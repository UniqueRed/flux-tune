"""flux-tune: recommend Deepgram Flux turn-taking settings for your own audio.

Input: a folder of user-channel recordings (wav/flac/ogg; any sample rate, mono or first channel),
each containing several user turns separated by the agent's speaking time (silence on this channel).

Pipeline:
  1. one manual-mode Flux pass (eot_threshold=1.0) per file
  2. hindsight turns from Flux's final word timestamps (gap split)
  3. 50/50 split by file: TUNE half picks the setting, REPORT half gives the numbers
  4. preview curve: offline re-thresholding of the manual-pass confidence stream (approximate;
     drifts on multi-turn audio, D-023)
  5. exact curve: real passes on a fixed threshold grid (REAL_GRID, includes the 0.7 default)
  6. recommendation per target cut-off rate = best real setting on TUNE meeting it; reported on
     REPORT with turn-level bootstrap CIs, paired against the default
  7. optional noise stress test: background talker mixed in at fixed TIRs, real pass at the
     recommended setting, scored against the clean hindsight turns

Usage: python scripts/flux_tune.py AUDIO_DIR --out reports/mine --target 0.1 [--noise] [--yes]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speakerturn.eval.bootstrap import bootstrap_metric, paired_bootstrap, select_operating_point  # noqa: E402
from speakerturn.eval.metrics import summarize  # noqa: E402
from speakerturn.flux.cache import FluxCache  # noqa: E402
from speakerturn.flux.client import run_batch  # noqa: E402
from speakerturn.flux.cost import CostMeter  # noqa: E402
from speakerturn.flux.events import parse_log  # noqa: E402
from speakerturn.fluxtune.core import agent_turns, arrays, hindsight_turns, score, single_turn, split_files  # noqa: E402
from speakerturn.fluxtune.simulate import real_eots, simulate_eot  # noqa: E402

SR = 16000
DEFAULT = 0.7
SIM_GRID = [round(x, 2) for x in np.arange(0.50, 0.99, 0.02)] + [0.99]
REAL_GRID = [0.6, 0.7, 0.8, 0.86, 0.9, 0.94, 0.98]  # exact curve (D-023)
TARGETS = (0.05, 0.10, 0.20)
TIRS = (10, 0)


def load_audio(p: Path) -> np.ndarray:
    a, sr = sf.read(p, dtype="float32", always_2d=True)
    a = a[:, 0]
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    return a


def background_talker(n: int, rng) -> np.ndarray:
    """Continuous raw AMI headset speech from TRAIN-split speakers (never the benchmark's speakers)."""
    idx = json.loads((ROOT / "data/pool/ami_v2/index.json").read_text())
    rows = [json.loads(l) for l in open(ROOT / "data/manifests/ami_turns.jsonl")]
    pool = [r for r in rows if r["split"] == "train" and r["turn_id"] in idx]
    out = np.zeros(0, np.float32)
    while len(out) < n:
        r = pool[rng.integers(len(pool))]
        a, _ = sf.read(ROOT / f"data/pool/ami_v2/{r['turn_id']}.flac", dtype="float32")
        a = a / (np.sqrt(np.mean(a ** 2)) + 1e-9)
        out = np.concatenate([out, a, np.zeros(int(rng.uniform(0.1, 0.8) * SR), np.float32)])
    return out[:n]


def active_rms(x):
    f = x[: len(x) // 320 * 320].reshape(-1, 320)
    e = np.sqrt((f ** 2).mean(1))
    act = e[e > e.max() * 0.05]
    return float(np.sqrt((act ** 2).mean())) if len(act) else 1e-5


async def passes(items, params, cache, meter, conc):
    res = await run_batch(items, params, cache, meter, concurrency=conc)
    return {cid: parse_log(recs) for cid, recs, _ in res}


def ci(a, metric, groups=None):
    r = bootstrap_metric(a, metric, groups=groups)
    return {k: r[k] for k in ("point", "lo", "hi")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=float, default=0.05, help="max cut-off rate")
    ap.add_argument("--gap", type=float, default=2.0, help="silence that separates user turns (s)")
    ap.add_argument("--agent_times", help="JSON {file_stem: [[start, end], ...]} of agent speech; "
                    "preferred over gap-based hindsight turns (which undercount cut-offs, D-021)")
    ap.add_argument("--single_turn", action="store_true", help="each file is one user turn (record_session.py)")
    ap.add_argument("--split", choices=["files", "turns"], default="files",
                    help="tune/report split unit; use 'turns' when there are only a few long recordings")
    ap.add_argument("--split_map", help="JSON {file_stem: 'tune'|'report'}: a fixed split, e.g. different corpora/speakers")
    ap.add_argument("--groups", help="JSON {file_stem: group}: bootstrap resamples whole groups (e.g. speakers)")
    ap.add_argument("--eval_dir", help="real recordings with background audio, scored at the recommended and default settings")
    ap.add_argument("--eval_agent_times", help="agent_times.json for --eval_dir")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    ap.add_argument("--concurrency", type=int, default=12)
    a = ap.parse_args()

    files = sorted(p for p in Path(a.audio_dir).iterdir() if p.suffix.lower() in (".wav", ".flac", ".ogg"))
    audio = {p.stem: load_audio(p) for p in files}
    minutes = sum(len(x) for x in audio.values()) / SR / 60
    if a.split_map:
        sm = json.loads(Path(a.split_map).read_text())
        tune, rep = {f for f in audio if sm[f] == "tune"}, {f for f in audio if sm[f] == "report"}
    else:
        tune, rep = split_files(list(audio))
    gmap = json.loads(Path(a.groups).read_text()) if a.groups else None
    G = (lambda ts: [gmap[t.file] for t in ts]) if gmap else (lambda ts: None)
    rep_min = sum(len(audio[f]) for f in rep) / SR / 60
    est = (minutes * (1 + 7) + (rep_min * 2 * len(TIRS) * 2 if a.noise else 0)) * 0.0077
    print(f"{len(audio)} files, {minutes:.1f} audio-min. Estimated Flux cost: ${est:.2f} "
          f"(1 manual pass + 7 real passes{' + noise passes' if a.noise else ''})", flush=True)
    if est > 5 and not a.yes:
        sys.exit("estimated cost > $5: re-run with --yes to proceed")

    cache = FluxCache(ROOT / "data" / "flux_cache")
    meter = CostMeter(ROOT / "experiments" / "api_spend.json", cap_usd=40.0)
    items = list(audio.items())
    man = asyncio.run(passes(items, {"eot_threshold": 1.0}, cache, meter, a.concurrency))
    if a.single_turn:
        turns = [t for f in audio for t in single_turn(f, man[f], len(audio[f]) / SR)]
        label_mode = "single_turn"
    elif a.agent_times:
        ag = json.loads(Path(a.agent_times).read_text())
        turns = [t for f in audio for t in agent_turns(f, man[f], len(audio[f]) / SR, ag.get(f, []))]
        label_mode = "agent_times"
    else:
        turns = [t for f in audio for t in hindsight_turns(f, man[f], len(audio[f]) / SR, a.gap)]
        label_mode = "gap_hindsight"
    if a.split == "turns" and not a.split_map:
        import hashlib
        h = lambda t: int(hashlib.sha256(f"{t.file}@{t.start:.2f}".encode()).hexdigest(), 16) % 2
        T = {"tune": [t for t in turns if h(t) == 0], "report": [t for t in turns if h(t) == 1]}
        rep = set(audio)  # noise test and leakage use every file; scoring still uses REPORT turns only
    else:
        T = {"tune": [t for t in turns if t.file in tune], "report": [t for t in turns if t.file in rep]}
    rep_min = sum(len(audio[f]) for f in rep) / SR / 60
    print(f"hindsight turns: {len(turns)} (tune {len(T['tune'])}, report {len(T['report'])})", flush=True)

    # approximate curve from one pass
    sim = {th: {f: simulate_eot(man[f], th) for f in audio} for th in SIM_GRID}
    sim_curve = {s: [{"threshold": th, **summarize(arrays(T[s], sim[th]))} for th in SIM_GRID] for s in T}
    real = {}
    for th in REAL_GRID:
        ev = asyncio.run(passes(items, {"eot_threshold": th}, cache, meter, a.concurrency))
        real[th] = {f: real_eots(ev[f]) for f in audio}
        print(f"real pass eot_threshold={th}: spend so far ${meter.state['usd']:.3f}", flush=True)
    def words_after(ts, dec):  # stricter count: cut-offs where the speaker said at least one more word
        det = score(ts, dec)[1]
        return sum(1 for d in det if d["cutoff"] and d["said_after"].strip()) / max(len(det), 1)

    real_curve = {s: [{"threshold": th, **summarize(arrays(T[s], real[th])), "fcr_words_after": words_after(T[s], real[th])}
                      for th in REAL_GRID] for s in T}
    tune_sweep = {th: arrays(T["tune"], real[th]) for th in REAL_GRID}
    ops = {t: select_operating_point(tune_sweep, t) for t in TARGETS}
    op = select_operating_point(tune_sweep, a.target)
    rec = op["param"]
    recs = {f"{t:.2f}": {"eot_threshold": o["param"], "reachable_on_tune": o["reachable"],
                         **{m: ci(arrays(T["report"], real[o["param"]]), m, G(T["report"])) for m in ("fcr", "lat_p50", "ter_1500")}}
            for t, o in ops.items()}

    A = {th: arrays(T["report"], real[th]) for th in (rec, DEFAULT)}
    out = {
        "files": len(audio), "audio_minutes": minutes,
        "bootstrap_unit": "group" if gmap else "turn",
        "n_groups": {k: len({gmap[t.file] for t in v}) for k, v in T.items()} if gmap else None, "target_cutoff": a.target, "gap_s": a.gap, "label_mode": label_mode,
        "turns": {k: len(v) for k, v in T.items()},
        "recommendation": {"eot_threshold": rec, "reachable_on_tune": op["reachable"]},
        "recommendations_by_target": recs,
        "report_half": {
            "recommended": {m: ci(A[rec], m, G(T["report"])) for m in ("fcr", "lat_p50", "lat_p90", "ter_1500")},
            "default_0.7": {m: ci(A[DEFAULT], m, G(T["report"])) for m in ("fcr", "lat_p50", "lat_p90", "ter_1500")},
            "rec_minus_default": {m: paired_bootstrap(A[DEFAULT], A[rec], m, stat=lambda b, n: n - b, groups=G(T["report"]))
                                  for m in ("fcr", "lat_p50")},
        },
        "curve_sim_preview": sim_curve, "curve_real": real_curve,
        "evidence": {str(th): [d for d in score(T["report"], real[th])[1] if d["cutoff"]][:25]
                     for th in (rec, DEFAULT)},
    }

    if a.noise:
        from collections import Counter
        from speakerturn.fluxtune.core import final_words

        def bag(ev):
            return Counter(w.word.lower().strip(".,?!") for w in final_words(ev))

        rng = np.random.default_rng(1337)
        noisy_items, out["noise"] = {}, {}
        for tir in TIRS:
            for f in sorted(rep):
                x = audio[f]
                b = background_talker(len(x), rng)
                noisy_items[f"{f}__tir{tir}"] = (x + b * active_rms(x) / 10 ** (tir / 20) / active_rms(b)).astype(np.float32)
        user_min = sum(t.end - t.start for t in T["report"]) / 60
        clean_bag = {f: bag(man[f]) for f in rep}
        for th in sorted({rec, DEFAULT}):
            ev = asyncio.run(passes(list(noisy_items.items()), {"eot_threshold": th}, cache, meter, a.concurrency))
            for tir in TIRS:
                dec = {f: real_eots(ev[f"{f}__tir{tir}"]) for f in rep}
                an = arrays(T["report"], dec)
                n_dec = sum(len(v) for v in dec.values())
                n_hit = sum(1 for d in score(T["report"], dec)[1] if d["decision"] is not None)
                inserted = sum(sum((bag(ev[f"{f}__tir{tir}"]) - clean_bag[f]).values()) for f in rep)
                out["noise"][f"th{th}_tir{tir}dB"] = {
                    "threshold": th, "tir_db": tir,
                    **{m: ci(an, m, G(T["report"])) for m in ("fcr", "lat_p50", "ter_1500", "no_decision_rate")},
                    "extra_eots_per_min": (n_dec - n_hit) / rep_min,
                    "inserted_words_per_audio_min": inserted / rep_min,
                    "inserted_words_per_user_min": inserted / user_min,
                    "vs_clean": {m: paired_bootstrap(A[th], an, m, stat=lambda b, n: n - b, groups=G(T["report"])) for m in ("ter_1500", "fcr")},
                }
            print(f"noise passes at eot_threshold={th}: done, spend ${meter.state['usd']:.3f}", flush=True)

    if a.eval_dir:
        ev_audio = {p.stem: load_audio(p) for p in sorted(Path(a.eval_dir).iterdir())
                    if p.suffix.lower() in (".wav", ".flac", ".ogg")}
        ev_items = list(ev_audio.items())
        ev_ag = json.loads(Path(a.eval_agent_times).read_text())
        ev_man = asyncio.run(passes(ev_items, {"eot_threshold": 1.0}, cache, meter, a.concurrency))
        ev_turns = [t for f in ev_audio for t in agent_turns(f, ev_man[f], len(ev_audio[f]) / SR, ev_ag.get(f, []))]
        out["real_background"] = {"files": len(ev_audio), "turns": len(ev_turns), "by_threshold": {}}
        for th in sorted({rec, DEFAULT}):
            ev = asyncio.run(passes(ev_items, {"eot_threshold": th}, cache, meter, a.concurrency))
            dec = {f: real_eots(ev[f]) for f in ev_audio}
            an = arrays(ev_turns, dec)
            out["real_background"]["by_threshold"][str(th)] = {m: ci(an, m) for m in ("fcr", "lat_p50", "ter_1500", "no_decision_rate")}
            out["real_background"].setdefault("evidence", {})[str(th)] = [d for d in score(ev_turns, dec)[1] if d["cutoff"]][:15]
        print(f"real background recordings: {len(ev_turns)} turns, spend ${meter.state['usd']:.3f}", flush=True)

    out["spend_usd_total"] = meter.state["usd"]
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "results.json").write_text(json.dumps(out, indent=1, default=float))
    r = out["report_half"]
    print(f"\nRECOMMENDED eot_threshold={rec} (target cut-off <= {a.target:.0%}; reachable on tune half: {op['reachable']})")
    for k in ("recommended", "default_0.7"):
        v = r[k]
        print(f"  {k:12s} cut-off {v['fcr']['point']:.1%} [{v['fcr']['lo']:.1%}, {v['fcr']['hi']:.1%}]  "
              f"median latency {v['lat_p50']['point']:.2f}s [{v['lat_p50']['lo']:.2f}, {v['lat_p50']['hi']:.2f}]")
    print(f"saved {od / 'results.json'}")


if __name__ == "__main__":
    main()
