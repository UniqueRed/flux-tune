"""Day-1 validation for flux-tune (DECISIONS D-019).

Test 1 (offline re-thresholding): run each clip in manual mode (eot_threshold=1.0) and for real at
0.7 and 0.85; compare simulated vs real EndOfTurn times.
Test 2 (hindsight labels): compare Flux-word-based hindsight turn ends with AMI manual turn ends.

Clips (AMI val, raw continuous headset segments from data/pool/ami_v2):
  single : one turn + 4 s noise-floor tail
  session: 3 turns of one speaker, 3-4 s noise-floor gaps (a "user channel" of a call)
Usage: python scripts/flux_validate.py --n_single 24 --n_session 10
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
from speakerturn.flux.cache import FluxCache  # noqa: E402
from speakerturn.flux.client import run_batch  # noqa: E402
from speakerturn.flux.cost import CostMeter  # noqa: E402

SR = 16000
POOL = ROOT / "data" / "pool" / "ami_v2"
PASSES = {"manual": {"eot_threshold": 1.0}, "t070": {"eot_threshold": 0.7}, "t085": {"eot_threshold": 0.85}}


def load_turns(split="val"):
    idx = json.loads((POOL / "index.json").read_text())
    rows = [json.loads(l) for l in open(ROOT / "data" / "manifests" / "ami_turns.jsonl")]
    return [r for r in rows if r["split"] == split and r["turn_id"] in idx], idx


def seg(r, idx):
    a, _ = sf.read(POOL / f"{r['turn_id']}.flac", dtype="float32")
    s0 = idx[r["turn_id"]]
    # keep [start-0.5, end+0.3] so the next turn's speaker bleed is minimal
    i, j = int(max(0, r["start"] - 0.5 - s0) * SR), int((r["end"] + 0.3 - s0) * SR)
    return a[i:j], r["start"] - (s0 + i / SR)  # audio, offset of turn start within it


def floor(n, level, rng):
    return (rng.normal(0, 1, n) * level).astype(np.float32)


def build(rng, turns, idx, n_single, n_session):
    clips = []
    by_spk = {}
    for r in turns:
        if 1.5 <= r["end"] - r["start"] <= 10:
            by_spk.setdefault(r["target_speaker"], []).append(r)
    spks = sorted(by_spk)
    for k in range(n_single):
        r = by_spk[spks[k % len(spks)]][rng.integers(len(by_spk[spks[k % len(spks)]]))]
        a, off = seg(r, idx)
        lvl = np.sqrt(np.mean(a ** 2))
        g = 0.1 / (lvl + 1e-9)
        a = a * g
        audio = np.concatenate([a, floor(4 * SR, np.abs(a).std() * 0.01 + 1e-4, rng)])
        clips.append({"id": f"single{k:02d}", "audio": audio,
                      "turns": [{"turn_id": r["turn_id"], "start": off, "end": off + r["end"] - r["start"],
                                 "text": r["text"]}]})
    for k in range(n_session):
        spk = spks[(k * 7) % len(spks)]
        rs = [by_spk[spk][i] for i in rng.permutation(len(by_spk[spk]))[:3]]
        parts, turns_meta, t = [], [], 0.0
        g = None
        for r in rs:
            a, off = seg(r, idx)
            g = g or 0.1 / (np.sqrt(np.mean(a ** 2)) + 1e-9)
            a = a * g
            turns_meta.append({"turn_id": r["turn_id"], "start": t + off, "end": t + off + r["end"] - r["start"],
                               "text": r["text"]})
            gap = floor(int(rng.uniform(3.0, 4.0) * SR), 1e-3, rng)
            parts += [a, gap]
            t += (len(a) + len(gap)) / SR
        clips.append({"id": f"session{k:02d}", "audio": np.concatenate(parts), "turns": turns_meta})
    return clips


async def main_async(a):
    rng = np.random.default_rng(a.seed)
    turns, idx = load_turns("val")
    clips = build(rng, turns, idx, a.n_single, a.n_session)
    total = sum(len(c["audio"]) for c in clips) / SR
    print(f"{len(clips)} clips, {total / 60:.1f} audio-min per pass, {len(PASSES)} passes", flush=True)
    cache = FluxCache(ROOT / "data" / "flux_cache")
    meter = CostMeter(ROOT / "experiments" / "api_spend.json", cap_usd=40.0)
    out = ROOT / "experiments" / "fluxtune_validate"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, params in PASSES.items():
        res = await run_batch([(c["id"], c["audio"]) for c in clips], params, cache, meter,
                              concurrency=a.concurrency)
        results[name] = {cid: recs for cid, recs, _ in res}
        print(f"pass {name}: done, spend ${meter.state['usd']:.3f}", flush=True)
    (out / "clips.json").write_text(json.dumps([{k: v for k, v in c.items() if k != "audio"} | {"dur": len(c["audio"]) / SR}
                                               for c in clips], indent=1))
    (out / "records.json").write_text(json.dumps(results))
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_single", type=int, default=24)
    ap.add_argument("--n_session", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1337)
    asyncio.run(main_async(ap.parse_args()))
