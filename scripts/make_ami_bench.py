"""Build the AMI flux-tune benchmark: tune on AMI dev (val) meetings, report on unseen AMI test meetings.

Each file is a user-channel "call": raw headset turns separated by 2.5-5 s of noise floor
for the agent's speaking time.
  - no turn is reused, and every speaker contributes at most --per_spk turns (no speaker dominates);
  - dev and test speakers/meetings are disjoint, so the report half is a different population;
  - writes agent_times.json (the gaps), split.json (file -> tune|report) and groups.json
    (file -> speaker) for the speaker-clustered bootstrap.
Usage: python scripts/make_ami_bench.py --out data/ami_bench --tune_per_spk 15 --report_per_spk 20
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import flux_validate  # noqa: E402
from flux_validate import load_turns, seg  # noqa: E402

SR = 16000


def build(split, half, per_spk, turns_per_file, rng, out, labels, agent, split_map, groups):
    turns, idx = load_turns(split)
    by_spk = {}
    for r in turns:
        if 1.0 <= r["end"] - r["start"] <= 10:
            by_spk.setdefault(r["target_speaker"], []).append(r)
    n = 0
    for spk in sorted(by_spk):
        rs = [by_spk[spk][i] for i in rng.permutation(len(by_spk[spk]))[:per_spk]]
        for k in range(0, len(rs) - turns_per_file + 1, turns_per_file):
            parts, meta, ag, t, g = [], [], [], 0.0, None
            for r in rs[k:k + turns_per_file]:
                x, off = seg(r, idx)
                g = g or 0.1 / (np.sqrt(np.mean(x ** 2)) + 1e-9)
                x = x * g
                meta.append({"turn_id": r["turn_id"], "start": t + off, "end": t + off + r["end"] - r["start"], "text": r["text"]})
                gap = (rng.normal(0, 1, int(rng.uniform(2.5, 5.0) * SR)) * 1e-3).astype(np.float32)
                ag.append([round(t + len(x) / SR, 3), round(t + (len(x) + len(gap)) / SR, 3)])
                parts += [x, gap]
                t += (len(x) + len(gap)) / SR
            name = f"{half}_{spk}_{k // turns_per_file}"
            sf.write(out / f"{name}.flac", np.concatenate(parts), SR)
            labels[name], agent[name], split_map[name], groups[name] = meta, ag, half, spk
            n += len(meta)
    print(f"{split} -> {half}: {len(by_spk)} speakers, {n} turns")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "ami_bench"))
    ap.add_argument("--tune_per_spk", type=int, default=15)
    ap.add_argument("--report_per_spk", type=int, default=20)
    ap.add_argument("--turns_per_file", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pool", default=str(flux_validate.POOL), help="AMI clips from fetch_ami_segments.py")
    a = ap.parse_args()
    flux_validate.POOL = Path(a.pool)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels, agent, split_map, groups = {}, {}, {}, {}
    build("val", "tune", a.tune_per_spk, a.turns_per_file, rng, out, labels, agent, split_map, groups)
    build("test", "report", a.report_per_spk, a.turns_per_file, rng, out, labels, agent, split_map, groups)
    for name, obj in (("labels", labels), ("agent_times", agent), ("split", split_map), ("groups", groups)):
        (out / f"{name}.json").write_text(json.dumps(obj, indent=1))
    print(f"wrote {len(labels)} files to {out}")


if __name__ == "__main__":
    main()
