"""Fetch only the needed AMI headset audio via HTTP range requests (DECISIONS D-008).

For each selected turn, download [start - PRE, end + POST] from the target's headset channel and
store it as 16 kHz mono FLAC in data/pool/ami/<turn_id>.flac. Full channels are never stored.

Usage: python scripts/fetch_ami_segments.py --split train --n 3000 --seed 1337
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speakerturn.data.ami import AUDIO_URL  # noqa: E402

PRE, POST = 0.3, 0.6
SR, HDR = 16000, 44
OUT = ROOT / "data" / "pool" / "ami"


def fetch(row: dict) -> tuple[str, float]:
    path = OUT / f"{row['turn_id']}.flac"
    PRE, POST = ARGS.pre, ARGS.post
    if path.exists():
        return row["turn_id"], 0.0
    s = max(0.0, row["start"] - PRE)
    e = row["end"] + POST
    b0 = HDR + 2 * int(s * SR)
    b1 = HDR + 2 * int(e * SR) - 1
    url = AUDIO_URL.format(m=row["meeting"], ch=row["channel"])
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={b0}-{b1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise IOError(f"no range support: HTTP {r.status}")
                data = r.read()
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
    tmp = path.with_suffix(".tmp.flac")
    sf.write(tmp, pcm, SR, subtype="PCM_16")
    tmp.replace(path)
    return row["turn_id"], s


def select(rows: list[dict], split: str, n: int, seed: int) -> list[dict]:
    """Spread the sample over speakers: round-robin across speakers in random order."""
    rng = np.random.default_rng(seed)
    by_spk: dict[str, list[dict]] = {}
    for r in rows:
        if r["split"] == split:
            by_spk.setdefault(r["target_speaker"], []).append(r)
    for v in by_spk.values():
        rng.shuffle(v)
    out, i = [], 0
    spks = sorted(by_spk)
    while len(out) < n and any(i < len(v) for v in by_spk.values()):
        for s in spks:
            if i < len(by_spk[s]) and len(out) < n:
                out.append(by_spk[s][i])
        i += 1
    return out


def main():
    global OUT, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pre", type=float, default=PRE)
    ap.add_argument("--post", type=float, default=POST)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    OUT, ARGS = Path(a.out), a
    rows = [json.loads(l) for l in open(ROOT / "data" / "manifests" / "ami_turns.jsonl")]
    sel = select(rows, a.split, a.n, a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, r) for r in sel]
        for f in as_completed(futs):
            tid, seg_start = f.result()
            if tid not in index:
                row = next(r for r in sel if r["turn_id"] == tid)
                index[tid] = round(max(0.0, row["start"] - a.pre), 6)
            done += 1
            if done % 250 == 0:
                print(f"{done}/{len(sel)}", flush=True)
                index_path.write_text(json.dumps(index))
    index_path.write_text(json.dumps(index))
    print(f"done: {len(sel)} turns for split={a.split}")


if __name__ == "__main__":
    main()
