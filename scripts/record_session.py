"""Record yourself answering voice-agent prompts, one answer per file, for flux-tune.

For each prompt: read it, press Enter, answer out loud as you naturally would (pauses, "um"s and
all), press Enter when you're done. Recording continues 3 s after your Enter so Flux sees the
silence. Files: <out>/<condition>_<NN>.wav (16 kHz mono). Each file holds exactly one user turn,
so run flux-tune with --single_turn.

Usage:
  python scripts/record_session.py --out data/my_recordings --condition quiet --n 15
  python scripts/record_session.py --out data/my_recordings --condition tv --n 15   # TV/podcast on
"""
import argparse
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speakerturn.voicelab.prompts import PROMPTS  # noqa: E402

SR = 16000
TAIL_S = 3.0


def record_one(path: Path):
    import sounddevice as sd

    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    chunks = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=cb):
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        while not stop.is_set():
            try:
                chunks.append(q.get(timeout=0.1))
            except queue.Empty:
                pass
        n_tail = int(TAIL_S * SR)
        got = 0
        while got < n_tail:
            c = q.get()
            chunks.append(c)
            got += len(c)
    sf.write(path, np.concatenate(chunks), SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/my_recordings")
    ap.add_argument("--condition", default="quiet", help="label, e.g. quiet / tv / cafe")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--start", type=int, default=0, help="first prompt index")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Recording {a.n} answers ({a.condition}). Answer naturally; pauses are fine.\n")
    for k in range(a.n):
        prompt = PROMPTS[(a.start + k) % len(PROMPTS)]
        print(f"[{k + 1}/{a.n}] AGENT: {prompt}")
        input("   press Enter, then answer...")
        print("   recording... press Enter when you're finished")
        record_one(out / f"{a.condition}_{a.start + k:02d}.wav")
        print("   saved.\n")
    print(f"Done. Files in {out}")


if __name__ == "__main__":
    sys.exit(main())
