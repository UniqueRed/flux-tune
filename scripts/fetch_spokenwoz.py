"""Fetch a sample of SpokenWOZ test dialogues (CC BY-NC 4.0) as a second flux-tune corpus.

SpokenWOZ: human crowdworkers role-playing customer and agent on phone calls; stereo 8 kHz WAV,
ch0 = user, ch1 = agent (docs/data_licenses.md). Only the dialogues we use are extracted from the
2.18 GB test zip via HTTP range requests. Saved per dialogue: the USER channel only, as FLAC.
agent_times.json comes from the annotated system turns (first word start -> last word end).
Usage: python scripts/fetch_spokenwoz.py --n 40 --out data/spokenwoz
"""
import argparse
import ast
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/datasets/ssz1111"
AUDIO_ZIP = f"{HF}/SpokenWOZ-Test-Audio-Fixed/resolve/main/audio_5700_test.zip"
TEXT = f"{HF}/SpokenWOZ-Test-Text-Fixed/resolve/main/data.json"


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = r.read()
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "data" / "spokenwoz"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    text_path = ROOT / "data" / "raw" / "spokenwoz_test.json"
    if not text_path.exists():
        text_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TEXT, text_path)
    dialogues = json.loads(text_path.read_text())

    z = zipfile.ZipFile(io.BufferedReader(RangeFile(AUDIO_ZIP), buffer_size=1 << 20))
    wavs = {Path(i.filename).stem: i for i in z.infolist() if i.filename.endswith(".wav")}
    ids = sorted(set(wavs) & set(dialogues))
    pick = [ids[i] for i in np.random.default_rng(a.seed).permutation(len(ids))[: a.n]]

    at_path = out / "agent_times.json"
    agent = json.loads(at_path.read_text()) if at_path.exists() else {}
    for k, did in enumerate(pick):
        if (out / f"{did}.flac").exists() and did in agent:
            continue
        x, sr = sf.read(io.BytesIO(z.open(wavs[did]).read()), dtype="int16", always_2d=True)
        sf.write(out / f"{did}.flac", x[:, 0], sr, subtype="PCM_16")
        spans = []
        for turn in dialogues[did]["log"]:
            ws = turn["words"]
            ws = ast.literal_eval(ws) if isinstance(ws, str) else ws
            if turn["tag"] == "system" and ws:
                spans.append([ws[0]["BeginTime"] / 1000, ws[-1]["EndTime"] / 1000])
        agent[did] = spans
        print(f"{k + 1}/{len(pick)} {did} {len(x) / sr:.0f}s, {len(spans)} agent turns", flush=True)
    at_path.write_text(json.dumps(agent, indent=1))
    (out / "groups.json").write_text(json.dumps({d: d for d in agent}, indent=1))


if __name__ == "__main__":
    main()
