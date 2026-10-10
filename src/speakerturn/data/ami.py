"""AMI corpus adapter: NXT word annotations + per-meeting headset audio (CC BY 4.0)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .labeling import Turn, Word, spurts, turns

AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Headset-{ch}.wav"
NITE = "{http://nite.sourceforge.net/}"


@dataclass
class Speaker:
    agent: str       # NXT letter A-E
    channel: int     # headset index
    global_id: str   # e.g. FEE013 (F/M = gender)
    role: str

    @property
    def gender(self) -> str:
        return self.global_id[0]


def series(meeting: str) -> str:
    return meeting[:6]


def load_meetings(root: Path) -> dict[str, dict]:
    tree = ET.parse(root / "corpusResources" / "meetings.xml")
    out = {}
    for m in tree.getroot().iter("meeting"):
        spk = {}
        for s in m.iter("speaker"):
            if s.get("channel") is None:
                continue
            spk[s.get("nxt_agent")] = Speaker(s.get("nxt_agent"), int(s.get("channel")),
                                              s.get("global_name"), s.get("role", ""))
        out[m.get("observation")] = {"type": m.get("type"), "duration": float(m.get("duration") or 0),
                                     "speakers": spk}
    return out


def load_words(root: Path, meeting: str, agent: str) -> list[Word]:
    p = root / "words" / f"{meeting}.{agent}.words.xml"
    if not p.exists():
        return []
    ws = []
    for w in ET.parse(p).getroot():
        if w.tag != "w" or w.get("punc") == "true" or w.get("starttime") is None or w.get("endtime") is None:
            continue
        s, e = float(w.get("starttime")), float(w.get("endtime"))
        if e > s:
            ws.append(Word((w.text or "").strip(), s, e))
    return ws


def meeting_turns(root: Path, meeting: str, meta: dict) -> tuple[list[Turn], dict[str, list[Word]]]:
    words = {a: load_words(root, meeting, a) for a in meta["speakers"]}
    words = {a: w for a, w in words.items() if w}
    return turns(spurts(words)), words
