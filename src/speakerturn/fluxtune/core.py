"""flux-tune core: hindsight turns from a manual-mode Flux pass, and scoring of EOT decisions.

Hindsight turns (validated in D-020): Flux's final word hypotheses (last word list per Flux turn
index) are merged and split wherever the gap between consecutive words is >= `gap` seconds.
A turn = [first word start, last word end]. In a user-channel recording the agent speaks between
turns, so inter-turn gaps are long; mid-turn hold pauses shorter than `gap` stay inside the turn.

Scoring: for each hindsight turn, the first decision at/after its start and before
the next turn starts is d. Cut-off if d < end - 0.1 s; else latency = d - end. No decision in the
window = censored at the window end (counted as latency, and as an extension if long).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eval.metrics import TurnResult, turn_arrays
from ..types import FluxEvent, FluxWord


@dataclass
class HTurn:
    file: str
    start: float
    end: float
    words: list[FluxWord]
    window_end: float  # next turn start, or file end

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


def final_words(events: list[FluxEvent]) -> list[FluxWord]:
    last: dict[int, list[FluxWord]] = {}
    for e in events:
        if e.words:
            last[e.turn_index] = e.words
    return sorted((w for ws in last.values() for w in ws if w.start is not None and w.end is not None),
                  key=lambda w: w.start)


def hindsight_turns(file: str, events: list[FluxEvent], dur: float, gap: float = 2.0) -> list[HTurn]:
    words = final_words(events)
    groups: list[list[FluxWord]] = []
    for w in words:
        if groups and w.start - groups[-1][-1].end < gap:
            groups[-1].append(w)
        else:
            groups.append([w])
    turns = []
    for i, g in enumerate(groups):
        nxt = groups[i + 1][0].start if i + 1 < len(groups) else dur
        turns.append(HTurn(file, g[0].start, g[-1].end, g, nxt))
    return turns


def agent_turns(file: str, events: list[FluxEvent], dur: float,
                agent: list[tuple[float, float]]) -> list[HTurn]:
    """Turns delimited by the agent's speaking intervals (stereo agent channel or agent timestamps).
    A user turn = the user's words between two agent intervals; it ends at the user's last word
    before the agent starts, the same "next speaker change" definition as docs/labeling.md.
    Most faithful for human-agent recordings: with an existing bot, agent onsets are that bot's own
    (possibly premature) decisions."""
    words = final_words(events)
    agent = sorted(agent)
    bounds = [0.0] + [a for a, _ in agent] + [dur]
    starts = [0.0] + [b for _, b in agent]
    turns = []
    for k in range(len(bounds) - 1):
        lo, hi = starts[k], bounds[k + 1]
        ws = [w for w in words if lo <= w.start < hi]
        if ws:
            turns.append(HTurn(file, ws[0].start, ws[-1].end, ws, hi))
    for i, t in enumerate(turns):  # decision window runs until the next user turn begins
        t.window_end = turns[i + 1].start if i + 1 < len(turns) else dur
    return turns


def single_turn(file: str, events: list[FluxEvent], dur: float) -> list[HTurn]:
    """One user turn per file (scripts/record_session.py): all of Flux's final words."""
    ws = final_words(events)
    return [HTurn(file, ws[0].start, ws[-1].end, ws, dur)] if ws else []


def score(turns: list[HTurn], decisions: dict[str, list[float]]) -> tuple[list[TurnResult], list[dict]]:
    """decisions: file -> decision times. Returns TurnResults (for metrics) and per-turn detail."""
    results, detail = [], []
    for t in turns:
        ds = [d for d in decisions.get(t.file, []) if t.start <= d < t.window_end]
        d = ds[0] if ds else None
        results.append(TurnResult(f"{t.file}@{t.start:.2f}", t.start, t.end, t.window_end, d))
        detail.append({"file": t.file, "start": t.start, "end": t.end, "decision": d,
                       "cutoff": d is not None and d < t.end - 0.1, "text": t.text,
                       "said_after": " ".join(w.word for w in t.words if d is not None and w.start >= d)})
    return results, detail


def arrays(turns, decisions):
    return turn_arrays(score(turns, decisions)[0])


def split_files(files: list[str], seed: int = 1337) -> tuple[set[str], set[str]]:
    """Deterministic 50/50 split by file: tune on one half, report on the other."""
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(sorted(files)))
    h = len(order) // 2
    return set(order[:h]), set(order[h:])
