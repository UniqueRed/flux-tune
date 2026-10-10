"""Operational turn labeling (documented in docs/labeling.md).

1. Per speaker, merge consecutive words into *spurts* when the inter-word gap < SPURT_GAP.
2. A spurt is a *backchannel* if it lasts < 1.0 s and has <= 3 words.
3. A target turn starts at a non-backchannel spurt of speaker S and extends over S's later spurts
   until the next non-backchannel spurt of a different speaker starts (the "change").
   Turn end = end of S's last word before the change. If S's last spurt straddles the change and
   ends within END_OVERLAP_MAX (1.0 s) after it, that is an overlapped ending and the turn ends
   at that spurt's end; a longer straddle is an interruption (excluded downstream).
   Gaps between S's spurts inside the turn are hold pauses.
Known label noise: overlapping interruptions, and backchannels that are really short turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SPURT_GAP = 0.2
BC_MAX_DUR = 1.0
BC_MAX_WORDS = 3
END_OVERLAP_MAX = 1.0


@dataclass
class Word:
    w: str
    s: float
    e: float


@dataclass
class Spurt:
    spk: str
    words: list[Word]

    @property
    def s(self): return self.words[0].s

    @property
    def e(self): return self.words[-1].e

    @property
    def backchannel(self) -> bool:
        return (self.e - self.s) < BC_MAX_DUR and len(self.words) <= BC_MAX_WORDS


@dataclass
class Turn:
    spk: str
    start: float
    end: float
    words: list[Word]
    holds: list[tuple[float, float]] = field(default_factory=list)
    prev_other_end: float | None = None   # end of the last other-speaker non-bc spurt before start
    next_change: float | None = None      # start of the change spurt (None = meeting end)
    overlap_at_start: bool = False        # another non-bc spurt still active at turn start
    end_overlap: bool = False             # next speaker started <= END_OVERLAP_MAX before S stopped
    interrupted: bool = False             # S kept talking > END_OVERLAP_MAX after the change


def spurts(words_by_spk: dict[str, list[Word]], gap: float = SPURT_GAP) -> list[Spurt]:
    out = []
    for spk, ws in words_by_spk.items():
        ws = sorted(ws, key=lambda w: w.s)
        cur: list[Word] = []
        for w in ws:
            if cur and w.s - cur[-1].e >= gap:
                out.append(Spurt(spk, cur))
                cur = []
            cur.append(w)
        if cur:
            out.append(Spurt(spk, cur))
    return sorted(out, key=lambda s: (s.s, s.e))


def turns(sp: list[Spurt]) -> list[Turn]:
    major = [s for s in sp if not s.backchannel]
    out: list[Turn] = []
    i = 0
    while i < len(major):
        spk = major[i].spk
        j = i
        while j + 1 < len(major) and major[j + 1].spk == spk:
            j += 1
        start = major[i].s
        change = major[j + 1].s if j + 1 < len(major) else None
        # all of S's spurts (incl. its own backchannel-sized ones) that start before the change
        own = [s for s in sp if s.spk == spk and s.s >= start and (change is None or s.s < change)]
        end_overlap = interrupted = False
        if own and change is not None and own[-1].e > change:
            # S still talking when the other speaker takes over: an overlapped ending if S stops
            # within END_OVERLAP_MAX, else an interruption (excluded downstream)
            if own[-1].e - change <= END_OVERLAP_MAX:
                end_overlap = True
            else:
                interrupted = True
        words = [w for s in own for w in s.words]
        if not words:
            i = j + 1
            continue
        holds = [(a.e, b.s) for a, b in zip(own, own[1:]) if b.s > a.e]
        prev = [m for m in major[:i] if m.spk != spk]
        t = Turn(spk, start, words[-1].e, words, holds,
                 prev_other_end=max((m.e for m in prev), default=None), next_change=change)
        t.overlap_at_start = any(m.s < start < m.e for m in prev)
        t.end_overlap, t.interrupted = end_overlap, interrupted
        out.append(t)
        i = j + 1
    return out
