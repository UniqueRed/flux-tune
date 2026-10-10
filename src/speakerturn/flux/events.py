"""Parse logged Flux messages into FluxEvents (schema: docs/external_facts.md §3).

Each logged record = {"msg": <server JSON>, "wall_rel": s since first audio byte sent,
"audio_sent": audio seconds sent at arrival}. Float fields may be strings; parse defensively.

t_avail policy (when could a causal system have this event?):
  realtime : max(audio_window_end, wall_rel)          -- real-time paced runs
  offset   : audio_window_end + latency_offset        -- faster-than-real-time runs, offset
                                                         calibrated from real-time runs (§6.0.3)
"""
from __future__ import annotations

from ..types import FluxEvent, FluxWord

TURN_EVENTS = {"StartOfTurn", "Update", "EagerEndOfTurn", "TurnResumed", "EndOfTurn"}


def _f(x):
    return None if x is None else float(x)


def parse_record(rec: dict, policy: str = "realtime", latency_offset: float = 0.0) -> FluxEvent | None:
    m = rec["msg"]
    if m.get("type") != "TurnInfo" or m.get("event") not in TURN_EVENTS:
        return None
    awe = _f(m.get("audio_window_end")) or 0.0
    if policy == "realtime":
        t_avail = max(awe, float(rec.get("wall_rel", awe)))
    elif policy == "offset":
        t_avail = awe + latency_offset
    else:
        raise ValueError(policy)
    words = [FluxWord(w.get("word", ""), _f(w.get("confidence")), _f(w.get("start")), _f(w.get("end")))
             for w in m.get("words") or []]
    return FluxEvent(event=m["event"], t_avail=t_avail, turn_index=int(m.get("turn_index", 0)),
                     audio_window_start=_f(m.get("audio_window_start")) or 0.0,
                     audio_window_end=awe, transcript=m.get("transcript", ""), words=words,
                     end_of_turn_confidence=_f(m.get("end_of_turn_confidence")), raw=m)


def parse_log(records: list[dict], **kw) -> list[FluxEvent]:
    evs = [e for e in (parse_record(r, **kw) for r in records) if e is not None]
    return sorted(evs, key=lambda e: e.t_avail)
