"""Offline re-thresholding of Flux end-of-turn decisions from one manual-mode pass.

A manual-mode pass (eot_threshold=1.0) never ends turns by model, so it exposes the raw
end_of_turn_confidence stream. `simulate_eot` replays that stream with a candidate threshold
and Flux's turn state machine (docs/external_facts.md §3):
  idle  --(speech onset: first word starting after the last decision)-->  in_turn
  in_turn --(event with conf >= threshold and a non-empty transcript)--> EOT, back to idle
Whether this reproduces real runs is an empirical question, tested by scripts/flux_validate.py.
"""
from __future__ import annotations

from ..types import FluxEvent


def simulate_eot(events: list[FluxEvent], threshold: float, min_words: int = 1) -> list[float]:
    """Decision times (t_avail of the triggering event) for a candidate threshold."""
    events = sorted(events, key=lambda e: e.t_avail)
    decisions: list[float] = []
    in_turn, last_dec = False, -1.0
    for i, e in enumerate(events):
        if not in_turn:
            # a new turn begins once the stream shows a word that started after the last decision
            if any(w.start is not None and w.start > last_dec + 1e-6 for w in e.words) or \
                    (e.event == "StartOfTurn" and e.t_avail > last_dec):
                in_turn = True
            else:
                continue
        conf = e.end_of_turn_confidence
        new_words = [w for w in e.words if w.start is None or w.start > last_dec + 1e-6]
        if conf is not None and conf >= threshold and len(new_words) >= min_words:
            decisions.append(e.t_avail)
            last_dec, in_turn = e.t_avail, False
    return decisions


def real_eots(events: list[FluxEvent]) -> list[float]:
    return [e.t_avail for e in sorted(events, key=lambda e: e.t_avail) if e.event == "EndOfTurn"]
