"""flux-tune core: hindsight segmentation, scoring and the offline re-threshold state machine."""
import pytest

from speakerturn.fluxtune.core import hindsight_turns, score, split_files
from speakerturn.fluxtune.simulate import simulate_eot
from speakerturn.types import FluxEvent, FluxWord


def W(w, s, e):
    return FluxWord(w, 0.9, s, e)


def test_hindsight_turns_split_on_gap_and_use_final_hypothesis():
    ev = [FluxEvent("Update", 1.0, turn_index=0, words=[W("helo", 0.5, 0.9)]),
          FluxEvent("Update", 2.0, turn_index=0, words=[W("hello", 0.5, 0.9), W("there", 1.0, 1.4), W("um", 2.5, 2.7)]),
          FluxEvent("Update", 9.0, turn_index=0, words=[W("hello", 0.5, 0.9), W("there", 1.0, 1.4), W("um", 2.5, 2.7),
                                                        W("yes", 6.0, 6.3)])]
    T = hindsight_turns("f", ev, dur=10.0, gap=2.0)
    assert [(t.start, t.end) for t in T] == [(0.5, 2.7), (6.0, 6.3)]   # 1.1 s hold stays inside turn 1
    assert T[0].text == "hello there um" and T[0].window_end == 6.0 and T[1].window_end == 10.0


def test_score_cutoff_latency_and_censoring():
    ev = [FluxEvent("Update", 9.0, words=[W("a", 0.5, 1.0), W("b", 1.5, 2.0), W("c", 6.0, 6.5)])]
    T = hindsight_turns("f", ev, dur=10.0, gap=2.0)
    res, det = score(T, {"f": [1.2, 7.0]})
    assert det[0]["cutoff"] and det[0]["said_after"] == "b"
    assert not det[1]["cutoff"] and res[1].decision == 7.0
    res, det = score(T, {"f": [2.5]})
    assert res[1].decision is None   # censored at file end by the metrics


def test_simulate_eot_respects_turn_state_machine():
    ev = [FluxEvent("StartOfTurn", 0.5, words=[W("hi", 0.3, 0.5)], end_of_turn_confidence=0.1),
          FluxEvent("Update", 0.75, words=[W("hi", 0.3, 0.5)], end_of_turn_confidence=0.8),
          FluxEvent("Update", 1.0, words=[W("hi", 0.3, 0.5)], end_of_turn_confidence=0.9),   # same turn: no 2nd EOT
          FluxEvent("Update", 3.0, words=[W("hi", 0.3, 0.5), W("ok", 2.5, 2.8)], end_of_turn_confidence=0.2),
          FluxEvent("Update", 3.25, words=[W("hi", 0.3, 0.5), W("ok", 2.5, 2.8)], end_of_turn_confidence=0.75)]
    assert simulate_eot(ev, 0.7) == [0.75, 3.25]
    assert simulate_eot(ev, 0.85) == [1.0]


def test_split_files_deterministic_and_disjoint():
    a, b = split_files([f"f{i}" for i in range(10)])
    assert a.isdisjoint(b) and len(a | b) == 10
    assert split_files([f"f{i}" for i in range(10)]) == (a, b)


def test_agent_turns_keep_long_holds_inside_one_turn():
    from speakerturn.fluxtune.core import agent_turns
    ev = [FluxEvent("Update", 9.0, words=[W("a", 0.5, 1.0), W("b", 3.6, 4.0), W("c", 7.0, 7.5)])]
    T = agent_turns("f", ev, dur=10.0, agent=[(4.5, 6.5)])
    assert [(t.start, t.end) for t in T] == [(0.5, 4.0), (7.0, 7.5)]   # 2.6 s hold stays inside
    assert T[0].window_end == 7.0


def test_single_turn_spans_all_words():
    from speakerturn.fluxtune.core import single_turn
    ev = [FluxEvent("Update", 9.0, words=[W("a", 0.5, 1.0), W("b", 4.0, 4.4)])]
    (t,) = single_turn("f", ev, dur=8.0)
    assert (t.start, t.end, t.window_end) == (0.5, 4.4, 8.0)
