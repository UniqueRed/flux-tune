"""Label sanity: toy cases + checks over real AMI annotations when present."""
from pathlib import Path

import numpy as np
import pytest

from speakerturn.data.labeling import Word, spurts, turns

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "ami"


def ws(spk_words):
    return {k: [Word(w, s, e) for w, s, e in v] for k, v in spk_words.items()}


def test_backchannel_does_not_end_turn_and_hold_is_detected():
    d = ws({"A": [("so", 0, .3), ("we", .35, .6), ("should", .6, 1.0), ("then", 2.0, 2.4), ("go", 2.4, 3.0)],
            "B": [("yeah", 1.2, 1.5), ("okay", 3.5, 3.9), ("sure", 3.9, 4.2), ("thing", 4.2, 4.6), ("now", 4.6, 5.0)]})
    T = turns(spurts(d))
    a = T[0]
    assert a.spk == "A" and a.start == 0 and a.end == 3.0
    assert a.holds == [(1.0, 2.0)] and a.next_change == 3.5


def test_overlapped_ending_and_interruption():
    over = ws({"A": [("one", 0, .5), ("two", .5, 1.0), ("three", 1.0, 1.6)],
               "B": [("well", 1.2, 1.6), ("i", 1.6, 1.8), ("think", 1.8, 2.4)]})
    t = turns(spurts(over))[0]
    assert t.end == 1.6 and t.end_overlap and not t.interrupted
    intr = ws({"A": [("one", 0, .5), ("two", .5, 1.0), ("three", 1.0, 3.0)],
               "B": [("well", 1.2, 1.6), ("i", 1.6, 1.8), ("think", 1.8, 2.4)]})
    assert turns(spurts(intr))[0].interrupted


@pytest.fixture(scope="module")
def ami_turns():
    if not (ROOT / "words").exists():
        pytest.skip("AMI annotations not downloaded")
    from speakerturn.data.ami import load_meetings, meeting_turns
    M = load_meetings(ROOT)
    out = []
    for m in ["ES2004a", "IS1009b", "TS3003c", "EN2002a"]:
        T, words = meeting_turns(ROOT, m, M[m])
        out += [(t, words) for t in T]
    return out


def test_no_target_words_after_end_inside_turn(ami_turns):
    for t, words in ami_turns:
        if t.next_change is None:
            continue
        own_after = [w for w in words[t.spk] if t.end < w.s < t.next_change]
        assert not own_after, (t.spk, t.start, t.end)


def test_hold_pauses_exist_and_distributions_sane(ami_turns):
    usable = [t for t, _ in ami_turns if t.next_change is not None and not t.interrupted]
    assert sum(bool(t.holds) for t in usable) > 10
    dur = np.array([t.end - t.start for t in usable])
    gaps = np.array([t.next_change - t.end for t in usable])
    assert (dur > 0).all() and np.median(dur) < 10
    assert np.median(gaps) > -1.0 and np.median(gaps) < 3.0
    assert all(h[1] > h[0] for t in usable for h in t.holds)
