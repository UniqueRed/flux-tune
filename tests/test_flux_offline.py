"""Offline tests for Flux parsing, cache, cost meter (no network, no key)."""
import numpy as np
import pytest

from speakerturn.flux.cache import FluxCache, cache_key
from speakerturn.flux.cost import BudgetExceeded, CostMeter
from speakerturn.flux.events import parse_log


def rec(event, awe, conf, wall=None, words=None, typ="TurnInfo"):
    return {"msg": {"type": typ, "event": event, "turn_index": 0, "audio_window_start": 0,
                    "audio_window_end": str(awe), "transcript": "hi", "words": words or [],
                    "end_of_turn_confidence": str(conf)},
            "wall_rel": wall if wall is not None else awe, "audio_sent": awe}


def test_parse_handles_string_floats_and_policies():
    recs = [{"msg": {"type": "Connected"}, "wall_rel": 0.0},
            rec("Update", 0.48, 0.1, wall=0.6),
            rec("EndOfTurn", 1.2, 0.8, wall=1.1, words=[{"word": "hi", "confidence": "0.9", "start": 0.4, "end": 0.6}])]
    ev = parse_log(recs)
    assert [e.event for e in ev] == ["Update", "EndOfTurn"]
    assert ev[0].t_avail == pytest.approx(0.6)        # wall later than window end
    assert ev[1].t_avail == pytest.approx(1.2)        # never earlier than window end
    assert ev[1].words[0].start == 0.4 and ev[1].end_of_turn_confidence == 0.8
    ev2 = parse_log(recs, policy="offset", latency_offset=0.15)
    assert ev2[0].t_avail == pytest.approx(0.63)


def test_cache_key_depends_on_audio_params_model(tmp_path):
    a = np.zeros(100, "<i2")
    b = a.copy(); b[5] = 1
    k = cache_key(a, {"eot_threshold": 0.7}, "m")
    assert k != cache_key(b, {"eot_threshold": 0.7}, "m")
    assert k != cache_key(a, {"eot_threshold": 0.8}, "m")
    assert k != cache_key(a, {"eot_threshold": 0.7}, "m2")
    c = FluxCache(tmp_path)
    assert c.get(k) is None
    c.put(k, [rec("Update", 0.5, 0.1)], meta={"dur": 1})
    assert FluxCache.strip_meta(c.get(k))[0]["msg"]["event"] == "Update"


def test_cost_meter_enforces_cap_and_persists(tmp_path):
    p = tmp_path / "spend.json"
    m = CostMeter(p, cap_usd=1.0, usd_per_min=0.5)
    m.check(60)
    m.record(60)
    assert m.state["usd"] == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded):
        m.check(61)
    assert CostMeter(p, cap_usd=1.0, usd_per_min=0.5).state["usd"] == pytest.approx(0.5)


def test_cost_meter_other_usd_and_cross_process_reload(tmp_path):
    p = tmp_path / "spend.json"
    a, b = CostMeter(p, 1.0, usd_per_min=0.6), CostMeter(p, 1.0, usd_per_min=0.6)
    a.record(60)           # $0.60 audio
    b.record_usd(0.25)     # another process adds TTS spend without clobbering a's audio
    assert CostMeter(p, 1.0, usd_per_min=0.6).state["usd"] == pytest.approx(0.85)
    with pytest.raises(BudgetExceeded):
        a.check(0, planned_usd=0.2)
