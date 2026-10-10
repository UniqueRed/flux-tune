"""Hand-built toy cases for every metric."""
import numpy as np
import pytest

from speakerturn.eval.bootstrap import (paired_bootstrap, rel_reduction, select_operating_point,
                                        unpaired_ratio)
from speakerturn.eval.metrics import TurnResult, evaluate, turn_arrays


def tr(d, t_end=5.0, clip_end=10.0, t_start=1.0, i=0):
    return TurnResult(f"t{i}", t_start, t_end, clip_end, d)


def test_false_cut_boundary():
    # exactly at t_end - 100 ms is NOT a false cut; just before is
    assert evaluate([tr(4.9)])["fcr"] == 0.0
    assert evaluate([tr(4.899)])["fcr"] == 1.0


def test_latency_and_negative_latency_allowed_within_tolerance():
    s = evaluate([tr(4.95), tr(5.5)])
    assert s["fcr"] == 0
    assert s["lat_p50"] == pytest.approx((-0.05 + 0.5) / 2)


def test_no_decision_is_censored_at_clip_end_and_counts_as_extension():
    a = turn_arrays([tr(None, t_end=5.0, clip_end=9.0)])
    assert a["lat"][0] == pytest.approx(4.0)
    s = evaluate([tr(None, t_end=5.0, clip_end=9.0)])
    assert s["ter_1500"] == 1.0 and s["ter_3000"] == 1.0 and s["no_decision_rate"] == 1.0


def test_ter_thresholds_and_excludes_false_cuts():
    rs = [tr(5.0 + 1.4, i=0), tr(5.0 + 1.6, i=1), tr(5.0 + 3.1, i=2), tr(2.0, i=3)]
    s = evaluate(rs)
    assert s["fcr"] == 0.25
    assert s["ter_1500"] == pytest.approx(2 / 3)   # over 3 non-FC turns
    assert s["ter_3000"] == pytest.approx(1 / 3)


def test_decision_before_turn_start_rejected():
    with pytest.raises(ValueError):
        evaluate([tr(0.5, t_start=1.0)])


def test_rel_reduction():
    assert rel_reduction(0.4, 0.2) == pytest.approx(0.5)
    assert rel_reduction(0.2, 0.4) == pytest.approx(-1.0)


def test_paired_bootstrap_detects_clear_improvement():
    base = [tr(5.0 + 2.0, i=i) for i in range(200)]
    new = [tr(5.0 + (2.0 if i < 40 else 0.3), i=i) for i in range(200)]
    r = paired_bootstrap(turn_arrays(base), turn_arrays(new), n_boot=300)
    assert r["point"] == pytest.approx(0.8)
    assert r["lo"] > 0.7 and r["hi"] <= 0.9


def test_paired_bootstrap_identical_systems_ci_is_zero():
    rng = np.random.default_rng(0)
    rs = [tr(5.0 + float(rng.uniform(0, 3)), i=i) for i in range(100)]
    a = turn_arrays(rs)
    r = paired_bootstrap(a, a, n_boot=200)
    assert r["lo"] == r["hi"] == 0.0


def test_unpaired_ratio():
    num = turn_arrays([tr(5 + (2.0 if i % 2 else 0.2), i=i) for i in range(400)])  # TER .5
    den = turn_arrays([tr(5 + (2.0 if i % 10 == 0 else 0.2), i=i) for i in range(400)])  # TER .1
    r = unpaired_ratio(num, den, n_boot=300)
    assert r["point"] == pytest.approx(5.0)
    assert r["lo"] > 2.0


def test_operating_point_selection():
    sweep = {
        "aggressive": turn_arrays([tr(3.0, i=0)] + [tr(5.2, i=i) for i in range(1, 10)]),  # FCR .1
        "mid": turn_arrays([tr(5.2, i=i) for i in range(9)] + [tr(7.0, i=9)]),            # FCR 0
        "slow": turn_arrays([tr(7.0, i=i) for i in range(10)]),                          # FCR 0
    }
    op = select_operating_point(sweep, 0.05)
    assert op["param"] == "mid" and op["reachable"]
    assert select_operating_point({"a": sweep["aggressive"]}, 0.05)["reachable"] is False


def test_group_bootstrap_resamples_whole_groups():
    from speakerturn.eval.bootstrap import bootstrap_metric
    # speaker A: 10 cut-offs; speaker B: 10 clean turns -> group resampling gives FCR in {0, .5, 1}
    a = turn_arrays([tr(2.0, i=i) for i in range(10)] + [tr(5.2, i=10 + i) for i in range(10)])
    turn = bootstrap_metric(a, "fcr")
    grp = bootstrap_metric(a, "fcr", groups=["A"] * 10 + ["B"] * 10)
    assert grp["point"] == turn["point"] == 0.5
    assert grp["hi"] - grp["lo"] > turn["hi"] - turn["lo"]
    one = bootstrap_metric(a, "fcr", groups=["A"] * 20)  # a single group: no between-group variance
    assert one["lo"] == one["hi"] == 0.5
