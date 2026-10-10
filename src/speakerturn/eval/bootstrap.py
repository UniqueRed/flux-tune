"""Turn-level bootstrap CIs and operating-point matching.

A criterion is met only if the CI bound satisfies it, not just the point estimate.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .metrics import summarize

N_BOOT = 1000


def _ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if not len(s):
        return float("nan"), float("nan")
    return float(np.percentile(s, 100 * alpha / 2)), float(np.percentile(s, 100 * (1 - alpha / 2)))


def _sampler(n: int, groups, rng):
    """Resample turns, or whole groups of turns (e.g. speakers) when `groups` gives each turn's group."""
    if groups is None:
        return lambda: rng.integers(0, n, n)
    g = np.asarray(groups)
    members = [np.flatnonzero(g == u) for u in np.unique(g)]
    return lambda: np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])


def bootstrap_metric(a: dict, metric: str, n_boot: int = N_BOOT, seed: int = 1337,
                     alpha: float = 0.05, groups=None) -> dict:
    """CI for one system's metric, resampling turns (or groups of turns, see _sampler)."""
    rng = np.random.default_rng(seed)
    draw = _sampler(len(a["fc"]), groups, rng)
    point = summarize(a)[metric]
    boots = np.array([summarize(a, draw())[metric] for _ in range(n_boot)])
    lo, hi = _ci(boots, alpha)
    return {"point": point, "lo": lo, "hi": hi, "n_boot": n_boot}


def rel_reduction(base: float, new: float) -> float:
    """Relative reduction of `new` vs `base`: 1 - new/base. Positive = improvement."""
    if base == 0:
        return float("nan") if new == 0 else -np.inf
    return 1.0 - new / base


def paired_bootstrap(a_base: dict, a_new: dict, metric: str = "ter_1500",
                     stat: Callable[[float, float], float] = rel_reduction,
                     n_boot: int = N_BOOT, seed: int = 1337, alpha: float = 0.05, groups=None) -> dict:
    """Paired turn-level (or group-level) bootstrap of stat(metric_base, metric_new). Both systems
    must be evaluated on the same turns in the same order."""
    n = len(a_base["fc"])
    if len(a_new["fc"]) != n:
        raise ValueError("paired bootstrap requires identical turn sets")
    rng = np.random.default_rng(seed)
    draw = _sampler(n, groups, rng)
    point = stat(summarize(a_base)[metric], summarize(a_new)[metric])
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = draw()
        boots[b] = stat(summarize(a_base, idx)[metric], summarize(a_new, idx)[metric])
    lo, hi = _ci(boots, alpha)
    return {"metric": metric, "point": point, "lo": lo, "hi": hi, "n_boot": n_boot, "n_turns": n}


def unpaired_ratio(a_num: dict, a_den: dict, metric: str = "ter_1500", n_boot: int = N_BOOT,
                   seed: int = 1337, alpha: float = 0.05) -> dict:
    """CI for metric(num)/metric(den) over disjoint turn sets (e.g., interference vs clean)."""
    rng = np.random.default_rng(seed)
    n1, n2 = len(a_num["fc"]), len(a_den["fc"])
    point = summarize(a_num)[metric] / max(summarize(a_den)[metric], 1e-12)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        num = summarize(a_num, rng.integers(0, n1, n1))[metric]
        den = summarize(a_den, rng.integers(0, n2, n2))[metric]
        boots[b] = num / den if den > 0 else np.inf
    lo, hi = _ci(np.where(np.isinf(boots), np.nan, boots), alpha)
    return {"metric": metric, "point": point, "lo": lo, "hi": hi,
            "n_inf": int(np.isinf(boots).sum()), "n_boot": n_boot}


def select_operating_point(sweep: dict, target_fcr: float) -> dict:
    """Pick a system's operating point on VALIDATION for a target FCR.

    sweep: {param_key: turn_arrays}. Among points with FCR <= target, choose minimum TER(1500),
    tie-break by lower median latency. If none reaches the target, return the minimum-FCR point
    flagged `reachable=False` (reported, never silently compared).
    """
    rows = [(k, summarize(a)) for k, a in sweep.items()]
    feasible = [r for r in rows if r[1]["fcr"] <= target_fcr]
    if feasible:
        k, s = min(feasible, key=lambda r: (r[1]["ter_1500"], r[1]["lat_p50"]))
        return {"param": k, "summary": s, "reachable": True, "target_fcr": target_fcr}
    k, s = min(rows, key=lambda r: (r[1]["fcr"], r[1]["ter_1500"]))
    return {"param": k, "summary": s, "reachable": False, "target_fcr": target_fcr}


def frontier(sweep: dict) -> list[dict]:
    """FCR vs latency/TER points for frontier plots, sorted by FCR."""
    pts = [{"param": k, **summarize(a)} for k, a in sweep.items()]
    return sorted(pts, key=lambda p: p["fcr"])
