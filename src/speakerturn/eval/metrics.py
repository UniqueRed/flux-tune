"""Turn-level metrics.

Per turn i: true end t_end, clip end clip_end, decision d (first decision at/after turn start,
or None).
  FC        : d < t_end - 0.100
  latency   : d - t_end for non-FC turns; None -> censored at clip end (clip_end - t_end)
  extension : latency > 1.5 s (primary) / 3.0 s (secondary), over non-FC turns incl. no-decision
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FC_TOL = 0.100
TER_PRIMARY = 1.5
TER_SECONDARY = 3.0


@dataclass
class TurnResult:
    turn_id: str
    t_start: float
    t_end: float
    clip_end: float
    decision: float | None
    strata: dict = field(default_factory=dict)


def turn_arrays(results: list[TurnResult]) -> dict[str, np.ndarray]:
    """Vectorize per-turn outcomes. All later statistics are computed from these arrays so that
    bootstrap resampling only needs index arrays."""
    n = len(results)
    fc = np.zeros(n, bool)
    lat = np.full(n, np.nan)
    nodec = np.zeros(n, bool)
    for i, r in enumerate(results):
        if r.decision is not None and r.decision < r.t_start:
            raise ValueError(f"{r.turn_id}: decision before turn start; harness must filter these")
        if r.decision is None:
            nodec[i] = True
            lat[i] = r.clip_end - r.t_end  # censored at clip end
        elif r.decision < r.t_end - FC_TOL:
            fc[i] = True
        else:
            lat[i] = r.decision - r.t_end
    return {"fc": fc, "lat": lat, "nodec": nodec}


def summarize(a: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict[str, float]:
    fc, lat, nodec = a["fc"], a["lat"], a["nodec"]
    if idx is not None:
        fc, lat, nodec = fc[idx], lat[idx], nodec[idx]
    n = len(fc)
    ok = ~fc
    L = lat[ok]
    out = {
        "n_turns": float(n),
        "fcr": float(fc.mean()) if n else float("nan"),
        "no_decision_rate": float(nodec.mean()) if n else float("nan"),
    }
    if len(L):
        out.update(
            ter_1500=float((L > TER_PRIMARY).mean()),
            ter_3000=float((L > TER_SECONDARY).mean()),
            lat_p50=float(np.percentile(L, 50)),
            lat_p90=float(np.percentile(L, 90)),
            lat_p95=float(np.percentile(L, 95)),
            lat_mean=float(L.mean()),
        )
    else:
        out.update({k: float("nan") for k in
                    ("ter_1500", "ter_3000", "lat_p50", "lat_p90", "lat_p95", "lat_mean")})
    return out


def evaluate(results: list[TurnResult]) -> dict[str, float]:
    return summarize(turn_arrays(results))


def stratify(results: list[TurnResult], key: str) -> dict[str, list[TurnResult]]:
    groups: dict[str, list[TurnResult]] = {}
    for r in results:
        groups.setdefault(str(r.strata.get(key)), []).append(r)
    return groups
