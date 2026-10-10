"""Deepgram spend meter. Persists cumulative audio-seconds and estimated USD.

Rate: regular pay-as-you-go Flux English $0.0077/min (conservative; promo is $0.0065/min).
Source: docs/external_facts.md §8, verified 2026-10-02. Other API costs (Flux TTS at $0.045 per
1,000 characters) are added via record_usd. The ledger is re-read before every write so separate
processes (voice lab server, flux-tune) don't overwrite each other.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

USD_PER_MIN = 0.0077
TTS_USD_PER_1K_CHARS = 0.045


class BudgetExceeded(RuntimeError):
    pass


class CostMeter:
    def __init__(self, path: str | Path, cap_usd: float, usd_per_min: float = USD_PER_MIN):
        self.path, self.cap, self.rate = Path(path), cap_usd, usd_per_min
        self._lock = threading.Lock()
        self.state = {"audio_seconds": 0.0, "usd": 0.0, "requests": 0, "usd_per_min": usd_per_min,
                      "other_usd": 0.0}
        self._reload()

    def _reload(self) -> None:
        if self.path.exists():
            self.state.update(json.loads(self.path.read_text()))
            self.state.setdefault("other_usd", 0.0)

    def _total(self) -> float:
        return self.cost(self.state["audio_seconds"]) + self.state.get("other_usd", 0.0)

    def cost(self, seconds: float) -> float:
        return seconds / 60.0 * self.rate

    def remaining(self) -> float:
        return self.cap - self.state["usd"]

    def check(self, planned_seconds: float, planned_usd: float = 0.0) -> None:
        """Raise before a batch if it would exceed the cap."""
        self._reload()
        if self.state["usd"] + self.cost(planned_seconds) + planned_usd > self.cap:
            raise BudgetExceeded(
                f"planned ${self.cost(planned_seconds):.2f} + spent ${self.state['usd']:.2f} > cap ${self.cap:.2f}")

    def record(self, seconds: float) -> None:
        with self._lock:
            self._reload()
            self.state["audio_seconds"] += seconds
            self.state["usd"] = self._total()
            self.state["requests"] += 1
            self._write()

    def record_usd(self, usd: float) -> None:
        with self._lock:
            self._reload()
            self.state["other_usd"] = self.state.get("other_usd", 0.0) + usd
            self.state["usd"] = self._total()
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.path)
