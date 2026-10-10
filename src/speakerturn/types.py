"""Core streaming contracts.

Every streaming component implements a causal step interface so the same code runs in
evaluation replay and live. Ground truth is never visible to controllers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

SR = 16000
CHUNK_MS = 80
CHUNK_SAMPLES = SR * CHUNK_MS // 1000  # 1280


@dataclass
class ClipMeta:
    """From the manifest. Only `clip_id` and `sr` are public; GT lives elsewhere."""
    clip_id: str
    sr: int = SR


@dataclass
class FluxWord:
    word: str
    confidence: float | None = None
    start: float | None = None  # audio-time seconds, if Flux provides them
    end: float | None = None


@dataclass
class FluxEvent:
    """One Flux TurnInfo message, normalized.

    `t_avail` is the audio-time (s, relative to clip start) at which a causal system could
    have this event in hand: max(audio_window_end, wall-clock arrival mapped to audio time).
    The replay harness only delivers events with t_avail <= t.
    """
    event: str  # StartOfTurn | Update | EagerEndOfTurn | TurnResumed | EndOfTurn
    t_avail: float
    turn_index: int = 0
    audio_window_start: float = 0.0
    audio_window_end: float = 0.0
    transcript: str = ""
    words: list[FluxWord] = field(default_factory=list)
    end_of_turn_confidence: float | None = None
    raw: dict | None = None


@dataclass
class Decision:
    t: float
    reason: str = ""
    score: float | None = None


@runtime_checkable
class TurnController(Protocol):
    def reset(self, meta_public: dict) -> None: ...

    def step(self, t: float, audio: np.ndarray, flux_events: list[FluxEvent]) -> Decision | None:
        """t = end time (s) of `audio` chunk. `flux_events` = events newly available in (t-chunk, t].
        Return Decision to declare end-of-turn, else None. Must be causal."""
        ...


@runtime_checkable
class TargetProfile(Protocol):
    def update(self, t: float, audio: np.ndarray) -> None: ...

    def embedding(self) -> np.ndarray | None: ...


# Keys allowed in meta_public. Anything else is a ground-truth leak.
PUBLIC_META_KEYS = frozenset({"clip_id", "sr", "agent_turn_end", "agent_last_utterance", "mode"})


def public_meta(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k in PUBLIC_META_KEYS}
