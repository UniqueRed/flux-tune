"""Content-addressed cache of Flux event logs.

key = sha256(audio bytes + canonical request params + model id). Only the hash and the
gzip-JSONL event log are stored, never audio.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def cache_key(pcm16: np.ndarray, params: dict, model: str) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(pcm16, dtype="<i2").tobytes())
    h.update(json.dumps(params, sort_keys=True, separators=(",", ":")).encode())
    h.update(model.encode())
    return h.hexdigest()


class FluxCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.jsonl.gz"

    def get(self, key: str) -> list[dict] | None:
        p = self._path(key)
        if not p.exists():
            return None
        with gzip.open(p, "rt") as f:
            return [json.loads(line) for line in f]

    def put(self, key: str, records: list[dict], meta: dict | None = None) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            f.write(json.dumps({"_meta": meta or {}}) + "\n")
            for r in records:
                f.write(json.dumps(r) + "\n")
        tmp.replace(p)

    @staticmethod
    def strip_meta(records: list[dict]) -> list[dict]:
        return [r for r in records if "_meta" not in r]
