"""Async Deepgram Flux streaming client. Facts: docs/external_facts.md.

Sends a clip as 80 ms linear16 chunks at `pace` x real time, records every server message with
wall-clock (`wall_rel`, s since first audio byte) and audio time sent at arrival (`audio_sent`).
The API key is read from env/.env and never logged.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from websockets.exceptions import ConnectionClosed

from .cache import FluxCache, cache_key
from .cost import CostMeter

URL = "wss://api.deepgram.com/v2/listen"
MODEL = "flux-general-en"
SR = 16000
CHUNK = 1280  # 80 ms


def api_key() -> str:
    k = os.environ.get("DEEPGRAM_API_KEY")
    if not k:
        env = Path(__file__).resolve().parents[3] / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("DEEPGRAM_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"\'')
    if not k:
        raise RuntimeError("DEEPGRAM_API_KEY not set: copy .env.example to .env and add your key")
    return k


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2")


async def stream_clip(pcm16: np.ndarray, params: dict, pace: float = 1.0,
                      model: str = MODEL) -> list[dict]:
    import websockets

    q = {"model": model, "encoding": "linear16", "sample_rate": SR, **params}
    headers = {"Authorization": f"Token {api_key()}"}
    records: list[dict] = []
    state = {"t0": None, "sent": 0.0}
    async with websockets.connect(f"{URL}?{urlencode(q)}", additional_headers=headers,
                                  max_size=None, open_timeout=30) as ws:
        async def sender():
            state["t0"] = t0 = time.monotonic()
            for i in range(0, len(pcm16), CHUNK):
                c = pcm16[i:i + CHUNK]
                await ws.send(c.tobytes())
                state["sent"] = (i + len(c)) / SR
                delay = t0 + state["sent"] / pace - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            await ws.send(json.dumps({"type": "CloseStream"}))

        async def receiver():
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                now = time.monotonic()
                records.append({"msg": json.loads(raw),
                                "wall_rel": None if state["t0"] is None else now - state["t0"],
                                "audio_sent": state["sent"]})

        recv = asyncio.create_task(receiver())
        await sender()
        try:
            await asyncio.wait_for(recv, timeout=30)
        except asyncio.TimeoutError:
            recv.cancel()
    errors = [r["msg"] for r in records if r["msg"].get("type") == "Error"]
    if errors:
        raise RuntimeError(f"Flux error: {errors[0].get('code')}: {errors[0].get('description')}")
    return records


async def run_clip(audio: np.ndarray, params: dict, cache: FluxCache, meter: CostMeter,
                   pace: float = 1.0, model: str = MODEL) -> tuple[list[dict], bool]:
    """Returns (records, from_cache). Never pays twice for the same request. Pacing is not part
    of the key on purpose; the replay-speed finding decides whether paced and fast runs are
    interchangeable, and runs at a different pace use a distinct params entry `_pace`."""
    pcm = to_pcm16(audio)
    p = dict(params) | ({"_pace": pace} if pace != 1.0 else {})
    key = cache_key(pcm, p, model)
    hit = cache.get(key)
    if hit is not None:
        return FluxCache.strip_meta(hit), True
    dur = len(pcm) / SR
    meter.check(dur)
    records = await stream_clip(pcm, params, pace=pace, model=model)
    meter.record(dur)
    cache.put(key, records, meta={"params": p, "model": model, "dur": dur})
    return records, False


async def run_batch(items, params: dict, cache: FluxCache, meter: CostMeter,
                    concurrency: int = 8, pace: float = 1.0):
    """items: iterable of (clip_id, audio). Checks the whole batch's uncached cost first."""
    items = list(items)
    sem = asyncio.Semaphore(concurrency)

    async def one(cid, audio):
        async with sem:
            for attempt in range(3):
                try:
                    recs, hit = await run_clip(audio, params, cache, meter, pace)
                    return cid, recs, hit
                except (OSError, asyncio.TimeoutError, ConnectionClosed) as e:  # network; retry
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)

    uncached = sum(len(a) / SR for cid, a in items
                   if cache.get(cache_key(to_pcm16(a), dict(params) | ({"_pace": pace} if pace != 1.0 else {}), MODEL)) is None)
    meter.check(uncached)
    return await asyncio.gather(*(one(c, a) for c, a in items))
