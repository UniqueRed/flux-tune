"""Voice lab: talk to a scripted voice agent in the browser; recordings feed flux-tune.

Runs locally only (browsers allow the microphone on localhost, and the Deepgram key stays here).
  GET  /               the page (static/index.html)
  GET  /api/prompts    agent questions
  POST /api/tts        {"text", "voice"} -> agent audio from Flux TTS (/v2/speak), cached on disk
  WS   /ws             browser -> 16 kHz int16 mic chunks + in-band markers; server -> live Flux events

In-band markers ({"type": "agent_start"} / {"type": "agent_end"}) arrive in order with the audio, so
each is stamped with the exact audio time = samples received so far. Those agent intervals become
flux-tune's turn boundaries (--agent_times). The agent waits for the user to press Next rather than
replying on Flux's EndOfTurn, so the recordings are not shaped by the threshold being tested.

Saved per session (data/voice_lab/<condition>/):
  <id>.wav            the user's mic, continuous (zeros while the agent speaks if muting is on)
  agent_times.json    {<id>: [[start, end], ...]} merged across sessions
  <id>.meta.json      prompts, intervals, live Flux log summary
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import aiohttp
import numpy as np
import soundfile as sf
from aiohttp import web

from ..flux.client import MODEL, URL, api_key
from ..flux.cost import TTS_USD_PER_1K_CHARS, BudgetExceeded, CostMeter
from .prompts import PROMPTS

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("VOICE_LAB_DATA", ROOT / "data" / "voice_lab"))
STATIC = Path(__file__).parent / "static"
SR = 16000
LIVE_PARAMS = {"eot_threshold": 0.7}  # display only; flux-tune re-runs every threshold afterwards
TTS_FALLBACK = "aura-2-thalia-en"


def meter() -> CostMeter:
    return CostMeter(ROOT / "experiments" / "api_spend.json", cap_usd=40.0)


async def index(request):
    return web.FileResponse(STATIC / "index.html")


async def prompts(request):
    return web.json_response({"prompts": PROMPTS})


async def tts(request):
    body = await request.json()
    text, voice = body["text"].strip()[:500], body.get("voice", "flux-haley-en")
    cache = DATA / "tts" / f"{hashlib.sha256(f'{voice}|{text}'.encode()).hexdigest()[:24]}.mp3"
    if not cache.exists():
        m = meter()
        usd = len(text) / 1000 * TTS_USD_PER_1K_CHARS
        try:
            m.check(0, planned_usd=usd)
        except BudgetExceeded as e:
            return web.json_response({"error": str(e)}, status=402)
        headers = {"Authorization": f"Token {api_key()}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as s:
            for url in (f"https://api.deepgram.com/v2/speak?model={voice}",
                        f"https://api.deepgram.com/v1/speak?model={TTS_FALLBACK}"):
                async with s.post(url, json={"text": text}, headers=headers) as r:
                    if r.status == 200:
                        audio = await r.read()
                        break
            else:
                return web.json_response({"error": f"TTS failed: HTTP {r.status}"}, status=502)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(audio)
        m.record_usd(usd)
    return web.FileResponse(cache, headers={"Content-Type": "audio/mpeg"})


class Session:
    def __init__(self, condition: str):
        self.condition = "".join(c for c in condition if c.isalnum() or c in "-_") or "quiet"
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.chunks: list[np.ndarray] = []
        self.n = 0
        self.agent: list[list[float]] = []
        self.prompts: list[str] = []
        self.flux_log: list[dict] = []

    @property
    def t(self) -> float:
        return self.n / SR

    def save(self) -> dict:
        d = DATA / self.condition
        d.mkdir(parents=True, exist_ok=True)
        audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.int16)
        sf.write(d / f"{self.id}.wav", audio, SR, subtype="PCM_16")
        # an agent turn still open when the session stops ends at the session end
        intervals = [[round(a, 3), round(b if b is not None else self.t, 3)] for a, b in self.agent
                     if (b if b is not None else self.t) > a]
        at = d / "agent_times.json"
        merged = json.loads(at.read_text()) if at.exists() else {}
        merged[self.id] = intervals
        at.write_text(json.dumps(merged, indent=1))
        eots = [e for e in self.flux_log if e.get("event") == "EndOfTurn"]
        meta = {"id": self.id, "condition": self.condition, "duration_s": round(self.t, 2),
                "prompts": self.prompts, "agent_intervals": intervals, "live_params": LIVE_PARAMS,
                "live_eots": [round(e["audio_window_end"], 2) for e in eots]}
        (d / f"{self.id}.meta.json").write_text(json.dumps(meta, indent=1))
        with open(d / f"{self.id}.flux_live.jsonl", "w") as f:
            for e in self.flux_log:
                f.write(json.dumps(e) + "\n")
        return meta


async def ws_handler(request):
    import websockets

    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    first = await ws.receive_json()
    sess = Session(first.get("condition", "quiet"))
    m = meter()
    try:
        m.check(15 * 60)  # room for a 15-minute session
    except BudgetExceeded as e:
        await ws.send_json({"type": "error", "message": str(e)})
        return ws
    q = {"model": MODEL, "encoding": "linear16", "sample_rate": SR, **LIVE_PARAMS}
    flux = await websockets.connect(f"{URL}?{urlencode(q)}", additional_headers={"Authorization": f"Token {api_key()}"},
                                    max_size=None, open_timeout=30)
    await ws.send_json({"type": "ready", "session": sess.id})

    async def relay():
        async for raw in flux:
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            if msg.get("type") != "TurnInfo":
                continue
            e = {"event": msg.get("event"), "audio_window_end": float(msg.get("audio_window_end") or 0),
                 "conf": float(msg.get("end_of_turn_confidence") or 0), "transcript": msg.get("transcript", ""),
                 "turn_index": msg.get("turn_index", 0)}
            sess.flux_log.append(e)
            if not ws.closed:
                await ws.send_json({"type": "flux", **e})

    relay_task = asyncio.create_task(relay())
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                pcm = np.frombuffer(msg.data, dtype="<i2")
                sess.chunks.append(pcm.copy())
                sess.n += len(pcm)
                await flux.send(msg.data)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                c = json.loads(msg.data)
                if c["type"] == "agent_start":
                    sess.agent.append([sess.t, None])
                    sess.prompts.append(c.get("prompt", ""))
                elif c["type"] == "agent_end" and sess.agent and sess.agent[-1][1] is None:
                    sess.agent[-1][1] = sess.t
                elif c["type"] == "stop":
                    break
    finally:
        try:
            await flux.send(json.dumps({"type": "CloseStream"}))
            await asyncio.wait_for(relay_task, timeout=10)
        except Exception:
            relay_task.cancel()
        await flux.close()
        if sess.n:
            m.record(sess.t)
            meta = sess.save()
            if not ws.closed:
                await ws.send_json({"type": "saved", **meta})
        await ws.close()
    return ws


def make_app() -> web.Application:
    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/api/prompts", prompts), web.post("/api/tts", tts),
                    web.get("/ws", ws_handler)])
    return app
