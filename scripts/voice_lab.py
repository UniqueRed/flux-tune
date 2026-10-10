"""Start the voice lab: http://localhost:8765 (use headphones so the agent's voice stays out of the mic).
Usage: python scripts/voice_lab.py [--port 8765]"""
import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aiohttp import web  # noqa: E402
from speakerturn.voicelab.server import make_app  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://localhost:{a.port}"
    print(f"Voice lab running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    web.run_app(make_app(), host="127.0.0.1", port=a.port, print=None)
