"""Analyze everything recorded in the voice lab and rebuild the report.

Quiet sessions pick the recommended setting (turn-level tune/report split) and get the synthetic
background-talker test; background sessions are scored as real-world evidence at the recommended
and default settings. Output: reports/voice_lab/{results.json, report.html}
Usage: python scripts/analyze_lab.py [--target 0.10]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "voice_lab"
PY = sys.executable

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=float, default=0.10)
a = ap.parse_args()
quiet, bg = LAB / "quiet", LAB / "background"
n_q = len(list(quiet.glob("*.wav"))) if quiet.exists() else 0
n_b = len(list(bg.glob("*.wav"))) if bg.exists() else 0
if not n_q:
    sys.exit("No quiet-room sessions yet: record one in the voice lab first (python scripts/voice_lab.py).")
cmd = [PY, str(ROOT / "scripts/flux_tune.py"), str(quiet), "--agent_times", str(quiet / "agent_times.json"),
       "--split", "turns", "--target", str(a.target), "--noise", "--yes", "--out", str(ROOT / "reports/voice_lab")]
if n_b:
    cmd += ["--eval_dir", str(bg), "--eval_agent_times", str(bg / "agent_times.json")]
subprocess.run(cmd, check=True)
mins = sum(p.stat().st_size for p in quiet.glob("*.wav")) / 32000 / 60
subprocess.run([PY, str(ROOT / "scripts/build_report.py"), str(ROOT / "reports/voice_lab"), "--title", "flux-tune Voice Lab",
                "--dataset", f"your own voice: {n_q} quiet session(s), {n_b} with background audio",
                "--caveat", "These are one person's recordings from the voice lab: one speaker, one microphone, a small number of turns, "
                "so the confidence intervals are wide. The background-talker test mixes a recorded AMI speaker in digitally; "
                "the real-background section uses sessions recorded with audio playing in the room. Read this alongside the AMI "
                "and SpokenWOZ benchmarks rather than on its own."], check=True)
print("Open reports/voice_lab/report.html")
