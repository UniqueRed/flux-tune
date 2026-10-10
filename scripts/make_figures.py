"""Figures for the write-up (writeup/figures/*.png), from results.json files and cached Flux passes.

Every number comes from reports/{ami_bench,spokenwoz}/results.json or the Flux cache; no API spend
(all passes are cached). Light theme for Medium; palette = dataviz reference instance (validated).
Usage: python scripts/make_figures.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flux_tune import SR, load_audio, passes  # noqa: E402
from speakerturn.flux.cache import FluxCache  # noqa: E402
from speakerturn.flux.cost import CostMeter  # noqa: E402
from speakerturn.fluxtune.core import agent_turns, score  # noqa: E402
from speakerturn.fluxtune.simulate import real_eots  # noqa: E402

OUT = ROOT / "writeup" / "figures"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = {"250": "#86b6ef", "400": "#3987e5", "600": "#184f95"}

plt.rcParams.update({
    "font.family": "Inter", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 200,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.titleweight": "semibold", "axes.titlesize": 13,
    "axes.titlelocation": "left", "axes.titlepad": 12,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


def results(name):
    return json.loads((ROOT / "reports" / name / "results.json").read_text())


def cached(d: Path, files, ths):
    cache = FluxCache(ROOT / "data" / "flux_cache")
    meter = CostMeter(ROOT / "experiments" / "api_spend.json", cap_usd=40.0)
    audio = {f: load_audio(d / f"{f}.flac") for f in files}
    items = list(audio.items())
    ev = {th: asyncio.run(passes(items, {"eot_threshold": th}, cache, meter, 12)) for th in ths}
    return audio, ev


# ---------------------------------------------------------------- 1. the problem, on one real turn
def fig_hero():
    d = ROOT / "data" / "ami_bench"
    f = "report_FEE051_2"
    audio, ev = cached(d, [f], (1.0, 0.7, 0.94))
    ag = json.loads((d / "agent_times.json").read_text())
    turns = agent_turns(f, ev[1.0][f], len(audio[f]) / SR, ag[f])
    t = next(t for t in turns if abs(t.start - 17.76) < 0.05)
    d7 = next(x for x in real_eots(ev[0.7][f]) if x >= t.start)
    d94 = next(x for x in real_eots(ev[0.94][f]) if x >= t.start)
    lo, hi = t.start - 0.25, d94 + 0.7
    conf = [(e.audio_window_end, e.end_of_turn_confidence) for e in ev[1.0][f]
            if e.end_of_turn_confidence is not None and t.start + 0.4 <= e.audio_window_end <= hi]
    x, y = zip(*conf)

    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(10, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.5, 3.2], "hspace": 0.06})
    phr, cur = [], [t.words[0]]
    for w in t.words[1:]:
        if w.start - cur[-1].end >= 0.35:
            phr.append(cur)
            cur = [w]
        else:
            cur.append(w)
    phr.append(cur)
    for k, p in enumerate(phr):
        ax0.add_patch(FancyBboxPatch((p[0].start, 0.12), p[-1].end - p[0].start, 0.42,
                                     boxstyle="round,pad=0,rounding_size=0.08", fc="#d9d8d3", ec="none"))
        import textwrap
        room = min(phr[k + 1][0].start if k + 1 < len(phr) else d94, d7 if p[0].start < d7 else d94) - p[0].start
        said = textwrap.fill(" ".join(w.word for w in p), width=max(12, int(room * 8)))
        ax0.text(p[0].start, 0.68, f"“{said}”", fontsize=9.5, va="bottom", linespacing=1.35,
                 color=INK if p[-1].end <= d7 + 0.05 else INK2, style="normal" if p[-1].end <= d7 + 0.05 else "italic")
    ax0.set_ylim(0, 1.95)
    ax0.set_yticks([])
    ax0.grid(False)
    for sp in ("left", "bottom"):
        ax0.spines[sp].set_visible(False)

    ax.plot(x, y, color=BLUE, lw=2, solid_capstyle="round", zorder=3)
    ax.scatter(x, y, s=9, color=BLUE, zorder=3)
    for th, c, lab in ((0.7, ORANGE, "Default threshold 0.7"), (0.94, AQUA, "Tuned threshold 0.94")):
        ax.axhline(th, color=c, lw=1.5, ls=(0, (4, 3)), zorder=2)
        ax.text(lo + 0.05, th + 0.015, lab, color=INK2, fontsize=9.5, ha="left", va="bottom")
    ax.axvline(d7, color=ORANGE, lw=2, zorder=2)
    ax0.axvline(d7, color=ORANGE, lw=2)
    ax.axvline(d94, color=AQUA, lw=2, zorder=2)
    ax0.axvline(d94, color=AQUA, lw=2)
    ax.annotate("Default (0.7) ends the turn\nduring the pause, cutting the\nspeaker off mid-thought", xy=(d7, 0.42),
                xytext=(d7 + 1.85, 0.42), fontsize=9.5, va="center", linespacing=1.35,
                arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1, "mutation_scale": 10})
    ax.text(d94 + 0.1, 0.40, "Tuned setting waits\nfor the real end", fontsize=9.5, va="center")
    ax.set_ylim(0, 1.04)
    ax.set_xlim(lo, hi + 1.5)
    ax.set_ylabel("Flux end-of-turn confidence")
    ticks = np.arange(0, hi - t.start + 0.01, 1.0)
    ax.set_xticks(t.start + ticks)
    ax.set_xticklabels([f"{v:.0f} s" for v in ticks])
    ax.spines["bottom"].set_bounds(lo, hi)
    fig.suptitle("A pause is not the end of a turn", x=0.125, ha="left", fontsize=14, fontweight="semibold", y=1.02)
    fig.text(0.125, 0.94, "One real turn from the AMI meeting corpus, run through Deepgram Flux", fontsize=10, color=INK2)
    save(fig, "01_pause_vs_end.png")


# ---------------------------------------------------------------- 2. pipeline diagram
def fig_pipeline():
    steps = [("Real calls", "User audio plus the\ntimes the agent spoke"),
             ("7 Flux passes", "eot_threshold\nfrom 0.6 to 0.98"),
             ("Score each turn", "Cut off? How long\nuntil the reply?"),
             ("Choose", "Best setting on\nthe tune speakers"),
             ("Verify", "On held-out speakers,\nwith 95% CIs")]
    W, H, gap, pad = 1.9, 1.55, 0.36, 0.2  # inches: card width/height, gap, inner padding
    n = len(steps)
    fw, fh = n * W + (n - 1) * gap, H + 0.65
    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.axis("off")
    ax.text(0, fh - 0.12, "How flux-tune picks a setting", fontsize=13, fontweight="semibold", va="top")
    top = H
    texts = []
    for i, (h, b) in enumerate(steps):
        x0 = i * (W + gap)
        hl = i == n - 1
        ax.add_patch(FancyBboxPatch((x0, 0), W, H, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="#eaf2fc" if hl else "#f4f4f2", ec=BLUE if hl else GRID, lw=1.4))
        ax.text(x0 + pad, top - pad, f"{i + 1:02d}", fontsize=9, color=BLUE, fontweight="semibold", va="top")
        texts.append(ax.text(x0 + pad, top - pad - 0.3, h, fontsize=12, fontweight="semibold", va="top"))
        texts.append(ax.text(x0 + pad, top - pad - 0.68, b, fontsize=9.5, color=INK2, va="top", linespacing=1.45))
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x0 + W + 0.07, H / 2), (x0 + W + gap - 0.07, H / 2),
                                         arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.4))
    # every text must keep at least `pad` from its card's right and bottom edges
    r = fig.canvas.get_renderer()
    for k, t in enumerate(texts):
        x0 = (k // 2) * (W + gap)
        bb = ax.transData.inverted().transform(t.get_window_extent(renderer=r))
        assert bb[1][0] <= x0 + W - pad + 1e-6 and bb[0][1] >= pad - 1e-6, (t.get_text(), bb)
    fig.savefig(OUT / "02_pipeline.png", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / "02_pipeline.png")


# ---------------------------------------------------------------- 3. trade-off curves, two corpora
def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sets = [("ami_bench", "Meetings (AMI, 43 held-out speakers)", BLUE, 0.94),
            ("spokenwoz", "Phone calls (SpokenWOZ, 20 held-out calls)", ORANGE, 0.86)]
    for name, lab, c, rec in sets:
        cur = results(name)["curve_real"]["report"]
        xs = [p["lat_p50"] for p in cur]
        ys = [100 * p["fcr"] for p in cur]
        ax.plot(xs, ys, color=c, lw=2, label=lab, zorder=3)
        ax.scatter(xs, ys, s=64, color=c, edgecolor=SURF, linewidth=2, zorder=4)
        for p in cur:
            if p["threshold"] in (0.7, rec):
                tag = "default 0.7" if p["threshold"] == 0.7 else f"tuned {rec}"
                txt = f"{tag}\n{100 * p['fcr']:.0f}% cut off · {p['lat_p50']:.2f} s"
                if name == "ami_bench":
                    ax.annotate(txt, (p["lat_p50"], 100 * p["fcr"]), xytext=(14, 6), textcoords="offset points",
                                fontsize=9.5, color=INK, linespacing=1.3)
                else:  # below the orange curve, where the chart is empty
                    at = (0.06, 4.2) if p["threshold"] == 0.7 else (0.8, 1.0)
                    ax.annotate(txt, (p["lat_p50"], 100 * p["fcr"]), xytext=at, textcoords="data", fontsize=9.5,
                                color=INK, linespacing=1.3,
                                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9, "shrinkB": 8})
                ax.scatter([p["lat_p50"]], [100 * p["fcr"]], s=150, facecolor="none", edgecolor=c, lw=1.6, zorder=5)
    ax.set_xlabel("Median wait after the speaker finishes (seconds)")
    ax.set_ylabel("Turns cut off (%)")
    ax.set_xlim(0, 3.4)
    ax.set_ylim(0, 37)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.set_title("Every setting trades interruptions for waiting, and the best one depends on the audio")
    ax.text(0, -0.17, "Each point is one real Flux pass at eot_threshold 0.6, 0.7, 0.8, 0.86, 0.9, 0.94, 0.98 (left to right). "
            "Held-out data only.", transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, "03_tradeoff.png")


# ---------------------------------------------------------------- 4. per-speaker effect (AMI held-out)
def fig_speakers():
    d = ROOT / "data" / "ami_bench"
    sm = json.loads((d / "split.json").read_text())
    groups = json.loads((d / "groups.json").read_text())
    files = sorted(f for f in sm if sm[f] == "report")
    audio, ev = cached(d, files, (1.0, 0.7, 0.94))
    ag = json.loads((d / "agent_times.json").read_text())
    T = [t for f in files for t in agent_turns(f, ev[1.0][f], len(audio[f]) / SR, ag.get(f, []))]
    rate = {}
    for th in (0.7, 0.94):
        det = score(T, {f: real_eots(ev[th][f]) for f in files})[1]
        by = {}
        for x in det:
            by.setdefault(groups[x["file"]], []).append(x["cutoff"])
        rate[th] = {s: 100 * np.mean(v) for s, v in by.items()}
    spk = sorted(rate[0.7], key=lambda s: rate[0.7][s])
    better = sum(rate[0.94][s] < rate[0.7][s] for s in spk)
    same = sum(rate[0.94][s] == rate[0.7][s] for s in spk)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    yy = np.arange(len(spk))
    for i, s in enumerate(spk):
        ax.plot([rate[0.94][s], rate[0.7][s]], [i, i], color="#cfcec9", lw=2, zorder=2, solid_capstyle="round")
    ax.scatter([rate[0.7][s] for s in spk], yy, s=42, color=ORANGE, label="Default 0.7", zorder=3, edgecolor=SURF, lw=1)
    ax.scatter([rate[0.94][s] for s in spk], yy, s=42, color=AQUA, label="Tuned 0.94", zorder=4, edgecolor=SURF, lw=1)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Turns cut off for that speaker (%)")
    ax.set_ylabel(f"{len(spk)} held-out speakers, sorted")
    ax.set_xlim(-2, 80)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    ax.set_title(f"Fewer cut-offs for {better} of {len(spk)} speakers the tool never saw ({same} unchanged)")
    ax.text(0, -0.15, "AMI test meetings, about 18 turns per speaker. The setting was chosen on 28 other speakers.",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, "04_per_speaker.png")
    return better, same, len(spk)


# ---------------------------------------------------------------- 5. background speech
def fig_background():
    r = results("ami_bench")
    by = {p["threshold"]: p for p in r["curve_real"]["report"]}
    n = r["noise"]
    conds = [("Quiet", lambda th: by[th]["no_decision_rate"], RAMP["250"]),
             ("Background talker 10 dB quieter", lambda th: n[f"th{th}_tir10dB"]["no_decision_rate"]["point"], RAMP["400"]),
             ("Background talker as loud", lambda th: n[f"th{th}_tir0dB"]["no_decision_rate"]["point"], RAMP["600"])]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ths = [0.7, 0.94]
    bw = 0.24
    for k, (lab, fn, c) in enumerate(conds):
        xs = np.arange(len(ths)) + (k - 1) * (bw + 0.02)
        vals = [100 * fn(th) for th in ths]
        ax.bar(xs, vals, width=bw, color=c, label=lab, zorder=3)
        for xv, v in zip(xs, vals):
            ax.text(xv, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(ths)))
    ax.set_xticklabels(["Default (0.7)", "Tuned (0.94)"], fontsize=11, color=INK)
    ax.set_ylabel("Turns that never ended (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_title("Background speech keeps turns open, and a higher threshold makes it worse")
    ax.text(0, -0.15, "AMI held-out speakers (787 turns) with a second AMI speaker mixed in behind them. "
            "'Never ended' = no EndOfTurn before the speaker's next turn.", transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, "05_background.png")


# ---------------------------------------------------------------- 6. a voice-lab session timeline
def fig_session():
    base = ROOT / "data" / "voice_lab"
    panels = []
    for cond in ("quiet", "background"):
        meta = sorted((base / cond).glob("*.meta.json"))[-1]
        sid = meta.name.split(".")[0]
        x, sr = sf.read(base / cond / f"{sid}.wav", dtype="float32")
        live = [json.loads(l) for l in open(base / cond / f"{sid}.flux_live.jsonl")]
        panels.append((cond, x, sr, live, json.loads(meta.read_text())["agent_intervals"]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"hspace": 0.35})
    span = 60
    for ax, (cond, x, sr, live, agent) in zip(axes, panels):
        ax.grid(axis="x", visible=False)
        hop = sr // 50
        env = np.sqrt(np.convolve(x[: len(x) // hop * hop].reshape(-1, hop).__pow__(2).mean(1), np.ones(3) / 3, "same"))
        tt = np.arange(len(env)) / 50
        m = tt <= span
        env = env / (env[m].max() + 1e-9)
        for a, b in agent:
            if a < span:
                ax.axvspan(a, min(b, span), color="#efeeea", zorder=0, lw=0)
        ax.fill_between(tt[m], 0, env[m] * 0.9, color="#c9c8c2", lw=0, zorder=1)
        c = [(e["audio_window_end"], e["conf"]) for e in live if e["event"] == "Update" and e["audio_window_end"] <= span]
        if c:
            cx, cy = zip(*c)
            ax.plot(cx, cy, color=BLUE, lw=1.6, zorder=3)
        ax.axhline(0.7, color=ORANGE, lw=1.2, ls=(0, (4, 3)), zorder=2)
        for e in live:
            if e["event"] == "EndOfTurn" and e["audio_window_end"] <= span:
                ax.plot([e["audio_window_end"]] * 2, [0, 1.02], color=ORANGE, lw=1.6, zorder=4)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0, span)
        ax.set_yticks([0, 0.7, 1])
        ax.set_title("Quiet room" if cond == "quiet" else "Podcast playing in the room", fontsize=11.5, pad=6)
    axes[-1].set_xlabel("Seconds into the session")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(fc="#efeeea", label="Agent talking"), Patch(fc="#c9c8c2", label="My microphone level"),
               Line2D([], [], color=BLUE, lw=1.8, label="Flux end-of-turn confidence"),
               Line2D([], [], color=ORANGE, lw=1.8, label="EndOfTurn at default 0.7")]
    axes[0].legend(handles=handles, frameon=False, loc="lower left", fontsize=9.5, ncol=4,
                   bbox_to_anchor=(0, 1.16), handlelength=1.6, columnspacing=1.4)
    fig.suptitle("What the voice lab records: my own first minute of each session", x=0.125, ha="left",
                 fontsize=13, fontweight="semibold", y=1.03)
    save(fig, "06_voice_lab_session.png")


# ---------------------------------------------------------------- 0. cover image (Medium thumbnail, 16:9)
def fig_cover():
    """Thumbnail with no text: a caller's voice message, and the agent's reply crashing into it mid-sentence.

    Both waveforms are real audio: the caller is the meeting turn from figure 1, the agent a stretch of
    another speaker in the same file."""
    from matplotlib.collections import LineCollection
    d = ROOT / "data" / "ami_bench"
    f = "report_FEE051_2"
    audio, ev = cached(d, [f], (1.0,))
    x = audio[f]

    def env(a, b, nb):
        seg = x[int(a * SR): int(b * SR)]
        e = np.linspace(0, len(seg), nb + 1).astype(int)
        amp = np.array([np.sqrt(np.mean(seg[i:j] ** 2)) for i, j in zip(e[:-1], e[1:])])
        return np.maximum(np.clip(amp / np.percentile(amp, 97), 0, 1) ** 0.6, 0.08)

    BG, FG, DIM, INKD = "#0d0e12", "#f2f1ec", "#2a2f36", "#0d0e12"
    AGENT = "#13ef93"  # voice-AI green for the agent
    fig = plt.figure(figsize=(8, 4.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(9, 0)
    ax.axis("off")

    def bars(x0, x1, yc, hmax, amp, cols, lw):
        xs = np.linspace(x0, x1, len(amp))
        segs = [[(u, yc - hmax * v), (u, yc + hmax * v)] for u, v in zip(xs, amp)]
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, capstyle="round", zorder=4))

    def bubble(x0, y0, w, h, fc, tail_x, tail_dir, z):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.75",
                                    fc=fc, ec=BG, lw=7, zorder=z))
        yb = y0 + h - 0.05
        ax.add_patch(plt.Polygon([(tail_x, yb), (tail_x + 0.9 * tail_dir, yb), (tail_x - 0.15 * tail_dir, yb + 0.75)],
                                 closed=True, fc=fc, ec="none", zorder=z))

    def play(cx, cy, r, fc, ic, z):
        ax.add_patch(plt.Circle((cx, cy), r, fc=fc, ec="none", zorder=z))
        ax.add_patch(plt.Polygon([(cx - 0.32 * r, cy - 0.45 * r), (cx - 0.32 * r, cy + 0.45 * r), (cx + 0.5 * r, cy)],
                                 closed=True, fc=ic, ec="none", zorder=z + 0.1))

    # caller: a long voice message; the half the agent talks over is dimmed
    cx0, cy0, cw, ch = 0.9, 1.7, 12.2, 2.6
    bubble(cx0, cy0, cw, ch, FG, cx0 + 0.9, 1, 2)
    play(cx0 + 1.35, cy0 + ch / 2, 0.62, INKD, FG, 3)
    amp = env(17.6, 21.6, 64)
    xa, xb, cut = cx0 + 2.4, cx0 + cw - 0.75, 8.6
    xs = np.linspace(xa, xb, len(amp))
    bars(xa, xb, cy0 + ch / 2, 0.85, amp, [INKD if u < cut else "#b9bcbf" for u in xs], 4.2)

    # agent: replies on top of the caller, mid-sentence
    ax0, ay0, aw, ah = 6.4, 3.8, 8.7, 2.6
    bubble(ax0, ay0, aw, ah, AGENT, ax0 + aw - 0.9, -1, 5)
    play(ax0 + 1.35, ay0 + ah / 2, 0.62, INKD, AGENT, 6)
    amp2 = env(45.2, 47.8, 42)
    ax.add_collection(LineCollection(
        [[(u, ay0 + ah / 2 - 0.85 * v), (u, ay0 + ah / 2 + 0.85 * v)] for u, v in
         zip(np.linspace(ax0 + 2.4, ax0 + aw - 0.75, len(amp2)), amp2)],
        colors=INKD, linewidths=4.2, capstyle="round", zorder=7))

    fig.savefig(OUT / "00_cover.png", dpi=200, facecolor=BG)
    plt.close(fig)
    print("wrote", OUT / "00_cover.png")


# ---------------------------------------------------------------- 8-10. tables as images (Medium has no tables)
def table_fig(name, title, subtitle, cols, rows, highlight=None, widths=None, note=None):
    """rows: list of lists of (text) or (text, sub) tuples; highlight: index of the row to emphasise."""
    n = len(rows)
    widths = widths or [1 / len(cols)] * len(cols)
    rh, hh = 0.62, 0.5
    H = hh + n * rh
    fig = plt.figure(figsize=(9, 1.25 + H * 0.62 + (0.3 if note else 0)))
    ax = fig.add_axes([0.03, 0.05 + (0.08 if note else 0), 0.94, 0.95 - 0.27 - (0.08 if note else 0)])
    ax.set_xlim(0, 1)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.text(0.03, 0.93, title, fontsize=14, fontweight="semibold", va="top")
    fig.text(0.03, 0.93 - 0.11 * 2.6 / fig.get_figheight(), subtitle, fontsize=10.5, color=INK2, va="top")
    xs = np.concatenate([[0], np.cumsum(widths)])
    for j, c in enumerate(cols):
        ax.text(xs[j] + 0.015, hh * 0.55, c.upper(), fontsize=9, color=MUTED, fontweight="semibold", va="center")
    ax.plot([0, 1], [hh, hh], color=INK2, lw=1)
    for i, row in enumerate(rows):
        y0 = hh + i * rh
        if i == highlight:
            ax.add_patch(FancyBboxPatch((0.003, y0 + 0.06), 0.994, rh - 0.12, boxstyle="round,pad=0,rounding_size=0.012",
                                        fc="#eaf2fc", ec="none", zorder=0))
        for j, cell in enumerate(row):
            txt, sub = cell if isinstance(cell, tuple) else (cell, None)
            main = ax.text(xs[j] + 0.015, y0 + rh / 2, txt, fontsize=12, va="center", color=INK,
                           fontweight="semibold" if i == highlight else "normal")
            if sub:  # place the CI right after the measured width of the main value
                bb = main.get_window_extent(renderer=fig.canvas.get_renderer())
                x_end = ax.transData.inverted().transform((bb.x1, bb.y0))[0]
                ax.text(x_end + 0.01, y0 + rh / 2, sub, fontsize=9.5, va="center", color=INK2)
        if i < n - 1:
            ax.plot([0, 1], [y0 + rh, y0 + rh], color=GRID, lw=0.8)
    if note:
        fig.text(0.03, 0.03, note, fontsize=9, color=MUTED, va="bottom")
    save(fig, name)


def pct(v):
    return f"{100 * v:.0f}%"


def ci(v, unit="%"):
    if unit == "%":
        return f"95% CI {100 * v['lo']:.0f}–{100 * v['hi']:.0f}%"
    return f"95% CI {v['lo']:.2f}–{v['hi']:.2f} s"


def fig_tables():
    a = results("ami_bench")
    rh, rec = a["report_half"], a["recommendations_by_target"]
    d, t = rh["default_0.7"], rh["recommended"]
    table_fig("08_table_meetings.png", "Tuning on meeting speech",
              f"Held-out AMI speakers: {a['turns']['report']} turns from {a['n_groups']['report']} people the tool never saw",
              ["Setting", "Turns cut off", "Median wait"],
              [["Default (0.7)", (pct(d["fcr"]["point"]), ci(d["fcr"])), (f"{d['lat_p50']['point']:.2f} s", ci(d["lat_p50"], "s"))],
               [f"Tuned ({a['recommendation']['eot_threshold']})", (pct(t["fcr"]["point"]), ci(t["fcr"])),
                (f"{t['lat_p50']['point']:.2f} s", ci(t["lat_p50"], "s"))]],
              highlight=1, widths=[0.28, 0.38, 0.34])
    by = {p["threshold"]: p for p in a["curve_real"]["report"]}
    rows = []
    for tgt in ("0.20", "0.10", "0.05"):
        th = rec[tgt]["eot_threshold"]
        rows.append([f"At most {float(tgt):.0%} cut off", str(th), pct(by[th]["fcr"]), f"{by[th]['lat_p50']:.2f} s"])
    table_fig("09_table_menu.png", "Pick the trade-off that fits your product",
              "Setting chosen on 28 AMI speakers for each goal, then measured on 43 held-out speakers",
              ["Goal", "Setting", "Turns cut off", "Median wait"], rows, highlight=1, widths=[0.36, 0.2, 0.22, 0.22])
    s = results("spokenwoz")
    sr, srec = s["report_half"], s["recommendations_by_target"]
    r86 = srec["0.10"]
    r94 = srec["0.05"]
    assert r86["eot_threshold"] == 0.86 and r94["eot_threshold"] == 0.94
    table_fig("10_table_phone.png", "Tuning on phone calls",
              f"Held-out SpokenWOZ customer-agent calls: {s['turns']['report']} turns from {s['n_groups']['report']} calls",
              ["Setting", "Turns cut off", "Median wait"],
              [["Default (0.7)", (pct(sr["default_0.7"]["fcr"]["point"]), ci(sr["default_0.7"]["fcr"])),
                (f"{sr['default_0.7']['lat_p50']['point']:.2f} s", ci(sr["default_0.7"]["lat_p50"], "s"))],
               ["Tuned (0.86)", (pct(r86["fcr"]["point"]), ci(r86["fcr"])), (f"{r86['lat_p50']['point']:.2f} s", ci(r86["lat_p50"], "s"))],
               ["Meeting setting (0.94)", (pct(r94["fcr"]["point"]), ci(r94["fcr"])), (f"{r94['lat_p50']['point']:.2f} s", ci(r94["lat_p50"], "s"))]],
              highlight=1, widths=[0.3, 0.35, 0.35])


if __name__ == "__main__":
    which = set(sys.argv[1:])
    for name, fn in [("cover", fig_cover), ("hero", fig_hero), ("pipeline", fig_pipeline), ("tradeoff", fig_tradeoff),
                     ("speakers", fig_speakers), ("background", fig_background), ("session", fig_session),
                     ("tables", fig_tables)]:
        if not which or name in which:
            out = fn()
            if out:
                print(name, out)
