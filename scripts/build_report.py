"""Render a flux-tune results.json as a self-contained, phone-friendly HTML report.

Usage: python scripts/build_report.py reports/fluxtune_demo_agent --title "flux-tune Demo" \
           --dataset "40 simulated calls built from AMI meeting speech (held-out speakers)"
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

DEFAULT = 0.7


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def esc(s):
    return html.escape(str(s))


def tradeoff_svg(curve, highlight):
    """Cut-off rate (y) vs median latency (x), one series, exact real-pass points."""
    W, H, L, R, T, B = 640, 340, 56, 20, 16, 44
    xs = [p["lat_p50"] for p in curve]
    xmax = max(4.0, max(xs) * 1.08)
    ymax = 0.6 if max(p["fcr"] for p in curve) <= 0.55 else 1.0
    X = lambda v: L + (W - L - R) * v / xmax
    Y = lambda v: T + (H - T - B) * (1 - v / ymax)
    g = []
    for k in range(0, int(ymax * 10) + 1):
        v = k / 10
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{int(v * 100)}%</text>')
    for v in range(0, int(xmax) + 1):
        g.append(f'<text x="{X(v):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{v} s</text>')
    pts = sorted(curve, key=lambda p: p["lat_p50"])
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(p['lat_p50']):.1f},{Y(p['fcr']):.1f}" for i, p in enumerate(pts))
    marks, labels = [], []
    for p in pts:
        th = p["threshold"]
        tip = (f"eot_threshold {th}|cut-off rate {pct(p['fcr'])}|median latency {p['lat_p50']:.2f} s|"
               f"p90 latency {p['lat_p90']:.2f} s|turns kept open &gt;1.5 s {pct(p['ter_1500'], 0)}")
        cls = "pt hl" if th in highlight or th == DEFAULT else "pt"
        marks.append(f'<g class="hit" tabindex="0" data-tip="{tip}"><circle cx="{X(p["lat_p50"]):.1f}" cy="{Y(p["fcr"]):.1f}" r="14" class="hitarea"/>'
                     f'<circle cx="{X(p["lat_p50"]):.1f}" cy="{Y(p["fcr"]):.1f}" r="{6.5 if "hl" in cls else 5}" class="{cls}"/></g>')
        if th == DEFAULT or th in highlight:
            name = "Flux default 0.7" if th == DEFAULT else f"{th}"
            anchor = "start"
            dx = 10
            if X(p["lat_p50"]) > W - 150:
                anchor, dx = "end", -10
            labels.append(f'<text x="{X(p["lat_p50"]) + dx:.1f}" y="{Y(p["fcr"]) - 9:.1f}" class="dlabel" text-anchor="{anchor}">{name}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Cut-off rate versus median response latency for each eot_threshold">'
            f'{"".join(g)}<line x1="{L}" x2="{W - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="axis"/>'
            f'<path d="{path}" class="line"/>{"".join(marks)}{"".join(labels)}'
            f'<text x="{(L + W - R) / 2:.0f}" y="{H - 6}" class="axlabel" text-anchor="middle">Median time from the end of the user\'s turn to Flux\'s EndOfTurn</text>'
            f'</svg>')


def noise_svg(rows):
    """Grouped horizontal bars: share of turns kept open > 1.5 s, by threshold and background level."""
    series = [("clean", "Quiet", "s1"), ("tir10", "Talker 10 dB quieter", "s2"), ("tir0", "Talker as loud", "s3")]
    W, L, R, T = 640, 112, 64, 8
    bar, gapb, group_gap = 18, 4, 22
    gh = len(series) * (bar + gapb) - gapb
    H = T + len(rows) * (gh + group_gap) + 26
    X = lambda v: L + (W - L - R) * v
    out = []
    for k in range(0, 11, 2):
        v = k / 10
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{H - 26}" class="grid"/>'
                   f'<text x="{X(v):.1f}" y="{H - 8}" class="tick" text-anchor="middle">{int(v * 100)}%</text>')
    y = T
    for r in rows:
        out.append(f'<text x="{L - 10}" y="{y + gh / 2 + 4:.1f}" class="glabel" text-anchor="end">θ = {r["th"]}</text>')
        for key, name, cls in series:
            v = r[key]
            w = max(2.0, X(v) - L)
            tip = f"eot_threshold {r['th']}|{name}|turns kept open &gt;1.5 s: {pct(v, 0)}"
            out.append(f'<g class="hit" tabindex="0" data-tip="{tip}"><rect x="{L}" y="{y}" width="{w:.1f}" height="{bar}" rx="4" class="bar {cls}"/>'
                       f'<rect x="{L}" y="{y}" width="{min(w, 4):.1f}" height="{bar}" class="bar {cls}"/>'
                       f'<text x="{L + w + 6:.1f}" y="{y + bar / 2 + 4:.1f}" class="vlabel">{pct(v, 0)}</text></g>')
            y += bar + gapb
        y += group_gap - gapb
    out.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{H - 26}" class="axis"/>')
    legend = "".join(f'<span class="key"><i class="sw {c}"></i>{n}</span>' for _, n, c in series)
    return (f'<div class="legend">{legend}</div><svg viewBox="0 0 {W} {H}" class="chart" role="img" '
            f'aria-label="Share of turns kept open more than 1.5 seconds, quiet versus background talker">{"".join(out)}</svg>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_dir")
    ap.add_argument("--title", default="flux-tune Demo")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--group_noun", default="speakers", help="what a bootstrap group is, e.g. 'calls'")
    ap.add_argument("--caveat", help="'What this report can and cannot say' paragraph (HTML); defaults to the AMI demo text")
    ap.add_argument("--label_check", default="", help="label_check.json from check_demo_labels.py (AMI demo only)")
    a = ap.parse_args()
    d = Path(a.report_dir)
    r = json.loads((d / "results.json").read_text())
    lc = json.loads(Path(a.label_check).read_text()) if a.label_check and Path(a.label_check).exists() else None
    curve = r["curve_real"]["report"]
    by_th = {p["threshold"]: p for p in curve}
    recs = r["recommendations_by_target"]
    hl = {v["eot_threshold"] for v in recs.values()}
    dflt = r["report_half"]["default_0.7"]
    ng = r.get("n_groups")
    split_note = (f"Each setting was chosen on {r['turns']['tune']} turns from {ng['tune']} {a.group_noun} and measured on "
                  f"{r['turns']['report']} turns from {ng['report']} different {a.group_noun}. "
                  f"95% confidence intervals resample whole {a.group_noun}, so they show how much results vary between them."
                  if ng else "Each setting was chosen on one half of the calls and measured on the other half, "
                  "with 95% confidence intervals from resampling turns.")

    cards = []
    for t, v in sorted(recs.items(), key=lambda kv: -float(kv[0])):
        p = by_th[v["eot_threshold"]]
        tgt = float(t)
        status = ("ok", "Meets target on held-out turns") if v["fcr"]["point"] <= tgt else \
            ("warn", "Within error of target") if v["fcr"]["lo"] <= tgt else ("warn", "Misses target on held-out turns")
        cards.append(f'''<article class="opt">
  <p class="eyebrow">Cut-offs at most {pct(float(t), 0)}</p>
  <p class="th"><span class="mono">eot_threshold</span> {v["eot_threshold"]}</p>
  <dl>
    <div><dt>Cut-off rate</dt><dd>{pct(v["fcr"]["point"])} <small>CI {pct(v["fcr"]["lo"])}–{pct(v["fcr"]["hi"])}</small></dd></div>
    <div><dt>Median latency</dt><dd>{v["lat_p50"]["point"]:.2f} s <small>CI {v["lat_p50"]["lo"]:.2f}–{v["lat_p50"]["hi"]:.2f}</small></dd></div>
    <div><dt>Kept open &gt;1.5 s</dt><dd>{pct(p["ter_1500"], 0)}</dd></div>
  </dl>
  <p class="chip {status[0]}">{status[1]}</p>
</article>''')
    cards.append(f'''<article class="opt base">
  <p class="eyebrow">For comparison</p>
  <p class="th"><span class="mono">eot_threshold</span> 0.7 <small>(Flux default)</small></p>
  <dl>
    <div><dt>Cut-off rate</dt><dd>{pct(dflt["fcr"]["point"])} <small>CI {pct(dflt["fcr"]["lo"])}–{pct(dflt["fcr"]["hi"])}</small></dd></div>
    <div><dt>Median latency</dt><dd>{dflt["lat_p50"]["point"]:.2f} s <small>CI {dflt["lat_p50"]["lo"]:.2f}–{dflt["lat_p50"]["hi"]:.2f}</small></dd></div>
    <div><dt>Kept open &gt;1.5 s</dt><dd>{pct(by_th[DEFAULT]["ter_1500"], 0)}</dd></div>
  </dl>
</article>''')

    table = "".join(f'<tr><td class="mono">{p["threshold"]}{" (default)" if p["threshold"] == DEFAULT else ""}</td><td>{pct(p["fcr"])}</td>'
                    f'<td>{pct(p["fcr_words_after"]) if "fcr_words_after" in p else "–"}</td>'
                    f'<td>{p["lat_p50"]:.2f} s</td><td>{p["lat_p90"]:.2f} s</td><td>{pct(p["ter_1500"], 0)}</td></tr>' for p in curve)
    wa = by_th.get(DEFAULT, {}).get("fcr_words_after")
    strict_note = (f"<h3>Strict count</h3><p>A turn counts as cut off when Flux fires more than 100 ms before the end of the speaker's last word. "
                   f"Some of those fire during the final word itself, with nothing said afterwards, which a listener may not notice. "
                   f"Counting only cut-offs where the speaker said at least one more word, the default's rate is {pct(wa)} instead of {pct(by_th[DEFAULT]['fcr'])}. "
                   "The table view under the trade-off chart gives both counts for every setting.</p>") if wa is not None else ""

    noise_rows, noise_table, n = [], [], r.get("noise", {})
    for th in sorted({v["threshold"] for v in n.values()}):
        a10, a0 = n.get(f"th{th}_tir10dB"), n.get(f"th{th}_tir0dB")
        if not (a10 and a0):
            continue
        noise_rows.append({"th": th, "clean": by_th[th]["ter_1500"] if th in by_th else float("nan"),
                           "tir10": a10["ter_1500"]["point"], "tir0": a0["ter_1500"]["point"]})
        for name, v in (("Talker 10 dB quieter", a10), ("Talker as loud", a0)):
            dv = v["vs_clean"]["ter_1500"]
            noise_table.append(f'<tr><td class="mono">{th}</td><td>{name}</td><td>{pct(v["ter_1500"]["point"], 0)}</td>'
                               f'<td>+{100 * dv["point"]:.0f} pts <small>({100 * dv["lo"]:.0f} to {100 * dv["hi"]:.0f})</small></td>'
                               f'<td>{pct(v["no_decision_rate"]["point"], 0)}</td><td>{v.get("inserted_words_per_audio_min", float("nan")):.0f}</td></tr>')
    d10 = n.get("th0.7_tir10dB")
    noise_section = (f'''<section>
  <h2>With someone else talking in the room</h2>
  <p>The same calls were re-run with another person's conversation mixed in behind the speaker. {f"At the default setting, the share of turns Flux kept open for more than 1.5 seconds went from {pct(by_th[DEFAULT]['ter_1500'], 0)} to {pct(d10['ter_1500']['point'], 0)}, and {pct(d10['no_decision_rate']['point'], 0)} of turns never ended before the user spoke again." if d10 else ""} No threshold fixes this: raising it removes cut-offs but leaves nearly every turn open.</p>
  <div class="figure">{noise_svg(noise_rows)}</div>
  <details><summary>Table view</summary><div class="tablewrap"><table><thead><tr><th>eot_threshold</th><th>Background</th><th>Kept open &gt;1.5 s</th><th>Change vs quiet (95% CI)</th><th>Turn never ended</th><th>Background words in transcript / min</th></tr></thead><tbody>{"".join(noise_table)}</tbody></table></div></details>
</section>''') if r.get("noise") else ""

    ev = []
    for e in r["evidence"].get(str(DEFAULT), [])[:8]:
        after = e["said_after"]
        before = e["text"][: len(e["text"]) - len(after)].rstrip() if after and e["text"].endswith(after) else e["text"]
        ev.append(f'<li><p class="said"><span>{esc(before)}</span><mark aria-label="Flux ended the turn here">EndOfTurn</mark>'
                  f'<span class="after">{esc(after)}</span></p><p class="where mono">{esc(e["file"])} · {e["decision"]:.2f} s</p></li>')

    real = ""
    rb = r.get("real_background")
    if rb:
        rows_rb = []
        for th, v in sorted(rb["by_threshold"].items()):
            q = by_th.get(float(th))
            rows_rb.append(f'<tr><td class="mono">{th}{" (default)" if float(th) == DEFAULT else ""}</td>'
                           f'<td>{pct(q["ter_1500"], 0) if q else "–"} / {pct(q["fcr"], 0) if q else "–"}</td>'
                           f'<td>{pct(v["ter_1500"]["point"], 0)} <small>({pct(v["ter_1500"]["lo"], 0)}–{pct(v["ter_1500"]["hi"], 0)})</small> / {pct(v["fcr"]["point"], 0)}</td>'
                           f'<td>{pct(v["no_decision_rate"]["point"], 0)}</td></tr>')
        ev_rb = []
        for e in rb.get("evidence", {}).get(str(DEFAULT), [])[:4]:
            after = e["said_after"]
            before = e["text"][: len(e["text"]) - len(after)].rstrip() if after and e["text"].endswith(after) else e["text"]
            ev_rb.append(f'<li><p class="said"><span>{esc(before)}</span><mark>EndOfTurn</mark><span class="after">{esc(after)}</span></p></li>')
        real = f'''<section>
  <h2>Recorded with real background audio</h2>
  <p>{rb["turns"]} turns from {rb["files"]} session(s) recorded with a TV, podcast or conversation playing in the room. These are separate recordings of the same questions, so compare them with the quiet sessions as a whole, not turn by turn.</p>
  <div class="tablewrap"><table><thead><tr><th>eot_threshold</th><th>Quiet: kept open &gt;1.5 s / cut off</th><th>Background: kept open &gt;1.5 s (95% CI) / cut off</th><th>Background: turn never ended</th></tr></thead><tbody>{"".join(rows_rb)}</tbody></table></div>
  {('<ul class="ev">' + "".join(ev_rb) + '</ul>') if ev_rb else ""}
</section>'''

    lab = ""
    if lc:
        rows_lc = "".join(f'<tr><td class="mono">{x["threshold"]}</td><td>{pct(x["fcr_human"])}</td><td>{pct(x["fcr_hindsight"])}</td></tr>' for x in lc["by_threshold"])
        lab = f'''<p>Without a speaker-change signal, flux-tune can fall back to splitting turns at 2-second silences. On this data that fallback ranked settings in the same order as human labels (rank correlation {lc["rank_corr_fcr"]:.2f}) but found {lc["turns_hindsight"]} turns where people heard {lc["turns_human"]}, so it counted about half the cut-offs. That is why this report uses the agent's speaking times, and why the fallback carries a warning.</p>
<div class="tablewrap"><table><thead><tr><th>Threshold</th><th>Cut-offs, human labels</th><th>Cut-offs, silence-split fallback</th></tr></thead><tbody>{rows_lc}</tbody></table></div>'''

    page = f'''<title>{esc(a.title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
/* Layout: one reading column; summary cards first, then evidence and method, like a lab report for one deployment. */
:root {{
  --bg: #f6f7f9; --surface: #ffffff; --fg: #15191f; --fg2: #4a5361; --muted: #737d8c; --line: #e2e6ec; --grid: #eceff3;
  --accent: #2a78d6; --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --ok-bg: #e3f4ea; --ok-fg: #13663a; --warn-bg: #fdf0dc; --warn-fg: #8a5300; --mark-bg: #fde4e0; --mark-fg: #a1271b;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #111418; --surface: #181c21; --fg: #eef1f5; --fg2: #b4bdc9; --muted: #8b95a3; --line: #2a3038; --grid: #232830;
  --accent: #3987e5; --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  --ok-bg: #12301f; --ok-fg: #7fd6a2; --warn-bg: #352610; --warn-fg: #f2c070; --mark-bg: #3a1a17; --mark-fg: #ff9d90; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #111418; --surface: #181c21; --fg: #eef1f5; --fg2: #b4bdc9; --muted: #8b95a3; --line: #2a3038; --grid: #232830;
  --accent: #3987e5; --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  --ok-bg: #12301f; --ok-fg: #7fd6a2; --warn-bg: #352610; --warn-fg: #f2c070; --mark-bg: #3a1a17; --mark-fg: #ff9d90; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }}
.wrap {{ max-width: 760px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 56px; display: grid; gap: 40px; }}
header {{ display: grid; gap: 10px; }}
.brand {{ font: 500 13px/1 var(--mono); color: var(--accent); letter-spacing: .02em; }}
h1 {{ font: 600 clamp(26px, 5vw, 34px)/1.15 var(--display); margin: 0; text-wrap: balance; }}
h2 {{ font: 600 21px/1.25 var(--display); margin: 0 0 6px; text-wrap: balance; }}
p {{ margin: 0; max-width: 65ch; }}
.lede {{ color: var(--fg2); font-size: 16px; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--muted); }}
.meta b {{ color: var(--fg2); font-weight: 500; }}
section {{ display: grid; gap: 14px; min-width: 0; }}
.mono {{ font-family: var(--mono); font-size: .92em; }}
small {{ color: var(--muted); font-size: 12px; }}
.opts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
.opt {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px; display: grid; gap: 10px; align-content: start; min-width: 0; }}
.opt.base {{ background: transparent; border-style: dashed; }}
.eyebrow {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }}
.th {{ font: 600 20px/1.2 var(--display); }}
.th .mono {{ font: 400 12px var(--mono); color: var(--muted); display: block; margin-bottom: 2px; }}
dl {{ margin: 0; display: grid; gap: 6px; font-variant-numeric: tabular-nums; }}
dl div {{ display: grid; gap: 0; }}
dt {{ font-size: 12px; color: var(--muted); }}
dd {{ margin: 0; font-weight: 500; }}
.chip {{ justify-self: start; font-size: 12px; font-weight: 500; padding: 3px 8px; border-radius: 999px; }}
.chip.ok {{ background: var(--ok-bg); color: var(--ok-fg); }}
.chip.warn {{ background: var(--warn-bg); color: var(--warn-fg); }}
.snippet {{ display: flex; gap: 8px; align-items: stretch; flex-wrap: wrap; }}
.snippet code {{ flex: 1 1 280px; min-width: 0; overflow-x: auto; white-space: nowrap; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; font: 13px/1.4 var(--mono); }}
button {{ font: 500 13px var(--body); color: var(--fg); background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 8px 14px; cursor: pointer; }}
button:hover {{ border-color: var(--accent); }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.figure {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 12px 8px; display: grid; gap: 8px; min-width: 0; }}
.chart {{ width: 100%; height: auto; display: block; overflow: visible; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.axis {{ stroke: var(--line); stroke-width: 1; }}
.tick {{ fill: var(--muted); font: 11px var(--mono); }}
.axlabel {{ fill: var(--muted); font: 12px var(--body); }}
.glabel {{ fill: var(--fg2); font: 500 12px var(--mono); }}
.dlabel {{ fill: var(--fg); font: 500 12px var(--body); paint-order: stroke; stroke: var(--surface); stroke-width: 4px; }}
.vlabel {{ fill: var(--fg2); font: 500 12px var(--mono); }}
.line {{ fill: none; stroke: var(--s1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.pt {{ fill: var(--s1); stroke: var(--surface); stroke-width: 2; }}
.hitarea {{ fill: transparent; }}
.bar.s1 {{ fill: var(--s1); }} .bar.s2 {{ fill: var(--s2); }} .bar.s3 {{ fill: var(--s3); }}
.hit {{ cursor: default; outline: none; }}
.hit:focus-visible .pt, .hit:hover .pt {{ stroke: var(--fg); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--fg2); }}
.key {{ display: inline-flex; align-items: center; gap: 6px; }}
.sw {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
.sw.s1 {{ background: var(--s1); }} .sw.s2 {{ background: var(--s2); }} .sw.s3 {{ background: var(--s3); }}
.tablewrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 7px 10px 7px 0; border-bottom: 1px solid var(--line); white-space: nowrap; }}
th {{ font-weight: 500; color: var(--muted); font-size: 12px; }}
details summary {{ cursor: pointer; color: var(--fg2); font-size: 13px; }}
details[open] summary {{ margin-bottom: 8px; }}
.ev {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }}
.ev li {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; display: grid; gap: 4px; min-width: 0; }}
.said {{ overflow-wrap: anywhere; }}
.said mark {{ background: var(--mark-bg); color: var(--mark-fg); font: 500 11px var(--mono); padding: 2px 6px; border-radius: 4px; margin: 0 6px; vertical-align: 1px; }}
.said .after {{ color: var(--muted); font-style: italic; }}
.where {{ color: var(--muted); font-size: 12px; }}
.method {{ display: grid; gap: 12px; color: var(--fg2); }}
.method h3 {{ font: 600 15px/1.3 var(--body); color: var(--fg); margin: 6px 0 0; }}
#tip {{ position: fixed; pointer-events: none; z-index: 10; background: var(--fg); color: var(--bg); font: 12px/1.45 var(--mono); padding: 8px 10px; border-radius: 6px; max-width: 260px; }}
footer {{ font-size: 13px; color: var(--muted); display: grid; gap: 6px; }}
footer code {{ font: 12px var(--mono); overflow-wrap: anywhere; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
<div class="wrap">
<header>
  <p class="brand">flux-tune</p>
  <h1>Which Flux turn-taking setting fits this audio?</h1>
  <p class="lede">Every threshold trades interrupting people for waiting too long. This report measures that trade on real recordings with Deepgram Flux{", then checks how the chosen setting holds up when someone else is talking in the room" if r.get("noise") else ""}.</p>
  <div class="meta"><span><b>Audio</b> {esc(a.dataset or f"{r['files']} files")}, {r["audio_minutes"]:.0f} min</span>
  <span><b>Turns scored</b> {r["turns"]["report"]} held-out{f" from {ng['report']} {a.group_noun}" if ng else ""} (chosen on {r["turns"]["tune"]} others)</span>
  <span><b>Turn boundaries</b> {"agent speaking times" if r["label_mode"] == "agent_times" else r["label_mode"].replace("_", " ")}</span>
  <span><b>API cost</b> ${r["spend_usd_total"]:.2f} cumulative</span></div>
</header>

<section>
  <h2>Recommended settings</h2>
  <p>Pick the row that matches how much interrupting you can tolerate. {split_note}</p>
  <div class="opts">{"".join(cards)}</div>
  <div class="snippet"><code id="cfg">wss://api.deepgram.com/v2/listen?model=flux-general-en&amp;eot_threshold={r["recommendation"]["eot_threshold"]}</code><button id="copy" type="button">Copy URL</button></div>
</section>

<section>
  <h2>The trade-off on this audio</h2>
  <p>Each point is one real Flux pass over the same calls. Moving right waits longer before replying; moving down interrupts fewer turns. A "cut-off" means Flux ended the turn while the speaker still had words to say.</p>
  <div class="figure">{tradeoff_svg(curve, hl)}</div>
  <details><summary>Table view</summary><div class="tablewrap"><table><thead><tr><th>eot_threshold</th><th>Cut-off rate</th><th>…with words after</th><th>Median latency</th><th>p90 latency</th><th>Kept open &gt;1.5 s</th></tr></thead><tbody>{table}</tbody></table></div></details>
</section>

{noise_section}

{real}

<section>
  <h2>Where the default cut people off</h2>
  <p>Held-out turns where Flux at <span class="mono">eot_threshold 0.7</span> fired EndOfTurn early. Grey text is what the speaker said after Flux had already decided they were done.</p>
  <ul class="ev">{"".join(ev)}</ul>
</section>

<section class="method">
  <h2>How this was measured</h2>
  <h3>Real passes, not simulation</h3>
  <p>Flux reports its end-of-turn confidence every 240 ms but decides on 80 ms frames, so replaying the reported confidence against a new threshold misses some decisions and drifts badly on multi-turn audio. Every number above comes from a real Flux pass at that threshold.</p>
  <h3>Turn boundaries</h3>
  <p>A turn ends at the speaker's last word before the agent starts talking, using Flux's own word timestamps. That matches how people label turns.</p>
  {strict_note}
  {lab}
  <h3>What this report can and cannot say</h3>
  <p>{a.caveat or "The speech comes from the AMI meeting corpus (CC BY 4.0): spontaneous meeting talk with long pauses, which is harder than most people's replies to a voice agent. The background talker is a second AMI speaker mixed in digitally, not a real room. Treat the numbers as a demonstration of the method; run flux-tune on your own call audio before changing production settings."}</p>
</section>

<footer>
  <p>Reproduce: <code>python scripts/flux_tune.py AUDIO_DIR --agent_times agent_times.json --target 0.10 --noise --out OUT</code></p>
  <p>Built with Deepgram Flux (<span class="mono">flux-general-en</span>, /v2/listen). Not affiliated with Deepgram.</p>
</footer>
</div>
<div id="tip" hidden></div>
<script>
(() => {{
  const tip = document.getElementById("tip");
  const show = (el, x, y) => {{
    tip.innerHTML = el.dataset.tip.split("|").map((s, i) => i === 0 ? "<b>" + s + "</b>" : s).join("<br>");
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(window.innerWidth - w - 8, Math.max(8, x + 14)) + "px";
    tip.style.top = Math.max(8, y - h - 12) + "px";
  }};
  document.querySelectorAll(".hit").forEach(el => {{
    el.addEventListener("pointermove", e => show(el, e.clientX, e.clientY));
    el.addEventListener("pointerleave", () => tip.hidden = true);
    el.addEventListener("focus", () => {{ const b = el.getBoundingClientRect(); show(el, b.left + b.width / 2, b.top); }});
    el.addEventListener("blur", () => tip.hidden = true);
  }});
  const btn = document.getElementById("copy"), cfg = document.getElementById("cfg");
  btn.addEventListener("click", () => {{
    const text = cfg.textContent;
    const done = () => {{ btn.textContent = "Copied"; setTimeout(() => btn.textContent = "Copy URL", 1600); }};
    const fallback = () => {{ const r = document.createRange(); r.selectNodeContents(cfg); const s = getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = "Selected, press Copy"; }};
    try {{ navigator.clipboard.writeText(text).then(done, fallback); }} catch (e) {{ fallback(); }}
  }});
}})();
</script>
'''
    (d / "report.html").write_text(page)
    print("wrote", d / "report.html", f"{len(page) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
