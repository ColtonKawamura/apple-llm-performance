#!/usr/bin/env python3
"""Render the open-weights LLM leaderboard page."""
import os
import html as html_mod
import json

from registry import (MODELS, MODEL_BY_ID, CATEGORIES, rank, DATA_FILE)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SITE = os.path.join(ROOT, "docs")


def fmt_context(tokens):
    if tokens is None:
        return "--"
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.0f}M" if tokens % 1_000_000 == 0 else f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens // 1_000}k"
    return str(tokens)


def fmt_cost(val):
    if val is None:
        return "--"
    return f"${val:.2f}" if val >= 0.01 else f"${val:.4f}"


def fmt_time(val):
    if val is None:
        return "--"
    return f"{val:.1f} min"


def fmt_latency(val):
    if val is None:
        return "--"
    return f"{val:.1f}s"


def fmt_intel(val):
    if val is None:
        return "--"
    return str(val)


def esc(s):
    return html_mod.escape(str(s))


HEADER_TMPL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open-Weights LLM Leaderboard</title>
<meta name="description" content="Ranking of open-weights LLMs by intelligence, cost, speed, latency. Data from Artificial Analysis.">
<link rel="stylesheet" href="style.css">
</head>
<body>
<header>
  <h1>Open-Weights LLM Leaderboard</h1>
  <p class="subtitle">Open-weights models ranked by Artificial Analysis benchmarks</p>
  <p class="meta">{total} models &middot; {intel} with intelligence scores &middot; Last updated {updated}</p>
</header>
<nav class="tabs">
  {tabs}
</nav>
<main>
"""

CAT_HEADER = """<section id="{cat_id}" class="category">
  <h2>{label}</h2>
  <p class="axis">{axis}</p>
  <table class="leaderboard">
    <thead>
      <tr><th class="rank">#</th><th class="model">Model</th><th class="creator">Creator</th><th class="metric">{metric_label}</th><th class="details">Details</th></tr>
    </thead>
    <tbody>
"""

ROW_TMPL = """      <tr>
        <td class="rank">{rank}</td>
        <td class="model"><a href="{url}" target="_blank">{name}</a>{bar}</td>
        <td class="creator">{creator}</td>
        <td class="metric">{metric_val}</td>
        <td class="details">
{details}        </td>
      </tr>
"""

FOOTER = """</main>
<footer>
  <p>Data from <a href="https://artificialanalysis.ai/leaderboards/models?weights=open" target="_blank">Artificial Analysis Open-Weights Leaderboard</a>. Updated {updated}.</p>
</footer>
<script>
document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', e => {{
    e.preventDefault();
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const target = document.querySelector(tab.getAttribute('href'));
    if (target) target.scrollIntoView({{behavior: 'smooth'}});
  }});
}});
</script>
</body></html>"""

CSS = """
:root {{ --bg: #0d1117; --fg: #c9d1d9; --muted: #8b949e; --border: #30363d;
  --accent: #58a6ff; --card: #161b22; --header: #f0f6fc; --bar: #238636; }}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); color: var(--fg); line-height: 1.5;
  padding: 2rem 1rem; max-width: 1200px; margin: 0 auto; }}
header {{ text-align: center; padding: 2rem 0; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }}
header h1 {{ color: var(--header); font-size: 2rem; margin-bottom: 0.5rem; }}
header .subtitle {{ color: var(--muted); font-size: 1.1rem; }}
header .meta {{ color: var(--muted); font-size: 0.85rem; margin-top: 0.5rem; }}
nav.tabs {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 2rem; justify-content: center; }}
.tab {{ padding: 0.5rem 1rem; background: var(--card); border: 1px solid var(--border);
  border-radius: 6px; color: var(--fg); text-decoration: none; font-size: 0.9rem; }}
.tab:hover, .tab.active {{ background: var(--accent); color: #000; border-color: var(--accent); }}
.category {{ margin-bottom: 3rem; }}
.category h2 {{ color: var(--header); font-size: 1.5rem; margin-bottom: 0.5rem; }}
.category .axis {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1rem; }}
table.leaderboard {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; }}
table.leaderboard thead th {{ text-align: left; padding: 0.75rem; border-bottom: 2px solid var(--border);
  color: var(--muted); font-weight: 500; font-size: 0.85rem; text-transform: uppercase;
  letter-spacing: 0.05em; }}
table.leaderboard tbody td {{ padding: 0.75rem; border-bottom: 1px solid var(--border); vertical-align: middle; }}
table.leaderboard tbody tr:hover {{ background: rgba(255,255,255,0.03); }}
.rank {{ color: var(--muted); font-size: 0.85rem; width: 40px; }}
.model {{ font-weight: 500; }}
.model a {{ color: var(--accent); text-decoration: none; }}
.model a:hover {{ text-decoration: underline; }}
.bar {{ display: inline-block; height: 4px; background: var(--bar); border-radius: 2px; margin-top: 4px; min-width: 2px; }}
.creator {{ color: var(--muted); font-size: 0.9rem; }}
.metric {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.details {{ display: flex; flex-wrap: wrap; gap: 0.5rem; font-size: 0.85rem; color: var(--muted); }}
.detail {{ background: var(--card); padding: 2px 8px; border-radius: 4px; border: 1px solid var(--border); }}
footer {{ text-align: center; padding: 2rem 0; border-top: 1px solid var(--border);
  margin-top: 2rem; color: var(--muted); font-size: 0.85rem; }}
footer a {{ color: var(--accent); text-decoration: none; }}
footer a:hover {{ text-decoration: underline; }}
@media (max-width: 768px) {{ body {{ padding: 1rem 0.5rem; }} header h1 {{ font-size: 1.5rem; }}
  table.leaderboard {{ font-size: 0.85rem; }} table.leaderboard thead th, table.leaderboard tbody td {{ padding: 0.5rem; }}
  .details {{ flex-direction: column; gap: 0.25rem; }} }}
""".lstrip()


def render():
    cat_data = []
    for cat in CATEGORIES:
        ranked = rank(cat["id"])
        entries = []
        for mid, val in ranked:
            m = MODEL_BY_ID.get(mid)
            if m:
                entries.append({"model": m, "value": val, "rank": len(entries) + 1})
        cat_data.append({
            "id": cat["id"], "label": cat["label"],
            "axis": cat["axis"], "ascending": cat["ascending"],
            "entries": entries,
        })

    FMT = {
        "intelligence": fmt_intel, "cost": fmt_cost,
        "speed": fmt_time, "latency": fmt_latency, "context": fmt_context,
    }

    total = len(MODELS)
    with_intel = sum(1 for m in MODELS if m["intelligenceIndex"] is not None)
    with open(DATA_FILE, encoding="utf-8") as f:
        meta = json.load(f).get("_meta", {})
    last_fetched = meta.get("lastFetched", "unknown")

    tabs = []
    for i, c in enumerate(cat_data):
        active = "active" if i == 0 else ""
        tabs.append(f'<a href="#{c["id"]}" class="tab {active}">{esc(c["label"])}</a>')

    parts = []
    parts.append(HEADER_TMPL.format(
        total=total, intel=with_intel, updated=esc(last_fetched),
        tabs=" ".join(tabs),
    ))

    for cat in cat_data:
        fmt = FMT.get(cat["id"], str)
        parts.append(CAT_HEADER.format(
            cat_id=esc(cat["id"]), label=esc(cat["label"]),
            axis=esc(cat["axis"]), metric_label=esc(cat["label"]),
        ))
        for entry in cat["entries"]:
            m = entry["model"]
            val = fmt(entry["value"])
            bar = ""
            if cat["id"] == "intelligence" and entry["value"] and cat["entries"][0]["value"]:
                pct = int((entry["value"] / cat["entries"][0]["value"]) * 100)
                bar = f'<span class="bar" style="width:{pct}%"></span>'
            details = ""
            if cat["id"] == "intelligence":
                details = (
                    f'<span class="detail">Cost: {fmt_cost(m.get("costPerTask"))}</span> '
                    f'<span class="detail">Time: {fmt_time(m.get("timePerTask"))}</span> '
                    f'<span class="detail">Latency: {fmt_latency(m.get("latency"))}</span> '
                    f'<span class="detail">Context: {fmt_context(m.get("contextWindow"))}</span> '
                )
            elif cat["id"] == "cost":
                details = (
                    f'<span class="detail">Intel: {fmt_intel(m.get("intelligenceIndex"))}</span> '
                    f'<span class="detail">Context: {fmt_context(m.get("contextWindow"))}</span> '
                )
            elif cat["id"] == "context":
                details = f'<span class="detail">Intel: {fmt_intel(m.get("intelligenceIndex"))}</span> '
            parts.append(ROW_TMPL.format(
                rank=entry["rank"], url=esc(m["aaUrl"]), name=esc(m["name"]),
                bar=bar, creator=esc(m["creator"]), metric_val=val, details=details,
            ))
        parts.append("    </tbody>\n  </table>\n</section>\n")

    parts.append(FOOTER.format(updated=esc(last_fetched)))
    return "\n".join(parts)


def main():
    os.makedirs(SITE, exist_ok=True)
    html_content = render()
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Built docs/index.html ({len(html_content)} bytes)")
    css_content = CSS
    with open(os.path.join(SITE, "style.css"), "w", encoding="utf-8") as f:
        f.write(css_content)
    print(f"Built docs/style.css ({len(css_content)} bytes)")


if __name__ == "__main__":
    main()
