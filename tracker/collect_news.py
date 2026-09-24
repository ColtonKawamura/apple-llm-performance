#!/usr/bin/env python3
"""Collect the daily news rollup into data/data.json.

    python3 tracker/collect_news.py            # gather feeds, show the rollup, write it
    python3 tracker/collect_news.py --dry-run  # show the rollup, touch nothing

The page's News section is a daily snapshot: the top three Apple-silicon
model stories, plus the new language models that shipped in the last day,
each with a one-line summary and the link to the story it came from. This
script gathers the signals from the discovery sources listed in AGENTS.md -
Hugging Face trending and newest, mlx-community uploads, the Ollama
catalogue, r/LocalLLaMA's new feed - and writes one rollup entry into the
`news` section of data/data.json, newest first.

The summary and the top-three pick are judgement, so they are written by an
agent or a person, not by this script: it fetches and ranks the raw signal,
prints the candidate list, and takes the final rollup from a JSON file when
one is supplied (-). Without - it writes the mechanical rollup - the newest
items, ranked by source order - which is a fine default on a quiet day.

Nothing here is a build step. The build reads data/data.json; this only
updates it, the same way tracker/add_model.py does, and validate.py decides
whether the result is shippable.

Everything is standard library. One fetch per source, a single retry on 429
for Reddit, and a hard timeout on each so a dead feed cannot hang the run.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "data.json")

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
TIMEOUT = 25
MAX_ROLLUPS = 7          # how many daily rollups data.json keeps
NEW_MODEL_HOURS = 36     # "new" means created inside this window


def die(msg):
    print(f"collect_news: {msg}", file=sys.stderr)
    raise SystemExit(1)


def fetch(url, headers=None, retries=1):
    """One GET. Returns bytes, or None on any failure - a dead feed degrades
    the rollup, it does not kill the run."""
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                # Reddit rate-limits hard; back off before the one retry.
                time.sleep(5)
                continue
            print(f"  ! {url} -> HTTP {e.code}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"  ! {url} -> {e.__class__.__name__}: {e}", file=sys.stderr)
            return None
    return None


# ------------------------------------------------------------------ sources
def hf_models(params):
    """Hugging Face API. params is the query string after ?"""
    raw = fetch(f"https://huggingface.co/api/models?{params}")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("  ! huggingface: not JSON", file=sys.stderr)
        return []


def hf_trending_text(limit=12):
    """What the community is pulling right now, text models only.

    trendingScore is the highest-signal feed AGENTS.md names: a brand-new
    flagship trends within hours. The pipeline filter drops the image and
    audio models so the rollup is about models this page ranks."""
    rows = hf_models(f"sort=trendingScore&direction=-1&limit=60"
                     f"&pipeline_tag=text-generation")
    out = []
    for m in rows:
        name = m.get("modelId") or m.get("id") or ""
        if not name:
            continue
        out.append({
            "title": name,
            "url": f"https://huggingface.co/{name}",
            "downloads": m.get("downloads") or 0,
            "likes": m.get("likes") or 0,
            "created": m.get("createdAt") or "",
        })
    return out[:limit]


def hf_newest_text(limit=12):
    """Text models published newest-first - the release feed proper."""
    rows = hf_models(f"sort=createdAt&direction=-1&limit=80"
                     f"&pipeline_tag=text-generation")
    out = []
    for m in rows:
        name = m.get("modelId") or m.get("id") or ""
        if not name:
            continue
        out.append({
            "title": name,
            "url": f"https://huggingface.co/{name}",
            "downloads": m.get("downloads") or 0,
            "likes": m.get("likes") or 0,
            "created": m.get("createdAt") or "",
        })
    return out[:limit]


def hf_mlx_newest(limit=12):
    """mlx-community uploads, newest first - Apple-silicon-specific.

    If a model appears here, someone has already converted it for the chips
    on this page, which is the closest thing to a confirmed-new-model signal
    without polling every engine."""
    rows = hf_models(f"author=mlx-community&sort=createdAt&direction=-1"
                     f"&limit=80&pipeline_tag=text-generation")
    out = []
    for m in rows:
        name = m.get("modelId") or m.get("id") or ""
        if not name:
            continue
        out.append({
            "title": name,
            "url": f"https://huggingface.co/{name}",
            "downloads": m.get("downloads") or 0,
            "likes": m.get("likes") or 0,
            "created": m.get("createdAt") or "",
        })
    return out[:limit]


def reddit_new(limit=8):
    """r/LocalLLaMA new feed via the public RSS endpoint.

    Works without a key but rate-limits hard (HTTP 429 on a second request),
    so exactly one retry with a backoff - never a tight poll. The feed gives
    titles and links only, which is what a headline rollup needs.

    The endpoint serves Atom, not RSS 2.0: entries are <entry> and the link is
    a <link href="..."> attribute, not a text node. Both shapes are handled so a
    format change on Reddit does not silently zero out the community signal.
    Reddit namespaces the Atom elements, so match on the local name, not the
    fully-qualified tag."""
    raw = fetch("https://www.reddit.com/r/LocalLLaMA/new/.rss",
                headers={"Accept": "application/atom+xml"}, retries=1)
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        print(f"  ! reddit: not XML ({e})", file=sys.stderr)
        return []

    def local(tag):
        return tag.rsplit("}", 1)[-1]

    out = []
    # Atom: <entry><title>..<link href=".."/>
    for entry in (e for e in root.iter() if local(e.tag) == "entry"):
        title = ""
        link = ""
        for c in entry:
            if local(c.tag) == "title" and not title:
                title = (c.text or "").strip()
            elif local(c.tag) == "link" and c.get("rel") in (None, "alternate"):
                link = c.get("href", "")
        if not title or not link.startswith("http"):
            continue
        out.append({"title": title, "url": link})
        if len(out) >= limit:
            break
    if not out:
        # RSS 2.0 fallback: <item><title>..<link>http://..</link>
        for item in (e for e in root.iter() if local(e.tag) == "item"):
            texts = {local(c.tag): (c.text or "").strip() for c in item}
            title, link = texts.get("title", ""), texts.get("link", "")
            if not title or not link.startswith("http"):
                continue
            out.append({"title": title, "url": link})
            if len(out) >= limit:
                break
    return out


def ollama_new(limit=8):
    """The Ollama catalogue, newest first. A curated list with very low
    noise - a name landing here is usually a usable model, not a lab
    prototype."""
    raw = fetch("https://ollama.com/search?o=newest",
                headers={"Accept": "text/html"})
    if not raw:
        return []
    html = raw.decode("utf-8", "replace")
    out = []
    for m in re.finditer(r'href="/library/([a-z0-9._-]+)[?/"]', html):
        name = m.group(1)
        if name in ("all", "help", "search"):
            continue
        out.append({"title": name, "url": f"https://ollama.com/library/{name}"})
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------- picks
def is_new(created, now, hours):
    if not created:
        return False
    try:
        c = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError:
        return False
    return c >= now - timedelta(hours=hours)


def pick_top(cands, take=3):
    """The three stories. Candidates arrive with their source rank in the
    list order; a model with real traffic beats a model with none, and the
    Reddit feed - the community's own front page - gets first look. No
    invented scores: order is the ranking."""
    ranked = []
    seen = set()
    # Reddit first: it is where the community says what matters, and a
    # trending model with no discussion in it is usually a quant refresh.
    for c in cands.get("reddit", []):
        key = c["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append((2, c))
    for c in cands.get("hf_trending", []):
        key = c["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        # A trending model with four-figure downloads is the flagship
        # signal; below that it is ordered behind any Reddit story.
        ranked.append((0 if (c.get("downloads") or 0) >= 1000 else 1, c))
    ranked.sort(key=lambda t: t[0])
    return ranked[:take]


def pick_new_models(cands, now, hours):
    """Models published inside the window, from the two newest feeds.
    mlx-community is checked second because an mlx-community upload is the
    same model the HF feed lists - dedupe by name keeps one row per model,
    and the mlx-community row is the one that proves a Mac can load it."""
    seen = set()
    out = []
    for c in cands.get("hf_newest", []) + cands.get("hf_mlx", []):
        name = c["title"].lower()
        if name in seen:
            continue
        if not is_new(c.get("created"), now, hours):
            continue
        seen.add(name)
        out.append(c)
    return out


# -------------------------------------------------------------------- write
def load_data():
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)


def save_data(d):
    tmp = DATA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, DATA)


def write_rollup(d, rollup):
    news = d.setdefault("news", [])
    if news and news[0].get("date") == rollup["date"]:
        # Same day: replace the rollup, do not stack two entries for one day.
        news[0] = rollup
    else:
        news.insert(0, rollup)
        del news[MAX_ROLLUPS:]


def print_rollup(r):
    print(f"\nrollup {r['date']}")
    print("  top stories:")
    for t, s, u in r["top"]:
        print(f"    - {t}\n      {s}\n      {u}")
    print("  new language models:")
    if r["newModels"]:
        for t, s, u in r["newModels"]:
            print(f"    - {t}\n      {s}\n      {u}")
    else:
        print("    (none in the window)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="gather and show the rollup, do not touch data/data.json")
    ap.add_argument("--in", dest="spec", default=None,
                    help="path to a finished rollup JSON {date, top, newModels}; "
                         "supplied instead of the mechanical pick (summaries are "
                         "written by the agent, not invented here)")
    ap.add_argument("--top", type=int, default=3, help="stories to take (default 3)")
    a = ap.parse_args()

    if a.spec:
        with open(a.spec, encoding="utf-8") as f:
            rollup = json.load(f)
        for k in ("date", "top"):
            if k not in rollup:
                die(f"the supplied rollup is missing {k!r}")
        rollup.setdefault("newModels", [])
        print_rollup(rollup)
        if a.dry_run:
            print("\ndry run: nothing written.")
            return 0
        d = load_data()
        write_rollup(d, rollup)
        save_data(d)
        print(f"wrote {DATA} (news now {len(d['news'])} rollup(s))")
        return 0

    now = datetime.now(timezone.utc)
    cands = {}
    print("gathering signals (one fetch per source, no retries beyond one on 429):")
    print("  huggingface trending (text-generation)...", flush=True)
    cands["hf_trending"] = hf_trending_text()
    print("  huggingface newest (text-generation)...", flush=True)
    cands["hf_newest"] = hf_newest_text()
    print("  mlx-community newest (text-generation)...", flush=True)
    cands["hf_mlx"] = hf_mlx_newest()
    print("  r/LocalLLaMA new (rss)...", flush=True)
    cands["reddit"] = reddit_new()
    print("  ollama newest...", flush=True)
    cands["ollama"] = ollama_new()

    top = pick_top(cands, take=a.top)
    new_models = pick_new_models(cands, now, NEW_MODEL_HOURS)

    # The mechanical default: headline, a one-line fact that needs no reading
    # of the story, and the link. An agent or person replaces the summary
    # with a real one via --in; a bare title is not a summary, so the
    # fallback says what the row actually is.
    top_rows = []
    for _, c in top:
        extra = f" {c['downloads']:,} downloads" if c.get("downloads") else ""
        top_rows.append([c["title"],
                         f"Trending on the sources polled today.{extra}",
                         c["url"]])
    new_rows = [[c["title"],
                 "New text model published inside the collection window; "
                 "see the link for weights, licence and what it can do.",
                 c["url"]] for c in new_models]

    rollup = {"date": now.strftime("%Y-%m-%d"),
              "top": top_rows,
              "newModels": new_rows}
    print_rollup(rollup)

    if a.dry_run:
        print("\ndry run: nothing written. To ship a curated version, edit the "
              "summaries and run: python3 tracker/collect_news.py --in rollup.json")
        return 0

    d = load_data()
    write_rollup(d, rollup)
    save_data(d)
    print(f"\nwrote {DATA} (news now {len(d['news'])} rollup(s))")
    print("run tracker/validate.py, then tracker/build.py - exactly what CI does")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
