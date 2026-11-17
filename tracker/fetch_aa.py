#!/usr/bin/env python3
"""Fetch open-weights models from the Artificial Analysis leaderboard.

    python3 tracker/fetch_aa.py            # fetch and update data.json
    python3 tracker/fetch_aa.py --dry-run  # print what would change, write nothing

Pulls the open-weights leaderboard table from artificialanalysis.ai and replaces
data.json in-place. This is the single source of truth: every model on the page
comes from this one API call, and every metric comes from AA. Models are only
included if they appear on the leaderboard with weights=open.

Nothing here is a build step. The build reads data/data.json; this only
updates it, and validate.py decides whether the result is shippable.

Everything is standard library. Hard timeout on every request.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "data.json")

LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models?weights=open"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
TIMEOUT = 30

# Creators whose models are proprietary (closed weights) and should be excluded
CLOSED_CREATORS = {"Anthropic", "OpenAI", "SpaceXAI"}

# Model patterns that are proprietary even from otherwise open creators
CLOSED_PATTERNS = [
    r"^Muse\s+(?:Wonder|Ember|Glimmer)",  # Meta proprietary line
    r"^Grok",  # SpaceXAI
    r"^Gemini\s+4",  # Gemini 4 series
    r"^Gemini\s+3\.\d+\s+(?:Argon|Carbon|Hydrogen)",
]


def die(msg):
    print(f"fetch_aa: {msg}", file=sys.stderr)
    raise SystemExit(1)


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        die(f"GET {url} -> HTTP {e.code}")
    except Exception as e:
        die(f"GET {url} -> {e.__class__.__name__}: {e}")


def slug(name):
    """Generate a URL-safe ID from a model name."""
    s = re.sub(r'[^a-zA-Z0-9\s-]', '', name)
    s = re.sub(r'\s+', '-', s).strip('-').lower()
    return re.sub(r'-+', '-', s)


def parse_float(val):
    if val is None or val == "--":
        return None
    m = re.search(r'[\d.]+', str(val))
    return float(m.group()) if m else None


def parse_context(val):
    """Parse context window string (e.g. '1M', '256k') to token count."""
    if not val or val == "--":
        return None
    m = re.search(r'([\d.]+)([kKmM]?)', val)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit == 'k':
        return int(num * 1000)
    elif unit == 'm':
        return int(num * 1_000_000)
    return int(num)


def is_open_weights(creator, name):
    """Check if a model is open-weights based on creator and name."""
    for closed in CLOSED_CREATORS:
        if closed in creator:
            return False
    for pat in CLOSED_PATTERNS:
        if re.search(pat, name, re.I):
            return False
    return True


def parse_table(html):
    """Parse the leaderboard table from the HTML page."""
    # Find the table in the HTML
    table_match = re.search(r'<table[^>]*>(.*?)</table>', html, re.S)
    if not table_match:
        die("no <table> found in leaderboard HTML; page layout changed")
        return []  # unreachable, silences type checker
    table_html = table_match.group(1)

    # Extract all table rows
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.S)
    models = []
    seen = set()

    for row in rows:
        # Skip header rows
        if '<th' in row or '<caption' in row:
            continue

        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
        if len(cells) < 7:
            continue

        # Cell 1: Model name (with logo)
        model_cell = cells[1]

        # Extract creator from logo alt text or image URL
        creator_match = re.search(r'alt="([^"]+)"', model_cell)
        if not creator_match:
            creator_match = re.search(r'logos/([^"/\s]+)', model_cell)
        creator = creator_match.group(1) if creator_match else "Unknown"

        # Clean model name
        name = re.sub(r'<[^>]+>', '', model_cell).strip()
        name = re.sub(r'\s*\(.*?\)\s*', '', name).strip()
        name = re.sub(r'\s+', ' ', name).strip()

        if not name or name in seen:
            continue
        seen.add(name)

        if not is_open_weights(creator, name):
            continue

        # Parse metrics
        intel_raw = re.sub(r'<[^>]+>', '', cells[2]).strip()
        cost_raw = re.sub(r'<[^>]+>', '', cells[3]).strip()
        time_raw = re.sub(r'<[^>]+>', '', cells[4]).strip()
        latency_raw = re.sub(r'<[^>]+>', '', cells[5]).strip()
        context_raw = re.sub(r'<[^>]+>', '', cells[6]).strip()

        intel = int(intel_raw) if intel_raw.isdigit() else None

        models.append({
            "id": slug(name),
            "name": name,
            "creator": creator,
            "intelligenceIndex": intel,
            "costPerTask": parse_float(cost_raw),
            "timePerTask": parse_float(time_raw),
            "latency": parse_float(latency_raw),
            "contextWindow": parse_context(context_raw),
            "aaUrl": f"https://artificialanalysis.ai/models/{slug(name)}",
        })

    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would change, write nothing")
    args = ap.parse_args()

    # Fetch the leaderboard
    print("Fetching leaderboard...")
    html = get(LEADERBOARD_URL)

    # Parse the table
    print("Parsing table...")
    models = parse_table(html)
    if not models:
        die("no open-weights models parsed; page layout may have changed")

    # Sort by intelligence index (descending), then name
    models.sort(key=lambda m: (-(m["intelligenceIndex"] or 0), m["name"]))

    # Count models with data
    with_intel = sum(1 for m in models if m["intelligenceIndex"] is not None)
    with_cost = sum(1 for m in models if m["costPerTask"] is not None)

    print(f"\nFound {len(models)} open-weights models")
    print(f"  {with_intel} with Intelligence Index")
    print(f"  {with_cost} with Cost data")
    print(f"\nTop 10:")
    for m in models[:10]:
        print(f"  {m['name']:35s} | I:{str(m['intelligenceIndex'] or '--'):>4}")

    if args.dry_run:
        print("\nDry run, wrote nothing")
        return

    # Load existing data
    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)

    # Preserve categories (they define the UI structure)
    categories = data.get("categories", [])

    # Update meta
    data["_meta"] = {
        "schema_version": 2,
        "note": "Open-weights LLM leaderboard from artificialanalysis.ai. Models are fetched from https://artificialanalysis.ai/leaderboards/models?weights=open and tracked exclusively with AA metrics. Run python3 tracker/fetch_aa.py to refresh.",
        "lastFetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Replace models
    data["models"] = models
    data["categories"] = categories

    # Write atomically
    tmp = DATA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, DATA)

    print(f"\nWrote {len(models)} models to {DATA}")


if __name__ == "__main__":
    main()
