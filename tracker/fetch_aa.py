#!/usr/bin/env python3
"""Fetch Artificial Analysis' Intelligence Index and refresh the aaindex category.

    python3 tracker/fetch_aa.py            # fetch AA, rewrite the aaindex RANK
    python3 tracker/fetch_aa.py --dry-run  # print the table, write nothing

The page's 'Overall (AA Index)' category is ordered on Artificial Analysis'
Intelligence Index: one composite score per model, on a single scale, so every
bar is directly comparable - unlike the task categories, whose suites do not
mix. Values are read from artificialanalysis.ai's embedded Next.js payload; no
API key is needed. Matching is by each model's Hugging Face repo, taken from
the page's modelWeightsSourceUrl field, so a rename on either side does not
break the join.

Fetching is two-stage, because the /models page only embeds its ~12 frontier
open-weights records:
1. one GET of /models covers the top of the index;
2. every one of our models not matched there is probed at /models/<slug>,
   where <slug> is the model NAME (then the HF repo basename) with every
   non-alphanumeric run collapsed to '-'. A wrong slug returns 200 with a
   DIFFERENT model's page, so a page is accepted only when its own record's
   modelWeightsSourceUrl equals this model's HF repo, case-insensitively.
   A small pause between probes keeps the traffic polite.

The script owns exactly one record: the useCases entry with ID aaindex. It
creates that entry if it is missing and rewrites its RANK in place; it never
touches any other entry, and on an update it leaves LABEL/AXIS alone so prose
edited by hand in data.json is not clobbered by a nightly run.

Nothing here is a build step. The build reads data/data.json; this only
updates it, the same way tracker/add_model.py does, and validate.py decides
whether the result is shippable.

Refusals, all of them intentional:
- a fetch fails or the page shape changed (no model records at all) -> die
- fewer than MIN_MATCHES of our models are found -> die; a partial harvest
  must never blank the category, so no write happens
- a model with intelligenceIndex null on AA is skipped, not written as 0
  (null means 'listed but not evaluated')

Everything is standard library. Hard timeout on every request; one retry on
429 for the /models page.
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

MODELS_URL = "https://artificialanalysis.ai/models"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
TIMEOUT = 25
PROBE_PAUSE = 0.4  # between per-model page fetches

CAT_ID = "aaindex"
CAT_LABEL = "Overall (AA Index)"
CAT_AXIS = ("Artificial Analysis' Intelligence Index: one composite score per "
            "model on a single scale, so every bar here is directly comparable "
            "- the task categories' suites do not mix. Values are fetched "
            "nightly from artificialanalysis.ai/models; a model AA has not "
            "yet evaluated is left out. It says overall quality, not fitness "
            "for one job.")
METRIC = "AA Index"
MIN_MATCHES = 3  # below this the harvest is dead; refuse to write


def die(msg):
    print(f"fetch_aa: {msg}", file=sys.stderr)
    raise SystemExit(1)


def get(url, retries=1):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(5)
                continue
            if e.code == 404:
                return None
            die(f"GET {url} -> HTTP {e.code}")
        except Exception as e:
            die(f"GET {url} -> {e.__class__.__name__}: {e}")
    die(f"GET {url} -> gave up after retry")


def payload(html):
    """Decode the __next_f flight chunks into one text blob."""
    chunks = re.findall(
        r'self\.__next_f\.push\(\[1,\s*("(?:[^"\\]|\\.)*")\s*\]\)', html, re.S)
    if not chunks:
        return ""
    return "".join(json.loads(c) for c in chunks)


def repo_of(url):
    return url.rsplit("/huggingface.co/", 1)[-1].strip("/").lower()


def index_after(blob, pos, window=4000):
    """The intelligenceIndex in the record that starts at `pos` (its
    modelWeightsSourceUrl match). None when absent or null."""
    im = re.search(r'"intelligenceIndex":(null|-?\d+(?:\.\d+)?)',
                   blob[pos:pos + window])
    if not im:
        return None
    return None if im.group(1) == "null" else float(im.group(1))


def parse_frontier(blob):
    """Every (hf repo, index) pair embedded in the /models page."""
    out, seen = [], set()
    for m in re.finditer(r'"modelWeightsSourceUrl":"(https://huggingface\.co/[^"]+)"',
                         blob):
        repo = repo_of(m.group(1))
        if repo in seen:
            continue
        seen.add(repo)
        out.append((repo, index_after(blob, m.end())))
    return out


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def probe_model(blob, repo):
    """The page's OWN record: first modelWeightsSourceUrl in the blob.
    Returns its index, or (None, False) when the page is a different model."""
    m = re.search(r'"modelWeightsSourceUrl":"(https://huggingface\.co/[^"]+)"',
                  blob)
    if not m or repo_of(m.group(1)) != repo.lower():
        return None, False
    return index_after(blob, m.end()), True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the table, write nothing")
    args = ap.parse_args()

    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)

    text_models = [m for m in data["models"]
                   if m["MODALITY"] == "text" and m.get("HF")]
    found = {}  # model id -> index

    # Stage 1: the /models page carries the top of the index in one GET.
    html = get(MODELS_URL, retries=1)
    blob = payload(html)
    frontier = parse_frontier(blob)
    if not frontier:
        die("no model records parsed from /models; page layout changed, "
            "refusing to guess")
    by_repo = {m["HF"].strip().lower(): m["ID"] for m in text_models}
    for repo, val in frontier:
        if repo in by_repo and val is not None:
            found[by_repo[repo]] = val

    # Stage 2: probe /models/<slug> for whatever stage 1 missed.
    for m in text_models:
        if m["ID"] in found:
            continue
        repo = m["HF"].strip().lower()
        candidates = [slug(m["NAME"])]
        base = repo.rsplit("/", 1)[-1]
        if slug(base) and slug(base) not in candidates:
            candidates.append(slug(base))
        for cand in candidates:
            page = get(f"https://artificialanalysis.ai/models/{cand}")
            time.sleep(PROBE_PAUSE)
            if page is None:
                continue
            val, own = probe_model(payload(page), repo)
            if own:
                if val is not None:
                    found[m["ID"]] = val
                break  # right model either way (evaluated or not)

    if len(found) < MIN_MATCHES:
        die(f"only {len(found)} of our models found on AA "
            f"(< {MIN_MATCHES}); refusing to touch the category")

    rows = sorted(((v, mid) for mid, v in found.items()), key=lambda r: -r[0])
    rows = [[mid, METRIC, f"{v:.1f}"] for v, mid in rows]

    old = []
    for u in data["useCases"]:
        if u["ID"] == CAT_ID:
            old = u.get("RANK", [])
            break

    old_by_id = {r[0]: r[2] for r in old}
    for row in rows:
        was = old_by_id.get(row[0])
        tag = "new" if was is None else ("changed" if was != row[2] else "")
        print(f"  {row[0]:12s} {row[2]:>6s}  {tag}")
    for mid in old_by_id:
        if mid not in {r[0] for r in rows}:
            print(f"  {mid:12s}     -  dropped (no longer on AA or evaluated)")

    if args.dry_run:
        print(f"\n{len(rows)} rows; dry run, wrote nothing")
        return

    entry = {"ID": CAT_ID, "LABEL": CAT_LABEL, "MODALITY": "text",
             "FIDELITY_GATE": "mild", "AXIS": CAT_AXIS,
             "RANK": [list(r) for r in rows], "DISPLAY_ORDER": 5}
    found_entry = False
    for i, u in enumerate(data["useCases"]):
        if u["ID"] == CAT_ID:
            data["useCases"][i]["RANK"] = [list(r) for r in rows]
            found_entry = True
            break
    if not found_entry:
        data["useCases"].append(entry)  # append; display order comes from DISPLAY_ORDER

    tmp = DATA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, DATA)
    print(f"\nwrote {len(rows)} rows into the aaindex category")


if __name__ == "__main__":
    main()
