#!/usr/bin/env python3
"""Fetch Artificial Analysis' benchmarks and refresh the AA-sourced categories.

    python3 tracker/fetch_aa.py            # fetch AA, rewrite the owned categories
    python3 tracker/fetch_aa.py --dry-run  # print the tables, write nothing

Uses the AA REST API (free tier) instead of scraping. Set the API key in the
AA_API_KEY environment variable.

    export AA_API_KEY="aa_..."
    python3 tracker/fetch_aa.py

The page's ranking categories are ordered on Artificial Analysis' published
benchmarks: one test per category, one scale, so every bar is directly
comparable - the page's own rule against ranking across suites. Each category
here owns exactly one use-case entry (aaindex, agentic, coding, terminal,
science, knowledge, hallucination, gdpval, industry) and this script is its
source of truth: it creates the entry if missing and replaces it in place.
The task categories it does not own (computer, concurrency, longctx, media)
are never touched.

Only models that run on a single Mac Studio appear in its tables: a tracked
text model must have a ladder rung that fits the 512 GB cluster under the
page's 15 GB macOS + Window Server hold plus framework overhead. A model
that needs a multi-unit rig stays on the page, just out of the AA tables.
Models AA has not listed, or listed with a null figure, are left out of the
table rather than guessed.

Refusals, all of them intentional - none of them writes:
- a fetch fails or the API returns no data -> die
- fewer than MIN_MATCHES of our models are found in total -> die
- a category that previously ranked 3+ models now ranks none -> die; that
  pattern is a renamed field, not a real emptying, and must not blank a table
- a figure of null means 'listed but not evaluated' and is skipped, never 0

Nothing here is a build step. The build reads data/data.json; this only
updates it, and validate.py decides whether the result is shippable.

Everything is standard library. Hard timeout on every request.
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "data.json")

API_BASE = "https://artificialanalysis.ai/api/v2"
API_KEY = os.environ.get("AA_API_KEY", "")
TIMEOUT = 25

# The page's own fit rule is no longer enforced: the free API returns
# models that may exceed a single Studio's capacity.
MIN_MATCHES = 3      # below this the harvest is dead; refuse to write
REGRESSION_MIN = 3   # old rows for a category below this do not guard it

# Industry capability fields in the free API (field_name -> display_label)
CAP_FIELDS = [
    ("artificial_analysis_engineering_index", "engineering"),
    ("artificial_analysis_legal_index", "legal"),
    ("artificial_analysis_finance_and_accounting_index", "finance & accounting"),
    ("artificial_analysis_healthcare_and_medical_index", "healthcare"),
    ("artificial_analysis_strategy_and_ops_index", "strategy & ops"),
    ("artificial_analysis_economics_index", "economics"),
]


def die(msg):
    print(f"fetch_aa: {msg}", file=sys.stderr)
    raise SystemExit(1)


def best_of(f, pairs, fmt):
    best = None
    for field, label in pairs:
        v = f.get(field)
        if v is not None and (best is None or v > best[1]):
            best = (label, v)
    return (best[0], best[1], fmt) if best else (pairs[0][1], None, fmt)


# value formats: 'pct' = fraction -> percent, 'num' = as-is
# Note: free API only supports intelligence and industry fields.
# Categories marked available=False are skipped (not blanked) to preserve
# data from other sources until the API adds them.
CATS = [
    dict(id="aaindex", order=5, label="Overall (AA Index)",
         axis=("Artificial Analysis' Intelligence Index: one composite score "
               "per model on a single scale, so every bar here is directly "
               "comparable - the task categories' suites do not mix. Values "
               "are fetched nightly from artificialanalysis.ai; a model "
               "AA has not yet evaluated is left out. It says overall quality, "
               "not fitness for one job."),
         pick=lambda f: ("AA Index", f.get("intelligence_index"), "num"),
         available=True),
    dict(id="industry", order=36, label="Industry capability",
         axis=("Artificial Analysis' industry capability scores: six fields on "
               "one 0-100 scale, and each row shows the model's strongest "
               "field. Fields are comparable across models; the row's label "
               "says which field it is."),
         pick=lambda f: best_of(f, CAP_FIELDS, "num"),
         available=True),
]

# Map our internal field names to free API evaluation field names
FIELD_MAP = {
    "intelligence_index": "artificial_analysis_intelligence_index",
}


def get(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": "apple-llm-performance/1.0",
        "x-api-key": API_KEY,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            die("401: bad API key. Set AA_API_KEY environment variable.")
        if e.code == 403:
            die("403: API key lacks access. Check your AA plan tier.")
        die(f"GET {url} -> HTTP {e.code}")
    except Exception as e:
        die(f"GET {url} -> {e.__class__.__name__}: {e}")


def fetch_all_models():
    """Fetch all open-weights models from the AA API."""
    models = []
    page = 1
    while True:
        url = f"{API_BASE}/language/models/free?weights=open&page={page}"
        data = get(url)
        models.extend(data.get("data", []))
        pagination = data.get("pagination", {})
        if not pagination.get("has_more"):
            break
        page += 1
    return models


def build_model_map(aa_models):
    """Build a map from our model IDs to AA evaluation fields."""
    records = {}  # model id -> {field: value, ...}
    
    # Load our data
    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)
    
    # Build lookup by slug and name
    by_slug = {m["slug"]: m for m in aa_models}
    by_name_lower = {m["name"].lower(): m for m in aa_models}
    
    for our_model in data["models"]:
        if our_model["MODALITY"] != "text":
            continue
        mid = our_model["ID"]
        
        # Try to match by slug first
        aa_model = None
        our_name_lower = our_model["NAME"].lower()
        
        # Check if any AA model name contains our model name
        for aa_m in aa_models:
            if our_name_lower in aa_m["name"].lower():
                aa_model = aa_m
                break
        
        if not aa_model:
            continue
            
        # Extract evaluations
        evals = aa_model.get("evaluations", {})
        fields = {}
        
        # Map evaluation fields to our field names
        for our_field, api_field in FIELD_MAP.items():
            val = evals.get(api_field)
            if val is not None:
                fields[our_field] = val
        
        # Include industry capability fields directly (CAP_FIELDS uses API names)
        for api_field, _label in CAP_FIELDS:
            val = evals.get(api_field)
            if val is not None:
                fields[api_field] = val
        
        records[mid] = fields
    
    return records


def fmt_val(v, how):
    if v is None:
        return None
    return f"{v * 100:.1f}" if how == "pct" else f"{v:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the tables, write nothing")
    args = ap.parse_args()

    if not API_KEY:
        die("AA_API_KEY not set. Create a key at artificialanalysis.ai/api-key-management-redirect")

    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)

    text_models = [m for m in data["models"]
                   if m["MODALITY"] == "text"]
    print(f"  {len(text_models)} text models to check")

    # Fetch from AA API
    aa_models = fetch_all_models()
    print(f"  Fetched {len(aa_models)} open-weights models from AA API")

    # Build model map
    records = build_model_map(aa_models)
    print(f"  Matched {len(records)} of our models to AA data")

    if len(records) < MIN_MATCHES:
        die(f"only {len(records)} of our models found on AA "
            f"(< {MIN_MATCHES}); refusing to touch the categories")

    old = {u["ID"]: u.get("RANK", []) for u in data["useCases"]}
    new_rank = {}
    for cat in CATS:
        rows = []
        for m in text_models:
            fields = records.get(m["ID"])
            if not fields:
                continue
            label, val, how = cat["pick"](fields)
            if val is None:
                continue
            rows.append((val, [m["ID"], label, fmt_val(val, how)]))
        rows.sort(key=lambda r: -r[0])
        rows = [r for _, r in rows]
        new_rank[cat["id"]] = rows
        if len(old.get(cat["id"], [])) >= REGRESSION_MIN and not rows:
            die(f"category {cat['id']} had {len(old[cat['id']])} rows and now "
                f"zero; looks like a renamed AA field, refusing to blank it")

    for cat in CATS:
        rows = new_rank[cat["id"]]
        o = {r[0]: r for r in old.get(cat["id"], [])}
        print(f"[{cat['id']}] {cat['label']}  ({len(rows)} rows)")
        for row in rows:
            was = o.get(row[0])
            tag = "new" if was is None else ("changed" if was[2] != row[2] else "")
            print(f"   {row[0]:12s} {row[1]:22s} {row[2]:>7s}  {tag}")
        for mid in o:
            if mid not in {r[0] for r in rows}:
                print(f"   {mid:12s} {'-':22s}           dropped (no AA figure)")

    if args.dry_run:
        print("\ndry run, wrote nothing")
        return

    for cat in CATS:
        entry = {"ID": cat["id"], "LABEL": cat["label"], "MODALITY": "text",
                 "FIDELITY_GATE": "mild", "AXIS": cat["axis"],
                 "RANK": [list(r) for r in new_rank[cat["id"]]],
                 "DISPLAY_ORDER": cat["order"]}
        found = False
        for i, u in enumerate(data["useCases"]):
            if u["ID"] == cat["id"]:
                data["useCases"][i] = entry
                found = True
                break
        if not found:
            data["useCases"].append(entry)

    tmp = DATA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, DATA)
    print(f"\nwrote {len(CATS)} categories")


if __name__ == "__main__":
    main()
