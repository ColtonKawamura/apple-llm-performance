#!/usr/bin/env python3
"""Fetch Artificial Analysis' benchmarks and refresh the AA-sourced categories.

    python3 tracker/fetch_aa.py            # fetch AA, rewrite the owned categories
    python3 tracker/fetch_aa.py --dry-run  # print the tables, write nothing

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

Fetching is two-stage, because the /models page only embeds its ~12 frontier
open-weights records:
1. one GET of /models covers the top of the index;
2. every tracked model not matched there is probed at /models/<slug>, where
   <slug> is the model NAME (then the HF repo basename) with every
   non-alphanumeric run collapsed to '-'. A wrong slug returns 200 with a
   DIFFERENT model's page, so a page is accepted only when its own record's
   modelWeightsSourceUrl equals this model's HF repo, case-insensitively.
   A small pause between probes keeps the traffic polite.

Refusals, all of them intentional - none of them writes:
- a fetch fails or the page shape changed (no model records at all) -> die
- fewer than MIN_MATCHES of our models are found in total -> die
- a category that previously ranked 3+ models now ranks none -> die; that
  pattern is a renamed field, not a real emptying, and must not blank a table
- a figure of null means 'listed but not evaluated' and is skipped, never 0

Nothing here is a build step. The build reads data/data.json; this only
updates it, and validate.py decides whether the result is shippable.

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
RECORD_WINDOW = 8000  # every field we read sits within this of the URL field

# The page's own fit rule, mirrored so a model only enters the AA tables when
# a single Studio can actually load it (render_status.py: g - OS_HOLD + overhead).
STUDIO_GB = 512
OS_HOLD = 15  # macOS + Window Server per machine
OVERHEAD = 10

MIN_MATCHES = 3      # below this the harvest is dead; refuse to write
REGRESSION_MIN = 3   # old rows for a category below this do not guard it

CAP_FIELDS = [("engineering", "engineering"), ("legal", "legal"),
              ("financeAndAccounting", "finance & accounting"),
              ("healthcareAndMedical", "healthcare"),
              ("strategyAndOps", "strategy & ops"), ("economics", "economics")]


def die(msg):
    print(f"fetch_aa: {msg}", file=sys.stderr)
    raise SystemExit(1)


def first_of(f, pairs, fmt):
    for field, label in pairs:
        if f.get(field) is not None:
            return (label, f[field], fmt)
    return (pairs[0][1], None, fmt)


def best_of(f, pairs, fmt):
    best = None
    for field, label in pairs:
        v = f.get(field)
        if v is not None and (best is None or v > best[1]):
            best = (label, v)
    return (best[0], best[1], fmt) if best else (pairs[0][1], None, fmt)


# value formats: 'pct' = fraction -> percent, 'num' = as-is
CATS = [
    dict(id="aaindex", order=5, label="Overall (AA Index)",
         axis=("Artificial Analysis' Intelligence Index: one composite score "
               "per model on a single scale, so every bar here is directly "
               "comparable - the task categories' suites do not mix. Values "
               "are fetched nightly from artificialanalysis.ai/models; a model "
               "AA has not yet evaluated is left out. It says overall quality, "
               "not fitness for one job."),
         pick=lambda f: ("AA Index", f.get("intelligenceIndex"), "num")),
    dict(id="agentic", order=10, label="Agentic & tool use",
         axis=("Artificial Analysis' agentic suites - tau2, IT-Bench SRE and "
               "CRITPT-hard, which are different tests, so the row's label says "
               "which one it is. Pass rates on a common 0-100 scale; they "
               "justify each placement rather than being directly comparable."),
         pick=lambda f: first_of(f, [("tau2", "tau2"),
                                     ("itBenchSre", "IT-Bench SRE"),
                                     ("critptAgenticHard", "CRITPT-hard")], "pct")),
    dict(id="coding", order=20, label="Coding",
         axis=("Artificial Analysis' Frontier Code: their own coding benchmark, "
               "one test for every model, so the bars are directly comparable. "
               "Fetched nightly from artificialanalysis.ai/models."),
         pick=lambda f: ("Frontier Code", f.get("frontierCode"), "pct")),
    dict(id="terminal", order=30, label="Terminal & CLI work",
         axis=("Terminal-Bench 4.0 as run by Artificial Analysis: one version "
               "for every model, so the bars are directly comparable. A model "
               "AA has not run through TB 4.0 is left out rather than mixed "
               "with other versions - it is a different test."),
         pick=lambda f: ("Terminal-Bench 4.0", f.get("terminalBench40"), "pct")),
    dict(id="science", order=32, label="Science",
         axis=("Artificial Analysis' SciCode: scientific reasoning with code, "
               "one test for every model, so the bars are directly comparable."),
         pick=lambda f: ("SciCode", f.get("scicode"), "pct")),
    dict(id="knowledge", order=33, label="Knowledge",
         axis=("Humanity's Last Exam, diamond set, as run by Artificial "
               "Analysis: one test for every model, so the bars are directly "
               "comparable."),
         pick=lambda f: ("HLE Diamond", f.get("hleDiamond"), "pct")),
    dict(id="hallucination", order=34, label="Hallucination",
         axis=("Omniscience accuracy: the share of factual questions a model "
               "answers correctly without hallucinating, per Artificial "
               "Analysis. Higher is better - it is the complement of their "
               "hallucination rate, and one metric, so directly comparable."),
         pick=lambda f: ("Omniscience accuracy", f.get("omniscienceAccuracy"), "pct")),
    dict(id="gdpval", order=35, label="Economic work (GDPval)",
         axis=("GDPval, Artificial Analysis' benchmark of real economic work, "
               "scored Elo-style: one scale for every model, directly "
               "comparable."),
         pick=lambda f: ("GDPval", f.get("gdpval"), "num")),
    dict(id="industry", order=36, label="Industry capability",
         axis=("Artificial Analysis' industry capability scores: six fields on "
               "one 0-100 scale, and each row shows the model's strongest "
               "field. Fields are comparable across models; the row's label "
               "says which field it is."),
         pick=lambda f: best_of(f, CAP_FIELDS, "num")),
]

FIELDS = ["intelligenceIndex", "gdpval", "omniscienceAccuracy",
          "omniscienceHallucinationRate", "terminalBench21", "terminalBench40",
          "terminalbenchHard", "terminalBenchScience", "frontierCode",
          "frontierCode20", "hleDiamond", "scicode", "vialsScore", "healthbench",
          "tau2", "tauBanking", "itBenchSre", "critptAgenticHard", "mlcrOverall",
          "gdpPdfAllPass"]


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


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def num(field, window):
    m = re.search(rf'"{field}":(null|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', window)
    if not m:
        return None
    return None if m.group(1) == "null" else float(m.group(1))


def parse_fields(window):
    f = {name: num(name, window) for name in FIELDS}
    caps = re.search(r'"capabilities":\{(.*?)\}', window, re.S)
    if caps:
        for field, _label in CAP_FIELDS:
            f[field] = num(field, caps.group(1))
    return f


def record_at(blob, pos):
    m = re.match(r'"modelWeightsSourceUrl":"(https://huggingface\.co/[^"]+)"',
                 blob[pos:])
    if not m:
        return None, None
    return repo_of(m.group(1)), parse_fields(blob[pos:pos + RECORD_WINDOW])


def parse_frontier(blob):
    """Every (hf repo, fields) pair embedded in the /models page."""
    out, seen = [], set()
    for m in re.finditer(r'"modelWeightsSourceUrl"', blob):
        repo, fields = record_at(blob, m.start())
        if repo and repo not in seen:
            seen.add(repo)
            out.append((repo, fields))
    return out


def probe_own(blob, repo):
    """The page's OWN record: first modelWeightsSourceUrl in the blob.
    Returns fields, or None when the page is a different model."""
    m = re.search(r'"modelWeightsSourceUrl"', blob)
    if not m:
        return None
    own, fields = record_at(blob, m.start())
    return fields if own == repo else None


def fits_studio(model):
    """True when some ladder rung loads on one 512 GB Studio under the
    page's own fit rule."""
    usable = STUDIO_GB - OS_HOLD
    for fam in (model.get("LADDER") or {}).values():
        if isinstance(fam, list):
            for r in fam:
                if r.get("kind") in ("quant", "pruned", "native") \
                        and r.get("gb", 10 ** 9) + OVERHEAD <= usable:
                    return True
        elif isinstance(fam, dict):
            for lst in fam.values():
                if isinstance(lst, list):
                    for r in lst:
                        if r.get("kind") in ("quant", "pruned", "native") \
                                and r.get("gb", 10 ** 9) + OVERHEAD <= usable:
                            return True
    return False


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

    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)

    text_models = [m for m in data["models"]
                   if m["MODALITY"] == "text" and m.get("HF")]
    studio = [m for m in text_models if fits_studio(m)]
    out_of_fit = [m["ID"] for m in text_models if m not in studio]
    if out_of_fit:
        print(f"  (out of the AA tables, no single Studio fits: {', '.join(out_of_fit)})")

    records = {}  # model id -> fields
    html = get(MODELS_URL, retries=1)
    blob = payload(html)
    frontier = parse_frontier(blob)
    if not frontier:
        die("no model records parsed from /models; page layout changed, "
            "refusing to guess")
    by_repo = {m["HF"].strip().lower(): m for m in studio}
    for repo, fields in frontier:
        if repo in by_repo:
            records[by_repo[repo]["ID"]] = fields

    for m in studio:
        if m["ID"] in records:
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
            fields = probe_own(payload(page), repo)
            if fields is not None:
                records[m["ID"]] = fields
                break  # right model either way (evaluated or not)

    if len(records) < MIN_MATCHES:
        die(f"only {len(records)} of our Studio-fitting models found on AA "
            f"(< {MIN_MATCHES}); refusing to touch the categories")

    old = {u["ID"]: u.get("RANK", []) for u in data["useCases"]}
    new_rank = {}
    for cat in CATS:
        rows = []
        for m in studio:
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
