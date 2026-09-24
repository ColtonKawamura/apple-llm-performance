#!/usr/bin/env python3
"""Validate data/data.json, the single file the whole page renders from.

    python3 tracker/validate.py

Reports every problem it finds, not the first, and exits non-zero if any is
fatal. Warnings do not fail the build but are printed, because a record that is
merely thin is worth seeing without blocking a contribution.

This is the contract an agent has to satisfy. If a rule here feels wrong, change
the rule in a separate commit from the data, so a reviewer can see which of the
two moved.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import registry as R  # noqa: E402
from registry import DATA_FILE  # noqa: E402

ERRORS = []
WARNINGS = []

SEVERITIES = {"critical", "high", "medium", "low"}
KINDS = {"quant", "pruned", "native"}
ID_RE = re.compile(r"^[a-z][a-z0-9]*$")
ISSUE_RE = re.compile(r"^[\w.-]+/[\w.-]+#\d+$")
FIDELITY_BANDS = {"full", "mild", "low", "unusable"}

# Every attribute a record must carry, by section. Adding a required field to
# the page is a two-line change here and in registry.py, and this list is the
# single place that states it.
MODEL_FIELDS = ["ID", "MODALITY", "NAME", "ARCH", "LICENSE", "CONTEXT", "HF",
                "NOTE", "SOURCES", "PARAMS_B", "BEST_ENGINE", "ENGINES"]
ENGINE_FIELDS = ["ID", "NAME", "MODALITIES", "FORMAT", "INTERFACE", "API",
                 "LICENSE", "SITE", "PROSE_ALIASES", "WHAT",
                 "DISPLAY_ORDER", "QUANT_FAMILY"]
USECASE_FIELDS = ["ID", "LABEL", "MODALITY", "FIDELITY_GATE", "AXIS",
                  "RANK", "DISPLAY_ORDER"]


def err(where, msg):
    ERRORS.append(f"{where}: {msg}")


def warn(where, msg):
    WARNINGS.append(f"{where}: {msg}")


def load_data():
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        err(DATA_FILE, "missing; it is the single source the page renders from")
        return None
    except json.JSONDecodeError as e:
        err(DATA_FILE, f"is not valid JSON: {e}")
        return None


def check_schema_version(d):
    meta = d.get("_meta") or {}
    if meta.get("schema_version") != 1:
        err(DATA_FILE, f"unsupported _meta.schema_version {meta.get('schema_version')!r}; "
                       "this checkout's tracker/ code speaks version 1")


# ------------------------------------------------------------------- engines
def check_engines(d):
    seen_order = {}
    alias_owner = {}
    for m in d.get("engines", []):
        w = f"engines[{m.get('ID', '?')}]"
        if not ID_RE.match(m.get("ID", "")):
            err(w, f"ID {m.get('ID')!r} must be lowercase alphanumeric")
        for field in ENGINE_FIELDS:
            if not m.get(field) and field != "ID":
                err(w, f"missing or empty {field}")
        for mod in m.get("MODALITIES", []):
            if mod not in R.MODALITIES:
                err(w, f"unknown modality {mod!r}; expected one of {R.MODALITIES}")
        if m.get("QUANT_FAMILY") not in {"gguf", "mlx", "ds4"}:
            err(w, f"unknown QUANT_FAMILY {m.get('QUANT_FAMILY')!r}")
        order = m.get("DISPLAY_ORDER")
        if order is None:
            err(w, "missing DISPLAY_ORDER")
        elif order in seen_order:
            err(w, f"DISPLAY_ORDER {order} already used by {seen_order[order]}")
        else:
            seen_order[order] = m["ID"]
        if m.get("SITE") and not str(m["SITE"]).startswith("http"):
            err(w, f"SITE {m['SITE']!r} must be a URL; it is linked from the prose")
        for alias in m.get("PROSE_ALIASES", []):
            if not alias or not alias.strip():
                err(w, "PROSE_ALIASES contains an empty name")
            elif alias in alias_owner and alias_owner[alias] != m["ID"]:
                err(w, f"alias {alias!r} is also claimed by {alias_owner[alias]!r}; "
                       "an ambiguous name would link to the wrong engine")
            else:
                alias_owner[alias] = m["ID"]
        if len(m.get("WHAT", "")) < 80:
            warn(w, "WHAT is very short; it is the reader's only description of this engine")
        feed = m.get("RELEASE_FEED")
        if feed is not None:
            if feed.get("scheme") not in {"release", "semver", "none"}:
                err(w, f"RELEASE_FEED scheme {feed.get('scheme')!r} must be release/semver/none")
            if feed.get("scheme") != "none" and not feed.get("repo"):
                err(w, "RELEASE_FEED needs a repo unless scheme is 'none'")
        for key in m.get("CROSS_ISSUES", []):
            if not ISSUE_RE.match(key):
                err(w, f"malformed issue key {key!r}; expected owner/repo#123")
            elif key not in R.EMETA:
                err(w, f"CROSS_ISSUES cites {key} which has no entry in the issues section")


# -------------------------------------------------------------------- models
def check_models(d):
    seen_ids = set()
    for m in d.get("models", []):
        w = f"models[{m.get('ID', '?')}]"
        if not ID_RE.match(m.get("ID", "")):
            err(w, f"ID {m.get('ID')!r} must be lowercase alphanumeric")
        if m["ID"] in seen_ids:
            err(w, "duplicate model ID")
        seen_ids.add(m["ID"])
        if m.get("MODALITY") not in R.MODALITIES:
            err(w, f"unknown MODALITY {m.get('MODALITY')!r}")
        for field in MODEL_FIELDS:
            if not m.get(field):
                err(w, f"missing or empty {field}")
        if m.get("PARAMS_B") in (None, 0):
            err(w, "PARAMS_B must be the total parameter count in billions")
        if not m.get("SOURCES"):
            err(w, "SOURCES must cite at least one link; every figure needs a provenance")
        for src in m.get("SOURCES", []):
            if not isinstance(src, list) or len(src) != 2 or not str(src[1]).startswith("http"):
                err(w, f"malformed SOURCES entry {src!r}; expected [label, url]")
        if len(m.get("NOTE", "")) < 80:
            warn(w, "NOTE is very short; it is the reason to read the page over a spec sheet")

        # engine cells
        cells = m.get("ENGINES") or {}
        if not cells:
            err(w, "ENGINES is empty; a model with no engine cell renders no tabs")
        for eid, c in cells.items():
            at = f"{w}.ENGINES[{eid}]"
            if eid not in R.ENGINE_BY_ID:
                err(at, f"unknown engine {eid!r}")
                continue
            eng = R.ENGINE_BY_ID[eid]
            if m.get("MODALITY") not in eng["mods"]:
                err(at, f"engine handles {eng['mods']} but this model is {m.get('MODALITY')!r}")
            if c.get("status") not in R.STATUSES:
                err(at, f"status {c.get('status')!r} must be one of {R.STATUSES}")
            if not c.get("label"):
                err(at, "missing label")
            if not c.get("note"):
                err(at, "missing note; a status with no explanation is not reviewable")
            for key in c.get("issues", []):
                if not ISSUE_RE.match(key):
                    err(at, f"malformed issue key {key!r}")
                elif key not in R.EMETA:
                    err(at, f"cites {key} which has no entry in the issues section")
        if m.get("BEST_ENGINE") not in cells:
            err(w, f"BEST_ENGINE {m.get('BEST_ENGINE')!r} has no cell in ENGINES")
        elif cells.get(m.get("BEST_ENGINE"), {}).get("status") == "none":
            err(w, f"BEST_ENGINE {m['BEST_ENGINE']!r} is marked out of scope")

        # quant ladder
        lad = m.get("LADDER") or {}
        # only families an in-scope engine would actually load
        fams = {R.FAM[eid] for eid, c in cells.items()
                if eid in R.FAM and c.get("status") != "none"}
        for fam in fams:
            if fam not in lad or not lad[fam]:
                warn(w, f"no measured ladder for the {fam!r} family, which its engines load")
        for fam, rungs in lad.items():
            last = None
            for r in rungs:
                at = f"{w}.LADDER[{fam} {r.get('label', '?')}]"
                if r.get("kind") not in KINDS:
                    err(at, f"kind {r.get('kind')!r} must be one of {sorted(KINDS)}")
                if not isinstance(r.get("gb"), (int, float)) or r["gb"] <= 0:
                    err(at, "gb must be a positive size in gigabytes")
                if r.get("kind") == "quant":
                    if r.get("bpw") is None:
                        err(at, "a quant rung needs bpw")
                    elif not 0.5 <= r["bpw"] <= 20:
                        err(at, f"bpw {r['bpw']} is outside a believable range; check PARAMS_B")
                elif r.get("bpw") is not None:
                    err(at, f"kind {r['kind']!r} must not carry a bpw")
                if not r.get("repo"):
                    err(at, "missing repo")
                if last is not None and r["gb"] > last:
                    err(at, "ladder must be ordered largest first")
                last = r["gb"]

        # KV geometry
        kv = m.get("KV")
        if kv is None or set(kv) != {"bytes_per_token", "max_context", "derivation"}:
            err(w, "KV must be {bytes_per_token, max_context, derivation}")
        elif kv["bytes_per_token"] is not None:
            if m.get("MODALITY") != "text":
                err(w, "only text models should declare a per-token KV cost")
            if not kv["derivation"]:
                err(w, "a KV figure needs a derivation saying which layers were counted")
            if not kv["max_context"]:
                err(w, "a KV figure needs max_context to size a stream against")


# ----------------------------------------------------------------- use cases
def check_use_cases(d):
    seen_order = {}
    for m in d.get("useCases", []):
        w = f"useCases[{m.get('ID', '?')}]"
        if not ID_RE.match(m.get("ID", "")):
            err(w, f"ID {m.get('ID')!r} must be lowercase alphanumeric")
        for field in USECASE_FIELDS:
            if not m.get(field):
                err(w, f"missing or empty {field}")
        if m.get("MODALITY") not in R.MODALITIES:
            err(w, f"unknown MODALITY {m.get('MODALITY')!r}")
        if m.get("FIDELITY_GATE") not in FIDELITY_BANDS:
            err(w, f"FIDELITY_GATE {m.get('FIDELITY_GATE')!r} is not a band name")
        order = m.get("DISPLAY_ORDER")
        if order is None:
            err(w, "missing DISPLAY_ORDER")
        elif order in seen_order:
            err(w, f"DISPLAY_ORDER {order} already used by {seen_order[order]}")
        else:
            seen_order[order] = m["ID"]
        seen = set()
        for entry in m.get("RANK", []):
            if not isinstance(entry, list) or len(entry) != 3:
                err(w, f"RANK entry {entry!r} must be [model_id, metric, value]")
                continue
            mid, metric, value = entry
            if mid not in R.MODEL_BY_ID:
                err(w, f"RANK cites unknown model {mid!r}")
                continue
            if mid in seen:
                err(w, f"RANK lists {mid!r} twice")
            seen.add(mid)
            if R.MODEL_BY_ID[mid]["mod"] != m.get("MODALITY"):
                err(w, f"RANK cites {mid!r} ({R.MODEL_BY_ID[mid]['mod']}) in a "
                       f"{m.get('MODALITY')} category")
            if not metric or not str(value):
                err(w, f"RANK entry for {mid!r} needs both a metric name and a value")


# -------------------------------------------------------------------- issues
def check_issues(d):
    repos = set()
    for m in R.ISSUE_MODULES:
        repo = m.get("repo", "")
        w = f"issues[{repo}]"
        if not re.match(r"^[\w.-]+/[\w.-]+$", repo):
            err(w, f"repo {repo!r} must be owner/name")
        elif repo in repos:
            err(w, "two issue trackers for the same repo; merge them into one entry")
        repos.add(repo)
        if not m.get("issues"):
            err(w, "tracks no issues; a tracker with an empty list is dead weight")
        # ISSUES is the registry's view of the raw JSON: string keys coerced to
        # ints, which is the form the page and probe.py consume.
        for num, meta in (m.get("ISSUES") or {}).items():
            at = f"{w}['{num}']"
            if not isinstance(num, int) or num <= 0:
                err(at, "issue number must be a positive integer")
            if meta.get("severity") not in SEVERITIES:
                err(at, f"severity {meta.get('severity')!r} must be one of {sorted(SEVERITIES)}")
            if not meta.get("headline"):
                err(at, "missing headline")
            if not meta.get("why"):
                err(at, "missing 'why'; an issue with no consequence stated is noise")


# ------------------------------------------------------------------- global
def check_global(d):
    cited = set()
    for mid, cells in R.MATRIX.items():
        for c in cells.values():
            cited.update(c["items"])
    for keys in R.CROSS_BY_ENGINE.values():
        cited.update(keys)
    orphans = sorted(set(R.EMETA) - cited)
    if orphans:
        warn("issues", f"{len(orphans)} issues are tracked but cited nowhere: "
                       + ", ".join(orphans[:6]) + ("..." if len(orphans) > 6 else ""))
    prose_blocks = [m["NOTE"] for m in d.get("models", [])]
    prose_blocks += [c["note"] for m in d.get("models", []) for c in (m.get("ENGINES") or {}).values()]
    prose_blocks += [m["WHAT"] for m in d.get("engines", [])]
    haystack = "\n".join(prose_blocks)
    for alias, eid, _site in R.ENGINE_PROSE_LINKS:
        # An alias drawn from the engine's own name is legitimate even when no
        # note happens to use it yet. Anything else that matches nothing is a
        # typo, and will silently never link.
        if alias in R.ENGINE_BY_ID[eid]["name"]:
            continue
        if not re.search(rf"(?<![\w.-]){re.escape(alias)}(?![\w-])", haystack):
            warn(f"engines[{eid}]",
                 f"PROSE_ALIASES lists {alias!r}, which is neither this engine's name "
                 "nor used in any note; nothing will ever link")

    for key in sorted(R.PR_KEYS):
        if key not in R.EMETA:
            err("prKeys", f"{key} is listed as a PR but has no issue entry")
    for mod in R.MODALITIES:
        if not any(m["mod"] == mod for m in R.MODELS):
            warn("models", f"no models with modality {mod!r}")
        if not any(u["mod"] == mod for u in R.USE_CASES):
            warn("useCases", f"no use case with modality {mod!r}")


# --------------------------------------------------------------- code shape
# The renderer imports its data from the registry. If it also defines one of
# those names at module level, the local definition silently wins and the data/
# file stops mattering - which is exactly how a hand-maintained PR_KEYS set and
# a 30-entry issue table survived the split and shadowed the real ones.
def check_no_shadowing():
    import ast
    path = os.path.join(HERE, "render_status.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    imported = set()
    assigned = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "registry":
            imported.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned[t.id] = node.lineno
    for name in sorted(imported & set(assigned)):
        err(f"tracker/render_status.py:{assigned[name]}",
            f"{name} is imported from the registry and then reassigned here; "
            "the local value would shadow data/data.json and render stale facts")


def check_watch_state():
    """The polled issue states the page renders its open/closed pills from.

    This is the one input that is written by redirecting a command's stdout over
    it, which is how it once got truncated to nothing. The page still rendered -
    build.py reads whatever is there - so neither the build nor the output check
    noticed. Verify it covers what is tracked.
    """
    path = os.path.join(HERE, "watch-state.txt")
    if not os.path.exists(path):
        err("tracker/watch-state.txt", "missing; run tracker/probe.py")
        return
    keys = set()
    for line in open(path, encoding="utf-8"):
        k = line.split("|", 1)[0].strip()
        if k:
            keys.add(k)
    if not keys:
        err("tracker/watch-state.txt", "is empty; the page would show no issue states at all")
        return
    tracked = set(R.EMETA)
    missing = tracked - {k for k in keys if "@" not in k}
    if len(missing) > len(tracked) // 4:
        err("tracker/watch-state.txt",
            f"has no state for {len(missing)} of {len(tracked)} tracked issues; "
            "it looks truncated - re-run tracker/probe.py")
    elif missing:
        warn("tracker/watch-state.txt",
             f"{len(missing)} tracked issues have no polled state yet: "
             + ", ".join(sorted(missing)[:4]))


def main():
    d = load_data()
    if d is None:
        print("\n1 error, 0 warnings")
        return 1
    check_schema_version(d)
    check_engines(d)
    check_models(d)
    check_use_cases(d)
    check_issues(d)
    check_global(d)
    check_no_shadowing()
    check_watch_state()

    for wmsg in WARNINGS:
        print(f"warning: {wmsg}")
    for e in ERRORS:
        print(f"error:   {e}")
    print(f"\n{len(R.MODELS)} models, {len(R.ENGINES)} engines, "
          f"{len(R.USE_CASES)} use cases, {len(R.EMETA)} tracked issues")
    print(f"{len(ERRORS)} errors, {len(WARNINGS)} warnings")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    raise SystemExit(main())
