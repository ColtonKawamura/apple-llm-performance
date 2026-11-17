#!/usr/bin/env python3
"""Validate data/data.json, the single file the whole page renders from.

    python3 tracker/validate.py

Reports every problem it finds and exits non-zero if any is fatal.
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

ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")

METRIC_FIELDS = [
    "intelligenceIndex", "costPerTask", "timePerTask", "latency", "contextWindow"
]


def err(where, msg):
    ERRORS.append(f"{where}: {msg}")


def warn(where, msg):
    WARNINGS.append(f"{where}: {msg}")


def load_data():
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        err(DATA_FILE, "missing")
        return None
    except json.JSONDecodeError as e:
        err(DATA_FILE, f"invalid JSON: {e}")
        return None


def check_schema(d):
    meta = d.get("_meta") or {}
    if meta.get("schema_version") != 2:
        err("_meta", f"schema_version {meta.get('schema_version')!r} is not supported (need 2)")


def check_models(d):
    seen_ids = set()
    for m in d.get("models", []):
        mid = m.get("id", "?")
        w = f"models[{mid}]"

        if not ID_RE.match(mid):
            err(w, f"id {mid!r} must be lowercase alphanumeric with hyphens")
        if mid in seen_ids:
            err(w, "duplicate id")
        seen_ids.add(mid)

        # Required fields
        for field in ["id", "name", "creator"]:
            if not m.get(field):
                err(w, f"missing {field}")

        # At least one metric should be present
        has_metric = any(m.get(f) is not None for f in METRIC_FIELDS)
        if not has_metric:
            warn(w, "no metrics set; model will not appear in any ranking")

        # Validate URLs
        if m.get("aaUrl") and not m["aaUrl"].startswith("http"):
            err(w, f"aaUrl must be an http(s) URL")

        # Cross-reference: check metric values are reasonable
        if m.get("intelligenceIndex") is not None:
            if not isinstance(m["intelligenceIndex"], (int, float)):
                err(w, "intelligenceIndex must be a number")
        if m.get("contextWindow") is not None:
            if not isinstance(m["contextWindow"], int):
                err(w, "contextWindow must be an integer")


def check_categories(d):
    seen_order = {}
    for c in d.get("categories", []):
        cid = c.get("id", "?")
        w = f"categories[{cid}]"

        if not ID_RE.match(cid):
            err(w, f"id {cid!r} must be lowercase alphanumeric with hyphens")
        for field in ["id", "label", "metric", "order"]:
            if not c.get(field) and field != "order":
                err(w, f"missing {field}")

        # Check metric exists
        if c.get("metric") not in METRIC_FIELDS:
            err(w, f"metric {c.get('metric')!r} is not a known field")

        # Check uniqueness of order
        order = c.get("order")
        if order is None:
            err(w, "missing order")
        elif order in seen_order:
            err(w, f"order {order} already used by {seen_order[order]}")
        else:
            seen_order[order] = cid


def check_cross_refs(d):
    """Check that categories reference valid model ids."""
    model_ids = {m["id"] for m in d.get("models", [])}
    for cat in d.get("categories", []):
        w = f"categories[{cat['id']}]"
        metric = cat.get("metric")
        if not metric:
            continue
        # Check at least some models have this metric
        count = sum(1 for m in d.get("models", []) if m.get(metric) is not None)
        if count == 0 and model_ids:
            warn(w, f"no models have metric {metric!r}")


def main():
    d = load_data()
    if d is None:
        print("\n1 error, 0 warnings")
        return 1

    check_schema(d)
    check_models(d)
    check_categories(d)
    check_cross_refs(d)

    for w in WARNINGS:
        print(f"warning: {w}")
    for e in ERRORS:
        print(f"error:   {e}")

    print(f"\n{len(d.get('models', []))} models, {len(d.get('categories', []))} categories")
    print(f"{len(ERRORS)} errors, {len(WARNINGS)} warnings")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    raise SystemExit(main())
