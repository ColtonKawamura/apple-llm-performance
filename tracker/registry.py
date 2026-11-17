#!/usr/bin/env python3
"""Load data/data.json and present it in the shape the renderer expects.

All records live in one file — data/data.json — models and categories.
The models come exclusively from the Artificial Analysis open-weights
leaderboard; the categories define which AA metrics the page ranks on.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_FILE = os.path.join(ROOT, "data", "data.json")


def _load():
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


_D = _load()

# Raw records from the file
RAW_MODELS = _D.get("models", [])
RAW_CATEGORIES = _D.get("categories", [])

# Processed categories, sorted by display order
CATEGORIES = []
for c in sorted(RAW_CATEGORIES, key=lambda x: x.get("order", 999)):
    CATEGORIES.append({
        "id": c["id"],
        "label": c["label"],
        "metric": c["metric"],
        "ascending": c.get("ascending", False),
        "axis": c.get("axis", ""),
    })

# Model lookup by ID
MODELS = []
for m in RAW_MODELS:
    MODELS.append({
        "id": m["id"],
        "name": m["name"],
        "creator": m["creator"],
        "intelligenceIndex": m.get("intelligenceIndex"),
        "costPerTask": m.get("costPerTask"),
        "timePerTask": m.get("timePerTask"),
        "latency": m.get("latency"),
        "contextWindow": m.get("contextWindow"),
        "aaUrl": m.get("aaUrl", f"https://artificialanalysis.ai/models/{m['id']}"),
    })

MODEL_BY_ID = {m["id"]: m for m in MODELS}

# Rankings for each category: sorted list of (model_id, value) pairs
def _rank_for(category):
    """Build a ranked list for a category."""
    metric = category["metric"]
    ascending = category.get("ascending", False)

    ranked = []
    for m in MODELS:
        val = m.get(metric)
        if val is not None:
            ranked.append((m["id"], val))

    ranked.sort(key=lambda x: x[1], reverse=not ascending)
    return ranked


def rank(category_id):
    """Get the ranked list for a category."""
    cat = next((c for c in CATEGORIES if c["id"] == category_id), None)
    if not cat:
        return []
    return _rank_for(cat)
