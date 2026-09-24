#!/usr/bin/env python3
"""Load data/data.json and present it in the shape the renderer expects.

All the records live in one file - data/data.json - one entry per model, engine,
use case and issue tracker. That makes the whole page's input a single artefact
you can pull, edit and push from any machine, and it means a schema change
happens here and in tracker/validate.py rather than scattered through the
renderer.

The JSON keeps the records' original attribute names, so the only work done here
is the mechanical conversion the renderer used to rely on from Python values:
tuples back to tuples where a renderer compares or formats them, issue numbers
back to ints (JSON object keys are strings), and the per-engine matrix
assembled. Nothing validates - loading is deliberately permissive so
validate.py can report every problem at once with a useful message.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
DATA_FILE = os.path.join(DATA, "data.json")

MODALITIES = ["text", "image", "video", "audio"]
STATUSES = ["works", "degraded", "blocked", "none"]

# works/degraded/blocked/none -> the CSS verdict classes the page uses
SCLASS = {"works": "ready", "degraded": "degraded", "blocked": "blocked", "none": "unknown"}


def _load():
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


_D = _load()


def _records(section):
    """Records in file order, each remembering where it lives for error messages."""
    for rec in _D.get(section, []):
        rec["sourceFile"] = f"data/data.json ({section})"
    return _D.get(section, [])


MODEL_MODULES = _records("models")
ENGINE_MODULES = _records("engines")
USE_CASE_MODULES = _records("useCases")
ISSUE_MODULES = _records("issues")

# NEWS is the one section that is not a record a machine can re-derive: it is
# a human-curated daily rollup written into data/data.json by tracker/collect_news.py.
# Newest rollup first, and only the first one renders - the page is a snapshot,
# and a dated second copy would read as stale the moment a newer one lands.
NEWS = list(_D.get("news") or [])

# ISSUE_MODULES are the two-level records {"repo", "issues"}; the validators
# access .REPO / .ISSUES, so expose them as attributes too.
for _m in ISSUE_MODULES:
    _m["REPO"] = _m.get("repo", "")
    _m["ISSUES"] = {int(k): dict(v) for k, v in (_m.get("issues") or {}).items()}


# --------------------------------------------------------------------- engines
def _engine(m):
    e = {"id": m["ID"], "name": m["NAME"], "mods": list(m["MODALITIES"]), "fmt": m["FORMAT"],
         "surface": m["INTERFACE"], "api": m["API"], "lic": m["LICENSE"], "repo": m["REPO"],
         "what": m["WHAT"], "api_kind": "api"}
    if m.get("API_DETAIL"):
        e["api_detail"] = m["API_DETAIL"]
    return e


# Tab order comes from each engine's DISPLAY_ORDER, not from the filename, so
# adding an engine cannot silently reshuffle every card.
ENGINE_MODULES.sort(key=lambda m: (m.get("DISPLAY_ORDER", 9999), m["ID"]))
ENGINES = [_engine(m) for m in ENGINE_MODULES]
ENGINE_BY_ID = {e["id"]: e for e in ENGINES}

ENGINE_SITES = {m["ID"]: m["SITE"] for m in ENGINE_MODULES}

# Longest alias first, so "vLLM Metal" is matched before a shorter alias could
# claim part of it, and "MLX-Audio" before anything that is a prefix of it.
ENGINE_PROSE_LINKS = sorted(
    ((alias, m["ID"], m["SITE"]) for m in ENGINE_MODULES for alias in m.get("PROSE_ALIASES", [])),
    key=lambda t: -len(t[0]))

FAM = {m["ID"]: m["QUANT_FAMILY"] for m in ENGINE_MODULES}
CROSS_BY_ENGINE = {m["ID"]: list(m.get("CROSS_ISSUES", [])) for m in ENGINE_MODULES}
RELEASE_FEEDS = [dict(m["RELEASE_FEED"], engine=m["ID"])
                 for m in ENGINE_MODULES if m.get("RELEASE_FEED")]


# ---------------------------------------------------------------------- models
def _model(m):
    d = {"id": m["ID"], "mod": m["MODALITY"], "name": m["NAME"], "arch": m["ARCH"],
         "lic": m["LICENSE"], "ctx": m["CONTEXT"], "hf": m["HF"], "note": m["NOTE"],
         "srcs": [tuple(x) for x in m.get("SOURCES", [])],
         "agentic": [tuple(x) for x in m.get("SCORES", {}).get("agentic", [])],
         "coding": [tuple(x) for x in m.get("SCORES", {}).get("coding", [])],
         "w": 0.0, "est": False}
    if m.get("CONTEXT_LABEL"):
        d["ctx_label"] = m["CONTEXT_LABEL"]
    lad = m.get("LADDER") or {}
    smallest = min((r["gb"] for rungs in lad.values() for r in rungs), default=0.0)
    d["w"] = smallest
    return d


MODELS = [_model(m) for m in MODEL_MODULES]
MODEL_BY_ID = {m["id"]: m for m in MODELS}
PARAMS = {m["ID"]: m["PARAMS_B"] for m in MODEL_MODULES}
LADDERS = {m["ID"]: (m.get("LADDER") or {}) for m in MODEL_MODULES}
QUANT_SOURCES = {m["ID"]: (m.get("QUANT_SOURCES") or {}) for m in MODEL_MODULES}
KV = {m["ID"]: (m["KV"]["bytes_per_token"], m["KV"]["max_context"], m["KV"]["derivation"])
      for m in MODEL_MODULES}
BEST = {m["ID"]: m["BEST_ENGINE"] for m in MODEL_MODULES}

MATRIX = {
    m["ID"]: {eid: {"s": c["status"], "label": c["label"], "w": None, "q": None,
                    "note": c["note"], "items": list(c["issues"])}
              for eid, c in m["ENGINES"].items()}
    for m in MODEL_MODULES
}


# ------------------------------------------------------------------ use cases
USE_CASE_MODULES.sort(key=lambda m: (m.get("DISPLAY_ORDER", 9999), m["ID"]))
USE_CASES = [{"id": m["ID"], "label": m["LABEL"], "mod": m["MODALITY"], "gate": m["FIDELITY_GATE"],
              "axis": m["AXIS"], "rank": [tuple(r) for r in m["RANK"]]}
             for m in USE_CASE_MODULES]


# ---------------------------------------------------------------------- issues
EMETA = {}
for _m in ISSUE_MODULES:
    for _num, _meta in _m["ISSUES"].items():
        EMETA[f"{_m['REPO']}#{_num}"] = (_meta["severity"], _meta["headline"], _meta["why"])

PR_KEYS = set(_D.get("prKeys", []))


# ------------------------------------------------------------------- derived
def modality(m):
    return m.get("mod", "text")


def engine_order(mid):
    """Engines that could plausibly load this model, best first.

    Filtered by modality: an image model never shows a chat server, and an LLM
    never shows mflux. Without this every media card would carry a column of
    "out of scope" for every engine it has nothing to do with.
    """
    best = BEST[mid]
    rest = [e["id"] for e in ENGINES
            if e["id"] != best and e["id"] in MATRIX[mid]
            and MATRIX[mid][e["id"]].get("s") != "none"]
    return [best] + rest


REPO_LABELS = {
    "waybarrios/vllm-mlx": "vllm-mlx",
    "ml-explore/mlx-lm": "mlx-lm",
    "ggml-org/llama.cpp": "llama.cpp",
    "jundot/omlx": "oMLX",
    "antirez/ds4": "ds4",
    "ollama/ollama": "ollama",
    "lmstudio-ai/lmstudio-bug-tracker": "LM Studio",
    "vllm-project/vllm-metal": "vLLM Metal",
    "Blaizzy/mlx-video": "MLX-Video",
    "Blaizzy/mlx-audio": "MLX-Audio",
    "filipstrand/mflux": "mflux",
    "argmaxinc/DiffusionKit": "DiffusionKit",
}


def repo_label(key):
    return REPO_LABELS.get(key.split("#")[0], key.split("#")[0])
