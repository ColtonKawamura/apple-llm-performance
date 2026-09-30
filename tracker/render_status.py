#!/usr/bin/env python3
"""Render the vllm-mlx watchlist state file into a model-first status page."""
import os, re, html, hashlib, json
from urllib.parse import urlparse
# One record per file under data/, assembled by tracker/registry.py. MODELS and
# the per-engine matrix used to live in this file and in engines.py; they were
# split so two agents editing two different models never touch the same file.
from registry import (ENGINES, ENGINE_BY_ID, EMETA, MATRIX, BEST, engine_order,
                      repo_label, CROSS_BY_ENGINE, RELEASE_FEEDS, FAM,
                      LADDERS, KV, PARAMS, USE_CASES, MODELS, modality, SCLASS,
                      ENGINE_PROSE_LINKS, PR_KEYS, NEWS)
from bands import BANDS, FAM_OVERRIDE, FIDELITY_NOTES


def card_name():
    """Content-hashed social card filename; CDN caches images for 4h."""
    p = os.path.join(ASSETS, "og-card.jpg")
    if not os.path.exists(p):
        return "card.jpg"
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:10]
    return f"card-{h}.jpg"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
SITE = os.path.join(ROOT, "docs")
STATE = os.path.join(HERE, "watch-state.txt")

# key -> (severity, headline, why it matters)
META = dict(EMETA)


# works/degraded/blocked/none -> the CSS verdict classes the page already uses

CROSS = ["waybarrios/vllm-mlx#619", "waybarrios/vllm-mlx#584", "waybarrios/vllm-mlx#672",
         "waybarrios/vllm-mlx#546", "waybarrios/vllm-mlx#627", "waybarrios/vllm-mlx#682",
         "waybarrios/vllm-mlx#732", "waybarrios/vllm-mlx#570"]

SEV_LABEL = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}
SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def read_state():
    rows, releases = {}, {}
    if not os.path.exists(STATE):
        return rows, releases
    for line in open(STATE):
        line = line.strip()
        if not line or "|" not in line:
            continue
        key, state, _label = (line.split("|", 2) + ["", ""])[:3]
        if key.endswith("@release"):
            releases[key[:-len("@release")]] = state
        else:
            rows[key] = state
    return rows, releases


def pill(state):
    s = (state or "open").lower()
    if s == "merged":
        return "merged", "Merged"
    if s == "closed":
        return "closed", "Closed"
    return "open", "Open"


def issue_url(key):
    repo, num = key.split("#")
    kind = "pull" if key in PR_KEYS else "issues"
    return f"https://github.com/{repo}/{kind}/{num}"


def slug(name):
    return "m-" + "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")


def best_cell(m):
    """The engine a model card opens on."""
    eid = BEST[m["id"]]
    return eid, MATRIX[m["id"]][eid]


def fam_for(eid, mid):
    return FAM_OVERRIDE.get((eid, mid), FAM[eid])


def ladder_for(eid, mid):
    """Measured rungs this engine can load for this model, largest first."""
    if MATRIX[mid][eid]["s"] == "none":
        return []
    return LADDERS.get(mid, {}).get(fam_for(eid, mid), [])


# Packagers that split a GGUF over SPLIT_GB with llama-gguf-split and file the
# shards under a folder named for the quant. Everyone else here publishes one
# file at the repo root however large it gets - antirez ships a 464 GB
# DeepSeek-V4-Pro in a single blob.
SPLIT_PACKAGERS = ("unsloth/",)
SPLIT_GB = 50.0


def rung_url(fam, r):
    """Where to send someone who has just been told to run this rung.

    An MLX repo holds exactly one precision, so the repo *is* the build. A GGUF
    repo holds every quant of a model at once - unsloth/Qwen3.8-27B-GGUF carries
    27 of them - so linking the repo drops the reader in a file list and leaves
    them to work out which of those files the page just recommended.

    The rung's label is the GGUF's filename, so a single-file quant links to the
    blob. A split quant has no single file to link, and its shards live in a
    folder named for the quant suffix, so that folder is the link instead.
    Checked against all 143 gguf rungs in data/models on 2026-08-29: 75 single
    files, 68 folders, nothing unresolved.
    """
    base = f"https://huggingface.co/{r['repo']}"
    if fam not in ("gguf", "ds4"):
        return base
    if r["gb"] > SPLIT_GB and r["repo"].startswith(SPLIT_PACKAGERS):
        stem = re.sub(r"-GGUF$", "", r["repo"].split("/")[-1], flags=re.I)
        if not r["label"].startswith(stem + "-"):
            return base
        return f"{base}/tree/main/{r['label'][len(stem) + 1:]}"
    return f"{base}/blob/main/{r['label']}.gguf"


def rungs_with_urls(fam, lad):
    """Rungs as the browser sees them: the repo is resolved to the rung's own
    URL here rather than shipped raw, because the page needs somewhere to send
    the reader and the repo on its own is not it."""
    out = []
    for r in lad:
        d = dict(r, url=rung_url(fam, r))
        d.pop("repo", None)
        out.append(d)
    return out


def engine_payload(m):
    """Every engine that can run this model, with its own quant ladder.

    Ordered by architecture support first, then by whether it is the
    recommended engine - the browser walks this and takes the first entry with a
    rung that fits, so preference only breaks ties between equally-supported
    engines.
    """
    mid = m["id"]
    best = BEST[mid]
    rank = {"ready": 0, "degraded": 1, "blocked": 2, "unknown": 3}
    out = []
    for eid in engine_order(mid):
        c = MATRIX[mid][eid]
        lad = ladder_for(eid, mid)
        if not lad:
            continue
        out.append({"id": eid, "name": ENGINE_BY_ID[eid]["name"], "s": SCLASS[c["s"]],
                    "label": c["label"], "fam": fam_for(eid, mid),
                    "note": FIDELITY_NOTES.get((mid, fam_for(eid, mid)), ""),
                    "ladder": rungs_with_urls(fam_for(eid, mid), lad)})
    out.sort(key=lambda d: (rank[d["s"]], 0 if d["id"] == best else 1,
                            -(d["ladder"][-1]["gb"] if d["ladder"] else 0)))
    return out


def model_payload(m):
    bpt, maxctx, why = KV.get(m["id"], (None, None, ""))
    return {"engines": engine_payload(m),
            "kv": {"bpt": bpt, "maxctx": maxctx, "why": why},
            "params": PARAMS.get(m["id"])}


def index_rows(rows):
    """One line per model. The size, engine and headroom cells are all filled in
    by the browser once a cluster is selected - the server-rendered values are
    just the default-cluster answer so the page is not blank without JS."""
    # One lane per job, server-rendered from USE_CASES in display order - the
    # same order the chips are built in. A frozen list here would silently drop
    # a lane when a category is added.
    lanes = "".join(f'<span class="ix-lane uc-slot-{u["id"]}"><i></i></span>'
                    for u in USE_CASES)
    out = []
    for m in sorted(MODELS, key=lambda m: m["name"].casefold()):
        eid, c = best_cell(m)
        lad = ladder_for(eid, m["id"])
        gb = lad[-1]["gb"] if lad else m["w"]
        payload = html.escape(json.dumps(model_payload(m)), quote=True)
        out.append(f"""
        <a class="ix-row v-{SCLASS[c['s']]}" href="#{m['id']}" data-model="{m['id']}"
           data-sw="{SCLASS[c['s']]}" data-swlabel="{html.escape(c['label'])}"
           data-mod="{modality(m)}" data-payload="{payload}">
          <span class="ix-rank" aria-hidden="true"></span>
          <span class="ix-id"><span class="ix-name"><em>{html.escape(m['name'])}</em></span>
          <span class="ix-bars" aria-hidden="true">{lanes}<span class="ix-lane ix-lane-avg"><i></i></span></span></span>
          {support_strip(m)}
          <span class="ix-verdict"><span class="ix-status v-{SCLASS[c['s']]}">{html.escape(c['label'])}</span>
          <span class="ix-eng">{html.escape(ENGINE_BY_ID[eid]['name'])}</span></span>
          <span class="ix-fit"><span class="ix-fit-top"><span class="ix-size">{gb:.0f} GB</span>
          <span class="ix-meta fit"></span></span>
          <span class="ix-meter" aria-hidden="true"><i></i></span></span>
        </a>""")
    return "".join(out)


def support_strip(m):
    """One square per engine of the model's modality, coloured by its status -
    a caniuse-style support row. Fixed engine order so the columns line up
    down the list; an engine with no cell for this model shows as empty."""
    mid, mod = m["id"], modality(m)
    cells = []
    for e in ENGINES:
        if mod not in e["mods"]:
            continue
        c = MATRIX[mid].get(e["id"])
        sc = SCLASS[c["s"]] if c else "unknown"
        tip = f"{e['name']}: {c['label'] if c and c['s'] != 'none' else 'not supported'}"
        cells.append(f'<i class="sq s-{sc}" title="{html.escape(tip, quote=True)}"></i>')
    return f'<span class="ix-strip" role="img" aria-label="Engine support">{"".join(cells)}</span>'


def src_links(m):
    out = [f"""<a class="src" href="https://huggingface.co/{m['hf']}" target="_blank" rel="noopener">{html.escape(m['hf'])}</a>"""]
    out += [f"""<a class="src" href="{u}" target="_blank" rel="noopener">{html.escape(lbl)}</a>""" for lbl, u in m["srcs"]]
    return "".join(out)


def scores(pairs):
    """Score rows with a bar behind every figure that sits on a 0-100 scale.
    An Elo or a note has no published maximum, so it gets the figure alone."""
    out = []
    for k, v in pairs:
        mm = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*%?\s*$", v)
        w = float(mm.group(1)) if mm else None
        bar = (f'<span class="score-bar"><i style="width:{max(0.0, min(100.0, w)):.1f}%"></i></span>'
               if w is not None and 0 <= w <= 100 else '<span class="score-bar none"></span>')
        out.append(f"""<div class="score"><span class="score-k">{html.escape(k)}</span>"""
                   f"""{bar}<span class="score-v">{html.escape(v)}</span></div>""")
    return "".join(out)


# Text that must not be linkified: an existing anchor, a code span, or any tag.
_SKIP = re.compile(r"(<a\b[^>]*>.*?</a>|<code>.*?</code>|<[^>]+>)", re.S)

# An engine name is only a mention if it stands alone. The lookarounds keep
# "ds4-server" and "mlx-lm.server" from being clipped mid-token.
_ENGINE_MENTIONS = [
    (re.compile(rf"(?<![\w.-]){re.escape(alias)}(?![\w-])"), eid, site)
    for alias, eid, site in ENGINE_PROSE_LINKS]


def link_engines(out):
    """Link the first mention of each engine to its own website.

    First mention only: the notes name an engine repeatedly, and linking every
    occurrence turns a paragraph into a wall of blue. Existing links, code spans
    and tag interiors are left alone, so a hand-written [text](url) always wins.
    """
    seen = set()
    parts = _SKIP.split(out)
    for i, part in enumerate(parts):
        if i % 2:                     # the captured skip-group; leave verbatim
            continue
        for pat, eid, site in _ENGINE_MENTIONS:
            if eid in seen:
                continue
            new_part, n = pat.subn(
                lambda m: f'<a href="{site}" target="_blank" rel="noopener">{m.group(0)}</a>',
                part, count=1)
            if n:
                seen.add(eid)
                part = new_part
        parts[i] = part
    return "".join(parts)


def prose(text, engines=True):
    """HTML-escape, then render inline `code` spans, [text](url) links, and
    link the first mention of each inference engine to its own website."""
    out = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", html.escape(text))
    out = re.sub(r"\*\*([^*]+)\*\*", lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
                 lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener">{m.group(1)}</a>',
                 out)
    return link_engines(out) if engines else out


API_ROWS = [("endpoints", "Endpoints"), ("streaming", "Streaming"), ("tools", "Tool calling"),
            ("structured", "Structured output"), ("concurrency", "Concurrency"), ("gotcha", "Watch for")]


def api_block(e):
    api = e.get("api_detail")
    if not api:
        return ""
    rows = "".join(
        f"""<div class="api-row"><dt>{html.escape(label)}</dt><dd>{prose(api[k])}</dd></div>"""
        for k, label in API_ROWS if api.get(k))
    return f"""<dl class="api">{rows}</dl>"""


def engine_build(mid, eid):
    """The chosen rung is filled in by the browser; this is the static fallback.

    The fallback names the smallest rung, which is the same default the glance
    rows render, so a reader without JS still gets a real link to real weights
    rather than an empty anchor pointing at "#".
    """
    lad = ladder_for(eid, mid)
    if not lad:
        c = MATRIX[mid][eid]
        label = c["q"][0] if c.get("q") else "no build published for this engine"
        return f"""<div class="eng-build none"><span class="eng-build-k">Build</span><span>{html.escape(label)}</span></div>"""
    r = lad[-1]
    return (f"""<div class="eng-build"><span class="eng-build-k">Build</span>"""
            f"""<a class="build-link" href="{html.escape(rung_url(fam_for(eid, mid), r), quote=True)}" """
            f"""target="_blank" rel="noopener">{html.escape(r['label'])}</a>"""
            f"""<span class="build-bpw"></span></div>""")


def engine_meta_line(e):
    bits = [("Interface", e["surface"]), ("Format", e["fmt"]), ("API", e["api"]), ("License", e["lic"])]
    return "".join(f"""<div><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>""" for k, v in bits)


def engine_tabs(m, rows):
    mid, order = m["id"], engine_order(m["id"])
    tabs, panes = [], []
    for i, eid in enumerate(order):
        c, e = MATRIX[mid][eid], ENGINE_BY_ID[eid]
        sc = SCLASS[c["s"]]
        sel = "true" if i == 0 else "false"
        n_open = sum(1 for k in c["items"] if rows.get(k, "open").lower() == "open")
        tabs.append(f"""
          <button type="button" role="tab" class="eng-tab s-{sc}" data-eng="{eid}"
                  aria-selected="{sel}" aria-controls="{mid}-{eid}" id="{mid}-{eid}-tab">
            <span class="eng-tab-n">{html.escape(e['name'])}</span>
            <span class="eng-tab-s s-{sc}">{html.escape(c['label'])}</span>
          </button>""")
        lad = ladder_for(eid, mid)
        w_attr = ' data-has-ladder="1"' if lad else ""
        items = render_items(c["items"], rows)
        body = (f"""<ul class="rows">{items}\n          </ul>"""
                if items else
                """<p class="eng-clear">Nothing open tracked against this engine for this model.</p>""")
        # A blocked engine cannot load the model at all, so quoting a build, a
        # resident size, a fidelity band or a context table for it is noise at
        # best and misleading at worst. Leave the reason and the issues.
        sizing = "" if c["s"] in ("blocked", "none") else f"""
          {engine_build(mid, eid)}
          <div class="eng-fit s-{sc}"{w_attr}></div>
          <div class="mem-bar" hidden aria-hidden="true"><i class="mb-w"></i><i class="mb-o"></i><i class="mb-f"></i></div>
          <div class="ladder-wrap" hidden><span class="ctx-k">Quant ladder <em>&middot; bar height = size on disk, dashed line = usable memory</em></span>
            <div class="ladder"></div></div>
          <p class="fidelity" hidden></p>
          <div class="ctx-wrap" hidden><span class="ctx-k">Concurrent contexts in the KV headroom</span>
            <div class="ctx"></div>
            <p class="ctx-why"></p></div>"""
        panes.append(f"""
        <div class="eng-pane" id="{mid}-{eid}" role="tabpanel" aria-labelledby="{mid}-{eid}-tab"
             data-eng="{eid}"{'' if i == 0 else ' hidden'}>
          <dl class="eng-meta">{engine_meta_line(e)}</dl>{sizing}
          <p class="eng-note">{prose(c['note'])}</p>
          <p class="blockers-label"><b>{n_open}</b> open <span>of {len(c['items'])} tracked on this engine</span></p>
          {body}
        </div>""")
    return f"""
      <div class="eng" data-model="{mid}">
        <div class="eng-tabs" role="tablist" aria-label="Engines for {html.escape(m['name'])}">{''.join(tabs)}
        </div>
        <div class="eng-panes">{''.join(panes)}
        </div>
      </div>"""


def cross_tabs(rows, releases):
    feeds = {f["engine"]: f for f in RELEASE_FEEDS}
    tabs, panes = [], []
    for i, e in enumerate(ENGINES):
        eid = e["id"]
        keys = CROSS_BY_ENGINE.get(eid, [])
        n_open = sum(1 for k in keys if rows.get(k, "open").lower() == "open")
        sel = "true" if i == 0 else "false"
        tabs.append(f"""
          <button type="button" role="tab" class="eng-tab" data-eng="{eid}"
                  aria-selected="{sel}" aria-controls="cross-{eid}" id="cross-{eid}-tab">
            <span class="eng-tab-n">{html.escape(e['name'])}</span>
            <span class="eng-tab-s">{n_open} open</span>
          </button>""")
        items = render_items(keys, rows)
        body = (f"""<ul class="cross-list">{items}\n          </ul>""" if items else
                """<p class="eng-clear">Nothing server-wide tracked against this engine.</p>""")
        panes.append(f"""
        <div class="eng-pane" id="cross-{eid}" role="tabpanel" aria-labelledby="cross-{eid}-tab"
             data-eng="{eid}"{'' if i == 0 else ' hidden'}>
          <dl class="eng-meta">{engine_meta_line(e)}{release_cell(feeds.get(eid), releases)}</dl>
          <p class="eng-note">{prose(e['what'])}</p>
          {api_block(e)}
          {body}
        </div>""")
    return f"""
      <div class="eng" data-model="cross">
        <div class="eng-tabs" role="tablist" aria-label="Engines">{''.join(tabs)}
        </div>
        <div class="eng-panes">{''.join(panes)}
        </div>
      </div>"""


def release_cell(feed, releases):
    """The engine's latest release, shown alongside its other facts."""
    if not feed:
        return ""
    if feed["scheme"] == "none":
        tag = "no tag feed"
    else:
        tag = releases.get(feed["repo"], "not seen yet")
    note = f" &middot; {html.escape(feed['note'])}" if feed["note"] else ""
    return (f"""<div><dt>Latest release</dt><dd>{html.escape(tag)}"""
            f"""<span class="rel-note">{note}</span></dd></div>""")


def release_rows(releases):
    out = []
    for f in RELEASE_FEEDS:
        e = ENGINE_BY_ID[f["engine"]]
        repo = f["repo"]
        name = (f"""<a href="https://github.com/{repo}" target="_blank" rel="noopener">{html.escape(e['name'])}</a>"""
                if repo else html.escape(e["name"]))
        tag = "no tag feed" if f["scheme"] == "none" else releases.get(repo, "not seen yet")
        note = f"""<span class="rel-note">{html.escape(f['note'])}</span>""" if f["note"] else ""
        out.append(f"""<div class="rel"><span class="rel-repo">{name}{note}</span>"""
                   f"""<span class="rel-tag">{html.escape(tag)}</span></div>""")
    return "".join(out)


def _news_item(title, summary, url, n=None):
    num = f'<span class="news-n">{n:02d}</span>' if n else ""
    host = re.sub(r"^www\.", "", urlparse(url).netloc)
    return f"""
        <li class="news-item">
          <a class="news-card" href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">
            {num}<span class="news-host">{html.escape(host)}</span>
            <h4>{html.escape(title)}</h4>
            <p>{html.escape(summary)}</p>
          </a>
        </li>"""


def _news_sublist(rows, empty):
    if rows:
        return f"""<ul class="news-list news-new">{"".join(rows)}</ul>"""
    return f"""<p class="news-empty">{html.escape(empty)}</p>"""


def news_panel():
    """The daily community snapshot.

    Rendered from the newest rollup in data/data.json (the `news` section,
    written by tracker/collect_news.py). One file, one rollup at a time - the
    reader gets today's picture without opening Reddit or a lab blog, and the
    date in the corner says how fresh the picture is.
    """
    if not NEWS:
        return ""
    n = NEWS[0]
    date = n.get("date", "")
    top = n.get("top") or []
    top_html = "".join(_news_item(t, s, u, i + 1) for i, (t, s, u) in enumerate(top))
    new = n.get("newModels", []) or []
    new_html = _news_sublist(
        [_news_item(t, s, u) for t, s, u in new],
        "Nothing new shipped in the last day. The rollup re-checks on the next update.")
    return f"""
  <section class="panel" id="news">
    <h2>News <span class="news-date">rollup of {html.escape(date)}</span></h2>
    <p class="panel-lead">What shipped, broke and moved on Apple silicon today &mdash; from
    r/LocalLLaMA, Hugging Face, the lab blogs and the engine release feeds.</p>
    <h3 class="news-h">Top stories</h3>
    <ul class="news-list news-top">{top_html}</ul>
    <h3 class="news-h">New language models for Apple silicon</h3>
    {new_html}
  </section>"""


def score_block(m):
    cols = [(cat, m[k]) for cat, k in (("Agentic", "agentic"), ("Coding", "coding")) if m[k]]
    if not cols:
        return ""
    body = "".join(f"""<div class="score-col"><span class="score-cat">{cat}</span>{scores(p)}</div>"""
                   for cat, p in cols)
    return f"""<div class="scores-wrap"><span class="q-cat">Benchmark scores</span>
          <div class="scores">{body}</div></div>"""


def render_items(keys, rows):
    present = [k for k in keys if k in META]
    present.sort(key=lambda k: (SEV_ORDER.get(META[k][0], 9), k))
    out = []
    for key in present:
        sev, headline, why = META[key]
        state = rows.get(key, "open")
        cls, txt = pill(state)
        repo, num = key.split("#")
        short = repo_label(key)
        out.append(f"""
        <li class="row sev-{sev}">
          <div class="row-head">
            <a class="ref" href="{issue_url(key)}" target="_blank" rel="noopener">{short}&thinsp;#{num}</a>
            <span class="pill {cls}">{txt}</span>
            <span class="sev-tag">{SEV_LABEL[sev]}</span>
          </div>
          <details class="row-body"><summary><h4>{html.escape(headline)}</h4></summary>
          <p>{prose(why)}</p></details>
        </li>""")
    return "".join(out)


def render():
    rows, releases = read_state()

    cards = []
    for m in MODELS:
        eid, c = best_cell(m)
        sc = SCLASS[c["s"]]
        payload = html.escape(json.dumps(model_payload(m)), quote=True)
        cards.append(f"""
    <section class="model v-{sc}" id="card-{m['id']}" data-model="{m['id']}" data-sw="{sc}"
             data-swlabel="{html.escape(c['label'])}" data-mod="{modality(m)}" data-payload="{payload}">
      <div class="model-head">
        <div class="model-id">
          <h2>{html.escape(m['name'])}</h2>
          <span class="verdict v-{sc}">{html.escape(c['label'])}</span>
        </div>
        <dl class="spec">
          <div><dt>Architecture</dt><dd>{html.escape(m['arch'])}</dd></div>
          <div><dt>License</dt><dd>{html.escape(m['lic'])}</dd></div>
          <div><dt>{html.escape(m.get('ctx_label', 'Context'))}</dt><dd>{html.escape(m['ctx'])}</dd></div>
        </dl>
        <div class="model-gauge" hidden aria-hidden="true">
          <div class="mem-bar"><i class="mb-w"></i><i class="mb-o"></i><i class="mb-f"></i></div>
          <div class="mg-key"><span class="k-w">weights</span><span class="k-o">overhead</span><span class="k-f">free for KV</span></div>
        </div>
        <p class="model-fit"></p>
        <p class="model-note">{prose(m['note'])}</p>
        <div class="srcs"><span class="q-cat">Sources</span>{src_links(m)}</div>
        {score_block(m)}
      </div>
      {engine_tabs(m, rows)}
    </section>""")

    # Serialised here rather than baked into TEMPLATE: a frozen literal silently
    # went stale once already when new models were added to USE_CASES.
    usecases = json.dumps([{"id": u["id"], "label": u["label"], "gate": u["gate"],
                            "axis": u["axis"], "mod": u.get("mod", "text"),
                            "rank": [[r[0], r[1], r[2]] for r in u["rank"]]}
                           for u in USE_CASES])
    bands = json.dumps([[b[0], b[1], b[2], b[3]] for b in BANDS])

    doc = TEMPLATE.format(style=STYLE, usecases=usecases, bands=bands,
                          cards="".join(cards), index=index_rows(rows),
                          cross=cross_tabs(rows, releases), news=news_panel())
    return doc.replace("/apple-llm-performance/card.jpg",
                       "/apple-llm-performance/" + card_name())


STYLE = r"""
  /* Visual-first dashboard: a blueprint grid, bento tiles, and every number
     that can be a shape drawn as one. Inspired by caniuse's support grid,
     Grafana stat and gauge panels, and leaderboard layouts. Plain CSS, no
     assets. Kept outside the str.format template so braces need no doubling. */
  :root {
    --bg: #e1e2e7; --bg-grid: rgba(55, 96, 191, .06);
    --panel: #ffffff; --panel-2: #eef0f6; --panel-3: #e0e3ee;
    --ink: #1f2335; --ink-2: #3760bf; --muted: #6172b0;
    --line: #c4c8da; --line-soft: #d8dbe8;
    --accent: #7847bd; --accent-2: #007197; --accent-ink: #ffffff;
    --lime: #9854f1; --cyan: #007197;
    --critical: #f52a65; --high: #b15c00; --medium: #6172b0; --low: #a8aecb;
    --ok: #118c74; --ok-tint: rgba(17, 140, 116, .09); --warn: #b15c00;
    --bar-avg: #16a34a; --bar-score-a: #0e7490; --bar-score-b: #84cc16;
    --bar-agentic: #ea580c; --bar-coding: #2563eb; --bar-terminal: #7c3aed;
    --bar-computer: #0891b2; --bar-concurrency: #d97706; --bar-longctx: #ca8a04;
    --bar-image: #db2777; --bar-video: #c026d3; --bar-voice: #0d9488;
    --bar-narration: #be123c; --bar-music: #4f46e5;
    --shadow: 0 1px 0 rgba(14, 18, 28, .04), 0 8px 24px -12px rgba(14, 18, 28, .18);
    --r: 18px; --r-sm: 10px;
    --f-disp: "Space Grotesk", "Inter", system-ui, sans-serif;
    --f-body: "Inter", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    --f-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #1a1b26; --bg-grid: rgba(122, 162, 247, .05);
      --panel: #1f2335; --panel-2: #24283b; --panel-3: #292e42;
      --ink: #c0caf5; --ink-2: #a9b1d6; --muted: #565f89;
      --line: #3b4261; --line-soft: #292e42;
      --accent: #bb9af7; --accent-2: #7dcfff; --accent-ink: #1a1b26;
      --lime: #bb9af7; --cyan: #2ac3de;
      --critical: #f7768e; --high: #ff9e64; --medium: #737aa2; --low: #565f89;
      --ok: #73daca; --ok-tint: rgba(115, 218, 202, .08); --warn: #ff9e64;
      --bar-avg: #3ee07f; --bar-score-a: #4fd8f0; --bar-score-b: #c5f24a;
      --bar-agentic: #ffa94d; --bar-coding: #6ea8fe; --bar-terminal: #b39dfa;
      --bar-computer: #38e1f5; --bar-concurrency: #f7c948; --bar-longctx: #f5d95a;
      --bar-image: #f783bf; --bar-video: #ea86fa; --bar-voice: #3fe0c5;
      --bar-narration: #ff8fa3; --bar-music: #93a0fb;
      --shadow: 0 1px 0 rgba(255, 255, 255, .03) inset, 0 16px 40px -20px rgba(0, 0, 0, .8);
    }
  }
  :root[data-theme="dark"] {
    --bg: #1a1b26; --bg-grid: rgba(122, 162, 247, .05);
    --panel: #1f2335; --panel-2: #24283b; --panel-3: #292e42;
    --ink: #c0caf5; --ink-2: #a9b1d6; --muted: #565f89;
    --line: #3b4261; --line-soft: #292e42;
    --accent: #bb9af7; --accent-2: #7dcfff; --accent-ink: #1a1b26;
    --lime: #bb9af7; --cyan: #2ac3de;
    --critical: #f7768e; --high: #ff9e64; --medium: #737aa2; --low: #565f89;
    --ok: #73daca; --ok-tint: rgba(115, 218, 202, .08); --warn: #ff9e64;
    --bar-avg: #3ee07f; --bar-score-a: #4fd8f0; --bar-score-b: #c5f24a;
    --shadow: 0 1px 0 rgba(255, 255, 255, .03) inset, 0 16px 40px -20px rgba(0, 0, 0, .8);
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  html, body, .wrap { overflow-anchor: none; }
  body {
    margin: 0; color: var(--ink); background-color: var(--bg);
    background-image:
      linear-gradient(var(--bg-grid) 1px, transparent 1px),
      linear-gradient(90deg, var(--bg-grid) 1px, transparent 1px);
    background-size: 44px 44px; background-position: -1px -1px;
    font-family: var(--f-body); font-size: 15px; line-height: 1.55;
    -webkit-font-smoothing: antialiased; min-height: 100vh; overflow-x: hidden;
  }
  body::before {
    content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background: radial-gradient(70rem 30rem at 50% -8rem, color-mix(in srgb, var(--lime) 16%, transparent), transparent 70%),
                linear-gradient(180deg, transparent 40rem, var(--bg) 70rem);
  }
  a { color: var(--accent-2); }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  [hidden] { display: none !important; }
  code { font-family: var(--f-mono); font-size: .84em; background: var(--panel-3);
    border-radius: 5px; padding: .08em .35em; }
  .wrap { max-width: 74rem; margin: 0 auto; padding: 1.25rem 1.5rem 5rem; }

  /* ---------- hero ---------- */
  .fork-note { margin: 0 0 2.2rem; display: flex; flex-wrap: wrap; align-items: center; gap: .15rem .45rem;
    font-size: .78rem; color: var(--muted); }
  .fork-note a { color: var(--ink-2); }
  .fork-tag { font-family: var(--f-mono); font-size: .64rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: var(--accent-ink); background: var(--accent);
    border-radius: 999px; padding: .12rem .5rem; }
  header { display: flex; flex-direction: column; gap: .9rem; margin-bottom: 2rem; }
  .hero-top { display: flex; justify-content: flex-end; align-items: center; gap: 1rem; }
  .live-dot { width: .55rem; height: .55rem; border-radius: 50%; background: var(--ok); flex: none;
    box-shadow: 0 0 0 0 color-mix(in srgb, var(--ok) 60%, transparent); animation: ping 2.4s ease-out infinite; }
  @keyframes ping { 0% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--ok) 55%, transparent); }
    80%, 100% { box-shadow: 0 0 0 .6rem transparent; } }
  .gh { display: inline-flex; align-items: center; gap: .45rem; text-decoration: none; color: var(--ink);
    background: var(--panel); border: 1px solid var(--line); border-radius: 999px;
    padding: .38rem .85rem .38rem .7rem; font-size: .8rem; font-weight: 600; box-shadow: var(--shadow); }
  .gh:hover { border-color: var(--accent); }
  h1 { font-family: var(--f-disp); font-size: clamp(2.4rem, 7vw, 4.6rem); font-weight: 700; margin: 0;
    letter-spacing: -.045em; line-height: .95; text-wrap: balance; }
  h1 .grad { background: linear-gradient(95deg, var(--lime), var(--cyan)); -webkit-background-clip: text;
    background-clip: text; color: transparent; }
  .hero-sub { margin: 0; font-size: 1.05rem; color: var(--ink-2); max-width: 40rem; }

  /* ---------- section labels ---------- */
  .sec-k, .uc-f > .sec-k { font-family: var(--f-mono); font-size: .68rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: var(--accent); display: block; margin-bottom: .75rem; }

  /* ---------- rig ---------- */
  .rig { background: var(--panel); border: 1px solid var(--line); border-radius: var(--r);
    padding: 1.2rem 1.3rem 1.25rem; margin: 0 0 1rem; box-shadow: var(--shadow); }
  .rig-controls { display: flex; gap: .7rem; flex-wrap: wrap; align-items: flex-end; }
  .rig-f { display: flex; flex-direction: column; gap: .3rem; min-width: 0; flex: 1 1 9rem; max-width: 14rem; }
  .rig-f > span { font-size: .7rem; font-weight: 600; color: var(--muted); }
  .rig select { font: inherit; font-size: .95rem; font-weight: 600; color: var(--ink); appearance: none;
    -webkit-appearance: none; background-color: var(--panel-2);
    background-image: linear-gradient(45deg, transparent 50%, var(--muted) 50%), linear-gradient(135deg, var(--muted) 50%, transparent 50%);
    background-position: calc(100% - 16px) 55%, calc(100% - 11px) 55%; background-size: 5px 5px; background-repeat: no-repeat;
    border: 1px solid var(--line); border-radius: 12px; padding: .6rem 2rem .6rem .8rem; width: 100%; cursor: pointer; }
  .rig select:hover { border-color: var(--muted); }
  .rig select optgroup { font-weight: 600; }
  .rig select.chip-glow { box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 30%, transparent); transition: box-shadow .3s ease; }
  .rig-out { margin: 1rem 0 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr)); gap: .5rem;
    transition: opacity .18s ease; }
  .rig-out.pulse { opacity: .35; }
  .tile { display: flex; flex-direction: column; gap: .12rem; padding: .7rem .85rem; border-radius: 12px;
    background: var(--panel-2); border: 1px solid var(--line-soft); min-width: 0; }
  .tile-k { font-family: var(--f-mono); font-size: .62rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
  .tile-v { font-family: var(--f-disp); font-size: 1.35rem; font-weight: 700; letter-spacing: -.03em;
    line-height: 1.15; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .tile-s { font-size: .72rem; color: var(--muted); }
  .tile-hi { background: color-mix(in srgb, var(--accent) 12%, var(--panel)); border-color: color-mix(in srgb, var(--accent) 45%, transparent); }
  .tile-hi .tile-v { color: var(--accent); }
  .rig-gauge { margin-top: .8rem; }
  .rg-bar { display: flex; gap: 4px; height: 14px; }
  .rg-seg { flex: 1; border-radius: 4px; overflow: hidden; position: relative;
    background: repeating-linear-gradient(135deg, var(--panel-3) 0 4px, transparent 4px 8px); border: 1px solid var(--line); }
  .rg-seg i { position: absolute; left: 0; top: 0; bottom: 0; background: linear-gradient(90deg, var(--lime), var(--cyan)); }
  .rg-key { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: .4rem; font-family: var(--f-mono); font-size: .66rem; color: var(--muted); }
  .rg-key span::before, .mg-key span::before { content: ""; display: inline-block; width: .6rem; height: .6rem;
    border-radius: 2px; margin-right: .35rem; vertical-align: -.05em; }
  .rg-key .k-u::before { background: linear-gradient(90deg, var(--lime), var(--cyan)); }
  .rg-key .k-r::before { background: repeating-linear-gradient(135deg, var(--muted) 0 2px, transparent 2px 4px); }
  .rig-warn { margin: .8rem 0 0; font-size: .82rem; color: var(--critical); padding: .55rem .75rem; border-radius: 10px;
    background: color-mix(in srgb, var(--critical) 9%, transparent); border: 1px solid color-mix(in srgb, var(--critical) 35%, transparent); }

  /* ---------- use-case chips ---------- */
  .index { margin: 0 0 2.5rem; background: var(--panel); border: 1px solid var(--line); border-radius: var(--r);
    padding: 1.2rem 1.3rem 1rem; box-shadow: var(--shadow); }
  .ix-top { margin-bottom: .8rem; }
  .uc-f { display: flex; flex-direction: column; }
  .uc-list { display: flex; flex-wrap: wrap; gap: .4rem; margin: 0; padding: 0; }
  .uc-chip { margin: 0; display: inline-flex; align-items: center; gap: .45rem; font: inherit; font-size: .82rem;
    font-weight: 600; color: var(--ink-2); cursor: pointer; background: var(--panel-2);
    border: 1px solid var(--line); border-radius: 10px; padding: .45rem .8rem .45rem .6rem;
    transition: border-color .12s ease, background .12s ease, color .12s ease; }
  .uc-chip:hover { border-color: var(--muted); color: var(--ink); }
  .uc-chip .uc-tick { display: none; }
  .uc-chip .uc-dot { width: .8rem; height: .8rem; border-radius: 4px; flex: none; background: var(--low);
    box-shadow: inset 0 0 0 2px var(--panel-2); border: 2px solid var(--low); }
  .uc-chip[aria-pressed="true"] { color: var(--ink); }
  .uc-chip[aria-pressed="true"] .uc-dot { box-shadow: none; }
  .uc-chip[data-uc="agentic"] { --c: var(--bar-agentic); }
  .uc-chip[data-uc="coding"] { --c: var(--bar-coding); }
  .uc-chip[data-uc="terminal"] { --c: var(--bar-terminal); }
  .uc-chip[data-uc="computer"] { --c: var(--bar-computer); }
  .uc-chip[data-uc="concurrency"] { --c: var(--bar-concurrency); }
  .uc-chip[data-uc="longctx"] { --c: var(--bar-longctx); }
  .uc-chip[data-uc="image"] { --c: var(--bar-image); }
  .uc-chip[data-uc="video"] { --c: var(--bar-video); }
  .uc-chip[data-uc="voice"] { --c: var(--bar-voice); }
  .uc-chip[data-uc="narration"] { --c: var(--bar-narration); }
  .uc-chip[data-uc="music"] { --c: var(--bar-music); }
  .uc-chip .uc-dot { background: var(--c, var(--low)); border-color: var(--c, var(--low)); }
  .uc-chip[aria-pressed="true"] { border-color: var(--c, var(--ok));
    background: color-mix(in srgb, var(--c, var(--ok)) 14%, var(--panel));
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--c, var(--ok)) 16%, transparent); }
  .uc-legend { display: none; flex-wrap: wrap; align-items: center; gap: .3rem .55rem; margin: 0 0 .6rem; }
  .uc-legend.on { display: flex; }
  .lg-it { display: inline-flex; align-items: center; gap: .35rem; }
  .lg-swatch { width: 16px; height: 6px; border-radius: 3px; background: var(--low); flex: none; }
  .lg-swatch-avg { width: 24px; height: 6px; border-radius: 3px; background: var(--bar-avg); flex: none; }
  .lg-arr { font-family: var(--f-mono); color: var(--muted); }
  .lg-word { font-size: .74rem; color: var(--muted); }
  .uc-legend .uc-slot-agentic { background: var(--bar-agentic); }
  .uc-legend .uc-slot-coding { background: var(--bar-coding); }
  .uc-legend .uc-slot-terminal { background: var(--bar-terminal); }
  .uc-legend .uc-slot-computer { background: var(--bar-computer); }
  .uc-legend .uc-slot-concurrency { background: var(--bar-concurrency); }
  .uc-legend .uc-slot-longctx { background: var(--bar-longctx); }
  .uc-legend .uc-slot-image { background: var(--bar-image); }
  .uc-legend .uc-slot-video { background: var(--bar-video); }
  .uc-legend .uc-slot-voice { background: var(--bar-voice); }
  .uc-legend .uc-slot-narration { background: var(--bar-narration); }
  .uc-legend .uc-slot-music { background: var(--bar-music); }
  .uc-out { margin: 0 0 .9rem; font-size: .88rem; color: var(--ink-2); }
  .uc-out:not(:empty) { padding: .85rem 1rem .85rem 3.1rem; border-radius: 14px; position: relative;
    background: color-mix(in srgb, var(--ok) 8%, var(--panel-2)); border: 1px solid color-mix(in srgb, var(--ok) 35%, transparent); }
  .uc-out:not(:empty)::before { content: "\2605"; position: absolute; left: .85rem; top: .7rem;
    width: 1.6rem; height: 1.6rem; display: grid; place-items: center; border-radius: 8px;
    background: var(--ok); color: var(--bg); font-size: .9rem; }
  .uc-out strong { color: var(--ink); font-weight: 700; }
  .uc-out .uc-why { display: block; margin-top: .3rem; font-size: .78rem; color: var(--muted); }

  /* ---------- model list ---------- */
  .ix-head, .ix-row { display: grid; align-items: center; gap: 1rem;
    grid-template-columns: 2rem minmax(0, 1fr) 7.6rem 10.5rem 13rem; }
  .ix-head { padding: .2rem .9rem .45rem; font-family: var(--f-mono); font-size: .6rem; letter-spacing: .1em;
    text-transform: uppercase; color: var(--muted); }
  .ix-head span:last-child { text-align: left; }
  .ix-rows { display: flex; flex-direction: column; gap: 4px; }
  .ix-row { padding: .7rem .9rem; background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 12px;
    text-decoration: none; color: inherit; position: relative;
    transition: background .15s ease, border-color .15s ease, transform .15s ease; }
  .ix-row:hover, .ix-row:focus-visible { background: var(--panel-3); border-color: var(--line); transform: translateY(-1px); }
  .ix-row.uc-out-of-scope { opacity: .45; }
  .ix-row.v-toolarge { opacity: .7; }
  .ix-rank { font-family: var(--f-disp); font-weight: 700; font-size: 1rem; color: var(--muted);
    font-variant-numeric: tabular-nums; text-align: center; display: grid; place-items: center; height: 2rem;
    border-radius: 8px; }
  .ix-rank:empty::before { content: ""; width: .6rem; height: .6rem; border-radius: 50%; background: var(--low); }
  .ix-row.v-ready .ix-rank:empty::before { background: var(--ok); box-shadow: 0 0 10px color-mix(in srgb, var(--ok) 60%, transparent); }
  .ix-row.v-degraded .ix-rank:empty::before { background: var(--warn); }
  .ix-row.v-blocked .ix-rank:empty::before { background: var(--critical); }
  .ix-row[data-rank="1"] .ix-rank { background: var(--accent); color: var(--accent-ink); }
  .ix-row[data-rank="2"] .ix-rank, .ix-row[data-rank="3"] .ix-rank { background: var(--panel); color: var(--ink);
    box-shadow: inset 0 0 0 1px var(--line); }
  .ix-row.uc-best { background: color-mix(in srgb, var(--accent) 9%, var(--panel-2));
    border-color: color-mix(in srgb, var(--accent) 55%, transparent); }
  .ix-row.uc-best .ix-name::after { content: "best"; margin-left: .5rem; font-family: var(--f-mono); font-size: .58rem;
    font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--accent-ink); background: var(--accent);
    border-radius: 999px; padding: .1em .5em; }
  .ix-id { display: flex; flex-direction: column; justify-content: center; gap: .35rem; min-width: 0; }
  .ix-name { font-weight: 600; font-size: .95rem; letter-spacing: -.01em; display: flex; align-items: center; min-width: 0; }
  .ix-name em { font-style: normal; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ix-row.no-bar .ix-name em::after { content: "\2020"; margin-left: .3rem; font-size: .8em; color: var(--muted); vertical-align: super; }
  .ix-bars { display: none; }
  .ix-row.j1 .ix-bars, .ix-row.multi .ix-bars { display: flex; flex-direction: column; gap: 3px; }
  .ix-lane { display: none; position: relative; height: 6px; background: var(--panel-3); border-radius: 3px; overflow: hidden; }
  .ix-lane.on { display: block; }
  .ix-lane > i { position: absolute; left: 0; top: 0; bottom: 0; width: 0; border-radius: 3px;
    transition: width .45s cubic-bezier(.2, .7, .3, 1); }
  .ix-lane-avg > i { background: var(--bar-avg); }
  .uc-slot-agentic > i { background: var(--bar-agentic); }
  .uc-slot-coding > i { background: var(--bar-coding); }
  .uc-slot-terminal > i { background: var(--bar-terminal); }
  .uc-slot-computer > i { background: var(--bar-computer); }
  .uc-slot-concurrency > i { background: var(--bar-concurrency); }
  .uc-slot-longctx > i { background: var(--bar-longctx); }
  .uc-slot-image > i { background: var(--bar-image); }
  .uc-slot-video > i { background: var(--bar-video); }
  .uc-slot-voice > i { background: var(--bar-voice); }
  .uc-slot-narration > i { background: var(--bar-narration); }
  .uc-slot-music > i { background: var(--bar-music); }
  .ix-strip { display: flex; gap: 3px; flex-wrap: wrap; }
  .sq { display: inline-block; width: 12px; height: 12px; border-radius: 3px; background: var(--panel-3);
    box-shadow: inset 0 0 0 1px var(--line); }
  .sq.s-ready { background: var(--ok); box-shadow: none; }
  .sq.s-degraded { background: var(--warn); box-shadow: none; }
  .sq.s-blocked { background: var(--critical); box-shadow: none; }
  .ix-verdict { display: flex; flex-direction: column; gap: .2rem; min-width: 0; }
  .ix-status { font-size: .74rem; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    display: inline-flex; align-items: center; gap: .35rem; color: var(--ink-2); }
  .ix-status::before { content: ""; width: .5rem; height: .5rem; border-radius: 50%; background: var(--low); flex: none; }
  .ix-status.v-ready { color: var(--ok); } .ix-status.v-ready::before { background: var(--ok); }
  .ix-status.v-degraded { color: var(--warn); } .ix-status.v-degraded::before { background: var(--warn); }
  .ix-status.v-blocked { color: var(--critical); } .ix-status.v-blocked::before { background: var(--critical); }
  .ix-status.v-toolarge, .ix-status.v-unknown, .ix-status.v-nofit { color: var(--muted); }
  .ix-eng { font-family: var(--f-mono); font-size: .7rem; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ix-fit { display: flex; flex-direction: column; gap: .3rem; min-width: 0; }
  .ix-fit-top { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
  .ix-size { font-family: var(--f-mono); font-size: .8rem; font-weight: 700; color: var(--ink); white-space: nowrap; }
  .ix-meta { font-family: var(--f-mono); font-size: .68rem; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ix-meter { display: block; height: 7px; border-radius: 4px; background: var(--panel-3); overflow: hidden;
    box-shadow: inset 0 0 0 1px var(--line-soft); }
  .ix-meter i { display: block; height: 100%; width: 0; border-radius: 4px; background: var(--medium);
    transition: width .45s cubic-bezier(.2, .7, .3, 1); }
  .ix-row.v-ready .ix-meter i { background: linear-gradient(90deg, var(--lime), var(--cyan)); }
  .ix-row.v-degraded .ix-meter i { background: var(--warn); }
  .ix-row.v-blocked .ix-meter i { background: var(--critical); }
  .ix-row.v-toolarge .ix-meter i { background: repeating-linear-gradient(135deg, var(--critical) 0 4px, transparent 4px 7px); }
  .ix-key { display: flex; flex-wrap: wrap; align-items: center; gap: .35rem .5rem; margin-top: .8rem;
    font-family: var(--f-mono); font-size: .66rem; color: var(--muted); }
  .ix-key .sq { width: 10px; height: 10px; margin-left: .4rem; }
  .ix-key-m { display: inline-block; width: 36px; height: 7px; border-radius: 4px; background: var(--panel-3); margin-left: .8rem; overflow: hidden; }
  .ix-key-m i { display: block; width: 60%; height: 100%; background: linear-gradient(90deg, var(--lime), var(--cyan)); }
  @media (max-width: 860px) {
    .ix-head { display: none; }
    .ix-row { grid-template-columns: 2rem minmax(0, 1fr) auto; row-gap: .5rem; column-gap: .7rem; }
    .ix-rank { grid-column: 1; grid-row: 1 / span 2; align-self: start; }
    .ix-id { grid-column: 2 / -1; grid-row: 1; }
    .ix-verdict { grid-column: 2; grid-row: 2; }
    .ix-strip { grid-column: 3; grid-row: 2; justify-content: flex-end; max-width: 7rem; }
    .ix-fit { grid-column: 2 / -1; grid-row: 3; }
  }
  @media (prefers-reduced-motion: reduce) { .ix-lane > i, .ix-meter i { transition: none; } .live-dot { animation: none; } }

  /* ---------- detail card ---------- */
  .detail { margin-bottom: 2.5rem; }
  .back { appearance: none; cursor: pointer; font: inherit; font-size: .84rem; font-weight: 600; color: var(--ink);
    background: var(--panel); border: 1px solid var(--line); border-radius: 999px; padding: .45rem 1rem .45rem .8rem;
    margin: 0 0 1rem; display: inline-flex; align-items: center; gap: .45rem; box-shadow: var(--shadow); }
  .back:hover { border-color: var(--accent); }
  .back-bottom { margin: 1.25rem 0 0; }
  .model { margin-bottom: 2.5rem; scroll-margin-top: 1rem; }
  .detail .model { margin-bottom: 0; }
  .model-head { position: relative; background: var(--panel); border: 1px solid var(--line); border-radius: var(--r) var(--r) 0 0;
    padding: 1.6rem 1.6rem 1.3rem; overflow: hidden; }
  .model-head::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px; background: var(--medium); }
  .model.v-ready .model-head::before { background: linear-gradient(90deg, var(--lime), var(--cyan)); }
  .model.v-degraded .model-head::before { background: var(--warn); }
  .model.v-blocked .model-head::before { background: var(--critical); }
  .model.v-toolarge .model-head::before, .model.v-unknown .model-head::before { background: var(--low); }
  .model-id { display: flex; align-items: center; gap: .8rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .model-id h2 { font-family: var(--f-disp); font-size: clamp(1.6rem, 4vw, 2.4rem); font-weight: 700; margin: 0; letter-spacing: -.035em; line-height: 1.05; }
  .verdict { font-size: .78rem; font-weight: 700; padding: .3rem .75rem; border-radius: 999px; color: var(--ink-2);
    background: var(--panel-3); display: inline-flex; align-items: center; gap: .4rem; }
  .verdict::before { content: ""; width: .5rem; height: .5rem; border-radius: 50%; background: currentColor; }
  .verdict.v-ready { color: var(--ok); background: color-mix(in srgb, var(--ok) 13%, transparent); }
  .verdict.v-degraded { color: var(--warn); background: color-mix(in srgb, var(--warn) 13%, transparent); }
  .verdict.v-blocked { color: var(--critical); background: color-mix(in srgb, var(--critical) 13%, transparent); }
  .verdict.v-toolarge, .verdict.v-unknown { color: var(--muted); }
  .spec { display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: .5rem; margin: 0 0 1rem; }
  .spec div { display: flex; flex-direction: column; gap: .15rem; min-width: 0; padding: .65rem .8rem;
    background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 12px; }
  .spec dt, .eng-meta dt, .api-row dt { font-family: var(--f-mono); font-size: .6rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
  .spec dd { margin: 0; font-size: .86rem; font-weight: 600; color: var(--ink); }
  .model-gauge { margin: 0 0 .5rem; }
  .mem-bar { display: flex; height: 16px; border-radius: 6px; overflow: hidden; background: var(--panel-3);
    box-shadow: inset 0 0 0 1px var(--line); }
  .mem-bar i { display: block; height: 100%; width: 0; transition: width .45s cubic-bezier(.2, .7, .3, 1); }
  .mem-bar .mb-w { background: linear-gradient(90deg, var(--lime), var(--cyan)); }
  .mem-bar .mb-o { background: repeating-linear-gradient(135deg, var(--muted) 0 3px, transparent 3px 6px); }
  .mem-bar .mb-f { background: color-mix(in srgb, var(--accent-2) 22%, transparent); }
  .mg-key { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: .4rem; font-family: var(--f-mono); font-size: .64rem; color: var(--muted); }
  .mg-key .k-w::before { background: linear-gradient(90deg, var(--lime), var(--cyan)); }
  .mg-key .k-o::before { background: repeating-linear-gradient(135deg, var(--muted) 0 2px, transparent 2px 4px); }
  .mg-key .k-f::before { background: color-mix(in srgb, var(--accent-2) 35%, transparent); }
  .model-fit { margin: .2rem 0 .9rem; font-family: var(--f-mono); font-size: .8rem; color: var(--ink); }
  .model-fit.toolarge { color: var(--critical); }
  .model-note { margin: 0 0 1rem; font-size: .92rem; color: var(--ink-2); max-width: 54rem; }
  .srcs { display: flex; align-items: center; gap: .4rem; flex-wrap: wrap; margin: 0 0 .4rem; }
  .q-cat { font-family: var(--f-mono); font-size: .6rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-right: .2rem; }
  .src { font-size: .74rem; font-weight: 500; color: var(--ink-2); text-decoration: none; border: 1px solid var(--line);
    border-radius: 999px; padding: .15rem .6rem; background: var(--panel-2); }
  .src:hover { border-color: var(--accent-2); color: var(--ink); }
  .scores-wrap { margin-top: 1.1rem; padding-top: 1rem; border-top: 1px dashed var(--line); }
  .scores { display: grid; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); gap: 1rem 2rem; margin-top: .6rem; }
  .score-col { display: flex; flex-direction: column; gap: .45rem; min-width: 0; }
  .score-cat { font-size: .78rem; font-weight: 700; color: var(--ink); }
  .score { display: grid; grid-template-columns: minmax(0, 9rem) minmax(0, 1fr) 4.2rem; align-items: center; gap: .7rem; }
  .score-k { font-size: .76rem; color: var(--ink-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .score-bar { height: 8px; border-radius: 4px; background: var(--panel-3); overflow: hidden; }
  .score-bar i { display: block; height: 100%; background: linear-gradient(90deg, var(--bar-score-a), var(--bar-score-b)); border-radius: 4px; }
  .score-bar.none { background: repeating-linear-gradient(90deg, var(--line) 0 3px, transparent 3px 7px); height: 2px; }
  .score-v { font-family: var(--f-mono); font-size: .78rem; font-weight: 700; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* engine rail */
  .eng { display: grid; grid-template-columns: 14rem 1fr; background: var(--panel); border: 1px solid var(--line);
    border-top: 0; border-radius: 0 0 var(--r) var(--r); overflow: hidden; }
  .eng-tabs { display: flex; flex-direction: column; gap: 3px; padding: .6rem; background: var(--panel-2);
    border-right: 1px solid var(--line); }
  .eng-panes { min-width: 0; }
  .eng-tab { appearance: none; border: 1px solid transparent; cursor: pointer; text-align: left; background: transparent;
    color: var(--ink-2); padding: .55rem .7rem; border-radius: 10px; display: flex; flex-direction: column; gap: .1rem; font: inherit; }
  .eng-tab:hover { background: var(--panel-3); color: var(--ink); }
  .eng-tab[aria-selected="true"] { background: var(--panel); color: var(--ink); border-color: var(--line); box-shadow: var(--shadow); }
  .eng-tab-n { font-size: .85rem; font-weight: 700; }
  .eng-tab-s { font-size: .68rem; color: var(--muted); display: inline-flex; align-items: center; gap: .35rem; }
  .eng-tab-s.s-ready::before, .eng-tab-s.s-degraded::before, .eng-tab-s.s-blocked::before, .eng-tab-s.s-unknown::before {
    content: ""; width: .5rem; height: .5rem; border-radius: 2px; flex: none; }
  .eng-tab-s.s-ready::before { background: var(--ok); }
  .eng-tab-s.s-degraded::before { background: var(--warn); }
  .eng-tab-s.s-blocked::before { background: var(--critical); }
  .eng-tab-s.s-unknown::before { background: var(--low); }
  .eng-pane { padding: 1.3rem 1.4rem 1.4rem; display: flex; flex-direction: column; gap: .9rem; min-width: 0; }
  .eng-meta { margin: 0; display: flex; flex-wrap: wrap; gap: .4rem; }
  .eng-meta div { display: flex; flex-direction: column; gap: .05rem; padding: .4rem .65rem; border-radius: 10px;
    background: var(--panel-2); border: 1px solid var(--line-soft); min-width: 0; }
  .eng-meta dd { margin: 0; font-size: .8rem; font-weight: 600; color: var(--ink); }
  .rel-note { font-weight: 400; color: var(--muted); font-size: .72rem; }
  .eng-build { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; padding: .7rem .85rem; border-radius: 12px;
    background: color-mix(in srgb, var(--accent-2) 7%, var(--panel-2)); border: 1px solid var(--line); }
  .eng-build-k { font-family: var(--f-mono); font-size: .6rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
  .eng-build a, .eng-build span:not(.eng-build-k) { font-family: var(--f-mono); font-size: .8rem; font-weight: 700; word-break: break-all; }
  .eng-build.none span:not(.eng-build-k) { color: var(--muted); font-weight: 400; }
  .build-bpw { font-size: .68rem !important; padding: .15em .55em; border-radius: 999px; background: var(--panel-3); }
  .build-bpw:empty { display: none; }
  .build-bpw.b-full { color: var(--ok); background: color-mix(in srgb, var(--ok) 13%, transparent); }
  .build-bpw.b-mild, .build-bpw.b-low, .build-bpw.b-pruned { color: var(--warn); background: color-mix(in srgb, var(--warn) 13%, transparent); }
  .build-bpw.b-unusable { color: var(--critical); background: color-mix(in srgb, var(--critical) 13%, transparent); }
  .eng-fit { font-family: var(--f-mono); font-size: .8rem; color: var(--ink); }
  .eng-fit:empty { display: none; }
  .eng-fit.toolarge { color: var(--critical); }
  .ladder-wrap, .ctx-wrap { display: flex; flex-direction: column; gap: .5rem; padding: .9rem 1rem; border-radius: 12px;
    background: var(--panel-2); border: 1px solid var(--line-soft); }
  .ctx-k { font-size: .78rem; font-weight: 700; color: var(--ink); }
  .ctx-k em { font-style: normal; font-weight: 400; font-size: .7rem; color: var(--muted); }
  .ladder { position: relative; display: flex; align-items: flex-end; gap: 3px; height: 84px;
    border-bottom: 1px solid var(--line); padding-top: 4px; }
  .lb { flex: 1 1 0; min-width: 3px; max-width: 26px; border-radius: 3px 3px 0 0; background: var(--panel-3);
    box-shadow: inset 0 0 0 1px var(--line); }
  .lb.fits.b-full { background: var(--ok); box-shadow: none; }
  .lb.fits.b-mild, .lb.fits.b-low, .lb.fits.b-pruned { background: var(--warn); box-shadow: none; }
  .lb.fits.b-unusable { background: var(--critical); box-shadow: none; }
  .lb.pick { outline: 2px solid var(--ink); outline-offset: 2px; }
  .lb-line { position: absolute; left: 0; right: 0; border-top: 1.5px dashed var(--critical); pointer-events: none; }
  .lb-line em { position: absolute; right: 0; bottom: 2px; font-style: normal; font-family: var(--f-mono); font-size: .6rem;
    color: var(--critical); background: var(--panel-2); padding: 0 .25rem; }
  .ladder-cap { margin: 0; font-family: var(--f-mono); font-size: .68rem; color: var(--muted); }
  .ladder-cap b { color: var(--ink); }
  .fidelity { margin: 0; font-size: .84rem; color: var(--ink-2); padding: .7rem .85rem; border-radius: 12px;
    background: color-mix(in srgb, var(--warn) 9%, transparent); border: 1px solid color-mix(in srgb, var(--warn) 35%, transparent); }
  .fidelity.b-unusable { background: color-mix(in srgb, var(--critical) 9%, transparent); border-color: color-mix(in srgb, var(--critical) 35%, transparent); }
  .fidelity strong { color: var(--ink); }
  .fidelity.b-unusable strong { color: var(--critical); }
  .ctx { display: flex; flex-direction: column; gap: .35rem; }
  .cx { display: grid; grid-template-columns: 11rem 4.5rem minmax(0, 1fr) 5.5rem; align-items: center; gap: .7rem;
    font-size: .78rem; color: var(--ink-2); font-variant-numeric: tabular-nums; }
  .cx-h { font-family: var(--f-mono); font-size: .58rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
  .cx-l em { font-style: normal; color: var(--muted); font-size: .7rem; }
  .cx-kv { font-family: var(--f-mono); font-size: .72rem; }
  .cx-bar { height: 10px; border-radius: 3px; background: var(--panel-3); overflow: hidden; }
  .cx-bar i { display: block; height: 100%; background: linear-gradient(90deg, var(--accent-2), var(--lime)); border-radius: 3px; }
  .ctx-n { font-family: var(--f-mono); font-weight: 700; color: var(--ink); text-align: right; }
  .ctx-n.none { color: var(--critical); font-weight: 400; font-size: .7rem; }
  .ctx-cap .ctx-n { color: var(--ok); }
  .ctx-why { margin: .2rem 0 0; font-size: .72rem; color: var(--muted); }
  .eng-note { margin: 0; font-size: .9rem; color: var(--ink-2); max-width: 56rem; }
  .eng-clear { margin: 0; font-size: .82rem; color: var(--muted); }
  .blockers-label { margin: .2rem 0 0; font-size: .78rem; color: var(--muted); display: flex; align-items: baseline; gap: .35rem; }
  .blockers-label b { font-family: var(--f-disp); font-size: 1.4rem; color: var(--ink); line-height: 1; }
  @media (max-width: 820px) {
    .eng { grid-template-columns: 1fr; }
    .eng-tabs { flex-direction: row; overflow-x: auto; border-right: 0; border-bottom: 1px solid var(--line); }
    .eng-tab { flex: 0 0 auto; }
    .cx { grid-template-columns: 7.5rem 3.8rem minmax(0, 1fr) 4.5rem; gap: .45rem; }
    .score { grid-template-columns: minmax(0, 7rem) minmax(0, 1fr) 3.8rem; }
  }

  /* issues */
  .rows, .cross-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
  .row { --sev: var(--medium); background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 12px;
    padding: .65rem .85rem .65rem 1rem; position: relative; overflow: hidden; }
  .row::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--sev); }
  .row.sev-critical { --sev: var(--critical); }
  .row.sev-high { --sev: var(--high); }
  .row.sev-medium { --sev: var(--medium); }
  .row.sev-low { --sev: var(--low); }
  .row-head { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; margin-bottom: .2rem; }
  .ref { font-family: var(--f-mono); font-size: .74rem; font-weight: 700; color: var(--accent-2); text-decoration: none; }
  .ref:hover { text-decoration: underline; }
  .pill { font-size: .64rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; padding: .12rem .5rem;
    border-radius: 999px; background: var(--panel-3); color: var(--ink-2); }
  .pill.merged { color: var(--ok); background: color-mix(in srgb, var(--ok) 14%, transparent); }
  .pill.closed { color: var(--muted); }
  .sev-tag { margin-left: auto; font-size: .64rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em;
    color: var(--sev); }
  .row-body > summary { cursor: pointer; list-style: none; display: flex; align-items: baseline; gap: .4rem; }
  .row-body > summary::-webkit-details-marker { display: none; }
  .row-body > summary::after { content: "+"; margin-left: auto; font-family: var(--f-mono); color: var(--muted); }
  .row-body[open] > summary::after { content: "\2212"; }
  .row h4 { font-size: .88rem; font-weight: 600; margin: 0; }
  .row p { margin: .35rem 0 0; font-size: .84rem; color: var(--ink-2); max-width: 54rem; }
  .row p code, .eng-note code, .model-note code { font-size: .8em; }

  /* ---------- panels ---------- */
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: var(--r); padding: 1.3rem 1.4rem;
    margin-bottom: 1rem; box-shadow: var(--shadow); }
  .panel h2 { font-family: var(--f-disp); font-size: 1.25rem; letter-spacing: -.02em; margin: 0 0 .8rem; font-weight: 700; }
  .panel-lead { margin: 0 0 1rem; font-size: .88rem; color: var(--ink-2); max-width: 56rem; }
  .panel > ul, .panel-fold > ul { margin: 0; padding-left: 1.1rem; display: flex; flex-direction: column; gap: .55rem; }
  .panel-fold > ul > li { font-size: .87rem; color: var(--ink-2); }
  .panel strong { color: var(--ink); }
  .panel-fold > summary { cursor: pointer; list-style: none; display: flex; align-items: center; gap: .6rem; }
  .panel-fold > summary::-webkit-details-marker { display: none; }
  .panel-fold > summary::before { content: "+"; display: grid; place-items: center; width: 1.6rem; height: 1.6rem;
    border-radius: 8px; background: var(--panel-3); color: var(--ink); font-family: var(--f-mono); font-weight: 700; flex: none; }
  .panel-fold[open] > summary::before { content: "\2212"; background: var(--accent); color: var(--accent-ink); }
  .panel-fold > summary h2 { margin: 0; }
  .panel-fold > ul, .panel-fold > .panel-lead { margin-top: 1.1rem; }
  .api { margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: .5rem; }
  .api-row { background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 12px; padding: .7rem .85rem;
    display: flex; flex-direction: column; gap: .25rem; }
  .api-row dd { margin: 0; font-size: .82rem; color: var(--ink-2); }
  .api-row:last-child { border-color: color-mix(in srgb, var(--warn) 40%, transparent); }
  .api-row:last-child dt { color: var(--warn); }

  /* ---------- news ---------- */
  #news { margin: 1.5rem 0 2.5rem; }
  #news h2 { display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; }
  .news-date { font-family: var(--f-mono); font-size: .7rem; font-weight: 500; letter-spacing: 0; color: var(--muted); }
  .news-h { font-family: var(--f-mono); font-size: .64rem; letter-spacing: .12em; text-transform: uppercase;
    color: var(--muted); font-weight: 700; margin: 1.2rem 0 .6rem; }
  .news-list { list-style: none; margin: 0; padding: 0; display: grid; gap: .6rem;
    grid-template-columns: repeat(auto-fill, minmax(17rem, 1fr)); }
  .news-card { display: flex; flex-direction: column; gap: .4rem; height: 100%; padding: 1rem 1.05rem; border-radius: 14px;
    background: var(--panel-2); border: 1px solid var(--line-soft); color: inherit; text-decoration: none;
    transition: transform .15s ease, border-color .15s ease; }
  .news-card:hover, .news-card:focus-visible { transform: translateY(-2px); border-color: var(--accent-2); }
  .news-n { font-family: var(--f-disp); font-size: 2rem; font-weight: 700; line-height: 1; letter-spacing: -.04em;
    background: linear-gradient(95deg, var(--lime), var(--cyan)); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .news-host { font-family: var(--f-mono); font-size: .62rem; color: var(--muted); text-transform: lowercase; }
  .news-item h4 { margin: 0; font-size: .92rem; font-weight: 700; line-height: 1.3; color: var(--ink); }
  .news-item p { margin: 0; font-size: .8rem; color: var(--ink-2); line-height: 1.45;
    display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; }
  .news-empty { font-size: .84rem; color: var(--muted); margin: 0; }

  /* ---------- footer ---------- */
  .disclaimer { margin: 2rem 0 0; font-size: .76rem; color: var(--muted); max-width: 56rem; }
  .disclaimer strong { color: var(--ink-2); }
  footer { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--line); font-size: .76rem; color: var(--muted);
    display: flex; align-items: center; gap: .5rem; }
  @media (max-width: 560px) {
    .wrap { padding: 1rem .9rem 4rem; }
    .rig, .index, .panel { padding: 1rem; }
    .model-head { padding: 1.3rem 1.1rem 1.1rem; }
    .eng-pane { padding: 1rem; }
    .cx-kv, .cx-h span:nth-child(2) { display: none; }
    .cx { grid-template-columns: 7.5rem minmax(0, 1fr) 4.5rem; }
  }
"""


TEMPLATE = """<title>Apple LLM Performance Tracker</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Open weight AI models and their Apple M-series compatibility.">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Colton Kawamura">
<meta property="og:url" content="https://coltonkawamura.github.io/apple-llm-performance/">
<meta property="og:title" content="Apple LLM Performance Tracker">
<meta property="og:description" content="Open weight AI models and their Apple M-series compatibility.">
<meta property="og:image" content="https://coltonkawamura.github.io/apple-llm-performance/card.jpg">
<meta property="og:image:type" content="image/jpeg">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Dark card over a glowing Apple Silicon die reading Can Your Mac Run It? - find the best LLM for your Mac, updated daily. Open source on GitHub.">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Apple LLM Performance Tracker">
<meta name="twitter:description" content="Open weight AI models and their Apple M-series compatibility.">
<meta name="twitter:image" content="https://coltonkawamura.github.io/apple-llm-performance/card.jpg">
<meta name="twitter:image:alt" content="Dark card over a glowing Apple Silicon die reading Can Your Mac Run It? - find the best LLM for your Mac, updated daily. Open source on GitHub.">
<link rel="canonical" href="https://coltonkawamura.github.io/apple-llm-performance/">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap">
<style>{style}</style>

<div class="wrap">
  <p class="fork-note">
    <span class="fork-tag">Fork</span>
    of <a href="https://github.com/dreamingwell/apple-llm-performance" target="_blank" rel="noopener">dreamingwell/apple-llm-performance</a>
    &mdash; credit to its author. Maintained for my own use; a work in progress, so some things may be wrong or break.
  </p>
  <header>
    <div class="hero-top">
      <a class="gh" href="https://github.com/ColtonKawamura/apple-llm-performance"
         target="_blank" rel="noopener" aria-label="Open source on GitHub">
        <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" focusable="false"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-2.98-.88-2.98-2.9 0-.83.3-1.51.79-2.04-.08-.2-.35-1 .08-2.07 0 0 .65-.2 2.13.79a7.2 7.2 0 0 1 1.94-.26c.66 0 1.32.09 1.94.26 1.48-1 2.13-.79 2.13-.79.43 1.07.16 1.87.08 2.07.49.53.79 1.21.79 2.04 0 2.03-1.21 2.7-2.99 2.9.31.27.58.79.58 1.6 0 1.15-.01 2.09-.01 2.38 0 .21.15.46.55.38A7.99 7.99 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>
        <span>GitHub</span>
      </a>
    </div>
    <h1>Apple LLM <span class="grad">Performance Tracker</span></h1>
    <p class="hero-sub">Open-weight models &times; Apple silicon. Pick your Mac, pick a job, see what fits.</p>
  </header>

  <form class="rig" id="rig" aria-label="Cluster configuration">
    <span class="sec-k">01 &middot; Your Mac</span>
    <div class="rig-controls">
      <label class="rig-f"><span>CPU Model</span>
        <select id="rig-chip"></select>
      </label>
      <label class="rig-f"><span>Memory each</span>
        <select id="rig-mem"></select>
      </label>
      <label class="rig-f"><span>Units</span>
        <select id="rig-n"></select>
      </label>
    </div>
    <div class="rig-out" id="rig-out"></div>
    <div class="rig-gauge" id="rig-gauge" aria-hidden="true"></div>
    <p class="rig-warn" id="rig-warn" hidden></p>
  </form>

  <nav class="index" id="list" aria-label="Model index">
    <div class="ix-top">
      <div class="uc-f">
        <span class="sec-k">02 &middot; What for?</span>
        <div class="uc-list" id="uc-sel" role="group" aria-label="Jobs to combine"></div>
      </div>
    </div>
    <div class="uc-legend" id="uc-legend" aria-hidden="true"></div>
    <p class="uc-out" id="uc-out"></p>
    <div class="ix-head" aria-hidden="true"><span>#</span><span>Model</span><span>Engines</span><span>Verdict &middot; engine</span><span>Build &middot; memory</span></div>
    <div class="ix-rows">{index}
    </div>
    <div class="ix-key" aria-hidden="true"><span class="sq s-ready"></span>runs <span class="sq s-degraded"></span>caveats
      <span class="sq s-blocked"></span>blocked <span class="sq s-unknown"></span>n/a
      <span class="ix-key-m"><i></i></span>share of usable memory</div>
  </nav>

  <div class="detail" id="detail" hidden>
    <button type="button" class="back" id="back">
      <span aria-hidden="true">&larr;</span> All models
    </button>
{cards}
    <button type="button" class="back back-bottom" id="back-bottom">
      <span aria-hidden="true">&larr;</span> All models
    </button>
  </div>

  {news}

  <div class="panel wide">
    <details class="panel-fold">
      <summary><h2>General engine information</h2></summary>
    <p class="panel-lead">What each engine is, what its API actually implements, and the defects that follow
    you whichever model you load on it. All seven speak OpenAI on <code>/v1/chat/completions</code> with SSE
    streaming, and none is desktop-only &mdash; but &ldquo;OpenAI-compatible&rdquo; covers a wide range, and the
    differences land on exactly the features an agent leans on: whether tool-call arguments stream as deltas or
    arrive only after the turn, whether constrained decoding exists at all, and whether <code>tool_choice</code>
    is implemented. Five of the seven also serve Anthropic <code>/v1/messages</code>, so Claude Code can point
    at them directly.</p>
    {cross}
    </details>
  </div>

  <div class="panel">
    <details class="panel-fold">
      <summary><h2>Reading the scores</h2></summary>
    <ul>
      <li><strong>Terminal-Bench 2.0 and 2.1 are different benchmarks.</strong> Qwen3-Coder-Next's 36.2 is on v2.0; Qwen3.8-27B's 73.0 and GLM-5.2's 81.0 are on v2.1. Do not rank across the two &mdash; they are shown labelled, not normalised.</li>
      <li>Scores are vendor-reported or aggregator-reported, not reproduced here. Treat them as a shortlist filter, then verify the shortlist on your own context-rot harness.</li>
      <li>Nothing on this page has been measured on M5 Ultra hardware. Everything else is published numbers.</li>
      <li>Weights are the summed file sizes of the linked repository &mdash; safetensors for MLX builds, GGUF for the rest &mdash; measured, not estimated. A <strong>*</strong> marks the exception: a figure derived from parameter count because no build has been published anywhere.</li>
      <li><strong>The same model weighs different amounts on different engines.</strong> GGUF has quant tiers MLX does not, so llama.cpp can often fit a model MLX cannot &mdash; Qwen3-Coder-Next's GGUF ladder reaches down to 18.9 GB while its MLX ladder stops at 42.4 GB. Each engine tab states its own build and its own fit.</li>
      <li>Issue lists are scoped to the engine tab you are on, and are filtered for what actually applies on a Mac. A CUDA-only or ROCm-only report is not listed here even when it dominates the upstream thread.</li>
      <li>Fit assumes a 90% wired-memory limit plus framework overhead &mdash; ~10 GB for an LLM server, which has a paged KV pool and Metal buffers to hold, and ~1.5 GB for an image or audio runtime, which does not &mdash; and that pooling shards weights evenly. It answers "does this load", not "does this run well" &mdash; a model spread across machines still pays the Thunderbolt hop on every token.</li>
    </ul>
    </details>
  </div>

  <p class="disclaimer">
    <strong>Disclaimer.</strong> All of this is best effort and provided for entertainment purposes only.
    No warranty is given as to its accuracy. Benchmark scores are vendor- or aggregator-reported and are not
    reproduced here; issue states are a twice-daily snapshot; hardware figures are arithmetic, not measurements.
    Verify anything you intend to spend money on.
  </p>

  <footer>
    <span class="live-dot" aria-hidden="true"></span>
    Polled twice daily against the GitHub API across llama.cpp, Ollama, LM Studio, oMLX, vllm-mlx, mlx-lm and ds4;
    state changes only &mdash; open&rarr;closed, merged, new release tag.
  </footer>
</div>

<script data-newblock="1">
(function () {{
  // Every M-series chip, its peak memory bandwidth, and the union of unified-memory
  // options across every Mac that shipped with it - laptops, mini, iMac, Studio and
  // Mac Pro - because the chip is what decides whether a model fits, not the case.
  // tb5 marks Thunderbolt 5, which is what RDMA and JACCL tensor parallelism need;
  // everything older pools only over the ring/pipeline path.
  // gen groups the dropdown. bwNote flags chips with a binned lower-bandwidth variant.
  var MACHINES = {{
    m1:       {{ label: "M1",       gen: "M1", bw: 68,   mem: [8, 16],                tb: "Thunderbolt 3 / USB4", link: 40, tb5: false, ports: 2 }},
    m1pro:    {{ label: "M1 Pro",   gen: "M1", bw: 200,  mem: [16, 32],               tb: "Thunderbolt 4", link: 40, tb5: false, ports: 3 }},
    m1max:    {{ label: "M1 Max",   gen: "M1", bw: 400,  mem: [32, 64],               tb: "Thunderbolt 4", link: 40, tb5: false, ports: 4 }},
    m1ultra:  {{ label: "M1 Ultra", gen: "M1", bw: 800,  mem: [64, 128],              tb: "Thunderbolt 4", link: 40, tb5: false, ports: 6 }},
    m2:       {{ label: "M2",       gen: "M2", bw: 100,  mem: [8, 16, 24],            tb: "Thunderbolt 4", link: 40, tb5: false, ports: 2 }},
    m2pro:    {{ label: "M2 Pro",   gen: "M2", bw: 200,  mem: [16, 32],               tb: "Thunderbolt 4", link: 40, tb5: false, ports: 4 }},
    m2max:    {{ label: "M2 Max",   gen: "M2", bw: 400,  mem: [32, 64, 96],           tb: "Thunderbolt 4", link: 40, tb5: false, ports: 4 }},
    m2ultra:  {{ label: "M2 Ultra", gen: "M2", bw: 800,  mem: [64, 128, 192],         tb: "Thunderbolt 4", link: 40, tb5: false, ports: 6 }},
    m3:       {{ label: "M3",       gen: "M3", bw: 100,  mem: [8, 16, 24],            tb: "Thunderbolt 4", link: 40, tb5: false, ports: 2 }},
    m3pro:    {{ label: "M3 Pro",   gen: "M3", bw: 150,  mem: [18, 36],               tb: "Thunderbolt 4", link: 40, tb5: false, ports: 3 }},
    m3max:    {{ label: "M3 Max",   gen: "M3", bw: 400,  mem: [36, 48, 64, 96, 128],  tb: "Thunderbolt 4", link: 40, tb5: false, ports: 3,
                bwNote: "300 GB/s on the binned 14-core CPU / 30-core GPU part" }},
    m3ultra:  {{ label: "M3 Ultra", gen: "M3", bw: 819,  mem: [96, 256, 512],         tb: "Thunderbolt 5", link: 80, tb5: true,  ports: 6 }},
    m4:       {{ label: "M4",       gen: "M4", bw: 120,  mem: [16, 24, 32],           tb: "Thunderbolt 4", link: 40, tb5: false, ports: 2 }},
    m4pro:    {{ label: "M4 Pro",   gen: "M4", bw: 273,  mem: [24, 48, 64],           tb: "Thunderbolt 5", link: 80, tb5: true,  ports: 3 }},
    m4max:    {{ label: "M4 Max",   gen: "M4", bw: 546,  mem: [36, 48, 64, 128],      tb: "Thunderbolt 5", link: 80, tb5: true,  ports: 4,
                bwNote: "410 GB/s on the binned 14-core CPU part" }},
    m5:       {{ label: "M5",       gen: "M5", bw: 153,  mem: [16, 24, 32],           tb: "Thunderbolt 4", link: 40, tb5: false, ports: 2 }},
    m5pro:    {{ label: "M5 Pro",   gen: "M5", bw: 307,  mem: [24, 48, 64],           tb: "Thunderbolt 5", link: 80, tb5: true,  ports: 3 }},
    m5max:    {{ label: "M5 Max",   gen: "M5", bw: 614,  mem: [36, 48, 64, 128],      tb: "Thunderbolt 5", link: 80, tb5: true,  ports: 4,
                bwNote: "460 GB/s on the 32-core GPU part; 614 on the 40-core" }},
    m5ultra:  {{ label: "M5 Ultra", gen: "M5", bw: 1200, mem: [96, 256, 512],         tb: "Thunderbolt 5", link: 80, tb5: true,  ports: 6 }}
  }};
  var GENS = ["M5", "M4", "M3", "M2", "M1"];
  var USE_CASES = {usecases};
  var BAND_RANK = {{ full: 0, mild: 1, low: 2, pruned: 2, unusable: 3 }};
  var ORDER = Object.keys(MACHINES);
  var MAX_UNITS = 6, PRACTICAL_UNITS = 4, WIRED = 0.90;
  // Framework overhead. The larger figure covers an LLM server's KV pool and
  // Metal buffers; a diffusion or TTS runtime carries far less, and charging it
  // 10 GB made a 310 MB model report "10 GB resident".
  var OVERHEAD_TEXT = 10, OVERHEAD_MEDIA = 1.5, OVERHEAD = OVERHEAD_TEXT;
  var BANDS = {bands};

  // The chosen jobs: every chip that is pressed, in display order. Empty means
  // "anything" - the plain list. Combining ANDs them: a model only counts for
  // the combined pick where it publishes a figure for EACH chosen job, and its
  // bar is the average of its share of the leader across those jobs.
  function ucSelected() {{
    var out = [];
    var chips = document.querySelectorAll("#uc-sel .uc-chip");
    for (var i = 0; i < chips.length; i++) {{
      if (chips[i].getAttribute("aria-pressed") === "true") {{
        var id = chips[i].getAttribute("data-uc");
        for (var ui = 0; ui < USE_CASES.length; ui++) {{
          if (USE_CASES[ui].id === id) {{ out.push(USE_CASES[ui]); break; }}
        }}
      }}
    }}
    return out;
  }}
  // The strictest fidelity gate among the chosen jobs wins: a build acceptable
  // for one of the jobs is not acceptable for the other.
  function ucGate(ucs) {{
    var g = "unusable";
    for (var i = 0; i < ucs.length; i++) {{
      if (BAND_RANK[ucs[i].gate] < BAND_RANK[g]) g = ucs[i].gate;
    }}
    return g;
  }}


  var chipSel = document.getElementById("rig-chip"),
      memSel  = document.getElementById("rig-mem"),
      nSel    = document.getElementById("rig-n"),
      out     = document.getElementById("rig-out"),
      warn    = document.getElementById("rig-warn");
  if (!chipSel) return;

  var DEF_CHIP = "m5ultra", DEF_MEM = 256, DEF_N = 1;

  var q = new URLSearchParams(location.search);
  var chip = MACHINES[q.get("chip")] ? q.get("chip") : DEF_CHIP;
  var mem  = parseInt(q.get("mem"), 10);
  if (!mem || MACHINES[chip].mem.indexOf(mem) === -1) {{
    mem = MACHINES[chip].mem.indexOf(DEF_MEM) !== -1 ? DEF_MEM : MACHINES[chip].mem[0];
  }}
  var n = parseInt(q.get("n"), 10);
  if (!n || n < 1 || n > MAX_UNITS) n = DEF_N;

  GENS.forEach(function (g) {{
    var grp = document.createElement("optgroup");
    grp.label = g;
    ORDER.filter(function (k) {{ return MACHINES[k].gen === g; }}).forEach(function (k) {{
      var o = document.createElement("option");
      o.value = k; o.textContent = MACHINES[k].label;
      grp.appendChild(o);
    }});
    chipSel.appendChild(grp);
  }});
  for (var i = 1; i <= MAX_UNITS; i++) {{
    var o = document.createElement("option");
    o.value = i; o.textContent = i + (i === 1 ? " machine" : " machines");
    nSel.appendChild(o);
  }}

  function fillMem(keepIfPossible) {{
    var opts = MACHINES[chip].mem;
    memSel.innerHTML = "";
    opts.forEach(function (g) {{
      var o = document.createElement("option");
      o.value = g; o.textContent = g + " GB";
      memSel.appendChild(o);
    }});
    if (opts.indexOf(keepIfPossible) === -1) {{
      keepIfPossible = opts.indexOf(DEF_MEM) !== -1 ? DEF_MEM : opts[opts.length - 1];
    }}
    memSel.value = keepIfPossible;
  }}

  function fmt(gb) {{
    if (gb >= 1000) return (gb / 1000).toFixed(2) + " TB";
    if (gb < 1) return Math.round(gb * 1000) + " MB";   // TTS models are sub-gigabyte
    return Math.round(gb) + " GB";
  }}

  function apply() {{
    chip = chipSel.value;
    var g = parseInt(memSel.value, 10);
    n = parseInt(nSel.value, 10);
    var perNode = g * WIRED, cluster = perNode * n, M = MACHINES[chip];

    // The interconnect only matters once there is something to interconnect.
    var tile = function (k, v, sub, cls) {{
      return '<div class="tile' + (cls ? " " + cls : "") + '"><span class="tile-k">' + k +
        '</span><span class="tile-v">' + v + "</span>" +
        (sub ? '<span class="tile-s">' + sub + "</span>" : "") + "</div>";
    }};
    out.innerHTML =
      tile("Cluster", n + " \u00d7 " + M.label, g + " GB each") +
      tile("Pooled", fmt(g * n), "unified memory") +
      tile("Usable", fmt(cluster), "after the wired-memory limit", "tile-hi") +
      tile("Bandwidth", M.bw >= 1000 ? (M.bw / 1000).toFixed(1) + " TB/s" : M.bw + " GB/s", "per machine") +
      (n > 1 ? tile("Link", M.link + " Gb/s", M.tb) : "");
    var gz = document.getElementById("rig-gauge");
    if (gz) {{
      var segs = "";
      for (var si = 0; si < n; si++) {{
        segs += '<span class="rg-seg"><i style="width:' + (WIRED * 100).toFixed(0) + '%"></i></span>';
      }}
      gz.innerHTML = '<div class="rg-bar">' + segs + "</div>" +
        '<div class="rg-key"><span class="k-u">usable ' + fmt(cluster) + "</span>" +
        '<span class="k-r">held back by macOS ' + fmt(g * n - cluster) + "</span></div>";
    }}
    out.classList.add("pulse");
    setTimeout(function () {{ out.classList.remove("pulse"); }}, 450);

    if (n > 1 && !M.tb5) {{
      warn.hidden = false;
      warn.textContent = "Clustering multiple " + M.label + " machines will be very slow: " + M.tb +
        " has no RDMA path, so pooling falls back to ring pipeline parallelism.";
    }} else if (n > PRACTICAL_UNITS) {{
      warn.hidden = false;
      warn.textContent = "Past " + PRACTICAL_UNITS + " machines a full Thunderbolt mesh runs out of ports. " +
        "Treat these rows as arithmetic, not a supported setup.";
    }} else {{ warn.hidden = true; }}

    function fitDetail(w, over, kvWord) {{
      var OVERHEAD = over === undefined ? OVERHEAD_TEXT : over;
      var kv = kvWord === undefined ? " for KV" : kvWord;
      var resident = w + OVERHEAD;
      var nodes = Math.ceil(resident / perNode);
      if (resident > cluster) {{
        return {{ tooBig: true, nodes: nodes, resident: resident, free: 0,
                 short: "needs " + nodes + " machine" + (nodes === 1 ? "" : "s"),
                 text: "needs " + nodes + " machine" + (nodes === 1 ? "" : "s") }};
      }}
      if (nodes > 1) {{
        var freeP = cluster - resident;
        return {{ tooBig: false, nodes: nodes, resident: resident, free: freeP, copies: 1,
                 short: fmt(freeP) + " free",
                 text: "pooled across " + nodes + " of your " + n + ", " + fmt(freeP) + " left" + kv }};
      }}
      // It fits on one machine, so every machine runs its own copy. Nothing is
      // pooled and nothing is shared - the capacity simply multiplies.
      var free = perNode - resident;
      return {{ tooBig: false, nodes: 1, resident: resident, free: free, copies: n,
               short: fmt(free) + " free",
               text: n > 1
                 ? fmt(free) + " free" + kv + " per machine \u2014 run as individual compute, not as a cluster"
                 : fmt(free) + " free" + kv }};
    }}

    // Pick the target build for this cluster - not simply the biggest thing that
    // fits. Measured KL divergence flattens above 4 bits per weight (0.41 at
    // Q4_K_XL against 0.24 at Q5 and 0.10 at Q8), so paying an extra 200 GB for
    // Q8 buys almost nothing and costs the KV headroom that decides how much
    // context and how many concurrent streams you get. So: the CHEAPEST rung
    // that still clears 4 bits, and only if none does, the best of what is left.
    function pickIn(ladder, over) {{
      var fits = ladder.filter(function (r) {{ return r.gb + over <= cluster; }});
      if (!fits.length) return null;
      var full = fits.filter(function (r) {{
        return r.kind === "native" || (r.kind !== "pruned" && r.bpw >= 4);
      }});
      if (full.length) return full[full.length - 1];   // ladder is largest-first
      return fits[0];                                  // best available below 4 bpw
    }}
    function pick(ladder) {{ return pickIn(ladder, OVERHEAD_TEXT); }}

    function band(rung) {{
      if (rung.kind === "native") {{
        // Media checkpoints ship at a stated precision and bundle encoders and a
        // VAE with the transformer, so a bits-per-weight band would be invented.
        return {{ k: "full", label: "As published", why: "" }};
      }}
      if (rung.kind === "pruned") {{
        return {{ k: "pruned", label: "Expert-pruned",
                 why: "This build was not quantised down, it was pruned: whole experts were scored and " +
                      "deleted. The surviving weights are near lossless, and the capacity they came from " +
                      "is gone. Bits per weight does not describe this loss." }};
      }}
      for (var i = 0; i < BANDS.length; i++) {{
        if (rung.bpw >= BANDS[i][0]) {{
          return {{ k: BANDS[i][1], label: BANDS[i][2], why: BANDS[i][3] }};
        }}
      }}
      return {{ k: "unusable", label: "Below agentic-usable", why: "" }};
    }}

    // freeGB is per machine when the model fits on one, so multiply the stream
    // count by the number of copies to get the total the whole group serves.
    function ctxRows(bpt, maxctx, freeGB, copies) {{
      if (!bpt || !maxctx || freeGB <= 0) return [];
      copies = copies || 1;
      var out = [];
      [[1, "full"], [0.75, "three quarters"], [0.5, "half"], [0.25, "a quarter"]].forEach(function (f) {{
        var tok = Math.round(maxctx * f[0]);
        var per = bpt * tok;                       // bytes
        out.push({{ tok: tok, frac: f[1], perGB: per / 1e9,
                   streams: Math.floor((freeGB * 1e9) / per) * copies }});
      }});
      // A model can load and still have no room for a quarter of its advertised
      // window - MiniMax M3 on one 256 GB machine is exactly that. Saying "runs"
      // above four rows of "does not fit" is useless, so state what does fit.
      if (out[out.length - 1].streams < 1) {{
        var maxTok = Math.floor((freeGB * 1e9) / bpt);
        out.push({{ tok: maxTok, frac: "largest that fits", cap: true,
                   perGB: (bpt * maxTok) / 1e9, streams: maxTok >= 2048 ? copies : 0 }});
      }}
      return out;
    }}

    // Paint a weights | overhead | free bar against the memory the build lives
    // in: one machine when it fits on one (each machine runs its own copy),
    // the whole pool when it is sharded.
    function setMem(bar, w, over, fd) {{
      if (!bar) return;
      var den = fd.nodes === 1 ? perNode : cluster;
      var seg = bar.querySelectorAll("i");
      [w, over, Math.max(0, fd.free)].forEach(function (v, i) {{
        if (seg[i]) seg[i].style.width = Math.max(0, Math.min(100, v / den * 100)).toFixed(1) + "%";
      }});
    }}
    function esc(t) {{
      return String(t).replace(/[&<>"]/g, function (c) {{
        return {{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }}[c];
      }});
    }}

    function tokFmt(t) {{
      return t >= 1000000 ? (t / 1000000).toFixed(t % 1000000 ? 2 : 0) + "M"
                          : Math.round(t / 1000) + "k";
    }}

    document.querySelectorAll("[data-payload]").forEach(function (el) {{
      var pl;
      try {{ pl = JSON.parse(el.getAttribute("data-payload")); }} catch (e) {{ return; }}
      var engs = pl.engines || [];
      var isText = (el.getAttribute("data-mod") || "text") === "text";
      var over = isText ? OVERHEAD_TEXT : OVERHEAD_MEDIA;
      var kvWord = pl.kv && pl.kv.bpt ? " for KV" : "";

      var chosen = null, rung = null;
      for (var i = 0; i < engs.length; i++) {{
        var r = pickIn(engs[i].ladder, over);
        if (r) {{ chosen = engs[i]; rung = r; break; }}
      }}
      if (!chosen && engs.length) {{
        // nothing fits anywhere: report whichever engine has the smallest build
        chosen = engs.reduce(function (a, b) {{
          return b.ladder[b.ladder.length - 1].gb < a.ladder[a.ladder.length - 1].gb ? b : a;
        }});
        rung = chosen.ladder[chosen.ladder.length - 1];
      }}
      if (!chosen) {{
        var f0 = el.querySelector(".fit");
        if (f0) f0.textContent = "no build published";
        return;
      }}

      var f = fitDetail(rung.gb, over, kvWord);
      var bd = band(rung);
      var cls, label;
      if (chosen.s === "blocked" || chosen.s === "unknown") {{
        cls = chosen.s; label = chosen.label;
      }} else if (f.tooBig) {{
        cls = "toolarge"; label = "Too large";
      }} else if (bd.k === "unusable") {{
        cls = "blocked"; label = "Too degraded";
      }} else if (bd.k === "mild" || bd.k === "low" || bd.k === "pruned") {{
        cls = "degraded"; label = chosen.s === "ready" ? "Runs, " + bd.label.toLowerCase() : chosen.label;
      }} else {{
        cls = chosen.s; label = chosen.label;
      }}

      if (el.classList.contains("ix-row")) {{
        el.__pick = {{ model: el.querySelector(".ix-name").textContent.trim(), engine: chosen.name,
                      engineId: chosen.id, gb: rung.gb, bpw: rung.bpw, band: bd.k,
                      tooBig: f.tooBig || bd.k === "unusable" || chosen.s === "blocked" }};
        el.className = "ix-row v-" + cls;
        var st = el.querySelector(".ix-status");
        st.className = "ix-status v-" + cls;
        st.textContent = label;
        el.querySelector(".ix-eng").textContent = chosen.name;
        el.querySelector(".ix-size").textContent = fmt(rung.gb);
        el.querySelector(".fit").textContent = f.short;
        var meter = el.querySelector(".ix-meter i");
        if (meter) {{
          var den = f.nodes === 1 ? perNode : cluster;
          meter.style.width = (f.tooBig ? 100 : Math.min(100, f.resident / den * 100)).toFixed(1) + "%";
        }}
        return;
      }}

      el.className = "model v-" + cls;
      // Open the card on the engine the glance row names for this cluster, so the
      // two never disagree - unless the reader has already picked a tab by hand.
      var grp = el.querySelector(".eng");
      if (grp && !grp.hasAttribute("data-user-picked")) {{
        var want = grp.querySelector('.eng-tab[data-eng="' + chosen.id + '"]');
        if (want && want.getAttribute("aria-selected") !== "true") {{
          grp.querySelectorAll(".eng-tab").forEach(function (t) {{
            t.setAttribute("aria-selected", t === want ? "true" : "false");
          }});
          grp.querySelectorAll(".eng-pane").forEach(function (pa) {{
            pa.hidden = pa.getAttribute("data-eng") !== chosen.id;
          }});
        }}
      }}
      var vb = el.querySelector(".verdict");
      vb.className = "verdict v-" + cls;
      vb.textContent = label;
      var mg = el.querySelector(".model-gauge");
      if (mg) {{
        mg.hidden = chosen.s === "blocked" || f.tooBig;
        if (!mg.hidden) setMem(mg.querySelector(".mem-bar"), rung.gb, over, f);
      }}
      var mf = el.querySelector(".model-fit");
      if (mf) {{
        mf.className = "model-fit" + (f.tooBig ? " toolarge" : "");
        mf.textContent = chosen.s === "blocked"
          ? "No engine here can load this yet \u2014 see the tabs below for why."
          : f.tooBig
            ? "Does not fit. Smallest build is " + chosen.name + " at " + fmt(rung.gb) + ", which " + f.text + "."
            : "Best fit here: " + chosen.name + ", " + fmt(rung.gb) + " of weights, " + f.text + ".";
      }}

      // Each engine tab reports its own rung on the same cluster.
      engs.forEach(function (eng) {{
        var pane = el.querySelector('.eng-pane[data-eng="' + eng.id + '"]');
        if (!pane) return;
        var fit = pane.querySelector(".eng-fit");
        if (!fit || !fit.hasAttribute("data-has-ladder")) return;
        var r = pickIn(eng.ladder, over) || eng.ladder[eng.ladder.length - 1];
        var pf = fitDetail(r.gb, over, kvWord), pb = band(r);

        fit.classList.toggle("toolarge", pf.tooBig);
        fit.textContent = pf.tooBig
          ? "Too large: " + fmt(pf.resident) + " resident, " + pf.text + "."
          : fmt(pf.resident) + " resident on this cluster, " + pf.text + ".";

        var mb = pane.querySelector(".mem-bar");
        if (mb) {{
          mb.hidden = pf.tooBig;
          if (!pf.tooBig) setMem(mb, r.gb, over, pf);
        }}
        var lw = pane.querySelector(".ladder-wrap");
        if (lw && eng.ladder.length) {{
          // The measured ladder as a bar chart: one bar per build, largest
          // first, height to scale. Builds that fit are lit; the picked one is
          // outlined; the dashed line is the biggest build this cluster holds.
          var maxGb = 0, nFit = 0;
          eng.ladder.forEach(function (x) {{ if (x.gb > maxGb) maxGb = x.gb; }});
          var limit = cluster - over;
          var bars = eng.ladder.map(function (x) {{
            var ok = x.gb + over <= cluster;
            if (ok) nFit++;
            var h = Math.max(4, x.gb / maxGb * 100);
            var tip = x.label + " \u00b7 " + fmt(x.gb) +
              (x.kind === "quant" && x.bpw ? " \u00b7 " + x.bpw.toFixed(2) + " bits/weight" : "");
            return '<span class="lb b-' + band(x).k + (ok ? " fits" : "") + (x === r ? " pick" : "") +
              '" style="height:' + h.toFixed(1) + '%" title="' + esc(tip) + '"></span>';
          }}).join("");
          var line = limit > 0 && limit < maxGb
            ? '<span class="lb-line" style="bottom:' + (limit / maxGb * 100).toFixed(1) + '%"><em>' +
              fmt(limit) + "</em></span>" : "";
          lw.hidden = false;
          lw.querySelector(".ladder").innerHTML = line + bars;
          var cap = lw.querySelector(".ladder-cap");
          if (!cap) {{
            cap = document.createElement("p");
            cap.className = "ladder-cap";
            lw.appendChild(cap);
          }}
          cap.innerHTML = "<b>" + nFit + "</b> of " + eng.ladder.length + " builds fit &middot; " +
            fmt(eng.ladder[eng.ladder.length - 1].gb) + " to " + fmt(maxGb);
        }}
        var link = pane.querySelector(".build-link"), bpw = pane.querySelector(".build-bpw");
        if (link) {{
          // r.url is the rung's own weights - the GGUF file or, for a split
          // quant, the folder its shards live in - not the repo that holds it.
          link.textContent = r.label;
          link.setAttribute("href", r.url);
        }}
        if (bpw) {{
          bpw.textContent = r.kind === "pruned" ? "expert-pruned"
                          : r.kind === "native" ? "as published"
                          : r.bpw.toFixed(2) + " bits/weight";
          bpw.className = "build-bpw b-" + pb.k;
        }}

        var fid = pane.querySelector(".fidelity");
        if (fid) {{
          if (pb.k === "full") {{
            fid.hidden = true;
          }} else {{
            fid.hidden = false;
            fid.className = "fidelity b-" + pb.k;
            fid.innerHTML = "<strong>" + pb.label + ".</strong> " + pb.why +
              (eng.note ? " " + eng.note : "");
          }}
        }}

        var wrap = pane.querySelector(".ctx-wrap");
        if (wrap) {{
          var rows = pf.tooBig ? [] : ctxRows(pl.kv.bpt, pl.kv.maxctx, pf.free, pf.copies);
          if (!rows.length) {{
            wrap.hidden = true;
          }} else {{
            wrap.hidden = false;
            var most = 1;
            rows.forEach(function (c) {{ if (c.streams > most) most = c.streams; }});
            wrap.querySelector(".ctx").innerHTML =
              '<div class="cx cx-h"><span>Context each</span><span>KV / stream</span><span></span><span>Streams</span></div>' +
              rows.map(function (c) {{
              var label = c.cap ? tokFmt(c.tok) + " &mdash; " + c.frac
                                : tokFmt(c.tok) + " <em>" + c.frac + "</em>";
              var pct = c.streams < 1 ? 0 : Math.max(2, c.streams / most * 100);
              return '<div class="cx' + (c.cap ? " ctx-cap" : "") + '"><span class="cx-l">' + label +
                     '</span><span class="cx-kv">' +
                     (c.perGB < 1 ? (c.perGB * 1000).toFixed(0) + " MB" : c.perGB.toFixed(1) + " GB") +
                     '</span><span class="cx-bar"><i style="width:' + pct.toFixed(1) + '%"></i></span>' +
                     '<span class="ctx-n' + (c.streams < 1 ? ' none' : '') + '">' +
                     (c.streams < 1 ? "does not fit" : c.streams) + "</span></div>";
            }}).join("");
            wrap.querySelector(".ctx-why").textContent =
              "KV at fp16: " + (pl.kv.bpt / 1024).toFixed(1) + " KiB per token. " + pl.kv.why + ". " +
              (pf.copies > 1 ? "Stream counts are the total across all " + pf.copies +
                               " machines, each running its own copy. " : "") +
              "Quantising the KV cache to 8-bit doubles every count above.";
          }}
        }}
      }});
    }});

    // The winner for the chosen job or jobs. One job: walk its curated ranking
    // and take the first model that fits and clears its fidelity gate. Several
    // jobs: AND them - a model only counts where it publishes a figure for
    // EACH chosen job - then take the strongest among those, ordered by the
    // first chosen job. Ranking is fixed at build time because the benchmarks
    // are not mutually comparable; what changes with the cluster is only which
    // entries are reachable.
    var ucs = ucSelected();
    var uc = ucs.length ? ucs[0] : null;
    var gate = ucs.length ? ucGate(ucs) : null;
    var combined = ucs.length > 1;
    var num = function (v) {{
      var m = String(v).match(/-?[0-9]+([.][0-9]+)?/);
      return m ? parseFloat(m[0]) : null;
    }};
    // The intersection of the chosen jobs' rankings: a map of model id to its
    // position in the first chosen job's rank plus one entry per job it appears in.
    var pos = {{}}, entries = {{}};
    if (ucs.length) {{
      ucs.forEach(function (u, ui) {{
        u.rank.forEach(function (r) {{
          if (pos[r[0]] === undefined) entries[r[0]] = [];
          if (ui === 0) pos[r[0]] = u.rank.indexOf(r);
          if (!entries[r[0]].some(function (e) {{ return e[0] === u.id; }})) {{
            entries[r[0]].push([u.id, r[1], r[2]]);
          }}
        }});
      }});
    }}
    // Models built for a different kind of output are not "unranked", they are
    // irrelevant - a text model has no place in an image-generation table. The
    // default view shows text models; the others appear with their categories.
    var mods = {{}};
    if (ucs.length) ucs.forEach(function (u) {{ mods[u.mod] = true; }});
    var inMod = function (mod) {{ return ucs.length ? !!mods[mod] : mod === "text"; }};
    document.querySelectorAll(".ix-row").forEach(function (r) {{
      r.classList.remove("uc-best", "uc-out-of-scope", "j1", "multi");
      r.hidden = !inMod(r.getAttribute("data-mod") || "text");
    }});
    document.querySelectorAll(".ix-row").forEach(function (r) {{
      r.removeAttribute("data-rank");
      var rk = r.querySelector(".ix-rank");
      if (rk) rk.textContent = "";
    }});
    if (!ucs.length) {{
      if (ucOut) ucOut.innerHTML = "";
      ucLegend.classList.remove("on");
      ucLegend.innerHTML = "";
      document.querySelectorAll(".ix-row").forEach(function (r) {{
        r.style.order = "";
        r.classList.remove("no-bar", "j1", "multi");
        var lanes = r.querySelectorAll(".ix-lane");
        for (var li = 0; li < lanes.length; li++) {{
          lanes[li].classList.remove("on");
          var fill = lanes[li].querySelector("i");
          if (fill) fill.style.width = "0";
        }}
      }});
    }} else {{
      // The legend is the whole of the page's bar grammar: one swatch per
      // picked job, in pick order, an arrow, and the green swatch labelled
      // avg for the average lane. It exists only in the combined view - with
      // one job, the row's own coloured lane is all there is, and the chip
      // above already maps that job to its colour.
      if (combined) {{
        ucLegend.innerHTML = ucs.map(function (u) {{
          return '<span class="lg-it"><span class="lg-swatch uc-slot-' + u.id + '"></span></span>';
        }}).join("") +
          '<span class="lg-arr" aria-hidden="true">\\u2192</span>' +
          '<span class="lg-it"><span class="lg-swatch-avg"></span><span class="lg-word">avg</span></span>';
        ucLegend.classList.add("on");
        ucLegend.setAttribute("aria-hidden", "false");
      }} else {{
        ucLegend.classList.remove("on");
        ucLegend.innerHTML = "";
        ucLegend.setAttribute("aria-hidden", "true");
      }}
      // Usable = ranked in EVERY chosen job (a true intersection), fits on this
      // cluster, and clears the strictest of the chosen jobs' fidelity gates.
      // For a single job that reduces to "ranked in the job."
      var usableSet = {{}};
      document.querySelectorAll(".ix-row").forEach(function (r) {{
        var mid = r.getAttribute("data-model");
        var inAll = ucs.every(function (u) {{
          return (entries[mid] || []).some(function (e) {{ return e[0] === u.id; }});
        }});
        usableSet[mid] = inAll && r.__pick && !r.__pick.tooBig &&
                         BAND_RANK[r.__pick.band] <= BAND_RANK[gate];
      }});
      // Full bar = a perfect score on the job's own benchmark. A percentage
      // suite is out of 100, so a perfect score is 100. A suite that is not a
      // percentage (an Elo, an active-parameter note) publishes no maximum, so
      // the best figure that still fits stands in for 100 - which is why the
      // leader below is still needed: only as the reference for those suites.
      var lead = [];
      ucs.forEach(function (u) {{
        var lm = null, lv = null;
        for (var li = 0; li < u.rank.length; li++) {{
          var lrow = document.querySelector('.ix-row[data-model="' + u.rank[li][0] + '"]');
          if (lrow && lrow.__pick && !lrow.__pick.tooBig &&
              BAND_RANK[lrow.__pick.band] <= BAND_RANK[gate]) {{
            lm = u.rank[li][1];
            lv = num(u.rank[li][2]);
            break;
          }}
        }}
        lead.push({{ id: u.id, metric: lm, val: lv }});
      }});
      var anyLead = lead.some(function (L) {{ return L.val !== null; }});
      // A percentage sits on the 0-100 scale and is compared against 100. A
      // non-percentage has no published maximum, so it is compared against the
      // best figure on its own scale (the leader's). An Elo next to a
      // percentage is never compared directly.
      var isPct = function (v) {{ return v !== null && v <= 100; }};
      // One job's share of a perfect score. The per-job lane is drawn from
      // this; barFor is the average of the same pieces across the chosen jobs.
      var barForJob = function (mid, L) {{
        if (pos[mid] === undefined) return null;
        var e = entries[mid].filter(function (x) {{ return x[0] === L.id; }});
        if (!e.length) return null;
        var v = num(e[0][2]);
        if (v === null) return null;
        if (isPct(v)) return Math.max(0, Math.min(100, v));
        if (!L.val) return null;
        if (isPct(L.val)) return null;
        return Math.max(0, Math.min(100, (v / L.val) * 100));
      }};
      var barFor = function (mid) {{
        if (pos[mid] === undefined) return null;
        var acc = 0, cnt = 0;
        lead.forEach(function (L) {{
          var w = barForJob(mid, L);
          if (w === null) return;   // no numeric leader, unranked, or off scale
          acc += w;
          cnt++;
        }});
        if (!cnt) return null;
        return acc / cnt;
      }};
      // With several jobs chosen, usable models are ordered by their average of
      // the per-job share of a perfect score (the green bar) rather than by any
      // single job's curated rank. A single job keeps its curated order.
      var rankBy = null;
      if (combined) {{
        rankBy = ucs[0].rank.map(function (x) {{ return x[0]; }})
          .filter(function (m) {{ return usableSet[m]; }})
          .sort(function (a, b) {{ return (barFor(b) || 0) - (barFor(a) || 0); }});
      }}
      // Winner: the top usable model. With one job that is the first entry of
      // its curated rank; with several it is the highest average of its per-job
      // share of a perfect score - the head of the same order the rows below use.
      var winner = null;
      var worder = combined ? rankBy : ucs[0].rank.map(function (x) {{ return x[0]; }});
      for (var ri = 0; ri < worder.length; ri++) {{
        var mid = worder[ri];
        if (!usableSet[mid]) continue;
        var row = document.querySelector('.ix-row[data-model="' + mid + '"]');
        if (!row) continue;
        winner = {{ row: row, mid: mid }};
        break;
      }}
      document.querySelectorAll(".ix-row").forEach(function (r) {{
        var mid = r.getAttribute("data-model");
        var ranked = pos[mid] !== undefined;
        if (!ranked) r.classList.add("uc-out-of-scope");
        var usable = usableSet[mid];
        var orderIdx = usable
          ? (combined ? rankBy.indexOf(mid) : pos[mid])
          : (ranked ? 1000 + pos[mid] : 4000);
        r.style.order = orderIdx;
        var w = usable ? barFor(mid) : null;
        // Only meaningful when at least one chosen job has a numeric leader to
        // compare against; with no benchmark at all, nothing is "off scale".
        r.classList.toggle("no-bar", usable && w === null && anyLead);
        // Every pick draws its thin lanes under the name: one job, its own
        // coloured lane; several, the picked jobs' lanes plus the green
        // average lane, which exists only in the combined view.
        r.classList.toggle("j1", !combined);
        r.classList.toggle("multi", combined);
        var lanes = r.querySelectorAll(".ix-lane");
        for (var li = 0; li < lanes.length; li++) {{
          lanes[li].classList.remove("on");
        }}
        var avgW = usable ? barFor(mid) : null;
        var hasAny = false;
        lead.forEach(function (L) {{
          if (!combined && L.id !== ucs[0].id) return;
          var lane = r.querySelector(".ix-lane.uc-slot-" + L.id);
          if (!lane) return;
          var lw = usable ? barForJob(mid, L) : null;
          lane.classList.toggle("on", lw !== null);
          if (lw !== null) hasAny = true;
          var fill = lane.querySelector("i");
          if (fill) fill.style.width = lw === null ? "0" : lw.toFixed(1) + "%";
        }});
        if (combined) {{
          var avgLane = r.querySelector(".ix-lane.ix-lane-avg");
          if (avgLane) {{
            avgLane.classList.toggle("on", avgW !== null || hasAny);
            var afill = avgLane.querySelector("i");
            if (afill) afill.style.width = avgW === null ? "0" : avgW.toFixed(1) + "%";
          }}
        }}
      }});
      // Rank numbers follow the visual order, which is CSS order, not DOM order.
      [].slice.call(document.querySelectorAll(".ix-row")).filter(function (r) {{
        return !r.hidden && usableSet[r.getAttribute("data-model")];
      }}).sort(function (a, b) {{ return (+a.style.order) - (+b.style.order); }})
        .forEach(function (r, i) {{
          r.setAttribute("data-rank", i + 1);
          var rk = r.querySelector(".ix-rank");
          if (rk) rk.textContent = i + 1;
        }});
      if (winner) {{
        winner.row.classList.add("uc-best");
        ucOut.innerHTML = "";
      }} else {{
        var jobs = combined
          ? "Every model ranked for all of " + ucs.map(function (u) {{ return u.label.toLowerCase(); }}).join(", ")
          : "Every model ranked for " + uc.label.toLowerCase();
        ucOut.innerHTML = "<strong>Nothing suitable fits this cluster.</strong>" +
          "<span class='uc-why'>" + jobs +
          " is either too large here, or only fits at a precision below what " +
          (combined ? "the strictest chosen job tolerates. Add memory, add a machine, or choose fewer or easier jobs"
                    : "this job tolerates. Add memory, add a machine, or pick a different job") +
          ".</span>";
      }}
    }}

    var p = new URLSearchParams();
    p.set("chip", chip); p.set("mem", g); p.set("n", n);
    if (ucs.length) p.set("uc", ucs.map(function (u) {{ return u.id; }}).join(","));
    history.replaceState(null, "", location.pathname + "?" + p.toString() + (location.hash || ""));
  }}

  var ucSel = document.getElementById("uc-sel"), ucOut = document.getElementById("uc-out"),
      ucLegend = document.getElementById("uc-legend");
  if (ucSel) {{
    // One chip per job, in display order. Pressed chips are the chosen jobs;
    // none pressed is the plain list. Chips, not a <select multiple>: the
    // current choice stays visible and multi-picking needs no modifier key.
    USE_CASES.forEach(function (u) {{
      var b = document.createElement("button");
      b.type = "button";
      b.className = "uc-chip";
      b.setAttribute("data-uc", u.id);
      b.setAttribute("aria-pressed", "false");
      b.innerHTML = "<span class='uc-dot' aria-hidden='true'></span>" +
                    "<span class='uc-tick' aria-hidden='true'>\u2713</span>" +
                    "<span>" + u.label + "</span>";
      b.addEventListener("click", function () {{
        var on = b.getAttribute("aria-pressed") === "true";
        b.setAttribute("aria-pressed", on ? "false" : "true");
        apply();
      }});
      ucSel.appendChild(b);
    }});
    var uc0 = q.get("uc");
    if (uc0) {{
      uc0.split(",").forEach(function (id) {{
        var chip = ucSel.querySelector(".uc-chip[data-uc='" + id + "']");
        if (chip) chip.setAttribute("aria-pressed", "true");
      }});
    }}
  }}

  // The hash is the view: no hash shows the list, #model-id shows that card.
  // Keeping the state in the URL means deep links, the browser Back button and
  // the on-page Back control are all the same mechanism.
  var listEl = document.getElementById("list"),
      detailEl = document.getElementById("detail"),
      backEl = document.getElementById("back"),
      cards = [].slice.call(document.querySelectorAll(".model"));

  function route(pin) {{
    // Swapping views changes the document height, and Chrome's scroll anchoring
    // reacts by moving the viewport to keep its anchor element stable - which is
    // what made this jump even with the anchor click prevented. Pin the scroll
    // position across the swap.
    var y = window.scrollY;
    var want = (location.hash || "").replace(/^#/, "");
    var found = null;
    cards.forEach(function (c) {{
      var mine = c.id === "card-" + want;
      c.hidden = !mine;
      if (mine) found = c;
    }});
    if (listEl) listEl.hidden = !!found;
    if (detailEl) detailEl.hidden = !found;
    // Restore now and again after layout: the adjustment does not always land in
    // the same tick as the attribute change.
    if (pin !== false) {{
      var hold = function () {{ if (window.scrollY !== y) window.scrollTo(0, y); }};
      hold();
      requestAnimationFrame(hold);
    }}
    return found;
  }}

  if (listEl) {{
    listEl.addEventListener("click", function (ev) {{
      var row = ev.target.closest ? ev.target.closest(".ix-row") : null;
      if (!row || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.button !== 0) return;
      ev.preventDefault();
      history.pushState(null, "", location.pathname + location.search +
                        "#" + row.getAttribute("data-model"));
      route();
    }});
  }}
  var backBottomEl = document.getElementById("back-bottom");
  if (backBottomEl) {{
    backBottomEl.addEventListener("click", function () {{
      history.pushState(null, "", location.pathname + location.search);
      // Skip the scroll pin here: from the bottom of a long card, holding
      // position would leave the reader staring at the reference panels.
      route(false);
      var rig = document.getElementById("rig");
      if (rig) {{
        window.scrollTo(0, Math.max(0, rig.getBoundingClientRect().top + window.scrollY - 14));
      }}
    }});
  }}
  if (backEl) {{
    backEl.addEventListener("click", function () {{
      // Keep the cluster and use-case selections, drop the card.
      history.pushState(null, "", location.pathname + location.search);
      route();
    }});
  }}
  window.addEventListener("hashchange", function () {{ route(); }});
  window.addEventListener("popstate", function () {{ route(); }});

  chipSel.value = chip;
  fillMem(mem);
  nSel.value = n;
  chipSel.addEventListener("change", function () {{
    chip = chipSel.value;
    fillMem(parseInt(memSel.value, 10));
    chipSel.classList.add("chip-glow");
    setTimeout(function () {{ chipSel.classList.remove("chip-glow"); }}, 700);
    apply();
  }});
  memSel.addEventListener("change", apply);
  nSel.addEventListener("change", apply);
  apply();
  route();
}})();
</script>

<script>
  (function () {{
    document.querySelectorAll(".eng").forEach(function (grp) {{
      var tabs = [].slice.call(grp.querySelectorAll(".eng-tab"));
      var panes = [].slice.call(grp.querySelectorAll(".eng-pane"));
      function show(t) {{
        tabs.forEach(function (x) {{ x.setAttribute("aria-selected", x === t ? "true" : "false"); }});
        var id = t.getAttribute("data-eng");
        panes.forEach(function (p) {{ p.hidden = p.getAttribute("data-eng") !== id; }});
      }}
      tabs.forEach(function (t, i) {{
        t.addEventListener("click", function () {{ grp.setAttribute("data-user-picked", "1"); show(t); }});
        t.addEventListener("keydown", function (ev) {{
          var d = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
          if (!d) return;
          ev.preventDefault();
          var next = tabs[(i + d + tabs.length) % tabs.length];
          show(next);
          next.focus();
        }});
      }});
    }});
  }})();
</script>
"""

if __name__ == "__main__":  # pragma: no cover - use tracker/build.py instead
    raise SystemExit("run: python3 tracker/build.py")
