#!/usr/bin/env python3
"""Nightly maintenance, split so a small local model only does the judgement.

The nightly job used to be one long agent session: pull, discover new models,
re-measure every ladder, refresh the Artificial Analysis ranks, re-read the
prose, validate, commit, curate the news, commit again. On a 27B local model
that was 70-90 turns and up to six hours, nearly all of it spent re-reading a
context that grew with every tool call - and most of those turns went into the
agent re-writing throwaway discovery scripts. Almost every step is mechanical,
so this script does them and leaves the agent a short, bounded work list with
one command per action.

  no LLM
    nightly.py refresh                  pull, measure --all, fetch_aa, gate, commit
                                        "chore: nightly refresh <date>", push
  work lists (nothing to do prints {"wakeAgent": false}, so no agent starts)
    nightly.py brief                    new-model candidates, tracked-issue state
                                        changes, prose to re-check after a ladder or
                                        AA change - a few of each per night
    nightly.py news-brief               candidate stories for today's rollup
  one command per agent action
    nightly.py entry <id>               one model, compact, ladder summarised
    nightly.py patch <id> <FIELD> <text>
                                        set one field: NOTE, NAME, ARCH, LICENSE,
                                        CONTEXT, CONTEXT_LABEL, BEST_ENGINE,
                                        ENGINES.<eng>.status|label|note, or
                                        ENGINES.<eng>.issues+ (append one issue key)
    nightly.py ship "<subject>"         gate, commit data/data.json, push
    nightly.py add-model <spec.json> [--candidate <repo>]
                                        tracker/add_model.py --commit --direct, then
                                        records the candidate as handled
    nightly.py decide <repo> skip|defer "<reason>"
    nightly.py ack <issue-key | prose:<id>> ...   (or --all)
    nightly.py publish-news <rollup.json>
                                        check the rollup, write it, gate, commit
                                        "chore: news rollup <date>", push

Every mutating command takes .cache/nightly/lock, so two jobs can never be
inside git at the same time. Local state lives in .cache/nightly/ (gitignored):
ledger.json (decisions on candidates, prose checks done), watch-acked.txt
(issue states already acted on), and one directory per night holding the
refresh logs and refresh.json, which `brief` reads.

Standard library only; runs on the macOS system Python (3.9).
"""
import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import NoReturn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "data.json")
WATCH = os.path.join(HERE, "watch-state.txt")
CACHE = os.path.join(ROOT, ".cache", "nightly")
LEDGER = os.path.join(CACHE, "ledger.json")
ACKED = os.path.join(CACHE, "watch-acked.txt")
PY = sys.executable
UA = "apple-llm-performance-nightly/1.0 (+https://github.com/coltonkawamura/apple-llm-performance)"
TIMEOUT = 25

WINDOW_DAYS = 14        # discovery looks at uploads this recent
DEFER_DAYS = 7          # a deferred candidate comes back after this long
STATUSES = ("works", "degraded", "blocked", "none")
AA_CATS = ("aaindex", "agentic", "coding", "terminal", "science", "knowledge",
           "hallucination", "gdpval", "industry")
PLACEHOLDERS = ("Trending on the sources polled today",
                "New text model published inside the collection window")

# Orgs whose own uploads are first-party releases. The orgs of every model
# already on the page are added at run time, so this only needs the labs the
# page has not covered yet; a wrong or dead name costs one failed request.
LAB_ORGS = [
    "Qwen", "deepseek-ai", "zai-org", "moonshotai", "MiniMaxAI", "mistralai",
    "nvidia", "google", "allenai", "meta-llama", "microsoft", "openai",
    "ibm-granite", "LiquidAI", "tencent", "baidu", "XiaomiMiMo", "stepfun-ai",
    "inclusionAI", "ByteDance-Seed", "thinkingmachines", "black-forest-labs",
    "stabilityai", "Lightricks", "Wan-AI", "HuggingFaceTB", "CohereLabs",
    "internlm", "OpenGVLab", "rednote-hilab", "meituan-longcat", "Zyphra",
    "hexgrad", "kyutai", "ResembleAI", "sesame", "ServiceNow-AI", "arcee-ai",
    "NousResearch", "swiss-ai", "upstage", "LGAI-EXAONE", "sarvamai",
    "PrimeIntellect", "ai21labs", "apple", "Salesforce", "FunAudioLLM",
]
# Third-party packagers: their uploads are quants of somebody else's model, so
# a hit is resolved to the base model it names.
PACKAGERS = {"mlx-community", "unsloth", "bartowski", "lmstudio-community",
             "ggml-org", "mradermacher", "QuantFactory", "MaziyarPanahi",
             "DevQuasar", "NexVeridian", "nightmedia", "inferencerlabs"}
GGUF_FEEDS = ("unsloth", "bartowski", "lmstudio-community", "ggml-org")
# Hugging Face pipelines that map onto the page's modalities.
KEEP_PIPES = {"text-generation", "image-text-to-text", "any-to-any",
              "text-to-image", "image-to-image", "text-to-video",
              "image-to-video", "text-to-speech", "text-to-audio",
              "automatic-speech-recognition", "audio-to-audio",
              "audio-text-to-text", "video-text-to-text"}
TEXT_PIPES = {"text-generation", "image-text-to-text", "any-to-any",
              "audio-text-to-text", "video-text-to-text"}
NOISE = re.compile(r"(\blora\b|adapter|embedding|rerank|reward|eagle|abliterat"
                   r"|uncensor|heretic|dflash|dspark|\bmtp\b|\bdraft\b)", re.I)
# Format and precision suffixes stripped before comparing names, so a quant
# upload matches the model it quantises.
SUFFIX = re.compile(r"[-_.](gguf|mlx|awq|gptq|exl2|onnx|fp\d+|nvfp\d+|mxfp\d+|bf16"
                    r"|int\d+|w\d+a\d+|[2-8]-?bits?|q[2-8](_k)?(_[msl])?|q8_0|iq\d\w*"
                    r"|ud|dwq|dq|oq\d\w*|mixed|quant|quantized|instruct|it|chat|hf|base)$", re.I)
# A candidate needs some signal before it takes a slot: two feeds, or likes.
# Anything below waits; it comes back by itself if it gains traction.
MIN_LIKES_LAB, MIN_LIKES_OTHER = 10, 50


# ------------------------------------------------------------------ helpers
def die(msg, code=1) -> NoReturn:
    print(f"nightly: {msg}", file=sys.stderr)
    raise SystemExit(code)


def today():
    return datetime.now().strftime("%Y-%m-%d")


def tail(text, n):
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def short(v, n=160):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 3] + "..."


def run(cmd, check=False, log=None, timeout=None):
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        die(f"`{' '.join(cmd)}` timed out after {timeout}s")
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd) + "\n" + r.stdout)
            if r.stderr:
                f.write("\n--- stderr ---\n" + r.stderr)
            f.write(f"\n[exit {r.returncode}]\n")
    if check and r.returncode != 0:
        die(f"`{' '.join(cmd)}` failed (exit {r.returncode}):\n"
            + tail(r.stdout + "\n" + r.stderr, 15))
    return r


def git(*args, check=True):
    return run(["git"] + list(args), check=check).stdout.strip()


def load(path=DATA):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(d):
    """Same serialisation as every other writer in tracker/, so an untouched
    record never moves in the diff."""
    tmp = DATA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, DATA)


def find_model(d, mid):
    return next((m for m in d["models"] if m["ID"] == mid), None)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


@contextlib.contextmanager
def repo_lock(wait=2700):
    """One nightly step inside git at a time, across cron jobs."""
    os.makedirs(CACHE, exist_ok=True)
    f = open(os.path.join(CACHE, "lock"), "w")
    start, said = time.time(), False
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - start > wait:
                die("another nightly step still holds .cache/nightly/lock")
            if not said:
                print("waiting for another nightly step to finish...", flush=True)
                said = True
            time.sleep(5)
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def load_ledger():
    try:
        led = load(LEDGER)
    except (OSError, ValueError):
        led = {}
    led.setdefault("candidates", {})
    led.setdefault("prose_done", {})
    return led


def save_ledger(led):
    cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    led["prose_done"] = {k: v for k, v in led["prose_done"].items() if k >= cutoff}
    os.makedirs(CACHE, exist_ok=True)
    tmp = LEDGER + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(led, f, indent=1, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(tmp, LEDGER)


def fetch(url, as_json=True):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read()
        return json.loads(raw) if as_json else raw.decode("utf-8", "replace")
    except Exception:
        return None


def parse_time(s):
    try:
        t = datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------- git
def data_changed():
    return run(["git", "diff", "--quiet", "--", "data/data.json"]).returncode != 0


def ensure_main(stash):
    """On main with a clean tree. A run that died half-way leaves edits behind;
    stashing them (never discarding) keeps one bad night from blocking the next."""
    notes = []
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        if not stash:
            die("the working tree has uncommitted changes:\n" + dirty +
                "\nship them, or `git checkout -- data/data.json` to drop them")
        git("stash", "push", "-m", f"nightly autostash {datetime.now():%Y-%m-%d %H:%M}")
        notes.append("stashed uncommitted changes an earlier run left behind "
                     "(see `git stash list`)")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        git("checkout", "main")
        notes.append(f"switched from branch {branch} back to main")
    return notes


def pull():
    r = run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    if r.returncode != 0:
        run(["git", "rebase", "--abort"])
        die("git pull --rebase origin main failed:\n" + tail(r.stdout + r.stderr, 10))


def push():
    """Push main; on a rejection rebase onto the new origin/main once and retry."""
    r = None
    for attempt in (1, 2):
        r = run(["git", "push", "origin", "main"])
        if r.returncode == 0:
            return git("rev-parse", "--short", "HEAD")
        if attempt == 1:
            p = run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
            if p.returncode != 0:
                run(["git", "rebase", "--abort"])
                break
    die("push to origin main failed; the commit is local only:\n"
        + tail(r.stdout + r.stderr, 10))


def gate():
    """validate, build, check_output - exactly what CI runs. Returns (ok, report)."""
    v = run([PY, "tracker/validate.py"], timeout=600)
    counts = tail(v.stdout, 1)
    if v.returncode != 0:
        errors = [ln for ln in v.stdout.splitlines() if ln.startswith("error")]
        return False, "validate failed (" + counts + "):\n" + "\n".join(errors[:15])
    b = run([PY, "tracker/build.py"], timeout=600)
    if b.returncode != 0:
        return False, "build failed:\n" + tail(b.stdout + b.stderr, 12)
    c = run([PY, "tools/check_output.py"], timeout=300)
    if c.returncode != 0:
        return False, "check_output failed:\n" + tail(c.stdout + c.stderr, 12)
    return True, f"gate ok (validate: {counts}; check_output: {tail(c.stdout, 1)})"


# ----------------------------------------------------------------- refresh
def ladder_change(old, new):
    """(description, structural) for one model's ladder. Structural means rungs
    appeared or vanished, which is what makes prose go stale; a resize is not."""
    old, new = old or {}, new or {}
    parts, structural = [], False
    for fam in list(dict.fromkeys(list(old) + list(new))):
        o = {r["label"]: r for r in old.get(fam) or []}
        n = {r["label"]: r for r in new.get(fam) or []}
        if o == n:
            continue
        added = [k for k in n if k not in o]
        removed = [k for k in o if k not in n]
        resized = [k for k in n if k in o and n[k] != o[k]]
        bits = []
        if fam not in old:
            bits.append("new family")
        if fam not in new:
            bits.append("family gone")
        if added:
            bits.append("+" + ", +".join(added[:4]) + (f" (+{len(added) - 4} more)" if len(added) > 4 else ""))
        if removed:
            bits.append("-" + ", -".join(removed[:4]) + (f" (+{len(removed) - 4} more)" if len(removed) > 4 else ""))
        if resized:
            bits.append(f"{len(resized)} rung(s) resized")
        structural = structural or bool(added or removed)
        parts.append(f"{fam}: " + "; ".join(bits))
    return "; ".join(parts), structural


def lost_repos(old_lad, new_lad, sources):
    """Families where a repo still listed in QUANT_SOURCES contributed rungs
    before and none now. measure.py turns a failed request into an empty file
    list, so this is almost always a network blip, not a deleted repo."""
    bad = []
    for fam, rungs in (old_lad or {}).items():
        before = {r["repo"] for r in rungs or []}
        after = {r["repo"] for r in (new_lad or {}).get(fam) or []}
        lost = sorted(r for r in before - after if r in sources)
        if lost:
            bad.append((fam, lost))
    return bad


def guard_ladders(before):
    """Re-measure a model whose ladder lost a whole repo; if it is still missing,
    keep the old family. Returns notes for the summary."""
    def suspects(d):
        out = {}
        bm = {m["ID"]: m for m in before["models"]}
        for m in d["models"]:
            b = bm.get(m["ID"])
            if not b:
                continue
            srcs = {r for repos in (m.get("QUANT_SOURCES") or {}).values() for r in repos}
            bad = lost_repos(b.get("LADDER"), m.get("LADDER"), srcs)
            if bad:
                out[m["ID"]] = bad
        return out

    first = suspects(load())
    if not first:
        return []
    for mid in first:
        run([PY, "tracker/measure.py", "--model", mid], timeout=900)
    after = load()
    still = suspects(after)
    notes = [f"{mid}: a repo came back empty on the sweep, re-measured fine"
             for mid in first if mid not in still]
    if still:
        bm = {m["ID"]: m for m in before["models"]}
        for m in after["models"]:
            if m["ID"] not in still:
                continue
            lad = m.get("LADDER") or {}
            for fam, lost in still[m["ID"]]:
                lad[fam] = bm[m["ID"]]["LADDER"][fam]
                notes.append(f"{m['ID']}: kept the old {fam} ladder - {', '.join(lost)} "
                             "returned no files twice (network, or the repo is gone: check by hand)")
            m["LADDER"] = lad
        save(after)
    return notes


def rank_changes(before, after):
    ob = {u["ID"]: u.get("RANK", []) for u in before["useCases"]}
    lines = []
    for u in after["useCases"]:
        o = {r[0]: r for r in ob.get(u["ID"], [])}
        n = {r[0]: r for r in u.get("RANK", [])}
        new = [m for m in n if m not in o]
        gone = [m for m in o if m not in n]
        moved = [f"{m} {o[m][2]}->{n[m][2]}" for m in n if m in o and o[m][2] != n[m][2]]
        parts = []
        if new:
            parts.append("new " + ", ".join(f"{m} {n[m][2]}" for m in new))
        if gone:
            parts.append("dropped " + ", ".join(gone))
        if moved:
            parts.append(", ".join(moved[:6]) + (f" (+{len(moved) - 6} more)" if len(moved) > 6 else ""))
        if parts:
            lines.append(f"{u['ID']}: " + "; ".join(parts))

    def in_aa(d):
        return {r[0] for u in d["useCases"] if u["ID"] in AA_CATS for r in u.get("RANK", [])}
    return lines, sorted(in_aa(after) - in_aa(before)), sorted(in_aa(before) - in_aa(after))


def watch_state_age_days():
    ts = git("log", "-1", "--format=%ct", "--", "tracker/watch-state.txt", check=False)
    return int((time.time() - int(ts)) // 86400) if ts.isdigit() else None


def print_refresh(s):
    head = f"nightly refresh {s['date']}: "
    if s.get("commit"):
        head += f"committed {s['commit']}"
    elif s.get("failed"):
        head += "FAILED - nothing committed"
    else:
        head += "no change, nothing committed"
    print(head)
    if s["ladders"]:
        print("ladders:")
        for mid, desc in s["ladders"].items():
            print(f"  {mid}: {desc}")
    if s["aa"]:
        print("AA ranks:")
        for ln in s["aa"]:
            print("  " + ln)
    if s["aa_entered"]:
        print("now in the AA tables: " + ", ".join(s["aa_entered"]))
    if s["aa_left"]:
        print("gone from the AA tables: " + ", ".join(s["aa_left"]))
    for n in s["guarded"]:
        print("ladder guard: " + n)
    if s.get("gate"):
        print(s["gate"])
    for n in s["notes"]:
        print("note: " + n)
    p = prose_targets(s)
    if p:
        print("prose for the maintenance agent to re-check: " + ", ".join(mid for mid, _ in p))


def cmd_refresh(a):
    date = today()
    logdir = os.path.join(CACHE, date)
    s = {"date": date, "notes": [], "ladders": {}, "prose_ladders": [], "aa": [],
         "aa_entered": [], "aa_left": [], "guarded": [], "gate": None, "commit": None}
    started = time.time()
    with repo_lock():
        s["notes"] += ensure_main(stash=True)
        pull()
        before = load()

        m = run([PY, "tracker/measure.py", "--all"], timeout=2700,
                log=os.path.join(logdir, "measure.log"))
        if m.returncode != 0:
            s["notes"].append(f"measure.py --all exited {m.returncode}: "
                              + short(tail(m.stderr or m.stdout, 2)))
        s["guarded"] = guard_ladders(before)

        f = run([PY, "tracker/fetch_aa.py"], timeout=1800,
                log=os.path.join(logdir, "fetch_aa.log"))
        if f.returncode != 0:
            s["notes"].append("fetch_aa.py refused, AA ranks left as they were: "
                              + short(tail(f.stderr or f.stdout, 2), 240))

        after = load()
        bm = {x["ID"]: x for x in before["models"]}
        for x in after["models"]:
            desc, structural = ladder_change((bm.get(x["ID"]) or {}).get("LADDER"), x.get("LADDER"))
            if desc:
                s["ladders"][x["ID"]] = desc
                if structural:
                    s["prose_ladders"].append(x["ID"])
        s["aa"], s["aa_entered"], s["aa_left"] = rank_changes(before, after)

        if data_changed():
            ok, report = gate()
            s["gate"] = report
            if not ok:
                os.makedirs(logdir, exist_ok=True)
                shutil.copy(DATA, os.path.join(logdir, "data.failed.json"))
                git("checkout", "--", "data/data.json")
                s["failed"] = True
                write_text(os.path.join(logdir, "refresh.json"), json.dumps(s, indent=1))
                print_refresh(s)
                die("gate failed; nothing committed (the failing file is in "
                    f".cache/nightly/{date}/data.failed.json)\n{report}")
            body = ["Mechanical refresh by tracker/nightly.py (no LLM)."]
            body += [f"- ladder {k}: {v}" for k, v in s["ladders"].items()]
            body += [f"- AA {ln}" for ln in s["aa"]]
            body += [f"- guard: {n}" for n in s["guarded"]]
            git("add", "data/data.json")
            git("commit", "-q", "-m", f"chore: nightly refresh {date}", "-m", "\n".join(body))
            s["commit"] = (git("rev-parse", "--short", "HEAD") + " (not pushed)"
                           if a.no_push else push() + " and pushed")

    age = watch_state_age_days()
    if age is not None and age > 7:
        s["notes"].append(
            f"tracker/watch-state.txt was last updated {age} days ago: the 'Refresh issue "
            "states' workflow is not running (GitHub disables scheduled workflows on forks "
            "until they are enabled), so issue pills and tracked-state changes are stale")
    s["seconds"] = round(time.time() - started)
    write_text(os.path.join(logdir, "refresh.json"), json.dumps(s, indent=1))
    print_refresh(s)
    return 0


# ------------------------------------------------------------------- brief
def latest_refresh():
    """The newest refresh.json written in the last 36 hours, or None."""
    if not os.path.isdir(CACHE):
        return None
    for name in sorted(os.listdir(CACHE), reverse=True):
        p = os.path.join(CACHE, name, "refresh.json")
        if os.path.exists(p) and time.time() - os.path.getmtime(p) < 36 * 3600:
            try:
                return load(p)
            except ValueError:
                return None
    return None


def prose_targets(s):
    """Models whose prose may now contradict the data: rungs appeared or vanished,
    or the model entered or left the AA tables."""
    out = []
    for mid in s.get("prose_ladders", []):
        out.append((mid, "ladder changed - " + s["ladders"].get(mid, "")))
    for mid in s.get("aa_entered", []):
        out.append((mid, "now listed in the Artificial Analysis tables"))
    for mid in s.get("aa_left", []):
        out.append((mid, "no longer in the Artificial Analysis tables"))
    seen, uniq = set(), []
    for mid, why in out:
        if mid not in seen:
            seen.add(mid)
            uniq.append((mid, why))
    return uniq


def parse_watch(text):
    out = {}
    for line in text.splitlines():
        parts = line.split("|", 2)
        if len(parts) >= 2 and parts[0].strip():
            out[parts[0].strip()] = (parts[1].strip(), parts[2].strip() if len(parts) > 2 else "")
    return out


def write_watch(states):
    write_text(ACKED, "".join(f"{k}|{st}|{lbl}\n" for k, (st, lbl) in sorted(states.items())))


def cited_by(d):
    out = {}
    for m in d["models"]:
        for eid, cell in (m.get("ENGINES") or {}).items():
            for k in cell.get("issues") or []:
                out.setdefault(k, []).append(f"{m['ID']}/{eid} ({cell.get('status')})")
    return out


def watch_changes(d):
    """Issue states that moved since they were last acknowledged. New keys and
    engine releases nothing on the page waits for are acknowledged on the spot."""
    cur = parse_watch(open(WATCH, encoding="utf-8").read())
    if not os.path.exists(ACKED):
        write_watch(cur)          # first run: baseline, nothing to report
        return []
    acked = parse_watch(open(ACKED, encoding="utf-8").read())
    cites = cited_by(d)
    engines_by_repo = {}
    for e in d["engines"]:
        if e.get("REPO"):
            engines_by_repo.setdefault(e["REPO"].lower(), []).append(e["ID"])
    items, auto = [], False
    for key, (st, label) in sorted(cur.items()):
        old = acked.get(key)
        if old is not None and old[0] == st:
            continue
        if old is None:
            acked[key], auto = (st, label), True
            continue
        if key.endswith("@release"):
            eids = engines_by_repo.get(key[:-len("@release")].lower(), [])
            waiting = [f"{m['ID']}/{eid}" for m in d["models"]
                       for eid, c in (m.get("ENGINES") or {}).items()
                       if eid in eids and c.get("status") == "degraded"
                       and "master" in (c.get("label", "") + " " + c.get("note", "")).lower()]
            if not waiting:
                acked[key], auto = (st, label), True
                continue
            items.append((key, old[0], st, "new release", waiting))
        else:
            items.append((key, old[0], st, label, cites.get(key, [])))
    if auto:
        write_watch(acked)
    return items


def norm(name):
    base = name.rsplit("/", 1)[-1].lower()
    prev = None
    while prev != base:
        prev = base
        base = SUFFIX.sub("", base)
    return re.sub(r"[^a-z0-9]", "", base)


def page_index(d):
    repos, names = set(), set()
    for m in d["models"]:
        for r in [m.get("HF")] + [q for qs in (m.get("QUANT_SOURCES") or {}).values() for q in qs]:
            if r:
                repos.add(r.lower())
                names.add(norm(r))
        names.add(norm(m.get("NAME", "")))
    names.discard("")
    return repos, names


def base_models(tags):
    """[(relation, repo)] from Hugging Face base_model:... tags."""
    out = []
    for t in tags or []:
        if not t.startswith("base_model:"):
            continue
        parts = t[len("base_model:"):].split(":")
        if len(parts) == 1 and "/" in parts[0]:
            out.append(("", parts[0]))
        elif len(parts) == 2 and "/" in parts[1]:
            out.append((parts[0], parts[1]))
    return out


def ledger_blocks(led, key):
    e = led["candidates"].get(key.lower())
    if not e:
        return False
    if e.get("decision") == "defer":
        return e.get("until", "") >= today()
    return True


class Discovery:
    """New-model candidates: what shipped recently that the page does not have
    and nobody has decided on yet. All mechanical; the agent only judges."""

    def __init__(self, d, led):
        self.d, self.led = d, led
        self.repos, self.names = page_index(d)
        orgs = {m["HF"].split("/")[0] for m in d["models"] if "/" in (m.get("HF") or "")}
        self.labs = {o.lower() for o in set(LAB_ORGS) | orgs} - {p.lower() for p in PACKAGERS}
        self.failed = []
        self._mlx = self._arch = None

    def on_page(self, repo):
        return repo.lower() in self.repos or norm(repo) in self.names

    def is_lab(self, repo):
        return repo.split("/")[0].lower() in self.labs

    def feeds(self):
        f = {"hf-trending": "sort=trendingScore&direction=-1&limit=100",
             "mlx-community": "author=mlx-community&sort=createdAt&direction=-1&limit=100"}
        for p in GGUF_FEEDS:
            f[p] = f"author={p}&sort=createdAt&direction=-1&limit=50"
        for org in sorted(set(LAB_ORGS) | {m["HF"].split("/")[0] for m in self.d["models"]
                                          if "/" in (m.get("HF") or "")}):
            if org.lower() in self.labs:
                f["lab:" + org] = f"author={urllib.parse.quote(org)}&sort=createdAt&direction=-1&limit=10"
        return f

    def collect(self, window=None, signal=True, known=True, pipes=None):
        """Candidates from every feed. window: how recent an upload must be;
        signal: drop rows below the likes/two-feed floor; known: drop models
        on the page or already decided; pipes: Hugging Face pipelines kept."""
        pipes = pipes or KEEP_PIPES
        feeds = self.feeds()
        with ThreadPoolExecutor(max_workers=8) as pool:
            got = dict(zip(feeds, pool.map(
                lambda q: fetch("https://huggingface.co/api/models?" + q), feeds.values())))
        self.failed = [k for k, v in got.items() if v is None]
        cutoff = datetime.now(timezone.utc) - (window or timedelta(days=WINDOW_DAYS))
        cands = {}
        for src, rows in got.items():
            for m in rows or []:
                rid = m.get("id") or m.get("modelId") or ""
                created = parse_time(m.get("createdAt"))
                if "/" not in rid or not created or created < cutoff:
                    continue
                pipe = m.get("pipeline_tag")
                if pipe and pipe not in pipes:
                    continue
                author = rid.split("/")[0]
                bases = base_models(m.get("tags"))
                quant = None
                if author in PACKAGERS or any(rel == "quantized" for rel, _ in bases):
                    pick = ([b for rel, b in bases if rel == "quantized"]
                            or [b for rel, b in bases if rel == ""]
                            or [b for rel, b in bases if rel == "finetune"])
                    if not pick:
                        continue
                    key, quant = pick[0], rid
                else:
                    derived = [b for rel, b in bases if rel in ("finetune", "merge", "adapter")]
                    if derived and derived[0].split("/")[0].lower() != author.lower() \
                            and author.lower() not in self.labs:
                        continue        # a third party's fine-tune or merge
                    key = rid
                if NOISE.search(key) or (quant and quant.lower() in self.repos):
                    continue
                if known and (self.on_page(key) or ledger_blocks(self.led, key)):
                    continue
                if key.split("/")[0] in PACKAGERS:
                    continue            # a packager's re-upload, not a release
                e = cands.setdefault(key, {"sources": set(), "likes": 0, "downloads": 0,
                                           "quants": set()})
                e["sources"].add(src.split(":")[0])
                e["likes"] = max(e["likes"], m.get("likes") or 0)
                e["downloads"] = max(e["downloads"], m.get("downloads") or 0)
                if quant:
                    e["quants"].add(quant)
        # One row per model: a lab's own -GGUF / -FP8 upload folds into the model.
        merged = {}
        for key in sorted(cands, key=lambda k: (len(k.rsplit("/", 1)[-1]), k)):
            group = (key.split("/")[0].lower(), norm(key))
            if group in merged:
                e, m = merged[group][1], cands[key]
                e["sources"] |= m["sources"]
                e["quants"] |= m["quants"] | {key}
                e["likes"] = max(e["likes"], m["likes"])
                e["downloads"] = max(e["downloads"], m["downloads"])
            else:
                merged[group] = (key, cands[key])
        out = {}
        for key, e in merged.values():
            floor = MIN_LIKES_LAB if self.is_lab(key) else MIN_LIKES_OTHER
            if not signal or len(e["sources"]) >= 2 or e["likes"] >= floor:
                out[key] = e
        return out

    def mlx_classes(self):
        if self._mlx is None:
            rows = fetch("https://api.github.com/repos/ml-explore/mlx-lm/contents/mlx_lm/models")
            self._mlx = ({r["name"][:-3] for r in rows if r.get("name", "").endswith(".py")}
                         if isinstance(rows, list) else set())
        return self._mlx

    def llama_arch(self):
        if self._arch is None:
            self._arch = fetch("https://raw.githubusercontent.com/ggml-org/llama.cpp/"
                               "master/src/llama-arch.cpp", as_json=False) or ""
        return self._arch

    def precheck(self, info, key, quants):
        pipe = info.get("pipeline_tag")
        mt = (info.get("config") or {}).get("model_type") or ""
        bits = []
        if pipe in TEXT_PIPES or not pipe:
            if mt:
                variants = {mt, mt.replace("_", ""), mt.replace("_", "-")}
                mlx = self.mlx_classes()
                arch = self.llama_arch()
                bits.append("mlx-lm class " + ("?" if not mlx else ("yes" if variants & mlx else "no")))
                bits.append("llama.cpp arch " + ("?" if not arch else
                            ("yes" if any(f'"{v}"' in arch for v in variants) else "no")))
            else:
                bits.append("no model_type in config.json")
        else:
            bits.append("media model: confirm in the mflux / MLX-Audio / MLX-Video / "
                        "DiffusionKit README tables")
        base = key.rsplit("/", 1)[-1]
        found = {"mlx": set(), "gguf": set()}
        for q in quants:
            low = q.lower()
            if "gguf" in low:
                found["gguf"].add(q)
            elif "mlx" in low:
                found["mlx"].add(q)
        for fmt in ("mlx", "gguf"):
            rows = fetch("https://huggingface.co/api/models?" + urllib.parse.urlencode(
                {"search": base, "filter": fmt, "sort": "downloads", "direction": "-1",
                 "limit": "8"})) or []
            for r in rows:
                rid = r.get("id") or ""
                if norm(rid) == norm(key):
                    found[fmt].add(rid)
        for fmt, label in (("mlx", "MLX quants"), ("gguf", "GGUF")):
            reps = sorted(found[fmt], key=lambda r: (r.split("/")[0] not in PACKAGERS, r))
            bits.append(f"{label}: " + (", ".join(reps[:3]) + (f" (+{len(reps) - 3})" if len(reps) > 3 else "")
                                        if reps else "none found"))
        return " | ".join(bits), mt

    def pick(self, limit):
        cands = self.collect()
        order = sorted(cands, key=lambda k: (not self.is_lab(k), -len(cands[k]["sources"]),
                                             -cands[k]["likes"], k))
        shown, examined = [], 0
        for key in order:
            if len(shown) >= limit or examined >= limit * 5:
                break
            examined += 1
            e = cands[key]
            info = fetch("https://huggingface.co/api/models/" + key)
            if not isinstance(info, dict):
                shown.append((key, e, None, "model page not reachable (renamed, private or gated?)", ""))
                continue
            pipe = info.get("pipeline_tag")
            if pipe and pipe not in KEEP_PIPES:
                self.auto(key, f"pipeline {pipe} is outside the page's modalities")
                continue
            author = key.split("/")[0].lower()
            derived = [b for rel, b in base_models(info.get("tags"))
                       if rel in ("finetune", "merge", "adapter")]
            if derived and derived[0].split("/")[0].lower() != author and author not in self.labs:
                self.auto(key, f"third-party fine-tune/merge of {derived[0]}")
                continue
            if any(rel == "quantized" for rel, _ in base_models(info.get("tags"))):
                self.auto(key, "a quantisation, not a model")
                continue
            pre, mt = self.precheck(info, key, e["quants"])
            shown.append((key, e, info, pre, mt))
        return shown, max(0, len(order) - examined)

    def auto(self, key, reason):
        self.led["candidates"][key.lower()] = {"decision": "skip", "reason": "auto: " + reason,
                                               "date": today()}

    def fresh_language_models(self, hours=36, limit=8):
        """Language models first published in the last `hours`, for the rollup's
        newModels: a quant uploaded today of an older model does not count, so
        each candidate's own model page decides."""
        cands = self.collect(window=timedelta(hours=hours), signal=False, known=False,
                             pipes=TEXT_PIPES)
        order = sorted(cands, key=lambda k: (not self.is_lab(k), -len(cands[k]["sources"]),
                                             -cands[k]["likes"], k))
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        out = []
        for key in order[:limit * 3]:
            if len(out) >= limit:
                break
            info = fetch("https://huggingface.co/api/models/" + key)
            if not isinstance(info, dict):
                continue
            created = parse_time(info.get("createdAt"))
            pipe = info.get("pipeline_tag")
            if not created or created < cutoff or (pipe and pipe not in TEXT_PIPES):
                continue
            author = key.split("/")[0].lower()
            bases = base_models(info.get("tags"))
            derived = [b for rel, b in bases if rel in ("finetune", "merge", "adapter")]
            if any(rel == "quantized" for rel, _ in bases) or (
                    derived and derived[0].split("/")[0].lower() != author
                    and author not in self.labs):
                continue
            out.append((key, info, cands[key]))
        return out


def describe_candidate(i, key, e, info, pre, mt):
    lines = []
    if info:
        lic = next((t[8:] for t in info.get("tags") or [] if t.startswith("license:")), "?")
        params = (info.get("safetensors") or {}).get("total")
        facts = [info.get("pipeline_tag") or "no pipeline tag",
                 "created " + (info.get("createdAt") or "?")[:10],
                 f"{info.get('likes', 0)} likes", f"{info.get('downloads', 0)} downloads", lic]
        if params:
            facts.append(f"{params / 1e9:.1f}B params")
        if mt:
            facts.append("model_type " + mt)
        if info.get("gated"):
            facts.append("gated")
        lines.append(f"M{i} {key}  [{', '.join(facts)}]")
    else:
        lines.append(f"M{i} {key}  [{pre}]")
        pre = ""
    lines.append(f"   seen in: {', '.join(sorted(e['sources']))}")
    if pre:
        lines.append(f"   pre-check: {pre}")
    lines.append(f"   https://huggingface.co/{key}")
    return lines


def cmd_brief(a):
    d, led = load(), load_ledger()
    states = watch_changes(d)[:a.max_states]
    s = latest_refresh()
    done = set(led["prose_done"].get(s["date"], [])) if s else set()
    prose = [(mid, why) for mid, why in (prose_targets(s) if s else [])
             if mid not in done and find_model(d, mid)][:a.max_prose]
    picks, queued, failed = [], 0, []
    if not a.no_discovery:
        disc = Discovery(d, led)
        picks, queued = disc.pick(a.max_models)
        failed = disc.failed
    save_ledger(led)            # auto-skips found during discovery

    if not (states or prose or picks):
        print(json.dumps({"wakeAgent": False}))
        return 0

    out = [f"Maintenance work list for {today()} (tracker/nightly.py brief). "
           "Work these items in order; nothing else."]
    if states:
        out.append("\nTRACKED ISSUE STATE CHANGES - resolve each with `nightly.py ack <key>`")
        for i, (key, old, new, label, cites) in enumerate(states, 1):
            out.append(f"S{i} {key}: {old} -> {new}  ({short(label, 90)})")
            out.append("   cited by: " + (", ".join(cites[:6]) if cites else "nothing on the page"))
    if prose:
        out.append("\nPROSE TO RE-CHECK - resolve each with `nightly.py ack prose:<id>`")
        for i, (mid, why) in enumerate(prose, 1):
            out.append(f"P{i} {mid}: {short(why, 200)}")
    if picks:
        out.append("\nNEW-MODEL CANDIDATES - resolve each with `nightly.py add-model <spec> "
                   "--candidate <repo>` or `nightly.py decide <repo> skip|defer \"<reason>\"`")
        for i, row in enumerate(picks, 1):
            out += describe_candidate(i, *row)
        if queued:
            out.append(f"({queued} more candidates wait for later nights)")
    if failed:
        out.append(f"\n(discovery: {len(failed)} feed(s) failed to load: {', '.join(failed[:5])})")
    print("\n".join(out))
    return 0


# -------------------------------------------------------------- news-brief
def models_added_since(hours):
    rev = git("rev-list", "-1", f"--before={hours} hours ago", "HEAD", check=False)
    if not rev:
        return []
    try:
        old = json.loads(git("show", f"{rev}:data/data.json"))
    except (ValueError, SystemExit):
        return []
    have = {m["ID"] for m in old["models"]}
    return [m for m in load()["models"] if m["ID"] not in have]


def cmd_news_brief(a):
    d, date = load(), today()
    news = d.get("news") or []
    if news and news[0].get("date") == date and not a.force:
        print(json.dumps({"wakeAgent": False}))
        return 0
    sys.path.insert(0, HERE)
    import collect_news as cn  # noqa: E402  (feeds and their parsers live there)

    with ThreadPoolExecutor(max_workers=4) as pool:
        fut = {"reddit": pool.submit(cn.reddit_new, 20),
               "trending": pool.submit(cn.hf_trending_text, 12),
               "mlx": pool.submit(cn.hf_mlx_newest, 12),
               "ollama": pool.submit(cn.ollama_new, 10)}
        got = {k: f.result() for k, f in fut.items()}
    fresh = Discovery(d, {"candidates": {}, "prose_done": {}}).fresh_language_models()
    added = models_added_since(36)
    prev = next((r for r in news if r.get("date") != date), None)
    s = latest_refresh()

    out = [f"News candidates for the {date} rollup (tracker/nightly.py news-brief)."]
    if added:
        out.append("\nADDED TO THE PAGE IN THE LAST 36 HOURS (strongest stories; the page's own note is the source)")
        for m in added:
            out.append(f"A {m['NAME']} ({m.get('HF') or m['ID']}, {m['MODALITY']}, best engine "
                       f"{m.get('BEST_ENGINE')}): {short(m.get('NOTE', ''), 260)}")
            out.append(f"   https://huggingface.co/{m['HF']}" if m.get("HF") else "")
    if s and (s.get("aa_entered") or s.get("prose_ladders")):
        out.append("\nTONIGHT'S DATA REFRESH")
        if s.get("aa_entered"):
            out.append("   newly ranked by Artificial Analysis: " + ", ".join(s["aa_entered"]))
        for mid in s.get("prose_ladders", []):
            out.append(f"   {mid} quants changed: {short(s['ladders'].get(mid, ''), 160)}")
    out.append("\nNEW LANGUAGE MODELS, FIRST PUBLISHED IN THE LAST 36 HOURS (the only candidates "
               "for newModels; already checked for date, fine-tunes and re-uploads)")
    for i, (key, info, e) in enumerate(fresh, 1):
        lic = next((t[8:] for t in info.get("tags") or [] if t.startswith("license:")), "?")
        params = (info.get("safetensors") or {}).get("total")
        out.append(f"N{i} {key}  [created {(info.get('createdAt') or '?')[:16].replace('T', ' ')} UTC, "
                   f"{info.get('pipeline_tag') or 'no pipeline tag'}, {lic}"
                   + (f", {params / 1e9:.1f}B params" if params else "")
                   + f", {info.get('likes', 0)} likes]  https://huggingface.co/{key}")
    if not fresh:
        out.append("   (none - newModels is [] today)")
    out.append("\nR/LOCALLLAMA NEW (titles only - a community post is a lead, never a source)")
    out += [f"R{i} {short(c['title'], 140)}  {c['url']}" for i, c in enumerate(got["reddit"], 1)] \
        or ["(feed unavailable)"]
    out.append("\nHUGGING FACE TRENDING (text)")
    out += [f"T{i} {c['title']}  created {c['created'][:10]}, {c['downloads']:,} downloads, "
            f"{c['likes']} likes" for i, c in enumerate(got["trending"], 1)] or ["(feed unavailable)"]
    out.append("\nMLX-COMMUNITY NEWEST (someone already converted it for Apple silicon)")
    out += [f"X{i} {c['title']}  created {c['created'][:10]}"
            for i, c in enumerate(got["mlx"], 1)] or ["(feed unavailable)"]
    out.append("\nOLLAMA NEWEST")
    out.append("   " + (", ".join(c["title"] for c in got["ollama"]) or "(feed unavailable)"))
    if prev:
        out.append(f"\nPREVIOUS ROLLUP ({prev['date']}) - do not repeat these unless something new happened")
        out += [f"   - {short(t[0], 120)}" for t in prev.get("top", [])]
    out.append("\nROLLUP FORMAT (publish-news checks all of it; the date is filled in for you)")
    out.append('{"top": [["<title>", "<one paragraph, 60+ characters>", "https://..."]],   '
               "<- 1 to 3 rows")
    out.append(' "newModels": [["<name>", "<what it is and what it costs to run>", "https://..."]]}'
               "   <- 0 to 10 rows, only from the N lines")
    print("\n".join(ln for ln in out if ln != ""))
    return 0


# ------------------------------------------------------------ agent actions
def ladder_summary(rungs):
    if not rungs:
        return "nothing measured"
    gbs = [r["gb"] for r in rungs]
    labels = [r["label"] for r in rungs]
    return (f"{len(rungs)} rung(s), {min(gbs):g}-{max(gbs):g} GB: " + ", ".join(labels[:6])
            + (f" (+{len(labels) - 6})" if len(labels) > 6 else ""))


def cmd_entry(a):
    d = load()
    m = find_model(d, a.id)
    if not m:
        die(f"no model {a.id!r}; ids: {', '.join(x['ID'] for x in d['models'])}")
    out = {k: v for k, v in m.items() if a.ladder or k != "LADDER"}
    if not a.ladder:
        out["LADDER (summary)"] = {f: ladder_summary(r) for f, r in (m.get("LADDER") or {}).items()}
    out["ranked in"] = [f"{u['ID']}: {r[1]} {r[2]}" for u in d["useCases"]
                        for r in u.get("RANK", []) if r[0] == a.id]
    if a.fields:
        want = [w.strip() for w in a.fields.split(",") if w.strip()]
        out = {k: v for k, v in out.items() if k == "ID" or any(k.startswith(w) for w in want)}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


EDITABLE = ("NAME", "ARCH", "LICENSE", "CONTEXT", "CONTEXT_LABEL", "NOTE", "BEST_ENGINE")


def issue_keys(d):
    return {f"{it['repo']}#{n}" for it in d["issues"] for n in it.get("issues", {})}


def cmd_patch(a):
    with repo_lock():
        d = load()
        m = find_model(d, a.id)
        if not m:
            die(f"no model {a.id!r}")
        field, text = a.field, a.text
        if field in EDITABLE:
            if field == "BEST_ENGINE" and text not in (m.get("ENGINES") or {}):
                die(f"BEST_ENGINE must be one of {a.id}'s cells: {', '.join(m.get('ENGINES') or {})}")
            old = m.get(field)
            m[field] = text
        elif field.startswith("ENGINES."):
            parts = field.split(".")
            if len(parts) != 3:
                die("engine fields look like ENGINES.<engine>.note")
            _, eid, key = parts
            cell = (m.get("ENGINES") or {}).get(eid)
            if cell is None:
                die(f"{a.id} has no {eid} cell (cells: {', '.join(m.get('ENGINES') or {})}); "
                    "adding a cell is an add-model update")
            if key == "issues+":
                if text not in issue_keys(d):
                    die(f"{text} is not in the issues section of data/data.json")
                old = list(cell.get("issues") or [])
                if text in old:
                    print(f"{a.id}/{eid} already cites {text}; nothing changed")
                    return 0
                cell.setdefault("issues", []).append(text)
            elif key in ("status", "label", "note"):
                if key == "status" and text not in STATUSES:
                    die(f"status must be one of {', '.join(STATUSES)}")
                old = cell.get(key)
                cell[key] = text
            else:
                die("engine fields are status, label, note, or issues+")
        else:
            die(f"cannot patch {field}; editable: {', '.join(EDITABLE)}, "
                "ENGINES.<eng>.status|label|note|issues+ (LADDER, ID and QUANT_SOURCES are not "
                "hand-edited: see AGENTS.md)")
        save(d)
    print(f"patched {a.id} {field}\n  was: {short(old)}\n  now: {short(text)}\n"
          f"next: python3 tracker/nightly.py ship \"fix({a.id}): <what changed>\"")
    return 0


def cmd_ship(a):
    with repo_lock():
        if git("rev-parse", "--abbrev-ref", "HEAD") != "main":
            die("not on main; `git checkout main` first")
        if not data_changed():
            print("data/data.json has no changes; nothing to ship")
            return 0
        stat = run(["git", "diff", "--numstat", "--", "data/data.json"]).stdout.split()
        moved = int(stat[0]) + int(stat[1]) if len(stat) >= 2 and stat[0].isdigit() else 0
        if moved > 400 and not a.force:
            die(f"the diff moves {moved} lines - that is a reformat or a mistake, not a "
                "patch. Inspect `git diff --stat`, or `git checkout -- data/data.json`")
        ok, report = gate()
        if not ok:
            die(report + "\nfix what it names and run ship again, or "
                "`git checkout -- data/data.json` to drop the change")
        git("add", "data/data.json")
        msg = ["-m", a.subject] + (["-m", a.body] if a.body else [])
        git("commit", "-q", *msg)
        sha = git("rev-parse", "--short", "HEAD") + " (not pushed)" if a.no_push else push()
    print(f"shipped {sha}: {a.subject} ({moved} lines). {report}")
    return 0


def cmd_add_model(a):
    try:
        spec = load(a.spec)
    except (OSError, ValueError) as e:
        die(f"cannot read {a.spec}: {e}")
    if not isinstance(spec, dict) or not spec.get("ID"):
        die("the spec must be a JSON object with an ID")
    mid = spec["ID"]
    if not find_model(load(), mid):
        need = ["NAME", "MODALITY", "ARCH", "LICENSE", "CONTEXT", "HF", "PARAMS_B", "NOTE",
                "SOURCES", "ENGINES", "BEST_ENGINE"] + (["KV"] if spec.get("MODALITY") == "text" else [])
        missing = [k for k in need if spec.get(k) in (None, "", [], {})]
        if missing:
            die(f"a new model needs {', '.join(missing)} in the spec (copy the shape of "
                "`nightly.py entry <similar-id>`; LADDER is measured, never supplied)")
    with repo_lock():
        dirty = git("status", "--porcelain", "--untracked-files=no")
        if dirty:
            die("uncommitted changes would block add_model.py:\n" + dirty +
                "\nship them first, or `git checkout -- data/data.json` to drop them")
        r = run([PY, "tracker/add_model.py", "--in", os.path.abspath(a.spec), "--commit", "--direct"],
                timeout=1800, log=os.path.join(CACHE, today(), f"add_model-{mid}.log"))
        if r.returncode != 0:
            restored = data_changed()
            if restored:
                git("checkout", "--", "data/data.json")
            out = r.stdout + "\n" + r.stderr
            errs = [ln for ln in out.splitlines()
                    if ln.startswith(("error", "add_model:", "unknown"))
                    or re.match(r"^[A-Za-z_.]*(Error|Exception)\b", ln)]
            die("add_model.py failed" + (" (data/data.json restored)" if restored else "") + ":\n"
                + "\n".join(errs[:15] or [tail(out, 15)])
                + "\nfix the spec and run add-model again")
    led = load_ledger()
    for key in filter(None, [a.candidate, spec.get("HF")]):
        led["candidates"][key.lower()] = {"decision": "added", "id": mid, "date": today()}
    save_ledger(led)
    print(tail(r.stdout, 4))
    print(f"added {mid}: {git('log', '-1', '--format=%h %s')}")
    return 0


def cmd_decide(a):
    led = load_ledger()
    e = {"decision": a.decision, "reason": a.reason, "date": today()}
    if a.decision == "defer":
        e["until"] = (datetime.now() + timedelta(days=a.days)).strftime("%Y-%m-%d")
    led["candidates"][a.repo.lower()] = e
    save_ledger(led)
    print(f"{a.repo}: {a.decision}" + (f" until {e['until']}" if "until" in e else "")
          + f" - {a.reason}")
    return 0


def cmd_ack(a):
    cur = parse_watch(open(WATCH, encoding="utf-8").read())
    acked = parse_watch(open(ACKED, encoding="utf-8").read()) if os.path.exists(ACKED) else {}
    led = load_ledger()
    s = latest_refresh()
    rdate = s["date"] if s else today()
    done = led["prose_done"].setdefault(rdate, [])
    if a.all:
        acked = dict(cur)
        done += [mid for mid, _ in (prose_targets(s) if s else []) if mid not in done]
    for key in a.keys:
        if key.startswith("prose:"):
            if key[6:] not in done:
                done.append(key[6:])
        elif key in cur:
            acked[key] = cur[key]
        else:
            die(f"{key} is not a tracked issue key (tracker/watch-state.txt) nor prose:<id>")
    write_watch(acked)
    save_ledger(led)
    print("acknowledged: " + (", ".join(a.keys) if a.keys else "everything"))
    return 0


def check_rows(rows, name, lo, hi):
    if not isinstance(rows, list):
        return [f"{name} must be a list of [title, summary, url] rows"]
    problems = []
    if not lo <= len(rows) <= hi:
        problems.append(f"{name} has {len(rows)} row(s); it needs {lo} to {hi}")
    for i, r in enumerate(rows, 1):
        if not (isinstance(r, list) and len(r) == 3 and all(isinstance(x, str) for x in r)):
            problems.append(f"{name}[{i}] must be [title, summary, url]: three strings")
            continue
        t, s, u = (x.strip() for x in r)
        if not t:
            problems.append(f"{name}[{i}] has no title")
        if len(s) < 60:
            problems.append(f"{name}[{i}] summary is {len(s)} characters; write a real "
                            "paragraph a reader can learn from without the link (60+)")
        if any(p in s for p in PLACEHOLDERS):
            problems.append(f"{name}[{i}] still carries the mechanical placeholder summary")
        if not u.startswith(("https://", "http://")):
            problems.append(f"{name}[{i}] url must start with https://")
    return problems


def cmd_publish_news(a):
    try:
        rollup = load(a.file)
    except (OSError, ValueError) as e:
        die(f"cannot read {a.file} as JSON: {e}")
    if not isinstance(rollup, dict):
        die('the rollup must be a JSON object: {"top": [...], "newModels": [...]}')
    date = today()
    if rollup.get("date") not in (None, date):
        print(f"note: date {rollup.get('date')!r} replaced with today, {date}")
    rollup = {"date": date, "top": rollup.get("top"), "newModels": rollup.get("newModels") or []}
    problems = check_rows(rollup["top"], "top", 1, 3) + check_rows(rollup["newModels"], "newModels", 0, 10)
    if problems:
        die("rollup rejected, nothing written:\n- " + "\n- ".join(problems))
    if a.dry_run:
        print("rollup ok (dry run, nothing written): "
              + "; ".join(t[0] for t in rollup["top"]))
        return 0
    sys.path.insert(0, HERE)
    import collect_news as cn  # noqa: E402
    with repo_lock():
        notes = ensure_main(stash=True)
        pull()
        d = load()
        cn.write_rollup(d, rollup)
        save(d)
        ok, report = gate()
        if not ok:
            git("checkout", "--", "data/data.json")
            die("gate failed, rollup not committed (data/data.json restored):\n" + report)
        git("add", "data/data.json")
        git("commit", "-q", "-m", f"chore: news rollup {date}", "-m",
            "\n".join(f"- top: {t[0]}" for t in rollup["top"])
            + "".join(f"\n- new: {t[0]}" for t in rollup["newModels"]))
        sha = git("rev-parse", "--short", "HEAD") + " (not pushed)" if a.no_push else push()
    for n in notes:
        print("note: " + n)
    print(f"published the {date} rollup: {sha}. {report}")
    return 0


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("refresh", help="no-LLM nightly refresh")
    p.add_argument("--no-push", action="store_true")
    p.set_defaults(fn=cmd_refresh)

    p = sub.add_parser("brief", help="work list for the maintenance agent")
    p.add_argument("--max-models", type=int, default=2)
    p.add_argument("--max-states", type=int, default=3)
    p.add_argument("--max-prose", type=int, default=3)
    p.add_argument("--no-discovery", action="store_true")
    p.set_defaults(fn=cmd_brief)

    p = sub.add_parser("news-brief", help="candidate stories for today's rollup")
    p.add_argument("--force", action="store_true", help="even if today's rollup exists")
    p.set_defaults(fn=cmd_news_brief)

    p = sub.add_parser("entry", help="print one model, ladder summarised")
    p.add_argument("id")
    p.add_argument("--ladder", action="store_true", help="include the full ladder")
    p.add_argument("--fields", help="comma-separated keys to keep, e.g. NOTE,ENGINES")
    p.set_defaults(fn=cmd_entry)

    p = sub.add_parser("patch", help="set one field of one model")
    p.add_argument("id")
    p.add_argument("field")
    p.add_argument("text")
    p.set_defaults(fn=cmd_patch)

    p = sub.add_parser("ship", help="gate, commit data/data.json, push")
    p.add_argument("subject")
    p.add_argument("--body")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--force", action="store_true", help="allow a diff of 400+ lines")
    p.set_defaults(fn=cmd_ship)

    p = sub.add_parser("add-model", help="add_model.py --commit --direct, then record it")
    p.add_argument("spec")
    p.add_argument("--candidate", help="the candidate repo from the brief")
    p.set_defaults(fn=cmd_add_model)

    p = sub.add_parser("decide", help="record a decision on a candidate")
    p.add_argument("repo")
    p.add_argument("decision", choices=("skip", "defer"))
    p.add_argument("reason")
    p.add_argument("--days", type=int, default=DEFER_DAYS)
    p.set_defaults(fn=cmd_decide)

    p = sub.add_parser("ack", help="acknowledge issue-state changes or prose checks")
    p.add_argument("keys", nargs="*")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_ack)

    p = sub.add_parser("publish-news", help="check, write, gate, commit and push a rollup")
    p.add_argument("file")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-push", action="store_true")
    p.set_defaults(fn=cmd_publish_news)

    a = ap.parse_args()
    if a.cmd == "ack" and not (a.keys or a.all):
        ap.error("ack needs at least one key, or --all")
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
