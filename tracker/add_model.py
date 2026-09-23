#!/usr/bin/env python3
"""Add or update one model in data/data.json, end to end.

This is the fast path for "update the data file with the details of this
model". Feed it a compact description of the model and it does everything the
rest of the pipeline does, in order:

    1. writes (or updates) the model's entry in data/data.json
    2. appends its curated ranks to the use cases it names
    3. measures its quant ladder from Hugging Face (tracker/measure.py)
    4. validates, builds and checks the page - exactly what CI runs
    5. commits and pushes, so the site gets rebuilt

Nothing here is clever. Policy - what a model entry must contain - is enforced
by tracker/validate.py, which this calls and refuses to bypass. The script only
does the mechanics: merging the entry, keeping the file's formatting, and
driving the pipeline in the right order.

The input is a JSON file (or - for stdin), one object:

    {
      "ID": "qwen38",                     lowercase alphanumeric; the deep-link
      "NAME": "Qwen3.8-27B",
      "MODALITY": "text",                 text | image | video | audio
      "ARCH": "Dense 27.8B - ...",
      "LICENSE": "Apache-2.0",
      "CONTEXT": "262k (to 1M)",
      "HF": "Qwen/Qwen3.8-27B",
      "PARAMS_B": 27.8,
      "NOTE": "engine-neutral summary, >= 80 chars to avoid a validator warning",
      "SOURCES": [["Model card", "https://huggingface.co/Qwen/Qwen3.8-27B"]],
      "SCORES": {"coding": [["SWE-bench Pro", "61.7"]]},
      "BEST_ENGINE": "llamacpp",
      "ENGINES": {"llamacpp": {"status": "works", "label": "Runs",
                               "note": "...", "issues": []}},
      "QUANT_SOURCES": {"gguf": ["unsloth/Qwen3.8-27B-GGUF"],
                        "mlx": ["mlx-community/Qwen3.8-27B-4bit"]},
      "KV": {"bytes_per_token": 65536, "max_context": 262144,
             "derivation": "16 of 64 layers full attention"},
      "ranks": [["coding", "SWE-bench Pro", "61.7"]]
    }

Everything except ID/NAME/MODALITY/NOTE/SOURCES is optional in the input;
validate.py decides what the finished entry still needs. `KV` is required for
text models. `LADDER` is never taken from the input - step 3 measures it.

    python3 tracker/add_model.py --in model.json            # dry-run: show the plan
    python3 tracker/add_model.py --in model.json --commit   # write, measure, build,
                                                            # branch, commit, push,
                                                            # open a PR
    python3 tracker/add_model.py --in model.json --commit --direct
                                                            # same, but push to main
                                                            # (the site deploys at once)

Existing model id = update in place: the supplied fields overwrite, `ENGINES`
cells are replaced per engine, `SCORES` merged per category, and `ranks` are
appended only where the model is not already listed.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "data.json")

ID_RE = re.compile(r"^[a-z][a-z0-9]*$")


def die(msg):
    print(f"add_model: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_data():
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)


def save_data(d):
    tmp = DATA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, DATA)


def find_model(d, mid):
    for m in d["models"]:
        if m["ID"] == mid:
            return m
    return None


def apply_entry(d, spec, dry):
    """Merge spec into d. Returns (entry, action, rank_notes)."""
    mid = spec["ID"]
    action = "add"
    m = find_model(d, mid)
    if m is not None:
        action = "update"
        for k, v in spec.items():
            if k in ("ENGINES", "SCORES", "ranks", "LADDER"):
                continue
            m[k] = v
    else:
        m = {k: v for k, v in spec.items()
             if k not in ("ranks", "LADDER") or v is not None}
        d["models"].append(m)

    # engine cells replace wholesale - a cell is one claim, not a patch
    for eid, cell in (spec.get("ENGINES") or {}).items():
        m.setdefault("ENGINES", {})[eid] = {
            "status": cell["status"], "label": cell.get("label", ""),
            "note": cell.get("note", ""), "issues": list(cell.get("issues", []))}
    # scores merge per category; a supplied category replaces its rows
    for cat, rows in (spec.get("SCORES") or {}).items():
        m.setdefault("SCORES", {})[cat] = [list(r) for r in rows]

    rank_notes = []
    for rank in spec.get("ranks") or []:
        if not (isinstance(rank, list) and len(rank) == 3):
            die(f"ranks entry {rank!r} must be [use_case_id, metric, value]")
        uc_id, metric, value = rank
        uc = next((u for u in d["useCases"] if u["ID"] == uc_id), None)
        if uc is None:
            die(f"ranks names use case {uc_id!r}, which does not exist in data/data.json")
        if any(r[0] == mid for r in uc["RANK"]):
            rank_notes.append(f"ranks: {uc_id} already lists {mid}; left untouched")
        elif dry:
            rank_notes.append(f"ranks: would append {mid} to {uc_id} RANK")
        else:
            uc["RANK"].append([mid, metric, value])
            rank_notes.append(f"ranks: appended {mid} to {uc_id} RANK "
                              "(at the end - reordering is editorial, not mechanical)")
    return m, action, rank_notes


def run(step, cmd):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        die(f"step {step!r} failed; nothing was committed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="spec", required=True,
                    help="path to the model's JSON description, or '-' for stdin")
    ap.add_argument("--commit", action="store_true",
                    help="write the file and drive measure/validate/build/commit/push")
    ap.add_argument("--direct", action="store_true",
                    help="with --commit: push to main so the site deploys at once "
                         "(default is branch + pull request)")
    ap.add_argument("--branch", default=None, help="branch name (default: data/<id>)")
    ap.add_argument("--skip-measure", action="store_true",
                    help="do not re-measure the ladder (no network / no QUANT_SOURCES)")
    ap.add_argument("--no-push", action="store_true",
                    help="with --commit: stop after the local commit")
    a = ap.parse_args()

    if a.spec == "-":
        spec = json.load(sys.stdin)
    else:
        with open(a.spec, encoding="utf-8") as f:
            spec = json.load(f)
    if not isinstance(spec, dict) or "ID" not in spec:
        die("the input must be a JSON object with an 'ID'")
    mid = spec["ID"]
    if not ID_RE.match(mid):
        die(f"ID {mid!r} must be lowercase alphanumeric - it is a deep-link and never changes")
    if "LADDER" in spec and spec["LADDER"]:
        die("LADDER is measured by tracker/measure.py, never supplied; drop it from the input")
    d = load_data()
    existing_kv = (find_model(d, mid) or {}).get("KV")
    if spec.get("MODALITY", "text") == "text" and not (spec.get("KV") or existing_kv):
        print("note: text models need a KV figure; validate.py will flag it if one is missing")

    m, action, rank_notes = apply_entry(d, spec, dry=not a.commit)

    print(f"\nadd_model: {action} model {mid!r} ({spec.get('NAME', '?')})")
    for n in rank_notes:
        print("  " + n)
    srcs = (m or {}).get("QUANT_SOURCES") or {}
    print(f"  ladder: will measure {', '.join(f'{k}: {len(v)} repo(s)' for k, v in srcs.items()) or 'nothing (no QUANT_SOURCES)'}")

    if not a.commit:
        print("\ndry run: nothing written. Re-run with --commit to apply and deploy.")
        return 0

    # git first, so the edit lands on a clean branch cut from current main:
    # fetch, branch (or main + rebase for --direct). If the working tree has
    # uncommitted changes to data/data.json the checkout refuses, and that is
    # the right behaviour - commit or stash them first.
    run("git fetch", ["git", "fetch", "origin", "main"])
    if a.direct:
        run("git checkout main", ["git", "checkout", "main"])
        run("git rebase origin/main", ["git", "rebase", "origin/main"])
        target = "main"
    else:
        branch = a.branch or f"data/{mid}"
        run("git checkout -B " + branch + " origin/main",
            ["git", "checkout", "-B", branch, "origin/main"])
        target = branch

    d = load_data()   # the file on the fresh branch, not whatever was edited
    apply_entry(d, spec, dry=False)
    save_data(d)
    print(f"on {target}: wrote data/data.json")

    if not a.skip_measure and srcs:
        run("measure", [sys.executable, "tracker/measure.py", "--model", mid])
    elif a.skip_measure:
        print("skipping ladder measurement (--skip-measure)")

    run("validate", [sys.executable, "tracker/validate.py"])
    run("build", [sys.executable, "tracker/build.py"])
    run("check output", [sys.executable, "tools/check_output.py"])
    print("page is valid and built - committing")

    run("git add", ["git", "add", "data/data.json"])
    msg = (f"Update {mid} in data/data.json\n\n"
           f"{action.capitalize()} the {spec.get('NAME', mid)} entry from a supplied "
           f"description: identity, scores, per-engine status and "
           f"{'a measured' if (srcs and not a.skip_measure) else 'the existing'} quant "
           f"ladder. {spec.get('NOTE', '')[:200]}")
    run("git commit", ["git", "commit", "-m", msg])
    if a.no_push:
        print(f"committed locally on {target}; push it to deploy (skipped: --no-push)")
        return 0
    run("git push", ["git", "push", "origin", target])
    if a.direct:
        print(f"pushed to main: CI rebuilds and deploys the page now")
    else:
        gh = shutil.which("gh")
        if gh:
            r = subprocess.run([gh, "pr", "create", "--base", "main", "--head", target,
                                "--title", f"Update {mid} in data/data.json",
                                "--body", f"Generated by tracker/add_model.py. {spec.get('NOTE', '')}"],
                               cwd=ROOT, capture_output=True, text=True)
            url = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
            if url:
                print(f"pull request opened: {url}")
                print("merge it (or: gh pr merge " + url.rsplit("/", 1)[-1] +
                      " --squash) and the site deploys")
            else:
                print(f"branch pushed: origin/{target}. Open a PR: "
                      f"gh pr create --base main --head {target}")
        else:
            print(f"branch pushed: origin/{target}. Open a PR against main to deploy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
