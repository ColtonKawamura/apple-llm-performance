#!/usr/bin/env python3
"""Sanity-check the rendered page. Run by CI after tracker/build.py.

validate.py checks the inputs; this checks the output. The failure it exists to
catch is a template field that renders as literal text because its braces got
doubled by accident - format() raises on a *missing* field, but an
over-escaped one substitutes nothing and ships silently.
"""
import os
import re
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tracker"))

import render_status as rs  # noqa: E402

html = open(os.path.join(ROOT, "docs", "index.html"), encoding="utf-8").read()
fields = {f for _, f, _, _ in string.Formatter().parse(rs.TEMPLATE) if f}
problems = []

leaked = sorted(f for f in fields if "{" + f + "}" in html)
if leaked:
    problems.append(f"template fields rendered as literal text: {', '.join(leaked)}")

if 'id="card-' not in html:
    problems.append("no model cards in the output")
if "data-payload" not in html:
    problems.append("no glance rows in the output")
if "dict_items" in html or "OrderedDict" in html:
    problems.append("a Python repr leaked into the page")

# A helper that returns a tuple and is interpolated straight into the template
# ships the tuple's repr, and "('', '')" under 179 model cards was invisible to
# every other check. Catch the shape, not just that one instance.
# Python reprs use single quotes; the page's JS uses double quotes, so this
# shape does not false-positive on setAttribute("a", "b") calls.
tuple_repr = re.search(r"\('[^']*',\s*'[^']*'\)", html)
if tuple_repr:
    problems.append(f"a tuple repr leaked into the page: {tuple_repr.group(0)[:40]}")

# `hf` is a repo path for HF models and a full URL for AA-listed ones. Blindly
# prefixing it produced https://huggingface.co/https://artificialanalysis.ai/...
if "huggingface.co/http" in html:
    problems.append("a source link is double-prefixed with huggingface.co/")

# The build already refuses to emit C0/C1 characters; this is the belt to that
# brace, because an octal escape in embedded CSS once shipped U+0091 to the page.
for ch in re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", html):
    problems.append(f"control character U+{ord(ch):04X} in the output")
    break

cards = html.count('id="card-')
rows = html.count("data-payload")
if cards < 10 or rows < 10:
    problems.append(f"suspiciously little content: {cards} cards, {rows} rows")

for p in problems:
    print(f"error: {p}")
if problems:
    raise SystemExit(1)
print(f"ok: {len(html):,} bytes, {cards} cards, {rows} rows, {len(fields)} fields substituted")
