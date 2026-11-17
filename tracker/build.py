#!/usr/bin/env python3
"""Build the site into docs/, which GitHub Pages serves.

    python3 tracker/build.py

Steps:
  1. Validate data
  2. Render site
  3. Verify output exists
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    sys.path.insert(0, HERE)

    # Step 1: Validate
    print("Validating data...")
    ret = os.system(f"python3 {os.path.join(HERE, 'validate.py')}")
    if ret:
        print("Data validation failed; aborting build", file=sys.stderr)
        return 1

    # Step 2: Render
    print("Rendering site...")
    from render_status import render, CSS
    SITE = os.path.join(ROOT, "docs")
    os.makedirs(SITE, exist_ok=True)

    html = render()
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  docs/index.html ({len(html)} bytes)")

    with open(os.path.join(SITE, "style.css"), "w", encoding="utf-8") as f:
        f.write(CSS)
    print(f"  docs/style.css ({len(CSS)} bytes)")

    # Step 3: Verify
    for name in ["index.html", "style.css"]:
        p = os.path.join(SITE, name)
        assert os.path.exists(p), f"{p} missing after render"

    print("Build complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
