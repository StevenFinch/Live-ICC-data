from __future__ import annotations
"""
Generate a repository-wide file listing as FILES.md (Markdown).
- Excludes the .git directory
- Shows path, size (bytes), and last-modified time (UTC)
"""

import datetime as dt
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "FILES.md"

EXCLUDES = {".git"}

def iter_files(base: pathlib.Path):
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(REPO)
        parts = rel.parts
        if any(part in EXCLUDES for part in parts):
            continue
        if p.is_file():
            yield p, rel

def main():
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = []
    lines.append(f"# Repository File Index")
    lines.append("")
    lines.append(f"_auto-generated: {now}_")
    lines.append("")
    lines.append("```text")
    for p, rel in iter_files(REPO):
        stat = p.stat()
        mtime = dt.datetime.utcfromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size = stat.st_size
        lines.append(f"{rel}   [{size} B]   (modified {mtime} UTC)")
    lines.append("```")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {OUT}")

if __name__ == "__main__":
    main()
