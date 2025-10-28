from __future__ import annotations
"""
Run daily ICC jobs (sp500, usall) after 16:00 US/Eastern and save into YYYYMM subfolders.
Also auto-sorts any legacy CSVs already in data/ into the correct YYYYMM folder.
"""

import datetime as dt
import os
import pathlib
import re
import shutil
import subprocess
import sys

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

REPO = pathlib.Path(__file__).resolve().parent
DATA = REPO / "data"
DATA.mkdir(exist_ok=True)

# produced by backend/icc_market_live.py at repo root
SAMPLE = REPO / "icc_live_sample.csv"

SCRIPT_CANDIDATES = [
    REPO / "backend" / "icc_market_live.py",
    REPO / "icc_market_live.py",
]

# match our filenames like: icc_live_sp500_2025_1027.csv
FNAME_RE = re.compile(
    r"^icc_live_(sp500|usall)_(\d{4})_(\d{4})\.csv$"
)

def eastern_now() -> dt.datetime:
    if ZoneInfo is None:
        raise RuntimeError("zoneinfo not available. Use Python>=3.9 and install 'tzdata'.")
    return dt.datetime.now(ZoneInfo("America/New_York"))

def yyyymm_from_parts(year: str, mmdd: str) -> str:
    # mmdd like "1027" -> mm = "10"
    return f"{year}{mmdd[:2]}"

def ensure_month_dir(yyyymm: str) -> pathlib.Path:
    d = DATA / yyyymm
    d.mkdir(parents=True, exist_ok=True)
    return d

def find_script() -> pathlib.Path:
    for p in SCRIPT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "icc_market_live.py not found (checked backend/icc_market_live.py and ./icc_market_live.py)"
    )

def move_unique(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    """Move src to dst; if dst exists, append _rN before suffix."""
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst
    stem, suf = dst.stem, dst.suffix
    for k in range(1, 100):
        cand = dst.with_name(f"{stem}_r{k}{suf}")
        if not cand.exists():
            shutil.move(str(src), str(cand))
            return cand
    raise RuntimeError(f"Too many duplicates for {dst.name}")

def autosort_existing_data() -> None:
    """
    Move any top-level CSVs inside data/ into data/YYYYMM/ subfolders
    based on their file name pattern.
    """
    for p in DATA.glob("*.csv"):
        m = FNAME_RE.match(p.name)
        if not m:
            # leave unknown files alone
            continue
        _, year, mmdd = m.groups()
        month = yyyymm_from_parts(year, mmdd)
        month_dir = ensure_month_dir(month)
        dst = month_dir / p.name
        if p.resolve() == dst.resolve():
            continue
        moved = move_unique(p, dst)
        print(f"[autosort] Moved {p.name} -> {moved.relative_to(REPO)}")

def run_once(arg: str, out_dir: pathlib.Path, now_et: dt.datetime) -> pathlib.Path:
    """
    Run icc_market_live.py once with arg ('sp500' or 'usall'), then move/rename
    icc_live_sample.csv into out_dir/icc_live_{arg}_{YYYY}_{MMDD}.csv
    """
    script = find_script()
    cmd = [sys.executable, str(script), arg]
    print(">> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not SAMPLE.exists():
        raise FileNotFoundError(f"Expected {SAMPLE} after running {arg}, but not found.")

    y = now_et.strftime("%Y")
    mmdd = now_et.strftime("%m%d")
    out_path = out_dir / f"icc_live_{arg}_{y}_{mmdd}.csv"
    saved = move_unique(SAMPLE, out_path)
    print(f">> Saved: {saved.relative_to(REPO)}")
    return saved

def main():
    # 1) guard by time
    now_et = eastern_now()
    if os.environ.get("IGNORE_TIME_GUARD") != "1" and now_et.hour < 16:
        print(f"[guard] {now_et.strftime('%H:%M')} ET < 16:00 — skip.")
        return

    # 2) sort any legacy files that are still under data/ root
    autosort_existing_data()

    # 3) ensure today's YYYYMM folder
    month_dir = ensure_month_dir(now_et.strftime("%Y%m"))

    # 4) run jobs into subfolder
    run_once("sp500", month_dir, now_et)
    run_once("usall",  month_dir, now_et)

if __name__ == "__main__":
    main()
