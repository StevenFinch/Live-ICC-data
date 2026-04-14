from __future__ import annotations

import pathlib
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = pathlib.Path(__file__).resolve().parents[1]

EXTRA_UNIVERSES = [
    "sp100",
    "dow30",
    "ndx100",
]


def today_et_str() -> str:
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.strftime("%Y_%m%d")


def ensure_month_dir() -> pathlib.Path:
    now = datetime.now(ZoneInfo("America/New_York"))
    month_dir = REPO / "data" / now.strftime("%Y%m")
    month_dir.mkdir(parents=True, exist_ok=True)
    return month_dir


def run_one(universe: str, month_dir: pathlib.Path) -> bool:
    out_path = month_dir / f"icc_live_{universe}_{today_et_str()}.csv"

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[run_extra_indices] skip existing {out_path.name}")
        return True

    cmd = [
        sys.executable,
        str(REPO / "backend" / "icc_market_live.py"),
        "--universe",
        universe,
        "--save",
        str(out_path),
    ]

    print(f"[run_extra_indices] running {universe}")
    print("[run_extra_indices] cmd =", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[run_extra_indices] done {universe}")
            return True
        print(f"[run_extra_indices] failed: empty output for {universe}")
        return False
    except Exception as e:
        print(f"[run_extra_indices] FAILED on {universe}: {type(e).__name__}: {e}")
        return False


def main() -> None:
    month_dir = ensure_month_dir()
    ok = []
    bad = []

    for universe in EXTRA_UNIVERSES:
        success = run_one(universe, month_dir)
        if success:
            ok.append(universe)
        else:
            bad.append(universe)

    print(f"[run_extra_indices] success = {ok}")
    print(f"[run_extra_indices] failed = {bad}")

if __name__ == "__main__":
    main()
