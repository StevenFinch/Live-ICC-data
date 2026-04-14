from __future__ import annotations

import pathlib
import subprocess
import sys
from datetime import datetime

# Make repo root importable even when running:
#   python backend/run_extra_indices.py
REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.icc_market_live import _today_et  # type: ignore


EXTRA_UNIVERSES = [
    "sp100",
    "dow30",
    "ndx100",
    "sp400",
    "sp600",
    "sp1500",
    "rut1000",
]


def ensure_month_dir(now_et: datetime) -> pathlib.Path:
    month_dir = REPO / "data" / now_et.strftime("%Y%m")
    month_dir.mkdir(parents=True, exist_ok=True)
    return month_dir


def run_one(universe: str, month_dir: pathlib.Path) -> None:
    out_path = month_dir / f"icc_live_{universe}_{_today_et()}.csv"

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
    subprocess.run(cmd, check=True)


def main() -> None:
    now_et = datetime.now()
    month_dir = ensure_month_dir(now_et)

    failed = []
    for universe in EXTRA_UNIVERSES:
        try:
            run_one(universe, month_dir)
        except Exception as e:
            print(f"[run_extra_indices] FAILED on {universe}: {type(e).__name__}: {e}")
            failed.append(universe)

    if failed:
        raise RuntimeError(f"Some extra universes failed: {failed}")

    print("[run_extra_indices] all done.")


if __name__ == "__main__":
    main()
