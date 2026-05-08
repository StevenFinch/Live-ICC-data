from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from backend.holdings_sources import build_etf_icc_panel
from backend.country_adr_sources import build_country_region_icc_panel

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SNAPSHOT_RE = re.compile(r"^icc_live_usall_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$")


@dataclass
class Snapshot:
    path: Path
    date: str
    df: pd.DataFrame


def load_usall_snapshot(path: Path) -> pd.DataFrame:
    """Load one all-market snapshot."""
    if path.stat().st_size <= 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[df["date"].notna()].copy()
    if df.empty:
        raise ValueError(f"No valid rows in {path}")
    return df


def latest_valid_usall() -> Snapshot:
    """Find the latest valid non-empty all-market snapshot."""
    candidates = []
    for p in DATA.glob("*/*.csv"):
        m = SNAPSHOT_RE.match(p.name)
        if not m:
            continue
        try:
            df = load_usall_snapshot(p)
        except Exception as exc:
            print(f"[ensure_daily_derived] skip invalid usall {p}: {type(exc).__name__}: {exc}")
            continue
        date = str(df["date"].dropna().iloc[0])
        rerun = int(m.group("rerun") or 0)
        candidates.append(((date, rerun, p.name), Snapshot(p, date, df)))
    if not candidates:
        raise RuntimeError("No valid usall snapshots found.")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def main() -> None:
    """Refresh ETF and Country / Region derived data for latest valid usall date."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    _args = parser.parse_args()

    snap = latest_valid_usall()
    asof = snap.date
    print(f"[ensure_daily_derived] latest valid usall date = {asof}")
    print(f"[ensure_daily_derived] source = {snap.path}")

    build_etf_icc_panel(
        snap.df,
        asof_date=asof,
        config_path=REPO / "config" / "etfs.csv",
        repo_root=REPO,
        output_root=DATA / "derived" / "etf",
    )
    print(f"[ensure_daily_derived] ETF derived data refreshed for {asof}")

    build_country_region_icc_panel(
        asof_date=asof,
        repo_root=REPO,
        output_root=DATA / "derived" / "country_adr",
    )
    print(f"[ensure_daily_derived] Country / Region derived data refreshed for {asof}")


if __name__ == "__main__":
    main()
