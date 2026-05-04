from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from backend.holdings_sources import build_etf_icc_panel
from backend.country_adr_sources import build_country_adr_icc_panel

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DERIVED_DIR = DATA_DIR / "derived"
DOCS_DATA = REPO / "docs" / "data"
CONFIG_DIR = REPO / "config"
SNAPSHOT_RE = re.compile(r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$")


def parse_snapshot_meta(path: Path) -> dict[str, Any] | None:
    """Parse metadata from an ICC snapshot filename."""
    m = SNAPSHOT_RE.match(path.name)
    if not m:
        return None
    year = m.group("year")
    mmdd = m.group("mmdd")
    return {
        "universe": m.group("universe"),
        "year": year,
        "mmdd": mmdd,
        "date": datetime.strptime(f"{year}{mmdd}", "%Y%m%d").date().isoformat(),
        "yyyymm": f"{year}{mmdd[:2]}",
        "rerun": int(m.group("rerun") or 0),
    }


def load_snapshot(path: Path) -> pd.DataFrame:
    """Load and normalize one firm-level ICC snapshot."""
    if path.stat().st_size <= 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise EmptyDataError(f"Empty CSV: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False).str.strip()
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    df["ICC"] = pd.to_numeric(df["ICC"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "name" not in df.columns:
        df["name"] = None
    if "sector" not in df.columns:
        df["sector"] = None
    return df.dropna(subset=["ticker", "mktcap", "ICC", "date"]).copy()


def latest_valid_snapshot(universe: str) -> tuple[Path, pd.DataFrame, str]:
    """Find the latest valid snapshot for a universe."""
    candidates: list[tuple[tuple[str, int, str], Path]] = []
    for path in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(path)
        if not meta or meta["universe"] != universe:
            continue
        try:
            if path.stat().st_size <= 0:
                continue
        except FileNotFoundError:
            continue
        candidates.append(((meta["date"], meta["rerun"], path.name), path))
    if not candidates:
        raise FileNotFoundError(f"No valid snapshot files found for universe={universe}")
    candidates.sort(key=lambda x: x[0])
    for _key, path in reversed(candidates):
        try:
            df = load_snapshot(path)
            if not df.empty:
                return path, df, str(df["date"].iloc[0])
        except Exception as exc:
            print(f"[ensure_daily_derived] skip invalid {path}: {type(exc).__name__}: {exc}")
    raise FileNotFoundError(f"No loadable snapshot files found for universe={universe}")


def safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-safe records."""
    records = []
    for row in df.to_dict(orient="records"):
        out: dict[str, Any] = {}
        for k, v in row.items():
            try:
                if pd.isna(v):
                    out[k] = None
                    continue
            except Exception:
                pass
            if isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                x = float(v)
                out[k] = x if np.isfinite(x) else None
            else:
                out[k] = v
        records.append(out)
    return records


def save_daily_panel(df: pd.DataFrame, root: Path, stem: str, asof_date: str) -> Path:
    """Save one dated derived panel."""
    dt = datetime.strptime(asof_date, "%Y-%m-%d")
    out_dir = root / dt.strftime("%Y%m")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{dt.strftime('%Y_%m%d')}.csv"
    out = df.copy()
    if "date" not in out.columns:
        out.insert(0, "date", asof_date)
    else:
        out["date"] = asof_date
    out.to_csv(out_path, index=False)
    return out_path


def patch_docs_json(name: str, df: pd.DataFrame, asof_date: str) -> None:
    """Patch lightweight docs JSON so the site never stays on an old date."""
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    payload = {"asof_date": asof_date, "latest": safe_records(df), "daily": safe_records(df), "monthly": safe_records(df)}
    names = [name]
    if name == "etf":
        names.append("etf_icc")
    if name == "country":
        names.append("country_icc")
    for n in names:
        with open(DOCS_DATA / f"{n}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)


def ensure_daily_derived(universe: str = "usall", force: bool = False) -> None:
    """Ensure ETF and Country / Region derived files exist for the latest raw snapshot date."""
    path, usall, asof_date = latest_valid_snapshot(universe)
    print(f"[ensure_daily_derived] latest raw snapshot = {path}")
    print(f"[ensure_daily_derived] asof_date = {asof_date}")

    dt = datetime.strptime(asof_date, "%Y-%m-%d")
    yyyymm = dt.strftime("%Y%m")
    etf_file = DERIVED_DIR / "etf" / yyyymm / f"etf_icc_{dt.strftime('%Y_%m%d')}.csv"
    country_file = DERIVED_DIR / "country_adr" / yyyymm / f"country_adr_icc_{dt.strftime('%Y_%m%d')}.csv"

    if force or not etf_file.exists() or etf_file.stat().st_size == 0:
        etf_panel, etf_members = build_etf_icc_panel(usall, CONFIG_DIR / "etfs.csv")
        etf_file = save_daily_panel(etf_panel, DERIVED_DIR / "etf", "etf_icc", asof_date)
        print(f"[ensure_daily_derived] wrote ETF file: {etf_file}")
        if etf_members is not None and not etf_members.empty:
            mem_dir = DERIVED_DIR / "etf_members" / yyyymm
            mem_dir.mkdir(parents=True, exist_ok=True)
            etf_members.to_csv(mem_dir / f"etf_members_{dt.strftime('%Y_%m%d')}.csv", index=False)
    else:
        etf_panel = pd.read_csv(etf_file)
        print(f"[ensure_daily_derived] ETF already exists: {etf_file}")

    if force or not country_file.exists() or country_file.stat().st_size == 0:
        country_panel, country_members = build_country_adr_icc_panel(DERIVED_DIR / "country_adr" / "adr_candidates_latest.csv")
        country_file = save_daily_panel(country_panel, DERIVED_DIR / "country_adr", "country_adr_icc", asof_date)
        print(f"[ensure_daily_derived] wrote country file: {country_file}")
        if country_members is not None and not country_members.empty:
            mem_dir = DERIVED_DIR / "country_adr_members" / yyyymm
            mem_dir.mkdir(parents=True, exist_ok=True)
            country_members.to_csv(mem_dir / f"country_adr_members_{dt.strftime('%Y_%m%d')}.csv", index=False)
    else:
        country_panel = pd.read_csv(country_file)
        print(f"[ensure_daily_derived] country already exists: {country_file}")

    patch_docs_json("etf", etf_panel, asof_date)
    patch_docs_json("country", country_panel, asof_date)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    force_env = os.environ.get("FORCE_DERIVED_REBUILD", "0") == "1"
    ensure_daily_derived(args.universe, force=args.force or force_env)


if __name__ == "__main__":
    main()
