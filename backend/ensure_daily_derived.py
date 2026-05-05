from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pandas.errors import EmptyDataError

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DERIVED_DIR = DATA_DIR / "derived"
CONFIG_DIR = REPO / "config"

SNAPSHOT_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of each duplicated column name."""
    if df is None or df.empty:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _load_snapshot(path: Path) -> pd.DataFrame:
    """Load a non-empty ICC snapshot and normalize key fields."""
    if path.stat().st_size <= 0:
        raise EmptyDataError(f"Empty snapshot: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise EmptyDataError(f"No rows in snapshot: {path}")
    df = _drop_duplicate_columns(df)
    required = {"ticker", "ICC", "mktcap", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["ICC"] = pd.to_numeric(df["ICC"], errors="coerce")
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[df["ticker"].notna() & df["ICC"].notna() & df["mktcap"].notna() & df["date"].notna()].copy()
    return df


def _snapshot_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort snapshots by date, rerun number, and filename."""
    m = SNAPSHOT_RE.match(path.name)
    if not m:
        return (0, 0, path.name)
    yyyymmdd = int(f"{m.group('year')}{m.group('mmdd')}")
    rerun = int(m.group("rerun") or 0)
    return (yyyymmdd, rerun, path.name)


def _find_latest_valid_usall() -> tuple[Path, pd.DataFrame, str]:
    """Find the latest valid usall snapshot under data/YYYYMM."""
    candidates = []
    for path in DATA_DIR.glob("*/*.csv"):
        m = SNAPSHOT_RE.match(path.name)
        if not m or m.group("universe") != "usall":
            continue
        candidates.append(path)

    for path in sorted(candidates, key=_snapshot_sort_key, reverse=True):
        try:
            df = _load_snapshot(path)
            if not df.empty:
                asof_date = str(df["date"].dropna().iloc[0])
                return path, df, asof_date
        except Exception as exc:
            print(f"[ensure_daily_derived] skip invalid usall snapshot {path}: {type(exc).__name__}: {exc}")

    raise RuntimeError("No valid non-empty usall snapshot found.")


def _call_with_fallbacks(label: str, funcs: list[Callable[[], Any]]) -> Any:
    """Try multiple call signatures and return the first successful result."""
    errors: list[str] = []
    for fn in funcs:
        try:
            return fn()
        except TypeError as exc:
            errors.append(f"TypeError: {exc}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError(f"All {label} call patterns failed. Errors: {' | '.join(errors[-5:])}")


def _normalize_result(result: Any) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Normalize helper return values into a main panel and optional members panel."""
    if isinstance(result, tuple):
        if len(result) >= 2:
            panel = result[0]
            members = result[1]
        elif len(result) == 1:
            panel = result[0]
            members = None
        else:
            panel = pd.DataFrame()
            members = None
    else:
        panel = result
        members = None

    if panel is None:
        panel = pd.DataFrame()
    if not isinstance(panel, pd.DataFrame):
        panel = pd.DataFrame(panel)
    panel = _drop_duplicate_columns(panel)

    if members is not None and not isinstance(members, pd.DataFrame):
        members = pd.DataFrame(members)
    if isinstance(members, pd.DataFrame):
        members = _drop_duplicate_columns(members)

    return panel, members


def _ensure_date_column(df: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    """Ensure a consistent date column exists in derived outputs."""
    df = _drop_duplicate_columns(df)
    if "date" not in df.columns:
        df["date"] = asof_date
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["date"] = df["date"].fillna(asof_date)
    return df


def _write_panel(df: pd.DataFrame, family_dir: Path, stem: str, asof_date: str) -> Path:
    """Write a derived daily panel to data/derived/<family>/YYYYMM."""
    yyyymm = asof_date[:7].replace("-", "")
    tag = asof_date.replace("-", "_")
    out_dir = family_dir / yyyymm
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{tag}.csv"
    df = _ensure_date_column(df, asof_date)
    df.to_csv(out_path, index=False)
    return out_path


def _build_etf(latest_usall: pd.DataFrame, asof_date: str) -> None:
    """Refresh ETF ICC outputs for the latest usall date."""
    from backend.holdings_sources import build_etf_icc_panel

    config_path = CONFIG_DIR / "etfs.csv"
    result = _call_with_fallbacks(
        "ETF ICC",
        [
            lambda: build_etf_icc_panel(latest_usall, asof_date=asof_date, config_path=config_path),
            lambda: build_etf_icc_panel(latest_usall, asof_date, config_path),
            lambda: build_etf_icc_panel(latest_usall, asof_date),
            lambda: build_etf_icc_panel(latest_usall, config_path=config_path, asof_date=asof_date),
            lambda: build_etf_icc_panel(latest_usall, config_path),
        ],
    )
    panel, members = _normalize_result(result)
    if panel.empty:
        raise RuntimeError("ETF ICC panel is empty after refresh.")
    out_path = _write_panel(panel, DERIVED_DIR / "etf", "etf_icc", asof_date)
    print(f"[ensure_daily_derived] wrote ETF panel: {out_path} ({len(panel)} rows)")
    if members is not None and not members.empty:
        members_path = _write_panel(members, DERIVED_DIR / "etf", "etf_members", asof_date)
        print(f"[ensure_daily_derived] wrote ETF members: {members_path} ({len(members)} rows)")


def _build_country_region(latest_usall: pd.DataFrame, asof_date: str) -> None:
    """Refresh Country / Region ADR ICC outputs for the latest usall date."""
    from backend.country_adr_sources import build_country_region_icc_panel

    result = _call_with_fallbacks(
        "Country / Region ICC",
        [
            lambda: build_country_region_icc_panel(latest_usall, asof_date=asof_date),
            lambda: build_country_region_icc_panel(latest_usall, asof_date),
            lambda: build_country_region_icc_panel(asof_date=asof_date, base_usall=latest_usall),
            lambda: build_country_region_icc_panel(asof_date=asof_date),
            lambda: build_country_region_icc_panel(latest_usall),
        ],
    )
    panel, members = _normalize_result(result)
    if panel.empty:
        raise RuntimeError("Country / Region ICC panel is empty after refresh.")
    out_path = _write_panel(panel, DERIVED_DIR / "country_adr", "country_adr_icc", asof_date)
    print(f"[ensure_daily_derived] wrote Country / Region panel: {out_path} ({len(panel)} rows)")
    if members is not None and not members.empty:
        members_path = _write_panel(members, DERIVED_DIR / "country_adr", "country_adr_members", asof_date)
        print(f"[ensure_daily_derived] wrote Country / Region members: {members_path} ({len(members)} rows)")


def main() -> None:
    """Ensure ETF and Country / Region derived data exist for the latest usall date."""
    source_path, latest_usall, asof_date = _find_latest_valid_usall()
    force = os.environ.get("FORCE_DERIVED_REFRESH", "0") == "1"
    print(f"[ensure_daily_derived] latest usall snapshot = {source_path}")
    print(f"[ensure_daily_derived] asof_date = {asof_date}")
    print(f"[ensure_daily_derived] force refresh = {force}")

    etf_existing = DERIVED_DIR / "etf" / asof_date[:7].replace("-", "") / f"etf_icc_{asof_date.replace('-', '_')}.csv"
    country_existing = DERIVED_DIR / "country_adr" / asof_date[:7].replace("-", "") / f"country_adr_icc_{asof_date.replace('-', '_')}.csv"

    if force or not (etf_existing.exists() and etf_existing.stat().st_size > 0):
        _build_etf(latest_usall, asof_date)
    else:
        print(f"[ensure_daily_derived] ETF panel already exists: {etf_existing}")

    if force or not (country_existing.exists() and country_existing.stat().st_size > 0):
        _build_country_region(latest_usall, asof_date)
    else:
        print(f"[ensure_daily_derived] Country / Region panel already exists: {country_existing}")


if __name__ == "__main__":
    main()
