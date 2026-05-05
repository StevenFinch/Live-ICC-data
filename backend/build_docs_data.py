from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from backend.holdings_sources import build_etf_icc_panel
from backend.country_adr_sources import build_country_region_icc_panel

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DIR = REPO / "docs"
DOCS_DATA = DOCS_DIR / "data"
DOWNLOADS = DOCS_DATA / "downloads"
RAW_DOWNLOADS = DOWNLOADS / "raw_snapshots"
FAMILY_DOWNLOADS = DOWNLOADS / "families"

SNAPSHOT_RE = re.compile(r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$")
INDEX_UNIVERSES = ["sp500", "sp100", "dow30", "ndx100", "sp400", "sp600", "sp1500", "rut1000"]
FAMILY_ORDER = ["marketwide", "value", "industry", "indices", "etf", "country"]
FAMILY_LABELS = {
    "marketwide": "Marketwide ICC",
    "value": "Value premium ICC",
    "industry": "Industry ICC",
    "indices": "Index ICC",
    "etf": "ETF ICC",
    "country": "Country / Region Level ICC",
}
FAMILY_NOTES = {
    "marketwide": "All U.S. market and S&P 500 aggregate ICC series.",
    "value": "Value, growth, and implied value premium series built from firm-level ICC snapshots.",
    "industry": "Industry-level ICC series aggregated from the all-market firm-level ICC panel.",
    "indices": "Index-level ICC series aggregated from available index constituent snapshots.",
    "etf": "ETF ICC series from online ETF holdings matched to firm-level ICC snapshots. History starts when archiving begins.",
    "country": "Country / Region Level ICC from ADR and foreign-listing top-N composites. History starts when archiving begins.",
}


@dataclass
class Snapshot:
    path: Path
    universe: str
    date: str
    yyyymm: str
    year: str
    month: str
    rerun: int
    df: pd.DataFrame


def ensure_dirs() -> None:
    """Create all docs data and download directories."""
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    RAW_DOWNLOADS.mkdir(parents=True, exist_ok=True)
    FAMILY_DOWNLOADS.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    """Convert numpy/pandas objects and non-finite values to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON without NaN values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)


def parse_snapshot_path(path: Path) -> dict[str, Any] | None:
    """Parse standard raw ICC snapshot filenames."""
    m = SNAPSHOT_RE.match(path.name)
    if not m:
        return None
    year = m.group("year")
    mmdd = m.group("mmdd")
    month = mmdd[:2]
    day = mmdd[2:]
    return {
        "universe": m.group("universe"),
        "year": year,
        "month": month,
        "day": day,
        "yyyymm": f"{year}{month}",
        "date": f"{year}-{month}-{day}",
        "rerun": int(m.group("rerun") or 0),
    }


def load_snapshot(path: Path) -> pd.DataFrame:
    """Load and normalize one raw ICC snapshot."""
    if path.stat().st_size <= 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        raise EmptyDataError(f"No rows or columns: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    df["ICC"] = pd.to_numeric(df["ICC"], errors="coerce")
    df["bm"] = pd.to_numeric(df["bm"], errors="coerce") if "bm" in df.columns else np.nan
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    if "name" not in df.columns:
        df["name"] = ""
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[
        df["ticker"].notna()
        & df["date"].notna()
        & df["mktcap"].notna()
        & df["ICC"].notna()
        & np.isfinite(df["mktcap"])
        & np.isfinite(df["ICC"])
        & (df["mktcap"] > 0)
    ].copy()
    if df.empty:
        raise ValueError(f"Cleaned snapshot is empty: {path}")
    return df


def discover_snapshots() -> list[Snapshot]:
    """Discover all valid raw ICC snapshots and deduplicate by universe/date."""
    candidates: list[tuple[tuple[str, str, int, str], Snapshot]] = []
    for path in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_path(path)
        if meta is None:
            continue
        try:
            df = load_snapshot(path)
        except Exception as exc:
            print(f"[build_docs_data] skip invalid raw snapshot {path}: {type(exc).__name__}: {exc}")
            continue
        actual_date = str(df["date"].dropna().iloc[0])
        snap = Snapshot(path, meta["universe"], actual_date, actual_date[:7].replace("-", ""), actual_date[:4], actual_date[5:7], int(meta["rerun"]), df)
        candidates.append(((snap.universe, snap.date, snap.rerun, snap.path.name), snap))
    candidates.sort(key=lambda x: x[0])
    latest: dict[tuple[str, str], Snapshot] = {}
    for _key, snap in candidates:
        latest[(snap.universe, snap.date)] = snap
    out = list(latest.values())
    out.sort(key=lambda s: (s.universe, s.date, s.rerun, s.path.name))
    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute a finite positive-weight weighted mean."""
    mask = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def latest_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Return latest row overall or latest row per group."""
    if df.empty:
        return df.copy()
    x = df.sort_values("date").copy()
    if not group_cols:
        return x.tail(1).reset_index(drop=True)
    idx = x.groupby(group_cols, dropna=False)["date"].idxmax()
    return x.loc[idx].sort_values(group_cols).reset_index(drop=True)


def month_end_rows(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    """Create monthly observations by keeping the latest daily row in each month/group."""
    if df.empty:
        return pd.DataFrame()
    x = df.copy()
    x["month"] = x["date"].astype(str).str.slice(0, 7)
    x = x.sort_values("date")
    idx = x.groupby(group_cols + ["month"], dropna=False)["date"].idxmax() if group_cols else x.groupby(["month"], dropna=False)["date"].idxmax()
    out = x.loc[idx].copy().sort_values(group_cols + ["month"] if group_cols else ["month"])
    out = out.rename(columns={"date": "month_end_date"})
    keep = group_cols + ["month", "month_end_date"] + value_cols
    keep = [c for c in keep if c in out.columns]
    return out[keep].reset_index(drop=True)


def build_marketwide(snapshots: list[Snapshot]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build marketwide daily/monthly/latest tables."""
    rows = []
    for snap in snapshots:
        if snap.universe not in {"usall", "sp500"}:
            continue
        family = "all_market" if snap.universe == "usall" else "sp500"
        df = snap.df
        rows.append({
            "date": snap.date,
            "family": family,
            "daily_icc": weighted_mean(df["ICC"], df["mktcap"]),
            "ew_icc": float(df["ICC"].mean()),
            "n_firms": int(len(df)),
            "method": "ICC calculation",
            "source_file": str(snap.path.relative_to(REPO)),
        })
    daily = pd.DataFrame(rows).sort_values(["family", "date"]).reset_index(drop=True) if rows else pd.DataFrame()
    monthly = month_end_rows(daily, ["family"], ["daily_icc", "ew_icc", "n_firms", "method", "source_file"]) if not daily.empty else pd.DataFrame()
    latest = latest_by_group(daily, ["family"])
    return latest, daily, monthly


def build_value(snapshots: list[Snapshot]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build value/growth/IVP daily/monthly/latest tables from usall snapshots."""
    rows = []
    for snap in snapshots:
        if snap.universe != "usall":
            continue
        df = snap.df.copy()
        df = df[df["bm"].notna() & np.isfinite(df["bm"]) & (df["bm"] > 0) & df["ICC"].notna()].copy()
        if df.empty:
            continue
        lo, hi = df["ICC"].quantile(0.005), df["ICC"].quantile(0.995)
        df = df[(df["ICC"] >= lo) & (df["ICC"] <= hi)].copy()
        if df.empty:
            continue
        size_median = df["mktcap"].median()
        bm30, bm70 = df["bm"].quantile(0.30), df["bm"].quantile(0.70)
        df["size_bucket"] = np.where(df["mktcap"] <= size_median, "S", "B")
        df["bm_bucket"] = pd.cut(df["bm"], bins=[-np.inf, bm30, bm70, np.inf], labels=["L", "M", "H"], include_lowest=True).astype(str)
        df["portfolio"] = df["size_bucket"] + "/" + df["bm_bucket"]
        d = {p: weighted_mean(g["ICC"], g["mktcap"]) for p, g in df.groupby("portfolio")}
        growth = float((d["S/L"] + d["B/L"]) / 2.0) if pd.notna(d.get("S/L")) and pd.notna(d.get("B/L")) else np.nan
        value = float((d["S/H"] + d["B/H"]) / 2.0) if pd.notna(d.get("S/H")) and pd.notna(d.get("B/H")) else np.nan
        ivp = value - growth if pd.notna(value) and pd.notna(growth) else np.nan
        rows.append({"date": snap.date, "value_icc": value, "growth_icc": growth, "ivp": ivp, "n_firms": int(len(df)), "method": "ICC calculation", "source_file": str(snap.path.relative_to(REPO))})
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()
    monthly = month_end_rows(daily, [], ["value_icc", "growth_icc", "ivp", "n_firms", "method", "source_file"]) if not daily.empty else pd.DataFrame()
    latest = latest_by_group(daily, [])
    return latest, daily, monthly


def build_industry(snapshots: list[Snapshot]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build industry daily/monthly/latest tables from usall snapshots."""
    rows = []
    for snap in snapshots:
        if snap.universe != "usall":
            continue
        df = snap.df.copy()
        df["sector"] = df["sector"].fillna("Unknown").astype(str)
        for sector, g in df.groupby("sector", dropna=False):
            rows.append({"date": snap.date, "group": sector, "daily_icc": weighted_mean(g["ICC"], g["mktcap"]), "ew_icc": float(g["ICC"].mean()), "n_firms": int(len(g)), "method": "ICC calculation", "source_file": str(snap.path.relative_to(REPO))})
    daily = pd.DataFrame(rows).sort_values(["group", "date"]).reset_index(drop=True) if rows else pd.DataFrame()
    monthly = month_end_rows(daily, ["group"], ["daily_icc", "ew_icc", "n_firms", "method", "source_file"]) if not daily.empty else pd.DataFrame()
    latest = latest_by_group(daily, ["group"])
    return latest, daily, monthly


def build_indices(snapshots: list[Snapshot]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build index daily/monthly/latest tables from index snapshots."""
    rows = []
    for snap in snapshots:
        if snap.universe not in INDEX_UNIVERSES:
            continue
        df = snap.df
        rows.append({"date": snap.date, "family": snap.universe, "daily_icc": weighted_mean(df["ICC"], df["mktcap"]), "ew_icc": float(df["ICC"].mean()), "n_firms": int(len(df)), "method": "ICC calculation", "source_file": str(snap.path.relative_to(REPO))})
    daily = pd.DataFrame(rows).sort_values(["family", "date"]).reset_index(drop=True) if rows else pd.DataFrame()
    monthly = month_end_rows(daily, ["family"], ["daily_icc", "ew_icc", "n_firms", "method", "source_file"]) if not daily.empty else pd.DataFrame()
    latest = latest_by_group(daily, ["family"])
    return latest, daily, monthly


def read_derived_family(kind: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read archived derived ETF/Country files if they exist."""
    root = DATA_DIR / "derived" / ("etf" if kind == "etf" else "country_adr")
    files = sorted(root.glob("*/*.csv")) if root.exists() else []
    if kind == "country":
        files = [p for p in files if "constituents" not in p.name]
    frames = []
    for path in files:
        try:
            if path.stat().st_size <= 0:
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["source_file"] = str(path.relative_to(REPO))
            frames.append(df)
        except Exception as exc:
            print(f"[build_docs_data] skip derived {kind} file {path}: {type(exc).__name__}: {exc}")
    if not frames:
        cols = ["date", "ticker", "label", "icc", "coverage_weight", "method", "holding_source", "status"] if kind == "etf" else ["date", "country", "icc", "n_selected", "n_icc_available", "coverage_mktcap", "method", "status"]
        empty = pd.DataFrame(columns=cols)
        return empty, empty, empty
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    daily = daily[daily["date"].notna()].copy()
    group_cols = ["ticker"] if kind == "etf" else ["country"]
    value_cols = [c for c in daily.columns if c not in {"date", "month"}]
    daily = daily.sort_values(group_cols + ["date"]).drop_duplicates(group_cols + ["date"], keep="last")
    monthly = month_end_rows(daily, group_cols, value_cols)
    latest = latest_by_group(daily, group_cols)
    return latest, daily.reset_index(drop=True), monthly


def to_web_path(path: Path) -> str:
    """Convert a docs file path to a relative browser path."""
    return "./" + path.relative_to(DOCS_DIR).as_posix()


def write_csv(path: Path, df: pd.DataFrame) -> str:
    """Write CSV and return a web-relative download path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return to_web_path(path)


def make_zip(zip_path: Path, files: list[Path]) -> str:
    """Create a zip archive from a list of files and return a web path."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.exists() and f.is_file():
                zf.write(f, arcname=f.name)
    return to_web_path(zip_path)


def build_family_time_tree(family: str, daily: pd.DataFrame) -> list[dict[str, Any]]:
    """Create year/month/day per-family archive files and zip archives."""
    if daily.empty or "date" not in daily.columns:
        return []
    x = daily.copy()
    x["year"] = x["date"].astype(str).str.slice(0, 4)
    x["month"] = x["date"].astype(str).str.slice(5, 7)
    base = FAMILY_DOWNLOADS / family / "archive"
    years = []
    for year in sorted(x["year"].dropna().unique(), reverse=True):
        year_df = x[x["year"] == year].copy()
        year_files = []
        months = []
        for month in sorted(year_df["month"].dropna().unique(), reverse=True):
            month_df = year_df[year_df["month"] == month].copy()
            month_dir = base / str(year) / str(month)
            month_files = []
            days = []
            for date in sorted(month_df["date"].dropna().unique(), reverse=True):
                day_df = month_df[month_df["date"] == date].copy()
                day_path = month_dir / f"{family}_{date}.csv"
                write_csv(day_path, day_df.drop(columns=["year", "month"], errors="ignore"))
                month_files.append(day_path)
                year_files.append(day_path)
                days.append({"date": date, "n_rows": int(len(day_df)), "csv": to_web_path(day_path)})
            months.append({"month": f"{year}-{month}", "n_files": len(month_files), "download_all_zip": make_zip(base / str(year) / f"{family}_{year}_{month}_all.zip", month_files), "days": days})
        years.append({"year": str(year), "n_files": len(year_files), "download_all_zip": make_zip(base / f"{family}_{year}_all.zip", year_files), "months": months})
    return years


def build_family_downloads(family: str, latest: pd.DataFrame, daily: pd.DataFrame, monthly: pd.DataFrame) -> dict[str, Any]:
    """Write standard family downloads and return catalog metadata."""
    base = FAMILY_DOWNLOADS / family
    latest_path = base / f"{family}_latest.csv"
    daily_path = base / f"{family}_daily_history.csv"
    monthly_path = base / f"{family}_monthly_history.csv"
    latest_web = write_csv(latest_path, latest)
    daily_web = write_csv(daily_path, daily)
    monthly_web = write_csv(monthly_path, monthly)
    all_zip = make_zip(base / f"{family}_all.zip", [latest_path, daily_path, monthly_path])
    return {"family": family, "label": FAMILY_LABELS[family], "note": FAMILY_NOTES[family], "latest_csv": latest_web, "daily_history_csv": daily_web, "monthly_history_csv": monthly_web, "all_zip": all_zip, "tree": build_family_time_tree(family, daily)}


def copy_raw_snapshots(snapshots: list[Snapshot]) -> list[dict[str, Any]]:
    """Copy raw snapshots into docs downloads and build raw snapshot metadata."""
    records = []
    for snap in snapshots:
        group = "usall" if snap.universe == "usall" else "sp500" if snap.universe == "sp500" else "other_indices"
        out_dir = RAW_DOWNLOADS / group / snap.year / snap.month
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / snap.path.name
        shutil.copy2(snap.path, out_path)
        records.append({"raw_group": group, "date": snap.date, "year": snap.year, "month": snap.month, "universe": snap.universe, "n_firms": int(len(snap.df)), "csv": to_web_path(out_path)})
    return records


def build_raw_tree(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build grouped raw snapshot tree with real year/month zip archives."""
    result: dict[str, Any] = {}
    for group in ["usall", "sp500", "other_indices"]:
        group_records = [r for r in records if r["raw_group"] == group]
        years = []
        for year in sorted({r["year"] for r in group_records}, reverse=True):
            year_records = [r for r in group_records if r["year"] == year]
            year_files = [DOCS_DIR / r["csv"].replace("./", "") for r in year_records]
            months = []
            for month in sorted({r["month"] for r in year_records}, reverse=True):
                month_records = [r for r in year_records if r["month"] == month]
                month_files = [DOCS_DIR / r["csv"].replace("./", "") for r in month_records]
                months.append({"month": f"{year}-{month}", "n_files": len(month_records), "download_all_zip": make_zip(RAW_DOWNLOADS / group / year / f"raw_{group}_{year}_{month}_all.zip", month_files), "days": sorted(month_records, key=lambda r: (r["date"], r["universe"]), reverse=True)})
            years.append({"year": year, "n_files": len(year_records), "download_all_zip": make_zip(RAW_DOWNLOADS / group / f"raw_{group}_{year}_all.zip", year_files), "months": months})
        result[group] = {"label": {"usall": "All-market raw snapshots", "sp500": "S&P 500 raw snapshots", "other_indices": "Other index raw snapshots"}[group], "years": years}
    return result


def trim_display_months(df: pd.DataFrame, group_col: str | None = None, n: int = 3) -> pd.DataFrame:
    """Keep last n monthly rows overall or per group for page display."""
    if df.empty:
        return df.copy()
    sort_col = "month_end_date" if "month_end_date" in df.columns else "date"
    if group_col and group_col in df.columns:
        return df.sort_values(sort_col, ascending=False).groupby(group_col, dropna=False).head(n).sort_values([group_col, sort_col], ascending=[True, False]).reset_index(drop=True)
    return df.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)


def write_family_json(family: str, latest: pd.DataFrame, daily: pd.DataFrame, monthly: pd.DataFrame, downloads: dict[str, Any]) -> None:
    """Write the page JSON for one family."""
    group_map = {"marketwide": "family", "industry": "group", "indices": "family", "etf": "ticker", "country": "country"}
    display_monthly = trim_display_months(monthly, group_map.get(family), 3)
    write_json(DOCS_DATA / f"{family}.json", {"family": family, "label": FAMILY_LABELS[family], "note": FAMILY_NOTES[family], "latest": latest.to_dict(orient="records"), "monthly": display_monthly.to_dict(orient="records"), "downloads": downloads})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    _args = parser.parse_args()
    ensure_dirs()
    snapshots = discover_snapshots()
    if not snapshots:
        raise RuntimeError("No valid raw ICC snapshots found under data/YYYYMM.")
    usall_snaps = [s for s in snapshots if s.universe == "usall"]
    if not usall_snaps:
        raise RuntimeError("No valid usall snapshot found.")
    latest_usall = sorted(usall_snaps, key=lambda s: s.date)[-1]

    # Always refresh ETF and Country / Region derived data for the latest available all-market date.
    print(f"[build_docs_data] refreshing ETF and Country / Region data for {latest_usall.date}")
    build_etf_icc_panel(latest_usall.df, latest_usall.date)
    build_country_region_icc_panel(latest_usall.date)

    latest_mkt, daily_mkt, monthly_mkt = build_marketwide(snapshots)
    latest_val, daily_val, monthly_val = build_value(snapshots)
    latest_ind, daily_ind, monthly_ind = build_industry(snapshots)
    latest_idx, daily_idx, monthly_idx = build_indices(snapshots)
    latest_etf, daily_etf, monthly_etf = read_derived_family("etf")
    latest_cty, daily_cty, monthly_cty = read_derived_family("country")

    family_data = {
        "marketwide": (latest_mkt, daily_mkt, monthly_mkt),
        "value": (latest_val, daily_val, monthly_val),
        "industry": (latest_ind, daily_ind, monthly_ind),
        "indices": (latest_idx, daily_idx, monthly_idx),
        "etf": (latest_etf, daily_etf, monthly_etf),
        "country": (latest_cty, daily_cty, monthly_cty),
    }
    family_catalog = {}
    for family in FAMILY_ORDER:
        latest, daily, monthly = family_data[family]
        downloads = build_family_downloads(family, latest, daily, monthly)
        family_catalog[family] = downloads
        write_family_json(family, latest, daily, monthly, downloads)

    raw_records = copy_raw_snapshots(snapshots)
    raw_tree = build_raw_tree(raw_records)
    write_json(DOCS_DATA / "downloads_catalog.json", {"families": family_catalog, "raw_snapshots": raw_tree})

    overview_rows = []
    for family in FAMILY_ORDER:
        latest, _daily, monthly = family_data[family]
        if family == "marketwide":
            for group in ["all_market", "sp500"]:
                lr = latest[latest["family"] == group].iloc[0].to_dict() if not latest.empty and (latest["family"] == group).any() else {}
                mm = trim_display_months(monthly[monthly["family"] == group] if not monthly.empty else monthly, None, 3)
                overview_rows.append({"dataset": "All market" if group == "all_market" else "S&P 500", "latest_daily": lr.get("daily_icc"), "method": lr.get("method"), "month_1": mm.iloc[0].get("daily_icc") if len(mm) > 0 else None, "month_2": mm.iloc[1].get("daily_icc") if len(mm) > 1 else None, "month_3": mm.iloc[2].get("daily_icc") if len(mm) > 2 else None})
            continue
        latest_row = latest.iloc[0].to_dict() if not latest.empty else {}
        value_key = "ivp" if family == "value" else "daily_icc" if family in {"industry", "indices"} else "icc"
        group_col = "group" if family == "industry" else "family" if family == "indices" else "ticker" if family == "etf" else "country" if family == "country" else None
        mm = trim_display_months(monthly, group_col, 3)
        overview_rows.append({"dataset": FAMILY_LABELS[family], "latest_daily": latest_row.get(value_key), "method": latest_row.get("method"), "month_1": mm.iloc[0].get(value_key) if len(mm) > 0 else None, "month_2": mm.iloc[1].get(value_key) if len(mm) > 1 else None, "month_3": mm.iloc[2].get(value_key) if len(mm) > 2 else None})
    write_json(DOCS_DATA / "overview.json", {"title": "Live ICC data library", "rows": overview_rows, "families": family_catalog})
    print(f"[build_docs_data] valid raw snapshots = {len(snapshots)}")
    print("[build_docs_data] wrote JSON pages, organized downloads, and zip archives")


if __name__ == "__main__":
    main()
