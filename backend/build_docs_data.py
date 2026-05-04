from __future__ import annotations

import argparse
import os
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from backend.ensure_daily_derived import ensure_daily_derived

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DERIVED_DIR = DATA_DIR / "derived"
DOCS_DATA = REPO / "docs" / "data"
DOWNLOADS = DOCS_DATA / "downloads"
ZIP_DIR = DOWNLOADS / "zip"
CONFIG_DIR = REPO / "config"

SNAPSHOT_RE = re.compile(r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$")
INDEX_UNIVERSES = ["sp500", "sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"]


def safe_json(obj):
    """Convert pandas/numpy objects into strict JSON-safe values."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [safe_json(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    return obj


def write_json(path: Path, payload: dict) -> None:
    """Write strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_json(payload), f, indent=2, allow_nan=False)


def parse_snapshot_meta(path: Path) -> dict | None:
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
        "date": f"{year}-{mmdd[:2]}-{mmdd[2:]}",
        "yyyymm": f"{year}{mmdd[:2]}",
        "yyyymmdd": f"{year}{mmdd}",
        "rerun": int(m.group("rerun") or 0),
    }


def load_snapshot(path: Path) -> pd.DataFrame:
    """Load one firm-level ICC snapshot."""
    if path.stat().st_size <= 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise EmptyDataError(f"Empty CSV: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    for c in ["mktcap", "ICC", "bm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "bm" not in df.columns:
        df["bm"] = np.nan
    if "sector" not in df.columns:
        df["sector"] = None
    if "name" not in df.columns:
        df["name"] = None
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[df["ticker"].notna() & df["date"].notna() & df["ICC"].notna() & df["mktcap"].notna() & (df["mktcap"] > 0)].copy()
    return df


def find_snapshots(universe: str | None = None) -> list[Path]:
    """Find snapshot CSV files under data/YYYYMM."""
    rows = []
    for p in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(p)
        if not meta:
            continue
        if universe is not None and meta["universe"] != universe:
            continue
        rows.append(((meta["universe"], meta["yyyymmdd"], meta["rerun"], p.name), p))
    rows.sort(key=lambda x: x[0])
    return [p for _, p in rows]


def get_valid_snapshots(paths: list[Path]) -> list[tuple[Path, pd.DataFrame]]:
    """Load and de-duplicate valid snapshots by universe and date."""
    keep: dict[tuple[str, str], tuple[Path, pd.DataFrame, int]] = {}
    for p in paths:
        meta = parse_snapshot_meta(p)
        if not meta:
            continue
        try:
            df = load_snapshot(p)
        except Exception as exc:
            print(f"[build_docs_data] skip invalid snapshot {p}: {type(exc).__name__}: {exc}")
            continue
        if df.empty:
            continue
        key = (meta["universe"], str(df["date"].iloc[0]))
        old = keep.get(key)
        if old is None or meta["rerun"] >= old[2]:
            keep[key] = (p, df, meta["rerun"])
    out = [(p, df) for p, df, _ in keep.values()]
    out.sort(key=lambda x: (parse_snapshot_meta(x[0])["universe"], str(x[1]["date"].iloc[0])))
    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute weighted mean with positive finite weights."""
    mask = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def build_market_history(valid: list[tuple[Path, pd.DataFrame]], family: str) -> pd.DataFrame:
    """Build daily marketwide/index ICC history."""
    rows = []
    for path, df in valid:
        rows.append({
            "date": str(df["date"].iloc[0]),
            "family": family,
            "daily_icc": weighted_mean(df["ICC"], df["mktcap"]),
            "ew_icc": float(df["ICC"].mean()),
            "n_firms": int(len(df)),
            "method": "ICC calculation",
            "source_file": str(path.relative_to(REPO)),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def build_value_history(valid: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    """Build daily value/growth/IVP history from all-market snapshots."""
    rows = []
    for path, raw in valid:
        df = raw.copy()
        df = df[df["bm"].notna() & np.isfinite(df["bm"]) & (df["bm"] > 0)].copy()
        if df.empty:
            continue
        lo, hi = df["ICC"].quantile(0.005), df["ICC"].quantile(0.995)
        df = df[(df["ICC"] >= lo) & (df["ICC"] <= hi)].copy()
        if df.empty:
            continue
        size_med = float(df["mktcap"].median())
        bm30, bm70 = float(df["bm"].quantile(0.30)), float(df["bm"].quantile(0.70))
        df["size_bucket"] = np.where(df["mktcap"] <= size_med, "S", "B")
        df["bm_bucket"] = pd.cut(df["bm"], [-np.inf, bm30, bm70, np.inf], labels=["L", "M", "H"]).astype(str)
        df["portfolio"] = df["size_bucket"] + "/" + df["bm_bucket"]
        d = df.groupby("portfolio").apply(lambda g: weighted_mean(g["ICC"], g["mktcap"])).to_dict()
        growth = np.nan if pd.isna(d.get("S/L", np.nan)) or pd.isna(d.get("B/L", np.nan)) else float((d["S/L"] + d["B/L"]) / 2)
        value = np.nan if pd.isna(d.get("S/H", np.nan)) or pd.isna(d.get("B/H", np.nan)) else float((d["S/H"] + d["B/H"]) / 2)
        ivp = np.nan if pd.isna(value) or pd.isna(growth) else float(value - growth)
        rows.append({"date": str(df["date"].iloc[0]), "family": "value_premium", "value_icc": value, "growth_icc": growth, "ivp": ivp, "daily_icc": ivp, "n_firms": int(len(df)), "method": "ICC calculation", "source_file": str(path.relative_to(REPO))})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def build_industry_history(valid: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    """Build daily industry ICC history."""
    rows = []
    for path, df in valid:
        df = df[df["sector"].notna()].copy()
        for sector, g in df.groupby("sector"):
            rows.append({"date": str(g["date"].iloc[0]), "family": "industry", "group": str(sector), "daily_icc": weighted_mean(g["ICC"], g["mktcap"]), "ew_icc": float(g["ICC"].mean()), "n_firms": int(len(g)), "method": "ICC calculation", "source_file": str(path.relative_to(REPO))})
    return pd.DataFrame(rows).sort_values(["date", "group"]).reset_index(drop=True) if rows else pd.DataFrame()


def monthly_from_daily(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Convert daily records into month-end records."""
    if df.empty:
        return df.copy()
    group_cols = group_cols or []
    out = df.copy()
    date_col = "date"
    out["month"] = pd.to_datetime(out[date_col]).dt.strftime("%Y-%m")
    out = out.sort_values(date_col)
    monthly = out.groupby(group_cols + ["month"], as_index=False).tail(1).copy()
    monthly = monthly.rename(columns={date_col: "month_end_date"})
    return monthly.reset_index(drop=True)


def latest_daily(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Return latest records, optionally by group."""
    if df.empty:
        return df.copy()
    out = df.sort_values("date")
    if group_cols:
        return out.groupby(group_cols, as_index=False).tail(1).reset_index(drop=True)
    return out.tail(1).reset_index(drop=True)


def save_family(name: str, daily: pd.DataFrame, monthly: pd.DataFrame, group_cols: list[str] | None = None) -> dict:
    """Save a dataset family to CSV and ZIP downloads."""
    fam_dir = DOWNLOADS / name
    fam_dir.mkdir(parents=True, exist_ok=True)
    latest = latest_daily(daily, group_cols)
    latest_path = fam_dir / f"{name}_latest.csv"
    daily_path = fam_dir / f"{name}_daily_history.csv"
    monthly_path = fam_dir / f"{name}_monthly_history.csv"
    latest.to_csv(latest_path, index=False)
    daily.to_csv(daily_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ZIP_DIR / f"{name}_all.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in [latest_path, daily_path, monthly_path]:
            z.write(p, p.relative_to(DOCS_DATA))
    return {"latest": f"./data/{latest_path.relative_to(DOCS_DATA).as_posix()}", "daily": f"./data/{daily_path.relative_to(DOCS_DATA).as_posix()}", "monthly": f"./data/{monthly_path.relative_to(DOCS_DATA).as_posix()}", "zip": f"./data/{zip_path.relative_to(DOCS_DATA).as_posix()}"}


def save_derived(kind: str, panel: pd.DataFrame, date: str) -> Path:
    """Save a derived daily panel with one file per date."""
    yyyymm = date[:7].replace("-", "")
    out_dir = DERIVED_DIR / kind / yyyymm
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{kind}_icc_{date.replace('-', '_')}.csv"
    panel = panel.copy()
    if "date" not in panel.columns:
        panel.insert(0, "date", date)
    else:
        panel["date"] = date
    panel.to_csv(out, index=False)
    return out


def load_derived(kind: str) -> pd.DataFrame:
    """Load derived history files."""
    frames = []
    for p in sorted((DERIVED_DIR / kind).glob("*/*.csv")):
        try:
            df = pd.read_csv(p)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "date" in out.columns:
        out = out.drop_duplicates([c for c in ["date", "ticker", "country"] if c in out.columns], keep="last")
    return out


def copy_raw(valid_all: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    """Copy raw snapshots into docs/data/downloads/raw and generate manifests."""
    raw_dir = DOWNLOADS / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, df in valid_all:
        meta = parse_snapshot_meta(path)
        if not meta:
            continue
        group = "usall" if meta["universe"] == "usall" else "sp500" if meta["universe"] == "sp500" else "other_indices"
        target_dir = raw_dir / group / meta["yyyymm"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)
        rows.append({"date": str(df["date"].iloc[0]), "year": meta["year"], "month": meta["mmdd"][:2], "yyyymm": meta["yyyymm"], "universe": meta["universe"], "raw_group": group, "n_firms": int(len(df)), "download_path": f"./data/{target.relative_to(DOCS_DATA).as_posix()}"})
    return pd.DataFrame(rows).sort_values(["raw_group", "date", "universe"]).reset_index(drop=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    all_valid = get_valid_snapshots(find_snapshots(None))
    by_universe: dict[str, list[tuple[Path, pd.DataFrame]]] = {}
    for p, df in all_valid:
        meta = parse_snapshot_meta(p)
        if meta:
            by_universe.setdefault(meta["universe"], []).append((p, df))
    usall_valid = by_universe.get("usall", [])
    if not usall_valid:
        raise RuntimeError("No valid usall snapshots found")
    latest_date = str(usall_valid[-1][1]["date"].iloc[0])
    latest_usall = usall_valid[-1][1]

    all_market = build_market_history(usall_valid, "all_market")
    sp500 = build_market_history(by_universe.get("sp500", []), "sp500")
    marketwide = pd.concat([all_market, sp500], ignore_index=True) if not sp500.empty else all_market
    value = build_value_history(usall_valid)
    industry = build_industry_history(usall_valid)
    index_frames = [build_market_history(by_universe[u], u) for u in INDEX_UNIVERSES if u in by_universe]
    indices = pd.concat(index_frames, ignore_index=True) if index_frames else pd.DataFrame()

    # Ensure ETF and Country / Region derived files exist for the latest raw usall date.
    # This prevents the website from staying on stale ETF/Country data when raw snapshots update.
    force_derived = os.environ.get("FORCE_DERIVED_REBUILD", "0") == "1"
    ensure_daily_derived(args.universe, force=force_derived)

    etf_daily = load_derived("etf")
    if not etf_daily.empty:
        if "icc" in etf_daily.columns:
            etf_daily["daily_icc"] = pd.to_numeric(etf_daily["icc"], errors="coerce")
        if "ticker" not in etf_daily.columns:
            etf_daily["ticker"] = "ETF"

    country_daily = load_derived("country_adr")
    if not country_daily.empty:
        if "icc" in country_daily.columns:
            country_daily["daily_icc"] = pd.to_numeric(country_daily["icc"], errors="coerce")
        if "country" not in country_daily.columns and "country_region" in country_daily.columns:
            country_daily["country"] = country_daily["country_region"]
        if "country_region" not in country_daily.columns and "country" in country_daily.columns:
            country_daily["country_region"] = country_daily["country"]

    families = {
        "marketwide": (marketwide, monthly_from_daily(marketwide, ["family"]), ["family"]),
        "value": (value, monthly_from_daily(value), None),
        "industry": (industry, monthly_from_daily(industry, ["group"]), ["group"]),
        "indices": (indices, monthly_from_daily(indices, ["family"]), ["family"]),
        "etf": (etf_daily, monthly_from_daily(etf_daily, ["ticker"]) if not etf_daily.empty else pd.DataFrame(), ["ticker"]),
        "country": (country_daily, monthly_from_daily(country_daily, ["country"]) if not country_daily.empty else pd.DataFrame(), ["country"]),
    }
    downloads = {k: save_family(k, v[0], v[1], v[2]) for k, v in families.items()}
    raw_manifest = copy_raw(all_valid)

    def payload(name: str) -> dict:
        daily, monthly, group = families[name]
        return {"latest": latest_daily(daily, group).to_dict(orient="records") if not daily.empty else [], "daily": daily.to_dict(orient="records") if not daily.empty else [], "monthly": monthly.to_dict(orient="records") if not monthly.empty else [], "downloads": downloads[name]}

    overview_rows = []
    for name, label in [("marketwide", "Marketwide ICC"), ("value", "Value Premium ICC"), ("industry", "Industry ICC"), ("indices", "Index ICC"), ("etf", "ETF ICC"), ("country", "Country / Region Level ICC")]:
        daily = families[name][0]
        latest = latest_daily(daily, families[name][2]) if not daily.empty else pd.DataFrame()
        val_col = "daily_icc" if "daily_icc" in latest.columns else "icc" if "icc" in latest.columns else "ivp" if "ivp" in latest.columns else None
        overview_rows.append({"dataset": label, "latest_date": latest_date, "latest_daily": float(latest[val_col].dropna().mean()) if val_col and not latest.empty and latest[val_col].notna().any() else None, "method": "ICC calculation" if name not in {"etf", "country"} else "Mixed; see dataset page"})
    write_json(DOCS_DATA / "overview.json", {"title": "Live ICC data library", "asof_date": latest_date, "rows": overview_rows, "downloads": downloads})
    write_json(DOCS_DATA / "marketwide.json", payload("marketwide"))
    write_json(DOCS_DATA / "value.json", payload("value"))
    write_json(DOCS_DATA / "industry.json", payload("industry"))
    write_json(DOCS_DATA / "indices.json", payload("indices"))
    write_json(DOCS_DATA / "etf.json", {**payload("etf"), "method_note": "ETF ICC is recalculated from the latest all-market snapshot date using online holdings sources."})
    write_json(DOCS_DATA / "country.json", {**payload("country"), "method_note": "Country / Region Level ICC uses a Country ADR Top-10 composite built from online ADR-like listings."})
    write_json(DOCS_DATA / "downloads_catalog.json", {"families": downloads, "raw_snapshots": raw_manifest.to_dict(orient="records") if not raw_manifest.empty else []})

    # Legacy aliases for older pages or cached frontend code.
    write_json(DOCS_DATA / "etf_icc.json", {**payload("etf")})
    write_json(DOCS_DATA / "country_icc.json", {**payload("country")})
    write_json(DOCS_DATA / "market_icc.json", {**payload("marketwide")})
    write_json(DOCS_DATA / "index_icc.json", {**payload("indices")})
    write_json(DOCS_DATA / "value_icc_bm.json", {**payload("value")})
    write_json(DOCS_DATA / "industry_icc.json", {**payload("industry")})
    print(f"[build_docs_data] wrote docs/data using latest usall date {latest_date}")


if __name__ == "__main__":
    main()
