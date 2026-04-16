
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pandas.errors import EmptyDataError

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DIR = REPO / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"
DOWNLOAD_DIR = DOCS_DATA_DIR / "downloads"
RAW_DIR = DOWNLOAD_DIR / "raw"
RAW_ZIP_DIR = DOWNLOAD_DIR / "raw_zips"
CONFIG_DIR = REPO / "config"

SNAPSHOT_RE = re.compile(r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$")
SPDR_TICKERS = {"SPY", "DIA", "XLF", "XLK", "XLE", "XLV", "XLY", "XLI"}

INDEX_FAMILIES = ["sp500", "sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"]


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)


def parse_snapshot_meta(path: Path) -> dict | None:
    m = SNAPSHOT_RE.match(path.name)
    if not m:
        return None
    year = m.group("year")
    mmdd = m.group("mmdd")
    rerun = int(m.group("rerun") or 0)
    return {
        "universe": m.group("universe"),
        "year": year,
        "mmdd": mmdd,
        "yyyymm": f"{year}{mmdd[:2]}",
        "yyyymmdd": f"{year}{mmdd}",
        "rerun": rerun,
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def normalize_symbol(s: str) -> str:
    return str(s).strip().upper().replace(".", "-")


def normalize_name(s: str) -> str:
    s = str(s).upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    for token in ["CORPORATION", "CORP", "INCORPORATED", "INC", "PLC", "LIMITED", "LTD", "CO", "COMPANY", "HOLDINGS", "HOLDING", "GROUP", "CLASS", "CL", "ADR", "ADS", "SA", "AG", "NV", "SPA", "THE"]:
        s = re.sub(rf"\b{token}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_float(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        y = float(x)
        return y if np.isfinite(y) else None
    except Exception:
        return None


def safe_int(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        return None


def find_all_snapshots(universe: str | None = None) -> list[Path]:
    files = []
    for p in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(p)
        if meta is None:
            continue
        if universe is not None and meta["universe"] != universe:
            continue
        files.append(((meta["universe"], meta["yyyymmdd"], meta["rerun"], p.name), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        raise EmptyDataError(f"No columns parsed: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
    for col in ["mktcap", "ICC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "bm" in df.columns:
        df["bm"] = pd.to_numeric(df["bm"], errors="coerce")
    else:
        df["bm"] = np.nan
    if "sector" not in df.columns:
        df["sector"] = None
    if "name" not in df.columns:
        df["name"] = None
    df["ticker"] = df["ticker"].astype(str).map(normalize_symbol)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["norm_name"] = df["name"].astype(str).map(normalize_name)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["ticker"].notna()
        & df["date"].notna()
        & df["mktcap"].notna()
        & df["ICC"].notna()
        & np.isfinite(df["mktcap"])
        & np.isfinite(df["ICC"])
        & (df["mktcap"] > 0)
    ].copy()


def get_valid_snapshots(paths: list[Path]) -> list[tuple[Path, pd.DataFrame]]:
    out = []
    for path in paths:
        try:
            df = clean_df(load_snapshot(path))
            if df.empty:
                continue
            out.append((path, df))
        except Exception as e:
            print(f"[build_docs_data] skip invalid snapshot {path}: {type(e).__name__}: {e}")
    return out


def dedup_valid_snapshots(valid: list[tuple[Path, pd.DataFrame]]) -> list[tuple[Path, pd.DataFrame]]:
    keep = {}
    for path, df in valid:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        dt = str(df["date"].dropna().iloc[0]) if df["date"].notna().any() else None
        if not dt:
            continue
        keep[(meta["universe"], dt)] = (path, df)
    out = list(keep.values())
    out.sort(key=lambda x: (parse_snapshot_meta(x[0])["universe"], str(x[1]["date"].iloc[0]), parse_snapshot_meta(x[0])["rerun"], x[0].name))
    return out


def build_market_history(valid: list[tuple[Path, pd.DataFrame]], family_name: str) -> pd.DataFrame:
    rows = []
    for path, df in valid:
        rows.append({
            "family": family_name,
            "date": str(df["date"].iloc[0]),
            "vw_icc": weighted_mean(df["ICC"], df["mktcap"]),
            "ew_icc": float(df["ICC"].mean()),
            "n_items": int(len(df)),
            "source_file": str(path.relative_to(REPO)),
        })
    if not rows:
        return pd.DataFrame(columns=["family", "date", "vw_icc", "ew_icc", "n_items", "source_file"])
    out = pd.DataFrame(rows).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    out["yyyymm"] = out["date"].str.slice(0, 7)
    return out


def build_value_history(valid: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, raw in valid:
        df = raw[
            raw["bm"].notna()
            & np.isfinite(raw["bm"])
            & (raw["bm"] > 0)
        ].copy()
        if df.empty:
            continue
        q_lo = float(df["ICC"].quantile(0.005))
        q_hi = float(df["ICC"].quantile(0.995))
        df = df[(df["ICC"] >= q_lo) & (df["ICC"] <= q_hi)].copy()
        if df.empty:
            continue
        size_median = float(df["mktcap"].median())
        bm30 = float(df["bm"].quantile(0.30))
        bm70 = float(df["bm"].quantile(0.70))
        df["size_bucket"] = np.where(df["mktcap"] <= size_median, "S", "B")
        df["bm_bucket"] = pd.cut(df["bm"], bins=[-np.inf, bm30, bm70, np.inf], labels=["L", "M", "H"], include_lowest=True)
        df["portfolio"] = df["size_bucket"] + "/" + df["bm_bucket"].astype(str)
        pt = df.groupby("portfolio", dropna=False).apply(lambda g: pd.Series({
            "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
            "n_items": int(len(g)),
        })).reset_index()
        d = dict(zip(pt["portfolio"], pt["vw_icc"]))
        value_icc = np.nan if pd.isna(d.get("S/H")) or pd.isna(d.get("B/H")) else float((d["S/H"] + d["B/H"]) / 2)
        growth_icc = np.nan if pd.isna(d.get("S/L")) or pd.isna(d.get("B/L")) else float((d["S/L"] + d["B/L"]) / 2)
        ivp_bm = np.nan if pd.isna(value_icc) or pd.isna(growth_icc) else float(value_icc - growth_icc)
        rows.append({
            "date": str(df["date"].iloc[0]),
            "value_icc": value_icc,
            "growth_icc": growth_icc,
            "ivp_bm": ivp_bm,
            "n_items": int(len(df)),
            "source_file": str(path.relative_to(REPO)),
        })
    if not rows:
        return pd.DataFrame(columns=["date", "value_icc", "growth_icc", "ivp_bm", "n_items", "source_file"])
    out = pd.DataFrame(rows).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    out["yyyymm"] = out["date"].str.slice(0, 7)
    return out


def build_industry_daily(valid: list[tuple[Path, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_table = pd.DataFrame()
    summary_rows = []
    for path, df in valid:
        day = str(df["date"].iloc[0])
        tab = df[df["sector"].notna()].groupby("sector", dropna=False).apply(lambda g: pd.Series({
            "date": day,
            "sector": str(g.name),
            "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
            "ew_icc": float(g["ICC"].mean()),
            "n_items": int(len(g)),
        })).reset_index(drop=True)
        if not tab.empty:
            latest_table = tab
            summary_rows.append({"date": day, "vw_icc": weighted_mean(tab["vw_icc"], pd.Series(np.ones(len(tab)))), "n_items": int(len(tab))})
    summary = pd.DataFrame(summary_rows).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    if not summary.empty:
        summary["yyyymm"] = summary["date"].str.slice(0, 7)
    return latest_table, summary


def monthly_last(df: pd.DataFrame, value_cols: list[str], family_col: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    tmp = df.copy()
    tmp["yyyymm"] = tmp["date"].str.slice(0, 7)
    sort_cols = [family_col, "date"] if family_col else ["date"]
    tmp = tmp.sort_values(sort_cols)
    group_cols = [family_col, "yyyymm"] if family_col else ["yyyymm"]
    out = tmp.groupby(group_cols, as_index=False).tail(1).reset_index(drop=True)
    return out


def latest_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values("date").tail(1).reset_index(drop=True)


def try_read_spdr_holdings(ticker: str) -> pd.DataFrame | None:
    if ticker.upper() not in SPDR_TICKERS:
        return None
    # This direct pattern is documented by the publicly accessible SPY holdings workbook URL.
    url = f"https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"
    try:
        df = pd.read_excel(url)
    except Exception:
        return None
    cols = {str(c).strip().lower(): c for c in df.columns}
    symbol_col = None
    name_col = None
    weight_col = None
    for k in cols:
        if k in {"ticker", "symbol", "sedol ticker"}:
            symbol_col = cols[k]
        if k in {"name", "security name"}:
            name_col = cols[k]
        if "weight" in k:
            weight_col = cols[k]
    if weight_col is None:
        return None
    out = pd.DataFrame({
        "symbol": df[symbol_col].astype(str).map(normalize_symbol) if symbol_col else None,
        "name": df[name_col].astype(str).map(normalize_name) if name_col else None,
        "weight": pd.to_numeric(df[weight_col], errors="coerce"),
    })
    out = out[out["weight"].notna()].copy()
    if out["weight"].max() > 1.5:
        out["weight"] = out["weight"] / 100.0
    out = out[out["weight"] > 0].copy()
    if out.empty:
        return None
    out["weight"] = out["weight"] / out["weight"].sum()
    return out[["symbol", "name", "weight"]]


def try_read_yf_top_holdings(ticker: str) -> pd.DataFrame | None:
    try:
        t = yf.Ticker(ticker)
        fd = getattr(t, "funds_data", None)
        if fd is None:
            return None
        holdings = getattr(fd, "top_holdings", None)
        if holdings is None:
            return None
        h = holdings.copy() if isinstance(holdings, pd.DataFrame) else pd.DataFrame(holdings)
        if h.empty:
            return None
    except Exception:
        return None
    cols = {str(c).strip().lower(): c for c in h.columns}
    symbol_col = next((cols[c] for c in ["symbol", "ticker", "holding"] if c in cols), None)
    name_col = next((cols[c] for c in ["name", "holding", "company"] if c in cols), None)
    weight_col = None
    for c in ["holdingpercent", "holding_percent", "weight", "percent", "pct"]:
        if c in cols:
            weight_col = cols[c]
            break
    if weight_col is None:
        # Last numeric-ish column fallback.
        for c in h.columns[::-1]:
            vals = pd.to_numeric(h[c], errors="coerce")
            if vals.notna().sum() >= max(1, len(h) // 2):
                weight_col = c
                break
    if weight_col is None:
        return None
    out = pd.DataFrame({
        "symbol": h[symbol_col].astype(str).map(normalize_symbol) if symbol_col else None,
        "name": h[name_col].astype(str).map(normalize_name) if name_col else None,
        "weight": pd.to_numeric(h[weight_col], errors="coerce"),
    })
    out = out[out["weight"].notna()].copy()
    if out["weight"].max() > 1.5:
        out["weight"] = out["weight"] / 100.0
    out = out[out["weight"] > 0].copy()
    if out.empty:
        return None
    out["weight"] = out["weight"] / out["weight"].sum()
    return out[["symbol", "name", "weight"]]


def fetch_online_holdings(ticker: str) -> tuple[pd.DataFrame | None, str]:
    spdr = try_read_spdr_holdings(ticker)
    if spdr is not None and not spdr.empty:
        return spdr, "spdr_official_holdings"
    yf_h = try_read_yf_top_holdings(ticker)
    if yf_h is not None and not yf_h.empty:
        return yf_h, "yfinance_top_holdings"
    return None, "unavailable"


def match_holdings_to_usall(holdings: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    if holdings is None or holdings.empty:
        return pd.DataFrame()
    base_sym = base[["ticker", "norm_name", "ICC"]].copy()
    by_sym = holdings.merge(base_sym, left_on="symbol", right_on="ticker", how="left")
    needs_name = by_sym["ICC"].isna()
    if needs_name.any():
        name_map = base[["norm_name", "ICC"]].dropna().drop_duplicates(subset=["norm_name"])
        fix = by_sym.loc[needs_name, ["name", "weight"]].merge(name_map, left_on="name", right_on="norm_name", how="left")
        by_sym.loc[needs_name, "ICC"] = fix["ICC"].values
    return by_sym


def try_etf_proxy_icc(ticker: str) -> tuple[float | None, str]:
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}
    except Exception:
        return None, "unavailable"
    pe = info.get("trailingPE") or info.get("forwardPE")
    yld = info.get("yield")
    if yld is None:
        yld = info.get("dividendYield")
    try:
        pe = float(pe) if pe is not None else None
    except Exception:
        pe = None
    try:
        yld = float(yld) if yld is not None else 0.0
    except Exception:
        yld = 0.0
    if yld is not None and yld > 1.5:
        yld = yld / 100.0
    if pe is not None and pe > 0:
        # Engineering proxy based on public ETF valuation metrics.
        return float(1.0 / pe + (yld or 0.0)), "public_valuation_proxy"
    return None, "unavailable"


def build_holdings_or_proxy(rows_cfg: pd.DataFrame, latest_usall: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = latest_usall.copy()
    rows = []
    summary_rows = []
    today = str(base["date"].iloc[0])

    for _, cfg in rows_cfg.iterrows():
        ticker = normalize_symbol(cfg.get("ticker", ""))
        if not ticker:
            continue
        label = str(cfg.get("label", ticker))
        category = str(cfg.get("category", "")) if "category" in cfg.index else ""
        country = str(cfg.get("country", "")) if "country" in cfg.index else ""

        holdings, source = fetch_online_holdings(ticker)
        matched = match_holdings_to_usall(holdings, base) if holdings is not None else pd.DataFrame()
        coverage = float(matched.loc[matched["ICC"].notna(), "weight"].sum()) if not matched.empty else 0.0
        holdings_icc = weighted_mean(matched["ICC"], matched["weight"]) if not matched.empty else np.nan

        proxy_icc, proxy_source = try_etf_proxy_icc(ticker)

        if safe_float(holdings_icc) is not None and coverage >= 0.10:
            final_icc = float(holdings_icc)
            final_source = source
            status = "ok"
        elif proxy_icc is not None:
            final_icc = float(proxy_icc)
            final_source = proxy_source
            status = "proxy"
        else:
            final_icc = np.nan
            final_source = source if source != "unavailable" else proxy_source
            status = "unavailable"

        row = {
            "date": today,
            "ticker": ticker,
            "label": label,
            "vw_icc": final_icc,
            "coverage_weight": coverage,
            "n_holdings": int(len(holdings)) if holdings is not None else None,
            "n_matched": int(matched["ICC"].notna().sum()) if not matched.empty else 0,
            "source": final_source,
            "status": status,
        }
        if kind == "etf":
            row["category"] = category
        else:
            row["country"] = country

        rows.append(row)

    latest_table = pd.DataFrame(rows)
    if latest_table.empty:
        if kind == "etf":
            latest_table = pd.DataFrame(columns=["date", "ticker", "label", "category", "vw_icc", "coverage_weight", "n_holdings", "n_matched", "source", "status"])
        else:
            latest_table = pd.DataFrame(columns=["date", "country", "ticker", "label", "vw_icc", "coverage_weight", "n_holdings", "n_matched", "source", "status"])

    agg_value = weighted_mean(latest_table["vw_icc"], pd.Series(np.ones(len(latest_table)))) if not latest_table.empty else np.nan
    summary = pd.DataFrame([{
        "date": today,
        f"{kind}_icc": agg_value,
        "n_items": int(len(latest_table)),
        "yyyymm": today[:7],
    }])
    return latest_table, summary


def merge_archived_summary(path: Path, new_summary: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, new_summary], ignore_index=True)
    else:
        out = new_summary.copy()
    out = out.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
    if "yyyymm" not in out.columns and date_col in out.columns:
        out["yyyymm"] = out[date_col].astype(str).str.slice(0, 7)
    return out


def merge_archived_table(path: Path, new_table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, new_table], ignore_index=True)
    else:
        out = new_table.copy()
    out = out.sort_values(keys).drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    return out


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def zip_paths(zip_path: Path, files: list[tuple[Path, str]]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files:
            if src.exists():
                zf.write(src, arcname=arc)


def build_family_downloads(family: str, latest_df: pd.DataFrame, daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> list[dict]:
    fam_dir = DOWNLOAD_DIR / family
    fam_dir.mkdir(parents=True, exist_ok=True)
    latest_path = fam_dir / "latest_daily_table.csv"
    daily_path = fam_dir / "daily_history.csv"
    monthly_path = fam_dir / "monthly_history.csv"
    save_df(latest_df, latest_path)
    save_df(daily_df, daily_path)
    save_df(monthly_df, monthly_path)
    zip_paths(fam_dir / "all_history_bundle.zip", [
        (latest_path, "latest_daily_table.csv"),
        (daily_path, "daily_history.csv"),
        (monthly_path, "monthly_history.csv"),
    ])
    return [
        {"label": "Latest daily table (CSV)", "path": f"./data/downloads/{family}/latest_daily_table.csv"},
        {"label": "Daily history (CSV)", "path": f"./data/downloads/{family}/daily_history.csv"},
        {"label": "Monthly history (CSV)", "path": f"./data/downloads/{family}/monthly_history.csv"},
        {"label": "Download all history (ZIP)", "path": f"./data/downloads/{family}/all_history_bundle.zip"},
    ]


def copy_raw_snapshots(valid: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, df in valid:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        target_dir = RAW_DIR / meta["yyyymm"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)
        universe = meta["universe"]
        bucket = "usall" if universe == "usall" else ("sp500" if universe == "sp500" else "other")
        rows.append({
            "bucket": bucket,
            "yyyymm": meta["yyyymm"],
            "year": meta["yyyymm"][:4],
            "month": meta["yyyymm"][4:6],
            "date": str(df["date"].iloc[0]),
            "universe": universe,
            "n_items": int(len(df)),
            "download_path": f"./data/downloads/raw/{meta['yyyymm']}/{path.name}",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["bucket", "date", "universe"]).drop_duplicates(subset=["bucket", "date", "universe"], keep="last").reset_index(drop=True)

    # Build year and month zip files.
    for bucket, g_bucket in out.groupby("bucket"):
        for year, g_year in g_bucket.groupby("year"):
            files = []
            for _, r in g_year.iterrows():
                rel = r["download_path"].replace("./data/downloads/", "")
                files.append((DOWNLOAD_DIR / rel, f"{r['yyyymm']}/{Path(rel).name}"))
            zip_paths(RAW_ZIP_DIR / bucket / f"{year}.zip", files)

        for (year, month), g_month in g_bucket.groupby(["year", "month"]):
            files = []
            for _, r in g_month.iterrows():
                rel = r["download_path"].replace("./data/downloads/", "")
                files.append((DOWNLOAD_DIR / rel, Path(rel).name))
            zip_paths(RAW_ZIP_DIR / bucket / year / f"{year}{month}.zip", files)

    return out


def build_downloads_catalog(family_downloads: dict, raw_rows: pd.DataFrame) -> dict:
    raw_tabs = {}
    for bucket in ["usall", "sp500", "other"]:
        g_bucket = raw_rows[raw_rows["bucket"] == bucket].copy() if not raw_rows.empty else pd.DataFrame()
        years = []
        if not g_bucket.empty:
            for year, g_year in sorted(g_bucket.groupby("year"), reverse=True):
                year_item = {
                    "year": year,
                    "zip_path": f"./data/downloads/raw_zips/{bucket}/{year}.zip",
                    "months": [],
                }
                for month, g_month in sorted(g_year.groupby("month"), reverse=True):
                    month_item = {
                        "month": month,
                        "zip_path": f"./data/downloads/raw_zips/{bucket}/{year}/{year}{month}.zip",
                        "files": g_month.sort_values("date", ascending=False).to_dict(orient="records"),
                    }
                    year_item["months"].append(month_item)
                years.append(year_item)
        raw_tabs[bucket] = years
    return {"families": family_downloads, "raw_tabs": raw_tabs}


def latest_month_values(df: pd.DataFrame, family_label: str, field: str) -> dict:
    if df.empty:
        return {
            "family": family_label,
            "latest_daily": None,
            "m1_date": None, "m1_value": None,
            "m2_date": None, "m2_value": None,
            "m3_date": None, "m3_value": None,
        }
    latest_daily = safe_float(df.sort_values("date").iloc[-1][field])
    monthly = monthly_last(df[["date", field]].copy(), [field]).sort_values("date", ascending=False).head(3).reset_index(drop=True)
    row = {"family": family_label, "latest_daily": latest_daily}
    for i in range(3):
        if i < len(monthly):
            row[f"m{i+1}_date"] = str(monthly.loc[i, "date"])
            row[f"m{i+1}_value"] = safe_float(monthly.loc[i, field])
        else:
            row[f"m{i+1}_date"] = None
            row[f"m{i+1}_value"] = None
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ZIP_DIR.mkdir(parents=True, exist_ok=True)

    all_valid = dedup_valid_snapshots(get_valid_snapshots(find_all_snapshots()))
    by_universe: dict[str, list[tuple[Path, pd.DataFrame]]] = {}
    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        if meta:
            by_universe.setdefault(meta["universe"], []).append((path, df))

    usall_valid = by_universe.get(args.universe, [])
    if not usall_valid:
        raise RuntimeError(f"No valid {args.universe} snapshots found.")
    latest_usall = usall_valid[-1][1]
    asof_date = str(latest_usall["date"].iloc[0])

    # Core families
    market_all_daily = build_market_history(usall_valid, "All U.S.")
    market_sp500_daily = build_market_history(by_universe.get("sp500", []), "S&P 500")
    marketwide_daily = pd.concat([market_all_daily, market_sp500_daily], ignore_index=True)
    marketwide_monthly = monthly_last(marketwide_daily, ["vw_icc", "ew_icc", "n_items"], family_col="family")
    marketwide_latest = marketwide_daily.sort_values("date").groupby("family", as_index=False).tail(1).reset_index(drop=True)

    value_daily = build_value_history(usall_valid)
    value_monthly = monthly_last(value_daily, ["value_icc", "growth_icc", "ivp_bm"])
    value_latest = latest_row(value_daily)

    industry_latest_table, industry_summary_daily = build_industry_daily(usall_valid)
    industry_summary_monthly = monthly_last(industry_summary_daily, ["vw_icc", "n_items"])
    industry_summary_latest = latest_row(industry_summary_daily)

    idx_daily_list = [build_market_history(by_universe.get(code, []), code.upper()) for code in INDEX_FAMILIES]
    idx_daily = pd.concat([df for df in idx_daily_list if not df.empty], ignore_index=True) if any(not df.empty for df in idx_daily_list) else pd.DataFrame(columns=["family", "date", "vw_icc", "ew_icc", "n_items", "source_file", "yyyymm"])
    idx_monthly = monthly_last(idx_daily, ["vw_icc", "ew_icc", "n_items"], family_col="family")
    idx_latest = idx_daily.sort_values("date").groupby("family", as_index=False).tail(1).reset_index(drop=True)

    # ETF / country online resources, archiving from now.
    etf_cfg = pd.read_csv(CONFIG_DIR / "etfs.csv")
    country_cfg = pd.read_csv(CONFIG_DIR / "country_etfs.csv")

    etf_latest_table, etf_today_summary = build_holdings_or_proxy(etf_cfg, latest_usall, "etf")
    country_latest_table, country_today_summary = build_holdings_or_proxy(country_cfg, latest_usall, "country")

    etf_daily_path = DOWNLOAD_DIR / "etf" / "daily_history.csv"
    etf_latest_path = DOWNLOAD_DIR / "etf" / "latest_daily_table.csv"
    country_daily_path = DOWNLOAD_DIR / "country" / "daily_history.csv"
    country_latest_path = DOWNLOAD_DIR / "country" / "latest_daily_table.csv"

    etf_daily = merge_archived_summary(etf_daily_path, etf_today_summary)
    country_daily = merge_archived_summary(country_daily_path, country_today_summary)
    etf_monthly = monthly_last(etf_daily, ["etf_icc", "n_items"])
    country_monthly = monthly_last(country_daily, ["country_icc", "n_items"])

    etf_latest_table_all = merge_archived_table(etf_latest_path, etf_latest_table, ["date", "ticker"])
    country_latest_table_all = merge_archived_table(country_latest_path, country_latest_table, ["date", "ticker"])

    # Persist family downloads
    family_downloads = {}
    family_downloads["marketwide"] = build_family_downloads("marketwide", marketwide_latest, marketwide_daily, marketwide_monthly)
    family_downloads["value"] = build_family_downloads("value", value_latest, value_daily, value_monthly)
    family_downloads["industry"] = build_family_downloads("industry", industry_latest_table, industry_summary_daily, industry_summary_monthly)
    family_downloads["indices"] = build_family_downloads("indices", idx_latest, idx_daily, idx_monthly)
    family_downloads["etf"] = build_family_downloads("etf", etf_latest_table_all[etf_latest_table_all["date"] == asof_date], etf_daily, etf_monthly)
    family_downloads["country"] = build_family_downloads("country", country_latest_table_all[country_latest_table_all["date"] == asof_date], country_daily, country_monthly)

    raw_rows = copy_raw_snapshots(all_valid)
    downloads_catalog = build_downloads_catalog(family_downloads, raw_rows)

    # Overview
    overview_rows = [
        latest_month_values(market_all_daily, "Marketwide (All U.S.)", "vw_icc"),
        latest_month_values(market_sp500_daily, "Marketwide (S&P 500)", "vw_icc"),
        latest_month_values(value_daily.rename(columns={"ivp_bm": "series_value"}), "Value Premium (IVP)", "series_value"),
        latest_month_values(industry_summary_daily.rename(columns={"vw_icc": "series_value"}), "Industry-wide ICC", "series_value"),
        latest_month_values(etf_daily.rename(columns={"etf_icc": "series_value"}), "ETF ICC", "series_value"),
        latest_month_values(country_daily.rename(columns={"country_icc": "series_value"}), "Country ICC", "series_value"),
        latest_month_values(idx_daily.rename(columns={"vw_icc": "series_value"}), "Indices (avg latest rows)", "series_value"),
    ]

    write_json(DOCS_DATA_DIR / "overview.json", {
        "asof_date": asof_date,
        "rows": overview_rows,
    })

    write_json(DOCS_DATA_DIR / "marketwide.json", {
        "asof_date": asof_date,
        "latest": marketwide_latest.to_dict(orient="records"),
        "daily": marketwide_daily.to_dict(orient="records"),
        "monthly": marketwide_monthly.to_dict(orient="records"),
        "downloads": family_downloads["marketwide"],
    })
    write_json(DOCS_DATA_DIR / "value.json", {
        "asof_date": asof_date,
        "latest": value_latest.to_dict(orient="records"),
        "daily": value_daily.to_dict(orient="records"),
        "monthly": value_monthly.to_dict(orient="records"),
        "downloads": family_downloads["value"],
    })
    write_json(DOCS_DATA_DIR / "industry.json", {
        "asof_date": asof_date,
        "latest_table": industry_latest_table.to_dict(orient="records"),
        "summary_latest": industry_summary_latest.to_dict(orient="records"),
        "summary_daily": industry_summary_daily.to_dict(orient="records"),
        "summary_monthly": industry_summary_monthly.to_dict(orient="records"),
        "downloads": family_downloads["industry"],
    })
    write_json(DOCS_DATA_DIR / "indices.json", {
        "asof_date": asof_date,
        "latest": idx_latest.to_dict(orient="records"),
        "daily": idx_daily.to_dict(orient="records"),
        "monthly": idx_monthly.to_dict(orient="records"),
        "downloads": family_downloads["indices"],
    })
    write_json(DOCS_DATA_DIR / "etf.json", {
        "asof_date": asof_date,
        "latest_summary": latest_row(etf_daily).to_dict(orient="records"),
        "latest_table": etf_latest_table.to_dict(orient="records"),
        "daily_history": etf_daily.to_dict(orient="records"),
        "monthly_history": etf_monthly.to_dict(orient="records"),
        "last_three_months": etf_monthly.sort_values("date", ascending=False).head(3).to_dict(orient="records"),
        "downloads": family_downloads["etf"],
        "note": "ETF ICC history starts when this archive pipeline began. Holdings-based ICC is used when online holdings can be matched; otherwise an online public valuation proxy is used.",
    })
    write_json(DOCS_DATA_DIR / "country.json", {
        "asof_date": asof_date,
        "latest_summary": latest_row(country_daily).to_dict(orient="records"),
        "latest_table": country_latest_table.to_dict(orient="records"),
        "daily_history": country_daily.to_dict(orient="records"),
        "monthly_history": country_monthly.to_dict(orient="records"),
        "last_three_months": country_monthly.sort_values("date", ascending=False).head(3).to_dict(orient="records"),
        "downloads": family_downloads["country"],
        "note": "Country ICC history starts when this archive pipeline began. Country ICC is based on country ETF proxies using online holdings or public valuation proxies.",
    })
    write_json(DOCS_DATA_DIR / "downloads_catalog.json", downloads_catalog)

    print(f"[build_docs_data] asof_date = {asof_date}")
    print("[build_docs_data] wrote overview/family data/downloads catalog")
