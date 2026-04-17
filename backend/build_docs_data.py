from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from pandas.errors import EmptyDataError

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DIR = REPO / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"
DOWNLOAD_DIR = DOCS_DATA_DIR / "downloads"
RAW_COPY_DIR = DOWNLOAD_DIR / "raw"
ZIP_DIR = DOWNLOAD_DIR / "zips"
DERIVED_DIR = DATA_DIR / "derived"
CONFIG_DIR = REPO / "config"

SNAPSHOT_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)
PE_REGEX = re.compile(r"P/E Ratio\s*(?:as of [A-Za-z]{3,9} \d{1,2},? \d{4}\s*)?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
SPY_HOLDINGS_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
INDEX_UNIVERSES = ["sp500", "sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"]


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


def safe_float(x):
    if pd.isna(x):
        return None
    try:
        value = float(x)
        if not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def normalize_symbol(text: str) -> str:
    text = str(text).strip().upper()
    text = text.replace(".", "-")
    text = re.sub(r"\s+", "", text)
    return text


def load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        raise EmptyDataError(f"No columns parsed: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    for col in ["mktcap", "ICC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).map(normalize_symbol)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "bm" in df.columns:
        df["bm"] = pd.to_numeric(df["bm"], errors="coerce")
    else:
        df["bm"] = np.nan
    if "sector" not in df.columns:
        df["sector"] = None
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


def find_all_snapshots(universe: str | None = None) -> list[Path]:
    items = []
    for p in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(p)
        if meta is None:
            continue
        if universe and meta["universe"] != universe:
            continue
        items.append(((meta["universe"], meta["yyyymmdd"], meta["rerun"], p.name), p))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def get_valid_snapshots(paths: Iterable[Path]) -> list[tuple[Path, pd.DataFrame]]:
    valid = []
    for path in paths:
        try:
            df = clean_df(load_snapshot(path))
            if df.empty:
                continue
            valid.append((path, df))
        except Exception as exc:
            print(f"[build_docs_data] skip {path}: {type(exc).__name__}: {exc}")
    return valid


def dedup_valid_snapshots(valid: list[tuple[Path, pd.DataFrame]]) -> list[tuple[Path, pd.DataFrame]]:
    latest = {}
    for path, df in valid:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        date_value = str(df["date"].iloc[0])
        key = (meta["universe"], date_value)
        latest[key] = (path, df)
    out = list(latest.values())
    out.sort(key=lambda x: (parse_snapshot_meta(x[0])["universe"], str(x[1]["date"].iloc[0])))
    return out


def month_end_last(df: pd.DataFrame, value_col: str, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or []
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["yyyymm"] = tmp["date"].dt.strftime("%Y-%m")
    sort_cols = group_cols + ["date"]
    tmp = tmp.sort_values(sort_cols)
    result = tmp.groupby(group_cols + ["yyyymm"], as_index=False).tail(1)
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    return result


def build_marketwide(base_valid: list[tuple[Path, pd.DataFrame]], sp500_valid: list[tuple[Path, pd.DataFrame]]) -> dict:
    daily_rows = []
    for path, df in base_valid:
        daily_rows.append({
            "date": str(df["date"].iloc[0]),
            "family": "all_market",
            "vw_icc": weighted_mean(df["ICC"], df["mktcap"]),
            "ew_icc": float(df["ICC"].mean()),
            "n_items": int(len(df)),
            "source_file": str(path.relative_to(REPO)),
        })
    for path, df in sp500_valid:
        daily_rows.append({
            "date": str(df["date"].iloc[0]),
            "family": "sp500",
            "vw_icc": weighted_mean(df["ICC"], df["mktcap"]),
            "ew_icc": float(df["ICC"].mean()),
            "n_items": int(len(df)),
            "source_file": str(path.relative_to(REPO)),
        })
    daily = pd.DataFrame(daily_rows).sort_values(["family", "date"]).reset_index(drop=True)
    monthly = month_end_last(daily, "vw_icc", ["family"])
    latest = daily.sort_values("date").groupby("family", as_index=False).tail(1).reset_index(drop=True)
    return {"latest": latest, "daily": daily, "monthly": monthly}


def build_value(base_valid: list[tuple[Path, pd.DataFrame]]) -> dict:
    rows = []
    latest_breakdown = []
    for path, raw in base_valid:
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
        bm30 = float(df["bm"].quantile(0.3))
        bm70 = float(df["bm"].quantile(0.7))
        df["size_bucket"] = np.where(df["mktcap"] <= size_median, "S", "B")
        df["bm_bucket"] = pd.cut(df["bm"], [-np.inf, bm30, bm70, np.inf], labels=["L", "M", "H"], include_lowest=True)
        df["bucket"] = df["size_bucket"] + "/" + df["bm_bucket"].astype(str)
        bucket_stats = (
            df.groupby("bucket", as_index=False)
            .apply(lambda g: pd.Series({
                "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
                "n_items": int(len(g))
            }))
            .reset_index(drop=True)
        )
        bucket_map = dict(zip(bucket_stats["bucket"], bucket_stats["vw_icc"]))
        value_icc = np.nanmean([bucket_map.get("S/H", np.nan), bucket_map.get("B/H", np.nan)])
        growth_icc = np.nanmean([bucket_map.get("S/L", np.nan), bucket_map.get("B/L", np.nan)])
        ivp = value_icc - growth_icc if np.isfinite(value_icc) and np.isfinite(growth_icc) else np.nan
        date_value = str(df["date"].iloc[0])
        rows.append({
            "date": date_value,
            "value_icc": value_icc,
            "growth_icc": growth_icc,
            "ivp": ivp,
            "n_items": int(len(df)),
            "source_file": str(path.relative_to(REPO)),
        })
        for _, row in bucket_stats.iterrows():
            latest_breakdown.append({
                "date": date_value,
                "bucket": row["bucket"],
                "vw_icc": row["vw_icc"],
                "n_items": row["n_items"],
            })
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    monthly = month_end_last(daily, "ivp")
    latest = daily.tail(1).reset_index(drop=True)
    latest_breakdown_df = pd.DataFrame(latest_breakdown)
    if not latest_breakdown_df.empty:
        latest_breakdown_df = latest_breakdown_df[latest_breakdown_df["date"] == latest.iloc[0]["date"]].reset_index(drop=True)
    return {"latest": latest, "daily": daily, "monthly": monthly, "latest_breakdown": latest_breakdown_df}


def build_industry(base_valid: list[tuple[Path, pd.DataFrame]]) -> dict:
    daily_rows = []
    latest_rows = []
    for path, df in base_valid:
        d2 = df[df["sector"].notna()].copy()
        if d2.empty:
            continue
        date_value = str(d2["date"].iloc[0])
        grp = (
            d2.groupby("sector", as_index=False)
            .apply(lambda g: pd.Series({
                "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
                "ew_icc": float(g["ICC"].mean()),
                "n_items": int(len(g)),
            }))
            .reset_index(drop=True)
        )
        grp["date"] = date_value
        grp["source_file"] = str(path.relative_to(REPO))
        latest_rows.append(grp)
        daily_rows.append({
            "date": date_value,
            "summary_icc": float(grp["vw_icc"].mean()),
            "n_groups": int(len(grp)),
            "source_file": str(path.relative_to(REPO)),
        })
    latest = pd.concat(latest_rows, ignore_index=True) if latest_rows else pd.DataFrame(columns=["date", "sector", "vw_icc", "ew_icc", "n_items", "source_file"])
    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    monthly = month_end_last(daily, "summary_icc")
    if not latest.empty:
        last_date = latest["date"].max()
        latest = latest[latest["date"] == last_date].sort_values("vw_icc", ascending=False).reset_index(drop=True)
    return {"latest": latest, "daily": daily, "monthly": monthly}


def build_indices(all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]]) -> dict:
    daily_rows = []
    latest_rows = []
    for universe in INDEX_UNIVERSES:
        valid = all_valid_by_universe.get(universe, [])
        if not valid:
            continue
        rows = []
        for path, df in valid:
            rows.append({
                "date": str(df["date"].iloc[0]),
                "universe": universe,
                "vw_icc": weighted_mean(df["ICC"], df["mktcap"]),
                "ew_icc": float(df["ICC"].mean()),
                "n_items": int(len(df)),
                "source_file": str(path.relative_to(REPO)),
            })
        one = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        daily_rows.append(one)
        latest_rows.append(one.tail(1))
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame(columns=["date", "universe", "vw_icc", "ew_icc", "n_items", "source_file"])
    monthly = month_end_last(daily, "vw_icc", ["universe"]) if not daily.empty else daily.copy()
    latest = pd.concat(latest_rows, ignore_index=True) if latest_rows else daily.head(0)
    return {"latest": latest.sort_values("universe").reset_index(drop=True), "daily": daily, "monthly": monthly}


def http_get_text(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def http_get_bytes(url: str) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.content


def parse_pe_from_text(text: str) -> float | None:
    cleaned = " ".join(BeautifulSoup(text, "html.parser").get_text(" ").split())
    m = PE_REGEX.search(cleaned)
    if not m:
        return None
    try:
        pe = float(m.group(1))
        if pe > 0:
            return pe
    except Exception:
        return None
    return None


def proxy_from_page_pe(url: str | None) -> tuple[float | None, str]:
    if not url:
        return None, "no_url"
    try:
        text = http_get_text(url)
        pe = parse_pe_from_text(text)
        if pe and pe > 0:
            return 1.0 / pe, "official_page_pe_ratio_proxy"
        return None, "page_pe_not_found"
    except Exception as exc:
        return None, f"page_error:{type(exc).__name__}"


def proxy_from_yfinance_pe(ticker: str) -> tuple[float | None, str]:
    try:
        info = yf.Ticker(ticker).get_info()
        pe = info.get("forwardPE") or info.get("trailingPE")
        pe = float(pe) if pe is not None else None
        if pe and np.isfinite(pe) and pe > 0:
            return 1.0 / pe, "yfinance_pe_ratio_proxy"
        return None, "yfinance_pe_not_found"
    except Exception as exc:
        return None, f"yfinance_info_error:{type(exc).__name__}"


def load_spy_holdings() -> pd.DataFrame | None:
    try:
        content = http_get_bytes(SPY_HOLDINGS_URL)
        xl = pd.ExcelFile(io.BytesIO(content))
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            cols = {str(c).strip().lower(): c for c in df.columns}
            if "ticker" in cols and ("weight" in cols or "weight (%)" in cols):
                ticker_col = cols["ticker"]
                weight_col = cols.get("weight") or cols.get("weight (%)")
                out = pd.DataFrame({
                    "symbol": df[ticker_col].astype(str).map(normalize_symbol),
                    "weight": pd.to_numeric(df[weight_col], errors="coerce"),
                })
                out = out[out["symbol"].ne("") & out["weight"].notna()].copy()
                if out["weight"].max() > 1.5:
                    out["weight"] = out["weight"] / 100.0
                out = out[out["weight"] > 0].copy()
                out["weight"] = out["weight"] / out["weight"].sum()
                return out[["symbol", "weight"]]
    except Exception as exc:
        print(f"[build_docs_data] SPY holdings fallback failed: {type(exc).__name__}: {exc}")
    return None


def load_top_holdings_from_yfinance(ticker: str) -> pd.DataFrame | None:
    try:
        fd = getattr(yf.Ticker(ticker), "funds_data", None)
        if fd is None:
            return None
        holdings = getattr(fd, "top_holdings", None)
        if holdings is None:
            return None
        h = holdings.copy() if isinstance(holdings, pd.DataFrame) else pd.DataFrame(holdings)
        if h.empty:
            return None
        cols = {str(c).strip().lower(): c for c in h.columns}
        symbol_col = cols.get("symbol") or cols.get("ticker")
        weight_col = cols.get("holdingpercent") or cols.get("holding_percent") or cols.get("weight")
        if symbol_col is None:
            for c in h.columns:
                if h[c].astype(str).str.contains(r"^[A-Za-z\.\-]{1,8}$", regex=True).mean() > 0.5:
                    symbol_col = c
                    break
        if weight_col is None:
            for c in h.columns[::-1]:
                vals = pd.to_numeric(h[c], errors="coerce")
                if vals.notna().sum() >= max(3, len(h) // 2):
                    weight_col = c
                    break
        if symbol_col is None or weight_col is None:
            return None
        out = pd.DataFrame({
            "symbol": h[symbol_col].astype(str).map(normalize_symbol),
            "weight": pd.to_numeric(h[weight_col], errors="coerce"),
        })
        out = out[out["symbol"].ne("") & out["weight"].notna()].copy()
        if out.empty:
            return None
        if out["weight"].max() > 1.5:
            out["weight"] = out["weight"] / 100.0
        out = out[out["weight"] > 0].copy()
        out["weight"] = out["weight"] / out["weight"].sum()
        return out[["symbol", "weight"]]
    except Exception as exc:
        print(f"[build_docs_data] yfinance holdings failed for {ticker}: {type(exc).__name__}: {exc}")
        return None


def holdings_based_proxy(ticker: str, base_usall: pd.DataFrame) -> tuple[float | None, float, int, int, str]:
    holdings = None
    source = ""
    if ticker == "SPY":
        holdings = load_spy_holdings()
        source = "ssga_holdings_download"
    if holdings is None:
        holdings = load_top_holdings_from_yfinance(ticker)
        source = "yfinance_top_holdings"
    if holdings is None or holdings.empty:
        return None, 0.0, 0, 0, "no_holdings"
    base = base_usall[["ticker", "ICC"]].copy()
    merged = holdings.merge(base, left_on="symbol", right_on="ticker", how="left")
    matched = merged[merged["ICC"].notna()].copy()
    coverage = float(matched["weight"].sum()) if not matched.empty else 0.0
    if matched.empty or coverage <= 0:
        return None, 0.0, int(len(holdings)), 0, source
    value = weighted_mean(matched["ICC"], matched["weight"])
    return value, coverage, int(len(holdings)), int(len(matched)), source


def read_config(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for c in columns:
        if c not in df.columns:
            df[c] = None
    return df[columns].copy()


def upsert_history(path: Path, df_new: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, df_new], ignore_index=True)
    else:
        combined = df_new.copy()
    combined = combined.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def build_online_proxy_family(
    config_df: pd.DataFrame,
    base_usall: pd.DataFrame,
    asof_date: str,
    kind: str,
) -> dict:
    latest_rows = []
    for _, cfg in config_df.iterrows():
        ticker = normalize_symbol(cfg.get("ticker", ""))
        if not ticker:
            continue
        label = str(cfg.get("label", ticker))
        category = str(cfg.get("category", "") or "")
        url = str(cfg.get("url", "") or "")
        allow_holdings = int(cfg.get("allow_holdings", 0) or 0)
        country = str(cfg.get("country", "") or "")

        value = None
        coverage = None
        n_items = None
        n_matched = None
        source = None
        status = None

        if allow_holdings:
            value, coverage, n_items, n_matched, source = holdings_based_proxy(ticker, base_usall)
            if value is not None and coverage >= 0.20 and n_matched >= 3:
                status = "ok_holdings"
            else:
                value = None

        if value is None:
            value, source = proxy_from_page_pe(url)
            if value is not None:
                status = "ok_page_proxy"

        if value is None:
            value, source = proxy_from_yfinance_pe(ticker)
            if value is not None:
                status = "ok_yfinance_proxy"

        if value is None:
            status = status or "unavailable"

        row = {
            "date": asof_date,
            "ticker": ticker,
            "label": label,
            "category": category,
            "vw_icc": value,
            "coverage_weight": coverage,
            "n_items": n_items,
            "n_matched": n_matched,
            "source": source,
            "status": status,
        }
        if kind == "country":
            row["country"] = country
        latest_rows.append(row)

    latest = pd.DataFrame(latest_rows)
    hist_path = DERIVED_DIR / f"{kind}_daily_history.csv"
    key_cols = ["date", "ticker"]
    history = upsert_history(hist_path, latest, key_cols) if not latest.empty else pd.DataFrame(columns=latest.columns)
    monthly = month_end_last(history, "vw_icc", ["ticker"]) if not history.empty else history.copy()

    summary_daily = pd.DataFrame(columns=["date", "summary_icc", "n_items"])
    summary_monthly = pd.DataFrame(columns=["date", "summary_icc", "n_items"])
    if not history.empty:
        summary_daily = (
            history.groupby("date", as_index=False)
            .agg(summary_icc=("vw_icc", "mean"), n_items=("ticker", "count"))
            .sort_values("date")
            .reset_index(drop=True)
        )
        summary_monthly = month_end_last(summary_daily, "summary_icc")

    return {
        "latest": latest,
        "daily": history,
        "monthly": monthly,
        "summary_daily": summary_daily,
        "summary_monthly": summary_monthly,
    }


def ensure_clean_download_dirs() -> None:
    for p in [DOCS_DATA_DIR, DOWNLOAD_DIR, RAW_COPY_DIR, ZIP_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def export_raw_snapshots(valid_all: list[tuple[Path, pd.DataFrame]]) -> tuple[dict, pd.DataFrame]:
    for child in RAW_COPY_DIR.glob("*"):
        if child.is_dir():
            shutil.rmtree(child)
    for child in ZIP_DIR.glob("*"):
        if child.is_dir():
            shutil.rmtree(child)
        elif child.is_file():
            child.unlink()

    file_rows = []
    tabs = {"usall": [], "sp500": [], "other_indices": []}

    def tab_name(universe: str) -> str:
        if universe == "usall":
            return "usall"
        if universe == "sp500":
            return "sp500"
        return "other_indices"

    for path, df in valid_all:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        yyyymm = meta["yyyymm"]
        date_value = str(df["date"].iloc[0])
        target_dir = RAW_COPY_DIR / yyyymm
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)
        row = {
            "date": date_value,
            "yyyymm": yyyymm,
            "universe": meta["universe"],
            "n_items": int(len(df)),
            "download_path": f"./data/downloads/raw/{yyyymm}/{path.name}",
        }
        file_rows.append(row)
        tabs[tab_name(meta["universe"])].append(row)

    def build_zip(files: list[dict], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for row in files:
                rel = row["download_path"].replace("./data/downloads/", "")
                src = DOWNLOAD_DIR / rel
                if src.exists():
                    zf.write(src, arcname=src.name)

    raw_tabs = {}
    for tab, rows in tabs.items():
        rows = sorted(rows, key=lambda x: (x["date"], x["universe"]), reverse=True)
        tab_years = []
        by_year: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_year[row["yyyymm"][:4]].append(row)
        for year in sorted(by_year.keys(), reverse=True):
            year_rows = by_year[year]
            by_month: dict[str, list[dict]] = defaultdict(list)
            for row in year_rows:
                by_month[row["yyyymm"]].append(row)
            year_zip_rel = f"./data/downloads/zips/raw/{tab}/{year}.zip"
            build_zip(year_rows, ZIP_DIR / "raw" / tab / f"{year}.zip")
            months = []
            for yyyymm in sorted(by_month.keys(), reverse=True):
                month_rows = sorted(by_month[yyyymm], key=lambda x: (x["date"], x["universe"]), reverse=True)
                month_zip_rel = f"./data/downloads/zips/raw/{tab}/{yyyymm}.zip"
                build_zip(month_rows, ZIP_DIR / "raw" / tab / f"{yyyymm}.zip")
                months.append({
                    "month": yyyymm[4:6],
                    "yyyymm": yyyymm,
                    "download_all_zip": month_zip_rel,
                    "files": month_rows,
                })
            tab_years.append({
                "year": year,
                "download_all_zip": year_zip_rel,
                "months": months,
            })
        raw_tabs[tab] = tab_years

    all_files = pd.DataFrame(file_rows).sort_values(["date", "universe"]).reset_index(drop=True)
    return raw_tabs, all_files


def save_family_downloads(name: str, daily: pd.DataFrame, monthly: pd.DataFrame, latest: pd.DataFrame | None) -> list[dict]:
    items = []
    if daily is not None and not daily.empty:
        p = DOWNLOAD_DIR / f"{name}_daily_history.csv"
        daily.to_csv(p, index=False)
        items.append({"label": "Daily history CSV", "path": f"./data/downloads/{p.name}"})
    if monthly is not None and not monthly.empty:
        p = DOWNLOAD_DIR / f"{name}_monthly_history.csv"
        monthly.to_csv(p, index=False)
        items.append({"label": "Monthly history CSV", "path": f"./data/downloads/{p.name}"})
    if latest is not None and not latest.empty:
        p = DOWNLOAD_DIR / f"{name}_latest.csv"
        latest.to_csv(p, index=False)
        items.append({"label": "Latest breakdown CSV", "path": f"./data/downloads/{p.name}"})
    return items


def overview_last_three(monthly_df: pd.DataFrame, family_col: str | None, value_col: str) -> dict[str, list[dict]]:
    if monthly_df.empty:
        return {}
    if family_col:
        groups = monthly_df.groupby(family_col)
    else:
        groups = [("_single", monthly_df)]
    out = {}
    for family, grp in groups:
        grp2 = grp.sort_values("date", ascending=False).head(3).reset_index(drop=True)
        out[str(family)] = grp2.to_dict(orient="records")
    return out


def build_overview(asof_date: str, marketwide: dict, value: dict, industry: dict, etf: dict, country: dict, indices: dict) -> dict:
    market_months = overview_last_three(marketwide["monthly"], "family", "vw_icc")
    value_months = overview_last_three(value["monthly"], None, "ivp")
    industry_months = overview_last_three(industry["monthly"], None, "summary_icc")
    etf_months = overview_last_three(etf["summary_monthly"], None, "summary_icc")
    country_months = overview_last_three(country["summary_monthly"], None, "summary_icc")
    index_months = overview_last_three(indices["monthly"], None, "vw_icc")

    def row_from_last_three(label: str, latest_daily: float | None, arr: list[dict], value_key: str) -> dict:
        row = {"family": label, "latest_daily": latest_daily}
        for i in range(3):
            if i < len(arr):
                row[f"m{i+1}_date"] = arr[i]["date"]
                row[f"m{i+1}_value"] = arr[i].get(value_key)
            else:
                row[f"m{i+1}_date"] = None
                row[f"m{i+1}_value"] = None
        return row

    rows = []
    all_latest = marketwide["latest"]
    all_val = safe_float(all_latest.loc[all_latest["family"] == "all_market", "vw_icc"].iloc[0]) if not all_latest.empty else None
    sp500_val = safe_float(all_latest.loc[all_latest["family"] == "sp500", "vw_icc"].iloc[0]) if not all_latest.empty else None
    value_latest = safe_float(value["latest"]["ivp"].iloc[0]) if not value["latest"].empty else None
    industry_latest = safe_float(industry["daily"]["summary_icc"].iloc[-1]) if not industry["daily"].empty else None
    etf_latest = safe_float(etf["summary_daily"]["summary_icc"].iloc[-1]) if not etf["summary_daily"].empty else None
    country_latest = safe_float(country["summary_daily"]["summary_icc"].iloc[-1]) if not country["summary_daily"].empty else None
    indices_latest = safe_float(indices["daily"]["vw_icc"].mean()) if not indices["daily"].empty else None

    rows.append(row_from_last_three("Marketwide (All U.S.)", all_val, market_months.get("all_market", []), "vw_icc"))
    rows.append(row_from_last_three("Marketwide (S&P 500)", sp500_val, market_months.get("sp500", []), "vw_icc"))
    rows.append(row_from_last_three("Value Premium (IVP)", value_latest, value_months.get("_single", []), "ivp"))
    rows.append(row_from_last_three("Industry Wide", industry_latest, industry_months.get("_single", []), "summary_icc"))
    rows.append(row_from_last_three("ETF ICC", etf_latest, etf_months.get("_single", []), "summary_icc"))
    rows.append(row_from_last_three("Country ICC", country_latest, country_months.get("_single", []), "summary_icc"))
    rows.append(row_from_last_three("Indices", indices_latest, index_months.get("_single", []), "vw_icc"))

    return {"title": "Fama-French style live ICC data library", "asof_date": asof_date, "rows": rows}


def family_payload(asof_date: str, latest: pd.DataFrame, daily: pd.DataFrame, monthly: pd.DataFrame, downloads: list[dict], summary_col: str | None = None) -> dict:
    payload = {
        "asof_date": asof_date,
        "latest": latest.to_dict(orient="records") if latest is not None else [],
        "daily": daily.to_dict(orient="records") if daily is not None else [],
        "monthly": monthly.to_dict(orient="records") if monthly is not None else [],
        "downloads": downloads,
    }
    if summary_col and daily is not None and not daily.empty and summary_col in daily.columns:
        payload["latest_summary"] = safe_float(daily[summary_col].iloc[-1])
    return payload


def write_page_jsons(asof_date: str, marketwide: dict, value: dict, industry: dict, etf: dict, country: dict, indices: dict, downloads_catalog: dict) -> None:
    overview = build_overview(asof_date, marketwide, value, industry, etf, country, indices)
    write_json(DOCS_DATA_DIR / "overview.json", overview)
    write_json(DOCS_DATA_DIR / "marketwide.json", family_payload(asof_date, marketwide["latest"], marketwide["daily"], marketwide["monthly"], downloads_catalog["families"]["marketwide"]))
    write_json(DOCS_DATA_DIR / "value.json", family_payload(asof_date, value["latest_breakdown"], value["daily"], value["monthly"], downloads_catalog["families"]["value"]))
    write_json(DOCS_DATA_DIR / "industry.json", family_payload(asof_date, industry["latest"], industry["daily"], industry["monthly"], downloads_catalog["families"]["industry"], summary_col="summary_icc"))
    write_json(DOCS_DATA_DIR / "etf.json", {
        "asof_date": asof_date,
        "summary_latest": etf["summary_daily"].tail(1).to_dict(orient="records"),
        "summary_monthly": etf["summary_monthly"].to_dict(orient="records"),
        "latest": etf["latest"].to_dict(orient="records"),
        "daily": etf["daily"].to_dict(orient="records"),
        "monthly": etf["monthly"].to_dict(orient="records"),
        "downloads": downloads_catalog["families"]["etf"],
    })
    write_json(DOCS_DATA_DIR / "country.json", {
        "asof_date": asof_date,
        "summary_latest": country["summary_daily"].tail(1).to_dict(orient="records"),
        "summary_monthly": country["summary_monthly"].to_dict(orient="records"),
        "latest": country["latest"].to_dict(orient="records"),
        "daily": country["daily"].to_dict(orient="records"),
        "monthly": country["monthly"].to_dict(orient="records"),
        "downloads": downloads_catalog["families"]["country"],
    })
    write_json(DOCS_DATA_DIR / "indices.json", family_payload(asof_date, indices["latest"], indices["daily"], indices["monthly"], downloads_catalog["families"]["indices"]))
    write_json(DOCS_DATA_DIR / "downloads_catalog.json", downloads_catalog)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    ensure_clean_download_dirs()
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    all_valid = dedup_valid_snapshots(get_valid_snapshots(find_all_snapshots(None)))
    if not all_valid:
        raise RuntimeError("No valid raw snapshots found under data/YYYYMM/")

    by_universe: dict[str, list[tuple[Path, pd.DataFrame]]] = defaultdict(list)
    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        by_universe[meta["universe"]].append((path, df))

    base_valid = by_universe.get(args.universe, [])
    sp500_valid = by_universe.get("sp500", [])
    if not base_valid:
        raise RuntimeError(f"No valid snapshots found for {args.universe}")

    latest_usall_path, latest_usall = base_valid[-1]
    asof_date = str(latest_usall["date"].iloc[0])

    marketwide = build_marketwide(base_valid, sp500_valid)
    value = build_value(base_valid)
    industry = build_industry(base_valid)
    indices = build_indices(by_universe)

    etf_cfg = read_config(CONFIG_DIR / "etfs.csv", ["ticker", "label", "category", "url", "allow_holdings"])
    country_cfg = read_config(CONFIG_DIR / "country_etfs.csv", ["ticker", "label", "country", "url", "allow_holdings"])
    etf = build_online_proxy_family(etf_cfg, latest_usall, asof_date, "etf")
    country = build_online_proxy_family(country_cfg, latest_usall, asof_date, "country")

    raw_tabs, all_snapshot_files = export_raw_snapshots(all_valid)

    families = {
        "marketwide": save_family_downloads("marketwide", marketwide["daily"], marketwide["monthly"], marketwide["latest"]),
        "value": save_family_downloads("value", value["daily"], value["monthly"], value["latest_breakdown"]),
        "industry": save_family_downloads("industry", industry["daily"], industry["monthly"], industry["latest"]),
        "etf": save_family_downloads("etf", etf["daily"], etf["monthly"], etf["latest"]),
        "country": save_family_downloads("country", country["daily"], country["monthly"], country["latest"]),
        "indices": save_family_downloads("indices", indices["daily"], indices["monthly"], indices["latest"]),
    }

    downloads_catalog = {
        "asof_date": asof_date,
        "families": families,
        "raw_tabs": raw_tabs,
    }

    write_page_jsons(asof_date, marketwide, value, industry, etf, country, indices, downloads_catalog)
    all_snapshot_files.to_csv(DOWNLOAD_DIR / "all_snapshot_files.csv", index=False)

    print(f"[build_docs_data] asof_date = {asof_date}")
    print(f"[build_docs_data] latest usall file = {latest_usall_path}")
    print("[build_docs_data] docs/data and downloads updated")


if __name__ == "__main__":
    main()
