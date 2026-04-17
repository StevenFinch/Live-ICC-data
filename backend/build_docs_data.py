from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
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
CONFIG_DIR = REPO / "config"
LIB_DIR = DATA_DIR / "library_history"

SNAPSHOT_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

INDEX_UNIVERSES = [
    "sp500",
    "sp100",
    "dow30",
    "ndx100",
    "sp400",
    "sp600",
    "rut1000",
]

FAMILY_LABELS = {
    "marketwide": "Marketwide",
    "value": "Value premium",
    "industry": "Industry",
    "etf": "ETF",
    "country": "Country",
    "indices": "Indices",
}

PAGE_PROXY_PATTERNS = [
    re.compile(r"Price/Earnings Ratio[^\d]{0,40}(\d+(?:\.\d+)?)", re.I),
    re.compile(r"P/E Ratio[^\d]{0,40}(\d+(?:\.\d+)?)", re.I),
    re.compile(r'"peRatio"\s*[:=]\s*"?(\d+(?:\.\d+)?)', re.I),
    re.compile(r'"pe_ratio"\s*[:=]\s*"?(\d+(?:\.\d+)?)', re.I),
]


@dataclass
class SnapshotMeta:
    universe: str
    year: str
    mmdd: str
    yyyymm: str
    yyyymmdd: str
    rerun: int


@dataclass
class FamilyDownloadSpec:
    family_key: str
    title: str
    latest_path: str
    daily_history_path: str
    monthly_history_path: str
    latest_table: list[dict]
    daily_history: list[dict]
    monthly_history: list[dict]


# -----------------------------
# Generic helpers
# -----------------------------


def requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def json_safe(obj):
    """Convert pandas/numpy values into JSON-safe Python values."""
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
        return None if not math.isfinite(x) else x
    return obj



def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)



def safe_float(x) -> float | None:
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None



def safe_int(x) -> int | None:
    try:
        return int(x)
    except Exception:
        return None



def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = (
        values.notna()
        & weights.notna()
        & np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))



def parse_snapshot_meta(path: Path) -> SnapshotMeta | None:
    m = SNAPSHOT_RE.match(path.name)
    if not m:
        return None
    year = m.group("year")
    mmdd = m.group("mmdd")
    rerun = int(m.group("rerun") or 0)
    return SnapshotMeta(
        universe=m.group("universe"),
        year=year,
        mmdd=mmdd,
        yyyymm=f"{year}{mmdd[:2]}",
        yyyymmdd=f"{year}{mmdd}",
        rerun=rerun,
    )



def load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    if path.stat().st_size == 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        raise EmptyDataError(f"No columns parsed from file: {path}")
    df.columns = [str(c).strip() for c in df.columns]

    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
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

    return df



def clean_snapshot(df: pd.DataFrame) -> pd.DataFrame:
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
    rows: list[tuple[tuple[str, int, str], Path]] = []
    for p in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(p)
        if meta is None:
            continue
        if universe is not None and meta.universe != universe:
            continue
        rows.append(((meta.yyyymmdd, meta.rerun, p.name), p))
    rows.sort(key=lambda x: x[0])
    return [p for _, p in rows]



def get_valid_snapshots(universe: str | None = None) -> list[tuple[Path, pd.DataFrame, SnapshotMeta]]:
    valid = []
    for path in find_all_snapshots(universe):
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        try:
            df = clean_snapshot(load_snapshot(path))
            if df.empty:
                continue
            valid.append((path, df, meta))
        except Exception as e:
            print(f"[build_docs_data] skip invalid snapshot {path}: {type(e).__name__}: {e}")

    # Deduplicate same universe/date and keep the latest rerun.
    dedup: dict[tuple[str, str], tuple[Path, pd.DataFrame, SnapshotMeta]] = {}
    for path, df, meta in valid:
        date_value = str(df["date"].dropna().iloc[0])
        dedup[(meta.universe, date_value)] = (path, df, meta)

    out = list(dedup.values())
    out.sort(key=lambda x: (x[2].universe, str(x[1]["date"].dropna().iloc[0])))
    return out



def latest_per_month(df: pd.DataFrame, group_cols: list[str], value_cols: list[str], n_months: int = 3) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["yyyymm"] = tmp["date"].dt.strftime("%Y-%m")
    tmp = tmp.sort_values(group_cols + ["date"])
    tmp = tmp.groupby(group_cols + ["yyyymm"], as_index=False).tail(1)
    tmp = tmp.sort_values(group_cols + ["date"], ascending=[True] * len(group_cols) + [False])
    tmp = tmp.groupby(group_cols, as_index=False).head(n_months)
    tmp["date"] = tmp["date"].dt.strftime("%Y-%m-%d")
    keep = group_cols + ["date", "yyyymm"] + value_cols
    return tmp[keep].reset_index(drop=True)



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def make_zip(zip_path: Path, file_paths: Iterable[Path], arcname_fn=None) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in file_paths:
            if not p.exists() or p.stat().st_size == 0:
                continue
            arcname = arcname_fn(p) if arcname_fn else p.name
            zf.write(p, arcname)


# -----------------------------
# Family builders from snapshots
# -----------------------------


def build_marketwide(base_valid: list[tuple[Path, pd.DataFrame, SnapshotMeta]], sp500_valid: list[tuple[Path, pd.DataFrame, SnapshotMeta]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = []
    for path, df, _ in base_valid:
        all_rows.append(
            {
                "date": str(df["date"].iloc[0]),
                "family": "all_market",
                "value": weighted_mean(df["ICC"], df["mktcap"]),
                "n_items": int(len(df)),
                "method": "ICC calculation",
                "source_file": str(path.relative_to(REPO)),
            }
        )
    for path, df, _ in sp500_valid:
        all_rows.append(
            {
                "date": str(df["date"].iloc[0]),
                "family": "sp500",
                "value": weighted_mean(df["ICC"], df["mktcap"]),
                "n_items": int(len(df)),
                "method": "ICC calculation",
                "source_file": str(path.relative_to(REPO)),
            }
        )
    daily = pd.DataFrame(all_rows).sort_values(["family", "date"]).reset_index(drop=True)
    monthly = latest_per_month(daily, ["family"], ["value", "n_items", "method"])
    latest = daily.sort_values(["family", "date"]).groupby("family", as_index=False).tail(1)
    return latest.reset_index(drop=True), daily, monthly



def build_value_family(base_valid: list[tuple[Path, pd.DataFrame, SnapshotMeta]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for path, raw_df, _ in base_valid:
        df = raw_df.copy()
        df = df[
            df["bm"].notna()
            & np.isfinite(df["bm"])
            & (df["bm"] > 0)
            & df["ICC"].notna()
            & np.isfinite(df["ICC"])
            & df["mktcap"].notna()
            & np.isfinite(df["mktcap"])
            & (df["mktcap"] > 0)
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
        df["bm_bucket"] = pd.cut(
            df["bm"],
            bins=[-np.inf, bm30, bm70, np.inf],
            labels=["L", "M", "H"],
            include_lowest=True,
            right=True,
        ).astype("string")
        df["bucket"] = df["size_bucket"] + "/" + df["bm_bucket"].astype(str)

        bucket_icc = df.groupby("bucket", as_index=False).apply(
            lambda g: pd.Series({"vw_icc": weighted_mean(g["ICC"], g["mktcap"])})
        )
        d = dict(zip(bucket_icc["bucket"], bucket_icc["vw_icc"]))
        growth = np.nan if pd.isna(d.get("S/L")) or pd.isna(d.get("B/L")) else float((d.get("S/L") + d.get("B/L")) / 2.0)
        value = np.nan if pd.isna(d.get("S/H")) or pd.isna(d.get("B/H")) else float((d.get("S/H") + d.get("B/H")) / 2.0)
        ivp = np.nan if pd.isna(value) or pd.isna(growth) else float(value - growth)
        rows.extend(
            [
                {
                    "date": str(df["date"].iloc[0]),
                    "family": "value_icc",
                    "value": value,
                    "n_items": int(len(df)),
                    "method": "ICC calculation",
                    "source_file": str(path.relative_to(REPO)),
                },
                {
                    "date": str(df["date"].iloc[0]),
                    "family": "growth_icc",
                    "value": growth,
                    "n_items": int(len(df)),
                    "method": "ICC calculation",
                    "source_file": str(path.relative_to(REPO)),
                },
                {
                    "date": str(df["date"].iloc[0]),
                    "family": "ivp_bm",
                    "value": ivp,
                    "n_items": int(len(df)),
                    "method": "ICC calculation",
                    "source_file": str(path.relative_to(REPO)),
                },
            ]
        )

    daily = pd.DataFrame(rows).sort_values(["family", "date"]).reset_index(drop=True)
    monthly = latest_per_month(daily, ["family"], ["value", "n_items", "method"])
    latest = daily.sort_values(["family", "date"]).groupby("family", as_index=False).tail(1)
    return latest.reset_index(drop=True), daily, monthly



def build_industry_family(base_valid: list[tuple[Path, pd.DataFrame, SnapshotMeta]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for path, df, _ in base_valid:
        g = df[df["sector"].notna()].copy()
        if g.empty:
            continue
        grp = (
            g.groupby("sector", as_index=False)
            .apply(lambda x: pd.Series({
                "value": weighted_mean(x["ICC"], x["mktcap"]),
                "n_items": int(len(x)),
            }))
            .reset_index(drop=True)
        )
        grp["date"] = str(g["date"].iloc[0])
        grp["family"] = grp["sector"].astype(str)
        grp["method"] = "ICC calculation"
        grp["source_file"] = str(path.relative_to(REPO))
        rows.append(grp[["date", "family", "value", "n_items", "method", "source_file"]])

    daily = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "family", "value", "n_items", "method", "source_file"])
    daily = daily.sort_values(["family", "date"]).reset_index(drop=True)
    monthly = latest_per_month(daily, ["family"], ["value", "n_items", "method"])
    latest = daily.sort_values(["family", "date"]).groupby("family", as_index=False).tail(1)
    return latest.reset_index(drop=True), daily, monthly



def build_indices_family(all_valid: list[tuple[Path, pd.DataFrame, SnapshotMeta]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for path, df, meta in all_valid:
        if meta.universe not in INDEX_UNIVERSES:
            continue
        rows.append(
            {
                "date": str(df["date"].iloc[0]),
                "family": meta.universe,
                "value": weighted_mean(df["ICC"], df["mktcap"]),
                "n_items": int(len(df)),
                "method": "ICC calculation",
                "source_file": str(path.relative_to(REPO)),
            }
        )
    daily = pd.DataFrame(rows).sort_values(["family", "date"]).reset_index(drop=True)
    monthly = latest_per_month(daily, ["family"], ["value", "n_items", "method"])
    latest = daily.sort_values(["family", "date"]).groupby("family", as_index=False).tail(1)
    return latest.reset_index(drop=True), daily, monthly


# -----------------------------
# ETF / Country online estimates
# -----------------------------


def read_cfg(name: str, cols: list[str]) -> pd.DataFrame:
    path = CONFIG_DIR / name
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].copy()



def normalize_symbol(s: str) -> str:
    return str(s).strip().upper().replace(".", "-")



def fetch_page_pe(url: str, session: requests.Session) -> tuple[float | None, str | None]:
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        text = r.text
    except Exception:
        return None, None

    for pat in PAGE_PROXY_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                pe = float(m.group(1))
                if math.isfinite(pe) and pe > 0:
                    return pe, "official_page_pe_ratio_proxy"
            except Exception:
                pass
    return None, None



def fetch_yfinance_pe(ticker: str) -> tuple[float | None, str | None]:
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        return None, None
    pe = info.get("forwardPE") or info.get("trailingPE")
    try:
        pe = float(pe)
        if math.isfinite(pe) and pe > 0:
            return pe, "yfinance_pe_ratio_proxy"
    except Exception:
        return None, None
    return None, None



def fetch_top_holdings(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    fd = getattr(t, "funds_data", None)
    if fd is None:
        return pd.DataFrame(columns=["symbol", "weight"])
    holdings = getattr(fd, "top_holdings", None)
    if holdings is None:
        return pd.DataFrame(columns=["symbol", "weight"])
    h = holdings.copy() if isinstance(holdings, pd.DataFrame) else pd.DataFrame(holdings)
    if h.empty:
        return pd.DataFrame(columns=["symbol", "weight"])

    cols = {str(c).lower(): c for c in h.columns}
    sym_col = cols.get("symbol") or cols.get("ticker") or cols.get("holding") or h.columns[0]
    weight_col = None
    for cand in ["holdingpercent", "holding_percent", "weight", "percent", "pct"]:
        if cand in cols:
            weight_col = cols[cand]
            break
    if weight_col is None:
        for c in h.columns[::-1]:
            vals = pd.to_numeric(h[c], errors="coerce")
            if vals.notna().sum() >= max(1, len(h) // 2):
                weight_col = c
                break
    if weight_col is None:
        return pd.DataFrame(columns=["symbol", "weight"])

    out = pd.DataFrame({
        "symbol": h[sym_col].astype(str).map(normalize_symbol),
        "raw_weight": pd.to_numeric(h[weight_col], errors="coerce"),
    })
    out = out[out["symbol"].notna() & out["raw_weight"].notna()].copy()
    if out.empty:
        return pd.DataFrame(columns=["symbol", "weight"])
    out["weight"] = out["raw_weight"] / 100.0 if out["raw_weight"].max() > 1.5 else out["raw_weight"]
    out = out[out["weight"] > 0].copy()
    if out.empty:
        return pd.DataFrame(columns=["symbol", "weight"])
    out["weight"] = out["weight"] / out["weight"].sum()
    return out[["symbol", "weight"]]



def holdings_based_proxy(ticker: str, base_usall: pd.DataFrame) -> tuple[float | None, dict]:
    h = fetch_top_holdings(ticker)
    if h.empty:
        return None, {"coverage_weight": 0.0, "n_items": 0, "n_matched": 0, "source": None, "status": "unavailable"}

    base = base_usall[["ticker", "ICC"]].copy()
    merged = h.merge(base, left_on="symbol", right_on="ticker", how="left")
    matched = merged["ICC"].notna()
    coverage = float(merged.loc[matched, "weight"].sum()) if len(merged) else 0.0
    n_matched = int(matched.sum())
    n_items = int(len(merged))

    if coverage >= 0.20 and n_matched >= 3:
        icc = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"])
        return icc, {
            "coverage_weight": coverage,
            "n_items": n_items,
            "n_matched": n_matched,
            "source": "yfinance_top_holdings",
            "status": "ICC calculation",
        }

    return None, {
        "coverage_weight": coverage,
        "n_items": n_items,
        "n_matched": n_matched,
        "source": "yfinance_top_holdings",
        "status": "unavailable",
    }



def estimate_online_row(row: pd.Series, base_usall: pd.DataFrame, session: requests.Session) -> dict:
    ticker = normalize_symbol(row["ticker"])
    label = str(row.get("label", ticker))
    category = str(row.get("category", ""))
    url = str(row.get("url", "") or "").strip()
    allow_holdings = str(row.get("allow_holdings", "1")).strip() in {"1", "true", "True", "yes", "YES"}

    value = None
    detail = {
        "coverage_weight": None,
        "n_items": None,
        "n_matched": None,
        "source": None,
        "status": "unavailable",
    }

    if allow_holdings:
        value, detail = holdings_based_proxy(ticker, base_usall)

    if value is None and url:
        pe, source = fetch_page_pe(url, session)
        if pe is not None and pe > 0:
            value = 1.0 / pe
            detail.update({"source": source, "status": "P/E estimate"})

    if value is None:
        pe, source = fetch_yfinance_pe(ticker)
        if pe is not None and pe > 0:
            value = 1.0 / pe
            detail.update({"source": source, "status": "P/E estimate"})

    return {
        "date": str(base_usall["date"].iloc[0]),
        "family": ticker,
        "ticker": ticker,
        "label": label,
        "category": category,
        "value": safe_float(value),
        "coverage_weight": safe_float(detail["coverage_weight"]),
        "n_items": safe_int(detail["n_items"]),
        "n_matched": safe_int(detail["n_matched"]),
        "source": detail["source"],
        "status": detail["status"],
        "method": detail["status"],
    }



def append_history_csv(path: Path, latest_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        hist = pd.read_csv(path)
    else:
        hist = pd.DataFrame(columns=latest_df.columns)
    merged = pd.concat([hist, latest_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=key_cols, keep="last").sort_values(key_cols).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged



def build_online_family(cfg_name: str, base_usall: pd.DataFrame, hist_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = read_cfg(cfg_name, ["ticker", "label", "category", "url", "allow_holdings"])
    session = requests_session()
    rows = [estimate_online_row(row, base_usall, session) for _, row in cfg.iterrows()]
    latest = pd.DataFrame(rows)
    hist_path = LIB_DIR / hist_name
    daily = append_history_csv(hist_path, latest, ["date", "ticker"])
    monthly = latest_per_month(daily.rename(columns={"ticker": "family"}), ["family"], ["value", "method"])
    monthly = monthly.rename(columns={"family": "ticker"})
    return latest, daily, monthly


# -----------------------------
# Downloads
# -----------------------------


def family_download_tree(family_key: str, latest: pd.DataFrame, daily: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    family_dir = DOWNLOAD_DIR / family_key
    ensure_dir(family_dir)

    latest_csv = family_dir / "latest.csv"
    daily_csv = family_dir / "daily_history.csv"
    monthly_csv = family_dir / "monthly_history.csv"
    write_csv(latest_csv, latest)
    write_csv(daily_csv, daily)
    write_csv(monthly_csv, monthly)

    # Build per-day CSV files.
    day_root = family_dir / "daily"
    if day_root.exists():
        shutil.rmtree(day_root)
    ensure_dir(day_root)

    if not daily.empty:
        for date_value, g in daily.groupby("date"):
            dt = pd.to_datetime(date_value)
            yyyy = dt.strftime("%Y")
            yyyymm = dt.strftime("%Y%m")
            out = day_root / yyyy / f"{date_value}.csv"
            write_csv(out, g.sort_values(g.columns.tolist()).reset_index(drop=True))

    years = []
    if day_root.exists():
        for year_dir in sorted([p for p in day_root.iterdir() if p.is_dir()], reverse=True):
            month_map: dict[str, list[Path]] = {}
            for f in sorted(year_dir.glob("*.csv")):
                dt = pd.to_datetime(f.stem)
                yyyymm = dt.strftime("%Y%m")
                month_map.setdefault(yyyymm, []).append(f)

            year_zip = family_dir / f"{family_key}_{year_dir.name}_all.zip"
            all_year_files = [f for arr in month_map.values() for f in arr]
            make_zip(year_zip, all_year_files, arcname_fn=lambda p: f"{year_dir.name}/{p.name}")

            months = []
            for yyyymm, files in sorted(month_map.items(), reverse=True):
                month_zip = family_dir / f"{family_key}_{yyyymm}_all.zip"
                make_zip(month_zip, files, arcname_fn=lambda p: f"{yyyymm}/{p.name}")
                months.append({
                    "month": yyyymm[4:6],
                    "yyyymm": yyyymm,
                    "download_all": f"./data/downloads/{family_key}/{month_zip.name}",
                    "files": [
                        {
                            "date": f.stem,
                            "download_path": f"./data/downloads/{family_key}/daily/{year_dir.name}/{f.name}",
                        }
                        for f in sorted(files, reverse=True)
                    ],
                })

            years.append({
                "year": year_dir.name,
                "download_all": f"./data/downloads/{family_key}/{year_zip.name}",
                "months": months,
            })

    return {
        "family_key": family_key,
        "title": FAMILY_LABELS[family_key],
        "latest_csv": f"./data/downloads/{family_key}/latest.csv",
        "daily_history_csv": f"./data/downloads/{family_key}/daily_history.csv",
        "monthly_history_csv": f"./data/downloads/{family_key}/monthly_history.csv",
        "years": years,
    }



def build_raw_downloads(all_valid: list[tuple[Path, pd.DataFrame, SnapshotMeta]]) -> dict:
    raw_root = DOWNLOAD_DIR / "raw"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    ensure_dir(raw_root)

    categories = {
        "usall": [],
        "sp500": [],
        "other_indices": [],
    }

    for path, _df, meta in all_valid:
        if meta.universe == "usall":
            cat = "usall"
        elif meta.universe == "sp500":
            cat = "sp500"
        elif meta.universe in INDEX_UNIVERSES:
            cat = "other_indices"
        else:
            continue

        target_dir = raw_root / cat / meta.year / meta.yyyymm
        ensure_dir(target_dir)
        target = target_dir / path.name
        shutil.copy2(path, target)
        categories[cat].append((target, meta))

    out = {}
    for cat, items in categories.items():
        year_map: dict[str, dict[str, list[Path]]] = {}
        for p, meta in items:
            year_map.setdefault(meta.year, {}).setdefault(meta.yyyymm, []).append(p)

        years = []
        for year, month_map in sorted(year_map.items(), reverse=True):
            all_year_files = [f for files in month_map.values() for f in files]
            year_zip = raw_root / cat / f"{cat}_{year}_all.zip"
            make_zip(year_zip, all_year_files, arcname_fn=lambda p: f"{year}/{p.parent.name}/{p.name}")
            months = []
            for yyyymm, files in sorted(month_map.items(), reverse=True):
                month_zip = raw_root / cat / year / f"{cat}_{yyyymm}_all.zip"
                make_zip(month_zip, files, arcname_fn=lambda p: f"{yyyymm}/{p.name}")
                day_rows = []
                for f in sorted(files, reverse=True):
                    meta = parse_snapshot_meta(f)
                    if meta is None:
                        continue
                    date_value = f"{meta.year}-{meta.mmdd[:2]}-{meta.mmdd[2:]}"
                    day_rows.append({
                        "date": date_value,
                        "universe": meta.universe,
                        "download_path": f"./data/downloads/raw/{cat}/{year}/{yyyymm}/{f.name}",
                    })
                months.append({
                    "month": yyyymm[4:6],
                    "yyyymm": yyyymm,
                    "download_all": f"./data/downloads/raw/{cat}/{year}/{month_zip.name}",
                    "files": day_rows,
                })
            years.append({
                "year": year,
                "download_all": f"./data/downloads/raw/{cat}/{year_zip.name}",
                "months": months,
            })
        out[cat] = years
    return out


# -----------------------------
# Page payloads
# -----------------------------


def overview_family_rows(latest: pd.DataFrame, monthly: pd.DataFrame, family_key: str) -> list[dict]:
    latest_map = latest.set_index("family")["value"].to_dict() if not latest.empty else {}
    method_map = latest.set_index("family")["method"].to_dict() if not latest.empty else {}
    month_map: dict[str, list[dict]] = {}
    for fam, grp in monthly.groupby("family"):
        month_map[fam] = grp.sort_values("date", ascending=False).to_dict(orient="records")

    rows = []
    for fam in sorted(set(list(latest_map.keys()) + list(month_map.keys()))):
        months = month_map.get(fam, [])[:3]
        row = {
            "family_key": family_key,
            "family": fam,
            "latest_daily": safe_float(latest_map.get(fam)),
            "method": method_map.get(fam),
            "m1_date": months[0]["date"] if len(months) > 0 else None,
            "m1_value": safe_float(months[0]["value"]) if len(months) > 0 else None,
            "m2_date": months[1]["date"] if len(months) > 1 else None,
            "m2_value": safe_float(months[1]["value"]) if len(months) > 1 else None,
            "m3_date": months[2]["date"] if len(months) > 2 else None,
            "m3_value": safe_float(months[2]["value"]) if len(months) > 2 else None,
        }
        rows.append(row)
    return rows



def build_family_page_payload(title: str, latest: pd.DataFrame, daily: pd.DataFrame, monthly: pd.DataFrame, downloads: dict, split_groups: bool = False) -> dict:
    payload = {
        "title": title,
        "latest": latest.to_dict(orient="records"),
        "daily": daily.to_dict(orient="records"),
        "monthly": monthly.to_dict(orient="records"),
        "downloads": downloads,
    }
    if split_groups and not monthly.empty:
        groups = {}
        for fam, grp in monthly.groupby("family"):
            groups[fam] = grp.sort_values("date", ascending=False).head(3).to_dict(orient="records")
        payload["monthly_groups"] = groups
    return payload


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    ensure_dir(DOCS_DATA_DIR)
    ensure_dir(DOWNLOAD_DIR)
    ensure_dir(LIB_DIR)

    all_valid = get_valid_snapshots(None)
    base_valid = [x for x in all_valid if x[2].universe == args.universe]
    sp500_valid = [x for x in all_valid if x[2].universe == "sp500"]

    if not base_valid:
        raise RuntimeError("No valid usall snapshots found.")

    latest_usall_path, latest_usall, _ = base_valid[-1]
    asof_date = str(latest_usall["date"].iloc[0])

    market_latest, market_daily, market_monthly = build_marketwide(base_valid, sp500_valid)
    value_latest, value_daily, value_monthly = build_value_family(base_valid)
    industry_latest, industry_daily, industry_monthly = build_industry_family(base_valid)
    indices_latest, indices_daily, indices_monthly = build_indices_family(all_valid)
    etf_latest, etf_daily, etf_monthly = build_online_family("etfs.csv", latest_usall, "etf_history.csv")
    country_latest, country_daily, country_monthly = build_online_family("country_etfs.csv", latest_usall, "country_history.csv")

    downloads_market = family_download_tree("marketwide", market_latest, market_daily, market_monthly)
    downloads_value = family_download_tree("value", value_latest, value_daily, value_monthly)
    downloads_industry = family_download_tree("industry", industry_latest, industry_daily, industry_monthly)
    downloads_etf = family_download_tree("etf", etf_latest, etf_daily, etf_monthly)
    downloads_country = family_download_tree("country", country_latest, country_daily, country_monthly)
    downloads_indices = family_download_tree("indices", indices_latest, indices_daily, indices_monthly)
    raw_downloads = build_raw_downloads(all_valid)

    overview_rows = []
    overview_rows.extend(overview_family_rows(market_latest, market_monthly, "marketwide"))
    overview_rows.extend(overview_family_rows(value_latest, value_monthly, "value"))
    overview_rows.extend(overview_family_rows(industry_latest, industry_monthly, "industry"))
    overview_rows.extend(overview_family_rows(etf_latest.rename(columns={"ticker": "family"}), etf_monthly.rename(columns={"ticker": "family"}), "etf"))
    overview_rows.extend(overview_family_rows(country_latest.rename(columns={"ticker": "family"}), country_monthly.rename(columns={"ticker": "family"}), "country"))
    overview_rows.extend(overview_family_rows(indices_latest, indices_monthly, "indices"))

    write_json(
        DOCS_DATA_DIR / "overview.json",
        {
            "title": "Live ICC data library",
            "asof_date": asof_date,
            "source_file": str(latest_usall_path.relative_to(REPO)),
            "rows": overview_rows,
        },
    )

    write_json(DOCS_DATA_DIR / "marketwide.json", build_family_page_payload("Marketwide", market_latest, market_daily, market_monthly, downloads_market, split_groups=True))
    write_json(DOCS_DATA_DIR / "value.json", build_family_page_payload("Value premium", value_latest, value_daily, value_monthly, downloads_value, split_groups=True))
    write_json(DOCS_DATA_DIR / "industry.json", build_family_page_payload("Industry", industry_latest, industry_daily, industry_monthly, downloads_industry))
    write_json(DOCS_DATA_DIR / "etf.json", build_family_page_payload("ETF", etf_latest, etf_daily, etf_monthly, downloads_etf))
    write_json(DOCS_DATA_DIR / "country.json", build_family_page_payload("Country", country_latest, country_daily, country_monthly, downloads_country))
    write_json(DOCS_DATA_DIR / "indices.json", build_family_page_payload("Indices", indices_latest, indices_daily, indices_monthly, downloads_indices))

    write_json(
        DOCS_DATA_DIR / "downloads_catalog.json",
        {
            "title": "Downloads",
            "asof_date": asof_date,
            "families": {
                "marketwide": downloads_market,
                "value": downloads_value,
                "industry": downloads_industry,
                "etf": downloads_etf,
                "country": downloads_country,
                "indices": downloads_indices,
            },
            "raw": raw_downloads,
        },
    )

    print(f"[build_docs_data] asof_date = {asof_date}")
    print(f"[build_docs_data] source_file = {latest_usall_path}")
    print("[build_docs_data] wrote docs/data/*.json and docs/data/downloads/*")


if __name__ == "__main__":
    main()
