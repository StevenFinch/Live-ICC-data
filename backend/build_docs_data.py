
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:
    yf = None


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DIR = REPO / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"
DOWNLOADS_DIR = DOCS_DATA_DIR / "downloads"
RAW_DIR = DOWNLOADS_DIR / "raw"
FAMILIES_DIR = DOWNLOADS_DIR / "families"
CONFIG_DIR = REPO / "config"

SNAP_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)

INDEX_UNIVERSES = ["sp500", "sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def json_safe(obj):
    """Convert pandas/numpy/time objects into JSON-safe Python values."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def write_json(path: Path, payload: dict) -> None:
    """Write JSON after converting unsafe values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)


def parse_snapshot_meta(path: Path) -> dict | None:
    m = SNAP_RE.match(path.name)
    if not m:
        return None
    year = m.group("year")
    mmdd = m.group("mmdd")
    rerun = int(m.group("rerun") or 0)
    d = datetime.strptime(f"{year}{mmdd}", "%Y%m%d").date()
    return {
        "universe": m.group("universe"),
        "date": d,
        "yyyymm": d.strftime("%Y%m"),
        "rerun": rerun,
    }


def find_snapshots(universe: str | None = None) -> list[Path]:
    """Find all snapshot files matching the naming convention."""
    rows = []
    for p in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(p)
        if meta is None:
            continue
        if universe is not None and meta["universe"] != universe:
            continue
        rows.append(((meta["universe"], meta["date"], meta["rerun"], p.name), p))
    rows.sort(key=lambda x: x[0])
    return [p for _, p in rows]


def load_snapshot(path: Path) -> pd.DataFrame:
    """Load one snapshot CSV and normalize columns."""
    if not path.exists() or path.stat().st_size == 0:
        raise pd.errors.EmptyDataError("No columns to parse from file")

    df = pd.read_csv(path)
    if df is None or df.empty or len(df.columns) == 0:
        raise pd.errors.EmptyDataError("No columns to parse from file")

    df.columns = [str(c).strip() for c in df.columns]

    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in ["mktcap", "ICC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "bm" in df.columns:
        df["bm"] = pd.to_numeric(df["bm"], errors="coerce")
    else:
        df["bm"] = np.nan

    if "sector" not in df.columns:
        df["sector"] = None
    if "name" not in df.columns:
        df["name"] = None

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)

    keep = (
        df["ticker"].notna()
        & df["date"].notna()
        & df["mktcap"].notna()
        & df["ICC"].notna()
        & np.isfinite(df["mktcap"])
        & np.isfinite(df["ICC"])
        & (df["mktcap"] > 0)
    )
    return df.loc[keep].copy()


def get_valid_snapshots(universe: str | None = None) -> list[tuple[Path, pd.DataFrame, dict]]:
    """Load all valid snapshots and skip malformed or empty files."""
    out = []
    for p in find_snapshots(universe):
        meta = parse_snapshot_meta(p)
        if meta is None:
            continue
        try:
            df = load_snapshot(p)
        except Exception as e:
            print(f"[build_docs_data] skip invalid snapshot {p}: {type(e).__name__}: {e}")
            continue
        if df.empty:
            continue
        out.append((p, df, meta))
    return out


def dedup_by_universe_date(items: list[tuple[Path, pd.DataFrame, dict]]) -> list[tuple[Path, pd.DataFrame, dict]]:
    """Keep only the latest rerun for each (universe, date)."""
    keep: dict[tuple[str, date], tuple[Path, pd.DataFrame, dict]] = {}
    for item in items:
        p, df, meta = item
        key = (meta["universe"], meta["date"])
        prev = keep.get(key)
        if prev is None or meta["rerun"] >= prev[2]["rerun"]:
            keep[key] = item
    out = list(keep.values())
    out.sort(key=lambda x: (x[2]["universe"], x[2]["date"], x[2]["rerun"], x[0].name))
    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    """Compute a robust weighted average."""
    mask = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return None
    return float(np.average(values[mask], weights=weights[mask]))


def latest_per_month(df: pd.DataFrame, group_cols: list[str], keep_cols: list[str]) -> pd.DataFrame:
    """Keep the latest observation in each month for each group."""
    if df is None or df.empty:
        cols = list(dict.fromkeys(group_cols + ["date", "month"] + keep_cols))
        return pd.DataFrame(columns=cols)

    tmp = df.copy()
    if tmp.columns.duplicated().any():
        tmp = tmp.loc[:, ~tmp.columns.duplicated()].copy()

    if "date" not in tmp.columns:
        cols = list(dict.fromkeys(group_cols + ["date", "month"] + keep_cols))
        return pd.DataFrame(columns=cols)

    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp[tmp["date"].notna()].copy()
    if tmp.empty:
        cols = list(dict.fromkeys(group_cols + ["date", "month"] + keep_cols))
        return pd.DataFrame(columns=cols)

    existing_groups = [c for c in group_cols if c in tmp.columns]
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)

    sort_cols = existing_groups + ["date"] if existing_groups else ["date"]
    tmp = tmp.sort_values(sort_cols).copy()

    dedup_cols = existing_groups + ["month"] if existing_groups else ["month"]
    tmp = tmp.drop_duplicates(subset=dedup_cols, keep="last").copy()

    cols = [c for c in existing_groups + ["date", "month"] + keep_cols if c in tmp.columns]
    cols = list(dict.fromkeys(cols))
    return tmp[cols].reset_index(drop=True)


def ensure_dirs() -> None:
    """Create output directories."""
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FAMILIES_DIR.mkdir(parents=True, exist_ok=True)


def to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame into JSON-safe records."""
    if df is None or df.empty:
        return []
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    return json_safe(out.to_dict(orient="records"))


def write_family_outputs(slug: str, latest_df: pd.DataFrame, daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> dict:
    """Write family JSON and CSV downloads."""
    family_dir = FAMILIES_DIR / slug
    family_dir.mkdir(parents=True, exist_ok=True)

    latest_csv = family_dir / "latest.csv"
    daily_csv = family_dir / "daily_history.csv"
    monthly_csv = family_dir / "monthly_history.csv"

    latest_df.to_csv(latest_csv, index=False)
    daily_df.to_csv(daily_csv, index=False)
    monthly_df.to_csv(monthly_csv, index=False)

    payload = {
        "family": slug,
        "latest": to_records(latest_df),
        "daily": to_records(daily_df),
        "monthly": to_records(monthly_df),
        "last_three_monthly": to_records(monthly_df.sort_values(["month"]).groupby("family", as_index=False).tail(3) if ("family" in monthly_df.columns and not monthly_df.empty) else monthly_df.tail(3)),
        "downloads": {
            "latest_csv": f"./data/downloads/families/{slug}/latest.csv",
            "daily_history_csv": f"./data/downloads/families/{slug}/daily_history.csv",
            "monthly_history_csv": f"./data/downloads/families/{slug}/monthly_history.csv",
        },
    }
    write_json(DOCS_DATA_DIR / f"{slug}.json", payload)
    return payload["downloads"]


def build_marketwide(base_valid: list[tuple[Path, pd.DataFrame, dict]], all_by_universe: dict[str, list[tuple[Path, pd.DataFrame, dict]]]):
    """Build marketwide outputs for all_market and sp500."""
    rows = []
    for _, df, meta in base_valid:
        rows.append({
            "date": meta["date"],
            "family": "all_market",
            "value": weighted_mean(df["ICC"], df["mktcap"]),
            "method": "ICC calculation",
            "n_items": int(len(df)),
        })

    for _, df, meta in all_by_universe.get("sp500", []):
        rows.append({
            "date": meta["date"],
            "family": "sp500",
            "value": weighted_mean(df["ICC"], df["mktcap"]),
            "method": "ICC calculation",
            "n_items": int(len(df)),
        })

    daily = pd.DataFrame(rows)
    if daily.empty:
        latest = pd.DataFrame(columns=["date", "family", "value", "method", "n_items"])
        monthly = pd.DataFrame(columns=["family", "date", "month", "value", "method"])
        return latest, daily, monthly

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["family", "date"]).drop_duplicates(["family", "date"], keep="last").reset_index(drop=True)
    latest = daily.sort_values(["family", "date"]).groupby("family", as_index=False).tail(1).reset_index(drop=True)
    monthly = latest_per_month(daily, ["family"], ["value", "method", "n_items"])
    return latest, daily, monthly


def build_value(base_valid: list[tuple[Path, pd.DataFrame, dict]]):
    """Build value, growth, and IVP series from usall."""
    rows = []
    for _, raw_df, meta in base_valid:
        df = raw_df.copy()
        df = df[
            df["bm"].notna() & np.isfinite(df["bm"]) & (df["bm"] > 0)
        ].copy()
        if df.empty:
            continue

        q_lo = float(df["ICC"].quantile(0.005))
        q_hi = float(df["ICC"].quantile(0.995))
        df = df[(df["ICC"] >= q_lo) & (df["ICC"] <= q_hi)].copy()
        if df.empty:
            continue

        size_med = float(df["mktcap"].median())
        bm30 = float(df["bm"].quantile(0.30))
        bm70 = float(df["bm"].quantile(0.70))

        df["size_bucket"] = np.where(df["mktcap"] <= size_med, "S", "B")
        df["bm_bucket"] = pd.cut(df["bm"], bins=[-np.inf, bm30, bm70, np.inf], labels=["L", "M", "H"], include_lowest=True)
        df["bucket"] = df["size_bucket"].astype(str) + "/" + df["bm_bucket"].astype(str)

        bucket_icc = {}
        for bucket, g in df.groupby("bucket", dropna=False):
            bucket_icc[bucket] = weighted_mean(g["ICC"], g["mktcap"])

        growth = None
        value = None
        if bucket_icc.get("S/L") is not None and bucket_icc.get("B/L") is not None:
            growth = (bucket_icc["S/L"] + bucket_icc["B/L"]) / 2.0
        if bucket_icc.get("S/H") is not None and bucket_icc.get("B/H") is not None:
            value = (bucket_icc["S/H"] + bucket_icc["B/H"]) / 2.0

        ivp = None if (value is None or growth is None) else value - growth

        rows.extend([
            {"date": meta["date"], "family": "value", "value": value, "method": "ICC calculation"},
            {"date": meta["date"], "family": "growth", "value": growth, "method": "ICC calculation"},
            {"date": meta["date"], "family": "ivp_bm", "value": ivp, "method": "ICC calculation"},
        ])

    daily = pd.DataFrame(rows)
    if daily.empty:
        latest = pd.DataFrame(columns=["date", "family", "value", "method"])
        monthly = pd.DataFrame(columns=["family", "date", "month", "value", "method"])
        return latest, daily, monthly

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["family", "date"]).drop_duplicates(["family", "date"], keep="last").reset_index(drop=True)
    latest = daily.sort_values(["family", "date"]).groupby("family", as_index=False).tail(1).reset_index(drop=True)
    monthly = latest_per_month(daily, ["family"], ["value", "method"])
    return latest, daily, monthly


def build_industry(base_valid: list[tuple[Path, pd.DataFrame, dict]]):
    """Build industry series from latest and historical usall snapshots."""
    rows = []
    for _, df, meta in base_valid:
        tmp = df[df["sector"].notna()].copy()
        if tmp.empty:
            continue
        for sector, g in tmp.groupby("sector", dropna=False):
            rows.append({
                "date": meta["date"],
                "family": str(sector),
                "value": weighted_mean(g["ICC"], g["mktcap"]),
                "method": "ICC calculation",
                "n_items": int(len(g)),
            })

    daily = pd.DataFrame(rows)
    if daily.empty:
        latest = pd.DataFrame(columns=["date", "family", "value", "method", "n_items"])
        monthly = pd.DataFrame(columns=["family", "date", "month", "value", "method", "n_items"])
        return latest, daily, monthly

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["family", "date"]).drop_duplicates(["family", "date"], keep="last").reset_index(drop=True)
    latest = daily.sort_values(["family", "date"]).groupby("family", as_index=False).tail(1).reset_index(drop=True)
    monthly = latest_per_month(daily, ["family"], ["value", "method", "n_items"])
    return latest, daily, monthly


def build_indices(all_by_universe: dict[str, list[tuple[Path, pd.DataFrame, dict]]]):
    """Build index family from all supported extra universes."""
    rows = []
    for universe in INDEX_UNIVERSES:
        for _, df, meta in all_by_universe.get(universe, []):
            rows.append({
                "date": meta["date"],
                "family": universe,
                "value": weighted_mean(df["ICC"], df["mktcap"]),
                "method": "ICC calculation",
                "n_items": int(len(df)),
            })

    daily = pd.DataFrame(rows)
    if daily.empty:
        latest = pd.DataFrame(columns=["date", "family", "value", "method", "n_items"])
        monthly = pd.DataFrame(columns=["family", "date", "month", "value", "method", "n_items"])
        return latest, daily, monthly

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["family", "date"]).drop_duplicates(["family", "date"], keep="last").reset_index(drop=True)
    latest = daily.sort_values(["family", "date"]).groupby("family", as_index=False).tail(1).reset_index(drop=True)
    monthly = latest_per_month(daily, ["family"], ["value", "method", "n_items"])
    return latest, daily, monthly


def read_csv_config(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    """Read one CSV config and ensure all expected columns exist."""
    if not path.exists():
        return pd.DataFrame(columns=expected_cols)
    df = pd.read_csv(path)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols].copy()


def fetch_url_text(url: str) -> str:
    """Fetch HTML text from one URL."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


PE_PATTERNS = [
    re.compile(r"P/E[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"Price\/Earnings[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"PE Ratio[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)", re.I),
]


def fetch_page_pe(url: str) -> float | None:
    """Parse a rough P/E value from a public ETF page."""
    if not url:
        return None
    try:
        text = fetch_url_text(url)
    except Exception:
        return None

    text = text.replace(",", " ")
    for pat in PE_PATTERNS:
        m = pat.search(text)
        if m:
            pe = float(m.group(1))
            if pe > 0:
                return pe
    return None


def fetch_yfinance_pe(ticker: str) -> float | None:
    """Fetch a rough PE value from yfinance."""
    if yf is None or not ticker:
        return None
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        return None
    for k in ["forwardPE", "trailingPE"]:
        v = info.get(k)
        try:
            v = float(v)
        except Exception:
            continue
        if np.isfinite(v) and v > 0:
            return v
    return None


def fetch_yfinance_top_holdings(ticker: str) -> pd.DataFrame:
    """Fetch top holdings from yfinance funds_data."""
    if yf is None:
        return pd.DataFrame(columns=["symbol", "weight"])
    try:
        fd = getattr(yf.Ticker(ticker), "funds_data", None)
        if fd is None:
            return pd.DataFrame(columns=["symbol", "weight"])
        holdings = getattr(fd, "top_holdings", None)
        if holdings is None:
            return pd.DataFrame(columns=["symbol", "weight"])
        h = holdings.copy() if isinstance(holdings, pd.DataFrame) else pd.DataFrame(holdings)
    except Exception:
        return pd.DataFrame(columns=["symbol", "weight"])

    if h.empty:
        return pd.DataFrame(columns=["symbol", "weight"])

    cols = {str(c).lower(): c for c in h.columns}
    symbol_col = None
    for cand in ["symbol", "ticker", "holding"]:
        if cand in cols:
            symbol_col = cols[cand]
            break
    weight_col = None
    for cand in ["holdingpercent", "holding_percent", "weight", "percent", "pct"]:
        if cand in cols:
            weight_col = cols[cand]
            break

    if symbol_col is None:
        symbol_col = h.columns[0]
    if weight_col is None:
        for c in h.columns[::-1]:
            vals = pd.to_numeric(h[c], errors="coerce")
            if vals.notna().sum() >= max(1, len(h) // 2):
                weight_col = c
                break

    if weight_col is None:
        return pd.DataFrame(columns=["symbol", "weight"])

    out = pd.DataFrame({
        "symbol": h[symbol_col].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False),
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


def holdings_based_proxy(ticker: str, base_usall: pd.DataFrame) -> dict | None:
    """Build a holdings-based proxy by merging top holdings into usall ICC."""
    holdings = fetch_yfinance_top_holdings(ticker)
    if holdings.empty:
        return None

    merged = holdings.merge(
        base_usall[["ticker", "ICC"]],
        left_on="symbol",
        right_on="ticker",
        how="left",
    )

    matched = merged["ICC"].notna()
    coverage = float(merged.loc[matched, "weight"].sum()) if len(merged) else 0.0
    n_matched = int(matched.sum())
    if coverage <= 0 or n_matched == 0:
        return None

    value = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"])
    if value is None:
        return None

    return {
        "value": value,
        "method": "ICC calculation",
        "coverage_weight": coverage,
        "n_items": int(len(holdings)),
        "n_matched": n_matched,
        "status": "ICC calculation",
    }


def build_online_family(config_name: str, latest_usall: pd.DataFrame, history_name: str):
    """Build latest, daily, and monthly outputs for online ETF/Country families."""
    config_path = CONFIG_DIR / config_name

    if "country" in config_name.lower():
        expected_cols = ["family", "ticker", "label", "url", "allow_holdings"]
        default_family_col = "family"
    else:
        expected_cols = ["ticker", "label", "category", "url", "allow_holdings"]
        default_family_col = "ticker"

    cfg = read_csv_config(config_path, expected_cols)
    if cfg.empty:
        empty_latest = pd.DataFrame(columns=expected_cols + ["date", "value", "method", "coverage_weight", "n_items", "n_matched", "status"])
        empty_daily = pd.DataFrame(columns=expected_cols + ["date", "value", "method", "coverage_weight", "n_items", "n_matched", "status"])
        empty_monthly = pd.DataFrame(columns=[default_family_col, "date", "month", "value", "method"])
        return empty_latest, empty_daily, empty_monthly

    date_value = str(pd.to_datetime(latest_usall["date"]).dt.strftime("%Y-%m-%d").iloc[0])
    base_usall = latest_usall.copy()

    rows = []
    for _, r in cfg.iterrows():
        ticker = str(r.get("ticker", "") or "").strip().upper()
        family = str(r.get("family", "") or "").strip()
        label = str(r.get("label", "") or "").strip()
        category = str(r.get("category", "") or "").strip()
        url = str(r.get("url", "") or "").strip()
        allow_holdings = int(pd.to_numeric(r.get("allow_holdings", 1), errors="coerce") or 0)

        if not family:
            family = ticker

        result = None

        if allow_holdings == 1 and ticker:
            try:
                result = holdings_based_proxy(ticker, base_usall)
                if result is not None:
                    coverage = float(result.get("coverage_weight", 0.0) or 0.0)
                    n_matched = int(result.get("n_matched", 0) or 0)
                    if not (coverage >= 0.20 and n_matched >= 3):
                        result = None
            except Exception:
                result = None

        if result is None and url:
            pe = fetch_page_pe(url)
            if pe is not None and pe > 0:
                result = {
                    "value": float(1.0 / pe),
                    "method": "P/E estimate",
                    "coverage_weight": None,
                    "n_items": None,
                    "n_matched": None,
                    "status": "P/E estimate",
                }

        if result is None and ticker:
            pe = fetch_yfinance_pe(ticker)
            if pe is not None and pe > 0:
                result = {
                    "value": float(1.0 / pe),
                    "method": "P/E estimate",
                    "coverage_weight": None,
                    "n_items": None,
                    "n_matched": None,
                    "status": "P/E estimate",
                }

        if result is None:
            result = {
                "value": None,
                "method": "Unavailable",
                "coverage_weight": None,
                "n_items": None,
                "n_matched": None,
                "status": "Unavailable",
            }

        rows.append({
            "date": date_value,
            "ticker": ticker,
            "family": family,
            "label": label,
            "category": category if "category" in cfg.columns else None,
            "url": url,
            "allow_holdings": allow_holdings,
            "value": result.get("value"),
            "method": result.get("method"),
            "coverage_weight": result.get("coverage_weight"),
            "n_items": result.get("n_items"),
            "n_matched": result.get("n_matched"),
            "status": result.get("status"),
        })

    latest = pd.DataFrame(rows)

    history_path = FAMILIES_DIR / history_name
    if history_path.exists():
        old = pd.read_csv(history_path)
        if old.columns.duplicated().any():
            old = old.loc[:, ~old.columns.duplicated()].copy()
    else:
        old = pd.DataFrame(columns=latest.columns)

    all_cols = list(dict.fromkeys(list(old.columns) + list(latest.columns)))
    for c in all_cols:
        if c not in old.columns:
            old[c] = None
        if c not in latest.columns:
            latest[c] = None

    old = old[all_cols].copy()
    latest = latest[all_cols].copy()

    key_cols = ["date", "family"]
    if not old.empty:
        old = old.drop_duplicates(subset=[c for c in key_cols if c in old.columns], keep="last").copy()

    daily = pd.concat([old, latest], ignore_index=True)
    daily = daily.drop_duplicates(subset=[c for c in key_cols if c in daily.columns], keep="last").copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily[daily["date"].notna()].copy()
    daily = daily.sort_values([c for c in ["family", "date"] if c in daily.columns]).reset_index(drop=True)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    out_daily = daily.copy()
    out_daily["date"] = out_daily["date"].dt.strftime("%Y-%m-%d")
    out_daily.to_csv(history_path, index=False)

    daily_for_monthly = daily.copy()
    if "family" not in daily_for_monthly.columns:
        daily_for_monthly["family"] = daily_for_monthly.get(default_family_col)

    if daily_for_monthly.columns.duplicated().any():
        daily_for_monthly = daily_for_monthly.loc[:, ~daily_for_monthly.columns.duplicated()].copy()

    monthly = latest_per_month(daily_for_monthly, ["family"], ["value", "method"])
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
    latest["date"] = latest["date"].dt.strftime("%Y-%m-%d")
    out_daily["date"] = pd.to_datetime(out_daily["date"], errors="coerce")
    out_daily["date"] = out_daily["date"].dt.strftime("%Y-%m-%d")

    return latest, out_daily, monthly


def copy_and_zip_raw(all_valid: list[tuple[Path, pd.DataFrame, dict]]) -> dict:
    """Copy raw snapshots into docs and create year/month zip files."""
    bucket_map = {}
    for p, _, meta in all_valid:
        universe = meta["universe"]
        if universe == "usall":
            bucket = "usall"
        elif universe == "sp500":
            bucket = "sp500"
        else:
            bucket = "other_indices"

        year = meta["date"].strftime("%Y")
        month = meta["date"].strftime("%Y%m")

        dst_dir = RAW_DIR / bucket / year / month
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / p.name
        shutil.copy2(p, dst)

        bucket_map.setdefault(bucket, {}).setdefault(year, {}).setdefault(month, []).append(dst)

    catalog = {}
    for bucket, years in bucket_map.items():
        catalog[bucket] = {"years": []}
        for year, months in sorted(years.items(), reverse=True):
            year_zip = RAW_DIR / bucket / year / f"{bucket}_{year}_all.zip"
            with zipfile.ZipFile(year_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for month_files in months.values():
                    for f in month_files:
                        zf.write(f, arcname=f.name)

            year_entry = {
                "year": year,
                "download_all": f"./data/downloads/raw/{bucket}/{year}/{year_zip.name}",
                "months": [],
            }

            for month, files in sorted(months.items(), reverse=True):
                month_zip = RAW_DIR / bucket / year / f"{bucket}_{month}_all.zip"
                with zipfile.ZipFile(month_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for f in files:
                        zf.write(f, arcname=f.name)

                month_entry = {
                    "month": month,
                    "download_all": f"./data/downloads/raw/{bucket}/{year}/{month_zip.name}",
                    "files": [],
                }

                for f in sorted(files, reverse=True):
                    month_entry["files"].append({
                        "label": f.name,
                        "path": f"./data/downloads/raw/{bucket}/{year}/{month}/{f.name}",
                    })

                year_entry["months"].append(month_entry)

            catalog[bucket]["years"].append(year_entry)

    return catalog


def build_overview_payload(marketwide_monthly, value_monthly, industry_latest, etf_latest, country_latest, indices_latest):
    """Build the homepage overview payload."""
    def latest_value(df, family):
        if df is None or df.empty:
            return None
        tmp = df[df["family"] == family].copy()
        if tmp.empty:
            return None
        tmp = tmp.sort_values("date")
        return tmp.iloc[-1].get("value")

    all_three = []
    sp500_three = []
    if marketwide_monthly is not None and not marketwide_monthly.empty:
        all_three = to_records(marketwide_monthly[marketwide_monthly["family"] == "all_market"].sort_values("date").tail(3))
        sp500_three = to_records(marketwide_monthly[marketwide_monthly["family"] == "sp500"].sort_values("date").tail(3))

    return {
        "title": "Live ICC data library",
        "overview_cards": [
            {"family": "All market", "value": latest_value(marketwide_monthly, "all_market"), "method": "ICC calculation"},
            {"family": "S&P 500", "value": latest_value(marketwide_monthly, "sp500"), "method": "ICC calculation"},
            {"family": "Value premium", "value": latest_value(value_monthly, "ivp_bm"), "method": "ICC calculation"},
            {"family": "Industry", "value": None if industry_latest.empty else industry_latest.iloc[0].get("value"), "method": None if industry_latest.empty else industry_latest.iloc[0].get("method")},
            {"family": "ETF", "value": None if etf_latest.empty else etf_latest.iloc[0].get("value"), "method": None if etf_latest.empty else etf_latest.iloc[0].get("method")},
            {"family": "Country", "value": None if country_latest.empty else country_latest.iloc[0].get("value"), "method": None if country_latest.empty else country_latest.iloc[0].get("method")},
            {"family": "Indices", "value": None if indices_latest.empty else indices_latest.iloc[0].get("value"), "method": None if indices_latest.empty else indices_latest.iloc[0].get("method")},
        ],
        "marketwide_last_three": {
            "all_market": all_three,
            "sp500": sp500_three,
        },
    }


def main() -> None:
    """Build all JSON and CSV outputs for the website."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    ensure_dirs()

    base_valid = dedup_by_universe_date(get_valid_snapshots(args.universe))
    if not base_valid:
        raise RuntimeError(f"No valid snapshots found for {args.universe}")

    all_valid = dedup_by_universe_date(get_valid_snapshots(None))
    all_by_universe: dict[str, list[tuple[Path, pd.DataFrame, dict]]] = {}
    for item in all_valid:
        all_by_universe.setdefault(item[2]["universe"], []).append(item)

    latest_usall = base_valid[-1][1].copy()
    latest_usall["date"] = pd.to_datetime(latest_usall["date"], errors="coerce")

    marketwide_latest, marketwide_daily, marketwide_monthly = build_marketwide(base_valid, all_by_universe)
    value_latest, value_daily, value_monthly = build_value(base_valid)
    industry_latest, industry_daily, industry_monthly = build_industry(base_valid)
    indices_latest, indices_daily, indices_monthly = build_indices(all_by_universe)
    etf_latest, etf_daily, etf_monthly = build_online_family("etfs.csv", latest_usall, "etf_history.csv")
    country_latest, country_daily, country_monthly = build_online_family("country_etfs.csv", latest_usall, "country_history.csv")

    downloads_catalog = {
        "marketwide": write_family_outputs("marketwide", marketwide_latest, marketwide_daily, marketwide_monthly),
        "value": write_family_outputs("value", value_latest, value_daily, value_monthly),
        "industry": write_family_outputs("industry", industry_latest, industry_daily, industry_monthly),
        "indices": write_family_outputs("indices", indices_latest, indices_daily, indices_monthly),
        "etf": write_family_outputs("etf", etf_latest, etf_daily, etf_monthly),
        "country": write_family_outputs("country", country_latest, country_daily, country_monthly),
    }

    raw_catalog = copy_and_zip_raw(all_valid)

    overview = build_overview_payload(
        marketwide_monthly,
        value_monthly,
        industry_latest,
        etf_latest,
        country_latest,
        indices_latest,
    )
    overview["as_of"] = base_valid[-1][2]["date"]
    write_json(DOCS_DATA_DIR / "overview.json", overview)

    write_json(
        DOCS_DATA_DIR / "downloads_catalog.json",
        {
            "title": "Live ICC data library",
            "family_downloads": downloads_catalog,
            "raw_downloads": raw_catalog,
        },
    )

    # Compatibility outputs for older front-end paths.
    write_json(DOCS_DATA_DIR / "market_icc.json", json.load(open(DOCS_DATA_DIR / "marketwide.json", "r", encoding="utf-8")))
    write_json(DOCS_DATA_DIR / "value_icc_bm.json", json.load(open(DOCS_DATA_DIR / "value.json", "r", encoding="utf-8")))
    write_json(DOCS_DATA_DIR / "industry_icc.json", json.load(open(DOCS_DATA_DIR / "industry.json", "r", encoding="utf-8")))
    write_json(DOCS_DATA_DIR / "etf_icc.json", json.load(open(DOCS_DATA_DIR / "etf.json", "r", encoding="utf-8")))
    write_json(DOCS_DATA_DIR / "country_icc.json", json.load(open(DOCS_DATA_DIR / "country.json", "r", encoding="utf-8")))
    write_json(DOCS_DATA_DIR / "index_icc.json", json.load(open(DOCS_DATA_DIR / "indices.json", "r", encoding="utf-8")))

    print(f"[build_docs_data] wrote docs data to {DOCS_DATA_DIR}")


if __name__ == "__main__":
    main()
