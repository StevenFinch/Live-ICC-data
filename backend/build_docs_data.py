
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import zipfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pandas.errors import EmptyDataError

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DATA_DIR = REPO / "docs" / "data"
DOWNLOAD_DIR = DOCS_DATA_DIR / "downloads"
RAW_DOWNLOAD_DIR = DOWNLOAD_DIR / "raw"
FAMILY_DOWNLOAD_DIR = DOWNLOAD_DIR / "families"
CONFIG_DIR = REPO / "config"

SNAPSHOT_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)

INDEX_UNIVERSES = [
    "sp500",
    "sp100",
    "dow30",
    "ndx100",
    "sp400",
    "sp600",
    "sp1500",
    "rut1000",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}


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


def safe_float(x: Any):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        val = float(x)
        if not np.isfinite(val):
            return None
        return val
    except Exception:
        return None


def safe_int(x: Any):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        return None


def json_safe(obj: Any):
    """Recursively convert pandas/numpy/date objects into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [json_safe(v) for v in sorted(obj, key=lambda x: str(x))]

    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.isoformat()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x
    if isinstance(obj, np.bool_):
        return bool(obj)

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)


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


def find_all_snapshots(universe: str | None = None) -> list[Path]:
    files = []
    for p in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(p)
        if meta is None:
            continue
        if universe is not None and meta["universe"] != universe:
            continue
        files.append(((meta["yyyymmdd"], meta["rerun"], p.name), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    if path.stat().st_size == 0:
        raise EmptyDataError("No columns to parse from file")

    df = pd.read_csv(path)
    if df is None or len(df.columns) == 0:
        raise EmptyDataError("No columns to parse from file")

    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for c in ["mktcap", "ICC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "bm" in df.columns:
        df["bm"] = pd.to_numeric(df["bm"], errors="coerce")
    else:
        df["bm"] = np.nan
    if "sector" not in df.columns:
        df["sector"] = None
    if "name" not in df.columns:
        df["name"] = None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def try_load_snapshot(path: Path) -> pd.DataFrame | None:
    try:
        return load_snapshot(path)
    except Exception as e:
        print(f"[build_docs_data] skip invalid snapshot {path}: {type(e).__name__}: {e}")
        return None


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        df["ticker"].notna()
        & df["date"].notna()
        & df["mktcap"].notna()
        & df["ICC"].notna()
        & np.isfinite(df["mktcap"])
        & np.isfinite(df["ICC"])
        & (df["mktcap"] > 0)
    ].copy()
    return out


def get_valid_snapshots(paths: list[Path]) -> list[tuple[Path, pd.DataFrame]]:
    valid = []
    for path in paths:
        df = try_load_snapshot(path)
        if df is None:
            continue
        df = clean_df(df)
        if df.empty:
            continue
        valid.append((path, df))
    return valid


def dedup_valid_snapshots(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> list[tuple[Path, pd.DataFrame]]:
    """Keep the last valid file for each universe/date pair."""
    keep = {}
    for path, df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        d = df["date"].dropna()
        if d.empty:
            continue
        date_value = d.iloc[0].strftime("%Y-%m-%d")
        keep[(meta["universe"], date_value)] = (path, df)
    out = list(keep.values())
    out.sort(
        key=lambda x: (
            parse_snapshot_meta(x[0])["universe"],
            x[1]["date"].iloc[0],
            parse_snapshot_meta(x[0])["rerun"],
            x[0].name,
        )
    )
    return out


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

    valid_groups = [c for c in group_cols if c in tmp.columns]
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    sort_cols = valid_groups + ["date"] if valid_groups else ["date"]
    tmp = tmp.sort_values(sort_cols).copy()
    dedup_cols = valid_groups + ["month"] if valid_groups else ["month"]
    tmp = tmp.drop_duplicates(subset=dedup_cols, keep="last").copy()
    wanted = [c for c in list(dict.fromkeys(valid_groups + ["date", "month"] + keep_cols)) if c in tmp.columns]
    return tmp[wanted].reset_index(drop=True)


def build_market_history(valid_snapshots: list[tuple[Path, pd.DataFrame]], family: str) -> pd.DataFrame:
    rows = []
    for path, df in valid_snapshots:
        d = df["date"].dropna()
        if d.empty:
            continue
        rows.append(
            {
                "date": d.iloc[0],
                "family": family,
                "value": weighted_mean(df["ICC"], df["mktcap"]),
                "method": "ICC calculation",
                "n_firms": int(len(df)),
                "source_file": str(path.relative_to(REPO)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "family", "value", "method", "n_firms", "source_file"])
    out = pd.DataFrame(rows).sort_values("date").drop_duplicates(["family", "date"], keep="last").reset_index(drop=True)
    return out


def build_value_history(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, raw_df in valid_snapshots:
        df = raw_df.copy()
        df = df[
            df["bm"].notna()
            & np.isfinite(df["bm"])
            & (df["bm"] > 0)
            & df["mktcap"].notna()
            & np.isfinite(df["mktcap"])
            & (df["mktcap"] > 0)
            & df["ICC"].notna()
            & np.isfinite(df["ICC"])
            & df["date"].notna()
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
            df["bm"], bins=[-np.inf, bm30, bm70, np.inf], labels=["L", "M", "H"], include_lowest=True
        ).astype("string")
        df["portfolio"] = df["size_bucket"] + "/" + df["bm_bucket"].astype(str)

        pt = (
            df.groupby("portfolio")
            .apply(lambda g: pd.Series({"vw_icc": weighted_mean(g["ICC"], g["mktcap"])}))
            .reset_index()
        )
        dmap = dict(zip(pt["portfolio"], pt["vw_icc"]))
        sl, bl = dmap.get("S/L", np.nan), dmap.get("B/L", np.nan)
        sh, bh = dmap.get("S/H", np.nan), dmap.get("B/H", np.nan)
        growth = np.nan if pd.isna(sl) or pd.isna(bl) else float((sl + bl) / 2.0)
        value = np.nan if pd.isna(sh) or pd.isna(bh) else float((sh + bh) / 2.0)
        ivp = np.nan if pd.isna(value) or pd.isna(growth) else float(value - growth)
        dt = df["date"].iloc[0]
        rows.extend(
            [
                {"date": dt, "family": "value", "value": value, "method": "ICC calculation", "n_firms": int(len(df))},
                {"date": dt, "family": "growth", "value": growth, "method": "ICC calculation", "n_firms": int(len(df))},
                {"date": dt, "family": "ivp_bm", "value": ivp, "method": "ICC calculation", "n_firms": int(len(df))},
            ]
        )

    if not rows:
        return pd.DataFrame(columns=["date", "family", "value", "method", "n_firms"])
    out = pd.DataFrame(rows).sort_values(["family", "date"]).drop_duplicates(["family", "date"], keep="last").reset_index(drop=True)
    return out


def build_latest_industry(latest_usall: pd.DataFrame) -> pd.DataFrame:
    df = clean_df(latest_usall)
    df = df[df["sector"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "family", "label", "value", "method", "n_firms"])
    dt = df["date"].iloc[0]
    out = (
        df.groupby("sector")
        .apply(
            lambda g: pd.Series(
                {
                    "value": weighted_mean(g["ICC"], g["mktcap"]),
                    "n_firms": int(len(g)),
                }
            )
        )
        .reset_index()
        .rename(columns={"sector": "family"})
    )
    out["date"] = dt
    out["label"] = out["family"]
    out["method"] = "ICC calculation"
    return out[["date", "family", "label", "value", "method", "n_firms"]].sort_values("family").reset_index(drop=True)


def read_csv_config(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=expected_cols)
    df = pd.read_csv(path)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols].copy()


def fetch_yfinance_top_holdings(ticker: str) -> pd.DataFrame:
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

    cols = {str(c).strip().lower(): c for c in h.columns}
    symbol_col = None
    weight_col = None

    for c in ["symbol", "ticker", "holding"]:
        if c in cols:
            symbol_col = cols[c]
            break
    for c in ["holdingpercent", "holding_percent", "weight", "percent", "pct"]:
        if c in cols:
            weight_col = cols[c]
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

    out = pd.DataFrame(
        {
            "symbol": h[symbol_col].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False),
            "raw_weight": pd.to_numeric(h[weight_col], errors="coerce"),
        }
    )
    out = out[out["symbol"].notna() & out["raw_weight"].notna()].copy()
    if out.empty:
        return pd.DataFrame(columns=["symbol", "weight"])

    out["weight"] = out["raw_weight"] / 100.0 if out["raw_weight"].max() > 1.5 else out["raw_weight"]
    out = out[out["weight"] > 0].copy()
    if out.empty:
        return pd.DataFrame(columns=["symbol", "weight"])
    out["weight"] = out["weight"] / out["weight"].sum()
    return out[["symbol", "weight"]]


def fetch_page_pe(url: str) -> float | None:
    try:
        text = requests.get(url, timeout=20, headers=HEADERS).text
    except Exception:
        return None
    patterns = [
        r"P/E(?:\s+Ratio)?[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)",
        r"Price/Earnings[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            try:
                v = float(m.group(1))
                if v > 0:
                    return v
            except Exception:
                pass
    return None


def fetch_yfinance_pe(ticker: str) -> float | None:
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        return None
    for key in ["forwardPE", "trailingPE"]:
        val = info.get(key)
        if val is not None:
            try:
                val = float(val)
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
    return None


def holdings_based_proxy(ticker: str, base_usall: pd.DataFrame) -> dict | None:
    h = fetch_yfinance_top_holdings(ticker)
    if h.empty:
        return None
    base = base_usall[["ticker", "ICC"]].copy()
    base["ticker"] = base["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    merged = h.merge(base, left_on="symbol", right_on="ticker", how="left")
    matched = merged["ICC"].notna()
    if matched.sum() == 0:
        return {
            "value": None,
            "coverage_weight": 0.0,
            "n_items": int(len(h)),
            "n_matched": 0,
        }
    coverage = float(merged.loc[matched, "weight"].sum())
    value = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"])
    return {
        "value": value,
        "coverage_weight": coverage,
        "n_items": int(len(h)),
        "n_matched": int(matched.sum()),
    }


def build_online_family(config_name: str, latest_usall: pd.DataFrame, history_name: str):
    """Build latest, daily, and monthly outputs for ETF or Country families."""
    config_path = CONFIG_DIR / config_name

    if "country" in config_name.lower():
        expected_cols = ["family", "ticker", "label", "url", "allow_holdings"]
        fallback_group_col = "family"
    else:
        expected_cols = ["ticker", "label", "category", "url", "allow_holdings"]
        fallback_group_col = "ticker"

    cfg = read_csv_config(config_path, expected_cols)

    if cfg.empty:
        empty_latest = pd.DataFrame(columns=expected_cols + ["date", "value", "method", "coverage_weight", "n_items", "n_matched", "status", "family"])
        empty_daily = empty_latest.copy()
        empty_monthly = pd.DataFrame(columns=["family", "date", "month", "value", "method"])
        return empty_latest, empty_daily, empty_monthly

    dt = latest_usall["date"].dropna().iloc[0]
    date_value = dt.strftime("%Y-%m-%d")
    base_usall = clean_df(latest_usall).copy()

    rows = []
    for _, r in cfg.iterrows():
        ticker = str(r.get("ticker", "") or "").upper().strip()
        family = str(r.get("family", "") or "").strip()
        label = str(r.get("label", "") or "").strip()
        category = str(r.get("category", "") or "").strip()
        url = str(r.get("url", "") or "").strip()
        allow_holdings = int(pd.to_numeric(r.get("allow_holdings", 1), errors="coerce") or 0)

        if not family:
            family = ticker if ticker else label

        result = None

        if allow_holdings == 1 and ticker:
            try:
                result = holdings_based_proxy(ticker, base_usall)
                if result is not None:
                    coverage = float(result.get("coverage_weight", 0.0) or 0.0)
                    n_matched = int(result.get("n_matched", 0) or 0)
                    if result.get("value") is not None and coverage >= 0.20 and n_matched >= 3:
                        result["method"] = "ICC calculation"
                        result["status"] = "ICC calculation"
                    else:
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

        row = {
            "date": date_value,
            "ticker": ticker,
            "family": family,
            "label": label if label else family,
            "category": category if category else None,
            "url": url,
            "allow_holdings": allow_holdings,
            "value": result.get("value"),
            "method": result.get("method"),
            "coverage_weight": result.get("coverage_weight"),
            "n_items": result.get("n_items"),
            "n_matched": result.get("n_matched"),
            "status": result.get("status"),
        }
        rows.append(row)

    latest = pd.DataFrame(rows)

    history_path = FAMILY_DOWNLOAD_DIR / history_name
    history_path.parent.mkdir(parents=True, exist_ok=True)

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

    key_cols = ["date", "family"] if "family" in latest.columns else ["date", fallback_group_col]

    if not old.empty:
        old = old.drop_duplicates(subset=[c for c in key_cols if c in old.columns], keep="last").copy()

    daily = pd.concat([old, latest], ignore_index=True)
    daily = daily.drop_duplicates(subset=[c for c in key_cols if c in daily.columns], keep="last").copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.sort_values([c for c in ["family", "date"] if c in daily.columns]).reset_index(drop=True)

    daily.to_csv(history_path, index=False)

    daily_for_monthly = daily.copy()
    if daily_for_monthly.columns.duplicated().any():
        daily_for_monthly = daily_for_monthly.loc[:, ~daily_for_monthly.columns.duplicated()].copy()
    if "family" not in daily_for_monthly.columns:
        if fallback_group_col in daily_for_monthly.columns:
            daily_for_monthly["family"] = daily_for_monthly[fallback_group_col]
        else:
            daily_for_monthly["family"] = None

    monthly = latest_per_month(daily_for_monthly, ["family"], ["value", "method"])
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return latest, daily, monthly


def build_index_outputs(all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_parts = []
    latest_rows = []
    for universe in INDEX_UNIVERSES:
        valid = all_valid_by_universe.get(universe, [])
        if not valid:
            continue
        hist = build_market_history(valid, universe)
        if hist.empty:
            continue
        daily_parts.append(hist)
        last = hist.sort_values("date").iloc[-1].to_dict()
        latest_rows.append(
            {
                "date": last["date"],
                "family": universe,
                "label": universe,
                "value": last["value"],
                "method": "ICC calculation",
                "n_firms": last["n_firms"],
                "source_file": last["source_file"],
            }
        )

    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame(columns=["date", "family", "value", "method", "n_firms", "source_file"])
    monthly = latest_per_month(daily, ["family"], ["value", "method", "n_firms", "source_file"])
    latest = pd.DataFrame(latest_rows) if latest_rows else pd.DataFrame(columns=["date", "family", "label", "value", "method", "n_firms", "source_file"])
    return latest, daily, monthly


def build_marketwide_outputs(all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_hist = build_market_history(all_valid_by_universe.get("usall", []), "all_market")
    sp_hist = build_market_history(all_valid_by_universe.get("sp500", []), "sp500")
    daily = pd.concat([all_hist, sp_hist], ignore_index=True)
    if daily.empty:
        return (
            pd.DataFrame(columns=["date", "family", "label", "value", "method", "n_firms", "source_file"]),
            daily,
            pd.DataFrame(columns=["family", "date", "month", "value", "method", "n_firms", "source_file"]),
        )
    daily["label"] = daily["family"]
    latest = (
        daily.sort_values("date")
        .drop_duplicates(subset=["family"], keep="last")
        .copy()[["date", "family", "label", "value", "method", "n_firms", "source_file"]]
    )
    monthly = latest_per_month(daily, ["family"], ["value", "method", "n_firms", "source_file"])
    return latest.reset_index(drop=True), daily.reset_index(drop=True), monthly.reset_index(drop=True)


def build_value_outputs(base_valid: list[tuple[Path, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = build_value_history(base_valid)
    if daily.empty:
        return (
            pd.DataFrame(columns=["date", "family", "label", "value", "method", "n_firms"]),
            daily,
            pd.DataFrame(columns=["family", "date", "month", "value", "method", "n_firms"]),
        )
    daily["label"] = daily["family"]
    latest = (
        daily.sort_values("date")
        .drop_duplicates(subset=["family"], keep="last")
        .copy()[["date", "family", "label", "value", "method", "n_firms"]]
    )
    monthly = latest_per_month(daily, ["family"], ["value", "method", "n_firms"])
    return latest.reset_index(drop=True), daily.reset_index(drop=True), monthly.reset_index(drop=True)


def build_industry_outputs(base_valid: list[tuple[Path, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not base_valid:
        empty = pd.DataFrame(columns=["date", "family", "label", "value", "method", "n_firms"])
        return empty, empty.copy(), pd.DataFrame(columns=["family", "date", "month", "value", "method", "n_firms"])
    daily_parts = []
    for _path, df in base_valid:
        out = build_latest_industry(df)
        if not out.empty:
            daily_parts.append(out)
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame(columns=["date", "family", "label", "value", "method", "n_firms"])
    if daily.empty:
        return daily.copy(), daily.copy(), pd.DataFrame(columns=["family", "date", "month", "value", "method", "n_firms"])
    latest = (
        daily.sort_values("date")
        .drop_duplicates(subset=["family"], keep="last")
        .copy()[["date", "family", "label", "value", "method", "n_firms"]]
    )
    monthly = latest_per_month(daily, ["family"], ["label", "value", "method", "n_firms"])
    return latest.reset_index(drop=True), daily.reset_index(drop=True), monthly.reset_index(drop=True)


def copy_raw_snapshots(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    RAW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        sub = "other_indices"
        if meta["universe"] == "usall":
            sub = "usall"
        elif meta["universe"] == "sp500":
            sub = "sp500"
        target_dir = RAW_DOWNLOAD_DIR / sub / meta["yyyymm"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)
        rows.append(
            {
                "category": sub,
                "yyyymm": meta["yyyymm"],
                "date": df["date"].iloc[0].strftime("%Y-%m-%d"),
                "universe": meta["universe"],
                "download_path": f"./data/downloads/raw/{sub}/{meta['yyyymm']}/{path.name}",
                "n_firms": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def make_zip_from_files(zip_path: Path, files: list[Path], arc_root: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            if p.exists() and p.is_file():
                zf.write(p, arcname=str(p.relative_to(arc_root)))


def build_raw_download_catalog(raw_rows: pd.DataFrame) -> dict:
    catalog = {"usall": [], "sp500": [], "other_indices": []}
    if raw_rows.empty:
        return catalog

    base_dir = RAW_DOWNLOAD_DIR
    for category in ["usall", "sp500", "other_indices"]:
        grp = raw_rows[raw_rows["category"] == category].copy()
        if grp.empty:
            continue
        for yyyymm, gmonth in grp.groupby("yyyymm"):
            year = yyyymm[:4]
            month = yyyymm[4:6]
            month_dir = base_dir / category / yyyymm
            month_files = sorted(month_dir.glob("*.csv"))
            month_zip = month_dir / f"{category}_{yyyymm}.zip"
            make_zip_from_files(month_zip, month_files, base_dir / category)

            # Year zip
            year_dir = base_dir / category
            year_pattern_files = sorted(year_dir.glob(f"{year}*/*.csv"))
            year_zip = year_dir / f"{category}_{year}.zip"
            make_zip_from_files(year_zip, year_pattern_files, year_dir)

            month_node = {
                "year": year,
                "month": month,
                "month_label": f"{year}-{month}",
                "download_all": f"./data/downloads/raw/{category}/{yyyymm}/{category}_{yyyymm}.zip",
                "days": [],
            }
            for _, row in gmonth.sort_values(["date", "universe"], ascending=[False, True]).iterrows():
                month_node["days"].append(
                    {
                        "date": row["date"],
                        "universe": row["universe"],
                        "n_firms": safe_int(row["n_firms"]),
                        "download_path": row["download_path"],
                    }
                )

            # Append into year node
            year_nodes = {node["year"]: node for node in catalog[category]}
            if year not in year_nodes:
                node = {
                    "year": year,
                    "download_all": f"./data/downloads/raw/{category}/{category}_{year}.zip",
                    "months": [],
                }
                catalog[category].append(node)
                year_nodes[year] = node
            year_nodes[year]["months"].append(month_node)

        catalog[category] = sorted(catalog[category], key=lambda x: x["year"], reverse=True)
        for ynode in catalog[category]:
            ynode["months"] = sorted(ynode["months"], key=lambda x: x["month"], reverse=True)
    return catalog


def build_family_download_tree(slug: str, daily: pd.DataFrame) -> dict:
    out_dir = FAMILY_DOWNLOAD_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = {"years": []}
    if daily is None or daily.empty or "date" not in daily.columns:
        return catalog

    tmp = daily.copy()
    if tmp.columns.duplicated().any():
        tmp = tmp.loc[:, ~tmp.columns.duplicated()].copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp[tmp["date"].notna()].copy()
    if tmp.empty:
        return catalog

    tmp["yyyymm"] = tmp["date"].dt.strftime("%Y%m")
    tmp["year"] = tmp["date"].dt.strftime("%Y")

    year_map = {}
    for year, gy in tmp.groupby("year"):
        year_dir = out_dir / year
        year_dir.mkdir(parents=True, exist_ok=True)
        year_csv = year_dir / f"{slug}_{year}.csv"
        gy.to_csv(year_csv, index=False)

        months = []
        for yyyymm, gm in gy.groupby("yyyymm"):
            month = yyyymm[4:6]
            month_dir = year_dir / month
            month_dir.mkdir(parents=True, exist_ok=True)
            month_csv = month_dir / f"{slug}_{yyyymm}.csv"
            gm.to_csv(month_csv, index=False)

            day_nodes = []
            for dt, gd in gm.groupby(gm["date"].dt.strftime("%Y-%m-%d")):
                day_csv = month_dir / f"{slug}_{dt.replace('-', '')}.csv"
                gd.to_csv(day_csv, index=False)
                day_nodes.append(
                    {
                        "date": dt,
                        "download_path": f"./data/downloads/families/{slug}/{year}/{month}/{day_csv.name}",
                    }
                )

            month_zip = month_dir / f"{slug}_{yyyymm}.zip"
            make_zip_from_files(month_zip, sorted(month_dir.glob("*.csv")), month_dir)
            months.append(
                {
                    "month": month,
                    "month_label": f"{year}-{month}",
                    "download_all": f"./data/downloads/families/{slug}/{year}/{month}/{month_zip.name}",
                    "days": sorted(day_nodes, key=lambda x: x["date"], reverse=True),
                }
            )

        year_zip = year_dir / f"{slug}_{year}.zip"
        files = [p for p in year_dir.rglob("*.csv")]
        make_zip_from_files(year_zip, files, year_dir)
        year_map[year] = {
            "year": year,
            "download_all": f"./data/downloads/families/{slug}/{year}/{year_zip.name}",
            "months": sorted(months, key=lambda x: x["month"], reverse=True),
        }

    catalog["years"] = sorted(year_map.values(), key=lambda x: x["year"], reverse=True)
    return catalog


def write_family_outputs(slug: str, latest_df: pd.DataFrame, daily_df: pd.DataFrame, monthly_df: pd.DataFrame):
    latest_csv = FAMILY_DOWNLOAD_DIR / f"{slug}_latest.csv"
    daily_csv = FAMILY_DOWNLOAD_DIR / f"{slug}_daily_history.csv"
    monthly_csv = FAMILY_DOWNLOAD_DIR / f"{slug}_monthly_history.csv"

    latest_df.to_csv(latest_csv, index=False)
    daily_df.to_csv(daily_csv, index=False)
    monthly_df.to_csv(monthly_csv, index=False)

    payload = {
        "slug": slug,
        "latest": latest_df.to_dict(orient="records"),
        "daily": daily_df.to_dict(orient="records"),
        "monthly": monthly_df.to_dict(orient="records"),
        "downloads": {
            "latest_csv": f"./data/downloads/families/{latest_csv.name}",
            "daily_csv": f"./data/downloads/families/{daily_csv.name}",
            "monthly_csv": f"./data/downloads/families/{monthly_csv.name}",
        },
        "tree": build_family_download_tree(slug, daily_df),
    }
    write_json(DOCS_DATA_DIR / f"{slug}.json", payload)


def write_compatibility_outputs(
    marketwide_latest: pd.DataFrame,
    value_latest: pd.DataFrame,
    industry_latest: pd.DataFrame,
    etf_latest: pd.DataFrame,
    country_latest: pd.DataFrame,
    indices_latest: pd.DataFrame,
    marketwide_daily: pd.DataFrame,
    value_daily: pd.DataFrame,
    industry_daily: pd.DataFrame,
    etf_daily: pd.DataFrame,
    country_daily: pd.DataFrame,
    indices_daily: pd.DataFrame,
):
    overview = {
        "title": "Live ICC data library",
        "cards": {
            "all_market_vw_icc": safe_float(marketwide_latest.loc[marketwide_latest["family"] == "all_market", "value"].iloc[0]) if not marketwide_latest[marketwide_latest["family"] == "all_market"].empty else None,
            "sp500_vw_icc": safe_float(marketwide_latest.loc[marketwide_latest["family"] == "sp500", "value"].iloc[0]) if not marketwide_latest[marketwide_latest["family"] == "sp500"].empty else None,
            "value_icc": safe_float(value_latest.loc[value_latest["family"] == "value", "value"].iloc[0]) if not value_latest[value_latest["family"] == "value"].empty else None,
            "growth_icc": safe_float(value_latest.loc[value_latest["family"] == "growth", "value"].iloc[0]) if not value_latest[value_latest["family"] == "growth"].empty else None,
            "ivp_bm": safe_float(value_latest.loc[value_latest["family"] == "ivp_bm", "value"].iloc[0]) if not value_latest[value_latest["family"] == "ivp_bm"].empty else None,
        },
        "downloads": [
            {"label": "Marketwide daily history", "path": "./data/downloads/families/marketwide_daily_history.csv"},
            {"label": "Value daily history", "path": "./data/downloads/families/value_daily_history.csv"},
            {"label": "Industry daily history", "path": "./data/downloads/families/industry_daily_history.csv"},
            {"label": "ETF daily history", "path": "./data/downloads/families/etf_daily_history.csv"},
            {"label": "Country daily history", "path": "./data/downloads/families/country_daily_history.csv"},
            {"label": "Indices daily history", "path": "./data/downloads/families/indices_daily_history.csv"},
        ],
    }
    write_json(DOCS_DATA_DIR / "overview.json", overview)

    def compat_family_json(path_name: str, latest_df: pd.DataFrame, daily_df: pd.DataFrame):
        write_json(DOCS_DATA_DIR / path_name, {"latest": latest_df.to_dict(orient="records"), "history": daily_df.to_dict(orient="records")})

    compat_family_json("market_icc.json", marketwide_latest, marketwide_daily)
    compat_family_json("value_icc_bm.json", value_latest, value_daily)
    compat_family_json("industry_icc.json", industry_latest, industry_daily)
    compat_family_json("etf_icc.json", etf_latest, etf_daily)
    compat_family_json("country_icc.json", country_latest, country_daily)
    compat_family_json("index_icc.json", indices_latest, indices_daily)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    FAMILY_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    base_paths = find_all_snapshots(args.universe)
    if not base_paths:
        raise FileNotFoundError(f"No {args.universe} snapshots found")

    base_valid = dedup_valid_snapshots(get_valid_snapshots(base_paths))
    if not base_valid:
        raise RuntimeError(f"No valid {args.universe} snapshots found")

    all_paths = find_all_snapshots(None)
    all_valid = dedup_valid_snapshots(get_valid_snapshots(all_paths))

    all_valid_by_universe = defaultdict(list)
    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        all_valid_by_universe[meta["universe"]].append((path, df))

    latest_usall_path, latest_usall = base_valid[-1]
    asof_date = latest_usall["date"].dropna().iloc[0].strftime("%Y-%m-%d")

    marketwide_latest, marketwide_daily, marketwide_monthly = build_marketwide_outputs(all_valid_by_universe)
    value_latest, value_daily, value_monthly = build_value_outputs(base_valid)
    industry_latest, industry_daily, industry_monthly = build_industry_outputs(base_valid)
    indices_latest, indices_daily, indices_monthly = build_index_outputs(all_valid_by_universe)
    etf_latest, etf_daily, etf_monthly = build_online_family("etfs.csv", latest_usall, "etf_history.csv")
    country_latest, country_daily, country_monthly = build_online_family("country_etfs.csv", latest_usall, "country_history.csv")

    raw_rows = copy_raw_snapshots(all_valid)
    raw_catalog = build_raw_download_catalog(raw_rows)

    write_family_outputs("marketwide", marketwide_latest, marketwide_daily, marketwide_monthly)
    write_family_outputs("value", value_latest, value_daily, value_monthly)
    write_family_outputs("industry", industry_latest, industry_daily, industry_monthly)
    write_family_outputs("indices", indices_latest, indices_daily, indices_monthly)
    write_family_outputs("etf", etf_latest, etf_daily, etf_monthly)
    write_family_outputs("country", country_latest, country_daily, country_monthly)

    write_json(
        DOCS_DATA_DIR / "downloads_catalog.json",
        {
            "title": "Live ICC data library",
            "asof_date": asof_date,
            "raw_snapshots": raw_catalog,
            "families": {
                "marketwide": "./data/marketwide.json",
                "value": "./data/value.json",
                "industry": "./data/industry.json",
                "indices": "./data/indices.json",
                "etf": "./data/etf.json",
                "country": "./data/country.json",
            },
            "source_file": str(latest_usall_path.relative_to(REPO)),
        },
    )

    write_compatibility_outputs(
        marketwide_latest, value_latest, industry_latest, etf_latest, country_latest, indices_latest,
        marketwide_daily, value_daily, industry_daily, etf_daily, country_daily, indices_daily
    )

    print(f"[build_docs_data] asof_date = {asof_date}")
    print(f"[build_docs_data] valid snapshots = {len(all_valid)}")
    print("[build_docs_data] wrote docs/data/*.json and downloads")


if __name__ == "__main__":
    main()
