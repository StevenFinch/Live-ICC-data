from __future__ import annotations

import argparse
import io
import json
import math
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
    keep: dict[tuple[str, str], tuple[Path, pd.DataFrame, SnapshotMeta]] = {}
    for path, df, meta in valid:
        dt = str(df["date"].iloc[0])
        keep[(meta.universe, dt)] = (path, df, meta)

    out = list(keep.values())
    out.sort(key=lambda x: (x[2].universe, x[1]["date"].iloc[0], x[2].rerun, x[0].name))
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

    actual_group_cols = [c for c in group_cols if c in tmp.columns]
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    sort_cols = actual_group_cols + ["date"] if actual_group_cols else ["date"]
    tmp = tmp.sort_values(sort_cols).copy()
    dedup_cols = actual_group_cols + ["month"] if actual_group_cols else ["month"]
    tmp = tmp.drop_duplicates(subset=dedup_cols, keep="last").copy()

    wanted_cols = [c for c in actual_group_cols + ["date", "month"] + keep_cols if c in tmp.columns]
    wanted_cols = list(dict.fromkeys(wanted_cols))
    return tmp[wanted_cols].reset_index(drop=True)


def keep_last_n_months(df: pd.DataFrame, n: int, by_col: str | None = None) -> pd.DataFrame:
    if df is None or df.empty or "month" not in df.columns:
        return df.copy()
    tmp = df.copy()
    if by_col is None or by_col not in tmp.columns:
        months = sorted(tmp["month"].dropna().unique())[-n:]
        return tmp[tmp["month"].isin(months)].copy()
    parts = []
    for _, g in tmp.groupby(by_col, dropna=False):
        months = sorted(g["month"].dropna().unique())[-n:]
        parts.append(g[g["month"].isin(months)].copy())
    return pd.concat(parts, ignore_index=True) if parts else tmp.iloc[0:0].copy()


def to_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return json_safe(df.to_dict(orient="records"))


def read_csv_config(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=expected_cols)
    df = pd.read_csv(path)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols].copy()


# -----------------------------
# ZIP helpers
# -----------------------------

def make_zip(zip_path: Path, members: Iterable[tuple[Path, str]]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for real_path, arcname in members:
            if real_path.exists() and real_path.is_file():
                zf.write(real_path, arcname)


# -----------------------------
# Family builders based on firm ICC snapshots
# -----------------------------

def build_marketwide(usall_valid, sp500_valid) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for family_name, snapshots in [("all_market", usall_valid), ("sp500", sp500_valid)]:
        for path, df, _meta in snapshots:
            rows.append(
                {
                    "date": str(df["date"].iloc[0]),
                    "family": family_name,
                    "value": weighted_mean(df["ICC"], df["mktcap"]),
                    "method": "ICC calculation",
                    "n_items": int(len(df)),
                    "source_file": str(path.relative_to(REPO)),
                }
            )
    daily = pd.DataFrame(rows)
    if daily.empty:
        empty = pd.DataFrame(columns=["date", "family", "value", "method", "n_items", "source_file"])
        return empty, empty, empty
    daily = daily.sort_values(["family", "date"]).reset_index(drop=True)
    latest = daily.sort_values(["family", "date"]).drop_duplicates(["family"], keep="last")
    monthly = latest_per_month(daily, ["family"], ["value", "method", "n_items", "source_file"])
    return latest.reset_index(drop=True), daily, monthly


def build_value(usall_valid) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for path, raw_df, _meta in usall_valid:
        df = raw_df.copy()
        df = df[
            df["date"].notna()
            & df["bm"].notna()
            & np.isfinite(df["bm"])
            & (df["bm"] > 0)
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
        df["portfolio"] = df["size_bucket"] + "/" + df["bm_bucket"].astype(str)
        pt = (
            df.groupby("portfolio", dropna=False)
            .apply(lambda g: pd.Series({"vw_icc": weighted_mean(g["ICC"], g["mktcap"])}))
            .reset_index()
        )
        d = dict(zip(pt["portfolio"], pt["vw_icc"]))
        sl = d.get("S/L", np.nan)
        bl = d.get("B/L", np.nan)
        sh = d.get("S/H", np.nan)
        bh = d.get("B/H", np.nan)
        growth = np.nan if (pd.isna(sl) or pd.isna(bl)) else float((sl + bl) / 2.0)
        value = np.nan if (pd.isna(sh) or pd.isna(bh)) else float((sh + bh) / 2.0)
        ivp = np.nan if (pd.isna(value) or pd.isna(growth)) else float(value - growth)
        date_value = str(df["date"].iloc[0])
        rows.extend(
            [
                {"date": date_value, "family": "value_icc", "value": value, "method": "ICC calculation", "source_file": str(path.relative_to(REPO))},
                {"date": date_value, "family": "growth_icc", "value": growth, "method": "ICC calculation", "source_file": str(path.relative_to(REPO))},
                {"date": date_value, "family": "ivp_bm", "value": ivp, "method": "ICC calculation", "source_file": str(path.relative_to(REPO))},
            ]
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        empty = pd.DataFrame(columns=["date", "family", "value", "method", "source_file"])
        return empty, empty, empty
    daily = daily.sort_values(["family", "date"]).reset_index(drop=True)
    latest = daily.sort_values(["family", "date"]).drop_duplicates(["family"], keep="last")
    monthly = latest_per_month(daily, ["family"], ["value", "method", "source_file"])
    return latest.reset_index(drop=True), daily, monthly


def build_industry(usall_valid) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for path, raw_df, _meta in usall_valid:
        df = raw_df.copy()
        df = df[df["sector"].notna()].copy()
        if df.empty:
            continue
        grouped = (
            df.groupby("sector", dropna=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "value": weighted_mean(g["ICC"], g["mktcap"]),
                        "n_items": int(len(g)),
                    }
                )
            )
            .reset_index()
        )
        date_value = str(df["date"].iloc[0])
        for _, r in grouped.iterrows():
            rows.append(
                {
                    "date": date_value,
                    "family": str(r["sector"]),
                    "label": str(r["sector"]),
                    "value": r["value"],
                    "method": "ICC calculation",
                    "n_items": int(r["n_items"]),
                    "source_file": str(path.relative_to(REPO)),
                }
            )
    daily = pd.DataFrame(rows)
    if daily.empty:
        empty = pd.DataFrame(columns=["date", "family", "label", "value", "method", "n_items", "source_file"])
        return empty, empty, empty
    daily = daily.sort_values(["family", "date"]).reset_index(drop=True)
    latest = daily.sort_values(["family", "date"]).drop_duplicates(["family"], keep="last")
    monthly = latest_per_month(daily, ["family"], ["label", "value", "method", "n_items", "source_file"])
    return latest.reset_index(drop=True), daily, monthly


def build_indices(all_valid) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for path, df, meta in all_valid:
        if meta.universe not in INDEX_UNIVERSES:
            continue
        rows.append(
            {
                "date": str(df["date"].iloc[0]),
                "family": meta.universe,
                "value": weighted_mean(df["ICC"], df["mktcap"]),
                "method": "ICC calculation",
                "n_items": int(len(df)),
                "source_file": str(path.relative_to(REPO)),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        empty = pd.DataFrame(columns=["date", "family", "value", "method", "n_items", "source_file"])
        return empty, empty, empty
    daily = daily.sort_values(["family", "date"]).reset_index(drop=True)
    latest = daily.sort_values(["family", "date"]).drop_duplicates(["family"], keep="last")
    monthly = latest_per_month(daily, ["family"], ["value", "method", "n_items", "source_file"])
    return latest.reset_index(drop=True), daily, monthly


# -----------------------------
# Online ETF/Country family helpers
# -----------------------------

def fetch_page_text(url: str) -> str:
    if not url:
        raise ValueError("Empty URL")
    s = requests_session()
    resp = s.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_page_pe(url: str) -> float | None:
    text = fetch_page_text(url)
    for pat in PAGE_PROXY_PATTERNS:
        m = pat.search(text)
        if m:
            pe = safe_float(m.group(1))
            if pe is not None and pe > 0:
                return pe
    return None


def fetch_yfinance_pe(ticker: str) -> float | None:
    if not ticker:
        return None
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    for key in ["forwardPE", "trailingPE", "forward_pe", "trailing_pe"]:
        pe = safe_float(info.get(key))
        if pe is not None and pe > 0:
            return pe
    return None


def normalize_symbols(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
    )


def fetch_yfinance_top_holdings(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    fd = getattr(t, "funds_data", None)
    if fd is None:
        raise ValueError("funds_data unavailable")
    h = getattr(fd, "top_holdings", None)
    if h is None:
        raise ValueError("top_holdings unavailable")
    df = h.copy() if isinstance(h, pd.DataFrame) else pd.DataFrame(h)
    if df.empty:
        raise ValueError("top_holdings empty")
    cols = {str(c).strip().lower(): c for c in df.columns}
    sym_col = None
    wt_col = None
    for cand in ["symbol", "ticker", "holding"]:
        if cand in cols:
            sym_col = cols[cand]
            break
    for cand in ["holdingpercent", "holding_percent", "weight", "percent", "pct"]:
        if cand in cols:
            wt_col = cols[cand]
            break
    if sym_col is None:
        sym_col = df.columns[0]
    if wt_col is None:
        for c in df.columns[::-1]:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() >= max(1, len(df) // 2):
                wt_col = c
                break
    if wt_col is None:
        raise ValueError("weight column not found")
    out = pd.DataFrame(
        {
            "symbol": normalize_symbols(df[sym_col]),
            "raw_weight": pd.to_numeric(df[wt_col], errors="coerce"),
        }
    )
    out = out[out["symbol"].notna() & out["raw_weight"].notna()].copy()
    if out.empty:
        raise ValueError("no valid holdings rows")
    out["weight"] = np.where(out["raw_weight"].abs() > 1.5, out["raw_weight"] / 100.0, out["raw_weight"])
    out = out[out["weight"] > 0].copy()
    if out.empty:
        raise ValueError("no positive weights")
    out["weight"] = out["weight"] / out["weight"].sum()
    return out[["symbol", "weight"]]


def holdings_based_proxy(ticker: str, base_usall: pd.DataFrame) -> dict | None:
    holdings = fetch_yfinance_top_holdings(ticker)
    merged = holdings.merge(
        base_usall[["ticker", "ICC"]].copy(),
        left_on="symbol",
        right_on="ticker",
        how="left",
    )
    matched = merged["ICC"].notna()
    coverage = float(merged.loc[matched, "weight"].sum()) if len(merged) else 0.0
    n_matched = int(matched.sum())
    if n_matched <= 0:
        return None
    value = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"])
    return {
        "value": value,
        "coverage_weight": coverage,
        "n_items": int(len(merged)),
        "n_matched": n_matched,
    }


def build_online_family(config_name: str, latest_usall: pd.DataFrame, history_name: str):
    """Build ETF/Country family outputs from online data and append archive history."""
    config_path = CONFIG_DIR / config_name

    if "country" in config_name.lower():
        expected_cols = ["family", "ticker", "label", "url", "allow_holdings"]
        fallback_family_col = "family"
    else:
        expected_cols = ["ticker", "label", "category", "url", "allow_holdings"]
        fallback_family_col = "ticker"

    cfg = read_csv_config(config_path, expected_cols)
    date_value = str(latest_usall["date"].dropna().iloc[0])
    base_usall = clean_snapshot(latest_usall).copy()

    rows = []
    for _, r in cfg.iterrows():
        ticker = str(r.get("ticker", "") or "").strip().upper()
        family = str(r.get("family", "") or "").strip() or ticker
        label = str(r.get("label", "") or "").strip() or family
        category = str(r.get("category", "") or "").strip()
        url = str(r.get("url", "") or "").strip()
        allow_holdings = int(pd.to_numeric(r.get("allow_holdings", 1), errors="coerce") or 0)

        result = None

        # Step 1: try holdings-based aggregation.
        if allow_holdings == 1 and ticker:
            try:
                result = holdings_based_proxy(ticker, base_usall)
                if result is not None:
                    coverage = float(result.get("coverage_weight", 0.0) or 0.0)
                    n_matched = int(result.get("n_matched", 0) or 0)
                    if coverage >= 0.20 and n_matched >= 3:
                        result["method"] = "ICC calculation"
                        result["status"] = "ICC calculation"
                    else:
                        result = None
            except Exception:
                result = None

        # Step 2: try page-based P/E estimate.
        if result is None and url:
            try:
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
            except Exception:
                result = None

        # Step 3: try yfinance-based P/E estimate.
        if result is None and ticker:
            try:
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
            except Exception:
                result = None

        if result is None:
            result = {
                "value": None,
                "method": "Unavailable",
                "coverage_weight": None,
                "n_items": None,
                "n_matched": None,
                "status": "Unavailable",
            }

        rows.append(
            {
                "date": date_value,
                "family": family,
                "ticker": ticker,
                "label": label,
                "category": category,
                "url": url,
                "allow_holdings": allow_holdings,
                "value": result.get("value"),
                "method": result.get("method"),
                "coverage_weight": result.get("coverage_weight"),
                "n_items": result.get("n_items"),
                "n_matched": result.get("n_matched"),
                "status": result.get("status"),
            }
        )

    latest = pd.DataFrame(rows)
    history_path = DOWNLOAD_DIR / history_name
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

    # Append the latest archive and deduplicate by family/date.
    daily = pd.concat([old, latest], ignore_index=True)
    if daily.columns.duplicated().any():
        daily = daily.loc[:, ~daily.columns.duplicated()].copy()
    daily = daily.drop_duplicates(subset=["date", "family"], keep="last").copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    daily = daily.sort_values(["family", "date"]).reset_index(drop=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(history_path, index=False)

    monthly = latest_per_month(daily, ["family"], ["ticker", "label", "category", "value", "method", "coverage_weight", "n_items", "n_matched", "status"])
    return latest, daily, monthly


# -----------------------------
# Download catalog helpers
# -----------------------------

def save_table(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return f"./data/downloads/{path.name}" if path.parent == DOWNLOAD_DIR else f"./data/downloads/{path.relative_to(DOWNLOAD_DIR).as_posix()}"


def build_family_tree(daily: pd.DataFrame, family_col: str, root_subdir: str) -> list[dict]:
    if daily is None or daily.empty:
        return []
    tmp = daily.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp[tmp["date"].notna()].copy()
    if tmp.empty:
        return []

    out = []
    for family, g in tmp.groupby(family_col, dropna=False):
        fam_key = str(family)
        rows_year = []
        for year, gy in g.groupby(g["date"].dt.strftime("%Y")):
            month_rows = []
            year_members = []
            for month, gm in gy.groupby(g["date"].dt.strftime("%m")):
                day_files = []
                month_members = []
                subdir = DOWNLOAD_DIR / root_subdir / fam_key / year / month
                subdir.mkdir(parents=True, exist_ok=True)
                for _, r in gm.sort_values("date", ascending=False).iterrows():
                    ds = r["date"].strftime("%Y-%m-%d")
                    day_path = subdir / f"{ds}.csv"
                    pd.DataFrame([r]).to_csv(day_path, index=False)
                    rel = f"./data/downloads/{day_path.relative_to(DOWNLOAD_DIR).as_posix()}"
                    day_files.append({"date": ds, "path": rel})
                    month_members.append((day_path, f"{ds}.csv"))
                    year_members.append((day_path, f"{month}/{ds}.csv"))
                month_zip = subdir / f"{year}-{month}.zip"
                make_zip(month_zip, month_members)
                month_rows.append(
                    {
                        "month": month,
                        "download_all_zip": f"./data/downloads/{month_zip.relative_to(DOWNLOAD_DIR).as_posix()}",
                        "days": day_files,
                    }
                )
            year_dir = DOWNLOAD_DIR / root_subdir / fam_key / year
            year_dir.mkdir(parents=True, exist_ok=True)
            year_zip = year_dir / f"{year}.zip"
            make_zip(year_zip, year_members)
            rows_year.append(
                {
                    "year": year,
                    "download_all_zip": f"./data/downloads/{year_zip.relative_to(DOWNLOAD_DIR).as_posix()}",
                    "months": month_rows,
                }
            )
        out.append({"family": fam_key, "years": sorted(rows_year, key=lambda x: x["year"], reverse=True)})
    return out


def build_raw_snapshot_tree(all_valid: list[tuple[Path, pd.DataFrame, SnapshotMeta]]) -> dict:
    def classify(universe: str) -> str:
        if universe == "usall":
            return "usall"
        if universe == "sp500":
            return "sp500"
        return "other_indices"

    groups = {"usall": [], "sp500": [], "other_indices": []}
    for path, df, meta in all_valid:
        groups[classify(meta.universe)].append((path, df, meta))

    tabs = []
    for tab_key, rows in groups.items():
        years_payload = []
        by_year: dict[str, list[tuple[Path, pd.DataFrame, SnapshotMeta]]] = {}
        for row in rows:
            by_year.setdefault(row[2].year, []).append(row)
        for year in sorted(by_year.keys(), reverse=True):
            months_payload = []
            year_members = []
            by_month: dict[str, list[tuple[Path, pd.DataFrame, SnapshotMeta]]] = {}
            for row in by_year[year]:
                by_month.setdefault(row[2].yyyymm, []).append(row)
            for yyyymm in sorted(by_month.keys(), reverse=True):
                month = yyyymm[-2:]
                day_rows = []
                month_members = []
                out_dir = DOWNLOAD_DIR / "raw" / tab_key / year / month
                out_dir.mkdir(parents=True, exist_ok=True)
                for path, df, meta in sorted(by_month[yyyymm], key=lambda x: x[2].yyyymmdd, reverse=True):
                    target = out_dir / path.name
                    if not target.exists():
                        shutil.copy2(path, target)
                    rel = f"./data/downloads/{target.relative_to(DOWNLOAD_DIR).as_posix()}"
                    day_rows.append({
                        "date": str(df["date"].iloc[0]),
                        "universe": meta.universe,
                        "path": rel,
                        "n_items": int(len(df)),
                    })
                    month_members.append((target, target.name))
                    year_members.append((target, f"{month}/{target.name}"))
                month_zip = out_dir / f"{yyyymm}.zip"
                make_zip(month_zip, month_members)
                months_payload.append(
                    {
                        "month": month,
                        "yyyymm": yyyymm,
                        "download_all_zip": f"./data/downloads/{month_zip.relative_to(DOWNLOAD_DIR).as_posix()}",
                        "days": day_rows,
                    }
                )
            year_dir = DOWNLOAD_DIR / "raw" / tab_key / year
            year_dir.mkdir(parents=True, exist_ok=True)
            year_zip = year_dir / f"{year}.zip"
            make_zip(year_zip, year_members)
            years_payload.append(
                {
                    "year": year,
                    "download_all_zip": f"./data/downloads/{year_zip.relative_to(DOWNLOAD_DIR).as_posix()}",
                    "months": months_payload,
                }
            )
        tabs.append({"tab": tab_key, "years": years_payload})
    return {"tabs": tabs}


def family_download_spec(family_key: str, latest: pd.DataFrame, daily: pd.DataFrame, monthly: pd.DataFrame) -> FamilyDownloadSpec:
    title = FAMILY_LABELS[family_key]
    latest_path = save_table(latest, DOWNLOAD_DIR / f"{family_key}_latest.csv")
    daily_path = save_table(daily, DOWNLOAD_DIR / f"{family_key}_daily_history.csv")
    monthly_path = save_table(monthly, DOWNLOAD_DIR / f"{family_key}_monthly_history.csv")
    return FamilyDownloadSpec(
        family_key=family_key,
        title=title,
        latest_path=latest_path,
        daily_history_path=daily_path,
        monthly_history_path=monthly_path,
        latest_table=to_records(latest),
        daily_history=to_records(daily),
        monthly_history=to_records(monthly),
    )


def build_overview_payload(asof_date: str, specs: dict[str, FamilyDownloadSpec]) -> dict:
    rows = []
    for key in ["marketwide", "value", "industry", "etf", "country", "indices"]:
        spec = specs.get(key)
        if spec is None:
            continue
        latest_df = pd.DataFrame(spec.latest_table)
        monthly_df = pd.DataFrame(spec.monthly_history)
        latest_value = None
        latest_method = None
        if not latest_df.empty:
            # Use the first row as the homepage summary row.
            latest_value = safe_float(latest_df.iloc[0].get("value"))
            latest_method = latest_df.iloc[0].get("method")
        last_three = keep_last_n_months(monthly_df, 3, by_col=None)
        last_three = last_three.sort_values("date", ascending=False) if not last_three.empty and "date" in last_three.columns else last_three
        rows.append(
            {
                "family": key,
                "title": spec.title,
                "latest_daily": latest_value,
                "latest_method": latest_method,
                "last_three_monthly": to_records(last_three),
                "latest_csv": spec.latest_path,
                "daily_history_csv": spec.daily_history_path,
                "monthly_history_csv": spec.monthly_history_path,
            }
        )
    return {"asof_date": asof_date, "title": "Live ICC data library", "rows": rows}


def marketwide_payload(asof_date: str, spec: FamilyDownloadSpec) -> dict:
    daily = pd.DataFrame(spec.daily_history)
    monthly = pd.DataFrame(spec.monthly_history)
    all_monthly = keep_last_n_months(monthly[monthly["family"] == "all_market"], 3)
    sp500_monthly = keep_last_n_months(monthly[monthly["family"] == "sp500"], 3)
    return {
        "asof_date": asof_date,
        "title": spec.title,
        "latest": spec.latest_table,
        "daily": spec.daily_history,
        "monthly": spec.monthly_history,
        "monthly_last_three": {
            "all_market": to_records(all_monthly.sort_values("date", ascending=False)),
            "sp500": to_records(sp500_monthly.sort_values("date", ascending=False)),
        },
        "downloads": {
            "latest_csv": spec.latest_path,
            "daily_history_csv": spec.daily_history_path,
            "monthly_history_csv": spec.monthly_history_path,
        },
    }


def generic_payload(asof_date: str, spec: FamilyDownloadSpec, family_col: str = "family") -> dict:
    monthly_df = pd.DataFrame(spec.monthly_history)
    last_three = keep_last_n_months(monthly_df, 3, by_col=family_col)
    return {
        "asof_date": asof_date,
        "title": spec.title,
        "latest": spec.latest_table,
        "daily": spec.daily_history,
        "monthly": spec.monthly_history,
        "monthly_last_three": to_records(last_three.sort_values([family_col, "date"], ascending=[True, False]) if not last_three.empty and family_col in last_three.columns else last_three),
        "downloads": {
            "latest_csv": spec.latest_path,
            "daily_history_csv": spec.daily_history_path,
            "monthly_history_csv": spec.monthly_history_path,
        },
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    all_valid = get_valid_snapshots(None)
    usall_valid = [x for x in all_valid if x[2].universe == args.universe]
    sp500_valid = [x for x in all_valid if x[2].universe == "sp500"]

    if not usall_valid:
        raise RuntimeError(f"No valid {args.universe} snapshots found")

    latest_usall = usall_valid[-1][1]
    asof_date = str(latest_usall["date"].iloc[0])

    market_latest, market_daily, market_monthly = build_marketwide(usall_valid, sp500_valid)
    value_latest, value_daily, value_monthly = build_value(usall_valid)
    industry_latest, industry_daily, industry_monthly = build_industry(usall_valid)
    indices_latest, indices_daily, indices_monthly = build_indices(all_valid)
    etf_latest, etf_daily, etf_monthly = build_online_family("etfs.csv", latest_usall, "etf_history.csv")
    country_latest, country_daily, country_monthly = build_online_family("country_etfs.csv", latest_usall, "country_history.csv")

    specs = {
        "marketwide": family_download_spec("marketwide", market_latest, market_daily, market_monthly),
        "value": family_download_spec("value", value_latest, value_daily, value_monthly),
        "industry": family_download_spec("industry", industry_latest, industry_daily, industry_monthly),
        "etf": family_download_spec("etf", etf_latest, etf_daily, etf_monthly),
        "country": family_download_spec("country", country_latest, country_daily, country_monthly),
        "indices": family_download_spec("indices", indices_latest, indices_daily, indices_monthly),
    }

    # Build time-tree downloads for each family.
    family_trees = {
        "marketwide": build_family_tree(market_daily, "family", "trees/marketwide"),
        "value": build_family_tree(value_daily, "family", "trees/value"),
        "industry": build_family_tree(industry_daily, "family", "trees/industry"),
        "etf": build_family_tree(etf_daily, "family", "trees/etf"),
        "country": build_family_tree(country_daily, "family", "trees/country"),
        "indices": build_family_tree(indices_daily, "family", "trees/indices"),
    }

    downloads_catalog = {
        "asof_date": asof_date,
        "title": "Live ICC data library",
        "families": [
            {
                "key": spec.family_key,
                "title": spec.title,
                "latest_csv": spec.latest_path,
                "daily_history_csv": spec.daily_history_path,
                "monthly_history_csv": spec.monthly_history_path,
                "time_tree": family_trees.get(spec.family_key, []),
            }
            for spec in specs.values()
        ],
        "raw_snapshots": build_raw_snapshot_tree(all_valid),
    }

    # Write JSON payloads consumed by the front-end.
    write_json(DOCS_DATA_DIR / "overview.json", build_overview_payload(asof_date, specs))
    write_json(DOCS_DATA_DIR / "marketwide.json", marketwide_payload(asof_date, specs["marketwide"]))
    write_json(DOCS_DATA_DIR / "value.json", generic_payload(asof_date, specs["value"]))
    write_json(DOCS_DATA_DIR / "industry.json", generic_payload(asof_date, specs["industry"]))
    write_json(DOCS_DATA_DIR / "etf.json", generic_payload(asof_date, specs["etf"]))
    write_json(DOCS_DATA_DIR / "country.json", generic_payload(asof_date, specs["country"]))
    write_json(DOCS_DATA_DIR / "indices.json", generic_payload(asof_date, specs["indices"]))
    write_json(DOCS_DATA_DIR / "downloads_catalog.json", downloads_catalog)

    print(f"[build_docs_data] asof_date={asof_date}")
    print(f"[build_docs_data] marketwide_daily={len(market_daily)} value_daily={len(value_daily)} industry_daily={len(industry_daily)}")
    print(f"[build_docs_data] etf_daily={len(etf_daily)} country_daily={len(country_daily)} indices_daily={len(indices_daily)}")
    print("[build_docs_data] wrote docs/data/*.json and docs/data/downloads/*")


if __name__ == "__main__":
    main()
