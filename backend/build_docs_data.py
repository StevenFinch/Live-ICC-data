from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from collections import defaultdict
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
RAW_ZIP_DIR = DOWNLOAD_DIR / "raw_zips"
CONFIG_DIR = REPO / "config"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

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


def parse_snapshot_meta(path: Path) -> dict[str, Any] | None:
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


def safe_float(x: Any) -> float | None:
    if pd.isna(x):
        return None
    try:
        val = float(x)
        if not np.isfinite(val):
            return None
        return val
    except Exception:
        return None


def safe_int(x: Any) -> int | None:
    if pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        return None


def json_safe(obj: Any) -> Any:
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


def read_csv_config(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=expected_cols)

    df = pd.read_csv(path)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols].copy()


def bucket_raw_universe(universe: str) -> str:
    if universe == "usall":
        return "usall"
    if universe == "sp500":
        return "sp500"
    if universe in INDEX_UNIVERSES:
        return "other_indices"
    return "other"


def find_all_snapshots(universe: str | None = None) -> list[Path]:
    rows = []
    for p in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(p)
        if meta is None:
            continue
        if universe is not None and meta["universe"] != universe:
            continue
        rows.append(((meta["yyyymmdd"], meta["rerun"], p.name), p))
    rows.sort(key=lambda x: x[0])
    return [p for _, p in rows]


def load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")

    if path.stat().st_size == 0:
        raise EmptyDataError("No columns to parse from file")

    df = pd.read_csv(path)
    if df is None or len(df.columns) == 0:
        raise EmptyDataError("No columns to parse from file")

    df.columns = [str(c).strip() for c in df.columns]
    req = {"ticker", "mktcap", "ICC", "date"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    df["ICC"] = pd.to_numeric(df["ICC"], errors="coerce")
    if "bm" in df.columns:
        df["bm"] = pd.to_numeric(df["bm"], errors="coerce")
    else:
        df["bm"] = np.nan
    if "sector" not in df.columns:
        df["sector"] = None
    if "name" not in df.columns:
        df["name"] = None
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def try_load_snapshot(path: Path) -> pd.DataFrame | None:
    try:
        return load_snapshot(path)
    except Exception as e:
        print(f"[build_docs_data] skip invalid snapshot {path}: {type(e).__name__}: {e}")
        return None


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
        df = try_load_snapshot(path)
        if df is None:
            continue
        df = clean_df(df)
        if df.empty:
            continue
        out.append((path, df))
    return out


def dedup_valid_snapshots(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> list[tuple[Path, pd.DataFrame]]:
    keep: dict[tuple[str, str], tuple[Path, pd.DataFrame]] = {}
    for path, df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        date_value = str(df["date"].dropna().iloc[0]) if df["date"].notna().any() else None
        if not date_value:
            continue
        keep[(meta["universe"], date_value)] = (path, df)

    rows = list(keep.values())
    rows.sort(
        key=lambda x: (
            parse_snapshot_meta(x[0])["universe"],
            str(x[1]["date"].dropna().iloc[0]),
            parse_snapshot_meta(x[0])["rerun"],
            x[0].name,
        )
    )
    return rows


def latest_per_month(df: pd.DataFrame, group_cols: list[str], keep_cols: list[str]) -> pd.DataFrame:
    """
    Keep the latest observation in each month for each group.

    This function is robust to duplicate column names and missing columns.
    """
    if df is None or df.empty:
        cols = group_cols + ["date", "month"] + keep_cols
        cols = list(dict.fromkeys(cols))
        return pd.DataFrame(columns=cols)

    tmp = df.copy()

    if tmp.columns.duplicated().any():
        tmp = tmp.loc[:, ~tmp.columns.duplicated()].copy()

    if "date" not in tmp.columns:
        cols = group_cols + ["date", "month"] + keep_cols
        cols = list(dict.fromkeys(cols))
        return pd.DataFrame(columns=cols)

    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp[tmp["date"].notna()].copy()
    if tmp.empty:
        cols = group_cols + ["date", "month"] + keep_cols
        cols = list(dict.fromkeys(cols))
        return pd.DataFrame(columns=cols)

    group_cols = [c for c in group_cols if c in tmp.columns]

    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    sort_cols = group_cols + ["date"] if group_cols else ["date"]
    tmp = tmp.sort_values(sort_cols).copy()

    dedup_cols = group_cols + ["month"] if group_cols else ["month"]
    tmp = tmp.drop_duplicates(subset=dedup_cols, keep="last").copy()

    wanted_cols = group_cols + ["date", "month"] + keep_cols
    wanted_cols = [c for c in wanted_cols if c in tmp.columns]
    wanted_cols = list(dict.fromkeys(wanted_cols))
    return tmp[wanted_cols].reset_index(drop=True)


def last_three_monthly_per_family(df: pd.DataFrame, family_col: str = "family") -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []

    out = []
    for fam, g in df.groupby(family_col, dropna=False):
        gg = g.sort_values("date", ascending=False).head(3).copy()
        for _, r in gg.iterrows():
            out.append(
                {
                    family_col: fam,
                    "date": str(r.get("date")),
                    "month": r.get("month"),
                    "value": safe_float(r.get("value")),
                    "method": r.get("method"),
                }
            )
    return out


def build_market_series(valid_snapshots: list[tuple[Path, pd.DataFrame]], family: str) -> pd.DataFrame:
    rows = []
    for path, df in valid_snapshots:
        rows.append(
            {
                "family": family,
                "date": str(df["date"].iloc[0]),
                "value": weighted_mean(df["ICC"], df["mktcap"]),
                "method": "ICC calculation",
                "n_items": int(len(df)),
                "source_file": str(path.relative_to(REPO)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["family", "date", "value", "method", "n_items", "source_file"])
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out = out.drop_duplicates(subset=["family", "date"], keep="last").reset_index(drop=True)
    return out


def build_value_daily(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []

    for path, raw_df in valid_snapshots:
        df = raw_df.copy()
        df = df[
            df["date"].notna()
            & df["mktcap"].notna()
            & df["bm"].notna()
            & df["ICC"].notna()
            & np.isfinite(df["mktcap"])
            & np.isfinite(df["bm"])
            & np.isfinite(df["ICC"])
            & (df["mktcap"] > 0)
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

        icc_sl = d.get("S/L", np.nan)
        icc_bl = d.get("B/L", np.nan)
        icc_sh = d.get("S/H", np.nan)
        icc_bh = d.get("B/H", np.nan)

        growth_icc = np.nan if (pd.isna(icc_sl) or pd.isna(icc_bl)) else float((icc_sl + icc_bl) / 2.0)
        value_icc = np.nan if (pd.isna(icc_sh) or pd.isna(icc_bh)) else float((icc_sh + icc_bh) / 2.0)
        ivp_bm = np.nan if (pd.isna(value_icc) or pd.isna(growth_icc)) else float(value_icc - growth_icc)

        date_value = str(df["date"].iloc[0])
        rows.extend(
            [
                {"family": "value", "date": date_value, "value": value_icc, "method": "ICC calculation"},
                {"family": "growth", "date": date_value, "value": growth_icc, "method": "ICC calculation"},
                {"family": "ivp", "date": date_value, "value": ivp_bm, "method": "ICC calculation"},
            ]
        )

    if not rows:
        return pd.DataFrame(columns=["family", "date", "value", "method"])
    out = pd.DataFrame(rows).sort_values(["family", "date"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["family", "date"], keep="last").reset_index(drop=True)
    return out


def build_industry_daily(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, raw_df in valid_snapshots:
        df = raw_df.copy()
        df = df[df["sector"].notna()].copy()
        if df.empty:
            continue
        date_value = str(df["date"].iloc[0])
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
        for _, r in grouped.iterrows():
            rows.append(
                {
                    "family": str(r["sector"]),
                    "date": date_value,
                    "value": safe_float(r["value"]),
                    "method": "ICC calculation",
                    "n_items": safe_int(r["n_items"]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["family", "date", "value", "method", "n_items"])
    out = pd.DataFrame(rows).sort_values(["family", "date"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["family", "date"], keep="last").reset_index(drop=True)
    return out


def build_indices_daily(all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]]) -> pd.DataFrame:
    rows = []
    for universe in INDEX_UNIVERSES:
        valid = all_valid_by_universe.get(universe, [])
        if not valid:
            continue
        daily = build_market_series(valid, universe)
        rows.extend(daily.to_dict(orient="records"))
    if not rows:
        return pd.DataFrame(columns=["family", "date", "value", "method", "n_items", "source_file"])
    out = pd.DataFrame(rows).sort_values(["family", "date"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["family", "date"], keep="last").reset_index(drop=True)
    return out


def fetch_text(url: str, timeout: int = 25) -> str:
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_page_pe(url: str) -> float | None:
    text = fetch_text(url)
    patterns = [
        r'P/?E(?:\s+Ratio)?[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)',
        r'Price/Earnings[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)',
        r'Forward P/E[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)',
        r'trailing PE[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            pe = float(m.group(1))
            if pe > 0:
                return pe
    return None


def fetch_yfinance_pe(ticker: str) -> float | None:
    tk = yf.Ticker(ticker)
    try:
        info = tk.get_info()
        pe = info.get("forwardPE") or info.get("trailingPE")
        if pe is not None and pe > 0:
            return float(pe)
    except Exception:
        pass
    return None


def get_top_holdings(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    fd = getattr(tk, "funds_data", None)
    if fd is None:
        raise ValueError("funds_data unavailable")
    holdings = getattr(fd, "top_holdings", None)
    if holdings is None:
        raise ValueError("top_holdings unavailable")

    if isinstance(holdings, pd.DataFrame):
        h = holdings.copy()
    else:
        h = pd.DataFrame(holdings)

    if h.empty:
        raise ValueError("top_holdings empty")

    cols = {str(c).lower(): c for c in h.columns}
    symbol_col = None
    weight_col = None

    for cand in ["symbol", "holding", "ticker"]:
        if cand in cols:
            symbol_col = cols[cand]
            break
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
        raise ValueError("weight column not found")

    out = pd.DataFrame(
        {
            "symbol": h[symbol_col].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False),
            "raw_weight": pd.to_numeric(h[weight_col], errors="coerce"),
        }
    )
    out = out[out["symbol"].notna() & out["raw_weight"].notna()].copy()
    out["weight"] = out["raw_weight"] / 100.0 if out["raw_weight"].max() > 1.5 else out["raw_weight"]
    out = out[out["weight"] > 0].copy()
    if out.empty:
        raise ValueError("no positive weights")
    out["weight"] = out["weight"] / out["weight"].sum()
    return out[["symbol", "weight"]]


def holdings_based_proxy(ticker: str, base_usall: pd.DataFrame) -> dict[str, Any] | None:
    h = get_top_holdings(ticker)
    base = base_usall[["ticker", "ICC"]].copy()
    base["ticker"] = base["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)

    merged = h.merge(base, left_on="symbol", right_on="ticker", how="left")
    matched = merged["ICC"].notna()

    if matched.sum() == 0:
        return None

    coverage = float(merged.loc[matched, "weight"].sum())
    value = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"])
    return {
        "value": value,
        "coverage_weight": coverage,
        "n_items": int(len(h)),
        "n_matched": int(matched.sum()),
    }


def sanitize_history_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def build_online_family(config_name: str, latest_usall: pd.DataFrame, history_name: str):
    """
    Build latest / daily / monthly outputs for ETF or Country families using online sources.

    Priority:
    1) holdings-based aggregation if matching coverage is sufficient
    2) page-based P/E estimate
    3) yfinance-based P/E estimate

    Historical series starts from the moment archiving begins.
    """
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

    date_value = str(latest_usall["date"].dropna().iloc[0])
    base_usall = clean_df(latest_usall).copy()

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

        if allow_holdings == 1:
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

        row = {
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
        }
        rows.append(row)

    latest = pd.DataFrame(rows)

    history_path = DOWNLOAD_DIR / history_name
    if history_path.exists():
        old = sanitize_history_df(pd.read_csv(history_path))
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

    key_cols = ["date", "family"] if "family" in latest.columns else ["date", default_family_col]
    if not old.empty:
        key_exist = [c for c in key_cols if c in old.columns]
        if key_exist:
            old = old.drop_duplicates(subset=key_exist, keep="last").copy()

    daily = pd.concat([old, latest], ignore_index=True)
    daily = sanitize_history_df(daily)
    key_exist = [c for c in key_cols if c in daily.columns]
    if key_exist:
        daily = daily.drop_duplicates(subset=key_exist, keep="last").copy()

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sort_cols = [c for c in ["family", "date"] if c in daily.columns]
    if sort_cols:
        daily = daily.sort_values(sort_cols).reset_index(drop=True)

    daily.to_csv(history_path, index=False)

    daily_for_monthly = sanitize_history_df(daily)
    if "family" not in daily_for_monthly.columns:
        if default_family_col in daily_for_monthly.columns:
            daily_for_monthly["family"] = daily_for_monthly[default_family_col]
        else:
            daily_for_monthly["family"] = None

    monthly = latest_per_month(daily_for_monthly, ["family"], ["value", "method"])

    return latest, daily, monthly


def mirror_raw_and_build_zips(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> list[dict[str, Any]]:
    RAW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ZIP_DIR.mkdir(parents=True, exist_ok=True)

    catalog_rows: list[dict[str, Any]] = []
    bucket_year_month_files: dict[tuple[str, str, str], list[Path]] = defaultdict(list)

    for path, df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue

        bucket = bucket_raw_universe(meta["universe"])
        year = meta["year"]
        month = meta["mmdd"][:2]
        yyyymm = meta["yyyymm"]

        target_dir = RAW_DOWNLOAD_DIR / bucket / year / yyyymm
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)

        bucket_year_month_files[(bucket, year, month)].append(target)

        catalog_rows.append(
            {
                "bucket": bucket,
                "universe": meta["universe"],
                "year": year,
                "month": month,
                "yyyymm": yyyymm,
                "date": str(df["date"].dropna().iloc[0]),
                "download_path": f"./data/downloads/raw/{bucket}/{year}/{yyyymm}/{path.name}",
                "filename": path.name,
                "n_items": int(len(df)),
            }
        )

    for (bucket, year, month), files in bucket_year_month_files.items():
        month_zip_dir = RAW_ZIP_DIR / bucket / year
        month_zip_dir.mkdir(parents=True, exist_ok=True)

        month_zip_path = month_zip_dir / f"{year}-{month}.zip"
        with zipfile.ZipFile(month_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(files):
                zf.write(f, arcname=f.name)

    bucket_year_groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for (bucket, year, _month), files in bucket_year_month_files.items():
        bucket_year_groups[(bucket, year)].extend(files)

    for (bucket, year), files in bucket_year_groups.items():
        year_zip_dir = RAW_ZIP_DIR / bucket
        year_zip_dir.mkdir(parents=True, exist_ok=True)
        year_zip_path = year_zip_dir / f"{year}.zip"
        with zipfile.ZipFile(year_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(files):
                arcname = f"{f.parent.name}/{f.name}"
                zf.write(f, arcname=arcname)

    return catalog_rows


def write_family_outputs(slug: str, latest: pd.DataFrame, daily: pd.DataFrame, monthly: pd.DataFrame) -> None:
    family_dir = DOWNLOAD_DIR / slug
    family_dir.mkdir(parents=True, exist_ok=True)

    latest.to_csv(family_dir / "latest.csv", index=False)
    daily.to_csv(family_dir / "daily.csv", index=False)
    monthly.to_csv(family_dir / "monthly.csv", index=False)

    payload = {
        "asof_date": str(latest["date"].dropna().iloc[0]) if not latest.empty and latest["date"].notna().any() else None,
        "latest": latest.to_dict(orient="records"),
        "daily": daily.to_dict(orient="records"),
        "monthly": monthly.to_dict(orient="records"),
        "last_three_monthly": last_three_monthly_per_family(monthly, "family"),
        "downloads": {
            "latest": f"./data/downloads/{slug}/latest.csv",
            "daily": f"./data/downloads/{slug}/daily.csv",
            "monthly": f"./data/downloads/{slug}/monthly.csv",
        },
    }
    write_json(DOCS_DATA_DIR / f"{slug}.json", payload)


def write_legacy_outputs(
    marketwide_daily: pd.DataFrame,
    value_daily: pd.DataFrame,
    industry_latest: pd.DataFrame,
    etf_latest: pd.DataFrame,
    country_latest: pd.DataFrame,
    indices_daily: pd.DataFrame,
    indices_latest: pd.DataFrame,
    asof_date: str,
) -> None:
    all_market_hist = marketwide_daily[marketwide_daily["family"] == "all_market"].copy()
    write_json(
        DOCS_DATA_DIR / "market_icc.json",
        {
            "asof_date": asof_date,
            "history": all_market_hist[["date", "value", "n_items"]].rename(columns={"value": "vw_icc", "n_items": "n_firms"}).to_dict(orient="records"),
        },
    )

    value_hist = value_daily.pivot(index="date", columns="family", values="value").reset_index()
    if "value" not in value_hist.columns:
        value_hist["value"] = None
    if "growth" not in value_hist.columns:
        value_hist["growth"] = None
    if "ivp" not in value_hist.columns:
        value_hist["ivp"] = None
    write_json(
        DOCS_DATA_DIR / "value_icc_bm.json",
        {
            "asof_date": asof_date,
            "history": value_hist.rename(columns={"value": "value_icc", "growth": "growth_icc", "ivp": "ivp_bm"}).to_dict(orient="records"),
        },
    )

    write_json(
        DOCS_DATA_DIR / "industry_icc.json",
        {
            "asof_date": asof_date,
            "latest": industry_latest.to_dict(orient="records"),
        },
    )

    write_json(
        DOCS_DATA_DIR / "etf_icc.json",
        {
            "asof_date": asof_date,
            "latest": etf_latest.to_dict(orient="records"),
        },
    )

    write_json(
        DOCS_DATA_DIR / "country_icc.json",
        {
            "asof_date": asof_date,
            "latest": country_latest.to_dict(orient="records"),
        },
    )

    write_json(
        DOCS_DATA_DIR / "index_icc.json",
        {
            "asof_date": asof_date,
            "history": indices_daily.rename(columns={"family": "universe", "value": "vw_icc", "n_items": "n_firms"}).to_dict(orient="records"),
            "latest": indices_latest.rename(columns={"family": "universe", "value": "vw_icc", "n_items": "n_firms"}).to_dict(orient="records"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ZIP_DIR.mkdir(parents=True, exist_ok=True)

    base_paths = find_all_snapshots(args.universe)
    if not base_paths:
        raise FileNotFoundError(f"No {args.universe} snapshot found under data/YYYYMM/")

    base_valid = dedup_valid_snapshots(get_valid_snapshots(base_paths))
    if not base_valid:
        raise RuntimeError(f"Found {args.universe} snapshot files, but none are valid.")

    latest_usall_path, latest_usall = base_valid[-1]
    latest_date = str(latest_usall["date"].dropna().iloc[0])

    all_paths = find_all_snapshots(None)
    all_valid = dedup_valid_snapshots(get_valid_snapshots(all_paths))

    all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]] = {}
    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        all_valid_by_universe.setdefault(meta["universe"], []).append((path, df))

    all_market_daily = build_market_series(base_valid, "all_market")
    sp500_daily = build_market_series(all_valid_by_universe.get("sp500", []), "sp500")
    marketwide_daily = pd.concat([all_market_daily, sp500_daily], ignore_index=True)
    marketwide_monthly = latest_per_month(marketwide_daily, ["family"], ["value", "method", "n_items", "source_file"])
    marketwide_latest = (
        marketwide_daily.sort_values("date").drop_duplicates(subset=["family"], keep="last").reset_index(drop=True)
        if not marketwide_daily.empty else pd.DataFrame(columns=marketwide_daily.columns)
    )

    value_daily = build_value_daily(base_valid)
    value_monthly = latest_per_month(value_daily, ["family"], ["value", "method"])
    value_latest = (
        value_daily.sort_values("date").drop_duplicates(subset=["family"], keep="last").reset_index(drop=True)
        if not value_daily.empty else pd.DataFrame(columns=value_daily.columns)
    )

    industry_daily = build_industry_daily(base_valid)
    industry_monthly = latest_per_month(industry_daily, ["family"], ["value", "method", "n_items"])
    industry_latest = (
        industry_daily.sort_values("date").drop_duplicates(subset=["family"], keep="last").reset_index(drop=True)
        if not industry_daily.empty else pd.DataFrame(columns=industry_daily.columns)
    )

    indices_daily = build_indices_daily(all_valid_by_universe)
    indices_monthly = latest_per_month(indices_daily, ["family"], ["value", "method", "n_items", "source_file"])
    indices_latest = (
        indices_daily.sort_values("date").drop_duplicates(subset=["family"], keep="last").reset_index(drop=True)
        if not indices_daily.empty else pd.DataFrame(columns=indices_daily.columns)
    )

    etf_latest, etf_daily, etf_monthly = build_online_family("etfs.csv", latest_usall, "etf_history.csv")
    country_latest, country_daily, country_monthly = build_online_family("country_etfs.csv", latest_usall, "country_history.csv")

    write_family_outputs("marketwide", marketwide_latest, marketwide_daily, marketwide_monthly)
    write_family_outputs("value", value_latest, value_daily, value_monthly)
    write_family_outputs("industry", industry_latest, industry_daily, industry_monthly)
    write_family_outputs("indices", indices_latest, indices_daily, indices_monthly)
    write_family_outputs("etf", etf_latest, etf_daily, etf_monthly)
    write_family_outputs("country", country_latest, country_daily, country_monthly)

    raw_catalog_rows = mirror_raw_and_build_zips(all_valid)

    overview_rows = []
    family_map = {
        "marketwide": marketwide_latest,
        "value": value_latest,
        "industry": industry_latest,
        "etf": etf_latest,
        "country": country_latest,
        "indices": indices_latest,
    }
    monthly_map = {
        "marketwide": marketwide_monthly,
        "value": value_monthly,
        "industry": industry_monthly,
        "etf": etf_monthly,
        "country": country_monthly,
        "indices": indices_monthly,
    }

    for slug, latest_df in family_map.items():
        overview_rows.append(
            {
                "family": slug,
                "latest_daily": latest_df.to_dict(orient="records"),
                "last_three_monthly": last_three_monthly_per_family(monthly_map[slug], "family"),
                "downloads": {
                    "latest": f"./data/downloads/{slug}/latest.csv",
                    "daily": f"./data/downloads/{slug}/daily.csv",
                    "monthly": f"./data/downloads/{slug}/monthly.csv",
                },
            }
        )

    downloads_catalog = {
        "title": "Live ICC data library",
        "asof_date": latest_date,
        "families": {
            slug: {
                "latest_csv": f"./data/downloads/{slug}/latest.csv",
                "daily_csv": f"./data/downloads/{slug}/daily.csv",
                "monthly_csv": f"./data/downloads/{slug}/monthly.csv",
                "history_tree": [],
            }
            for slug in ["marketwide", "value", "industry", "etf", "country", "indices"]
        },
        "raw_snapshots": {
            "usall": [],
            "sp500": [],
            "other_indices": [],
            "other": [],
        },
    }

    raw_grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in raw_catalog_rows:
        raw_grouped[r["bucket"]][r["year"]][r["month"]].append(r)

    for bucket, years in raw_grouped.items():
        year_rows = []
        for year, months in sorted(years.items(), reverse=True):
            month_rows = []
            for month, rows in sorted(months.items(), reverse=True):
                month_rows.append(
                    {
                        "month": month,
                        "download_all_zip": f"./data/downloads/raw_zips/{bucket}/{year}/{year}-{month}.zip",
                        "days": sorted(
                            [
                                {
                                    "date": x["date"],
                                    "universe": x["universe"],
                                    "filename": x["filename"],
                                    "download_path": x["download_path"],
                                    "n_items": x["n_items"],
                                }
                                for x in rows
                            ],
                            key=lambda z: (z["date"], z["universe"]),
                            reverse=True,
                        ),
                    }
                )
            year_rows.append(
                {
                    "year": year,
                    "download_all_zip": f"./data/downloads/raw_zips/{bucket}/{year}.zip",
                    "months": month_rows,
                }
            )
        downloads_catalog["raw_snapshots"][bucket] = year_rows

    write_json(
        DOCS_DATA_DIR / "overview.json",
        {
            "title": "Live ICC data library",
            "asof_date": latest_date,
            "source_file": str(latest_usall_path.relative_to(REPO)),
            "sections": overview_rows,
        },
    )

    write_json(DOCS_DATA_DIR / "downloads_catalog.json", downloads_catalog)

    write_legacy_outputs(
        marketwide_daily=marketwide_daily,
        value_daily=value_daily,
        industry_latest=industry_latest,
        etf_latest=etf_latest,
        country_latest=country_latest,
        indices_daily=indices_daily,
        indices_latest=indices_latest,
        asof_date=latest_date,
    )

    print(f"[build_docs_data] total snapshots found = {len(all_paths)}")
    print(f"[build_docs_data] total valid deduped snapshots = {len(all_valid)}")
    print(f"[build_docs_data] latest valid usall asof_date = {latest_date}")
    print("[build_docs_data] wrote docs/data outputs successfully")


if __name__ == "__main__":
    main()
