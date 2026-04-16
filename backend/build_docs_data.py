from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.errors import EmptyDataError

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DIR = REPO / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"
DOWNLOAD_DIR = DOCS_DATA_DIR / "downloads"
RAW_DOWNLOAD_DIR = DOWNLOAD_DIR / "raw"
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
    "rut1000",
]


def parse_snapshot_meta(path: Path) -> dict | None:
    m = SNAPSHOT_RE.match(path.name)
    if not m:
        return None
    year = m.group("year")
    mmdd = m.group("mmdd")
    rerun = int(m.group("rerun") or 0)
    yyyymm = f"{year}{mmdd[:2]}"
    yyyymmdd = f"{year}{mmdd}"
    return {
        "universe": m.group("universe"),
        "year": year,
        "mmdd": mmdd,
        "yyyymm": yyyymm,
        "yyyymmdd": yyyymmdd,
        "rerun": rerun,
    }


def safe_float(x):
    if pd.isna(x):
        return None
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def safe_int(x):
    if pd.isna(x):
        return None
    try:
        return int(x)
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
        files.append(((meta["universe"], meta["yyyymmdd"], meta["rerun"], p.name), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise EmptyDataError(f"Empty file: {path}")

    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        raise EmptyDataError(f"Empty CSV content: {path}")

    if df is None or len(df.columns) == 0:
        raise EmptyDataError(f"No columns parsed from {path}")

    df.columns = [str(c).strip() for c in df.columns]

    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

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

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
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
    valid: list[tuple[Path, pd.DataFrame]] = []
    for path in paths:
        try:
            df = clean_df(load_snapshot(path))
        except Exception as e:
            print(f"[build_docs_data] skipping invalid snapshot: {path} | {type(e).__name__}: {e}")
            continue
        if df.empty:
            continue
        valid.append((path, df))
    return valid


def dedup_valid_snapshots(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> list[tuple[Path, pd.DataFrame]]:
    keep: dict[tuple[str, str], tuple[Path, pd.DataFrame]] = {}
    for path, df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        date_value = str(df["date"].dropna().iloc[0])
        keep[(meta["universe"], date_value)] = (path, df)
    out = list(keep.values())
    out.sort(key=lambda x: (parse_snapshot_meta(x[0])["universe"], str(x[1]["date"].dropna().iloc[0]), parse_snapshot_meta(x[0])["rerun"], x[0].name))
    return out


def read_csv_config(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=expected_cols)
    df = pd.read_csv(path)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols].copy()


def month_end_rows(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    tmp = tmp.sort_values(keys + ["date"]).reset_index(drop=True)
    out = tmp.groupby(keys + ["month"], dropna=False).tail(1).reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.drop(columns=["month"])


def latest_three_months(df: pd.DataFrame, keys: list[str], value_cols: list[str]) -> list[dict]:
    monthly = month_end_rows(df, keys)
    if monthly.empty:
        return []
    monthly = monthly.copy()
    monthly["date"] = pd.to_datetime(monthly["date"])
    if keys:
        monthly = monthly.sort_values(keys + ["date"])
    else:
        monthly = monthly.sort_values(["date"])
    out = monthly.groupby(keys if keys else lambda _: 0, dropna=False).tail(3) if keys else monthly.tail(3)
    out = out.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out[[*(keys or []), "date", *value_cols]].to_dict(orient="records")


def build_market_history(valid_snapshots: list[tuple[Path, pd.DataFrame]], label: str) -> pd.DataFrame:
    rows = []
    for path, df in valid_snapshots:
        rows.append(
            {
                "date": str(df["date"].iloc[0]),
                "family": label,
                "vw_icc": weighted_mean(df["ICC"], df["mktcap"]),
                "ew_icc": float(df["ICC"].mean()),
                "n_items": int(len(df)),
                "source_file": str(path.relative_to(REPO)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "family", "vw_icc", "ew_icc", "n_items", "source_file"])
    out = pd.DataFrame(rows).sort_values("date").drop_duplicates(["date", "family"], keep="last").reset_index(drop=True)
    return out


def build_value_history(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
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
        df["bm_bucket"] = pd.cut(df["bm"], bins=[-np.inf, bm30, bm70, np.inf], labels=["L", "M", "H"], include_lowest=True, right=True).astype("string")
        df["portfolio"] = df["size_bucket"] + "/" + df["bm_bucket"].astype(str)
        pt = (
            df.groupby("portfolio", dropna=False)
            .apply(lambda g: pd.Series({"vw_icc": weighted_mean(g["ICC"], g["mktcap"]), "n_items": int(len(g))}))
            .reset_index()
        )
        d = dict(zip(pt["portfolio"], pt["vw_icc"]))
        growth_icc = np.nan if pd.isna(d.get("S/L", np.nan)) or pd.isna(d.get("B/L", np.nan)) else float((d["S/L"] + d["B/L"]) / 2.0)
        value_icc = np.nan if pd.isna(d.get("S/H", np.nan)) or pd.isna(d.get("B/H", np.nan)) else float((d["S/H"] + d["B/H"]) / 2.0)
        ivp_bm = np.nan if pd.isna(value_icc) or pd.isna(growth_icc) else float(value_icc - growth_icc)
        rows.append(
            {
                "date": str(df["date"].iloc[0]),
                "value_icc": value_icc,
                "growth_icc": growth_icc,
                "ivp_bm": ivp_bm,
                "n_items": int(len(df)),
                "source_file": str(path.relative_to(REPO)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "value_icc", "growth_icc", "ivp_bm", "n_items", "source_file"])
    return pd.DataFrame(rows).sort_values("date").drop_duplicates(["date"], keep="last").reset_index(drop=True)


def build_industry_daily(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, df in valid_snapshots:
        d = clean_df(df)
        d = d[d["sector"].notna()].copy()
        if d.empty:
            continue
        by_sector = (
            d.groupby("sector", dropna=False)
            .apply(lambda g: pd.Series({
                "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
                "ew_icc": float(g["ICC"].mean()),
                "n_items": int(len(g)),
                "total_mktcap": float(g["mktcap"].sum()),
            }))
            .reset_index()
        )
        by_sector["date"] = str(d["date"].iloc[0])
        rows.append(by_sector)
    if not rows:
        return pd.DataFrame(columns=["date", "sector", "vw_icc", "ew_icc", "n_items", "total_mktcap"])
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["sector", "date"]).drop_duplicates(["date", "sector"], keep="last").reset_index(drop=True)
    return out


def summarize_industry(industry_daily: pd.DataFrame) -> pd.DataFrame:
    if industry_daily.empty:
        return pd.DataFrame(columns=["date", "family", "vw_icc", "ew_icc", "n_items"])
    rows = []
    for date, g in industry_daily.groupby("date", dropna=False):
        rows.append({
            "date": date,
            "family": "industry_wide",
            "vw_icc": float(g["vw_icc"].mean()),
            "ew_icc": float(g["ew_icc"].mean()),
            "n_items": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def normalize_weight_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out[out["weight"].notna() & (out["weight"] > 0)].copy()
    if out.empty:
        return out
    if out["weight"].max() > 1.5:
        out["weight"] = out["weight"] / 100.0
    out["weight"] = out["weight"] / out["weight"].sum()
    return out


def get_online_holdings(ticker: str) -> tuple[pd.DataFrame, str]:
    # Universal online method: yfinance fund top holdings.
    t = yf.Ticker(ticker)
    fd = getattr(t, "funds_data", None)
    if fd is None:
        raise ValueError("funds_data unavailable")
    holdings = getattr(fd, "top_holdings", None)
    if holdings is None:
        raise ValueError("top_holdings unavailable")
    h = holdings.copy() if isinstance(holdings, pd.DataFrame) else pd.DataFrame(holdings)
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
        raise ValueError("weight column unavailable")
    out = pd.DataFrame({
        "symbol": h[symbol_col].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False),
        "weight": pd.to_numeric(h[weight_col], errors="coerce"),
    })
    out = normalize_weight_col(out)
    if out.empty:
        raise ValueError("no usable holdings")
    return out[["symbol", "weight"]], "yfinance_top_holdings"


def compute_holdings_family(rows_cfg: pd.DataFrame, latest_usall: pd.DataFrame, key_col: str) -> pd.DataFrame:
    base = clean_df(latest_usall)[["ticker", "ICC", "mktcap", "sector", "name"]].copy()
    base["ticker"] = base["ticker"].astype(str).str.upper().str.strip()
    out_rows = []
    asof_date = str(base.index.size and latest_usall["date"].dropna().iloc[0])
    for _, cfg in rows_cfg.iterrows():
        ticker = str(cfg.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        label = str(cfg.get("label", ticker))
        key_val = str(cfg.get(key_col, label))
        category = str(cfg.get("category", ""))
        try:
            h, source = get_online_holdings(ticker)
            merged = h.merge(base, left_on="symbol", right_on="ticker", how="left")
            matched = merged["ICC"].notna()
            coverage = float(merged.loc[matched, "weight"].sum()) if len(merged) else 0.0
            if matched.sum() == 0:
                out_rows.append({
                    "date": asof_date,
                    key_col: key_val,
                    "ticker": ticker,
                    "label": label,
                    "category": category,
                    "vw_icc": np.nan,
                    "coverage_weight": 0.0,
                    "n_items": int(len(h)),
                    "n_matched": 0,
                    "source": source,
                    "status": "no_matched_holdings",
                })
                continue
            icc = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"])
            out_rows.append({
                "date": asof_date,
                key_col: key_val,
                "ticker": ticker,
                "label": label,
                "category": category,
                "vw_icc": icc,
                "coverage_weight": coverage,
                "n_items": int(len(h)),
                "n_matched": int(matched.sum()),
                "source": source,
                "status": "ok" if coverage > 0 else "low_coverage",
            })
        except Exception as e:
            out_rows.append({
                "date": asof_date,
                key_col: key_val,
                "ticker": ticker,
                "label": label,
                "category": category,
                "vw_icc": np.nan,
                "coverage_weight": np.nan,
                "n_items": np.nan,
                "n_matched": np.nan,
                "source": "online",
                "status": f"error: {type(e).__name__}",
            })
    return pd.DataFrame(out_rows)


def upsert_history(latest_df: pd.DataFrame, history_path: Path, key_cols: list[str]) -> pd.DataFrame:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        hist = pd.read_csv(history_path)
    else:
        hist = pd.DataFrame(columns=latest_df.columns)
    combined = pd.concat([hist, latest_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(key_cols).reset_index(drop=True)
    combined.to_csv(history_path, index=False)
    return combined


def summarize_holdings_family(daily_df: pd.DataFrame, family_name: str) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(columns=["date", "family", "vw_icc", "ew_icc", "n_items"])
    rows = []
    for date, g in daily_df.groupby("date", dropna=False):
        good = g[pd.to_numeric(g["vw_icc"], errors="coerce").notna()].copy()
        if good.empty:
            rows.append({"date": date, "family": family_name, "vw_icc": np.nan, "ew_icc": np.nan, "n_items": 0})
        else:
            rows.append({
                "date": date,
                "family": family_name,
                "vw_icc": float(pd.to_numeric(good["vw_icc"], errors="coerce").mean()),
                "ew_icc": float(pd.to_numeric(good["vw_icc"], errors="coerce").mean()),
                "n_items": int(len(good)),
            })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_index_outputs(all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_rows = []
    latest_rows = []
    for universe in INDEX_UNIVERSES:
        valid = all_valid_by_universe.get(universe, [])
        if not valid:
            continue
        hist = build_market_history(valid, universe)
        if hist.empty:
            continue
        history_rows.append(hist)
        latest_rows.append(hist.sort_values("date").iloc[-1].to_dict())
    history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame(columns=["date", "family", "vw_icc", "ew_icc", "n_items", "source_file"])
    latest = pd.DataFrame(latest_rows) if latest_rows else pd.DataFrame(columns=["date", "family", "vw_icc", "ew_icc", "n_items", "source_file"])
    if not latest.empty:
        latest = latest.sort_values("family").reset_index(drop=True)
    return history, latest


def copy_raw_snapshots(valid_snapshots: Iterable[tuple[Path, pd.DataFrame]]) -> None:
    RAW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for path, _df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        target_dir = RAW_DOWNLOAD_DIR / meta["yyyymm"]
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_dir / path.name)


def build_download_tabs(all_valid: list[tuple[Path, pd.DataFrame]]) -> dict:
    groups = {"usall": [], "sp500": [], "other_indices": []}
    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        universe = meta["universe"]
        row = {
            "date": str(df["date"].dropna().iloc[0]),
            "yyyymm": meta["yyyymm"],
            "universe": universe,
            "n_items": int(len(df)),
            "download_path": f"./data/downloads/raw/{meta['yyyymm']}/{path.name}",
        }
        if universe == "usall":
            groups["usall"].append(row)
        elif universe == "sp500":
            groups["sp500"].append(row)
        else:
            groups["other_indices"].append(row)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: (x["date"], x["universe"]), reverse=True)
    return groups


def build_family_downloads() -> dict:
    families = {
        "marketwide": [
            {"label": "Daily history CSV", "path": "./data/downloads/marketwide_daily_history.csv"},
            {"label": "Monthly history CSV", "path": "./data/downloads/marketwide_monthly_history.csv"},
            {"label": "Latest breakdown CSV", "path": "./data/downloads/marketwide_latest.csv"},
        ],
        "value": [
            {"label": "Daily history CSV", "path": "./data/downloads/value_daily_history.csv"},
            {"label": "Monthly history CSV", "path": "./data/downloads/value_monthly_history.csv"},
        ],
        "industry": [
            {"label": "Daily history CSV", "path": "./data/downloads/industry_daily_history.csv"},
            {"label": "Monthly history CSV", "path": "./data/downloads/industry_monthly_history.csv"},
            {"label": "Latest breakdown CSV", "path": "./data/downloads/industry_latest.csv"},
        ],
        "etf": [
            {"label": "Daily history CSV", "path": "./data/downloads/etf_daily_history.csv"},
            {"label": "Monthly history CSV", "path": "./data/downloads/etf_monthly_history.csv"},
            {"label": "Latest breakdown CSV", "path": "./data/downloads/etf_latest.csv"},
        ],
        "country": [
            {"label": "Daily history CSV", "path": "./data/downloads/country_daily_history.csv"},
            {"label": "Monthly history CSV", "path": "./data/downloads/country_monthly_history.csv"},
            {"label": "Latest breakdown CSV", "path": "./data/downloads/country_latest.csv"},
        ],
        "indices": [
            {"label": "Daily history CSV", "path": "./data/downloads/index_daily_history.csv"},
            {"label": "Monthly history CSV", "path": "./data/downloads/index_monthly_history.csv"},
            {"label": "Latest breakdown CSV", "path": "./data/downloads/index_latest.csv"},
        ],
    }
    return families


def summary_row(name: str, latest_value: float | None, monthly_records: list[dict], value_key: str) -> dict:
    ordered = sorted(monthly_records, key=lambda x: x["date"], reverse=True)[:3]
    while len(ordered) < 3:
        ordered.append({"date": None, value_key: None})
    return {
        "family": name,
        "latest_daily": latest_value,
        "m1_date": ordered[0].get("date"),
        "m1_value": ordered[0].get(value_key),
        "m2_date": ordered[1].get("date"),
        "m2_value": ordered[1].get(value_key),
        "m3_date": ordered[2].get("date"),
        "m3_value": ordered[2].get(value_key),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    base_valid = dedup_valid_snapshots(get_valid_snapshots(find_all_snapshots(args.universe)))
    if not base_valid:
        raise RuntimeError(f"No valid {args.universe} snapshots found")
    latest_usall_path, latest_usall = base_valid[-1]
    latest_date = str(latest_usall["date"].dropna().iloc[0])

    all_valid = dedup_valid_snapshots(get_valid_snapshots(find_all_snapshots(None)))
    all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]] = {}
    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        all_valid_by_universe.setdefault(meta["universe"], []).append((path, df))

    market_all_daily = build_market_history(base_valid, "all_market")
    sp500_daily = build_market_history(all_valid_by_universe.get("sp500", []), "sp500")
    marketwide_daily = pd.concat([market_all_daily, sp500_daily], ignore_index=True)
    marketwide_monthly = month_end_rows(marketwide_daily, ["family"])
    marketwide_latest = marketwide_daily.sort_values("date").groupby("family", dropna=False).tail(1).reset_index(drop=True)

    value_daily = build_value_history(base_valid)
    value_monthly = month_end_rows(value_daily, [])

    industry_daily = build_industry_daily(base_valid)
    industry_monthly = month_end_rows(industry_daily, ["sector"])
    industry_summary_daily = summarize_industry(industry_daily)
    industry_summary_monthly = month_end_rows(industry_summary_daily, ["family"])
    industry_latest = industry_daily[industry_daily["date"] == latest_date].sort_values(["total_mktcap", "sector"], ascending=[False, True]).reset_index(drop=True)

    etf_cfg = read_csv_config(CONFIG_DIR / "etfs.csv", ["ticker", "label", "category"])
    country_cfg = read_csv_config(CONFIG_DIR / "country_etfs.csv", ["ticker", "label", "country"])

    etf_latest = compute_holdings_family(etf_cfg, latest_usall, key_col="label")
    country_latest = compute_holdings_family(country_cfg, latest_usall, key_col="country")

    etf_daily = upsert_history(etf_latest, DOWNLOAD_DIR / "etf_daily_history.csv", ["date", "label"])
    country_daily = upsert_history(country_latest, DOWNLOAD_DIR / "country_daily_history.csv", ["date", "country"])
    etf_monthly = month_end_rows(etf_daily, ["label"])
    country_monthly = month_end_rows(country_daily, ["country"])
    etf_summary_daily = summarize_holdings_family(etf_daily, "etf")
    country_summary_daily = summarize_holdings_family(country_daily, "country")
    etf_summary_monthly = month_end_rows(etf_summary_daily, ["family"])
    country_summary_monthly = month_end_rows(country_summary_daily, ["family"])

    index_daily, index_latest = build_index_outputs(all_valid_by_universe)
    index_monthly = month_end_rows(index_daily, ["family"])
    index_summary_daily = summarize_holdings_family(index_daily.rename(columns={"family": "label"}), "indices")
    index_summary_monthly = month_end_rows(index_summary_daily, ["family"])

    copy_raw_snapshots(all_valid)

    marketwide_daily.to_csv(DOWNLOAD_DIR / "marketwide_daily_history.csv", index=False)
    marketwide_monthly.to_csv(DOWNLOAD_DIR / "marketwide_monthly_history.csv", index=False)
    marketwide_latest.to_csv(DOWNLOAD_DIR / "marketwide_latest.csv", index=False)
    value_daily.to_csv(DOWNLOAD_DIR / "value_daily_history.csv", index=False)
    value_monthly.to_csv(DOWNLOAD_DIR / "value_monthly_history.csv", index=False)
    industry_daily.to_csv(DOWNLOAD_DIR / "industry_daily_history.csv", index=False)
    industry_monthly.to_csv(DOWNLOAD_DIR / "industry_monthly_history.csv", index=False)
    industry_latest.to_csv(DOWNLOAD_DIR / "industry_latest.csv", index=False)
    etf_monthly.to_csv(DOWNLOAD_DIR / "etf_monthly_history.csv", index=False)
    etf_latest.to_csv(DOWNLOAD_DIR / "etf_latest.csv", index=False)
    country_monthly.to_csv(DOWNLOAD_DIR / "country_monthly_history.csv", index=False)
    country_latest.to_csv(DOWNLOAD_DIR / "country_latest.csv", index=False)
    index_daily.to_csv(DOWNLOAD_DIR / "index_daily_history.csv", index=False)
    index_monthly.to_csv(DOWNLOAD_DIR / "index_monthly_history.csv", index=False)
    index_latest.to_csv(DOWNLOAD_DIR / "index_latest.csv", index=False)

    overview_rows = [
        summary_row(
            "Marketwide (All U.S.)",
            safe_float(market_all_daily.sort_values("date").iloc[-1]["vw_icc"]) if not market_all_daily.empty else None,
            latest_three_months(market_all_daily, ["family"], ["vw_icc"]),
            "vw_icc",
        ),
        summary_row(
            "Marketwide (S&P 500)",
            safe_float(sp500_daily.sort_values("date").iloc[-1]["vw_icc"]) if not sp500_daily.empty else None,
            latest_three_months(sp500_daily, ["family"], ["vw_icc"]),
            "vw_icc",
        ),
        summary_row(
            "Value Premium (IVP)",
            safe_float(value_daily.sort_values("date").iloc[-1]["ivp_bm"]) if not value_daily.empty else None,
            latest_three_months(value_daily, [], ["ivp_bm"]),
            "ivp_bm",
        ),
        summary_row(
            "Industry Wide",
            safe_float(industry_summary_daily.sort_values("date").iloc[-1]["vw_icc"]) if not industry_summary_daily.empty else None,
            latest_three_months(industry_summary_daily, ["family"], ["vw_icc"]),
            "vw_icc",
        ),
        summary_row(
            "ETF ICC",
            safe_float(etf_summary_daily.sort_values("date").iloc[-1]["vw_icc"]) if not etf_summary_daily.empty else None,
            latest_three_months(etf_summary_daily, ["family"], ["vw_icc"]),
            "vw_icc",
        ),
        summary_row(
            "Country ICC",
            safe_float(country_summary_daily.sort_values("date").iloc[-1]["vw_icc"]) if not country_summary_daily.empty else None,
            latest_three_months(country_summary_daily, ["family"], ["vw_icc"]),
            "vw_icc",
        ),
    ]

    write_json(DOCS_DATA_DIR / "overview.json", {
        "title": "Fama-French style live ICC data library",
        "asof_date": latest_date,
        "rows": overview_rows,
    })

    write_json(DOCS_DATA_DIR / "marketwide.json", {
        "asof_date": latest_date,
        "latest": marketwide_latest.to_dict(orient="records"),
        "daily": marketwide_daily.to_dict(orient="records"),
        "monthly": marketwide_monthly.to_dict(orient="records"),
    })
    write_json(DOCS_DATA_DIR / "value.json", {
        "asof_date": latest_date,
        "latest": value_daily.tail(1).to_dict(orient="records"),
        "daily": value_daily.to_dict(orient="records"),
        "monthly": value_monthly.to_dict(orient="records"),
    })
    write_json(DOCS_DATA_DIR / "industry.json", {
        "asof_date": latest_date,
        "summary_latest": industry_summary_daily.tail(1).to_dict(orient="records"),
        "summary_monthly": industry_summary_monthly.to_dict(orient="records"),
        "latest": industry_latest.to_dict(orient="records"),
        "daily": industry_daily.to_dict(orient="records"),
        "monthly": industry_monthly.to_dict(orient="records"),
    })
    write_json(DOCS_DATA_DIR / "etf.json", {
        "asof_date": latest_date,
        "summary_latest": etf_summary_daily.tail(1).to_dict(orient="records"),
        "summary_monthly": etf_summary_monthly.to_dict(orient="records"),
        "latest": etf_latest.to_dict(orient="records"),
        "daily": etf_daily.to_dict(orient="records"),
        "monthly": etf_monthly.to_dict(orient="records"),
    })
    write_json(DOCS_DATA_DIR / "country.json", {
        "asof_date": latest_date,
        "summary_latest": country_summary_daily.tail(1).to_dict(orient="records"),
        "summary_monthly": country_summary_monthly.to_dict(orient="records"),
        "latest": country_latest.to_dict(orient="records"),
        "daily": country_daily.to_dict(orient="records"),
        "monthly": country_monthly.to_dict(orient="records"),
    })
    write_json(DOCS_DATA_DIR / "indices.json", {
        "asof_date": latest_date,
        "summary_latest": index_summary_daily.tail(1).to_dict(orient="records"),
        "summary_monthly": index_summary_monthly.to_dict(orient="records"),
        "latest": index_latest.to_dict(orient="records"),
        "daily": index_daily.to_dict(orient="records"),
        "monthly": index_monthly.to_dict(orient="records"),
    })
    write_json(DOCS_DATA_DIR / "downloads_catalog.json", {
        "asof_date": latest_date,
        "families": build_family_downloads(),
        "raw_tabs": build_download_tabs(all_valid),
    })

    print(f"[build_docs_data] latest asof_date = {latest_date}")
    print(f"[build_docs_data] total valid deduped snapshots used = {len(all_valid)}")
    print("[build_docs_data] wrote docs/data and docs/data/downloads")


if __name__ == "__main__":
    main()
