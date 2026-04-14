
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.errors import EmptyDataError

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DIR = REPO / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"
DOWNLOAD_DIR = DOCS_DATA_DIR / "downloads"
RAW_DIR = DOWNLOAD_DIR / "raw"
CONFIG_DIR = REPO / "config"

SNAPSHOT_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)

INDEX_UNIVERSES = ["sp500", "sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"]


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
    mask = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def safe_int(x):
    try:
        if pd.isna(x):
            return None
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
        return x if math.isfinite(x) else None
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)


def load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise EmptyDataError(f"empty file: {path}")
    df = pd.read_csv(path)
    if df is None or len(df.columns) == 0:
        raise EmptyDataError(f"no columns: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
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
    df = df[
        df["ticker"].notna()
        & df["date"].notna()
        & df["mktcap"].notna()
        & df["ICC"].notna()
        & np.isfinite(df["mktcap"])
        & np.isfinite(df["ICC"])
        & (df["mktcap"] > 0)
    ].copy()
    return df


def find_snapshot_paths(universe: str | None = None) -> list[Path]:
    out = []
    for p in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(p)
        if meta is None:
            continue
        if universe is not None and meta["universe"] != universe:
            continue
        out.append(((meta["universe"], meta["yyyymmdd"], meta["rerun"], p.name), p))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def get_valid_snapshots(universe: str | None = None) -> list[tuple[Path, pd.DataFrame]]:
    raw = find_snapshot_paths(universe)
    keep: dict[tuple[str, str], tuple[Path, pd.DataFrame, int]] = {}
    for path in raw:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        try:
            df = load_snapshot(path)
            if df.empty:
                continue
            date_value = str(df["date"].iloc[0])
            key = (meta["universe"], date_value)
            score = meta["rerun"]
            prev = keep.get(key)
            if prev is None or score >= prev[2]:
                keep[key] = (path, df, score)
        except Exception as e:
            print(f"[build_docs_data] skip {path}: {type(e).__name__}: {e}")
    out = [(p, df) for (p, df, _score) in keep.values()]
    out.sort(key=lambda x: (parse_snapshot_meta(x[0])["universe"], str(x[1]["date"].iloc[0])))
    return out


def latest_by_month(df_hist: pd.DataFrame, value_cols: list[str]) -> tuple[list[str], list[dict]]:
    if df_hist.empty:
        return [], []
    d = df_hist.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.to_period("M").astype(str)
    month_ends = d.sort_values("date").groupby("month", as_index=False).tail(1)
    month_ends = month_ends.sort_values("date").tail(3)
    cols = month_ends["month"].tolist()
    rows = []
    for col in value_cols:
        rows.append({
            "label": col["label"],
            "values": [safe_float(x) for x in month_ends[col["key"]].tolist()]
        })
    return cols, rows


def align_rows(columns: list[str], rows: list[dict]) -> list[dict]:
    return rows


def build_market_history(valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]], universe: str) -> pd.DataFrame:
    rows = []
    for path, df in valid_by_universe.get(universe, []):
        rows.append({
            "date": str(df["date"].iloc[0]),
            "vw_icc": weighted_mean(df["ICC"], df["mktcap"]),
            "ew_icc": float(df["ICC"].mean()),
            "n_firms": int(len(df)),
            "source_file": str(path.relative_to(REPO)),
        })
    if not rows:
        return pd.DataFrame(columns=["date", "vw_icc", "ew_icc", "n_firms", "source_file"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_value_history(usall_valid: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, raw_df in usall_valid:
        df = raw_df.copy()
        df = df[
            df["bm"].notna()
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
        tmp = df.groupby("portfolio").apply(lambda g: weighted_mean(g["ICC"], g["mktcap"])).to_dict()
        sl, bl, sh, bh = tmp.get("S/L", np.nan), tmp.get("B/L", np.nan), tmp.get("S/H", np.nan), tmp.get("B/H", np.nan)
        growth = np.nan if (pd.isna(sl) or pd.isna(bl)) else float((sl + bl) / 2)
        value = np.nan if (pd.isna(sh) or pd.isna(bh)) else float((sh + bh) / 2)
        rows.append({
            "date": str(df["date"].iloc[0]),
            "value_icc": value,
            "growth_icc": growth,
            "ivp_bm": np.nan if (pd.isna(value) or pd.isna(growth)) else float(value - growth),
            "n_firms": int(len(df)),
            "source_file": str(path.relative_to(REPO)),
        })
    if not rows:
        return pd.DataFrame(columns=["date","value_icc","growth_icc","ivp_bm","n_firms","source_file"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def latest_table_by_sector(latest_usall: pd.DataFrame) -> pd.DataFrame:
    df = latest_usall.copy()
    df = df[df["sector"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["sector","vw_icc","ew_icc","n_firms"])
    out = df.groupby("sector").apply(lambda g: pd.Series({
        "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
        "ew_icc": float(g["ICC"].mean()),
        "n_firms": int(len(g))
    })).reset_index()
    return out.sort_values("sector").reset_index(drop=True)


def industry_overview_series(usall_valid: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for _path, df in usall_valid:
        sector_tbl = latest_table_by_sector(df)
        if sector_tbl.empty:
            continue
        rows.append({
            "date": str(df["date"].iloc[0]),
            "median_icc": float(sector_tbl["vw_icc"].median()),
            "mean_icc": float(sector_tbl["vw_icc"].mean()),
            "n_groups": int(len(sector_tbl)),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame(columns=["date","median_icc","mean_icc","n_groups"])


def get_top_holdings(etf_ticker: str) -> pd.DataFrame:
    t = yf.Ticker(etf_ticker)
    fd = getattr(t, "funds_data", None)
    if fd is None:
        raise ValueError("funds_data unavailable")
    h = getattr(fd, "top_holdings", None)
    if h is None:
        raise ValueError("top_holdings unavailable")
    h = h.copy() if isinstance(h, pd.DataFrame) else pd.DataFrame(h)
    if h.empty:
        raise ValueError("top_holdings empty")
    cols = {str(c).lower(): c for c in h.columns}
    symbol_col = cols.get("symbol") or cols.get("holding") or cols.get("ticker") or h.columns[0]
    weight_col = cols.get("holdingpercent") or cols.get("holding_percent") or cols.get("weight") or cols.get("percent")
    if weight_col is None:
        for c in h.columns[::-1]:
            vals = pd.to_numeric(h[c], errors="coerce")
            if vals.notna().sum() >= max(1, len(h)//2):
                weight_col = c
                break
    if weight_col is None:
        raise ValueError("weight column not found")
    out = pd.DataFrame({
        "symbol": h[symbol_col].astype(str).str.upper().str.strip(),
        "raw_weight": pd.to_numeric(h[weight_col], errors="coerce"),
    })
    out = out[out["symbol"].notna() & out["raw_weight"].notna()].copy()
    out["weight"] = np.where(out["raw_weight"].abs().max() > 1.5, out["raw_weight"]/100.0, out["raw_weight"])
    out = out[out["weight"] > 0].copy()
    if out.empty:
        raise ValueError("no positive weights")
    out["weight"] = out["weight"] / out["weight"].sum()
    return out[["symbol","weight"]]


def build_holdings_cache(cfg: pd.DataFrame, id_col: str) -> dict[str, pd.DataFrame]:
    cache = {}
    for _, row in cfg.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        if not ticker:
            continue
        try:
            cache[ticker] = get_top_holdings(ticker)
        except Exception as e:
            print(f"[build_docs_data] holdings failed for {ticker}: {type(e).__name__}: {e}")
            cache[ticker] = pd.DataFrame(columns=["symbol", "weight"])
    return cache


def latest_proxy_table(cfg: pd.DataFrame, latest_usall: pd.DataFrame, holdings_cache: dict[str, pd.DataFrame], kind: str) -> pd.DataFrame:
    base = latest_usall[["ticker","ICC","mktcap","sector","name"]].copy()
    base["ticker"] = base["ticker"].astype(str).str.upper().str.strip()
    rows = []
    for _, row in cfg.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        h = holdings_cache.get(ticker, pd.DataFrame(columns=["symbol","weight"]))
        label = str(row.get("label", ticker))
        category = str(row.get("category", "")) if "category" in row.index else ""
        country = str(row.get("country", "")) if "country" in row.index else ""
        if h.empty:
            rows.append({
                "ticker": ticker, "label": label, "category": category, "country": country,
                "vw_icc": np.nan, "coverage_weight": np.nan, "n_holdings": 0, "n_matched": 0, "status": "no_online_holdings"
            })
            continue
        merged = h.merge(base, left_on="symbol", right_on="ticker", how="left")
        matched = merged["ICC"].notna()
        coverage = float(merged.loc[matched, "weight"].sum()) if len(merged) else np.nan
        icc = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"]) if matched.sum() else np.nan
        rows.append({
            "ticker": ticker, "label": label, "category": category, "country": country,
            "vw_icc": icc, "coverage_weight": coverage, "n_holdings": int(len(h)), "n_matched": int(matched.sum()),
            "status": "ok" if matched.sum() else "no_match"
        })
    df = pd.DataFrame(rows)
    if kind == "country":
        cols = ["country","ticker","label","vw_icc","coverage_weight","n_holdings","n_matched","status"]
        return df[cols].sort_values("country").reset_index(drop=True) if not df.empty else pd.DataFrame(columns=cols)
    cols = ["ticker","label","category","vw_icc","coverage_weight","n_holdings","n_matched","status"]
    return df[cols].sort_values(["category","ticker"]).reset_index(drop=True) if not df.empty else pd.DataFrame(columns=cols)


def proxy_overview_series(cfg: pd.DataFrame, usall_valid: list[tuple[Path, pd.DataFrame]], holdings_cache: dict[str, pd.DataFrame], kind: str) -> pd.DataFrame:
    rows = []
    for _path, df in usall_valid:
        latest = latest_proxy_table(cfg, df, holdings_cache, kind)
        if latest.empty:
            continue
        s = pd.to_numeric(latest["vw_icc"], errors="coerce")
        s = s[np.isfinite(s)]
        if len(s) == 0:
            continue
        rows.append({
            "date": str(df["date"].iloc[0]),
            "median_icc": float(np.median(s)),
            "mean_icc": float(np.mean(s)),
            "n_groups": int(len(s)),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame(columns=["date","median_icc","mean_icc","n_groups"])


def build_indices_outputs(valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_rows = []
    latest_rows = []
    for universe in INDEX_UNIVERSES:
        hist = build_market_history(valid_by_universe, universe)
        if hist.empty:
            continue
        hist = hist.copy()
        hist["universe"] = universe
        hist_rows.append(hist)
        latest = hist.sort_values("date").iloc[-1].to_dict()
        latest_rows.append({
            "universe": universe,
            "date": latest["date"],
            "vw_icc": latest["vw_icc"],
            "ew_icc": latest["ew_icc"],
            "n_firms": latest["n_firms"],
        })
    hist_all = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame(columns=["date","universe","vw_icc","ew_icc","n_firms"])
    latest_df = pd.DataFrame(latest_rows).sort_values("universe").reset_index(drop=True) if latest_rows else pd.DataFrame(columns=["universe","date","vw_icc","ew_icc","n_firms"])
    return hist_all, latest_df


def month_end_rows(df_hist: pd.DataFrame, value_specs: list[dict]) -> tuple[list[str], list[dict]]:
    cols, rows = latest_by_month(df_hist, value_specs)
    return cols, rows


def align_month_columns(all_months: list[str], blocks: list[tuple[list[str], list[dict]]]) -> list[dict]:
    out = []
    for cols, rows in blocks:
        idx = {m:i for i,m in enumerate(cols)}
        for row in rows:
            vals = []
            for m in all_months:
                if m in idx:
                    vals.append(row["values"][idx[m]])
                else:
                    vals.append(None)
            out.append({"label": row["label"], "values": vals})
    return out


def make_zip_if_possible(paths: list[Path], zip_path: Path) -> str | None:
    if not paths:
        return None
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                if p.exists():
                    zf.write(p, arcname=p.name)
        return f"./data/downloads/{zip_path.relative_to(DOWNLOAD_DIR).as_posix()}"
    except Exception as e:
        print(f"[build_docs_data] zip failed for {zip_path}: {e}")
        return None


def copy_raw_snapshots(all_valid: list[tuple[Path, pd.DataFrame]]) -> list[dict]:
    rows = []
    by_month: dict[tuple[str,str], list[Path]] = defaultdict(list)
    by_year: dict[tuple[str,str], list[Path]] = defaultdict(list)

    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        target_dir = RAW_DIR / meta["universe"] / meta["yyyymm"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)
        rel = f"./data/downloads/raw/{meta['universe']}/{meta['yyyymm']}/{path.name}"
        by_month[(meta["universe"], meta["yyyymm"])].append(target)
        by_year[(meta["universe"], meta["year"])].append(target)
        rows.append({
            "date": str(df["date"].iloc[0]),
            "yyyymm": meta["yyyymm"],
            "year": meta["year"],
            "month": meta["yyyymm"][4:6],
            "universe": meta["universe"],
            "n_firms": int(len(df)),
            "download_path": rel,
            "month_download": None,
            "year_download": {},
        })

    month_zip_map = {}
    for (universe, yyyymm), paths in by_month.items():
        rel = make_zip_if_possible(paths, RAW_DIR / universe / yyyymm / f"{universe}_{yyyymm}_all.zip")
        month_zip_map[(universe, yyyymm)] = rel

    year_zip_map = {}
    for (universe, year), paths in by_year.items():
        rel = make_zip_if_possible(paths, RAW_DIR / universe / year / f"{universe}_{year}_all.zip")
        year_zip_map[(universe, year)] = rel

    for r in rows:
        r["month_download"] = month_zip_map.get((r["universe"], r["yyyymm"]))
        r["year_download"] = {r["year"]: year_zip_map.get((r["universe"], r["year"]))}

    return rows


def build_download_catalog(raw_rows: list[dict]) -> dict:
    fam = {"usall": [], "sp500": [], "other_indices": []}
    for r in sorted(raw_rows, key=lambda x: (x["date"], x["universe"])):
        if r["universe"] == "usall":
            fam["usall"].append(r)
        elif r["universe"] == "sp500":
            fam["sp500"].append(r)
        elif r["universe"] in INDEX_UNIVERSES:
            fam["other_indices"].append(r)
    return fam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_valid = get_valid_snapshots(None)
    if not all_valid:
        raise RuntimeError("No valid snapshots found under data/YYYYMM/")
    valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]] = defaultdict(list)
    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        if meta:
            valid_by_universe[meta["universe"]].append((path, df))

    usall_valid = valid_by_universe.get(args.universe, [])
    if not usall_valid:
        raise RuntimeError(f"No valid snapshots for {args.universe}")
    latest_usall = usall_valid[-1][1]
    latest_date = str(latest_usall["date"].iloc[0])

    market_all = build_market_history(valid_by_universe, "usall")
    market_sp500 = build_market_history(valid_by_universe, "sp500")
    value_hist = build_value_history(usall_valid)

    latest_industry = latest_table_by_sector(latest_usall)
    industry_series = industry_overview_series(usall_valid)

    etf_cfg = pd.read_csv(CONFIG_DIR / "etfs.csv")
    country_cfg = pd.read_csv(CONFIG_DIR / "country_etfs.csv")
    etf_holdings_cache = build_holdings_cache(etf_cfg, "ticker")
    country_holdings_cache = build_holdings_cache(country_cfg, "ticker")

    latest_etf = latest_proxy_table(etf_cfg, latest_usall, etf_holdings_cache, "etf")
    etf_series = proxy_overview_series(etf_cfg, usall_valid, etf_holdings_cache, "etf")
    latest_country = latest_proxy_table(country_cfg, latest_usall, country_holdings_cache, "country")
    country_series = proxy_overview_series(country_cfg, usall_valid, country_holdings_cache, "country")

    indices_hist, latest_indices = build_indices_outputs(valid_by_universe)

    raw_rows = copy_raw_snapshots(all_valid)
    download_tree = build_download_catalog(raw_rows)

    # CSV downloads
    market_all.to_csv(DOWNLOAD_DIR / "marketwide_all_history.csv", index=False)
    market_sp500.to_csv(DOWNLOAD_DIR / "marketwide_sp500_history.csv", index=False)
    value_hist.to_csv(DOWNLOAD_DIR / "value_history.csv", index=False)
    latest_industry.to_csv(DOWNLOAD_DIR / "industry_latest.csv", index=False)
    industry_series.to_csv(DOWNLOAD_DIR / "industry_overview_history.csv", index=False)
    latest_etf.to_csv(DOWNLOAD_DIR / "etf_latest.csv", index=False)
    etf_series.to_csv(DOWNLOAD_DIR / "etf_overview_history.csv", index=False)
    latest_country.to_csv(DOWNLOAD_DIR / "country_latest.csv", index=False)
    country_series.to_csv(DOWNLOAD_DIR / "country_overview_history.csv", index=False)
    indices_hist.to_csv(DOWNLOAD_DIR / "indices_history.csv", index=False)
    latest_indices.to_csv(DOWNLOAD_DIR / "indices_latest.csv", index=False)

    # monthly summaries
    cols_all, rows_all = month_end_rows(market_all, [{"key":"vw_icc","label":"All market"}])
    cols_sp500, rows_sp500 = month_end_rows(market_sp500, [{"key":"vw_icc","label":"S&P 500"}])
    cols_value, rows_value = month_end_rows(value_hist, [
        {"key":"value_icc","label":"Value ICC"},
        {"key":"growth_icc","label":"Growth ICC"},
        {"key":"ivp_bm","label":"IVP"},
    ])
    cols_ind, rows_ind = month_end_rows(industry_series, [{"key":"median_icc","label":"Industry median"}])
    cols_etf, rows_etf = month_end_rows(etf_series, [{"key":"median_icc","label":"ETF median"}])
    cols_country, rows_country = month_end_rows(country_series, [{"key":"median_icc","label":"Country median"}])

    all_months = []
    for arr in [cols_all, cols_sp500, cols_value, cols_ind, cols_etf, cols_country]:
        for m in arr:
            if m not in all_months:
                all_months.append(m)
    all_months = sorted(all_months)[-3:]

    overview_monthly_rows = align_month_columns(all_months, [
        (cols_all, rows_all),
        (cols_sp500, rows_sp500),
        (cols_value, rows_value),
        (cols_ind, rows_ind),
        (cols_etf, rows_etf),
        (cols_country, rows_country),
    ])

    latest_daily = {
        "all_market_vw_icc": safe_float(market_all.iloc[-1]["vw_icc"]) if not market_all.empty else None,
        "sp500_vw_icc": safe_float(market_sp500.iloc[-1]["vw_icc"]) if not market_sp500.empty else None,
        "value_icc": safe_float(value_hist.iloc[-1]["value_icc"]) if not value_hist.empty else None,
        "growth_icc": safe_float(value_hist.iloc[-1]["growth_icc"]) if not value_hist.empty else None,
        "ivp_bm": safe_float(value_hist.iloc[-1]["ivp_bm"]) if not value_hist.empty else None,
        "industry_median_icc": safe_float(industry_series.iloc[-1]["median_icc"]) if not industry_series.empty else None,
        "etf_median_icc": safe_float(etf_series.iloc[-1]["median_icc"]) if not etf_series.empty else None,
        "country_median_icc": safe_float(country_series.iloc[-1]["median_icc"]) if not country_series.empty else None,
    }

    overview_downloads = {
        "marketwide": [
            {"label": "All market history", "path": "./data/downloads/marketwide_all_history.csv"},
            {"label": "S&P 500 history", "path": "./data/downloads/marketwide_sp500_history.csv"},
        ],
        "value": [{"label": "Value / Growth / IVP history", "path": "./data/downloads/value_history.csv"}],
        "industry": [
            {"label": "Industry overview history", "path": "./data/downloads/industry_overview_history.csv"},
            {"label": "Industry latest table", "path": "./data/downloads/industry_latest.csv"},
        ],
        "etf": [
            {"label": "ETF overview history", "path": "./data/downloads/etf_overview_history.csv"},
            {"label": "ETF latest table", "path": "./data/downloads/etf_latest.csv"},
        ],
        "country": [
            {"label": "Country overview history", "path": "./data/downloads/country_overview_history.csv"},
            {"label": "Country latest table", "path": "./data/downloads/country_latest.csv"},
        ],
        "indices": [
            {"label": "Indices history", "path": "./data/downloads/indices_history.csv"},
            {"label": "Indices latest table", "path": "./data/downloads/indices_latest.csv"},
        ],
    }

    write_json(DOCS_DATA_DIR / "overview.json", {
        "asof_date": latest_date,
        "latest_daily": latest_daily,
        "three_months": {"columns": all_months, "rows": overview_monthly_rows},
        "downloads": overview_downloads,
    })

    write_json(DOCS_DATA_DIR / "marketwide.json", {
        "asof_date": latest_date,
        "all_history": market_all.to_dict(orient="records"),
        "sp500_history": market_sp500.to_dict(orient="records"),
        "three_months": {"columns": all_months, "rows": align_month_columns(all_months, [(cols_all, rows_all), (cols_sp500, rows_sp500)])},
    })

    write_json(DOCS_DATA_DIR / "value.json", {
        "asof_date": latest_date,
        "history": value_hist.to_dict(orient="records"),
        "three_months": {"columns": all_months, "rows": align_month_columns(all_months, [(cols_value, rows_value)])},
    })

    write_json(DOCS_DATA_DIR / "industry.json", {
        "asof_date": latest_date,
        "latest": latest_industry.to_dict(orient="records"),
        "overview_history": industry_series.to_dict(orient="records"),
        "three_months": {"columns": all_months, "rows": align_month_columns(all_months, [(cols_ind, rows_ind)])},
    })

    write_json(DOCS_DATA_DIR / "etf.json", {
        "asof_date": latest_date,
        "latest": latest_etf.to_dict(orient="records"),
        "overview_history": etf_series.to_dict(orient="records"),
        "three_months": {"columns": all_months, "rows": align_month_columns(all_months, [(cols_etf, rows_etf)])},
    })

    write_json(DOCS_DATA_DIR / "country.json", {
        "asof_date": latest_date,
        "latest": latest_country.to_dict(orient="records"),
        "overview_history": country_series.to_dict(orient="records"),
        "three_months": {"columns": all_months, "rows": align_month_columns(all_months, [(cols_country, rows_country)])},
    })

    write_json(DOCS_DATA_DIR / "indices.json", {
        "asof_date": latest_date,
        "latest": latest_indices.to_dict(orient="records"),
        "history": indices_hist.to_dict(orient="records"),
    })

    write_json(DOCS_DATA_DIR / "downloads_catalog.json", {
        "asof_date": latest_date,
        "families": download_tree,
        "aggregate": overview_downloads,
    })

    print(f"[build_docs_data] asof_date={latest_date}")
    print(f"[build_docs_data] usall snapshots={len(usall_valid)}")
    print(f"[build_docs_data] indices available={[u for u in INDEX_UNIVERSES if len(valid_by_universe.get(u, [])) > 0]}")
    print("[build_docs_data] docs/data generated successfully")


if __name__ == "__main__":
    main()
