
from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.errors import EmptyDataError

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DATA_DIR = REPO / "docs" / "data"
DOWNLOAD_DIR = DOCS_DATA_DIR / "downloads"
FAMILY_DIR = DOWNLOAD_DIR / "families"
ARCHIVE_DIR = DOWNLOAD_DIR / "archives"
CONFIG_DIR = REPO / "config"

SNAPSHOT_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)

INDEX_UNIVERSES = ["sp500", "sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"]
DERIVED_FAMILIES = ["marketwide", "value", "industry", "etf", "country", "indices"]


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


def safe_float(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        x = float(x)
        if not np.isfinite(x):
            return None
        return x
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
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    if df is None or len(df.columns) == 0:
        raise EmptyDataError(f"No columns parsed from: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
    for col in ["mktcap", "ICC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["bm"] = pd.to_numeric(df["bm"], errors="coerce") if "bm" in df.columns else np.nan
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "sector" not in df.columns:
        df["sector"] = None
    if "name" not in df.columns:
        df["name"] = None
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
    keep = {}
    for path, df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        date_value = str(df["date"].dropna().iloc[0])
        keep[(meta["universe"], date_value)] = (path, df)
    out = list(keep.values())
    out.sort(key=lambda x: (parse_snapshot_meta(x[0])["universe"], str(x[1]["date"].iloc[0]), parse_snapshot_meta(x[0])["rerun"], x[0].name))
    return out


def build_market_history(valid_snapshots: list[tuple[Path, pd.DataFrame]], universe_label: str) -> pd.DataFrame:
    rows = []
    for path, df in valid_snapshots:
        rows.append(
            {
                "date": str(df["date"].iloc[0]),
                "universe": universe_label,
                "vw_icc": weighted_mean(df["ICC"], df["mktcap"]),
                "ew_icc": float(df["ICC"].mean()),
                "n_firms": int(len(df)),
                "source_file": str(path.relative_to(REPO)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "universe", "vw_icc", "ew_icc", "n_firms", "source_file"])
    out = pd.DataFrame(rows).sort_values("date").drop_duplicates(["date", "universe"], keep="last").reset_index(drop=True)
    return out


def build_value_history(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, raw_df in valid_snapshots:
        df = raw_df[
            raw_df["bm"].notna()
            & np.isfinite(raw_df["bm"])
            & (raw_df["bm"] > 0)
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
        df["bm_bucket"] = pd.cut(df["bm"], bins=[-np.inf, bm30, bm70, np.inf], labels=["L", "M", "H"], include_lowest=True).astype(str)
        df["portfolio"] = df["size_bucket"] + "/" + df["bm_bucket"]
        pt = (
            df.groupby("portfolio", dropna=False)
            .apply(lambda g: pd.Series({"vw_icc": weighted_mean(g["ICC"], g["mktcap"]), "n_firms": int(len(g))}))
            .reset_index()
        )
        d = dict(zip(pt["portfolio"], pt["vw_icc"]))
        icc_sl, icc_bl = d.get("S/L", np.nan), d.get("B/L", np.nan)
        icc_sh, icc_bh = d.get("S/H", np.nan), d.get("B/H", np.nan)
        growth_icc = np.nan if (pd.isna(icc_sl) or pd.isna(icc_bl)) else float((icc_sl + icc_bl) / 2.0)
        value_icc = np.nan if (pd.isna(icc_sh) or pd.isna(icc_bh)) else float((icc_sh + icc_bh) / 2.0)
        ivp_bm = np.nan if (pd.isna(value_icc) or pd.isna(growth_icc)) else float(value_icc - growth_icc)
        rows.append(
            {
                "date": str(df["date"].iloc[0]),
                "value_icc": value_icc,
                "growth_icc": growth_icc,
                "ivp_bm": ivp_bm,
                "n_firms": int(len(df)),
                "source_file": str(path.relative_to(REPO)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "value_icc", "growth_icc", "ivp_bm", "n_firms", "source_file"])
    return pd.DataFrame(rows).sort_values("date").drop_duplicates(["date"], keep="last").reset_index(drop=True)


def build_industry_daily(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    daily_tables = {}
    series_rows = []
    for path, df in valid_snapshots:
        g = df[df["sector"].notna()].groupby("sector", dropna=False).apply(
            lambda x: pd.Series(
                {
                    "vw_icc": weighted_mean(x["ICC"], x["mktcap"]),
                    "ew_icc": float(x["ICC"].mean()),
                    "n_firms": int(len(x)),
                    "total_mktcap": float(x["mktcap"].sum()),
                }
            )
        ).reset_index()
        if g.empty:
            continue
        date = str(df["date"].iloc[0])
        g.insert(0, "date", date)
        daily_tables[date] = g.sort_values(["total_mktcap", "sector"], ascending=[False, True]).reset_index(drop=True)
        series_rows.append(
            {
                "date": date,
                "median_icc": float(g["vw_icc"].median()),
                "mean_icc": float(g["vw_icc"].mean()),
                "n_groups": int(len(g)),
            }
        )
    series = pd.DataFrame(series_rows).sort_values("date").reset_index(drop=True) if series_rows else pd.DataFrame(columns=["date","median_icc","mean_icc","n_groups"])
    return daily_tables, series


def read_csv_config(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].copy()


def get_top_holdings(etf_ticker: str) -> pd.DataFrame:
    t = yf.Ticker(etf_ticker)
    fd = getattr(t, "funds_data", None)
    if fd is None:
        raise ValueError("funds_data unavailable")
    h = getattr(fd, "top_holdings", None)
    if h is None:
        raise ValueError("top_holdings unavailable")
    if not isinstance(h, pd.DataFrame):
        h = pd.DataFrame(h)
    if h.empty:
        raise ValueError("top_holdings empty")
    cols = {str(c).lower(): c for c in h.columns}
    symbol_col = None
    for cand in ["symbol", "holding", "ticker"]:
        if cand in cols:
            symbol_col = cols[cand]
            break
    if symbol_col is None:
        symbol_col = h.columns[0]
    weight_col = None
    for cand in ["holdingpercent", "holding_percent", "weight", "percent", "pct"]:
        if cand in cols:
            weight_col = cols[cand]
            break
    if weight_col is None:
        # choose last numeric-ish column
        for c in reversed(list(h.columns)):
            vals = pd.to_numeric(h[c], errors="coerce")
            if vals.notna().sum() >= max(1, len(h)//2):
                weight_col = c
                break
    if weight_col is None:
        raise ValueError("weight column not found")
    out = pd.DataFrame({"symbol": h[symbol_col].astype(str).str.upper().str.strip(), "raw_weight": pd.to_numeric(h[weight_col], errors="coerce")})
    out = out[out["symbol"].notna() & out["raw_weight"].notna()].copy()
    out["weight"] = out["raw_weight"] / 100.0 if out["raw_weight"].max() > 1.5 else out["raw_weight"]
    out = out[out["weight"] > 0].copy()
    if out.empty:
        raise ValueError("no positive weights")
    out["weight"] = out["weight"] / out["weight"].sum()
    return out[["symbol", "weight"]]


def get_holdings_cache(cfg: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cache = {}
    for _, row in cfg.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        try:
            cache[ticker] = get_top_holdings(ticker)
        except Exception as e:
            print(f"[build_docs_data] holdings unavailable for {ticker}: {type(e).__name__}: {e}")
    return cache


def build_holdings_daily(cfg: pd.DataFrame, holdings_cache: dict[str, pd.DataFrame], valid_snapshots: list[tuple[Path, pd.DataFrame]], kind: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    daily_tables = {}
    series_rows = []
    for path, df in valid_snapshots:
        base = df[["ticker", "ICC", "mktcap", "sector", "name"]].copy()
        date = str(df["date"].iloc[0])
        rows = []
        for _, r in cfg.iterrows():
            ticker = str(r.get("ticker", "")).upper().strip()
            if not ticker or ticker not in holdings_cache:
                continue
            label = str(r.get("label", ticker))
            category = str(r.get("category", "")) if "category" in r.index else ""
            country = str(r.get("country", "")) if "country" in r.index else ""
            h = holdings_cache[ticker]
            merged = h.merge(base, left_on="symbol", right_on="ticker", how="left")
            matched = merged["ICC"].notna()
            coverage = float(merged.loc[matched, "weight"].sum()) if len(merged) else 0.0
            icc = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"]) if matched.sum() else np.nan
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "label": label,
                    "category": category,
                    "country": country,
                    "vw_icc": icc,
                    "coverage_weight": coverage,
                    "n_holdings": int(len(h)),
                    "n_matched": int(matched.sum()),
                    "status": "ok" if matched.sum() else "no_match",
                }
            )
        if not rows:
            continue
        table = pd.DataFrame(rows)
        daily_tables[date] = table
        good = table["vw_icc"].notna()
        if good.sum():
            series_rows.append(
                {
                    "date": date,
                    "median_icc": float(table.loc[good, "vw_icc"].median()),
                    "mean_icc": float(table.loc[good, "vw_icc"].mean()),
                    "n_groups": int(good.sum()),
                }
            )
    series = pd.DataFrame(series_rows).sort_values("date").reset_index(drop=True) if series_rows else pd.DataFrame(columns=["date","median_icc","mean_icc","n_groups"])
    return daily_tables, series


def build_indices_data(all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    history_rows = []
    daily_tables = {}
    all_dates = set()
    for u in INDEX_UNIVERSES:
        hist = build_market_history(all_valid_by_universe.get(u, []), u)
        if hist.empty:
            continue
        history_rows.append(hist)
        all_dates.update(hist["date"].tolist())
    history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame(columns=["date","universe","vw_icc","ew_icc","n_firms","source_file"])
    history = history.sort_values(["date","universe"]).reset_index(drop=True)
    for date, sub in history.groupby("date"):
        daily_tables[str(date)] = sub.sort_values("universe").reset_index(drop=True)
    return history, daily_tables


def month_end_rows(df: pd.DataFrame, value_cols: list[str], label_map: dict[str, str] | None = None, date_col: str = "date") -> tuple[list[str], list[dict]]:
    if df is None or df.empty:
        return [], []
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col])
    x["yyyymm"] = x[date_col].dt.strftime("%Y-%m")
    month_last = (
        x.sort_values(date_col)
         .groupby("yyyymm", as_index=False)
         .tail(1)
         .sort_values(date_col)
    )
    month_last = month_last.tail(3).reset_index(drop=True)
    cols = month_last["yyyymm"].tolist()
    rows = []
    for col in value_cols:
        rows.append({"label": label_map.get(col, col) if label_map else col, "values": [safe_float(v) for v in month_last[col].tolist()]})
    return cols, rows


def ensure_clean_download_dirs():
    if FAMILY_DIR.exists():
        shutil.rmtree(FAMILY_DIR)
    if ARCHIVE_DIR.exists():
        shutil.rmtree(ARCHIVE_DIR)
    FAMILY_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def write_daily_family_files(family: str, daily_tables: dict[str, pd.DataFrame]) -> list[dict]:
    entries = []
    fam_root = FAMILY_DIR / family
    for date, df in sorted(daily_tables.items()):
        d = pd.to_datetime(date)
        year = d.strftime("%Y")
        month = d.strftime("%m")
        out_dir = fam_root / year / month
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{family}_{date}.csv"
        path = out_dir / filename
        df.to_csv(path, index=False)
        entries.append(
            {
                "family": family,
                "date": date,
                "year": year,
                "month": month,
                "label": filename,
                "download_path": f"./data/downloads/families/{family}/{year}/{month}/{filename}",
            }
        )
    return entries


def build_archives_for_family(family: str, entries: list[dict]) -> dict:
    family_archives = {"family": family, "years": []}
    if not entries:
        return family_archives

    grouped_year = defaultdict(list)
    grouped_month = defaultdict(list)
    for e in entries:
        grouped_year[e["year"]].append(e)
        grouped_month[(e["year"], e["month"])].append(e)

    for year in sorted(grouped_year.keys(), reverse=True):
        year_entries = grouped_year[year]
        y_dir = ARCHIVE_DIR / family / year
        y_dir.mkdir(parents=True, exist_ok=True)

        year_zip = y_dir / f"{family}_{year}_all.zip"
        with zipfile.ZipFile(year_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for e in sorted(year_entries, key=lambda x: x["date"]):
                rel = e["download_path"].replace("./data/downloads/", "")
                src = DOWNLOAD_DIR / rel
                if src.exists():
                    zf.write(src, arcname=f"{family}/{year}/{src.name}")

        months_payload = []
        months = sorted({e["month"] for e in year_entries}, reverse=True)
        for month in months:
            month_entries = grouped_month[(year, month)]
            m_dir = y_dir / month
            m_dir.mkdir(parents=True, exist_ok=True)
            month_zip = m_dir / f"{family}_{year}_{month}_all.zip"
            with zipfile.ZipFile(month_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for e in sorted(month_entries, key=lambda x: x["date"]):
                    rel = e["download_path"].replace("./data/downloads/", "")
                    src = DOWNLOAD_DIR / rel
                    if src.exists():
                        zf.write(src, arcname=f"{family}/{year}/{month}/{src.name}")
            months_payload.append(
                {
                    "month": month,
                    "download_all": f"./data/downloads/archives/{family}/{year}/{month}/{month_zip.name}",
                    "days": sorted(month_entries, key=lambda x: x["date"], reverse=True),
                }
            )

        family_archives["years"].append(
            {
                "year": year,
                "download_all": f"./data/downloads/archives/{family}/{year}/{year_zip.name}",
                "months": months_payload,
            }
        )
    return family_archives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ensure_clean_download_dirs()

    all_valid = dedup_valid_snapshots(get_valid_snapshots(find_all_snapshots(None)))
    if not all_valid:
        raise RuntimeError("No valid snapshots found.")
    all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]] = defaultdict(list)
    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        if meta:
            all_valid_by_universe[meta["universe"]].append((path, df))

    usall_valid = all_valid_by_universe.get("usall", [])
    sp500_valid = all_valid_by_universe.get("sp500", [])
    if not usall_valid:
        raise RuntimeError("No valid usall snapshots found.")

    latest_usall_path, latest_usall = usall_valid[-1]
    latest_date = str(latest_usall["date"].iloc[0])

    market_all = build_market_history(usall_valid, "usall")
    market_sp500 = build_market_history(sp500_valid, "sp500")
    marketwide_daily = {}
    for date in sorted(set(market_all["date"]).union(set(market_sp500["date"]))):
        rows = []
        sub = market_all[market_all["date"] == date]
        if not sub.empty:
            rows.append({"date": date, "market": "All U.S.", "vw_icc": sub.iloc[0]["vw_icc"], "ew_icc": sub.iloc[0]["ew_icc"], "n_firms": sub.iloc[0]["n_firms"]})
        sub2 = market_sp500[market_sp500["date"] == date]
        if not sub2.empty:
            rows.append({"date": date, "market": "S&P 500", "vw_icc": sub2.iloc[0]["vw_icc"], "ew_icc": sub2.iloc[0]["ew_icc"], "n_firms": sub2.iloc[0]["n_firms"]})
        if rows:
            marketwide_daily[date] = pd.DataFrame(rows)

    value_hist = build_value_history(usall_valid)
    value_daily = {}
    for _, row in value_hist.iterrows():
        value_daily[str(row["date"])] = pd.DataFrame([row])

    industry_daily, industry_series = build_industry_daily(usall_valid)

    etf_cfg = read_csv_config(CONFIG_DIR / "etfs.csv", ["ticker", "label", "category"])
    country_cfg = read_csv_config(CONFIG_DIR / "country_etfs.csv", ["ticker", "label", "country"])
    etf_holdings = get_holdings_cache(etf_cfg)
    country_holdings = get_holdings_cache(country_cfg)
    etf_daily, etf_series = build_holdings_daily(etf_cfg, etf_holdings, usall_valid, "etf")
    country_daily, country_series = build_holdings_daily(country_cfg, country_holdings, usall_valid, "country")

    indices_hist, indices_daily = build_indices_data(all_valid_by_universe)

    # write daily family files + archives
    family_entries = {}
    family_entries["marketwide"] = write_daily_family_files("marketwide", marketwide_daily)
    family_entries["value"] = write_daily_family_files("value", value_daily)
    family_entries["industry"] = write_daily_family_files("industry", industry_daily)
    family_entries["etf"] = write_daily_family_files("etf", etf_daily)
    family_entries["country"] = write_daily_family_files("country", country_daily)
    family_entries["indices"] = write_daily_family_files("indices", indices_daily)

    download_tree = {fam: build_archives_for_family(fam, entries) for fam, entries in family_entries.items()}

    # aggregate CSV downloads
    market_all.to_csv(DOWNLOAD_DIR / "marketwide_all_history.csv", index=False)
    market_sp500.to_csv(DOWNLOAD_DIR / "marketwide_sp500_history.csv", index=False)
    value_hist.to_csv(DOWNLOAD_DIR / "value_history.csv", index=False)
    industry_series.to_csv(DOWNLOAD_DIR / "industry_overview_history.csv", index=False)
    etf_series.to_csv(DOWNLOAD_DIR / "etf_overview_history.csv", index=False)
    country_series.to_csv(DOWNLOAD_DIR / "country_overview_history.csv", index=False)
    indices_hist.to_csv(DOWNLOAD_DIR / "indices_history.csv", index=False)

    latest_industry = industry_daily.get(latest_date, pd.DataFrame())
    latest_etf = etf_daily.get(latest_date, pd.DataFrame())
    latest_country = country_daily.get(latest_date, pd.DataFrame())
    latest_indices = indices_daily.get(latest_date, pd.DataFrame())

    if not latest_industry.empty:
        latest_industry.to_csv(DOWNLOAD_DIR / "industry_latest.csv", index=False)
    if not latest_etf.empty:
        latest_etf.to_csv(DOWNLOAD_DIR / "etf_latest.csv", index=False)
    if not latest_country.empty:
        latest_country.to_csv(DOWNLOAD_DIR / "country_latest.csv", index=False)
    if not latest_indices.empty:
        latest_indices.to_csv(DOWNLOAD_DIR / "indices_latest.csv", index=False)

    # overview monthly summary
    cols_all, rows_all = month_end_rows(market_all, ["vw_icc"], {"vw_icc": "All U.S. market"})
    cols_sp500, rows_sp500 = month_end_rows(market_sp500, ["vw_icc"], {"vw_icc": "S&P 500"})
    cols_value, rows_value = month_end_rows(value_hist, ["value_icc", "growth_icc", "ivp_bm"], {"value_icc": "Value ICC", "growth_icc": "Growth ICC", "ivp_bm": "IVP (B/M)"})
    cols_ind, rows_ind = month_end_rows(industry_series, ["median_icc"], {"median_icc": "Industry median ICC"})
    cols_etf, rows_etf = month_end_rows(etf_series, ["median_icc"], {"median_icc": "ETF median ICC"})
    cols_country, rows_country = month_end_rows(country_series, ["median_icc"], {"median_icc": "Country median ICC"})

    # unify last 3 month columns
    all_months = sorted(set(cols_all + cols_sp500 + cols_value + cols_ind + cols_etf + cols_country))
    all_months = all_months[-3:]

    def align_rows(cols, rows):
        if not cols:
            return []
        idx = {c:i for i,c in enumerate(cols)}
        out = []
        for r in rows:
            vals = []
            for m in all_months:
                vals.append(r["values"][idx[m]] if m in idx else None)
            out.append({"label": r["label"], "values": vals})
        return out

    latest_daily = {
        "all_market_vw_icc": safe_float(market_all.iloc[-1]["vw_icc"]) if not market_all.empty else None,
        "all_market_ew_icc": safe_float(market_all.iloc[-1]["ew_icc"]) if not market_all.empty else None,
        "all_n_firms": safe_int(market_all.iloc[-1]["n_firms"]) if not market_all.empty else None,
        "sp500_vw_icc": safe_float(market_sp500.iloc[-1]["vw_icc"]) if not market_sp500.empty else None,
        "sp500_ew_icc": safe_float(market_sp500.iloc[-1]["ew_icc"]) if not market_sp500.empty else None,
        "sp500_n_firms": safe_int(market_sp500.iloc[-1]["n_firms"]) if not market_sp500.empty else None,
        "value_icc": safe_float(value_hist.iloc[-1]["value_icc"]) if not value_hist.empty else None,
        "growth_icc": safe_float(value_hist.iloc[-1]["growth_icc"]) if not value_hist.empty else None,
        "ivp_bm": safe_float(value_hist.iloc[-1]["ivp_bm"]) if not value_hist.empty else None,
        "industry_median_icc": safe_float(industry_series.iloc[-1]["median_icc"]) if not industry_series.empty else None,
        "etf_median_icc": safe_float(etf_series.iloc[-1]["median_icc"]) if not etf_series.empty else None,
        "country_median_icc": safe_float(country_series.iloc[-1]["median_icc"]) if not country_series.empty else None,
    }

    overview_monthly_rows = []
    for rows in [align_rows(cols_all, rows_all), align_rows(cols_sp500, rows_sp500), align_rows(cols_value, rows_value), align_rows(cols_ind, rows_ind), align_rows(cols_etf, rows_etf), align_rows(cols_country, rows_country)]:
        overview_monthly_rows.extend(rows)

    overview_downloads = {
        "marketwide": [
            {"label": "All U.S. market history", "path": "./data/downloads/marketwide_all_history.csv"},
            {"label": "S&P 500 history", "path": "./data/downloads/marketwide_sp500_history.csv"},
        ],
        "value": [{"label": "Value / Growth / IVP history", "path": "./data/downloads/value_history.csv"}],
        "industry": [{"label": "Industry overview history", "path": "./data/downloads/industry_overview_history.csv"}, {"label": "Industry latest table", "path": "./data/downloads/industry_latest.csv"}],
        "etf": [{"label": "ETF overview history", "path": "./data/downloads/etf_overview_history.csv"}, {"label": "ETF latest table", "path": "./data/downloads/etf_latest.csv"}],
        "country": [{"label": "Country overview history", "path": "./data/downloads/country_overview_history.csv"}, {"label": "Country latest table", "path": "./data/downloads/country_latest.csv"}],
        "indices": [{"label": "Indices history", "path": "./data/downloads/indices_history.csv"}, {"label": "Indices latest table", "path": "./data/downloads/indices_latest.csv"}],
    }

    # write JSON payloads
    write_json(
        DOCS_DATA_DIR / "overview.json",
        {
            "asof_date": latest_date,
            "latest_daily": latest_daily,
            "three_months": {"columns": all_months, "rows": overview_monthly_rows},
            "downloads": overview_downloads,
        },
    )

    write_json(
        DOCS_DATA_DIR / "marketwide.json",
        {
            "asof_date": latest_date,
            "all_history": market_all.to_dict(orient="records"),
            "sp500_history": market_sp500.to_dict(orient="records"),
            "three_months": {
                "columns": all_months,
                "rows": align_rows(cols_all, rows_all) + align_rows(cols_sp500, rows_sp500),
            },
        },
    )

    write_json(
        DOCS_DATA_DIR / "value.json",
        {
            "asof_date": latest_date,
            "history": value_hist.to_dict(orient="records"),
            "three_months": {"columns": all_months, "rows": align_rows(cols_value, rows_value)},
        },
    )

    write_json(
        DOCS_DATA_DIR / "industry.json",
        {
            "asof_date": latest_date,
            "latest": latest_industry.to_dict(orient="records"),
            "overview_history": industry_series.to_dict(orient="records"),
            "three_months": {"columns": all_months, "rows": align_rows(cols_ind, rows_ind)},
        },
    )

    write_json(
        DOCS_DATA_DIR / "etf.json",
        {
            "asof_date": latest_date,
            "latest": latest_etf.to_dict(orient="records"),
            "overview_history": etf_series.to_dict(orient="records"),
            "three_months": {"columns": all_months, "rows": align_rows(cols_etf, rows_etf)},
        },
    )

    write_json(
        DOCS_DATA_DIR / "country.json",
        {
            "asof_date": latest_date,
            "latest": latest_country.to_dict(orient="records"),
            "overview_history": country_series.to_dict(orient="records"),
            "three_months": {"columns": all_months, "rows": align_rows(cols_country, rows_country)},
        },
    )

    write_json(
        DOCS_DATA_DIR / "indices.json",
        {
            "asof_date": latest_date,
            "latest": latest_indices.to_dict(orient="records"),
            "history": indices_hist.to_dict(orient="records"),
        },
    )

    write_json(DOCS_DATA_DIR / "downloads_catalog.json", {"asof_date": latest_date, "families": download_tree, "aggregate": overview_downloads})

    print(f"[build_docs_data] asof_date={latest_date}")
    print(f"[build_docs_data] usall snapshots={len(usall_valid)}")
    print(f"[build_docs_data] indices available={[u for u in INDEX_UNIVERSES if len(all_valid_by_universe.get(u, [])) > 0]}")
    print("[build_docs_data] docs/data generated successfully")


if __name__ == "__main__":
    main()
