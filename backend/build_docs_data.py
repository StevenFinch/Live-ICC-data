
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pandas.errors import EmptyDataError

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
CONFIG_DIR = REPO / "config"
DOCS_DIR = REPO / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"
DOWNLOAD_DIR = DOCS_DATA_DIR / "downloads"
RAW_DOWNLOAD_DIR = DOWNLOAD_DIR / "raw"
FAMILY_DOWNLOAD_DIR = DOWNLOAD_DIR / "family"
LIBRARY_DIR = DATA_DIR / "library"

SNAPSHOT_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)
BAD_SYM = re.compile(r"[^A-Z\.\-]")
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

INDEX_UNIVERSES = ["sp500", "sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"]
RAW_GROUPS = {
    "usall": {"usall"},
    "sp500": {"sp500"},
    "other_indices": {"sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"},
}


@dataclass
class Snapshot:
    path: Path
    universe: str
    date: str
    yyyymm: str
    rerun: int
    df: pd.DataFrame


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
    if pd.isna(x):
        return None
    try:
        x = float(x)
        if not np.isfinite(x):
            return None
        return x
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


def parse_snapshot_meta(path: Path) -> dict | None:
    m = SNAPSHOT_RE.match(path.name)
    if not m:
        return None
    year = m.group("year")
    mmdd = m.group("mmdd")
    rerun = int(m.group("rerun") or 0)
    yyyymmdd = f"{year}{mmdd}"
    date = datetime.strptime(yyyymmdd, "%Y%m%d").date().isoformat()
    return {
        "universe": m.group("universe"),
        "yyyymm": f"{year}{mmdd[:2]}",
        "date": date,
        "rerun": rerun,
        "yyyymmdd": yyyymmdd,
    }


def load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    if df is None or len(df.columns) == 0:
        raise EmptyDataError(f"No columns parsed from file: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")
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


def clean_snapshot_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["ticker"].notna()
        & df["date"].notna()
        & df["mktcap"].notna()
        & df["ICC"].notna()
        & np.isfinite(df["mktcap"])
        & np.isfinite(df["ICC"])
        & (df["mktcap"] > 0)
    ].copy()


def find_valid_snapshots() -> list[Snapshot]:
    keep: dict[tuple[str, str], Snapshot] = {}
    for path in sorted(DATA_DIR.glob("*/*.csv")):
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        try:
            df = clean_snapshot_df(load_snapshot(path))
        except Exception as exc:
            print(f"[build_docs_data] skip {path}: {type(exc).__name__}: {exc}")
            continue
        if df.empty:
            continue
        snap = Snapshot(
            path=path,
            universe=meta["universe"],
            date=meta["date"],
            yyyymm=meta["yyyymm"],
            rerun=meta["rerun"],
            df=df,
        )
        key = (snap.universe, snap.date)
        prev = keep.get(key)
        if prev is None or snap.rerun >= prev.rerun:
            keep[key] = snap
    out = list(keep.values())
    out.sort(key=lambda s: (s.universe, s.date, s.rerun, s.path.name))
    return out


def latest_snapshot(snaps: list[Snapshot], universe: str) -> Snapshot:
    candidates = [s for s in snaps if s.universe == universe]
    if not candidates:
        raise RuntimeError(f"No valid snapshots for {universe}")
    return sorted(candidates, key=lambda s: (s.date, s.rerun))[-1]


def read_csv_config(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns].copy()


def snapshot_market_row(snap: Snapshot, family: str, label: str | None = None) -> dict:
    return {
        "family": family,
        "label": label or family,
        "date": snap.date,
        "value": weighted_mean(snap.df["ICC"], snap.df["mktcap"]),
        "method": "ICC calculation",
        "source": "Constituent ICC aggregation",
        "n_items": int(len(snap.df)),
        "coverage_weight": None,
        "status": "available",
    }


def build_marketwide_daily(snaps: list[Snapshot]) -> pd.DataFrame:
    rows = []
    for s in snaps:
        if s.universe == "usall":
            rows.append(snapshot_market_row(s, "all_market", "All market"))
        elif s.universe == "sp500":
            rows.append(snapshot_market_row(s, "sp500", "S&P 500"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["family", "date"]).reset_index(drop=True)


def build_value_rows(snap: Snapshot) -> list[dict]:
    df = snap.df.copy()
    df = df[
        df["bm"].notna()
        & np.isfinite(df["bm"])
        & (df["bm"] > 0)
    ].copy()
    if df.empty:
        return []

    q_lo = float(df["ICC"].quantile(0.005))
    q_hi = float(df["ICC"].quantile(0.995))
    df = df[(df["ICC"] >= q_lo) & (df["ICC"] <= q_hi)].copy()
    if df.empty:
        return []

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
    ).astype(str)
    df["portfolio"] = df["size_bucket"] + "/" + df["bm_bucket"]

    pt = (
        df.groupby("portfolio", dropna=False)
        .apply(lambda g: pd.Series({"value": weighted_mean(g["ICC"], g["mktcap"])}))
        .reset_index()
    )

    d = dict(zip(pt["portfolio"], pt["value"]))
    growth = np.nan if pd.isna(d.get("S/L")) or pd.isna(d.get("B/L")) else (d["S/L"] + d["B/L"]) / 2.0
    value = np.nan if pd.isna(d.get("S/H")) or pd.isna(d.get("B/H")) else (d["S/H"] + d["B/H"]) / 2.0
    ivp = np.nan if pd.isna(value) or pd.isna(growth) else value - growth

    rows = []
    for family, label, val in [
        ("value_icc", "Value ICC", value),
        ("growth_icc", "Growth ICC", growth),
        ("ivp_bm", "IVP (B/M)", ivp),
    ]:
        rows.append(
            {
                "family": family,
                "label": label,
                "date": snap.date,
                "value": val,
                "method": "ICC calculation",
                "source": "Constituent ICC aggregation",
                "n_items": int(len(df)),
                "coverage_weight": None,
                "status": "available",
            }
        )
    return rows


def build_value_daily(snaps: list[Snapshot]) -> pd.DataFrame:
    rows = []
    for s in snaps:
        if s.universe == "usall":
            rows.extend(build_value_rows(s))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["family", "date"]).reset_index(drop=True)


def build_industry_rows(snap: Snapshot) -> list[dict]:
    df = snap.df.copy()
    df = df[df["sector"].notna()].copy()
    if df.empty:
        return []
    out = (
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
    rows = []
    for _, r in out.iterrows():
        rows.append(
            {
                "family": str(r["sector"]),
                "label": str(r["sector"]),
                "date": snap.date,
                "value": r["value"],
                "method": "ICC calculation",
                "source": "Constituent ICC aggregation",
                "n_items": int(r["n_items"]),
                "coverage_weight": None,
                "status": "available",
            }
        )
    return rows


def build_industry_daily(snaps: list[Snapshot]) -> pd.DataFrame:
    rows = []
    for s in snaps:
        if s.universe == "usall":
            rows.extend(build_industry_rows(s))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["family", "date"]).reset_index(drop=True)


def build_indices_daily(snaps: list[Snapshot]) -> pd.DataFrame:
    rows = []
    for s in snaps:
        if s.universe in INDEX_UNIVERSES:
            label = {
                "sp500": "S&P 500",
                "sp100": "S&P 100",
                "dow30": "Dow 30",
                "ndx100": "Nasdaq-100",
                "sp400": "S&P 400",
                "sp600": "S&P 600",
                "rut1000": "Russell 1000",
            }.get(s.universe, s.universe)
            rows.append(snapshot_market_row(s, s.universe, label))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["family", "date"]).reset_index(drop=True)


def _normalize_symbols(series: pd.Series) -> list[str]:
    vals = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )
    return [v for v in vals if v and not BAD_SYM.search(v)]


def parse_pe_from_text(text: str) -> float | None:
    text = text.replace(",", "")
    patterns = [
        r"P/E Ratio(?:[^0-9]{0,40})(\d+(?:\.\d+)?)",
        r"Price/Earnings Ratio FY1(?:[^0-9]{0,40})(\d+(?:\.\d+)?)",
        r"Price/Earnings(?:[^0-9]{0,40})(\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I | re.S)
        if m:
            try:
                pe = float(m.group(1))
                if np.isfinite(pe) and pe > 0:
                    return pe
            except Exception:
                pass
    return None


def get_text(url: str, timeout: int = 25) -> str:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def fetch_page_pe(url: str | None) -> float | None:
    if not url or str(url).strip() == "":
        return None
    try:
        text = get_text(str(url))
        return parse_pe_from_text(text)
    except Exception as exc:
        print(f"[build_docs_data] page fetch failed for {url}: {exc}")
        return None


def fetch_yfinance_pe(ticker: str) -> float | None:
    try:
        info = yf.Ticker(ticker).info or {}
        for key in ("forwardPE", "trailingPE"):
            pe = info.get(key)
            if pe is not None:
                pe = float(pe)
                if np.isfinite(pe) and pe > 0:
                    return pe
    except Exception as exc:
        print(f"[build_docs_data] yfinance info failed for {ticker}: {exc}")
    return None


def fetch_top_holdings(ticker: str) -> pd.DataFrame:
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
    cols = {str(c).lower(): c for c in df.columns}
    sym_col = None
    wt_col = None
    for key in ("symbol", "holding", "ticker"):
        if key in cols:
            sym_col = cols[key]
            break
    for key in ("holdingpercent", "holding_percent", "weight", "percent", "pct"):
        if key in cols:
            wt_col = cols[key]
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
        raise ValueError("could not identify weight column")
    out = pd.DataFrame(
        {
            "symbol": _normalize_symbols(df[sym_col]),
            "raw_weight": pd.to_numeric(df[wt_col], errors="coerce"),
        }
    )
    out = out[out["raw_weight"].notna()].copy()
    if out.empty:
        raise ValueError("no usable weights")
    out["weight"] = out["raw_weight"] / 100.0 if out["raw_weight"].max() > 1.5 else out["raw_weight"]
    out = out[out["weight"] > 0].copy()
    out["weight"] = out["weight"] / out["weight"].sum()
    return out[["symbol", "weight"]]


def compute_holdings_based_value(ticker: str, latest_usall: pd.DataFrame) -> tuple[float | None, float | None, int, str]:
    base = latest_usall[["ticker", "ICC"]].copy()
    try:
        holdings = fetch_top_holdings(ticker)
        merged = holdings.merge(base, left_on="symbol", right_on="ticker", how="left")
        matched = merged["ICC"].notna()
        coverage = float(merged.loc[matched, "weight"].sum()) if len(merged) else 0.0
        n_matched = int(matched.sum())
        if n_matched >= 3 and coverage >= 0.20:
            value = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"])
            return value, coverage, n_matched, "holdings-based"
        return None, coverage, n_matched, "holdings-insufficient"
    except Exception as exc:
        print(f"[build_docs_data] holdings failed for {ticker}: {exc}")
        return None, None, 0, "holdings-error"


def proxy_value_from_pe(pe: float) -> float:
    return 1.0 / pe


def build_proxy_family_daily(
    cfg: pd.DataFrame,
    latest_usall: pd.DataFrame,
    family_kind: str,
    date: str,
) -> pd.DataFrame:
    rows = []
    for _, row in cfg.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        label = str(row.get("label", ticker))
        url = row.get("url")
        allow_holdings = str(row.get("allow_holdings", "0")).strip() in {"1", "true", "True", "yes", "YES"}

        value = None
        coverage = None
        n_matched = None
        method = None
        source = None
        status = "unavailable"

        if allow_holdings:
            h_value, h_cov, h_n, h_source = compute_holdings_based_value(ticker, latest_usall)
            if h_value is not None:
                value = h_value
                coverage = h_cov
                n_matched = h_n
                method = "ICC calculation"
                source = h_source
                status = "available"

        if value is None:
            pe = fetch_page_pe(url)
            if pe is not None:
                value = proxy_value_from_pe(pe)
                method = "P/E estimate"
                source = "Provider page"
                status = "available"

        if value is None:
            pe = fetch_yfinance_pe(ticker)
            if pe is not None:
                value = proxy_value_from_pe(pe)
                method = "P/E estimate"
                source = "Yahoo Finance"
                status = "available"

        if family_kind == "etf":
            family = ticker
        else:
            family = str(row.get("country", ticker))

        rows.append(
            {
                "family": family,
                "ticker": ticker,
                "label": label,
                "date": date,
                "value": value,
                "method": method or "Unavailable",
                "source": source or "Unavailable",
                "n_items": safe_int(n_matched),
                "coverage_weight": safe_float(coverage),
                "status": status,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["family", "ticker"]).reset_index(drop=True)


def ensure_library_dirs() -> None:
    for family in ["marketwide", "value", "industry", "indices", "etf", "country"]:
        (LIBRARY_DIR / family / "daily").mkdir(parents=True, exist_ok=True)


def archive_family_daily(family: str, daily_df: pd.DataFrame) -> None:
    if daily_df.empty:
        return
    date = str(daily_df["date"].iloc[0])
    dt = datetime.strptime(date, "%Y-%m-%d")
    yyyymm = dt.strftime("%Y%m")
    tag = dt.strftime("%Y_%m%d")
    out_dir = LIBRARY_DIR / family / "daily" / yyyymm
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{family}_{tag}.csv"
    daily_df.to_csv(out_path, index=False)


def load_family_archives(family: str) -> pd.DataFrame:
    base = LIBRARY_DIR / family / "daily"
    rows = []
    for p in sorted(base.glob("*/*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df["__source_file"] = str(p.relative_to(REPO))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    for col in ["value", "coverage_weight"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "n_items" in out.columns:
        out["n_items"] = pd.to_numeric(out["n_items"], errors="coerce")
    sort_cols = [c for c in ["family", "ticker", "date", "__source_file"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    dedup_keys = [c for c in ["family", "ticker", "date"] if c in out.columns]
    out = out.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
    return out


def monthly_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out_rows = []
    series_keys = [c for c in ["family", "ticker", "label", "method", "source"] if c in df.columns]
    for _, g in df.groupby([c for c in ["family", "ticker"] if c in df.columns], dropna=False):
        g = g.copy()
        g["date_dt"] = pd.to_datetime(g["date"])
        g["yyyymm"] = g["date_dt"].dt.strftime("%Y-%m")
        g = g.sort_values("date_dt")
        monthly = g.groupby("yyyymm", dropna=False).tail(1)
        out_rows.append(monthly[df.columns])
    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=df.columns)
    out = out.sort_values([c for c in ["family", "ticker", "date"] if c in out.columns]).reset_index(drop=True)
    return out


def last_three_months_by_series(monthly_df: pd.DataFrame) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if monthly_df.empty:
        return out
    group_cols = ["family"]
    if "ticker" in monthly_df.columns:
        group_cols = ["family", "ticker"]
    for _, g in monthly_df.groupby(group_cols, dropna=False):
        g = g.sort_values("date", ascending=False).head(3).copy()
        if "family" in g.columns:
            family = str(g["family"].iloc[0])
        else:
            family = str(g["ticker"].iloc[0])
        out[family] = g.to_dict(orient="records")
    return out


def latest_by_series(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df.copy()
    group_cols = [c for c in ["family", "ticker"] if c in daily_df.columns]
    out = (
        daily_df.sort_values("date")
        .groupby(group_cols, dropna=False)
        .tail(1)
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return out


def build_overview_payload(data_map: dict[str, dict]) -> dict:
    rows = []

    def add_rows(family_key: str, latest_df: pd.DataFrame, monthly_df: pd.DataFrame):
        groups = last_three_months_by_series(monthly_df)
        for _, latest in latest_df.iterrows():
            family = str(latest.get("family") or latest.get("ticker"))
            monthly_rows = groups.get(family, [])
            padded = monthly_rows + [{}] * max(0, 3 - len(monthly_rows))
            rows.append(
                {
                    "family_key": family_key,
                    "family": family,
                    "label": latest.get("label", family),
                    "latest_daily": latest.get("value"),
                    "method": latest.get("method"),
                    "m1_date": padded[0].get("date"),
                    "m1_value": padded[0].get("value"),
                    "m2_date": padded[1].get("date"),
                    "m2_value": padded[1].get("value"),
                    "m3_date": padded[2].get("date"),
                    "m3_value": padded[2].get("value"),
                }
            )

    add_rows("marketwide", data_map["marketwide"]["latest"], data_map["marketwide"]["monthly"])
    add_rows("value premium", data_map["value"]["latest"], data_map["value"]["monthly"])
    add_rows("industry", data_map["industry"]["latest"], data_map["industry"]["monthly"])
    add_rows("ETF", data_map["etf"]["latest"], data_map["etf"]["monthly"])
    add_rows("country", data_map["country"]["latest"], data_map["country"]["monthly"])
    add_rows("indices", data_map["indices"]["latest"], data_map["indices"]["monthly"])

    rows = sorted(rows, key=lambda x: (str(x["family_key"]), str(x["family"])))
    asof = ""
    for family_data in data_map.values():
        latest = family_data["latest"]
        if not latest.empty:
            d = str(latest["date"].max())
            if d > asof:
                asof = d

    return {"asof_date": asof, "rows": rows}


def family_title_map() -> dict[str, str]:
    return {
        "marketwide": "Marketwide",
        "value": "Value Premium",
        "industry": "Industry",
        "etf": "ETF",
        "country": "Country",
        "indices": "Indices",
    }


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_zip(zip_path: Path, members: Iterable[Path], base_dir: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in members:
            if p.exists() and p.is_file():
                zf.write(p, arcname=str(p.relative_to(base_dir)))


def build_family_downloads(family: str, daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> dict:
    base = FAMILY_DOWNLOAD_DIR / family
    latest_df = latest_by_series(daily_df)
    latest_path = base / "latest.csv"
    daily_path = base / "daily_history.csv"
    monthly_path = base / "monthly_history.csv"
    write_csv(latest_path, latest_df)
    write_csv(daily_path, daily_df)
    write_csv(monthly_path, monthly_df)

    years = defaultdict(lambda: defaultdict(list))
    archive_dir = LIBRARY_DIR / family / "daily"
    for p in sorted(archive_dir.glob("*/*.csv")):
        m = re.search(r"_(\d{4})_(\d{4})\.csv$", p.name)
        if not m:
            continue
        year = m.group(1)
        month = m.group(2)[:2]
        date = datetime.strptime(f"{year}{m.group(2)}", "%Y%m%d").date().isoformat()
        rel = f"./data/downloads/family/{family}/archive/{year}{month}/{p.name}"
        target_dir = base / "archive" / f"{year}{month}"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target_dir / p.name)
        years[year][f"{year}{month}"].append(
            {
                "date": date,
                "download_path": rel,
                "filename": p.name,
            }
        )

    year_entries = []
    for year in sorted(years.keys(), reverse=True):
        month_entries = []
        year_files = []
        for yyyymm in sorted(years[year].keys(), reverse=True):
            files = sorted(years[year][yyyymm], key=lambda x: x["date"], reverse=True)
            month_dir = base / "archive" / yyyymm
            month_zip = base / "archive" / f"{yyyymm}.zip"
            month_members = list(month_dir.glob("*.csv"))
            make_zip(month_zip, month_members, month_dir.parent)
            month_entries.append(
                {
                    "yyyymm": yyyymm,
                    "download_all": f"./data/downloads/family/{family}/archive/{yyyymm}.zip",
                    "files": files,
                }
            )
            year_files.extend(month_members)

        year_zip = base / "archive" / f"{year}.zip"
        make_zip(year_zip, year_files, base / "archive")
        year_entries.append(
            {
                "year": year,
                "download_all": f"./data/downloads/family/{family}/archive/{year}.zip",
                "months": month_entries,
            }
        )

    return {
        "title": family_title_map()[family],
        "latest_csv": f"./data/downloads/family/{family}/latest.csv",
        "daily_history_csv": f"./data/downloads/family/{family}/daily_history.csv",
        "monthly_history_csv": f"./data/downloads/family/{family}/monthly_history.csv",
        "years": year_entries,
    }


def build_raw_downloads(valid_snaps: list[Snapshot]) -> dict:
    # Copy raw snapshots into docs/data/downloads/raw/<group>/<yyyymm>/
    group_years: dict[str, dict[str, dict[str, list[dict]]]] = {
        "usall": defaultdict(lambda: defaultdict(list)),
        "sp500": defaultdict(lambda: defaultdict(list)),
        "other_indices": defaultdict(lambda: defaultdict(list)),
    }

    for snap in valid_snaps:
        group = None
        for k, universes in RAW_GROUPS.items():
            if snap.universe in universes:
                group = k
                break
        if group is None:
            continue

        yyyymm = snap.yyyymm
        year = yyyymm[:4]
        target_dir = RAW_DOWNLOAD_DIR / group / yyyymm
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / snap.path.name
        shutil.copy2(snap.path, target)
        rel = f"./data/downloads/raw/{group}/{yyyymm}/{snap.path.name}"
        group_years[group][year][yyyymm].append(
            {
                "date": snap.date,
                "universe": snap.universe,
                "n_firms": int(len(snap.df)),
                "download_path": rel,
                "filename": snap.path.name,
            }
        )

    result = {}
    for group, years in group_years.items():
        year_entries = []
        for year in sorted(years.keys(), reverse=True):
            month_entries = []
            year_members = []
            for yyyymm in sorted(years[year].keys(), reverse=True):
                files = sorted(years[year][yyyymm], key=lambda x: (x["date"], x["filename"]), reverse=True)
                month_dir = RAW_DOWNLOAD_DIR / group / yyyymm
                month_members = list(month_dir.glob("*.csv"))
                month_zip = RAW_DOWNLOAD_DIR / group / f"{yyyymm}.zip"
                make_zip(month_zip, month_members, month_dir.parent)
                month_entries.append(
                    {
                        "yyyymm": yyyymm,
                        "download_all": f"./data/downloads/raw/{group}/{yyyymm}.zip",
                        "files": files,
                    }
                )
                year_members.extend(month_members)
            year_zip = RAW_DOWNLOAD_DIR / group / f"{year}.zip"
            make_zip(year_zip, year_members, RAW_DOWNLOAD_DIR / group)
            year_entries.append(
                {
                    "year": year,
                    "download_all": f"./data/downloads/raw/{group}/{year}.zip",
                    "months": month_entries,
                }
            )
        result[group] = year_entries
    return result


def build_json_payload_for_family(daily_df: pd.DataFrame, downloads: dict) -> dict:
    monthly_df = monthly_from_daily(daily_df)
    latest_df = latest_by_series(daily_df)
    return {
        "latest": json_safe(latest_df.to_dict(orient="records")),
        "daily": json_safe(daily_df.sort_values(["family", "date"]).to_dict(orient="records")),
        "monthly": json_safe(monthly_df.sort_values(["family", "date"]).to_dict(orient="records")),
        "monthly_groups": json_safe(last_three_months_by_series(monthly_df)),
        "downloads": json_safe(downloads),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ensure_library_dirs()

    valid_snaps = find_valid_snapshots()
    if not valid_snaps:
        raise RuntimeError("No valid snapshots found under data/YYYYMM/")

    latest_usall = latest_snapshot(valid_snaps, args.universe).df

    # Build latest daily family rows, then archive them.
    marketwide_latest = latest_by_series(build_marketwide_daily(valid_snaps))
    value_latest = latest_by_series(build_value_daily(valid_snaps))
    industry_latest = latest_by_series(build_industry_daily([latest_snapshot(valid_snaps, args.universe)]))
    indices_latest = latest_by_series(build_indices_daily(valid_snaps))
    etf_cfg = read_csv_config(CONFIG_DIR / "etfs.csv", ["ticker", "label", "category", "url", "allow_holdings"])
    country_cfg = read_csv_config(CONFIG_DIR / "country_etfs.csv", ["ticker", "label", "country", "url", "allow_holdings"])
    latest_date = latest_snapshot(valid_snaps, args.universe).date
    etf_latest = build_proxy_family_daily(etf_cfg, latest_usall, "etf", latest_date)
    country_latest = build_proxy_family_daily(country_cfg, latest_usall, "country", latest_date)

    # Archive latest rows so history can start accumulating from now on.
    archive_family_daily("marketwide", marketwide_latest)
    archive_family_daily("value", value_latest)
    archive_family_daily("industry", industry_latest)
    archive_family_daily("indices", indices_latest)
    archive_family_daily("etf", etf_latest)
    archive_family_daily("country", country_latest)

    # Load archived families to build history and downloads.
    family_daily = {
        "marketwide": load_family_archives("marketwide"),
        "value": load_family_archives("value"),
        "industry": load_family_archives("industry"),
        "indices": load_family_archives("indices"),
        "etf": load_family_archives("etf"),
        "country": load_family_archives("country"),
    }

    family_monthly = {k: monthly_from_daily(v) for k, v in family_daily.items()}
    family_downloads = {k: build_family_downloads(k, family_daily[k], family_monthly[k]) for k in family_daily}
    raw_downloads = build_raw_downloads(valid_snaps)

    data_map = {}
    for family in family_daily:
        payload = build_json_payload_for_family(family_daily[family], family_downloads[family])
        data_map[family] = {
            "latest": latest_by_series(family_daily[family]),
            "monthly": family_monthly[family],
        }
        write_json(DOCS_DATA_DIR / f"{family}.json", payload)

    write_json(
        DOCS_DATA_DIR / "downloads_catalog.json",
        {
            "families": family_downloads,
            "raw": raw_downloads,
        },
    )
    write_json(DOCS_DATA_DIR / "overview.json", build_overview_payload(data_map))

    print("[build_docs_data] completed successfully")


if __name__ == "__main__":
    main()
