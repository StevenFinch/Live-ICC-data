from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pandas.errors import EmptyDataError

try:
    from backend.icc_market_live import get_live_panel
except Exception:
    get_live_panel = None  # type: ignore

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DERIVED_DIR = DATA_DIR / "derived"
ETF_DERIVED = DERIVED_DIR / "etf"
COUNTRY_DERIVED = DERIVED_DIR / "country_adr"
DOCS_DATA = REPO / "docs" / "data"
SNAPSHOT_RE = re.compile(r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$")
BAD_SYM = re.compile(r"[^A-Z\.\-]")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

DEFAULT_ETFS = [
    ("SPY", "SPDR S&P 500 ETF Trust", "US broad market"),
    ("IVV", "iShares Core S&P 500 ETF", "US broad market"),
    ("VOO", "Vanguard S&P 500 ETF", "US broad market"),
    ("VTI", "Vanguard Total Stock Market ETF", "US broad market"),
    ("QQQ", "Invesco QQQ Trust", "US growth / Nasdaq"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF", "US blue chip"),
    ("IWM", "iShares Russell 2000 ETF", "US small cap"),
    ("MDY", "SPDR S&P MidCap 400 ETF", "US mid cap"),
    ("IJR", "iShares Core S&P Small-Cap ETF", "US small cap"),
    ("XLK", "Technology Select Sector SPDR Fund", "Sector"),
    ("XLF", "Financial Select Sector SPDR Fund", "Sector"),
    ("XLE", "Energy Select Sector SPDR Fund", "Sector"),
    ("XLV", "Health Care Select Sector SPDR Fund", "Sector"),
    ("XLY", "Consumer Discretionary Select Sector SPDR Fund", "Sector"),
    ("XLI", "Industrial Select Sector SPDR Fund", "Sector"),
    ("XLP", "Consumer Staples Select Sector SPDR Fund", "Sector"),
    ("XLU", "Utilities Select Sector SPDR Fund", "Sector"),
    ("XLB", "Materials Select Sector SPDR Fund", "Sector"),
    ("XLRE", "Real Estate Select Sector SPDR Fund", "Sector"),
    ("XLC", "Communication Services Select Sector SPDR Fund", "Sector"),
]


def parse_snapshot_meta(path: Path) -> dict[str, Any] | None:
    m = SNAPSHOT_RE.match(path.name)
    if not m:
        return None
    year = m.group("year")
    mmdd = m.group("mmdd")
    return {
        "universe": m.group("universe"),
        "year": year,
        "mmdd": mmdd,
        "yyyymm": f"{year}{mmdd[:2]}",
        "date": datetime.strptime(f"{year}{mmdd}", "%Y%m%d").date().isoformat(),
        "rerun": int(m.group("rerun") or 0),
    }


def load_snapshot(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    df["ICC"] = pd.to_numeric(df["ICC"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.dropna(subset=["ticker", "mktcap", "ICC", "date"])


def latest_valid_snapshot(universe: str) -> tuple[Path, pd.DataFrame, str]:
    candidates: list[tuple[tuple[str, int, str], Path]] = []
    for path in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(path)
        if not meta or meta["universe"] != universe:
            continue
        try:
            if path.stat().st_size <= 0:
                continue
        except FileNotFoundError:
            continue
        candidates.append(((meta["date"], meta["rerun"], path.name), path))
    if not candidates:
        raise FileNotFoundError(f"No valid snapshot files found for universe={universe}")
    candidates.sort(key=lambda x: x[0])
    for _key, path in reversed(candidates):
        try:
            df = load_snapshot(path)
            if not df.empty:
                date_value = str(df["date"].dropna().iloc[0])
                return path, df, date_value
        except Exception as exc:
            print(f"[ensure_daily_derived] skip invalid snapshot {path}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"No readable non-empty snapshot found for universe={universe}")


def normalize_ticker(x: Any) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper().replace(".", "-")
    s = re.sub(r"\s+", "", s)
    if not s or BAD_SYM.search(s):
        return None
    return s


def parse_weight(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().replace(",", "")
    if not s or s in {"-", "--", "nan"}:
        return None
    pct = "%" in s
    s = re.sub(r"[^0-9\.\-]", "", s)
    if not s:
        return None
    try:
        val = float(s)
    except Exception:
        return None
    if pct or val > 1.5:
        val = val / 100.0
    if val <= 0 or not np.isfinite(val):
        return None
    return val


def request_text(url: str, timeout: int = 30, retries: int = 3) -> str:
    headers = {"User-Agent": USER_AGENT}
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_err = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def request_bytes(url: str, timeout: int = 45, retries: int = 3) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            last_err = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    cols = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in cols}
    sym_col = None
    weight_col = None
    for key in ["symbol", "ticker", "holding ticker", "identifier", "ticker symbol"]:
        if key in lower:
            sym_col = lower[key]
            break
    if sym_col is None:
        for c in cols:
            sample = df[c].dropna().astype(str).str.upper().head(20).tolist()
            hits = sum(1 for s in sample if normalize_ticker(s) is not None and len(str(s)) <= 8)
            if hits >= max(3, min(10, len(sample) // 2)):
                sym_col = c
                break
    for key in ["% assets", "weight", "weight (%)", "market value weight", "holding percent", "holdingpercent", "% of fund", "fund weight"]:
        if key in lower:
            weight_col = lower[key]
            break
    if weight_col is None:
        for c in reversed(cols):
            vals = df[c].dropna().head(50).map(parse_weight)
            if vals.notna().sum() >= 5:
                weight_col = c
                break
    return sym_col, weight_col


def clean_holdings(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"empty holdings from {source}")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    sym_col, weight_col = detect_columns(df)
    if sym_col is None or weight_col is None:
        raise ValueError(f"could not detect symbol/weight columns from {source}; columns={list(df.columns)}")
    out = pd.DataFrame({
        "ticker": df[sym_col].map(normalize_ticker),
        "weight": df[weight_col].map(parse_weight),
    })
    out = out.dropna(subset=["ticker", "weight"])
    out = out[out["weight"] > 0]
    out = out.groupby("ticker", as_index=False)["weight"].sum()
    if out.empty:
        raise ValueError(f"no usable holdings rows from {source}")
    total = out["weight"].sum()
    if total > 0:
        out["weight"] = out["weight"] / total
    out["holding_source"] = source
    return out


def fetch_stockanalysis_holdings(ticker: str) -> pd.DataFrame:
    url = f"https://stockanalysis.com/etf/{ticker.lower()}/holdings/"
    html = request_text(url)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No holdings table found at {url}")
    best = max(tables, key=len)
    return clean_holdings(best, "stockanalysis_full_holdings")


def fetch_spdr_xlsx_holdings(ticker: str) -> pd.DataFrame:
    url = f"https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"
    raw = request_bytes(url)
    tables = pd.read_excel(io.BytesIO(raw), sheet_name=None, header=None)
    frames = []
    for _name, sheet in tables.items():
        if sheet.empty:
            continue
        for header_row in range(min(20, len(sheet))):
            tmp = sheet.iloc[header_row + 1 :].copy()
            tmp.columns = sheet.iloc[header_row].astype(str).tolist()
            if any("ticker" in str(c).lower() or "identifier" in str(c).lower() for c in tmp.columns):
                frames.append(tmp)
    if not frames:
        raise RuntimeError("No usable SPDR holdings sheet")
    best = max(frames, key=len)
    return clean_holdings(best, "spdr_official_full_holdings")


def fetch_ishares_csv_holdings(ticker: str, product_id: str | None = None, slug: str | None = None) -> pd.DataFrame:
    if not product_id or not slug:
        raise RuntimeError("missing iShares product id or slug")
    url = f"https://www.ishares.com/us/products/{product_id}/{slug}/1467271812596.ajax?fileType=csv&fileName={ticker.upper()}_holdings&dataType=fund"
    text = request_text(url)
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        lower = line.lower()
        if "ticker" in lower and ("weight" in lower or "% of fund" in lower or "market value" in lower):
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))
    return clean_holdings(df, "ishares_official_full_holdings")


def fetch_yfinance_holdings(ticker: str) -> pd.DataFrame:
    fd = getattr(yf.Ticker(ticker), "funds_data", None)
    if fd is None:
        raise RuntimeError("yfinance funds_data unavailable")
    h = getattr(fd, "top_holdings", None)
    if h is None:
        raise RuntimeError("yfinance top_holdings unavailable")
    df = h.copy() if isinstance(h, pd.DataFrame) else pd.DataFrame(h)
    return clean_holdings(df, "yfinance_top_holdings")


def etf_config() -> list[dict[str, Any]]:
    path = REPO / "config" / "etfs.csv"
    rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        try:
            cfg = pd.read_csv(path).fillna("")
            for _, r in cfg.iterrows():
                t = normalize_ticker(r.get("ticker"))
                if not t:
                    continue
                rows.append({
                    "ticker": t,
                    "label": str(r.get("label") or t),
                    "category": str(r.get("category") or ""),
                    "provider": str(r.get("provider") or "").lower(),
                    "product_id": str(r.get("product_id") or ""),
                    "slug": str(r.get("slug") or ""),
                })
        except Exception as exc:
            print(f"[ensure_daily_derived] could not read config/etfs.csv: {exc}")
    existing = {r["ticker"] for r in rows}
    for t, label, category in DEFAULT_ETFS:
        if t not in existing:
            rows.append({"ticker": t, "label": label, "category": category, "provider": "", "product_id": "", "slug": ""})
    return rows


def fetch_best_etf_holdings(row: dict[str, Any]) -> pd.DataFrame:
    ticker = row["ticker"]
    errors = []
    attempts = []
    provider = str(row.get("provider") or "").lower()
    if provider == "spdr" or ticker in {"SPY", "DIA", "MDY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC"}:
        attempts.append(lambda: fetch_spdr_xlsx_holdings(ticker))
    if provider == "ishares" or row.get("product_id"):
        attempts.append(lambda: fetch_ishares_csv_holdings(ticker, row.get("product_id") or None, row.get("slug") or None))
    attempts.append(lambda: fetch_stockanalysis_holdings(ticker))
    attempts.append(lambda: fetch_yfinance_holdings(ticker))
    for fn in attempts:
        try:
            return fn()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors[-3:]))


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def build_etf_panel(usall: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    base = usall[["ticker", "ICC", "mktcap"]].copy()
    base["ticker"] = base["ticker"].map(normalize_ticker)
    rows = []
    for cfg in etf_config():
        ticker = cfg["ticker"]
        try:
            h = fetch_best_etf_holdings(cfg)
            merged = h.merge(base, on="ticker", how="left")
            matched = merged["ICC"].notna()
            coverage = float(merged.loc[matched, "weight"].sum()) if matched.any() else 0.0
            icc = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"]) if matched.any() else np.nan
            source = str(h["holding_source"].iloc[0]) if "holding_source" in h.columns and not h.empty else "unknown"
            if coverage >= 0.80 and int(matched.sum()) >= 10 and "top" not in source:
                method = "ICC calculation"
                status = "ok"
            elif coverage >= 0.20 and int(matched.sum()) >= 3:
                method = "Partial holdings estimate"
                status = "partial"
            else:
                method = "Unavailable"
                status = "no_matched_holdings"
                icc = np.nan
            rows.append({
                "date": asof_date,
                "ticker": ticker,
                "label": cfg.get("label") or ticker,
                "category": cfg.get("category") or "",
                "method": method,
                "icc": icc,
                "vw_icc": icc,
                "coverage_weight": coverage,
                "n_holdings_total": int(len(h)),
                "n_holdings_matched": int(matched.sum()),
                "holding_source": source,
                "status": status,
            })
        except Exception as exc:
            rows.append({
                "date": asof_date,
                "ticker": ticker,
                "label": cfg.get("label") or ticker,
                "category": cfg.get("category") or "",
                "method": "Unavailable",
                "icc": np.nan,
                "vw_icc": np.nan,
                "coverage_weight": 0.0,
                "n_holdings_total": 0,
                "n_holdings_matched": 0,
                "holding_source": "unavailable",
                "status": f"error: {type(exc).__name__}",
            })
    return pd.DataFrame(rows)


def fetch_nasdaq_symbol_directory() -> pd.DataFrame:
    urls = [
        "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt",
    ]
    frames = []
    for url in urls:
        text = request_text(url)
        lines = [line for line in text.splitlines() if "|" in line and not line.startswith("File Creation")]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.columns = [str(c).strip() for c in out.columns]
    return out


def adr_candidates() -> list[str]:
    df = fetch_nasdaq_symbol_directory()
    symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    name_col = "Security Name" if "Security Name" in df.columns else ("Security Name" if "Security Name" in df.columns else None)
    if name_col is None:
        name_col = [c for c in df.columns if "name" in c.lower()][0]
    etf_col = "ETF" if "ETF" in df.columns else None
    names = df[name_col].fillna("").astype(str)
    mask = names.str.contains(r"ADR|ADS|American Depositary|Depositary Shares|Depositary Receipt", case=False, regex=True)
    if etf_col:
        mask = mask & (df[etf_col].fillna("N").astype(str).str.upper() != "Y")
    syms = [normalize_ticker(x) for x in df.loc[mask, symbol_col].tolist()]
    return [s for s in syms if s]


def fetch_info_safe(ticker: str) -> dict[str, Any]:
    try:
        info = yf.Ticker(ticker).info or {}
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def build_country_panel(asof_date: str) -> pd.DataFrame:
    syms = adr_candidates()
    records = []
    for i, t in enumerate(syms, 1):
        info = fetch_info_safe(t)
        country = info.get("country") or info.get("headquartersCountry")
        mktcap = info.get("marketCap")
        name = info.get("shortName") or info.get("longName")
        if country and mktcap and pd.notna(mktcap) and float(mktcap) > 0:
            records.append({"ticker": t, "country_region": str(country), "company": name or t, "mktcap": float(mktcap)})
        if i % 50 == 0:
            time.sleep(0.5)
    if not records:
        return pd.DataFrame(columns=["date", "country_region", "method", "icc", "vw_icc", "n_candidates", "n_selected", "n_icc_available", "coverage_mktcap", "status", "constituents"])
    universe = pd.DataFrame(records).drop_duplicates("ticker")
    selected = []
    for country, g in universe.groupby("country_region"):
        g = g.sort_values("mktcap", ascending=False).head(10).copy()
        if len(g) >= 3:
            selected.append(g)
    if not selected:
        return pd.DataFrame(columns=["date", "country_region", "method", "icc", "vw_icc", "n_candidates", "n_selected", "n_icc_available", "coverage_mktcap", "status", "constituents"])
    selected_df = pd.concat(selected, ignore_index=True)
    tickers = selected_df["ticker"].dropna().unique().tolist()
    if get_live_panel is None:
        live = pd.DataFrame()
    else:
        live = get_live_panel(tickers)
    if live.empty:
        live = pd.DataFrame(columns=["ticker", "ICC"])
    live["ticker"] = live.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().str.replace(".", "-", regex=False)
    merged = selected_df.merge(live[["ticker", "ICC"]], on="ticker", how="left")
    rows = []
    for country, g in merged.groupby("country_region"):
        available = g[g["ICC"].notna()].copy()
        n_avail = int(len(available))
        n_selected = int(len(g))
        coverage = float(available["mktcap"].sum() / g["mktcap"].sum()) if g["mktcap"].sum() > 0 else 0.0
        if n_avail >= 5:
            icc = weighted_mean(available["ICC"], available["mktcap"])
            method = "ADR Top-10 ICC calculation" if n_avail >= min(10, n_selected) else "Partial ADR estimate"
            status = "ok" if method.startswith("ADR") else "partial"
        else:
            icc = np.nan
            method = "Unavailable"
            status = "too_few_adr_icc"
        rows.append({
            "date": asof_date,
            "country_region": country,
            "method": method,
            "icc": icc,
            "vw_icc": icc,
            "n_candidates": int((universe["country_region"] == country).sum()),
            "n_selected": n_selected,
            "n_icc_available": n_avail,
            "coverage_mktcap": coverage,
            "status": status,
            "constituents": ", ".join(g["ticker"].tolist()),
        })
    return pd.DataFrame(rows).sort_values(["status", "country_region"]).reset_index(drop=True)


def save_daily_panel(df: pd.DataFrame, root: Path, stem: str, asof_date: str) -> Path:
    dt = datetime.strptime(asof_date, "%Y-%m-%d")
    out_dir = root / dt.strftime("%Y%m")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}_{dt.strftime('%Y_%m%d')}.csv"
    df.to_csv(out, index=False)
    return out


def json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    clean = []
    for row in records:
        out = {}
        for k, v in row.items():
            if pd.isna(v):
                out[k] = None
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                out[k] = float(v) if np.isfinite(float(v)) else None
            else:
                out[k] = v
        clean.append(out)
    return clean


def patch_docs_json(family: str, df: pd.DataFrame, asof_date: str) -> None:
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    payload = {"asof_date": asof_date, "latest": json_safe_records(df), "daily": json_safe_records(df), "monthly": json_safe_records(df)}
    names = [family]
    if family == "etf":
        names.append("etf_icc")
    if family == "country":
        names.append("country_icc")
    for name in names:
        with open(DOCS_DATA / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)


def ensure_daily_derived(universe: str = "usall", force: bool = False) -> None:
    path, usall, asof_date = latest_valid_snapshot(universe)
    print(f"[ensure_daily_derived] latest raw snapshot = {path}")
    print(f"[ensure_daily_derived] asof_date = {asof_date}")
    dt = datetime.strptime(asof_date, "%Y-%m-%d")
    etf_file = ETF_DERIVED / dt.strftime("%Y%m") / f"etf_icc_{dt.strftime('%Y_%m%d')}.csv"
    country_file = COUNTRY_DERIVED / dt.strftime("%Y%m") / f"country_adr_icc_{dt.strftime('%Y_%m%d')}.csv"
    if force or not etf_file.exists() or etf_file.stat().st_size == 0:
        etf = build_etf_panel(usall, asof_date)
        etf_file = save_daily_panel(etf, ETF_DERIVED, "etf_icc", asof_date)
        print(f"[ensure_daily_derived] wrote ETF file: {etf_file}")
    else:
        etf = pd.read_csv(etf_file)
        print(f"[ensure_daily_derived] ETF already exists: {etf_file}")
    if force or not country_file.exists() or country_file.stat().st_size == 0:
        country = build_country_panel(asof_date)
        country_file = save_daily_panel(country, COUNTRY_DERIVED, "country_adr_icc", asof_date)
        print(f"[ensure_daily_derived] wrote country file: {country_file}")
    else:
        country = pd.read_csv(country_file)
        print(f"[ensure_daily_derived] country already exists: {country_file}")
    patch_docs_json("etf", etf, asof_date)
    patch_docs_json("country", country, asof_date)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    force_env = os.environ.get("FORCE_DERIVED_REBUILD", "0") == "1"
    ensure_daily_derived(args.universe, force=args.force or force_env)


if __name__ == "__main__":
    main()
