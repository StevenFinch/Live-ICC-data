from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

DEFAULT_ETFS = [
    {"ticker": "SPY", "label": "SPDR S&P 500 ETF Trust", "category": "US equity"},
    {"ticker": "IVV", "label": "iShares Core S&P 500 ETF", "category": "US equity", "ishares_id": "239726", "ishares_slug": "ishares-core-sp-500-etf"},
    {"ticker": "VOO", "label": "Vanguard S&P 500 ETF", "category": "US equity"},
    {"ticker": "VTI", "label": "Vanguard Total Stock Market ETF", "category": "US equity"},
    {"ticker": "QQQ", "label": "Invesco QQQ Trust", "category": "US equity"},
    {"ticker": "DIA", "label": "SPDR Dow Jones Industrial Average ETF Trust", "category": "US equity"},
    {"ticker": "IWB", "label": "iShares Russell 1000 ETF", "category": "US equity", "ishares_id": "239707", "ishares_slug": "ishares-russell-1000-etf"},
    {"ticker": "IWM", "label": "iShares Russell 2000 ETF", "category": "US equity", "ishares_id": "239710", "ishares_slug": "ishares-russell-2000-etf"},
    {"ticker": "IWD", "label": "iShares Russell 1000 Value ETF", "category": "US value", "ishares_id": "239708", "ishares_slug": "ishares-russell-1000-value-etf"},
    {"ticker": "IWF", "label": "iShares Russell 1000 Growth ETF", "category": "US growth", "ishares_id": "239725", "ishares_slug": "ishares-russell-1000-growth-etf"},
    {"ticker": "EFA", "label": "iShares MSCI EAFE ETF", "category": "International equity", "ishares_id": "239623", "ishares_slug": "ishares-msci-eafe-etf"},
    {"ticker": "EEM", "label": "iShares MSCI Emerging Markets ETF", "category": "Emerging markets", "ishares_id": "239637", "ishares_slug": "ishares-msci-emerging-markets-etf"},
    {"ticker": "EWJ", "label": "iShares MSCI Japan ETF", "category": "Country ETF", "ishares_id": "239665", "ishares_slug": "ishares-msci-japan-etf"},
    {"ticker": "EWG", "label": "iShares MSCI Germany ETF", "category": "Country ETF", "ishares_id": "239650", "ishares_slug": "ishares-msci-germany-etf"},
    {"ticker": "MCHI", "label": "iShares MSCI China ETF", "category": "Country ETF", "ishares_id": "239619", "ishares_slug": "ishares-msci-china-etf"},
    {"ticker": "INDA", "label": "iShares MSCI India ETF", "category": "Country ETF", "ishares_id": "239659", "ishares_slug": "ishares-msci-india-etf"},
    {"ticker": "EWZ", "label": "iShares MSCI Brazil ETF", "category": "Country ETF", "ishares_id": "239612", "ishares_slug": "ishares-msci-brazil-etf"},
    {"ticker": "XLK", "label": "Technology Select Sector SPDR Fund", "category": "Sector US"},
    {"ticker": "XLF", "label": "Financial Select Sector SPDR Fund", "category": "Sector US"},
    {"ticker": "XLE", "label": "Energy Select Sector SPDR Fund", "category": "Sector US"},
]

SPDR_TICKERS = {"SPY", "DIA", "XLK", "XLF", "XLE", "XLY", "XLP", "XLV", "XLI", "XLB", "XLU", "XLRE", "XLC"}


def _clean_symbol(x: Any) -> str:
    s = str(x).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace(".", "-")
    s = re.sub(r"[^A-Z0-9\-]", "", s)
    return s


def _clean_weight(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).replace("%", "").replace(",", "").strip()
    try:
        v = float(s)
    except Exception:
        return None
    if not np.isfinite(v) or v <= 0:
        return None
    if v > 1.5:
        v = v / 100.0
    return float(v)


def _request(url: str, timeout: int = 30) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r


def _standardize_holdings(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["symbol", "weight", "holding_source"])
    cols = {str(c).strip().lower(): c for c in df.columns}
    symbol_col = None
    weight_col = None
    for c in ["ticker", "symbol", "identifier", "holding ticker", "ticker symbol"]:
        if c in cols:
            symbol_col = cols[c]
            break
    for c in ["weight (%)", "weight", "% weight", "market value weight", "weighting", "holding percent", "holdingpercent"]:
        if c in cols:
            weight_col = cols[c]
            break
    if symbol_col is None:
        for c in df.columns:
            vals = df[c].astype(str).str.upper().str.strip()
            hit = vals.str.match(r"^[A-Z][A-Z0-9\.-]{0,8}$", na=False).sum()
            if hit >= max(3, int(len(df) * 0.2)):
                symbol_col = c
                break
    if weight_col is None:
        for c in df.columns:
            vals = pd.to_numeric(df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
            if vals.notna().sum() >= max(3, int(len(df) * 0.2)) and vals.max(skipna=True) > 0:
                weight_col = c
                break
    if symbol_col is None or weight_col is None:
        return pd.DataFrame(columns=["symbol", "weight", "holding_source"])
    out = pd.DataFrame({
        "symbol": df[symbol_col].map(_clean_symbol),
        "weight": df[weight_col].map(_clean_weight),
    })
    out = out[(out["symbol"] != "") & out["weight"].notna() & (out["weight"] > 0)].copy()
    if out.empty:
        return pd.DataFrame(columns=["symbol", "weight", "holding_source"])
    out = out.drop_duplicates("symbol", keep="first")
    total = float(out["weight"].sum())
    if total > 0:
        out["weight"] = out["weight"] / total
    out["holding_source"] = source
    return out[["symbol", "weight", "holding_source"]]


def fetch_spdr_holdings(ticker: str) -> pd.DataFrame:
    url = f"https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"
    r = _request(url)
    sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None, header=None)
    frames: list[pd.DataFrame] = []
    for raw in sheets.values():
        if raw.empty:
            continue
        header_idx = None
        for i in range(min(30, len(raw))):
            row = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
            if any("ticker" == x or "identifier" == x for x in row) and any("weight" in x for x in row):
                header_idx = i
                break
        if header_idx is None:
            continue
        df = raw.iloc[header_idx + 1:].copy()
        df.columns = [str(x).strip() for x in raw.iloc[header_idx].tolist()]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["symbol", "weight", "holding_source"])
    return _standardize_holdings(pd.concat(frames, ignore_index=True), "spdr_full_holdings")


def fetch_ishares_holdings(ticker: str, product_id: str | None, slug: str | None) -> pd.DataFrame:
    if not product_id or not slug:
        return pd.DataFrame(columns=["symbol", "weight", "holding_source"])
    url = f"https://www.ishares.com/us/products/{product_id}/{slug}/1467271812596.ajax?fileType=csv&fileName={ticker.upper()}_holdings&dataType=fund"
    r = _request(url)
    text = r.text
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:80]):
        low = line.lower()
        if ("ticker" in low or "symbol" in low) and "weight" in low:
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame(columns=["symbol", "weight", "holding_source"])
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))
    return _standardize_holdings(df, "ishares_full_holdings")


def fetch_stockanalysis_holdings(ticker: str) -> pd.DataFrame:
    url = f"https://stockanalysis.com/etf/{ticker.lower()}/holdings/"
    try:
        html = _request(url).text
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return pd.DataFrame(columns=["symbol", "weight", "holding_source"])
    best = pd.DataFrame()
    for t in tables:
        if len(t) > len(best):
            best = t
    return _standardize_holdings(best, "stockanalysis_holdings")


def fetch_yfinance_top_holdings(ticker: str) -> pd.DataFrame:
    try:
        fd = yf.Ticker(ticker).funds_data
        h = fd.top_holdings
        if h is None:
            return pd.DataFrame(columns=["symbol", "weight", "holding_source"])
        df = h.copy() if isinstance(h, pd.DataFrame) else pd.DataFrame(h)
        return _standardize_holdings(df, "yfinance_top_holdings")
    except Exception:
        return pd.DataFrame(columns=["symbol", "weight", "holding_source"])


def fetch_best_holdings(row: pd.Series) -> pd.DataFrame:
    ticker = str(row.get("ticker", "")).upper().strip()
    product_id = str(row.get("ishares_id", "") or "").strip() or None
    slug = str(row.get("ishares_slug", "") or "").strip() or None
    providers = []
    if ticker in SPDR_TICKERS:
        providers.append(lambda: fetch_spdr_holdings(ticker))
    if product_id and slug:
        providers.append(lambda: fetch_ishares_holdings(ticker, product_id, slug))
    providers.extend([lambda: fetch_stockanalysis_holdings(ticker), lambda: fetch_yfinance_top_holdings(ticker)])
    best = pd.DataFrame(columns=["symbol", "weight", "holding_source"])
    for fn in providers:
        try:
            h = fn()
        except Exception:
            h = pd.DataFrame(columns=["symbol", "weight", "holding_source"])
        if not h.empty and len(h) > len(best):
            best = h
        if not h.empty and h["holding_source"].iloc[0] in {"spdr_full_holdings", "ishares_full_holdings"} and len(h) >= 50:
            return h
        time.sleep(0.2)
    return best


def _read_etf_config(config_path: Path | None = None) -> pd.DataFrame:
    if config_path and config_path.exists() and config_path.stat().st_size > 0:
        df = pd.read_csv(config_path)
    else:
        df = pd.DataFrame(DEFAULT_ETFS)
    for c in ["ticker", "label", "category", "ishares_id", "ishares_slug"]:
        if c not in df.columns:
            df[c] = ""
    return df


def build_etf_icc_panel(latest_usall: pd.DataFrame, config_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = latest_usall.copy()
    base["ticker"] = base["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False).str.strip()
    base = base[["ticker", "ICC", "mktcap"] + (["name"] if "name" in base.columns else [])].dropna(subset=["ticker", "ICC", "mktcap"])
    cfg = _read_etf_config(config_path)
    rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for _, r in cfg.iterrows():
        ticker = str(r.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        label = str(r.get("label", ticker) or ticker)
        category = str(r.get("category", "") or "")
        h = fetch_best_holdings(r)
        if h.empty:
            rows.append({"ticker": ticker, "label": label, "category": category, "icc": np.nan, "coverage_weight": 0.0, "n_holdings_total": 0, "n_holdings_matched": 0, "method": "Unavailable", "holding_source": "none", "status": "unavailable"})
            continue
        merged = h.merge(base, left_on="symbol", right_on="ticker", how="left")
        matched = merged["ICC"].notna() & merged["weight"].notna()
        coverage = float(merged.loc[matched, "weight"].sum()) if matched.any() else 0.0
        n_matched = int(matched.sum())
        n_total = int(len(h))
        source = str(h["holding_source"].iloc[0])
        if n_matched > 0 and coverage > 0:
            icc = float(np.average(merged.loc[matched, "ICC"], weights=merged.loc[matched, "weight"]))
            if source in {"spdr_full_holdings", "ishares_full_holdings"} and coverage >= 0.80 and n_matched >= 10:
                method = "ICC calculation"
                status = "icc_calculation"
            else:
                method = "Partial holdings estimate"
                status = "partial_holdings"
        else:
            icc = np.nan
            method = "Unavailable"
            status = "unavailable"
        rows.append({"ticker": ticker, "label": label, "category": category, "icc": icc, "coverage_weight": coverage, "n_holdings_total": n_total, "n_holdings_matched": n_matched, "method": method, "holding_source": source, "status": status})
        for _, m in merged.iterrows():
            member_rows.append({"etf": ticker, "symbol": m.get("symbol"), "weight": m.get("weight"), "matched": bool(pd.notna(m.get("ICC"))), "constituent_icc": m.get("ICC"), "holding_source": source})
    return pd.DataFrame(rows), pd.DataFrame(member_rows)
