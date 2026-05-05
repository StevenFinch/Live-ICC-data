from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

DEFAULT_ETFS = [
    ("SPY", "SPDR S&P 500 ETF Trust", "Large Blend"),
    ("IVV", "iShares Core S&P 500 ETF", "Large Blend"),
    ("VOO", "Vanguard S&P 500 ETF", "Large Blend"),
    ("VTI", "Vanguard Total Stock Market ETF", "Total Market"),
    ("QQQ", "Invesco QQQ Trust", "Large Growth"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF", "Large Blend"),
    ("IWM", "iShares Russell 2000 ETF", "Small Blend"),
    ("IJR", "iShares Core S&P Small-Cap ETF", "Small Blend"),
    ("MDY", "SPDR S&P MidCap 400 ETF", "Mid Blend"),
    ("VUG", "Vanguard Growth ETF", "Large Growth"),
    ("VTV", "Vanguard Value ETF", "Large Value"),
    ("IWF", "iShares Russell 1000 Growth ETF", "Large Growth"),
    ("IWD", "iShares Russell 1000 Value ETF", "Large Value"),
    ("XLK", "Technology Select Sector SPDR Fund", "Sector"),
    ("XLF", "Financial Select Sector SPDR Fund", "Sector"),
    ("XLV", "Health Care Select Sector SPDR Fund", "Sector"),
    ("XLY", "Consumer Discretionary Select Sector SPDR Fund", "Sector"),
    ("XLP", "Consumer Staples Select Sector SPDR Fund", "Sector"),
    ("XLE", "Energy Select Sector SPDR Fund", "Sector"),
    ("XLI", "Industrial Select Sector SPDR Fund", "Sector"),
    ("XLB", "Materials Select Sector SPDR Fund", "Sector"),
    ("XLU", "Utilities Select Sector SPDR Fund", "Sector"),
    ("XLRE", "Real Estate Select Sector SPDR Fund", "Sector"),
    ("XLC", "Communication Services Select Sector SPDR Fund", "Sector"),
    ("SMH", "VanEck Semiconductor ETF", "Industry"),
    ("XBI", "SPDR S&P Biotech ETF", "Industry"),
]

ISHARES_PRODUCTS = {
    "IVV": ("239726", "ishares-core-sp-500-etf"),
    "IWM": ("239710", "ishares-russell-2000-etf"),
    "IJR": ("239774", "ishares-core-sp-smallcap-etf"),
    "IWF": ("239707", "ishares-russell-1000-growth-etf"),
    "IWD": ("239706", "ishares-russell-1000-value-etf"),
}


def _http_get(url: str, timeout: int = 30, retries: int = 2) -> requests.Response:
    """Fetch a URL with a browser-like user agent and retries."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(0.7 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def _clean_symbol(value: Any) -> str | None:
    """Normalize a ticker into Yahoo-compatible style."""
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().upper()
    if not s or s in {"NAN", "--", "-"}:
        return None
    s = s.replace(".", "-").replace(" ", "")
    if re.search(r"[^A-Z0-9\-]", s):
        return None
    return s


def _parse_weight(value: Any) -> float | None:
    """Parse percentage or decimal weights into decimal format."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        x = value.replace("%", "").replace(",", "").strip()
    else:
        x = value
    try:
        w = float(x)
    except Exception:
        return None
    if not np.isfinite(w) or w <= 0:
        return None
    return w / 100.0 if w > 1.5 else w


def _standardize_holdings(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Standardize arbitrary holdings tables into symbol/weight/source columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "weight", "source"])

    cols = {str(c).strip().lower(): c for c in df.columns}
    symbol_col = None
    weight_col = None

    for cand in ["ticker", "symbol", "holding ticker", "identifier", "local ticker"]:
        if cand in cols:
            symbol_col = cols[cand]
            break

    for cand in ["weight", "weight (%)", "% weight", "market value weight", "holdingpercent", "holding_percent"]:
        if cand in cols:
            weight_col = cols[cand]
            break

    if symbol_col is None:
        for c in df.columns:
            vals = df[c].astype(str).str.upper()
            plausible = vals.str.match(r"^[A-Z0-9][A-Z0-9\.\-]{0,8}$", na=False).sum()
            if plausible >= max(3, len(df) // 4):
                symbol_col = c
                break

    if weight_col is None:
        for c in df.columns:
            vals = df[c].map(_parse_weight)
            if vals.notna().sum() >= max(3, len(df) // 4):
                weight_col = c
                break

    if symbol_col is None or weight_col is None:
        return pd.DataFrame(columns=["symbol", "weight", "source"])

    out = pd.DataFrame({
        "symbol": df[symbol_col].map(_clean_symbol),
        "weight": df[weight_col].map(_parse_weight),
    })
    out = out[out["symbol"].notna() & out["weight"].notna() & (out["weight"] > 0)].copy()
    if out.empty:
        return pd.DataFrame(columns=["symbol", "weight", "source"])
    out = out.groupby("symbol", as_index=False)["weight"].sum()
    total = out["weight"].sum()
    if total > 0:
        out["weight"] = out["weight"] / total
    out["source"] = source
    return out[["symbol", "weight", "source"]]


def fetch_stockanalysis_holdings(ticker: str) -> pd.DataFrame:
    """Fetch ETF holdings from StockAnalysis where available."""
    url = f"https://stockanalysis.com/etf/{ticker.lower()}/holdings/"
    html = _http_get(url).text
    tables = pd.read_html(io.StringIO(html))
    best = pd.DataFrame()
    for table in tables:
        lower = [str(c).strip().lower() for c in table.columns]
        if any(c in lower for c in ["symbol", "ticker"]) and any("weight" in c for c in lower):
            if len(table) > len(best):
                best = table
    return _standardize_holdings(best, "stockanalysis_full_holdings")


def fetch_spdr_holdings(ticker: str) -> pd.DataFrame:
    """Fetch SPDR holdings workbook using the common public workbook pattern."""
    url = f"https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"
    content = _http_get(url).content
    xls = pd.ExcelFile(io.BytesIO(content))
    best = pd.DataFrame()
    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        for header_row in range(min(20, len(raw))):
            candidate = pd.read_excel(xls, sheet_name=sheet, header=header_row)
            parsed = _standardize_holdings(candidate, "spdr_full_holdings")
            if len(parsed) > len(best):
                best = parsed
    return best


def _read_ishares_csv_text(text: str, source: str) -> pd.DataFrame:
    """Parse iShares CSV text with variable metadata header rows."""
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:80]):
        lower = line.lower()
        if "ticker" in lower and ("weight" in lower or "market value" in lower):
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame(columns=["symbol", "weight", "source"])
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))
    return _standardize_holdings(df, source)


def fetch_ishares_holdings(ticker: str) -> pd.DataFrame:
    """Fetch iShares holdings CSV for mapped iShares products."""
    if ticker not in ISHARES_PRODUCTS:
        return pd.DataFrame(columns=["symbol", "weight", "source"])
    product_id, slug = ISHARES_PRODUCTS[ticker]
    url = (
        f"https://www.ishares.com/us/products/{product_id}/{slug}/"
        f"1467271812596.ajax?fileType=csv&fileName={ticker}_holdings&dataType=fund"
    )
    text = _http_get(url).text
    return _read_ishares_csv_text(text, "ishares_full_holdings")


def fetch_yfinance_top_holdings(ticker: str) -> pd.DataFrame:
    """Fetch yfinance top holdings as a fallback, not as a full-holdings source."""
    fund = yf.Ticker(ticker).funds_data
    holdings = getattr(fund, "top_holdings", None)
    if holdings is None:
        return pd.DataFrame(columns=["symbol", "weight", "source"])
    df = holdings.copy() if isinstance(holdings, pd.DataFrame) else pd.DataFrame(holdings)
    return _standardize_holdings(df, "yfinance_top_holdings")


def fetch_best_holdings(ticker: str) -> pd.DataFrame:
    """Fetch the best available online holdings table for one ETF."""
    fetchers = [
        fetch_spdr_holdings,
        fetch_ishares_holdings,
        fetch_stockanalysis_holdings,
        fetch_yfinance_top_holdings,
    ]
    errors: list[str] = []
    for fetcher in fetchers:
        try:
            h = fetcher(ticker)
            if not h.empty:
                return h
        except Exception as exc:
            errors.append(f"{fetcher.__name__}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(columns=["symbol", "weight", "source"])


def _etf_list_from_config() -> list[tuple[str, str, str]]:
    """Read config/etfs.csv if available, otherwise use the built-in list."""
    path = REPO / "config" / "etfs.csv"
    if not path.exists() or path.stat().st_size == 0:
        return DEFAULT_ETFS
    try:
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            return DEFAULT_ETFS
        rows = []
        for _, r in df.iterrows():
            ticker = str(r.get("ticker", "")).upper().strip()
            if not ticker:
                continue
            label = str(r.get("label", ticker))
            category = str(r.get("category", "ETF"))
            rows.append((ticker, label, category))
        return rows or DEFAULT_ETFS
    except Exception:
        return DEFAULT_ETFS


def build_etf_icc_panel(base_usall: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    """Build and archive ETF ICC using online holdings and current firm-level ICC snapshots."""
    base = base_usall.copy()
    base["ticker"] = base["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    base = base[["ticker", "ICC", "mktcap"]].dropna().copy()

    rows: list[dict[str, Any]] = []
    for ticker, label, category in _etf_list_from_config():
        try:
            holdings = fetch_best_holdings(ticker)
            if holdings.empty:
                rows.append({
                    "date": asof_date,
                    "ticker": ticker,
                    "label": label,
                    "category": category,
                    "icc": np.nan,
                    "coverage_weight": 0.0,
                    "n_holdings": 0,
                    "n_matched": 0,
                    "method": "Unavailable",
                    "holding_source": "none",
                    "status": "no_online_holdings",
                })
                continue
            merged = holdings.merge(base, left_on="symbol", right_on="ticker", how="left")
            matched = merged["ICC"].notna() & merged["weight"].notna() & (merged["weight"] > 0)
            coverage = float(merged.loc[matched, "weight"].sum()) if matched.any() else 0.0
            n_matched = int(matched.sum())
            source = str(holdings["source"].iloc[0]) if "source" in holdings.columns and len(holdings) else "unknown"
            if n_matched > 0:
                icc = float(np.average(merged.loc[matched, "ICC"], weights=merged.loc[matched, "weight"]))
            else:
                icc = np.nan
            if coverage >= 0.75 and n_matched >= 10 and "top" not in source:
                method = "ICC calculation"
                status = "ok_full_holdings"
            elif coverage >= 0.30 and n_matched >= 5:
                method = "Partial holdings estimate"
                status = "ok_partial_holdings"
            else:
                method = "Unavailable"
                status = "insufficient_matching_holdings"
            rows.append({
                "date": asof_date,
                "ticker": ticker,
                "label": label,
                "category": category,
                "icc": icc,
                "coverage_weight": coverage,
                "n_holdings": int(len(holdings)),
                "n_matched": n_matched,
                "method": method,
                "holding_source": source,
                "status": status,
            })
        except Exception as exc:
            rows.append({
                "date": asof_date,
                "ticker": ticker,
                "label": label,
                "category": category,
                "icc": np.nan,
                "coverage_weight": 0.0,
                "n_holdings": 0,
                "n_matched": 0,
                "method": "Unavailable",
                "holding_source": "error",
                "status": f"error:{type(exc).__name__}",
            })

    out = pd.DataFrame(rows)
    yyyymm = asof_date[:7].replace("-", "")
    tag = asof_date[:4] + "_" + asof_date[5:7] + asof_date[8:10]
    out_dir = DATA_DIR / "derived" / "etf" / yyyymm
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"etf_icc_{tag}.csv", index=False)
    return out
