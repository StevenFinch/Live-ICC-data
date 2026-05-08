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
    ("SPY", "SPDR S&P 500 ETF Trust", "Broad Market"),
    ("IVV", "iShares Core S&P 500 ETF", "Broad Market"),
    ("VOO", "Vanguard S&P 500 ETF", "Broad Market"),
    ("VTI", "Vanguard Total Stock Market ETF", "Broad Market"),
    ("QQQ", "Invesco QQQ Trust", "Growth / Nasdaq"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF", "Index"),
    ("IWM", "iShares Russell 2000 ETF", "Small Cap"),
    ("IJR", "iShares Core S&P Small-Cap ETF", "Small Cap"),
    ("MDY", "SPDR S&P MidCap 400 ETF Trust", "Mid Cap"),
    ("VUG", "Vanguard Growth ETF", "Style"),
    ("VTV", "Vanguard Value ETF", "Style"),
    ("IWF", "iShares Russell 1000 Growth ETF", "Style"),
    ("IWD", "iShares Russell 1000 Value ETF", "Style"),
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
    ("SMH", "VanEck Semiconductor ETF", "Theme"),
    ("XBI", "SPDR S&P Biotech ETF", "Theme"),
]


def http_get(url: str, timeout: int = 25, retries: int = 3) -> requests.Response:
    """Fetch a URL with retries."""
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def normalize_symbol(x: Any) -> str:
    """Normalize symbols for Yahoo-compatible matching."""
    s = "" if x is None else str(x).strip().upper()
    s = re.sub(r"\s+", "", s).replace(".", "-")
    return re.sub(r"[^A-Z0-9\-\^]", "", s)


def normalize_weight(x: Any) -> float:
    """Normalize a weight into decimal units."""
    if x is None:
        return np.nan
    s = str(x).strip().replace("%", "").replace(",", "")
    try:
        v = float(s)
    except Exception:
        return np.nan
    return v / 100.0 if v > 1.5 else v


def read_etf_config(config_path: Path | None) -> pd.DataFrame:
    """Read ETF config or fall back to a built-in list."""
    if config_path is not None and config_path.exists() and config_path.stat().st_size > 0:
        try:
            df = pd.read_csv(config_path)
            if "ticker" in df.columns:
                if "label" not in df.columns:
                    df["label"] = df["ticker"]
                if "category" not in df.columns:
                    df["category"] = ""
                return df[["ticker", "label", "category"]].dropna(subset=["ticker"]).copy()
        except Exception:
            pass
    return pd.DataFrame(DEFAULT_ETFS, columns=["ticker", "label", "category"])


def parse_holdings_table(df: pd.DataFrame) -> pd.DataFrame:
    """Parse a generic holdings table into symbol and weight columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "weight"])
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    sym_col = None
    for key in ["ticker", "symbol", "holding ticker", "identifier"]:
        if key in lower:
            sym_col = lower[key]
            break
    if sym_col is None:
        for c in df.columns:
            vals = df[c].astype(str).str.upper()
            if vals.str.match(r"^[A-Z0-9\.\-]{1,8}$").mean() > 0.4:
                sym_col = c
                break

    weight_col = None
    for key in ["weight", "% weight", "weight (%)", "holding weight", "market value weight"]:
        if key in lower:
            weight_col = lower[key]
            break
    if weight_col is None:
        for c in df.columns[::-1]:
            vals = df[c].map(normalize_weight)
            if vals.notna().sum() >= max(3, len(df) * 0.2) and vals.dropna().between(0, 1).mean() > 0.5:
                weight_col = c
                break

    if sym_col is None:
        return pd.DataFrame(columns=["symbol", "weight"])

    out = pd.DataFrame({
        "symbol": df[sym_col].map(normalize_symbol),
        "weight": df[weight_col].map(normalize_weight) if weight_col is not None else np.nan,
    })
    out = out[(out["symbol"] != "") & out["symbol"].notna()].copy()
    if out.empty:
        return pd.DataFrame(columns=["symbol", "weight"])
    if out["weight"].isna().all():
        out["weight"] = 1.0 / len(out)
    else:
        out = out[out["weight"].notna() & (out["weight"] > 0)].copy()
        total = out["weight"].sum()
        if total > 0:
            out["weight"] = out["weight"] / total
    return out.groupby("symbol", as_index=False)["weight"].sum()


def fetch_stockanalysis_holdings(ticker: str) -> pd.DataFrame:
    """Fetch ETF holdings from StockAnalysis holdings page."""
    url = f"https://stockanalysis.com/etf/{ticker.lower()}/holdings/"
    r = http_get(url)
    tables = pd.read_html(io.StringIO(r.text))
    best = pd.DataFrame()
    for table in tables:
        parsed = parse_holdings_table(table)
        if len(parsed) > len(best):
            best = parsed
    if best.empty:
        raise RuntimeError(f"No StockAnalysis holdings for {ticker}")
    best["source"] = "stockanalysis_full_holdings"
    return best


def fetch_spdr_holdings(ticker: str) -> pd.DataFrame:
    """Fetch SPDR holdings workbook when available."""
    url = f"https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"
    r = http_get(url)
    xls = pd.read_excel(io.BytesIO(r.content), sheet_name=None, header=None)
    best = pd.DataFrame()
    for sheet in xls.values():
        for header in range(min(15, len(sheet))):
            candidate = sheet.iloc[header:].copy()
            candidate.columns = candidate.iloc[0].astype(str).tolist()
            candidate = candidate.iloc[1:].reset_index(drop=True)
            parsed = parse_holdings_table(candidate)
            if len(parsed) > len(best):
                best = parsed
    if best.empty:
        raise RuntimeError(f"No SPDR holdings parsed for {ticker}")
    best["source"] = "spdr_full_holdings"
    return best


def fetch_yfinance_top_holdings(ticker: str) -> pd.DataFrame:
    """Fetch yfinance top holdings as a partial-holdings fallback."""
    fund = yf.Ticker(ticker).funds_data
    h = getattr(fund, "top_holdings", None)
    if h is None:
        raise RuntimeError(f"No yfinance top holdings for {ticker}")
    df = h.copy() if isinstance(h, pd.DataFrame) else pd.DataFrame(h)
    parsed = parse_holdings_table(df)
    if parsed.empty:
        raise RuntimeError(f"No yfinance top holdings parsed for {ticker}")
    parsed["source"] = "yfinance_top_holdings"
    return parsed


def fetch_best_holdings(ticker: str) -> pd.DataFrame:
    """Fetch the best available online holdings."""
    errors = []
    for fetcher in (fetch_spdr_holdings, fetch_stockanalysis_holdings, fetch_yfinance_top_holdings):
        try:
            h = fetcher(ticker)
            if not h.empty:
                return h
        except Exception as exc:
            errors.append(f"{fetcher.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def build_etf_icc_panel(
    base_usall: pd.DataFrame,
    asof_date: str,
    config_path: Path | None = None,
    repo_root: Path | None = None,
    output_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and optionally archive ETF ICC from online ETF holdings."""
    cfg = read_etf_config(config_path)
    base = base_usall.copy()
    base.columns = [str(c).strip() for c in base.columns]
    base["ticker"] = base["ticker"].astype(str).map(normalize_symbol)
    base["ICC"] = pd.to_numeric(base["ICC"], errors="coerce")
    base = base[base["ticker"].notna() & (base["ticker"] != "")].copy()

    rows, members = [], []
    for _, row in cfg.iterrows():
        etf = normalize_symbol(row["ticker"])
        label = str(row.get("label", etf))
        category = str(row.get("category", ""))
        icc, coverage, n_holdings, n_matched = np.nan, 0.0, 0, 0
        method, source, status = "Unavailable", "", "unavailable"
        try:
            holdings = fetch_best_holdings(etf)
            n_holdings = int(len(holdings))
            source = str(holdings["source"].iloc[0]) if "source" in holdings.columns and not holdings.empty else "online_holdings"
            merged = holdings.merge(base[["ticker", "ICC"]], left_on="symbol", right_on="ticker", how="left")
            matched = merged["ICC"].notna() & np.isfinite(merged["ICC"])
            coverage = float(merged.loc[matched, "weight"].sum()) if matched.any() else 0.0
            n_matched = int(matched.sum())
            if n_matched > 0:
                icc = float(np.average(merged.loc[matched, "ICC"], weights=merged.loc[matched, "weight"]))
                method = "Partial holdings estimate" if source == "yfinance_top_holdings" or coverage < 0.80 else "ICC calculation"
                status = "partial_holdings_estimate" if method.startswith("Partial") else "icc_calculation"
            for _, m in merged.iterrows():
                members.append({
                    "date": asof_date, "etf": etf, "symbol": m.get("symbol"),
                    "weight": m.get("weight"), "matched": bool(pd.notna(m.get("ICC"))),
                    "constituent_icc": m.get("ICC"), "holding_source": source,
                })
        except Exception as exc:
            status = f"unavailable: {type(exc).__name__}"
            source = "none"
        rows.append({
            "date": asof_date, "ticker": etf, "label": label, "category": category,
            "icc": icc, "coverage_weight": coverage, "n_holdings": n_holdings,
            "n_matched": n_matched, "method": method, "holding_source": source, "status": status,
        })
    panel = pd.DataFrame(rows)
    member_df = pd.DataFrame(members)
    if output_root is not None:
        yyyymm = asof_date[:7].replace("-", "")
        tag = asof_date.replace("-", "_")
        out_dir = Path(output_root) / yyyymm
        out_dir.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out_dir / f"etf_icc_{tag}.csv", index=False)
        if not member_df.empty:
            member_df.to_csv(out_dir / f"etf_constituents_{tag}.csv", index=False)
    return panel, member_df
