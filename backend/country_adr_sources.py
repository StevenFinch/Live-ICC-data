
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from backend.icc_market_live import get_live_panel
except Exception:
    get_live_panel = None  # type: ignore


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

ADR_TERMS = re.compile(
    r"(american depositary|american depository| adr| ads|depositary shares|depository shares)",
    re.I,
)

EXCLUDE_TERMS = re.compile(
    r"(etf|fund|warrant|right|unit|notes due|preferred|preference|closed end|bond)",
    re.I,
)


def http_get_text(url: str, timeout: int = 30, retries: int = 3) -> str:
    """Fetch text content with retries."""
    headers = {"User-Agent": USER_AGENT}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def read_pipe_directory(url: str) -> pd.DataFrame:
    """Read Nasdaq Trader pipe-delimited symbol directory."""
    text = http_get_text(url)
    lines = [ln for ln in text.splitlines() if "|" in ln and not ln.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str).fillna("")


def normalize_symbol(x: str) -> str:
    """Normalize ticker for Yahoo-style usage."""
    return str(x).strip().upper().replace(".", "-").replace("/", "-")


def fetch_online_adr_candidates() -> pd.DataFrame:
    """Build an ADR-like candidate universe from Nasdaq Trader online symbol directories."""
    frames = []
    try:
        n = read_pipe_directory(NASDAQ_LISTED_URL)
        n = n.rename(columns={"Symbol": "ticker", "Security Name": "name", "ETF": "etf"})
        n["exchange"] = "NASDAQ"
        frames.append(n[["ticker", "name", "exchange", "etf"]])
    except Exception as exc:
        print(f"[country_adr] failed to read Nasdaq listed directory: {exc}")

    try:
        o = read_pipe_directory(OTHER_LISTED_URL)
        o = o.rename(columns={"ACT Symbol": "ticker", "Security Name": "name", "Exchange": "exchange", "ETF": "etf"})
        frames.append(o[["ticker", "name", "exchange", "etf"]])
    except Exception as exc:
        print(f"[country_adr] failed to read other listed directory: {exc}")

    if not frames:
        return pd.DataFrame(columns=["ticker", "name", "exchange", "source"])

    df = pd.concat(frames, ignore_index=True)
    df["ticker"] = df["ticker"].map(normalize_symbol)
    df["name"] = df["name"].astype(str)
    df["etf"] = df["etf"].astype(str).str.upper()
    df = df[(df["etf"] != "Y") & df["name"].str.contains(ADR_TERMS, na=False)]
    df = df[~df["name"].str.contains(EXCLUDE_TERMS, na=False)]
    df = df.drop_duplicates("ticker").copy()
    df["source"] = "nasdaq_trader_symbol_directories"
    return df[["ticker", "name", "exchange", "source"]].reset_index(drop=True)


def enrich_candidate_with_yfinance(ticker: str) -> dict:
    """Fetch country and market cap for one ADR candidate."""
    try:
        info = yf.Ticker(ticker).info or {}
        country = info.get("country")
        market_cap = info.get("marketCap")
        sector = info.get("sector")
        short_name = info.get("shortName") or info.get("longName")
        quote_type = info.get("quoteType")
        if not country or country == "United States":
            return {}
        if not market_cap or market_cap <= 0:
            return {}
        return {
            "ticker": ticker,
            "country": country,
            "market_cap": float(market_cap),
            "sector": sector,
            "yf_name": short_name,
            "quote_type": quote_type,
        }
    except Exception:
        return {}


def load_cached_or_build_adr_universe(cache_path: Path, max_age_days: int = 7) -> pd.DataFrame:
    """Load cached ADR universe if fresh; otherwise rebuild from online sources."""
    if cache_path.exists():
        try:
            mtime = pd.Timestamp(cache_path.stat().st_mtime, unit="s")
            age_days = (pd.Timestamp.utcnow().tz_localize(None) - mtime.tz_localize(None)).days
            cached = pd.read_csv(cache_path)
            if age_days <= max_age_days and not cached.empty:
                return cached
        except Exception:
            pass

    raw = fetch_online_adr_candidates()
    rows = []
    for i, r in raw.iterrows():
        ticker = r["ticker"]
        rec = enrich_candidate_with_yfinance(ticker)
        if rec:
            rec.update({"name": r["name"], "exchange": r["exchange"], "source": r["source"]})
            rows.append(rec)
        if (i + 1) % 30 == 0:
            time.sleep(0.6)

    out = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def build_country_adr_icc_panel(cache_path: Path, min_available: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute country ADR Top-10 ICC using online ADR candidates and live ICC estimates."""
    adr = load_cached_or_build_adr_universe(cache_path)
    if adr.empty or get_live_panel is None:
        return (
            pd.DataFrame(columns=[
                "country", "icc", "method", "n_candidates", "n_selected",
                "n_icc_available", "coverage_mktcap", "status"
            ]),
            pd.DataFrame(),
        )

    selected_rows = []
    for country, g in adr.groupby("country"):
        g2 = g.sort_values("market_cap", ascending=False).head(10).copy()
        g2["rank_in_country"] = range(1, len(g2) + 1)
        selected_rows.append(g2)
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()

    tickers = selected["ticker"].dropna().astype(str).unique().tolist()
    live = get_live_panel(tickers)
    if live is None or live.empty:
        live = pd.DataFrame(columns=["ticker", "ICC", "mktcap", "name", "sector"])

    live = live.copy()
    live["ticker"] = live["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    selected["ticker"] = selected["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    merged = selected.merge(live[["ticker", "ICC", "mktcap", "name", "sector"]], on="ticker", how="left", suffixes=("", "_icc"))

    country_rows = []
    for country, g in merged.groupby("country"):
        total_mcap = float(g["market_cap"].sum()) if len(g) else 0.0
        valid = g[g["ICC"].notna() & g["market_cap"].notna() & (g["market_cap"] > 0)].copy()
        n_candidates = int((adr["country"] == country).sum())
        n_selected = int(len(g))
        n_available = int(len(valid))
        if n_available > 0 and total_mcap > 0:
            coverage = float(valid["market_cap"].sum() / total_mcap)
        else:
            coverage = 0.0

        if n_available >= 10:
            method = "ADR Top-10 ICC calculation"
            status = "icc_calculation"
        elif n_available >= min_available:
            method = "Partial ADR estimate"
            status = "partial_adr_estimate"
        else:
            method = "Unavailable"
            status = "unavailable"

        if n_available >= min_available:
            icc = float(np.average(valid["ICC"], weights=valid["market_cap"]))
        else:
            icc = None

        country_rows.append(
            {
                "country": country,
                "icc": icc,
                "method": method,
                "n_candidates": n_candidates,
                "n_selected": n_selected,
                "n_icc_available": n_available,
                "coverage_mktcap": coverage,
                "status": status,
            }
        )

    country_panel = pd.DataFrame(country_rows).sort_values(["status", "country"]).reset_index(drop=True)
    return country_panel, merged
