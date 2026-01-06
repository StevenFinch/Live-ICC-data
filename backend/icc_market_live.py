from __future__ import annotations
"""
backend/icc_market_live.py – live Implied Cost of Capital (ICC) estimator
=========================================================================
CLI examples
------------
# S&P 500
python backend/icc_market_live.py sp500
# entire US market (slow)
python backend/icc_market_live.py usall
# custom list
python backend/icc_market_live.py AAPL MSFT NVDA
"""

# stdlib
import datetime as dt
import logging
import pathlib
import random
import re
import sys
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional
import io
import time

import requests

# third-party
import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import exceptions as yfexc

# -----------------------------------------------------------------------------
ROOT   = pathlib.Path(__file__).resolve().parents[1]
TODAY  = dt.date.today().isoformat()

PAUSE_EVERY, PAUSE_SEC = 20, 0.6       # rate-limit safety
G_MIN, G_MAX           = -0.20, 0.25   # clamp long-run growth
MAX_TICKERS            = 6000          # keep sane

BAD_SYM                = re.compile(r"[^A-Z\\.\-]")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# -----------------------------------------------------------------------------
# 1  Ticker universes
# -----------------------------------------------------------------------------
_DATAHUB = {
    "nasdaq": "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed-symbols.csv",
    "nyse":   "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv",
    "amex":   "https://datahub.io/core/nyse-other-listings/r/amex-listed.csv",
}

_LOCAL_TICKERS_CSV = ROOT / "tickers" / "tickers.csv"

def _load_local_tickers_csv(path: pathlib.Path = _LOCAL_TICKERS_CSV) -> List[str]:
    """
    Fallback universe for `usall` when DataHub endpoints fail.

    Expected format (example):
      ticker,date
      A,2026-01-02
      AAPL,2026-01-02

    Notes:
    - Only the ticker column is used; date is ignored.
    - Dots are converted to dashes to match yfinance (e.g., BRK.B -> BRK-B).
    """
    if not path.exists():
        logging.warning("local tickers fallback not found: %s", path)
        return []
    try:
        df = pd.read_csv(path, dtype=str)
        col = "ticker" if "ticker" in df.columns else df.columns[0]
        raw = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.", "-", regex=True)
            .str.upper()
        )
        syms = sorted({s for s in raw if s and not BAD_SYM.search(s)})
        logging.info("local tickers loaded: %d (%s)", len(syms), path)
        return syms
    except Exception as exc:
        logging.warning("failed to load local tickers from %s: %s", path, exc)
        return []

@lru_cache(maxsize=None)
def get_us_tickers() -> List[str]:
    symbols: set[str] = set()
    for name, url in _DATAHUB.items():
        try:
            df = pd.read_csv(url, dtype=str, usecols=[0])
            col = df.columns[0]
            raw = df[col].astype(str).str.replace(r"\.", "-", regex=True).str.upper()
            symbols.update(s for s in raw if not BAD_SYM.search(s))
            logging.info("%s tickers loaded: %d", name.upper(), len(raw))
        except Exception as exc:
            logging.warning("failed to load %s: %s", name, exc)

    # Fallback: if DataHub produced nothing, use local ./tickers/tickers.csv
    if not symbols:
        logging.warning(
            "DataHub returned 0 symbols; falling back to local tickers file: %s",
            _LOCAL_TICKERS_CSV,
        )
        symbols.update(_load_local_tickers_csv())

    logging.info("total US symbols: %d", len(symbols))
    return sorted(symbols)

_WIKI_TICK = re.compile(r"(symbol|ticker)", re.I)
_INDEX_URL = {
    "sp500":  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp100":  "https://en.wikipedia.org/wiki/S%26P_100",
    "dow30":  "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "ndx100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}

def _get(url: str, max_retries: int = 3, timeout: int = 20) -> str:
    """GET with a real User-Agent and a few retries (helps avoid 403s)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    last_exc: Optional[Exception] = None
    for k in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5 * (k + 1))
    raise RuntimeError(f"GET failed after {max_retries} tries: {url} ({last_exc})")

def _index_tickers(kind: str) -> List[str]:
    """Scrape Wikipedia constituents tables for common indices."""
    url = _INDEX_URL[kind]
    html = _get(url)
    tables = pd.read_html(io.StringIO(html))
    out: List[str] = []
    for tdf in tables:
        # pick a column named Symbol/Ticker, etc.
        cols = list(map(str, tdf.columns))
        sym_col = None
        for c in cols:
            if _WIKI_TICK.search(c):
                sym_col = c
                break
        if sym_col is None:
            continue
        raw = (
            tdf[sym_col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.", "-", regex=True)
            .str.upper()
        )
        out.extend([s for s in raw if s and not BAD_SYM.search(s)])
    out = sorted(set(out))
    logging.info("%s tickers loaded: %d", kind.upper(), len(out))
    return out


# -----------------------------------------------------------------------------
# 2  Data fetch helpers (yfinance)
# -----------------------------------------------------------------------------
def _yf_info(ticker: str) -> Dict[str, Any]:
    """Fetch yfinance info with retry / error handling."""
    try:
        tkr = yf.Ticker(ticker)
        # .info can be slow; keep it but wrap errors
        info = tkr.info or {}
        return info
    except (yfexc.YFinanceException, Exception) as exc:
        logging.debug("yfinance info failed for %s: %s", ticker, exc)
        return {}

def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def _clamp(x: float, lo: float, hi: float) -> float:
    if np.isnan(x):
        return x
    return float(min(max(x, lo), hi))


# -----------------------------------------------------------------------------
# 3  Simple ICC estimator (kept lightweight for “live” mode)
# -----------------------------------------------------------------------------
def estimate_icc(
    price: float,
    fwd_eps: float,
    growth: float,
    payout: float = 0.0,
) -> float:
    """
    Very lightweight implied cost of capital proxy:
      r ≈ (fwd_eps / price) + g
    with optional payout term if you want:
      r ≈ (fwd_eps * (1 - payout) / price) + g
    """
    if price <= 0 or np.isnan(price) or np.isnan(fwd_eps) or np.isnan(growth):
        return float("nan")
    # payout is fraction of earnings paid out (0..1)
    payout = 0.0 if np.isnan(payout) else float(min(max(payout, 0.0), 1.0))
    ey = (fwd_eps * (1.0 - payout)) / price
    return float(ey + growth)


def get_live_panel(universe: List[str]) -> pd.DataFrame:
    """Build a per-ticker panel with live-ish ICC inputs."""
    universe = [u.strip().upper() for u in universe if u and isinstance(u, str)]
    universe = [u.replace(".", "-") for u in universe]
    universe = [u for u in universe if not BAD_SYM.search(u)]

    if len(universe) > MAX_TICKERS:
        logging.warning("universe too large (%d), truncating to %d", len(universe), MAX_TICKERS)
        universe = universe[:MAX_TICKERS]

    rows: List[Dict[str, Any]] = []
    for i, sym in enumerate(universe, 1):
        if (i % PAUSE_EVERY) == 0:
            time.sleep(PAUSE_SEC + random.random() * 0.2)

        info = _yf_info(sym)

        price = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
        fwd_eps = _safe_float(info.get("forwardEps"))
        trailing_eps = _safe_float(info.get("trailingEps"))

        # Try to derive a reasonable growth proxy:
        # 1) use analyst long-term growth if present
        g = _safe_float(info.get("earningsGrowth"))
        if np.isnan(g):
            g = _safe_float(info.get("revenueGrowth"))

        # clamp
        g = _clamp(g, G_MIN, G_MAX)

        # payout ratio
        payout = _safe_float(info.get("payoutRatio"))

        icc = estimate_icc(price=price, fwd_eps=fwd_eps, growth=g, payout=payout)

        rows.append(
            {
                "date": TODAY,
                "ticker": sym,
                "price": price,
                "forward_eps": fwd_eps,
                "trailing_eps": trailing_eps,
                "growth": g,
                "payout_ratio": payout,
                "icc": icc,
                "source": "yfinance",
            }
        )

    df = pd.DataFrame(rows)
    return df


# -----------------------------------------------------------------------------
# 4  CLI
# -----------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("usage: icc_market_live.py {usall|sp500|sp100|dow30|ndx100|TICKER...}")
        raise SystemExit(2)

    arg = sys.argv[1].lower()
    if arg == "usall":
        universe = get_us_tickers()
    elif arg in _INDEX_URL:
        universe = _index_tickers(arg)
    else:
        universe = [s.upper() for s in sys.argv[1:]]

    panel = get_live_panel(universe)
    out = ROOT / "icc_live_sample.csv"
    panel.to_csv(out, index=False)
    print(panel.head())
    logging.info("saved %s (%d rows)", out, len(panel))


if __name__ == "__main__":
    main()
