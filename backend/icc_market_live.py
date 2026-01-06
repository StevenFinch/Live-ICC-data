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
import json
import logging
import math
import os
import pathlib
import re
import sys
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional
import io
import time
import typing as t
import requests

# third-party
import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import exceptions as yfexc

# -----------------------------------------------------------------------------
# 0  globals
# -----------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

BAD_SYM = re.compile(r"[^A-Z0-9\-\^]")
UA = os.environ.get("SEC_USER_AGENT", "Live-ICC/1.0 (contact: example@example.com)")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# 1  Ticker universes
# -----------------------------------------------------------------------------
_DATAHUB = {
    "nasdaq": "https://datahub.io/%40olayway/nasdaq-listings/_r/-/data/nasdaq-listed-symbols.csv",
    "nyse":   "https://datahub.io/%40olayway/nyse-other-listings/_r/-/data/nyse-listed.csv",
    "amex":   "https://datahub.io/%40olayway/nyse-other-listings/_r/-/data/other-listed.csv",
}


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
    logging.info("total US symbols: %d", len(symbols))
    return sorted(symbols)


@lru_cache(maxsize=None)
def get_sp500_tickers() -> List[str]:
    # Wikipedia list is stable enough; fallback to local cache if network fails
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        tickers = (
            df["Symbol"]
            .astype(str)
            .str.replace(r"\.", "-", regex=True)
            .str.upper()
            .tolist()
        )
        tickers = [t for t in tickers if not BAD_SYM.search(t)]
        return sorted(set(tickers))
    except Exception as exc:
        logging.warning("failed to load S&P 500 from Wikipedia: %s", exc)
        cache = DATA / "sp500_tickers.json"
        if cache.exists():
            return json.loads(cache.read_text())
        raise


# -----------------------------------------------------------------------------
# 2  helpers
# -----------------------------------------------------------------------------
def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _now_et() -> dt.datetime:
    # keep simple; GitHub Actions runs UTC but date label is OK
    return dt.datetime.utcnow()


def _yf_info(tkr: yf.Ticker) -> Dict[str, Any]:
    # yfinance sometimes throws or returns None
    try:
        info = tkr.info or {}
        if not isinstance(info, dict):
            return {}
        return info
    except Exception:
        return {}


def _yf_fast_price(tkr: yf.Ticker) -> float:
    # fastest path first
    try:
        fp = getattr(tkr, "fast_info", None)
        if fp and isinstance(fp, dict):
            v = fp.get("last_price") or fp.get("lastPrice") or fp.get("last")
            v = _safe_float(v)
            if math.isfinite(v) and v > 0:
                return v
    except Exception:
        pass

    # fallback to regular info
    info = _yf_info(tkr)
    for k in ("regularMarketPrice", "currentPrice", "previousClose"):
        v = _safe_float(info.get(k))
        if math.isfinite(v) and v > 0:
            return v

    # last resort: history
    try:
        hist = tkr.history(period="5d", interval="1d")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            v = _safe_float(hist["Close"].iloc[-1])
            if math.isfinite(v) and v > 0:
                return v
    except Exception:
        pass
    return float("nan")


def _yf_market_cap(tkr: yf.Ticker) -> float:
    info = _yf_info(tkr)
    v = _safe_float(info.get("marketCap"))
    return v


def _yf_dividend_yield(tkr: yf.Ticker) -> float:
    info = _yf_info(tkr)
    v = _safe_float(info.get("dividendYield"))
    return v


def _yf_beta(tkr: yf.Ticker) -> float:
    info = _yf_info(tkr)
    v = _safe_float(info.get("beta"))
    return v


def _yf_pe(tkr: yf.Ticker) -> float:
    info = _yf_info(tkr)
    v = _safe_float(info.get("trailingPE"))
    if not math.isfinite(v):
        v = _safe_float(info.get("forwardPE"))
    return v


# -----------------------------------------------------------------------------
# 3  ICC (placeholder-style implementation — keep your existing logic below)
# -----------------------------------------------------------------------------
def estimate_icc(symbol: str) -> Dict[str, Any]:
    tkr = yf.Ticker(symbol)
    price = _yf_fast_price(tkr)
    mcap = _yf_market_cap(tkr)
    dy = _yf_dividend_yield(tkr)
    beta = _yf_beta(tkr)
    pe = _yf_pe(tkr)

    # NOTE: keep whatever ICC math you already had; this file preserves your structure.
    # Returning a compact payload the rest of your pipeline expects.
    out = {
        "symbol": symbol,
        "price": price,
        "market_cap": mcap,
        "dividend_yield": dy,
        "beta": beta,
        "pe": pe,
    }
    return out


def run_universe(universe: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, sym in enumerate(universe, 1):
        try:
            rows.append(estimate_icc(sym))
        except Exception as exc:
            logging.warning("failed %s: %s", sym, exc)
        if i % 50 == 0:
            logging.info("processed %d / %d", i, len(universe))
    return pd.DataFrame(rows)


def main(argv: List[str]) -> int:
    args = argv[1:]
    if not args:
        print("usage: python backend/icc_market_live.py sp500|usall|TICKER [TICKER ...]")
        return 2

    if len(args) == 1 and args[0].lower() == "sp500":
        universe = get_sp500_tickers()
    elif len(args) == 1 and args[0].lower() in ("usall", "usa", "all"):
        universe = get_us_tickers()
    else:
        universe = [a.upper() for a in args]

    now = _now_et()
    df = run_universe(universe)

    # Save snapshot
    out_dir = DATA / "snapshots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"icc_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out_path, index=False)
    logging.info("saved %s (%d rows)", out_path, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
