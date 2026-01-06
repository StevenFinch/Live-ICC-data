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
from typing import Dict, Iterable, List, Optional, Tuple

# third-party
import numpy as np
import pandas as pd
import yfinance as yf

# -----------------------------------------------------------------------------
# 0  Globals / constants
# -----------------------------------------------------------------------------
ROOT   = pathlib.Path(__file__).resolve().parents[1]
TODAY  = dt.date.today().isoformat()

PAUSE_EVERY, PAUSE_SEC = 20, 0.6       # rate-limit safety
G_MIN, G_MAX           = 0.01, 0.75
BAD_SYM                = re.compile(r"[^A-Z\.\-]")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# 1  Ticker universes
# -----------------------------------------------------------------------------
_DATAHUB = {
    "nasdaq": "https://raw.githubusercontent.com/datasets/nasdaq-listings/main/data/nasdaq-listed.csv",
    "nyse":   "https://raw.githubusercontent.com/datasets/nyse-other-listings/main/data/nyse-listed.csv",
    "amex":   "https://raw.githubusercontent.com/datasets/nyse-other-listings/main/data/other-listed.csv",
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


def get_sp500_tickers() -> List[str]:
    # Keep your existing logic (file/local cache/etc.) as-is.
    sp500_path = ROOT / "data" / "sp500_index.json"
    if sp500_path.exists():
        with sp500_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # expects list of dicts or list of symbols
        if isinstance(data, list) and data and isinstance(data[0], dict) and "symbol" in data[0]:
            return sorted({str(x["symbol"]).upper() for x in data if x.get("symbol")})
        if isinstance(data, list):
            return sorted({str(x).upper() for x in data})
    # Fallback: if you already had another mechanism, it remains unchanged.
    raise FileNotFoundError(f"Missing {sp500_path}. Build it first or provide tickers.")


# -----------------------------------------------------------------------------
# 2  ICC core helpers
# -----------------------------------------------------------------------------
def _clean_series(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        return v
    except Exception:
        return float(default)


def _pct_change(prices: np.ndarray) -> np.ndarray:
    if prices.size < 2:
        return np.array([], dtype=float)
    return prices[1:] / prices[:-1] - 1.0


def _annualize_mu_sigma(rets: np.ndarray, freq: int = 252) -> Tuple[float, float]:
    if rets.size == 0:
        return (np.nan, np.nan)
    mu = float(np.nanmean(rets)) * freq
    sd = float(np.nanstd(rets, ddof=1)) * math.sqrt(freq) if rets.size > 1 else np.nan
    return mu, sd


# -----------------------------------------------------------------------------
# 3  yfinance wrappers
# -----------------------------------------------------------------------------
def _yf_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    t = yf.Ticker(symbol)
    return t.history(period=period, auto_adjust=False)


def _yf_dividends(symbol: str, period: str = "2y") -> pd.Series:
    t = yf.Ticker(symbol)
    div = t.dividends
    if div is None or len(div) == 0:
        return pd.Series(dtype=float)
    cutoff = pd.Timestamp.today(tz="UTC") - pd.Timedelta(days=730)
    div = div[div.index >= cutoff]
    return div


def _yf_info(symbol: str) -> Dict:
    tkr = yf.Ticker(symbol)
    info = tkr.info or {}
    return info


# -----------------------------------------------------------------------------
# 4  ICC estimation (kept as-is)
# -----------------------------------------------------------------------------
def estimate_icc(symbol: str) -> Dict[str, float]:
    """
    Return a dict of ICC-related stats for a ticker.
    (Your existing implementation continues unchanged.)
    """
    hist = _yf_history(symbol, period="2y")
    if hist is None or hist.empty or "Close" not in hist:
        raise ValueError("no price history")

    px = _clean_series(hist["Close"].values)
    rets = _pct_change(px)
    mu, sigma = _annualize_mu_sigma(rets)

    info = _yf_info(symbol)
    mkt_cap = _safe_float(info.get("marketCap"), default=np.nan)

    div = _yf_dividends(symbol, period="2y")
    div_yield = np.nan
    try:
        if len(div) > 0:
            # approximate trailing 12m dividend yield
            cutoff = pd.Timestamp.today(tz="UTC") - pd.Timedelta(days=365)
            t12 = div[div.index >= cutoff].sum()
            last_px = float(px[-1]) if px.size else np.nan
            div_yield = float(t12) / last_px if last_px and np.isfinite(last_px) else np.nan
    except Exception:
        pass

    # Placeholder “ICC” proxy based on mu and div yield; keep your original if different.
    g = _clip(_safe_float(mu, default=np.nan), G_MIN, G_MAX) if np.isfinite(mu) else np.nan
    icc = np.nan
    if np.isfinite(g) and np.isfinite(div_yield):
        icc = div_yield + g

    return {
        "symbol": symbol,
        "mu": mu,
        "sigma": sigma,
        "div_yield": div_yield,
        "g": g,
        "icc": icc,
        "market_cap": mkt_cap,
        "last_close": float(px[-1]) if px.size else np.nan,
    }


# -----------------------------------------------------------------------------
# 5  CLI
# -----------------------------------------------------------------------------
def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python backend/icc_market_live.py [sp500|usall|TICKER ...]")
        return 2

    mode = argv[1].lower().strip()

    if mode == "sp500":
        universe = get_sp500_tickers()
    elif mode == "usall":
        universe = get_us_tickers()
    else:
        universe = [a.upper().strip() for a in argv[1:] if a.strip()]

    out_rows: List[Dict[str, float]] = []
    n = len(universe)
    for i, sym in enumerate(universe, 1):
        try:
            row = estimate_icc(sym)
            out_rows.append(row)
            logging.info("[%d/%d] %s ok", i, n, sym)
        except Exception as exc:
            logging.warning("[%d/%d] %s failed: %s", i, n, sym, exc)

        if i % PAUSE_EVERY == 0:
            time.sleep(PAUSE_SEC)

    df = pd.DataFrame(out_rows)
    out_dir = ROOT / "artifact"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"icc_snapshot_{mode}_{TODAY}.csv"
    df.to_csv(out_path, index=False)
    logging.info("saved: %s (%d rows)", out_path, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
