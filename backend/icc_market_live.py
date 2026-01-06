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
import io
import logging
import pathlib
import re
import sys
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

# timezone (py3.11 has zoneinfo)
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

import requests

# third-party
import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import exceptions as yfexc

# -----------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]

PAUSE_EVERY, PAUSE_SEC = 20, 0.6
G_MIN, G_MAX = 0.01, 0.75
BAD_SYM = re.compile(r"[^A-Z\.\-]")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s: %(message)s")


def _today_et() -> dt.date:
    """Return 'today' in America/New_York for consistent file naming in GH Actions."""
    if ZoneInfo is None:
        return dt.datetime.utcnow().date()
    return dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo("America/New_York")).date()


# -----------------------------------------------------------------------------
# 1  Ticker universes
# -----------------------------------------------------------------------------
_DATAHUB = {
    # datahub.io core links 404 now -> GitHub raw CSV mirrors
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


_INDEX_URL = {
    # avoid Wikipedia (403 in CI); use stable CSV for SP500
    "sp500":  "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
    "sp100":  "https://en.wikipedia.org/wiki/S%26P_100",
    "dow30":  "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "ndx100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}


def _get(url: str, max_retries: int = 3, timeout: int = 20) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to GET {url} after {max_retries} tries") from last_err


def _find_symbol_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns]
    lowers = {c.lower(): c for c in candidates}
    for key in ("symbol", "ticker", "code"):
        if key in lowers:
            return lowers[key]
    return candidates[0]


def _fetch_index_tickers(url: str) -> list[str]:
    # CSV mode (SP500)
    if url.lower().endswith(".csv"):
        df = pd.read_csv(url, dtype=str)
        sym_col = _find_symbol_column(df)
        ytickers = (
            df[sym_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", "", regex=True)
            .str.replace(".", "-", regex=False)
            .str.upper()
            .tolist()
        )
        ytickers = [s for s in ytickers if s and not BAD_SYM.search(s)]
        if len(ytickers) < 30:
            raise RuntimeError(f"Only parsed {len(ytickers)} symbols from {url}; source may have changed.")
        return ytickers

    # Wikipedia HTML mode (sp100/dow30/ndx100)
    html = _get(url)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No tables found on {url}")
    best = max(tables, key=len)
    sym_col = _find_symbol_column(best)
    df = best.dropna(subset=[sym_col]).copy()
    ytickers = (
        df[sym_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )
    ytickers = [s for s in ytickers if s and not BAD_SYM.search(s)]
    if len(ytickers) < 30:
        raise RuntimeError(f"Only parsed {len(ytickers)} symbols from {url}; page format may have changed.")
    return ytickers


@lru_cache(maxsize=None)
def _index_tickers(code: str) -> List[str]:
    return _fetch_index_tickers(_INDEX_URL[code])


# -----------------------------------------------------------------------------
# 2  Li–Ng–Swaminathan helpers
# -----------------------------------------------------------------------------
def _eps_path(fe1: float, g2: float, T: int = 15, g_long: float = 0.04) -> List[float]:
    out = [fe1, fe1 * (1 + g2)]
    fade = np.exp(np.log(g_long / g2) / T)
    for _ in range(3, T + 2):
        g2 *= fade
        out.append(out[-1] * (1 + g2))
    return out


def _pv(eps: List[float], b1: float, r: float, T: int = 15, g_long: float = 0.04) -> float:
    b_ss = np.clip(g_long / r, 0, 1)
    step = (b1 - b_ss) / T
    pv = 0.0
    for k in range(1, T + 1):
        b_k = np.clip(b1 - step * (k - 1), 0, 1)
        pv += eps[k - 1] * (1 - b_k) / (1 + r) ** k
    tv = eps[-1] / (r * (1 + r) ** T)
    return pv + tv


def _solve_icc(price: float, fe1: float, g2: float, div: float) -> Optional[float]:
    if min(price, fe1) <= 0 or not np.isfinite(price):
        return None
    eps = _eps_path(fe1, g2)
    b1 = np.clip(1 - div / fe1, 0, 1)
    lo, hi = 0.01, 0.40
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pv = _pv(eps, b1, mid)
        lo, hi = (mid, hi) if pv > price else (lo, mid)
    return mid


# -----------------------------------------------------------------------------
# 3  yfinance fetch (sequential)
# -----------------------------------------------------------------------------
def _fetch(sym: str) -> Optional[Dict[str, Any]]:
    if BAD_SYM.search(sym):
        return None
    try:
        tkr = yf.Ticker(sym)
        info = tkr.info or {}
        price = info.get("regularMarketPrice")
        if price is None:
            return None

        fe1, g2 = None, None
        try:
            trend = tkr.get_eps_trend(as_dict=True) or {}
            curr = trend.get("current", {})
            fe_now, fe_next = curr.get("0y"), curr.get("+1y")
            if fe_now and fe_next and fe_now > 0:
                fe1 = fe_now
                g2 = np.clip(fe_next / fe_now - 1, G_MIN, G_MAX)
        except Exception:
            pass

        if fe1 is None:
            fe1 = info.get("forwardEps")
        if g2 is None and info.get("earningsGrowth") is not None:
            g2 = np.clip(info["earningsGrowth"], G_MIN, G_MAX)
        if fe1 is None or fe1 <= 0:
            return None
        if g2 is None:
            g2 = 0.04

        return dict(
            ticker=sym,
            price=price,
            dividend=info.get("dividendRate") or 0.0,
            mktcap=info.get("marketCap"),
            shares=info.get("sharesOutstanding"),
            bvps=info.get("bookValue"),
            sector=info.get("sector"),
            name=info.get("shortName") or info.get("longName"),
            FE1=fe1,
            g2=g2,
        )
    except yfexc.YFRateLimitError:
        time.sleep(1.5)
        return _fetch(sym)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# 4  Panel builder
# -----------------------------------------------------------------------------
def get_live_panel(symbols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, sym in enumerate(symbols, 1):
        rec = _fetch(sym)
        if rec:
            rows.append(rec)
        if i % PAUSE_EVERY == 0:
            time.sleep(PAUSE_SEC)

    df = pd.DataFrame(rows)
    if df.empty:
        logging.warning("all fetches failed")
        return df

    df["bm"] = np.where(
        (df.bvps > 0) & (df.shares > 0) & (df.mktcap > 0),
        (df.bvps * df.shares) / df.mktcap,
        np.nan,
    )
    df["ICC"] = df.apply(lambda r: _solve_icc(r.price, r.FE1, r.g2, r.dividend), axis=1)
    df = df.dropna(subset=["ICC"]).reset_index(drop=True)
    df["date"] = _today_et().isoformat()
    logging.info("panel rows: %d / universe %d", len(df), len(symbols))
    return df


# -----------------------------------------------------------------------------
# 5  CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: icc_market_live.py {usall|sp500|sp100|dow30|ndx100|TICKERS…}")
        sys.exit(0)

    arg = sys.argv[1].lower()
    if arg == "usall":
        universe = get_us_tickers()
    elif arg in _INDEX_URL:
        universe = _index_tickers(arg)
    else:
        universe = [s.upper() for s in sys.argv[1:]]

    panel = get_live_panel(universe)

    # --- OUTPUT LOCATION (your requested path) ---
    d_et = _today_et()
    yyyymm = f"{d_et.year}{d_et.month:02d}"
    tag = f"{d_et.year}_{d_et.month:02d}{d_et.day:02d}"

    out_dir = ROOT / "data" / yyyymm
    out_dir.mkdir(parents=True, exist_ok=True)

    universe_name = arg if (arg == "usall" or arg in _INDEX_URL) else "custom"
    out = out_dir / f"icc_live_{universe_name}_{tag}.csv"
    panel.to_csv(out, index=False)

    # --- COMPAT FILE (so run_daily_icc.py stops crashing) ---
    # run_daily_icc.py expects this file to exist after running the command.
    sample = ROOT / "icc_live_sample.csv"
    panel.to_csv(sample, index=False)

    print(panel.head())
    logging.info("saved %s (%d rows)", out, len(panel))
    logging.info("saved %s (%d rows)", sample, len(panel))
