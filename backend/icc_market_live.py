from __future__ import annotations
"""
backend/icc_market_live.py
- Market-wide ICC (Li–Ng–Swaminathan style) for a chosen universe (sp500/sp100/dow30/ndx100)
- Uses yfinance for prices / fundamentals
- Writes JSON snapshots for the frontend

(Only URL-related fixes applied: DataHub 404 -> GitHub raw CSV. Also sp500 uses a CSV source to avoid Wikipedia 403 in CI.)
"""

import io
import json
import logging
import os
import re
import sys
import time
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# If you run in GitHub Actions, keep requests polite
MAX_TICKERS, PAUSE_SEC = 20, 0.6       # rate-limit safety
G_MIN, G_MAX = -0.5, 0.8               # growth clamp
R_F = 0.04                             # simple rf placeholder

BAD_SYM = re.compile(r"[^A-Z0-9\-\^]")  # allow '-' and '^' (Yahoo style)

# -----------------------------------------------------------------------------
# 0  US symbol universe (DataHub links used to 404; now use GitHub raw)
# -----------------------------------------------------------------------------
_DATAHUB = {
    "nasdaq": "https://raw.githubusercontent.com/datasets/nasdaq-listings/main/data/nasdaq-listed.csv",
    "nyse":   "https://raw.githubusercontent.com/datasets/nyse-other-listings/main/data/nyse-listed.csv",
    "amex":   "https://raw.githubusercontent.com/datasets/nyse-other-listings/main/data/other-listed.csv",
}


@lru_cache(maxsize=None)
def get_us_tickers() -> t.List[str]:
    symbols: t.Set[str] = set()
    for name, url in _DATAHUB.items():
        try:
            df = pd.read_csv(url, usecols=[0], dtype=str)
            raw = (
                df.iloc[:, 0]
                .dropna()
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", "", regex=True)
                .str.replace(".", "-", regex=False)
                .str.upper()
                .tolist()
            )
            raw = [s for s in raw if s and not BAD_SYM.search(s)]
            symbols.update(raw)
            logging.info("%s tickers loaded: %d", name.upper(), len(raw))
        except Exception as exc:
            logging.warning("failed to load %s: %s", name, exc)
    logging.info("total US symbols: %d", len(symbols))
    return sorted(symbols)


_WIKI_TICK = re.compile(r"(symbol|ticker)", re.I)
_INDEX_URL = {
    # Wikipedia 403s in some CI environments -> use a stable CSV source for SP500
    "sp500":  "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
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
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"GET failed after {max_retries} tries: {url}") from last_err


def _find_symbol_column(df: pd.DataFrame) -> str:
    """Best-effort symbol/ticker column finder."""
    lowers = {c.lower(): c for c in df.columns}
    for key in ["symbol", "ticker", "act symbol"]:
        if key in lowers:
            return lowers[key]
    # fallback: fuzzy match
    candidates = list(df.columns)
    for c in candidates:
        if _WIKI_TICK.search(str(c)):
            return c
    # fallback: first column
    return candidates[0]


def _fetch_index_from_wikipedia(url: str) -> list[str]:
    """Return Yahoo-friendly tickers from an index URL.

    Supports:
      * Wikipedia HTML pages (parsed via read_html)
      * CSV constituents (e.g., GitHub raw constituents.csv)
    """
    # CSV mode (avoids Wikipedia 403s in some CI environments)
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
        ytickers = [sym for sym in ytickers if sym and not BAD_SYM.search(sym)]
        if len(ytickers) < 30:
            raise RuntimeError(f"Only parsed {len(ytickers)} symbols from {url}; source may have changed.")
        return ytickers

    # HTML mode (Wikipedia)
    html = _get(url)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No tables found on {url}")

    # Heuristic: pick the table with the most rows that also has a symbol-like column
    best: t.Optional[pd.DataFrame] = None
    best_rows = -1
    for tbl in tables:
        try:
            sym_col = _find_symbol_column(tbl)
        except Exception:
            continue
        rows = len(tbl)
        if rows > best_rows and sym_col in tbl.columns:
            best = tbl
            best_rows = rows
    if best is None:
        best = max(tables, key=len)

    sym_col = _find_symbol_column(best)
    df = best.dropna(subset=[sym_col]).copy()
    df[sym_col] = df[sym_col].astype(str).str.strip()

    # raw symbols (e.g., "BRK.B") → Yahoo symbols ("BRK-B")
    ytickers = (
        df[sym_col]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )
    # filter weird symbols
    ytickers = [s for s in ytickers if s and not BAD_SYM.search(s)]

    # sanity check (avoid partial parses)
    if len(ytickers) < 30:
        raise RuntimeError(f"Only parsed {len(ytickers)} symbols from {url}; page format may have changed.")

    return ytickers


@lru_cache(maxsize=None)
def _index_tickers(code: str) -> t.List[str]:
    """Fetch index constituents and return Yahoo-friendly tickers ('.' → '-')."""
    url = _INDEX_URL[code]
    return _fetch_index_from_wikipedia(url)

# -----------------------------------------------------------------------------
# 2  Li–Ng–Swaminathan helpers
# -----------------------------------------------------------------------------
def _eps_path(fe1: float, g2: float,
              T: int = 15,
              g_terminal: float = 0.03) -> np.ndarray:
    """Construct a 15-year EPS path with short-term FE and long-term terminal growth."""
    fe1 = float(np.clip(fe1, G_MIN, G_MAX))
    g2 = float(np.clip(g2, G_MIN, G_MAX))
    eps = np.zeros(T)
    eps[0] = 1.0 * (1.0 + fe1)  # year1
    if T > 1:
        # years 2..5: blend towards g2
        for k in range(1, min(5, T)):
            w = (k) / 4.0  # 0.25..1.0
            g = (1 - w) * fe1 + w * g2
            eps[k] = eps[k - 1] * (1.0 + g)
        # years 6..15: fade g2 to terminal
        for k in range(5, T):
            w = (k - 5) / (T - 6 + 1e-9)
            g = (1 - w) * g2 + w * g_terminal
            eps[k] = eps[k - 1] * (1.0 + g)
    return eps


def _pv_cashflows(cf: np.ndarray, r: float) -> float:
    """Present value of annual cashflows with discount rate r."""
    r = float(max(r, -0.99))
    disc = np.array([(1.0 + r) ** (i + 1) for i in range(len(cf))], dtype=float)
    return float(np.sum(cf / disc))


def _solve_icc(price: float,
               fe1: float,
               g2: float,
               payout: float = 0.5,
               g_terminal: float = 0.03) -> float:
    """Solve for r such that PV(dividends) == price using a simple bisection."""
    if price <= 0 or not np.isfinite(price):
        return float("nan")

    eps = _eps_path(fe1=fe1, g2=g2, T=15, g_terminal=g_terminal)
    div = payout * eps

    # terminal value: Gordon on year 15 dividend
    dT = div[-1]
    def f(r: float) -> float:
        if r <= g_terminal + 1e-6:
            return 1e9
        pv = _pv_cashflows(div, r)
        tv = dT * (1.0 + g_terminal) / (r - g_terminal)
        pv += tv / ((1.0 + r) ** len(div))
        return pv - price

    lo, hi = -0.5, 1.5
    flo, fhi = f(lo), f(hi)
    if not np.isfinite(flo) or not np.isfinite(fhi):
        return float("nan")
    if flo * fhi > 0:
        # no sign change; return midpoint-ish
        return float("nan")

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not np.isfinite(fm):
            return float("nan")
        if abs(fm) < 1e-6:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


# -----------------------------------------------------------------------------
# 3  yfinance helpers
# -----------------------------------------------------------------------------
def _yf_info(sym: str) -> dict:
    """Robust yfinance.info fetch (handles occasional None)."""
    tkr = yf.Ticker(sym)
    try:
        info = tkr.info or {}
    except Exception:
        info = {}
    return info


def _latest_price(sym: str) -> float:
    """Fetch last close/regular price."""
    tkr = yf.Ticker(sym)
    try:
        hist = tkr.history(period="5d", auto_adjust=False)
        if len(hist):
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    # fallback
    info = _yf_info(sym)
    for k in ["regularMarketPrice", "previousClose"]:
        v = info.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return float("nan")


def _fe1_g2_from_info(info: dict) -> tuple[float, float]:
    """Approximate FE1 and g2 from yfinance fields (very rough)."""
    # Try forward EPS vs trailing EPS as FE1 proxy
    fe1 = np.nan
    try:
        fwd = info.get("forwardEps")
        trl = info.get("trailingEps")
        if fwd and trl and float(trl) != 0:
            fe1 = float(fwd) / float(trl) - 1.0
    except Exception:
        fe1 = np.nan

    # Long-term growth proxy: earningsGrowth if available
    g2 = np.nan
    try:
        eg = info.get("earningsGrowth")
        if eg is not None:
            g2 = float(eg)
    except Exception:
        g2 = np.nan

    # Fill defaults if missing
    if not np.isfinite(fe1):
        fe1 = 0.05
    if not np.isfinite(g2):
        g2 = 0.05
    return float(fe1), float(g2)


# -----------------------------------------------------------------------------
# 4  Main pipeline
# -----------------------------------------------------------------------------
@dataclass
class Row:
    symbol: str
    price: float
    icc: float
    fe1: float
    g2: float


def build_index_snapshot(code: str) -> dict:
    tickers = _index_tickers(code)
    rows: t.List[dict] = []
    out_rows: t.List[Row] = []

    # limit to reduce API load
    todo = tickers[: MAX_TICKERS] if os.environ.get("FAST_TEST") else tickers

    for i, sym in enumerate(todo, 1):
        try:
            info = _yf_info(sym)
            price = _latest_price(sym)
            fe1, g2 = _fe1_g2_from_info(info)
            icc = _solve_icc(price=price, fe1=fe1, g2=g2, payout=0.5, g_terminal=0.03)
            out_rows.append(Row(sym, price, icc, fe1, g2))
        except Exception as exc:
            logging.warning("ticker failed %s: %s", sym, exc)

        if i % 5 == 0:
            time.sleep(PAUSE_SEC)

    for r in out_rows:
        rows.append({
            "symbol": r.symbol,
            "price": None if not np.isfinite(r.price) else r.price,
            "icc": None if not np.isfinite(r.icc) else r.icc,
            "fe1": r.fe1,
            "g2": r.g2,
        })

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "meta": {"updated_at": now, "universe": code, "n": len(rows)},
        "rows": rows,
    }
    return payload


def main(argv: t.List[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s: %(message)s",
    )

    code = (argv[1] if len(argv) > 1 else "sp500").lower().strip()
    if code not in _INDEX_URL:
        raise SystemExit(f"Unknown universe: {code}. Choose from {sorted(_INDEX_URL)}")

    snap = build_index_snapshot(code)

    out_path = DATA_DIR / f"{code}_index.json"
    out_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    logging.info("wrote %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
