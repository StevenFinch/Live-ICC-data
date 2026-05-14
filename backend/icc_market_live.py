from __future__ import annotations

"""
backend/icc_market_live.py

Live Implied Cost of Capital (ICC) estimator.

This version preserves the original ICC estimator and output filenames while
upgrading the U.S. universe source to Nasdaq Trader official symbol directories
plus the previous DataHub fallback.
"""

import datetime as dt
import io
import json
import logging
import os
import pathlib
import re
import sys
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from yfinance import exceptions as yfexc

try:
    from backend.us_universe_sources import get_us_common_tickers
except Exception:
    get_us_common_tickers = None  # type: ignore


ROOT = pathlib.Path(__file__).resolve().parents[1]
COVERAGE_DIR = ROOT / "data" / "coverage"
PAUSE_EVERY, PAUSE_SEC = 20, 0.6
G_MIN, G_MAX = 0.01, 0.75
BAD_SYM = re.compile(r"[^A-Z\.\-]")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _today_et() -> dt.date:
    """Return today in America/New_York for consistent file naming."""
    if ZoneInfo is None:
        return dt.datetime.utcnow().date()
    return dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo("America/New_York")).date()


_DATAHUB = {
    "nasdaq": "https://raw.githubusercontent.com/datasets/nasdaq-listings/main/data/nasdaq-listed.csv",
    "nyse": "https://raw.githubusercontent.com/datasets/nyse-other-listings/main/data/nyse-listed.csv",
    "amex": "https://raw.githubusercontent.com/datasets/nyse-other-listings/main/data/other-listed.csv",
}


@lru_cache(maxsize=None)
def _legacy_get_us_tickers() -> List[str]:
    """Load the legacy DataHub ticker universe as a fallback."""
    symbols: set[str] = set()
    for name, url in _DATAHUB.items():
        try:
            df = pd.read_csv(url, dtype=str, usecols=[0])
            col = df.columns[0]
            raw = df[col].astype(str).str.replace(r"\.", "-", regex=True).str.upper()
            symbols.update(s for s in raw if s and not BAD_SYM.search(s))
            logging.info("%s legacy tickers loaded: %d", name.upper(), len(raw))
        except Exception as exc:
            logging.warning("failed to load legacy %s: %s", name, exc)
    logging.info("total legacy US symbols: %d", len(symbols))
    return sorted(symbols)


@lru_cache(maxsize=None)
def get_us_tickers() -> List[str]:
    """Return broad U.S. common-equity candidates with legacy fallback."""
    if get_us_common_tickers is not None:
        try:
            tickers = get_us_common_tickers(write_report=True)
            if len(tickers) >= 3000:
                logging.info("Nasdaq Trader US common-equity candidates: %d", len(tickers))
                return tickers
            logging.warning("Nasdaq Trader universe too small (%d); falling back to legacy.", len(tickers))
        except Exception as exc:
            logging.warning("Nasdaq Trader universe failed; falling back to legacy: %s", exc)
    return _legacy_get_us_tickers()


_INDEX_URL = {
    "sp500": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
    "sp100": "https://en.wikipedia.org/wiki/S%26P_100",
    "dow30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "ndx100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}


def _get(url: str, max_retries: int = 3, timeout: int = 20) -> str:
    """Fetch a URL with retry logic."""
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
    """Find the likely symbol/ticker column."""
    candidates = [str(c) for c in df.columns]
    lowers = {c.lower(): c for c in candidates}
    for key in ("symbol", "ticker", "code"):
        if key in lowers:
            return lowers[key]
    return candidates[0]


def _fetch_index_tickers(url: str) -> list[str]:
    """Fetch index tickers from CSV or HTML tables."""
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
    """Return index constituent tickers."""
    return _fetch_index_tickers(_INDEX_URL[code])


def _eps_path(fe1: float, g2: float, T: int = 15, g_long: float = 0.04) -> List[float]:
    """Build the expected EPS path with a fade to long-run growth."""
    out = [fe1, fe1 * (1 + g2)]
    fade = np.exp(np.log(g_long / g2) / T)
    for _ in range(3, T + 2):
        g2 *= fade
        out.append(out[-1] * (1 + g2))
    return out


def _pv(eps: List[float], b1: float, r: float, T: int = 15, g_long: float = 0.04) -> float:
    """Compute present value of payout stream."""
    b_ss = np.clip(g_long / r, 0, 1)
    step = (b1 - b_ss) / T
    pv = 0.0
    for k in range(1, T + 1):
        b_k = np.clip(b1 - step * (k - 1), 0, 1)
        pv += eps[k - 1] * (1 - b_k) / (1 + r) ** k
    tv = eps[-1] / (r * (1 + r) ** T)
    return pv + tv


def _solve_icc(price: float, fe1: float, g2: float, div: float) -> Optional[float]:
    """Solve for the implied cost of capital."""
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


def _trailing_fallback_enabled() -> bool:
    """Return whether trailing EPS fallback is enabled."""
    return os.environ.get("ICC_ALLOW_TRAILING_EPS_FALLBACK", "1").strip() not in {"0", "false", "False"}


def _fetch(sym: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Fetch one ticker and return both the ICC record and diagnostics."""
    diag: Dict[str, Any] = {
        "ticker": sym,
        "fetch_status": "unknown",
        "has_price": False,
        "has_market_cap": False,
        "has_forward_eps": False,
        "has_trailing_eps": False,
        "has_icc": False,
        "icc_method": None,
        "unavailable_reason": None,
        "mktcap": None,
    }

    if BAD_SYM.search(sym):
        diag["fetch_status"] = "failed"
        diag["unavailable_reason"] = "invalid_symbol"
        return None, diag

    try:
        tkr = yf.Ticker(sym)
        info = tkr.info or {}

        price = info.get("regularMarketPrice")
        diag["has_price"] = price is not None and np.isfinite(price)
        diag["mktcap"] = info.get("marketCap")
        diag["has_market_cap"] = diag["mktcap"] is not None and pd.notna(diag["mktcap"])

        if price is None or not np.isfinite(price):
            diag["fetch_status"] = "failed"
            diag["unavailable_reason"] = "missing_price"
            return None, diag

        fe1, g2 = None, None
        icc_method = "forecast_icc"

        try:
            trend = tkr.get_eps_trend(as_dict=True) or {}
            curr = trend.get("current", {})
            fe_now, fe_next = curr.get("0y"), curr.get("+1y")
            if fe_now and fe_next and fe_now > 0:
                fe1 = float(fe_now)
                g2 = float(np.clip(fe_next / fe_now - 1, G_MIN, G_MAX))
                diag["has_forward_eps"] = True
        except Exception:
            pass

        if fe1 is None:
            fwd = info.get("forwardEps")
            if fwd is not None and pd.notna(fwd) and fwd > 0:
                fe1 = float(fwd)
                diag["has_forward_eps"] = True
                icc_method = "forecast_icc"

        if g2 is None and info.get("earningsGrowth") is not None:
            try:
                g2 = float(np.clip(info["earningsGrowth"], G_MIN, G_MAX))
            except Exception:
                g2 = None

        trailing = info.get("trailingEps")
        diag["has_trailing_eps"] = trailing is not None and pd.notna(trailing) and trailing > 0

        if (fe1 is None or fe1 <= 0) and _trailing_fallback_enabled() and diag["has_trailing_eps"]:
            fe1 = float(trailing)
            icc_method = "trailing_eps_fallback_icc"

        if fe1 is None or fe1 <= 0:
            diag["fetch_status"] = "failed"
            diag["unavailable_reason"] = "missing_positive_eps"
            return None, diag

        if g2 is None:
            g2 = 0.04

        rec = dict(
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
            icc_method=icc_method,
        )

        icc = _solve_icc(rec["price"], rec["FE1"], rec["g2"], rec["dividend"])
        if icc is None or pd.isna(icc):
            diag["fetch_status"] = "failed"
            diag["unavailable_reason"] = "icc_solver_failed"
            return None, diag

        rec["ICC"] = icc
        diag["fetch_status"] = "ok"
        diag["has_icc"] = True
        diag["icc_method"] = icc_method
        return rec, diag

    except yfexc.YFRateLimitError:
        time.sleep(1.5)
        return _fetch(sym)
    except Exception as exc:
        diag["fetch_status"] = "failed"
        diag["unavailable_reason"] = f"exception:{type(exc).__name__}"
        return None, diag


def get_live_panel_with_diagnostics(symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the live ICC panel and a diagnostics table for all input symbols."""
    rows: List[Dict[str, Any]] = []
    diags: List[Dict[str, Any]] = []

    for i, sym in enumerate(symbols, 1):
        rec, diag = _fetch(sym)
        diags.append(diag)
        if rec:
            rows.append(rec)
        if i % PAUSE_EVERY == 0:
            time.sleep(PAUSE_SEC)

    df = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diags)

    if df.empty:
        logging.warning("all fetches failed")
        return df, diag_df

    for col in ["bvps", "shares", "mktcap"]:
        if col not in df.columns:
            df[col] = np.nan

    df["bm"] = np.where(
        (df.bvps > 0) & (df.shares > 0) & (df.mktcap > 0),
        (df.bvps * df.shares) / df.mktcap,
        np.nan,
    )

    df = df.dropna(subset=["ICC"]).reset_index(drop=True)
    df["date"] = _today_et().isoformat()

    logging.info("panel rows: %d / universe %d", len(df), len(symbols))
    return df, diag_df


def get_live_panel(symbols: List[str]) -> pd.DataFrame:
    """Compatibility wrapper that returns only the successful ICC panel."""
    panel, _diag = get_live_panel_with_diagnostics(symbols)
    return panel


def _write_coverage(universe_name: str, symbols: List[str], panel: pd.DataFrame, diag: pd.DataFrame) -> None:
    """Write coverage diagnostics for the current run."""
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    d_et = _today_et()
    tag = f"{d_et.year}_{d_et.month:02d}{d_et.day:02d}"

    diag_out = COVERAGE_DIR / f"{universe_name}_diagnostics_{tag}.csv"
    summary_out = COVERAGE_DIR / f"{universe_name}_coverage_summary_{tag}.json"

    diag.to_csv(diag_out, index=False)

    n_input = len(symbols)
    n_ok = int(len(panel))
    method_series = panel.get("icc_method", pd.Series(dtype=str)) if not panel.empty else pd.Series(dtype=str)
    n_forecast = int((method_series == "forecast_icc").sum())
    n_trailing = int((method_series == "trailing_eps_fallback_icc").sum())

    fetched_mktcap_total = pd.to_numeric(diag.get("mktcap", pd.Series(dtype=float)), errors="coerce").dropna().sum()
    icc_mktcap_total = pd.to_numeric(panel.get("mktcap", pd.Series(dtype=float)), errors="coerce").dropna().sum()
    mktcap_coverage = float(icc_mktcap_total / fetched_mktcap_total) if fetched_mktcap_total and fetched_mktcap_total > 0 else None

    summary = {
        "date": d_et.isoformat(),
        "universe": universe_name,
        "n_input_symbols": int(n_input),
        "n_icc_eligible": int(n_ok),
        "count_coverage": float(n_ok / n_input) if n_input else None,
        "n_forecast_icc": n_forecast,
        "n_trailing_eps_fallback_icc": n_trailing,
        "fetched_mktcap_total": float(fetched_mktcap_total) if pd.notna(fetched_mktcap_total) else None,
        "icc_mktcap_total": float(icc_mktcap_total) if pd.notna(icc_mktcap_total) else None,
        "mktcap_coverage_among_fetched": mktcap_coverage,
        "diagnostics_csv": str(diag_out.relative_to(ROOT)),
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _save_outputs(universe_name: str, panel: pd.DataFrame) -> pathlib.Path:
    """Save standard panel outputs."""
    d_et = _today_et()
    yyyymm = f"{d_et.year}{d_et.month:02d}"
    tag = f"{d_et.year}_{d_et.month:02d}{d_et.day:02d}"

    out_dir = ROOT / "data" / yyyymm
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / f"icc_live_{universe_name}_{tag}.csv"
    panel.to_csv(out, index=False)

    sample = ROOT / "icc_live_sample.csv"
    panel.to_csv(sample, index=False)

    logging.info("saved %s (%d rows)", out, len(panel))
    logging.info("saved %s (%d rows)", sample, len(panel))
    return out


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

    panel, diag = get_live_panel_with_diagnostics(universe)
    universe_name = arg if (arg == "usall" or arg in _INDEX_URL) else "custom"

    _save_outputs(universe_name, panel)
    _write_coverage(universe_name, universe, panel, diag)

    print(panel.head())
