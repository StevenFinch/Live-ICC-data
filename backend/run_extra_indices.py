from __future__ import annotations

import io
import pathlib
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Callable

import pandas as pd
import requests

# Make the repo root importable when this file is executed as:
#   python backend/run_extra_indices.py
REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.icc_market_live import get_live_panel  # type: ignore


BAD_SYM = re.compile(r"[^A-Z\.\-]")
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

INDEX_URLS = {
    "sp100": "https://en.wikipedia.org/wiki/S%26P_100",
    "dow30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "ndx100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    "rut1000": "https://en.wikipedia.org/wiki/Russell_1000_Index",
}

RUN_ORDER = [
    "sp100",
    "dow30",
    "ndx100",
    "sp400",
    "sp600",
    "rut1000",
]


def today_et() -> datetime:
    """Return the current America/New_York datetime."""
    return datetime.now(ZoneInfo("America/New_York"))


def month_dir() -> pathlib.Path:
    """Return the output month folder under data/YYYYMM."""
    now = today_et()
    out_dir = REPO / "data" / f"{now.year}{now.month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def date_tag() -> str:
    """Return the filename date tag in YYYY_MMDD format."""
    now = today_et()
    return f"{now.year}_{now.month:02d}{now.day:02d}"


def existing_nonempty_file(universe: str) -> pathlib.Path | None:
    """Return an existing non-empty output file for today if it exists."""
    out_dir = month_dir()
    pattern = f"icc_live_{universe}_{date_tag()}*.csv"
    for p in sorted(out_dir.glob(pattern)):
        try:
            if p.stat().st_size > 0:
                return p
        except FileNotFoundError:
            continue
    return None


def save_panel(universe: str, panel: pd.DataFrame) -> pathlib.Path:
    """Save the computed ICC panel to the standard output location."""
    out_path = month_dir() / f"icc_live_{universe}_{date_tag()}.csv"
    panel.to_csv(out_path, index=False)
    return out_path


def get_text(url: str, timeout: int = 30, retries: int = 3) -> str:
    """Fetch webpage HTML with retries and a browser-like user agent."""
    headers = {"User-Agent": USER_AGENT}
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)

    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_err}")


def read_html_tables(url: str) -> list[pd.DataFrame]:
    """Parse all HTML tables from a page."""
    html = get_text(url)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No HTML tables found at {url}")
    return tables


def normalize_symbols(series: pd.Series) -> list[str]:
    """Normalize symbols into Yahoo-compatible format."""
    out = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )
    out = [s for s in out if s and not BAD_SYM.search(s)]
    return out


def find_table_by_columns(
    tables: list[pd.DataFrame],
    required_cols: list[str],
    min_rows: int,
) -> pd.DataFrame:
    """Find the first table containing all required columns."""
    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        lower_cols = {c.lower(): c for c in cols}

        if all(col.lower() in lower_cols for col in required_cols) and len(df) >= min_rows:
            out = df.copy()
            out.columns = cols
            return out

    raise RuntimeError(
        f"Could not find a table with columns {required_cols} and at least {min_rows} rows"
    )


def fetch_sp100() -> list[str]:
    """Fetch S&P 100 constituents from Wikipedia."""
    tables = read_html_tables(INDEX_URLS["sp100"])
    df = find_table_by_columns(tables, ["Symbol", "Name", "Sector"], min_rows=90)
    syms = normalize_symbols(df["Symbol"])
    if len(syms) < 90:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for sp100")
    return syms


def fetch_dow30() -> list[str]:
    """Fetch Dow Jones Industrial Average constituents from Wikipedia."""
    tables = read_html_tables(INDEX_URLS["dow30"])
    df = find_table_by_columns(tables, ["Company", "Exchange", "Symbol", "Sector"], min_rows=30)
    syms = normalize_symbols(df["Symbol"])
    if len(syms) < 30:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for dow30")
    return syms[:30]


def fetch_ndx100() -> list[str]:
    """Fetch Nasdaq-100 constituents from Wikipedia."""
    tables = read_html_tables(INDEX_URLS["ndx100"])
    df = find_table_by_columns(tables, ["Ticker", "Company"], min_rows=90)
    syms = normalize_symbols(df["Ticker"])
    if len(syms) < 90:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for ndx100")
    return syms


def fetch_sp400() -> list[str]:
    """Fetch S&P 400 constituents from Wikipedia."""
    tables = read_html_tables(INDEX_URLS["sp400"])
    df = find_table_by_columns(tables, ["Symbol", "Security"], min_rows=350)
    syms = normalize_symbols(df["Symbol"])
    if len(syms) < 350:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for sp400")
    return syms


def fetch_sp600() -> list[str]:
    """Fetch S&P 600 constituents from Wikipedia."""
    tables = read_html_tables(INDEX_URLS["sp600"])
    df = find_table_by_columns(tables, ["Symbol", "Security"], min_rows=500)
    syms = normalize_symbols(df["Symbol"])
    if len(syms) < 500:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for sp600")
    return syms


def fetch_rut1000() -> list[str]:
    """Fetch Russell 1000 constituents from Wikipedia."""
    tables = read_html_tables(INDEX_URLS["rut1000"])
    df = find_table_by_columns(tables, ["Company", "Symbol"], min_rows=900)
    syms = normalize_symbols(df["Symbol"])
    if len(syms) < 900:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for rut1000")
    return syms


FETCHERS: dict[str, Callable[[], list[str]]] = {
    "sp100": fetch_sp100,
    "dow30": fetch_dow30,
    "ndx100": fetch_ndx100,
    "sp400": fetch_sp400,
    "sp600": fetch_sp600,
    "rut1000": fetch_rut1000,
}


def run_one(universe: str) -> tuple[bool, str]:
    """Fetch constituents, compute live ICC panel, and save the result."""
    existing = existing_nonempty_file(universe)
    if existing is not None:
        return True, f"skip existing {existing.name}"

    tickers = FETCHERS[universe]()

    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for t in tickers:
        if t not in seen:
            deduped.append(t)
            seen.add(t)

    panel = get_live_panel(deduped)
    if panel.empty:
        raise RuntimeError(f"ICC panel is empty for {universe}")

    out_path = save_panel(universe, panel)
    return True, f"saved {out_path.name} with {len(panel)} rows"


def main() -> None:
    """Run all extra index jobs."""
    ok: list[str] = []
    bad: list[str] = []

    for universe in RUN_ORDER:
        try:
            success, msg = run_one(universe)
            print(f"[run_extra_indices] {universe}: {msg}")
            if success:
                ok.append(universe)
            else:
                bad.append(universe)
        except Exception as e:
            print(f"[run_extra_indices] FAILED on {universe}: {type(e).__name__}: {e}")
            bad.append(universe)

    print(f"[run_extra_indices] success={ok}")
    print(f"[run_extra_indices] failed={bad}")

    # Fail only if everything failed.
    if len(ok) == 0:
        raise RuntimeError(f"All extra index runs failed: {bad}")


if __name__ == "__main__":
    main()
