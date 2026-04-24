
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

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.icc_market_live import get_live_panel  # type: ignore


BAD_SYM = re.compile(r"[^A-Z\.\-]")
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

INDEX_URLS = {
    "sp100": "https://en.wikipedia.org/wiki/S%26P_100",
    "dow30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "ndx100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    "rut1000": "https://en.wikipedia.org/wiki/Russell_1000_Index",
}

RUN_ORDER = ["sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"]


def today_et() -> datetime:
    """Return the current America/New_York datetime."""
    return datetime.now(ZoneInfo("America/New_York"))


def month_dir() -> pathlib.Path:
    """Return output folder data/YYYYMM."""
    now = today_et()
    out_dir = REPO / "data" / f"{now.year}{now.month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def date_tag() -> str:
    """Return date tag YYYY_MMDD."""
    now = today_et()
    return f"{now.year}_{now.month:02d}{now.day:02d}"


def existing_nonempty_file(universe: str) -> pathlib.Path | None:
    """Return today's existing file if available."""
    for p in sorted(month_dir().glob(f"icc_live_{universe}_{date_tag()}*.csv")):
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def http_get(url: str, timeout: int = 30, retries: int = 3) -> str:
    """Fetch HTML with retries."""
    headers = {"User-Agent": USER_AGENT}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def read_tables(url: str) -> list[pd.DataFrame]:
    """Read HTML tables."""
    html = http_get(url)
    return pd.read_html(io.StringIO(html))


def normalize_symbols(series: pd.Series) -> list[str]:
    """Normalize ticker symbols."""
    syms = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )
    return [s for s in syms if s and not BAD_SYM.search(s)]


def find_table(tables: list[pd.DataFrame], required_cols: list[str], min_rows: int) -> pd.DataFrame:
    """Find the first table with required columns."""
    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        lower = {c.lower(): c for c in cols}
        if all(c.lower() in lower for c in required_cols) and len(df) >= min_rows:
            out = df.copy()
            out.columns = cols
            return out
    raise RuntimeError(f"Could not find table with columns {required_cols}")


def fetch_sp100() -> list[str]:
    """Fetch S&P 100 constituents."""
    df = find_table(read_tables(INDEX_URLS["sp100"]), ["Symbol", "Name"], 90)
    return normalize_symbols(df["Symbol"])


def fetch_dow30() -> list[str]:
    """Fetch Dow 30 constituents."""
    df = find_table(read_tables(INDEX_URLS["dow30"]), ["Company", "Symbol"], 30)
    return normalize_symbols(df["Symbol"])[:30]


def fetch_ndx100() -> list[str]:
    """Fetch Nasdaq 100 constituents."""
    df = find_table(read_tables(INDEX_URLS["ndx100"]), ["Ticker", "Company"], 90)
    return normalize_symbols(df["Ticker"])


def fetch_sp400() -> list[str]:
    """Fetch S&P 400 constituents."""
    df = find_table(read_tables(INDEX_URLS["sp400"]), ["Symbol", "Security"], 350)
    return normalize_symbols(df["Symbol"])


def fetch_sp600() -> list[str]:
    """Fetch S&P 600 constituents."""
    df = find_table(read_tables(INDEX_URLS["sp600"]), ["Symbol", "Security"], 500)
    return normalize_symbols(df["Symbol"])


def fetch_rut1000() -> list[str]:
    """Fetch Russell 1000 constituents."""
    df = find_table(read_tables(INDEX_URLS["rut1000"]), ["Company", "Symbol"], 900)
    return normalize_symbols(df["Symbol"])


FETCHERS: dict[str, Callable[[], list[str]]] = {
    "sp100": fetch_sp100,
    "dow30": fetch_dow30,
    "ndx100": fetch_ndx100,
    "sp400": fetch_sp400,
    "sp600": fetch_sp600,
    "rut1000": fetch_rut1000,
}


def run_one(universe: str) -> tuple[bool, str]:
    """Run one extra index job."""
    existing = existing_nonempty_file(universe)
    if existing:
        return True, f"skip existing {existing.name}"
    tickers = FETCHERS[universe]()
    deduped = list(dict.fromkeys(tickers))
    panel = get_live_panel(deduped)
    if panel.empty:
        raise RuntimeError(f"empty panel for {universe}")
    out = month_dir() / f"icc_live_{universe}_{date_tag()}.csv"
    panel.to_csv(out, index=False)
    return True, f"saved {out.name} with {len(panel)} rows"


def main() -> None:
    """Run all extra index jobs."""
    ok, bad = [], []
    for universe in RUN_ORDER:
        try:
            success, msg = run_one(universe)
            print(f"[run_extra_indices] {universe}: {msg}")
            ok.append(universe) if success else bad.append(universe)
        except Exception as exc:
            print(f"[run_extra_indices] FAILED on {universe}: {type(exc).__name__}: {exc}")
            bad.append(universe)
    print(f"[run_extra_indices] success={ok}")
    print(f"[run_extra_indices] failed={bad}")
    if not ok:
        raise RuntimeError(f"All extra index runs failed: {bad}")


if __name__ == "__main__":
    main()
