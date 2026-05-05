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
RUN_ORDER = ["sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"]


def today_et() -> datetime:
    """Return current New York datetime."""
    return datetime.now(ZoneInfo("America/New_York"))


def month_dir() -> pathlib.Path:
    """Return current month output directory."""
    now = today_et()
    out = REPO / "data" / f"{now.year}{now.month:02d}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def date_tag() -> str:
    """Return date tag used by raw ICC filenames."""
    now = today_et()
    return f"{now.year}_{now.month:02d}{now.day:02d}"


def existing_nonempty_file(universe: str) -> pathlib.Path | None:
    """Return existing non-empty output file for today's index universe."""
    for p in sorted(month_dir().glob(f"icc_live_{universe}_{date_tag()}*.csv")):
        try:
            if p.stat().st_size > 0:
                return p
        except FileNotFoundError:
            continue
    return None


def get_text(url: str, timeout: int = 30, retries: int = 3) -> str:
    """Fetch HTML with retries."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def read_html_tables(url: str) -> list[pd.DataFrame]:
    """Read all HTML tables from a URL."""
    tables = pd.read_html(io.StringIO(get_text(url)))
    if not tables:
        raise RuntimeError(f"No HTML tables found at {url}")
    return tables


def normalize_symbols(series: pd.Series) -> list[str]:
    """Normalize symbols into Yahoo-compatible tickers."""
    out = series.astype(str).str.strip().str.replace(r"\s+", "", regex=True).str.replace(".", "-", regex=False).str.upper().tolist()
    return [s for s in out if s and not BAD_SYM.search(s)]


def find_table_by_columns(tables: list[pd.DataFrame], required_cols: list[str], min_rows: int) -> pd.DataFrame:
    """Find the first table with all required columns and enough rows."""
    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        lower_cols = {c.lower(): c for c in cols}
        if all(col.lower() in lower_cols for col in required_cols) and len(df) >= min_rows:
            out = df.copy()
            out.columns = cols
            return out
    raise RuntimeError(f"Could not find table with columns {required_cols}")


def fetch_sp100() -> list[str]:
    """Fetch S&P 100 constituents."""
    df = find_table_by_columns(read_html_tables(INDEX_URLS["sp100"]), ["Symbol", "Name", "Sector"], 90)
    return normalize_symbols(df["Symbol"])


def fetch_dow30() -> list[str]:
    """Fetch Dow Jones Industrial Average constituents."""
    df = find_table_by_columns(read_html_tables(INDEX_URLS["dow30"]), ["Company", "Exchange", "Symbol", "Sector"], 30)
    return normalize_symbols(df["Symbol"])[:30]


def fetch_ndx100() -> list[str]:
    """Fetch Nasdaq-100 constituents."""
    df = find_table_by_columns(read_html_tables(INDEX_URLS["ndx100"]), ["Ticker", "Company"], 90)
    return normalize_symbols(df["Ticker"])


def fetch_sp400() -> list[str]:
    """Fetch S&P 400 constituents."""
    df = find_table_by_columns(read_html_tables(INDEX_URLS["sp400"]), ["Symbol", "Security"], 350)
    return normalize_symbols(df["Symbol"])


def fetch_sp600() -> list[str]:
    """Fetch S&P 600 constituents."""
    df = find_table_by_columns(read_html_tables(INDEX_URLS["sp600"]), ["Symbol", "Security"], 500)
    return normalize_symbols(df["Symbol"])


def fetch_rut1000() -> list[str]:
    """Fetch Russell 1000 constituents."""
    df = find_table_by_columns(read_html_tables(INDEX_URLS["rut1000"]), ["Company", "Symbol"], 800)
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
    """Fetch constituents, compute ICC panel, and save the result."""
    existing = existing_nonempty_file(universe)
    if existing is not None:
        return True, f"skip existing {existing.name}"
    tickers = FETCHERS[universe]()
    deduped = list(dict.fromkeys(tickers))
    panel = get_live_panel(deduped)
    if panel is None or panel.empty:
        raise RuntimeError(f"ICC panel is empty for {universe}")
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
