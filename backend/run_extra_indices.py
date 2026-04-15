from __future__ import annotations

import io
import pathlib
import re
import sys
import time
from typing import Callable

import pandas as pd
import requests

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.icc_market_live import get_live_panel, _today_et  # type: ignore


BAD_SYM = re.compile(r"[^A-Z\.\-]")
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

INDEX_URLS = {
    "sp500": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
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


def _get_text(url: str, timeout: int = 30, retries: int = 3) -> str:
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


def _read_html_tables(url: str) -> list[pd.DataFrame]:
    html = _get_text(url)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No HTML tables found at {url}")
    return tables


def _normalize_symbols(series: pd.Series) -> list[str]:
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


def _find_table_by_columns(
    tables: list[pd.DataFrame],
    required_cols: list[str],
    min_rows: int,
) -> pd.DataFrame:
    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        lower_cols = {c.lower(): c for c in cols}
        if all(col.lower() in lower_cols for col in required_cols) and len(df) >= min_rows:
            df = df.copy()
            df.columns = cols
            return df
    raise RuntimeError(
        f"Could not find a table with columns {required_cols} and at least {min_rows} rows"
    )


def _fetch_sp500() -> list[str]:
    url = INDEX_URLS["sp500"]
    df = pd.read_csv(url, dtype=str)
    cols = {str(c).strip().lower(): c for c in df.columns}
    sym_col = cols.get("symbol")
    if sym_col is None:
        raise RuntimeError("sp500 CSV does not contain a Symbol column")
    syms = _normalize_symbols(df[sym_col])
    if len(syms) < 450:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for sp500")
    return syms


def _fetch_sp100() -> list[str]:
    tables = _read_html_tables(INDEX_URLS["sp100"])
    df = _find_table_by_columns(tables, ["Symbol", "Name", "Sector"], min_rows=90)
    syms = _normalize_symbols(df["Symbol"])
    if len(syms) < 90:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for sp100")
    return syms


def _fetch_dow30() -> list[str]:
    tables = _read_html_tables(INDEX_URLS["dow30"])
    df = _find_table_by_columns(tables, ["Company", "Exchange", "Symbol", "Sector"], min_rows=30)
    syms = _normalize_symbols(df["Symbol"])
    if len(syms) < 30:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for dow30")
    return syms[:30]


def _fetch_ndx100() -> list[str]:
    tables = _read_html_tables(INDEX_URLS["ndx100"])
    df = _find_table_by_columns(tables, ["Ticker", "Company"], min_rows=90)
    syms = _normalize_symbols(df["Ticker"])
    if len(syms) < 90:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for ndx100")
    return syms


def _fetch_sp400() -> list[str]:
    tables = _read_html_tables(INDEX_URLS["sp400"])
    df = _find_table_by_columns(tables, ["Symbol", "Security"], min_rows=350)
    syms = _normalize_symbols(df["Symbol"])
    if len(syms) < 350:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for sp400")
    return syms


def _fetch_sp600() -> list[str]:
    tables = _read_html_tables(INDEX_URLS["sp600"])
    df = _find_table_by_columns(tables, ["Symbol", "Security"], min_rows=500)
    syms = _normalize_symbols(df["Symbol"])
    if len(syms) < 500:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for sp600")
    return syms


def _fetch_rut1000() -> list[str]:
    tables = _read_html_tables(INDEX_URLS["rut1000"])
    df = _find_table_by_columns(tables, ["Company", "Symbol"], min_rows=900)
    syms = _normalize_symbols(df["Symbol"])
    if len(syms) < 900:
        raise RuntimeError(f"Parsed only {len(syms)} symbols for rut1000")
    return syms


FETCHERS: dict[str, Callable[[], list[str]]] = {
    "sp500": _fetch_sp500,
    "sp100": _fetch_sp100,
    "dow30": _fetch_dow30,
    "ndx100": _fetch_ndx100,
    "sp400": _fetch_sp400,
    "sp600": _fetch_sp600,
    "rut1000": _fetch_rut1000,
}


def _month_dir() -> pathlib.Path:
    d_et = _today_et()
    out_dir = REPO / "data" / f"{d_et.year}{d_et.month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _date_tag() -> str:
    d_et = _today_et()
    return f"{d_et.year}_{d_et.month:02d}{d_et.day:02d}"


def _save_panel(universe: str, panel: pd.DataFrame) -> pathlib.Path:
    out_dir = _month_dir()
    out_path = out_dir / f"icc_live_{universe}_{_date_tag()}.csv"
    panel.to_csv(out_path, index=False)
    return out_path


def _existing_nonempty_file(universe: str) -> pathlib.Path | None:
    out_dir = _month_dir()
    pattern = f"icc_live_{universe}_{_date_tag()}*.csv"
    for p in sorted(out_dir.glob(pattern)):
        try:
            if p.stat().st_size > 0:
                return p
        except FileNotFoundError:
            continue
    return None


def run_one(universe: str) -> tuple[bool, str]:
    existing = _existing_nonempty_file(universe)
    if existing is not None:
        return True, f"skip existing {existing.name}"

    fetcher = FETCHERS[universe]
    tickers = fetcher()

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for t in tickers:
        if t not in seen:
            deduped.append(t)
            seen.add(t)

    panel = get_live_panel(deduped)
    if panel.empty:
        raise RuntimeError(f"ICC panel is empty for {universe}")

    out_path = _save_panel(universe, panel)
    return True, f"saved {out_path.name} with {len(panel)} rows"


def main() -> None:
    ok = []
    bad = []

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

    print(f"[run_extra_indices] success = {ok}")
    print(f"[run_extra_indices] failed = {bad}")

    # Fail only if everything failed.
    if len(ok) == 0:
        raise RuntimeError(f"All extra index runs failed: {bad}")


if __name__ == "__main__":
    main()
