
from __future__ import annotations

import io
import pathlib
import re
import subprocess
import sys
import time
from typing import Callable

import pandas as pd
import requests

REPO = pathlib.Path(__file__).resolve().parents[1]
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
BAD_SYM = re.compile(r"[^A-Z\.\-]")

INDEX_URLS = {
    "sp100": "https://en.wikipedia.org/wiki/S%26P_100",
    "dow30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "ndx100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    "rut1000": "https://en.wikipedia.org/wiki/Russell_1000_Index",
}

RUN_ORDER = ["sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"]


def get_text(url: str, timeout: int = 30, retries: int = 3) -> str:
    headers = {"User-Agent": USER_AGENT}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_err = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_err}")


def read_html_tables(url: str) -> list[pd.DataFrame]:
    html = get_text(url)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No HTML tables found at {url}")
    return tables


def normalize_symbols(series: pd.Series) -> list[str]:
    vals = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )
    out = [v for v in vals if v and not BAD_SYM.search(v)]
    deduped = []
    seen = set()
    for v in out:
        if v not in seen:
            deduped.append(v)
            seen.add(v)
    return deduped


def find_table_by_columns(tables: list[pd.DataFrame], required_cols: list[str], min_rows: int) -> pd.DataFrame:
    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        lower_cols = {c.lower(): c for c in cols}
        if all(col.lower() in lower_cols for col in required_cols) and len(df) >= min_rows:
            df = df.copy()
            df.columns = cols
            return df
    raise RuntimeError(f"Could not find table with columns={required_cols}, min_rows={min_rows}")


def fetch_sp100() -> list[str]:
    tables = read_html_tables(INDEX_URLS["sp100"])
    df = find_table_by_columns(tables, ["Symbol", "Name"], 90)
    syms = normalize_symbols(df["Symbol"])
    if len(syms) < 90:
        raise RuntimeError(f"Only parsed {len(syms)} symbols for sp100")
    return syms


def fetch_dow30() -> list[str]:
    tables = read_html_tables(INDEX_URLS["dow30"])
    df = find_table_by_columns(tables, ["Company", "Exchange", "Symbol", "Sector"], 30)
    syms = normalize_symbols(df["Symbol"])
    if len(syms) < 30:
        raise RuntimeError(f"Only parsed {len(syms)} symbols for dow30")
    return syms[:30]


def fetch_ndx100() -> list[str]:
    tables = read_html_tables(INDEX_URLS["ndx100"])
    df = find_table_by_columns(tables, ["Ticker", "Company"], 90)
    syms = normalize_symbols(df["Ticker"])
    if len(syms) < 90:
        raise RuntimeError(f"Only parsed {len(syms)} symbols for ndx100")
    return syms


def fetch_sp400() -> list[str]:
    tables = read_html_tables(INDEX_URLS["sp400"])
    df = find_table_by_columns(tables, ["Symbol", "Security"], 350)
    syms = normalize_symbols(df["Symbol"])
    if len(syms) < 350:
        raise RuntimeError(f"Only parsed {len(syms)} symbols for sp400")
    return syms


def fetch_sp600() -> list[str]:
    tables = read_html_tables(INDEX_URLS["sp600"])
    df = find_table_by_columns(tables, ["Symbol", "Security"], 500)
    syms = normalize_symbols(df["Symbol"])
    if len(syms) < 500:
        raise RuntimeError(f"Only parsed {len(syms)} symbols for sp600")
    return syms


def fetch_rut1000() -> list[str]:
    tables = read_html_tables(INDEX_URLS["rut1000"])
    df = find_table_by_columns(tables, ["Company", "Symbol"], 900)
    syms = normalize_symbols(df["Symbol"])
    if len(syms) < 900:
        raise RuntimeError(f"Only parsed {len(syms)} symbols for rut1000")
    return syms


FETCHERS: dict[str, Callable[[], list[str]]] = {
    "sp100": fetch_sp100,
    "dow30": fetch_dow30,
    "ndx100": fetch_ndx100,
    "sp400": fetch_sp400,
    "sp600": fetch_sp600,
    "rut1000": fetch_rut1000,
}


def output_file_for(universe: str) -> pathlib.Path:
    from backend.icc_market_live import _today_et  # Imported lazily to match repo runtime.

    d_et = _today_et()
    out_dir = REPO / "data" / f"{d_et.year}{d_et.month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{d_et.year}_{d_et.month:02d}{d_et.day:02d}"
    return out_dir / f"icc_live_{universe}_{tag}.csv"


def run_one(universe: str) -> bool:
    out_path = output_file_for(universe)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[run_extra_indices] skip existing {out_path.name}")
        return True

    tickers = FETCHERS[universe]()
    if not tickers:
        raise RuntimeError(f"No tickers fetched for {universe}")

    cmd = [sys.executable, str(REPO / "backend" / "icc_market_live.py"), *tickers]
    print(f"[run_extra_indices] {universe}: {len(tickers)} tickers")
    subprocess.run(cmd, check=True)

    sample = REPO / "icc_live_sample.csv"
    if not sample.exists() or sample.stat().st_size <= 0:
        raise RuntimeError(f"Sample output not found after running {universe}")

    sample.replace(out_path)
    print(f"[run_extra_indices] saved {out_path.name}")
    return True


def main() -> None:
    ok = []
    bad = []
    for universe in RUN_ORDER:
        try:
            run_one(universe)
            ok.append(universe)
        except Exception as exc:
            print(f"[run_extra_indices] FAILED on {universe}: {type(exc).__name__}: {exc}")
            bad.append(universe)

    print(f"[run_extra_indices] success={ok}")
    print(f"[run_extra_indices] failed={bad}")

    if len(ok) == 0:
        raise RuntimeError(f"All extra index runs failed: {bad}")


if __name__ == "__main__":
    main()
