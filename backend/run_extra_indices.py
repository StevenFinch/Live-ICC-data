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

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
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


def _get_text(url: str, timeout: int = 30, retries: int = 3) -> str:
    headers = {"User-Agent": USER_AGENT}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_err = exc
            time.sleep(0.7 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def _read_html_tables(url: str) -> list[pd.DataFrame]:
    tables = pd.read_html(io.StringIO(_get_text(url)))
    if not tables:
        raise RuntimeError(f"No HTML tables parsed from {url}")
    return tables


def _normalize_symbols(series: pd.Series) -> list[str]:
    values = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )
    return [x for x in values if x and not BAD_SYM.search(x)]


def _find_table_by_columns(tables: list[pd.DataFrame], required_cols: list[str], min_rows: int) -> pd.DataFrame:
    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        lower = {c.lower(): c for c in cols}
        if all(c.lower() in lower for c in required_cols) and len(df) >= min_rows:
            df = df.copy()
            df.columns = cols
            return df
    raise RuntimeError(f"Could not find a table with columns {required_cols}")


def _fetch_sp100() -> list[str]:
    df = _find_table_by_columns(_read_html_tables(INDEX_URLS["sp100"]), ["Symbol", "Name", "Sector"], 90)
    return _normalize_symbols(df["Symbol"])


def _fetch_dow30() -> list[str]:
    df = _find_table_by_columns(_read_html_tables(INDEX_URLS["dow30"]), ["Company", "Exchange", "Symbol", "Sector"], 30)
    return _normalize_symbols(df["Symbol"])[:30]


def _fetch_ndx100() -> list[str]:
    df = _find_table_by_columns(_read_html_tables(INDEX_URLS["ndx100"]), ["Ticker", "Company"], 90)
    return _normalize_symbols(df["Ticker"])


def _fetch_sp400() -> list[str]:
    df = _find_table_by_columns(_read_html_tables(INDEX_URLS["sp400"]), ["Symbol", "Security"], 350)
    return _normalize_symbols(df["Symbol"])


def _fetch_sp600() -> list[str]:
    df = _find_table_by_columns(_read_html_tables(INDEX_URLS["sp600"]), ["Symbol", "Security"], 500)
    return _normalize_symbols(df["Symbol"])


def _fetch_rut1000() -> list[str]:
    df = _find_table_by_columns(_read_html_tables(INDEX_URLS["rut1000"]), ["Company", "Symbol"], 900)
    return _normalize_symbols(df["Symbol"])


FETCHERS: dict[str, Callable[[], list[str]]] = {
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


def _existing_nonempty_file(universe: str) -> pathlib.Path | None:
    for p in sorted(_month_dir().glob(f"icc_live_{universe}_{_date_tag()}*.csv")):
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def run_one(universe: str) -> tuple[bool, str]:
    existing = _existing_nonempty_file(universe)
    if existing is not None:
        return True, f"skip existing {existing.name}"

    tickers = FETCHERS[universe]()
    tickers = list(dict.fromkeys(tickers))
    panel = get_live_panel(tickers)
    if panel.empty:
        raise RuntimeError(f"Empty panel for {universe}")

    out_path = _month_dir() / f"icc_live_{universe}_{_date_tag()}.csv"
    panel.to_csv(out_path, index=False)
    return True, f"saved {out_path.name} with {len(panel)} rows"


def main() -> None:
    ok, bad = [], []
    for universe in RUN_ORDER:
        try:
            success, msg = run_one(universe)
            print(f"[run_extra_indices] {universe}: {msg}")
            if success:
                ok.append(universe)
            else:
                bad.append(universe)
        except Exception as exc:
            print(f"[run_extra_indices] FAILED on {universe}: {type(exc).__name__}: {exc}")
            bad.append(universe)

    print(f"[run_extra_indices] success = {ok}")
    print(f"[run_extra_indices] failed = {bad}")
    if len(ok) == 0:
        raise RuntimeError(f"All extra index runs failed: {bad}")


if __name__ == "__main__":
    main()
