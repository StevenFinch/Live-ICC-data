from __future__ import annotations

import io
import pathlib
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
    "Chrome/123.0.0.0 Safari/537.36"
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


def get_text(url: str, timeout: int = 30, retries: int = 3) -> str:
    last_err = None
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(0.6 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")



def read_tables(url: str) -> list[pd.DataFrame]:
    html = get_text(url)
    return pd.read_html(io.StringIO(html))



def normalize_symbols(series: pd.Series) -> list[str]:
    out = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )
    out = [x for x in out if x and x not in {"NAN", "NONE"}]
    seen = set()
    deduped = []
    for x in out:
        if x not in seen:
            deduped.append(x)
            seen.add(x)
    return deduped



def find_table(tables: list[pd.DataFrame], required_cols: list[str], min_rows: int) -> pd.DataFrame:
    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        cols_lower = {c.lower(): c for c in cols}
        if all(col.lower() in cols_lower for col in required_cols) and len(df) >= min_rows:
            df = df.copy()
            df.columns = cols
            return df
    raise RuntimeError(f"Could not find table with columns {required_cols}")



def fetch_sp100() -> list[str]:
    df = find_table(read_tables(INDEX_URLS["sp100"]), ["Symbol", "Name", "Sector"], 90)
    return normalize_symbols(df["Symbol"])



def fetch_dow30() -> list[str]:
    df = find_table(read_tables(INDEX_URLS["dow30"]), ["Company", "Exchange", "Symbol", "Sector"], 30)
    return normalize_symbols(df["Symbol"])[:30]



def fetch_ndx100() -> list[str]:
    df = find_table(read_tables(INDEX_URLS["ndx100"]), ["Ticker", "Company"], 90)
    return normalize_symbols(df["Ticker"])



def fetch_sp400() -> list[str]:
    df = find_table(read_tables(INDEX_URLS["sp400"]), ["Symbol", "Security"], 350)
    return normalize_symbols(df["Symbol"])



def fetch_sp600() -> list[str]:
    df = find_table(read_tables(INDEX_URLS["sp600"]), ["Symbol", "Security"], 500)
    return normalize_symbols(df["Symbol"])



def fetch_rut1000() -> list[str]:
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



def month_dir() -> pathlib.Path:
    d = _today_et()
    out = REPO / "data" / f"{d.year}{d.month:02d}"
    out.mkdir(parents=True, exist_ok=True)
    return out



def date_tag() -> str:
    d = _today_et()
    return f"{d.year}_{d.month:02d}{d.day:02d}"



def existing_nonempty(universe: str) -> pathlib.Path | None:
    out_dir = month_dir()
    pattern = f"icc_live_{universe}_{date_tag()}*.csv"
    for p in sorted(out_dir.glob(pattern)):
        try:
            if p.stat().st_size > 0:
                return p
        except FileNotFoundError:
            continue
    return None



def run_one(universe: str) -> tuple[bool, str]:
    existing = existing_nonempty(universe)
    if existing is not None:
        return True, f"skip existing {existing.name}"

    tickers = FETCHERS[universe]()
    panel = get_live_panel(tickers)
    if panel.empty:
        raise RuntimeError(f"Empty ICC panel for {universe}")

    out_path = month_dir() / f"icc_live_{universe}_{date_tag()}.csv"
    panel.to_csv(out_path, index=False)
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

    if len(ok) == 0:
        raise RuntimeError(f"All extra index runs failed: {bad}")


if __name__ == "__main__":
    main()
