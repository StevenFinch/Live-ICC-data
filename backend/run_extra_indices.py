
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from backend.icc_market_live import _index_tickers, _today_et, get_live_panel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EXTRA_SOURCES = {
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    "rut1000": "https://en.wikipedia.org/wiki/Russell_1000_Index",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_symbol_table(url: str) -> list[str]:
    html = fetch_html(url)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No tables found on {url}")

    best = None
    best_count = -1
    for t in tables:
        cols = {str(c).lower(): c for c in t.columns}
        sym_col = None
        for key in ["symbol", "ticker"]:
            if key in cols:
                sym_col = cols[key]
                break
        if sym_col is None:
            continue
        vals = (
            t[sym_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", "", regex=True)
            .str.replace(".", "-", regex=False)
            .str.upper()
        )
        vals = [x for x in vals if x]
        if len(vals) > best_count:
            best_count = len(vals)
            best = vals

    if best is None or len(best) < 50:
        raise RuntimeError(f"Could not parse a valid symbol table from {url}")

    return sorted(set(best))


def save_universe(universe_name: str, tickers: list[str]) -> None:
    panel = get_live_panel(tickers)
    d_et = _today_et()
    yyyymm = f"{d_et.year}{d_et.month:02d}"
    tag = f"{d_et.year}_{d_et.month:02d}{d_et.day:02d}"
    out_dir = DATA / yyyymm
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"icc_live_{universe_name}_{tag}.csv"
    panel.to_csv(out, index=False)
    print(f"[run_extra_indices] saved {out} rows={len(panel)}")


def main() -> None:
    sp500 = _index_tickers("sp500")
    sp100 = _index_tickers("sp100")
    dow30 = _index_tickers("dow30")
    ndx100 = _index_tickers("ndx100")

    sp400 = parse_symbol_table(EXTRA_SOURCES["sp400"])
    sp600 = parse_symbol_table(EXTRA_SOURCES["sp600"])
    rut1000 = parse_symbol_table(EXTRA_SOURCES["rut1000"])

    sp1500 = sorted(set(sp500) | set(sp400) | set(sp600))

    save_universe("sp100", sp100)
    save_universe("dow30", dow30)
    save_universe("ndx100", ndx100)
    save_universe("sp400", sp400)
    save_universe("sp600", sp600)
    save_universe("sp1500", sp1500)
    save_universe("rut1000", rut1000)


if __name__ == "__main__":
    main()
