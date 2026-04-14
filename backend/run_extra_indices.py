
from __future__ import annotations

import io
import pathlib
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.icc_market_live import _index_tickers, get_live_panel  # type: ignore


HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
}


def _today_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def find_symbol_column(df: pd.DataFrame) -> str:
    lowers = {str(c).lower(): c for c in df.columns}
    for cand in ["symbol", "ticker", "company", "security"]:
        if cand in lowers:
            return lowers[cand]
    return df.columns[0]


def parse_table_symbols(url: str) -> list[str]:
    html = fetch_html(url)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No tables found for {url}")
    best = max(tables, key=len)
    col = find_symbol_column(best)
    symbols = (
        best[col]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )
    symbols = [s for s in symbols if s and s not in {"NAN", "NONE"}]
    if len(symbols) < 30:
        raise RuntimeError(f"Only parsed {len(symbols)} symbols from {url}")
    return symbols


def extra_index_universes() -> dict[str, list[str]]:
    extra = {
        "sp400": parse_table_symbols("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"),
        "sp600": parse_table_symbols("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"),
        "rut1000": parse_table_symbols("https://en.wikipedia.org/wiki/Russell_1000_Index"),
    }
    return extra


def save_panel(universe_name: str, symbols: list[str]) -> bool:
    now = _today_et()
    out_dir = REPO / "data" / now.strftime("%Y%m")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"icc_live_{universe_name}_{now.strftime('%Y_%m%d')}.csv"

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[run_extra_indices] skip existing {out_path.name}")
        return True

    print(f"[run_extra_indices] computing {universe_name} | n={len(symbols)}")
    panel = get_live_panel(symbols)
    if panel.empty:
        print(f"[run_extra_indices] empty panel for {universe_name}")
        return False

    panel.to_csv(out_path, index=False)
    print(f"[run_extra_indices] wrote {out_path}")
    return True


def main() -> None:
    builtins = {
        "sp100": _index_tickers("sp100"),
        "dow30": _index_tickers("dow30"),
        "ndx100": _index_tickers("ndx100"),
    }
    online_extra = extra_index_universes()

    ok, bad = [], []
    for universe, symbols in {**builtins, **online_extra}.items():
        try:
            success = save_panel(universe, symbols)
            if success:
                ok.append(universe)
            else:
                bad.append(universe)
        except Exception as e:
            print(f"[run_extra_indices] FAILED on {universe}: {type(e).__name__}: {e}")
            bad.append(universe)

    print(f"[run_extra_indices] success={ok}")
    print(f"[run_extra_indices] failed={bad}")


if __name__ == "__main__":
    main()
