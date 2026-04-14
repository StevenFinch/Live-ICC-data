
from __future__ import annotations

import io
import pathlib
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

REPO = pathlib.Path(__file__).resolve().parents[1]

INDEX_SOURCES = {
    "sp100": {
        "url": "https://en.wikipedia.org/wiki/S%26P_100",
        "mode": "html",
    },
    "dow30": {
        "url": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        "mode": "html",
    },
    "ndx100": {
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "mode": "html",
    },
    "sp400": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "mode": "html",
    },
    "sp600": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        "mode": "html",
    },
    "rut1000": {
        "url": "https://en.wikipedia.org/wiki/Russell_1000_Index",
        "mode": "html",
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}


def today_et_str() -> str:
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.strftime("%Y_%m%d")


def ensure_month_dir() -> pathlib.Path:
    now = datetime.now(ZoneInfo("America/New_York"))
    month_dir = REPO / "data" / now.strftime("%Y%m")
    month_dir.mkdir(parents=True, exist_ok=True)
    return month_dir


def http_get(url: str, tries: int = 3, timeout: int = 30) -> str:
    last = None
    for i in range(1, tries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(i * 1.0)
    raise RuntimeError(f"Failed to download {url}") from last


def find_symbol_col(df: pd.DataFrame) -> str:
    lowers = {str(c).lower(): c for c in df.columns}
    for key in ["symbol", "ticker", "code"]:
        if key in lowers:
            return lowers[key]
    return df.columns[0]


def clean_symbols(vals) -> list[str]:
    s = (
        pd.Series(list(vals))
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.upper()
    )
    s = s[s.str.match(r"^[A-Z0-9\.\-]+$")]
    s = s[s.str.len() > 0]
    return sorted(s.unique().tolist())


def fetch_index_symbols(name: str) -> list[str]:
    spec = INDEX_SOURCES[name]
    if spec["mode"] == "csv":
        df = pd.read_csv(spec["url"], dtype=str)
        col = find_symbol_col(df)
        return clean_symbols(df[col].tolist())

    html = http_get(spec["url"])
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No HTML tables found for {name}")

    # Choose the table with the most rows and a plausible symbol column
    best = None
    best_n = -1
    for t in tables:
        if len(t) > best_n:
            best = t
            best_n = len(t)

    if best is None or best.empty:
        raise RuntimeError(f"No usable table found for {name}")

    col = find_symbol_col(best)
    symbols = clean_symbols(best[col].tolist())
    if len(symbols) < 20:
        raise RuntimeError(f"Too few symbols parsed for {name}: {len(symbols)}")
    return symbols


def run_one(index_name: str, month_dir: pathlib.Path) -> bool:
    out_path = month_dir / f"icc_live_{index_name}_{today_et_str()}.csv"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[run_extra_indices] skip existing {out_path.name}")
        return True

    try:
        symbols = fetch_index_symbols(index_name)
        print(f"[run_extra_indices] {index_name}: parsed {len(symbols)} symbols")
    except Exception as e:
        print(f"[run_extra_indices] failed to fetch symbols for {index_name}: {type(e).__name__}: {e}")
        return False

    cmd = [
        sys.executable,
        str(REPO / "backend" / "icc_market_live.py"),
        *symbols,
    ]

    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[run_extra_indices] icc_market_live failed for {index_name}: {type(e).__name__}: {e}")
        return False

    # icc_market_live saves custom universe as icc_live_custom_YYYY_MMDD.csv; rename it.
    month_glob = sorted(month_dir.glob(f"icc_live_custom_{today_et_str()}*.csv"))
    if not month_glob:
        print(f"[run_extra_indices] missing custom output for {index_name}")
        return False

    src = month_glob[-1]
    try:
        src.replace(out_path)
    except Exception:
        # fallback copy+remove
        out_path.write_bytes(src.read_bytes())
        src.unlink(missing_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[run_extra_indices] done {index_name}: {out_path.name}")
        return True

    print(f"[run_extra_indices] empty output for {index_name}")
    return False


def main() -> None:
    month_dir = ensure_month_dir()
    ok, bad = [], []
    for idx in INDEX_SOURCES:
        success = run_one(idx, month_dir)
        if success:
            ok.append(idx)
        else:
            bad.append(idx)
    print(f"[run_extra_indices] success={ok}")
    print(f"[run_extra_indices] failed={bad}")


if __name__ == "__main__":
    main()
