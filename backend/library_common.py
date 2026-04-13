from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import exceptions as yfexc

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # pragma: no cover


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCS = ROOT / "docs"
CONFIG = ROOT / "config"
LIBRARY = DATA / "library"
CACHE = DATA / "cache"

for p in [DATA, DOCS, CONFIG, LIBRARY, CACHE]:
    p.mkdir(parents=True, exist_ok=True)

BAD_SYM = re.compile(r"[^A-Z0-9\.\-\^=]")
G_MIN, G_MAX = 0.01, 0.75
PAUSE_EVERY, PAUSE_SEC = 20, 0.6

FNAME_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)


def today_et() -> dt.date:
    if ZoneInfo is None:
        return dt.datetime.utcnow().date()
    return dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo("America/New_York")).date()


def ensure_dir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if int(mask.sum()) == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def parse_snapshot_key(path: pathlib.Path, universe: str):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    if m.group("universe") != universe:
        return None
    year = m.group("year")
    mmdd = m.group("mmdd")
    rerun = int(m.group("rerun") or 0)
    yyyymmdd = int(f"{year}{mmdd}")
    return (yyyymmdd, rerun, path.name)


def find_latest_snapshot(universe: str = "usall") -> pathlib.Path:
    candidates = []
    for p in DATA.glob("*.csv"):
        key = parse_snapshot_key(p, universe)
        if key is not None:
            candidates.append((key, p))
    for p in DATA.glob("*/*.csv"):
        key = parse_snapshot_key(p, universe)
        if key is not None:
            candidates.append((key, p))

    if not candidates:
        raise FileNotFoundError(f"No ICC snapshot found for universe={universe!r} under {DATA}")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_snapshot(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    for col in ["mktcap", "bm", "ICC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        df["date"] = today_et().isoformat()

    for col in ["ticker", "name", "sector"]:
        if col not in df.columns:
            df[col] = None

    required = {"ticker", "mktcap", "bm", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in snapshot {path.name}: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def safe_float(x):
    if x is None or pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x):
    if x is None or pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        return None


def df_records(df: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, (np.floating,)):
                clean[k] = None if pd.isna(v) else float(v)
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif pd.isna(v):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


def write_json(obj: dict, path: pathlib.Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def upsert_history(
    new_df: pd.DataFrame,
    history_path: pathlib.Path,
    key_cols: list[str],
    sort_cols: list[str],
) -> pd.DataFrame:
    if history_path.exists():
        hist = pd.read_csv(history_path)
        keep = pd.Series(True, index=hist.index)
        for _, row in new_df[key_cols].drop_duplicates().iterrows():
            mask = pd.Series(True, index=hist.index)
            for c in key_cols:
                mask &= hist[c].astype(str) == str(row[c])
            keep &= ~mask
        hist = hist.loc[keep].copy()
        hist = pd.concat([hist, new_df], ignore_index=True)
    else:
        hist = new_df.copy()

    for c in sort_cols:
        hist[c] = hist[c].astype(str)
    hist = hist.sort_values(sort_cols).reset_index(drop=True)
    return hist


# -----------------------------------------------------------------------------
# Live ICC helper for arbitrary symbols (used for ETF / country proxy holdings)
# -----------------------------------------------------------------------------

def _eps_path(fe1: float, g2: float, T: int = 15, g_long: float = 0.04) -> list[float]:
    out = [fe1, fe1 * (1 + g2)]
    fade = np.exp(np.log(g_long / g2) / T)
    for _ in range(3, T + 2):
        g2 *= fade
        out.append(out[-1] * (1 + g2))
    return out


def _pv(eps: list[float], b1: float, r: float, T: int = 15, g_long: float = 0.04) -> float:
    b_ss = np.clip(g_long / r, 0, 1)
    step = (b1 - b_ss) / T
    pv = 0.0
    for k in range(1, T + 1):
        b_k = np.clip(b1 - step * (k - 1), 0, 1)
        pv += eps[k - 1] * (1 - b_k) / (1 + r) ** k
    tv = eps[-1] / (r * (1 + r) ** T)
    return pv + tv


def solve_icc(price: float, fe1: float, g2: float, div: float) -> Optional[float]:
    if min(price, fe1) <= 0 or not np.isfinite(price):
        return None
    eps = _eps_path(fe1, g2)
    b1 = np.clip(1 - div / fe1, 0, 1)
    lo, hi = 0.01, 0.40
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pv = _pv(eps, b1, mid)
        lo, hi = (mid, hi) if pv > price else (lo, mid)
    return mid


def fetch_live_icc_row(sym: str) -> Optional[Dict[str, Any]]:
    sym = str(sym).upper().strip()
    if not sym or BAD_SYM.search(sym):
        return None

    try:
        tkr = yf.Ticker(sym)
        info = tkr.info or {}
        price = info.get("regularMarketPrice")
        if price is None or not np.isfinite(price):
            return None

        fe1, g2 = None, None
        try:
            trend = tkr.get_eps_trend(as_dict=True) or {}
            curr = trend.get("current", {}) or {}
            fe_now, fe_next = curr.get("0y"), curr.get("+1y")
            if fe_now and fe_next and fe_now > 0:
                fe1 = fe_now
                g2 = np.clip(fe_next / fe_now - 1, G_MIN, G_MAX)
        except Exception:
            pass

        if fe1 is None:
            fe1 = info.get("forwardEps")

        if g2 is None and info.get("earningsGrowth") is not None:
            g2 = np.clip(info["earningsGrowth"], G_MIN, G_MAX)

        if fe1 is None or fe1 <= 0:
            return None
        if g2 is None:
            g2 = 0.04

        shares = info.get("sharesOutstanding")
        bvps = info.get("bookValue")
        mktcap = info.get("marketCap")
        dividend = info.get("dividendRate") or 0.0

        bm = np.nan
        if shares and bvps and mktcap and shares > 0 and bvps > 0 and mktcap > 0:
            bm = (bvps * shares) / mktcap

        icc = solve_icc(price, fe1, g2, dividend)
        if icc is None:
            return None

        return {
            "ticker": sym,
            "price": price,
            "dividend": dividend,
            "mktcap": mktcap,
            "shares": shares,
            "bvps": bvps,
            "bm": bm,
            "FE1": fe1,
            "g2": g2,
            "ICC": icc,
            "sector": info.get("sector"),
            "name": info.get("shortName") or info.get("longName"),
            "country": info.get("country"),
            "industry": info.get("industry"),
            "date": today_et().isoformat(),
        }
    except yfexc.YFRateLimitError:
        time.sleep(1.5)
        return fetch_live_icc_row(sym)
    except Exception:
        return None


def build_live_lookup(snapshot_df: pd.DataFrame) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    keep_cols = [c for c in ["ticker", "ICC", "mktcap", "bm", "sector", "name", "date"] if c in snapshot_df.columns]
    for _, row in snapshot_df[keep_cols].iterrows():
        lookup[str(row["ticker"]).upper()] = row.to_dict()
    return lookup


def read_custom_etf_holdings(etf_ticker: str) -> Optional[pd.DataFrame]:
    p = CONFIG / "etf_holdings" / f"{etf_ticker.upper()}.csv"
    if not p.exists():
        return None

    df = pd.read_csv(p)
    cols = {c.lower().strip(): c for c in df.columns}
    if "ticker" not in cols or "weight" not in cols:
        raise ValueError(f"{p} must contain columns ticker, weight")

    out = pd.DataFrame(
        {
            "holding_ticker": df[cols["ticker"]].astype(str).str.upper().str.strip(),
            "weight": pd.to_numeric(df[cols["weight"]], errors="coerce"),
        }
    )
    out = out[out["holding_ticker"] != ""].dropna(subset=["weight"]).copy()
    if out.empty:
        return None

    # normalize to 1
    total = out["weight"].sum()
    if total > 0:
        out["weight"] = out["weight"] / total
    return out


def _normalize_weight_col(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.dropna().empty:
        return x
    # If percentages are in 0-100 scale, convert to 0-1.
    if float(x.dropna().max()) > 1.5:
        x = x / 100.0
    return x


def fetch_etf_holdings_yahoo(etf_ticker: str) -> Optional[pd.DataFrame]:
    """
    Documented yfinance fallback:
    Ticker.funds_data.top_holdings is used here.
    If you want full holdings, put config/etf_holdings/<ETF>.csv in the repo.
    """
    try:
        tkr = yf.Ticker(etf_ticker)
        funds = tkr.funds_data
        df = getattr(funds, "top_holdings", None)
        if df is None or len(df) == 0:
            return None

        tmp = df.reset_index().copy()
        tmp.columns = [str(c).strip() for c in tmp.columns]

        sym_col = None
        wt_col = None

        for c in tmp.columns:
            cl = c.lower()
            if sym_col is None and ("symbol" in cl or "ticker" in cl or cl == "index"):
                sym_col = c
            if wt_col is None and ("weight" in cl or "holding" in cl or "%" in cl or "asset" in cl):
                wt_col = c

        if sym_col is None:
            for c in tmp.columns:
                if tmp[c].astype(str).str.contains(r"[A-Za-z]").mean() > 0.8:
                    sym_col = c
                    break

        if wt_col is None:
            for c in tmp.columns:
                xc = pd.to_numeric(tmp[c], errors="coerce")
                if xc.notna().sum() >= max(3, len(tmp) // 2):
                    wt_col = c
                    break

        if sym_col is None or wt_col is None:
            return None

        out = pd.DataFrame(
            {
                "holding_ticker": tmp[sym_col].astype(str).str.upper().str.strip(),
                "weight": _normalize_weight_col(tmp[wt_col]),
            }
        )
        out = out[
            (out["holding_ticker"] != "")
            & out["weight"].notna()
            & np.isfinite(out["weight"])
            & (out["weight"] > 0)
        ].copy()

        if out.empty:
            return None

        total = out["weight"].sum()
        if total > 0:
            out["weight"] = out["weight"] / total
        return out
    except Exception:
        return None


def get_etf_holdings(etf_ticker: str) -> tuple[Optional[pd.DataFrame], str]:
    custom = read_custom_etf_holdings(etf_ticker)
    if custom is not None and len(custom) > 0:
        return custom, "custom_csv"

    yahoo = fetch_etf_holdings_yahoo(etf_ticker)
    if yahoo is not None and len(yahoo) > 0:
        return yahoo, "yfinance_top_holdings"

    return None, "unavailable"


def resolve_holdings_icc(
    holdings: pd.DataFrame,
    snapshot_lookup: dict[str, dict],
    fetched_cache: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for i, (_, r) in enumerate(holdings.iterrows(), 1):
        sym = str(r["holding_ticker"]).upper().strip()
        wt = float(r["weight"])
        rec = snapshot_lookup.get(sym)

        source = "snapshot"
        if rec is None:
            rec = fetched_cache.get(sym)
            source = "live_fetch_cache"

        if rec is None:
            rec = fetch_live_icc_row(sym)
            source = "live_fetch"
            if rec is not None:
                fetched_cache[sym] = rec

        rows.append(
            {
                "holding_ticker": sym,
                "weight": wt,
                "icc": np.nan if rec is None else rec.get("ICC"),
                "name": None if rec is None else rec.get("name"),
                "sector": None if rec is None else rec.get("sector"),
                "country": None if rec is None else rec.get("country"),
                "source": source if rec is not None else "missing",
            }
        )

        if i % PAUSE_EVERY == 0:
            time.sleep(PAUSE_SEC)

    out = pd.DataFrame(rows)
    return out
