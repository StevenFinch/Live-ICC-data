from __future__ import annotations

import io
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from backend.icc_market_live import get_live_panel
except Exception:
    get_live_panel = None  # type: ignore

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

CITI_GLOBAL_URLS = [
    "https://depositaryreceipts.citi.com/adr/guides/uig.aspx?active=A&pageId=8&subPageId=34",
    "https://depositaryreceipts.citi.com/adr/guides/uig.aspx?active=A&pageId=4&subpageid=34",
    "https://depositaryreceipts.citi.com/adr/guides/uig.aspx?pageId=4&subpageid=34",
]

CITI_UNSPONSORED_URLS = [
    "https://depositaryreceipts.citi.com/adr/guides/unspresource.aspx?pageId=5&subpageid=173",
]

DB_DR_DIRECTORY_URLS = [
    "https://adr.db.com/drwebrebrand/dr-universe/dr_universe_type_e.html",
]

ADR_TERMS = re.compile(
    r"(american depositary|american depository|\badr\b|\bads\b|depositary shares|depository shares|global depositary|\bgdr\b)",
    re.I,
)

FOREIGN_EQUITY_TERMS = re.compile(
    r"(ordinary shares|ord shares|common shares|\bplc\b|\bn\.v\.\b|\bs\.a\.\b|\bag\b|\bse\b|\bltd\b)",
    re.I,
)

EXCLUDE_TERMS = re.compile(
    r"(etf|fund|warrant|right|unit|notes due|preferred|preference|closed end|bond|etn|trust preferred|debt)",
    re.I,
)

BAD_SYM = re.compile(r"[^A-Z0-9\.\-]")

COUNTRY_ALIASES = {
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "Great Britain": "United Kingdom",
    "England": "United Kingdom",
    "PRC": "China",
    "People's Republic of China": "China",
    "Hong Kong SAR": "Hong Kong",
    "Russian Federation": "Russia",
    "Korea, Republic of": "South Korea",
    "Republic of Korea": "South Korea",
    "Taiwan, Province of China": "Taiwan",
    "Brasil": "Brazil",
}

SOURCE_PRIORITY = {
    "citi_global_dr_directory": 1,
    "citi_unsponsored_adr_center": 2,
    "deutsche_bank_dr_directory": 3,
    "nasdaq_trader_adr_like": 4,
    "nasdaq_trader_foreign_listing": 5,
    "yfinance_enrichment": 9,
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def request_text(url: str, timeout: int = 35, retries: int = 3) -> str:
    """Fetch text with retries and a browser-like user agent."""
    headers = {"User-Agent": USER_AGENT}
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_err = exc
            time.sleep(0.9 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def normalize_ticker(x: Any) -> str | None:
    """Normalize ticker symbols to Yahoo-compatible notation where possible."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip().upper()
    if not s or s in {"NAN", "--", "-", "NONE"}:
        return None
    s = s.replace(" ", "")
    s = s.replace("/", "-")
    s = s.replace(".", "-")
    s = re.sub(r"[^A-Z0-9\-]", "", s)
    if not s or BAD_SYM.search(s):
        return None
    if len(s) > 12:
        return None
    return s


def normalize_country(x: Any) -> str | None:
    """Normalize country names."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = re.sub(r"\s+", " ", str(x).strip())
    if not s or s.lower() in {"nan", "none", "--", "-"}:
        return None
    return COUNTRY_ALIASES.get(s, s)


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column by case-insensitive name."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def make_records(
    df: pd.DataFrame,
    ticker_candidates: list[str],
    country_candidates: list[str],
    name_candidates: list[str],
    exchange_candidates: list[str],
    source: str,
) -> pd.DataFrame:
    """Convert a source table into normalized ADR candidate rows."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "company", "country", "exchange", "source"])

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    tcol = first_existing_col(df, ticker_candidates)
    ccol = first_existing_col(df, country_candidates)
    ncol = first_existing_col(df, name_candidates)
    ecol = first_existing_col(df, exchange_candidates)

    if tcol is None:
        return pd.DataFrame(columns=["ticker", "company", "country", "exchange", "source"])

    out = pd.DataFrame()
    out["ticker"] = df[tcol].map(normalize_ticker)
    out["company"] = df[ncol].astype(str).str.strip() if ncol else ""
    out["country"] = df[ccol].map(normalize_country) if ccol else None
    out["exchange"] = df[ecol].astype(str).str.strip() if ecol else ""
    out["source"] = source
    out = out.dropna(subset=["ticker"]).drop_duplicates("ticker")
    return out[["ticker", "company", "country", "exchange", "source"]]


def read_pipe_directory(url: str) -> pd.DataFrame:
    """Read a Nasdaq Trader pipe-delimited symbol directory."""
    text = request_text(url)
    lines = [ln for ln in text.splitlines() if "|" in ln and not ln.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str).fillna("")


def fetch_nasdaq_trader_candidates() -> pd.DataFrame:
    """Fetch ADR-like and foreign-listing candidates from Nasdaq Trader symbol directories."""
    frames: list[pd.DataFrame] = []

    try:
        n = read_pipe_directory(NASDAQ_LISTED_URL)
        n = n.rename(columns={"Symbol": "ticker", "Security Name": "company", "ETF": "etf"})
        n["exchange"] = "NASDAQ"
        frames.append(n[["ticker", "company", "exchange", "etf"]])
    except Exception as exc:
        print(f"[country_adr] Nasdaq listed directory failed: {type(exc).__name__}: {exc}")

    try:
        o = read_pipe_directory(OTHER_LISTED_URL)
        o = o.rename(columns={"ACT Symbol": "ticker", "Security Name": "company", "Exchange": "exchange", "ETF": "etf"})
        frames.append(o[["ticker", "company", "exchange", "etf"]])
    except Exception as exc:
        print(f"[country_adr] Other listed directory failed: {type(exc).__name__}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["ticker", "company", "country", "exchange", "source"])

    raw = pd.concat(frames, ignore_index=True)
    raw["ticker"] = raw["ticker"].map(normalize_ticker)
    raw["company"] = raw["company"].astype(str)
    raw["etf"] = raw["etf"].astype(str).str.upper()
    raw = raw[(raw["ticker"].notna()) & (raw["etf"] != "Y")]
    raw = raw[~raw["company"].str.contains(EXCLUDE_TERMS, na=False)]

    adr_like = raw[raw["company"].str.contains(ADR_TERMS, na=False)].copy()
    adr_like["source"] = "nasdaq_trader_adr_like"

    foreign_like = raw[raw["company"].str.contains(FOREIGN_EQUITY_TERMS, na=False)].copy()
    foreign_like["source"] = "nasdaq_trader_foreign_listing"

    out = pd.concat([adr_like, foreign_like], ignore_index=True)
    out["country"] = None
    out = out.drop_duplicates("ticker")
    return out[["ticker", "company", "country", "exchange", "source"]]


def fetch_citi_unsponsored() -> pd.DataFrame:
    """Fetch Citi unsponsored ADR center tables."""
    frames: list[pd.DataFrame] = []
    for url in CITI_UNSPONSORED_URLS:
        try:
            html = request_text(url)
            tables = pd.read_html(io.StringIO(html))
            for tb in tables:
                rec = make_records(
                    tb,
                    ticker_candidates=["DR Ticker", "Ticker", "Symbol"],
                    country_candidates=["Country"],
                    name_candidates=["Company", "Issuer"],
                    exchange_candidates=["Exchange"],
                    source="citi_unsponsored_adr_center",
                )
                if not rec.empty and rec["country"].notna().any():
                    frames.append(rec)
        except Exception as exc:
            print(f"[country_adr] Citi unsponsored failed: {type(exc).__name__}: {exc}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("ticker") if frames else pd.DataFrame(columns=["ticker", "company", "country", "exchange", "source"])


def fetch_citi_global_directory() -> pd.DataFrame:
    """Fetch Citi global DR directory when static tables are available."""
    frames: list[pd.DataFrame] = []
    for url in CITI_GLOBAL_URLS:
        try:
            html = request_text(url)
            tables = pd.read_html(io.StringIO(html))
            for tb in tables:
                rec = make_records(
                    tb,
                    ticker_candidates=["DR Ticker", "Ticker", "Symbol", "DR Symbol"],
                    country_candidates=["Country"],
                    name_candidates=["Company", "Issuer", "Issuer Name"],
                    exchange_candidates=["Exchange"],
                    source="citi_global_dr_directory",
                )
                if not rec.empty and rec["country"].notna().any():
                    frames.append(rec)
        except Exception as exc:
            print(f"[country_adr] Citi global directory failed: {type(exc).__name__}: {exc}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("ticker") if frames else pd.DataFrame(columns=["ticker", "company", "country", "exchange", "source"])


def fetch_deutsche_bank_directory() -> pd.DataFrame:
    """Fetch Deutsche Bank DR directory when static tables are available."""
    frames: list[pd.DataFrame] = []
    for url in DB_DR_DIRECTORY_URLS:
        try:
            html = request_text(url)
            tables = pd.read_html(io.StringIO(html))
            for tb in tables:
                rec = make_records(
                    tb,
                    ticker_candidates=["DR Ticker", "Ticker", "Symbol"],
                    country_candidates=["Country"],
                    name_candidates=["Company", "Issuer"],
                    exchange_candidates=["Exchange"],
                    source="deutsche_bank_dr_directory",
                )
                if not rec.empty and rec["country"].notna().any():
                    frames.append(rec)
        except Exception as exc:
            print(f"[country_adr] Deutsche Bank directory failed: {type(exc).__name__}: {exc}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("ticker") if frames else pd.DataFrame(columns=["ticker", "company", "country", "exchange", "source"])


def merge_candidates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge ADR candidate sources while preserving best country attribution."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["ticker", "company", "country", "exchange", "source"])

    raw = pd.concat(frames, ignore_index=True)
    raw["ticker"] = raw["ticker"].map(normalize_ticker)
    raw["country"] = raw["country"].map(normalize_country)
    raw["company"] = raw["company"].fillna("").astype(str)
    raw["exchange"] = raw["exchange"].fillna("").astype(str)
    raw["source"] = raw["source"].fillna("unknown").astype(str)
    raw = raw.dropna(subset=["ticker"])
    raw = raw[~raw["ticker"].str.contains(r"\$|\^", regex=True, na=False)]
    raw["source_priority"] = raw["source"].map(SOURCE_PRIORITY).fillna(99)
    raw["has_country"] = raw["country"].notna().astype(int)
    raw = raw.sort_values(["ticker", "has_country", "source_priority"], ascending=[True, False, True])

    rows = []
    for ticker, g in raw.groupby("ticker", sort=False):
        best = g.iloc[0].to_dict()
        country_rows = g[g["country"].notna()]
        if not country_rows.empty:
            best["country"] = country_rows.iloc[0]["country"]
        best["sources_all"] = ";".join(sorted(set(g["source"].astype(str))))
        rows.append(best)

    out = pd.DataFrame(rows)
    return out[["ticker", "company", "country", "exchange", "source", "sources_all"]].drop_duplicates("ticker")


def fetch_yfinance_metadata(ticker: str) -> dict[str, Any]:
    """Fetch country, market cap, sector, and name from yfinance."""
    rec: dict[str, Any] = {"ticker": ticker}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        country = normalize_country(info.get("country") or info.get("headquartersCountry"))
        market_cap = info.get("marketCap")
        if not market_cap:
            try:
                market_cap = getattr(t.fast_info, "market_cap", None) or t.fast_info.get("market_cap")
            except Exception:
                market_cap = None
        rec.update({
            "yf_country": country,
            "market_cap": float(market_cap) if market_cap and float(market_cap) > 0 else np.nan,
            "sector": info.get("sector"),
            "yf_name": info.get("shortName") or info.get("longName"),
            "quote_type": info.get("quoteType"),
        })
    except Exception:
        rec.update({
            "yf_country": None,
            "market_cap": np.nan,
            "sector": None,
            "yf_name": None,
            "quote_type": None,
        })
    return rec


def enrich_candidates(candidates: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """Enrich candidates with yfinance metadata, using a persistent cache."""
    cache_meta = cache_path.with_name(cache_path.stem + "_yf_meta.csv")
    existing = pd.DataFrame()
    if cache_meta.exists():
        try:
            existing = pd.read_csv(cache_meta)
            existing["ticker"] = existing["ticker"].map(normalize_ticker)
        except Exception:
            existing = pd.DataFrame()

    existing_tickers = set(existing["ticker"].dropna().astype(str)) if not existing.empty else set()
    all_tickers = candidates["ticker"].dropna().astype(str).unique().tolist()

    max_enrich = _env_int("COUNTRY_ADR_MAX_ENRICH", 2000)
    missing = [t for t in all_tickers if t not in existing_tickers]
    missing = missing[:max_enrich]

    rows = []
    for i, ticker in enumerate(missing, start=1):
        rows.append(fetch_yfinance_metadata(ticker))
        if i % 40 == 0:
            print(f"[country_adr] yfinance enriched {i}/{len(missing)} new tickers")
            time.sleep(0.5)

    new_meta = pd.DataFrame(rows)
    combined = pd.concat([existing, new_meta], ignore_index=True) if not new_meta.empty else existing
    if not combined.empty:
        combined = combined.drop_duplicates("ticker", keep="last")
        cache_meta.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_meta, index=False)

    out = candidates.merge(combined, on="ticker", how="left") if not combined.empty else candidates.copy()
    out["country_final"] = out["country"].where(out["country"].notna(), out.get("yf_country"))
    out["country_final"] = out["country_final"].map(normalize_country)
    out["company_final"] = out["company"].where(out["company"].astype(str).str.len() > 0, out.get("yf_name"))
    out["market_cap"] = pd.to_numeric(out.get("market_cap"), errors="coerce")

    out = out[out["country_final"].notna()].copy()
    out = out[out["country_final"] != "United States"].copy()
    out = out[out["market_cap"].notna() & (out["market_cap"] > 0)].copy()
    out = out.drop_duplicates("ticker")
    return out


def load_cached_or_build_adr_universe(cache_path: Path, max_age_days: int | None = None) -> pd.DataFrame:
    """Build or load the multi-source ADR universe."""
    force = os.environ.get("COUNTRY_ADR_FORCE_REBUILD", "0") == "1"
    max_age_days = max_age_days if max_age_days is not None else _env_int("COUNTRY_ADR_CACHE_MAX_AGE_DAYS", 2)

    if cache_path.exists() and not force:
        try:
            mtime = pd.Timestamp(cache_path.stat().st_mtime, unit="s")
            age_days = (pd.Timestamp.utcnow().tz_localize(None) - mtime.tz_localize(None)).days
            cached = pd.read_csv(cache_path)
            if age_days <= max_age_days and not cached.empty:
                print(f"[country_adr] using cached ADR universe: {cache_path}")
                return cached
        except Exception:
            pass

    print("[country_adr] rebuilding ADR universe from online sources")
    frames = [
        fetch_citi_unsponsored(),
        fetch_citi_global_directory(),
        fetch_deutsche_bank_directory(),
        fetch_nasdaq_trader_candidates(),
    ]
    candidates = merge_candidates(frames)
    print(f"[country_adr] raw merged candidates = {len(candidates)}")

    enriched = enrich_candidates(candidates, cache_path)
    enriched = enriched.sort_values(["country_final", "market_cap"], ascending=[True, False]).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(cache_path, index=False)
    print(f"[country_adr] enriched ADR universe = {len(enriched)}")
    return enriched


def build_country_adr_icc_panel(cache_path: Path, min_available: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Country / Region Level ICC from ADR composites.

    The standard portfolio uses up to the 10 largest ADR-like U.S.-listed firms by market cap.
    A country / region is still computed when at least 4 constituents have valid ICC estimates.
    """
    min_available = min_available if min_available is not None else _env_int("COUNTRY_ADR_MIN_AVAILABLE", 4)
    min_available = max(1, min_available)

    adr = load_cached_or_build_adr_universe(cache_path)
    output_cols = [
        "country", "country_region", "icc", "vw_icc", "method", "n_candidates", "n_selected",
        "n_icc_available", "coverage_mktcap", "status", "constituents", "source_note",
    ]
    if adr.empty or get_live_panel is None:
        return pd.DataFrame(columns=output_cols), pd.DataFrame()

    selected_rows = []
    for country, g in adr.groupby("country_final"):
        g2 = g.sort_values("market_cap", ascending=False).head(10).copy()
        g2["rank_in_country"] = range(1, len(g2) + 1)
        selected_rows.append(g2)
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()

    if selected.empty:
        return pd.DataFrame(columns=output_cols), pd.DataFrame()

    tickers = selected["ticker"].dropna().astype(str).unique().tolist()
    print(f"[country_adr] computing firm-level ICC for {len(tickers)} selected ADR tickers")
    try:
        live = get_live_panel(tickers)
    except Exception as exc:
        print(f"[country_adr] get_live_panel failed: {type(exc).__name__}: {exc}")
        live = pd.DataFrame(columns=["ticker", "ICC", "mktcap", "name", "sector"])

    if live is None or live.empty:
        live = pd.DataFrame(columns=["ticker", "ICC", "mktcap", "name", "sector"])

    live = live.copy()
    live["ticker"] = live["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    selected["ticker"] = selected["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    merged = selected.merge(live[["ticker", "ICC", "mktcap", "name", "sector"]], on="ticker", how="left", suffixes=("", "_icc"))

    rows = []
    for country, g in merged.groupby("country_final"):
        total_selected_mcap = float(g["market_cap"].sum()) if len(g) else 0.0
        valid = g[g["ICC"].notna() & g["market_cap"].notna() & (g["market_cap"] > 0)].copy()
        n_candidates = int((adr["country_final"] == country).sum())
        n_selected = int(len(g))
        n_available = int(len(valid))
        coverage = float(valid["market_cap"].sum() / total_selected_mcap) if total_selected_mcap > 0 and n_available > 0 else 0.0

        if n_available >= 10:
            method = "ICC calculation"
            status = "icc_calculation"
        elif n_available >= min_available:
            method = "Partial ADR ICC calculation"
            status = "partial_adr_icc_calculation"
        else:
            method = "Unavailable"
            status = "unavailable"

        if n_available >= min_available:
            icc_value = float(np.average(valid["ICC"], weights=valid["market_cap"]))
            constituents = ", ".join(valid.sort_values("market_cap", ascending=False)["ticker"].astype(str).tolist())
        else:
            icc_value = None
            constituents = ", ".join(g.sort_values("market_cap", ascending=False)["ticker"].astype(str).head(10).tolist())

        rows.append({
            "country": country,
            "country_region": country,
            "icc": icc_value,
            "vw_icc": icc_value,
            "method": method,
            "n_candidates": n_candidates,
            "n_selected": n_selected,
            "n_icc_available": n_available,
            "coverage_mktcap": coverage,
            "status": status,
            "constituents": constituents,
            "source_note": "Multi-source ADR universe; top-10 by market cap; minimum 4 valid ICC constituents.",
        })

    panel = pd.DataFrame(rows)
    status_order = {"icc_calculation": 0, "partial_adr_icc_calculation": 1, "unavailable": 2}
    panel["_status_order"] = panel["status"].map(status_order).fillna(9)
    panel = panel.sort_values(["_status_order", "country_region"]).drop(columns=["_status_order"]).reset_index(drop=True)

    members = merged.copy()
    members["country"] = members["country_final"]
    members["country_region"] = members["country_final"]
    members = members.sort_values(["country_region", "rank_in_country"]).reset_index(drop=True)
    return panel[output_cols], members
