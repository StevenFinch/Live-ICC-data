from __future__ import annotations

import io
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from backend.icc_market_live import get_live_panel
except Exception:
    get_live_panel = None  # type: ignore


SCHEMA_VERSION = "country_adr_sources_v12_multi_source"

USER_AGENT = (
    os.getenv("SEC_USER_AGENT")
    or os.getenv("SEC_CONTACT_EMAIL")
    or "LiveICCDataLibrary/1.0 contact@example.com"
)
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
SEC_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"

CITI_BASE_URL = (
    "https://depositaryreceipts.citi.com/adr/guides/uig.aspx"
    "?active=A&company_name={query}&pageId=8&subPageId=159"
)
CITI_QUERIES = ["0-9"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

ADR_NAME_RE = re.compile(
    r"(american depositar|american depositar?y| depository | depositary |\badr\b|\bads\b|"
    r"global depositary|global depository|\bgdr\b)",
    re.I,
)
FOREIGN_LISTING_HINT_RE = re.compile(
    r"(ordinary shares|common shares|class a ordinary|class b ordinary|\bplc\b|\bn\.v\.\b|"
    r"\bs\.a\.\b|\bs\.a\.b\.\b|\bse\b|\bag\b|\bltd\b|\blimited\b)",
    re.I,
)
EXCLUDE_NAME_RE = re.compile(
    r"(etf|exchange traded fund|fund|warrant|rights?|units?|notes due|preferred|preference|"
    r"closed end|closed-end|bond|debenture|trust preferred|series [a-z] preferred)",
    re.I,
)
BAD_SYMBOL_RE = re.compile(r"[^A-Z0-9\.-]")

US_COUNTRY_NAMES = {
    "United States",
    "United States of America",
    "USA",
    "U.S.A.",
    "US",
}

COUNTRY_ALIASES = {
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "Britain": "United Kingdom",
    "Great Britain": "United Kingdom",
    "Korea": "South Korea",
    "Korea, Republic of": "South Korea",
    "Republic of Korea": "South Korea",
    "S. Korea": "South Korea",
    "S. Africa": "South Africa",
    "Russia": "Russian Federation",
    "UAE": "United Arab Emirates",
    "Czechia": "Czech Republic",
    "China, People's Republic of": "China",
    "People's Republic of China": "China",
    "Hong Kong, China": "Hong Kong",
    "Macao": "Macau",
}


def now_utc_iso() -> str:
    """Return current UTC timestamp as an ISO string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def http_get_text(url: str, *, sec: bool = False, timeout: int = 30, retries: int = 3) -> str:
    """Fetch text content with retries."""
    headers = {"User-Agent": USER_AGENT if sec else BROWSER_USER_AGENT}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def normalize_symbol(x: object) -> str:
    """Normalize a ticker for Yahoo-style lookup."""
    s = str(x or "").strip().upper()
    s = s.replace("/", "-").replace(".", "-").replace(" ", "")
    return s


def normalize_country(x: object) -> str | None:
    """Normalize country names and remove empty values."""
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "n/a", "--"}:
        return None
    s = re.sub(r"\s+", " ", s)
    return COUNTRY_ALIASES.get(s, s)


def is_us_country(country: object) -> bool:
    """Return whether a country label should be treated as the United States."""
    c = normalize_country(country)
    return bool(c and c in US_COUNTRY_NAMES)


def dedupe_rows(rows: Iterable[dict]) -> pd.DataFrame:
    """Deduplicate candidate rows while preserving useful source metadata."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=["ticker", "name", "exchange", "country_hint", "source"])

    for col in ["ticker", "name", "exchange", "country_hint", "source"]:
        if col not in df.columns:
            df[col] = None

    df["ticker"] = df["ticker"].map(normalize_symbol)
    df = df[df["ticker"].astype(str).ne("")]
    df = df[~df["ticker"].astype(str).str.contains(BAD_SYMBOL_RE, na=False)]
    df["country_hint"] = df["country_hint"].map(normalize_country)

    # Prefer rows with explicit country information, then rows from DR directories.
    df["_priority"] = 0
    df.loc[df["country_hint"].notna(), "_priority"] += 10
    df.loc[df["source"].astype(str).str.contains("citi|deutsche|bny|adr", case=False, na=False), "_priority"] += 5
    df = df.sort_values(["ticker", "_priority"], ascending=[True, False])

    agg = (
        df.groupby("ticker", as_index=False)
        .agg(
            name=("name", lambda x: next((v for v in x if isinstance(v, str) and v.strip()), None)),
            exchange=("exchange", lambda x: next((v for v in x if isinstance(v, str) and v.strip()), None)),
            country_hint=("country_hint", lambda x: next((v for v in x if isinstance(v, str) and v.strip()), None)),
            source=("source", lambda x: ",".join(sorted(set(str(v) for v in x if str(v).strip())))),
        )
    )
    return agg.reset_index(drop=True)


def read_pipe_directory(url: str) -> pd.DataFrame:
    """Read Nasdaq Trader pipe-delimited symbol directory."""
    text = http_get_text(url)
    lines = [ln for ln in text.splitlines() if "|" in ln and not ln.startswith("File Creation Time")]
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str).fillna("")


def fetch_nasdaq_adr_like_candidates() -> pd.DataFrame:
    """Build ADR-like and foreign-listing candidates from Nasdaq Trader symbol directories."""
    rows: list[dict] = []

    try:
        n = read_pipe_directory(NASDAQ_LISTED_URL)
        for _, r in n.iterrows():
            name = str(r.get("Security Name", ""))
            etf = str(r.get("ETF", "")).upper()
            test = str(r.get("Test Issue", "")).upper()
            if etf == "Y" or test == "Y" or EXCLUDE_NAME_RE.search(name):
                continue
            if not (ADR_NAME_RE.search(name) or FOREIGN_LISTING_HINT_RE.search(name)):
                continue
            rows.append(
                {
                    "ticker": r.get("Symbol"),
                    "name": name,
                    "exchange": "NASDAQ",
                    "country_hint": None,
                    "source": "nasdaq_trader_nasdaqlisted",
                }
            )
    except Exception as exc:
        print(f"[country_adr] Nasdaq listed source failed: {type(exc).__name__}: {exc}")

    try:
        o = read_pipe_directory(OTHER_LISTED_URL)
        exchange_map = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "BATS", "V": "IEX"}
        for _, r in o.iterrows():
            name = str(r.get("Security Name", ""))
            etf = str(r.get("ETF", "")).upper()
            test = str(r.get("Test Issue", "")).upper()
            if etf == "Y" or test == "Y" or EXCLUDE_NAME_RE.search(name):
                continue
            if not (ADR_NAME_RE.search(name) or FOREIGN_LISTING_HINT_RE.search(name)):
                continue
            rows.append(
                {
                    "ticker": r.get("ACT Symbol"),
                    "name": name,
                    "exchange": exchange_map.get(str(r.get("Exchange", "")).upper(), r.get("Exchange")),
                    "country_hint": None,
                    "source": "nasdaq_trader_otherlisted",
                }
            )
    except Exception as exc:
        print(f"[country_adr] Other-listed source failed: {type(exc).__name__}: {exc}")

    return dedupe_rows(rows)


def parse_citi_table_from_html(html: str) -> pd.DataFrame:
    """Parse Citi DR directory search results from static HTML tables."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        lower = {c.lower(): c for c in cols}
        if "ticker" in lower and "country" in lower:
            df = t.copy()
            df.columns = cols
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_citi_dr_candidates() -> pd.DataFrame:
    """Fetch active DR candidates from Citi's public DR directory pages."""
    rows: list[dict] = []
    for q in CITI_QUERIES:
        url = CITI_BASE_URL.format(query=q)
        try:
            html = http_get_text(url)
            df = parse_citi_table_from_html(html)
            if df.empty:
                continue
            cols = {str(c).strip().lower(): c for c in df.columns}
            ticker_col = cols.get("ticker")
            country_col = cols.get("country")
            issuer_col = cols.get("issuer") or cols.get("company") or cols.get("company name")
            exchange_col = cols.get("exchange")
            active_col = cols.get("active")
            if ticker_col is None or country_col is None:
                continue
            for _, r in df.iterrows():
                if active_col is not None and str(r.get(active_col, "")).strip().upper() not in {"A", "ACTIVE", ""}:
                    continue
                ticker = r.get(ticker_col)
                country = normalize_country(r.get(country_col))
                if not ticker or not country or is_us_country(country):
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "name": r.get(issuer_col) if issuer_col else None,
                        "exchange": r.get(exchange_col) if exchange_col else None,
                        "country_hint": country,
                        "source": "citi_dr_directory",
                    }
                )
        except Exception as exc:
            print(f"[country_adr] Citi DR query {q!r} failed: {type(exc).__name__}: {exc}")
        time.sleep(0.2)
    return dedupe_rows(rows)


def load_sec_ticker_map(cache_dir: Path) -> pd.DataFrame:
    """Load SEC ticker-CIK-exchange associations."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "sec_company_tickers_exchange.csv"
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, dtype=str)
            if {"ticker", "cik"}.issubset(df.columns):
                return df
        except Exception:
            pass

    try:
        text = http_get_text(SEC_TICKERS_EXCHANGE_URL, sec=True)
        data = pd.read_json(io.StringIO(text))
        fields = data.get("fields")
        records = data.get("data")
        if fields is not None and records is not None:
            df = pd.DataFrame(records, columns=list(fields))
        else:
            df = data
        df.columns = [str(c).lower() for c in df.columns]
        if "ticker" in df.columns and "cik" in df.columns:
            df["ticker"] = df["ticker"].map(normalize_symbol)
            df.to_csv(cache_path, index=False)
            return df
    except Exception as exc:
        print(f"[country_adr] SEC ticker map failed: {type(exc).__name__}: {exc}")

    return pd.DataFrame(columns=["ticker", "cik", "title", "exchange"])


def sec_country_for_cik(cik: object, cache_dir: Path) -> str | None:
    """Fetch issuer country from SEC submissions JSON when available."""
    if cik is None or pd.isna(cik):
        return None
    try:
        cik_int = int(float(cik))
    except Exception:
        return None
    cik10 = f"{cik_int:010d}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"CIK{cik10}.json"

    try:
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
        else:
            text = http_get_text(SEC_SUBMISSIONS_URL.format(cik10=cik10), sec=True, timeout=20, retries=2)
            cache_path.write_text(text, encoding="utf-8")
            time.sleep(0.12)
        import json

        obj = json.loads(text)
        addresses = obj.get("addresses") or {}
        business = addresses.get("business") or {}
        country = (
            business.get("stateOrCountryDescription")
            or obj.get("stateOfIncorporationDescription")
            or obj.get("stateOfIncorporation")
        )
        return normalize_country(country)
    except Exception:
        return None


def yfinance_profile(ticker: str) -> dict:
    """Fetch market cap, country, and descriptive fields from yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        market_cap = info.get("marketCap")
        if not market_cap:
            try:
                market_cap = getattr(t, "fast_info", {}).get("market_cap")
            except Exception:
                market_cap = None
        return {
            "ticker": ticker,
            "yf_country": normalize_country(info.get("country")),
            "market_cap": float(market_cap) if market_cap and float(market_cap) > 0 else np.nan,
            "yf_name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "quote_type": info.get("quoteType"),
        }
    except Exception:
        return {"ticker": ticker, "yf_country": None, "market_cap": np.nan}


def build_candidate_universe(cache_path: Path, *, max_age_days: int = 5) -> pd.DataFrame:
    """Build or load a multi-source ADR and foreign-listing candidate universe."""
    force = os.getenv("COUNTRY_ADR_FORCE_REBUILD", "0") == "1"
    if cache_path.exists() and not force:
        try:
            cached = pd.read_csv(cache_path)
            if not cached.empty and "schema_version" in cached.columns:
                version_ok = str(cached["schema_version"].iloc[0]) == SCHEMA_VERSION
                mtime = pd.Timestamp(cache_path.stat().st_mtime, unit="s")
                age = (pd.Timestamp.utcnow().tz_localize(None) - mtime.tz_localize(None)).days
                if version_ok and age <= max_age_days:
                    return cached
        except Exception:
            pass

    candidate_frames = []
    citi = fetch_citi_dr_candidates()
    if not citi.empty:
        candidate_frames.append(citi)
    nasdaq = fetch_nasdaq_adr_like_candidates()
    if not nasdaq.empty:
        candidate_frames.append(nasdaq)

    if not candidate_frames:
        return pd.DataFrame(columns=["ticker", "name", "exchange", "country", "market_cap", "source", "schema_version"])

    candidates = dedupe_rows(pd.concat(candidate_frames, ignore_index=True).to_dict(orient="records"))
    candidates = candidates[~candidates["ticker"].astype(str).str.endswith(("W", "R", "U"))].copy()

    # Prioritize candidates with explicit countries and ADR directory sources, then enrich with profiles.
    candidates["_priority"] = 0
    candidates.loc[candidates["country_hint"].notna(), "_priority"] += 10
    candidates.loc[candidates["source"].astype(str).str.contains("citi", case=False, na=False), "_priority"] += 5
    candidates = candidates.sort_values(["_priority", "ticker"], ascending=[False, True]).reset_index(drop=True)

    max_candidates = int(os.getenv("COUNTRY_ADR_MAX_CANDIDATES", "1500"))
    candidates = candidates.head(max_candidates).copy()

    sec_cache = cache_path.parent / "sec_submissions_cache"
    sec_map = load_sec_ticker_map(cache_path.parent)
    sec_map = sec_map[[c for c in ["ticker", "cik", "exchange", "title"] if c in sec_map.columns]].copy()
    if "ticker" in sec_map.columns:
        sec_map["ticker"] = sec_map["ticker"].map(normalize_symbol)

    rows: list[dict] = []
    for i, r in candidates.iterrows():
        ticker = str(r["ticker"])
        profile = yfinance_profile(ticker)
        country = normalize_country(r.get("country_hint")) or profile.get("yf_country")

        if not country and not sec_map.empty:
            m = sec_map.loc[sec_map["ticker"] == ticker]
            if not m.empty:
                country = sec_country_for_cik(m.iloc[0].get("cik"), sec_cache)

        country = normalize_country(country)
        if not country or is_us_country(country):
            continue

        market_cap = profile.get("market_cap")
        if market_cap is None or pd.isna(market_cap) or float(market_cap) <= 0:
            continue

        rows.append(
            {
                "ticker": ticker,
                "name": r.get("name") or profile.get("yf_name"),
                "exchange": r.get("exchange"),
                "country": country,
                "market_cap": float(market_cap),
                "sector": profile.get("sector"),
                "industry": profile.get("industry"),
                "quote_type": profile.get("quote_type"),
                "source": r.get("source"),
                "schema_version": SCHEMA_VERSION,
                "built_at_utc": now_utc_iso(),
            }
        )

        if (i + 1) % 25 == 0:
            time.sleep(0.5)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["country", "market_cap"], ascending=[True, False]).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def select_country_top_n(adr: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Select top-N securities by market cap within each country or region."""
    if adr.empty:
        return pd.DataFrame()
    selected = []
    for country, g in adr.groupby("country"):
        g2 = g.sort_values("market_cap", ascending=False).head(top_n).copy()
        g2["rank_in_country"] = range(1, len(g2) + 1)
        selected.append(g2)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def compute_country_panel(selected: pd.DataFrame, *, min_available: int = 3, full_threshold: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Country / Region Level ICC from selected ADR or foreign-listing constituents."""
    if selected.empty or get_live_panel is None:
        return (
            pd.DataFrame(
                columns=[
                    "country",
                    "icc",
                    "method",
                    "n_candidates",
                    "n_selected",
                    "n_icc_available",
                    "coverage_mktcap",
                    "status",
                ]
            ),
            pd.DataFrame(),
        )

    tickers = selected["ticker"].dropna().astype(str).map(normalize_symbol).drop_duplicates().tolist()
    live = get_live_panel(tickers)
    if live is None or live.empty:
        live = pd.DataFrame(columns=["ticker", "ICC", "mktcap", "name", "sector"])

    live = live.copy()
    live["ticker"] = live["ticker"].astype(str).map(normalize_symbol)
    if "ICC" not in live.columns:
        live["ICC"] = np.nan
    if "mktcap" not in live.columns:
        live["mktcap"] = np.nan

    selected = selected.copy()
    selected["ticker"] = selected["ticker"].astype(str).map(normalize_symbol)
    merged = selected.merge(
        live[[c for c in ["ticker", "ICC", "mktcap", "name", "sector"] if c in live.columns]],
        on="ticker",
        how="left",
        suffixes=("", "_icc"),
    )

    rows = []
    for country, g in merged.groupby("country"):
        total_mcap = float(g["market_cap"].sum()) if len(g) else 0.0
        valid = g[g["ICC"].notna() & g["market_cap"].notna() & (g["market_cap"] > 0)].copy()
        n_selected = int(len(g))
        n_available = int(len(valid))
        coverage = float(valid["market_cap"].sum() / total_mcap) if total_mcap > 0 and n_available > 0 else 0.0

        if n_available >= full_threshold:
            method = "ICC calculation"
            status = "icc_calculation"
            icc = float(np.average(valid["ICC"], weights=valid["market_cap"]))
        elif n_available >= min_available:
            method = "Partial ADR estimate"
            status = "partial_adr_estimate"
            icc = float(np.average(valid["ICC"], weights=valid["market_cap"]))
        else:
            method = "Unavailable"
            status = "unavailable"
            icc = None

        rows.append(
            {
                "country": country,
                "icc": icc,
                "method": method,
                "n_candidates": int((merged["country"] == country).sum()),
                "n_selected": n_selected,
                "n_icc_available": n_available,
                "coverage_mktcap": coverage,
                "status": status,
            }
        )

    panel = pd.DataFrame(rows)
    if not panel.empty:
        status_order = {"icc_calculation": 0, "partial_adr_estimate": 1, "unavailable": 2}
        panel["_status_order"] = panel["status"].map(status_order).fillna(9)
        panel = panel.sort_values(["_status_order", "country"]).drop(columns="_status_order").reset_index(drop=True)
    return panel, merged


def build_country_adr_icc_panel(cache_path: Path, min_available: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Country / Region Level ICC using top ADR or foreign-listing constituents by market cap."""
    adr = build_candidate_universe(cache_path)
    if adr.empty:
        return (
            pd.DataFrame(
                columns=[
                    "country",
                    "icc",
                    "method",
                    "n_candidates",
                    "n_selected",
                    "n_icc_available",
                    "coverage_mktcap",
                    "status",
                ]
            ),
            pd.DataFrame(),
        )
    selected = select_country_top_n(adr, top_n=10)
    return compute_country_panel(selected, min_available=min_available, full_threshold=8)


# Backward-compatible aliases for older build scripts.
def build_country_region_icc_panel(cache_path: Path, min_available: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alias for build_country_adr_icc_panel."""
    return build_country_adr_icc_panel(cache_path, min_available=min_available)


def build_country_icc_panel(cache_path: Path, min_available: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alias for build_country_adr_icc_panel."""
    return build_country_adr_icc_panel(cache_path, min_available=min_available)
