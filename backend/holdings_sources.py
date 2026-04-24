
from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
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

BAD_SYM = re.compile(r"[^A-Z0-9\.\-]")


@dataclass
class ETFSpec:
    ticker: str
    label: str
    category: str = "Core"
    provider: str = "auto"
    product_id: str = ""
    slug: str = ""


DEFAULT_ETFS: list[ETFSpec] = [
    ETFSpec("SPY", "SPDR S&P 500 ETF Trust", "US equity", "spdr"),
    ETFSpec("IVV", "iShares Core S&P 500 ETF", "US equity", "ishares", "239726", "ishares-core-sp-500-etf"),
    ETFSpec("VOO", "Vanguard S&P 500 ETF", "US equity", "stockanalysis"),
    ETFSpec("VTI", "Vanguard Total Stock Market ETF", "US equity", "stockanalysis"),
    ETFSpec("QQQ", "Invesco QQQ Trust", "US equity", "stockanalysis"),
    ETFSpec("DIA", "SPDR Dow Jones Industrial Average ETF Trust", "US equity", "spdr"),
    ETFSpec("IWM", "iShares Russell 2000 ETF", "US equity", "ishares"),
    ETFSpec("IWB", "iShares Russell 1000 ETF", "US equity", "ishares"),
    ETFSpec("IWD", "iShares Russell 1000 Value ETF", "US value", "ishares"),
    ETFSpec("IWF", "iShares Russell 1000 Growth ETF", "US growth", "ishares"),
    ETFSpec("EFA", "iShares MSCI EAFE ETF", "International equity", "ishares", "239623", "ishares-msci-eafe-etf"),
    ETFSpec("EEM", "iShares MSCI Emerging Markets ETF", "Emerging markets", "ishares"),
    ETFSpec("EWJ", "iShares MSCI Japan ETF", "Country ETF", "ishares"),
    ETFSpec("EWG", "iShares MSCI Germany ETF", "Country ETF", "ishares", "239650", "ishares-msci-germany-etf"),
    ETFSpec("MCHI", "iShares MSCI China ETF", "Country ETF", "ishares"),
    ETFSpec("INDA", "iShares MSCI India ETF", "Country ETF", "ishares", "239659", "ishares-msci-india-etf"),
    ETFSpec("EWZ", "iShares MSCI Brazil ETF", "Country ETF", "ishares"),
]


def http_get(url: str, timeout: int = 30, retries: int = 3) -> bytes:
    """Download URL content with retries."""
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def normalize_symbol(x: Any) -> str | None:
    """Normalize a holding symbol to Yahoo-style US ticker format when possible."""
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s or s in {"-", "--", "NAN", "CASH", "USD", "US DOLLAR"}:
        return None
    s = re.sub(r"\s+", "", s)
    s = s.replace(".", "-")
    s = s.replace("/", "-")
    if BAD_SYM.search(s):
        return None
    return s


def clean_holdings(df: pd.DataFrame, ticker_col: str, weight_col: str, name_col: str | None = None) -> pd.DataFrame:
    """Return normalized holdings with symbol, weight, and optional name."""
    out = pd.DataFrame()
    out["symbol"] = df[ticker_col].map(normalize_symbol)
    out["weight"] = pd.to_numeric(
        df[weight_col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )
    if out["weight"].dropna().max() is not None and out["weight"].dropna().max() > 1.5:
        out["weight"] = out["weight"] / 100.0
    out["name"] = df[name_col].astype(str) if name_col and name_col in df.columns else None
    out = out[out["symbol"].notna() & out["weight"].notna() & (out["weight"] > 0)].copy()
    out = out.groupby("symbol", as_index=False).agg(weight=("weight", "sum"), name=("name", "first"))
    total = out["weight"].sum()
    if total > 0:
        out["weight"] = out["weight"] / total
    return out.sort_values("weight", ascending=False).reset_index(drop=True)


def find_col(cols: list[str], patterns: list[str]) -> str | None:
    """Find a column by case-insensitive pattern matching."""
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in cols:
            if rx.search(str(c)):
                return c
    return None


def fetch_stockanalysis_holdings(ticker: str) -> pd.DataFrame:
    """Fetch full ETF holdings from stockanalysis.com when available."""
    url = f"https://stockanalysis.com/etf/{ticker.lower()}/holdings/"
    html = http_get(url).decode("utf-8", errors="ignore")
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No holdings table found at {url}")
    candidates = []
    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        df.columns = cols
        sym_col = find_col(cols, [r"^symbol$", r"ticker"])
        weight_col = find_col(cols, [r"^weight$", r"%\s*weight", r"weight\s*\(%" ])
        name_col = find_col(cols, [r"^name$", r"company", r"holding"])
        if sym_col and weight_col and len(df) >= 10:
            candidates.append((len(df), df, sym_col, weight_col, name_col))
    if not candidates:
        raise RuntimeError(f"Could not identify a holdings table at {url}")
    _, df, sym_col, weight_col, name_col = max(candidates, key=lambda x: x[0])
    h = clean_holdings(df, sym_col, weight_col, name_col)
    if h.empty:
        raise RuntimeError(f"Parsed empty holdings from {url}")
    h["holding_source"] = "stockanalysis_full_holdings"
    return h


def fetch_spdr_holdings(ticker: str) -> pd.DataFrame:
    """Fetch SPDR full holdings XLSX using the public daily holdings workbook."""
    url = f"https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"
    content = http_get(url)
    xls = pd.ExcelFile(io.BytesIO(content))
    best: tuple[int, pd.DataFrame, str, str, str | None] | None = None
    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        for i in range(min(len(raw), 40)):
            row = [str(x).strip() for x in raw.iloc[i].tolist()]
            if any(x.lower() == "ticker" for x in row) and any("weight" in x.lower() for x in row):
                df = pd.read_excel(xls, sheet_name=sheet, header=i)
                cols = [str(c).strip() for c in df.columns]
                df.columns = cols
                sym_col = find_col(cols, [r"^ticker$", r"symbol"])
                weight_col = find_col(cols, [r"weight", r"%"])
                name_col = find_col(cols, [r"name", r"holding"])
                if sym_col and weight_col:
                    item = (len(df), df, sym_col, weight_col, name_col)
                    if best is None or item[0] > best[0]:
                        best = item
    if best is None:
        raise RuntimeError(f"Could not identify holdings table in SPDR workbook for {ticker}")
    _, df, sym_col, weight_col, name_col = best
    h = clean_holdings(df, sym_col, weight_col, name_col)
    if h.empty:
        raise RuntimeError(f"Parsed empty SPDR holdings for {ticker}")
    h["holding_source"] = "spdr_full_holdings"
    return h


def parse_ishares_csv(content: bytes) -> pd.DataFrame:
    """Parse iShares holdings CSV with variable metadata rows."""
    text = content.decode("utf-8-sig", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header_idx = None
    for i, line in enumerate(lines[:80]):
        lower = line.lower()
        if "ticker" in lower and ("weight" in lower or "market value" in lower):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find iShares holdings header row")
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text), on_bad_lines="skip")
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    sym_col = find_col(cols, [r"^ticker$", r"symbol"])
    weight_col = find_col(cols, [r"weight\s*\(%\)", r"weight", r"% of fund"])
    name_col = find_col(cols, [r"name", r"holding"])
    if not sym_col or not weight_col:
        raise RuntimeError("Could not identify iShares ticker/weight columns")
    h = clean_holdings(df, sym_col, weight_col, name_col)
    if h.empty:
        raise RuntimeError("Parsed empty iShares holdings")
    h["holding_source"] = "ishares_full_holdings"
    return h


def fetch_ishares_holdings(spec: ETFSpec) -> pd.DataFrame:
    """Fetch iShares full holdings CSV using the public ajax endpoint."""
    if not spec.product_id or not spec.slug:
        raise RuntimeError(f"Missing iShares product_id/slug for {spec.ticker}")
    url = (
        f"https://www.ishares.com/us/products/{spec.product_id}/{spec.slug}/"
        f"1467271812596.ajax?fileType=csv&fileName={spec.ticker}_holdings&dataType=fund"
    )
    return parse_ishares_csv(http_get(url))


def fetch_yfinance_top_holdings(ticker: str) -> pd.DataFrame:
    """Fetch yfinance top holdings as a partial-holdings fallback."""
    fund = yf.Ticker(ticker)
    fd = getattr(fund, "funds_data", None)
    if fd is None:
        raise RuntimeError("funds_data is unavailable")
    h = getattr(fd, "top_holdings", None)
    if h is None:
        raise RuntimeError("top_holdings is unavailable")
    df = h.copy() if isinstance(h, pd.DataFrame) else pd.DataFrame(h)
    if df.empty:
        raise RuntimeError("top_holdings is empty")
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    sym_col = find_col(cols, [r"symbol", r"ticker", r"holding"])
    weight_col = find_col(cols, [r"holdingPercent", r"holding_percent", r"weight", r"percent", r"pct"])
    if sym_col is None:
        sym_col = cols[0]
    if weight_col is None:
        numeric_cols = []
        for c in cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() >= max(1, len(df) // 2):
                numeric_cols.append(c)
        if numeric_cols:
            weight_col = numeric_cols[-1]
    if weight_col is None:
        raise RuntimeError("Could not identify yfinance top holdings weight column")
    h2 = clean_holdings(df, sym_col, weight_col, None)
    h2["holding_source"] = "yfinance_top_holdings"
    return h2


def load_etf_specs(config_path: Path | None = None) -> list[ETFSpec]:
    """Load ETF specs from config if available; otherwise use built-in defaults."""
    specs = DEFAULT_ETFS.copy()
    if config_path and config_path.exists():
        try:
            cfg = pd.read_csv(config_path, dtype=str).fillna("")
            loaded = []
            for _, r in cfg.iterrows():
                ticker = str(r.get("ticker", "")).strip().upper()
                if not ticker:
                    continue
                loaded.append(
                    ETFSpec(
                        ticker=ticker,
                        label=str(r.get("label", ticker)).strip() or ticker,
                        category=str(r.get("category", "ETF")).strip() or "ETF",
                        provider=str(r.get("provider", "auto")).strip() or "auto",
                        product_id=str(r.get("product_id", "")).strip(),
                        slug=str(r.get("slug", "")).strip(),
                    )
                )
            if loaded:
                specs = loaded
        except Exception:
            pass
    seen = set()
    out = []
    for s in specs:
        if s.ticker not in seen:
            seen.add(s.ticker)
            out.append(s)
    return out


def fetch_full_or_partial_holdings(spec: ETFSpec) -> pd.DataFrame:
    """Fetch ETF holdings using official provider sources first, then public full holdings, then top holdings."""
    errors = []
    providers = []
    if spec.provider.lower() in {"spdr", "auto"}:
        providers.append(("spdr", lambda: fetch_spdr_holdings(spec.ticker)))
    if spec.provider.lower() in {"ishares", "auto"}:
        providers.append(("ishares", lambda: fetch_ishares_holdings(spec)))
    providers.append(("stockanalysis", lambda: fetch_stockanalysis_holdings(spec.ticker)))
    providers.append(("yfinance", lambda: fetch_yfinance_top_holdings(spec.ticker)))
    for source, fn in providers:
        try:
            h = fn()
            h["holding_source"] = h.get("holding_source", source)
            return h
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def compute_icc_from_holdings(
    holdings: pd.DataFrame,
    base_panel: pd.DataFrame,
    max_extra_symbols: int = 250,
) -> tuple[float | None, float, int, int, pd.DataFrame]:
    """Compute holdings-weighted ICC using base panel plus extra live ICC fetches for unmatched symbols."""
    if holdings.empty:
        return None, 0.0, 0, 0, pd.DataFrame()
    h = holdings.copy()
    h["symbol"] = h["symbol"].map(normalize_symbol)
    h = h[h["symbol"].notna() & h["weight"].notna()].copy()
    h["weight"] = pd.to_numeric(h["weight"], errors="coerce")
    h = h[h["weight"] > 0].copy()
    h["weight"] = h["weight"] / h["weight"].sum()

    base = base_panel.copy()
    base["ticker"] = base["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    base = base[["ticker", "ICC", "mktcap", "name", "sector"]].dropna(subset=["ICC"]).drop_duplicates("ticker")

    merged = h.merge(base, left_on="symbol", right_on="ticker", how="left")
    current_cov = float(merged.loc[merged["ICC"].notna(), "weight"].sum()) if len(merged) else 0.0

    if current_cov < 0.80 and get_live_panel is not None:
        missing = merged.loc[merged["ICC"].isna(), ["symbol", "weight"]].sort_values("weight", ascending=False)
        extra_symbols = missing["symbol"].head(max_extra_symbols).tolist()
        if extra_symbols:
            extra_panel = get_live_panel(extra_symbols)
            if extra_panel is not None and not extra_panel.empty:
                extra = extra_panel[["ticker", "ICC", "mktcap", "name", "sector"]].copy()
                extra["ticker"] = extra["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
                combined = pd.concat([base, extra], ignore_index=True).drop_duplicates("ticker", keep="last")
                merged = h.merge(combined, left_on="symbol", right_on="ticker", how="left")

    matched = merged["ICC"].notna()
    coverage = float(merged.loc[matched, "weight"].sum()) if len(merged) else 0.0
    n_total = int(len(h))
    n_matched = int(matched.sum())
    if n_matched == 0 or coverage <= 0:
        return None, 0.0, n_total, 0, merged
    icc = float(np.average(merged.loc[matched, "ICC"], weights=merged.loc[matched, "weight"]))
    return icc, coverage, n_total, n_matched, merged


def build_etf_icc_panel(base_panel: pd.DataFrame, config_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ETF ICC panel from online holdings without requiring manual holdings files."""
    rows = []
    members = []
    for spec in load_etf_specs(config_path):
        try:
            holdings = fetch_full_or_partial_holdings(spec)
            icc, coverage, n_total, n_matched, merged = compute_icc_from_holdings(holdings, base_panel)
            source = str(holdings["holding_source"].iloc[0]) if "holding_source" in holdings.columns and len(holdings) else "unknown"
            if icc is None:
                method = "Unavailable"
                status = "unavailable"
            elif source == "yfinance_top_holdings" or coverage < 0.80:
                method = "Partial holdings estimate"
                status = "partial_holdings"
            else:
                method = "ICC calculation"
                status = "icc_calculation"
            rows.append(
                {
                    "ticker": spec.ticker,
                    "label": spec.label,
                    "category": spec.category,
                    "icc": icc,
                    "coverage_weight": coverage,
                    "n_holdings_total": n_total,
                    "n_holdings_matched": n_matched,
                    "method": method,
                    "holding_source": source,
                    "status": status,
                }
            )
            if not merged.empty:
                tmp = merged.copy()
                tmp["etf"] = spec.ticker
                tmp["etf_label"] = spec.label
                members.append(tmp)
        except Exception as exc:
            rows.append(
                {
                    "ticker": spec.ticker,
                    "label": spec.label,
                    "category": spec.category,
                    "icc": None,
                    "coverage_weight": 0.0,
                    "n_holdings_total": 0,
                    "n_holdings_matched": 0,
                    "method": "Unavailable",
                    "holding_source": "",
                    "status": f"error: {type(exc).__name__}",
                }
            )
    panel = pd.DataFrame(rows)
    members_df = pd.concat(members, ignore_index=True) if members else pd.DataFrame()
    return panel, members_df
