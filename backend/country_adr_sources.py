from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"

BUILTIN_ADR_CANDIDATES = {
    "China": ["BABA", "JD", "PDD", "BIDU", "NTES", "TME", "BEKE", "YUMC", "ZTO", "LI", "XPEV", "NIO"],
    "Japan": ["TM", "SONY", "MUFG", "SMFG", "MFG", "HMC", "TAK", "NMR", "IX"],
    "United Kingdom": ["SHEL", "HSBC", "BP", "GSK", "AZN", "UL", "BTI", "LYG", "BCS", "RELX"],
    "Netherlands": ["ASML", "ING", "PHG", "NXPI", "QGEN"],
    "Germany": ["SAP", "DB", "DTEGY", "ADDYY", "BAYRY", "VWAGY", "BMWYY", "SIEGY", "BASFY"],
    "France": ["TTE", "SNY", "ORAN", "BNPQY", "LVMUY", "LRLCY", "AXAHY"],
    "Italy": ["E", "RACE", "CNHI", "ENLAY", "ISNPY", "MONRY", "TI"],
    "Brazil": ["PBR", "VALE", "ITUB", "BBD", "ABEV", "NU", "XP", "ERJ", "SBS"],
    "India": ["INFY", "WIT", "HDB", "IBN", "RDY", "TTM", "VEDL", "MMYT"],
    "Taiwan": ["TSM", "UMC", "AUOTY", "ASX", "CHT"],
    "South Korea": ["KB", "SHG", "WF", "PKX", "KT", "SKM", "KEP"],
    "Switzerland": ["NVS", "UBS", "LOGI", "TEL", "RHHBY"],
    "Canada": ["SHOP", "RY", "TD", "BNS", "BMO", "CM", "TRI", "ENB", "CNQ", "CNI", "CP", "BAM"],
    "Mexico": ["AMX", "FMX", "CX", "KOF", "PAC", "ASR", "OMAB", "TV"],
    "Chile": ["SQM", "BCH", "BSAC", "CCU", "ENIC"],
    "Argentina": ["MELI", "YPF", "GGAL", "BBAR", "BMA", "TEO", "LOMA", "IRS"],
    "Spain": ["SAN", "BBVA", "TEF", "GRFS"],
    "Ireland": ["ACN", "AON", "MDT", "TT", "CRH", "JCI"],
    "Israel": ["TEVA", "CHKP", "NICE", "WIX", "FROG", "ICL"],
    "Australia": ["BHP", "RIO", "WDS"],
    "South Africa": ["NPSNY", "SSL", "GFI", "HMY", "SBSW", "AU"],
    "Luxembourg": ["SPOT", "GLOB", "MT"],
    "Jersey": ["APTV", "WPP", "FERG", "MANU"],
}


def normalize_symbol(x: Any) -> str:
    """Normalize symbols for Yahoo-compatible matching."""
    s = "" if x is None else str(x).strip().upper()
    s = re.sub(r"\s+", "", s).replace(".", "-")
    return re.sub(r"[^A-Z0-9\-]", "", s)


def fetch_nasdaq_trader_candidates() -> pd.DataFrame:
    """Fetch ADR-like and foreign-listing candidates from Nasdaq Trader symbol directories."""
    urls = [
        "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt",
    ]
    rows = []
    patterns = re.compile(r"ADR|ADS|DEPOSITARY|AMERICAN DEPOSITARY|ORDINARY SHARES| PLC| N\.V\.| S\.A\.| LTD| AG", re.I)
    headers = {"User-Agent": USER_AGENT}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="|", dtype=str)
            for _, rr in df.iterrows():
                symbol = normalize_symbol(rr.get("Symbol") or rr.get("ACT Symbol"))
                name = str(rr.get("Security Name") or rr.get("Company Name") or "")
                etf = str(rr.get("ETF", "")).upper()
                test = str(rr.get("Test Issue", "")).upper()
                if symbol and etf != "Y" and test != "Y" and patterns.search(name):
                    rows.append({"ticker": symbol, "name": name, "source": "nasdaq_trader", "country_hint": ""})
        except Exception:
            continue
    return pd.DataFrame(rows).drop_duplicates("ticker") if rows else pd.DataFrame(columns=["ticker", "name", "source", "country_hint"])


def build_candidate_universe() -> pd.DataFrame:
    """Build a broad candidate universe without requiring user-maintained files."""
    rows = []
    for country, symbols in BUILTIN_ADR_CANDIDATES.items():
        for sym in symbols:
            rows.append({"ticker": normalize_symbol(sym), "country_hint": country, "name": "", "source": "builtin_adr_candidate"})
    base = pd.DataFrame(rows)
    online = fetch_nasdaq_trader_candidates()
    if not online.empty:
        base = pd.concat([base, online], ignore_index=True, sort=False)
    base["ticker"] = base["ticker"].map(normalize_symbol)
    return base[base["ticker"] != ""].drop_duplicates("ticker").reset_index(drop=True)


def enrich_with_yfinance(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add country and market cap using yfinance."""
    rows = []
    for _, r in candidates.iterrows():
        ticker = r["ticker"]
        country = str(r.get("country_hint") or "")
        name = str(r.get("name") or "")
        market_cap = np.nan
        try:
            info = yf.Ticker(ticker).get_info()
            country = info.get("country") or country
            market_cap = info.get("marketCap") or np.nan
            name = info.get("shortName") or info.get("longName") or name
        except Exception:
            pass
        rows.append({"ticker": ticker, "country": country, "name": name, "mktcap": market_cap, "source": r.get("source", "")})
    out = pd.DataFrame(rows)
    out["mktcap"] = pd.to_numeric(out["mktcap"], errors="coerce")
    out = out[out["country"].notna() & (out["country"].astype(str).str.strip() != "")]
    return out.drop_duplicates("ticker")


def get_live_panel_for_tickers(tickers: list[str]) -> pd.DataFrame:
    """Compute firm-level ICC panel using the existing repo engine."""
    from backend.icc_market_live import get_live_panel
    return get_live_panel(tickers)


def build_country_region_icc_panel(
    asof_date: str,
    repo_root: Path | None = None,
    output_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and optionally archive Country / Region ADR Top-N ICC."""
    candidates = enrich_with_yfinance(build_candidate_universe())
    rows, members = [], []
    if candidates.empty:
        panel = pd.DataFrame(columns=["date", "country", "icc", "n_selected", "n_icc_available", "coverage_mktcap", "method", "status"])
        return panel, pd.DataFrame()

    for country, g in candidates.groupby("country", dropna=False):
        g = g.copy()
        g = g[g["mktcap"].notna() & np.isfinite(g["mktcap"]) & (g["mktcap"] > 0)]
        g = g.sort_values("mktcap", ascending=False).head(10)
        n_selected = int(len(g))
        icc, n_icc, coverage = np.nan, 0, 0.0
        method, status = "Unavailable", "unavailable"
        if n_selected > 0:
            try:
                live = get_live_panel_for_tickers(g["ticker"].astype(str).tolist())
                if not live.empty:
                    live = live.copy()
                    live["ticker"] = live["ticker"].astype(str).map(normalize_symbol)
                    live["ICC"] = pd.to_numeric(live["ICC"], errors="coerce")
                    merged = g.merge(live[["ticker", "ICC"]], on="ticker", how="left")
                    ok = merged["ICC"].notna() & np.isfinite(merged["ICC"])
                    n_icc = int(ok.sum())
                    if n_icc > 0:
                        denom = float(merged["mktcap"].sum())
                        coverage = float(merged.loc[ok, "mktcap"].sum() / denom) if denom > 0 else 0.0
                        icc = float(np.average(merged.loc[ok, "ICC"], weights=merged.loc[ok, "mktcap"]))
                        if n_icc >= 4:
                            method, status = "ADR Top-10 ICC calculation", "adr_top10_icc_calculation"
                        elif n_icc >= 2:
                            method, status = "Partial ADR estimate", "partial_adr_estimate"
                        else:
                            method, status = "Unavailable", "too_few_adr_icc"
                    for _, m in merged.iterrows():
                        members.append({"date": asof_date, "country": country, "ticker": m.get("ticker"), "name": m.get("name"), "mktcap": m.get("mktcap"), "icc": m.get("ICC"), "selected": True})
            except Exception as exc:
                status = f"unavailable: {type(exc).__name__}"
        rows.append({"date": asof_date, "country": country, "icc": icc, "n_selected": n_selected, "n_icc_available": n_icc, "coverage_mktcap": coverage, "method": method, "status": status})
    panel = pd.DataFrame(rows).sort_values("country").reset_index(drop=True)
    members_df = pd.DataFrame(members)
    if output_root is not None:
        yyyymm = asof_date[:7].replace("-", "")
        tag = asof_date.replace("-", "_")
        out_dir = Path(output_root) / yyyymm
        out_dir.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out_dir / f"country_adr_icc_{tag}.csv", index=False)
        if not members_df.empty:
            members_df.to_csv(out_dir / f"country_adr_constituents_{tag}.csv", index=False)
    return panel, members_df
