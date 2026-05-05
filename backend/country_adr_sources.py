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

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"

try:
    from backend.icc_market_live import get_live_panel  # type: ignore
except Exception:  # pragma: no cover
    get_live_panel = None  # type: ignore

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# Built-in public ADR/foreign-listing universe. This is maintained in code so the user does not need to supply seeds.
BUILTIN_ADR_SEEDS: list[tuple[str, str, str]] = [
    ("BABA", "China", "Alibaba Group"), ("PDD", "China", "PDD Holdings"), ("BIDU", "China", "Baidu"), ("JD", "China", "JD.com"), ("NTES", "China", "NetEase"), ("TCOM", "China", "Trip.com"), ("ZTO", "China", "ZTO Express"), ("YUMC", "China", "Yum China"), ("TME", "China", "Tencent Music"), ("BEKE", "China", "KE Holdings"), ("LI", "China", "Li Auto"), ("NIO", "China", "NIO"), ("XPEV", "China", "XPeng"), ("FUTU", "China", "Futu"),
    ("TSM", "Taiwan", "Taiwan Semiconductor"), ("UMC", "Taiwan", "United Microelectronics"), ("ASX", "Taiwan", "ASE Technology"), ("HIMX", "Taiwan", "Himax"), ("AUOTY", "Taiwan", "AUO"),
    ("TM", "Japan", "Toyota"), ("SONY", "Japan", "Sony"), ("MUFG", "Japan", "Mitsubishi UFJ"), ("SMFG", "Japan", "Sumitomo Mitsui"), ("MFG", "Japan", "Mizuho"), ("HMC", "Japan", "Honda"), ("NMR", "Japan", "Nomura"), ("IX", "Japan", "ORIX"),
    ("INFY", "India", "Infosys"), ("WIT", "India", "Wipro"), ("HDB", "India", "HDFC Bank"), ("IBN", "India", "ICICI Bank"), ("RDY", "India", "Dr Reddy's"), ("WNS", "India", "WNS"), ("MMYT", "India", "MakeMyTrip"), ("SIFY", "India", "Sify"),
    ("PBR", "Brazil", "Petrobras"), ("VALE", "Brazil", "Vale"), ("ITUB", "Brazil", "Itau"), ("BBD", "Brazil", "Bradesco"), ("BSBR", "Brazil", "Banco Santander Brasil"), ("ABEV", "Brazil", "Ambev"), ("GGB", "Brazil", "Gerdau"), ("SID", "Brazil", "CSN"), ("SUZ", "Brazil", "Suzano"), ("VIV", "Brazil", "Telefonica Brasil"),
    ("ARM", "United Kingdom", "Arm Holdings"), ("AZN", "United Kingdom", "AstraZeneca"), ("HSBC", "United Kingdom", "HSBC"), ("BP", "United Kingdom", "BP"), ("SHEL", "United Kingdom", "Shell"), ("GSK", "United Kingdom", "GSK"), ("UL", "United Kingdom", "Unilever"), ("DEO", "United Kingdom", "Diageo"), ("BTI", "United Kingdom", "British American Tobacco"), ("BCS", "United Kingdom", "Barclays"), ("LYG", "United Kingdom", "Lloyds"), ("RIO", "United Kingdom", "Rio Tinto"), ("NGG", "United Kingdom", "National Grid"), ("RELX", "United Kingdom", "RELX"),
    ("ASML", "Netherlands", "ASML"), ("NXPI", "Netherlands", "NXP"), ("ING", "Netherlands", "ING"), ("PHG", "Netherlands", "Philips"), ("CNH", "Netherlands", "CNH Industrial"), ("QGEN", "Netherlands", "Qiagen"), ("AER", "Netherlands", "AerCap"), ("STLA", "Netherlands", "Stellantis"),
    ("SAP", "Germany", "SAP"), ("DB", "Germany", "Deutsche Bank"), ("DTEGY", "Germany", "Deutsche Telekom"), ("SIEGY", "Germany", "Siemens"), ("BASFY", "Germany", "BASF"), ("BMWYY", "Germany", "BMW"), ("VWAGY", "Germany", "Volkswagen"), ("BAYRY", "Germany", "Bayer"), ("DHLGY", "Germany", "DHL"), ("ALIZY", "Germany", "Allianz"),
    ("TTE", "France", "TotalEnergies"), ("SNY", "France", "Sanofi"), ("LVMUY", "France", "LVMH"), ("AXAHY", "France", "AXA"), ("DANOY", "France", "Danone"), ("EADSY", "France", "Airbus"), ("ORANY", "France", "Orange"), ("LRLCY", "France", "L'Oreal"), ("BNPQY", "France", "BNP Paribas"),
    ("NVS", "Switzerland", "Novartis"), ("RHHBY", "Switzerland", "Roche"), ("NSRGY", "Switzerland", "Nestle"), ("UBS", "Switzerland", "UBS"), ("ABB", "Switzerland", "ABB"), ("ZURVY", "Switzerland", "Zurich Insurance"),
    ("NVO", "Denmark", "Novo Nordisk"), ("DNNGY", "Denmark", "Orsted"), ("VWSYF", "Denmark", "Vestas"),
    ("ERIC", "Sweden", "Ericsson"), ("VOLVY", "Sweden", "Volvo"), ("ASAZY", "Sweden", "Assa Abloy"),
    ("NOK", "Finland", "Nokia"), ("EQNR", "Norway", "Equinor"),
    ("SAN", "Spain", "Santander"), ("BBVA", "Spain", "BBVA"), ("TEF", "Spain", "Telefonica"), ("IBDRY", "Spain", "Iberdrola"), ("GRFS", "Spain", "Grifols"),
    ("E", "Italy", "Eni"), ("RACE", "Italy", "Ferrari"), ("ENLAY", "Italy", "Enel"), ("ISNPY", "Italy", "Intesa Sanpaolo"),
    ("SHOP", "Canada", "Shopify"), ("RY", "Canada", "Royal Bank of Canada"), ("TD", "Canada", "Toronto-Dominion"), ("BNS", "Canada", "Scotiabank"), ("BMO", "Canada", "Bank of Montreal"), ("CM", "Canada", "CIBC"), ("ENB", "Canada", "Enbridge"), ("CNQ", "Canada", "Canadian Natural Resources"), ("TRI", "Canada", "Thomson Reuters"), ("CP", "Canada", "Canadian Pacific Kansas City"), ("CNI", "Canada", "Canadian National Railway"),
    ("TEVA", "Israel", "Teva"), ("CHKP", "Israel", "Check Point"), ("NICE", "Israel", "NICE"), ("WIX", "Israel", "Wix"), ("CYBR", "Israel", "CyberArk"), ("ICL", "Israel", "ICL"), ("ZIM", "Israel", "ZIM"),
    ("KB", "South Korea", "KB Financial"), ("SHG", "South Korea", "Shinhan"), ("SKM", "South Korea", "SK Telecom"), ("KT", "South Korea", "KT"), ("PKX", "South Korea", "POSCO"), ("LPL", "South Korea", "LG Display"),
    ("AMX", "Mexico", "America Movil"), ("FMX", "Mexico", "Fomento Economico Mexicano"), ("KOF", "Mexico", "Coca-Cola FEMSA"), ("PAC", "Mexico", "Grupo Aeroportuario del Pacifico"), ("OMAB", "Mexico", "OMA"), ("ASR", "Mexico", "ASUR"), ("VLRS", "Mexico", "Volaris"), ("CX", "Mexico", "Cemex"),
    ("YPF", "Argentina", "YPF"), ("GGAL", "Argentina", "Grupo Financiero Galicia"), ("BMA", "Argentina", "Banco Macro"), ("PAM", "Argentina", "Pampa Energia"), ("TEO", "Argentina", "Telecom Argentina"), ("LOMA", "Argentina", "Loma Negra"), ("DESP", "Argentina", "Despegar"),
    ("SQM", "Chile", "SQM"), ("BCH", "Chile", "Banco de Chile"), ("CCU", "Chile", "CCU"), ("ENIC", "Chile", "Enel Chile"), ("LTM", "Chile", "LATAM Airlines"),
    ("CIB", "Colombia", "Bancolombia"), ("EC", "Colombia", "Ecopetrol"), ("BAP", "Peru", "Credicorp"), ("BVN", "Peru", "Buenaventura"),
    ("BUD", "Belgium", "Anheuser-Busch InBev"), ("UCBJY", "Belgium", "UCB"),
    ("BHP", "Australia", "BHP"), ("WDS", "Australia", "Woodside"), ("CSLLY", "Australia", "CSL"), ("MQBKY", "Australia", "Macquarie"),
    ("NPSNY", "South Africa", "Naspers"), ("SBSW", "South Africa", "Sibanye Stillwater"), ("GFI", "South Africa", "Gold Fields"), ("SSL", "South Africa", "Sasol"),
    ("SE", "Singapore", "Sea Limited"), ("GRAB", "Singapore", "Grab"), ("DBSDY", "Singapore", "DBS Group"), ("UOVEY", "Singapore", "UOB"),
    ("MELI", "Uruguay", "MercadoLibre"), ("GLOB", "Luxembourg", "Globant"), ("SPOT", "Luxembourg", "Spotify"),
]


def _http_get(url: str, timeout: int = 30, retries: int = 2) -> str:
    """Fetch text with retries."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_error = exc
            time.sleep(0.6 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def _clean_symbol(value: Any) -> str | None:
    """Normalize ticker symbols."""
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().upper().replace(".", "-").replace(" ", "")
    if not s or s in {"NAN", "N/A", "--"}:
        return None
    if re.search(r"[^A-Z0-9\-]", s):
        return None
    return s


def _seed_candidates() -> pd.DataFrame:
    """Return built-in ADR/foreign-listing candidates."""
    return pd.DataFrame(BUILTIN_ADR_SEEDS, columns=["ticker", "country", "company"]).assign(source="builtin_major_adr_universe")


def _nasdaq_trader_candidates() -> pd.DataFrame:
    """Fetch ADR-like and foreign-listing candidates from Nasdaq Trader symbol directories."""
    urls = [
        "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt",
    ]
    rows: list[dict[str, Any]] = []
    patterns = re.compile(r"ADR|ADS|DEPOSITARY|AMERICAN DEPOSITARY|NEW YORK REGISTRY| PLC| N\.V\.| S\.A\.| LTD| LIMITED| AG$| SE$| OYJ| AB$", re.I)
    for url in urls:
        try:
            text = _http_get(url)
            df = pd.read_csv(io.StringIO(text), sep="|")
            symbol_col = "Symbol" if "Symbol" in df.columns else "ACT Symbol" if "ACT Symbol" in df.columns else None
            name_col = "Security Name" if "Security Name" in df.columns else "Security Name" if "Security Name" in df.columns else None
            if symbol_col is None or name_col is None:
                continue
            if "ETF" in df.columns:
                df = df[df["ETF"].astype(str).str.upper() != "Y"].copy()
            for _, r in df.iterrows():
                name = str(r.get(name_col, ""))
                if not patterns.search(name):
                    continue
                ticker = _clean_symbol(r.get(symbol_col))
                if ticker:
                    rows.append({"ticker": ticker, "country": None, "company": name, "source": "nasdaq_trader_symbol_directory"})
        except Exception as exc:
            print(f"[country_adr_sources] Nasdaq Trader source failed: {type(exc).__name__}: {exc}")
    return pd.DataFrame(rows)


def _citi_candidates() -> pd.DataFrame:
    """Best-effort scrape of Citi Global DR Directory search result pages."""
    rows: list[dict[str, Any]] = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        url = f"https://depositaryreceipts.citi.com/adr/guides/uig.aspx?active=A&company_name={letter}&pageId=8&subPageId=159"
        try:
            html = _http_get(url, retries=1)
            tables = pd.read_html(io.StringIO(html))
            for table in tables:
                lower = {str(c).strip().lower(): c for c in table.columns}
                ticker_col = next((lower[c] for c in lower if "ticker" in c), None)
                country_col = next((lower[c] for c in lower if "country" in c), None)
                company_col = next((lower[c] for c in lower if "issuer" in c or "company" in c), None)
                if ticker_col is None or country_col is None:
                    continue
                for _, r in table.iterrows():
                    ticker = _clean_symbol(r.get(ticker_col))
                    country = r.get(country_col)
                    if ticker and pd.notna(country):
                        rows.append({"ticker": ticker, "country": str(country).strip(), "company": str(r.get(company_col, "")), "source": "citi_dr_directory"})
        except Exception:
            continue
    return pd.DataFrame(rows)


def build_country_candidates() -> pd.DataFrame:
    """Build a multi-source ADR/foreign-listing candidate universe."""
    frames = [_seed_candidates()]
    for loader in [_nasdaq_trader_candidates, _citi_candidates]:
        try:
            df = loader()
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"[country_adr_sources] {loader.__name__} failed: {type(exc).__name__}: {exc}")
    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = out["ticker"].map(_clean_symbol)
    out = out[out["ticker"].notna()].copy()
    out["country"] = out["country"].replace({"nan": None, "None": None, "": None})
    out = out.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return out


def _fast_market_cap(ticker: str) -> float | None:
    """Fetch market cap using yfinance fast_info/info fallbacks."""
    try:
        t = yf.Ticker(ticker)
        try:
            cap = t.fast_info.get("market_cap")  # type: ignore[attr-defined]
            if cap and np.isfinite(float(cap)) and float(cap) > 0:
                return float(cap)
        except Exception:
            pass
        info = t.get_info()
        cap = info.get("marketCap")
        if cap and np.isfinite(float(cap)) and float(cap) > 0:
            return float(cap)
    except Exception:
        return None
    return None


def _yf_country(ticker: str) -> str | None:
    """Fetch company country from yfinance if available."""
    try:
        info = yf.Ticker(ticker).get_info()
        country = info.get("country")
        if country:
            return str(country).strip()
    except Exception:
        return None
    return None


def enrich_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Fill missing country and market cap for candidate securities."""
    rows: list[dict[str, Any]] = []
    for _, r in candidates.iterrows():
        ticker = str(r["ticker"])
        country = r.get("country")
        if not country or pd.isna(country):
            country = _yf_country(ticker)
        if not country:
            continue
        cap = _fast_market_cap(ticker)
        rows.append({
            "ticker": ticker,
            "country": str(country).strip(),
            "company": r.get("company", ""),
            "source": r.get("source", ""),
            "market_cap_hint": cap,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["ticker"], keep="first")
    return out


def _get_live_panel_safe(tickers: list[str]) -> pd.DataFrame:
    """Run the existing firm-level ICC engine in chunks."""
    if get_live_panel is None:
        return pd.DataFrame()
    frames = []
    chunk_size = 60
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            df = get_live_panel(chunk)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"[country_adr_sources] get_live_panel failed for chunk {i}: {type(exc).__name__}: {exc}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    return out


def build_country_region_icc_panel(asof_date: str) -> pd.DataFrame:
    """Build and archive Country / Region Level ICC using ADR top-N composites."""
    candidates = enrich_candidates(build_country_candidates())
    if candidates.empty:
        out = pd.DataFrame(columns=["date", "country", "icc", "n_candidates", "n_selected", "n_icc_available", "coverage_mktcap", "method", "status"])
    else:
        selected_tickers: list[str] = []
        selected_by_country: dict[str, pd.DataFrame] = {}
        for country, g in candidates.groupby("country"):
            g = g.copy()
            g["rank_cap"] = pd.to_numeric(g["market_cap_hint"], errors="coerce").fillna(0.0)
            g = g.sort_values("rank_cap", ascending=False).head(25)
            selected_by_country[str(country)] = g
            selected_tickers.extend(g["ticker"].astype(str).tolist())
        selected_tickers = sorted(set(selected_tickers))
        panel = _get_live_panel_safe(selected_tickers)
        if panel.empty:
            out = pd.DataFrame(columns=["date", "country", "icc", "n_candidates", "n_selected", "n_icc_available", "coverage_mktcap", "method", "status"])
        else:
            panel["ticker"] = panel["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
            rows = []
            constituent_rows = []
            for country, g in selected_by_country.items():
                merged = g.merge(panel, on="ticker", how="left", suffixes=("", "_icc"))
                merged["mktcap_final"] = pd.to_numeric(merged.get("mktcap"), errors="coerce").fillna(pd.to_numeric(merged["market_cap_hint"], errors="coerce"))
                ok = merged["ICC"].notna() & merged["mktcap_final"].notna() & (merged["mktcap_final"] > 0)
                avail = merged.loc[ok].copy().sort_values("mktcap_final", ascending=False).head(10)
                n_available = int(len(avail))
                if n_available >= 4:
                    weights = avail["mktcap_final"] / avail["mktcap_final"].sum()
                    icc = float(np.average(avail["ICC"], weights=weights))
                    method = "ADR Top-10 ICC calculation"
                    status = "ok_adr_top_n"
                    coverage = float(avail["mktcap_final"].sum() / merged["mktcap_final"].dropna().sum()) if merged["mktcap_final"].dropna().sum() > 0 else np.nan
                elif n_available >= 2:
                    weights = avail["mktcap_final"] / avail["mktcap_final"].sum()
                    icc = float(np.average(avail["ICC"], weights=weights))
                    method = "Partial ADR estimate"
                    status = "partial_adr_composite"
                    coverage = float(avail["mktcap_final"].sum() / merged["mktcap_final"].dropna().sum()) if merged["mktcap_final"].dropna().sum() > 0 else np.nan
                else:
                    icc = np.nan
                    method = "Unavailable"
                    status = "insufficient_adr_icc"
                    coverage = np.nan
                rows.append({
                    "date": asof_date,
                    "country": country,
                    "icc": icc,
                    "n_candidates": int(len(candidates[candidates["country"] == country])),
                    "n_selected": int(len(g)),
                    "n_icc_available": n_available,
                    "coverage_mktcap": coverage,
                    "method": method,
                    "status": status,
                })
                for _, rr in avail.iterrows():
                    constituent_rows.append({
                        "date": asof_date,
                        "country": country,
                        "ticker": rr.get("ticker"),
                        "company": rr.get("company"),
                        "icc": rr.get("ICC"),
                        "mktcap": rr.get("mktcap_final"),
                        "source": rr.get("source"),
                    })
            out = pd.DataFrame(rows).sort_values(["method", "country"]).reset_index(drop=True)
            cons = pd.DataFrame(constituent_rows)
            yyyymm = asof_date[:7].replace("-", "")
            tag = asof_date[:4] + "_" + asof_date[5:7] + asof_date[8:10]
            cons_dir = DATA_DIR / "derived" / "country_adr" / yyyymm
            cons_dir.mkdir(parents=True, exist_ok=True)
            cons.to_csv(cons_dir / f"country_adr_constituents_{tag}.csv", index=False)
    yyyymm = asof_date[:7].replace("-", "")
    tag = asof_date[:4] + "_" + asof_date[5:7] + asof_date[8:10]
    out_dir = DATA_DIR / "derived" / "country_adr" / yyyymm
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"country_adr_icc_{tag}.csv", index=False)
    return out
