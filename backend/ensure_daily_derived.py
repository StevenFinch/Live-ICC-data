from __future__ import annotations

import io
import re
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DERIVED_ETF_DIR = DATA_DIR / "derived" / "etf"
DERIVED_COUNTRY_DIR = DATA_DIR / "derived" / "country_adr"
CONFIG_DIR = REPO / "config"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

SNAP_RE = re.compile(r"^icc_live_usall_(\d{4})_(\d{4})(?:_r\d+)?\.csv$")

DEFAULT_ETFS = [
    ("SPY", "SPDR S&P 500 ETF Trust", "Broad market"),
    ("IVV", "iShares Core S&P 500 ETF", "Broad market"),
    ("VOO", "Vanguard S&P 500 ETF", "Broad market"),
    ("VTI", "Vanguard Total Stock Market ETF", "Broad market"),
    ("QQQ", "Invesco QQQ Trust", "Large growth"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF", "Large cap"),
    ("IWM", "iShares Russell 2000 ETF", "Small cap"),
    ("IJR", "iShares Core S&P Small-Cap ETF", "Small cap"),
    ("MDY", "SPDR S&P MidCap 400 ETF", "Mid cap"),
    ("VUG", "Vanguard Growth ETF", "Style"),
    ("VTV", "Vanguard Value ETF", "Style"),
    ("IWF", "iShares Russell 1000 Growth ETF", "Style"),
    ("IWD", "iShares Russell 1000 Value ETF", "Style"),
    ("XLK", "Technology Select Sector SPDR Fund", "Sector"),
    ("XLF", "Financial Select Sector SPDR Fund", "Sector"),
    ("XLV", "Health Care Select Sector SPDR Fund", "Sector"),
    ("XLY", "Consumer Discretionary Select Sector SPDR Fund", "Sector"),
    ("XLP", "Consumer Staples Select Sector SPDR Fund", "Sector"),
    ("XLE", "Energy Select Sector SPDR Fund", "Sector"),
    ("XLI", "Industrial Select Sector SPDR Fund", "Sector"),
    ("XLB", "Materials Select Sector SPDR Fund", "Sector"),
    ("XLU", "Utilities Select Sector SPDR Fund", "Sector"),
    ("XLRE", "Real Estate Select Sector SPDR Fund", "Sector"),
    ("XLC", "Communication Services Select Sector SPDR Fund", "Sector"),
    ("SMH", "VanEck Semiconductor ETF", "Theme"),
    ("XBI", "SPDR S&P Biotech ETF", "Theme"),
]

ADR_SEEDS = [
    ("BABA", "China", "Alibaba Group"), ("PDD", "China", "PDD Holdings"), ("BIDU", "China", "Baidu"),
    ("JD", "China", "JD.com"), ("NTES", "China", "NetEase"), ("TCOM", "China", "Trip.com"),
    ("ZTO", "China", "ZTO Express"), ("YUMC", "China", "Yum China"), ("TME", "China", "Tencent Music"),
    ("BEKE", "China", "KE Holdings"), ("LI", "China", "Li Auto"), ("NIO", "China", "NIO"),
    ("XPEV", "China", "XPeng"), ("FUTU", "China", "Futu"), ("MNSO", "China", "MINISO"),
    ("TSM", "Taiwan", "Taiwan Semiconductor"), ("UMC", "Taiwan", "United Microelectronics"),
    ("ASX", "Taiwan", "ASE Technology"), ("HIMX", "Taiwan", "Himax"),
    ("TM", "Japan", "Toyota"), ("SONY", "Japan", "Sony"), ("MUFG", "Japan", "Mitsubishi UFJ"),
    ("SMFG", "Japan", "Sumitomo Mitsui"), ("MFG", "Japan", "Mizuho"), ("HMC", "Japan", "Honda"),
    ("NMR", "Japan", "Nomura"), ("IX", "Japan", "ORIX"), ("TAK", "Japan", "Takeda"),
    ("INFY", "India", "Infosys"), ("WIT", "India", "Wipro"), ("HDB", "India", "HDFC Bank"),
    ("IBN", "India", "ICICI Bank"), ("RDY", "India", "Dr Reddy's"), ("WNS", "India", "WNS"),
    ("MMYT", "India", "MakeMyTrip"), ("SIFY", "India", "Sify"),
    ("PBR", "Brazil", "Petrobras"), ("VALE", "Brazil", "Vale"), ("ITUB", "Brazil", "Itau"),
    ("BBD", "Brazil", "Bradesco"), ("BSBR", "Brazil", "Santander Brasil"), ("ABEV", "Brazil", "Ambev"),
    ("GGB", "Brazil", "Gerdau"), ("SID", "Brazil", "CSN"), ("SUZ", "Brazil", "Suzano"),
    ("VIV", "Brazil", "Telefonica Brasil"),
    ("ARM", "United Kingdom", "Arm Holdings"), ("AZN", "United Kingdom", "AstraZeneca"),
    ("HSBC", "United Kingdom", "HSBC"), ("BP", "United Kingdom", "BP"), ("SHEL", "United Kingdom", "Shell"),
    ("GSK", "United Kingdom", "GSK"), ("UL", "United Kingdom", "Unilever"), ("DEO", "United Kingdom", "Diageo"),
    ("BTI", "United Kingdom", "British American Tobacco"), ("BCS", "United Kingdom", "Barclays"),
    ("LYG", "United Kingdom", "Lloyds"), ("RIO", "United Kingdom", "Rio Tinto"), ("NGG", "United Kingdom", "National Grid"),
    ("RELX", "United Kingdom", "RELX"),
    ("ASML", "Netherlands", "ASML"), ("NXPI", "Netherlands", "NXP"), ("ING", "Netherlands", "ING"),
    ("PHG", "Netherlands", "Philips"), ("CNH", "Netherlands", "CNH Industrial"), ("QGEN", "Netherlands", "Qiagen"),
    ("AER", "Netherlands", "AerCap"), ("STLA", "Netherlands", "Stellantis"),
    ("SAP", "Germany", "SAP"), ("DB", "Germany", "Deutsche Bank"), ("DTEGY", "Germany", "Deutsche Telekom"),
    ("SIEGY", "Germany", "Siemens"), ("BASFY", "Germany", "BASF"), ("BMWYY", "Germany", "BMW"),
    ("VWAGY", "Germany", "Volkswagen"), ("BAYRY", "Germany", "Bayer"), ("DHLGY", "Germany", "DHL"),
    ("ALIZY", "Germany", "Allianz"),
    ("TTE", "France", "TotalEnergies"), ("SNY", "France", "Sanofi"), ("LVMUY", "France", "LVMH"),
    ("AXAHY", "France", "AXA"), ("DANOY", "France", "Danone"), ("EADSY", "France", "Airbus"),
    ("ORANY", "France", "Orange"), ("LRLCY", "France", "L'Oreal"), ("BNPQY", "France", "BNP Paribas"),
    ("NVS", "Switzerland", "Novartis"), ("RHHBY", "Switzerland", "Roche"), ("NSRGY", "Switzerland", "Nestle"),
    ("UBS", "Switzerland", "UBS"), ("ABB", "Switzerland", "ABB"), ("ZURVY", "Switzerland", "Zurich Insurance"),
    ("NVO", "Denmark", "Novo Nordisk"), ("DNNGY", "Denmark", "Orsted"), ("VWSYF", "Denmark", "Vestas"),
    ("ERIC", "Sweden", "Ericsson"), ("VOLVY", "Sweden", "Volvo"), ("ASAZY", "Sweden", "Assa Abloy"),
    ("NOK", "Finland", "Nokia"), ("EQNR", "Norway", "Equinor"),
    ("SAN", "Spain", "Santander"), ("BBVA", "Spain", "BBVA"), ("TEF", "Spain", "Telefonica"),
    ("IBDRY", "Spain", "Iberdrola"), ("GRFS", "Spain", "Grifols"),
    ("E", "Italy", "Eni"), ("RACE", "Italy", "Ferrari"), ("ENLAY", "Italy", "Enel"),
    ("ISNPY", "Italy", "Intesa Sanpaolo"),
    ("SHOP", "Canada", "Shopify"), ("RY", "Canada", "Royal Bank of Canada"), ("TD", "Canada", "Toronto-Dominion"),
    ("BNS", "Canada", "Scotiabank"), ("BMO", "Canada", "Bank of Montreal"), ("CM", "Canada", "CIBC"),
    ("ENB", "Canada", "Enbridge"), ("CNQ", "Canada", "Canadian Natural Resources"), ("TRI", "Canada", "Thomson Reuters"),
    ("CP", "Canada", "Canadian Pacific Kansas City"), ("CNI", "Canada", "Canadian National Railway"),
    ("TEVA", "Israel", "Teva"), ("CHKP", "Israel", "Check Point"), ("NICE", "Israel", "NICE"),
    ("WIX", "Israel", "Wix"), ("CYBR", "Israel", "CyberArk"), ("ICL", "Israel", "ICL"), ("ZIM", "Israel", "ZIM"),
    ("KB", "South Korea", "KB Financial"), ("SHG", "South Korea", "Shinhan"), ("SKM", "South Korea", "SK Telecom"),
    ("KT", "South Korea", "KT"), ("PKX", "South Korea", "POSCO"), ("LPL", "South Korea", "LG Display"),
    ("AMX", "Mexico", "America Movil"), ("FMX", "Mexico", "FEMSA"), ("KOF", "Mexico", "Coca-Cola FEMSA"),
    ("PAC", "Mexico", "Pacific Airport Group"), ("OMAB", "Mexico", "OMA"), ("ASR", "Mexico", "ASUR"),
    ("VLRS", "Mexico", "Volaris"), ("CX", "Mexico", "Cemex"),
    ("YPF", "Argentina", "YPF"), ("GGAL", "Argentina", "Grupo Financiero Galicia"), ("BMA", "Argentina", "Banco Macro"),
    ("PAM", "Argentina", "Pampa Energia"), ("TEO", "Argentina", "Telecom Argentina"), ("LOMA", "Argentina", "Loma Negra"),
    ("DESP", "Argentina", "Despegar"),
    ("SQM", "Chile", "SQM"), ("BCH", "Chile", "Banco de Chile"), ("CCU", "Chile", "CCU"),
    ("ENIC", "Chile", "Enel Chile"), ("LTM", "Chile", "LATAM Airlines"),
    ("CIB", "Colombia", "Bancolombia"), ("EC", "Colombia", "Ecopetrol"),
    ("BAP", "Peru", "Credicorp"), ("BVN", "Peru", "Buenaventura"),
    ("BUD", "Belgium", "Anheuser-Busch InBev"), ("UCBJY", "Belgium", "UCB"),
    ("BHP", "Australia", "BHP"), ("WDS", "Australia", "Woodside"), ("CSLLY", "Australia", "CSL"),
    ("MQBKY", "Australia", "Macquarie"),
    ("NPSNY", "South Africa", "Naspers"), ("SBSW", "South Africa", "Sibanye Stillwater"),
    ("GFI", "South Africa", "Gold Fields"), ("SSL", "South Africa", "Sasol"),
    ("SE", "Singapore", "Sea Limited"), ("GRAB", "Singapore", "Grab"), ("DBSDY", "Singapore", "DBS Group"),
    ("UOVEY", "Singapore", "UOB"),
    ("MELI", "Uruguay", "MercadoLibre"), ("GLOB", "Luxembourg", "Globant"), ("SPOT", "Luxembourg", "Spotify"),
]

COUNTRY_ALIASES = {
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "Great Britain": "United Kingdom",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Korea": "South Korea",
    "Republic of Korea": "South Korea",
    "Taiwan Province of China": "Taiwan",
    "Hong Kong SAR China": "Hong Kong",
}


def normalize_ticker(x: Any) -> str | None:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().upper().replace(".", "-").replace(" ", "")
    if not s or s in {"NAN", "N/A", "--"}:
        return None
    if re.search(r"[^A-Z0-9\-]", s):
        return None
    return s


def normalize_country(x: Any) -> str | None:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "n/a"}:
        return None
    return COUNTRY_ALIASES.get(s, s)


def latest_usall_snapshot() -> tuple[str, pd.DataFrame]:
    candidates: list[tuple[datetime, Path]] = []
    for p in DATA_DIR.glob("*/*.csv"):
        m = SNAP_RE.match(p.name)
        if not m:
            continue
        if p.stat().st_size <= 0:
            continue
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d")
        candidates.append((dt, p))
    if not candidates:
        raise FileNotFoundError("No valid usall snapshot found under data/YYYYMM")
    candidates.sort(key=lambda x: x[0])
    dt, path = candidates[-1]
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["ICC"] = pd.to_numeric(df["ICC"], errors="coerce")
    df["mktcap"] = pd.to_numeric(df.get("mktcap"), errors="coerce")
    df = df[df["ticker"].notna() & df["ICC"].notna() & df["mktcap"].notna() & (df["mktcap"] > 0)].copy()
    return dt.strftime("%Y-%m-%d"), df


def read_etf_config() -> pd.DataFrame:
    path = CONFIG_DIR / "etfs.csv"
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path)
        for col in ["ticker", "label", "category"]:
            if col not in df.columns:
                df[col] = ""
        df = df[["ticker", "label", "category"]].copy()
    else:
        df = pd.DataFrame(DEFAULT_ETFS, columns=["ticker", "label", "category"])
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df = df[df["ticker"].notna()].drop_duplicates("ticker")
    return df


def http_get(url: str, timeout: int = 25, retries: int = 2) -> str:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_error = exc
            time.sleep(0.6 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def parse_percent(x: Any) -> float | None:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().replace(",", "")
    if not s:
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return None
    try:
        v = float(s)
        return v / 100.0 if v > 1.5 else v
    except Exception:
        return None


def fetch_stockanalysis_holdings(etf: str) -> pd.DataFrame:
    url = f"https://stockanalysis.com/etf/{etf.lower()}/holdings/"
    html = http_get(url)
    tables = pd.read_html(io.StringIO(html))
    for table in tables:
        cols = {str(c).strip().lower(): c for c in table.columns}
        sym_col = None
        weight_col = None
        for key in cols:
            if key in {"symbol", "ticker"} or "symbol" in key:
                sym_col = cols[key]
            if "weight" in key or "%" in key:
                weight_col = cols[key]
        if sym_col is None or weight_col is None:
            continue
        out = pd.DataFrame({
            "ticker": table[sym_col].map(normalize_ticker),
            "weight": table[weight_col].map(parse_percent),
        })
        out = out[out["ticker"].notna() & out["weight"].notna() & (out["weight"] > 0)].copy()
        if len(out) >= 5:
            out["source"] = "stockanalysis_full_holdings"
            return out
    raise RuntimeError(f"No usable holdings table for {etf}")


def build_etf(asof_date: str, usall: pd.DataFrame) -> pd.DataFrame:
    base = usall[["ticker", "ICC", "mktcap"]].drop_duplicates("ticker").copy()
    rows = []
    members = []
    for _, r in read_etf_config().iterrows():
        etf = str(r["ticker"])
        label = str(r.get("label") or etf)
        category = str(r.get("category") or "")
        try:
            holdings = fetch_stockanalysis_holdings(etf)
            merged = holdings.merge(base, on="ticker", how="left")
            matched = merged["ICC"].notna() & merged["weight"].notna()
            n_holdings = int(len(holdings))
            n_matched = int(matched.sum())
            coverage = float(merged.loc[matched, "weight"].sum()) if n_matched else 0.0
            if n_matched >= 5 and coverage >= 0.70:
                weights = merged.loc[matched, "weight"]
                icc = float(np.average(merged.loc[matched, "ICC"], weights=weights))
                method = "ICC calculation"
                status = "ok_full_holdings"
            elif n_matched >= 3:
                weights = merged.loc[matched, "weight"]
                icc = float(np.average(merged.loc[matched, "ICC"], weights=weights))
                method = "Partial holdings estimate"
                status = "partial_holdings"
            else:
                icc = np.nan
                method = "Unavailable"
                status = "insufficient_matched_holdings"
            rows.append({
                "date": asof_date,
                "ticker": etf,
                "label": label,
                "category": category,
                "icc": icc,
                "coverage_weight": coverage,
                "n_holdings": n_holdings,
                "n_matched": n_matched,
                "method": method,
                "holding_source": "stockanalysis_full_holdings",
                "status": status,
            })
            for _, m in merged.loc[matched].iterrows():
                members.append({"date": asof_date, "etf": etf, "ticker": m["ticker"], "weight": m["weight"], "icc": m["ICC"]})
        except Exception as exc:
            rows.append({
                "date": asof_date,
                "ticker": etf,
                "label": label,
                "category": category,
                "icc": np.nan,
                "coverage_weight": 0.0,
                "n_holdings": 0,
                "n_matched": 0,
                "method": "Unavailable",
                "holding_source": "stockanalysis_full_holdings",
                "status": f"error: {type(exc).__name__}",
            })
    out = pd.DataFrame(rows)
    mem = pd.DataFrame(members)
    yyyymm = asof_date[:7].replace("-", "")
    tag = asof_date[:4] + "_" + asof_date[5:7] + asof_date[8:10]
    out_dir = DERIVED_ETF_DIR / yyyymm
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"etf_icc_{tag}.csv", index=False)
    mem.to_csv(out_dir / f"etf_constituents_{tag}.csv", index=False)
    return out


def build_country(asof_date: str, usall: pd.DataFrame) -> pd.DataFrame:
    seeds = pd.DataFrame(ADR_SEEDS, columns=["ticker", "country", "company"])
    seeds["ticker"] = seeds["ticker"].map(normalize_ticker)
    seeds["country"] = seeds["country"].map(normalize_country)
    seeds = seeds.dropna(subset=["ticker", "country"]).drop_duplicates("ticker")
    base = usall[["ticker", "ICC", "mktcap"]].drop_duplicates("ticker").copy()
    merged = seeds.merge(base, on="ticker", how="left")
    merged = merged[merged["ICC"].notna() & merged["mktcap"].notna() & (merged["mktcap"] > 0)].copy()
    rows = []
    members = []
    for country, g in merged.groupby("country"):
        g = g.sort_values("mktcap", ascending=False).head(10).copy()
        n = int(len(g))
        if n >= 4:
            method = "ADR Top-10 ICC calculation"
            status = "adr_top10_icc_calculation"
        elif n >= 2:
            method = "Partial ADR estimate"
            status = "partial_adr_estimate"
        else:
            continue
        weights = g["mktcap"] / g["mktcap"].sum()
        icc = float(np.average(g["ICC"], weights=weights))
        rows.append({
            "date": asof_date,
            "country": country,
            "icc": icc,
            "n_candidates": int((seeds["country"] == country).sum()),
            "n_selected": n,
            "n_icc_available": n,
            "coverage_mktcap": 1.0,
            "method": method,
            "status": status,
        })
        for _, m in g.iterrows():
            members.append({"date": asof_date, "country": country, "ticker": m["ticker"], "company": m["company"], "icc": m["ICC"], "mktcap": m["mktcap"]})
    out = pd.DataFrame(rows).sort_values("country") if rows else pd.DataFrame(columns=["date", "country", "icc", "n_candidates", "n_selected", "n_icc_available", "coverage_mktcap", "method", "status"])
    mem = pd.DataFrame(members)
    yyyymm = asof_date[:7].replace("-", "")
    tag = asof_date[:4] + "_" + asof_date[5:7] + asof_date[8:10]
    out_dir = DERIVED_COUNTRY_DIR / yyyymm
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"country_adr_icc_{tag}.csv", index=False)
    mem.to_csv(out_dir / f"country_adr_constituents_{tag}.csv", index=False)
    return out


def main() -> None:
    asof_date, usall = latest_usall_snapshot()
    print(f"[ensure_daily_derived] latest usall date: {asof_date}, rows={len(usall)}")
    etf = build_etf(asof_date, usall)
    country = build_country(asof_date, usall)
    print(f"[ensure_daily_derived] ETF rows: {len(etf)}")
    print(f"[ensure_daily_derived] Country / Region rows: {len(country)}")


if __name__ == "__main__":
    main()
