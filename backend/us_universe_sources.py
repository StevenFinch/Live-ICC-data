from __future__ import annotations

import datetime as dt
import io
import json
import pathlib
import re
import time

import pandas as pd
import requests


ROOT = pathlib.Path(__file__).resolve().parents[1]
COVERAGE_DIR = ROOT / "data" / "coverage"

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

DATAHUB_FALLBACK = {
    "nasdaq": "https://raw.githubusercontent.com/datasets/nasdaq-listings/main/data/nasdaq-listed.csv",
    "nyse": "https://raw.githubusercontent.com/datasets/nyse-other-listings/main/data/nyse-listed.csv",
    "amex": "https://raw.githubusercontent.com/datasets/nyse-other-listings/main/data/other-listed.csv",
}

BAD_SYM = re.compile(r"[^A-Z\.\-]")
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

EXCLUDE_NAME_PATTERNS = [
    "warrant",
    "rights",
    "right",
    "unit",
    "units",
    "preferred",
    "preference",
    "depositary shares each representing",
    "depositary share each representing",
    "senior notes",
    "senior note",
    "notes due",
    "note due",
    "debenture",
    "bond",
    "baby bond",
    "etf",
    "exchange traded fund",
    "fund",
    "etn",
    "exchange traded note",
    "closed end fund",
    "closed-end fund",
    "trust units",
    "subscription receipt",
    "contingent value right",
    "cvr",
]

INCLUDE_NAME_HINTS = [
    "common stock",
    "common shares",
    "common share",
    "ordinary shares",
    "ordinary share",
    "american depositary",
    "american depositary shares",
    "american depository",
    "ads",
    "adr",
    "class a",
    "class b",
    "class c",
    "limited voting shares",
    "subordinate voting shares",
    "common units",
]


def _now_tag() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _get_text(url: str, timeout: int = 30, retries: int = 3) -> str:
    """Fetch text with retry logic."""
    headers = {"User-Agent": USER_AGENT}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_err = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def _normalize_symbol(x: object) -> str:
    """Normalize ticker symbols into Yahoo-compatible notation."""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace(".", "-")
    return s


def _valid_symbol(s: str) -> bool:
    if not s or s.lower() == "nan":
        return False
    if BAD_SYM.search(s):
        return False
    bad_suffixes = ("WS", "WT", "WTR")
    if any(s.endswith(suf) for suf in bad_suffixes):
        return False
    return True


def _parse_pipe_file(text: str) -> pd.DataFrame:
    """Parse Nasdaq Trader pipe-delimited symbol directory text."""
    lines = [line for line in text.splitlines() if line and "|" in line]
    if not lines:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)
    if "Symbol" in df.columns:
        df = df[df["Symbol"].astype(str).str.upper() != "FILE CREATION TIME"].copy()
    if "ACT Symbol" in df.columns:
        df = df[df["ACT Symbol"].astype(str).str.upper() != "FILE CREATION TIME"].copy()
    return df


def _fetch_nasdaq_listed() -> pd.DataFrame:
    """Fetch Nasdaq-listed securities from Nasdaq Trader."""
    raw = _parse_pipe_file(_get_text(NASDAQ_LISTED_URL))
    if raw.empty:
        return pd.DataFrame()

    df = pd.DataFrame()
    df["ticker"] = raw["Symbol"].map(_normalize_symbol)
    df["security_name"] = raw.get("Security Name", "").astype(str)
    df["exchange"] = "NASDAQ"
    df["source"] = "nasdaqtrader_nasdaqlisted"
    df["is_etf"] = raw.get("ETF", "N").astype(str).str.upper().eq("Y")
    df["is_test_issue"] = raw.get("Test Issue", "N").astype(str).str.upper().eq("Y")
    df["raw_category"] = raw.get("Market Category", "")
    return df


def _fetch_other_listed() -> pd.DataFrame:
    """Fetch NYSE/NYSE American/other listed securities from Nasdaq Trader."""
    raw = _parse_pipe_file(_get_text(OTHER_LISTED_URL))
    if raw.empty:
        return pd.DataFrame()

    symbol_col = "ACT Symbol" if "ACT Symbol" in raw.columns else raw.columns[0]
    df = pd.DataFrame()
    df["ticker"] = raw[symbol_col].map(_normalize_symbol)
    df["security_name"] = raw.get("Security Name", "").astype(str)
    df["exchange"] = raw.get("Exchange", "").astype(str)
    df["source"] = "nasdaqtrader_otherlisted"
    df["is_etf"] = raw.get("ETF", "N").astype(str).str.upper().eq("Y")
    df["is_test_issue"] = raw.get("Test Issue", "N").astype(str).str.upper().eq("Y")
    df["raw_category"] = raw.get("Exchange", "")
    return df


def _load_datahub_fallback() -> pd.DataFrame:
    """Load the legacy DataHub universe as a fallback source."""
    rows = []
    for name, url in DATAHUB_FALLBACK.items():
        try:
            raw = pd.read_csv(url, dtype=str)
            sym_col = raw.columns[0]
            tmp = pd.DataFrame()
            tmp["ticker"] = raw[sym_col].map(_normalize_symbol)
            tmp["security_name"] = raw.iloc[:, 1].astype(str) if raw.shape[1] > 1 else ""
            tmp["exchange"] = name.upper()
            tmp["source"] = f"datahub_{name}"
            tmp["is_etf"] = False
            tmp["is_test_issue"] = False
            tmp["raw_category"] = ""
            rows.append(tmp)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def classify_common_equity(df: pd.DataFrame) -> pd.DataFrame:
    """Classify likely common-equity candidates while preserving a broad universe."""
    out = df.copy()
    out["ticker"] = out["ticker"].map(_normalize_symbol)
    out["security_name"] = out["security_name"].fillna("").astype(str)
    name_l = out["security_name"].str.lower()

    excluded_by_name = pd.Series(False, index=out.index)
    for pat in EXCLUDE_NAME_PATTERNS:
        excluded_by_name = excluded_by_name | name_l.str.contains(re.escape(pat), na=False)

    included_by_name = pd.Series(False, index=out.index)
    for pat in INCLUDE_NAME_HINTS:
        included_by_name = included_by_name | name_l.str.contains(re.escape(pat), na=False)

    valid_symbol = out["ticker"].map(_valid_symbol)
    out["exclude_reason"] = ""
    out.loc[~valid_symbol, "exclude_reason"] = "invalid_symbol"
    out.loc[out["is_test_issue"], "exclude_reason"] = "test_issue"
    out.loc[out["is_etf"], "exclude_reason"] = "etf"
    out.loc[excluded_by_name & ~included_by_name, "exclude_reason"] = "non_common_security_name"

    out["is_common_equity_candidate"] = (
        valid_symbol
        & ~out["is_test_issue"]
        & ~out["is_etf"]
        & ~(excluded_by_name & ~included_by_name)
    )
    return out


def build_us_universe(write_report: bool = True) -> pd.DataFrame:
    """Build a broad U.S.-listed common-equity candidate universe."""
    frames = []
    errors = []

    for loader in (_fetch_nasdaq_listed, _fetch_other_listed):
        try:
            frames.append(loader())
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")

    try:
        frames.append(_load_datahub_fallback())
    except Exception as exc:
        errors.append(f"datahub_fallback: {exc}")

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise RuntimeError(f"No ticker universe source succeeded: {errors}")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["ticker"]).copy()
    df["ticker"] = df["ticker"].map(_normalize_symbol)
    df = df[df["ticker"].map(_valid_symbol)].copy()
    df = classify_common_equity(df)

    df["source_rank"] = df["source"].map(
        {"nasdaqtrader_nasdaqlisted": 0, "nasdaqtrader_otherlisted": 1}
    ).fillna(9)
    df = df.sort_values(["ticker", "source_rank"]).drop_duplicates("ticker", keep="first")
    df = df.sort_values("ticker").reset_index(drop=True)

    if write_report:
        COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
        tag = _now_tag()
        df.to_csv(COVERAGE_DIR / f"us_universe_candidates_{tag}.csv", index=False)
        summary = {
            "generated_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "n_total_listed_candidates": int(len(df)),
            "n_common_equity_candidates": int(df["is_common_equity_candidate"].sum()),
            "n_excluded": int((~df["is_common_equity_candidate"]).sum()),
            "sources": sorted(df["source"].dropna().unique().tolist()),
            "source_errors": errors,
        }
        with open(COVERAGE_DIR / f"us_universe_summary_{tag}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return df


def get_us_common_tickers(write_report: bool = True) -> list[str]:
    """Return broad U.S.-listed common-equity candidate tickers."""
    df = build_us_universe(write_report=write_report)
    tickers = df.loc[df["is_common_equity_candidate"], "ticker"].dropna().astype(str).tolist()
    return sorted(dict.fromkeys(tickers))
