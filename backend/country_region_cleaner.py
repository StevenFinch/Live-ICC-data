from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COUNTRY_ALIASES = {
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "u.k": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "united kingdom": "United Kingdom",
    "united kingdom of great britain and northern ireland": "United Kingdom",
    "england": "United Kingdom",
    "south korea": "South Korea",
    "korea, south": "South Korea",
    "republic of korea": "South Korea",
    "korea republic of": "South Korea",
    "korea": "South Korea",
    "people's republic of china": "China",
    "prc": "China",
    "china mainland": "China",
    "mainland china": "China",
    "hong kong sar": "Hong Kong",
    "hong kong sar china": "Hong Kong",
    "hong kong, china": "Hong Kong",
    "taiwan province of china": "Taiwan",
    "taiwan, china": "Taiwan",
    "russian federation": "Russia",
    "viet nam": "Vietnam",
    "u.a.e.": "United Arab Emirates",
    "uae": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "slovak republic": "Slovakia",
    "macau": "Macao",
    "macao sar china": "Macao",
}

COUNTRY_COL_CANDIDATES = ["country", "country_region", "Country / Region", "Country", "region"]
DATE_COL_CANDIDATES = ["date", "month_end_date", "asof_date", "Date"]
ICC_COL_CANDIDATES = ["icc", "daily_icc", "Monthly ICC", "ICC", "value"]
METHOD_COL_CANDIDATES = ["method", "Method", "status"]
COVERAGE_COL_CANDIDATES = ["coverage_mktcap", "coverage_weight", "coverage", "Coverage"]
N_COL_CANDIDATES = ["n_icc_available", "n_available", "n_selected", "Available ADRs", "n_firms"]


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def normalize_country_region(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    key = " ".join(text.replace("_", " ").split()).lower()
    return COUNTRY_ALIASES.get(key, text)


def _is_unavailable_series(df: pd.DataFrame) -> pd.Series:
    method_col = _first_existing(list(df.columns), METHOD_COL_CANDIDATES)
    icc_col = _first_existing(list(df.columns), ICC_COL_CANDIDATES)
    unavailable = pd.Series(False, index=df.index)
    if method_col:
        m = df[method_col].astype(str).str.lower()
        unavailable = unavailable | m.str.contains("unavailable", na=False)
    if icc_col:
        icc = pd.to_numeric(df[icc_col], errors="coerce")
        unavailable = unavailable | icc.isna()
    return unavailable


def _row_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    method_col = _first_existing(list(df.columns), METHOD_COL_CANDIDATES)
    icc_col = _first_existing(list(df.columns), ICC_COL_CANDIDATES)
    coverage_col = _first_existing(list(df.columns), COVERAGE_COL_CANDIDATES)
    n_col = _first_existing(list(df.columns), N_COL_CANDIDATES)

    if icc_col:
        score += pd.to_numeric(df[icc_col], errors="coerce").notna().astype(float) * 1000
    if method_col:
        m = df[method_col].astype(str).str.lower()
        score += (~m.str.contains("unavailable", na=False)).astype(float) * 500
        score += m.str.contains("adr top-10|icc calculation", na=False).astype(float) * 100
        score += m.str.contains("partial", na=False).astype(float) * 50
    if n_col:
        score += pd.to_numeric(df[n_col], errors="coerce").fillna(0).clip(lower=0, upper=100)
    if coverage_col:
        score += pd.to_numeric(df[coverage_col], errors="coerce").fillna(0).clip(lower=0, upper=1) * 10
    return score


def clean_country_region_df(df: pd.DataFrame, drop_unavailable: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    country_col = _first_existing(list(df.columns), COUNTRY_COL_CANDIDATES)
    if country_col is None:
        return df

    df[country_col] = df[country_col].map(normalize_country_region)
    df = df[df[country_col].astype(str).str.strip() != ""].copy()

    if drop_unavailable:
        df = df[~_is_unavailable_series(df)].copy()

    if df.empty:
        return df.reset_index(drop=True)

    date_col = _first_existing(list(df.columns), DATE_COL_CANDIDATES)
    group_cols = [country_col]
    if date_col:
        group_cols = [country_col, date_col]

    df["__score"] = _row_score(df)
    sort_cols = group_cols + ["__score"]
    ascending = [True] * len(group_cols) + [False]
    df = df.sort_values(sort_cols, ascending=ascending)
    df = df.drop_duplicates(subset=group_cols, keep="first")
    df = df.drop(columns=["__score"], errors="ignore")

    if date_col:
        df = df.sort_values([country_col, date_col], ascending=[True, False])
    else:
        df = df.sort_values(country_col)

    return df.reset_index(drop=True)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x
    return obj


def clean_country_json(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ["latest", "daily", "monthly", "display_monthly", "rows"]:
        if isinstance(data.get(key), list):
            df = pd.DataFrame(data[key])
            cleaned = clean_country_region_df(df, drop_unavailable=True)
            data[key] = cleaned.to_dict(orient="records")

    data["label"] = "Country / Region Level ICC"
    data["note"] = "Country / Region Level ICC is based on available ADR and foreign-listing composites. Unavailable country/region rows are hidden from public tables."

    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, allow_nan=False)
