from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.errors import EmptyDataError

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DATA_DIR = REPO / "docs" / "data"
DOWNLOADS_DIR = DOCS_DATA_DIR / "downloads"
LIBRARY_DIR = REPO / "data" / "library"
RAW_DIR = DOWNLOADS_DIR / "raw"
RAW_ARCHIVE_DIR = DOWNLOADS_DIR / "raw_archives"
CONFIG_DIR = REPO / "config"

SNAPSHOT_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_\-]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)

INDEX_UNIVERSES = [
    "sp500",
    "sp100",
    "dow30",
    "ndx100",
    "sp400",
    "sp600",
    "rut1000",
]

RAW_GROUPS = {
    "usall": {"usall"},
    "sp500": {"sp500"},
    "other_indices": set(INDEX_UNIVERSES) - {"sp500"},
}


def parse_snapshot_meta(path: Path) -> dict[str, Any] | None:
    m = SNAPSHOT_RE.match(path.name)
    if not m:
        return None
    year = m.group("year")
    mmdd = m.group("mmdd")
    rerun = int(m.group("rerun") or 0)
    return {
        "universe": m.group("universe"),
        "year": year,
        "mmdd": mmdd,
        "yyyymm": f"{year}{mmdd[:2]}",
        "yyyymmdd": f"{year}{mmdd}",
        "rerun": rerun,
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = (
        values.notna()
        & weights.notna()
        & np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def find_all_snapshots(universe: str | None = None) -> list[Path]:
    files: list[tuple[tuple[str, int, str], Path]] = []
    for p in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(p)
        if meta is None:
            continue
        if universe is not None and meta["universe"] != universe:
            continue
        files.append(((meta["yyyymmdd"], meta["rerun"], p.name), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        raise EmptyDataError(f"No columns parsed from file: {path}")

    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    for col in ["mktcap", "ICC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "bm" in df.columns:
        df["bm"] = pd.to_numeric(df["bm"], errors="coerce")
    else:
        df["bm"] = np.nan

    if "sector" not in df.columns:
        df["sector"] = None
    if "name" not in df.columns:
        df["name"] = None

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def try_load_snapshot(path: Path) -> pd.DataFrame | None:
    try:
        return load_snapshot(path)
    except Exception as e:
        print(f"[build_docs_data] skipping invalid snapshot: {path} | {type(e).__name__}: {e}")
        return None


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["ticker"].notna()
        & df["date"].notna()
        & df["mktcap"].notna()
        & df["ICC"].notna()
        & np.isfinite(df["mktcap"])
        & np.isfinite(df["ICC"])
        & (df["mktcap"] > 0)
    ].copy()


def get_valid_snapshots(paths: list[Path]) -> list[tuple[Path, pd.DataFrame]]:
    valid: list[tuple[Path, pd.DataFrame]] = []
    for path in paths:
        df = try_load_snapshot(path)
        if df is None:
            continue
        df = clean_df(df)
        if df.empty:
            continue
        valid.append((path, df))
    return valid


def dedup_valid_snapshots(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> list[tuple[Path, pd.DataFrame]]:
    keep: dict[tuple[str, str], tuple[Path, pd.DataFrame]] = {}
    for path, df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        date_value = str(df["date"].dropna().iloc[0]) if df["date"].notna().any() else None
        if not date_value:
            continue
        keep[(meta["universe"], date_value)] = (path, df)
    out = list(keep.values())
    out.sort(
        key=lambda x: (
            parse_snapshot_meta(x[0])["universe"],
            str(x[1]["date"].dropna().iloc[0]),
            parse_snapshot_meta(x[0])["rerun"],
            x[0].name,
        )
    )
    return out


def build_market_history(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, df in valid_snapshots:
        rows.append(
            {
                "date": str(df["date"].iloc[0]),
                "vw_icc": weighted_mean(df["ICC"], df["mktcap"]),
                "ew_icc": float(df["ICC"].mean()),
                "n_firms": int(len(df)),
                "source_file": str(path.relative_to(REPO)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "vw_icc", "ew_icc", "n_firms", "source_file"])
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def build_monthly_from_daily(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    group_cols = group_cols or []
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["yyyymm"] = tmp["date"].dt.strftime("%Y-%m")
    sort_cols = group_cols + ["date"]
    tmp = tmp.sort_values(sort_cols)
    if group_cols:
        monthly = tmp.groupby(group_cols + ["yyyymm"], as_index=False).tail(1)
    else:
        monthly = tmp.groupby(["yyyymm"], as_index=False).tail(1)
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m-%d")
    return monthly.reset_index(drop=True)


def last_n_months(df: pd.DataFrame, n: int = 3, group_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    group_cols = group_cols or []
    if group_cols:
        return (
            df.sort_values(group_cols + ["yyyymm"])
            .groupby(group_cols, as_index=False, group_keys=False)
            .tail(n)
            .reset_index(drop=True)
        )
    return df.sort_values("yyyymm").tail(n).reset_index(drop=True)


def build_value_history(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, raw_df in valid_snapshots:
        df = raw_df.copy()
        df = df[
            df["date"].notna()
            & df["mktcap"].notna()
            & df["bm"].notna()
            & df["ICC"].notna()
            & np.isfinite(df["mktcap"])
            & np.isfinite(df["bm"])
            & np.isfinite(df["ICC"])
            & (df["mktcap"] > 0)
            & (df["bm"] > 0)
        ].copy()
        if df.empty:
            continue
        q_lo = float(df["ICC"].quantile(0.005))
        q_hi = float(df["ICC"].quantile(0.995))
        df = df[(df["ICC"] >= q_lo) & (df["ICC"] <= q_hi)].copy()
        if df.empty:
            continue
        size_median = float(df["mktcap"].median())
        bm30 = float(df["bm"].quantile(0.30))
        bm70 = float(df["bm"].quantile(0.70))
        df["size_bucket"] = np.where(df["mktcap"] <= size_median, "S", "B")
        df["bm_bucket"] = pd.cut(
            df["bm"],
            bins=[-np.inf, bm30, bm70, np.inf],
            labels=["L", "M", "H"],
            include_lowest=True,
            right=True,
        ).astype("string")
        df["portfolio"] = df["size_bucket"] + "/" + df["bm_bucket"].astype(str)

        pt = (
            df.groupby("portfolio", dropna=False)
            .apply(lambda g: pd.Series({"vw_icc": weighted_mean(g["ICC"], g["mktcap"]), "n_firms": len(g)}))
            .reset_index()
        )
        d = dict(zip(pt["portfolio"], pt["vw_icc"]))
        icc_sl = d.get("S/L", np.nan)
        icc_bl = d.get("B/L", np.nan)
        icc_sh = d.get("S/H", np.nan)
        icc_bh = d.get("B/H", np.nan)
        growth_icc = np.nan if (pd.isna(icc_sl) or pd.isna(icc_bl)) else float((icc_sl + icc_bl) / 2.0)
        value_icc = np.nan if (pd.isna(icc_sh) or pd.isna(icc_bh)) else float((icc_sh + icc_bh) / 2.0)
        ivp_bm = np.nan if (pd.isna(value_icc) or pd.isna(growth_icc)) else float(value_icc - growth_icc)
        rows.append(
            {
                "date": str(df["date"].iloc[0]),
                "value_icc": value_icc,
                "growth_icc": growth_icc,
                "ivp_bm": ivp_bm,
                "n_firms": int(len(df)),
                "source_file": str(path.relative_to(REPO)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "value_icc", "growth_icc", "ivp_bm", "n_firms", "source_file"])
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def build_industry_daily(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    latest_rows = []
    for path, raw_df in valid_snapshots:
        df = raw_df.copy()
        df = df[df["sector"].notna()].copy()
        if df.empty:
            continue
        sector_df = (
            df.groupby("sector", dropna=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
                        "ew_icc": float(g["ICC"].mean()),
                        "n_firms": int(len(g)),
                        "total_mktcap": float(g["mktcap"].sum()),
                    }
                )
            )
            .reset_index()
        )
        date_value = str(df["date"].iloc[0])
        sector_df["date"] = date_value
        latest_rows.append(sector_df)
        summary_rows.append(
            {
                "date": date_value,
                "industry_icc_eq_sector": float(sector_df["vw_icc"].mean()),
                "industry_icc_cap_sector": weighted_mean(sector_df["vw_icc"], sector_df["total_mktcap"]),
                "n_sectors": int(len(sector_df)),
                "source_file": str(path.relative_to(REPO)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("date").drop_duplicates("date", keep="last") if summary_rows else pd.DataFrame(columns=["date", "industry_icc_eq_sector", "industry_icc_cap_sector", "n_sectors", "source_file"])
    latest_all = pd.concat(latest_rows, ignore_index=True) if latest_rows else pd.DataFrame(columns=["sector", "vw_icc", "ew_icc", "n_firms", "total_mktcap", "date"])
    if not latest_all.empty:
        latest_day = latest_all["date"].max()
        latest_all = latest_all[latest_all["date"] == latest_day].sort_values(["total_mktcap", "sector"], ascending=[False, True]).reset_index(drop=True)
    return summary.reset_index(drop=True), latest_all.reset_index(drop=True)


def upsert_history(current_df: pd.DataFrame, history_path: Path, key_cols: list[str]) -> pd.DataFrame:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        hist = pd.read_csv(history_path)
    else:
        hist = pd.DataFrame(columns=current_df.columns)
    combined = pd.concat([hist, current_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    sort_cols = [c for c in ["date"] + key_cols if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    combined.to_csv(history_path, index=False)
    return combined


def read_csv_config(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=expected_cols)
    df = pd.read_csv(path)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols].copy()


def get_top_holdings(etf_ticker: str) -> pd.DataFrame:
    t = yf.Ticker(etf_ticker)
    fd = getattr(t, "funds_data", None)
    if fd is None:
        raise ValueError("funds_data is unavailable")
    holdings = getattr(fd, "top_holdings", None)
    if holdings is None:
        raise ValueError("top_holdings is unavailable")
    h = holdings.copy() if isinstance(holdings, pd.DataFrame) else pd.DataFrame(holdings)
    if h.empty:
        raise ValueError("top_holdings is empty")
    cols = {str(c).lower(): c for c in h.columns}
    symbol_col = None
    weight_col = None
    for cand in ["symbol", "holding", "ticker"]:
        if cand in cols:
            symbol_col = cols[cand]
            break
    for cand in ["holdingpercent", "holding_percent", "weight", "percent", "pct"]:
        if cand in cols:
            weight_col = cols[cand]
            break
    if symbol_col is None:
        symbol_col = h.columns[0]
    if weight_col is None:
        for c in h.columns[::-1]:
            vals = pd.to_numeric(h[c], errors="coerce")
            if vals.notna().sum() >= max(1, len(h) // 2):
                weight_col = c
                break
    if weight_col is None:
        raise ValueError("could not identify weight column")
    out = pd.DataFrame(
        {
            "symbol": h[symbol_col].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False),
            "raw_weight": pd.to_numeric(h[weight_col], errors="coerce"),
        }
    )
    out = out[out["symbol"].notna() & out["raw_weight"].notna()].copy()
    out["weight"] = np.where(out["raw_weight"].abs() > 1.5, out["raw_weight"] / 100.0, out["raw_weight"])
    out = out[out["weight"] > 0].copy()
    if out.empty:
        raise ValueError("no positive weights")
    out["weight"] = out["weight"] / out["weight"].sum()
    return out[["symbol", "weight"]]


def build_holdings_proxy(config_df: pd.DataFrame, latest_usall: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = clean_df(latest_usall)[["ticker", "ICC", "mktcap", "sector", "name"]].copy()
    base["ticker"] = base["ticker"].astype(str).str.upper().str.strip()

    rows = []
    member_rows = []
    today = str(base.shape[0] and latest_usall["date"].dropna().iloc[0]) if not latest_usall.empty else None

    for _, cfg in config_df.iterrows():
        ticker = str(cfg.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        label = str(cfg.get("label", ticker))
        category = str(cfg.get("category", "")) if "category" in cfg.index else ""
        country = str(cfg.get("country", "")) if "country" in cfg.index else ""
        try:
            h = get_top_holdings(ticker)
            merged = h.merge(base, left_on="symbol", right_on="ticker", how="left")
            matched = merged["ICC"].notna()
            coverage = float(merged.loc[matched, "weight"].sum()) if len(merged) else 0.0
            icc = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"]) if matched.any() else np.nan
            rows.append(
                {
                    "date": today,
                    "ticker": ticker,
                    "label": label,
                    "category": category,
                    "country": country,
                    "vw_icc": icc,
                    "coverage_weight": coverage,
                    "n_holdings": int(len(h)),
                    "n_matched": int(matched.sum()),
                    "holding_source": "yfinance_top_holdings",
                    "status": "ok" if matched.any() else "no_matched_holdings",
                }
            )
            if matched.any():
                top = merged.loc[matched, ["symbol", "weight", "ICC", "sector", "name"]].copy()
                top["parent_ticker"] = ticker
                top["parent_label"] = label
                top["date"] = today
                member_rows.append(top)
        except Exception as e:
            rows.append(
                {
                    "date": today,
                    "ticker": ticker,
                    "label": label,
                    "category": category,
                    "country": country,
                    "vw_icc": np.nan,
                    "coverage_weight": np.nan,
                    "n_holdings": np.nan,
                    "n_matched": np.nan,
                    "holding_source": "yfinance_top_holdings",
                    "status": f"error: {type(e).__name__}",
                }
            )
    current = pd.DataFrame(rows)
    members = pd.concat(member_rows, ignore_index=True) if member_rows else pd.DataFrame(columns=["symbol", "weight", "ICC", "sector", "name", "parent_ticker", "parent_label", "date"])
    if kind == "country" and not current.empty:
        current = current[["date", "country", "ticker", "label", "vw_icc", "coverage_weight", "n_holdings", "n_matched", "holding_source", "status"]]
    elif kind == "etf" and not current.empty:
        current = current[["date", "ticker", "label", "category", "vw_icc", "coverage_weight", "n_holdings", "n_matched", "holding_source", "status"]]
    return current, members


def build_history_summary(history_df: pd.DataFrame, family: str) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=["date", f"{family}_icc", "n_items"])
    tmp = history_df.copy()
    if family == "etf":
        value_col = "vw_icc"
        group_col = "ticker"
    else:
        value_col = "vw_icc"
        group_col = "country"
    summary = (
        tmp.groupby("date", as_index=False)
        .apply(lambda g: pd.Series({f"{family}_icc": float(g[value_col].dropna().mean()) if g[value_col].notna().any() else np.nan, "n_items": int(g[group_col].nunique())}))
        .reset_index(drop=True)
    )
    return summary


def build_index_outputs(all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history_rows = []
    latest_rows = []
    summary_rows = []
    for universe in ["sp500"] + [u for u in INDEX_UNIVERSES if u != "sp500"]:
        valid = all_valid_by_universe.get(universe, [])
        if not valid:
            continue
        hist = build_market_history(valid)
        if hist.empty:
            continue
        hist = hist.copy()
        hist["universe"] = universe
        history_rows.append(hist)
        latest = hist.sort_values("date").iloc[-1].to_dict()
        latest_rows.append(
            {
                "universe": universe,
                "date": latest.get("date"),
                "vw_icc": latest.get("vw_icc"),
                "ew_icc": latest.get("ew_icc"),
                "n_firms": latest.get("n_firms"),
                "source_file": latest.get("source_file"),
            }
        )
    history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame(columns=["date", "vw_icc", "ew_icc", "n_firms", "source_file", "universe"])
    latest = pd.DataFrame(latest_rows).sort_values("universe") if latest_rows else pd.DataFrame(columns=["universe", "date", "vw_icc", "ew_icc", "n_firms", "source_file"])
    if not history.empty:
        summary = (
            history.groupby("date", as_index=False)
            .apply(lambda g: pd.Series({"index_icc": float(g["vw_icc"].dropna().mean()) if g["vw_icc"].notna().any() else np.nan, "n_indices": int(g["universe"].nunique())}))
            .reset_index(drop=True)
            .sort_values("date")
        )
    else:
        summary = pd.DataFrame(columns=["date", "index_icc", "n_indices"])
    return history.reset_index(drop=True), latest.reset_index(drop=True), summary.reset_index(drop=True)


def archive_raw_snapshots(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for path, df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        group_name = "other_indices"
        if meta["universe"] == "usall":
            group_name = "usall"
        elif meta["universe"] == "sp500":
            group_name = "sp500"
        target_dir = RAW_DIR / group_name / meta["yyyymm"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)
        rows.append(
            {
                "group": group_name,
                "date": str(df["date"].iloc[0]),
                "yyyymm": meta["yyyymm"],
                "universe": meta["universe"],
                "n_firms": int(len(df)),
                "download_path": f"./data/downloads/raw/{group_name}/{meta['yyyymm']}/{path.name}",
                "filename": path.name,
            }
        )
    out = pd.DataFrame(rows).sort_values(["group", "date", "universe"]).reset_index(drop=True) if rows else pd.DataFrame(columns=["group", "date", "yyyymm", "universe", "n_firms", "download_path", "filename"])
    return out


def _zip_files(paths: list[Path], zip_path: Path, arc_prefix: str) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p.exists() and p.is_file():
                zf.write(p, arcname=f"{arc_prefix}/{p.name}")


def build_raw_download_catalog(raw_manifest: pd.DataFrame) -> dict[str, Any]:
    catalog: dict[str, Any] = {"tabs": {}}
    if raw_manifest.empty:
        return catalog
    for group_name in ["usall", "sp500", "other_indices"]:
        group_df = raw_manifest[raw_manifest["group"] == group_name].copy()
        if group_df.empty:
            catalog["tabs"][group_name] = {"years": []}
            continue
        years = []
        for year, year_df in group_df.groupby(group_df["yyyymm"].str[:4]):
            year_paths: list[Path] = []
            months = []
            for yyyymm, month_df in year_df.groupby("yyyymm"):
                month_paths = []
                day_rows = []
                for _, row in month_df.sort_values(["date", "universe"]).iterrows():
                    rel = row["download_path"].replace("./", "")
                    abs_path = REPO / rel
                    month_paths.append(abs_path)
                    year_paths.append(abs_path)
                    day_rows.append(
                        {
                            "date": row["date"],
                            "universe": row["universe"],
                            "n_firms": int(row["n_firms"]),
                            "path": row["download_path"],
                            "label": row["filename"],
                        }
                    )
                month_zip_rel = f"./data/downloads/raw_archives/{group_name}/{yyyymm}.zip"
                _zip_files(month_paths, DOCS_DATA_DIR / "downloads" / "raw_archives" / group_name / f"{yyyymm}.zip", f"{group_name}/{yyyymm}")
                months.append(
                    {
                        "yyyymm": yyyymm,
                        "download_all": month_zip_rel,
                        "days": sorted(day_rows, key=lambda x: x["date"], reverse=True),
                    }
                )
            year_zip_rel = f"./data/downloads/raw_archives/{group_name}/{year}.zip"
            _zip_files(year_paths, DOCS_DATA_DIR / "downloads" / "raw_archives" / group_name / f"{year}.zip", f"{group_name}/{year}")
            years.append(
                {
                    "year": year,
                    "download_all": year_zip_rel,
                    "months": sorted(months, key=lambda x: x["yyyymm"], reverse=True),
                }
            )
        catalog["tabs"][group_name] = {"years": sorted(years, key=lambda x: x["year"], reverse=True)}
    return catalog


def build_overview_rows(latest_dict: dict[str, dict[str, Any]], monthly_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    label_map = {
        "marketwide": "Marketwide",
        "value": "Value Premium",
        "industry": "Industry",
        "etf": "ETF",
        "country": "Country",
        "indices": "Indices",
    }
    metric_map = {
        "marketwide": "vw_icc",
        "value": "ivp_bm",
        "industry": "industry_icc_eq_sector",
        "etf": "etf_icc",
        "country": "country_icc",
        "indices": "index_icc",
    }
    for key, label in label_map.items():
        latest = latest_dict.get(key, {})
        monthly = monthly_dict.get(key, pd.DataFrame())
        metric_col = metric_map[key]
        month_vals = monthly[metric_col].tolist() if metric_col in monthly.columns else []
        month_labels = monthly["yyyymm"].tolist() if "yyyymm" in monthly.columns else []
        row = {
            "family": key,
            "label": label,
            "latest_date": latest.get("date"),
            "latest_value": latest.get(metric_col),
        }
        for i in range(3):
            row[f"m{i+1}_label"] = month_labels[i] if i < len(month_labels) else None
            row[f"m{i+1}_value"] = month_vals[i] if i < len(month_vals) else None
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

    base_paths = find_all_snapshots(args.universe)
    if not base_paths:
        raise FileNotFoundError(f"No {args.universe} snapshots found under data/YYYYMM/")
    base_valid = dedup_valid_snapshots(get_valid_snapshots(base_paths))
    if not base_valid:
        raise RuntimeError(f"No valid {args.universe} snapshots found.")

    latest_usall_path, latest_usall = base_valid[-1]
    latest_date = str(latest_usall["date"].dropna().iloc[0])

    all_paths = find_all_snapshots(None)
    all_valid = dedup_valid_snapshots(get_valid_snapshots(all_paths))
    all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]] = defaultdict(list)
    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        all_valid_by_universe[meta["universe"]].append((path, df))

    # Marketwide
    market_daily = build_market_history(base_valid)
    market_monthly = build_monthly_from_daily(market_daily)
    market_monthly_3 = last_n_months(market_monthly, 3)
    write_csv(market_daily, DOWNLOADS_DIR / "marketwide" / "daily_history.csv")
    write_csv(market_monthly, DOWNLOADS_DIR / "marketwide" / "monthly_history.csv")

    # Value
    value_daily = build_value_history(base_valid)
    value_monthly = build_monthly_from_daily(value_daily)
    value_monthly_3 = last_n_months(value_monthly, 3)
    write_csv(value_daily, DOWNLOADS_DIR / "value" / "daily_history.csv")
    write_csv(value_monthly, DOWNLOADS_DIR / "value" / "monthly_history.csv")

    # Industry
    industry_summary_daily, industry_latest = build_industry_daily(base_valid)
    industry_monthly = build_monthly_from_daily(industry_summary_daily)
    industry_monthly_3 = last_n_months(industry_monthly, 3)
    write_csv(industry_summary_daily, DOWNLOADS_DIR / "industry" / "daily_history.csv")
    write_csv(industry_monthly, DOWNLOADS_DIR / "industry" / "monthly_history.csv")
    write_csv(industry_latest, DOWNLOADS_DIR / "industry" / "latest_daily_table.csv")

    # ETF and Country: start archiving from now
    etf_cfg = read_csv_config(CONFIG_DIR / "etfs.csv", ["ticker", "label", "category"])
    country_cfg = read_csv_config(CONFIG_DIR / "country_etfs.csv", ["ticker", "label", "country"])

    etf_current, etf_members = build_holdings_proxy(etf_cfg, latest_usall, "etf")
    country_current, country_members = build_holdings_proxy(country_cfg, latest_usall, "country")

    etf_daily = upsert_history(etf_current, LIBRARY_DIR / "etf" / "history_daily.csv", ["date", "ticker"])
    country_daily = upsert_history(country_current, LIBRARY_DIR / "country" / "history_daily.csv", ["date", "country", "ticker"])
    if not etf_members.empty:
        upsert_history(etf_members, LIBRARY_DIR / "etf" / "members_daily.csv", ["date", "parent_ticker", "symbol"])
    if not country_members.empty:
        upsert_history(country_members, LIBRARY_DIR / "country" / "members_daily.csv", ["date", "parent_ticker", "symbol"])

    etf_summary_daily = build_history_summary(etf_daily, "etf")
    country_summary_daily = build_history_summary(country_daily, "country")
    etf_monthly = build_monthly_from_daily(etf_summary_daily)
    country_monthly = build_monthly_from_daily(country_summary_daily)
    etf_monthly_3 = last_n_months(etf_monthly, 3)
    country_monthly_3 = last_n_months(country_monthly, 3)
    write_csv(etf_daily, DOWNLOADS_DIR / "etf" / "daily_history.csv")
    write_csv(etf_monthly, DOWNLOADS_DIR / "etf" / "monthly_history.csv")
    write_csv(etf_current, DOWNLOADS_DIR / "etf" / "latest_daily_table.csv")
    write_csv(country_daily, DOWNLOADS_DIR / "country" / "daily_history.csv")
    write_csv(country_monthly, DOWNLOADS_DIR / "country" / "monthly_history.csv")
    write_csv(country_current, DOWNLOADS_DIR / "country" / "latest_daily_table.csv")

    # Indices
    index_daily, index_latest, index_summary_daily = build_index_outputs(all_valid_by_universe)
    index_monthly = build_monthly_from_daily(index_summary_daily)
    index_monthly_3 = last_n_months(index_monthly, 3)
    write_csv(index_daily, DOWNLOADS_DIR / "indices" / "daily_history.csv")
    write_csv(index_monthly, DOWNLOADS_DIR / "indices" / "monthly_history.csv")
    write_csv(index_latest, DOWNLOADS_DIR / "indices" / "latest_daily_table.csv")

    # Raw snapshot archives
    raw_manifest = archive_raw_snapshots(all_valid)
    write_csv(raw_manifest, DOWNLOADS_DIR / "raw" / "manifest.csv")
    raw_catalog = build_raw_download_catalog(raw_manifest)

    latest_market = market_daily.iloc[-1].to_dict() if not market_daily.empty else {}
    latest_value = value_daily.iloc[-1].to_dict() if not value_daily.empty else {}
    latest_industry = industry_summary_daily.iloc[-1].to_dict() if not industry_summary_daily.empty else {}
    latest_etf = etf_summary_daily.iloc[-1].to_dict() if not etf_summary_daily.empty else {}
    latest_country = country_summary_daily.iloc[-1].to_dict() if not country_summary_daily.empty else {}
    latest_indices = index_summary_daily.iloc[-1].to_dict() if not index_summary_daily.empty else {}

    overview_rows = build_overview_rows(
        {
            "marketwide": latest_market,
            "value": latest_value,
            "industry": latest_industry,
            "etf": latest_etf,
            "country": latest_country,
            "indices": latest_indices,
        },
        {
            "marketwide": market_monthly_3,
            "value": value_monthly_3,
            "industry": industry_monthly_3,
            "etf": etf_monthly_3,
            "country": country_monthly_3,
            "indices": index_monthly_3,
        },
    )
    write_csv(overview_rows, DOWNLOADS_DIR / "overview_last_3_months.csv")

    family_downloads = {
        "marketwide": [
            {"label": "Daily history (CSV)", "path": "./data/downloads/marketwide/daily_history.csv"},
            {"label": "Monthly history (CSV)", "path": "./data/downloads/marketwide/monthly_history.csv"},
        ],
        "value": [
            {"label": "Daily history (CSV)", "path": "./data/downloads/value/daily_history.csv"},
            {"label": "Monthly history (CSV)", "path": "./data/downloads/value/monthly_history.csv"},
        ],
        "industry": [
            {"label": "Daily summary history (CSV)", "path": "./data/downloads/industry/daily_history.csv"},
            {"label": "Monthly summary history (CSV)", "path": "./data/downloads/industry/monthly_history.csv"},
            {"label": "Latest daily table (CSV)", "path": "./data/downloads/industry/latest_daily_table.csv"},
        ],
        "etf": [
            {"label": "Daily summary history (CSV)", "path": "./data/downloads/etf/daily_history.csv"},
            {"label": "Monthly summary history (CSV)", "path": "./data/downloads/etf/monthly_history.csv"},
            {"label": "Latest daily table (CSV)", "path": "./data/downloads/etf/latest_daily_table.csv"},
        ],
        "country": [
            {"label": "Daily summary history (CSV)", "path": "./data/downloads/country/daily_history.csv"},
            {"label": "Monthly summary history (CSV)", "path": "./data/downloads/country/monthly_history.csv"},
            {"label": "Latest daily table (CSV)", "path": "./data/downloads/country/latest_daily_table.csv"},
        ],
        "indices": [
            {"label": "Daily history by index (CSV)", "path": "./data/downloads/indices/daily_history.csv"},
            {"label": "Monthly summary history (CSV)", "path": "./data/downloads/indices/monthly_history.csv"},
            {"label": "Latest daily table (CSV)", "path": "./data/downloads/indices/latest_daily_table.csv"},
        ],
        "raw": [
            {"label": "Raw manifest (CSV)", "path": "./data/downloads/raw/manifest.csv"},
        ],
    }

    # JSON payloads for pages
    write_json(
        DOCS_DATA_DIR / "overview.json",
        {
            "asof_date": latest_date,
            "source_file": str(latest_usall_path.relative_to(REPO)),
            "overview_rows": overview_rows.to_dict(orient="records"),
            "family_downloads": family_downloads,
        },
    )
    write_json(
        DOCS_DATA_DIR / "marketwide.json",
        {
            "asof_date": latest_date,
            "latest_daily": latest_market,
            "daily_history": market_daily.to_dict(orient="records"),
            "monthly_history": market_monthly.to_dict(orient="records"),
            "last_three_months": market_monthly_3.to_dict(orient="records"),
            "downloads": family_downloads["marketwide"],
        },
    )
    write_json(
        DOCS_DATA_DIR / "value.json",
        {
            "asof_date": latest_date,
            "latest_daily": latest_value,
            "daily_history": value_daily.to_dict(orient="records"),
            "monthly_history": value_monthly.to_dict(orient="records"),
            "last_three_months": value_monthly_3.to_dict(orient="records"),
            "downloads": family_downloads["value"],
        },
    )
    write_json(
        DOCS_DATA_DIR / "industry.json",
        {
            "asof_date": latest_date,
            "latest_summary": latest_industry,
            "latest_table": industry_latest.to_dict(orient="records"),
            "daily_history": industry_summary_daily.to_dict(orient="records"),
            "monthly_history": industry_monthly.to_dict(orient="records"),
            "last_three_months": industry_monthly_3.to_dict(orient="records"),
            "downloads": family_downloads["industry"],
        },
    )
    write_json(
        DOCS_DATA_DIR / "etf.json",
        {
            "asof_date": latest_date,
            "latest_summary": latest_etf,
            "latest_table": etf_current.to_dict(orient="records"),
            "daily_history": etf_summary_daily.to_dict(orient="records"),
            "monthly_history": etf_monthly.to_dict(orient="records"),
            "last_three_months": etf_monthly_3.to_dict(orient="records"),
            "downloads": family_downloads["etf"],
            "note": "ETF ICC history starts when this archive pipeline began. Historical backfill before archive start is not guaranteed.",
        },
    )
    write_json(
        DOCS_DATA_DIR / "country.json",
        {
            "asof_date": latest_date,
            "latest_summary": latest_country,
            "latest_table": country_current.to_dict(orient="records"),
            "daily_history": country_summary_daily.to_dict(orient="records"),
            "monthly_history": country_monthly.to_dict(orient="records"),
            "last_three_months": country_monthly_3.to_dict(orient="records"),
            "downloads": family_downloads["country"],
            "note": "Country ICC history starts when this archive pipeline began. Country ICC is based on country ETF proxies using online holdings data.",
        },
    )
    write_json(
        DOCS_DATA_DIR / "indices.json",
        {
            "asof_date": latest_date,
            "latest_summary": latest_indices,
            "latest_table": index_latest.to_dict(orient="records"),
            "daily_history": index_daily.to_dict(orient="records"),
            "monthly_history": index_monthly.to_dict(orient="records"),
            "last_three_months": index_monthly_3.to_dict(orient="records"),
            "downloads": family_downloads["indices"],
        },
    )
    write_json(
        DOCS_DATA_DIR / "downloads_catalog.json",
        {
            "asof_date": latest_date,
            "family_downloads": family_downloads,
            "raw_catalog": raw_catalog,
        },
    )

    print(f"[build_docs_data] latest valid {args.universe} date = {latest_date}")
    print(f"[build_docs_data] total valid snapshots = {len(all_valid)}")
    print("[build_docs_data] docs/data and downloads written")


if __name__ == "__main__":
    main()
