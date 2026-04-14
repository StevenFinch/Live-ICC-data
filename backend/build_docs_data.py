from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.errors import EmptyDataError


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DATA_DIR = REPO / "docs" / "data"
DOWNLOAD_DIR = DOCS_DATA_DIR / "downloads"
RAW_DOWNLOAD_DIR = DOWNLOAD_DIR / "raw"
CONFIG_DIR = REPO / "config"

SNAPSHOT_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)

INDEX_UNIVERSES = [
    "sp500",
    "sp100",
    "dow30",
    "ndx100",
    "sp400",
    "sp600",
    "sp1500",
    "rut1000",
]


def parse_snapshot_meta(path: Path) -> dict | None:
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


def safe_float(x):
    if pd.isna(x):
        return None
    try:
        val = float(x)
        if not np.isfinite(val):
            return None
        return val
    except Exception:
        return None


def safe_int(x):
    if pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        return None


def json_safe(obj):
    """
    Recursively convert NaN/Inf into JSON-safe None.
    """
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_payload = json_safe(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_payload, f, indent=2, allow_nan=False)


def find_all_snapshots(universe: str | None = None) -> list[Path]:
    files = []
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
        raise FileNotFoundError(f"Snapshot not found: {path}")

    if path.stat().st_size == 0:
        raise EmptyDataError(f"Empty file (0 bytes): {path}")

    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        raise EmptyDataError(f"Empty CSV content: {path}")

    if df is None or df.empty or len(df.columns) == 0:
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
    valid = []
    for path in paths:
        df = try_load_snapshot(path)
        if df is None:
            continue
        df = clean_df(df)
        if df.empty:
            print(f"[build_docs_data] skipping cleaned-empty snapshot: {path}")
            continue
        valid.append((path, df))
    return valid


def read_csv_config(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        print(f"[build_docs_data] config missing, using empty config: {path}")
        return pd.DataFrame(columns=expected_cols)

    df = pd.read_csv(path)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols].copy()


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
            print(f"[build_docs_data] skipping value sort on empty bm sample: {path}")
            continue

        q_lo = float(df["ICC"].quantile(0.005))
        q_hi = float(df["ICC"].quantile(0.995))
        df = df[(df["ICC"] >= q_lo) & (df["ICC"] <= q_hi)].copy()

        if df.empty:
            print(f"[build_docs_data] skipping value sort after ICC trim: {path}")
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
            .apply(
                lambda g: pd.Series(
                    {
                        "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
                        "n_firms": int(len(g)),
                    }
                )
            )
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


def build_latest_industry(latest_usall: pd.DataFrame) -> pd.DataFrame:
    df = clean_df(latest_usall)
    df = df[df["sector"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=["sector", "vw_icc", "ew_icc", "n_firms", "total_mktcap"])

    out = (
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
        .sort_values(["total_mktcap", "sector"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return out


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
    symbol_col, weight_col = None, None

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
        raise ValueError("could not identify weight column in top_holdings")

    out = pd.DataFrame(
        {
            "symbol": h[symbol_col].astype(str).str.upper().str.strip(),
            "raw_weight": pd.to_numeric(h[weight_col], errors="coerce"),
        }
    )

    out = out[out["symbol"].notna() & out["raw_weight"].notna()].copy()
    out["weight"] = out["raw_weight"] / 100.0 if out["raw_weight"].max() > 1.5 else out["raw_weight"]
    out = out[out["weight"] > 0].copy()

    if out.empty:
        raise ValueError("no positive weights in top_holdings")

    out["weight"] = out["weight"] / out["weight"].sum()
    return out[["symbol", "weight"]]


def build_holdings_proxy(rows_cfg: pd.DataFrame, latest_usall: pd.DataFrame, kind: str) -> pd.DataFrame:
    base = clean_df(latest_usall)[["ticker", "ICC", "mktcap", "sector", "name"]].copy()
    base["ticker"] = base["ticker"].astype(str).str.upper().str.strip()

    out_rows = []

    for _, cfg in rows_cfg.iterrows():
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

            if matched.sum() == 0:
                out_rows.append(
                    {
                        "ticker": ticker,
                        "label": label,
                        "category": category,
                        "country": country,
                        "vw_icc": np.nan,
                        "coverage_weight": 0.0,
                        "n_holdings": int(len(h)),
                        "n_matched": 0,
                        "holding_source": "yfinance_top_holdings",
                        "status": "no_matched_holdings",
                    }
                )
                continue

            icc = weighted_mean(merged.loc[matched, "ICC"], merged.loc[matched, "weight"])

            out_rows.append(
                {
                    "ticker": ticker,
                    "label": label,
                    "category": category,
                    "country": country,
                    "vw_icc": icc,
                    "coverage_weight": coverage,
                    "n_holdings": int(len(h)),
                    "n_matched": int(matched.sum()),
                    "holding_source": "yfinance_top_holdings",
                    "status": "ok" if coverage > 0 else "low_coverage",
                }
            )
        except Exception as e:
            out_rows.append(
                {
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

    df = pd.DataFrame(out_rows)

    if df.empty:
        if kind == "country":
            return pd.DataFrame(
                columns=[
                    "country",
                    "ticker",
                    "label",
                    "vw_icc",
                    "coverage_weight",
                    "n_holdings",
                    "n_matched",
                    "holding_source",
                    "status",
                ]
            )
        return pd.DataFrame(
            columns=[
                "ticker",
                "label",
                "category",
                "vw_icc",
                "coverage_weight",
                "n_holdings",
                "n_matched",
                "holding_source",
                "status",
            ]
        )

    if kind == "country":
        keep_cols = [
            "country",
            "ticker",
            "label",
            "vw_icc",
            "coverage_weight",
            "n_holdings",
            "n_matched",
            "holding_source",
            "status",
        ]
        return df[keep_cols].sort_values("country").reset_index(drop=True)

    keep_cols = [
        "ticker",
        "label",
        "category",
        "vw_icc",
        "coverage_weight",
        "n_holdings",
        "n_matched",
        "holding_source",
        "status",
    ]
    return df[keep_cols].sort_values(["category", "ticker"]).reset_index(drop=True)


def build_index_outputs(all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_rows = []
    latest_rows = []

    for universe in INDEX_UNIVERSES:
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

    if history_rows:
        hist_all = pd.concat(history_rows, ignore_index=True)
        hist_all = hist_all.sort_values(["universe", "date"]).reset_index(drop=True)
    else:
        hist_all = pd.DataFrame(columns=["universe", "date", "vw_icc", "ew_icc", "n_firms", "source_file"])

    latest_df = pd.DataFrame(latest_rows)
    if latest_df.empty:
        latest_df = pd.DataFrame(columns=["universe", "date", "vw_icc", "ew_icc", "n_firms", "source_file"])
    else:
        latest_df = latest_df.sort_values("universe").reset_index(drop=True)

    return hist_all, latest_df


def copy_raw_snapshots(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> None:
    RAW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for path, _df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue

        target_dir = RAW_DOWNLOAD_DIR / meta["yyyymm"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)


def build_year_month_manifest(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, _df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        rows.append(
            {
                "yyyymm": meta["yyyymm"],
                "universe": meta["universe"],
                "filename": path.name,
                "download_folder": f"./data/downloads/raw/{meta['yyyymm']}/",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["yyyymm", "n_files", "universes", "download_folder"])

    tmp = pd.DataFrame(rows)
    out = (
        tmp.groupby("yyyymm", dropna=False)
        .agg(
            n_files=("filename", "count"),
            universes=("universe", lambda x: ", ".join(sorted(set(map(str, x))))),
            download_folder=("download_folder", "first"),
        )
        .reset_index()
        .sort_values("yyyymm")
        .reset_index(drop=True)
    )
    return out


def build_all_snapshot_files_manifest(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        date_value = str(df["date"].dropna().iloc[0]) if df["date"].notna().any() else None
        rows.append(
            {
                "date": date_value,
                "yyyymm": meta["yyyymm"],
                "universe": meta["universe"],
                "n_firms": int(len(df)),
                "source_file": str(path.relative_to(REPO)),
                "download_path": f"./data/downloads/raw/{meta['yyyymm']}/{path.name}",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "yyyymm", "universe", "n_firms", "source_file", "download_path"])

    out = pd.DataFrame(rows).sort_values(["date", "universe"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Base universe (normally usall)
    base_paths = find_all_snapshots(args.universe)
    if not base_paths:
        raise FileNotFoundError(f"No {args.universe} snapshot found under data/YYYYMM/")

    base_valid = get_valid_snapshots(base_paths)
    if not base_valid:
        raise RuntimeError(f"Found {args.universe} snapshot files, but none of them are valid non-empty CSVs.")

    latest_usall_path, latest_usall = base_valid[-1]
    latest_date = str(latest_usall["date"].dropna().iloc[0])

    # All universes
    all_paths = find_all_snapshots(None)
    all_valid = get_valid_snapshots(all_paths)

    all_valid_by_universe: dict[str, list[tuple[Path, pd.DataFrame]]] = {}
    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        all_valid_by_universe.setdefault(meta["universe"], []).append((path, df))

    market_history = build_market_history(base_valid)
    value_history = build_value_history(base_valid)
    industry_latest = build_latest_industry(latest_usall)

    etf_cfg = read_csv_config(CONFIG_DIR / "etfs.csv", ["ticker", "label", "category"])
    country_cfg = read_csv_config(CONFIG_DIR / "country_etfs.csv", ["ticker", "label", "country"])

    etf_latest = build_holdings_proxy(etf_cfg, latest_usall, kind="etf")
    country_latest = build_holdings_proxy(country_cfg, latest_usall, kind="country")

    index_history, index_latest = build_index_outputs(all_valid_by_universe)

    year_month_manifest = build_year_month_manifest(all_valid)
    all_snapshot_files = build_all_snapshot_files_manifest(all_valid)

    copy_raw_snapshots(all_valid)

    # CSV downloads
    market_history.to_csv(DOWNLOAD_DIR / "market_icc_history.csv", index=False)
    value_history.to_csv(DOWNLOAD_DIR / "value_icc_bm_history.csv", index=False)
    industry_latest.to_csv(DOWNLOAD_DIR / "industry_icc_latest.csv", index=False)
    etf_latest.to_csv(DOWNLOAD_DIR / "etf_icc_latest.csv", index=False)
    country_latest.to_csv(DOWNLOAD_DIR / "country_icc_latest.csv", index=False)
    index_history.to_csv(DOWNLOAD_DIR / "index_icc_history.csv", index=False)
    index_latest.to_csv(DOWNLOAD_DIR / "index_icc_latest.csv", index=False)
    year_month_manifest.to_csv(DOWNLOAD_DIR / "year_month_manifest.csv", index=False)
    all_snapshot_files.to_csv(DOWNLOAD_DIR / "all_snapshot_files.csv", index=False)

    latest_market = market_history.iloc[-1].to_dict() if not market_history.empty else {}
    latest_value = value_history.iloc[-1].to_dict() if not value_history.empty else {}

    write_json(
        DOCS_DATA_DIR / "overview.json",
        {
            "asof_date": latest_date,
            "source_file": str(latest_usall_path.relative_to(REPO)),
            "cards": {
                "market_vw_icc": safe_float(latest_market.get("vw_icc")),
                "value_icc": safe_float(latest_value.get("value_icc")),
                "growth_icc": safe_float(latest_value.get("growth_icc")),
                "ivp_bm": safe_float(latest_value.get("ivp_bm")),
            },
            "downloads": [
                {"label": "Market ICC history (CSV)", "path": "./data/downloads/market_icc_history.csv"},
                {"label": "Value / Growth / IVP history (CSV)", "path": "./data/downloads/value_icc_bm_history.csv"},
                {"label": "Industry ICC latest (CSV)", "path": "./data/downloads/industry_icc_latest.csv"},
                {"label": "ETF ICC latest (CSV)", "path": "./data/downloads/etf_icc_latest.csv"},
                {"label": "Country ICC latest (CSV)", "path": "./data/downloads/country_icc_latest.csv"},
                {"label": "Index ICC history (CSV)", "path": "./data/downloads/index_icc_history.csv"},
                {"label": "Index ICC latest (CSV)", "path": "./data/downloads/index_icc_latest.csv"},
                {"label": "Year-Month manifest (CSV)", "path": "./data/downloads/year_month_manifest.csv"},
                {"label": "All snapshot files manifest (CSV)", "path": "./data/downloads/all_snapshot_files.csv"},
            ],
        },
    )

    write_json(
        DOCS_DATA_DIR / "market_icc.json",
        {"asof_date": latest_date, "history": market_history.to_dict(orient="records")},
    )

    write_json(
        DOCS_DATA_DIR / "value_icc_bm.json",
        {"asof_date": latest_date, "history": value_history.to_dict(orient="records")},
    )

    write_json(
        DOCS_DATA_DIR / "industry_icc.json",
        {"asof_date": latest_date, "latest": industry_latest.to_dict(orient="records")},
    )

    write_json(
        DOCS_DATA_DIR / "etf_icc.json",
        {"asof_date": latest_date, "latest": etf_latest.to_dict(orient="records")},
    )

    write_json(
        DOCS_DATA_DIR / "country_icc.json",
        {"asof_date": latest_date, "latest": country_latest.to_dict(orient="records")},
    )

    write_json(
        DOCS_DATA_DIR / "index_icc.json",
        {
            "asof_date": latest_date,
            "history": index_history.to_dict(orient="records"),
            "latest": index_latest.to_dict(orient="records"),
        },
    )

    write_json(
        DOCS_DATA_DIR / "year_month_manifest.json",
        {"asof_date": latest_date, "rows": year_month_manifest.to_dict(orient="records")},
    )

    write_json(
        DOCS_DATA_DIR / "all_snapshot_files.json",
        {"asof_date": latest_date, "rows": all_snapshot_files.to_dict(orient="records")},
    )

    print(f"[build_docs_data] total snapshot files found = {len(all_paths)}")
    print(f"[build_docs_data] total valid snapshots used = {len(all_valid)}")
    print(f"[build_docs_data] latest valid {args.universe} asof_date = {latest_date}")
    print("[build_docs_data] wrote docs/data and docs/data/downloads")


if __name__ == "__main__":
    main()
