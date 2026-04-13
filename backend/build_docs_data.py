from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.errors import EmptyDataError

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DATA_DIR = REPO / "docs" / "data"
DOWNLOAD_DIR = DOCS_DATA_DIR / "downloads"
CONFIG_DIR = REPO / "config"

SNAPSHOT_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)


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


def find_all_snapshots(universe: str) -> list[Path]:
    files = []
    for p in DATA_DIR.glob("*/*.csv"):
        m = SNAPSHOT_RE.match(p.name)
        if not m:
            continue
        if m.group("universe") != universe:
            continue

        yyyymmdd = int(f"{m.group('year')}{m.group('mmdd')}")
        rerun = int(m.group("rerun") or 0)
        files.append(((yyyymmdd, rerun, p.name), p))

    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def load_snapshot(path: Path) -> pd.DataFrame:
    """
    Strict loader.
    Raise on empty / malformed / missing-required-columns files.
    """
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
    """
    Safe loader: return None if file is empty / malformed / unusable.
    """
    try:
        df = load_snapshot(path)
        return df
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


def read_csv_config(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return pd.read_csv(path)


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
        ticker = str(cfg["ticker"]).upper().strip()
        label = str(cfg.get("label", ticker))
        category = str(cfg.get("category", "")) if "category" in cfg.index else ""
        country = str(cfg.get("country", "")) if "country" in cfg.index else ""

        try:
            h = get_top_holdings(ticker)
            merged = h.merge(base, left_on="symbol", right_on="ticker", how="left")

            matched = merged["ICC"].notna()
            coverage = float(merged.loc[matched, "weight"].sum())

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


def safe_val(x):
    if pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return x


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = find_all_snapshots(args.universe)
    if not all_paths:
        raise FileNotFoundError("No usall snapshot found under data/YYYYMM/")

    valid_snapshots = get_valid_snapshots(all_paths)
    if not valid_snapshots:
        raise RuntimeError("Found snapshot files, but none of them are valid non-empty CSVs.")

    latest_usall_path, latest_usall = valid_snapshots[-1]
    latest_date = str(latest_usall["date"].dropna().iloc[0])

    market_history = build_market_history(valid_snapshots)
    value_history = build_value_history(valid_snapshots)
    industry_latest = build_latest_industry(latest_usall)

    etf_cfg = read_csv_config(CONFIG_DIR / "etfs.csv")
    country_cfg = read_csv_config(CONFIG_DIR / "country_etfs.csv")

    etf_latest = build_holdings_proxy(etf_cfg, latest_usall, kind="etf")
    country_latest = build_holdings_proxy(country_cfg, latest_usall, kind="country")

    market_history.to_csv(DOWNLOAD_DIR / "market_icc_history.csv", index=False)
    value_history.to_csv(DOWNLOAD_DIR / "value_icc_bm_history.csv", index=False)
    industry_latest.to_csv(DOWNLOAD_DIR / "industry_icc_latest.csv", index=False)
    etf_latest.to_csv(DOWNLOAD_DIR / "etf_icc_latest.csv", index=False)
    country_latest.to_csv(DOWNLOAD_DIR / "country_icc_latest.csv", index=False)

    latest_market = market_history.iloc[-1].to_dict() if not market_history.empty else {}
    latest_value = value_history.iloc[-1].to_dict() if not value_history.empty else {}

    write_json(
        DOCS_DATA_DIR / "overview.json",
        {
            "asof_date": latest_date,
            "source_file": str(latest_usall_path.relative_to(REPO)),
            "cards": {
                "market_vw_icc": safe_val(latest_market.get("vw_icc")),
                "value_icc": safe_val(latest_value.get("value_icc")),
                "growth_icc": safe_val(latest_value.get("growth_icc")),
                "ivp_bm": safe_val(latest_value.get("ivp_bm")),
            },
            "downloads": [
                {"label": "Market ICC history (CSV)", "path": "./data/downloads/market_icc_history.csv"},
                {"label": "Value / Growth / IVP history (CSV)", "path": "./data/downloads/value_icc_bm_history.csv"},
                {"label": "Industry ICC latest (CSV)", "path": "./data/downloads/industry_icc_latest.csv"},
                {"label": "ETF ICC latest (CSV)", "path": "./data/downloads/etf_icc_latest.csv"},
                {"label": "Country ICC latest (CSV)", "path": "./data/downloads/country_icc_latest.csv"},
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

    print(f"[build_docs_data] total snapshot files found = {len(all_paths)}")
    print(f"[build_docs_data] valid snapshots used = {len(valid_snapshots)}")
    print(f"[build_docs_data] latest valid asof_date = {latest_date}")
    print("[build_docs_data] wrote docs/data and docs/data/downloads")


if __name__ == "__main__":
    main()
