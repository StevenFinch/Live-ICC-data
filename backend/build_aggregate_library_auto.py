
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from library_common import (
    DATA,
    DOCS_DATA,
    REPO,
    find_latest_snapshot,
    load_snapshot,
    publish_df,
    quantile_trim,
    standardize_ticker,
    to_json,
    upsert_history,
    weighted_mean,
)

CONFIG = REPO / "config"
ETF_LIST = CONFIG / "etfs.csv"
COUNTRY_ETF_LIST = CONFIG / "country_etfs.csv"


def safe_float(x):
    if pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _read_config(path: Path, required: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return df


def build_market_icc(df: pd.DataFrame) -> dict:
    used = df[df["ICC"].notna() & df["mktcap"].notna() & (df["mktcap"] > 0)].copy()
    date_value = str(used["date"].dropna().iloc[0])
    row = {
        "date": date_value,
        "id": "market",
        "label": "US Market",
        "n_firms": int(len(used)),
        "vw_icc": weighted_mean(used["ICC"], used["mktcap"]),
        "ew_icc": float(used["ICC"].mean()),
        "source_file": str(used["source_file"].iloc[0]),
    }
    out_dir = DATA / "library" / "market_icc"
    hist = upsert_history(out_dir / "history.csv", row)
    publish_df(pd.DataFrame([row]), out_dir / "latest.csv")
    publish_df(hist, out_dir / "history.csv")
    to_json(
        {
            "asof_date": row["date"],
            "latest": row,
            "history": hist.sort_values("date").to_dict(orient="records"),
        },
        out_dir / "latest.json",
    )
    return row


def build_value_icc_bm(df: pd.DataFrame) -> dict:
    used = df.copy()
    used = used[
        used["ICC"].notna()
        & used["mktcap"].notna()
        & used["bm"].notna()
        & np.isfinite(used["ICC"])
        & np.isfinite(used["mktcap"])
        & np.isfinite(used["bm"])
        & (used["mktcap"] > 0)
        & (used["bm"] > 0)
    ].copy()
    used = quantile_trim(used, "ICC", q=0.005)

    size_median = float(used["mktcap"].median())
    bm30 = float(used["bm"].quantile(0.30))
    bm70 = float(used["bm"].quantile(0.70))

    used["size"] = np.where(used["mktcap"] <= size_median, "S", "B")
    used["bm_bucket"] = pd.cut(
        used["bm"],
        bins=[-np.inf, bm30, bm70, np.inf],
        labels=["L", "M", "H"],
        include_lowest=True,
    ).astype(str)
    used["portfolio"] = used["size"] + "/" + used["bm_bucket"]

    ports = []
    for p in ["S/L", "S/M", "S/H", "B/L", "B/M", "B/H"]:
        g = used.loc[used["portfolio"] == p].copy()
        ports.append(
            {
                "portfolio": p,
                "n_firms": int(len(g)),
                "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
                "ew_icc": float(g["ICC"].mean()) if len(g) else np.nan,
                "vw_bm": weighted_mean(g["bm"], g["mktcap"]),
            }
        )
    pt = pd.DataFrame(ports).set_index("portfolio")

    value_icc = np.nanmean([pt.loc["S/H", "vw_icc"], pt.loc["B/H", "vw_icc"]])
    growth_icc = np.nanmean([pt.loc["S/L", "vw_icc"], pt.loc["B/L", "vw_icc"]])
    ivp_bm = value_icc - growth_icc

    row = {
        "date": str(used["date"].dropna().iloc[0]),
        "id": "ivp_bm",
        "label": "Value minus Growth ICC (B/M)",
        "n_firms": int(len(used)),
        "market_icc": weighted_mean(used["ICC"], used["mktcap"]),
        "value_icc": float(value_icc),
        "growth_icc": float(growth_icc),
        "ivp_bm": float(ivp_bm),
        "size_median": size_median,
        "bm30": bm30,
        "bm70": bm70,
        "source_file": str(used["source_file"].iloc[0]),
    }

    out_dir = DATA / "library" / "value_icc_bm"
    hist = upsert_history(out_dir / "history.csv", row)
    publish_df(pd.DataFrame([row]), out_dir / "latest.csv")
    publish_df(pt.reset_index(), out_dir / "latest_portfolios.csv")
    publish_df(hist, out_dir / "history.csv")
    to_json(
        {
            "asof_date": row["date"],
            "latest": row,
            "portfolios": pt.reset_index().to_dict(orient="records"),
            "history": hist.sort_values("date").to_dict(orient="records"),
        },
        out_dir / "latest.json",
    )
    return row


def build_industry_icc(df: pd.DataFrame) -> pd.DataFrame:
    used = df[df["ICC"].notna() & df["mktcap"].notna() & (df["mktcap"] > 0) & df["sector"].notna()].copy()
    rows = []
    for sec, g in used.groupby("sector", dropna=True):
        rows.append(
            {
                "date": str(g["date"].dropna().iloc[0]),
                "id": f"industry::{sec}",
                "sector": sec,
                "label": sec,
                "n_firms": int(len(g)),
                "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
                "ew_icc": float(g["ICC"].mean()),
                "total_mktcap": float(g["mktcap"].sum()),
                "source_file": str(g["source_file"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows).sort_values("sector").reset_index(drop=True)
    out_dir = DATA / "library" / "industry_icc"

    latest_path = out_dir / "latest.csv"
    hist_path = out_dir / "history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        today = str(out["date"].iloc[0])
        hist = hist.loc[hist["date"].astype(str) != today].copy()
        hist = pd.concat([hist, out], ignore_index=True)
    else:
        hist = out.copy()
    hist = hist.sort_values(["date", "sector"]).reset_index(drop=True)

    publish_df(out, latest_path)
    publish_df(hist, hist_path)
    to_json(
        {
            "asof_date": str(out["date"].iloc[0]) if len(out) else None,
            "latest": out.to_dict(orient="records"),
        },
        out_dir / "latest.json",
    )
    return out


def _coerce_holding_table(raw) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame(columns=["holding_ticker", "holding_name", "holding_weight"])
    if isinstance(raw, pd.Series):
        df = raw.to_frame().T
    else:
        df = pd.DataFrame(raw).copy()

    if df.empty:
        return pd.DataFrame(columns=["holding_ticker", "holding_name", "holding_weight"])

    cols = {str(c).strip(): c for c in df.columns}
    out = pd.DataFrame()

    ticker_candidates = ["Symbol", "symbol", "Holding", "holding", "Ticker", "ticker"]
    name_candidates = ["Name", "name", "Holding Name", "holding_name"]
    weight_candidates = ["Holding Percent", "holdingPercent", "holding_percent", "% Assets", "Percent", "percent"]

    tick_col = next((cols[c] for c in ticker_candidates if c in cols), None)
    name_col = next((cols[c] for c in name_candidates if c in cols), None)
    w_col = next((cols[c] for c in weight_candidates if c in cols), None)

    if tick_col is None:
        # try first column if it looks like strings/tickers
        tick_col = df.columns[0]

    out["holding_ticker"] = df[tick_col].astype(str).map(standardize_ticker)
    out["holding_name"] = df[name_col].astype(str) if name_col is not None else out["holding_ticker"]

    if w_col is not None:
        w = pd.to_numeric(df[w_col], errors="coerce")
    else:
        # equal-weight fallback if Yahoo did not provide weights
        w = pd.Series(np.ones(len(df)), index=df.index, dtype=float)

    # Normalize percentages whether 0-1 or 0-100.
    if w.dropna().max() is not np.nan and len(w.dropna()) > 0:
        max_w = float(w.dropna().max())
        if max_w > 1.5:
            w = w / 100.0

    out["holding_weight"] = pd.to_numeric(w, errors="coerce")
    out = out[out["holding_ticker"].ne("") & out["holding_weight"].notna() & (out["holding_weight"] > 0)].copy()
    return out.reset_index(drop=True)


def _load_manual_holdings_if_any(etf: str) -> Optional[pd.DataFrame]:
    folder = CONFIG / "etf_holdings"
    candidates = [
        folder / f"{etf}.csv",
        folder / f"{etf.lower()}.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [str(c).strip() for c in df.columns]
            if {"holding_ticker", "holding_weight"} <= set(df.columns):
                df["holding_ticker"] = df["holding_ticker"].map(standardize_ticker)
                df["holding_weight"] = pd.to_numeric(df["holding_weight"], errors="coerce")
                return df[["holding_ticker", "holding_weight"]].copy()
    return None


def get_etf_holdings_auto(etf: str) -> pd.DataFrame:
    # Priority 1: optional manual file if user wants to override.
    manual = _load_manual_holdings_if_any(etf)
    if manual is not None and len(manual):
        out = manual.copy()
        out["holding_name"] = out["holding_ticker"]
        out["holding_source"] = "manual_file"
        return out

    # Priority 2: yfinance automatic holdings.
    try:
        t = yf.Ticker(etf)
        raw = getattr(getattr(t, "funds_data", None), "top_holdings", None)
        df = _coerce_holding_table(raw)
        if len(df):
            df["holding_source"] = "yfinance_top_holdings"
            return df
    except Exception:
        pass

    return pd.DataFrame(columns=["holding_ticker", "holding_name", "holding_weight", "holding_source"])


def _calc_etf_icc_one(etf_row: pd.Series, snapshot: pd.DataFrame) -> dict:
    etf = standardize_ticker(etf_row["ticker"])
    label = str(etf_row.get("label", etf))
    category = str(etf_row.get("category", "ETF"))
    holdings = get_etf_holdings_auto(etf)

    if holdings.empty:
        return {
            "date": str(snapshot["date"].dropna().iloc[0]),
            "id": f"etf::{etf}",
            "ticker": etf,
            "label": label,
            "category": category,
            "n_holdings": 0,
            "n_matched": 0,
            "coverage_weight": np.nan,
            "matched_weight": np.nan,
            "vw_icc": np.nan,
            "holding_source": None,
            "status": "no_holdings",
        }

    snap = snapshot[["ticker", "ICC", "mktcap", "name", "sector"]].copy()
    merged = holdings.merge(snap, left_on="holding_ticker", right_on="ticker", how="left")

    matched = merged[merged["ICC"].notna()].copy()
    raw_coverage = float(matched["holding_weight"].sum()) if len(matched) else 0.0

    if len(matched):
        matched["norm_weight"] = matched["holding_weight"] / matched["holding_weight"].sum()
        vw_icc = float(np.average(matched["ICC"], weights=matched["norm_weight"]))
    else:
        vw_icc = np.nan

    top_constituents = (
        matched.sort_values("holding_weight", ascending=False)[
            ["holding_ticker", "holding_name", "holding_weight", "ICC", "sector"]
        ]
        .head(10)
        .to_dict(orient="records")
    )

    return {
        "date": str(snapshot["date"].dropna().iloc[0]),
        "id": f"etf::{etf}",
        "ticker": etf,
        "label": label,
        "category": category,
        "n_holdings": int(len(holdings)),
        "n_matched": int(len(matched)),
        "coverage_weight": raw_coverage,
        "matched_weight": 1.0 if len(matched) else np.nan,
        "vw_icc": vw_icc,
        "holding_source": str(holdings["holding_source"].iloc[0]) if len(holdings) else None,
        "status": "ok" if len(matched) else "no_snapshot_match",
        "top_constituents": top_constituents,
    }


def build_etf_icc(snapshot: pd.DataFrame) -> pd.DataFrame:
    cfg = _read_config(ETF_LIST, ["ticker", "label", "category"])
    rows = [_calc_etf_icc_one(r, snapshot) for _, r in cfg.iterrows()]
    out = pd.DataFrame(rows).sort_values(["category", "ticker"]).reset_index(drop=True)

    out_dir = DATA / "library" / "etf_icc"
    today = str(snapshot["date"].dropna().iloc[0])
    hist_path = out_dir / "history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist = hist.loc[hist["date"].astype(str) != today].copy()
        hist = pd.concat([hist, out.drop(columns=["top_constituents"], errors="ignore")], ignore_index=True)
    else:
        hist = out.drop(columns=["top_constituents"], errors="ignore").copy()

    publish_df(out.drop(columns=["top_constituents"], errors="ignore"), out_dir / "latest.csv")
    publish_df(hist, hist_path)
    to_json({"asof_date": today, "latest": out.to_dict(orient="records")}, out_dir / "latest.json")
    return out


def build_country_icc(snapshot: pd.DataFrame) -> pd.DataFrame:
    cfg = _read_config(COUNTRY_ETF_LIST, ["country", "ticker", "label"])
    rows = []
    for _, r in cfg.iterrows():
        base = _calc_etf_icc_one(pd.Series({"ticker": r["ticker"], "label": r["label"], "category": "Country"}), snapshot)
        base["country"] = r["country"]
        base["id"] = f"country::{r['country']}"
        rows.append(base)
    out = pd.DataFrame(rows).sort_values("country").reset_index(drop=True)

    out_dir = DATA / "library" / "country_icc"
    today = str(snapshot["date"].dropna().iloc[0])
    hist_path = out_dir / "history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist = hist.loc[hist["date"].astype(str) != today].copy()
        hist = pd.concat([hist, out.drop(columns=["top_constituents"], errors="ignore")], ignore_index=True)
    else:
        hist = out.drop(columns=["top_constituents"], errors="ignore").copy()

    publish_df(out.drop(columns=["top_constituents"], errors="ignore"), out_dir / "latest.csv")
    publish_df(hist, hist_path)
    to_json({"asof_date": today, "latest": out.to_dict(orient="records")}, out_dir / "latest.json")
    return out


def build_docs_payload(
    market_row: dict,
    value_row: dict,
    industry_df: pd.DataFrame,
    etf_df: pd.DataFrame,
    country_df: pd.DataFrame,
) -> None:
    DOCS_DATA.mkdir(parents=True, exist_ok=True)

    overview = {
        "asof_date": market_row["date"],
        "cards": {
            "market_vw_icc": safe_float(market_row["vw_icc"]),
            "value_icc": safe_float(value_row["value_icc"]),
            "growth_icc": safe_float(value_row["growth_icc"]),
            "ivp_bm": safe_float(value_row["ivp_bm"]),
        },
    }
    to_json(overview, DOCS_DATA / "overview.json")

    market_hist = pd.read_csv(DATA / "library" / "market_icc" / "history.csv")
    value_hist = pd.read_csv(DATA / "library" / "value_icc_bm" / "history.csv")

    to_json(
        {
            "asof_date": market_row["date"],
            "history": market_hist.sort_values("date").to_dict(orient="records"),
        },
        DOCS_DATA / "market_icc.json",
    )
    to_json(
        {
            "asof_date": value_row["date"],
            "history": value_hist.sort_values("date").to_dict(orient="records"),
        },
        DOCS_DATA / "value_icc_bm.json",
    )
    to_json(
        {
            "asof_date": market_row["date"],
            "latest": industry_df.to_dict(orient="records"),
        },
        DOCS_DATA / "industry_icc.json",
    )
    to_json(
        {
            "asof_date": market_row["date"],
            "latest": etf_df.to_dict(orient="records"),
        },
        DOCS_DATA / "etf_icc.json",
    )
    to_json(
        {
            "asof_date": market_row["date"],
            "latest": country_df.to_dict(orient="records"),
        },
        DOCS_DATA / "country_icc.json",
    )

    downloads = DOCS_DATA / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    publish_df(market_hist, downloads / "market_icc_history.csv")
    publish_df(value_hist, downloads / "value_icc_bm_history.csv")
    publish_df(industry_df, downloads / "industry_icc_latest.csv")
    publish_df(etf_df.drop(columns=["top_constituents"], errors="ignore"), downloads / "etf_icc_latest.csv")
    publish_df(country_df.drop(columns=["top_constituents"], errors="ignore"), downloads / "country_icc_latest.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="usall")
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()

    snapshot_path = Path(args.snapshot).resolve() if args.snapshot else find_latest_snapshot(args.universe)
    snapshot = load_snapshot(snapshot_path, universe=args.universe)

    market_row = build_market_icc(snapshot)
    value_row = build_value_icc_bm(snapshot)
    industry_df = build_industry_icc(snapshot)
    etf_df = build_etf_icc(snapshot)
    country_df = build_country_icc(snapshot)
    build_docs_payload(market_row, value_row, industry_df, etf_df, country_df)

    print(f"[ok] aggregate library built from {snapshot_path}")
    print(f"[ok] market vw_icc = {market_row['vw_icc']:.6f}")
    print(f"[ok] ivp_bm = {value_row['ivp_bm']:.6f}")


if __name__ == "__main__":
    main()
