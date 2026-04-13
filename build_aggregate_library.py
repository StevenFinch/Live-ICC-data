from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from library_common import (
    CONFIG,
    DOCS,
    LIBRARY,
    build_live_lookup,
    df_records,
    ensure_dir,
    find_latest_snapshot,
    get_etf_holdings,
    load_snapshot,
    resolve_holdings_icc,
    safe_float,
    safe_int,
    today_et,
    upsert_history,
    weighted_mean,
    write_json,
)


VALUE_PORTS = ["S/L", "S/M", "S/H", "B/L", "B/M", "B/H"]


def build_market(snapshot: pd.DataFrame, out_dir: Path) -> dict:
    row = {
        "date": str(snapshot["date"].dropna().iloc[0]),
        "universe": "usall",
        "market_icc_vw": weighted_mean(snapshot["ICC"], snapshot["mktcap"]),
        "market_icc_ew": float(snapshot["ICC"].mean()),
        "n_firms": int(snapshot["ICC"].notna().sum()),
    }
    latest_df = pd.DataFrame([row])
    history_path = out_dir / "history.csv"
    hist = upsert_history(latest_df, history_path, ["date", "universe"], ["date", "universe"])

    ensure_dir(out_dir)
    latest_df.to_csv(out_dir / "latest.csv", index=False)
    hist.to_csv(history_path, index=False)

    payload = {
        "asof_date": row["date"],
        "series": {
            "market_icc_vw": safe_float(row["market_icc_vw"]),
            "market_icc_ew": safe_float(row["market_icc_ew"]),
            "n_firms": safe_int(row["n_firms"]),
        },
        "history": df_records(hist),
        "files": {
            "latest_csv": str((out_dir / "latest.csv").relative_to(LIBRARY.parent.parent)),
            "history_csv": str(history_path.relative_to(LIBRARY.parent.parent)),
        },
    }
    write_json(payload, out_dir / "latest.json")
    return payload


def build_value_icc_bm(snapshot: pd.DataFrame, out_dir: Path) -> dict:
    raw_n = len(snapshot)

    df = snapshot.copy()
    df = df[
        df["mktcap"].notna()
        & df["bm"].notna()
        & df["ICC"].notna()
        & np.isfinite(df["mktcap"])
        & np.isfinite(df["bm"])
        & np.isfinite(df["ICC"])
        & (df["mktcap"] > 0)
        & (df["bm"] > 0)
    ].copy()

    basic_filtered_n = len(df)
    q_lo = float(df["ICC"].quantile(0.005))
    q_hi = float(df["ICC"].quantile(0.995))
    df = df[(df["ICC"] >= q_lo) & (df["ICC"] <= q_hi)].copy()

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
    ).astype(str)
    df["portfolio"] = df["size_bucket"] + "/" + df["bm_bucket"]

    port_rows = []
    for p in VALUE_PORTS:
        g = df.loc[df["portfolio"] == p].copy()
        port_rows.append(
            {
                "portfolio": p,
                "n_firms": int(len(g)),
                "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
                "ew_icc": float(g["ICC"].mean()) if len(g) else np.nan,
                "vw_bm": weighted_mean(g["bm"], g["mktcap"]) if len(g) else np.nan,
                "total_mktcap": float(g["mktcap"].sum()) if len(g) else np.nan,
            }
        )
    pt = pd.DataFrame(port_rows)
    d = pt.set_index("portfolio")["vw_icc"].to_dict()

    icc_l = np.nan
    if pd.notna(d.get("S/L", np.nan)) and pd.notna(d.get("B/L", np.nan)):
        icc_l = float((d["S/L"] + d["B/L"]) / 2.0)

    icc_h = np.nan
    if pd.notna(d.get("S/H", np.nan)) and pd.notna(d.get("B/H", np.nan)):
        icc_h = float((d["S/H"] + d["B/H"]) / 2.0)

    ivp = icc_h - icc_l if (pd.notna(icc_h) and pd.notna(icc_l)) else np.nan

    summary = {
        "date": str(df["date"].dropna().iloc[0]),
        "universe": "usall",
        "n_firms_raw": raw_n,
        "n_firms_basic_filtered": basic_filtered_n,
        "n_firms_used": int(len(df)),
        "trim_lo_icc": q_lo,
        "trim_hi_icc": q_hi,
        "size_median": size_median,
        "bm30": bm30,
        "bm70": bm70,
        "market_icc": weighted_mean(df["ICC"], df["mktcap"]),
        "value_icc": icc_h,
        "growth_icc": icc_l,
        "ivp_bm": ivp,
    }

    ensure_dir(out_dir)
    pd.DataFrame([summary]).to_csv(out_dir / "latest.csv", index=False)
    pt.to_csv(out_dir / "latest_portfolios.csv", index=False)

    members = df[
        ["date", "ticker", "name", "sector", "portfolio", "size_bucket", "bm_bucket", "mktcap", "bm", "ICC"]
    ].copy()
    members = members.sort_values(["portfolio", "mktcap"], ascending=[True, False]).reset_index(drop=True)
    members.to_csv(out_dir / "latest_members.csv", index=False)

    history_path = out_dir / "history.csv"
    hist = upsert_history(pd.DataFrame([summary]), history_path, ["date", "universe"], ["date", "universe"])
    hist.to_csv(history_path, index=False)

    chart_payload = {
        "dates": hist["date"].astype(str).tolist(),
        "ivp_bm": [None if pd.isna(x) else float(x) for x in hist["ivp_bm"]],
        "value_icc": [None if pd.isna(x) else float(x) for x in hist["value_icc"]],
        "growth_icc": [None if pd.isna(x) else float(x) for x in hist["growth_icc"]],
        "market_icc": [None if pd.isna(x) else float(x) for x in hist["market_icc"]],
    }
    write_json(chart_payload, out_dir / "chart.json")

    payload = {
        "asof_date": summary["date"],
        "methodology": {
            "label": "Live Daily Approximation of IVP(B/M)",
            "sort": "2x3 sort on size and current-snapshot book-to-market within the latest usall snapshot",
            "trim_icc": "Remove top and bottom 0.5% of firm ICC values in the current cross-section",
            "bucket_icc": "Value-weighted ICC within each bucket using market cap",
            "value_icc": "Average of S/H and B/H",
            "growth_icc": "Average of S/L and B/L",
            "ivp_bm": "value_icc - growth_icc",
        },
        "summary": {k: safe_float(v) if isinstance(v, (float, np.floating)) else v for k, v in summary.items()},
        "portfolios": df_records(pt),
        "history": df_records(hist),
        "files": {
            "latest_csv": str((out_dir / "latest.csv").relative_to(LIBRARY.parent.parent)),
            "latest_portfolios_csv": str((out_dir / "latest_portfolios.csv").relative_to(LIBRARY.parent.parent)),
            "latest_members_csv": str((out_dir / "latest_members.csv").relative_to(LIBRARY.parent.parent)),
            "history_csv": str(history_path.relative_to(LIBRARY.parent.parent)),
        },
    }
    write_json(payload, out_dir / "latest.json")
    return payload


def build_industry(snapshot: pd.DataFrame, out_dir: Path, min_firms: int = 5) -> dict:
    df = snapshot.copy()
    df["industry_bucket"] = df["sector"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    rows = []
    for industry, g in df.groupby("industry_bucket", dropna=False):
        if len(g) < min_firms:
            continue
        rows.append(
            {
                "date": str(g["date"].dropna().iloc[0]),
                "industry": industry,
                "n_firms": int(len(g)),
                "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
                "ew_icc": float(g["ICC"].mean()),
                "vw_bm": weighted_mean(g["bm"], g["mktcap"]),
                "total_mktcap": float(g["mktcap"].sum()),
            }
        )

    latest_df = pd.DataFrame(rows).sort_values("vw_icc", ascending=False).reset_index(drop=True)

    ensure_dir(out_dir)
    latest_df.to_csv(out_dir / "latest.csv", index=False)

    history_path = out_dir / "history.csv"
    hist = upsert_history(latest_df, history_path, ["date", "industry"], ["date", "industry"])
    hist.to_csv(history_path, index=False)

    payload = {
        "asof_date": None if latest_df.empty else str(latest_df["date"].iloc[0]),
        "bucket_definition": "Yahoo Finance sector field from the live snapshot (industry proxy for the website layer)",
        "latest": df_records(latest_df),
        "history": df_records(hist),
        "files": {
            "latest_csv": str((out_dir / "latest.csv").relative_to(LIBRARY.parent.parent)),
            "history_csv": str(history_path.relative_to(LIBRARY.parent.parent)),
        },
    }
    write_json(payload, out_dir / "latest.json")
    return payload


def build_etf(
    snapshot: pd.DataFrame,
    out_dir: Path,
    config_path: Path,
    snapshot_lookup: dict[str, dict],
    fetched_cache: dict[str, dict],
) -> tuple[dict, pd.DataFrame]:
    cfg = pd.read_csv(config_path)
    cfg.columns = [str(c).strip().lower() for c in cfg.columns]
    required = {"ticker", "label", "group"}
    missing = required - set(cfg.columns)
    if missing:
        raise ValueError(f"{config_path} missing required columns: {sorted(missing)}")

    etf_rows = []
    holdings_rows = []

    for _, row in cfg.iterrows():
        etf = str(row["ticker"]).upper().strip()
        label = str(row["label"]).strip()
        group = str(row["group"]).strip()

        holdings, holdings_source = get_etf_holdings(etf)
        if holdings is None or holdings.empty:
            etf_rows.append(
                {
                    "date": today_et().isoformat(),
                    "etf": etf,
                    "label": label,
                    "group": group,
                    "holdings_source": holdings_source,
                    "n_holdings_total": 0,
                    "n_holdings_used": 0,
                    "coverage_weight": 0.0,
                    "etf_icc": np.nan,
                }
            )
            continue

        resolved = resolve_holdings_icc(holdings, snapshot_lookup, fetched_cache)
        merged = holdings.merge(resolved, on=["holding_ticker", "weight"], how="left")
        coverage = float(merged.loc[merged["icc"].notna(), "weight"].sum())
        etf_icc = weighted_mean(merged["icc"], merged["weight"])

        etf_rows.append(
            {
                "date": today_et().isoformat(),
                "etf": etf,
                "label": label,
                "group": group,
                "holdings_source": holdings_source,
                "n_holdings_total": int(len(merged)),
                "n_holdings_used": int(merged["icc"].notna().sum()),
                "coverage_weight": coverage,
                "etf_icc": etf_icc,
            }
        )

        tmp = merged.copy()
        tmp.insert(1, "etf", etf)
        tmp.insert(2, "label", label)
        tmp.insert(3, "group", group)
        holdings_rows.append(tmp)

    latest_df = pd.DataFrame(etf_rows).sort_values(["group", "label"]).reset_index(drop=True)
    holdings_df = pd.concat(holdings_rows, ignore_index=True) if holdings_rows else pd.DataFrame()

    ensure_dir(out_dir)
    latest_df.to_csv(out_dir / "latest.csv", index=False)
    if not holdings_df.empty:
        holdings_df.to_csv(out_dir / "latest_holdings.csv", index=False)

    history_path = out_dir / "history.csv"
    hist = upsert_history(latest_df, history_path, ["date", "etf"], ["date", "etf"])
    hist.to_csv(history_path, index=False)

    payload = {
        "asof_date": None if latest_df.empty else str(latest_df["date"].iloc[0]),
        "methodology": {
            "etf_icc": "Holdings-based ETF ICC. Use config/etf_holdings/<ETF>.csv if available; otherwise use yfinance documented top_holdings fallback.",
            "coverage_weight": "Sum of constituent weights with a successfully resolved constituent ICC.",
        },
        "latest": df_records(latest_df),
        "history": df_records(hist),
        "files": {
            "latest_csv": str((out_dir / "latest.csv").relative_to(LIBRARY.parent.parent)),
            "history_csv": str(history_path.relative_to(LIBRARY.parent.parent)),
            "latest_holdings_csv": None if holdings_df.empty else str((out_dir / "latest_holdings.csv").relative_to(LIBRARY.parent.parent)),
        },
    }
    write_json(payload, out_dir / "latest.json")
    return payload, latest_df


def build_country_proxy(country_cfg_path: Path, etf_latest_df: pd.DataFrame, out_dir: Path) -> dict:
    cfg = pd.read_csv(country_cfg_path)
    cfg.columns = [str(c).strip().lower() for c in cfg.columns]
    required = {"country", "etf", "label", "region"}
    missing = required - set(cfg.columns)
    if missing:
        raise ValueError(f"{country_cfg_path} missing required columns: {sorted(missing)}")

    merged = cfg.merge(etf_latest_df, left_on="etf", right_on="etf", how="left")
    latest = pd.DataFrame(
        {
            "date": merged["date"].fillna(today_et().isoformat()),
            "country": merged["country"],
            "region": merged["region"],
            "label": merged["label_x"] if "label_x" in merged.columns else merged["label"],
            "proxy_etf": merged["etf"],
            "proxy_etf_icc": merged["etf_icc"],
            "coverage_weight": merged["coverage_weight"],
            "holdings_source": merged["holdings_source"],
        }
    ).sort_values(["region", "country"]).reset_index(drop=True)

    ensure_dir(out_dir)
    latest.to_csv(out_dir / "latest.csv", index=False)

    history_path = out_dir / "history.csv"
    hist = upsert_history(latest, history_path, ["date", "country"], ["date", "country"])
    hist.to_csv(history_path, index=False)

    payload = {
        "asof_date": None if latest.empty else str(latest["date"].iloc[0]),
        "methodology": {
            "country_icc": "Country proxy ICC built from single-country ETF holdings-based ICCs.",
            "note": "This is the website-ready proxy layer. A true FF-style country-stock library would require separate non-US stock universes.",
        },
        "latest": df_records(latest),
        "history": df_records(hist),
        "files": {
            "latest_csv": str((out_dir / "latest.csv").relative_to(LIBRARY.parent.parent)),
            "history_csv": str(history_path.relative_to(LIBRARY.parent.parent)),
        },
    }
    write_json(payload, out_dir / "latest.json")
    return payload




def mirror_downloads() -> None:
    dl = ensure_dir(DOCS / "data" / "downloads")

    copies = {
        LIBRARY / "market_icc" / "history.csv": dl / "market_icc_history.csv",
        LIBRARY / "value_icc_bm" / "history.csv": dl / "value_icc_bm_history.csv",
        LIBRARY / "value_icc_bm" / "latest_portfolios.csv": dl / "value_icc_bm_latest_portfolios.csv",
        LIBRARY / "value_icc_bm" / "latest_members.csv": dl / "value_icc_bm_latest_members.csv",
        LIBRARY / "industry_icc" / "history.csv": dl / "industry_icc_history.csv",
        LIBRARY / "etf_icc" / "history.csv": dl / "etf_icc_history.csv",
        LIBRARY / "country_icc" / "history.csv": dl / "country_icc_history.csv",
    }

    for src, dst in copies.items():
        if src.exists():
            dst.write_bytes(src.read_bytes())


def write_docs_json(outputs: dict[str, dict]) -> None:
    data_dir = ensure_dir(DOCS / "data")
    manifest = {
        "asof_date": today_et().isoformat(),
        "datasets": {
            name: f"data/{name}.json" for name in outputs.keys()
        },
    }
    for name, payload in outputs.items():
        write_json(payload, data_dir / f"{name}.json")
    write_json(manifest, data_dir / "manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="usall")
    ap.add_argument("--input-file", default=None)
    args = ap.parse_args()

    input_file = Path(args.input_file).resolve() if args.input_file else find_latest_snapshot(args.universe)
    snapshot = load_snapshot(input_file)
    snapshot_lookup = build_live_lookup(snapshot)
    fetched_cache: dict[str, dict] = {}

    outputs = {}

    outputs["market_icc"] = build_market(snapshot, ensure_dir(LIBRARY / "market_icc"))
    outputs["value_icc_bm"] = build_value_icc_bm(snapshot, ensure_dir(LIBRARY / "value_icc_bm"))
    outputs["industry_icc"] = build_industry(snapshot, ensure_dir(LIBRARY / "industry_icc"))

    etf_payload, etf_latest_df = build_etf(
        snapshot=snapshot,
        out_dir=ensure_dir(LIBRARY / "etf_icc"),
        config_path=CONFIG / "etfs.csv",
        snapshot_lookup=snapshot_lookup,
        fetched_cache=fetched_cache,
    )
    outputs["etf_icc"] = etf_payload

    outputs["country_icc"] = build_country_proxy(
        country_cfg_path=CONFIG / "country_etfs.csv",
        etf_latest_df=etf_latest_df,
        out_dir=ensure_dir(LIBRARY / "country_icc"),
    )

    overview = {
        "asof_date": today_et().isoformat(),
        "source_snapshot": str(input_file.relative_to(input_file.parents[1])),
        "cards": {
            "market_icc_vw": outputs["market_icc"]["series"]["market_icc_vw"],
            "value_icc": outputs["value_icc_bm"]["summary"]["value_icc"],
            "growth_icc": outputs["value_icc_bm"]["summary"]["growth_icc"],
            "ivp_bm": outputs["value_icc_bm"]["summary"]["ivp_bm"],
        },
        "downloads": {
            "market_icc": "data/downloads/market_icc_history.csv",
            "value_icc_bm": "data/downloads/value_icc_bm_history.csv",
            "industry_icc": "data/downloads/industry_icc_history.csv",
            "etf_icc": "data/downloads/etf_icc_history.csv",
            "country_icc": "data/downloads/country_icc_history.csv",
        },
    }
    outputs["overview"] = overview

    write_docs_json(outputs)
    mirror_downloads()
    print("Built aggregate library and docs JSON.")
    print(f"Snapshot: {input_file}")


if __name__ == "__main__":
    main()
