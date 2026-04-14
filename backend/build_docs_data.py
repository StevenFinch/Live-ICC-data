
from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DOCS_DIR = REPO / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"
DOWNLOAD_DIR = DOCS_DATA_DIR / "downloads"

SNAPSHOT_RE = re.compile(
    r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$"
)

# Keep this list conservative so the page only shows indices that the pipeline actually knows how to build.
INDEX_UNIVERSES = [
    "sp500",
    "sp100",
    "dow30",
    "ndx100",
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


def json_safe(obj):
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)


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

    if df is None or len(df.columns) == 0:
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
            continue
        valid.append((path, df))
    return valid


def dedup_valid_snapshots(valid_snapshots: list[tuple[Path, pd.DataFrame]]) -> list[tuple[Path, pd.DataFrame]]:
    """
    Deduplicate same (universe, date): keep the last one only.
    Input is already sorted by date -> rerun -> filename, so the last one is the newest rerun.
    """
    keep: dict[tuple[str, str], tuple[Path, pd.DataFrame]] = {}
    for path, df in valid_snapshots:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        if not df["date"].notna().any():
            continue
        date_value = str(df["date"].dropna().iloc[0])
        key = (meta["universe"], date_value)
        keep[key] = (path, df)

    out = list(keep.values())
    out.sort(
        key=lambda x: (
            str(x[1]["date"].dropna().iloc[0]),
            parse_snapshot_meta(x[0])["universe"],
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
            .apply(lambda g: pd.Series({"vw_icc": weighted_mean(g["ICC"], g["mktcap"])}))
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


def build_industry_daily_history(base_valid: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for path, df in base_valid:
        d = df[df["sector"].notna()].copy()
        if d.empty:
            continue

        agg = (
            d.groupby("sector", dropna=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "date": str(g["date"].iloc[0]),
                        "sector": str(g["sector"].iloc[0]),
                        "vw_icc": weighted_mean(g["ICC"], g["mktcap"]),
                        "ew_icc": float(g["ICC"].mean()),
                        "n_firms": int(len(g)),
                        "source_file": str(path.relative_to(REPO)),
                    }
                )
            )
            .reset_index(drop=True)
        )
        rows.append(agg)

    if not rows:
        return pd.DataFrame(columns=["date", "sector", "vw_icc", "ew_icc", "n_firms", "source_file"])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["date", "sector"]).reset_index(drop=True)
    return out


def to_month_end_panel(df: pd.DataFrame, date_col: str, value_col: str, row_col: str | None = None, months_keep: int = 3) -> tuple[list[str], list[dict]]:
    if df.empty:
        return [], []

    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        return [], []

    tmp["ym"] = tmp[date_col].dt.strftime("%Y-%m")
    # keep month-end (latest available date) within each month and row group
    if row_col is None:
        latest_per_month = tmp.groupby("ym")[date_col].max().reset_index()
        tmp = tmp.merge(latest_per_month, on=["ym", date_col], how="inner")
        tmp = tmp.sort_values("ym")
        months = sorted(tmp["ym"].dropna().unique())[-months_keep:]
        tmp = tmp[tmp["ym"].isin(months)].copy()

        rows = []
        row = {"series": value_col}
        series = tmp.sort_values("ym").set_index("ym")[value_col]
        for m in months:
            row[m] = series.get(m, np.nan)
        rows.append(row)
        return months, rows

    latest_per_group_month = tmp.groupby([row_col, "ym"])[date_col].max().reset_index()
    tmp = tmp.merge(latest_per_group_month, on=[row_col, "ym", date_col], how="inner")
    months = sorted(tmp["ym"].dropna().unique())[-months_keep:]
    tmp = tmp[tmp["ym"].isin(months)].copy()

    rows = []
    for key, g in tmp.groupby(row_col, dropna=False):
        row = {"series": str(key)}
        s = g.sort_values("ym").set_index("ym")[value_col]
        for m in months:
            row[m] = s.get(m, np.nan)
        rows.append(row)

    rows.sort(key=lambda x: x["series"])
    return months, rows


def make_home_monthly_payload(market_all: pd.DataFrame, sp500: pd.DataFrame, value_hist: pd.DataFrame, industry_daily: pd.DataFrame, index_daily: pd.DataFrame) -> dict:
    # Core table
    def _month_rows_from_single(df: pd.DataFrame, value_col: str, label: str, months_keep: int = 3):
        months, rows = to_month_end_panel(df, "date", value_col, None, months_keep)
        if not rows:
            return [], []
        rows[0]["series"] = label
        return months, rows

    months_all, rows_all_vw = _month_rows_from_single(market_all, "vw_icc", "All market VW ICC")
    _, rows_all_ew = _month_rows_from_single(market_all, "ew_icc", "All market EW ICC")
    months_sp, rows_sp_vw = _month_rows_from_single(sp500, "vw_icc", "S&P 500 VW ICC")
    _, rows_sp_ew = _month_rows_from_single(sp500, "ew_icc", "S&P 500 EW ICC")
    months_val, rows_val = to_month_end_panel(value_hist, "date", "value_icc", None, 3)
    _, rows_growth = to_month_end_panel(value_hist, "date", "growth_icc", None, 3)
    _, rows_ivp = to_month_end_panel(value_hist, "date", "ivp_bm", None, 3)

    if rows_val:
        rows_val[0]["series"] = "Value ICC"
    if rows_growth:
        rows_growth[0]["series"] = "Growth ICC"
    if rows_ivp:
        rows_ivp[0]["series"] = "IVP (B/M)"

    core_months = months_all or months_sp or months_val
    core_rows = rows_all_vw + rows_all_ew + rows_sp_vw + rows_sp_ew + rows_val + rows_growth + rows_ivp

    index_months, index_rows = to_month_end_panel(index_daily, "date", "vw_icc", "universe", 3)
    industry_months, industry_rows = to_month_end_panel(industry_daily, "date", "vw_icc", "sector", 3)

    # Order index rows
    index_order = {u: i for i, u in enumerate(INDEX_UNIVERSES)}
    index_rows.sort(key=lambda x: index_order.get(x["series"], 999))

    return {
        "core": {"months": core_months, "rows": core_rows},
        "index": {"months": index_months, "rows": index_rows},
        "industry": {"months": industry_months, "rows": industry_rows},
    }


def filter_last_31_days(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        return tmp
    max_date = tmp[date_col].max()
    cutoff = max_date - pd.Timedelta(days=31)
    tmp = tmp[tmp[date_col] >= cutoff].copy()
    tmp[date_col] = tmp[date_col].dt.strftime("%Y-%m-%d")
    return tmp.sort_values(date_col, ascending=False).reset_index(drop=True)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_family_daily_outputs(market_all: pd.DataFrame, sp500: pd.DataFrame, value_hist: pd.DataFrame, index_daily: pd.DataFrame, industry_daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # marketwide family: one daily file containing All + SP500 rows
    m1 = market_all.copy()
    m1["universe"] = "usall"
    m2 = sp500.copy()
    m2["universe"] = "sp500"
    marketwide = pd.concat([m1, m2], ignore_index=True) if not m1.empty or not m2.empty else pd.DataFrame(
        columns=["date", "universe", "vw_icc", "ew_icc", "n_firms", "source_file"]
    )
    marketwide = marketwide[["date", "universe", "vw_icc", "ew_icc", "n_firms", "source_file"]].sort_values(["date", "universe"]).reset_index(drop=True)

    value_family = value_hist[["date", "value_icc", "growth_icc", "ivp_bm", "n_firms", "source_file"]].copy()
    index_family = index_daily[["date", "universe", "vw_icc", "ew_icc", "n_firms", "source_file"]].copy()
    industry_family = industry_daily[["date", "sector", "vw_icc", "ew_icc", "n_firms", "source_file"]].copy()

    return {
        "marketwide": marketwide,
        "value_icc": value_family,
        "index_icc": index_family,
        "industry_icc": industry_family,
    }


def write_family_downloads(family_name: str, df: pd.DataFrame) -> dict:
    family_dir = DOWNLOAD_DIR / family_name
    family_dir.mkdir(parents=True, exist_ok=True)

    tree = {"family": family_name, "label": family_name.replace("_", " ").title(), "years": []}
    if df.empty:
        return tree

    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp[tmp["date"].notna()].copy()
    if tmp.empty:
        return tree

    tmp["year"] = tmp["date"].dt.strftime("%Y")
    tmp["month"] = tmp["date"].dt.strftime("%Y-%m")
    tmp["date_str"] = tmp["date"].dt.strftime("%Y-%m-%d")

    for year, g_year in sorted(tmp.groupby("year"), reverse=True):
        year_dir = family_dir / year
        year_dir.mkdir(parents=True, exist_ok=True)

        year_entry = {
            "year": year,
            "download_all_label": f"Download all {family_name} files for {year}",
            "download_all_href": f"data/downloads/{family_name}/{year}/all-{family_name}-{year}.zip",
            "months": [],
        }

        month_files_for_zip: list[Path] = []

        for month, g_month in sorted(g_year.groupby("month"), reverse=True):
            month_dir = year_dir / month
            month_dir.mkdir(parents=True, exist_ok=True)

            days = []
            daily_files: list[Path] = []

            for date_str, g_day in sorted(g_month.groupby("date_str"), reverse=True):
                out_name = f"{family_name}_{date_str}.csv"
                out_path = month_dir / out_name

                out_df = g_day.drop(columns=["year", "month", "date_str"]).copy()
                out_df["date"] = out_df["date"].dt.strftime("%Y-%m-%d")
                out_df.to_csv(out_path, index=False)

                daily_files.append(out_path)
                month_files_for_zip.append(out_path)

                days.append(
                    {
                        "date": date_str,
                        "label": date_str,
                        "href": f"data/downloads/{family_name}/{year}/{month}/{out_name}",
                    }
                )

            month_zip = month_dir / f"all-{family_name}-{month}.zip"
            with zipfile.ZipFile(month_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fp in daily_files:
                    zf.write(fp, arcname=fp.name)

            year_entry["months"].append(
                {
                    "month": month,
                    "download_all_label": f"Download all {family_name} files for {month}",
                    "download_all_href": f"data/downloads/{family_name}/{year}/{month}/{month_zip.name}",
                    "days": days,
                }
            )

        year_zip = year_dir / f"all-{family_name}-{year}.zip"
        with zipfile.ZipFile(year_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fp in month_files_for_zip:
                zf.write(fp, arcname=f"{fp.parent.name}/{fp.name}")

        tree["years"].append(year_entry)

    return tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_clean_dir(DOWNLOAD_DIR)

    # Base universe and all valid snapshots
    all_paths = find_all_snapshots(None)
    if not all_paths:
        raise FileNotFoundError("No snapshot files found under data/YYYYMM/")

    all_valid = dedup_valid_snapshots(get_valid_snapshots(all_paths))
    if not all_valid:
        raise RuntimeError("Found snapshot files, but none were valid non-empty CSVs.")

    by_universe: dict[str, list[tuple[Path, pd.DataFrame]]] = {}
    for path, df in all_valid:
        meta = parse_snapshot_meta(path)
        if meta is None:
            continue
        by_universe.setdefault(meta["universe"], []).append((path, df))

    base_valid = by_universe.get(args.universe, [])
    if not base_valid:
        raise RuntimeError(f"No valid snapshots for universe={args.universe}")

    latest_usall_path, latest_usall = base_valid[-1]
    latest_date = str(latest_usall["date"].dropna().iloc[0])

    market_all = build_market_history(base_valid)
    sp500_hist = build_market_history(by_universe.get("sp500", []))
    value_hist = build_value_history(base_valid)

    # index daily history
    index_rows = []
    for universe in INDEX_UNIVERSES:
        hist = build_market_history(by_universe.get(universe, []))
        if hist.empty:
            continue
        hist = hist.copy()
        hist["universe"] = universe
        index_rows.append(hist)

    index_daily = pd.concat(index_rows, ignore_index=True) if index_rows else pd.DataFrame(
        columns=["date", "vw_icc", "ew_icc", "n_firms", "source_file", "universe"]
    )
    index_daily = index_daily.sort_values(["date", "universe"]).reset_index(drop=True)

    industry_daily = build_industry_daily_history(base_valid)
    latest_industry = (
        industry_daily[industry_daily["date"] == industry_daily["date"].max()].copy()
        if not industry_daily.empty
        else pd.DataFrame(columns=["date", "sector", "vw_icc", "ew_icc", "n_firms", "source_file"])
    )

    home_monthly = make_home_monthly_payload(market_all, sp500_hist, value_hist, industry_daily, index_daily)

    # Recent page payloads (recent month only)
    marketwide_recent = filter_last_31_days(
        pd.concat(
            [
                market_all.assign(series="All U.S. market"),
                sp500_hist.assign(series="S&P 500"),
            ],
            ignore_index=True,
        ),
        "date",
    )

    value_recent = filter_last_31_days(value_hist, "date")
    index_recent = filter_last_31_days(index_daily, "date")
    industry_recent = filter_last_31_days(industry_daily, "date")

    # Write page data JSON
    write_json(
        DOCS_DATA_DIR / "overview.json",
        {
            "asof_date": latest_date,
            "source_file": str(latest_usall_path.relative_to(REPO)),
            "nav": [
                {"label": "Home", "href": "index.html"},
                {"label": "Marketwide", "href": "marketwide.html"},
                {"label": "Value", "href": "value.html"},
                {"label": "Indices", "href": "indices.html"},
                {"label": "Industry", "href": "industry.html"},
                {"label": "Downloads", "href": "downloads.html"},
            ],
        },
    )

    write_json(
        DOCS_DATA_DIR / "home_monthly.json",
        {
            "asof_date": latest_date,
            "core": home_monthly["core"],
            "index": home_monthly["index"],
            "industry": home_monthly["industry"],
        },
    )

    write_json(
        DOCS_DATA_DIR / "marketwide_page.json",
        {
            "asof_date": latest_date,
            "recent_daily": marketwide_recent.to_dict(orient="records"),
            "monthly": home_monthly["core"],
        },
    )

    write_json(
        DOCS_DATA_DIR / "value_page.json",
        {
            "asof_date": latest_date,
            "recent_daily": value_recent.to_dict(orient="records"),
            "monthly_rows": [r for r in home_monthly["core"]["rows"] if r["series"] in ["Value ICC", "Growth ICC", "IVP (B/M)"]],
            "months": home_monthly["core"]["months"],
        },
    )

    # Build latest index latest table
    latest_index = (
        index_daily.sort_values(["universe", "date"]).groupby("universe", as_index=False).tail(1).sort_values("universe")
        if not index_daily.empty
        else pd.DataFrame(columns=["date", "vw_icc", "ew_icc", "n_firms", "source_file", "universe"])
    )

    write_json(
        DOCS_DATA_DIR / "indices_page.json",
        {
            "asof_date": latest_date,
            "recent_daily": index_recent.to_dict(orient="records"),
            "latest": latest_index.to_dict(orient="records"),
            "months": home_monthly["index"]["months"],
            "monthly_rows": home_monthly["index"]["rows"],
        },
    )

    write_json(
        DOCS_DATA_DIR / "industry_page.json",
        {
            "asof_date": latest_date,
            "recent_daily": industry_recent.to_dict(orient="records"),
            "latest": latest_industry.sort_values("sector").to_dict(orient="records"),
            "months": home_monthly["industry"]["months"],
            "monthly_rows": home_monthly["industry"]["rows"],
        },
    )

    # Family downloads
    family_frames = build_family_daily_outputs(market_all, sp500_hist, value_hist, index_daily, industry_daily)
    trees = [write_family_downloads(name, frame) for name, frame in family_frames.items()]

    write_json(
        DOCS_DATA_DIR / "download_tree.json",
        {
            "asof_date": latest_date,
            "families": trees,
        },
    )

    print(f"[build_docs_data] total valid deduped snapshots used = {len(all_valid)}")
    print(f"[build_docs_data] latest valid {args.universe} asof_date = {latest_date}")
    print("[build_docs_data] wrote FF-style site data and collapsible download tree.")
