
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

from backend.holdings_sources import build_etf_icc_panel
from backend.country_adr_sources import build_country_adr_icc_panel


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
DERIVED_DIR = DATA_DIR / "derived"
DOCS_DIR = REPO / "docs"
DOCS_DATA = DOCS_DIR / "data"
DOWNLOADS = DOCS_DATA / "downloads"
ZIP_DIR = DOWNLOADS / "zip"
CONFIG_DIR = REPO / "config"

SNAPSHOT_RE = re.compile(r"^icc_live_(?P<universe>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})(?:_r(?P<rerun>\d+))?\.csv$")
DATE_RE = re.compile(r"(?P<year>\d{4})_(?P<mmdd>\d{4})")


INDEX_UNIVERSES = ["sp500", "sp100", "dow30", "ndx100", "sp400", "sp600", "rut1000"]


def parse_snapshot_meta(path: Path) -> dict | None:
    """Parse metadata from an ICC snapshot filename."""
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
        "date": f"{year}-{mmdd[:2]}-{mmdd[2:]}",
        "yyyymm": f"{year}{mmdd[:2]}",
        "yyyymmdd": f"{year}{mmdd}",
        "rerun": rerun,
    }


def safe_json(obj):
    """Convert numpy/pandas objects into JSON-safe Python objects."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [safe_json(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    return obj


def write_json(path: Path, payload: dict) -> None:
    """Write strict JSON without NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_json(payload), f, indent=2, allow_nan=False)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute weighted mean with finite positive weights."""
    mask = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def load_snapshot(path: Path) -> pd.DataFrame:
    """Load one firm-level ICC snapshot."""
    if path.stat().st_size <= 0:
        raise EmptyDataError(f"Empty file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise EmptyDataError(f"Empty CSV: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"ticker", "mktcap", "ICC", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    for c in ["mktcap", "ICC", "bm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "bm" not in df.columns:
        df["bm"] = np.nan
    for c in ["sector", "name"]:
        if c not in df.columns:
            df[c] = None
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[df["ticker"].notna() & df["date"].notna() & df["ICC"].notna() & df["mktcap"].notna() & (df["mktcap"] > 0)].copy()
    return df


def find_snapshots(universe: str | None = None) -> list[Path]:
    """Find snapshot CSV files under data/YYYYMM."""
    rows = []
    for p in DATA_DIR.glob("*/*.csv"):
        meta = parse_snapshot_meta(p)
        if not meta:
            continue
        if universe is not None and meta["universe"] != universe:
            continue
        rows.append(((meta["universe"], meta["yyyymmdd"], meta["rerun"], p.name), p))
    rows.sort(key=lambda x: x[0])
    return [p for _, p in rows]


def get_valid_snapshots(paths: list[Path]) -> list[tuple[Path, pd.DataFrame]]:
    """Load and de-duplicate valid snapshots by universe and date."""
    keep: dict[tuple[str, str], tuple[Path, pd.DataFrame, int]] = {}
    for p in paths:
        meta = parse_snapshot_meta(p)
        if not meta:
            continue
        try:
            df = load_snapshot(p)
        except Exception as exc:
            print(f"[build_docs_data] skip invalid snapshot {p}: {type(exc).__name__}: {exc}")
            continue
        if df.empty:
            continue
        key = (meta["universe"], str(df["date"].iloc[0]))
        old = keep.get(key)
        if old is None or meta["rerun"] >= old[2]:
            keep[key] = (p, df, meta["rerun"])
    out = [(p, df) for p, df, _ in keep.values()]
    out.sort(key=lambda x: (parse_snapshot_meta(x[0])["universe"], str(x[1]["date"].iloc[0])))
    return out


def build_market_history(valid: list[tuple[Path, pd.DataFrame]], family: str) -> pd.DataFrame:
    """Build daily marketwide/index ICC history."""
    rows = []
    for path, df in valid:
        rows.append({
            "date": str(df["date"].iloc[0]),
            "family": family,
            "daily_icc": weighted_mean(df["ICC"], df["mktcap"]),
            "ew_icc": float(df["ICC"].mean()),
            "n_firms": int(len(df)),
            "method": "ICC calculation",
            "source_file": str(path.relative_to(REPO)),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def build_value_history(valid: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    """Build daily value/growth/IVP history from usall snapshots."""
    rows = []
    for path, raw in valid:
        df = raw.copy()
        df = df[df["bm"].notna() & np.isfinite(df["bm"]) & (df["bm"] > 0)].copy()
        if df.empty:
            continue
        lo, hi = df["ICC"].quantile(0.005), df["ICC"].quantile(0.995)
        df = df[(df["ICC"] >= lo) & (df["ICC"] <= hi)].copy()
        if df.empty:
            continue
        size_med = float(df["mktcap"].median())
        bm30, bm70 = float(df["bm"].quantile(0.30)), float(df["bm"].quantile(0.70))
        df["size_bucket"] = np.where(df["mktcap"] <= size_med, "S", "B")
        df["bm_bucket"] = pd.cut(df["bm"], [-np.inf, bm30, bm70, np.inf], labels=["L", "M", "H"]).astype(str)
        df["portfolio"] = df["size_bucket"] + "/" + df["bm_bucket"]
        d = df.groupby("portfolio").apply(lambda g: weighted_mean(g["ICC"], g["mktcap"])).to_dict()
        growth = np.nan if pd.isna(d.get("S/L", np.nan)) or pd.isna(d.get("B/L", np.nan)) else float((d["S/L"] + d["B/L"]) / 2)
        value = np.nan if pd.isna(d.get("S/H", np.nan)) or pd.isna(d.get("B/H", np.nan)) else float((d["S/H"] + d["B/H"]) / 2)
        ivp = np.nan if pd.isna(value) or pd.isna(growth) else float(value - growth)
        rows.append({
            "date": str(df["date"].iloc[0]),
            "family": "value_premium",
            "value_icc": value,
            "growth_icc": growth,
            "ivp": ivp,
            "daily_icc": ivp,
            "n_firms": int(len(df)),
            "method": "ICC calculation",
            "source_file": str(path.relative_to(REPO)),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def build_industry_history(valid: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    """Build daily industry ICC history from usall snapshots."""
    rows = []
    for path, df in valid:
        df = df[df["sector"].notna()].copy()
        for sector, g in df.groupby("sector"):
            rows.append({
                "date": str(g["date"].iloc[0]),
                "family": "industry",
                "group": str(sector),
                "daily_icc": weighted_mean(g["ICC"], g["mktcap"]),
                "ew_icc": float(g["ICC"].mean()),
                "n_firms": int(len(g)),
                "method": "ICC calculation",
                "source_file": str(path.relative_to(REPO)),
            })
    return pd.DataFrame(rows).sort_values(["date", "group"]).reset_index(drop=True) if rows else pd.DataFrame()


def monthly_from_daily(df: pd.DataFrame, value_cols: list[str], group_cols: list[str] | None = None) -> pd.DataFrame:
    """Convert daily history into month-end observations."""
    if df.empty:
        return df.copy()
    group_cols = group_cols or []
    out = df.copy()
    out["month"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m")
    out = out.sort_values("date")
    idx_cols = group_cols + ["month"]
    monthly = out.groupby(idx_cols, as_index=False).tail(1).copy()
    monthly = monthly.rename(columns={"date": "month_end_date"})
    return monthly.reset_index(drop=True)


def latest_daily(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Return latest daily observation for a family."""
    if df.empty:
        return df.copy()
    group_cols = group_cols or []
    out = df.sort_values("date")
    if group_cols:
        return out.groupby(group_cols, as_index=False).tail(1).reset_index(drop=True)
    return out.tail(1).reset_index(drop=True)


def save_family(name: str, daily: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    """Save family daily/monthly CSV downloads and create a zip."""
    fam_dir = DOWNLOADS / name
    fam_dir.mkdir(parents=True, exist_ok=True)
    daily_path = fam_dir / f"{name}_daily_history.csv"
    monthly_path = fam_dir / f"{name}_monthly_history.csv"
    latest_path = fam_dir / f"{name}_latest.csv"
    daily.to_csv(daily_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    latest_daily(daily, ["group"] if "group" in daily.columns else None).to_csv(latest_path, index=False)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ZIP_DIR / f"{name}_all.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(daily_path, daily_path.relative_to(DOCS_DATA))
        z.write(monthly_path, monthly_path.relative_to(DOCS_DATA))
        z.write(latest_path, latest_path.relative_to(DOCS_DATA))
    return {
        "latest": f"./data/{latest_path.relative_to(DOCS_DATA).as_posix()}",
        "daily": f"./data/{daily_path.relative_to(DOCS_DATA).as_posix()}",
        "monthly": f"./data/{monthly_path.relative_to(DOCS_DATA).as_posix()}",
        "zip": f"./data/{zip_path.relative_to(DOCS_DATA).as_posix()}",
    }


def save_derived_panel(kind: str, panel: pd.DataFrame, date: str) -> Path:
    """Save a derived daily panel such as ETF or country ADR."""
    yyyymm = date[:7].replace("-", "")
    out_dir = DERIVED_DIR / kind / yyyymm
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{kind}_icc_{date.replace('-', '_')}.csv"
    panel.insert(0, "date", date)
    panel.to_csv(out, index=False)
    return out


def load_derived_history(kind: str) -> pd.DataFrame:
    """Load all derived daily CSV files."""
    files = sorted((DERIVED_DIR / kind).glob("*/*.csv"))
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def copy_raw_and_make_zips(valid_all: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    """Copy raw snapshots into docs downloads and create year/month zips."""
    raw_dir = DOWNLOADS / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, df in valid_all:
        meta = parse_snapshot_meta(path)
        if not meta:
            continue
        group = "usall" if meta["universe"] == "usall" else "sp500" if meta["universe"] == "sp500" else "other_indices"
        target_dir = raw_dir / group / meta["yyyymm"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)
        rows.append({
            "date": str(df["date"].iloc[0]),
            "year": meta["year"],
            "month": meta["mmdd"][:2],
            "yyyymm": meta["yyyymm"],
            "universe": meta["universe"],
            "raw_group": group,
            "n_firms": int(len(df)),
            "download_path": f"./data/{target.relative_to(DOCS_DATA).as_posix()}",
        })

    manifest = pd.DataFrame(rows).sort_values(["raw_group", "date", "universe"]) if rows else pd.DataFrame()
    manifest_path = raw_dir / "raw_snapshot_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    zip_rows = []
    if not manifest.empty:
        for (group, year), g in manifest.groupby(["raw_group", "year"]):
            zip_path = ZIP_DIR / f"raw_{group}_{year}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for _, r in g.iterrows():
                    file_path = DOCS_DATA / r["download_path"].replace("./data/", "")
                    if file_path.exists():
                        z.write(file_path, file_path.relative_to(DOCS_DATA))
            zip_rows.append({"scope": "year", "raw_group": group, "year": year, "month": "", "zip_path": f"./data/{zip_path.relative_to(DOCS_DATA).as_posix()}"})
        for (group, yyyymm), g in manifest.groupby(["raw_group", "yyyymm"]):
            zip_path = ZIP_DIR / f"raw_{group}_{yyyymm}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for _, r in g.iterrows():
                    file_path = DOCS_DATA / r["download_path"].replace("./data/", "")
                    if file_path.exists():
                        z.write(file_path, file_path.relative_to(DOCS_DATA))
            zip_rows.append({"scope": "month", "raw_group": group, "year": yyyymm[:4], "month": yyyymm[4:], "zip_path": f"./data/{zip_path.relative_to(DOCS_DATA).as_posix()}"})
    zips = pd.DataFrame(zip_rows)
    zips.to_csv(raw_dir / "raw_snapshot_zips.csv", index=False)
    return manifest


def main() -> None:
    """Build all website JSON/CSV/ZIP outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="usall")
    args = parser.parse_args()

    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    all_valid = get_valid_snapshots(find_snapshots(None))
    by_universe: dict[str, list[tuple[Path, pd.DataFrame]]] = {}
    for p, df in all_valid:
        meta = parse_snapshot_meta(p)
        if meta:
            by_universe.setdefault(meta["universe"], []).append((p, df))

    usall_valid = by_universe.get("usall", [])
    sp500_valid = by_universe.get("sp500", [])
    if not usall_valid:
        raise RuntimeError("No valid usall snapshots found")

    latest_date = str(usall_valid[-1][1]["date"].iloc[0])
    latest_usall = usall_valid[-1][1]

    all_market_daily = build_market_history(usall_valid, "all_market")
    sp500_daily = build_market_history(sp500_valid, "sp500") if sp500_valid else pd.DataFrame()
    marketwide_daily = pd.concat([all_market_daily, sp500_daily], ignore_index=True) if not sp500_daily.empty else all_market_daily
    marketwide_monthly = monthly_from_daily(marketwide_daily, ["daily_icc"], ["family"])

    value_daily = build_value_history(usall_valid)
    value_monthly = monthly_from_daily(value_daily, ["value_icc", "growth_icc", "ivp"], [])

    industry_daily = build_industry_history(usall_valid)
    industry_monthly = monthly_from_daily(industry_daily, ["daily_icc"], ["group"])

    index_frames = []
    for u in INDEX_UNIVERSES:
        if u in by_universe:
            index_frames.append(build_market_history(by_universe[u], u))
    indices_daily = pd.concat(index_frames, ignore_index=True) if index_frames else pd.DataFrame()
    indices_monthly = monthly_from_daily(indices_daily, ["daily_icc"], ["family"]) if not indices_daily.empty else pd.DataFrame()

    etf_panel, etf_members = build_etf_icc_panel(latest_usall, CONFIG_DIR / "etfs.csv")
    save_derived_panel("etf", etf_panel.copy(), latest_date)
    if not etf_members.empty:
        members_dir = DERIVED_DIR / "etf_members" / latest_date[:7].replace("-", "")
        members_dir.mkdir(parents=True, exist_ok=True)
        etf_members.to_csv(members_dir / f"etf_members_{latest_date.replace('-', '_')}.csv", index=False)
    etf_daily = load_derived_history("etf")
    etf_monthly = monthly_from_daily(etf_daily.rename(columns={"icc": "daily_icc"}), ["daily_icc"], ["ticker"]) if not etf_daily.empty else pd.DataFrame()

    country_panel, country_members = build_country_adr_icc_panel(DERIVED_DIR / "country_adr" / "adr_candidates_latest.csv")
    save_derived_panel("country_adr", country_panel.copy(), latest_date)
    if not country_members.empty:
        mem_dir = DERIVED_DIR / "country_adr_members" / latest_date[:7].replace("-", "")
        mem_dir.mkdir(parents=True, exist_ok=True)
        country_members.to_csv(mem_dir / f"country_adr_members_{latest_date.replace('-', '_')}.csv", index=False)
    country_daily = load_derived_history("country_adr")
    country_monthly = monthly_from_daily(country_daily.rename(columns={"icc": "daily_icc"}), ["daily_icc"], ["country"]) if not country_daily.empty else pd.DataFrame()

    downloads = {
        "marketwide": save_family("marketwide", marketwide_daily, marketwide_monthly),
        "value": save_family("value", value_daily, value_monthly),
        "industry": save_family("industry", industry_daily, industry_monthly),
        "indices": save_family("indices", indices_daily, indices_monthly),
        "etf": save_family("etf", etf_daily, etf_monthly),
        "country": save_family("country", country_daily, country_monthly),
    }

    raw_manifest = copy_raw_and_make_zips(all_valid)

    write_json(DOCS_DATA / "overview.json", {
        "title": "Live ICC data library",
        "asof_date": latest_date,
        "downloads": downloads,
    })
    write_json(DOCS_DATA / "marketwide.json", {
        "latest": latest_daily(marketwide_daily, ["family"]).to_dict(orient="records"),
        "daily": marketwide_daily.to_dict(orient="records"),
        "monthly": marketwide_monthly.to_dict(orient="records"),
        "downloads": downloads["marketwide"],
    })
    write_json(DOCS_DATA / "value.json", {
        "latest": latest_daily(value_daily).to_dict(orient="records"),
        "daily": value_daily.to_dict(orient="records"),
        "monthly": value_monthly.to_dict(orient="records"),
        "downloads": downloads["value"],
    })
    write_json(DOCS_DATA / "industry.json", {
        "latest": latest_daily(industry_daily, ["group"]).to_dict(orient="records"),
        "daily": industry_daily.to_dict(orient="records"),
        "monthly": industry_monthly.to_dict(orient="records"),
        "downloads": downloads["industry"],
    })
    write_json(DOCS_DATA / "indices.json", {
        "latest": latest_daily(indices_daily, ["family"]).to_dict(orient="records"),
        "daily": indices_daily.to_dict(orient="records"),
        "monthly": indices_monthly.to_dict(orient="records"),
        "downloads": downloads["indices"],
    })
    write_json(DOCS_DATA / "etf.json", {
        "latest": latest_daily(etf_daily, ["ticker"]).to_dict(orient="records") if not etf_daily.empty else [],
        "daily": etf_daily.to_dict(orient="records") if not etf_daily.empty else [],
        "monthly": etf_monthly.to_dict(orient="records") if not etf_monthly.empty else [],
        "downloads": downloads["etf"],
        "method_note": "ETF ICC uses online full holdings when available; otherwise partial holdings estimates are clearly labeled.",
    })
    write_json(DOCS_DATA / "country.json", {
        "latest": latest_daily(country_daily, ["country"]).to_dict(orient="records") if not country_daily.empty else [],
        "daily": country_daily.to_dict(orient="records") if not country_daily.empty else [],
        "monthly": country_monthly.to_dict(orient="records") if not country_monthly.empty else [],
        "downloads": downloads["country"],
        "method_note": "Country ICC uses a Country ADR Top-10 composite from online US-listed ADR-like securities.",
    })
    write_json(DOCS_DATA / "downloads_catalog.json", {
        "families": downloads,
        "raw_snapshots": raw_manifest.to_dict(orient="records") if not raw_manifest.empty else [],
    })

    print(f"[build_docs_data] wrote docs/data for {latest_date}")


if __name__ == "__main__":
    main()
