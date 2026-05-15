from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DOCS_DATA = REPO / "docs" / "data"

ALIASES = {
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "Great Britain": "United Kingdom",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Korea": "South Korea",
    "Republic of Korea": "South Korea",
    "Taiwan Province of China": "Taiwan",
    "Hong Kong SAR China": "Hong Kong",
}


def norm_country(x: Any) -> str:
    s = "" if x is None else str(x).strip()
    return ALIASES.get(s, s)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[public_data_sanitizer] cannot parse {path}: {type(exc).__name__}: {exc}")
        return None


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def usable(row: dict[str, Any], value_key: str) -> bool:
    method = str(row.get("method") or "")
    status = str(row.get("status") or "")
    value = row.get(value_key)
    if "Unavailable" in method or "unavailable" in status.lower() or "insufficient" in status.lower():
        return False
    return value is not None


def latest_only(rows: list[dict[str, Any]], date_key: str = "date", group_key: str | None = None, value_key: str = "icc") -> list[dict[str, Any]]:
    cleaned = [dict(r) for r in rows if r.get(date_key) and usable(r, value_key)]
    if not cleaned:
        return []
    max_date = max(str(r[date_key]) for r in cleaned)
    cleaned = [r for r in cleaned if str(r[date_key]) == max_date]
    if group_key:
        best: dict[str, dict[str, Any]] = {}
        for r in cleaned:
            if group_key == "country":
                r[group_key] = norm_country(r.get(group_key))
            key = str(r.get(group_key) or "")
            if not key:
                continue
            old = best.get(key)
            if old is None:
                best[key] = r
            else:
                n_old = float(old.get("n_icc_available") or old.get("n_matched") or 0)
                n_new = float(r.get("n_icc_available") or r.get("n_matched") or 0)
                cov_old = float(old.get("coverage_mktcap") or old.get("coverage_weight") or 0)
                cov_new = float(r.get("coverage_mktcap") or r.get("coverage_weight") or 0)
                if (n_new, cov_new) > (n_old, cov_old):
                    best[key] = r
        return sorted(best.values(), key=lambda x: str(x.get(group_key)))
    return cleaned


def sanitize_family(filename: str, group_key: str | None, value_key: str) -> None:
    path = DOCS_DATA / filename
    data = load_json(path)
    if not data:
        return
    if isinstance(data.get("latest"), list):
        data["latest"] = latest_only(data["latest"], group_key=group_key, value_key=value_key)
    if isinstance(data.get("daily"), list):
        # Keep historical daily data for downloads, but remove unavailable rows.
        daily = [dict(r) for r in data["daily"] if usable(r, value_key)]
        if group_key == "country":
            for r in daily:
                r["country"] = norm_country(r.get("country"))
        data["daily"] = daily
    if isinstance(data.get("monthly"), list):
        monthly = [dict(r) for r in data["monthly"] if usable(r, value_key)]
        if group_key == "country":
            for r in monthly:
                r["country"] = norm_country(r.get("country"))
        data["monthly"] = monthly
    write_json(path, data)
    print(f"[public_data_sanitizer] sanitized {filename}")


def main() -> None:
    sanitize_family("country.json", group_key="country", value_key="icc")
    sanitize_family("etf.json", group_key="ticker", value_key="icc")


if __name__ == "__main__":
    main()
