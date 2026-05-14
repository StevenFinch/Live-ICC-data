from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.country_region_cleaner import clean_country_json, clean_country_region_df

REPO = Path(__file__).resolve().parents[1]

CSV_PATTERNS = [
    "data/derived/country_adr/**/*.csv",
    "docs/data/downloads/families/country/**/*.csv",
]


def clean_csv_file(path: Path) -> None:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return
        df = pd.read_csv(path)
        cleaned = clean_country_region_df(df, drop_unavailable=True)
        cleaned.to_csv(path, index=False)
        print(f"[clean_country_outputs] cleaned {path} rows={len(cleaned)}")
    except Exception as exc:
        print(f"[clean_country_outputs] skip {path}: {type(exc).__name__}: {exc}")


def main() -> None:
    for pattern in CSV_PATTERNS:
        for path in REPO.glob(pattern):
            clean_csv_file(path)

    clean_country_json(REPO / "docs" / "data" / "country.json")
    print("[clean_country_outputs] done")


if __name__ == "__main__":
    main()
