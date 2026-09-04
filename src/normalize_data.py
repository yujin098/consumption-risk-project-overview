"""Normalize the BC Card contest dataset to stable analysis columns."""

from __future__ import annotations

import pandas as pd


COLUMN_MAPPING = {
    "STRD_YYMM": "year_month",
    "SIDO_NM": "province",
    "CCG_NM": "district",
    "GENDER_CD": "gender_code",
    "AGE_CD": "age_code",
    "TP_BUZ_NO": "industry_code",
    "TP_BUZ_NM": "industry_name",
    "amt": "amount",
    "cnt": "transactions",
}

KEY_COLUMNS = [
    "year_month",
    "province",
    "district",
    "gender_code",
    "age_code",
    "industry_code",
]


def normalize_consumption(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename, type, and validate the contest dataset's required fields."""
    missing_columns = [column for column in COLUMN_MAPPING if column not in frame]
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    result = frame[list(COLUMN_MAPPING)].rename(columns=COLUMN_MAPPING).copy()
    result["year_month"] = pd.to_datetime(
        result["year_month"].astype("string"), format="%Y%m", errors="coerce"
    )
    result["gender_code"] = result["gender_code"].astype("string")
    result["age_code"] = result["age_code"].astype("string")

    for column in ["province", "district", "industry_name"]:
        result[column] = result[column].astype("string").str.strip()

    for column in KEY_COLUMNS:
        if result[column].isna().any():
            raise ValueError(f"{column} contains missing or invalid values")

    if (result["transactions"] < 0).any():
        raise ValueError("transactions contains negative values")

    return result
