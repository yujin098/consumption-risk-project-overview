import unittest

import pandas as pd

from src.normalize_data import normalize_consumption


class NormalizeConsumptionTests(unittest.TestCase):
    def sample(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "STRD_YYMM": [202601],
                "SIDO_NM": ["서울특별시"],
                "CCG_NM": ["종로구"],
                "GENDER_CD": [1],
                "AGE_CD": [3],
                "TP_BUZ_NO": [8001],
                "TP_BUZ_NM": ["일반한식"],
                "amt": [100_000],
                "cnt": [10],
            }
        )

    def test_standardizes_columns_and_types(self):
        result = normalize_consumption(self.sample())

        self.assertEqual(
            list(result.columns),
            [
                "year_month",
                "province",
                "district",
                "gender_code",
                "age_code",
                "industry_code",
                "industry_name",
                "amount",
                "transactions",
            ],
        )
        self.assertEqual(result.loc[0, "year_month"], pd.Timestamp("2026-01-01"))
        self.assertEqual(result.loc[0, "gender_code"], "1")
        self.assertEqual(result.loc[0, "age_code"], "3")

    def test_rejects_negative_transactions(self):
        frame = self.sample()
        frame.loc[0, "cnt"] = -1

        with self.assertRaisesRegex(ValueError, "transactions"):
            normalize_consumption(frame)

    def test_rejects_invalid_month(self):
        frame = self.sample()
        frame.loc[0, "STRD_YYMM"] = 202613

        with self.assertRaisesRegex(ValueError, "year_month"):
            normalize_consumption(frame)

    def test_rejects_missing_required_key(self):
        frame = self.sample()
        frame.loc[0, "CCG_NM"] = None

        with self.assertRaisesRegex(ValueError, "district"):
            normalize_consumption(frame)


if __name__ == "__main__":
    unittest.main()
