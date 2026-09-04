import unittest

import pandas as pd

from src.features import (
    add_national_benchmarks,
    add_two_way_benchmarks,
    build_region_industry_month,
)


class RegionIndustryFeatureTests(unittest.TestCase):
    def test_aggregates_demographics_and_calculates_monthly_changes(self):
        frame = pd.DataFrame(
            {
                "year_month": pd.to_datetime(
                    ["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"]
                ),
                "province": ["서울특별시"] * 4,
                "district": ["종로구"] * 4,
                "gender_code": ["1", "2", "1", "2"],
                "age_code": ["3", "3", "3", "3"],
                "industry_code": [8001] * 4,
                "industry_name": ["일반한식"] * 4,
                "amount": [60_000, 40_000, 72_000, 48_000],
                "transactions": [6, 4, 6, 4],
            }
        )

        result = build_region_industry_month(frame)

        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[0, "amount"], 100_000)
        self.assertEqual(result.loc[0, "transactions"], 10)
        self.assertEqual(result.loc[0, "average_ticket"], 10_000)
        self.assertAlmostEqual(result.loc[1, "amount_mom"], 0.2)
        self.assertAlmostEqual(result.loc[1, "transactions_mom"], 0.0)
        self.assertAlmostEqual(result.loc[1, "average_ticket_mom"], 0.2)

    def test_keeps_first_month_changes_missing(self):
        frame = pd.DataFrame(
            {
                "year_month": pd.to_datetime(["2026-01-01"]),
                "province": ["서울특별시"],
                "district": ["종로구"],
                "gender_code": ["1"],
                "age_code": ["3"],
                "industry_code": [8001],
                "industry_name": ["일반한식"],
                "amount": [100_000],
                "transactions": [10],
            }
        )

        result = build_region_industry_month(frame)

        self.assertTrue(pd.isna(result.loc[0, "amount_mom"]))
        self.assertTrue(pd.isna(result.loc[0, "transactions_mom"]))

    def test_compares_each_region_with_national_industry_change(self):
        monthly = pd.DataFrame(
            {
                "year_month": pd.to_datetime(
                    ["2026-01-01", "2026-02-01", "2026-01-01", "2026-02-01"]
                ),
                "province": ["서울특별시", "서울특별시", "부산광역시", "부산광역시"],
                "district": ["종로구", "종로구", "중구", "중구"],
                "industry_code": [8001] * 4,
                "industry_name": ["일반한식"] * 4,
                "amount": [100, 120, 100, 100],
                "transactions": [10, 10, 10, 10],
                "average_ticket": [10, 12, 10, 10],
                "amount_mom": [None, 0.2, None, 0.0],
                "transactions_mom": [None, 0.0, None, 0.0],
                "average_ticket_mom": [None, 0.2, None, 0.0],
            }
        )

        result = add_national_benchmarks(monthly)
        February = result.loc[result["year_month"] == "2026-02-01"].set_index(
            "province"
        )

        self.assertAlmostEqual(February.loc["서울특별시", "national_amount_mom"], 0.1)
        self.assertAlmostEqual(February.loc["서울특별시", "relative_amount_mom"], 0.1)
        self.assertAlmostEqual(February.loc["부산광역시", "relative_amount_mom"], -0.1)

    def test_two_way_benchmark_removes_region_wide_common_change(self):
        rows = []
        for province, district, february_value in [
            ("서울특별시", "종로구", 120),
            ("부산광역시", "중구", 100),
        ]:
            for industry_code, industry_name in [(8001, "일반한식"), (8002, "중국음식")]:
                rows.extend(
                    [
                        {
                            "year_month": pd.Timestamp("2026-01-01"),
                            "province": province,
                            "district": district,
                            "industry_code": industry_code,
                            "industry_name": industry_name,
                            "amount": 100,
                            "transactions": 100,
                            "average_ticket": 1,
                        },
                        {
                            "year_month": pd.Timestamp("2026-02-01"),
                            "province": province,
                            "district": district,
                            "industry_code": industry_code,
                            "industry_name": industry_name,
                            "amount": february_value,
                            "transactions": february_value,
                            "average_ticket": 1,
                        },
                    ]
                )
        monthly = pd.DataFrame(rows).sort_values(
            ["province", "district", "industry_code", "year_month"]
        )
        grouped = monthly.groupby(
            ["province", "district", "industry_code"], sort=False
        )
        for column in ["amount", "transactions", "average_ticket"]:
            monthly[f"{column}_mom"] = grouped[column].pct_change(fill_method=None)

        result = add_two_way_benchmarks(add_national_benchmarks(monthly))
        february = result.loc[result["year_month"] == "2026-02-01"]

        self.assertTrue(
            (february["two_way_relative_amount_mom"].abs() < 1e-10).all()
        )
        self.assertTrue(
            (february["two_way_relative_transactions_mom"].abs() < 1e-10).all()
        )


if __name__ == "__main__":
    unittest.main()
