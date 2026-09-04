import unittest

import pandas as pd

from src.features import add_national_benchmarks
from src.risk_model import (
    add_cpi_adjustment,
    build_core_age_metrics,
    fit_kmeans,
    summarize_panel_stress,
)


class RiskModelTests(unittest.TestCase):
    def test_adds_cpi_adjusted_real_amount(self):
        monthly = pd.DataFrame(
            {
                "year_month": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                "province": ["서울특별시"] * 2,
                "district": ["종로구"] * 2,
                "industry_code": [8001] * 2,
                "industry_name": ["일반한식"] * 2,
                "amount": [100.0, 110.0],
                "transactions": [10, 10],
                "average_ticket": [10.0, 11.0],
            }
        )
        cpi = pd.DataFrame(
            {
                "year_month": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                "cpi_index": [100.0, 110.0],
            }
        )

        result = add_cpi_adjustment(monthly, cpi)

        self.assertEqual(result.loc[0, "real_amount"], 100.0)
        self.assertEqual(result.loc[1, "real_amount"], 100.0)
        self.assertAlmostEqual(result.loc[1, "real_amount_mom"], 0.0)

    def test_core_age_metric_uses_national_industry_core_age(self):
        frame = pd.DataFrame(
            {
                "year_month": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-01",
                        "2026-06-01",
                        "2026-06-01",
                        "2026-01-01",
                        "2026-01-01",
                        "2026-06-01",
                        "2026-06-01",
                    ]
                ),
                "province": ["서울특별시"] * 4 + ["부산광역시"] * 4,
                "district": ["종로구"] * 4 + ["중구"] * 4,
                "industry_code": [8001] * 8,
                "industry_name": ["일반한식"] * 8,
                "age_code": ["3", "4", "3", "4"] * 2,
                "transactions": [80, 20, 50, 50, 60, 40, 60, 40],
            }
        )

        result = build_core_age_metrics(frame).set_index("province")

        self.assertEqual(result.loc["서울특별시", "core_age_code"], "3")
        self.assertAlmostEqual(result.loc["서울특별시", "core_age_share_change"], -0.3)
        self.assertLess(
            result.loc["서울특별시", "relative_core_age_share_change"], 0
        )

    def test_stress_score_ranks_persistent_decline_above_stable_panel(self):
        months = pd.date_range("2026-01-01", periods=6, freq="MS")
        rows = []
        for month_index, month in enumerate(months):
            rows.extend(
                [
                    {
                        "year_month": month,
                        "province": "서울특별시",
                        "district": "종로구",
                        "industry_code": 8001,
                        "industry_name": "일반한식",
                        "amount": 1000 - 80 * month_index,
                        "transactions": 100 - 8 * month_index,
                    },
                    {
                        "year_month": month,
                        "province": "부산광역시",
                        "district": "중구",
                        "industry_code": 8001,
                        "industry_name": "일반한식",
                        "amount": 1000,
                        "transactions": 100,
                    },
                ]
            )
        monthly = pd.DataFrame(rows)
        monthly["average_ticket"] = monthly["amount"] / monthly["transactions"]
        grouped = monthly.groupby(
            ["province", "district", "industry_code"], sort=False
        )
        for column in ["amount", "transactions", "average_ticket"]:
            monthly[f"{column}_mom"] = grouped[column].pct_change(fill_method=None)
        monthly = add_national_benchmarks(monthly)
        core = pd.DataFrame(
            {
                "province": ["서울특별시", "부산광역시"],
                "district": ["종로구", "중구"],
                "industry_code": [8001, 8001],
                "core_age_code": ["3", "3"],
                "core_age_share_change": [-0.2, 0.0],
                "national_core_age_share_change": [-0.1, -0.1],
                "relative_core_age_share_change": [-0.1, 0.1],
            }
        )

        result = summarize_panel_stress(
            monthly, core, minimum_monthly_transactions=50
        ).set_index("province")

        self.assertGreater(
            result.loc["서울특별시", "csdi"], result.loc["부산광역시", "csdi"]
        )
        self.assertEqual(result.loc["서울특별시", "signal_quality"], "usable")

    def test_kmeans_is_deterministic_and_separates_obvious_groups(self):
        frame = pd.DataFrame(
            {
                "x": [-3.0, -2.8, -3.2, 3.0, 2.8, 3.2],
                "y": [-2.0, -2.2, -1.8, 2.0, 2.2, 1.8],
            }
        )

        first = fit_kmeans(frame, ["x", "y"], k=2, seed=7)
        second = fit_kmeans(frame, ["x", "y"], k=2, seed=7)

        self.assertEqual(first["labels"].tolist(), second["labels"].tolist())
        self.assertEqual(len(set(first["labels"])), 2)
        self.assertEqual(len(set(first["labels"][:3])), 1)
        self.assertEqual(len(set(first["labels"][3:])), 1)


if __name__ == "__main__":
    unittest.main()
