"""Feature construction for region-industry consumption analysis."""

from __future__ import annotations

import pandas as pd


def build_region_industry_month(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate demographics and calculate within-panel monthly changes."""
    working = frame.copy()
    working["industry_name"] = working["industry_name"].str.replace(
        r"\s+", "", regex=True
    )
    group_columns = [
        "year_month",
        "province",
        "district",
        "industry_code",
        "industry_name",
    ]
    panel_columns = ["province", "district", "industry_code"]
    result = (
        working.groupby(group_columns, as_index=False)
        .agg(amount=("amount", "sum"), transactions=("transactions", "sum"))
        .sort_values(panel_columns + ["year_month"])
        .reset_index(drop=True)
    )
    result["average_ticket"] = result["amount"] / result["transactions"]

    grouped = result.groupby(panel_columns, sort=False)
    for column in ["amount", "transactions", "average_ticket"]:
        result[f"{column}_mom"] = grouped[column].pct_change(fill_method=None)

    return result


def add_national_benchmarks(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compare each panel's monthly change with its national industry change."""
    national = (
        monthly.groupby(
            ["year_month", "industry_code", "industry_name"], as_index=False
        )
        .agg(
            national_amount=("amount", "sum"),
            national_transactions=("transactions", "sum"),
        )
        .sort_values(["industry_code", "year_month"])
    )
    national["national_average_ticket"] = (
        national["national_amount"] / national["national_transactions"]
    )
    grouped = national.groupby("industry_code", sort=False)
    for column in ["amount", "transactions", "average_ticket"]:
        national[f"national_{column}_mom"] = grouped[f"national_{column}"].pct_change(
            fill_method=None
        )

    benchmark_columns = [
        "year_month",
        "industry_code",
        "national_amount_mom",
        "national_transactions_mom",
        "national_average_ticket_mom",
    ]
    result = monthly.merge(
        national[benchmark_columns],
        on=["year_month", "industry_code"],
        how="left",
        validate="many_to_one",
    )
    for column in ["amount", "transactions", "average_ticket"]:
        result[f"relative_{column}_mom"] = (
            result[f"{column}_mom"] - result[f"national_{column}_mom"]
        )
    return result


def add_two_way_benchmarks(monthly: pd.DataFrame) -> pd.DataFrame:
    """Remove both national industry and region-wide monthly changes.

    This multiplicative residual helps prevent a tourism or local-event shock that
    hits every industry in one district from being mistaken for industry-specific
    stress.
    """
    required = {
        "national_amount_mom",
        "national_transactions_mom",
        "national_average_ticket_mom",
    }
    missing = required.difference(monthly.columns)
    if missing:
        raise ValueError(
            "National benchmarks must be added first; missing " f"{sorted(missing)}"
        )

    region_keys = ["province", "district"]
    region = (
        monthly.groupby(["year_month"] + region_keys, as_index=False)
        .agg(
            region_amount=("amount", "sum"),
            region_transactions=("transactions", "sum"),
        )
        .sort_values(region_keys + ["year_month"])
    )
    region["region_average_ticket"] = (
        region["region_amount"] / region["region_transactions"]
    )
    grouped_region = region.groupby(region_keys, sort=False)
    for column in ["amount", "transactions", "average_ticket"]:
        region[f"region_{column}_mom"] = grouped_region[f"region_{column}"].pct_change(
            fill_method=None
        )

    national = (
        monthly.groupby("year_month", as_index=False)
        .agg(
            total_amount=("amount", "sum"),
            total_transactions=("transactions", "sum"),
        )
        .sort_values("year_month")
    )
    national["total_average_ticket"] = (
        national["total_amount"] / national["total_transactions"]
    )
    for column in ["amount", "transactions", "average_ticket"]:
        national[f"total_{column}_mom"] = national[f"total_{column}"].pct_change(
            fill_method=None
        )

    result = monthly.merge(
        region[
            ["year_month"]
            + region_keys
            + [
                "region_amount_mom",
                "region_transactions_mom",
                "region_average_ticket_mom",
            ]
        ],
        on=["year_month"] + region_keys,
        how="left",
        validate="many_to_one",
    ).merge(
        national[
            [
                "year_month",
                "total_amount_mom",
                "total_transactions_mom",
                "total_average_ticket_mom",
            ]
        ],
        on="year_month",
        how="left",
        validate="many_to_one",
    )
    for column in ["amount", "transactions", "average_ticket"]:
        region_relative_factor = (1 + result[f"region_{column}_mom"]) / (
            1 + result[f"total_{column}_mom"]
        )
        result[f"two_way_relative_{column}_mom"] = (
            (1 + result[f"{column}_mom"])
            / (1 + result[f"national_{column}_mom"])
            / region_relative_factor
            - 1
        )
    return result
