"""CSDI scoring and consumption-pattern clustering.

The model is deliberately portfolio-level: one row represents a
province/district/industry combination, never an individual merchant or borrower.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PANEL_KEYS = ["province", "district", "industry_code"]


def add_cpi_adjustment(monthly: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    """Deflate nominal card spending with the national CPI (2020=100)."""
    required = {"year_month", "cpi_index"}
    missing = required.difference(cpi.columns)
    if missing:
        raise ValueError(f"CPI data is missing columns: {sorted(missing)}")

    cpi_values = cpi[list(required)].copy()
    cpi_values["year_month"] = pd.to_datetime(cpi_values["year_month"])
    if cpi_values["year_month"].duplicated().any():
        raise ValueError("CPI data must have one row per month")
    if (cpi_values["cpi_index"] <= 0).any():
        raise ValueError("CPI values must be positive")

    result = monthly.copy()
    result["year_month"] = pd.to_datetime(result["year_month"])
    result = result.merge(
        cpi_values, on="year_month", how="left", validate="many_to_one"
    )
    if result["cpi_index"].isna().any():
        missing_months = result.loc[result["cpi_index"].isna(), "year_month"].unique()
        raise ValueError(f"Missing CPI values for months: {missing_months}")

    result["real_amount"] = result["amount"] / result["cpi_index"] * 100
    result = result.sort_values(PANEL_KEYS + ["year_month"]).reset_index(drop=True)
    result["real_amount_mom"] = result.groupby(PANEL_KEYS, sort=False)[
        "real_amount"
    ].pct_change(fill_method=None)
    return result


def build_core_age_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure erosion of each industry's nationally dominant age segment.

    The core age is selected using national transaction counts in the first month.
    Corporate/non-age rows (``x``) are excluded from the share denominator.
    """
    working = frame.copy()
    working["year_month"] = pd.to_datetime(working["year_month"])
    working["age_code"] = working["age_code"].astype("string")
    working = working.loc[working["age_code"].isin(list("123456"))]
    if working.empty:
        raise ValueError("No valid age-coded rows are available")

    first_month = working["year_month"].min()
    last_month = working["year_month"].max()
    national_first = (
        working.loc[working["year_month"] == first_month]
        .groupby(["industry_code", "age_code"], as_index=False)["transactions"]
        .sum()
        .sort_values(
            ["industry_code", "transactions", "age_code"],
            ascending=[True, False, True],
        )
    )
    core_map = (
        national_first.drop_duplicates("industry_code")
        .rename(columns={"age_code": "core_age_code"})[
            ["industry_code", "core_age_code"]
        ]
        .reset_index(drop=True)
    )

    panel_age = (
        working.groupby(PANEL_KEYS + ["year_month", "age_code"], as_index=False)[
            "transactions"
        ].sum()
    ).merge(core_map, on="industry_code", how="left", validate="many_to_one")
    panel_age["core_transactions"] = np.where(
        panel_age["age_code"] == panel_age["core_age_code"],
        panel_age["transactions"],
        0,
    )
    panel_month = panel_age.groupby(
        PANEL_KEYS + ["year_month", "core_age_code"], as_index=False
    ).agg(
        age_coded_transactions=("transactions", "sum"),
        core_transactions=("core_transactions", "sum"),
    )
    panel_month["core_age_share"] = (
        panel_month["core_transactions"] / panel_month["age_coded_transactions"]
    )

    national_month = panel_age.groupby(
        ["industry_code", "year_month", "core_age_code"], as_index=False
    ).agg(
        age_coded_transactions=("transactions", "sum"),
        core_transactions=("core_transactions", "sum"),
    )
    national_month["national_core_age_share"] = (
        national_month["core_transactions"]
        / national_month["age_coded_transactions"]
    )
    panel_month = panel_month.merge(
        national_month[
            [
                "industry_code",
                "year_month",
                "core_age_code",
                "national_core_age_share",
            ]
        ],
        on=["industry_code", "year_month", "core_age_code"],
        how="left",
        validate="many_to_one",
    )

    start = panel_month.loc[panel_month["year_month"] == first_month].set_index(
        PANEL_KEYS
    )
    end = panel_month.loc[panel_month["year_month"] == last_month].set_index(
        PANEL_KEYS
    )
    result = end[["core_age_code", "core_age_share", "national_core_age_share"]].join(
        start[["core_age_share", "national_core_age_share"]],
        how="inner",
        lsuffix="_end",
        rsuffix="_start",
    )
    result["core_age_share_change"] = (
        result["core_age_share_end"] - result["core_age_share_start"]
    )
    result["national_core_age_share_change"] = (
        result["national_core_age_share_end"]
        - result["national_core_age_share_start"]
    )
    result["relative_core_age_share_change"] = (
        result["core_age_share_change"]
        - result["national_core_age_share_change"]
    )
    return result.reset_index()[
        PANEL_KEYS
        + [
            "core_age_code",
            "core_age_share_change",
            "national_core_age_share_change",
            "relative_core_age_share_change",
        ]
    ]


def summarize_panel_stress(
    monthly: pd.DataFrame,
    core_age_metrics: pd.DataFrame,
    minimum_monthly_transactions: int = 100,
) -> pd.DataFrame:
    """Create an interpretable 0--100 consumption-stress score (CSDI)."""
    working = monthly.copy()
    working["year_month"] = pd.to_datetime(working["year_month"])
    expected_months = working["year_month"].nunique()
    complete_index = (
        working.groupby(PANEL_KEYS)["year_month"]
        .nunique()
        .loc[lambda values: values == expected_months]
        .index
    )
    complete = working.set_index(PANEL_KEYS).loc[complete_index].reset_index()
    first_month = complete["year_month"].min()
    last_month = complete["year_month"].max()

    first = complete.loc[complete["year_month"] == first_month].set_index(PANEL_KEYS)
    last = complete.loc[complete["year_month"] == last_month].set_index(PANEL_KEYS)
    summary = last[["industry_name", "amount", "transactions", "average_ticket"]].join(
        first[["amount", "transactions", "average_ticket"]],
        lsuffix="_end",
        rsuffix="_start",
    )

    national = working.groupby(["year_month", "industry_code"], as_index=False).agg(
        amount=("amount", "sum"), transactions=("transactions", "sum")
    )
    national["average_ticket"] = national["amount"] / national["transactions"]
    national_first = national.loc[national["year_month"] == first_month].set_index(
        "industry_code"
    )
    national_last = national.loc[national["year_month"] == last_month].set_index(
        "industry_code"
    )
    national_total = working.groupby("year_month", as_index=False).agg(
        amount=("amount", "sum"), transactions=("transactions", "sum")
    )
    national_total["average_ticket"] = (
        national_total["amount"] / national_total["transactions"]
    )
    total_first = national_total.loc[
        national_total["year_month"] == first_month
    ].iloc[0]
    total_last = national_total.loc[
        national_total["year_month"] == last_month
    ].iloc[0]
    region = working.groupby(
        ["year_month", "province", "district"], as_index=False
    ).agg(amount=("amount", "sum"), transactions=("transactions", "sum"))
    region["average_ticket"] = region["amount"] / region["transactions"]
    region_first = region.loc[region["year_month"] == first_month].set_index(
        ["province", "district"]
    )
    region_last = region.loc[region["year_month"] == last_month].set_index(
        ["province", "district"]
    )
    for column in ["amount", "transactions", "average_ticket"]:
        summary[f"{column}_change"] = (
            summary[f"{column}_end"] / summary[f"{column}_start"] - 1
        )
        national_ratio = (
            national_last[column] / national_first[column]
        ).rename(f"national_{column}_ratio")
        summary = summary.join(national_ratio, on="industry_code")
        summary[f"relative_{column}_change"] = (
            (summary[f"{column}_end"] / summary[f"{column}_start"])
            / summary[f"national_{column}_ratio"]
            - 1
        )
        region_relative_ratio = (
            (region_last[column] / region_first[column])
            / (total_last[column] / total_first[column])
        ).rename(f"region_relative_{column}_ratio")
        summary = summary.join(
            region_relative_ratio, on=["province", "district"]
        )
        summary[f"two_way_relative_{column}_change"] = (
            (summary[f"{column}_end"] / summary[f"{column}_start"])
            / summary[f"national_{column}_ratio"]
            / summary[f"region_relative_{column}_ratio"]
            - 1
        )

    change_rows = complete.loc[complete["year_month"] != first_month].copy()
    amount_path_column = (
        "two_way_relative_amount_mom"
        if "two_way_relative_amount_mom" in change_rows
        else "relative_amount_mom"
    )
    transaction_path_column = (
        "two_way_relative_transactions_mom"
        if "two_way_relative_transactions_mom" in change_rows
        else "relative_transactions_mom"
    )
    change_rows["joint_underperformance"] = (
        (change_rows[amount_path_column] < 0)
        & (change_rows[transaction_path_column] < 0)
    ).astype(float)
    path_metrics = change_rows.groupby(PANEL_KEYS).agg(
        joint_underperformance_rate=("joint_underperformance", "mean"),
        two_way_transaction_volatility=(transaction_path_column, "std"),
    )
    volume = complete.groupby(PANEL_KEYS)["transactions"].agg(
        minimum_monthly_transactions="min",
        total_transactions="sum",
    )
    summary = summary.join(path_metrics).join(volume).reset_index()
    summary = summary.merge(
        core_age_metrics,
        on=PANEL_KEYS,
        how="left",
        validate="one_to_one",
    )
    summary["signal_quality"] = np.where(
        summary["minimum_monthly_transactions"] >= minimum_monthly_transactions,
        "usable",
        "low_volume",
    )
    summary["csdi"] = np.nan

    eligible = summary["signal_quality"] == "usable"
    eligible_rows = summary.loc[eligible]
    stress_inputs = {
        "amount_stress": -eligible_rows["two_way_relative_amount_change"],
        "transaction_stress": -eligible_rows[
            "two_way_relative_transactions_change"
        ],
        "demographic_stress": -eligible_rows["relative_core_age_share_change"],
        "volatility_stress": eligible_rows["two_way_transaction_volatility"],
    }
    for name, values in stress_inputs.items():
        summary.loc[eligible, name] = values.rank(pct=True, method="average").fillna(
            0.5
        )
    summary.loc[eligible, "persistence_stress"] = summary.loc[
        eligible, "joint_underperformance_rate"
    ].fillna(0)
    summary.loc[eligible, "csdi"] = 100 * (
        0.25 * summary.loc[eligible, "amount_stress"]
        + 0.30 * summary.loc[eligible, "transaction_stress"]
        + 0.20 * summary.loc[eligible, "persistence_stress"]
        + 0.15 * summary.loc[eligible, "demographic_stress"]
        + 0.10 * summary.loc[eligible, "volatility_stress"]
    )
    summary["risk_tier"] = "low_signal"
    summary.loc[eligible & (summary["csdi"] < 40), "risk_tier"] = "stable"
    summary.loc[
        eligible & summary["csdi"].between(40, 55, inclusive="left"), "risk_tier"
    ] = "watch"
    summary.loc[
        eligible & summary["csdi"].between(55, 70, inclusive="left"), "risk_tier"
    ] = "warning"
    summary.loc[eligible & (summary["csdi"] >= 70), "risk_tier"] = "high_risk"
    return summary.sort_values("csdi", ascending=False, na_position="last").reset_index(
        drop=True
    )


def fit_kmeans(
    frame: pd.DataFrame,
    feature_columns: list[str],
    k: int,
    seed: int = 42,
    max_iterations: int = 200,
) -> dict[str, object]:
    """Fit deterministic standardized K-Means without an optional ML dependency."""
    values = frame[feature_columns].astype(float)
    if values.isna().any().any():
        raise ValueError("K-Means features must not contain missing values")
    if not 2 <= k < len(values):
        raise ValueError("k must be at least 2 and smaller than the row count")

    raw = values.to_numpy()
    means = raw.mean(axis=0)
    scales = raw.std(axis=0)
    scales[scales == 0] = 1.0
    standardized = (raw - means) / scales
    rng = np.random.default_rng(seed)

    centroid_indices = [int(rng.integers(len(standardized)))]
    while len(centroid_indices) < k:
        chosen = standardized[centroid_indices]
        squared = ((standardized[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2)
        closest = squared.min(axis=1)
        closest[centroid_indices] = 0
        if closest.sum() == 0:
            next_index = next(
                index for index in range(len(standardized)) if index not in centroid_indices
            )
        else:
            next_index = int(rng.choice(len(standardized), p=closest / closest.sum()))
        centroid_indices.append(next_index)

    centroids = standardized[centroid_indices].copy()
    labels = np.zeros(len(standardized), dtype=int)
    for iteration in range(1, max_iterations + 1):
        distances = ((standardized[:, None, :] - centroids[None, :, :]) ** 2).sum(
            axis=2
        )
        next_labels = distances.argmin(axis=1)
        next_centroids = centroids.copy()
        for cluster in range(k):
            members = standardized[next_labels == cluster]
            if len(members):
                next_centroids[cluster] = members.mean(axis=0)
        if np.array_equal(labels, next_labels) and np.allclose(
            centroids, next_centroids
        ):
            labels = next_labels
            centroids = next_centroids
            break
        labels = next_labels
        centroids = next_centroids

    final_distances = (
        (standardized - centroids[labels]) ** 2
    ).sum(axis=1)
    inertia = float(final_distances.sum())
    silhouette = _silhouette_score(standardized, labels)
    original_centroids = centroids * scales + means
    return {
        "labels": labels,
        "centroids": pd.DataFrame(original_centroids, columns=feature_columns),
        "inertia": inertia,
        "silhouette": silhouette,
        "iterations": iteration,
    }


def _silhouette_score(values: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean silhouette coefficient; singleton clusters score zero."""
    squared_norms = (values**2).sum(axis=1)
    squared_distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2 * values @ values.T
    )
    distances = np.sqrt(np.maximum(squared_distances, 0))
    scores = np.zeros(len(values))
    unique_labels = np.unique(labels)
    for row_index, cluster in enumerate(labels):
        same = labels == cluster
        same[row_index] = False
        if not same.any():
            continue
        within = distances[row_index, same].mean()
        other_means = [
            distances[row_index, labels == other].mean()
            for other in unique_labels
            if other != cluster
        ]
        nearest_other = min(other_means)
        denominator = max(within, nearest_other)
        scores[row_index] = (
            (nearest_other - within) / denominator if denominator else 0.0
        )
    return float(scores.mean())
