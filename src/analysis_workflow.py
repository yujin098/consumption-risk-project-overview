"""Public quality, sensitivity and score-trace workflow for normalized inputs.

No file reads or private-data dependencies. Calendar and thresholds are explicit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    build_region_industry_month, add_national_benchmarks, add_two_way_benchmarks,
)
from src.normalize_data import KEY_COLUMNS
from src.risk_model import (
    PANEL_KEYS, SCORE_WEIGHTS, build_core_age_metrics, summarize_panel_stress,
)


def assess_quality(frame, expected_months, minimum_transactions=100):
    """Retain every observed panel; never silently deduplicate financial rows.

    A duplicated demographic key excludes the whole panel from benchmarks and
    scoring. Invalid amounts/counts also exclude it; low volume only defers scoring.
    Panels entirely absent from the input require a separate universe list.
    """
    calendar = pd.DatetimeIndex(expected_months)
    if (len(calendar) < 2 or calendar.hasnans or calendar.has_duplicates
            or not calendar.equals(pd.date_range(calendar.min(), calendar.max(), freq='MS'))):
        raise ValueError('expected_months must be consecutive month starts (at least two)')
    if not np.isfinite(minimum_transactions) or minimum_transactions <= 0:
        raise ValueError('minimum_transactions must be positive')
    required = set(KEY_COLUMNS + ['industry_name', 'amount', 'transactions'])
    if required.difference(frame.columns) or frame.empty:
        raise ValueError('A nonempty normalized input with all required columns is needed')
    work = frame.copy()
    work['year_month'] = pd.to_datetime(work.year_month, errors='coerce')
    if work[KEY_COLUMNS].isna().any().any():
        raise ValueError('Panel keys and dates must not be missing')
    if not work.year_month.isin(calendar).all():
        raise ValueError('Input contains dates outside expected_months')
    work['_duplicate'] = work.duplicated(KEY_COLUMNS, keep=False)
    work['_industry_label'] = work.industry_name.astype('string').str.replace(r'\s+', '', regex=True)
    work['_industry_label_count'] = work.groupby('industry_code')['_industry_label'].transform('nunique')
    for column in ['amount', 'transactions']:
        work[column] = pd.to_numeric(work[column], errors='coerce')
    work['_invalid'] = (
        ~np.isfinite(work.amount) | (work.amount <= 0)
        | ~np.isfinite(work.transactions) | (work.transactions <= 0)
        | work.transactions.mod(1).ne(0)
        | ~work.age_code.astype(str).isin(list('123456'))
    )
    rows = []
    for key, group in work.groupby(PANEL_KEYS, sort=True):
        missing = calendar.difference(pd.DatetimeIndex(group.year_month.unique()))
        duplicates = int(group['_duplicate'].sum())
        invalid = int(group['_invalid'].sum())
        # Global code/name conflicts would split national code/month benchmarks.
        labels = group['_industry_label']
        invalid_labels = labels.isna().any() or labels.eq('').any() or group['_industry_label_count'].gt(1).any()
        reasons = []
        if len(missing):
            reasons.append('missing_months')
        if duplicates:
            reasons.append('duplicate_keys')
        if invalid or invalid_labels:
            reasons.append('invalid_values')
        benchmark_eligible = not reasons
        minimum = group.groupby('year_month').transactions.sum().min()
        if minimum < minimum_transactions:
            reasons.append('low_volume')
        rows.append(dict(zip(PANEL_KEYS, key)) | dict(
            observed_months=group.year_month.nunique(), expected_months=len(calendar),
            missing_months=','.join(missing.strftime('%Y-%m')),
            duplicate_rows=duplicates, invalid_rows=invalid,
            minimum_monthly_transactions=minimum,
            benchmark_eligible=benchmark_eligible, eligible=not reasons,
            exclusion_reason=';'.join(reasons),
        ))
    return pd.DataFrame(rows)


def run_analysis(frame, expected_months, thresholds=(100, 50, 200), top_n=3):
    """Recompute percentile scores per threshold; compare with the first scenario.

    Weight comparisons share eligibility. Threshold comparisons may change both
    the scored population and its percentile reference distribution. Benchmarks
    include structurally valid low-volume panels under every scenario.
    """
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError('Provide unique positive thresholds; first is the baseline')
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError('top_n must be a positive integer')
    qualities = [assess_quality(frame, expected_months, threshold) for threshold in thresholds]
    quality = qualities[0]
    clean = frame.merge(quality.loc[quality.benchmark_eligible, PANEL_KEYS], on=PANEL_KEYS)
    for column in ['amount', 'transactions']:
        clean[column] = pd.to_numeric(clean[column])
    monthly = pd.DataFrame()
    core = pd.DataFrame()
    if not clean.empty:
        monthly = add_two_way_benchmarks(add_national_benchmarks(build_region_industry_month(clean)))
        core = build_core_age_metrics(clean)

    scenarios = []
    for threshold, check in zip(thresholds, qualities):
        # The older score function assumes at least one usable panel.
        detail = None
        if check.eligible.any():
            detail = summarize_panel_stress(monthly, core, threshold)
        for name, weights in [('baseline', SCORE_WEIGHTS), ('equal', dict.fromkeys(SCORE_WEIGHTS, .2))]:
            scored = check.copy()
            if detail is not None:
                # Keep quality status from the audit, and score/trace fields from the model.
                extra = [c for c in detail if c not in scored or c in PANEL_KEYS]
                scored = scored.merge(detail[extra], on=PANEL_KEYS, how='left', validate='one_to_one')
            for component in SCORE_WEIGHTS:
                if component not in scored:
                    scored[component] = np.nan
            scored['csdi'] = 100 * sum(scored[c] * w for c, w in weights.items())
            scored.loc[~scored.eligible, 'csdi'] = np.nan
            # Do not retain a baseline tier after reweighting.
            scored = scored.drop(columns=['risk_tier', 'signal_quality'], errors='ignore')
            scored['signal_quality'] = np.where(scored.eligible, 'usable', scored.exclusion_reason)
            scored['scenario'] = f'{name}_{threshold}'
            scored['minimum_transactions_threshold'] = threshold
            scored['rank'] = scored.csdi.rank(ascending=False, method='min')
            scored = scored.sort_values(['csdi'] + PANEL_KEYS, ascending=[False, True, True, True], na_position='last')
            scenarios.append(scored)
    scores = pd.concat(scenarios, ignore_index=True)
    reference = scenarios[0].loc[lambda x: x.csdi.notna()].set_index(PANEL_KEYS)
    reference_top = set(reference.head(top_n).index)
    comparisons = []
    for scenario in scenarios:
        usable = scenario.loc[scenario.csdi.notna()].set_index(PANEL_KEYS)
        selected = set(usable.head(top_n).index)
        common = reference.index.intersection(usable.index)
        rank_change = (usable.loc[common, 'rank'] - reference.loc[common, 'rank']).abs()
        comparisons.append(dict(
            scenario=scenario.scenario.iloc[0], eligible_count=len(usable),
            excluded_count=len(scenario) - len(usable), requested_top_n=top_n,
            selected_count=len(selected), reference_selected_count=len(reference_top),
            retained_count=len(selected & reference_top),
            entered_count=len(selected - reference_top), exited_count=len(reference_top - selected),
            top_overlap_rate=len(selected & reference_top) / len(reference_top) if reference_top else np.nan,
            common_eligible_count=len(common),
            mean_absolute_rank_change=rank_change.mean() if len(common) else np.nan,
        ))

    walkthrough = None
    if not reference.empty:
        key = reference.index[0]
        row = reference.iloc[0]
        selected_monthly = monthly.loc[
            monthly[PANEL_KEYS].eq(pd.Series(key, index=PANEL_KEYS)).all(axis=1)
        ].sort_values('year_month')
        contributions = pd.DataFrame([
            dict(component=c, normalized_value=row[c], weight=w, contribution=100 * row[c] * w)
            for c, w in SCORE_WEIGHTS.items()
        ])
        walkthrough = dict(panel=dict(zip(PANEL_KEYS, key)), score=float(row.csdi),
                           monthly=selected_monthly, contributions=contributions, endpoint=row.to_dict())
    return dict(quality=quality, monthly=monthly, scores=scores,
                comparison=pd.DataFrame(comparisons), walkthrough=walkthrough)
