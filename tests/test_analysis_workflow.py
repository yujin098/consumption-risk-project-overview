import unittest

import pandas as pd

from src.analysis_workflow import assess_quality, run_analysis


def sample():
    rows = []
    for district, amounts, counts in [
        ('가상A', [10000, 9000, 8000], [100, 90, 80]),
        ('가상B', [10000, 11000, 12000], [100, 110, 120]),
        ('가상C', [1000, 1000, 1000], [10, 10, 10]),
    ]:
        for month, amount, count in zip(pd.date_range('2026-01-01', periods=3, freq='MS'), amounts, counts):
            rows.append(dict(year_month=month, province='가상도', district=district,
                             industry_code=1, industry_name='가상업종', gender_code='1',
                             age_code='3', amount=amount, transactions=count))
    return pd.DataFrame(rows)


MONTHS = pd.date_range('2026-01-01', periods=3, freq='MS')


class QualityTests(unittest.TestCase):
    def test_missing_and_duplicate_panels_are_retained_with_reasons(self):
        frame = sample()
        frame = frame.loc[~((frame.district == '가상A') & (frame.year_month == MONTHS[1]))]
        frame = pd.concat([frame, frame.loc[frame.district == '가상B'].iloc[[0]]], ignore_index=True)
        quality = assess_quality(frame, MONTHS, 50).set_index('district')
        self.assertEqual(len(quality), 3)
        self.assertIn('missing_months', quality.loc['가상A', 'exclusion_reason'])
        self.assertEqual(quality.loc['가상A', 'missing_months'], '2026-02')
        self.assertIn('duplicate_keys', quality.loc['가상B', 'exclusion_reason'])
        self.assertEqual(quality.loc['가상C', 'exclusion_reason'], 'low_volume')

    def test_whole_month_missing_is_detected_against_explicit_calendar(self):
        frame = sample().loc[lambda x: x.year_month != MONTHS[1]]
        quality = assess_quality(frame, MONTHS, 50)
        self.assertTrue(quality.missing_months.eq('2026-02').all())
        self.assertFalse(quality.eligible.any())

    def test_conflicting_duplicate_keys_are_not_silently_summed(self):
        frame = sample()
        extra = frame.iloc[[0]].copy()
        extra['amount'] = 99999
        quality = assess_quality(pd.concat([frame, extra]), MONTHS, 50)
        self.assertFalse(quality.loc[quality.district == '가상A', 'eligible'].item())

    def test_zero_transactions_are_excluded_before_division(self):
        frame = sample()
        frame.loc[0, 'transactions'] = 0
        quality = assess_quality(frame, MONTHS, 50).set_index('district')
        self.assertIn('invalid_values', quality.loc['가상A', 'exclusion_reason'])

    def test_conflicting_industry_names_across_regions_are_reported_not_crashed(self):
        frame = sample()
        frame.loc[frame.district == '가상A', 'industry_name'] = '다른업종명'
        result = run_analysis(frame, MONTHS, thresholds=(50,))
        self.assertEqual(len(result['quality']), 3)
        self.assertTrue(result['quality'].exclusion_reason.str.contains('invalid_values').all())
        self.assertTrue(result['scores'].csdi.isna().all())


class WorkflowTests(unittest.TestCase):
    def test_weight_and_threshold_comparisons_match_independent_arithmetic(self):
        result = run_analysis(sample(), MONTHS, thresholds=(50, 100), top_n=1)
        baseline = result['scores'].query("scenario == 'baseline_50'")
        equal = result['scores'].query("scenario == 'equal_50'")
        weights = [.25, .30, .20, .15, .10]
        columns = ['amount_stress', 'transaction_stress', 'persistence_stress',
                   'demographic_stress', 'volatility_stress']
        for _, row in baseline.loc[baseline.csdi.notna()].iterrows():
            self.assertAlmostEqual(row.csdi, sum(row[c] * w * 100 for c, w in zip(columns, weights)))
        for _, row in equal.loc[equal.csdi.notna()].iterrows():
            self.assertAlmostEqual(row.csdi, sum(row[c] for c in columns) * 20)
        self.assertEqual(result['comparison'].set_index('scenario').loc['baseline_50', 'eligible_count'], 2)
        self.assertEqual(result['comparison'].set_index('scenario').loc['baseline_100', 'eligible_count'], 1)
        self.assertAlmostEqual(result['walkthrough']['contributions'].contribution.sum(), result['walkthrough']['score'])
        self.assertEqual(len(result['walkthrough']['monthly']), 3)
        trace = result['walkthrough']['endpoint']
        self.assertAlmostEqual(trace['amount_change'], trace['amount_end'] / trace['amount_start'] - 1)
        self.assertAlmostEqual(trace['two_way_relative_amount_change'],
                               (trace['amount_end'] / trace['amount_start']) /
                               trace['national_amount_ratio'] / trace['region_relative_amount_ratio'] - 1)

    def test_all_excluded_produces_quality_rows_and_no_fabricated_scores(self):
        frame = sample().loc[lambda x: x.year_month != MONTHS[1]]
        result = run_analysis(frame, MONTHS, thresholds=(50,), top_n=1)
        self.assertEqual(len(result['quality']), 3)
        self.assertTrue(result['scores'].csdi.isna().all())
        self.assertTrue(result['comparison'].eligible_count.eq(0).all())
        self.assertTrue(result['comparison'].top_overlap_rate.isna().all())
        self.assertIsNone(result['walkthrough'])

    def test_duplicate_panel_is_in_output_but_not_benchmarks(self):
        frame = sample()
        extra = frame.iloc[[0]].copy()
        extra['amount'] = 999999
        result = run_analysis(pd.concat([frame, extra]), MONTHS, thresholds=(50,))
        rows = result['scores'].query("district == '가상A'")
        self.assertTrue(rows.csdi.isna().all())
        self.assertTrue(rows.exclusion_reason.str.contains('duplicate_keys').all())
        self.assertFalse(result['monthly'].district.eq('가상A').any())
