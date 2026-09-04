import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class SyntheticDemoTests(unittest.TestCase):
    def test_demo_scores_artificial_panels_without_private_inputs(self):
        # Removing the generator or low-volume handling must fail this test.
        self.assertIsNotNone(importlib.util.find_spec('src.synthetic_demo'))
        from src.synthetic_demo import run_demo

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'demo'
            run_demo(target)
            panel = pd.read_csv(target / 'synthetic_panel_scores.csv')
            self.assertEqual(len(panel), 14)
            self.assertTrue(panel.province.str.startswith('가상').all())
            self.assertEqual(panel.signal_quality.eq('usable').sum(), 11)
            low = panel.loc[panel.signal_quality.eq('low_volume')]
            self.assertEqual(len(low), 1)
            self.assertTrue(low.csdi.isna().all())
            self.assertTrue(panel.loc[panel.signal_quality.eq('usable'), 'csdi'].between(0, 100).all())
            excluded = panel.set_index('district')
            self.assertIn('duplicate_keys', excluded.loc['가상중복구', 'exclusion_reason'])
            self.assertIn('missing_months', excluded.loc['가상누락구', 'exclusion_reason'])
            self.assertTrue(excluded.loc[['가상중복구', '가상누락구'], 'csdi'].isna().all())
            quality = pd.read_csv(target / 'quality_report.csv')
            comparison = pd.read_csv(target / 'sensitivity_comparison.csv')
            self.assertEqual(len(quality), 14)
            self.assertEqual(comparison.set_index('scenario').loc['baseline_100', 'eligible_count'], 11)
            self.assertEqual(comparison.set_index('scenario').loc['baseline_200', 'eligible_count'], 10)
            contributions = pd.read_csv(target / 'walkthrough_contributions.csv')
            self.assertAlmostEqual(contributions.contribution.sum(), panel.csdi.max())
            monthly = pd.read_csv(target / 'walkthrough_monthly.csv')
            self.assertEqual(len(monthly), 6)
            self.assertTrue((target / 'REPORT.md').stat().st_size > 0)
            self.assertIn('SYNTHETIC', (target / 'NOTICE.txt').read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):
                run_demo(target)
