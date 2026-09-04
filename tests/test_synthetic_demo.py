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
            self.assertEqual(len(panel), 12)
            self.assertTrue(panel.province.str.startswith('가상').all())
            self.assertEqual(panel.signal_quality.eq('usable').sum(), 11)
            low = panel.loc[panel.signal_quality.eq('low_volume')]
            self.assertEqual(len(low), 1)
            self.assertTrue(low.csdi.isna().all())
            self.assertTrue(panel.loc[panel.signal_quality.eq('usable'), 'csdi'].between(0, 100).all())
            self.assertIn('SYNTHETIC', (target / 'NOTICE.txt').read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):
                run_demo(target)
