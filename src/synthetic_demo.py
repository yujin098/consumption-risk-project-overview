"""Artificial, non-calibrated demo; never reads contest or external data."""
import argparse
from pathlib import Path

import pandas as pd

from src.normalize_data import normalize_consumption
from src.analysis_workflow import run_analysis
from src.demo_report import render_report


def run_demo(output: Path) -> None:
    """Generate invented panels and run the existing score functions, not K-means."""
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for region in range(4):
        for industry in range(3):
            for month in range(6):
                for age in [3, 4]:
                    # Invented trends; no fitted parameters or sampled real records.
                    base = 8 if (region, industry) == (3, 2) else (60 if (region, industry) == (2, 2) else 250)
                    trend = 1 + (region - industry) * month * 0.025
                    count = max(1, round(base * trend * (1 if age == 3 else 0.7)))
                    rows.append({
                        'STRD_YYMM': 202601 + month,
                        'SIDO_NM': '가상도', 'CCG_NM': f'가상구{region + 1}',
                        'GENDER_CD': '1', 'AGE_CD': str(age),
                        'TP_BUZ_NO': 90001 + industry,
                        'TP_BUZ_NM': f'가상업종{industry + 1}',
                        'amt': count * (10000 + 100 * industry * month), 'cnt': count,
                    })
    raw = pd.DataFrame(rows)
    # Two intentionally faulty panels illustrate review instead of silent removal.
    duplicate = raw.loc[(raw.CCG_NM == '가상구1') & (raw.TP_BUZ_NO == 90001)].copy()
    duplicate['CCG_NM'] = '가상중복구'
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
    missing = raw.loc[(raw.CCG_NM == '가상구1') & (raw.TP_BUZ_NO == 90001) & (raw.STRD_YYMM != 202603)].copy()
    missing['CCG_NM'] = '가상누락구'
    raw = pd.concat([raw, duplicate, missing], ignore_index=True)
    normalized = normalize_consumption(raw)
    result = run_analysis(normalized, pd.date_range('2026-01-01', periods=6, freq='MS'))
    monthly = result['monthly']
    panel = result['scores'].loc[result['scores'].scenario == 'baseline_100']
    raw.to_csv(output / 'synthetic_input.csv', index=False, encoding='utf-8-sig')
    monthly.to_csv(output / 'synthetic_monthly.csv', index=False, encoding='utf-8-sig')
    panel.to_csv(output / 'synthetic_panel_scores.csv', index=False, encoding='utf-8-sig')
    for name, key in [('quality_report', 'quality'), ('sensitivity_comparison', 'comparison'), ('scenario_scores', 'scores')]:
        result[key].to_csv(output / f'{name}.csv', index=False, encoding='utf-8-sig')
    trace = result['walkthrough']
    if trace is not None:
        trace['monthly'].to_csv(output / 'walkthrough_monthly.csv', index=False, encoding='utf-8-sig')
        trace['contributions'].to_csv(output / 'walkthrough_contributions.csv', index=False, encoding='utf-8-sig')
    (output / 'REPORT.md').write_text(render_report(result), encoding='utf-8')
    (output / 'NOTICE.txt').write_text(
        'SYNTHETIC ONLY — 가상데이터 실행 예제. 실제 소비·신용 위험 결과가 아닙니다.\n'
        'No contest data, CPI, K-means, predictive validation or real regional estimates.\n',
        encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run synthetic-only CSDI demo')
    parser.add_argument('output', type=Path, help='New directory; must not exist')
    run_demo(parser.parse_args().output)
