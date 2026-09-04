"""Artificial, non-calibrated demo; never reads contest or external data."""
import argparse
from pathlib import Path

import pandas as pd

from src.normalize_data import normalize_consumption
from src.features import (
    build_region_industry_month, add_national_benchmarks, add_two_way_benchmarks,
)
from src.risk_model import build_core_age_metrics, summarize_panel_stress


def run_demo(output: Path) -> None:
    """Generate invented panels and run the existing score functions, not K-means."""
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for region in range(4):
        for industry in range(3):
            for month in range(6):
                for age in [3, 4]:
                    # Invented trends; no fitted parameters or sampled real records.
                    base = 8 if (region, industry) == (3, 2) else 250
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
    normalized = normalize_consumption(raw)
    monthly = add_two_way_benchmarks(add_national_benchmarks(
        build_region_industry_month(normalized)))
    panel = summarize_panel_stress(monthly, build_core_age_metrics(normalized))
    raw.to_csv(output / 'synthetic_input.csv', index=False, encoding='utf-8-sig')
    monthly.to_csv(output / 'synthetic_monthly.csv', index=False, encoding='utf-8-sig')
    panel.to_csv(output / 'synthetic_panel_scores.csv', index=False, encoding='utf-8-sig')
    (output / 'NOTICE.txt').write_text(
        'SYNTHETIC ONLY — 가상데이터 실행 예제. 실제 소비·신용 위험 결과가 아닙니다.\n'
        'No contest data, CPI, K-means, predictive validation or real regional estimates.\n',
        encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run synthetic-only CSDI demo')
    parser.add_argument('output', type=Path, help='New directory; must not exist')
    run_demo(parser.parse_args().output)
