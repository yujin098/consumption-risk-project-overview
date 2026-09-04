"""Readable report of artificial examples; no external rendering dependency."""
import pandas as pd


def markdown_table(frame):
    """Render small result tables without requiring tabulate."""
    def cell(value):
        if pd.isna(value):
            return '—'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(cell(v) for v in row) + ' |'
                 for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def render_report(result):
    quality = result['quality']
    comparison = result['comparison']
    sections = [
        '# 가상데이터로 따라가는 소비 점검 분석',
        '이 보고서의 지역·업종·금액·건수·점수는 모두 인공 예제다. 실제 BC카드 분석 결과가 아니다.',
        '생성 명령: `python -m src.synthetic_demo outputs/demo`\n\n'
        '분석 기간은 2026년 1~6월로 명시했다. 품질점검 → 기준 비교 → 한 대상의 점수 계산 순서로 읽는다.',
        '## 1. 어떤 대상을 계산하고, 왜 제외했는가',
        f'관측된 지역·업종 {len(quality)}개 중 기본 기준에서 {int(quality.eligible.sum())}개에 점수를 계산했다. '
        '제외 대상도 아래 표와 점수 파일에 남긴다.',
        markdown_table(quality[['district', 'industry_code', 'observed_months',
                                'minimum_monthly_transactions', 'exclusion_reason']].rename(columns={
            'district': '가상 지역', 'industry_code': '업종', 'observed_months': '관측 월 수',
            'minimum_monthly_transactions': '월 최소 건수', 'exclusion_reason': '제외 이유'})),
        '`missing_months`: 지정 기간 누락 · `duplicate_keys`: 동일 집계 키 중복 · '
        '`low_volume`: 월 최소 건수 미달 · `invalid_values`: 값 또는 연령코드·업종명 점검 실패. '
        '빈 제외 이유는 점수 산출 가능을 뜻한다. 중복은 임의로 합치거나 삭제하지 않고 해당 조합을 보류한다.',
        '## 2. 기준을 바꾸면 무엇이 달라지는가',
        '`baseline`: 금액 25%·건수 30%·동반부진 20%·연령 구성비 15%·변동성 10%. '
        '`equal`: 다섯 항목에 각각 20%. 이름 뒤 숫자는 매월 최소 거래 건수다.',
        markdown_table(comparison[['scenario', 'eligible_count', 'retained_count',
                                   'entered_count', 'exited_count', 'mean_absolute_rank_change']].rename(columns={
            'scenario': '설정', 'eligible_count': '산출 대상', 'retained_count': '상위 3 유지',
            'entered_count': '진입', 'exited_count': '이탈', 'mean_absolute_rank_change': '공통 대상 평균 순위 변화'})),
        '모든 비교의 기준은 `baseline_100`이다. 상위 목록의 동점은 지역·업종 키 순서로 정리한다. '
        '순위는 동점에 같은 최상위 순위를 부여한다. 거래량 기준을 바꾸면 대상과 백분위 기준 분포도 함께 바뀐다. '
        '가중치 비교는 같은 대상 안의 비중만 바꾼다. 목록 유지율은 예측력이나 최적성의 증거가 아니다.',
    ]
    trace = result['walkthrough']
    if trace is None:
        sections += ['## 3. 한 대상의 점수 따라가기', '기본 설정의 산출 대상이 없어 계산 예제를 만들지 않았다.']
    else:
        row = trace['endpoint']
        monthly = trace['monthly']
        raw = monthly[['year_month', 'amount', 'transactions', 'average_ticket']].copy()
        raw['year_month'] = raw.year_month.dt.strftime('%Y-%m')
        sections += [
            '## 3. 한 대상의 점수 따라가기',
            f"기본 설정에서 점수가 가장 높은 **{trace['panel']['district']} · 업종 {trace['panel']['industry_code']}**를 예제로 골랐다.",
            '### 월별 입력을 먼저 확인',
            markdown_table(raw.rename(columns={'year_month': '월', 'amount': '금액',
                                               'transactions': '건수', 'average_ticket': '건당 금액'})),
            '건당 금액 = 이용금액 ÷ 이용건수. 이용건수는 고객 수가 아니다.',
            '### 기간 처음과 끝의 상대변화를 계산',
            f"금액 변화율 = {row['amount_end']:.0f} ÷ {row['amount_start']:.0f} − 1 = {row['amount_change']:.6f}\n\n"
            f"전국 동일 업종 성장배율 = {row['national_amount_ratio']:.6f}\n\n"
            f"지역 상대 성장배율 = 지역 전체 성장배율 ÷ 입력 전체 성장배율 = {row['region_relative_amount_ratio']:.6f}\n\n"
            f"이중 상대 금액 변화 = ({row['amount_end']:.0f} ÷ {row['amount_start']:.0f}) ÷ "
            f"{row['national_amount_ratio']:.6f} ÷ {row['region_relative_amount_ratio']:.6f} − 1 = {row['two_way_relative_amount_change']:.6f}",
            '절대 금액이 증가했어도 비교 집단보다 성장 폭이 작으면 상대변화는 음수가 될 수 있다. '
            '이 예제의 높은 점수를 실제 매출 감소나 부실로 해석하면 안 된다.',
            '같은 방식으로 건수 상대변화를 계산한다. 월별 상대변화로 반복 부진과 변동성을 계산하며, '
            '연령 구성비는 업종별 첫 달 최다 결제 연령층을 기준으로 처음과 끝을 비교한다.',
            markdown_table(pd.DataFrame([
                {'점수 입력': '금액 이중 상대변화', '값': row['two_way_relative_amount_change']},
                {'점수 입력': '건수 이중 상대변화', '값': row['two_way_relative_transactions_change']},
                {'점수 입력': '월별 금액·건수 동반부진 비율', '값': row['joint_underperformance_rate']},
                {'점수 입력': '핵심 연령 구성비의 상대변화', '값': row['relative_core_age_share_change']},
                {'점수 입력': '월별 상대 건수 변화의 표준편차', '값': row['two_way_transaction_volatility']},
            ])),
            '### 구성요소를 합쳐 최종 점수를 계산',
            '금액·건수·연령 변화는 부호를 뒤집고, 변동성은 그대로 산출 대상 내 백분위로 바꾼다. '
            '동반부진은 관측 변화 구간의 비율을 사용한다. 백분위는 전체 비교 대상이 있어야 계산할 수 있다.',
            markdown_table(trace['contributions'].rename(columns={'component': '구성요소',
                'normalized_value': '정규화 값', 'weight': '가중치', 'contribution': '점수 기여'})),
            f"각 기여 = 정규화 값 × 가중치 × 100. 합계 **{trace['score']:.4f}점**. "
            '표시는 반올림했으며 계산은 원래 정밀도를 사용한다. 점수는 부도확률이 아니다.',
        ]
    sections += ['## 확인할 파일',
                 '`quality_report.csv`: 전 대상의 품질·제외 이유\n\n'
                 '`sensitivity_comparison.csv` / `scenario_scores.csv`: 설정별 비교와 전체 점수\n\n'
                 '`walkthrough_monthly.csv` / `walkthrough_contributions.csv`: 위 예제의 월별 계산과 기여도\n\n'
                 '보고서의 파일은 실행한 출력 폴더에서 찾을 수 있다. 실제 자료의 요약 수치는 별도 분석 과정 문서에 있다.']
    return '\n\n'.join(sections) + '\n'
