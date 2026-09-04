# CSDI · 소비 변화로 점검 우선순위 설명하기

지역·업종별 소비 흐름에서 추가 확인할 대상을 정리하는 분석 프로젝트입니다.
금액과 건수를 나눠 보고, 전국 동일 업종과 지역 전체 흐름을 비교한 뒤
여러 신호를 CSDI 점수로 정리합니다.

**가상데이터로 품질점검 → 기준 비교 → 개별 점수 계산을 직접 따라갈 수 있습니다.**
실제 분석의 판단 과정은 별도 문서에서 설명합니다.

## 먼저 읽을 것

| 궁금한 점 | 읽을 문서 |
| --- | --- |
| 무엇을 위해 시작했나? | [프로젝트 소개](docs/PROJECT_OVERVIEW.md) |
| 왜 이 분석 방식을 선택했나? | [분석 과정과 실제 요약 결과](docs/ANALYSIS_STORY.md) |
| 실행하면 무엇을 볼 수 있나? | [가상데이터 결과 예시](docs/DEMO_RESULTS.md) |
| 계산을 어떻게 설명하나? | [계산과 질문](docs/EXPLAIN_IT.md) |
| 어디까지 검증할 수 있나? | [공개 범위와 한계](docs/SCOPE.md) |

## 바로 실행하기

Python 3.12에서 저장소 폴더를 열고 실행합니다.

```text
python -m pip install -r requirements.txt
python -m src.synthetic_demo outputs/demo
python -m unittest discover -s tests -v
```

`outputs/demo`는 새 폴더여야 합니다. 재실행할 때는 다른 출력 이름을 지정하세요.
완료 후 **`outputs/demo/REPORT.md`**를 읽으면 됩니다. 원본 데이터나 외부 API는 필요하지 않습니다.

## 실행으로 확인하는 세 가지

1. **품질점검:** 14개 가상 지역·업종 중 기본 기준에서 11개를 계산합니다.
   중복·기간 누락·저거래량 3개도 제외 이유와 함께 남깁니다.
2. **기준 비교:** 기본·동일 가중치와 월 최소 거래량 50·100·200건을 조합한
   6개 설정에서 대상 수, 상위 목록, 순위 변화를 비교합니다.
3. **점수 설명:** 한 대상의 월별 금액·건수에서 상대변화, 구성요소, 최종 점수까지 보여줍니다.

| 결과 파일 | 내용 |
| --- | --- |
| `REPORT.md` | 순서대로 읽는 결과 보고서 |
| `quality_report.csv` | 모든 관측 대상의 품질·제외 이유 |
| `synthetic_panel_scores.csv` | 기본 설정의 점수와 제외 대상 |
| `sensitivity_comparison.csv` | 설정별 대상 수·상위 목록 유지·순위 변화 |
| `scenario_scores.csv` | 6개 설정의 전체 점수·순위 |
| `walkthrough_monthly.csv`, `walkthrough_contributions.csv` | 개별 대상의 계산 과정 |
| `synthetic_input.csv`, `synthetic_monthly.csv` | 가상 입력과 품질점검 후 월별 지표 |

## 코드 읽는 순서

| 파일 | 역할 |
| --- | --- |
| [synthetic_demo.py](src/synthetic_demo.py) | 가상 사례 생성과 전체 실행 |
| [normalize_data.py](src/normalize_data.py) | 입력 컬럼·날짜·키 정리 |
| [analysis_workflow.py](src/analysis_workflow.py) | 품질점검, 설정별 재계산, 순위 비교, 개별 계산 추출 |
| [features.py](src/features.py) | 금액·건수·건당 금액 및 비교 기준 |
| [risk_model.py](src/risk_model.py) | 핵심 연령 구성비, 점수 산식, CPI·군집 보조 함수 |
| [demo_report.py](src/demo_report.py) | 계산 결과를 읽기 쉬운 보고서로 구성 |

점수는 부도확률이나 검증된 신용등급이 아닙니다. 가상 예제는 계산 흐름을 확인하는 용도입니다.
원본 데이터·상세 결과표의 공개 범위, 실제 결과 재현의 한계, 본인 역할과 AI 활용은
[공개 범위와 한계](docs/SCOPE.md)에 모았습니다.
