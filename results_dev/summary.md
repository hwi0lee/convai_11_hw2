# 평가 결과

시나리오 30개 × (초기 계획 + 후속 발화 2턴). 모든 지표는 **gold 상태** 기준으로 자동 채점.


## 전체 (모든 턴)

| 지표 | ours | baseline | baseline_cot |
|---|---|---|---|
| 제약 완전 준수 계획 비율 ↑ | 93.3% | 10.0% | 28.9% |
| 일정 충돌 블록 비율 ↓ | 0.0% | 6.0% | 6.7% |
| 마감 위반 블록 비율 ↓ | 0.0% | 1.5% | 1.7% |
| 하루 최대시간 초과 일수(평균) ↓ | 0.00 | 1.38 | 0.34 |
| 필요 작업시간 커버율(실현가능 케이스) ↑ | 98.8% | 83.7% | 68.4% |
| 완전·유효 계획 비율(실현가능 케이스) ↑ | 92.1% | 6.3% | 23.8% |
| 과잉 배치(분, 평균) ↓ | 0.00 | 290.67 | 3.33 |

## 초기 계획 (turn 0)

| 지표 | ours | baseline | baseline_cot |
|---|---|---|---|
| 제약 완전 준수 계획 비율 ↑ | 100.0% | 10.0% | 36.7% |
| 일정 충돌 블록 비율 ↓ | 0.0% | 4.8% | 4.8% |
| 마감 위반 블록 비율 ↓ | 0.0% | 0.2% | 0.8% |
| 하루 최대시간 초과 일수(평균) ↓ | 0.00 | 1.47 | 0.40 |
| 필요 작업시간 커버율(실현가능 케이스) ↑ | 100.0% | 89.6% | 69.7% |
| 완전·유효 계획 비율(실현가능 케이스) ↑ | 100.0% | 9.5% | 28.6% |
| 과잉 배치(분, 평균) ↓ | 0.00 | 308.00 | 5.00 |

## 재조정 (후속 발화 turn 1-2)

| 지표 | ours | baseline | baseline_cot |
|---|---|---|---|
| 제약 완전 준수 계획 비율 ↑ | 90.0% | 10.0% | 25.0% |
| 일정 충돌 블록 비율 ↓ | 0.0% | 6.6% | 7.6% |
| 마감 위반 블록 비율 ↓ | 0.0% | 2.1% | 2.2% |
| 하루 최대시간 초과 일수(평균) ↓ | 0.00 | 1.33 | 0.32 |
| 필요 작업시간 커버율(실현가능 케이스) ↑ | 98.2% | 80.8% | 67.7% |
| 완전·유효 계획 비율(실현가능 케이스) ↑ | 88.1% | 4.8% | 21.4% |
| 과잉 배치(분, 평균) ↓ | 0.00 | 282.00 | 2.50 |

## 재조정 품질

| 지표 | ours | baseline | baseline_cot |
|---|---|---|---|
| 계획 안정성 (유효한 기존 블록 유지율) ↑ | 94.4% | 85.0% | 73.9% |
| 설명 사실 일치율 (LLM judge) ↑ | 78.7% | 66.4% | 65.4% |
| 응답 지연(초, 평균) | 2.47 | 4.11 | 15.95 |

## 실현 불가능 탐지 (positive = 불가능)

| 시스템 | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| ours | 100.0% | 100.0% | 100.0% | 100.0% |
| baseline | 55.6% | 74.1% | 63.5% | 74.4% |
| baseline_cot | 100.0% | 37.0% | 54.1% | 81.1% |

## 후속 발화 유형별 제약 완전 준수 비율

| 유형 | n | ours | baseline | baseline_cot |
|---|---|---|---|---|
| add | 13 | 85% | 15% | 38% |
| block_evening | 4 | 100% | 25% | 0% |
| block_range | 6 | 100% | 0% | 33% |
| deadline | 5 | 100% | 20% | 20% |
| missed | 6 | 100% | 17% | 17% |
| more | 7 | 100% | 0% | 29% |
| partial | 11 | 100% | 0% | 18% |
| remove | 8 | 50% | 12% | 25% |

## 발화 → 상태 변경 파싱 정확도 (ours)

상태가 gold와 정확히 일치한 턴: **85.0%** (51/60)

| 유형 | 정확도 |
|---|---|
| add | 85% (11/13) |
| block_evening | 100% (4/4) |
| block_range | 100% (6/6) |
| deadline | 80% (4/5) |
| missed | 100% (6/6) |
| more | 86% (6/7) |
| partial | 91% (10/11) |
| remove | 50% (4/8) |

## 오류 분석 (위반 유형별 블록 수 합계)

| 위반 유형 | ours | baseline | baseline_cot |
|---|---|---|---|
| conflict | 0 | 54 | 50 |
| after_deadline | 0 | 21 | 17 |
| outside_window | 0 | 118 | 82 |
| in_past | 0 | 0 | 0 |
| self_overlap | 0 | 2 | 0 |
| unknown_task | 16 | 1 | 0 |
| malformed | 0 | 0 | 0 |
| cap_exceeded_days | 0 | 124 | 31 |

### ours 파싱 오류 유형

extra_task: 6, remaining: 2, deadline: 1

### ours 실패 사례

- **S05 (remove)** “발표 준비는 이번 주에 안 하기로 했어. 빼줘.” → ops=[('update_task', {'name': '발표 준비', 'duration_hours': 0})] / 오류=['extra_task:발표 준비']
- **S07 (remove)** “논문 읽기는 이번 주에 안 하기로 했어. 빼줘.” → ops=[('update_task', {'name': '논문 읽기', 'duration_hours': 0})] / 오류=['extra_task:논문 읽기']
- **S07 (add)** “일요일까지 블로그 글 작성 3시간짜리 추가해줘.” → ops=[('add_task', {'name': '블로그 글 작성', 'duration_hours': 3, 'deadline': '2026-11-15 23:59'})] / 오류=['extra_task:논문 읽기']
- **S08 (deadline)** “코딩 테스트 연습 마감이 수요일로 당겨졌어.” → ops=[('update_task', {'name': '코딩 테스트 연습', 'deadline': '2026-11-11 18:00'})] / 오류=['deadline:코딩 테스트 연습:11-11 18:00!=11-11 23:59']
- **S16 (remove)** “중간고사 공부는 이번 주에 안 하기로 했어. 빼줘.” → ops=[('report_missed', {'date': '2026-11-09', 'task_name': '중간고사 공부'})] / 오류=['extra_task:중간고사 공부']
- **S26 (partial)** “오늘 프로젝트 기획서는 1.5시간 했고 나머지는 못 했어.” → ops=[('report_progress', {'task_name': '프로젝트 기획서', 'done_hours': 1.5, 'date': '2026-11-09'}), ('report_missed', {'task_name': '프로젝트 기획서', 'date': '2026-11-09'})] / 오류=['remaining:영어 회화 준비:0!=300']
- **S26 (more)** “중간고사 공부 생각보다 오래 걸릴 것 같아. 2시간 더 필요해.” → ops=[('update_task', {'name': '중간고사 공부', 'add_hours': 2})] / 오류=['remaining:영어 회화 준비:0!=300']
- **S29 (remove)** “알고리즘 과제는 이번 주에 안 하기로 했어. 빼줘.” → ops=[('report_missed', {'date': '2026-11-09', 'task_name': '알고리즘 과제'})] / 오류=['extra_task:알고리즘 과제']
- **S29 (add)** “금요일까지 포트폴리오 정리 2시간짜리 추가해줘.” → ops=[('add_task', {'name': '포트폴리오 정리', 'duration_hours': 2, 'deadline': '2026-11-13 23:59', 'priority': 2})] / 오류=['extra_task:알고리즘 과제']

제약 위반이 발생한 ours 턴: 6개 [('S05', 'remove', {'unknown_task': 2, 'in_past': 0, 'outside_window': 0, 'conflict': 0, 'after_deadline': 0, 'self_overlap': 0}), ('S07', 'remove', {'unknown_task': 2, 'in_past': 0, 'outside_window': 0, 'conflict': 0, 'after_deadline': 0, 'self_overlap': 0}), ('S07', 'add', {'unknown_task': 2, 'in_past': 0, 'outside_window': 0, 'conflict': 0, 'after_deadline': 0, 'self_overlap': 0}), ('S16', 'remove', {'unknown_task': 4, 'in_past': 0, 'outside_window': 0, 'conflict': 0, 'after_deadline': 0, 'self_overlap': 0}), ('S29', 'remove', {'unknown_task': 3, 'in_past': 0, 'outside_window': 0, 'conflict': 0, 'after_deadline': 0, 'self_overlap': 0}), ('S29', 'add', {'unknown_task': 3, 'in_past': 0, 'outside_window': 0, 'conflict': 0, 'after_deadline': 0, 'self_overlap': 0})]

## 비용 (토큰 사용량)

| 구분 | 호출 수 | 입력 토큰 | 출력 토큰 |
|---|---|---|---|
| ours | 120 | 264567 | 8770 |
| judge | 180 | 251039 | 30656 |
| baseline | 90 | 135504 | 55147 |
| baseline_cot | 90 | 520954 | 327323 |
