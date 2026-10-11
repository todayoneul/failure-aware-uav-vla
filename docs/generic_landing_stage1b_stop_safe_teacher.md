# Stage 1b — Stop-Safe Generic Landing

2026-10-11, 브랜치 `exp/generic-landing-stage1b-stop-safe-teacher`. Stage 1([Generic Landing + Active Search v2](generic_landing_active_search_v2.md))은 gate를 통과하지 못했고, 그 기록은 그대로 둔다. Stage 1b는 한 가지만 바꾼다: 착륙 teacher의 label.

**질문:** Teacher가 touchdown 전에 준 label이 student의 stop과 부딪치고 있었는가? 그 충돌을 없애면 cube와 cylinder 착륙이 안정적으로 배워지는가?

<!-- STATUS -->

동결한 baseline(`grounding_film`, canonical 47/48), Gaussian Blur 결과(46 / 40 / 24 / 8), Arbitrary Start 보고, Stage 1의 실패 결과는 바꾸지 않았다. Canonical test 48개와 Depot set은 비행하지 않았다. Active Search v2(Stage 2)는 이 단계의 결과와 무관하게 시작하지 않는다.

## Stage 1에서 본 것

| | Stage 1 (update 2000) |
|---|---:|
| Validation success | 30 / 36 |
| Pad 착륙 | 10 / 10 |
| **Cube 착륙** | **6 / 10** (기준 8) |
| Cylinder 착륙 | 8 / 10 |
| 같은 시작을 한 번 더 | cube 5 / 10, cylinder 6 / 10. 20개 중 5개가 뒤집힘 |
| 실패 | 모두 공중에서 스스로 멈춤. 상승 중 2, 정렬 중 3, 하강 전 1 |

## 1. Student의 stop: 코드에서 읽은 것

보고서의 "0.08m × 4번"을 믿지 않고 코드를 다시 읽었다.

| | 위치 | 내용 |
|---|---|---|
| Stop 함수 | `src/aerovla_oft/spec.py`, `is_stop` | 세 축 모두 `stop_threshold`(0.08) × 그 축의 범위 이하이면 stop |
| 축별 한계 | `configs/aerovla_oft.json` | 전진 0.08m (범위 0–1m), 상하 0.08m (±0.5m), 회전 0.0336 rad (±0.21 rad). 물리 단위, 범위로 자른 뒤 |
| 연속 규칙 | `scripts/visual_search.py`, `run_episode` | Stop인 decision이 `stop_ticks`(4)번 이어지면 비행 종료. Stop이 아닌 decision 하나가 count를 0으로 되돌린다 |
| Chunk | `src/aerovla_oft/model.py`, `infer` | 4개를 예측하고 첫 번째만 비행한다(execute 1). 매 decision 새로 예측하므로, 네 번은 연속한 네 추론의 첫 행동이다 |
| 예외 | `run_episode` | Finalizer가 접촉을 판정하는 중(`CONTACT_CANDIDATE`)이면 끝내지 않고 기다린다. 강제 이동(push, kick)은 세지 않는다 |
| Teacher 비행 | 같은 줄 | Teacher의 비행도 같은 규칙으로 끝난다 |

Teacher의 label은 정확히 나오므로 stop 규칙 안에 든 적이 없다(Stage 1 data에서도 0개). Student의 출력은 label보다 조금 작고, 그래서 label이 규칙 가까이에 있으면 student만 걸린다.

## 2. Stop ratio와 규칙

`stop_ratio` = 행동의 세 축을 각 축의 stop 한계로 나눈 값 중 가장 큰 것. 1 이하이면 stop이다. Teacher를 재는 데 따로 만든 근사 규칙은 없다. `src/landing/stop_safety.py`는 student가 쓰는 `is_stop`을 그대로 불러 쓴다.

**규칙 (녹화 전에 `configs/generic_landing_stage1b.json`에 고정):**

- 착륙 임무에서 touchdown 전의 모든 label은 stop이 아니어야 하고, `stop_ratio`가 **3.0 이상**이어야 한다.
- Touchdown 전에 stop인 label이 4번 이어지는 구간이 없어야 한다.
- 착륙의 stop label은 기체가 면 위에 선 뒤에만 나온다.
- Approach 임무는 목표 근처 공중에서 멈추는 것이 정답이다. Canonical teacher는 그 정지점을 향해 일부러 감속한다(0.2m, 2.5배까지). 이 label은 세어서 보고하고 3.0 기준은 적용하지 않는다. Canonical approach teacher는 바꾸지 않았다.

**왜 3.0인가.** 기존 data를 같은 도구로 쟀다.

| Data | 착륙 label (touchdown 전) | 가장 작은 ratio | 3.0 미만 |
|---|---:|---:|---:|
| Gen-v3 / v3b / v3c의 pad 착륙 | 17,233 | 3.125 | 0 |
| Stage 1의 pad 착륙 | 3,259 | 3.125 | 0 |
| Stage 1의 cube 착륙 | 4,556 | **1.875** | 1,120 |
| Stage 1의 cylinder 착륙 | 4,493 | **1.875** | 1,097 |

Pad teacher의 가장 작은 label은 마지막 하강(0.25m)이고 3.125배다. Student는 pad에 내린다(10/10, 26/27). Stage 1의 cube·cylinder label은 1.875배(수평 전진 0.15m)까지 내려갔고 그 착륙이 공중에서 멈췄다. 3.0은 pad teacher가 늘 지켜 온 여유를 모든 면에 요구하는 값이다. 처음 권장된 2.0은 전진 0.16m를 통과시키는데, 그것은 실패한 label과 같다.

진단용으로 "stop에 가까운 구간"도 센다: ratio 2.0 미만이 4번 이어지는 구간. 1.5가 아니라 2.0인 것은 Stage 1의 0.15m(1.875)가 세어지게 하기 위해서다.

## 3. Teacher를 어떻게 고쳤는가

[src/landing/teacher.py](../src/landing/teacher.py)의 `StopSafeTeacher`. Stage 1의 `SurfaceTeacher`는 지우지 않았고, Stage 1의 계획과 data는 그 teacher로 계속 재현된다.

| | Stage 1 | Stage 1b |
|---|---|---|
| 작은 면 위의 진입 속도 | 남은 거리 × 0.05, 하한 **0.15m** (가운데 3m 전부터 0.15m로 기어감) | 남은 거리 × 0.05, 하한 **0.35m** (가운데 7m 전에 0.35m에 도달하고 그대로 들어감) |
| 하강을 시작하는 반경 | 면 크기에 맞춰 줄임 (0.95m) | Canonical 그대로 (1.75m). Pad와 같다 |
| 하강 전 12m 안의 수평 구간 | 30–40 decision | 20–25 decision |
| 물체 근처 상승 | 필요한 높이까지 제자리에서 (전진 0) | 윗면보다 1m 높아질 때까지만 제자리, 그 뒤로는 **전진하면서** 상승 |
| Pad 착륙, 탐색, approach | Canonical | Canonical |

- **0.35m의 근거:** 기체는 명령을 약 2.4초(4.8 decision) 늦게 따른다. Canonical 반경 1.75m를 그 시간에 지나는 속도다. 그 반경에서 하강 명령을 받으면 기체가 가운데쯤까지 밀려간다. Stop 한계의 4.4배.
- **전진하며 상승:** Stage 1의 상승 중 정지 둘은 윗면보다 0.7m, 1.7m 높은 곳에서 일어났다. Label은 "전진 0, 상승 0.5m"였고, student의 상승 출력이 줄어드는 동안 전진은 0에 머물렀다. 이제 윗면을 벗어난 뒤에는 상승의 끝과 수평 비행의 시작이 전진을 공유한다.
- Stop을 피하려고 넣은 움직임은 없다. 모든 축의 값은 그 순간 기체가 해야 할 일이다.

Teacher의 상태와 계획의 이름:

| 계획의 이름 | 코드의 state | Label |
|---|---|---|
| SEARCH | `search` | 오른쪽 회전 0.21 rad (canonical) |
| APPROACH | `approach` | 12m 밖에서 목표로 전진 |
| SURFACE_APPROACH | `final` | 12m 안에서 수평 전진. 작은 면: 0.35m 이상 |
| CLIMB | `climb` | 상승 0.25–0.5m. 윗면을 벗어나면 전진과 함께 |
| ALIGN | `align` | 가운데가 옆에 있을 때 제자리 회전 0.21 rad |
| DESCENT | `descend` | 하강 0.5m, 낮아지면 0.25m (canonical) |
| TERMINAL | `landed` (착륙), `stop` (approach) | 0. 착륙은 면 위에 선 뒤에만 |

## 4. 바꾸지 않은 것

시작 checkpoint(`grounding_film`), 구조, FiLM, LoRA, action head, Front + Down 입력, 문장만 주는 입력, chunk 4 / 실행 1, finalizer, evaluator, stop 규칙, 착륙 영역, update 수(2,500), learning rate, 새 data 비율(45%), 기존 data, 후보(1000 / 1500 / 2000 / 2500), Stage 1 gate의 모든 숫자. Stage 1의 checkpoint에서 이어 학습하지 않는다. Stage 1의 generic landing episode 233개는 학습에 넣지 않는다.

## 5. 녹화 전 확인

**Label 비교 (같은 도구, 같은 규칙).**

| | Stage 1 (녹화된 233 episode) | Stage 1b (계획 285개의 dry run) |
|---|---:|---:|
| 착륙 label (touchdown 전) | 12,308 | 12,991 |
| 그중 stop | 0 | 0 |
| 3.0 미만 | **2,217** | **0** |
| Stop 4연속 구간 | 0 | 0 |
| Stop에 가까운 4연속 구간 | **1,184** | **0** |
| 가장 작은 ratio | 1.875 | 3.125 |
| 5번째 백분위 | 1.875 | 3.125 |
| SURFACE_APPROACH: 3.0 미만 | 2,217 / 4,619 | 0 / 4,418 |
| CLIMB: 3.0 미만 | 0 / 1,233 | 0 / 1,403 |
| DESCENT: 3.0 미만 | 0 / 2,478 | 0 / 2,868 |

Dry run은 지연이 없는 운동학 계산이다. 녹화가 끝나면 녹화된 label로 다시 잰다.

**Teacher pilot (simulator, finalizer 포함, 새 pilot 시작 15개).**

| | 착륙 | Approach twin | 착륙 위치 (가운데에서) |
|---|---:|---:|---|
| Pad | 4 / 4 | 1 / 1 | 0.28–1.11m |
| Cube | 4 / 4 | 1 / 1 | 0.07–0.21m |
| Cylinder | 4 / 4 | 1 / 1 | 0.07–0.12m |

15 / 15, 충돌 0, wrong target 0. 비행 끝 전의 label 758개 중 규칙을 어긴 것 0(착륙 label 693개의 최소 ratio 3.125). 착륙 12개 중 9개가 1.5m 이상 상승했다(최대 8.9m). 시작은 10.8–36.3m, 높이 3.4–11.5m.

## 6. 계획 (녹화 전에 고정)

숫자는 모두 [configs/generic_landing_stage1b.json](../configs/generic_landing_stage1b.json), 시작은 `configs/generic_landing_stage1b/`에 있다.

- **Data:** Stage 1과 같은 구성을 새 seed(50000번대)에서 다시 뽑았다. 녹화 210 episode(pad / cube / cylinder 각각 착륙 54 + approach twin 16) + validation loss용 24. 10–44m, 시작 높이 3–12m, 절반은 목표가 시야 밖. 44m 밖, 가려진 목표, 새 물체는 없다.
- **Seed:** 저장소의 모든 계획·set·dataset을 읽어 쓰인 범위를 확인했다(0–24999, 30000–31002, 40000–43999). 50000 이상은 쓰인 적이 없다.
- **Pilot:** 새 시작 15개. Baseline과 후보 넷이 한 번씩. 순서: cube·cylinder 착륙 수 → 공중 정지 수 → pad 착륙 수 → wrong target → 충돌 → validation L1 → 이른 update.
- **Validation:** 새 시작 36개(종류별 착륙 10 + approach twin 2), 녹화한 적 없는 layout. Gate는 Stage 1의 숫자 그대로: success 30/36, 착륙 24/30, pad 9, cube 8, cylinder 8, wrong target 2 이하, 충돌 1 이하, approach 중 touchdown 0.
- **Repeat gate (새로 추가):** Cube·cylinder 착륙 20개 시작을 같은 checkpoint로 한 번 더 비행한다. 종류마다 첫 비행 8/10 이상, 두 비행 합 16/20 이상, 두 번 다 실패한 시작 1개 이하.
- **Regression:** Canonical validation과 같은 구성의 새 36개 시작을 baseline과 고른 checkpoint가 한 번씩. Success는 baseline − 2 이상, pad 착륙 baseline − 2 이상, wrong target baseline + 1 이하, 충돌은 baseline보다 많지 않을 것, approach 중 touchdown 0.
- **통과하면** checkpoint 이름 `generic_landing_v1`. 통과하지 못하면 만들지 않는다.

**시작 간격의 예외 둘 (set을 뽑기 전에 config에 적음).** 모든 시작은 같은 장면의 예전 시작에서 6m 이상 떨어진다. 다음 둘만 예외다.

1. 녹화 시작은 Stage 1의 녹화 시작에서 6m를 지키지 않아도 된다. 그 episode들은 이번 학습 data에 없다. 지키게 하면 pad 가까이(10–20m, pad가 처음부터 보이는 시작)에 자리가 남지 않아 16개 layout 어디에서도 뽑히지 않았다. Stage 1의 시작과 같은 시작은 하나도 없다.
2. Regression set은 Stage 1의 regression 시작에서 6m를 지키지 않아도 된다. 그 시작들은 녹화된 적이 없다. 지키게 하면 범위의 seed 10개 모두에서 set이 만들어지지 않았다(가까이에 다른 색 pad가 보이는 시작 자리가 없음).

Pilot과 validation 시작은 예외 없이 모든 예전 시작에서 6m 이상 떨어져 있다.

**실패의 이름.** 실패한 착륙은 어떻게 끝났는지로 먼저 부른다: `AIR_SELF_STOP`, `COLLISION`, `TIMEOUT`, `TARGET_SELECTION`, `TOUCHDOWN`, `STABILITY`, `OTHER`. 공중에서 멈춘 비행은 마지막 decision의 위치로 단계를 붙인다: `SURFACE_APPROACH`(12m 밖), `CLIMB`(12m 안, 물체 위가 아님, 진입 높이보다 낮음), `ALIGNMENT`(진입 높이 또는 물체 위, 하강 반경 밖), `DESCENT`(canonical teacher가 하강하는 자리). 정책의 출력이 아니라 위치만으로 읽으므로 Stage 1의 비행도 같은 규칙으로 읽는다(Stage 1의 여섯: CLIMB 2, ALIGNMENT 3, DESCENT 1).

**Ceiling.** Stage 1에서 14m를 넘은 비행이 둘 있었다. 이번 질문은 teacher의 label이므로 executor는 바꾸지 않고, 넘은 비행의 수와 최고 높이만 기록한다.

<!-- RESULTS -->
