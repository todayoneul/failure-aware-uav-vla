# Gen-v3 Canonical Clean Baseline — 44m 이하, Landing Finalizer

2026-10-09 실행. 브랜치 `exp/aerovla-oft-gen-v3-canonical-baseline`(`exp/aerovla-oft-gen-v3-grounding-landing`에서 분기). 새 학습은 없다. [Gen-v3](aerovla_oft_gen_v3.md)의 두 번째 checkpoint(`generalization_v3b`)를 그대로 쓴다. Gen-v3의 checkpoint, dataset, 결과, 그리고 Depot의 156개 test set은 그대로 있다.

> **이후 (Gen-v3c, 2026-10-09):** 이 문서의 숫자는 **legacy evaluator result**다. 그 뒤 "서 있음" 판정을 접촉과 위치로 읽는 `canonical_evaluator_v2`를 넣었고, 목표 선택 병목을 겨냥한 hard-negative 보정(Gen-v3c)을 새 검증 set에서 비행했다. 결과: 33/36, 다른 물체에서 끝난 비행 3회로 gate를 다시 통과하지 못했다. Canonical test 48개는 여전히 비행하지 않았다. → [Gen-v3c](gen_v3c_hard_negative_grounding.md). 아래의 30/36과 착륙 21/27은 새 판정으로 다시 계산하지 않았다.

**요약**

- **Canonical clean baseline은 아직 서지 않았다.** Gate를 통과하지 못했다. Smoke는 9/10으로 통과했고, representative는 30/36(0.83)에서 한 줄로 멈췄다: 다른 물체에서 끝난 비행이 3회였다(기준 2회 이하). 기준은 비행 전에 고정한 그대로다.
- **그래서 하지 않은 것:** 모델·finalizer·config 동결, canonical test set 48개 비행. Test set은 teacher만 비행했고(48/48) 모델은 한 번도 비행하지 않았다.
- **가장 큰 병목 하나는 목표 선택이다.** 지시한 것과 색이나 형태가 같은 다른 물체를 목표로 삼는 경우가 있다(3/36). 2회는 그 물체가 먼저 보이고 지시한 pad는 안 보이는 시작에서 났고(그런 시작 9개 중 2회), 1회는 지시한 pad로 가다가 놓친 뒤에 났다. 모델 자신의 실패 4회 중 3회다. 범위를 44m 이하로 줄여도 사라지지 않았다.
- **Landing Finalizer는 설계대로 동작했다.** 접촉 24회 전부 latch하고 disarm했다. Approach 9회에서는 한 번도 켜지지 않았고 내려앉은 비행도 없다(9/9 성공). 다만 이번 set에서 결과를 바꾼 것은 1회뿐이다. 정책이 23회 중 22회 스스로 멈췄기 때문이다.
- **평가 쪽 결함을 하나 찾았다.** 착륙 2회가 실패로 기록됐는데, 기록상 기체는 지시한 pad 위에서 움직이지 않았고 latch·disarm도 됐다. "서 있음" 판정이 simulator가 보고한 수직 속도에 기대고 있어서다. Gate의 숫자는 고치지 않았다. 고쳐 읽어도 gate는 통과하지 못한다.
- **Gaussian Blur로 넘어갈 준비는 되지 않았다.**

## 무엇을 정했나

```
Canonical Failure-Aware operating range:
<=44m

Extended long-range evaluation:
>44m, currently limited by same-color object grounding
```

- **Canonical mission:** `Find the blue landing pad and land on it.`
- **범위:** 시작할 때 목표까지 0 초과 44m 이하. Failure-Aware 실험(Blur부터)의 clean 기준은 이 범위에서 잰다.
- **44m 초과는 지우지 않는다.** 따로 [Long-range same-color grounding](long_range_same_color_grounding.md)으로 보존하고, canonical 점수에 섞지 않는다.
- **Depot의 156개 test set은 계속 봉인한다.** 이번 기준을 위해 그중 유리한 것을 골라 쓰지 않았다. canonical용 검증 set과 test set을 새 seed로 따로 만들었다.

## 왜 44m 이하인가

- **Gen-v3의 검증에서 44m 이하는 두 checkpoint 모두 19/22였고, 54m 이상은 8–9/14였다.** 먼 쪽의 실패는 한 가지로 모였다: 파란 물체 넷 중 지시한 것을 확정하지 못한다(54m 이상 파란 목표 3/9, 파랑이 아닌 목표 5/5).
- **그 구간은 감지 한계에 가깝다.** 70m 밖의 pad는 모델 입력에서 약 25×6px이고, Front 화면의 수평선 부근은 기체의 팔이 가려 가운데 16°로만 보인다.
- **Blur 실험의 기준으로 쓸 수 없다.** 작은 형태 구분은 Blur가 가장 먼저 무너뜨린다. clean에서도 불안정한 구간을 기준에 넣으면 Blur의 영향을 가려낼 수 없다.
- **이 범위 안이 안정적이라는 것은 이번 검증으로 확인되지 않았다.** 위의 19/22는 범위를 정한 근거이지 그 결과가 아니다. 새 검증 set에서는 44m 안에서도 같은 종류의 혼동이 났다([아래](#가장-큰-병목-목표-선택)).

## 왜 Landing Finalizer인가

- **VLA는 이미 내려앉는다.** Gen-v3 검증의 착륙 18회 중 16회가 지시한 pad에 touchdown했다.
- **남은 실패는 내려앉은 뒤였다.** pad 위에 가만히 서서 하강 명령 0.04–0.19m를 계속 내거나(정지 기준 0.08m), 약한 상승 명령으로 떴다 내렸다를 반복했다. 기체는 착륙해 있는데 "정책이 0 행동을 4번 연속 낼 때까지" 성공을 기다리는 구조였다.
- **착륙의 끝은 저수준의 일이다.** 실제 기체에서도 접촉과 정지를 확인해 모터를 끄는 것은 비행 제어기가 한다. 그 부분을 실행기로 옮겼다.

## 책임의 구분

| | 하는 일 | 받는 것 |
|---|---|---|
| **VLA** | 문장 이해, 탐색, 목표 grounding, 접근, Down 화면으로 정렬, 하강 | Front RGB, Down RGB, 문장 |
| **Low-level executor** | 연속 명령 실행, 안정 접촉 감지, LANDED latch, 모터 disarm | VLA의 행동(전진·하강·yaw), 접촉 보고, 기체 자신의 위치 |
| **Evaluator** | 맞는 목표인가, 맞는 pad의 착륙 구역 안인가, 거리, touchdown 정밀도, 충돌, 성공 지표 | simulator의 ground truth 전부 |

- **VLA에는 여전히 목표의 좌표·거리·방위·가시성이 들어가지 않는다.**
- **실행기는 무엇 위에 서 있는지 모른다.** finalizer는 "안정된 접촉 상태인가"만 본다. 그것이 지시한 pad였는지는 평가에서 따로 확인한다. 다른 pad에 내려앉아도 finalizer는 똑같이 latch하고, 평가는 실패로 센다.
- **VLA가 모터를 직접 다루지 않는다.** 행동은 계속 전진·하강·yaw이고, disarm은 저장소의 기존 Project AirSim 추상화(`drone.disarm()`)를 쓴다. simulator의 자체 착륙 루틴(`land_async`)은 여전히 쓰지 않는다.

## Landing Finalizer

`src/visual_search/finalizer.py`. 숫자는 [configs/targets/landing_pads.json](../configs/targets/landing_pads.json)의 `finalizer`에 있고, 어떤 비행도 판정하기 전에 commit했다(`7249006`).

```
FLYING → LANDING_DESCENT → CONTACT_CANDIDATE → STABLE_CONTACT → LANDED_LATCHED → DISARMED
```

| 전환 | 조건 |
|---|---|
| 활성화 | 문장이 착륙을 시킬 때만. `land`가 동사로 쓰인 문장("… and land on it", "Land on …", "… and land"). "the landing pad"라는 명사만으로는 켜지지 않는다 |
| FLYING → LANDING_DESCENT | 최근 6 판단 안에 실행기가 0.08m 넘는 하강을 받았다(정책 자신의 정지 범위 밖의 하강) |
| → CONTACT_CANDIDATE | 하강 중에 접촉이 보고됐다. 수평 비행 중의 접촉은 착륙으로 보지 않는다 |
| → STABLE_CONTACT → LANDED_LATCHED | 접촉한 높이에서 5cm 안에 있고, 한 판단 사이의 수직·수평 속도가 각각 0.1m/s 이하인 상태가 **3 판단 연속** |
| CONTACT_CANDIDATE → 되돌아감 | 접촉한 높이에서 15cm 넘게 떠났다 |
| LANDED_LATCHED → DISARMED | 실행기가 disarm을 호출했다 |

- **Latch 뒤에는 어떤 동작 명령도 넘기지 않는다.** 정책이 무엇을 내든 실행기는 0 명령을 보낸다. 그다음 disarm한다.
- **접촉 전에는 아무것도 끄지 않는다.** "pad가 화면에 크게 보인다"는 조건이 아니다. 접촉이 없으면 아무리 가만히 떠 있어도 latch하지 않는다.
- **임계값의 근거:** 기록된 teacher 착륙 61회에서, pad 위에 서 있을 때 한 판단 사이의 움직임은 수직 최대 0.029m/s, 수평 0.0m/s였고 하강 중에는 중앙값 0.51m/s였다. 0.1m/s는 앞의 세 배, 뒤의 1/5다. 3 판단은 첫 판단과 마지막 판단 사이 1초다.
- **속도는 위치 차이로 잰다.** simulator가 서 있는 기체의 상태를 갱신하지 않는 경우가 있어서, 보고된 속도 대신 판단 사이의 위치 변화와 실제 시간을 쓴다.
- **Approach mission은 건드리지 않는다.** "Approach …", "Find …", "Find and approach …"에서는 finalizer가 만들어지되 꺼져 있고, 기존처럼 공중에서 정책이 스스로 멈춰야 한다.
- **Descent는 바꾸지 않았다.** 하강 속도 조절(flare), 카메라, 구조는 이번에 손대지 않았다. 한 번에 하나만 바꾼다.

### Episode마다 남기는 것

`touchdown_detected`, `stable_contact_detected`, `finalizer_triggered`, `disarm_triggered`, `touchdown_velocity_mps`, `horizontal_velocity_mps`, `stable_duration_s`(finalizer의 기록), 그리고 평가의 `touchdown_success`, `stable_physical_landing`, `strict_policy_zero_action`, `system_land_success`, 맞는 pad 여부.

## 착륙을 읽는 세 가지

과거 결과를 소급해서 성공으로 바꾸지 않는다. Gen-v3의 `touchdown 16/18`, `규칙 전체 14/18`은 그 문서에 그대로 있다. 아래 정의는 **Gen-v3 개발 뒤에 도입한 canonical system 정의**이고, 이후의 비행에만 쓴다.

| 이름 | 조건 | 쓰임 |
|---|---|---|
| **Touchdown success** | 지시한 pad + 착륙 구역(중심에서 가로·세로 6.6m) 안의 접촉 + 접촉 직전 수직 속도 0.75m/s 이하 | 단계 지표 |
| **Stable physical landing** | 위 + pad 위에 선 채(touchdown 높이에서 3cm 이내, 수직 0.05m/s 이하, 수평 0.1m/s 이하) 3 판단 이상 + 끝날 때도 서 있음. 기록에서 따로 계산하고 finalizer의 말을 빌리지 않는다 | 단계 지표 |
| **System landing success** (VLA + finalizer) | 위 + finalizer가 latch하고 disarm함 + 다른 접촉 없음 | **mission의 성공** |
| **Strict policy zero-action** | 지시한 pad에 touchdown한 뒤 정책이 스스로 0에 가까운 행동을 4 판단 연속 냄 | 보조 지표. Gen-v3와의 비교용. 성공의 조건이 아님 |

- **Stable physical landing의 판정에는 알려진 결함이 있다.** "수직 0.05m/s 이하"를 simulator가 보고한 속도로 확인하는데, 서 있는 기체에 0이 아닌 속도가 보고되는 경우가 있다. 이번 검증의 착륙 2회가 그렇게 빠졌다([아래](#평가의-서-있음-판정에서-찾은-결함)). 정의와 숫자는 고치지 않았다.
- **Strict 지표도 계속 잰다.** latch 뒤에도 정책에 화면을 계속 주고(명령은 넘기지 않음) touchdown 뒤 20 판단까지 0 행동이 나오는지 본다. "finalizer 없이도 스스로 멈췄을까"에 답하기 위해서다.
- **Approach의 성공은 그대로다.** 스스로 멈춤 + 목표 15m 이내 + 공중에 있고 pad 위로 내려오지 않음 + 충돌 없음.

## 검증 set과 test set

둘 다 새 seed로 만들었고, 이전의 어떤 계획(학습, Gen-v3 검증, Depot의 156개, set L)과도 seed와 시작 위치가 겹치지 않는다(`tests/test_canonical.py`).

| | 검증 (gate 전용) | Test (gate 통과 뒤 한 번) |
|---|---|---|
| 파일 | `configs/gen_v3_canonical_44m_validation.json` | `configs/gen_v3_canonical_44m_test.json` |
| Seed | 8500–8525 | 9500–9530 |
| 배치 | Field·Lot의 a–d. 학습에 쓴 배치, 학습에 안 쓴 시작 | Field·Lot의 g–j. 학습에도 검증에도 없는 배치 |
| 시작 수 | 36 (band마다 12) | 48 (band마다 16) |
| 착륙 / 접근 | 27 / 9 | 36 / 12 |
| blue pad 착륙 / red pad 착륙 | 21 / 6 | 30 / 6 |
| Teacher | 36/36 | 48/48 |
| 모델 | 비행함 (아래) | **비행하지 않음** |

- **Band:** near 10–20m, mid 20–32m, far 32–44m. 시작은 모든 물체에서 5m 이상 떨어져 있어서 가장 가까운 시작이 pad 중심에서 약 12m다.
- **시작의 종류:** 처음부터 보임(`visible`), 찾아야 함(`search`, `behind`), 화면 가장자리(`edge`), 다른 pad나 같은 색 물체가 곁에 있음(`pair`), 다른 물체가 먼저 보이고 목표는 안 보임(`past`), red pad(`red_visible`, `red_search`).
- **Twin:** 같은 자세에서 문장만 "Approach the blue landing pad."로(`approach`), 두 물체의 자리를 바꿈(`swap`), 같은 자세에서 곁의 물체를 물음(`query`). Twin은 짝이 된 시작의 band에 넣었다. 자리를 바꾼 twin은 목표까지의 거리가 달라지지만 44m 이하다.
- **Test는 새 장면이 아니다.** 같은 simulator의 학습 장면(Field·Lot) 안에 만든 새 배치다. 새 held-out 장면은 Depot의 156개이고, 계속 봉인돼 있다.

## Gate

기준은 [configs/visual_search.json](../configs/visual_search.json)의 `canonical.gates`에 비행 전에 고정했다(`7249006`). 숫자는 Gen-v3의 gate와 같고, smoke의 먼 구간 줄만 canonical의 far band(32–44m)로 읽는다. 두 gate 모두 검증 set만 쓴다. 판정 기록: `outputs/generalization/canonical_baseline/gate_smoke.json`, `gate_representative.json`.

| Smoke (검증 시작 10개) | 값 | 기준 | |
|---|---:|---|---|
| 비행 | 10 | 10 이상 | 통과 |
| 성공 | 9 | 7 이상 | 통과 |
| 착륙 성공(system) | 6 | 1 이상 | 통과 |
| 32–44m에서의 성공 | 3 | 1 이상 | 통과 |
| 충돌 | 0 | 1 이하 | 통과 |
| 실행 오류 | 0 | 0 | 통과 |

| Representative (검증 시작 36개) | 값 | 기준 | |
|---|---:|---|---|
| 비행 | 36 | 30 이상 | 통과 |
| 성공률 | 30/36 (0.833) | 0.80 이상 | 통과 |
| 착륙 성공률(system) | 21/27 (0.778) | 0.75 이상 | 통과 |
| 목표가 보인 뒤 접근을 시작함 | 32/34 (0.941) | 0.90 이상 | 통과 |
| 문장이 시킨 대로 끝남 | 33/36 (0.917) | 0.90 이상 | 통과 |
| **다른 물체에서 끝난 비행** | **3** | **2 이하** | **통과 못 함** |
| 충돌 | 0 | 1 이하 | 통과 |
| 실행 오류 | 0 | 0 | 통과 |

- **Smoke: PASS. Representative: FAIL.** 기준을 결과에 맞춰 고치지 않았다.
- **통과한 줄도 여유가 크지 않다.** 착륙 성공률과 "시킨 대로 끝남"은 각각 한 비행이 더 실패했으면 기준 아래였다.
- **계획대로 여기서 멈췄다.** Test set은 비행하지 않았고, 재학습도 하지 않았다.

## 검증에서 본 것

표와 episode별 기록: `outputs/generalization/canonical_baseline/`(`tables.md`, `summary.json`, `episodes.csv`, `failures/`).

| Band | 전체 | 착륙 | 접근 |
|---|---:|---:|---:|
| near (10–20m) | 11/12 | 8/9 | 3/3 |
| mid (20–32m) | 11/12 | 8/9 | 3/3 |
| far (32–44m) | 8/12 | 5/9 | 3/3 |
| 합계 | 30/36 | 21/27 | 9/9 |

| 착륙 27회 | 수 |
|---|---:|
| 지시한 pad로 감 | 23 |
| 지시한 pad에 touchdown (구역 안, 부드럽게) | 23 |
| Stable physical landing (평가의 판정) | 21 |
| System landing success (위 + latch + disarm) | 21 |
| Strict policy zero-action (보조) | 22 |
| blue pad / red pad | 17/21, 4/6 |

- **문장이 끝을 정한다.** "Approach …"와 다른 물체를 묻는 문장 9회는 모두 공중에서 스스로 멈췄고(9/9) 내려앉은 비행은 없다. 착륙을 시킨 27회 중 공중 정지로 끝난 것은 3회이고, 셋 다 아래의 실패다.
- **충돌은 0이다.**
- **Touchdown은 구역 안이지만 teacher보다 부정확하다.** 오차 중앙값 2.40m, 최대 4.03m(구역 6.6m). 직전 수직 속도 중앙값 0.58m/s, 최대 0.62m/s(기준 0.75m/s). Teacher는 0.62m, 0.36m/s다.

### 실패 6회

| Episode | 시작 | 일어난 일 | 단계 |
|---|---|---|---|
| `cv-8504-…-near_pair-swapped` | blue pad는 뒤쪽 121° 방향 28.5m, red pad가 정면 18m에 보임 | red pad로 곧장 가서 내려앉았다. Finalizer는 latch하고 disarm했고(무엇 위인지 모르므로 설계대로), 평가는 실패로 셌다 | **WRONG TARGET** — 형태가 같은 다른 색 pad |
| `cv-8522-…-far_past` | blue pad는 84° 방향 32.8m로 안 보임, blue cube가 18° 방향 42.7m에 보임 | blue cube로 가서 5.8m 앞 공중에서 멈췄다 | **WRONG TARGET** — 같은 색 |
| `cv-8524-red_pad-far_red_visible` | red pad가 정면 39.8m에 보임 | 25m까지 다가가다 방향을 틀어 놓쳤고, 한 바퀴 돈 뒤 red cube 7m 앞 공중에서 멈췄다 | **WRONG TARGET** — 같은 색 |
| `cv-8523-…-far_behind` | blue pad는 정반대 방향 38.7m | 돌다가 pad가 화면 가장자리(42°)에 들어온 순간 0 행동을 4번 내고 공중에서 멈췄다 | **GROUNDING** — 조기 정지 |
| `cv-8510-…-mid_search` | 24.0m, 안 보임 | 찾아가서 blue pad에 내려앉았다(오차 1.85m). 정책은 0 행동을 냈고 finalizer는 latch·disarm했다. 평가의 "서 있음" 판정이 세지 않았다 | **PHYSICAL LANDING**으로 기록. 실제로는 평가의 읽기 문제(아래) |
| `cv-8525-red_pad-far_red_search` | 32.5m, 안 보임 | 같음(red pad, 오차 1.13m) | 같음 |

단계별로 나누면:

| 단계 | 수 | |
|---|---:|---|
| SEARCH | 0 | 자동 분류는 2회를 SEARCH로 적는다(지시한 pad가 한 번도 화면에 들어오지 않음). 둘 다 다른 물체로 먼저 갔기 때문이라 아래에 넣었다. Gate의 "다른 물체에서 끝남"은 처음부터 3으로 센다 |
| GROUNDING | 1 | |
| **WRONG TARGET** | **3** | |
| APPROACH | 0 | |
| ALIGNMENT | 0 | |
| DESCENT | 0 | |
| PHYSICAL LANDING | 2 | 둘 다 평가의 읽기 문제 |
| FINALIZER | 0 | |
| COLLISION | 0 | |

## 가장 큰 병목: 목표 선택

**지시한 것과 색이나 형태가 같은 다른 물체를 목표로 삼는 경우가 있다.** 36회 중 3회다.

| 시작 | 수 | 성공 | 다른 물체에서 끝남 |
|---|---:|---:|---:|
| 곁의 관련 물체가 먼저 보이고 지시한 목표는 안 보임 (`swap`, `query`, `past`) | 9 | 7 | 2 |
| 그 밖 | 27 | 23 | 1 |

("관련 물체" = 지시한 것과 색이 같거나 형태가 같은 다른 물체. "그 밖"의 나머지 실패 3회는 조기 정지 1회와 평가의 읽기 문제 2회다.)

- **Gate에서 걸린 유일한 줄이고, 모델 자신의 실패 4회 중 3회다.**
- **두 가지 모양으로 났다.** 2회는 관련 물체가 처음부터 보이고(18m, 43m) 지시한 pad는 화면 밖에 있던 시작에서, 찾지 않고 보이는 물체로 갔다. 1회는 정면에 보이는 red pad로 25m까지 가다가 방향을 틀어 놓쳤고, 한 바퀴 돌다 만난 red cube로 갔다.
- **44m 이하로 줄여도 사라지지 않았다.** 3회 중 2회가 32–44m, 1회가 가장 가까운 band다. [Gen-v3](aerovla_oft_gen_v3.md)의 검증에서 44m 이하의 혼동은 1/15였다. 이번 set은 관련 물체가 먼저 보이는 시작을 band마다 3개씩 일부러 넣었고, 실패는 주로 거기서 났다.
- **색만의 문제가 아니다.** 2회는 같은 색의 다른 형태(pad 대 cube), 1회는 같은 형태의 다른 색(blue pad 대 red pad)이다.
- **항상 틀리는 것은 아니다.** 같은 종류의 시작에서 성공도 한다: 자리를 바꾼 twin 2/3, 곁의 물체를 묻는 twin 3/3, 다른 물체가 먼저 보이는 시작 2/3. 불안정하다는 뜻이다.
- **끝맺음은 고른 물체를 따른다.** Cube로 간 2회는 그 앞 공중에서 멈췄고, 다른 pad로 간 1회는 내려앉았다. 착륙을 시켰는데 공중 정지로 끝난 비행은 이 선택 오류와 조기 정지에서만 나왔다.
- **내 해석:** 화면 한 장만 보는 정책이라, "문장이 말한 것은 아직 안 보이니 더 찾는다"보다 "보이는 비슷한 물체로 간다"가 이기는 장면이 있다. 이번 기록만으로 원인을 확정한 것은 아니다.

## Landing Finalizer가 한 일

| | 수 |
|---|---:|
| 착륙 mission에서 접촉이 있었던 비행 | 24 (지시한 pad 23, 다른 pad 1) |
| Latch | 24 |
| Disarm | 24 |
| 접촉에서 latch까지 (중앙값) | 1.55초 |
| 접촉 없이 끝난 착륙 mission(공중 정지 3회)에서 latch나 disarm | 0 |
| Approach mission 9회에서 켜진 횟수 | 0 |
| Finalizer가 결과를 바꾼 비행 | 1 (`cv-8505-…-near_past`) |

- **상태 기계는 요구한 대로 동작했다.** 접촉 전에는 아무것도 끄지 않았고, approach에서는 켜지지 않았고, 안정 접촉이면 latch한 뒤 disarm했다.
- **무엇 위에 섰는지는 평가가 판정했다.** red pad에 내려앉은 1회에서도 finalizer는 똑같이 latch했고, 평가는 실패로 셌다.
- **이번 set에서의 기여는 작았다.** 지시한 pad에 내려앉은 23회 중 22회는 정책이 스스로 0 행동을 냈다. Finalizer가 없었으면 실패였을 비행은 1회다(touchdown 뒤 20 판단 동안 0 행동이 나오지 않음). Gen-v3 검증에서 16회 중 2회였던 "내려앉고도 멈추지 못함"이 여기서는 23회 중 1회였다.

## 평가의 "서 있음" 판정에서 찾은 결함

실패로 기록된 착륙 2회(`cv-8510`, `cv-8525`)는 비행의 실패가 아니다.

| Episode | 접촉 뒤 판단 수 | 높이 변화 | 수평 이동 | Simulator가 보고한 수직 속도 (m/s) | "서 있음"으로 센 판단 | 정책의 0 행동 |
|---|---:|---:|---:|---|---:|---|
| `cv-8510-…-mid_search` | 4 | 0.0004m | 0.0m | 0.095, 0.074, 0.052, 0.041 | 1 | 예 |
| `cv-8525-red_pad-far_red_search` | 4 | 0.0002m | 0.0m | 0.085, 0.080, 0.080, 0.063 | 0 | 예 |

- **기체는 pad 위에서 움직이지 않았다.** 접촉 뒤 4 판단 동안 위치가 1mm도 변하지 않았고, 정책은 0 행동을 냈고, finalizer는 latch하고 disarm했다.
- **평가는 "서 있음"을 simulator가 보고한 수직 속도 0.05m/s 이하로 확인한다**(`SearchEnv.resting`). 이 두 비행에서는 기체가 멈춘 뒤에도 보고된 속도가 천천히 줄어들기만 했다. 접촉이 있었던 나머지 22회에서는 접촉한 판단부터 0으로 보고됐다. Finalizer는 같은 이유로 처음부터 보고된 속도 대신 위치 차이를 쓴다.
- **Gate의 숫자는 고치지 않았다.** 이 둘은 기록대로 실패다(30/36, 착륙 21/27). 결과를 본 뒤 판정 방식을 바꾸지 않는다는 원칙대로다.
- **고쳐 읽어도 gate는 통과하지 못한다.** 위치로 읽으면 32/36, 착륙 23/27이 되지만 "다른 물체에서 끝남 3회"는 그대로다.
- **다음 gate 전에 고칠 것:** "서 있음"을 판단 사이의 높이 변화로 읽는다(finalizer와 같은 방식). 이번에는 적용하지 않았다. 고친 뒤에는 검증 시작도 새 seed로 다시 뽑는다. 이 36개는 이미 봤다.
- **그렇게 했다:** [Gen-v3c](gen_v3c_hard_negative_grounding.md#evaluator-v2)의 `canonical_evaluator_v2`와 seed 22000번대의 새 검증 set. 이 문서의 run에는 소급하지 않았다.

## 하지 않은 것

- **동결하지 않았다.** `scripts/freeze_baseline.py write canonical_baseline …`은 gate 통과 뒤의 단계다. `outputs/generalization/canonical_baseline_frozen.json`은 없다.
- **Canonical test set 48개를 비행하지 않았다.** 다음 gate가 통과할 때까지 그대로 둔다.
- **Depot의 156개를 비행하지 않았다.**
- **학습하지 않았다.** Gen-v4, query twin 추가, FiLM, proprio, search_scan_progress 모두 없다.
- **Blur, Occlusion, Drift를 실행하지 않았다.**
- **Threshold를 바꾸지 않았다.** Finalizer의 숫자, 착륙 규칙, gate 기준 모두 `7249006`의 것이다.

## 먼 거리의 한계

44m를 넘는 시작은 [Long-range same-color grounding](long_range_same_color_grounding.md)에 따로 둔다. 기존 결과(set L, Gen-v3 검증의 54m 이상, Depot의 Q7·Q8)는 그대로다. 이번 검증은 같은 종류의 혼동이 44m 안에서도, 관련 물체가 먼저 보일 때 난다는 것을 더했다.

## 다음 Failure-Aware 단계

**Gaussian Blur는 아직 아니다.** Clean에서 100회 중 8회꼴로 다른 물체로 가는 기준 위에서는, Blur가 더한 선택 오류를 가려낼 수 없다.

먼저 정할 것은 하나다: **목표 선택 병목을 어떻게 다룰 것인가.** 이번 작업의 범위(재학습 없음, 데이터 추가 없음, 구조 변경 없음) 안에는 모델 쪽에서 이것을 고칠 수단이 없다. 그와 별개로, 다음 gate 전에는 평가의 "서 있음" 판정을 고치고 검증 시작을 새 seed로 다시 뽑아야 한다.

**그 뒤 (Gen-v3c):** 데이터로 고쳐 보았다. Hard-negative 96 episode를 더한 짧은 보정 학습은 pilot 12회를 통과했지만(11/12), 새 검증 36회에서 다시 다른 물체 3회로 gate를 통과하지 못했다. 관련 물체가 먼저 보이는 시작에서의 오류는 다섯 번에 한 번꼴 그대로다. 다음 후보는 FiLM 등 더 강한 language–vision 결합이다. → [Gen-v3c](gen_v3c_hard_negative_grounding.md)

## 재현

```powershell
# 시작 set (파일이 있으면 거부한다)
python scripts/canonical_baseline.py plan validation
python scripts/canonical_baseline.py plan test

# Teacher로 두 set 확인
scripts/run_visual_search.ps1 -Policy teacher -Plan configs/gen_v3_canonical_44m_validation.json -Set episodes -Output outputs/visual_search/canonical_teacher_validation
scripts/run_visual_search.ps1 -Policy teacher -Plan configs/gen_v3_canonical_44m_test.json -Set episodes -Output outputs/visual_search/canonical_teacher_test

# Gate: smoke 10개 → 판정 → 나머지 검증 시작 → 판정 (finalizer는 기본으로 켜져 있다)
$ids = (python scripts/canonical_baseline.py ids) -split ' '
scripts/run_visual_search.ps1 -Policy oft -Checkpoint outputs/aerovla_oft/checkpoints/generalization_v3b -Plan configs/gen_v3_canonical_44m_validation.json -Set episodes -Only $ids -Output outputs/visual_search/canonical_gate -Resume
python scripts/canonical_baseline.py gate smoke --run outputs/visual_search/canonical_gate --teacher outputs/visual_search/canonical_teacher_validation
scripts/run_visual_search.ps1 -Policy oft -Checkpoint outputs/aerovla_oft/checkpoints/generalization_v3b -Plan configs/gen_v3_canonical_44m_validation.json -Set episodes -Output outputs/visual_search/canonical_gate -Resume
python scripts/canonical_baseline.py gate representative --run outputs/visual_search/canonical_gate --teacher outputs/visual_search/canonical_teacher_validation

# 표
python scripts/canonical_baseline.py report "OFT Gen-v3:validation=outputs/visual_search/canonical_gate" "Teacher:validation=outputs/visual_search/canonical_teacher_validation" "Teacher:test=outputs/visual_search/canonical_teacher_test" --teacher validation=outputs/visual_search/canonical_teacher_validation test=outputs/visual_search/canonical_teacher_test --output outputs/generalization/canonical_baseline
```

Checkpoint와 비행 기록(`outputs/visual_search/`, `outputs/aerovla_oft/`)은 Git에 없다.

## Tests

`tests/test_canonical.py`:

- Approach mission은 finalizer를 켜지 않는다. 착륙을 시키는 문장만 켠다.
- Latch에는 하강 중의 실제 접촉과 3 판단의 정지가 필요하다. 접촉 전에는 아무것도 끄지 않는다.
- 안정 접촉이면 LANDED로 latch하고, 그 뒤의 동작 명령은 넘기지 않는다.
- Finalizer는 무엇 위에 섰는지 받지 않는다. 지시한 pad인지는 평가가 따로 판정하고, 다른 pad에 latch한 비행은 실패다.
- Touchdown, stable physical landing, strict zero-action, system 결과는 따로 계산된다.
- Finalizer가 없던 Gen-v2·Gen-v3의 기록은 이전과 똑같이 읽힌다.
- 두 시작 set은 생성기로 그대로 재현되고, 44m 이하이고, band가 고르고, 이전의 모든 계획과 겹치지 않는다.
- Gate의 기준은 Gen-v3 gate의 숫자와 같다.
- Latch됐지만 "서 있음"으로 읽히지 않은 착륙은 목록에 나오고, 세는 숫자는 바뀌지 않는다.
