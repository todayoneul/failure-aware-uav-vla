# Stage 1c — Cube Descent Recovery + Same-Color Landing Grounding

2026-10-12, 브랜치 `exp/generic-landing-stage1c-recovery-grounding`. Stage 1([문서](generic_landing_active_search_v2.md))과 Stage 1b([문서](generic_landing_stage1b_stop_safe_teacher.md))는 gate를 통과하지 못했고, 그 기록은 그대로 둔다.

**질문:** Stage 1b가 얻은 stop-safe 착륙을 유지하면서, cube 착륙의 마지막 단계와 같은 색 물체 사이의 선택을 함께 고칠 수 있는가?

<!-- STATUS -->

고치는 것은 둘이고 그 밖의 능력은 더하지 않는다.

- **A. Cube descent recovery.** Stage 1b의 cube 실패는 모두 내려가는 단계였다. 그런 상태에서 빠져나오는 길을 가르친다.
- **B. Same-colour landing grounding.** 착륙을 배운 정책이 "파란 pad"를 찾으라는 문장에 다른 파란 물체로 갔다. 색만 맞는 물체를 착륙 대상으로 삼지 않게 한다.

동결한 baseline(`grounding_film`, canonical 47/48), Gaussian Blur 결과, Arbitrary Start 보고, Stage 1과 Stage 1b의 결과는 바꾸지 않았다. Canonical test 48개와 Depot set은 비행하지 않았다. Active Search v2(Stage 2)는 이 단계의 결과와 무관하게 시작하지 않는다.

## 출발점

| | Stage 1 | Stage 1b |
|---|---:|---:|
| Gate | 실패 | 실패 |
| Cube 착륙 (첫 비행 / 두 비행 합) | 6 / 10, 11 / 20 | 7 / 10, 16 / 20 |
| Cylinder 착륙 | 8 / 10, 14 / 20 | 10 / 10, 20 / 20 |
| 공중 정지 (cube·cylinder 40번 중) | 15 | 1 |
| Regression 충돌 | 0 | 2 (재비행에서도 같음) |

Stage 1c는 **Stage 1b가 고른 checkpoint(update 2000)에서 이어서** 학습한다. 이 checkpoint는 Stage 1b의 gate를 통과한 것이 아니므로 `generic_landing_v1`이라 부르지 않고 경로와 지문으로 적는다(`configs/generic_landing_stage1c.json`의 `start_checkpoint_sha256`). `grounding_film`에서 다시 시작하지 않는 이유: Stage 1b가 이미 얻은 것(cylinder 20/20, 공중 정지 15 → 1)을 유지한 채 남은 둘만 고치는 보정이기 때문이다.

## 1. Stage 1b의 cube 실패를 다시 읽음

Cube 착륙 20번(validation 10개 시작 × 2)과 같은 시작에서의 teacher 비행 10번을 위치와 비행한 행동으로 읽었다(`src/landing/failures.descent_profile`).

| | Teacher (10) | Stage 1b student (20) |
|---|---|---|
| 물체 위로 들어올 때의 높이 (윗면 위) | 3.3–4.0m | 0.4–4.3m |
| 하강을 시작한 곳 (가운데에서) | 1.44–1.75m | 0.83–6.16m |
| 하강 중 상하 명령의 부호가 바뀐 횟수 | 0 | 18번은 0, 한 번은 2, 한 번은 58 |
| 내린 곳 (가운데에서) | 0.06–0.36m | 성공 0.42–3.35m, 실패 3.8m와 5.7m (영역 밖) |

- 가운데에서 5m 넘게 떨어진 곳에서 하강을 시작한 비행 셋(5.26m, 5.68m, 6.16m)은 모두 실패했다. 4.1m 안에서 시작한 비행 중 실패는 하강을 끝내지 못한 하나뿐이다.
- `52411`: 윗면보다 0.4–0.9m 높은 채로 들어와 가장자리에 내렸다(3.8m, 3.35m).
- `52407`: 가운데 전 5.7m, 6.2m에서 내려가기 시작해 모서리에서 끝났다.
- `52412`: 가운데에서 1.6m, 윗면 1.1–2.3m 위에 머물며 상하 명령의 부호를 58번 바꿨다.

Teacher는 이런 상태에 스스로 들어가지 않는다. 그래서 data에는 이런 상태에서 나오는 길이 없었다.

## 2. Stage 1b의 같은 색 물체 실패를 다시 읽음

Regression의 세 비행. 문장은 모두 "Find the blue landing pad and land on it."

| 시작 | 첫 시야 | Stage 1b | `grounding_film` | Stage 1의 checkpoint |
|---|---|---|---|---|
| `53471` | Blue cone이 18.5m 앞 가운데. Pad는 25m 뒤 | Cone으로 직진, 9.6m까지 올라 37 decision에 충돌 | 다른 물체 앞 공중 정지 | 다른 물체 앞 공중 정지 |
| `53479` | Blue cone이 49.5m 앞 가운데. Pad는 31m, 60° 옆 | Cone으로 가서 73 decision에 충돌 | Pad에 착륙 | Pad에 착륙 |
| `53101` | 아무것도 없음. Pad는 16.6m, 89° 옆 | Pad를 한 번도 보지 못하고 65m 떨어진 blue cube에 착륙 | Pad에 착륙 | Pad에 착륙 |

세 비행 모두 pad를 한 번도 시야에 넣지 못했다. Cone은 높이 10m이고 평평한 꼭대기가 없다. Stage 1과 Stage 1b의 generic landing data에는 "같은 색의 다른 물체가 먼저 보이는 착륙"을 넣은 적이 없다. 계획한 착륙 180개 중 60m 안의 같은 색 물체가 첫 시야에 있고 목표는 없는 시작은 Stage 1에 2개, Stage 1b에 3개였고 우연히 생긴 것이다.

## 3. Teacher: 하강 규칙 하나

[src/landing/teacher.py](../src/landing/teacher.py)의 `RecoveryTeacher`. Stage 1b의 teacher에 규칙 하나를 더했다. 탐색, 접근, 상승, 진입 속도(0.35m), 하강 속도는 그대로다.

| | Stage 1b | Stage 1c |
|---|---|---|
| 하강을 시작하는 곳 | 가운데에서 1.75m 안 (canonical) | 같음 |
| 낮게 내려온 뒤(윗면 위 3m 이하) 하강을 계속하는 범위 | 3.5m (pad의 값) | **2.675m** (cube, cylinder). Pad는 3.5m 그대로 |
| 그 범위 밖, 낮은 높이 | 계속 하강 | **하강하지 않음.** 올라가서 가운데로 다시 들어온 뒤 1.75m 안에서 하강 |

- **2.675m의 근거:** 하강을 시작하는 반경(1.75m)과 쓸 수 있는 윗면의 끝(3.6m)의 한가운데다. Canonical 3.5m는 영역이 6.6m인 pad의 값이고, cube에서는 끝에서 0.1m다.
- **Hysteresis:** 하강은 1.75m 안에서만 시작하고, 일단 낮아지면 2.675m까지는 계속한다. 작은 위치 오차로 label이 "상승"으로 뒤집히지 않는다. 그보다 크게 벗어나면 되돌아간다.
- Teacher의 정상 비행은 바뀌지 않는다(1.75m에서 내려가 0.4m 안에 내린다). Stage 1b의 data는 이 규칙과도 맞는다.

Stop-safety 규칙은 Stage 1b 그대로다: 착륙의 touchdown 전 label은 stop이 아니고 stop 규칙의 3.0배 이상이며 stop 4연속 구간이 없다.

## 4. Recovery data: 네 가지 상태

Teacher가 스스로 들어가지 않는 상태는 **강제 이동**으로 만든다(`RecoveryPushes`). 강제 이동은 비행하지만 label이 아니다(canonical push와 같은 방식. 강제 이동이 들어간 frame은 학습 sample이 되지 않는다). 강제 이동이 끝난 뒤의 frame이 teacher의 "빠져나오는 길"을 담는다.

| 상태 | 어떻게 만드는가 | Teacher의 답 |
|---|---|---|
| `under_climb` | 상승 도중(윗면 위 0.6–2.1m) 더 오르지 않고 물체 쪽 3.0–5.5m까지 직진시킨다 | 상승 → 가운데로 → 하강 |
| `edge` | 진입 높이에서 가운데를 2.5–4.5m 옆으로 비켜 지나가게 한다 | 하강하지 않음. 가운데로 돌아서 들어온 뒤 하강 |
| `premature` | 가운데 4.5–6.5m 전에서 윗면 위 1.3–2.3m까지 내려가게 한다 | 하강 중단 → 상승하며 가운데로 → 하강 |
| `stall` | 가운데 위에서 내려가는 중(윗면 위 1.2–2.5m 이하)에 두 번 들어 올린다 | 멈춤 없이 하강. 되올라가지 않음 |

- 범위는 1절의 실패에서 왔다. 같은 시작을 베끼지 않았다. Stage 1b의 validation·regression 시작은 이 data의 시작이 아니다.
- 강제 하강은 윗면 위 1.3m보다 낮게, 강제 진입은 0.6m보다 낮게 내리지 않는다. 기체가 명령을 약 2.4초 늦게 따르기 때문에 물체 옆면과 거리를 둔다.
- Episode는 다음을 모두 만족할 때만 data가 된다: teacher가 영역 안에 내리고 스스로 멈췄다, 계획한 강제 이동이 모두 일어났다, 강제 이동이 끝난 첫 decision에 camera가 물체를 보고 있었다(사진으로 알 수 없는 답을 label로 주지 않는다).
- 강제 이동은 teacher 상태가 아니라 **위치**에서 일어난다. 그래서 같은 episode를 학습된 정책에게도 그대로 날릴 수 있다(pilot의 8개).

## 5. Same-colour data

첫 시야에 이름과 같은 색의 **다른** 물체가 있고 이름의 물체는 없다. 이름의 물체는 돌면 보인다. 이동해야 보이는 시작, 44m 밖, 가려진 목표는 없다.

- **Teacher:** 이름의 물체가 camera에 보일 때까지 canonical 탐색(오른쪽 회전)을 한다. 무엇이 보이든 상관하지 않고, 목표가 어디 있는지 모른다. 보이면 접근해서 착륙한다. 다른 물체 쪽으로 가는 label은 없다(계획 단계에서 teacher의 경로가 다른 물체 12m 안으로 들어가는 시작은 버린다).
- **Twin:** 같은 시작에 문장만 바꾼다. 첫 화면은 같고 명사가 다르며 내릴 물체가 다르다. Cone에는 착륙면이 없어 twin이 없다.
- 일부는 다른 물체도 착륙할 수 있는 물체다(pad ↔ cube). "내리기 쉬운 같은 색 물체"로 가는 지름길을 막기 위해서다.

| 이름의 물체 ← 먼저 보이는 물체 | Episode |
|---|---:|
| Blue pad ← blue cube | 20 |
| Blue pad ← blue cone | 12 |
| Blue cube ← blue pad | 16 (twin 10 + 단독 6) |
| Red pad ← red cube | 12 |
| Red cube ← red pad | 12 (twin 6 + 단독 6) |
| 합 | 72 (twin pair 16쌍) |

**Student에게 가는 것은 Front·Down 사진과 문장뿐이다.** 목표의 좌표, 거리, 방향, 물체 id, 다른 물체의 id는 teacher와 evaluator만 쓴다. 녹화된 data의 문장이 모두 catalogue의 문장인지(숫자와 id 없음), proprio가 꺼져 있는지, twin이 시작을 공유하고 다른 물체를 가리키는지를 학습 전에 검사한다(`scripts/generic_landing_1c.py inputs`).

## 6. 계획 (녹화 전에 고정)

숫자는 모두 [configs/generic_landing_stage1c.json](../configs/generic_landing_stage1c.json), 시작은 `configs/generic_landing_stage1c/`에 있다.

**Data.** 녹화 152 episode + validation loss용 16.

| | Episode |
|---|---:|
| Recovery: cube (상태마다 16) | 64 |
| Recovery: cylinder (상태마다 2) | 8 |
| Recovery: pad (`edge` 4, `stall` 4) | 8 |
| Same colour | 72 |

**학습.** Stage 1b의 update 2000에서 시작. LoRA, action head, FiLM만 학습하고 module은 더하지 않는다. Learning rate는 Stage 1b의 절반(2.5e-5), 2,000 update. 뽑는 frame의 35%는 Stage 1c의 새 data, 35%는 Stage 1b의 stop-safe generic landing data, 30%는 기존 data다. Stage 1의 generic landing episode는 0개다. 후보는 update 500 / 1000 / 1500 / 2000.

**Seed.** 저장소의 모든 계획·set·dataset을 읽었다(0–24999, 30000–31002, 40000–43999, 50000–53999 사용 중). Stage 1c는 60000번대를 쓴다.

**Pilot (새 시작 24개).** 강제 cube 착륙 8(상태마다 2), same-colour 착륙 8, pad 4, cylinder 4. Stage 1b의 checkpoint와 후보 넷이 한 번씩. 순서: same-colour 충돌 → 다른 물체 착륙 → 강제 착륙 회복 수 → cube 착륙 → cylinder → pad → 공중 정지 → validation L1 → 이른 update. Same-colour 충돌이 있거나 Stage 1b보다 pad·cylinder·회복이 나빠진 후보는 뒤로 보낸다.

**Gate A — Generic landing (새 시작 36개, Stage 1의 형식과 숫자 그대로).** Success 30/36, 착륙 24/30, pad 9, cube 8, cylinder 8, wrong target 2 이하, 충돌 1 이하, approach 중 touchdown 0. 강제 이동은 없다.

**Gate A의 두 번째 비행.** Cube와 cylinder 착륙 20개 시작을 한 번 더. Cube: 첫 비행 8 이상, 합 16/20 이상, 둘 다 실패 1개 이하. Cylinder: 첫 비행 8 이상, 합 18/20 이상, 둘 다 실패 1개 이하.

**Gate B — Same-colour (새 시작 24개).** 모두 다른 물체가 첫 시야에 있고 이름의 물체는 없다.

| | 기준 |
|---|---:|
| 이름의 물체에 착륙 | 22 / 24 이상 |
| 다른 물체에 착륙 | 0 |
| 다른 물체와 충돌 | 0 |
| 충돌 (모두) | 0 |
| 다른 물체에서 끝난 비행 | 0 |
| 먼저 보인 물체를 거절 | 22 / 24 이상 (90%) |

- **다른 물체에 접근:** 어느 decision에서든 다른 물체의 12m 안에 있고 이름의 물체보다 그 물체에 더 가깝다.
- **다른 물체에서 끝남:** 이름의 물체가 아닌 것에 착륙·접촉·충돌했거나, 다른 물체 12m 안에서 그쪽에 더 가까운 채 스스로 멈췄다.
- **거절:** 접근하지도, 거기서 끝나지도 않았다.

**Gate B의 두 번째 비행.** 24개 중 12개(cone이 다른 물체인 4개 모두 + 나머지 중 다른 물체가 가장 가까운 8개)를 한 번 더. 둘 다 실패한 시작 1개 이하, 두 번째 비행의 충돌 0.

**Regression (새 시작 36개, canonical validation의 구성).** `grounding_film`, Stage 1b의 checkpoint, 고른 checkpoint가 한 번씩. 고른 checkpoint를 `grounding_film`과 비교한다: success는 baseline − 2 이상, pad 착륙은 baseline − 1 이상, **충돌 0**, approach 중 touchdown 0, wrong target은 baseline + 1 이하.

**통과 조건.** 녹화 label의 stop 규칙, data의 입력 검사, Gate A, Gate A의 두 번째 비행, Gate B, Gate B의 두 번째 비행, regression이 **모두** 통과해야 Stage 1c 통과다. 그때에만 `generic_landing_v1`을 만든다.

**Gate 뒤의 기록용 비행 (판정과 무관, 미리 정함).** Gate B 시작의 twin 11개, teacher pilot의 강제 16개를 Stage 1b의 checkpoint와 고른 checkpoint로, Stage 1b가 실패한 validation 시작 셋과 regression 시작 셋을 고른 checkpoint로.

**시작 간격 규칙을 다시 읽음 (set을 뽑기 전에 config에 적음).**

- 6m 규칙은 "check로 비행하는 시작"을 "학습에 쓴 시작"에서 떼어 놓기 위한 것이다. 지금까지는 녹화 시작끼리도 6m를 지키게 했는데, 그 읽기로는 자리가 없다. Blue cone이 먼저 보이고 blue pad가 돌면 보이는 후보 시작의 99% 이상이 예전 시작의 6m 안에 있어 하나도 뽑히지 않았다.
- Stage 1c의 **녹화 시작**은 check로 비행하는 모든 시작(저장소 전체와 이 stage)에서 6m를 지키고, 예전 녹화 시작과는 가까워도 된다. 둘 다 data다.
- Stage 1c의 **check 시작**(gate A, gate B, pilot, teacher pilot)은 이 stage와 그 앞 checkpoint들이 학습에 쓴 모든 시작, 이전의 다른 check 시작, 서로에게서 6m를 지킨다. Stage 1과 Stage 1b의 check 시작(녹화된 적 없음)과 Stage 1의 녹화 시작(이 계열의 어느 checkpoint도 학습하지 않음)과는 가까워도 된다. 이렇게 하지 않으면 pad 가까이의 착륙 시작이 뽑히지 않았다.
- Regression set은 Stage 1b와 같이 예전 regression set의 시작과 가까워도 된다.
- 어느 set에도 예전 set의 시작과 같은 시작은 없다.

**Ceiling.** Executor는 바꾸지 않는다. 14m를 넘은 비행의 수와 최고 높이를 기록한다.

## 7. 녹화 전 확인

**Label (dry run).** 계획한 295개 episode를 새 teacher로 dry run했다(강제 이동 포함). 착륙 label 17,623개 중 stop 0, 3.0배 미만 0, stop 4연속 구간 0, 최소 3.125배. 끝나지 않은 dry run은 없다.

**Teacher pilot (simulator, finalizer 포함, 32개).**

| 강제 상태 | 비행 | 강제 이동이 일어나고 물체가 보임 | 회복해서 착륙 | 강제 이동이 끝난 곳 (가운데에서 / 윗면 위) | 내린 곳 |
|---|---:|---:|---:|---|---|
| `under_climb` | 4 | 4 | 4 | 2.7–4.4m / 1.6–2.0m | 0.08–0.92m |
| `edge` | 4 | 4 | 4 | 3.0–4.2m / 3.7–4.0m | 0.30–0.72m |
| `premature` | 4 | 4 | 4 | 3.6–3.8m / 1.3–2.0m | 0.17–1.32m |
| `stall` (두 번씩) | 4 | 4 | 4 | 0.05–0.64m / 0.7–2.7m | 0.13–0.64m |

| Same colour | 비행 | 이름의 물체에 착륙 |
|---|---:|---:|
| 다른 물체가 먼저 보이는 시작 | 12 | 12 |
| Twin (문장이 먼저 보이는 물체를 가리킴) | 4 | 4 |

- 강제 16 / 16, same colour 16 / 16, 충돌 0, wrong target 0, stop 규칙을 어긴 label 0.
- **목표를 아는 탐색 label 0.** 어느 camera에도 이름의 물체가 보이지 않던 270 decision에서 teacher의 label은 모두 canonical 탐색(오른쪽 회전, 또는 앞이 막혔을 때의 상승)이었다.
- 기준(강제 15 / 16 이상, same colour 16 / 16, 충돌 0, wrong target 0, 위반 0)을 통과했다.
- Teacher pilot은 두 번 비행했다. 처음 뽑은 계획에서 이 set의 seed가 선언한 범위를 넘어간 것을 발견해(다른 set과 겹친 seed는 없었다) seed 사용 방식을 고치고 모든 계획을 다시 뽑았다. 처음 계획의 32개도 같은 기준을 통과했다. 위 표는 다시 뽑은 계획의 것이다.

<!-- RESULTS -->
