# Stage 1d — Final Descent Continuation Correction

2026-10-12–13, 브랜치 `exp/generic-landing-stage1d-final-descent`. Stage 1([문서](generic_landing_active_search_v2.md)), Stage 1b([문서](generic_landing_stage1b_stop_safe_teacher.md)), Stage 1c([문서](generic_landing_stage1c_recovery_grounding.md))는 통과하지 못했고, 그 기록은 그대로 둔다.

**질문:** Stage 1c가 얻은 같은 색 물체 구분과 generic landing 능력을 유지하면서, 윗면 위 마지막 하강을 끝까지 이어가도록 정책을 고칠 수 있는가?

<!-- STATUS -->

고치는 것은 하나다: 윗면 위 마지막 2m에서 정책이 하는 일. 새 능력은 더하지 않는다. Model 구조, action 공간, stop 기준, `SAFE_RATIO`, finalizer, evaluator, 착륙면, executor는 바꾸지 않았다. Same-colour episode는 새로 녹화하지 않고 Stage 1c의 것을 replay한다.

동결한 baseline(`grounding_film`), Gaussian Blur 결과, Stage 1 / 1b / 1c의 결과는 바꾸지 않았다. Canonical test 48개와 Depot set은 비행하지 않았다. Active Search v2는 이 단계의 결과와 무관하게 시작하지 않는다.

## 출발점

| | Stage 1 | Stage 1b | Stage 1c |
|---|---|---|---|
| 결과 | 실패 | 실패 | 후보 pilot에서 멈춤 (gate는 비행하지 않음) |
| 남긴 것 | cube 6/10, 공중 정지 | 공중 정지 15 → 1, cube 7/10, 같은 색 물체로 감 | 늦은 후보: 같은 색 오착륙 0, cube 8/10, **공중 정지 2건 재발** |

- **시작 checkpoint:** Stage 1c의 update 2000 (`outputs/aerovla_oft/checkpoints/generic_landing_stage1c_run/update-2000`, hash는 config에 기록). Stage 1c의 규칙이 고른 후보가 아니었고 `generic_landing_v1`이 아니다. Stage 1c pilot에서 다른 물체 착륙 0, cube 8/10, cylinder 4/4, pad 10/10, validation L1이 넷 중 가장 낮았다.
- **봉인한 set:** Stage 1c가 만든 gate A(36), gate B(24)와 그 twin(11)은 어떤 학습된 정책도 비행하지 않았다. Regression(36)은 `grounding_film`과 Stage 1b만 비행했다. 이 넷은 Stage 1d의 학습, pilot, 후보 선택, debugging에 쓰지 않는다. 파일의 hash를 config에 적고 test가 확인한다.

## 1. 실패한 비행을 다시 읽음

Stage 1c pilot에서 update 1000 / 1500 / 2000이 같은 두 시작에서 스스로 멈춘 여섯 비행이다(`outputs/generic_landing_1d/summary/descent_failure_audit.json`).

| | Case 1: 63006 (red cube) | Case 2: 63009 (blue cube) |
|---|---|---|
| 윗면 높이 | 8.0m | 8.0m |
| 멈춘 높이 (윗면 위) | 0.21 / 1.38 / 1.07m | 1.87 / 2.13 / 2.35m |
| 가운데에서의 거리 | 2.76 / 2.60 / 2.82m (윗면 위) | 5.38 / 5.49 / 5.44m (**윗면 밖**) |
| 가운데의 방향 | 116–118° (옆·뒤) | 70–78° (옆) |
| Front / Down camera | 안 보임 / 보임 | 안 보임 / 보임 |
| 마지막 12 decision의 전진, 회전 | 최대 0.012m, 0.007 rad (사실상 0) | 최대 0.008m, 0.079 rad |
| 하강 명령 (update 2000, 마지막 12) | 0.49 → 0.41 → 0.42 → 0.37 → 0.26 → 0.11 → 0.21 → 0.11 → 0 → 0 → 0 → 0 | 0.24 → 0.08 → 0.14 → 0.16 → 0.18 → 0.10 → 0.05 → 0.10 → 0.07 → 0.01 → 0.07 → 0.07 |
| 마지막 4 decision의 stop ratio (2000) | 0.08, 0.07, 0.04, 0.08 | 0.93, 0.17, 0.84, 0.87 |
| 수직 명령의 부호 뒤집힘 | 0 | 0 (1500만 1) |
| 하강을 시작한 자리 (가운데에서, 윗면 위) | 2.93–2.94m, 5.9–7.6m | 5.74–6.01m, 4.6–5.2m |
| 하강을 시작한 뒤 멈출 때까지 | 17–23 decision | 21–28 decision |

여섯 모두 같은 모양이다. **전진도 회전도 없이 곧게 내려오다가 하강 명령만 0으로 줄었다.** 흔들린 것이 아니다.

**그 자리의 label.** 녹화된 cube 착륙 label을 같은 자리(거리 ±0.4m, 높이 ±0.6m)에서 세었다.

| | Stage 1b data | Stage 1c data |
|---|---|---|
| Case 1 근처의 label | 0개 | 0–5개. 있는 것은 거의 **상승**(4), 가운데가 앞에 있는 것뿐 |
| Case 2 근처의 label | 0개 | 2–4개, 모두 **상승**, 가운데가 앞에 있는 것뿐 |
| 그 가운데 가운데가 45° 이상 옆에 있는 것 | 0 | 0 |

윗면 위 낮은 곳(3m 이하)의 label을 가운데에서의 거리로 나누면:

| 가운데에서 | Stage 1b data | Stage 1c data |
|---|---|---|
| 0–1.75m | 하강 591 | 하강 1,173 |
| 1.75–2.675m | 없음 | 하강 31 |
| 2.675–3.1m | 없음 | **상승 20** |
| 3.1–3.6m | 없음 | 상승 43 |
| 3.6m 밖 | 없음 | 상승 66 |

- Stage 1c의 teacher는 2.675m에서 label을 "계속 하강(+0.25)"에서 "상승(−0.5)"으로 바꾼다. Case 1은 그 경계 바로 위(2.6–2.8m)다.
- 그리고 그 상태 — 낮고, 가운데가 옆이나 뒤에 있는 — 의 label은 어느 data에도 없었다. Teacher는 늘 가운데를 앞에 두고 내려오기 때문이다.
- 학습된 정책은 사진 한 쌍만 보고 그 전에 무엇을 했는지 모른다. 평평한 윗면 가까이에서는 down camera에 윗면만 보인다. 양쪽 label이 만나는 자리에서 수직 명령이 0이 되고, 전진과 회전도 0이면 그것은 stop이다.
- 이것은 관찰과 맞는 설명이고, 이 비행들만으로 증명한 것은 아니다.

**기존 teacher의 마지막 하강 크기.** 모든 teacher의 하강 label은 두 값뿐이다: 윗면 위 2.5m 초과에서 `[0, 0.5, 0]`, 그 아래에서 `[0, 0.25, 0]`(stop ratio 3.125). Canonical pad data(읽은 400 episode에서 263 / 452개), Stage 1b의 pad·cube·cylinder(361/524, 234/530, 246/558), Stage 1c(429/525, 394/1,082, 20/92) 모두 같다. **Cube의 마지막 하강도 이 값을 그대로 쓴다.** 더 큰 값을 넣지 않았다.

## 2. Teacher: 자리로 정하는 하강 규칙

`src/landing/teacher.FinalDescentTeacher`. Stage 1c의 teacher에서 둘만 바꿨다(pad보다 작은 윗면에서만).

| | Stage 1c | Stage 1d |
|---|---|---|
| 낮게 내려온 뒤 하강을 이어가는 반경 (B) | 2.675m | **3.3m** |
| 윗면 위에서 올라가야 할 때 | 윗면 위 1m 아래면 제자리 상승 | 가운데로 가면서 상승 (제자리 상승은 윗면 **옆**이고 1m 아래일 때만) |

**두 반경.**

- **A = 1.75m (descent safe).** Canonical align 반경. 이 안에서는 어느 높이에서든 하강을 시작한다(`DESCENT_COMMIT`). 모든 이전 단계와 같다.
- **B = 3.3m (hard recovery).** 쓸 수 있는 윗면의 끝(3.6m)에서 하강 중의 수평 drift를 뺀 값. Stage 1b teacher가 녹화한 cube·cylinder 착륙 117개에서 마지막 1.5m 동안의 수평 이동은 중앙값 0.18m, 최대 0.28m였다. 올림해서 0.3m. B 안에서 낮게 있는 기체는 그대로 내려가면 영역 안에 닿는다. Pad는 canonical 3.5m 그대로다.

**규칙 (자리의 함수, 기억이 아님).**

| 기체가 있는 곳 | Label | 이름 |
|---|---|---|
| 가운데에서 A 이내 | 하강 | `DESCENT_COMMIT` |
| 윗면 위 3m 이하, A–B 사이 | 하강 (가운데가 어느 쪽에 있든) | `DESCENT_CONTINUE` |
| 윗면 위 3m 초과(들어오는 높이), A–B 사이 | 가운데로 전진, 하강 없음 | `RECENTER` |
| B 밖이거나 윗면 옆, 낮음 | 하강 없음. 상승 (가운데 쪽으로 돌면서, 길이 열려 있으면 전진) | `RECOVER_CLEARANCE` |
| 들어오는 높이(윗면 위 3.5m)를 되찾은 뒤 | 가운데로 → A 안에서 하강 | `RECENTER` → `DESCENT_COMMIT` |
| 닿은 뒤 | 0 (stop) | `TOUCHDOWN` |

- **Hysteresis.** A와 B 사이 1.55m가 그 띠다. 들어오는 기체는 A 전에 내려가지 않고, 낮은 기체는 B 전에 하강을 멈추지 않는다. 작은 흔들림으로 label이 하강 ↔ 상승으로 바뀌지 않는다.
- **왜 기억이 아니라 자리인가.** Teacher가 "아까 하강을 시작했으니 계속"이라고 기억으로 정하면, 같은 사진에 두 가지 label이 생긴다. Student는 사진 한 쌍만 본다. 그래서 띠 안에서의 label은 높이로 갈린다: 낮으면 하강, 들어오는 높이면 가운데로.
- **Label의 이름.** 학습의 balance가 teacher state 이름을 쓰므로 기록되는 state는 canonical 이름(`descend`, `climb`, `final`, `align`, `landed`) 그대로다. 위 표의 이름은 `descent_phase()`가 같은 decision에 붙이는 설명이다.
- **Stop 규칙.** 닿기 전의 모든 label은 stop 규칙의 3.0배 이상이다(하강 0.25m = 3.125배, 상승 0.25m 이상, 제자리 회전 0.21 rad = 6.25배).

## 3. 강제 상태 넷

Teacher는 스스로 이런 상태에 들어가지 않는다. 그래서 Stage 1c처럼 **강제 이동**으로 만든다. 강제 이동은 비행하지만 label이 아니고, 그 decision이 든 label window는 data에서 빠진다. 강제 이동이 끝난 뒤의 frame이 label이다.

| | 상태 | 만드는 법 | Teacher의 label |
|---|---|---|---|
| D1 | `CENTERED_LOW_HOVER` (`low_hover`) | 가운데 옆 0.3–2.6m를 지나게 한 뒤 윗면 위 0.3–2.0m로 내려 4–8 decision 동안 거의 정지 | 하강, 닿을 때까지 |
| D2 | `CENTERED_DESCENT_DECAY` (`descent_decay`) | 같은 자리에서 윗면 위 1.8–2.8m까지 내린 뒤, 하강을 매 decision 0.55–0.75배로 줄임 (5–8 decision) | 하강, 같은 크기로, 닿을 때까지 |
| D3 | `EDGE_LOW` (`edge_low`) | 가운데 옆 4.0–5.2m를 지나게 한 뒤 윗면 높이 위 0.5–2.0m로 내림 (가장자리·모서리 위 또는 윗면 옆) | 하강 없음. 상승 → 가운데 → 하강 |
| D4 | `EDGE_AFTER_PARTIAL_DESCENT` (`edge_drift`) | 가운데 가까이에서 윗면 위 1.2–2.5m까지 내린 뒤 수평으로 밀어 3.6–4.4m 밖으로 (가운데가 뒤에 옴) | 하강 없음. 돌면서 상승 → 가운데 → 하강 |

- 한 episode의 강제 이동은 둘이다: 가운데에서 얼마나 떨어져 지나갈지 정하는 Stage 1c의 `edge` 이동과, 상태를 만드는 이동.
- 둘 다 teacher 상태가 아니라 **위치**에서 일어난다. 그래서 같은 episode를 학습된 정책에게도 그대로 날릴 수 있다(pilot의 12개, qualification의 4개).
- D1과 D2의 "centered"는 "하강을 이어가는 부분(B) 안"이라는 뜻이다. 실패한 비행이 2.6–2.8m였으므로 가운데 바로 위만이 아니라 B 안 전체에 걸쳐 만든다.
- 녹화에 넣는 조건: teacher가 영역 안에 착륙하고 스스로 멈췄고, 두 강제 이동이 모두 일어났고, 각각 끝난 decision에 camera가 물체를 보고 있었고, 두 번째가 끝났을 때 기체가 아직 공중(윗면 위 0.15m 이상)이었다.

## 4. 계획 (녹화 전에 고정)

숫자는 모두 [configs/generic_landing_stage1d.json](../configs/generic_landing_stage1d.json), 시작은 `configs/generic_landing_stage1d/`에 있다.

**Data.** 녹화 80 episode + validation loss용 8.

| | Cube | Cylinder | Pad | 합 |
|---|---:|---:|---:|---:|
| D1 `low_hover` | 20 | 2 | 2 | 24 |
| D2 `descent_decay` | 16 | 2 | 2 | 20 |
| D3 `edge_low` | 18 | 1 | 1 | 20 |
| D4 `edge_drift` | 14 | 1 | 1 | 16 |
| 합 | 68 (85%) | 6 | 6 | 80 |

**Replay.** 뽑는 frame의 25%는 Stage 1d, 30%는 Stage 1c의 교정 data(recovery와 same-colour 전부), 25%는 Stage 1b의 stop-safe data, 20%는 그 이전 data다. Stage 1의 generic landing episode는 0개다.

**Replay에서 빼는 decision.** Stage 1c가 녹화한 label 가운데, 같은 자리에서 Stage 1d teacher가 주지 않을 것은 replay하지 않는다: 작은 윗면 위, 윗면 위 3m 이하, A와 B 사이에 있는데 녹화된 label이 상승인 decision. **40 decision, 25 episode**다(Stage 1b data에는 0). 그 decision이 든 label window만 빠지고 episode의 나머지는 그대로이며, label을 고쳐 쓰지 않는다. 같은 사진에 반대 label 둘을 주지 않기 위해서다. 목록은 `outputs/generic_landing_1d/summary/left_out_labels.json`.

**학습.** Stage 1c의 update 2000에서 시작. LoRA, FiLM, action head만 학습하고 module은 더하지 않는다. Learning rate는 Stage 1c의 절반(1.25e-5), 1,500 update. 후보는 update 500 / 1000 / 1500. 결과를 보고 늘리지 않는다.

**Seed.** Stage 1d는 80000번대를 쓴다(Stage 1c는 60000번대).

**Pilot (새 시작 24개).** 강제 cube 착륙 12(상태마다 3), same-colour 착륙 6, pad 3, cylinder 3. Stage 1c의 update 2000이 먼저 비행하고 후보 셋이 한 번씩. 하강 비교를 위해 Stage 1b의 checkpoint가 강제 12개를 비행한다(판정과 무관).

- 순서: same-colour 충돌 → 다른 물체 착륙 → 강제 cube 착륙 성공 → 공중 정지 → cube 착륙 → cylinder → pad → 14m 초과 비행 수 → validation L1 → 이른 update.
- 기준(`pass`): same-colour 충돌 0, 다른 물체 착륙 0, 강제 cube 착륙 10/12 이상, 그 12개의 공중 정지 1 이하, pad 3/3, cylinder 2/3 이상, 강제 cube 착륙이 Stage 1c의 update 2000보다 적지 않음.
- **고르는 법.** 후보마다 기준을 댄다. **기준을 모두 넘은 후보들**을 순서대로 세우고 첫째가 qualification으로 간다. 아무도 넘지 못하면 pilot은 통과하지 못한 것이고, qualification도 gate도 비행하지 않으며 더 학습하지 않는다. (Stage 1c는 순서의 첫째만 기준에 댔다. 이번 계획서는 "모두 실패하면" gate를 열지 않는다고 했으므로, 기준은 후보마다 댄다.)

**Qualification (새 시작 20개).** 고른 후보가 한 번 비행한다. Cube 착륙 10(상태마다 강제 1개씩 4 + 강제 이동 없는 6), same-colour 4, pad 3, cylinder 3.

| | 기준 |
|---|---:|
| Cube 착륙 | 8 / 10 이상 |
| Same-colour: 이름의 물체에 착륙 | 4 / 4, 충돌 0 |
| Pad | 3 / 3 |
| Cylinder | 2 / 3 이상 |
| 공중 정지 (20개 전체) | 1 이하 |
| Wrong target | 0 |
| 충돌 | 0 |

통과하지 못하면 gate는 비행하지 않고 Stage 1d는 실패다.

**봉인한 Gate A / Gate B / Regression.** Pilot과 qualification을 **둘 다** 통과해야 연다. Stage 1c의 파일을 Stage 1c의 code로, Stage 1c의 숫자 그대로 판정한다(`scripts/generic_landing_1c.py gate / repeat`).

- **Gate A (36):** success 30, 착륙 24/30, pad 9, cube 8, cylinder 8, wrong target 2 이하, 충돌 1 이하, approach 중 touchdown 0. 두 번째 비행: cube 첫 비행 8 이상·합 16/20 이상·둘 다 실패 1 이하, cylinder 합 18/20 이상.
- **Gate B (24):** 이름의 물체에 착륙 22 이상, 다른 물체 착륙 0, 다른 물체와 충돌 0, 충돌 0, 다른 물체에서 끝남 0, 거절 22 이상. 두 번째 비행(정해 둔 12개): 둘 다 실패 1 이하, 충돌 0.
- **Regression (36):** `grounding_film`과 비교해 success −2 이내, pad 착륙 −1 이내, wrong target +1 이내, 충돌 0, approach 중 touchdown 0. Stage 1d는 한 줄을 더한다: **이름이 아닌 물체에 착륙 0**.

**통과 조건.** 녹화 label의 stop 규칙, 입력 검사, replay 검사, pilot, qualification, Gate A와 두 번째 비행, Gate B와 두 번째 비행, regression이 **모두** 통과해야 Stage 1d 통과다. 그때에만 `generic_landing_v1`을 만든다.

**시작 간격.** Stage 1c의 규칙에 하나를 더했다. Stage 1d의 check 시작(pilot, qualification, teacher pilot)은 이 계열의 checkpoint가 학습한 모든 시작, 서로, Stage 1c의 봉인한 gate·regression·pilot에서 6m를 지킨다. Stage 1c의 **teacher pilot** 시작과는 가까워도 된다. Teacher만 비행했고 어떤 정책도 비행하지 않았으며 녹화되지 않은 시작이다. 그것까지 피하면 "blue cube가 먼저 보이고 blue pad는 안 보이는" 새 시작이 하나도 남지 않았다(계획을 처음 뽑을 때 발견, 계획 파일이 생기기 전에 config에 적음). 녹화 시작은 그것도 포함해 check로 비행하는 모든 시작에서 6m를 지킨다.

**Ceiling.** Executor는 바꾸지 않고 clamp도 넣지 않는다. 모든 pilot, qualification, gate에서 14m를 넘은 비행의 수, 최고 높이, 이름을 기록한다.

## 5. 녹화 전 확인

**강제 이동의 물리 확인 (계획을 뽑기 전, 어느 set에도 없는 시작 8개).** 명령을 2.4초 늦게 따르는 기체가 강제 이동에서 얼마나 가라앉고 흘러가는지 보려고 teacher가 simulator에서 비행했다. 8개 모두 착륙했다.

| 상태 | 두 번째 강제 이동이 끝난 자리 (가운데에서, 윗면 위) | 그 뒤 |
|---|---|---|
| `low_hover` | 1.19m, 1.68m / 2.48m, 1.65m | 곧바로 하강, 그 자리에 착륙 (오차 1.19m, 2.48m) |
| `descent_decay` | 1.21m, 1.52m / 2.03m, 1.07m | 곧바로 하강, 그 자리에 착륙 |
| `edge_low` | 3.59m, 2.03m / 3.85m, 1.57m | 상승 1번 → 가운데 → 하강 (오차 0.28m, 0.47m) |
| `edge_drift` | 3.80m, 1.03m / 4.13m, 1.03m | 상승 1번 → 가운데 → 하강 (오차 0.43m, 0.18m) |

`edge_low`에서 기체가 가라앉는 동안 안쪽으로 0.45–0.6m 흘렀다. 그래서 그 상태의 지나가는 거리를 3.6–5.0m에서 4.0–5.2m로 옮겼다(계획을 뽑기 전).

**Dry run 감사.** 계획한 모든 episode(녹화 88 + check 64)를 teacher의 dry run으로 비행해 label을 검사했다: 닿기 전 stop 0, 3.0배 미만 0, stop 4연속 0, 끝나지 않은 dry run 0.

**Teacher pilot (simulator, finalizer 포함, 강제 cube 착륙 20개).**

| | 결과 | 기준 |
|---|---:|---:|
| 착륙 (두 강제 이동이 일어났고, 끝난 자리에서 물체가 보였고, 두 번째가 공중에서 끝났고, 착륙함) | 20 / 20 | 19 이상 |
| 충돌 | 0 | 0 |
| Wrong target | 0 | 0 |
| 닿기 전의 stop label, stop 4연속 | 0 | 0 |
| 3.0배 미만 label | 0 | 0 |
| 강제 상태가 보이지 않았거나 일어나지 않음 | 0 | 0 |

**PASS.**

| 상태 | 상태가 놓인 자리 (가운데에서 / 윗면 위) | 하강이 시작된 자리 (가운데에서 / 윗면 위) | Touchdown 오차 | 수직 label의 부호 바뀜 | 상승 횟수 |
|---|---|---|---|---:|---:|
| D1 `low_hover` | 1.13–1.84m / 0.59–1.84m | 같은 자리 | 1.13–1.84m | 0 | 0 |
| D2 `descent_decay` | 0.76–2.20m / 1.19–1.78m | 같은 자리 | 0.76–2.20m | 0 | 0 |
| D3 `edge_low` | 3.86–4.52m / 1.50–1.97m | 1.43–1.74m / 3.70–3.90m | 0.13–0.52m | 비행마다 1 | 비행마다 1 |
| D4 `edge_drift` | 3.92–4.38m / 1.02–1.93m | 1.50–1.72m / 3.77–3.90m | 0.20–0.43m | 비행마다 1 | 비행마다 1 |

- D1, D2: 놓인 자리에서 곧바로 내려가 그 자리에 닿는다. 상승도, 부호 바뀜도 없다.
- D3, D4: 한 번 올라가고(부호 바뀜 1번) 가운데로 가서 A 안에서 내려온다. Teacher가 오르내림을 되풀이한 비행은 없다.
- 20 비행 모두 실시간 배율 0.997 이상, 최고 높이 11.6–11.9m(ceiling 14m 아래)다.

<!-- TEACHER_PILOT -->

<!-- RESULTS -->
