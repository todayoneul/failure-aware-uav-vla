# AeroVLA-OFT Gen-v3c — hard-negative grounding 보정과 canonical baseline

2026-10-09 실행. 브랜치 `exp/aerovla-oft-gen-v3c-hard-negative`(`exp/aerovla-oft-gen-v3-canonical-baseline`에서 분기). [Gen-v3](aerovla_oft_gen_v3.md)의 두 번째 checkpoint(`generalization_v3b`), [canonical 검증 결과](canonical_clean_baseline.md)(30/36), canonical test set 48개, Depot의 156개, Gen-v2의 동결 파일은 그대로 있다.

> **이후 (구조 실험, 2026-10-10):** 이 문서의 checkpoint에서 같은 data로 2,000 update를 더 학습하자 pilot의 실패가 사라졌다(FiLM을 넣은 쪽 32/32, 넣지 않은 대조 30/32). FiLM을 넣은 checkpoint는 새 검증 gate(34/36)와 canonical test 48개(47/48)를 통과했다. 아래의 "데이터 보정으로는 부족하다"는 판정은 검증 L1 규칙이 남긴 update 1,250의 checkpoint에 대한 것으로 읽어야 한다. → [Language–vision grounding architecture](language_vision_grounding_architecture.md). 이 문서의 숫자는 그대로 두었다.

**요약**

- **Gate를 다시 통과하지 못했다.** 새 검증 36회에서 smoke 10/10, representative 33/36(0.92). 성공률, 착륙, 접근 시작, 끝맺음, 충돌은 기준을 넘었고, **다른 물체에서 끝난 비행이 3회**(기준 2회 이하)였다. 이전과 같은 줄이다.
- **그래서 하지 않은 것:** 동결, canonical test 48개 비행. Test set은 여전히 어떤 모델도 비행하지 않았다.
- **이번 질문의 답은 NO다.** "먼저 보이는 비슷한 물체"를 가르치는 hard-negative 96 episode와 짧은 보정 학습으로는, 그 조건의 오류가 다섯 번에 한 번꼴에서 구별될 만큼 줄지 않았다(시작 checkpoint 15회 중 4회, Gen-v3c 21회 중 4회).
- **줄어든 것은 있다.** Pilot의 같은 12개 시작에서 9/12 → 11/12(다른 물체 2 → 1, 충돌 1 → 0). 같은 색 물체가 화면 가운데 있는 frame에서 그쪽으로 전진하는 비율은 33% → 21%.
- **한 번의 비행으로는 단정할 수 없다.** Pilot에서 성공한 `blue pad ← blue cone` 시작은 세 번 비행에 한 번 충돌했고, 성공한 두 번도 cone 쪽으로 몇 번 전진했다가 돌아선 것이다. 이 조건에서의 행동은 경계 위에 있다.
- **그 조건 밖은 깨끗하다.** 관련 물체가 먼저 보이지 않는 검증 시작은 21/21. 지시한 pad로 간 착륙 24회는 전부 안정 착륙·latch·disarm까지 갔고 충돌은 0이다.
- **실패가 나는 자리가 더 좁혀졌다.** 관련 물체가 화면에 오래 머무는 시작이다: 탐색이 도는 쪽(오른쪽)에 있거나, 정면 20m 이내. 그런 18회 중 8회가 다른 물체에서 끝났고, 나머지 18회에서는 0회다.
- **Evaluator v2를 넣었다.** "서 있음"을 보고된 속도 대신 접촉과 위치로 읽는다. 이전 결과(30/36)는 다시 계산하지 않았다.
- **다음은 데이터가 아니라 구조다.** 계획의 원칙대로 같은 데이터를 더 넣지 않고, FiLM 또는 더 강한 language–vision 결합을 다음 후보로 권한다. 이번에는 구현하지 않았다.
- **Gaussian Blur로 넘어갈 준비는 되지 않았다.**

## 이 문서의 세 가지 결과는 서로 다른 것이다

| 이름 | 모델 | 시작 set | 평가 방식 | 어디 |
|---|---|---|---|---|
| Gen-v3b의 canonical 검증 (이전) | `generalization_v3b` | 검증 36개, seed 8500번대, 학습 배치 a–d | legacy evaluator | [canonical_clean_baseline.md](canonical_clean_baseline.md). 30/36. 다시 계산하지 않았다 |
| Gen-v3c의 새 검증 | `generalization_v3c` | 검증 36개, seed 22000번대, 새 배치 p–s | `canonical_evaluator_v2` | 이 문서 |
| Gen-v3c의 canonical test | `generalization_v3c` | test 48개, seed 9500번대, 새 배치 g–j | `canonical_evaluator_v2` | 이 문서. gate 통과 뒤 한 번만 |

## 왜 Gen-v3c인가

- **이전 canonical 검증은 30/36이었고, gate는 한 줄에서 걸렸다:** 다른 물체에서 끝난 비행 3회(기준 2회 이하).
- **실패의 모양은 한 가지였다.** 지시한 것과 색이나 형태가 같은 다른 물체를 목표로 삼는다: blue pad 대신 blue cube, red pad 대신 red cube, blue pad 대신 red pad.
- **조건도 좁다.** 2회는 그 물체가 먼저 보이고 지시한 pad는 화면 밖에 있던 시작에서 났다. 1회는 지시한 pad로 가다가 놓친 뒤 같은 색 물체로 갔다.
- **이번 질문은 하나다.** "먼저 보이는 비슷한 물체"를 데이터로 분명히 가르치면, 지금의 구조(Front/Down RGB와 문장, LoRA r16, L1 head, FiLM·proprio 없음)가 색과 명사를 묶어 맞는 목표를 안정적으로 고를 수 있는가.

바꾼 것은 둘뿐이다: **(A) 착륙 평가의 "서 있음" 판정**, **(B) hard-negative 약 100 episode를 더한 짧은 보정 학습**. 카메라, 탐색 고도, action head, chunk, FiLM, proprio는 그대로다.

## Evaluator v2

### 무엇이 문제였나

이전 검증에서 착륙 2회가 실패로 기록됐다. 기체는 지시한 pad 위에서 4 판단 동안 1mm도 움직이지 않았고, 정책은 0 행동을 냈고, finalizer는 latch하고 disarm했다. 그런데 "서 있음" 판정이 simulator가 보고한 수직 속도 0.05m/s 이하를 요구했고, 그 두 비행에서는 0.04–0.095m/s가 보고됐다. 모델의 실패가 아니라 측정의 문제다.

### 새 정의 (`canonical_evaluator_v2`)

[configs/targets/landing_pads.json](../configs/targets/landing_pads.json)의 `evaluator`, [scripts/visual_search.py](../scripts/visual_search.py)의 `standing`·`landing_summary`. 보고된 속도는 "서 있음"에 쓰지 않는다.

| 조건 | 읽는 것 |
|---|---|
| 지시한 pad의 착륙 구역 | pad 위쪽에서의 접촉 + 접촉 위치가 중심에서 가로·세로 6.6m 이내 (이전과 같음) |
| 접촉 높이와 일치 | 마지막 3 판단의 높이가 모두 접촉 높이에서 0.05m 이내 |
| 위치가 안정 | 그 3 판단 사이의 수평 이동 0.05m 이하, 수직 이동 0.05m 이하 |
| Latch와 disarm | finalizer가 latch하고 disarm함 |
| 다른 접촉 없음 | pad 말고 닿은 것이 없음 |

- **Touchdown success:** 지시한 pad + 구역 안 + 접촉 직전 수직 속도 0.75m/s 이하. 속도를 읽는 판단을 위치로 찾는다(pad 위에 선 첫 판단의 바로 앞).
- **Physical landing:** 어느 pad든 그 위에 안정되게 서 있음(위의 둘째·셋째 줄). 어느 pad인지는 따로 본다.
- **Stable physical landing:** touchdown success + physical landing + 다른 접촉 없음. Finalizer의 말을 빌리지 않는다.
- **System landing success:** 위 + latch + disarm. 착륙 mission의 성공이다.
- **Strict policy zero-action:** 지시한 pad에 내려앉은 뒤 정책이 스스로 0 행동을 4번 냄. 보조 지표. `finalizer_rescued_success`는 system은 성공했는데 strict는 아닌 비행이다.

### 임계값은 기록에서 나왔고, 비행 전에 고정했다

Teacher 착륙 63회와 모델 착륙 24회의 기록에서 3 판단짜리 window를 쟀다(`bd79191`에 고정, 어떤 새 비행도 판정하기 전).

| 3 판단 사이의 위치 변화 | 수평 | 수직 |
|---|---|---|
| Pad 위에 서 있음 (window 194개) | 최대 0.0000m | 최대 0.0003m |
| 하강의 마지막 판단이 섞인 window | 최대 0.03m | 최대 0.20m |
| 마지막 하강 중 | 최대 0.67m | 0.45m 이상 |
| 접근 뒤 공중 정지 | 0.0–1.4m (중앙값 0.6m) | 0.001–0.05m |

- **0.05m는 "서 있음"의 100배가 넘고 "하강 중"의 1/9이다.** Finalizer 자신의 한계(0.1m/s)가 한 판단에 허용하는 거리이기도 하다.
- **가만히 있다는 것만으로는 착륙이 아니다.** 공중 정지도 수직으로는 0.05m 안에서 떠 있다. 접촉과 높이가 둘을 가른다.

### Finalizer는 그대로다

안정 접촉 → latch → 명령 차단 → disarm. 숫자도 `7249006`의 것 그대로다. 목표가 무엇인지, 좌표, 거리, 맞는 pad인지는 여전히 받지 않는다. 맞는 pad인지는 평가만 판정한다.

### 이전 결과는 고치지 않았다

- **Legacy evaluator result:** 이전 canonical 검증의 30/36, 착륙 21/27은 그대로다. 새 판정으로 다시 계산해 gate를 통과시키지 않았다.
- **새 비행은 두 판정을 함께 저장한다.** `legacy_evaluator` 아래에 이전 방식의 값이 남는다.
- **확인:** 그 두 비행의 기록을 test 입력으로 두었다(`tests/data/noisy_velocity_landings.json`). 새 판정은 둘 다 안정 착륙으로 읽고, 이전 판정은 저장된 값과 같게 읽는다. 이것은 판정의 회귀 test이지 이전 gate의 재판정이 아니다.

## Hard-negative data

### 가르치는 규칙

1. **먼저 보이는 물체가 문장과 완전히 맞지 않으면, 다가가지 말고 계속 찾는다.**
2. **맞는 목표를 따라가다 잠깐 놓쳤다고 해서, 같은 색의 다른 물체로 갈아타지 않는다.**

Teacher는 원래 이렇게 한다(목표가 화면에 없으면 다른 것이 보여도 계속 돈다). 더한 것은 그 장면을 일부러 만드는 시작이다. 목표의 좌표·거리·방위는 여전히 teacher와 평가만 쓴다.

### 종류

| 종류 | 무엇 | Train | Val |
|---|---|---:|---:|
| `color_first` | 지시한 것과 **색이 같고 형태가 다른** 물체가 첫 화면 가운데(±20°, 48m 이내)에 있고, 지시한 것은 화면 밖 | 30 | 4 |
| `shape_first` | **형태가 같고 색이 다른** 물체가 첫 화면 가운데에 있고, 지시한 것은 화면 밖 | 14 | 2 |
| `lost` | 지시한 것이 보이고 접근을 시작한 뒤, 강제 회전(비행은 하되 label은 아님)으로 관련 물체가 화면 가운데에 오고 지시한 것은 화면 밖으로 | 20 | 2 |
| `swap` | `*_first`의 자세 그대로, 두 물체의 자리를 바꾼 배치에서. 같은 화면에 이제 지시한 것이 있다 | 16 | 2 |
| `query` | `*_first`의 자세 그대로, 화면에 보이는 물체를 지시. 문장만 다르다 | 16 | 2 |
| 합계 | | **96** | **12** |

- **첫 화면에 지시한 것이 없는 시작:** `color_first` + `shape_first` = 44(train). `swap`·`query`·`lost`는 보이는 상태에서 시작한다.
- **`lost`의 강제 회전은 양쪽이다.** Train에서 왼쪽 9, 오른쪽 11. Teacher는 항상 오른쪽으로 찾으므로, 왼쪽으로 틀어지면 금방 되찾고 오른쪽으로 틀어지면 거의 한 바퀴를 돌면서 관련 물체를 다시 지나친다.
- **전부 query twin으로 채우지 않았다.** 96개 중 16개다.
- **녹화:** 108개 모두 teacher가 끝냈다(착륙 36회 포함, 버린 episode 0). Frame 4,933개(train 4,437, val 496). 지시한 물체가 화면에 없는 frame이 1,156개다.

예시: [dataset_samples.jpg](../outputs/examples/gen_v3c/dataset_samples.jpg). 같은 화면에서 문장이나 배치만 바뀌면 teacher의 답이 "계속 돈다"와 "간다"로 갈린다. 계획 전체: `outputs/examples/gen_v3c/hard_negative_plan.json`.

### 색, 형태, 거리의 분포 (train 96)

| 지시한 물체 | 수 | | 지시한 것 ← 화면에 보이는 것 (base episode와 query twin) | 수 |
|---|---:|---|---|---:|
| blue pad | 27 | | blue pad ← blue cube / blue cylinder / blue cone | 5 / 4 / 5 |
| red pad | 19 | | blue pad ← red pad | 8 |
| blue cube | 13 | | red pad ← red cube | 7 |
| red cube | 12 | | red pad ← blue pad | 8 |
| blue cylinder | 14 | | blue cube ← blue pad / blue cylinder / red cube | 4 / 3 / 3 |
| blue cone | 11 | | red cube ← red pad / blue cube | 7 / 3 |
| | | | blue cylinder ← blue pad / blue cube / blue cone / green cylinder | 5 / 3 / 3 / 2 |
| | | | blue cone ← blue pad / blue cylinder / blue cube | 5 / 3 / 2 |

- **색:** 파랑 65, 빨강 31. 빨간 물체는 pad와 cube뿐이라 반반은 되지 않는다. "파랑이면 더 조심한다"는 지름길을 만들지 않으려고 red pad ↔ red cube, red ↔ blue 쌍을 양방향으로 넣었다.
- **형태:** pad 46, cube 25, cylinder 14, cone 11. blue pad만 목표인 것이 아니다(27/96).
- **관계:** 같은 색·다른 형태 56, 같은 형태·다른 색 24(swap twin 16개는 관계가 없다. 화면에 지시한 물체가 있다).
- **거리:** 모든 시작이 지시한 물체에서 12–44m. 32–44m 37, 20–32m 39, 12–20m 20. 첫 화면의 관련 물체는 15–47m 거리다.
- **끝:** 접근 63, 착륙 33. pad를 지시한 46개 중 33개가 착륙이다.
- **장면:** Field 54, Lot 42.

### 배치: 네 가지 용도가 서로 겹치지 않는다

Field와 Lot에 배치를 열 개씩 더했다([configs/maps/field.json](../configs/maps/field.json), [lot.json](../configs/maps/lot.json)). 자리(site)는 그대로이고, 어느 물체가 어디 서는지가 다르다.

| 용도 | 배치 | 관련 물체 쌍 (42–65m 간격) |
|---|---|---|
| 기존 학습 (replay) | a–d | 그대로 |
| Hard-negative 학습 | k, l, m, n | blue pad–blue cone / blue cube / blue cylinder / red pad, red pad–red cube, blue cube–red cube, blue cylinder–blue cone / green cylinder |
| Pilot | t, u | blue pad–blue cube / blue cone, red cube–red pad, blue cube–red cube |
| 새 검증 | p, q, r, s | blue pad–blue cone / blue cube / red pad, red pad–red cube |
| Canonical test (봉인) | g, h, i, j | 그대로 |

- **Hard-negative episode는 평가 set의 시작도 배치도 쓰지 않는다.** 이전 canonical 검증 36개, canonical test 48개, Depot의 156개, pilot 12개, 새 검증 36개의 모든 시작에서 6m 이상 떨어져 있다. 이전 검증의 실패 위치를 다시 비행한 episode는 없다. 실패의 종류만 썼다.
- **학습 배치는 test 배치와 이웃 쌍을 하나도 공유하지 않는다.** "어느 두 자리에 어느 두 물체"가 겹치지 않게 했다. 새 검증 배치는 학습·pilot·test 어느 것과도 겹치지 않는다.
- **자리를 바꾼 배치 쌍(k↔l, m↔n, t↔u, p↔q, r↔s)은 바닥과 조명이 같다.** 두 물체의 자리만 다르다.
- **이전 검증 36개는 학습 배치(a–d)에서 비행한 것이다.** 그 배치는 그 전부터 학습 data의 것이라 replay에는 남아 있다. 새 episode는 그 배치를 쓰지 않는다.

### Replay

Hard-negative만으로 학습하지 않는다. 뽑는 frame의 절반은 새 episode에서, 절반은 checkpoint가 학습한 전체 data에서 온다. 탐색, 보통의 grounding, 접근, 착륙, approach 대 land twin이 모두 그 안에 있다.

## 보정 학습

구조는 그대로다: OpenVLA-7B NF4, 고정한 AeroVLA adapter, LoRA r16, 선형 L1 head, chunk 4·execute 1, FiLM·proprio 없음. 설정은 [configs/visual_search.json](../configs/visual_search.json)의 `gen_v3c.fine_tune`에 학습 전에 적었다.

| | 값 | 근거 |
|---|---|---|
| 시작 | `generalization_v3b` | 처음부터 다시 학습하지 않는다 |
| Update | 2,000 (frame 4개씩) | 계획의 1,000–3,000 범위 |
| Learning rate | 5e-5 → 5e-6 (선형 감소) | 처음 학습의 1/4. 그 학습은 2e-4에서 줄어 2.5e-5에서 끝났다 |
| 뽑는 비율 | 새 episode 50%, 이전 data 50% | `--share generalization_v3c_added=0.5` |
| 더 자주 뽑는 frame | 목표 선택이 일어나는 frame 3배, 먼 작은 목표 2배(이전과 같음) | 아래 |
| 검증 frame | 새 episode의 검증 frame 전부 + 같은 수의 이전 검증 frame | 새것과 옛것을 반씩 본다 |
| Checkpoint 규칙 | 이전과 같음: 검증 L1이 가장 낮은 것, 2% 안에서는 행동 점수 | `--select-band 0.02` |

- **"목표 선택이 일어나는 frame"** = hard-negative episode에서 지시한 물체가 화면에 없는 모든 frame(teacher는 계속 돈다)과, swap·query twin의 처음 8 frame(같은 화면에 지시한 물체가 있다). 새 frame의 약 1/5이고, 이 가중이 없으면 한 frame이 평균 한 번쯤 뽑힌다.
- **이 가중은 계획을 commit한 뒤, 학습을 시작하기 전에 더했다**(`89a64bf`). 그때까지 학습된 것도 비행한 것도 없었다.

### 결과

2,000 update를 모두 돌았다(84분, RTX 5070 12GB). 기록: `outputs/generalization/gen_v3c/training.json`(설정, 검증 이력, offline 판독 전부). 병합한 data는 1,160 episode / 57,329 frame이고, 그중 새 것이 108 episode / 4,933 frame이다.

| | 시작 (`generalization_v3b`) | Gen-v3c pilot checkpoint |
|---|---:|---:|
| 남긴 checkpoint | | update 1,250 (규칙대로. 마지막 update는 2% band 밖) |
| 검증 L1, 새것·옛것 반반 600 frame | 0.0607 | 0.0601 |
| 같은 frame의 행동 점수 (접근·회전 / 정지·하강이 teacher와 같은 비율) | 0.923 | 0.940 |
| 이전 검증 frame 400개의 L1 | 0.0561 | 0.0581 |
| 이전 검증 frame의 행동 점수 | 0.923 | 0.933 |
| 새 episode의 검증 frame 중 "계속 돈다" frame(90개)의 첫 행동 L1 | 0.0387 | 0.0067 |

- **검증 L1은 거의 그대로다.** 이미 낮았고, 새 frame의 대부분은 보통의 접근·착륙 frame이다. 바뀐 것은 "지시한 물체가 화면에 없을 때"의 행동이다.
- **이전 능력은 offline에서는 유지됐다.** 이전 검증 frame의 L1이 0.002 올랐고 행동 점수는 오히려 올랐다. 비행에서의 회귀는 아래에서 따로 본다.

### 같은 frame, 다른 문장 (offline probe)

새 episode에서 "관련 물체가 화면에 있고 지시한 물체는 없는" frame을 골라, episode의 문장과 화면에 보이는 물체를 지시하는 문장을 각각 주고 첫 행동의 종류를 봤다. 처음 세 판단의 frame이다.

| Episode의 문장 → "계속 돈다" (맞는 답) | 시작 checkpoint | Gen-v3c |
|---|---:|---:|
| 같은 색이 먼저 보임 (`color_first`), train 87 frame | 58 (67%) | 69 (79%) |
| 같은 형태가 먼저 보임 (`shape_first`), train 42 frame | 38 (90%) | 39 (93%) |
| 놓친 뒤 (`lost`), train 57 frame | 52 (91%) | 54 (95%) |
| 합계, train 186 frame | 148 (80%) | 162 (87%) |
| 합계, val 24 frame (학습에 안 쓴 episode) | 23 (96%) | 23 (96%) |

- **시작 checkpoint는 같은 색 물체가 화면 가운데에 있으면 세 번에 한 번은 그쪽으로 전진했다.** 비행에서 본 실패율(9회 중 2회)과 맞는다. 같은 형태·다른 색은 열 번에 한 번이다. 병목은 색이 같을 때다.
- **보정 뒤 그 비율은 33% → 21%로 줄었다. 없어지지 않았다.** 이 frame들은 Gen-v3c가 학습한 frame인데도 다섯 번에 한 번은 여전히 전진한다.
- **"화면에 보이는 물체를 지시하는 문장"으로 바꿨을 때 전진으로 바뀐 비율은 34% → 35%로 그대로다.** 문장만 바꿔서는 이 frame들에서 행동이 잘 바뀌지 않는다.
- **이 probe만으로 통과·실패를 정하지 않는다.** 기준은 비행이다.

## Pilot: 12회의 빠른 확인

새 검증 set을 쓰기 전에, 잘못된 목표로 가는 행동이 실제로 줄었는지 본다. 어떤 episode도 녹화하지 않은 배치 둘(t, u)에서 12회다([configs/gen_v3c_pilot.json](../configs/gen_v3c_pilot.json)). 기준은 비행 전에 고정했다: 성공 9회 이상, 다른 물체에서 끝난 비행 1회 이하, 충돌 1회 이하, 실행 오류 0. 통과하지 못하면 이 단계의 답은 **DATA CORRECTION INSUFFICIENT**이고, episode를 더 넣지 않는다.

| 시작 | 수 | 무엇 |
|---|---:|---|
| 같은 색·다른 형태가 먼저 보임 | 4 | blue pad ← blue cube, red pad ← red cube, blue pad ← blue cone, blue cube ← blue pad |
| 같은 형태·다른 색이 먼저 보임 | 2 | blue cube ← red cube, red cube ← blue cube |
| 놓친 뒤 관련 물체가 보임 | 2 | blue pad를 따라가다 강제 회전으로 blue cube / blue cone이 화면 가운데에 |
| 자리 바꿈 | 2 | 위 첫째·셋째 시작에서 두 물체의 자리만 바꿈 |
| 관련 물체 없는 탐색 | 2 | blue pad, red pad가 화면 밖. 탐색이 그대로인지 |

### 결과: 통과

Teacher는 12/12다. 시작 checkpoint도 같은 12개를 비행했다(비교용, 판정 대상이 아님).

| | 성공 | 다른 물체에서 끝남 | 충돌 | 판정 |
|---|---:|---:|---:|---|
| 시작 checkpoint (`generalization_v3b`) | 9/12 | 2 | 1 | 통과 못 함 |
| **Gen-v3c** | **11/12** | **1** | **0** | **통과** |
| 기준 | 9 이상 | 1 이하 | 1 이하 | |

| 시작 | 시작 checkpoint | Gen-v3c |
|---|---|---|
| blue pad ← blue cube (38.7m) | 성공 | 성공 |
| red pad ← red cube (30.8m) | red cube 앞에서 정지 | **red cube 앞에서 정지** |
| blue pad ← blue cone (34.8m) | blue cone으로 가서 충돌 | 성공 (blue pad에 착륙) |
| blue cube ← blue pad (24.5m) | 공중에서 정지 | 성공 |
| blue cube ← red cube, red cube ← blue cube | 2/2 | 2/2 |
| 놓친 뒤 blue cube / blue cone이 보임 | 2/2 | 2/2 |
| 자리 바꿈 | 2/2 | 2/2 |
| 관련 물체 없는 탐색 | 2/2 | 2/2 |

- **줄었다.** 같은 색 물체가 먼저 보이는 네 시작에서 1/4 → 3/4. 시작 checkpoint가 실패한 셋 중 둘이 고쳐졌다.
- **남았다.** red pad를 지시했는데 red cube가 먼저 보이는 시작은 두 checkpoint 모두 red cube로 갔다.
- **잃은 것은 없다.** 형태가 같은 물체, 놓친 뒤, 자리 바꿈, 보통의 탐색은 그대로 전부 성공이다.
- **기준을 통과했으므로 새 검증 set으로 넘어갔다.**

### 같은 시작을 다시 비행하면 (판정과 무관한 확인)

예시 화면을 녹화하려고 `blue pad ← blue cone` 시작 하나를 다시 비행했더니, pilot에서 성공했던 그 시작이 이번에는 blue cone에 충돌했다. 그래서 12개를 전부 한 번 더 비행했다(`outputs/generalization/gen_v3c/pilot_repeat.json`). Pilot의 판정은 처음 비행의 것이고, 바꾸지 않았다.

| | 성공 | 다른 물체에서 끝남 | 충돌 |
|---|---:|---:|---:|
| Pilot (판정에 쓴 비행) | 11/12 | 1 | 0 |
| 12개 전부 다시 | 11/12 | 1 | 0 |
| `blue pad ← blue cone` 하나만 따로 다시 | 0/1 | — | 1 |

- **12개를 다시 비행한 결과는 시작마다 처음과 같았다.** 대부분의 시작은 반복해도 같은 끝이 난다.
- **그런데 `blue pad ← blue cone` 시작은 세 번 중 한 번 충돌했다.** 세 비행 모두 처음 세 판단은 오른쪽으로 돌고(cone이 화면 가운데로 온다), 네 번째 판단에서 cone 쪽으로 전진한다. 성공한 두 번은 두세 번 전진하다가 다시 돌아섰고, 실패한 한 번은 돌아서지 않고 계속 전진해 부딪혔다.
- **즉 "성공"한 비행도 그 물체로 가다가 돌아선 것이다.** 이 조건에서의 행동은 경계 위에 있고, 같은 색 물체가 가운데 보이는 판단마다 "간다"와 "돈다"가 갈린다. Offline probe의 21%와 같은 이야기다.
- **읽을 때:** 이 조건의 시작 하나하나는 한 번의 비행으로 "고쳐졌다"고 말할 수 없다. Pilot의 11/12도, 검증의 "다른 물체 3회"도 한두 비행의 폭을 가진 숫자다.

## 새 검증 set (학습 전에 고정)

[configs/gen_v3c_canonical_validation.json](../configs/gen_v3c_canonical_validation.json). 이전 검증 36개는 개발에 쓰였으므로 다시 쓰지 않았다. Seed 22000–22028, 배치 p–s(어떤 episode도 녹화하지 않았고 다른 set도 쓰지 않는다), 모든 시작이 44m 이하. 계획·gate 기준과 함께 `bd79191`에 commit했고, 그 뒤에 녹화와 학습을 했다.

| Band마다 12개 (near 12–20m, mid 20–32m, far 32–44m) | 문장 | 수 |
|---|---|---:|
| `visible`, `search`, `edge` | blue pad 착륙 | 3 |
| `past_color`: 같은 색 물체(cube 또는 cone)가 첫 화면에, blue pad는 화면 밖 | blue pad 착륙 | 1 |
| `past_shape`: red pad가 첫 화면에, blue pad는 화면 밖 | blue pad 착륙 | 1 |
| `pair` + 자리 바꾼 twin(`swap`) + 곁의 물체를 묻는 twin(`query`) | blue pad 착륙 2, "Find the …" 1 | 3 |
| `red_past`: red cube(2 band) 또는 blue pad(1 band)가 첫 화면에, red pad는 화면 밖 | red pad 착륙 | 1 |
| `red_visible` | red pad 착륙 | 1 |
| `visible`·`search`의 approach twin | "Approach the blue landing pad." | 2 |

- **착륙 27, 접근 9.** Canonical 문장 21개. Field 17, Lot 19. 이전 canonical 검증과 같은 비율이다.
- **목표 선택을 묻는 시작이 15개다:** 첫 화면에 관련 물체가 있고 지시한 것은 화면 밖(`past_color` 3, `past_shape` 3, `red_past` 3, `swap` 3, `query` 3). 이전 검증에는 9개였다.
- **"먼저 보임"의 기준은 canonical set과 같다:** 첫 화면 어디든, 60m 이내. 학습 episode는 더 어렵게 만들었다(화면 가운데, 48m 이내).
- **강제 회전은 없다.** 검증과 test는 clean 비행이다. "놓친 뒤"는 pilot의 강제 회전 2회와, 검증 비행 중 저절로 놓친 경우를 기록에서 세어 본다.
- **Teacher는 36/36이다**(착륙 27회 전부 latch·disarm).

## Gate

Canonical gate의 숫자 그대로다([configs/visual_search.json](../configs/visual_search.json)의 `gen_v3c.gates`, test가 확인한다). 판정은 `canonical_evaluator_v2`.

### 결과: smoke 통과, representative 통과 못 함

판정 기록: `outputs/generalization/gen_v3c/gate_smoke.json`, `gate_representative.json`. 표와 episode별 기록: 같은 폴더의 `tables.md`, `summary.json`, `episodes.csv`, `failures/`.

| Smoke (검증 시작 10개) | 값 | 기준 | |
|---|---:|---|---|
| 성공 | 10 | 7 이상 | 통과 |
| 착륙 성공(system) | 7 | 1 이상 | 통과 |
| 32–44m에서의 성공 | 3 | 1 이상 | 통과 |
| 충돌 | 0 | 1 이하 | 통과 |
| 실행 오류 | 0 | 0 | 통과 |

| Representative (검증 시작 36개) | 값 | 기준 | |
|---|---:|---|---|
| 전체 성공률 | 33/36 (0.917) | 0.80 이상 | 통과 |
| 착륙 성공률(system) | 24/27 (0.889) | 0.75 이상 | 통과 |
| 목표가 보인 뒤 접근을 시작함 | 32/33 (0.970) | 0.90 이상 | 통과 |
| 문장이 시킨 대로 끝남 | 34/36 (0.944) | 0.90 이상 | 통과 |
| **다른 물체에서 끝난 비행** | **3** | **2 이하** | **통과 못 함** |
| 충돌 | 0 | 1 이하 | 통과 |
| 실행 오류 | 0 | 0 | 통과 |

- **Gate: FAIL.** 이전과 같은 줄에서, 같은 숫자로 걸렸다. 기준은 고치지 않았다.
- **나머지 줄은 이전보다 여유가 생겼다.** 성공률 0.83 → 0.92, 착륙 0.78 → 0.89. 다만 set도 판정 방식도 다르므로 나란히 놓고 "올랐다"고 읽을 수는 없다.
- **계획대로 여기서 멈췄다.** 동결하지 않았고, canonical test 48개는 비행하지 않았다.

| Band | 전체 | 착륙 | 접근 |
|---|---:|---:|---:|
| near (12–20m) | 11/12 | 8/9 | 3/3 |
| mid (20–32m) | 10/12 | 7/9 | 3/3 |
| far (32–44m) | 12/12 | 9/9 | 3/3 |

### 실패 3회

셋 다 "관련 물체가 첫 화면에 있고 지시한 pad는 화면 밖"인 시작이다. 셋 다 지시한 pad가 한 번도 화면에 들어오지 않았다.

| Episode | 첫 화면 | 일어난 일 |
|---|---|---|
| `v3c-22005-…-near_pair-swapped` | blue cone이 정면(−5°) 15m. blue pad는 오른쪽 93°, 38m | blue cone으로 천천히 전진해 그 앞 공중에서 멈췄다 |
| `v3c-22011-…-mid_past_color` | blue cone이 오른쪽 21°, 23m. blue pad는 뒤쪽 150°, 24m | 오른쪽으로 세 번 돌자 cone이 화면 가운데로 왔고, 그쪽으로 전진해 그 앞에서 멈췄다 |
| `v3c-22012-…-mid_past_shape` | red pad가 오른쪽 32°, 16m. blue pad는 왼쪽 99°, 29m | 오른쪽으로 세 번 돌자 red pad가 가운데로 왔고, 그 위에 내려앉았다. Finalizer는 latch했고 평가는 실패로 셌다 |

## Grounding 진단

### 목표 선택 (새 검증 36회)

| 시작이 가진 것 | 비행 | 성공 | 지시한 물체에서 끝남 | 다른 물체에서 끝남 |
|---|---:|---:|---:|---:|
| 색이 같고 형태가 다른 물체가 곁이나 첫 화면에 | 11 | 9 | 9 | 2 |
| 형태가 같고 색이 다른 물체가 곁이나 첫 화면에 | 7 | 6 | 6 | 1 |
| **관련 물체가 첫 화면에 있고 지시한 것은 화면 밖** | **15** | **12** | **12** | **3** |
| … 그중 같은 색 | 9 | 7 | 7 | 2 |
| … 그중 같은 형태 | 6 | 5 | 5 | 1 |
| 두 물체의 자리를 바꿈 (swap) | 3 | 2 | 2 | 1 |
| 곁의 물체를 지시 (query) | 3 | 3 | 3 | 0 |
| 지시한 물체가 첫 화면 밖 | 21 | 18 | 18 | 3 |
| 지시한 물체가 첫 화면 안 | 15 | 15 | 15 | 0 |
| 가는 길에 지시한 물체를 놓침 (강제 회전 없이) | 1 | 1 | 1 | 0 |

- **"먼저 보이는 비슷한 물체"를 거절한 비율: 12/15.** 같은 색 7/9, 같은 형태 5/6.
- **그 조건 밖은 21/21이다.** 실패는 전부 그 15개 안에 있다.
- **놓친 뒤 되찾기:** pilot의 강제 회전 2/2(시작 checkpoint도 2/2), 검증에서 저절로 놓친 1회도 되찾아 성공. 이번 비행들에서 "놓친 뒤 다른 물체로 갈아탐"은 나오지 않았다.

### 조건은 같고, 비율도 거의 같다

"관련 물체가 첫 화면에 있고 지시한 것은 화면 밖"인 시작만 모으면:

| | 시작 checkpoint | Gen-v3c |
|---|---:|---:|
| 이전 canonical 검증 (legacy evaluator) | 9회 중 2회가 다른 물체로 | — |
| Pilot | 6회 중 2회 (+ 공중 정지 1회) | 6회 중 1회 |
| 새 검증 | — | 15회 중 3회 |
| 합계 | 15회 중 4회 (27%) | 21회 중 4회 (19%) |

- **줄었다고 말하기에는 차이가 작다.** 같은 시작 6개(pilot)에서는 2 → 1이지만, 새 검증의 15개에서는 다섯 번에 한 번꼴 그대로다. 표본이 작아서 27%와 19%를 다르다고 할 근거가 없다.
- **Offline probe와 맞는다.** 같은 색 물체가 화면 가운데에 있는 frame에서 그쪽으로 전진하는 비율이 33% → 21%였다. 줄었지만 남았다.

### 실패가 나는 자리: 관련 물체가 화면에 오래 머무는 시작

두 checkpoint의 그런 시작 36회(위 표)를 관련 물체의 첫 위치로 나누면:

| 첫 화면에서 관련 물체의 위치 | 비행 | 다른 물체에서 끝남 |
|---|---:|---:|
| 오른쪽(+10° 이상): 오른쪽으로 도는 탐색이 그것을 화면 가운데로 끌고 지나간다 | 14 | 6 (+ 공중 정지 1) |
| 정면(±10°)이고 20m 이내 | 4 | 2 |
| 왼쪽이거나, 정면이고 20m보다 멂 | 18 | 0 |

- **탐색은 항상 오른쪽으로 돈다.** 관련 물체가 왼쪽에 있으면 한두 판단 만에 화면 밖으로 나가고, 오른쪽에 있으면 가운데를 지나 다섯 판단쯤 화면에 머문다.
- **한 장만 보는 정책에게는 판단마다 "저것으로 갈까"가 새로 걸린다.** 한 번이라도 전진하면 그 물체가 가운데에 더 크게 보이고, 그대로 굳는다.
- **내 해석:** 화면에 관련 물체가 가운데에 있을 때, 문장의 명사가 행동을 바꾸는 힘이 약하다. Offline probe에서 그런 frame의 문장을 "보이는 물체를 지시하는 문장"으로 바꿔도 전진으로 바뀐 것은 세 번에 한 번이었고, 보정 뒤에도 그대로였다. 데이터를 더한 것은 "전진하지 않을" 확률을 조금 올렸을 뿐, 문장과 화면을 묶는 방식은 바꾸지 못했다.

## Examples

모두 이미 쓴 시작(pilot, 새 검증)을 `-Record`로 다시 비행한 것이다. Test set의 비행이 아니다.

| | 무엇 |
|---|---|
| [dataset_samples.jpg](../outputs/examples/gen_v3c/dataset_samples.jpg) | Hard-negative episode의 첫 화면과 twin. 같은 화면, 다른 문장이나 배치 |
| [related_first_recovered.gif](../outputs/examples/gen_v3c/related_first_recovered.gif) | `blue pad ← blue cone`(pilot). blue cone이 화면 가운데로 오자 그쪽으로 몇 번 전진했다가 돌아서고, blue pad를 찾아 착륙한다 |
| [failure_related_first.gif](../outputs/examples/gen_v3c/failure_related_first.gif) | 새 검증의 실패(`v3c-22011`). 같은 종류의 시작에서 blue cone으로 가서 그 앞에 멈춘다 |
| [lost_reacquired.gif](../outputs/examples/gen_v3c/lost_reacquired.gif) | 강제 회전으로 blue pad를 놓치고 blue cone이 가운데 보이지만, 계속 돌아 blue pad를 되찾고 착륙한다(pilot) |

## Canonical test 48개

**실행하지 않았다.** Gate를 통과하지 못했기 때문이다. Test set은 teacher만 비행한 상태 그대로다(48/48). 이번 결과를 보고 test를 미리 들여다보거나 일부만 비행하지 않았다.

## Finalizer와 착륙 (새 검증)

| 착륙 mission 27회 | 수 |
|---|---:|
| 지시한 pad에 touchdown (구역 안, 부드럽게) | 24 |
| Physical landing (어느 pad든 안정되게 섬) | 25 |
| Stable physical landing (지시한 pad) | 24 |
| System landing success | 24 |
| Strict policy zero-action (보조) | 23 |
| blue pad / red pad | 18/21, 6/6 |

| Finalizer | 수 |
|---|---:|
| 접촉이 있었던 비행 | 25 (지시한 pad 24, 다른 pad 1) |
| Latch / disarm | 25 / 25 |
| `finalizer_rescued_success` (정책은 0 행동을 못 냈지만 system은 착륙) | 1 |
| Approach mission 9회에서 켜진 횟수 | 0 |

- **착륙 쪽은 안정적이다.** 지시한 pad로 간 24회는 전부 touchdown, 안정 착륙, latch, disarm까지 갔다. 실패한 착륙 3회는 모두 pad에 가지 못한 것이다.
- **Evaluator v2와 이전 판정이 이번에는 같았다.** 이 run에서는 "서 있는데 속도가 보고된" 경우가 나오지 않았다(24회 모두 두 판정이 일치). 이전 run의 2회는 기록된 test 입력으로 확인한다.
- **Touchdown:** 오차 중앙값 2.56m, 최대 3.89m(구역 6.6m). 직전 수직 속도 중앙값 0.61m/s, 최대 0.65m/s(기준 0.75m/s).
- **착륙 data는 더 넣지 않는다.** 남은 strict 차이(23 대 24)는 보조 지표로만 둔다.

## 회귀: 보정이 이전 능력을 깎았는가

Gate는 통과하지 못했지만, 다음 단계가 이 checkpoint에서 출발해도 되는지 알기 위해 비행했다. 이전 set의 시작 33개를 Gen-v3c의 어떤 비행보다 먼저 정해 두었고(`scripts/gen_v3c.py regression`), 같은 시작에서 이전 checkpoint들이 한 것과 나란히 놓았다(`outputs/generalization/gen_v3c/regression.md`, `regression.json`). 이 set들은 pad도 착륙도 없는 접근 과제이고, canonical baseline의 일부가 아니다.

| Set | Gen-v2 | Gen-v3 (두 번째) | Gen-v3c |
|---|---:|---:|---:|
| G1 — 처음 보는 시작 (Blocks) | 8/8 | 7/8 | 7/8 |
| G3 — held-out 장면 (Yard) | 8/8 | 8/8 | 8/8 |
| P — 안 쓴 문장 | 8/8 | 7/8 | 6/8 |
| L — 시작 40–90m (canonical 범위 밖) | 6/9 | 7/9 | 7/9 |
| 합계 | 30/33 | 29/33 | 28/33 |

- **탐색, 접근, 정지, 충돌에서는 달라진 것이 없다.** 33회 모두 목표를 찾았고(33/33), 충돌은 0이다. Yard는 8/8 그대로다.
- **잃은 것은 한 비행이고, 이미 알던 자리다.** Blocks의 blue cone 46m 시작을 다른 문장(`Locate …`)으로 시켰을 때다. 그 시작은 두 번째 Gen-v3 checkpoint부터 실패하던 것이고(정면에 보이는 cone을 두고 계속 돈다), 이번에 같은 시작의 세 문장 중 둘에서 셋으로 늘었다.
- **내 해석:** "파란 물체가 보여도 확실하지 않으면 가지 않는다"를 더 가르친 것이, 파란 물체가 하나뿐인 장면의 먼 blue cone에서 망설임을 조금 더 키웠을 수 있다. 한 비행이라 단정하지 않는다.
- **장거리의 실패 둘은 처음 보는 물체(yellow pyramid)의 55m 이상 시작이다.** 세 checkpoint 모두 그 둘 중 하나 이상에서 실패한다. 44m 초과는 여전히 따로 둔다.

## 결론

### 이번 질문의 답: NO

> "먼저 보이는 비슷한 물체"라는 hard negative를 데이터로 분명히 가르치면, 지금의 AeroVLA-OFT 구조가 색과 명사를 묶어 맞는 목표를 안정적으로 고를 수 있는가?

- **이 데이터와 이 보정으로는 아니다.** 새 검증에서 그런 시작 15회 중 3회가 다시 다른 물체에서 끝났고, gate는 같은 줄에서 걸렸다.
- **Pilot은 통과했지만 그것으로 충분하지 않았다.** Pilot의 12회는 방향을 보는 빠른 확인이었고, 판정은 새 검증 36회의 것이다.
- **계획의 원칙대로, 같은 종류의 데이터를 더 넣지 않는다.**

### 44m 이하 clean canonical baseline은 안정적인가: PARTIALLY

- **관련 물체가 먼저 보이지 않는 시작에서는 안정적이다.** 새 검증에서 21/21, 착륙은 지시한 pad로 간 24회가 전부 system landing까지 갔고, 충돌은 0이다.
- **관련 물체가 먼저 보이는 시작에서는 다섯 번에 한 번 틀린다.** 그 조건이 canonical mission의 장면(색과 형태가 겹치는 물체들 사이의 pad)에서 드문 것이 아니다.
- **Gate를 통과하지 못했으므로 baseline으로 동결하지 않았다.**

### 남은 병목은 여전히 language–vision grounding인가: YES

- **새 검증의 실패 3회가 전부 같은 조건이다.** 두 checkpoint의 pilot·검증 비행을 합치면 다른 물체로 간 9회 중 8회가 "관련 물체가 먼저 보임"이고, 나머지 1회는 이전 검증에서 목표를 놓친 뒤였다. 새 검증에서 찾기, 접근, 정렬, 하강, 착륙, 문장에 따른 끝맺음에서는 실패가 나오지 않았다.
- **데이터로 줄일 수 있는 만큼은 줄였다.** 화면 가운데의 같은 색 물체로 전진하는 비율은 33% → 21%(offline)였고, 비행에서의 비율은 구별되지 않을 만큼만 움직였다.
- **다음 후보: FiLM 또는 더 강한 language–vision 결합.** 지금은 문장이 LLM 입력으로만 들어가고, 화면 가운데에 관련 물체가 있을 때 명사가 행동을 바꾸는 힘이 약하다. 문장으로 vision 쪽 feature를 직접 조절하는 구조를 다음 단계로 권한다. 12GB에서는 vision 쪽 학습이 걸려 있어 별도의 설계가 필요하다. **이번 branch에서는 구현하지 않았다.**
- **구조를 바꾸지 않고 볼 수 있는 것도 하나 있다.** 실패는 관련 물체가 화면에 오래 머무는 시작에 몰려 있다. 이것은 "한 장만 보는 정책 + 한쪽으로만 도는 탐색"의 성질이라, FiLM을 넣은 뒤에도 같은 표로 다시 재야 한다.

### Gaussian Blur로 넘어갈 준비: NO

Clean 기준이 gate를 통과하지 못했다. Blur는 실행하지 않았다.

### 44m 초과는 여전히 별개다

회귀 표의 장거리 시작에서 숫자가 어떻게 나왔든, 44m 초과를 canonical baseline에 다시 섞지 않는다. [Long-range same-color grounding](long_range_same_color_grounding.md)에 따로 둔다. 풀렸다고 주장하지 않는다.

## 한계

1. **Held-out test 결과가 없다.** Canonical test 48개는 비행하지 않았다. 이 문서의 모델 숫자는 pilot과 새 검증의 것이다.
2. **표본이 작고, 비행은 반복하면 달라질 수 있다.** 다른 물체로 간 비행 3회와 2회의 차이는 한 비행이다. 같은 시작·같은 checkpoint에서 성공과 충돌이 갈린 경우가 있었다. Gate는 미리 정한 숫자로 판정했고, 그 숫자 근처에서의 우연을 가려낼 만큼 크지 않다. 다음 gate에서는 목표 선택을 묻는 시작을 두 번 이상 비행하는 것을 권한다.
3. **새 검증 set은 이전 것보다 그 조건을 많이 넣었다**(15개 대 9개). 그래서 "다른 물체 3회"가 같은 숫자라도 같은 어려움은 아니다. 비율로 보면 거의 같다.
4. **보정은 한 가지 설정으로 한 번 했다.** Update 수, learning rate, 뽑는 비율을 바꿔 가며 찾지 않았다(결과를 보고 고르지 않기로 했다). 더 길거나 강한 보정이 더 줄였을 가능성은 남아 있다.
5. **Offline probe의 "보이는 물체를 지시하는 문장"은 `Find …` 한 가지 문형이다.**
6. **Evaluator v2의 임계값은 이 simulator의 기록에서 나왔다.** 다른 물리 설정에서는 다시 재야 한다.
7. **장면은 모두 같은 simulator 안에 실행 중에 만든 것이다.** 새 배치는 새 장면이 아니다.

## 재현

```powershell
# 계획: 검증 → pilot → 녹화할 episode (앞의 둘은 파일이 있으면 거부한다)
python scripts/gen_v3c.py plan validation
python scripts/gen_v3c.py plan pilot
python scripts/gen_v3c.py plan train --output datasets/projectairsim_visual_search/generalization_v3c_added
python scripts/gen_v3c.py check

# 녹화(teacher, finalizer 없이: 이전 data와 같은 방식), teacher로 검증·pilot 확인
scripts/run_visual_search.ps1 -Policy teacher -Finalizer off -Plan datasets/projectairsim_visual_search/generalization_v3c_added/plan.json -Set train -Record datasets/projectairsim_visual_search/generalization_v3c_added -Output outputs/visual_search/gen_v3c_collect_train
scripts/run_visual_search.ps1 -Policy teacher -Plan configs/gen_v3c_canonical_validation.json -Set episodes -Output outputs/visual_search/gen_v3c_teacher_validation
```

```bash
# 병합, 검증 frame 섞기, 보정 학습 (WSL)
python scripts/build_visual_search_dataset.py $DATA/generalization_v3c_added
python scripts/build_visual_search_dataset.py $DATA/generalization_v3c --sources $DATA/generalization_v1 $DATA/generalization_v2_added $DATA/generalization_v3_added $DATA/generalization_v3b_added $DATA/generalization_v3c_added
python scripts/gen_v3c.py val-mix --dataset $DATA/generalization_v3c --added generalization_v3c_added
python scripts/train_aerovla_oft.py --dataset $DATA/generalization_v3c --output outputs/aerovla_oft/checkpoints/generalization_v3c --init outputs/aerovla_oft/checkpoints/generalization_v3b \
  --steps 2000 --lr 5e-5 --eval-every 250 --eval-samples 600 --val-file val_mix --strategies right --boost small_visible=2 selection=3 --share generalization_v3c_added=0.5 --select-band 0.02
python scripts/train_aerovla_oft.py --negative-probe outputs/aerovla_oft/checkpoints/generalization_v3c --dataset $DATA/generalization_v3c_added
```

```powershell
# Pilot → smoke → representative. 각 단계는 앞 단계를 통과해야 넘어간다
scripts/run_visual_search.ps1 -Policy oft -Checkpoint outputs/aerovla_oft/checkpoints/generalization_v3c -Plan configs/gen_v3c_pilot.json -Set episodes -Output outputs/visual_search/gen_v3c_pilot
python scripts/gen_v3c.py gate pilot --run outputs/visual_search/gen_v3c_pilot --teacher outputs/visual_search/gen_v3c_teacher_pilot
scripts/run_visual_search.ps1 -Policy oft -Checkpoint outputs/aerovla_oft/checkpoints/generalization_v3c -Plan configs/gen_v3c_canonical_validation.json -Set episodes -Output outputs/visual_search/gen_v3c_gate
python scripts/gen_v3c.py gate representative --run outputs/visual_search/gen_v3c_gate --teacher outputs/visual_search/gen_v3c_teacher_validation
```

Checkpoint, dataset, 비행 기록(`outputs/aerovla_oft/`, `datasets/`, `outputs/visual_search/`)은 Git에 없다.

## Tests

`tests/test_canonical.py`의 `EvaluatorV2Tests`:

- 실제로 안정되게 선 착륙 + 흔들리는 보고 속도 → 착륙으로 읽는다(기록된 두 비행 포함). 이전 판정은 아니라고 읽고, 그 값도 함께 남는다.
- Pad 위지만 위치가 계속 움직임 → 착륙이 아니다. 한 번 닿고 다시 뜬 것도 아니다.
- 다른 pad에 안정되게 섬 → physical landing은 참, 과제는 실패.
- 공중 정지 → 아무리 가만히 있어도 착륙이 아니다.
- Approach mission → finalizer가 켜지지 않고, 착륙 판정도 없다.
- 안정 접촉 + latch + disarm → system landing success. Latch나 disarm이 빠지면 아니다.
- Finalizer의 숫자는 이전 그대로다.

`tests/test_gen_v3c.py`:

- 학습·pilot·검증·test·기존 학습의 배치가 서로 겹치지 않는다. 학습 배치는 test 배치와 이웃 쌍을 공유하지 않고, 검증 배치는 어느 것과도 공유하지 않는다.
- 녹화 계획이 생성기로 그대로 재현되고, 종류별 수가 설정과 같고, 어떤 평가 set의 시작에서도 6m 이상 떨어져 있다.
- `*_first`는 관련 물체가 화면 가운데에 있고 지시한 물체는 화면에 전혀 없다. `lost`는 접근을 시작한 뒤 강제 회전으로 관련 물체가 가운데에 온다. Twin은 자세가 같고 하나만 다르다.
- 색·형태·거리·끝·장면이 한쪽으로 쏠리지 않았다.
- 새 검증 set과 pilot이 생성기로 재현되고, 구성이 문서와 같다. Smoke와 회귀 시작이 고정돼 있다.
- Gate의 숫자가 canonical gate와 같다. 보정 학습의 설정이 config에 있다.
- 뽑는 비율, 더 자주 뽑는 frame, 검증 frame 섞기, 기록에서 "놓침"을 찾는 것, 자리 바꾼 twin의 병합.

