# 개입 data 학습 1회차 (Takeover Round 1) — 계획 초안

**상태: 초안.** 사용자가 확인하면 평가 시작 위치를 뽑고 gate 파일을 봉인한 뒤, 결정까지 숫자를 바꾸지 않습니다. 설정 파일: [`configs/human_takeover_r1.json`](../configs/human_takeover_r1.json).

## 무엇을 확인하는가

개입 모드에서 사람이 고친 조작으로 학습하면, 모델이 다음 세 가지를 **스스로** 하게 되는지 봅니다. 세 가지 모두 사람이 고친 바로 그 장면입니다.

1. **정밀 착륙:** 가운데에서 1.75m 안(공 위 판은 1.3m). 지금 모델이 혼자 내린 pad 착륙 6번은 1.77–3.46m로 모두 기준 밖이었습니다.
2. **pad가 아닌 표면에 착륙:** cube, cylinder, 공 위 판. 지금 모델은 공중에서 멈추거나 가장자리에 닿습니다.
3. **44m 밖에서 찾아가기:** 평가 범위 밖. 출발할 때 목표가 안 보이는 경우가 많습니다.

동시에, 원래 잘 하던 12–44m pad 착륙은 나빠지지 않아야 합니다.

## 시작 checkpoint: 동결 baseline `grounding_film`

- 개입 비행을 이 모델로 날렸으므로, 사람이 고친 장면은 모두 이 모델이 실제로 간 상태입니다. 그래서 이 모델에서 이어서 학습합니다(DAgger 1회차).
- generic landing 단계(1b–1d)의 checkpoint는 쓰지 않습니다. gate를 넘지 못했고, 다른 teacher의 data로 학습됐습니다.
- 이번 회차의 data는 **기본 checkpoint로 날린 개입 비행만** 씁니다. `-Checkpoint`로 다른 모델을 날린 비행은 쓰지 않습니다.

## Data: 학습을 시작하기 전에 모을 것 (data gate)

| 기준 | 최소 | 지금 |
|---|---:|---:|
| 기록된 개입 비행 | 100 | 14 |
| 착륙: pad / cube / cylinder / 공 위 판 | 각 12 | 1 / 1 / 7 / 0 |
| 접근(APPROACH) | 15 | 5 |
| 44m 밖에서 출발 (mid, far) | 30 | 6 |
| 장면별: field / lot | 각 25 | 14 / 0 |

- **진행 상황 보기:** `assets\projectairsim-env\Scripts\python.exe scripts\human_data_status.py`
- **고르게 모으기:** 무작위 모드(R, `-Random`)는 비행 종류 다섯 가지(pad·cube·cylinder·판 착륙, 접근)를 같은 확률로 뽑습니다. 물체부터 뽑으면 판 착륙은 30번에 한 번꼴이라 이렇게 바꿨습니다. 장면(M)과 거리 단계(D)는 직접 바꿔 가며 채웁니다.
- **사람 조종만 한 비행**(`human_pilot`)도 따로 세어 같이 학습에 넣습니다. gate 기준에는 넣지 않습니다.
- **시작 조건:** 기준을 모두 채우고 사용자가 시작하라고 해야 학습이 시작됩니다. 자동으로 시작하지 않습니다.

## 학습 data 구성

| | 출처 | 뽑히는 비율 |
|---|---|---:|
| 사람 | 개입 비행 (`takeover`) | 40% |
| 사람 | 사람 조종 (`human_pilot`) | 5% |
| 재생(replay) | `grounding_film`이 학습한 data (generalization v1, v2·v3·v3b·v3c added: 1,160 비행, 57,329 샘플) | 55% |

- **학습에서 빠지는 decision:** 모델이 날린 decision(`model`), 넘겨받은 순간(`handover`), 손을 뗀 순간(`pause`, `below_safe`), 후진(`backward`), 모델보다 빠름(`over_bound`)은 기록 그대로 빠집니다.
- **generic landing teacher의 data는 넣지 않습니다.** 바뀌는 것이 사람 data 하나뿐이어야 결과를 해석할 수 있습니다.
- **validation:** 비행 단위로 20%를 떼어 둡니다(builder의 seed 분할).
- **기록:** 학습을 시작할 때 쓴 사람 비행의 id와 sha256을 `outputs/human_r1/data_manifest.json`에 남깁니다.

## 학습 설정

Stage 1b가 같은 checkpoint에서 학습한 방식과 같습니다. LoRA, action head, FiLM을 학습하고, 아무것도 더하지 않습니다.

| | 값 |
|---|---|
| update | 2,000 (batch 1 × accumulation 4) |
| learning rate | 5e-5 (FiLM도 5e-5) |
| 후보 | update 500 / 1000 / 1500 / 2000 |
| validation | 250 update마다, 600 샘플 |

## 오프라인 측정 (판정에 쓰지 않음)

사람 data의 validation 비행에서, `grounding_film`과 각 후보의 첫 action이 사람 label과 얼마나 다른지 잽니다(`train_aerovla_oft.py --breakdown --source takeover`). 목표에서 먼 구간, 가까운 구간, 표면 위 구간으로 나눠 봅니다. 학습이 사람의 조작 쪽으로 움직였는지 비행 전에 보는 숫자입니다.

## 평가

- **장소:** check layout `t u p q r s`. 사람 조종은 이 layout을 거절하므로, 사람이 본 적 없는 배치입니다. 판 착륙은 이 layout들의 판 변형(`t_cap` …)에서 납니다.
- **새 시작 위치:** seed 범위 90000–92999에서 뽑습니다. 저장소의 모든 check 시작, 모든 사람 비행의 시작, 서로에서 6m 이상 떨어뜨립니다.
- **착륙 판정:** surface evaluator의 착륙(이름의 표면, 착륙 영역 안, soft, 멈춰 섬). 정밀 착륙은 그중 가운데 기준 안인 것입니다.
- **결과가 갈리는 시작:** 같은 시작도 결과가 바뀔 수 있습니다. 그래서 1–2 비행 차이로 기준이 갈리면 그 시작을 두 모델 모두 한 번 더 날리고 둘 다 셉니다.

### Pilot (후보 고르기, 새 시작 32개)

`grounding_film`이 먼저 날고, 후보 넷이 한 번씩 납니다.

| | near | mid |
|---|---:|---:|
| 착륙: pad, cube, cylinder, 판 | 각 3 | 각 3 |
| 접근 | 4 | 4 |

후보의 통과 기준은 `grounding_film`과 비교합니다.

| 항목 | 기준 |
|---|---|
| 착륙 | +4 이상 (24개 중) |
| 정밀 착륙 | +4 이상 |
| 공중 정지 | 늘지 않음 |
| 충돌 | 늘지 않음 |
| 다른 물체에 착륙 | 늘지 않음 |
| 접근 성공 | 1개 넘게 줄지 않음 |

기준을 넘은 후보 중에서 정밀 착륙 → 착륙 → 공중 정지 → 사람 validation L1 → 이른 update 순으로 하나를 고릅니다. 넘은 후보가 없으면 gate는 날리지 않고 1회차는 통과하지 못합니다.

### Gate (봉인, 새 시작 48개)

계획을 고정할 때 뽑아 봉인합니다. 학습, 후보 고르기, 디버깅에 쓰지 않습니다. 고른 후보와 `grounding_film`이 한 번씩 납니다.

| | near | mid | far |
|---|---:|---:|---:|
| 착륙: pad, cube, cylinder, 판 | 각 3 | 각 3 | 각 2 |
| 접근 | 6 | 6 | 4 |

통과 기준은 다음과 같습니다(`grounding_film` 대비).

| 항목 | 기준 |
|---|---|
| 착륙 | +6 이상 (32개 중) |
| 정밀 착륙 | +6 이상 |
| 공중 정지 | 늘지 않음 |
| 충돌 | 늘지 않음 |
| 다른 물체에 착륙 | 늘지 않음 |
| 접근 성공 | 2개 넘게 줄지 않음 |

### Regression (원래 하던 것)

12–44m pad 착륙 새 시작 24개입니다. 착륙이 `grounding_film`보다 1개 넘게 줄지 않고, 충돌 0, 다른 물체 착륙 0이어야 합니다.

## 결정

1회차는 다음이 모두 맞을 때만 통과합니다.

1. data gate를 채웠다.
2. 후보 하나가 pilot을 통과했다.
3. 그 후보가 gate와 regression을 통과했다.

통과하면 `outputs/aerovla_oft/checkpoints/human_r1`로 저장하고, 2회차 개입 비행은 이 모델로 날립니다. 통과하지 못하면 data는 그대로 두고, 이유를 보고하고, 다음 단계는 사용자가 정합니다.

## 계획을 고정한 뒤 만들 것

- **평가 시작 생성:** check layout과 그 판 변형에서 pilot, gate, regression 시작을 뽑고 gate를 봉인하는 도구.
- **판 layout 평가 실행기:** 판 layout에서 checkpoint를 날리는 실행기. canonical loop(`scripts/visual_search.py`)는 동결 파일이라 고치지 않고, 사람 조종처럼 판 layout을 넣은 map을 environment에 건넵니다.
- **dataset 조립과 학습 명령:** `scripts/build_visual_search_dataset.py --sources …` 후 `scripts/train_aerovla_oft.py --init … --share takeover=0.4 human_pilot=0.05 …`.
- **판정 요약:** pilot, gate, regression의 표와 통과 여부.

## 하지 않는 것

- 동결 baseline의 파일, canonical 평가와 그 map, 이미 쓴 44m canonical test는 바꾸거나 다시 쓰지 않습니다.
- Stage 1c의 봉인 gate는 열지 않습니다.
- 학습은 data gate를 채우고 사용자가 시작하라고 할 때만 합니다.

## 위험과 한계

- **사람 label과 teacher label의 충돌:** 같은 장면에서 둘이 다르게 행동하면, 섞어 학습할 때 평균이 되어 머뭇거릴 수 있습니다. pad 착륙에서 특히 그렇습니다(재생 data에 teacher의 pad 착륙이 많음). Regression과 공중 정지 수로 봅니다.
- **사람의 기억과 바깥 시점:** 사람은 아까 본 것을 기억하고, 대부분 Unreal 창(바깥 시점)을 보며 날렸습니다(지금 14개 모두 `view: unreal`). 모델은 지금 사진 한 쌍만 봅니다. 사진만으로 설명되지 않는 조작은 배우기 어렵습니다.
- **분량:** 100 비행이면 사람 샘플이 약 5,000개로, `grounding_film` 학습 data의 9% 정도입니다. 효과가 작으면 2회차 data를 더해 다시 봅니다.
- **far 구간:** 평가 gate에는 있지만 사람 data는 적을 수 있습니다. far는 학습이 아닌 측정 위주로 봅니다.
