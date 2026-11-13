# AeroVLA-OFT Gen-v2 — 실패 분석과 전환 데이터 보강

2026-10-08 실행. 브랜치 `exp/aerovla-oft-generalization`. [Gen-v1](aerovla_oft_generalization.md)의 checkpoint, dataset, 평가 결과는 그대로 두고 그 위에 추가했다.

**요약**

- **Gen-v1의 실패는 목표를 못 찾아서가 아니었다.** 처음에 안 보이던 목표 73개를 모두 찾았다. 본 물체의 실패 9회 중 8회는 찾은 뒤에, 높은 곳에서 일찍 멈추거나 목표 앞에서 계속 올라가서 생겼다.
- **그 부분의 데이터만 더했다.** 구조, 학습 설정, teacher 규칙, held-out set은 그대로 두고 "찾은 뒤의 행동"을 담은 153 episode를 Gen-v1의 292 episode에 더해 처음부터 다시 학습했다.
- **Gen-v2는 처음 보는 시작에서 32/32다**(Gen-v1 27/32). 이른 정지와 계속 상승이 8회에서 0회가 됐고, 학습에 없던 문장도 21/24에서 24/24가 됐다. 충돌은 계속 0이다.
- **처음 보는 물체도 같이 올랐다**(G2 9 → 11/12, G4 2 → 4/6). 그 물체는 여전히 학습에 없다. 실패의 상당수가 grounding이 아니라 찾은 뒤의 머뭇거림이었다는 뜻이다.
- **Yard는 그대로다**(19/20 → 18/20). 남은 실패 5회는 모두 44m 이상에서 시작해, 목표를 보고도 다가가지 않거나 멈추지 않은 경우다.
- **이 개선은 같은 held-out set을 보고 고친 결과이고, 학습량도 함께 늘었다.** 새 held-out 추정치로 읽으면 안 된다.
- **다음 단계 하나:** 새 seed의 held-out 시작 상태에서 Visual Search + Gaussian Blur robustness 평가.

## 무엇을 묻는가

Gen-v1에서 남은 실패가 (1) 목표를 못 찾아서인지, (2) 찾은 뒤 전환·접근·정지를 제대로 못 해서인지, (3) 새 물체의 언어 grounding이 약해서인지를 평가 기록으로 나누고, 가장 큰 병목 하나만 고쳐서 같은 held-out set으로 다시 본다.

## Gen-v1 결과 동결

`scripts/freeze_generalization_results.py`가 모든 실행 폴더를 읽어 [outputs/generalization/gen_v1_final/](../outputs/generalization/gen_v1_final/)에 표를 만든다. 원래 실행 폴더는 읽기만 한다.

| 파일 | 내용 |
|---|---|
| `summary.json` | 아래 숫자 전부와 checkpoint의 학습 기록 |
| `generalization_levels.csv` | 모델 × level: 성공과 그 앞 단계 |
| `failure_taxonomy.csv` | 실패마다 분류 하나와 대표 episode |
| `motion_comparison.csv` | 판단 주기, 추론 시간, 행동·heading 변화, clip, 충돌 |
| `episodes.csv` | episode별 한 줄 |
| `representative_failures/` | 분류별 대표 episode의 기록과 그림(지도 위 궤적, 거리·높이 변화) |

학습 정보는 checkpoint 기록과 맞는지 확인했다: dataset 292 episode / 11,745 sample, best 1,250 update, 2,250에서 조기 종료, train L1 0.115, val L1 0.092, peak 9.80GiB.

### Generalization matrix

| Model | G0 | G1 | G2 | G3 | G4 |
|---|---:|---:|---:|---:|---:|
| AeroVLA Step | 3/18 ¹ | 1/16 | - | - | - |
| AeroVLA Continuous | - | 0/16 | - | - | - |
| OFT Pilot | 23/24 ¹ | 4/32 | 0/6 | 6/10 ² | - |
| OFT Gen-v1 | 20/20 | 27/32 | 9/12 | 19/20 | 2/6 |
| Teacher (목표 위치를 앎) | - | 32/32 | 12/12 | 20/20 | 6/6 |

¹ pilot 단계의 결과(자기 학습 구역)다. ² pilot이 아는 물체(cone, ball)만.

- **비교군은 줄여서 돌렸다.** 기존 AeroVLA는 회당 2–3분이 걸려 G1의 16회(visible, peripheral, search, altitude 각 4)만 비행했다. pilot은 held-out episode 대부분에서 시간 초과로 끝나, G1 전부와 자기 장면(G1의 cone·ball 16회를 벽과 추가 물체가 없는 layout과 pilot의 문장으로) 5/16, G3의 cone·ball 10회, G2 6회만 비행했다. 표의 `-`는 돌리지 않은 것이다.
- **같은 episode끼리:** G1에서 기존 AeroVLA가 비행한 16회에 대해 Gen-v1은 12/16, cone·ball 16회에 대해 pilot 4/16과 Gen-v1 13/16, G3의 cone·ball 10회에 대해 pilot 6/10과 Gen-v1 9/10이다.

| Gen-v1 | 결과 | |
|---|---:|---|
| P 문장 변형 | 21/24 | `Locate` 7/8, `Search for` 8/8, `Find and approach` 6/8. 같은 시작의 원래 문장은 8/8 |
| S 좌우 시험 | 16/16 | 첫 회전은 16회 모두 오른쪽. 획득까지 오른쪽 8, 오른쪽 뒤 14, 왼쪽 뒤 26, 왼쪽 32 판단 |

### 단계별로 나누기

성공 기준(스스로 정지 + 15m 이내 + 충돌 없음)은 그대로 두고, 그 앞 단계를 따로 센다.

- **Acquisition:** 처음에 안 보이던 목표가 Front 또는 Down 시야에 들어왔는가
- **Approach:** 목표 20m 안까지 갔는가
- **Stop:** 정책이 스스로 멈췄는가

| Gen-v1 | Acquisition | Approach | Stop | 성공 |
|---|---:|---:|---:|---:|
| G0 | 8/8 | 20/20 | 20/20 | 20/20 |
| G1 | 16/16 | 32/32 | 31/32 | 27/32 |
| G2 | 6/6 | 11/12 | 9/12 | 9/12 |
| G3 | 12/12 | 19/20 | 19/20 | 19/20 |
| G4 | 3/3 | 4/6 | 3/6 | 2/6 |
| P | 12/12 | 24/24 | 23/24 | 21/24 |
| S | 16/16 | 16/16 | 16/16 | 16/16 |
| 합계 | 73/73 | 126/130 | 121/130 | 114/130 |

### Failure taxonomy

실패한 episode마다 기록에서 다음 순서로 하나를 정한다: 충돌 → 한 번도 못 봄 → (P에서) 같은 시작이 원래 문장으로는 성공 → 다른 물체로 감 → 본 뒤 5m 이상 더 상승 → 15m 안에 들어갔는데 안 멈춤 → 20m 안에서 멈춤 → 그 밖.

| 분류 | G1 | G3 | P | G2 | G4 | 합계 | 대표 episode |
|---|---:|---:|---:|---:|---:|---:|---|
| F1 Search: 목표를 한 번도 못 봄 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| F2 Grounding: 다른 물체로 감 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| F3 Approach: 봤지만 가까이 못 감 | 0 | 1 | 0 | 1 | 2 | 4 | `g3-0137-orange_ball-search` |
| F4 Early stop: 15m 밖에서 정지 | 3 | 0 | 0 | 0 | 1 | 4 | `g1-0024-blue_cone-altitude` |
| F5 No stop: 근처에서 안 멈춤 | 0 | 0 | 0 | 0 | 1 | 1 | `g4-0162-yellow_pyramid-peripheral` |
| F6 Altitude transition: 찾은 뒤 계속 상승 | 2 | 0 | 0 | 2 | 0 | 4 | `g1-0008-blue_cone-peripheral` |
| F7 Collision | 0 | 0 | 0 | 0 | 0 | 0 | - |
| F8 Language: 문장을 바꿨을 때만 실패 | 0 | 0 | 3 | 0 | 0 | 3 | `p-0000-blue_cone-visible-find_and_approach` |
| F9 Domain: Yard에서만 반복 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| F10 Other | 0 | 0 | 0 | 0 | 0 | 0 | - |
| 실패 합계 | 5 | 1 | 3 | 3 | 4 | 16 | |

- **F8의 3회도 모습은 F4/F6과 같다.** 먼 거리에서 보이는 목표로 가다가 7–12m 올라간 뒤 15.1–16.7m에서 멈추거나 시간 초과됐다. 문장이 바뀌자 같은 약점이 더 쉽게 드러난 것이다.
- **F9는 0이다.** Yard(G3)의 실패는 1회뿐이고 같은 모습이 반복되지 않았다. G4의 실패 4회는 Yard라서가 아니라 처음 보는 물체라서 생긴 것으로 본다(Blocks의 G2에서도 3회 실패).

### 여섯 질문

| 질문 | 답 | 근거 |
|---|---|---|
| Q1. G1은 충분히 개선됐는가 | 그렇다 | pilot 4/32 → Gen-v1 27/32(같은 cone·ball 16회로는 4/16 → 13/16), 충돌 3 → 0. 학습에서 비워 둔 방향에서도 15/16 |
| Q2. held-out 장면(G3)에서도 탐색이 유지됐는가 | 그렇다 | 처음에 안 보인 12회 모두 찾았고 19/20 성공. 다른 물체가 먼저 보인 7회에도 지시한 물체로 갔다 |
| Q3. G2/G4에서 처음 보는 물체의 grounding이 되는가 | 부분적으로만 | 문장은 따른다(같은 자리에서 "red cube"라고 하면 pyramid로 0/12, "yellow pyramid"라고 하면 9/12). 그러나 본 물체보다 두 배 걸리고, Yard에서는 2/6이다. 실패 7회 중 3회는 보이는데도 다가가지 못했다 |
| Q4. 병목은 search인가 stop/transition인가 | stop/transition | acquisition 73/73. 본 물체(G1, G3, P)의 실패 9회 중 8회가 상승 뒤의 이른 정지이거나 찾은 뒤의 계속 상승이다 |
| Q5. 오른쪽만 도는 탐색이 성공률을 제한하는가 | 아니다. 시간만 든다 | S 16/16, 탐색 시작 episode 전부 획득. 목표가 왼쪽이면 획득까지 32 판단, 오른쪽이면 8 판단 |
| Q6. 먼 거리·고도 상황에서 "한 바퀴 돌고 상승"이 필요한가 | 지금 결과로는 필요하지 않다 | 먼 탐색(44–70m)과 고도 episode에서 획득 실패가 0이다. 고도 episode는 "앞이 막히면 상승" 규칙으로 8/8 찾았다 |

## 병목 판단 — CASE A (찾은 뒤의 전환과 정지)

- **찾기는 병목이 아니다.** 처음에 안 보이던 목표 73개를 모두 시야에 넣었다. scan progress 같은 상태를 추가할 근거(CASE B)가 없다.
- **처음 보는 물체(CASE C)는 두 번째 문제다.** G2 9/12, G4 2/6으로 약하지만, 이번에는 고치지 않는다. 한 번에 하나만 바꿔야 원인을 가릴 수 있다. yellow pyramid는 Gen-v2 학습에도 넣지 않았다.
- **본 물체에서 남은 실패는 두 가지 모습이다.**
  1. **높은 곳에서의 이른 정지.** 벽을 넘으려고 올라간 뒤(높이 11–20m) 목표로 가다가 15.0–16.8m에서 멈춘다. teacher는 12m에서 멈춘다. 본 시작(G0)에서도 고도 episode의 정지 거리는 11.6–14.6m로 길다. 높은 곳에서는 목표가 화면 아래쪽에 크게 보여, 낮은 곳에서 가까이 갔을 때와 비슷해 보이는 것으로 추정한다.
  2. **가까운 큰 물체 앞에서의 상승.** 목표가 가까이(11–19m) 있고 화면 가장자리에 걸리면, 멈추거나 그쪽으로 도는 대신 올라간다. 학습 데이터에서 "가까운 것이 화면을 채움 → 상승"은 벽과 건물에서만 나왔고, 가까운 목표가 화면을 채운 장면은 적었다.

## Gen-v2에서 바꾼 것

**dataset만 바꿨다.** Gen-v1의 292 episode를 그대로 두고, 찾은 뒤의 행동을 담은 episode를 더했다.

| | Gen-v1 | 추가 | Gen-v2 |
|---|---:|---:|---:|
| Episode (train / val) | 292 (260 / 32) | 153 (135 / 18) | 445 (395 / 50) |
| Sample (train / val) | 11,745 (10,495 / 1,250) | 5,020 (4,480 / 540) | 16,765 (14,975 / 1,790) |
| 정지 sample | 1,168 | 612 | 1,780 |
| 그중 높이 9m 이상 | 176 | 364 | 540 |
| 상승 sample | 530 | 508 | 1,038 |

추가한 episode의 종류(`configs/visual_search.json`의 `generalization.targeted_v2`). 물체 4개에 고르게, 좌우 반반이다.

| 종류 | 수 (train / val) | 시작 | 가르치려는 전환 | 겨냥한 Gen-v1 실패 |
|---|---:|---|---|---|
| `climb_to_view` | 39 (35 / 4) | 벽 뒤 32–69m, 높이 5–9m | 상승 → 목표가 보임 → 상승을 멈춤 → 정렬 → 접근 → 정지 (정지 높이 12–15m) | 올라간 뒤 15–17m에서의 이른 정지 |
| `high_approach` | 32 (28 / 4) | 목표가 보임, 14–66m, **높이 9–14m** | 높은 곳에서의 접근 → 정지 | 같은 실패. 벽 없이 높이만 다르게 |
| `stop_zone` | 40 (36 / 4) | 목표가 보임, **10–16m**, 높이 5–14m | 접근 → 정지의 경계. 12m 안이면 바로 정지, 밖이면 조금 더 가서 정지 | 이른 정지, 목표 앞에서 멈추지 않음 |
| `near_edge` | 28 (24 / 4) | 목표가 12–24m, 화면 가장자리(좌우 25–40°) | 정렬 → 접근 | 가까운 왼쪽 목표로 돌지 않고 한 바퀴 돎, 목표 앞 상승 |
| `near_hidden` | 14 (12 / 2) | 목표가 13–21m, 시야 밖이거나 일부만 보임(45–120°) | 탐색 → 목표가 보임 | 가까운 큰 물체 앞에서의 상승 |

- **목표를 찾는 데이터는 늘리지 않았다.** 추가분의 대부분은 목표가 이미 보이거나 곧 보이는 상황이다. 추가한 5,020 sample 중 탐색은 579개(12%)다.
- **정지 label 규칙은 그대로다.** teacher는 여전히 "보이고 12m 이내면 정지"다. 거리는 label을 만드는 데만 쓰고, 모델은 영상에서 판단해야 한다. 그래서 정지 장면이 여러 높이와 방향에서 나오도록 시작 상태를 넓혔다(정지 높이 5–15m).
- **계획 → 수집 → 병합의 검사.** 추가 episode도 held-out 시작점에서 6m 이상, 비워 둔 방향 밖에 있다(`plan_generalization.py check`). seed는 train 2000–2135, val 5100–5117로 Gen-v1(1000–1259, 5000–5031)과 겹치지 않는다. 병합할 때 같은 episode, seed, 시작점, 궤적 폴더가 두 번 나오면 멈추게 했고 중복은 0이다. 영상은 복사하지 않고 sample이 원래 폴더를 가리킨다.
- **수집:** simulator 2개로 48분. 154 episode 중 153개를 썼다. 1개(red cube를 70m 밖 벽 뒤에서 찾는 경우)는 teacher가 240 tick 안에 도착하지 못해 버렸다.
- **이 보강은 held-out 결과를 보고 정한 것이다.** G1–G3의 실패 모습을 보고 종류를 골랐으므로, 같은 set에서 Gen-v2가 좋아지는 것은 새 held-out 추정치가 아니라 개발 결과다. 학습 시작점은 여전히 held-out 시작점과 떨어져 있다.

**바꾸지 않은 것:** 구조(NF4, 고정된 AeroVLA adapter, LoRA r16, L1 head, chunk 4, 실행 1, proprio OFF, FiLM OFF, linear head), optimizer 설정, 검증 방식과 checkpoint 선택 규칙, teacher의 규칙(정지 12m 포함), 실행기, held-out 시작 상태 파일, 성공 기준.

## Training

Gen-v1과 같은 명령, 같은 규칙으로 처음부터 다시 학습했다(Gen-v1 checkpoint에서 이어 학습하지 않았다).

| | Gen-v1 | Gen-v2 |
|---|---:|---:|
| 학습 sample | 10,495 | 14,975 |
| 계획 / 실제 update | 4,000 / 2,250 (조기 종료) | 4,000 / 4,000 |
| **선택된 checkpoint (검증 최저)** | 1,250 update | **3,999 update** (마지막) |
| 본 sample 수 (epoch) | 5,004 (0.48) | 16,000 (1.07) |
| Train L1 (이동 평균) | 0.115 | 0.049 |
| Val L1 (자기 검증 320 sample) | 0.092 | 0.070 |
| Peak VRAM | 9.80GiB | 9.80GiB |
| 시간 | 108분 | 140분 |
| 재로드 차이 | 0.0 | 0.0 |

검증 L1은 250 update마다 0.235, 0.168, 0.133, 0.126, 0.134, 0.104, 0.094, 0.087, 0.094, 0.082, 0.081, 0.084, 0.081, 0.074, 0.072, 0.070이었다.

두 checkpoint를 서로의 검증 frame으로도 쟀다(`train_aerovla_oft.py --score`). 검증 set이 달라서 생기는 차이를 빼고 보기 위해서다.

| Checkpoint | Gen-v1 검증 frame | Gen-v2 검증 frame |
|---|---:|---:|
| Gen-v1 | 0.092 | 0.097 |
| Gen-v2 | **0.067** | **0.070** |

- **Gen-v2는 Gen-v1이 원래 하던 일도 더 잘 맞춘다.** Gen-v1의 검증 frame에서 0.092 → 0.067이다. 상태별로는 stop 0.031 → 0.005, align 0.041 → 0.006, approach 0.099 → 0.064이고, search는 0.065 → 0.081로 조금 나빠졌다.
- **데이터와 학습량이 함께 달라졌다.** 같은 규칙(검증 최저, 4번 연속 개선이 없으면 종료)을 썼지만 Gen-v1은 1,250 update에서, Gen-v2는 3,999 update에서 골라졌다. Gen-v1의 검증 L1은 0.09–0.15 사이에서 흔들리다 멈췄고, Gen-v2는 끝까지 내려갔다. 아래 결과의 차이를 "보강 데이터만의 효과"로 읽을 수는 없다.

## 결과

Gen-v1과 정확히 같은 held-out 파일, 같은 실행기, 같은 성공 기준이다. 표는 [outputs/generalization/gen_v2_final/](../outputs/generalization/gen_v2_final/)에 있다.

### Gen-v1 vs Gen-v2

| Level | Gen-v1 | Gen-v2 |
|---|---:|---:|
| G0 본 시작 | 20/20 | 20/20 |
| G1 처음 보는 시작 | 27/32 | **32/32** |
| G2 처음 보는 물체 | 9/12 | 11/12 |
| G3 학습하지 않은 장면 (Yard) | 19/20 | 18/20 |
| G4 Yard + 처음 보는 물체 | 2/6 | 4/6 |
| P 문장 변형 | 21/24 | **24/24** |
| S 좌우 시험 | 16/16 | 16/16 |
| 합계 | 114/130 | 125/130 |

| 단계 (130회) | Gen-v1 | Gen-v2 |
|---|---:|---:|
| Acquisition (처음에 안 보인 73회) | 73/73 | 73/73 |
| Approach (20m 안) | 126/130 | 126/130 |
| Stop (스스로 정지) | 121/130 | 126/130 |
| 성공 | 114/130 | 125/130 |
| 충돌 | 0 | 0 |

![같은 G1 시작 상태에서 Gen-v1과 Gen-v2의 실제 궤적](../outputs/examples/generalization/flown_g1_v1_v2.jpg)

위는 Gen-v1, 아래는 Gen-v2다. 네모(실패)가 Gen-v2에서는 없다.

### Failure taxonomy의 변화

| 분류 | Gen-v1 | Gen-v2 |
|---|---:|---:|
| F1 Search | 0 | 0 |
| F2 Grounding | 0 | 0 |
| F3 Approach: 봤지만 가까이 못 감 | 4 | 4 |
| **F4 Early stop** | 4 | **0** |
| F5 No stop | 1 | 1 |
| **F6 Altitude transition: 찾은 뒤 계속 상승** | 4 | **0** |
| F7 Collision | 0 | 0 |
| **F8 Language: 문장을 바꿨을 때만 실패** | 3 | **0** |
| F9 Domain | 0 | 0 |
| F10 Other | 0 | 0 |
| 합계 | 16 | 5 |

- **겨냥한 두 가지가 없어졌다.** 이른 정지 4 → 0, 찾은 뒤의 계속 상승 4 → 0이다.
- **문장 변형의 실패 3회도 같이 없어졌다.** 그 3회는 모습이 F4/F6과 같았다(위). 문장 데이터는 추가하지 않았다.
- **F3(봤지만 가까이 못 감)은 4회 그대로다.** 이번에 손대지 않은 부분이다.

### 겨냥한 실패가 실제로 어떻게 바뀌었나

| Gen-v1의 약점 | Gen-v1 | Gen-v2 |
|---|---|---|
| 벽 넘어 올라간 뒤의 정지 거리 (G1 고도 4회) | 15.0, 16.0, 16.3, 16.8m → 0/4 | 10.3, 11.4, 11.4, 12.3m → 4/4 |
| 같은 것, 본 시작 (G0 고도 4회) | 11.6–14.6m | 10.0–12.4m |
| 성공했을 때 가장 먼 정지 거리 (G1) | 14.7m | 12.5m |
| 가까운 목표가 왼쪽 가장자리에 있을 때 (G1의 4회) | 240(실패), 91, 64, 57 판단 | 15, 18, 23, 16 판단 (teacher 15, 17, 22, 15) |
| 목표 앞에서의 불필요한 상승 | G1 2회, G2 2회 | 0회 |
| 학습에 없던 방향의 먼 visible 시작 (G1의 3회) | 97, 170, 209 판단 | 64, 67, 46 판단 |

[같은 시작에서 Gen-v1은 목표 앞에서 계속 올라가고](../outputs/examples/generalization/failure_g1_v1.gif), [Gen-v2는 15 판단 만에 멈춘다](../outputs/examples/generalization/fixed_g1_v2.gif). [벽을 넘어 올라가 찾고 10.5m에서 멈추는 Gen-v2의 비행](../outputs/examples/generalization/climb_g1_v2.gif)도 있다.

### 소요 시간

| episode 평균 판단 수 | Teacher | Gen-v1 | Gen-v2 |
|---|---:|---:|---:|
| G1 | 43 | 77 | 51 |
| G2 (yellow pyramid) | 38 | 148 | 70 |
| G3 (Yard) | 34 | 54 | 62 |
| P | - | 91 | 52 |

찾은 뒤 머뭇거리며 한 바퀴 더 도는 일이 줄어서다. G1에서 목표가 시야 밖인 판단이 episode당 35 → 13으로 줄었다. 탐색 자체(처음 시야에 넣기까지 20.6 → 21.4 판단)는 그대로다.

### 남은 실패 5회

| Episode | 무슨 일이 있었나 | 분류 | Gen-v1에서는 |
|---|---|---|---|
| `g3-0136` Yard, blue cone, 54m 뒤쪽 | 27번째 판단에 찾았지만 다가가지 않고 55m에서 스스로 멈춤 | F3 | 성공 (129 판단) |
| `g3-0139` Yard, green cylinder, 44m 뒤쪽 | 13.3m까지 갔지만 240 판단 안에 멈추지 않음 | F5 | 성공 |
| `g2-0110` pyramid, 49m 오른쪽 | 5번째 판단에 봤지만 다가가지 않음 | F3 | 같은 실패 |
| `g4-0161` Yard pyramid, 47m 정면 | 보이는데 다가가지 않음 | F3 | 같은 실패 |
| `g4-0165` Yard pyramid, 51m 뒤쪽 | 18번째 판단에 봤지만 다가가지 않음 | F3 | 같은 실패 |

**다섯 모두 44m 이상에서 시작한 경우다.** 셋은 처음 보는 물체, 둘은 Yard의 본 물체다. Blocks의 본 물체(G0, G1, P, S)에서는 92회 중 실패가 없다.

### Regression

- G0 20/20 유지, S 16/16 유지, 충돌 0 유지.
- **G3는 19/20 → 18/20이다.** 1회 차이이고 20회 표본에서는 의미를 두기 어렵지만, 좋아지지 않은 것은 사실이다. Gen-v1이 성공했던 먼 탐색 2회를 놓쳤고, Gen-v1이 놓쳤던 1회는 성공했다. 보강 데이터는 모두 Blocks에서 모았다.
- 어느 level도 10%p 이상 떨어지지 않았다.

## 움직임

같은 G1 시작 상태다. 기존 AeroVLA 두 방식은 16회, 나머지는 32회다.

| Model | 판단 주기 | 추론 시간 | 연속한 두 행동의 차이 | 판단당 heading 변화: 평균 / 최대 | 평균 속도 | 충돌 | 성공 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AeroVLA Step | 6.42초 | 1,110ms | 6.9% | 4.2° / 69° | 0.65m/s | 1/16 | 1/16 |
| AeroVLA Continuous | 2.97초 | 1,157ms | 4.6% | 3.1° / 65° | 0.89m/s | 0/16 | 0/16 |
| OFT Pilot | 0.55초 | 425ms | 2.4% | 4.2° / 14° | 0.22m/s | 3/32 | 4/32 |
| OFT Gen-v1 | 0.53초 | 398ms | 2.8% | 3.6° / 15° | 0.63m/s | 0/32 | 27/32 |
| OFT Gen-v2 | 0.51초 | 396ms | 3.3% | 2.6° / 19° | 0.99m/s | 0/32 | 32/32 |
| Teacher | 0.51초 | - | 3.0% | 3.4° / 16° | 1.20m/s | 0/32 | 32/32 |

| 범위 밖 예측 (G1) | 빈도 | 초과량 중앙값 | 최대 |
|---|---:|---:|---:|
| Gen-v1 | 54% | 0.031 | 0.19 |
| Gen-v2 | 26% | 0.008 | 0.07 |

- **무엇이 무엇을 바꿨는지는 나눠서 읽어야 한다.**
  - *연속 실행기*: 기존 AeroVLA의 판단 주기를 6.4초 → 3.0초로 줄였다. 성공률은 바꾸지 못했다(1/16, 0/16).
  - *OFT의 병렬 디코딩*: 추론을 1.1초 → 0.4초로 줄였다.
  - *작은 연속 행동*: 판단당 heading 변화의 최댓값을 65–69° → 14–19°로 줄였다.
  - *학습 데이터*: 구조와 실행기가 같은 pilot, Gen-v1, Gen-v2의 성공이 4/32, 27/32, 32/32다.
- **Gen-v2는 머뭇거림이 줄어 평균 속도가 teacher에 가까워졌다**(0.63 → 0.99m/s, teacher 1.20m/s).
- **범위 밖 예측은 줄었지만 목표로 삼지 않았다.** head는 그대로 linear이고 실행 때 한계값으로 자른다. tanh head는 쓰지 않았다(같은 조건에서 검증 L1 0.151 대 0.383).
- **탐색 회전이 teacher의 절반 속도인 것은 그대로다.** 실행기 쪽 성질이고 이번에도 고치지 않았다([Gen-v1 문서](aerovla_oft_generalization.md#움직임)).

## 해석

**Gen-v1에서 남은 실패는 어디서 왔는가.**

| 가설 | 판정 | 근거 |
|---|---|---|
| 목표를 못 찾아서 | 아니다 | 두 모델 모두 처음에 안 보이던 목표 73개를 전부 시야에 넣었다 |
| 찾은 뒤 전환·접근·정지를 못 해서 | **그렇다 (본 물체의 주된 원인)** | 본 물체의 실패 9회 중 8회가 이 모습이었고, 그 부분의 데이터만 더하자 8회가 모두 없어졌다(G1 27 → 32/32, P 21 → 24/24) |
| 새 물체의 언어 grounding이 약해서 | 일부만 | 문장은 따른다(대조 시험). 처음 보는 물체의 실패 7회 중 4회는 전환 데이터만으로 없어졌고(G2 9 → 11, G4 2 → 4), 남은 3회는 44m 이상에서 "보이는데 다가가지 않음"이다 |

**중요한 발견**

1. **병목은 탐색이 아니라 찾은 뒤였다.** scan progress 같은 상태를 추가하지 않고, 구조를 바꾸지 않고, 전환 장면 153 episode를 더한 것으로 이른 정지와 계속 상승이 8 → 0이 됐다.
2. **높은 곳에서의 정지는 데이터로 고쳐졌다.** 벽을 넘은 뒤의 정지 거리가 15.0–16.8m에서 10.3–12.3m가 됐다. teacher의 정지 규칙(12m)은 바꾸지 않았고, 정지 장면을 높이 5–15m에 걸쳐 보여 줬을 뿐이다.
3. **처음 보는 물체에서의 실패 상당수는 grounding 문제가 아니었다.** yellow pyramid는 여전히 학습에 없는데 G2 9 → 11/12, G4 2 → 4/6이 됐고, 걸리는 시간도 절반이 됐다(148 → 70 판단). 물체를 알아보지 못한 것이 아니라 찾은 뒤에 머뭇거린 것이었다.
4. **문장이 행동을 정한다.** 같은 자리에서 다른 물체를 지시하면 가까이 보이는 물체로 가지 않는다(Gen-v1 대조 시험: pyramid 옆에서 0/12, Yard에서 2/20). 학습에 없던 문장 3종은 Gen-v2에서 24/24다.
5. **오른쪽만 도는 탐색은 성공률을 깎지 않는다.** 좌우 시험 16/16이고 목표가 왼쪽이면 12초 더 걸릴 뿐이다. 탐색 시간의 더 큰 몫은 실행기에 있다(추론 0.4초 동안의 회전이 반영되지 않아 teacher의 절반 속도로 돈다).

**다음 단계 (하나만).** Blocks의 본 물체는 92회 중 92회, Yard는 20회 중 18회, 충돌 0이다. 탐색과 정지가 안정됐으므로 **Visual Search + Gaussian Blur robustness 평가**로 넘어간다. 이때 흐림 없는 조건을 **새 seed로 만든 held-out 시작 상태**에서 먼저 재서 기준선으로 삼는다. 지금의 held-out set은 Gen-v2를 만드는 데 쓰였기 때문이다(아래 한계 1). 처음 보는 물체의 grounding(FiLM, 물체·문장 다양화)과 실행기의 회전 속도는 그다음 후보다.

## 한계

1. **Gen-v2의 개선은 같은 held-out set을 보고 고친 결과다.** 보강할 종류를 G1–G3의 실패 모습에서 골랐다. 학습 시작점은 held-out 시작점과 여전히 6m 이상 떨어져 있고 비워 둔 방향도 그대로지만, 이 숫자는 새 held-out 추정치가 아니다. 새 seed의 시작 상태로 다시 재야 한다.
2. **데이터와 학습량이 함께 달라졌다.** 같은 선택 규칙에서 Gen-v1은 1,250 update, Gen-v2는 3,999 update가 골라졌다. "Gen-v1 데이터로 4,000 update"는 돌리지 않았으므로 보강 데이터만의 몫은 분리되지 않았다.
3. **표본이 작고 한 번씩만 비행했다.** 같은 시작도 다시 비행하면 결과가 갈린다. Gen-v1의 고도 실패 2개를 GIF용으로 다시 비행했을 때 14.9m와 14.3m에서 멈춰 성공이었다. G3의 19 → 18, G4의 2 → 4 같은 1–2회 차이는 방향만 참고해야 한다.
4. **Yard는 같은 simulator 안의 다른 장면이다.** 구조물, 바닥, 조명은 다르지만 renderer와 하늘은 같다. "처음 보는 simulator 환경"에서의 결과가 아니고, 일반적인 UAV 탐색이 풀렸다는 뜻도 아니다.
5. **먼 거리(44m 이상)에서 보이는 목표로 다가가지 않는 실패가 남았다.** 5회 모두 이 모습이고 Yard나 처음 보는 물체에서 나왔다. 이번에 손대지 않았다.
6. **처음 보는 물체는 하나(yellow pyramid)뿐이고 학습에 넣지 않았다.** G2/G4가 오른 이유를 "그 물체를 더 잘 알아봐서"라고 할 근거는 없다.
7. **비교군은 줄여서 돌렸다.** pilot과 기존 AeroVLA는 G1 중심이고, G0·G4 일부는 비어 있다.
8. **탐색은 여전히 기억 없는 한 방향 회전이고, 고도 상황은 벽 4개가 만든 것이다.** 흐림, 바람, 센서 잡음은 없다.

## 실행

```powershell
# 데모와 평가는 Gen-v1과 같고 checkpoint만 다르다
.\scripts\run_visual_search_demo.ps1 -Model oft -Checkpoint outputs/aerovla_oft/checkpoints/generalization_v2 -Level G1 -Target red_cube
.\scripts\run_visual_search.ps1 -Policy oft -Checkpoint outputs/aerovla_oft/checkpoints/generalization_v2 -Plan configs/generalization_test_spawns.json -Set G1 -Output outputs/visual_search/gen_v2_g1

# 대조 시험: 같은 시작에서 다른 물체를 지시한다 (-Named 물체 또는 another)
.\scripts\run_visual_search.ps1 -Policy oft -Checkpoint outputs/aerovla_oft/checkpoints/generalization_v1 -Plan configs/generalization_test_spawns.json -Set G3 -Named another -Output outputs/visual_search/control_g3_told_another

# 결과 동결과 두 모델 비교 (실행 폴더는 읽기만 한다)
.\assets\projectairsim-env\Scripts\python.exe scripts\freeze_generalization_results.py "OFT Gen-v1:G1=outputs/visual_search/gen_v1_g1" "OFT Gen-v2:G1=outputs/visual_search/gen_v2_g1" --compare "OFT Gen-v1" "OFT Gen-v2" --output outputs/generalization/gen_v2_final

# 보강 episode: 계획(검사 포함) → 수집. 두 번째 simulator는 -TopicsPort 9089 -ServicesPort 9090과 -Skip/-Limit으로 나눠 돌린다
.\assets\projectairsim-env\Scripts\python.exe scripts\plan_generalization.py targeted --output datasets/projectairsim_visual_search/generalization_v2_added
.\scripts\run_visual_search.ps1 -Policy teacher -Plan datasets/projectairsim_visual_search/generalization_v2_added/plan.json -Set train -Record datasets/projectairsim_visual_search/generalization_v2_added -Output outputs/visual_search/gen_v2_collect_train
.\scripts\run_visual_search.ps1 -Policy teacher -Plan datasets/projectairsim_visual_search/generalization_v2_added/plan.json -Set val -Record datasets/projectairsim_visual_search/generalization_v2_added -Output outputs/visual_search/gen_v2_collect_val
```

```bash
# 병합(복사 없이 참조, 중복이 있으면 멈춤) → 학습 → 다른 dataset의 검증 파일로 점수
python scripts/build_visual_search_dataset.py datasets/projectairsim_visual_search/generalization_v2 \
  --sources datasets/projectairsim_visual_search/generalization_v1 datasets/projectairsim_visual_search/generalization_v2_added
python scripts/train_aerovla_oft.py --dataset datasets/projectairsim_visual_search/generalization_v2 --output outputs/aerovla_oft/checkpoints/generalization_v2 \
  --steps 4000 --eval-every 250 --eval-samples 320 --patience 4 --strategies right --name OFT-Generalization-v2
python scripts/train_aerovla_oft.py --score outputs/aerovla_oft/checkpoints/generalization_v2 --dataset datasets/projectairsim_visual_search/generalization_v1
```

checkpoint와 dataset은 Git에 없다. Gen-v1의 checkpoint(`checkpoints/generalization_v1`), dataset, 실행 폴더는 그대로 있다.

## Tests

- **Python 146/146, PowerShell 4/4 통과**(WSL). 이 단계에서 추가한 것: 보강 종류가 기반 종류의 규칙과 자기 범위(거리, 방향, 높이)를 지킴, 기존 종류의 계획은 바뀌지 않음(held-out 파일이 여전히 생성기 출력과 같음), 보강 계획이 자기 seed를 쓰고 held-out 시작점·wedge와 떨어져 있음, 병합이 중복 episode·seed·시작점·폴더를 거부, 실패마다 분류가 하나, 단계(찾기·접근·정지) 계산, 다른 물체를 지시하는 옵션이 점수 대상은 그대로 둠.
- **Checkpoint 재로드:** `generalization_v2`를 새 프로세스에서 불러 저장 시점 예측과 비교했고 차이는 0.0이다.
- **Live:** 보강 수집 154회(teacher 미도착 1회는 dataset에서 제외), Gen-v2 held-out 130회, 기록용 비행 9회다. G2 실행의 첫 두 episode가 이륙 단계의 simulator 응답 시간 초과로 오류가 났고, 정책과 무관한 프로그램 오류라 그 둘만 다시 비행했다(오류 episode는 결과에 넣지 않고 다시 비행하는 것이 pilot 때부터의 규칙이다). 충돌은 0건이다.
- **바꾸지 않은 것의 확인:** Gen-v1의 checkpoint 폴더, dataset, 실행 폴더는 읽기만 했다. 학습 스크립트는 checkpoint가 있는 폴더로의 학습을 거부한다.
