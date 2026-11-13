# AeroVLA-OFT Gen-v2 — 새 장거리 held-out 평가

2026-10-08 실행. 브랜치 `exp/aerovla-oft-generalization`. [Gen-v2](aerovla_oft_gen_v2.md)의 checkpoint를 그대로 썼다. 이 단계에서 학습, 모델 수정, teacher·executor·성공 기준 변경은 없다.

**요약**

- **Gen-v2에 남았던 실패 5회가 모두 44m 이상에서 시작했던 이유를 따로 떼어 봤다.** 새 seed로 시작 거리 40–90m의 held-out 시작 36개(L1 40–55m, L2 55–70m, L3 70–90m, 각 12개)를 비행 전에 고정했다.
- **결과는 27/36이다.** L1 11/12, L2 8/12, L3 8/12. 충돌 0, 다른 물체 옆에 멈춘 경우 0. teacher는 같은 시작에서 36/36이다.
- **거리 자체가 원인은 아니다.** 학습한 장면(Blocks)에서는 17/18이고, 학습 범위(최대 70m) 밖인 70–90m도 6/6이다. 실패 9회 중 8회는 Yard이고 그중 8회 모두 56m 이상이다(Yard: 55m 이하 6/6, 그 위 4/12).
- **탐색 실패는 0이다.** simulator 기준으로 목표는 36회 모두 시야에 들어왔다. 실패한 9회 중 8회는 목표가 시야를 4–12번 지나가는 동안 계속 회전만 했다(240 판단 동안 약 7바퀴). 목표가 화면에 있어도 모델이 그것을 목표로 반응하지 않았다는 뜻이다.
- **접근을 시작한 비행은 모두 끝까지 갔다.** 20m 안에 들어온 27회가 전부 15m 안에서 스스로 멈췄다(정지 거리 9.6–12.5m). 정지 실패 0.
- **처음 보는 물체는 더 약하다**(7/12 대 본 물체 20/24). 다만 Yard에서는 본 물체도 56m 이상에서 4/8이라, 물체만의 문제는 아니다.
- **판정: 부분적으로 일반화한다.** 다음에 바꿀 것은 탐색 구조가 아니라, 멀리 작게 보이는 목표에 접근을 시작하는 장면의 데이터(장면 외관을 달리해서)다.

## 무엇을 묻는가

Gen-v2의 남은 실패가 (A) 찾은 뒤 먼 거리를 다가가지 못해서인지, (B) 먼 거리에서는 찾지 못해서인지, (C) 처음 보는 물체에서만 그런지를, 기존 평가와 겹치지 않는 시작 상태에서 단계별로 나눈다.

## 평가 set `L`

[configs/long_range_test_spawns.json](../configs/long_range_test_spawns.json). `scripts/plan_generalization.py long`이 한 번 쓰고 다시 쓰지 않는다. 비행 전에 commit했다(`75b8d22`).

| | L1 | L2 | L3 |
|---|---:|---:|---:|
| 시작 거리 | 40–55m | 55–70m | 70–90m |
| Episode | 12 | 12 | 12 |
| Blocks / Yard | 6 / 6 | 6 / 6 | 6 / 6 |
| 처음부터 보임 / 안 보임 | 6 / 6 | 6 / 6 | 6 / 6 |
| 본 물체 / 처음 보는 물체(yellow pyramid) | 8 / 4 | 8 / 4 | 8 / 4 |

- **Seed:** 6000–6096. 기존 범위(test 0–199, train 1000–4999, val 5000–5999) 밖이다.
- **분리:** 모든 시작은 같은 맵의 학습·검증 시작 446개(G0는 그중 20개의 재비행), 기존 held-out 시작(G1–G4, P, S)과 6m 이상 떨어져 있다. 가장 가까운 학습 시작까지 6.7m, 기존 held-out 시작까지 6.8m다. 10m로 잡으면 Blocks의 40–55m 구간에 시작할 자리가 없어 기존 기준 6m를 썼다. Yard에는 학습 시작이 없다.
- **학습 범위와의 관계:** 학습 episode의 시작 거리는 최대 70.0m다. L1·L2는 Blocks에서 학습 범위 안의 거리이고, L3는 어느 장면에서도 학습 범위 밖이다.
- **시작 조건:** 기존 held-out과 같은 규칙이다. 목표까지 직선이 트여 있고, teacher가 맵의 box 위 dry run에서 도착한다. "안 보임"은 목표가 앞 카메라 시야 밖에 있는 시작(옆·뒤)이다.
- **맵의 한계:** Blocks의 blue cone은 70–90m에서 직선이 트인 자리가 없다. L3의 그 두 자리는 orange ball과 red cube가 대신했다(파일의 `planned_object`). 그래서 L3 Blocks의 본 물체는 ball 1, cube 2, cylinder 1이다.
- **기록:** episode마다 map, layout, object, 시작 거리, 시작 방위, 시작 높이, seed, 본 물체 여부, 가장 가까운 기존 시작까지의 거리.

## 측정

`scripts/freeze_long_range.py`가 실행 폴더를 읽어 [outputs/generalization/long_range_v2/](../outputs/generalization/long_range_v2/)에 쓴다(`summary.json`, `episodes.csv`, `range_results.csv`, `seen_unseen.csv`, `map_results.csv`, `failure_types.csv`, `failures/`에 실패 9회 각각의 기록과 그림).

| 단계 | 기준 |
|---|---|
| SEARCH SUCCESS | 목표가 앞 또는 아래 카메라 시야에 한 번이라도 들어옴(simulator의 depth로 판정) |
| APPROACH SUCCESS | 목표 20m 안에 들어옴 |
| TASK SUCCESS | 스스로 정지 + 15m 이내 + 충돌 없음(기존 기준 그대로) |

실패 분류는 다음 순서로 하나만 붙는다: `COLLISION` → `LONG_SEARCH_FAILURE`(한 번도 안 보임) → `UNSEEN_OBJECT_GROUNDING_FAILURE`(처음 보는 물체를 지시했는데 다른 물체 옆에서 끝남) → `LONG_APPROACH_FAILURE`(보였지만 20m 안에 못 들어옴) → `LONG_STOP_FAILURE`(20m 안에 들어왔지만 반경 안 정지로 끝나지 않음). `TIMEOUT`은 끝난 방식이라 따로 센다(240 판단).

"시야에 들어옴"은 simulator의 말이지 모델의 말이 아니다. 그래서 목표가 시야에 있을 때 모델이 무엇을 했는지도 적는다: 목표가 시야에 들어온 횟수, 시야에 있던 판단 수, 그때의 전진량, 전체 회전량. 목표가 두 번 이상 시야를 지나갔는데 계속 회전한 실패는 `passed_over`, 반경 밖에서 스스로 멈춘 실패는 `early_stop`으로 표시한다.

## 거리별 결과

| Range | Acquisition | 안 보이던 시작의 획득 | <=20m | Self-stop | Final Success | Collision | Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| 40–55m | 12/12 | 6/6 | 11/12 | 11/12 | 11/12 | 0 | 1 |
| 55–70m | 12/12 | 6/6 | 8/12 | 8/12 | 8/12 | 0 | 4 |
| 70–90m | 12/12 | 6/6 | 8/12 | 9/12 | 8/12 | 0 | 3 |
| 합계 | 36/36 | 18/18 | 27/36 | 28/36 | 27/36 | 0 | 8 |

Teacher는 같은 시작에서 36/36이다(획득 18/18, timeout 0, 최대 106 판단). 시작 상태는 모두 풀 수 있고, 240 판단은 가장 먼 시작에도 넉넉하다. 모델의 성공 비행은 중앙값 78, 최대 203 판단이었다.

거리와 장면을 나누면 다음과 같다.

| Range | Blocks | Yard |
|---|---:|---:|
| 40–55m | 5/6 | 6/6 |
| 55–70m | 6/6 | 2/6 |
| 70–90m | 6/6 | 2/6 |
| 합계 | 17/18 | 10/18 |

- **Blocks에서는 거리에 따라 떨어지지 않는다.** 학습 범위 밖인 70–90m도 6/6이다. 유일한 실패는 44m에서 시작한 처음 보는 물체다.
- **Yard에서는 약 55m가 경계다.** 55m 이하 6/6(기존 G3의 18/20도 54m 이하였다), 56m 이상 4/12. 성공한 4회는 64, 67, 84, 88m 시작이라 "멀수록 못 한다"는 단조로운 관계도 아니다.

## Acquisition 대 Approach 대 Stop

| 단계 | 결과 |
|---|---:|
| SEARCH SUCCESS (목표가 시야에 들어옴) | 36/36 |
| 그중 처음에 안 보이던 시작 | 18/18 (획득까지 중앙값 21 판단, teacher 10.5) |
| APPROACH SUCCESS (20m 이내) | 27/36 |
| 20m 안에 들어온 뒤 TASK SUCCESS | 27/27 |

- **정지는 문제가 아니다.** 20m 안에 들어온 27회가 모두 9.6–12.5m에서 스스로 멈췄다. Gen-v1의 병목이었던 이른 정지·계속 상승은 이 set에서도 나타나지 않았다(1회 예외는 아래).
- **실패 9회는 모두 "보였지만 다가가지 않음"이다.** 그런데 접근하다 만 것이 아니다. 9회 중 8회는 거의 전진하지 않았다(목표가 시야에 있던 판단에서 전진 평균 0.01–0.29m, 성공 비행은 0.76m).
- **그 8회는 목표를 지나치며 계속 돌았다.** 목표가 시야에 4–12번 들어왔고(시야에 있던 판단 11–72개), 240 판단 동안 2,100–2,750° 회전했다. teacher는 그 판단들에서 `approach`를 냈고 모델은 탐색 회전을 냈다.
- **처음부터 목표가 정면에 있던 시작도 6회 실패했다**(보임 시작 12/18, 안 보임 시작 15/18). 정면의 목표를 두고 회전을 시작했다. 탐색 순서의 문제가 아니라 그 화면을 "목표가 보인다"로 읽지 못한 것이다.
- **나머지 1회**(`l3-6091`, Yard, red cube, 85m)는 목표가 2 판단만 시야에 있다가 벗어났고, 다른 방향으로 약 20m 전진한 뒤 72m에서 스스로 멈췄다. 근처에 다른 물체는 없었다(가장 가까운 물체 116m).

즉 단계 표만 보면 "획득은 높고 최종 성공만 낮음"(A)이지만, 기록을 보면 접근 궤적이 길어서 실패한 것이 아니라 먼 목표를 목표로 알아보지 못해 접근을 시작하지 않은 것이다.

## 본 물체 대 처음 보는 물체

| Range | 본 물체 | 처음 보는 물체 |
|---|---:|---:|
| 40–55m | 8/8 | 3/4 |
| 55–70m | 6/8 | 2/4 |
| 70–90m | 6/8 | 2/4 |
| 합계 | 20/24 | 7/12 |

| | Blocks | Yard |
|---|---:|---:|
| 본 물체 | 12/12 | 8/12 |
| 처음 보는 물체 | 5/6 | 2/6 |

- **처음 보는 물체가 더 약한 것은 맞다.** 다만 실패 방식이 본 물체와 같다(목표를 지나치며 회전). 다른 물체로 간 경우는 0이다.
- **물체만으로는 설명되지 않는다.** Yard의 본 물체도 56m 이상에서 4/8이다. 장면과 물체가 각각 낮추고, 둘이 겹친 Yard의 처음 보는 물체가 가장 낮다(2/6).
- **표본이 작다.** 칸마다 4–12회라 칸 사이의 차이를 크기로 읽으면 안 된다. 방향만 본다.

## 실패 분류

| Type | Count | Timeout | 지나치며 회전 | 이른 정지 | 처음 보는 물체 | L1 | L2 | L3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG_SEARCH_FAILURE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LONG_APPROACH_FAILURE | 9 | 8 | 8 | 1 | 5 | 1 | 4 | 4 |
| LONG_STOP_FAILURE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| UNSEEN_OBJECT_GROUNDING_FAILURE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| COLLISION | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Timeout 8회는 모두 위의 회전 실패다. 끝날 때 가까워지고 있던 비행은 없어서(마지막 20 판단의 거리 변화 0.4m 이하) 판단 수를 늘려도 결과가 달라지지 않는다.

| Episode | 장면 | 물체 | 시작 | 최소 거리 | 끝 | 시야에 들어온 횟수 | 회전 |
|---|---|---|---:|---:|---|---:|---:|
| `l1-6008-yellow_pyramid-visible` | blocks | yellow_pyramid | 43.8m | 42.0m | timeout | 5 | 2092° |
| `l2-6017-blue_cone-search` | yard | blue_cone | 65.4m | 63.8m | timeout | 8 | 2523° |
| `l2-6018-orange_ball-visible` | yard | orange_ball | 64.3m | 56.4m | timeout | 6 | 2660° |
| `l2-6023-yellow_pyramid-search` | yard | yellow_pyramid | 65.2m | 58.5m | timeout | 9 | 2547° |
| `l2-6024-yellow_pyramid-visible` | yard | yellow_pyramid | 59.1m | 58.6m | timeout | 12 | 2383° |
| `l3-6089-blue_cone-visible` | yard | blue_cone | 77.7m | 73.9m | timeout | 6 | 2442° |
| `l3-6091-red_cube-visible` | yard | red_cube | 85.3m | 71.9m | 정지 | 1 | 511° |
| `l3-6095-yellow_pyramid-visible` | yard | yellow_pyramid | 77.7m | 77.7m | timeout | 11 | 2590° |
| `l3-6096-yellow_pyramid-search` | yard | yellow_pyramid | 89.1m | 88.7m | timeout | 4 | 2746° |

대표 그림: [Yard, orange ball, 64m](../outputs/generalization/long_range_v2/failures/OFT_Gen-v2_L2_LONG_APPROACH_FAILURE_l2-6018-orange_ball-visible.jpg). 나머지는 같은 폴더에 있다.

## Gen-v2는 장거리로 일반화하는가

**부분적으로(PARTIALLY).**

- **예:** 학습한 장면에서는 학습 범위 밖 거리(70–90m)까지 그대로 된다. 어느 장면에서도 목표는 시야에 들어오고, 접근을 시작하면 끝까지 가서 제대로 멈춘다. 충돌과 다른 물체로 가는 경우는 없다.
- **아니오:** 학습하지 않은 장면에서 56m 이상이면 4/12다. 멀리 작게 보이는 목표를 알아보는 능력이 학습한 장면의 외관에 묶여 있다.
- Gen-v2 문서의 "남은 실패는 44m 이상"은 거리의 문제처럼 읽혔지만, 새 set에서 보면 **거리 × 장면(과 물체)** 의 문제다.

## 다음에 바꿀 것

미리 정한 세 갈래에 비추면 다음과 같다.

| 갈래 | 조건 | 이 결과 |
|---|---|---|
| A | 획득은 높고 최종 성공만 낮음 | 조건은 맞다(36/36 대 27/36). 다만 원인은 긴 접근 궤적이 아니라 접근을 시작하지 않는 것 |
| B | 장거리에서 획득 자체가 낮음 | 아니다(18/18). 목표는 시야를 여러 번 지나갔다 |
| C | 처음 보는 물체에서만 낮음 | 아니다. 더 낮지만 본 물체도 Yard에서 실패한다 |

- **탐색 구조는 바꾸지 않는다.** `search_scan_progress`와 "한 바퀴 → 상승 → 재탐색"은 목표가 시야에 안 들어올 때의 처방이다. 여기서는 목표가 시야에 4–12번 들어왔고, 한 바퀴를 셀 수 있어도 알아보지 못하는 화면은 그대로다.
- **권장하는 다음 변경 하나:** 먼 거리에서 목표가 작게 보이는 순간부터 접근을 시작해 정지까지 가는 궤적(far → medium → near → stop)을, **장면 외관을 달리해서** 더한다. 학습용 장면 변형(바닥 색, 조명 시각, 배경 구조물)을 Blocks 쪽에 만들고 55–90m 시작을 포함한다. Yard와 `L`은 계속 평가에만 쓴다.
- **그다음 후보:** 물체 다양성과 FiLM. 처음 보는 물체의 격차(7/12 대 20/24)가 위 변경 뒤에도 남으면 그때 다룬다.
- **주의:** `L`의 결과를 보고 다음 변경을 정했으므로, 그 변경 뒤 `L`의 점수는 새 held-out 추정치가 아니다. 그때는 seed 7000번대로 새 set을 고정해야 한다.

## Gaussian Blur로 넘어갈 수 있는가

**예(YES), 범위를 정해서.**

- clean 장거리 기준선이 이제 고정됐다(`outputs/generalization/long_range_v2/`). Blur에서의 실패를 이 표와 episode 단위로 비교할 수 있다.
- **Blur 효과를 읽을 수 있는 구간:** Blocks 전 거리(clean 17/18)와 Yard 55m 이하(clean 6/6, G3 18/20). 여기서는 clean 실패가 거의 없어 떨어진 만큼이 Blur의 영향이다.
- **읽을 수 없는 구간:** Yard 56m 이상(clean 4/12). 여기서는 Blur 없이도 대부분 실패하므로 Blur의 영향을 가려낼 수 없다. 같이 비행하되 따로 보고한다.
- 이번 단계에서 Blur는 실행하지 않았다.

## 이 결과가 말하지 않는 것

- **표본이 작다.** 36회, 칸마다 4–12회다. 27/36의 95% 구간은 대략 59–86%다.
- **Yard는 같은 simulator 안의 다른 장면이다.** 다른 simulator나 실제 환경에 대한 결과가 아니다. Yard에서 무엇이 화면을 달라 보이게 했는지(저녁 조명, 모래 바닥, 배경)는 나누지 않았다.
- **시작은 모두 목표까지 직선이 트인 자리다.** 먼 거리에서 구조물에 가려진 목표를 찾는 경우는 이 set에 없다.
- **`in view`는 depth로 판정한 기하학적 가시성이다.** 90m에서 10m 물체는 모델 입력(224px)에서 약 18px이다. 그 크기에서도 Blocks에서는 접근했다.
- **Gen-v1은 이 set에서 비행하지 않았다.** Gen-v2의 보강이 장거리에 준 영향은 여기서 알 수 없다.

## 재현

```powershell
# 시작 상태(이미 있으면 거부한다) → Gen-v2 그대로 비행 → teacher로 같은 시작 확인 → 동결
.\assets\projectairsim-env\Scripts\python.exe scripts\plan_generalization.py long
foreach ($s in 'L1','L2','L3') { .\scripts\run_visual_search.ps1 -Policy oft -Checkpoint outputs/aerovla_oft/checkpoints/generalization_v2 -Plan configs/long_range_test_spawns.json -Set $s -Output "outputs/visual_search/gen_v2_$($s.ToLower())" }
foreach ($s in 'L1','L2','L3') { .\scripts\run_visual_search.ps1 -Policy teacher -Plan configs/long_range_test_spawns.json -Set $s -Output "outputs/visual_search/teacher_$($s.ToLower())" }
.\assets\projectairsim-env\Scripts\python.exe scripts\freeze_long_range.py "OFT Gen-v2:L1=outputs/visual_search/gen_v2_l1" "OFT Gen-v2:L2=outputs/visual_search/gen_v2_l2" "OFT Gen-v2:L3=outputs/visual_search/gen_v2_l3" "Teacher:L1=outputs/visual_search/teacher_l1" "Teacher:L2=outputs/visual_search/teacher_l2" "Teacher:L3=outputs/visual_search/teacher_l3" --output outputs/generalization/long_range_v2
```

## Tests

- **Python 151/151 통과**(WSL). 추가한 것: 장거리 파일이 생성기 출력과 같음, 구간마다 두 장면·두 시작 종류·처음 보는 물체 4개, seed가 새 범위이고 모든 기존 시작과 6m 이상 떨어짐, L3가 학습 범위 밖, 단계와 실패 분류가 하나씩 붙음, 목표를 지나치며 회전한 실패의 구분, 표의 집계.
- **Live:** Gen-v2 36회(45분), teacher 36회(26분). 오류로 다시 비행한 episode 0, 충돌 0.
- **바꾸지 않은 것:** checkpoint 폴더 `generalization_v2`는 읽기만 했다. 기존 held-out 파일과 G0–G4, P, S의 결과는 그대로다.

## 전체 episode

| Episode | 장면 | 물체 | 본 물체 | 거리 m | 방위 ° | 높이 m | Seed | 처음 보임 | 획득 판단 | 최소 m | 최종 m | 20m | 15m | 끝 | 성공 |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| `l1-6000-blue_cone-visible` | blocks | blue_cone | 예 | 41.0 | 2 | 5.6 | 6000 | 예 | 0 | 11.0 | 11.0 | 예 | 예 | 정지 | 예 |
| `l1-6001-orange_ball-search` | blocks | orange_ball | 예 | 48.0 | 173 | 8.5 | 6001 | 아니오 | 19 | 10.0 | 10.0 | 예 | 예 | 정지 | 예 |
| `l1-6002-red_cube-visible` | blocks | red_cube | 예 | 46.9 | -7 | 6.3 | 6002 | 예 | 0 | 10.5 | 10.5 | 예 | 예 | 정지 | 예 |
| `l1-6003-green_cylinder-search` | blocks | green_cylinder | 예 | 42.4 | -74 | 6.0 | 6003 | 아니오 | 35 | 10.5 | 10.5 | 예 | 예 | 정지 | 예 |
| `l1-6004-blue_cone-visible` | yard | blue_cone | 예 | 54.6 | 0 | 6.0 | 6004 | 예 | 0 | 10.0 | 10.0 | 예 | 예 | 정지 | 예 |
| `l1-6005-orange_ball-search` | yard | orange_ball | 예 | 47.6 | 178 | 6.5 | 6005 | 아니오 | 21 | 11.6 | 11.6 | 예 | 예 | 정지 | 예 |
| `l1-6006-red_cube-visible` | yard | red_cube | 예 | 48.8 | 3 | 5.0 | 6006 | 예 | 0 | 11.8 | 11.8 | 예 | 예 | 정지 | 예 |
| `l1-6007-green_cylinder-search` | yard | green_cylinder | 예 | 41.8 | 105 | 6.4 | 6007 | 아니오 | 10 | 10.8 | 10.8 | 예 | 예 | 정지 | 예 |
| `l1-6008-yellow_pyramid-visible` | blocks | yellow_pyramid | 아니오 | 43.8 | -8 | 7.4 | 6008 | 예 | 0 | 42.0 | 66.3 | 아니오 | 아니오 | timeout | 아니오 |
| `l1-6009-yellow_pyramid-search` | blocks | yellow_pyramid | 아니오 | 44.1 | -156 | 6.2 | 6009 | 아니오 | 29 | 11.0 | 11.0 | 예 | 예 | 정지 | 예 |
| `l1-6010-yellow_pyramid-visible` | yard | yellow_pyramid | 아니오 | 42.9 | 0 | 8.7 | 6010 | 예 | 0 | 10.9 | 10.9 | 예 | 예 | 정지 | 예 |
| `l1-6011-yellow_pyramid-search` | yard | yellow_pyramid | 아니오 | 44.1 | -165 | 8.3 | 6011 | 아니오 | 22 | 11.4 | 11.4 | 예 | 예 | 정지 | 예 |
| `l2-6013-blue_cone-search` | blocks | blue_cone | 예 | 57.0 | -111 | 8.5 | 6013 | 아니오 | 51 | 12.3 | 12.3 | 예 | 예 | 정지 | 예 |
| `l2-6014-orange_ball-visible` | blocks | orange_ball | 예 | 61.8 | -6 | 6.3 | 6014 | 예 | 0 | 10.8 | 10.8 | 예 | 예 | 정지 | 예 |
| `l2-6015-red_cube-search` | blocks | red_cube | 예 | 56.2 | 173 | 7.2 | 6015 | 아니오 | 19 | 9.8 | 9.8 | 예 | 예 | 정지 | 예 |
| `l2-6016-green_cylinder-visible` | blocks | green_cylinder | 예 | 55.3 | 7 | 8.2 | 6016 | 예 | 0 | 11.2 | 11.2 | 예 | 예 | 정지 | 예 |
| `l2-6017-blue_cone-search` | yard | blue_cone | 예 | 65.4 | 88 | 8.6 | 6017 | 아니오 | 8 | 63.8 | 63.8 | 아니오 | 아니오 | timeout | 아니오 |
| `l2-6018-orange_ball-visible` | yard | orange_ball | 예 | 64.3 | -10 | 5.2 | 6018 | 예 | 0 | 56.4 | 56.4 | 아니오 | 아니오 | timeout | 아니오 |
| `l2-6019-red_cube-search` | yard | red_cube | 예 | 64.0 | -163 | 8.3 | 6019 | 아니오 | 22 | 12.4 | 12.4 | 예 | 예 | 정지 | 예 |
| `l2-6020-green_cylinder-visible` | yard | green_cylinder | 예 | 66.5 | 9 | 7.5 | 6020 | 예 | 0 | 12.5 | 12.5 | 예 | 예 | 정지 | 예 |
| `l2-6021-yellow_pyramid-search` | blocks | yellow_pyramid | 아니오 | 55.2 | 160 | 8.5 | 6021 | 아니오 | 96 | 11.5 | 11.5 | 예 | 예 | 정지 | 예 |
| `l2-6022-yellow_pyramid-visible` | blocks | yellow_pyramid | 아니오 | 56.5 | 1 | 8.9 | 6022 | 예 | 0 | 11.2 | 11.2 | 예 | 예 | 정지 | 예 |
| `l2-6023-yellow_pyramid-search` | yard | yellow_pyramid | 아니오 | 65.2 | 96 | 7.6 | 6023 | 아니오 | 9 | 58.5 | 58.5 | 아니오 | 아니오 | timeout | 아니오 |
| `l2-6024-yellow_pyramid-visible` | yard | yellow_pyramid | 아니오 | 59.1 | -2 | 7.1 | 6024 | 예 | 0 | 58.6 | 73.1 | 아니오 | 아니오 | timeout | 아니오 |
| `l3-6045-orange_ball-visible` | blocks | orange_ball | 예 | 79.1 | 9 | 7.6 | 6045 | 예 | 0 | 9.6 | 9.6 | 예 | 예 | 정지 | 예 |
| `l3-6086-red_cube-search` | blocks | red_cube | 예 | 81.8 | 166 | 8.4 | 6086 | 아니오 | 18 | 11.1 | 11.1 | 예 | 예 | 정지 | 예 |
| `l3-6087-red_cube-visible` | blocks | red_cube | 예 | 77.9 | -6 | 6.0 | 6087 | 예 | 0 | 10.0 | 10.0 | 예 | 예 | 정지 | 예 |
| `l3-6088-green_cylinder-search` | blocks | green_cylinder | 예 | 85.8 | 150 | 6.6 | 6088 | 아니오 | 20 | 11.0 | 11.0 | 예 | 예 | 정지 | 예 |
| `l3-6089-blue_cone-visible` | yard | blue_cone | 예 | 77.7 | 0 | 8.0 | 6089 | 예 | 0 | 73.9 | 73.9 | 아니오 | 아니오 | timeout | 아니오 |
| `l3-6090-orange_ball-search` | yard | orange_ball | 예 | 84.3 | -178 | 8.4 | 6090 | 아니오 | 20 | 11.6 | 11.6 | 예 | 예 | 정지 | 예 |
| `l3-6091-red_cube-visible` | yard | red_cube | 예 | 85.3 | -6 | 6.2 | 6091 | 예 | 0 | 71.9 | 71.9 | 아니오 | 아니오 | 정지 | 아니오 |
| `l3-6092-green_cylinder-search` | yard | green_cylinder | 예 | 87.5 | -167 | 8.2 | 6092 | 아니오 | 26 | 11.2 | 11.2 | 예 | 예 | 정지 | 예 |
| `l3-6093-yellow_pyramid-visible` | blocks | yellow_pyramid | 아니오 | 70.9 | 10 | 6.8 | 6093 | 예 | 0 | 11.5 | 11.5 | 예 | 예 | 정지 | 예 |
| `l3-6094-yellow_pyramid-search` | blocks | yellow_pyramid | 아니오 | 74.0 | 125 | 6.5 | 6094 | 아니오 | 17 | 11.3 | 11.3 | 예 | 예 | 정지 | 예 |
| `l3-6095-yellow_pyramid-visible` | yard | yellow_pyramid | 아니오 | 77.7 | -5 | 9.0 | 6095 | 예 | 0 | 77.7 | 80.3 | 아니오 | 아니오 | timeout | 아니오 |
| `l3-6096-yellow_pyramid-search` | yard | yellow_pyramid | 아니오 | 89.1 | -81 | 5.6 | 6096 | 아니오 | 38 | 88.7 | 89.0 | 아니오 | 아니오 | timeout | 아니오 |
