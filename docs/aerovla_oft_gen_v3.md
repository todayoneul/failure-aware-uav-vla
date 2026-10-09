# AeroVLA-OFT Gen-v3 — 정확한 언어 grounding, 먼 목표 인식, 착륙

2026-10-08 시작. 브랜치 `exp/aerovla-oft-gen-v3-grounding-landing`(`exp/aerovla-oft-generalization`에서 분기). [Gen-v2](aerovla_oft_gen_v2.md)의 checkpoint, dataset, 평가 결과와 [장거리 평가](aerovla_oft_long_range.md)는 그대로 두고 그 위에 추가했다.

**요약**

- **Gen-v3는 아직 clean baseline이 아니다.** 평가 앞에 둔 gate(pilot → smoke 10회 → representative 36회)를 끝까지 통과하지 못했고, 그래서 새 held-out test set 156개(Scene C, Depot)는 **한 번도 비행하지 않았다.** 아래 숫자는 검증 시작(학습 장면, 학습에 안 쓴 seed)과 기존 평가 set의 것이다.
- **두 번 학습했다.** 첫 checkpoint는 smoke에서 6/10으로 멈췄다. 검증 실패를 보고 199 episode를 더해 처음부터 다시 학습한 두 번째 checkpoint는 smoke를 통과했고(7/10) representative에서 27/36(0.75, 기준 0.80)으로 멈췄다.
- **된 것:** 문장이 비행의 끝을 정한다(approach를 시켰는데 내려앉은 비행 0/18). 착륙 18회 중 16회가 지시한 pad에 내려앉았고 충돌은 0이다. 파랑이 아닌 목표는 56–70m 시작을 포함해 12/12다. 기존 set은 Gen-v2의 101/112에서 103/112이고, Yard의 먼 "본 물체"는 8/12에서 11/12가 됐다.
- **안 된 것 하나:** 장면에 파란 물체가 넷(pad, cube, cylinder, cone)인데, 54m 이상에서 그중 지시한 것을 확정하지 못한다(파란 목표 3/9). 44m 이하에서는 헷갈림이 1/15다. 보강은 "다른 파란 물체로 감"을 "먼 파란 목표 앞에서 망설임"으로 바꿨을 뿐 합계는 그대로였다.
- **그다음 것:** 내려앉은 뒤 0 행동을 내지 못한 착륙이 2/18이다. pad 위에 가만히 서서 약한 하강 명령을 계속 낸다.
- **다음에 정할 것은 모델이 아니라 과제의 범위다.** 권장: canonical clean mission을 시작 44m 이하로 정하고(두 checkpoint 모두 19/22), 먼 거리의 같은 색 구분은 한계로 남긴다. 세 번째 재학습은 그 결정 뒤의 일이다.

## 왜 Gen-v3인가

- **Gen-v2에 남은 한계는 "보이는데 가지 않는 것"이었다.** 새 장거리 평가 36회에서 목표는 매번 시야에 들어왔고(36/36), 20m 안에 들어온 27회는 전부 제대로 멈췄다. 실패 9회 중 8회는 목표가 시야를 4–12번 지나가는 동안 회전만 했다. 학습한 장면에서는 70–90m도 6/6이었고 실패는 학습하지 않은 장면의 56m 이상에 몰렸다.
- **지금까지의 과제는 Failure-Aware 실험의 기준 과제로 쓰기에 좁았다.** 장면에 이름 붙은 물체가 2–4개뿐이었고 색과 형태가 모두 달라서, 문장을 정확히 읽지 않아도 맞을 수 있었다. 끝도 "15m 안에서 멈춤"뿐이었다.
- **Gen-v3는 세 가지를 clean 조건에서 먼저 안정화한다.** (1) 색이나 형태가 같은 다른 물체가 있어도 문장이 가리킨 물체를 고른다. (2) 멀리 작게 보이는 목표를, 장면 외관이 달라도, 목표로 알아보고 접근을 시작한다. (3) 찾고, 다가가고, 위에서 맞추고, 내려가서 착륙한다.
- **숫자를 올리는 것이 목적이 아니다.** 이후 Blur·Occlusion·Drift의 영향을 따로 잴 수 있을 만큼 clean 과제가 안정적인지를 본다.

## Canonical mission

```
Find the blue landing pad and land on it.
```

- **입력은 그대로다.** Front RGB, Down RGB, 문장 하나. 목표의 좌표, 거리, 방위, 화면 속 위치, 가시성은 어떤 형태로도 모델에 들어가지 않는다(`policy_inputs`에 그런 인자가 없다).
- **출력도 그대로다.** 전진, 하강, yaw의 연속값 4-step chunk, 한 step만 실행. 착륙도 이 세 축으로 한다. episode 안에서 simulator의 자체 착륙 루틴(`land_async`)은 어디에서도 부르지 않는다.
- **문장이 끝을 정한다.** "Approach …"와 "Find …"는 물체 근처 공중에서 멈추는 것이고, "… and land on it"과 "Land on …"은 pad 위에 내려앉는 것이다. 같은 장면, 같은 시작에서 문장만 바꿔 둘 다 학습하고 평가한다.

## 이 단계에서 프롬프트에 더한 조건

계획을 그대로 실행하면 결과를 해석할 수 없게 되는 지점들이 있어, 학습 전에 다음을 정했다.

| 무엇 | 왜 |
|---|---|
| Pad를 14×14m, 높이 2.5m의 단(platform)으로 만들고 위에 흰 H와 가는 격자를 올림 | 바닥에 붙은 평평한 pad는 8m 높이에서 80m 밖이면 세로 2px 미만이다. 높이 2.5m면 옆면이 보여 같은 거리의 cone과 비슷한 화면 면적이 된다. H는 정렬의 단서, 격자는 마지막 1m의 높이 단서다(격자는 첫 pilot 뒤에 더했다) |
| 착륙 판정을 collision topic으로 | 실행 중에 만든 mesh에도 충돌이 있고 topic이 닿은 물체의 이름을 준다는 것을 먼저 확인했다. pad(또는 그 H)를 위에서 닿으면 touchdown, 그 밖의 모든 접촉은 collision이다 |
| 빨간 pad에도 착륙 문장을 학습 | 파란 pad에만 착륙을 가르치면 "land"가 곧 "파란 pad"가 된다 |
| Pad에 대한 approach 문장을 학습, 같은 시작의 twin 포함 | pad만 보면 내려가는 것을 막는다. 착륙 시작 20개는 같은 자세에서 approach 문장으로 한 번 더 비행한다 |
| 물체마다 색 하나, 형태 하나. 파란 pad·cube·cylinder·cone은 색이 같고, pad 둘·cube 둘·cylinder 둘은 형태가 같음 | 어느 속성을 놓쳤는지(색 혼동인지 형태 혼동인지)를 끝난 위치로 가를 수 있다 |
| 모든 물체가 layout마다 자리를 바꿈. 두 물체가 42m 떨어진 "pair" 자리 3쌍 | 자리로 맞히는 것을 막고, 두 후보가 한 화면에 들어오는 시작을 만든다. 서로 다른 물체의 성공 반경(15m)은 겹치지 않는다 |
| yellow pyramid는 어떤 학습 장면에도 두지 않음 | Gen-v1부터 "처음 보는 물체"였다. distractor로라도 학습 장면에 넣으면 G2·G4의 뜻이 바뀐다. 평가 장면에만 distractor로 선다 |
| 기존 평가 시작(G0–G4, P, S, L)과 6m 이상 분리, Yard·Depot 학습 0회 | L과 Yard는 이미 결과를 보고 설계에 쓴 set이다. 궤적도 시작점도 학습에 넣지 않는다 |
| "보임"과 "알아봄"을 따로 측정 | 장거리 평가의 교훈이다. 시야에 있다는 것은 simulator의 말이고, 모델이 그쪽으로 가기 시작했는지는 행동에서 읽는다(Grounding transition) |
| Checkpoint 선택 규칙과 학습 길이를 학습 전에 고정 | 결과를 보고 고르지 않기 위해서다 |
| Teacher가 simulator에서 끝내지 못한 평가 episode는 모든 모델의 점수에서 제외 | 풀 수 없는 시작으로 모델을 재지 않는다. 규칙을 비행 전에 정했다 |

## 물체와 distractor

[configs/targets/objects.json](../configs/targets/objects.json). 새 장면의 모든 물체는 이 한 파일에서 온다.

| 물체 | 형태 | 크기 (m) | 색 | 비고 |
|---|---|---|---|---|
| blue landing pad | pad | 14 × 14 × 2.5 | 파랑 | 착륙 대상. 위에 흰 H |
| red landing pad | pad | 14 × 14 × 2.5 | 빨강 | 형태가 같은 distractor. 착륙 대상이기도 함 |
| blue cube | cube | 8 × 8 × 8 | 파랑(pad와 같은 값) | 색이 같은 distractor |
| red cube | cube | 8 × 8 × 8 | 빨강 | |
| blue cylinder | cylinder | 8 × 8 × 9 | 파랑(pad와 같은 값) | 색이 같은 distractor |
| green cylinder | cylinder | 8 × 8 × 9 | 초록 | |
| orange ball | ball | 10 × 10 × 10 | 주황 | simulator에 들어 있는 asset |
| blue cone | cone | 10 × 10 × 10 | 청록에 가까운 파랑 | Blocks의 cone을 본뜬 mesh |
| yellow pyramid | pyramid | 9 × 9 × 9 | 노랑 | 평가 장면에만. 학습에 없음 |

## 장면

Project AirSim 1.0.1에는 환경이 하나뿐이라, Yard와 같은 방식으로 Blocks 구조물이 보이지 않는 먼 곳에 실행 중에 장면을 만든다. 모두 **같은 simulator, 같은 runtime 안의 다른 장면**이다. 새 simulator domain이 아니다.

| 장면 | 쓰임 | 바닥 | 시각 | 구조물 | 물체 자리 |
|---|---|---|---|---|---|
| Blocks | 학습(기존 Gen-v2 data + pad layout c·d) | 원래 맵 | 기본 | 블록 탑, 벽 4개 | cube·cylinder가 서던 두 자리에 pad 둘 |
| Field | 학습 a–d, 평가 e | 풀(a, b, e) / 마른 풀(c, d) | 10:00 / 15:30 | 생울타리, 헛간, silo, 벽, 탑 | 8곳(pair 3쌍) |
| Lot | 학습 a–d, 평가 e | 아스팔트(a, b, e) / 콘크리트(c, d) | 12:30 / 17:00 | 창고, 트럭, 차단벽, 탱크, 기둥 | 8곳(pair 3쌍), Field와 다른 배치 |
| **Depot (Scene C)** | **평가만. 학습·검증 비행 0회** | 자갈 | 08:00 | 컨테이너 열, crane, 탱크, 벽, gantry | 9곳(pair 3쌍), yellow pyramid 포함 |
| Yard | 기존 평가(G3, G4, L)만 | 모래 | 19:00 | 그대로 | 그대로 |

- **자리 바꿈:** Field와 Lot에서 blue pad는 학습 layout 4개에서 서로 다른 자리 4곳에 선다. 평가용 layout e에서는 학습에서 한 번도 서지 않은 자리에 선다.
- **Depot의 layout:** c1↔c2는 blue pad와 red pad만, c3↔c4는 blue pad와 blue cylinder만, c5↔c6는 blue pad와 blue cube만 자리를 바꾼 쌍이다. 같은 기체 자세에서 두 layout을 모두 비행해 자리로 맞히는지를 본다.
- **외관을 일부러 다르게:** 학습의 시각은 네 가지(10:00, 12:30, 15:30, 17:00), 바닥은 네 가지다. Depot의 바닥과 시각(08:00)은 그 어느 것과도 다르고, Yard(모래, 19:00)와도 다르다. Yard의 외관을 학습 장면에 베끼지 않았다.

## Simulator에서의 착륙

학습 전에 probe로 확인한 것(`outputs/visual_search/probe/`, Git에는 없음):

- **실행 중에 만든 mesh에 내려앉을 수 있다.** pad 중앙 위 8m에서 tick 실행기로 내려가면 pad 윗면에서 멈춘다.
- **닿으면 기체가 그 자리에 선다.** `landed_state`가 1이 되고 전진·하강 명령으로는 움직이지 않는다. 상승 명령이 0.6m쯤 쌓이면 다시 뜬다(나중에 모델의 비행에서 확인).
- **collision topic이 닿은 물체의 이름을 준다.** 첫 event의 기체 위치를 touchdown 위치로 쓴다. event는 touchdown마다 한 번 온다(첫 probe에서는 H 표시 위에서 계속 왔지만, 격자를 올린 뒤의 실비행에서는 한 번이었다). 그래서 "지금 pad 위에 서 있는가"는 event가 아니라 기체의 높이와 수직 속도로 읽는다.
- **쉬는 높이:** 기체 원점은 서 있는 면보다 0.19m(지면)–0.16m(H 위) 높다. 높이는 처음 쉬던 자리에서 재므로 pad 위에 서면 높이가 2.5m로 읽힌다.
- **Front 화면의 가운데 띠는 기체의 팔과 프로펠러가 가린다.** 수평선 위아래 약 7°가 가려지고 가운데에 약 16° 너비의 틈만 열려 있다. 5–9m 높이에서 55m 이상 떨어진 물체는 거의 이 띠 안에 있어서, 회전 중에는 틈을 지나는 1–2 판단 동안만 온전히 보인다. 먼 목표를 "지나치는" 실패의 물리적 배경이다.
- **화면 속 크기:** 68m에서 pad의 bounding box는 256px 화면에서 가로 36px, 세로 7.6px다.

### 착륙 성공의 정의

[configs/targets/landing_pads.json](../configs/targets/landing_pads.json). 기존의 "15m 안에서 멈춤"은 착륙 성공에 쓰지 않는다. 숫자는 모두 pad나 기체의 치수, 또는 위 측정에서 나온다.

| 조건 | 값 | 근거 |
|---|---|---|
| 맞는 pad에 touchdown | collision topic이 그 pad(또는 H)를 가리키고 기체가 pad 윗면보다 위에 있음 | 옆면에 부딪힌 것은 collision |
| 착륙 구역 안 | pad 중심에서 가로·세로 각 6.6m 이내 | pad 반폭 7.0m − 기체 반폭 0.4m(rotor 0.253m + 프로펠러): 기체 전체가 pad 위 |
| 수직 속도 | 0.75m/s 이하 | 행동 범위가 허용하는 하강 1.0m/s와 teacher의 마지막 하강 0.5m/s의 중간 |
| 수평 속도 | 0.5m/s 이하 | 행동 범위가 허용하는 전진 2.0m/s의 1/4 |
| 스스로 정지 | 0에 가까운 행동 4 판단 연속(기존 정지 기준) | touchdown 뒤 20 판단 안에 멈추지 않으면 실패로 끝냄 |
| 멈췄을 때 pad 위에 서 있음 | touchdown 때와 높이 3cm 이내, 수직 속도 0.05m/s 이하 | 떴다가 공중에서 멈춘 것은 착륙이 아님 |
| 그 밖의 접촉 없음 | | |

Approach 과제의 성공은 그대로다: 스스로 멈춤 + 목표 15m 이내 + 충돌 없음. 여기에 "공중에 있고 pad 위로 내려오지 않았음"이 더해진다(approach를 시켰는데 pad에 내려앉거나, pad 위에서 1m 이상 내려오면 실패).

### 단계

episode마다 `TARGET_ACQUIRED`(Front나 Down에 보임) → `CORRECT_TARGET`(끝난 곳이 지시한 물체) → `APPROACHED`(20m 이내) → `DOWN_CAMERA_ALIGNED`(pad 위에서 중심 3.5m 이내, pad 폭의 1/4) → `DESCENT_STARTED`(pad 위에서 1m 이상 내려감) → `TOUCHDOWN` → `SELF_STOP` → `LAND_SUCCESS`를 기록한다.

## Teacher

기존 teacher(`SearchTeacher`)에 착륙을 더했다. 목표의 위치를 알지만 label은 여전히 두 화면에서 정해지는 것만 쓴다(기억이 필요한 규칙 없음). 목표를 찾기 전까지는 과제와 무관하게 똑같이 행동한다.

| 화면 | Approach 문장 | Land 문장 |
|---|---|---|
| 목표가 안 보임 | 오른쪽으로 회전(앞이 가까이 막혀 있으면 상승) | 같음 |
| 보이고 12m보다 멂 | 목표 쪽으로 돌고 전진 | 같음 |
| 보이고 12m 이내 | **정지**(0 행동) | **계속 전진**. pad 위에서는 거리의 1/6로 속도를 줄임 |
| pad 중심 1.75m 이내(pad 폭의 1/8) | — | **하강** 0.5m/tick, 윗면 2.5m 위부터 0.25m/tick |
| pad 위에서 이미 낮음(윗면 3m 이내, 중심 3.5m 이내) | **정지** | **하강 유지**(첫 pilot 뒤에 더한 규칙) |
| touchdown | — | **정지**(0 행동) |

높이 올라온 상태(pad 위 7m 초과)에서 25m 안에 들어오면 전진하면서 0.25m/tick씩 내려와, pad가 두 화면 중 하나에 계속 남게 한다.

## 평가 앞의 gate

156개 test 시작은 아래 세 단계를 차례로 통과한 뒤에만 비행한다. 기준은 `configs/visual_search.json`의 `gen_v3.gates`에 있고, 판정 대상 checkpoint가 생기기 전에 commit했다. 판정은 `scripts/gate_gen_v3.py`가 하고, 평가 체인은 그 종료 코드를 보고 스스로 멈춘다.

| Gate | 무엇으로 | 통과 기준 |
|---|---|---|
| Pilot | pilot episode만으로 학습한 checkpoint, 같은 자세 4곳 × 두 문장 | land 비행 4회 중 3회 이상 지시한 pad에 touchdown, approach 비행의 touchdown 0회, approach 3회 이상 공중 정지 성공, 문장만 바꾼 probe 세 줄 모두 0.7 이상, 실행 오류 0 |
| Smoke | 최종 checkpoint, 종류별 첫 검증 시작 10개 | 성공 7/10 이상, 착륙 성공 1 이상, 원거리 성공 1 이상, 충돌 1 이하, 실행 오류 0 |
| Representative | 최종 checkpoint, 검증 시작 전부(smoke 10개 포함) | 성공률 80% 이상, 착륙 성공률 75% 이상, 시야에 들어온 목표로 접근 시작 90% 이상, 문장대로 끝남 90% 이상, 다른 물체에서 끝남 2 이하, 충돌 1 이하, 실행 오류 0 |

- **Smoke와 representative는 test set의 일부가 아니다.** 학습 계획의 검증 split 시작(학습 장면, 학습에 쓰지 않은 seed)이다. gate에서 떨어져 학습으로 돌아가더라도 156개는 개발에 노출되지 않는다.
- **Teacher가 simulator에서 끝내지 못한 시작과, 강제로 밀어 넣는 episode는 gate에서 뺀다.**

## Pilot: 문장이 비행의 끝을 정하는가

본 수집 전에 Field의 layout 하나에서 작은 set을 기록해, 그것만으로 학습한 checkpoint를 같은 자세에서 두 문장으로 비행시켰다. 두 번 했다.

### 첫 pilot — 통과하지 못함

16 episode(8개 자세 × land·approach), 652 sample, 600 update(21분). train L1 0.88 → 0.040, val 0.239 → 0.049, 재로드 차이 0.0.

| 문장 | 4회 비행 | 결과 |
|---|---|---|
| Find … and land on it. | 4회 모두 pad를 찾아 위에서 맞추고 하강을 시작 | touchdown 2/4, 성공 1/4. 2회는 pad 0.3m 위에서 스스로 정지, 1회는 touchdown 뒤 정지 판정을 못 받음 |
| Approach … | 2회는 약 10m에서 정지 | 2회는 정지 경계를 지나친 뒤 착륙 동작을 따라감(1회 touchdown, 1회는 pad 표면 바로 위에서 정지) |

모델이 아니라 장면·teacher·데이터에 원인이 셋 있었다.

1. **Pad 윗면에 높이 단서가 없었다.** 균일한 파랑이라 Down 화면은 0.3m 위에서도 착지 상태에서도 같았다. → 윗면 전체에 0.4m 간격, 3cm 폭의 흰 격자를 올렸다. 하강하는 동안 칸이 화면에서 계속 커진다.
2. **Teacher의 label이 표면 바로 위에서 뒤집혔다.** 하강은 중심 1.75m 안에서만 하는 규칙이라, 하강 중 몇십 cm만 밀려나도 label이 "돌아서 다시 맞춤"이 됐다. 모델은 그 사이에서 하강 명령을 0으로 줄였고, 그것이 정지로 판정됐다(`pilot-9000`). → pad 위에서 이미 낮게 내려왔으면(윗면 3m 이내, 중심 3.5m 이내) 계속 하강한다. 여전히 화면에서 정해지는 규칙이다.
3. **Approach 문장을 정지 거리 안쪽에서 본 적이 없었다.** approach 비행은 12m에서 멈추므로 그보다 가까운 장면은 전부 land 문장의 것이었다. 경계를 조금 지나친 approach 비행은 본 적 없는 상태에 들어가 착륙을 따라갔다. → approach episode에서 teacher가 멈춘 뒤 기체를 목표 쪽으로(일부는 아래로도) 강제로 밀고, 거기서 다시 "정지"를 label로 한다. 민 구간은 비행만 하고 label로 쓰지 않는다.

같이 고친 것: approach를 시켰는데 pad 위로 내려와 표면 바로 위에서 멈춘 비행이 "공중 정지 성공"으로 잡히던 것을 막았다(pad 위에서 1m 이상 내려오면 approach 실패). touchdown 뒤 정지를 기다리는 시간을 8 판단에서 20 판단으로 늘렸다(멈춘 기체에서 yaw 출력이 기준 0.034rad 근처에서 흔들려 4연속을 채우지 못한 경우가 있었다).

Gate의 probe 줄도 이때 바로잡았다. 처음에는 "착륙 중의 모든 근접 장면에 approach 문장을 주면 멈추는가"였는데, 이는 approach 비행이 갈 일이 없는 장면(착륙 하강 한가운데)까지 섞은 것이었다. 결정이 실제로 내려지는 장면, 즉 approach가 멈춘 장면에서 두 문장을 비교하도록 바꿨다. 원래 숫자도 계속 기록한다(0.20 → 0.68).

### 두 번째 pilot — 통과

22 episode(위 16 + 밀어 넣은 approach 6), 700 update(25분). 재로드 차이 0.0.

| 검사 | 값 | 기준 |
|---|---:|---:|
| land 비행 중 지시한 pad에 touchdown | 4/4 | 3 이상 |
| approach 비행 중 touchdown | 0/4 | 0 |
| approach 비행 중 공중 정지 성공 | 4/4 (9.5–11.5m) | 3 이상 |
| 착륙 근접 장면에서 land 문장 → 계속 진행 | 0.98 | 0.7 이상 |
| approach가 멈춘 장면에서 approach 문장 → 정지 | 1.00 | 0.7 이상 |
| 같은 장면에서 land 문장 → 계속 진행 | 1.00 | 0.7 이상 |

- **같은 자세, 다른 문장, 다른 끝.** land 문장은 4회 모두 내려앉았고 approach 문장은 4회 모두 공중에서 멈췄다.
- **착륙 완전 성공은 3/4다.** 나머지 1회는 touchdown하고 하강 명령도 0으로 내렸지만, yaw 출력이 0.065–0.09rad에 머물러 정지 기준(0.034rad)을 20 판단 동안 넘지 못했다. 본 학습 뒤에도 남는지 볼 항목이다.
- **pilot은 22 episode의 과적합 확인이다.** 같은 자세에서 비행했으므로 일반화에 대해서는 아무것도 말하지 않는다.

## 평가 set (학습 전에 고정)

[configs/gen_v3_test_spawns.json](../configs/gen_v3_test_spawns.json). 156개 시작, seed 8000–8150. `scripts/plan_generalization.py v3-test`가 한 번 쓰고 다시 쓰지 않는다. 학습 데이터를 계획하기 전에 commit했다(`9b1c42a`). teacher는 simulator에서 156개를 모두 끝냈다(156/156).

| Set | 수 | 장면 | 무엇을 보는가 |
|---|---:|---|---|
| Q1 | 8 | Depot | "Find the blue landing pad." 기본 |
| Q2 | 8 | Depot | 첫 화면에 red pad가 함께 보임(나란히 / 한 줄로 / pad는 가장자리·red pad는 가운데 / pad는 시야 밖) |
| Q3 | 8 | Depot | 같은 구성으로 blue cylinder 또는 blue cube |
| Q4 | 16 | Depot | Q2·Q3의 기체 자세 그대로, 두 물체의 자리를 바꾼 layout |
| Q5 | 16 | Depot | 자세 4곳 × 문장 4개(blue pad, red pad, 색이 같은 다른 물체, 색도 형태도 다른 물체) |
| Q6 | 18 | Depot | 자세 6곳 × 문장 3개: land / "Approach …"(학습한 말) / "Find and approach …"(안 쓴 말) |
| Q7 | 18 | Depot | 40–55m, 55–70m, 70–90m 각 6. approach. blue pad 9, red pad 3, 그 밖 6 |
| Q8 | 24 | Depot | canonical mission. near(14–24m) 8, medium(24–44m) 8, far(55–90m) 8 |
| Q8R | 6 | Depot | 같은 문장을 red pad로 |
| Q10 | 18 | Depot | Q8의 시작 6개 × 학습에 안 쓴 착륙 문장 3개 |
| QS | 16 | Field e, Lot e | 학습 장면의 미사용 layout(pad가 학습에서 서지 않던 자리). approach 8, 착륙 8 |

- **문장 split:** 학습은 "Find X." "Approach X." "Find X and land on it." "Land on X."만 쓴다. "Locate X." "Search for X." "Find and approach X."(기존 P set과 같은 held-out), "Locate X and land on it." "Search for X, approach it, and land." "Find the blue rectangular landing pad and land on it."는 평가에만 쓴다.
- **첫 pilot 뒤에도 시작은 그대로다.** teacher의 착륙 규칙이 바뀌어 7개 episode에 저장된 dry-run 기록(판단 수, 상태 수)만 갱신했다. 시작 위치·방향·높이·문장·layout은 156개 모두 같다(스크립트로 비교).

## Dataset

Gen-v2의 445 episode(Gen-v1 292 + Gen-v2 보강 153)는 그대로 쓰고, 그 위에 두 차례 더했다. 모두 teacher가 simulator에서 비행한 기록이고, test set의 장면(Depot)·layout·시작·문장은 들어 있지 않다(`plan_generalization.py v3-check`).

### 첫 번째 보강 (`generalization_v3_added`)

계획 415, 기록 414(teacher가 320 판단 안에 못 끝낸 1개 제외). train 375 / val 39, 23,667 sample. simulator 3개로 110분.

| 범주 | 종류 | Episode (train + val) |
|---|---|---:|
| A. Distractor 속 grounding | 첫 화면에 다른 물체가 함께(나란히·한 줄), 목표는 가장자리·다른 물체는 가운데, 목표는 시야 밖·다른 물체는 시야 안 | 83 |
| B. 먼 목표 | 55–90m, 보임 / 안 보임 | 92 |
| C. 착륙 | 보임, 가장자리, 안 보임, 12–18m 근접, 높은 시작 | 128 |
| C′. 착륙 시작의 approach twin, 정지 뒤 안쪽으로 밀린 approach | 문장만 다름 / 정지 거리 안쪽에서 "정지" | 22 + 33 |
| D. 먼 시작에서 착륙 | 55–90m, 보임 / 안 보임 | 56 |

- **장면:** Field 172, Lot 184, Blocks(pad layout) 58.
- **과제:** approach 230, land 184(blue pad 2 : red pad 1). teacher의 착륙 184회는 모두 착륙 규칙을 만족했다(touchdown 오차 중앙값 0.75m, 최대 1.2m, 수직 속도 중앙값 0.37m/s).
- **시작 거리:** 30m 미만 161, 30–60m 134, 60m 이상 119.
- **화면 속 크기:** 목표가 보이는 frame을 bounding box가 차지하는 화면 비율로 삼등분했다. 작음은 1.6% 미만, 큼은 9.0% 이상. 작게 보이는 frame은 학습에서 두 배 자주 뽑는다. 크기는 기록에만 있고 모델 입력이 아니다. Gen-v1·v2의 frame에는 이 기록이 없다.
- **합친 dataset(`generalization_v3`):** 859 episode, 40,432 sample(train 36,465 / val 3,967). train·val 사이에 겹치는 episode, 폴더, seed, 시작 0.

## 첫 번째 학습과 smoke gate

구조와 recipe는 Gen-v2와 같다: OpenVLA-7B NF4 + 고정한 AeroVLA adapter + 새 LoRA r16 + L1 linear head, chunk 4, 실행 1, FiLM OFF, proprio OFF, batch 1 × accumulation 4, 처음부터 학습. tanh head는 쓰지 않았다.

학습 전에 고정한 것: 최대 8,000 update, 500 update마다 검증 frame 400개로 평가, 4회 연속 개선 없으면 중단, 작게 보이는 frame 2배, checkpoint 규칙(검증 L1 최저, 그 2% 안에서는 행동 종류 정확도가 높은 쪽).

| 항목 | 값 |
|---|---:|
| Update | 8,000(조기 종료 없음, 마지막까지 검증 loss가 내려감) |
| 시간 / peak VRAM | 262분 / 9.82GiB |
| 고른 checkpoint | update 7,999 |
| Train L1(EMA) / Val L1 | 0.024 / 0.066 |
| 행동 종류 정확도(검증) | 전진·회전 0.90, 정지·하강 0.97 |
| Gen-v2 검증 frame에서 | 0.071 (Gen-v2 자신은 0.071) |
| 재로드 차이 | 0.0 |
| 문장만 바꾼 probe | approach가 멈춘 장면: approach 문장 → 정지 0.98, land 문장 → 계속 1.00. 다른 pad를 지시하면 계속 전진 0.33 |

**Smoke gate: 통과하지 못함(6/10, 기준 7). Full evaluation은 시작하지 않았다.**

원인을 보려고 검증 시작 나머지를 진단용으로 비행했다(gate 통과가 아니다. test set도 아니다).

| 검증 시작 36개 | 값 | Representative 기준 |
|---|---:|---:|
| 성공 | 28/36 (0.78) | 0.80 이상 |
| 착륙 성공 | 13/18 (0.72) | 0.75 이상 |
| 시야에 들어온 목표로 접근 시작 | 0.94 | 0.90 이상 |
| 문장대로 끝남 | 0.92 | 0.90 이상 |
| 다른 물체에서 끝남 | 4 | 2 이하 |
| 충돌 | 0 | 1 이하 |

| 실패 8회 | 수 | 내용 |
|---|---:|---|
| 색이 같은 다른 물체에서 끝남 | 4 | blue cone → blue pad, blue cylinder → blue pad, blue pad 착륙 → blue cylinder 2회 |
| touchdown 뒤 정지 못 함 | 3 | 착지한 화면에서 약한 상승 명령(−0.1~−0.19m)이 나와 떴다 내렸다를 반복 |
| 먼 시작 timeout | 1 | blue cube, 60m |

- **찾기, 접근 시작, 문장에 따른 끝은 기준을 넘었다.** 착륙도 18회 중 16회는 지시한 pad에 touchdown했다.
- **병목 하나는 "색이 같고 형태가 다른 물체"다.** 원인은 데이터에서 셀 수 있다. 새 장면에서 "보이면 접근" frame의 정답은 blue pad가 5,320개, blue cylinder·cube는 각 1,400개 안팎, blue cone은 606개였다. 목표와 색이 같은 다른 물체가 목표보다 가까웠던 시작은 240개 중 5개였다. 모델은 "파란 것이 보이면 가서 12m 앞에서 멈춘다"에 가깝게 배웠다.
- **다른 하나는 착지한 뒤다.** teacher는 중심 1.2m 안에 내려앉았는데 모델은 0.9–5.9m에 내려앉았다. 그 위치의 착지 화면은 학습에 없었다.

### 두 번째 보강 (`generalization_v3b_added`)

검증 시작의 실패만 보고 만들었다. test set과 Depot은 보지 않았고 gate의 검증 시작도 바꾸지 않았다. 계획 199(train 179 / val 20).

| 종류 | Episode (train) | 내용 |
|---|---:|---|
| Query twin | 32 자세 + twin 67 | 같은 자세에서 명사만 바꿔, 색이나 형태가 같은 다른 물체로 각각 비행. 두 개 이상 갈 수 있는 자세만 씀 |
| 같은 색 물체가 더 가까움(approach) | 24 | 목표보다 8m 이상 가까운 같은 색 물체가 첫 화면에 있음 |
| 같은 색 물체를 지나 착륙 | 20 + 12 | pad는 시야 밖이고 같은 색 물체가 첫 화면에 / pad는 보이고 같은 색 물체가 더 가까움 |
| 중심에서 벗어난 착륙과 hop | 24 | 하강 마지막 3m에서 앞으로 밀어 중심 2–3m 밖에 내려앉게 하고, touchdown 뒤 0.3–0.7m 띄운 다음 다시 내려앉아 멈추게 함. 민 구간과 띄운 구간은 label이 아님 |

이때 같이 바로잡은 것: simulator는 touchdown마다 접촉 event를 한 번만 보낸다(쉬는 동안 계속 오지 않는다). "지금 pad 위에 서 있는가"는 기체의 높이(touchdown 때와 3cm 이내)와 수직 속도(0.05m/s 이하)로 읽는다. 착륙 성공에는 "멈췄을 때 pad 위에 서 있음"이 더해졌다(떴다가 공중에서 멈춘 것은 착륙이 아니다).

기록 193(teacher가 못 끝낸 6개 제외), 11,964 sample. 합친 dataset(`generalization_v3b`)은 1,052 episode, 52,396 sample(train 47,203 / val 5,193)이다.

## 두 번째 학습과 gate

같은 recipe로 처음부터 다시 학습했다. 첫 checkpoint(`generalization_v3`)는 그대로 있다. 학습 전에 고정한 것은 첫 학습과 같고 update 상한만 12,000이다(첫 학습이 8,000에서도 내려가고 있었다).

| 항목 | 첫 번째 (`generalization_v3`) | 두 번째 (`generalization_v3b`) |
|---|---:|---:|
| Episode / train sample | 859 / 36,465 | 1,052 / 47,203 |
| Update | 8,000 | 12,000(조기 종료 없음) |
| 시간 / peak VRAM | 262분 / 9.82GiB | 406분 / 9.83GiB |
| 고른 checkpoint | update 7,999 | update 10,500 |
| Val L1(두 번째 dataset의 검증 frame 400개) | 0.070 | **0.056** |
| 같은 frame에서 Gen-v2 | 0.206 | |
| 행동 종류 정확도: 전진·회전 / 정지·하강 | 0.89 / 0.93 | 0.92 / 0.93 |
| Gen-v2 검증 frame에서 | 0.071 | 0.076 (Gen-v2 자신은 0.071) |
| 재로드 차이 | 0.0 | 0.0 |

| Gate | 첫 번째 | 두 번째 |
|---|---|---|
| Smoke (10) | 6/10, 통과 못 함 | **7/10, 통과** |
| Representative (36) | 진단으로만 비행: 28/36 | **27/36, 통과 못 함** |

**Representative gate를 통과하지 못했으므로 156개 test 시작은 한 번도 비행하지 않았다.** 아래 숫자는 모두 검증 시작(학습 장면, 학습에 안 쓴 seed)에서의 것이고, 새 held-out 추정치가 아니다.

| 검증 시작 36개 | 첫 번째 | 두 번째 | 기준 |
|---|---:|---:|---:|
| 성공 | 28/36 (0.78) | 27/36 (0.75) | 0.80 이상 |
| 착륙 성공 | 13/18 (0.72) | 14/18 (0.78) | 0.75 이상 |
| 시야에 들어온 목표로 접근 시작 | 0.94 | 0.91 | 0.90 이상 |
| 문장대로 끝남 | 0.92 | 0.83 | 0.90 이상 |
| 다른 물체에서 끝남 | 4 | 2 | 2 이하 |
| 충돌 | 0 | 0 | 1 이하 |

## 검증 시작에서 본 것

### 문장이 끝을 정한다

- **approach를 시켰는데 내려앉은 비행은 두 checkpoint 모두 0/18이다.** 착륙 시작과 같은 자세의 approach twin 2개는 둘 다 12m 부근 공중에서 멈췄다.
- **land를 시켰는데 공중에서 멈춘 비행**은 첫 번째 2/18, 두 번째 1/18이고, 모두 다른 파란 물체 앞에서였다(아래).
- 같은 frame에서 문장만 바꾸면(두 번째 checkpoint): approach가 멈춘 장면에서 approach 문장 → 정지 0.96, land 문장 → 계속 진행 1.00.

### 정확한 grounding

| 검증 시작 | 첫 번째 | 두 번째 |
|---|---:|---:|
| 첫 화면에 다른 물체가 함께 있는 시작 8개(나란히, 한 줄, 가운데의 다른 물체, 시야 밖 목표) | 8/8 | 8/8 |
| 44m 이하 전체 | 19/22 | 19/22 |
| 파랑이 아닌 목표(red pad, red cube, green cylinder, orange ball) 전체 | 11/12 | **12/12** |
| 파란 목표(pad, cube, cylinder, cone) 전체 | 17/24 | 15/24 |

- **색은 구분한다.** red pad와 blue pad가 한 화면에 있을 때 지시한 쪽으로 갔고, 다른 pad를 지시하면 보이는 pad로 계속 전진하는 비율은 0.28이다(pilot 0.85).
- **실패는 전부 파란 물체 넷 사이에서 난다.** 새 장면에는 파란 물체가 넷(pad, cube, cylinder, cone)이고 빨강은 둘, 나머지는 하나씩이다. 파랑이 아닌 목표는 거리와 관계없이 12/12다.
- **두 번째 보강은 "다른 파란 물체로 감"을 줄였다(4 → 2).** 대신 먼 파란 목표 앞에서 접근을 시작했다가 그만두는 비행이 늘었다(1 → 5). 합계는 그대로다.

### 먼 목표

| 시작 54m 이상 (14개) | 첫 번째 | 두 번째 |
|---|---:|---:|
| 전체 | 9/14 | 8/14 |
| 파랑이 아닌 목표 | 4/5 | 5/5 |
| 파란 목표 | 5/9 | 3/9 |

- **외관이 다른 장면의 먼 목표는, 헷갈릴 물체가 없으면 된다.** Field·Lot의 서로 다른 바닥과 시각에서 56–70m의 red pad·red cube는 두 번째 checkpoint에서 5/5다.
- **파란 목표는 멀면 확정하지 못한다.** 두 번째 checkpoint의 실패 6회 중 3회는 처음부터 정면에 보이는 목표로 전진을 시작했다가 몇 판단 뒤 회전으로 바뀌었다(`val-5405`: 전진 1.0, 1.0, 0, 1.0, 0, 0.7, 0 …). 한 번 틀어지면 목표는 기체의 팔 뒤로 들어가고 탐색으로 돌아간다.
- **그 배경은 teacher 기록에서 잴 수 있다.** 55m 이상에서 목표가 기수 8° 안에 있으면 100% 보이지만, 8–25° 벗어나면 31–35%만 보인다(20–40m에서는 62%). 먼 목표는 가운데 틈으로만 보인다. teacher는 항상 정확히 정면으로 접근하므로, 조금 틀어진 뒤의 장면은 학습에 거의 없다(55m 이상 3,570 frame 중 332개).

### 착륙

| 착륙 검증 시작 18개 | 첫 번째 | 두 번째 |
|---|---:|---:|
| 지시한 pad에 touchdown | 16/18 | 16/18 |
| 착륙 성공(규칙 전체) | 13/18 | 14/18 |
| pad 위에 서 있지만 0 행동을 내지 못함 | 3 | 2 |
| 다른 파란 물체로 감 | 2 | 1 |
| 먼 시작에서 접근하지 못함 | 0 | 1 |
| Touchdown 오차 중앙값 / 최대 | 2.05m / 5.93m | 2.69m / 4.99m |
| Touchdown 직전 수직 속도 중앙값 | 0.58m/s | 0.59m/s |

- **찾고, 다가가고, 위에서 맞추고, 내려앉는 것까지는 된다.** touchdown은 모두 착륙 구역(중심 6.6m) 안이고 수직 속도는 기준(0.75m/s) 아래다. 충돌 0.
- **남은 것은 내려앉은 뒤의 0 행동이다.** 두 번째 checkpoint의 2회는 pad 위에 가만히 서서 하강 명령 0.04–0.19m를 계속 냈다(정지 기준 0.08m). 기체는 움직이지 않지만 "스스로 정지"로는 판정되지 않는다. 첫 번째 checkpoint에서는 반대로 약한 상승 명령이 나와 떴다 내렸다를 반복한 경우가 있었고, 두 번째 보강 뒤 그 형태는 나오지 않았다.
- **모델은 teacher보다 중심에서 멀리 내려앉는다**(teacher 중앙값 0.75m). 접근 마지막의 감속이 teacher만큼 정확하지 않다.

## Fresh Scene C 평가

**하지 않았다.** representative gate를 통과하지 못했기 때문이다. Depot의 156개 시작(Q1–Q10, QS)은 teacher만 비행했고(156/156, 착륙 62회 touchdown 오차 중앙값 0.67m, 최대 1.03m), 어떤 학습된 모델도 비행하지 않았다. 따라서 다음 질문에는 아직 답이 없다.

- 색·형태 distractor, 위치 swap, query swap에서의 정확도(Q1–Q5)
- 학습에 안 쓴 문장에서의 착륙(Q10)
- 처음 보는 장면에서 같은 행동이 유지되는지(Scene C 대 QS)
- Gen-v2와 Gen-v3의 같은 시작 비교

test set은 그대로 fresh하다. gate에서 본 것은 검증 시작뿐이다.

## 기존 평가 set (회귀)

두 번째 checkpoint로 G1, G3, P, L을 다시 비행했다. 이 set들은 Gen-v2 때 이미 결과를 본 것이고 L은 Gen-v3 설계의 근거였으므로, 새 추정치가 아니라 "잃은 것이 있는가"의 확인이다. 실행기와 판단 수 한도(240)는 Gen-v2 때와 같다.

| Set | Gen-v2 | Gen-v3 |
|---|---:|---:|
| G1 (본 맵, 처음 보는 시작) | 32/32 | 31/32 |
| G3 (Yard) | 18/20 | 20/20 |
| P (학습에 없던 문장) | 24/24 | 22/24 |
| L1 40–55m | 11/12 | 12/12 |
| L2 55–70m | 8/12 | 10/12 |
| L3 70–90m | 8/12 | 8/12 |
| 합계 | 101/112 | 103/112 |
| 충돌 | 0 | 0 |

| 장거리 set L | Gen-v2 | Gen-v3 |
|---|---:|---:|
| Blocks | 17/18 | 17/18 |
| Yard | 10/18 | 13/18 |
| 본 물체 | 20/24 | 23/24 |
| 그중 Yard | 8/12 | 11/12 |
| 처음 보는 물체(yellow pyramid) | 7/12 | 7/12 |
| 시야에 들어온 목표로 접근 시작 | 30/36 | 32/36 |

- **기존 능력은 유지됐다.** 탐색, 다시 찾기, 높이 전환, 접근, 정지, 다른 문장.
- **Yard의 먼 본 물체가 나아졌다(8/12 → 11/12).** Yard는 여전히 학습에 없다. 외관을 달리한 장면의 먼 목표 데이터가 한 일로 보이지만, L을 보고 설계한 것이므로 그만큼만 읽어야 한다.
- **처음 보는 물체는 그대로다(7/12).** Gen-v3는 물체 종류를 늘렸지만 yellow pyramid는 여전히 어디에도 없다.
- **새로 잃은 것:** Blocks의 blue cone 46m 시작 하나(G1 1회, 같은 시작의 다른 문장 2회). 정면에 보이는 cone을 두고 회전을 계속했다. 파란 물체에 대한 망설임이 파란 물체가 하나뿐인 기존 장면에도 번졌다.

## 한계

1. **Fresh held-out 결과가 없다.** 이 문서의 모델 숫자는 전부 검증 시작(학습 장면)이나 이미 본 set의 것이다. Scene C에서의 일반화, 문장 일반화, 정밀 grounding의 held-out 정확도는 측정하지 않았다.
2. **검증 시작 36개는 두 번 쓰였다.** 첫 checkpoint의 실패를 보고 두 번째 보강을 만들었으므로, 두 번째 checkpoint의 27/36은 그 36개에 대해 낙관적일 수 있는 숫자다. 그런데도 오르지 않았다.
3. **먼 거리의 같은 색 구분은 이 구성의 감지 한계에 가깝다.** 모델 입력에서 70m 밖의 pad는 약 25×6px, cube는 13×13px이고, 그나마 화면 가운데 16° 틈으로만 보인다. teacher는 정답을 알고 곧장 가지만, 화면에는 그만한 정보가 없을 수 있다.
4. **파란 물체를 넷 둔 것은 이 단계의 선택이다.** 계획의 distractor 목록에는 blue cone이 없었다. Gen-v1부터의 물체라 넣었고, 그것이 파란 쪽의 혼동을 키웠다.
5. **착륙의 "스스로 정지"는 엄격하다.** pad 위에 서서 하강 명령 0.1m를 내는 것은 기체를 움직이지 않지만 실패로 센다. touchdown 기준(16/18)과 규칙 전체(14/18)를 함께 봐야 한다.
6. **Touchdown은 teacher보다 부정확하다**(오차 중앙값 2.7m 대 0.75m). 착륙 구역(6.6m) 안이긴 하다.
7. **장면은 모두 같은 simulator 안에 실행 중에 만든 것이다.** 다른 simulator, 다른 렌더러, 실제 환경에 대한 결과가 아니다.
8. **FiLM과 proprio는 쓰지 않았다.** FiLM은 구현돼 있지 않다(vision 쪽 학습이 필요해 12GB 범위 밖으로 보류했던 것). proprio는 스위치만 있다.
9. **Gaussian Blur, Occlusion, Control Drift, 장애물 회피, recovery는 하지 않았다.**

## 다음 단계

**남은 병목 하나:** 54m 이상에서 색이 같은 물체 넷 중 지시한 것을 확정하는 것. 검증 실패 9회 중 6회다.

**가장 작은 다음 변경(권장):** 모델을 다시 학습하기 전에 canonical clean mission의 범위를 정한다.

| 선택 | 내용 | 비용 | 근거 |
|---|---|---|---|
| **A. 범위를 44m 이하로** | "Find the blue landing pad and land on it."의 clean baseline을 시작 44m 이하로 정하고, 먼 거리의 같은 색 구분은 한계로 기록 | 재학습 없음. 새 검증 시작으로 gate를 한 번 더 비행 | 44m 이하는 두 checkpoint 모두 19/22. Blur는 먼 거리의 작은 형태 구분을 가장 먼저 무너뜨리므로, 그 구간은 Failure-Aware 실험의 기준으로 쓰기 어렵다 |
| B. 탐색·접근 고도를 12–14m로 | 먼 목표가 기체의 팔 아래로 보이게 하는 teacher 변경, 원거리 재수집, 재학습 | 약 10시간 | 물리적 원인을 직접 다루지만 Gen-v2와 비행 방식이 달라진다 |
| C. 원거리 보강만 추가 | 55–90m의 query twin, 틀어진 뒤 되찾는 장면 | 약 8시간 | 가장 작은 데이터 변경. 두 번째 보강이 실패의 형태만 바꾼 전례가 있다 |

- **A를 고르면 그 안의 병목은 "내려앉은 뒤의 0 행동"이다**(44m 이하 착륙 12회 중 2회). 정의를 바꾸지 말고 두 숫자를 같이 보고하거나, 계획대로 고도·수직 속도만 주는 최소 proprio를 별도 실험으로 본다(조건이 맞는다: 목표 선택, 접근, 정렬, 하강까지는 되고 마지막만 남았다).
- **FiLM은 계획의 진입 조건(데이터 보강 뒤에도 목표 선택이 가장 큰 병목)에 해당하지만, 이 구성에서는 작은 변경이 아니다.**
- **어느 쪽이든 Blur는 아직 아니다.** gate를 통과한 clean baseline이 먼저다.

## Examples

검증 시작에서 두 번째 checkpoint를 `-Record`로 다시 비행한 것이다(test set이 아니다). 한 장은 2 판단이고, `failure_far_blue.gif`만 6 판단이다.

| 파일 | 내용 |
|---|---|
| [gen_v3/mission_land.gif](../outputs/examples/gen_v3/mission_land.gif) | 60m, red pad가 시야 밖 → 탐색 → 획득 → 접근 → Down 화면에서 정렬 → 하강 → touchdown → 정지 |
| [gen_v3/land.gif](../outputs/examples/gen_v3/land.gif), [gen_v3/approach.gif](../outputs/examples/gen_v3/approach.gif) | 같은 자세, 다른 문장: "Find the blue landing pad and land on it." 대 "Approach the blue landing pad." |
| [gen_v3/two_pads.gif](../outputs/examples/gen_v3/two_pads.gif) | red pad와 blue pad가 한 화면에 있는 시작에서 blue pad로 |
| [gen_v3/blue_cylinder.gif](../outputs/examples/gen_v3/blue_cylinder.gif) | 다른 물체를 지나 blue cylinder로 |
| [gen_v3/failure_far_blue.gif](../outputs/examples/gen_v3/failure_far_blue.gif) | 실패: 85m의 blue pad로 전진을 시작했다가 회전으로 바뀜 |
| [gen_v3/failure_no_stop.gif](../outputs/examples/gen_v3/failure_no_stop.gif) | 실패: touchdown 뒤 0 행동을 내지 못함 |
| [gen_v3/dataset_samples.jpg](../outputs/examples/gen_v3/dataset_samples.jpg) | 기록한 data의 frame 12개(위 Front, 아래 Down): 먼 pad, 두 pad, 착륙의 각 단계, approach의 정지 |
| [gen_v3/plan_depot.jpg](../outputs/examples/gen_v3/plan_depot.jpg), [gen_v3/plan_field.jpg](../outputs/examples/gen_v3/plan_field.jpg) | Depot의 test 시작(layout c1), Field의 학습 시작과 QS 시작 |

동결한 표: [outputs/generalization/gen_v3_dev/](../outputs/generalization/gen_v3_dev/)(`summary.json`, `episodes.csv`, `validation.csv`, `regression.csv`, `long_range.csv`, `failures/`에 Gen-v3의 실패 18회 각각의 기록과 그림). Gen-v2의 파일은 [gen_v2_frozen.json](../outputs/generalization/gen_v2_frozen.json)의 hash와 같다.

## 재현

```powershell
# 평가 set(있으면 거부) → pilot → 수집 → 학습 → gate. 각 단계의 체인은 outputs/visual_search/gen_v3*_master.ps1에 있다(Git에는 없음)
.\assets\projectairsim-env\Scripts\python.exe scripts\plan_generalization.py v3-test
.\assets\projectairsim-env\Scripts\python.exe scripts\plan_generalization.py v3-pilot --output datasets/projectairsim_visual_search/gen_v3_pilot2
.\assets\projectairsim-env\Scripts\python.exe scripts\plan_generalization.py v3-train --output datasets/projectairsim_visual_search/generalization_v3_added
.\assets\projectairsim-env\Scripts\python.exe scripts\plan_generalization.py v3-train --recipe round_b --output datasets/projectairsim_visual_search/generalization_v3b_added
.\assets\projectairsim-env\Scripts\python.exe scripts\plan_generalization.py v3-check --plan datasets/projectairsim_visual_search/generalization_v3b_added/plan.json
.\scripts\run_visual_search.ps1 -Policy teacher -Plan datasets/projectairsim_visual_search/generalization_v3_added/plan.json -Set train -Record datasets/projectairsim_visual_search/generalization_v3_added -Output outputs/visual_search/gen_v3_collect_train
# gate: smoke 10개 → 판정 → 검증 시작 전부 → 판정
.\scripts\run_visual_search.ps1 -Policy oft -Checkpoint outputs/aerovla_oft/checkpoints/generalization_v3b -Plan datasets/projectairsim_visual_search/generalization_v3_added/plan.json -Set val -Output outputs/visual_search/gen_v3b_gate
.\assets\projectairsim-env\Scripts\python.exe scripts\gate_gen_v3.py representative --run outputs/visual_search/gen_v3b_gate --teacher datasets/projectairsim_visual_search/generalization_v3_added
.\assets\projectairsim-env\Scripts\python.exe scripts\freeze_baseline.py verify gen_v2
```

```bash
python scripts/build_visual_search_dataset.py datasets/projectairsim_visual_search/generalization_v3b \
  --sources datasets/projectairsim_visual_search/generalization_v1 datasets/projectairsim_visual_search/generalization_v2_added \
            datasets/projectairsim_visual_search/generalization_v3_added datasets/projectairsim_visual_search/generalization_v3b_added
python scripts/train_aerovla_oft.py --dataset datasets/projectairsim_visual_search/generalization_v3b --output outputs/aerovla_oft/checkpoints/generalization_v3b \
  --steps 12000 --eval-every 500 --eval-samples 400 --patience 4 --strategies right --boost small_visible=2 --select-band 0.02 --name OFT-Generalization-v3
python scripts/train_aerovla_oft.py --probe outputs/aerovla_oft/checkpoints/generalization_v3b --dataset datasets/projectairsim_visual_search/generalization_v3b
```

checkpoint와 dataset은 Git에 없다. Gen-v2의 checkpoint, dataset, 결과는 읽기만 했다.

## Tests

- **Python 189/189 통과**(WSL). 이 단계에서 추가한 것: 장면과 물체(속성이 주장대로만 다름, 자리 바꿈, 학습 장면에 held-out 물체 없음, Depot 학습 0), pad의 가시성과 표시, 착륙 teacher(문장이 끝을 정함, 가운데에서만 하강, 낮게 내려오면 하강 유지, touchdown 뒤에만 정지, 찾기 전에는 과제와 무관), 강제로 민 approach와 hop 착륙, twin이 바꾸라는 것만 바꿈, test 파일이 생성기 출력과 같음, set별 구성과 swap, 학습 계획이 test 장면·layout·문장·시작과 겹치지 않음, 물체가 한 문장에 묶이지 않음, touchdown과 collision의 구분, 착륙 규칙, approach는 공중에서 끝남, grounding 전환의 정의, 실패 단계 분류, checkpoint 규칙, 크기 삼등분, gate.
- **Live:** pilot 16 + 22 episode 수집과 gate 비행 8 + 8회, 본 수집 415 + 199 episode, teacher의 test set 156회, 검증 시작 39 × 2회, 회귀 112회, 예시용 7회. 충돌 0. 오류로 다시 비행한 episode 0.
- **Checkpoint 재로드:** 네 checkpoint(pilot 둘, Gen-v3 둘) 모두 새 프로세스에서 불러 저장 시점 예측과 비교했고 차이는 0.0이다.
- **Gen-v2 동결 확인:** 학습 전에 기록한 120개 파일의 hash를 끝에 다시 확인했다(`freeze_baseline.py verify gen_v2`).

