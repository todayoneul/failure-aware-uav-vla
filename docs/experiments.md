# 실제 실행한 테스트

2026-10-04~05에 수행한 주요 시험만 정리했다. 서로 다른 시험의 latency나 메모리를 한 결과로 합치지 않는다.

| 시험 | 결과 | 핵심 기록 |
|---|---|---|
| CUDA / BF16 / tiny NF4 | PASS | RTX 5070, CC 12.0, PyTorch cu128, BF16 및 HF NF4 loading |
| TravelUAV WSL rendering | GPU 조건 FAIL | camera/state는 응답했지만 llvmpipe CPU rendering |
| Windows Project AirSim smoke | GPU·camera·script motion PASS | Front/Down 각각 100프레임, 평균 40.76 / 40.89ms. takeoff False 반환 제한 |
| 기본 blur / control drift | Model-free demo PASS | blur 적용 영상과 scripted lateral/yaw disturbance 확인 |
| 실제 AeroVLA NF4 | PASS | OpenVLA-7B + LoRA, 단독 11회 생성 |
| 최초 relay 통합 | FAIL | 모델 로딩 후 scene topic 초기화 timeout. 실패 기록 보존 |
| Direct WSL reconnect | PASS | 20/20, 60초 idle 후 5/5 |
| Resource pressure | Tested levels PASS | RAM 2/4GiB, GPU 7GiB. RAM 8/12/16GiB는 여유 부족으로 미실행 |
| Single-step VLA | PASS | live camera → action → 실제 이동 0.336m |
| 10-step closed loop | PASS | 10/10 완료, inference 평균 1.05초, 전체 step 4.53초 |

최종 loop의 GPU global peak는 **9.76GiB**, Windows RAM peak는 **30.07GiB**다. Timeout·OOM·simulator crash는 없었다. 비행 중 collision event는 0건이며, 초기 platform 접촉과 착륙 접촉은 별도로 남겼다. 자동 장애 감지·복구 또는 navigation benchmark는 수행하지 않았다.

초기 repository 정리에서는 새 비행·모델·simulator 다운로드를 하지 않았다. 당시 adapter·설정 회귀 테스트 **6/6**, Python compile, PowerShell 8개·Bash 3개 문법 검사도 통과했다. baseline 카메라 설정과 기존 공개 예제 이미지 3장의 원본 일치를 확인했다.

## 외부 관찰 화면 개선 — 2026-10-05

기존 Blocks에서 **모델 없는 별도 scripted flight**를 촬영했다. AeroVLA 성능 시험을 다시 수행한 것이 아니다. 원본 Front/Down·vehicle geometry·dynamics·integration 코드와 기존 3장 이미지는 유지했다.

| Chase 후보 | 후방 / 위쪽 | FOV | 관찰 |
|---|---|---|---|
| 기존 | 10 / 1m | 90° | 드론이 작아 처음 보는 사람이 구별하기 어려움 |
| Close | 1.9 / 0.55m | 60° | 기체와 네 로터가 가장 명확해 **선택** |
| Medium | 2.8 / 0.75m | 60° | 주변 공간이 더 넓게 보임 |
| Elevated | 2.4 / 1m | 60° | 기체 위쪽 형상이 잘 보임 |

후보 3개의 `set_camera_pose` / `set_field_of_view`는 실제 True를 반환했다. Raw Chase는 960×540이며, 외부 관찰 창에는 Front/Down과 command·clearance·heading·position을 표시했다. 기존 baseline의 Chase 256×256과 구별한다.

첫 촬영은 순차 camera RPC와 저장 때문에 약 3fps였다. Chase topic 구독으로 바꾼 최종 촬영은 **90개 실제 frame, 약 10.67fps**였다. GIF는 **8.34초 / 480×270 / 3,448,620 bytes (3.29MiB)**이며 실제 기록 간격을 유지한다. Frame 생성·보간·시간 가속은 하지 않았다.

전진·yaw·전진·hover 명령은 True, 실제 이동을 state로 확인했다. 기존과 같이 takeoff 반환은 False였지만 실제 climb·airborne 조건을 통과했다. 마지막 land는 True, disarm 후 landed state 0, cleanup error는 없었다. Simulator/client는 종료했다. 새 모델·map·package를 다운로드하거나 failure 기능을 구현하지 않았다.

회귀 테스트는 **10/10**이다. Chase만 바뀌고 physics와 AI camera 설정은 보존되는지, camera optical axis, GIF의 기록 시각 보존을 추가 검사했다. 후보·최종 촬영 raw frames·세 차례 촬영의 결과 JSON은 로컬 `outputs/demo_views/`에 보존하고 선택한 media만 공개한다.

상세 기록: [최종 loop](archive/final_closed_loop_validation.md), [통신](archive/communication_stability_test.md), [NF4](archive/aerovla_int4_validation.md), [Project AirSim](archive/projectairsim_smoke_test.md), [이전 기록 목록](archive/README.md).

## Gaussian Blur live input injection — 2026-10-05

현재 구현은 [Gaussian Blur 실행 안내](gaussian_blur_demo.md)에 있다. 기존 model-free blur/drift와 달리 **실제 Front/Down → GaussianBlur → 기존 AeroVLA preprocessing/inference**에 연결했다.

- NORMAL 2 → MEDIUM BLUR 2 → NORMAL 2, **6/6 완료**.
- 실제 CUDA BF16 tensor의 원본 기준 대비 차이: `false,false,true,true,false,false`. 표시 입력과 모델 입력의 frame SHA도 일치했다.
- 실제 movement: 0.336 / 0.239 / 0.386 / 0.410 / 0.305 / 0.387m. Timeout/OOM 없이 land/disarm 완료.
- Blur ON pair 평균 **4.776ms**; 첫 9.020ms, 다음 0.533ms. 첫 generation 2.880s, 이후 5회 평균 1.075s.
- 리뷰 보완 뒤 새 launcher의 추가 1-step도 PASS, actual movement와 정상 종료/worker 소멸을 확인했다. 이전 ON/OFF 결과는 `outputs/failure_demo/runs/`에 보존하고 새 실행의 파일과 분리했다.
- 회귀 테스트 **24/24**, Windows native/WSL argv 경계 smoke 2개 통과. 종료 helper는 무해한 소유 테스트 worker만 종료하고 다른 run-token 프로세스는 거부하는 테스트로 확인했다.

물리 키를 자동으로 눌러 시험한 것은 아니다. 동일 B 상태 변경 경로의 live 검증과 키/버튼 handler 테스트를 수행했다. LOW/HIGH는 모듈 테스트에서 실행했으며, 실제 VLA live 기록은 MEDIUM이다. 모델 행동 차이의 원인을 blur로 단정하지 않으며 detection/recovery, 재학습, benchmark는 수행하지 않았다.

## Harness fix and model self-evaluation — 2026-10-06

이전 Near/Medium/Far 미션 실패를 로그로 다시 분석해 실행 하네스를 고치고, AeroVLA NF4를 고정 조건에서 평가했다. 상세: [Model Self-Evaluation](model_evaluation.md).

- **원인:** 모델 행동(전진 0–5m, 회전 ±63°)을 0.5m·15°로 잘랐고, 속도 명령이 step마다 약 6cm씩 고도를 잃어 0.8m 하한에서 미션이 종료됐다. 모델의 LAND는 무시했다.
- **수정:** upstream과 같은 의미·크기의 실행(위치·고도 유지 API), 고도 범위 0.8–30m를 실패가 아닌 목표 제한으로 적용, 모델 정지로 episode 종료 후 착륙·판정, 미션 종류 통합, landmark 목표와 방향 힌트 ON/OFF.
- **실행기 실측:** 1m 이상 이동 441회의 명령 대비 오차 중앙값 0.16m, 고도 오차 중앙값 0.02m. 69 trial에서 고도 이탈·runtime·cleanup 오류 0건.
- **모델 평가 69 trial / 654 decision / 1,873m:** 힌트가 목표를 가리킨 35회(초기 20m 초과) 중 33회 접근, 20m 안 모델 정지 12회, 충돌 7회, invalid 출력 6회. 힌트 없는 13회는 모두 첫 step LAND. 설명과 힌트가 다르면 힌트 쪽으로 이동.
- **대화형 데모:** 2세션 3미션. 좌표 목표 7.85m 정지·착륙, blue cone 102m 비행 후 3.59m 정지·착륙, 다른 1회는 `diverging` 실패.
- **Blur 데모:** `-AutoTest` 6/6 PASS. 이동 거리는 1/10 배율을 쓰며, 6 step 동안 고도 변화 2.4cm.
- **Tests:** Python 73/73, Windows PowerShell 5.1 검사 4개 통과. PowerShell 7은 이 PC에 없어 미실행.

같은 조건의 반복 실행이 서로 다른 결과를 냈다. 위 수치는 성공률 주장이 아니라 관찰 기록이다. Failure 비교, BF16 원본과의 비교, TravelUAV 환경 평가는 수행하지 않았다.

### 연속 비행 추가 — 2026-10-06

step마다 멈추는 대신 도착 2초 전에 다음 판단을 시작해 명령을 이어 붙이는 방식을 추가했다. 상세: [연속 비행](model_evaluation.md#연속-비행).

- **실측:** 이동 중 명령 교체 시 속도 0.80m/s 이상 유지. 2초 전 교체에서 최저 0.96m/s.
- **비교(조건당 5회, hint + landmark):** 정지 시간 비율 21% / 18% → 8% / 5%, 1m당 2.05 / 1.67초 → 1.29 / 1.12초. 20m 안 정지는 wall 3/5 대 3/5, cone 3/5 대 1/5. decision 수가 약 1.9배라 invalid 출력으로 끝나는 trial이 늘었다.
- **대화형 데모:** 기본을 연속으로 변경. 좌표 목표 2.27m, Blue cone 15.9m에서 모델 정지·착륙, 정상 종료.
- **Tests:** Python 80/80, Windows PowerShell 5.1 검사 4개 통과.

### 깨진 출력, 블록 위 목표, 지시문 방식 — 2026-10-06

블록 위를 목표로 한 대화형 미션 3개가 모두 `invalid_action`으로 끝난 로그에서 출발했다. 상세: [해당 절](model_evaluation.md#깨진-출력-블록-위-목표-지시문만으로-찾아가기).

- **invalid 출력:** 기록된 18개 모두 `LAND`가 없고, 13개는 숫자 자리에 다른 토큰이 낀 이동 명령이었다. 낀 토큰 20개 중 17개가 원본 OpenVLA의 action 토큰(id 31744–31999). 세션의 마지막 입력으로 `9식 49 49`를 재현했고, 유효 토큰만 고르면 `97 49 49`였다.
- **수정:** 출력 형식 `DD DD DD[ LAND]`만 허용하는 디코더를 기본으로 했다. 이후 평가 34 trial / 757 decision에서 토큰 교체 27회, `invalid_action` 종료 0회.
- **디코더 전후(연속 비행, 조건당 5회):** cone 20m 안 정지 1/5 → 3/5, wall 3/5 → 1/5. invalid 종료 4/10 → 0/10, 충돌 2/10 → 6/10. wall 충돌 4회에는 토큰 교체가 없었다.
- **블록 위 목표:** 42m 거리에서 낮은 고도 그대로는 지붕 착륙 0/3, 지붕보다 6m 위로 올라간 뒤에는 6회 모두 10m 안 정지·5회 지붕 착륙. 97m 출발점에서는 세 방식 9회 모두 실패. 데모는 목표 45m 안에서 올라가는 방식을 쓴다.
- **지시문만:** 저장 프레임 135장에서 회전 방향이 목표 쪽 10 / 반대 8(힌트가 있으면 21 / 3). 실제 비행 6회 중 지시한 물체 20m 안 정지 0회.
- **대화형 데모:** 2세션 3미션 정상 종료. 지붕 목표는 30.7m, 21.0m에서 정지해 실패, 지시문 모드는 첫 step LAND.
- **Tests:** Python 94/94, Windows PowerShell 5.1 검사 4개 통과.

## AeroVLA-OFT와 Visual Search — 2026-10-07

브랜치 `feat/aerovla-oft-visual-search`. 상세: [AeroVLA-OFT](aerovla_oft.md).

- **조사:** 공식 AeroVLA 학습 코드·데이터 형식(텍스트 bin 99개, LoRA r64, 방향 문장이 데이터에 포함)과 OpenVLA-OFT 구현(병렬 디코딩, L1 head, chunk, proprio, FiLM)을 확인했다.
- **Baseline (AeroVLA, 문장만):** A 3/6, B 0/6, C 0/6. C에서 목표가 시야에 들어온 적 없음.
- **학습 가능성:** NF4 + LoRA r16 + head, batch 1에서 최대 9.8GiB, step당 0.42초. 추론 forward 181ms.
- **Dataset:** teacher 72 episode, 2,923 sample (search 514, stop 288). episode 단위 58 / 14 split.
- **Overfit:** 48 sample에서 L1 0.64 → 0.044, 재로드 차이 0.0.
- **Pilot:** 1,200 update, 39분. train L1 0.023, val 0.322(첫 행동 0.254).
- **AeroVLA-OFT:** A 6/6, B 6/6, C 5/6, D 6/6. 판단 주기 6.26초 → 0.52초. 학습 범위 밖 시작에서는 2/6.
- **하지 않은 것:** FiLM, proprio 비교, 다른 맵, Blur와의 결합, LAND.

## AeroVLA-OFT 일반화 (Gen-v1) — 2026-10-07~08

브랜치 `exp/aerovla-oft-generalization`. 상세: [일반화](aerovla_oft_generalization.md).

- **준비:** 맵을 `configs/maps/`로 옮김. Blocks에 red cube, green cylinder, 벽 4개, 평가 전용 yellow pyramid를 실행 중에 띄움. 같은 simulator 안에 두 번째 장면 Yard를 만듦(학습 비행 0회).
- **Held-out:** 학습 계획 전에 G1 32, G2 12, G3 20, G4 6, 문장 변형 24, 좌우 시험 16회의 시작 상태를 고정. 학습 시작점은 6m 이상 떨어지고 물체 자리마다 60° 방향을 비움.
- **Teacher:** 영상으로 정해지는 규칙만 학습에 사용(오른쪽 회전, 앞이 막히면 상승). 좌우를 50/50으로 섞으면 탐색 frame의 yaw 예측이 +0.88 → +0.07.
- **Dataset:** 292 episode(260 / 32), 11,745 sample, teacher 실패 0, train/val 누출 0.
- **학습:** 2,250 update에서 조기 종료, 검증 최저 1,250 update. train 0.115 / val 0.092, peak 9.8GiB, 108분.
- **Action range:** pilot의 범위 밖 예측 24%는 label이 한계값인 축의 1–2% 초과. tanh head는 같은 300 update에서 val 0.151 → 0.383이라 쓰지 않음.
- **결과:** G0 20/20, G1 27/32, G2 9/12, G3 19/20, G4 2/6, 문장 변형 21/24, 좌우 시험 16/16, 충돌 0. 처음에 안 보이던 목표 73/73 획득.
- **비교:** pilot G1 4/32(자기 장면·문장으로 5/16), G3 cone·ball 6/10, G2 0/6. 기존 AeroVLA G1 step 1/16, continuous 0/16.
- **대조:** 다른 물체를 지시하면 가까운 물체에 멈춘 것은 pyramid 옆 0/12, Yard 2/20.
- **찾은 문제:** 연속 실행 시 simulator 포트 해제 대기, scene reload 때 조명 초기화(Yard 저녁 조명). 둘 다 고침. 모델의 탐색 회전이 teacher의 절반 속도(14°/s 대 27°/s)인 것은 고치지 않음.

## AeroVLA-OFT Gen-v2 — 2026-10-08

같은 브랜치. 상세: [Gen-v2](aerovla_oft_gen_v2.md). 동결한 표: `outputs/generalization/gen_v1_final/`, `gen_v2_final/`.

- **분석:** Gen-v1의 실패 16회를 10개 분류로 나눔. 탐색 실패 0, 다른 물체로 감 0, 이른 정지 4, 찾은 뒤 계속 상승 4, 접근 실패 4, 문장 변형에서만 3, 미정지 1. 병목은 찾은 뒤의 전환과 정지.
- **바꾼 것:** dataset만. 전환 장면 153 episode(벽 넘어 상승 39, 높은 곳 접근 32, 정지 거리 부근 40, 가장자리의 가까운 목표 28, 가깝고 시야 밖 14)를 더해 445 episode, 16,765 sample.
- **학습:** 처음부터 같은 설정으로 4,000 update(검증이 끝까지 내려감). train 0.049 / val 0.070. Gen-v1의 검증 frame으로는 0.092 → 0.067.
- **결과:** G0 20/20, G1 32/32, G2 11/12, G3 18/20, G4 4/6, 문장 변형 24/24, 좌우 시험 16/16, 충돌 0. 합계 114 → 125/130.
- **실패 분류:** 이른 정지 4 → 0, 계속 상승 4 → 0, 문장 변형 3 → 0, 접근 실패 4 → 4, 미정지 1 → 1. 남은 5회는 모두 44m 이상 시작(Yard 2, 처음 보는 물체 3).
- **주의:** 보강 종류를 같은 held-out set의 실패에서 골랐고 학습량도 늘었다. 새 held-out 추정치가 아니다.
- **Tests:** Python 146/146, PowerShell 4/4.
- **다음:** 새 seed의 held-out 시작 상태에서 Visual Search + Gaussian Blur.

## AeroVLA-OFT Gen-v2 장거리 held-out — 2026-10-08

같은 브랜치. 상세: [장거리 평가](aerovla_oft_long_range.md). 동결한 표: `outputs/generalization/long_range_v2/`. 학습·모델 변경 없음.

- **Set:** 새 seed(6000번대)의 시작 36개를 비행 전에 고정. L1 40–55m, L2 55–70m, L3 70–90m 각 12개(Blocks 6 / Yard 6, 보임 6 / 안 보임 6, 본 물체 8 / 처음 보는 물체 4). 모든 기존 시작과 6m 이상.
- **결과:** 27/36(L1 11/12, L2 8/12, L3 8/12). 충돌 0. Teacher 36/36.
- **단계:** 목표가 시야에 들어옴 36/36(안 보이던 시작 18/18), 20m 이내 27/36, 그 뒤 정지 27/27.
- **장면:** Blocks 17/18(70–90m 6/6), Yard 10/18(55m 이하 6/6, 56m 이상 4/12).
- **물체:** 본 물체 20/24, 처음 보는 물체 7/12. Yard의 본 물체도 56m 이상에서 4/8.
- **실패 9회:** 전부 "보였지만 20m 안에 못 들어옴". 8회는 목표가 시야를 4–12번 지나가는 동안 계속 회전(약 7바퀴), 1회는 72m에서 스스로 정지.
- **판정:** 부분적으로 일반화. 병목은 탐색도 정지도 아니고, 학습하지 않은 장면에서 멀리 작게 보이는 목표를 알아보는 것.
- **다음:** 탐색 구조는 그대로. 장면 외관을 달리한 장거리 접근 시작 데이터가 다음 후보. Gaussian Blur는 Blocks 전 거리와 Yard 55m 이하에서 이 기준선과 비교.
- **Tests:** Python 151/151.

## AeroVLA-OFT Gen-v3 (grounding과 착륙) — 2026-10-08~09

브랜치 `exp/aerovla-oft-gen-v3-grounding-landing`. 상세: [Gen-v3](aerovla_oft_gen_v3.md). 동결한 표: `outputs/generalization/gen_v3_dev/`. **Gate를 통과하지 못해 새 test set 156개는 비행하지 않았다.**

- **과제:** "Find the blue landing pad and land on it." 입력·출력·구조는 Gen-v2와 같음(FiLM, proprio 없음). 착륙도 전진·하강·yaw로 하고 `land_async`는 쓰지 않음.
- **준비:** 물체 9종(파랑·빨강 pad 14×14×2.5m, cube 2, cylinder 2, ball, cone, 학습에 없는 pyramid)을 한 catalogue로. 학습 장면 Field·Lot, 평가 전용 Depot(Scene C)을 실행 중에 생성. 착륙 teacher, collision topic으로 읽는 touchdown, 착륙 규칙(`configs/targets/landing_pads.json`).
- **Test set:** seed 8000번대 156개를 학습 계획 전에 고정(색·형태 distractor, 위치 swap, query swap, approach 대 land, 40–90m, canonical 착륙 24, 안 쓴 문장). teacher 156/156.
- **Gate:** pilot → smoke 10 → representative 36을 통과해야 test set을 비행. 기준은 config에 미리 고정. smoke·representative는 검증 split의 시작.
- **Pilot:** 첫 시도는 통과 못 함(착륙 4회 중 2회가 pad 0.3m 위에서 정지, approach 2회가 경계를 지나 착륙 동작). pad 윗면에 격자, 낮게 내려온 뒤 하강 유지, 경계 안쪽으로 민 approach를 넣고 두 번째에 통과(착륙 touchdown 4/4, approach 공중 정지 4/4).
- **첫 학습:** 859 episode / 40,432 sample, 8,000 update, 262분, val L1 0.066. Smoke 6/10으로 통과 못 함. 검증 시작 36개 진단: 28/36, 실패는 같은 색 다른 물체 4, touchdown 뒤 미정지 3, 원거리 1.
- **두 번째 학습:** 같은 색 대비(query twin 등)와 착륙 끝부분 보강 199 episode를 더해 1,052 episode / 52,396 sample, 처음부터 12,000 update, 406분, val L1 0.056. Smoke 7/10 통과, representative 27/36(0.75, 기준 0.80)으로 통과 못 함.
- **검증 시작에서(두 번째):** approach를 시켰는데 내려앉음 0/18, 지시한 pad에 touchdown 16/18, 착륙 규칙 전체 14/18, 충돌 0. 파랑이 아닌 목표 12/12, 파란 목표 15/24(54m 이상 3/9).
- **회귀(두 번째, 기존 set 112회):** Gen-v2 101 → 103. G1 31/32, G3 20/20, P 22/24, L 30/36(Yard의 본 물체 8/12 → 11/12, 처음 보는 물체 7/12 그대로).
- **찾은 것:** Front 화면의 수평선 부근은 기체의 팔이 가리고 가운데 16°만 열림(55m 이상, 8–25° 벗어난 목표는 31–35%만 보임). simulator는 touchdown마다 접촉 event를 한 번만 보냄.
- **남은 병목:** 54m 이상에서 파란 물체 넷 중 지시한 것을 확정하지 못함. 그다음은 touchdown 뒤의 0 행동(2/18).
- **Tests:** Python 189/189.
- **다음:** canonical mission의 범위(44m 이하 권장)를 먼저 정함. Blur는 gate를 통과한 뒤.

## Gen-v3 canonical clean baseline (44m 이하, Landing Finalizer) — 2026-10-09

브랜치 `exp/aerovla-oft-gen-v3-canonical-baseline`. 상세: [Canonical clean baseline](canonical_clean_baseline.md), [Long-range same-color grounding](long_range_same_color_grounding.md). 표: `outputs/generalization/canonical_baseline/`. **학습 없음. Gate를 통과하지 못해 동결하지 않았고 canonical test set 48개는 비행하지 않았다.**

- **정의:** canonical mission "Find the blue landing pad and land on it.", 시작 44m 이하. 44m 초과는 지우지 않고 long-range 과제로 따로 둠. Depot의 156개는 계속 봉인.
- **Landing Finalizer:** 실행기 안의 상태 기계(FLYING → LANDING_DESCENT → CONTACT_CANDIDATE → STABLE_CONTACT → LANDED_LATCHED → DISARMED). 착륙을 시키는 문장에서만 켜짐. 하강 중의 접촉 + 3 판단 정지면 latch, 그 뒤 동작 명령을 넘기지 않고 disarm. 목표의 좌표·거리·방위나 "맞는 pad인가"는 받지 않음. 숫자는 비행 전에 config에 고정.
- **착륙을 읽는 세 가지:** touchdown / stable physical landing / strict policy zero-action. Mission의 성공은 system landing(VLA + finalizer). Gen-v3의 기존 숫자는 다시 세지 않음.
- **새 시작 set:** 검증 36(seed 8500–8525, 학습 배치·안 쓴 시작), test 48(seed 9500–9530, Field·Lot의 새 배치 g–j). band 10–20 / 20–32 / 32–44m. approach·swap·query twin 포함. Teacher 36/36, 48/48(착륙 63회 전부 latch·disarm).
- **Gate (Gen-v3 두 번째 checkpoint 그대로 + finalizer):** smoke 9/10 통과. Representative 30/36: 성공률 0.83, 착륙 21/27, 접근 시작 32/34, 시킨 대로 끝남 33/36, 충돌 0은 기준을 넘었고, 다른 물체에서 끝난 비행 3회(기준 2회 이하)로 통과 못 함.
- **실패 6회:** 목표 선택 3(blue pad → blue cube, red pad → red cube, blue pad → red pad), 조기 공중 정지 1, 평가의 읽기 문제 2.
- **가장 큰 병목:** 목표 선택. 색이나 형태가 같은 다른 물체를 목표로 삼음. 2회는 그 물체가 먼저 보이고 지시한 pad는 안 보이는 시작(그런 시작 9개 중 2회). 44m 안에서도 남.
- **Finalizer:** 접촉 24회 모두 latch·disarm, approach 9회에서 켜진 적 없음. 결과를 바꾼 비행은 1회(정책이 23회 중 22회 스스로 멈춤).
- **찾은 결함:** 평가의 "서 있음" 판정이 simulator가 보고한 수직 속도에 기대는데, 서 있는 기체에 0이 아닌 값이 보고된 착륙 2회가 실패로 기록됨(위치 변화 0.0004m 이하). 숫자는 고치지 않음. 고쳐 읽어도 gate는 통과 못 함.
- **Tests:** Python 207/207.
- **다음:** Blur는 아직 아님. 목표 선택 병목을 어떻게 다룰지 먼저 정함. 다음 gate 전에 "서 있음" 판정을 위치 차이로 고치고 검증 시작을 새 seed로 다시 뽑음.

## Gen-v3c (hard-negative grounding 보정, evaluator v2) — 2026-10-09

브랜치 `exp/aerovla-oft-gen-v3c-hard-negative`. 상세: [Gen-v3c](gen_v3c_hard_negative_grounding.md). 표: `outputs/generalization/gen_v3c/`. **새 검증 set의 gate를 통과하지 못해 동결하지 않았고 canonical test set 48개는 비행하지 않았다.**

- **질문:** "먼저 보이는 비슷한 물체"를 데이터로 가르치면 지금 구조가 맞는 목표를 안정적으로 고르는가. 바꾼 것은 둘: 착륙의 "서 있음" 판정, hard-negative 보정 학습. 카메라·탐색 고도·head·chunk·FiLM·proprio는 그대로.
- **Evaluator v2:** 안정 착륙을 접촉 + 마지막 3 판단의 높이(접촉 높이 ±0.05m)와 위치 변화(수평·수직 0.05m 이하)로 읽음. 보고된 속도는 쓰지 않음. 임계값은 기록된 착륙 87회의 window에서(서 있음 최대 0.0003m, 하강 중 0.45m 이상). 이전 결과(30/36)는 legacy evaluator result로 두고 다시 계산하지 않음. Finalizer는 그대로.
- **Data:** Field·Lot의 새 배치 k–n에서 96 + 12 episode(4,933 frame). 같은 색·다른 형태가 먼저 보임 30, 같은 형태·다른 색이 먼저 보임 14, 놓친 뒤 관련 물체가 보임 20, 자리 바꾼 twin 16, query twin 16. 파랑 65·빨강 31, pad 46·cube 25·cylinder 14·cone 11, 모두 44m 이하. 평가 set(이전 검증 36, test 48, Depot 156, pilot 12, 새 검증 36)의 시작과 배치를 쓰지 않음.
- **평가를 먼저 고정:** 새 검증 36개(seed 22000번대, 배치 p–s), pilot 12개(배치 t, u), gate 기준, 회귀 시작 33개를 녹화·학습 전에 commit.
- **보정 학습:** `generalization_v3b`에서 2,000 update, lr 5e-5, 뽑는 frame의 절반은 새 episode·절반은 이전 data, 목표 선택이 일어나는 frame 3배. 84분. 남긴 checkpoint는 update 1,250. 검증 L1(반반 600 frame) 0.0607 → 0.0601, 이전 검증 frame 0.0561 → 0.0581.
- **Offline probe:** 같은 색 물체가 화면 가운데 있고 지시한 것은 없는 frame에서 그쪽으로 전진하는 비율 33% → 21%. 보이는 물체를 지시하는 문장으로 바꿨을 때 전진으로 바뀌는 비율은 34% → 35%로 그대로.
- **Pilot (12회, 비행 전 기준: 성공 9 이상·다른 물체 1 이하):** Gen-v3c 11/12, 다른 물체 1 → 통과. 같은 12개에서 시작 checkpoint는 9/12, 다른 물체 2, 충돌 1.
- **새 검증 gate:** smoke 10/10 통과. Representative 33/36(0.92): 착륙 24/27, 접근 시작 32/33, 시킨 대로 끝남 34/36, 충돌 0은 기준을 넘었고, **다른 물체에서 끝난 비행 3회(기준 2회 이하)로 통과 못 함.**
- **진단:** 실패 3회 전부 "관련 물체가 첫 화면에 있고 지시한 것은 화면 밖"(15회 중 3회: 같은 색 7/9, 같은 형태 5/6). 그 밖은 21/21. 두 checkpoint를 합쳐 보면 그 조건의 오류는 15회 중 4회 → 21회 중 4회로, 구별될 만큼 달라지지 않음. 오류는 관련 물체가 화면에 오래 머무는 시작(탐색이 도는 오른쪽, 또는 정면 20m 이내)에 몰림: 18회 중 8회, 나머지 18회 중 0회.
- **착륙과 finalizer:** 지시한 pad로 간 24회 전부 안정 착륙·latch·disarm. Finalizer가 구한 착륙 1회. Approach 9회에서 켜진 적 없음. 이 run에서는 두 판정이 일치.
- **회귀 (이전 set의 33개 시작):** Gen-v2 30, 두 번째 Gen-v3 29, Gen-v3c 28. 잃은 한 비행은 이미 실패하던 Blocks의 blue cone 46m 시작의 다른 문장. 충돌 0, Yard 8/8.
- **판정:** 데이터 보정으로는 부족(질문의 답 NO). 같은 데이터를 더 넣지 않음. 다음 후보는 FiLM 등 더 강한 language–vision 결합(구현하지 않음). 44m 초과는 계속 별도.
- **Tests:** Python 241/241, PowerShell 4/4. Gen-v2 지문 그대로.
- **다음:** Blur는 아직 아님.

## Language–vision grounding architecture (FiLM, 대조 실험, canonical test) — 2026-10-10

브랜치 `exp/aerovla-oft-grounding-architecture`. 상세: [Language–vision grounding architecture](language_vision_grounding_architecture.md). 표: `outputs/generalization/grounding_architecture/`. **Canonical clean baseline이 gate와 test를 통과했다.**

- **질문:** wrong-target 실패가 문장이 시각 표현을 약하게 조건화해서 생기는가. Data는 Gen-v3c 그대로(episode 추가 없음), teacher·finalizer·evaluator v2·action·44m 범위 고정. 구조 실험은 둘까지: FiLM, 실패 시에만 cross-attention.
- **평가:** 새 pilot set 16개(seed 23000번대, 녹화한 적 없는 배치)를 모델마다 두 번씩 비행. 9개는 관련 물체가 "어려운 자리"(탐색이 도는 쪽, 또는 정면 가까이)에 있고 지시한 것은 화면 밖. 통과 기준(다른 물체 32회 중 2 이하, baseline보다 2 이상 적고 절반 이하, 두 번 다 틀린 시작 1 이하, 그 밖은 baseline보다 나쁘지 않음, 회귀 26/33 이상)은 학습 전에 commit.
- **구조:** 두 모듈 모두 vision encoder와 projector 사이(patch feature 256×2176). FiLM은 문장 token embedding의 평균으로 채널마다 γ·β(4.34M parameter). Cross-attention은 단어가 patch를 읽고 읽은 자리에 되돌려 씀(2.74M, 구현만 하고 학습 안 함). 학습 전에는 예측을 바꾸지 않음(차이 0.0). Vision encoder·projector·LLM·AeroVLA adapter는 고정.
- **Baseline (Gen-v3c):** 28/32, 다른 물체 4, 충돌 2. 실패는 `red pad ← red cube` 두 시작에서 두 번씩.
- **FiLM (Gen-v3c에서 2,000 update, 모듈 lr 5e-4·LoRA lr 5e-5, 87분, 남긴 checkpoint update 1,750):** pilot 32/32, 다른 물체 0, 16개 시작 모두 두 번 다 맞음, 회귀 26/33. Pilot PASS. Experiment 2는 실행하지 않음.
- **대조 (모듈 없이 같은 2,000 update):** 30/32, 다른 물체 2, 두 번 다 틀린 시작 0. Pilot의 줄을 지킴. 같은 두 red pad 시작을 고침. 회귀 27/33. 미리 정한 규칙대로 개선을 FiLM의 효과로 인정하지 않음.
- **FiLM을 들여다보면:** blue pad 문장과 red pad 문장의 γ·β가 사실상 같음(차이 1.7%). 문장에 따라 달라지긴 하지만(조절 에너지의 71%) 고쳐진 실패의 색 구분을 실어 준 것은 아님.
- **다음 단계로 간 것:** 미리 정한 선택 순서로 pilot에서 가장 좋은 FiLM checkpoint. 구조의 증거가 아니라 baseline 후보로서. 이 결정은 검증 비행 전에 config에 적음.
- **새 검증 gate (seed 24000번대 36개, 첫 비행으로 판정):** smoke 10/10. Representative 34/36(0.94), 착륙 25/27, 접근 시작 34/34, 시킨 대로 끝남 35/36, 다른 물체 2(기준 2 이하), 충돌 0. PASS. 목표 선택 시작 17개의 두 번째 비행(판정 아님)에서는 13/17, 다른 물체 3.
- **동결:** checkpoint(FiLM 포함), 착륙 규칙(finalizer·evaluator), config, 지도, test 시작, 비행·판정 code의 지문을 `outputs/generalization/grounding_film_frozen.json`에.
- **Canonical test 48개 (한 번):** **47/48(0.979)**. 착륙 36/36(touchdown·안정 착륙·system·strict 모두), 다른 물체 1, 접근 시작 46/47, 시킨 대로 끝남 48/48, 충돌 0. 12–20m 15/16, 20–32m 16/16, 32–44m 16/16. 실패 하나는 "Find the blue cube."에서 먼저 보이는 blue pad 곁에 멈춤.
- **읽는 법:** Gen-v3c의 "데이터로는 부족하다"는 판정은 검증 L1이 고른 update 1,250의 checkpoint에 대한 것이었다. 같은 data를 더 학습하면 그 실패는 고쳐지고, 검증 L1은 그 차이를 보여 주지 못한다. 목표 선택은 여전히 가장 약한 곳이다.
- **비용:** FiLM은 추론 시간 +4%(216 → 225ms), VRAM 차이 없음(추론 6.94 GiB, 학습 9.88 GiB).
- **Tests:** Python 260/260, PowerShell 4/4. Gen-v2 지문 그대로.
- **다음:** Gaussian Blur 준비 완료(실행하지 않음). Depot의 156개와 44m 초과는 계속 별도.

## grounding_film Interactive Mission Control — 2026-10-10

브랜치 `feat/grounding-film-interactive-mission`. 상세: [grounding_film Interactive Mission Control](grounding_film_interactive_demo.md). **학습도 평가도 아니다.** 기존 Mission Control 화면에 동결한 baseline을 연결한 시연이고, canonical test의 47/48은 그대로다.

- **구조:** 미션 한 번은 canonical 평가의 `run_episode` 그대로(관측, 문장, chunk 4개 중 하나 실행, 연속 명령, stop 판정, Landing Finalizer). 동결 파일은 수정하지 않고 `SearchEnv`를 상속해 화면과 로그만 붙였다. 동결 지문(`grounding_film` 30개, `gen_v2` 120개) 그대로.
- **모델 입력:** Front RGB, Down RGB, 문장 하나. 목표 XYZ·거리·bearing·방향 힌트·지도·표시는 화면과 판정 전용. 모델로 가는 호출은 하나이고 목표를 넘길 parameter가 없다(테스트로 확인).
- **장면:** 기존 Blocks 지도 그대로에 catalogue의 pad 둘, cube 둘, cylinder 하나를 빈 자리에 spawn. 원래 있던 blue cone, orange ball도 목표. 평지의 launch 자세 두 곳에서 canonical episode와 같은 방식으로 시작(기존 platform은 1.5m 높아 pad 접촉 판정이 어긋난다).
- **조작:** 지도 클릭/N(물체), T(LAND/APPROACH), G, R(launch 자세로), L(launch 변경). Cube + LAND와 빈 땅 클릭은 시작하지 않는다. Prompt mode(M)는 이 mode에서 없다.
- **수동 확인(각 1회, 합계 13회: 성공 10, 실패 2, 중단 1):** 최종 배치에서 blue pad 착륙(65 decision), 같은 자세에서 접근(40, 공중 정지), red pad 착륙(106), blue cube 접근(37), blue cone 접근(27), green cylinder 접근(66) 모두 지시대로 끝남. 그 전 배치에서는 5회 중 4회 성공, blue cone 접근 1회 실패(색 블록 벽을 정면으로 본 자세에서 320 decision 동안 망설임). 배치는 이 비행들을 보고 고쳤다. 마지막 code 확인에서는 조작 실수로 110m 밖 건물 뒤의 cone을 지시했고 실패(120 decision).
- **Decision 주기:** canonical 0.513초. 비행 중 지도 캡처(0.18초)를 5초마다 하던 때는 평균 0.536초에 10번에 한 번 0.7초. 캡처를 없앤 뒤 0.515–0.530초.
- **기존 데모:** `run_mission_demo.ps1`은 그대로 동작(실행해 확인).
- **Tests:** Python 291/291, PowerShell 4/4.
- **하지 않은 것:** 재학습, canonical 검증·test 재실행, Depot 156, Blur 평가.

## Gaussian Blur robustness characterization — 2026-10-10

브랜치 `exp/failure-gaussian-blur-characterization`. 상세: [Gaussian Blur Robustness Characterization](failure_gaussian_blur.md). 표: `outputs/failures/gaussian_blur/summary/`. **측정만 했다.** 학습, 감지, 복구는 없다. 동결 지문은 launch 전후 8번 모두 그대로.

- **조건:** 정책이 받는 Front·Down RGB에 Gaussian Blur. Clean / Low 7×7 σ1.5 / Medium 15×15 σ3.0 / High 31×31 σ6.0. 비행 전에 config와 repeat subset을 commit(`90a1850`).
- **비행:** canonical test의 48개 시작 × 4조건 = 192회(시작마다 조건 순서를 다르게, 각 조건이 1–4번째로 12번씩), 계획만 보고 고른 12개 시작을 한 번 더 48회. 비행은 canonical `run_episode` 그대로이고 `BlurEnv.observe`가 frame 둘만 바꾼다. Ground truth, finalizer, evaluator는 blur 전의 장면을 읽는다.
- **확인:** clean 3,109 decision 모두 raw = model input, blur 조건 6,978 decision 모두 raw ≠ model input. 정책이 받은 배열과 기록된 hash가 모든 decision에서 일치.
- **Mission success:** 46 / 40 / 24 / 8 (of 48). 착륙 35 / 31 / 15 / 0 (of 36), 접근 11 / 9 / 9 / 8 (of 12). 충돌 0, wrong target 1 / 2 / 2 / 2.
- **실패 방식:** 정책이 공중에서 스스로 정지 행동을 낸다. 성공이 아닌 정지 1 / 8 / 24 / 38. High의 착륙 36회는 전부 공중 정지(15회는 pad 바로 위)이고 하강을 시작한 비행이 없다.
- **단계:** Low는 search·grounding만(Lot 장면의 목표가 첫 화면에 없는 시작 18 → 11), 착륙은 그대로. Medium부터 착륙이 가장 큰 몫(pad 위에 간 것 중 하강 시작 19/25, high 0/17).
- **거리:** 세 구간이 거의 같다(15/14/8/2, 16/14/8/3, 15/12/8/3).
- **Paired:** clean 대비 잃은 시작 7 / 22 / 38, 얻은 시작 1 / 0 / 0. Repeat은 48쌍 중 47쌍이 같은 결과.
- **Clean 재현:** 예전 47/48, 이번 46/48, 47개 시작이 같은 결과.
- **선명도(분석 전용):** Front의 Laplacian variance 중앙값 467 / 17.7 / 3.7 / 1.7로 네 조건이 겹치지 않는다. Down은 겹친다.
- **지연:** blur 처리 1.1ms 이하, decision 주기 0.511초로 네 조건 같음.
- **Runtime:** simulator 연결이 한 번 끊겨 두 비행을 다시 비행(하나는 자동, 하나는 disarm 응답이 없던 clean 비행을 기록을 남기고 따로 뺀 뒤). Simulator 창은 띄우고 비행(세션이 끊긴 상태에서 숨긴 simulator가 멈춤).
- **Interactive:** `run_grounding_film_mission_demo.ps1`의 B, 1/2/3이 같은 주입기를 쓰고 화면에 강도·kernel·sigma를 표시. 정성 확인 두 세션에서 clean·low 착륙, medium은 pad 위 정지, high는 26–28m 앞 정지.
- **Tests:** Python 309/309, PowerShell 4/4.
- **다음:** Front만 / Down만 blur하는 ablation, 그다음 detector와 recovery. 이 48개 시작은 더 이상 held-out이 아니므로 recovery는 새 시작에서 평가한다.
