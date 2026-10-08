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
