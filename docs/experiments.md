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
