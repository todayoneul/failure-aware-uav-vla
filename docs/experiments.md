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

이 repository 정리에서는 새 비행·모델·simulator 다운로드를 하지 않았다. 기존 adapter 테스트와 Git에 없는 조사 파일 없이 scene 설정을 준비하는 회귀 테스트 **6/6이 통과**했다. Python compile, PowerShell 8개·Bash 3개 문법 검사도 통과했다. baseline 카메라 설정과 공개 예제 이미지 3장의 원본 일치를 확인했다.

상세 기록: [최종 loop](archive/final_closed_loop_validation.md), [통신](archive/communication_stability_test.md), [NF4](archive/aerovla_int4_validation.md), [Project AirSim](archive/projectairsim_smoke_test.md), [이전 기록 목록](archive/README.md).
