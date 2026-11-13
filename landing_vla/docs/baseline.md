# Current Baseline

2026-10-05에 실제로 확인한 정상 동작 상태다. 앞으로 기능을 추가할 때 이 상태와 비교한다.

| 항목 | 현재 구성 |
|---|---|
| OS | Windows 11 + WSL2 Ubuntu 24.04.4 |
| GPU | RTX 5070 12GB, driver 591.86 |
| Simulator | Windows Project AirSim Blocks, 배포 1.0.1 / Python client 1.0.2 |
| VLA | OpenVLA-7B + AeroVLA LoRA |
| Inference | 언어 모델 NF4 4-bit + BF16 compute, 나머지 모듈은 혼합 정밀도 |
| Input | Front RGB + Down RGB, 자연어 지시 |
| Output | forward / down / yaw |
| Connection | WSL2 → Windows host의 직접 NNG 통신, topics 8989 / services 8990 |

```text
Camera → AeroVLA → Action Adapter → Drone Movement → 새 Camera
```

| 실측 | 결과 |
|---|---:|
| Single-step | PASS, 실제 이동 0.336m |
| Closed loop | 10/10 completed |
| 평균 AeroVLA inference | 1.05초 |
| 평균 전체 step | 4.53초, 약 0.22Hz |
| Combined global GPU peak | 9.76GiB, desktop 등 포함 |
| Windows system RAM peak / 최소 여유 | 30.07 / 1.05GiB |
| Timeout / OOM / simulator crash | 없음 |

이 baseline 측정 당시에는 forward 최대 0.5m, vertical 최대 ±0.3m, yaw 최대 ±15°로 제한했고 고도 범위 이탈은 거부했다. takeoff 반환값이 False였던 이전 시험이 있어, 실제 상승·airborne 상태를 함께 확인한다.

**2026-10-06 이후:** 미션 실행기는 모델 행동을 upstream과 같은 의미·크기로 실행한다(전진 0–5m, 상하 ±5m, 회전 ±63°; 큰 회전은 제자리 회전). 이동은 위치·고도 유지 명령을 쓰고, 고도 범위는 실패가 아니라 목표 고도 제한으로 적용한다. Blur 데모만 시작 플랫폼 위에 머물도록 이동 거리를 1/10로 줄여 쓴다. 근거와 측정은 [Model Self-Evaluation](model_evaluation.md).

Blocks 한 환경의 기능 연결 시험이며 목표 도달 성능을 평가한 결과는 아니다. 기본 blur/drift는 별도의 model-free 데모다. 모델을 연결한 failure injection·자동 감지·복구는 아직 없다.

실행 준비: [setup](setup.md). 테스트 요약: [experiments](experiments.md). 상세 측정: [최종 검증 기록](archive/final_closed_loop_validation.md).
