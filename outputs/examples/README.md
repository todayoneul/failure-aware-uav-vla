# 실제 데모 이미지

2026-10-05의 저장 프레임에서 3장만 골랐다. 이미지 편집이나 재실행 없이 원본을 복사했다. desktop screenshot이 아닌 simulator camera/debug export다.

| 파일 | 원본 로컬 기록 | 의미 |
|---|---|---|
| [closed_loop.png](closed_loop.png) | `outputs/communication_final/debug_10.png` | 실제 AeroVLA live closed-loop의 10번째 decision, Front/Down 영상과 raw action |
| [control_drift.png](control_drift.png) | `outputs/platform_final/demo-control-drift.png` | 모델 없는 script 비행 중 blur와 lateral/yaw disturbance를 함께 표시한 데모 |
| [drone_view.png](drone_view.png) | `outputs/platform_final/Chase-first.png` | Project AirSim Blocks의 실제 Chase camera |

`control_drift.png`는 AeroVLA가 장애를 감지하거나 복구했다는 결과가 아니다. 전체 frame dump와 raw 로그는 Git에서 제외한다.
