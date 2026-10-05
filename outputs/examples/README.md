# 실제 데모 이미지

2026-10-05의 실제 simulator camera/debug export다. 기존 3장은 그대로 보존했다. 외부 Chase framing 개선 후 새 hero·관찰 창·짧은 scripted flight GIF를 추가했다. AI 생성 이미지나 desktop screenshot이 아니다.

| 파일 | 원본 로컬 기록 | 의미 |
|---|---|---|
| [closed_loop.png](closed_loop.png) | `outputs/communication_final/debug_10.png` | 실제 AeroVLA live closed-loop의 10번째 decision, Front/Down 영상과 raw action |
| [control_drift.png](control_drift.png) | `outputs/platform_final/demo-control-drift.png` | 모델 없는 script 비행 중 blur와 lateral/yaw disturbance를 함께 표시한 데모 |
| [drone_view.png](drone_view.png) | `outputs/platform_final/Chase-first.png` | Project AirSim Blocks의 실제 Chase camera |
| [hero_drone.png](hero_drone.png) | `outputs/demo_views/hero_drone.png` | Close Chase, 실제 airborne drone; raw camera 960×540 |
| [observer_view.png](observer_view.png) | `outputs/demo_views/observer_hero.png` | 큰 외부 시점 + Front/Down + 실제 command/telemetry, model-free |
| [drone_flight.gif](drone_flight.gif) | `outputs/demo_views/frames/`, `capture-result.json` | 실제 전진·yaw·전진; 기록된 wall-time 간격대로 GIF 인코딩 |

`control_drift.png`는 AeroVLA가 장애를 감지하거나 복구했다는 결과가 아니다. 전체 frame dump와 raw 로그는 Git에서 제외한다.

새 비행 GIF도 AeroVLA 조종 성능을 보여주는 영상이 아니다. 기존 AI 화면과 별도의 scripted visualization이다. Hero PNG와 관찰 PNG는 원본 export를 복사했다. GIF만 크기·색상 수를 줄였으며 새 frame을 생성하거나 보간하지 않았다. 테스트용 close/medium/elevated 후보와 raw frames는 로컬 `outputs/demo_views/`에 남기고 Git에 넣지 않는다.
