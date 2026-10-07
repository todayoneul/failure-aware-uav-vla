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
| [gaussian_blur_comparison.png](gaussian_blur_comparison.png) | Gaussian Blur 6-step live 검증의 `comparison.png`, 이후 `outputs/failure_demo/runs/`에 보존 | 동일 관측의 원본 기준 vs 실제 AeroVLA input, MEDIUM |
| [gaussian_blur_observer.png](gaussian_blur_observer.png) | 같은 검증의 `observer_blur.png` | 실제 input·prompt·raw/decoded/bounded action을 표시한 관찰 창 |
| [mission_runner.png](mission_runner.png) | `outputs/mission_demo/` Test 5 GO_TO MEDIUM Blur live 관측을 최종 UI renderer로 export | 실제 맵·depth 목표·궤적·input·행동; Final은 판정 시점, live는 마지막 관측의 거리 |
| [mission_landing_failed.png](mission_landing_failed.png) | `outputs/mission_demo/runs/20261006T012742Z-6cbfc3b7/mission_failed.png` | 실제 GO_TO_AND_LAND 실패: 최종 XY 0.5044m >0.45m |
| [full_map_grounding.png](full_map_grounding.png) | `outputs/full_map/far-run/inspector.png` | 실제 68m 목표의 12번째 decision, 전체 맵·실제 hint·point visibility와 실패 결과; camera cross는 display copy에만 있음 |

| [model_evaluation.jpg](model_evaluation.jpg) | `outputs/model_eval/` 7개 run의 `results.json`을 `scripts/summarize_model_evaluation.py`로 그림 | 69 trial의 실제 궤적, 조건별 8칸. 배경은 simulator top-down 촬영이며 모델 입력이 아님 |
| [model_evaluation_frames.jpg](model_evaluation_frames.jpg) | `outputs/model_eval/run1/frames/`의 step별 모델 입력 | 실제 Front/Down 입력과 그 step의 모델 출력. 위: 102m 비행 후 cone 옆 LAND, 아래: 설명은 cone인데 힌트가 가리킨 ball 위에서 LAND |
| [model_evaluation_roof.jpg](model_evaluation_roof.jpg) | `outputs/model_eval/roof`, `instruction`의 `results.json`을 `outputs/model_eval/make_roof_figure.py`(로컬)로 그림 | 지붕 목표 6회의 평면도와 측면도, 지시문만 준 6회의 평면도. 배경은 simulator top-down 영상 |
| [visual_search_demo.png](visual_search_demo.png) | `outputs/visual_search/demo/viewer.png`, `run_visual_search_demo.ps1 -Model oft -Cases C` | AeroVLA-OFT visual search의 실제 관찰 창: 뒤돌아 시작해 43 tick 뒤 목표 11.1m에서 정지. 십자는 화면용 복사본에만 있음 |
| [visual_search_dataset/](visual_search_dataset/) | `datasets/projectairsim_visual_search/pilot`에서 sample 3개 | 학습 데이터 예시: `samples.json`(search / approach / stop), `samples.jpg`(각 Front 위·Down 아래), `summary.json`(전체 통계) |
| [mission_landmark.png](mission_landmark.png) | `outputs/mission_demo/` 대화형 세션의 `current_view.png` | Blue cone landmark 미션: 43 step, 모델 LAND, 착륙, 목표 3.59m |

`control_drift.png`는 AeroVLA가 장애를 감지하거나 복구했다는 결과가 아니다. 전체 frame dump와 raw 로그는 Git에서 제외한다.

새 비행 GIF도 AeroVLA 조종 성능을 보여주는 영상이 아니다. 기존 AI 화면과 별도의 scripted visualization이다. Hero PNG와 관찰 PNG는 원본 export를 복사했다. GIF만 크기·색상 수를 줄였으며 새 frame을 생성하거나 보간하지 않았다. 테스트용 close/medium/elevated 후보와 raw frames는 로컬 `outputs/demo_views/`에 남기고 Git에 넣지 않는다.

새 `gaussian_blur_*` 두 이미지는 **실제 AeroVLA input injection** 검증에서 가져왔다. 기존 `control_drift.png`와 달리 실제 NF4 model inference를 수행했다. 자동 detection/recovery의 구현이나 성공을 보여주는 자료는 아니다.

`model_evaluation*` 두 장과 `mission_landmark.png`는 2026-10-06 하네스 수정 뒤의 실제 기록이다. 궤적 그림과 입력 모음은 저장된 실제 데이터로 그린 것이고 JPEG로 저장했다. 한 번의 성공 화면이 일반적인 성공률을 뜻하지 않는다. 같은 조건의 반복 결과는 [Model Self-Evaluation](../../docs/model_evaluation.md)에 있다.
