# Failure 기능 개발 계획

목표는 simulator를 보면서 장애를 직접 켜고 드론 반응을 관찰하는 재미있는 텀프로젝트다. 아래는 **개발 후보**이며 구현이나 모델 영향 측정 결과가 아니다. 기존 model-free blur/drift prototype는 참고할 수 있지만, AeroVLA에 연결된 공통 failure 모듈·자동 감지·복구는 아직 없다.

## Visual failures

실제 Front/Down camera를 받은 뒤, AeroVLA의 RGB 변환·resize·mosaic 전 단계에 적용하는 후보다. 원본 영상과 주입된 영상을 같이 표시한다.

| Failure | What happens | How to simulate | Difficulty | Demo value |
|---|---|---|---|---|
| Gaussian Blur | 영상이 갑자기 흐려진다 | Gaussian kernel로 처리 | 쉬움 | 높음 |
| Brightness Change | 너무 어둡거나 밝아진다 | 밝기 gain을 바꾸고 uint8 범위로 제한 | 쉬움 | 보통 |
| Image Noise | 영상에 잡음이 생긴다 | 강도를 제한한 noise 추가 | 쉬움 | 보통 |
| Partial Occlusion | 화면 일부가 가려진다 | 지정 영역을 검은 사각형으로 덮기 | 쉬움 | 높음 |
| Temporary Full Occlusion | 잠깐 아무것도 보이지 않는다 | 일정 시간 전체를 가리고 원본으로 복귀 | 쉬움 | 높음 |
| Frame Drop | 새 영상을 못 받고 화면이 멈춘다 | 마지막 frame 재사용, timestamp도 기록 | 보통 | 높음 |

## Control failures

모델 행동에 제한된 편향을 더하는 후보다. 주입 뒤에도 기존 속도·이동 거리·고도·회전 제한을 지킨다. 간단한 disturbance 데모이며 정확한 wind/actuator 물리 모델을 주장하지 않는다.

| Failure | What happens | How to simulate | Difficulty | Demo value |
|---|---|---|---|---|
| Lateral Drift | 의도한 방향에서 옆으로 밀린다 | body lateral velocity bias 추가 | 쉬움~보통 | 매우 높음 |
| Yaw Drift | heading이 계속 빗나간다 | 작은 yaw bias 추가 | 쉬움~보통 | 높음 |
| Vertical Drift | 고도가 의도와 다르게 변한다 | 제한된 vertical velocity bias 추가 | 보통 | 높음 |
| Delayed Action | 판단과 이동 사이가 늦어진다 | action 실행을 일정 시간 지연 | 보통 | 보통 |
| Action Scaling | 전진·회전이 약하거나 과하다 | 배율 조정 뒤 다시 제한 | 쉬움 | 높음 |
| Temporary Control Loss | 잠깐 새 명령이 적용되지 않는다 | 제한된 시간에 새 action을 막고 hover 유지 | 보통~높음 | 높음 |

기존 W 데모는 hover에서 lateral/yaw disturbance를 예약하는 제한된 script 기능이다. VLA action에 적용하는 모듈은 다음 작업에서 만든다.

## Environment failures

현재 Blocks에서 아래 조작 API·scene 변경을 검증하지 않았다. 새 map이나 simulator를 도입하기 전에 기존 환경에서 가능한지 확인한다.

| Failure | What happens | How to simulate, 후보 | Difficulty | Demo value |
|---|---|---|---|---|
| Unexpected Obstacle | 갑자기 장애물이 나타난다 | 기존 scene에서 물체 배치/이동 가능 여부 확인 | 높음 | 매우 높음 |
| Target Occlusion | 목표물이 물체 뒤에 숨는다 | 목표 앞에 가림 물체 배치 | 높음 | 높음 |
| Target Movement | 목표가 다른 위치로 이동한다 | scene object 이동과 instruction 갱신 | 높음 | 매우 높음 |
| Blocked Path | 통로가 막힌다 | 통로를 object로 막고 안전 경로 확보 | 높음 | 높음 |
| Wrong Landmark | 잘못된 표식을 따라간다 | 비슷한 표식과 지시 구성 | 높음 | 높음 |

## 첫 세 후보 비교

난이도·재미는 개발 판단이다. VLA 영향은 아래 가설이며 아직 비교 실행하지 않았다.

| 후보 | 구현 | 화면 재미 | Project AirSim 적용 | 예상 VLA 영향 | 추천 |
|---|---|---|---|---|---|
| Gaussian Blur | 쉬움 | 좋음 | camera frame에 적용, 기존 prototype 참고 | 물체/방향 인식이 흐려질 수 있음 | **첫 구현** |
| Partial Occlusion | 쉬움 | 좋음 | camera frame에 적용 | 가려진 영역에 따라 판단이 달라질 수 있음 | 두 번째 |
| Control Drift | 쉬움~보통 | 매우 좋음 | 기존 motion API와 안전 제한 활용 | 출력은 같아도 실제 이동이 빗나갈 수 있음 | 세 번째 |

**첫 작업은 Gaussian Blur 주입기**다. 기존 카메라 경로에 작은 모듈로 붙이고 비행 명령은 유지할 수 있어 baseline 보존이 쉽다. 원본/변경 영상과 ON/OFF 상태를 즉시 보여줄 수 있다. 다음 작업에서 `src/failures/`에 재사용 가능한 주입기와 토글 연결을 만들고, 한 효과부터 확인한다.

## Interactive demo 아이디어

향후 통합 창의 단축키 후보다. 현재 model-free 창에는 B/W 일부 기능만 있고, 아래 전체 인터페이스는 아직 없다.

```text
B → Blur ON/OFF
O → Partial Occlusion ON/OFF
W → Wind-like / Control Drift ON/OFF
R → Failure 설정 Reset
```

R은 우선 장애 설정만 초기화하는 후보다. Drone pose/scene 재시작은 별도 조작으로 구분할 수 있다.

```text
Current Failure: CONTROL DRIFT
AeroVLA Action: Forward ... / Down ... / Yaw ...
Drone State: Position ... / Heading ...
```

raw 모델 행동과 제한된 실제 명령을 구분해 보여준다. 장애가 켜진 동안의 상태 표시와 원본/주입 영상 비교를 우선한다.

향후 연결:

```text
Camera → Failure Injection → AeroVLA → Drone
                                       ↓
                              Failure Detection → Recovery
```

자동 감지·진단·복구는 주입 데모 다음 단계다. 이번 작업에서는 새 failure 기능, Detector, Recovery, 재학습, benchmark 또는 복잡한 metric을 구현하지 않는다.
