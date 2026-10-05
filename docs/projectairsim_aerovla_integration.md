# Project AirSim × AeroVLA adapter 및 통신 검증

> **후속 결과:** 최초 reverse relay 실패 뒤 direct WSL2 연결로 전환해 reconnect 20/20, idle 5/5 및 live 10-step을 통과했다. [최종 통합 검증](final_closed_loop_validation.md), [통신 검증](communication_stability_test.md). 아래는 최초 통합 시도의 기록이다.

2026-10-05 KST. **카메라·상태·조종의 최초 WSL probe는 통과. 모델 로딩 뒤 재연결은 scene topic timeout으로 실패.** 따라서 통신이 안정적이라고 판정하지 않는다. 이전 문서는 수정하지 않고 이번 추가 검증을 별도로 기록했다.

## 실제 구성과 통신

Windows Project AirSim Blocks v1.0.1 (Unreal5.2), Windows GPU rendering + WSL2 Ubuntu Python. 이는 TravelUAV map 또는 final benchmark 환경이 아니다. 이번 Python client는 `projectairsim==1.0.2`, 공식 통신은 NNG Pair0 topics/Req0 services이다. Legacy AirSim의 msgpack RPC client와 호환된다고 가정하지 않았다.

```text
WSL Python local18989/18990
  ↕ WSL reverse listener41510/41511
Windows가 WSL localhost-forwarding으로 연결 시작
  ↕ Windows helper TCP stream
Windows Project AirSim8989(topics)/8990(services)
```

기존 WSL→Windows 직접 inbound 경로의 차단을 피하려고 이 실험용 TCP relay를 사용했다. OS 방화벽·WSL networking mode·upstream simulator/client는 수정하지 않았다. **직접 NAT 연결 성공이나 production bridge 안정성을 의미하지 않는다.**

첫 실행 Simulator PID49136, 재실행 PID23192. Helpers는 hidden window로 실행하고 simulator/debug viewer는 사용자가 볼 수 있도록 실행했다. 테스트 후 생성한 모든 simulator/helper/monitor/viewer 및 WSL relay를 종료했다. 이번 실패는 화면용 debug frame 생성 전이므로 실제 end-to-end debug display를 확인했다는 주장도 하지 않는다.

| 최초 WSL probe | 결과 | 측정 |
|---|---|---:|
| FrontCamera scene RGB | PASS, BGR encoding, 256×256×3 | 46.983ms |
| DownCamera scene RGB | PASS, BGR encoding, 256×256×3 | 84.382ms |
| ground truth kinematics | pose/quaternion/linear velocity 수신 | 1.102ms |
| API control/arm | True | — |
| takeoff | 반환 False, 실제 상승 | 1.403m / flying state1 |
| forward body velocity | 0.5m/s × 1s, 반환 True | XY 이동 0.321384m |
| hover/land/disarm | 실행, 최종 landed state0 | land True |

Latency는 각각 **1회 호출**이며 앞선 100-frame simulator 단독 통계와 합치지 않는다. 첫 연결 시간은 11.365ms였다. 상태 before/after 전체와 프레임 SHA는 `communication.json`에 있다.

## takeoff=False의 사용 기준

전후 NED z는 -2.692681789 → -4.095816612. NED의 +z는 아래이므로 상승량은 `z_before-z_after=1.403134823m`. `get_landed_state()`는 takeoff 뒤1, 착륙/disarm 뒤0이었다. 그러므로 연구용 launch success를 **실제 climb>0.5m 및 airborne state**로 검사하는 방식은 가능하다. Hover 후 속도/고도 안정화도 확인해야 한다. 한 번 상승한 사실만으로 목표 고도 도달 또는 takeoff 반환 False의 원인을 규명했다고 할 수 없다. Upstream patch는 하지 않았다.

## Observation adapter — 정적/저장 프레임 검증 PASS

구현: [projectairsim_observation_adapter.py](../src/integration/projectairsim_observation_adapter.py).

1. `FrontCamera`/`DownCamera`의 BGR uint8 이미지에 정확히 BGR→RGB 적용.
2. 각 PIL RGB 이미지를 BICUBIC 224×224로 resize.
3. front를 위, down을 아래에 붙여 **224×448** 세로 mosaic 작성.
4. Pinned OpenVLA `AutoImageProcessor`를 그대로 적용하여 실제 **[1,6,224,224]** 확인. 6-channel은 front/down을 별도 채널로 쌓은 결과가 아니라 DINOv2/SigLIP 두 backbone의 동일 mosaic transform이다.
5. Simulator pose의 `position`은 `[x,y,z]`, quaternion key는 `[x,y,z,w]`로 명시적으로 추출한다. dict 저장 순서를 quaternion 순서로 사용하지 않는다. velocity도 `[x,y,z]`로 기록한다.

Prompt는 [AeroVLA wrapper](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/src/model_wrapper/aerovla_wrapper_ui.py)의 template와 instruction split을 유지했다.

```text
<image>
Fly straight ahead and find the target. Find the colored blocks ahead.
Action:
```

실제 prompt의 마지막 `Action:` 뒤에는 공백 한 칸을 붙인다.

Instruction의 `degrees from you.` 이후부터 ` Please control` 이전까지를 object description으로 추출한다. Target position은 모델에 수치 tensor로 주입하지 않고 `target-current` world vector를 quaternion inverse로 body frame에 변환해 semantic direction을 생성한다.

```text
vector_world = target_position - current_position
vector_body = R_xyzw(current_orientation)^(-1) * vector_world
angle = atan2(vector_body.y, vector_body.x)
```

방향 구간은 원본의 ±15/60/120/180° 분기를 유지한다. 이번 target은 최초 heading 앞3m에 설정하려던 **통합 실험용 synthetic target**이며 실제 scene landmark/TravelUAV ground truth로 검증된 위치가 아니다. 단독 추론에서는 저장 pose 앞 약3m를 사용했다. 통합 실행은 초기화 실패로 target 설정에 도달하지 못했다.

## Action adapter — 정적 테스트 PASS, 실제 VLA 조종 미실행

구현: [projectairsim_action_adapter.py](../src/integration/projectairsim_action_adapter.py). [원본 AeroVLA simulator client](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/airsim_plugin/AirVLNSimulatorClientTool_AeroVLA.py)의 거리/상대 yaw 의미를 유지하고 [Project AirSim 공식 API](https://github.com/iamaisim/ProjectAirSim/blob/v1.0.1/client/python/projectairsim/src/projectairsim/drone.py)의 NED velocity 및 rad yaw로 변환한다.

| 의미 | TravelUAV/legacy AirSim | Project AirSim 실험 |
|---|---|---|
| World axes | NED: X north, Y east, Z down | 동일 NED; origin 수치는 환경별로 다름 |
| Body axes | X forward, Y right, Z down | 동일 해석 |
| Quaternion | scipy/API 변환 시 xyzw | RPC dict의 x,y,z,w 명시 추출 |
| yaw + | NED에서 위에서 보면 시계 방향 | 동일 부호 |
| AeroVLA yaw | 현재 heading에 더하는 상대 rad | absolute target rad로 변환 |
| legacy rotate API | degree 인수 | Project `rotate_to_yaw_async`는 rad |
| forward/down | 예측 **거리(m)** | duration1s에 대해 velocity=distance/duration |

99-bin의 유효한 정수 `b ∈ [0,98]`를 원본대로 역양자화한다.

```text
forward = 5*b_f/98
down = -5 + 10*b_d/98
yaw = -1.1 + 2.2*b_y/98

f = clip(forward, 0, 0.5m)
d = clip(down, -0.3m, +0.3m)
delta_yaw = clip(yaw, -15°, +15°)  # 이후 rad로 유지
target_yaw = wrap(current_yaw + delta_yaw)
delta_world = [f*cos(target_yaw), f*sin(target_yaw), d]
velocity_world = delta_world / 1s
```

실행 순서는 필요 시 공식 `rotate_to_yaw_async(target_yaw, yaw_rate=.3rad/s)` → `move_by_velocity_async(vx,vy,vz, duration=1, yaw_is_rate=False, yaw=target_yaw)` → hover. Legacy controller와 완전히 같은 dynamics를 재현한다고 주장하지 않는다.

Z target은 초기 정지 platform의 `ground_z` 기준 clearance 0.8–4m 안으로 제한한다. 이 값은 terrain-ray AGL이 아니며 이동한 지형의 고도를 추적하지 않는다. 최초 takeoff clearance가 범위 안인 이번 실험을 전제로 한다. 범위를 이미 벗어난 state에 대한 추가 guard/vertical bound 검증과 collision 검증은 아직 필요하다.

LAND 문자열 또는 `(0,49,49)`는 stop으로 파싱한다. 초기 smoke에서는 stop을 hover로 매핑하고 종료 cleanup에서 land/disarm한다. 원본이 malformed output을 0으로 처리하는 것과 달리, 실험 파서는 세 유효 정수가 아니면 오류로 기록한다. 모델이 실제로 LAND를 출력했다는 증거는 없다.

## 재연결 실패 — 여기서 중단

두 번째 실행에서 NF4+LoRA가 CUDA에 로딩된 뒤 simulator의 scene-load 요청까지 전달됐다. 이후 `/$topics` subscription callback이 60초 안에 업데이트되지 않았다.

```text
01:40:11.181 Connection opened.
01:40:11.181 Loading scene config: scene_basic_drone.jsonc
01:40:11.972 Getting the list of available topic info...
01:41:12.119 Timeout waiting to get topic info, disconnecting client.
RuntimeError: Timeout waiting to get topic info from sim server.
```

Client 코드의 `get_topic_info()`는 Pair0에서 `/$topics`를 subscribe하고 callback flag를 기다린다. 서비스 socket 연결이 열렸다는 사실만으로 topic 연결 성공을 판정할 수 없다. Relay 로그에는 두 port의 여러 `SESSION_CLOSED`/재연결이 있다. **최종 원인은 확정하지 못했다.** 모델 로딩 전부터 연결된 relay의 stale session/NNG handshake·재연결 순서와 host RAM 압박은 다음에 분리 검증할 후보이며 확인된 원인으로 기록하지 않는다.

통신 실패가 사용자 지정 중단 조건이므로 추가 flag/version 변경, simulator 재실행, 우회 service-only 조종을 시도하지 않았다. Single-step/10-step/Failure-aware 구현은 하지 않았다.

## 검증 및 원시 파일

- 4개 adapter test: BGR/RGB와 mosaic 순서, NED quaternion/방향, valid/invalid bin·LAND, 거리·고도 clipping. 실제 GPU inference는 별도 측정이다.
- `scripts/projectairsim_wsl_probe.py`: 최초 WSL camera/state/control 실행
- `scripts/integration_reverse_bridge.py`, `.ps1`: 이번에 사용한 experimental route
- `scripts/run_aerovla_projectairsim.py`: 단일→10step 조건부 runner, 이번 실행은 초기화에서 중단
- `communication.json`, `front_communication.png`, `down_communication.png`
- `end-to-end.json`, `end-to-end.log`, `e2e-reverse-bridge-wsl.log`, `e2e-bridge-*.log`
- `integration_log.jsonl`: failure event 및 원시 측정 요약. 나중에 append한 summary event는 측정 시간순이라고 해석하지 않는다.
