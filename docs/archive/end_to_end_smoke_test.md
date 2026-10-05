# AeroVLA × Project AirSim Integration Result

> **후속 결과:** 아래 최초 reverse relay 통합 실패 이후 direct WSL2 연결에서 live single-step과 10/10 반복 실행에 성공했다. 현재 판정은 **FEASIBLE WITH LIMITATIONS**이며 [최종 통합 검증](final_closed_loop_validation.md)에 기록했다. 이 문서는 이전 실패 결과를 보존한다.

2026-10-05 KST. **실제 7B NF4+LoRA 실행은 성공했지만 최종 통합 초기화에서 통신 timeout으로 중단했다.** 아래 PASS는 각 측정 범위에 한정된다. 이전 플랫폼 문서는 보존했다.

## 1. Windows ↔ WSL communication

Result: **⚠️ 최초 probe PASS / 통합 재연결 FAIL**

| 항목 | 최초 WSL probe | 최종 통합 실행 |
|---|---|---|
| Front RGB | ✅ 256×256 BGR, RPC46.983ms | 미도달 |
| Down RGB | ✅ 256×256 BGR, RPC84.382ms | 미도달 |
| State RPC | ✅ pose/quaternion/velocity, RPC1.102ms | 미도달 |
| Control RPC | ✅ forward0.5m/s×1s → XY0.321m; hover/land | 미도달 |
| Scene topic initialization | ✅ 최초 성공 | ❌ 60s timeout |

Latency는 단일 호출 측정이다. Windows가 연결을 시작하는 두-port reverse TCP route를 사용했다. 직접 WSL→Windows NAT 및 안정적인 재연결은 검증되지 않았다. Takeoff False에도 실제 상승1.403m, airborne1, 최종 landed0을 확인했다.

## 2. AeroVLA INT4

Result: **✅ 단독 로딩/생성/파싱 PASS**

- Base: `openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f`
- LoRA: `XuPeng23/AerialVLA@196f2f3253b69df6e90ac10b6ae041c7b3a9569e/aero_vla`
- Precision: language model224 Linear4bit, NF4 double quant + BF16 compute. Vision/projector/head 유지, FP32 parameter 일부 존재.
- Device map: `{"":0}`, 모든 parameter cuda:0. 모델 전체 `.to()`/LoRA merge/CPU offload 없음.
- Embeddings: 32064→32001, adapter 로딩 전 resize 성공.

| 측정 | 값 |
|---|---:|
| 로딩 직후 allocated / reserved | 6.503 / 7.061 GiB |
| Inference peak allocated / reserved | 6.801 / 7.145 GiB |
| 첫 inference | 1418.046ms |
| 이후10회 mean / median / p95 | **733.371 / 751.497 / 883.571ms** |
| 유효 동작 | 11/11; 동일 저장 프레임의 `96 49 49` |

범위는 generation이며 live image RPC나 action execution은 포함하지 않는다. 상세는 [aerovla_int4_validation.md](aerovla_int4_validation.md).

## 3. Observation Adapter

Result: **⚠️ 저장된 실제 프레임 + 모델 단독 PASS; live 통합 미검증**

Front/Down은 공식 camera RPC로 획득했다. BGR→RGB, 각각 PIL BICUBIC224², front위/down아래224×448 mosaic, official processor [1,6,224,224]를 확인했다. Position=NED xyz, orientation=xyzw. Target은 synthetic integration-only 위치이며 semantic body direction 생성에만 사용한다. Instruction/prompt는 upstream delimiter/template를 유지했다. Scene landmark ground truth가 있는 navigation task가 아니다.

## 4. Action Adapter

Result: **⚠️ 정적 테스트 PASS; VLA 출력으로 실제 조종 미실행**

99-bin forward/down distance(m), relative yaw(rad)를 현재 NED heading으로 회전한 world displacement로 변환하고 duration1s로 나눠 공식 world-velocity API에 전달하도록 구현했다. Yaw는 상대값을 absolute rad로 변환한다. Forward≤0.5m, vertical≤±0.3m, yaw≤±15°, 초기 platform 기준 clearance0.8–4m. LAND/stop은 초기 smoke에서 hover로 처리한다. 좌표·순서·제한 테스트는 통과했지만 실제 VLA→command 실행 동등성은 아직 없다. 식과 제한은 [projectairsim_aerovla_integration.md](projectairsim_aerovla_integration.md)에 기록했다.

## 5. Single-step End-to-End

Result: **NOT RUN / ❌ 성공 기준 미충족**

| 기준 | 이번 단일 통합 실행 |
|---|---|
| Live observation | NOT RUN, topic 초기화에서 중단 |
| AeroVLA inference | NOT RUN (단독은 별도 PASS) |
| Valid VLA action 전달 | NOT RUN |
| 해당 action 때문에 drone state 변화 | NOT RUN |

STEP1의 수동 forward로 얻은 이동을 VLA로 만든 이동으로 기록하지 않았다.

## 6. 10-step Closed Loop

Result: **NOT RUN**. Completed steps: **0/10**.

Crash/OOM: **통합 runner는 timeout으로 FAIL/exit1; OOM은 NO**. Simulator crash 증거는 없다. 사용자의 통신 실패 중단 조건에 따라 추가 실행은 하지 않았다.

오류 전문:

```text
Traceback (most recent call last):
  File "$REPO_ROOT/scripts/run_aerovla_projectairsim.py", line 78, in main
    world=World(client,'scene_basic_drone.jsonc',delay_after_load_sec=2,sim_config_path=str(CONFIG))
  File "$UAV_VLA_HOME/integration/lib/python3.10/site-packages/projectairsim/world.py", line 72, in __init__
    self.load_scene(config_dict, delay_after_load_sec=delay_after_load_sec)
  File "$UAV_VLA_HOME/integration/lib/python3.10/site-packages/projectairsim/world.py", line 1073, in load_scene
    self.client.get_topic_info()  # get new scene's list of registered topic info
  File "$UAV_VLA_HOME/integration/lib/python3.10/site-packages/projectairsim/client.py", line 123, in get_topic_info
    raise RuntimeError(
RuntimeError: Timeout waiting to get topic info from sim server.
```

Confirmed: `/$topics` callback flag가60s 내 갱신되지 않음; relay 세션이 여러 번 닫히고 재연결됨. Root cause: **UNKNOWN**. Stale relay pairing/NNG reconnect timing 또는 host RAM 압박은 가설이다. CUDA/NF4 실패나 OOM으로 원인을 바꾸어 해석하지 않는다.

## 7. VRAM / RAM

| 범위 | 실제 측정 |
|---|---:|
| RTX5070 total (nvidia-smi) | **12227MiB / 11.940GiB**, 제품표기12GB |
| Project AirSim 단독 dedicated peak (이전 플랫폼 시험) | **0.919GiB**; 당시100-frame RPC+비행 |
| 이번 simulator dedicated peak | **0.676GiB**, loading/scene-init 구간만 |
| 이번 simulator shared peak / 3D engine peak | 0.161GiB / 86% |
| AeroVLA 단독 inference allocated / reserved peak | 6.801 / 7.145GiB → tensor 기준 GREEN |
| 모델 단독 구간 GPU 전체 peak | 9272MiB / 9.055GiB (desktop/context 포함) |
| **Combined loading/scene-init 전체 peak** | **9912MiB / 9.680GiB** |
| 그 순간 remaining | **2315MiB / 2.261GiB** |
| Combined inference/flight peak | **NOT MEASURED** |
| Combined host Windows 전체 RAM peak | **30.957GiB** |
| Combined 모델 프로세스 RSS peak | 7.128GiB (loading의 mmap 포함) |

Dedicated와 shared는 구분하고 합산 값을 VRAM으로 표시하지 않았다. GPU 전체 사용량에는 desktop/other process/context가 포함된다. Resource sampling은 WSL0.5s, Windows 약1s이며 더 짧은 peak를 놓칠 수 있다. Windows 전체 RAM은 다른 작업과 file cache도 포함한다. Memory load 공존은 성공했지만 전체 파이프라인 공존은 **⚠️ 미검증**이다.

## 8. Domain Shift Observation

단독 출력: **형식은 유효, quality는 판정 불가**. 저장된 Project AirSim Blocks 프레임·동일 prompt에서 11회 모두 `96 49 49` → forward4.898m/down0/yaw0이었다. 동일 입력 greedy repeat이므로 반복 출력만으로 degenerate navigation이라고 판정하지 않는다. Live pose 변화·landmark 도달·task success는 측정하지 않았다. **No performance conclusion yet.**

## Final Decision

```yaml
Project AirSim + AeroVLA integration: ❌ NOT FEASIBLE (현재 통합 구성의 성공 기준 미충족)
RTX 5070 12GB coexistence: ⚠️ (동시 로딩 성공; live inference+flight 미검증)
Failure-aware research implementation 진행: NO

가장 큰 남은 문제:
  1: scene reload 이후 topic/NNG 재연결 안정성
  2: Windows 전체 RAM 여유 및 실제 추론·비행 공존 peak 미확인
  3: 동작 adapter의 실제 VLA 제어 검증과 Blocks domain shift 미평가

다음 추천 작업:
  1: 모델 추론을 제외하고 relay 연결 순서와 scene reload 재연결을 반복 검증
  2: Windows native client가 simulator 연결을 소유하는 관측/명령 gateway 대안 비교
  3: 통신 안정화 후 이번 checkpoint로 single-step → 10-step 및 공존 메모리 재검증
```

NOT FEASIBLE은 이번 실행의 판정이며 RTX5070에서 모델 실행 자체가 불가능하거나 연구 프로젝트를 중단해야 한다는 결론이 아니다. GPU/RAM/NNG의 원인을 혼동하지 않고, 먼저 통신 한 가지를 해결하는 다음 작업이 필요하다. CPU offload·버전 변경·추가 다운로드·Failure 연구는 수행하지 않았다.

## 산출물과 정리

- 요청한 세 문서 생성, 기존 문서 보존.
- `src/integration/` 별도 loader/observation/action adapter, upstream 수정 없음.
- `outputs/integration/communication.json`, `model-validation.json`, `end-to-end.json`, `measurement-summary.json` 및 원시 로그/메모리 sampling 보존.
- `front_communication.png`, `down_communication.png`는 실제 STEP1 프레임. E2E `front_single/debug_single`은 실행 미도달로 생성되지 않았다.
- `debug_communication.png`는 종료 후 저장 프레임으로 만든 진단 자료이며 live closed-loop 화면이 아니다.
- Simulator, model process, relay helpers, resource monitor, viewer 종료. Checkpoint/cache와 환경은 보존했다.
