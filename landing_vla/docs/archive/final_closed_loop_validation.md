# Final Integration Validation

2026-10-05 KST. **Windows Project AirSim live camera → WSL AeroVLA NF4 → safe action → actual drone movement가 single-step과10/10 loop에서 성공했다.** 요청된 feasibility 실험은 종료했다. 이전 실패 문서는 과거 결과로 보존했다. Failure-aware 구현, 추가 episode, 학습/benchmark 또는 새 모델/simulator 다운로드는 하지 않았다.

## Gate A — Direct reconnect

- 20-run reconnect: **20/20**
- Idle reconnect: **5/5**, disconnect 후60s
- Result: **✅ PASS**
- Route: WSL→Windows host192.168.160.1, native direct NNG8989/8990, relay 사용 없음
- Root cause: 직접 경로가 현재 blocker를 해소한 것은 검증. 이전 relay 내부 실패 원인은 미확정.

A1 평균 connect3.248ms, topic initialization488.894ms, Front47.475ms, state0.733ms. [상세 통신 기록](communication_stability_test.md).

## Gate B — Resource pressure

RAM2/4GiB 단계에서 각2/2, GPU7.000GiB에서5/5 reconnect 성공. RAM8/12/16GiB는 commit margin 때문에 **NOT RUN**. Communication failure correlated: **NO within tested range**, 높은 RAM 단계는 미확정. Result: **⚠️ tested levels PASS / larger RAM levels SKIPPED**.

## Gate C — Windows Gateway

Used: **NO**. Observation/action0, failure/latencyN/A. **NOT REQUIRED**. Native direct로 뒤의 두 gate까지 통과했다. [Gateway decision](gateway_evaluation.md).

## Gate D — Single-step VLA

| 기준 | 실제 결과 |
|---|---|
| Live observation | ✅ fresh Front/Down256×256 BGR |
| AeroVLA inference | ✅ existing OpenVLA7B NF4 + AeroVLA LoRA |
| Parsed action | ✅ `95 49 49` |
| Action transport / execution | ✅ move/hover returnsTrue |
| Drone movement | ✅ **0.336016m** |
| Timeout / OOM | **NO / NO** |
| Result | **✅ PASS** |

Raw action: forward4.846939m/down0m/yaw0rad. Safety clipping: forward0.5m, current heading-45°, NED displacement `[+0.353553,-0.353553,0]`, duration1s. Clipped command만 실행했다.

Generation1727.996ms, observation block111.671ms, command wall1007.238ms. Command wall에는1s 실제 이동이 포함되며 순수 network latency가 아니다. Instruction/target, state_before/execution/after, raw text, parsed/clipped action, images/SHA, latency, GPU/Windows RAM을 `closed-loop.json`/`closed-loop-log.jsonl`에 저장했다.

## Gate E —10-step closed loop

| 지표 | 실제 측정 |
|---|---:|
| Completed | **10/10** |
| Average generation | **1047.559ms** |
| Median / p95 generation | 1054.333 /1106.270ms |
| Average observation transport block | **104.478ms** |
| Average command wall, actual yaw/move 포함 | **2729.768ms** |
| Average entire step wall | **4527.246ms** |
| Effective serial decision cadence | **0.221Hz** |
| Sum of10 step displacement lengths | **3.062978m** |
| Timeout / OOM / simulator crash | **NO / NO / NO** |
| NaN / abnormal state / in-flight collision | **NO / NO / NO** |
| Move/hover/yaw returns | **모두True** |
| Final landed state / cleanup error | **0 / none** |
| Result | **✅ functional loop PASS** |

제안된 약1Hz 시작 주기는 **이번 순차 실행에서 달성하지 못했다**. 약1s generation 후 이동1s, 필요한 yaw 완료, hover/안정화와 logging을 기다린다. Nonzero yaw의 command 완료는 약4.45s였다. Simulator 내부 controller가 decision 사이의 physics를 처리한다.20Hz VLA 제어는 시도하지 않았다. Scheduling 개선은 남은 engineering task이다.

| Step | Raw bins | Actual displacement m |
|---|---|---:|
| 1 | 95 49 49 | 0.241475 |
| 2 | 96 49 49 | 0.246071 |
| 3 | 96 49 49 | 0.253110 |
| 4 | 25 49 98 | 0.386991 |
| 5 | 95 49 49 | 0.203609 |
| 6 | 57 49 49 | 0.220599 |
| 7 | 52 35 12 | 0.469656 |
| 8 | 25 49 98 | 0.392155 |
| 9 | 9 49 0 | 0.396184 |
| 10 | 00 39 00 | 0.253128 |

Forward≤0.5m, vertical≤±0.3m, yaw≤±15°를 유지했다. Generation 뒤 state를 다시 읽고 absolute command를 계산했다. 초기 platform 기준 clearance0.8–4m와 bounded state-monitor tolerance를 적용했다. Nonfinite/out-of-envelope state는 명령을 거부한다. 새 altitude/NaN 회귀 테스트는 수정 전 실패, integration-layer guard 추가 후 통과. 전체 adapter suite5/5.

Collision은 official topic으로 감시했다. [v1.0.1 message definition](https://github.com/iamaisim/ProjectAirSim/blob/v1.0.1/core_sim/src/message/collision_info_message.cpp)은 event `time_stamp`/object/impact 정보를 가지며 `has_collided` boolean은 없다. 기록된2건은 초기 platform 및 cleanup 착륙 contact이며 flight interval5.829–54.534s 밖이다. **비행 중 event0건**. Ground contact를 삭제하지 않았다.

## Resource Measurement

RTX5070 driver591.86, total12227MiB=11.940GiB. 기존 고정 CUDA/NF4 환경과 checkpoint를 재사용했다. 모델 전체 `.to()`, CPU offload, LoRA merge, version 교체는 하지 않았다.

| Pipeline/loop scope | Peak / minimum |
|---|---:|
| Combined global GPU | **9990MiB /9.756GiB** |
| Remaining GPU at peak | **2237MiB /2.185GiB** |
| Model PyTorch peak allocated / reserved | **6.937 /7.145GiB** |
| Simulator dedicated GPU peak | **0.638GiB** |
| Simulator shared GPU peak | 0.171GiB, VRAM과 구분 |
| Windows system RAM peak | **30.066GiB** |
| Windows physical available minimum | **1.048GiB** |
| Windows committed bytes peak | **54.705GiB** |
| Pagefile actual current-use peak | **1005MiB /0.981GiB** |
| Pagefile allocated-capacity peak | **25279MiB /24.687GiB** |
| WSL process RSS peak | 7.430GiB, loading mmap 포함 |
| WSL psutil used peak / available minimum | 2.415 /12.766GiB |
| WSL swap used peak | 0.762MiB |

PyTorch peak는 loading 전 reset하여 **loading+flight**를 포함한다. 독립 inference-only peak가 아니며 이전 단독 inference6.801GiB 측정은 그대로 보존했다. Global GPU에는 desktop/context/other process도 포함한다. 이번 simulator workload는 이전0.919GiB demo와 달라 수치가 다르며 과거 raw 결과를 대체하지 않는다.

Pagefile capacity24.687GiB는 **실제24.687GiB swapping을 의미하지 않는다**. Current-use peak는0.981GiB이다. Windows managed capacity가 변화했지만 수동 설정은 하지 않았다. RAM/commit/WSL counters는 회계 범위가 다르므로 서로 대체하지 않는다. Host physical 여유가 약1GiB뿐이므로 장시간 실행/추가 프로그램에 대한 여유는 제한적이다.

Sampling: Windows 약1s+CIM latency, WSL 약0.5s. Labelled pipeline44/loop29 Windows samples, model-to-cleanup154 WSL samples. 더 짧은 peak를 놓칠 수 있다.

## Domain Shift Quick Check

- Unique scenes: **1 Blocks environment /10 distinct live Front/Down pairs**.10개 독립 map이 아니다.
- Unique raw actions: **7**.
- Repeated-action ratio: **20%**, 가장 빈번한 exact bin triplet의 횟수/10으로 정의.
- Observation: **Healthy variation / not obviously degenerate within this sequence**.

Forward/yaw와 일부 vertical 값이 변했으며 모든 frame에서96 49 49가 반복되지 않았다. Frame SHA는 perceptual diversity 자체가 아니며 다른 실제 pose/저장 view도 보존했다. 하나의 synthetic target/instruction 시퀀스이다. Navigation quality, goal success, quantization accuracy 또는 TravelUAV benchmark 성능은 결론 내리지 않는다.

## Final Decision

```yaml
Project AirSim + AeroVLA: ⚠️ FEASIBLE WITH LIMITATIONS
Functional live pipeline: ✅ single-step +10/10 loop
Recommended integration: Direct WSL client → Windows Project AirSim
Windows Gateway: NOT REQUIRED
Failure-aware research phase: START (next task: research design only)

Remaining engineering blockers:
  1: Host RAM/commit margin 및 resource headroom
  2: Serial cadence0.221Hz; 제안된1Hz는 아직 미달
  3: Current host IP / single client ownership / unattended lifecycle

Remaining research risks:
  1: Blocks→TravelUAV domain gap; native TravelUAV benchmark 미검증
  2: BF16-vs-NF4 action/navigation accuracy 비교 없음
  3: Synthetic target/instruction와 clipped dynamics는 benchmark ground truth가 아님

Next recommended task: Failure Taxonomy / Injection / Detection / Diagnosis / Recovery / Evaluation 설계
```

START는 다음 연구 설계를 진행할 수 있다는 판정이다. 이번 작업에서 해당 연구 구현이나 추가 environment episode를 진행하지 않았다.

## Artifacts / cleanup

- `src/integration/closed_loop_runner.py`: final native direct runner, conditional single→10, collision/state checks.
- 기존 NF4 loader/observation preprocessing은 변경하지 않았다. Action guard만 integration layer에서 보완했다. Upstream hashes unchanged.
- `outputs/communication_final/closed-loop.json`, `closed-loop-log.jsonl`, `final-summary.json`, 원시 client/resource logs와 socket snapshots.
- `front_single/down_single/debug_single.png`, `front_01..10/down_01..10/debug_01..10.png`, mosaics/`debug_latest.png`.
- Simulator/debug viewer를 표시했고 저장 panel도 직접 확인했다. Window evidence는 venv launcher의 child Python named-window handle로 확인했다.
- Drone land/disarm, client disconnect, model process exit 완료. Simulator, monitor, viewer launcher/child 종료. Checkpoint/cache/runtime 보존. 완료된 feasibility gate는 자동 반복하지 않는다.
