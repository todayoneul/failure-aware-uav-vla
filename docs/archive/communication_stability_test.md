# Communication Stability Final Validation

2026-10-05 KST, 약02:09–02:23. **Native direct WSL client가 reconnect/idle/resource-pressure 시험을 통과했다.** 이전 reverse NNG relay는 사용하지 않았다. 기존 simulator, checkpoint와 라이브러리 버전을 유지했다.

## Gate A — Direct reconnect

Windows Project AirSim v1.0.1 Blocks의 topics8989(NNG Pair0)/services8990(NNG Req0)에 WSL official client1.0.2가 직접 연결했다. 실측 Windows host는192.168.160.1, WSL source는192.168.175.49이다. 주소가 영구적이라고 가정하지 않는다. Firewall/WSL networking/upstream 설정을 변경하지 않았다.

`scripts/communication_reconnect_probe.py`는 각 시도마다 TCP preflight → 새 client connect → scene reload와 topic initialization → FrontRGB → kinematics → disconnect/client destroy를 수행했다.

| Test | 실제 시도 | 성공 | 실패 |
|---|---:|---:|---:|
| A1, disconnect 후5s 간격 | 20 | **20** | **0** |
| A2, disconnect 후60s idle | 5 | **5** | **0** |

A2의 initial connection도5/5 성공했다. 실제60s sleep 후 reconnect했다. 저장된 `idle_seconds`는 timer 종료가 reconnect 후이므로 **idle+reconnect**60.644–60.735s이다.

| A1 latency, n20 | Mean ms | Median ms | p95 ms |
|---|---:|---:|---:|
| NNG connect | 3.248 | 2.780 | 4.132 |
| Scene reload + topic initialization | 488.894 | 478.114 | 561.692 |
| Front RPC + unpack | 47.475 | 47.099 | 49.729 |
| State RPC | 0.733 | 0.656 | 1.006 |

Probe의 `delay_after_load_sec=0`, final flight는2s이다. Topic 시간은 scene loading도 포함한다. 원래60s timeout을 늘리거나 library를 patch하지 않았다. Front256×256 BGR, position/orientation/velocity를 실제 수신했다.

### 원인 판정

이전 실패는 모델 로딩 후 **experimental reverse relay** 경로의 topic 초기화에서 발생했다. 이번에는 native direct host TCP를 사용하고 모델 로딩 후 새 NNG client를 연결했다. A1/A2와 실제 loop 성공으로 현재 실행 blocker가 해소된 것은 검증됐다. **이전 relay의 정확한 내부 실패 원인은 미확정**이다. Stale pairing/handshake timing은 가설이며 packet trace/relay 단독 수정은 하지 않았다. CUDA/OOM/RAM을 확인된 원인으로 기록하지 않는다.

### Lifecycle/networking 자료

- `gate-a.json`, `gate-a-trials.jsonl`, `gate-a.log`: 모든 시도, stage latency, success/error, pose와 camera timestamp.
- `windows-ips.json`, `firewall-profiles.json`, `windows-tcp-*.json`: host IP, firewall profile, port 소유/연결/TIME_WAIT 상태 snapshots. Profile의 enum0만으로 실제 방화벽 allow/deny를 추정하지 않는다.
- `wsl-sockets-during-a.txt`, `wsl-sockets-final.txt`: `ss -anp`, route/process 상태.
- `client-combined.log`: official client 로그. Probe는 정상 종료했고 task의 orphan process는 최종 정리 시 남지 않았다.
- 확인한 packaged Saved/Logs와 user Blocks/Saved/Logs 경로에서 readable Unreal/PAServer log는 제공되지 않았다. Simulator log는 **unavailable**, client/socket/resource 기록으로 보완했다.

초기 inline shell preflight 명령은 quoting SyntaxError 때문에 실행되지 않았다. 실제 네트워크 시도/성공 count에 포함하지 않았고, 이후 file-based probe로 위 시험을 수행했다.

## Gate B — Resource pressure

Gate A PASS 이후, AeroVLA를 로딩하지 않은 상태에서 수행했다. RAM은 numpy uint8 arrays를 실제 할당·touch·유지했으며 각 tested level에서 full reconnect/image/state를2회 수행했다.

| Target | 실제 유지 | Reconnect | 결과 |
|---|---:|---|---|
| 2GiB | 2GiB | **2/2** | PASS |
| 4GiB | 4GiB | **2/2** | PASS |
| 8GiB | 추가 할당 없음, 기존4GiB 유지 | NOT RUN | SKIPPED_HEADROOM |
| 12GiB | 동일 | NOT RUN | SKIPPED_HEADROOM |
| 16GiB | 동일 | NOT RUN | SKIPPED_HEADROOM |

| RAM pressure | Windows used / available | Commit / limit | Pagefile current use |
|---|---|---|---:|
| 2GiB | 17.953 /13.161GiB | 34.653 /37.364GiB | 997MiB |
| 4GiB | 20.080 /11.034GiB | 36.712 /37.364GiB | 997MiB |

4GiB 단계에서 commit margin이0.652GiB여서 다음 increment를 할당하지 않았다. Guard는 projected Windows available≥3GiB, WSL available≥2GiB, commit margin≥0.5GiB이다.16GiB는 현재 WSL 약15GiB 한도에도 맞지 않는다. **8/12/16GiB에서 PASS 또는 OOM을 관측한 것이 아니다.**

RAM 해제 후 GPU uint8 tensor를 실제 **7.000GiB allocated/reserved**로 할당·fill·유지했다. 이 상태 full reconnect/image/state **5/5 PASS**. 사전 free GPU≥7.75GiB 조건을 확인했고 종료 후 tensor/context를 해제했다.

Communication failure correlated: **NO, tested2/4GiB RAM 및7GiB GPU 범위에서만**. 더 큰 RAM과 모든 pressure 조합은 검증하지 않았다. 실제 model+simulator 공존은 뒤의 live loop에서 별도 확인했다.

Windows used/available/commit/limit, pagefile allocated/current/peak, simulator WDDM memory/global GPU는 `windows-resources.jsonl`에 있다. Windows가 관리하는 pagefile 용량은 시험 중 변했지만 수동 설정 변경은 하지 않았다.

## Gate C와 재현

**NOT REQUIRED.** Native direct route에서 A와 tested B를 통과했으므로 gateway를 추가하지 않았다. 자세한 사유는 [gateway_evaluation.md](gateway_evaluation.md).

실제 실행 명령:

```bash
$UAV_VLA_HOME/projectairsim-client/bin/python \
  $REPO_ROOT/scripts/communication_reconnect_probe.py \
  --host 192.168.160.1
```

Simulator를 먼저 실행하고 현재 host 주소를 확인해야 한다. 완료된 feasibility gate를 자동 반복하지 않는다. 모든 raw evidence는 `outputs/communication_final/`; 이전 실패 자료는 `outputs/integration/`에 보존했다.
