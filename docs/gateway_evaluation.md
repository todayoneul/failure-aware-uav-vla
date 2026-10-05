# Windows Gateway Evaluation

2026-10-05 KST.

**NOT REQUIRED** for the validated current architecture.

Native direct WSL NNG connection은 Gate A20/20 reconnect와5/5 idle reconnect를 통과했다. Gate B의 실제 RAM2/4GiB, GPU7GiB pressure에서도 통신 실패가 없었다. 이어 같은 native direct route로 live single-step 및10/10 loop가 성공했다.

| Gateway 지표 | 결과 |
|---|---|
| Used | **NO** |
| Observation requests | 0; gateway test 미실행 |
| Action requests | 0; gateway test 미실행 |
| Failures / disconnects | N/A |
| Mean / median / p95 | N/A |
| 100 observation +20 action stability gate | NOT RUN / NOT REQUIRED |

`windows_gateway/`, ZeroMQ/FastAPI dependency, gateway protocol을 추가하지 않았다. Simulator NNG를 직접 사용하는 official client가 기존 WSL runtime에서 Windows host의8989/8990에 연결한다. 이전 two-port reverse NNG relay는 현재 구성에 포함하지 않는다. 과거 relay failure의 내부 원인을 해결/확정했다고 주장하지 않는다.

남은 route 조건:

1. 시작할 때 현재 Windows host IP와 simulator port를 확인한다.192.168.160.1은 이번 세션의 측정 주소이다.
2. 한 실행에 하나의 NNG client owner를 유지한다. 모델 로딩 뒤 connect하고 loop 동안 유지한다. 경쟁 topic consumer/기존 eager relay session을 추가하지 않는다.
3. 임의의 host firewall/network 변경 및 장시간 unattended reconnect는 미검증이다. 향후 direct failure가 재현될 경우에만 Windows-owned client gateway를 다시 평가한다.

그 경우에는 NNG/topic을 Windows에서 끝내고 binary image/state 및 bounded action만 simple transport로 교환하며100-observation/20-action 안정성 기준과 command replay 방지를 검증해야 한다. 이번에는 불필요했으며 구현/검증하지 않았다.

실측: [communication_stability_test.md](communication_stability_test.md), [final_closed_loop_validation.md](final_closed_loop_validation.md).
