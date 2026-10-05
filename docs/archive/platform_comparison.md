# Final Simulation Platform Recommendation

2026-10-05 KST. **Strategy 1: Windows Project AirSim 개발 + Native Ubuntu TravelUAV 최종 benchmark 후보**를 권장한다. Windows GPU/camera/demo를 실제 확인했다. Native는 boot 환경 부재로 NOT TESTED이며 최종 benchmark 통과로 판정하지 않는다.

## 두 플랫폼 비교

| 항목 | Project AirSim Windows | TravelUAV Native Ubuntu |
|---|---|---|
| GPU rendering | ✅ RTX 5070 PID 3D engine 확인 | NOT TESTED |
| Setup difficulty | 646.565 MiB Blocks + client venv 실제 실행 | 별도 boot 환경 필요; 계획상 setup 부담이 더 큼 |
| Simulator stability | 최종 약 54초 probe, camera 각 100개, script 비행; takeoff False 제한 | NOT TESTED |
| Visual quality | 단순 Blocks; front/down/chase demo Good 잠정 평가 | Native 화면 미관측 |
| User-visible fun | 실제 debug demo 가능, 5.24 FPS; 사용자 평가는 미수집 | NOT TESTED |
| Camera latency | front 40.759/40.587/42.019 ms, down 40.885/40.765/42.048 ms (mean/median/p95) | N/A |
| VRAM | Simulator dedicated peak **0.919 GiB**, global peak 3.289 GiB | UNKNOWN |
| VLA integration | primitives 확인, 실제 VLA 미실행 | upstream 경로와 가까우나 실제 inference 미검증 |
| AeroVLA compatibility | **Medium**: NNG/camera/state/action adapter 필요 | 설계상 높은 호환성 후보; NF4 loader 미검증 |
| Failure injection ease | 임시 blur/drift config demo ✅ | 이번에 구현/검증 안 함 |
| Benchmark value | custom development/demo; TravelUAV score를 대신하지 않음 | 원래 benchmark 후보; 전체 protocol 미검증 |
| Reproducibility | release/client pin, archive hash, config/raw metrics 확보 | 동일 archive 확보; Native 측정은 없음 |
| Research value | 실험 pipeline/시각 demo 개발에 유용하다고 추론 | AeroVLA/TravelUAV 재현에 적합한 후보라고 추론 |

연구 가치/재미/호환성은 source audit과 관찰에 따른 판단이며, 미실행 model/benchmark 성능 수치가 아니다. 이전 **Microsoft AirSim 1.8.1 Blocks**와 이번 Project AirSim은 분리한다.

상세: [Project AirSim](projectairsim_smoke_test.md), [Native 미실행/계획](native_traveluav_smoke_test.md), [WSL 판정](traveluav_rendering_decision.md). Project AirSim provenance: [official v1.0.1 release](https://github.com/iamaisim/ProjectAirSim/releases/tag/v1.0.1).

## 전략 판단

| Strategy | 판단 |
|---|---|
| 1. Windows Project AirSim 개발 + Native TravelUAV 최종 평가 | **권장, 조건부**. 지금 볼 수 있는 개발 경로와 원래 benchmark 목표 유지; 두 환경 adapter 비용 있음 |
| 2. 모두 Native TravelUAV | Native 환경/GPU 검증 통과 후 재검토. 현재 즉시 실행 불가 |
| 3. 모두 Windows Project AirSim | Demo/custom benchmark는 후보이나 원래 TravelUAV 평가를 대체하므로 현재 목표에는 추천하지 않음 |

가장 재미있게 개발할 **현재 후보**는 Windows Project AirSim: 창/camera/failure 표시가 실제 가능했다. 사용자 재미 평가와 GUI 고속화는 아직이다.

가장 연구적으로 신뢰할 **최종 benchmark 후보**는 Native TravelUAV: AeroVLA의 원래 map/평가 정의를 유지할 수 있다. Native가 더 안정적이라고 실측한 것은 아니며 hardware Vulkan/protocol 검증이 선행돼야 한다.

```yaml
Project AirSim Result: ⚠️  # GPU/camera/demo PASS; takeoff boolean 제한
Native TravelUAV Result: NOT TESTED
Recommended architecture:
  Development: Windows native Project AirSim, model-free demo
  Future inference: WSL2 AeroVLA NF4 후보, NNG dual-port 및 adapter 먼저 검증
  Final benchmark: Native Ubuntu + 동일 TravelUAV map + AeroVLA 후보
```

## RTX 5070 메모리

| 값 | GiB | 측정 / 추정 |
|---|---:|---|
| GPU capacity | 11.940 | measured, 12,227 MiB |
| Project AirSim dedicated peak | 0.919 | measured, 941.184 MiB PID WDDM |
| Simulator 포함 전체 GPU peak | 3.289 | measured, 3,368 MiB; desktop/다른 앱 포함 |
| AeroVLA NF4 resident budget | 6–8 | **estimated**, 7B 미실행 |
| 추가 activation/transient peak | 1–2 | **estimated**, actual peak 미측정 |
| Windows global peak + model/runtime | **10.289–13.289** | estimated peak co-residency |
| Native TravelUAV simulator | UNKNOWN | NOT MEASURED |

**Combined feasibility: Uncertain.** 낮은 추정은 11.940 GiB 이내지만 높은 쪽은 초과한다. Global peak에 simulator dedicated 값을 다시 더하면 중복 합산이다. Tiny NF4 peak를 7B peak로 환산하지 않았다.

## 다음 단계

1. Takeoff False의 의미/원인과 WSL↔Windows NNG 두 포트 연결을 최소 검증한다. GUI는 topic streaming/PNG 저장 제거로 개선할 후보지만 이번에는 추가 최적화를 하지 않는다.
2. 별도 SSD/준비된 Native Ubuntu 확보 후 Vulkan → 동일 BrushifyUrban camera 100-frame → simulator 자원 순으로 Test A를 수행한다. 현재 partition/EFI/OS/bootloader는 수정하지 않는다.
3. 이후 원본 수정 없는 AeroVLA NF4 loader 검증과 실제 combined VRAM 계획을 확인하고 checkpoint 다운로드 여부를 다시 결정한다.

**AeroVLA checkpoint 다운로드 진행: NO.** Custom OpenVLA + LoRA + quantization loader, split inference, Native TravelUAV 자원은 아직 검증되지 않았다.

세 문서와 raw evidence를 남기고 simulator/client 창과 listener를 정리했다. Model/전체 dataset 다운로드, Failure Detection/Recovery, fine-tuning, 대규모 evaluation으로 진행하지 않았다. Blur/drift는 이번 프롬프트가 허용한 폐기 가능한 demo에 한정했다.
