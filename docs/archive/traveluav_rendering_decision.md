# TravelUAV rendering decision

실행일: 2026-10-04 23시대 (Asia/Seoul). 범위: 기존 BrushifyUrban map의 WSLg/OpenGL 재시도 1개 구성, 30-frame RPC 계측, 동일 map Windows 배포 조사. 상세 evidence는 `outputs/rendering_retry/`에 보존했다.

**결론: WSLg OpenGL/D3D12는 RTX 5070으로 동작하지만, 현재 TravelUAV UE 4.27.2 binary는 OpenGL 요청을 Vulkan으로 돌려 llvmpipe를 사용한다.** 공식 공개 BrushifyUrban Windows build도 없다. 같은 TravelUAV 환경 재현의 우선 경로는 **C. Native Ubuntu**, 현재 Windows 장비에서 개발을 이어가는 대안은 **D. Windows AirSim 개발 + 추후 native Linux TravelUAV 평가**다. Native Ubuntu는 권장 경로이며 아직 이 PC에서 실행 검증한 환경이 아니다. Gate 2의 hardware rendering 조건은 미충족이고 Gate 3 진행은 **NO**다.

## 1. WSLg backend 실측

필요한 진단 도구 `mesa-utils 9.0.0-2`만 Ubuntu 공식 저장소에서 설치했다. Mesa/NVIDIA driver/PyTorch 버전을 바꾸거나 PPA를 추가하지 않았다. env 변수는 해당 실행 프로세스에만 적용했다. profile/WSL 설정/Windows firewall/파티션은 수정하지 않았다.

| 진단 | 실제 결과 | 판정 |
|---|---|---|
| nvidia-smi | RTX 5070, driver 591.86, total 12227MiB | CUDA 장치 노출 정상; graphics 증거와 구별 |
| 기본 glxinfo -B | llvmpipe (LLVM 20.1.2, 256 bits), Accelerated: no | ❌ 기본 OpenGL은 CPU |
| GALLIUM_DRIVER=d3d12 + MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA | **D3D12 (NVIDIA GeForce RTX 5070), Accelerated: yes**, OpenGL core 4.6 | ✅ WSLg OpenGL hardware backend |
| vulkaninfo --summary | llvmpipe, PHYSICAL_DEVICE_TYPE_CPU | ❌ TravelUAV에 필요한 Vulkan GPU 경로 없음 |

실행 명령:

```bash
/usr/lib/wsl/lib/nvidia-smi
glxinfo -B
vulkaninfo --summary
GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA glxinfo -B
```

기본 OpenGL 로그 (`outputs/rendering_retry/glx-default.log`, 로컬 기록·Git 제외), NVIDIA D3D12 로그 (`outputs/rendering_retry/glx-nvidia-d3d12.log`, 로컬 기록·Git 제외), Vulkan 로그 (`outputs/rendering_retry/vulkan-summary.log`, 로컬 기록·Git 제외), nvidia-smi (`outputs/rendering_retry/nvidia-smi.log`, 로컬 기록·Git 제외), 설치 로그 (`outputs/rendering_retry/mesa-utils-install.log`, 로컬 기록·Git 제외).

Microsoft의 [WSLg GPU 선택 문서](https://github.com/microsoft/wslg/wiki/GPU-selection-in-WSLg)는 이 adapter 선택을 Mesa D3D12 경로로 설명한다. 이번 OpenGL 성공은 Vulkan까지 hardware로 바뀌었다는 뜻이 아니다.

## 2. 실제 TravelUAV 실행과 RHI 판정

UE 버전은 이전 map 로그 및 이번 로그의 **4.27.2**다. [Epic 4.27 공식 사양](https://dev.epicgames.com/documentation/unreal-engine/hardware-and-software-specifications?application_version=4.27)은 4.26부터 OpenGL 지원 제거를 명시한다. 실행 전 binary의 ASCII/UTF-16 strings에서 `opengl`, OpenGLDrv, RHI 선택/진단 메시지를 확인했다. legacy parser가 남아 있어 **알려진 -opengl 옵션 하나만** 실제 확인했다. 여러 flag/driver 버전 조합을 반복하지 않았다.

```bash
GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  $UAV_VLA_HOME/assets/travel-urban/BrushifyUrban/BrushifyUrban.sh \
  -opengl -ResX=640 -ResY=480 -windowed -NoSound -NoVSync \
  -settings=$REPO_ROOT/scripts/gate2-settings.json
```

원본 map/pak/settings에는 변경을 가하지 않았다. 최초 bounded launch에서 fallback을 확인하고, **같은 구성**을 30-frame 계측 목적으로 한 번 더 실행했다. 추가 해결 시도가 아니다. 이번에는 windowed 실행이며 이전 RenderOffscreen 한 프레임 측정과 직접 비교하지 않는다.

UE 로그의 결정적인 실제 메시지:

```text
Warning: OpenGL is no longer supported for desktop platforms. Vulkan will be used instead.
LogInit: Using SDL_WINDOW_VULKAN
LogVulkanRHI: Display: Device 0: llvmpipe (LLVM 20.1.2, 256 bits)
LogVulkanRHI: Display: - DeviceID 0x0 Type CPU
```

따라서 driver 선택 변수는 정상 적용되었어도, **해당 Unreal binary가 OpenGL RHI를 사용하지 않는다**. `-opengl4` 검토는 이 명시적 desktop 지원 종료에서 끝냈으며 추가 launch하지 않았다. Vulkan/D3D12 driver를 새로 빌드하거나 Engine/pak을 고치는 작업도 하지 않았다.

전체 UE 로그 (`outputs/rendering_retry/travel-opengl-unreal.log`, 로컬 기록·Git 제외), stdout (`outputs/rendering_retry/travel-opengl-stdout.log`, 로컬 기록·Git 제외), binary strings (`outputs/rendering_retry/unreal-rhi-wide-strings.log`, 로컬 기록·Git 제외), 최초 시도 JSON (`outputs/rendering_retry/first-attempt/travel-opengl-launch.json`, 로컬 기록·Git 제외), [계측 코드](../../scripts/traveluav_opengl_probe.py).

## 3. 30-frame RPC와 메모리

scene에 모델을 띄우지 않았고 takeoff/arm/move/episode 명령도 호출하지 않았다. `ping`, `getMultirotorState`, `simGetVehiclePose` 성공. FrontCamera / Drone_1 / uncompressed Scene RGB 256×256의 **simGetImages를 연속 30회** 요청했고 모두 유효한 image payload를 받았다.

| 항목 | CPU Vulkan fallback 실측 |
|---|---:|
| simGetImages 요청 / 성공 | 30 / 30 |
| Mean | **391.426ms** |
| Median | **286.870ms** |
| p95 | **324.601ms** |
| 최초 요청 | 3318.664ms |
| Simulator process-group RSS sampled peak | **1407.805MiB (1.375GiB)** |
| global nvidia-smi usage 범위 | 1622–1643MiB |
| global GPU utilization sampled 최대 | 5% |
| Simulator 전용 RTX scene-rendering VRAM | **N/A: hardware scene rendering이 성립하지 않음** |

첫 요청을 제외하지 않은 30개 전체 통계다. p95는 numpy percentile의 linear interpolation이며 30 samples의 작은 표본이다. 첫 요청의 shader/capture 초기화 원인을 세부 profiling한 것은 아니다. 이 결과는 RPC 응답 가능성을 보여주지만 **GPU 성능 수치가 아니고 Gate 2 GPU PASS가 아니다**. 과거 1-frame latency와 map/window/cache 조건이 달라 NOT_COMPARABLE이다.

nvidia-smi는 desktop/WSLg presentation/다른 Windows process를 포함한다. scene renderer가 CPU인데 GPU 사용률 몇 %나 global usage 증가만으로 RTX 렌더링을 주장하지 않는다. Windows WDDM의 GPU Engine/Process Memory counters도 저장했다. **Task Manager UI 자체는 직접 열람하지 않았고**, 같은 WDDM 계측 경로의 raw counters로 보완했다. 이 VM/presentation 점유를 simulator 전용 VRAM으로 분리할 근거는 없어 global 1643MiB를 simulator VRAM이라고 적지 않았다.

30개 latency + resource samples (`outputs/rendering_retry/travel-opengl-launch.json`, 로컬 기록·Git 제외), 실행 로그 (`outputs/rendering_retry/travel-opengl-profile.log`, 로컬 기록·Git 제외), Windows WDDM counters (`outputs/rendering_retry/windows-wddm-during-profile.json`, 로컬 기록·Git 제외).

## 4. 동일 map Windows build 조사

판정: **NO — 현재 공식 공개 배포 기준**. 저자의 비공개/local build까지 없다고 단정하지 않는다.

- [TravelUAV README](https://github.com/prince687028/TravelUAV/blob/5cc26e9a4a55b9c788e918f7c3bb2dc5076a85e6/README.md)의 environment download는 공식 HF 저장소로 연결된다. [공식 프로젝트 페이지](https://prince687028.github.io/Travel/)에도 추가 Windows package 링크가 없다.
- HF 최신 revision은 이전과 같은 `44de5739a95a2f6a88767446421cddada9606642`. 동일 BrushifyUrban은 ZIP 하나다. 이미 받은 해당 ZIP의 **27 entries 전체**를 검사했고 `.exe`, WindowsNoEditor, Win64는 없으며 Linux executable과 LinuxNoEditor pak만 있다. ZIP digest는 기존 HF digest와 일치한다.
- 공식 GitHub main `5cc26e9a4a55b9c788e918f7c3bb2dc5076a85e6` 전체 tree도 Windows executable/package/build 경로를 제공하지 않는다. [GitHub releases](https://github.com/prince687028/TravelUAV/releases)는 없고 현재 API도 `[]`다.
- [유지관리자 issue #6 답변](https://github.com/prince687028/TravelUAV/issues/6#issuecomment-2677291805)은 작업을 Linux에서 수행했다고 밝힌다. 이 답변만으로 Windows 불가능을 단정한 것이 아니라 실제 배포 목록/동일 ZIP과 함께 판정했다.

동일 ZIP 전체 목록 (`outputs/rendering_retry/same-map-package-audit.json`, 로컬 기록·Git 제외), 최신 HF metadata (`outputs/rendering_retry/travel-env-metadata.json`, 로컬 기록·Git 제외), GitHub tree (`outputs/rendering_retry/travel-github-tree.json`, 로컬 기록·Git 제외), releases API (`outputs/rendering_retry/travel-github-releases.json`, 로컬 기록·Git 제외), issue 답변 API (`outputs/rendering_retry/travel-issue6-comments.json`, 로컬 기록·Git 제외).

다른 map 전체 ZIP을 다운로드하거나 Windows Blocks를 같은 TravelUAV map으로 취급하지 않았다. Linux cooked pak이 Windows cooked build를 대체한다는 가정도 하지 않는다.

## TravelUAV Rendering Decision

```yaml
WSL OpenGL/D3D12 renderer: D3D12 (NVIDIA GeForce RTX 5070), Accelerated yes
TravelUAV renderer: Vulkan llvmpipe CPU; UE 4.27.2가 -opengl 요청을 거부하고 Vulkan 선택
RTX 5070 사용: NO - TravelUAV scene rendering 기준; WSLg OpenGL 진단은 YES
simGetImages mean: 391.426 ms (CPU fallback, 30 samples)
median: 286.870 ms
p95: 324.601 ms
Simulator VRAM: N/A - hardware rendering 미성립, global GPU 값과 분리 불가
TravelUAV Windows build: NO - 동일 BrushifyUrban 공식 공개 배포 기준
최종 권장 구조: C. Native Ubuntu - 실제 TravelUAV baseline/benchmark용
현재 Windows 개발 대안: D. Windows AirSim 개발 후 native Linux에서 TravelUAV final evaluation
Gate 3 진행: NO
근거:
  - WSLg hardware OpenGL은 작동하나 배포 Unreal은 OpenGL을 사용하지 않는다.
  - 실제 Unreal RHI는 Vulkan/llvmpipe CPU이며 RPC 성공으로 GPU gate를 대체할 수 없다.
  - 동일 공개 map의 Windows packaged build가 없다.
  - Native Ubuntu는 기존 Linux binary와 native GPU driver 경로를 사용하는 권장 후보이며 아직 실측 전이다.
```

C를 택하면 확보한 native Linux 환경에서 먼저 같은 ZIP으로 Vulkan adapter/RGB/state를 검증한다. 현재 Windows 파티션 수정, dual boot 설치, driver 변경은 수행하지 않았다. D는 개발 환경 대안이고 TravelUAV 원본 benchmark 재현을 의미하지 않는다. **이번 작업은 결정 보고에서 멈추며 checkpoint 다운로드, Gate 3 prototype, failure module, 학습/evaluation을 시작하지 않는다.**
