# Native Ubuntu + TravelUAV smoke test

2026-10-05 KST. **NOT EXECUTED — hardware boot environment unavailable.** Native Vulkan, TravelUAV GPU, latency, VRAM은 측정하지 않았다.

## A0 — 읽기 전용 disk / boot 확인

Project AirSim 테스트 이후 Get-Disk, Get-Partition, Get-Volume, Get-PhysicalDisk, bcdedit /enum, wsl -l -v를 조회했다. [Inventory](../outputs/platform_final/disk-boot-inventory.json).

| 대상 | 관찰 |
|---|---|
| Disk 0 | SHGP31-2000GM, 2,000,398,934,016 bytes NVMe SSD, Windows IsBoot/IsSystem=true |
| Partitions | EFI 200 MiB, MSR 16 MiB, C: NTFS, Windows Recovery 약 901 MiB |
| Disk 1 / 2 | CalDigit SD reader, size 0 / No Media; SSD가 아님 |
| 다른 physical SSD | 확인 안 됨 |
| 연결된 external SSD | usable media 확인 안 됨 |
| Native Linux partition | 조회된 GPT 목록에 확인 안 됨 |
| Firmware / BCD | access denied; 전체 boot entries는 UNKNOWN |
| WSL Ubuntu | 설치돼 있지만 Native Ubuntu가 아님 |

**현재 세션에서 사용할 Native Ubuntu: NOT AVAILABLE.** Firmware 접근 제한 때문에 모든 기존/외부 Ubuntu 설치가 존재하지 않는다고 단정하지 않는다. C: 빈 공간을 별도 SSD로 취급하지 않았다. OS 설치, EFI mounting, format, partition resize, bootloader 변경, 재부팅은 하지 않았다.

## A1 — 준비할 OS와 SSD 조건

**Ubuntu 24.04 LTS 최신 point release를 우선 검증 후보**로 둔다. 동일 TravelUAV ELF가 기존 WSL Ubuntu 24.04.4에서 launch/RPC에 성공해 user-space 실행 증거는 있다. 이것은 Native GPU 성공 증거가 아니다. Upstream 환경 재현이 필요하면 22.04 LTS도 후보이며 RTX 5070 지원 driver/kernel부터 확인해야 한다.

Blackwell은 NVIDIA **open kernel modules**를 사용해야 한다. [NVIDIA 공식 설명](https://download.nvidia.com/XFree86/Linux-x86_64/570.133.07/README/kernel_open.html), [Ubuntu driver 안내](https://ubuntu.com/desktop/docs/en/latest/how-to/graphics/install-nvidia-drivers/). Native 설치 시 실제 ubuntu-drivers list에 있는 RTX 5070 지원 stable desktop *-open 조합을 선정한다. 과거 branch 번호를 무조건 고정하거나 WSL에 Linux graphics driver를 설치하지 않는다.

별도 SSD의 프로젝트 계획 예산: **최소 128 GB, 운영 여유는 256 GB**, SSD 자체 UEFI/EFI와 ext4 설치 공간. OS 25–35 GB, Python/cache 10–15 GB, 현재 map 약 3 GB, 향후 모델/cache 20–30 GB와 log 여유를 잡은 **추정**이다. 전체 benchmark dataset 저장량은 포함하지 않는다. External USB SSD는 안정적인 연결과 지속적 I/O가 필요하다. 설치/포맷/boot 설정은 별도 승인 이후 작업이다.

## A2–A5 — Native 실행 계획

1. 준비된 Native Ubuntu에서 OS/kernel/driver를 기록하고 nvidia-smi, vulkaninfo --summary를 실행한다. **RTX 5070 hardware Vulkan**이 기준이다. CPU/llvmpipe면 FAIL.
2. 기존 **동일 BrushifyUrban archive**만 복사한다. 1,530,641,125 bytes, SHA256 `5faf7a9b475a4adc5e106c8a9711b975497cab9482f188ceb06cd557017c8d09`. WSL 보유 위치 `/home/gyuhan/uav-vla-smoke/assets/archives/BrushifyUrban.zip`. 전체 dataset을 다시 받지 않는다.
3. UE4.27.2 Linux binary를 640×480 windowed Vulkan으로 실행하고 UE RHI가 RTX 5070을 선택하는지 확인한다. 종료된 WSL OpenGL 수정을 재개하지 않는다.
4. 기존 isolated AirSim client/settings를 복사해 ping, state/pose, front/down scene을 확인한다. Camera별 warm-up과 **새 frame 최소 100개**. RPC round-trip, decode 포함 FPS, dual-view timestamp 차이를 구분한다.
5. Simulator만 로드한 상태에서 process RSS/CPU, GPU memory/util을 주기적으로 수집한다. Process NVIDIA accounting이 없으면 전후 global delta를 proxy라고 명시한다. Desktop 사용량과 simulator 전용 VRAM을 구분한다.
6. 이 단계 통과 후에만 별도 AeroVLA NF4 loader 및 공존 예산을 점검한다. 이번 요청의 checkpoint download 금지는 유지한다.

## A6 — 공존 VRAM 추정

**Native TravelUAV simulator VRAM: UNKNOWN.** WSL llvmpipe, old AirSim Blocks 또는 Project AirSim의 0.919 GiB를 Native TravelUAV 값으로 대신 쓰지 않는다.

Gate 1은 tiny random Llama의 NF4 stack만 검증했다. OpenVLA 7B actual peak는 측정하지 않았다. AeroVLA NF4 resident **6–8 GiB + 추가 activation/transient 1–2 GiB는 계획용 추정**이다. Vision/projector/resize/LoRA/generation 설정에 따라 달라진다.

```text
RTX 5070 capacity = 11.940 GiB (Windows 측정)
Native simulator = S GiB (UNKNOWN)
Native desktop/driver = D GiB (UNKNOWN)
Estimated model resident = 6–8 GiB
Estimated extra runtime peak = 1–2 GiB
Estimated total = S + D + 7–10 GiB
Coexistence = Uncertain
```

## A7 — 결과

```yaml
Native TravelUAV Result: NOT TESTED
Execution status: NOT EXECUTED — hardware boot environment unavailable
Native Ubuntu: NOT AVAILABLE in current session
RTX Vulkan: ⚠️ NOT TESTED
TravelUAV GPU rendering: ⚠️ NOT TESTED
AirSim RPC: ⚠️ NOT TESTED on native Ubuntu
Camera: ⚠️ NOT TESTED on native Ubuntu
Camera latency: N/A
Simulator VRAM: N/A
System RAM: N/A
Estimated AeroVLA INT4 coexistence: Uncertain
Major blockers:
  1: usable Native Ubuntu boot environment / separate SSD 없음
  2: native Vulkan 및 동일 map GPU 실행 미검증
  3: actual AeroVLA NF4 loader / combined peak VRAM 미검증
```

[기존 WSL 판정](traveluav_rendering_decision.md)은 그대로 유지한다. Native 미실행은 Native 실패 또는 프로젝트 중단을 뜻하지 않는다.
