# RTX 5070 compatibility smoke test

> **최신 상태 (2026-10-05 최종 검증):** 실제 OpenVLA-7B + AeroVLA LoRA NF4 로딩·생성과 Windows Project AirSim live camera → action → 이동 10/10이 성공했다. [최종 결과](final_closed_loop_validation.md), [실제 모델 NF4 검증](aerovla_int4_validation.md). 아래 필수 판정표와 본문은 최초 TravelUAV Gate 1/2/3 시점의 기록이다. TravelUAV Gate 2의 GPU 렌더링 실패는 유지하며, Project AirSim에서의 후속 성공과 구별한다.

> 2026-10-05: [Project AirSim smoke test](projectairsim_smoke_test.md), [Native TravelUAV 계획/미실행](native_traveluav_smoke_test.md) 추가. Windows Project AirSim GPU 성공은 기존 **TravelUAV Gate 2** GPU 성공이나 **AeroVLA Gate 3** 통과로 간주하지 않는다. Gate 1 NF4 결과는 그대로이며 actual AeroVLA INT4는 미검증이다.

실행일: 2026-10-04 (Asia/Seoul). Git branch: `researchuav-vla-feasibility`.

**Gate 1 통과, Gate 2 부분 검증, Gate 3 미실행.** CUDA/NF4와 Windows Blocks → WSL Python RPC는 실제 동작했다. 이후 요청한 WSLg 재시도에서 NVIDIA D3D12/OpenGL backend는 성공했으나 UE 4.27.2가 -opengl을 Vulkan으로 되돌려 TravelUAV는 계속 llvmpipe CPU renderer였다. 30-frame CPU RPC를 측정했고 동일 BrushifyUrban의 공개 Windows build가 없음을 확인했다. 최종 권장은 **C. Native Ubuntu**, Windows 개발 차선은 **D**로 갱신한다. [최신 rendering decision](traveluav_rendering_decision.md). 전체 AeroVLA/TravelUAV GPU 실행은 미입증이고 checkpoint 다운로드 권고는 **NO**다.

전체 dataset, OpenVLA/AeroVLA checkpoint, navigation episode, failure injection/detection/recovery, fine-tuning, evaluation은 수행하지 않았다. upstream 소스는 수정하지 않았다. 고정 revision 대조 (`outputs/compatibility/upstream-integrity.json`, 로컬 기록·Git 제외)에서 wrapper 내용이 일치했으며 기존 Windows audit copy의 CRLF와 GitHub LF만 다르다. AirSim client 수정은 격리한 Gate 2 환경에만 적용하고 원본/수정 hash와 patch를 보존했다.

## 필수 판정표

| Gate | 결과 | 실제 측정값 | 판정 |
|---|---|---|---|
| CUDA/PyTorch | PASS, WSL GPU kernel 실행 | torch `2.7.1+cu128`, CUDA `12.8`, `cuda.is_available=True`, RTX 5070, CC `12.0` | ✅ |
| BF16 | PASS, matmul 결과 유한 | 256×256, FP32 기준 relative L2 `0.00165626`, BF16 지원 True | ✅ |
| WSLg OpenGL/D3D12 | env 변수로 NVIDIA hardware 선택 성공 | D3D12 (NVIDIA GeForce RTX 5070), Accelerated yes, OpenGL 4.6 | ✅ backend / ⚠️ UE가 OpenGL 미사용 |
| bitsandbytes NF4 | PASS, kernel 및 HF loader/generate | bnb `0.48.2`, uint8 packed NF4/BF16 compute, 작은 Llama의 14개 Linear4bit 변환 | ✅ |
| Unreal rendering | -opengl 재시도도 Vulkan CPU fallback; Windows Blocks는 RTX 5070 | UE `4.27.2`: OpenGL desktop 지원 종료 warning, Vulkan `llvmpipe`; Blocks는 D3D11 RTX | ⚠️ Travel gate 미충족 |
| AirSim RPC | 두 환경 ping/state/pose 성공; Windows 직접 NAT 실패, 중계 성공 | WSL state `0.473ms`, pose `0.354ms`; Windows 중계 state `0.780ms`, pose `0.567ms` | ✅ 연결 / ⚠️ topology 조건 |
| Camera RPC | 기존 단일 요청 및 최신 CPU fallback 30/30 성공 | 최신 windowed CPU mean `391.426ms`, median `286.870ms`, p95 `324.601ms`; 기존 Blocks 중계 단일 `179.195ms`는 다른 조건 | ✅ 응답 / ⚠️ Travel GPU 미검증 |
| Simulator VRAM | Windows Blocks 단독 측정; Travel hardware 값 미측정 | Blocks PID dedicated GPU peak `216.160MiB`, shared `80.770MiB`; Travel WSL은 CPU renderer | ⚠️ 다른 map의 실측 |
| INT4 AeroVLA loader | NOT_RUN: Gate 2의 Travel GPU 조건 미충족 | prototype·adapter load·OpenVLA forward·동시 VRAM 측정 없음 | ⚠️ |

✅는 해당 API/연산만의 판정이다. **Gate 2 전체 통과에는 TravelUAV map + RTX 5070 hardware renderer + RGB/state가 함께 필요**하므로 generic Blocks 성공으로 대체하지 않았다.

## Gate 1 — modern CUDA / NF4

WSL Ubuntu 24.04.4, kernel `6.6.87.2-microsoft-standard-WSL2`. uv `0.12.23`으로 Python 3.10.14와 별도 venv를 만들었다. system Python 3.12.3과 shell profile을 변경하지 않았다. 환경: `$UAV_VLA_HOME/gate1`.

```yaml
Python: 3.10.14
PyTorch: 2.7.1+cu128
Torch CUDA: "12.8"
NVIDIA Driver: "591.86"
GPU: NVIDIA GeForce RTX 5070
Compute capability: "12.0 / sm_120"
Transformers: 4.42.4
BitsAndBytes: 0.48.2
PEFT: 0.11.1
Accelerate: 0.32.1
NF4 test: PASS - Linear4bit and HF load/forward/4-token generation
BF16 test: PASS - finite matmul, relative L2 0.00165626
Peak VRAM:
  Torch allocated: 9.250 MiB
  Torch reserved: 22.000 MiB
  Global nvidia-smi sampled peak: 1833 MiB
  Global baseline: 1608 MiB
```

Torch peak는 작은 테스트의 allocator 통계다. global 값에는 Windows desktop, 다른 프로세스, CUDA context가 포함된다. `1833-1608=225MiB`는 전체 GPU 점유 변화이며 모델 전용 메모리로 단정하지 않는다. GPU 총량은 `12227MiB`. AeroVLA 7B peak와 별개다.

BF16 256×256 matmul, `bnb.nn.Linear4bit(512,128)` NF4 forward가 성공했다. NF4 packed weight uint8/output BF16, dequantized matmul 기준 relative L2 `0.0`. wheel arch list에 `sm_120`이 있고 `torch.cuda.is_bf16_supported()` True.

**로컬 random Llama 361,088 parameters**를 생성해 725,170-byte checkpoint로 저장했다. 외부 pretrained model 다운로드 없이 아래 설정으로 HF load/logits/greedy 4-token generation을 성공했다. 14개 projection이 실제 Linear4bit로 변환되었다.

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.bfloat16)
# AutoModelForCausalLM.from_pretrained: device_map={"": 0},
# torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
# attn_implementation="eager", local_files_only=True
```

이 결과는 AutoModelForVision2Seq/OpenVLA custom code/timm vision/AeroVLA LoRA/resize 호환성을 포함하지 않는다. `past_key_values` tuple deprecation 경고는 있었지만 실행 성공. flash-attn/xformers는 설치하지 않았다.

첫 실행은 bnb import → Triton 3.3.1의 C helper compilation에서 `RuntimeError: Failed to find C compiler. Please specify via CC environment variable.`로 실패했다. `gcc`/`cc` 부재를 확인하고 **패키지 버전을 유지**한 채 WSL root로 `gcc libc6-dev`를 설치한 뒤 재실행해 통과했다. gcc `13.3.0`, libc6-dev `2.39-0ubuntu8.9`. apt가 필요한 libc 관련 패키지 3개도 갱신했지만 전체 OS upgrade는 하지 않았다. 기본 사용자 sudo는 password를 요구해 WSL `-u root` 경로를 사용했다. 임의 버전 반복 설치는 하지 않았다.

오류 전문: 최초 stdout/stderr (`outputs/compatibility/gate1-run.log`, 로컬 기록·Git 제외), 실패 JSON (`outputs/compatibility/gate1-initial-failure.json`, 로컬 기록·Git 제외). 성공 증거: 복구 후 로그 (`outputs/compatibility/gate1-run-after-gcc.log`, 로컬 기록·Git 제외), 측정 JSON (`outputs/compatibility/gate1-result.json`, 로컬 기록·Git 제외), 전체 freeze (`outputs/compatibility/gate1-freeze.txt`, 로컬 기록·Git 제외). [설치 script](../../scripts/setup_gate1.sh), [측정 코드](../../scripts/compatibility_gate1.py).

## Gate 2 — 단일 TravelUAV map과 Windows fallback

### 자원 / 공통 설정

[고정한 TravelUAV 환경 revision](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env/tree/44de5739a95a2f6a88767446421cddada9606642)의 **완전한 독립 ZIP 중 가장 작은 `BrushifyUrban.zip` 하나**를 받았다. Desert의 320MB ZIP과 carla의 338MB ZIP은 분할 압축 마지막 조각이므로 단독 환경으로 세지 않았다.

| 자원 | 다운로드 / 압축 해제 | SHA256 |
|---|---|---|
| TravelUAV BrushifyUrban | 1,530,641,125 bytes (1.426GiB) / 1,656,328,410 bytes, 27 entries | `5faf7a9b475a4adc5e106c8a9711b975497cab9482f188ceb06cd557017c8d09`, HF digest와 일치 |
| [AirSim Windows Blocks 1.8.1](https://github.com/microsoft/AirSim/releases/tag/v1.8.1-windows) | 259,463,081 bytes (247.443MiB), 별도 fallback map | `47c526a5f0acff42c211d2479b9de9f10286a8162678ef19adc09040a23ca1db`, 로컬 계산; 공급자 digest 미제공 |

BrushifyUrban은 Linux ELF와 `.sh`만 포함하고 Windows executable은 없다. WSL 경로는 `$UAV_VLA_HOME/assets/travel-urban`. Blocks는 `assets/windows-blocks-1.8.1`. raw dataset/closed-loop 전체 maps/trajectory/spawn metadata는 받지 않았다. default spawn은 유효한 navigation 시작 위치로 확인하지 않았다. 물리 엔진으로 pose가 변했지만 takeoff/arm/move 명령은 호출하지 않았다.

모델을 종료한 뒤 simulator만 실행하여 ping/version/listVehicles, **simGetImages 1회**, state 1회, pose 1회를 요청하고 종료했다. [settings](../../scripts/gate2-settings.json): Drone_1/SimpleFlight, FrontCamera/DownCamera, Scene 256×256, ClockSpeed **1**, NoDisplay, RPC 41461. launcher RenderOffscreen/NoSound/NoVSync, 640×480. 원본 ClockSpeed 10/5-view/recording과 다른 **smoke 전용 설정**이다.

Gate 2 env `$UAV_VLA_HOME/gate2`: Python 3.10.14, AirSim 1.8.1, numpy 1.26.3, tornado 4.5.3, msgpack 1.1.2, AeroVLA 수정 msgpack-rpc-python 0.4, OpenCV contrib 4.11.0.86. `uv pip check` 통과. freeze (`outputs/compatibility/gate2-freeze.txt`, 로컬 기록·Git 제외), [설치 순서를 기록한 script](../../scripts/setup_gate2.sh). 기존 env를 덮어쓰는 설치 재실행은 피한다.

[AeroVLA 공식 troubleshooting](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/docs/assets/troubleshooting.md)에 따라 수정 RPC zip을 설치하고 AirSim client encoding kwargs를 제거했다. zip SHA256 `c0d7df3fe91271ea052384ca7150c7f6730eeed63672168d08a0f27946322197`. patch (`outputs/compatibility/airsim-client-encoding.patch`, 로컬 기록·Git 제외), 원본/수정 hash (`outputs/compatibility/airsim-client-patch-manifest.json`, 로컬 기록·Git 제외). upstream checkout에는 적용하지 않았다.

### A. WSL simulator + WSL Python

실제 `BrushifyUrban.sh -RenderOffscreen -vulkan ...` 실행. `ldd` unresolved library 없음. UE 로그는 **4.27.2**, AirSim plugin을 확인했다. `vulkaninfo`와 UE 모두 **llvmpipe (LLVM 20.1.2, 256 bits), Type CPU**를 선택했다. **RTX 5070 hardware rendering 조건 미충족.** CUDA 성공과 Vulkan graphics를 구별하며 Linux NVIDIA display driver를 WSL에 설치하지 않았다.

최초 camera `8051.535ms`. resource sampler가 blocking RPC 중 멈추는 문제를 확인해 별도 thread로 바꾸고 동일 설정으로 한 번 재측정했다. 재측정 camera `3051.437ms`, ping `0.744ms`, state `0.473ms`, pose `0.354ms`. 각 실행에서 한 프레임만 받았고 latency 분포/FPS/최대치로 해석하지 않는다.

재측정 process-group sampled RSS peak **1359.879MiB (1.328GiB)**. global GPU usage 1645MiB가 일정했으나 CPU renderer의 점유로 귀속하지 않는다. **TravelUAV GPU-rendering VRAM은 MISSING**.

최초 RPC (`outputs/compatibility/wsl-first-run/wsl-travel-rpc.json`, 로컬 기록·Git 제외), 재측정 RPC (`outputs/compatibility/wsl-travel-rpc.json`, 로컬 기록·Git 제외), resource samples (`outputs/compatibility/wsl-travel-launch.json`, 로컬 기록·Git 제외), UE 전체 로그 (`outputs/compatibility/wsl-travel-unreal.log`, 로컬 기록·Git 제외), Vulkan 진단 (`outputs/compatibility/wsl-vulkan-summary.log`, 로컬 기록·Git 제외), 프레임 (`outputs/compatibility/wsl-travel-camera.png`, 로컬 기록·Git 제외).

### B. Windows simulator + WSL Python

Blocks를 `-d3d11 -RenderOffscreen`으로 실행. UE **4.27.2**의 `Chosen D3D11 Adapter`와 device 생성에서 **RTX 5070 / driver 591.86** 확인. UE 로그에 SUV glass가 참조하는 EditorMaterials package 누락 등 warning/error가 남았지만 이번 drone scene/한 프레임 RPC는 성공했다. 전체 assets의 무결성이 검증되었다고 보고하지 않는다.

직접 NAT `WSL → 192.168.160.1:41461`은 timeout. Windows는 `0.0.0.0:41461`에서 listen했으나 WSL TCP 접속은 완료되지 않았다. 셸에 Windows 관리자 token이 없었다. 방화벽/host filtering은 의심되지만 정확한 차단 rule은 미확정. 방화벽을 끄거나 규칙을 추가하지 않았다.

임시 역방향 TCP 경로에서는 실제 RPC가 성공했다:

```text
WSL AirSim client → 127.0.0.1:41501 → WSL bridge
  → Windows에서 WSL:41500으로 먼저 연결한 stream
  → Windows bridge → Windows 127.0.0.1:41461 → AirSim Blocks
```

RPC message를 변경하지 않고 양방향 byte stream을 중계했다. camera 파일 공유를 RPC 대신 사용하지 않았다. [Python bridge](../../scripts/gate2_reverse_bridge.py), [Windows bridge](../../scripts/gate2_reverse_bridge.ps1). 테스트 후 simulator/helper/listen sockets를 종료했다. persistent service/firewall 변경 없음.

camera **179.195ms**, ping `2.114ms`, state `0.780ms`, pose `0.567ms`. latency는 network/render/readback을 포함한 최초 요청이며 GPU render time/지속 FPS가 아니다.

simulator PID GPU counter의 sampled peak **216.160MiB (0.211GiB) dedicated**, **80.770MiB shared**. 시스템 RAM working-set peak **367.750MiB**, private bytes **604.637MiB**. global nvidia-smi baseline `1639MiB`, peak `1860MiB`. GPU counter와 nvidia-smi 범위가 달라 더하지 않는다. 작은 Blocks의 값이며 TravelUAV 도시 scene으로 외삽하지 않는다.

RPC JSON (`outputs/compatibility/windows-blocks-bridge-rpc.json`, 로컬 기록·Git 제외), resource samples (`outputs/compatibility/windows-blocks-bridge-launch.json`, 로컬 기록·Git 제외), UE 로그 (`outputs/compatibility/windows-blocks-unreal.log`, 로컬 기록·Git 제외), 직접 NAT 실패 전문 (`outputs/compatibility/windows-blocks-rpc.json`, 로컬 기록·Git 제외), 프레임 (`outputs/compatibility/windows-blocks-bridge-camera.png`, 로컬 기록·Git 제외).

**A/B는 map·renderer·cache·중계 조건이 달라 NOT_COMPARABLE.** 8.05s 대 179ms를 GPU 개선 배율이나 TravelUAV 성능 차이로 보고하지 않는다. B는 hardware/RPC 구조를 입증했지만 동일 TravelUAV Windows scene/build가 필요하다. Linux pak을 Windows executable에 복사하면 동등 환경이 된다고 가정하지 않는다.

### 이후 한정 재시도 — 2026-10-04 23시대

`GALLIUM_DRIVER=d3d12`, `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`로 glxinfo가 RTX 5070 hardware를 확인했다. 그러나 동일 Unreal에 `-opengl -windowed`를 주면 desktop OpenGL 지원 종료 warning 후 Vulkan llvmpipe를 사용한다. 동일 설정에서 30회 camera/state RPC를 계측했고 mean 391.426 / median 286.870 / p95 324.601ms였다. 최초 요청 3318.664ms도 포함한다. system RAM RSS sampled peak 1407.805MiB, global GPU range 1622–1643MiB. hardware scene VRAM으로 귀속하지 않는다. 이전 RenderOffscreen 단일 측정과 NOT_COMPARABLE이다.

공식 README/HF 최신 metadata/동일 ZIP 전체 contents/GitHub tree/releases를 조사한 결과 **동일 BrushifyUrban의 공개 Windows packaged build는 NO**다. 비공개 build 존재는 알 수 없다. WSL graphics 추가 수정은 종료하고 C 또는 D 경로만 권장한다. [전체 근거와 30개 samples](traveluav_rendering_decision.md).

## Gate 3 — NOT_RUN

사용자의 **Gate 1과 Gate 2 모두 통과 후 prototype 진행** 조건에 따라, TravelUAV GPU rendering이 미검증인 상태에서 INT4 loader/patch를 만들거나 checkpoint를 받지 않았다. 다음은 기존 정적 조사에서 확인한 쟁점이며 실제 adapter/forward 검증 결과가 아니다.

| 항목 | 원본 코드 / 향후 검증 조건 |
|---|---|
| loader / device_map | AutoModelForVision2Seq BF16, trust_remote_code; NF4/device_map 없음. 향후 load-time GPU placement 필요 |
| resize_token_embeddings | tokenizer/config/embedding row 대조 필요. quantized output head의 resize 경로 미검증 |
| PEFT LoRA | rank 64, unmerged, projector modules_to_save; quantization 대상과 adapter/vision/projector dtype/device 확인 필요 |
| self.model.to(self.device) | 원본은 PEFT 후 호출. transformers 4.42.4의 bnb PreTrainedModel.to 제한 때문에 일반적으로 그대로 유지할 수 있다고 볼 수 없음. 바깥 PeftModel의 이동 경로가 달라 같은 예외를 보장하지는 않음 |
| OpenVLA custom code | 작은 Llama PASS가 custom multimodal/timm/vision dtype 호환성을 입증하지 않음 |

근거: [원본 wrapper](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/src/model_wrapper/aerovla_wrapper_ui.py#L20-L44), [Transformers to 코드](https://github.com/huggingface/transformers/blob/v4.42.4/src/transformers/modeling_utils.py#L2756-L2796), [adapter config](https://huggingface.co/XuPeng23/AerialVLA/raw/196f2f3253b69df6e90ac10b6ae041c7b3a9569e/aero_vla/adapter_config.json). AeroVLA INT4는 **미검증**이며 실패 실험으로 표시하지 않는다.

## 재실행과 자원 보존

```powershell
# 최초 설치 시 C compiler 필요; setup은 별도 Python env를 생성한다
wsl -d Ubuntu -u root -- apt-get install -y --no-install-recommends gcc libc6-dev
wsl -d Ubuntu -- bash $REPO_ROOT/scripts/setup_gate1.sh
wsl -d Ubuntu -- $UAV_VLA_HOME/gate1/bin/python $REPO_ROOT/scripts/compatibility_gate1.py --output $REPO_ROOT/outputs/compatibility --model-dir $UAV_VLA_HOME/tiny-hf-llama
# 이미 설치된 Gate 2 env와 압축 해제 자원을 사용
wsl -d Ubuntu -- $UAV_VLA_HOME/gate2/bin/python $REPO_ROOT/scripts/compatibility_gate2_linux.py --executable $UAV_VLA_HOME/assets/travel-urban/BrushifyUrban/BrushifyUrban.sh --output $REPO_ROOT/outputs/compatibility
& .\scripts\compatibility_gate2_windows.ps1 -UseReverseBridge
```

재실행은 결과 파일을 갱신하므로 기존 evidence를 보존하고 실행해야 한다. Travel archive+extract 약 3.0GiB, Gate 1 env 약 6.7GiB, Gate 2 env 약 197MiB. hardlink/cache 중복으로 단순 합은 실제 추가 disk usage와 다를 수 있다. CUDA runtime 다운로드는 테스트 설치이며 모델/dataset 다운로드와 구별된다. `outputs/compatibility/`에 원본 로그/versions/hash/resource samples를 남겼고 assets/output은 Git ignore 대상이다.

## Recommendation

```yaml
권장 구조: C. Native Ubuntu - 동일 TravelUAV baseline용, 아직 native 실측 전
현재 Windows 개발 차선: D. Windows AirSim 개발 + 추후 native Linux TravelUAV 평가
NF4: ✅
Simulator: ⚠️ - Windows Blocks RTX rendering/RPC 성공, TravelUAV GPU rendering 미확인
AeroVLA INT4: ⚠️ - NOT_RUN, Gate 2 통과 대기
현재 예상 VRAM:
  Model: MISSING - AeroVLA 실측 없음; 6–8 GiB는 계획용 예약 가정
  Simulator: MISSING - TravelUAV hardware 실측 없음; Blocks만 0.211 GiB dedicated 실측
  Total: MISSING - 동시 실행 없음, 12 GB 이하 보장 불가
다음 단계에서 OpenVLA/AeroVLA checkpoint 다운로드를 진행해도 되는가: NO
근거:
  - RTX 5070의 WSL CUDA/BF16/NF4/HF 로딩 stack은 실제 통과했다.
  - WSLg NVIDIA OpenGL은 정상이나 UE 4.27.2는 -opengl을 Vulkan으로 돌려 CPU renderer를 사용한다.
  - 동일 BrushifyUrban Windows map/build는 공식 공개 배포에 없다.
  - 직접 NAT 연결 문제는 남아 있으며 임시 중계 경로만 확인했다.
  - AeroVLA custom loader/LoRA/resize 및 model+scene 메모리는 미검증이다.
```

다음 후보는 확보한 native Linux 환경에서 같은 map의 hardware renderer를 확인하는 C, 또는 개발과 최종 TravelUAV 평가 환경을 나누는 D다. 이번 단계에서 checkpoint/Gate 3/episode로 넘어가지 않는다. 파티션/dual boot 설치는 수행하지 않았다.
