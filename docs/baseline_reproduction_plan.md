# Minimal Baseline Reproduction Plan

작성일: 2026-10-04 (Asia/Seoul). 관련 조사: [feasibility_report.md](feasibility_report.md).

## 1. 첫 milestone와 현재 상태

목표는 **TravelUAV ModernCityMap의 원본 navigation episode 한 개에서 instruction → AeroVLA action → UAV 이동 → 종료/결과 저장을 확인**하는 것이다. 모델 학습이나 Failure Detection/Diagnosis/Recovery는 포함하지 않는다.

**2026-10-04 실제 smoke 갱신:** Python 3.10.14/torch 2.7.1+cu128/bnb 0.48.2의 CUDA·BF16·NF4·작은 HF 모델 로딩은 통과했다. TravelUAV BrushifyUrban은 WSL에서 CPU renderer와 RGB/state/pose가 성공했고 Windows Blocks의 RTX 5070 rendering + WSL 중계 RPC가 성공했다. 동일 TravelUAV GPU rendering은 미확인이므로 Gate 2 부분 검증, INT4 prototype Gate 3 미실행, checkpoint 다운로드 권고 **NO**다. [전체 실측/로그](compatibility_smoke_test.md). 아래 episode/raw/7B/checkpoint 명령은 향후 계획이며 이번 단계에서 실행하지 않았다. 사용자 요청에 따라 episode로 넘어가지 않고 중단한다.

이번 기술 Gate 1/2/3와 아래 episode 계획의 G0~G5는 구별한다. 기술 Gate 3은 INT4 prototype이고 아래 G3은 episode 자료 준비다.

| Gate | 다음 단계 진입 조건 |
|---|---|
| G0: 자원·환경 | 별도 branch `researchuav-vla-feasibility` 및 격리 uv env 생성 완료; 기존 Python 보존 |
| G1: CUDA/의존성 | CUDA/BF16/NF4/HF tiny load/generate 통과. VLA/timm/adapter 전체 import는 별도 |
| G2: simulator | 보류: BrushifyUrban WSL은 CPU renderer. Windows Blocks RTX/RPC 성공은 동일 Travel scene 검증을 대체하지 않음 |
| G3: episode 자료 | 원본 episode/mark/description/merged JSON·meta가 연결되고 hash 기록 |
| G4: model | direct NF4 load + unmerged adapter + 한 observation 추론; peak/dtype/유효한 action 확인 |
| G5: closed loop | 한 episode를 원본 horizon·종료 규칙으로 완료하고 로그/metric 저장 |

G5의 **pipeline 완료**와 **navigation 성공**은 다르다. collision/timeout으로 끝나면 연결 smoke는 통과할 수 있지만 “목표를 정상 달성했다”고 보고하지 않는다. navigation 성공은 원본 evaluator의 success 판정과 raw contact collision 기록을 함께 확인한다. 첫 episode가 실패하면 실패를 숨기거나 horizon을 줄이지 않는다.

## 2. Exact environment 후보

| 항목 | 선택 후보 / 현재 근거 |
|---|---|
| Host | Windows 11 Pro, RTX 5070, driver 591.86 (현재 읽기 전용 확인) |
| 실행 OS | B안 조건부: Windows simulator + 기존 WSL2 Ubuntu 24.04.4 LTS inference, kernel 6.6.87.2 |
| Python | 실제 uv venv `/home/gyuhan/uav-vla-smoke/gate1`와 `gate2`, **3.10.14**. Conda는 설치하지 않음 |
| PyTorch runtime | 실제 **torch 2.7.1+cu128**, torchvision 0.22.1+cu128; torchaudio 미설치 |
| CUDA | wheel 제공 CUDA **12.8 runtime**; host driver 사용, 전체 toolkit/source-build 기본 경로 제외 |
| model precision | **NF4 4-bit base + BF16 compute**; nonquantized vision/projector/adapter dtype는 검사 후 고정 |
| attention | `sdpa` 후보, 배포 custom code가 거부하면 `eager` 검사; flash-attn/xformers 기본 제외 |
| 작업 파일시스템 | WSL Linux filesystem의 `~/uav-vla`; 수많은 trajectory 파일을 `/mnt/c`에서 처리하는 경로는 우선 피함 |
| 자원 topology | GPU 0, model 1 process, simulator scene 1개, batchSize 1 |

system Python 3.12.3과 테스트 환경을 분리했다. compiled UE 4.27.2 BrushifyUrban은 실행했으나 llvmpipe CPU renderer였다. Windows Blocks D3D11 RTX 렌더링과 중계 RPC는 성공했다. 동일 TravelUAV Windows scene/build 확보 또는 native Linux hardware renderer 확보가 남았다. Ubuntu/renderer/map을 바꾸면 manifest와 결과 scope도 변경한다.

PyTorch 2.7.1/cu128 대응 조합은 [공식 설치 표](https://pytorch.org/get-started/previous-versions/)에 있고, [bnb 0.48.2](https://huggingface.co/docs/bitsandbytes/v0.48.2/en/installation)는 sm120 및 NF4 binary 지원 근거가 있다. 최신 package를 모두 쓰는 조합 대신 upstream API 변경을 줄이는 후보를 고정한다.

## 3. Repository와 경로 전략

이번 smoke에서 `C:\Users\leegy\Desktop\drone`에 Git을 초기화하고 `researchuav-vla-feasibility` branch를 생성했다. upstream checkout은 외부 TEMP 경로에서 원본 그대로 보존했다. 아래 future baseline에는 **외부 별도 clone + immutable SHA + patch 기록**을 선택한다. 동작이 안정된 뒤 필요하면 submodule로 전환할 수 있다. assets/output/venv는 Git ignore 대상으로 두었다.

제안 경로는 아래와 같다. 이것은 이번에 생성한 구조가 아니다.

```text
Windows: C:\Users\leegy\Desktop\drone\docs\  ← 이번 두 문서
WSL: ~/uav-vla/
  external/AeroVLA/       ← 실험 branch, upstream SHA 보존
  external/TravelUAV/     ← 전처리 source, SHA 고정
  assets/archives/        ← 원본 분할 archive
  assets/envs/closeloop_envs/
  assets/dataset_raw/ModernCityMap/<episode UUID>/
  assets/openvla-7b/
  assets/adapter-repository/aero_vla/
  manifests/             ← versions, commands, hashes, patch와 episode 선정
```

future shell 경로 정의 예시 (`UAV_*`는 이 계획 전용 변수):

```bash
export UAV_WORK="$HOME/uav-vla"
export UAV_AERO="$UAV_WORK/external/AeroVLA"
export UAV_TRAVEL="$UAV_WORK/external/TravelUAV"
export UAV_ASSETS="$UAV_WORK/assets"
export UAV_ENV_ROOT="$UAV_ASSETS/envs"
export UAV_DATA_ROOT="$UAV_ASSETS/dataset_raw"
```

`HOME` 자체를 재정의하지 않는다. 환경/data/model은 Git에서 제외하고 code patch와 manifest만 version control에 남긴다. 외부 clone에서 변경할 경우 먼저 branch를 만든다.

```bash
mkdir -p "$UAV_WORK/external" "$UAV_ASSETS/archives" \
  "$UAV_ENV_ROOT" "$UAV_DATA_ROOT" "$UAV_ASSETS/staging/raw" "$UAV_WORK/manifests"
git clone --filter=blob:none https://github.com/XuPeng23/AeroVLA.git "$UAV_AERO"
git -C "$UAV_AERO" switch --detach 2c5ae0987a484ab92f00dd9d9ed493cb3e98e492
git -C "$UAV_AERO" switch -c researchuav-vla-feasibility
git clone --filter=blob:none https://github.com/prince687028/TravelUAV.git "$UAV_TRAVEL"
git -C "$UAV_TRAVEL" switch --detach 5cc26e9a4a55b9c788e918f7c3bb2dc5076a85e6
```

detached TravelUAV source는 읽기/전처리 실행만 하며 generator를 수정해야 하면 그곳에서도 별도 branch를 생성한다. source snapshot 자체가 보장하는 것은 코드 동일성이고 package/asset 실행 성공이 아니다.

## 4. 설치 순서와 package 고정 후보 — G0/G1

1. WSL host driver, RAM, disk 확인. 현재 WSL RAM은 약 15GiB로 CPU checkpoint loading transient가 문제일 수 있다. Windows의 31.11GiB를 전부 WSL이 쓰는 것으로 가정하지 않는다. `.wslconfig` 변경이 필요하면 별도로 기록한다.
2. 실제 smoke는 uv 0.12.23 managed Python과 별도 venv를 사용했다. [실행한 설치 script](../scripts/setup_gate1.sh)/freeze를 우선 재사용한다. 아래 Conda 명령은 이전 계획의 대안이며 현재 설치 방식으로 보고하지 않는다.
3. Python 3.10 환경 → CUDA wheel → model utility → patched RPC → AirSim 순서.
4. OS library/graphics tool은 실제 필요한 항목만 설치하고 `dpkg-query` 결과를 기록한다. 현재 libvulkan/UE runtime 의존성과 `net-tools`, `7z`, Tk library의 확정 OS package lock은 `MISSING`이다.

```bash
conda create -n uav-vla-baseline python=3.10.14 -y
conda activate uav-vla-baseline
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  transformers==4.42.4 peft==0.11.1 accelerate==0.32.1 \
  bitsandbytes==0.48.2 tokenizers==0.19.1 timm==0.9.10 \
  huggingface-hub==0.23.5 safetensors==0.4.5 \
  einops==0.6.1 numpy==1.26.3 scipy==1.15.0 \
  opencv-python==4.10.0.84 Pillow==10.2.0 tqdm==4.67.1 \
  yacs==0.1.8 psutil==6.1.1 \
  tornado==4.5.3 msgpack==1.1.2
python -m pip install "$UAV_AERO/msgpack-rpc-python-fix-msgpack-dep.zip"
python -m pip install airsim==1.8.1 --no-build-isolation
python -m pip check
```

`pip install -r upstream/requirements.txt`는 이 후보에 사용하지 않는다. torch/vision/bnb/flash-attn pin을 재적용하기 때문이다. 위 **직접 의존성 버전은 구체적인 후보**지만 pip/setuptools와 전이 dependency·OS library의 전체 lock은 아직 없다. 실제 resolver/설치 검증 후 `pip freeze`, `conda list --explicit`, OS package inventory를 저장해야 재현 가능한 설치 결과가 된다. 선언된 package 범위가 맞는 것만으로 “호환 확인 완료”라고 보고하지 않는다.

torchaudio가 필요한 import가 발견되면 별도로 `torchaudio==2.7.1`을 같은 cu128 index에서 설치한다. `flash-attn==2.5.8`을 설치하지 않고 SDPA/eager로 검증한다. 현재 FA4에 sm120 코드가 있어도 다른 API의 beta 경로를 첫 baseline에 추가하지 않는다.

RPC zip은 [AeroVLA 공식 troubleshooting](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/docs/assets/troubleshooting.md)의 수정 소스다. 조사한 16,338-byte zip의 SHA256:

```text
c0d7df3fe91271ea052384ca7150c7f6730eeed63672168d08a0f27946322197
```

공식 수정 RPC zip과 `VehicleClient` encoding kwargs 제거를 실제 Gate 2 env에 적용했다. [원본/수정 hash](../outputs/compatibility/airsim-client-patch-manifest.json)와 [정확한 patch](../outputs/compatibility/airsim-client-encoding.patch)를 기록했고 두 환경 RGB/state RPC를 성공했다. upstream은 수정하지 않았다. Tornado 4.5.3을 이유 없이 최신 major로 바꾸지 않았다.

다음은 향후 전체 baseline import 확인 예시다. 현재 Gate 1의 실행 결과는 smoke 문서/freeze에 있으며 timm/airsim/tk를 포함한 이 combined import 명령은 실행하지 않았다.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list()); x=torch.ones((16,16),device='cuda'); print((x@x).sum().item())"
python -c "import bitsandbytes, transformers, peft, accelerate, timm, airsim, msgpackrpc, tkinter; print('imports OK')"
python -m pip check
```

bnb NF4 Linear4bit quantize/forward와 로컬 random Llama의 HF load/generate를 실제 실행해 통과했다. 첫 Triton import는 C compiler 부재로 실패했으며 gcc/libc6-dev를 추가하고 같은 버전으로 복구했다. 이 결과는 OpenVLA/AeroVLA inference 검증과 구별한다.

## 5. 최소 environment 다운로드·renderer/RPC — G2

이번 camera/state smoke에는 **BrushifyUrban ZIP 하나(1.531GB)**만 받아 WSL에서 실행했고 Windows Blocks(0.259GB)를 별도로 비교했다. UE 4.27.2/RPC는 확인했으나 Travel hardware rendering이 미충족이다. 따라서 아래 18.65GB closed-loop 묶음과 raw/7B 다운로드는 **실행하지 않고 보류**했다. 작은 map smoke의 ClockSpeed 1/두 camera와 아래 원본 평가 설정을 혼동하지 않는다.

**사전 용량 보고:** environment 묶음 18,645,702,495 bytes, raw map 9,847,057,165 bytes, base+adapter 약 15.55GB. 전체 알려진 최소 다운로드 약 **44.05GB / 41.03GiB**. 추출/중복/cache/환경 포함 **100–150GiB 여유 예약은 계획 가정**이다. 먼저 environment만 받고 G2를 통과한 뒤 raw/checkpoint 다운로드로 진행한다. 전체 raw+env는 약 527.53GiB로 현재 C: 여유보다 크므로 받지 않는다.

HF repository는 다음 revision으로 고정한다. public 자원이며 명령은 아직 수행하지 않았다.

```bash
huggingface-cli download wangxiangyu0814/TravelUAV_env \
  --repo-type dataset --revision 44de5739a95a2f6a88767446421cddada9606642 \
  --include 'closeloop_envs.z*' 'closeloop_envs.zip' \
  --local-dir "$UAV_ASSETS/archives/env"
7z l "$UAV_ASSETS/archives/env/closeloop_envs.zip"
7z x "$UAV_ASSETS/archives/env/closeloop_envs.zip" -o"$UAV_ENV_ROOT"
```

분할 파일을 모두 같은 directory에 둔다. `7z` extraction은 [사용자 보고](https://github.com/prince687028/TravelUAV/issues/51#issuecomment-3336545506)를 근거로 한 후보이며 이 archive를 직접 열어 검증하지 않았다. listing에서 예상 root·uncompressed 크기·누락 part를 확인한다. unzip 하나로 마지막 zip만 풀 수 있다고 가정하지 않는다. archive에 ModernCityMap/shared Engine만 선택 추출할 수 있으면 원래 relative layout을 유지한다. 필요한 shared files와 추출 크기가 아직 불명확하므로 무조건 일부 폴더만 지우지 않는다.

서버 root는 `ModernCityMap.sh`가 들어 있는 `closeloop_envs`의 **부모**여야 한다. 이 경우 `$UAV_ENV_ROOT/closeloop_envs/ModernCityMap.sh`가 존재해야 한다. `.sh`와 Linux 실행 파일 permission, runtime library를 점검한다.

```bash
cd "$UAV_AERO"
python airsim_plugin/AirVLNSimulatorServerTool.py \
  --gpus 0 --port 30000 --root_path "$UAV_ENV_ROOT"
```

이것은 **AeroVLA의 server** command이다. TravelUAV README의 별도 server/gpu default/25000 eval port를 섞지 않는다. management RPC는 127.0.0.1:30000, 첫 scene AirSim port는 일반적으로 30001이지만 실제 반환 port를 따른다. 원본 `reopen_scenes` API로 model 없이 한 scene을 요청하는 확인 예시:

```bash
python -c "import msgpackrpc; c=msgpackrpc.Client(msgpackrpc.Address('127.0.0.1',30000),timeout=180); print(c.call('ping')); print(c.call('reopen_scenes','127.0.0.1',[['ModernCityMap',0]]))"
```

반환값이 성공이고 렌더링 로그가 정상인지 확인한 뒤 반환된 AirSim port에서 `confirmConnection`, RGB/DepthPerspective 두 camera 응답·pose를 읽는다. `Drone_1`, `FrontCamera`, `DownCamera`는 current settings에 정의된 이름이다. settings를 임의로 stock AirSim camera 이름으로 바꾸지 않는다. 이후 G3에서 episode의 pose/target object를 설정해 실제 이동을 확인한다.

G2 성공 조건:

- RHI/renderer 로그와 graphics adapter가 하드웨어 경로를 사용하며 software fallback만으로 실행되고 있지 않음.
- RGB가 빈 이미지/검은 화면이 아니고 depth/pose RPC가 유효함. CUDA `nvidia-smi` 존재만으로 통과하지 않음.
- shader/material/asset 누락과 RPC timeout이 없음. 응답시간·simulator-only VRAM peak 기록.
- scene 1개, model 0개 조건에서 메모리·RAM 여유 확인.

서버는 원본 template에서 runtime settings를 다시 생성한다. `ClockSpeed=10`, 5-view RGB/depth 256×256 및 front/down record camera 1024×1024 등을 baseline 평가 시 보존한다. 현재 model이 두 RGB view를 써도 evaluator/recording의 추가 camera를 누락하지 않는다. 이번 gate는 별도 ClockSpeed 1/두 camera/RPC 설정을 사용했다. WSL hardware graphics 조건이 미충족이므로 checkpoint/episode를 보류하고 조건부 Windows split 구조를 다음 후보로 선택했다.

## 6. Dataset·one-episode 준비 — G3

```bash
huggingface-cli download wangxiangyu0814/TravelUAV \
  --repo-type dataset --revision faa8f2514156455ea7423464cc1295e6f92575cb \
  --include 'ModernCityMap.z*' 'ModernCityMap.zip' \
  --local-dir "$UAV_ASSETS/archives/data"
7z l "$UAV_ASSETS/archives/data/ModernCityMap.zip"
7z x "$UAV_ASSETS/archives/data/ModernCityMap.zip" -o"$UAV_ASSETS/staging/raw"
```

archive listing에서 root를 확인해 `dataset_raw/ModernCityMap/...` 경로가 두 번 중첩되지 않도록 한다. staging에서 아래 선택 episode만 `$UAV_DATA_ROOT/ModernCityMap/<UUID>/`로 배치하고 `log`, RGB, mark, description을 함께 보존한다. 가능하면 7z listing 후 해당 경로만 선택 추출해 staging 크기를 줄인다. closed-loop inference에는 full-map `rgb_imgs.tensor` 또는 feature 전처리가 필요하지 않다.

첫 후보는 공식 AeroVLA split에 존재하는 다음 trajectory이다. 난이도와 성공 여부는 아직 확인하지 않았다.

```text
ModernCityMap/84eabab1-99ff-4cb8-bde9-ad9459196148/merged_data.json
```

원본 `data/uav_dataset/seen_valset_splits/ModernCityMap.json`에서 같은 `json` path에 해당하는 27개 records (frames 1–27)를 별도 파일 `data/uav_dataset/smoke/ModernCityMap_one_episode.json`에 저장하는 계획이다. 한 record만 있어도 현재 loader의 dedupe 결과는 한 episode지만, 전체 records를 보존하면 선정 근거를 추적하기 쉽다. 원본 split은 변경하지 않는다. 존재하지 않는 `--max_episodes`나 평가 환경에 전달되지 않는 `--activate_maps`에 의존하지 않는다.

metadata는 고정한 AeroVLA checkout의 `data/meta/map_spawnarea_info.json`과 `object_description.json`을 사용하고 hash를 남긴다. TravelUAV HF meta revision도 대조용으로 기록한다. 공개 HF 전체 seen split과 AeroVLA map split의 episode 수를 혼동하지 않는다.

merged JSON이 없을 때 사용하는 정확한 원본 tool 경로:

이 tool은 해당 map 아래의 모든 trajectory folder를 순회하며 episode 선택 flag가 없다. 따라서 `$UAV_DATA_ROOT/ModernCityMap/`에는 선택한 UUID folder만 배치하고, 다른 추출 episode는 staging에 둔다. 이 조건을 확인한 뒤 다음 명령으로 한 episode의 metadata만 생성한다.

```bash
python "$UAV_TRAVEL/Model/LLaMA-UAV/tools/generate_merged_json.py" \
  --root_dir "$UAV_DATA_ROOT" --map_list ModernCityMap
```

이 tool의 `random.seed = 1`은 seed 함수를 호출한 것이 아니라 덮어쓴 것이므로, description 선택이 재생성할 때 달라질 수 있다. 기존 merged JSON을 우선 보존한다. 생성이 필요하면 선택된 instruction과 file hash를 기록한 뒤 재사용한다. seed를 수정하면 별도 branch의 명시적인 patch로 기록한다. baseline instruction을 설명 없이 고정 문장으로 바꾸지 않는다.

G3에서 선택한 folder의 `mark.json`, `object_description.json`, merged의 `trajectory_raw_detailed`, `conversations[0].value`와 spawn metadata를 대조한다. target spawn, 초기 pose, 원본 instruction을 원래 값으로 설정할 수 있는지 확인한다.

## 7. Checkpoint·NF4 adaptation — G4

```bash
huggingface-cli download openvla/openvla-7b \
  --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --local-dir "$UAV_ASSETS/openvla-7b"
huggingface-cli download XuPeng23/AerialVLA \
  aero_vla/adapter_config.json aero_vla/adapter_model.safetensors \
  --revision 196f2f3253b69df6e90ac10b6ae041c7b3a9569e \
  --local-dir "$UAV_ASSETS/adapter-repository"
```

training JSON은 다운로드에서 제외한다. AeroVLA root의 `./openvla-7b`가 base directory를, `./checkpoints/aero_vla`가 `adapter-repository/aero_vla`를 참조하도록 symlink 등을 준비한다. 기존 directory가 있으면 먼저 확인한다. LoRA HF 저장소 전체를 adapter path로 잘못 지정하지 않는다.

**다음 단계에서 검증할 최소 변경이다. 이번에는 코드 파일을 만들거나 수정하지 않았다.**

| 위치 | 검증할 변경 |
|---|---|
| wrapper `from_pretrained` | NF4, double quant, BF16 compute의 `BitsAndBytesConfig`, load-time `device_map={"":0}`, low_cpu_mem_usage |
| placement | quantized branch에서 모델 전체 `.to(cuda)`/dtype 재변환을 피하고 adapter/input placement 검사 |
| adapter | unmerged LoRA + saved projector 로드; missing/unexpected keys, tensor dtype, 실제 bnb Linear4bit 확인 |
| attention | 고정한 custom model에서 SDPA 선택 여부 확인; eager로 바꾸면 memory/latency 재측정 |
| tokenizer/vision | 원본 embedding resize, token IDs, mosaic/image processor 유지; 비양자화 vision dtype 확인 |
| logging | raw generated text, parse 성립/stop 이유, load/inference peak와 시간 기록; 이 milestone에서 policy action/stop 판정은 바꾸지 않음 |

`llm_int8_skip_modules` 등으로 vision/projector를 제외한다면 4-bit에서도 실제 해당 module이 제외되었는지 변환된 module 목록으로 확인한다. 전체 parameters × 0.5 byte라는 하한을 실제 allocation으로 취급하지 않는다. BF16 base를 먼저 GPU에 올려 merge하는 경로는 선택하지 않는다.

G4에서 camera observation과 원본 instruction으로 한 번 추론해 유한한 범위 내 action, 유효한 token parse, 정상 stop flag, CPU/GPU dtype을 확인한다. 잘못된 parse가 0-vector → stop이 된 것을 올바른 LAND 예측으로 해석하지 않는다. NF4에서 이 checkpoint가 작동한다는 증거가 아직 없으므로 실패하면 G5로 넘어가지 않는다.

## 8. 한 개의 closed-loop episode — G5

**다음은 G1–G4를 통과하고 NF4 loader patch가 적용된 경우에만 실행하는 제안 명령이다. 원본 BF16 loader로 실행하지 않는다.** cwd를 AeroVLA root로 고정하고 server는 별도 terminal에서 실행한다.

```bash
cd "$UAV_AERO"
CUDA_VISIBLE_DEVICES=0 python -u src/vlnce_src/eval_aerovla.py \
  --run_type eval --name AeroVLA_NF4_Smoke \
  --gpu_id 0 --simulator_tool_port 30000 --DDP_MASTER_PORT 20001 \
  --batchSize 1 --maxWaypoints 200 \
  --dataset_path "$UAV_DATA_ROOT" \
  --eval_save_path ./eval_results/aero_vla_nf4_smoke/seen_valset/ModernCityMap \
  --model_path ./checkpoints/aero_vla \
  --eval_json_path ./data/uav_dataset/smoke/ModernCityMap_one_episode.json \
  --map_spawn_area_json_path ./data/meta/map_spawnarea_info.json \
  --object_name_json_path ./data/meta/object_description.json
```

새 output directory를 사용하고 재시도할 때마다 별도 directory를 기록한다. 기존 result를 섞으면 완료 trajectory를 건너뛰거나 집계가 틀릴 수 있다. DDP용 `RANK/WORLD_SIZE/LOCAL_RANK`가 우연히 상속되지 않았는지, port가 비어 있는지 확인한다. 원본 `80005` → `20001` 변경도 manifest에 남긴다.

metric.sh의 고정 path 불일치 때문에 실제 metric CLI를 직접 사용한다.

```bash
python utils/metric.py \
  --root_dir ./eval_results/aero_vla_nf4_smoke \
  --analysis_list seen_valset/ModernCityMap \
  --path_type_list full --map_filter ModernCityMap
```

성공 판정에 필요한 증거:

1. 원본 instruction, raw camera → processed input, raw generated text, decoded action을 추적할 수 있다.
2. UAV pose 변화가 실제 simulator 관측에 기록된다. action API 호출만으로 이동 완료를 판단하지 않는다.
3. 한 distinct trajectory가 200-waypoint 상한/원본 종료 규칙으로 끝나고 `ori_info.json`, `log/*.json`, RGB/depth output이 존재한다.
4. 원본 SR/OSR/NE/SPL을 계산하며 pipeline 종료와 navigation 성공을 구별한다. model stop + target 거리 ≤20m 등 SR의 원본 조건을 확인한다.
5. contact collision과 depth/stuck/distance-growth에 따른 pseudo-collision을 구별해 기록한다. 현재 evaluator의 감시를 새로운 failure detector의 성과로 세지 않는다.
6. model load, model-only, scene-only, 동시 실행 peak와 host RAM, latency/RPC 시간이 기록된다.

원본 metric 코드의 NE는 비행 종점과 원본 GT trajectory 종점의 거리이며 spawn target 거리와 같다고 보장할 수 없다. SPL은 GT polyline 길이에서 20m를 빼는 구현이다. AeroVLA OSR은 action 이후 반환된 state에서 갱신하므로 TravelUAV의 predicted-waypoint 기반 코드와 동일시하지 않는다. metric 이름만 보고 정의를 바꾸지 않는다.

한 episode의 SR=0/100%는 smoke 결과다. NF4 precision, 최신 split, WSL/runtime timing은 논문 조건과 달라 `NOT_COMPARABLE`이다. 이 단계에서 논문 표의 benchmark를 재현했다고 주장하지 않는다.

## 9. Resource budget과 manifest

| 항목 | 현재 수치 / 다음에 측정할 값 |
|---|---|
| base BF16 weights 산술 | 14.047GiB; 전체 GPU BF16 불가 |
| 모든 weights INT4인 이상적 하한 | 3.512GiB; 실제 allocation이 아님 |
| NF4 model 예산 | 6–8GiB를 계획 가정으로 예약 가능; 보장된 예측값 아님 |
| simulator peak | Windows Blocks dedicated GPU 216.160MiB, shared 80.770MiB; RAM working set 367.750MiB / private bytes 604.637MiB. 동일 TravelUAV hardware 값 `MISSING` |
| 동시 peak | `MISSING`; 7B 모델과 scene 동시 실행 없음 |
| 12GB 용량 합격 여부 | CUDA allocation과 Windows display를 포함한 total 측정; OOM/공유 memory 넘침을 성공으로 세지 않음 |
| CPU RAM | 현재 WSL 약 15GiB; load/quant 순간값과 swap 사용 확인 |
| knownminimumdownload | 約44.05GB / 41.03GiB |
| 계획 disk 여유 | 100–150GiB (staging/cache/env/output 포함 가정); archive listing 이후 갱신 |
| full assets | raw+env만 약 527.53GiB; 현재 PC에는 한꺼번에 받지 않음 |

측정할 때 PyTorch `max_memory_allocated/reserved`와 Windows `nvidia-smi` total usage를 함께 기록한다. 후자만으로 model/simulator 내역을 확정하지 않고 WSL NVML의 제약도 기록한다. 모델 6–8GiB 예산을 실측 최대값이라고 표기하지 않는다. load peak가 steady-state보다 높아도 누락하지 않는다.

재현 manifest에는 두 upstream SHA, patch/diff hash, HF revisions, download bytes/checksums, episode/subset/instruction hash, OS/Python/freeze, driver/torch/bnb/attention/quant/dtype/device map, renderer/assets/settings hash, horizon/ClockSpeed/종료 규칙, commands/cwd, peak/latency/실패 원인이 필요하다. 실제 smoke의 freeze/patch/hash/resource samples는 `outputs/compatibility/`에 보존했다. episode/model 관련 미실행 항목은 `MISSING`으로 남겼다.

## 10. 중단과 다음 단계 조건

| 실패 | 이 단계에서의 대응 |
|---|---|
| cu128/bnb kernel 실패 | driver/wheel/sm120/ABI 원인 기록; 구버전 cu118로 돌아가 실험을 계속하지 않음 |
| WSL UE hardware rendering 미충족 | model/raw 다운로드 보류. Windows Blocks split 경로 성공을 바탕으로 동일 Travel Windows build 또는 native Linux hardware 환경 확보 |
| assets/RPC/merged 부족 | 필요한 file/byte/version 특정; 전체 저장소/거대 dataset을 무차별 다운로드하지 않음 |
| NF4 adapter/API 실패 | 최소 loader 변경과 package 범위 조사; 성능 동등성을 가정하지 않음 |
| 동시 실행 OOM | 측정 내역으로 예산 재평가; renderer 변경/offload는 다른 구성으로 기록 |
| navigation 실패 | 원인과 raw trajectory 보존; failure module이나 training을 임의로 추가하지 않음 |

현재 단계는 Gate 1 PASS / Gate 2 PARTIAL / Gate 3 NOT_RUN으로 중단한다. 다음 checkpoint 다운로드 권고는 **NO**. Gate 2 충족 후 INT4 prototype을 검증하고, 별도 단계에서만 실제 한 episode를 고려한다. failure injection/schema/detection은 시작하지 않는다.

## 11. 이 계획의 직접 근거

- [AeroVLA requirements](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/requirements.txt), [eval command](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/scripts/eval_aerovla.sh), [wrapper](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/src/model_wrapper/aerovla_wrapper_ui.py).
- [AeroVLA server](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/airsim_plugin/AirVLNSimulatorServerTool.py), [episode loader](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/src/vlnce_src/env_uav.py), [termination](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/src/vlnce_src/closeloop_util.py), [metrics](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/utils/metric.py).
- [TravelUAV generator](https://github.com/prince687028/TravelUAV/blob/5cc26e9a4a55b9c788e918f7c3bb2dc5076a85e6/Model/LLaMA-UAV/tools/generate_merged_json.py), [single-scene guidance](https://github.com/prince687028/TravelUAV/issues/27#issuecomment-2892760034), [no official mini set](https://github.com/XuPeng23/AeroVLA/issues/16#issuecomment-5550862979).
- [고정한 환경 파일](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env/tree/44de5739a95a2f6a88767446421cddada9606642), [raw map 파일](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV/tree/faa8f2514156455ea7423464cc1295e6f92575cb), [base](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f), [adapter](https://huggingface.co/XuPeng23/AerialVLA/tree/196f2f3253b69df6e90ac10b6ae041c7b3a9569e).

공식 code/metadata에서 확인한 command 인자를 사용하며 존재하지 않는 기능이나 측정 결과를 만들지 않는다. package/OS 전체 lock과 실제 동작 성공은 다음 단계에서 확인한다.
