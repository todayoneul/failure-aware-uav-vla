# Setup

현재 Baseline은 **Windows Project AirSim + WSL2 AeroVLA**다. 아래는 실제 사용한 환경과 실행 순서다. 이번 repository 정리에서는 설치·다운로드·비행을 다시 실행하지 않았다.

## 환경

- Windows 11, RTX 5070 12GB, NVIDIA driver 591.86
- WSL2 Ubuntu 24.04.4, Python 3.10.14
- Windows Blocks 배포 1.0.1, Windows demo Python 3.12.14
- PyTorch 2.7.1+cu128 / torchvision 0.22.1+cu128
- transformers 4.42.4 / bitsandbytes 0.48.2 / PEFT 0.11.1 / accelerate 0.32.1
- projectairsim 1.0.2 / pynng 0.9.0 / timm 0.9.10 / scipy 1.15.3 / OpenCV 4.11.0.86

직접 사용한 패키지는 [integration-requirements.txt](../configs/integration-requirements.txt), Windows model-free client는 [windows-client-requirements.txt](../configs/windows-client-requirements.txt)에 고정했다. 전체 간접 의존성 freeze는 로컬 기록에 있다. 새 PC의 전체 설치는 이번 정리에서 재검증하지 않았다.

## 경로

PowerShell은 저장소 루트에서 실행한다. WSL에서도 **같은 저장소**를 열고 다음 변수를 잡는다.

```bash
export REPO_ROOT="$(pwd)"  # WSL에서 저장소 루트로 이동한 후
export UAV_VLA_HOME="${UAV_VLA_HOME:-$HOME/uav-vla-smoke}"
PYTHON="$UAV_VLA_HOME/integration/bin/python"
```

| 대상 | 위치 |
|---|---|
| Windows simulator | `assets/projectairsim-blocks-1.0.1/Blocks/Binaries/Win64/Blocks-Win64-Shipping.exe` |
| Windows demo client | `assets/projectairsim-env/Scripts/python.exe` |
| WSL inference env | `$UAV_VLA_HOME/integration/` |
| OpenVLA cache | `$UAV_VLA_HOME/models/hf/models--openvla--openvla-7b/snapshots/<revision>/` |
| LoRA cache | `$UAV_VLA_HOME/models/hf/models--XuPeng23--AerialVLA/snapshots/<revision>/aero_vla/` |
| Local model manifest | `outputs/integration/model-downloads.json` |
| Scene/camera settings | `configs/scene_basic_drone.jsonc`, `configs/robot_quadrotor_fastphysics.jsonc` |

Base revision은 `47a0ec7fc4ec123775a391911046cf33cf9ed83f`, LoRA revision은 `196f2f3253b69df6e90ac10b6ae041c7b3a9569e`다. 로더는 local manifest의 실제 snapshot 경로를 읽고 offline으로 실행한다. 다른 PC에서는 해당 PC의 manifest를 준비해야 한다.

## 처음 준비하는 환경에서만

기존 정상 환경·모델·simulator를 재사용할 수 있다. 아래 설치 명령은 준비가 안 된 환경에서 사용할 안내이며 자동으로 실행되지 않는다.

PowerShell에서 pinned Blocks와 Windows demo client를 준비한다. Blocks archive는 약 646.6MiB다.

```powershell
.\scripts\prepare_projectairsim.ps1
py -3.12 -m venv assets/projectairsim-env
.\assets\projectairsim-env\Scripts\python.exe -m pip install -r configs/windows-client-requirements.txt
```

WSL에서 `uv`와 Python 3.10.14를 준비한 뒤 inference env를 만든다. 기존 bundled uv는 `$UAV_VLA_HOME/tools/uv-x86_64-unknown-linux-gnu/uv`에 있다.

```bash
uv venv --python 3.10.14 "$UAV_VLA_HOME/integration"
uv pip install --python "$PYTHON" -r configs/integration-requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip check --python "$PYTHON"
```

Bundled uv를 쓰는 기존 환경은 `bash scripts/setup_integration.sh`를 사용할 수 있다. 모델이 없는 경우에만 `"$PYTHON" scripts/download_integration_models.py`를 실행한다. 두 고정 checkpoint의 합계는 약 **15.55GB**이며 `configs/*-model.json`의 크기·SHA와 대조하고 local manifest를 만든다. 이번 작업에서는 모델을 다운로드하지 않았다.

## 실행 순서

1. Windows에서 Project AirSim을 실행한다.
2. WSL inference 환경과 현재 Windows host 주소를 확인한다.
3. resource monitor와 기존 gate 결과 파일을 확인한다.
4. integration runner가 모델을 로딩한 뒤 직접 client를 연결하고 single-step → 10-step을 실행한다.

Windows에서 simulator를 별도 터미널로 실행한다.

```powershell
& '.\assets\projectairsim-blocks-1.0.1\Blocks\Binaries\Win64\Blocks-Win64-Shipping.exe' -windowed -ResX=960 -ResY=540
```

다른 PowerShell에서 현재 simulator의 resource monitor를 실행한다.

```powershell
New-Item -ItemType Directory -Force outputs/communication_final | Out-Null
$sim = Get-Process -Name 'Blocks-Win64-Shipping'
.\scripts\communication_final_resources.ps1 -SimulatorProcessId $sim.Id
```

WSL에서 기존 cache와 manifest를 사용한다.

```bash
WINDOWS_HOST=$(ip -4 route show default | awk '/default/ {print $3; exit}')
"$PYTHON" src/integration/closed_loop_runner.py --host "$WINDOWS_HOST"
```

현재 runner는 로컬 `outputs/communication_final/gate-a.json`, `gate-b.json`의 PASS와 monitor의 `windows-latest.json`을 요구한다. 이는 완료한 reconnect·제한된 resource-pressure 시험 결과다. **Git에는 raw 결과가 없으므로 새 clone만으로 위 명령이 즉시 비행하지 않는다.** 현재 시험 PC에서는 기존 파일을 재사용한다. 다른 PC·네트워크에서는 이전 PASS를 가져오는 대신 그 환경의 연결을 먼저 검증해야 한다. NAT 주소는 매번 확인하고 simulator client는 한 개만 연결한다.

Runner를 재실행하면 기존 로컬 결과 파일 일부가 갱신된다. 완료된 feasibility 시험을 자동 반복하지 않는다. 디버그 viewer는 Windows client 환경에서 `integration_debug_viewer.py --output outputs/communication_final`을 별도로 실행할 수 있다.

## 모델 없는 기존 데모 / 테스트

`run_projectairsim_demo.ps1`는 자체 simulator를 시작하므로 기존 simulator와 동시에 실행하지 않는다. 기존 B/W toggle 및 자동 blur/drift를 보여주는 **model-free** 시험이다.

```powershell
.\scripts\run_projectairsim_demo.ps1 -AutoFailures
```

WSL에서 simulator·checkpoint를 로딩하지 않는 회귀 테스트:

```bash
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
```

상세 조건: [communication](archive/communication_stability_test.md), [NF4 loader](archive/aerovla_int4_validation.md), [최종 loop](archive/final_closed_loop_validation.md).
