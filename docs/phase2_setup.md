# Phase 2 환경 준비: CARLA 차량 추종

이 문서는 Phase 2(자연어로 지정한 차량을 찾아 락온하고 따라가는 드론)의 실행 환경을 기록한다. 현재 상태는 다음과 같다.

| 항목 | 상태 |
|---|---|
| Python 환경 (uv) | 완료 (2026-11-12) |
| CARLA 버전 선택 | 0.9.16으로 정함 (3절) |
| CARLA simulator 설치와 실행 확인 | 아직 하지 않음 (4절) |

Phase 1의 환경(WSL의 inference env, `assets/projectairsim-env`)은 바꾸지 않았다. Phase 1 환경은 [setup](setup.md)에 기록되어 있다.

## 1. 구성

Phase 2는 모두 Windows에서 실행한다. CARLA simulator, Python client, 관찰 화면, 이후의 인식 모델이 한 곳에서 돈다.

- CARLA 0.9.16은 Windows 패키지와 Windows용 Python wheel을 함께 제공한다.
- client와 simulator가 같은 PC의 `localhost`로 통신한다. Phase 1처럼 매번 WSL NAT의 host 주소를 확인할 필요가 없다.
- GPU(RTX 5070 12GB)는 하나이므로 simulator와 인식 모델이 같은 메모리를 나눠 쓴다. 그래서 4절에서 simulator가 차지하는 메모리를 먼저 잰다.

WSL은 지금 쓰지 않는다. lock 파일은 Linux x64도 함께 풀어 두었으므로, 나중에 uv 0.13 이상을 WSL에 설치하면 같은 lock으로 환경을 만들 수 있다. 다만 이것은 아직 확인하지 않았다(현재 WSL의 uv는 0.12.23).

## 2. Python 환경 (uv)

### 2.1 버전

| 항목 | 값 |
|---|---|
| uv | 0.13.7 (`%USERPROFILE%\.local\bin\uv.exe`) |
| Python | CPython 3.12.16 (uv가 관리) |
| carla | 0.9.16 |
| numpy | 2.5.4 |
| opencv-python | 5.0.0.93 |

- 의존성은 [`pyproject.toml`](../pyproject.toml)에 적고, 정확한 버전과 hash는 [`uv.lock`](../uv.lock)에 고정했다. Python 버전은 `.python-version`(3.12)에 있다.
- Python을 3.12로 정한 이유: carla 0.9.16의 wheel은 CPython 3.10–3.12만 있다. 그중 가장 최신을 골랐다.
- 대상 플랫폼은 Windows x64와 Linux x64 두 가지로 제한했다(`tool.uv.environments`). carla wheel이 이 두 플랫폼에만 있다.
- `carla` client 버전은 simulator 버전과 같아야 하므로 0.9.16으로 고정했다.

### 2.2 만들기와 확인

저장소 루트에서 실행한다.

```powershell
uv sync
uv run python -c "import carla; print(carla.Client('localhost', 2000).get_client_version())"
```

두 번째 명령은 `0.9.16`을 출력한다. 이 명령은 simulator에 연결하지 않고 client 라이브러리만 확인한다. 환경은 저장소의 `.venv/`에 만들어지며 Git에는 올리지 않는다.

### 2.3 Python 설치 위치

uv가 받은 Python은 `%USERPROFILE%\.local\share\uv\python`에 두었다. uv의 기본 위치(`%APPDATA%\uv\python`)를 쓰지 않은 이유는 다음과 같다.

- Claude 데스크톱 앱 안에서 실행한 명령이 AppData에 쓰면, 그 파일은 실제 AppData가 아니라 앱 전용 폴더에 저장된다.
- 그러면 `.venv`가 가리키는 Python이 앱 밖의 PowerShell에서는 보이지 않고, `.venv`가 깨진다.
- 위 위치는 AppData 밖이므로 앱 안과 밖에서 같은 파일이 보인다.

앱 밖의 PowerShell에서 `.venv`를 새로 만들 때는 그대로 `uv sync`를 실행하면 된다. 앱 안에서 다시 만들 때는 먼저 위치를 지정한다.

```powershell
$env:UV_PYTHON_INSTALL_DIR = "$env:USERPROFILE\.local\share\uv\python"
uv sync
```

## 3. CARLA 버전 선택

2026-11-12 기준으로 공개된 최신 release는 두 가지다.

| 항목 | 0.9.16 | 0.10.0 |
|---|---|---|
| 엔진 | Unreal Engine 4 | Unreal Engine 5 |
| 공개일 | 2025-09-16 | 2024-12-19 |
| Windows 패키지 | `CARLA_0.9.16.zip` 7.81GB, 추가 맵 `AdditionalMaps_0.9.16.zip` 7.25GB | `Carla-0.10.0-Win64-Shipping.zip` 10.20GB |
| PyPI client | `carla` 0.9.16: Windows와 Linux, CPython 3.10–3.12 | `carla-ue5-api` 0.10.0: Linux만, CPython 3.11–3.12 |

0.9.16을 고른 이유는 다음과 같다.

1. **GPU 메모리.** 12GB를 인식 모델과 나눠 써야 한다. UE4 기반이 UE5 기반보다 가볍다.
2. **Windows client.** Windows용 wheel이 PyPI에 있어 uv로 바로 설치된다. 0.10.0은 PyPI에 Linux wheel만 있다.
3. **자동 정답.** 0.9.16 changelog에 bounding box의 `actor_id`와, instance segmentation에서 actor ID를 쓰는 기능이 들어 있다. 화면 속 어느 차가 지정한 차인지 simulator가 직접 알려 주므로, 추적 정답과 채점을 자동으로 만들 수 있다.

맵 목록과 차량 종류는 설치 후 실제 패키지에서 확인해 4절에 기록한다.

## 4. 설치 확인 계획

아직 하지 않았다. 다음 순서로 진행한다.

1. **내려받기.** `CARLA_0.9.16.zip`(7.81GB)을 `downloads.carlasim.com`에서 받아 `assets/carla-0.9.16/`에 푼다. `assets/`는 Git에서 제외되어 있다. 추가 맵(7.25GB)은 첫 확인에 필요하지 않으므로 나중에 정한다.
2. **실행.** simulator를 품질 Low와 Epic에서 각각 켜고, client가 `localhost:2000`에 연결되는지 본다.
3. **차량.** Traffic Manager로 차량 여러 대를 자동 주행시킨다.
4. **드론 시점.** 공중에 카메라를 두고 매 tick 직접 옮겨 드론처럼 쓴다. 1인칭 영상과 3인칭 영상(드론과 차량이 함께 보이는 시점)을 함께 받는다.
5. **측정.** 품질별로 FPS와 GPU 메모리를 잰다. 인식 모델에 남는 메모리가 이 값으로 정해진다.

결과는 이 문서에 더한다.

## 참고

- CARLA releases: https://github.com/carla-simulator/carla/releases
- PyPI `carla`: https://pypi.org/project/carla/
- PyPI `carla-ue5-api`: https://pypi.org/project/carla-ue5-api/
- uv: https://docs.astral.sh/uv/
