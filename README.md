# Failure-Aware UAV VLA

**Project AirSim의 가상 드론을 Vision-Language-Action 모델(AeroVLA)로 조종하고, 영상 흐림 같은 장애를 직접 넣어 보는 오픈소스소프트웨어 과목 텀프로젝트입니다.** 지금은 "모델이 무엇을 할 수 있고 무엇을 못 하는지"를 실제 비행으로 측정하는 단계이고, 장애 감지와 복구는 그다음입니다.

![Project AirSim에서 비행 중인 드론](outputs/examples/hero_drone.png)

## 한눈에 보기

| 기능 | 상태 | 실행 |
|---|---|---|
| Front/Down 영상 → AeroVLA → 드론 이동 (closed loop) | 동작 | `.\scripts\run_blur_demo.ps1` |
| Gaussian Blur를 실제 모델 입력에 주입 | 동작 | 같은 창에서 B, 1/2/3 |
| 맵에서 목표를 골라 보내기 (방향 힌트 사용) | 동작. 20m 안 정지는 조건에 따라 0–60% | `.\scripts\run_mission_demo.ps1` |
| 문장만 주고 찾아가기 (방향 힌트 없음) | AeroVLA-OFT pilot으로 이 맵에서 동작. 다른 맵은 미확인 | `.\scripts\run_visual_search_demo.ps1` |
| 장애 자동 감지·복구 | 아직 없음 | - |

모델과 simulator는 저장소에 없습니다. 준비 방법은 [setup](docs/setup.md)에 있습니다.

## Demo

### 1. 문장만 주고 찾아가기 — Visual Search

![AeroVLA-OFT가 뒤돌아 선 상태에서 파란 원뿔을 찾아 접근하는 실제 비행](outputs/examples/visual_search_oft.gif)

`Find the blue cone.` 한 문장과 Front/Down 영상만 받은 **AeroVLA-OFT**의 실제 비행입니다(2배속). 목표가 뒤에 있어 처음에는 보이지 않습니다. 오른쪽으로 돌며 찾고, 보이면 그쪽으로 틀어 접근하고, 11m 앞에서 스스로 멈춥니다. 목표 좌표와 방향 힌트는 모델에 들어가지 않습니다. 십자 표시는 화면용 복사본에만 있고, 모델 입력이 카메라 원본과 같은지 hash로 확인해 표시합니다.

```powershell
.\scripts\run_visual_search_demo.ps1 -Model baseline   # 기존 AeroVLA
.\scripts\run_visual_search_demo.ps1 -Model oft        # AeroVLA-OFT (checkpoint를 학습한 PC에서)
```

![같은 시작 조건에서 기존 AeroVLA와 AeroVLA-OFT의 실제 궤적](outputs/examples/visual_search_trajectories.jpg)

같은 시작 조건에서의 실제 궤적입니다. 위는 기존 AeroVLA, 아래는 AeroVLA-OFT입니다.

| 시작 조건 (각 6회) | 기존 AeroVLA | AeroVLA-OFT |
|---|---:|---:|
| A. 목표가 정면에 보임 | 3/6 | 6/6 |
| B. 목표가 화면 가장자리에 보임 | 0/6 | 6/6 |
| C. 목표가 안 보임 | 0/6 | 5/6 |
| D. 접근 중 강제로 돌려 놓침 ([GIF](outputs/examples/visual_search_oft_reacquire.gif)) | - | 6/6 |
| 판단 주기 | 6.3초 | 0.52초 |

- **AeroVLA-OFT**는 AeroVLA adapter를 고정한 채 OpenVLA-OFT 방식의 head를 얹은 것입니다. 한 번의 forward로 연속값 행동 4개를 내고 그중 1개를 실행합니다. teacher가 비행한 72 episode로 RTX 5070 12GB에서 39분 학습한 pilot입니다.
- **같은 맵, 같은 두 물체에서만 확인했습니다.** 학습에 없던 위치에서 시작하면 2/6이고, 다른 맵은 시험하지 않았습니다.

[구조·데이터·학습·결과·한계](docs/aerovla_oft.md)

### 2. 맵에서 목표를 골라 보내기 — Coordinate Goal Mode

![Blue cone 미션의 실제 화면: 102m 비행 뒤 모델이 LAND를 내고 목표 3.6m에 착륙](outputs/examples/mission_landmark.png)

```powershell
.\scripts\run_mission_demo.ps1
```

맵의 평평한 지점이나 landmark를 클릭하고 **G**로 시작합니다. 이 모드는 목표 좌표로 계산한 방향 문장(`Fly forward-left and find the target. …`)을 모델에 줍니다. 모델이 LAND를 내면 착륙하고, 그 지점이 목표 20m 안이면 성공입니다.

| 키 | 기능 |
|---|---|
| 맵 클릭 / N | 목표 선택 / landmark 차례로 선택 |
| G / R | 시작 / 끝난 미션 초기화 |
| M | prompt 모드: 방향 힌트 + 설명 → 설명만 → 지시문만 |
| B, 1/2/3 | Gaussian Blur ON/OFF, 강도 |
| Q / Esc | 중단하고 착륙 |

- 기본은 멈추지 않고 이어서 나는 **연속 비행**입니다. `-Flight step`을 붙이면 판단마다 멈춥니다.
- 블록 위를 클릭하면 드론은 목표 45m 안에서 그 면보다 6m 위로 올라갑니다.

[조작과 화면 설명](docs/mission_demo.md)

### 3. 장애 주입 — Gaussian Blur

![같은 시점의 원본 영상과 실제로 모델에 들어간 흐린 영상](outputs/examples/gaussian_blur_comparison.png)

```powershell
.\scripts\run_blur_demo.ps1
```

왼쪽은 원본, 오른쪽은 실제 모델 입력입니다. 관찰 창에서 **B**로 켜고 끄고 **1/2/3**으로 강도를 바꿉니다. [관찰 창](outputs/examples/gaussian_blur_observer.png)에는 prompt, 모델 출력, 실행 명령이 함께 나옵니다. 자동 감지와 복구는 아직 없습니다. [안내](docs/gaussian_blur_demo.md)

### 4. 드론과 모델 입력

| | |
|---|---|
| ![전진과 회전](outputs/examples/drone_flight.gif) | ![10번째 decision의 Front/Down 영상과 AeroVLA 출력](outputs/examples/closed_loop.png) |
| 관찰 카메라로 본 드론. 화면을 보여주기 위한 scripted 비행입니다 | AeroVLA가 실제로 받은 Front/Down 영상과 그때의 출력 |

모델 없이 영상 흐림과 조종 편향을 넣어 본 초기 시연은 [control_drift.png](outputs/examples/control_drift.png)에 있습니다.

## 측정 결과

모두 Project AirSim Blocks 맵에서의 실제 비행입니다. 같은 조건도 실행마다 결과가 갈리므로 성공률이 아니라 관찰 기록으로 읽어야 합니다.

![조건별 실제 궤적 69회](outputs/examples/model_evaluation.jpg)

| 질문 | 결과 | 문서 |
|---|---|---|
| 방향 힌트가 있으면 목표로 가는가 | 35회 중 33회 접근, 12회는 20m 안에서 스스로 정지 | [Model Self-Evaluation](docs/model_evaluation.md) |
| 방향 힌트를 빼면 | 13회 모두 첫 step에 LAND | 같은 문서 |
| 설명과 힌트가 다른 물체를 가리키면 | 힌트 쪽으로 감 | 같은 문서 |
| 블록 위 목표 ([그림](outputs/examples/model_evaluation_roof.jpg)) | 42m 거리에서 위로 접근하면 6/6, 97m 출발점에서는 0/9 | [해당 절](docs/model_evaluation.md#2-블록-위-목표) |
| 말로만 시키면 (기존 AeroVLA) | 지시한 물체 20m 안 정지 0/6 | [해당 절](docs/model_evaluation.md#3-지시문만으로-찾아가기) |
| 미션이 `invalid_action`으로 끝나던 이유 | 숫자 토큰 하나가 원본 OpenVLA의 action 토큰으로 바뀜. 디코더를 고친 뒤 0회 | [해당 절](docs/model_evaluation.md#1-invalid-출력은-토큰-하나가-바뀐-이동-명령이었다) |
| 연속 비행의 효과 | 정지 시간 18–21% → 5–8%. 벽 충돌은 더 잦았음(10회 중 5회 대 8회 중 1회) | [연속 비행](docs/model_evaluation.md#연속-비행) |
| 힌트 없이 찾게 학습시킬 수 있는가 | 이 맵에서는 가능(위 Visual Search 표) | [AeroVLA-OFT](docs/aerovla_oft.md) |

## System Architecture

```text
                      ┌─ Coordinate Goal Mode ──────────────────────────────┐
Project AirSim        │ Front + Down RGB + "Fly {방향} and find the target" │
(Windows)             │        → AeroVLA NF4 → 숫자 3개 또는 LAND            │
   │  Front / Down    └─────────────────────────────────────────────────────┘
   ├────────────────►
   │                  ┌─ Visual Search Mode ────────────────────────────────┐
   │                  │ Front + Down RGB + "Find the blue cone."            │
   │                  │        → AeroVLA-OFT → 연속값 행동 4개, 1개 실행      │
   │                  └─────────────────────────────────────────────────────┘
   ◄──────────────── Action Adapter ← forward / down / yaw        (WSL2)
```

simulator는 Windows에서, 모델은 WSL2 Ubuntu에서 실행하고 직접 통신합니다. 두 모드는 서로 독립이고 기존 AeroVLA baseline은 그대로 남아 있습니다.

## Environment

RTX 5070 12GB / Windows + WSL2 / Project AirSim Blocks / OpenVLA-7B + AeroVLA LoRA / NF4 + BF16 compute.

- 기존 AeroVLA: 추론 약 1.0초, 판단 한 번 약 4.5–6.3초, GPU peak 9.76GiB ([baseline](docs/baseline.md))
- AeroVLA-OFT: 추론 약 0.4초(simulator와 함께), 학습 peak 9.8GiB

## Repository Structure

```text
src/integration/     모델 loader, 카메라·행동 변환, 실행기
src/mission/         지도 좌표, landmark, 미션 상태와 판정
src/failures/        Gaussian Blur와 control protocol
src/visual_search/   Visual Search episode, teacher 정책
src/aerovla_oft/     AeroVLA-OFT 모델과 행동·chunk 규칙
scripts/             launcher(.ps1), 평가·학습·요약 스크립트
configs/             비행·미션 한도, landmark, 평가 protocol, OFT 설정
tests/               simulator 없이 도는 테스트 (Python 110개 + PowerShell 4개)
docs/                아래 문서
outputs/examples/    README에 쓰는 실제 화면과 GIF
```

## 문서

| 문서 | 내용 |
|---|---|
| [setup](docs/setup.md) | 환경 준비와 실행 순서 |
| [baseline](docs/baseline.md) | 초기 closed loop 측정 |
| [gaussian_blur_demo](docs/gaussian_blur_demo.md) | Blur 주입 데모 |
| [mission_demo](docs/mission_demo.md) · [full_map_grounding](docs/full_map_grounding.md) | 미션 데모 조작, 지도와 Inspector |
| [model_evaluation](docs/model_evaluation.md) | 하네스 수정, 69회 평가, 연속 비행, 디코더, 블록 위 목표, 지시문 |
| [aerovla_oft](docs/aerovla_oft.md) | Visual Search Mode와 AeroVLA-OFT |
| [experiments](docs/experiments.md) | 날짜별 실험 요약 |
| [failure_plan](docs/failure_plan.md) | 이후 넣을 장애 후보 |

## 한계

- **맵 하나에서 본 결과입니다.** Blocks 맵, 물체 몇 개, 조건당 3–6회입니다.
- **Coordinate Goal Mode의 방향 힌트는 목표 좌표에서 나옵니다.** 카메라만으로 얻는 정보가 아닙니다.
- **AeroVLA-OFT는 pilot입니다.** teacher의 고정 규칙을 흉내 내고, 학습 구역 밖에서는 약합니다. LAND와 착륙은 넣지 않았습니다.
- **장애물 회피가 없습니다.** 충돌하면 simulator가 기체를 고정해 그 비행은 끝납니다.
- **모델 전체 fine-tuning, TravelUAV benchmark는 하지 않았습니다.** 학습은 LoRA pilot뿐입니다.
- 모델 weight, checkpoint, dataset, simulator, raw 로그는 Git에 넣지 않습니다.

## Roadmap

- [x] Project AirSim setup, AeroVLA NF4 inference, live closed loop
- [x] Gaussian Blur input injection with interactive controls
- [x] Interactive target selection and mission runner
- [x] Upstream-equivalent action execution, model-decided landing
- [x] Model self-evaluation with landmarks and prompt ablations
- [x] Continuous flight, grammar-constrained decoding, roof approach
- [x] Visual Search Mode and the AeroVLA-OFT pilot
- [ ] Visual search beyond the training map and start region
- [ ] Visual Search + Gaussian Blur
- [ ] Additional failure types
- [ ] Failure detection and basic recovery

다음은 **Visual Search의 일반화 확인**입니다. 시작 구역과 거리를 넓힌 데이터로 다시 학습해, 학습 범위 밖에서의 결과(현재 2/6)가 오르는지 봅니다. 그 뒤에 Visual Search에 Blur를 넣어 봅니다.
