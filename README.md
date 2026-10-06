# Failure-Aware UAV VLA

## Overview

**Project AirSim에서 AeroVLA가 가상 드론을 조종하고, 예상치 못한 장애를 직접 넣어 보는 오픈소스소프트웨어 과목 텀프로젝트입니다.** 영상 흐림, 가림, 조종 편향 같은 상황을 만들고, 이후 감지·복구 기능으로 확장할 계획입니다.

![Project AirSim에서 비행 중인 드론의 가까운 외부 시점](outputs/examples/hero_drone.png)

Project AirSim의 실제 드론입니다. 관찰 카메라를 가까이 배치해 기체와 주변 장애물을 함께 볼 수 있습니다.

## What It Does

앞·아래 카메라 영상과 자연어 지시를 AeroVLA에 전달합니다. 모델이 출력한 전진·상하 이동·회전 값을 안전 범위로 제한해 드론을 움직이고, 새 영상을 받아 반복합니다.

## Current Demo

- ✅ Project AirSim 드론·Front/Down RGB 카메라
- ✅ AeroVLA NF4 추론과 실제 영상 → 행동 → 이동
- ✅ single-step 및 **10/10 closed loop**
- ✅ **Interactive Gaussian Blur injection — 실제 AeroVLA 입력에 적용**
- ✅ **Interactive mission target selection** — 맵 클릭으로 실제 좌표 선택
- ✅ **Full-map / Target Grounding Inspector** — 전체 Blocks와 실제 방향 prompt·camera visibility 표시
- ✅ **Goal-based mission runner** — 이동·호버 성공, 목표 착륙은 아직 실패
- ✅ 기본 blur·control drift 데모 — **모델 없는 별도 script 시험**

### 드론이 어떻게 움직이나요?

![실제 scripted flight의 전진과 회전](outputs/examples/drone_flight.gif)

실제 simulator에서 촬영한 약 8초의 전진·회전 시연입니다. 이 GIF는 관찰 화면을 보여주기 위한 scripted control이며, 아래의 AeroVLA 실행 기록과 구분합니다.

큰 Chase 화면 옆에 Front/Down 영상과 현재 명령·고도·방향을 함께 보여주는 [관찰 창](outputs/examples/observer_view.png)도 준비했습니다. 실행: `./scripts/run_demo_view.ps1` — [사용 안내](docs/setup.md#관찰-화면-demo).

### AeroVLA는 무엇을 보고 판단하나요?

![실제 10번째 decision의 Front/Down 영상과 AeroVLA 출력](outputs/examples/closed_loop.png)

AeroVLA에 전달된 Front/Down RGB와 생성된 forward/down/yaw 행동입니다. 기존 10-step closed loop의 실제 저장 화면입니다.

### 장애 상황은 어떻게 보여주나요?

실제 AI 입력을 흐리게 하는 Gaussian Blur를 켜고 끌 수 있습니다. 관찰 창에서 **B: ON/OFF, 1/2/3: 강도, Q/Esc: 종료** 또는 버튼을 사용합니다.

```powershell
.\scripts\run_blur_demo.ps1
```

![같은 시점의 원본 영상과 실제 blurred AeroVLA 입력](outputs/examples/gaussian_blur_comparison.png)

왼쪽은 원본 기준, 오른쪽은 실제 모델 입력입니다. [새 관찰 창](outputs/examples/gaussian_blur_observer.png)에는 prompt·모델 원문 행동·해석된 값·안전 제한 후 명령이 함께 표시됩니다. [실행과 관찰 안내](docs/gaussian_blur_demo.md).

### 기존 모델 없는 장애 시연

![기존 blur와 control drift 시연](outputs/examples/control_drift.png)

모델 없는 별도 script에서 영상 흐림과 조종 편향을 주입한 데모입니다. 자동 감지·복구는 이후 구현합니다. [이전 드론 화면](outputs/examples/drone_view.png)도 그대로 보존했습니다.

### 맵에서 목표를 골라 이동시키려면?

```powershell
.\scripts\run_mission_demo.ps1
```

![목표·경로·모델 입력과 행동을 함께 보여주는 실제 미션 화면](outputs/examples/mission_runner.png)

맵의 평평한 지점을 클릭하고 **G: 이동, H: 호버, L: 착륙 시험**을 누릅니다. **V: 시점, +/-: 확대·축소, R: 완료 후 초기화, B: Blur, Q/Esc: 중단·착륙**입니다. 모델은 첫 미션에서 기존 캐시로 로딩합니다. 실제 이동·호버가 한 번씩 성공했으며, 착륙은 XY 오차 **0.50m > 기준 0.45m**로 실패했습니다. [실행 방법·실제 결과·한계](docs/mission_demo.md).

이제 **F: 전체 맵, C: 드론 중심, WASD: pan**을 지원합니다. 오른쪽 Inspector에서 목표 좌표가 모델의 **coarse 방향 문장**으로 바뀌는 과정을 볼 수 있습니다. Exact XYZ·거리·빨간 X는 모델 입력이 아닙니다. Near/Medium/Far(약 2/9.5/68m)는 이번 시험에서 실패했으며, [실제 화면·결과·visibility 해석](docs/full_map_grounding.md)에 그대로 기록했습니다. Visual landmark navigation은 아직 구현하지 않았습니다.

## System Architecture

```text
Project AirSim → Front + Down RGB → AeroVLA NF4
       ↑                                ↓
     Drone ← Action Adapter ← forward / down / yaw
```

시뮬레이터는 Windows에서, 모델은 WSL2 Ubuntu에서 실행하며 직접 통신합니다.

## Environment

RTX 5070 12GB / Windows + WSL2 / Project AirSim / OpenVLA-7B + AeroVLA LoRA / NF4 + BF16 compute. 환경과 실행 순서는 [setup](docs/setup.md)에 있습니다.

## Repository Structure

```text
docs/              setup · baseline · experiments · failure_plan
docs/archive/      이전 기술 검증 기록
src/integration/   모델·카메라·행동 변환과 실행 코드
src/failures/      Gaussian Blur와 작은 control protocol
src/mission/       지도 좌표·미션 상태·성공/실패 판정
configs/           baseline 설정과 패키지 버전
scripts/ · tests/  준비·측정 스크립트와 테스트
outputs/examples/  드론·관찰·AI·장애 화면과 짧은 GIF
```

## Current Status

[Current Baseline](docs/baseline.md): 추론 평균 **1.05초**, 전체 step **4.53초**, 전체 GPU peak **9.76GiB**. 10단계 동안 timeout·OOM·simulator crash는 없었습니다. RAM 여유가 약 1GiB이고 반복 속도는 약 0.22Hz여서 장시간 자율 비행은 아직 확인하지 않았습니다. [실제 테스트 요약](docs/experiments.md).

실패 자동 감지·진단·복구는 아직 구현하지 않았습니다. TravelUAV benchmark와 모델 재학습은 현재 과제 범위에 포함하지 않습니다. 모델·시뮬레이터 바이너리와 raw 로그는 Git에 넣지 않습니다.

## Roadmap

- [x] Project AirSim setup
- [x] AeroVLA NF4 inference
- [x] Live closed-loop drone control
- [x] Gaussian Blur input injection
- [x] Interactive controls for Gaussian Blur
- [x] Interactive target selection / GO_TO / GO_TO_AND_HOVER
- [ ] GO_TO_AND_LAND success
- [ ] Additional failure types
- [ ] Failure detection
- [ ] Basic recovery behavior
- [ ] Demo and evaluation

다음 권장 작업은 **미션 목표가 현재 coarse 방향 prompt에 어떻게 표현되는지 분석**하는 것입니다. 나머지 Failure 후보는 [failure plan](docs/failure_plan.md)에 계획으로 남겨두었습니다.
