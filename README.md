# Failure-Aware UAV VLA

## Overview

**Project AirSim에서 AeroVLA가 가상 드론을 조종하고, 예상치 못한 장애를 직접 넣어 보는 오픈소스소프트웨어 과목 텀프로젝트입니다.** 영상 흐림, 가림, 조종 편향 같은 상황을 만들고, 이후 감지·복구 기능으로 확장할 계획입니다.

![Project AirSim에서 비행 중인 드론의 가까운 외부 시점](outputs/examples/hero_drone.png)

Project AirSim의 실제 드론입니다. 관찰 카메라를 가까이 배치해 기체와 주변 장애물을 함께 볼 수 있습니다.

## What It Does

앞·아래 카메라 영상과 자연어 지시를 AeroVLA에 전달합니다. 모델이 출력한 전진·상하 이동·회전 값대로 드론을 움직이고, 새 영상을 받아 반복합니다. 모델이 LAND를 출력하면 착륙합니다.

## Current Demo

- ✅ Project AirSim 드론·Front/Down RGB 카메라
- ✅ AeroVLA NF4 추론과 실제 영상 → 행동 → 이동
- ✅ single-step 및 **10/10 closed loop**
- ✅ **Interactive Gaussian Blur injection — 실제 AeroVLA 입력에 적용**
- ✅ **Interactive mission target selection** — 맵 클릭 또는 landmark(파란 원뿔·주황 공·색 벽) 선택
- ✅ **Full-map / Target Grounding Inspector** — 전체 Blocks와 실제 prompt·camera visibility 표시
- ✅ **Mission runner** — 모델 행동을 원래 크기로 실행하고, 모델의 LAND로 끝내 착륙 후 판정
- ✅ **Model self-evaluation** — 고정 시작점·landmark·prompt 조건별 측정: 방향 힌트가 있으면 35회 중 33회가 목표 쪽으로 접근하고 12회는 20m 안에서 스스로 정지, 힌트가 없으면 13회 모두 출발하지 않음
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

![Blue cone landmark 미션의 실제 화면: 102m 비행 뒤 모델 LAND, 목표 3.6m에 착륙](outputs/examples/mission_landmark.png)

맵의 평평한 지점이나 landmark를 클릭하고 **G**로 시작합니다. **N: landmark 차례로 선택, M: prompt 모드(방향 힌트 + 설명 → 설명만 → 지시문만), R: 완료 후 초기화, F/C/WASD/+/-/V: 지도 조작, B: Blur, Q/Esc: 중단·착륙**입니다. 블록 위를 클릭하면 그 블록의 종류와 색이 설명 문장이 되고, 드론은 목표 45m 안에 들어왔을 때 그 면보다 6m 위로 올라갑니다. 모델은 첫 미션에서 기존 캐시로 로딩합니다. 드론은 모델이 낸 거리만큼(최대 5m/step) 실제로 이동하고, 모델이 LAND를 내면 착륙한 뒤 그 지점이 목표 20m 안인지로 성공을 판정합니다. [현재 조작](docs/mission_demo.md#현재-조작-2026-10-06-이후).

오른쪽 Inspector는 모델에 실제로 들어간 prompt를 보여줍니다. 좌표·거리·지도·빨간 X는 모델 입력이 아닙니다.

### 모델은 실제로 목표를 찾아가나요?

![조건별 실제 궤적](outputs/examples/model_evaluation.jpg)

하네스를 고친 뒤 고정 시작점에서 69회 비행시켜 측정했습니다. **방향 힌트가 있으면 날아갑니다.** 힌트가 목표를 가리킨 35회(초기 거리 20m 초과) 중 33회가 목표 쪽으로 접근했고, 12회는 목표 20m 안에서 모델이 스스로 LAND를 냈습니다. 위 대화형 데모 화면도 102m 떨어진 파란 원뿔까지 가서 3.6m 지점에 착륙한 실제 실행입니다.

**카메라만으로 설명된 물체를 찾아가지는 못합니다.** 방향 힌트를 빼면 13회 모두 첫 step에서 LAND였고, 설명과 힌트가 다른 물체를 가리키면 힌트 쪽으로 갔습니다. 정지 위치도 실행마다 달라, 같은 조건이 8m에서 멈추기도 하고 34m에서 멈추거나 목표물에 부딪히기도 했습니다. [무엇을 고쳤는지·방법·전체 결과·한계](docs/model_evaluation.md).

### 블록 위에도 착륙하나요? 말로만 시켜도 되나요?

![지붕 목표와 지시문 방식의 실제 궤적](outputs/examples/model_evaluation_roof.jpg)

**블록 위는 가까운 곳에서 위로 접근할 때만 됩니다.** 모델은 고도를 거의 바꾸지 않아서, 낮게 날아가면 16m 건물의 벽 앞에서 멈추거나 부딪힙니다(지붕 착륙 0/3). 42m 거리에서 지붕보다 6m 위로 올라간 뒤 출발하면 6회 모두 목표 10m 안에서 모델이 스스로 멈췄고 5회는 그 지붕에 착륙했습니다. 97m 떨어진 기본 출발점에서는 9회 모두 실패했습니다. 가는 도중에 멈추거나, 지붕 위까지 가서 멈추지 않았습니다. 데모는 블록 위 목표가 45m 안에 들어오면 그 면보다 6m 위로 올라갑니다.

**"파란 원뿔 위에 착륙해라, 돌아다니며 카메라로 찾아라"처럼 말로만 시키면 찾아가지 못합니다.** 출발은 하지만 지시한 물체 20m 안에서 멈춘 경우가 0/6입니다. 이 모델은 매 step 방향 힌트를 받는 형태로만 학습됐고, 영상 한 장만 보고 판단해 지나온 곳을 기억하지 못합니다.

그 과정에서 미션이 `invalid_action`으로 끝나던 원인도 찾았습니다. 이동 명령의 숫자 토큰 하나가 원본 OpenVLA의 action 토큰으로 바뀐 출력이었고, 유효한 토큰만 고르도록 디코더를 바꾼 뒤로는 그런 종료가 없습니다. [상세](docs/model_evaluation.md#깨진-출력-블록-위-목표-지시문만으로-찾아가기).

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
docs/              setup · baseline · experiments · model_evaluation · failure_plan
docs/archive/      이전 기술 검증 기록
src/integration/   모델·카메라·행동 변환과 실행 코드
src/failures/      Gaussian Blur와 작은 control protocol
src/mission/       지도 좌표·landmark·미션 상태·성공/실패 판정
configs/           baseline 설정, 비행·미션 한도, landmark, 평가 protocol
scripts/ · tests/  준비·측정 스크립트와 테스트
outputs/examples/  드론·관찰·AI·장애 화면과 짧은 GIF
```

## Current Status

[Current Baseline](docs/baseline.md): 추론 평균 **1.05초**, 전체 step **4.53초**, 전체 GPU peak **9.76GiB**. 10단계 동안 timeout·OOM·simulator crash는 없었습니다. RAM 여유가 약 1GiB이고 반복 속도는 약 0.22Hz여서 장시간 자율 비행은 아직 확인하지 않았습니다. [실제 테스트 요약](docs/experiments.md).

미션 실행기는 모델 행동을 원래 크기로 실행하며, 69회 평가에서 추론 평균 **0.99초**, 이동 명령 평균 **4.4초**, step당 이동 평균 **2.9m**였습니다. 고도 이탈 실패와 프로그램 오류는 없었습니다. [Model Self-Evaluation](docs/model_evaluation.md).

실패 자동 감지·진단·복구는 아직 구현하지 않았습니다. TravelUAV benchmark와 모델 재학습은 현재 과제 범위에 포함하지 않습니다. 모델·시뮬레이터 바이너리와 raw 로그는 Git에 넣지 않습니다.

## Roadmap

- [x] Project AirSim setup
- [x] AeroVLA NF4 inference
- [x] Live closed-loop drone control
- [x] Gaussian Blur input injection
- [x] Interactive controls for Gaussian Blur
- [x] Interactive target selection and mission runner
- [x] Upstream-equivalent action execution, model-decided landing
- [x] Model self-evaluation with landmarks and prompt ablations
- [ ] Continuous flight: re-plan in the air instead of stopping every step (kept on the `feat/continuous-flight` branch)
- [x] Grammar-constrained action decoding; roof targets approached from above
- [x] Instruction-only prompt mode and its evaluation (the model does not follow it)
- [ ] Reaching a described landmark and stopping near it
- [ ] Additional failure types
- [ ] Failure detection
- [ ] Basic recovery behavior
- [ ] Demo and evaluation

다음 권장 작업은 **가장 안정적이었던 조건(hint + landmark, colored wall / blue cone)을 기준선으로 삼아 Blur 같은 failure의 영향을 정지 거리 분포로 비교**하는 것입니다. 실행 간 편차가 커서 조건당 5회 이상이 필요합니다. 나머지 Failure 후보는 [failure plan](docs/failure_plan.md)에 계획으로 남겨두었습니다.
