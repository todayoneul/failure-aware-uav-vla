# Failure-Aware UAV VLA

## Overview

**Project AirSim에서 AeroVLA가 가상 드론을 조종하고, 예상치 못한 장애를 직접 넣어 보는 오픈소스소프트웨어 과목 텀프로젝트입니다.** 영상 흐림, 가림, 조종 편향 같은 상황을 만들고, 이후 감지·복구 기능으로 확장할 계획입니다.

## What It Does

앞·아래 카메라 영상과 자연어 지시를 AeroVLA에 전달합니다. 모델이 출력한 전진·상하 이동·회전 값을 안전 범위로 제한해 드론을 움직이고, 새 영상을 받아 반복합니다.

## Current Demo

- ✅ Project AirSim 드론·Front/Down RGB 카메라
- ✅ AeroVLA NF4 추론과 실제 영상 → 행동 → 이동
- ✅ single-step 및 **10/10 closed loop**
- ✅ 기본 blur·control drift 데모 — **모델 없는 별도 script 시험**

![실제 10번째 decision의 Front/Down 영상과 AeroVLA 출력](outputs/examples/closed_loop.png)

저장된 실제 실행 화면입니다. [드론 화면](outputs/examples/drone_view.png)과 [blur·drift 데모](outputs/examples/control_drift.png)도 볼 수 있습니다.

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
src/failures/      다음 기능을 위한 빈 자리
configs/           baseline 설정과 패키지 버전
scripts/ · tests/  준비·측정 스크립트와 테스트
outputs/examples/  실제 데모 이미지 3장
```

## Current Status

[Current Baseline](docs/baseline.md): 추론 평균 **1.05초**, 전체 step **4.53초**, 전체 GPU peak **9.76GiB**. 10단계 동안 timeout·OOM·simulator crash는 없었습니다. RAM 여유가 약 1GiB이고 반복 속도는 약 0.22Hz여서 장시간 자율 비행은 아직 확인하지 않았습니다. [실제 테스트 요약](docs/experiments.md).

실패 자동 감지·진단·복구는 아직 구현하지 않았습니다. TravelUAV benchmark와 모델 재학습은 현재 과제 범위에 포함하지 않습니다. 모델·시뮬레이터 바이너리와 raw 로그는 Git에 넣지 않습니다.

## Roadmap

- [x] Project AirSim setup
- [x] AeroVLA NF4 inference
- [x] Live closed-loop drone control
- [ ] Failure injection framework
- [ ] Interactive failure controls
- [ ] Failure detection
- [ ] Basic recovery behavior
- [ ] Demo and evaluation

다음 후보는 **Gaussian Blur → Partial Occlusion → Control Drift**입니다. 구현 전 계획과 단축키 아이디어는 [failure plan](docs/failure_plan.md)에 있습니다.
