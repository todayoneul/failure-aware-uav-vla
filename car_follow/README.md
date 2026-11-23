# Phase 2: 자연어로 지정한 차량을 따라가는 드론 (진행 중)

CARLA에서 자연어로 지정한 차량("오른쪽 차선의 흰색 SUV를 따라가")을 찾아 락온하고, 일정 거리와 고도를 유지하며 따라가는 드론을 만듭니다. 핵심 질문은 "비슷한 차가 많고 가려짐이 있어도 지정한 그 차를 놓치지 않는가"입니다. 연구 주제는 **실패를 스스로 알아차리는 자연어 지시 드론 추적**입니다. 닮은 차로 갈아타거나, 목표를 놓치거나, 명령이 모호할 때 드론이 이를 알아차리고 대응하게 합니다([연구 계획](docs/research_plan.md)). 저장소 전체 개요는 [상위 README](../README.md)에 있습니다.

**이 폴더에서 실행합니다.** 아래와 각 문서의 경로와 명령은 모두 `car_follow/` 기준이고, Python 환경은 이 폴더의 uv 프로젝트입니다.

## 1. 진행

| 단계 | 상태 | 문서 |
|---|---|---|
| 환경과 CARLA 0.9.16 설치 점검 | 완료 | [setup](docs/setup.md) |
| 첫 데모 계획: 같은 차종·색의 차 2–3대 사이에서 지정한 차 따라가기 (Town05) | 완료 | [demo1_plan](docs/demo1_plan.md) |
| 첫 데모 구현 1: 장면과 시험 목록 (시험 63개, 본평가 30개 봉인) | 완료 | [demo1](docs/demo1.md) 1절 |
| 연구 계획: 주제, 가설, 실패 시험 설계, 진행 순서 | 완료 | [research_plan](docs/research_plan.md) |
| CARLA-Air 확인 (CARLA 0.9.16 + 쿼드로터 비행 모델) | 완료. 쓰지 않음(드론 물리가 동기 모드 tick을 따르지 않음). CARLA 0.9.16 + 자체 동역학으로 감 | [carla_air_check](docs/carla_air_check.md) |
| 첫 데모 구현 2: 드론 이동과 추종 제어 | 다음 | |

![시험 시작 예시](outputs/examples/demo1_trials.jpg)

*첫 데모 시험의 시작 화면. 초록 상자가 목표, 주황 상자가 같은 차종·색의 차이고, 위 띠의 문장이 목표 한 대만 가리킵니다.*

## 2. 실행

```powershell
uv sync                                               # Python 3.12 환경(.venv)을 만든다
uv run python scripts/carla_check.py --quality Low   # simulator를 켜서 속도와 GPU 메모리를 잰다
uv run python scripts/make_demo1_trials.py           # 첫 데모 시험 목록을 만든다 (이미 만들어 둠)
uv run python -m unittest discover -s tests          # simulator 없이 도는 테스트
```

- CARLA 0.9.16 Windows 패키지는 `assets/carla-0.9.16/`에 풀어 둡니다. Git에는 없고, 받는 곳은 [setup](docs/setup.md) 3절에 있습니다.
- Claude 데스크톱 앱 안에서 환경을 새로 만들 때의 주의점은 [setup](docs/setup.md) 2.3절에 있습니다.

## 3. 폴더 구조

```text
pyproject.toml, uv.lock, .python-version   Python 환경 (uv)
src/car_follow/    simulator 시작·연결과 카메라(sim.py), 첫 데모의 시험·문장·봉인(trials.py)
scripts/           설치 확인(carla_check.py), 시험 목록 생성(make_demo1_trials.py), CARLA-Air 확인(carla_air/)
configs/           차종 이름표, 첫 데모 시험 목록 (본평가는 봉인)
tests/             simulator 없이 도는 테스트
docs/              아래 문서
outputs/examples/  문서에 쓰는 그림 (나머지 실행 결과는 로컬에만)
assets/            (로컬에만) CARLA 0.9.16
```

## 4. 문서

| 문서 | 내용 |
|---|---|
| [research_plan](docs/research_plan.md) | 연구 계획: 주제와 차별점, 가설, 실패 종류와 방해 차 단계, 감시기와 대응, 지표, 진행 순서, CARLA-Air 확인 항목 |
| [carla_air_check](docs/carla_air_check.md) | CARLA-Air 확인 기록: client 연결, 동기 모드, 드론에 붙인 카메라, 속도 응답, 지도와 자원, 판정 |
| [setup](docs/setup.md) | uv Python 환경, CARLA 버전 선택, 설치 확인(속도, GPU 메모리, 자동 정답, 다른 GPU 작업과 함께) |
| [demo1_plan](docs/demo1_plan.md) | 첫 데모 계획: 범위, 구성(처음 지정, 추적, 추종 제어, 화면), 채점 지표와 합격선, 구현 순서 |
| [demo1](docs/demo1.md) | 첫 데모 구현 기록: 차종 이름표, 색 이름 규칙, 시험 구성과 문장, 확인과 버림, 나누기와 봉인 |
