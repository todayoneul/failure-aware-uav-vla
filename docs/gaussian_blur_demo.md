# Gaussian Blur — 실행과 관찰

Windows Project AirSim의 실제 Front/Down 영상을 **blur → resize/mosaic → processor → AeroVLA** 순서로 전달한다. 화면만 흐리게 만드는 기존 model-free 데모와 다르다. 기본은 **NORMAL / OFF / MEDIUM**이다.

## 이 PC에서 실행

PowerShell을 열어 저장소 폴더로 이동한 뒤 한 명령을 실행한다.

```powershell
.\scripts\run_blur_demo.ps1
```

이미 준비한 Windows Blocks와 client, WSL2 Ubuntu inference 환경, local OpenVLA/AeroVLA checkpoint를 재사용한다. Launcher가 simulator, 관찰 창, resource monitor, WSL model runner를 시작한다. 모델·map·package는 다운로드하지 않는다. 다른 simulator/client가 켜져 있으면 먼저 그 창을 종료한다.

Windows PowerShell 5.1과 PowerShell 7 모두 지원한다. 초기화·종료 제어는 Python 파일을 실행하므로 `python -c`의 따옴표 전달 방식에 의존하지 않는다. Redirect된 WSL process의 handle을 먼저 보존해 5.1에서 정상 종료 코드를 놓치지 않도록 했다.

모델 로딩 중에는 `Loading OpenVLA NF4 + AeroVLA LoRA`가 표시된다. 로딩과 camera 연결이 끝나면 비행과 실제 모델 추론이 시작된다. 별도 WSL 터미널 명령은 필요 없다. 기본 최대 30 decisions이며 바꾸려면 `-MaxSteps 12`처럼 지정한다.

## 조작

관찰 창 제목은 **Gaussian Blur Demo | AeroVLA inputs and actions**다. 창을 클릭한 뒤 영문 입력 상태로 키를 누르거나, 위쪽 버튼을 클릭한다.

| 입력 | 동작 |
|---|---|
| B 또는 Blur 버튼 | ON/OFF 전환 |
| 1 / Low | LOW: 7×7, sigma 1.5 |
| 2 / Medium | MEDIUM: 15×15, sigma 3.0 |
| 3 / High | HIGH: 31×31, sigma 6.0 |
| Q / Esc / 창 닫기 | 종료 요청 → 현재 제한된 작업 종료 → 착륙·disarm |

키 변경은 **다음 camera observation부터** 적용한다. 진행 중인 inference의 입력을 중간에 바꾸지 않는다. 요청이 대기 중이면 `Request applies at NEXT observation`을 표시한다. 기본 target은 Front/Down 둘 다이며 Chase는 blur하지 않는다.

로딩 중 종료 요청은 모델 로딩 경계를 지난 후 처리될 수 있다. 최대 decision 수에 도달하면 착륙하고 마지막 화면을 남긴다. Q/Esc로 관찰 창을 닫으면 launcher가 자기 simulator와 monitor를 정리한다.

## 무엇을 보면 되나

- **큰 왼쪽 화면:** 실제 드론 외부 시점. Blur 상태에서도 드론은 선명하게 보인다.
- **오른쪽 Front/Down:** 해당 `input step`에서 AeroVLA가 실제 받은 영상. Blur ON이면 여기부터 흐려진다.
- **FAILURE / SEVERITY:** 현재 모델 decision에 적용된 상태. 위쪽 버튼은 다음 관측에 대한 요청이다.
- **파란 진행 상태:** camera 수신, 모델 loading/inference, bounded action, cleanup 순서다.
- **Prompt given to model:** 해당 관측과 pose로 구성해 모델에 전달한 실제 prompt다.
- **Model output / bins:** 모델이 생성한 원문 행동 토큰이다.
- **Decoded model action:** 토큰을 거리(m)와 yaw(deg)로 해석한 값이다.
- **Actual bounded command:** 기존 안전 범위로 제한해 실제 drone API에 전달한 값이다.
- **Last checked state:** 마지막으로 확인한 높이·방향·위치다. 실시간 Chase와 달리 decision 경계에서 갱신한다.
- **Actual input: VERIFIED:** 표시된 입력 hash와 실제 `generate()`에 전달한 tensor 기록을 검증했다는 뜻이다.

이 VLA는 내부 판단을 설명하는 문장을 출력하지 않는다. 따라서 ‘사고 과정’은 위의 **관측 → prompt → 생성값 → 해석 → 실제 명령 → 이동 결과**를 확인하는 형태로 제공한다. 출력만 보고 행동의 원인을 추정해 표시하지 않는다.

예를 들어 저장된 blur decision은 raw `75 49 61`, 해석된 forward 3.827m / yaw 15.4°였다. 실제 명령은 forward 0.5m / yaw 15°로 제한했다. 화면에서 세 값을 나란히 볼 수 있다.

## 직접 확인할 순서

1. NORMAL 상태로 2~3 decisions를 본다.
2. B를 눌러 `GAUSSIAN BLUR`로 바꾼다. 다음 입력에서 Front/Down이 흐려지는지 본다.
3. 2~3 decisions 동안 모델 출력과 실제 명령을 관찰한다. 1/2/3으로 강도를 바꿔도 된다.
4. B를 다시 눌러 NORMAL로 복귀하는지 본다.
5. Q/Esc로 끝내고 착륙·종료를 기다린다.

Blur 때문에 모델이 실패해야 이 기능이 성공한 것은 아니다. 성공 기준은 **실제 입력에 blur 적용, ON/OFF 조작, inference와 action 실행**이다.

## 기록 확인

`outputs/failure_demo/`에 최근 실행을 기록한다. 전체 camera frame dump는 만들지 않는다.

재실행하면 이전에 프로그램이 만든 사진·로그는 `outputs/failure_demo/runs/<UTC시각-식별자>/`에 보존하고 현재 파일을 새로 시작한다. 사용자가 따로 만든 메모나 다른 디렉터리는 이동하지 않는다. `run-info.json`으로 이번 실행과 이전 archive를 구분한다.

| 파일 | 확인할 것 |
|---|---|
| `worker.log`, `worker-errors.log` | 모델 로딩·각 decision·오류 |
| `closed-loop.json`, `closed-loop-log.jsonl` | 단계별 failure·raw/parsed/clipped action·pose·이동·latency |
| `control-events.jsonl` | B/강도/종료 요청과 자동 검증 여부 |
| `normal_front/down.png`, `blur_front/down.png`, `comparison.png` | 동일 blur decision의 원본 기준과 실제 failed input 비교 |
| `restored_front/down.png` | OFF 복귀 후 대표 입력 |
| `observer_normal/blur/restored.png` | 실제 관찰 창의 대표 화면 |

`input_evidence`에는 사용한 frame SHA, mosaic SHA, GPU BF16 tensor SHA, 원본 기준 tensor SHA와 비교 결과가 있다. OFF에서 원본 기준과 일치하는지 확인한다. ON에서 실제 영상이 변했다면 tensor도 달라지는지 확인한다. 균일한 단색 영상은 Gaussian blur를 적용해도 같을 수 있으며, 이 경우 차이를 만들어 기록하지 않는다.

## 수행한 live 검증

2026-10-05, 다음 명령으로 동일 B 제어 경로를 자동 호출했다. 물리 키 누름을 자동화한 시험은 아니다.

```powershell
.\scripts\run_blur_demo.ps1 -AutoTest
```

NORMAL 2 → MEDIUM BLUR 2 → NORMAL 2, **6/6 완료**. 실제 GPU tensor의 원본 기준 대비 변경 여부는 `false,false,true,true,false,false`였다. Device `cuda:0`, BF16, shape `[1,6,224,224]`. 각 decision에서 실제 이동을 확인했고 timeout/OOM은 없었다. Land True, disarm 뒤 LANDED, cleanup error 없음.

Blur ON 두 pair의 처리 시간은 약 **9.02 / 0.53ms**였다. 첫 AeroVLA generation은 **2.880초**, 이후 5회 평균은 약 **1.075초**였다. 이 작은 시연에서의 측정이며, 강도별 성능 비교나 navigation 평가가 아니다. 다른 pose의 행동 차이를 blur의 인과 효과로 단정하지 않는다.

자동 감지·분류·복구·replanning·재학습·benchmark·다른 failure는 이번 작업에 포함하지 않는다.

실행/표시/종료 경계 보완 뒤 추가 정상 1-step도 실제 추론·이동·land/disarm에 성공했다. 소유 worker 종료 확인, 현재 run의 stale blur 파일 부재를 검사했다. 24개 Python 회귀 테스트와 Windows/WSL 인자 경계 smoke 2개가 통과했다. LOW/HIGH는 모듈 테스트, 실제 live ON/OFF 기록은 MEDIUM이다.

종료 요청에 정상 응답하지 않는 경우에는 제한된 대기 뒤 **동일 run-token과 script 경로로 확인한 worker만** 종료하고 자기 simulator를 닫는다. 이 강제 종료는 정상 착륙으로 기록하지 않는다. `launcher-cleanup.json`의 `terminated` / `forced`를 확인할 수 있다.
