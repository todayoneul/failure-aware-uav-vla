# Human Pilot Mode — 사람이 카메라만 보고 조종하고, 그 비행을 data로 기록하기

사람이 **앞(Front)과 아래(Down) 카메라와 문장을 보고** 드론을 조종합니다. 그 비행은 teacher의 녹화와 같은 형식으로 기록되어, 나중에 모델 학습 data로 쓸 수 있습니다. 모델, gate, 이전 결과는 바꾸지 않습니다.

```powershell
.\scripts\run_human_pilot.ps1                # 조종 창만 (simulator 창은 숨김)
.\scripts\run_human_pilot.ps1 -UnrealView    # simulator(Unreal) 창도 띄우고, 그 창을 보며 조종
.\scripts\run_human_pilot.ps1 -Takeover      # 개입 모드: 모델이 먼저 날고, 막히면 사람이 넘겨받음
```

## 화면에 보이는 것

| 보임 | 보이지 않음 |
|---|---|
| Front 카메라, Down 카메라 (768×768, 아래 설명) | 지도, 위치, 높이 |
| 문장 ("Find the blue cube and land on it.") | 목표까지의 거리·방향 |
| 상태 한 줄: 비행 중 / 속도 단계 / 녹화 여부 / 모델 범위 밖 경고 | (기본 실행에서) Simulator 창 |
| 내 조작 값 막대 (전진, 상승, 회전) | |

- **고화질 화면.** 화면의 두 사진은 모델 카메라와 같은 위치·같은 화각(90°)에 붙인 표시용 카메라 두 대가 768×768로 찍어 보내는 영상입니다(초당 약 28장). 모델 카메라(256×256)를 확대한 것이 아니어서 선명합니다. **기록되는 사진은 여전히 모델이 받는 256×256 그대로**이고, 표시용 카메라는 기록에 쓰지 않습니다. 해상도는 `-ViewPx`로 바꿀 수 있습니다.
- **Decision 수는 화면에 없습니다.** Decision은 기록 단위일 뿐입니다: 모델은 0.5초마다 한 번 판단하므로, 사람의 연속 조작을 0.5초씩 묶어 모델 action 하나로 바꿔 저장합니다(아래 "무엇이 기록되는가"). 조종하는 사람은 신경 쓸 필요가 없습니다.
- 비행을 고를 때도 물체 **이름**과 **거리 단계**만 고르고, 시작 위치는 고르지 않습니다. 시작은 그 단계 안의 무작위 자리, 무작위 방향입니다. 사람도 모델처럼 목표가 어디 있는지 모른 채 출발합니다(아래 "시작 거리").
- 비행이 끝난 뒤에는 결과(착륙한 물체, 성공 여부, 기록 여부)를 보여 줍니다.

### Unreal 창을 보며 조종하기 (`-UnrealView`)

- Simulator 창(드론을 뒤에서 따라가는 시점)이 화면 왼쪽 위에 1280×720으로 뜨고, 조종 창은 그 오른쪽(x=1300)에 작게(사진 300×300) 붙습니다.
- 키는 두 창 어느 쪽이 앞에 있어도 먹힙니다. 다른 프로그램이 앞에 있으면 먹히지 않습니다.
- 이 시점은 모델이 받지 않는 정보입니다(드론 밖에서 본 모습). 그래서 이렇게 날린 비행은 기록에 `view: unreal`로 표시됩니다(`cameras`와 구분). 학습에 넣을지, 넣는다면 어떻게 다룰지는 학습 단계에서 정합니다. 기록되는 사진과 label은 두 방식이 같습니다.

## 조작

| 키보드 | Pad (Xbox식) | 동작 |
|---|---|---|
| W / S | 왼쪽 stick 위 / 아래 | 전진 / 후진 |
| A D / ← → | 왼쪽 stick 좌우 | 회전 |
| ↑ / ↓ (R / F, PageUp / PageDown) | 오른쪽 stick 위 / 아래 | 상승 / 하강 |
| 1 2 3 4 | LB / RB | 속도 단계 (느림 / 보통 / 빠름 / 부스트) |
| Enter | A | 끝 (APPROACH: 여기서 멈춤 / LAND: 포기) |
| Backspace | B | 이번 비행 버리기 |
| Esc | | 종료 |

비행 전: **N / P** 물체, **T** LAND / APPROACH, **L** layout(공 위 착륙판 layout 포함), **M** 장면(field / lot), **D** 거리 단계, **R** 무작위 목표 켜고 끄기, **Enter** (pad Start) 출발.

- **세 축을 동시에 조작할 수 있습니다.** 전진·후진하면서 상승·하강하고 회전까지 함께 됩니다(W와 ↑를 같이 누르면 앞으로 가며 올라감). 확인 비행에서 앞으로 가며 올라간 0.5초가 label 하나에 전진 1.0m, 상승 0.5m로 함께 기록됐습니다.
- **축은 정책과 같은 셋(전진, 상하, 회전)입니다.** 옆으로 미끄러지기는 없습니다. 정책은 옆으로 갈 수 없어서, 그 움직임은 label로 바꿀 수 없기 때문입니다.
- **후진은 됩니다. 다만 학습에는 쓰지 않습니다.** 정책의 전진 action은 0–1m뿐이라 뒤로 가는 action이 없습니다. 후진한 0.5초는 기록은 되지만 `backward`로 표시되어 label에서 빠집니다.
- **부드러운 조작.** 명령은 1초에 20번 갱신되고, 키를 누르고 떼는 변화는 가속 한도(전진·상하 4 m/s², 회전 3 rad/s²) 안에서 바뀝니다. 전진 0→2 m/s가 0.5초, 회전이 최고 속도까지 0.14초입니다.
- **기체 자체의 반응은 그대로입니다.** Simulator 기체는 속도 명령을 약 2.4초 늦게 따라갑니다. 이 지연은 바꾸지 않았습니다. 기록된 비행이 정책의 비행과 같은 물리로 움직여야 data가 맞기 때문입니다. 그래서 조작이 묵직하게 느껴지고, 미리 감속해야 합니다.
- **속도 단계.** 시작은 3(빠름)입니다.

| | 전진 | 상하 | 회전 | 학습 |
|---|---:|---:|---:|---|
| 1 느림 | 0.6 m/s | 0.53 m/s | 12.7°/s | 씀 |
| 2 보통 | 1.2 m/s | 0.6 m/s | 14.4°/s | 씀 |
| 3 빠름 (모델의 최고 속도) | 2.0 m/s | 1.0 m/s | 24°/s | 씀 |
| 4 부스트 | 5.0 m/s | 2.5 m/s | 60°/s | **안 씀** |

**착륙할 때는 마지막 하강을 1 또는 2단계로 하세요.** 착륙으로 인정되는 접지 속도는 0.75 m/s 이하인데, 3단계의 하강 속도는 1.0 m/s입니다. LAND 비행에서 이보다 빠르게 내려가면 화면에 `DESCENDING TOO FAST TO TOUCH DOWN`이 뜹니다. 기체가 2.4초 늦게 따라오므로 표면에 닿기 몇 초 전에 바꿔야 합니다.

1–3은 정책의 stop 기준의 3배 이상이고 action 한도 이하입니다. 4는 멀리 이동할 때 쓰는 단계로, 모델이 낼 수 없는 속도입니다. 부스트(또는 후진) 중에는 화면 상태 줄에 `BEYOND THE MODEL: not used for training`이 뜹니다. Stick은 dead zone(20%)을 넘으면 그 하한부터 고른 단계의 속도까지 비례합니다.

## 무엇이 기록되는가

0.5초(정책의 decision 간격)마다:

1. 두 카메라 사진을 canonical loop와 똑같이 받아 PNG로 저장합니다(`scripts/visual_search.SearchEnv.observe`, 256×256).
2. 그 뒤 0.5초 동안 사람이 준 명령을 **정책의 action 하나**로 바꿉니다: 전진 m, 하강 m, 회전 rad per tick. 0.5초가 조금 길거나 짧았으면 정규화하고, action 한도 안으로 자릅니다. 이것이 그 decision의 label(`teacher_action`)입니다.

정책이 이 label을 그대로 내면 executor는 같은 속도·고도·방향 명령을 내므로, 사람이 한 비행을 재현하는 action이 됩니다.

**학습에 쓰지 않는 decision (기록은 함).**

| 표시 | 뜻 | 이유 |
|---|---|---|
| `pause` | 세 축 모두 stop 기준 안 (손을 뗌) | 정책은 이것을 stop으로 읽고, 4번 이어지면 비행을 끝냅니다. 지난 세 단계에서 싸운 공중 정지의 원인입니다 |
| `below_safe` | stop은 아니지만 3배 미만 (누르기 시작·끝) | Stage 1b부터 모든 녹화 label이 지킨 여유 |
| `backward` | 그 0.5초 중 한 순간이라도 후진 | 정책에는 뒤로 가는 action이 없습니다 |
| `over_bound` | 그 0.5초 중 한 순간이라도 모델 한도보다 빠름 (부스트, 부스트에서 감속하는 중) | 정책이 낼 수 없는 속도입니다. 0.5초 평균이 한도 안이어도 뺍니다 |

Dataset builder는 이 표시가 붙은 decision이 든 label window를 뺍니다(`excluded`). 비행 끝의 stop(APPROACH에서 Enter 뒤 4번, LAND에서 touchdown 뒤 4번)은 정책의 정상적인 끝이므로 label입니다.

**어떤 비행이 data가 되는가.** 문장대로 했고, 사람이 끝냈거나 착륙했고, 아무것도 부딪치지 않았고, simulator가 실시간의 0.8배 이상을 유지한 비행만 `episodes/`에 들어갑니다. 문장대로 했는지는 이렇게 읽습니다.

- LAND: surface evaluator의 접지 기준. 이름의 물체 윗면, 착륙 영역 안, 접지 속도 0.75 m/s 이하, 그 뒤 멈춰 서 있음(`stable_physical_landing`). 그리고 **정밀 착륙**: 윗면 가운데에서 1.75m 안(pad, cube, cylinder), 공 위 착륙판은 1.3m 안.
- APPROACH: canonical 판정(그 물체 가까이에서 멈춤).

LAND에 canonical 판정을 쓰지 않는 이유가 있습니다. Canonical 판정은 simulator가 보고하는 수직 속도가 가라앉은 순간을 접지로 봅니다. 사람 비행의 마지막 4번(2초) 동안에는 이 값이 늦게 가라앉거나 아예 가라앉지 않을 수 있습니다. 실제로 첫 사람 비행 여섯 번 중에 두 번이 틀리게 분류됐습니다. 2.25 m/s의 세게 닿은 착륙은 들어갔고, 0.41 m/s로 가운데에 닿은 착륙은 수직 속도 보고가 0.06에 머물러 빠졌습니다. 기준을 고친 뒤 이 두 비행의 파일을 옮겼습니다. 나머지는 `rejected/`에 사진과 함께 남아 볼 수 있지만, builder는 읽지 않습니다.

```
datasets/projectairsim_visual_search/human_pilot/
  episodes/hp-<시각>-<물체>-<task>.json      학습에 쓸 수 있는 비행
  rejected/...json                           실패·포기한 비행 (학습 안 함)
  field/hp-.../frontcamera/000000.png ...    사진 (모델 입력 그대로, 256×256)
outputs/human_pilot/flights.jsonl            비행마다 한 줄: 결과, label 수, 뺀 decision 수, view
```

각 decision에는 simulator가 아는 값(거리, 방향, 카메라에 보였는지)도 분석용으로 함께 저장됩니다. Builder는 이를 `meta`에만 두고, 모델은 읽지 않습니다. Teacher 녹화와 같습니다. 비행 요약에는 `view`(`cameras` / `unreal`)와 표시 해상도(`view_px`)도 남습니다.

## 개입 모드 — 모델이 날다가 막히면 사람이 넘겨받기

```powershell
.\scripts\run_human_pilot.ps1 -Takeover
.\scripts\run_human_pilot.ps1 -Takeover -UnrealView
.\scripts\run_human_pilot.ps1 -Takeover -Checkpoint outputs/aerovla_oft/checkpoints/generic_landing_stage1c_run/update-2000
```

모델이 먼저 날고, 사람은 보고 있다가 모델이 잘못 갈 때 넘겨받아 고칩니다. 이렇게 모이는 data는 **모델이 실제로 간 상태에서 사람이 고친 조작**입니다. 사람이 처음부터 날린 비행과 달리, 모델이 막히는 장면(가려진 목표 앞에서 제자리 회전, 윗면 위 공중 정지)에서의 정답이 모입니다.

| 키보드 | Pad | 동작 |
|---|---|---|
| Space 또는 아무 비행 키 (W S A D ↑ ↓ …) | Y 또는 stick | 넘겨받기 (바로) |
| Space | Y | 모델에게 돌려주기 (다음 decision부터, 다시 누르면 취소) |
| Enter | A | 모델이 멈춰 기다릴 때: 그 자리에서 비행 끝내기. 사람이 날 때: 사람 조종과 같음 |

- **모델의 비행은 canonical 평가와 같습니다.** 0.5초마다 같은 함수로 사진을 받아 모델에 주고, 모델의 첫 action을 canonical executor(`SearchEnv.tick`)로 날립니다. Landing finalizer는 canonical 평가처럼 꺼져 있습니다.
- **모델은 WSL의 GPU에서 돕니다**(`src/human_pilot/remote.py`). 추론은 한 decision에 약 0.35초(첫 decision만 약 1.2초)이고, decision 간격은 0.53–0.56초입니다. `-UnrealView`를 함께 쓰면 Unreal 창과 모델이 GPU를 나눠 써서 약 0.62초로 늘어납니다. 실행하면 checkpoint를 먼저 확인하고(동결 baseline이면 fingerprint까지), WSL에서 불러오는 동안(약 1분) 메뉴에 `loading`이 뜹니다.
- **모델이 받는 것은 사진 두 장과 문장뿐입니다.** 답에는 모델이 받은 사진의 sha256이 붙어 오고, 저장한 사진과 다르면 비행을 멈춥니다.
- **넘겨받는 순간 사람의 조작은 모델이 내던 속도와 방향에서 시작합니다.** 덜컥이지 않습니다.
- **모델이 멈추면 기다립니다.** 모델이 stop을 4번 연속 내면 canonical 평가에서는 비행이 끝납니다. 개입 모드에서는 기체가 그 자리에 머물고 `MODEL STOPPED`가 뜹니다. 넘겨받아 고치거나(공중에서 멈췄으면 내려가서 착륙), Enter로 그대로 끝냅니다.
- **화면:** 상태 줄이 `MODEL FLYING`(보라) / `YOU ARE FLYING` / `MODEL STOPPED`로 바뀝니다. 모델이 날 때 막대는 모델의 명령을 보여 줍니다.
- **시작:** 사람 조종과 같은 규칙입니다(녹화 layout의 무작위 자리, check 시작에서 6m 이상, `-Seed`로 반복).

**무엇이 label인가.**

| 표시 | 뜻 |
|---|---|
| `model` | 모델이 날린 decision. 사진과 모델의 답(chunk, 추론 시간)까지 기록하지만 label이 아님 |
| `handover` | 넘겨받은 순간의 decision(앞은 모델, 뒤는 사람). label이 아님 |
| 표시 없음 | 사람이 날린 decision. 사람 조종의 규칙(`pause`, `below_safe`, `backward`, `over_bound`)을 그대로 적용 |

**어떤 비행이 data가 되는가.** 사람 조종과 같은 조건(문장대로 함, 부딪치지 않음, 실시간 0.8배 이상)에 하나가 더 붙습니다: 사람의 label이 하나 이상 있어야 합니다. 모델이 혼자 끝까지 한 비행에는 배울 것이 없기 때문입니다. 모델의 stop을 Enter로 받아들여 끝낸 비행(`model_stop`)도 성공이면 들어갑니다. 저장 위치는 `datasets/projectairsim_visual_search/takeover/`이고 id는 `tk-`로 시작합니다. 사람 혼자 날린 비행과 따로 둡니다. 비행 요약에는 checkpoint, 동결 baseline 확인 결과, 넘겨받고 돌려준 decision 번호가 남습니다.

**Checkpoint.** 기본은 Interactive Mission Control의 기본값인 동결 baseline `grounding_film`입니다. `-Checkpoint`로 다른 것을 날릴 수 있고, 그때는 확인되지 않은 checkpoint라는 경고가 나옵니다.

## 정밀 착륙

착륙 지점이 큽니다. pad는 14×14m, cube·cylinder는 8m라서 윗면 어디에 내려도 착륙으로 인정됩니다. 그런 data로 배우면 모델도 대충 내립니다. 그래서 사람 비행이 data가 되려면 **윗면 가운데 가까이** 내려야 합니다.

| 표면 | 가운데에서 | 근거 |
|---|---:|---|
| pad (14m), cube (8m), cylinder (지름 8m) | 1.75m | canonical teacher가 하강을 시작하는 반경(pad 폭의 1/8, `configs/targets/landing_pads.json` teacher.align_radius_m) |
| 공 위 착륙판 (지름 6m) | 1.3m | 위와 같은 반경이되, 표면의 착륙 영역 반경(2.6m)의 절반을 넘지 않음 |

- 장면은 바꾸지 않았습니다. pad와 물체의 크기, canonical 착륙 판정, 이전 평가는 그대로입니다. 바뀐 것은 사람 비행 중 어떤 것을 data로 남기느냐뿐입니다.
- pad에는 흰 H가 있습니다. 아래 카메라에서 H 가운데를 맞추면 됩니다.
- 비행이 끝나면 메뉴에 `landed 2.4 m from the middle (data needs 1.75 m or less)`처럼 결과가 나옵니다. 비행 중에는 거리를 보여 주지 않습니다(모델도 모르는 정보).

## 시작 거리

| 단계 (D) | 목표에서 | 뜻 |
|---|---:|---|
| near | 12–44m | canonical 평가가 확인한 범위. 대개 출발할 때 목표가 보임 |
| mid (기본) | 44–100m | 평가 범위 밖. 목표가 안 보이는 경우가 많아 찾아야 함 |
| far | 100–160m | 맵의 먼 쪽까지. 찾기와 긴 이동 |

- 160m까지는 두 장면의 모든 녹화 layout, 모든 물체에서 시작을 뽑을 수 있음을 확인했습니다.
- 단계는 비행 기록에 `distance_band`로 남습니다. 나중에 거리별로 나눠 학습하거나 평가할 수 있습니다.
- 먼 단계는 비행이 길어집니다. 4단계(부스트)로 빨리 가도 되지만, 그 구간은 학습에 쓰이지 않습니다.

## 목표: 모든 물체, 무작위

- 녹화 layout마다 물체 8개가 있습니다: blue·red pad, blue·red cube, blue·green cylinder, orange ball, blue cone. **N / P**로 고르고 **T**로 LAND / APPROACH를 바꿉니다. cone은 끝이 뾰족해서 APPROACH만 됩니다.
- **R**(또는 `-Random`)을 켜면 비행마다 layout(공 위 착륙판이 있는 것 포함), 물체, LAND / APPROACH가 무작위로 정해집니다. 착륙할 수 있는 물체는 절반은 LAND, 절반은 APPROACH입니다. 장면(field / lot)은 M으로 고른 것을 유지합니다. 무작위 비행은 기록에 `random_target: true`로 남습니다.

### 공 위 착륙판

- 공은 윗면이 둥글어서 그대로는 착륙할 수 없습니다. 녹화 layout마다 공 위에 평평한 둥근 판을 얹은 변형 `<layout>_cap`(예: `a_cap`)을 두었습니다. 판 외에는 원래 layout과 똑같습니다.
- 판은 Interactive Mission Control의 `mission_cap` 판과 같은 모양입니다: 지름 6m, 두께 1.2m, 흰색. 판의 가장자리에서 공의 표면에 걸쳐 놓이고, 판 윗면은 공 꼭대기보다 0.2m 위입니다. 판 위에 내리면 공에 착륙한 것으로 판정되고, 공의 다른 곳에 닿으면 충돌입니다(`configs/landing/surfaces.json`의 `cap` 규칙).
- 판 layout은 실행할 때 메모리에서 만듭니다. `configs/maps/field.json`과 `lot.json`은 동결 baseline의 fingerprint에 들어 있어서 고치지 않았습니다(고치면 동결 확인이 실패하고 개입 모드가 시작되지 않음).

## 어디서 날 수 있는가

- 장면: `field`, `lot`. Layout: `a b c d k l m n` (모든 generic landing 단계가 녹화한 layout)과 각각의 공 위 착륙판 변형 `a_cap … n_cap`.
- Check를 비행하는 layout(`t u p q r s`)과 canonical layout(`e g h i j`)은 거절합니다. 시작은 저장소의 모든 check 시작(봉인한 gate 포함)에서 6m 이상 떨어집니다. 사람이 나중의 평가 장면을 미리 보지 않게 하기 위해서입니다.

## 실행 옵션

```powershell
.\scripts\run_human_pilot.ps1 -Scene lot -Layout k -Target green_cylinder -Task land
```

- `-Takeover`: 개입 모드(위). `-Checkpoint DIR`로 다른 checkpoint를, `-Distro`로 다른 WSL 배포판을 씁니다.
- `-UnrealView`: simulator 창을 1280×720으로 띄우고 그 창을 보며 조종합니다(`view: unreal`로 기록).
- `-ViewPx 1024`: 표시용 카메라 해상도(기본 768). 기록되는 사진에는 영향이 없습니다.
- `-Band near|mid|far`: 시작 거리 단계(기본 mid). 메뉴에서 D로도 바꿉니다.
- `-Random`: 비행마다 layout, 물체, LAND / APPROACH를 무작위로 정합니다. 메뉴에서 R로도 켜고 끕니다.
- `-Layout a_cap`: 공 위 착륙판 layout에서 시작합니다.
- `-Seed 24`: 같은 시작에서 다시 날립니다. 시작은 seed로 정해지므로, 같은 seed는 같은 자리·방향입니다(비행마다 1씩 늘어남).
- `-NoRecord`: 기록하지 않고 연습합니다.
- `-Record DIR`: 다른 dataset 폴더에 기록합니다.
- `-TopicsPort 9089 -ServicesPort 9090`: 다른 simulator가 떠 있을 때 다른 port로 띄웁니다.
- `-Script FILE`: 사람 대신 정해진 입력으로 날립니다(전체 loop 점검용).

## 자주 보는 메시지

- `Timed out while trying to unsubscribe from all topics.` Simulator client가 카메라 영상 구독을 끊으려 했는데, simulator가 1초 안에 요청을 받지 않았다는 경고입니다. 새 비행의 장면을 불러올 때와 종료할 때 구독을 끊습니다. Simulator가 먼저 닫혔거나 장면을 불러오느라 바쁘면 나옵니다. 구독은 장면·연결과 함께 사라지고, 비행 기록은 그 전에 이미 저장되므로 무시해도 됩니다.
- 종료할 때 `pynng ... Object closed`가 붙은 traceback: 같은 client library가 종료 순서에 따라 가끔 내는 메시지입니다. 이전 데모들에서도 나왔고 결과에 영향이 없습니다.

## 확인 비행

사람 대신 정해진 입력(`-Script`)으로 날렸습니다. 이 비행들은 확인용 폴더에만 기록했고(또는 기록하지 않았고) dataset에는 넣지 않았습니다.

| 비행 | 화면 | 결과 | Decision | Label / pause / below_safe / backward / over_bound / 끝의 stop | Decision 간격 | 화면 frame |
|---|---|---|---:|---|---:|---:|
| blue cube 접근 (첫 버전) | 256 확대 | 성공 | 36 | 28 / 3 / 1 / – / – / 4 | 0.52초 | 초당 약 28 |
| blue pad 착륙 (첫 버전) | 256 확대 | pad 위에 착륙, 영역 안, soft, 성공 | 50 | 34 / 10 / 2 / – / – / 4 (닿은 뒤) | 0.52초 | 초당 약 28 |
| 전진 → 전진+상승 → 후진 → 부스트 → 멈춤 | 768 | 끝까지 기록 | 32 | 9 / 9 / 0 / 6 / 4 / 4 | 0.52초 | 초당 약 28 |
| blue pad 착륙, 3단계로 하강 | 768 | pad 위, 영역 안, 0.96 m/s로 접지 → 기록 안 됨(rejected) | 46 | 31 / 10 / 1 / 0 / 0 / 4 | 0.51초 | 초당 약 28 |
| blue pad 착륙, 2단계로 하강 (같은 시작) | 768 | pad 위, 영역 안, soft → 기록됨(episodes) | 50 | 33 / 11 / 2 / 0 / 0 / 4 | 0.51초 | 초당 약 28 |
| 짧은 접근, `-UnrealView` | Unreal + 300 | 끝까지 기록, 실시간 0.999배 | 15 | 6 / 5 / 0 / 0 / 0 / 4 | 0.51초 | 초당 약 28 |

- 전진+상승 구간의 label은 `[1.0, -0.5, 0]`(전진 1m, 상승 0.5m per 0.5초)로, 두 축이 한 label에 함께 들어갔습니다.
- 후진 2.5초는 6개 decision(5개 + 바뀌는 순간 1개)이 `backward`로, 부스트 1.5초는 3개 + 부스트에서 감속하던 1개가 `over_bound`로 표시됐습니다.
- `-UnrealView`: simulator 창 안쪽이 1280×720, 화면 (8, 31)에 놓였고(제목 표시줄이 화면 안), 조종 창은 그 오른쪽 (1300, 0)에 붙었습니다. Unreal 창을 띄운 접근 비행 일곱 번의 실시간 비율은 0.90–1.00이었습니다.
- 회귀 확인: 비행 loop를 개입 모드에 맞게 바꾼 뒤 위의 2단계 pad 착륙(기록됨, label 34)과 후진·부스트 비행(`backward` 6, `over_bound` 4)을 다시 날려 같은 결과를 얻었습니다.
- 첫 버전 비행에서 decision 간격이 0.5초보다 길었던 것(0.6–0.78초)은 0.5초를 사진을 요청하는 순간부터 세도록 고쳐 0.52초가 됐습니다. Canonical loop은 0.51초입니다.

### 거리 단계, 무작위 목표, 공 위 착륙판 확인

- **공과 판의 실제 위치(simulator에서 측정):** field와 lot 모두 공은 지면에서 0–10.0m, 판은 9.0–10.2m에 있습니다. 계산한 대로 판이 공 위에 걸쳐 있습니다.
- **판 위 착륙:** 공의 위치를 아는 간단한 제어기로 두 장면의 `a_cap`에서 판 위에 내렸습니다(장면과 판정 점검용, data 아님). 두 번 다 접지가 공(`orange_ball`)으로 판정됐고, 가운데에서 0.30m, 0.20 m/s, 멈춰 섬, 정밀 착륙 기준(1.3m) 통과였습니다.
- **조종 loop에서:** `-Layout a_cap -Target orange_ball -Task land -Band far`로 153m에서 출발했고, 판이 있는 장면이 그대로 떴습니다. `-Random -Band near`는 layout c, red cube, APPROACH, 24m 시작을 뽑았습니다. 개입 모드도 같은 메뉴로 날려 이전과 같은 결과를 얻었습니다.
- **정밀 착륙 기준을 이미 기록된 비행에 적용:** 사람 조종 2개와 개입 모드 7개가 data로 남아 있었는데, 모두 가운데 1.75m 안(개입 모드는 0.29–1.62m)이라 옮겨진 것이 없습니다. 반면 사람이 넘겨받지 않은 동결 모델의 pad 착륙 6번은 가운데에서 1.77–3.46m로 모두 기준 밖입니다. 기준을 바꾸면 `scripts/refile_human_flights.py --apply`로 다시 나눕니다.

### 개입 모드 확인 비행

동결 baseline `grounding_film`(fingerprint 확인됨)으로, 사람 대신 정해진 입력으로 날렸습니다. 모두 field / layout a / seed 7 시작이고, 확인용 폴더에만 기록했습니다.

| 비행 | 무슨 일이 있었나 | 결과 | Decision | model / handover / 사람 label / 그 외 사람 decision |
|---|---|---|---:|---|
| blue cube 접근 | 모델 13 decision → 넘겨받아 전진·회전 → 돌려줌 → 모델이 7.2m에서 stop 4번 → 기다림 → Enter | 성공, 기록됨 (`model_stop`) | 28 | 17 / 1 / 9 / pause 1 |
| blue cube 접근 | 모델이 10.5m에서 stop 4번 → 18초 기다림 → 넘겨받아 전진 → Enter | 성공, 기록됨 (`human_stop`) | 35 | 22 / 0 / 4 / pause 4, below_safe 1, 끝의 stop 4 |
| blue cube 착륙, 개입 없음 | 모델이 혼자 cube 윗면 가장자리(가운데에서 4.9m, 영역 밖)에 닿음 | 실패, 기록 안 됨 | 72 | 72 / 0 / 0 / – |
| blue cube 착륙, 개입 | 같은 시작. 모델이 cube 12m 앞에 왔을 때 넘겨받아 조금 올라가 다가간 뒤 2단계로 하강 | 윗면 영역 안(가운데에서 2.45m), 0.57 m/s, 성공, 기록됨 | 88 | 57 / 1 / 20 / pause 6, 끝의 stop 4 |
| blue cube 접근, `-UnrealView` | 넘겨받기 2번, 돌려주기 1번 | 성공 (기록하지 않음) | 54 | 32 / 2 / 10 / pause 6, 끝의 stop 4 |

- 모델의 추론은 decision마다 약 0.35초였고(첫 decision 1.2초), 모델이 받은 사진의 sha256은 매번 저장한 사진과 같았습니다(다르면 비행이 멈춤).
- 넘겨받은 순간 사람의 조작은 모델이 내던 1.94 m/s에서 시작해 그 decision의 사람 몫이 2 m/s로 이어졌습니다.
- Dataset builder의 규칙으로 첫 비행을 읽으면 label window는 사람이 날린 decision 14–19에서만 시작합니다. 넘겨받은 decision(13)이나 손을 뗀 decision(23)이 든 window는 빠집니다.
- 매 비행 뒤 WSL의 policy server는 pilot이 연결을 닫자 스스로 끝났습니다.
- 개입 없는 착륙과 개입한 착륙은 같은 시작에서 날렸습니다. 모델이 혼자 가면 가장자리에 닿고(Stage 1d의 가장자리 문제와 같은 모습), 사람이 넘겨받아 고치면 영역 안에 내립니다. 개입 모드가 모으려는 data가 바로 이 차이입니다.

## 한계

- 사람은 기억을 쓰지만(아까 왼쪽에 보였다) 모델은 지금 사진 한 쌍만 봅니다. 기억에 기댄 조작은 모델이 따라 하기 어렵습니다. Unreal 창을 보며 날린 비행은 더 그렇습니다(밖에서 본 시점).
- 같은 장면에서 사람과 scripted teacher가 다르게 행동하면, 둘을 섞어 학습할 때 다시 평균되어 멈출 수 있습니다. 사람 data는 teacher가 못 하는 상황(가려진 목표 찾기, 가장자리 회복)에 쓰는 편이 낫습니다.
- 부스트와 후진은 빠르게 자리를 옮기는 데는 편하지만, 그 구간은 학습에 쓰이지 않습니다. 학습할 비행은 1–3단계로 날리는 편이 data가 많이 남습니다.
- 개입 모드에서 모델이 표면에 닿으면 비행은 4 decision 뒤 끝나고, 그 뒤에는 넘겨받을 수 없습니다. 판정이 첫 접지로 정해지기 때문입니다. 가장자리로 가는 것이 보이면 닿기 전에 넘겨받아야 합니다.
- 아직 이 data로 학습하지 않았습니다. 학습에 넣는 비율과 기준은 다음 단계에서 정합니다.
