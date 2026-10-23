# Human Pilot Mode — 사람이 카메라만 보고 조종하고, 그 비행을 data로 기록하기

사람이 **앞(Front)과 아래(Down) 카메라 두 장과 문장만 보고** 드론을 조종합니다. 그 비행은 teacher의 녹화와 같은 형식으로 기록되어, 나중에 모델 학습 data로 쓸 수 있습니다. 모델, gate, 이전 결과는 바꾸지 않습니다.

```powershell
.\scripts\run_human_pilot.ps1
```

## 화면에 보이는 것

| 보임 | 보이지 않음 |
|---|---|
| Front 카메라, Down 카메라 (모델이 받는 256×256 그대로, 2배 확대) | 지도, 위치, 높이 |
| 문장 ("Find the blue cube and land on it.") | 목표까지의 거리·방향 |
| 내 조작 값 (전진, 상승, 회전), 속도 단계, decision 수, 녹화 여부 | Simulator 창 (밖에서 드론을 보는 시점) |

- Simulator 창은 실행 script가 숨깁니다. 정책은 그 시점을 본 적이 없기 때문입니다. 설정 확인용으로만 `-ShowSimulator`로 띄울 수 있습니다.
- 비행을 고를 때도 물체 **이름**만 고르고, 시작 위치는 고르지 않습니다. 시작은 목표에서 12–44m 떨어진 무작위 자리, 무작위 방향입니다. 사람도 모델처럼 목표가 어디 있는지 모른 채 출발합니다.
- 비행이 끝난 뒤에는 결과(착륙한 물체, 성공 여부, 기록 여부)를 보여 줍니다.

## 조작

| 키보드 | Pad (Xbox식) | 동작 |
|---|---|---|
| W / ↑ | 왼쪽 stick 위 | 전진 |
| A D / ← → | 왼쪽 stick 좌우 | 회전 |
| R / PageUp | 오른쪽 stick 위 | 상승 |
| F / PageDown | 오른쪽 stick 아래 | 하강 |
| 1 2 3 | LB / RB | 속도 단계 (느림 / 보통 / 빠름) |
| Enter | A | 끝 (APPROACH: 여기서 멈춤 / LAND: 포기) |
| Backspace | B | 이번 비행 버리기 |
| Esc | | 종료 |

비행 전: **N / P** 물체, **T** LAND / APPROACH, **L** layout, **M** 장면(field / lot), **Enter** (pad Start) 출발.

- **축은 정책과 같은 셋뿐입니다.** 전진(뒤로는 안 됨), 상승·하강, 회전. 옆으로 미끄러지기나 후진은 없습니다. 정책이 낼 수 없는 움직임을 기록하면 label로 바꿀 수 없기 때문입니다.
- **부드러운 조작.** 명령은 1초에 20번 갱신되고, 키를 누르고 떼는 변화는 가속 한도(전진·상하 1.5 m/s², 회전 1.5 rad/s²) 안에서 서서히 바뀝니다. 0.5초마다 계단식으로 바뀌는 모델 비행과 다릅니다. 화면도 조작과 함께 1초에 20번 갱신됩니다(카메라 stream을 받아 보여 줌).
- **기체 자체의 반응은 그대로입니다.** Simulator 기체는 속도 명령을 약 2.4초 늦게 따라갑니다. 이 지연은 바꾸지 않았습니다. 기록된 비행이 정책의 비행과 같은 물리로 움직여야 data가 맞기 때문입니다. 그래서 조작이 묵직하게 느껴지고, 미리 감속해야 합니다.
- **속도 단계.**

| | 전진 | 상하 | 회전 |
|---|---:|---:|---:|
| 1 느림 | 0.6 m/s | 0.53 m/s | 12.7°/s |
| 2 보통 | 1.2 m/s | 0.6 m/s | 14.4°/s |
| 3 빠름 | 2.0 m/s | 1.0 m/s | 24°/s |

모든 단계는 정책의 stop 기준의 3배 이상이고 action 한도 이하입니다. Stick은 dead zone(20%)을 넘으면 그 하한부터 한도까지 비례합니다.

## 무엇이 기록되는가

0.5초(정책의 decision 간격)마다:

1. 두 카메라 사진을 canonical loop와 똑같이 받아 PNG로 저장합니다(`scripts/visual_search.SearchEnv.observe`).
2. 그 뒤 0.5초 동안 사람이 준 명령을 **정책의 action 하나**로 바꿉니다: 전진 m, 하강 m, 회전 rad per tick. 0.5초가 조금 길거나 짧았으면 정규화하고, action 한도 안으로 자릅니다. 이것이 그 decision의 label(`teacher_action`)입니다.

정책이 이 label을 그대로 내면 executor는 같은 속도·고도·방향 명령을 내므로, 사람이 한 비행을 재현하는 action이 됩니다.

**학습에 쓰지 않는 decision (기록은 함).**

| 표시 | 뜻 | 이유 |
|---|---|---|
| `pause` | 세 축 모두 stop 기준 안 (손을 뗌) | 정책은 이것을 stop으로 읽고, 4번 이어지면 비행을 끝냅니다. 지난 세 단계에서 싸운 공중 정지의 원인입니다 |
| `below_safe` | stop은 아니지만 3배 미만 (누르기 시작·끝) | Stage 1b부터 모든 녹화 label이 지킨 여유 |

Dataset builder는 이 표시가 붙은 decision이 든 label window를 뺍니다(`excluded`). 비행 끝의 stop(APPROACH에서 Enter 뒤 4번, LAND에서 touchdown 뒤 4번)은 정책의 정상적인 끝이므로 label입니다.

**어떤 비행이 data가 되는가.** Surface evaluator가 문장대로 했다고 읽고(이름의 물체에 착륙 / 그 가까이에서 멈춤), 사람이 끝냈거나 착륙했고, 아무것도 부딪치지 않았고, simulator가 실시간의 0.8배 이상을 유지한 비행만 `episodes/`에 들어갑니다. 나머지는 `rejected/`에 사진과 함께 남아 볼 수 있지만, builder는 읽지 않습니다.

```
datasets/projectairsim_visual_search/human_pilot/
  episodes/hp-<시각>-<물체>-<task>.json      학습에 쓸 수 있는 비행
  rejected/...json                           실패·포기한 비행 (학습 안 함)
  field/hp-.../frontcamera/000000.png ...    사진 (모델 입력 그대로)
outputs/human_pilot/flights.jsonl            비행마다 한 줄: 결과, label 수, 뺀 decision 수
```

각 decision에는 simulator가 아는 값(거리, 방향, 카메라에 보였는지)도 분석용으로 함께 저장됩니다. Builder는 이를 `meta`에만 두고, 모델은 읽지 않습니다. Teacher 녹화와 같습니다.

## 어디서 날 수 있는가

- 장면: `field`, `lot`. Layout: `a b c d k l m n` (모든 generic landing 단계가 녹화한 layout).
- Check를 비행하는 layout(`t u p q r s`)과 canonical layout(`e g h i j`)은 거절합니다. 시작은 저장소의 모든 check 시작(봉인한 gate 포함)에서 6m 이상 떨어집니다. 사람이 나중의 평가 장면을 미리 보지 않게 하기 위해서입니다.

## 실행 옵션

```powershell
.\scripts\run_human_pilot.ps1 -Scene lot -Layout k -Target green_cylinder -Task land
```

- `-NoRecord`: 기록하지 않고 연습합니다.
- `-Record DIR`: 다른 dataset 폴더에 기록합니다.
- `-ShowSimulator`: simulator 창을 띄웁니다(설정 확인용, 녹화에는 쓰지 마세요).
- `-TopicsPort 9089 -ServicesPort 9090`: 다른 simulator가 떠 있을 때 다른 port로 띄웁니다.
- `-Script FILE`: 사람 대신 정해진 입력으로 날립니다(전체 loop 점검용).

## 확인 비행

사람 대신 정해진 입력(`-Script`)으로 simulator를 숨긴 채 날렸습니다. 이 비행들은 확인용 폴더에만 기록했고 dataset에는 넣지 않았습니다.

| 비행 | 결과 | Decision | Label / pause / below_safe / 끝의 stop | Decision 간격 | 화면 frame |
|---|---|---:|---|---:|---:|
| blue cube 접근 (1차) | 9.1m에서 멈춤, 성공 | 30 | 21 / 3 / 2 / 4 | 0.6–0.78초 | (세지 않음) |
| blue cube 접근 (2차, 간격 수정 뒤) | 성공 | 36 | 28 / 3 / 1 / 4 | 0.52초 | 초당 약 29 |
| blue pad 착륙 | pad 위 3.06m 지점에 착륙, 영역 안, soft, 성공 | 50 | 34 / 10 / 2 / 4 (닿은 뒤) | 0.52초 | 초당 약 29 |

- 1차 비행에서 decision 간격이 0.5초보다 길었습니다(사진을 받는 0.1–0.25초만큼 밀림). 0.5초를 사진을 요청하는 순간부터 세도록 고쳐 0.52초가 됐습니다. Canonical loop은 0.51초입니다.
- 키를 누르고 있는 동안의 label은 고른 속도 그대로였습니다: 보통 단계에서 회전 0.126 rad, 전진 0.6m, 상승 0.3m(0.5초당).
- 손을 뗀 구간(입력 script의 대기)은 모두 `pause`로, 누르기 시작·끝은 `below_safe`로 표시됐습니다.
- Simulator 창을 숨겨도 두 카메라 stream이 들어왔습니다.

## 한계

- 사람은 기억을 쓰지만(아까 왼쪽에 보였다) 모델은 지금 사진 한 쌍만 봅니다. 기억에 기댄 조작은 모델이 따라 하기 어렵습니다.
- 같은 장면에서 사람과 scripted teacher가 다르게 행동하면, 둘을 섞어 학습할 때 다시 평균되어 멈출 수 있습니다. 사람 data는 teacher가 못 하는 상황(가려진 목표 찾기, 가장자리 회복)에 쓰는 편이 낫습니다.
- 아직 이 data로 학습하지 않았습니다. 학습에 넣는 비율과 기준은 다음 단계에서 정합니다.
