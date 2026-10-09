# AeroVLA-OFT — Language–vision grounding architecture 실험

2026-10-10 실행. 브랜치 `exp/aerovla-oft-grounding-architecture`(`exp/aerovla-oft-gen-v3c-hard-negative`에서 분기). [Gen-v3c](gen_v3c_hard_negative_grounding.md)의 checkpoint, 결과, 문서는 그대로 있다. Canonical test set 48개와 Depot의 156개는 계속 봉인돼 있고, 이번에도 비행하지 않았다.

**요약**

- **Canonical clean baseline이 처음으로 섰다.** FiLM을 넣은 checkpoint가 pilot, 새 검증 gate(34/36), 그리고 봉인해 둔 canonical test 48개를 한 번에 통과했다: **47/48**, 착륙 36/36, 다른 물체 1회, 충돌 0, 문장에 따른 끝맺음 48/48.
- **그러나 그것이 FiLM 덕이라는 근거는 없다.** 같은 일정을 FiLM 없이 돌린 대조 실험도 pilot을 통과했고(30/32, 회귀 27/33), 같은 실패(`red pad ← red cube`)를 고쳤다. Pilot에서의 개선은 같은 data로 2,000 update를 더 학습한 결과다.
- **FiLM만의 몫은 32회 중 2회 차이(0 대 2)이고, 반복 편차와 구별되지 않는다.** 학습된 FiLM은 "blue landing pad"와 "red landing pad" 문장에 사실상 같은 조절을 낸다.
- **Experiment 2(cross-attention)는 실행하지 않았다.** FiLM이 pilot을 통과했기 때문이다. 모듈은 구현돼 있고 학습하지 않았다.
- **검증 gate의 통과는 한 비행 차이였다.** 다른 물체 2회(기준 2 이하). 목표 선택을 묻는 17개 시작을 한 번 더 비행하자 3회였다. 목표 선택은 여전히 가장 약한 곳이다.
- **회귀는 조금 깎였다.** 이전 set의 33개 시작에서 28 → 26(허용 한계).
- **Gen-v3c의 "데이터로는 부족하다"는 판정은 일찍 멈춘 checkpoint에 대한 것이었다.** 검증 L1로 고른 update 1,250에서는 gate를 못 넘었다. 같은 data를 2,000 update 더 학습한 두 checkpoint는 pilot의 그 실패를 고쳤고, 그중 FiLM을 넣은 쪽이 gate와 test까지 갔다(control은 pilot까지만 비행했다). 검증 L1은 이 차이를 보여 주지 못한다.
- **Gaussian Blur로 넘어갈 준비: YES.** 동결한 checkpoint와 test 48개의 결과가 clean 기준이다. Blur는 실행하지 않았다.

## 질문

> 지금의 wrong-target 실패는, 문장이 시각 표현을 충분히 강하게 조건화하지 못해서 생기는가?

[Gen-v3c](gen_v3c_hard_negative_grounding.md)에서 hard-negative data를 더한 보정은 gate를 다시 통과하지 못했다(33/36, 다른 물체에서 끝난 비행 3회). 실패는 전부 "지시한 것과 색이나 형태가 같은 물체가 먼저 보이는" 시작이었고, 그 밖은 21/21이었다. 그래서 데이터를 더 넣지 않고 구조를 두 단계로 시험한다.

1. **Experiment 1 — Visual feature FiLM.** 문장으로 시각 feature의 채널을 조절한다.
2. **Experiment 2 — Language→vision cross-attention adapter.** FiLM이 통과하지 못할 때만. 문장의 단어가 patch를 직접 고른다.

한 번에 구조 실험은 둘까지만 한다.

## 고정한 것

| | |
|---|---|
| Dataset | Gen-v3c의 병합 data 그대로(1,160 episode / 57,329 frame). Episode를 더하지 않았다 |
| 뽑는 방식 | Gen-v3c 보정과 같음: 절반은 hard-negative episode, 목표 선택 frame 3배, 먼 작은 목표 2배 |
| Teacher, 착륙 규칙, finalizer | 그대로 |
| Evaluator | `canonical_evaluator_v2` |
| Action | 전진·하강·yaw, L1 head, chunk 4, execute 1 |
| 범위 | 시작 44m 이하. 44m 초과는 이 실험과 무관하다 |
| 시작 checkpoint | `generalization_v3c`. 두 실험 모두 여기서 시작한다(Experiment 2는 FiLM checkpoint 위에 쌓지 않는다) |
| 봉인 | Canonical test 48개, Depot 156개 |

## 평가: 같은 16개 시작을 모든 모델이 두 번씩

Gen-v3c에서 같은 시작·같은 checkpoint의 비행이 성공과 충돌로 갈린 적이 있었다. 그래서 이번 pilot은 **시작마다 두 번** 비행하고, 한 번의 성공보다 "두 번 중 몇 번 맞았는가"를 본다.

[configs/grounding_architecture_pilot.json](../configs/grounding_architecture_pilot.json). Seed 23000번대. 어떤 episode도 녹화한 적 없는 배치 여섯(t, u, p–s)에서, 이전의 모든 시작(학습, 검증, pilot, test)에서 6m 이상 떨어진 새 시작이다. Gen-v3c의 검증 시작을 다시 쓰지 않았다. 구조를 학습하기 전에 commit했다(`e1ec665`).

| 묶음 | 시작 | 무엇 |
|---|---:|---|
| 같은 색·다른 형태 | 6 | blue pad ← blue cube ×2, blue pad ← blue cone ×2, red pad ← red cube ×2. 관련 물체가 **어려운 자리**에 있고 지시한 pad는 화면 밖 |
| 같은 형태·다른 색 | 3 | blue pad ← red pad ×2, blue cube ← red cube. 같은 규칙 |
| 먼저 보이지만 다른 자리 | 4 | blue cube ← blue pad, blue cone ← blue pad(화면 어디든), blue pad ← blue cube, red pad ← blue pad(왼쪽) |
| 놓친 뒤 | 3 | blue pad를 따라가다 강제 회전으로 blue cube / blue cone / red pad가 화면 가운데에. 오른쪽으로 두 번, 왼쪽으로 한 번 |

- **"어려운 자리"** = 탐색이 도는 쪽(오른쪽 8–35°)이거나 정면 10° 이내·25m 이내. Gen-v3b와 Gen-v3c의 비행 36회에서, 관련 물체가 그런 자리에 있던 18회 중 8회가 다른 물체에서 끝났고 나머지 18회에서는 0회였다. Pilot의 16개 중 9개가 그 자리다.
- **거리:** 12–20m 2개, 20–32m 6개, 32–44m 8개. Pad를 지시한 13개는 착륙, 나머지 3개는 접근이다.
- **모델마다 32회.** Gen-v3c checkpoint가 먼저 비행해 baseline이 된다. Teacher는 한 번 비행해 16개가 모두 끝낼 수 있는 시작인지 확인한다.

### 통과 기준 (결과를 보기 전에 고정)

[configs/visual_search.json](../configs/visual_search.json)의 `grounding_architecture.pilot.pass`. 한 줄이라도 어기면 FAIL이다. "맞음"은 비행이 지시한 물체에서 끝난 것이다.

| 줄 | 기준 |
|---|---|
| 다른 물체에서 끝난 비행 | 32회 중 2회 이하 |
| Baseline 대비 | 2회 이상 적고, baseline의 절반 이하 (한 비행 차이는 반복 편차다) |
| 두 번 다 틀린 시작 | 1개 이하 |
| 충돌 | baseline보다 많지 않음 |
| 문장이 시킨 대로 끝남 | baseline보다 적지 않음 |
| Approach 비행의 touchdown | 0 |
| 성공 | baseline보다 적지 않음 |
| 착륙 완료율 | 지시한 pad로 간 비행 중 system landing 0.9 이상 |
| 지시한 물체가 화면에 들어온 비행 | baseline보다 1회 넘게 적지 않음 |
| 실행 오류 | 0 |
| 회귀 | 위 줄을 모두 지킨 모델만 이전 set의 33개 시작을 비행. Gen-v3c(28/33)보다 2회 넘게 적지 않고 충돌 0 |

- **전체 성공만 오르고 다른 물체로 간 횟수가 그대로면 FAIL이다.**
- **통과한 구조에는 대조 실험을 붙인다.** LoRA와 head도 모듈과 함께 학습하므로, 좋아진 것이 2,000 update를 더 돈 덕일 수 있다. 통과한 구조가 나오면 모듈 없이 같은 일정으로 학습한 것(control)을 같은 32회에 비행하고, control이 통과하지 못할 때만 구조의 효과로 인정한다.
- **둘 다 비행한 경우의 선택 순서:** pilot 통과 → 다른 물체로 간 횟수 → 두 번 다 맞은 시작 수 → 성공 수 → 회귀 → 비용.

## 시각 경로 audit

```
Front + Down (224×448 mosaic → 224×224)
  → vision encoder: DINOv2 ViT-L/14 (1024) ⊕ SigLIP SO400M/14 (1152), 뒤에서 두 번째 층의 patch feature
  → 256 patch × 2176                                  ← 두 모듈이 들어가는 자리
  → projector (2176 → 8704 → 4096 → 4096, AeroVLA의 것)
  → 256 visual token × 4096
  → [BOS] + visual token + 문장 token + 빈 action token 12개 → Llama-2 7B (NF4) + AeroVLA LoRA(고정) + 보정 LoRA r16
  → action token의 마지막 hidden state → L1 head → 4 step × (전진, 하강, yaw)
```

- **문장은 지금까지 LLM의 입력 token으로만 들어갔다.** Vision encoder와 projector는 문장을 모른 채 화면을 256개 token으로 만든다. 문장과 화면이 만나는 곳은 LLM의 self-attention뿐이다.
- **넣을 수 있는 자리는 둘이다:** projector 직전의 patch feature(2176), projector 직후의 visual token(4096). 둘 다 vision encoder를 학습하지 않아도 되고, 추가 비용도 같다.
- **Projector 직전을 골랐다.** 계획이 먼저 든 자리다. 거기서는 feature가 아직 순수하게 시각적이라(DINOv2와 SigLIP의 채널), 문장이 "어떤 채널을 키우고 줄일지"를 projector가 LLM의 공간으로 섞기 전에 정할 수 있다. Projector는 고정한 채 gradient만 지나간다.
- **한 자리에만 넣었다.** 두 실험이 같은 자리를 쓴다.

## Baseline: Gen-v3c가 같은 32회를 먼저 비행

Teacher는 16/16이다. Gen-v3c checkpoint(`generalization_v3c`)의 두 번 비행:

| | Gen-v3c |
|---|---:|
| 성공 | 28/32 |
| 다른 물체에서 끝남 | 4 |
| 충돌 | 2 |
| 두 번 다 맞은 시작 / 한 번 / 한 번도 | 14 / 0 / 2 |
| 같은 색·다른 형태 | 18/22 |
| 같은 형태·다른 색 | 10/10 |
| 먼저 보이는 관련 물체를 거절 | 22/26 |
| 놓친 뒤 되찾음 | 6/6 |

- **실패는 두 시작에 몰려 있고, 둘 다 `red pad ← red cube`다.** 두 번 비행 모두 red cube로 갔다(한 시작은 두 번 다 충돌, 다른 시작은 두 번 다 그 앞에서 정지). 반복해도 같은 결과였다.
- **blue pad가 목표인 "어려운 자리" 시작 6개는 12/12였다.** Gen-v3c 검증에서 본 blue cone·red pad 쪽 실패는 이 set에서는 나오지 않았다.
- **그래서 이 pilot은 사실상 "red pad와 red cube를 가르는가"를 묻는다.** 통과하려면 다른 물체 2회 이하이고 두 번 다 틀린 시작이 1개 이하여야 하므로, 두 시작 중 적어도 하나를 두 번 다 맞혀야 한다.

## Experiment 1 — Visual feature FiLM

### 구조

```
문장 ──tokenizer──> token id ──LLM의 embedding table(고정)──> token embedding의 평균 (4096)
                                                                  │
                                      LayerNorm → Linear(4096→512) → GELU → Linear(512→2×2176)
                                                                  │
                                                         γ = 1 + Δγ,   β = Δβ      (채널마다 하나씩)
                                                                  │
vision encoder(고정) ──> patch feature (256 × 2176) ──> γ ⊙ x + β ──> projector(고정) ──> LLM
```

| | |
|---|---|
| 넣은 자리 | Projector 직전의 patch feature. 한 자리에만 |
| 문장 표현 | 문장만(prompt의 틀 없이) token으로 나눠 LLM 자신의 embedding table로 embed한 뒤 평균. 새 text encoder를 넣지 않았다 |
| 초기화 | 마지막 층이 0. 학습 전에는 γ=1, β=0이라 모델이 Gen-v3c와 정확히 같다(저장된 예측과의 차이 0.0으로 확인) |
| 더한 parameter | 4,338,432 |
| 학습한 것 | FiLM 4.34M + 보정 LoRA 39.98M + action head 14.72M = 59.03M |
| 학습하지 않은 것 | Vision encoder, projector, LLM(NF4), AeroVLA LoRA |
| 모듈이 받는 것 | Patch feature와 문장. 목표의 좌표·거리·방위·가시성은 들어가지 않는다 |

- **FiLM은 모든 patch를 같은 방식으로 바꾼다.** γ와 β가 채널마다 하나씩이고 patch 위치와 무관하다.
- **모듈 안의 계산은 float32로 한다.** 변화가 작게 시작하므로 half precision에서 반올림돼 사라지지 않게 했다.

### 학습

[configs/visual_search.json](../configs/visual_search.json)의 `grounding_architecture.training`. Gen-v3c 보정과 같은 일정이다.

| | 값 |
|---|---|
| 시작 | `generalization_v3c` |
| Update | 2,000 (frame 4개씩), 87분 |
| Learning rate | LoRA·head 5e-5 → 5e-6, FiLM 5e-4 → 5e-5 (새 모듈은 10배. 학습 전에 정했다) |
| Data | Gen-v3c와 같음. 절반은 hard-negative episode에서 |
| 남긴 checkpoint | update 1,750 (같은 규칙: 검증 L1이 가장 낮은 것의 2% 안에서 행동 점수) |
| 검증 L1 (새것·옛것 반반 600 frame) | 0.0601 → 0.0609 |
| 이전 검증 frame 400개의 L1 | 0.0581 → 0.0588 |
| 학습 중 최대 VRAM | 9.88 GiB (Gen-v3c 9.82) |

- **학습 초반에 한 번 크게 흔들렸다.** Update 500에서 검증 L1이 0.166까지 올랐다가 750에서 0.075로 돌아왔다. 새 모듈의 학습률이 높았던 탓으로 보인다. 남긴 checkpoint는 그 뒤의 것이다.
- **Offline probe:** 같은 색 물체가 화면 가운데 있고 지시한 것은 없는 frame에서 "계속 돈다"를 낸 비율(학습 frame) 79% → 94%. 보이는 물체를 지시하는 문장으로 바꿨을 때 전진으로 바뀐 비율은 35% → 33%로 그대로다.

### Pilot 결과

| | Gen-v3c | FiLM |
|---|---:|---:|
| 성공 | 28/32 | **32/32** |
| 다른 물체에서 끝남 | 4 | **0** |
| 두 번 다 맞은 시작 / 한 번 / 한 번도 | 14 / 0 / 2 | **16 / 0 / 0** |
| 같은 색·다른 형태 | 18/22 | 22/22 |
| 같은 형태·다른 색 | 10/10 | 10/10 |
| 먼저 보이는 관련 물체를 거절 | 22/26 | 26/26 |
| 놓친 뒤 되찾음 | 6/6 | 6/6 |
| 충돌 | 2 | 0 |
| 문장이 시킨 대로 끝남 | 28/32 | 32/32 |
| 착륙 완료 / 지시한 pad로 간 비행 | 22/22 | 26/26 |
| 회귀 (이전 set 33개 시작) | 28 | 26 |

| 통과 기준 | 값 | 필요 | |
|---|---:|---|---|
| 다른 물체에서 끝난 비행 | 0 | 2 이하 | 통과 |
| Baseline 대비 (2 이상 적고 절반 이하) | 0 | 2 이하 | 통과 |
| 두 번 다 틀린 시작 | 0 | 1 이하 | 통과 |
| 충돌 | 0 | 2 이하 | 통과 |
| 시킨 대로 끝남 | 32 | 28 이상 | 통과 |
| Approach의 touchdown | 0 | 0 | 통과 |
| 성공 | 32 | 28 이상 | 통과 |
| 착륙 완료율 | 1.0 | 0.9 이상 | 통과 |
| 지시한 물체가 화면에 들어옴 | 32 | 27 이상 | 통과 |
| 회귀: 성공 | 26 | 26 이상 | 통과 (한계에 닿음) |
| 회귀: 충돌 | 0 | 0 | 통과 |

- **Pilot: PASS.** 두 `red pad ← red cube` 시작을 두 번 다 맞혔고, 다른 14개 시작은 그대로 두 번 다 맞혔다.
- **회귀는 허용 한계에 정확히 닿았다.** 26/33으로 Gen-v3c보다 둘 적다. 잃은 둘은 Yard의 blue cone이 정면에 보이는 시작(계속 돌며 지나침)과 장거리 orange ball 시작(시간 초과)이다. Blocks의 blue cone 46m 시작은 여전히 세 문장 모두 실패한다. "보이는 파란 물체 앞에서 망설임"이 한 장면 더 번졌다.

### FiLM이 실제로 배운 것

학습된 모듈에 문장 15개를 넣어 γ와 β를 직접 봤다(`scripts/inspect_film.py`, base model은 돌리지 않는다).

| | |
|---|---|
| 조절의 크기 | \|γ−1\| 평균 0.042(최대 0.67), \|β\| 평균 0.036(최대 0.56). DINOv2 쪽과 SigLIP 쪽이 비슷하다 |
| 문장에 따라 달라지는 부분 | 조절 전체 에너지의 71% |
| "Find the **blue** landing pad and land on it." 대 "Find the **red** landing pad and land on it." | 방향이 같고(cosine 1.00) 크기 차이는 1.7% |
| "Find the blue landing pad." 대 "Find the blue cube." | cosine 0.99, 차이 15% |
| "Find the red landing pad." 대 "Find the red cube." | cosine 0.56, 차이 171% |
| 조절 벡터의 크기 | pad를 지시하는 문장 0.78–0.79, "Find the blue cube." 0.76, "Find the red cube." 7.6, "Approach the red cube." 11.0 |

- **모듈은 문장에 따라 다른 조절을 낸다.** 고정된 변환이 아니다.
- **그런데 pad를 지시하는 문장들에는 색과 무관하게 거의 같은 조절을 낸다.** blue pad 문장과 red pad 문장의 γ·β는 사실상 같다. Pilot에서 고쳐진 두 시작은 "red landing pad"를 지시한 것인데, 그 문장에 대한 조절이 blue pad 문장의 것과 구별되지 않는다.
- **즉 이 FiLM이 "빨강"과 "파랑"을 시각 feature에 실어 준 것은 아니다.** red pad와 red cube를 가른 것이 FiLM의 조건화인지, 함께 2,000 update를 더 학습한 LoRA와 head인지는 이 표만으로는 알 수 없다. 그것을 가리는 것이 아래의 대조 실험이다.

## 대조 실험: 모듈 없이 같은 2,000 update

FiLM 실험에서는 LoRA와 head도 함께 2,000 update를 더 학습했다. 그래서 통과한 것이 모듈 덕인지 추가 학습 덕인지 가리려고, **모듈만 뺀 같은 일정**(같은 시작 checkpoint, 같은 data, 같은 뽑는 순서, 같은 학습률)을 돌려 같은 32회를 비행했다. 이 대조와 그 판정 규칙은 학습 전에 적어 둔 것이다(`e1ec665`).

| | Gen-v3c | FiLM | Control (모듈 없음, +2,000) |
|---|---:|---:|---:|
| 성공 | 28/32 | 32/32 | 30/32 |
| 다른 물체에서 끝남 | 4 | 0 | 2 |
| 두 번 다 맞은 시작 / 한 번 / 한 번도 | 14 / 0 / 2 | 16 / 0 / 0 | 14 / 2 / 0 |
| `red pad ← red cube` 두 시작 (4회) | 0/4 | 4/4 | 4/4 |
| 충돌 | 2 | 0 | 0 |
| Offline: 같은 색 물체가 가운데 있을 때 "계속 돈다" | 79% | 94% | 97% |
| 이전 검증 frame의 L1 | 0.0581 | 0.0588 | 0.0582 |
| 회귀 (이전 set 33개 시작) | 28 | 26 | 27 |
| 남긴 checkpoint | — | update 1,750 | update 1,999 |

- **Control도 pilot을 통과했다.** 다른 물체 2회(기준 2 이하, baseline의 절반), 두 번 다 틀린 시작 0개, 충돌 0, 회귀 27/33(기준 26 이상). 회귀 시작까지 비행해 판정을 끝냈다.
- **Control도 같은 두 시작을 고쳤다.** `red pad ← red cube`는 모듈이 없어도 네 번 모두 맞았다. Offline에서도 같은 색 frame의 "계속 돈다" 비율이 FiLM과 같거나 조금 높다.
- **Control이 틀린 두 비행은 다른 시작에서, 한 번씩이다:** `blue pad ← blue cube`에서 blue cube 앞 정지 1회, `blue cone ← blue pad`에서 blue pad로 감 1회. 같은 시작의 다른 한 번은 맞았다.

### 판정: 개선을 FiLM의 효과로 인정하지 않는다

미리 정한 규칙은 "통과한 구조의 control이 pilot을 통과하지 못할 때만 구조의 효과로 인정한다"였다. Control이 통과했으므로:

- **Pilot에서 고쳐진 것은 같은 data로 2,000 update를 더 학습한 결과다.** FiLM 없이도 일어났다.
- **FiLM이 더한 것은 32회 중 2회의 차이다(0 대 2).** 서로 다른 두 시작에서 한 번씩이고, 같은 시작을 반복했을 때 본 편차와 구별되지 않는다. FiLM 쪽이 낫다는 방향이지만 근거로 삼을 크기가 아니다.
- **모듈을 들여다본 결과와도 맞는다.** 학습된 FiLM은 blue pad 문장과 red pad 문장에 사실상 같은 조절을 낸다.
- **Gen-v3c에서 내린 "데이터 보정으로는 부족하다"는 판정은 일부 고쳐 읽어야 한다.** 그때는 검증 L1 규칙이 update 1,250의 checkpoint를 남겼다. 같은 data를 더 오래 학습하면(합계 3,250 update) 이 pilot의 실패는 사라진다. 검증 L1은 그 차이를 보여 주지 못했다(0.060 안팎에서 그대로).

### 그래도 FiLM checkpoint를 다음 단계로 보낸 이유

계획은 pilot의 승자를 gate 뒤로 보내라고 한다. 미리 정한 선택 순서(pilot 통과 → 다른 물체로 간 횟수가 적은 쪽 → 두 번 다 맞은 시작이 많은 쪽)로는, control까지 놓고 봐도 FiLM checkpoint가 앞선다(0회, 16개).

- **구조의 증거로서가 아니라, pilot에서 가장 좋은 checkpoint로서 보낸다.** 이 결정은 검증 비행 전에 config에 적고 commit했다(`grounding_architecture.outcome`, `edb34f1`).
- **이 한 가지는 미리 적은 문장과 다르다.** 검증은 "control이 통과하지 못한 구조"에만 한다고 적어 두었었다. 계획의 흐름(승자 → 새 검증 → 통과 시 test)을 따르기 위해 그 문장을 이 목적에 한해 접었고, gate의 숫자는 그대로다.
- **Experiment 2(cross-attention)는 실행하지 않았다.** FiLM이 pilot을 통과했기 때문이다. Control의 결과로 보면, 이 pilot의 수준에서는 구조를 바꿀 이유가 드러나지 않았다.

## 새 검증 set (FiLM checkpoint)

[configs/grounding_architecture_validation.json](../configs/grounding_architecture_validation.json). Seed 24000–24268, 36개. Gen-v3c 검증과 같은 구성(band마다 12개, 착륙 27·접근 9, 목표 선택을 묻는 시작 17개)이고 시작은 모두 새것이다. Gen-v3c의 검증 36개를 다시 쓰지 않았다. 어떤 모델도 비행하기 전에 commit했다(`26058f5`). Teacher 36/36.

Gate는 canonical gate의 숫자 그대로이고, **시작마다 첫 비행**으로 판정한다. 판정: `canonical_evaluator_v2`.

| Smoke (10개) | 값 | 기준 | |
|---|---:|---|---|
| 성공 | 10 | 7 이상 | 통과 |
| 착륙 성공(system) | 7 | 1 이상 | 통과 |
| 32–44m에서의 성공 | 3 | 1 이상 | 통과 |
| 충돌 / 실행 오류 | 0 / 0 | 1 이하 / 0 | 통과 |

| Representative (36개) | 값 | 기준 | |
|---|---:|---|---|
| 전체 성공률 | 34/36 (0.944) | 0.80 이상 | 통과 |
| 착륙 성공률(system) | 25/27 (0.926) | 0.75 이상 | 통과 |
| 목표가 보인 뒤 접근을 시작함 | 34/34 (1.000) | 0.90 이상 | 통과 |
| 문장이 시킨 대로 끝남 | 35/36 (0.972) | 0.90 이상 | 통과 |
| 다른 물체에서 끝난 비행 | 2 | 2 이하 | 통과 (한계에 닿음) |
| 충돌 | 0 | 1 이하 | 통과 |
| 실행 오류 | 0 | 0 | 통과 |

- **Gate: PASS.** Canonical gate를 처음으로 통과했다. 이전 두 번(30/36, 33/36)은 둘 다 "다른 물체 3회"에서 걸렸고, 이번에는 2회다.
- **한 비행 차이의 통과다.** 다른 물체로 간 비행이 하나만 더 있었으면 같은 줄에서 걸렸다.

| Band | 전체 | 착륙 | 접근 |
|---|---:|---:|---:|
| near (12–20m) | 11/12 | 8/9 | 3/3 |
| mid (20–32m) | 12/12 | 9/9 | 3/3 |
| far (32–44m) | 11/12 | 8/9 | 3/3 |

| 목표 선택 | 비행 | 성공 | 다른 물체에서 끝남 |
|---|---:|---:|---:|
| 관련 물체가 첫 화면에 있고 지시한 것은 화면 밖 | 17 | 15 | 2 |
| … 그중 같은 색 | 13 | 12 | 1 |
| … 그중 같은 형태 | 4 | 3 | 1 |
| 지시한 물체가 첫 화면 안 | 15 | 15 | 0 |
| 그 밖 (관련 물체 없는 탐색 등) | 4 | 4 | 0 |

- **실패 2회는 둘 다 목표 선택이다.** `av-24172-…-near_pair-swapped`: blue cone이 정면 15m에 있는 시작에서 cone 앞에 멈췄다. `av-24267-red_pad-far_red_past`: red pad를 지시했는데 먼저 보이는 blue pad에 내려앉았다(같은 형태).
- **그 조건 밖은 19/19다.** 착륙은 지시한 pad로 간 25회가 전부 안정 착륙·latch·disarm까지 갔다. Finalizer가 구한 착륙 2회, approach 9회에서 켜진 적 없음. Touchdown 오차 중앙값 1.92m, 최대 3.86m.

### 목표 선택 시작의 두 번째 비행 (판정에 쓰지 않음)

목표 선택을 묻는 17개 시작을 한 번 더 비행했다. 이것은 gate가 아니고, 미리 그렇게 적어 두었다.

| 17개 시작 | 첫 비행 | 두 번째 비행 |
|---|---:|---:|
| 성공 | 15 | 13 |
| 다른 물체에서 끝남 | 2 | 3 |
| 두 번 다 성공 / 한 번 / 한 번도 | 13 / 2 / 2 | |

- **두 시작은 두 번 다 틀렸다.** 위의 두 실패가 그대로 반복됐다.
- **두 시작은 한 번씩 갈렸다.** `…-find-blue_cone`(blue cone을 지시, blue pad가 정면 15m)은 두 번째 비행에서 바로 앞의 blue pad 곁에 멈췄다. `…-far_past_color`는 두 번째 비행에서 blue pad에 안정되게 내려앉고 latch까지 됐는데 disarm 호출이 확인되지 않아 system landing이 아닌 것으로 기록됐다. 이 종류의 실패는 처음이다.
- **읽을 때:** 두 번째 비행으로 판정했다면 다른 물체 3회로 걸렸을 것이다. Gate의 통과는 미리 정한 대로 첫 비행의 것이지만, 이 조건에서의 오류율은 여전히 열 번에 한두 번이다(34회 중 5회).

## 동결

두 gate를 통과한 뒤 `outputs/generalization/grounding_film_frozen.json`에 지문을 남겼다(`85d5f3d`): checkpoint(FiLM 모듈 포함), 착륙 규칙(finalizer·evaluator 포함), config, 지도, test 시작 파일, 비행·판정 code. Test는 이 지문이 그대로일 때만 비행한다.

## Canonical test 48개 (한 번)

[configs/gen_v3_canonical_44m_test.json](../configs/gen_v3_canonical_44m_test.json). Canonical 기준을 정할 때 만들어 봉인해 둔 set이다(seed 9500번대, Field·Lot의 배치 g–j: 학습에도 검증에도 쓴 적 없는 배치). 지금까지 teacher만 비행했고, 어떤 모델도 비행한 적이 없다. 동결한 FiLM checkpoint로 **한 번** 비행했다. 미리 보거나 일부만 비행하지 않았다. 표: `outputs/generalization/grounding_architecture/test/`.

| | 결과 | 검증 gate의 기준으로 읽으면 |
|---|---:|---|
| Episode | 48 | |
| **전체 성공** | **47/48 (0.979)** | 0.80 이상 |
| 지시한 물체에서 끝남 | 47/48 | |
| 다른 물체에서 끝남 | 1 | 2 이하 |
| 목표가 보인 뒤 접근을 시작함 | 46/47 (0.979) | 0.90 이상 |
| 문장이 시킨 대로 끝남 (approach 대 land) | 48/48 | 0.90 이상 |
| 충돌 | 0 | 1 이하 |

| 착륙 36회 | 결과 |
|---|---:|
| 지시한 pad에 touchdown | 36/36 |
| Stable physical landing | 36/36 |
| System landing success | 36/36 |
| Strict policy zero-action (보조) | 36/36 |

| 시작 거리 | 전체 | 착륙 | 접근 |
|---|---:|---:|---:|
| 12–20m | 15/16 | 12/12 | 3/4 |
| 20–32m | 16/16 | 12/12 | 4/4 |
| 32–44m | 16/16 | 12/12 | 4/4 |

| 목표 선택 | 비행 | 성공 | 다른 물체에서 끝남 |
|---|---:|---:|---:|
| 색이 같고 형태가 다른 물체가 곁이나 첫 화면에 | 22 | 21 | 1 |
| 형태가 같고 색이 다른 물체가 곁이나 첫 화면에 | 26 | 26 | 0 |
| 두 물체의 자리를 바꿈 | 6 | 6 | 0 |
| 지시한 물체가 첫 화면 밖 | 27 | 26 | 1 |
| 관련 물체가 먼저 보이고 지시한 것은 화면 밖 | 17 | 16 | 1 |
| 곁의 물체를 지시 (query) | 6 | 5 | 1 |

- **실패는 하나다.** `ct-9507-…-near_pair2-find-blue_cube`: "Find the blue cube."인데 먼저 보이는 blue pad 곁에서 멈췄다. 이번에도 목표 선택이고, 같은 색이다.
- **Finalizer:** 착륙 36회 모두 latch하고 disarm했다. Finalizer가 구한 착륙은 0회다(36회 모두 정책이 스스로 멈췄다). Approach 12회에서 켜진 적은 없다.
- **Touchdown:** 오차 중앙값 2.76m, 최대 4.69m(구역 6.6m). 직전 수직 속도 중앙값 0.60m/s, 최대 0.65m/s(기준 0.75m/s).
- **이 48개는 다시 개발에 쓰지 않는다.** 이 결과가 canonical clean baseline의 추정치다.

## Examples

동결한 checkpoint를 이미 쓴 검증 시작에서 `-Record`로 다시 비행한 것이다. Test set의 비행이 아니다.

| | 무엇 |
|---|---|
| [related_first_landed.gif](../outputs/examples/grounding_architecture/related_first_landed.gif) | `av-24213-…-mid_pair-swapped`. blue cube가 정면에 먼저 보이고 blue pad는 화면 밖. cube로 가지 않고 계속 돌아 pad를 찾아 내려앉는다 |
| [failure_other_pad.gif](../outputs/examples/grounding_architecture/failure_other_pad.gif) | `av-24267-red_pad-far_red_past`. red pad를 지시했는데 먼저 보이는 blue pad에 내려앉는다. 검증에서 두 번 다 틀린 시작이고, 다시 비행해도 같았다 |

## 비교

Pilot의 같은 16개 시작 × 2회, 그리고 그 뒤의 단계.

| | Gen-v3c | FiLM | Control (모듈 없음, +2,000) | Cross-Attention |
|---|---:|---:|---:|---:|
| Pilot 성공 | 28/32 | 32/32 | 30/32 | N/A |
| 다른 물체에서 끝남 | 4 | 0 | 2 | N/A |
| 같은 색·다른 형태 | 18/22 | 22/22 | 20/22 | N/A |
| 같은 형태·다른 색 | 10/10 | 10/10 | 10/10 | N/A |
| 먼저 보이는 관련 물체를 거절 | 22/26 | 26/26 | 24/26 | N/A |
| 놓친 뒤 되찾음 | 6/6 | 6/6 | 6/6 | N/A |
| 두 번 다 맞은 시작 (16개 중) | 14 | 16 | 14 | N/A |
| 한 번 / 한 번도 | 0 / 2 | 0 / 0 | 2 / 0 | N/A |
| 충돌 | 2 | 0 | 0 | N/A |
| 시킨 대로 끝남 | 28/32 | 32/32 | 30/32 | N/A |
| 착륙 완료 / 지시한 pad로 감 | 22/22 | 26/26 | 25/25 | N/A |
| 회귀 (이전 set 33개) | 28 | 26 | 27 | N/A |
| 판단당 추론 시간 (비행 중) | 396ms | 401ms | 406ms | N/A |
| 추론 시간 (같은 frame 59개) | 216ms | 225ms | 216ms | N/A |
| 추론 최대 VRAM | 6.94 GiB | 6.94 GiB | 6.94 GiB | N/A |
| 학습 최대 VRAM | 9.82 GiB | 9.88 GiB | 9.82 GiB | N/A |
| 더한 parameter | 0 | 4.34M | 0 | 2.74M (학습 안 함) |
| 학습한 parameter | 54.69M | 59.03M | 54.69M | N/A |
| 새 검증 gate | 33/36, FAIL (이전 set) | 34/36, PASS | 비행 안 함 | N/A |
| Canonical test 48 | 비행 안 함 | 47/48 | 비행 안 함 | N/A |

- **Cross-attention은 실행하지 않았다.** 모듈은 구현하고 test했지만(학습 전에는 예측을 바꾸지 않음을 확인), 학습도 비행도 하지 않았다.
- **FiLM의 비용은 작다.** Parameter 4.34M, 추론 시간 +4%, VRAM 차이 없음.

## 결론

### 질문의 답

> 지금의 wrong-target 실패는 문장이 시각 표현을 충분히 강하게 조건화하지 못해서 생기는가?

**이 실험으로는 그렇다고 할 수 없다.**

- **FiLM을 넣은 checkpoint는 pilot, 새 검증 gate, canonical test를 모두 통과했다.**
- **그러나 같은 일정을 FiLM 없이 돌린 control도 pilot을 통과했고, 같은 실패를 고쳤다.** Pilot에서의 개선은 추가 학습으로 설명된다.
- **FiLM만의 몫은 32회 중 2회(0 대 2)다.** 방향은 FiLM 쪽이지만 반복 편차와 구별되지 않는다. 학습된 FiLM은 pad 문장에 색과 무관하게 같은 조절을 낸다.
- **즉 계획의 세 결론(A: FiLM이 풀었다, B: cross-attention이 필요했다, C: 둘 다 부족하다) 중 어느 것도 그대로는 맞지 않는다.** 일어난 일은 "같은 data로 더 오래 학습하자 pilot의 실패가 사라졌고, 그 위에 FiLM을 얹은 checkpoint가 가장 좋아서 gate와 test까지 갔다"이다. 결과만 보면 A에 가장 가깝지만, A의 인과("FiLM이 병목을 풀었다")는 확인되지 않았다.

### Gen-v3c의 판정에 대해

Gen-v3c에서 "데이터 보정으로는 부족하다"고 한 것은 update 1,250에서 멈춘 checkpoint에 대한 판정이었다. 같은 data를 2,000 update 더 학습한 control은 그 판정의 근거가 된 종류의 시작에서 4회 중 4회(red pad ← red cube)를 맞혔다. Checkpoint를 고른 규칙(검증 L1)이 이 행동의 차이를 보지 못했다는 것이 이번에 드러난 점이다.

### 남은 것

- **목표 선택은 여전히 가장 약한 곳이다.** 검증에서 관련 물체가 먼저 보이는 시작은 34회 중 5회가 다른 물체에서 끝났고(두 번 비행 합계), test의 유일한 실패도 그것이다. 사라진 것이 아니라 gate 아래로 내려간 것이다.
- **회귀는 조금 깎였다.** 이전 set의 33개 시작에서 28 → 26(control은 27). "보이는 파란 물체 앞에서 망설임"이 Yard의 blue cone 시작으로 번졌다.
- **구조 쪽 질문은 열려 있다.** FiLM의 효과를 가리려면 control과 FiLM을 같은 시작에서 더 많이(예: 시작당 4–6회) 비행하거나, 학습된 FiLM checkpoint에서 모듈만 끄고 비교해야 한다. 이번에는 하지 않았다.

### Gaussian Blur로 넘어갈 준비: YES

- **새 검증 gate를 통과했고, canonical test는 47/48이다.** 착륙 36/36, 충돌 0, 문장에 따른 끝맺음 48/48.
- **Clean 기준으로 쓸 숫자가 생겼다.** Blur 실험은 이 checkpoint(`grounding_film`, 동결)와 이 48개에서의 결과를 기준으로 삼는다.
- **Blur 실험에서 따로 볼 것:** clean에서도 목표 선택 오류가 0이 아니다. Blur가 더한 오류와 구별하려면 관련 물체가 먼저 보이는 시작을 따로 세고, 가능하면 반복 비행한다.
- **Blur는 이번에 실행하지 않았다.**

### 44m 초과

이 실험과 무관하다. Canonical 범위는 44m 이하 그대로이고, 회귀에 섞인 장거리 시작의 숫자로 범위를 넓히지 않는다. [Long-range same-color grounding](long_range_same_color_grounding.md)에 따로 둔다.

## 한계

1. **Test는 새 장면이 아니다.** Canonical test 48개는 학습 장면(Field·Lot) 안의 새 배치다. 새 held-out 장면은 Depot의 156개이고, 봉인돼 있다.
2. **검증 gate의 통과는 한 비행 차이였다.** 다른 물체 2회(기준 2 이하)였고, 목표 선택 시작의 두 번째 비행에서는 3회였다. Test의 47/48은 그보다 좋지만 한 번의 비행이다.
3. **구조의 효과는 가려지지 않았다.** Control을 gate와 test에 올리지 않았으므로, FiLM 없는 checkpoint가 같은 결과를 냈을지는 모른다.
4. **Pilot은 결국 한 종류의 실패(red pad ← red cube)만 갈랐다.** Baseline의 실패가 그 두 시작에 몰려 있었기 때문이다.
5. **FiLM의 학습률은 한 값만 썼다.** 학습 초반의 흔들림(update 500)은 그 값이 높았다는 신호다.
6. **문장 표현은 token embedding의 평균이다.** 어순을 보지 않는다. 지금의 문장들에서는 차이가 없지만 더 복잡한 지시에서는 부족할 수 있다.
7. **Checkpoint를 고르는 규칙이 이 행동을 보지 못한다.** 검증 L1은 0.060 안팎에서 그대로인데 비행에서의 목표 선택은 달라졌다. 다음에는 선택 규칙에 목표 선택 frame의 행동을 넣는 것을 권한다.

## 재현

```bash
# 학습 전 확인: 학습 안 된 모듈은 예측을 바꾸지 않는다 (WSL)
python scripts/check_grounding_identity.py --checkpoint outputs/aerovla_oft/checkpoints/generalization_v3c --dataset $DATA/generalization_v3c

# Experiment 1 (FiLM), 그리고 모듈만 뺀 control. Data와 일정은 Gen-v3c 보정과 같다
python scripts/train_aerovla_oft.py --dataset $DATA/generalization_v3c --output outputs/aerovla_oft/checkpoints/grounding_film --init outputs/aerovla_oft/checkpoints/generalization_v3c \
  --steps 2000 --lr 5e-5 --eval-every 250 --eval-samples 600 --val-file val_mix --strategies right --boost small_visible=2 selection=3 --share generalization_v3c_added=0.5 --select-band 0.02 --grounding film
python scripts/train_aerovla_oft.py --cost outputs/aerovla_oft/checkpoints/grounding_film --dataset $DATA/generalization_v3c --val-file val_mix --eval-samples 60
python scripts/inspect_film.py --checkpoint outputs/aerovla_oft/checkpoints/grounding_film
```

```powershell
# Pilot: 16개 시작을 모델마다 두 번, 그리고 판정
python scripts/grounding_arch.py plan pilot
scripts/run_visual_search.ps1 -Policy oft -Checkpoint outputs/aerovla_oft/checkpoints/grounding_film -Plan configs/grounding_architecture_pilot.json -Set episodes -Output outputs/visual_search/arch_film_run1
python scripts/grounding_arch.py judge film --runs outputs/visual_search/arch_film_run1 outputs/visual_search/arch_film_run2 --baseline outputs/visual_search/arch_gen_v3c_run1 outputs/visual_search/arch_gen_v3c_run2 --teacher outputs/visual_search/arch_teacher_pilot --regression outputs/visual_search/arch_film_regression/regression.json
python scripts/grounding_arch.py compare "Gen-v3c=...run1,...run2" "FiLM=...run1,...run2" --teacher outputs/visual_search/arch_teacher_pilot --output outputs/generalization/grounding_architecture

# 새 검증 set → gate → 동결 → test (한 번)
python scripts/grounding_arch.py plan validation
python scripts/grounding_arch.py gate representative --run outputs/visual_search/arch_film_validation --teacher outputs/visual_search/arch_teacher_validation
python scripts/freeze_baseline.py verify grounding_film
python scripts/gen_v3c.py gate test --run outputs/visual_search/arch_film_test --teacher outputs/visual_search/canonical_teacher_test
```

Checkpoint, dataset, 비행 기록은 Git에 없다. 표와 판정 기록은 `outputs/generalization/grounding_architecture/`에 있다.

## Tests

`tests/test_grounding_arch.py`:

- Pilot set이 생성기로 그대로 재현되고, 네 묶음 16개이고, 관련 물체가 묶음이 말하는 자리에 있고, 이전의 모든 시작에서 6m 이상 떨어져 있고, 녹화한 적 없는 배치에 있다.
- 새 검증 set이 재현되고, 구성이 Gen-v3c 검증과 같고, seed와 시작이 새것이다.
- Pilot의 줄: 뚜렷한 감소는 통과, 다른 물체 3회는 실패, baseline이 이미 낮으면 한 비행 차이로는 통과하지 못함, 성공만 오르고 다른 물체가 그대로면 실패, 다른 것이 나빠지면 실패, 회귀를 비행한 뒤에는 그 줄도 판정에 들어감.
- 학습 일정이 Gen-v3c 보정과 같고, 새 모듈의 학습률만 10배다.
- 두 모듈은 학습 전에는 입력을 그대로 내보내고, 첫 gradient가 마지막 층에 닿는다.
- FiLM은 모든 patch를 같은 방식으로 바꾸고, 문장으로만 달라지고, padding을 읽지 않는다.
- Cross-attention은 단어가 patch를 읽고 읽은 자리에 되돌려 쓴다. 어떤 단어도 보지 않은 patch는 그대로다.
- 두 모듈 모두 patch와 문장 말고는 아무것도 받지 않는다(목표의 좌표·거리·방위·가시성 없음). Vision encoder는 학습하지 않는다.




