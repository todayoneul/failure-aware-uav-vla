# AeroVLA-OFT — Language–vision grounding architecture 실험

2026-10-10 실행. 브랜치 `exp/aerovla-oft-grounding-architecture`(`exp/aerovla-oft-gen-v3c-hard-negative`에서 분기). [Gen-v3c](gen_v3c_hard_negative_grounding.md)의 checkpoint, 결과, 문서는 그대로 있다. Canonical test set 48개와 Depot의 156개는 계속 봉인돼 있고, 이번에도 비행하지 않았다.

{{SUMMARY}}

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

{{CONTROL}}

