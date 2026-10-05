# AeroVLA INT4/NF4 실제 검증

> **후속 결과:** 이 문서의 최초 relay 통합 실패 뒤, direct WSL2 경로에서 live single-step과 10/10 반복 실행에 성공했다. [최종 통합 검증](final_closed_loop_validation.md). 아래 단독 모델 실측과 최초 실패는 당시 기록으로 보존한다.

검증일: 2026-10-05 KST. **모델 단독 로딩과 11회 생성은 PASS.** Windows Project AirSim과 함께 로딩도 성공했으나, scene topic 초기화 timeout 때문에 동시 camera → inference → flight는 검증하지 못했다. 이번 문서는 이전 작은 모델 NF4 smoke test와 구별되는 실제 7B base + AeroVLA LoRA 측정이다.

## 다운로드 범위와 재현 정보

통신 STEP 1을 통과한 뒤 사용자에게 총 용량을 먼저 알리고 아래 파일만 다운로드했다. HF API의 pinned metadata와 실제 파일 크기를 대조하고 LFS SHA-256을 검증했다. 다른 모델, 학습 데이터, TravelUAV 전체 데이터/환경은 받지 않았다.

| 항목 | Revision | 실제 다운로드 |
|---|---|---:|
| [OpenVLA-7B](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f) | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` | 15,085,147,325 bytes / 16 files |
| [AeroVLA LoRA: aero_vla](https://huggingface.co/XuPeng23/AerialVLA/tree/196f2f3253b69df6e90ac10b6ae041c7b3a9569e/aero_vla) | `196f2f3253b69df6e90ac10b6ae041c7b3a9569e` | 462,655,797 bytes / 2 files |
| 합계 | 고정 revision, snapshot symlink로 중복 복사 방지 | **15,547,803,122 bytes / 14.480 GiB** |

Base의 세 safetensors shard와 config/tokenizer/processor/custom Python code, LoRA의 `adapter_config.json`/`adapter_model.safetensors`만 선택했다. 모델 저장 위치는 WSL `/home/gyuhan/uav-vla-smoke/models/hf`. 다운로드 사전 예산은 실행 환경 포함 30GB였으며 C: 여유는 다운로드 전 400,527,294,464 bytes, 정리 시 384,480,014,336 bytes였다. 이 차이는 전체 드라이브 측정으로 다른 쓰기도 포함할 수 있다. venv의 apparent size 7.1G는 uv cache hardlink를 포함하므로 독립 추가 디스크 사용량으로 합산하지 않는다.

Python 3.10.14 별도 venv: `/home/gyuhan/uav-vla-smoke/integration`. 기존 Gate 1 환경은 수정하지 않았다.

| Library | Version |
|---|---|
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| transformers / tokenizers | 4.42.4 / 0.19.1 |
| bitsandbytes / accelerate / PEFT | 0.48.2 / 0.32.1 / 0.11.1 |
| timm / scipy / numpy | 0.9.10 / 1.15.3 / 1.26.3 |
| Project AirSim / pynng / OpenCV | 1.0.2 / 0.9.0 / 4.11.0.86 |

초기 offline cache 재사용은 `filelock` registry metadata 부족으로 실패했다. 이후 **동일한 고정 버전**을 online index에서 설치했으며 dependency check는 통과했다. 임의 버전 변경이나 반복 downgrade는 하지 않았다. 전체 freeze와 설치 오류도 원시 로그에 보존했다.

## 별도 로더의 실제 동작

구현: [aerovla_int4_loader.py](../src/integration/aerovla_int4_loader.py). [공개 AeroVLA wrapper](https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/src/model_wrapper/aerovla_wrapper_ui.py)는 읽기만 했으며 복사본과 원본의 SHA-256을 보존했다.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["vision_backbone", "projector", "lm_head"],
)
# from_pretrained: device_map={"": 0}, torch_dtype=BF16,
# low_cpu_mem_usage=True, trust_remote_code=True,
# local_files_only=True, attn_implementation="eager"
```

이는 **언어 모델 NF4 + 나머지 혼합 정밀도** 구성이다. 모든 parameter를 4-bit로 만든 구성이 아니다. DINOv2/SigLIP 비전 backbone 및 projector는 실제 BF16, 전체 parameter dtype은 uint8/BF16/FP32였다. LoRA와 일부 유지 parameter에는 FP32가 존재한다.

| 충돌 후보 | 실제 확인 |
|---|---|
| `device_map` | 모든 parameter가 `cuda:0`; 224개 `bnb.nn.Linear4bit` 확인. CPU/disk offload 없음 |
| `.to(device)` | 모델 전체에 호출하지 않음. 입력 tensor만 CUDA/BF16으로 이동 |
| embedding resize | 양자화하지 않은 embedding/output head에서 **32064 → 32001**; LoRA 로딩 **전** 수행, 성공 |
| PEFT | rank64, alpha128, q/k/v/o/gate/up/down projection + `modules_to_save=[projector]`; NF4 base 위 로딩/생성 성공 |
| merge | adapter를 merge하지 않음 |
| custom code | pinned local OpenVLA code 사용. 별도 backbone weight 다운로드 없음 |
| attention | `eager` 지정. custom `_supports_sdpa`가 초기화 전 `language_model`을 참조하는 자동 선택 경로를 피함; FlashAttention 설치 없음 |

[Custom OpenVLA code](https://huggingface.co/openvla/openvla-7b/blob/47a0ec7fc4ec123775a391911046cf33cf9ed83f/modeling_prismatic.py)는 transformers 4.40.1을 기대한다는 warning을 출력했다. 이번 고정 4.42.4에서 실제 11회 생성은 통과했다. tuple `past_key_values` deprecation warning도 기록했다. 이를 이후 라이브러리 버전에서의 호환성 보장으로 일반화하지 않는다.

## 모델 단독 실측

Simulator를 종료한 상태에서 STEP 1에서 저장한 실제 Front/Down 프레임을 사용했다. 단독 반복은 동일 이미지·동일 synthetic target·동일 prompt의 greedy generation이며 장면 변화 또는 navigation 평가가 아니다. `inference_mode`, batch1, `max_new_tokens=20`, `do_sample=False`; CUDA synchronize로 generation latency를 측정했다. 이미지 RPC/preprocessing 시간은 이 latency에 포함하지 않았다.

| 지표 | 실측 |
|---|---:|
| Base + resize + LoRA 로딩 | 17.010 s (tokenizer/processor 사전 로드 제외) |
| 로딩 직후 allocated / reserved | 6.503 / 7.061 GiB |
| 로딩 과정 peak allocated | 6.937 GiB |
| 첫 inference latency | 1418.046 ms |
| 이후 10회 mean / median / p95 | **733.371 / 751.497 / 883.571 ms** |
| Inference peak allocated / reserved | **6.801 / 7.145 GiB** |
| 로딩 직후 GPU 전체 nvidia-smi | 9163 MiB |
| 모델 단독 구간 GPU 전체 peak | **9272 MiB / 9.055 GiB**, desktop/context 포함 |
| 프로세스 RSS 로딩 peak / 추론 종료 | 7.196 / 1.856 GiB |
| WSL `psutil.virtual_memory().used` peak | 2.412 GiB; file cache를 포함하는 물리 RAM 합계와 다른 회계 범위 |
| Windows 전체 RAM peak | 30.804 GiB; 다른 프로그램·WSL cache 등 포함 |

Tensor allocated/reserved 기준으로 사용자 정의 **GREEN(≤8GiB)**이다. GPU 전체 9.055GiB는 desktop과 CUDA context까지 포함하므로 모델 tensor peak로 대체하지 않는다. RSS는 file-mapped checkpoint pages도 포함한다. Host RAM이 높은 상태였으므로 추론 RAM만으로 전체 시스템 여유를 판단하지 않는다.

11회 모두 raw output은 `Action: 96 49 49</s>`. 원본 99-bin 규칙으로 `forward=4.897959m, down=0m, yaw=0rad, stop=False`로 파싱됐다. LAND와 zero-action stop 분기는 별도 adapter test에서 확인했으며 **실제 모델이 LAND를 생성했다는 증거는 없다**.

## 공존 측정의 범위

Windows simulator가 실행된 상태에서 두 번째 NF4+LoRA 로딩도 성공했다. 로딩/scene 초기화 동안 전체 GPU peak **9912MiB(9.680GiB)**, 남은 공간 **2315MiB(2.261GiB)**, OOM 없음. 그러나 topic 초기화 timeout 이후 중단하여 **동시 카메라 획득·추론·비행의 peak는 미측정**이다. 이 수치로 완성된 closed-loop 공존을 승인하지 않는다.

## 원시 자료와 재현 명령

- `outputs/integration/model-downloads.json`: 파일별 크기/SHA와 revision
- `base-metadata.json`, `adapter-metadata.json`: 다운로드 직전 HF API 결과
- `model-validation.json`, `model-validation.log`: 11회 raw 출력/파싱/시간/메모리
- `model-resource-samples.jsonl`, `model-windows-resources.jsonl`: GPU 전체/RAM sampling
- `integration-freeze.txt`, `install.log`, `install-online.log`, `upstream-hashes.json`

```bash
# 반드시 simulator를 끈 상태에서 단독 검증
/home/gyuhan/uav-vla-smoke/integration/bin/python \
  /mnt/c/Users/leegy/Desktop/drone/scripts/validate_aerovla_int4.py
```

현재 실패 상태에서는 이 명령을 자동으로 재실행하지 않는다. 다운로드된 base/LoRA와 실험 환경은 다음 통신 수정 검증을 위해 보존했다.
