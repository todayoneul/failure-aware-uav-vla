# Failure-aware UAV VLA 실행 가능성 조사

조사일: 2026-10-04 (Asia/Seoul). 대상: RTX 5070 12GB Windows PC, TravelUAV + AeroVLA.

## 1. 결론과 검증 범위

**갱신 판정: ⚠️ 기술 경로 일부를 실측했으나 baseline은 미재현이다.** WSL의 torch 2.7.1/cu128 + BF16 + bnb 0.48.2 NF4 + 작은 HF 모델 로딩은 통과했다. TravelUAV BrushifyUrban은 WSL에서 CPU Vulkan renderer를 사용했다. Windows AirSim Blocks는 RTX 5070 렌더링과 WSL RPC 중계가 성공했다. 이에 권장 후보를 **B. Windows simulator + WSL inference**로 갱신하되 동일 TravelUAV Windows scene 확보를 조건으로 둔다. Gate 2는 부분 검증이고 Gate 3 prototype은 미실행이다. checkpoint 다운로드 권고는 **NO**. 상세 로그·필수 표: [compatibility_smoke_test.md](compatibility_smoke_test.md).

| 근거 표시 | 의미 |
|---|---|
| `VERIFIED_SOURCE` | 현재 공식 코드·설정·배포 메타데이터를 직접 확인 |
| `VERIFIED_LOCAL` | 이 PC에서 하드웨어 조회 또는 실제 smoke 실행·측정으로 확인; 각 범위를 명시 |
| `AUTHOR_REPORTED` | 논문 또는 유지관리자의 보고; 이 PC에서 재측정하지 않음 |
| `ESTIMATED` | 명시한 산술 또는 용량 계획 가정; 측정 결과가 아님 |
| `UNTESTED` | 제안하는 변경/조합; 설치·실행 검증 전 |
| `MISSING` / `CONFLICT` | 근거 부재 / 문서와 코드 또는 자료 사이 불일치 |
| `NOT_COMPARABLE` | 실행 조건·split·precision 등이 달라 직접 비교할 수 없음 |

최초 조사에서는 작은 source/metadata만 읽었다. 이후 사용자가 승인한 3단계 smoke 범위에서 Git을 초기화하고 `researchuav-vla-feasibility` branch를 만들었다. 별도 Python 환경, TravelUAV BrushifyUrban ZIP 하나(1.531GB), Windows fallback Blocks(0.259GB)를 설치/실행했다. 작은 random Llama는 로컬 생성했다. OpenVLA/AeroVLA checkpoint·raw dataset·navigation episode·학습·benchmark·failure module은 수행하지 않았다. 임시 upstream checkout은 수정하지 않았으며 AirSim client encoding patch는 격리한 테스트 환경에만 적용했다.

## 2. 확인한 버전과 로컬 환경

### 공식 자원 고정점

| 자원 | 조회 시 revision | 확인 내용 |
|---|---|---|
| [AeroVLA][A-repo] | `2c5ae0987a484ab92f00dd9d9ed493cb3e98e492` (2026-09-16) | README, requirements, 로더, 학습·평가·제어·서버 코드, open/closed issues |
| [TravelUAV][T-repo] | `5cc26e9a4a55b9c788e918f7c3bb2dc5076a85e6` (2026-01-22 UTC) | README, requirement, LLaMA-UAV 문서, 전처리·평가·서버 코드, issues |
| [OpenVLA base][B-files] | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` | config, custom Python, 가중치 크기·parameter 메타데이터 |
| [AeroVLA LoRA][L-files] | `196f2f3253b69df6e90ac10b6ae041c7b3a9569e` | adapter config, 파일 크기·digest |
| [TravelUAV raw][D-files] | `faa8f2514156455ea7423464cc1295e6f92575cb` | map별 분할 압축 파일 목록·byte 크기 |
| [TravelUAV environments][E-files] | `44de5739a95a2f6a88767446421cddada9606642` | 배포 환경 파일 목록·byte 크기 |
| [TravelUAV split/meta][J-files] | `5a1ed4c99d3e7bb2b35fa34135910cb7687fc831` | validation frame index와 map/object metadata |

AeroVLA는 이전 이름 AerialVLA에서 변경되었으며 HF adapter 저장소 이름은 여전히 `XuPeng23/AerialVLA`이다. 논문은 [arXiv 2603.14363v1][A-paper], TravelUAV 논문은 [arXiv 2410.07087][T-paper]를 확인했다. 현재 README의 ECCV 2026 표기와 arXiv의 이전 제목은 자원 이름 변경을 반영한다.

### 이 PC의 읽기 전용 확인

| 항목 | `VERIFIED_LOCAL` |
|---|---|
| OS | Windows 11 Pro `10.0.26300` |
| GPU | NVIDIA GeForce RTX 5070; `nvidia-smi` 총량 12,227 MiB (약 11.94 GiB) |
| Windows NVIDIA driver | `591.86` |
| GPU 현재 점유 | 조회 순간 약 1,557 MiB 사용, 10,387 MiB free; 실행 peak가 아님 |
| 시스템 RAM | 약 31.11 GiB |
| C: 여유 | 약 351.21 GiB; 조회 이후 변동 가능 |
| WSL | Ubuntu, WSL version 2; Ubuntu `24.04.4 LTS` |
| WSL kernel / CUDA 장치 인터페이스 | `6.6.87.2-microsoft-standard-WSL2`, `/dev/dxg` 존재, WSL `nvidia-smi` 경로 존재 |
| WSL RAM / swap | `free -h` 약 15 GiB / 4 GiB |
| WSL system Python | `3.12.3`; 제안 환경의 Python 3.10과 구별 |
| 그래픽 진단 | vulkan-tools 설치 후 `vulkaninfo`: llvmpipe, physical device Type CPU. UE도 같은 renderer 선택 |
| 실제 GPU 연산 | Python 3.10.14 / torch 2.7.1+cu128 / CC 12.0 / BF16·NF4·HF tiny load PASS |
| Windows simulator 실측 | Blocks UE 4.27.2 / D3D11 RTX 5070 / 중계 camera RPC 179.195ms |

CUDA/NF4와 두 simulator의 RGB/state/pose를 실제 실행했다. TravelUAV GPU rendering과 AeroVLA 전체 checkpoint 양자화 RAM/VRAM은 `MISSING`이다. WSL `df`의 약 954GiB 가상 여유는 C:의 실제 저장공간에 추가되는 용량이 아니다. 위 초기 idle/disk 수치는 조회 시점 값이며 실행 peak는 smoke 보고서의 범위별 계측을 따른다.

## A. AeroVLA

### 모델·입출력·checkpoint

| 항목 | 조사 결과 |
|---|---|
| 공식 저장소 | `XuPeng23/AeroVLA` |
| 구조 / base | OpenVLA-7B: Llama-2-7B language backbone + DINOv2/SigLIP fused vision backbone + projector |
| 크기 | HF API 기준 base 7,541,237,184 parameters; 이름의 7B보다 전체 vision 포함 수가 큼 |
| 기본 precision | 로더에서 `torch.bfloat16` 명시 |
| 추가 가중치 | rank 64, alpha 128, dropout 0.05 LoRA; q/k/v/o 및 gate/up/down projection에 적용, `projector`는 `modules_to_save` |
| 입력 | 마지막 observation의 front/down RGB를 세로 mosaic로 결합, image processor와 language prompt 사용 |
| 출력 | 숫자 token으로 forward/down/yaw 값을 복원하고 LAND/near-zero 조건으로 stop flag 생성 |
| 실제 inference entry | `src/vlnce_src/eval_aerovla.py` → `src/model_wrapper/aerovla_wrapper_ui.py` |
| 생성 방식 | greedy generation, 최대 20 new tokens; 현재 wrapper는 confidence/logit score를 반환하지 않음 |
| 학습 데이터 | 공개 `aerovla_train_dataset.json`와 TravelUAV raw frames; 추론 milestone에는 학습 JSON 불필요 |

근거: [base config][B-config], [adapter config][L-config], [로더][A-loader], [입력·생성·파싱][A-input]. base checkpoint에 vision/LLM 가중치가 포함되므로 Llama-2·DINOv2·SigLIP를 별도로 중복 다운로드할 필요는 없다. adapter의 실제 tensor dtype/정확한 parameter 수는 작은 메타데이터 조회로 확정하지 못했다.

| 다운로드 항목 | `VERIFIED_SOURCE` 크기 |
|---|---:|
| base weight 3 shards | 15,082,600,824 bytes (약 14.05 GiB) |
| base 저장소 전체, config/tokenizer/custom Python 포함 | 15,085,153,882 bytes |
| adapter_model.safetensors | 462,655,064 bytes (약 0.431 GiB) |
| adapter_config.json | 733 bytes |
| 학습 JSON, 이번에는 제외 | 257,288,037 bytes |

공개 LoRA가 논문 checkpoint이고 base가 `openvla/openvla-7b`라는 것은 [유지관리자 보고][A-issue12]이다. adapter config는 base의 HF revision을 고정하지 않는다. 위 revision은 이번 조사 고정점이며 논문 당시 revision과 동일하다고 입증한 값은 아니다.

### requirements와 양자화 판정

[현재 requirements][A-req]는 torch 2.1.2 / torchvision 0.16.2, transformers 4.42.4, peft 0.11.1, accelerate 0.32.1, bitsandbytes 0.43.1, timm 0.9.10, flash-attn 2.5.8을 고정한다. README의 CUDA 11.8 명령까지 그대로 적용하면 RTX 5070 이행 후보와 충돌한다. 현대 torch 설치 후 upstream requirements를 그대로 설치하는 것도 torch를 다시 구버전으로 바꾼다.

**INT4는 upstream에서 구현된 추론 기능이 아니다.** 실제 wrapper는 BF16 base → tokenizer resize → unmerged LoRA 부착 → 모델 전체 `.to(cuda)` 순서다. `merge_and_unload`는 주석이다. `BitsAndBytesConfig`, `load_in_4bit`, `device_map`은 없다. 따라서 NF4 사용은 다음 단계의 별도 변경·검증 대상이다. [실제 loader][A-loader]

제안 경로는 base를 처음부터 NF4로 로드하고 load-time placement를 정한 뒤 LoRA/projector를 부착하는 방식이다. BF16 전체를 GPU에 올려 merge한 후 양자화하는 방식은 12GB 제약을 통과하지 못한다. Transformers 4.42.4의 base model `.to()`는 bnb 모델을 제한하지만 바깥 PeftModel의 이동 경로까지 동일 예외라고 단정하지 않는다. 전체 모델 이동을 제거/분기하고 image·adapter·vision dtype을 확인해야 한다. adapter merge 없이 시작하며, 단순 package 교체만으로 해결되었다고 판단하지 않는다. [Transformers 코드][H-to], [PEFT bnb 코드][H-peft]

CUDA/PyTorch/VRAM 최소값을 AeroVLA가 공식 5070 기준으로 제시한 근거는 없다. 논문 §4.2의 **RTX 4090에서 17GB 추론, 4×24GB 4090 BF16 학습**은 `AUTHOR_REPORTED`이며 5070·NF4·WSL의 측정값으로 사용할 수 없다. [논문][A-paper]

### TravelUAV 연결과 재현상 불일치

| 항목 | 현재 근거와 처리 |
|---|---|
| 환경 연결 | TravelUAV의 precompiled 환경·raw trajectory·map/object meta를 AeroVLA 자체 AirSim server/client/evaluator가 사용 |
| 전처리 위치 | TravelUAV root의 `tools/`가 아니라 `Model/LLaMA-UAV/tools/generate_merged_json.py` 사용 |
| adapter 위치 | README `checkpoints/aerial_vla`와 eval script `checkpoints/aero_vla` 불일치; script/HF folder 기준으로 후자 사용 |
| metric 명령 | README `bash scripts/metric.sh aero_vla`와 달리 script는 argument를 읽지 않고 `MODEL_NAME="aerial_vla"` 고정; 직접 metric Python CLI 사용 권장 |
| DDP port | eval script `80005`는 TCP 범위 밖; plain single process에서는 DDP 미초기화 가능, custom command는 유효한 `20001` 사용 |
| UI | README의 실시간 UI 안내와 달리 GUI 생성/갱신은 주석; Tk import는 남아 있어 import dependency 확인 필요 |
| yaw/control | 논문의 yaw 범위와 실제 `[-1.1,1.1]` 범위가 다름; 회전 후 조건부 이동이 실행되며 LAND는 evaluator stop flag |

근거: [평가 script][A-eval], [metric script][A-metric], [전처리 tool][T-merge], [wrapper][A-input], [실제 controller][A-control]. 수정할 때 원본 SHA와 patch를 함께 기록해야 한다.

**연구에서 보존할 baseline의 특권 정보와 종료 규칙:** 현재 방향 prompt는 `mark.json`에서 유래한 정확한 target 위치와 simulator pose로 계산된다. 모델 입력은 coarse text이지만 전처리를 IMU만으로 수행한다고 표현할 수 없다. [GT 전달 코드][A-gt]와 [저자 설명][A-issue2]가 이를 확인한다. 또한 evaluator는 depth proximity, stuck, target 거리 10-frame 연속 증가를 collision/done으로 처리한다. 이 값들은 simulator의 실제 접촉 collision과 별도로 기록할 필요가 있다. 현재 parser는 잘못된 출력을 0-vector로 바꾸고 stop으로 해석할 수 있다. 이번에는 변경하지 않고 향후 logging 필요성을 기록한다. [종료 규칙][A-termination], [parser][A-input]

### 최신 issue 상태에서 확인한 위험

2026-10-04 API snapshot에서 AeroVLA는 open 12 / closed 6 issues였다. issue가 closed라는 사실을 호환성 인증으로 보지 않는다.

- [#18][A-i18]: 현재 열려 있는 재현/렌더링·physics 문제. 저자의 하드웨어 원인 설명은 가설이며 5070 검증 결과가 아니다.
- [#16][A-i16]: 공식 mini evaluation set 없음; 한 simulator와 한 model process 단위 재현부터 시작해야 함.
- [#12][A-issue12]: 공개 base/LoRA 확인과 사용자 결과 차이, runtime timing 논의가 함께 존재.
- [#9 및 공식 troubleshooting][A-rpc]: RPC dependency/encoding 수정 안내. 평균 지연 보고를 로컬 latency로 옮기지 않음.
- [#10][A-i10], [#11][A-i11], [#6][A-i6]: Vulkan/환경 바이너리/누락된 merged JSON 관련 설치·준비 위험.

조회한 issue들에서 **RTX 5070 + WSL2 + NF4 AeroVLA의 성공을 입증하는 측정 근거는 찾지 못했다**.

## B. TravelUAV

### simulator·OS·데이터·평가

| 항목 | 판정 |
|---|---|
| simulator | Unreal Engine 기반 AirSim UAV 환경; 별도 simulator 제작 불필요 |
| AirSim dependency | Python API와 RPC 필요. 환경 바이너리에 plugin 포함; 초기 목표에 AirSim/UE 소스 빌드 불필요 |
| 엔진 버전 | 실제 BrushifyUrban과 Windows Blocks 로그에서 UE 4.27.2 확인. 다른 배포 map 및 AirSim C++ plugin revision은 `MISSING`; Python SDK version과 동일시하지 않음 |
| 공개 실행 경로 | Linux `.sh`/`LinuxNoEditor`, `netstat`, `grep`, `pkill`, POSIX signal 기반 |
| Windows native | Blocks RTX 5070 rendering/RPC는 성공. TravelUAV의 동등 Windows scene/build·launcher는 미확인 |
| WSL2 | BrushifyUrban 실행/RGB/state/pose 성공, 그러나 llvmpipe CPU rendering. RTX rendering 조건 미충족 |
| raw 구조 | map/trajectory UUID 아래 `log`, 5-view RGB/depth, `mark.json`, `object_description.json`; generator로 `merged_data.json` 생성 |
| instruction | merged JSON의 `conversations[0].value`: target 방향·각도·object description이 포함된 prompt |
| trajectory | 원본 pose series와 시작 좌표 기준 변환 trajectory; AeroVLA action과 동일한 파일 형식이 아님 |
| split 형식 | `{"json":"map/uuid/merged_data.json","frame":...}`인 frame index; episode 수로 바로 해석하면 안 됨 |
| closed loop | management server가 map을 실행하고 evaluator가 RGB/pose → model → action → 재관측 반복 |
| 공식 metric | SR, OSR, NE, SPL; 원본 정의·termination 코드 보존 |

근거: [TravelUAV README][T-readme], [dataset 안내][T-modelreadme], [server][T-server], [generator][T-merge], [AeroVLA dataset loader][A-dataset], [metric 코드][A-metricpy]. TravelUAV 자체 모델은 LLaMA-UAV와 추가 GroundingDINO/trajectory 모델 등을 사용하므로, 12GB에서 더 가벼운 대체 baseline이라는 근거가 없다. random/fixed action은 연결 확인에 사용할 수 있는 하한선이지 AeroVLA 재현을 대체하는 성공 결과가 아니다.

현재 metadata split의 seen frame index는 75,374개, 고유 trajectory는 1,418개다. ModernCityMap seen 파일은 2,873개 frame index / 124개 trajectory다. TravelUAV 원논문의 seen episode 수 1,410개와 현재 배포 수 차이를 기록해야 한다. AeroVLA 논문은 1,418개를 사용하므로 그 논문에도 같은 수 불일치가 있다고 주장하지 않는다. 버전 대조 없이 TravelUAV 원논문 benchmark 재현을 선언할 수 없다 (`NOT_COMPARABLE`). 한 episode smoke test는 split 성능의 추정치가 아니다.

### 확인된 다운로드 용량

단위: GB = 10^9 bytes, GiB = 2^30 bytes. 아래는 **압축 파일 크기**이며 압축 해제 크기가 아니다. HF 파일 메타데이터에서 현재 파일 크기를 합산했다. [raw][D-files], [env][E-files], [meta][J-files]

| 항목 | bytes | 약 GiB |
|---|---:|---:|
| ModernCityMap raw: z01 + z02 + zip | 9,847,057,165 | 9.171 |
| closeloop_envs: z01~z04 + zip | 18,645,702,495 | 17.366 |
| seen split + map meta + object meta | 10,060,038 | 0.009 |
| 위 최소 map/environment/meta 합계 | 28,502,819,698 | 26.545 |
| raw repository 파일 전체 | 482,695,739,270 | 449.546 |
| environment repository 파일 전체 | 83,732,236,876 | 77.981 |
| 전체 raw + env | 566,427,976,146 | 527.527 |

ModernCityMap은 script·spawn meta·seen split에 연결된 최소 **navigation episode 후보**다. camera/state smoke에서는 spawn/split 연결을 요구하지 않아, 가장 작은 완전한 단일 환경 BrushifyUrban(1,530,641,125 bytes; 추출 1,656,328,410 bytes)을 사용했다. 이 smoke 결과를 ModernCityMap episode 재현으로 세지 않는다. 단일 episode라도 raw는 map archive 단위이고 ModernCityMap 환경은 여러 맵을 묶은 archive이다. 개별 episode 전용 다운로드는 확인되지 않았다.

**모델까지 포함한 최소 알려진 다운로드는 약 44.05GB / 41.03GiB**다. archive 재결합·추출, Python/CUDA 환경, HF cache 중복, 로그를 고려해 **100–150GiB의 작업 여유를 계획 가정으로 예약**한다 (`ESTIMATED`, 실제 최종 용량 아님). map별 추출 크기와 output peak는 `MISSING`이므로 archive listing 후 예산을 다시 정해야 한다. 현재 C: 여유는 최소 경로를 조사할 여지는 있지만 전체 assets는 압축 상태만으로도 들어가지 않는다.

### issue로 확인한 재현성 제약

- [TravelUAV #2][T-i2]는 업로드한 Linux package의 직접 실행을 설명하고, [#6][T-i6]는 연구를 Linux에서 수행했다고 확인한다.
- [#27][T-i27]: single scene을 validation JSON에서 필터링하는 접근. episode 한 개는 별도의 subset manifest로 지정.
- [#14][T-i14], [#17][T-i17]: 공개 simulator/assistant 설정과 논문 시점 조건 차이 논의. 최신 자료와 논문 성능을 자동으로 동일시하지 않음.
- [#66][T-i66]: TravelUAV의 원래 collision code에서 `diff`/`diffs` 처리 차이 보고. AeroVLA의 별도 Assist 코드는 다르므로 이 issue를 AeroVLA의 동일 버그라고 옮기지 않는다. 수정하면 변경 baseline으로 구분.

## C. RTX 5070 compatibility

5070은 NVIDIA 표 기준 compute capability **12.0 (`sm_120`)**이며 datacenter Blackwell의 `sm_100`과 구별된다. PyTorch 2.7은 Blackwell/CUDA 12.8 지원을 도입했고, 공식 설치 표에 torch 2.7.1 + torchvision 0.22.1 + torchaudio 2.7.1의 cu128 Windows/Linux wheel이 있다. 이는 최신 버전 주장 대신 코드 이행용 고정 후보다. 현재 Windows driver 591.86은 5070 지원 driver 572.70보다 높다. [NVIDIA GPU 표][N-gpu], [PyTorch 발표][P-blog], [공식 version table][P-versions], [5070 driver][N-driver]

| Component | RTX 5070 12GB 판정 | 이유 / 미확인 사항 |
|---|---|---|
| 고정 호환 PyTorch cu128 | ✅ 로컬 실행 통과 | torch 2.7.1+cu128, CC 12.0, BF16 GPU kernel 및 tiny HF NF4 load/generate 성공 |
| upstream torch 2.1.2 + cu118 그대로 | ❌ 채택 불가 | sm120에 맞춘 현재 후보가 아님 |
| TravelUAV simulator | ⚠️ 부분 검증 | WSL BrushifyUrban CPU renderer/RPC 성공; Windows Blocks RTX/RPC 성공, 동일 TravelUAV Windows build 없음 |
| AeroVLA BF16 / FP16 전체 GPU 추론 | ❌ 어려움 | base weights만 약 14.047GiB로 GPU 총량보다 큼 |
| AeroVLA INT8 | ⚠️ 여유 부족 가능 | 이상적 base 하한 7.023GiB에 adapter/vision/runtime/simulator 추가; 구현도 없음 |
| AeroVLA INT4/NF4 | ⚠️ Gate 3 미실행 | NF4 stack 통과와 VLA 호환성은 구별. Gate 2 미충족으로 prototype/adapter/dtype/동시 peak 보류 |
| 일반 BF16 LoRA training | ❌ 현재 장비 목표로 부적합 | frozen BF16 base도 VRAM에 맞지 않음 |
| QLoRA / 매우 작은 batch training | ⚠️ 미래 검토 | NF4 base 가능성만 있음; activations·학습 code 변경 미검증, 현재 scope 밖 |
| Full fine-tuning | ❌ 현실적으로 어려움 | 18 bytes/parameter 예시로 약 126.42GiB + activations; 12GB에 맞지 않음 |

마지막 행은 [HF 메모리 구성 예시][H-memory]를 parameter 수에 적용한 산술이며 모델별 실측 요구량이 아니다.

### 충돌 package와 처리 후보

| dependency | 현재 → 후보 / 원인 |
|---|---|
| torch / torchvision | 2.1.2 / 0.16.2 → 2.7.1 cu128 / 0.22.1; 대응 wheel을 함께 선택 |
| bitsandbytes | 0.43.1 → 0.48.2; 공식 cu128/12.9 Linux·Windows 빌드에 sm120, NF4 지원 명시 |
| transformers / peft / accelerate | 4.42.4 / 0.11.1 / 0.32.1 유지로 Gate 1 통과. AeroVLA adapter/custom multimodal path는 미검증 |
| timm / OpenVLA remote code | 0.9.10 유지; 배포 custom model이 허용하는 timm/version/API 범위를 확인 |
| flash-attn | 기존 2.5.8은 sm120 지원으로 인증되지 않음. 현재 FA4의 sm120 코드 존재와 별개로 drop-in 호환은 미검증; 초기 후보에서 제외, SDPA/eager 확인 |
| xformers | AeroVLA requirements/평가 로더에 요구되지 않으므로 추가하지 않음 |
| AirSim / msgpack / tornado | 공식 수정 RPC + msgpack 1.1.2 + tornado 4.5.3 + airsim 1.8.1 실제 RPC 성공. 격리 env client encoding patch/hash 보존 |

근거: [bnb 0.48.2 versioned docs][H-bnb], [FA2 기존 지원][F-old], [현재 FA4 sm120 코드][F-new], [OpenVLA custom model][B-model], [RPC troubleshooting][A-rpc]. Gate 1은 eager를 사용했고 custom VLA 전체 조합은 미검증이다. WSL에는 Windows host GPU driver를 사용하고 Linux display driver를 설치하지 않았다. full CUDA toolkit/flash-attn build는 하지 않았지만 Triton import에는 C compiler가 필요해 gcc/libc6-dev를 설치했다. 첫 실패 전문과 원인별 복구를 smoke 보고서에 보존했다. [CUDA on WSL][N-wsl]

### VRAM: 산술과 측정을 분리

| 항목 | 현재 확인 |
|---|---|
| base BF16 weight 저장 산술 | 7,541,237,184 × 2 bytes = 14.047GiB (`ESTIMATED`) |
| 모든 base weights가 4-bit라는 이상적 하한 | parameters × 0.5 byte = 3.512GiB; nonquantized modules/metadata를 제외한 하한 |
| NF4 실제 model resident / load peak | `MISSING` |
| simulator-only peak | Windows Blocks sampled dedicated GPU 216.160MiB, shared 80.770MiB, RAM working set 367.750MiB / private bytes 604.637MiB. TravelUAV hardware VRAM `MISSING` |
| TravelUAV WSL CPU rendering RAM | BrushifyUrban process-group RSS sampled peak 1359.879MiB; hardware VRAM으로 해석하지 않음 |
| co-resident model + simulator total peak | `MISSING`; 두 구성 동시 실행 없음 |
| 실제 AeroVLA inference latency / FPS | `MISSING` |

6–8GiB를 모델 예산으로 예약할 수는 있으나 이는 **계획 가정**이고 예상 최대치로 검증한 수치가 아니다. simulator에 몇 GiB가 필요한지도 scene/renderer별 측정 전에는 확정할 수 없다. 따라서 **총 최대 VRAM을 12GB 이하라고 보장하지 않는다**. load·최장 prompt·action 생성·scene 초기화 peak를 모두 측정해야 한다. NF4는 base checkpoint 다운로드 크기를 자동으로 4분의 1로 줄이지 않는다.

## D. 권장 실행 환경

**실측 후 권장 후보: B. Windows native simulator + WSL2 inference.** WSL CUDA/NF4는 정상이고 WSL TravelUAV Vulkan은 CPU renderer였다. Windows Blocks에서는 RTX 5070 graphics와 WSL camera/state RPC가 동작했다. 단, 동일 TravelUAV Windows scene/build 확보와 launcher 변경을 조건으로 둔다. all-in-one 실행의 실패를 프로젝트 전체 불가능으로 판단하지 않는다.

| 판단 기준 | Windows native | WSL2 Ubuntu |
|---|---|---|
| 현재 배포 map/launcher와 일치 | Windows 대응 package/build 미확인 | Linux binary/bash/server를 사용 가능 |
| Linux 프로세스·netstat·signal 코드 | launcher 변경 필요 | 원본 코드와 가깝게 사용 |
| PyTorch/bnb sm120 지원 | 현대 wheel 존재, 로컬 Windows ML 미검증 | 별도 env BF16/NF4/HF load PASS |
| UE GPU 렌더링 | Blocks D3D11 RTX 5070 확인; Travel scene 미확보 | Travel BrushifyUrban llvmpipe CPU, RTX 조건 미충족 |
| 갱신 결정 | simulator 측 조건부 권장 | inference 측 권장; simulator all-in-one 보류 |

Microsoft의 WSLg/OpenGL 가속 설명과 CUDA 지원만으로 이 compiled UE map의 Vulkan 요구사항까지 충족했다고 판단할 수 없다. `vulkaninfo`/엔진 RHI 로그, software renderer 여부, 실제 RGB·depth·pose RPC와 UAV 이동을 먼저 확인한다. [Microsoft graphics][M-gui], [WSLg][M-wslg]

직접 NAT `WSL → 192.168.160.1:41461`은 timeout였고 Windows listen은 정상 확인했다. Windows에서 WSL로 먼저 연결하는 임시 TCP 중계에서는 camera/state/pose가 성공했다. 방화벽/영구 networking 설정은 변경하지 않았다. 이것은 split 구조의 실제 경로를 입증하지만 안정적 운영 topology와 같은 Travel scene은 남은 조건이다. 같은 GPU의 VRAM 경쟁도 계속된다. native Linux GPU 환경은 별도 대안으로 남긴다. [Microsoft networking][M-net]

## 3. 다음 단계 결정

1. Gate 1 완료. Gate 2는 동일 TravelUAV map의 GPU rendering을 확보해 재검증해야 한다.
2. Gate 2 통과 후에만 별도 INT4 loader prototype/Gate 3를 검증한다. 현재 OpenVLA/AeroVLA checkpoint 다운로드 권고는 **NO**다.
3. 이후 checkpoint·한 observation·동시 VRAM·episode는 별도 단계다. 이번 요청은 세 Gate 결과 보고에서 멈추며 navigation/failure 구현으로 넘어가지 않는다.

이번 조사 단계의 최종 상태는 **⚠️ 조건부 가능, 실제 baseline 미재현**이다. Failure Detection/Diagnosis/Recovery, dataset 대규모 생성, full training, world model, diffusion, RL은 수행하지 않는다.

[A-repo]: https://github.com/XuPeng23/AeroVLA
[T-repo]: https://github.com/prince687028/TravelUAV
[A-paper]: https://arxiv.org/html/2603.14363v1
[T-paper]: https://arxiv.org/abs/2410.07087
[B-files]: https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f
[B-config]: https://huggingface.co/openvla/openvla-7b/raw/47a0ec7fc4ec123775a391911046cf33cf9ed83f/config.json
[B-model]: https://huggingface.co/openvla/openvla-7b/raw/47a0ec7fc4ec123775a391911046cf33cf9ed83f/modeling_prismatic.py
[L-files]: https://huggingface.co/XuPeng23/AerialVLA/tree/196f2f3253b69df6e90ac10b6ae041c7b3a9569e
[L-config]: https://huggingface.co/XuPeng23/AerialVLA/raw/196f2f3253b69df6e90ac10b6ae041c7b3a9569e/aero_vla/adapter_config.json
[D-files]: https://huggingface.co/datasets/wangxiangyu0814/TravelUAV/tree/faa8f2514156455ea7423464cc1295e6f92575cb
[E-files]: https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env/tree/44de5739a95a2f6a88767446421cddada9606642
[J-files]: https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_data_json/tree/5a1ed4c99d3e7bb2b35fa34135910cb7687fc831
[A-loader]: https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/src/model_wrapper/aerovla_wrapper_ui.py#L20-L44
[A-input]: https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/src/model_wrapper/aerovla_wrapper_ui.py#L78-L223
[A-req]: https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/requirements.txt
[A-eval]: https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/scripts/eval_aerovla.sh
[A-metric]: https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/scripts/metric.sh
[A-metricpy]: https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/utils/metric.py
[A-control]: https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/airsim_plugin/AirVLNSimulatorClientTool_AeroVLA.py#L312-L384
[A-gt]: https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/src/vlnce_src/env_uav.py#L133-L157
[A-dataset]: https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/src/vlnce_src/env_uav.py#L114-L159
[A-termination]: https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/src/vlnce_src/closeloop_util.py#L162-L245
[A-issue2]: https://github.com/XuPeng23/AeroVLA/issues/2#issuecomment-4150273686
[A-issue12]: https://github.com/XuPeng23/AeroVLA/issues/12
[A-i18]: https://github.com/XuPeng23/AeroVLA/issues/18
[A-i16]: https://github.com/XuPeng23/AeroVLA/issues/16
[A-i10]: https://github.com/XuPeng23/AeroVLA/issues/10
[A-i11]: https://github.com/XuPeng23/AeroVLA/issues/11
[A-i6]: https://github.com/XuPeng23/AeroVLA/issues/6
[A-rpc]: https://github.com/XuPeng23/AeroVLA/blob/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/docs/assets/troubleshooting.md
[T-readme]: https://github.com/prince687028/TravelUAV/blob/5cc26e9a4a55b9c788e918f7c3bb2dc5076a85e6/README.md
[T-modelreadme]: https://github.com/prince687028/TravelUAV/blob/5cc26e9a4a55b9c788e918f7c3bb2dc5076a85e6/Model/LLaMA-UAV/README.md
[T-merge]: https://github.com/prince687028/TravelUAV/blob/5cc26e9a4a55b9c788e918f7c3bb2dc5076a85e6/Model/LLaMA-UAV/tools/generate_merged_json.py
[T-server]: https://github.com/prince687028/TravelUAV/blob/5cc26e9a4a55b9c788e918f7c3bb2dc5076a85e6/airsim_plugin/AirVLNSimulatorServerTool.py
[T-i2]: https://github.com/prince687028/TravelUAV/issues/2#issuecomment-2668599542
[T-i6]: https://github.com/prince687028/TravelUAV/issues/6#issuecomment-2677291805
[T-i27]: https://github.com/prince687028/TravelUAV/issues/27
[T-i14]: https://github.com/prince687028/TravelUAV/issues/14
[T-i17]: https://github.com/prince687028/TravelUAV/issues/17
[T-i66]: https://github.com/prince687028/TravelUAV/issues/66
[N-gpu]: https://developer.nvidia.com/cuda/gpus
[N-driver]: https://www.nvidia.com/ko-kr/geforce/news/geforce-rtx-5070-game-ready-driver/
[N-wsl]: https://docs.nvidia.com/cuda/archive/12.8.2/wsl-user-guide/index.html
[P-blog]: https://pytorch.org/blog/pytorch-2-7/
[P-versions]: https://pytorch.org/get-started/previous-versions/
[H-bnb]: https://huggingface.co/docs/bitsandbytes/v0.48.2/en/installation
[H-to]: https://github.com/huggingface/transformers/blob/v4.42.4/src/transformers/modeling_utils.py#L2756-L2796
[H-peft]: https://github.com/huggingface/peft/blob/v0.11.1/src/peft/tuners/lora/bnb.py
[H-memory]: https://huggingface.co/docs/transformers/v4.50.0/model_memory_anatomy
[F-old]: https://github.com/Dao-AILab/flash-attention/blob/v2.5.8/README.md
[F-new]: https://github.com/Dao-AILab/flash-attention/blob/e9515d5dee6ade134a33d6020d38d01ef0596996/flash_attn/cute/interface.py
[M-gui]: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps
[M-wslg]: https://github.com/microsoft/wslg#opengl-accelerated-rendering-in-wslg
[M-net]: https://learn.microsoft.com/en-us/windows/wsl/networking
