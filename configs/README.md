# Baseline 설정

- `scene_basic_drone.jsonc`, `robot_quadrotor_fastphysics.jsonc`: 성공한 실험의 실제 설정을 그대로 복사했다. Drone1, Front/Down/Chase 256×256, camera interval 0.0333333초를 유지한다. `projectairsim_probe.prepare_config()`가 로컬 `outputs/platform_final/sim_config/`에 복사한다.
- `base-model.json`, `adapter-model.json`: 다운로드를 검증한 고정 Hugging Face revision, 파일 크기와 제공된 LFS SHA-256. 모델 파일이나 개인 캐시 경로는 포함하지 않는다.
- `projectairsim-release.json`: 실제 사용한 Windows Blocks 배포의 URL·크기·SHA-256이다.
- `*-requirements.txt`: 각 환경에서 직접 사용하는 패키지 버전이다. 전체 간접 의존성 freeze는 로컬 실험 기록에 보존한다.

JSONC 설정은 [Project AirSim의 공식 예제](https://github.com/iamaisim/ProjectAirSim/tree/v1.0.1/client/python/example_user_scripts/sim_config)에서 파생되었다. 카메라 구성만 당시 smoke 설정대로 변경했다. 해당 설정의 원본 MIT 고지는 [ProjectAirSim-LICENSE](ProjectAirSim-LICENSE)에 보존한다. 이 고지는 위 Project AirSim 파생 설정에 적용된다.

forward/down/yaw 해석과 안전 범위는 현재 `src/integration/projectairsim_action_adapter.py`에 있다. 이번 정리에서 동작을 바꾸거나 새 failure 설정을 만들지 않았다.
