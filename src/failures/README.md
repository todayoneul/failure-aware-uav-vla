# 앞으로 추가할 failure 모듈

아직 구현하지 않았다. 현재 model-free blur/drift 데모는 `scripts/projectairsim_probe.py`에 있으며, 재사용 가능한 failure framework는 별도 다음 작업이다.

- Visual: blur, brightness, noise, occlusion, frame drop
- Control: lateral/yaw/vertical drift, delay, scaling, control loss
- Environment: obstacle, target occlusion/movement, blocked path, wrong landmark

첫 후보와 데모 아이디어: [failure_plan.md](../../docs/failure_plan.md).
