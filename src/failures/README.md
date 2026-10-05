# Failure 모듈

`gaussian_blur.py`는 실제 AeroVLA preprocessing 전에 Front/Down에 적용하는 Gaussian Blur다. 기본 OFF/MEDIUM, LOW/MEDIUM/HIGH, 비파괴 apply를 제공한다. `control.py`는 Windows B·1/2/3 버튼과 WSL runner 사이의 작은 atomic file protocol이다.

기존 model-free blur/drift 데모는 `scripts/projectairsim_probe.py`에 별도로 보존했다. 자동 감지·분류·복구·replanning 및 다른 injection은 구현하지 않았다.

실행: [Gaussian Blur 사용 안내](../../docs/gaussian_blur_demo.md). 다음 후보: [failure_plan.md](../../docs/failure_plan.md).
