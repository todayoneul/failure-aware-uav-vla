"""Vehicle detectors for demo 1 config 2 (docs/demo1.md 5.5): one interface, one adapter per model.

    detector = make("rfdetr-small")          # or "gt"; later e.g. "yolo26s", "dfine-s"
    boxes = detector.detect(Frame(bgr))      # [Box] of vehicles in the drone's RGB frame

Every adapter takes the same `Frame` (the 960 x 540 BGR image of the drone camera; the ground-truth adapter also
reads the instance-ID image) and returns `Box`es in that frame's pixels, keeping only vehicle classes
(`VEHICLE_LABELS`), with the model's own confidence down to `threshold`. The tracker (src/car_follow/tracker.py)
uses the scores for its two matching rounds, so adapters should report low-confidence boxes too.

To add a model: write a class with `name`, `detect(frame)` and `info()` (and `close()` if it holds GPU memory), and
register it in `MODELS`. Model packages are imported only when their adapter is made, so the rest of the project
runs without them.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "assets" / "models"            # local only (git-ignored)
VEHICLE_LABELS = ("car", "truck", "bus")        # COCO names; CARLA's traffic here has 4-wheel vehicles only
LOW_THRESHOLD = 0.1                             # the tracker's second round goes down to 0.1


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    label: str

    @property
    def xyxy(self):
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Frame:
    bgr: np.ndarray                 # H x W x 3, uint8
    ids: np.ndarray = None          # instance actor IDs (ground-truth adapter only)
    car_ids: tuple = ()             # actor IDs of the recorded cars (ground-truth adapter only)


class GroundTruth:
    """A perfect detector: the visible-pixel box of every recorded car with at least `min_px` pixels, score 1."""
    name = "gt"

    def __init__(self, min_px=30):
        self.min_px = min_px

    def detect(self, frame):
        ids, counts = np.unique(frame.ids, return_counts=True)
        wanted = {int(a) for a in frame.car_ids}
        out = []
        for a, n in zip(ids.tolist(), counts.tolist()):
            if a in wanted and n >= self.min_px:
                ys, xs = np.nonzero(frame.ids == a)
                out.append(Box(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1), 1.0, "car"))
        return out

    def info(self):
        return {"name": self.name, "license": "n/a", "vram_mib": 0}


class RFDETR:
    """RF-DETR (Roboflow, Apache-2.0) pretrained on COCO, through the `rfdetr` package. Sizes: nano, small, medium,
    base. The package fetches the weights from repo.roboflow.com into assets/models/rfdetr/ on first use and checks
    their MD5. The predictor is compiled for the fixed input (a CUDA graph on this PC) and resizes every frame to the
    model's square resolution (512 for small)."""
    SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "base": "RFDETRBase"}

    def __init__(self, size="small", threshold=LOW_THRESHOLD, compile=True, dtype="float32"):
        import torch
        import rfdetr
        from rfdetr.assets.coco_classes import COCO_CLASSES
        self.name = f"rfdetr-{size}"
        self.threshold = threshold
        folder = WEIGHTS / "rfdetr"
        folder.mkdir(parents=True, exist_ok=True)
        weights = folder / f"rf-detr-{size}.pth"   # fetched by the package (MD5 checked) when missing
        torch.cuda.reset_peak_memory_stats()
        source = getattr(rfdetr, self.SIZES[size])(pretrain_weights=str(weights))
        self.resolution = int(source.model_config.resolution)
        self.compile_error = None
        try:   # an optimized predictor (fixed shape; CUDA graph where the installed PyTorch supports it)
            self.model = source.inference(compile=compile, dtype=dtype)
            self.compiled = compile
        except Exception as e:  # fall back to the eager predictor; recorded in info()
            self.compile_error = repr(e)[:300]
            self.model = source.inference(compile=False, dtype=dtype)
            self.compiled = False
        del source                                   # the predictor owns its own copy of the weights
        torch.cuda.empty_cache()
        self.dtype = dtype
        self.labels = {int(k): v for k, v in COCO_CLASSES.items()}
        self.weights = weights
        self._torch = torch

    def detect(self, frame):
        rgb = np.ascontiguousarray(frame.bgr[:, :, ::-1])
        d = self.model.predict(rgb, threshold=self.threshold, include_source_image=False)
        out = []
        for (x0, y0, x1, y1), score, cid in zip(d.xyxy, d.confidence, d.class_id):
            label = self.labels.get(int(cid), str(int(cid)))
            if label in VEHICLE_LABELS:
                out.append(Box(float(x0), float(y0), float(x1), float(y1), float(score), label))
        return dedupe(out)

    def info(self):
        t = self._torch
        return {"name": self.name, "license": "Apache-2.0", "package": "rfdetr", "resolution": self.resolution,
                "compiled": self.compiled, "compile_error": self.compile_error, "dtype": self.dtype,
                "weights": str(self.weights.relative_to(ROOT)),
                "weights_mib": round(self.weights.stat().st_size / 2 ** 20, 1) if self.weights.exists() else None,
                "vram_peak_mib": round(t.cuda.max_memory_allocated() / 2 ** 20, 1),
                "vram_reserved_mib": round(t.cuda.memory_reserved() / 2 ** 20, 1),
                "torch": t.__version__, "cuda": t.version.cuda, "gpu": t.cuda.get_device_name(0)}


MODELS = {
    "gt": lambda: GroundTruth(),
    "rfdetr-nano": lambda: RFDETR("nano"),
    "rfdetr-small": lambda: RFDETR("small"),
    "rfdetr-medium": lambda: RFDETR("medium"),
}


def make(name):
    if name not in MODELS:
        raise ValueError(f"unknown detector {name!r}; known: {', '.join(MODELS)}")
    return MODELS[name]()


def timed(detector, frame):
    """(boxes, seconds) with the GPU work finished inside the timing."""
    t0 = time.perf_counter()
    boxes = detector.detect(frame)
    torch = getattr(detector, "_torch", None)
    if torch is not None:
        torch.cuda.synchronize()
    return boxes, time.perf_counter() - t0


def dedupe(boxes, max_iou=0.7):
    """Drop a box that overlaps a higher-scoring one by more than `max_iou`, whatever the labels. DETR-style models
    have no NMS and can return one car twice, e.g. as "car" 0.84 and "truck" 0.19."""
    kept = []
    for b in sorted(boxes, key=lambda b: -b.score):
        if all(iou(b.xyxy, k.xyxy) <= max_iou for k in kept):
            kept.append(b)
    return kept


def iou(a, b):
    """IoU of two (x0, y0, x1, y1) boxes."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_truth(boxes, truth, min_iou=0.5):
    """Greedy one-to-one matching of detections to ground-truth boxes by IoU (highest first).
    Returns ({truth index: (box index, iou)}, [unmatched box indices])."""
    pairs = sorted(((iou(b.xyxy, t), i, j) for i, b in enumerate(boxes) for j, t in enumerate(truth)), reverse=True)
    got, used = {}, set()
    for v, i, j in pairs:
        if v < min_iou:
            break
        if i in used or j in got:
            continue
        got[j] = (i, v)
        used.add(i)
    return got, [i for i in range(len(boxes)) if i not in used]
