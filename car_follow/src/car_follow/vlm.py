"""Vision-language first pick (docs/demo1_plan.md 2.1, docs/demo1.md 6): which detected car does the sentence name?

Set-of-Mark: every candidate box from the detector gets a number drawn on the drone's image, and a VLM is asked
which number the sentence names. The answer is read as probabilities, not as generated text:

    picker = make("qwen3-vl-2b")                     # or "internvl3.5-2b", "qwen3-vl-4b-4bit", ...
    result = pick(picker, bgr, boxes, sentence)      # boxes: [(x0, y0, x1, y1)] from the detector
    result.top, result.probs, result.margin, result.agreement

One forward pass per question gives the logit of each candidate number's token at the answer position (the
numbers 1-9 are single tokens). The question is asked `permutations` times (3 by default) with the numbers shuffled
over the boxes, and the probabilities are averaged per box; this removes the model's preference for particular
numbers, and how often the top box changes between shuffles is itself an ambiguity signal (`agreement`).
Returned per pick, for the layer-1 ambiguity monitor (docs/research_plan.md):
    probs        probability of each box (averaged over the shuffles)
    top          the box with the highest probability
    margin       p(top-1) - p(top-2)
    log_ratio    log(p(top-1) / p(top-2))
    agreement    share of shuffles whose own top-1 is `top`
    shuffles     per shuffle: the numbers given to the boxes and the probabilities

Adapters share one interface (`label_logits`, `info`, `unload`) so models can be swapped like the detectors
(src/car_follow/detector.py). Model files live in assets/models/hf (Hugging Face cache layout, local only) and are
loaded offline; only models whose code ships with the installed transformers are used (no remote code).
"""
import gc
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = ROOT / "assets" / "models" / "hf"
MAX_CANDIDATES = 9        # the numbers 1-9 are single tokens
# The files fetched from Hugging Face (no README, no remote code); the same list finds them offline.
FILES = ["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "*.model", "merges.txt", "vocab*"]

PROMPTS = {
    "v1": ("The image is from a drone camera looking down at a road from behind and above. Each candidate vehicle "
           "is marked with a number in a white box. Command: \"{sentence}\"\n"
           "Which numbered vehicle does the command refer to? Answer with the number only."),
    # v2 (docs/demo1.md 6.3): v1 plus the camera geometry the drone knows. Zero-shot models read "furthest back" as
    # "farthest from the camera"; from behind the traffic it is the car nearest the drone.
    "v2": ("The image is from a drone camera flying behind the traffic and looking forward along the road, so the "
           "vehicles drive away from the camera, toward the top of the image. \"Ahead\" or \"front\" means farther "
           "along the road: higher in the image and farther from the drone. \"Back\" or \"behind\" means closer to "
           "the drone: lower in the image. Left and right are as seen in the image, which is also the drivers' left "
           "and right. Each candidate vehicle is marked with a number in a white box. Command: \"{sentence}\"\n"
           "Which numbered vehicle does the command refer to? Answer with the number only."),
}


# --- marks ---

def draw_marks(bgr, boxes, labels, thickness=2, font_scale=0.6):
    """The image with each box outlined and its label in a white tag above its top-left corner."""
    img = bgr.copy()
    h, w = img.shape[:2]
    for (x0, y0, x1, y1), label in zip(boxes, labels):
        x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), thickness)
        text = str(label)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        tx = min(max(0, x0), w - tw - 6)
        ty = y0 - 4 if y0 - th - 8 >= 0 else min(h - 2, y1 + th + 6)
        cv2.rectangle(img, (tx, ty - th - 4), (tx + tw + 6, ty + 2), (255, 255, 255), -1)
        cv2.rectangle(img, (tx, ty - th - 4), (tx + tw + 6, ty + 2), (0, 0, 0), 1)
        cv2.putText(img, text, (tx + 3, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def candidates(boxes, scores, limit=MAX_CANDIDATES):
    """Indices of at most `limit` boxes, the highest scores first."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return [int(i) for i in order[:limit]]


# --- the pick ---

@dataclass
class Pick:
    probs: np.ndarray
    top: int
    margin: float
    log_ratio: float
    agreement: float
    shuffles: list = field(default_factory=list)
    seconds: float = 0.0


def softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def pick(picker, bgr, boxes, sentence, permutations=3, prompt="v1", seed=0):
    """Ask `picker` which of `boxes` the sentence names, with the numbers shuffled `permutations` times."""
    n = len(boxes)
    if n == 0:
        return None
    if n > MAX_CANDIDATES:
        raise ValueError(f"at most {MAX_CANDIDATES} candidates (got {n}); choose them with candidates()")
    rng = np.random.default_rng(seed)
    rgb_cache = {}
    t0 = time.perf_counter()
    total = np.zeros(n)
    shuffles = []
    for s in range(permutations):
        numbers = np.arange(1, n + 1) if s == 0 else rng.permutation(np.arange(1, n + 1))
        key = tuple(numbers)
        if key not in rgb_cache:
            rgb_cache[key] = np.ascontiguousarray(draw_marks(bgr, boxes, numbers)[:, :, ::-1])
        text = PROMPTS[prompt].format(sentence=sentence)
        logits = picker.label_logits(rgb_cache[key], text, [str(k) for k in range(1, n + 1)])
        p_number = softmax(logits)                      # over the numbers 1..n
        p_box = p_number[numbers - 1]                   # box i carries number numbers[i]
        total += p_box
        shuffles.append({"numbers": numbers.tolist(), "probs": p_box.round(4).tolist(), "top": int(p_box.argmax())})
    probs = total / permutations
    order = np.argsort(-probs)
    top = int(order[0])
    p1, p2 = probs[order[0]], (probs[order[1]] if n > 1 else 0.0)
    return Pick(probs=probs, top=top, margin=float(p1 - p2),
                log_ratio=float(math.log(max(p1, 1e-9) / max(p2, 1e-9))) if n > 1 else float("inf"),
                agreement=sum(s["top"] == top for s in shuffles) / permutations, shuffles=shuffles,
                seconds=time.perf_counter() - t0)


# --- adapters ---

class TransformersVLM:
    """A VLM whose code ships with transformers, loaded from the local cache (offline)."""

    def __init__(self, name, repo, quant=None, max_pixels=None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        self.name, self.repo, self.quant = name, repo, quant
        self._torch = torch
        path = snapshot_download(repo, cache_dir=str(HF_CACHE), local_files_only=True, allow_patterns=FILES)
        kwargs = {"dtype": torch.bfloat16, "device_map": "cuda"}
        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.bfloat16)
        elif quant == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        proc_kwargs = {"max_pixels": max_pixels} if max_pixels else {}
        self.processor = AutoProcessor.from_pretrained(path, **proc_kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(path, **kwargs).eval()
        self.load_s = time.perf_counter() - t0
        self.weights_mib = (torch.cuda.memory_allocated() - before) / 2 ** 20
        self.tokenizer = self.processor.tokenizer
        self._label_ids = {}
        torch.cuda.reset_peak_memory_stats()

    def token_id(self, label):
        if label not in self._label_ids:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{self.name}: label {label!r} is {len(ids)} tokens")
            self._label_ids[label] = ids[0]
        return self._label_ids[label]

    def prompt_inputs(self, rgb, text):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return chat, self.processor(text=[chat], images=[rgb], return_tensors="pt").to(self.model.device)

    def label_logits(self, rgb, text, labels):
        """Logit of each label's token as the first token of the answer."""
        torch = self._torch
        _, inputs = self.prompt_inputs(rgb, text)
        with torch.inference_mode():
            out = self.model(**inputs, logits_to_keep=1)
        last = out.logits[0, -1].float()
        return last[[self.token_id(l) for l in labels]].cpu().numpy()

    def generate(self, rgb, text, max_new_tokens=8):
        """Free generation (to check what the model would answer by itself)."""
        torch = self._torch
        _, inputs = self.prompt_inputs(rgb, text)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def info(self):
        t = self._torch
        return {"name": self.name, "repo": self.repo, "quant": self.quant or "bf16", "load_s": round(self.load_s, 1),
                "weights_mib": round(self.weights_mib), "peak_mib_since_load": round(t.cuda.max_memory_allocated()
                                                                                     / 2 ** 20),
                "license": "Apache-2.0"}

    def unload(self):
        self.model = None
        gc.collect()
        self._torch.cuda.empty_cache()


MODELS = {
    "qwen3-vl-2b": lambda: TransformersVLM("qwen3-vl-2b", "Qwen/Qwen3-VL-2B-Instruct"),
    "qwen3-vl-4b-4bit": lambda: TransformersVLM("qwen3-vl-4b-4bit", "Qwen/Qwen3-VL-4B-Instruct", quant="4bit"),
    "qwen3-vl-4b-8bit": lambda: TransformersVLM("qwen3-vl-4b-8bit", "Qwen/Qwen3-VL-4B-Instruct", quant="8bit"),
    "internvl3.5-2b": lambda: TransformersVLM("internvl3.5-2b", "OpenGVLab/InternVL3_5-2B-HF"),
}


def make(name):
    if name not in MODELS:
        raise ValueError(f"unknown VLM {name!r}; known: {', '.join(MODELS)}")
    return MODELS[name]()
