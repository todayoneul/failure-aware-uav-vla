"""Demo 1 step 6 (docs/demo1.md 6.5): does a VLM fit on the GPU next to the simulator and the detector?

Runs one trial live as config 3 will: the simulator replays the traffic with the demo's cameras (instance
segmentation, depth, RGB), RF-DETR-Small detects on every tick, and at tick 0 the VLM makes the first pick from the
detector's boxes (3 shuffles of the numbers). A thread samples the whole GPU's used memory (all processes, from
nvidia-smi) every 50 ms. Reports the used memory before loading the models, after loading them, the peak
during the pick and the peak during the rest of the trial, against the card's total.

Run from the car_follow folder:
    uv run python scripts/demo1_vlm_memory_check.py --vlm qwen3-vl-2b --trial dev-013
    uv run python scripts/demo1_vlm_memory_check.py --vlm qwen3-vl-4b-4bit --unload-after-pick
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo1_eval import OUT as EVAL_OUT, fly_trial  # noqa: E402
from scripts.demo1_follow import HZ, OUT as FOLLOW_OUT, select  # noqa: E402
from src.car_follow import detector, scorer, sim, trials, vlm  # noqa: E402


class Sampler(threading.Thread):
    """Streams the whole GPU's used memory from nvidia-smi (NVML) every `every_ms`; each sample is tagged with the
    phase the run is in. (cudaMemGetInfo is not device-wide under Windows' WDDM driver model.)"""

    def __init__(self, every_ms=50):
        super().__init__(daemon=True)
        self.samples, self.phase = [], "start"
        self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                      "--format=csv,noheader,nounits", f"-lms={every_ms}"],
                                     stdout=subprocess.PIPE, text=True)

    def run(self):
        for line in self.proc.stdout:
            try:
                used, total = (float(v) for v in line.strip().split(","))
            except ValueError:
                continue
            self.samples.append((time.monotonic(), self.phase, used, total))

    def stop(self):
        self.proc.terminate()

    def peak(self, phase):
        v = [s[2] for s in self.samples if s[1] == phase]
        return round(max(v)) if v else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vlm", default="qwen3-vl-2b", choices=sorted(vlm.MODELS))
    p.add_argument("--trial", default="dev-013")
    p.add_argument("--run", default="rec1")
    p.add_argument("--unload-after-pick", action="store_true")
    p.add_argument("--no-start", action="store_true")
    args = p.parse_args()
    import torch
    torch.cuda.init()
    sampler = Sampler()
    sampler.start()
    out = {"vlm": args.vlm, "trial": args.trial, "unload_after_pick": args.unload_after_pick}
    with sim.session(trials.MAP, hz=HZ, start=not args.no_start) as (client, world, _tm):
        sim.clear(client, world)
        world.tick()
        time.sleep(1.0)
        sampler.phase = "simulator only"
        time.sleep(2.0)
        sampler.phase = "loading models"
        det = detector.make("rfdetr-small")
        picker = vlm.make(args.vlm)
        sampler.phase = "models loaded"
        time.sleep(1.0)
        trial = select("dev", [args.trial])[0]
        rec = dict(np.load(FOLLOW_OUT / "traffic" / args.run / f"{trial['id']}.npz"))
        world.reset_all_traffic_lights()
        state = {"picker": picker}

        def probe(k, bgr, ids_img, rp, rec_, s):
            if k == 0:
                sampler.phase = "first pick"
                boxes = [b for b in det.detect(detector.Frame(bgr)) if b.score >= 0.3]
                keep = vlm.candidates([b.xyxy for b in boxes], [b.score for b in boxes])
                boxes = [boxes[i].xyxy for i in keep]
                r = vlm.pick(state["picker"], bgr, boxes, trial["sentence"]["en"])
                truth = scorer.mask_box(ids_img, rp.ids[0])
                out["pick"] = {"candidates": len(boxes), "right": bool(truth and detector.iou(boxes[r.top], truth)
                                                                       >= 0.5),
                               "margin": round(r.margin, 3), "agreement": r.agreement, "seconds": round(r.seconds, 3)}
                if args.unload_after_pick:
                    state["picker"].unload()
                    state["picker"] = None
                sampler.phase = "following"

        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        ev, log, wall = fly_trial(client, world, trial, rec, 2, "on", det=det, probe=probe)
        out["trial_seconds"] = round(time.monotonic() - t0, 1)
        out["ticks_per_s"] = round(len(ev["owner"]) / wall, 1)
        sampler.phase = "done"
        time.sleep(0.5)
    sampler.stop()
    total = sampler.samples[-1][3]
    out["gpu_total_mib"] = round(total)
    out["used_mib"] = {ph: sampler.peak(ph) for ph in ("start", "simulator only", "loading models", "models loaded",
                                                      "first pick", "following")}
    out["before_simulator_mib"] = round(min(x[2] for x in sampler.samples if x[1] == "start"))
    out["peak_mib"] = max(v for v in out["used_mib"].values() if v is not None)
    out["headroom_mib"] = round(total - out["peak_mib"])
    out["this_process_torch_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    print(json.dumps(out, indent=1))
    path = EVAL_OUT / f"vlm_memory_{args.vlm}{'_unload' if args.unload_after_pick else ''}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
