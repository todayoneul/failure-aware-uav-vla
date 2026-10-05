"""Publish only selected real camera images and a wall-time flight GIF."""
import json
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/demo_views"
EXAMPLES = ROOT / "outputs/examples"


def frame_durations(samples, end_elapsed_s):
    times = [row["elapsed_s"] for row in samples] + [end_elapsed_s]
    if len(samples) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("At least two chronological frames and a later recording end are required")
    return [max(10, round((b-a)*100)*10) for a, b in zip(times, times[1:])]


def main():
    capture = json.loads((OUT / "capture-result.json").read_text())
    if capture["status"] != "PASS" or capture.get("cleanup_error"):
        raise RuntimeError("A completed, safely cleaned-up capture is required")
    samples = capture["samples"]
    # Older capture logs predate the explicit recording end; hold their last frame for one measured interval.
    end = capture.get("recording_end_elapsed_s", samples[-1]["elapsed_s"] + samples[-1]["elapsed_s"]-samples[-2]["elapsed_s"])
    durations = frame_durations(samples, end)
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    for source, target in (("hero_drone.png", "hero_drone.png"), ("observer_hero.png", "observer_view.png")):
        shutil.copyfile(OUT / source, EXAMPLES / target)
    frames = []
    for row in samples:
        with Image.open(OUT / "frames" / row["frame"]) as image:
            rgb = image.convert("RGB").resize((480, 270), Image.Resampling.LANCZOS)
            frames.append(rgb.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    gif = OUT / "drone_flight.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=2)
    size = gif.stat().st_size
    if size > 4*1024**2:
        raise RuntimeError(f"GIF is too large to publish: {size} bytes; retain hero only")
    shutil.copyfile(gif, EXAMPLES / gif.name)
    result = {"view": capture["selected"], "frame_count": len(frames), "duration_ms": sum(durations),
              "size_bytes": size, "resolution": [480, 270],
              "kind": "actual model-free scripted flight; recorded wall-time intervals; no generated frames"}
    (OUT / "media-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
