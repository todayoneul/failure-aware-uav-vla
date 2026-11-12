"""Phase 2 install check: start CARLA, spawn traffic, fly a kinematic camera "drone" after one car,
and measure FPS and GPU memory for each quality level.

The drone is a free camera moved every tick to a point behind and above the target car (no flight model).
A chase camera behind the drone gives the third-person view. Each run also checks the simulator's own
ground truth: the target's bounding box projected into the drone image, and the target's visible pixels
in an instance segmentation image.

Run from the repository root:
    uv run python scripts/carla_check.py --quality Low Epic
"""
import argparse
import json
import math
import queue
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import carla
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = ROOT / "assets" / "carla-0.9.16" / "CarlaUE4.exe"
OUTPUT = ROOT / "outputs" / "phase2" / "carla_check"


def gpu_memory_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    used, total = (int(v) for v in out.strip().splitlines()[0].split(","))
    return used, total


def server_args(quality, port, offscreen):
    args = [f"-quality-level={quality}", f"-carla-rpc-port={port}", "-nosound"]
    return args + (["-RenderOffScreen"] if offscreen else ["-windowed", "-ResX=640", "-ResY=360"])


def start_server(exe, args):
    # With the console handles inherited, the very first start never answered within 10 minutes; detached, every
    # start answered within seconds (the cause was not isolated: it may also have been the first shader warm-up).
    return subprocess.Popen([str(exe), *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_server(process):
    # CarlaUE4.exe starts the shipping binary as a child, so end the whole tree.
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    process.wait(timeout=60)


def connect(port, timeout_s):
    client = carla.Client("localhost", port)
    client.set_timeout(10.0)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            client.get_server_version()
            return client
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(2.0)


def look_at(eye, target):
    """Rotation that points a camera at eye towards target (CARLA: x forward, y right, z up, degrees)."""
    d = target - eye
    yaw = math.degrees(math.atan2(d.y, d.x))
    pitch = math.degrees(math.atan2(d.z, math.hypot(d.x, d.y)))
    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def drone_pose(vehicle, distance, height):
    t = vehicle.get_transform()
    forward = t.get_forward_vector()
    flat = carla.Vector3D(forward.x, forward.y, 0.0)
    norm = math.hypot(flat.x, flat.y) or 1.0
    eye = t.location - carla.Location(flat.x / norm * distance, flat.y / norm * distance, -height)
    return carla.Transform(eye, look_at(eye, t.location))


def chase_pose(drone, target_location, back, up):
    forward = drone.get_forward_vector()
    norm = math.hypot(forward.x, forward.y) or 1.0
    eye = drone.location - carla.Location(forward.x / norm * back, forward.y / norm * back, -up)
    return carla.Transform(eye, look_at(eye, target_location))


def camera(world, kind, width, height, fov, transform):
    bp = world.get_blueprint_library().find(kind)
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    sensor = world.spawn_actor(bp, transform)
    frames = queue.Queue()
    sensor.listen(frames.put)
    return sensor, frames


def frame_at(frames, frame, timeout=10.0):
    while True:
        image = frames.get(timeout=timeout)
        if image.frame == frame:
            return image


def bgr(image):
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3].copy()


def project_box(vehicle, cam_transform, width, height, fov):
    """Pixel box (x0, y0, x1, y1) of the vehicle's 3D bounding box in the camera image, or None if behind."""
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    world_to_cam = np.array(cam_transform.get_inverse_matrix())
    points = []
    for v in vehicle.bounding_box.get_world_vertices(vehicle.get_transform()):
        x, y, z, _ = world_to_cam @ np.array([v.x, v.y, v.z, 1.0])
        if x <= 0.1:
            return None
        # CARLA camera axes (forward, right, up) to image axes.
        points.append((focal * y / x + width / 2.0, -focal * z / x + height / 2.0))
    xs, ys = zip(*points)
    return (min(xs), min(ys), max(xs), max(ys))


def instance_ids(image):
    """Actor ids per pixel under the two plausible encodings of the instance camera (G, B channels)."""
    raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4).astype(np.int32)
    blue, green = raw[:, :, 0], raw[:, :, 1]
    return {"g+256b": green + 256 * blue, "b+256g": blue + 256 * green}


def run_quality(args, quality, out_dir):
    record = {"quality": quality}
    process = None
    if not args.no_start:
        record["server_args"] = server_args(quality, args.port, args.offscreen)
        record["gpu_before_server_mib"] = gpu_memory_mib()[0]
        process = start_server(args.server, record["server_args"])
    actors, sensors = [], []
    world = original = None
    try:
        started = time.monotonic()
        client = connect(args.port, args.connect_timeout)
        record["connect_s"] = round(time.monotonic() - started, 1)
        record["client_version"] = client.get_client_version()
        record["server_version"] = client.get_server_version()
        record["maps"] = sorted(m.split("/")[-1] for m in client.get_available_maps())
        world = client.load_world(args.map) if args.map else client.get_world()
        record["map"] = world.get_map().name.split("/")[-1]
        time.sleep(5.0)
        record["gpu_server_idle_mib"] = gpu_memory_mib()[0]

        original = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.hz
        world.apply_settings(settings)
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(args.seed)

        library = world.get_blueprint_library()
        vehicles = library.filter("vehicle.*")
        kinds = {}
        for bp in vehicles:
            kind = bp.get_attribute("base_type").as_str() if bp.has_attribute("base_type") else "unknown"
            kinds[kind] = kinds.get(kind, 0) + 1
        record["vehicle_blueprints"] = {"total": len(vehicles), "by_base_type": kinds}
        record["drone_like_blueprints"] = [bp.id for bp in library if "drone" in bp.id or "uav" in bp.id]

        rng = random.Random(args.seed)
        cars = [bp for bp in vehicles if bp.has_attribute("base_type") and bp.get_attribute("base_type").as_str() == "car"]
        if not cars:
            cars = [bp for bp in vehicles if bp.get_attribute("number_of_wheels").as_int() == 4]
        spawn_points = world.get_map().get_spawn_points()
        rng.shuffle(spawn_points)
        record["spawn_points"] = len(spawn_points)
        batch = []
        for sp in spawn_points[:args.vehicles]:
            bp = rng.choice(cars)
            if bp.has_attribute("color"):
                bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
            batch.append(carla.command.SpawnActor(bp, sp).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
        for result in client.apply_batch_sync(batch, True):
            if not result.error:
                actors.append(result.actor_id)
        record["vehicles_spawned"] = len(actors)
        world.tick()
        target = world.get_actor(actors[0])
        record["target"] = {"id": target.id, "type_id": target.type_id, "color": target.attributes.get("color")}

        pose = drone_pose(target, args.distance, args.height)
        fpv, fpv_frames = camera(world, "sensor.camera.rgb", args.width, args.height_px, args.fov, pose)
        chase, chase_frames = camera(world, "sensor.camera.rgb", args.width, args.height_px, args.fov,
                                     chase_pose(pose, target.get_location(), 12.0, 6.0))
        sensors += [fpv, chase]
        streams = [(fpv, fpv_frames), (chase, chase_frames)]
        dt = settings.fixed_delta_seconds

        def step(save_as=None, instance=None):
            pose = drone_pose(target, args.distance, args.height)
            chase_t = chase_pose(pose, target.get_location(), 12.0, 6.0)
            fpv.set_transform(pose)
            chase.set_transform(chase_t)
            if instance is not None:
                instance[0].set_transform(pose)
            # Stand-in for the drone body in the chase view, just behind the drone camera so that camera
            # does not see it (debug shapes show up in every camera).
            forward = pose.get_forward_vector()
            body = pose.location - carla.Location(forward.x, forward.y, forward.z)
            world.debug.draw_box(carla.BoundingBox(body, carla.Vector3D(0.4, 0.4, 0.15)), pose.rotation,
                                 0.1, carla.Color(255, 140, 0), 2.0 * dt)
            frame = world.tick()
            images = [frame_at(q, frame) for _, q in streams]
            seg = frame_at(instance[1], frame) if instance is not None else None
            if save_as:
                save_views(out_dir / save_as, images, target, pose, args, seg, record)
            return images

        def measure(name, ticks, instance=None):
            gpu = []
            started = time.monotonic()
            for i in range(ticks):
                step(instance=instance)
                if i % 50 == 0:
                    gpu.append(gpu_memory_mib()[0])
            wall = time.monotonic() - started
            record[name] = {"ticks": ticks, "wall_s": round(wall, 2), "fps": round(ticks / wall, 1),
                            "sim_speed": round(ticks * dt / wall, 2), "gpu_used_max_mib": max(gpu),
                            "gpu_total_mib": gpu_memory_mib()[1]}

        for _ in range(args.warmup):
            step()
        measure("two_cameras", args.ticks)
        step(save_as="views_a.png")

        seg_cam = camera(world, "sensor.camera.instance_segmentation", args.width, args.height_px, args.fov,
                         drone_pose(target, args.distance, args.height))
        sensors.append(seg_cam[0])
        for _ in range(10):
            step(instance=seg_cam)
        measure("with_instance_camera", args.ticks, instance=seg_cam)
        step(save_as="views_b.png", instance=seg_cam)
    finally:
        for sensor in sensors:
            sensor.stop()
            sensor.destroy()
        if world is not None and actors:
            client.apply_batch_sync([carla.command.DestroyActor(a) for a in actors], True)
        if world is not None and original is not None:
            original.synchronous_mode = False
            world.apply_settings(original)
        if process is not None:
            stop_server(process)
    return record


def save_views(path, images, target, pose, args, seg, record):
    fpv, chase = (bgr(i) for i in images)
    box = project_box(target, pose, args.width, args.height_px, args.fov)
    if box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(fpv, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(fpv, f"target {target.id}", (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    panels = [fpv, chase]
    if seg is not None:
        ids = instance_ids(seg)
        counts = {name: int((grid == target.id).sum()) for name, grid in ids.items()}
        record.setdefault("instance_check", []).append({"target_pixels": counts, "projected_box": box})
        mask = np.zeros_like(fpv)
        best = max(counts, key=counts.get)
        mask[ids[best] == target.id] = (0, 0, 255)
        panels.append(cv2.addWeighted(bgr(seg), 0.5, mask, 0.5, 0))
    for panel, label in zip(panels, ["drone (1st person)", "chase (3rd person)", "instance: target in red"]):
        cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite(str(path), np.hstack(panels))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quality", nargs="+", default=["Low", "Epic"], choices=["Low", "Epic"])
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--no-start", action="store_true", help="use a simulator that is already running")
    parser.add_argument("--offscreen", action="store_true", help="start the simulator without its window")
    parser.add_argument("--label", default="", help="suffix for the output folder of each quality")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--connect-timeout", type=float, default=600.0)
    parser.add_argument("--map", default=None, help="map to load, e.g. Town03 (default: the one the simulator opens)")
    parser.add_argument("--vehicles", type=int, default=40)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--distance", type=float, default=20.0, help="drone distance behind the target (m)")
    parser.add_argument("--height", type=float, default=15.0, help="drone height above the target (m)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height-px", type=int, default=540)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    run_dir = OUTPUT / datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    for quality in args.quality:
        out_dir = run_dir / (quality.lower() + (f"_{args.label}" if args.label else ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        record = run_quality(args, quality, out_dir)
        (out_dir / "summary.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "maps"}, ensure_ascii=False))
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"results: {run_dir}")


if __name__ == "__main__":
    main()
