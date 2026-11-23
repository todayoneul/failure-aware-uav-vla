"""The CARLA side of Phase 2: start and stop the simulator, connect, run it in synchronous mode, and read cameras.

Everything here talks to a running simulator except `server_args`, `look_at`, `image_box` and `instance_ids`.
"""
import contextlib
import math
import queue
import socket
import subprocess
import time
from pathlib import Path

import carla
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER = ROOT / "assets" / "carla-0.9.16" / "CarlaUE4.exe"


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


def port_in_use(port, host="localhost"):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_and_connect(exe, args, port, timeout_s, tries=2):
    """Start the simulator and connect to it, starting it again if it does not answer within `timeout_s`.

    Three times in about twenty starts the simulator never answered: it listened on its ports and rendered, but no
    client (not even a fresh one) got an answer within 10 minutes. Started again it answered within seconds. The
    cause was not found."""
    if port_in_use(port):
        raise RuntimeError(f"something already listens on port {port}: stop the running simulator first")
    for attempt in range(tries):
        process = start_server(exe, args)
        try:
            return process, connect(port, timeout_s)
        except RuntimeError:
            stop_server(process)
            if attempt == tries - 1:
                raise


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


@contextlib.contextmanager
def session(map_name, hz=20.0, quality="Low", start=True, offscreen=False, port=2000, tm_port=8000, seed=0,
            server=DEFAULT_SERVER, connect_timeout=120.0):
    """Yield (client, world, traffic_manager) with `map_name` loaded in synchronous mode at `hz`.

    Starts the simulator first unless `start` is false, and on the way out destroys every vehicle and sensor,
    switches synchronous mode off and stops the simulator it started."""
    if start:
        process, client = start_and_connect(server, server_args(quality, port, offscreen), port, connect_timeout)
    else:
        process, client = None, connect(port, connect_timeout)
    world = None
    try:
        client.set_timeout(60.0)
        world = client.get_world()
        if world.get_map().name.split("/")[-1] != map_name:
            world = client.load_world(map_name)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / hz
        world.apply_settings(settings)
        tm = client.get_trafficmanager(tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(seed)
        yield client, world, tm
    finally:
        if world is not None:
            clear(client, world)
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        if process is not None:
            stop_server(process)


def clear(client, world):
    """Destroy every vehicle and sensor in the world."""
    actors = world.get_actors()
    sensors = list(actors.filter("sensor.*"))
    for sensor in sensors:
        if sensor.is_listening():
            sensor.stop()
    doomed = sensors + list(actors.filter("vehicle.*"))
    client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in doomed], True)
    world.tick()


def look_at(eye, target):
    """Rotation that points a camera at eye towards target (CARLA: x forward, y right, z up, degrees)."""
    d = target - eye
    yaw = math.degrees(math.atan2(d.y, d.x))
    pitch = math.degrees(math.atan2(d.z, math.hypot(d.x, d.y)))
    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def camera(world, kind, width, height, fov, transform):
    """Spawn a camera and return (sensor, queue of its images)."""
    bp = world.get_blueprint_library().find(kind)
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    sensor = world.spawn_actor(bp, transform)
    frames = queue.Queue()
    sensor.listen(frames.put)
    return sensor, frames


def frame_at(frames, frame, timeout=10.0):
    """The image of simulator frame `frame` from a camera queue (older images are dropped)."""
    while True:
        image = frames.get(timeout=timeout)
        if image.frame == frame:
            return image


def bgr(image):
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3].copy()


def instance_ids(image):
    """Actor id of every pixel of an instance segmentation image (CARLA 0.9.16: G + 256 * B, see docs/setup.md 4.4)."""
    raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4).astype(np.int32)
    return raw[:, :, 1] + 256 * raw[:, :, 0]


def semantic_tags(image):
    """Semantic tag of every pixel of an instance segmentation image (its red channel)."""
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, 2].copy()


def image_box(world_to_camera, vertices, width, height, fov):
    """Pixel box (x0, y0, x1, y1) of 3D points (N x 3, world) seen by a camera, or None if any is behind it.

    `world_to_camera` is the camera transform's inverse matrix (4 x 4)."""
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    points = np.c_[np.asarray(vertices, dtype=float), np.ones(len(vertices))] @ np.asarray(world_to_camera).T
    forward, right, up = points[:, 0], points[:, 1], points[:, 2]
    if (forward <= 0.1).any():
        return None
    # CARLA camera axes (forward, right, up) to image axes.
    xs, ys = focal * right / forward + width / 2.0, -focal * up / forward + height / 2.0
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
