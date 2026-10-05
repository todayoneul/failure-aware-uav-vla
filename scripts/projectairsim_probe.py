"""Disposable, model-free Project AirSim feasibility probe. No upstream edits."""
import argparse
import asyncio
import copy
import importlib.metadata
import json
import math
import platform
import time
import traceback
from pathlib import Path

import commentjson
import cv2
import numpy as np
import psutil
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.utils import unpack_image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "platform_final"
CONFIG = OUT / "sim_config"
SOURCE = OUT / "source" / "client" / "python" / "example_user_scripts" / "sim_config"


def prepare_config():
    CONFIG.mkdir(exist_ok=True)
    scene = commentjson.loads((SOURCE / "scene_basic_drone.jsonc").read_text())
    robot = commentjson.loads((SOURCE / "robot_quadrotor_fastphysics.jsonc").read_text())
    down = next(x for x in robot["sensors"] if x["id"] == "DownCamera")
    front = copy.deepcopy(down)
    front["id"] = "FrontCamera"
    front["origin"]["rpy-deg"] = "0 0 0"
    robot["sensors"].append(front)
    for sensor in robot["sensors"]:
        if sensor["type"] == "camera":
            sensor["capture-interval"] = 0.0333333
            sensor["capture-settings"] = [dict(sensor["capture-settings"][0], width=256, height=256,
                **{"image-type": 0, "capture-enabled": True, "streaming-enabled": sensor["id"] == "Chase"})]
    (CONFIG / "robot_quadrotor_fastphysics.jsonc").write_text(json.dumps(robot, indent=2))
    (CONFIG / "scene_basic_drone.jsonc").write_text(json.dumps(scene, indent=2))


def stats(values):
    return {"n": len(values), "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def pose_data(drone):
    return drone.get_ground_truth_kinematics()


def image(drone, sensor):
    before = time.perf_counter()
    message = drone.get_images(sensor, [0])[0]
    elapsed_ms = (time.perf_counter() - before) * 1000
    frame = unpack_image(message)
    if frame.shape[:2] != (256, 256) or not frame.size:
        raise RuntimeError(f"Invalid {sensor} frame {frame.shape}")
    return frame, message, elapsed_ms


async def main(args):
    prepare_config()
    client = ProjectAirSimClient()
    result = {"python": platform.python_version(), "client": importlib.metadata.version("projectairsim"),
              "started": time.time(), "camera_method": "Drone.get_images(sensor,[0]) request/response",
              "resolution": [256, 256], "commands": [], "raw_camera": {}, "errors": []}
    armed = False
    try:
        client.connect()
        world = World(client, "scene_basic_drone.jsonc", delay_after_load_sec=2, sim_config_path=str(CONFIG))
        drone = Drone(client, world, "Drone1")
        result["spawn_state"] = pose_data(drone)
        result["spawn_pose"] = drone.get_ground_truth_pose()
        result["sensors"] = drone.sensors
        print("SPAWN", json.dumps(result["spawn_state"]), flush=True)
        for sensor in ("FrontCamera", "DownCamera", "Chase"):
            frame, message, elapsed = image(drone, sensor)
            cv2.imwrite(str(OUT / f"{sensor}-first.png"), frame)
            result[f"{sensor}_first"] = {"latency_ms": elapsed,
                "metadata": {k: v for k, v in message.items() if k != "data"}, "std": float(frame.std())}
        if args.initial_only:
            return
        # Measure warm request/response latency without the GUI/flight overhead.
        for sensor in ("FrontCamera", "DownCamera"):
            for _ in range(5):
                image(drone, sensor)
            latencies, stamps = [], []
            start = time.perf_counter()
            for i in range(args.frames):
                frame, message, elapsed = image(drone, sensor)
                latencies.append(elapsed)
                stamps.append(message.get("time_stamp", message.get("timestamp", message.get("timestamp_ns"))))
            duration = time.perf_counter() - start
            result["raw_camera"][sensor] = {"latencies_ms": latencies, "timestamps": stamps}
            result[sensor + "_measurement"] = {**stats(latencies), "elapsed_s": duration,
                "effective_fps": args.frames / duration, "unique_timestamps": len(set(stamps))}
            cv2.imwrite(str(OUT / f"{sensor}-measured.png"), frame)
            print(sensor, result[sensor + "_measurement"], flush=True)
        if not args.flight:
            return

        name = "Project AirSim | front / down / chase | disposable demo"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 800, 590)
        cv2.moveWindow(name, 970, 40)
        process = psutil.Process(args.sim_pid) if args.sim_pid else None
        if process:
            process.cpu_percent()
        resources, trajectory, failure_events = [], [], []
        blur, drift_until = False, 0.0
        current = "spawn / camera benchmark"
        stop = False
        drift_requested = False
        auto_blur = auto_drift = False
        flight_start = time.perf_counter()

        async def display():
            nonlocal blur, drift_requested, stop, auto_blur, auto_drift
            while not stop:
                frames = [image(drone, s)[0] for s in ("FrontCamera", "DownCamera", "Chase")]
                raw_front = frames[0].copy()
                state = pose_data(drone)
                trajectory.append({"wall_elapsed_s": time.perf_counter() - flight_start, "command": current, "state": state})
                elapsed = time.perf_counter() - flight_start
                if args.auto_failures and current.startswith("forward") and not auto_blur:
                    blur = auto_blur = True
                    failure_events.append({"type": "visual_blur", "source": "config", "time_s": elapsed})
                if args.auto_failures and current == "hover" and not auto_drift:
                    drift_requested = auto_drift = True
                if blur:
                    frames[0] = cv2.GaussianBlur(frames[0], (21, 21), 6)
                    frames[1] = cv2.GaussianBlur(frames[1], (21, 21), 6)
                panel = np.zeros((500, 800, 3), dtype=np.uint8)
                for idx, frame in enumerate(frames):
                    panel[35:291, idx*266:idx*266+256] = frame[:, :, :3]
                    cv2.putText(panel, ("FRONT RGB", "DOWN RGB", "CHASE")[idx], (idx*266+5, 25), 0, .6, (230,230,230), 1)
                pos = state.get("pose", {}).get("position", {})
                ori = state.get("pose", {}).get("orientation", {})
                yaw = math.degrees(math.atan2(2*(ori.get("w",1)*ori.get("z",0)+ori.get("x",0)*ori.get("y",0)),
                    1-2*(ori.get("y",0)**2+ori.get("z",0)**2)))
                lines = [f"Command: {current}", f"Position NED: {pos}  Yaw: {yaw:.1f} deg",
                    f"Altitude relative to NED origin: {-pos.get('z',0):.2f} m",
                    "B: toggle blur | W: queue drift at hover | Esc: land and exit"]
                if blur: lines.append("FAILURE: VISUAL BLUR")
                if time.perf_counter() < drift_until: lines.append("FAILURE: CONTROL DRIFT")
                for i, line in enumerate(lines):
                    cv2.putText(panel, line, (8, 320+i*28), 0, .52, (50,190,255) if line.startswith("FAILURE") else (230,230,230), 1)
                cv2.imshow(name, panel)
                key = cv2.waitKey(1) & 0xff
                if key in (ord("b"), ord("B")):
                    blur = not blur
                    failure_events.append({"type":"visual_blur", "source":"keyboard", "time_s":elapsed, "active":blur})
                elif key in (ord("w"), ord("W")): drift_requested = True
                elif key == 27: stop = True
                if process:
                    mem = process.memory_info()
                    resources.append({"elapsed_s": elapsed, "rss_bytes": mem.rss, "private_bytes": getattr(mem,"private",None),
                        "cpu_percent_one_core": process.cpu_percent(), "system_ram_used_bytes": psutil.virtual_memory().used})
                if blur and not (OUT / "demo-blur.png").exists():
                    cv2.imwrite(str(OUT / "demo-blur.png"), panel)
                    cv2.imwrite(str(OUT / "blur-source-front.png"), raw_front)
                    cv2.imwrite(str(OUT / "blur-output-front.png"), frames[0])
                if time.perf_counter() < drift_until:
                    cv2.imwrite(str(OUT / "demo-control-drift.png"), panel)
                cv2.imwrite(str(OUT / "demo-latest.png"), panel)
                await asyncio.sleep(.03)

        async def command(label, operation, hold=0):
            nonlocal current
            current = label
            before = pose_data(drone)
            start = time.perf_counter()
            task = await operation
            response = await asyncio.wait_for(task, timeout=35)
            await asyncio.sleep(hold)
            after = pose_data(drone)
            result["commands"].append({"command": label, "response": response, "duration_s":time.perf_counter()-start,
                "before":before, "after":after, "landed_state":drone.get_landed_state()})
            print("COMMAND", label, response, json.dumps(after), flush=True)
            if stop: raise RuntimeError("Demo stop requested")

        result["enable_api_control"] = drone.enable_api_control()
        result["arm"] = drone.arm()
        armed = True
        viewer = asyncio.create_task(display())
        try:
            await command("takeoff", drone.takeoff_async(timeout_sec=20), 1)
            await command("up", drone.move_by_velocity_body_frame_async(0, 0, -.6, duration=2), 1)
            await command("forward", drone.move_by_velocity_body_frame_async(1, 0, 0, duration=2), 1)
            await command("yaw right", drone.rotate_by_yaw_rate_async(math.radians(30), duration=2), 1)
            await command("forward after yaw", drone.move_by_velocity_body_frame_async(1, 0, 0, duration=2), 1)
            await command("down", drone.move_by_velocity_body_frame_async(0, 0, .5, duration=1), 1)
            await command("hover", drone.hover_async(), 2)
            if drift_requested:
                drift_until = time.perf_counter()+3
                failure_events.append({"type":"control_drift", "source":"config_or_keyboard", "time_s":time.perf_counter()-flight_start})
                await command("temporary lateral/yaw disturbance", drone.move_by_velocity_body_frame_async(0, .7, 0, duration=2,
                    yaw_is_rate=True, yaw=math.radians(15)), 1)
                await command("hover after disturbance", drone.hover_async(), 2)
            await command("land", drone.land_async(timeout_sec=20), 1)
            result["final_landed_state"] = drone.get_landed_state()
            result["final_state"] = pose_data(drone)
            result["disarm"] = drone.disarm()
            armed = False
            await asyncio.sleep(2)
            result["landed_after_disarm"] = drone.get_landed_state()
            result["disable_api_control"] = drone.disable_api_control()
            current = "landed | demo complete"
            await asyncio.sleep(3)
        finally:
            stop = True
            await viewer
            result["trajectory"] = trajectory
            result["failure_events"] = failure_events
            result["resources"] = resources
            cv2.destroyAllWindows()
    except Exception:
        result["errors"].append(traceback.format_exc())
        print(result["errors"][-1], flush=True)
    finally:
        if armed:
            try:
                if drone.get_landed_state() != 0:
                    await asyncio.wait_for(await drone.land_async(timeout_sec=20), 25)
                drone.disarm()
                drone.disable_api_control()
            except Exception:
                result["cleanup_error"] = traceback.format_exc()
        client.disconnect()
        result["finished"] = time.time()
        (OUT / args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["errors"]: raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--flight", action="store_true")
    parser.add_argument("--auto-failures", action="store_true")
    parser.add_argument("--sim-pid", type=int)
    parser.add_argument("--output", default="projectairsim-result.json")
    asyncio.run(main(parser.parse_args()))
