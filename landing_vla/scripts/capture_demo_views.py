"""Real Blocks camera framing and bounded scripted flight; no model or failures."""
import argparse
import asyncio
import importlib.metadata
import json
import math
import platform
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.types import Pose
from projectairsim.utils import unpack_image
from demo_view_config import VIEW_PROFILES, build_demo_robot, camera_pose

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/demo_views"
CONFIG = OUT / "sim_config"
WINDOW = "UAV observation | Chase + camera inputs | Scripted flight"


def prepare_demo_config():
    CONFIG.mkdir(parents=True, exist_ok=True)
    robot = json.loads((ROOT / "configs/robot_quadrotor_fastphysics.jsonc").read_text())
    (CONFIG / "robot_quadrotor_fastphysics.jsonc").write_text(json.dumps(build_demo_robot(robot), indent=2))
    (CONFIG / "scene_basic_drone.jsonc").write_text((ROOT / "configs/scene_basic_drone.jsonc").read_text())


def image(drone, camera):
    message = drone.get_images(camera, [0])[0]
    frame = unpack_image(message)
    expected = (540, 960, 3) if camera == "Chase" else (256, 256, 3)
    if frame.shape != expected or message["encoding"] != "BGR":
        raise RuntimeError(f"Camera contract failed: {camera}, {frame.shape}")
    return frame, message["time_stamp"]


def panel(chase, front, down, state, phase, ground_z):
    canvas = np.full((710, 1224, 3), (248, 245, 241), dtype=np.uint8)
    ink, accent = (58, 42, 26), (180, 105, 35)
    cv2.putText(canvas, "Project AirSim / UAV observation", (20, 35), 0, .8, ink, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Scripted flight - model inference is off", (20, 61), 0, .48, ink, 1, cv2.LINE_AA)
    canvas[85:625, :960] = chase
    for frame, label, top in ((front, "Front / sees ahead", 85), (down, "Down / sees below", 355)):
        cv2.putText(canvas, label, (974, top-8), 0, .5, ink, 1, cv2.LINE_AA)
        canvas[top:top+248, 976:1224] = cv2.resize(frame, (248, 248))
    p, q = state["pose"]["position"], state["pose"]["orientation"]
    yaw = math.degrees(math.atan2(2*(q["w"]*q["z"]+q["x"]*q["y"]), 1-2*(q["y"]**2+q["z"]**2)))
    cv2.putText(canvas, f"Command: {phase}", (20, 656), 0, .66, accent, 2, cv2.LINE_AA)
    cv2.putText(canvas, f'Clearance {ground_z-p["z"]:.2f} m   Heading {yaw:.1f} deg   Position {p["x"]:.2f}, {p["y"]:.2f}, {p["z"]:.2f} m NED',
                (20, 692), 0, .55, ink, 1, cv2.LINE_AA)
    return canvas


async def main(args):
    prepare_demo_config()
    result = {"status": "RUNNING", "profiles": VIEW_PROFILES, "selected": args.view,
              "commands": [], "samples": [], "errors": [],
              "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                  for name in ("projectairsim", "opencv-python", "numpy", "pillow")}},
              "kind": "model-free scripted visualization; not AeroVLA navigation"}
    client, drone, viewer, armed = ProjectAirSimClient(), None, None, False
    stop, phase, recording, ground_z = False, "Preparing camera views", False, 0
    hero_saved = False
    latest_chase = None
    stream_count = 0
    started, collision_events, flight_stamp = time.perf_counter(), [], None

    def save():
        (OUT / "capture-result.json").write_text(json.dumps(result, indent=2))

    def check_flight():
        state = drone.get_ground_truth_kinematics()
        p = state["pose"]["position"]
        if not all(math.isfinite(v) for v in p.values()) or not -.05 <= ground_z-p["z"] <= 4.5:
            raise RuntimeError("Invalid state or clearance outside capture envelope")
        if flight_stamp is not None and any(event.get("time_stamp", 0) > flight_stamp for event in collision_events):
            raise RuntimeError("Collision during capture flight")
        return state

    def apply_view(name):
        profile = VIEW_PROFILES[name]
        pose_return = drone.set_camera_pose("Chase", Pose(camera_pose(profile)))
        fov_return = drone.set_field_of_view("Chase", 0, profile["fov_deg"])
        result.setdefault("camera_api", []).append({"view": name, "set_pose": pose_return, "set_fov": fov_return})
        if not pose_return or not fov_return:
            raise RuntimeError(f"Chase camera API rejected {name}")

    def chase_callback(_, message):
        nonlocal latest_chase, stream_count
        frame = unpack_image(message)
        if frame.shape == (540, 960, 3) and message["encoding"] == "BGR":
            latest_chase = (frame, message["time_stamp"])
            stream_count += 1

    async def display():
        nonlocal hero_saved
        front, down, preview_at = None, None, 0
        last_stamp = None
        while not stop:
            if latest_chase is None or latest_chase[1] == last_stamp:
                await asyncio.sleep(.01)
                continue
            chase, stamp = latest_chase
            last_stamp = stamp
            refresh_preview = front is None or time.perf_counter()-preview_at >= .6
            if refresh_preview:
                front, _ = image(drone, "FrontCamera")
                down, _ = image(drone, "DownCamera")
                preview_at = time.perf_counter()
            state = check_flight()
            rendered = panel(chase, front, down, state, phase, ground_z)
            if refresh_preview:
                cv2.imwrite(str(OUT / "observer_latest.png"), rendered, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if args.window and refresh_preview:
                cv2.imshow(WINDOW, rendered)
            if args.window:
                if cv2.waitKey(1) & 0xff == 27:
                    raise RuntimeError("Capture stopped by Escape")
            if recording:
                index = len(result["samples"])
                path = OUT / "frames" / f"frame_{index:04d}.png"
                cv2.imwrite(str(path), chase, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                result["samples"].append({"elapsed_s": time.perf_counter()-started, "phase": phase,
                                          "camera_timestamp": stamp, "source": "Chase topic subscription",
                                          "state": state, "frame": path.name})
                if not hero_saved and phase == "Forward":
                    cv2.imwrite(str(OUT / "hero_drone.png"), chase)
                    cv2.imwrite(str(OUT / "observer_hero.png"), rendered)
                    hero_saved = True
            await asyncio.sleep(.02)

    async def command(label, operation, hold=0):
        nonlocal phase
        if viewer and viewer.done():
            viewer.result()
        phase = label
        before = check_flight()
        start = time.perf_counter()
        response = await asyncio.wait_for(await operation, timeout=30)
        await asyncio.sleep(hold)
        after = check_flight()
        result["commands"].append({"label": label, "return": response, "before": before, "after": after,
                                   "wall_s": time.perf_counter()-start})
        save()
        print(f"COMMAND {label} return={response}", flush=True)

    try:
        client.connect()
        world = World(client, "scene_basic_drone.jsonc", delay_after_load_sec=2, sim_config_path=str(CONFIG))
        drone = Drone(client, world, "Drone1")
        client.subscribe(drone.robot_info["collision_info"], lambda _, message: collision_events.append(message))
        ground = drone.get_ground_truth_kinematics()
        ground_z = ground["pose"]["position"]["z"]
        # Capture baseline origin/FOV at the same high resolution for a fair framing comparison.
        original = next(s for s in json.loads((ROOT / "configs/robot_quadrotor_fastphysics.jsonc").read_text())["sensors"] if s["id"] == "Chase")
        drone.enable_api_control(); drone.arm(); armed = True
        await command("Takeoff", drone.takeoff_async(timeout_sec=20), .4)
        await command("Hover for camera comparison", drone.hover_async(), .3)
        air = drone.get_ground_truth_kinematics()
        result["takeoff_criterion"] = {"actual_climb_m": ground_z-air["pose"]["position"]["z"],
                                      "landed_state": drone.get_landed_state()}
        if result["takeoff_criterion"]["actual_climb_m"] < .5 or result["takeoff_criterion"]["landed_state"] == 0:
            raise RuntimeError("Actual takeoff failed")
        flight_stamp = air["time_stamp"]
        # Baseline has pitch -11.46deg, not atan(1/10); reproduce that exactly.
        pitch = math.radians(float(original["origin"]["rpy-deg"].split()[1]))
        baseline_pose = camera_pose({"back_m": 10, "height_m": 1})
        baseline_pose["rotation"].update(w=math.cos(pitch/2), y=math.sin(pitch/2))
        drone.set_camera_pose("Chase", Pose(baseline_pose)); drone.set_field_of_view("Chase", 0, 90)
        await asyncio.sleep(.2)
        cv2.imwrite(str(OUT / "chase_before.png"), image(drone, "Chase")[0])
        for name in VIEW_PROFILES:
            apply_view(name)
            await asyncio.sleep(.2)
            cv2.imwrite(str(OUT / f"chase_{name}.png"), image(drone, "Chase")[0])
        apply_view(args.view)
        client.subscribe(drone.sensors["Chase"]["scene_camera"], chase_callback)
        (OUT / "frames").mkdir(exist_ok=True)
        if args.window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1224, 710)
            cv2.moveWindow(WINDOW, 20, 35)
        viewer = asyncio.create_task(display())
        recording = True
        await command("Hover", drone.hover_async(), .6)
        await command("Forward", drone.move_by_velocity_body_frame_async(.35, 0, 0, duration=2), .2)
        await command("Yaw right", drone.rotate_by_yaw_rate_async(math.radians(12), duration=2), .2)
        await command("Forward after yaw", drone.move_by_velocity_body_frame_async(.35, 0, 0, duration=2), .2)
        await command("Hover", drone.hover_async(), .6)
        recording = False
        result["recording_end_elapsed_s"] = time.perf_counter()-started
        if len(result["samples"]) < 2:
            raise RuntimeError("Chase stream did not provide sufficient actual frames")
        result["status"] = "PASS"
    except Exception:
        result["status"] = "FAIL"; result["errors"].append(traceback.format_exc())
        print(result["errors"][-1], flush=True)
    finally:
        stop = True
        if viewer:
            try:
                await viewer
            except Exception:
                result["status"] = "FAIL"; result["errors"].append(traceback.format_exc())
        if drone and armed:
            try:
                result["land_return"] = await asyncio.wait_for(await drone.land_async(timeout_sec=20), 25)
                drone.disarm(); await asyncio.sleep(2); result["final_landed_state"] = drone.get_landed_state()
                drone.disable_api_control()
            except Exception:
                result["cleanup_error"] = traceback.format_exc()
        client.disconnect(); cv2.destroyAllWindows()
        result["collision_events"] = collision_events
        result["chase_stream_messages"] = stream_count
        result["elapsed_s"] = time.perf_counter()-started
        save()
    print(f'CAPTURE {result["status"]} frames={len(result["samples"])}', flush=True)
    if result["status"] != "PASS" or result.get("cleanup_error"):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=VIEW_PROFILES, default="close")
    parser.add_argument("--window", action="store_true")
    asyncio.run(main(parser.parse_args()))
