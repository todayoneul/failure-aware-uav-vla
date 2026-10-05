"""Chase-only framing; never mutate the baseline vehicle or AI cameras."""
import copy
import math

VIEW_PROFILES = {
    "close": {"back_m": 1.9, "height_m": 0.55, "fov_deg": 60},
    "medium": {"back_m": 2.8, "height_m": 0.75, "fov_deg": 60},
    "elevated": {"back_m": 2.4, "height_m": 1.0, "fov_deg": 60},
}


def camera_pose(profile):
    pitch = -math.atan2(profile["height_m"], profile["back_m"])
    return {"translation": {"x": -profile["back_m"], "y": 0, "z": -profile["height_m"]},
            "rotation": {"w": math.cos(pitch / 2), "x": 0,
                         "y": math.sin(pitch / 2), "z": 0}}


def build_demo_robot(original):
    robot = copy.deepcopy(original)
    chase = next(sensor for sensor in robot["sensors"] if sensor["id"] == "Chase")
    profile = VIEW_PROFILES["close"]
    chase["origin"] = {"xyz": f'-{profile["back_m"]} 0 -{profile["height_m"]}',
                       "rpy-deg": f'0 {math.degrees(-math.atan2(profile["height_m"], profile["back_m"]))} 0'}
    for capture in chase["capture-settings"]:
        capture.update(width=960, height=540, **{"fov-degrees": profile["fov_deg"]})
    return robot
