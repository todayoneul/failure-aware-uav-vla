"""CARLA-Air probe 4: a CARLA instance-segmentation camera attached to the AirSim drone actor (check 3),
then whether Town05 loads with the drone still present."""
import json
from pathlib import Path
import queue
import time

import airsim
import carla
import numpy as np

res = {}
client = carla.Client("localhost", 2000)
client.set_timeout(60.0)
world = client.get_world()
ac = airsim.MultirotorClient(port=41451, timeout_value=15)
ac.confirmConnection()
ac.enableApiControl(True)
ac.hoverAsync().join()

drone = [a for a in world.get_actors() if a.type_id == "airsim.drone"][0]
dl = drone.get_location()
# A car on the road nearest the point under the drone.
wp = world.get_map().get_waypoint(carla.Location(dl.x, dl.y, 0.0))
car = world.try_spawn_actor(world.get_blueprint_library().find("vehicle.tesla.model3"),
                            carla.Transform(wp.transform.location + carla.Location(z=0.5), wp.transform.rotation))
res["car"] = None if car is None else car.id
# Bring the drone 25 m above the car (AirSim NED = CARLA + offset, z flipped).
off = (172.19658, -183.85911, 27.45434)
cl = wp.transform.location
ac.moveToPositionAsync(cl.x + off[0], cl.y + off[1], -(cl.z + 25.0) + off[2], 8, timeout_sec=40).join()
ac.hoverAsync().join()
time.sleep(2.0)
dl = drone.get_location()
res["drone_minus_car_m"] = [round(dl.x - cl.x, 2), round(dl.y - cl.y, 2), round(dl.z - cl.z, 2)]

bp = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
for k, v in (("image_size_x", "640"), ("image_size_y", "360"), ("fov", "90")):
    bp.set_attribute(k, v)
q = queue.Queue()
cam = world.spawn_actor(bp, carla.Transform(carla.Location(z=-0.5), carla.Rotation(pitch=-90.0)), attach_to=drone)
cam.listen(q.put)
try:
    img = q.get(timeout=10.0)
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    ids = a[:, :, 1].astype(np.int32) + 256 * a[:, :, 0].astype(np.int32)  # BGRA: G + 256 * B
    uniq, counts = np.unique(ids, return_counts=True)
    res["camera_parent"] = cam.parent.type_id if cam.parent else None
    res["car_pixels"] = int(counts[uniq == car.id][0]) if car and car.id in uniq else 0
    res["top_ids"] = {int(u): int(c) for u, c in sorted(zip(uniq, counts), key=lambda x: -x[1])[:6]}
    ct = cam.get_transform().location
    res["camera_minus_drone_m"] = [round(ct.x - dl.x, 2), round(ct.y - dl.y, 2), round(ct.z - dl.z, 2)]
finally:
    cam.stop()
    cam.destroy()
    if car:
        car.destroy()
print(json.dumps(res, indent=1), flush=True)

# Town05: the command-line map argument was ignored, so try loading it from the client.
t = time.monotonic()
try:
    world = client.load_world("Town05")
    res["town05_load_s"] = round(time.monotonic() - t, 1)
    res["town05_map"] = world.get_map().name
    time.sleep(5.0)
    drones = [a for a in world.get_actors() if a.type_id == "airsim.drone"]
    res["town05_drone_actor"] = len(drones)
    ac2 = airsim.MultirotorClient(port=41451, timeout_value=15)
    ac2.confirmConnection()
    p = ac2.getMultirotorState().kinematics_estimated.position
    res["town05_airsim_ned"] = [round(p.x_val, 2), round(p.y_val, 2), round(p.z_val, 2)]
    if drones:
        l = drones[0].get_location()
        res["town05_offset"] = [round(p.x_val - l.x, 2), round(p.y_val - l.y, 2), round(p.z_val + l.z, 2)]
except Exception as e:  # noqa: BLE001
    res["town05_error"] = repr(e)
print(json.dumps(res, indent=1))
OUT = Path(__file__).resolve().parents[2] / "outputs" / "carla_air_check"
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / (Path(__file__).stem + ".json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
