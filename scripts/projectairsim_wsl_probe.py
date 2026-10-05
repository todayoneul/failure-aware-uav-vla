"""Actual Project AirSim Python client, run inside isolated WSL runtime."""
import asyncio
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts"))
from projectairsim_probe import prepare_config, CONFIG
from projectairsim import ProjectAirSimClient, World, Drone
from projectairsim.utils import unpack_image
import cv2

OUT = ROOT / "outputs" / "integration"

async def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prepare_config()
    client = ProjectAirSimClient(address="127.0.0.1",port_topics=18989,port_services=18990)
    result={"route":"transparent reverse TCP; WSL NNG client -> Windows Project AirSim", "errors":[],"latencies_ms":{}}
    drone=None
    try:
        start=time.perf_counter()
        client.connect()
        result["connect_ms"]=(time.perf_counter()-start)*1000
        world=World(client,"scene_basic_drone.jsonc",delay_after_load_sec=2,sim_config_path=str(CONFIG))
        drone=Drone(client,world,"Drone1")
        for sensor,name in (("FrontCamera","front"),("DownCamera","down")):
            start=time.perf_counter()
            msg=drone.get_images(sensor,[0])[0]
            result["latencies_ms"][name]=(time.perf_counter()-start)*1000
            frame=unpack_image(msg)
            assert frame.shape==(256,256,3)
            path=OUT/f"{name}_communication.png"
            cv2.imwrite(str(path),frame)
            result[name]={"shape":list(frame.shape),"encoding":msg["encoding"],"time_stamp":msg["time_stamp"],
                "path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"std":float(frame.std())}
        start=time.perf_counter()
        result["state_before"]=drone.get_ground_truth_kinematics()
        result["latencies_ms"]["state"]=(time.perf_counter()-start)*1000
        result["enable_api_control"]=drone.enable_api_control()
        result["arm"]=drone.arm()
        start=time.perf_counter()
        result["takeoff_return"]=await (await drone.takeoff_async(timeout_sec=20))
        result["takeoff_wall_ms"]=(time.perf_counter()-start)*1000
        result["state_after_takeoff"]=drone.get_ground_truth_kinematics()
        result["takeoff_landed_state"]=drone.get_landed_state()
        before=drone.get_ground_truth_kinematics()
        start=time.perf_counter()
        result["forward_return"]=await (await drone.move_by_velocity_body_frame_async(.5,0,0,duration=1))
        result["forward_wall_ms"]=(time.perf_counter()-start)*1000
        await (await drone.hover_async())
        await asyncio.sleep(.5)
        result["state_before_forward"]=before
        result["state_after"]=drone.get_ground_truth_kinematics()
        result["land_return"]=await (await drone.land_async(timeout_sec=20))
        drone.disarm()
        await asyncio.sleep(2)
        result["final_landed_state"]=drone.get_landed_state()
        drone.disable_api_control()
    except Exception:
        result["errors"].append(traceback.format_exc())
        print(result["errors"][-1],flush=True)
    finally:
        if drone:
            try:
                drone.disarm()
                drone.disable_api_control()
            except Exception: pass
        client.disconnect()
        (OUT/"communication.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if result["errors"]: raise SystemExit(1)

asyncio.run(main())
