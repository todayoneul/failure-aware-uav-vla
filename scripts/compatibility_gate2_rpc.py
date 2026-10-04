"""One camera request and read-only drone state/pose; never moves the vehicle."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True)
parser.add_argument('--port', type=int, default=41461)
parser.add_argument('--label', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
result = {'label': args.label, 'host': args.host, 'port': args.port, 'calls': {}}
args.output.mkdir(parents=True, exist_ok=True)

try:
    import airsim
    import cv2
    import numpy as np
    client = airsim.MultirotorClient(ip=args.host, port=args.port, timeout_value=10)
    start = time.perf_counter()
    assert client.ping()
    result['calls']['ping'] = {'status': 'PASS', 'latency_ms': (time.perf_counter()-start)*1000}
    result['server_version'] = client.getServerVersion()
    result['client_version'] = client.getClientVersion()
    result['vehicles'] = client.listVehicles()
    assert 'Drone_1' in result['vehicles'], result['vehicles']
    start = time.perf_counter()
    images = client.simGetImages([airsim.ImageRequest('FrontCamera', airsim.ImageType.Scene,
                                                    pixels_as_float=False, compress=False)],
                                 vehicle_name='Drone_1')
    latency = (time.perf_counter() - start) * 1000
    assert len(images) == 1 and images[0].width > 0 and images[0].height > 0
    response = images[0]
    pixels = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(
        response.height, response.width, 3)
    image_path = args.output / f'{args.label}-camera.png'
    assert cv2.imwrite(str(image_path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    result['calls']['simGetImages'] = {
        'status': 'PASS', 'latency_ms': latency, 'requests': 1,
        'width': response.width, 'height': response.height, 'bytes': len(response.image_data_uint8),
        'pixel_min': int(pixels.min()), 'pixel_max': int(pixels.max()),
        'pixel_std': float(pixels.std()), 'timestamp': int(response.time_stamp),
        'raw_sha256': hashlib.sha256(response.image_data_uint8).hexdigest(),
        'image': str(image_path), 'camera': 'FrontCamera', 'vehicle': 'Drone_1'}
    start = time.perf_counter()
    state = client.getMultirotorState(vehicle_name='Drone_1')
    result['calls']['getMultirotorState'] = {
        'status': 'PASS', 'latency_ms': (time.perf_counter()-start)*1000,
        'timestamp': int(state.timestamp), 'landed_state': int(state.landed_state),
        'position': vars(state.kinematics_estimated.position)}
    start = time.perf_counter()
    pose = client.simGetVehiclePose(vehicle_name='Drone_1')
    result['calls']['simGetVehiclePose'] = {
        'status': 'PASS', 'latency_ms': (time.perf_counter()-start)*1000,
        'position': vars(pose.position), 'orientation': vars(pose.orientation)}
    client.client.close()
    result['rpc'] = 'PASS'
except Exception:
    result['rpc'] = 'FAIL'
    result['full_traceback'] = traceback.format_exc()
finally:
    args.output.joinpath(f'{args.label}-rpc.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
raise SystemExit(0 if result['rpc'] == 'PASS' else 1)
