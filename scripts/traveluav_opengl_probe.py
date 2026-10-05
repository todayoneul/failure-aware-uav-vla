"""One bounded, source-informed OpenGL attempt on the existing TravelUAV map."""
import json
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

workspace = Path('/mnt/c/Users/leegy/Desktop/drone')
output = workspace / 'outputs/rendering_retry'
executable = Path('/home/gyuhan/uav-vla-smoke/assets/travel-urban/BrushifyUrban/BrushifyUrban.sh')
env = os.environ.copy()
env.update(GALLIUM_DRIVER='d3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
command = [str(executable), '-opengl', '-ResX=640', '-ResY=480', '-windowed',
           '-NoSound', '-NoVSync', f'-settings={workspace}/scripts/gate2-settings.json',
           f'-AbsLog={output}/travel-opengl-unreal.log', '-stdout', '-FullStdOutLogOutput']
result = {'command': command, 'environment': {k:env[k] for k in
          ['GALLIUM_DRIVER','MESA_D3D12_DEFAULT_ADAPTER_NAME']}, 'samples':[],
          'rpc_port_open':False, 'camera_requests':0}
stop = threading.Event()

def monitor():
    while not stop.is_set():
        rows = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        used, utilization = (int(x.strip()) for x in rows.strip().split(','))
        rss = sum(int(row.split()[1]) for row in subprocess.check_output(
            ['ps','-e','-o','pgid=,rss='], text=True, stderr=subprocess.DEVNULL).splitlines()
            if int(row.split()[0]) == child.pid)
        result['samples'].append({'elapsed_s':time.monotonic()-started,
                                  'global_gpu_used_mib':used,'global_gpu_util_percent':utilization,
                                  'simulator_group_rss_mib':rss/1024})
        stop.wait(0.25)

try:
    with output.joinpath('travel-opengl-stdout.log').open('w') as log:
        child = subprocess.Popen(command, cwd=executable.parent, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        result['pid'] = child.pid
        worker = threading.Thread(target=monitor,daemon=True)
        worker.start()
        while time.monotonic()-started < 30:
            if child.poll() is not None:
                break
            try:
                with socket.create_connection(('127.0.0.1',41461),timeout=0.2):
                    result['rpc_port_open'] = True
                unreal_log = output.joinpath('travel-opengl-unreal.log').read_text(errors='replace')
                hardware = 'NVIDIA GeForce RTX 5070' in unreal_log and 'llvmpipe' not in unreal_log
                result['hardware_verified_before_rpc'] = hardware
                # Profile the observed fallback too; never count CPU RPC success as Gate 2 GPU PASS.
                if result['rpc_port_open']:
                    import airsim
                    import numpy as np
                    client = airsim.MultirotorClient(ip='127.0.0.1',port=41461,timeout_value=10)
                    assert client.ping()
                    result['ping'] = 'PASS'
                    state = client.getMultirotorState(vehicle_name='Drone_1')
                    pose = client.simGetVehiclePose(vehicle_name='Drone_1')
                    result['state'] = {'timestamp':state.timestamp,'position':vars(pose.position)}
                    result['renderer_for_profile'] = 'hardware' if hardware else 'CPU Vulkan llvmpipe fallback'
                    latencies = []
                    for i in range(30):
                        frame_started = time.perf_counter()
                        frames = client.simGetImages([airsim.ImageRequest('FrontCamera',airsim.ImageType.Scene,
                            pixels_as_float=False,compress=False)],vehicle_name='Drone_1')
                        latencies.append((time.perf_counter()-frame_started)*1000)
                        assert len(frames)==1 and frames[0].width==256 and len(frames[0].image_data_uint8)>0
                        result['camera_requests'] += 1
                        if (i+1)%10 == 0:
                            print(json.dumps({'frames':i+1,'latest_latency_ms':latencies[-1],
                                              'renderer':result['renderer_for_profile']}),flush=True)
                    result['camera_latency_ms'] = {'values':latencies,'mean':float(np.mean(latencies)),
                        'median':float(np.median(latencies)),'p95':float(np.percentile(latencies,95))}
                    client.client.close()
                break
            except (OSError,ConnectionError):
                pass
            time.sleep(0.25)
        result['exit_before_cleanup'] = child.poll()
        result['duration_s'] = time.monotonic()-started
finally:
    stop.set()
    if 'worker' in locals():
        worker.join(timeout=3)
    if 'child' in locals() and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait()
    result['sampled_group_rss_peak_mib'] = max((x['simulator_group_rss_mib'] for x in result['samples']),default=0)
    result['sampled_global_gpu_peak_mib'] = max((x['global_gpu_used_mib'] for x in result['samples']),default=0)
    output.joinpath('travel-opengl-launch.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2),flush=True)
