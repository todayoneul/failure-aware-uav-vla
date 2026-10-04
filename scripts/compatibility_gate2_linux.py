"""Bounded Linux simulator launch; capture renderer, resources and RPC readiness."""
import argparse
import json
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--executable', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
command = [str(args.executable), '-RenderOffscreen', '-vulkan', '-NoSound', '-NoVSync',
           '-ResX=640', '-ResY=480', '-windowed',
           '-settings=/mnt/c/Users/leegy/Desktop/drone/scripts/gate2-settings.json',
           f'-AbsLog={args.output}/wsl-travel-unreal.log', '-stdout', '-FullStdOutLogOutput']
result = {'command': command, 'samples': [], 'rpc_port_open': False}
stop = threading.Event()

def monitor_resources():
    while not stop.is_set():
        rss = 0
        for line in subprocess.check_output(['ps', '-e', '-o', 'pid=,pgid=,rss='],
                                             text=True, stderr=subprocess.DEVNULL).splitlines():
            pid, group, memory = map(int, line.split())
            if group == child.pid:
                rss += memory
        gpu = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()
        result['samples'].append({'seconds': time.monotonic()-start,
                                  'process_group_rss_mib': rss/1024, 'global_gpu_used_mib': int(gpu)})
        stop.wait(0.25)

try:
    with args.output.joinpath('wsl-travel-stdout.log').open('w') as log:
        child = subprocess.Popen(command, cwd=args.executable.parent, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        result['pid'] = child.pid
        start = time.monotonic()
        worker = threading.Thread(target=monitor_resources, daemon=True)
        worker.start()
        while time.monotonic() - start < 45:
            if child.poll() is not None:
                break
            try:
                with socket.create_connection(('127.0.0.1', 41461), timeout=0.2):
                    result['rpc_port_open'] = True
                rpc = subprocess.run([
                    '/home/gyuhan/uav-vla-smoke/gate2/bin/python',
                    '/mnt/c/Users/leegy/Desktop/drone/scripts/compatibility_gate2_rpc.py',
                    '--host', '127.0.0.1', '--label', 'wsl-travel', '--output', str(args.output)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=35)
                result['rpc_probe_exit'] = rpc.returncode
                break
            except (ConnectionError, OSError):
                pass
            time.sleep(0.5)
        result['exit_before_cleanup'] = child.poll()
finally:
    stop.set()
    if 'worker' in locals():
        worker.join(timeout=3)
    if 'child' in locals() and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    result['peak_process_group_rss_mib'] = max((x['process_group_rss_mib'] for x in result['samples']), default=0)
    args.output.joinpath('wsl-travel-launch.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'samples'}, indent=2))
