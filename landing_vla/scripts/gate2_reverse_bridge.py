"""One local read-only RPC probe through a Windows-initiated TCP connection."""
import socket
import subprocess
import sys
import threading
from pathlib import Path

evidence = Path(__file__).resolve().parents[1] / 'outputs/compatibility'
remote_listener = socket.socket()
remote_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
remote_listener.bind(('0.0.0.0', 41500))
remote_listener.listen(1)
remote_listener.settimeout(45)
local_listener = socket.socket()
local_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
local_listener.bind(('127.0.0.1', 41501))
local_listener.listen(1)
local_listener.settimeout(45)
print('Reverse bridge listeners ready', flush=True)
remote, _ = remote_listener.accept()
remote_listener.close()

def pump(source, target):
    try:
        while True:
            data = source.recv(65536)
            if not data:
                break
            target.sendall(data)
    except OSError:
        pass
    finally:
        try:
            target.shutdown(socket.SHUT_WR)
        except OSError:
            pass

def connect_local():
    local, _ = local_listener.accept()
    local_listener.close()
    first = threading.Thread(target=pump, args=(local, remote), daemon=True)
    second = threading.Thread(target=pump, args=(remote, local), daemon=True)
    first.start()
    second.start()
    first.join(30)
    second.join(2)
    local.close()
    remote.close()

worker = threading.Thread(target=connect_local, daemon=True)
worker.start()
probe = subprocess.run([
    sys.executable,
    str(Path(__file__).resolve().parent / 'compatibility_gate2_rpc.py'),
    '--host', '127.0.0.1', '--port', '41501', '--label', 'windows-blocks-bridge',
    '--output', str(evidence)], timeout=30)
worker.join(3)
raise SystemExit(probe.returncode)
