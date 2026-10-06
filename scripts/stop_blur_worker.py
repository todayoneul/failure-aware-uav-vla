"""Last-resort stop of an exact owned WSL worker, after graceful quit timed out."""
import argparse
import json
import os
import signal
import time
from pathlib import Path


def alive(pid):
    try:
        state = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[0]
        return state != 'Z'
    except FileNotFoundError:
        return False


def verify_owner(pid, expected_script, run_token, marker='--blur-demo'):
    if pid <= 1:
        raise PermissionError('Refusing invalid/system PID')
    args = (Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
    valid = len(args) > 1 and Path(args[1]).resolve() == Path(expected_script).resolve()
    valid = valid and marker in args and '--run-token' in args
    if valid:
        index = args.index('--run-token')
        valid = index+1 < len(args) and args[index+1] == run_token
    if not valid:
        if not alive(pid):
            return False
        raise PermissionError('PID does not belong to this exact demo run')
    return True


def stop_owned_worker(pid, expected_script, run_token, marker='--blur-demo'):
    if not alive(pid):
        return {'pid': pid, 'terminated': True, 'forced': False}
    try:
        owned = verify_owner(pid, expected_script, run_token, marker)
    except FileNotFoundError:
        if alive(pid):
            raise
        owned = False
    if not owned:
        return {'pid': pid, 'terminated': True, 'forced': False}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not alive(pid):
            break
        try:
            if not verify_owner(pid, expected_script, run_token, marker):
                break
        except FileNotFoundError:
            if not alive(pid):
                break
            raise
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic()+3
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(.05)
    if alive(pid):
        raise RuntimeError('Owned worker did not exit')
    return {'pid': pid, 'terminated': True, 'forced': True,
            'note': 'Graceful quit exceeded its deadline; launcher closes the owned simulator'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid-file', type=Path, required=True)
    parser.add_argument('--run-token', required=True)
    parser.add_argument('--mode',choices=('blur','mission'),default='blur')
    args = parser.parse_args()
    if args.pid_file.exists():
        pid = int(args.pid_file.read_text().strip())
        root=Path(__file__).resolve().parents[1]
        script=root/('src/mission/runner.py' if args.mode=='mission' else 'src/integration/closed_loop_runner.py')
        marker='--mission-demo' if args.mode=='mission' else '--blur-demo'
        print(json.dumps(stop_owned_worker(pid, script, args.run_token, marker)))
    else:
        print(json.dumps({'pid': None, 'terminated': True, 'forced': False, 'note': 'Python worker not started'}))
