"""File-based launcher controls, compatible with Windows PowerShell 5.1 and 7."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.failures.control import initialize_run_output, default_control, read_control, write_control, apply_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('init', 'quit'), required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/failure_demo'))
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--mode', choices=('blur','mission'), default='blur')
    args = parser.parse_args()
    if not 1 <= args.steps <= 60:
        parser.error('--steps must be within 1..60')
    output = args.output if args.output.is_absolute() else ROOT/args.output
    path = output/'control.json'
    if args.action == 'init':
        from src.integration.blur_demo_support import BlurDemoSession
        initialize_run_output(output)
        if args.mode=='mission':
            from src.mission.control import default_mission_control
            write_control(path, default_mission_control())
        else:write_control(path, default_control())
        session=BlurDemoSession(ROOT, output, args.steps)
        if args.mode=='mission':
            session.update(mission={'state':'IDLE','target':None,'errors':None})
        print('Control initialized: blur OFF, severity MEDIUM', flush=True)
    else:
        state = read_control(path)
        if not state['quit']:
            state = apply_key(state, ord('q'))
            write_control(path, state)
            with (output/'control-events.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps({'source': 'launcher', 'key': ord('q'),
                                      'state': state, 'epoch': time.time()})+'\n')


if __name__ == '__main__':
    main()
