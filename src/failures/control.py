"""Small atomic file protocol between the Windows keys and the WSL runner."""
import json
import os
import time
import tempfile
import datetime
import uuid
from pathlib import Path


def default_control():
    return {'enabled': False, 'severity': 'medium', 'quit': False, 'revision': 0}


def validate_control(state):
    if not isinstance(state, dict):
        raise ValueError('Control must be an object')
    if type(state.get('enabled')) is not bool or type(state.get('quit')) is not bool:
        raise ValueError('Control enabled/quit must be bool')
    if state.get('severity') not in ('low', 'medium', 'high'):
        raise ValueError('Unknown control severity')
    if type(state.get('revision')) is not int or state['revision'] < 0:
        raise ValueError('Invalid control revision')
    return state


def read_control(path):
    return validate_control(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def write_control(path, state):
    validate_control(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as file:
        json.dump(state, file)
        temporary = Path(file.name)
    # Short Windows reader leases can briefly deny replacement.
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.01)


def apply_key(state, key):
    validate_control(state)
    updated = state.copy()
    if key in (ord('b'), ord('B')):
        updated['enabled'] = not state['enabled']
    elif key in (ord('1'), ord('2'), ord('3')):
        updated['severity'] = ('low', 'medium', 'high')[key-ord('1')]
    elif key in (27, ord('q'), ord('Q')):
        updated['quit'] = True
    else:
        return updated
    updated['revision'] += 1
    return updated


def initialize_run_output(path):
    """Archive only known current-run files; leave unrelated notes and directories alone."""
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = {'closed-loop.json','closed-loop-log.jsonl','closed-loop-wsl-resources.jsonl',
             'control.json','control-events.jsonl','telemetry.json','worker-pid.txt','run-info.json',
             'worker.log','worker-errors.log','viewer.log','viewer-errors.log',
             'windows-resources.jsonl','windows-latest.json','windows-latest.tmp','active-stage.txt',
             'chase_latest.png','comparison.png','launcher-cleanup.json'}
    names.update(f'{prefix}_{camera}.png' for prefix in ('normal','blur','restored','input_a','input_b')
                 for camera in ('front','down'))
    names.update(f'observer_{label}.png' for label in ('normal','blur','restored'))
    names.update(Path(name).stem+'.publish.png' for name in list(names) if name.endswith('.png'))
    names.add('telemetry.publish.tmp')
    old = [file for file in output.iterdir() if file.is_file() and
           (file.name in names or file.name.startswith('control.json.') and file.name.endswith('.tmp'))]
    archive = None
    if old:
        tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        archive = output/'runs'/tag
        archive.mkdir(parents=True)
        for file in old:
            # Single known file moves, never recursive cleanup or removal.
            file.replace(archive/file.name)
    (output/'run-info.json').write_text(json.dumps({'id': uuid.uuid4().hex,
        'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'previous_run_archive': str(archive.relative_to(output)) if archive else None}), encoding='utf-8')
    return archive
