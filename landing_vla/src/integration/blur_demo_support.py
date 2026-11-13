"""Minimal observation/control/telemetry hooks for the existing closed loop."""
import hashlib
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
from src.failures.control import default_control, read_control, write_control, apply_key
from src.failures.gaussian_blur import GaussianBlurFailure, SEVERITIES


class DemoQuit(Exception):
    pass


def publish_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.publish.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.01)


def publish_image(path, image):
    path = Path(path)
    temporary = path.with_name(path.stem+'.publish.png')
    if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise RuntimeError(f'Cannot publish image: {path.name}')
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.01)


class BlurDemoSession:
    def __init__(self, root, output, steps):
        self.root, self.output = Path(root), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.control_path = self.output/'control.json'
        self.control = default_control()
        if not self.control_path.exists():
            write_control(self.control_path, self.control)
        self.failure = GaussianBlurFailure()
        self.saved_blur = self.saved_restored = False
        self.last_chase = 0
        self.chase_ready = False
        self.telemetry = {'status': 'RUNNING', 'phase': 'Starting', 'step': 0,
                          'completed_steps': 0, 'max_steps': steps, 'failure': self.control,
                          'input_step': 0, 'input_files': {}}
        self.update()

    def update(self, **fields):
        self.telemetry.update(fields)
        self.telemetry['updated_at'] = time.time()
        self.telemetry['chase_ready'] = self.chase_ready
        publish_json(self.output/'telemetry.json', self.telemetry)

    def poll_control(self):
        try:
            self.control = read_control(self.control_path)
        except (OSError, ValueError) as error:
            self.update(control_warning=str(error))
        if self.control['quit']:
            raise DemoQuit('Exit requested; landing and disarming')
        return self.control.copy()

    def auto_toggle(self, step):
        state = apply_key(read_control(self.control_path), ord('b'))
        write_control(self.control_path, state)
        with (self.output/'control-events.jsonl').open('a') as file:
            file.write(json.dumps({'source': 'automatic live validation', 'key': 'B',
                                   'before_step': step, 'state': state})+'\n')

    def prepare_config(self):
        from scripts.demo_view_config import build_demo_robot
        directory = self.output/'sim_config'
        directory.mkdir(exist_ok=True)
        robot = json.loads((self.root/'configs/robot_quadrotor_fastphysics.jsonc').read_text())
        robot = build_demo_robot(robot)
        chase = next(sensor for sensor in robot['sensors'] if sensor['id'] == 'Chase')
        for settings in chase['capture-settings']:
            settings.update(width=640, height=360)
        (directory/'robot_quadrotor_fastphysics.jsonc').write_text(json.dumps(robot, indent=2))
        (directory/'scene_basic_drone.jsonc').write_text((self.root/'configs/scene_basic_drone.jsonc').read_text())
        return directory

    def chase_callback(self, _, message):
        if time.monotonic()-self.last_chase < .25:
            return
        from projectairsim.utils import unpack_image
        frame = unpack_image(message)
        if frame.shape == (360, 640, 3) and message['encoding'] == 'BGR':
            publish_image(self.output/'chase_latest.png', frame)
            self.last_chase = time.monotonic()
            self.chase_ready = True

    def inject(self, raw_frames, step, state, instruction):
        control = self.poll_control()
        self.failure.enabled = control['enabled']
        self.failure.set_severity(control['severity'])
        started = time.perf_counter()
        used = [self.failure.apply(image) for image in raw_frames]
        latency = (time.perf_counter()-started)*1000
        hashes = lambda frames: {name: hashlib.sha256(frame.tobytes()).hexdigest()
                                 for name, frame in zip(('front', 'down'), frames)}
        metadata = {'failure_enabled': control['enabled'], 'failure_type': 'gaussian_blur' if control['enabled'] else 'normal',
                    'severity': control['severity'], 'revision': control['revision'], 'target': 'all',
                    'kernel': SEVERITIES[control['severity']][0], 'sigma': SEVERITIES[control['severity']][1],
                    'blur_processing_ms': latency, 'raw_frame_sha256': hashes(raw_frames),
                    'used_frame_sha256': hashes(used)}
        slot = 'a' if step % 2 else 'b'
        input_files = {name: f'input_{slot}_{name}.png' for name in ('front','down')}
        for name, frame in zip(('front', 'down'), used):
            publish_image(self.output/input_files[name], frame)
        if control['enabled'] and not self.saved_blur:
            for name, raw, failed in zip(('front', 'down'), raw_frames, used):
                publish_image(self.output/f'normal_{name}.png', raw)
                publish_image(self.output/f'blur_{name}.png', failed)
            comparison = np.full((620, 560, 3), 245, dtype=np.uint8)
            cv2.putText(comparison, 'Original reference', (12, 28), 0, .6, (45,45,45), 1, cv2.LINE_AA)
            cv2.putText(comparison, 'Actual VLA input', (292, 28), 0, .6, (45,45,45), 1, cv2.LINE_AA)
            for row, (raw, failed, name) in enumerate(zip(raw_frames, used, ('Front', 'Down'))):
                y = 60+row*280
                cv2.putText(comparison, name, (12, y-8), 0, .5, (45,45,45), 1, cv2.LINE_AA)
                comparison[y:y+256, 12:268] = raw
                comparison[y:y+256, 292:548] = failed
            publish_image(self.output/'comparison.png', comparison)
            self.saved_blur = True
        elif not control['enabled']:
            prefix = 'restored' if self.saved_blur else 'normal'
            if prefix == 'normal' or not self.saved_restored:
                for name, frame in zip(('front', 'down'), used):
                    publish_image(self.output/f'{prefix}_{name}.png', frame)
                self.saved_restored = prefix == 'restored'
        self.update(phase='AeroVLA inference', step=step, input_step=step, input_files=input_files,
                    failure=metadata, state=state,
                    instruction=instruction, inference=None, clipped_action=None, input_verified=None)
        return used, metadata

    def verify_input(self, inference, failure):
        evidence = inference['input_evidence']
        if evidence['used_frame_sha256'] != failure['used_frame_sha256']:
            raise RuntimeError('Displayed/injected observations differ from actual model input')
        if not failure['failure_enabled'] and evidence['tensor_differs_from_reference']:
            raise RuntimeError('Blur OFF tensor differs from the raw reference')
        pixels_changed = failure['raw_frame_sha256'] != failure['used_frame_sha256']
        if failure['failure_enabled'] and pixels_changed and not evidence['tensor_differs_from_reference']:
            raise RuntimeError('Blur ON did not modify the actual processed tensor')
        self.update(inference=inference, input_verified=True)
