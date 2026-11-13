"""Windows keys and readable model-I/O view; never opens a simulator client."""
import argparse
import json
import hashlib
import math
import sys
import textwrap
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.failures.control import default_control, read_control, write_control, apply_key

WINDOW = 'Gaussian Blur Demo | AeroVLA inputs and actions'
BUTTONS = [(780, 20, 250, 40, ord('b')),
           (780, 76, 76, 34, ord('1')), (866, 76, 76, 34, ord('2')), (952, 76, 76, 34, ord('3'))]


def control_key_for_click(x, y):
    for left, top, width, height, key in BUTTONS:
        if left <= x < left+width and top <= y < top+height:
            return key
    return -1


def read_input_pair(output, telemetry):
    files = telemetry.get('input_files') or {}
    expected = (telemetry.get('failure') or {}).get('used_frame_sha256') or {}
    if not all(name in files and name in expected for name in ('front','down')):
        return None
    pair = {}
    for name in ('front','down'):
        image = cv2.imread(str(Path(output)/files[name]))
        if image is None or image.shape != (256,256,3) or hashlib.sha256(image.tobytes()).hexdigest() != expected[name]:
            return None
        pair[name] = image
    return pair


def write_lines(canvas, lines, x, y, width=56, scale=.48, color=(55, 43, 31)):
    for line in lines:
        for wrapped in textwrap.wrap(str(line), width) or ['']:
            cv2.putText(canvas, wrapped, (x, y), 0, scale, color, 1, cv2.LINE_AA)
            y += 22
    return y


def draw_view(telemetry, control, images):
    canvas = np.full((960, 1120, 3), (248, 245, 241), dtype=np.uint8)
    ink = (55, 43, 31)
    failure = telemetry.get('failure') or default_control()
    active = failure.get('failure_enabled', failure.get('enabled', False))
    severity = failure.get('severity', 'medium')
    cv2.putText(canvas, 'Gaussian Blur / AeroVLA live', (20, 35), 0, .82, ink, 2, cv2.LINE_AA)
    cv2.putText(canvas, 'FAILURE: '+('GAUSSIAN BLUR' if active else 'NORMAL'), (20, 75), 0, .75,
                (20, 110, 210) if active else (80, 120, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, f'Severity {severity.upper()}  |  Decision {telemetry.get("step",0)} / {telemetry.get("max_steps",0)}',
                (20, 105), 0, .52, ink, 1, cv2.LINE_AA)
    for left, top, width, height, key in BUTTONS:
        requested = control['enabled'] if key == ord('b') else control['severity'] == ('low','medium','high')[key-ord('1')]
        cv2.rectangle(canvas, (left, top), (left+width, top+height), (185,115,40) if requested else (220,220,220), -1)
        label = ('B: request BLUR OFF' if control['enabled'] else 'B: request BLUR ON') if key == ord('b') else ('1 Low','2 Medium','3 High')[key-ord('1')]
        cv2.putText(canvas, label, (left+7, top+24), 0, .45, (255,255,255) if requested else ink, 1, cv2.LINE_AA)
    pending = control['revision'] != failure.get('revision', 0)
    cv2.putText(canvas, 'Request applies at NEXT observation' if pending else 'B / 1 / 2 / 3 or click buttons | Q / Esc: exit',
                (20, 143), 0, .52, (20,110,210) if pending else ink, 1, cv2.LINE_AA)
    phase = telemetry.get('phase', 'Waiting for WSL runner')
    if telemetry.get('status') in ('FAIL', 'STOPPED', 'PASS'):
        phase = telemetry['status']+' / '+phase
    cv2.putText(canvas, phase, (20, 176), 0, .61, (175, 100, 30), 2, cv2.LINE_AA)
    for name, left, top, width, height, label in (
        ('chase', 20, 214, 720, 405, 'Drone / external view - never blurred'),
        ('front', 780, 214, 256, 256, f'Front / input step {telemetry.get("input_step",0)}'),
        ('down', 780, 504, 256, 256, f'Down / input step {telemetry.get("input_step",0)}')):
        cv2.putText(canvas, label, (left, top-10), 0, .51, ink, 1, cv2.LINE_AA)
        if images.get(name) is not None:
            canvas[top:top+height, left:left+width] = cv2.resize(images[name], (width, height))
        else:
            cv2.rectangle(canvas, (left, top), (left+width, top+height), (210,210,210), -1)
            cv2.putText(canvas, 'Waiting for camera / model load...', (left+10, top+height//2), 0, .48, ink, 1)
    state = telemetry.get('state')
    if state:
        p, q = state['position'], state['orientation']
        yaw = math.degrees(math.atan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)))
        write_lines(canvas, [f'Last checked state: height {telemetry.get("ground_z",0)-p[2]:.2f} m | heading {yaw:.1f} deg',
                             f'Position NED: {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f} m'], 20, 654, width=85, scale=.52)
    inference = telemetry.get('inference') or {}
    prompt = inference.get('prompt', telemetry.get('instruction', 'Model loading; existing cached checkpoints only'))
    write_lines(canvas, ['Prompt given to model: '+prompt.replace('<image>', '').replace('\n',' ').strip()],
                20, 710, width=85, scale=.48)
    parsed = inference.get('parsed_action') or {}
    clipped = telemetry.get('clipped_action') or {}
    raw = inference.get('raw_output', 'Waiting...').split('Action:')[-1].strip()
    write_lines(canvas, ['Model output / bins', raw[:100]], 20, 804, width=34, scale=.55)
    decoded = [f'Forward {parsed["fwd"]:.3f} m', f'Down {parsed["down"]:.3f} m', f'Yaw {math.degrees(parsed["yaw"]):.1f} deg'] if parsed else ['Waiting for generation']
    write_lines(canvas, ['Decoded model action']+decoded, 385, 804, width=35, scale=.54)
    if clipped:
        delta = clipped['displacement_ned']
        issued = [f'Forward {math.hypot(*delta[:2]):.3f} m', f'Down {delta[2]:.3f} m', f'Yaw {math.degrees(clipped["yaw_delta_rad"]):.1f} deg']
    else:
        issued = ['Waiting for safety bounds']
    write_lines(canvas, ['Actual bounded command']+issued, 780, 804, width=35, scale=.54)
    latency = inference.get('inference_ms')
    evidence = 'VERIFIED' if telemetry.get('input_verified') else 'pending'
    text = f'Actual input: {evidence} | Blur pair {failure.get("blur_processing_ms",0):.3f} ms'
    if latency is not None:
        text += f' | Inference {latency:.0f} ms'
    cv2.putText(canvas, text, (20, 935), 0, .48, ink, 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto-close', action='store_true')
    args = parser.parse_args()
    output = ROOT/'outputs/failure_demo'
    output.mkdir(parents=True, exist_ok=True)
    control_path = output/'control.json'
    control = read_control(control_path) if control_path.exists() else default_control()
    telemetry, images, stamps = {}, {}, {}
    completed_at = None
    saved = set()

    def dispatch(key, source):
        nonlocal control
        try:
            control = read_control(control_path)
        except (OSError, ValueError):
            pass
        updated = apply_key(control, key)
        if updated != control:
            write_control(control_path, updated)
            control = updated
            with (output/'control-events.jsonl').open('a') as file:
                file.write(json.dumps({'source': source, 'key': key, 'state': updated, 'epoch': time.time()})+'\n')

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1064, 912)
    cv2.moveWindow(WINDOW, 20, 20)
    cv2.setMouseCallback(WINDOW, lambda event,x,y,*_: dispatch(control_key_for_click(x,y), 'mouse') if event == cv2.EVENT_LBUTTONDOWN else None)
    try:
        while not control['quit']:
            try:
                control = read_control(control_path)
                telemetry = json.loads((output/'telemetry.json').read_text())
            except (OSError, ValueError):
                pass
            pair = read_input_pair(output, telemetry)
            if pair is None:
                images['front'] = images['down'] = None
            else:
                images.update(pair)
            sources = []
            if telemetry.get('chase_ready'):
                sources.append(('chase','chase_latest.png'))
            for name, filename in sources:
                path = output/filename
                try:
                    stamp = path.stat().st_mtime_ns
                    if stamp != stamps.get(name):
                        image = cv2.imread(str(path))
                        if image is not None:
                            expected = (telemetry.get('failure') or {}).get('used_frame_sha256',{}).get(name)
                            if expected and hashlib.sha256(image.tobytes()).hexdigest() != expected:
                                continue
                            images[name], stamps[name] = image, stamp
                except OSError:
                    pass
            rendered = draw_view(telemetry, control, images)
            cv2.imshow(WINDOW, rendered)
            if pair is not None and telemetry.get('input_verified') and telemetry.get('clipped_action') is not None:
                active = (telemetry.get('failure') or {}).get('failure_enabled', False)
                label = 'blur' if active else 'restored' if 'blur' in saved else 'normal'
                if label not in saved:
                    cv2.imwrite(str(output/f'observer_{label}.png'), rendered)
                    saved.add(label)
            key = cv2.waitKey(40) & 0xff
            dispatch(key, 'keyboard')
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                dispatch(ord('q'), 'window close')
            if telemetry.get('phase') == 'Finished' and args.auto_close:
                completed_at = completed_at or time.monotonic()
                if time.monotonic()-completed_at > 3:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
