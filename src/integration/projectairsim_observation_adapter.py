"""AeroVLA's exact BGR -> RGB -> 224x448 mosaic and semantic prompt."""
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

def adapt_state(kinematics):
    pose = kinematics['pose']
    q = pose['orientation']
    return {'position': [pose['position'][k] for k in 'xyz'],
            'orientation': [q[k] for k in 'xyzw'],
            'velocity': [kinematics['twist']['linear'][k] for k in 'xyz']}

def make_mosaic(front_bgr, down_bgr):
    frames = []
    for frame in (front_bgr, down_bgr):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected uint8 HWC BGR camera image')
        frames.append(Image.fromarray(frame[..., ::-1].copy()).resize((224,224), Image.Resampling.BICUBIC))
    mosaic = Image.new('RGB', (224,448), (0,0,0))
    mosaic.paste(frames[0], (0,0))
    mosaic.paste(frames[1], (0,224))
    return mosaic

def semantic_direction(state, target_position):
    vector = np.asarray(target_position) - np.asarray(state['position'])
    if np.linalg.norm(vector[:2]) < .01:
        return ''
    body = Rotation.from_quat(state['orientation']).inv().apply(vector)
    angle = np.degrees(np.arctan2(body[1],body[0]))
    if -15 <= angle <= 15: return 'straight ahead '
    if 15 < angle <= 60: return 'forward-right '
    if 60 < angle <= 120: return 'to your right '
    if 120 < angle <= 180: return 'to your right rear '
    if -60 <= angle < -15: return 'forward-left '
    if -120 <= angle < -60: return 'to your left '
    if -180 <= angle < -120: return 'to your left rear '
    return ''

def make_prompt(state, target_position, instruction, direction_hint=True, freeform=None):
    if freeform:
        # Evaluation only: a sentence the model was not trained on replaces the whole template.
        return f'<image>\n{freeform.strip()}\nAction: '
    # Retain the original wrapper's instruction extraction and exact template.
    if 'degrees from you.' not in instruction or ' Please control' not in instruction:
        raise ValueError('Instruction needs upstream AeroVLA delimiters')
    description = instruction.split('degrees from you.')[1].split(' Please control')[0].strip()
    # Without the hint the template is the one upstream emits at zero target distance.
    direction = semantic_direction(state,target_position) if direction_hint else ''
    return f'<image>\nFly {direction}and find the target. {description}\nAction: '
