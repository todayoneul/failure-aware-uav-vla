"""AeroVLA-OFT action, proprio and chunk conventions; no torch, no simulator.

Actions keep AeroVLA's three axes and signs (forward, down, yaw) but are continuous, one small
motion per control tick, and predicted as a short chunk as in OpenVLA-OFT.
"""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'configs/aerovla_oft.json'
AXES=('forward_m','down_m','yaw_rad')
# Vehicle-only quantities. Absolute position and heading are left out on purpose: in a single map they
# would let the policy memorise where a target is instead of finding it in the images.
PROPRIO_FIELDS=('forward_speed_mps','lateral_speed_mps','down_speed_mps','height_m','yaw_rate_rps')
FORBIDDEN_PROPRIO=('target','bearing','distance','visible','position','yaw_rad','heading')


def load_config(path=None):
    config=json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding='utf-8'))
    bounds=config['action_bounds']
    if tuple(bounds)!=AXES or any(not low<high for low,high in bounds.values()):
        raise ValueError('action_bounds must list forward_m, down_m, yaw_rad with low < high')
    if not 1<=config['execute_horizon']<=config['chunk_size']:
        raise ValueError('execute_horizon must lie within the chunk')
    fields=config['proprio']['fields']
    if any(word in field for field in fields for word in FORBIDDEN_PROPRIO) or any(field not in PROPRIO_FIELDS for field in fields):
        raise ValueError('proprio may only carry vehicle motion fields')
    if len(config['proprio']['scale'])!=len(fields):
        raise ValueError('proprio scale needs one value per field')
    return config


def normalize_action(action,config):
    """Physical [forward, down, yaw] -> [-1, 1] per axis (OFT `bounds` normalisation), clipped."""
    result=[]
    for value,(low,high) in zip(action,config['action_bounds'].values()):
        value=min(high,max(low,float(value)))
        result.append(2*(value-low)/(high-low)-1)
    return result


def denormalize_action(values,config):
    result=[]
    for value,(low,high) in zip(values,config['action_bounds'].values()):
        value=min(1.,max(-1.,float(value)))
        result.append(low+(value+1)/2*(high-low))
    return result


def chunk_labels(actions,chunk_size):
    """Future action chunk for every step of one episode.

    Past the end of the episode the last action is repeated, as OpenVLA-OFT's `chunk_act_obs` does by
    clamping the future indices to the final timestep."""
    if not actions:return []
    last=len(actions)-1
    return [[list(actions[min(last,step+offset)]) for offset in range(chunk_size)] for step in range(len(actions))]


def is_stop(action,config):
    """The policy's stop is a near-zero action on every axis (AeroVLA's numeric stop, without the LAND text)."""
    limit=config['stop_threshold']
    return all(abs(value)<=limit*(high-low) for value,(low,high) in zip(action,config['action_bounds'].values()))


def proprio_vector(state,ground_z,config,yaw_rate=0.):
    """Vehicle motion in the body frame, scaled; nothing about the target or the map frame."""
    x,y,z,w=state['orientation']
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    vx,vy,vz=state['velocity']
    values={'forward_speed_mps':vx*math.cos(yaw)+vy*math.sin(yaw),'lateral_speed_mps':-vx*math.sin(yaw)+vy*math.cos(yaw),
            'down_speed_mps':vz,'height_m':ground_z-state['position'][2],'yaw_rate_rps':yaw_rate}
    return [values[field]/scale for field,scale in zip(config['proprio']['fields'],config['proprio']['scale'])]


def split_episodes(episode_ids,validation_fraction=.2,seed=0):
    """Episode-level split: no frame of a validation trajectory is ever trained on."""
    import random
    unique=sorted(set(episode_ids));random.Random(seed).shuffle(unique)
    count=max(1,round(len(unique)*validation_fraction)) if len(unique)>1 else 0
    validation=set(unique[:count])
    return [e for e in unique if e not in validation],sorted(validation)
