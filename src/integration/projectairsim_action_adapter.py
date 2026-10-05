"""Strict bin parsing and bounded NED displacement -> official flight APIs."""
import math
import re
from scipy.spatial.transform import Rotation

def parse_action(text):
    output = text.split('Action:')[-1].replace('</s>','').replace('<pad>','').strip()
    if 'LAND' in output:
        return {'fwd':0.,'down':0.,'yaw':0.,'stop':True,'bins':None,'output':output}
    # Three integers only. Reject malformed output rather than fabricate a stop.
    tokens = re.findall(r'(?<![\w.])-?\d+(?![\w.])', output)
    if len(tokens) != 3 or any(not 0 <= int(t) <= 98 for t in tokens):
        raise ValueError(f'Invalid AeroVLA action bins: {output!r}')
    bins = list(map(int,tokens))
    fwd=bins[0]/98*5
    down=bins[1]/98*10-5
    yaw=bins[2]/98*2.2-1.1
    return {'fwd':fwd,'down':down,'yaw':yaw,
            'stop':fwd<.01 and abs(down)<.01 and abs(yaw)<.01,
            'bins':bins,'output':output}

def convert_action(action, state, ground_z):
    if not all(math.isfinite(action[k]) for k in ('fwd','down','yaw')):
        raise ValueError('Non-finite model action')
    if not math.isfinite(ground_z) or not all(math.isfinite(v) for v in state['position']):
        raise ValueError('Non-finite vehicle state')
    clearance=ground_z-state['position'][2]
    if not .78 <= clearance <= 4.02:
        raise ValueError(f'Vehicle outside altitude envelope: clearance={clearance}')
    yaw = float(Rotation.from_quat(state['orientation']).as_euler('xyz')[2])
    delta = max(-math.radians(15), min(math.radians(15), action['yaw']))
    heading = math.atan2(math.sin(yaw+delta), math.cos(yaw+delta))
    fwd = max(0., min(.5, action['fwd']))
    down = max(-.3, min(.3, action['down']))
    z = state['position'][2]
    # Clearance relative to the initial resting platform (not terrain-ray AGL).
    target_z = max(ground_z-4., min(ground_z-.8, z+down))
    displacement = [fwd*math.cos(heading), fwd*math.sin(heading), target_z-z]
    if action['stop']:
        displacement=[0.,0.,0.]
        delta=0.
        heading=yaw
    return {'stop':action['stop'], 'yaw_delta_rad':delta,'target_yaw_rad':heading,
            'duration_sec':1.,'displacement_ned':displacement,
            'velocity_ned_mps':displacement.copy(), 'target_z':target_z,
            'ground_reference_z':ground_z, 'clearance_limits_m':[.8,4.]}

async def execute_action(drone, command):
    if command['stop']:
        return {'hover': await (await drone.hover_async())}
    returns={}
    if abs(command['yaw_delta_rad'])>.001:
        returns['yaw']=await (await drone.rotate_to_yaw_async(
            command['target_yaw_rad'], timeout_sec=5, margin=math.radians(3), yaw_rate=.3))
    vx,vy,vz=command['velocity_ned_mps']
    returns['move']=await (await drone.move_by_velocity_async(vx,vy,vz,
        duration=command['duration_sec'], yaw_is_rate=False, yaw=command['target_yaw_rad']))
    returns['hover']=await (await drone.hover_async())
    return returns
