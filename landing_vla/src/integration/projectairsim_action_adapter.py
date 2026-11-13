"""Strict bin parsing and upstream-equivalent NED displacement -> official flight APIs."""
import asyncio
import math
import re
from time import monotonic
from scipy.spatial.transform import Rotation

# Upstream AeroVLA move_path(): rotate to yaw+pred_yaw, then translate (fwd, down) at
# 1 m/s only when |pred_yaw| < 0.25 rad; a larger turn changes altitude in place.
DEFAULT_LIMITS={'minimum_clearance_m':.8,'maximum_clearance_m':30.,'displacement_scale':1.,
                'cruise_speed_mps':1.,'vertical_speed_mps':2.,'yaw_move_threshold_rad':.25,
                'micro_move_threshold_m':1.,'continuous_lead_s':2.,'target_clearance_m':6.,
                'approach_distance_m':45.}
# Small steps that keep the older platform demos near the start platform.
PLATFORM_DEMO_LIMITS={'displacement_scale':.1,'maximum_clearance_m':4.}
MODEL_VERTICAL_RANGE_M=5.
EPSILON=.01
YAW_SETTLE_RATE=1.2  # rad/s; measured: 63 deg settles within 3 deg in about 1 s.
TRIM_TOLERANCE_M=.15
# A turn is held this long past its settle time so the vehicle keeps its place while the next decision is computed.
HOLD_AFTER_TURN_S=4.


def flight_limits(overrides=None):
    limits={**DEFAULT_LIMITS,**(overrides or {})}
    if not all(math.isfinite(limits[key]) for key in DEFAULT_LIMITS):
        raise ValueError('Invalid flight limits')
    if not 0<limits['minimum_clearance_m']<limits['maximum_clearance_m']:
        raise ValueError('Invalid clearance limits')
    if min(limits[key] for key in ('displacement_scale','cruise_speed_mps','vertical_speed_mps',
                                   'yaw_move_threshold_rad','micro_move_threshold_m','continuous_lead_s',
                                   'approach_distance_m'))<=0:
        raise ValueError('Flight scales, speeds and thresholds must be positive')
    if limits['target_clearance_m']<0:
        raise ValueError('target_clearance_m is zero (off) or positive')
    if type(limits.setdefault('continuous',False)) is not bool:
        raise ValueError('continuous must be true or false')
    return limits


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


def allowed_action_tokens(generated, digits, space, land, end):
    """Token ids that may follow `generated` in AeroVLA's output: `DD DD DD`, an optional ` LAND`, the end token.

    `digits` are the ids of '0'..'9', `land` the two ids of ' LAND'. Bins stop at 98, so a 9 is never followed by a 9.
    """
    position=len(generated)
    if position in (2,5): return [space]
    if position in (0,3,6): return list(digits)
    if position in (1,4,7): return list(digits[:9]) if generated[-1]==digits[9] else list(digits)
    if position==8: return [end,land[0]]
    if position==9 and generated[-1]==land[0]: return [land[1]]
    return [end]


def convert_action(action, state, ground_z, limits=None, altitude_reference=None):
    """`altitude_reference` is the previous commanded height. Continuous flight re-plans before a leg
    settles, so basing each step on the measured height would integrate the tracking error into a drift."""
    limits=flight_limits(limits)
    if not all(math.isfinite(action[k]) for k in ('fwd','down','yaw')):
        raise ValueError('Non-finite model action')
    if not math.isfinite(ground_z) or not all(math.isfinite(v) for v in state['position']):
        raise ValueError('Non-finite vehicle state')
    x,y,z=state['position'];scale=limits['displacement_scale']
    yaw=float(Rotation.from_quat(state['orientation']).as_euler('xyz')[2])
    # Clearance relative to the initial resting platform (not terrain-ray AGL).
    floor_z=ground_z-limits['minimum_clearance_m'];ceiling_z=ground_z-limits['maximum_clearance_m']
    command={'stop':bool(action['stop']),'mode':'stop','yaw_delta_rad':0.,'target_yaw_rad':yaw,
             'displacement_ned':[0.,0.,0.],'target_position':[x,y,z],'target_z':z,
             'altitude_clamped':False,'expected_duration_sec':1.,'ground_reference_z':ground_z,
             'clearance_limits_m':[limits['minimum_clearance_m'],limits['maximum_clearance_m']],
             'cruise_speed_mps':limits['cruise_speed_mps'],'vertical_speed_mps':limits['vertical_speed_mps'],
             'micro_move_threshold_m':limits['micro_move_threshold_m'],'displacement_scale':scale,
             'continuous':limits['continuous'],'continuous_lead_s':limits['continuous_lead_s']}
    if action['stop']:
        return command
    delta=max(-math.pi,min(math.pi,action['yaw']))
    heading=math.atan2(math.sin(yaw+delta),math.cos(yaw+delta))
    turning=abs(delta)>=limits['yaw_move_threshold_rad']
    fwd=0. if turning else max(0.,action['fwd'])*scale
    base_z=z if altitude_reference is None or not math.isfinite(altitude_reference) else altitude_reference
    wanted_z=base_z+action['down']*scale
    # The envelope clamps the target; it never ends a mission. A vehicle outside it is
    # brought back by at most the model's own vertical range per step.
    bound=MODEL_VERTICAL_RANGE_M*scale
    down=max(-bound,min(bound,min(floor_z,max(ceiling_z,wanted_z))-z))
    target_z=z+down
    displacement=[fwd*math.cos(heading),fwd*math.sin(heading),down]
    translate=fwd>EPSILON
    travel=math.hypot(fwd,down)
    duration=max(1.,travel/limits['cruise_speed_mps'])+abs(down)/limits['vertical_speed_mps'] if translate else \
        max(1.+abs(delta)/YAW_SETTLE_RATE,abs(down)/limits['vertical_speed_mps']+1.)
    command.update(mode='translate' if translate else 'turn',yaw_delta_rad=delta,target_yaw_rad=heading,
                   displacement_ned=displacement,target_position=[x+displacement[0],y+displacement[1],target_z],
                   target_z=target_z,altitude_clamped=abs(target_z-wanted_z)>1e-6,expected_duration_sec=duration)
    return command


def turn_duration(command):
    return max(1.+abs(command['yaw_delta_rad'])/YAW_SETTLE_RATE,abs(command['displacement_ned'][2])/command['vertical_speed_mps']+1.)


async def execute_action(drone, command):
    """Position/altitude-hold primitives; plain velocity commands lose about 6 cm of height per step."""
    if command['stop']:
        return {'hover': await (await drone.hover_async())}
    x,y,z=command['target_position'];dx,dy,dz=command['displacement_ned']
    hold={'yaw_is_rate':False,'yaw':command['target_yaw_rad']}
    returns={}
    if command['mode']=='turn':
        returns['turn']=await (await drone.move_by_velocity_z_async(0.,0.,z,duration=turn_duration(command),**hold))
    elif math.sqrt(dx*dx+dy*dy+dz*dz)<command['micro_move_threshold_m']:
        # The position controller ignores sub-metre goals; upstream also uses a 1 s velocity move here.
        returns['move']=await (await drone.move_by_velocity_z_async(dx,dy,z,duration=1.,**hold))
    else:
        travel=math.sqrt(dx*dx+dy*dy+dz*dz)
        returns['move']=await (await drone.move_to_position_async(
            x,y,z,command['cruise_speed_mps'],timeout_sec=travel/command['cruise_speed_mps']*2+5,**hold))
        returns['settle']=await (await drone.hover_async())
        residual=z-drone.get_ground_truth_kinematics()['pose']['position']['z']
        if abs(residual)>TRIM_TOLERANCE_M:
            returns['trim']=await (await drone.move_by_velocity_z_async(
                0.,0.,z,duration=abs(residual)/command['vertical_speed_mps']+1.,**hold))
    returns['hover']=await (await drone.hover_async())
    return returns


async def begin_action(drone, command):
    """Issue a command without waiting for it to finish; a later command replaces it in flight."""
    x,y,z=command['target_position'];dx,dy,dz=command['displacement_ned']
    hold={'yaw_is_rate':False,'yaw':command['target_yaw_rad']}
    if command['mode']=='turn':
        return await drone.move_by_velocity_z_async(0.,0.,z,duration=turn_duration(command)+HOLD_AFTER_TURN_S,**hold)
    travel=math.sqrt(dx*dx+dy*dy+dz*dz)
    if travel<command['micro_move_threshold_m']:
        return await drone.move_by_velocity_z_async(dx,dy,z,duration=1.,**hold)
    return await drone.move_to_position_async(x,y,z,command['cruise_speed_mps'],
                                              timeout_sec=travel/command['cruise_speed_mps']*2+5,**hold)


async def fly_until_handoff(drone, command, interrupted=lambda:False, poll_s=.05, clock=monotonic):
    """Continuous flight: return while the vehicle is still moving, `continuous_lead_s` before it would
    arrive, so the next observation and decision are made in flight and the next command joins this one
    without a stop. A turn in place is waited out, because a view taken mid-turn is not what the model acts on.
    """
    if command['stop']:
        return {'hover': await (await drone.hover_async()),'handoff':'stop'}
    task=await begin_action(drone,command);started=clock()
    x,y,_=command['target_position'];dx,dy,dz=command['displacement_ned']
    lead=command['continuous_lead_s'];travel=math.sqrt(dx*dx+dy*dy+dz*dz)
    deadline=2*command['expected_duration_sec']+15
    while True:
        elapsed=clock()-started
        if interrupted():reason='interrupted';break
        if command['mode']=='turn':
            if elapsed>=turn_duration(command):reason='turned';break
        elif travel<command['micro_move_threshold_m']:
            if elapsed>=1.-lead:reason='lead';break
        else:
            position=drone.get_ground_truth_kinematics()['pose']['position']
            if math.hypot(x-position['x'],y-position['y'])<=lead*command['cruise_speed_mps']:reason='lead';break
            if task.done():reason='arrived';break
        if elapsed>deadline:reason='timeout';break
        await asyncio.sleep(poll_s)
    return {'handoff':reason,'elapsed_s':clock()-started,'completed':task.done()}
