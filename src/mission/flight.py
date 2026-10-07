"""Guarded command execution and landing; the manager never computes a navigation command."""
import asyncio
from time import monotonic
import math
from src.integration.projectairsim_action_adapter import execute_action,fly_until_handoff

PRE_DESCENT_CLEARANCE_M=1.5
PRE_DESCENT_SPEED_MPS=2.


async def guarded_execute(drone,manager,state,events,flight_stamp,command,now=None):
    """Recheck conditions after inference, immediately before the next motion RPC."""
    if any(event.get('time_stamp',0)>flight_stamp for event in events):manager.fail('collision')
    manager.observe(state,now=now)
    if manager.status!='NAVIGATING':return None
    limit=2*command.get('expected_duration_sec',1.)+15
    if command.get('continuous'):
        # The command keeps flying after this returns; a collision ends the wait at once.
        collided=lambda:any(event.get('time_stamp',0)>flight_stamp for event in events)
        return await asyncio.wait_for(fly_until_handoff(drone,command,interrupted=collided),timeout=limit+5)
    return await asyncio.wait_for(execute_action(drone,command),timeout=limit)


def approach_altitude(current_z,surface_z,ground_z,limits,distance_m):
    """NED height to climb to, or None when no climb is due.

    The policy keeps its height in almost every decision and the direction hint is horizontal, so a
    surface at or above the flight level (a roof) is only reachable from above it. The climb waits until
    the vehicle is within `approach_distance_m` of the selected point: 22 m above its launch platform the
    policy stopped at once (5 of 6 runs), while a climb made 42 m from the target was followed by a
    flight onto the roof (3 of 3)."""
    if not limits['target_clearance_m'] or current_z<=surface_z-limits['minimum_clearance_m']:return None
    if distance_m>limits['approach_distance_m']:return None
    wanted=max(surface_z-limits['target_clearance_m'],ground_z-limits['maximum_clearance_m'])
    return wanted if wanted<current_z else None


async def climb_to(drone,z,speed):
    """Vertical move in place; the heading is kept."""
    start=drone.get_ground_truth_kinematics()['pose']['position']['z']
    duration=abs(z-start)/speed+1.
    await asyncio.wait_for(await drone.move_by_velocity_z_async(0.,0.,z,duration=duration),timeout=2*duration+10)
    await asyncio.wait_for(await drone.hover_async(),timeout=10)
    return {'from_z':start,'to_z':z}


async def descend_before_landing(drone,surface_z):
    """Native Land descends slowly and times out from height; close most of the gap first."""
    sample=drone.get_ground_truth_kinematics();z=sample['pose']['position']['z']
    low=surface_z-PRE_DESCENT_CLEARANCE_M
    if not math.isfinite(surface_z) or not math.isfinite(z) or low-z<.5:return None
    duration=(low-z)/PRE_DESCENT_SPEED_MPS+1.
    await asyncio.wait_for(await drone.move_by_velocity_z_async(0.,0.,low,duration=duration),timeout=2*duration+10)
    await asyncio.wait_for(await drone.hover_async(),timeout=10)
    return {'from_z':z,'to_z':low,'surface_z':surface_z}


async def land_and_confirm(drone,confirmation_seconds=5,stable_seconds=1,surface_z=None):
    """SimpleFlight marks LANDED only after low throttle, often after disarm.

    Its official Land RPC confirms a commanded descent followed by one second
    of near-zero Z velocity. Require that RPC plus independently stable finite
    kinematics before disarm, then require the final LANDED state. Never accept
    a rejected landing while the vehicle still reports FLYING.
    """
    descent=None
    if surface_z is not None:
        # A failed fast descent must never replace or prevent the landing itself.
        try:descent=await descend_before_landing(drone,surface_z)
        except Exception as error:descent={'error':repr(error),'surface_z':surface_z}
    response=await asyncio.wait_for(await drone.land_async(timeout_sec=20),timeout=25)
    before=drone.get_landed_state();sample=None;stable_since=None;anchor=None
    deadline=monotonic()+confirmation_seconds
    if before!=0:
        if response is not True:
            raise RuntimeError(f'Landing rejected: RPC return={response}, landed_state={before}')
        while True:
            sample=drone.get_ground_truth_kinematics();now=monotonic()
            position=[sample['pose']['position'][k] for k in 'xyz']
            velocities=[sample['twist'][kind][k] for kind in ('linear','angular') for k in 'xyz']
            stable=all(math.isfinite(v) for v in position+velocities) and all(abs(v)<=.08 for v in velocities)
            if anchor is not None:stable=stable and math.dist(anchor,position)<=.02
            if stable:
                if stable_since is None:stable_since=now;anchor=position
                if now-stable_since>=stable_seconds:break
            else:stable_since=None;anchor=None
            if now>=deadline:raise RuntimeError('Land RPC completed but touchdown kinematics are not stable')
            await asyncio.sleep(.1)
    disarmed=drone.disarm()
    if disarmed is False:raise RuntimeError('Disarm command rejected after touchdown')
    await asyncio.sleep(2)
    landed=drone.get_landed_state()
    if landed!=0:raise RuntimeError(f'Post-disarm LANDED state not confirmed: {landed}')
    return {'land_return':response,'landed_state_before_disarm':before,'landed_state':landed,
            'touchdown_kinematics':sample,'stable_seconds':stable_seconds if before!=0 else None,
            'pre_descent':descent}


def surface_below(world,state):
    """Simulator ground height under the vehicle; None when the query is unavailable."""
    try:
        value=float(world.get_surface_elevation_at_point(state['position'][0],state['position'][1]))
    except Exception:
        return None
    return value if math.isfinite(value) else None
