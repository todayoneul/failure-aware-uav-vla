"""Landing provenance; the manager never computes a navigation command."""
import asyncio
from time import monotonic
import math
from src.integration.projectairsim_action_adapter import execute_action


async def guarded_execute(drone,manager,state,ground_z,events,flight_stamp,command,now=None):
    """Recheck conditions after inference, immediately before the next motion RPC."""
    if any(event.get('time_stamp',0)>flight_stamp for event in events):manager.fail('collision')
    if not .75<=ground_z-state['position'][2]<=4.05:manager.fail('extreme_altitude')
    manager.observe(state,now=now)
    if manager.status!='NAVIGATING':return None
    return await asyncio.wait_for(execute_action(drone,command),timeout=20)


async def land_and_confirm(drone,confirmation_seconds=5,stable_seconds=1):
    """SimpleFlight marks LANDED only after low throttle, often after disarm.

    Its official Land RPC confirms a commanded descent followed by one second
    of near-zero Z velocity. Require that RPC plus independently stable finite
    kinematics before disarm, then require the final LANDED state. Never accept
    a rejected landing while the vehicle still reports FLYING.
    """
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
            'touchdown_kinematics':sample,'stable_seconds':stable_seconds if before!=0 else None}


def landing_source(action, mission_state):
    if mission_state != 'LANDING':
        return None
    if action and action.get('output', '').strip().strip('<>') == 'LAND':
        return 'AeroVLA LAND signal at target tolerance'
    if action and action.get('stop'):
        return 'AeroVLA stop signal at target tolerance'
    return 'Mission Manager fallback after target tolerance'
