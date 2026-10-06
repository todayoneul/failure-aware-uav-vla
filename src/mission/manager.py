"""One mission target and evaluator. This module contains no drone control APIs.

The policy ends an episode itself (LAND or a zero action), as in the upstream AeroVLA
evaluator. Ground-truth coordinates only score the result; they never end navigation.
"""
from dataclasses import dataclass
import math
import time

TERMINAL={'SUCCESS','FAILED','ABORTED'}


@dataclass(frozen=True)
class MissionTarget:
    name: str
    position: tuple
    source: str
    description: str = None
    landmark: str = None

    def __post_init__(self):
        if not self.name or len(self.position)!=3 or not all(math.isfinite(v) for v in self.position):
            raise ValueError('Target requires a name and finite NED surface position')


class MissionManager:
    def __init__(self,success_radius_m=20.,max_steps=60,max_duration=900,diverging_steps=10,
                 stuck_steps=15,stuck_distance_m=.05):
        for value in (success_radius_m,max_steps,max_duration,diverging_steps,stuck_steps,stuck_distance_m):
            if not math.isfinite(value) or value<=0: raise ValueError('Mission limits must be positive and finite')
        self.success_radius_m=success_radius_m;self.max_steps=max_steps;self.max_duration=max_duration
        self.diverging_steps=diverging_steps;self.stuck_steps=stuck_steps;self.stuck_distance_m=stuck_distance_m
        self.reset()

    def reset(self):
        if getattr(self,'status','IDLE') in ('NAVIGATING','LANDING'):
            raise ValueError('Abort an active mission before resetting')
        self.status='IDLE';self.target=None;self.reason=None
        self.started=None;self.steps=0;self.hover_z=None
        self.distances=[];self.stuck=0;self.stop=None

    def select(self,target):
        if self.status not in {'IDLE','TARGET_SELECTED'}|TERMINAL: raise ValueError('Cannot retarget an active mission')
        self.reset();self.target=target;self.status='TARGET_SELECTED'

    def start(self,state,now=None):
        if self.status!='TARGET_SELECTED': raise ValueError('Select a target first')
        self.hover_z=float(state['position'][2])
        self.started=time.monotonic() if now is None else now
        self.status='NAVIGATING';self.reason=None;self.steps=0;self.stuck=0;self.stop=None
        self.distances=[self.metrics(state)['horizontal_m']]

    @property
    def goal_position(self):
        if self.target is None: return None
        return [self.target.position[0],self.target.position[1],self.hover_z if self.hover_z is not None else self.target.position[2]]

    def metrics(self,state):
        if self.target is None: return None
        x,y,z=state['position'];gx,gy,gz=self.goal_position
        if not all(math.isfinite(v) for v in (x,y,z)): raise ValueError('Invalid vehicle position')
        horizontal=math.hypot(gx-x,gy-y);vertical=abs(gz-z)
        return {'horizontal_m':horizontal,'vertical_m':vertical,'distance_3d_m':math.hypot(horizontal,vertical)}

    def observe(self,state=None,now=None):
        """Budget checks before a new decision; position never ends navigation here."""
        if self.status!='NAVIGATING': return self.status
        now=time.monotonic() if now is None else now
        if now-self.started>=self.max_duration: return self.fail('timeout')
        if self.steps>=self.max_steps: return self.fail('max_steps')
        return self.status

    def record_step(self,before,after,stop):
        """Score one executed decision. `stop` is the policy's own LAND / zero action."""
        if self.status!='NAVIGATING': return self.status
        self.steps+=1
        distance=self.metrics(after)['horizontal_m'];self.distances.append(distance)
        moved=math.hypot(after['position'][0]-before['position'][0],after['position'][1]-before['position'][1])
        self.stuck=self.stuck+1 if moved<self.stuck_distance_m else 0
        if stop:
            self.stop={'step':self.steps,'distance_m':distance,'within_radius':distance<=self.success_radius_m}
            self.status='LANDING'
        elif self.stuck>self.stuck_steps: self.fail('stuck')
        elif self.diverging(): self.fail('diverging')
        elif self.steps>=self.max_steps: self.fail('max_steps')
        return self.status

    def diverging(self):
        """Upstream ends an episode once the target distance has grown for ten frames in a row."""
        recent=self.distances[-(self.diverging_steps+1):]
        return len(recent)==self.diverging_steps+1 and all(b-a>self.stuck_distance_m for a,b in zip(recent,recent[1:]))

    def finish_stop(self,landed=None):
        """Judge the policy's stop; the vehicle's response to LAND is recorded, not required."""
        if self.status!='LANDING': return self.status
        self.stop['landed']=landed
        if self.stop['within_radius']:
            self.status='SUCCESS';self.reason=f'model stop {self.stop["distance_m"]:.2f} m from target'
        else:self.fail(f'stopped_outside_radius ({self.stop["distance_m"]:.2f} m)')
        return self.status

    def fail(self,reason):
        if self.status not in TERMINAL:self.status='FAILED';self.reason=reason
        return self.status

    def abort(self):
        if self.status not in TERMINAL:self.status='ABORTED';self.reason='manual abort'
        return self.status

    def snapshot(self,state=None):
        return {'state':self.status,'reason':self.reason,'steps':self.steps,
                'target':{'name':self.target.name,'surface_position':list(self.target.position),'source':self.target.source,
                          'description':self.target.description,'landmark':self.target.landmark} if self.target else None,
                'goal_position':self.goal_position,'errors':self.metrics(state) if state and self.target else None,
                'success_radius_m':self.success_radius_m,'stop':self.stop,
                'initial_distance_m':self.distances[0] if self.distances else None,
                'minimum_distance_m':min(self.distances) if self.distances else None}
