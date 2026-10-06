"""One mission target and evaluator. This module contains no drone control APIs."""
from dataclasses import dataclass
import math
import time

TERMINAL={'SUCCESS','FAILED','ABORTED'}
KINDS={'GO_TO','GO_TO_AND_HOVER','GO_TO_AND_LAND'}


@dataclass(frozen=True)
class MissionTarget:
    name: str
    position: tuple
    source: str

    def __post_init__(self):
        if not self.name or len(self.position)!=3 or not all(math.isfinite(v) for v in self.position):
            raise ValueError('Target requires a name and finite NED surface position')


class MissionManager:
    def __init__(self, horizontal_tolerance=.45, altitude_tolerance=.4, velocity_tolerance=.25,
                 hover_seconds=1.0,max_steps=60,max_duration=300):
        for value in (horizontal_tolerance,altitude_tolerance,velocity_tolerance,hover_seconds,max_steps,max_duration):
            if not math.isfinite(value) or value<=0: raise ValueError('Mission limits must be positive and finite')
        self.horizontal_tolerance=horizontal_tolerance;self.altitude_tolerance=altitude_tolerance
        self.velocity_tolerance=velocity_tolerance;self.hover_seconds=hover_seconds
        self.max_steps=max_steps;self.max_duration=max_duration
        self.reset()

    def reset(self):
        if getattr(self,'status','IDLE') in ('NAVIGATING','HOVERING','LANDING'):
            raise ValueError('Abort an active mission before resetting')
        self.status='IDLE';self.target=None;self.kind=None;self.reason=None
        self.started=None;self.hover_started=None;self.steps=0;self.hover_z=None

    def select(self,target):
        if self.status not in {'IDLE','TARGET_SELECTED'}|TERMINAL: raise ValueError('Cannot retarget an active mission')
        self.target=target;self.status='TARGET_SELECTED';self.reason=None

    def start(self,kind,state,now=None):
        if self.status!='TARGET_SELECTED' or kind not in KINDS: raise ValueError('Select a target and a supported mission')
        self.hover_z=float(state['position'][2])
        if self.target.position[2]-self.hover_z<.8:
            raise ValueError('Target surface is above the supported flight clearance')
        self.kind=kind;self.started=time.monotonic() if now is None else now
        self.status='NAVIGATING';self.reason=None;self.steps=0;self.hover_started=None

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

    def observe(self,state,now=None):
        if self.status in TERMINAL or self.status in ('IDLE','TARGET_SELECTED'): return self.status
        now=time.monotonic() if now is None else now
        if now-self.started>=self.max_duration: return self.fail('timeout')
        errors=self.metrics(state)
        reached=errors['horizontal_m']<=self.horizontal_tolerance
        hover_altitude=errors['vertical_m']<=self.altitude_tolerance
        if self.status=='NAVIGATING' and reached and (self.kind!='GO_TO_AND_HOVER' or hover_altitude):
            self.status={'GO_TO':'SUCCESS','GO_TO_AND_HOVER':'HOVERING','GO_TO_AND_LAND':'LANDING'}[self.kind]
            if self.status=='SUCCESS': self.reason='horizontal goal tolerance reached'
        if self.status=='HOVERING':
            if not reached or not hover_altitude:
                self.status='NAVIGATING';self.hover_started=None
                return self.status
            speed=math.sqrt(sum(v*v for v in state['velocity']))
            stable=reached and errors['vertical_m']<=self.altitude_tolerance and speed<=self.velocity_tolerance
            if stable:
                self.hover_started=now if self.hover_started is None else self.hover_started
                if now-self.hover_started>=self.hover_seconds:
                    self.status='SUCCESS';self.reason='position/altitude/velocity stable during hover'
            else:self.hover_started=None
        if self.status=='NAVIGATING' and self.steps>=self.max_steps:self.fail('max_steps')
        return self.status

    def confirm_landing(self,state,landed):
        if self.status!='LANDING':return self.status
        errors=self.metrics(state)
        if landed and errors['horizontal_m']<=self.horizontal_tolerance and abs(state['position'][2]-self.target.position[2])<=.5:
            self.status='SUCCESS';self.reason='aligned target landing confirmed'
        elif landed:self.fail('landed outside target tolerance')
        return self.status

    def fail(self,reason):
        if self.status not in TERMINAL:self.status='FAILED';self.reason=reason
        return self.status

    def abort(self):
        if self.status not in TERMINAL:self.status='ABORTED';self.reason='manual abort'
        return self.status

    def snapshot(self,state=None):
        return {'state':self.status,'type':self.kind,'reason':self.reason,'steps':self.steps,
                'target':{'name':self.target.name,'surface_position':list(self.target.position),'source':self.target.source} if self.target else None,
                'goal_position':self.goal_position,'errors':self.metrics(state) if state and self.target else None,
                'horizontal_tolerance':self.horizontal_tolerance}
