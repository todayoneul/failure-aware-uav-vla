"""Landing finalizer: the low-level end of a landing. No simulator, no torch, and nothing about targets.

The policy brings the vehicle down onto a surface with the same three action axes it uses for
everything else. What happens once the vehicle rests there is not left to the policy: the executor
recognises a stable contact, latches LANDED, passes no further motion command on and disarms.

    FLYING -> LANDING_DESCENT -> CONTACT_CANDIDATE -> STABLE_CONTACT -> LANDED_LATCHED -> DISARMED

It is active only when the mission's sentence asks for a landing, and it reads three things, all of
them the vehicle's own: whether a contact was reported, how fast the vehicle is moving, and whether the
executor was recently told to descend. It is never told where the target is or what the vehicle stands
on. Whether that was the pad the sentence named is decided afterwards by the evaluator.
"""
import math
import re

STATES=('FLYING','LANDING_DESCENT','CONTACT_CANDIDATE','STABLE_CONTACT','LANDED_LATCHED','DISARMED')


def landing_requested(instruction):
    """A mission asks for a landing when its sentence uses `land` as a verb: "... and land on it", "Land on ...",
    "... approach it, and land". The noun in "the landing pad" alone does not ask for one."""
    return bool(re.search(r'\bland\b',instruction.lower()))


class LandingFinalizer:
    def __init__(self,settings,instruction,enabled=True):
        self.settings=settings;self.active=bool(enabled and landing_requested(instruction));self.state='FLYING'
        self.recent=[];self.previous=None;self.speed=(0.,0.);self.contact=None;self.stable=0;self.stable_duration_s=None
        self.path=['FLYING'];self.disarm_ok=None

    def _go(self,state):
        if state!=self.state:self.state=state;self.path.append(state)

    @property
    def latched(self):
        """Once latched, the executor passes no motion command on."""
        return self.state in ('LANDED_LATCHED','DISARMED')

    def command(self,action):
        """What the executor flies this decision: the policy's action, or nothing once the landing is latched."""
        return [0.,0.,0.] if self.latched else list(action)

    def update(self,position,stamp_s,contact_event,executed_down_m):
        """One decision of the executor.

        position: the vehicle's own position (x, y, z down); stamp_s: when it was read; contact_event: a contact
        was reported since the last decision; executed_down_m: the descent the executor was last told to fly."""
        settings=self.settings;before=self.speed
        if self.previous is not None and stamp_s>self.previous[1]:
            # Speed over the decision that just ended, from the vehicle's own positions: (vertical, horizontal).
            (x,y,z),then=self.previous;dt=stamp_s-then
            self.speed=(abs(position[2]-z)/dt,math.hypot(position[0]-x,position[1]-y)/dt)
        self.previous=(tuple(position),stamp_s)
        if not self.active or self.latched:return self.state
        self.recent=(self.recent+[executed_down_m>=settings['descent_down_m']])[-settings['descent_window']:]
        descending=any(self.recent)
        if self.state=='FLYING' and descending:self._go('LANDING_DESCENT')
        elif self.state=='LANDING_DESCENT' and not descending:self._go('FLYING')
        if self.state=='LANDING_DESCENT' and contact_event:
            # A contact while coming down. The speed just before it is the touchdown speed.
            self.contact={'z':position[2],'stamp_s':stamp_s,'vertical_mps':before[0],'horizontal_mps':before[1]};self.stable=0
            self._go('CONTACT_CANDIDATE')
        elif self.state=='CONTACT_CANDIDATE':
            if abs(position[2]-self.contact['z'])>settings['leave_m']:
                # Off the surface again: not a landing yet.
                self.contact=None;self.stable=0;self._go('LANDING_DESCENT' if descending else 'FLYING')
            else:
                still=(self.speed[0]<=settings['max_vertical_mps'] and self.speed[1]<=settings['max_horizontal_mps']
                       and abs(position[2]-self.contact['z'])<=settings['height_tolerance_m'])
                self.stable=self.stable+1 if still else 0
                if self.stable>=settings['stable_decisions']:
                    self.stable_duration_s=stamp_s-self.contact['stamp_s'];self._go('STABLE_CONTACT');self._go('LANDED_LATCHED')
        return self.state

    def disarmed(self,ok):
        """The executor has switched the motors off (or failed to)."""
        self.disarm_ok=bool(ok)
        if ok and self.state=='LANDED_LATCHED':self._go('DISARMED')

    def report(self):
        contact=self.contact or {}
        return {'active':self.active,'state':self.state,'path':list(self.path),'touchdown_detected':self.contact is not None or self.latched,
                'stable_contact_detected':'STABLE_CONTACT' in self.path,'finalizer_triggered':'LANDED_LATCHED' in self.path,
                'disarm_triggered':bool(self.disarm_ok),'touchdown_velocity_mps':contact.get('vertical_mps'),
                'horizontal_velocity_mps':contact.get('horizontal_mps'),'stable_duration_s':self.stable_duration_s}
