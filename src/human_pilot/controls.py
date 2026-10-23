"""What a human pilot's keys and sticks become, and what is recorded of it. No simulator, no window.

A human flies with forward and backward, up and down, and a turn (no sideways). The flight is continuous: the command is
updated twenty times a second, every change is eased in (`Smoother`), and the vehicle never waits for a decision. What is
recorded is what a policy would have had to say to fly the same way: every 0.5 s the commands of the last 0.5 s are turned
into one action in the policy's units (`TickLabel`): forward metres, downward metres and turn radians per tick.

A policy can do less than a human: it never flies backward, and its action bounds cap it at 2 m/s forward, 1 m/s up or
down and 24 deg/s of turn. A human may do more; what a policy could not have said is recorded and marked, not dropped. A
label is marked, and the dataset builder leaves the decision out of every label window
(scripts/build_visual_search_dataset.py), when:

    backward    the vehicle was flown backward during the tick (a policy has no backward)
    over_bound  an axis went past the policy's action bound (the boost tier): the label would be clipped and wrong
    pause       every axis inside the stop band of src/aerovla_oft/spec.is_stop (the human let go of everything): the
                policy would read it as a stop, and four in a row end a flight
    below_safe  outside the stop band but less than `safe_ratio` times it on every axis (the start or the end of a key
                press): the margin stage 1b fixed for every recorded label (configs/generic_landing_stage1b.json)

The stop at the end of a flight (after the pilot says it is done, or after a touchdown) is the policy's stop and is not
marked.

Speeds: a key held flies at the chosen tier. Tiers 1-3 lie inside the policy's bounds and are at least `safe_ratio` times
the stop band on each axis, so a held key there records a label far from a stop; tier 4 (boost) is faster than a policy
can fly and is never a label. A stick past its dead zone is scaled from that floor to the tier's rate.
"""
import math
from dataclasses import dataclass, field

STOP_THRESHOLD=.08      # src/aerovla_oft/spec.is_stop: every axis within 8 % of its range
SAFE_RATIO=3.          # configs/generic_landing_stage1b.json, stop_safety.safe_ratio
DEAD_ZONE=.2           # of a stick's travel
TIER_NAMES=('slow','normal','fast','boost')


@dataclass(frozen=True)
class Axes:
    """Rates: forward m/s (- backward), down m/s (+ descends), turn rad/s (+ turns right)."""
    forward:float=0.
    down:float=0.
    turn:float=0.


@dataclass
class Inputs:
    """What the pilot holds at this moment. Keys are booleans; a stick is -1..1 and wins over the keys of its axis when it is past the dead zone."""
    forward:bool=False
    backward:bool=False
    up:bool=False
    down:bool=False
    left:bool=False
    right:bool=False
    stick_forward:float=0.    # + forward, - backward
    stick_turn:float=0.
    stick_vertical:float=0.   # + up


def limits_of(oft):
    """Largest rate of each axis a policy can command: an action bound per tick over the tick."""
    tick=oft['tick_s'];bounds=oft['action_bounds']
    return Axes(bounds['forward_m'][1]/tick,bounds['down_m'][1]/tick,bounds['yaw_rad'][1]/tick)


def stop_rates(oft):
    """The stop band of each axis as a rate: 8 % of the axis range per tick."""
    tick=oft['tick_s'];bounds=oft['action_bounds'];width=lambda pair:pair[1]-pair[0]
    return Axes(STOP_THRESHOLD*width(bounds['forward_m'])/tick,STOP_THRESHOLD*width(bounds['down_m'])/tick,STOP_THRESHOLD*width(bounds['yaw_rad'])/tick)


def tiers_of(oft,boost=None):
    """Four speed tiers per axis: slow, normal and fast inside a policy's bounds (the slowest just above `SAFE_RATIO`
    times the stop band, the fastest the bound), and `boost` (an Axes of rates) beyond them."""
    top=limits_of(oft);floor=stop_rates(oft)
    tier=lambda axis,share:max(getattr(floor,axis)*SAFE_RATIO*1.1,getattr(top,axis)*share)
    tiers=[Axes(*(tier(axis,share) for axis in ('forward','down','turn'))) for share in (.3,.6,1.)]
    return tiers+[boost or Axes(2.5*top.forward,2.5*top.down,2.5*top.turn)]


def stick(value,floor,top):
    """A stick past its dead zone, from the floor of the axis to `top`; inside the dead zone nothing."""
    magnitude=abs(value)
    if magnitude<=DEAD_ZONE:return 0.
    return math.copysign(floor+max(0.,top-floor)*min(1.,(magnitude-DEAD_ZONE)/(1-DEAD_ZONE)),value)


def wanted(inputs,tier,oft):
    """The rates the pilot asks for now, at most the tier's on each axis."""
    floor=stop_rates(oft);low=lambda axis:min(getattr(floor,axis)*SAFE_RATIO*1.1,getattr(tier,axis))
    forward=(tier.forward if inputs.forward else 0.)-(tier.forward if inputs.backward else 0.)
    if abs(inputs.stick_forward)>DEAD_ZONE:forward=stick(inputs.stick_forward,low('forward'),tier.forward)
    vertical=(tier.down if inputs.up else 0.)-(tier.down if inputs.down else 0.)
    if abs(inputs.stick_vertical)>DEAD_ZONE:vertical=stick(inputs.stick_vertical,low('down'),tier.down)
    turn=(tier.turn if inputs.right else 0.)-(tier.turn if inputs.left else 0.)
    if abs(inputs.stick_turn)>DEAD_ZONE:turn=stick(inputs.stick_turn,low('turn'),tier.turn)
    return Axes(forward,-vertical,turn)


@dataclass
class Smoother:
    """Eases the commanded rates toward what the pilot asks: each axis changes by at most its acceleration per second."""
    acceleration:Axes=field(default_factory=lambda:Axes(4.,4.,3.))
    now:Axes=field(default_factory=Axes)

    def step(self,target,dt):
        ease=lambda axis:getattr(self.now,axis)+max(-getattr(self.acceleration,axis)*dt,min(getattr(self.acceleration,axis)*dt,getattr(target,axis)-getattr(self.now,axis)))
        self.now=Axes(ease('forward'),ease('down'),ease('turn'));return self.now

    def stop(self):self.now=Axes();return self.now


class TickLabel:
    """The commands flown between two decisions, as one action of the policy. `add` is called with the rates flown for
    `dt` seconds; `close` gives the action the policy would have had to give at the decision that opened the tick, and
    `reason` says why it cannot be a label (backward, over_bound) or None. Either holds when it happened at any moment of
    the tick, even if the tick's sum fits a policy's bounds (slowing down from the fourth tier, say)."""
    def __init__(self,oft):
        self.oft=oft;self.tick=oft['tick_s'];self.reason=None;self.reset()

    def reset(self):self.forward=self.down=self.turn=self.elapsed=0.;self.back=self.fast=False

    def add(self,rates,dt):
        self.forward+=rates.forward*dt;self.down+=rates.down*dt;self.turn+=rates.turn*dt;self.elapsed+=dt
        self.back=self.back or rates.forward<0.;self.fast=self.fast or beyond(rates,self.oft)

    def close(self):
        """[forward m, down m, turn rad] per tick (scaled to the nominal tick when the tick ran long or short), clipped to the bounds."""
        scale=self.tick/self.elapsed if self.elapsed>0 else 0.;bounds=self.oft['action_bounds']
        raw=[self.forward*scale,self.down*scale,self.turn*scale];pairs=[bounds['forward_m'],bounds['down_m'],bounds['yaw_rad']]
        label=[max(pair[0],min(pair[1],value)) for value,pair in zip(raw,pairs)]
        self.reason='backward' if self.back else 'over_bound' if self.fast or any(abs(a-b)>1e-6 for a,b in zip(raw,label)) else None
        self.reset();return label


def stop_ratio(action,oft):
    """Largest axis of an action over its stop band (as src/landing/stop_safety.stop_ratio): <= 1 is a stop."""
    bounds=oft['action_bounds'];limits=[oft['stop_threshold']*(pair[1]-pair[0]) for pair in bounds.values()]
    return max(abs(value)/limit for value,limit in zip(action,limits))


def mark(action,oft,terminal=False,reason=None):
    """Why a recorded decision stays out of the label windows, or None. `reason`: what the tick itself reported."""
    if terminal:return None
    if reason:return reason
    ratio=stop_ratio(action,oft)
    if ratio<=1.:return 'pause'
    if ratio<SAFE_RATIO:return 'below_safe'
    return None


def beyond(rates,oft):
    """Is the vehicle being flown in a way a policy could not command (backward, or faster than its bounds)?"""
    top=limits_of(oft)
    return rates.forward<-1e-6 or rates.forward>top.forward+1e-6 or abs(rates.down)>top.down+1e-6 or abs(rates.turn)>top.turn+1e-6
