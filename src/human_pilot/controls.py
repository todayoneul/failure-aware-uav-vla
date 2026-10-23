"""What a human pilot's keys and sticks become, and what is recorded of it. No simulator, no window.

A human flies with the three axes a policy has and nothing else: forward (never backward or sideways), up and down, and a
turn. Inside those axes the flight is continuous: the command is updated twenty times a second, every change is eased in
(`Smoother`), and the vehicle never waits for a decision. What is recorded is what a policy would have had to say to fly
the same way: every 0.5 s the commands of the last 0.5 s are turned into one action in the policy's units
(`TickLabel`): forward metres, downward metres and turn radians per tick, inside the action bounds.

A label a policy must never learn before the vehicle has done what the sentence asked is marked, not dropped:

    pause       every axis inside the stop band of src/aerovla_oft/spec.is_stop (the human let go of everything): the
                policy would read it as a stop, and four in a row end a flight
    below_safe  outside the stop band but less than `safe_ratio` times it on every axis (the start or the end of a key
                press): the margin stage 1b fixed for every recorded label (configs/generic_landing_stage1b.json)

The dataset builder leaves a marked decision out of every label window (scripts/build_visual_search_dataset.py). The
stop at the end of a flight (after the pilot says it is done, or after a touchdown) is the policy's stop and is not marked.

Speeds: a key held flies at the chosen tier; every tier is at least `safe_ratio` times the stop band on its axis, so a held
key never records a label near a stop. A stick past its dead zone is scaled from that floor to the axis maximum.
"""
import math
from dataclasses import dataclass, field

STOP_THRESHOLD=.08      # src/aerovla_oft/spec.is_stop: every axis within 8 % of its range
SAFE_RATIO=3.          # configs/generic_landing_stage1b.json, stop_safety.safe_ratio
DEAD_ZONE=.2           # of a stick's travel


@dataclass(frozen=True)
class Axes:
    """Rates in the policy's frame: forward m/s (>= 0), down m/s (+ descends), turn rad/s (+ turns right)."""
    forward:float=0.
    down:float=0.
    turn:float=0.


@dataclass
class Inputs:
    """What the pilot holds at this moment. Keys are booleans; a stick is -1..1 and wins over the keys of its axis when it is past the dead zone."""
    forward:bool=False
    up:bool=False
    down:bool=False
    left:bool=False
    right:bool=False
    stick_forward:float=0.
    stick_turn:float=0.
    stick_vertical:float=0.   # + up


def limits_of(oft):
    """Largest rate of each axis: an action bound per tick over the tick."""
    tick=oft['tick_s'];bounds=oft['action_bounds']
    return Axes(bounds['forward_m'][1]/tick,bounds['down_m'][1]/tick,bounds['yaw_rad'][1]/tick)


def stop_rates(oft):
    """The stop band of each axis as a rate: 8 % of the axis range per tick."""
    tick=oft['tick_s'];bounds=oft['action_bounds'];width=lambda pair:pair[1]-pair[0]
    return Axes(STOP_THRESHOLD*width(bounds['forward_m'])/tick,STOP_THRESHOLD*width(bounds['down_m'])/tick,STOP_THRESHOLD*width(bounds['yaw_rad'])/tick)


def tiers_of(oft):
    """Three speed tiers (slow, normal, fast) per axis. The slowest is just above `SAFE_RATIO` times the stop band; the
    fastest is the action bound."""
    top=limits_of(oft);floor=stop_rates(oft)
    tier=lambda axis,share:max(getattr(floor,axis)*SAFE_RATIO*1.1,getattr(top,axis)*share)
    return [Axes(*(tier(axis,share) for axis in ('forward','down','turn'))) for share in (.3,.6,1.)]


def stick(value,floor,top):
    """A stick past its dead zone, from the floor of the axis to its top; inside the dead zone nothing."""
    magnitude=abs(value)
    if magnitude<=DEAD_ZONE:return 0.
    return math.copysign(floor+(top-floor)*min(1.,(magnitude-DEAD_ZONE)/(1-DEAD_ZONE)),value)


def wanted(inputs,tier,oft):
    """The rates the pilot asks for now."""
    top=limits_of(oft);floor=stop_rates(oft);low=lambda axis:getattr(floor,axis)*SAFE_RATIO*1.1
    forward=tier.forward if inputs.forward else 0.
    if abs(inputs.stick_forward)>DEAD_ZONE:forward=max(0.,stick(inputs.stick_forward,low('forward'),top.forward))
    vertical=(tier.down if inputs.up else 0.)-(tier.down if inputs.down else 0.)
    if abs(inputs.stick_vertical)>DEAD_ZONE:vertical=stick(inputs.stick_vertical,low('down'),top.down)
    turn=(tier.turn if inputs.right else 0.)-(tier.turn if inputs.left else 0.)
    if abs(inputs.stick_turn)>DEAD_ZONE:turn=stick(inputs.stick_turn,low('turn'),top.turn)
    return Axes(min(forward,top.forward),max(-top.down,min(top.down,-vertical)),max(-top.turn,min(top.turn,turn)))


@dataclass
class Smoother:
    """Eases the commanded rates toward what the pilot asks: each axis changes by at most its acceleration per second."""
    acceleration:Axes=field(default_factory=lambda:Axes(1.5,1.5,1.5))
    now:Axes=field(default_factory=Axes)

    def step(self,target,dt):
        ease=lambda axis:getattr(self.now,axis)+max(-getattr(self.acceleration,axis)*dt,min(getattr(self.acceleration,axis)*dt,getattr(target,axis)-getattr(self.now,axis)))
        self.now=Axes(ease('forward'),ease('down'),ease('turn'));return self.now

    def stop(self):self.now=Axes();return self.now


class TickLabel:
    """The commands flown between two decisions, as one action of the policy. `add` is called with the rates flown for
    `dt` seconds; `close` gives the action the policy would have had to give at the decision that opened the tick."""
    def __init__(self,oft):
        self.oft=oft;self.tick=oft['tick_s'];self.reset()

    def reset(self):self.forward=self.down=self.turn=self.elapsed=0.

    def add(self,rates,dt):
        self.forward+=rates.forward*dt;self.down+=rates.down*dt;self.turn+=rates.turn*dt;self.elapsed+=dt

    def close(self):
        """[forward m, down m, turn rad] per tick (scaled to the nominal tick when the tick ran long or short), clipped to the bounds."""
        scale=self.tick/self.elapsed if self.elapsed>0 else 0.;bounds=self.oft['action_bounds']
        clip=lambda value,pair:max(pair[0],min(pair[1],value))
        label=[clip(self.forward*scale,bounds['forward_m']),clip(self.down*scale,bounds['down_m']),clip(self.turn*scale,bounds['yaw_rad'])]
        self.reset();return label


def stop_ratio(action,oft):
    """Largest axis of an action over its stop band (as src/landing/stop_safety.stop_ratio): <= 1 is a stop."""
    bounds=oft['action_bounds'];limits=[oft['stop_threshold']*(pair[1]-pair[0]) for pair in bounds.values()]
    return max(abs(value)/limit for value,limit in zip(action,limits))


def mark(action,oft,terminal=False):
    """Why a recorded decision stays out of the label windows, or None."""
    if terminal:return None
    ratio=stop_ratio(action,oft)
    if ratio<=1.:return 'pause'
    if ratio<SAFE_RATIO:return 'below_safe'
    return None
