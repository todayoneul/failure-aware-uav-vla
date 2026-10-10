"""A scripted pilot for checking the landing evaluator. Not a policy, and nothing learned.

Whether the evaluator reads a landing on a cube correctly cannot be learned from a policy that may or may not manage one.
This pilot is told where to go (it reads the vehicle's true state and is given a point and a height), and flies there with
the same three action axes and through the same canonical loop as a policy: it stands where the policy stands, in
`policy.infer`, and ignores the two frames and the sentence it is handed. What it produces are flights whose outcome is
known beforehand, so that what the evaluator says about them can be checked.

    land    climb above the surface, fly over the aim point, come down on it, give the zero action once it stands
    hover   the same, but stop in the air over the aim point and never come down
    ram     fly straight at the aim point at the height the flight started at (into the side of something taller)
    hold    give the zero action from the first decision (a start that is only looked at)

The rates are the canonical teacher's (configs/targets/landing_pads.json, `teacher`): full descent down to its slow height
above the surface, its final descent rate below that.
"""
import math
from src.visual_search.episodes import relative_target
from src.aerovla_oft.spec import normalize_action,is_stop

MODES=('land','hover','ram','hold')
ARRIVE_M=.1          # the aim point is reached within this; the transit below never commands less than the stop band allows
GAIN=.5              # share of the remaining distance flown per decision when close: no overshoot at one decision of lag
FACE_DEG=20.         # the vehicle turns on the spot until the aim point is within this of straight ahead


class ControlledPilot:
    def __init__(self,env,oft,landing,aim_xy,top_m=0.,mode='land'):
        if mode not in MODES:raise ValueError(f'Unknown pilot mode: {mode!r}')
        self.env=env;self.oft=oft;self.aim=list(aim_xy);self.top=float(top_m);self.mode=mode;self.teacher=landing['teacher']
        (_,self.forward_max),(_,self.down_max),(_,self.yaw_max)=oft['action_bounds'].values()
        # Flown over at the height above the surface from which the canonical descent is committed.
        self.cruise=self.top+self.teacher['commit_height_m'];self.arrived=False;self.phases=[]

    def act(self):
        """(action, phase) from the vehicle's true state: the pilot's privilege, which no policy has."""
        env=self.env;state=env.last_state;bearing,distance,_=relative_target(state,self.aim);height=env.ground_z-state['position'][2]
        if self.mode=='hold' or env.touchdown is not None or env.hit:return [0.,0.,0.],'rest'
        turn=max(-self.yaw_max,min(self.yaw_max,math.radians(bearing)))
        if self.mode=='ram':return [self.forward_max if abs(bearing)<=FACE_DEG else 0.,0.,turn],'ram'
        if not self.arrived:
            if height<self.cruise-.25:return [0.,-min(self.down_max,self.cruise-height),turn if distance>ARRIVE_M else 0.],'climb'
            if distance>ARRIVE_M:
                forward=0. if abs(bearing)>FACE_DEG else max(.1,min(self.forward_max,GAIN*distance))
                return [forward,0.,turn],'transit'
            self.arrived=True
        if self.mode=='hover':return [0.,0.,0.],'hover'
        above=height-self.top
        return [0.,self.down_max if above>self.teacher['slow_height_m'] else self.teacher['final_descent_m'],0.],'descend'

    def infer(self,front,down,instruction,proprio=None):
        """Called as a policy is called. The frames and the sentence are not looked at."""
        action,phase=self.act()
        if not self.phases or self.phases[-1]!=phase:self.phases.append(phase)
        chunk=[[float(value) for value in action]]*self.oft['chunk_size']
        return {'prompt':'(controlled pilot: no prompt)','chunk':chunk,'normalised_chunk':[normalize_action(item,self.oft) for item in chunk],
                'inference_ms':0.,'stop':is_stop(action,self.oft),'proprio':None,'pilot_phase':phase}


def aim_point(pilot,surfaces):
    """The point a plan's pilot is sent to: an object's surface centre (or a named [x, y]), moved by an offset."""
    aim=pilot.get('aim');offset=pilot.get('offset_m',[0.,0.])
    # A pilot that holds from the first decision is sent nowhere.
    if aim is None:return [0.,0.],0.
    if isinstance(aim,str):
        surface=surfaces[aim];centre=surface.centre_xy;top=surface.top_m
    else:centre=aim;top=pilot.get('top_m',0.)
    return [centre[0]+offset[0],centre[1]+offset[1]],top


def compare(reading,expect):
    """Each expected value of a controlled flight against what the evaluator read: [(key, expected, read, same)].
    `collision: true` asks for a collision with anything; `landed_on: null` for no touchdown."""
    rows=[]
    for key,wanted in expect.items():
        got=reading.get(key)
        if key=='collision' and isinstance(wanted,bool):got=got is not None
        rows.append((key,wanted,got,got==wanted))
    return rows
