"""Mission requests share the small blur-control file without sharing its state."""
import copy
from src.failures.control import default_control, apply_key


GROUNDING='grounding-film'
INSTRUCTION_ONLY='instruction-only'
TASKS=('land','approach')
# Arrow keys as OpenCV reports them on Windows (waitKeyEx): they pan the map like W / S / A / D.
ARROWS={2490368:'w',2621440:'s',2424832:'a',2555904:'d'}


def default_mission_control(policy='legacy'):
    state={**default_control(),'mission_request_id':0,'mission_request':None,'mission_requests':[],
           'overview_view':'top','overview_zoom':1.,'overview_pan':[0,0],'overview_focus':'map',
           'direction_hint':True,'prompt_mode':'hint'}
    if policy==GROUNDING:
        # An AeroVLA-OFT policy is given the sentence alone: no prompt mode can be chosen, and the task is part of the sentence.
        state.update(policy=GROUNDING,direction_hint=False,prompt_mode=INSTRUCTION_ONLY,main_view='map',start_mode=False)
    return state


def enqueue(state,request):
    result=copy.deepcopy(state)
    request_id=state.get('mission_request_id',0)+1
    result['mission_request_id']=request_id;result['mission_request']=request
    result['mission_requests']=(state.get('mission_requests',[])+[{'id':request_id,'request':request}])[-16:]
    return result


def pending_requests(state,after):
    return [(item['id'],item['request']) for item in state.get('mission_requests',[]) if item['id']>after]


def request_selection(state, frame_id, pixel):
    return enqueue(state,{'action':'select','frame_id':int(frame_id),'pixel':list(map(int,pixel))})


def request_start_pose(state,frame_id,pixel,heading_pixel=None):
    """A start chosen on the map: the pixel pressed is where the vehicle stands, the pixel released at (after a drag) is what
    it faces. Without a drag the heading stays as it was."""
    request={'action':'start_pose','frame_id':int(frame_id),'pixel':list(map(int,pixel)),
             'heading_pixel':list(map(int,heading_pixel)) if heading_pixel is not None else None}
    return enqueue(state,request)


def pan(state,letter):
    result=copy.deepcopy(state);offset=list(state.get('overview_pan',[0,0]))
    axis,change={'w':(0,1),'s':(0,-1),'a':(1,-1),'d':(1,1)}[letter]
    offset[axis]+=change;result['overview_pan']=offset
    return result


def grounding_key(state,key):
    """Keys that mean something else, or nothing, when the policy is an AeroVLA-OFT checkpoint. None: handled as before."""
    letter=chr(key).lower() if 0<=key<256 else ''
    placing=bool(state.get('start_mode'))
    if letter=='m':return copy.deepcopy(state)          # the prompt is the sentence; there is no mode to change
    if letter=='t':return enqueue(state,{'action':'task'})   # LAND <-> APPROACH, for the next mission
    if letter=='s':
        # S: the map is clicked for the vehicle's start, not for a target, until S is pressed again. (S no longer pans the
        # map down for this policy; the Down arrow does.)
        result=enqueue(state,{'action':'start_mode','on':not placing});result['start_mode']=not placing
        return result
    if key in (ord('['),ord(']')) or letter in ('j','k'):
        # Height and heading of the start, while it is being placed; outside that mode the keys do nothing.
        if not placing:return copy.deepcopy(state)
        if letter in ('j','k'):return enqueue(state,{'action':'start_yaw','step':-1 if letter=='j' else 1})
        return enqueue(state,{'action':'start_height','step':-1 if key==ord('[') else 1})
    if letter in ('l','g','r'):
        # A preset, a mission or a reset ends the placing of a start; G and R use the start as it stands.
        result=enqueue(state,{'action':{'l':'launch','g':'start','r':'reset'}[letter]});result['start_mode']=False
        return result
    if letter=='f':
        result=copy.deepcopy(state);result.update(main_view='map',overview_pan=[0,0],overview_view='top',overview_zoom=1.,overview_focus='map')
        return result
    if letter=='c':
        # The simulator's chase camera fills the main panel; the map keeps its own view.
        result=copy.deepcopy(state);result['main_view']='chase'
        return result
    return None


def mission_key(state,key):
    if key in ARROWS:return pan(state,ARROWS[key])
    if state.get('policy')==GROUNDING:
        result=grounding_key(state,key)
        if result is not None:return result
    if key in (ord('+'),ord('='),ord('-')):
        result=copy.deepcopy(state)
        result['overview_zoom']=max(.5,min(12,state.get('overview_zoom',1.)*(1.25 if key!=ord('-') else .8)))
        return result
    if key in (ord('f'),ord('F'),ord('c'),ord('C')):
        result=copy.deepcopy(state)
        result.update(overview_pan=[0,0],overview_view='top',overview_zoom=1. if chr(key).lower()=='f' else 8.,
                      overview_focus='map' if chr(key).lower()=='f' else 'drone')
        return result
    if 0<=key<256 and chr(key).lower() in 'wasd':return pan(state,chr(key).lower())
    if key in (ord('g'),ord('G'),ord('r'),ord('R'),ord('n'),ord('N')):
        # One mission type: the model flies and ends the episode itself. N cycles named landmarks.
        return enqueue(state,{'action':{'g':'start','r':'reset','n':'landmark'}[chr(key).lower()]})
    if key in (ord('m'),ord('M')):
        # hint -> description only -> free-form instruction -> hint
        modes=('hint','description','instruction')
        current=state.get('prompt_mode','hint' if state.get('direction_hint',True) else 'description')
        result=copy.deepcopy(state);result['prompt_mode']=modes[(modes.index(current)+1)%len(modes)]
        result['direction_hint']=result['prompt_mode']=='hint'
        return result
    if key in (ord('v'),ord('V')):
        result=copy.deepcopy(state);result['overview_view']='elevated' if state.get('overview_view')=='top' else 'top'
        return result
    return apply_key(state,key)
