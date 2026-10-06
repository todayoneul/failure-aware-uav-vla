"""Mission requests share the small blur-control file without sharing its state."""
import copy
from src.failures.control import default_control, apply_key


def default_mission_control():
    return {**default_control(),'mission_request_id':0,'mission_request':None,'mission_requests':[],
            'overview_view':'elevated','overview_height':50}


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


def mission_key(state,key):
    if key in (ord('+'),ord('='),ord('-')):
        result=copy.deepcopy(state)
        result['overview_height']=max(18,min(65,state.get('overview_height',50)+(-5 if key!=ord('-') else 5)))
        return result
    if key in (ord('g'),ord('G'),ord('h'),ord('H'),ord('l'),ord('L'),ord('r'),ord('R')):
        action='reset' if chr(key).lower()=='r' else 'start'
        request={'action':action}
        if action=='start':request['type']={'g':'GO_TO','h':'GO_TO_AND_HOVER','l':'GO_TO_AND_LAND'}[chr(key).lower()]
        return enqueue(state,request)
    if key in (ord('v'),ord('V')):
        result=copy.deepcopy(state);result['overview_view']='elevated' if state.get('overview_view')=='top' else 'top'
        return result
    return apply_key(state,key)
