"""Point visibility and coordinate-to-prompt evidence; no steering or recognition."""
import math
import cv2
import numpy as np
from .geometry import rotation_matrix,intrinsics


def grounding_report(state,goal,surface,instruction,direction_hint=True,landmark=None):
    # Use the exact existing model adapter; this lazy import is WSL-side only.
    from src.integration.projectairsim_observation_adapter import semantic_direction,make_prompt
    relative=np.asarray(goal)-np.asarray(state['position'])
    body=rotation_matrix(state['orientation']).T@relative
    mode=('DIRECTION HINT + ' if direction_hint else 'NO HINT + ')+('LANDMARK DESCRIPTION' if landmark else 'GENERIC DESCRIPTION')
    return {'mode':mode,'world_surface':list(surface),'navigation_goal':list(goal),
            'relative_world':relative.tolist(),'relative_body':body.tolist(),
            'horizontal_distance':float(np.linalg.norm(relative[:2])),
            'bearing_deg':math.degrees(math.atan2(relative[1],relative[0])),
            'body_bearing_deg':math.degrees(math.atan2(body[1],body[0])),
            'semantic_direction':semantic_direction(state,goal).strip(),
            'prompt':make_prompt(state,goal,instruction,direction_hint),'prompt_scope':'preview',
            'decision_pose':state,'direction_hint':bool(direction_hint),'landmark':landmark,
            'vla_receives':{'exact_xyz_tokens':False,'distance_token':False,'overview_image':False,'red_x_marker':False}}


def target_visibility(point,meta,depth=None):
    local=rotation_matrix(meta['orientation']).T@(np.asarray(point)-np.asarray(meta['position']))
    result={'status':'BEHIND CAMERA','pixel':None,'in_fov':False,'visible':False,
            'expected_depth_m':float(local[0]),'camera':meta}
    if not np.isfinite(local).all():result['status']='INVALID PROJECTION';return result
    if local[0]<=0:return result
    focal,cx,cy=intrinsics(meta);pixel=[float(cx+focal*local[1]/local[0]),float(cy+focal*local[2]/local[0])]
    result['pixel']=pixel
    if not 0<=pixel[0]<meta['width'] or not 0<=pixel[1]<meta['height']:
        result['status']='OUT OF FOV';return result
    result['in_fov']=True;result['status']='IN FOV / DEPTH UNAVAILABLE';result['visible']=None
    if depth is None:return result
    u,v=map(lambda value:int(round(value)),pixel)
    u=min(meta['width']-1,u);v=min(meta['height']-1,v)
    observed=float(depth[v,u]);result['observed_depth_m']=observed if math.isfinite(observed) else None
    if not math.isfinite(observed) or not 0<observed<65000:result['status']='UNKNOWN DEPTH';return result
    tolerance=max(.12,.004*local[0]);result['tolerance_m']=float(tolerance)
    delta=observed-local[0];result['depth_residual_m']=float(delta)
    if delta < -tolerance:result['status']='OCCLUDED';result['visible']=False
    elif abs(delta)<=tolerance:result['status']='VISIBLE';result['visible']=True
    else:result['status']='DEPTH MISMATCH'
    return result


def target_overlay(frame,visibility):
    debug=frame.copy()
    if visibility and visibility.get('in_fov') and visibility.get('pixel'):
        pixel=tuple(int(round(v)) for v in visibility['pixel'])
        color=(45,165,40) if visibility.get('visible') else (30,145,235)
        cv2.drawMarker(debug,pixel,color,cv2.MARKER_CROSS,13,1,cv2.LINE_AA)
    return debug


def matched_camera_images(messages):
    """One camera's RGB/depth; raw RGB remains the unchanged model observation."""
    from projectairsim.utils import unpack_image
    from .geometry import camera_metadata
    rgb,depth=messages[0],messages[1]
    fields=('time_stamp','pos_x','pos_y','pos_z','rot_x','rot_y','rot_z','rot_w')
    if any(rgb[k]!=depth[k] for k in fields):raise RuntimeError('Debug RGB/depth mismatch')
    image=unpack_image(rgb);values=unpack_image(depth)
    if rgb['encoding']!='BGR' or image.shape!=(256,256,3) or depth['encoding']!='16FC1' or values.shape!=(256,256):
        raise RuntimeError('AI RGB / debug depth contract changed')
    return image,values,camera_metadata(rgb,fov=90)
