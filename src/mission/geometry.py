"""Depth-planar deprojection in Project AirSim's NED camera axes."""
import copy
import math
import numpy as np

def rotation_matrix(quaternion):
    q=np.asarray(quaternion,dtype=float)
    norm=np.linalg.norm(q)
    if q.shape!=(4,) or not np.isfinite(q).all() or norm<1e-12:raise ValueError('Invalid camera quaternion')
    x,y,z,w=q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def intrinsics(meta):
    width,height=meta['width'],meta['height']
    fov=meta['fov_deg']
    if width<=0 or height<=0 or not 0<fov<180: raise ValueError('Invalid camera intrinsics')
    focal=width/(2*math.tan(math.radians(fov)/2))
    return focal,width/2,height/2


def pixel_to_world(pixel, depth, meta):
    u,v=pixel
    if not 0<=u<meta['width'] or not 0<=v<meta['height'] or not math.isfinite(depth) or not 0<depth<65000:
        raise ValueError('Pixel/depth is outside the valid scene capture')
    focal,cx,cy=intrinsics(meta)
    # Camera x is optical forward; y is image right and z image down.
    # DepthPlanar is optical-axis distance, not a Euclidean ray length.
    local=np.array([depth,(u-cx)*depth/focal,(v-cy)*depth/focal])
    world=rotation_matrix(meta['orientation'])@local+np.asarray(meta['position'])
    if not np.isfinite(world).all(): raise ValueError('Nonfinite camera pose')
    return world.tolist()


def world_to_pixel(point, meta):
    local=rotation_matrix(meta['orientation']).T@(np.asarray(point)-np.asarray(meta['position']))
    if local[0]<=0: raise ValueError('Point is behind the overview camera')
    focal,cx,cy=intrinsics(meta)
    return [cx+focal*local[1]/local[0],cy+focal*local[2]/local[0]]


def camera_metadata(message, fov=80):
    return {'width':message['width'],'height':message['height'],'fov_deg':fov,
            'position':[message[k] for k in ('pos_x','pos_y','pos_z')],
            'orientation':[message[k] for k in ('rot_x','rot_y','rot_z','rot_w')],
            'time_stamp':message['time_stamp']}


def select_surface(pixel, depth, meta):
    u,v=map(int,pixel)
    if not 1<=u<depth.shape[1]-1 or not 1<=v<depth.shape[0]-1: raise ValueError('Click inside the map')
    point=pixel_to_world((u,v),float(depth[v,u]),meta)
    right=pixel_to_world((u+1,v),float(depth[v,u+1]),meta)
    down=pixel_to_world((u,v+1),float(depth[v+1,u]),meta)
    normal=np.cross(np.subtract(right,point),np.subtract(down,point))
    length=np.linalg.norm(normal)
    if length<1e-9 or abs(normal[2])/length<math.cos(math.radians(20)):
        raise ValueError('Choose a flat visible surface, away from edges')
    return point


def mission_robot_config(base):
    from scripts.demo_view_config import build_demo_robot
    robot=build_demo_robot(base)
    for s in robot['sensors']:
        if s['id'] in ('FrontCamera','DownCamera'):
            # Append debug-only planar depth; original RGB/settings/pose are untouched.
            s['capture-settings'].append(dict(s['capture-settings'][0],
                **{'image-type':1,'pixels-as-float':True,'streaming-enabled':False}))
        if s['id']=='Chase':
            s['capture-settings'][0].update(width=640,height=360)
    overview=copy.deepcopy(next(s for s in base['sensors'] if s['id']=='DownCamera'))
    overview.update(id='Overview',origin={'xyz':'0 0 -28','rpy-deg':'0 -90 0'},
                    gimbal={'lock-roll':False,'lock-pitch':False,'lock-yaw':False})
    rgb=dict(overview['capture-settings'][0],width=640,height=360,
             **{'fov-degrees':80,'streaming-enabled':False})
    overview['capture-settings']=[rgb,dict(rgb,**{'image-type':1,'pixels-as-float':True})]
    overview['capture-interval']=.1
    robot['sensors'].append(overview)
    return robot


def relative_camera_pose(world_position, world_rpy, state):
    from scipy.spatial.transform import Rotation  # Existing WSL inference dependency; not needed by Windows UI.
    body=Rotation.from_quat(state['orientation'])
    translation=body.inv().apply(np.asarray(world_position)-np.asarray(state['position']))
    rotation=(body.inv()*Rotation.from_euler('xyz',world_rpy,degrees=True)).as_quat()
    return {'translation':dict(zip('xyz',map(float,translation))),
            'rotation':dict(zip('xyzw',map(float,rotation)))}
