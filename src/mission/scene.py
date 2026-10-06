"""Authoritative scene geometry and matched overview snapshots."""
import json
import hashlib
import time
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
from projectairsim.types import BoxAlignment,Pose
from projectairsim.utils import unpack_image
from src.integration.blur_demo_support import publish_json,publish_image
from .geometry import mission_robot_config,relative_camera_pose,camera_metadata,select_surface,pixel_to_world
from .bounds import scene_bounds,fit_camera
from .landmarks import load_landmarks,landmark_at

LANDMARKS=Path(__file__).resolve().parents[2]/'configs/landmarks.json'


def scene_records(world,names=None):
    records=[]
    for name in (names if names is not None else world.list_objects('.*')):
        record={'name':name}
        try:
            record.update(bbox=world.get_3d_bounding_box(name,BoxAlignment.WORLD_AXIS),pose=dict(world.get_object_pose(name)))
        except Exception as error:record['error']=str(error)
        records.append(record)
    return records


class OverviewScene:
    def __init__(self,world,drone,output):
        self.world,self.drone,self.output=world,drone,Path(output)
        records=scene_records(world)
        self.region=scene_bounds(records)
        self.landmarks=load_landmarks(LANDMARKS,records)
        self.frames=OrderedDict();self.counter=0
        publish_json(self.output/'scene-geometry.json',{'bounds':self.region,'landmarks':self.landmarks,'objects':records})

    def capture(self,state,view='top',zoom=1.,pan=(0,0),focus='map'):
        region=self.region
        if focus=='drone':
            center=(np.asarray(region['minimum'])+region['maximum'])/2
            shift=np.array([state['position'][0]-center[0],state['position'][1]-center[1],0])
            region={**region,'minimum':(np.asarray(region['minimum'])+shift).tolist(),
                    'maximum':(np.asarray(region['maximum'])+shift).tolist()}
        rig=fit_camera(region,view,max(.5,min(12,float(zoom))),pan)
        if not self.drone.set_camera_pose('Overview',Pose(relative_camera_pose(rig['position'],rig['rpy'],state))):
            raise RuntimeError('Overview pose update failed')
        messages=self.drone.get_images('Overview',[0,1])
        rgb,depth=messages[0],messages[1]
        if rgb['time_stamp']!=depth['time_stamp'] or any(rgb[k]!=depth[k] for k in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z','rot_w')):
            raise RuntimeError('Overview RGB/depth are not one capture')
        image=unpack_image(rgb);values=unpack_image(depth)
        if image.shape!=(360,640,3) or values.shape!=(360,640):raise RuntimeError('Unexpected Overview contract')
        self.counter+=1
        meta=camera_metadata(rgb)
        packet={'frame_id':self.counter,'camera':meta,'view':view,'image_file':f'overview_{self.counter%2}.png',
                'image_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'bounds':self.region,'rig':rig}
        self.frames[self.counter]=(values.copy(),meta)
        while len(self.frames)>4:self.frames.popitem(last=False)
        publish_image(self.output/packet['image_file'],image)
        publish_json(self.output/'overview.json',packet)
        return image,packet

    def landmark_at(self,frame_id,pixel):
        """A click on a configured landmark selects the object itself, whatever its surface slope."""
        if frame_id not in self.frames:raise ValueError('Map refreshed; select on the current frame again')
        depth,meta=self.frames[frame_id];u,v=map(int,pixel)
        try:point=pixel_to_world((u,v),float(depth[v,u]),meta)
        except (ValueError,IndexError):return None
        return landmark_at(self.landmarks,point)

    def select(self,frame_id,pixel):
        if frame_id not in self.frames:raise ValueError('Map refreshed; select on the current frame again')
        depth,meta=self.frames[frame_id]
        point=select_surface(pixel,depth,meta)
        if hasattr(self,'region') and any(not self.region['minimum'][i]-5<=point[i]<=self.region['maximum'][i]+5 for i in (0,1)):
            raise ValueError('Point is outside the bounded Blocks geometry area (+5m margin)')
        return point, f'Overview depth-planar frame {frame_id}, pixel {tuple(pixel)}'


def prepare_mission_config(root,output):
    directory=Path(output)/'sim_config';directory.mkdir(parents=True,exist_ok=True)
    root=Path(root)
    robot=json.loads((root/'configs/robot_quadrotor_fastphysics.jsonc').read_text())
    (directory/'robot_quadrotor_fastphysics.jsonc').write_text(json.dumps(mission_robot_config(robot),indent=2))
    (directory/'scene_basic_drone.jsonc').write_text((root/'configs/scene_basic_drone.jsonc').read_text())
    return directory
