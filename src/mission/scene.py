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
from .geometry import mission_robot_config,relative_camera_pose,camera_metadata,select_surface


class OverviewScene:
    def __init__(self,world,drone,output):
        self.world,self.drone,self.output=world,drone,Path(output)
        self.bounds=world.get_3d_bounding_box('TemplateCube_Rounded_1',BoxAlignment.WORLD_AXIS)
        self.pose=dict(world.get_object_pose('TemplateCube_Rounded_1'))
        c,s=self.bounds['center'],self.bounds['size']
        self.anchor=[c['x'],c['y'],c['z']-s['z']/2]
        self.frames=OrderedDict();self.counter=0
        publish_json(self.output/'scene-geometry.json',{'object':'TemplateCube_Rounded_1','pose':self.pose,'bounds':self.bounds})

    def capture(self,state,view='top',height=50):
        height=max(18,min(65,float(height)))
        if view=='top':
            position=[self.anchor[0],self.anchor[1],self.anchor[2]-height];rpy=[0,-90,0]
        else:
            position=[self.anchor[0]-height*.7,self.anchor[1]+height*.7,self.anchor[2]-height]
            delta=np.subtract(self.anchor,position)
            rpy=[0,-np.degrees(np.arctan2(delta[2],np.linalg.norm(delta[:2]))),np.degrees(np.arctan2(delta[1],delta[0]))]
        if not self.drone.set_camera_pose('Overview',Pose(relative_camera_pose(position,rpy,state))):
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
                'image_sha256':hashlib.sha256(image.tobytes()).hexdigest()}
        self.frames[self.counter]=(values.copy(),meta)
        while len(self.frames)>4:self.frames.popitem(last=False)
        publish_image(self.output/packet['image_file'],image)
        publish_json(self.output/'overview.json',packet)
        return image,packet

    def select(self,frame_id,pixel):
        if frame_id not in self.frames:raise ValueError('Map refreshed; select on the current frame again')
        depth,meta=self.frames[frame_id]
        point=select_surface(pixel,depth,meta)
        return point, f'Overview depth-planar frame {frame_id}, pixel {tuple(pixel)}'


def prepare_mission_config(root,output):
    directory=Path(output)/'sim_config';directory.mkdir(parents=True,exist_ok=True)
    root=Path(root)
    robot=json.loads((root/'configs/robot_quadrotor_fastphysics.jsonc').read_text())
    (directory/'robot_quadrotor_fastphysics.jsonc').write_text(json.dumps(mission_robot_config(robot),indent=2))
    (directory/'scene_basic_drone.jsonc').write_text((root/'configs/scene_basic_drone.jsonc').read_text())
    return directory
