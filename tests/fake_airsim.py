"""A vehicle over flat ground with boxes on it: as much of the simulator as the canonical episode loop asks for.

Used to fly `scripts.visual_search.run_episode` in tests without a simulator or a model. The vehicle moves by one tick per
velocity command, stands still once it touches something (as FastPhysics holds it), and reports that contact on the
collision topic. A `ScriptedModel` stands in for the policy and records exactly what it is called with.
"""
import asyncio
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from src.visual_search.maps import MapGeometry

REST=.19
TICK=.5


class FakeSim:
    def __init__(self,map_config,layout):
        self.map=map_config;self.geometry=MapGeometry(map_config,layout);self.ground=map_config['ground_z']
        self.stamp=1_000_000;self.callbacks={};self.calls=[];self.spawned=[];self.loads=0;self.captures=0
        self.x=self.y=0.;self.z=self.ground-REST;self.yaw=0.;self.velocity=[0.,0.,0.];self.held=False

    def now(self):
        self.stamp+=1_000_000;return self.stamp

    def load(self,origin):
        x,y,_=map(float,origin['xyz'].split());self.x,self.y=x,y;self.z=self.ground-REST
        self.yaw=math.radians(float(origin['rpy-deg'].split()[2]));self.velocity=[0.,0.,0.];self.held=False;self.loads+=1

    def objects_at(self,x,y):
        """(simulator name, top z) of every object whose footprint holds a point."""
        found=[]
        for target,item in self.geometry.objects.items():
            (cx,cy),(sx,sy,height)=item['centre'],item['size_m']
            if abs(x-cx)<=sx/2 and abs(y-cy)<=sy/2:
                spec=self.map['objects'][target];found.append((spec.get('object') or spec.get('native'),self.ground-height))
        return found

    def contact(self,name):
        self.held=True;self.velocity=[0.,0.,0.]
        event={'time_stamp':self.now(),'object_name':name,'position':{'x':self.x,'y':self.y,'z':self.z}}
        if 'collision' in self.callbacks:self.callbacks['collision']('collision',event)

    def move(self,vx,vy,z,yaw=None):
        if self.held:return
        if yaw is not None:self.yaw=yaw
        self.x+=vx*TICK;self.y+=vy*TICK;before=self.z
        for name,top in self.objects_at(self.x,self.y):
            if before>top-REST+1e-6:
                # Flown into the side of something taller than the vehicle is high.
                self.velocity=[vx,vy,0.];return self.contact(name)
            if z>=top-REST:
                # Come down onto its top.
                self.z=top-REST;self.velocity=[vx,vy,(self.z-before)/TICK];return self.contact(name)
        self.z=z;self.velocity=[vx,vy,(z-before)/TICK]

    def kinematics(self):
        half=self.yaw/2
        return {'pose':{'position':{'x':self.x,'y':self.y,'z':self.z},'orientation':{'x':0.,'y':0.,'z':math.sin(half),'w':math.cos(half)}},
                'twist':{'linear':dict(zip('xyz',self.velocity))},'time_stamp':self.now()}

    def camera(self,messages):
        """Stands in for `matched_camera_images`: a frame no other capture has, a depth map with nothing in the way, the pose."""
        self.captures+=1;sensor=messages[0]['sensor']
        image=np.full((256,256,3),40+self.captures%200,dtype=np.uint8);image[0,0]=[self.captures%256,(self.captures//256)%256,7 if sensor=='FrontCamera' else 9]
        turn=Rotation.from_euler('z',self.yaw)
        if sensor=='DownCamera':turn=turn*Rotation.from_euler('y',-90,degrees=True)
        meta={'width':256,'height':256,'fov_deg':90,'position':[self.x,self.y,self.z],'orientation':turn.as_quat().tolist(),'time_stamp':self.now()}
        return image,np.full((256,256),1000.,dtype=np.float32),meta


class FakeClient:
    def __init__(self,sim):self.sim=sim;self.socket_services=type('Services',(),{'recv_timeout':0})()
    def connect(self):pass
    def disconnect(self):self.sim.calls.append('disconnect')
    def subscribe(self,topic,callback):self.sim.callbacks[topic]=callback


def world_class(sim):
    class FakeWorld:
        def __init__(self,client,scene,delay_after_load_sec=0,sim_config_path=None):
            sim.load(json.loads((Path(sim_config_path)/scene).read_text())['actors'][0]['origin'])
        def destroy_all_spawned_objects(self):sim.spawned.clear()
        def spawn_object(self,name,*_):sim.spawned.append(name)
        def spawn_object_from_file(self,name,*_):sim.spawned.append(name)
        def set_time_of_day(self,*_):pass
        def list_objects(self,_):return []
        def get_3d_bounding_box(self,name,_):
            x,y=sim.map['native_objects'][name]['center'];return {'center':{'x':x,'y':y,'z':0.},'size':{'x':1.,'y':1.,'z':1.}}
    return FakeWorld


def drone_class(sim):
    class FakeDrone:
        def __init__(self,*_):
            self.sensors={'Chase':{'scene_camera':'chase'}};self.robot_info={'collision_info':'collision'}
        def get_ground_truth_kinematics(self):return sim.kinematics()
        def enable_api_control(self):sim.calls.append('enable')
        def disable_api_control(self):sim.calls.append('disable')
        def arm(self):sim.calls.append('arm')
        def disarm(self):sim.calls.append('disarm');return True
        def get_landed_state(self):return 0 if sim.held else 1
        def get_images(self,sensor,_):return {0:{'sensor':sensor},1:{'sensor':sensor}}
        def set_camera_pose(self,*_):return True
        async def _done(self,value=True):
            async def command():return value
            return asyncio.create_task(command())
        async def takeoff_async(self,**_):
            sim.calls.append('takeoff');sim.move(0.,0.,sim.z-1.);return await self._done()
        async def hover_async(self):
            sim.calls.append('hover');sim.velocity=[0.,0.,0.];return await self._done()
        async def move_by_velocity_z_async(self,vx,vy,z,duration=None,yaw_is_rate=False,yaw=None):
            sim.move(vx,vy,z,yaw);return await self._done()
        async def land_async(self,**_):
            sim.calls.append('land_async');raise AssertionError('the simulator landing routine must not be called')
    return FakeDrone


class ScriptedModel:
    """Stands in for AeroVLA-OFT. It is called as the policy is called, keeps every argument, and answers from a script."""
    def __init__(self,script,oft):self.script=script;self.oft=oft;self.calls=[]

    def infer(self,*arguments,**keywords):
        from src.aerovla_oft.spec import normalize_action,is_stop
        self.calls.append((arguments,keywords));action=[float(v) for v in self.script(len(self.calls)-1)]
        chunk=[action]*self.oft['chunk_size']
        return {'prompt':f'<image>\n{arguments[2].strip()}\nAction: ','chunk':chunk,'normalised_chunk':[normalize_action(a,self.oft) for a in chunk],
                'inference_ms':1.,'stop':is_stop(action,self.oft),'proprio':None}


def fly_to(sim,point,descend=False,stop=True,yaw_max=.21,arrive_m=12.):
    """A script that turns to a point and flies to it, reading the fake vehicle's own state. A test's stand-in for a policy
    that has found its target; what it knows comes from the test, not from anything a mission passes on."""
    def script(_):
        dx,dy=point[0]-sim.x,point[1]-sim.y;distance=math.hypot(dx,dy)
        turn=math.atan2(math.sin(math.atan2(dy,dx)-sim.yaw),math.cos(math.atan2(dy,dx)-sim.yaw))
        if abs(turn)>.02 and distance>1.5:return [0.,0.,max(-yaw_max,min(yaw_max,turn))]
        if descend:
            if distance>1.:return [min(1.,distance),0.,0.]
            return [0.,.25,0.]
        if distance>arrive_m:return [min(1.,distance-arrive_m+.05),0.,0.]
        return [0.,0.,0.] if stop else [0.,0.,yaw_max]
    return script
