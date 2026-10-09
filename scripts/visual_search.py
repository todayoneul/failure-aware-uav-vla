"""Visual search on Project AirSim: evaluate a policy, or record the privileged teacher as a dataset.

Policies see Front/Down RGB and one sentence (`policy_inputs`). The target centre is used here to place
the vehicle, to score, and by the teacher; it is never passed to a learned policy.

A sentence either asks to approach an object (stop in the air near it) or to land on a pad. Landing is
flown with the same three action axes as everything else: there is no call to the simulator's own
landing routine anywhere in an episode. Touching a pad from above ends the flight where it stands
(the simulator holds the vehicle); touching anything else is a collision.

  --policy baseline   AeroVLA NF4, instruction only, its own step actions and LAND (--flight continuous
                      hands each action over in flight instead of stopping after it)
  --policy teacher    scripted expert; with --record it writes an AeroVLA-format dataset
  --policy oft        AeroVLA-OFT checkpoint, short action chunk, receding horizon

Episodes come either from the pilot's cases (--cases A B C ...) or from a plan file written by
scripts/plan_generalization.py (--plan FILE --set NAME), which fixes the map, the layout, the object
and the start state of every episode.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from projectairsim import ProjectAirSimClient,World,Drone
from projectairsim.types import Pose,Vector3,Quaternion,BoxAlignment
from src.integration.projectairsim_observation_adapter import adapt_state
from src.integration.projectairsim_action_adapter import convert_action,execute_action,fly_until_handoff,flight_limits
from src.mission.scene import prepare_mission_config
from src.mission.grounding import target_visibility,matched_camera_images
from src.mission.flight import climb_to
from src.visual_search.episodes import load_config,make_episode,relative_target,policy_inputs,Teacher
from src.visual_search.generalization import SearchTeacher,STOP_STATES,Pushes
from src.visual_search.maps import load_map,MapGeometry,scene_objects,load_landing,owner_of,lighting
from src.visual_search.finalizer import LandingFinalizer
from src.visual_search.shapes import mesh
from src.aerovla_oft.spec import load_config as load_oft_config,proprio_vector,is_stop

MANIFEST=ROOT/'outputs/integration/model-downloads.json'
LEVELS={'G0':'G0 - seen start','G1':'G1 - unseen start','G2':'G2 - unseen object','G3':'G3 - held-out scene','G4':'G4 - held-out scene and object',
        'P':'G1 - unseen words','S':'G1 - side probe','L1':'L1 - start 40-55 m','L2':'L2 - start 55-70 m','L3':'L3 - start 70-90 m',
        'Q1':'Q1 - basic grounding','Q2':'Q2 - colour distractor','Q3':'Q3 - shape distractor','Q4':'Q4 - position swap','Q5':'Q5 - query swap',
        'Q6':'Q6 - approach or land','Q7':'Q7 - far grounding','Q8':'Q8 - landing','Q8R':'Q8 - landing, red pad','Q10':'Q10 - unseen words',
        'QS':'seen scene, held-out layout'}


def digest(frame):return hashlib.sha256(frame.tobytes()).hexdigest()


def pose_at(x,y,z):
    return Pose({'translation':Vector3({'x':x,'y':y,'z':z}),'rotation':Quaternion({'w':1,'x':0,'y':0,'z':0}),'frame_id':'DEFAULT_ID'})


class SearchEnv:
    def __init__(self,client,config,oft,output):
        self.client,self.config,self.oft,self.output=client,config,oft,Path(output)
        self.maps={};self.verified=set();self.meshes={};self.landing=load_landing()
        self.limits=flight_limits(json.loads((ROOT/'configs/flight_limits.json').read_text()))

    def map(self,name):
        if name not in self.maps:self.maps[name]=load_map(name)
        return self.maps[name]

    def verify_natives(self,map_config):
        """The map file's native objects must be where the simulator says they are."""
        for name,audited in map_config.get('native_objects',{}).items():
            box=self.world.get_3d_bounding_box(name,BoxAlignment.WORLD_AXIS)
            if math.hypot(box['center']['x']-audited['center'][0],box['center']['y']-audited['center'][1])>.5:
                raise RuntimeError(f'{name} is not where configs/maps/{map_config["id"]} says; run scripts/audit_map_obstacles.py again')
        self.verified.add(map_config['id'])

    def build_scene(self,map_config,layout):
        """Add the layout's structures and objects, and set the map's lighting."""
        self.world.destroy_all_spawned_objects()
        for item in scene_objects(map_config,layout):
            x,y=item['position'];z=map_config['ground_z']-item['size_m'][2]/2-item['lift_m']
            if 'asset' in item:self.world.spawn_object(item['name'],item['asset'],pose_at(x,y,z),item['scale'],False)
            else:
                key=(item['shape'],tuple(item['size_m']),tuple(item['color']))
                if key not in self.meshes:self.meshes[key]=mesh(item['shape'],item['size_m'],item['color'],item['name'],**item.get('detail',{}))
                self.world.spawn_object_from_file(item['name'],'gltf',self.meshes[key],True,pose_at(x,y,z),[1,1,1],False)
        # Loading a scene puts the sun back to its default, so a map's lighting is set again for every episode.
        light=lighting(map_config,layout)
        if light:self.world.set_time_of_day(True,light['time_of_day'],False,1.,1.,True)

    async def reset(self,episode):
        map_config=self.map(episode.get('map','blocks'));layout=episode.get('layout','pilot')
        directory=prepare_mission_config(ROOT,self.output)
        scene=json.loads((directory/'scene_basic_drone.jsonc').read_text())
        x,y=episode['start_xy']
        scene['actors'][0]['origin']={'xyz':f'{x} {y} -3.0','rpy-deg':f'0 0 {episode["start_yaw_deg"]}'}
        (directory/'scene_basic_drone.jsonc').write_text(json.dumps(scene,indent=2))
        self.world=World(self.client,'scene_basic_drone.jsonc',delay_after_load_sec=2,sim_config_path=str(directory))
        if map_config['id'] not in self.verified:self.verify_natives(map_config)
        self.build_scene(map_config,layout)
        self.drone=Drone(self.client,self.world,'Drone1');self.collisions=[]
        self.client.subscribe(self.drone.robot_info['collision_info'],lambda _,message:self.collisions.append(message))
        self.geometry=MapGeometry(map_config,layout);self.target=episode['target'];ground=map_config['ground_z']
        self.map_config=map_config;self.layout=layout;self.task=episode.get('task','approach');self.scanned=0;self.touchdown=None;self.hit=None
        self.contact_event=False
        item=self.geometry.objects[self.target];self.surface=item['size_m'][2] if item['landable'] else 0.
        self.corners=[[item['centre'][0]+dx*item['size_m'][0]/2,item['centre'][1]+dy*item['size_m'][1]/2,ground-h*item['size_m'][2]]
                      for dx in (-1,1) for dy in (-1,1) for h in (0,1)]
        # Scored point: the object's centre at half its height. Sight is tested at several heights because
        # the vehicle's own arms cross the middle rows of the front view.
        self.centre=[item['centre'][0],item['centre'][1],ground-item['size_m'][2]/2]
        self.sight_points=[[point[0],point[1],ground-point[2]] for point in self.geometry.sight_points(self.target)]
        self.ceiling=map_config['altitude']['ceiling_m']
        rest=adapt_state(self.drone.get_ground_truth_kinematics());self.ground_z=rest['position'][2]
        self.drone.enable_api_control();self.drone.arm()
        await asyncio.wait_for(await self.drone.takeoff_async(timeout_sec=20),timeout=25)
        await climb_to(self.drone,self.ground_z-episode.get('start_height_m',self.config['cruise_height_m']),2.);await asyncio.sleep(.8)
        sample=self.drone.get_ground_truth_kinematics();self.flight_stamp=sample['time_stamp']
        self.setpoint_z=adapt_state(sample)['position'][2];self.previous_yaw=None;self.previous_stamp=None

    def contacts(self):
        """Read the collision events that arrived since the last call: a touchdown on a pad, or a collision.

        A pad (or its marking) touched while the vehicle is above its top is a touchdown; the first one is
        kept with the vehicle's place at that moment. Anything else touched is a collision."""
        tolerance=self.landing['touchdown']['from_above_tolerance_m']
        events=self.collisions[self.scanned:];self.scanned+=len(events)
        # For the executor: was any contact reported since the last call? What was touched is not passed on to it.
        self.contact_event=any(event.get('time_stamp',0)>self.flight_stamp for event in events)
        for event in events:
            if event.get('time_stamp',0)<=self.flight_stamp or self.hit:continue
            owner=owner_of(self.map_config,self.layout,event.get('object_name',''))
            top=self.geometry.objects[owner]['size_m'][2] if owner and self.geometry.objects[owner]['landable'] else None
            # Heights are counted from where the vehicle rested at the start, so standing on a pad reads as the pad's height.
            if top is not None and self.ground_z-event['position']['z']>=top-tolerance:
                if self.touchdown is None:
                    centre=self.geometry.objects[owner]['centre']
                    self.touchdown={'object':owner,'position':[event['position']['x'],event['position']['y'],event['position']['z']],
                                    'offset_m':[event['position']['x']-centre[0],event['position']['y']-centre[1]],'time_stamp':event['time_stamp']}
            else:self.hit=event.get('object_name') or 'unknown'
        return self.touchdown,self.hit

    def collided(self):return self.contacts()[1] is not None

    def resting(self,state):
        """The pad the vehicle stands on now, or None. The simulator reports one contact per touchdown, not a stream while the
        vehicle rests, so this is read from the vehicle itself: at the height it touched down at and not moving vertically.
        A vehicle told to climb leaves the pad again; the first touchdown stays on record either way."""
        if self.touchdown is None:return None
        still=abs(state['position'][2]-self.touchdown['position'][2])<=.03 and abs(state['velocity'][2])<=.05
        return self.touchdown['object'] if still else None

    def apparent(self,meta):
        """Size of the target's bounding box in one camera's image, in pixels and as a share of the image. Occlusion is ignored;
        this is how large the object could look, kept with the log and never shown to a policy."""
        pixels=[item['pixel'] for item in (target_visibility(corner,meta) for corner in self.corners) if item['pixel']]
        if len(pixels)<len(self.corners):return None
        width,height=meta['width'],meta['height']
        x0,x1=max(0.,min(p[0] for p in pixels)),min(float(width),max(p[0] for p in pixels))
        y0,y1=max(0.,min(p[1] for p in pixels)),min(float(height),max(p[1] for p in pixels))
        if x1<=x0 or y1<=y0:return None
        return {'width_px':round(x1-x0,1),'height_px':round(y1-y0,1),'area_px':round((x1-x0)*(y1-y0),1),'image_fraction':round((x1-x0)*(y1-y0)/(width*height),5)}

    def observe(self):
        """Raw frames for the policy plus privileged target geometry for the teacher and the score."""
        frames={};visibility={};sizes={}
        for sensor,name in (('FrontCamera','front'),('DownCamera','down')):
            for attempt in range(4):
                # A camera request now and then comes back empty; ask again rather than lose the episode.
                messages=self.drone.get_images(sensor,[0,1])
                if 0 in messages and 1 in messages:break
                time.sleep(.1)
            image,depth,meta=matched_camera_images(messages)
            frames[name]=image;items=[target_visibility(point,meta,depth) for point in self.sight_points]
            seen=any(item['in_fov'] and item.get('observed_depth_m') is not None
                     and item['observed_depth_m']>=item['expected_depth_m']-self.config['visibility_margin_m'] for item in items)
            visibility[name]={'in_fov':any(item['in_fov'] for item in items),'seen':bool(seen),'pixel':items[1]['pixel']}
            sizes[name]=self.apparent(meta)
        sample=self.drone.get_ground_truth_kinematics();state=adapt_state(sample)
        bearing,distance,yaw=relative_target(state,self.centre)
        rate=0.
        if self.previous_yaw is not None and sample['time_stamp']>self.previous_stamp:
            delta=math.atan2(math.sin(yaw-self.previous_yaw),math.cos(yaw-self.previous_yaw))
            rate=delta/((sample['time_stamp']-self.previous_stamp)/1e9)
        self.previous_yaw,self.previous_stamp=yaw,sample['time_stamp']
        settings=self.config['generalization'];height=self.ground_z-state['position'][2]
        ahead=self.geometry.ahead(state['position'][0],state['position'][1],yaw,height,settings['ahead_reach_m'],
                                  settings['corridor_clearance_m'],ignore=(self.target,))
        return {'frames':frames,'state':state,'yaw':yaw,'yaw_rate':rate,'stamp':sample['time_stamp'],
                'privileged':{'bearing_deg':bearing,'distance_m':distance,'front':visibility['front'],'down':visibility['down'],
                              'ahead_m':ahead,'height_m':height,'over':self.geometry.over(state['position'][0],state['position'][1],self.target),
                              'surface_m':self.surface,'size':sizes}}

    async def tick(self,action,yaw):
        """One small continuous motion; the command outlasts the tick so the next one takes over in flight."""
        forward,down,turn=action;tick=self.oft['tick_s'];heading=yaw+turn
        self.setpoint_z=max(self.ground_z-self.limits['maximum_clearance_m'],min(self.ground_z-self.limits['minimum_clearance_m'],self.setpoint_z+down))
        await self.drone.move_by_velocity_z_async(forward/tick*math.cos(heading),forward/tick*math.sin(heading),self.setpoint_z,
                                                  duration=tick*3,yaw_is_rate=False,yaw=heading)

    def others(self,position):
        """Horizontal distance from a position to every object of the layout other than the target."""
        return {name:math.hypot(item['centre'][0]-position[0],item['centre'][1]-position[1])
                for name,item in self.geometry.objects.items() if name!=self.target}

    async def close(self):
        try:
            await asyncio.wait_for(await self.drone.hover_async(),timeout=10)
            self.drone.disarm();self.drone.disable_api_control()
        except Exception:pass

    def disarm(self):
        """Motors off after a latched landing. No hover first: the vehicle stands on a surface."""
        try:
            self.drone.disarm();self.drone.disable_api_control();return True
        except Exception:return False


def landing_summary(episode,steps,config,landing,stopped,touchdown,hit,others,finalizer=None):
    """What an episode did about its target and, for a landing, how far through the landing it got.

    A landing is read three ways, kept apart: touchdown (the named pad, inside its landing region, a soft contact),
    stable physical landing (and standing still on it for the finalizer's duration), and strict policy zero-action
    (and the policy gave the zero action by itself). With the finalizer in use the result of the mission is the
    system's: a stable physical landing on the named pad that the finalizer latched and disarmed on."""
    radius=config['success_radius_m'];stage=landing['stages'];rule=landing['touchdown'];task=episode.get('task','approach');last=steps[-1]
    distances=[s['distance_m'] for s in steps];seen=[bool(s['front_seen'] or s['down_seen']) for s in steps]
    # The object the vehicle ended at: the one it stands on, or else the nearest one within the success radius.
    near=sorted((distance,name) for name,distance in {**(others or {}),episode['target']:last['distance_m']}.items() if distance<=radius)
    selected=touchdown['object'] if touchdown else near[0][1] if near and stopped else None
    over=[index for index,s in enumerate(steps) if s.get('over')]
    descent=bool(over) and min(s['height_m'] for s in steps[over[0]:])<=steps[over[0]]['height_m']-stage['descent_started_m']
    result={'task':task,'selected':selected,'correct_target':selected==episode['target'],'wrong_target':selected is not None and selected!=episode['target'],
            'landed':touchdown is not None,'landed_on':touchdown['object'] if touchdown else None,'hit':hit,
            'stages':{'target_acquired':any(seen),'correct_target':selected==episode['target'],'approached':min(distances)<=stage['approach_m'],
                      'down_camera_aligned':any(s.get('over') and s['distance_m']<=stage['aligned_m'] for s in steps),
                      'descent_started':descent,'touchdown':bool(touchdown and touchdown['object']==episode['target']),'self_stop':bool(stopped)}}
    if touchdown:
        # The speed of the first touchdown: what the vehicle was doing at the last decision before it.
        first=next((index for index,s in enumerate(steps) if s.get('landed')),len(steps)-1)
        velocity=steps[max(0,first-1)].get('velocity',[0.,0.,0.])
        error=math.hypot(*touchdown['offset_m'])
        result['touchdown']={'object':touchdown['object'],'offset_m':touchdown['offset_m'],'horizontal_error_m':error,
                             'inside_region':max(abs(v) for v in touchdown['offset_m'])<=rule['landing_region_half_width_m'],
                             'vertical_speed_mps':velocity[2],'horizontal_speed_mps':math.hypot(velocity[0],velocity[1]),
                             'step':first,'left_the_pad_again':any(not s.get('landed') for s in steps[first:])}
        result['touchdown']['soft']=(abs(velocity[2])<=rule['max_vertical_speed_mps'] and result['touchdown']['horizontal_speed_mps']<=rule['max_horizontal_speed_mps'])
    if task=='land':
        down=result.get('touchdown')
        # ... and still standing on it when the flight ends (a vehicle that hopped off and stopped in the air has not landed).
        result['land_success']=bool(down and down['object']==episode['target'] and down['inside_region'] and down['soft'] and stopped and not hit
                                    and last.get('landed'))
        result['stages']['land_success']=result['land_success']
        rule=landing['finalizer'];on_target=bool(down and down['object']==episode['target'])
        # Longest stretch of decisions standing still on the pad, from the log alone (not from what the finalizer said).
        still=best=0
        for before,after in zip(steps,steps[1:]):
            dt=max(1e-6,after['epoch']-before['epoch'])
            slow=math.hypot(after['position'][0]-before['position'][0],after['position'][1]-before['position'][1])/dt<=rule['max_horizontal_mps']
            still=still+1 if before.get('landed') and after.get('landed') and slow else 0;best=max(best,still)
        result['touchdown_success']=bool(on_target and down['inside_region'] and abs(down['vertical_speed_mps'])<=landing['touchdown']['max_vertical_speed_mps'])
        result['stable_physical_landing']=bool(result['touchdown_success'] and best+1>=rule['stable_decisions'] and last.get('landed'))
        result['strict_policy_zero_action']=bool(on_target and stopped)
        result['strict_land_success']=result['land_success']
        if finalizer and finalizer['active']:
            result['finalizer']=finalizer
            result['system_land_success']=bool(result['stable_physical_landing'] and finalizer['finalizer_triggered'] and finalizer['disarm_triggered'] and not hit)
            result['stages']['system_land_success']=result['system_land_success']
    return result


def summarise(episode,steps,config,reason,stopped,others=None,landing=None,touchdown=None,hit=None,finalizer=None):
    distances=[s['distance_m'] for s in steps];first=steps[0];last=steps[-1]
    turns=[s['action'][2] for s in steps if abs(s['action'][2])>1e-3]
    toward=None
    if abs(first['bearing_deg'])>15 and turns:toward=(turns[0]>0)==(first['bearing_deg']>0)
    seen=[s['front_seen'] for s in steps]
    acquired=next((i for i,flag in enumerate(seen) if flag),None)
    lost_then_seen=any(seen[i] and not seen[i-1] for i in range(1,len(seen)) if any(seen[:i]))
    jumps=[sum(abs(a-b)/span for a,b,span in zip(s['normalised'],p['normalised'],(2,2,2))) for p,s in zip(steps,steps[1:])
           if 'normalised' in s and 'normalised' in p]
    radius=config['success_radius_m'];heights=[s['height_m'] for s in steps]
    # A stop beside another object of the scene, while the named one is out of range.
    wrong=[name for name,distance in (others or {}).items() if stopped and distance<=radius and last['distance_m']>radius]
    extra=landing_summary(episode,steps,config,landing,stopped,touchdown,hit,others,finalizer) if landing else {}
    # Approach: stopped by itself in the air within the radius, without having come down over a pad (a forced push aside).
    # Land: with the finalizer, the system's landing (stable physical landing on the named pad, latched and disarmed);
    # without it, the strict rule that also needs the policy's own zero action (configs/targets/landing_pads.json).
    came_down=bool(extra and extra['stages']['descent_started'] and not episode.get('pushes'))
    success=(extra.get('system_land_success',extra['land_success']) if extra.get('task')=='land'
             else bool(stopped and last['distance_m']<=radius and reason!='collision' and not extra.get('landed') and not came_down))
    return {**episode,**extra,'reason':reason,'stopped':stopped,'steps':len(steps),'initial_distance_m':first['distance_m'],
            'final_distance_m':last['distance_m'],'minimum_distance_m':min(distances),
            'success':success,
            'reached':min(distances)<=radius,
            'initially_seen':first['front_seen'],'first_turn_toward_target':toward,
            'first_turn':None if not turns else ('right' if turns[0]>0 else 'left'),
            'centred':any(abs(s['bearing_deg'])<=15 for s in steps),'acquired_step':acquired,'reacquired':lost_then_seen,
            'search_steps':sum(not flag for flag in seen),
            'yaw_swept_deg':sum(abs(math.degrees(s['action'][2])) for s in steps),
            'height_range_m':[min(heights),max(heights)],'climbed_m':max(heights)-heights[0],
            'stop_distance_m':last['distance_m'] if stopped else None,'other_objects_m':others,'stopped_at_other':wrong,
            'action_jump':sum(jumps)/len(jumps) if jumps else None,
            'seconds':last['epoch']-first['epoch'],'mean_decision_s':(last['epoch']-first['epoch'])/max(1,len(steps)-1)}


async def run_episode(env,episode,policy,args,record):
    config,oft=env.config,env.oft;steps=[];reason='max_steps';stopped=False;stop_count=0
    planned='strategy' in episode;kick=episode.get('kick');kick_left=0.;altitude_setpoint=None
    teacher=SearchTeacher(config,oft,episode['strategy'],env.ceiling,env.landing) if planned else Teacher(config,oft)
    task=episode.get('task','approach');landed_ticks=0;pushes=Pushes(episode)
    # The executor's own end of a landing. It is given the sentence and the vehicle's state, never the target.
    finalizer=LandingFinalizer(env.landing['finalizer'],episode['instruction'],enabled=planned and args.finalizer=='on');executed_down=0.
    frames_dir=(record['root']/record['rel']) if record else None
    if record:
        for name in ('frontcamera','downcamera'):(frames_dir/name).mkdir(parents=True,exist_ok=True)
    limit=config['max_decisions_baseline'] if args.policy=='baseline' else episode.get('max_ticks',config['generalization']['max_ticks']) if planned else config['max_ticks']
    for index in range(limit):
        started=time.monotonic()
        touchdown,hit=env.contacts()
        if hit:reason='collision';break
        if touchdown:
            # The vehicle stands on a pad and no longer moves; a policy that has not stopped a few decisions later never will.
            # (With the finalizer latched the landing is already complete; the wait only records whether the policy stops too.)
            landed_ticks+=1
            if landed_ticks>env.landing['after_touchdown_ticks']:reason='landed' if finalizer.latched else 'landed_no_stop';break
        observation=env.observe();private=observation['privileged'];state=observation['state']
        resting=env.resting(state);on_pad=resting is not None
        # Wall-clock time between decisions: the simulator's own state can stand still, stamp included, while the vehicle rests.
        finalizer.update(state['position'],time.time(),env.contact_event,executed_down)
        proprio=proprio_vector(state,env.ground_z,oft,observation['yaw_rate'])
        inputs=policy_inputs(observation['frames']['front'],observation['frames']['down'],episode['instruction'],
                             proprio if oft['proprio']['enabled'] else None)
        step={'step':index,'epoch':time.time(),'position':state['position'],'yaw_rad':observation['yaw'],
              'height_m':private['height_m'],'bearing_deg':private['bearing_deg'],'distance_m':private['distance_m'],
              'front_seen':private['front']['seen'],'front_in_fov':private['front']['in_fov'],'down_seen':private['down']['seen'],
              'front_pixel':private['front']['pixel'],'down_pixel':private['down']['pixel'],'ahead_m':private['ahead_m'],
              'over':private['over'],'landed':on_pad,'velocity':state['velocity'],'target_size':private['size'],'finalizer':finalizer.state,
              'input_sha256':{name:digest(inputs[name]) for name in ('front','down')},'proprio':proprio}
        perturbed=False
        if args.policy=='baseline':
            inference=policy.infer(inputs['front'],inputs['down'],None,None,'',freeform=inputs['instruction'])
            action=inference['parsed_action']
            step.update(raw_output=inference['raw_output'].split('Action:')[-1].strip(),prompt=inference['prompt'],inference_ms=inference['inference_ms'])
            if action is None:step['action']=[0.,0.,0.];steps.append(step);reason='invalid_action';break
            step['action']=[action['fwd'],action['down'],action['yaw']]
            if action['stop']:steps.append(step);reason='model_stop';stopped=True;break
            if args.flight=='continuous':
                # The vehicle flew on during inference; the action applies to the observed pose and the
                # next decision is made before this leg ends.
                command=convert_action(action,state,env.ground_z,limits=dict(env.limits,continuous=True),altitude_reference=altitude_setpoint)
                altitude_setpoint=command['target_z']
                await asyncio.wait_for(fly_until_handoff(env.drone,command,interrupted=env.collided),timeout=2*command['expected_duration_sec']+20)
            else:
                command=convert_action(action,state,env.ground_z,limits=env.limits)
                await asyncio.wait_for(execute_action(env.drone,command),timeout=2*command['expected_duration_sec']+15)
        else:
            if planned:
                label,mode=teacher.act({'bearing_deg':private['bearing_deg'],'distance_m':private['distance_m'],'visible':private['front']['seen'],
                                        'below':private['down']['seen'],'ahead_m':private['ahead_m'],'height_m':private['height_m'],
                                        'over':private['over'],'surface_m':private['surface_m'],'task':task,
                                        'landed':resting==episode['target']})
            else:label,mode=teacher.act(private['bearing_deg'],private['distance_m'],private['front']['seen'])
            if args.policy=='teacher':
                action=label;step['teacher_state']=mode;step['teacher_action']=list(label)
            else:
                inference=policy.infer(inputs['front'],inputs['down'],inputs['instruction'],inputs['proprio'])
                chunk=inference['chunk'];action=chunk[0]
                step.update(chunk=chunk,normalised=inference['normalised_chunk'][0],prompt=inference['prompt'],
                            inference_ms=inference['inference_ms'],teacher_action=label,teacher_state=mode)
            if kick and index==kick['after_ticks']:kick_left=math.radians(kick['degrees'])
            if abs(kick_left)>1e-6:
                # A forced turn away from the target, for the teacher and for a tested policy alike: flown, never a label.
                turn=max(-oft['action_bounds']['yaw_rad'][1],min(oft['action_bounds']['yaw_rad'][1],kick_left))
                kick_left-=turn;action=[0.,0.,turn];perturbed=True;step['teacher_state']='kick'
            if planned:
                # A forced move past the place an approach stops at, flown by the teacher and by a tested policy alike: never a label.
                forced=pushes.step(mode,{'bearing_deg':private['bearing_deg'],'distance_m':private['distance_m'],'height_m':private['height_m'],
                                         'surface_m':private['surface_m']},oft['action_bounds']['yaw_rad'][1])
                if forced:action=forced;perturbed=True;step['teacher_state']='push'
            step['action']=list(action);step['perturbed']=perturbed
            if record:
                name=f'{index:06d}.png'
                cv2.imwrite(str(frames_dir/'frontcamera'/name),inputs['front']);cv2.imwrite(str(frames_dir/'downcamera'/name),inputs['down'])
                step['img_name']=name
            halted=is_stop(action,oft) and not perturbed
            stop_count=stop_count+1 if halted else 0
            if stop_count>=config['stop_ticks'] and not pushes.pending():
                # The policy's own stop. On a surface the finalizer may be a decision or two from latching; it is given them.
                stopped=True
                if finalizer.state!='CONTACT_CANDIDATE':
                    steps.append(step);reason='landed' if finalizer.latched else 'model_stop' if args.policy=='oft' else 'teacher_stop';break
            # Once the landing is latched no motion command is passed on, whatever the policy says.
            executed=finalizer.command(action)
            step['executed']=list(executed);executed_down=executed[1]
            await env.tick(executed,observation['yaw'])
            await asyncio.sleep(max(0.,oft['tick_s']-(time.monotonic()-started)))
        steps.append(step)
        if args.live:
            cv2.imwrite(str(Path(args.output)/'live_front.png'),inputs['front']);cv2.imwrite(str(Path(args.output)/'live_down.png'),inputs['down'])
            temporary=Path(args.output)/'live.tmp'
            temporary.write_text(json.dumps({'mode':'VISUAL SEARCH','model':args.policy,'model_name':args.model_name,'episode':episode,'step':step,
                                             'map':env.geometry.config['name'],'target':env.geometry.config['objects'][env.target]['noun'],
                                             'level':LEVELS.get(args.set or '',args.set),
                                             'success_radius_m':config['success_radius_m'],'failure':'NORMAL',
                                             'raw_sha256':{name:digest(observation['frames'][name]) for name in ('front','down')}}))
            temporary.replace(Path(args.output)/'live.json')
    touchdown,hit=env.contacts()
    if hit:reason='collision'
    others=env.others(steps[-1]['position']) if steps else None
    if finalizer.latched:finalizer.disarmed(env.disarm())
    else:await env.close()
    if args.live:
        (Path(args.output)/'live_result.json').write_text(json.dumps({'episode':episode['id'],'reason':reason,'stopped':stopped,
                                                                      'final_distance_m':steps[-1]['distance_m'] if steps else None}))
    return summarise(episode,steps,config,reason,stopped,others,env.landing,touchdown,hit,finalizer.report()),steps


def planned_episodes(args,config):
    """Episodes of one set of a plan file: `sets[NAME]` of the held-out file, or a split of a dataset plan."""
    data=json.loads(Path(args.plan).read_text(encoding='utf-8'))
    episodes=data['sets'][args.set] if 'sets' in data and args.set in data['sets'] else data[args.set]
    if args.only:episodes=[episode for episode in episodes if episode['id'] in args.only or episode['target'] in args.only or episode['kind'] in args.only]
    if args.strategy:episodes=[dict(episode,strategy=args.strategy,id=f'{episode["id"]}-{args.strategy}') for episode in episodes]
    if args.layout:episodes=[dict(episode,layout=args.layout,planned_layout=episode['layout']) for episode in episodes]
    if args.pilot_verbs:
        # The pilot was trained with `Approach` when the target started in view and `Find` when it did not.
        episodes=[dict(episode,planned_instruction=episode['instruction'],
                       instruction=config['instructions']['approach' if episode['kind'] in ('visible','peripheral') else 'find'].format(
                           noun=load_map(episode['map'])['objects'][episode['target']]['noun'])) for episode in episodes]
    if args.named:
        # Control: the same start, but the sentence names another object of the scene. The score still follows the
        # planned object, so a `success` here is a stop beside the object that was not asked for.
        renamed=[]
        for episode in episodes:
            objects=load_map(episode['map'])
            others=[name for name in objects['layouts'][episode['layout']] if name!=episode['target']]
            named=others[episode['seed']%len(others)] if args.named=='another' else args.named
            renamed.append(dict(episode,id=f'{episode["id"]}-told-{named}',named_object=named,planned_instruction=episode['instruction'],
                                instruction=config['instructions'][episode['instruction_id']].format(noun=objects['objects'][named]['noun'])))
        episodes=renamed
    return episodes[args.skip:args.skip+args.limit] if args.limit else episodes[args.skip:]


async def main(args):
    config=load_config();oft=load_oft_config(args.oft_config) if args.oft_config else load_oft_config()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    policy=None
    if args.policy=='baseline':
        from src.integration.aerovla_int4_loader import AeroVLAInt4
        policy=AeroVLAInt4(MANIFEST);args.model_name=args.model_name or f'AeroVLA ({args.flight})'
    elif args.policy=='oft':
        from src.aerovla_oft.model import AeroVLAOFT
        saved=json.loads((Path(args.checkpoint)/'manifest.json').read_text())['config']
        oft=saved;policy=AeroVLAOFT(MANIFEST,saved,checkpoint=args.checkpoint)
        args.model_name=args.model_name or saved.get('name')
    client=ProjectAirSimClient(address=args.host,port_topics=args.ports[0],port_services=args.ports[1]);client.connect();client.socket_services.recv_timeout=60000
    env=SearchEnv(client,config,oft,output)
    results_path=output/'results.json'
    done=json.loads(results_path.read_text())['episodes'] if results_path.exists() and args.resume else []
    done=[item for item in done if not item.get('error')]
    finished={item['id'] for item in done};status=0
    if args.plan:episodes=planned_episodes(args,config)
    else:episodes=[{'case':case,'target':target,'seed':seed,'id':f'{case}-{target}-{seed}'}
                   for seed in range(args.seed_start,args.seed_start+args.episodes) for case in args.cases for target in args.targets]
    record_root=Path(args.record) if args.record else None
    try:
        for episode in episodes:
            identifier=episode['id']
            if identifier in finished:continue
            try:
                if not args.plan:
                    centre=env.map('blocks')['native_objects'][env.map('blocks')['objects'][episode['target']]['native']]['center']
                    episode=make_episode(config,episode['case'],episode['target'],centre,episode['seed'])
                await env.reset(episode)
                record={'root':record_root,'rel':f'{episode.get("map","blocks")}/{identifier}'} if record_root else None
                summary,steps=await run_episode(env,episode,policy,args,record)
            except Exception:
                summary={**episode,'error':traceback.format_exc()};steps=[]
                print(summary['error'],flush=True)
                if 'Connection' in summary['error'] or 'closed' in summary['error']:status=1
            summary['policy']=args.policy;summary['set']=args.set
            done.append(summary)
            with (output/'steps.jsonl').open('a') as file:
                for step in steps:file.write(json.dumps({'episode':identifier,**step})+'\n')
            if record_root and steps:
                (record_root/'episodes').mkdir(parents=True,exist_ok=True)
                (record_root/'episodes'/f'{identifier}.json').write_text(json.dumps({'summary':summary,'traj_rel_dir':record['rel'],'steps':steps}))
            results_path.write_text(json.dumps({'policy':args.policy,'flight':args.flight if args.policy=='baseline' else 'tick',
                                                'checkpoint':args.checkpoint,'model_name':args.model_name,'plan':args.plan,'set':args.set,
                                                'layout_override':args.layout,'config':config,'oft':oft,'episodes':done},indent=1))
            print(f'EPISODE {identifier} {summary.get("reason")} success={summary.get("success")} steps={summary.get("steps")} '
                  f'initial={summary.get("initial_distance_m",float("nan")):.1f} final={summary.get("final_distance_m",float("nan")):.1f} '
                  f'min={summary.get("minimum_distance_m",float("nan")):.1f} seen0={summary.get("initially_seen")} acquired={summary.get("acquired_step")} '
                  f'climbed={summary.get("climbed_m",float("nan")):.1f}',flush=True)
            if status:break
    finally:
        client.disconnect()
    raise SystemExit(status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True)
    parser.add_argument('--ports',type=int,nargs=2,default=[8989,8990],help='simulator topic and service ports')
    parser.add_argument('--policy',choices=('baseline','teacher','oft'),required=True)
    parser.add_argument('--flight',choices=('step','continuous'),default='step',help='how the baseline executes its actions')
    parser.add_argument('--checkpoint');parser.add_argument('--oft-config');parser.add_argument('--model-name')
    parser.add_argument('--cases',nargs='+',default=['A','B','C']);parser.add_argument('--targets',nargs='+',default=['blue_cone','orange_ball'])
    parser.add_argument('--episodes',type=int,default=2,help='seeds per case and target');parser.add_argument('--seed-start',type=int,default=0)
    parser.add_argument('--plan',help='plan file with fixed start states');parser.add_argument('--set',help='set or split of the plan file to fly')
    parser.add_argument('--only',nargs='+',help='episode ids, objects or kinds to keep');parser.add_argument('--skip',type=int,default=0)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--layout',help='fly the planned starts in another layout of the same map')
    parser.add_argument('--strategy',help='teacher search strategy to use instead of the planned one')
    parser.add_argument('--pilot-verbs',action='store_true',help="word the instruction as the pilot's training data did")
    parser.add_argument('--named',help="control: name this object (or 'another') in the sentence instead of the planned one")
    parser.add_argument('--finalizer',choices=('on','off'),default='on',help='the low-level landing finalizer for sentences that ask for a landing; off flies as before it existed')
    parser.add_argument('--output',default=str(ROOT/'outputs/visual_search/run'));parser.add_argument('--record',help='dataset root to write teacher episodes into')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--live',action='store_true')
    arguments=parser.parse_args()
    if arguments.policy=='oft' and not arguments.checkpoint:parser.error('--checkpoint is required for --policy oft')
    if arguments.plan and not arguments.set:parser.error('--plan needs --set')
    asyncio.run(main(arguments))
