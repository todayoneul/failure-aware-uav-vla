"""Headless AeroVLA evaluation: fixed starts, scene landmarks and prompt ablations.

Every trial reloads the scene, so each starts from the same physical state. The model
flies at the configured action scale and ends the episode itself; ground truth only scores.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import asyncio
import collections
import json
import math
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
from scipy.spatial.transform import Rotation
from projectairsim import ProjectAirSimClient,World,Drone
from projectairsim.types import Pose
from projectairsim.utils import unpack_image
from src.integration.projectairsim_observation_adapter import adapt_state,make_mosaic
from src.integration.projectairsim_action_adapter import convert_action,flight_limits
from src.mission.manager import MissionManager,MissionTarget,TERMINAL
from src.mission.scene import prepare_mission_config,scene_records,LANDMARKS
from src.mission.flight import land_and_confirm,guarded_execute,surface_below,approach_altitude,climb_to
from src.mission.grounding import grounding_report,target_visibility,matched_camera_images
from src.mission.geometry import relative_camera_pose,camera_metadata
from src.mission.landmarks import load_landmarks,instruction_for,prompt_arguments

# condition -> (prompt mode, whether the target's description is given)
PROMPTS={'hint+generic':('hint',False),'hint+landmark':('hint',True),'landmark-only':('description',True),
         'instruction':('instruction',True)}
MAP_VIEW={'position':[45.,0.,-150.],'rpy':[0,-90,0]}


def yaw_of(state):return float(Rotation.from_quat(state['orientation']).as_euler('xyz')[2])


def motion_profile(poses,start_stamp,end_stamp):
    """How steadily the vehicle moved between the first decision and the end of navigation (simulation time)."""
    window=[pose for pose in poses if start_stamp<=pose[0]<=end_stamp]
    speeds=[math.hypot(b[1]-a[1],b[2]-a[2])/((b[0]-a[0])/1e9) for a,b in zip(window,window[1:]) if b[0]>a[0]]
    if not speeds:return None
    return {'navigation_sim_s':(window[-1][0]-window[0][0])/1e9,'mean_speed_mps':sum(speeds)/len(speeds),
            'stationary_fraction':sum(speed<.1 for speed in speeds)/len(speeds)}


def resolve_target(name,protocol,landmarks,described):
    spec=protocol['targets'][name]
    if 'landmark' in spec:
        landmark=next(item for item in landmarks if item['id']==spec['landmark'])
        return MissionTarget(landmark['name'],tuple(landmark['position']),'scene bounding box '+'+'.join(landmark['objects']),
                             landmark['description'] if described else None,landmark['id'],landmark.get('noun'))
    return MissionTarget(name,tuple(spec['position']),spec.get('note','protocol coordinate'),
                         spec.get('description') if described else None,None,spec.get('noun'))


async def run_trial(trial,protocol,client,model,output,limits,envelope,landmarks_cache):
    start=protocol['starts'][trial['start']];mode,described=PROMPTS[trial['prompt']]
    config=prepare_mission_config(ROOT,output)
    scene=json.loads((config/'scene_basic_drone.jsonc').read_text())
    scene['actors'][0]['origin']={'xyz':' '.join(str(v) for v in start['xyz']),'rpy-deg':f'0 0 {start["yaw_deg"]}'}
    (config/'scene_basic_drone.jsonc').write_text(json.dumps(scene,indent=2))
    world=World(client,'scene_basic_drone.jsonc',delay_after_load_sec=2,sim_config_path=str(config))
    drone=Drone(client,world,'Drone1');collisions=[];poses=[]
    client.subscribe(drone.robot_info['collision_info'],lambda _,message:collisions.append(message))
    def pose_callback(_,message):
        # Ten samples per simulated second are enough to tell flying from standing still.
        if not poses or message['time_stamp']-poses[-1][0]>=1e8:
            poses.append((message['time_stamp'],message['position']['x'],message['position']['y']))
    client.subscribe(drone.robot_info['actual_pose'],pose_callback)
    if not landmarks_cache:
        names=[name for entry in json.loads(LANDMARKS.read_text()) for name in entry['objects']]
        landmarks_cache.extend(load_landmarks(LANDMARKS,scene_records(world,names)))
    target=resolve_target(trial['target'],protocol,landmarks_cache,described)
    hint_target=resolve_target(trial.get('hint_target',trial['target']),protocol,landmarks_cache,described)
    rest=adapt_state(drone.get_ground_truth_kinematics());ground_z=rest['position'][2]
    hint,freeform=prompt_arguments(target,mode)
    result={'id':trial['id'],'base_id':trial.get('base_id',trial['id']),'group':trial['group'],'start':trial['start'],'target':trial['target'],
            'hint_target':trial.get('hint_target'),'hint_mode':trial.get('hint_mode'),
            'prompt_condition':trial['prompt'],'direction_hint':hint,'decoder':model.decoder,'approach':trial.get('approach'),
            'target_position':list(target.position),'instruction':instruction_for(target),'rest_position':rest['position'],
            'max_steps':trial['max_steps'],'flight':'continuous' if envelope['continuous'] else 'step',
            'started_epoch':time.time(),'steps':[],'error':None}
    manager=MissionManager(**{**limits,'max_steps':trial['max_steps']})
    frames=output/'frames'/trial['id'];frames.mkdir(parents=True,exist_ok=True)
    airborne=False;flight_stamp=None;current=rest;altitude_setpoint=None

    def state():return adapt_state(drone.get_ground_truth_kinematics())
    def collided():return any(event.get('time_stamp',0)>flight_stamp for event in collisions)
    try:
        if not (output/'map.png').exists():
            # One top-down picture for the trajectory figure; it is never a model input.
            drone.set_camera_pose('Overview',Pose(relative_camera_pose(MAP_VIEW['position'],MAP_VIEW['rpy'],rest)))
            time.sleep(.6);message=drone.get_images('Overview',[0])[0]
            cv2.imwrite(str(output/'map.png'),unpack_image(message))
            (output/'map.json').write_text(json.dumps(camera_metadata(message)))
        drone.enable_api_control();drone.arm();airborne=True
        result['takeoff_return']=await asyncio.wait_for(await drone.takeoff_async(timeout_sec=20),timeout=25)
        await asyncio.wait_for(await drone.hover_async(),timeout=10);await asyncio.sleep(.5)
        sample=drone.get_ground_truth_kinematics();current=adapt_state(sample)
        if ground_z-current['position'][2]<.5 or drone.get_landed_state()==0:raise RuntimeError('Actual takeoff criterion failed')
        climb_z=ground_z-trial['start_clearance_m'] if trial.get('start_clearance_m') else None
        if climb_z is not None:
            await climb_to(drone,climb_z,envelope['vertical_speed_mps']);await asyncio.sleep(.5)
            sample=drone.get_ground_truth_kinematics();current=adapt_state(sample)
        flight_stamp=sample['time_stamp'];result['start_state']=current
        result['start_clearance_m']=ground_z-current['position'][2]
        manager.select(target);manager.start(current)
        result['initial_distance_m']=manager.distances[0];trajectory=[current['position']]
        while manager.status=='NAVIGATING':
            current=state()
            if collided():manager.fail('collision');break
            if manager.observe(current)!='NAVIGATING':break
            # `approach: above` rises over the target surface once it is near, as the demo does for a selected roof.
            climb_z=approach_altitude(current['position'][2],target.position[2],ground_z,envelope,
                                      manager.metrics(current)['horizontal_m']) if trial.get('approach')=='above' else None
            if climb_z is not None:
                await asyncio.wait_for(await drone.hover_async(),timeout=10)
                result.setdefault('climbs',[]).append({'step':manager.steps+1,**await climb_to(drone,climb_z,envelope['vertical_speed_mps']),
                                                      'distance_m':manager.metrics(current)['horizontal_m']})
                current=state();manager.hover_z=current['position'][2];altitude_setpoint=None;trajectory.append(current['position'])
            raw=[];visibility={}
            for sensor,name in (('FrontCamera','front'),('DownCamera','down')):
                image,depth,meta=matched_camera_images(drone.get_images(sensor,[0,1]))
                raw.append(image);visibility[name]=target_visibility(target.position,meta,depth)['status']
            instruction=instruction_for(target)
            hint_goal=[hint_target.position[0],hint_target.position[1],manager.hover_z]
            if trial.get('hint_mode')=='ahead':
                # Uninformative hint: always "straight ahead", whatever the target's real bearing.
                heading=yaw_of(current)
                hint_goal=[current['position'][0]+1000*math.cos(heading),current['position'][1]+1000*math.sin(heading),manager.hover_z]
            grounding=grounding_report(current,hint_goal,hint_target.position,instruction,hint,target.landmark if described else None,freeform)
            inference=model.infer(*raw,current,hint_goal,instruction,direction_hint=hint,freeform=freeform)
            # Bearing of the scored target in the body frame, independent of which hint was sent.
            bearing=math.atan2(target.position[1]-current['position'][1],target.position[0]-current['position'][0])-yaw_of(current)
            if inference['prompt']!=grounding['prompt']:raise RuntimeError('Logged prompt differs from model prompt')
            action=inference['parsed_action']
            step={'step':manager.steps+1,'epoch':time.time(),'pose':current,'distance_before_m':manager.metrics(current)['horizontal_m'],
                  'semantic_direction':grounding['semantic_direction'],'body_bearing_deg':grounding['body_bearing_deg'],
                  'target_bearing_deg':math.degrees(math.atan2(math.sin(bearing),math.cos(bearing))),
                  'prompt':inference['prompt'],'raw_output':inference['raw_output'].split('Action:')[-1].strip(),
                  'parsed_action':action,'inference_ms':inference['inference_ms'],'visibility':visibility,
                  'decoder':inference.get('decoder')}
            make_mosaic(*raw).save(frames/f'{step["step"]:02d}.jpg',quality=85)
            if action is None:
                step['parse_error']=inference['parse_error'];result['steps'].append(step);manager.fail('invalid_action');break
            # Continuous flight has moved on during inference; the action applies to the observed pose.
            execution=current if envelope['continuous'] else state()
            command=convert_action(action,execution,ground_z,limits=envelope,
                                   altitude_reference=altitude_setpoint if envelope['continuous'] else None)
            if not command['stop']:altitude_setpoint=command['target_z']
            started=time.time()
            returns=await guarded_execute(drone,manager,execution,collisions,flight_stamp,command)
            if returns is None:result['steps'].append(step);break
            if not envelope['continuous']:await asyncio.sleep(.3)
            after=state()
            manager.record_step(execution,after,action['stop'],collided());trajectory.append(after['position'])
            step.update(command={k:command[k] for k in ('mode','yaw_delta_rad','displacement_ned','target_position','altitude_clamped')},
                        command_returns=returns,command_s=time.time()-started,after_position=after['position'],
                        moved_xy_m=math.dist(execution['position'][:2],after['position'][:2]),
                        distance_after_m=manager.metrics(after)['horizontal_m'],clearance_m=ground_z-after['position'][2],
                        mission_state=manager.status)
            result['steps'].append(step)
            with (output/'steps.jsonl').open('a') as file:file.write(json.dumps({'trial':trial['id'],**step})+'\n')
            print(f'{trial["id"]} step={step["step"]} out={step["raw_output"]!r} mode={command["mode"]} '
                  f'd={step["distance_after_m"]:.1f}m clr={step["clearance_m"]:.1f} state={manager.status}',flush=True)
        result['motion']=motion_profile(poses,flight_stamp,drone.get_ground_truth_kinematics()['time_stamp'])
        if envelope['continuous'] and manager.status!='LANDING' and airborne:
            # An unfinished leg must not carry on while the trial is being closed.
            await asyncio.wait_for(await drone.hover_async(),timeout=10)
        current=state()
        if manager.status=='LANDING':
            try:
                result['landing']=await land_and_confirm(drone,surface_z=surface_below(world,current))
                result['landing'].pop('touchdown_kinematics',None)
            except (RuntimeError,asyncio.TimeoutError) as error:result['landing_error']=repr(error)
            landed=drone.get_landed_state()==0;airborne=not landed
            current=state();manager.finish_stop(landed=landed,touchdown_z=current['position'][2])
        result['trajectory']=trajectory
    except Exception:
        result['error']=traceback.format_exc();manager.fail('runtime_error');print(result['error'],flush=True)
    finally:
        try:
            if airborne and drone.get_landed_state()!=0:
                # FastPhysics holds a collided vehicle in place, so it cannot land; the next trial reloads the scene.
                if flight_stamp is not None and collided():drone.disarm();result['cleanup']='disarmed while held by a collision'
                else:await land_and_confirm(drone,surface_z=surface_below(world,adapt_state(drone.get_ground_truth_kinematics())))
            drone.disable_api_control()
        except Exception:result['cleanup_error']=traceback.format_exc()
    final=manager.metrics(current)['horizontal_m'] if manager.target else None
    executed=[step for step in result['steps'] if 'command' in step]
    result.update(state=manager.status,reason=manager.reason,executed_steps=manager.steps,stop=manager.stop,
                  final_distance_m=final,minimum_distance_m=min(manager.distances) if manager.distances else None,
                  final_position=current['position'],path_length_m=sum(step['moved_xy_m'] for step in executed),
                  # Touchdown after the model's LAND also raises contact events; report only an in-flight collision.
                  collision=next(({'object':event.get('object_name'),'impact_point':event.get('impact_point')} for event in collisions
                                  if flight_stamp is not None and event.get('time_stamp',0)>flight_stamp),None)
                            if manager.reason=='collision' else None,
                  elapsed_s=time.time()-result['started_epoch'],
                  actions=collections.Counter(step['raw_output'].replace('</s>','').strip() for step in result['steps']).most_common(),
                  final_distance_to_hint_target_m=math.dist(current['position'][:2],hint_target.position[:2]))
    return result


async def main(args):
    protocol=json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    limits=json.loads((ROOT/'configs/mission_limits.json').read_text())
    envelope=flight_limits(json.loads((ROOT/'configs/flight_limits.json').read_text()))
    if args.flight:envelope['continuous']=args.flight=='continuous'
    trials=[trial for trial in protocol['trials'] if not args.only or trial['id'] in args.only or trial['group'] in args.only]
    if args.max_steps:trials=[{**trial,'max_steps':args.max_steps} for trial in trials]
    if args.repeats>1:
        trials=[{**trial,'base_id':trial['id'],'id':f'{trial["id"]}#{number}'} for number in range(1,args.repeats+1) for trial in trials]
    if args.suffix:
        # Repeats of the same trial definition; runs are not exactly repeatable, so they are counted separately.
        trials=[{**trial,'base_id':trial['id'],'id':trial['id']+args.suffix} for trial in trials]
    results_path=output/'results.json'
    done=json.loads(results_path.read_text())['trials'] if results_path.exists() and args.resume else []
    finished={item['id'] for item in done}
    from src.integration.aerovla_int4_loader import AeroVLAInt4
    model=AeroVLAInt4(ROOT/'outputs/integration/model-downloads.json',decoder=args.decoder)
    client=ProjectAirSimClient(address=args.host);client.connect();client.socket_services.recv_timeout=60000
    landmarks=[];status=0
    try:
        for trial in trials:
            if trial['id'] in finished:continue
            result=await run_trial(trial,protocol,client,model,output,limits,envelope,landmarks)
            done.append(result)
            results_path.write_text(json.dumps({'protocol':args.protocol,'mission_limits':limits,'flight_limits':envelope,'decoder':args.decoder,
                                                'landmarks':landmarks,'load_info':model.load_info,'trials':done},indent=1))
            stop=result['stop']
            print(f'TRIAL {result["id"]} {result["state"]} reason={result["reason"]} steps={result["executed_steps"]} '
                  f'initial={result.get("initial_distance_m") or float("nan"):.1f} final={result["final_distance_m"] or float("nan"):.1f} '
                  f'min={result["minimum_distance_m"] or float("nan"):.1f} stop={"%.1f m"%stop["distance_m"] if stop else None} '
                  f'elapsed={result["elapsed_s"]:.0f}s',flush=True)
            if result['error'] and 'Connection' in result['error']:status=1;break
    finally:
        client.disconnect()
    raise SystemExit(status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True)
    parser.add_argument('--protocol',default=str(ROOT/'configs/evaluation_protocol.json'))
    parser.add_argument('--output',default=str(ROOT/'outputs/model_eval/run'))
    parser.add_argument('--only',nargs='*',help='trial ids or group names');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--suffix',default='',help='appended to trial ids for a repeat pass, e.g. "#2"')
    parser.add_argument('--flight',choices=('step','continuous'),help='override configs/flight_limits.json')
    parser.add_argument('--max-steps',type=int,help='override every selected trial budget')
    parser.add_argument('--repeats',type=int,default=1,help='run the selected trials this many times in one session')
    parser.add_argument('--decoder',choices=('grammar','free'),default='grammar',
                        help='grammar keeps output inside the action format; free is plain greedy decoding')
    asyncio.run(main(parser.parse_args()))
