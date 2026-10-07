"""Visual search on Project AirSim: evaluate a policy, or record the privileged teacher as a dataset.

Policies see Front/Down RGB and one sentence (`policy_inputs`). The target centre is used here to place
the vehicle, to score, and by the teacher; it is never passed to a learned policy.

  --policy baseline   AeroVLA NF4, instruction only, its own step actions and LAND
  --policy teacher    scripted expert; with --record it writes an AeroVLA-format dataset
  --policy oft        AeroVLA-OFT checkpoint, short action chunk, receding horizon
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
from src.integration.projectairsim_observation_adapter import adapt_state
from src.integration.projectairsim_action_adapter import convert_action,execute_action,flight_limits
from src.mission.scene import prepare_mission_config,scene_records,LANDMARKS
from src.mission.landmarks import load_landmarks
from src.mission.grounding import target_visibility,matched_camera_images
from src.mission.flight import climb_to
from src.visual_search.episodes import load_config,make_episode,relative_target,policy_inputs,Teacher
from src.aerovla_oft.spec import load_config as load_oft_config,proprio_vector,is_stop

MANIFEST=ROOT/'outputs/integration/model-downloads.json'


def digest(frame):return hashlib.sha256(frame.tobytes()).hexdigest()


class SearchEnv:
    def __init__(self,client,config,oft,output):
        self.client,self.config,self.oft,self.output=client,config,oft,Path(output)
        self.landmarks={};self.limits=flight_limits(json.loads((ROOT/'configs/flight_limits.json').read_text()))

    def prepare(self):
        """Read the landmark boxes from the simulator once; no flight."""
        directory=prepare_mission_config(ROOT,self.output)
        world=World(self.client,'scene_basic_drone.jsonc',delay_after_load_sec=2,sim_config_path=str(directory))
        names=[name for entry in json.loads(LANDMARKS.read_text()) for name in entry['objects']]
        self.landmarks={item['id']:item for item in load_landmarks(LANDMARKS,scene_records(world,names))}

    async def reset(self,episode):
        directory=prepare_mission_config(ROOT,self.output)
        scene=json.loads((directory/'scene_basic_drone.jsonc').read_text())
        x,y=episode['start_xy']
        scene['actors'][0]['origin']={'xyz':f'{x} {y} -3.0','rpy-deg':f'0 0 {episode["start_yaw_deg"]}'}
        (directory/'scene_basic_drone.jsonc').write_text(json.dumps(scene,indent=2))
        self.world=World(self.client,'scene_basic_drone.jsonc',delay_after_load_sec=2,sim_config_path=str(directory))
        self.drone=Drone(self.client,self.world,'Drone1');self.collisions=[]
        self.client.subscribe(self.drone.robot_info['collision_info'],lambda _,message:self.collisions.append(message))
        landmark=self.landmarks[episode['target']]
        # Scored point: the object's centre at half its height.
        self.centre=[landmark['position'][0],landmark['position'][1],(landmark['minimum'][2]+landmark['maximum'][2])/2]
        # The vehicle's own arms cross the middle rows of the front view, so sight is tested at several heights.
        low,high=landmark['maximum'][2],landmark['minimum'][2]
        self.sight_points=[[self.centre[0],self.centre[1],low+(high-low)*fraction] for fraction in (.25,.5,.75,.9)]
        rest=adapt_state(self.drone.get_ground_truth_kinematics());self.ground_z=rest['position'][2]
        self.drone.enable_api_control();self.drone.arm()
        await asyncio.wait_for(await self.drone.takeoff_async(timeout_sec=20),timeout=25)
        await climb_to(self.drone,self.ground_z-self.config['cruise_height_m'],2.);await asyncio.sleep(.8)
        sample=self.drone.get_ground_truth_kinematics();self.flight_stamp=sample['time_stamp']
        self.setpoint_z=adapt_state(sample)['position'][2];self.previous_yaw=None;self.previous_stamp=None

    def collided(self):return any(event.get('time_stamp',0)>self.flight_stamp for event in self.collisions)

    def observe(self):
        """Raw frames for the policy plus privileged target geometry for the teacher and the score."""
        frames={};visibility={}
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
        sample=self.drone.get_ground_truth_kinematics();state=adapt_state(sample)
        bearing,distance,yaw=relative_target(state,self.centre)
        rate=0.
        if self.previous_yaw is not None and sample['time_stamp']>self.previous_stamp:
            delta=math.atan2(math.sin(yaw-self.previous_yaw),math.cos(yaw-self.previous_yaw))
            rate=delta/((sample['time_stamp']-self.previous_stamp)/1e9)
        self.previous_yaw,self.previous_stamp=yaw,sample['time_stamp']
        return {'frames':frames,'state':state,'yaw':yaw,'yaw_rate':rate,'stamp':sample['time_stamp'],
                'privileged':{'bearing_deg':bearing,'distance_m':distance,'front':visibility['front'],'down':visibility['down']}}

    async def tick(self,action,yaw):
        """One small continuous motion; the command outlasts the tick so the next one takes over in flight."""
        forward,down,turn=action;tick=self.oft['tick_s'];heading=yaw+turn
        self.setpoint_z=max(self.ground_z-self.limits['maximum_clearance_m'],min(self.ground_z-self.limits['minimum_clearance_m'],self.setpoint_z+down))
        await self.drone.move_by_velocity_z_async(forward/tick*math.cos(heading),forward/tick*math.sin(heading),self.setpoint_z,
                                                  duration=tick*3,yaw_is_rate=False,yaw=heading)

    async def close(self):
        try:
            await asyncio.wait_for(await self.drone.hover_async(),timeout=10)
            self.drone.disarm();self.drone.disable_api_control()
        except Exception:pass


def summarise(episode,steps,config,reason,stopped):
    distances=[s['distance_m'] for s in steps];first=steps[0];last=steps[-1]
    turns=[s['action'][2] for s in steps if abs(s['action'][2])>1e-3]
    toward=None
    if abs(first['bearing_deg'])>15 and turns:toward=(turns[0]>0)==(first['bearing_deg']>0)
    seen=[s['front_seen'] for s in steps]
    acquired=next((i for i,flag in enumerate(seen) if flag),None)
    lost_then_seen=any(seen[i] and not seen[i-1] for i in range(1,len(seen)) if any(seen[:i]))
    jumps=[sum(abs(a-b)/span for a,b,span in zip(s['normalised'],p['normalised'],(2,2,2))) for p,s in zip(steps,steps[1:])
           if 'normalised' in s and 'normalised' in p]
    return {**episode,'reason':reason,'stopped':stopped,'steps':len(steps),'initial_distance_m':first['distance_m'],
            'final_distance_m':last['distance_m'],'minimum_distance_m':min(distances),
            'success':bool(stopped and last['distance_m']<=config['success_radius_m'] and reason!='collision'),
            'reached':min(distances)<=config['success_radius_m'],
            'initially_seen':first['front_seen'],'first_turn_toward_target':toward,
            'centred':any(abs(s['bearing_deg'])<=15 for s in steps),'acquired_step':acquired,'reacquired':lost_then_seen,
            'yaw_swept_deg':sum(abs(math.degrees(s['action'][2])) for s in steps),
            'height_range_m':[min(s['height_m'] for s in steps),max(s['height_m'] for s in steps)],
            'action_jump':sum(jumps)/len(jumps) if jumps else None,
            'mean_decision_s':(last['epoch']-first['epoch'])/max(1,len(steps)-1)}


async def run_episode(env,episode,policy,args,record):
    config,oft=env.config,env.oft;steps=[];reason='max_steps';stopped=False;stop_count=0
    teacher=Teacher(config,oft);kick=episode.get('kick');kick_left=0.
    frames_dir=(record['root']/record['rel']) if record else None
    if record:
        for name in ('frontcamera','downcamera'):(frames_dir/name).mkdir(parents=True,exist_ok=True)
    limit=config['max_decisions_baseline'] if args.policy=='baseline' else config['max_ticks']
    for index in range(limit):
        started=time.monotonic()
        if env.collided():reason='collision';break
        observation=env.observe();private=observation['privileged'];state=observation['state']
        proprio=proprio_vector(state,env.ground_z,oft,observation['yaw_rate'])
        inputs=policy_inputs(observation['frames']['front'],observation['frames']['down'],episode['instruction'],
                             proprio if oft['proprio']['enabled'] else None)
        step={'step':index,'epoch':time.time(),'position':state['position'],'yaw_rad':observation['yaw'],
              'height_m':env.ground_z-state['position'][2],'bearing_deg':private['bearing_deg'],'distance_m':private['distance_m'],
              'front_seen':private['front']['seen'],'front_in_fov':private['front']['in_fov'],'down_seen':private['down']['seen'],
              'front_pixel':private['front']['pixel'],'down_pixel':private['down']['pixel'],
              'input_sha256':{name:digest(inputs[name]) for name in ('front','down')},'proprio':proprio}
        perturbed=False
        if args.policy=='baseline':
            inference=policy.infer(inputs['front'],inputs['down'],None,None,'',freeform=inputs['instruction'])
            action=inference['parsed_action']
            step.update(raw_output=inference['raw_output'].split('Action:')[-1].strip(),prompt=inference['prompt'],inference_ms=inference['inference_ms'])
            if action is None:step['action']=[0.,0.,0.];steps.append(step);reason='invalid_action';break
            step['action']=[action['fwd'],action['down'],action['yaw']]
            if action['stop']:steps.append(step);reason='model_stop';stopped=True;break
            command=convert_action(action,state,env.ground_z,limits=env.limits)
            await asyncio.wait_for(execute_action(env.drone,command),timeout=2*command['expected_duration_sec']+15)
        else:
            label,mode=teacher.act(private['bearing_deg'],private['distance_m'],private['front']['seen'])
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
            step['action']=list(action);step['perturbed']=perturbed
            if record:
                name=f'{index:06d}.png'
                cv2.imwrite(str(frames_dir/'frontcamera'/name),inputs['front']);cv2.imwrite(str(frames_dir/'downcamera'/name),inputs['down'])
                step['img_name']=name
            halted=is_stop(action,oft) and not perturbed
            stop_count=stop_count+1 if halted else 0
            if stop_count>=config['stop_ticks']:steps.append(step);reason='model_stop' if args.policy=='oft' else 'teacher_stop';stopped=True;break
            await env.tick(action,observation['yaw'])
            await asyncio.sleep(max(0.,oft['tick_s']-(time.monotonic()-started)))
        steps.append(step)
        if args.live:
            cv2.imwrite(str(Path(args.output)/'live_front.png'),inputs['front']);cv2.imwrite(str(Path(args.output)/'live_down.png'),inputs['down'])
            temporary=Path(args.output)/'live.tmp'
            temporary.write_text(json.dumps({'mode':'VISUAL SEARCH','model':args.policy,'episode':episode,'step':step,
                                             'success_radius_m':config['success_radius_m'],'failure':'NORMAL',
                                             'raw_sha256':{name:digest(observation['frames'][name]) for name in ('front','down')}}))
            temporary.replace(Path(args.output)/'live.json')
    if env.collided():reason='collision'
    await env.close()
    if args.live:
        (Path(args.output)/'live_result.json').write_text(json.dumps({'episode':episode['id'],'reason':reason,'stopped':stopped,
                                                                      'final_distance_m':steps[-1]['distance_m'] if steps else None}))
    return summarise(episode,steps,config,reason,stopped),steps


async def main(args):
    config=load_config();oft=load_oft_config(args.oft_config) if args.oft_config else load_oft_config()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    policy=None
    if args.policy=='baseline':
        from src.integration.aerovla_int4_loader import AeroVLAInt4
        policy=AeroVLAInt4(MANIFEST)
    elif args.policy=='oft':
        from src.aerovla_oft.model import AeroVLAOFT
        saved=json.loads((Path(args.checkpoint)/'manifest.json').read_text())['config']
        oft=saved;policy=AeroVLAOFT(MANIFEST,saved,checkpoint=args.checkpoint)
    client=ProjectAirSimClient(address=args.host);client.connect();client.socket_services.recv_timeout=60000
    env=SearchEnv(client,config,oft,output)
    results_path=output/'results.json'
    done=json.loads(results_path.read_text())['episodes'] if results_path.exists() and args.resume else []
    done=[item for item in done if not item.get('error')]
    finished={item['id'] for item in done};status=0
    episodes=[(case,target,seed) for seed in range(args.seed_start,args.seed_start+args.episodes) for case in args.cases for target in args.targets]
    record_root=Path(args.record) if args.record else None
    try:
        for case,target,seed in episodes:
            identifier=f'{case}-{target}-{seed}'
            if identifier in finished:continue
            try:
                if not env.landmarks:env.prepare()
                centre=env.landmarks[target]['position']
                episode=make_episode(config,case,target,centre,seed)
                await env.reset(episode)
                record={'root':record_root,'rel':f'blocks/{identifier}'} if record_root else None
                summary,steps=await run_episode(env,episode,policy,args,record)
            except Exception:
                summary={'id':identifier,'case':case,'target':target,'seed':seed,'error':traceback.format_exc()};steps=[]
                print(summary['error'],flush=True)
                if 'Connection' in summary['error']:status=1
            summary['policy']=args.policy
            done.append(summary)
            with (output/'steps.jsonl').open('a') as file:
                for step in steps:file.write(json.dumps({'episode':identifier,**step})+'\n')
            if record_root and steps:
                (record_root/'episodes').mkdir(parents=True,exist_ok=True)
                (record_root/'episodes'/f'{identifier}.json').write_text(json.dumps({'summary':summary,'traj_rel_dir':f'blocks/{identifier}','steps':steps}))
            results_path.write_text(json.dumps({'policy':args.policy,'checkpoint':args.checkpoint,'config':config,'oft':oft,'episodes':done},indent=1))
            print(f'EPISODE {identifier} {summary.get("reason")} success={summary.get("success")} steps={summary.get("steps")} '
                  f'initial={summary.get("initial_distance_m",float("nan")):.1f} final={summary.get("final_distance_m",float("nan")):.1f} '
                  f'min={summary.get("minimum_distance_m",float("nan")):.1f} seen0={summary.get("initially_seen")} acquired={summary.get("acquired_step")}',flush=True)
            if status:break
    finally:
        client.disconnect()
    raise SystemExit(status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True)
    parser.add_argument('--policy',choices=('baseline','teacher','oft'),required=True)
    parser.add_argument('--checkpoint');parser.add_argument('--oft-config')
    parser.add_argument('--cases',nargs='+',default=['A','B','C']);parser.add_argument('--targets',nargs='+',default=['blue_cone','orange_ball'])
    parser.add_argument('--episodes',type=int,default=2,help='seeds per case and target');parser.add_argument('--seed-start',type=int,default=0)
    parser.add_argument('--output',default=str(ROOT/'outputs/visual_search/run'));parser.add_argument('--record',help='dataset root to write teacher episodes into')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--live',action='store_true')
    arguments=parser.parse_args()
    if arguments.policy=='oft' and not arguments.checkpoint:parser.error('--checkpoint is required for --policy oft')
    asyncio.run(main(arguments))
