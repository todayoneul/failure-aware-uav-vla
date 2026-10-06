"""Interactive goals; all navigation commands come from existing AeroVLA actions."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import asyncio
import json
import math
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/mission_demo'
OUT.mkdir(parents=True,exist_ok=True)
if '--mission-demo' in sys.argv:(OUT/'worker-pid.txt').write_text(str(os.getpid()))
import cv2
import numpy as np
from projectairsim import ProjectAirSimClient,World,Drone
from projectairsim.utils import unpack_image
from src.integration.blur_demo_support import BlurDemoSession,DemoQuit,publish_json,publish_image
from src.integration.projectairsim_observation_adapter import adapt_state
from src.integration.projectairsim_action_adapter import convert_action
from src.mission.manager import MissionManager,MissionTarget,TERMINAL
from src.mission.scene import OverviewScene,prepare_mission_config
from src.mission.flight import landing_source,land_and_confirm,guarded_execute
from src.mission.control import pending_requests


def instruction_for(target):
    # Keep the exact delimiters expected by the upstream wrapper. Numerical goal
    # position enters the existing semantic direction function, not a new parser.
    return 'The target is selected on the overview and 0 degrees from you. Find the selected target location. Please control the drone.'


async def main(args):
    limits=json.loads((ROOT/'configs/mission_limits.json').read_text());limits['max_steps']=args.steps
    manager=MissionManager(**limits)
    session=BlurDemoSession(ROOT,OUT,args.steps)
    session.update(mission=manager.snapshot(),capabilities={'GO_TO':True,'GO_TO_AND_HOVER':False,'GO_TO_AND_LAND':False},
                   phase='Preparing mission map',trajectory=[])
    capabilities=session.telemetry['capabilities']
    verified=json.loads((ROOT/'configs/mission_capabilities.json').read_text())
    capabilities['GO_TO_AND_HOVER']=verified['go_to_live_verified']
    capabilities['GO_TO_AND_LAND']=verified['hover_live_verified']
    history=[];trajectory=[];rows=[]
    client=None;drone=None;model=None;airborne=False;mission_id=0;last_request=0
    flight_stamp=None;collision_events=[];program_status='RUNNING'
    last_state=None;last_state_epoch=None;initial_distance=None;landing_controller=None

    def state():
        nonlocal last_state,last_state_epoch
        result=adapt_state(drone.get_ground_truth_kinematics())
        if not all(math.isfinite(v) for key in ('position','orientation','velocity') for v in result[key]):
            raise RuntimeError('Nonfinite vehicle state')
        last_state=result;last_state_epoch=time.time()
        return result

    def update(current,phase=None):
        snapshot=manager.snapshot(current)
        session.update(mission=snapshot,state=current,trajectory=trajectory,capabilities=capabilities,
                       mission_id=mission_id,**({'phase':phase} if phase else {}))

    def save_history():publish_json(OUT/'mission-results.json',{'missions':history,'program_status':program_status})

    def save_mission(current,observation_fresh=True):
        entry={**manager.snapshot(current),'mission_id':mission_id,'initial_distance_m':initial_distance,
               'final_distance_m':manager.metrics(current)['horizontal_m'],'vla_steps':len(rows),
               'steps':rows.copy(),'trajectory':trajectory.copy(),'landing_controller':landing_controller,
               'final_observation_status':'fresh' if observation_fresh else 'last_validated_before_exit',
               'final_observation_epoch':last_state_epoch}
        history.append(entry);save_history()
        if manager.status=='SUCCESS':
            if manager.kind=='GO_TO' and rows:capabilities['GO_TO_AND_HOVER']=True
            if manager.kind=='GO_TO_AND_HOVER':capabilities['GO_TO_AND_LAND']=True
        update(current,'Mission '+manager.status+' - R resets, select next target')

    async def hover():
        result=await asyncio.wait_for(await drone.hover_async(),timeout=10)
        if not result:raise RuntimeError('Hover command rejected')

    try:
        client=ProjectAirSimClient(address=args.host);client.connect()
        client.socket_services.recv_timeout=10000
        world=World(client,'scene_basic_drone.jsonc',delay_after_load_sec=2,sim_config_path=str(prepare_mission_config(ROOT,OUT)))
        drone=Drone(client,world,'Drone1')
        client.subscribe(drone.robot_info['collision_info'],lambda _,message:collision_events.append(message))
        client.subscribe(drone.sensors['Chase']['scene_camera'],session.chase_callback)
        overview=OverviewScene(world,drone,OUT)
        ground=drone.get_ground_truth_kinematics();ground_z=ground['pose']['position']['z']
        session.update(ground_z=ground_z)
        current=state()
        overview.capture(current,'top');publish_image(OUT/'overview_top.png',cv2.imread(str(OUT/'overview_1.png')))
        elevated_image,_=overview.capture(current,'elevated');publish_image(OUT/'overview_elevated.png',elevated_image)
        last_map=0
        while True:
            control=session.poll_control();current=state()
            if time.monotonic()-last_map>.4:
                overview.capture(current,control.get('overview_view','elevated'),control.get('overview_height',50));last_map=time.monotonic()
            for request_id,request in pending_requests(control,last_request):
                last_request=request_id
                try:
                    if request['action']=='select':
                        point,source=overview.select(request['frame_id'],request['pixel'])
                        if manager.status in TERMINAL:manager.reset()
                        manager.select(MissionTarget(f'Visible surface ({point[0]:.2f}, {point[1]:.2f})',tuple(point),source))
                        update(current,'Target selected - G starts GO_TO')
                    elif request['action']=='reset':
                        manager.reset();trajectory=[];rows=[];update(current,'Mission reset - choose a target')
                    elif request['action']=='start':
                        kind=request['type']
                        if not capabilities.get(kind):raise ValueError('Complete the preceding mission type first')
                        if manager.status!='TARGET_SELECTED':raise ValueError('Select a visible map target first')
                        if model is None:
                            session.update(phase='Loading cached OpenVLA NF4 + AeroVLA LoRA')
                            from src.integration.aerovla_int4_loader import AeroVLAInt4
                            model=AeroVLAInt4(ROOT/'outputs/integration/model-downloads.json')
                            session.poll_control()
                        if not airborne:
                            session.update(phase='Takeoff for mission')
                            drone.enable_api_control();drone.arm()
                            airborne=True  # Takeoff/hover may fail after the motors have started.
                            takeoff=await asyncio.wait_for(await drone.takeoff_async(timeout_sec=20),timeout=25)
                            await hover();await asyncio.sleep(.5)
                            sample=drone.get_ground_truth_kinematics();current=adapt_state(sample)
                            if ground_z-current['position'][2]<.5 or drone.get_landed_state()==0:
                                raise RuntimeError('Actual takeoff criterion failed')
                            airborne=True;flight_stamp=sample['time_stamp']
                        manager.start(kind,current);mission_id+=1;rows=[];trajectory=[current['position']]
                        initial_distance=manager.metrics(current)['horizontal_m'];landing_controller=None
                        session.update(inference=None,clipped_action=None,input_verified=None,input_files={},input_step=0,completed_steps=0)
                        update(current,'AeroVLA goal navigation')
                    session.update(mission_request_ack=request_id,control_message=None)
                except ValueError as error:session.update(mission_request_ack=request_id,control_message=str(error))

            if manager.status in ('NAVIGATING','HOVERING'):
                if any(event.get('time_stamp',0)>flight_stamp for event in collision_events):manager.fail('collision')
                if not .75<=ground_z-current['position'][2]<=4.05:manager.fail('extreme_altitude')
            if manager.status=='NAVIGATING':
                manager.observe(current)
                if manager.status=='NAVIGATING':
                    control=session.poll_control();step=manager.steps+1;raw=[];camera={}
                    for sensor,name in (('FrontCamera','front'),('DownCamera','down')):
                        message=drone.get_images(sensor,[0])[0];frame=unpack_image(message)
                        if message['encoding']!='BGR' or frame.shape!=(256,256,3):raise RuntimeError('AI camera contract changed')
                        raw.append(frame);camera[name]={'time_stamp':message['time_stamp']}
                    instruction=instruction_for(manager.target)
                    used,failure=session.inject(raw,step,current,instruction)
                    inference=model.infer(*used,current,manager.goal_position,instruction,trace_inputs=True,reference_frames=raw)
                    session.verify_input(inference,failure);session.poll_control()
                    action=inference['parsed_action']
                    if action is None:manager.fail('invalid_action');raise RuntimeError(inference['parse_error'])
                    execution=state();command=convert_action(action,execution,ground_z)
                    session.update(phase='Executing AeroVLA bounded action',clipped_action=command)
                    returns=await guarded_execute(drone,manager,execution,ground_z,collision_events,flight_stamp,command)
                    if returns is None:
                        update(execution,'Navigation stopped before command: '+manager.status)
                        continue
                    if not all(returns.values()):
                        manager.fail('command_rejected')
                        raise RuntimeError(f'Flight command rejected: {returns}')
                    await asyncio.sleep(.3);after=state()
                    if any(event.get('time_stamp',0)>flight_stamp for event in collision_events):manager.fail('collision')
                    if not .75<=ground_z-after['position'][2]<=4.05:manager.fail('extreme_altitude')
                    manager.steps+=1;manager.observe(after);trajectory.append(after['position'])
                    row={'mission_id':mission_id,'step':manager.steps,'epoch':time.time(),'position':after['position'],
                         'target':manager.goal_position,'distance':manager.metrics(after),'instruction':instruction,
                         'inference':inference,'parsed_action':action,'executed_action':command,'command_returns':returns,
                         'failure':failure,'camera':camera,'mission_state':manager.status}
                    rows.append(row)
                    with (OUT/'mission-steps.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
                    session.update(completed_steps=manager.steps,inference=inference,clipped_action=command)
                    update(after,'AeroVLA goal navigation')
                    print(f'MISSION {mission_id} step={manager.steps} distance={row["distance"]["horizontal_m"]:.3f} state={manager.status}',flush=True)
            if manager.status=='HOVERING':
                await hover();await asyncio.sleep(.2);current=state();manager.observe(current)
                update(current,'Holding target position / velocity check')
            if manager.status=='LANDING':
                landing_controller=landing_source(rows[-1]['parsed_action'] if rows else None,manager.status)
                await hover();session.update(phase=landing_controller)
                confirmation=await land_and_confirm(drone)
                session.update(landing_confirmation=confirmation)
                current=state();airborne=False
                manager.confirm_landing(current,drone.get_landed_state()==0)
                if manager.status!='SUCCESS':manager.fail('landing_not_confirmed')
                update(current,'Landing evaluation')
            if manager.status in TERMINAL and (not history or history[-1]['mission_id']!=mission_id):
                current=state();save_mission(current)
            if manager.status in ('IDLE','TARGET_SELECTED'):
                if airborne:await hover()
                update(state(),'Choose target on Overview; G starts')
            elif manager.status in TERMINAL:
                # Hover is a native pause, independent of target coordinates.
                # Keep the drone stopped while the user reads a result or selects a new goal.
                if airborne:await hover()
                session.update(state=state())
            await asyncio.sleep(.1)
    except DemoQuit:
        manager.abort();program_status='STOPPED'
    except Exception:
        manager.fail('runtime_error');program_status='FAIL';session.update(error=traceback.format_exc())
        print(traceback.format_exc(),flush=True)
    finally:
        if manager.target and manager.kind and (not history or history[-1]['mission_id']!=mission_id):
            # A broken RPC must not erase the failed/aborted mission record.
            save_mission(last_state,observation_fresh=False)
        if drone:
            try:
                if airborne and drone.get_landed_state()!=0:
                    session.update(cleanup_landing=await land_and_confirm(drone))
                else:drone.disarm()
                drone.disable_api_control()
            except Exception:
                program_status='FAIL';session.update(cleanup_error=traceback.format_exc())
        if client:client.disconnect()
        save_history()
        final_snapshot=manager.snapshot()
        if manager.kind and history and history[-1]['mission_id']==mission_id:
            final_snapshot['errors']=history[-1]['errors']
        session.update(status=program_status,phase='Finished',mission=final_snapshot)
    if program_status=='FAIL':raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True)
    parser.add_argument('--mission-demo',action='store_true');parser.add_argument('--steps',type=int,default=60)
    parser.add_argument('--run-token');args=parser.parse_args()
    if not 1<=args.steps<=60:parser.error('--steps within 1..60')
    asyncio.run(main(args))
