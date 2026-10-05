"""Final direct-WSL single step + ten decisions; model and upstream unchanged."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from pathlib import Path
if '--blur-demo' in sys.argv:
    # Publish ownership before slow ML imports/loading so cancellation can locate this process.
    early_output = Path(__file__).resolve().parents[2]/'outputs/failure_demo'
    early_output.mkdir(parents=True, exist_ok=True)
    (early_output/'worker-pid.txt').write_text(str(os.getpid()))
import cv2
import numpy as np
import psutil
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.integration.aerovla_int4_loader import AeroVLAInt4
from src.integration.projectairsim_observation_adapter import adapt_state,make_mosaic
from src.integration.projectairsim_action_adapter import convert_action,execute_action
from projectairsim_probe import prepare_config,CONFIG
from projectairsim import ProjectAirSimClient,World,Drone
from projectairsim.utils import unpack_image
from validate_aerovla_int4 import memory
from src.integration.blur_demo_support import BlurDemoSession, DemoQuit
OUT=ROOT/'outputs/communication_final'
INSTRUCTION='The target is 3 meters away and 0 degrees from you. Find the colored blocks ahead. Please control the drone.'

def host_memory():return json.loads((OUT/'windows-latest.json').read_text(encoding='utf-8-sig'))

def debug_panel(front,down,state,label,inference=None):
    panel=np.zeros((500,640,3),dtype=np.uint8)
    panel[:240,:320]=cv2.resize(front,(320,240));panel[:240,320:]=cv2.resize(down,(320,240))
    lines=[f'Direct WSL | Step {label} | Front / Down',f'Position NED: {state["position"]}',INSTRUCTION]
    if inference:lines += [f'Generation: {inference["inference_ms"]:.1f} ms',
        'Raw: '+inference['raw_output'].split('Action:')[-1],'Parsed: '+str(inference['parsed_action'])]
    y=260
    for line in lines:
        for wrapped in textwrap.wrap(line,84):
            cv2.putText(panel,wrapped,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA);y+=19
    cv2.imwrite(str(OUT/f'debug_{label}.png'),panel)
    temp=OUT/'debug_publish.png';cv2.imwrite(str(temp),panel);os.replace(temp,OUT/'debug_latest.png')

async def main(args):
    global OUT
    session = None
    if args.blur_demo:
        OUT = ROOT/'outputs/failure_demo'
        session = BlurDemoSession(ROOT, OUT, args.steps)
        (OUT/'worker-pid.txt').write_text(str(os.getpid()))
    else:
        gate_a=json.loads((OUT/'gate-a.json').read_text())
        gate_b=json.loads((OUT/'gate-b.json').read_text())
        if gate_a.get('status')!='PASS' or gate_b.get('status')!='PASS':
            raise RuntimeError('Direct route requires successful Gate A and tested Gate B levels')
    result={'status':'RUNNING','route':'Direct WSL NNG -> Windows host','address':args.host,
            'single_step':None,'closed_loop':[],'errors':[],'domain_check':None}
    stop=threading.Event();client=None;drone=None;collision_events=[];flight_start_stamp=None
    def sample():
        with (OUT/'closed-loop-wsl-resources.jsonl').open('w') as file:
            while not stop.is_set():
                row={'epoch':time.time(),'process_rss_GiB':psutil.Process().memory_info().rss/1024**3,
                    'wsl_memory':dict(psutil.virtual_memory()._asdict()),'wsl_swap':dict(psutil.swap_memory()._asdict()),
                    'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits'],text=True).strip()}
                file.write(json.dumps(row)+'\n');file.flush();stop.wait(.5)
    def save():(OUT/'closed-loop.json').write_text(json.dumps(result,indent=2))
    def log(row):
        with (OUT/'closed-loop-log.jsonl').open('a') as file:file.write(json.dumps({'epoch':time.time(),**row})+'\n')
    def collision_callback(_,message):
        collision_events.append({'received_epoch':time.time(),'message':message})
    def check_collisions():
        # Official CollisionInfoMessage has a collision event timestamp, no has_collided flag.
        recent=[event for event in collision_events if flight_start_stamp is not None
                and event['message'].get('time_stamp',0)>flight_start_stamp]
        if recent:raise RuntimeError('Collision during flight: '+json.dumps(recent[-1]))
    def state_is_finite(state):
        if not all(math.isfinite(v) for key in ('position','orientation','velocity') for v in state[key]):
            raise RuntimeError('Non-finite simulator state')
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    try:
        (OUT/'active-stage.txt').write_text('Gate D model load')
        if session:
            session.update(phase='Loading OpenVLA NF4 + AeroVLA LoRA', instruction=INSTRUCTION)
            session.poll_control()
        torch.cuda.reset_peak_memory_stats()
        model=AeroVLAInt4(ROOT/'outputs/integration/model-downloads.json')
        result['load_info']=model.load_info;result['after_load']=memory();save()
        if session:
            session.poll_control()
            session.update(phase='Connecting to Project AirSim')
        # Start the NNG client only after model loading, with no eager proxy session.
        if session:
            config_path = session.prepare_config()
        else:
            prepare_config()
            config_path = CONFIG
        client=ProjectAirSimClient(address=args.host)
        start=time.perf_counter();client.connect();result['connect_ms']=(time.perf_counter()-start)*1000
        start=time.perf_counter();world=World(client,'scene_basic_drone.jsonc',delay_after_load_sec=2,sim_config_path=str(config_path))
        result['topic_init_ms']=(time.perf_counter()-start)*1000
        drone=Drone(client,world,'Drone1')
        client.subscribe(drone.robot_info['collision_info'],collision_callback)
        if session:
            client.subscribe(drone.sensors['Chase']['scene_camera'], session.chase_callback)
        rest=drone.get_ground_truth_kinematics();ground_z=rest['pose']['position']['z']
        if session:
            session.update(phase='Takeoff', ground_z=ground_z)
        drone.enable_api_control();drone.arm()
        takeoff_return=await (await drone.takeoff_async(timeout_sec=20))
        await (await drone.hover_async());await asyncio.sleep(.7)
        after=drone.get_ground_truth_kinematics();initial=adapt_state(after)
        result['takeoff']={'return':takeoff_return,'before':rest,'after':after,
            'actual_climb_m':ground_z-after['pose']['position']['z'],'landed_state':drone.get_landed_state()}
        if result['takeoff']['actual_climb_m']<.5 or result['takeoff']['landed_state']==0:
            raise RuntimeError('Actual takeoff criterion failed')
        flight_start_stamp=after['time_stamp']
        heading=Rotation.from_quat(initial['orientation']).as_euler('xyz')[2]
        target=[initial['position'][0]+3*math.cos(heading),initial['position'][1]+3*math.sin(heading),initial['position'][2]]
        result['target']={'position':target,'kind':'synthetic integration-only; not a benchmark landmark'}
        save();log({'event':'takeoff','takeoff':result['takeoff'],'target':result['target']})

        async def step(label):
            check_collisions();wall=time.perf_counter();transport_start=time.perf_counter()
            if session:
                session.poll_control()
                session.update(phase='Receiving live Front / Down', step=int(label))
            row={'step':label,'instruction':INSTRUCTION,'target':target,'camera':{}}
            row['state_before']=drone.get_ground_truth_kinematics();state=adapt_state(row['state_before']);state_is_finite(state)
            frames=[]
            for sensor,name in (('FrontCamera','front'),('DownCamera','down')):
                start=time.perf_counter();message=drone.get_images(sensor,[0])[0]
                latency=(time.perf_counter()-start)*1000;frame=unpack_image(message)
                if message['encoding']!='BGR' or frame.shape!=(256,256,3):raise RuntimeError('Unexpected camera contract')
                row['camera'][name]={'rpc_ms':latency,'timestamp':message['time_stamp'],
                    'sha256':hashlib.sha256(frame.tobytes()).hexdigest(),'shape':list(frame.shape)};frames.append(frame)
                if not session:
                    path=OUT/f'{name}_{label}.png';cv2.imwrite(str(path),frame)
                    row['camera'][name]['path']=str(path)
            row['observation_transport_ms']=(time.perf_counter()-transport_start)*1000
            if session:
                raw_frames = frames
                frames, row['failure'] = session.inject(raw_frames, int(label), state, INSTRUCTION)
                inference = model.infer(*frames, state, target, INSTRUCTION,
                                        trace_inputs=True, reference_frames=raw_frames)
                session.verify_input(inference, row['failure'])
                session.poll_control()
            else:
                debug_panel(*frames,state,label)
                inference=model.infer(*frames,state,target,INSTRUCTION)
            row['inference']=inference
            if inference['parsed_action'] is None:log({'event':'invalid_action',**row});raise RuntimeError(inference['parse_error'])
            if not session:
                debug_panel(*frames,state,label,inference);make_mosaic(*frames).save(OUT/f'mosaic_{label}.png')
            check_collisions()
            # Re-read pose immediately before converting/issuing a command after generation.
            execution_state=adapt_state(drone.get_ground_truth_kinematics());state_is_finite(execution_state)
            row['execution_state']=execution_state
            row['clipped_action']=convert_action(inference['parsed_action'],execution_state,ground_z)
            if session:
                session.update(phase='Executing bounded action', clipped_action=row['clipped_action'], state=execution_state)
            command_start=time.perf_counter()
            if session:
                row['command_returns'] = await asyncio.wait_for(execute_action(drone, row['clipped_action']), timeout=20)
            else:
                row['command_returns']=await execute_action(drone,row['clipped_action'])
            row['command_rpc_wall_ms']=(time.perf_counter()-command_start)*1000
            await asyncio.sleep(.5);check_collisions()
            row['state_after']=drone.get_ground_truth_kinematics();after_state=adapt_state(row['state_after']);state_is_finite(after_state)
            row['movement_m']=float(np.linalg.norm(np.array(after_state['position'])-np.array(execution_state['position'])))
            row['orientation_delta_rad']=float((Rotation.from_quat(execution_state['orientation']).inv()*Rotation.from_quat(after_state['orientation'])).magnitude())
            row['clearance_m']=ground_z-after_state['position'][2];row['landed_state']=drone.get_landed_state()
            if row['landed_state']==0 or not .75 <= row['clearance_m'] <= 4.05:
                log({'event':'abnormal_state',**row});raise RuntimeError('Unexpected landed/altitude state')
            row['memory']=memory();row['windows']=host_memory();row['step_wall_ms']=(time.perf_counter()-wall)*1000
            # Command wall includes the actual1s flight/yaw; do not label it wire latency.
            log({'event':'step',**row});print(f'STEP {label}: {inference["parsed_action"]["bins"]} movement={row["movement_m"]:.4f}m inference={inference["inference_ms"]:.1f}ms',flush=True)
            if session:
                session.update(phase='Step complete', completed_steps=int(label), state=after_state,
                               movement_m=row['movement_m'], inference=inference, step_wall_ms=row['step_wall_ms'],
                               command_returns=row['command_returns'])
            return row

        if not session:
            (OUT/'active-stage.txt').write_text('Gate D single step')
            result['single_step']=await step('single');save()
            if result['single_step']['clipped_action']['stop'] or result['single_step']['movement_m']<.03:
                raise RuntimeError('Single-step action did not cause measurable movement')
        (OUT/'active-stage.txt').write_text('Gate E ten-step loop')
        count = args.steps if session else 10
        for index in range(1,count+1):
            if session and args.auto_test and index in (3, 5):
                session.auto_toggle(index)
            result['closed_loop'].append(await step(f'{index:02d}'));save()
        result['loop_summary']={'steps':count,
            'inference_ms':{'mean':float(np.mean([r['inference']['inference_ms'] for r in result['closed_loop']])),
                'median':float(np.median([r['inference']['inference_ms'] for r in result['closed_loop']])),
                'p95':float(np.percentile([r['inference']['inference_ms'] for r in result['closed_loop']],95))},
            'observation_transport_mean_ms':float(np.mean([r['observation_transport_ms'] for r in result['closed_loop']])),
            'command_rpc_wall_mean_ms':float(np.mean([r['command_rpc_wall_ms'] for r in result['closed_loop']])),
            'step_wall_mean_ms':float(np.mean([r['step_wall_ms'] for r in result['closed_loop']])),
            'movement_total_m':float(sum(r['movement_m'] for r in result['closed_loop']))}
        frames_unique=len({(r['camera']['front']['sha256'],r['camera']['down']['sha256']) for r in result['closed_loop']})
        actions=[tuple(r['inference']['parsed_action']['bins'] or ['LAND']) for r in result['closed_loop']]
        most_common=max(actions.count(a) for a in set(actions))
        result['domain_check']={'distinct_live_frame_pairs':frames_unique,'unique_actions':len(set(actions)),
            'modal_action_ratio':most_common/len(actions),'actions':[list(a) for a in actions],
            'scope':'one Blocks flight, same synthetic instruction/target; not independent scene/target sweep',
            'observation':'Inconclusive' if len(set(actions))==1 else 'not obviously degenerate on this sequence'}
        result['peak']=memory();result['status']='PASS';save()
    except DemoQuit:
        result['status'] = 'STOPPED'; save()
    except Exception:
        result['status']='FAIL';result['errors'].append(traceback.format_exc());print(result['errors'][-1],flush=True)
        result['failure_memory']=memory();log({'event':'failure','error':result['errors'][-1]});save()
    finally:
        (OUT/'active-stage.txt').write_text('Cleanup')
        if session:
            session.update(phase='Landing and cleanup')
        # Collision events observed after the flight start remain in the raw result.
        result['collision_events']=collision_events
        if drone:
            try:
                result['land_return']=await (await drone.land_async(timeout_sec=20))
                drone.disarm();await asyncio.sleep(2);result['final_landed_state']=drone.get_landed_state();drone.disable_api_control()
            except Exception:result['cleanup_error']=traceback.format_exc()
        if client:client.disconnect()
        stop.set();sampler.join(timeout=2);save()
        if session:
            session.update(status=result['status'], phase='Finished', errors=result['errors'],
                           cleanup_error=result.get('cleanup_error'), result_file='closed-loop.json')
    print(f'FINAL {result["status"]} loop_steps={len(result["closed_loop"])}',flush=True)
    if result['status'] not in ('PASS','STOPPED') or result.get('cleanup_error'):raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True)
    parser.add_argument('--blur-demo', action='store_true')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--auto-test', action='store_true')
    parser.add_argument('--run-token')
    args = parser.parse_args()
    if not 1 <= args.steps <= 60:
        parser.error('--steps must be within 1..60')
    if args.auto_test and (not args.blur_demo or args.steps != 6):
        parser.error('--auto-test requires --blur-demo --steps 6')
    asyncio.run(main(args))
