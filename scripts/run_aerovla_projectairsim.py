"""Actual single-step gate followed by exactly ten model-driven closed-loop steps."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
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
import cv2
import numpy as np
import psutil
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.integration.aerovla_int4_loader import AeroVLAInt4
from src.integration.projectairsim_observation_adapter import adapt_state, make_mosaic
from src.integration.projectairsim_action_adapter import convert_action, execute_action
from projectairsim_probe import prepare_config, CONFIG
from projectairsim import ProjectAirSimClient, World, Drone
from projectairsim.utils import unpack_image
from validate_aerovla_int4 import memory
OUT=ROOT/'outputs/integration'
INSTRUCTION='The target is 3 meters away and 0 degrees from you. Find the colored blocks ahead. Please control the drone.'

def debug_panel(front,down,state,label,inference=None):
    panel=np.zeros((500,640,3),dtype=np.uint8)
    panel[:240,:320]=cv2.resize(front,(320,240))
    panel[:240,320:]=cv2.resize(down,(320,240))
    lines=[f'Step: {label} | Front RGB (left) / Down RGB (right)',
           f'Position NED: {state["position"]}',
           'Instruction: '+INSTRUCTION]
    if inference:
        lines += [f'Inference: {inference["inference_ms"]:.1f} ms',
                  'Raw action: '+inference['raw_output'].split('Action:')[-1],
                  'Parsed: '+str(inference['parsed_action'])]
    y=262
    for line in lines:
        for wrapped in textwrap.wrap(line,85):
            cv2.putText(panel,wrapped,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.42,(240,240,240),1,cv2.LINE_AA)
            y+=19
    cv2.imwrite(str(OUT/f'debug_{label}.png'),panel)
    temp=OUT/'debug_publish.png';cv2.imwrite(str(temp),panel)
    os.replace(temp,OUT/'debug_latest.png')

async def main():
    result={'status':'RUNNING','single_step':None,'closed_loop':[], 'errors':[]}
    stop=threading.Event()
    def sample():
        with (OUT/'combined-resource-samples.jsonl').open('w') as file:
            while not stop.is_set():
                row={'epoch':time.time(),'process_rss_GiB':psutil.Process().memory_info().rss/1024**3,
                     'wsl_system_used_GiB':psutil.virtual_memory().used/1024**3,
                     'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                         '--format=csv,noheader,nounits'],text=True).strip()}
                file.write(json.dumps(row)+'\n');file.flush();stop.wait(.5)
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    client=None;drone=None
    def save(): (OUT/'end-to-end.json').write_text(json.dumps(result,indent=2))
    def log(row):
        with (OUT/'integration_log.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch':time.time(),**row})+'\n')
    try:
        torch.cuda.reset_peak_memory_stats()
        model=AeroVLAInt4(OUT/'model-downloads.json')
        result['model_load']=model.load_info
        result['after_model_load']=memory();save()
        prepare_config()
        client=ProjectAirSimClient(address='127.0.0.1',port_topics=18989,port_services=18990)
        client.connect()
        world=World(client,'scene_basic_drone.jsonc',delay_after_load_sec=2,sim_config_path=str(CONFIG))
        drone=Drone(client,world,'Drone1')
        rest=drone.get_ground_truth_kinematics()
        ground_z=rest['pose']['position']['z']
        drone.enable_api_control();drone.arm()
        result['takeoff']={'return':await (await drone.takeoff_async(timeout_sec=20)), 'before':rest}
        await (await drone.hover_async());await asyncio.sleep(.7)
        after=drone.get_ground_truth_kinematics()
        result['takeoff']['after']=after
        result['takeoff']['landed_state']=drone.get_landed_state()
        result['takeoff']['actual_climb_m']=ground_z-after['pose']['position']['z']
        if result['takeoff']['actual_climb_m']<.5 or result['takeoff']['landed_state']==0:
            raise RuntimeError('Actual takeoff state criterion failed')
        initial=adapt_state(after)
        heading=Rotation.from_quat(initial['orientation']).as_euler('xyz')[2]
        target=[initial['position'][0]+3*math.cos(heading),initial['position'][1]+3*math.sin(heading),initial['position'][2]]
        result['target']={'position':target,'kind':'integration-only synthetic target; not a TravelUAV benchmark goal'}
        save();log({'event':'takeoff','result':result['takeoff'],'target':result['target']})

        async def step(label):
            row={'step':label,'instruction':INSTRUCTION,'target_position':target,
                 'state_before':drone.get_ground_truth_kinematics(),'camera':{}}
            state=adapt_state(row['state_before'])
            frames=[]
            for sensor,name in (('FrontCamera','front'),('DownCamera','down')):
                start=time.perf_counter();message=drone.get_images(sensor,[0])[0]
                latency=(time.perf_counter()-start)*1000
                frame=unpack_image(message)
                if message['encoding']!='BGR': raise RuntimeError('Unexpected camera channel encoding')
                path=OUT/f'{name}_{label}.png';cv2.imwrite(str(path),frame)
                row['camera'][name]={'path':str(path),'rpc_ms':latency,'time_stamp':message['time_stamp'],
                    'shape':list(frame.shape),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                frames.append(frame)
            debug_panel(*frames,state,label)
            start=time.perf_counter()
            inference=model.infer(*frames,state,target,INSTRUCTION)
            row['inference']=inference
            if inference['parsed_action'] is None:
                log({**row,'event':'invalid_output'})
                raise RuntimeError(inference['parse_error'])
            debug_panel(*frames,state,label,inference)
            make_mosaic(*frames).save(OUT/f'mosaic_{label}.png')
            command=convert_action(inference['parsed_action'],state,ground_z)
            row['converted_action']=command
            row['command_returns']=await execute_action(drone,command)
            await asyncio.sleep(.5)
            row['state_after']=drone.get_ground_truth_kinematics()
            p1=np.array(state['position']);s2=adapt_state(row['state_after']);p2=np.array(s2['position'])
            row['movement_m']=float(np.linalg.norm(p2-p1))
            q1=Rotation.from_quat(state['orientation']);q2=Rotation.from_quat(s2['orientation'])
            row['orientation_delta_rad']=float((q1.inv()*q2).magnitude())
            row['landed_state']=drone.get_landed_state()
            row['memory']=memory()
            row['observe_infer_command_ms']=(time.perf_counter()-start)*1000
            log({'event':'step',**row})
            print(f'STEP {label} bins={inference["parsed_action"]["bins"]} movement={row["movement_m"]:.6f}m inference={inference["inference_ms"]:.2f}ms',flush=True)
            return row

        result['single_step']=await step('single');save()
        if result['single_step']['converted_action']['stop'] or result['single_step']['movement_m']<.03:
            result['status']='SINGLE_STEP_NO_MOVEMENT';save();return
        # Single-step gate has passed. Only now allow the ten-step closed loop.
        for index in range(1,11):
            row=await step(f'{index:02d}')
            result['closed_loop'].append(row);save()
            if row['landed_state']==0:
                raise RuntimeError('Drone unexpectedly landed during closed loop')
        result['status']='PASS';result['combined_peak']=memory();save()
    except Exception:
        result['status']='FAIL';result['errors'].append(traceback.format_exc())
        print(result['errors'][-1],flush=True)
        result['failure_memory']=memory();save();log({'event':'failure','error':result['errors'][-1]})
    finally:
        if drone:
            try:
                result['land_return']=await (await drone.land_async(timeout_sec=20))
                drone.disarm();await asyncio.sleep(2)
                result['final_landed_state']=drone.get_landed_state()
                result['final_state']=drone.get_ground_truth_kinematics()
                drone.disable_api_control()
            except Exception:
                result['cleanup_error']=traceback.format_exc()
        if client:client.disconnect()
        stop.set();sampler.join(timeout=2);save()
    print(f'FINAL {result["status"]} completed_loop_steps={len(result["closed_loop"])}',flush=True)
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
