"""Model-free fresh-client reconnect trials. No timeout inflation/upstream patch."""
import argparse
import gc
import json
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from projectairsim_probe import prepare_config, CONFIG
from projectairsim import ProjectAirSimClient,World,Drone
from projectairsim.utils import unpack_image
OUT=ROOT/'outputs/communication_final'

def trial(host,ports,iteration,phase):
    row={'iteration':iteration,'phase':phase,'epoch':time.time(),'address':host,'ports':ports,
         'success':False,'connect_ms':None,'topic_init_ms':None,'front_ms':None,'state_ms':None}
    client=None;stage='TCP_preflight';start=time.perf_counter()
    try:
        for port in ports:
            with socket.create_connection((host,port),timeout=1.5):pass
        row['tcp_preflight_ms']=(time.perf_counter()-start)*1000
        stage='NNG_connect';start=time.perf_counter()
        client=ProjectAirSimClient(address=host,port_topics=ports[0],port_services=ports[1])
        client.connect();row['connect_ms']=(time.perf_counter()-start)*1000
        stage='topic_initialization';start=time.perf_counter()
        world=World(client,'scene_basic_drone.jsonc',delay_after_load_sec=0,sim_config_path=str(CONFIG))
        row['topic_init_ms']=(time.perf_counter()-start)*1000
        drone=Drone(client,world,'Drone1')
        stage='front';start=time.perf_counter();message=drone.get_images('FrontCamera',[0])[0]
        frame=unpack_image(message);row['front_ms']=(time.perf_counter()-start)*1000
        row['front_shape']=list(frame.shape);row['front_timestamp']=message['time_stamp']
        stage='state';start=time.perf_counter();row['state']=drone.get_ground_truth_kinematics()
        row['state_ms']=(time.perf_counter()-start)*1000
        row['success']=True
    except Exception:
        row['exception']=traceback.format_exc();row['failed_stage']=stage
        row['failed_stage_ms']=(time.perf_counter()-start)*1000
        row['socket_state']=subprocess.check_output(['ss','-anp'],text=True)
    finally:
        if client is not None:
            try:client.disconnect()
            except Exception:row['cleanup_exception']=traceback.format_exc()
            row['subscriber_thread_alive_after_disconnect']=bool(client.recv_topic_thread and client.recv_topic_thread.is_alive())
            del client
        gc.collect()
    with (OUT/'gate-a-trials.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(f'{phase} {iteration} success={row["success"]} stage={row.get("failed_stage","complete")}',flush=True)
    return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True)
    parser.add_argument('--topics',type=int,default=8989);parser.add_argument('--services',type=int,default=8990)
    args=parser.parse_args();prepare_config()
    result={'route':'native WSL NNG client -> Windows host address, no reverse relay',
            'host':args.host,'short_idle':[],'long_idle':[]}
    path=OUT/'gate-a.json'
    for i in range(1,21):
        result['short_idle'].append(trial(args.host,[args.topics,args.services],i,'A1'))
        path.write_text(json.dumps(result,indent=2))
        if i<20:time.sleep(5)
    for i in range(1,6):
        initial=trial(args.host,[args.topics,args.services],i,'A2_before_idle')
        start=time.monotonic();print(f'A2 {i} IDLE_BEGIN 60s',flush=True);time.sleep(60)
        reconnect=trial(args.host,[args.topics,args.services],i,'A2_reconnect')
        result['long_idle'].append({'initial':initial,'reconnect':reconnect,'idle_seconds':time.monotonic()-start})
        path.write_text(json.dumps(result,indent=2))
    result['short_success']=sum(row['success'] for row in result['short_idle'])
    result['long_success']=sum(row['initial']['success'] and row['reconnect']['success'] for row in result['long_idle'])
    result['status']='PASS' if result['short_success']==20 and result['long_success']==5 else 'FAIL'
    path.write_text(json.dumps(result,indent=2));print('GATE_A '+result['status'],flush=True)

if __name__=='__main__':main()
