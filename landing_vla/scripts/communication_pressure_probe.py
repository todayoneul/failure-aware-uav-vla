"""Bounded synthetic RAM/GPU pressure; never load AeroVLA in Gate B."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from communication_reconnect_probe import trial
OUT=ROOT/'outputs/communication_final'

def host_memory():
    return json.loads((OUT/'windows-latest.json').read_text(encoding='utf-8-sig'))

def resources():
    return {'windows':host_memory(),'wsl':dict(psutil.virtual_memory()._asdict()),
            'process_rss_GiB':psutil.Process().memory_info().rss/1024**3}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True);args=parser.parse_args()
    gate_a=json.loads((OUT/'gate-a.json').read_text())
    if gate_a.get('status')!='PASS':raise RuntimeError('Gate A must pass before resource pressure')
    result={'ram':[],'gpu':None,'errors':[]};chunks=[]
    def save(): (OUT/'gate-b.json').write_text(json.dumps(result,indent=2))
    (OUT/'active-stage.txt').write_text('Gate B RAM pressure')
    try:
        for target in (2,4,8,12,16):
            row={'target_GiB':target,'before':resources()}
            current=len(chunks)*.5;increment=target-current
            win=row['before']['windows'];wsl_available=row['before']['wsl']['available']/1024**3
            commit_headroom=win['commit_limit_GiB']-win['commit_GiB']
            if (wsl_available<increment+2 or win['windows_available_GiB']<increment+3
                    or commit_headroom<increment+.5):
                row.update(status='SKIPPED_HEADROOM',held_GiB=current,
                    reason='Need >=2GiB WSL available, >=3GiB Windows physical available and >=0.5GiB projected commit margin')
                result['ram'].append(row);save();print(f'RAM {target}GiB SKIPPED_HEADROOM',flush=True);continue
            for _ in range(int(increment*2)):
                block=np.empty(512*1024**2,dtype=np.uint8);block.fill(1);chunks.append(block)
            time.sleep(3)
            row['during']=resources();row['held_GiB']=len(chunks)*.5
            row['trials']=[trial(args.host,[8989,8990],i,f'B1_{target}GiB') for i in (1,2)]
            row['status']='PASS' if all(t['success'] for t in row['trials']) else 'FAIL'
            result['ram'].append(row);save();print(f'RAM {target}GiB {row["status"]}',flush=True)
            if row['status']=='FAIL':break
    finally:
        chunks.clear()
        try:del block
        except UnboundLocalError:pass
        gc.collect()
    time.sleep(3)
    (OUT/'active-stage.txt').write_text('Gate B GPU pressure')
    import torch
    free,total=torch.cuda.mem_get_info()
    result['gpu']={'target_GiB':7,'free_before_GiB':free/1024**3,'total_GiB':total/1024**3}
    if free<7.75*1024**3:
        result['gpu']['status']='SKIPPED_HEADROOM';save();return
    reservation=None
    try:
        reservation=torch.empty(7*1024**3,dtype=torch.uint8,device='cuda');reservation.fill_(0)
        torch.cuda.synchronize()
        result['gpu']['allocated_GiB']=torch.cuda.memory_allocated()/1024**3
        result['gpu']['reserved_GiB']=torch.cuda.memory_reserved()/1024**3
        result['gpu']['during']=resources()
        result['gpu']['trials']=[trial(args.host,[8989,8990],i,'B2_7GiB') for i in range(1,6)]
        result['gpu']['status']='PASS' if all(t['success'] for t in result['gpu']['trials']) else 'FAIL'
    finally:
        del reservation;torch.cuda.empty_cache();save()
    result['status']='PASS' if all(r['status']!='FAIL' for r in result['ram']) and result['gpu']['status']=='PASS' else 'FAIL_OR_INCOMPLETE'
    save();print('GATE_B '+result['status'],flush=True)

if __name__=='__main__':main()
