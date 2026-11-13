"""Model-only load + first inference + ten synchronized generation measurements."""
import json
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
import cv2
import numpy as np
import psutil
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.integration.aerovla_int4_loader import AeroVLAInt4
OUT=ROOT/'outputs/integration'

def memory():
    return {'allocated_GiB':torch.cuda.memory_allocated()/1024**3,
        'reserved_GiB':torch.cuda.memory_reserved()/1024**3,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
        'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,
        'process_rss_GiB':psutil.Process().memory_info().rss/1024**3,
        'wsl_system_used_GiB':psutil.virtual_memory().used/1024**3,
        'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True).strip()}

stop=threading.Event()
def sample():
    with (OUT/'model-resource-samples.jsonl').open('w') as file:
        while not stop.is_set():
            try:
                row={'epoch':time.time(),'process_rss_GiB':psutil.Process().memory_info().rss/1024**3,
                     'wsl_system_used_GiB':psutil.virtual_memory().used/1024**3,
                     'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits'],text=True).strip()}
                file.write(json.dumps(row)+'\n');file.flush()
            except Exception as error: print(f'RESOURCE_SAMPLE_ERROR {error}',flush=True)
            stop.wait(.5)

def main():
    result={'status':'RUNNING','simulator':'stopped for model-only test', 'runs':[]}
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    try:
        torch.cuda.reset_peak_memory_stats()
        result['before_load']=memory()
        model=AeroVLAInt4(OUT/'model-downloads.json')
        result['load_info']=model.load_info
        result['after_load']=memory()
        (OUT/'model-validation.json').write_text(json.dumps(result,indent=2))
        frames=[cv2.imread(str(OUT/f'{camera}_communication.png')) for camera in ('front','down')]
        communication=json.loads((OUT/'communication.json').read_text())
        from src.integration.projectairsim_observation_adapter import adapt_state
        state=adapt_state(communication['state_before'])
        target=[state['position'][0]+2.12,state['position'][1]-2.12,state['position'][2]]
        instruction='The target is 3 meters away and 0 degrees from you. Find the colored blocks ahead. Please control the drone.'
        torch.cuda.reset_peak_memory_stats()
        for index in range(11):
            run=model.infer(*frames,state,target,instruction)
            run['index']=index;run['memory']=memory()
            result['runs'].append(run)
            print(f'INFERENCE {index} {run["inference_ms"]:.3f}ms action={run["parsed_action"]} error={run["parse_error"]}',flush=True)
            (OUT/'model-validation.json').write_text(json.dumps(result,indent=2))
        timing=[r['inference_ms'] for r in result['runs'][1:]]
        result['ten_inference_ms']={'count':10,'mean':float(np.mean(timing)),
                                   'median':float(np.median(timing)),'p95':float(np.percentile(timing,95))}
        result['peak']=memory()
        result['valid_actions']=sum(r['parsed_action'] is not None for r in result['runs'])
        result['status']='PASS' if result['valid_actions']==11 else 'INVALID_ACTION_OUTPUT'
    except Exception:
        result['status']='FAIL';result['error']=traceback.format_exc()
        result['failure_memory']=memory()
        print(result['error'],flush=True)
    finally:
        stop.set();sampler.join(timeout=2)
        (OUT/'model-validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if result['status']!='PASS': raise SystemExit(1)

if __name__=='__main__':main()
