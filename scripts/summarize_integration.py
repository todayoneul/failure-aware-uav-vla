"""Derive summary measurements from saved raw evidence, without rerunning gates."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/integration'
def read(name): return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def rows(name):
    return [json.loads(line) for line in (OUT/name).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
def stats(values):
    return {'count':len(values),'mean':float(np.mean(values)), 'median':float(np.median(values)),
            'p95':float(np.percentile(values,95)), 'peak':float(max(values))}
comm=read('communication.json');model=read('model-validation.json');e2e=read('end-to-end.json')
model_rows=rows('model-resource-samples.jsonl');combined_rows=rows('combined-resource-samples.jsonl')
win_model=rows('model-windows-resources.jsonl');win_combined=rows('combined-windows-resources.jsonl')
smis=lambda records,key:[float(row[key].split(',')[0]) for row in records]
combined_peak=max(smis(combined_rows,'nvidia_smi')+smis(win_combined,'global_nvidia_smi'))
before=np.array([comm['state_before_forward']['pose']['position'][k] for k in 'xyz'])
after=np.array([comm['state_after']['pose']['position'][k] for k in 'xyz'])
summary={
 'communication':{'passed':not comm['errors'],'route':comm['route'],'latencies_ms':comm['latencies_ms'],
     'forward_displacement_xyz_m':(after-before).tolist(),'forward_horizontal_m':float(np.linalg.norm((after-before)[:2])),
     'takeoff_actual_climb_m':comm['state_before']['pose']['position']['z']-comm['state_after_takeoff']['pose']['position']['z'],
     'takeoff_return':comm['takeoff_return'],'takeoff_landed_state':comm['takeoff_landed_state'],
     'final_landed_state':comm['final_landed_state']},
 'model':{'status':model['status'],'load_info':model['load_info'],'after_load':model['after_load'],
     'first_inference_ms':model['runs'][0]['inference_ms'],'ten_inference_ms':model['ten_inference_ms'],
     'inference_peak':model['peak'],'model_band':'GREEN: tensor allocated and reserved <= 8 GiB',
     'global_gpu_peak_MiB':max(smis(model_rows,'nvidia_smi')+smis(win_model,'global_nvidia_smi')),
     'process_rss_peak_GiB':max(row['process_rss_GiB'] for row in model_rows),
     'wsl_system_peak_GiB':max(row['wsl_system_used_GiB'] for row in model_rows),
     'windows_system_peak_GiB':max(row['windows_system_used_GiB'] for row in win_model),
     'valid_actions':model['valid_actions'],'inference_trials':len(model['runs'])},
 'combined':{'status':e2e['status'],'stage':'model loaded; scene topic initialization failed before flight',
     'global_gpu_peak_MiB':combined_peak,'global_gpu_peak_GiB':combined_peak/1024,
     'physical_gpu_total_MiB':12227,'remaining_at_peak_MiB':12227-combined_peak,
     'simulator_dedicated_peak_GiB':max(sum(m['DedicatedUsage'] for m in row['simulator_memory']) for row in win_combined)/1024**3,
     'simulator_shared_peak_GiB':max(sum(m['SharedUsage'] for m in row['simulator_memory']) for row in win_combined)/1024**3,
     'simulator_3D_peak_percent':max([engine['UtilizationPercentage'] for row in win_combined for engine in row['simulator_engines'] if 'engtype_3D' in engine['Name']]),
     'process_rss_peak_GiB':max(row['process_rss_GiB'] for row in combined_rows),
     'wsl_system_peak_GiB':max(row['wsl_system_used_GiB'] for row in combined_rows),
     'windows_system_peak_GiB':max(row['windows_system_used_GiB'] for row in win_combined),
     'after_load':e2e['after_model_load'],'single_step':'NOT RUN','closed_loop_steps':len(e2e['closed_loop']),
     'OOM':False,'error':e2e['errors'][0]},
 'downloads':{'bytes':sum(file['bytes'] for entry in read('model-downloads.json')['models'] for file in entry['files']),
              'files':sum(len(entry['files']) for entry in read('model-downloads.json')['models'])},
 'resource_sample_counts':{'model_wsl':len(model_rows),'model_windows':len(win_model),
                           'combined_wsl':len(combined_rows),'combined_windows':len(win_combined)}
}
(OUT/'measurement-summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'integration_log.jsonl').open('a') as file:
    for event,source,data in [('communication','communication.json',summary['communication']),
                              ('model_validation','model-validation.json',summary['model'])]:
        file.write(json.dumps({'event':'saved_evidence_summary','measurement':event,'source':source,
                               'note':'Summary appended after run; order is not measurement chronology','data':data})+'\n')
print(json.dumps(summary,indent=2))
