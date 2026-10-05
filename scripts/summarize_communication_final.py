"""Summarize saved final validation evidence; no model/simulator execution."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/communication_final'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def rows(name):return [json.loads(s) for s in (OUT/name).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def stats(values):
    return {'count':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),
            'p95':float(np.percentile(values,95)),'max':float(max(values))}
a=read('gate-a.json');b=read('gate-b.json');loop=read('closed-loop.json')
win=rows('windows-resources.jsonl');wsl=rows('closed-loop-wsl-resources.jsonl')
windows_loop=[r for r in win if r['stage']=='Gate E ten-step loop']
windows_pipeline=[r for r in win if r['stage'].startswith(('Gate D','Gate E'))]
flight_start=loop['takeoff']['after']['time_stamp']
flight_end=loop['closed_loop'][-1]['state_after']['time_stamp']
flight_collisions=[e for e in loop['collision_events'] if flight_start<e['message']['time_stamp']<=flight_end]
all_steps=[loop['single_step']]+loop['closed_loop']
resource=lambda records,key:[float(r[key].split(',')[0]) for r in records]
summary={
 'gate_a':{'status':a['status'],'route':a['route'],'host':a['host'],
     'short_success':a['short_success'],'long_success':a['long_success'],
     'short_latencies_ms':{key:stats([r[key] for r in a['short_idle']]) for key in ('connect_ms','topic_init_ms','front_ms','state_ms')},
     'idle_seconds': [row['idle_seconds'] for row in a['long_idle']],
     'root_cause':'Native direct path passed; exact prior reverse-relay failure mechanism remains unproven'},
 'gate_b':{'status':b['status'],'ram_stages':[{'target_GiB':r['target_GiB'],'status':r['status'],
         'held_GiB':r['held_GiB'],'trials_success':sum(t['success'] for t in r.get('trials',[])),
         'windows':r.get('during',{}).get('windows')} for r in b['ram']],
     'gpu_allocated_GiB':b['gpu']['allocated_GiB'],'gpu_reserved_GiB':b['gpu']['reserved_GiB'],
     'gpu_success':sum(r['success'] for r in b['gpu']['trials']),
     'communication_failure_correlated':'NO in tested2/4GiB RAM and7GiB GPU conditions;8/12/16GiB RAM untested'},
 'gate_c':read('gate-c.json'),
 'gate_d':{'status':'PASS','movement_m':loop['single_step']['movement_m'],
     'inference_ms':loop['single_step']['inference']['inference_ms'],
     'observation_transport_ms':loop['single_step']['observation_transport_ms'],
     'command_wall_ms':loop['single_step']['command_rpc_wall_ms'],
     'action':loop['single_step']['inference']['parsed_action'],
     'clipped_action':loop['single_step']['clipped_action']},
 'gate_e':{'status':loop['status'],**loop['loop_summary'],
     'effective_decisions_Hz':1000/loop['loop_summary']['step_wall_mean_ms'],
     'command_wall_ms':stats([r['command_rpc_wall_ms'] for r in loop['closed_loop']]),
     'rpc_exception_count':len(loop['errors']),
     'all_command_returns_true':all(all(r['command_returns'].values()) for r in all_steps),
     'flight_collision_count':len(flight_collisions),'ground_contact_messages':len(loop['collision_events']),
     'flight_start_timestamp':flight_start,'flight_end_timestamp':flight_end,
     'movement_each_m':[r['movement_m'] for r in loop['closed_loop']],
     'final_landed_state':loop['final_landed_state'],'cleanup_error':loop.get('cleanup_error')},
 'resources':{
     'physical_gpu_total_MiB':12227,
     'pipeline_gpu_peak_MiB':max(resource(windows_pipeline,'global_nvidia_smi')),
     'loop_gpu_peak_MiB':max(resource(windows_loop,'global_nvidia_smi')),
     'wsl_sampler_global_peak_MiB':max(resource(wsl,'nvidia_smi')),
     'model_peak_allocated_GiB':loop['peak']['peak_allocated_GiB'],
     'model_peak_reserved_GiB':loop['peak']['peak_reserved_GiB'],
     'simulator_dedicated_peak_GiB':max(sum(m['DedicatedUsage'] for m in r['simulator_gpu_memory']) for r in windows_pipeline)/1024**3,
     'simulator_shared_peak_GiB':max(sum(m['SharedUsage'] for m in r['simulator_gpu_memory']) for r in windows_pipeline)/1024**3,
     'windows_used_peak_GiB':max(r['windows_used_GiB'] for r in windows_pipeline),
     'windows_available_min_GiB':min(r['windows_available_GiB'] for r in windows_pipeline),
     'windows_commit_peak_GiB':max(r['commit_GiB'] for r in windows_pipeline),
     'pagefile_used_peak_MiB':max(sum(p['CurrentUsage'] for p in r['pagefiles']) for r in windows_pipeline),
     'pagefile_capacity_peak_MiB':max(sum(p['AllocatedBaseSize'] for p in r['pagefiles']) for r in windows_pipeline),
     'wsl_process_rss_peak_GiB':max(r['process_rss_GiB'] for r in wsl),
     'wsl_psutil_used_peak_GiB':max(r['wsl_memory']['used'] for r in wsl)/1024**3,
     'wsl_available_min_GiB':min(r['wsl_memory']['available'] for r in wsl)/1024**3,
     'wsl_swap_used_peak_GiB':max(r['wsl_swap']['used'] for r in wsl)/1024**3,
     'sample_counts':{'windows_pipeline':len(windows_pipeline),'windows_loop':len(windows_loop),'wsl_model_to_cleanup':len(wsl)}},
 'domain_check':loop['domain_check'],
 'final_decision':'FEASIBLE WITH LIMITATIONS:10/10 functional integration, RAM margin and decision cadence remain limited'}
summary['resources']['combined_gpu_peak_MiB']=max(summary['resources']['pipeline_gpu_peak_MiB'],summary['resources']['wsl_sampler_global_peak_MiB'])
summary['resources']['combined_gpu_peak_GiB']=summary['resources']['combined_gpu_peak_MiB']/1024
summary['resources']['remaining_at_peak_GiB']=(12227-summary['resources']['combined_gpu_peak_MiB'])/1024
(OUT/'final-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
