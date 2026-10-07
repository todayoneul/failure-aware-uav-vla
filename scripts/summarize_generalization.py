"""Generalisation tables from visual-search runs; no simulator, no model.

Each run is given as MODEL:LEVEL=DIRECTORY, for example
  "Gen-v1:G1=outputs/visual_search/gen_v1_g1"  "Pilot:G1=outputs/visual_search/pilot_g1"
Prints the level-by-model matrix, a breakdown of each run, how the search turns, why failures failed,
and the motion of each run. --json keeps the same numbers.
"""
import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_visual_search import load,motion
from src.visual_search.maps import load_map,MapGeometry

LEVELS=('G0','G1','G2','G3','G4','P','S')
NAMES={'G0':'G0 seen map, seen object, seen start','G1':'G1 seen map, seen object, unseen start','G2':'G2 seen map, unseen object',
       'G3':'G3 unseen map, seen object','G4':'G4 unseen map, unseen object','P':'G1 starts, unseen wording','S':'G1 side probe'}


def mean(values):
    values=[value for value in values if value is not None]
    return statistics.mean(values) if values else None


def failure(episode,radius):
    """One reason per episode, read from what happened rather than from why it might have."""
    if episode['success']:return 'success'
    if episode['reason']=='collision':return 'collision'
    if episode['reason']=='invalid_action':return 'invalid model output'
    if episode['stopped']:
        return 'stopped at another object' if episode.get('stopped_at_other') else 'stopped away from the target'
    if episode['acquired_step'] is None:return 'target never came into view'
    if episode['minimum_distance_m']<=radius:return 'reached the target but did not stop'
    if episode['initial_distance_m']-episode['minimum_distance_m']<5.:return 'target in view but no approach'
    return 'approached but did not arrive'


def breakdown(episodes,radius):
    def cell(items):return f'{sum(e["success"] for e in items)}/{len(items)}'
    unseen=[e for e in episodes if not e['initially_seen']]
    return {'n':len(episodes),'success':sum(e['success'] for e in episodes),'collisions':sum(e['reason']=='collision' for e in episodes),
            'reached':sum(e['reached'] for e in episodes),'stopped':sum(e['stopped'] for e in episodes),
            'mean_initial_m':mean(e['initial_distance_m'] for e in episodes),'mean_final_m':mean(e['final_distance_m'] for e in episodes),
            'mean_minimum_m':mean(e['minimum_distance_m'] for e in episodes),
            'mean_stop_distance_m':mean(e.get('stop_distance_m') for e in episodes if e['success']),
            'started_unseen':len(unseen),'acquired':sum(e['acquired_step'] is not None for e in unseen),
            'mean_acquire_step':mean(e['acquired_step'] for e in unseen),'mean_search_steps':mean(e.get('search_steps') for e in episodes),
            'mean_yaw_swept_deg':mean(e['yaw_swept_deg'] for e in episodes),'mean_climbed_m':mean(e.get('climbed_m') for e in episodes),
            'mean_seconds':mean(e.get('seconds') for e in episodes),
            'by_kind':{kind:cell([e for e in episodes if e.get('kind',e['case'])==kind]) for kind in sorted({e.get('kind',e['case']) for e in episodes})},
            'by_target':{target:cell([e for e in episodes if e['target']==target]) for target in sorted({e['target'] for e in episodes})},
            'by_wedge':{name:cell([e for e in episodes if e.get('in_wedge') is flag]) for name,flag in (('inside the held-out wedge',True),('outside it',False))
                        if any(e.get('in_wedge') is flag for e in episodes)},
            'by_instruction':{name:cell([e for e in episodes if e.get('instruction_id')==name]) for name in sorted({e.get('instruction_id') for e in episodes if e.get('instruction_id')})},
            'wrong_object_stops':sum(bool(e.get('stopped_at_other')) for e in episodes),
            'failures':dict(collections.Counter(failure(e,radius) for e in episodes if not e['success']))}


def distractors(episodes,steps,reach_m=70.,half_fov_deg=40.):
    """How often another object of the scene stood in the front view, and whether the vehicle went to it.

    Recomputed from the flown poses and the map's boxes: inside the view angle, with a free line of sight, and no farther than
    the longest start distance of the training data."""
    exposed=before=stops=0;geometries={}
    for episode in episodes:
        key=(episode.get('map','blocks'),episode.get('layout','pilot'))
        if key not in geometries:geometries[key]=MapGeometry(load_map(key[0]),key[1])
        geometry=geometries[key];rows=steps.get(episode['id'],[]);seen_other=seen_before=False
        first=episode['acquired_step'] if episode['acquired_step'] is not None else len(rows)
        for index,row in enumerate(rows):
            x,y=row['position'][:2]
            for name,item in geometry.objects.items():
                if name==episode['target']:continue
                dx,dy=item['centre'][0]-x,item['centre'][1]-y
                bearing=math.degrees(math.atan2(math.sin(math.atan2(dy,dx)-row['yaw_rad']),math.cos(math.atan2(dy,dx)-row['yaw_rad'])))
                if abs(bearing)<=half_fov_deg and math.hypot(dx,dy)<=reach_m and geometry.sees(x,y,row['height_m'],name):
                    seen_other=True;seen_before=seen_before or index<first
        exposed+=seen_other;before+=seen_before;stops+=bool(episode.get('stopped_at_other'))
    return {'episodes':len(episodes),'another_object_in_view':exposed,'before_the_named_one_was_seen':before,'stopped_at_another_object':stops}


def turns(episodes):
    """First turn of episodes that began with the target out of view, by the side it was on."""
    table={}
    for episode in episodes:
        if episode['initially_seen'] or episode.get('kind') not in ('search','altitude'):continue
        side=episode.get('side') or ('behind' if abs(episode['target_bearing_deg'])>=150 else 'left' if episode['target_bearing_deg']<0 else 'right')
        row=table.setdefault(side,{'right':0,'left':0,'none':0,'acquired':0,'n':0})
        row[episode.get('first_turn') or 'none']+=1;row['acquired']+=episode['acquired_step'] is not None;row['n']+=1
    return table


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',help='MODEL:LEVEL=DIRECTORY');parser.add_argument('--json')
    args=parser.parse_args();runs=[];output={'runs':{}}
    for item in args.runs:
        label,directory=item.split('=',1);model,level=label.rsplit(':',1)
        policy,episodes,steps=load(directory);data=json.loads((Path(directory)/'results.json').read_text())
        radius=data['config']['success_radius_m']
        runs.append({'model':model,'level':level,'directory':directory,'policy':policy,'episodes':episodes,'steps':steps,
                     'summary':breakdown(episodes,radius),'turns':turns(episodes),'distractors':distractors(episodes,steps),
                     'motion':motion(policy,episodes,steps)})
        output['runs'][label]={'directory':directory,'summary':runs[-1]['summary'],'turns':runs[-1]['turns'],'distractors':runs[-1]['distractors'],'motion':runs[-1]['motion'],
                               'episodes':[{k:e.get(k) for k in ('id','target','kind','success','reason','initial_distance_m','final_distance_m','minimum_distance_m',
                                                                 'acquired_step','first_turn','climbed_m','stopped_at_other','in_wedge','instruction')}
                                           |{'outcome':failure(e,radius)} for e in episodes]}
    models=list(dict.fromkeys(run['model'] for run in runs))
    print('## Generalisation matrix\n');print('| Level | '+' | '.join(models)+' |');print('|---|'+'---:|'*len(models))
    for level in LEVELS:
        row=[]
        for model in models:
            found=[run for run in runs if run['model']==model and run['level']==level]
            row.append(' + '.join(f'{run["summary"]["success"]}/{run["summary"]["n"]}' for run in found) if found else '-')
        if any(cell!='-' for cell in row):print(f'| {NAMES[level]} | '+' | '.join(row)+' |')
    for run in runs:
        s=run['summary'];fmt=lambda value,unit='':'-' if value is None else f'{value:.1f}{unit}'
        print(f'\n## {run["model"]}, {NAMES[run["level"]]}\n')
        print(f'- success {s["success"]}/{s["n"]}, collisions {s["collisions"]}, came within the radius {s["reached"]}/{s["n"]}, stopped by itself {s["stopped"]}/{s["n"]}')
        print(f'- distance: start {fmt(s["mean_initial_m"]," m")}, closest {fmt(s["mean_minimum_m"]," m")}, end {fmt(s["mean_final_m"]," m")}; stop distance when successful {fmt(s["mean_stop_distance_m"]," m")}')
        print(f'- started with the target out of view: {s["started_unseen"]}, brought it into view: {s["acquired"]}/{s["started_unseen"]}, after {fmt(s["mean_acquire_step"])} decisions on average')
        print(f'- per episode: {fmt(s["mean_search_steps"])} decisions without the target in view, {fmt(s["mean_yaw_swept_deg"]," deg")} turned, {fmt(s["mean_climbed_m"]," m")} climbed, {fmt(s["mean_seconds"]," s")}')
        for title,key in (('by kind','by_kind'),('by object','by_target'),('by start region','by_wedge'),('by wording','by_instruction')):
            if s[key]:print(f'- {title}: '+', '.join(f'{name} {value}' for name,value in s[key].items()))
        if s['failures']:print('- failures: '+', '.join(f'{name} {value}' for name,value in s['failures'].items()))
        d=run['distractors']
        print(f'- another object of the scene was in the front view (within 70 m) in {d["another_object_in_view"]}/{d["episodes"]} episodes '
              f'({d["before_the_named_one_was_seen"]} of them before the named one was first seen); stopped at another object: {d["stopped_at_another_object"]}')
        if run['turns']:
            print('- first turn when the target started out of view: '+'; '.join(
                f'target {side}: right {row["right"]}, left {row["left"]}, none {row["none"]} (found {row["acquired"]}/{row["n"]})' for side,row in sorted(run['turns'].items())))
    print('\n## Motion\n');print('| Run | Decisions | Seconds per decision | Inference (ms) | Action change (fraction of range) | Heading change per decision (deg) | Largest (deg) | Outside range | Collisions |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for run in runs:
        m=run['motion'];s=run['summary']
        print(f'| {run["model"]} {run["level"]} | {m["decisions"]} | {m["seconds_per_decision"]:.2f} | {m["inference_ms"]:.0f} | {m["action_change_fraction_of_range"]:.3f} | '
              f'{m["vehicle_turn_per_decision_deg"]:.1f} | {m["largest_turn_per_decision_deg"]:.0f} | {m["predictions_outside_range"]}/{m["decisions"]} | {s["collisions"]}/{s["n"]} |')
    if args.json:Path(args.json).write_text(json.dumps(output,indent=1))


if __name__=='__main__':main()
