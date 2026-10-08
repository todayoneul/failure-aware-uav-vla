"""Freeze a long-range evaluation (sets L1-L3 of configs/long_range_test_spawns.json); no simulator, no model.

Runs are given as "MODEL:SET=DIRECTORY" and only read. An episode is taken through three stages:

  search    the target was in the front or down view at some decision
  approach  the vehicle came within 20 m of the target
  task      it stopped by itself within the success radius without a collision (the unchanged criterion)

and a failed episode gets exactly one type, decided in this order:

  COLLISION                        the episode ended in a collision
  LONG_SEARCH_FAILURE              the target never came into view
  UNSEEN_OBJECT_GROUNDING_FAILURE  an object never trained on was named and the vehicle ended beside another object
  LONG_APPROACH_FAILURE            the target was in view at some point but the vehicle never came within 20 m
  LONG_STOP_FAILURE                it came within 20 m and did not end stopped inside the radius

How the episode ended (self-stop, timeout, collision) is recorded separately, so a timeout is counted
both as a timeout and under the stage it ran out in.

The output directory receives summary.json, episodes.csv, range_results.csv, seen_unseen.csv,
map_results.csv, failure_types.csv and failures/ (a trace and a picture of every failed episode).
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_visual_search import load
from scripts.freeze_generalization_results import nearest_other,picture,climb_after_sight

APPROACH_M=20.
TYPES=('COLLISION','LONG_SEARCH_FAILURE','UNSEEN_OBJECT_GROUNDING_FAILURE','LONG_APPROACH_FAILURE','LONG_STOP_FAILURE')


def in_view(row):return bool(row.get('front_seen') or row.get('down_seen'))


def measure(episode,rows,radius,geometries):
    """Everything recorded about one flown episode."""
    seen=[in_view(row) for row in rows];first=next((index for index,flag in enumerate(seen) if flag),None)
    distances=[row['distance_m'] for row in rows];other=nearest_other(episode,rows,geometries)
    tail=distances[-21:]
    after=seen[first:] if first is not None else []
    return {'initially_visible':seen[0],'ever_acquired':any(seen),'first_acquisition_step':first,
            'first_acquisition_s':None if first is None else round(rows[first]['epoch']-rows[0]['epoch'],1),
            'initial_distance_m':round(distances[0],1),'minimum_distance_m':round(min(distances),1),'final_distance_m':round(distances[-1],1),
            'entered_20m':min(distances)<=APPROACH_M,'entered_15m':min(distances)<=radius,
            'self_stop':bool(episode['stopped']),'collision':episode['reason']=='collision','timeout':episode['reason']=='max_steps',
            'steps':len(rows),'seconds':round(rows[-1]['epoch']-rows[0]['epoch'],1),
            'search_success':any(seen),'approach_success':min(distances)<=APPROACH_M,'task_success':bool(episode['success']),
            # Diagnostics for a failure: was it still getting closer when it ended, how long it kept the target in view, how it moved.
            'closing_at_end_m':round(tail[0]-tail[-1],1),'in_view_after_acquisition':round(sum(after)/len(after),2) if after else None,
            'forward_m_per_step':round(statistics.mean(row['action'][0] for row in rows),3),
            'climbed_m':round(max(row['height_m'] for row in rows)-rows[0]['height_m'],1),'climb_after_sight_m':round(climb_after_sight(rows),1),
            'nearest_other_m':round(other[0],1),'nearest_other':other[1]}


def failure_type(episode,measured,radius):
    if measured['task_success']:return None
    if measured['collision']:return 'COLLISION'
    if not measured['ever_acquired']:return 'LONG_SEARCH_FAILURE'
    if not episode.get('object_seen_in_training',True) and (episode.get('stopped_at_other') or
                                                           (measured['nearest_other_m']<=radius and not measured['approach_success'])):
        return 'UNSEEN_OBJECT_GROUNDING_FAILURE'
    return 'LONG_STOP_FAILURE' if measured['approach_success'] else 'LONG_APPROACH_FAILURE'


def tally(items):
    """One row of a results table."""
    hidden=[item for item in items if not item['initially_visible']]
    count=lambda key,rows=items:sum(bool(row[key]) for row in rows)
    steps=[item['first_acquisition_step'] for item in hidden if item['first_acquisition_step'] is not None]
    return {'episodes':len(items),'acquired':count('ever_acquired'),'started_hidden':len(hidden),'hidden_acquired':count('ever_acquired',hidden),
            'median_steps_to_acquire':statistics.median(steps) if steps else None,
            'entered_20m':count('entered_20m'),'entered_15m':count('entered_15m'),'self_stop':count('self_stop'),'success':count('task_success'),
            'collision':count('collision'),'timeout':count('timeout')}


def table(title,first,rows):
    print(f'\n{title}\n');print(f'| {first} | Acquisition | Hidden at start: acquired | <=20m | <=15m | Self-stop | Final Success | Collision | Timeout |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for name,row in rows:
        n=row['episodes']
        print(f'| {name} | {row["acquired"]}/{n} | {row["hidden_acquired"]}/{row["started_hidden"]} | {row["entered_20m"]}/{n} | {row["entered_15m"]}/{n} | '
              f'{row["self_stop"]}/{n} | {row["success"]}/{n} | {row["collision"]} | {row["timeout"]} |')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',help='MODEL:SET=DIRECTORY');parser.add_argument('--output',required=True)
    parser.add_argument('--plan',default=str(ROOT/'configs/long_range_test_spawns.json'));parser.add_argument('--no-pictures',action='store_true')
    args=parser.parse_args();output=Path(args.output);(output/'failures').mkdir(parents=True,exist_ok=True)
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8'));ranges={name:episodes[0]['range_m'] for name,episodes in plan['sets'].items()}
    geometries={};items=[];models=[];errors={}
    for item in args.runs:
        label,directory=item.split('=',1);model,level=label.rsplit(':',1);_,episodes,steps=load(directory)
        data=json.loads((Path(directory)/'results.json').read_text());radius=data['config']['success_radius_m']
        errors[label]=[episode['id'] for episode in data['episodes'] if episode.get('error')]
        if model not in models:models.append(model)
        for episode in episodes:
            rows=steps[episode['id']];measured=measure(episode,rows,radius,geometries);kind=failure_type(episode,measured,radius)
            record={'model':model,'range':level,'range_m':f'{ranges[level][0]:.0f}-{ranges[level][1]:.0f}','episode':episode['id'],'map':episode['map'],
                    'layout':episode['layout'],'object':episode['target'],'object_seen_in_training':episode['object_seen_in_training'],'kind':episode['kind'],
                    'instruction':episode['instruction'],'seed':episode['seed'],'planned_distance_m':episode['start_distance_m'],
                    'initial_bearing_deg':episode['target_bearing_deg'],'initial_altitude_m':episode['start_height_m'],**measured,
                    'end':'collision' if measured['collision'] else 'self_stop' if measured['self_stop'] else 'timeout' if measured['timeout'] else episode['reason'],
                    'failure_type':kind or ''}
            items.append(record)
            if kind and model!='Teacher':
                name=f'{model}_{level}_{kind}_{episode["id"]}'.replace(' ','_')
                trace=[{key:row.get(key) for key in ('step','distance_m','bearing_deg','height_m','front_seen','down_seen','action','teacher_state')} for row in rows]
                (output/'failures'/f'{name}.json').write_text(json.dumps({**record,'trace':trace},indent=1))
                if not args.no_pictures:picture(episode,rows,output/'failures'/f'{name}.jpg',f'{model} {level} {kind}: {episode["id"]}')
    summary={'plan':str(Path(args.plan).resolve().relative_to(ROOT)).replace('\\','/'),'definitions':{'approach_m':APPROACH_M,'in_view':'front or down camera',
             'success':'stopped by itself within the success radius, no collision (unchanged)','failure_types':list(TYPES)},
             'episodes_with_errors':errors,'models':{}}
    for model in models:
        mine=[item for item in items if item['model']==model];failed=[item for item in mine if item['failure_type']]
        by_range=[(f'{name} {ranges[name][0]:.0f}-{ranges[name][1]:.0f}m',tally([i for i in mine if i['range']==name])) for name in ranges if any(i['range']==name for i in mine)]
        by_range.append(('All',tally(mine)))
        groups=(('seen object',True),('unseen object',False))
        by_object=[(f'{name} {label}',tally([i for i in mine if i['range']==name and i['object_seen_in_training']==flag])) for name in ranges for label,flag in groups]
        by_object+=[(f'All {label}',tally([i for i in mine if i['object_seen_in_training']==flag])) for label,flag in groups]
        by_map=[(f'{name} {scene}',tally([i for i in mine if i['range']==name and i['map']==scene])) for name in ranges for scene in ('blocks','yard')]
        by_map+=[(f'All {scene}',tally([i for i in mine if i['map']==scene])) for scene in ('blocks','yard')]
        by_start=[(label,tally([i for i in mine if i['kind']==kind])) for label,kind in (('in view at the start','visible'),('out of view at the start','search'))]
        types=[{'model':model,'failure_type':kind,'count':len(found),'timeouts':sum(i['timeout'] for i in found),
                'unseen_object':sum(not i['object_seen_in_training'] for i in found),'by_range':{name:sum(i['range']==name for i in found) for name in ranges},
                'episodes':[i['episode'] for i in found]} for kind in TYPES for found in [[i for i in failed if i['failure_type']==kind]]]
        summary['models'][model]={'by_range':dict(by_range),'by_object':dict(by_object),'by_map':dict(by_map),'by_start':dict(by_start),'failure_types':types,
                                  'failures':len(failed),'timeouts':sum(i['timeout'] for i in mine)}
        print(f'\n# {model}');table('## Distance','Range',by_range);table('## Seen and unseen objects','Range and object',by_object)
        table('## Scene','Range and scene',by_map);table('## Start','Start',by_start)
        print('\n## Failure types\n');print('| Type | Count | of which timeout | Unseen object | L1 | L2 | L3 | Episodes |');print('|---|---:|---:|---:|---:|---:|---:|---|')
        for row in types:
            print(f'| {row["failure_type"]} | {row["count"]} | {row["timeouts"]} | {row["unseen_object"]} | '+' | '.join(str(row['by_range'].get(name,0)) for name in ('L1','L2','L3'))
                  +f' | {", ".join(row["episodes"])} |')
        if model!='Teacher':
            flat=lambda rows,first:[{first:name,**row} for name,row in rows]
            for name,rows in (('range_results.csv',flat(by_range,'range')),('seen_unseen.csv',flat(by_object,'range_and_object')),('map_results.csv',flat(by_map,'range_and_scene')),
                              ('failure_types.csv',[{**row,'by_range':json.dumps(row['by_range']),'episodes':' '.join(row['episodes'])} for row in types])):
                with (output/name).open('w',newline='',encoding='utf-8') as file:
                    writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/'episodes.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    (output/'summary.json').write_text(json.dumps(summary,indent=1))


if __name__=='__main__':main()
