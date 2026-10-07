"""Freeze the results of a set of visual-search runs as small files; no simulator, no model.

Runs are given as "MODEL:LEVEL=DIRECTORY". The raw run directories are only read. The output
directory receives:

  summary.json               every number below, plus the checkpoint's training record if one is given
  generalization_levels.csv  model x level: success and the stages before it
  failure_taxonomy.csv       every failure put in exactly one category, with a representative episode
  motion_comparison.csv      decision cycle, latency, action and heading change, clipping, collisions
  episodes.csv               one row per episode
  representative_failures/   for each category of each model: the episode's trace and a picture of it

Stages of an episode (the success criterion itself is unchanged: stopped by itself, within the
radius, no collision):
  acquisition  the target started out of view and came into the front or down view
  approach     the vehicle came within 20 m of the target
  stop         the policy stopped by itself
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_visual_search import load,motion
from scripts.summarize_generalization import breakdown,turns,distractors,LEVELS
from src.visual_search.maps import load_map,MapGeometry
from src.aerovla_oft.spec import normalize_action

APPROACH_M=20.
CLIMB_AFTER_SIGHT_M=5.
CATEGORIES={'F1':'search: the target never came into view','F2':'grounding: went to or stopped at another object',
            'F3':'approach: found the target but did not get close','F4':'early stop: stopped by itself outside the radius',
            'F5':'no stop: was near the target and did not stop there','F6':'altitude transition: kept climbing after the target was found',
            'F7':'collision','F8':'language: fails only with the unseen wording','F9':'domain: repeated in the held-out scene only','F10':'other'}


def stages(episode,rows):
    seen=[bool(row.get('front_seen') or row.get('down_seen')) for row in rows]
    started_hidden=not seen[0] if seen else not episode['initially_seen']
    return {'started_hidden':started_hidden,'acquired':any(seen) if started_hidden else None,
            'approached':episode['minimum_distance_m']<=APPROACH_M,'stopped':bool(episode['stopped']),'success':bool(episode['success'])}


def climb_after_sight(rows):
    """How much higher the vehicle went after the target was first in view."""
    first=next((index for index,row in enumerate(rows) if row.get('front_seen') or row.get('down_seen')),None)
    if first is None:return 0.
    return max(row['height_m'] for row in rows[first:])-rows[first]['height_m']


def nearest_other(episode,rows,geometries):
    """Closest the vehicle came to any other object of the scene: (distance, name)."""
    key=(episode.get('map','blocks'),episode.get('layout','pilot'))
    if key not in geometries:geometries[key]=MapGeometry(load_map(key[0]),key[1])
    best=(float('inf'),None)
    for name,item in geometries[key].objects.items():
        if name==episode['target']:continue
        for row in rows:
            distance=math.hypot(item['centre'][0]-row['position'][0],item['centre'][1]-row['position'][1])
            if distance<best[0]:best=(distance,name)
    return best


def classify(episode,rows,radius,geometries,base_success=None):
    """One category for a failed episode, decided by what the log shows in this order."""
    if episode['success']:return None
    if episode['reason']=='collision':return 'F7'
    if episode['reason']=='invalid_action':return 'F10'
    if not any(row.get('front_seen') or row.get('down_seen') for row in rows):return 'F1'
    if base_success:return 'F8'
    other=nearest_other(episode,rows,geometries)[0]
    if episode.get('stopped_at_other') or (other<=radius and episode['minimum_distance_m']>APPROACH_M):return 'F2'
    if climb_after_sight(rows)>=CLIMB_AFTER_SIGHT_M:return 'F6'
    if episode['stopped']:
        if episode['minimum_distance_m']<=radius:return 'F5'
        return 'F4' if episode['minimum_distance_m']<=APPROACH_M else 'F3'
    return 'F5' if episode['minimum_distance_m']<=APPROACH_M else 'F3'


def clipping(policy,rows_by_episode,config):
    """How often and by how much the head leaves the normalised range (flown values are clipped)."""
    values=[row['normalised'] for rows in rows_by_episode.values() for row in rows if 'normalised' in row and not row.get('perturbed')]
    if not values:return None
    excess=[max(abs(v)-1 for v in value) for value in values];outside=[e for e in excess if e>0]
    return {'decisions':len(values),'outside':len(outside),'frequency':len(outside)/len(values),
            'median_overshoot':statistics.median(outside) if outside else 0.,'max_overshoot':max(outside) if outside else 0.}


def picture(episode,rows,output,title):
    """A failure at a glance: the flown path on the map, and distance and height over the decisions."""
    import cv2
    import numpy as np
    from scripts.plan_generalization import figure
    canvas=figure(load_map(episode.get('map','blocks')),episode.get('layout','pilot'),[],None,title=title,
                  paths=[([row['position'] for row in rows],episode['target'],episode['success'])],
                  legend='white dot: start   square: end   colour: the object named')
    height=canvas.shape[0];chart=np.full((height,520,3),255,dtype=np.uint8);left,right,top,bottom=60,500,40,height-60
    def draw(values,color,label,scale,row):
        points=[(int(left+(right-left)*index/max(1,len(values)-1)),int(bottom-(bottom-top)*min(1.,value/scale))) for index,value in enumerate(values)]
        for a,b in zip(points,points[1:]):cv2.line(chart,a,b,color,2,cv2.LINE_AA)
        cv2.putText(chart,f'{label} (full scale {scale:.0f} m)',(left,20+18*row),0,.45,color,1,cv2.LINE_AA)
    cv2.rectangle(chart,(left,top),(right,bottom),(120,120,120),1)
    draw([row['distance_m'] for row in rows],(40,40,200),'distance to the target',80.,0)
    draw([row['height_m'] for row in rows],(180,110,20),'height above ground',40.,1)
    for index,row in enumerate(rows):
        if row.get('front_seen'):
            x=int(left+(right-left)*index/max(1,len(rows)-1));cv2.line(chart,(x,bottom+6),(x,bottom+14),(40,150,40),1)
    cv2.putText(chart,'green ticks: target in the front view',(left,bottom+34),0,.42,(40,150,40),1,cv2.LINE_AA)
    cv2.putText(chart,f'{len(rows)} decisions',(right-110,bottom+34),0,.42,(60,60,60),1,cv2.LINE_AA)
    cv2.imwrite(str(output),np.hstack([canvas,chart]),[cv2.IMWRITE_JPEG_QUALITY,85])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',help='MODEL:LEVEL=DIRECTORY');parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint',action='append',default=[],help='MODEL=DIRECTORY of a checkpoint whose training record is copied into the summary')
    parser.add_argument('--no-pictures',action='store_true')
    args=parser.parse_args();output=Path(args.output);(output/'representative_failures').mkdir(parents=True,exist_ok=True)
    runs=[];geometries={};rows_out=[]
    for item in args.runs:
        label,directory=item.split('=',1);model,level=label.rsplit(':',1)
        policy,episodes,steps=load(directory);data=json.loads((Path(directory)/'results.json').read_text())
        runs.append({'model':model,'level':level,'directory':directory,'policy':policy,'episodes':episodes,'steps':steps,'config':data['config'],'oft':data['oft']})
    # A prompt-variation failure counts as a language failure only if the same start succeeded with the trained wording.
    base={(run['model'],episode['id']):episode['success'] for run in runs if run['level']=='G1' for episode in run['episodes']}
    for run in runs:
        radius=run['config']['success_radius_m']
        for episode in run['episodes']:
            rows=run['steps'].get(episode['id'],[]);stage=stages(episode,rows)
            code=classify(episode,rows,radius,geometries,base.get((run['model'],episode.get('base'))) if run['level']=='P' else None)
            episode['_stage']=stage;episode['_failure']=code;episode['_climb_after_sight']=climb_after_sight(rows)
    # Domain failure: a category that repeats in the held-out scene and never occurs in the training map for the same model.
    for model in {run['model'] for run in runs}:
        mine=[episode for run in runs if run['model']==model for episode in run['episodes'] if episode['_failure']]
        for code in set(e['_failure'] for e in mine):
            yard=[e for e in mine if e['_failure']==code and e.get('map')!='blocks'];blocks=[e for e in mine if e['_failure']==code and e.get('map','blocks')=='blocks']
            if len(yard)>=2 and not blocks and code not in ('F7','F8'):
                for episode in yard:episode['_mechanism']=code;episode['_failure']='F9'
    summary={'runs':{},'categories':CATEGORIES,'definitions':{'approach_m':APPROACH_M,'climb_after_sight_m':CLIMB_AFTER_SIGHT_M,
                                                              'success':'stopped by itself within the radius, no collision (unchanged)'}}
    levels=[];taxonomy=[];motions=[]
    for run in runs:
        episodes=run['episodes'];radius=run['config']['success_radius_m'];label=f'{run["model"]}:{run["level"]}'
        hidden=[e for e in episodes if e['_stage']['started_hidden']];failed=[e for e in episodes if e['_failure']]
        stage={'episodes':len(episodes),'success':sum(e['success'] for e in episodes),'collisions':sum(e['reason']=='collision' for e in episodes),
               'started_hidden':len(hidden),'acquired':sum(bool(e['_stage']['acquired']) for e in hidden),
               'approached':sum(e['_stage']['approached'] for e in episodes),'stopped':sum(e['_stage']['stopped'] for e in episodes)}
        levels.append({'model':run['model'],'level':run['level'],**stage,'success_rate':round(stage['success']/max(1,len(episodes)),3)})
        counts={}
        for code in CATEGORIES:
            items=[e for e in failed if e['_failure']==code]
            if not items:continue
            representative=sorted(items,key=lambda e:e['id'])[0];counts[code]={'count':len(items),'episodes':[e['id'] for e in items],'representative':representative['id']}
            taxonomy.append({'model':run['model'],'level':run['level'],'category':code,'name':CATEGORIES[code],'count':len(items),
                             'percent_of_failures':round(100*len(items)/len(failed),1),'percent_of_episodes':round(100*len(items)/len(episodes),1),
                             'representative_episode':representative['id']})
            rows=run['steps'].get(representative['id'],[]);name=f'{run["model"]}_{run["level"]}_{code}_{representative["id"]}'.replace(' ','_')
            trace=[{k:row.get(k) for k in ('step','distance_m','bearing_deg','height_m','front_seen','down_seen','action','teacher_state')} for row in rows]
            (output/'representative_failures'/f'{name}.json').write_text(json.dumps({'model':run['model'],'level':run['level'],'category':code,'name':CATEGORIES[code],
                'episode':{k:v for k,v in representative.items() if not k.startswith('_') and k!='plan'},'climb_after_sight_m':representative['_climb_after_sight'],'trace':trace},indent=1))
            if not args.no_pictures and rows:
                picture(representative,rows,output/'representative_failures'/f'{name}.jpg',f'{run["model"]} {run["level"]} {code}: {representative["id"]}')
        move=motion(run['policy'],episodes,run['steps']);clip=clipping(run['policy'],run['steps'],run['oft'])
        motions.append({'model':run['model'],'level':run['level'],'episodes':len(episodes),'success':stage['success'],'collisions':stage['collisions'],
                        'decisions':move['decisions'],'seconds_per_decision':round(move['seconds_per_decision'],3),
                        'inference_ms':None if move['inference_ms']!=move['inference_ms'] else round(move['inference_ms'],1),
                        'action_change_fraction_of_range':round(move['action_change_fraction_of_range'],4),
                        'heading_change_deg_mean':round(move['vehicle_turn_per_decision_deg'],2),'heading_change_deg_max':round(move['largest_turn_per_decision_deg'],1),
                        'mean_speed_mps':round(move['mean_speed_mps'],3),
                        'clip_frequency':None if clip is None else round(clip['frequency'],3),'clip_median_overshoot':None if clip is None else round(clip['median_overshoot'],4),
                        'clip_max_overshoot':None if clip is None else round(clip['max_overshoot'],3)})
        summary['runs'][label]={'directory':run['directory'],'stages':stage,'breakdown':breakdown(episodes,radius),'failures':counts,'turns':turns(episodes),
                                'distractors':distractors(episodes,run['steps']),'motion':motions[-1]}
        for episode in episodes:
            rows_out.append({'model':run['model'],'level':run['level'],'episode':episode['id'],'map':episode.get('map','blocks'),'target':episode['target'],
                             'kind':episode.get('kind',episode.get('case')),'instruction':episode.get('instruction'),'reason':episode['reason'],
                             'steps':episode['steps'],'initial_distance_m':round(episode['initial_distance_m'],1),'minimum_distance_m':round(episode['minimum_distance_m'],1),
                             'final_distance_m':round(episode['final_distance_m'],1),'climbed_m':round(episode.get('climbed_m',0.),1),
                             'climb_after_sight_m':round(episode['_climb_after_sight'],1),'started_hidden':episode['_stage']['started_hidden'],
                             'acquired':episode['_stage']['acquired'],'approached':episode['_stage']['approached'],'stopped':episode['_stage']['stopped'],
                             'success':episode['success'],'failure':episode['_failure'] or '','mechanism':episode.get('_mechanism','')})
    for item in args.checkpoint:
        model,directory=item.split('=',1);training=json.loads((Path(directory)/'manifest.json').read_text())['training'];data=training['dataset_summary']
        summary.setdefault('training',{})[model]={'checkpoint':directory,'dataset_episodes':data['episodes'],'dataset_samples':data['samples'],
            'train_samples_used':training['train_samples_used'],'best_update':training['best']['update'],'updates_run':training.get('updates_run',training['updates']),
            'stopped_early':training.get('stopped_early'),'train_l1_at_best':training['best']['train_l1_ema'],'val_l1_at_best':training['best']['val_l1'],
            'peak_allocated_GiB':training['peak_allocated_GiB'],'validation_history':[[h['update'],h['val']['l1']] for h in training['history'] if 'val' in h]}
    (output/'summary.json').write_text(json.dumps(summary,indent=1))
    for name,table in (('generalization_levels.csv',levels),('failure_taxonomy.csv',taxonomy),('motion_comparison.csv',motions),('episodes.csv',rows_out)):
        with (output/name).open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=list(table[0]) if table else ['empty']);writer.writeheader();writer.writerows(table)
    models=list(dict.fromkeys(run['model'] for run in runs))
    print('| Model | '+' | '.join(level for level in LEVELS)+' |');print('|---|'+'---:|'*len(LEVELS))
    for model in models:
        cells=[' + '.join(f'{row["success"]}/{row["episodes"]}' for row in levels if row['model']==model and row['level']==level) or '-' for level in LEVELS]
        print(f'| {model} | '+' | '.join(cells)+' |')
    print('\n| Model | Level | Hidden at start: acquired | Came within 20 m | Stopped by itself | Success |');print('|---|---|---:|---:|---:|---:|')
    for row in levels:
        print(f'| {row["model"]} | {row["level"]} | {row["acquired"]}/{row["started_hidden"]} | {row["approached"]}/{row["episodes"]} | {row["stopped"]}/{row["episodes"]} | {row["success"]}/{row["episodes"]} |')
    print('\n| Model | Level | Category | Count | Representative |');print('|---|---|---|---:|---|')
    for row in taxonomy:print(f'| {row["model"]} | {row["level"]} | {row["category"]} {row["name"]} | {row["count"]} | {row["representative_episode"]} |')


if __name__=='__main__':main()
