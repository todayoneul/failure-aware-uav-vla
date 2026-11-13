"""Freeze a Gen-v3 evaluation as small files; no simulator, no model.

Runs are given as "MODEL:SET=DIRECTORY" and only read. Every episode is read the same way, whichever
model flew it and whenever, from its step log:

  acquired     the target was in the front or down view at some decision (the simulator's statement)
  grounded     while it was in view the policy began to fly toward it: at least three decisions with a
               forward command of 0.25 m or more in one stretch of it being in view. This is the step
               the long-range evaluation found missing: in view, and the vehicle kept turning.
  approached   the vehicle came within 20 m of the target
  selected     the object the flight ended at: the pad it stands on, or the nearest object within the
               success radius if it stopped by itself
  success      approach: stopped by itself in the air within the radius of the target, without having come down over a pad
               land: the landing rule of configs/targets/landing_pads.json

A failed episode gets exactly one stage, the first one it did not pass:
  COLLISION, SEARCH, GROUNDING, WRONG_TARGET, APPROACH, then for an approach STOP and for a landing
  ALIGNMENT, DESCENT, LANDING. A landing flown with the low-level finalizer has, in place of LANDING,
  PHYSICAL_LANDING (it touched the named pad but did not come to rest on it inside the landing region)
  and FINALIZER (it rested there and the finalizer did not latch or disarm). Runs from before the
  finalizer are read exactly as they were.

Output: summary.json, episodes.csv, one csv per table and failures/ (a trace and a picture per failure
of the models named with --pictures).
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_visual_search import load
from scripts.freeze_generalization_results import nearest_other,picture
from src.visual_search.maps import load_map,load_landing

FORWARD_M=.25
FAR_M=54.
GROUNDED_STEPS=3
STAGES=('COLLISION','SEARCH','GROUNDING','WRONG_TARGET','APPROACH','STOP','ALIGNMENT','DESCENT','LANDING','PHYSICAL_LANDING','FINALIZER')
GROUNDING_SETS=(('Q1','Basic'),('Q2','Colour distractor'),('Q3','Shape distractor'),('Q4','Position swap'),('Q5','Query swap'),('Q10','Paraphrase (landing)'))
REGRESSION_SETS=('G1','G3','P','L1','L2','L3')


def in_view(row):return bool(row.get('front_seen') or row.get('down_seen'))


def transition(rows):
    """Did the policy start toward the target while it was in view, and on which pass of it through the view?"""
    stretches=[];current=None
    for index,row in enumerate(rows):
        if in_view(row):
            if current is None:current=[];stretches.append(current)
            current.append(index)
        else:current=None
    def begun(stretch):
        run=best=0
        for index in stretch:
            run=run+1 if rows[index]['action'][0]>=FORWARD_M else 0;best=max(best,run)
        return best>=GROUNDED_STEPS
    passes=[begun(stretch) for stretch in stretches];first=next((index for index,flag in enumerate(passes) if flag),None)
    return {'views':len(stretches),'decisions_in_view':sum(len(stretch) for stretch in stretches),'grounded':first is not None,
            'grounded_on_first_view':first==0,'views_before_grounding':first,
            'turns_deg':round(sum(abs(math.degrees(row['action'][2])) for row in rows))}


def read(episode,rows,radius,landing,geometries,maps):
    """Everything recorded about one flown episode, from an old run or a new one."""
    task=episode.get('task','approach');target=episode['target'];distances=[row['distance_m'] for row in rows];stage=landing['stages']
    move=transition(rows);other=nearest_other(episode,rows,geometries)
    landed=bool(episode.get('landed'));down=episode.get('touchdown') or {}
    selected=episode.get('selected')
    if 'selected' not in episode:
        # A run from before the landing phase: the object it stopped at.
        selected=target if episode['stopped'] and distances[-1]<=radius else (episode.get('stopped_at_other') or [None])[0] if episode['stopped'] else None
    over=[index for index,row in enumerate(rows) if row.get('over')]
    aligned=any(row.get('over') and row['distance_m']<=stage['aligned_m'] for row in rows)
    descended=bool(over) and min(row['height_m'] for row in rows[over[0]:])<=rows[over[0]]['height_m']-stage['descent_started_m']
    # Stopped in the air without having come down over a pad: what an approach asks for.
    hovered=bool(episode['stopped'] and not landed and not descended)
    # The three readings of a landing and the system's result; absent in runs from before the finalizer.
    finalizer=episode.get('finalizer') or {}
    scene=maps.setdefault(episode.get('map','blocks'),load_map(episode.get('map','blocks')));objects=scene['objects']
    def relation(name):
        if name is None:return 'none'
        if name==target:return 'target'
        a,b=objects[target].get('attributes',{}),objects[name].get('attributes',{})
        return 'same colour' if a.get('color') and a.get('color')==b.get('color') else 'same shape' if a.get('shape') and a.get('shape')==b.get('shape') else 'other'
    # A wrong object: the flight ended at one, or it came within the radius of one without ever nearing the target.
    wrong=selected not in (None,target) or (other[0]<=radius and min(distances)>stage['approach_m'])
    record={'task':task,'target':target,'success':bool(episode['success']),'acquired':any(in_view(row) for row in rows),'initially_visible':in_view(rows[0]),
            **move,'approached':min(distances)<=stage['approach_m'],'selected':selected,'selected_relation':relation(selected),
            'correct_target':selected==target,'wrong_target':bool(wrong),'wrong_object':selected if selected not in (None,target) else other[1] if wrong else None,
            'aligned':aligned,'descent_started':descended,'landed':landed,'landed_on':episode.get('landed_on'),
            'touchdown_on_target':landed and episode.get('landed_on')==target,'self_stop':bool(episode['stopped']),'hovered':hovered,
            'terminal':'landed' if landed else 'hovered' if hovered else 'neither',
            'terminal_as_asked':('landed' if task=='land' else 'hovered')==('landed' if landed else 'hovered' if hovered else 'neither'),
            'collision':episode['reason']=='collision','timeout':episode['reason']=='max_steps','reason':episode['reason'],'steps':len(rows),
            'initial_distance_m':round(distances[0],1),'minimum_distance_m':round(min(distances),1),'final_distance_m':round(distances[-1],1),
            'final_height_m':round(rows[-1]['height_m'],1),'nearest_other_m':round(other[0],1),'nearest_other':other[1],
            'touchdown_error_m':None if not down else round(down['horizontal_error_m'],2),
            'touchdown_vertical_mps':None if not down else round(abs(down['vertical_speed_mps']),2),
            'touchdown_horizontal_mps':None if not down else round(down['horizontal_speed_mps'],2),
            'touchdown_inside':down.get('inside_region'),'touchdown_soft':down.get('soft'),
            'finalizer_active':bool(finalizer.get('active')),'touchdown_success':episode.get('touchdown_success'),
            'stable_physical_landing':episode.get('stable_physical_landing'),'system_landing':episode.get('system_land_success'),
            # From evaluator v2 on: standing still on a pad whichever pad it is, and which reading the run was recorded with.
            'physical_landing':episode.get('physical_landing'),'evaluator':episode.get('evaluator','legacy') if task=='land' else None,
            'finalizer_rescued_success':episode.get('finalizer_rescued_success'),
            'strict_policy_zero_action':episode.get('strict_policy_zero_action',bool(landed and episode.get('landed_on')==target and episode['stopped'])) if task=='land' else None,
            'finalizer_triggered':finalizer.get('finalizer_triggered'),'disarm_triggered':finalizer.get('disarm_triggered'),
            'stable_duration_s':finalizer.get('stable_duration_s'),'finalizer_state':finalizer.get('state')}
    record['failure'],record['mechanism']=classify(record)
    return record


def classify(r):
    """The first stage a failed episode did not pass, and a word on how."""
    if r['success']:return '',''
    how='timeout' if r['timeout'] else 'stopped by itself' if r['self_stop'] else r['reason']
    if r['collision']:return 'COLLISION',how
    if not r['acquired']:return 'SEARCH',how
    if r['wrong_target']:return 'WRONG_TARGET',f'at {r["wrong_object"]}'
    if not r['grounded']:return 'GROUNDING','passed over' if r['views']>=2 else how
    if not r['approached']:return 'APPROACH','stopped early' if r['self_stop'] else how
    if r['task']=='approach':return 'STOP','landed instead of hovering' if r['landed'] else 'stopped outside the radius' if r['self_stop'] else 'did not stop'
    if not r['aligned']:return 'ALIGNMENT','hovered instead of landing' if r['hovered'] else how
    if not r['touchdown_on_target']:return 'DESCENT','hovered instead of landing' if r['hovered'] else 'did not start' if not r['descent_started'] else how
    if r.get('finalizer_active'):
        if not r['stable_physical_landing']:
            return 'PHYSICAL_LANDING','outside the landing region' if not r['touchdown_inside'] else 'hard touchdown' if not r['touchdown_soft'] else 'did not stay on the pad'
        return 'FINALIZER','did not latch' if not r['finalizer_triggered'] else 'did not disarm'
    return 'LANDING','no stop after touchdown' if not r['self_stop'] else 'outside the landing region' if not r['touchdown_inside'] else 'hard touchdown'


def rate(items,key):
    return f'{sum(bool(item[key]) for item in items)}/{len(items)}' if items else '-'


def percentile(values,fraction):
    values=sorted(values)
    return None if not values else values[min(len(values)-1,int(math.ceil(fraction*len(values)))-1)]


def markdown(title,header,rows):
    print(f'\n### {title}\n');print('| '+' | '.join(header)+' |');print('|'+'---|'*len(header))
    for row in rows:print('| '+' | '.join(str(cell) for cell in row)+' |')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',help='MODEL:SET=DIRECTORY');parser.add_argument('--output',required=True)
    parser.add_argument('--pictures',nargs='*',default=[],help='models whose failures are drawn')
    parser.add_argument('--solvable-by',help='a model (the teacher) whose failed starts are left out of every other model: a start nobody can finish measures nothing')
    args=parser.parse_args();output=Path(args.output);(output/'failures').mkdir(parents=True,exist_ok=True)
    landing=load_landing();geometries={};maps={};items=[];models=[];errors={};tables={}
    for item in args.runs:
        label,directory=item.split('=',1);model,level=label.rsplit(':',1);_,episodes,steps=load(directory)
        data=json.loads((Path(directory)/'results.json').read_text());radius=data['config']['success_radius_m']
        errors[label]=[episode['id'] for episode in data['episodes'] if episode.get('error')]
        if model not in models:models.append(model)
        for episode in episodes:
            # A start with forced pushes is a training device, not a test of a policy.
            if episode.get('pushes'):continue
            rows=steps[episode['id']];record=read(episode,rows,radius,landing,geometries,maps)
            record={'model':model,'set':level,'episode':episode['id'],'map':episode.get('map','blocks'),'layout':episode.get('layout','pilot'),
                    'kind':episode.get('kind',episode.get('case')),'instruction':episode['instruction'],'range':episode.get('range',''),
                    'pose':episode.get('query_pose',episode.get('terminal_pose','')),'twin_of':episode.get('swap_of',episode.get('base','')),**record}
            items.append(record)
            if record['failure'] and model in args.pictures:
                name=f'{model}_{level}_{record["failure"]}_{episode["id"]}'.replace(' ','_')
                trace=[{key:row.get(key) for key in ('step','distance_m','bearing_deg','height_m','front_seen','down_seen','over','landed','action','teacher_state')} for row in rows]
                (output/'failures'/f'{name}.json').write_text(json.dumps({**record,'trace':trace},indent=1))
                picture(episode,rows,output/'failures'/f'{name}.jpg',f'{model} {level} {record["failure"]}: {episode["id"]}')
    unsolved=sorted({(item['set'],item['episode']) for item in items if item['model']==args.solvable_by and not item['success']}) if args.solvable_by else []
    left_out=[item for item in items if (item['set'],item['episode']) in set(unsolved) and item['model']!=args.solvable_by]
    items=[item for item in items if item not in left_out]
    if unsolved:print(f'Left out of every model because {args.solvable_by} did not finish them: '+', '.join(f'{level}/{name}' for level,name in unsolved))
    summary={'left_out':{'because':f'{args.solvable_by} did not finish the start in the simulator','episodes':[f'{level}/{name}' for level,name in unsolved]},
             'definitions':{'forward_m':FORWARD_M,'grounded_steps':GROUNDED_STEPS,'approach_m':landing['stages']['approach_m'],'aligned_m':landing['stages']['aligned_m'],
                            'landing_rule':landing['touchdown'],'stages':list(STAGES)},'episodes_with_errors':errors,'models':{}}
    for model in models:
        mine=[item for item in items if item['model']==model];of=lambda *names:[item for item in mine if item['set'] in names]
        print(f'\n# {model}');result={}
        # Precise grounding: which object the flight ended at.
        rows=[(name,rate(of(level),'success'),rate(of(level),'correct_target'),sum(item['wrong_target'] for item in of(level))) for level,name in GROUNDING_SETS if of(level)]
        swaps=of('Q4');bases={item['episode']:item for item in of('Q2','Q3')}
        pairs=[(bases[item['twin_of']],item) for item in swaps if item['twin_of'] in bases]
        poses=sorted({item['pose'] for item in of('Q5')})
        result['grounding']={'sets':[dict(zip(('test','success','correct_target','wrong_target'),row)) for row in rows],
                             'position_swap_pairs_both_correct':f'{sum(a["success"] and b["success"] for a,b in pairs)}/{len(pairs)}' if pairs else '-',
                             'query_swap_poses_all_correct':f'{sum(all(i["success"] for i in of("Q5") if i["pose"]==pose) for pose in poses)}/{len(poses)}' if poses else '-'}
        if rows:
            markdown('Precise grounding',('Test','Success','Ended at the named object','Wrong object'),rows)
            print(f'\nPosition swap, both members of a pair correct: {result["grounding"]["position_swap_pairs_both_correct"]}; '
                  f'query swap, all four sentences correct from one pose: {result["grounding"]["query_swap_poses_all_correct"]}')
        fresh=of('Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q8R','Q10','QS')
        result['wrong_target']={'episodes':len(fresh),'wrong':sum(item['wrong_target'] for item in fresh)}
        # Which attribute was missed when the flight ended at something.
        columns=('target','same colour','same shape','other','none');targets=sorted({item['target'] for item in fresh})
        matrix=[(target,*(sum(item['selected_relation']==column for item in fresh if item['target']==target) for column in columns)) for target in targets]
        result['confusion']=[dict(zip(('named',)+columns,row)) for row in matrix]
        if matrix:markdown('Where flights ended, by the object named',('Named',)+tuple(f'Ended at: {c}' for c in columns),matrix)
        # Far grounding: in view, grounded, approached, final.
        for title,levels in (('Far grounding (Q7)',(('F1','40-55m'),('F2','55-70m'),('F3','70-90m'))),):
            rows=[(name,rate(group,'acquired'),rate(group,'grounded'),rate(group,'grounded_on_first_view'),rate(group,'approached'),rate(group,'success'))
                  for band,name in levels for group in [[item for item in of('Q7') if item['range']==band]] if group]
            result['far']=[dict(zip(('range','in_view','grounded','grounded_on_first_view','approached','success'),row)) for row in rows]
            if rows:markdown(title,('Range','In view','Grounded','Grounded on first view','Came within 20 m','Final'),rows)
        rows=[(level,rate(of(level),'acquired'),rate(of(level),'grounded'),rate(of(level),'grounded_on_first_view'),rate(of(level),'approached'),rate(of(level),'success'))
              for level in ('L1','L2','L3') if of(level)]
        result['long_range']=[dict(zip(('range','in_view','grounded','grounded_on_first_view','approached','success'),row)) for row in rows]
        if rows:markdown('Long-range set L (flown before; development-exposed)',('Range','In view','Grounded','Grounded on first view','Came within 20 m','Final'),rows)
        # Landing by start distance.
        for level,title in (('Q8','Landing: find the blue landing pad and land on it (Q8)'),):
            rows=[(band.capitalize(),rate(group,'correct_target'),rate(group,'approached'),rate(group,'aligned'),rate(group,'descent_started'),
                   rate(group,'touchdown_on_target'),rate(group,'success')) for band in ('near','medium','far') for group in [[i for i in of(level) if i['range']==band]] if group]
            if of(level):rows.append(('All',*(rate(of(level),key) for key in ('correct_target','approached','aligned','descent_started','touchdown_on_target','success'))))
            result['landing']=[dict(zip(('range','correct_target','approached','aligned','descent_started','touchdown','landing'),row)) for row in rows]
            if rows:markdown(title,('Range','Correct target','Approach','Alignment','Descent started','Touchdown','Landing success'),rows)
        landings=[item for item in fresh if item['task']=='land'];touched=[item for item in landings if item['touchdown_on_target']]
        if landings:
            errors_m=[item['touchdown_error_m'] for item in touched]
            result['touchdown']={'landing_episodes':len(landings),'success':sum(i['success'] for i in landings),'touchdowns_on_target':len(touched),
                                 'horizontal_error_m':{'median':statistics.median(errors_m) if errors_m else None,'p90':percentile(errors_m,.9),'max':max(errors_m) if errors_m else None},
                                 'vertical_speed_mps':{'median':statistics.median(i['touchdown_vertical_mps'] for i in touched) if touched else None,
                                                       'max':max((i['touchdown_vertical_mps'] for i in touched),default=None)},
                                 'horizontal_speed_mps':{'median':statistics.median(i['touchdown_horizontal_mps'] for i in touched) if touched else None,
                                                         'max':max((i['touchdown_horizontal_mps'] for i in touched),default=None)},
                                 'landed_on_another_pad':sum(i['landed'] and not i['touchdown_on_target'] for i in landings)}
            print('\nTouchdown on the named pad:',json.dumps(result['touchdown']))
            rows=[(level,rate(of(level),'success')) for level in ('Q8','Q8R','Q10','QS') for _ in [0] if [i for i in of(level) if i['task']=='land']]
            rows=[(level,rate([i for i in of(level) if i['task']=='land'],'success')) for level,_ in rows]
            result['landing_sets']=dict(rows);markdown('Landing missions by set',('Set','Landing success'),rows)
        # The sentence decides the end of the flight.
        group=of('Q6')
        if group:
            asked=lambda task,words=None:[i for i in group if i['task']==task and (words is None or words in i['instruction'])]
            rows=[('Find ... and land on it. (trained words)',rate(asked('land'),'landed'),rate(asked('land'),'hovered'),rate(asked('land'),'success')),
                  ('Approach ... (trained words)',rate(asked('approach','Approach the'),'landed'),rate(asked('approach','Approach the'),'hovered'),rate(asked('approach','Approach the'),'success')),
                  ('Find and approach ... (unseen words)',rate(asked('approach','Find and approach'),'landed'),rate(asked('approach','Find and approach'),'hovered'),
                   rate(asked('approach','Find and approach'),'success'))]
            poses=sorted({item['pose'] for item in group})
            result['terminal']={'rows':[dict(zip(('sentence','landed','hovered','success'),row)) for row in rows],'as_asked':rate(group,'terminal_as_asked'),
                                'poses_all_as_asked':f'{sum(all(i["terminal_as_asked"] for i in group if i["pose"]==pose) for pose in poses)}/{len(poses)}'}
            markdown('Approach or land from one pose (Q6)',('Sentence','Landed','Stopped in the air','Success'),rows)
            print(f'\nEnd of flight as the sentence asked: {result["terminal"]["as_asked"]}; all three sentences from one pose: {result["terminal"]["poses_all_as_asked"]}')
        # Scene C against the training scenes, and the earlier sets.
        depot=[item for item in fresh if item['map']=='depot'];seen=of('QS')
        rows=[('Depot (scene C), approach',rate([i for i in depot if i['task']=='approach'],'success')),('Depot (scene C), landing',rate([i for i in depot if i['task']=='land'],'success')),
              ('Training scenes, held-out layout, approach',rate([i for i in seen if i['task']=='approach'],'success')),
              ('Training scenes, held-out layout, landing',rate([i for i in seen if i['task']=='land'],'success'))]
        result['scenes']=dict(rows)
        if fresh:markdown('Scene',('Where','Success'),rows)
        # Validation starts of the training plan: training scenes, seeds no training episode used. Development data, not a test.
        dev=of('VAL')
        if dev:
            blue=lambda item:maps[item['map']]['objects'][item['target']].get('attributes',{}).get('color')=='blue'
            far=lambda item:item['initial_distance_m']>=FAR_M
            groups=(('All',dev),('Approach',[i for i in dev if i['task']=='approach']),('Landing',[i for i in dev if i['task']=='land']),
                    (f'Start under {FAR_M:.0f} m',[i for i in dev if not far(i)]),(f'Start {FAR_M:.0f} m and more',[i for i in dev if far(i)]),
                    ('Blue target',[i for i in dev if blue(i)]),('Target of another colour',[i for i in dev if not blue(i)]),
                    (f'Blue target, {FAR_M:.0f} m and more',[i for i in dev if blue(i) and far(i)]),(f'Other colour, {FAR_M:.0f} m and more',[i for i in dev if not blue(i) and far(i)]))
            rows=[(name,rate(group,'success'),rate([i for i in group if i['acquired']],'grounded'),rate(group,'terminal_as_asked'),sum(i['wrong_target'] for i in group),
                   rate([i for i in group if i['task']=='land'],'touchdown_on_target')) for name,group in groups]
            result['validation']=[dict(zip(('group','success','grounded','terminal_as_asked','wrong_target','touchdown_on_named_pad'),row)) for row in rows]
            markdown('Validation starts (development data, not the test set)',('Group','Success','Grounded','End as asked','Wrong object','Touchdown on the named pad'),rows)
        rows=[(level,rate(of(level),'success'),rate(of(level),'grounded'),sum(i['collision'] for i in of(level))) for level in REGRESSION_SETS if of(level)]
        result['regression']=[dict(zip(('set','success','grounded','collisions'),row)) for row in rows]
        if rows:markdown('Earlier sets',('Set','Success','Grounded','Collisions'),rows)
        failed=[item for item in mine if item['failure']]
        rows=[(stage,len(found),', '.join(sorted({item['mechanism'] for item in found})),' '.join(item['episode'] for item in found[:6])+(' ...' if len(found)>6 else ''))
              for stage in STAGES for found in [[item for item in failed if item['failure']==stage]]]
        result['failures']=[{'stage':stage,'count':count,'how':how,'by_set':{level:sum(i['set']==level for i in failed if i['failure']==stage) for level in sorted({i['set'] for i in failed})}}
                            for stage,count,how,_ in rows]
        markdown('Failures by the first stage not passed',('Stage','Count','How','Episodes'),rows)
        result['totals']={'episodes':len(mine),'success':sum(i['success'] for i in mine),'collisions':sum(i['collision'] for i in mine),'timeouts':sum(i['timeout'] for i in mine),
                          'acquired':rate(mine,'acquired'),'grounded':rate([i for i in mine if i['acquired']],'grounded'),
                          'hidden_at_start_acquired':rate([i for i in mine if not i['initially_visible']],'acquired'),
                          'by_set':{level:rate(of(level),'success') for level in sorted({i['set'] for i in mine})}}
        print('\nTotals:',json.dumps(result['totals']))
        summary['models'][model]=result;tables[model]=result
    with (output/'episodes.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    for name,key in (('grounding.csv','grounding'),('far_grounding.csv','far'),('landing.csv','landing'),('confusion.csv','confusion'),('failures.csv','failures'),
                     ('regression.csv','regression'),('long_range.csv','long_range'),('validation.csv','validation')):
        rows=[]
        for model,result in tables.items():
            block=result.get(key) or [];block=block['sets'] if isinstance(block,dict) else block
            rows+=[{'model':model,**{k:(json.dumps(v) if isinstance(v,dict) else v) for k,v in row.items()}} for row in block]
        if rows:
            with (output/name).open('w',newline='',encoding='utf-8') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=1))


if __name__=='__main__':main()
