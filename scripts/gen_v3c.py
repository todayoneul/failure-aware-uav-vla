"""Gen-v3c: plans, gates and tables of the hard-negative correction; no simulator, no model.

  plan validation|pilot     write configs/gen_v3c_canonical_validation.json / configs/gen_v3c_pilot.json once
  plan train --output DIR   write the plan of the recorded episodes (DIR/plan.json) and its tracked copy
  check                     confirm that no recorded episode touches a start or a layout of any evaluation set
  ids                       the ten smoke starts of the validation set
  regression                the starts of the earlier sets that are flown again (ids by set)
  val-mix --dataset MERGED --added NAME
                            write MERGED/val_mix.json: the added validation frames and as many earlier ones
  gate pilot|smoke|representative|test --run DIR --teacher DIR [--output JSON]
                            exit 0 when passed. `test` reads the 48 canonical test starts by the representative thresholds.
  report MODEL:SET=DIR ... --teacher SET=DIR ... --output DIR
                            tables and files for the document, with the target-selection breakdown

Landings are read by canonical_evaluator_v2 (configs/targets/landing_pads.json, `evaluator`): the runs judged here were
recorded with it. Runs recorded before it keep the reading they were stored with.
"""
import argparse
import collections
import csv
import json
import math
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.canonical_baseline import flown,solved,measures,landing_rows,disagreements
from scripts.freeze_gen_v3 import rate,markdown,STAGES
from scripts.freeze_generalization_results import picture
from scripts.gate_gen_v3 import judge
from src.visual_search.episodes import load_config
from src.visual_search.maps import load_map
from src.visual_search import canonical,gen_v3c
from src.aerovla_oft.spec import load_config as load_oft_config

SETS={'validation':gen_v3c.FILES['validation'],'pilot':gen_v3c.FILES['pilot'],'test':canonical.FILES['test']}


def histogram(items,key):return dict(sorted(collections.Counter(key(item) for item in items).items(),key=lambda pair:str(pair[0])))


def describe_training(config,plans):
    """What the recorded episodes hold, split by split: kinds, objects, colours, shapes, bands, tasks."""
    scene=load_map('field');attribute=lambda name,key:scene['objects'][name]['attributes'][key];result={}
    for split,episodes in plans.items():
        result[split]={'episodes':len(episodes),'by_kind':histogram(episodes,lambda e:e['kind']),'by_task':histogram(episodes,lambda e:e.get('task','approach')),
                       'by_layout':histogram(episodes,lambda e:f'{e["map"]}/{e["layout"]}'),'by_target':histogram(episodes,lambda e:e['target']),
                       'by_target_colour':histogram(episodes,lambda e:attribute(e['target'],'color')),'by_target_shape':histogram(episodes,lambda e:attribute(e['target'],'shape')),
                       'by_band':histogram(episodes,lambda e:e['band']),'by_instruction':histogram(episodes,lambda e:e['instruction']),
                       'by_pair':histogram([e for e in episodes if e.get('distractor')],lambda e:f'{e["target"]} <- {e["distractor"]}'),
                       'by_relation':histogram([e for e in episodes if e.get('relation')],lambda e:e['relation']),
                       'target_out_of_view_at_start':sum(e['plan']['acquired_tick']!=0 for e in episodes),
                       'forced_turns':histogram([e for e in episodes if e.get('kick')],lambda e:'left' if e['kick']['degrees']<0 else 'right'),
                       'start_distance_m':[min(e['start_distance_m'] for e in episodes),max(e['start_distance_m'] for e in episodes)],
                       'teacher_ticks':[min(e['plan']['ticks'] for e in episodes),max(e['plan']['ticks'] for e in episodes)]}
    return result


def check_training(config,plans):
    """A recorded episode may not use a layout or come near a start of any evaluation set, old or new; nor leave the range."""
    settings=config['gen_v3c'];problems=[];radius=settings['separation_m'];points=gen_v3c.earlier_points(*gen_v3c.FILES.values())
    unseen={phrase for task in config['gen_v3']['instructions'].values() for phrase in task['held_out']}
    reserved={name:set(settings['pilot_layouts'][name])|set(settings['validation_layouts'][name])|set(config['canonical']['test_layouts'][name])|set(load_map(name)['held_out_layouts'])
              for name in gen_v3c.SCENES}
    for split,episodes in plans.items():
        low,high=settings['seeds'][split]
        for episode in episodes:
            name=episode['map']
            if name not in settings['train_layouts'] or episode['layout'] not in settings['train_layouts'][name]:problems.append(f'{episode["id"]}: {name}/{episode["layout"]} is not a Gen-v3c training layout')
            if episode['layout'] in reserved.get(name,set()):problems.append(f'{episode["id"]}: flown in a layout of an evaluation set')
            if episode['instruction_id'] in unseen:problems.append(f'{episode["id"]}: told in held-out words')
            if not low<=episode['seed']<=high:problems.append(f'{episode["id"]}: seed outside the {split} range')
            if episode['start_distance_m']>settings['max_start_m']+1e-6:problems.append(f'{episode["id"]}: starts farther than {settings["max_start_m"]} m')
            if 'yellow_pyramid' in (episode['target'],episode.get('distractor')):problems.append(f'{episode["id"]}: the held-out object')
            for where,(px,py) in points:
                if where==name and math.hypot(episode['start_xy'][0]-px,episode['start_xy'][1]-py)<radius:problems.append(f'{episode["id"]}: within {radius} m of an earlier start');break
    ids=[e['id'] for episodes in plans.values() for e in episodes]
    if len(ids)!=len(set(ids)):problems.append('duplicate episode ids')
    # The two splits do not share a start either (twins share theirs inside one split).
    for episode in plans.get('val',[]):
        if any(other['map']==episode['map'] and math.dist(other['start_xy'],episode['start_xy'])<radius for other in plans.get('train',[])):
            problems.append(f'{episode["id"]}: within {radius} m of a training start')
    return problems


def regression_ids(config):
    """The starts of the earlier sets flown again. A set lists its starts in cycles (objects, sentences), so a fixed step that
    shares no factor with the set's size is taken through it: every object and every kind of start comes up."""
    chosen={}
    def pick(episodes,count):
        size=len(episodes);step=next(value for value in range(math.ceil(size/count),size) if math.gcd(value,size)==1)
        return [episodes[index]['id'] for index in sorted((k*step)%size for k in range(count))]
    for name,(path,count) in config['gen_v3c']['regression']['sets'].items():
        sets=json.loads((ROOT/path).read_text(encoding='utf-8'))['sets']
        if name=='L':chosen.update({band:pick(sets[band],count//3) for band in ('L1','L2','L3')})
        else:chosen[name]=pick(sets[name],count)
    return chosen


def val_mix(dataset,added,seed=0):
    """Validation frames for the fine-tune: all of the added episodes' and as many of the earlier ones', by the teacher's states."""
    samples=json.loads((Path(dataset)/'val.json').read_text());prefix=f'../{added}/'
    new=[s for s in samples if s['traj_rel_dir'].startswith(prefix)];old=[s for s in samples if not s['traj_rel_dir'].startswith(prefix)]
    groups={}
    for sample in old:groups.setdefault(sample['meta']['teacher_state'],[]).append(sample)
    chosen=[]
    for name in sorted(groups):
        items=groups[name];random.Random(f'{seed}|{name}').shuffle(items);chosen+=items[:max(1,round(len(new)*len(items)/len(old)))]
    (Path(dataset)/'val_mix.json').write_text(json.dumps(new+chosen))
    return {'added_frames':len(new),'earlier_frames':len(chosen),'earlier_by_state':histogram(chosen,lambda s:s['meta']['teacher_state'])}


def checks(name,config,items,errors,validation=None):
    """The lines of one gate: {label: (value, comparison, limit)}. `validation` names another validation file with the same make-up."""
    settings=config['gen_v3c'];m=measures(items);total=max(1,m['flights'])
    if name=='pilot':
        gates=settings['pilot']
        return items,{'flights':(m['flights'],'>=',gates['episodes']),'successes':(m['success'],'>=',gates['success_min']),
                      'flights that ended at a wrong object':(m['wrong_target'],'<=',gates['wrong_target_max']),
                      'collisions':(m['collisions'],'<=',gates['collisions_max']),'episodes with a runner error':(len(errors),'<=',gates['errors_max'])}
    if name=='smoke':
        gates=settings['gates']['smoke'];wanted=set(canonical.smoke_ids(config,json.loads(Path(validation or SETS['validation']).read_text(encoding='utf-8'))['episodes']))
        items=[i for i in items if i['episode'] in wanted];m=measures(items)
        return items,{'flights':(m['flights'],'>=',gates['episodes']),'successes':(m['success'],'>=',gates['success_min']),
                      'landing successes (system)':(m['landing_success'],'>=',gates['landing_success_min']),
                      'successes from 32-44 m':(sum(i['success'] for i in items if i['band']=='far'),'>=',gates['far_band_success_min']),
                      'collisions':(m['collisions'],'<=',gates['collisions_max']),'episodes with a runner error':(len(errors),'<=',gates['errors_max'])}
    gates=settings['gates']['representative']
    return items,{'flights':(m['flights'],'>=',gates['episodes_min']),'success rate':(round(m['success']/total,3),'>=',gates['success_rate_min']),
                  'landing success rate (system: stable physical landing on the named pad, latched and disarmed)':(round(m['landing_success']/max(1,m['landings']),3),'>=',gates['landing_success_rate_min']),
                  'grounded when the target came into view':(round(m['grounded']/max(1,m['in_view']),3),'>=',gates['grounded_rate_min']),
                  'end of flight as the sentence asked':(round(m['terminal_as_asked']/total,3),'>=',gates['terminal_as_asked_rate_min']),
                  'flights that ended at a wrong object':(m['wrong_target'],'<=',gates['wrong_target_max']),
                  'collisions':(m['collisions'],'<=',gates['collisions_max']),'episodes with a runner error':(len(errors),'<=',gates['errors_max'])}


def lost_and_found(rows,stage):
    """Was the named object in view, then out of it for three decisions or more before the vehicle had come near, and seen again?
    Returns (lost, seen again) for one flight's decisions."""
    seen=[bool(row['front_seen'] or row['down_seen']) for row in rows];first=next((index for index,flag in enumerate(seen) if flag),None)
    if first is None:return False,False
    gap=0
    for index in range(first+1,len(rows)):
        if rows[index]['distance_m']<=stage['approach_m']:break
        gap=0 if seen[index] else gap+1
        if gap>=3:return True,any(seen[index+1:])
    return False,False


def selection(items,episodes,steps,landing,within_m=60.):
    """Target selection by what a start holds: [what, flights, successes, ended at the named object, ended at a wrong object].

    Read from the plan itself, so that any set can be broken down the same way: the object beside the named one (the
    `distractor` or `neighbour` of the start) and the objects the first view shows within `within_m`."""
    by_id={episode['id']:episode for episode in episodes};scenes={};rows=[]
    def facts(i):
        plan=by_id[i['episode']];scene=scenes.setdefault(plan['map'],load_map(plan['map']));target=plan['target']
        beside=plan.get('distractor') or plan.get('neighbour')
        shown={gen_v3c.relation_of(scene,target,item['name']) for item in plan.get('others_in_view',[]) if item['distance_m']<=within_m}-{None}
        hidden=plan['plan']['acquired_tick']!=0
        return {'beside':gen_v3c.relation_of(scene,target,beside) if beside else None,'first':shown if hidden else set(),'role':plan.get('role') or plan.get('kind',''),
                'hidden':hidden,'kick':bool(plan.get('kick'))}
    known={i['episode']:facts(i) for i in items}
    def line(name,chosen):
        if chosen:rows.append((name,len(chosen),sum(bool(i['success']) for i in chosen),sum(bool(i['correct_target']) for i in chosen),sum(i['wrong_target'] for i in chosen)))
    of=lambda i:known[i['episode']]
    line('an object of the same colour and another shape beside the named one or in the first view',[i for i in items if of(i)['beside']=='same_color' or 'same_color' in of(i)['first']])
    line('an object of the same shape and another colour beside the named one or in the first view',[i for i in items if of(i)['beside']=='same_shape' or 'same_shape' in of(i)['first']])
    line('a related object in the first view, the named one outside it',[i for i in items if of(i)['first']])
    line('... of the same colour',[i for i in items if 'same_color' in of(i)['first']])
    line('... of the same shape',[i for i in items if 'same_shape' in of(i)['first']])
    line('the two objects exchanged (position swap)',[i for i in items if of(i)['role']=='swap'])
    line('the neighbour named instead (query)',[i for i in items if of(i)['role']=='query'])
    line('the named object outside the first view',[i for i in items if of(i)['hidden']])
    line('the named object in the first view',[i for i in items if not of(i)['hidden']])
    line('named object lost to a forced turn, a related object in view',[i for i in items if of(i)['kick']])
    natural=[];again=[]
    for i in items:
        if of(i)['kick']:continue
        lost,found=lost_and_found(steps[i['episode']],landing['stages'])
        if lost:natural.append(i);again+=[i] if found else []
    if natural:
        rows.append(('named object lost on the way without a forced turn',len(natural),sum(bool(i['success']) for i in natural),sum(bool(i['correct_target']) for i in natural),sum(i['wrong_target'] for i in natural)))
        rows.append(('... and seen again',len(again),sum(bool(i['success']) for i in again),sum(bool(i['correct_target']) for i in again),sum(i['wrong_target'] for i in again)))
    return rows


def report(args,config):
    from src.visual_search.maps import load_landing
    output=Path(args.output);(output/'failures').mkdir(parents=True,exist_ok=True);summary={'models':{},'left_out':{}};rows_out=[];landing=load_landing()
    teacher={item.split('=',1)[0]:solved(item.split('=',1)[1]) for item in args.teacher}
    for item in args.runs:
        label,directory=item.split('=',1);model,level=label.rsplit(':',1);items,errors,left,(episodes,steps)=flown(directory,teacher.get(level))
        summary['left_out'][label]=left;m=measures(items);land=[i for i in items if i['task']=='land'];touched=[i for i in land if i['touchdown_on_target']]
        print();print(f'# {model}, {level}');print(json.dumps(m))
        markdown('Landings by start distance',('Band','Correct target','Approach','Alignment','Touchdown','Stable physical landing','System landing','Strict zero-action'),landing_rows(items))
        by_role=[(role,rate([i for i in items if i['role']==role],'success')) for role in dict.fromkeys(i['role'] for i in items)]
        markdown('By kind of start',('Start','Success'),by_role)
        by_band=[(band,rate([i for i in items if i['band']==band],'success'),rate([i for i in items if i['band']==band and i['task']=='land'],'success'),
                  rate([i for i in items if i['band']==band and i['task']=='approach'],'success')) for band in ('near','mid','far')]
        markdown('By band',('Band','All','Landing','Approach'),by_band)
        chosen=selection(items,episodes,steps,landing)
        markdown('Target selection',('Start holds','Flights','Success','Ended at the named object','Ended at a wrong object'),chosen)
        failed=[i for i in items if i['failure']]
        stages=[(stage,sum(i['failure']==stage for i in failed),', '.join(sorted({i['mechanism'] for i in failed if i['failure']==stage})),
                 ' '.join(i['episode'] for i in failed if i['failure']==stage)) for stage in STAGES]
        markdown('Failures by the first stage not passed',('Stage','Count','How','Episodes'),[row for row in stages if row[1]])
        active=[i for i in items if i['finalizer_active']];by_id={episode['id']:episode for episode in episodes}
        finalizer={'landing_missions':len(land),'contacts':sum(bool(by_id[i['episode']].get('landed')) for i in land),
                   'triggered':sum(bool(i['finalizer_triggered']) for i in land),'disarmed':sum(bool(i['disarm_triggered']) for i in land),
                   'rescued_success':sum(bool(by_id[i['episode']].get('finalizer_rescued_success')) for i in land),
                   'approach_missions':sum(i['task']=='approach' for i in items),'active_in_approach_missions':sum(i['task']=='approach' for i in active),
                   'evaluator':sorted({by_id[i['episode']].get('evaluator','legacy') for i in land}),
                   'legacy_reading_disagrees':[i['episode'] for i in land if by_id[i['episode']].get('legacy_evaluator') and
                                               by_id[i['episode']]['legacy_evaluator']['stable_physical_landing']!=bool(i['stable_physical_landing'])]}
        print();print('Finalizer:',json.dumps(finalizer))
        errors_m=[i['touchdown_error_m'] for i in touched];speeds=[i['touchdown_vertical_mps'] for i in touched]
        precision={'touchdowns_on_the_named_pad':len(touched),'horizontal_error_m':{'median':sorted(errors_m)[len(errors_m)//2] if errors_m else None,'max':max(errors_m,default=None)},
                   'vertical_speed_mps':{'median':sorted(speeds)[len(speeds)//2] if speeds else None,'max':max(speeds,default=None)}}
        print('Touchdown:',json.dumps(precision))
        apart=disagreements(items,steps)
        summary['models'][label]={'measures':m,'landings':[dict(zip(('band','correct_target','approach','alignment','touchdown','stable_physical_landing','system_landing','strict_zero_action'),row)) for row in landing_rows(items)],
                                  'by_role':dict(by_role),'by_band':[dict(zip(('band','all','landing','approach'),row)) for row in by_band],
                                  'target_selection':[dict(zip(('start_holds','flights','success','named_object','wrong_object'),row)) for row in chosen],
                                  'failures':{stage:{'count':count,'how':how,'episodes':names.split()} for stage,count,how,names in stages if count},
                                  'finalizer':finalizer,'touchdown':precision,'latched_but_not_read_as_stable':apart,'episodes_with_errors':errors,
                                  'rates':{'correct_target':rate(items,'correct_target'),'approach':rate(items,'approached'),'touchdown':rate(land,'touchdown_success'),
                                           'stable_physical_landing':rate(land,'stable_physical_landing'),'system_landing':rate(land,'system_landing'),
                                           'strict_policy_zero_action':rate(land,'strict_policy_zero_action'),'terminal_as_asked':rate(items,'terminal_as_asked'),
                                           'wrong_target':m['wrong_target'],'collisions':m['collisions']}}
        for record in items:
            rows_out.append({'model':model,'set':level,**{k:v for k,v in record.items()}})
            if record['failure'] and model!='Teacher':
                name=f'{model}_{level}_{record["failure"]}_{record["episode"]}'.replace(' ','_');trace_rows=steps[record['episode']]
                trace=[{key:row.get(key) for key in ('step','distance_m','bearing_deg','height_m','front_seen','down_seen','over','landed','on_surface','finalizer','velocity','action','executed')} for row in trace_rows]
                (output/'failures'/f'{name}.json').write_text(json.dumps({**record,'trace':trace},indent=1))
                picture(by_id[record['episode']],trace_rows,output/'failures'/f'{name}.jpg',f'{model} {level} {record["failure"]}: {record["episode"]}')
    with (output/'episodes.csv').open('w',newline='',encoding='utf-8') as file:
        fields=list(dict.fromkeys(key for row in rows_out for key in row));writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows_out)
    (output/'summary.json').write_text(json.dumps(summary,indent=1))


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    plan=commands.add_parser('plan');plan.add_argument('which',choices=('validation','pilot','train'));plan.add_argument('--output',help='dataset folder of the recorded episodes (train)')
    commands.add_parser('check').add_argument('--plan',default=str(gen_v3c.TRAIN_PLAN));commands.add_parser('ids');commands.add_parser('regression')
    mix=commands.add_parser('val-mix');mix.add_argument('--dataset',required=True);mix.add_argument('--added',required=True)
    gate=commands.add_parser('gate');gate.add_argument('name',choices=('pilot','smoke','representative','test'));gate.add_argument('--run',required=True)
    gate.add_argument('--teacher',required=True,help="the teacher's run over the same set");gate.add_argument('--output')
    rep=commands.add_parser('report');rep.add_argument('runs',nargs='+',help='MODEL:SET=DIRECTORY');rep.add_argument('--teacher',nargs='*',default=[],help="SET=DIRECTORY of the teacher's run")
    rep.add_argument('--output',required=True)
    args=parser.parse_args();config=load_config();line=chr(10)
    if args.command=='plan':
        oft=load_oft_config()
        if args.which=='train':
            if not all(path.exists() for path in gen_v3c.FILES.values()):raise SystemExit('the validation and the pilot starts are written first; the recorded episodes keep away from them')
            plans={'train':gen_v3c.build_training(config,oft,'train')};plans['val']=gen_v3c.build_training(config,oft,'val',plans['train']);problems=check_training(config,plans)
            if problems:raise SystemExit(line.join(problems[:20]))
            data={'recipe':'gen_v3c.hard_negative','summary':describe_training(config,plans),**plans}
            for path in ([Path(args.output)/'plan.json'] if args.output else [])+[gen_v3c.TRAIN_PLAN]:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=1)+line,encoding='utf-8',newline=line)
            for split,text in data['summary'].items():print(split,json.dumps(text))
            return
        path=gen_v3c.FILES[args.which]
        if path.exists():raise SystemExit(f'{path} exists; the Gen-v3c {args.which} starts are fixed and are not regenerated')
        if args.which=='pilot' and not gen_v3c.FILES['validation'].exists():raise SystemExit('the validation starts are written first')
        episodes=gen_v3c.build_validation(config,oft) if args.which=='validation' else gen_v3c.build_pilot(config,oft)
        path.write_text(json.dumps(gen_v3c.describe(config,args.which,episodes),indent=1)+line,encoding='utf-8',newline=line)
        print(args.which,len(episodes),json.dumps({'by_band':histogram(episodes,lambda e:e['band']),'by_role':histogram(episodes,lambda e:e['role']),
                                                   'by_layout':histogram(episodes,lambda e:f'{e["map"]}/{e["layout"]}'),'landings':sum(e['task']=='land' for e in episodes),
                                                   'seeds':[min(e['seed'] for e in episodes),max(e['seed'] for e in episodes)]}));return
    if args.command=='check':
        data=json.loads(Path(args.plan).read_text(encoding='utf-8'));problems=check_training(config,{split:data[split] for split in ('train','val')})
        if problems:raise SystemExit(line.join(problems[:20]))
        print(f'OK: {len(data["train"])+len(data["val"])} recorded starts are in the training layouts, inside the range and clear of '
              f'{len(gen_v3c.earlier_points(*gen_v3c.FILES.values()))} earlier starts (radius {config["gen_v3c"]["separation_m"]} m)');return
    if args.command=='ids':print(' '.join(canonical.smoke_ids(config,json.loads(SETS['validation'].read_text(encoding='utf-8'))['episodes'])));return
    if args.command=='regression':print(json.dumps(regression_ids(config)));return
    if args.command=='val-mix':print(json.dumps(val_mix(args.dataset,args.added)));return
    if args.command=='report':report(args,config);return
    items,errors,left,_=flown(args.run,solved(args.teacher));items,lines=checks(args.name,config,items,errors);table,passed=judge(lines)
    result={'gate':args.name,'run':args.run,'passed':passed,'checks':table,**measures(items),'left_out_because_the_teacher_did_not_finish':left,
            'failures':{i['episode']:f'{i["failure"]} ({i["mechanism"]})' for i in items if i['failure']}}
    print(json.dumps(result,indent=1))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=1))
    print(f'GEN-V3C {args.name.upper()}: {"PASSED" if passed else "NOT PASSED"}')
    sys.exit(0 if passed else 1)


if __name__=='__main__':main()
