"""Phase B, stage 1 (generic landing): plans, gates and tables; no simulator, no model.

  plan                      write configs/generic_landing/{pilot,validation,regression,train_plan}.json once
  check                     confirm the plans are what the configuration gives, that every recorded start is in a training
                            layout and every check start in an evaluation layout, and that all starts keep their distance
  ids smoke|critical|repeat the validation starts flown first; the ones flown a second time (stage 1b: repeat, the cube and
                            cylinder landing starts)
  pilot --runs NAME=DIR ... --teacher DIR [--output JSON]
                            the pilot's table and the candidate that goes on (exit 0: one meets the pilot's lines)
  gate smoke|validation --run DIR [--output JSON]            exit 0 when passed
  regression --run DIR --baseline DIR [--output JSON]        exit 0 when the checkpoint is not worse than the baseline

  --stage FILE              before the command: another stage's configuration (stage 1b: configs/generic_landing_stage1b.json);
                            its plans, its teacher, its pilot order and its gates are used. Stage 1b adds:
  repeat --run DIR --again DIR [--output JSON]               the second flights of the cube and cylinder starts (exit 0: stable)
  failures --run DIR [--output JSON]                         every failed landing named: how it ended, and where it stopped
  outputs --runs NAME=DIR ... [--output JSON]                how far the flown actions of the landings were from a stop
  ceiling --runs NAME=DIR ... [--output JSON]                flights above the start rule's ceiling

A run folder is one scripts/surface_flights.py wrote (results.json: per episode the canonical summary and the surface
evaluator's reading). Landings are read by the surface evaluator, approaches by the canonical evaluator.
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.visual_search import generic_landing as stage1
from src.visual_search.episodes import load_config
from src.aerovla_oft.spec import load_config as load_oft_config


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=1)+'\n',encoding='utf-8',newline='\n')


def build_all(config,oft,stage):
    """Every set in the order it is drawn: the checks first (the regression set, which has the least room, before the others),
    then the recorded episodes, which keep away from all of them."""
    points=stage1.earlier_points(*stage1.previous_files(stage));sets={};own=[]
    starts=lambda episodes:[(episode['map'],episode['start_xy']) for episode in episodes if not episode.get('twin_of')]
    if stage['regression'].get('may_be_near'):
        # A later stage's regression set replaces the one before it and need not keep away from that one's starts (never
        # recorded): the row with a pad of the other colour close by in the first view has no room left otherwise.
        sets['regression']=stage1.build_regression(config,oft,stage,loose_points(stage,'regression'))
        points+=starts(sets['regression'])
    else:sets['regression']=stage1.build_regression(config,oft,stage,points)
    own+=starts(sets['regression'])
    for name,build in (('pilot',stage1.build_pilot),('validation',stage1.build_validation)):
        sets[name]=build(config,oft,stage,points);own+=starts(sets[name])
    # The recorded episodes of a later stage take the place of the earlier stage's, which are not in its data, and need not
    # keep away from them; they keep away from everything else, this stage's own checks included.
    sets['train_plan']=stage1.build_training(config,oft,stage,loose_points(stage,'training')+own if stage['training'].get('may_be_near') else points)
    return sets


def loose_points(stage,which='regression'):
    """The earlier starts a set of a later stage keeps away from: all but the sets named in its `may_be_near`."""
    near={(ROOT/path).resolve() for path in stage[which].get('may_be_near',[])}
    return stage1.earlier_points(*[path for path in stage1.previous_files(stage) if Path(path).resolve() not in near])


def histogram(items,key):return dict(sorted(collections.Counter(key(item) for item in items).items(),key=lambda pair:str(pair[0])))


def describe_training(stage,plan):
    teacher={'surface':'SurfaceTeacher','stop_safe':'StopSafeTeacher'}[stage1.teacher_of(stage)];name=f', stage {stage["stage"]}' if stage.get('stage') else ''
    out={'description':f'Generic landing{name}: the plan of the recorded episodes. Landings on pads, cubes and cylinders from arbitrary starts, and approach twins of some of '
                       f'them. Recorded with src/landing/teacher.{teacher} in layouts earlier episodes were recorded in.',
         'seeds':{split:stage['seeds'][split] for split in ('train','val')}}
    for split,episodes in plan.items():
        out[f'{split}_summary']={'episodes':len(episodes),'tasks':histogram(episodes,lambda e:e['task']),'surface_kinds':histogram(episodes,lambda e:e['surface_kind']),
                                 'targets':histogram(episodes,lambda e:e['target']),'bands':histogram(episodes,lambda e:e['band']),
                                 'first_view':histogram(episodes,lambda e:'visible' if e['initially_visible'] else 'outside'),
                                 'scenes':histogram(episodes,lambda e:e['map']),'layouts':histogram(episodes,lambda e:e['layout']),
                                 'start_height_m':histogram(episodes,lambda e:f'{int(e["start_height_m"]//3*3)}-{int(e["start_height_m"]//3*3+3)}'),
                                 'needs_climb':sum(e['plan']['climbed_m']>=1. for e in episodes if e['task']=='land')}
        out[split]=episodes
    return out


def check(config,oft,stage):
    files=stage1.files_of(stage);errors=[];again=build_all(config,oft,stage);saved={name:json.loads(files[name].read_text(encoding='utf-8'))['episodes'] for name in files}
    plan=json.loads(stage1.train_plan_of(stage).read_text(encoding='utf-8'))
    for name in files:
        if again[name]!=saved[name]:errors.append(f'{name}: the file is not what the configuration gives')
    if any(again['train_plan'][split]!=plan[split] for split in ('train','val')):errors.append('train_plan: the file is not what the configuration gives')
    recorded=plan['train']+plan['val'];checks=[episode for name in files for episode in saved[name]]
    for episode in recorded:
        if episode['layout'] not in stage['layouts']['train']:errors.append(f'{episode["id"]}: recorded in an evaluation layout')
        if not stage['bands_m']['near'][0]-.01<=episode['start_distance_m']<=stage['bands_m']['far'][1]+.01:errors.append(f'{episode["id"]}: outside the range')
    for episode in checks:
        if episode['layout'] not in stage['layouts']['evaluation']:errors.append(f'{episode["id"]}: a check start in a layout episodes were recorded in')
    # Starts of one scene keep their distance, across all sets (a twin shares its start with the landing it is a twin of).
    firsts=[episode for episode in recorded+checks if not episode.get('twin_of')];earlier=stage1.earlier_points(*stage1.previous_files(stage))
    loose={name:loose_points(stage,key) for key,names in (('regression',('regression',)),('training',('train','val'))) if stage[key].get('may_be_near') for name in names}
    for index,a in enumerate(firsts):
        for b in firsts[index+1:]:
            if a['map']==b['map'] and math.dist(a['start_xy'],b['start_xy'])<stage['separation_m']:errors.append(f'{a["id"]} and {b["id"]} start too near each other')
        mine=loose.get(a.get('set'),earlier)
        if any(where==a['map'] and math.dist(a['start_xy'],xy)<stage['separation_m'] for where,xy in mine):errors.append(f'{a["id"]} starts too near an earlier start')
    seeds=collections.Counter((episode['map'],episode['seed']) for episode in firsts)
    errors+=[f'seed used twice: {key}' for key,count in seeds.items() if count>1]
    return errors


def flights(run,only=None):
    """The flown episodes of a run folder (scripts/surface_flights.py), in flying order."""
    path=Path(run) if Path(run).is_absolute() else ROOT/run
    records=[item for item in json.loads((path/'results.json').read_text(encoding='utf-8'))['episodes'] if item.get('flown')]
    return [item for item in records if only is None or item['id'] in only]


def measure(records):
    """What a set of flights did, counted the way the gates are written. A landing is read by the surface evaluator
    (system landing: the named object, inside its usable region, soft, standing, latched), an approach by the canonical one."""
    out={'flights':len(records),'success':0,'land':0,'land_success':0,'approach':0,'approach_success':0,'wrong_target':0,'collisions':0,'approach_touchdowns':0,
         'timeouts':0,'stopped_in_air':0,'kinds':{},'failed':[]}
    for item in records:
        episode=item['episode'];summary=item['summary'];reading=item['surface_landing'];task=episode['task'];good=bool(reading['success'])
        kind=episode.get('surface_kind') or ('pad' if episode['target'].endswith('_pad') else 'other')
        entry=out['kinds'].setdefault(kind,{'land':0,'land_success':0,'touchdown':0,'approach':0,'approach_success':0,'wrong_target':0,'collisions':0})
        out['success']+=good;out[task]+=1;out[f'{task}_success']+=good;entry[task]+=1;entry[f'{task}_success']+=good
        wrong=bool(summary.get('wrong_target'));hit=summary.get('hit') is not None or summary.get('reason')=='collision'
        out['wrong_target']+=wrong;out['collisions']+=hit;entry['wrong_target']+=wrong;entry['collisions']+=hit
        out['timeouts']+=summary.get('reason')=='max_steps'
        if task=='approach':out['approach_touchdowns']+=bool(summary.get('landed'))
        else:
            entry['touchdown']+=bool(reading['correct_object'] and reading['inside_region']);out['stopped_in_air']+=bool(summary.get('stopped') and not reading['contact_from_above'] and not hit)
        if not good:out['failed'].append({'id':item['id'],'task':task,'reason':summary.get('reason'),'landed_on':reading['landed_on'],'collision':reading['collision'],
                                          'wrong_target':wrong,'final_distance_m':round(summary.get('final_distance_m') or 0.,1),'steps':summary.get('steps')})
    return out


def failure_stage(item):
    """Where a failed landing stopped being a landing: the first of these that did not happen.
    search (the object never came into view) -> selection (ended at another object) -> approach (never within the arrival
    distance) -> climb (never higher than the top) -> alignment (never over the usable region) -> descent (over it, no
    contact) -> touchdown (a contact that was not a stable, latched landing inside the region). A collision is its own kind."""
    summary=item['summary'];reading=item['surface_landing'];stages=summary.get('stages',{})
    if reading['collision'] or summary.get('reason')=='collision':return 'collision'
    if summary.get('wrong_target'):return 'selection'
    if not stages.get('target_acquired'):return 'search'
    if summary.get('minimum_distance_m',1e9)>12.:return 'approach'
    surface=reading['target_surface']
    if surface.get('landable') and summary.get('height_range_m',[0,0])[1]<surface['top_m']+.5:return 'climb'
    if not reading.get('decisions_over_usable_region'):return 'alignment'
    if not reading['contact_from_above']:return 'descent'
    return 'touchdown'


def table(records):
    """Rows of the validation table, one per kind of surface."""
    numbers=measure(records);rows=[]
    for kind in stage1.KINDS:
        entry=numbers['kinds'].get(kind,{});mine=[item for item in records if item['episode'].get('surface_kind')==kind and item['episode']['task']=='land']
        failed=[item for item in mine if not item['surface_landing']['success']]
        errors=sorted(item['surface_landing']['horizontal_error_m'] for item in mine if item['surface_landing']['success'])
        rows.append({'kind':kind,'landings_flown':entry.get('land',0),'system_landing':entry.get('land_success',0),'touchdown_in_region':entry.get('touchdown',0),
                     'wrong_target':entry.get('wrong_target',0),'collisions':entry.get('collisions',0),'approach_flown':entry.get('approach',0),
                     'approach_success':entry.get('approach_success',0),'median_touchdown_error_m':round(errors[len(errors)//2],2) if errors else None,
                     'failure_stages':histogram(failed,failure_stage),'failed':[{'id':item['id'],'stage':failure_stage(item),'reason':item['summary']['reason'],
                                                                              'final_distance_m':round(item['summary']['final_distance_m'],1)} for item in failed]})
    return {'numbers':{key:value for key,value in numbers.items() if key not in ('kinds','failed')},'rows':rows,'failed':numbers['failed']}


def lines(checks):
    passed=all(ok for _,_,_,ok in checks)
    for name,value,limit,ok in checks:print(f'  {"PASS" if ok else "FAIL"}  {name}: {value} ({limit})')
    return passed


def judge_validation(stage,which,records):
    numbers=measure(records);kinds=numbers['kinds']
    if which=='smoke':
        rule=stage['validation']['smoke']
        checks=[('flights',numbers['flights'],f'= {rule["of"]}',numbers['flights']==rule['of']),('success',numbers['success'],f'>= {rule["success_min"]}',numbers['success']>=rule['success_min']),
                ('collisions',numbers['collisions'],f'<= {rule["collisions_max"]}',numbers['collisions']<=rule['collisions_max'])]
    else:
        gates=stage['validation']['gates'];land=lambda kind:kinds.get(kind,{}).get('land_success',0)
        checks=[('flights',numbers['flights'],f'= {gates["of"]}',numbers['flights']==gates['of']),
                ('mission success',numbers['success'],f'>= {gates["success_min"]} of {gates["of"]}',numbers['success']>=gates['success_min']),
                ('system landings',numbers['land_success'],f'>= {gates["land_system_success_min"]} of {gates["land_of"]}',numbers['land_success']>=gates['land_system_success_min']),
                ('pad landings',land('pad'),f'>= {gates["pad_land_min"]} of {gates["per_kind_of"]}',land('pad')>=gates['pad_land_min']),
                ('cube landings',land('cube'),f'>= {gates["cube_land_min"]} of {gates["per_kind_of"]}',land('cube')>=gates['cube_land_min']),
                ('cylinder landings',land('cylinder'),f'>= {gates["cylinder_land_min"]} of {gates["per_kind_of"]}',land('cylinder')>=gates['cylinder_land_min']),
                ('flights at a wrong object',numbers['wrong_target'],f'<= {gates["wrong_target_max"]}',numbers['wrong_target']<=gates['wrong_target_max']),
                ('collisions',numbers['collisions'],f'<= {gates["collisions_max"]}',numbers['collisions']<=gates['collisions_max']),
                ('touchdowns in an approach',numbers['approach_touchdowns'],f'<= {gates["approach_touchdowns_max"]}',numbers['approach_touchdowns']<=gates['approach_touchdowns_max'])]
    return numbers,checks


def judge_pilot(stage,runs,losses):
    """The candidates' pilot flights by the fixed order, and whether the first of them meets the pilot's lines."""
    rows=[]
    for name,records in runs.items():
        numbers=measure(records);kinds=numbers['kinds'];tall=kinds.get('cube',{}).get('land_success',0)+kinds.get('cylinder',{}).get('land_success',0)
        rows.append({'candidate':name,'flights':numbers['flights'],'cube_cylinder_landings':tall,'pad_landings':kinds.get('pad',{}).get('land_success',0),
                     'cube_landings':kinds.get('cube',{}).get('land_success',0),'cylinder_landings':kinds.get('cylinder',{}).get('land_success',0),
                     'air_self_stops':numbers['stopped_in_air'],
                     'wrong_target':numbers['wrong_target'],'collisions':numbers['collisions'],'approach_touchdowns':numbers['approach_touchdowns'],
                     'approach_success':numbers['approach_success'],'success':numbers['success'],'val_l1':losses.get(name),'failed':numbers['failed']})
    candidates=[row for row in rows if row['candidate'].isdigit()];loss=lambda row:row['val_l1'] if row['val_l1'] is not None else float('inf')
    if stage.get('stage'):
        # Stage 1b: landings on cubes and cylinders, then the flights that stopped by themselves in the air, pads, wrong objects, collisions.
        ranked=sorted(candidates,key=lambda row:(-row['cube_cylinder_landings'],row['air_self_stops'],-row['pad_landings'],row['wrong_target'],row['collisions'],
                                                 loss(row),int(row['candidate'])))
    else:
        ranked=sorted(candidates,key=lambda row:(-row['cube_cylinder_landings'],row['wrong_target'],-row['pad_landings'],row['collisions'],loss(row),int(row['candidate'])))
    rule=stage['pilot']['pass'];chosen=ranked[0] if ranked else None;checks=[]
    if chosen:
        checks=[('landings on cubes and cylinders',chosen['cube_cylinder_landings'],f'>= {rule["cube_cylinder_landings_min"]} of {rule["cube_cylinder_landings_of"]}',chosen['cube_cylinder_landings']>=rule['cube_cylinder_landings_min']),
                ('landings on pads',chosen['pad_landings'],f'>= {rule["pad_landings_min"]} of {rule["pad_landings_of"]}',chosen['pad_landings']>=rule['pad_landings_min']),
                ('touchdowns in an approach',chosen['approach_touchdowns'],f'<= {rule["approach_touchdowns_max"]}',chosen['approach_touchdowns']<=rule['approach_touchdowns_max']),
                ('collisions',chosen['collisions'],f'<= {rule["collisions_max"]}',chosen['collisions']<=rule['collisions_max'])]
    return rows,[row['candidate'] for row in ranked],checks


def judge_regression(stage,model,baseline):
    rule=stage['regression'];mine=measure(model);base=measure(baseline)
    checks=[('flights',mine['flights'],f'= {base["flights"]}',mine['flights']==base['flights']),
            ('mission success',mine['success'],f'>= baseline {base["success"]} - {rule["success_below_baseline_max"]}',mine['success']>=base['success']-rule['success_below_baseline_max']),
            ('landings',mine['land_success'],f'>= baseline {base["land_success"]} - {rule["landings_below_baseline_max"]}',mine['land_success']>=base['land_success']-rule['landings_below_baseline_max']),
            ('flights at a wrong object',mine['wrong_target'],f'<= baseline {base["wrong_target"]} + {rule["wrong_target_above_baseline_max"]}',mine['wrong_target']<=base['wrong_target']+rule['wrong_target_above_baseline_max']),
            ('collisions',mine['collisions'],f'<= baseline {base["collisions"]} + {rule["collisions_above_baseline_max"]}',mine['collisions']<=base['collisions']+rule['collisions_above_baseline_max'])]
    if 'approach_touchdowns_max' in rule:
        checks.append(('touchdowns in an approach',mine['approach_touchdowns'],f'<= {rule["approach_touchdowns_max"]}',mine['approach_touchdowns']<=rule['approach_touchdowns_max']))
    return {'model':mine,'baseline':base},checks


def judge_repeat(stage,first,second):
    """Stage 1b: the cube and cylinder landing starts flown twice by one checkpoint. Per kind the first flights, the two flights
    together, and the starts that failed both times."""
    rule=stage['validation']['repeat'];checks=[];table={}
    for kind in rule['kinds']:
        landed=lambda records:{item['id']:bool(item['surface_landing']['success']) for item in records
                               if item['episode'].get('surface_kind')==kind and item['episode']['task']=='land'}
        one,two=landed(first),landed(second);ids=sorted(one);twice=[name for name in ids if name in two]
        both=[name for name in twice if one[name] and two[name]];neither=[name for name in twice if not one[name] and not two[name]]
        mixed=[name for name in twice if one[name]!=two[name]];total=sum(one[name] for name in ids)+sum(two[name] for name in twice)
        table[kind]={'starts':len(ids),'flown_twice':len(twice),'first_flights':sum(one.values()),'second_flights':sum(two[name] for name in twice),'both':len(both),'one_of_two':len(mixed),
                     'neither':len(neither),'aggregate':total,'starts_that_failed_both_times':neither,'starts_that_flipped':mixed}
        checks+=[(f'{kind} starts flown twice',len(twice),f'= {rule["first_flight_of"]}',len(ids)==len(twice)==rule['first_flight_of']),
                 (f'{kind} landings, first flights',sum(one.values()),f'>= {rule["first_flight_min"]} of {rule["first_flight_of"]}',sum(one.values())>=rule['first_flight_min']),
                 (f'{kind} landings, both flights',total,f'>= {rule["aggregate_min"]} of {rule["aggregate_of"]}',total>=rule['aggregate_min']),
                 (f'{kind} starts that failed both times',len(neither),f'<= {rule["both_failed_max"]}',len(neither)<=rule['both_failed_max'])]
    return table,checks


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('plan');commands.add_parser('check')
    ids=commands.add_parser('ids');ids.add_argument('which',choices=('smoke','critical','repeat'))
    pilot=commands.add_parser('pilot');pilot.add_argument('--runs',nargs='+',required=True,help='NAME=DIR; a candidate is named by its update');pilot.add_argument('--output')
    pilot.add_argument('--training',help='the training run folder, for the validation L1 of each candidate')
    gate=commands.add_parser('gate');gate.add_argument('which',choices=('smoke','validation'));gate.add_argument('--run',required=True);gate.add_argument('--output')
    regression=commands.add_parser('regression');regression.add_argument('--run',required=True);regression.add_argument('--baseline',required=True);regression.add_argument('--output')
    report=commands.add_parser('table');report.add_argument('--run',required=True);report.add_argument('--output')
    again=commands.add_parser('repeat');again.add_argument('--run',required=True);again.add_argument('--again',required=True);again.add_argument('--output')
    failed=commands.add_parser('failures');failed.add_argument('--run',required=True);failed.add_argument('--output')
    for name in ('outputs','ceiling'):
        other=commands.add_parser(name);other.add_argument('--runs',nargs='+',required=True,help='NAME=DIR');other.add_argument('--output')
    parser.add_argument('--stage',help='the configuration of a later stage (default: stage 1, configs/generic_landing.json)')
    args=parser.parse_args();config=load_config();oft=load_oft_config();stage=stage1.load(args.stage);files=stage1.files_of(stage);train_plan=stage1.train_plan_of(stage)
    if args.stage:stage['file']=str(Path(args.stage).as_posix())
    where=lambda path:Path(path) if Path(path).is_absolute() else ROOT/path
    if args.command in ('failures','outputs','ceiling'):
        from src.landing import failures as reading
        from src.visual_search.maps import load_landing,load_map
        landing=load_landing()
        if args.command=='failures':
            result=reading.failures(flights(args.run),reading.steps_of(where(args.run)),landing,config,oft)
            print(json.dumps({key:value for key,value in result.items() if key!='flights'}))
            for row in result['flights']:print(json.dumps({key:value for key,value in row.items() if key!='self_stop'}))
        elif args.command=='outputs':
            result={name:reading.output_margins(flights(folder),reading.steps_of(where(folder)),landing,config,oft) for name,folder in (item.split('=',1) for item in args.runs)}
            for name,entry in result.items():
                print(name,json.dumps(entry['flights']))
                for phase,row in entry['by_phase'].items():print('  ',phase,json.dumps(row))
        else:
            limit=load_map(stage['scenes'][0])['altitude']['ceiling_m']
            result={name:reading.ceiling(flights(folder),limit) for name,folder in (item.split('=',1) for item in args.runs)}
            for name,entry in result.items():print(name,json.dumps(entry))
        if args.output:write(where(args.output),result)
        return
    if args.command=='repeat':
        pairs,checks=judge_repeat(stage,flights(args.run),flights(args.again))
        for kind,row in pairs.items():print(kind,json.dumps(row))
        passed=lines(checks);result={'table':pairs,'checks':checks,'passed':passed}
        if args.output:write(where(args.output),result)
        print('PASSED' if passed else 'NOT PASSED');raise SystemExit(0 if passed else 1)
    if args.command=='table':
        result=table(flights(args.run))
        for row in result['rows']:print(json.dumps({key:value for key,value in row.items() if key!='failed'}))
        print(json.dumps(result['numbers']))
        if args.output:write(Path(args.output) if Path(args.output).is_absolute() else ROOT/args.output,result)
        return
    if args.command in ('pilot','gate','regression'):
        if args.command=='pilot':
            runs={name:flights(folder) for name,folder in (item.split('=',1) for item in args.runs)};losses={}
            for name in runs:
                manifest=(Path(args.training) if args.training else ROOT/stage['run'])/f'update-{name}'/'manifest.json'
                if manifest.exists():losses[name]=json.loads(manifest.read_text(encoding='utf-8'))['candidate']['val_l1']
            rows,ranked,checks=judge_pilot(stage,runs,losses)
            for row in rows:print(json.dumps({key:value for key,value in row.items() if key!='failed'}))
            print('order:',ranked);passed=bool(checks) and lines(checks);result={'rows':rows,'order':ranked,'chosen':ranked[0] if ranked else None,'checks':checks,'passed':passed}
        elif args.command=='gate':
            smoke=None
            if args.which=='smoke':
                count=stage['validation']['smoke']['starts_per_kind'];episodes=json.loads(files['validation'].read_text(encoding='utf-8'))['episodes'];smoke=[]
                for kind in stage1.KINDS:smoke+=[e['id'] for e in episodes if e.get('surface_kind')==kind and e['task']=='land'][:count]
            numbers,checks=judge_validation(stage,args.which,flights(args.run,smoke));passed=lines(checks);result={'which':args.which,'numbers':numbers,'checks':checks,'passed':passed}
        else:
            numbers,checks=judge_regression(stage,flights(args.run),flights(args.baseline));passed=lines(checks);result={'numbers':numbers,'checks':checks,'passed':passed}
        if args.output:write(Path(args.output) if Path(args.output).is_absolute() else ROOT/args.output,result)
        print('PASSED' if passed else 'NOT PASSED');raise SystemExit(0 if passed else 1)
    if args.command=='plan':
        if any(path.exists() for path in list(files.values())+[train_plan]):raise SystemExit('the plans exist; they are written once')
        sets=build_all(config,oft,stage)
        for name in files:write(files[name],stage1.describe(stage,name,sets[name]))
        write(train_plan,describe_training(stage,sets['train_plan']))
        for name in files:print(name,len(sets[name]),histogram(sets[name],lambda e:(e.get('surface_kind') or e.get('role'),e['task'])))
        described=describe_training(stage,sets['train_plan'])
        for split in ('train','val'):print(split,json.dumps(described[f'{split}_summary']))
    elif args.command=='check':
        errors=check(config,oft,stage)
        for text in errors:print('ERROR: '+text)
        if errors:raise SystemExit(1)
        print('OK: the plans are what the configuration gives; recorded starts are in training layouts, check starts in evaluation layouts, and all keep their distance')
    elif args.command=='ids':
        episodes=json.loads(files['validation'].read_text(encoding='utf-8'))['episodes']
        if args.which=='critical':print(json.dumps(stage1.critical_ids(episodes)))
        elif args.which=='repeat':
            # Stage 1b: the landing starts on the kinds of surface whose second flights are part of the gate.
            print(json.dumps([e['id'] for e in episodes if e['task']=='land' and e.get('surface_kind') in stage['validation']['repeat']['kinds']]))
        else:
            count=stage['validation']['smoke']['starts_per_kind'];chosen=[]
            for kind in stage1.KINDS:chosen+=[e['id'] for e in episodes if e.get('surface_kind')==kind and e['task']=='land'][:count]
            print(json.dumps(chosen))


if __name__=='__main__':main()
