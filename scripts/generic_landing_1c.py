"""Phase B, stage 1c (cube descent recovery and same-colour landing grounding): plans, gates and tables; no simulator, no model.

  plan                      write configs/generic_landing_stage1c/*.json once
  check                     confirm the plans are what the configuration gives and that the starts keep the distances they must
  ids smoke|repeat|same-color-repeat|forced
                            the starts of gate A flown first; its cube and cylinder landing starts (flown twice); the twelve
                            starts of gate B flown twice; the forced sixteen of the teacher pilot
  inputs DATASET            what a recorded dataset gives a student: sentences only from the catalogue, no number, no object
                            id, proprio off; and that every twin shares its start with its landing and names another object
  teacher-pilot --run DIR   the teacher's thirty-two flights before the recording (exit 0: the pilot's lines are met)
  pilot --runs stage1b=DIR 500=DIR ...
                            the candidates' table and the one that goes on (exit 0: it meets the pilot's lines)
  gate smoke|a|b --run DIR  gate A (and its smoke), gate B
  repeat a|b --run DIR --again DIR
                            the second flights of gate A's cube and cylinder starts; of gate B's twelve
  regression --run DIR --baseline DIR [--stage1b DIR]
  cube --runs NAME=DIR ...  how cube landings went wrong, counted from their descent profiles
  descent --runs NAME=DIR ...   the descent of every cube landing: where and how high it began, sign changes, touchdown
  commitment --runs NAME=DIR ...  every same-colour flight read against the other object
  decision --audit JSON --inputs JSON --a JSON --repeat-a JSON --b JSON --repeat-b JSON --regression JSON

Every command takes --output JSON. A run folder is one scripts/surface_flights.py wrote. Landings are read by the surface
evaluator, approaches by the canonical one; the gates' numbers are the stage's configuration.
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.generic_landing as base
from src.landing import failures as reading
from src.landing.teacher import FORCED
from src.visual_search import generic_landing as stages
from src.visual_search import generic_landing_1c as plans
from src.visual_search.episodes import load_config
from src.aerovla_oft.spec import load_config as load_oft_config

flights=base.flights;measure=base.measure;lines=base.lines;write=base.write
where=lambda path:Path(path) if Path(path).is_absolute() else ROOT/path


def describe_training(stage,plan):
    out={'description':'Generic landing, stage 1c: the plan of the recorded episodes. Forced landings (recovery) and same-colour landings with their twins, recorded with '
                       'src/landing/teacher.RecoveryTeacher in layouts earlier episodes were recorded in.',
         'seeds':{name:stage['seeds'][name] for name in ('train_recovery','train_same_color','val_recovery','val_same_color')}}
    for split,episodes in plan.items():
        count=lambda key:dict(sorted(collections.Counter(key(e) for e in episodes).items(),key=lambda pair:str(pair[0])))
        out[f'{split}_summary']={'episodes':len(episodes),'recovery':count(lambda e:e.get('recovery') or '-'),'surface_kinds':count(lambda e:e['surface_kind']),'targets':count(lambda e:e['target']),
                                 'same_colour':count(lambda e:f'{e["target"]} / {e["distractor"]}'+(' (twin)' if e.get('twin_of') else '') if e.get('distractor') else '-'),
                                 'twin_pairs':sum(bool(e.get('twin_of')) for e in episodes),'bands':count(lambda e:e['band']),'scenes':count(lambda e:e['map']),'layouts':count(lambda e:e['layout'])}
        out[split]=episodes
    return out


def check(config,oft,stage):
    """The files are what the configuration gives; recorded starts are in recording layouts and check starts in evaluation
    layouts; no check start is within the separation of a recorded start of the data or of another check start of this
    stage; no seed is used twice."""
    files=plans.files_of(stage);errors=[];again=plans.build_all(config,oft,stage)
    saved={name:json.loads(files[name].read_text(encoding='utf-8'))['episodes'] for name in files};plan=json.loads(plans.train_plan_of(stage).read_text(encoding='utf-8'))
    for name in files:
        if again[name]!=saved[name]:errors.append(f'{name}: the file is not what the configuration gives')
    if any(again['train_plan'][split]!=plan[split] for split in ('train','val')):errors.append('train_plan: the file is not what the configuration gives')
    recorded=plan['train']+plan['val'];checks=[episode for name in files for episode in saved[name]];radius=stage['separation_m']
    for episode in recorded:
        if episode['layout'] not in stage['layouts']['train']:errors.append(f'{episode["id"]}: recorded in an evaluation layout')
    for episode in checks:
        if episode['layout'] not in stage['layouts']['evaluation']:errors.append(f'{episode["id"]}: a check start in a layout episodes were recorded in')
    near=lambda a,b:a['map']==b['map'] and math.dist(a['start_xy'],b['start_xy'])<radius
    first=lambda episodes:[e for e in episodes if not e.get('twin_of')]
    # A recorded start and a check start never stand near each other; the check starts of the gates and the pilots keep
    # away from each other (the regression set is drawn first and the others keep away from it).
    for a in first(recorded):
        for b in first(checks):
            if near(a,b):errors.append(f'{a["id"]} (recorded) and {b["id"]} (check) start too near each other')
    mine=first(checks)
    for index,a in enumerate(mine):
        for b in mine[index+1:]:
            if near(a,b):errors.append(f'{a["id"]} and {b["id"]} start too near each other')
    flown=plans.check_points(stage)
    for a in first(recorded):
        if any(name==a['map'] and math.dist(a['start_xy'],xy)<radius for name,xy in flown):errors.append(f'{a["id"]} (recorded) starts too near an earlier check start')
    spots=collections.Counter((e['map'],tuple(e['start_xy'])) for e in first(recorded)+mine);errors+=[f'start used twice: {key}' for key,count in spots.items() if count>1]
    seeds=collections.Counter((e['map'],e['seed'],e.get('set')) for e in first(recorded)+mine);errors+=[f'seed used twice: {key}' for key,count in seeds.items() if count>1]
    return errors


def inputs_audit(root,stage,config):
    """What a recorded dataset would give a student beside the two pictures: the sentence. Every sentence has to be one the
    catalogue builds for an object and a task, with no number and no object id in it; proprio is off; a twin shares its
    start with the landing it is a twin of and names another object."""
    from src.visual_search.generalization import instruction_for_id
    from src.visual_search.maps import load_map
    allowed=set()
    for name in stage['scenes']:
        scene=load_map(name)
        for target in scene['objects']:
            for task in stage['instructions'].values():
                try:allowed.add(instruction_for_id(config,scene,target,task))
                except Exception:pass
    records=[json.loads(path.read_text(encoding='utf-8')) for path in sorted((where(root)/'episodes').glob('*.json'))]
    summaries={record['summary']['id']:record['summary'] for record in records};sentences=collections.Counter(summary['instruction'] for summary in summaries.values())
    ids={target for name in stage['scenes'] for target in load_map(name)['objects']}
    bad=[text for text in sentences if text not in allowed or any(letter.isdigit() for letter in text) or any(name in text for name in ids)]
    twins=[summary for summary in summaries.values() if summary.get('twin_of') and summary.get('distractor')];broken=[]
    for twin in twins:
        first=summaries.get(twin['twin_of'])
        if first is None:continue    # the landing was not kept; the twin stands alone
        same=(twin['map'],twin['layout'],twin['start_xy'],twin['start_yaw_deg'],twin['start_height_m'])==(first['map'],first['layout'],first['start_xy'],first['start_yaw_deg'],first['start_height_m'])
        if not same or twin['target']==first['target'] or twin['instruction']==first['instruction'] or twin['target']!=first['distractor']:broken.append(twin['id'])
    oft=load_oft_config();step_keys=collections.Counter(key for record in records[:20] for step in record['steps'][:3] for key in step)
    result={'what':str(root),'episodes':len(records),'sentences':len(sentences),'sentences_not_from_the_catalogue_or_with_a_number_or_an_id':bad,
            'proprio_enabled':bool(oft['proprio']['enabled']),'twins':len(twins),'twins_with_their_landing':sum(twin['twin_of'] in summaries for twin in twins),'twins_that_do_not_fit':broken,
            'what_reaches_the_student':'the front and down pictures (one mosaic) and the sentence; the recorded steps also hold what the teacher and the evaluator were told '
                                       f'({", ".join(sorted(key for key in step_keys if key in ("bearing_deg","distance_m","position","front_seen","down_seen","over")))}), which the '
                                       'dataset builder keeps as analysis fields and the model never reads',
            'examples':sorted(sentences)[:12]}
    result['passed']=not bad and not result['proprio_enabled'] and not broken
    return result


def search_check(steps_by_flight,oft):
    """Decisions of teacher flights at which no camera showed the named object and the vehicle was not over it, and the
    teacher's label was something else than the canonical search without a target (turn right, or climb when blocked)."""
    turn=oft['action_bounds']['yaw_rad'][1];up=oft['action_bounds']['down_m'][0];blind=0;other=[]
    for name,steps in steps_by_flight.items():
        for step in steps:
            if step.get('perturbed') or step['front_seen'] or step['down_seen'] or step.get('over') or step.get('landed'):continue
            blind+=1;label=[round(value,4) for value in step['teacher_action']]
            if label not in ([0.,0.,round(turn,4)],[0.,round(up,4),0.]) and len(other)<10:other.append({'id':name,'step':step['step'],'label':label,'state':step.get('teacher_state')})
            elif label not in ([0.,0.,round(turn,4)],[0.,round(up,4),0.]):other.append(None)
    return {'decisions_without_the_target_in_view':blind,'labels_that_are_not_the_blind_search':len(other),'examples':[item for item in other if item]}


def judge_teacher_pilot(stage,records,steps,oft,config):
    from src.landing import stop_safety
    rule=stage['teacher_pilot']['pass'];forced=[item for item in records if item['episode'].get('recovery')];same=[item for item in records if item['episode'].get('distractor')]
    good=lambda item:bool(item['surface_landing']['success']);numbers=measure(records)
    pushed=[item for item in forced if item.get('forced',{}).get('all_came_to_pass_and_seen')]
    episodes=[{'id':item['id'],'task':item['episode']['task'],'kind':item['episode'].get('surface_kind'),'start_distance_m':item['episode'].get('start_distance_m',-1.),'steps':steps[item['id']]} for item in records]
    labels=stop_safety.audit(episodes,oft,{key:stage['stop_safety'][key] for key in ('safe_ratio','near_stop_ratio')},config['stop_ticks'])
    broken=sum(check['count'] for check in labels['checks']);blind=search_check({item['id']:steps[item['id']] for item in records},oft)
    recovered=sum(good(item) and bool(item.get('forced',{}).get('all_came_to_pass_and_seen')) for item in forced)
    checks=[('forced landings recovered (the push came to pass, the object was in view where it let go, the teacher landed)',recovered,f'>= {rule["recovery_min"]} of {rule["recovery_of"]}',
             recovered>=rule['recovery_min'] and len(forced)==rule['recovery_of']),
            ('same-colour starts landed on the named object',sum(good(item) for item in same),f'>= {rule["same_color_min"]} of {rule["same_color_of"]}',sum(good(item) for item in same)>=rule['same_color_min'] and len(same)==rule['same_color_of']),
            ('collisions',numbers['collisions'],f'<= {rule["collisions_max"]}',numbers['collisions']<=rule['collisions_max']),
            ('flights at a wrong object',numbers['wrong_target'],f'<= {rule["wrong_target_max"]}',numbers['wrong_target']<=rule['wrong_target_max']),
            ('labels that break the stop rule',broken,f'<= {rule["stop_safety_violations_max"]}',broken<=rule['stop_safety_violations_max']),
            ('labels of the search that are not the blind search',blind['labels_that_are_not_the_blind_search'],f'<= {rule["search_that_depends_on_the_target_max"]}',
             blind['labels_that_are_not_the_blind_search']<=rule['search_that_depends_on_the_target_max'])]
    by_state={state:{'flown':sum(item['episode']['recovery']==state for item in forced),'pushed_and_seen':sum(item['episode']['recovery']==state for item in pushed),
                     'recovered':sum(item['episode']['recovery']==state and good(item) for item in pushed)} for state in FORCED}
    return {'numbers':{key:numbers[key] for key in ('flights','success','wrong_target','collisions')},'recovery':by_state,'same_colour':{'flown':len(same),'landed_on_the_named_object':sum(good(item) for item in same),
            'twins':sum(bool(item['episode'].get('twin_of')) for item in same)},'labels':{'checks':labels['checks'],'by_task':labels['by']['task']},'search':blind},checks


def pilot_row(stage,name,records,loss):
    numbers=measure(records);land=[item for item in records if item['episode']['task']=='land'];good=lambda item:bool(item['surface_landing']['success'])
    kind=lambda item:item['episode'].get('surface_kind');same=[item for item in land if item['episode'].get('distractor')];forced=[item for item in land if item['episode'].get('recovery')]
    hit=lambda item:bool(item['surface_landing']['collision'] or item['summary'].get('hit') is not None or item['summary'].get('reason')=='collision')
    wrong=lambda item:bool(item['surface_landing'].get('landed_on')) and item['surface_landing']['landed_on']!=item['episode']['target']
    return {'candidate':name,'flights':len(records),'same_color_collisions':sum(hit(item) for item in same),'wrong_target_landings':sum(wrong(item) for item in land),
            'recovered':sum(good(item) for item in forced),'recovery_of':len(forced),'recovered_by_state':{state:sum(good(item) for item in forced if item['episode']['recovery']==state) for state in FORCED},
            # For reading the table, not for the order: in how many of the forced landings the push came to pass (a flight that never reaches the place is not pushed).
            'pushed':sum(item.get('forced',{}).get('came_to_pass',0)>=1 for item in forced),'pushed_and_recovered':sum(good(item) and item.get('forced',{}).get('came_to_pass',0)>=1 for item in forced),
            'cube_landings':sum(good(item) for item in land if kind(item)=='cube'),'cube_of':sum(kind(item)=='cube' for item in land),
            'cylinder_landings':sum(good(item) for item in land if kind(item)=='cylinder'),'cylinder_of':sum(kind(item)=='cylinder' for item in land),
            'pad_landings':sum(good(item) for item in land if kind(item)=='pad'),'pad_of':sum(kind(item)=='pad' for item in land),
            'same_color_landings':sum(good(item) for item in same),'same_color_of':len(same),'air_self_stops':numbers['stopped_in_air'],'collisions':numbers['collisions'],
            'wrong_target':numbers['wrong_target'],'success':numbers['success'],'val_l1':loss,'failed':numbers['failed']}


def judge_pilot(stage,runs,losses):
    """The table of the pilot, the candidates in their fixed order (a candidate that meets one of `put_behind` after every
    candidate that meets none), and whether the first meets the pilot's lines."""
    rows=[pilot_row(stage,name,records,losses.get(name)) for name,records in runs.items()];before=next((row for row in rows if row['candidate']=='stage1b'),None)
    behind=stage['pilot']['put_behind'];candidates=[row for row in rows if row['candidate'].isdigit()]
    for row in candidates:
        reasons=[]
        if row['same_color_collisions']>=behind['same_color_collisions_min']:reasons.append('a collision in a same-colour start')
        if before:
            if row['pad_landings']<=before['pad_landings']-behind['pad_landings_below_stage_1b_by']:reasons.append('pad landings below stage 1b')
            if row['cylinder_landings']<=before['cylinder_landings']-behind['cylinder_landings_below_stage_1b_by']:reasons.append('cylinder landings below stage 1b')
            if row['air_self_stops']>=before['air_self_stops']+behind['air_self_stops_above_stage_1b_by']:reasons.append('stops in the air above stage 1b')
            if row['recovered']<=before['recovered']-behind['recovered_below_stage_1b_by']:reasons.append('forced landings recovered below stage 1b')
        row['put_behind']=reasons
    loss=lambda row:row['val_l1'] if row['val_l1'] is not None else float('inf')
    ranked=sorted(candidates,key=lambda row:(bool(row['put_behind']),row['same_color_collisions'],row['wrong_target_landings'],-row['recovered'],-row['cube_landings'],-row['cylinder_landings'],
                                             -row['pad_landings'],row['air_self_stops'],loss(row),int(row['candidate'])))
    rule=stage['pilot']['pass'];chosen=ranked[0] if ranked else None;checks=[]
    if chosen:
        checks=[('collisions in the same-colour starts',chosen['same_color_collisions'],f'<= {rule["same_color_collisions_max"]}',chosen['same_color_collisions']<=rule['same_color_collisions_max']),
                ('landings on a wrong object',chosen['wrong_target_landings'],f'<= {rule["wrong_target_landings_max"]}',chosen['wrong_target_landings']<=rule['wrong_target_landings_max']),
                ('landings on cubes',chosen['cube_landings'],f'>= {rule["cube_landings_min"]} of {rule["cube_landings_of"]}',chosen['cube_landings']>=rule['cube_landings_min']),
                ('landings on pads',chosen['pad_landings'],f'>= {rule["pad_landings_min"]} of {rule["pad_landings_of"]}',chosen['pad_landings']>=rule['pad_landings_min']),
                ('landings on cylinders',chosen['cylinder_landings'],f'>= {rule["cylinder_landings_min"]} of {rule["cylinder_landings_of"]}',chosen['cylinder_landings']>=rule['cylinder_landings_min'])]
    return rows,[row['candidate'] for row in ranked],checks


def judge_repeat_a(stage,first,second):
    """The cube and cylinder landing starts of gate A flown twice: per kind the first flights, the two together (the cube's and
    the cylinder's own number), and the starts that failed both times."""
    rule=stage['validation']['repeat'];checks=[];table={}
    for kind in rule['kinds']:
        landed=lambda records:{item['id']:bool(item['surface_landing']['success']) for item in records if item['episode'].get('surface_kind')==kind and item['episode']['task']=='land'}
        one,two=landed(first),landed(second);ids=sorted(one);twice=[name for name in ids if name in two];total=sum(one[name] for name in ids)+sum(two[name] for name in twice)
        neither=[name for name in twice if not one[name] and not two[name]];mixed=[name for name in twice if one[name]!=two[name]];least=rule['aggregate_min'][kind]
        table[kind]={'starts':len(ids),'flown_twice':len(twice),'first_flights':sum(one.values()),'second_flights':sum(two[name] for name in twice),'both':sum(one[name] and two[name] for name in twice),
                     'one_of_two':len(mixed),'neither':len(neither),'aggregate':total,'starts_that_failed_both_times':neither,'starts_that_flipped':mixed}
        checks+=[(f'{kind} starts flown twice',len(twice),f'= {rule["first_flight_of"]}',len(ids)==len(twice)==rule['first_flight_of']),
                 (f'{kind} landings, first flights',sum(one.values()),f'>= {rule["first_flight_min"]} of {rule["first_flight_of"]}',sum(one.values())>=rule['first_flight_min']),
                 (f'{kind} landings, both flights',total,f'>= {least} of {rule["aggregate_of"]}',total>=least),
                 (f'{kind} starts that failed both times',len(neither),f'<= {rule["both_failed_max"]}',len(neither)<=rule['both_failed_max'])]
    return table,checks


def same_color_rows(config,records,steps):
    """Every same-colour flight of a run read against the other object."""
    from src.visual_search.maps import MapGeometry,load_map
    scenes={};rows=[]
    for item in records:
        episode=item['episode']
        if not episode.get('distractor') or episode['task']!='land':continue
        scene=scenes.setdefault(episode['map'],load_map(episode['map']));centre=MapGeometry(scene,episode['layout']).objects[episode['distractor']]['centre']
        row=reading.same_color_reading(item,steps[item['id']],centre,config['arrive_distance_m']);row.update(twin=bool(episode.get('twin_of')),other_first=bool(episode.get('distractor_first')))
        rows.append(row)
    return rows


def judge_b(stage,config,records,steps):
    gates=stage['same_color']['gates'];rows=same_color_rows(config,records,steps);count=lambda key:sum(bool(row[key]) for row in rows)
    hits=sum(bool(row['collision']) for row in rows)
    numbers={'flights':len(rows),'landed_on_the_named_object':count('success'),'wrong_target_landings':count('wrong_target_landing'),'collisions_with_the_other':count('collision_with_the_other'),
             'collisions':hits,'ended_at_the_wrong_object':count('wrong_terminal'),'approached_the_other':count('approached_the_other'),'rejected_the_other':count('rejected_the_other'),
             'stopped_at_the_other':count('stopped_at_the_other')}
    pairs=collections.defaultdict(lambda:[0,0])
    for row in rows:pairs[f'{row["target"]} past {row["other"]}'][0]+=row['success'];pairs[f'{row["target"]} past {row["other"]}'][1]+=1
    numbers['by_pair']={name:f'{good} / {total}' for name,(good,total) in sorted(pairs.items())}
    checks=[('flights',numbers['flights'],f'= {gates["of"]}',numbers['flights']==gates['of']),
            ('landings on the named object',numbers['landed_on_the_named_object'],f'>= {gates["correct_completion_min"]} of {gates["of"]}',numbers['landed_on_the_named_object']>=gates['correct_completion_min']),
            ('landings on a wrong object',numbers['wrong_target_landings'],f'<= {gates["wrong_target_landings_max"]}',numbers['wrong_target_landings']<=gates['wrong_target_landings_max']),
            ('collisions with the other object',numbers['collisions_with_the_other'],f'<= {gates["collisions_with_the_other_max"]}',numbers['collisions_with_the_other']<=gates['collisions_with_the_other_max']),
            ('collisions',numbers['collisions'],f'<= {gates["collisions_max"]}',numbers['collisions']<=gates['collisions_max']),
            ('flights that ended at the wrong object',numbers['ended_at_the_wrong_object'],f'<= {gates["wrong_terminal_max"]}',numbers['ended_at_the_wrong_object']<=gates['wrong_terminal_max']),
            ('the other object, first in view, rejected',numbers['rejected_the_other'],f'>= {gates["rejection_min"]} of {gates["of"]}',numbers['rejected_the_other']>=gates['rejection_min'])]
    return numbers,rows,checks


def judge_repeat_b(stage,config,first,second,first_steps,second_steps):
    rule=stage['same_color']['repeat'];one={row['id']:row for row in same_color_rows(config,first,first_steps)};two={row['id']:row for row in same_color_rows(config,second,second_steps)}
    ids=sorted(two);both=[name for name in ids if one[name]['success'] and two[name]['success']];neither=[name for name in ids if not one[name]['success'] and not two[name]['success']]
    mixed=[name for name in ids if one[name]['success']!=two[name]['success']];hits=sum(bool(two[name]['collision']) for name in ids)
    table={'starts':len(ids),'both':len(both),'one_of_two':len(mixed),'neither':len(neither),'collisions_in_the_second_flights':hits,
           'wrong_target_landings_in_the_second_flights':sum(two[name]['wrong_target_landing'] for name in ids),'starts_that_failed_both_times':neither,'starts_that_flipped':mixed,
           'second_flights':[two[name] for name in ids]}
    checks=[('starts flown twice',len(ids),f'= {rule["starts"]}',len(ids)==rule['starts'] and all(name in one for name in ids)),
            ('starts that failed both times',len(neither),f'<= {rule["both_failed_max"]}',len(neither)<=rule['both_failed_max']),
            ('collisions in the second flights',hits,f'<= {rule["collisions_max"]}',hits<=rule['collisions_max'])]
    return table,checks


def judge_regression(stage,model,baseline):
    rule=stage['regression'];mine=measure(model);base_numbers=measure(baseline)
    checks=[('flights',mine['flights'],f'= {base_numbers["flights"]}',mine['flights']==base_numbers['flights']),
            ('mission success',mine['success'],f'>= baseline {base_numbers["success"]} - {rule["success_below_baseline_max"]}',mine['success']>=base_numbers['success']-rule['success_below_baseline_max']),
            ('pad landings',mine['land_success'],f'>= baseline {base_numbers["land_success"]} - {rule["landings_below_baseline_max"]}',mine['land_success']>=base_numbers['land_success']-rule['landings_below_baseline_max']),
            ('flights at a wrong object',mine['wrong_target'],f'<= baseline {base_numbers["wrong_target"]} + {rule["wrong_target_above_baseline_max"]}',mine['wrong_target']<=base_numbers['wrong_target']+rule['wrong_target_above_baseline_max']),
            ('collisions',mine['collisions'],f'<= {rule["collisions_max"]}',mine['collisions']<=rule['collisions_max']),
            ('touchdowns in an approach',mine['approach_touchdowns'],f'<= {rule["approach_touchdowns_max"]}',mine['approach_touchdowns']<=rule['approach_touchdowns_max'])]
    return {'model':mine,'baseline':base_numbers},checks


def cube_profiles(records,steps,landing,config,oft,kind='cube'):
    return [reading.descent_profile(item,steps[item['id']],landing,config,oft) for item in records if item['episode']['task']=='land' and item['episode'].get('surface_kind')==kind]


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);parser.add_argument('--stage',default=str(plans.CONFIG))
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('plan','check'):commands.add_parser(name).add_argument('--output')
    one=commands.add_parser('ids');one.add_argument('which',choices=('smoke','repeat','same-color-repeat','forced'))
    one=commands.add_parser('inputs');one.add_argument('dataset');one.add_argument('--output')
    one=commands.add_parser('teacher-pilot');one.add_argument('--run',required=True);one.add_argument('--output')
    one=commands.add_parser('pilot');one.add_argument('--runs',nargs='+',required=True);one.add_argument('--output');one.add_argument('--training')
    one=commands.add_parser('gate');one.add_argument('which',choices=('smoke','a','b'));one.add_argument('--run',required=True);one.add_argument('--output')
    one=commands.add_parser('repeat');one.add_argument('which',choices=('a','b'));one.add_argument('--run',required=True);one.add_argument('--again',required=True);one.add_argument('--output')
    one=commands.add_parser('regression');one.add_argument('--run',required=True);one.add_argument('--baseline',required=True);one.add_argument('--stage1b');one.add_argument('--output')
    for name in ('cube','descent','commitment'):
        one=commands.add_parser(name);one.add_argument('--runs',nargs='+',required=True);one.add_argument('--output')
    one=commands.add_parser('decision');one.add_argument('--output')
    for name in ('audit','inputs','a','repeat-a','b','repeat-b','regression'):one.add_argument(f'--{name}',required=True)
    args=parser.parse_args();config=load_config();oft=load_oft_config();stage=plans.load(args.stage);files=plans.files_of(stage);result=None;passed=None
    named=lambda items:[tuple(item.split('=',1)) for item in items]
    if args.command=='plan':
        if any(path.exists() for path in list(files.values())+[plans.train_plan_of(stage)]):raise SystemExit('the plans exist; they are written once')
        sets=plans.build_all(config,oft,stage)
        for name in files:write(files[name],plans.describe(stage,name,sets[name]));print(name,len(sets[name]))
        described=describe_training(stage,sets['train_plan']);write(plans.train_plan_of(stage),described)
        for split in ('train','val'):print(split,json.dumps(described[f'{split}_summary']))
        return
    if args.command=='check':
        errors=check(config,oft,stage)
        for text in errors:print('ERROR: '+text)
        if errors:raise SystemExit(1)
        print('OK: the plans are what the configuration gives; recorded starts are in recording layouts, check starts in evaluation layouts; no recorded start is near a check start');return
    if args.command=='ids':
        load=lambda name:json.loads(files[name].read_text(encoding='utf-8'))['episodes']
        if args.which=='smoke':
            chosen=[]
            for kind in stages.KINDS:chosen+=[e['id'] for e in load('validation') if e.get('surface_kind')==kind and e['task']=='land'][:stage['validation']['smoke']['starts_per_kind']]
        elif args.which=='repeat':chosen=[e['id'] for e in load('validation') if e['task']=='land' and e.get('surface_kind') in stage['validation']['repeat']['kinds']]
        elif args.which=='same-color-repeat':chosen=plans.repeat_ids(stage,load('same_color'))
        else:chosen=[e['id'] for e in load('teacher_pilot') if e.get('recovery')]
        print(json.dumps(chosen));return
    if args.command=='inputs':
        result=inputs_audit(args.dataset,stage,config);passed=result['passed']
        print(json.dumps({key:value for key,value in result.items() if key not in ('examples','what_reaches_the_student')}))
    elif args.command=='teacher-pilot':
        numbers,checks=judge_teacher_pilot(stage,flights(args.run),reading.steps_of(where(args.run)),oft,config)
        print(json.dumps(numbers['recovery']));print(json.dumps(numbers['same_colour']));print(json.dumps(numbers['search']));passed=lines(checks);result={**numbers,'checks':checks,'passed':passed}
    elif args.command=='pilot':
        runs={name:flights(folder) for name,folder in named(args.runs)};losses={}
        for name in runs:
            manifest=(where(args.training) if args.training else ROOT/stage['run'])/f'update-{name}'/'manifest.json'
            if manifest.exists():losses[name]=json.loads(manifest.read_text(encoding='utf-8'))['candidate']['val_l1']
        rows,ranked,checks=judge_pilot(stage,runs,losses)
        for row in rows:print(json.dumps({key:value for key,value in row.items() if key!='failed'}))
        print('order:',ranked);passed=bool(checks) and lines(checks);result={'rows':rows,'order':ranked,'chosen':ranked[0] if ranked else None,'checks':checks,'passed':passed}
    elif args.command=='gate':
        if args.which=='b':
            numbers,rows,checks=judge_b(stage,config,flights(args.run),reading.steps_of(where(args.run)));print(json.dumps(numbers));passed=lines(checks)
            result={'which':'b','numbers':numbers,'flights':rows,'checks':checks,'passed':passed}
        else:
            smoke=None
            if args.which=='smoke':
                smoke=[]
                for kind in stages.KINDS:smoke+=[e['id'] for e in json.loads(files['validation'].read_text(encoding='utf-8'))['episodes'] if e.get('surface_kind')==kind and e['task']=='land'][:stage['validation']['smoke']['starts_per_kind']]
            numbers,checks=base.judge_validation(stage,'smoke' if args.which=='smoke' else 'validation',flights(args.run,smoke));passed=lines(checks)
            result={'which':args.which,'numbers':numbers,'checks':checks,'passed':passed}
    elif args.command=='repeat':
        if args.which=='a':
            table,checks=judge_repeat_a(stage,flights(args.run),flights(args.again))
            for kind,row in table.items():print(kind,json.dumps(row))
        else:
            table,checks=judge_repeat_b(stage,config,flights(args.run),flights(args.again),reading.steps_of(where(args.run)),reading.steps_of(where(args.again)))
            print(json.dumps({key:value for key,value in table.items() if key!='second_flights'}))
        passed=lines(checks);result={'table':table,'checks':checks,'passed':passed}
    elif args.command=='regression':
        numbers,checks=judge_regression(stage,flights(args.run),flights(args.baseline))
        if args.stage1b:numbers['stage_1b']=measure(flights(args.stage1b))
        for name,row in numbers.items():print(name,json.dumps({key:row[key] for key in ('flights','success','land_success','approach_success','wrong_target','collisions','approach_touchdowns','stopped_in_air')}))
        passed=lines(checks);result={'numbers':numbers,'checks':checks,'passed':passed}
    elif args.command in ('cube','descent'):
        from src.visual_search.maps import load_landing
        landing=load_landing();result={}
        for name,folder in named(args.runs):
            profiles=cube_profiles(flights(folder),reading.steps_of(where(folder)),landing,config,oft)
            if args.command=='cube':result[name]=reading.cube_counts(profiles,stage['teacher'],3.6);print(name,json.dumps(result[name]))
            else:
                result[name]=profiles
                for row in profiles:print(name,json.dumps(row))
    elif args.command=='commitment':
        result={}
        for name,folder in named(args.runs):
            result[name]=same_color_rows(config,flights(folder),reading.steps_of(where(folder)))
            for row in result[name]:print(name,json.dumps(row))
    elif args.command=='decision':
        load=lambda path:json.loads(where(path).read_text(encoding='utf-8'))
        parts={'the recorded landing labels keep the stop rule':load(args.audit)['passed'],'what the data gives a student is pictures and a sentence':load(args.inputs)['passed'],
               'gate A (generic landing)':load(args.a)['passed'],'gate A, second flights of the cube and cylinder starts':load(getattr(args,'repeat_a'))['passed'],
               'gate B (same colour)':load(args.b)['passed'],'gate B, second flights':load(getattr(args,'repeat_b'))['passed'],'regression':load(args.regression)['passed']}
        passed=all(parts.values());result={'parts':parts,'stage_1c':'PASSED' if passed else 'NOT PASSED','generic_landing_v1_exists':(ROOT/stage['checkpoint']).exists()}
        for name,good in parts.items():print(f'  {"PASS" if good else "FAIL"}  {name}')
    if args.output and result is not None:write(where(args.output),result)
    if passed is not None:print('PASSED' if passed else 'NOT PASSED');raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
