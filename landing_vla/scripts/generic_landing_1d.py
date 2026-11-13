"""Phase B, stage 1d (final descent continuation): plans, audits, the pilot, the qualification and the tables; no simulator, no model.

  plan                      write configs/generic_landing_stage1d/*.json once
  check                     confirm the plans are what the configuration gives, that the starts keep the distances they must
                            (from stage 1c's sealed sets among others) and that every seed lies inside its declared range
  ids critical              the twelve forced starts of the pilot (stage 1b's checkpoint flies them for the comparison)
  failures --runs NAME=DIR ... --ids ID ...
                            the audit of flights that stopped in the air: the last decisions, and the recorded labels near them
  left-out                  the recorded labels of stage 1c and stage 1b that the stage 1d teacher would not give (not replayed)
  inputs DATASET            what a recorded dataset gives a student: sentences from the catalogue, no number, no id, proprio off
  replay DATASET            what the merged data holds: none of stage 1, some of stage 1b, stage 1c's same-colour, stage 1d's
  teacher-pilot --run DIR   the teacher's twenty forced flights before the recording (exit 0: the pilot's lines are met)
  pilot --runs stage1c=DIR 500=DIR ...
                            the candidates' table, the eligible ones in their order and the one that goes on (exit 0: there is one)
  qualification --run DIR   the chosen candidate's twenty flights before the sealed gates (exit 0: its lines are met)
  regression --run DIR --baseline DIR [--stage1b DIR]
                            stage 1c's regression lines (read from stage 1c's file) and stage 1d's one more
  descent --runs NAME=DIR ...   the last metres of every forced cube landing, from the decisions each policy made itself
  decision --audit JSON --inputs JSON --replay JSON --pilot JSON --qualification JSON --a JSON --repeat-a JSON --b JSON
           --repeat-b JSON --regression JSON

Gate A, gate B and their second flights are stage 1c's and are judged by stage 1c's tool (scripts/generic_landing_1c.py gate /
repeat), with stage 1c's numbers. Every command takes --output JSON. A run folder is one scripts/surface_flights.py wrote.
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.generic_landing as base
import scripts.generic_landing_1c as tool1c
from src.landing import failures as reading
from src.visual_search import generic_landing as stages
from src.visual_search import generic_landing_1c as before
from src.visual_search import generic_landing_1d as plans
from src.visual_search.episodes import load_config
from src.aerovla_oft.spec import load_config as load_oft_config

flights=base.flights;measure=base.measure;lines=base.lines;write=base.write
where=lambda path:Path(path) if Path(path).is_absolute() else ROOT/path
DATA=ROOT/'datasets/projectairsim_visual_search'


def describe_training(stage,plan):
    out={'description':'Generic landing, stage 1d: the plan of the recorded episodes. Landings pushed into one of four low states over a top and left to the teacher '
                       '(src/landing/teacher.FinalDescentTeacher), recorded in layouts earlier episodes were recorded in.',
         'seeds':{name:stage['seeds'][name] for name in ('train_final','val_final')}}
    for split,episodes in plan.items():
        count=lambda key:dict(sorted(collections.Counter(key(e) for e in episodes).items(),key=lambda pair:str(pair[0])))
        out[f'{split}_summary']={'episodes':len(episodes),'states':count(lambda e:e['recovery']),'surface_kinds':count(lambda e:e['surface_kind']),'targets':count(lambda e:e['target']),
                                 'bands':count(lambda e:e['band']),'scenes':count(lambda e:e['map']),'layouts':count(lambda e:e['layout'])}
        out[split]=episodes
    return out


def check(config,oft,stage):
    """The files are what the configuration gives; recorded starts are in recording layouts and check starts in evaluation
    layouts; no start of this stage is within the separation of a start of stage 1c's sets or of another check start; no
    recorded start is near any start flown for a check; every seed lies inside the range its set declares."""
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
    for index,a in enumerate(checks):
        for b in checks[index+1:]:
            if near(a,b):errors.append(f'{a["id"]} and {b["id"]} start too near each other')
    for a in recorded:
        for b in checks:
            if near(a,b):errors.append(f'{a["id"]} (recorded) and {b["id"]} (check) start too near each other')
    # Stage 1c's sets, the sealed gates among them: nothing of this stage starts near any of their starts.
    read=lambda paths:[episode for path in paths for episode in json.loads(Path(path).read_text(encoding='utf-8'))['episodes']]
    for mine,theirs in ((recorded,read(plans.earlier_sets(stage,strict=True))),(checks,read(plans.earlier_sets(stage)))):
        for a in mine:
            for b in theirs:
                if near(a,b):errors.append(f'{a["id"]} starts too near {b["id"]} of stage 1c')
    flown=before.check_points(stage)
    for a in recorded:
        if any(name==a['map'] and math.dist(a['start_xy'],xy)<radius for name,xy in flown):errors.append(f'{a["id"]} (recorded) starts too near an earlier check start')
    spots=collections.Counter((e['map'],tuple(e['start_xy'])) for e in recorded+checks);errors+=[f'start used twice: {key}' for key,count in spots.items() if count>1]
    seeds=collections.Counter(e['seed'] for e in recorded+checks);errors+=[f'seed used twice: {key}' for key,count in seeds.items() if count>1]
    ranges={'pilot':saved['pilot'],'qualification':saved['qualification'],'teacher_pilot':saved['teacher_pilot'],'train_final':plan['train'],'val_final':plan['val']}
    for name,episodes in ranges.items():
        low,high=stage['seeds'][name];errors+=[f'{e["id"]}: seed outside {name} {low}-{high}' for e in episodes if not low<=e['seed']<=high]
    return errors


def recorded_steps(root):
    """(summary, step) of every recorded decision of a dataset."""
    for path in sorted((where(root)/'episodes').glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        for step in record['steps']:yield record['summary'],step


def left_out(stage,config,landing_rule,names):
    """Recorded decisions whose label the stage 1d teacher would not give in the same place (configs, dataset.left_out):
    over a small top, no higher than the commit height over it, between the align radius and the hard recovery radius,
    and the recorded label is a climb."""
    from src.landing.surface import SurfaceSet
    from src.landing.teacher import hard_recovery_radius
    from src.visual_search.maps import load_map
    rule=landing_rule['teacher'];pad=landing_rule['touchdown']['landing_region_half_width_m'];surfaces={};out={}
    for name in names:
        chosen=collections.defaultdict(list);episodes=0
        for summary,step in recorded_steps(DATA/name):
            if summary.get('task')!='land':continue
            key=(summary['map'],summary['layout']);surface=surfaces.setdefault(key,SurfaceSet(load_map(summary['map']),summary['layout']))[summary['target']]
            if not surface.landable or surface.usable_reach_m>=pad or step.get('perturbed') or step.get('landed') or step.get('on_surface') or not step.get('over'):continue
            above=step['height_m']-surface.top_m;hard=hard_recovery_radius(surface,stage['teacher']['drift_margin_m'])
            if above<=rule['commit_height_m'] and rule['align_radius_m']<step['distance_m']<=hard and step['teacher_state']=='climb' and step['teacher_action'][1]<0:
                chosen[summary['id']].append(step['step'])
        out[name]={'decisions':sum(len(steps) for steps in chosen.values()),'episodes':len(chosen),'steps':dict(sorted(chosen.items()))}
    return out


def replay_audit(stage,merged):
    """What the merged data holds, by where each episode was recorded."""
    summary=json.loads((where(merged)/'summary.json').read_text(encoding='utf-8'));sources=summary['sources'];data=stage['dataset']
    same=[summary_ for summary_,step in recorded_steps(DATA/data['corrective']) if step['step']==0 and summary_.get('distractor')]
    twins=sum(bool(item.get('twin_of')) for item in same)
    result={'what':str(merged),'episodes':summary['episodes'],'samples':summary['samples'],'by_source':sources,
            'stage_1_generic_landing_episodes':sum(sources.get(name,0) for name in data['not_used']),'stage_1b_episodes':sources.get(data['stop_safe'],0),
            'stage_1c_episodes':sources.get(data['corrective'],0),'stage_1c_same_colour_episodes_recorded':len(same),'stage_1c_twins_recorded':twins,
            'stage_1d_episodes':sources.get(data['added'],0),'older_episodes':sum(sources.get(name,0) for name in data['replay'])}
    # The merge takes every episode of a source it is given: stage 1c's count in the merge is all of its recorded episodes, the same-colour ones among them.
    whole=len(list((DATA/data['corrective']/'episodes').glob('*.json')))
    result['all_of_stage_1c_is_in']=result['stage_1c_episodes']==whole
    result['passed']=bool(result['stage_1_generic_landing_episodes']==0 and result['stage_1b_episodes']>0 and result['stage_1d_episodes']>0 and result['all_of_stage_1c_is_in'] and len(same)>0 and twins>0)
    return result


def label_profile(steps,top):
    """A teacher flight after its last push, from its labels: where the descent began, how often the vertical label changed
    sign and how many separate climbs there were."""
    pushed=[index for index,step in enumerate(steps) if step.get('perturbed')];after=[step for step in (steps[pushed[-1]+1:] if pushed else steps) if not step.get('landed')]
    signs=[1 if step['teacher_action'][1]>0 else -1 for step in after if step['teacher_action'][1]!=0];names=[step['teacher_state'] for step in after]
    climbs=sum(1 for index,name in enumerate(names) if name=='climb' and (index==0 or names[index-1]!='climb'))
    first=next((step for step in after if step['teacher_state']=='descend'),None);last=next((step for step in reversed(after) if step['teacher_state']!='descend'),None)
    final=next((step for index,step in enumerate(after) if step['teacher_state']=='descend' and all(other['teacher_state']=='descend' for other in after[index:])),None)
    return {'let_go':None if not after else {'distance_m':round(after[0]['distance_m'],2),'above_top_m':round(after[0]['height_m']-top,2),'label':after[0]['teacher_state']},
            'descent_began':None if final is None else {'distance_m':round(final['distance_m'],2),'above_top_m':round(final['height_m']-top,2)},
            'sign_changes_of_the_vertical_label':sum(a!=b for a,b in zip(signs,signs[1:])),'separate_climbs':climbs,'decisions_after_the_push':len(after)}


def judge_teacher_pilot(stage,records,steps,oft,config):
    from src.landing import stop_safety
    rule=stage['teacher_pilot']['pass'];good=lambda item:bool(item['surface_landing']['success']);numbers=measure(records)
    episodes=[{'id':item['id'],'task':item['episode']['task'],'kind':item['episode'].get('surface_kind'),'start_distance_m':item['episode'].get('start_distance_m',-1.),'steps':steps[item['id']]} for item in records]
    labels=stop_safety.audit(episodes,oft,{key:stage['stop_safety'][key] for key in ('safe_ratio','near_stop_ratio')},config['stop_ticks'])
    # The audit's lines: labels that are a stop, labels below the safe ratio, runs of stops (src/landing/stop_safety.audit).
    stops=sum(check['count'] for check in labels['checks'] if 'below' not in check['what']);margins=sum(check['count'] for check in labels['checks'] if 'below' in check['what'])
    reached=lambda item:bool(item.get('forced',{}).get('all_came_to_pass_and_seen') and item['forced'].get('last_let_go_in_the_air'))
    unseen=[item['id'] for item in records if not reached(item)];table=[]
    for item in records:
        top=item['surface_landing']['target_surface']['top_m'];row={'id':item['id'],'state':item['episode']['recovery'],'landed':good(item),'reached':reached(item),
                                                                    'touchdown_error_m':None if item['surface_landing'].get('horizontal_error_m') is None else round(item['surface_landing']['horizontal_error_m'],2)}
        row.update(label_profile(steps[item['id']],top));table.append(row)
    checks=[('forced landings the teacher finished (both pushes came to pass, the object in view where each let go, the vehicle in the air at the second, and it landed)',
             sum(good(item) and reached(item) for item in records),f'>= {rule["success_min"]} of {rule["of"]}',sum(good(item) and reached(item) for item in records)>=rule['success_min'] and len(records)==rule['of']),
            ('collisions',numbers['collisions'],f'<= {rule["collisions_max"]}',numbers['collisions']<=rule['collisions_max']),
            ('flights at a wrong object',numbers['wrong_target'],f'<= {rule["wrong_target_max"]}',numbers['wrong_target']<=rule['wrong_target_max']),
            ('labels that are a stop, or runs of stops, before the touchdown',stops,f'<= {rule["stop_violations_max"]}',stops<=rule['stop_violations_max']),
            ('landing labels before the touchdown below the safe ratio',margins,f'<= {rule["safe_ratio_violations_max"]}',margins<=rule['safe_ratio_violations_max']),
            ('forced states the pictures did not show or that did not come to pass',len(unseen),f'<= {rule["forced_state_visibility_failures_max"]}',len(unseen)<=rule['forced_state_visibility_failures_max'])]
    by_state={state:{'flown':sum(row['state']==state for row in table),'reached':sum(row['state']==state and row['reached'] for row in table),'landed':sum(row['state']==state and row['landed'] for row in table),
                     'sign_changes':sum(row['sign_changes_of_the_vertical_label'] for row in table if row['state']==state),'separate_climbs':sum(row['separate_climbs'] for row in table if row['state']==state)}
              for state in plans.STATES}
    return {'numbers':{key:numbers[key] for key in ('flights','success','wrong_target','collisions')},'by_state':by_state,'labels':{'checks':labels['checks'],'by_task':labels['by']['task']},
            'not_reached':unseen,'flights':table},checks


def pilot_row(stage,name,records,loss,limit_m):
    numbers=measure(records);land=[item for item in records if item['episode']['task']=='land'];good=lambda item:bool(item['surface_landing']['success'])
    kind=lambda item:item['episode'].get('surface_kind');same=[item for item in land if item['episode'].get('distractor')];forced=[item for item in land if item['episode'].get('recovery')]
    plain=[item for item in land if not item['episode'].get('distractor') and not item['episode'].get('recovery')]
    hit=lambda item:bool(item['surface_landing']['collision'] or item['summary'].get('hit') is not None or item['summary'].get('reason')=='collision')
    wrong=lambda item:bool(item['surface_landing'].get('landed_on')) and item['surface_landing']['landed_on']!=item['episode']['target']
    air=lambda item:bool(item['summary'].get('stopped') and not item['surface_landing']['contact_from_above'] and not hit(item))
    pushed=lambda item:bool(item.get('forced',{}).get('all_came_to_pass_and_seen') and item['forced'].get('last_let_go_in_the_air'))
    return {'candidate':name,'flights':len(records),'same_color_collisions':sum(hit(item) for item in same),'wrong_target_landings':sum(wrong(item) for item in land),
            'critical':sum(good(item) for item in forced),'critical_of':len(forced),'critical_by_state':{state:sum(good(item) for item in forced if item['episode']['recovery']==state) for state in plans.STATES},
            # For reading the table, not for the order: in how many of the forced landings the state came to pass in the air.
            'critical_pushed':sum(pushed(item) for item in forced),'critical_pushed_and_landed':sum(good(item) and pushed(item) for item in forced),
            'critical_air_self_stops':sum(air(item) for item in forced),'air_self_stops':sum(air(item) for item in land),
            'cube_landings':sum(good(item) for item in land if kind(item)=='cube'),'cube_of':sum(kind(item)=='cube' for item in land),
            'cylinder_landings':sum(good(item) for item in plain if kind(item)=='cylinder'),'cylinder_of':sum(kind(item)=='cylinder' for item in plain),
            'pad_landings':sum(good(item) for item in plain if kind(item)=='pad'),'pad_of':sum(kind(item)=='pad' for item in plain),
            'pad_landings_all':sum(good(item) for item in land if kind(item)=='pad'),'pad_of_all':sum(kind(item)=='pad' for item in land),
            'same_color_landings':sum(good(item) for item in same),'same_color_of':len(same),'collisions':numbers['collisions'],'wrong_target':numbers['wrong_target'],'success':numbers['success'],
            'ceiling_overruns':sum(item['summary']['height_range_m'][1]>limit_m for item in records),'highest_m':round(max(item['summary']['height_range_m'][1] for item in records),1),
            'val_l1':loss,'failed':numbers['failed']}


def judge_pilot(stage,runs,losses,limit_m):
    """The table of the pilot; for each candidate the lines it does not meet; the candidates that meet every line in their
    fixed order."""
    rows=[pilot_row(stage,name,records,losses.get(name),limit_m) for name,records in runs.items()];before_=next((row for row in rows if row['candidate']=='stage1c'),None)
    rule=stage['pilot']['pass'];candidates=[row for row in rows if row['candidate'].isdigit()]
    for row in candidates:
        checks=[('collisions in the same-colour starts',row['same_color_collisions'],f'<= {rule["same_color_collisions_max"]}',row['same_color_collisions']<=rule['same_color_collisions_max']),
                ('landings on a wrong object',row['wrong_target_landings'],f'<= {rule["wrong_target_landings_max"]}',row['wrong_target_landings']<=rule['wrong_target_landings_max']),
                ('forced cube landings that landed',row['critical'],f'>= {rule["critical_min"]} of {rule["critical_of"]}',row['critical']>=rule['critical_min'] and row['critical_of']==rule['critical_of']),
                ('stops in the air among the forced cube landings',row['critical_air_self_stops'],f'<= {rule["critical_air_self_stops_max"]}',row['critical_air_self_stops']<=rule['critical_air_self_stops_max']),
                ('pad landings',row['pad_landings'],f'>= {rule["pad_landings_min"]} of {rule["pad_landings_of"]}',row['pad_landings']>=rule['pad_landings_min'] and row['pad_of']==rule['pad_landings_of']),
                ('cylinder landings',row['cylinder_landings'],f'>= {rule["cylinder_landings_min"]} of {rule["cylinder_landings_of"]}',row['cylinder_landings']>=rule['cylinder_landings_min'] and row['cylinder_of']==rule['cylinder_landings_of']),
                ("forced cube landings against stage 1c's checkpoint on the same starts",row['critical'],f'>= {before_["critical"] if before_ else "?"}',bool(before_) and row['critical']>=before_['critical'])]
        row['checks']=checks;row['eligible']=all(ok for _,_,_,ok in checks)
    loss=lambda row:row['val_l1'] if row['val_l1'] is not None else float('inf')
    order=lambda row:(row['same_color_collisions'],row['wrong_target_landings'],-row['critical'],row['air_self_stops'],-row['cube_landings'],-row['cylinder_landings'],-row['pad_landings_all'],
                      row['ceiling_overruns'],loss(row),int(row['candidate']))
    ranked=sorted((row for row in candidates if row['eligible']),key=order)
    return rows,[row['candidate'] for row in ranked],[row['candidate'] for row in sorted(candidates,key=order)]


def judge_qualification(stage,records,limit_m):
    rule=stage['qualification']['pass'];row=pilot_row(stage,'chosen',records,None,limit_m);land=[item for item in records if item['episode']['task']=='land']
    cylinders=[item for item in land if item['episode'].get('surface_kind')=='cylinder'];pads=[item for item in land if item['episode'].get('surface_kind')=='pad' and not item['episode'].get('distractor')]
    checks=[('flights',len(records),'= 20',len(records)==20),
            ('cube landings (four forced, six plain)',row['cube_landings']-sum(bool(item['surface_landing']['success']) for item in land if item['episode'].get('distractor') and item['episode'].get('surface_kind')=='cube'),
             f'>= {rule["cube_landings_min"]} of {rule["cube_landings_of"]}',None),
            ('same-colour starts landed on the named object',row['same_color_landings'],f'>= {rule["same_color_landings_min"]} of {rule["same_color_landings_of"]}',row['same_color_landings']>=rule['same_color_landings_min'] and row['same_color_of']==rule['same_color_landings_of']),
            ('collisions in the same-colour starts',row['same_color_collisions'],f'<= {rule["same_color_collisions_max"]}',row['same_color_collisions']<=rule['same_color_collisions_max']),
            ('pad landings',row['pad_landings'],f'>= {rule["pad_landings_min"]} of {rule["pad_landings_of"]}',row['pad_landings']>=rule['pad_landings_min'] and len(pads)==rule['pad_landings_of']),
            ('cylinder landings',row['cylinder_landings'],f'>= {rule["cylinder_landings_min"]} of {rule["cylinder_landings_of"]}',row['cylinder_landings']>=rule['cylinder_landings_min'] and len(cylinders)==rule['cylinder_landings_of']),
            ('flights that stop by themselves in the air',row['air_self_stops'],f'<= {rule["air_self_stops_max"]}',row['air_self_stops']<=rule['air_self_stops_max']),
            ('flights at a wrong object',max(row['wrong_target'],row['wrong_target_landings']),f'<= {rule["wrong_target_max"]}',max(row['wrong_target'],row['wrong_target_landings'])<=rule['wrong_target_max']),
            ('collisions',row['collisions'],f'<= {rule["collisions_max"]}',row['collisions']<=rule['collisions_max'])]
    name,value,limit,_=checks[1];own=[item for item in land if item['episode'].get('surface_kind')=='cube' and not item['episode'].get('distractor')]
    checks[1]=(name,value,limit,value>=rule['cube_landings_min'] and len(own)==rule['cube_landings_of'])
    return row,checks


def judge_regression(stage,model,baseline):
    """Stage 1c's lines with stage 1c's numbers, and stage 1d's one more: no landing on another object than the named one."""
    numbers,checks=tool1c.judge_regression(plans.gates_of(stage),model,baseline)
    wrong=[item['id'] for item in model if item['episode']['task']=='land' and item['surface_landing'].get('landed_on') and item['surface_landing']['landed_on']!=item['episode']['target']]
    limit=stage['regression']['wrong_object_landings_max'];checks.append(('landings on another object than the named one',len(wrong),f'<= {limit}',len(wrong)<=limit))
    numbers['landings_on_another_object']=wrong;return numbers,checks


def failure_audit(config,oft,landing,runs,ids,names):
    """Flights that ended by the policy's own stop in the air: the surface, the last twelve decisions, the descent, and what
    the recorded data of earlier stages labels near the place the flight ended at."""
    from src.landing.stop_safety import stop_ratio
    fields=[]
    for name in names:
        rows=[]
        for summary,step in recorded_steps(DATA/name):
            if summary.get('surface_kind')!='cube' or summary.get('task')!='land' or step.get('perturbed') or step.get('landed') or step.get('on_surface'):continue
            rows.append((step['distance_m'],step['height_m']-8.,step['teacher_action'],step['teacher_state'],abs(step['bearing_deg'])))
        fields.append((name,rows))
    out=[]
    for name,folder in runs:
        records={item['id']:item for item in flights(folder)};steps=reading.steps_of(where(folder))
        for key in ids:
            if key not in records:continue
            item=records[key];rows=steps[key];surface=item['surface_landing']['target_surface'];top=surface['top_m'];last=rows[-1];profile=reading.final_descent_profile(item,rows,landing,config,oft)
            start=next((index for index,step in enumerate(rows) if not step.get('perturbed') and step['distance_m']<=config['arrive_distance_m'] and step['action'][1]>oft['stop_threshold']),None)
            centre=surface['centre_xy'];place=last['position'];around=[]
            for source,field in fields:
                hit=[row for row in field if abs(row[0]-last['distance_m'])<=.4 and abs(row[1]-(last['height_m']-top))<=.6];abeam=[row for row in hit if row[4]>=45.]
                around.append({'data':source,'labels_within_0.4_m_and_0.6_m_of_height':len(hit),'descend':sum(row[3]=='descend' for row in hit),'climb':sum(row[3]=='climb' for row in hit),
                               'with_the_middle_45_deg_or_more_off_the_nose':len(abeam)})
            out.append({'checkpoint':name,'id':key,'reason':item['summary']['reason'],'decisions':len(rows),'surface':{'object':surface['object_id'],'top_m':top,'usable_half_extent_m':surface.get('usable_half_extent_m')},
                        'end':{'height_m':round(last['height_m'],2),'above_top_m':round(last['height_m']-top,2),'distance_from_the_middle_m':round(last['distance_m'],2),
                               'offset_from_the_middle_m':[round(place[0]-centre[0],2),round(place[1]-centre[1],2)],'over_the_footprint':bool(last.get('over')),'bearing_of_the_middle_deg':round(last['bearing_deg'],1),
                               'front_camera_sees_it':bool(last['front_seen']),'down_camera_sees_it':bool(last['down_seen'])},
                        'last_twelve':[{'step':step['step'],'above_top_m':round(step['height_m']-top,2),'distance_m':round(step['distance_m'],2),'forward_m':round(step['action'][0],3),'down_m':round(step['action'][1],3),
                                        'yaw_rad':round(step['action'][2],4),'stop_ratio':round(stop_ratio(step['action'],oft),2),'forced':bool(step.get('perturbed'))} for step in rows[-12:]],
                        'last_four_stop_ratios':[round(stop_ratio(step['action'],oft),2) for step in rows[-4:]],
                        'decisions_since_the_descent_began':None if start is None else len(rows)-start,'sign_flips':profile['sign_flips'],
                        'descent_began':{'distance_m':profile['descent_start_distance_m'],'above_top_m':profile['descent_start_height_above_top_m']},
                        'largest_forward_command_of_the_last_twelve_m':round(max(abs(step['action'][0]) for step in rows[-12:]),3),
                        'largest_turn_command_of_the_last_twelve_rad':round(max(abs(step['action'][2]) for step in rows[-12:]),4),'recorded_labels_near_the_end':around})
    return out


def named(items):return [(text.split('=',1)[0],text.split('=',1)[1]) for text in items]


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);parser.add_argument('--stage',default=str(plans.CONFIG))
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('plan','check','left-out'):commands.add_parser(name).add_argument('--output')
    one=commands.add_parser('ids');one.add_argument('which',choices=('critical',))
    one=commands.add_parser('failures');one.add_argument('--runs',nargs='+',required=True);one.add_argument('--ids',nargs='+',required=True);one.add_argument('--output')
    for name in ('inputs','replay'):one=commands.add_parser(name);one.add_argument('dataset');one.add_argument('--output')
    for name in ('teacher-pilot','qualification'):one=commands.add_parser(name);one.add_argument('--run',required=True);one.add_argument('--output')
    one=commands.add_parser('pilot');one.add_argument('--runs',nargs='+',required=True);one.add_argument('--output');one.add_argument('--training')
    one=commands.add_parser('regression');one.add_argument('--run',required=True);one.add_argument('--baseline',required=True);one.add_argument('--stage1b');one.add_argument('--output')
    one=commands.add_parser('descent');one.add_argument('--runs',nargs='+',required=True);one.add_argument('--output')
    one=commands.add_parser('decision')
    for name in ('audit','inputs','replay','pilot','qualification','a','repeat-a','b','repeat-b','regression'):one.add_argument(f'--{name}')
    one.add_argument('--output')
    args=parser.parse_args();stage=plans.load(args.stage);config=load_config();oft=load_oft_config();files=plans.files_of(stage);passed=None;result=None
    from src.visual_search.maps import load_landing,load_map
    landing=load_landing();limit_m=load_map(stage['scenes'][0])['altitude']['ceiling_m']
    if args.command=='plan':
        if any(path.exists() for path in list(files.values())+[plans.train_plan_of(stage)]):raise SystemExit('the plans of stage 1d exist; they are written once')
        sets=plans.build_all(config,oft,stage)
        for name in files:write(files[name],plans.describe(stage,name,sets[name]));print(name,len(sets[name]))
        described=describe_training(stage,sets['train_plan']);write(plans.train_plan_of(stage),described)
        for split in ('train','val'):print(split,json.dumps(described[f'{split}_summary']))
        return
    if args.command=='check':
        errors=check(config,oft,stage)
        for text in errors:print('ERROR: '+text)
        if errors:raise SystemExit(1)
        print("OK: the plans are what the configuration gives; no start is near a start of stage 1c's sets; recorded starts keep away from every check start; every seed is inside its range");return
    if args.command=='ids':
        print(json.dumps([e['id'] for e in json.loads(files['pilot'].read_text(encoding='utf-8'))['episodes'] if e.get('recovery')]));return
    if args.command=='failures':
        result={'description':"Flights that ended by the policy's own stop in the air, read decision by decision, and the labels the recorded data holds near where each ended.",
                'flights':failure_audit(config,oft,landing,named(args.runs),args.ids,[stage['dataset']['stop_safe'],stage['dataset']['corrective']])}
        for row in result['flights']:print(row['checkpoint'],row['id'],json.dumps(row['end']),row['last_four_stop_ratios'],json.dumps(row['recorded_labels_near_the_end']))
    elif args.command=='left-out':
        result={'rule':stage['dataset']['left_out']['rule'],**left_out(stage,config,landing,[stage['dataset']['stop_safe'],stage['dataset']['corrective']])}
        for name in (stage['dataset']['stop_safe'],stage['dataset']['corrective']):print(name,result[name]['decisions'],'decisions in',result[name]['episodes'],'episodes')
    elif args.command=='inputs':
        result=tool1c.inputs_audit(args.dataset,stage,config);records=[summary for summary,step in recorded_steps(args.dataset) if step['step']==0]
        result['same_colour_episodes']=sum(bool(summary.get('distractor')) for summary in records);result['states']=dict(collections.Counter(summary.get('recovery') for summary in records))
        result['passed']=passed=bool(result['passed'] and result['same_colour_episodes']==0)
        print(json.dumps({key:value for key,value in result.items() if key not in ('examples','what_reaches_the_student')}))
    elif args.command=='replay':
        result=replay_audit(stage,args.dataset);passed=result['passed'];print(json.dumps(result))
    elif args.command=='teacher-pilot':
        numbers,checks=judge_teacher_pilot(stage,flights(args.run),reading.steps_of(where(args.run)),oft,config)
        print(json.dumps(numbers['by_state']));passed=lines(checks);result={**numbers,'checks':checks,'passed':passed}
    elif args.command=='pilot':
        runs={name:flights(folder) for name,folder in named(args.runs)};losses={}
        for name in runs:
            manifest=(where(args.training) if args.training else ROOT/stage['run'])/f'update-{name}'/'manifest.json'
            if manifest.exists():losses[name]=json.loads(manifest.read_text(encoding='utf-8'))['candidate']['val_l1']
        rows,eligible,ranked=judge_pilot(stage,runs,losses,limit_m)
        for row in rows:
            print(json.dumps({key:value for key,value in row.items() if key not in ('failed','checks')}))
            if 'checks' in row:lines(row['checks'])
        print('order of all:',ranked);print('eligible, in order:',eligible);passed=bool(eligible)
        result={'rows':rows,'order_of_all':ranked,'eligible':eligible,'chosen':eligible[0] if eligible else None,'passed':passed}
    elif args.command=='qualification':
        row,checks=judge_qualification(stage,flights(args.run),limit_m);print(json.dumps({key:value for key,value in row.items() if key!='failed'}));passed=lines(checks)
        result={'numbers':row,'checks':checks,'passed':passed}
    elif args.command=='regression':
        numbers,checks=judge_regression(stage,flights(args.run),flights(args.baseline))
        if args.stage1b:numbers['stage_1b']=measure(flights(args.stage1b))
        for name in ('model','baseline','stage_1b'):
            if name in numbers:print(name,json.dumps({key:numbers[name][key] for key in ('flights','success','land_success','approach_success','wrong_target','collisions','approach_touchdowns','stopped_in_air')}))
        passed=lines(checks);result={'numbers':numbers,'checks':checks,'passed':passed}
    elif args.command=='descent':
        result={}
        for name,folder in named(args.runs):
            steps=reading.steps_of(where(folder));result[name]=[reading.final_descent_profile(item,steps[item['id']],landing,config,oft,tuple(stage['descent_reading']['band_over_top_m']))
                                                              for item in flights(folder) if item['episode']['task']=='land' and item['episode'].get('recovery')]
            for row in result[name]:print(name,json.dumps(row))
    elif args.command=='decision':
        load=lambda path:json.loads(where(path).read_text(encoding='utf-8'))['passed'] if path and where(path).exists() else None
        parts={'the recorded landing labels keep the stop rule':load(args.audit),'what the data gives a student is pictures and a sentence':load(args.inputs),
               "the merged data holds none of stage 1 and some of stage 1b, stage 1c's same-colour and stage 1d":load(args.replay),'the pilot':load(args.pilot),'the qualification':load(args.qualification),
               'gate A (generic landing)':load(args.a),'gate A, second flights of the cube and cylinder starts':load(getattr(args,'repeat_a')),
               'gate B (same colour)':load(args.b),'gate B, second flights':load(getattr(args,'repeat_b')),'regression':load(args.regression)}
        passed=all(value is True for value in parts.values())
        result={'parts':{name:('not flown' if value is None else value) for name,value in parts.items()},'stage_1d':'PASSED' if passed else 'NOT PASSED','generic_landing_v1_exists':(ROOT/stage['checkpoint']).exists()}
        for name,good in parts.items():print(f'  {"PASS" if good else "not flown" if good is None else "FAIL"}  {name}')
    if args.output and result is not None:write(where(args.output),result)
    if passed is not None:print('PASSED' if passed else 'NOT PASSED');raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
