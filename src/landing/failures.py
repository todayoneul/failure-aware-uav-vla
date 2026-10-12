"""How a failed landing is named, and how far a policy's flown actions were from a stop. No simulator, no torch.

A failed landing gets two names. `end`: how the flight ended (the policy's own stop in the air, a collision, the decision
limit, another object, a contact that was no good landing). `phase`: where in the landing the vehicle was at a decision,
read from its position against the teacher's own boundaries (the arrival distance, the fly-in height over the top, the
radii at which the canonical teacher descends) and not from anything the policy put out. The same rule reads the flights
of stage 1 and of stage 1b, so the two can be compared.

Reads the records of scripts/surface_flights.py: per flight the canonical summary and the surface evaluator's reading
(results.json), per decision the vehicle's state and the action that was flown (steps.jsonl).
"""
import collections
import json
import math
from pathlib import Path
from src.aerovla_oft.spec import is_stop
from src.landing.stop_safety import stop_ratio,spread
from src.landing.teacher import LEVEL_BAND_M,COMMIT_MARGIN_M

ENDS=('AIR_SELF_STOP','COLLISION','TIMEOUT','TARGET_SELECTION','TOUCHDOWN','STABILITY','OTHER')
PHASES=('SEARCH','SURFACE_APPROACH','CLIMB','ALIGNMENT','DESCENT')


def steps_of(run):
    """Every decision of a run folder, by episode."""
    rows=collections.defaultdict(list)
    with (Path(run)/'steps.jsonl').open(encoding='utf-8') as file:
        for line in file:
            step=json.loads(line);rows[step['episode']].append(step)
    return rows


def fly_in_height(surface,landing):
    """How high over a top the teacher flies in (src/landing/teacher.py): a pad's or a smaller top's."""
    reach=surface.get('usable_radius_m') or min(surface.get('usable_half_extent_m') or [landing['touchdown']['landing_region_half_width_m']])
    small=reach<landing['touchdown']['landing_region_half_width_m']
    return landing['teacher']['commit_height_m']+COMMIT_MARGIN_M if small else landing['teacher']['slow_height_m']


def phase_of(step,surface,landing,config):
    """Where in a landing the vehicle is at this decision, from its position alone."""
    rule=landing['teacher'];above=step['height_m']-surface['top_m'];distance=step['distance_m'];over=bool(step.get('over'))
    if distance<=rule['align_radius_m'] or (over and above<=rule['commit_height_m'] and distance<=rule['commit_radius_m']):return 'DESCENT'
    if distance>config['arrive_distance_m']:return 'SURFACE_APPROACH' if step.get('front_seen') or step.get('down_seen') else 'SEARCH'
    if not over and above<fly_in_height(surface,landing)-LEVEL_BAND_M:return 'CLIMB'
    return 'ALIGNMENT'


def end_of(record):
    """How a failed flight ended."""
    summary=record['summary'];reading=record['surface_landing']
    if reading.get('collision') or summary.get('reason')=='collision' or summary.get('hit') is not None:return 'COLLISION'
    if summary.get('wrong_target') or reading.get('wrong_object') or reading.get('wrong_object_landing'):return 'TARGET_SELECTION'
    if reading.get('contact_from_above') and reading.get('correct_object'):
        return 'STABILITY' if reading.get('inside_region') and reading.get('acceptable_touchdown') else 'TOUCHDOWN'
    if summary.get('reason')=='max_steps':return 'TIMEOUT'
    if summary.get('stopped') and not reading.get('contact_from_above'):return 'AIR_SELF_STOP'
    return 'OTHER'


def flown(step):
    """The action the vehicle was given at a decision (the first action of the predicted chunk)."""
    return step['action']


def classify(record,steps,landing,config,oft):
    """A failed landing: its end, its phase at the last decision, and for a stop in the air everything about its last moments."""
    episode=record['episode'];summary=record['summary'];surface=record['surface_landing']['target_surface'];last=steps[-1]
    out={'id':record['id'],'kind':episode.get('surface_kind') or episode.get('role'),'task':episode['task'],'end':end_of(record),
         'phase':phase_of(last,surface,landing,config) if surface.get('landable') else None,'reason':summary['reason'],'decisions':len(steps)}
    if out['end']=='AIR_SELF_STOP':
        tail=steps[-config['stop_ticks']:];angle=last.get('yaw_rad',0.)+math.radians(last['bearing_deg'])
        out['self_stop']={'height_m':round(last['height_m'],2),'height_above_top_m':round(last['height_m']-surface['top_m'],2) if surface.get('landable') else None,
                          'target_distance_m':round(last['distance_m'],2),
                          # The vehicle's place seen from the middle of the surface (metres along the map's axes).
                          'offset_from_centre_m':[round(-last['distance_m']*math.cos(angle),2),round(-last['distance_m']*math.sin(angle),2)],
                          'over_the_object':bool(last.get('over')),'front_sees_target':bool(last['front_seen']),'down_sees_target':bool(last['down_seen']),
                          'last_actions':[[round(value,4) for value in flown(step)] for step in tail],
                          'last_stop_ratios':[round(stop_ratio(flown(step),oft),3) for step in tail],
                          'start_distance_m':episode.get('start_distance_m'),'start_height_m':episode.get('start_height_m'),'finalizer_state':last.get('finalizer'),
                          'phases_of_the_last_decisions':[phase_of(step,surface,landing,config) for step in tail] if surface.get('landable') else None}
    return out


def failures(records,steps,landing,config,oft):
    """Every failed landing of a run, named; and the counts by end and by phase."""
    rows=[classify(record,steps[record['id']],landing,config,oft) for record in records if record['episode']['task']=='land' and not record['surface_landing']['success']]
    count=lambda key,names:{name:sum(row[key]==name for row in rows) for name in names}
    stops=[row for row in rows if row['end']=='AIR_SELF_STOP']
    return {'failed_landings':len(rows),'by_end':count('end',ENDS),'air_self_stops_by_phase':{name:sum(row['phase']==name for row in stops) for name in PHASES},
            'by_kind':{kind:{'failed':sum(row['kind']==kind for row in rows),'air_self_stop':sum(row['kind']==kind for row in stops)} for kind in ('pad','cube','cylinder')},
            'flights':rows}


def output_margins(records,steps,landing,config,oft,kinds=('cube','cylinder')):
    """The stop ratio of every action the policy flew in the landings of a run before the vehicle touched anything, by phase."""
    groups=collections.defaultdict(list);runs=collections.Counter()
    for record in records:
        episode=record['episode'];surface=record['surface_landing']['target_surface']
        if episode['task']!='land' or episode.get('surface_kind') not in kinds or not surface.get('landable'):continue
        longest=run=0
        for step in steps[record['id']]:
            if step.get('landed') or step.get('on_surface'):break
            ratio=stop_ratio(flown(step),oft);phase=phase_of(step,surface,landing,config)
            for name in (phase,'ALL'):groups[name].append(ratio)
            run=run+1 if is_stop(flown(step),oft) else 0;longest=max(longest,run)
        runs['flights']+=1;runs['with_a_stop_decision']+=longest>=1;runs['with_two_in_a_row']+=longest>=2;runs['with_three_in_a_row']+=longest>=3
        runs['ended_by_four']+=longest>=config['stop_ticks']
    table={}
    for name in PHASES+('ALL',):
        values=groups.get(name,[])
        table[name]=None if not values else {**spread(values),'stops':sum(value<=1. for value in values),'share_stop':round(sum(value<=1. for value in values)/len(values),4),
                                             'share_below_2':round(sum(value<2. for value in values)/len(values),4),
                                             'share_below_3':round(sum(value<3. for value in values)/len(values),4)}
    return {'kinds':list(kinds),'flights':dict(runs),'by_phase':table}


def ceiling(records,limit_m):
    """Flights that went higher than the start rule's ceiling (nothing in the executor holds a policy below it)."""
    over=[{'id':record['id'],'highest_m':round(record['summary']['height_range_m'][1],1),'success':bool(record['surface_landing']['success'])}
          for record in records if record['summary']['height_range_m'][1]>limit_m]
    return {'limit_m':limit_m,'flights':len(records),'over':len(over),'highest_m':round(max((record['summary']['height_range_m'][1] for record in records),default=0.),1),'episodes':over}


def descent_profile(record,steps,landing,config,oft):
    """The last part of a landing, read from where the vehicle was and what it flew: how high it came in over the object,
    where and how high it began to come down, how often its vertical command changed sign, and how the flight ended."""
    surface=record['surface_landing']['target_surface'];top=surface['top_m'];limit=oft['stop_threshold']*(oft['action_bounds']['down_m'][1]-oft['action_bounds']['down_m'][0])
    flown_steps=[step for step in steps if not (step.get('landed') or step.get('on_surface'))];reading=record['surface_landing']
    out={'id':record['id'],'kind':record['episode'].get('surface_kind'),'success':bool(reading['success']),'reason':record['summary']['reason'],'decisions':len(steps),
         'highest_above_top_m':round(max((step['height_m'] for step in steps),default=top)-top,2),
         'touchdown_error_m':None if reading.get('horizontal_error_m') is None else round(reading['horizontal_error_m'],2),'inside_region':bool(reading.get('inside_region')) if reading.get('landed_on') else None}
    near=[step for step in flown_steps if step['distance_m']<=config['arrive_distance_m']]
    over=next((step for step in near if step.get('over')),None)
    out['entry_height_above_top_m']=None if over is None else round(over['height_m']-top,2)
    # The descent starts at the first decision near the object whose flown vertical command is downward and larger than the stop band.
    start=next((index for index,step in enumerate(near) if step['action'][1]>limit),None)
    if start is None:out.update(descent_start_distance_m=None,descent_start_height_above_top_m=None,sign_flips=0,descent_decisions=0,climbs_after_the_descent_began=0);return out
    after=near[start:];signs=[1 if step['action'][1]>limit else -1 for step in after if abs(step['action'][1])>limit]
    out.update(descent_start_distance_m=round(after[0]['distance_m'],2),descent_start_height_above_top_m=round(after[0]['height_m']-top,2),
               sign_flips=sum(a!=b for a,b in zip(signs,signs[1:])),descent_decisions=len(after),climbs_after_the_descent_began=sum(sign<0 for sign in signs))
    return out


def plain(name):
    """An object's name as the catalogue and the simulator both spell it, reduced to letters: blue_cone, SceneBlueCone -> bluecone."""
    return ''.join(letter for letter in str(name).lower() if letter.isalnum()).replace('scene','')


def cube_counts(profiles,rule,usable_reach_m):
    """Stage 1c: how the cube landings of a run went wrong, counted from their descent profiles (reported, not a gate)."""
    failed=[row for row in profiles if not row['success']]
    return {'landings':len(profiles),'landed':sum(row['success'] for row in profiles),
            'insufficient_climb':sum(row['entry_height_above_top_m'] is not None and row['entry_height_above_top_m']<rule['clear_above_top_m'] for row in profiles),
            'premature_descent':sum(row['descent_start_distance_m'] is not None and row['descent_start_distance_m']>usable_reach_m for row in profiles),
            'edge_touchdown':sum(row['inside_region'] is False for row in profiles),
            'descent_oscillation':sum(row['sign_flips']>=2 for row in profiles),
            'air_self_stop':sum(row['reason']=='model_stop' and row['touchdown_error_m'] is None for row in failed),
            'timeout':sum(row['reason']=='max_steps' for row in failed)}


def same_color_reading(record,steps,other_centre,arrive_m):
    """A landing whose first view held another object of the named colour, read against that object: did the flight go to
    it, did it end at it? From the vehicle's positions and the evaluator's reading, not from anything the policy put out."""
    episode=record['episode'];summary=record['summary'];reading=record['surface_landing'];other=episode['distractor'];target=episode['target']
    away=[math.hypot(other_centre[0]-step['position'][0],other_centre[1]-step['position'][1]) for step in steps]
    near=[index for index,(distance,step) in enumerate(zip(away,steps)) if distance<=arrive_m and distance<step['distance_m']]
    acquired=next((index for index,step in enumerate(steps) if step['front_seen'] or step['down_seen']),None)
    hit=reading.get('collision') or summary.get('hit');last=steps[-1]
    hit_other=bool(hit) and plain(hit).startswith(plain(other));hit_wrong=bool(hit) and not plain(hit).startswith(plain(target))
    wrong_landing=bool(reading.get('landed_on')) and reading['landed_on']!=target
    stopped_at_other=bool(summary.get('stopped') and not reading.get('contact_from_above') and not hit and away[-1]<=arrive_m and away[-1]<last['distance_m'])
    wrong_terminal=bool(wrong_landing or hit_wrong or stopped_at_other)
    return {'id':record['id'],'target':target,'other':other,'success':bool(reading['success']),'reason':summary['reason'],'landed_on':reading.get('landed_on'),
            'collision':hit,'collision_with_the_other':hit_other,'wrong_target_landing':wrong_landing,'stopped_at_the_other':stopped_at_other,'wrong_terminal':wrong_terminal,
            'approached_the_other':bool(near),'rejected_the_other':not near and not wrong_terminal,
            'nearest_to_the_other_m':round(min(away),1),'start_distance_to_the_other_m':round(away[0],1),'target_acquired_at_decision':acquired,
            'nearest_to_the_other_before_the_target_was_seen_m':round(min(away[:acquired+1] if acquired is not None else away),1),
            'first_approach_of_the_other_at_decision':near[0] if near else None,'decisions':len(steps),'final_distance_to_target_m':round(last['distance_m'],1),
            'final_distance_to_the_other_m':round(away[-1],1)}
