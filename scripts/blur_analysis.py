"""Read the Gaussian blur characterization flights and write the tables; no simulator, no model.

  blur_analysis.py [--root outputs/failures/gaussian_blur] [--output DIR] [--figures]

Every flight is read by the reader of the canonical tables (scripts/freeze_gen_v3.read through scripts/canonical_baseline.flown),
unchanged, so each number means what it meant for the clean canonical test. On top of that: the experiment's failure class,
the comparison of the four conditions start by start, the second flights of the repeat subset, and what was logged per
decision about the frames (hashes, sharpness, timings). Nothing here feeds back into a flight.

Output (small files, kept in Git): summary.json, tables.md, episodes.csv, paired_starts.csv and, with --figures, plots.
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.canonical_baseline import flown,measures
from scripts.freeze_gen_v3 import in_view,percentile
from scripts.gen_v3c import selection
from src.failures import blur_experiment as blur
from src.visual_search.maps import load_landing,load_map

CONDITIONS=blur.SEVERITY_ORDER
LANDING_KEYS=(('alignment','aligned'),('descent','descent_started'),('touchdown','touchdown_success'),('stable_landing','stable_physical_landing'),
              ('system_landing','system_landing'),('strict_zero_action','strict_policy_zero_action'))


def share(count,total):return f'{count}/{total}'


def rate(count,total):return round(count/total,3) if total else None


def spread(values):
    values=[value for value in values if value is not None]
    if not values:return None
    return {'n':len(values),'mean':statistics.mean(values),'median':statistics.median(values),'p5':percentile(values,.05),'p95':percentile(values,.95)}


def load_condition(folder):
    """One condition's flights: canonical records, the plan and step log, and the per-decision extra records."""
    if not (folder/'results.json').exists():return None
    items,errors,left,(episodes,steps)=flown(folder);extra={}
    for line in (folder/'extra.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line);extra.setdefault(row['episode'],[]).append(row)
    by_id={episode['id']:episode for episode in episodes}
    for item in items:
        item['class']=blur.failure_class(item);item['cell']=blur.cell(item);item['blur']=by_id[item['episode']].get('blur',{})
        rows=steps[item['episode']];first=next((index for index,row in enumerate(rows) if in_view(row)),None)
        size=lambda row:((row.get('target_size') or {}).get('front') or {}).get('image_fraction')
        item.update(first_view_step=first,down_seen=any(row.get('down_seen') for row in rows),over_pad=any(row.get('over') for row in rows),
                    yaw_before_first_view_deg=None if first is None else round(sum(abs(math.degrees(row['action'][2])) for row in rows[:first])),
                    seconds_to_first_view=None if first is None else rows[first]['epoch']-rows[0]['epoch'],
                    front_fraction_at_start=size(rows[0]),front_fraction_at_first_view=None if first is None else size(rows[first]),
                    mean_inference_ms=statistics.mean(row['inference_ms'] for row in rows))
    return {'items':items,'by_start':{item['episode']:item for item in items},'episodes':episodes,'steps':steps,'extra':extra,'errors':errors}


def stage_counts(items):
    """The rows of the severity table: [count, of how many]."""
    land=[i for i in items if i['task']=='land'];seen=[i for i in items if i['acquired']];total=len(items)
    counts={'mission_success':[sum(i['success'] for i in items),total],'acquisition':[len(seen),total],
            'correct_grounding':[sum(i['grounded'] for i in seen),len(seen)],'ended_at_named_object':[sum(i['correct_target'] for i in items),total],
            'wrong_target':[sum(i['wrong_target'] for i in items),total],'approach':[sum(i['approached'] for i in items),total]}
    for name,key in LANDING_KEYS:counts[name]=[sum(bool(i[key]) for i in land),len(land)]
    counts.update(end_as_the_sentence_asked=[sum(i['terminal_as_asked'] for i in items),total],collision=[sum(i['collision'] for i in items),total],
                  timeout=[sum(i['timeout'] for i in items),total])
    return counts


def by_group(data,key,names):
    """Mission success per condition for the starts of each group: {group: {condition: [successes, flights]}}."""
    return {name:{condition:[sum(i['success'] for i in data[condition]['items'] if key(i)==name),sum(key(i)==name for i in data[condition]['items'])]
                  for condition in data} for name in names}


def landing_table(items):
    land=[i for i in items if i['task']=='land'];total=len(land)
    after=[i for i in land if i['aligned'] and not i['success']]
    return {'landings':total,'down_view_saw_the_pad':sum(i['down_seen'] for i in land),'over_the_pad':sum(i['over_pad'] for i in land),
            **{name:sum(bool(i[key]) for i in land) for name,key in LANDING_KEYS},
            'correct_pad_touched':sum(bool(i['touchdown_on_target']) for i in land),'touched_another_pad':sum(bool(i['landed'] and not i['touchdown_on_target']) for i in land),
            'finalizer_latched':sum(bool(i['finalizer_triggered']) for i in land),'disarmed':sum(bool(i['disarm_triggered']) for i in land),
            # Found, grounded and aligned over the pad, and the mission still failed: a failure of the landing itself.
            'failed_after_alignment':[i['episode'] for i in after],
            'touchdown_error_m':spread([i['touchdown_error_m'] for i in land if i['touchdown_on_target']]),
            'touchdown_vertical_mps':spread([i['touchdown_vertical_mps'] for i in land if i['touchdown_on_target']])}


def search_table(items):
    hidden=[i for i in items if not i['initially_visible']];found=[i for i in hidden if i['acquired']]
    return {'starts_with_the_target_outside_the_first_view':len(hidden),'acquired':len(found),'success':sum(i['success'] for i in hidden),
            'first_view_step':spread([i['first_view_step'] for i in found]),'seconds_to_first_view':spread([i['seconds_to_first_view'] for i in found]),
            'yaw_before_first_view_deg':spread([i['yaw_before_first_view_deg'] for i in found]),
            'decisions':spread([i['steps'] for i in hidden])}


def grounding_table(items):
    seen=[i for i in items if i['acquired']]
    return {'in_view':len(seen),'grounded':sum(i['grounded'] for i in seen),'grounded_on_first_view':sum(bool(i['grounded_on_first_view']) for i in seen),
            'ended_at_named_object':sum(i['correct_target'] for i in seen),'decisions_in_view':spread([i['decisions_in_view'] for i in seen]),
            'not_grounded':[i['episode'] for i in seen if not i['grounded']]}


def wrong_targets(condition,items,facts,scenes):
    from src.visual_search.gen_v3c import relation_of
    rows=[]
    for i in items:
        if not i['wrong_target']:continue
        fact=facts[i['episode']];scene=scenes[i['map']]
        rows.append({'condition':condition,'start':i['episode'],'instruction':i['instruction'],'named':i['target'],'went_to':i['wrong_object'],
                     'relation':relation_of(scene,i['target'],i['wrong_object']) if i['wrong_object'] else None,'landed_on':i['landed_on'],
                     'related_object_in_first_view':fact['first_view'],'target_initially_visible':i['initially_visible'],'named_object_ever_seen':i['acquired'],
                     'band':i['band'],'task':i['task'],'class':i['class'],'front_fraction_at_first_view':i['front_fraction_at_first_view'],'steps':i['steps']})
    return rows


def size_table(data):
    """Does the size of the target in the image at its first sighting separate what blur breaks? Starts are split at the
    clean flights' median first-sighting size; a start keeps its group in every condition."""
    clean={i['episode']:i['front_fraction_at_first_view'] for i in data['clean']['items'] if i['front_fraction_at_first_view'] is not None}
    if not clean:return None
    middle=statistics.median(clean.values());group=lambda i:None if i['episode'] not in clean else 'small' if clean[i['episode']]<middle else 'large'
    return {'split_at_front_image_fraction':middle,'starts':{name:sum(group(i)==name for i in data['clean']['items']) for name in ('small','large')},
            'success':by_group(data,group,('small','large')),
            'first_sighting_fraction_of_successes_and_failures':{condition:{'success':spread([i['front_fraction_at_first_view'] for i in data[condition]['items'] if i['success']]),
                                                                           'failure':spread([i['front_fraction_at_first_view'] for i in data[condition]['items'] if not i['success']])} for condition in data}}


def frame_tables(data):
    """Per condition and camera: sharpness of what the policy was given and of the raw frame, timings, and the input checks."""
    quality={};latency={};checks={};values={}
    for condition,item in data.items():
        rows=[row for rows in item['extra'].values() for row in rows]
        quality[condition]={kind:{camera:{metric:spread([row['quality'][kind][camera][metric] for row in rows]) for metric in ('laplacian_variance','gradient_magnitude','local_contrast')}
                                  for camera in blur.CAMERAS} for kind in ('input','raw')}
        values[condition]={camera:[row['quality']['input'][camera]['laplacian_variance'] for row in rows] for camera in blur.CAMERAS}
        latency[condition]={name:spread([row[key] for row in rows]) for name,key in (('blur_ms','blur_ms'),('inference_ms','inference_ms'),('decision_cycle_s','cycle_s'))}
        checks[condition]={'decisions':len(rows),**{f'{camera}_input_identical_to_raw':sum(row['input_is_raw'][camera] for row in rows) for camera in blur.CAMERAS}}
    order=[name for name in CONDITIONS if name in values]
    separation={f'{sharper} vs {blurrier}':{camera:blur.auroc(values[blurrier][camera],values[sharper][camera]) for camera in blur.CAMERAS}
                for sharper,blurrier in zip(order,order[1:])}
    return quality,latency,checks,separation


def repeat_table(main,again,ids):
    """The repeat subset: of two flights per start and condition, how many succeeded, and how often the two agreed."""
    table={};detail={}
    for condition in main:
        if again.get(condition) is None:continue
        pairs={start:(main[condition]['by_start'][start]['success'],again[condition]['by_start'][start]['success'])
               for start in ids if start in main[condition]['by_start'] and start in again[condition]['by_start']}
        table[condition]={'starts':len(pairs),'2/2':sum(a and b for a,b in pairs.values()),'1/2':sum(a!=b for a,b in pairs.values()),'0/2':sum(not a and not b for a,b in pairs.values()),
                          'flights_succeeded':sum(a+b for a,b in pairs.values()),'flights':2*len(pairs)}
        detail[condition]={start:f'{int(a)+int(b)}/2' for start,(a,b) in pairs.items()}
    return table,detail


def analyse(root,starts=None):
    experiment=blur.load();landing=load_landing();root=Path(root)
    data={name:load_condition(root/name) for name in CONDITIONS};data={name:item for name,item in data.items() if item}
    again={name:load_condition(root/'repeat'/name) for name in CONDITIONS};again={name:item for name,item in again.items() if item}
    episodes=json.loads((ROOT/(starts or experiment['starts'])).read_text(encoding='utf-8'))['episodes'];ids=[episode['id'] for episode in episodes]
    if starts:ids=[start for start in ids if any(start in item['by_start'] for item in data.values())]      # a check of the harness on other starts
    scenes={name:load_map(name) for name in {episode['map'] for episode in episodes}};facts={episode['id']:blur.start_facts(episode,scenes[episode['map']]) for episode in episodes}
    summary={'experiment':{'conditions':experiment['conditions'],'starts':experiment['starts'],'checkpoint':experiment['checkpoint']},
             'flights':{name:len(item['items']) for name,item in data.items()},'severity':{name:stage_counts(item['items']) for name,item in data.items()},
             'canonical_measures':{name:measures(item['items']) for name,item in data.items()}}
    clean=summary['severity']['clean']['mission_success'] if 'clean' in data else None
    if clean:
        summary['retention']={name:{**blur.retention(clean[0]/clean[1],counts['mission_success'][0]/counts['mission_success'][1]),
                                    'landing':blur.retention(summary['canonical_measures']['clean']['landing_success']/max(1,summary['canonical_measures']['clean']['landings']),
                                                             summary['canonical_measures'][name]['landing_success']/max(1,summary['canonical_measures'][name]['landings'])),
                                    'success_rate':rate(*counts['mission_success'])} for name,counts in summary['severity'].items()}
    kind=lambda i:'same_color' if facts[i['episode']]['same_color'] else 'same_shape' if facts[i['episode']]['same_shape'] else 'none'
    summary['by_band']=by_group(data,lambda i:i['band'],('near','mid','far'))
    summary['by_band_landing']=by_group({name:{'items':[i for i in item['items'] if i['task']=='land']} for name,item in data.items()},lambda i:i['band'],('near','mid','far'))
    summary['by_initial_visibility']=by_group(data,lambda i:'visible' if i['initially_visible'] else 'invisible',('visible','invisible'))
    summary['by_task']=by_group(data,lambda i:i['task'],('land','approach'))
    summary['by_distractor']=by_group(data,kind,('none','same_color','same_shape'))
    summary['by_related_object_in_first_view']=by_group(data,lambda i:'related first' if facts[i['episode']]['first_view'] else 'no related object first',('related first','no related object first'))
    summary['by_scene']=by_group(data,lambda i:i['map'],sorted(scenes))
    summary['target_selection']={name:[dict(zip(('start_holds','flights','success','ended_at_named_object','ended_at_wrong_object'),row))
                                       for row in selection(item['items'],item['episodes'],item['steps'],landing)] for name,item in data.items()}
    summary['landing']={name:landing_table(item['items']) for name,item in data.items()}
    summary['search']={name:search_table(item['items']) for name,item in data.items()}
    summary['grounding']={name:grounding_table(item['items']) for name,item in data.items()}
    summary['wrong_targets']=[row for name,item in data.items() for row in wrong_targets(name,item['items'],facts,scenes)]
    confusions={}
    for row in summary['wrong_targets']:confusions.setdefault(f'{row["named"]} -> {row["went_to"]}',{name:0 for name in data})[row['condition']]+=1
    summary['confusions']=confusions
    summary['taxonomy']={name:{label:[i['episode'] for i in item['items'] if i['class']==label] for label in blur.ALL_CLASSES} for name,item in data.items()}
    summary['canonical_stage']={name:{f'{i["failure"]} ({i["mechanism"]})':0 for i in item['items'] if i['failure']} for name,item in data.items()}
    for name,item in data.items():
        for i in item['items']:
            if i['failure']:summary['canonical_stage'][name][f'{i["failure"]} ({i["mechanism"]})']+=1
    summary['how_flights_ended']={name:{reason:sum(i['reason']==reason for i in item['items']) for reason in sorted({i['reason'] for i in item['items']})} for name,item in data.items()}
    summary['apparent_size']=size_table(data) if 'clean' in data else None
    quality,latency,checks,separation=frame_tables(data)
    summary.update(image_quality=quality,latency=latency,input_checks=checks,laplacian_variance_separation_auroc=separation)
    # Start by start.
    cells={start:{name:(item['by_start'][start]['cell'] if start in item['by_start'] else 'E_RUNTIME') for name,item in data.items()} for start in ids}
    outcome={name:{start:item['by_start'][start]['success'] for start in ids if start in item['by_start']} for name,item in data.items()}
    summary['paired']={f'clean -> {name}':blur.paired(outcome['clean'],outcome[name]) for name in CONDITIONS[1:] if name in outcome and 'clean' in outcome}
    complete=[start for start in ids if all(start in outcome.get(name,{}) for name in CONDITIONS)]
    onsets={start:blur.onset({name:outcome[name][start] for name in CONDITIONS}) for start in complete}
    summary['onset']={label:[start for start,value in onsets.items() if value['onset']==label] for label in ('NONE','LOW','MEDIUM','HIGH','CLEAN')}
    summary['non_monotonic']=[start for start,value in onsets.items() if value['non_monotonic']]
    summary['first_failed_stage_of_starts_clean_succeeded']={name:{label:sum(1 for start in complete if outcome['clean'][start] and data[name]['by_start'][start]['class']==label)
                                                                    for label in blur.ALL_CLASSES} for name in CONDITIONS[1:] if name in data and 'clean' in data}
    summary['position_in_start']={name:{str(position):[sum(i['success'] for i in item['items'] if i['blur'].get('position_in_start')==position),
                                                       sum(i['blur'].get('position_in_start')==position for i in item['items'])] for position in range(4)} for name,item in data.items()}
    # The clean flights of this run against the clean test flown before.
    historical=ROOT/experiment['historical_clean']['run']/'episodes.csv'
    if historical.exists() and 'clean' in data:
        with historical.open(encoding='utf-8') as file:before={row['episode']:row['success']=='True' for row in csv.DictReader(file)}
        now=outcome['clean']
        summary['clean_reproducibility']={'historical':share(sum(before.values()),len(before)),'this_run':share(sum(now.values()),len(now)),
                                          'same_outcome':sum(before[start]==now[start] for start in now if start in before),
                                          'succeeded_before_failed_now':[start for start in now if before.get(start) and not now[start]],
                                          'failed_before_succeeded_now':[start for start in now if start in before and not before[start] and now[start]]}
    # The second flights of the repeat subset.
    subset=[item['id'] for item in json.loads((ROOT/experiment['repeat']['subset']).read_text(encoding='utf-8'))['starts']]
    if again:
        table,detail=repeat_table(data,again,subset)
        summary['repeat']={'starts':subset,'by_condition':table,'by_start':{start:{name:detail[name].get(start) for name in detail} for start in subset},
                           'second_flight':{name:stage_counts(item['items']) for name,item in again.items()},
                           'agreement':{name:sum(1 for start in subset if start in data[name]['by_start'] and start in again[name]['by_start']
                                                 and data[name]['by_start'][start]['success']==again[name]['by_start'][start]['success']) for name in again if name in data}}
    # Attempts that ended in a harness error (flown again) and flights given up.
    attempts=[];given_up=[]
    for phase,path in (('main',root/'flights.jsonl'),('repeat',root/'repeat/flights.jsonl')):
        if path.exists():attempts+=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    errors=[entry for entry in attempts if entry['outcome']=='runtime_error']
    summary['runtime']={'attempts':len(attempts),'harness_errors':[{key:entry[key] for key in ('phase','start','condition','attempt','error')} for entry in errors],
                        'flights_never_completed':[start+' '+name for name in data for start in ids if start not in data[name]['by_start']]}
    return summary,data,again,cells,facts


def lines_of(summary,cells,facts):
    """tables.md"""
    out=[];conditions=[name for name in CONDITIONS if name in summary['severity']]
    def table(title,header,rows):
        out.extend(['',f'### {title}','','| '+' | '.join(header)+' |','|'+'---|'*len(header)]+['| '+' | '.join(str(cell) for cell in row)+' |' for row in rows])
    head=tuple(name.capitalize() for name in conditions);pair=lambda value:share(*value)
    labels=(('Mission success','mission_success'),('Acquisition','acquisition'),('Correct grounding (of acquired)','correct_grounding'),('Ended at the named object','ended_at_named_object'),
            ('Wrong target','wrong_target'),('Approach (within 20 m)','approach'),('Alignment (landings)','alignment'),('Descent started (landings)','descent'),
            ('Touchdown (landings)','touchdown'),('Stable landing (landings)','stable_landing'),('System landing (landings)','system_landing'),
            ('Strict zero action (landings)','strict_zero_action'),('End as the sentence asked','end_as_the_sentence_asked'),('Collision','collision'),('Timeout','timeout'))
    table('Severity summary',('Metric',)+head,[(label,)+tuple(pair(summary['severity'][name][key]) if key not in ('wrong_target','collision','timeout') else summary['severity'][name][key][0]
                                                          for name in conditions) for label,key in labels])
    if 'retention' in summary:
        table('Success retention against clean',('Condition','Success rate','Change (points)','Retention','Landing change (points)','Landing retention'),
              [(name,summary['retention'][name]['success_rate'],summary['retention'][name]['drop_points'],summary['retention'][name]['retention'],
                summary['retention'][name]['landing']['drop_points'],summary['retention'][name]['landing']['retention']) for name in conditions])
    for title,key,names in (('Distance band','by_band',('near','mid','far')),('Distance band, landings only','by_band_landing',('near','mid','far')),
                            ('Named object in the first view','by_initial_visibility',('visible','invisible')),('Task','by_task',('land','approach')),
                            ('Related object beside the named one or first in view','by_distractor',('none','same_color','same_shape')),
                            ('A related object in the first view, the named one outside it','by_related_object_in_first_view',('related first','no related object first')),
                            ('Scene','by_scene',tuple(summary['by_scene']))):
        table(title,('Group',)+head,[(name,)+tuple(pair(summary[key][name][condition]) for condition in conditions) for name in names])
    keys=('landings','down_view_saw_the_pad','over_the_pad','alignment','descent','correct_pad_touched','touched_another_pad','touchdown','stable_landing','system_landing','strict_zero_action','finalizer_latched','disarmed')
    table('Landing',('Metric',)+head,[(key.replace('_',' '),)+tuple(summary['landing'][name][key] for name in conditions) for key in keys]
          +[('failed after alignment',)+tuple(len(summary['landing'][name]['failed_after_alignment']) for name in conditions)])
    table('Search (starts with the named object outside the first view)',('Metric',)+head,
          [('starts',)+tuple(summary['search'][name]['starts_with_the_target_outside_the_first_view'] for name in conditions),
           ('acquired',)+tuple(summary['search'][name]['acquired'] for name in conditions),('mission success',)+tuple(summary['search'][name]['success'] for name in conditions),
           ('median decision of first sighting',)+tuple((summary['search'][name]['first_view_step'] or {}).get('median') for name in conditions),
           ('median yaw before first sighting (deg)',)+tuple((summary['search'][name]['yaw_before_first_view_deg'] or {}).get('median') for name in conditions)])
    table('Grounding transition',('Metric',)+head,[(key.replace('_',' '),)+tuple(summary['grounding'][name][key] for name in conditions)
                                                  for key in ('in_view','grounded','grounded_on_first_view','ended_at_named_object')])
    table('Failure class (first stage not passed)',('Class',)+head,[(label,)+tuple(len(summary['taxonomy'][name][label]) for name in conditions) for label in blur.ALL_CLASSES])
    table('How flights ended',('Reason',)+head,[(reason,)+tuple(summary['how_flights_ended'][name].get(reason,0) for name in conditions)
                                              for reason in sorted({reason for name in conditions for reason in summary['how_flights_ended'][name]})])
    if summary['wrong_targets']:
        table('Flights that ended at a wrong object',('Condition','Start','Named','Went to','Relation','Related object first in view','Named object seen','Band','Class'),
              [(row['condition'],row['start'],row['named'],row['went_to'],row['relation'],', '.join(row['related_object_in_first_view']) or '-',row['named_object_ever_seen'],row['band'],row['class'])
               for row in summary['wrong_targets']])
    if summary.get('paired'):
        table('Paired starts against clean',('Comparison','S -> S','S -> F','F -> S','F -> F','McNemar exact p'),
              [(name,value['S_S'],value['S_F'],value['F_S'],value['F_F'],round(value['mcnemar_exact_p'],4)) for name,value in summary['paired'].items()])
        table('Lowest severity at which a start failed',('Onset','Starts'),[(label,len(summary['onset'][label])) for label in ('NONE','LOW','MEDIUM','HIGH','CLEAN')]
              +[('non-monotonic (a higher severity succeeded again)',len(summary['non_monotonic']))])
        table('Failure class of starts that succeeded clean',('Class',)+tuple(name.capitalize() for name in conditions[1:]),
              [(label,)+tuple(summary['first_failed_stage_of_starts_clean_succeeded'][name][label] for name in conditions[1:]) for label in blur.ALL_CLASSES])
    if summary.get('repeat'):
        value=summary['repeat']['by_condition']
        table('Repeat subset: two flights per start',('Outcome',)+tuple(name.capitalize() for name in value),
              [(key,)+tuple(value[name][key] for name in value) for key in ('2/2','1/2','0/2')]+[('same outcome in both flights',)+tuple(summary['repeat']['agreement'][name] for name in value)])
        table('Repeat subset by start (successes of two flights)',('Start','Kind')+tuple(name.capitalize() for name in value),
              [(start,'same colour' if facts[start]['same_color'] else 'same shape')+tuple(summary['repeat']['by_start'][start][name] for name in value) for start in summary['repeat']['starts']])
    table('Laplacian variance of the frames the policy was given',('Camera','Statistic',)+head,
          [(camera,statistic)+tuple(round(summary['image_quality'][name]['input'][camera]['laplacian_variance'][statistic],1) for name in conditions)
           for camera in blur.CAMERAS for statistic in ('median','p5','p95')])
    table('Separation of adjacent severities by Laplacian variance (AUROC; 1 = no overlap)',('Pair','Front','Down'),
          [(name,round(value['front'],3),round(value['down'],3)) for name,value in summary['laplacian_variance_separation_auroc'].items()])
    table('Latency',('Condition','Blur ms (median / p95)','Inference ms (median / p95)','Decision cycle s (mean / median / p95)'),
          [(name,f'{summary["latency"][name]["blur_ms"]["median"]:.2f} / {summary["latency"][name]["blur_ms"]["p95"]:.2f}',
            f'{summary["latency"][name]["inference_ms"]["median"]:.0f} / {summary["latency"][name]["inference_ms"]["p95"]:.0f}',
            f'{summary["latency"][name]["decision_cycle_s"]["mean"]:.3f} / {summary["latency"][name]["decision_cycle_s"]["median"]:.3f} / {summary["latency"][name]["decision_cycle_s"]["p95"]:.3f}')
           for name in conditions])
    table('Input checks',('Condition','Decisions','Front identical to raw','Down identical to raw'),
          [(name,summary['input_checks'][name]['decisions'],summary['input_checks'][name]['front_input_identical_to_raw'],summary['input_checks'][name]['down_input_identical_to_raw']) for name in conditions])
    table('Every start',('Start','Kind','Band','Task')+head,
          [(start,'same colour' if facts[start]['same_color'] else 'same shape',facts[start]['band'],facts[start]['task'])+tuple(cells[start][name] for name in conditions) for start in cells])
    return out


def write(summary,data,again,cells,facts,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=1)+'\n',encoding='utf-8',newline='\n')
    (output/'tables.md').write_text('\n'.join(lines_of(summary,cells,facts)).lstrip('\n')+'\n',encoding='utf-8',newline='\n')
    skip=('blur','grounded_on_first_view','views_before_grounding','turns_deg','twin_of')
    rows=[{'phase':phase,'condition':name,**{key:value for key,value in item.items() if key not in skip},'position_in_start':item['blur'].get('position_in_start'),
           'flight_index':item['blur'].get('flight_index'),'same_color':facts[item['episode']]['same_color'],'same_shape':facts[item['episode']]['same_shape'],
           'related_first':','.join(facts[item['episode']]['first_view'])}
          for phase,group in (('main',data),('repeat',again)) for name,flights in group.items() for item in flights['items']]
    if rows:
        with (output/'episodes.csv').open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    with (output/'paired_starts.csv').open('w',newline='',encoding='utf-8') as file:
        conditions=[name for name in CONDITIONS if name in data];writer=csv.writer(file,lineterminator='\n')
        writer.writerow(['start','band','task','same_color','same_shape','initially_invisible']+conditions+['onset','non_monotonic']+[f'repeat_{name}' for name in again])
        for start in cells:
            complete=all(start in data[name]['by_start'] for name in conditions) and len(conditions)==4
            value=blur.onset({name:data[name]['by_start'][start]['success'] for name in conditions}) if complete else {'onset':'','non_monotonic':''}
            writer.writerow([start,facts[start]['band'],facts[start]['task'],facts[start]['same_color'],facts[start]['same_shape'],facts[start]['initially_invisible']]
                            +[cells[start][name] for name in conditions]+[value['onset'],value['non_monotonic']]
                            +[again[name]['by_start'][start]['cell'] if start in again[name]['by_start'] else '' for name in again])


def figures(summary,data,cells,output):
    """Small plots for the document."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);conditions=[name for name in CONDITIONS if name in data];colours={'clean':'#2c7fb8','low':'#7fcdbb','medium':'#fdae61','high':'#d7191c'}
    # Stage by stage: how many flights passed each stage, per condition.
    stages=(('acquisition','Acquired'),('correct_grounding','Grounded'),('approach','Approached'),('alignment','Aligned\n(landings)'),('descent','Descent\n(landings)'),
            ('touchdown','Touchdown\n(landings)'),('system_landing','System landing\n(landings)'),('mission_success','Mission\nsuccess'))
    figure,axis=plt.subplots(figsize=(10,4.2));width=.2
    for offset,name in enumerate(conditions):
        values=[100*summary['severity'][name][key][0]/max(1,summary['severity'][name][key][1]) for key,_ in stages]
        axis.bar([index+(offset-1.5)*width for index in range(len(stages))],values,width,label=name,color=colours[name])
    axis.set_xticks(range(len(stages)));axis.set_xticklabels([label for _,label in stages],fontsize=8);axis.set_ylabel('% of flights (of landings where marked)');axis.set_ylim(0,105)
    axis.legend(ncol=4,fontsize=8,loc='lower left');axis.set_title('Stages passed, by blur severity (48 starts, one flight each)',fontsize=10);figure.tight_layout()
    figure.savefig(output/'stages_by_severity.png',dpi=110);plt.close(figure)
    # Success by distance band.
    figure,axis=plt.subplots(figsize=(6,3.6));bands=('near','mid','far')
    for offset,name in enumerate(conditions):
        axis.bar([index+(offset-1.5)*width for index in range(3)],[100*summary['by_band'][band][name][0]/max(1,summary['by_band'][band][name][1]) for band in bands],width,label=name,color=colours[name])
    axis.set_xticks(range(3));axis.set_xticklabels(('10-20 m','20-32 m','32-44 m'));axis.set_ylabel('mission success %');axis.set_ylim(0,105);axis.legend(ncol=4,fontsize=8,loc='lower left')
    axis.set_title('Mission success by start distance (16 starts per band)',fontsize=10);figure.tight_layout();figure.savefig(output/'success_by_distance.png',dpi=110);plt.close(figure)
    # Every start in every condition.
    codes=sorted({cell for row in cells.values() for cell in row.values()});palette={'S':'#1a9641','E_RUNTIME':'#bbbbbb'}
    shades=plt.get_cmap('tab10');others=[code for code in codes if code not in palette]
    for index,code in enumerate(others):palette[code]=shades(index%10)
    figure,axis=plt.subplots(figsize=(7,9.5))
    for row,(start,values) in enumerate(cells.items()):
        for column,name in enumerate(conditions):
            axis.add_patch(plt.Rectangle((column,row),1,1,color=palette[values[name]],ec='white',lw=.5))
    axis.set_xlim(0,len(conditions));axis.set_ylim(len(cells),0);axis.set_xticks([index+.5 for index in range(len(conditions))]);axis.set_xticklabels(conditions)
    axis.set_yticks([index+.5 for index in range(len(cells))]);axis.set_yticklabels([start.replace('ct-','') for start in cells],fontsize=5.5)
    axis.legend(handles=[plt.Rectangle((0,0),1,1,color=palette[code]) for code in codes],labels=codes,fontsize=6,loc='upper left',bbox_to_anchor=(1.01,1))
    axis.set_title('Each start in each condition (S = mission success)',fontsize=9);figure.tight_layout();figure.savefig(output/'paired_starts.png',dpi=110);plt.close(figure)
    # Sharpness of the frames the policy was given.
    figure,axes=plt.subplots(1,2,figsize=(9,3.4))
    for axis,camera in zip(axes,blur.CAMERAS):
        values=[[max(row['quality']['input'][camera]['laplacian_variance'],1e-3) for rows in data[name]['extra'].values() for row in rows] for name in conditions]
        parts=axis.boxplot(values,labels=conditions,showfliers=False,patch_artist=True)
        for patch,name in zip(parts['boxes'],conditions):patch.set_facecolor(colours[name])
        axis.set_yscale('log');axis.set_title(f'{camera.capitalize()} camera',fontsize=9);axis.set_ylabel('Laplacian variance of the model input (log)')
    figure.suptitle('Sharpness of the frames the policy was given, every decision',fontsize=10);figure.tight_layout();figure.savefig(output/'laplacian_variance.png',dpi=110);plt.close(figure)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='outputs/failures/gaussian_blur');parser.add_argument('--output');parser.add_argument('--figures',action='store_true')
    parser.add_argument('--starts',help='another starts file, for a check of the harness')
    args=parser.parse_args();root=Path(args.root);root=root if root.is_absolute() else ROOT/root;output=Path(args.output) if args.output else root/'summary'
    summary,data,again,cells,facts=analyse(root,args.starts);write(summary,data,again,cells,facts,output)
    if args.figures:figures(summary,data,cells,output)
    print('\n'.join(lines_of(summary,cells,facts)[:60]))


if __name__=='__main__':main()
