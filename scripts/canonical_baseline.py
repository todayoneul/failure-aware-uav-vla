"""The canonical clean baseline: its start sets, its gates and its tables; no simulator, no model.

Mission: "Find the blue landing pad and land on it." from at most 44 m, flown by the second Gen-v3
checkpoint as it is, with the low-level landing finalizer. Nothing is trained here.

  plan validation|test     write configs/gen_v3_canonical_44m_<set>.json once; the test needs the validation file
  ids                      the ten smoke starts
  gate smoke|representative|test --run DIR --teacher DIR [--output JSON]
                           exit 0 when passed. `test` reads the test set by the representative thresholds.
  report MODEL:SET=DIR ... --teacher SET=DIR ... --output DIR
                           tables and files for the document

A landing is read three ways that are kept apart (scripts/visual_search.py, landing_summary):
  touchdown        the named pad, inside its landing region, a soft contact
  stable physical  and standing still on it for the finalizer's duration
  strict           and the policy gave the zero action by itself (the measure Gen-v3 was judged by)
The mission's result is the system's: a stable physical landing on the named pad that the finalizer
latched and disarmed on, with no other contact. An approach is judged as before.
A start the teacher did not finish in the simulator is left out of every model.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_visual_search import load
from scripts.freeze_gen_v3 import read,rate,percentile,markdown,STAGES
from scripts.freeze_generalization_results import picture
from scripts.gate_gen_v3 import judge
from src.visual_search.episodes import load_config
from src.visual_search.maps import load_landing
from src.visual_search import canonical
from src.aerovla_oft.spec import load_config as load_oft_config


def flown(directory,solvable=None):
    """Every flown episode of a run with the plan's own fields kept: (records, ids with a runner error, ids left out)."""
    _,episodes,steps=load(directory);data=json.loads((Path(directory)/'results.json').read_text());landing=load_landing();geometries={};maps={};items=[]
    kept=[episode for episode in episodes if solvable is None or episode['id'] in solvable]
    for episode in kept:
        record=read(episode,steps[episode['id']],data['config']['success_radius_m'],landing,geometries,maps)
        items.append({'episode':episode['id'],'map':episode['map'],'layout':episode['layout'],'band':episode.get('band'),'role':episode.get('role'),
                      'twin_of':episode.get('twin_of',''),'instruction':episode['instruction'],**record})
    return items,[episode['id'] for episode in data['episodes'] if episode.get('error')],[episode['id'] for episode in episodes if episode not in kept],(episodes,steps)


def solved(directory):
    """Ids the teacher finished in the simulator."""
    return {episode['id'] for episode in json.loads((Path(directory)/'results.json').read_text())['episodes'] if not episode.get('error') and episode['success']}


def measures(items):
    """The numbers the gates and the report use."""
    land=[i for i in items if i['task']=='land'];approach=[i for i in items if i['task']=='approach'];seen=[i for i in items if i['acquired']]
    return {'flights':len(items),'success':sum(i['success'] for i in items),'landings':len(land),'landing_success':sum(i['success'] for i in land),
            'approaches':len(approach),'approach_success':sum(i['success'] for i in approach),
            'grounded':sum(i['grounded'] for i in seen),'in_view':len(seen),'terminal_as_asked':sum(i['terminal_as_asked'] for i in items),
            'wrong_target':sum(i['wrong_target'] for i in items),'collisions':sum(i['collision'] for i in items)}


def checks(name,gates,items,errors,config):
    m=measures(items);total=max(1,m['flights'])
    if name=='smoke':
        wanted=set(canonical.smoke_ids(config,json.loads(canonical.FILES['validation'].read_text(encoding='utf-8'))['episodes']))
        items=[i for i in items if i['episode'] in wanted];m=measures(items)
        return items,{'flights':(m['flights'],'>=',gates['episodes']),'successes':(m['success'],'>=',gates['success_min']),
                      'landing successes (system)':(m['landing_success'],'>=',gates['landing_success_min']),
                      'successes from 32-44 m':(sum(i['success'] for i in items if i['band']=='far'),'>=',gates['far_band_success_min']),
                      'collisions':(m['collisions'],'<=',gates['collisions_max']),'episodes with a runner error':(len(errors),'<=',gates['errors_max'])}
    return items,{'flights':(m['flights'],'>=',gates['episodes_min']),'success rate':(round(m['success']/total,3),'>=',gates['success_rate_min']),
                  'landing success rate (system: stable physical landing on the named pad, latched and disarmed)':(round(m['landing_success']/max(1,m['landings']),3),'>=',gates['landing_success_rate_min']),
                  'grounded when the target came into view':(round(m['grounded']/max(1,m['in_view']),3),'>=',gates['grounded_rate_min']),
                  'end of flight as the sentence asked':(round(m['terminal_as_asked']/total,3),'>=',gates['terminal_as_asked_rate_min']),
                  'flights that ended at a wrong object':(m['wrong_target'],'<=',gates['wrong_target_max']),
                  'collisions':(m['collisions'],'<=',gates['collisions_max']),'episodes with a runner error':(len(errors),'<=',gates['errors_max'])}


def landing_rows(items):
    """Landing flights by start band: correct pad, approach, alignment, the three readings, the system's result."""
    land=[i for i in items if i['task']=='land'];rows=[]
    for name,group in [(band,[i for i in land if i['band']==band]) for band in ('near','mid','far')]+[('all',land)]:
        if group:rows.append((name,rate(group,'correct_target'),rate(group,'approached'),rate(group,'aligned'),rate(group,'touchdown_success'),
                              rate(group,'stable_physical_landing'),rate(group,'system_landing'),rate(group,'strict_policy_zero_action')))
    return rows


def report(args,config):
    output=Path(args.output);(output/'failures').mkdir(parents=True,exist_ok=True);summary={'models':{},'left_out':{}};rows_out=[]
    teacher={item.split('=',1)[0]:solved(item.split('=',1)[1]) for item in args.teacher}
    for item in args.runs:
        label,directory=item.split('=',1);model,level=label.rsplit(':',1);items,errors,left,(episodes,steps)=flown(directory,teacher.get(level))
        summary['left_out'][label]=left;m=measures(items);land=[i for i in items if i['task']=='land'];touched=[i for i in land if i['touchdown_on_target']]
        print(f'\\n# {model}, {level}');print(json.dumps(m))
        markdown('Landings by start distance',('Band','Correct target','Approach','Alignment','Touchdown','Stable physical landing','System landing','Strict zero-action'),landing_rows(items))
        by_role=[(role,rate([i for i in items if i['role']==role],'success')) for role in dict.fromkeys(i['role'] for i in items)]
        markdown('By kind of start',('Start','Success'),by_role)
        by_band=[(band,rate([i for i in items if i['band']==band],'success'),rate([i for i in items if i['band']==band and i['task']=='land'],'success'),
                  rate([i for i in items if i['band']==band and i['task']=='approach'],'success')) for band in ('near','mid','far')]
        markdown('By band',('Band','All','Landing','Approach'),by_band)
        failed=[i for i in items if i['failure']]
        stages=[(stage,sum(i['failure']==stage for i in failed),', '.join(sorted({i['mechanism'] for i in failed if i['failure']==stage})),
                 ' '.join(i['episode'] for i in failed if i['failure']==stage)) for stage in STAGES]
        markdown('Failures by the first stage not passed',('Stage','Count','How','Episodes'),[row for row in stages if row[1]])
        errors_m=[i['touchdown_error_m'] for i in touched]
        precision={'touchdowns_on_the_named_pad':len(touched),
                   'horizontal_error_m':{'median':statistics.median(errors_m) if errors_m else None,'p90':percentile(errors_m,.9),'max':max(errors_m,default=None)},
                   'vertical_speed_mps':{'median':statistics.median(i['touchdown_vertical_mps'] for i in touched) if touched else None,'max':max((i['touchdown_vertical_mps'] for i in touched),default=None)},
                   'stable_duration_s':{'median':statistics.median(i['stable_duration_s'] for i in touched if i['stable_duration_s'] is not None) if any(i['stable_duration_s'] is not None for i in touched) else None}}
        print('\\nTouchdown:',json.dumps(precision))
        summary['models'][label]={'measures':m,'landings':[dict(zip(('band','correct_target','approach','alignment','touchdown','stable_physical_landing','system_landing','strict_zero_action'),row)) for row in landing_rows(items)],
                                  'by_role':dict(by_role),'by_band':[dict(zip(('band','all','landing','approach'),row)) for row in by_band],
                                  'failures':{stage:{'count':count,'how':how,'episodes':names.split()} for stage,count,how,names in stages if count},
                                  'touchdown':precision,'episodes_with_errors':errors,
                                  'rates':{'correct_target':rate(items,'correct_target'),'approach':rate(items,'approached'),'touchdown':rate(land,'touchdown_success'),
                                           'stable_physical_landing':rate(land,'stable_physical_landing'),'system_landing':rate(land,'system_landing'),
                                           'strict_policy_zero_action':rate(land,'strict_policy_zero_action'),'wrong_target':m['wrong_target'],'collisions':m['collisions']}}
        by_id={episode['id']:episode for episode in episodes}
        for record in items:
            rows_out.append({'model':model,'set':level,**{k:v for k,v in record.items()}})
            if record['failure'] and model!='Teacher':
                name=f'{model}_{level}_{record["failure"]}_{record["episode"]}'.replace(' ','_');trace_rows=steps[record['episode']]
                trace=[{key:row.get(key) for key in ('step','distance_m','bearing_deg','height_m','front_seen','down_seen','over','landed','finalizer','action','executed')} for row in trace_rows]
                (output/'failures'/f'{name}.json').write_text(json.dumps({**record,'trace':trace},indent=1))
                picture(by_id[record['episode']],trace_rows,output/'failures'/f'{name}.jpg',f'{model} {level} {record["failure"]}: {record["episode"]}')
    with (output/'episodes.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows_out[0]));writer.writeheader();writer.writerows(rows_out)
    (output/'summary.json').write_text(json.dumps(summary,indent=1))


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('plan').add_argument('which',choices=('validation','test'));commands.add_parser('ids')
    gate=commands.add_parser('gate');gate.add_argument('name',choices=('smoke','representative','test'));gate.add_argument('--run',required=True)
    gate.add_argument('--teacher',required=True,help="the teacher's run over the same set");gate.add_argument('--output')
    rep=commands.add_parser('report');rep.add_argument('runs',nargs='+',help='MODEL:SET=DIRECTORY');rep.add_argument('--teacher',nargs='*',default=[],help="SET=DIRECTORY of the teacher's run")
    rep.add_argument('--output',required=True)
    args=parser.parse_args();config=load_config()
    if args.command=='plan':
        path=canonical.FILES[args.which]
        if path.exists():raise SystemExit(f'{path} exists; the canonical {args.which} starts are fixed and are not regenerated')
        episodes=canonical.build(config,load_oft_config(),args.which)
        path.write_text(json.dumps(canonical.describe(config,args.which,episodes),indent=1)+chr(10),encoding='utf-8',newline=chr(10))
        print(args.which,len(episodes),json.dumps({band:sum(e['band']==band for e in episodes) for band in config['canonical']['bands_m']}),
              'landings',sum(e['task']=='land' for e in episodes),'seeds',min(e['seed'] for e in episodes),max(e['seed'] for e in episodes));return
    if args.command=='ids':
        print(' '.join(canonical.smoke_ids(config,json.loads(canonical.FILES['validation'].read_text(encoding='utf-8'))['episodes'])));return
    if args.command=='report':report(args,config);return
    gates=config['canonical']['gates']['smoke' if args.name=='smoke' else 'representative']
    items,errors,left,_=flown(args.run,solved(args.teacher));items,lines=checks(args.name,gates,items,errors,config);table,passed=judge(lines)
    result={'gate':args.name,'run':args.run,'passed':passed,'checks':table,**measures(items),'left_out_because_the_teacher_did_not_finish':left,
            'failures':{i['episode']:f'{i["failure"]} ({i["mechanism"]})' for i in items if i['failure']}}
    print(json.dumps(result,indent=1))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=1))
    print(f'CANONICAL {args.name.upper()}: {"PASSED" if passed else "NOT PASSED"}')
    sys.exit(0 if passed else 1)


if __name__=='__main__':main()
