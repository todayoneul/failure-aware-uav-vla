"""Language-vision grounding architectures: start sets, the pilot's verdict and the comparison; no simulator, no model.

  plan pilot|validation     write configs/grounding_architecture_<set>.json once
  critical                  the validation starts that are flown a second time (target selection)
  judge NAME --runs DIR DIR --baseline DIR DIR --teacher DIR [--regression JSON] [--output JSON]
                            the pilot's lines for one model against the Gen-v3c flights of the same starts; exit 0 when
                            every line holds. Without --regression the regression line is left open and the verdict
                            says so (exit 2): the regression starts are flown only for a model that holds the others.
  compare NAME=DIR,DIR ... --teacher DIR --output DIR [--cost NAME=JSON ...]
                            the comparison table of every model flown over the pilot set
  gate smoke|representative --run DIR --teacher DIR [--output JSON]
                            the canonical gates on the architecture validation set (first flight of every start)

Every model flies each of the sixteen pilot starts twice; a start is counted by how many of its two flights ended at
the object the sentence named.
"""
import argparse
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.canonical_baseline import flown,solved,measures
from scripts.freeze_gen_v3 import markdown
from scripts.gate_gen_v3 import judge
from scripts import gen_v3c as gen_v3c_tool
from src.visual_search.episodes import load_config
from src.visual_search.maps import load_map
from src.visual_search import grounding_arch,gen_v3c
from src.aerovla_oft.spec import load_config as load_oft_config

FIRST_VIEW=('same_color','same_shape','first_view')


def pilot_metrics(runs,teacher):
    """Everything the pilot's lines and the comparison read, over the flights of one model (one folder per repeat)."""
    plan={episode['id']:episode for episode in json.loads(grounding_arch.FILES['pilot'].read_text(encoding='utf-8'))['episodes']}
    scenes={};items=[];errors=[];latency=[];periods=[]
    for index,directory in enumerate(runs):
        mine,failed,_,(episodes,steps)=flown(directory,solved(teacher));errors+=failed
        for item in mine:
            episode=plan[item['episode']];scene=scenes.setdefault(episode['map'],load_map(episode['map']))
            items.append({**item,'run':index,'group':episode['role'],'relation':gen_v3c.relation_of(scene,episode['target'],episode['distractor'])})
            latency+=[row['inference_ms'] for row in steps[item['episode']] if row.get('inference_ms') is not None]
        periods+=[episode['mean_decision_s'] for episode in episodes if episode.get('mean_decision_s')]
    count=lambda chosen,key='correct_target':sum(bool(i[key]) for i in chosen)
    land=[i for i in items if i['task']=='land'];reached=[i for i in land if i['correct_target']];approach=[i for i in items if i['task']=='approach']
    first=[i for i in items if i['group'] in FIRST_VIEW];lost=[i for i in items if i['group']=='lost']
    starts={}
    for item in items:starts.setdefault(item['episode'],[]).append(bool(item['correct_target']))
    spread={f'{k}/{len(runs)}':sum(sum(flags)==k for flags in starts.values()) for k in range(len(runs),-1,-1)}
    return {'runs':list(runs),'flights':len(items),'starts':len(starts),'errors':len(errors),'success':count(items,'success'),'correct_target':count(items),
            'wrong_target':count(items,'wrong_target'),'collisions':count(items,'collision'),'acquired':count(items,'acquired'),
            'terminal_as_asked':count(items,'terminal_as_asked'),'approach_touchdowns':count(approach,'landed'),
            'same_color':[count([i for i in items if i['relation']=='same_color']),sum(i['relation']=='same_color' for i in items)],
            'same_shape':[count([i for i in items if i['relation']=='same_shape']),sum(i['relation']=='same_shape' for i in items)],
            'first_view_rejected':[sum(not i['wrong_target'] for i in first),len(first)],'first_view_correct':[count(first),len(first)],
            'lost_reacquired':[count(lost),len(lost)],'landings_at_named_pad':len(reached),'system_landings':count(reached,'system_landing'),
            'landing_rate':count(reached,'system_landing')/len(reached) if reached else None,'strict_zero_action':count(reached,'strict_policy_zero_action'),
            'finalizer_rescued':count(reached,'finalizer_rescued_success'),'by_start':{name:sum(flags) for name,flags in starts.items()},'consistency':spread,
            'starts_always_correct':spread[f'{len(runs)}/{len(runs)}'],'starts_never_correct':spread[f'0/{len(runs)}'],
            'by_group':{group:[count([i for i in items if i['group']==group]),sum(i['group']==group for i in items)] for group in dict.fromkeys(i['group'] for i in items)},
            'wrong_objects':sorted(f'{i["episode"]}#{i["run"]+1}: {i["wrong_object"]}' for i in items if i['wrong_target']),
            'failures':sorted(f'{i["episode"]}#{i["run"]+1}: {i["failure"]} ({i["mechanism"]})' for i in items if i['failure']),
            'inference_ms':sum(latency)/len(latency) if latency else None,'decision_s':sum(periods)/len(periods) if periods else None}


def pilot_lines(model,baseline,rule,regression=None):
    """The pilot's lines for one model: {label: (value, comparison, limit)}. The regression line is present only when flown."""
    limit=min(rule['wrong_target_max'],baseline['wrong_target']-rule['wrong_target_drop_min'])
    if rule['wrong_target_halved']:limit=min(limit,math.floor(baseline['wrong_target']/2))
    lines={'flights':(model['flights'],'>=',baseline['flights']),
           'flights that ended at a wrong object':(model['wrong_target'],'<=',rule['wrong_target_max']),
           'flights at a wrong object, against the baseline (at least two fewer and at most half)':(model['wrong_target'],'<=',limit),
           'starts never correct':(model['starts_never_correct'],'<=',rule['starts_never_correct_max']),
           'collisions':(model['collisions'],'<=',baseline['collisions']+rule['collisions_above_baseline_max']),
           'end of flight as the sentence asked':(model['terminal_as_asked'],'>=',baseline['terminal_as_asked']-rule['terminal_as_asked_below_baseline_max']),
           'approach flights that touched down':(model['approach_touchdowns'],'<=',rule['approach_touchdowns_max']),
           'successes':(model['success'],'>=',baseline['success']-rule['success_below_baseline_max']),
           'landings completed among flights that went to the named pad':(round(model['landing_rate'] or 0.,3),'>=',rule['landing_rate_min']),
           'flights in which the named object came into view':(model['acquired'],'>=',baseline['acquired']-rule['acquired_below_baseline_max']),
           'episodes with a runner error':(model['errors'],'<=',rule['errors_max'])}
    if regression is not None:
        mine=regression['groups']['all'];names=list(mine)
        lines['regression starts: successes']=(mine[names[-1]]['success'],'>=',mine['Gen-v3c']['success']-rule['regression']['success_below_gen_v3c_max'])
        lines['regression starts: collisions']=(mine[names[-1]]['collisions'],'<=',rule['regression']['collisions_max'])
    return lines


def compare(args,config):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);models={};costs={}
    for item in args.cost:
        name,path=item.split('=',1);costs[name]=json.loads(Path(path).read_text())
    for item in args.models:
        name,folders=item.split('=',1);models[name]=pilot_metrics(folders.split(','),args.teacher)
    share=lambda pair:f'{pair[0]}/{pair[1]}'
    rows=[('Overall success',lambda m:f'{m["success"]}/{m["flights"]}'),('Ended at the named object',lambda m:f'{m["correct_target"]}/{m["flights"]}'),
          ('Wrong target',lambda m:m['wrong_target']),('Same colour, another shape',lambda m:share(m['same_color'])),('Same shape, another colour',lambda m:share(m['same_shape'])),
          ('First-visible related object rejected',lambda m:share(m['first_view_rejected'])),('Temporary loss, named object reacquired',lambda m:share(m['lost_reacquired'])),
          ('Starts correct 2/2',lambda m:m['consistency'].get('2/2')),('Starts correct 1/2',lambda m:m['consistency'].get('1/2')),('Starts correct 0/2',lambda m:m['consistency'].get('0/2')),
          ('Collision',lambda m:m['collisions']),('End of flight as asked',lambda m:f'{m["terminal_as_asked"]}/{m["flights"]}'),
          ('Landings completed / went to the named pad',lambda m:f'{m["system_landings"]}/{m["landings_at_named_pad"]}'),
          ('Inference per decision (ms)',lambda m:None if m['inference_ms'] is None else round(m['inference_ms']))]
    table=[(label,)+tuple(value(models[name]) for name in models) for label,value in rows]
    for label,key in (('Peak memory at inference (GiB)','inference_peak_GiB'),('Peak memory in training (GiB)','training_peak_GiB'),('Added parameters','added_parameters'),
                      ('Trainable parameters','trainable_parameters')):
        table.append((label,)+tuple(costs.get(name,{}).get(key,'-') for name in models))
    markdown('Architecture pilot: every model over the same sixteen starts, twice',('Metric',)+tuple(models),table)
    starts=json.loads(grounding_arch.FILES['pilot'].read_text(encoding='utf-8'))['episodes']
    markdown('Flights that ended at the named object, by start (of 2)',('Start','Group','Named <- related','Where')+tuple(models),
             [(e['id'],e['role'],f'{e["target"]} <- {e["distractor"]}',e['where'])+tuple(models[name]['by_start'].get(e['id'],'-') for name in models) for e in starts])
    (output/'pilot_comparison.json').write_text(json.dumps({'models':models,'costs':costs},indent=1))


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('plan').add_argument('which',choices=('pilot','validation'));commands.add_parser('critical')
    one=commands.add_parser('judge');one.add_argument('name');one.add_argument('--runs',nargs='+',required=True);one.add_argument('--baseline',nargs='+',required=True)
    one.add_argument('--teacher',required=True);one.add_argument('--regression');one.add_argument('--output')
    many=commands.add_parser('compare');many.add_argument('models',nargs='+',help='NAME=DIR,DIR');many.add_argument('--teacher',required=True);many.add_argument('--output',required=True)
    many.add_argument('--cost',nargs='*',default=[],help='NAME=JSON with the cost fields of a model')
    gate=commands.add_parser('gate');gate.add_argument('name',choices=('smoke','representative'));gate.add_argument('--run',required=True);gate.add_argument('--teacher',required=True)
    gate.add_argument('--output')
    args=parser.parse_args();config=load_config();line=chr(10)
    if args.command=='plan':
        path=grounding_arch.FILES[args.which]
        if path.exists():raise SystemExit(f'{path} exists; the {args.which} starts are fixed and are not regenerated')
        if args.which=='validation' and not grounding_arch.FILES['pilot'].exists():raise SystemExit('the pilot starts are written first')
        oft=load_oft_config();episodes=grounding_arch.build_pilot(config,oft) if args.which=='pilot' else grounding_arch.build_validation(config,oft)
        path.write_text(json.dumps(grounding_arch.describe(config,args.which,episodes),indent=1)+line,encoding='utf-8',newline=line)
        print(args.which,len(episodes),json.dumps({'by_role':gen_v3c_tool.histogram(episodes,lambda e:e['role']),'by_band':gen_v3c_tool.histogram(episodes,lambda e:e['band']),
                                                   'by_layout':gen_v3c_tool.histogram(episodes,lambda e:f'{e["map"]}/{e["layout"]}'),'landings':sum(e['task']=='land' for e in episodes),
                                                   'seeds':[min(e['seed'] for e in episodes),max(e['seed'] for e in episodes)]}));return
    if args.command=='critical':
        print(' '.join(grounding_arch.critical_ids(json.loads(grounding_arch.FILES['validation'].read_text(encoding='utf-8'))['episodes'])));return
    if args.command=='compare':compare(args,config);return
    if args.command=='gate':
        items,errors,left,_=flown(args.run,solved(args.teacher))
        items,lines=gen_v3c_tool.checks(args.name,config,items,errors,validation=grounding_arch.FILES['validation']);table,passed=judge(lines)
        result={'gate':args.name,'run':args.run,'passed':passed,'checks':table,**measures(items),'left_out_because_the_teacher_did_not_finish':left,
                'failures':{i['episode']:f'{i["failure"]} ({i["mechanism"]})' for i in items if i['failure']}}
        print(json.dumps(result,indent=1))
        if args.output:Path(args.output).write_text(json.dumps(result,indent=1))
        print(f'ARCHITECTURE {args.name.upper()}: {"PASSED" if passed else "NOT PASSED"}');sys.exit(0 if passed else 1)
    rule=config['grounding_architecture']['pilot']['pass'];model=pilot_metrics(args.runs,args.teacher);baseline=pilot_metrics(args.baseline,args.teacher)
    regression=json.loads(Path(args.regression).read_text()) if args.regression else None
    table,passed=judge(pilot_lines(model,baseline,rule,regression))
    verdict='PASSED' if passed and regression is not None else 'HOLDS SO FAR, REGRESSION NOT FLOWN' if passed else 'NOT PASSED'
    result={'model':args.name,'verdict':verdict,'passed':bool(passed and regression is not None),'lines_hold':passed,'regression_flown':regression is not None,'checks':table,
            'model_metrics':model,'baseline_metrics':baseline}
    print(json.dumps({key:value for key,value in result.items() if key not in ('model_metrics','baseline_metrics')},indent=1))
    print(json.dumps({key:model[key] for key in ('flights','success','correct_target','wrong_target','collisions','consistency','wrong_objects')}))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=1))
    print(f'ARCHITECTURE PILOT, {args.name}: {verdict}')
    sys.exit(0 if result['passed'] else 2 if passed else 1)


if __name__=='__main__':main()
