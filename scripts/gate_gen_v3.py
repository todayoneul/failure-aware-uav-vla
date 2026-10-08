"""Go / no-go gates in front of the Gen-v3 full evaluation; no simulator, no model.

The 156 fresh test starts are flown only after three smaller checks have passed, in this order:

  pilot           a checkpoint trained on a dozen episodes alone, flown from the same poses: does the sentence
                  decide between stopping in the air and landing?
  smoke           the final checkpoint on 10 validation starts, one of each kind: does anything work at all?
  representative  the final checkpoint on every validation start (the 10 included): is it worth the full run?

Smoke and representative starts come from the validation split of the training plan: training scenes, seeds
no training episode used. They are not part of the fresh test set, so a gate that fails and sends the work
back to training tells nothing about the test set. The thresholds are in configs/visual_search.json
(gen_v3.gates) and were fixed before the checkpoint they judge existed.

  gate_gen_v3.py ids                                  print the 10 smoke starts
  gate_gen_v3.py pilot --run DIR [--probe JSON]       exit 0 when the gate is passed, 1 when it is not
  gate_gen_v3.py smoke|representative --run DIR --teacher DATASET_ROOT [--output JSON]
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_visual_search import load
from scripts.freeze_gen_v3 import read
from src.visual_search.episodes import load_config
from src.visual_search.maps import load_landing

PLAN=ROOT/'datasets/projectairsim_visual_search/generalization_v3_added/plan.json'


def smoke_ids(config,plan):
    """The first validation start of each kind the smoke gate names."""
    return [next(e['id'] for e in plan['val'] if e['kind']==kind) for kind in config['gen_v3']['gates']['smoke']['kinds']]


def records(directory,solvable=None):
    """Every flown episode of a run, read as the frozen tables read it. Starts the teacher did not finish are left out."""
    _,episodes,steps=load(directory);data=json.loads((Path(directory)/'results.json').read_text());landing=load_landing();geometries={};maps={}
    kept=[episode for episode in episodes if solvable is None or episode['id'] in solvable]
    items=[{'episode':episode['id'],'kind':episode.get('kind'),**read(episode,steps[episode['id']],data['config']['success_radius_m'],landing,geometries,maps)} for episode in kept]
    return items,[episode['id'] for episode in data['episodes'] if episode.get('error')],[episode['id'] for episode in episodes if episode not in kept]


def teacher_solved(root):
    """Ids of the recorded episodes the teacher finished in the simulator (a landing must have met the landing rule)."""
    solved=set()
    for path in (Path(root)/'episodes').glob('*.json'):
        summary=json.loads(path.read_text())['summary']
        if summary.get('success'):solved.add(summary['id'])
    return solved


def judge(checks):
    """checks: name -> (value, comparison, limit). Returns the table and whether every line holds."""
    table=[{'check':name,'value':value,'needs':f'{comparison} {limit}','passed':bool(value>=limit if comparison=='>=' else value<=limit)} for name,(value,comparison,limit) in checks.items()]
    return table,all(row['passed'] for row in table)


def pilot(args,gates):
    items,errors,_=records(args.run);land=[i for i in items if i['task']=='land'];approach=[i for i in items if i['task']=='approach']
    checks={'landing flights that touched down on the named pad':(sum(i['touchdown_on_target'] for i in land),'>=',gates['land_touchdowns_min']),
            'approach flights that touched down':(sum(i['landed'] for i in approach),'<=',gates['approach_touchdowns_max']),
            'approach flights that stopped in the air within the radius':(sum(i['success'] for i in approach),'>=',gates['approach_hover_min']),
            'episodes with a runner error':(len(errors),'<=',0)}
    if args.probe:
        probe=json.loads(Path(args.probe).read_text())['task_swap']
        checks['near-pad frames where the landing sentence keeps going']=(round(probe['land_sentence_keeps_going'],2),'>=',gates['probe_min'])
        checks['the same frames where the approach sentence stops']=(round(probe['approach_sentence_stops'],2),'>=',gates['probe_min'])
    return checks,{'flights':len(items),'landing_success':sum(i['success'] for i in land),'touchdown_errors_m':[i['touchdown_error_m'] for i in land if i['touchdown_on_target']],
                   'stopped_after_touchdown':sum(i['touchdown_on_target'] and i['self_stop'] for i in land)}


def flown(args,gates,config,name):
    solvable=teacher_solved(args.teacher);items,errors,unsolved=records(args.run,solvable)
    if name=='smoke':
        wanted=set(smoke_ids(config,json.loads(PLAN.read_text())));items=[i for i in items if i['episode'] in wanted]
    land=[i for i in items if i['task']=='land'];far=[i for i in items if i['initial_distance_m']>=config['gen_v3']['ranges_m']['far'][0]-1.]
    seen=[i for i in items if i['acquired']];total=max(1,len(items))
    extra={'flights':len(items),'success':sum(i['success'] for i in items),'left_out_because_the_teacher_did_not_finish':unsolved,
           'by_kind':{kind:f'{sum(i["success"] for i in items if i["kind"]==kind)}/{sum(i["kind"]==kind for i in items)}' for kind in sorted({i['kind'] for i in items})},
           'failures':{i['episode']:f'{i["failure"]} ({i["mechanism"]})' for i in items if i['failure']}}
    if name=='smoke':
        checks={'flights':(len(items),'>=',gates['episodes']),'successes':(sum(i['success'] for i in items),'>=',gates['success_min']),
                'landing successes':(sum(i['success'] for i in land),'>=',gates['landing_success_min']),'far-start successes':(sum(i['success'] for i in far),'>=',gates['far_success_min']),
                'collisions':(sum(i['collision'] for i in items),'<=',gates['collisions_max']),'episodes with a runner error':(len(errors),'<=',gates['errors_max'])}
    else:
        checks={'flights':(len(items),'>=',gates['episodes_min']),'success rate':(round(sum(i['success'] for i in items)/total,3),'>=',gates['success_rate_min']),
                'landing success rate':(round(sum(i['success'] for i in land)/max(1,len(land)),3),'>=',gates['landing_success_rate_min']),
                'grounded when the target came into view':(round(sum(i['grounded'] for i in seen)/max(1,len(seen)),3),'>=',gates['grounded_rate_min']),
                'end of flight as the sentence asked':(round(sum(i['terminal_as_asked'] for i in items)/total,3),'>=',gates['terminal_as_asked_rate_min']),
                'flights that ended at a wrong object':(sum(i['wrong_target'] for i in items),'<=',gates['wrong_target_max']),
                'collisions':(sum(i['collision'] for i in items),'<=',gates['collisions_max']),'episodes with a runner error':(len(errors),'<=',gates['errors_max'])}
    return checks,extra


def main():
    parser=argparse.ArgumentParser();parser.add_argument('gate',choices=('ids','pilot','smoke','representative'));parser.add_argument('--run')
    parser.add_argument('--teacher',help='dataset root whose recorded validation episodes say which starts the teacher finished')
    parser.add_argument('--probe');parser.add_argument('--output')
    args=parser.parse_args();config=load_config();gates=config['gen_v3']['gates']
    if args.gate=='ids':print(' '.join(smoke_ids(config,json.loads(PLAN.read_text()))));return
    checks,extra=pilot(args,gates['pilot']) if args.gate=='pilot' else flown(args,gates[args.gate],config,args.gate)
    table,passed=judge(checks);result={'gate':args.gate,'run':args.run,'passed':passed,'checks':table,**extra}
    print(json.dumps(result,indent=1))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=1))
    print(f'GATE {args.gate.upper()}: {"PASSED" if passed else "NOT PASSED"}')
    sys.exit(0 if passed else 1)


if __name__=='__main__':main()
