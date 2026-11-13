"""Are a teacher's labels clear of the student's stop? (src/landing/stop_safety.py; no simulator, no model)

  audit_generic_landing_teacher.py [--stage FILE] plans [--output JSON]
        dry runs (kinematics, no lag) of every planned episode of the stage with the stage's teacher: the recorded
        episodes, the pilot starts and the validation starts
  audit_generic_landing_teacher.py [--stage FILE] run DIR [--output JSON]
        teacher flights in the simulator (scripts/surface_flights.py --policy teacher): the labels the teacher gave
  audit_generic_landing_teacher.py [--stage FILE] dataset ROOT [--output JSON]
        a recorded dataset (ROOT/episodes/*.json): the labels that would be trained on
  audit_generic_landing_teacher.py [--stage FILE] compare OLD.json NEW.json [--output JSON]
        two audits side by side

The stop rule is the student's (src/aerovla_oft/spec.is_stop, and stop_ticks stops in a row end a flight); the margin a
landing's labels have to keep is the stage's (`stop_safety` of its configuration, default configs/generic_landing_stage1b.json).
Exit 0: no landing label before a touchdown is a stop or below the margin, and nowhere do stop_ticks stops stand in a row.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.landing import stop_safety
from src.visual_search import generic_landing as stages
from src.visual_search.episodes import load_config
from src.aerovla_oft.spec import load_config as load_oft_config

STAGE=ROOT/'configs/generic_landing_stage1b.json'


def planned(config,oft,stage):
    """Every planned episode of the stage flown by its teacher in a dry run."""
    from src.landing.surface import SurfaceSet
    from src.landing.teacher import rollout
    from src.visual_search.maps import load_map
    plan=json.loads(stages.train_plan_of(stage).read_text(encoding='utf-8'));episodes=plan['train']+plan['val'];scenes={};out=[];files=stages.files_of(stage);names=('pilot','validation')
    if stage.get('stage')=='1c':
        # Stage 1c has more sets, and some of its episodes are pushed: the dry run flies the forced moves too.
        from src.visual_search import generic_landing_1c as later
        files=later.files_of(stage);names=('validation','same_color','same_color_twins','pilot','teacher_pilot')
    if stage.get('stage')=='1d':
        # Stage 1d: its own three check sets (the gates are stage 1c's files and are audited with stage 1c).
        from src.visual_search import generic_landing_1d as later
        files=later.files_of(stage);names=later.SETS
    for name in names:episodes+=json.loads(files[name].read_text(encoding='utf-8'))['episodes']
    for episode in episodes:
        scene=scenes.setdefault(episode['map'],load_map(episode['map']));trace=[]
        result=rollout(config,oft,scene,episode['layout'],episode,SurfaceSet(scene,episode['layout'])[episode['target']],kind=stages.teacher_of(stage),trace=trace)
        out.append({'id':episode['id'],'task':episode['task'],'kind':episode.get('surface_kind'),'start_distance_m':episode['start_distance_m'],'steps':trace,
                    'finished':result['reason']=='teacher_stop' and result['landed']==(episode['task']=='land')})
    return out


def flown(run):
    """The teacher flights of a run folder."""
    from src.landing.failures import steps_of
    path=Path(run) if Path(run).is_absolute() else ROOT/run;steps=steps_of(path)
    records=[item for item in json.loads((path/'results.json').read_text(encoding='utf-8'))['episodes'] if item.get('flown')]
    return [{'id':item['id'],'task':item['episode']['task'],'kind':item['episode'].get('surface_kind'),'start_distance_m':item['episode'].get('start_distance_m',-1.),
             'steps':steps[item['id']]} for item in records]


def show(result):
    total=result['total'];print(f'episodes {total.get("episodes",0)}, labels {total.get("actions",0)}: {total.get("nonterminal",0)} before the end of a flight, {total.get("terminal",0)} at it')
    for group in ('task','kind','state','band'):
        for name,row in result['by'].get(group,{}).items():
            ratio=row['stop_ratio'] or {}
            print(f'  {group:5s} {name:26s} labels {row.get("nonterminal",0):6d}  stops {int(row.get("nonterminal_stop",0)):4d}  below the margin {int(row.get("below_safe",0)):5d}'
                  +(f'  runs of stops {int(row["stop_windows"]):3d}  runs near a stop {int(row["near_stop_windows"]):5d}' if 'stop_windows' in row else f'  near a stop {int(row.get("near_stop",0)):5d}')
                  +f'  ratio min {ratio.get("min")} p5 {ratio.get("p5")} median {ratio.get("median")} p95 {ratio.get("p95")}')
    for check in result['checks']:print(f'  {"PASS" if check["passed"] else "FAIL"}  {check["what"]}: {check["count"]} (allowed {check["allowed"]})')
    print('PASSED' if result['passed'] else 'NOT PASSED')


def compare(old,new):
    """Stage 1's labels and stage 1b's side by side: the landings, and the states the plan names."""
    def side(result):
        land=result['by']['task'].get('LAND',{});state=lambda name:result['by']['state'].get(f'LAND {name}',{})
        ratio=land.get('stop_ratio') or {}
        return {'landing labels before the touchdown':land.get('nonterminal',0),'of them a stop':int(land.get('nonterminal_stop',0)),
                'below the margin':int(land.get('below_safe',0)),'runs of stops before the touchdown':int(land.get('stop_windows',0)),
                'runs near a stop before the touchdown':int(land.get('near_stop_windows',0)),'smallest stop ratio':ratio.get('min'),'stop ratio at the 5th percentile':ratio.get('p5'),
                'median stop ratio':ratio.get('median'),
                **{f'{name}: labels / below the margin / near a stop':f'{state(name).get("nonterminal",0)} / {int(state(name).get("below_safe",0))} / {int(state(name).get("near_stop",0))}'
                   for name in ('SURFACE_APPROACH','ALIGN','CLIMB','DESCENT')}}
    left,right=side(old),side(new)
    return {'rule':new['rule'],'rows':[{'metric':key,'old':left[key],'new':right[key]} for key in left]}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);parser.add_argument('--stage',default=str(STAGE))
    commands=parser.add_subparsers(dest='command',required=True)
    for name,argument in (('plans',None),('run','folder'),('dataset','root')):
        command=commands.add_parser(name);command.add_argument('--output')
        if argument:command.add_argument(argument)
    both=commands.add_parser('compare');both.add_argument('old');both.add_argument('new');both.add_argument('--output')
    args=parser.parse_args();stage=stages.load(args.stage);rule={key:stage['stop_safety'][key] for key in ('safe_ratio','near_stop_ratio')}
    where=lambda path:Path(path) if Path(path).is_absolute() else ROOT/path
    if args.command=='compare':
        result=compare(json.loads(where(args.old).read_text(encoding='utf-8')),json.loads(where(args.new).read_text(encoding='utf-8')))
        for row in result['rows']:print(f'  {row["metric"]:58s} {str(row["old"]):>22s} {str(row["new"]):>22s}')
    else:
        config=load_config();oft=load_oft_config()
        episodes=planned(config,oft,stage) if args.command=='plans' else flown(args.folder) if args.command=='run' else stop_safety.recorded(where(args.root))
        result=stop_safety.audit(episodes,oft,rule,config['stop_ticks']);result['what']=args.command if args.command=='plans' else str(getattr(args,'folder',None) or args.root)
        if args.command=='plans':
            # A dry run that does not finish is no episode of the plan (the plans keep only starts the teacher finishes from).
            result['dry_runs_not_finished']=[episode['id'] for episode in episodes if not episode['finished']]
            if result['dry_runs_not_finished']:result['passed']=False
        show(result)
        if args.command=='plans':print('dry runs not finished:',result['dry_runs_not_finished'])
    if args.output:
        path=where(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=1)+'\n',encoding='utf-8',newline='\n')
    if args.command!='compare':raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
