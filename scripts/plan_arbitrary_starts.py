"""Plan starts for one object from a seed, once; no simulator, no model.

A plan file holds the starts and the arguments they were drawn with. It is written once and flown as written: nothing
that flies a plan draws a start again. `--check FILE` draws the starts of an existing file again from its own recorded
arguments and reports whether they are the same, which is how a plan can be shown to be what its seed gives.

  python scripts/plan_arbitrary_starts.py --target blue_pad --task land --count 20 --seed 30000 \
      --min-distance 20 --max-distance 80 --output configs/experiments/arbitrary_start_example.json
  python scripts/plan_arbitrary_starts.py --check configs/experiments/arbitrary_start_example.json

Every start meets the rule a start has to meet (src/mission/start.py: open ground inside the map's area, the planner's
clearance from every obstacle and object, a height between the executor's floor and the map's ceiling), lies between the
two distances from the object, and keeps its distance from the plan's other starts and from every start of the files named
with --avoid. The distances are not limited to the range the current model was validated in: a plan may ask for 45-60 m or
80-110 m. What a start shows of the object (in the first view, hidden, hidden even from the ceiling) is recorded with it.

Planning a start says nothing about whether a policy can fly from it.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.mission import start as starts
from src.visual_search.episodes import load_config
from src.visual_search.maps import load_map

DEFAULT_MAP='configs/mission/grounding_film_demo.json'
ARGUMENTS=('map','layout','target','task','count','seed','min_distance','max_distance','height','yaw','avoid','separation','max_ticks')


def resolve(path):
    path=Path(path);return path if path.is_absolute() else ROOT/path


def relative(path):
    """A path as it is written into a plan: from the repository root, so that the plan reads the same on every machine."""
    try:return resolve(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:return Path(path).as_posix()


def build(arguments):
    """The plan that a set of arguments gives. Nothing here depends on the clock or on the machine."""
    config=load_config();map_config=load_map(str(resolve(arguments['map'])))
    layout=arguments['layout'] or map_config.get('mission',{}).get('layout')
    if layout not in map_config['layouts']:raise SystemExit(f'No layout {layout!r} in {arguments["map"]}: '+', '.join(map_config['layouts']))
    avoid=[]
    for name in arguments['avoid']:avoid+=starts.starts_of(json.loads(resolve(name).read_text(encoding='utf-8')),map_config['id'])
    episodes=starts.plan_starts(config,map_config,layout,arguments['target'],arguments['task'],arguments['count'],arguments['seed'],
                                (arguments['min_distance'],arguments['max_distance']),arguments['height'],arguments['yaw'],avoid,
                                arguments['separation'],max_ticks=arguments['max_ticks'],map_name=arguments['map'])
    rule=starts.rule(config,map_config)
    return {'description':'Starts drawn once from a seed by scripts/plan_arbitrary_starts.py and flown as written. `arguments` are what they were drawn '
                          'with; `--check` draws them again and compares. Each episode has the fields the canonical environment reads its start from '
                          '(start_xy, start_yaw_deg, start_height_m) and what the start shows of the object, from the map\'s boxes. The map path is '
                          'from the repository root.',
            'generator':'scripts/plan_arbitrary_starts.py','arguments':{**arguments,'layout':layout},'seed':arguments['seed'],
            'map':arguments['map'],'map_id':map_config['id'],'layout':layout,'target':arguments['target'],'task':arguments['task'],
            'rule':{key:rule[key] for key in ('clearance_m','clearance_from','height_m','height_from')},
            'avoided_starts':len(avoid),'canonical_range_m':config['canonical']['max_start_m'],
            'counts':{'starts':len(episodes),'initially_visible':sum(item['initially_visible'] for item in episodes),
                      'occluded':sum(item['occluded'] for item in episodes),'requires_translation':sum(item['requires_translation'] for item in episodes),
                      'beyond_canonical_range':sum(item['start_distance_m']>config['canonical']['max_start_m'] for item in episodes)},
            'episodes':episodes}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check',help='draw the starts of this plan file again from its own arguments and compare')
    parser.add_argument('--target');parser.add_argument('--task',choices=('land','approach'),default='approach')
    parser.add_argument('--count',type=int,default=10);parser.add_argument('--seed',type=int)
    parser.add_argument('--min-distance',type=float);parser.add_argument('--max-distance',type=float)
    parser.add_argument('--height',type=float,nargs=2,metavar=('LOW','HIGH'),help="start heights in metres; default: the map's own start band")
    parser.add_argument('--yaw',choices=starts.YAW_MODES,default='random',help='random: any heading; toward / away: facing the object or away from it')
    parser.add_argument('--map',default=DEFAULT_MAP,help='map file; default: the interactive mission map');parser.add_argument('--layout')
    parser.add_argument('--avoid',nargs='*',default=[],help='plan files whose starts the new ones keep away from (training or evaluation starts)')
    parser.add_argument('--separation',type=float,help='metres kept from other starts; default: the planner\'s exclusion radius')
    parser.add_argument('--max-ticks',type=int,help='decisions per flight written into the plan');parser.add_argument('--output')
    parser.add_argument('--force',action='store_true',help='write over an existing plan file')
    args=parser.parse_args()
    if args.check:
        saved=json.loads(resolve(args.check).read_text(encoding='utf-8'));again=build(saved['arguments'])
        same=again['episodes']==saved['episodes']
        print(f'{args.check}: {len(saved["episodes"])} starts, seed {saved["seed"]}: '+('the same starts are drawn again' if same else 'DIFFERENT starts are drawn now'))
        sys.exit(0 if same else 1)
    missing=[name for name in ('target','seed','min_distance','max_distance','output') if getattr(args,name) is None]
    if missing:parser.error('needed: '+', '.join('--'+name.replace('_','-') for name in missing))
    output=resolve(args.output)
    if output.exists() and not args.force:parser.error(f'{args.output} exists: a plan is written once (--force writes over it)')
    arguments={name:getattr(args,name) for name in ARGUMENTS};arguments['map']=relative(args.map);arguments['avoid']=[relative(name) for name in args.avoid]
    try:plan=build(arguments)
    except (ValueError,RuntimeError) as error:raise SystemExit(str(error))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(plan,indent=1)+'\n',encoding='utf-8',newline='\n')
    print(f'{args.output}: {plan["counts"]}')


if __name__=='__main__':main()
