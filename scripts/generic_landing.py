"""Phase B, stage 1 (generic landing): plans, gates and tables; no simulator, no model.

  plan                      write configs/generic_landing/{pilot,validation,regression,train_plan}.json once
  check                     confirm the plans are what the configuration gives, that every recorded start is in a training
                            layout and every check start in an evaluation layout, and that all starts keep their distance
  ids smoke|critical        the validation starts flown first; the ones flown a second time
  pilot --runs NAME=DIR ... --teacher DIR [--output JSON]
                            the pilot's table and the candidate that goes on (exit 0: one meets the pilot's lines)
  gate smoke|validation --run DIR [--output JSON]            exit 0 when passed
  regression --run DIR --baseline DIR [--output JSON]        exit 0 when the checkpoint is not worse than the baseline

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
    points=stage1.earlier_points();sets={}
    sets['regression']=stage1.build_regression(config,oft,stage,points)
    sets['pilot']=stage1.build_pilot(config,oft,stage,points)
    sets['validation']=stage1.build_validation(config,oft,stage,points)
    sets['train_plan']=stage1.build_training(config,oft,stage,points)
    return sets


def histogram(items,key):return dict(sorted(collections.Counter(key(item) for item in items).items(),key=lambda pair:str(pair[0])))


def describe_training(stage,plan):
    out={'description':'Generic landing: the plan of the recorded episodes. Landings on pads, cubes and cylinders from arbitrary starts, and approach twins of some of '
                       'them. Recorded with src/landing/teacher.SurfaceTeacher in layouts earlier episodes were recorded in.',
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
    errors=[];again=build_all(config,oft,stage);saved={name:json.loads(stage1.FILES[name].read_text(encoding='utf-8'))['episodes'] for name in stage1.FILES}
    plan=json.loads(stage1.TRAIN_PLAN.read_text(encoding='utf-8'))
    for name in stage1.FILES:
        if again[name]!=saved[name]:errors.append(f'{name}: the file is not what the configuration gives')
    if any(again['train_plan'][split]!=plan[split] for split in ('train','val')):errors.append('train_plan: the file is not what the configuration gives')
    recorded=plan['train']+plan['val'];checks=[episode for name in stage1.FILES for episode in saved[name]]
    for episode in recorded:
        if episode['layout'] not in stage['layouts']['train']:errors.append(f'{episode["id"]}: recorded in an evaluation layout')
        if not stage['bands_m']['near'][0]-.01<=episode['start_distance_m']<=stage['bands_m']['far'][1]+.01:errors.append(f'{episode["id"]}: outside the range')
    for episode in checks:
        if episode['layout'] not in stage['layouts']['evaluation']:errors.append(f'{episode["id"]}: a check start in a layout episodes were recorded in')
    # Starts of one scene keep their distance, across all sets (a twin shares its start with the landing it is a twin of).
    firsts=[episode for episode in recorded+checks if not episode.get('twin_of')];earlier=stage1.earlier_points()
    for index,a in enumerate(firsts):
        for b in firsts[index+1:]:
            if a['map']==b['map'] and math.dist(a['start_xy'],b['start_xy'])<stage['separation_m']:errors.append(f'{a["id"]} and {b["id"]} start too near each other')
        if any(where==a['map'] and math.dist(a['start_xy'],xy)<stage['separation_m'] for where,xy in earlier):errors.append(f'{a["id"]} starts too near an earlier start')
    seeds=collections.Counter((episode['map'],episode['seed']) for episode in firsts)
    errors+=[f'seed used twice: {key}' for key,count in seeds.items() if count>1]
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('plan');commands.add_parser('check')
    ids=commands.add_parser('ids');ids.add_argument('which',choices=('smoke','critical'))
    args=parser.parse_args();config=load_config();oft=load_oft_config();stage=stage1.load()
    if args.command=='plan':
        if any(path.exists() for path in list(stage1.FILES.values())+[stage1.TRAIN_PLAN]):raise SystemExit('the plans exist; they are written once')
        sets=build_all(config,oft,stage)
        for name in stage1.FILES:write(stage1.FILES[name],stage1.describe(stage,name,sets[name]))
        write(stage1.TRAIN_PLAN,describe_training(stage,sets['train_plan']))
        for name in stage1.FILES:print(name,len(sets[name]),histogram(sets[name],lambda e:(e.get('surface_kind') or e.get('role'),e['task'])))
        described=describe_training(stage,sets['train_plan'])
        for split in ('train','val'):print(split,json.dumps(described[f'{split}_summary']))
    elif args.command=='check':
        errors=check(config,oft,stage)
        for text in errors:print('ERROR: '+text)
        if errors:raise SystemExit(1)
        print('OK: the plans are what the configuration gives; recorded starts are in training layouts, check starts in evaluation layouts, and all keep their distance')
    elif args.command=='ids':
        episodes=json.loads(stage1.FILES['validation'].read_text(encoding='utf-8'))['episodes']
        if args.which=='critical':print(json.dumps(stage1.critical_ids(episodes)))
        else:
            count=stage['validation']['smoke']['starts_per_kind'];chosen=[]
            for kind in stage1.KINDS:chosen+=[e['id'] for e in episodes if e.get('surface_kind')==kind and e['task']=='land'][:count]
            print(json.dumps(chosen))


if __name__=='__main__':main()
