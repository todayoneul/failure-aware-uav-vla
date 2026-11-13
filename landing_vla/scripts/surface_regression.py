"""Read recorded flights again with the surface evaluator and compare with what the canonical evaluator recorded.

  python scripts/surface_regression.py RUN_FOLDER [RUN_FOLDER ...] [--output FILE]

A run folder is one an evaluation wrote (results.json and steps.jsonl of scripts/visual_search.py, or of the blur
characterization). Nothing is flown and no recorded result is changed: each flight's canonical summary and its decisions
are handed to src/landing/evaluator.read, and the two readings are counted against each other. Every landing surface of
those runs is a pad, so every flight has to be read as it was: the same result, the same landing region, the same decision
of the touchdown and the same speed. A difference would mean the general rule changed what a pad landing is.
"""
import argparse
import collections
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.landing import evaluator
from src.landing.surface import SurfaceSet
from src.visual_search.maps import load_map,load_landing

KEYS=('touchdown_success','stable_physical_landing','system_land_success','physical_landing')


def read_run(folder,landing,surfaces):
    """Counts for one run folder, and the flights that are read differently."""
    folder=Path(folder);results=json.loads((folder/'results.json').read_text(encoding='utf-8'))['episodes'];steps=collections.defaultdict(list)
    with (folder/'steps.jsonl').open(encoding='utf-8') as file:
        for line in file:
            row=json.loads(line);steps[row['episode']].append(row)
    counts=collections.Counter();different=[]
    for summary in results:
        if summary.get('error') or 'target' not in summary or not steps.get(summary['id']):continue
        key=(summary.get('map','blocks'),summary.get('layout','pilot'))
        if key not in surfaces:surfaces[key]=SurfaceSet(load_map(key[0]),key[1],landing)
        reading=evaluator.read(summary,surfaces[key],landing,steps[summary['id']]);task=summary.get('task','approach')
        counts['flights']+=1;counts[task]+=1;problems=[]
        if reading['success']!=bool(summary.get('success')):problems.append('success')
        if summary.get('touchdown'):
            counts['touchdowns']+=1
            if task=='land':
                counts['landings_with_touchdown']+=1;counts['landings_read_as_success']+=reading['success']
            if reading['inside_region']!=summary['touchdown']['inside_region']:problems.append('inside_region')
            if reading['touchdown_decision']!=summary['touchdown'].get('step'):problems.append('touchdown_decision')
            if reading['vertical_speed_mps']!=summary['touchdown']['vertical_speed_mps']:problems.append('vertical_speed')
            if reading['landed_on']!=summary.get('landed_on'):problems.append('landed_on')
        if task=='land':
            problems+=[name for name in KEYS if name in summary and reading[name]!=bool(summary[name])]
        if (reading['collision'] is not None)!=bool(summary.get('hit')):problems.append('collision')
        counts['same']+=not problems
        if problems:different.append({'id':summary['id'],'differs_in':problems})
    return dict(counts),different


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs',nargs='+');parser.add_argument('--output')
    args=parser.parse_args();landing=load_landing();surfaces={};rows=[];total=collections.Counter()
    for name in args.runs:
        folder=Path(name) if Path(name).is_absolute() else ROOT/name
        counts,different=read_run(folder,landing,surfaces);total.update(counts)
        rows.append({'run':Path(name).as_posix(),**counts,'different':different})
        print(f'{name}: {counts.get("flights",0)} flights, {counts.get("touchdowns",0)} touchdowns, read the same: {counts.get("same",0)}'
              +(f' | DIFFERENT: {different}' if different else ''))
    report={'description':'Recorded flights read again by the surface evaluator (src/landing/evaluator.py) against the canonical summary each was recorded '
                          'with (scripts/surface_regression.py). Every landing surface of these runs is a pad. `same` counts the flights whose result, '
                          'landing region, decision of the touchdown, touchdown speed, object landed on and collision are read as they were recorded.',
            'evaluator':evaluator.VERSION,'total':dict(total),'runs':rows}
    if args.output:
        output=Path(args.output) if Path(args.output).is_absolute() else ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=1)+'\n',encoding='utf-8',newline='\n')
    print(f'TOTAL: {total["flights"]} flights, {total["touchdowns"]} touchdowns, read the same: {total["same"]}')
    sys.exit(0 if total['same']==total['flights'] else 1)


if __name__=='__main__':main()
