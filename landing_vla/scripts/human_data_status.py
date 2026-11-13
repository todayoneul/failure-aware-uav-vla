"""Where the human data stands against a training plan's data gate (configs/human_takeover_r1.json by default).

  human_data_status.py [--plan configs/human_takeover_r1.json]

Counts the kept flights (episodes/) of the takeover dataset by kind (a landing on a pad, a cube, a cylinder or the plate on
the ball, or an approach), by distance band and by scene, and shows each line of the gate with what is still missing. The
human pilot's own kept flights are counted apart. Reads the recorded summaries only.
"""
import argparse
import collections
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.human_pilot import session


def tally(folder):
    """Kept flights of one dataset folder: kinds, bands, scenes, views, labels."""
    counts={'flights':0,'kind':collections.Counter(),'band':collections.Counter(),'scene':collections.Counter(),'view':collections.Counter(),'labels':0}
    for path in sorted((folder/'episodes').glob('*.json')):
        summary=json.loads(path.read_text(encoding='utf-8'))['summary'];reading=summary.get('surface_landing') or {}
        counts['flights']+=1;counts['labels']+=summary['labels']['labels']
        counts['kind']['approach' if summary['task']=='approach' else session.surface_kind(reading.get('target_surface')) or 'other']+=1
        # Flights recorded before the bands existed started 12-44 m away: the near band.
        counts['band'][summary.get('distance_band') or 'near']+=1;counts['scene'][summary['map']]+=1;counts['view'][summary.get('view','cameras')]+=1
    return counts


def lines(counts,gate):
    """(what, have, need) for every line of the gate."""
    out=[('kept takeover flights',counts['flights'],gate['flights_min'])]
    out+=[(f'landings on a {kind}',counts['kind'][kind],need) for kind,need in gate['land_min'].items()]
    out+=[('approaches',counts['kind']['approach'],gate['approach_min']),
          ('flights started beyond 44 m (mid, far)',counts['band']['mid']+counts['band']['far'],gate['beyond_44m_min'])]
    out+=[(f'flights in {scene}',counts['scene'][scene],need) for scene,need in gate['per_scene_min'].items()]
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan',default=str(ROOT/'configs/human_takeover_r1.json'));args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8'));gate=plan['data_gate'];settings=session.load()
    takeover=tally(ROOT/settings['takeover']['dataset']);pilot=tally(ROOT/settings['dataset'])
    print(f'{Path(args.plan).name} data gate')
    for what,have,need in lines(takeover,gate):
        print(f'  {"OK  " if have>=need else "    "}{what:42s} {have:4d} / {need:<4d}'+('' if have>=need else f'   {need-have} more'))
    met=all(have>=need for _,have,need in lines(takeover,gate))
    print(f'  labels of the pilot in kept takeover flights: {takeover["labels"]}; bands {dict(takeover["band"])}; views {dict(takeover["view"])}')
    print(f'  human pilot flights (counted apart): {pilot["flights"]} kept, {pilot["labels"]} labels')
    print('  data gate: MET (training waits for the user to say so)' if met else '  data gate: not met yet')


if __name__=='__main__':main()
