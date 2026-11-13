"""Judge recorded human (and takeover) flights again by the current keep rule, and move each to episodes/ or rejected/.

  refile_human_flights.py [--dataset DIR ...] [--apply]

The keep rule is src/human_pilot/session.kept (and, for a takeover flight, at least one label of the pilot's own). It is
read from what each flight's summary already holds (the surface evaluator's reading, the real-time factor, the reason, the
labels); nothing is flown again and no frame or step is changed. Without --apply it only prints what would move. A flight
that would become data but whose pictures were not kept is left where it is.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.human_pilot import session


def verdict(summary,settings):
    good=session.kept(summary['surface_landing'],summary,summary.get('real_time_factor'),settings)
    if summary.get('pilot')=='takeover':good=good and summary['labels']['labels']>0
    return good


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset',action='append',help='dataset folder (default: both of configs/human_pilot.json)')
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    settings=session.load();folders=[Path(item) for item in args.dataset or [ROOT/settings['dataset'],ROOT/settings['takeover']['dataset']]]
    for root in folders:
        counts={'kept':0,'rejected':0,'moved':0,'left':0}
        for folder in ('episodes','rejected'):
            for path in sorted((root/folder).glob('*.json')):
                data=json.loads(path.read_text(encoding='utf-8'));good=verdict(data['summary'],settings)
                counts['kept' if good else 'rejected']+=1;target=root/('episodes' if good else 'rejected')/path.name
                if target.parent.name==folder:continue
                if good and not (root/data['traj_rel_dir']/'frontcamera').exists():
                    print(f'{path.name}: would be data, but its pictures were not kept; left in {folder}/');counts['left']+=1;continue
                print(f'{path.name}: {folder}/ -> {target.parent.name}/'+('' if args.apply else ' (not moved: --apply)'))
                if args.apply:target.parent.mkdir(parents=True,exist_ok=True);path.replace(target);counts['moved']+=1
        print(root.relative_to(ROOT) if root.is_relative_to(ROOT) else root,counts)


if __name__=='__main__':main()
