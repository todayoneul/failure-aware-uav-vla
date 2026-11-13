"""Check a start given to the interactive mission's launcher before the simulator starts; writes start-check.json.

The start is the map's first preset launch with whatever was given in its place (--start-x / --start-y / --start-yaw /
--start-height), exactly as the worker builds it (src/mission/oft_runner.py), and it is held to the same rule
(src/mission/start.py). Exit 0: a flight may start there. Exit 1: it may not, and every reason is printed.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.mission import semantic
from src.mission import start as starts
from src.visual_search.episodes import load_config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--demo');parser.add_argument('--layout');parser.add_argument('--output',type=Path)
    parser.add_argument('--start-x',type=float);parser.add_argument('--start-y',type=float)
    parser.add_argument('--start-yaw',type=float);parser.add_argument('--start-height',type=float)
    args=parser.parse_args();config=load_config();demo=semantic.load_demo(args.demo);settings=demo['mission']
    layout=args.layout or settings['layout']
    if layout not in demo['layouts']:
        print(f'ERROR: no layout {layout!r} in the map file: '+', '.join(demo['layouts']),flush=True);sys.exit(1)
    given={'x':args.start_x,'y':args.start_y,'yaw_deg':args.start_yaw,'height_m':args.start_height}
    chosen=semantic.start_of(settings['launches'][0],config['cruise_height_m']).moved(**{key:value for key,value in given.items() if value is not None})
    custom=any(value is not None for value in given.values())
    result={'layout':layout,'custom':custom,'start':chosen.to_dict(),**starts.validate(chosen,demo,layout,config)}
    if args.output:
        output=args.output if args.output.is_absolute() else ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
        (output/'start-check.json').write_text(json.dumps(result,indent=1),encoding='utf-8')
    words=f'x {chosen.x:+.2f}, y {chosen.y:+.2f}, yaw {chosen.yaw_deg:+.1f} deg, height {chosen.height_m:.2f} m (layout {layout})'
    if result['valid']:
        print(('Custom start: ' if custom else 'Start: the first preset, ')+words+f' | clearance {result["facts"].get("clearance_m")} m',flush=True)
    else:print('INVALID START: '+words+'\nReason: '+'; '.join(result['reasons']),flush=True)
    sys.exit(0 if result['valid'] else 1)


if __name__=='__main__':main()
