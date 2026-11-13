"""Check an interactive mission's checkpoint before the simulator starts; writes checkpoint-check.json for the worker.

Without --checkpoint the interactive map's own default is checked (the frozen baseline), against the frozen record that
map names. Exit 0: usable (warnings, if any, are printed). Exit 1: missing, incomplete, or the frozen baseline has changed.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.mission.checkpoint import inspect
from src.mission.semantic import load_demo


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint');parser.add_argument('--frozen');parser.add_argument('--demo')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();settings=load_demo(args.demo)['mission']
    result=inspect(args.checkpoint or settings['checkpoint'],args.frozen or settings['frozen_record'])
    if args.output:
        output=args.output if args.output.is_absolute() else ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
        (output/'checkpoint-check.json').write_text(json.dumps(result,indent=1),encoding='utf-8')
    for text in result['warnings']:print('WARNING: '+text,flush=True)
    for text in result['problems']:print('ERROR: '+text,flush=True)
    if not result['problems']:
        print(f'Checkpoint {result["checkpoint"]}: {result.get("name")} | grounding {result.get("grounding")} | frozen baseline {result["frozen"]["state"]}',flush=True)
    sys.exit(1 if result['problems'] else 0)


if __name__=='__main__':main()
