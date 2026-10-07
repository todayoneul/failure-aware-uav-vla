"""Where do AeroVLA-OFT predictions leave the action range, and does it matter? No simulator, no model.

Reads the steps of flown AeroVLA-OFT runs (the head's normalised first action and the teacher's action
for the same frame) and, optionally, a dataset, and reports per axis how many predictions fall outside
[-1, 1], by how much, and how many of those sit on an axis where the teacher's own action is exactly at
that limit. Anything outside the range is clipped to the limit before it is flown.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.aerovla_oft.spec import normalize_action

AXES=('forward','down','yaw')


def analyse(run):
    data=json.loads((Path(run)/'results.json').read_text());config=data['oft'];rows=[]
    for line in (Path(run)/'steps.jsonl').read_text().splitlines():
        row=json.loads(line)
        if 'normalised' in row and not row.get('perturbed'):rows.append(row)
    result={'run':str(run),'decisions':len(rows),'any_axis_outside':sum(any(abs(v)>1 for v in row['normalised']) for row in rows),'axes':{}}
    for index,axis in enumerate(AXES):
        outside=[row for row in rows if abs(row['normalised'][index])>1]
        excess=[abs(row['normalised'][index])-1 for row in outside]
        # The teacher's action for the same frame, in the same units: is the limit itself what was asked for?
        at_limit=sum(normalize_action(row['teacher_action'],config)[index]*row['normalised'][index]>=.999 for row in outside)
        labels_at_limit=sum(abs(normalize_action(row['teacher_action'],config)[index])>=.999 for row in rows)
        result['axes'][axis]={'outside':len(outside),'teacher_at_that_limit':at_limit,'median_excess':statistics.median(excess) if excess else 0.,
                              'largest_excess':max(excess) if excess else 0.,'teacher_actions_at_a_limit':labels_at_limit}
    return result


def dataset_labels(root,config):
    samples=json.loads((Path(root)/'train.json').read_text());counts={axis:0 for axis in AXES};outside=0
    for sample in samples:
        first=normalize_action(sample['chunk'][0],config)
        for index,axis in enumerate(AXES):counts[axis]+=abs(first[index])>=.999
        outside+=any(not low-1e-9<=value<=high+1e-9 for value,(low,high) in zip(sample['chunk'][0],config['action_bounds'].values()))
    return {'samples':len(samples),'first_action_at_a_limit':{axis:value/len(samples) for axis,value in counts.items()},'labels_outside_bounds':outside}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+');parser.add_argument('--dataset');parser.add_argument('--json')
    args=parser.parse_args();output={'runs':[analyse(run) for run in args.runs]}
    print('| Run | Decisions | Any axis outside | '+' | '.join(f'{axis}: outside / teacher at that limit / median excess / largest' for axis in AXES)+' |')
    print('|---|---:|---:|'+'---|'*len(AXES))
    for result in output['runs']:
        cells=[f'{a["outside"]} / {a["teacher_at_that_limit"]} / {a["median_excess"]:.3f} / {a["largest_excess"]:.2f}' for a in result['axes'].values()]
        print(f'| {Path(result["run"]).name} | {result["decisions"]} | {result["any_axis_outside"]} ({result["any_axis_outside"]/max(1,result["decisions"]):.0%}) | '+' | '.join(cells)+' |')
    if args.dataset:
        config=json.loads((Path(args.runs[0])/'results.json').read_text())['oft'];output['dataset']=dataset_labels(args.dataset,config)
        share=output['dataset']['first_action_at_a_limit']
        print(f'\nTraining labels ({output["dataset"]["samples"]} samples): first action exactly at a limit on '
              +', '.join(f'{axis} {value:.0%}' for axis,value in share.items())+f'; outside the bounds: {output["dataset"]["labels_outside_bounds"]}')
    if args.json:Path(args.json).write_text(json.dumps(output,indent=1))
