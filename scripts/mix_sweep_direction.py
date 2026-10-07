"""Relabel part of a dataset as if a left-sweeping teacher had flown it; no simulator, no torch.

A policy that sees one frame cannot know which way an ongoing sweep is going. This writes a copy of a
dataset's train/val files in which a fraction of the episodes have their sweep labels mirrored, to
measure what such a mixture does to the learned search (docs/aerovla_oft_generalization.md).
"""
import argparse
import hashlib
import json
from pathlib import Path


def mirrored(sample,sweep):
    """The same frame as a left-sweeping teacher would label it: sweep turns change sign, nothing else."""
    copy=json.loads(json.dumps(sample))
    for step in copy['chunk']:
        if abs(step[0])<1e-6 and abs(abs(step[2])-sweep)<1e-6:step[2]=-step[2]
    copy['label']['yaw']=copy['chunk'][0][2];copy['meta']['strategy']='left'
    return copy


def mix(samples,fraction,sweep):
    result=[]
    for sample in samples:
        chosen=int(hashlib.sha256(sample['meta']['episode_id'].encode()).hexdigest(),16)%1000<fraction*1000
        result.append(mirrored(sample,sweep) if chosen and sample['meta']['teacher_state']=='search' else sample)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--fraction',type=float,default=.5)
    parser.add_argument('--sweep-rad',type=float,default=.21);parser.add_argument('--suffix',default='mixed')
    args=parser.parse_args();root=Path(args.root)
    for name in ('train','val'):
        samples=json.loads((root/f'{name}.json').read_text());mixed=mix(samples,args.fraction,args.sweep_rad)
        (root/f'{name}_{args.suffix}.json').write_text(json.dumps(mixed))
        search=[s for s in mixed if s['meta']['teacher_state']=='search']
        print(name,len(mixed),'search',len(search),'mirrored',sum(s['meta'].get('strategy')=='left' for s in search))
