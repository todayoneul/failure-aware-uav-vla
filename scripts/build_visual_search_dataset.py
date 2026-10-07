"""Turn recorded teacher episodes into an AeroVLA-format training file with OFT action chunks.

Each sample keeps AeroVLA's fields (traj_rel_dir, img_name, instruction, label, is_last_step,
is_penultimate) and adds the future chunk, the optional proprio vector and analysis-only metadata.
The split is by episode. No simulator, no torch.
"""
import argparse
import collections
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.aerovla_oft.spec import load_config,chunk_labels,split_episodes


def episode_samples(record,chunk_size):
    steps=record['steps'];labels=chunk_labels([step.get('teacher_action',step['action']) for step in steps],chunk_size)
    samples=[]
    for index,step in enumerate(steps):
        window=steps[index:index+chunk_size]
        # A forced turn is flown but is not the teacher's choice, so no label may contain one.
        if any(item.get('perturbed') for item in window) or 'img_name' not in step:continue
        fwd,down,yaw=labels[index][0]
        samples.append({'traj_rel_dir':record['traj_rel_dir'],'img_name':step['img_name'],
                        'instruction':'<image>\n'+record['summary']['instruction'],
                        'label':{'fwd':fwd,'down':down,'yaw':yaw},
                        'is_last_step':index==len(steps)-1,'is_penultimate':index==len(steps)-2,
                        'chunk':labels[index],'proprio':step['proprio'],
                        'meta':{'episode_id':record['summary']['id'],'case':record['summary']['case'],'target':record['summary']['target'],
                                'step':step['step'],'target_visible':step['front_seen'],'target_in_front_fov':step['front_in_fov'],
                                'target_visible_down':step['down_seen'],'bearing_deg':step['bearing_deg'],'distance_m':step['distance_m'],
                                'teacher_state':step['teacher_state']}})
    return samples


def build(root,config,validation_fraction,seed):
    records=[json.loads(path.read_text()) for path in sorted((root/'episodes').glob('*.json'))]
    records=[record for record in records if record['steps'] and not record['summary'].get('error') and record['summary']['reason']=='teacher_stop']
    train_ids,validation_ids=split_episodes([record['summary']['id'] for record in records],validation_fraction,seed)
    splits={'train':[],'val':[]}
    for record in records:
        splits['val' if record['summary']['id'] in validation_ids else 'train']+=episode_samples(record,config['chunk_size'])
    def count(samples,key):return dict(collections.Counter(key(sample) for sample in samples))
    everything=splits['train']+splits['val']
    summary={'episodes':len(records),'train_episodes':len(train_ids),'val_episodes':len(validation_ids),
             'samples':len(everything),'train_samples':len(splits['train']),'val_samples':len(splits['val']),
             'by_case_episodes':dict(collections.Counter(record['summary']['case'] for record in records)),
             'by_target_episodes':dict(collections.Counter(record['summary']['target'] for record in records)),
             'by_teacher_state':count(everything,lambda s:s['meta']['teacher_state']),
             'target_visible':count(everything,lambda s:'visible' if s['meta']['target_visible'] else 'not visible'),
             'chunk_size':config['chunk_size'],'tick_s':config['tick_s'],'action_bounds':config['action_bounds'],
             'val_episode_ids':validation_ids}
    return splits,summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--oft-config')
    parser.add_argument('--validation-fraction',type=float,default=.2);parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();root=Path(args.root)
    splits,summary=build(root,load_config(args.oft_config),args.validation_fraction,args.seed)
    for name,samples in splits.items():(root/f'{name}.json').write_text(json.dumps(samples))
    (root/'summary.json').write_text(json.dumps(summary,indent=1));print(json.dumps(summary,indent=1))
