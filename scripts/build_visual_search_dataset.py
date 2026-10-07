"""Turn recorded teacher episodes into an AeroVLA-format training file with OFT action chunks.

Each sample keeps AeroVLA's fields (traj_rel_dir, img_name, instruction, label, is_last_step,
is_penultimate) and adds the future chunk, the optional proprio vector and analysis-only metadata.
Episodes planned by scripts/plan_generalization.py carry their split (decided by seed range) and are
split accordingly; older recordings are split at random by episode. Either way no trajectory has
frames on both sides, and the summary says how the data is distributed. No simulator, no torch.
"""
import argparse
import collections
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.aerovla_oft.spec import load_config,chunk_labels,split_episodes
from src.visual_search.generalization import sector_of,STATELESS_STRATEGIES

DISTANCE_EDGES=(0,12,20,30,40,50,60,70,1e9)


def episode_samples(record,chunk_size):
    steps=record['steps'];labels=chunk_labels([step.get('teacher_action',step['action']) for step in steps],chunk_size)
    summary=record['summary'];samples=[]
    for index,step in enumerate(steps):
        window=steps[index:index+chunk_size]
        # A forced turn is flown but is not the teacher's choice, so no label may contain one.
        if any(item.get('perturbed') for item in window) or 'img_name' not in step:continue
        fwd,down,yaw=labels[index][0]
        samples.append({'traj_rel_dir':record['traj_rel_dir'],'img_name':step['img_name'],
                        'instruction':'<image>\n'+summary['instruction'],
                        'label':{'fwd':fwd,'down':down,'yaw':yaw},
                        'is_last_step':index==len(steps)-1,'is_penultimate':index==len(steps)-2,
                        'chunk':labels[index],'proprio':step['proprio'],
                        'meta':{'episode_id':summary['id'],'case':summary['case'],'target':summary['target'],
                                'map':summary.get('map','blocks'),'layout':summary.get('layout','pilot'),'seed':summary.get('seed'),
                                'kind':summary.get('kind',summary['case']),'strategy':summary.get('strategy','right'),
                                'step':step['step'],'target_visible':step['front_seen'],'target_in_front_fov':step['front_in_fov'],
                                'target_visible_down':step['down_seen'],'bearing_deg':step['bearing_deg'],'distance_m':step['distance_m'],
                                'height_m':step.get('height_m'),'teacher_state':step['teacher_state']}})
    return samples


def count(items,key):return dict(sorted(collections.Counter(key(item) for item in items).items(),key=lambda pair:str(pair[0])))


def distance_bin(value):
    for low,high in zip(DISTANCE_EDGES,DISTANCE_EDGES[1:]):
        if low<=value<high:return f'{low}-{high} m' if high<1e9 else f'{low}+ m'


def leaks(splits,records):
    """Anything shared between the training and validation sides: an episode, a trajectory folder, a seed or a start."""
    sides={name:{'ids':{s['meta']['episode_id'] for s in samples},'folders':{s['traj_rel_dir'] for s in samples}} for name,samples in splits.items()}
    seeds={name:{(r['summary'].get('map','blocks'),r['summary']['seed'],r['summary']['target'],r['summary']['case']) for r in records if r['summary']['id'] in sides[name]['ids']} for name in splits}
    starts={name:{tuple(round(v,2) for v in r['summary'].get('start_xy',[])) for r in records if r['summary']['id'] in sides[name]['ids']} for name in splits}
    return {'episodes':sorted(sides['train']['ids']&sides['val']['ids']),'folders':sorted(sides['train']['folders']&sides['val']['folders']),
            'seeds':len(seeds['train']&seeds['val']),'starts':len((starts['train']&starts['val'])-{()})}


def build(root,config,validation_fraction,seed):
    everything=[json.loads(path.read_text()) for path in sorted((root/'episodes').glob('*.json'))]
    records=[record for record in everything if record['steps'] and not record['summary'].get('error') and record['summary']['reason']=='teacher_stop']
    dropped=count([record for record in everything if record not in records],lambda r:r['summary'].get('reason','error'))
    planned=all(record['summary'].get('split') in ('train','val') for record in records)
    if planned:validation_ids={record['summary']['id'] for record in records if record['summary']['split']=='val'}
    else:validation_ids=set(split_episodes([record['summary']['id'] for record in records],validation_fraction,seed)[1])
    splits={'train':[],'val':[]}
    for record in records:
        splits['val' if record['summary']['id'] in validation_ids else 'train']+=episode_samples(record,config['chunk_size'])
    samples=splits['train']+splits['val'];bounds=config['action_bounds'];summaries=[record['summary'] for record in records]
    outside=sum(any(not low-1e-9<=value<=high+1e-9 for value,(low,high) in zip(step,bounds.values())) for sample in samples for step in sample['chunk'])
    search=[s for s in summaries if not s.get('initially_seen',True) or s.get('kind') in ('search','altitude','reacquire')]
    climbed=[s for s in search if s.get('climbed_m',0.)>=1.]
    summary={'episodes':len(records),'train_episodes':len(records)-len(validation_ids),'val_episodes':len(validation_ids),
             'dropped_episodes':dropped,'split_by':'seed range of the plan' if planned else 'random by episode',
             'samples':len(samples),'train_samples':len(splits['train']),'val_samples':len(splits['val']),
             'maps':count(summaries,lambda s:s.get('map','blocks')),'layouts':count(summaries,lambda s:s.get('layout','pilot')),
             'objects':count(summaries,lambda s:s['target']),'instructions':count(summaries,lambda s:s['instruction']),
             'by_case_episodes':count(summaries,lambda s:s['case']),'by_target_episodes':count(summaries,lambda s:s['target']),
             'start_distance_m':count(summaries,lambda s:distance_bin(s['initial_distance_m'])),
             'start_bearing_sector':count(summaries,lambda s:sector_of(s['target_bearing_deg'])),
             'start_height_m':count(summaries,lambda s:f'{int(s.get("start_height_m",6.))}-{int(s.get("start_height_m",6.))+1} m'),
             'search_strategy_episodes':count(summaries,lambda s:s.get('strategy','right')),
             'search_episodes':{'yaw_only':len(search)-len(climbed),'yaw_and_altitude':len(climbed)},
             'forced_turns':count([s for s in summaries if s.get('kick')],lambda s:'left' if s['kick']['degrees']<0 else 'right'),
             'seeds':{name:[min(values),max(values)] if values else None for name,values in
                      (('train',[s['seed'] for s in summaries if s['id'] not in validation_ids]),('val',[s['seed'] for s in summaries if s['id'] in validation_ids]))},
             'sample_distance_m':count(samples,lambda s:distance_bin(s['meta']['distance_m'])),
             'sample_bearing_sector':count(samples,lambda s:sector_of(s['meta']['bearing_deg'])),
             'by_teacher_state':count(samples,lambda s:s['meta']['teacher_state']),
             'target_visible':count(samples,lambda s:'visible' if s['meta']['target_visible'] else 'not visible'),
             'labels_outside_action_bounds':outside,
             'memory_strategy_samples':sum(s['meta']['strategy'] not in STATELESS_STRATEGIES for s in samples),
             'leaks':leaks(splits,records),
             'chunk_size':config['chunk_size'],'tick_s':config['tick_s'],'action_bounds':config['action_bounds'],
             'val_episode_ids':sorted(validation_ids)}
    return splits,summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--oft-config')
    parser.add_argument('--validation-fraction',type=float,default=.2);parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();root=Path(args.root)
    splits,summary=build(root,load_config(args.oft_config),args.validation_fraction,args.seed)
    if any(summary['leaks'].values()):raise SystemExit(f'Training and validation overlap: {summary["leaks"]}')
    for name,samples in splits.items():(root/f'{name}.json').write_text(json.dumps(samples))
    (root/'summary.json').write_text(json.dumps(summary,indent=1));print(json.dumps(summary,indent=1))
