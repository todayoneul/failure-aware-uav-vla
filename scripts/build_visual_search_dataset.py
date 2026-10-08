"""Turn recorded teacher episodes into an AeroVLA-format training file with OFT action chunks.

Each sample keeps AeroVLA's fields (traj_rel_dir, img_name, instruction, label, is_last_step,
is_penultimate) and adds the future chunk, the optional proprio vector and analysis-only metadata.
Episodes planned by scripts/plan_generalization.py carry their split (decided by seed range) and are
split accordingly; older recordings are split at random by episode. Either way no trajectory has
frames on both sides, and the summary says how the data is distributed. No simulator, no torch.

  build_visual_search_dataset.py ROOT                       one recorded dataset, written into ROOT
  build_visual_search_dataset.py ROOT --sources A B ...     several recorded datasets merged into ROOT; nothing is
                                                            copied, samples point at the source folders, and the
                                                            merge stops if an episode, seed, start or folder repeats
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
                                'height_m':step.get('height_m'),'teacher_state':step['teacher_state'],
                                # Analysis only: what the sentence asks for, whether the vehicle stands on a pad, and how large
                                # the target's bounding box is in the front image (share of the image; never a model input).
                                'task':summary.get('task','approach'),'landed':bool(step.get('landed')),'over':bool(step.get('over')),
                                'target_size':((step.get('target_size') or {}).get('front') or {}).get('image_fraction')}})
    return samples


def size_buckets(samples):
    """Small, medium and large by the thirds of the recorded sizes of the frames that show the target. Returns the two edges."""
    sizes=sorted(s['meta']['target_size'] for s in samples if s['meta']['target_visible'] and s['meta'].get('target_size'))
    if len(sizes)<3:return None
    edges=[sizes[len(sizes)//3],sizes[2*len(sizes)//3]]
    for sample in samples:
        size=sample['meta'].get('target_size')
        sample['meta']['size_bucket']=None if not (size and sample['meta']['target_visible']) else 'small' if size<edges[0] else 'medium' if size<edges[1] else 'large'
    return edges


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


def load_records(root,prefix=''):
    """Recorded episodes of one dataset; `prefix` is put in front of each trajectory folder when datasets are merged."""
    records=[json.loads(path.read_text()) for path in sorted((Path(root)/'episodes').glob('*.json'))]
    for record in records:
        record['traj_rel_dir']=prefix+record['traj_rel_dir'];record['source']=Path(root).name
    return records


def collisions(records):
    """Anything two recordings share that would make them the same data twice."""
    def repeated(values):
        counts=collections.Counter(values);return sorted(str(value) for value,count in counts.items() if count>1)
    return {'episode_ids':repeated(record['summary']['id'] for record in records),
            # A landing start flown again as an approach shares its seed and its pose on purpose; the task tells the two apart.
            'seeds':repeated((record['summary'].get('map','blocks'),record['summary'].get('seed'),record['summary'].get('task','approach'),
                              record['summary']['target'] if record['summary'].get('twin_of') else None)
                             for record in records if 'split' in record['summary']),
            'folders':repeated(record['traj_rel_dir'] for record in records),
            'starts':repeated(tuple(record['summary']['start_xy'])+(record['summary']['target'],record['summary'].get('task','approach'))
                              for record in records if 'start_xy' in record['summary'])}


def build(root,config,validation_fraction,seed):
    return build_records(load_records(root),config,validation_fraction,seed)


def build_records(everything,config,validation_fraction,seed):
    # A landing episode is kept only if the teacher actually landed by the landing rule.
    records=[record for record in everything if record['steps'] and not record['summary'].get('error') and record['summary']['reason']=='teacher_stop'
             and (record['summary'].get('task','approach')!='land' or record['summary'].get('land_success'))]
    dropped=count([record for record in everything if record not in records],lambda r:r['summary'].get('reason','error'))
    planned=all(record['summary'].get('split') in ('train','val') for record in records)
    if planned:validation_ids={record['summary']['id'] for record in records if record['summary']['split']=='val'}
    else:
        # A start flown twice with different sentences stays on one side: the two flights share their first frames.
        group=lambda record:record['summary'].get('twin_of') or record['summary']['id']
        chosen=set(split_episodes([group(record) for record in records],validation_fraction,seed)[1])
        validation_ids={record['summary']['id'] for record in records if group(record) in chosen}
    splits={'train':[],'val':[]}
    for record in records:
        splits['val' if record['summary']['id'] in validation_ids else 'train']+=episode_samples(record,config['chunk_size'])
    samples=splits['train']+splits['val'];bounds=config['action_bounds'];summaries=[record['summary'] for record in records]
    edges=size_buckets(samples)
    outside=sum(any(not low-1e-9<=value<=high+1e-9 for value,(low,high) in zip(step,bounds.values())) for sample in samples for step in sample['chunk'])
    search=[s for s in summaries if not s.get('initially_seen',True) or s.get('base',s.get('kind')) in ('search','altitude','reacquire')]
    climbed=[s for s in search if s.get('climbed_m',0.)>=1.]
    summary={'sources':count(records,lambda r:r.get('source','')),
             'episodes':len(records),'train_episodes':len(records)-len(validation_ids),'val_episodes':len(validation_ids),
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
             'tasks':count(summaries,lambda s:s.get('task','approach')),'samples_by_task':count(samples,lambda s:s['meta']['task']),
             'landing_episodes':sum(bool(s.get('land_success')) for s in summaries),
             'apparent_size':None if edges is None else {'what':"share of the front image covered by the target's bounding box, frames that show the target",
                                                         'small_below':edges[0],'large_from':edges[1],
                                                         'samples':count([s for s in samples if s['meta'].get('size_bucket')],lambda s:s['meta']['size_bucket']),
                                                         'frames_without_a_recorded_size':sum(s['meta'].get('target_size') is None for s in samples)},
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
    parser.add_argument('--sources',nargs='+',help='recorded datasets to merge; they must sit next to ROOT')
    args=parser.parse_args();root=Path(args.root)
    if args.sources:
        records=[]
        for source in args.sources:
            if Path(source).resolve().parent!=root.resolve().parent:raise SystemExit(f'{source} must be in the same folder as {root}')
            records+=load_records(source,f'../{Path(source).name}/')
        clashes=collisions(records)
        if any(clashes.values()):raise SystemExit(f'The datasets overlap: {clashes}')
        root.mkdir(parents=True,exist_ok=True)
        splits,summary=build_records(records,load_config(args.oft_config),args.validation_fraction,args.seed)
    else:splits,summary=build(root,load_config(args.oft_config),args.validation_fraction,args.seed)
    if any(summary['leaks'].values()):raise SystemExit(f'Training and validation overlap: {summary["leaks"]}')
    for name,samples in splits.items():(root/f'{name}.json').write_text(json.dumps(samples))
    (root/'summary.json').write_text(json.dumps(summary,indent=1));print(json.dumps(summary,indent=1))
