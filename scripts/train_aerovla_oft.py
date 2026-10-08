"""Train or verify AeroVLA-OFT on a visual-search dataset (LoRA on NF4, one 12 GB GPU).

  train:   --dataset <root> --output <checkpoint dir> [--steps N] [--limit-samples N for an overfit test]
           The directory receives the checkpoint with the lowest validation loss; `last/` holds the final
           weights. An existing checkpoint directory is never overwritten.
  verify:  --verify <checkpoint dir> --dataset <root>   reloads the checkpoint in a fresh process and
           compares its predictions with the ones stored at save time
  score:   --score <checkpoint dir> --dataset <root>    validation L1 of a checkpoint on any dataset
  probe:   --probe <checkpoint dir> --dataset <root>    the same frames with the sentence changed: does the action follow the words?

Which checkpoint is kept is decided by a rule given before training starts (--select-band): the lowest
validation L1, and among evaluations within the band of it the one whose first action most often has the
teacher's kind (advance or turn; stop or descend). With a band of 0 this is the lowest validation L1.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import json
import random
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.aerovla_oft.spec import load_config,normalize_action,denormalize_action,is_stop
from src.aerovla_oft.model import AeroVLAOFT

MANIFEST=ROOT/'outputs/integration/model-downloads.json'
AXES=('forward','down','yaw')
# Frames that can be drawn more often than their share: far, small targets in view, and the last part of a landing.
BOOSTS={'small_visible':lambda meta:meta.get('size_bucket')=='small' and meta['target_visible'],
        'landing':lambda meta:meta['teacher_state'] in ('final','descend','landed')}
GROUNDING=('advance','turn');TERMINAL=('stop','descend')


def behaviour(normalised,config):
    """The kind of a first action: stop, descend, climb, advance, or turn."""
    action=denormalize_action(normalised,config);forward,down,_=action
    if is_stop(action,config):return 'stop'
    if down>=.1 and forward<.15:return 'descend'
    if down<=-.1:return 'climb'
    return 'advance' if forward>=.15 else 'turn'


def select(kept,lowest,entry,band):
    """Should this evaluation replace the kept checkpoint? `kept` and `entry` carry val_l1 and score; `lowest` already includes the entry."""
    limit=lowest*(1+band)+1e-12
    if entry['val_l1']>limit:return False
    return kept is None or kept['val_l1']>limit or entry['score']>kept['score']+1e-9


def load_samples(root,name):return json.loads((Path(root)/f'{name}.json').read_text())


def mosaic(root,sample):
    """AeroVLA's input image: Front over Down, each 224 x 224, exactly as its dataset class builds it."""
    frames=[Image.open(Path(root)/sample['traj_rel_dir']/camera/sample['img_name']).convert('RGB').resize((224,224),Image.Resampling.BICUBIC)
            for camera in ('frontcamera','downcamera')]
    canvas=Image.new('RGB',(224,448),(0,0,0));canvas.paste(frames[0],(0,0));canvas.paste(frames[1],(0,224))
    return canvas


def batch_of(model,root,samples,config):
    pixels=model.pixel_values([mosaic(root,sample) for sample in samples])
    texts=[sample['instruction'].replace('<image>\n','',1) for sample in samples]
    targets=torch.tensor([[normalize_action(step,config) for step in sample['chunk']] for sample in samples],dtype=torch.float32)
    proprio=torch.tensor([sample['proprio'] for sample in samples],dtype=torch.float32) if config['proprio']['enabled'] else None
    return pixels,texts,targets,proprio


def stratified(samples,limit,seed=0):
    """A fixed subset with every teacher state represented in proportion (at least a few of each)."""
    if limit is None or limit>=len(samples):return list(samples)
    groups={}
    for sample in samples:groups.setdefault(sample['meta']['teacher_state'],[]).append(sample)
    chosen=[]
    for name in sorted(groups):
        items=groups[name];random.Random(f'{seed}|{name}').shuffle(items)
        chosen+=items[:max(min(8,len(items)),round(limit*len(items)/len(samples)))]
    return chosen[:max(limit,len(groups)*8)]


@torch.no_grad()
def evaluate(model,root,samples,config):
    """Mean L1 on normalised chunks: overall, for the executed first action, and by teacher state.

    Also how far the head leaves the normalised range, since anything beyond it is clipped when flown."""
    model.head.eval();total=first=0.;states={};predictions=[];outside=[0,0,0];any_outside=0;at_bound=[0,0,0];kinds=[]
    for sample in samples:
        pixels,texts,targets,proprio=batch_of(model,root,[sample],config)
        with torch.autocast('cuda',dtype=torch.bfloat16):predicted=model.forward(pixels,texts,proprio).float().cpu()
        error=(predicted-targets).abs();total+=error.mean().item();first+=error[:,0].mean().item()
        bucket=states.setdefault(sample['meta']['teacher_state'],{'error':0.,'count':0,'sum':[0.,0.,0.],'label':[0.,0.,0.]})
        bucket['error']+=error[:,0].mean().item();bucket['count']+=1
        for axis in range(3):
            value=predicted[0,0,axis].item();bucket['sum'][axis]+=value;bucket['label'][axis]+=targets[0,0,axis].item()
            outside[axis]+=abs(value)>1;at_bound[axis]+=abs(targets[0,0,axis].item())>=.999
        any_outside+=bool((predicted[0,0].abs()>1).any())
        predictions.append(predicted[0].tolist())
        kinds.append((behaviour(targets[0,0].tolist(),config),behaviour(predicted[0,0].tolist(),config),sample['meta'].get('task','approach')))
    model.head.train()
    count=max(1,len(samples))
    def accuracy(labels,task=None):
        chosen=[(label,got) for label,got,what in kinds if label in labels and (task is None or what==task)]
        return sum(label==got for label,got in chosen)/len(chosen) if chosen else None
    grounding,terminal=accuracy(GROUNDING),accuracy(TERMINAL)
    return {'l1':total/count,'first_action_l1':first/count,'samples':len(samples),
            # Does the first action have the teacher's kind? Advance or turn is the grounding decision; stop or descend is how a flight ends.
            'behaviour':{'grounding_accuracy':grounding,'terminal_accuracy':terminal,'all':accuracy(GROUNDING+TERMINAL+('climb',)),
                         'terminal_accuracy_land':accuracy(TERMINAL,'land'),'terminal_accuracy_approach':accuracy(TERMINAL,'approach'),
                         'score':sum(value for value in (grounding,terminal) if value is not None)/max(1,sum(value is not None for value in (grounding,terminal)))},
            'first_action_l1_by_state':{name:b['error']/b['count'] for name,b in states.items()},
            'first_action_mean_by_state':{name:[value/b['count'] for value in b['sum']] for name,b in states.items()},
            'label_mean_by_state':{name:[value/b['count'] for value in b['label']] for name,b in states.items()},
            'first_action_outside_range':any_outside/count,
            'outside_range_by_axis':dict(zip(AXES,(value/count for value in outside))),
            'labels_at_bound_by_axis':dict(zip(AXES,(value/count for value in at_bound)))},predictions


def train(args):
    output=Path(args.output)
    if (output/'manifest.json').exists():
        raise SystemExit(f'{output} already holds a checkpoint; choose another --output (checkpoints are never overwritten)')
    config=load_config(args.oft_config)
    if args.proprio is not None:config['proprio']['enabled']=args.proprio=='on'
    if args.head_output:config['head']['output']=args.head_output
    if args.name:config['name']=args.name
    settings=config['train'];random.seed(settings['seed']);torch.manual_seed(settings['seed'])
    samples=load_samples(args.dataset,args.train_file);validation=load_samples(args.dataset,args.val_file)
    if args.strategies:
        # Strategies whose labels depend on what the teacher remembers cannot be learned from one frame.
        samples=[s for s in samples if s['meta'].get('strategy','right') in args.strategies]
        validation=[s for s in validation if s['meta'].get('strategy','right') in args.strategies]
    if args.limit_samples:samples=samples[::max(1,len(samples)//args.limit_samples)][:args.limit_samples]
    validation=stratified(validation,args.eval_samples)
    torch.cuda.reset_peak_memory_stats()
    model=AeroVLAOFT(MANIFEST,config,checkpoint=args.init,trainable=True)
    learning_rate=args.lr or settings['learning_rate'];accumulation=args.accumulation or settings['gradient_accumulation']
    batch_size=args.batch or settings['batch_size']
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=learning_rate,weight_decay=settings['weight_decay'] if args.weight_decay is None else args.weight_decay)
    updates=args.steps;warmup=min(settings['warmup_steps'],max(1,updates//10))
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/warmup)*max(.1,1-step/max(1,updates)))
    history=[];started=time.time();running=None;best=None;stale=0;stopped_early=False
    # Searching, aligning and stopping are a minority of the frames but are what is being taught, so each
    # teacher state is drawn equally often up to `balance` times its natural share.
    counts={};weights=None
    for sample in samples:counts[sample['meta']['teacher_state']]=counts.get(sample['meta']['teacher_state'],0)+1
    if args.balance>1:
        weights=[min(args.balance,len(samples)/(len(counts)*counts[sample['meta']['teacher_state']])) for sample in samples]
    boosts={name:float(factor) for name,factor in (item.split('=') for item in args.boost)};boosted={}
    if boosts:
        weights=weights or [1.]*len(samples)
        for index,sample in enumerate(samples):
            for name,factor in boosts.items():
                if BOOSTS[name](sample['meta']):weights[index]*=factor;boosted[name]=boosted.get(name,0)+1
    lowest=float('inf')

    def record(update,final_train=None,final_val=None,stored=None):
        return {'dataset':str(args.dataset),'dataset_summary':json.loads((Path(args.dataset)/'summary.json').read_text()),
                'train_file':args.train_file,'val_file':args.val_file,'strategies':args.strategies,'init':args.init,
                'train_samples_used':len(samples),'val_samples_used':len(validation),'eval_samples':args.eval_samples,'updates':update+1,'planned_updates':updates,
                'batch_size':batch_size,'gradient_accumulation':accumulation,
                'samples_seen':(update+1)*batch_size*accumulation,'epochs':(update+1)*batch_size*accumulation/len(samples),
                'learning_rate':learning_rate,'balance':args.balance,'train_state_counts':counts,'boost':boosts,'boosted_samples':boosted,
                'selection':{'band':args.select_band,'rule':'lowest validation L1; among evaluations within the band of it, the highest mean of grounding and terminal accuracy'},
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
                'train_seconds':time.time()-started,'final_train':final_train,'final_val':final_val,'best':best,
                'stopped_early':stopped_early,'history':history,'gpu':torch.cuda.get_device_name(0),'reload_reference':stored}

    for update in range(updates):
        for _ in range(accumulation):
            chosen=random.choices(samples,weights=weights,k=batch_size)
            pixels,texts,targets,proprio=batch_of(model,args.dataset,chosen,config)
            loss,_=model.loss(pixels,texts,targets,proprio);(loss/accumulation).backward()
            running=loss.item() if running is None else .95*running+.05*loss.item()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.)
        optimizer.step();schedule.step();optimizer.zero_grad(set_to_none=True)
        last=update==updates-1
        if update%args.log_every==0 or last:
            entry={'update':update,'train_l1_ema':running,'elapsed_s':time.time()-started}
            if validation and ((update%args.eval_every==0 and update>0) or last):
                entry['val'],stored=evaluate(model,args.dataset,validation,config)
                # The checkpoint that is kept follows the rule fixed before training (see the top of this file), not the last evaluation.
                improved=entry['val']['l1']<lowest-1e-4;lowest=min(lowest,entry['val']['l1']);stale=0 if improved else stale+1
                candidate={'update':update,'val_l1':entry['val']['l1'],'val_first_action_l1':entry['val']['first_action_l1'],'train_l1_ema':running,
                           'score':entry['val']['behaviour']['score'],'behaviour':entry['val']['behaviour']}
                if select(best,lowest,candidate,args.select_band):
                    best=candidate;model.save(output,{'training':record(update,None,entry['val'],stored[:8])})
                entry['best_update']=best['update']
            history.append(entry);print(json.dumps(entry),flush=True)
            if args.patience and stale>=args.patience:stopped_early=True;break
    final_train,_=evaluate(model,args.dataset,stratified(samples,args.eval_samples),config)
    final_val,stored=evaluate(model,args.dataset,validation or stratified(samples,args.eval_samples),config)
    summary=record(update,final_train,final_val,stored[:8])
    if best is None:model.save(output,{'training':summary})
    else:
        model.save(output/'last',{'training':summary})
        # Complete the kept checkpoint's record with the whole run's history.
        kept=json.loads((output/'manifest.json').read_text())
        kept['training'].update(history=history,best=best,stopped_early=stopped_early,final_train_last=final_train,final_val_last=final_val,
                                updates_run=update+1,train_seconds=summary['train_seconds'],peak_allocated_GiB=summary['peak_allocated_GiB'],
                                peak_reserved_GiB=summary['peak_reserved_GiB'])
        (output/'manifest.json').write_text(json.dumps(kept,indent=1))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('history','reload_reference','dataset_summary')},indent=1))


def score(args):
    """L1 of a checkpoint on a dataset's validation file, with the same fixed subset the trainer uses."""
    saved=json.loads((Path(args.score)/'manifest.json').read_text());model=AeroVLAOFT(MANIFEST,saved['config'],checkpoint=args.score)
    samples=stratified(load_samples(args.dataset,args.val_file),args.eval_samples)
    metrics,_=evaluate(model,args.dataset,samples,saved['config'])
    result={'checkpoint':str(args.score),'dataset':str(args.dataset),'val_file':args.val_file,'metrics':metrics}
    print(json.dumps(result,indent=1))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=1))


@torch.no_grad()
def probe(args):
    """The same frames with only the sentence changed.

    Frames near a pad from landing episodes are shown with the landing sentence and with an approach sentence: the
    first should keep flying or descend, the second should stop. Frames that show the target are shown with the
    sentence naming the other pad: the vehicle should stop advancing on it."""
    saved=json.loads((Path(args.probe)/'manifest.json').read_text());config=saved['config'];model=AeroVLAOFT(MANIFEST,config,checkpoint=args.probe)
    samples=load_samples(args.dataset,args.val_file) or load_samples(args.dataset,args.train_file);rng=random.Random(0)
    def first(sample,text):
        pixels,_,_,proprio=batch_of(model,args.dataset,[sample],config)
        with torch.autocast('cuda',dtype=torch.bfloat16):return model.forward(pixels,[text],proprio).float().cpu()[0,0].tolist()
    near=[s for s in samples if s['meta'].get('task')=='land' and s['meta']['teacher_state'] in ('final','descend') and 'landing pad' in s['instruction']]
    seen=[s for s in samples if s['meta']['target_visible'] and s['meta']['teacher_state']=='approach' and s['meta']['target'] in ('blue_pad','red_pad')]
    rng.shuffle(near);rng.shuffle(seen);result={'checkpoint':str(args.probe),'dataset':str(args.dataset)}
    rows=[]
    for sample in near[:args.probe_samples]:
        text=sample['instruction'].replace('<image>\n','',1);noun='the blue landing pad' if 'blue' in text else 'the red landing pad'
        land=first(sample,text);approach=first(sample,f'Approach {noun}.')
        rows.append({'state':sample['meta']['teacher_state'],'distance_m':sample['meta']['distance_m'],'land':behaviour(land,config),'approach':behaviour(approach,config),
                     'difference':sum(abs(a-b) for a,b in zip(land,approach))/3,'approach_forward':denormalize_action(approach,config)[0],
                     'approach_down':denormalize_action(approach,config)[1]})
    def part(chosen):
        return {'frames':len(chosen),'land_sentence_keeps_going':sum(r['land'] in ('advance','descend') for r in chosen)/max(1,len(chosen)),
                'approach_sentence_stops':sum(r['approach']=='stop' for r in chosen)/max(1,len(chosen)),
                'approach_sentence_descends':sum(r['approach']=='descend' for r in chosen)/max(1,len(chosen)),
                'approach_sentence_kinds':{kind:sum(r['approach']==kind for r in chosen) for kind in sorted({r['approach'] for r in chosen})},
                'mean_action_difference':sum(r['difference'] for r in chosen)/max(1,len(chosen))}
    # An approach flight is never closer than where it stops, so frames from deeper inside a landing are new to that sentence.
    edge=9.
    result['task_swap']={**part(rows),'by_state':{state:part([r for r in rows if r['state']==state]) for state in ('final','descend')},
                         'where_an_approach_stops':part([r for r in rows if r['state']=='final' and r['distance_m']>=edge])}
    # The other way round: frames where an approach flight stopped, shown with the landing sentence.
    stops=[s for s in samples if s['meta'].get('task')=='approach' and s['meta']['teacher_state']=='stop' and 'landing pad' in s['instruction']]
    rng.shuffle(stops);rows=[]
    for sample in stops[:args.probe_samples]:
        text=sample['instruction'].replace('<image>\n','',1);noun='the blue landing pad' if 'blue' in text else 'the red landing pad'
        rows.append({'own':behaviour(first(sample,text),config),'land':behaviour(first(sample,f'Find {noun} and land on it.'),config)})
    result['stop_frames']={'frames':len(rows),'approach_sentence_stops':sum(r['own']=='stop' for r in rows)/max(1,len(rows)),
                           'land_sentence_keeps_going':sum(r['land'] in ('advance','descend') for r in rows)/max(1,len(rows))}
    rows=[]
    for sample in seen[:args.probe_samples]:
        text=sample['instruction'].replace('<image>\n','',1);other=text.replace('blue','\0').replace('red','blue').replace('\0','red')
        own=first(sample,text);swapped=first(sample,other)
        rows.append({'own':behaviour(own,config),'other':behaviour(swapped,config),'difference':sum(abs(a-b) for a,b in zip(own,swapped))/3})
    result['object_swap']={'frames':len(rows),'own_sentence_advances':sum(r['own']=='advance' for r in rows)/max(1,len(rows)),
                           'other_pad_named_still_advances':sum(r['other']=='advance' for r in rows)/max(1,len(rows)),
                           'mean_action_difference':sum(r['difference'] for r in rows)/max(1,len(rows))}
    print(json.dumps(result,indent=1))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=1))


def verify(args):
    saved=json.loads((Path(args.verify)/'manifest.json').read_text())
    config=saved['config'];model=AeroVLAOFT(MANIFEST,config,checkpoint=args.verify)
    training=saved['training'];name=training.get('val_file','val')
    samples=load_samples(args.dataset,name) or load_samples(args.dataset,training.get('train_file','train'))
    if training.get('strategies'):samples=[s for s in samples if s['meta'].get('strategy','right') in training['strategies']]
    reference=training['reload_reference']
    # Checkpoints written before the stratified validation subset used the first samples of the file.
    chosen=stratified(samples,training['eval_samples']) if 'eval_samples' in training else samples
    metrics,predicted=evaluate(model,args.dataset,chosen[:len(reference)],config)
    difference=max(abs(a-b) for p,r in zip(predicted,reference) for x,y in zip(p,r) for a,b in zip(x,y))
    result={'checkpoint':str(args.verify),'compared_samples':len(reference),'max_abs_difference':difference,'reload_ok':difference<.02,'metrics':metrics}
    (Path(args.verify)/'reload_check.json').write_text(json.dumps(result,indent=1));print(json.dumps(result,indent=1))
    if not result['reload_ok']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--output');parser.add_argument('--verify')
    parser.add_argument('--oft-config');parser.add_argument('--steps',type=int,default=300);parser.add_argument('--batch',type=int)
    parser.add_argument('--accumulation',type=int);parser.add_argument('--lr',type=float);parser.add_argument('--weight-decay',type=float)
    parser.add_argument('--limit-samples',type=int)
    parser.add_argument('--balance',type=float,default=3.,help='largest oversampling factor for a rare teacher state; 1 turns it off')
    parser.add_argument('--proprio',choices=('on','off'));parser.add_argument('--log-every',type=int,default=10)
    parser.add_argument('--eval-every',type=int,default=100);parser.add_argument('--eval-samples',type=int,default=120)
    parser.add_argument('--patience',type=int,default=0,help='stop after this many evaluations without a better validation loss; 0 keeps going')
    parser.add_argument('--train-file',default='train');parser.add_argument('--val-file',default='val')
    parser.add_argument('--strategies',nargs='+',help='keep only samples recorded with these teacher search strategies')
    parser.add_argument('--head-output',choices=('linear','tanh'));parser.add_argument('--name',help='model name stored with the checkpoint')
    parser.add_argument('--init',help='checkpoint to continue from')
    parser.add_argument('--score',help='checkpoint to evaluate on the validation file; --output then names a JSON file')
    parser.add_argument('--probe',help='checkpoint to show the same frames to with the sentence changed; --output then names a JSON file')
    parser.add_argument('--probe-samples',type=int,default=60)
    parser.add_argument('--boost',nargs='*',default=[],help='NAME=FACTOR: draw these frames more often (small_visible, landing)')
    parser.add_argument('--select-band',type=float,default=0.,help='relative band above the lowest validation L1 inside which the behaviour score decides')
    arguments=parser.parse_args()
    if arguments.probe:probe(arguments)
    elif arguments.score:score(arguments)
    elif arguments.verify:verify(arguments)
    else:train(arguments)
