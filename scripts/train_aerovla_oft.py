"""Train or verify AeroVLA-OFT on a visual-search dataset (LoRA on NF4, one 12 GB GPU).

  train:   --dataset <root> --output <checkpoint dir> [--steps N] [--limit-samples N for an overfit test]
  verify:  --verify <checkpoint dir> --dataset <root>   reloads the checkpoint in a fresh process and
           compares its predictions with the ones stored at save time
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
from src.aerovla_oft.spec import load_config,normalize_action
from src.aerovla_oft.model import AeroVLAOFT

MANIFEST=ROOT/'outputs/integration/model-downloads.json'


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


@torch.no_grad()
def evaluate(model,root,samples,config,limit=None):
    """Mean L1 on normalised chunks, overall, for the executed first action, and by teacher state."""
    model.head.eval();total=first=0.;states={};predictions=[]
    chosen=samples if limit is None else samples[:limit]
    for sample in chosen:
        pixels,texts,targets,proprio=batch_of(model,root,[sample],config)
        with torch.autocast('cuda',dtype=torch.bfloat16):predicted=model.forward(pixels,texts,proprio).float().cpu()
        error=(predicted-targets).abs();total+=error.mean().item();first+=error[:,0].mean().item()
        bucket=states.setdefault(sample['meta']['teacher_state'],[0.,0]);bucket[0]+=error[:,0].mean().item();bucket[1]+=1
        predictions.append(predicted[0].tolist())
    model.head.train()
    count=max(1,len(chosen))
    return {'l1':total/count,'first_action_l1':first/count,'samples':len(chosen),
            'first_action_l1_by_state':{name:value/number for name,(value,number) in states.items()}},predictions


def train(args):
    config=load_config(args.oft_config)
    if args.proprio is not None:config['proprio']['enabled']=args.proprio=='on'
    settings=config['train'];random.seed(settings['seed']);torch.manual_seed(settings['seed'])
    samples=load_samples(args.dataset,'train');validation=load_samples(args.dataset,'val')
    if args.limit_samples:samples=samples[::max(1,len(samples)//args.limit_samples)][:args.limit_samples]
    torch.cuda.reset_peak_memory_stats()
    model=AeroVLAOFT(MANIFEST,config,trainable=True)
    learning_rate=args.lr or settings['learning_rate'];accumulation=args.accumulation or settings['gradient_accumulation']
    batch_size=args.batch or settings['batch_size']
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=learning_rate,weight_decay=settings['weight_decay'])
    updates=args.steps;warmup=min(settings['warmup_steps'],max(1,updates//10))
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/warmup)*max(.1,1-step/max(1,updates)))
    history=[];started=time.time();running=None
    # Searching, aligning and stopping are a minority of the frames but are what is being taught, so each
    # teacher state is drawn equally often up to `balance` times its natural share.
    counts={};weights=None
    for sample in samples:counts[sample['meta']['teacher_state']]=counts.get(sample['meta']['teacher_state'],0)+1
    if args.balance>1:
        weights=[min(args.balance,len(samples)/(len(counts)*counts[sample['meta']['teacher_state']])) for sample in samples]
    for update in range(updates):
        for _ in range(accumulation):
            chosen=random.choices(samples,weights=weights,k=batch_size)
            pixels,texts,targets,proprio=batch_of(model,args.dataset,chosen,config)
            loss,_=model.loss(pixels,texts,targets,proprio);(loss/accumulation).backward()
            running=loss.item() if running is None else .95*running+.05*loss.item()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.)
        optimizer.step();schedule.step();optimizer.zero_grad(set_to_none=True)
        if update%args.log_every==0 or update==updates-1:
            entry={'update':update,'train_l1_ema':running,'elapsed_s':time.time()-started}
            if validation and (update%args.eval_every==0 or update==updates-1):
                entry['val'],_=evaluate(model,args.dataset,validation,config,args.eval_samples)
            history.append(entry);print(json.dumps(entry),flush=True)
    final_train,_=evaluate(model,args.dataset,samples,config,args.eval_samples)
    final_val,stored=evaluate(model,args.dataset,validation or samples,config,args.eval_samples)
    record={'dataset':str(args.dataset),'dataset_summary':json.loads((Path(args.dataset)/'summary.json').read_text()),
            'train_samples_used':len(samples),'updates':updates,'batch_size':batch_size,'gradient_accumulation':accumulation,
            'samples_seen':updates*batch_size*accumulation,'epochs':updates*batch_size*accumulation/len(samples),
            'learning_rate':learning_rate,'balance':args.balance,'train_state_counts':counts,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'train_seconds':time.time()-started,
            'final_train':final_train,'final_val':final_val,'history':history,'gpu':torch.cuda.get_device_name(0),
            'reload_reference':stored[:8]}
    model.save(args.output,{'training':record})
    print(json.dumps({k:v for k,v in record.items() if k not in ('history','reload_reference','dataset_summary')},indent=1))


def verify(args):
    saved=json.loads((Path(args.verify)/'manifest.json').read_text())
    config=saved['config'];model=AeroVLAOFT(MANIFEST,config,checkpoint=args.verify)
    samples=load_samples(args.dataset,'val') or load_samples(args.dataset,'train')
    reference=saved['training']['reload_reference']
    metrics,predicted=evaluate(model,args.dataset,samples,config,len(reference))
    difference=max(abs(a-b) for p,r in zip(predicted,reference) for x,y in zip(p,r) for a,b in zip(x,y))
    result={'checkpoint':str(args.verify),'compared_samples':len(reference),'max_abs_difference':difference,'reload_ok':difference<.02,'metrics':metrics}
    (Path(args.verify)/'reload_check.json').write_text(json.dumps(result,indent=1));print(json.dumps(result,indent=1))
    if not result['reload_ok']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--output');parser.add_argument('--verify')
    parser.add_argument('--oft-config');parser.add_argument('--steps',type=int,default=300);parser.add_argument('--batch',type=int)
    parser.add_argument('--accumulation',type=int);parser.add_argument('--lr',type=float);parser.add_argument('--limit-samples',type=int)
    parser.add_argument('--balance',type=float,default=3.,help='largest oversampling factor for a rare teacher state; 1 turns it off')
    parser.add_argument('--proprio',choices=('on','off'));parser.add_argument('--log-every',type=int,default=10)
    parser.add_argument('--eval-every',type=int,default=100);parser.add_argument('--eval-samples',type=int,default=120)
    arguments=parser.parse_args()
    verify(arguments) if arguments.verify else train(arguments)
