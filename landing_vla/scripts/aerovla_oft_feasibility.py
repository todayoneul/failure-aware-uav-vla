"""Can the OFT-style head be trained on this GPU? Synthetic batch, real model, measured memory and time.

Fits a handful of fixed random-image samples to fixed random chunks. It only shows that gradients reach
the new LoRA and head and what that costs; it says nothing about flying.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.aerovla_oft.spec import load_config
from src.aerovla_oft.model import AeroVLAOFT


def main(args):
    config=load_config();config['proprio']['enabled']=args.proprio
    torch.cuda.reset_peak_memory_stats()
    model=AeroVLAOFT(ROOT/'outputs/integration/model-downloads.json',config,trainable=True)
    loaded=torch.cuda.max_memory_allocated()/2**30
    if args.checkpointing:model.core.language_model.gradient_checkpointing_enable()
    generator=np.random.default_rng(0)
    mosaics=[Image.fromarray(generator.integers(0,256,(448,224,3),dtype=np.uint8)) for _ in range(args.samples)]
    texts=['Find the blue cone.','Approach the orange ball.']*args.samples
    targets=torch.tensor(generator.uniform(-1,1,(args.samples,config['chunk_size'],3)),dtype=torch.float32)
    proprio=torch.tensor(generator.uniform(-1,1,(args.samples,len(config['proprio']['fields']))),dtype=torch.float32)
    pixels=model.pixel_values(mosaics)
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=args.lr)
    losses=[];times=[]
    for step in range(args.steps):
        index=[(step*args.batch+i)%args.samples for i in range(args.batch)]
        torch.cuda.synchronize();start=time.perf_counter()
        loss,_=model.loss(pixels[index],[texts[i] for i in index],targets[index],proprio[index] if args.proprio else None)
        loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();times.append(time.perf_counter()-start);losses.append(loss.item())
        if step%10==0:print(f'step {step} loss {loss.item():.4f} {times[-1]:.2f}s',flush=True)
    peak=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
    model.head.eval();latency=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(8):
            torch.cuda.synchronize();start=time.perf_counter()
            model.forward(pixels[:1],texts[:1],proprio[:1] if args.proprio else None)
            torch.cuda.synchronize();latency.append(time.perf_counter()-start)
    result={'gpu':torch.cuda.get_device_name(0),'batch':args.batch,'gradient_checkpointing':args.checkpointing,'proprio':args.proprio,
            'loaded_GiB':loaded,'peak_allocated_GiB':peak,'peak_reserved_GiB':reserved,'median_step_s':float(np.median(times[2:])),
            'first_loss':losses[0],'last_loss':float(np.mean(losses[-5:])),'steps':args.steps,
            'forward_ms':float(np.median(latency[2:])*1000),**model.load_info}
    print(json.dumps(result,indent=1))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=1))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=60);parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--samples',type=int,default=4);parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--checkpointing',action='store_true');parser.add_argument('--proprio',action='store_true')
    parser.add_argument('--output');main(parser.parse_args())
