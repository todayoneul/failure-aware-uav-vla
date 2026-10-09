"""A grounding module that has not been trained must leave the model exactly as it was.

  check_grounding_identity.py --checkpoint <dir without a module> --dataset <root> [--modules film cross_attention]

Loads the checkpoint once per module with the module added and untrained, and compares the predictions on the frames the
checkpoint stored at save time with the stored ones. Exit 1 if any differs by more than the reload tolerance.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.train_aerovla_oft import MANIFEST,load_samples,stratified,evaluate
from src.aerovla_oft.model import AeroVLAOFT
from src.visual_search.episodes import load_config as load_search_config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--dataset',required=True)
    parser.add_argument('--modules',nargs='+',default=['film','cross_attention']);parser.add_argument('--output')
    args=parser.parse_args();saved=json.loads((Path(args.checkpoint)/'manifest.json').read_text());training=saved['training'];reference=training['reload_reference']
    samples=stratified(load_samples(args.dataset,training.get('val_file','val')),training['eval_samples'])[:len(reference)]
    experiments=load_search_config()['grounding_architecture']['experiments'];result={'checkpoint':str(args.checkpoint),'frames':len(reference),'modules':{}};worst=0.
    for name in args.modules:
        config=dict(saved['config'],grounding=experiments[name]['module']);model=AeroVLAOFT(MANIFEST,config,checkpoint=args.checkpoint)
        _,predicted=evaluate(model,args.dataset,samples,config)
        difference=max(abs(a-b) for p,r in zip(predicted,reference) for x,y in zip(p,r) for a,b in zip(x,y));worst=max(worst,difference)
        result['modules'][name]={'max_abs_difference':difference,'grounding_parameters':model.load_info['grounding_parameters'],'module_loaded_from_checkpoint':model.load_info['grounding_loaded']}
        del model;torch.cuda.empty_cache()
    result['identity']=worst<.02
    print(json.dumps(result,indent=1))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=1))
    if not result['identity']:raise SystemExit(1)


if __name__=='__main__':main()
