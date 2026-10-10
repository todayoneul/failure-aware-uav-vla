"""GPU smoke test of the interactive mission's policy: load the checkpoint the way the mission does, say what was read from
it, and ask for one action from a Front and a Down frame and one sentence. No simulator and no training; run in WSL.

  check_mission_model.py [--checkpoint DIR] [--front PNG --down PNG] [--target blue_pad] [--task land]
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint');parser.add_argument('--front');parser.add_argument('--down')
    parser.add_argument('--target',default='blue_pad');parser.add_argument('--task',default='land');parser.add_argument('--output')
    args=parser.parse_args()
    from src.mission import semantic
    from src.mission.oft_runner import Loader,WatchedPolicy
    from src.visual_search.episodes import load_config,policy_inputs
    config=load_config();demo=semantic.load_demo();settings=demo['mission']
    checkpoint=Path(args.checkpoint or settings['checkpoint']);checkpoint=checkpoint if checkpoint.is_absolute() else ROOT/checkpoint
    saved=json.loads((checkpoint/'manifest.json').read_text(encoding='utf-8'))['config']
    loader=Loader(checkpoint,saved);loader.thread.join()
    if loader.error:raise SystemExit(loader.error)
    target=next(item for item in semantic.semantic_targets(demo) if item['object_id']==args.target)
    semantic.check_start(target,args.task);text=semantic.sentence(config,demo,target,args.task)
    def frame(path,shade):
        if path:
            image=cv2.imread(str(path))
            if image is None or image.shape!=(256,256,3):raise SystemExit(f'{path}: a 256x256 colour image is needed')
            return image
        return np.full((256,256,3),shade,dtype=np.uint8)
    class Seen:
        """Stands in for the display: keeps what the policy was given and what it answered."""
        def model_input(self,front,down,instruction,proprio):self.given=(front.shape,down.shape,instruction,proprio)
        def model_output(self,result):self.result=result
    seen=Seen();inputs=policy_inputs(frame(args.front,110),frame(args.down,90),text)
    result=WatchedPolicy(loader.model,seen).infer(inputs['front'],inputs['down'],inputs['instruction'],inputs['proprio'])
    import torch
    report={'checkpoint':checkpoint.name,'loaded':loader.info,'given_to_the_model':{'front':seen.given[0],'down':seen.given[1],'instruction':seen.given[2],'proprio':seen.given[3]},
            'prompt':result['prompt'],'chunk':result['chunk'],'first_action':dict(zip(saved['action_bounds'],result['chunk'][0])),'stop':result['stop'],
            'inference_ms':result['inference_ms'],'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'gpu':torch.cuda.get_device_name(0)}
    print(json.dumps(report,indent=1))
    if args.output:Path(args.output).write_text(json.dumps(report,indent=1),encoding='utf-8')


if __name__=='__main__':main()
