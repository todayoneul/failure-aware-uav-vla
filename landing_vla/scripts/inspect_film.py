"""What a trained FiLM module does with different sentences; CPU only, the base model is not run.

  inspect_film.py --checkpoint <dir with a FiLM module> [--output JSON]

Reads the language model's embedding table from the base snapshot (nothing else of it), embeds a set of instructions the way
the model does (the sentence alone, mean of its token embeddings) and reports gamma and beta: how large the modulation is,
how much of it changes with the sentence, and how alike the modulations of two sentences are. A module whose gamma and beta did not depend on the sentence would be a fixed change of the features, not
language conditioning.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from src.aerovla_oft.model import VisualFiLM

MANIFEST=ROOT/'outputs/integration/model-downloads.json'
SENTENCES=('Find the blue landing pad and land on it.','Find the red landing pad and land on it.','Land on the blue landing pad.','Land on the red landing pad.',
           'Approach the blue landing pad.','Find the blue landing pad.','Find the red landing pad.','Find the blue cube.','Find the red cube.','Find the blue cylinder.',
           'Find the blue cone.','Find the green cylinder.','Find the orange ball.','Approach the blue cube.','Approach the red cube.')


def embedding_table(snapshot):
    index=json.loads((Path(snapshot)/'model.safetensors.index.json').read_text())['weight_map']
    name=next(key for key in index if key.endswith('embed_tokens.weight'))
    with safe_open(str(Path(snapshot)/index[name]),framework='pt') as file:return file.get_tensor(name).float()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output')
    args=parser.parse_args();saved=json.loads((Path(args.checkpoint)/'manifest.json').read_text());settings=saved['config']['grounding']
    if settings['type']!='film':raise SystemExit('this checkpoint has no FiLM module')
    snapshot=next(item['snapshot'] for item in json.loads(MANIFEST.read_text())['models'] if item['repo']=='openvla/openvla-7b')
    tokenizer=AutoTokenizer.from_pretrained(snapshot,trust_remote_code=True,local_files_only=True);table=embedding_table(snapshot)
    state=torch.load(Path(args.checkpoint)/'head.pt',map_location='cpu')['grounding'];visual=state['net.3.bias'].numel()//2
    module=VisualFiLM(table.shape[1],visual,settings['hidden_dim']);module.load_state_dict(state);module.eval()
    with torch.no_grad():
        pooled=torch.stack([table[tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids[0]].mean(0) for text in SENTENCES])
        delta=module.net(pooled)
    gamma,beta=delta[:,:visual],delta[:,visual:];common=delta.mean(0,keepdim=True);own=delta-common
    # Two sentences' modulations side by side: the angle between them, and how far apart they are against their own size.
    def pair(a,b):
        x,y=delta[SENTENCES.index(a)],delta[SENTENCES.index(b)]
        return {'cosine':round(float(torch.nn.functional.cosine_similarity(x,y,dim=0)),3),'distance_over_size':round(float((x-y).norm()/((x.norm()+y.norm())/2)),3)}
    result={'checkpoint':str(args.checkpoint),'visual_dim':visual,'sentences':len(SENTENCES),
            'gamma_minus_one':{'mean_abs':float(gamma.abs().mean()),'max_abs':float(gamma.abs().max())},'beta':{'mean_abs':float(beta.abs().mean()),'max_abs':float(beta.abs().max())},
            'by_encoder':{'dinov2_first_1024':{'gamma':float(gamma[:,:1024].abs().mean()),'beta':float(beta[:,:1024].abs().mean())},
                          'siglip_rest':{'gamma':float(gamma[:,1024:].abs().mean()),'beta':float(beta[:,1024:].abs().mean())}},
            # How much of the modulation is the sentence's own: energy of what is left after the part every sentence shares.
            'share_that_depends_on_the_sentence':float(own.pow(2).sum()/delta.pow(2).sum()),
            'pairs':{
                'blue pad land / red pad land':pair(SENTENCES[0],SENTENCES[1]),'blue pad land / blue pad land (other words)':pair(SENTENCES[0],SENTENCES[2]),
                'blue pad land / blue pad approach':pair(SENTENCES[0],SENTENCES[4]),'blue pad find / blue cube find':pair(SENTENCES[5],SENTENCES[7]),
                'red pad find / red cube find':pair(SENTENCES[6],SENTENCES[8]),'blue cube find / red cube find':pair(SENTENCES[7],SENTENCES[8]),
                'blue cube find / blue cube approach':pair(SENTENCES[7],SENTENCES[13]),'blue pad find / orange ball find':pair(SENTENCES[5],SENTENCES[12])},
            'norm_by_sentence':{text:round(float(row.norm()),3) for text,row in zip(SENTENCES,delta)}}
    print(json.dumps(result,indent=1))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=1))


if __name__=='__main__':main()
