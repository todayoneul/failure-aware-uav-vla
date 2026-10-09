"""Gen-v3c regression: the starts of the earlier sets flown again, beside what earlier checkpoints did on the same starts.

  gen_v3c_regression.py --output DIR [--runs outputs/visual_search]

Reads the run folders gen_v2_<set>, gen_v3b_<set> and gen_v3c_reg_<set> for G1, G3, P, L1, L2, L3 and keeps only the
starts named by `scripts/gen_v3c.py regression`. No simulator, no model.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gen_v3c import regression_ids
from scripts.freeze_gen_v3 import read,markdown
from scripts.summarize_visual_search import load
from src.visual_search.episodes import load_config
from src.visual_search.maps import load_landing

MODELS=(('Gen-v2','gen_v2_{}'),('Gen-v3 (second)','gen_v3b_{}'),('Gen-v3c','gen_v3c_reg_{}'))
GROUPS=(('G1 - unseen start (Blocks)',('G1',)),('G3 - held-out scene (Yard)',('G3',)),('P - unseen words',('P',)),('L - start 40-90 m',('L1','L2','L3')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--runs',default='outputs/visual_search')
    args=parser.parse_args();config=load_config();landing=load_landing();chosen=regression_ids(config);geometries={};maps={};records={};missing=[]
    for model,pattern in MODELS:
        for name,ids in chosen.items():
            folder=ROOT/args.runs/pattern.format(name.lower())
            if not (folder/'results.json').exists():missing.append(str(folder));continue
            _,episodes,steps=load(folder);data=json.loads((folder/'results.json').read_text());by_id={episode['id']:episode for episode in episodes}
            for episode_id in ids:
                if episode_id in by_id:
                    records[(model,name,episode_id)]=read(by_id[episode_id],steps[episode_id],data['config']['success_radius_m'],landing,geometries,maps)
    rows=[];summary={'starts':chosen,'missing_runs':missing,'groups':{},'changed':[]}
    def cell(items):return f'{sum(i["success"] for i in items)}/{len(items)}' if items else '-'
    for title,names in GROUPS+(('all',tuple(chosen)),):
        line=[title];summary['groups'][title]={}
        for model,_ in MODELS:
            items=[records[(model,name,episode_id)] for name in names for episode_id in chosen[name] if (model,name,episode_id) in records]
            line.append(cell(items));summary['groups'][title][model]={'success':sum(i['success'] for i in items),'flights':len(items),'collisions':sum(i['collision'] for i in items),
                                                                    'acquired':sum(i['acquired'] for i in items),'grounded':sum(i['grounded'] for i in items),
                                                                    'self_stop':sum(i['self_stop'] for i in items)}
        rows.append(tuple(line))
    markdown('Regression starts: successes',('Set',)+tuple(model for model,_ in MODELS),rows)
    # What Gen-v3c did differently from the checkpoint it was corrected from, start by start.
    changed=[]
    for name,ids in chosen.items():
        for episode_id in ids:
            before=records.get((MODELS[1][0],name,episode_id));after=records.get((MODELS[2][0],name,episode_id))
            if before and after and before['success']!=after['success']:
                changed.append((episode_id,'lost' if before['success'] else 'gained',f'{after["failure"]} ({after["mechanism"]})' if after['failure'] else f'was {before["failure"]} ({before["mechanism"]})'))
    markdown('Starts where Gen-v3c and the checkpoint it started from differ',('Start','Gen-v3c','How'),changed)
    summary['changed']=[dict(zip(('start','gen_v3c','how'),row)) for row in changed]
    detail=[]
    for (model,name,episode_id),record in records.items():
        detail.append({'model':model,'set':name,'episode':episode_id,**{key:record[key] for key in ('success','acquired','grounded','approached','self_stop','collision','failure','mechanism','steps','final_distance_m')}})
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'regression.json').write_text(json.dumps({**summary,'episodes':detail},indent=1))


if __name__=='__main__':main()
