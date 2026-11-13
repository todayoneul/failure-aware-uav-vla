"""Demo media from recorded visual-search runs; no simulator, no model.

  gif           one recorded episode drawn with the observer's own renderer, frame by frame
  trajectories  top-down paths of several runs on the simulator's map picture, one row per run
  flown         paths of generalisation runs on the map's own boxes, one panel per run and layout
"""
import argparse
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.visual_search_viewer import render
from src.mission.geometry import world_to_pixel

INK=(40,40,40);PAPER=(245,243,240)
TARGET={'blue_cone':(200,150,20),'orange_ball':(20,120,240)}
CASES={'A':'A  target centred','B':'B  target at the edge','C':'C  target not visible','D':'D  lost after a forced turn','L':'L  shared start'}
NAMES={'baseline':'AeroVLA, sentence only','oft':'AeroVLA-OFT, sentence only','teacher':'Teacher (uses the target position)'}


LEVELS={'G0':'G0 - seen start','G1':'G1 - unseen start','G2':'G2 - unseen object','G3':'G3 - held-out scene','G4':'G4 - held-out scene and object',
        'P':'G1 - unseen words','S':'G1 - side probe'}


def export_gif(record,episode,output,scale,seconds,hold,model_name=None,every=1):
    from src.visual_search.maps import load_map
    data=json.loads((Path(record)/'episodes'/f'{episode}.json').read_text());summary=data['summary'];frames=[]
    directory=Path(record)/data['traj_rel_dir'];map_config=load_map(summary.get('map','blocks'))
    for index,step in enumerate(data['steps']):
        # A long flight is thinned to every n-th decision; the last one is always kept.
        if index%every and index!=len(data['steps'])-1:continue
        front=cv2.imread(str(directory/'frontcamera'/step['img_name']));down=cv2.imread(str(directory/'downcamera'/step['img_name']))
        live={'mode':'VISUAL SEARCH','model':summary['policy'],'model_name':model_name,'episode':summary,'step':step,
              'map':map_config['name'],'target':map_config['objects'][summary['target']]['noun'],'level':LEVELS.get(summary.get('set')),
              'success_radius_m':data['summary'].get('success_radius_m',15.),'failure':'NORMAL','raw_sha256':step['input_sha256']}
        last=index==len(data['steps'])-1
        result={'episode':summary['id'],'reason':summary['reason'],'stopped':summary['stopped'],'final_distance_m':summary['final_distance_m']} if last else None
        canvas=render(live,front,down,result)
        canvas=cv2.resize(canvas,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)))
    palette=frames[len(frames)//2].quantize(colors=128,method=Image.Quantize.MEDIANCUT)
    frames=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
    durations=[int(seconds*1000)]*(len(frames)-1)+[int(hold*1000)]
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True)
    print(f'{output}: {len(frames)} frames, {Path(output).stat().st_size/2**20:.2f} MiB')


def load_run(path):
    data=json.loads((Path(path)/'results.json').read_text());steps={}
    for line in (Path(path)/'steps.jsonl').read_text().splitlines():
        row=json.loads(line);steps.setdefault(row['episode'],[]).append(row['position'])
    return data,steps


def export_trajectories(runs,map_directory,output,cases):
    image=cv2.imread(str(Path(map_directory)/'map.png'));meta=json.loads((Path(map_directory)/'map.json').read_text());scale=2.4
    base=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    def pixel(point):
        u,v=world_to_pixel([point[0],point[1],-1.],meta);return int(round(u*scale)),int(round(v*scale))
    (left,top),(right,bottom)=pixel([122.,-78.]),pixel([18.,74.]);rows=[]
    left,top=max(0,left),max(0,top);right,bottom=min(base.shape[1],right),min(base.shape[0],bottom)
    for run in runs:
        data,steps=load_run(run);radius=data['config']['success_radius_m'];panels=[]
        for case in cases:
            panel=base.copy();items=[e for e in data['episodes'] if e.get('case')==case and not e.get('error')]
            for episode in items:
                color=TARGET[episode['target']]
                path=[pixel(p) for p in steps.get(episode['id'],[])]
                target=target_centre[episode['target']]
                edge=pixel([target[0]+radius,target[1]]);middle=pixel(target)
                cv2.circle(panel,middle,int(math.hypot(edge[0]-middle[0],edge[1]-middle[1])),color,1,cv2.LINE_AA)
                for a,b in zip(path,path[1:]):cv2.line(panel,a,b,color,2,cv2.LINE_AA)
                if path:
                    cv2.circle(panel,path[0],5,(255,255,255),-1,cv2.LINE_AA);cv2.circle(panel,path[0],5,INK,1,cv2.LINE_AA)
                    if episode['success']:cv2.circle(panel,path[-1],7,color,-1,cv2.LINE_AA);cv2.circle(panel,path[-1],7,(255,255,255),1,cv2.LINE_AA)
                    else:cv2.rectangle(panel,(path[-1][0]-6,path[-1][1]-6),(path[-1][0]+6,path[-1][1]+6),(40,40,230),2)
            panel=panel[top:bottom,left:right].copy()
            done=sum(e['success'] for e in items)
            cv2.rectangle(panel,(0,0),(panel.shape[1],30),PAPER,-1)
            cv2.putText(panel,f'{CASES[case]}   {done}/{len(items)}' if items else f'{CASES[case]}   not run',(8,21),0,.55,INK,1,cv2.LINE_AA)
            panels.append(panel)
        row=np.hstack(panels);label=np.full((34,row.shape[1],3),255,dtype=np.uint8)
        cv2.putText(label,NAMES.get(data['policy'],data['policy']),(8,24),0,.7,INK,2,cv2.LINE_AA)
        rows+=[label,row]
    figure=np.vstack(rows);legend=np.full((40,figure.shape[1],3),PAPER,dtype=np.uint8)
    cv2.putText(legend,'blue: told to reach the cone | orange: the ball | white dot: start | ring: 15 m | filled dot: stopped by itself inside the ring | red square: did not',
                (8,26),0,.5,INK,1,cv2.LINE_AA)
    cv2.imwrite(str(output),np.vstack([figure,legend]),[cv2.IMWRITE_JPEG_QUALITY,88]);print(output,figure.shape)


def export_flown(runs,output,columns):
    """Each run (LABEL=DIRECTORY) drawn on the map its episodes were flown in, split by layout."""
    from scripts.plan_generalization import figure
    from src.visual_search.maps import load_map
    panels=[]
    for item in runs:
        label,directory=item.split('=',1);data,steps=load_run(directory)
        episodes=[e for e in data['episodes'] if not e.get('error')];groups={}
        for episode in episodes:groups.setdefault((episode.get('map','blocks'),episode.get('layout','pilot')),[]).append(episode)
        for (map_id,layout),items in sorted(groups.items()):
            paths=[(steps.get(e['id'],[]),e['target'],e['success']) for e in items];done=sum(e['success'] for e in items)
            panels.append(figure(load_map(map_id),layout,[],None,title=f'{label}: {done}/{len(items)}   ({map_id}, layout {layout})',paths=paths,
                                 legend='white dot: start   filled dot: stopped by itself within 15 m of the named object   square: did not   colour: the object named'))
    height=max(panel.shape[0] for panel in panels);width=max(panel.shape[1] for panel in panels);rows=[]
    padded=[cv2.copyMakeBorder(panel,0,height-panel.shape[0],0,width-panel.shape[1],cv2.BORDER_CONSTANT,value=(238,236,232)) for panel in panels]
    while len(padded)%columns:padded.append(np.full_like(padded[0],(238,236,232)))
    for index in range(0,len(padded),columns):rows.append(np.hstack(padded[index:index+columns]))
    sheet=np.vstack(rows);scale=min(1.,2400/sheet.shape[1])
    cv2.imwrite(str(output),cv2.resize(sheet,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,88]);print(output,sheet.shape)


# Landmark centres are read from the first run's results; the map picture itself carries no markers.
target_centre={}


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    gif=commands.add_parser('gif');gif.add_argument('--record',required=True);gif.add_argument('--episode',required=True);gif.add_argument('--output',required=True)
    gif.add_argument('--scale',type=float,default=.6);gif.add_argument('--seconds',type=float,default=.25);gif.add_argument('--hold',type=float,default=2.5)
    gif.add_argument('--model-name');gif.add_argument('--every',type=int,default=1,help='keep every n-th decision')
    paths=commands.add_parser('trajectories');paths.add_argument('runs',nargs='+');paths.add_argument('--map',required=True);paths.add_argument('--output',required=True)
    paths.add_argument('--cases',nargs='+',default=['A','B','C']);paths.add_argument('--targets',required=True,help='JSON file with landmark id -> [x, y]')
    flown=commands.add_parser('flown');flown.add_argument('runs',nargs='+',help='LABEL=DIRECTORY');flown.add_argument('--output',required=True)
    flown.add_argument('--columns',type=int,default=2)
    args=parser.parse_args()
    if args.command=='gif':export_gif(args.record,args.episode,args.output,args.scale,args.seconds,args.hold,args.model_name,args.every)
    elif args.command=='flown':export_flown(args.runs,args.output,args.columns)
    else:
        target_centre.update(json.loads(Path(args.targets).read_text()));export_trajectories(args.runs,args.map,args.output,args.cases)


if __name__=='__main__':main()
