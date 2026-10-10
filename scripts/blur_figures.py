"""Pictures of the Gaussian blur characterization, from the frames the flights saved; no simulator, no model.

  blur_figures.py same-start START [START ...] --output FILE      what the policy was given at the first decision of a
                                                                  start, in each condition (the pose is the same in all four)
  blur_figures.py strip CONDITION START --output FILE [--count N] [--repeat]
                                                                  one flight at evenly spaced decisions: the Front and
                                                                  Down frames the policy was given, with what the log says

Every picture is drawn from frames a flight actually used; captions come from the flight's own log (distance and whether
the target was in view are the simulator's statements, not the policy's).
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.failures import blur_experiment as blur

INK=(40,40,40)


def given(folder,start,index):
    """(Front, Down) as the policy received them at one saved decision. A clean flight saved only the camera's frames, which are what it was given."""
    image=cv2.imread(str(folder/'frames'/start/f'{index:04d}.jpg'))
    if image is None:raise SystemExit(f'no saved frame {index} of {start} in {folder}')
    parts=[image[:,left:left+256] for left in range(0,image.shape[1],256)]
    return (parts[2],parts[3]) if len(parts)==4 else (parts[0],parts[1])


def label(canvas,text,x,y,scale=.45,colour=INK,thickness=1):
    cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,colour,thickness,cv2.LINE_AA)


def same_start(root,starts,output):
    conditions=blur.SEVERITY_ORDER;config=blur.load();left,top,gap=150,46,10
    canvas=np.full((top+len(conditions)*(256+gap)+10,left+len(starts)*(2*256+2*gap)+10,3),255,np.uint8)
    for column,start in enumerate(starts):
        x=left+column*(2*256+2*gap);label(canvas,start,x,18,.42);label(canvas,'Front',x,38,.42);label(canvas,'Down',x+256+gap,38,.42)
        for row,name in enumerate(conditions):
            front,down=given(root/name,start,0);y=top+row*(256+gap);canvas[y:y+256,x:x+256]=front;canvas[y:y+256,x+256+gap:x+512+gap]=down
    for row,name in enumerate(conditions):
        y=top+row*(256+gap);spec=config['conditions'][name];label(canvas,name.upper(),10,y+120,.6,INK,2)
        label(canvas,'camera frame' if spec is None else f'{spec["kernel"]} x {spec["kernel"]}, sigma {spec["sigma"]}',10,y+145,.4)
    cv2.imwrite(str(output),canvas,[cv2.IMWRITE_JPEG_QUALITY,88]);print(output,canvas.shape)


def strip(root,condition,start,output,count=8,repeat=False):
    folder=(root/'repeat' if repeat else root)/condition;rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()];rows=[row for row in rows if row['episode']==start]
    summary=next(episode for episode in json.loads((folder/'results.json').read_text())['episodes'] if episode['id']==start)
    saved=sorted(int(path.stem) for path in (folder/'frames'/start).glob('*.jpg'));picked=[saved[round(index*(len(saved)-1)/max(1,count-1))] for index in range(min(count,len(saved)))]
    picked=list(dict.fromkeys(picked));gap=6;top=44;canvas=np.full((top+2*256+gap+58,10+len(picked)*(256+gap),3),255,np.uint8)
    spec=blur.load()['conditions'][condition];what='camera frame' if spec is None else f'{spec["kernel"]} x {spec["kernel"]}, sigma {spec["sigma"]}'
    result='SUCCESS' if summary['success'] else 'FAILED'
    label(canvas,f'{condition.upper()} ({what})  |  {start}  |  "{summary["instruction"]}"  |  {result}: {summary["reason"]}, {summary["steps"]} decisions',10,18,.45,INK,1)
    label(canvas,'Front and Down as given to the policy; the line under each column is the simulator\'s own statement, never given to the policy',10,36,.38,(110,110,110))
    for column,index in enumerate(picked):
        front,down=given(folder,start,index);x=10+column*(256+gap);row=rows[index]
        canvas[top:top+256,x:x+256]=front;canvas[top+256+gap:top+512+gap,x:x+256]=down
        seen='in view' if row['front_seen'] or row['down_seen'] else 'not in view'
        label(canvas,f'decision {index}: {row["distance_m"]:.0f} m, {seen}',x,top+512+gap+18,.4)
        action=row['action'];label(canvas,f'fwd {action[0]:+.2f} down {action[1]:+.2f} yaw {np.degrees(action[2]):+.0f}',x,top+512+gap+36,.38,(110,110,110))
        label(canvas,f'height {row["height_m"]:.1f} m, {row["finalizer"]}',x,top+512+gap+52,.36,(110,110,110))
    cv2.imwrite(str(output),canvas,[cv2.IMWRITE_JPEG_QUALITY,86]);print(output,canvas.shape)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='outputs/failures/gaussian_blur');commands=parser.add_subparsers(dest='command',required=True)
    same=commands.add_parser('same-start');same.add_argument('starts',nargs='+');same.add_argument('--output',required=True)
    one=commands.add_parser('strip');one.add_argument('condition',choices=blur.SEVERITY_ORDER);one.add_argument('start');one.add_argument('--output',required=True)
    one.add_argument('--count',type=int,default=8);one.add_argument('--repeat',action='store_true')
    args=parser.parse_args();root=Path(args.root);root=root if root.is_absolute() else ROOT/root;Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    if args.command=='same-start':same_start(root,args.starts,Path(args.output))
    else:strip(root,args.condition,args.start,Path(args.output),args.count,args.repeat)


if __name__=='__main__':main()
