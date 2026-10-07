"""Observer for a visual-search run. Reads the files the worker publishes; sends nothing to the model.

The target marker is drawn on a display copy only. The panel states whether the frames the policy
received are byte-identical to the raw camera frames (hashes published by the worker).
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

MODELS={'baseline':'Baseline AeroVLA','oft':'AeroVLA-OFT','teacher':'Teacher (privileged)'}
INK=(65,47,30);PURPLE=(150,75,115);GREEN=(40,140,35);RED=(45,75,200)


def marked(frame,pixel,seen):
    """Display copy with the ground-truth target marker; the frame passed in is left untouched."""
    copy=frame.copy()
    if pixel and all(math.isfinite(v) for v in pixel) and 0<=pixel[0]<frame.shape[1] and 0<=pixel[1]<frame.shape[0]:
        cv2.drawMarker(copy,(int(pixel[0]),int(pixel[1])),GREEN if seen else (30,145,235),cv2.MARKER_CROSS,15,2,cv2.LINE_AA)
    return copy


def input_matches(live,front,down):
    """True when the frames shown are byte-identical to what the camera gave and to what the policy received."""
    if front is None or down is None:return None
    return all(hashlib.sha256(frame.tobytes()).hexdigest()==live['step']['input_sha256'][name]==live['raw_sha256'][name]
               for name,frame in (('front',front),('down',down)))


def render(live,front,down,result=None):
    canvas=np.full((760,1180,3),(247,243,238),dtype=np.uint8)
    def text(value,x,y,scale=.55,color=INK,thickness=1):cv2.putText(canvas,str(value),(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)
    step=live['step'];episode=live['episode']
    text('MODE',20,34,.45);text(live['mode'],20,62,.8,PURPLE,2)
    text('MODEL',270,34,.45);text(live.get('model_name') or MODELS.get(live['model'],live['model']),270,62,.7,INK,2)
    # Where this episode sits among the generalisation tests; absent for the pilot's own cases.
    text('MAP',610,34,.45);text(live.get('map') or 'Blocks',610,62,.7,INK,2)
    text('TARGET',730,34,.45);text(live.get('target') or episode.get('target','--'),730,62,.6,INK,2)
    text('GENERALIZATION LEVEL',930,34,.45);text(live.get('level') or '--',930,62,.6,PURPLE,2)
    text('INSTRUCTION',20,98,.45);text(episode['instruction'],20,126,.8,INK,2)
    text('TARGET GROUND TRUTH: evaluation and display only; never sent to the model',20,156,.5,PURPLE)
    for name,frame,left in (('FRONT',front,20),('DOWN',down,420)):
        key=name.lower();seen=step[f'{key}_seen']
        text(f'{name} VISIBILITY: '+('target visible' if seen else 'not visible'),left,190,.52,GREEN if seen else INK)
        if frame is not None:canvas[202:586,left:left+384]=cv2.resize(marked(frame,step.get(f'{key}_pixel'),seen),(384,384),interpolation=cv2.INTER_NEAREST)
    same=input_matches(live,front,down)
    text('Model input = raw camera frame (no marker): '+{True:'VERIFIED',False:'MISMATCH',None:'--'}[same],20,612,.5,GREEN if same else RED)
    text('SEARCH STATE',830,190,.45);text('Target visible' if step['front_seen'] else 'Target not visible',830,218,.7,GREEN if step['front_seen'] else RED,2)
    text('DISTANCE',830,254,.45);text(f'{step["distance_m"]:.1f} m   bearing {step["bearing_deg"]:+.0f} deg',830,280,.62)
    text(f'success radius {live["success_radius_m"]:.0f} m   step {step["step"]}',830,304,.45)
    text('ACTION CHUNK  (forward m / down m / yaw deg)',830,344,.45)
    chunk=step.get('chunk') or [step['action']]
    for index,(forward,down_value,yaw) in enumerate(chunk):
        text(f't{index}  {forward:+.2f}  {down_value:+.2f}  {math.degrees(yaw):+6.1f}',830,374+index*28,.62,PURPLE if index==0 else INK,2 if index==0 else 1)
    text('EXECUTED',830,498,.45);text('t0' if 'chunk' in step else 'single action',830,524,.62,PURPLE,2)
    if 'raw_output' in step:text('model text: '+step['raw_output'][:28],830,552,.5)
    if 'inference_ms' in step:text(f'inference {step["inference_ms"]:.0f} ms',830,578,.5)
    text('FAILURE',830,618,.45);text(live.get('failure','NORMAL'),830,644,.7)
    if result and result.get('episode')==episode['id']:
        text(f'Episode ended: {result["reason"]}, {result["final_distance_m"]:.1f} m from the target',20,660,.6,GREEN if result['stopped'] and result['final_distance_m']<=live['success_radius_m'] else RED,2)
    text('Q / Esc closes this window; the run continues in the worker',20,740,.45)
    return canvas


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'outputs/visual_search/demo'))
    parser.add_argument('--snapshot');args=parser.parse_args();output=Path(args.output)
    window='VISUAL SEARCH | Front/Down + instruction only';live=None;front=down=None;result=None
    if not args.snapshot:cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,1180,760)
    while True:
        try:
            live=json.loads((output/'live.json').read_text())
            front=cv2.imread(str(output/'live_front.png'));down=cv2.imread(str(output/'live_down.png'))
            result=json.loads((output/'live_result.json').read_text()) if (output/'live_result.json').exists() else None
        except (OSError,ValueError):pass
        canvas=render(live,front,down,result) if live else np.full((760,1180,3),(247,243,238),dtype=np.uint8)
        if args.snapshot:
            cv2.imwrite(args.snapshot,canvas);return
        if live is None:cv2.putText(canvas,'Waiting for the model to load and the first observation...',(20,60),0,.7,INK,1,cv2.LINE_AA)
        cv2.imshow(window,canvas)
        if live:cv2.imwrite(str(output/'viewer.png'),canvas)
        key=cv2.waitKey(120)&0xff
        if key in (27,ord('q'),ord('Q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
        if (output/'done.flag').exists():time.sleep(.2)
    cv2.destroyAllWindows()


if __name__=='__main__':main()
