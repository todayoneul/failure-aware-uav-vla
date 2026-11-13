"""Write a map's obstacle list from the simulator's own bounding boxes; no flight, no model.

Input is the scene-geometry.json that the mission overview publishes (every object's world-axis box
as the simulator reported it). Output is the small file a map config points at: one footprint per
column of stacked blocks with its height above the ground, plus the boxes of the native objects that
can be named as targets.
"""
import argparse
import json
import math
from pathlib import Path


def audit(records,objects,ground_z):
    columns={};natives={}
    for record in records:
        try:
            box=record['bbox'];centre=[float(box['center'][k]) for k in 'xyz'];size=[float(box['size'][k]) for k in 'xyz']
        except (KeyError,TypeError,ValueError):continue
        # Helper actors report an empty box, the floor an unbounded one, and the vehicle is not scenery.
        if not all(math.isfinite(v) for v in centre+size) or min(size)<=0 or max(size)>500 or record['name'].startswith('Drone'):continue
        top=ground_z-(centre[2]-size[2]/2)
        if record['name'] in objects:
            natives[record['name']]={'center':[round(centre[0],2),round(centre[1],2)],'size_m':[round(size[0],2),round(size[1],2),round(top,2)]}
            continue
        low=(round(centre[0]-size[0]/2,1),round(centre[1]-size[1]/2,1));high=(round(centre[0]+size[0]/2,1),round(centre[1]+size[1]/2,1))
        column=columns.setdefault((low,high),{'name':record['name'],'min':list(low),'max':list(high),'top_m':0.,'parts':0})
        column['parts']+=1
        if top>column['top_m']:column.update(top_m=round(top,2),name=record['name'])
    missing=[name for name in objects if name not in natives]
    if missing:raise SystemExit(f'Objects absent from the scene: {missing}')
    return sorted(columns.values(),key=lambda b:(b['min'],b['max'])),natives


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scene-geometry',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--objects',nargs='*',default=[]);parser.add_argument('--ground-z',type=float,default=-1.,help='NED z of the ground surface')
    parser.add_argument('--source',default='Project AirSim Blocks 1.0.1')
    args=parser.parse_args();data=json.loads(Path(args.scene_geometry).read_text())
    boxes,natives=audit(data['objects'],args.objects,args.ground_z)
    Path(args.output).write_text(json.dumps({'source':f'{args.source}: world-axis bounding boxes reported by the simulator',
                                             'ground_z':args.ground_z,'boxes':boxes,'objects':natives},indent=1)+'\n')
    print(f'{len(boxes)} obstacle columns, tallest {max(b["top_m"] for b in boxes)} m; native objects: {natives}')
