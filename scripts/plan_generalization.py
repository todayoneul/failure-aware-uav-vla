"""Plan generalisation episodes; no simulator, no model.

  test    write configs/generalization_test_spawns.json once: the held-out start states (G1 unseen
          start, G2 unseen object, G3 unseen map, G4 both, prompt variations, side probe) and the
          wedge of start directions each object keeps for testing. It is never rewritten.
  train   write the training and validation plans for a dataset, kept away from every held-out start
  targeted  the same for a recipe of extra kinds (config: generalization.<recipe>), with its own seeds
  g0      pick training starts to replay as the seen-start test
  check   confirm that no planned training or validation start is near a held-out start or in a wedge
  figure  draw the map with every planned start
  long    write configs/long_range_test_spawns.json once: fresh held-out starts in three bands of start
          distance (L1, L2, L3), in both scenes, for seen and unseen objects, away from every earlier start
  v3-test   write configs/gen_v3_test_spawns.json once: the fresh Gen-v3 sets (grounding among distractors,
            position and query swaps, approach or land, far starts, the canonical landing mission)
  v3-train  write the added Gen-v3 training and validation plans, kept away from every evaluation start
  v3-pilot  write the small plan of the canonical check (each pose once to land and once to approach)
  v3-check  confirm that a Gen-v3 plan stays out of the test scene, the test layouts and the test starts
"""
import argparse
import collections
import copy
import json
import math
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.visual_search.episodes import load_config
from src.visual_search.maps import load_map,MapGeometry,object_position
from src.visual_search.generalization import (make_start,plan_split,plan_targeted,rollout,sector_of,band_of,in_wedge,exclusion,
                                              instruction_for_id,BANDS,KINDS)
from src.visual_search import gen_v3
from src.aerovla_oft.spec import load_config as load_oft_config

TEST_FILE=ROOT/'configs/generalization_test_spawns.json'
# One held-out episode per row: kind, distance band, inside the held-out wedge, side of the target.
G1_ROWS=(('visible','far',True,None),('peripheral','medium',False,1),('peripheral','near',True,-1),('search','medium',True,-1),
         ('search','far',False,1),('search','near',False,'behind'),('altitude','far',False,None),('reacquire','medium',True,None))
G2_ROWS=(('visible','near',None),('visible','medium',None),('visible','far',None),('peripheral','medium',1),('peripheral','near',-1),
         ('peripheral','far',1),('search','near',-1),('search','medium',1),('search','far','behind'),('search','medium',-1),
         ('search','far',1),('search','near','behind'))
G3_ROWS=(('visible','medium',None),('peripheral','near',1),('search','near',-1),('search','medium',1),('search','far','behind'))
G4_ROWS=(('visible','medium',None),('visible','far',None),('peripheral','near',-1),('search','near',1),('search','medium',-1),('search','far','behind'))
LONG_FILE=ROOT/'configs/long_range_test_spawns.json'
# Earlier plans a long-range start must keep away from: both recorded datasets and the replayed seen starts.
EARLIER_PLANS=('outputs/examples/generalization/dataset_plan.json','outputs/examples/generalization/dataset_v2_added_plan.json',
               'configs/generalization_seen_starts.json')
# One long-range episode per row and band: map, layout, object. Half start with the object in view, half do not.
LONG_ROWS=(('blocks','a','blue_cone'),('blocks','b','orange_ball'),('blocks','a','red_cube'),('blocks','b','green_cylinder'),
           ('yard','a','blue_cone'),('yard','a','orange_ball'),('yard','a','red_cube'),('yard','a','green_cylinder'),
           ('blocks','a_new','yellow_pyramid'),('blocks','b_new','yellow_pyramid'),('yard','a_new','yellow_pyramid'),('yard','a_new','yellow_pyramid'))
V3_FILE=ROOT/'configs/gen_v3_test_spawns.json'
SIDE_BEARINGS=(('left',-90.),('right',90.),('behind_left',-135.),('behind_right',135.))
WEDGE_DEG=60.


def site_key(map_config,layout,target):
    x,y=object_position(map_config,layout,target);return f'{x:.1f},{y:.1f}'


def choose_wedges(config,oft,map_config):
    """For every place an object stands, one 60 degree range of start directions that training never uses.

    The wedge belongs to the place, not to the object: the cube and the cylinder swap places between
    layouts and the views from a wedge must be unseen whichever of them stands there."""
    by_site={};wedges={}
    for layout in map_config['train_layouts']:
        wedges[layout]={}
        for target in map_config['train_objects']:
            key=site_key(map_config,layout,target)
            if key not in by_site:
                usable=[]
                for start in range(0,360,30):
                    span=(float(start),start+WEDGE_DEG)
                    # A wedge is only worth holding out if the map lets a vehicle start in it at every distance.
                    if all(make_start(config,oft,map_config,layout,target,kind,band,0,'wedge',direction_deg=span)
                           for kind in ('visible','search') for band in BANDS):usable.append(span)
                if not usable:raise RuntimeError(f'No usable wedge for {target} in layout {layout}')
                by_site[key]=random.Random(f'wedge|{map_config["id"]}|{key}').choice(usable)
            wedges[layout][target]=list(by_site[key])
    return wedges


def held_out_start(config,oft,map_config,layout,target,row,seed,split,wedge=None,instruction_id='find'):
    kind,band,inside,side=row if len(row)==4 else (row[0],row[1],None,row[2])
    options=dict(side=side,instruction_id=instruction_id)
    if wedge is not None and inside:options['direction_deg']=tuple(wedge)
    if wedge is not None and inside is False:options['excluded']=lambda x,y,direction:in_wedge(direction,wedge)
    for attempt_kind in dict.fromkeys((kind,'search')):
        for attempt_band in dict.fromkeys((band,)+BANDS):
            episode=make_start(config,oft,map_config,layout,target,attempt_kind,attempt_band,seed,split,**options)
            if episode:
                episode['requested_kind']=kind;episode['in_wedge']=bool(inside) if wedge is not None else None
                return episode
    raise RuntimeError(f'No held-out start for {target} {row} in {map_config["id"]}/{layout}')


def turned(config,oft,map_config,episode,bearing,name,label):
    """The same place and height with the vehicle facing another way."""
    geometry=MapGeometry(map_config,episode['layout']);centre=geometry.objects[episode['target']]['centre']
    to_target=math.degrees(math.atan2(centre[1]-episode['start_xy'][1],centre[0]-episode['start_xy'][0]))
    copy_=copy.deepcopy(episode);copy_.update(id=name,target_bearing_deg=bearing,start_yaw_deg=round(to_target-bearing,2),sector=sector_of(bearing),side=label)
    plan=rollout(config,oft,geometry,copy_,map_config['altitude']['ceiling_m'])
    if plan['reason']!='teacher_stop' or plan['climbed_m']>0:return None
    copy_['plan']={'ticks':plan['ticks'],'clearance_m':round(plan['clearance_m'],2),'climbed_m':0.,'acquired_tick':plan['acquired_tick'],'states':plan['states']}
    return copy_


def build_test(config,oft):
    blocks=load_map('blocks');yard=load_map('yard');wedges={'blocks':choose_wedges(config,oft,blocks)};sets={name:[] for name in ('G1','G2','G3','G4','P','S')}
    targets=blocks['train_objects'];layouts=blocks['train_layouts']
    for index,row in enumerate(G1_ROWS):
        for number,target in enumerate(targets):
            seed=index*len(targets)+number;layout=layouts[(index+number)%len(layouts)]
            sets['G1'].append(held_out_start(config,oft,blocks,layout,target,row,seed,'g1',wedges['blocks'][layout][target],
                                             instruction_id=('find','approach')[seed%2]))
    for index,row in enumerate(G2_ROWS):
        layout=blocks['held_out_layouts'][index%len(blocks['held_out_layouts'])]
        sets['G2'].append(held_out_start(config,oft,blocks,layout,blocks['held_out_objects'][0],row,100+index,'g2',instruction_id=('find','approach')[index%2]))
    for index,row in enumerate(G3_ROWS):
        for number,target in enumerate(yard['train_objects']):
            seed=120+index*len(yard['train_objects'])+number
            sets['G3'].append(held_out_start(config,oft,yard,'a',target,row,seed,'g3',instruction_id=('find','approach')[seed%2]))
    for index,row in enumerate(G4_ROWS):
        sets['G4'].append(held_out_start(config,oft,yard,'a_new',yard['held_out_objects'][0],row,160+index,'g4',instruction_id=('find','approach')[index%2]))
    # Prompt variations: the same held-out starts, told in words the training data never used.
    for episode in sets['G1']:
        if episode['seed']//len(targets) in (0,4):
            for phrase in config['generalization']['instructions']['held_out']:
                copy_=copy.deepcopy(episode)
                copy_.update(id=f'p-{episode["seed"]:04d}-{episode["target"]}-{episode["kind"]}-{phrase}',split='p',instruction_id=phrase,
                             instruction=instruction_for_id(config,blocks,episode['target'],phrase),base=episode['id'])
                sets['P'].append(copy_)
    # Side probe: one place per object, the target put to the left, right, behind-left and behind-right by turning the vehicle.
    for number,target in enumerate(targets):
        layout=layouts[number%len(layouts)];seed=180+number*4;group=None
        for attempt in range(40):
            base=make_start(config,oft,blocks,layout,target,'search','medium',seed+1000*attempt,'s',side=-1,instruction_id='find',
                            excluded=lambda x,y,direction,w=wedges['blocks'][layout][target]:in_wedge(direction,w))
            if base is None:continue
            group=[turned(config,oft,blocks,base,bearing,f's-{seed+offset:04d}-{target}-{label}',label) for offset,(label,bearing) in enumerate(SIDE_BEARINGS)]
            if all(group):break
        if not group or not all(group):raise RuntimeError(f'No side-probe start for {target}')
        for offset,episode in enumerate(group):episode['seed']=seed+offset
        sets['S']+=group
    return {'description':'Held-out start states for AeroVLA-OFT generalisation tests. Written once, before any generalisation '
                          'training data was planned; training and validation starts are kept away from every start listed here '
                          '(scripts/plan_generalization.py check).',
            'levels':{'G1':'seen map, seen object, unseen start','G2':'seen map, unseen object, unseen start',
                      'G3':'unseen map, seen object, unseen start','G4':'unseen map, unseen object, unseen start',
                      'P':'G1 starts told in unseen words','S':'one unseen place per object, target to the left, right, behind-left, behind-right'},
            'exclusion_radius_m':config['generalization']['exclusion_radius_m'],'wedge_deg':WEDGE_DEG,'wedges':wedges,'sets':sets}


def earlier_starts(held_out):
    """Every start planned before the long-range set: (set name, id, map, xy)."""
    points=[(name,e['id'],e['map'],e['start_xy']) for name,episodes in held_out['sets'].items() for e in episodes]
    for path in EARLIER_PLANS:
        data=json.loads((ROOT/path).read_text(encoding='utf-8'))
        for name,episodes in (data['sets'] if 'sets' in data else {key:data[key] for key in ('train','val')}).items():
            points+=[(name,e['id'],e['map'],e['start_xy']) for e in episodes]
    return points


def nearest_earlier(points,map_id,xy,training):
    """Distance to the closest earlier start in the same map, among training/validation starts or among held-out ones."""
    distances=[math.hypot(xy[0]-p[0],xy[1]-p[1]) for name,_,where,p in points if where==map_id and (name in ('train','val','G0'))==training]
    return round(min(distances),1) if distances else None


def build_long(config,oft,held_out):
    settings=config['generalization']['long_range'];points=earlier_starts(held_out);sets={};seed=settings['seeds'][0];maps={}
    for number,(name,span) in enumerate(settings['ranges_m'].items()):
        sets[name]=[]
        for index,(map_id,layout,target) in enumerate(LONG_ROWS):
            map_config=maps.setdefault(map_id,load_map(map_id));unseen=target in map_config['held_out_objects']
            # In view for every other row, shifted by one from band to band so each object gets both over the set.
            kind=('visible','search')[(index+number)%2];side=(1,-1,'behind')[(index//2+number)%3];episode=None
            excluded=lambda x,y,direction,m=map_id:any(where==m and math.hypot(x-p[0],y-p[1])<settings['separation_m'] for _,_,where,p in points)
            # Where the map has no clear line of this length to an object, the next seen object takes the row.
            planned=target;order=[target] if unseen else [target]+[other for other in map_config['train_objects'] if other!=target]
            for target in order:
                for attempt in range(20):
                    episode=make_start(config,oft,map_config,layout,target,kind,'far',seed,name.lower(),side=side if kind=='search' else None,
                                       excluded=excluded,instruction_id=('find','approach')[index%2],
                                       spec={'base':kind,'distance_m':span},name=f'{name.lower()}-{seed:04d}-{target}-{kind}')
                    seed+=1
                    if episode:break
                if episode:break
            if episode is None:raise RuntimeError(f'No {name} start for {planned} ({kind}) in {map_id}/{layout}')
            if target!=planned:episode['planned_object']=planned
            wedge=held_out['wedges'].get(map_id,{}).get(layout,{}).get(target)
            episode.update(range=name,range_m=span,object_seen_in_training=not unseen,in_wedge=bool(wedge and in_wedge(episode['start_direction_deg'],wedge)),
                           nearest_training_start_m=nearest_earlier(points,map_id,episode['start_xy'],True),
                           nearest_held_out_start_m=nearest_earlier(points,map_id,episode['start_xy'],False))
            points.append((name,episode['id'],map_id,episode['start_xy']));sets[name].append(episode)
    return {'description':'Fresh held-out start states by start distance, planned after Gen-v2 was trained and evaluated. Seeds '
                          f'{settings["seeds"][0]}+ were never used before; every start is at least {settings["separation_m"]} m from every training, '
                          'validation and earlier held-out start of its map. No model was trained on, selected with or changed after these.',
            'levels':{name:f'start {span[0]:.0f}-{span[1]:.0f} m from the target' for name,span in settings['ranges_m'].items()},
            'separation_m':settings['separation_m'],'earlier_plans':list(EARLIER_PLANS)+['configs/generalization_test_spawns.json'],
            'training_start_distance_max_m':max(e['start_distance_m'] for path in EARLIER_PLANS[:2] for key in ('train','val')
                                                for e in json.loads((ROOT/path).read_text(encoding='utf-8'))[key]),'sets':sets}


def describe_v3(episodes):
    return {'episodes':len(episodes),'by_kind':histogram(episodes,lambda e:e['kind']),'by_task':histogram(episodes,lambda e:e.get('task','approach')),
            'by_map':histogram(episodes,lambda e:e['map']),'by_layout':histogram(episodes,lambda e:f'{e["map"]}/{e["layout"]}'),
            'by_target':histogram(episodes,lambda e:e['target']),'by_instruction':histogram(episodes,lambda e:e['instruction']),
            'by_start_distance':histogram(episodes,lambda e:'55 m and more' if e['start_distance_m']>=55 else '24-55 m' if e['start_distance_m']>=24 else 'under 24 m'),
            'start_in_view':sum(e['plan']['acquired_tick']==0 for e in episodes),
            'another_object_in_first_view':sum(any(o['distance_m']<=90 for o in e.get('others_in_view',[])) for e in episodes),
            'teacher_ticks':[min(e['plan']['ticks'] for e in episodes),max(e['plan']['ticks'] for e in episodes)]}


def check_v3(config,test,plans,held_out,long_range):
    """A Gen-v3 training plan may not touch the test scene, a test layout, a held-out object, a test sentence or a test start."""
    settings=config['gen_v3'];problems=[];points=gen_v3.test_points(test,held_out,long_range);radius=settings['separation_m'];maps={}
    unseen={phrase for task in settings['instructions'].values() for phrase in task['held_out']}
    for split,episodes in plans.items():
        low,high=settings['seeds'][split]
        for episode in episodes:
            scene=maps.setdefault(episode['map'],load_map(episode['map']))
            if episode['map'] not in settings['train_layouts'] or episode['layout'] not in settings['train_layouts'][episode['map']]:
                problems.append(f'{episode["id"]}: {episode["map"]}/{episode["layout"]} is not a training layout')
            if set(scene['layouts'][episode['layout']])&set(scene['held_out_objects']) or episode['target'] in scene['held_out_objects']:
                problems.append(f'{episode["id"]}: a held-out object stands in the layout')
            if episode['instruction_id'] in unseen:problems.append(f'{episode["id"]}: told in held-out words')
            if not low<=episode['seed']<=high:problems.append(f'{episode["id"]}: seed outside the {split} range')
            for where,(px,py) in points:
                if where==episode['map'] and math.hypot(episode['start_xy'][0]-px,episode['start_xy'][1]-py)<radius:
                    problems.append(f'{episode["id"]}: within {radius} m of an evaluation start');break
            wedge=gen_v3.site_wedge(held_out,scene,episode['layout'],episode['target'])
            if wedge and in_wedge(episode['start_direction_deg'],wedge):problems.append(f'{episode["id"]}: inside a held-out wedge')
    ids=[e['id'] for episodes in plans.values() for e in episodes]
    if len(ids)!=len(set(ids)):problems.append('duplicate episode ids')
    return problems


def histogram(episodes,key):return dict(sorted(collections.Counter(key(episode) for episode in episodes).items()))


def describe(config,episodes):
    return {'episodes':len(episodes),'by_target':histogram(episodes,lambda e:e['target']),'by_kind':histogram(episodes,lambda e:e['kind']),
            'by_band':histogram(episodes,lambda e:band_of(config,e['start_distance_m'])),'by_sector':histogram(episodes,lambda e:e['sector']),
            'by_layout':histogram(episodes,lambda e:e['layout']),'by_instruction':histogram(episodes,lambda e:e['instruction_id']),
            'by_strategy':histogram(episodes,lambda e:e['strategy']),
            'requested_kind_changed':sum(e.get('requested_kind',e['kind'])!=e['kind'] for e in episodes)}


def check(config,held_out,plans):
    """Every planned training or validation start must be clear of all held-out starts and wedges."""
    radius=held_out['exclusion_radius_m'];problems=[]
    points=[(e['id'],e['map'],e['start_xy']) for episodes in held_out['sets'].values() for e in episodes]
    test_seeds=range(config['generalization']['seeds']['test'][0],config['generalization']['seeds']['test'][1]+1)
    for split,episodes in plans.items():
        low,high=config['generalization']['seeds'][split]
        for episode in episodes:
            if not low<=episode['seed']<=high or episode['seed'] in test_seeds:problems.append(f'{episode["id"]}: seed outside the {split} range')
            for name,map_id,(px,py) in points:
                if map_id==episode['map'] and math.hypot(episode['start_xy'][0]-px,episode['start_xy'][1]-py)<radius:
                    problems.append(f'{episode["id"]}: {math.hypot(episode["start_xy"][0]-px,episode["start_xy"][1]-py):.1f} m from held-out {name}')
            wedge=held_out['wedges'].get(episode['map'],{}).get(episode['layout'],{}).get(episode['target'])
            if wedge and in_wedge(episode['start_direction_deg'],wedge):problems.append(f'{episode["id"]}: inside the held-out wedge of {episode["target"]}')
    ids=[e['id'] for episodes in plans.values() for e in episodes]
    if len(ids)!=len(set(ids)):problems.append('duplicate episode ids across splits')
    return problems


COLORS={'blue_cone':(200,150,20),'orange_ball':(20,120,240),'red_cube':(40,40,215),'green_cylinder':(40,150,30),'yellow_pyramid':(20,200,230),
        'blue_pad':(230,90,10),'red_pad':(60,20,170),'blue_cube':(170,70,40),'blue_cylinder':(210,140,70)}


def figure(map_config,layout,groups,output,wedges=None,title='',paths=(),legend=None):
    """Top-down picture of one map layout: obstacles by height, objects, one mark per start, and flown paths.

    `paths` holds (points, object, success) for flown episodes; returns the picture when `output` is None."""
    import cv2
    import numpy as np
    area=map_config['area'];scale=5.;margin=8.
    x0,x1=area['x'][0]-margin,area['x'][1]+margin;y0,y1=area['y'][0]-margin,area['y'][1]+margin
    # North (x) up, east (y) to the right, as in the simulator's map view.
    def pixel(x,y):return int(round((y-y0)*scale)),int(round((x1-x)*scale))
    canvas=np.full((int((x1-x0)*scale)+70,int((y1-y0)*scale),3),(238,236,232),dtype=np.uint8);geometry=MapGeometry(map_config,layout)
    colors=COLORS
    for box in geometry.boxes:
        shade=int(205-min(25.,box['top_m'])*5);color=colors.get(box['name'],(shade,shade,shade))
        cv2.rectangle(canvas,pixel(box['max'][0],box['min'][1]),pixel(box['min'][0],box['max'][1]),color,-1)
        cv2.rectangle(canvas,pixel(box['max'][0],box['min'][1]),pixel(box['min'][0],box['max'][1]),(110,110,110),1)
    for target,span in (wedges or {}).items():
        if target not in geometry.objects:continue
        centre=geometry.objects[target]['centre']
        for angle in span:
            end=(centre[0]+70*math.cos(math.radians(angle)),centre[1]+70*math.sin(math.radians(angle)))
            cv2.line(canvas,pixel(*centre),pixel(*end),colors.get(target,(90,90,90)),1,cv2.LINE_AA)
    for label,episodes,style in groups:
        for episode in episodes:
            if episode['layout']!=layout and not (episode['layout'].startswith(layout) or layout.startswith(episode['layout'])):continue
            point=pixel(*episode['start_xy']);color=colors.get(episode['target'],(60,60,60))
            if style=='dot':cv2.circle(canvas,point,3,color,-1,cv2.LINE_AA)
            else:
                cv2.drawMarker(canvas,point,(20,20,20),cv2.MARKER_TILTED_CROSS,13,3,cv2.LINE_AA);cv2.drawMarker(canvas,point,color,cv2.MARKER_TILTED_CROSS,11,1,cv2.LINE_AA)
    for points,target,success in paths:
        color=colors.get(target,(60,60,60));track=[pixel(point[0],point[1]) for point in points]
        for a,b in zip(track,track[1:]):cv2.line(canvas,a,b,color,2,cv2.LINE_AA)
        if track:
            cv2.circle(canvas,track[0],4,(255,255,255),-1,cv2.LINE_AA);cv2.circle(canvas,track[0],4,(40,40,40),1,cv2.LINE_AA)
            if success:cv2.circle(canvas,track[-1],6,color,-1,cv2.LINE_AA);cv2.circle(canvas,track[-1],6,(255,255,255),1,cv2.LINE_AA)
            else:cv2.rectangle(canvas,(track[-1][0]-6,track[-1][1]-6),(track[-1][0]+6,track[-1][1]+6),(20,20,20),2)
    base=canvas.shape[0]-52
    cv2.putText(canvas,title,(10,base),0,.6,(40,40,40),2,cv2.LINE_AA)
    cv2.putText(canvas,legend or 'dot: training start   cross: held-out start   colour: the object named   lines: wedge of start directions kept for testing',
                (10,base+26),0,.45,(40,40,40),1,cv2.LINE_AA)
    if output is None:return canvas
    cv2.imwrite(str(output),canvas,[cv2.IMWRITE_JPEG_QUALITY,90]);print(output,canvas.shape)


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('test')
    train=commands.add_parser('train');train.add_argument('--output',required=True);train.add_argument('--episodes',type=int,default=240)
    train.add_argument('--validation',type=int,default=28);train.add_argument('--map',default='blocks')
    extra=commands.add_parser('targeted');extra.add_argument('--output',required=True);extra.add_argument('--recipe',default='targeted_v2')
    extra.add_argument('--map',default='blocks')
    replay=commands.add_parser('g0');replay.add_argument('--plan',required=True);replay.add_argument('--output',required=True);replay.add_argument('--episodes',type=int,default=20)
    replay.add_argument('--recorded',help='dataset root; only episodes the teacher completed there are replayed')
    checker=commands.add_parser('check');checker.add_argument('--plan',required=True)
    draw=commands.add_parser('figure');draw.add_argument('--plan');draw.add_argument('--map',default='blocks');draw.add_argument('--layout',default='a')
    draw.add_argument('--sets',nargs='+',default=['G1','S']);draw.add_argument('--output',required=True)
    draw.add_argument('--test-file',help='another file of held-out sets to draw instead of the first one (e.g. configs/gen_v3_test_spawns.json)')
    commands.add_parser('long');commands.add_parser('v3-test')
    for name in ('v3-train','v3-pilot'):commands.add_parser(name).add_argument('--output',required=True)
    commands.add_parser('v3-check').add_argument('--plan',required=True)
    args=parser.parse_args();config=load_config();oft=load_oft_config()
    if args.command=='test':
        if TEST_FILE.exists():raise SystemExit(f'{TEST_FILE} exists; the held-out starts are fixed and are not regenerated')
        data=build_test(config,oft);TEST_FILE.write_text(json.dumps(data,indent=1)+'\n',encoding='utf-8',newline='\n')
        for name,episodes in data['sets'].items():print(name,json.dumps(describe(config,episodes)))
        return
    held_out=json.loads(TEST_FILE.read_text(encoding='utf-8'))
    if args.command=='long':
        if LONG_FILE.exists():raise SystemExit(f'{LONG_FILE} exists; the long-range starts are fixed and are not regenerated')
        data=build_long(config,oft,held_out);LONG_FILE.write_text(json.dumps(data,indent=1)+chr(10),encoding='utf-8',newline=chr(10))
        for name,episodes in data['sets'].items():
            print(name,json.dumps({'episodes':len(episodes),'by_map':histogram(episodes,lambda e:e['map']),'by_kind':histogram(episodes,lambda e:e['kind']),
                                   'unseen_object':sum(not e['object_seen_in_training'] for e in episodes),
                                   'distance':[min(e['start_distance_m'] for e in episodes),max(e['start_distance_m'] for e in episodes)],
                                   'nearest_training':min((e['nearest_training_start_m'] for e in episodes if e['nearest_training_start_m'] is not None),default=None),
                                   'nearest_held_out':min((e['nearest_held_out_start_m'] for e in episodes if e['nearest_held_out_start_m'] is not None),default=None),
                                   'teacher_ticks':[min(e['plan']['ticks'] for e in episodes),max(e['plan']['ticks'] for e in episodes)]}))
        return
    if args.command.startswith('v3'):
        if args.command=='v3-test':
            if V3_FILE.exists():raise SystemExit(f'{V3_FILE} exists; the Gen-v3 test starts are fixed and are not regenerated')
            data=gen_v3.build_test(config,oft);V3_FILE.write_text(json.dumps(data,indent=1)+chr(10),encoding='utf-8',newline=chr(10))
            for name,episodes in data['sets'].items():print(name,json.dumps(describe_v3(episodes)))
            return
        test=json.loads(V3_FILE.read_text(encoding='utf-8'));long_range=json.loads(LONG_FILE.read_text(encoding='utf-8'))
        if args.command=='v3-check':
            plan=json.loads(Path(args.plan).read_text());plans={split:plan[split] for split in ('train','val','pilot') if split in plan}
        elif args.command=='v3-pilot':plans={'pilot':gen_v3.plan_pilot(config,oft)}
        else:
            points=gen_v3.test_points(test,held_out,long_range)
            plans={split:gen_v3.plan_training(config,oft,split,points,held_out) for split in ('train','val')}
        problems=check_v3(config,test,plans,held_out,long_range)
        if problems:raise SystemExit(chr(10).join(problems))
        if args.command=='v3-check':
            print(f'OK: {sum(len(v) for v in plans.values())} planned starts are clear of the test scene, the test layouts, the held-out words and '
                  f'{len(gen_v3.test_points(test,held_out,long_range))} evaluation starts (radius {config["gen_v3"]["separation_m"]} m)');return
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        (output/'plan.json').write_text(json.dumps({'recipe':'gen_v3','test_file':V3_FILE.relative_to(ROOT).as_posix(),
                                                    'summary':{split:describe_v3(episodes) for split,episodes in plans.items()},**plans},indent=1))
        for split,episodes in plans.items():print(split,json.dumps(describe_v3(episodes)))
        return
    if args.command=='train':
        map_config=load_map(args.map);seeds=config['generalization']['seeds']
        plans={'train':plan_split(config,oft,map_config,'train',range(seeds['train'][0],seeds['train'][0]+args.episodes),held_out),
               'val':plan_split(config,oft,map_config,'val',range(seeds['val'][0],seeds['val'][0]+args.validation),held_out)}
        problems=check(config,held_out,plans)
        if problems:raise SystemExit('\n'.join(problems))
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        (output/'plan.json').write_text(json.dumps({'map':args.map,'held_out_file':str(TEST_FILE.relative_to(ROOT)).replace('\\','/'),
                                                    'summary':{split:describe(config,episodes) for split,episodes in plans.items()},**plans},indent=1))
        for split,episodes in plans.items():print(split,json.dumps(describe(config,episodes)))
    elif args.command=='targeted':
        map_config=load_map(args.map);recipe=config['generalization'][args.recipe]
        plans={split:plan_targeted(config,oft,map_config,split,recipe,held_out) for split in ('train','val')}
        problems=check(config,held_out,plans)
        if problems:raise SystemExit('\n'.join(problems))
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        (output/'plan.json').write_text(json.dumps({'map':args.map,'recipe':args.recipe,'held_out_file':str(TEST_FILE.relative_to(ROOT)).replace('\\','/'),
                                                    'summary':{split:describe(config,episodes) for split,episodes in plans.items()},**plans},indent=1))
        for split,episodes in plans.items():print(split,json.dumps(describe(config,episodes)))
    elif args.command=='g0':
        plan=json.loads(Path(args.plan).read_text());chosen=[];targets=sorted({e['target'] for e in plan['train']})
        recorded={path.stem for path in (Path(args.recorded)/'episodes').glob('*.json')
                  if json.loads(path.read_text())['summary'].get('reason')=='teacher_stop'} if args.recorded else None
        # Seen starts: training episodes themselves, one of every kind for every object.
        for kind in KINDS:
            for target in targets:
                pool=[e for e in plan['train'] if e['target']==target and e['kind']==kind and (recorded is None or e['id'] in recorded)]
                if pool and len(chosen)<args.episodes:
                    copy_=copy.deepcopy(pool[len(chosen)%len(pool)]);copy_.update(base=copy_['id'],id='g0-'+copy_['id'].split('-',1)[1],split='g0');chosen.append(copy_)
        Path(args.output).write_text(json.dumps({'description':'Seen starts: training episodes replayed','sets':{'G0':chosen}},indent=1))
        print('G0',json.dumps(describe(config,chosen)))
    elif args.command=='check':
        plan=json.loads(Path(args.plan).read_text());problems=check(config,held_out,{'train':plan['train'],'val':plan['val']})
        print('\n'.join(problems) if problems else f'OK: {len(plan["train"])} training and {len(plan["val"])} validation starts are clear of '
              f'{sum(len(v) for v in held_out["sets"].values())} held-out starts (radius {held_out["exclusion_radius_m"]} m) and of every wedge')
        if problems:raise SystemExit(1)
    else:
        map_config=load_map(args.map);groups=[]
        if args.plan:
            plan=json.loads(Path(args.plan).read_text());groups.append(('train',[e for e in plan['train']+plan['val'] if e['map']==args.map],'dot'))
        shown=json.loads(Path(args.test_file).read_text(encoding='utf-8')) if args.test_file else held_out
        groups.append(('held-out',[e for name in args.sets for e in shown['sets'][name] if e['map']==args.map],'cross'))
        figure(map_config,args.layout,groups,args.output,None if args.test_file else held_out['wedges'].get(args.map,{}).get(args.layout),
               f'{map_config["name"]}, layout {args.layout}',legend='dot: training start   cross: held-out start   colour: the object named' if args.test_file else None)


if __name__=='__main__':main()
