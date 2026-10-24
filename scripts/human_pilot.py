"""Human pilot mode: a person flies the vehicle from the front and down pictures alone, and the flight is recorded as data.

  human_pilot.py --host 127.0.0.1 [--scene field] [--layout a] [--target blue_cube] [--task land] [--no-record]

The window shows the front and down views, the sentence, and the pilot's own commands; no map, no position, no height
and nothing of where the object is. The views come from two display cameras with the pose and the field of view of the
policy's cameras at a higher resolution (`--view-px`); what is recorded is the policy's own 256 x 256 pictures. With
`--unreal-view` the simulator's own window (the vehicle seen from outside) is shown beside it and the flight is recorded
with `view: unreal`. Keys and pad: src/human_pilot/devices.py. What is recorded and what is left out of the labels:
src/human_pilot/controls.py and src/human_pilot/session.py; docs/human_pilot_mode.md.

The vehicle is flown as a policy's flights are, with the same call (a velocity along the heading, a height to hold, a
heading to turn to) and the same limits; only the commands come twenty times a second, eased in, instead of once per
decision. Every 0.5 s the pictures are taken as the canonical loop takes them (scripts/visual_search.SearchEnv.observe)
and the commands of the following 0.5 s become that decision's label.

Takeover mode (`--takeover READY_FILE`, src/human_pilot/remote.py serving the checkpoint in WSL): the model flies each
flight first, as the canonical loop flies it, and the pilot takes over when it goes wrong (Space, or any flight key) and
gives it back (Space). Only the pilot's decisions are labels; the model's are recorded and marked `model`, the decision of
a takeover `handover`. The flights go to their own dataset folder.

--script FILE flies a fixed list of inputs instead of a person: a check of the whole loop with the simulator.
"""
import argparse
import asyncio
import copy
import json
import math
import random
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.human_pilot import controls,session
from src.human_pilot.controls import Axes,Inputs,Smoother,TickLabel

WINDOW='HUMAN PILOT | front + down cameras only'
STATES={'human':'FLYING','model':'MODEL FLYING','taken':'YOU ARE FLYING','stop':'DONE - holding','landed':'TOUCHDOWN - holding'}
UNREAL='Blocks Environment'     # the simulator's own window
VIEWS={'front':'FrontView','down':'DownView'}


class Stream:
    """The newest picture of each camera, as the simulator publishes it (thirty a second), for the display only."""
    def __init__(self):
        self.lock=threading.Lock();self.frames={};self.stamps={};self.topics=[];self.counts={}

    def attach(self,client,drone):
        from projectairsim.utils import unpack_image
        if self.topics:
            try:client.unsubscribe(self.topics)
            except Exception:pass
        self.topics=[];self.counts={'front':0,'down':0}
        for name,sensor in VIEWS.items():
            topic=drone.sensors[sensor]['scene_camera'];self.topics.append(topic)
            def keep(_,message,name=name):
                try:image=unpack_image(message)
                except Exception:return
                with self.lock:self.frames[name]=image;self.stamps[name]=time.monotonic();self.counts[name]=self.counts.get(name,0)+1
            client.subscribe(topic,keep)

    def latest(self,name,fresh_s=.6):
        with self.lock:
            if name in self.frames and time.monotonic()-self.stamps[name]<=fresh_s:return self.frames[name]
        return None


def display_cameras(prepare,size,chase=None):
    """The canonical scene configuration with two more cameras for the display: each a copy of the policy's camera (the
    same place on the vehicle, the same direction and field of view), at `size` x `size` and published as a stream. The
    policy's own cameras are left as they are; the recorded pictures are still requested from them as the canonical loop
    requests them."""
    def prepared(root,output):
        directory=prepare(root,output);path=directory/'robot_quadrotor_fastphysics.jsonc';robot=json.loads(path.read_text())
        sensors={sensor['id']:sensor for sensor in robot['sensors']}
        for source,name in (('FrontCamera',VIEWS['front']),('DownCamera',VIEWS['down'])):
            view=copy.deepcopy(sensors[source]);rgb=next(setting for setting in view['capture-settings'] if setting['image-type']==0)
            view.update({'id':name,'capture-settings':[dict(rgb,width=size,height=size,**{'streaming-enabled':True})]})
            view.pop('noise-settings',None);robot['sensors'].append(view)
        # With the simulator's window shown, its view (the camera behind the vehicle) at `chase` (width, height).
        if chase and 'Chase' in sensors:sensors['Chase']['capture-settings'][0].update(width=chase[0],height=chase[1])
        path.write_text(json.dumps(robot,indent=2));return directory
    return prepared


def place_simulator(width=1280,height=720):
    """Puts the simulator's own window at the top left with a `width` x `height` inside. Windows only; a no-op elsewhere."""
    if sys.platform!='win32':return False
    import ctypes
    from ctypes import wintypes
    user32=ctypes.windll.user32;found=[]
    def visit(handle,_):
        buffer=ctypes.create_unicode_buffer(256);user32.GetWindowTextW(handle,buffer,256)
        if buffer.value.strip()==UNREAL and user32.IsWindowVisible(handle):found.append(handle)
        return True
    user32.EnumWindows(ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)(visit),0)
    if not found:return False
    outer=wintypes.RECT();inner=wintypes.RECT();user32.GetWindowRect(found[0],ctypes.byref(outer));user32.GetClientRect(found[0],ctypes.byref(inner))
    extra_w=(outer.right-outer.left)-(inner.right-inner.left);extra_h=(outer.bottom-outer.top)-(inner.bottom-inner.top)
    return bool(user32.MoveWindow(found[0],0,0,width+extra_w,height+extra_h,True))


def environment(canonical,view_px,chase=None):
    from src.landing.env import SurfaceContacts

    class PilotEnv(SurfaceContacts,canonical.SearchEnv):
        stream=None

        async def reset(self,episode):
            original=canonical.prepare_mission_config;canonical.prepare_mission_config=display_cameras(original,view_px,chase)
            try:await super().reset(episode)
            finally:canonical.prepare_mission_config=original
            named=self.surfaces[self.target];self.surface=named.top_m if named.landable else 0.
            self.stream.attach(self.client,self.drone)

    return PilotEnv


class Script:
    """Inputs read from a file instead of a person: [[seconds, {"forward": true, ...}], ...]. A segment may press a key once
    when it begins (`"press": "enter"`). In the menu, before each flight, it presses Enter; after the script, nothing."""
    def __init__(self,path):
        self.segments=json.loads(Path(path).read_text(encoding='utf-8'));self.started=None;self.done=set()

    def read(self):
        if self.started is None:return Inputs(),{'enter'},False
        elapsed=time.monotonic()-self.started;clock=0.
        for index,(seconds,keys) in enumerate(self.segments):
            if elapsed<clock+seconds:
                pressed=set()
                if keys.get('press') and index not in self.done:self.done.add(index);pressed.add(keys['press'])
                return Inputs(**{key:value for key,value in keys.items() if key!='press'}),pressed,False
            clock+=seconds
        return Inputs(),set(),False

    def flying(self):self.started=time.monotonic()

    def menu(self):self.started=None;self.done=set()


def bar(canvas,x,y,width,value,top,label):
    """A centred bar for a signed rate."""
    cv2.putText(canvas,label,(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1,cv2.LINE_AA)
    cv2.rectangle(canvas,(x,y),(x+width,y+14),(200,200,200),1)
    middle=x+width//2;end=middle+int(width/2*max(-1.,min(1.,value/top if top else 0.)))
    cv2.rectangle(canvas,(min(middle,end),y+2),(max(middle,end),y+12),(60,120,200),-1)


def render(front,down,lines,rates=None,limits=None,size=560,footer=''):
    """The pilot's window: the sentence and state on top, the two views (`size` pixels square), the pilot's own commands.
    Nothing else."""
    # The views start below the text: at 100 px for up to three lines, lower for more (the menu).
    top=max(100,28*len(lines)+16);width=max(2*size+60,780);height=top+size+90;canvas=np.full((height,width,3),245,np.uint8)
    for row,(text,colour,big) in enumerate(lines):
        cv2.putText(canvas,text,(20,32+row*28),cv2.FONT_HERSHEY_SIMPLEX,.8 if big else .55,colour,2 if big else 1,cv2.LINE_AA)
    for column,(name,frame) in enumerate((('FRONT',front),('DOWN',down))):
        x=20+column*(size+20);y=top
        if frame is None:cv2.rectangle(canvas,(x,y),(x+size,y+size),(30,30,30),-1)
        else:canvas[y:y+size,x:x+size]=cv2.resize(frame,(size,size),interpolation=cv2.INTER_AREA if frame.shape[0]>size else cv2.INTER_LINEAR)
        cv2.putText(canvas,name,(x+8,y+24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
    if rates is not None and limits is not None:
        y=top+size+25
        width_bar=max(160,(width-80)//3)
        bar(canvas,20,y,width_bar,rates.forward,limits.forward,f'forward {rates.forward:+.2f} m/s')
        bar(canvas,40+width_bar,y,width_bar,-rates.down,limits.down,f'up {0.-rates.down:+.2f} m/s')
        bar(canvas,60+2*width_bar,y,width_bar,rates.turn,limits.turn,f'turn {math.degrees(rates.turn):+.1f} deg/s')
    if footer:cv2.putText(canvas,footer,(20,height-14),cv2.FONT_HERSHEY_SIMPLEX,.45,(90,90,90),1,cv2.LINE_AA)
    return canvas


KEYS_FLYING='W/S forward/back | A D turn | Up/Down climb/descend | 1-4 speed | Enter done | Backspace discard | Esc quit'
KEYS_MODEL='Space or any flight key: take over | 1-4 your speed | Enter: end (after the model stops) | Backspace discard | Esc quit'
KEYS_TAKEN='W/S | A D | Up/Down | 1-4 speed | Space: give back to the model | Enter done | Backspace discard | Esc quit'
KEYS_MENU='N/P object | T land/approach | L layout | M scene | D distance | R random | Enter fly | Esc quit'


def show(canvas):
    cv2.imshow(WINDOW,canvas);return cv2.waitKey(1)


def make_step(env,observation,index,instruction,mode,recorder,oft):
    """One decision as the canonical loop writes it (scripts/visual_search.run_episode), with the pictures saved. Returns the
    step and what a policy is given at it (the canonical loop's own `policy_inputs`)."""
    from src.aerovla_oft.spec import proprio_vector
    from src.visual_search.episodes import policy_inputs
    private=observation['privileged'];state=observation['state']
    proprio=proprio_vector(state,env.ground_z,oft,observation['yaw_rate'])
    inputs=policy_inputs(observation['frames']['front'],observation['frames']['down'],instruction,proprio if oft['proprio']['enabled'] else None)
    step={'step':index,'epoch':time.time(),'position':state['position'],'yaw_rad':observation['yaw'],'height_m':private['height_m'],'bearing_deg':private['bearing_deg'],
          'distance_m':private['distance_m'],'front_seen':private['front']['seen'],'front_in_fov':private['front']['in_fov'],'down_seen':private['down']['seen'],
          'front_pixel':private['front']['pixel'],'down_pixel':private['down']['pixel'],'ahead_m':private['ahead_m'],'over':private['over'],
          'landed':env.resting(state) is not None,'on_surface':env.on_surface(state),'velocity':state['velocity'],'target_size':private['size'],'finalizer':'OFF',
          'input_sha256':{name:session.digest(inputs[name]) for name in ('front','down')},'proprio':proprio,'teacher_state':mode,'perturbed':False,
          'sim_stamp':observation['stamp']}
    if recorder:step['img_name']=recorder.frames(index,inputs['front'],inputs['down'])
    return step,inputs


async def fly(env,episode,devices,settings,oft,config,recorder,size,tier_index=2,snapshot=None,policy=None):
    """One flight. Returns (reason, stopped, steps, touchdown, hit, switches) or None when the pilot discarded it or quit
    (then the second value says which).

    With `policy` (src/human_pilot/remote.RemotePolicy) the model flies first, as the canonical loop flies it: at each
    decision the pictures, the model's answer, and the canonical executor's command (SearchEnv.tick). The pilot takes over
    at any moment (Space, or any flight key) and gives control back with Space at the next decision. After the model's own
    stop (four in a row) the vehicle holds and waits: the pilot takes over, or ends the flight there (Enter)."""
    from src.aerovla_oft.spec import is_stop
    boost=settings['boost'];tiers=controls.tiers_of(oft,Axes(boost['forward_mps'],boost['vertical_mps'],math.radians(boost['turn_degps'])))
    tick=oft['tick_s'];dt=1./settings['control_hz'];limits=tiers[-1]
    acceleration=settings['acceleration'];smoother=Smoother(Axes(acceleration['forward_mps2'],acceleration['down_mps2'],acceleration['turn_radps2']))
    labeler=TickLabel(oft);steps=[];index=0;terminal=None;terminal_left=settings['terminal_ticks'];reason='max_steps';stopped=False
    soft=env.landing['touchdown']['max_vertical_speed_mps'];low=env.ground_z-env.limits['maximum_clearance_m'];high=env.ground_z-env.limits['minimum_clearance_m']
    # Who flies now; who flew during the open decision; the model's decision under way and whether its action is flown;
    # its stops in a row; whether it stopped and the vehicle waits; whether the pilot asked to give control back.
    pilot='model' if policy else 'human';flown={pilot};thinking=None;applied=True;stop_count=0;held=False;giving_back=False
    switches=[];model_rates=Axes()

    def open_decision():
        nonlocal opened,observation,pending,thinking,applied
        opened=time.monotonic();observation=env.observe()
        pending,given=make_step(env,observation,index,episode['instruction'],terminal or pilot,recorder,oft)
        if pilot=='model' and terminal is None:thinking=asyncio.ensure_future(asyncio.to_thread(policy.infer,given));applied=False

    def switch(to):
        nonlocal pilot,flown
        pilot=to;flown.add(to);switches.append({'decision':index,'to':to,'epoch':time.time()})

    opened=observation=pending=None;open_decision();heading=observation['yaw'];rates=Axes();saved=False
    if hasattr(devices,'flying'):devices.flying()
    last=time.monotonic();touchdown=hit=None
    while True:
        begun=time.monotonic();elapsed=max(1e-3,begun-last);last=begun
        inputs,pressed,_=devices.read()
        if 'escape' in pressed:return None,'quit'
        if 'backspace' in pressed or 'pad_b' in pressed:return None,'discard'
        for key,number in (('1',0),('2',1),('3',2),('4',3)):
            if key in pressed:tier_index=number
        if 'pad_lb' in pressed:tier_index=max(0,tier_index-1)
        if 'pad_rb' in pressed:tier_index=min(len(tiers)-1,tier_index+1)
        touchdown,hit=env.contacts()
        if hit:reason='collision';break
        if terminal is None and episode['task']=='land' and touchdown:
            terminal='landed';smoother.stop()
            if held:held=False;labeler.close();open_decision()
        if policy and terminal is None:
            asked=bool({'space','pad_y'}&pressed)
            if pilot=='model' and (asked or controls.active(inputs)):
                # The pilot's commands start from what the model was commanding (its decision's rates and heading).
                switch('human');smoother.now=model_rates
                if held:held=False;flown={'human'};labeler.close();open_decision()
            elif pilot=='human' and asked:giving_back=not giving_back
        if terminal is None and ('enter' in pressed or 'pad_a' in pressed):
            if pilot=='model':
                if held:reason='model_stop';stopped=True;break
            elif episode['task']=='approach':terminal='stop';smoother.stop()
            else:reason='gave_up';break
        if pilot=='human' or terminal or held:
            target=Axes() if terminal or held else controls.wanted(inputs,tiers[tier_index],oft)
            rates=smoother.stop() if terminal or held else smoother.step(target,elapsed)
            before=env.setpoint_z;heading+=rates.turn*elapsed;env.setpoint_z=max(low,min(high,env.setpoint_z+rates.down*elapsed))
            labeler.add(Axes(rates.forward,(env.setpoint_z-before)/elapsed,rates.turn),elapsed)
            await env.drone.move_by_velocity_z_async(rates.forward*math.cos(heading),rates.forward*math.sin(heading),env.setpoint_z,duration=settings['command_duration_s'],
                                                     yaw_is_rate=False,yaw=heading)
        elif not applied and thinking.done():
            # The model's decision, flown as the canonical loop flies it (no finalizer: the canonical evaluation's setting).
            answer=thinking.result();action=answer['chunk'][0]
            stop_count=stop_count+1 if is_stop(action,oft) else 0
            pending.update(chunk=answer['chunk'],normalised=answer['normalised_chunk'][0],prompt=answer['prompt'],inference_ms=answer['inference_ms'],
                           action=list(action),executed=list(action),teacher_action=list(action))
            await env.tick(action,observation['yaw']);heading=observation['yaw']+action[2];applied=True
            model_rates=rates=Axes(action[0]/tick,action[1]/tick,action[2]/tick)
        if pending is not None and time.monotonic()-opened>=tick and (pilot=='human' or terminal or applied):
            label=labeler.close();mark=controls.pilot_mark(flown)
            if 'action' not in pending:pending.update(teacher_action=label,action=label,executed=label)
            if 'human' in flown:pending['human_action']=label
            pending['pilot']=mark or 'human'
            if mark is None:
                pending.update(teacher_action=label,action=label,executed=label)
                mark=controls.mark(label,oft,terminal=pending['teacher_state'] in ('stop','landed'),reason=labeler.reason)
            if mark:pending['excluded']=mark
            steps.append(pending)
            if pending['teacher_state'] in ('stop','landed'):
                terminal_left-=1
                if terminal_left<=0:reason='human_stop' if pilot=='human' else 'model_stop';stopped=True;break
            index+=1
            if index>=episode.get('max_ticks',settings['max_ticks']):reason='max_steps';break
            if pilot=='model' and terminal is None and stop_count>=config['stop_ticks']:
                # The model's own stop: the canonical loop would end the flight here. The vehicle holds; the pilot decides.
                held=True;pending=None
            else:
                if giving_back and terminal is None:giving_back=False;switch('model');stop_count=0
                flown={pilot};open_decision()
        if held:state='MODEL STOPPED'
        elif terminal:state=STATES[terminal]
        else:state=(STATES['model'] if pilot=='model' else STATES['taken']) if policy else STATES['human']
        tier=controls.TIER_NAMES[tier_index]
        # Backward, or faster than a policy can fly: flown, recorded, never a label; the pilot is told so.
        note=' | BEYOND THE MODEL: not used for training' if pilot=='human' and controls.beyond(rates,oft) else ''
        colour=(150,40,150) if pilot=='model' and not terminal else (0,90,200) if note else (0,0,170) if recorder else (90,90,90)
        lines=[(episode['instruction'],(20,20,20),True),(f'{state} | speed {tier_index+1} {tier} | {"REC" if recorder else "not recorded"}{note}',colour,False)]
        if held:lines.append(('THE MODEL STOPPED IN THE AIR: take over (Space or a flight key) or end the flight here (Enter)',(0,0,200),False))
        elif giving_back:lines.append(('Control goes back to the model at the next decision (Space: cancel)',(150,40,150),False))
        # A landing counts only when the touchdown is soft; speed 3 descends faster than that.
        elif pilot=='human' and episode['task']=='land' and not terminal and rates.down>soft:
            lines.append((f'DESCENDING TOO FAST TO TOUCH DOWN: a landing needs {soft:.2f} m/s or less (speed 1 or 2)',(0,0,200),False))
        footer=KEYS_FLYING if not policy else KEYS_MODEL if pilot=='model' else KEYS_TAKEN
        # The stream when it runs; else the pictures of the last decision (the stream is for the display only).
        front=env.stream.latest('front');down=env.stream.latest('down')
        canvas=render(front if front is not None else observation['frames']['front'],down if down is not None else observation['frames']['down'],lines,rates,limits,size,footer)
        show(canvas)
        if snapshot and not saved and index>=12:cv2.imwrite(str(snapshot),canvas);saved=True
        await asyncio.sleep(max(0.,dt-(time.monotonic()-begun)))
    if thinking is not None and not thinking.done():
        # The model's answer to a decision the pilot took over is not flown; the connection is free again once it comes.
        try:await thinking
        except Exception:pass
    return (reason,stopped,steps,touchdown,hit,switches),None


async def pumping(period=.1):
    while True:
        cv2.waitKey(1);await asyncio.sleep(period)


async def main(args):
    import scripts.visual_search as canonical
    from projectairsim import ProjectAirSimClient
    from src.aerovla_oft.spec import load_config as load_oft
    from src.landing import evaluator
    from src.visual_search.episodes import load_config
    from src.visual_search.finalizer import LandingFinalizer
    from src.visual_search.maps import load_map
    settings=session.load(args.config);config=load_config();oft=load_oft();size=settings['display_px_unreal' if args.unreal_view else 'display_px']
    scenes=settings['scenes'];layouts=session.recording_layouts(settings);bands=list(settings['distance_bands'])
    scene=args.scene or scenes[0];layout=args.layout or layouts[0];band=args.band or settings['default_band'];drawn=bool(args.random)
    if scene not in scenes or layout not in layouts:raise SystemExit(f'scene and layout must be recording ones: {scenes}, {layouts}')
    record_root=None if args.no_record else Path(args.record or ROOT/(settings['takeover']['dataset'] if args.takeover else settings['dataset']))
    policy=None;model=None
    if args.takeover:
        from src.human_pilot.remote import RemotePolicy
        policy=RemotePolicy(args.takeover);check=Path(args.takeover).parent/'checkpoint-check.json'
        model=json.loads(check.read_text(encoding='utf-8')) if check.exists() else {}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    from src.human_pilot.devices import Devices
    devices=Script(args.script) if args.script else Devices(WINDOW,also=(UNREAL,) if args.unreal_view else ())
    # Each scene with the variants of its recording layouts that have the plate on the ball (built in memory).
    maps={name:session.with_caps(load_map(name),settings) for name in scenes};avoid={name:session.check_starts(name) for name in scenes}
    objects=session.targets_of(maps[scene],layout);chosen=next((index for index,item in enumerate(objects) if item['name']==args.target),0)
    task=args.task or ('land' if objects[chosen]['landable'] else 'approach');seed=args.seed if args.seed is not None else int(time.time())%100000
    client=ProjectAirSimClient(address=args.host,port_topics=args.ports[0],port_services=args.ports[1]);client.connect();client.socket_services.recv_timeout=60000
    env=environment(canonical,args.view_px or settings['view_px'],(1280,720) if args.unreal_view else None)(client,config,oft,output);env.stream=Stream();flown=0;message=''
    # The environment reads a map by its name through this cache: it is given the maps with the plate layouts.
    env.maps.update(maps)
    cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
    # Beside the simulator's window (1280 x 720 at the top left) when it is shown.
    if args.unreal_view:cv2.moveWindow(WINDOW,1300,0)
    try:
        while True:
            item=objects[chosen]
            if task=='land' and not item['landable']:task='approach'
            lines=[('TAKEOVER - the model flies, you take over' if policy else 'HUMAN PILOT - choose what to fly',(20,20,20),True),
                   (f'scene {scene} | layout {layout} | {item["noun"]} | {task.upper()}' if not drawn else f'scene {scene} | RANDOM: layout, object and LAND / APPROACH drawn for each flight',(20,20,20),False),
                   (f'start: {band.upper()}, a random place {settings["distance_bands"][band][0]:.0f}-{settings["distance_bands"][band][1]:.0f} m from the object (D: change)',(20,20,20),False),
                   (message,(0,110,0) if message.startswith('KEPT') else (0,0,170),False)]
            status=policy.status() if policy else 'ready'
            if policy:
                name=f'{Path(model.get("checkpoint","")).name} ({model.get("name")}, frozen baseline {(model.get("frozen") or {}).get("state")})'
                lines.append((f'MODEL {name}: ready' if status=='ready' else f'MODEL {name}: loading in WSL (about a minute) ...' if status=='loading'
                              else 'MODEL FAILED TO LOAD: see outputs/human_pilot/takeover/policy-server-errors.log',(150,40,150) if status in ('ready','loading') else (0,0,200),False))
            show(render(None,None,lines,size=size,footer=KEYS_MENU))
            inputs,pressed,_=devices.read()
            if 'escape' in pressed:break
            if 'n' in pressed:chosen=(chosen+1)%len(objects)
            if 'p' in pressed:chosen=(chosen-1)%len(objects)
            if 't' in pressed:task='approach' if task=='land' else ('land' if item['landable'] else 'approach')
            if 'd' in pressed:band=bands[(bands.index(band)+1)%len(bands)]
            if 'r' in pressed:drawn=not drawn
            if 'l' in pressed or 'm' in pressed:
                if 'l' in pressed:layout=layouts[(layouts.index(layout)+1)%len(layouts)]
                if 'm' in pressed:scene=scenes[(scenes.index(scene)+1)%len(scenes)]
                objects=session.targets_of(maps[scene],layout);chosen=min(chosen,len(objects)-1)
            if not ({'enter','pad_start'}&pressed):
                await asyncio.sleep(.05);continue
            if args.episodes and flown>=args.episodes:break
            if status!='ready':
                await asyncio.sleep(.05);continue
            if drawn:
                # The layout, the object and the task of this flight, from the flight's own seed.
                rng=random.Random(f'human-pilot|{scene}|{seed}');layout=rng.choice(layouts);objects=session.targets_of(maps[scene],layout)
                chosen=rng.randrange(len(objects));item=objects[chosen]
                task='land' if item['landable'] and rng.random()<settings['random_land_share'] else 'approach'
            episode=session.draw(settings,config,maps[scene],layout,item['name'],task,seed,avoid[scene],band=band);seed+=1
            episode['random_target']=drawn
            if policy:episode.update(id='tk'+episode['id'][2:],kind='takeover',pilot='takeover')
            show(render(None,None,[(episode['instruction'],(20,20,20),True),('PREPARING: loading the scene and flying to the start ...',(0,0,170),False)],size=size))
            recorder=session.Recorder(record_root,episode) if record_root else None
            # The window is kept answering while the scene loads (Windows marks it "not responding" otherwise).
            pump=asyncio.ensure_future(pumping())
            try:await env.reset(episode)
            finally:pump.cancel()
            clock=time.time();stamp=env.drone.get_ground_truth_kinematics()['time_stamp']
            if args.unreal_view:place_simulator()
            try:result,why=await fly(env,episode,devices,settings,oft,config,recorder,size,tier_index=settings['default_tier']-1,snapshot=args.snapshot,policy=policy)
            except Exception:
                traceback.print_exc();result,why=None,'error'
            if hasattr(devices,'menu'):devices.menu()
            if result is None:
                if recorder:recorder.discard()
                await env.close();message=f'{episode["id"]}: {why}, nothing kept'
                if why=='quit':break
                continue
            reason,stopped,steps,touchdown,hit,switches=result;flown+=1
            if not steps:
                await env.close();message=f'{episode["id"]}: {reason} before the first decision, nothing kept'
                if recorder:recorder.discard()
                continue
            finalizer=LandingFinalizer(env.landing['finalizer'],episode['instruction'],enabled=False)
            summary=canonical.summarise(episode,steps,config,reason,stopped,env.others(steps[-1]['position']),env.landing,touchdown,hit,finalizer.report())
            reading=evaluator.read(summary,env.surfaces,env.landing,steps)
            wall=steps[-1]['epoch']-clock;factor=((steps[-1]['sim_stamp']-stamp)/1e9/wall) if wall>0 else None
            good=session.kept(reading,summary,factor,settings);rows=session.label_rows(steps)
            # A takeover flight is data only for what the pilot flew: with no decision of the pilot's a label, there is none.
            if policy:good=good and rows['labels']>0
            target_surface=reading.get('target_surface') or {}
            landing=({'precision_m':round(session.precision_m(target_surface),3),'error_m':reading.get('horizontal_error_m'),'precise':session.precise(reading)}
                     if task=='land' and target_surface.get('landable') else None)
            extra={'takeover':{'checkpoint':model.get('checkpoint'),'model':model.get('name'),'frozen':(model.get('frozen') or {}).get('state'),
                               'loaded':policy.info.get('loaded'),'switches':switches,'takeovers':sum(item['to']=='human' for item in switches)}} if policy else {}
            if recorder:
                recorder.finish({**summary,'surface_landing':reading,'real_time_factor':factor,'pilot':'takeover' if policy else 'human','controls':'script' if args.script else 'keyboard/pad','labels':rows,
                                 'view':'unreal' if args.unreal_view else 'cameras','view_px':args.view_px or settings['view_px'],'distance_band':band,
                                 'random_target':drawn,'precision':landing,**extra},
                                steps,good,settings['keep_rejected_frames'])
            await env.close()
            line={'id':episode['id'],'scene':scene,'layout':layout,'target':item['name'],'task':task,'reason':reason,'success':bool(reading.get('success')),
                  'landed_on':reading.get('landed_on'),'collision':reading.get('collision'),'decisions':len(steps),'labels':rows,'real_time_factor':factor,'kept':good and bool(recorder),
                  # The median: in takeover mode the vehicle may wait between two decisions after the model's stop.
                  'decision_period_s':round(statistics.median([b['epoch']-a['epoch'] for a,b in zip(steps,steps[1:])] or [0.]),3),'stream_frames':dict(env.stream.counts),
                  'view':'unreal' if args.unreal_view else 'cameras','mode':'takeover' if policy else 'human','band':band,'random':drawn,
                  'start_distance_m':round(steps[0]['distance_m'],1),'precision':landing}
            if policy:line.update(model=model.get('name'),takeovers=extra['takeover']['takeovers'],switches=[[item['decision'],item['to']] for item in switches])
            with (output/'flights.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(line)+'\n')
            print('FLIGHT',json.dumps(line),flush=True)
            message=(f'KEPT {episode["id"]}' if good and recorder else f'NOT KEPT {episode["id"]}')+f': {reason}, success={bool(reading.get("success"))}, {rows["labels"]} labels, {rows["decisions"]-rows["labels"]-rows["terminal"]} left out'
            if policy:message+=f', taken over {extra["takeover"]["takeovers"]} times'
            if landing and landing['error_m'] is not None:message+=f', landed {landing["error_m"]:.1f} m from the middle (data needs {landing["precision_m"]:.2f} m or less)'
            if args.episodes and flown>=args.episodes:break
    finally:
        cv2.destroyAllWindows()
        if policy:policy.close()
        try:client.disconnect()
        except Exception:pass


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--ports',type=int,nargs=2,default=[8989,8990])
    parser.add_argument('--config',default=str(session.CONFIG));parser.add_argument('--scene');parser.add_argument('--layout')
    parser.add_argument('--target');parser.add_argument('--task',choices=('land','approach'));parser.add_argument('--seed',type=int)
    parser.add_argument('--record',help='dataset folder (default: the configuration\'s)');parser.add_argument('--no-record',action='store_true')
    parser.add_argument('--output',default=str(ROOT/'outputs/human_pilot'),help='logs and the simulator configuration of this session')
    parser.add_argument('--script',help='fly a fixed list of inputs instead of a person (a check of the loop)')
    parser.add_argument('--episodes',type=int,default=0,help='stop after this many flights (0: until Esc)')
    parser.add_argument('--snapshot',help='save the window once, at the twelfth decision of a flight (to show what the pilot sees)')
    parser.add_argument('--view-px',type=int,help='resolution of the display cameras (default: the configuration\'s)')
    parser.add_argument('--unreal-view',action='store_true',help='the simulator\'s own window is shown; keys work in it too; the flights are recorded with view: unreal')
    parser.add_argument('--band',choices=('near','mid','far'),help='distance band of the starts (default: the configuration\'s); D changes it before a flight')
    parser.add_argument('--random',action='store_true',help='draw the layout, the object and the task for every flight; R toggles it before a flight')
    parser.add_argument('--takeover',metavar='READY_FILE',help='takeover mode: the model served by src/human_pilot/remote.py (which writes READY_FILE once loaded) flies first')
    asyncio.run(main(parser.parse_args()))
