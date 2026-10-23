"""Human pilot mode: a person flies the vehicle from the front and down pictures alone, and the flight is recorded as data.

  human_pilot.py --host 127.0.0.1 [--scene field] [--layout a] [--target blue_cube] [--task land] [--no-record]

The window shows the two pictures a policy is given (256 x 256 each, enlarged), the sentence, and the pilot's own
commands. It shows no map, no position, no height and nothing of where the object is; the simulator's own window is
hidden by the launcher (scripts/run_human_pilot.ps1). Keys and pad: src/human_pilot/devices.py. What is recorded and
what is left out of the labels: src/human_pilot/controls.py and src/human_pilot/session.py; docs/human_pilot_mode.md.

The vehicle is flown as a policy's flights are, with the same call (a velocity along the heading, a height to hold, a
heading to turn to) and the same limits; only the commands come twenty times a second, eased in, instead of once per
decision. Every 0.5 s the pictures are taken as the canonical loop takes them (scripts/visual_search.SearchEnv.observe)
and the commands of the following 0.5 s become that decision's label.

--script FILE flies a fixed list of inputs instead of a person: a check of the whole loop with the simulator.
"""
import argparse
import asyncio
import json
import math
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
STATES={'human':'FLYING','stop':'DONE - holding','landed':'TOUCHDOWN - holding'}


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
        for sensor,name in (('FrontCamera','front'),('DownCamera','down')):
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


def streaming(prepare):
    """The canonical scene configuration with the two RGB cameras also published as a stream (for the display; the
    recorded pictures are still requested as the canonical loop requests them)."""
    def prepared(root,output):
        directory=prepare(root,output);path=directory/'robot_quadrotor_fastphysics.jsonc';robot=json.loads(path.read_text())
        for sensor in robot.get('sensors',[]):
            if sensor.get('id') in ('FrontCamera','DownCamera'):
                for setting in sensor.get('capture-settings',[]):
                    if setting.get('image-type')==0:setting['streaming-enabled']=True
        path.write_text(json.dumps(robot,indent=2));return directory
    return prepared


def environment(canonical):
    from src.landing.env import SurfaceContacts

    class PilotEnv(SurfaceContacts,canonical.SearchEnv):
        stream=None

        async def reset(self,episode):
            original=canonical.prepare_mission_config;canonical.prepare_mission_config=streaming(original)
            try:await super().reset(episode)
            finally:canonical.prepare_mission_config=original
            named=self.surfaces[self.target];self.surface=named.top_m if named.landable else 0.
            self.stream.attach(self.client,self.drone)

    return PilotEnv


class Script:
    """Inputs read from a file instead of a person: [[seconds, {"forward": true, ...}], ...]. A segment may press a key once
    when it begins (`"press": "enter"`). In the menu, before the flight, it presses Enter; after the script, nothing."""
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


def bar(canvas,x,y,width,value,top,label):
    """A centred bar for a signed rate."""
    cv2.putText(canvas,label,(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1,cv2.LINE_AA)
    cv2.rectangle(canvas,(x,y),(x+width,y+14),(200,200,200),1)
    middle=x+width//2;end=middle+int(width/2*max(-1.,min(1.,value/top if top else 0.)))
    cv2.rectangle(canvas,(min(middle,end),y+2),(max(middle,end),y+12),(60,120,200),-1)


def render(front,down,lines,rates=None,limits=None,scale=2,footer=''):
    """The pilot's window: the sentence and state on top, the two pictures, the pilot's own commands. Nothing else."""
    size=256*scale;width=2*size+60;height=size+190;canvas=np.full((height,width,3),245,np.uint8)
    for row,(text,colour,big) in enumerate(lines):
        cv2.putText(canvas,text,(20,32+row*28),cv2.FONT_HERSHEY_SIMPLEX,.8 if big else .55,colour,2 if big else 1,cv2.LINE_AA)
    for column,(name,frame) in enumerate((('FRONT',front),('DOWN',down))):
        x=20+column*(size+20);y=100
        if frame is None:cv2.rectangle(canvas,(x,y),(x+size,y+size),(30,30,30),-1)
        else:canvas[y:y+size,x:x+size]=cv2.resize(frame,(size,size),interpolation=cv2.INTER_NEAREST)
        cv2.putText(canvas,name,(x+8,y+24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
    if rates is not None and limits is not None:
        y=size+125
        bar(canvas,20,y,220,rates.forward,limits.forward,f'forward {rates.forward:+.2f} m/s')
        bar(canvas,270,y,220,-rates.down,limits.down,f'up {-rates.down:+.2f} m/s')
        bar(canvas,520,y,220,rates.turn,limits.turn,f'turn {math.degrees(rates.turn):+.1f} deg/s')
    if footer:cv2.putText(canvas,footer,(20,height-14),cv2.FONT_HERSHEY_SIMPLEX,.45,(90,90,90),1,cv2.LINE_AA)
    return canvas


KEYS_FLYING='W/Up forward | A D turn | R/PgUp up | F/PgDn down | 1 2 3 speed | Enter done | Backspace discard | Esc quit'
KEYS_MENU='N/P object | T land/approach | L layout | M scene | Enter fly | Esc quit'


def show(canvas):
    cv2.imshow(WINDOW,canvas);return cv2.waitKey(1)


def make_step(env,observation,index,instruction,mode,recorder,oft):
    """One decision as the canonical loop writes it (scripts/visual_search.run_episode), with the pictures saved."""
    from src.aerovla_oft.spec import proprio_vector
    from src.visual_search.episodes import policy_inputs
    private=observation['privileged'];state=observation['state']
    proprio=proprio_vector(state,env.ground_z,oft,observation['yaw_rate'])
    inputs=policy_inputs(observation['frames']['front'],observation['frames']['down'],instruction,None)
    step={'step':index,'epoch':time.time(),'position':state['position'],'yaw_rad':observation['yaw'],'height_m':private['height_m'],'bearing_deg':private['bearing_deg'],
          'distance_m':private['distance_m'],'front_seen':private['front']['seen'],'front_in_fov':private['front']['in_fov'],'down_seen':private['down']['seen'],
          'front_pixel':private['front']['pixel'],'down_pixel':private['down']['pixel'],'ahead_m':private['ahead_m'],'over':private['over'],
          'landed':env.resting(state) is not None,'on_surface':env.on_surface(state),'velocity':state['velocity'],'target_size':private['size'],'finalizer':'OFF',
          'input_sha256':{name:session.digest(inputs[name]) for name in ('front','down')},'proprio':proprio,'teacher_state':mode,'perturbed':False,
          'sim_stamp':observation['stamp']}
    if recorder:step['img_name']=recorder.frames(index,inputs['front'],inputs['down'])
    return step


async def fly(env,episode,devices,settings,oft,config,recorder,scale,tier_index=1,snapshot=None):
    """One flight. Returns (reason, stopped, steps, touchdown, hit) or None when the pilot discarded it or quit (then the
    second value says which)."""
    tick=oft['tick_s'];dt=1./settings['control_hz'];tiers=controls.tiers_of(oft);limits=controls.limits_of(oft)
    acceleration=settings['acceleration'];smoother=Smoother(Axes(acceleration['forward_mps2'],acceleration['down_mps2'],acceleration['turn_radps2']))
    labeler=TickLabel(oft);steps=[];index=0;terminal=None;terminal_left=settings['terminal_ticks'];reason='max_steps';stopped=False
    opened=time.monotonic();observation=env.observe();heading=observation['yaw'];z_set=env.setpoint_z
    pending=make_step(env,observation,index,episode['instruction'],'human',recorder,oft);rates=Axes();saved=False
    if hasattr(devices,'flying'):devices.flying()
    last=time.monotonic();touchdown=hit=None
    while True:
        begun=time.monotonic();elapsed=max(1e-3,begun-last);last=begun
        inputs,pressed,_=devices.read()
        if 'escape' in pressed:return None,'quit'
        if 'backspace' in pressed or 'pad_b' in pressed:return None,'discard'
        for key,number in (('1',0),('2',1),('3',2)):
            if key in pressed:tier_index=number
        if 'pad_lb' in pressed:tier_index=max(0,tier_index-1)
        if 'pad_rb' in pressed:tier_index=min(2,tier_index+1)
        touchdown,hit=env.contacts()
        if hit:reason='collision';break
        if terminal is None and episode['task']=='land' and touchdown:terminal='landed';smoother.stop()
        if terminal is None and ('enter' in pressed or 'pad_a' in pressed):
            if episode['task']=='approach':terminal='stop';smoother.stop()
            else:reason='gave_up';break
        target=Axes() if terminal else controls.wanted(inputs,tiers[tier_index],oft)
        rates=smoother.step(target,elapsed) if not terminal else smoother.stop()
        before=z_set;heading+=rates.turn*elapsed
        z_set=max(env.ground_z-env.limits['maximum_clearance_m'],min(env.ground_z-env.limits['minimum_clearance_m'],z_set+rates.down*elapsed))
        labeler.add(Axes(rates.forward,(z_set-before)/elapsed,rates.turn),elapsed)
        await env.drone.move_by_velocity_z_async(rates.forward*math.cos(heading),rates.forward*math.sin(heading),z_set,duration=settings['command_duration_s'],
                                                 yaw_is_rate=False,yaw=heading)
        if time.monotonic()-opened>=tick:
            label=labeler.close();pending.update(teacher_action=label,action=label,executed=label)
            mark=controls.mark(label,oft,terminal=pending['teacher_state']!='human')
            if mark:pending['excluded']=mark
            steps.append(pending)
            if pending['teacher_state']!='human':
                terminal_left-=1
                if terminal_left<=0:reason='human_stop';stopped=True;break
            index+=1
            if index>=episode.get('max_ticks',settings['max_ticks']):reason='max_steps';break
            opened=time.monotonic();observation=env.observe()
            pending=make_step(env,observation,index,episode['instruction'],terminal or 'human',recorder,oft)
        state=STATES[terminal or 'human'];tier=['slow','normal','fast'][tier_index]
        lines=[(episode['instruction'],(20,20,20),True),(f'{state} | decision {index} | speed {tier} | {"REC" if recorder else "not recorded"}',(0,0,170) if recorder else (90,90,90),False)]
        # The stream when it runs; else the pictures of the last decision (the stream is for the display only).
        front=env.stream.latest('front');down=env.stream.latest('down')
        canvas=render(front if front is not None else observation['frames']['front'],down if down is not None else observation['frames']['down'],lines,rates,limits,scale,KEYS_FLYING)
        show(canvas)
        if snapshot and not saved and index>=12:cv2.imwrite(str(snapshot),canvas);saved=True
        await asyncio.sleep(max(0.,dt-(time.monotonic()-begun)))
    return (reason,stopped,steps,touchdown,hit),None


async def main(args):
    import scripts.visual_search as canonical
    from projectairsim import ProjectAirSimClient
    from src.aerovla_oft.spec import load_config as load_oft
    from src.landing import evaluator
    from src.visual_search.episodes import load_config
    from src.visual_search.finalizer import LandingFinalizer
    from src.visual_search.maps import load_map
    settings=session.load(args.config);config=load_config();oft=load_oft();scale=settings['display_scale']
    scenes=settings['scenes'];scene=args.scene or scenes[0];layout=args.layout or settings['layouts'][0]
    if scene not in scenes or layout not in settings['layouts']:raise SystemExit(f'scene and layout must be recording ones: {scenes}, {settings["layouts"]}')
    record_root=None if args.no_record else Path(args.record or ROOT/settings['dataset'])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    from src.human_pilot.devices import Devices
    devices=Script(args.script) if args.script else Devices(WINDOW)
    maps={name:load_map(name) for name in scenes};avoid={name:session.check_starts(name) for name in scenes}
    objects=session.targets_of(maps[scene],layout);chosen=next((index for index,item in enumerate(objects) if item['name']==args.target),0)
    task=args.task or ('land' if objects[chosen]['landable'] else 'approach');seed=args.seed if args.seed is not None else int(time.time())%100000
    client=ProjectAirSimClient(address=args.host,port_topics=args.ports[0],port_services=args.ports[1]);client.connect();client.socket_services.recv_timeout=60000
    env=environment(canonical)(client,config,oft,output);env.stream=Stream();flown=0;message=''
    cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            item=objects[chosen]
            if task=='land' and not item['landable']:task='approach'
            lines=[('HUMAN PILOT - choose what to fly',(20,20,20),True),
                   (f'scene {scene} | layout {layout} | {item["noun"]} | {task.upper()} | start: a random place {settings["distance_m"][0]:.0f}-{settings["distance_m"][1]:.0f} m away',(20,20,20),False),
                   (message,(0,110,0) if message.startswith('KEPT') else (0,0,170),False)]
            show(render(None,None,lines,scale=scale,footer=KEYS_MENU))
            inputs,pressed,_=devices.read()
            if 'escape' in pressed:break
            if 'n' in pressed:chosen=(chosen+1)%len(objects)
            if 'p' in pressed:chosen=(chosen-1)%len(objects)
            if 't' in pressed:task='approach' if task=='land' else ('land' if item['landable'] else 'approach')
            if 'l' in pressed or 'm' in pressed:
                if 'l' in pressed:layout=settings['layouts'][(settings['layouts'].index(layout)+1)%len(settings['layouts'])]
                if 'm' in pressed:scene=scenes[(scenes.index(scene)+1)%len(scenes)]
                objects=session.targets_of(maps[scene],layout);chosen=min(chosen,len(objects)-1)
            if not ({'enter','pad_start'}&pressed):
                await asyncio.sleep(.05);continue
            if args.episodes and flown>=args.episodes:break
            episode=session.draw(settings,config,maps[scene],layout,item['name'],task,seed,avoid[scene]);seed+=1
            show(render(None,None,[(episode['instruction'],(20,20,20),True),('PREPARING: loading the scene and flying to the start ...',(0,0,170),False)],scale=scale))
            recorder=session.Recorder(record_root,episode) if record_root else None
            await env.reset(episode);clock=time.time();stamp=env.drone.get_ground_truth_kinematics()['time_stamp']
            try:result,why=await fly(env,episode,devices,settings,oft,config,recorder,scale,snapshot=args.snapshot)
            except Exception:
                traceback.print_exc();result,why=None,'error'
            if result is None:
                if recorder:recorder.discard()
                await env.close();message=f'{episode["id"]}: {why}, nothing kept'
                if why=='quit':break
                continue
            reason,stopped,steps,touchdown,hit=result;flown+=1
            if not steps:
                await env.close();message=f'{episode["id"]}: {reason} before the first decision, nothing kept'
                if recorder:recorder.discard()
                continue
            finalizer=LandingFinalizer(env.landing['finalizer'],episode['instruction'],enabled=False)
            summary=canonical.summarise(episode,steps,config,reason,stopped,env.others(steps[-1]['position']),env.landing,touchdown,hit,finalizer.report())
            reading=evaluator.read(summary,env.surfaces,env.landing,steps)
            wall=steps[-1]['epoch']-clock;factor=((steps[-1]['sim_stamp']-stamp)/1e9/wall) if wall>0 else None
            good=session.kept(reading,summary,factor,settings);rows=session.label_rows(steps)
            if recorder:
                recorder.finish({**summary,'surface_landing':reading,'real_time_factor':factor,'pilot':'human','controls':'script' if args.script else 'keyboard/pad','labels':rows},
                                steps,good,settings['keep_rejected_frames'])
            await env.close()
            line={'id':episode['id'],'scene':scene,'layout':layout,'target':item['name'],'task':task,'reason':reason,'success':bool(reading.get('success')),
                  'landed_on':reading.get('landed_on'),'collision':reading.get('collision'),'decisions':len(steps),'labels':rows,'real_time_factor':factor,'kept':good and bool(recorder),
                  'decision_period_s':round((steps[-1]['epoch']-steps[0]['epoch'])/max(1,len(steps)-1),3),'stream_frames':dict(env.stream.counts)}
            with (output/'flights.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(line)+'\n')
            print('FLIGHT',json.dumps(line),flush=True)
            message=(f'KEPT {episode["id"]}' if good and recorder else f'NOT KEPT {episode["id"]}')+f': {reason}, success={bool(reading.get("success"))}, {rows["labels"]} labels, {rows["pause"]+rows["below_safe"]} left out'
            if args.episodes and flown>=args.episodes:break
    finally:
        cv2.destroyAllWindows()
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
    asyncio.run(main(parser.parse_args()))
