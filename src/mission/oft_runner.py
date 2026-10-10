"""Interactive missions flown by an AeroVLA-OFT checkpoint (the frozen `grounding_film` by default).

Mission Control is the map, the evaluator and the log. The policy is the canonical one: every flight here is
`scripts.visual_search.run_episode` itself, started as an evaluation episode starts (`SearchEnv.reset`: the scene loaded
with the vehicle at its launch pose, take-off, climb to the cruise height). The observation, the sentence, the chunk of
which one action is executed, the continuous command, the stop rule and the landing finalizer are that loop's; none of
them is written again here. This module watches the loop through the environment's own methods and through the call to
the policy, and adds what an interactive session needs around it.

What the policy receives is what `policy_inputs` builds: the Front frame, the Down frame and one sentence. The selected
object's place, its distance and bearing, the map and its markers are used here for the display, the score and the log;
`WatchedPolicy.infer` has no parameter through which they could pass.

A landing is flown by the policy and ended by the canonical finalizer. The simulator's own landing routine is not called
anywhere in this mode, not even when the session is closed with the vehicle in the air.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import asyncio
import contextlib
import datetime
import json
import sys
import threading
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/mission_demo_grounding'
OUT.mkdir(parents=True,exist_ok=True)
if '--mission-demo' in sys.argv:(OUT/'worker-pid.txt').write_text(str(os.getpid()))
from projectairsim import ProjectAirSimClient
import scripts.visual_search as canonical
from scripts.visual_search import SearchEnv,run_episode,MANIFEST
from src.integration.blur_demo_support import DemoQuit
from src.integration.projectairsim_observation_adapter import adapt_state
from src.mission import semantic
from src.mission.checkpoint import inspect
from src.mission.control import pending_requests,TASKS,GROUNDING,INSTRUCTION_ONLY
from src.mission.feed import Feed,sha256
from src.mission.grounding import target_visibility,matched_camera_images
from src.mission.scene import OverviewScene
from src.visual_search.episodes import load_config,relative_target

MODEL_INPUT='Front RGB + Down RGB + Instruction ONLY'
NOT_SENT=('target XYZ','distance','bearing','direction hint','overview','marker')
CONNECTION_WORDS=('Connection','closed','NNG','Timed out','timed out')


class MissionAbort(Exception):
    """R during a flight: the flight ends where it is and is recorded as aborted."""


class WatchedPolicy:
    """The policy as the canonical loop calls it, with the display told what went in and what came out.

    `infer` is the only way from a mission to the model, and its parameters are the canonical loop's own: two frames, one
    sentence, and the vehicle's motion vector for a checkpoint trained with one. There is no parameter for a target."""
    def __init__(self,model,watch):self.model,self.watch=model,watch

    def infer(self,front,down,instruction,proprio=None):
        self.watch.model_input(front,down,instruction,proprio)
        result=self.model.infer(front,down,instruction,proprio)
        self.watch.model_output(result)
        return result


@contextlib.contextmanager
def watched_finalizer(listener):
    """Let the display see the canonical landing finalizer while a flight runs.

    The loop creates its finalizer itself, so for the length of one flight the class it creates is the canonical one with
    a report added after each of its own steps. It decides nothing differently and is told nothing more."""
    original=canonical.LandingFinalizer
    class Watched(original):
        def update(self,*arguments):
            state=super().update(*arguments);listener(self);return state
        def disarmed(self,ok):
            super().disarmed(ok);listener(self)
    canonical.LandingFinalizer=Watched
    try:yield
    finally:canonical.LandingFinalizer=original


def loaded_modules(model,saved):
    """What was read from the checkpoint. A checkpoint trained with a grounding module must have brought that module's weights:
    without them the module is the identity and the model flown would silently be another one."""
    from src.aerovla_oft.model import OFT_ADAPTER
    lora={name:p for name,p in model.peft.named_parameters() if 'lora_' in name and f'.{OFT_ADAPTER}.' in name}
    kind=(saved.get('grounding') or {}).get('type','none')
    info={'name':saved.get('name'),'grounding':kind,'grounding_parameters':model.load_info['grounding_parameters'],
          'grounding_loaded':bool(model.load_info['grounding_loaded']),
          'lora_parameters':sum(p.numel() for p in lora.values()),
          # A new adapter's B matrices are zero; trained ones are not.
          'lora_loaded':any(float(p.float().abs().sum())>0 for name,p in lora.items() if 'lora_B' in name),
          'head_parameters':model.load_info['head_parameters'],'head_loaded':True,
          'chunk_size':saved['chunk_size'],'execute_horizon':saved['execute_horizon'],'tick_s':saved['tick_s'],
          'proprio':bool(saved['proprio']['enabled']),'load_seconds':model.load_info['load_seconds']}
    if kind!='none':
        # The module's last layer starts at zero (the identity); trained weights are not zero.
        info['grounding_loaded']=bool(info['grounding_loaded'] and any(float(p.abs().sum())>0 for p in model.grounding_parameters()))
        if not info['grounding_loaded']:raise RuntimeError(f'The checkpoint names a {kind} module but brought no trained weights for it')
    if not info['lora_loaded']:raise RuntimeError('The checkpoint brought no trained LoRA weights')
    return info


class Loader:
    """Loads the checkpoint beside the map, so that the first mission does not wait for it."""
    def __init__(self,checkpoint,saved):
        self.model=None;self.info=None;self.error=None
        self.thread=threading.Thread(target=self._load,args=(checkpoint,saved),name='model-loader',daemon=True);self.thread.start()

    def _load(self,checkpoint,saved):
        try:
            from src.aerovla_oft.model import AeroVLAOFT
            model=AeroVLAOFT(MANIFEST,saved,checkpoint=str(checkpoint))
            # (The first pass through a freshly loaded model takes about a second longer than any other, in an evaluation run
            # as here. The vehicle hovers through it; no extra call is made to the model to hide it.)
            self.info=loaded_modules(model,saved);self.model=model
        except Exception:self.error=traceback.format_exc()


class Watch:
    """What the display and the log are told. Data flows one way: nothing kept here is read by the policy."""
    def __init__(self,feed,limit):
        self.feed=feed;self.last_request=0;self.overview=None;self.view=None;self.last_map=0.
        self.telemetry={'status':'RUNNING','phase':'Starting','step':0,'input_step':0,'completed_steps':0,'max_steps':limit,
                        'input_files':{},'failure':{'failure_enabled':False},'trajectory':[],'mission':{'state':'IDLE','target':None,'errors':None}}
        self.reset_flight()

    def reset_flight(self):
        self.step=0;self.inputs=[];self.trajectory=[];self.raw={};self.blur={'failure_enabled':False};self.clock=None;self.periods=[];self.map_ms=None
        self.telemetry.update(step=0,input_step=0,completed_steps=0,input_files={},trajectory=[],inference=None,executed=None,finalizer=None,
                              contact=None,evaluator=None,model_input=None,timing=None,input_verified=None)

    def update(self,**fields):
        self.telemetry.update(fields);self.publish()

    def publish(self):
        self.telemetry.update(updated_at=time.time(),chase_ready=self.feed.chase_ready,control_warning=self.feed.control_warning,feed_errors=self.feed.errors[-3:])
        self.feed.json('telemetry.json',self.telemetry)

    # --- the map ------------------------------------------------------------------------------------------------
    def refresh_map(self,state,period=None):
        """Capture the overview when its view was changed, or (between flights) when it has grown old.

        A capture takes the simulator about 0.18 s, a third of a decision. In flight none is made unless the user changes the
        view: the map is a still picture then, and the viewer draws the vehicle and its path on it from the telemetry."""
        if self.overview is None:return
        control=self.feed.control
        view=(control.get('overview_view','top'),float(control.get('overview_zoom',1.)),tuple(control.get('overview_pan',[0,0])))
        if view==self.view and (period is None or time.monotonic()-self.last_map<period):return
        started=time.perf_counter()
        self.overview.capture(state,view[0],view[1],view[2],'map')
        self.view=view;self.last_map=time.monotonic();self.map_ms=(time.perf_counter()-started)*1000

    # --- a flight, as the canonical loop goes through it ---------------------------------------------------------------
    def poll(self):
        """Keys read between two decisions: Q/Esc leaves the session, R the flight. Anything else waits until the flight ends."""
        control=self.feed.control
        if control['quit']:raise DemoQuit('Exit requested')
        for request_id,request in pending_requests(control,self.last_request):
            self.last_request=request_id
            if request['action']=='reset':raise MissionAbort('Reset requested during the flight')
            self.telemetry.update(mission_request_ack=request_id,control_message='A mission is flying: R ends it and returns to the launch pose, Q/Esc leaves')

    def decision(self):
        now=time.monotonic()
        if self.clock is not None:self.periods=(self.periods+[now-self.clock])[-20:]
        self.clock=now

    def contact(self,touchdown,hit):
        if touchdown or hit:
            self.telemetry['contact']={'touchdown_on':touchdown['object'] if touchdown else None,
                                       'touchdown_offset_m':touchdown['offset_m'] if touchdown else None,'collision':hit}

    def observed(self,observation,raw,blur,target):
        state=observation['state'];private=observation['privileged'];self.raw=raw;self.blur=blur
        self.trajectory.append(list(state['position']))
        # For the evaluator's panel and the log. None of this is in what the policy is given.
        self.telemetry.update(state=state,trajectory=self.trajectory,
                              evaluator={'target_xyz':target['position'],'distance_m':private['distance_m'],'bearing_deg':private['bearing_deg'],
                                         'height_m':private['height_m'],'front_seen':private['front']['seen'],'down_seen':private['down']['seen'],
                                         'front_pixel':private['front']['pixel'],'down_pixel':private['down']['pixel'],'over_target':private['over']})

    def model_input(self,front,down,instruction,proprio):
        self.step+=1;slot='a' if self.step%2 else 'b'
        files={name:f'input_{slot}_{name}.png' for name in ('front','down')};hashes={'front':sha256(front),'down':sha256(down)}
        # The files the viewer shows are these very arrays, and the hashes let it check that.
        self.feed.image(files['front'],front);self.feed.image(files['down'],down)
        unchanged=front is self.raw.get('front') and down is self.raw.get('down')
        self.inputs.append({'step':self.step,'input_sha256':hashes,'instruction':instruction,'proprio':proprio,'camera_frames_unchanged':unchanged,
                            'failure':self.blur['failure_type'] if self.blur.get('failure_enabled') else 'normal'})
        self.telemetry.update(step=self.step,input_step=self.step,input_files=files,phase='AeroVLA-OFT inference',
                              failure={**self.blur,'used_frame_sha256':hashes,'camera_frames_unchanged':unchanged},
                              model_input={'step':self.step,'instruction':instruction,'sha256':hashes,'proprio_sent':proprio is not None,
                                           'scope':'actual model input'},input_verified=None)
        self.publish()

    def model_output(self,result):
        self.telemetry['inference']={'chunk':result['chunk'],'action':result['chunk'][0],'normalised':result['normalised_chunk'][0],
                                     'inference_ms':result['inference_ms'],'stop':result['stop'],'prompt':result['prompt']}

    def finalizer(self,finalizer):
        self.telemetry['finalizer']=finalizer.report()

    def executed(self,action):
        self.telemetry['executed']=list(action)

    def ticked(self,state):
        period=sum(self.periods)/len(self.periods) if self.periods else None
        self.telemetry.update(completed_steps=self.step,phase='AeroVLA-OFT flight',
                              timing={'decision_s':period,'map_capture_ms':self.map_ms,
                                      'inference_ms':(self.telemetry.get('inference') or {}).get('inference_ms')})
        self.refresh_map(state);self.publish()

    def disarmed(self,ok):
        self.telemetry['disarmed']=bool(ok)


class MissionEnv(SearchEnv):
    """The canonical environment with a display watching it. Every override calls the canonical method and changes nothing in
    what it does, except `close`: a flight that ends in the air is left hovering there."""
    def __init__(self,client,config,oft,output,watch):
        super().__init__(client,config,oft,output);self.watch=watch;self._drone=None;self.semantic=None;self.last_state=None

    @property
    def drone(self):return self._drone

    @drone.setter
    def drone(self,value):
        # A loaded scene has a new vehicle; the observer's chase camera follows it from its first moment.
        self._drone=value
        self.client.subscribe(value.sensors['Chase']['scene_camera'],self.watch.feed.chase_callback)

    def contacts(self):
        result=super().contacts();self.watch.contact(*result);return result

    def observe(self):
        self.watch.decision();self.watch.poll()
        observation=super().observe();raw=observation['frames']
        # The blur switch of the earlier demos. Off (the default) the policy is given the camera's own arrays.
        used,blur=self.watch.feed.fail(raw)
        if used is not raw:observation=dict(observation,frames=used)
        self.last_state=observation['state'];self.watch.observed(observation,raw,blur,self.semantic)
        return observation

    async def tick(self,action,yaw):
        self.watch.executed(action)
        await super().tick(action,yaw)
        self.watch.ticked(self.last_state)

    async def close(self):
        """A flight that ended in the air: hover, motors on. (The evaluation switches them off here, because its next episode
        loads the scene again; a mission's vehicle stays where the policy stopped it, for the user to look at.)"""
        try:await asyncio.wait_for(await self.drone.hover_async(),timeout=10)
        except Exception:pass

    def disarm(self):
        ok=super().disarm();self.watch.disarmed(ok);return ok

    def state(self):
        return adapt_state(self.drone.get_ground_truth_kinematics())

    def preview(self,point=None):
        """Front and Down as they are now, for the display while no mission flies, and where a point falls in each."""
        frames={};visibility={}
        for sensor,name in (('FrontCamera','front'),('DownCamera','down')):
            messages=self.drone.get_images(sensor,[0,1])
            if 0 not in messages or 1 not in messages:return None,None
            image,depth,meta=matched_camera_images(messages);frames[name]=image
            if point is not None:
                item=target_visibility(point,meta,depth);visibility[name]={'status':item['status'],'pixel':item['pixel'],'in_fov':item['in_fov'],'visible':item['visible']}
        return frames,visibility


def run_folder(token):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder=OUT/'missions'/f'{stamp}-{(token or "direct")[:8]}';folder.mkdir(parents=True,exist_ok=True)
    return folder


async def main(args):
    config=load_config();demo_path=Path(args.demo) if args.demo else semantic.DEMO;demo=semantic.load_demo(demo_path);settings=demo['mission']
    checkpoint=Path(args.checkpoint or settings['checkpoint']);checkpoint=checkpoint if checkpoint.is_absolute() else ROOT/checkpoint
    # The launcher has compared the frozen baseline's fingerprints on the Windows side; started without it, that is done here.
    recorded=OUT/'checkpoint-check.json'
    check=json.loads(recorded.read_text(encoding='utf-8')) if recorded.exists() else inspect(checkpoint,settings['frozen_record'])
    if check['problems']:raise SystemExit('; '.join(check['problems']))
    saved=json.loads((checkpoint/'manifest.json').read_text(encoding='utf-8'))['config']
    limit=args.steps or config['gen_v3']['max_ticks'];radius=config['success_radius_m'];canonical_range=config['canonical']['max_start_m']
    targets=semantic.semantic_targets(demo);launches=settings['launches'];launch=launches[0]
    feed=Feed(OUT);watch=Watch(feed,limit);folder=run_folder(args.run_token)
    policy_info={'mode':GROUNDING,'model':checkpoint.name,'name':saved.get('name'),'checkpoint':check['checkpoint'],
                 'grounding':(saved.get('grounding') or {}).get('type','none'),'frozen':check['frozen']['state'],'warnings':check['warnings'],
                 'loaded':None,'prompt_mode':INSTRUCTION_ONLY,'model_input':MODEL_INPUT,'not_sent':list(NOT_SENT),
                 'chunk_size':saved['chunk_size'],'execute_horizon':saved['execute_horizon'],'tick_s':saved['tick_s'],
                 'simulator_landing_routine':'not used'}
    client=None;env=None;loader=None;overview=None;target=None;index=-1;task='land';mission_id=0;ready=None;vehicle='on the ground'
    state_name='IDLE';reason=None;result=None;flown=None;program_status='RUNNING';history=[];last_preview=last_hover=0.;preview_counter=0

    def snapshot(state=None):
        waiting=state_name in ('IDLE','TARGET_SELECTED');refused=semantic.refusal(target,task) if waiting else None
        # While a mission flies or its result is shown, the task and the sentence are the ones it was flown with.
        shown_task,text=flown if flown and not waiting else (task,semantic.sentence(config,demo,target,task) if target and not refused else None)
        view=semantic.launch_view(launch,target) if target else None
        live=relative_target(state,target['position']) if target and state else None
        return {'state':state_name,'reason':reason,'steps':watch.step,'task':shown_task,'instruction':text,'refusal':refused,
                'target':dict(target,surface_position=target['position'],
                              launch_distance_m=view['distance_m'],launch_bearing_deg=view['bearing_deg'],
                              in_canonical_range=view['distance_m']<=canonical_range) if target else None,
                'errors':{'horizontal_m':live[1],'bearing_deg':live[0]} if live else None,
                'success_radius_m':radius,'canonical_range_m':canonical_range,'result':result,'vehicle':vehicle,'mission_id':mission_id}

    def show(phase=None,state=None,**fields):
        state=state if state is not None else watch.telemetry.get('state')
        watch.update(mission=snapshot(state),policy=policy_info,launch=dict(launch,index=launches.index(launch),count=len(launches)),
                     targets=targets,**({'phase':phase} if phase else {}),**({'state':state} if state is not None else {}),**fields)

    async def prepare(wanted=None):
        """Start as an evaluation episode starts: the scene loaded with the vehicle at the launch pose, take-off, climb to the
        cruise height. The object named here only prepares the evaluator's view of it; a mission to another object prepares again."""
        nonlocal ready,vehicle
        chosen=wanted or targets[0]
        show(f'Loading the scene: take-off at {launch["name"]}');feed.flush(1.)
        episode=semantic.mission_episode(demo_path,demo,chosen,'approach',semantic.sentence(config,demo,chosen,'approach'),launch,mission_id,limit)
        await env.reset(episode);env.semantic=chosen
        ready=(launch['id'],chosen['object_id']);vehicle='hovering at the launch pose'
        watch.reset_flight();watch.telemetry.update(ground_z=env.ground_z)
        if overview is not None:
            # The map of the scene just loaded is taken now, so that a flight starting next does not pay for it.
            overview.rebind(env.world,env.drone);watch.view=None;watch.refresh_map(env.state(),0.)

    def record(entry):
        history.append(entry)
        (folder/f'mission-{entry["mission_id"]:03d}.json').write_text(json.dumps(entry),encoding='utf-8')
        brief={key:value for key,value in entry.items() if key not in ('steps','model_inputs')}
        with (folder/'missions.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(brief)+'\n')

    async def fly():
        """One mission: the canonical episode loop from the launch pose, watched."""
        nonlocal mission_id,ready,vehicle,state_name,reason,result,flown
        text=semantic.sentence(config,demo,target,task);mission_id+=1;flown=(task,text)
        episode=semantic.mission_episode(demo_path,demo,target,task,text,launch,mission_id,limit)
        if ready!=(launch['id'],target['object_id']):await prepare(target)
        ready=None;env.semantic=target;watch.reset_flight();state_name='NAVIGATING';reason=None;result=None;vehicle='flying'
        show('AeroVLA-OFT flight',control_message=None,disarmed=None)
        arguments=argparse.Namespace(policy='oft',finalizer='on',live=False,output=str(OUT),set=None,model_name=saved.get('name'))
        entry={'mission_id':mission_id,'started_epoch':time.time(),'checkpoint':check['checkpoint'],'model':policy_info['loaded'],
               'frozen_baseline':check['frozen']['state'],'target_object':target['object_id'],'target_noun':target['noun'],'target':target,
               'task':task,'instruction':text,'launch':launch,'model_input':MODEL_INPUT,'not_sent_to_model':list(NOT_SENT)}
        summary=None;steps=[];error=None;back=False
        try:
            with watched_finalizer(watch.finalizer):
                summary,steps=await run_episode(env,episode,WatchedPolicy(loader.model,watch),arguments,None)
        except MissionAbort:
            await env.close();state_name='ABORTED';reason='manual abort (R)';vehicle='hovering';back=True
        except DemoQuit:
            state_name='ABORTED';reason='manual abort (Q/Esc)';vehicle='hovering'
            record({**entry,'state':state_name,'reason':reason,'success':False,'model_inputs':list(watch.inputs),'steps':[]})
            raise
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            # A lost simulator ends the session; anything else ends this mission and is shown.
            if any(word in error for word in CONNECTION_WORDS):raise
            await env.close();state_name='FAILED';reason='runtime_error';vehicle='unknown'
        if summary is not None:
            result=semantic.outcome(summary);state_name='SUCCESS' if result['success'] else 'FAILED';reason=result['text']
            final=(summary.get('finalizer') or {})
            vehicle=('disarmed on '+str(summary.get('landed_on'))) if final.get('disarm_triggered') else \
                    'held by a contact' if summary.get('landed') or summary.get('hit') else 'hovering'
        agreed=bool(steps) and len(steps)<=len(watch.inputs) and all(step['input_sha256']==seen['input_sha256'] for step,seen in zip(steps,watch.inputs))
        record({**entry,'state':state_name,'reason':reason,'success':bool(result and result['success']),'outcome':result,'summary':summary,
                'error':error,'steps':steps,'model_inputs':list(watch.inputs),
                # The hashes the canonical loop logged for what it gave the policy, against those of what the display was given.
                'input_hash_agreement':agreed if steps else None,
                'camera_frames_unchanged':all(item['camera_frames_unchanged'] for item in watch.inputs) if watch.inputs else None,
                'mean_decision_s':summary.get('mean_decision_s') if summary else None})
        print(f'MISSION {mission_id} {target["object_id"]} {task} -> {state_name}: {reason}',flush=True)
        if back:
            # R was the request: the aborted flight is on record, and the scene goes back to the launch pose with its target kept.
            await prepare(target);state_name='TARGET_SELECTED';reason=None
            show(state=env.state(),control_message=f'Mission {mission_id} aborted; back at the launch pose');return
        show('Mission '+state_name+' - R returns to the launch pose',state=env.state(),input_verified=agreed if steps else None,
             control_message=error.strip().splitlines()[-1] if error else None)

    try:
        show('Connecting to the simulator',checkpoint_check=check)
        client=ProjectAirSimClient(address=args.host);client.connect();client.socket_services.recv_timeout=60000
        env=MissionEnv(client,config,saved,OUT,watch)
        loader=Loader(checkpoint,saved)
        await prepare()
        overview=OverviewScene(env.world,env.drone,OUT)
        # Captures go through the writer thread, so one taken in flight costs the loop the request and nothing more.
        overview.publish_image=lambda path,image:feed.image(Path(path).name,image)
        overview.publish_json=lambda path,data:feed.json(Path(path).name,data)
        watch.overview=overview
        while True:
            control=feed.control
            if control['quit']:raise DemoQuit('Exit requested')
            if loader.error:raise RuntimeError('The checkpoint could not be loaded:\n'+loader.error)
            if loader.info and policy_info['loaded'] is None:
                policy_info['loaded']=loader.info;print('MODEL '+json.dumps(loader.info),flush=True)
            state=env.state();watch.refresh_map(state,1.)
            for request_id,request in pending_requests(control,watch.last_request):
                watch.last_request=request_id;message=None;action=request['action']
                try:
                    if action!='start' and state_name not in ('IDLE','TARGET_SELECTED'):
                        # Any choice for the next mission puts the finished one's result and its panels away.
                        state_name='TARGET_SELECTED' if target else 'IDLE';result=None;reason=None;watch.reset_flight()
                    if action=='task':task=TASKS[(TASKS.index(task)+1)%len(TASKS)]
                    elif action in ('select','landmark'):
                        if action=='landmark':index=(index+1)%len(targets);chosen=targets[index]
                        else:
                            point=overview.point_at(request['frame_id'],request['pixel'])
                            chosen=semantic.target_at(targets,point,settings['click_margin_m']) if point is not None else None
                        if chosen is None:
                            # Bare ground or a building: nothing a sentence of this policy can name.
                            target=None;state_name='IDLE';result=None;reason=None
                            raise ValueError(semantic.NOT_SEMANTIC)
                        target=chosen;index=targets.index(chosen);state_name='TARGET_SELECTED';result=None;reason=None
                    elif action=='launch':
                        launch=launches[(launches.index(launch)+1)%len(launches)]
                        await prepare(target);state_name='TARGET_SELECTED' if target else 'IDLE';result=None;reason=None
                    elif action=='reset':
                        if ready is None:
                            # After a flight: the scene and the vehicle go back to the launch pose; the target stays for the next run.
                            await prepare(target)
                        else:target=None
                        state_name='TARGET_SELECTED' if target else 'IDLE';result=None;reason=None
                    elif action=='start':
                        semantic.check_start(target,task)
                        while loader.model is None and not loader.error:
                            show('Loading the checkpoint (first mission waits for it)');await asyncio.sleep(.3)
                            if feed.control['quit']:raise DemoQuit('Exit requested')
                        if loader.error:raise RuntimeError('The checkpoint could not be loaded:\n'+loader.error)
                        policy_info['loaded']=loader.info
                        await fly()
                        # Keys pressed during the flight were answered there.
                        watch.update(mission_request_ack=watch.last_request);break
                except ValueError as error:message=str(error)
                show(state=env.state(),mission_request_ack=request_id,control_message=message)
            now=time.monotonic()
            if vehicle.startswith('hovering') and now-last_hover>1.:
                # Hover is the simulator's own pause; it knows nothing of a target.
                await asyncio.wait_for(await env.drone.hover_async(),timeout=10);last_hover=now
            if state_name in ('IDLE','TARGET_SELECTED') and now-last_preview>1.:
                frames,visibility=env.preview(target['position'] if target else None)
                if frames:
                    preview_counter+=1;slot=preview_counter%2;files={name:f'preview_{slot}_{name}.png' for name in frames}
                    for name,frame in frames.items():feed.image(files[name],frame)
                    watch.telemetry.update(preview_files=files,preview_hashes={name:sha256(frame) for name,frame in frames.items()},preview_visibility=visibility)
                last_preview=now
            waiting={'IDLE':'Click a semantic object or press N','TARGET_SELECTED':'T chooses LAND / APPROACH; G starts'}.get(state_name)
            show(waiting,state=env.state())
            await asyncio.sleep(.1)
    except DemoQuit:
        program_status='STOPPED'
    except Exception:
        program_status='FAIL';watch.telemetry['error']=traceback.format_exc();print(watch.telemetry['error'],flush=True)
    finally:
        cleanup={'vehicle':vehicle,'simulator_landing_routine':'not used'}
        if env is not None and env.drone is not None and program_status!='FAIL' and not vehicle.startswith('disarmed'):
            # Left hovering (or standing where a contact holds it); the launcher closes the simulator. No landing routine is called.
            try:
                await asyncio.wait_for(await env.drone.hover_async(),timeout=10);cleanup['hover']=True
            except Exception as error:cleanup['hover_error']=repr(error)
        if client:
            try:client.disconnect()
            except Exception:pass
        (folder/'session.json').write_text(json.dumps({'status':program_status,'missions':len(history),'cleanup':cleanup,'policy':policy_info,
                                                       'run_token':args.run_token}),encoding='utf-8')
        watch.update(status=program_status,phase='Finished',cleanup=cleanup,mission=snapshot())
        feed.close()
    if program_status=='FAIL':raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True)
    parser.add_argument('--mission-demo',action='store_true');parser.add_argument('--checkpoint',help='AeroVLA-OFT checkpoint directory; default: the frozen baseline')
    parser.add_argument('--steps',type=int,default=0,help='decisions per mission; default: the evaluation budget')
    parser.add_argument('--demo',help='interactive map file; default configs/mission/grounding_film_demo.json');parser.add_argument('--run-token')
    arguments=parser.parse_args()
    if not 0<=arguments.steps<=400:parser.error('--steps within 1..400')
    asyncio.run(main(arguments))
