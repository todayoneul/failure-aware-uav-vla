"""Gaussian blur robustness characterization: fly the canonical starts clean and at three blur severities.

  blur_characterization.py subset                 write configs/failures/gaussian_blur_repeat_subset.json once (no flight is looked at)
  blur_characterization.py order [--phase P]      print the order of the flights and how the conditions are balanced
  blur_characterization.py fly --host H [--phase main|repeat] [--output DIR] [--limit N] [--starts FILE]
                                                  fly what is not flown yet (WSL, simulator running)

A flight is the canonical evaluation's own: `SearchEnv.reset` and `run_episode` of scripts/visual_search.py, with the
frozen checkpoint, the finalizer and the evaluator as they are. The one thing added is the failure between the camera and
the policy: `BlurEnv.observe` takes the canonical observation (frames, and the simulator's ground truth computed from the
unblurred scene) and replaces the two frames by their blurred versions. In the clean condition it hands on the very same
arrays. The policy is called as always, with two frames and a sentence; it is told nothing about the condition.

Per condition a folder in the shape of an evaluation run (results.json, steps.jsonl) plus, per decision, the hash of the raw
frames, sharpness numbers and timings (extra.jsonl), and frames for figures. flights.jsonl lists every attempt, also the
ones that ended in a harness error and were flown again. Exit 0: everything asked for is flown; 3: some flights remain;
4: the simulator stopped answering (start it again and run this again); anything else: the experiment must not go on.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.failures import blur_experiment as blur

CONNECTION_WORDS=('Connection','closed','NNG','Timed out','timed out')


def starts_of(experiment,path=None):
    return json.loads((ROOT/(path or experiment['starts'])).read_text(encoding='utf-8'))['episodes']


def subset_file(experiment):return ROOT/experiment['repeat']['subset']


def phase_ids(experiment,phase,episodes):
    if phase=='main':return [episode['id'] for episode in episodes]
    return [item['id'] for item in json.loads(subset_file(experiment).read_text(encoding='utf-8'))['starts']]


def write_subset(experiment):
    from src.visual_search.maps import load_map
    path=subset_file(experiment)
    if path.exists():raise SystemExit(f'{path} exists; the repeat subset is fixed and is not chosen again')
    episodes=starts_of(experiment);scenes={name:load_map(name) for name in {episode['map'] for episode in episodes}}
    chosen=blur.repeat_subset(experiment,episodes,scenes)
    path.write_text(json.dumps({'description':'The twelve canonical test starts flown a second time in every condition (configs/failures/gaussian_blur_characterization.json, '
                                              '`repeat`). Chosen from the plan of the starts before any flight of the experiment; a start may be of several kinds, `chosen_as` '
                                              'is the kind it was picked for.','rule':experiment['repeat']['rule'],'composition':experiment['repeat']['composition'],
                                'selection_seed':experiment['repeat']['selection_seed'],'starts':chosen},indent=1)+'\n',encoding='utf-8',newline='\n')
    for item in chosen:print(item['chosen_as'],item['id'],item['band'],item['task'],'beside',item['beside'],'first view',item['first_view'])


def digest(frame):
    import hashlib
    return hashlib.sha256(frame.tobytes()).hexdigest()


def make_environment(canonical):
    """The canonical environment with the failure between the camera and the policy."""
    class BlurEnv(canonical.SearchEnv):
        injector=None;trace=()

        def observe(self):
            # The canonical observation first: the frames, and everything the teacher and the evaluator read of the scene.
            observation=super().observe();raw=observation['frames']
            started=time.perf_counter();used=self.injector.apply(raw);finished=time.perf_counter()
            self.trace.append({'raw':raw,'used':used,'blur_ms':(finished-started)*1000,'clock':started,'simulator_ns':observation['stamp']})
            return observation if used is raw else dict(observation,frames=used)
    return BlurEnv


class RecordedPolicy:
    """The policy as the evaluation calls it: two frames, a sentence, and the motion vector of a checkpoint that uses one.
    What it was handed is kept, so that it can be compared with the log afterwards."""
    def __init__(self,model):self.model=model;self.seen=[]

    def infer(self,front,down,instruction,proprio=None):
        self.seen.append((front,down,instruction,proprio))
        return self.model.infer(front,down,instruction,proprio)


def examine(injector,trace,seen,steps,episode,uses_proprio):
    """After a flight: what the log, the policy's own record and the camera's frames say about every decision's input.

    Returns the per-decision records and a summary of the checks. A harness that handed the policy anything but the logged
    arrays is an error, not a result."""
    if not len(trace)==len(seen)==len(steps):raise RuntimeError(f'{len(trace)} observations, {len(seen)} policy calls, {len(steps)} logged decisions')
    rows=[];equal={camera:0 for camera in blur.CAMERAS}
    for index,(item,(front,down,sentence,proprio),step) in enumerate(zip(trace,seen,steps)):
        given={'front':front,'down':down}
        if any(given[camera] is not item['used'][camera] for camera in blur.CAMERAS):raise RuntimeError('the policy was not given the arrays the failure produced')
        if sentence!=episode['instruction'] or (proprio is not None)!=uses_proprio:raise RuntimeError('the policy was given something other than the start\'s sentence')
        hashes={camera:digest(given[camera]) for camera in blur.CAMERAS}
        if hashes!=step['input_sha256']:raise RuntimeError('the evaluation loop logged another input than the policy received')
        raw={camera:digest(item['raw'][camera]) for camera in blur.CAMERAS}
        if injector.failure is None and any(item['used'][camera] is not item['raw'][camera] for camera in blur.CAMERAS):raise RuntimeError('clean: the camera frame was not handed on as it is')
        for camera in blur.CAMERAS:equal[camera]+=raw[camera]==hashes[camera]
        rows.append({'episode':episode['id'],'condition':injector.name,'step':step['step'],'raw_sha256':raw,'input_sha256':hashes,
                     'input_is_raw':{camera:raw[camera]==hashes[camera] for camera in blur.CAMERAS},'blur_ms':item['blur_ms'],
                     'cycle_s':trace[index+1]['clock']-item['clock'] if index+1<len(trace) else None,'inference_ms':step.get('inference_ms'),
                     'quality':{'raw':{camera:blur.quality(item['raw'][camera]) for camera in blur.CAMERAS},
                                'input':{camera:blur.quality(given[camera]) for camera in blur.CAMERAS}}})
    # Did the simulator keep real time over the flight? (Its clock follows the wall clock when it is working.)
    wall=trace[-1]['clock']-trace[0]['clock'];simulated=(trace[-1]['simulator_ns']-trace[0]['simulator_ns'])/1e9
    return rows,{'decisions':len(steps),'policy_received_the_logged_arrays':True,'sentence':episode['instruction'],
                 'decisions_with_input_identical_to_raw':equal,'kernel':injector.kernel,'sigma':injector.sigma,
                 'wall_s':wall,'simulator_s':simulated,'real_time_factor':simulated/wall if wall>0 else None}


def save_frames(folder,trace,every,quality,blurred):
    """Frames for figures: raw Front | raw Down, and in a blurred condition | given Front | given Down."""
    import cv2
    import numpy as np
    folder.mkdir(parents=True,exist_ok=True)
    for index in range(0,len(trace),every):
        item=trace[index];parts=[item['raw']['front'],item['raw']['down']]+([item['used']['front'],item['used']['down']] if blurred else [])
        cv2.imwrite(str(folder/f'{index:04d}.jpg'),np.hstack(parts),[cv2.IMWRITE_JPEG_QUALITY,quality])


def condition_folder(root,phase,condition):return root/condition if phase=='main' else root/'repeat'/condition


def flown(folder):
    path=folder/'results.json'
    return json.loads(path.read_text(encoding='utf-8'))['episodes'] if path.exists() else []


async def fly(args):
    import scripts.visual_search as canonical
    from projectairsim import ProjectAirSimClient
    from src.mission.oft_runner import loaded_modules
    from src.visual_search.episodes import load_config
    config=load_config();experiment=blur.load(args.config);root=Path(args.output);root=root if root.is_absolute() else ROOT/root;root.mkdir(parents=True,exist_ok=True)
    episodes=starts_of(experiment,args.starts);by_id={episode['id']:episode for episode in episodes}
    flights=blur.flight_order(experiment,phase_ids(experiment,args.phase,episodes),args.phase)
    if args.limit:
        kept=list(dict.fromkeys(start for start,_ in flights))[:args.limit];flights=[flight for flight in flights if flight[0] in kept]
    done={name:flown(condition_folder(root,args.phase,name)) for name in experiment['conditions']}
    finished={(episode['id'],name) for name,items in done.items() for episode in items}
    log=root/('flights.jsonl' if args.phase=='main' else 'repeat/flights.jsonl');log.parent.mkdir(parents=True,exist_ok=True)
    attempts={}
    if log.exists():
        for line in log.read_text(encoding='utf-8').splitlines():
            entry=json.loads(line);key=(entry['start'],entry['condition']);attempts[key]=attempts.get(key,0)+1
    limit=experiment['runtime']['attempts_per_flight']
    pending=[flight for flight in flights if flight not in finished and attempts.get(flight,0)<limit]
    given_up=[flight for flight in flights if flight not in finished and attempts.get(flight,0)>=limit]
    print(f'PLAN {args.phase}: {len(flights)} flights, {len(flights)-len(pending)-len(given_up)} flown, {len(pending)} to fly, {len(given_up)} given up after {limit} attempts',flush=True)
    if not pending:raise SystemExit(0)
    checkpoint=ROOT/experiment['checkpoint'];saved=json.loads((checkpoint/'manifest.json').read_text(encoding='utf-8'))['config']
    from src.aerovla_oft.model import AeroVLAOFT
    model=AeroVLAOFT(canonical.MANIFEST,saved,checkpoint=str(checkpoint));info=loaded_modules(model,saved);print('MODEL '+json.dumps(info),flush=True)
    # One pass on blank frames: whichever flight comes first after loading would otherwise carry the model's slow first pass.
    import numpy as np
    blank=np.zeros((256,256,3),dtype=np.uint8);model.infer(blank,blank,'Warm up.',[0.]*len(saved['proprio']['fields']) if saved['proprio']['enabled'] else None)
    client=ProjectAirSimClient(address=args.host,port_topics=args.ports[0],port_services=args.ports[1]);client.connect();client.socket_services.recv_timeout=60000
    env=make_environment(canonical)(client,config,saved,root/('sim' if args.phase=='main' else 'repeat/sim'))
    arguments=argparse.Namespace(policy='oft',finalizer=experiment['runtime']['finalizer'],live=False,output=str(root),set='test',model_name=saved.get('name'))
    status=0;uses_proprio=bool(saved['proprio']['enabled'])
    try:
        for index,(start,condition) in enumerate(flights):
            if (start,condition) not in pending:continue
            episode=dict(by_id[start]);folder=condition_folder(root,args.phase,condition);folder.mkdir(parents=True,exist_ok=True)
            injector=blur.Injector(condition);env.injector=injector;env.trace=[];policy=RecordedPolicy(model)
            entry={'phase':args.phase,'index':index,'start':start,'condition':condition,'position_in_start':index%len(experiment['conditions']),
                   'attempt':attempts.get((start,condition),0)+1,'started_epoch':time.time()}
            try:
                await env.reset(episode)
                summary,steps=await canonical.run_episode(env,episode,policy,arguments,None)
            except Exception:
                error=traceback.format_exc();print(error,flush=True)
                entry.update(outcome='runtime_error',error=error.strip().splitlines()[-1],seconds=time.time()-entry['started_epoch'])
                attempts[(start,condition)]=entry['attempt']
                with log.open('a',encoding='utf-8') as file:file.write(json.dumps(entry)+'\n')
                # A simulator that no longer answers ends this process; the launcher starts a new one and the flight is flown again.
                if any(word in error for word in CONNECTION_WORDS):status=4;break
                status=3;continue
            # Outside the retry: a harness that gave the policy anything but what it logged stops the experiment.
            rows,checks=examine(injector,env.trace,policy.seen,steps,episode,uses_proprio)
            factor=checks['real_time_factor']
            if factor is not None and factor<experiment['runtime']['real_time']['minimum_factor']:
                # The simulator did not keep real time: not a flight of the policy. Logged, and flown again.
                entry.update(outcome='runtime_error',error=f'simulator advanced {factor:.2f} of real time',seconds=time.time()-entry['started_epoch'])
                attempts[(start,condition)]=entry['attempt']
                with log.open('a',encoding='utf-8') as file:file.write(json.dumps(entry)+'\n')
                print(f'INVALID {start} {condition}: {entry["error"]}',flush=True);status=3;continue
            summary['policy']='oft';summary['set']='test'
            summary['blur']={'condition':condition,'phase':args.phase,'flight_index':index,'position_in_start':entry['position_in_start'],'attempt':entry['attempt'],**checks}
            done[condition].append(summary)
            with (folder/'steps.jsonl').open('a') as file:
                for step in steps:file.write(json.dumps({'episode':start,**step})+'\n')
            with (folder/'extra.jsonl').open('a') as file:
                for row in rows:file.write(json.dumps(row)+'\n')
            save_frames(folder/'frames'/start,env.trace,experiment['frames']['every_decisions'],experiment['frames']['jpeg_quality'],injector.failure is not None)
            (folder/'results.json').write_text(json.dumps({'policy':'oft','flight':'tick','checkpoint':str(checkpoint),'model_name':saved.get('name'),'plan':args.starts or experiment['starts'],
                                                           'set':'test','layout_override':None,'condition':condition,'blur':experiment['conditions'][condition],
                                                           'config':config,'oft':saved,'episodes':done[condition]},indent=1))
            entry.update(outcome='flown',success=summary['success'],reason=summary['reason'],steps=summary['steps'],real_time_factor=factor,seconds=time.time()-entry['started_epoch'])
            with log.open('a',encoding='utf-8') as file:file.write(json.dumps(entry)+'\n')
            print(f'FLIGHT {index+1}/{len(flights)} {start} {condition} {summary["reason"]} success={summary["success"]} steps={summary["steps"]} '
                  f'final={summary["final_distance_m"]:.1f} selected={summary.get("selected")}',flush=True)
    finally:
        client.disconnect()
    remaining=[flight for flight in flights if flight not in {(episode['id'],name) for name,items in done.items() for episode in items}]
    raise SystemExit(status or (3 if [flight for flight in remaining if attempts.get(flight,0)<limit] else 0))


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True);commands.add_parser('subset')
    order=commands.add_parser('order');order.add_argument('--phase',choices=('main','repeat'),default='main')
    run=commands.add_parser('fly');run.add_argument('--host',required=True);run.add_argument('--ports',type=int,nargs=2,default=[8989,8990])
    run.add_argument('--phase',choices=('main','repeat'),default='main');run.add_argument('--output',default='outputs/failures/gaussian_blur')
    run.add_argument('--limit',type=int,default=0,help='only the first N starts of the phase');run.add_argument('--starts',help='another starts file (for a check of the harness; never mixed into the experiment)')
    parser.add_argument('--config');args=parser.parse_args();experiment=blur.load(args.config)
    if args.command=='subset':write_subset(experiment);return
    if args.command=='order':
        episodes=starts_of(experiment);flights=blur.flight_order(experiment,phase_ids(experiment,args.phase,episodes),args.phase)
        print(json.dumps({'phase':args.phase,'flights':len(flights),'position_of_each_condition':blur.positions(flights,list(experiment['conditions'])),'order':flights},indent=1));return
    asyncio.run(fly(args))


if __name__=='__main__':main()
