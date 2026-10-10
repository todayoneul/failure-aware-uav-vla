"""Fly a plan of starts with landing surfaces read as surfaces (WSL, simulator running).

  surface_flights.py --host H --plan FILE --output DIR                    controlled flights: each episode's own scripted pilot
  surface_flights.py --host H --plan FILE --output DIR --policy oft --checkpoint DIR [--limit N]
                                                                          the same starts flown by an AeroVLA-OFT checkpoint

A flight is the canonical evaluation's own: `SearchEnv.reset` and `run_episode` of scripts/visual_search.py, from the
episode's start fields (start_xy, start_yaw_deg, start_height_m), with the finalizer as it is. Two things are added around
it and nothing inside: a contact with any landing surface is read the way a contact with a pad is (src/landing/env.py),
and the finished flight is read by the surface evaluator (src/landing/evaluator.py) beside the canonical summary.

A controlled plan (configs/experiments/controlled_landings.json) gives every episode a scripted pilot (src/landing/controlled.py)
and what the evaluator has to say about the flight, written before it is flown; the result of each such check is PASS or
FAIL. A controlled flight says nothing about any policy. A plan of starts (scripts/plan_arbitrary_starts.py) flown with
--policy oft records what the checkpoint did from each start.

results.json holds one record per episode: the start asked for and the start reached, the canonical summary, the surface
reading and the checks. steps.jsonl holds every decision. Exit 0: every episode flown and every check passed; 1: a check
failed; 4: the simulator stopped answering (start it again and run this again: what is flown is kept).
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

CONNECTION_WORDS=('Connection','closed','NNG','Timed out','timed out')


def load_plan(path,config):
    """The episodes of a plan in the form the canonical environment flies. A controlled plan names its map once and leaves
    the sentence to the evaluation's own templates; a planned-starts file has every field already."""
    from src.mission import semantic
    from src.visual_search.maps import load_map
    plan=json.loads(Path(path).read_text(encoding='utf-8'));episodes=[];maps={}
    for item in plan['episodes']:
        episode=dict(item);name=episode.get('map') or plan['map'];absolute=str(name if Path(name).is_absolute() else ROOT/name)
        if absolute not in maps:maps[absolute]=load_map(absolute)
        map_config=maps[absolute];layout=episode.get('layout') or plan.get('layout') or map_config['mission']['layout']
        episode.update(map=absolute,layout=layout,kind=episode.get('kind','controlled'),strategy=episode.get('strategy',map_config.get('mission',{}).get('strategy','right')))
        if 'instruction' not in episode:
            target={item['object_id']:item for item in semantic.semantic_targets(map_config,layout)}[episode['target']]
            # The sentence of a landing that would be refused is never built: the episode is not flown.
            episode['instruction']=None if semantic.refusal(target,episode['task']) else semantic.sentence(config,map_config,target,episode['task'])
        episode.setdefault('start_height_m',config['cruise_height_m'])
        episodes.append(episode)
    return plan,episodes,maps


def start_checks(episode,env,tolerance):
    """Was the start that was asked for the start that was flown from?"""
    from src.mission import start as starts
    wanted=starts.StartState.from_episode(episode,env.config['cruise_height_m']);error=starts.reproduction(wanted,env.reached)
    rows=[('spawn_on_ground',True,env.spawn['on_ground'],env.spawn['on_ground']),
          ('start_xy_error_m',f'<= {tolerance["xy_m"]}',round(error['xy_error_m'],4),error['xy_error_m']<=tolerance['xy_m']),
          ('start_yaw_error_deg',f'<= {tolerance["yaw_deg"]}',round(error['yaw_error_deg'],4),error['yaw_error_deg']<=tolerance['yaw_deg']),
          ('start_height_error_m',f'within {tolerance["height_m"]}',round(error['height_error_m'],4),abs(error['height_error_m'])<=tolerance['height_m'])]
    return wanted,rows


async def fly(args):
    from projectairsim import ProjectAirSimClient
    import scripts.visual_search as canonical
    from src.integration.projectairsim_observation_adapter import adapt_state
    from src.landing import evaluator
    from src.landing.controlled import ControlledPilot,aim_point,compare
    from src.landing.env import SurfaceContacts
    from src.landing.surface import SurfaceSet,refusal
    from src.mission import start as starts
    from src.visual_search.episodes import load_config

    class FlightEnv(SurfaceContacts,canonical.SearchEnv):
        """The canonical environment with landing surfaces; it also keeps the vehicle's last true state for a scripted pilot."""
        async def reset(self,episode):
            self.last_state=None
            await super().reset(episode)
            self.reached=starts.reached(adapt_state(self.drone.get_ground_truth_kinematics()),self.ground_z)

        def observe(self):
            observation=super().observe();self.last_state=observation['state'];return observation

    config=load_config();plan,episodes,maps=load_plan(args.plan,config)
    if args.only:episodes=[episode for episode in episodes if episode['id'] in args.only]
    if args.limit:episodes=episodes[:args.limit]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);results_path=output/'results.json'
    done=json.loads(results_path.read_text(encoding='utf-8'))['episodes'] if results_path.exists() and not args.again else []
    finished={item['id'] for item in done};tolerance=plan.get('start_tolerance') or {'xy_m':.05,'yaw_deg':.5,'height_m':.15}
    model=None;oft=json.loads((ROOT/'configs/aerovla_oft.json').read_text(encoding='utf-8'))
    if args.policy=='oft':
        from src.aerovla_oft.model import AeroVLAOFT
        oft=json.loads((Path(args.checkpoint)/'manifest.json').read_text(encoding='utf-8'))['config']
        model=AeroVLAOFT(canonical.MANIFEST,oft,checkpoint=args.checkpoint)

    def save():
        results_path.write_text(json.dumps({'plan':str(args.plan),'policy':args.policy,'checkpoint':args.checkpoint,'evaluator':evaluator.VERSION,
                                            'start_tolerance':tolerance,'episodes':done},indent=1),encoding='utf-8')

    client=None;env=None;status=0
    try:
        for episode in episodes:
            if episode['id'] in finished:continue
            surfaces=SurfaceSet(maps[episode['map']],episode['layout']);record={'id':episode['id'],'episode':episode,'policy':args.policy,'started_epoch':time.time()}
            reason=refusal(surfaces[episode['target']]) if episode['task']=='land' else None
            if reason or episode.get('expect_refusal'):
                # A landing on an object without a landing surface is refused before anything flies.
                rows=[('refused',bool(episode.get('expect_refusal')),bool(reason),bool(reason)==bool(episode.get('expect_refusal')))]
                record.update(flown=False,refusal=reason,checks=rows,passed=all(row[3] for row in rows))
                done.append(record);save();print(f'EPISODE {episode["id"]} not flown: {reason} -> {"PASS" if record["passed"] else "FAIL"}',flush=True)
                continue
            if client is None:
                client=ProjectAirSimClient(address=args.host,port_topics=args.ports[0],port_services=args.ports[1]);client.connect()
                client.socket_services.recv_timeout=60000;env=FlightEnv(client,config,oft,output)
            try:
                await env.reset(episode)
                wanted,rows=start_checks(episode,env,tolerance)
                if args.policy=='oft':policy=model
                else:
                    aim,top=aim_point(episode['pilot'],env.surfaces);policy=ControlledPilot(env,oft,env.landing,aim,top,episode['pilot']['mode'])
                arguments=argparse.Namespace(policy='oft',finalizer='on',live=False,output=str(output),set=None,model_name=args.policy)
                summary,steps=await canonical.run_episode(env,episode,policy,arguments,None)
            except Exception:
                error=traceback.format_exc();print(error,flush=True)
                # A flight the harness lost is not a result: it is reported, kept out of results.json and flown again next time.
                with (output/'harness_errors.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps({'id':episode['id'],'epoch':time.time(),'error':error})+'\n')
                status=4 if any(word in error for word in CONNECTION_WORDS) else 1
                if status==4:break
                continue
            reading=evaluator.read(summary,env.surfaces,env.landing,steps)
            if args.policy!='oft':rows+=compare({**reading,'reason':summary['reason'],'stopped':summary['stopped']},episode.get('expect',{}))
            record.update(flown=True,start=wanted.to_dict(),start_reached=env.reached.to_dict(),spawn=env.spawn,summary=summary,surface_landing=reading,
                          words=evaluator.words(reading,summary) if episode['task']=='land' else None,
                          pilot_phases=getattr(policy,'phases',None),checks=rows,passed=all(row[3] for row in rows))
            done.append(record);save()
            with (output/'steps.jsonl').open('a',encoding='utf-8') as file:
                for step in steps:file.write(json.dumps({'episode':episode['id'],**step})+'\n')
            failed=[row[0] for row in rows if not row[3]]
            print(f'EPISODE {episode["id"]} {summary["reason"]} success={reading["success"]} landed_on={reading["landed_on"]} collision={reading["collision"]} '
                  f'steps={summary["steps"]} -> {"PASS" if record["passed"] else "FAIL "+", ".join(failed)}',flush=True)
    finally:
        if client is not None:
            try:client.disconnect()
            except Exception:pass
    if status:raise SystemExit(status)
    wanted={episode['id'] for episode in episodes};checked=[item for item in done if item['id'] in wanted]
    raise SystemExit(0 if len(checked)==len(wanted) and all(item['passed'] for item in checked) else 1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True);parser.add_argument('--ports',type=int,nargs=2,default=[8989,8990])
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--policy',choices=('controlled','oft'),default='controlled');parser.add_argument('--checkpoint')
    parser.add_argument('--only',nargs='+',help='episode ids to fly');parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--again',action='store_true',help='fly everything again instead of keeping what results.json already holds')
    arguments=parser.parse_args()
    if arguments.policy=='oft' and not arguments.checkpoint:parser.error('--checkpoint is required for --policy oft')
    asyncio.run(fly(arguments))
