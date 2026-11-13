"""Interactive Mission Control flown by an AeroVLA-OFT checkpoint: the scene, the sentence, what reaches the policy, the
landing, the keys and the canvas. The flights here are the canonical episode loop over a fake simulator; no model is loaded."""
import argparse
import asyncio
import inspect
import json
import math
import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'outputs/generalization/grounding_film_frozen.json'


def demo():
    from src.mission import semantic
    from src.visual_search.episodes import load_config
    config=load_config();scene=semantic.load_demo();return config,scene,semantic.semantic_targets(scene)


class DemoSceneTests(unittest.TestCase):
    def test_the_targets_are_the_catalogues_objects_with_what_a_mission_needs_to_know(self):
        config,scene,targets=demo();catalogue=json.loads((ROOT/'configs/targets/objects.json').read_text(encoding='utf-8'))['objects']
        names=[item['object_id'] for item in targets]
        self.assertEqual(names,['blue_pad','red_pad','blue_cube','red_cube','blue_cone','green_cylinder','orange_ball'])
        for item in targets:
            for key in ('object_id','noun','color','shape','position','landable'):self.assertIsNotNone(item[key],(item['object_id'],key))
            self.assertEqual(item['noun'],catalogue[item['object_id']]['noun'])
            self.assertEqual((item['color'],item['shape']),tuple(catalogue[item['object_id']]['attributes'].values()))
        # What may be landed on is what has a landing surface: the pads, the cubes and the cylinder; not the cone or the ball.
        self.assertEqual([item['object_id'] for item in targets if item['landable']],['blue_pad','red_pad','blue_cube','red_cube','green_cylinder'])
        self.assertEqual({item['object_id']:item['surface']['surface'] for item in targets},
                         {'blue_pad':'rectangle','red_pad':'rectangle','blue_cube':'rectangle','red_cube':'rectangle','blue_cone':'none','green_cylinder':'circle','orange_ball':'none'})
        # Objects added to the map are the catalogue's own: same size, same colour, same shape.
        for name,spec in scene['objects'].items():
            if 'native' not in spec:self.assertEqual(spec,catalogue[name])
        self.assertEqual({name for name,spec in scene['objects'].items() if 'native' in spec and name in scene['layouts']['mission']},{'blue_cone','orange_ball'})

    def test_nothing_is_added_to_the_map_but_objects_on_open_ground(self):
        from src.visual_search.maps import MapGeometry,scene_objects
        config,scene,targets=demo();geometry=MapGeometry(scene,'mission')
        from src.visual_search.maps import structures
        # (The one structure of the map file is the plate on the orange ball, and it belongs to the layout `mission_cap` alone.)
        self.assertEqual(structures(scene,'mission'),[]);self.assertEqual([item['name'] for item in structures(scene,'mission_cap')],['OrangeBallCap'])
        self.assertIsNone(scene['lighting'])
        blocks=[box for box in geometry.boxes if box['name'] not in geometry.objects]
        for item in targets:
            if item['native']:continue
            for box in blocks+[other for other in geometry.boxes if other['name'] in geometry.objects and other['name']!=item['object_id']]:
                gap=max(box['min'][0]-item['maximum'][0],item['minimum'][0]-box['max'][0],box['min'][1]-item['maximum'][1],item['minimum'][1]-box['max'][1])
                self.assertGreaterEqual(gap,5.,(item['object_id'],box['name']))
        # The whole map file is read by the evaluation's own loader, and what is spawned is spawned as it is there.
        spawned={entry['name'] for entry in scene_objects(scene,'mission')}
        self.assertTrue({'BluePad','RedPad','BlueCube','RedCube','GreenCylinder'}<=spawned)
        self.assertFalse({'Cone_5','OrangeBall'}&spawned)

    def test_every_launch_stands_on_open_ground_and_the_core_targets_are_in_the_validated_range(self):
        from src.mission.semantic import launch_view
        from src.visual_search.maps import MapGeometry
        config,scene,targets=demo();geometry=MapGeometry(scene,'mission');settings=scene['mission'];limit=config['canonical']['max_start_m']
        for launch in settings['launches']:
            # Clear of every object as an evaluation start is, and far enough from the blocks that turning on the spot
            # never faces a wall within the distance at which the teacher (and the policy after it) climbs.
            self.assertGreaterEqual(geometry.nearest(*launch['xy'],0.)[0],config['generalization']['start_margin_m'])
            blocks=[box for box in geometry.boxes if box['name'] not in geometry.objects and box['top_m']>config['cruise_height_m']]
            self.assertGreater(min(geometry._gap(box,*launch['xy']) for box in blocks),config['generalization']['teacher']['blocked_m'])
            self.assertTrue(geometry.inside_area(*launch['xy']))
        reach={item['object_id']:min(launch_view(launch,item)['distance_m'] for launch in settings['launches']
                                     if geometry.sees(*launch['xy'],config['cruise_height_m'],item['object_id'])) for item in targets}
        for name in ('blue_pad','red_pad','blue_cube','red_cube','blue_cone','green_cylinder'):
            self.assertLessEqual(reach[name],limit,name);self.assertGreaterEqual(reach[name],12.,name)
        north=settings['launches'][0];views={item['object_id']:launch_view(north,item) for item in targets}
        # The default launch shows both pads at once, each with the cube of its own colour on its outer side.
        self.assertLess(abs(views['blue_pad']['bearing_deg']),45);self.assertLess(abs(views['red_pad']['bearing_deg']),45)
        self.assertGreater(views['blue_cube']['bearing_deg'],views['blue_pad']['bearing_deg'])
        self.assertLess(views['red_cube']['bearing_deg'],views['red_pad']['bearing_deg'])

    def test_the_demonstration_adds_nothing_to_the_frozen_folders(self):
        self.assertFalse(list((ROOT/'configs/maps').glob('*mission*'))+list((ROOT/'configs/maps').glob('*demo*')))
        self.assertTrue((ROOT/'configs/mission/grounding_film_demo.json').exists())


class SentenceTests(unittest.TestCase):
    def test_the_sentences_are_the_evaluations_own(self):
        from src.mission.semantic import sentence
        config,scene,targets=demo();by={item['object_id']:item for item in targets}
        self.assertEqual(sentence(config,scene,by['blue_pad'],'land'),'Find the blue landing pad and land on it.')
        self.assertEqual(sentence(config,scene,by['blue_pad'],'approach'),'Approach the blue landing pad.')
        self.assertEqual(sentence(config,scene,by['red_pad'],'land'),'Find the red landing pad and land on it.')
        self.assertEqual(sentence(config,scene,by['blue_cube'],'approach'),'Approach the blue cube.')
        self.assertEqual(sentence(config,scene,by['blue_cone'],'approach'),'Approach the blue cone.')
        # Built from the templates the canonical test was flown with, not from words kept here.
        test=json.loads((ROOT/'configs/gen_v3_canonical_44m_test.json').read_text(encoding='utf-8'))
        flown={episode['instruction'] for group in test.values() if isinstance(group,list) for episode in group if isinstance(episode,dict) and 'instruction' in episode}
        if flown:self.assertIn('Find the blue landing pad and land on it.',flown)
        self.assertEqual(scene['mission']['instructions'],{'land':'land','approach':'approach'})

    def test_no_sentence_carries_a_direction_a_distance_or_a_coordinate(self):
        from src.mission.semantic import sentence,TASKS
        from src.visual_search.episodes import DIRECTION_WORDS
        from src.visual_search.finalizer import landing_requested
        config,scene,targets=demo()
        for item in targets:
            for task in TASKS:
                if task=='land' and not item['landable']:continue
                text=sentence(config,scene,item,task)
                self.assertFalse(re.search(r'\d',text),text);self.assertFalse(any(word in text for word in DIRECTION_WORDS),text)
                # The sentence alone says whether the finalizer is armed.
                self.assertEqual(landing_requested(text),task=='land',text)

    def test_a_landing_on_something_without_a_landing_surface_and_a_target_that_is_not_an_object_are_refused(self):
        from src.mission import semantic
        config,scene,targets=demo();by={item['object_id']:item for item in targets}
        for name in ('orange_ball','blue_cone'):
            self.assertEqual(semantic.refusal(by[name],'land'),'Selected object has no valid landing surface. Use APPROACH.')
            with self.assertRaisesRegex(ValueError,'no valid landing surface'):semantic.check_start(by[name],'land')
            self.assertIsNone(semantic.refusal(by[name],'approach'))
        for name in ('blue_pad','blue_cube','green_cylinder'):self.assertIsNone(semantic.refusal(by[name],'land'))
        self.assertEqual(semantic.sentence(config,scene,by['blue_cube'],'land'),'Find the blue cube and land on it.')
        self.assertEqual(semantic.sentence(config,scene,by['green_cylinder'],'land'),'Find the green cylinder and land on it.')
        self.assertEqual(semantic.sentence(config,scene,by['orange_ball'],'approach'),'Approach the orange ball.')
        self.assertEqual(semantic.refusal(None,'land'),'grounding_film requires a semantic object target. Select a configured landmark/object.')
        with self.assertRaisesRegex(ValueError,'semantic object target'):semantic.check_start(None,'approach')

    def test_a_click_selects_an_object_only_inside_its_footprint(self):
        from src.mission.semantic import target_at
        config,scene,targets=demo()
        self.assertEqual(target_at(targets,[12.,90.,-3.5])['object_id'],'blue_pad')
        self.assertEqual(target_at(targets,[91.,-36.,-9.])['object_id'],'blue_cone')
        self.assertIsNone(target_at(targets,[30.,70.,-1.]));self.assertIsNone(target_at(targets,[85.,0.,-16.]))


class ControlTests(unittest.TestCase):
    def test_the_task_key_the_launch_key_and_the_frozen_prompt_mode(self):
        from src.mission.control import default_mission_control,mission_key,pending_requests
        state=default_mission_control('grounding-film')
        self.assertEqual((state['prompt_mode'],state['direction_hint'],state['enabled']),('instruction-only',False,False))
        # T asks the session for the other task; the session, not the window, holds which one the next sentence uses.
        self.assertEqual([request['action'] for _,request in pending_requests(mission_key(mission_key(state,ord('t')),ord('T')),0)],['task','task'])
        # M changes nothing: no prompt mode exists that could put a hint or a coordinate in front of the model.
        for _ in range(3):
            state=mission_key(state,ord('m'));self.assertEqual((state['prompt_mode'],state['direction_hint']),('instruction-only',False))
        state=mission_key(mission_key(mission_key(state,ord('n')),ord('l')),ord('g'))
        self.assertEqual([request['action'] for _,request in pending_requests(state,0)],['landmark','launch','start'])
        self.assertEqual(mission_key(state,ord('c'))['main_view'],'chase');self.assertEqual(mission_key(mission_key(state,ord('c')),ord('f'))['main_view'],'map')
        # Blur is off until B is pressed, and no mission key touches it.
        self.assertFalse(state['enabled']);self.assertTrue(mission_key(state,ord('b'))['enabled'])

    def test_the_earlier_mission_control_keeps_its_keys(self):
        from src.mission.control import default_mission_control,mission_key,pending_requests
        state=default_mission_control()
        self.assertNotIn('policy',state);self.assertNotIn('task',state);self.assertEqual(state['prompt_mode'],'hint')
        self.assertEqual(mission_key(state,ord('m'))['prompt_mode'],'description')
        self.assertEqual(mission_key(state,ord('t')),state);self.assertEqual(pending_requests(mission_key(state,ord('l')),0),[])
        self.assertEqual(mission_key(state,ord('c'))['overview_focus'],'drone')


class CheckpointTests(unittest.TestCase):
    def make(self,root,grounding=True):
        path=root/'outputs/aerovla_oft/checkpoints/demo';(path/'oft').mkdir(parents=True)
        config=json.loads((ROOT/'configs/aerovla_oft.json').read_text(encoding='utf-8'))
        if grounding:config['grounding']={'type':'film','position':'before_projector','hidden_dim':512}
        (path/'manifest.json').write_text(json.dumps({'config':config}));(path/'head.pt').write_bytes(b'head')
        (path/'oft/adapter_model.safetensors').write_bytes(b'lora');(path/'oft/adapter_config.json').write_text('{}')
        return path

    def test_a_missing_or_incomplete_checkpoint_stops_the_demo(self):
        from src.mission.checkpoint import inspect as check
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.assertTrue(check('outputs/none',root=root)['problems'])
            path=self.make(root);(path/'head.pt').unlink()
            self.assertIn('head.pt',check(path,root=root)['problems'][0])

    def test_the_frozen_baseline_is_verified_and_any_change_to_it_stops_the_demo(self):
        from src.mission.checkpoint import inspect as check
        import scripts.freeze_baseline as freeze
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=self.make(root);relative='outputs/aerovla_oft/checkpoints/demo'
            (root/'outputs/generalization').mkdir(parents=True)
            with patch.object(freeze,'ROOT',root):files=freeze.collect([relative])
            (root/'outputs/generalization/demo_frozen.json').write_text(json.dumps({'paths':[relative],'files':files}))
            result=check(relative,'demo',root)
            self.assertEqual((result['frozen']['state'],result['grounding'],result['problems'],result['warnings']),('verified','film',[],[]))
            (path/'head.pt').write_bytes(b'other')
            changed=check(relative,'demo',root)
            self.assertEqual(changed['frozen']['state'],'changed');self.assertIn('head.pt',changed['problems'][0])

    def test_a_custom_checkpoint_runs_with_a_clear_warning(self):
        from src.mission.checkpoint import inspect as check
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.make(root,grounding=False);(root/'outputs/generalization').mkdir(parents=True)
            (root/'outputs/generalization/demo_frozen.json').write_text(json.dumps({'paths':['outputs/aerovla_oft/checkpoints/other'],'files':{}}))
            result=check('outputs/aerovla_oft/checkpoints/demo','demo',root)
            self.assertEqual(result['problems'],[]);self.assertEqual(result['frozen']['state'],'custom')
            self.assertTrue(any('custom checkpoint' in text for text in result['warnings']));self.assertTrue(any('no FiLM' in text for text in result['warnings']))

    @unittest.skipUnless((ROOT/'outputs/aerovla_oft/checkpoints/grounding_film/head.pt').exists() and FROZEN.exists(),'the frozen checkpoint is not on this machine')
    def test_this_work_left_the_frozen_baseline_as_it_was(self):
        from src.mission.checkpoint import inspect as check
        result=check('outputs/aerovla_oft/checkpoints/grounding_film','grounding_film')
        self.assertEqual((result['frozen']['state'],result['grounding'],result['problems']),('verified','film',[]))


class FeedTests(unittest.TestCase):
    def test_files_are_written_off_the_loop_and_blur_off_hands_the_same_arrays_on(self):
        from src.failures.control import write_control
        from src.mission.control import default_mission_control,mission_key
        from src.mission.feed import Feed,sha256
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);write_control(output/'control.json',default_mission_control('grounding-film'))
            feed=Feed(output);frame=np.arange(256*256*3,dtype=np.uint32).astype(np.uint8).reshape(256,256,3)
            frames={'front':frame,'down':frame[::-1].copy()}
            used,report=feed.fail(frames)
            self.assertIs(used,frames);self.assertIs(used['front'],frame);self.assertFalse(report['failure_enabled'])
            feed.image('input_a_front.png',frame);feed.json('telemetry.json',{'step':1});feed.json('telemetry.json',{'step':2});feed.append('log.jsonl',{'a':1})
            self.assertTrue(feed.flush(5.))
            self.assertEqual(json.loads((output/'telemetry.json').read_text())['step'],2)
            self.assertEqual(sha256(cv2.imread(str(output/'input_a_front.png'))),sha256(frame))
            # B, read by the same thread: only then is a frame changed, and it is a new array.
            write_control(output/'control.json',mission_key(default_mission_control('grounding-film'),ord('b')))
            for _ in range(100):
                if feed.control['enabled']:break
                import time;time.sleep(.02)
            blurred,report=feed.fail(frames)
            self.assertTrue(report['failure_enabled']);self.assertIsNot(blurred['front'],frame);self.assertNotEqual(sha256(blurred['front']),sha256(frame))
            feed.close();self.assertEqual(feed.errors,[])


class Flight:
    """One mission of the interactive runner's own parts over the fake simulator: MissionEnv, WatchedPolicy, the canonical loop."""
    def __init__(self,directory,blur=False,launch=None,layout='mission'):
        from src.failures.control import write_control
        from src.mission import oft_runner
        from src.mission.control import default_mission_control,mission_key
        from src.mission.feed import Feed
        from tests.fake_airsim import FakeSim,FakeClient
        from src.mission import semantic
        self.runner=oft_runner;self.config,self.scene,self.targets=demo();self.launch=launch
        if layout!='mission':self.scene['mission']['layout']=layout;self.targets=semantic.semantic_targets(self.scene)
        self.by={item['object_id']:item for item in self.targets}
        self.oft=json.loads((ROOT/'configs/aerovla_oft.json').read_text(encoding='utf-8'))
        self.output=Path(directory);control=default_mission_control('grounding-film')
        write_control(self.output/'control.json',mission_key(control,ord('b')) if blur else control)
        self.sim=FakeSim(self.scene,layout);self.feed=Feed(self.output);self.watch=oft_runner.Watch(self.feed,320)
        self.env=oft_runner.MissionEnv(FakeClient(self.sim),self.config,self.oft,self.output,self.watch)

    async def fly(self,target,task,script,told=None):
        from src.mission import semantic
        from tests.fake_airsim import ScriptedModel,world_class,drone_class
        import scripts.visual_search as canonical
        named=self.by[told or target];text=semantic.sentence(self.config,self.scene,named,task);launch=self.scene['mission']['launches'][0]
        launch=self.launch or launch
        episode=semantic.mission_episode(semantic.DEMO,self.scene,named,task,text,launch,1,320,self.config['cruise_height_m'])
        self.model=ScriptedModel(script(self.sim) if callable(script) else script,self.oft);self.text=text;self.finalizers=[]
        async def no_wait(_):return None
        clock=types.SimpleNamespace(sleep=no_wait,wait_for=asyncio.wait_for)
        def listen(finalizer):
            self.finalizers.append(finalizer.state);self.watch.finalizer(finalizer)
        with patch.object(canonical,'World',world_class(self.sim)),patch.object(canonical,'Drone',drone_class(self.sim)), \
             patch.object(canonical,'matched_camera_images',self.sim.camera),patch.object(canonical,'asyncio',clock),patch('builtins.print'):
            await self.env.reset(episode);self.env.semantic=named
            arguments=argparse.Namespace(policy='oft',finalizer='on',live=False,output=str(self.output),set=None,model_name='test')
            with self.runner.watched_finalizer(listen):
                self.summary,self.steps=await canonical.run_episode(self.env,episode,self.runner.WatchedPolicy(self.model,self.watch),arguments,None)
        from src.landing import evaluator
        self.reading=evaluator.read(self.summary,self.env.surfaces,self.env.landing,self.steps)
        self.restored=canonical.LandingFinalizer
        self.feed.close();return self.summary


class CanonicalLoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):asyncio.get_running_loop().slow_callback_duration=60.

    async def test_grounding_film_policy_input_has_no_target_geometry(self):
        from src.mission import oft_runner
        from src.visual_search.episodes import policy_inputs
        from tests.fake_airsim import fly_to
        with tempfile.TemporaryDirectory() as directory:
            flight=Flight(directory);target=flight.by['blue_pad']
            await flight.fly('blue_pad','land',lambda sim:fly_to(sim,target['position'],descend=True))
        self.assertGreater(len(flight.model.calls),40)
        numbers={round(abs(value),1) for value in target['position'][:2]}
        for (arguments,keywords),seen in zip(flight.model.calls,flight.watch.inputs):
            # Two frames, one sentence, and no motion vector for a checkpoint trained without one. Nothing else, by position or by name.
            self.assertEqual(keywords,{});self.assertEqual(len(arguments),4)
            front,down,sentence,proprio=arguments
            for frame in (front,down):self.assertEqual((type(frame),frame.shape,frame.dtype),(np.ndarray,(256,256,3),np.uint8))
            self.assertEqual(sentence,'Find the blue landing pad and land on it.');self.assertIsNone(proprio)
            self.assertFalse(re.search(r'\d',sentence))
            self.assertFalse(numbers&{round(abs(float(token)),1) for token in re.findall(r'-?\d+\.?\d*',sentence)})
            # The camera's own arrays, untouched: blur is off.
            self.assertTrue(seen['camera_frames_unchanged']);self.assertEqual(seen['failure'],'normal')
        # The call into the model has no parameter a target could pass through, and the gate before it refuses one in the sentence.
        self.assertEqual(list(inspect.signature(oft_runner.WatchedPolicy.infer).parameters),['self','front','down','instruction','proprio'])
        for leak in ('Find the blue landing pad 41.2 meters away.','Fly forward-right and land on the blue landing pad.','Go to 10.0 82.0.'):
            with self.assertRaises(ValueError):policy_inputs(None,None,leak)
        # What the display was given is what the evaluation loop logged as given to the policy.
        self.assertEqual([step['input_sha256'] for step in flight.steps],[seen['input_sha256'] for seen in flight.watch.inputs][:len(flight.steps)])
        evaluator=flight.watch.telemetry['evaluator']
        self.assertEqual(evaluator['target_xyz'],target['position']);self.assertNotIn('target_xyz',flight.watch.telemetry['model_input'])

    async def test_the_runner_has_one_way_to_the_model_and_no_way_to_the_simulators_landing(self):
        source=(ROOT/'src/mission/oft_runner.py').read_text(encoding='utf-8');code=source.split('"""',2)[2]
        self.assertEqual(len(re.findall(r'\.infer\(',code)),1)
        for word in ('land_async','land_and_confirm','semantic_direction','make_prompt','grounding_report','prompt_arguments','AeroVLAInt4','convert_action'):
            self.assertNotIn(word,code,word)
        self.assertIn('run_episode(env,episode,WatchedPolicy(',code)

    async def test_a_landing_is_flown_by_the_policy_and_ended_by_the_canonical_finalizer(self):
        import scripts.visual_search as canonical
        from src.visual_search.finalizer import LandingFinalizer,STATES
        from tests.fake_airsim import fly_to
        with tempfile.TemporaryDirectory() as directory:
            flight=Flight(directory);target=flight.by['blue_pad']
            summary=await flight.fly('blue_pad','land',lambda sim:fly_to(sim,target['position'],descend=True))
        self.assertTrue(summary['success']);self.assertEqual((summary['landed_on'],summary['reason']),('blue_pad','landed'))
        self.assertTrue(summary['system_land_success']);self.assertEqual(summary['evaluator'],'canonical_evaluator_v2')
        # contact -> stable -> latch -> no command passed on -> disarm, in that order.
        self.assertEqual(summary['finalizer']['path'],list(STATES))
        # The display is told after each decision; STABLE_CONTACT and the latch fall in the same one, so it sees the latch.
        order=[state for index,state in enumerate(flight.finalizers) if index==0 or state!=flight.finalizers[index-1]]
        self.assertEqual(order,[state for state in STATES if state!='STABLE_CONTACT'])
        latched=[step for step in flight.steps if step['finalizer']=='LANDED_LATCHED']
        self.assertTrue(latched);self.assertTrue(all(step['executed']==[0.,0.,0.] and step['action'][1]>0 for step in latched))
        self.assertEqual(flight.sim.calls[-2:],['disarm','disable']);self.assertTrue(flight.watch.telemetry['disarmed'])
        # The descent was the policy's: every downward metre was an action of the model, and no landing routine was called.
        self.assertNotIn('land_async',flight.sim.calls)
        self.assertGreater(sum(step['executed'][1] for step in flight.steps),3.)
        self.assertEqual(flight.watch.telemetry['finalizer']['state'],'DISARMED');self.assertEqual(flight.watch.telemetry['contact']['touchdown_on'],'blue_pad')
        # The loop is handed its own finalizer class back.
        self.assertIs(flight.restored,LandingFinalizer);self.assertIs(canonical.LandingFinalizer,LandingFinalizer)

    async def test_an_approach_has_no_finalizer_and_ends_hovering(self):
        from tests.fake_airsim import fly_to
        with tempfile.TemporaryDirectory() as directory:
            flight=Flight(directory);target=flight.by['blue_pad']
            summary=await flight.fly('blue_pad','approach',lambda sim:fly_to(sim,target['position']))
        self.assertEqual(flight.text,'Approach the blue landing pad.')
        self.assertTrue(summary['success']);self.assertEqual(summary['reason'],'model_stop');self.assertFalse(summary['landed'])
        self.assertNotIn('finalizer',summary);self.assertEqual(set(flight.finalizers),{'FLYING'});self.assertFalse(flight.watch.telemetry['finalizer']['active'])
        # Left in the air where the policy stopped: a hover, motors on.
        self.assertEqual(flight.sim.calls[-1],'hover');self.assertNotIn('disarm',flight.sim.calls);self.assertNotIn('land_async',flight.sim.calls)
        self.assertLess(flight.sim.z,flight.sim.ground-4.)

    async def test_the_same_pad_and_the_same_start_end_differently_by_the_sentence_alone(self):
        from tests.fake_airsim import fly_to
        ends={}
        for task in ('land','approach'):
            with tempfile.TemporaryDirectory() as directory:
                flight=Flight(directory);target=flight.by['blue_pad']
                await flight.fly('blue_pad',task,lambda sim:fly_to(sim,target['position'],descend=task=='land'))
                ends[task]=(flight.steps[0]['position'],flight.summary['landed'],flight.sim.held)
        self.assertEqual(ends['land'][0],ends['approach'][0])
        self.assertEqual((ends['land'][1:],ends['approach'][1:]),((True,True),(False,False)))

    async def test_a_stable_landing_on_the_other_pad_is_latched_and_is_a_failed_mission(self):
        from src.mission.semantic import outcome
        from tests.fake_airsim import fly_to
        with tempfile.TemporaryDirectory() as directory:
            flight=Flight(directory);other=flight.by['blue_pad']
            summary=await flight.fly('blue_pad','land',lambda sim:fly_to(sim,other['position'],descend=True),told='red_pad')
        self.assertEqual(flight.text,'Find the red landing pad and land on it.')
        # The finalizer was never told which pad was meant: it latched and disarmed. The evaluator says it was the wrong one.
        self.assertEqual(summary['finalizer']['state'],'DISARMED');self.assertTrue(summary['physical_landing'])
        self.assertFalse(summary['success']);self.assertTrue(summary['wrong_target']);self.assertEqual(summary['landed_on'],'blue_pad')
        self.assertIn('blue_pad',outcome(summary)['text'])

    async def test_touching_down_during_an_approach_is_a_failed_approach(self):
        from src.mission.semantic import outcome
        from tests.fake_airsim import fly_to
        with tempfile.TemporaryDirectory() as directory:
            flight=Flight(directory);target=flight.by['blue_pad']
            summary=await flight.fly('blue_pad','approach',lambda sim:fly_to(sim,target['position'],descend=True))
        self.assertTrue(summary['landed']);self.assertFalse(summary['success']);self.assertEqual(set(flight.finalizers),{'FLYING'})
        self.assertNotIn('disarm',flight.sim.calls);self.assertIn('touched down',outcome(summary)['text'])

    async def test_flying_into_an_object_is_a_collision(self):
        from src.mission.semantic import outcome
        from tests.fake_airsim import fly_to
        with tempfile.TemporaryDirectory() as directory:
            flight=Flight(directory);cube=flight.by['blue_cube']
            summary=await flight.fly('blue_cube','approach',lambda sim:fly_to(sim,cube['position'],arrive_m=0.,stop=False))
        self.assertEqual(summary['reason'],'collision');self.assertFalse(summary['success']);self.assertEqual(summary['hit'],'BlueCube')
        self.assertTrue(outcome(summary)['text'].startswith('COLLISION'));self.assertEqual(flight.watch.telemetry['contact']['collision'],'BlueCube')

    async def test_blur_changes_the_models_input_only_when_it_is_switched_on(self):
        from tests.fake_airsim import fly_to
        with tempfile.TemporaryDirectory() as directory:
            flight=Flight(directory,blur=True);target=flight.by['blue_pad']
            await flight.fly('blue_pad','approach',lambda sim:fly_to(sim,target['position']))
        self.assertTrue(all(not seen['camera_frames_unchanged'] and seen['failure']=='gaussian_blur' for seen in flight.watch.inputs))
        # Even then, the evaluation loop's log and the display agree on what the model was given.
        self.assertEqual([step['input_sha256'] for step in flight.steps],[seen['input_sha256'] for seen in flight.watch.inputs][:len(flight.steps)])


class ScriptedFeed:
    """A feed without files: the session's telemetry is kept, and the keys come from a script that reads it."""
    def __init__(self,operator):
        from src.mission.control import default_mission_control
        self.control=default_mission_control('grounding-film');self.operator=operator;self.states=[];self.images={};self.chase_ready=False
        self.control_warning=None;self.errors=[];self.telemetry={}
    def json(self,name,data):
        if name=='telemetry.json':
            self.telemetry=json.loads(json.dumps(data));mission=self.telemetry.get('mission') or {}
            if not self.states or self.states[-1]!=mission.get('state'):self.states.append(mission.get('state'))
            self.control=self.operator(self.control,self.telemetry) or self.control
    def image(self,name,frame):self.images[name]=frame
    def append(self,*_):pass
    def fail(self,frames):return frames,{'failure_enabled':False,'failure_type':'normal','severity':'medium','revision':0}
    def chase_callback(self,*_):pass
    def flush(self,*_):return True
    def close(self):pass


class SessionHarness:
    async def session(self,operator,scripts):
        """Run the interactive runner's main loop with the fake simulator and a scripted operator."""
        from src.mission import oft_runner
        from tests.fake_airsim import FakeSim,FakeClient,ScriptedModel,world_class,drone_class
        import scripts.visual_search as canonical
        config,scene,targets=demo();self.by={item['object_id']:item for item in targets};sim=FakeSim(scene,getattr(self,'options',{}).get('layout') or 'mission');self.sim=sim
        oft=json.loads((ROOT/'configs/aerovla_oft.json').read_text(encoding='utf-8'));oft['grounding']={'type':'film','position':'before_projector','hidden_dim':512}
        feed=ScriptedFeed(operator);self.feed=feed;models=[]
        def script(index):
            # Each mission flies the next script; the stand-in policy reads the fake vehicle, never the mission's target.
            mission=feed.telemetry.get('mission',{}).get('mission_id',1) or 1
            return scripts[min(len(scripts),mission)-1](sim)(index)
        model=ScriptedModel(script,oft);self.model=model
        loader=types.SimpleNamespace(model=model,error=None,info={'grounding_loaded':True,'chunk_size':4,'execute_horizon':1,'lora_loaded':True,'head_loaded':True})
        overview=Mock();overview.point_at.side_effect=lambda frame,pixel:self.clicks[tuple(pixel)]
        async def no_wait(_):return None
        clock=types.SimpleNamespace(sleep=no_wait,wait_for=asyncio.wait_for)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);checkpoint=output/'checkpoint';checkpoint.mkdir();(checkpoint/'manifest.json').write_text(json.dumps({'config':oft}))
            (output/'checkpoint-check.json').write_text(json.dumps({'checkpoint':'checkpoint','problems':[],'warnings':[],'frozen':{'state':'verified'}}))
            with patch.object(oft_runner,'OUT',output),patch.object(oft_runner,'Feed',lambda _:feed),patch.object(oft_runner,'Loader',lambda *_:loader), \
                 patch.object(oft_runner,'ProjectAirSimClient',lambda **_:FakeClient(sim)),patch.object(oft_runner,'OverviewScene',lambda *_:overview), \
                 patch.object(canonical,'World',world_class(sim)),patch.object(canonical,'Drone',drone_class(sim)), \
                 patch.object(canonical,'matched_camera_images',sim.camera),patch.object(oft_runner,'matched_camera_images',sim.camera), \
                 patch.object(canonical,'asyncio',clock),patch('builtins.print'):
                options=dict(host='fake',checkpoint=str(checkpoint),steps=0,demo=None,run_token='test',layout=None,start_x=None,start_y=None,start_yaw=None,start_height=None)
                options.update(getattr(self,'options',{}))
                await oft_runner.main(argparse.Namespace(**options))
            folder=next((output/'missions').iterdir())
            self.missions=[json.loads(line) for line in (folder/'missions.jsonl').read_text().splitlines()] if (folder/'missions.jsonl').exists() else []
            self.records=[json.loads(path.read_text()) for path in sorted(folder.glob('mission-*.json'))]
            self.session_record=json.loads((folder/'session.json').read_text())
        return feed

    def operator(self,plan):
        """Keys pressed when the session reaches a state: a list of (state the mission must be in, keys or a map click)."""
        from src.mission.control import mission_key,request_selection,request_start_pose
        steps=list(plan);self.messages=[];self.seen=[]
        def press(control,telemetry):
            mission=telemetry.get('mission') or {};self.seen.append(json.loads(json.dumps({'mission':mission,'message':telemetry.get('control_message')})))
            if not steps:return None
            wanted,keys=steps[0]
            # A key is pressed once the session has answered the last one, or at once while a mission flies.
            if wanted!='NAVIGATING' and telemetry.get('mission_request_ack',0)<control.get('mission_request_id',0):return None
            if mission.get('state')!=wanted or telemetry.get('phase','').startswith('Loading the scene'):return None
            if wanted=='NAVIGATING' and telemetry.get('step',0)<5:return None
            steps.pop(0)
            for key in keys:
                # A tuple is a map click on a target; a dictionary is a start placed on the map (pressed at, released at).
                if isinstance(key,dict):control=request_start_pose(control,1,key['pixel'],key.get('heading'))
                else:control=request_selection(control,1,list(key)) if isinstance(key,tuple) else mission_key(control,ord(key))
            return control
        return press


class SessionTests(SessionHarness,unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):asyncio.get_running_loop().slow_callback_duration=60.

    async def test_land_then_reset_then_approach_from_the_same_pose(self):
        from tests.fake_airsim import fly_to
        pad=demo()[2][0]['position'];self.clicks={}
        plan=[('IDLE','n'),('TARGET_SELECTED','g'),('SUCCESS','r'),('TARGET_SELECTED','t'),('TARGET_SELECTED','g'),('SUCCESS','q')]
        feed=await self.session(self.operator(plan),[lambda sim:fly_to(sim,pad,descend=True),lambda sim:fly_to(sim,pad)])
        self.assertEqual([(m['target_object'],m['task'],m['instruction'],m['success']) for m in self.missions],
                         [('blue_pad','land','Find the blue landing pad and land on it.',True),('blue_pad','approach','Approach the blue landing pad.',True)])
        first,second=self.records
        # Both flights began at the launch pose; R put the vehicle back there, and the target stayed selected.
        self.assertEqual(first['steps'][0]['position'][:2],second['steps'][0]['position'][:2]);self.assertEqual(first['steps'][0]['position'][:2],[30.,56.])
        self.assertEqual(first['summary']['finalizer']['state'],'DISARMED');self.assertNotIn('finalizer',second['summary'])
        self.assertTrue(first['summary']['landed']);self.assertFalse(second['summary']['landed'])
        for record in self.records:
            self.assertTrue(record['input_hash_agreement']);self.assertTrue(record['camera_frames_unchanged'])
            self.assertEqual(record['model_input'],'Front RGB + Down RGB + Instruction ONLY')
            for key in ('mission_id','checkpoint','target_object','target_noun','task','instruction','steps','success'):self.assertIn(key,record)
            for step in record['steps']:
                for key in ('input_sha256','chunk','normalised','action','front_seen','down_seen','distance_m','finalizer'):self.assertIn(key,step)
            # (The decision on which the policy stops by itself is logged without a command: none follows it.)
            self.assertTrue(all('executed' in step for step in record['steps'][:-1]))
        # Every call the stand-in model received: two frames, the mission's sentence, nothing else.
        sentences={arguments[2] for arguments,keywords in self.model.calls}
        self.assertEqual(sentences,{'Find the blue landing pad and land on it.','Approach the blue landing pad.'})
        self.assertTrue(all(len(arguments)==4 and not keywords and arguments[3] is None for arguments,keywords in self.model.calls))
        self.assertNotIn('land_async',self.sim.calls);self.assertEqual(self.session_record['cleanup']['simulator_landing_routine'],'not used')
        self.assertEqual(feed.telemetry['status'],'STOPPED');self.assertEqual(feed.telemetry['policy']['prompt_mode'],'instruction-only')
        self.assertEqual(feed.telemetry['failure']['failure_enabled'],False)

    async def test_bare_ground_and_a_landing_on_the_sphere_never_start_a_mission(self):
        cube=demo()[2][6]['position'];self.clicks={(100,100):[30.,70.,-1.],(200,200):cube}
        plan=[('IDLE',[(100,100)]),('IDLE','g'),('IDLE',[(200,200)]),('TARGET_SELECTED','g'),('TARGET_SELECTED','q')]
        feed=await self.session(self.operator(plan),[lambda sim:(lambda index:[0.,0.,0.])])
        messages=[item['message'] for item in self.seen if item['message']]
        self.assertIn('grounding_film requires a semantic object target. Select a configured landmark/object.',messages)
        self.assertIn('Selected object has no valid landing surface. Use APPROACH.',messages)
        self.assertEqual(self.missions,[]);self.assertEqual(self.model.calls,[]);self.assertNotIn('NAVIGATING',feed.states)
        # The ball stays selected, the display says why G will not start, and APPROACH would.
        self.assertEqual(feed.telemetry['mission']['target']['object_id'],'orange_ball')
        self.assertEqual(feed.telemetry['mission']['refusal'],'Selected object has no valid landing surface. Use APPROACH.')
        self.assertEqual((feed.telemetry['mission']['target']['surface']['surface'],feed.telemetry['mission']['target']['landable']),('none',False))

    async def test_r_during_a_flight_records_it_as_aborted_and_returns_to_the_launch_pose(self):
        from tests.fake_airsim import fly_to
        pad=demo()[2][0]['position'];self.clicks={}
        plan=[('IDLE','n'),('TARGET_SELECTED','tg'),('NAVIGATING','r'),('TARGET_SELECTED','q')]
        feed=await self.session(self.operator(plan),[lambda sim:fly_to(sim,pad)])
        self.assertEqual([(m['state'],m['success']) for m in self.missions],[('ABORTED',False)])
        self.assertEqual([round(self.sim.x,1),round(self.sim.y,1)],[30.,56.]);self.assertEqual(feed.telemetry['mission']['target']['object_id'],'blue_pad')
        self.assertNotIn('land_async',self.sim.calls)


class ViewerTests(unittest.TestCase):
    def telemetry(self,**fields):
        config,scene,targets=demo();target=dict(targets[0],surface_position=targets[0]['position'],launch_distance_m=41.2,launch_bearing_deg=29.1,in_canonical_range=True)
        base={'status':'RUNNING','phase':'AeroVLA-OFT flight','step':12,'input_step':12,'max_steps':320,'targets':targets,'ground_z':-1.19,
              'launch':dict(scene['mission']['launches'][0],index=0,count=2),
              'policy':{'mode':'grounding-film','model':'grounding_film','grounding':'film','frozen':'verified','model_input':'Front RGB + Down RGB + Instruction ONLY',
                        'not_sent':['target XYZ','distance','bearing','direction hint','overview','marker'],'tick_s':.5,
                        'loaded':{'grounding_loaded':True,'chunk_size':4,'execute_horizon':1}},
              'mission':{'state':'NAVIGATING','task':'land','instruction':'Find the blue landing pad and land on it.','target':target,'success_radius_m':15.,
                         'canonical_range_m':44.,'errors':{'horizontal_m':41.2,'bearing_deg':29.1},'result':None,'refusal':None},
              'model_input':{'step':12,'instruction':'Find the blue landing pad and land on it.','sha256':{'front':'a'*64,'down':'b'*64}},
              'evaluator':{'target_xyz':target['position'],'distance_m':20.,'bearing_deg':3.,'height_m':6.,'front_seen':True,'down_seen':False,'front_pixel':[120.,130.],'down_pixel':None},
              'inference':{'chunk':[[.8,0.,.05]]*4,'action':[.8,0.,.05],'normalised':[.6,0.,.24],'inference_ms':410.,'stop':False},'executed':[.8,0.,.05],
              'finalizer':{'active':True,'state':'LANDING_DESCENT','path':['FLYING','LANDING_DESCENT'],'finalizer_triggered':False},
              'failure':{'failure_enabled':False,'camera_frames_unchanged':True},'timing':{'decision_s':.52,'inference_ms':410.}}
        base.update(fields);return base

    def test_the_canvas_draws_every_state_of_a_mission(self):
        from scripts.grounding_mission_view import render,BUTTONS,annotate_map
        from src.mission.control import default_mission_control
        control=default_mission_control('grounding-film');frame=np.full((256,256,3),90,dtype=np.uint8)
        images={'front':frame,'down':frame,'chase':np.zeros((360,640,3),np.uint8),'chase_cam':np.zeros((360,640,3),np.uint8),'verified':True}
        flying=render(self.telemetry(),control,images);self.assertEqual(flying.shape,(1040,1480,3))
        idle=self.telemetry(mission={'state':'IDLE','target':None,'task':'land','success_radius_m':15.,'refusal':'x'},model_input=None,evaluator=None,inference=None,executed=None,finalizer=None)
        refused=self.telemetry();refused['mission'].update(state='TARGET_SELECTED',refusal='Selected object has no valid landing surface. Use APPROACH.')
        done=self.telemetry(contact={'touchdown_on':'blue_pad','touchdown_offset_m':[1.,2.],'collision':None},disarmed=True,input_verified=True)
        done['mission'].update(state='SUCCESS',result={'success':True,'text':'landed on the named pad; latched and disarmed','selected':'blue_pad','stopped':False,'steps':80})
        done['finalizer']={'active':True,'state':'DISARMED','path':['FLYING','LANDING_DESCENT','CONTACT_CANDIDATE','STABLE_CONTACT','LANDED_LATCHED','DISARMED'],'finalizer_triggered':True}
        crashed=self.telemetry(contact={'touchdown_on':None,'touchdown_offset_m':None,'collision':'BlueCube'})
        approach=self.telemetry(finalizer={'active':False,'state':'FLYING','path':['FLYING']});approach['mission']['task']='approach'
        for telemetry in (idle,refused,done,crashed,approach,{}):
            for overlay in (False,True):
                self.assertEqual(render(telemetry,dict(control,main_view='chase'),images,overlay).shape,(1040,1480,3))
        self.assertEqual(render({},control,{}).shape,(1040,1480,3))
        # The frames are drawn as they are: the overlay, when asked for, goes on a copy.
        before=frame.copy();shown=render(self.telemetry(),control,images,True)
        np.testing.assert_array_equal(frame,before);self.assertFalse(np.array_equal(shown[754:1010,20:276],frame))
        np.testing.assert_array_equal(render(self.telemetry(),control,images,False)[754:1010,20:276],frame)
        self.assertEqual(len({key for *_,key,_ in BUTTONS}),len(BUTTONS));self.assertIn(ord('t'),{key for *_,key,_ in BUTTONS})
        self.assertNotIn(ord('m'),{key for *_,key,_ in BUTTONS})
        packet={'camera':{'width':640,'height':360,'fov_deg':80,'position':[30.,60.,-200.],'orientation':[0.,-math.sin(math.pi/4),0.,math.cos(math.pi/4)]}}
        annotate_map(np.zeros((360,640,3),np.uint8),packet,self.telemetry())

    def test_the_earlier_canvas_is_drawn_as_before(self):
        from scripts.mission_viewer import render_canvas,MISSION_BUTTONS
        from src.mission.control import default_mission_control
        self.assertEqual(render_canvas({},default_mission_control(),{}).shape,(1040,1480,3))
        self.assertIn(ord('m'),{key for *_,key,_ in MISSION_BUTTONS})


class LauncherTests(unittest.TestCase):
    def test_the_launchers_keep_the_legacy_policy_as_the_default_and_add_the_new_one(self):
        blur=(ROOT/'scripts/run_blur_demo.ps1').read_text(encoding='utf-8');mission=(ROOT/'scripts/run_mission_demo.ps1').read_text(encoding='utf-8')
        short=(ROOT/'scripts/run_grounding_film_mission_demo.ps1').read_text(encoding='utf-8')
        for text in (blur,mission):self.assertIn("[ValidateSet('legacy','grounding-film')][string]$Policy='legacy'",text)
        self.assertIn("-Mode mission -Policy grounding-film -Checkpoint $Checkpoint",short)
        self.assertIn("[string]$Checkpoint='outputs/aerovla_oft/checkpoints/grounding_film'",short)
        self.assertIn("'src/mission/oft_runner.py'",blur);self.assertIn("'src/mission/runner.py'",blur)
        for line in ('GROUNDING-FILM MISSION CONTROL','1. Click a semantic object or press N.','2. Press T to choose LAND / APPROACH.','4. Front/Down show actual model inputs.',
                     '5. Target XYZ/distance/bearing are evaluator-only.','7. Q/Esc safely exits.','Failure injection:'):self.assertIn(line,blur)
        for path in ('scripts/run_blur_demo.ps1','scripts/run_mission_demo.ps1','scripts/run_grounding_film_mission_demo.ps1'):
            data=(ROOT/path).read_bytes();self.assertEqual(data.count(b'\n'),data.count(b'\r\n'),path)

    def test_the_worker_can_be_stopped_as_the_earlier_one_is(self):
        text=(ROOT/'scripts/stop_blur_worker.py').read_text(encoding='utf-8')
        self.assertIn("'grounding':'src/mission/oft_runner.py'",text)
        self.assertIn("if '--mission-demo' in sys.argv:(OUT/'worker-pid.txt')",(ROOT/'src/mission/oft_runner.py').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
