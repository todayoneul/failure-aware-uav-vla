"""Gaussian blur robustness characterization: the fixed conditions, the failure between camera and policy, the order of
flights, and how flights are read and compared. Flights here are the canonical loop over a fake simulator; no model."""
import argparse
import asyncio
import inspect
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def experiment():
    from src.failures import blur_experiment as blur
    return blur,blur.load()


def sharp(seed=0):
    return np.random.default_rng(seed).integers(0,256,(256,256,3),dtype=np.uint8)


class ConditionTests(unittest.TestCase):
    def test_the_severities_are_fixed_and_are_the_demos(self):
        from src.failures.gaussian_blur import SEVERITIES
        blur,config=experiment()
        self.assertEqual(config['conditions'],{'clean':None,'low':{'kernel':7,'sigma':1.5},'medium':{'kernel':15,'sigma':3.0},'high':{'kernel':31,'sigma':6.0}})
        self.assertEqual(SEVERITIES,{'low':(7,1.5),'medium':(15,3.0),'high':(31,6.0)})
        self.assertEqual(config['cameras'],['front','down'])
        self.assertEqual((config['checkpoint'],config['frozen_record'],config['starts']),
                         ('outputs/aerovla_oft/checkpoints/grounding_film','grounding_film','configs/gen_v3_canonical_44m_test.json'))
        # A configuration that disagrees with the demo's keys is refused, not silently used.
        with tempfile.TemporaryDirectory() as directory:
            changed=dict(config,conditions=dict(config['conditions'],medium={'kernel':13,'sigma':3.0}));path=Path(directory)/'c.json';path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):blur.load(path)

    def test_clean_hands_on_the_cameras_own_arrays(self):
        blur,_=experiment();frames={'front':sharp(1),'down':sharp(2)};before={name:frame.copy() for name,frame in frames.items()}
        used=blur.Injector('clean').apply(frames)
        self.assertIs(used,frames);self.assertIs(used['front'],frames['front']);self.assertIs(used['down'],frames['down'])
        np.testing.assert_array_equal(frames['front'],before['front'])

    def test_each_severity_is_exactly_its_kernel_and_sigma_on_both_cameras(self):
        import cv2
        blur,config=experiment();frames={'front':sharp(1),'down':sharp(2)};before={name:frame.copy() for name,frame in frames.items()};results={}
        for name in ('low','medium','high'):
            spec=config['conditions'][name];injector=blur.Injector(name);used=injector.apply(frames);results[name]=used
            self.assertEqual((injector.kernel,injector.sigma),(spec['kernel'],spec['sigma']))
            for camera in ('front','down'):
                np.testing.assert_array_equal(used[camera],cv2.GaussianBlur(before[camera],(spec['kernel'],spec['kernel']),spec['sigma']))
                self.assertIsNot(used[camera],frames[camera]);self.assertFalse(np.array_equal(used[camera],before[camera]))
                self.assertEqual((used[camera].shape,used[camera].dtype),((256,256,3),np.uint8))
            # The camera's arrays are left as they were, and the same frame always gives the same result.
            for camera in ('front','down'):np.testing.assert_array_equal(frames[camera],before[camera])
            again=blur.Injector(name).apply(frames)
            for camera in ('front','down'):np.testing.assert_array_equal(again[camera],used[camera])
        self.assertFalse(np.array_equal(results['low']['front'],results['medium']['front']));self.assertFalse(np.array_equal(results['medium']['front'],results['high']['front']))
        with self.assertRaises(ValueError):blur.Injector('extreme')

    def test_sharpness_numbers_fall_with_severity_and_are_only_measured(self):
        blur,config=experiment();frame=sharp(3);values=[blur.quality(frame)]+[blur.quality(blur.Injector(name).apply({'front':frame,'down':frame})['front']) for name in ('low','medium','high')]
        for key in ('laplacian_variance','gradient_magnitude','local_contrast'):
            numbers=[value[key] for value in values];self.assertEqual(numbers,sorted(numbers,reverse=True),key);self.assertGreater(numbers[0],numbers[-1])
        self.assertEqual(set(values[0]),set(config['image_quality'])-{'note'})
        # Nothing of this, and nothing naming the condition, is among what a policy can be given.
        from scripts.blur_characterization import RecordedPolicy
        self.assertEqual(list(inspect.signature(RecordedPolicy.infer).parameters),['self','front','down','instruction','proprio'])

    def test_switching_the_demos_blur_off_gives_the_original_frame_back(self):
        from src.failures.control import write_control
        from src.mission.control import default_mission_control,mission_key
        from src.mission.feed import Feed
        import time
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);control=default_mission_control('grounding-film');write_control(output/'control.json',control)
            feed=Feed(output);frames={'front':sharp(4),'down':sharp(5)};before=frames['front'].copy()
            def wait(check):
                for _ in range(200):
                    if check(feed.control):return
                    time.sleep(.02)
                self.fail('the control file was not read')
            used,report=feed.fail(frames);self.assertIs(used,frames);self.assertEqual((report['failure_enabled'],report['kernel']),(False,None))
            for key,name,kernel,sigma in (('1','low',7,1.5),('2','medium',15,3.0),('3','high',31,6.0)):
                control=mission_key(control,ord(key));control=control if control['enabled'] else mission_key(control,ord('b'));write_control(output/'control.json',control)
                wait(lambda c:c['enabled'] and c['severity']==name)
                used,report=feed.fail(frames)
                self.assertEqual((report['failure_enabled'],report['severity'],report['kernel'],report['sigma']),(True,name,kernel,sigma))
                self.assertIsNot(used['front'],frames['front']);self.assertIsNot(used['down'],frames['down'])
            control=mission_key(control,ord('b'));write_control(output/'control.json',control);wait(lambda c:not c['enabled'])
            used,report=feed.fail(frames)
            self.assertIs(used,frames);self.assertIs(used['front'],frames['front']);np.testing.assert_array_equal(used['front'],before);self.assertFalse(report['failure_enabled'])
            feed.close()


class OrderTests(unittest.TestCase):
    def starts(self):
        blur,config=experiment();return blur,config,json.loads((ROOT/config['starts']).read_text(encoding='utf-8'))['episodes']

    def test_every_start_is_flown_once_in_every_condition_in_a_balanced_fixed_order(self):
        blur,config,episodes=self.starts();ids=[episode['id'] for episode in episodes];flights=blur.flight_order(config,ids,'main')
        self.assertEqual(len(flights),192);self.assertEqual(len(set(flights)),192)
        self.assertEqual({flight for flight in flights},{(start,name) for start in ids for name in config['conditions']})
        # Start by start, and every condition first, second, third and last twelve times.
        self.assertEqual([flights[index][0] for index in range(0,192,4)],ids)
        self.assertEqual(blur.positions(flights,list(config['conditions'])),{name:[12,12,12,12] for name in config['conditions']})
        orders=[tuple(condition for _,condition in flights[index:index+4]) for index in range(0,192,4)]
        self.assertEqual(len(set(orders)),24);self.assertTrue(all(orders.count(order)==2 for order in set(orders)))
        self.assertEqual(flights,blur.flight_order(config,ids,'main'))
        self.assertNotEqual(flights,blur.flight_order(dict(config,order=dict(config['order'],seed=1)),ids,'main'))

    def test_the_repeat_subset_was_chosen_from_the_plan_and_is_what_its_rule_gives(self):
        from src.visual_search.maps import load_map
        blur,config,episodes=self.starts();saved=json.loads((ROOT/config['repeat']['subset']).read_text(encoding='utf-8'))
        scenes={name:load_map(name) for name in {episode['map'] for episode in episodes}}
        self.assertEqual(saved['starts'],blur.repeat_subset(config,episodes,scenes))
        ids=[item['id'] for item in saved['starts']];self.assertEqual(len(ids),12);self.assertEqual(len(set(ids)),12)
        self.assertTrue(set(ids)<={episode['id'] for episode in episodes})
        kinds=[item['chosen_as'] for item in saved['starts']]
        self.assertEqual({kind:kinds.count(kind) for kind in dict.fromkeys(kinds)},{'same_color':3,'same_shape':2,'initially_invisible':2,'far':2,'landing_heavy':2,'easy_control':1})
        for item in saved['starts']:
            if item['chosen_as']=='same_color':self.assertTrue(item['same_color'])
            if item['chosen_as']=='same_shape':self.assertTrue(item['same_shape'])
            if item['chosen_as']=='initially_invisible':self.assertTrue(item['initially_invisible'])
            if item['chosen_as']=='far':self.assertEqual(item['band'],'far')
            if item['chosen_as']=='landing_heavy':self.assertEqual((item['task'],item['band'],item['initially_invisible']),('land','near',False))
            if item['chosen_as']=='easy_control':self.assertEqual((item['task'],item['band'],item['initially_invisible']),('approach','near',False))
        flights=blur.flight_order(config,ids,'repeat');self.assertEqual(len(flights),48)
        self.assertEqual(blur.positions(flights,list(config['conditions'])),{name:[3,3,3,3] for name in config['conditions']})

    def test_the_kinds_of_start_are_read_as_the_canonical_tables_read_them(self):
        from src.visual_search.maps import load_map
        blur,config,episodes=self.starts();scenes={name:load_map(name) for name in {episode['map'] for episode in episodes}}
        facts=[blur.start_facts(episode,scenes[episode['map']]) for episode in episodes]
        # The numbers of the canonical test's own table: 22 same colour, 26 same shape, 17 with a related object first, 27 hidden.
        self.assertEqual((sum(f['same_color'] for f in facts),sum(f['same_shape'] for f in facts),sum(bool(f['first_view']) for f in facts),sum(f['initially_invisible'] for f in facts)),(22,26,17,27))


class ReadingTests(unittest.TestCase):
    def test_a_failed_flight_gets_the_class_of_the_first_stage_it_did_not_pass(self):
        blur,_=experiment();record=lambda stage,how='',**more:{'success':False,'failure':stage,'mechanism':how,**more}
        self.assertEqual(blur.failure_class({'success':True,'failure':''}),'')
        for stage,expected in (('COLLISION','B9_COLLISION'),('SEARCH','B1_SEARCH_FAILURE'),('WRONG_TARGET','B8_WRONG_TARGET'),('GROUNDING','B2_GROUNDING_FAILURE'),
                               ('APPROACH','B3_APPROACH_FAILURE'),('ALIGNMENT','B4_ALIGNMENT_FAILURE'),('PHYSICAL_LANDING','B7_STABLE_LANDING_FAILURE'),('FINALIZER','B7_STABLE_LANDING_FAILURE')):
            self.assertEqual(blur.failure_class(record(stage)),expected)
        self.assertEqual(blur.failure_class(record('DESCENT','did not start',descent_started=False)),'B5_DESCENT_FAILURE')
        self.assertEqual(blur.failure_class(record('DESCENT','timeout',descent_started=True)),'B6_TOUCHDOWN_FAILURE')
        self.assertEqual(blur.failure_class(record('STOP','did not stop')),'B10_TIMEOUT')
        self.assertEqual(blur.failure_class(record('STOP','stopped outside the radius')),'B3_APPROACH_FAILURE')
        self.assertEqual(blur.failure_class(record('STOP','landed instead of hovering')),'B12_OTHER')
        self.assertEqual((blur.cell({'success':True}),blur.cell(record('SEARCH')),blur.cell(record('WRONG_TARGET')),blur.cell(None)),('S','F_SEARCH','F_WRONG_TARGET','E_RUNTIME'))
        self.assertTrue(set(blur.CLASSES.values())<=set(blur.ALL_CLASSES))
        # Every stage the canonical classifier can name is mapped.
        from scripts.freeze_gen_v3 import STAGES
        self.assertTrue(set(STAGES)<=set(blur.CLASSES)|{'STOP','DESCENT'})

    def test_paired_counts_onset_retention_and_separation(self):
        blur,_=experiment()
        clean={'a':True,'b':True,'c':True,'d':False,'e':True};high={'a':True,'b':False,'c':False,'d':True,'e':False}
        counts=blur.paired(clean,high);self.assertEqual((counts['S_S'],counts['S_F'],counts['F_S'],counts['F_F']),(1,3,1,0))
        self.assertAlmostEqual(counts['mcnemar_exact_p'],2*(1+4)/16)          # 1 of 4 discordant or fewer, both sides
        self.assertEqual(blur.paired(clean,clean)['mcnemar_exact_p'],1.)
        self.assertEqual(blur.onset({'clean':True,'low':True,'medium':True,'high':True}),{'onset':'NONE','non_monotonic':False})
        self.assertEqual(blur.onset({'clean':True,'low':True,'medium':False,'high':False}),{'onset':'MEDIUM','non_monotonic':False})
        self.assertEqual(blur.onset({'clean':True,'low':False,'medium':True,'high':False}),{'onset':'LOW','non_monotonic':True})
        self.assertEqual(blur.onset({'clean':False,'low':True,'medium':True,'high':True}),{'onset':'CLEAN','non_monotonic':True})
        self.assertEqual(blur.retention(.96,.72),{'drop_points':-24.0,'retention':.75})
        self.assertEqual(blur.auroc([1,2,3],[4,5,6]),1.);self.assertEqual(blur.auroc([4,5,6],[1,2,3]),0.);self.assertEqual(blur.auroc([1,2],[1,2]),.5)


class Flight:
    """One canonical flight over the fake simulator with the experiment's own environment and policy wrapper."""
    def __init__(self,directory,condition):
        from scripts import blur_characterization as worker
        from src.mission import semantic
        from src.visual_search.episodes import load_config
        from tests.fake_airsim import FakeSim,FakeClient
        import scripts.visual_search as canonical
        self.worker=worker;self.canonical=canonical;self.config=load_config();self.scene=semantic.load_demo()
        self.targets={item['object_id']:item for item in semantic.semantic_targets(self.scene)}
        self.oft=json.loads((ROOT/'configs/aerovla_oft.json').read_text(encoding='utf-8'));self.sim=FakeSim(self.scene,'mission')
        self.env=worker.make_environment(canonical)(FakeClient(self.sim),self.config,self.oft,Path(directory))
        from src.failures import blur_experiment as blur
        self.injector=blur.Injector(condition);self.env.injector=self.injector;self.env.trace=[]

    async def fly(self,target,task,script):
        from src.mission import semantic
        from tests.fake_airsim import ScriptedModel,world_class,drone_class
        named=self.targets[target];text=semantic.sentence(self.config,self.scene,named,task)
        self.episode=semantic.mission_episode(semantic.DEMO,self.scene,named,task,text,self.scene['mission']['launches'][0],1,320)
        self.model=ScriptedModel(script(self.sim),self.oft);self.policy=self.worker.RecordedPolicy(self.model)
        async def no_wait(_):return None
        clock=types.SimpleNamespace(sleep=no_wait,wait_for=asyncio.wait_for);canonical=self.canonical
        with patch.object(canonical,'World',world_class(self.sim)),patch.object(canonical,'Drone',drone_class(self.sim)), \
             patch.object(canonical,'matched_camera_images',self.sim.camera),patch.object(canonical,'asyncio',clock),patch('builtins.print'):
            await self.env.reset(self.episode)
            arguments=argparse.Namespace(policy='oft',finalizer='on',live=False,output='unused',set='test',model_name='test')
            self.summary,self.steps=await canonical.run_episode(self.env,self.episode,self.policy,arguments,None)
        self.rows,self.checks=self.worker.examine(self.injector,self.env.trace,self.policy.seen,self.steps,self.episode,False)
        return self.summary


class InjectionInTheLoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):asyncio.get_running_loop().slow_callback_duration=60.

    async def flights(self,task='land'):
        from tests.fake_airsim import fly_to
        flown={}
        for condition in ('clean','low','medium','high'):
            with tempfile.TemporaryDirectory() as directory:
                flight=Flight(directory,condition);pad=flight.targets['blue_pad']['position']
                await flight.fly('blue_pad',task,lambda sim:fly_to(sim,pad,descend=task=='land'));flown[condition]=flight
        return flown

    async def test_blur_reaches_the_model_and_only_the_model(self):
        flown=await self.flights();clean=flown['clean']
        self.assertEqual(clean.checks['decisions_with_input_identical_to_raw'],{'front':len(clean.steps),'down':len(clean.steps)})
        self.assertTrue(all(item['used'] is item['raw'] for item in clean.env.trace))
        for name in ('low','medium','high'):
            flight=flown[name]
            # Every decision's Front and Down were changed, and the log, the policy's own record and the arrays agree on what was given.
            self.assertEqual(flight.checks['decisions_with_input_identical_to_raw'],{'front':0,'down':0})
            self.assertTrue(all(row['raw_sha256']['front']!=row['input_sha256']['front'] and row['raw_sha256']['down']!=row['input_sha256']['down'] for row in flight.rows))
            self.assertEqual([row['input_sha256'] for row in flight.rows],[step['input_sha256'] for step in flight.steps])
            self.assertEqual((flight.checks['kernel'],flight.checks['sigma']),(flight.injector.kernel,flight.injector.sigma))
            # The model is called as in the clean flight: two frames, the sentence, nothing about the condition.
            for arguments,keywords in flight.model.calls:
                self.assertEqual(keywords,{});self.assertEqual(len(arguments),4);self.assertEqual(arguments[2],'Find the blue landing pad and land on it.');self.assertIsNone(arguments[3])
                self.assertEqual((arguments[0].shape,arguments[0].dtype,arguments[1].shape),((256,256,3),np.uint8,(256,256,3)))
            # What the simulator says about the scene is what it says without blur: the same flight, decision for decision.
            self.assertEqual(len(flight.steps),len(clean.steps))
            for key in ('position','distance_m','bearing_deg','front_seen','down_seen','front_pixel','over','height_m','target_size','finalizer','teacher_action','action','landed'):
                self.assertEqual([step.get(key) for step in flight.steps],[step.get(key) for step in clean.steps],key)
            self.assertEqual({key:flight.summary[key] for key in ('success','reason','selected','landed_on','system_land_success','steps')},
                             {key:clean.summary[key] for key in ('success','reason','selected','landed_on','system_land_success','steps')})
        self.assertTrue(clean.summary['success']);self.assertEqual(clean.summary['finalizer']['state'],'DISARMED')

    async def test_a_harness_that_gave_the_policy_something_else_is_an_error(self):
        flown=await self.flights('approach');flight=flown['medium'];worker=flight.worker
        other=[(front.copy(),down,sentence,proprio) for front,down,sentence,proprio in flight.policy.seen]
        with self.assertRaisesRegex(RuntimeError,'not given the arrays'):worker.examine(flight.injector,flight.env.trace,other,flight.steps,flight.episode,False)
        with self.assertRaisesRegex(RuntimeError,'logged decisions'):worker.examine(flight.injector,flight.env.trace,flight.policy.seen[:-1],flight.steps,flight.episode,False)
        told=[(front,down,sentence+' The image is blurred.',proprio) for front,down,sentence,proprio in flight.policy.seen]
        with self.assertRaisesRegex(RuntimeError,'sentence'):worker.examine(flight.injector,flight.env.trace,told,flight.steps,flight.episode,False)
        # Per decision: sharpness of the raw frame and of the given one, the cost of the blur, the length of the decision.
        row=flight.rows[0];self.assertEqual(set(row['quality']),{'raw','input'});self.assertEqual(set(row['quality']['input']),{'front','down'})
        self.assertGreaterEqual(row['blur_ms'],0.);self.assertIsNotNone(row['cycle_s']);self.assertIsNone(flight.rows[-1]['cycle_s'])
        # The simulator's own clock against the wall clock over the flight is part of every flight's record.
        self.assertGreater(flight.checks['simulator_s'],0.);self.assertGreater(flight.checks['wall_s'],0.);self.assertIsNotNone(flight.checks['real_time_factor'])


class LauncherTests(unittest.TestCase):
    def test_the_simulator_is_started_as_the_evaluation_starts_it_and_the_freeze_is_verified_around_it(self):
        mine=(ROOT/'scripts/run_blur_characterization.ps1').read_text(encoding='utf-8');theirs=(ROOT/'scripts/run_visual_search.ps1').read_text(encoding='utf-8')
        line="$taskSimArgs=@('-windowed','-ResX=640','-ResY=480')";self.assertIn(line,mine);self.assertIn(line,theirs)
        # Same arguments; the window is shown here (a hidden simulator stops when the desktop session is disconnected).
        self.assertIn("$taskWindow=if($HideSimulator){'Hidden'}else{'Normal'}",mine)
        self.assertLess(mine.index('Confirm-FrozenBaseline "before launch'),mine.index('Start-Process -FilePath $taskExe'))
        self.assertLess(mine.index('$taskWorker.WaitForExit()'),mine.index('Confirm-FrozenBaseline "after launch'))
        blur,config=experiment();self.assertEqual(config['runtime']['simulator_arguments'],['-windowed','-ResX=640','-ResY=480'])
        self.assertEqual((config['runtime']['simulator_window'],config['runtime']['real_time']['minimum_factor']),('shown',.8))
        data=(ROOT/'scripts/run_blur_characterization.ps1').read_bytes();self.assertEqual(data.count(b'\n'),data.count(b'\r\n'))

    def test_the_experiment_trains_detects_and_recovers_nothing(self):
        code=(ROOT/'scripts/blur_characterization.py').read_text(encoding='utf-8').split('"""',2)[2]+(ROOT/'src/failures/blur_experiment.py').read_text(encoding='utf-8').split('"""',2)[2]
        for word in ('optimizer','backward(','.train(','save(','threshold','recover','hover_async','land_async'):self.assertNotIn(word,code,word)
        self.assertIn('canonical.run_episode(env,episode,policy,arguments,None)',code);self.assertIn('await env.reset(episode)',code)


class AnalysisTests(unittest.TestCase):
    def record(self,start,success=True,task='land',**more):
        base={'episode':start,'task':task,'success':success,'acquired':True,'grounded':True,'correct_target':success,'wrong_target':False,'approached':True,
              'aligned':True,'descent_started':True,'touchdown_success':success,'stable_physical_landing':success,'system_landing':success,
              'strict_policy_zero_action':success,'terminal_as_asked':success,'collision':False,'timeout':False,'band':'near','initially_visible':True}
        return {**base,**more}

    def test_the_severity_table_counts_each_stage_over_the_flights_it_applies_to(self):
        from scripts.blur_analysis import stage_counts,by_group
        items=[self.record('a'),self.record('b',False,aligned=False,descent_started=False,timeout=True),self.record('c',task='approach'),
               self.record('d',False,task='approach',acquired=False,grounded=False,approached=False,band='far'),
               self.record('e',False,wrong_target=True,collision=True,band='far')]
        counts=stage_counts(items)
        self.assertEqual(counts['mission_success'],[2,5]);self.assertEqual(counts['acquisition'],[4,5])
        # Grounding is asked of the flights that had the object in view; the landing stages of the landings.
        self.assertEqual(counts['correct_grounding'],[4,4]);self.assertEqual(counts['alignment'],[2,3]);self.assertEqual(counts['system_landing'],[1,3])
        self.assertEqual((counts['wrong_target'][0],counts['collision'][0],counts['timeout'][0]),(1,1,1))
        data={'clean':{'items':items},'high':{'items':[dict(item,success=False) for item in items]}}
        self.assertEqual(by_group(data,lambda i:i['band'],('near','far')),{'near':{'clean':[2,3],'high':[0,3]},'far':{'clean':[0,2],'high':[0,2]}})

    def test_two_flights_of_a_start_are_counted_together(self):
        from scripts.blur_analysis import repeat_table
        flights=lambda outcomes:{'by_start':{start:{'success':value} for start,value in outcomes.items()}}
        main={'clean':flights({'a':True,'b':True,'c':False}),'high':flights({'a':True,'b':False,'c':False})}
        again={'clean':flights({'a':True,'b':False,'c':False}),'high':flights({'a':False,'b':False,'c':False})}
        table,detail=repeat_table(main,again,['a','b','c'])
        self.assertEqual({key:table['clean'][key] for key in ('2/2','1/2','0/2')},{'2/2':1,'1/2':1,'0/2':1})
        self.assertEqual({key:table['high'][key] for key in ('2/2','1/2','0/2')},{'2/2':0,'1/2':1,'0/2':2})
        self.assertEqual(detail['high'],{'a':'1/2','b':'0/2','c':'0/2'});self.assertEqual((table['clean']['flights_succeeded'],table['clean']['flights']),(3,6))

    def test_the_viewer_names_the_failure_its_kernel_and_its_sigma(self):
        from scripts.grounding_mission_view import render
        from src.mission.control import default_mission_control,mission_key
        control=default_mission_control('grounding-film');frame=np.full((256,256,3),90,dtype=np.uint8);images={'front':frame,'down':frame,'verified':True}
        telemetry=lambda failure:{'mission':{'state':'NAVIGATING','task':'land','instruction':'Find the blue landing pad and land on it.','target':None},
                                  'model_input':{'step':3,'instruction':'Find the blue landing pad and land on it.','sha256':{'front':'a'*64,'down':'b'*64}},'input_step':3,'failure':failure}
        normal=render(telemetry({'failure_enabled':False,'camera_frames_unchanged':True}),control,images)
        blurred=render(telemetry({'failure_enabled':True,'severity':'high','kernel':31,'sigma':6.0,'camera_frames_unchanged':False}),control,images)
        # The failure block of the header and the line under the frames change; the frames themselves are drawn as given.
        self.assertFalse(np.array_equal(normal[40:100,630:935],blurred[40:100,630:935]));self.assertFalse(np.array_equal(normal[1012:1034,10:600],blurred[1012:1034,10:600]))
        np.testing.assert_array_equal(blurred[754:1010,20:276],frame)
        # Between flights the header shows what the keys ask for: off until B is pressed.
        waiting={'mission':{'state':'IDLE','target':None}}
        self.assertFalse(np.array_equal(render(waiting,control,images)[40:100,630:935],render(waiting,mission_key(mission_key(control,ord('b')),ord('3')),images)[40:100,630:935]))


class SetAsideTests(unittest.TestCase):
    def test_a_flight_the_harness_decided_is_taken_out_whole_kept_and_can_be_flown_again(self):
        from scripts import blur_characterization as worker
        lines=lambda rows:''.join(json.dumps(row)+chr(10) for row in rows)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'clean';(folder/'frames/a').mkdir(parents=True);(folder/'frames/a/0000.jpg').write_bytes(b'x')
            episode=lambda name,ok:{'id':name,'success':ok,'reason':'landed','steps':2,'landed_on':'blue_pad','stable_physical_landing':True,'system_land_success':ok}
            (folder/'results.json').write_text(json.dumps({'policy':'oft','config':{},'episodes':[episode('a',False),episode('b',True)]}))
            (folder/'steps.jsonl').write_text(lines({'episode':name,'step':index} for name in 'ab' for index in range(2)))
            (folder/'extra.jsonl').write_text(lines({'episode':name,'step':index} for name in 'ab' for index in range(2)))
            (root/'flights.jsonl').write_text(lines([{'start':'a','condition':'clean','outcome':'flown','attempt':1}]))
            with patch('builtins.print'):worker.set_aside(root,'main','a','clean','the simulator did not answer the disarm call')
            self.assertEqual([item['id'] for item in json.loads((folder/'results.json').read_text())['episodes']],['b'])
            self.assertEqual({json.loads(line)['episode'] for line in (folder/'steps.jsonl').read_text().splitlines()},{'b'})
            kept=json.loads((folder/'set_aside/a.json').read_text());self.assertEqual((kept['summary']['id'],len(kept['steps']),len(kept['extra'])),('a',2,2))
            self.assertTrue((folder/'set_aside/a-frames/0000.jpg').exists())
            log=[json.loads(line) for line in (root/'flights.jsonl').read_text().splitlines()]
            self.assertEqual((log[-1]['outcome'],log[-1]['recorded_as']['stable_physical_landing'],log[-1]['recorded_as']['success']),('set_aside_as_runtime_error',True,False))
            with self.assertRaises(SystemExit),patch('builtins.print'):worker.set_aside(root,'main','a','clean','again')


if __name__=='__main__':unittest.main()
