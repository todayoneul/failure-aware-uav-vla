"""Human pilot mode: what keys and sticks become, what is recorded of them, where a flight may start, that the
recording reads like the teacher's, and how takeover mode asks the model for a decision. No simulator, no window, no model."""
import hashlib
import json
import os
import threading
import math
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def oft():
    from src.aerovla_oft.spec import load_config
    return load_config()


class ControlTests(unittest.TestCase):
    def setUp(self):
        from src.human_pilot import controls
        self.c=controls;self.oft=oft()

    def test_the_axes_are_the_policys_and_their_limits_are_its_action_bounds(self):
        limits=self.c.limits_of(self.oft);bounds=self.oft['action_bounds'];tick=self.oft['tick_s']
        self.assertEqual((limits.forward,limits.down,limits.turn),(bounds['forward_m'][1]/tick,bounds['down_m'][1]/tick,bounds['yaw_rad'][1]/tick))
        self.assertEqual(bounds['forward_m'][0],0.)      # a policy has no backward: a human's backward is never a label
        self.assertEqual(set(self.c.Inputs.__dataclass_fields__),{'forward','backward','up','down','left','right','stick_forward','stick_turn','stick_vertical'})

    def test_three_tiers_lie_far_from_a_stop_inside_the_bounds_and_the_fourth_beyond_them(self):
        stop=self.c.stop_rates(self.oft);limits=self.c.limits_of(self.oft);tiers=self.c.tiers_of(self.oft,self.c.Axes(5.,2.5,1.05))
        self.assertEqual(len(tiers),4);self.assertEqual(self.c.TIER_NAMES,('slow','normal','fast','boost'))
        for tier in tiers[:3]:
            for axis in ('forward','down','turn'):
                self.assertGreaterEqual(getattr(tier,axis),self.c.SAFE_RATIO*getattr(stop,axis),(tier,axis));self.assertLessEqual(getattr(tier,axis),getattr(limits,axis)+1e-9)
        self.assertEqual(tiers[2],limits);self.assertEqual(tiers[3],self.c.Axes(5.,2.5,1.05))
        self.assertTrue(all(getattr(tiers[3],axis)>getattr(limits,axis) for axis in ('forward','down','turn')))
        settings=json.loads((ROOT/'configs/human_pilot.json').read_text(encoding='utf-8'));self.assertEqual(settings['default_tier'],3)
        self.assertGreater(settings['boost']['forward_mps'],limits.forward);self.assertGreater(math.radians(settings['boost']['turn_degps']),limits.turn)

    def test_keys_and_sticks_ask_for_rates(self):
        Inputs=self.c.Inputs;tier=self.c.tiers_of(self.oft)[1];wanted=lambda **keys:self.c.wanted(Inputs(**keys),tier,self.oft)
        self.assertEqual(wanted(),self.c.Axes())
        self.assertEqual(wanted(forward=True).forward,tier.forward);self.assertEqual(wanted(up=True).down,-tier.down);self.assertEqual(wanted(down=True).down,tier.down)
        self.assertEqual(wanted(right=True).turn,tier.turn);self.assertEqual(wanted(left=True).turn,-tier.turn);self.assertEqual(wanted(left=True,right=True).turn,0.)
        self.assertEqual(wanted(backward=True).forward,-tier.forward);self.assertEqual(wanted(forward=True,backward=True).forward,0.)
        # A stick: nothing inside its dead zone, at least the safe floor past it, the tier's rate at full travel, both ways.
        self.assertEqual(wanted(stick_forward=.15).forward,0.);self.assertAlmostEqual(wanted(stick_forward=-1.).forward,-tier.forward)
        floor=self.c.SAFE_RATIO*self.c.stop_rates(self.oft).forward
        self.assertGreaterEqual(wanted(stick_forward=.25).forward,floor);self.assertAlmostEqual(wanted(stick_forward=1.).forward,tier.forward)
        self.assertLess(wanted(stick_vertical=1.).down,0.);self.assertGreater(wanted(stick_turn=-.9).turn*-1,0.)

    def test_the_smoother_eases_a_press_in_and_out_within_its_acceleration(self):
        smoother=self.c.Smoother(self.c.Axes(1.5,1.5,1.5));target=self.c.Axes(2.,0.,.42);previous=smoother.now
        for _ in range(40):
            now=smoother.step(target,.05)
            for axis in ('forward','down','turn'):self.assertLessEqual(abs(getattr(now,axis)-getattr(previous,axis)),1.5*.05+1e-9)
            previous=now
        self.assertEqual(smoother.now,target);self.assertEqual(smoother.stop(),self.c.Axes())

    def test_a_tick_of_commands_becomes_one_action_in_the_policys_units(self):
        label=self.c.TickLabel(self.oft)
        for _ in range(10):label.add(self.c.Axes(1.2,.6,.252),.05)
        forward,down,turn=label.close();self.assertAlmostEqual(forward,.6);self.assertAlmostEqual(down,.3);self.assertAlmostEqual(turn,.126)
        # A tick that ran long (the pictures took a while) is scaled to the nominal tick: the same rates, the same action.
        for _ in range(12):label.add(self.c.Axes(1.2,.6,.252),.05)
        self.assertEqual([round(value,6) for value in label.close()],[.6,.3,.126])
        self.assertIsNone(label.reason)
        # Past the bounds: clipped, and the tick says it is no label; a fresh tick starts from nothing.
        label.add(self.c.Axes(9.,-9.,9.),.5);self.assertEqual(label.close(),[1.,-.5,.21]);self.assertEqual(label.reason,'over_bound')
        self.assertEqual(label.close(),[0.,0.,0.]);self.assertIsNone(label.reason)
        # Backward at any moment of the tick: the label's forward is clipped to nothing and the tick is no label.
        label.add(self.c.Axes(1.,0.,.1),.4);label.add(self.c.Axes(-.6,0.,.1),.1);label.close();self.assertEqual(label.reason,'backward')
        # Faster than the bounds for a moment (slowing down from the fourth tier): no label, though the sum fits.
        label.add(self.c.Axes(3.,0.,0.),.1);label.add(self.c.Axes(1.,0.,0.),.4);self.assertAlmostEqual(label.close()[0],.7);self.assertEqual(label.reason,'over_bound')

    def test_a_label_near_a_stop_is_marked_and_a_stop_at_the_end_is_not(self):
        self.assertEqual(self.c.mark([0.,0.,0.],self.oft),'pause');self.assertEqual(self.c.mark([.05,.02,.01],self.oft),'pause')
        self.assertEqual(self.c.mark([.15,0.,0.],self.oft),'below_safe');self.assertIsNone(self.c.mark([.3,0.,0.],self.oft))
        self.assertIsNone(self.c.mark([0.,0.,0.],self.oft,terminal=True));self.assertEqual(self.c.mark([.6,0.,0.],self.oft,reason='backward'),'backward')
        # Held keys of the three tiers inside the bounds, once eased in, give labels with the margin; the fourth and backward give none.
        tiers=self.c.tiers_of(self.oft)
        for index,tier in enumerate(tiers):
            for keys in ({'forward':True},{'up':True},{'down':True},{'left':True},{'right':True},{'backward':True}):
                smoother=self.c.Smoother();label=self.c.TickLabel(self.oft)
                for _ in range(40):smoother.step(self.c.wanted(self.c.Inputs(**keys),tier,self.oft),.05)
                for _ in range(10):label.add(smoother.step(self.c.wanted(self.c.Inputs(**keys),tier,self.oft),.05),.05)
                action=label.close();reason=self.c.mark(action,self.oft,reason=label.reason)
                expected='backward' if 'backward' in keys else 'over_bound' if index==3 else None
                self.assertEqual(reason,expected,(index,keys));self.assertEqual(self.c.beyond(smoother.now,self.oft),expected is not None,(index,keys))
        # The same ratio as the stage 1b audit.
        from src.landing.stop_safety import stop_ratio
        for action in ([.3,0.,0.],[0.,.1,.02],[.05,.05,.1]):self.assertAlmostEqual(self.c.stop_ratio(action,self.oft),stop_ratio(action,self.oft))


    def test_any_flight_input_takes_over_and_a_decision_is_marked_by_who_flew_it(self):
        Inputs=self.c.Inputs
        self.assertFalse(self.c.active(Inputs()));self.assertFalse(self.c.active(Inputs(stick_turn=.15)))
        for keys in ({'forward':True},{'backward':True},{'up':True},{'down':True},{'left':True},{'right':True},{'stick_forward':-.5},{'stick_vertical':.3}):
            self.assertTrue(self.c.active(Inputs(**keys)),keys)
        self.assertIsNone(self.c.pilot_mark({'human'}));self.assertEqual(self.c.pilot_mark({'model'}),'model')
        self.assertEqual(self.c.pilot_mark({'model','human'}),'handover')


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.human_pilot import session
        from src.visual_search.episodes import load_config
        from src.visual_search.maps import load_map
        cls.s=session;cls.settings=session.load();cls.config=load_config();cls.maps={name:load_map(name) for name in ('field','lot')}

    def test_a_flight_starts_in_a_recording_layout_away_from_every_check_start(self):
        settings=self.settings;self.assertEqual(settings['layouts'],['a','b','c','d','k','l','m','n'])
        stage=json.loads((ROOT/'configs/generic_landing_stage1d.json').read_text(encoding='utf-8'));self.assertEqual(settings['layouts'],stage['layouts']['train'])
        self.assertFalse(set(settings['layouts'])&set(stage['layouts']['evaluation']))
        for scene in ('field','lot'):
            checks=self.s.check_starts(scene);self.assertGreater(len(checks),100)
            for index,(layout,target,task) in enumerate((('a','blue_cube','land'),('k','green_cylinder','land'),('n','orange_ball','approach'))):
                episode=self.s.draw(settings,self.config,self.maps[scene],layout,target,task,1000+index,checks,stamp='20261023T000000')
                self.assertEqual((episode['map'],episode['layout'],episode['target'],episode['task'],episode['pilot']),(scene,layout,target,task,'human'))
                low,high=settings['distance_bands'][settings['default_band']];self.assertTrue(low<=episode['start_distance_m']<=high+1e-6)
                self.assertEqual(episode['distance_band'],settings['default_band'])
                self.assertTrue(all(math.dist(episode['start_xy'],point)>=6. for point in checks))
                self.assertEqual(episode['id'],f'hp-20261023T000000-{target}-{task}');self.assertEqual(episode['max_ticks'],settings['max_ticks'])
                self.assertFalse(any(letter.isdigit() for letter in episode['instruction']))
        for layout in ('t','p','e'):
            with self.assertRaises(ValueError):self.s.draw(settings,self.config,self.maps['field'],layout,'blue_cube','land',1)
        with self.assertRaises(ValueError):self.s.draw(dict(settings,scenes=['lot']),self.config,self.maps['field'],'a','blue_cube','land',1)

    def test_every_band_starts_where_it_says_and_the_far_band_reaches_160_m(self):
        settings=self.settings;bands=settings['distance_bands']
        self.assertEqual(bands,{'near':[12.,44.],'mid':[44.,100.],'far':[100.,160.]});self.assertEqual(settings['default_band'],'mid')
        for scene in ('field','lot'):
            checks=self.s.check_starts(scene)
            for index,band in enumerate(bands):
                episode=self.s.draw(settings,self.config,self.maps[scene],'d','red_pad','land',2000+index,checks,band=band)
                self.assertTrue(bands[band][0]<=episode['start_distance_m']<=bands[band][1]+1e-6,(scene,band,episode['start_distance_m']))

    def test_the_plate_layouts_make_the_ball_a_landing_surface_without_changing_the_map(self):
        from src.landing.surface import SurfaceSet
        from src.visual_search.maps import load_map,scene_objects,lighting
        for scene in ('field','lot'):
            base=load_map(scene);capped=self.s.with_caps(base,self.settings)
            self.assertEqual(base,load_map(scene))      # the map itself (part of the frozen baseline's fingerprints) is untouched
            self.assertEqual(self.s.recording_layouts(self.settings)[8:],[layout+'_cap' for layout in self.settings['layouts']])
            for layout in self.settings['layouts']:
                plain,cap=SurfaceSet(capped,layout)['orange_ball'],SurfaceSet(capped,layout+'_cap')['orange_ball']
                self.assertFalse(plain.landable);self.assertTrue(cap.landable,cap.reason)
                self.assertEqual(capped['layouts'][layout+'_cap'],capped['layouts'][layout]);self.assertEqual(lighting(capped,layout+'_cap'),lighting(capped,layout))
                plates=[item for item in scene_objects(capped,layout+'_cap') if item['name'].startswith('BallCap')]
                self.assertEqual(len(plates),1);self.assertFalse(any(item['name'].startswith('BallCap') for item in scene_objects(capped,layout)))
                # Every other thing spawned is the plain layout's.
                self.assertEqual([item['name'] for item in scene_objects(capped,layout+'_cap') if not item['name'].startswith('BallCap')],[item['name'] for item in scene_objects(capped,layout)])
                # The plate rests on the ball: its underside where the ball's surface is at its edge, its top just above the ball's.
                plate=plates[0];self.assertEqual(plate['position'],list(cap.centre_xy));self.assertAlmostEqual(plate['lift_m'],9.);self.assertAlmostEqual(cap.top_m,10.2)
            episode=self.s.draw(self.settings,self.config,capped,'k_cap','orange_ball','land',5,band='near')
            self.assertEqual((episode['layout'],episode['task']),('k_cap','land'))

    def test_a_landing_is_data_only_near_the_middle_of_its_surface(self):
        from src.landing.surface import SurfaceSet
        capped=self.s.with_caps(self.maps['field'],self.settings);surfaces=SurfaceSet(capped,'a_cap')
        radius={name:self.s.precision_m(surfaces[name].describe()) for name in ('blue_pad','red_pad','blue_cube','green_cylinder','orange_ball')}
        self.assertEqual(radius,{'blue_pad':1.75,'red_pad':1.75,'blue_cube':1.75,'green_cylinder':1.75,'orange_ball':1.3})
        summary={'reason':'human_stop','hit':None,'task':'land'};pad=surfaces['blue_pad'].describe()
        landed=lambda error:{'success':True,'stable_physical_landing':True,'target_surface':pad,'horizontal_error_m':error}
        self.assertTrue(self.s.kept(landed(1.7),summary,.95,self.settings));self.assertFalse(self.s.kept(landed(1.8),summary,.95,self.settings))
        self.assertFalse(self.s.kept({**landed(1.),'stable_physical_landing':False},summary,.95,self.settings))
        # Recorded flights are judged again from their summaries alone; a takeover flight also needs a label of the pilot's.
        from scripts.refile_human_flights import verdict
        stored={**summary,'surface_landing':landed(1.),'real_time_factor':.95,'labels':{'labels':3}}
        self.assertTrue(verdict(stored,self.settings));self.assertFalse(verdict({**stored,'surface_landing':landed(2.)},self.settings))
        self.assertFalse(verdict({**stored,'pilot':'takeover','labels':{'labels':0}},self.settings));self.assertTrue(verdict({**stored,'pilot':'takeover'},self.settings))

    def test_a_landing_is_asked_only_of_an_object_with_a_top(self):
        objects={item['name']:item['landable'] for item in self.s.targets_of(self.maps['field'],'a')}
        self.assertTrue(objects['blue_cube'] and objects['blue_pad'] and objects['green_cylinder']);self.assertFalse(objects['orange_ball'] or objects['blue_cone'])
        with self.assertRaises(ValueError):self.s.draw(self.settings,self.config,self.maps['field'],'a','blue_cone','land',1)

    def test_a_recorded_flight_reads_like_the_teachers_and_marked_decisions_are_in_no_label(self):
        import numpy as np
        from scripts.build_visual_search_dataset import load_records,episode_samples
        episode={'id':'hp-test-blue_cube-land','map':'field','task':'land','target':'blue_cube','instruction':'Find the blue cube and land on it.','case':'human','split':'train'}
        with tempfile.TemporaryDirectory() as root:
            recorder=self.s.Recorder(root,episode);steps=[]
            states=['human']*8+['landed']*4;marks={2:'pause',3:'below_safe'}
            for index,state in enumerate(states):
                frame=np.full((256,256,3),index,np.uint8)
                step={'step':index,'teacher_state':state,'teacher_action':[.3,.25,0.] if state=='human' else [0.,0.,0.],'action':[.3,.25,0.],'proprio':[0.]*4,'front_seen':True,
                      'front_in_fov':True,'down_seen':False,'bearing_deg':5.,'distance_m':20.-index,'height_m':9.,'perturbed':False,'img_name':recorder.frames(index,frame,frame)}
                if index in marks:step['excluded']=marks[index]
                steps.append(step)
            summary={**episode,'reason':'human_stop','success':True}
            path=recorder.finish(summary,steps,kept=True)
            self.assertEqual(path.parent.name,'episodes');self.assertTrue((Path(root)/'field/hp-test-blue_cube-land/frontcamera/000011.png').exists())
            self.assertEqual(self.s.label_rows(steps),{'decisions':12,'labels':6,'pause':1,'below_safe':1,'backward':0,'over_bound':0,'model':0,'handover':0,'terminal':4})
            records=load_records(root);self.assertEqual(len(records),1);samples=episode_samples(records[0],4)
            used={sample['meta']['step'] for sample in samples}
            # A window holding decision 2 or 3 is left out (decisions 0-3); the stops after the touchdown are labels.
            self.assertEqual(used,set(range(4,12)));self.assertTrue(any(sample['meta']['teacher_state']=='landed' for sample in samples))
            self.assertEqual(samples[0]['instruction'],'<image>\nFind the blue cube and land on it.')
            self.assertEqual(set(samples[0]),{'traj_rel_dir','img_name','instruction','label','is_last_step','is_penultimate','chunk','proprio','meta'})
            # Not kept: under rejected/, which the builder does not read.
            other=self.s.Recorder(root,dict(episode,id='hp-test-2'));other.frames(0,frame,frame);other.finish(summary,steps[:1],kept=False,keep_rejected_frames=False)
            self.assertTrue((Path(root)/'rejected/hp-test-2.json').exists());self.assertFalse((Path(root)/'field/hp-test-2').exists());self.assertEqual(len(load_records(root)),1)

    def test_only_a_flight_that_did_what_was_asked_is_kept(self):
        good={'success':True};summary={'reason':'human_stop','hit':None};settings=self.settings
        self.assertTrue(self.s.kept(good,summary,.95,settings));self.assertTrue(self.s.kept(good,summary,None,settings))
        self.assertFalse(self.s.kept({'success':False},summary,.95,settings));self.assertFalse(self.s.kept(good,{**summary,'reason':'gave_up'},.95,settings))
        self.assertFalse(self.s.kept(good,{**summary,'hit':'Wall'},.95,settings));self.assertFalse(self.s.kept(good,summary,.7,settings))
        # A landing: the surface evaluator's own soft touchdown that stood, whatever the canonical rule read (and near the middle).
        land={**summary,'task':'land'};pad={'landable':True,'outline':'rectangle','usable_half_extent_m':[6.6,6.6]}
        self.assertFalse(self.s.kept({'success':True,'stable_physical_landing':False,'target_surface':pad,'horizontal_error_m':.5},land,.95,settings))
        self.assertTrue(self.s.kept({'success':False,'stable_physical_landing':True,'target_surface':pad,'horizontal_error_m':.5},land,.95,settings))
        # Takeover mode: the pilot may accept the model's own stop at the end.
        self.assertTrue(self.s.kept(good,{**summary,'reason':'model_stop'},.95,settings))

    def test_a_takeover_flights_model_and_handover_decisions_are_in_no_label(self):
        import numpy as np
        from scripts.build_visual_search_dataset import load_records,episode_samples
        episode={'id':'tk-test-blue_cube-land','map':'field','task':'land','target':'blue_cube','instruction':'Find the blue cube and land on it.','case':'human','split':'train'}
        with tempfile.TemporaryDirectory() as root:
            recorder=self.s.Recorder(root,episode);steps=[]
            pilots=['model']*5+['handover']+['human']*6+['human']*4
            for index,pilot in enumerate(pilots):
                frame=np.full((256,256,3),index,np.uint8);state='landed' if index>=12 else 'model' if pilot=='model' else 'human'
                step={'step':index,'teacher_state':state,'teacher_action':[.3,.25,0.],'action':[.3,.25,0.],'proprio':[0.]*4,'front_seen':True,'pilot':pilot,
                      'front_in_fov':True,'down_seen':False,'bearing_deg':5.,'distance_m':20.-index,'height_m':9.,'perturbed':False,'img_name':recorder.frames(index,frame,frame)}
                if pilot!='human':step['excluded']=pilot
                steps.append(step)
            recorder.finish({**episode,'reason':'human_stop','success':True},steps,kept=True)
            self.assertEqual(self.s.label_rows(steps),{'decisions':16,'labels':6,'pause':0,'below_safe':0,'backward':0,'over_bound':0,'model':5,'handover':1,'terminal':4})
            # Only windows of the pilot's own decisions (6 onward) are labels.
            samples=episode_samples(load_records(root)[0],4);self.assertEqual({sample['meta']['step'] for sample in samples},set(range(6,16)))


class RemoteTests(unittest.TestCase):
    """The pilot's side and the server's side of takeover mode, over a real local connection, with a stand-in for the model."""
    class Model:
        def __init__(self):self.calls=[]
        def infer(self,front,down,instruction,proprio=None):
            self.calls.append((front.copy(),down.copy(),instruction,proprio))
            return {'chunk':[[float(front[0,0,0])/100,0.,0.]]*4,'normalised_chunk':[[0.,0.,0.]]*4,'prompt':'p','inference_ms':1.,'stop':False,'proprio':proprio}

    def test_a_decision_goes_to_the_server_and_back_with_proof_of_its_pictures(self):
        import numpy as np
        from multiprocessing.connection import Listener
        from src.human_pilot import remote
        model=self.Model();key=os.urandom(16);listener=Listener(('127.0.0.1',0),authkey=key)
        server=threading.Thread(target=remote.serve,args=(model,listener,lambda text:None),daemon=True);server.start()
        with tempfile.TemporaryDirectory() as folder:
            ready=Path(folder)/'policy-ready.json';policy=remote.RemotePolicy(ready)
            self.assertEqual(policy.status(),'loading')
            ready.write_text(json.dumps({'port':listener.address[1],'key':key.hex(),'loaded':{'name':'stand-in'}}),encoding='utf-8')
            self.assertEqual(policy.status(),'ready')
            front=np.full((256,256,3),42,np.uint8);down=np.zeros((256,256,3),np.uint8);down[5,6]=[1,2,3]
            answer=policy.infer({'front':front,'down':down,'instruction':'Find the blue cube and land on it.','proprio':None})
            self.assertEqual(answer['chunk'][0],[.42,0.,0.])
            # The model was given these pictures and this sentence, nothing else; the answer says which pictures it was given.
            given=model.calls[0];self.assertTrue((given[0]==front).all() and (given[1]==down).all())
            self.assertEqual(given[2:],('Find the blue cube and land on it.',None))
            self.assertEqual(answer['input_sha256'],{'front':hashlib.sha256(front.tobytes()).hexdigest(),'down':hashlib.sha256(down.tobytes()).hexdigest()})
            policy.close()
        server.join(5);self.assertFalse(server.is_alive());listener.close()

    def test_other_pictures_or_a_failed_server_stop_the_flight(self):
        import numpy as np
        from src.human_pilot import remote
        class Connection:
            def __init__(self,answer):self.answer=answer
            def send_bytes(self,data):pass
            def recv_bytes(self):return json.dumps(self.answer).encode()
        frame=np.zeros((256,256,3),np.uint8)
        with self.assertRaisesRegex(RuntimeError,'other pictures'):remote.ask(Connection({'chunk':[],'input_sha256':{'front':'x','down':'y'}}),frame,frame,'s')
        with self.assertRaisesRegex(RuntimeError,'failed'):remote.ask(Connection({'error':'Traceback ...'}),frame,frame,'s')
        with tempfile.TemporaryDirectory() as folder:
            ready=Path(folder)/'policy-ready.json';ready.write_text(json.dumps({'error':'CUDA is required'}),encoding='utf-8')
            self.assertEqual(remote.RemotePolicy(ready).status(),'CUDA is required')


class WindowTests(unittest.TestCase):
    def test_the_display_cameras_copy_the_policys_cameras_and_leave_them_as_they_are(self):
        from scripts.human_pilot import display_cameras,VIEWS
        from src.mission.scene import prepare_mission_config
        original=(ROOT/'configs/robot_quadrotor_fastphysics.jsonc').read_text()
        with tempfile.TemporaryDirectory() as plain,tempfile.TemporaryDirectory() as output:
            before={sensor['id']:sensor for sensor in json.loads((prepare_mission_config(ROOT,plain)/'robot_quadrotor_fastphysics.jsonc').read_text())['sensors']}
            directory=display_cameras(prepare_mission_config,768)(ROOT,output);robot={sensor['id']:sensor for sensor in json.loads((directory/'robot_quadrotor_fastphysics.jsonc').read_text())['sensors']}
            # Every sensor the policy and the canonical loop use is the same; two more are added.
            self.assertEqual(set(robot)-set(before),set(VIEWS.values()))
            for name,sensor in before.items():self.assertEqual(robot[name],sensor,name)
            for source,view in (('FrontCamera',VIEWS['front']),('DownCamera',VIEWS['down'])):
                copy,policy=robot[view],before[source];rgb=next(setting for setting in policy['capture-settings'] if setting['image-type']==0)
                self.assertEqual((copy['origin'],copy['parent-link'],copy['capture-interval']),(policy['origin'],policy['parent-link'],policy['capture-interval']))
                self.assertEqual(len(copy['capture-settings']),1);setting=copy['capture-settings'][0]
                self.assertEqual((setting['width'],setting['height'],setting['fov-degrees'],setting['image-type'],setting['streaming-enabled']),(768,768,rgb['fov-degrees'],0,True))
                self.assertEqual((rgb['width'],rgb['height']),(256,256))
        self.assertEqual((ROOT/'configs/robot_quadrotor_fastphysics.jsonc').read_text(),original)

    def test_the_window_shows_the_two_pictures_the_sentence_and_the_pilots_own_commands(self):
        import inspect
        import numpy as np
        import scripts.human_pilot as pilot
        from src.human_pilot import controls
        self.assertEqual(list(inspect.signature(pilot.render).parameters),['front','down','lines','rates','limits','size','footer'])
        canvas=pilot.render(np.full((768,768,3),200,np.uint8),np.full((768,768,3),50,np.uint8),[('Find the blue cube and land on it.',(0,0,0),True)],
                            controls.Axes(1.,.2,.1),controls.limits_of(oft()),560,pilot.KEYS_FLYING)
        self.assertEqual(canvas[100+300,20+300].tolist(),[200,200,200]);self.assertEqual(canvas[100+300,20+560+20+300].tolist(),[50,50,50])
        # The policy's 256 x 256 pictures (when no stream has come yet) fill the same panel.
        self.assertEqual(pilot.render(np.full((256,256,3),90,np.uint8),None,[],size=560)[100+500,20+500].tolist(),[90,90,90])
        # More than three lines of text (the menu) push the pictures down instead of being hidden under them.
        lines=[(f'line {index}',(0,0,0),False) for index in range(5)];canvas=pilot.render(np.full((256,256,3),90,np.uint8),None,lines,size=560)
        self.assertEqual(canvas[156+5,20+5].tolist(),[90,90,90]);self.assertEqual(canvas[150,20+5].tolist(),[245,245,245])
        source=inspect.getsource(pilot.fly);status=source[source.index('lines=['):];self.assertNotIn('decision',status[:status.index(chr(10))])
        # What the flight loop puts on the window: the sentence, its own state, the decision count, the speed, recording; never
        # what the simulator knows of the object.
        source=inspect.getsource(pilot.fly);shown=source[source.index('lines=['):source.index('canvas=render(')]
        for word in ('private','privileged','distance','bearing','height_m','position','target'):self.assertNotIn(word,shown)

    def test_a_script_flies_its_segments_and_presses_once(self):
        import time
        from scripts.human_pilot import Script
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'script.json';path.write_text(json.dumps([[.2,{'forward':True}],[.2,{'press':'enter'}]]),encoding='utf-8')
            script=Script(path);self.assertEqual(script.read()[1],{'enter'})
            script.flying();inputs,pressed,_=script.read();self.assertTrue(inputs.forward);self.assertEqual(pressed,set())
            time.sleep(.25);self.assertEqual(script.read()[1],{'enter'});self.assertEqual(script.read()[1],set())
            time.sleep(.2);inputs,pressed,_=script.read();self.assertFalse(inputs.forward);self.assertEqual(pressed,set())


if __name__=='__main__':unittest.main()
