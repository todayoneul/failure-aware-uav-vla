"""Human pilot mode: what keys and sticks become, what is recorded of them, where a flight may start, and that the
recording reads like the teacher's. No simulator, no window."""
import json
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
                centre=self.maps[scene]['objects'][target];self.assertTrue(settings['distance_m'][0]<=episode['start_distance_m']<=settings['distance_m'][1]+1e-6)
                self.assertTrue(all(math.dist(episode['start_xy'],point)>=6. for point in checks))
                self.assertEqual(episode['id'],f'hp-20261023T000000-{target}-{task}');self.assertEqual(episode['max_ticks'],settings['max_ticks'])
                self.assertFalse(any(letter.isdigit() for letter in episode['instruction']))
        for layout in ('t','p','e'):
            with self.assertRaises(ValueError):self.s.draw(settings,self.config,self.maps['field'],layout,'blue_cube','land',1)
        with self.assertRaises(ValueError):self.s.draw(dict(settings,scenes=['lot']),self.config,self.maps['field'],'a','blue_cube','land',1)

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
            self.assertEqual(self.s.label_rows(steps),{'decisions':12,'labels':6,'pause':1,'below_safe':1,'backward':0,'over_bound':0,'terminal':4})
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
        # A landing: the surface evaluator's own soft touchdown that stood, whatever the canonical rule read.
        land={**summary,'task':'land'}
        self.assertFalse(self.s.kept({'success':True,'stable_physical_landing':False},land,.95,settings))
        self.assertTrue(self.s.kept({'success':False,'stable_physical_landing':True},land,.95,settings))


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
