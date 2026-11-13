"""A flight's start: the one representation, the rule a start has to meet, the planner, the keys and the map that place one,
and that none of it reaches the policy. Sessions run the interactive runner over a fake simulator; no model is loaded."""
import asyncio
import inspect
import json
import math
import re
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXAMPLE=ROOT/'configs/experiments/arbitrary_start_example.json'


def demo():
    from src.mission import semantic
    from src.visual_search.episodes import load_config
    return load_config(),semantic.load_demo()


class StartStateTests(unittest.TestCase):
    def test_one_representation_for_the_map_the_plan_and_the_log(self):
        from src.mission.start import StartState
        start=StartState(123.4,-45.6,135,10)
        self.assertEqual(start.to_dict(),{'x':123.4,'y':-45.6,'yaw_deg':135.,'height_m':10.})
        self.assertEqual(StartState.from_dict(json.loads(json.dumps(start.to_dict()))),start)
        # The fields the canonical environment reads its start from, with the same numbers.
        self.assertEqual(start.episode_fields(),{'start_xy':[123.4,-45.6],'start_yaw_deg':135.,'start_height_m':10.})
        self.assertEqual(StartState.from_episode(start.episode_fields(),6.),start)
        self.assertEqual(StartState.from_episode({'start_xy':[1,2],'start_yaw_deg':3},6.),StartState(1,2,3,6.))
        self.assertEqual(start.moved(height_m=4.).height_m,4.);self.assertEqual(start.height_m,10.)
        with self.assertRaises(Exception):start.x=0.

    def test_a_launch_of_the_map_file_is_a_start_at_the_cruise_height(self):
        from src.mission import semantic
        config,scene=demo();north=scene['mission']['launches'][0]
        start=semantic.start_of(north,config['cruise_height_m'])
        self.assertEqual(start.to_dict(),{'x':30.,'y':56.,'yaw_deg':90.,'height_m':6.})
        custom=semantic.custom_launch(start.moved(yaw_deg=270.,height_m=9.))
        self.assertEqual((custom['id'],custom['xy'],custom['yaw_deg'],custom['height_m'],custom['custom']),('custom',[30.,56.],-90.,9.,True))
        # The episode of a mission carries the start exactly, and nothing of it is in the sentence.
        target=semantic.semantic_targets(scene)[2];text=semantic.sentence(config,scene,target,'land')
        episode=semantic.mission_episode(semantic.DEMO,scene,target,'land',text,custom,1,320,config['cruise_height_m'])
        self.assertEqual((episode['start_xy'],episode['start_yaw_deg'],episode['start_height_m']),([30.,56.],-90.,9.))
        self.assertEqual(episode['instruction'],'Find the blue cube and land on it.');self.assertFalse(re.search(r'\d',episode['instruction']))

    def test_headings_and_angles(self):
        from src.mission.start import heading,wrap_deg
        self.assertEqual(heading([0.,0.],[10.,0.]),0.);self.assertEqual(heading([0.,0.],[0.,10.]),90.)
        self.assertAlmostEqual(heading([20.,75.],[10.,85.]),135.);self.assertAlmostEqual(heading([0.,0.],[-5.,-5.]),-135.)
        self.assertIsNone(heading([3.,4.],[3.,4.]))
        self.assertEqual([wrap_deg(value) for value in (0.,180.,-180.,190.,-190.,360.,725.)],[0.,180.,180.,-170.,170.,0.,5.])


class ValidatorTests(unittest.TestCase):
    def check(self,x,y,yaw=0.,height=6.,**keywords):
        from src.mission.start import StartState,validate
        config,scene=demo();return validate(StartState(x,y,yaw,height),scene,'mission',config,**keywords)

    def test_open_flat_ground_is_a_start(self):
        result=self.check(18.,72.,30.,4.)
        self.assertEqual((result['valid'],result['reasons']),(True,[]));self.assertGreater(result['facts']['clearance_m'],5.)
        for launch in demo()[1]['mission']['launches']:self.assertTrue(self.check(*launch['xy'],launch['yaw_deg'])['valid'])

    def test_inside_a_wall_or_an_object_is_not(self):
        from src.visual_search.maps import MapGeometry
        config,scene=demo();geometry=MapGeometry(scene,'mission')
        block=next(box for box in geometry.boxes if box['name'] not in geometry.objects)
        middle=[(block['min'][0]+block['max'][0])/2,(block['min'][1]+block['max'][1])/2]
        result=self.check(*middle);self.assertFalse(result['valid']);self.assertIn(f'inside {block["name"]}',result['reasons'][0])
        for point,name in (([10.,92.],'blue_pad'),([2.,66.],'blue_cube'),([55.,30.],'green_cylinder'),([91.,32.],'orange_ball')):
            result=self.check(*point);self.assertFalse(result['valid']);self.assertIn(f'inside {name}',result['reasons'][0])

    def test_too_little_clearance_is_not(self):
        # 2.5 m from a block, and 3 m from the side of the blue pad: both nearer than a planned evaluation start may be.
        result=self.check(30.,40.);self.assertFalse(result['valid'])
        self.assertRegex(result['reasons'][0],r'obstacle clearance 2\.5 m < 5\.0 m \(TemplateCube_Rounded_\d+\)')
        result=self.check(20.,92.);self.assertEqual(result['reasons'],['obstacle clearance 3.0 m < 5.0 m (blue_pad)'])
        self.assertTrue(self.check(22.1,92.)['valid'])

    def test_too_low_too_high_and_outside_the_map_are_not(self):
        config,scene=demo();low,high=.8,scene['altitude']['ceiling_m']
        self.assertEqual(self.check(18.,72.,height=.5)['reasons'],[f'height 0.5 m is below the minimum flight clearance ({low:.1f} m)'])
        self.assertEqual(self.check(18.,72.,height=14.5)['reasons'],[f'height 14.5 m is above the ceiling ({high:.1f} m)'])
        self.assertTrue(self.check(18.,72.,height=low)['valid']);self.assertTrue(self.check(18.,72.,height=high)['valid'])
        self.assertIn('outside the map area',self.check(-60.,0.)['reasons'][0]);self.assertIn('outside the map area',self.check(0.,130.)['reasons'][0])
        self.assertFalse(self.check(float('nan'),0.)['valid']);self.assertFalse(self.check(0.,0.,height=float('inf'))['valid'])

    def test_a_click_that_did_not_land_on_the_ground_is_not(self):
        # The same place on the map, clicked on the ground and "clicked" on something 6 m above it.
        self.assertTrue(self.check(18.,72.,surface_point=[18.,72.,-1.])['valid'])
        result=self.check(18.,72.,surface_point=[18.,72.,-7.]);self.assertEqual(result['reasons'],['not on the ground: the clicked surface is 6.0 m above it'])
        # The top of a pad: inside the pad, and not the ground.
        self.assertEqual(len(self.check(10.,92.,surface_point=[10.,92.,-3.5])['reasons']),2)

    def test_every_number_of_the_rule_exists_already(self):
        from src.mission.start import rule,check,StartState
        config,scene=demo();terms=rule(config,scene);limits=json.loads((ROOT/'configs/flight_limits.json').read_text(encoding='utf-8'))
        self.assertEqual(terms['clearance_m'],config['generalization']['start_margin_m'])
        self.assertEqual(terms['height_m'],[limits['minimum_clearance_m'],scene['altitude']['ceiling_m']])
        self.assertEqual(terms['default_height_m'],config['cruise_height_m'])
        # The interactive map adds two step sizes for the keys and no limit of its own.
        self.assertEqual({key for key in scene['mission']['start'] if key!='description'},{'height_step_m','yaw_step_deg'})
        with self.assertRaisesRegex(ValueError,r'INVALID START\. Reason: inside blue_pad'):check(StartState(10.,92.,0.,6.),scene,'mission',config)

    def test_the_height_keys_stop_at_the_rules_bounds(self):
        from src.mission.start import StartState,step_height
        config,scene=demo();start=StartState(18.,72.,0.,13.)
        up,message=step_height(start,1,scene,config);self.assertEqual((up.height_m,message),(14.,None))
        same,message=step_height(up,1,scene,config);self.assertEqual(same.height_m,14.);self.assertIn('ceiling is 14.0 m',message)
        low=StartState(18.,72.,0.,1.);same,message=step_height(low,-1,scene,config)
        self.assertEqual(same.height_m,1.);self.assertIn('minimum flight clearance is 0.8 m',message)

    def test_a_vehicle_that_did_not_come_to_rest_on_the_ground_is_noticed(self):
        from src.mission.start import spawn_report
        config,scene=demo()
        self.assertTrue(spawn_report(-1.1927,scene)['on_ground'])
        # The map's own spawn platform is 1.5 m above the ground; a pad's top 2.5 m.
        self.assertFalse(spawn_report(-2.69,scene)['on_ground']);self.assertFalse(spawn_report(-3.69,scene)['on_ground'])


class PlannerTests(unittest.TestCase):
    def plan(self,**changes):
        from src.mission.start import plan_starts
        config,scene=demo()
        arguments=dict(layout='mission',target='blue_pad',task='land',count=12,seed=30000,distance_m=(20.,80.));arguments.update(changes)
        return plan_starts(config,scene,**arguments)

    def test_the_same_seed_gives_the_same_starts_and_another_seed_others(self):
        first,again,other=self.plan(),self.plan(),self.plan(seed=30001)
        self.assertEqual(first,again);self.assertNotEqual([item['start_xy'] for item in first],[item['start_xy'] for item in other])
        self.assertNotEqual(first,self.plan(target='red_pad'));self.assertEqual(len(first),12)
        self.assertEqual(len({item['id'] for item in first}),12);self.assertEqual(first[0]['id'],'as-30000-000-blue_pad-land')

    def test_every_start_meets_the_rule_the_distances_and_the_separation(self):
        from src.mission.start import StartState,validate
        config,scene=demo();episodes=self.plan(count=20);centre=[10.,92.]
        for episode in episodes:
            start=StartState.from_episode(episode,config['cruise_height_m']);self.assertTrue(validate(start,scene,'mission',config)['valid'],episode['id'])
            distance=math.hypot(start.x-centre[0],start.y-centre[1]);self.assertTrue(19.99<=distance<=80.01,distance)
            self.assertAlmostEqual(distance,episode['start_distance_m'],delta=.02)
            low,high=scene['altitude']['start_m'];self.assertTrue(low<=start.height_m<=high);self.assertTrue(-180.<start.yaw_deg<=180.)
            for key in ('map','layout','target','task','instruction','strategy','seed','start_xy','start_yaw_deg','start_height_m',
                        'initially_visible','initially_invisible','occluded','requires_translation','visible_after_climb','others_in_view'):self.assertIn(key,episode)
            self.assertEqual(episode['instruction'],'Find the blue landing pad and land on it.')
        for a in episodes:
            for b in episodes:
                if a is not b:self.assertGreaterEqual(math.hypot(a['start_xy'][0]-b['start_xy'][0],a['start_xy'][1]-b['start_xy'][1]),config['generalization']['exclusion_radius_m'])
        # Arbitrary means arbitrary: headings all round, starts with the pad in view and without, some of them hidden.
        self.assertGreater(len({round(item['start_yaw_deg']/90) for item in episodes}),2)
        self.assertTrue(any(item['initially_visible'] for item in episodes));self.assertTrue(any(item['initially_invisible'] for item in episodes))

    def test_the_distance_band_is_not_limited_to_the_validated_range(self):
        config,scene=demo()
        for low,high in ((45.,60.),(60.,80.),(80.,110.)):
            episodes=self.plan(count=4,distance_m=(low,high));self.assertTrue(all(low-.01<=item['start_distance_m']<=high+.01 for item in episodes))
            self.assertTrue(all(item['start_distance_m']>config['canonical']['max_start_m'] for item in episodes))

    def test_starts_keep_away_from_the_ones_they_are_told_to_avoid(self):
        from src.mission.start import starts_of
        first=self.plan(count=6);held=[item['start_xy'] for item in first]
        # Drawn again with the first plan to avoid, none of the starts is near one of the first.
        again=self.plan(count=6,avoid=held)
        for item in again:self.assertTrue(all(math.hypot(item['start_xy'][0]-x,item['start_xy'][1]-y)>=6. for x,y in held))
        test=json.loads((ROOT/'configs/gen_v3_canonical_44m_test.json').read_text(encoding='utf-8'))
        self.assertEqual(len(starts_of(test)),48);self.assertEqual(starts_of(test,'blocks'),[]);self.assertEqual(len(starts_of({'episodes':first},'blocks')),6)

    def test_headings_can_face_the_object_or_away_from_it(self):
        self.assertTrue(all(abs(item['target_bearing_deg'])<.02 for item in self.plan(count=4,yaw='toward')))
        self.assertTrue(all(abs(abs(item['target_bearing_deg'])-180.)<.02 for item in self.plan(count=4,yaw='away')))
        self.assertTrue(all(item['start_height_m']==7.5 for item in self.plan(count=3,height_m=(7.5,7.5))))

    def test_a_landing_is_not_planned_on_an_object_without_a_surface(self):
        with self.assertRaisesRegex(ValueError,'no valid landing surface'):self.plan(target='orange_ball')
        self.assertEqual(len(self.plan(target='orange_ball',task='approach',count=3)),3)
        self.assertEqual(self.plan(target='blue_cube',count=2)[0]['instruction'],'Find the blue cube and land on it.')
        self.assertEqual(self.plan(target='green_cylinder',count=2)[0]['instruction'],'Find the green cylinder and land on it.')
        with self.assertRaisesRegex(RuntimeError,'Only'):self.plan(count=50,distance_m=(0.,6.))

    def test_the_committed_plan_is_what_its_own_arguments_give(self):
        import scripts.plan_arbitrary_starts as planner
        saved=json.loads(EXAMPLE.read_text(encoding='utf-8'))
        self.assertEqual(planner.build(saved['arguments'])['episodes'],saved['episodes'])
        self.assertEqual((saved['seed'],saved['target'],saved['task'],len(saved['episodes'])),(30000,'blue_pad','land',20))
        self.assertEqual(saved['map'],'configs/mission/grounding_film_demo.json');self.assertEqual(saved['arguments']['min_distance'],20.)
        self.assertEqual(saved['counts']['beyond_canonical_range'],sum(item['start_distance_m']>44. for item in saved['episodes']))


class ControlKeyTests(unittest.TestCase):
    def test_s_places_a_start_and_the_other_keys_of_that_mode_wait_for_it(self):
        from src.mission.control import default_mission_control,mission_key,pending_requests,request_start_pose
        state=default_mission_control('grounding-film');self.assertFalse(state['start_mode'])
        for key in '[]jk':self.assertEqual(pending_requests(mission_key(state,ord(key)),0),[])
        placing=mission_key(state,ord('s'));self.assertTrue(placing['start_mode'])
        self.assertEqual(pending_requests(placing,0)[-1][1],{'action':'start_mode','on':True})
        asked=placing
        for key in '[]jK':asked=mission_key(asked,ord(key))
        self.assertEqual([request for _,request in pending_requests(asked,1)],
                         [{'action':'start_height','step':-1},{'action':'start_height','step':1},{'action':'start_yaw','step':-1},{'action':'start_yaw','step':1}])
        done=mission_key(asked,ord('S'));self.assertFalse(done['start_mode']);self.assertEqual(pending_requests(done,5)[-1][1],{'action':'start_mode','on':False})
        # A mission, a reset or a preset ends the placing; the request itself is the one it always was.
        for key,action in (('g','start'),('r','reset'),('l','launch')):
            after=mission_key(placing,ord(key));self.assertFalse(after['start_mode']);self.assertEqual(pending_requests(after,1)[-1][1],{'action':action})
        posed=request_start_pose(placing,7,(300.,200.),(310.,210.))
        self.assertEqual(pending_requests(posed,1)[-1][1],{'action':'start_pose','frame_id':7,'pixel':[300,200],'heading_pixel':[310,210]})
        self.assertIsNone(pending_requests(request_start_pose(placing,7,(300,200)),1)[-1][1]['heading_pixel'])

    def test_the_arrows_pan_the_map_and_s_no_longer_does_for_this_policy(self):
        from src.mission.control import default_mission_control,mission_key,ARROWS
        state=default_mission_control('grounding-film')
        self.assertEqual(mission_key(state,ord('s'))['overview_pan'],[0,0])
        moves={code:mission_key(state,code)['overview_pan'] for code in ARROWS}
        self.assertEqual(sorted(moves.values()),[[-1,0],[0,-1],[0,1],[1,0]])
        for code,letter in ARROWS.items():
            if letter!='s':self.assertEqual(mission_key(state,ord(letter))['overview_pan'],moves[code])
        # The earlier Mission Control keeps W A S D, and knows no start mode.
        legacy=default_mission_control();self.assertNotIn('start_mode',legacy)
        self.assertEqual(mission_key(legacy,ord('s'))['overview_pan'],[-1,0]);self.assertEqual(mission_key(legacy,ord('w'))['overview_pan'],[1,0])
        self.assertEqual(mission_key(legacy,ord(']')),legacy)
        # A key code no one handles (a function key, as waitKeyEx reports it) changes nothing.
        self.assertEqual(mission_key(state,7340032),state)


class StartSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from tests.test_grounding_mission import SessionHarness
        asyncio.get_running_loop().slow_callback_duration=60.
        self.harness=SessionHarness()

    async def session(self,plan,scripts,clicks=None,**options):
        harness=self.harness;harness.clicks=clicks or {};harness.options=options
        feed=await harness.session(harness.operator(plan),scripts)
        self.sim=harness.sim;self.missions=harness.missions;self.records=harness.records;self.model=harness.model;self.seen=harness.seen
        return feed

    async def test_a_start_placed_on_the_map_is_flown_from_exactly_and_returned_to(self):
        from tests.fake_airsim import fly_to
        from tests.test_grounding_mission import demo as mission_demo
        pad=mission_demo()[2][0]['position']
        # Pressed at (20, 75) on the ground and dragged toward (10, 85): a heading of 135 degrees.
        clicks={(300,200):[20.,75.,-1.],(310,210):[10.,85.,-4.]}
        plan=[('IDLE','n'),('TARGET_SELECTED','s'),('TARGET_SELECTED',[{'pixel':(300,200),'heading':(310,210)}]),('TARGET_SELECTED',']]k'),
              ('TARGET_SELECTED','s'),('TARGET_SELECTED','g'),('SUCCESS','r'),('TARGET_SELECTED','t'),('TARGET_SELECTED','g'),('SUCCESS','q')]
        feed=await self.session(plan,[lambda sim:fly_to(sim,pad,descend=True),lambda sim:fly_to(sim,pad)],clicks)
        self.assertEqual([(m['target_object'],m['task'],m['success']) for m in self.missions],[('blue_pad','land',True),('blue_pad','approach',True)])
        first,second=self.records
        for record in (first,second):
            # The start is in the log in the canonical fields, and the first decision of the flight was made exactly there.
            self.assertEqual((record['start_xy'],record['start_yaw_deg'],record['start_height_m']),([20.,75.],150.,8.))
            self.assertEqual(record['start']['source'],'custom');self.assertTrue(record['start']['valid']);self.assertTrue(record['start']['applied'])
            step=record['steps'][0];self.assertEqual(step['position'][:2],[20.,75.])
            self.assertAlmostEqual(math.degrees(step['yaw_rad']),150.,places=6);self.assertAlmostEqual(step['height_m'],8.,places=6)
            self.assertEqual(record['start']['reached'],{'x':20.,'y':75.,'yaw_deg':record['start']['reached']['yaw_deg'],'height_m':record['start']['reached']['height_m']})
            self.assertLess(record['start']['reproduction']['xy_error_m'],1e-9);self.assertLess(record['start']['reproduction']['yaw_error_deg'],1e-6)
            self.assertLess(abs(record['start']['reproduction']['height_error_m']),1e-6);self.assertTrue(record['start']['spawn']['on_ground'])
            self.assertEqual(record['reproduce'],'.\\scripts\\run_grounding_film_mission_demo.ps1 -StartX 20 -StartY 75 -StartYaw 150 -StartHeight 8')
        # R put the vehicle back at the same start with the target kept: LAND, R, APPROACH from one place.
        self.assertEqual(first['steps'][0]['position'],second['steps'][0]['position'])
        self.assertTrue(first['summary']['landed']);self.assertFalse(second['summary']['landed'])
        # Nothing of the start reached the policy: two frames and the mission's sentence, in every call.
        self.assertEqual({arguments[2] for arguments,keywords in self.model.calls},{'Find the blue landing pad and land on it.','Approach the blue landing pad.'})
        for arguments,keywords in self.model.calls:
            self.assertEqual((len(arguments),keywords,arguments[3]),(4,{},None));self.assertFalse(re.search(r'\d',arguments[2]))
            for frame in arguments[:2]:self.assertEqual((type(frame),frame.shape,frame.dtype),(np.ndarray,(256,256,3),np.uint8))
        self.assertIn('start pose',first['not_sent_to_model']);self.assertIn('landing surface',first['not_sent_to_model'])
        self.assertNotIn('land_async',self.sim.calls)

    async def test_a_start_that_may_not_be_flown_is_shown_as_invalid_and_g_does_nothing(self):
        # The first click is on top of the blue pad, the second 2.5 m from a block; then L goes back to a preset.
        clicks={(100,100):[10.,92.,-3.5],(120,120):[30.,40.,-1.]}
        plan=[('IDLE','n'),('TARGET_SELECTED','s'),('TARGET_SELECTED',[{'pixel':(100,100)}]),('TARGET_SELECTED','g'),
              ('TARGET_SELECTED','s'),('TARGET_SELECTED',[{'pixel':(120,120)}]),('TARGET_SELECTED','s'),('TARGET_SELECTED','r'),('TARGET_SELECTED','l'),('TARGET_SELECTED','q')]
        feed=await self.session(plan,[lambda sim:(lambda index:[0.,0.,0.])],clicks)
        messages=[item['message'] for item in self.seen if item['message']]
        self.assertTrue(any(text.startswith('INVALID START. Reason: inside blue_pad') and 'not on the ground' in text for text in messages),messages)
        self.assertTrue(any(re.match(r'INVALID START\. Reason: obstacle clearance 2\.5 m < 5\.0 m',text) for text in messages),messages)
        # No mission flew, the model was never called, and the vehicle never left the preset it stood at.
        self.assertEqual(self.missions,[]);self.assertEqual(self.model.calls,[]);self.assertNotIn('NAVIGATING',feed.states)
        self.assertEqual(self.sim.loads,2);self.assertEqual([round(self.sim.x,1),round(self.sim.y,1)],[30.,56.])
        # After L the start is a preset again (from a start of the user's own, the first one), and valid.
        self.assertEqual((feed.telemetry['start']['source'],feed.telemetry['start']['valid'],feed.telemetry['launch']['id']),('preset',True,'north'))

    async def test_a_click_outside_start_mode_never_moves_the_start(self):
        from tests.test_grounding_mission import demo as mission_demo
        cube=mission_demo()[2][2]['position'];clicks={(200,200):cube}
        plan=[('IDLE',[{'pixel':(200,200)}]),('IDLE','[k'),('IDLE',[(200,200)]),('TARGET_SELECTED','q')]
        feed=await self.session(plan,[lambda sim:(lambda index:[0.,0.,0.])],clicks)
        self.assertTrue(any('Press S to place a start' in (item['message'] or '') for item in self.seen))
        self.assertEqual((feed.telemetry['start']['source'],feed.telemetry['start']['x'],feed.telemetry['start']['height_m']),('preset',30.,6.))
        self.assertEqual(feed.telemetry['mission']['target']['object_id'],'blue_cube');self.assertEqual(self.sim.loads,1)

    async def test_a_start_given_to_the_launcher_starts_the_session_there(self):
        from tests.fake_airsim import land_on
        from tests.test_grounding_mission import demo as mission_demo
        cube=mission_demo()[2][2];clicks={(200,200):cube['position']}
        plan=[('IDLE',[(200,200)]),('TARGET_SELECTED','g'),('SUCCESS','q')]
        feed=await self.session(plan,[lambda sim:land_on(sim,cube['position'],8.)],clicks,start_x=18.,start_y=72.,start_yaw=30.,start_height=4.)
        record=self.records[0]
        self.assertEqual((record['start_xy'],record['start_yaw_deg'],record['start_height_m'],record['start']['source']),([18.,72.],30.,4.,'custom'))
        self.assertEqual(record['steps'][0]['position'][:2],[18.,72.]);self.assertAlmostEqual(record['steps'][0]['height_m'],4.,places=6)
        # A landing on a cube, asked for by its sentence, read by the surface evaluator, ended by the finalizer.
        self.assertEqual((record['target_object'],record['task'],record['instruction']),('blue_cube','land','Find the blue cube and land on it.'))
        self.assertTrue(record['success']);self.assertEqual(record['surface_landing']['landed_on'],'blue_cube')
        self.assertEqual((record['surface_landing']['target_surface']['surface'],record['outcome']['surface']),('rectangle','rectangle'))
        self.assertEqual(record['summary']['finalizer']['state'],'DISARMED');self.assertIn('rectangle surface',record['outcome']['text'])
        self.assertEqual({arguments[2] for arguments,keywords in self.model.calls},{'Find the blue cube and land on it.'})
        self.assertTrue(all(len(arguments)==4 and not keywords and arguments[3] is None for arguments,keywords in self.model.calls))

    async def test_a_start_given_to_the_launcher_that_may_not_be_flown_stops_the_session(self):
        with self.assertRaises(SystemExit) as refused:
            await self.session([('IDLE','q')],[lambda sim:(lambda index:[0.,0.,0.])],start_x=10.,start_y=92.)
        self.assertIn('INVALID START. Reason: inside blue_pad',str(refused.exception))
        with self.assertRaises(SystemExit) as refused:
            await self.session([('IDLE','q')],[lambda sim:(lambda index:[0.,0.,0.])],start_height=20.)
        self.assertIn('above the ceiling',str(refused.exception))

    async def test_the_capped_sphere_can_be_landed_on_in_its_own_layout(self):
        from src.mission import semantic
        from tests.fake_airsim import land_on
        scene=semantic.load_demo();ball=semantic.semantic_targets(scene,'mission_cap')[6];clicks={(200,200):ball['position']}
        self.assertEqual((ball['object_id'],ball['landable'],ball['surface']['surface']),('orange_ball',True,'cap'))
        plan=[('IDLE',[(200,200)]),('TARGET_SELECTED','g'),('SUCCESS','q')]
        feed=await self.session(plan,[lambda sim:land_on(sim,ball['position'],9.9)],clicks,layout='mission_cap',start_x=70.,start_y=32.1)
        record=self.records[0];self.assertTrue(record['success'])
        self.assertEqual((record['layout'],record['instruction'],record['surface_landing']['target_surface']['surface']),('mission_cap','Find the orange ball and land on it.','cap'))
        self.assertTrue(record['reproduce'].endswith('-Layout mission_cap'))


class LeakTests(unittest.TestCase):
    def test_no_way_from_a_start_or_a_surface_to_the_policy(self):
        from src.mission import oft_runner
        self.assertEqual(list(inspect.signature(oft_runner.WatchedPolicy.infer).parameters),['self','front','down','instruction','proprio'])
        source=(ROOT/'src/mission/oft_runner.py').read_text(encoding='utf-8').split('"""',2)[2]
        self.assertEqual(len(re.findall(r'\.infer\(',source)),1)
        policy=source[source.index('class WatchedPolicy'):source.index('def watched_finalizer')]
        for word in ('start','surface','launch','height','yaw','target','semantic'):self.assertNotIn(word,policy.split('"""')[-1],word)
        # The frozen model takes no motion vector, and this work did not switch one on.
        saved=json.loads((ROOT/'configs/aerovla_oft.json').read_text(encoding='utf-8'));self.assertFalse(saved['proprio']['enabled'])
        for name in ('src/mission/start.py','src/landing/surface.py','src/landing/evaluator.py','src/landing/env.py'):
            code=(ROOT/name).read_text(encoding='utf-8').split('"""',2)[2]
            for word in ('.infer(','policy_inputs','AeroVLAOFT','proprio_vector'):self.assertNotIn(word,code,(name,word))

    def test_the_scripted_pilot_is_not_a_policy_and_only_the_flight_worker_uses_it(self):
        users=[path.relative_to(ROOT).as_posix() for folder in ('src','scripts') for path in (ROOT/folder).rglob('*.py')
               if 'ControlledPilot' in path.read_text(encoding='utf-8')]
        self.assertEqual(sorted(users),['scripts/surface_flights.py','src/landing/controlled.py'])


class ViewerStartTests(unittest.TestCase):
    def telemetry(self,**start):
        from tests.test_grounding_mission import ViewerTests
        base=ViewerTests('test_the_earlier_canvas_is_drawn_as_before').telemetry()
        base['mission'].update(state='TARGET_SELECTED');base['launch']=dict(base['launch'],index=None)
        base['start']={'x':20.,'y':75.,'yaw_deg':150.,'height_m':8.,'source':'custom','name':'Custom start','valid':True,'reasons':[],'placing':False,'applied':True}
        base['start'].update(start);return base

    def test_a_drag_gives_a_place_and_a_heading_and_a_click_only_a_place(self):
        from scripts.grounding_mission_view import StartDrag
        drag=StartDrag();self.assertIsNone(drag.release((5,5)));self.assertIsNone(drag.preview())
        drag.press((300,200));drag.move((303,202));self.assertIsNone(drag.preview())
        self.assertEqual(drag.release((303,202)),([300,200],None))
        drag.press((300,200));drag.move((340,230));self.assertEqual(drag.preview(),([300,200],[340,230]))
        # Released outside the map: the last place the pointer was over it stands for the heading.
        self.assertEqual(drag.release(None),([300,200],[340,230]));self.assertIsNone(drag.preview())
        drag.move((1,1));self.assertIsNone(drag.release())

    def test_the_mouse_on_the_map_is_a_start_and_on_a_button_is_a_button(self):
        import cv2
        from scripts.grounding_mission_view import StartDrag,start_gesture,map_pixel,MAP_RECT,INSET,BUTTONS
        meta={'width':640,'height':360};left,top,width,height=MAP_RECT;drag=StartDrag()
        # The canvas shows the 640 x 360 capture in a 900 x 506 panel: the middle of the panel is the middle of the capture.
        self.assertEqual(map_pixel(left+width//2,top+height//2,meta),[320,180]);self.assertEqual(map_pixel(left,top,meta),[0,0])
        self.assertIsNone(map_pixel(left-1,top+10,meta));self.assertIsNone(map_pixel(left+10,top+height,meta))
        # The corner the chase-camera inset covers is not the map.
        self.assertIsNone(map_pixel(left+INSET[0]//2,top+height-INSET[1]//2,meta))
        # Press, drag, release: the place pressed and the place released, in the capture's pixels.
        self.assertEqual(start_gesture(drag,cv2.EVENT_LBUTTONDOWN,left+450,top+253,meta),'taken')
        self.assertEqual(start_gesture(drag,cv2.EVENT_MOUSEMOVE,left+500,top+300,meta),'taken');self.assertIsNotNone(drag.preview())
        self.assertEqual(start_gesture(drag,cv2.EVENT_LBUTTONUP,left+520,top+310,meta),([320,180],[369,220]))
        # A press and a release in one place: the place alone.
        start_gesture(drag,cv2.EVENT_LBUTTONDOWN,left+450,top+253,meta)
        self.assertEqual(start_gesture(drag,cv2.EVENT_LBUTTONUP,left+452,top+254,meta),([320,180],None))
        # Released off the map: the heading is toward where the pointer left it.
        start_gesture(drag,cv2.EVENT_LBUTTONDOWN,left+450,top+253,meta);start_gesture(drag,cv2.EVENT_MOUSEMOVE,left+880,top+253,meta)
        start_gesture(drag,cv2.EVENT_MOUSEMOVE,left+width+60,top+253,meta)
        self.assertEqual(start_gesture(drag,cv2.EVENT_LBUTTONUP,left+width+60,top+253,meta),([320,180],[625,180]))
        # A press on a button is left to the buttons, and begins nothing.
        button=BUTTONS[0];self.assertIsNone(start_gesture(drag,cv2.EVENT_LBUTTONDOWN,button[0]+5,button[1]+5,meta))
        self.assertEqual(start_gesture(drag,cv2.EVENT_LBUTTONUP,button[0]+5,button[1]+5,meta),'taken');self.assertIsNone(drag.preview())
        self.assertIsNone(start_gesture(drag,cv2.EVENT_RBUTTONDOWN,left+450,top+253,meta))

    def test_the_canvas_shows_the_start_its_mode_and_why_g_is_off(self):
        from scripts.grounding_mission_view import render,annotate_map,BUTTONS,start_lines,START,INVALID
        from src.mission.control import default_mission_control,mission_key
        control=default_mission_control('grounding-film');frame=np.full((256,256,3),90,dtype=np.uint8)
        images={'front':frame,'down':frame,'chase':np.zeros((360,640,3),np.uint8),'chase_cam':np.zeros((360,640,3),np.uint8),'verified':True}
        keys=[key for *_,key,_ in BUTTONS];self.assertEqual(len(set(keys)),len(keys));self.assertIn(ord('s'),keys)
        self.assertLess(max(left+width for left,_,width,*_ in BUTTONS),1480)
        go=next(item for item in BUTTONS if item[4]==ord('g'));place=next(item for item in BUTTONS if item[4]==ord('s'))
        colour=lambda canvas,button:tuple(int(value) for value in canvas[button[1]+3,button[0]+3])
        valid=render(self.telemetry(),control,images);invalid=render(self.telemetry(valid=False,reasons=['obstacle clearance 2.5 m < 5.0 m (a block)']),control,images)
        self.assertNotEqual(colour(valid,go),(205,205,205));self.assertEqual(colour(invalid,go),(205,205,205))
        placing=render(self.telemetry(placing=True),mission_key(control,ord('s')),images)
        self.assertNotEqual(colour(placing,place),colour(valid,place));self.assertEqual(placing.shape,(1040,1480,3))
        for telemetry in (self.telemetry(applied=False),self.telemetry(source='preset'),{}):self.assertEqual(render(telemetry,control,images).shape,(1040,1480,3))
        self.assertEqual(start_lines(self.telemetry()['start']),['START','Height 8.0 m','Yaw +150 deg'])
        # On the map: a marker at the start, in its own colour, red when the start may not be flown; and the arrow of a drag.
        packet={'camera':{'width':640,'height':360,'fov_deg':80,'position':[30.,60.,-200.],'orientation':[0.,-math.sin(math.pi/4),0.,math.cos(math.pi/4)]}}
        blank=np.zeros((360,640,3),np.uint8);marked=annotate_map(blank.copy(),packet,self.telemetry());refused=annotate_map(blank.copy(),packet,self.telemetry(valid=False))
        self.assertTrue((marked==np.array(START,dtype=np.uint8)).all(axis=2).any());self.assertTrue((refused==np.array(INVALID,dtype=np.uint8)).all(axis=2).any())
        self.assertFalse((refused==np.array(START,dtype=np.uint8)).all(axis=2).any())
        dragged=annotate_map(blank.copy(),packet,{},drag=([300,200],[340,230]));self.assertTrue(dragged.any())

    def test_the_surface_of_the_selected_object_is_in_the_evaluators_panel_only(self):
        from scripts.grounding_mission_view import render
        from src.mission import semantic
        from src.mission.control import default_mission_control
        control=default_mission_control('grounding-film');targets={item['object_id']:item for item in semantic.semantic_targets(semantic.load_demo())}
        for name in ('green_cylinder','blue_cube','orange_ball'):
            telemetry=self.telemetry();target=targets[name]
            telemetry['mission']['target']=dict(target,surface_position=target['position'],launch_distance_m=30.,launch_bearing_deg=10.,in_canonical_range=True)
            self.assertEqual(render(telemetry,control,{}).shape,(1040,1480,3))
        self.assertEqual((targets['green_cylinder']['surface']['surface'],targets['green_cylinder']['surface']['usable_radius_m']),('circle',3.6))


class LauncherTests(unittest.TestCase):
    def test_the_launchers_take_a_start_and_check_it_before_the_simulator(self):
        blur=(ROOT/'scripts/run_blur_demo.ps1').read_text(encoding='utf-8')
        for name in ('run_blur_demo.ps1','run_grounding_film_mission_demo.ps1','run_mission_demo.ps1'):
            text=(ROOT/'scripts'/name).read_text(encoding='utf-8')
            for parameter in ('[Nullable[double]]$StartX=$null','[Nullable[double]]$StartY=$null','[Nullable[double]]$StartYaw=$null','[Nullable[double]]$StartHeight=$null',"[string]$Layout=''"):
                self.assertIn(parameter,text,(name,parameter))
        self.assertLess(blur.index("'mission_start.py'"),blur.index('$taskSim=Start-Process'))
        self.assertIn("'--start-x',$StartX",blur);self.assertIn('$taskRunnerArgs+=$taskStartArgs',blur)
        self.assertIn("are for -Policy grounding-film",blur);self.assertIn("are for -Policy grounding-film",(ROOT/'scripts/run_mission_demo.ps1').read_text(encoding='utf-8'))
        for path in ('scripts/run_blur_demo.ps1','scripts/run_mission_demo.ps1','scripts/run_grounding_film_mission_demo.ps1','scripts/run_surface_flights.ps1'):
            data=(ROOT/path).read_bytes();self.assertEqual(data.count(b'\n'),data.count(b'\r\n'),path)

    def test_the_flight_worker_verifies_the_frozen_baselines_and_trains_nothing(self):
        launcher=(ROOT/'scripts/run_surface_flights.ps1').read_text(encoding='utf-8')
        self.assertIn("@('grounding_film','gen_v2')",launcher);self.assertEqual(launcher.count('Confirm-FrozenBaselines "'),2)
        for name in ('scripts/surface_flights.py','scripts/plan_arbitrary_starts.py','src/landing/controlled.py','src/mission/start.py'):
            code=(ROOT/name).read_text(encoding='utf-8')
            for word in ('optimizer','backward(','.train(','save_pretrained','torch.save'):self.assertNotIn(word,code,(name,word))


if __name__=='__main__':unittest.main()
