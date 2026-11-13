import copy
import json
import math
import unittest
from unittest.mock import AsyncMock,Mock,patch
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


class MissionGeometryTests(unittest.TestCase):
    def test_numpy_quaternion_projection_matches_ned_camera_rotation(self):
        from src.mission.geometry import rotation_matrix
        q=Rotation.from_euler('xyz',[0,-90,-45],degrees=True).as_quat()
        np.testing.assert_allclose(rotation_matrix(q),Rotation.from_quat(q).as_matrix(),atol=1e-8)
    def test_depth_projection_round_trip_with_actual_camera_axes(self):
        from src.mission.geometry import pixel_to_world, world_to_pixel
        q = Rotation.from_euler('xyz', [0,-90,0], degrees=True).as_quat()
        meta = {'width':640,'height':480,'fov_deg':80,'position':[3.1,4,-30], 'orientation':q.tolist()}
        point = pixel_to_world((360,280), 27.5, meta)
        np.testing.assert_allclose(world_to_pixel(point,meta), [360,280], atol=1e-6)
        self.assertAlmostEqual(point[2], -2.5, places=6)
        for depth in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): pixel_to_world((360,280),depth,meta)

    def test_overview_does_not_change_ai_cameras_or_physics(self):
        from src.mission.geometry import mission_robot_config
        base = json.loads((ROOT/'configs/robot_quadrotor_fastphysics.jsonc').read_text())
        saved = copy.deepcopy(base)
        changed = mission_robot_config(base)
        self.assertEqual(base,saved)
        for original in (s for s in base['sensors'] if s['id'] in ('FrontCamera','DownCamera')):
            debug=next(s for s in changed['sensors'] if s['id']==original['id'])
            self.assertEqual(debug['capture-settings'][0],original['capture-settings'][0])
            self.assertEqual({k:v for k,v in debug.items() if k!='capture-settings'},
                             {k:v for k,v in original.items() if k!='capture-settings'})
            self.assertEqual(debug['capture-settings'][1]['image-type'],1)
        for key in base:
            if key!='sensors': self.assertEqual(base[key],changed[key])
        self.assertEqual(sum(s['id']=='Overview' for s in changed['sensors']),1)

    def test_flat_depth_surface_and_invalid_sky_selection(self):
        from src.mission.geometry import select_surface
        q=Rotation.from_euler('xyz',[0,-90,0],degrees=True).as_quat()
        meta={'width':640,'height':360,'fov_deg':80,'position':[3.1,4,-30],'orientation':q.tolist()}
        depth=np.full((360,640),27.5,dtype=np.float32)
        self.assertAlmostEqual(select_surface((320,180),depth,meta)[2],-2.5)
        depth[180,320]=0
        with self.assertRaises(ValueError):select_surface((320,180),depth,meta)

    def test_selection_rejects_expired_frame_and_uses_captured_pose(self):
        from src.mission.scene import OverviewScene
        q=Rotation.from_euler('xyz',[0,-90,0],degrees=True).as_quat()
        meta={'width':640,'height':360,'fov_deg':80,'position':[3.1,4,-30],'orientation':q.tolist()}
        scene=object.__new__(OverviewScene)
        scene.frames={42:(np.full((360,640),27.5,dtype=np.float32),meta)}
        with self.assertRaises(ValueError):scene.select(41,(320,180))
        point,source=scene.select(42,(320,180))
        np.testing.assert_allclose(point,[3.1,4,-2.5],atol=1e-6)
        self.assertIn('frame 42',source)

    def test_clicked_object_is_described_by_its_kind_and_map_colour(self):
        from src.mission.scene import OverviewScene
        from src.mission.landmarks import describe_surface,color_name
        def box(name,center,size):return {'name':name,'bbox':{'center':dict(zip('xyz',center)),'size':dict(zip('xyz',size))}}
        records=[box('Ground',[0,0,.5],[2000,2000,1]),box('TemplateCube_Rounded_84',[85,2.5,-3.5],[10,10,5]),
                 box('TemplateCube_Rounded_96',[95,2.5,-13.5],[10,10,5]),box('Cylinder4',[19.9,-96,-2.5],[1,1,3]),
                 {'name':'Broken','error':'no box'}]
        gray,orange,shadow=(118,118,112),(30,120,235),(40,25,20)
        self.assertEqual([color_name(c) for c in (gray,orange,shadow,(200,80,30),(60,200,60),(250,250,250))],
                         ['gray','orange',None,'blue','green','white'])
        roof=describe_surface(records,[96,3,-16.],gray)
        self.assertEqual(roof,{'object':'TemplateCube_Rounded_96','noun':'the gray block','description':'The target is the top of a gray block.'})
        self.assertEqual(describe_surface(records,[84,1,-6.],orange)['description'],'The target is the top of an orange block.')
        self.assertEqual(describe_surface(records,[19.9,-96,-4.],shadow)['noun'],'the cylinder')
        # Plain ground and the side of a block (not its top) have no words.
        self.assertIsNone(describe_surface(records,[40,40,0.],gray));self.assertIsNone(describe_surface(records,[96,3,-12.],gray))
        scene=object.__new__(OverviewScene);scene.records=records
        image=np.zeros((360,640,3),dtype=np.uint8);image[180,320]=gray
        scene.frames={42:(np.zeros((360,640),dtype=np.float32),{},image)}
        self.assertEqual(scene.describe(42,(320,180),[96,3,-16.])['noun'],'the gray block')
        self.assertIsNone(scene.describe(41,(320,180),[96,3,-16.]))


class MissionManagerTests(unittest.TestCase):
    def state(self, x=0, y=0, z=-4, speed=0):
        return {'position':[x,y,z], 'orientation':[0,0,0,1], 'velocity':[speed,0,0]}

    def started(self,target=(40,0,-2.5),**limits):
        from src.mission.manager import MissionManager,MissionTarget
        manager=MissionManager(**limits)
        manager.select(MissionTarget('surface',target,'depth frame'))
        manager.start(self.state(),now=0)
        return manager

    def test_reaching_the_target_never_ends_navigation_without_a_model_stop(self):
        from src.mission.manager import MissionManager, MissionTarget
        manager=MissionManager()
        self.assertEqual(manager.status,'IDLE')
        manager.select(MissionTarget('surface',(40,0,-2.5),'depth frame'))
        self.assertEqual(manager.status,'TARGET_SELECTED')
        manager.start(self.state(),now=0)
        self.assertEqual(manager.status,'NAVIGATING')
        # Passing directly over the target is not a stop decision.
        manager.record_step(self.state(),self.state(x=40),stop=False)
        manager.observe(self.state(x=40),now=1)
        self.assertEqual(manager.status,'NAVIGATING')
        self.assertEqual(manager.snapshot()['minimum_distance_m'],0)
        manager.abort();manager.reset()
        self.assertEqual(manager.status,'IDLE');self.assertIsNone(manager.target)

    def test_model_stop_is_terminal_and_scored_by_radius(self):
        manager=self.started(success_radius_m=20)
        manager.record_step(self.state(),self.state(x=22),stop=True)
        self.assertEqual(manager.status,'LANDING')
        self.assertAlmostEqual(manager.stop['distance_m'],18)
        manager.finish_stop(landed=True)
        self.assertEqual(manager.status,'SUCCESS');self.assertTrue(manager.stop['landed'])
        manager.fail('cleanup issue');self.assertEqual(manager.status,'SUCCESS')
        # The same stop is a failure under a stricter radius, and a failed landing does not rescue or veto it.
        strict=self.started(success_radius_m=10)
        strict.record_step(self.state(),self.state(x=22),stop=True);strict.finish_stop(landed=False)
        self.assertEqual(strict.status,'FAILED');self.assertIn('stopped_outside_radius',strict.reason)
        inside=self.started(success_radius_m=20)
        inside.record_step(self.state(),self.state(x=22),stop=True);inside.finish_stop(landed=False)
        self.assertEqual(inside.status,'SUCCESS');self.assertFalse(inside.stop['landed'])

    def test_step_budget_stuck_and_diverging_rules(self):
        budget=self.started(max_steps=2)
        budget.record_step(self.state(),self.state(x=5),stop=False)
        self.assertEqual(budget.status,'NAVIGATING')
        budget.record_step(self.state(x=5),self.state(x=10),stop=False)
        self.assertEqual(budget.status,'FAILED');self.assertEqual(budget.reason,'max_steps')
        stuck=self.started(stuck_steps=3)
        for _ in range(3):stuck.record_step(self.state(),self.state(),stop=False)
        self.assertEqual(stuck.status,'NAVIGATING')
        stuck.record_step(self.state(),self.state(),stop=False)
        self.assertEqual(stuck.reason,'stuck')
        away=self.started(diverging_steps=3)
        for x in (-5,-10):away.record_step(self.state(x=x+5),self.state(x=x),stop=False)
        self.assertEqual(away.status,'NAVIGATING')
        away.record_step(self.state(x=-10),self.state(x=-15),stop=False)
        self.assertEqual(away.reason,'diverging')
        # Turning in place between retreats is not a monotonic retreat.
        search=self.started(diverging_steps=3)
        for before,after in ((0,-5),(-5,-5),(-5,-10),(-10,-15)):
            search.record_step(self.state(x=before),self.state(x=after),stop=False)
        self.assertEqual(search.status,'NAVIGATING')

    def test_timeout_abort_and_distance(self):
        from src.mission.manager import MissionManager,MissionTarget
        manager=MissionManager(max_duration=10)
        manager.select(MissionTarget('surface',(3,4,-2.5),'depth frame'))
        manager.start(self.state(),now=0)
        self.assertEqual(manager.metrics(self.state())['horizontal_m'],5)
        manager.observe(self.state(),now=10)
        self.assertEqual(manager.status,'FAILED');self.assertEqual(manager.reason,'timeout')
        manager.reset();manager.select(MissionTarget('surface',(2,0,-2.5),'depth frame'))
        manager.start(self.state(),now=0);manager.abort()
        self.assertEqual(manager.status,'ABORTED')


class MissionControlTests(unittest.TestCase):
    def test_map_mouse_coordinates_use_original_rgb_resolution(self):
        from scripts.mission_viewer import map_pixel,render_canvas,MAP_RECT
        meta={'width':640,'height':360}
        left,top,width,height=MAP_RECT
        self.assertEqual(map_pixel(left+width//2,top+height//2,meta),[320,180])
        self.assertIsNone(map_pixel(left-1,top+height//2,meta))
        telemetry={'mission':{'state':'TARGET_SELECTED','errors':None,'success_radius_m':20.,
                             'stop':{'step':3,'distance_m':4.2,'within_radius':True},
                             'target':{'name':'Blue cone','surface_position':[1,2,-2.5]}},'phase':'Select target'}
        from src.mission.control import default_mission_control
        canvas=render_canvas(telemetry,default_mission_control(),{})
        self.assertEqual(canvas.shape,(1040,1480,3))

    def test_click_then_start_before_poll_preserves_both_requests(self):
        from src.mission.control import default_mission_control,request_selection,mission_key,pending_requests
        clicked=request_selection(default_mission_control(),42,(320,180))
        started=mission_key(clicked,ord('g'))
        pending=pending_requests(started,0)
        self.assertEqual([r['action'] for _,r in pending],['select','start'])
        self.assertEqual(pending[0][1]['frame_id'],42)
        self.assertEqual(len(pending_requests(started,1)),1)

    def test_landmark_and_hint_keys(self):
        from src.mission.control import default_mission_control,mission_key,pending_requests
        control=default_mission_control()
        self.assertTrue(control['direction_hint']);self.assertEqual(control['prompt_mode'],'hint')
        without=mission_key(control,ord('m'))
        self.assertFalse(without['direction_hint']);self.assertEqual(without['mission_request_id'],0)
        # M cycles hint -> description only -> free-form instruction -> hint.
        spoken=mission_key(without,ord('M'))
        self.assertEqual([without['prompt_mode'],spoken['prompt_mode']],['description','instruction'])
        self.assertFalse(spoken['direction_hint'])
        again=mission_key(spoken,ord('m'))
        self.assertEqual(again['prompt_mode'],'hint');self.assertTrue(again['direction_hint'])
        chosen=mission_key(without,ord('n'))
        self.assertEqual([r['action'] for _,r in pending_requests(chosen,0)],['landmark'])
        # The former hover/land mission keys are no longer mission requests.
        for key in 'hl':self.assertEqual(mission_key(control,ord(key))['mission_request_id'],0)

    def test_landmarks_use_scene_boxes_and_instruction_carries_their_description(self):
        from src.mission.landmarks import load_landmarks,landmark_at,instruction_for,GENERIC_DESCRIPTION
        from src.mission.manager import MissionTarget
        def box(name,center,size):return {'name':name,'bbox':{'center':dict(zip('xyz',center)),'size':dict(zip('xyz',size))}}
        records=[box('Cone_5',[91.4,-35.4,-6],[10,10,10]),box('OrangeBall',[91.15,32.1,-5.7],[10,10,10]),
                 {'name':'Broken','error':'no box'}]
        landmarks=load_landmarks(ROOT/'configs/landmarks.json',records)
        # The colored wall's cubes are absent from this scene, so it is not offered.
        self.assertEqual([item['id'] for item in landmarks],['blue_cone','orange_ball'])
        np.testing.assert_allclose(landmarks[0]['position'],[91.4,-35.4,-11])
        self.assertEqual(landmark_at(landmarks,[95,-33,-3])['id'],'blue_cone')
        self.assertIsNone(landmark_at(landmarks,[60,0,-1]))
        # Seen from above the colored wall is part of a roof, so a click there is a surface, not the wall.
        wall={'id':'colored_wall','minimum':[80,-12.5,-16],'maximum':[90,7.5,-6],'click':False}
        self.assertIsNone(landmark_at([wall],[85,0,-16]));self.assertEqual(landmark_at([{**wall,'click':True}],[85,0,-16])['id'],'colored_wall')
        self.assertFalse(next(item for item in json.loads((ROOT/'configs/landmarks.json').read_text(encoding='utf-8'))
                              if item['id']=='colored_wall')['click'])
        self.assertEqual(landmarks[0]['noun'],'the large blue cone')
        cone=MissionTarget('Blue cone',tuple(landmarks[0]['position']),'box',landmarks[0]['description'],'blue_cone')
        self.assertIn('large blue cone',instruction_for(cone))
        self.assertIn(GENERIC_DESCRIPTION,instruction_for(MissionTarget('surface',(1,2,-2.5),'depth')))
        for text in (instruction_for(cone),instruction_for(None)):
            self.assertIn('degrees from you.',text);self.assertIn(' Please control',text)

    def test_selection_and_mission_keys_do_not_change_blur_state(self):
        from src.mission.control import default_mission_control,request_selection,mission_key
        control=default_mission_control()
        selected=request_selection(control,4,(300,250))
        self.assertEqual(selected['mission_request']['frame_id'],4)
        self.assertEqual(selected['revision'],0)
        started=mission_key(selected,ord('g'))
        self.assertEqual(started['mission_request'],{'action':'start'})
        blurred=mission_key(started,ord('b'))
        self.assertTrue(blurred['enabled'])
        self.assertEqual(blurred['mission_request'],started['mission_request'])
        reset=mission_key(blurred,ord('r'))
        self.assertTrue(reset['enabled'])
        self.assertEqual(reset['mission_request']['action'],'reset')


class MissionFlightTests(unittest.IsolatedAsyncioTestCase):
    async def test_touchdown_hold_restarts_after_drift_or_nonfinite_velocity(self):
        from src.mission.flight import land_and_confirm
        def sample(x=0,speed=0):
            return {'pose':{'position':dict(x=x,y=0,z=-2.69)},
                    'twist':{'linear':dict(x=speed,y=0,z=0),'angular':dict(x=0,y=0,z=0)}}
        for interruption in (sample(x=.03),sample(speed=float('nan'))):
            drone=Mock();response=AsyncMock(return_value=True)
            drone.land_async=AsyncMock(return_value=response())
            drone.get_landed_state.side_effect=[1,0]
            drone.get_ground_truth_kinematics.side_effect=[sample(),interruption,*[sample(x=.03) for _ in range(4)]]
            with patch('src.mission.flight.monotonic',side_effect=[0,0,.5,.6,.9,1.5,1.6]), \
                 patch('src.mission.flight.asyncio.sleep',new=AsyncMock()):
                result=await land_and_confirm(drone)
            self.assertEqual(drone.get_ground_truth_kinematics.call_count,6)
            self.assertEqual(result['stable_seconds'],1)

    async def test_post_disarm_flying_flag_is_a_cleanup_failure(self):
        from src.mission.flight import land_and_confirm
        drone=Mock();response=AsyncMock(return_value=True)
        drone.land_async=AsyncMock(return_value=response())
        drone.get_landed_state.side_effect=[0,1]
        with patch('src.mission.flight.asyncio.sleep',new=AsyncMock()):
            with self.assertRaisesRegex(RuntimeError,'Post-disarm'):await land_and_confirm(drone)

    async def test_simpleflight_touchdown_requires_rpc_success_and_stationary_kinematics(self):
        from src.mission.flight import land_and_confirm
        for speed in (0,1):
            drone=Mock();response=AsyncMock(return_value=True)
            drone.land_async=AsyncMock(return_value=response())
            drone.get_landed_state.side_effect=[1,0]
            drone.get_ground_truth_kinematics.return_value={
                'pose':{'position':dict(x=0,y=0,z=-2.69)},
                'twist':{'linear':dict(x=speed,y=0,z=0),'angular':dict(x=0,y=0,z=0)}}
            with patch('src.mission.flight.asyncio.sleep',new=AsyncMock()):
                if speed:
                    with self.assertRaises(RuntimeError):await land_and_confirm(drone,confirmation_seconds=0,stable_seconds=0)
                    drone.disarm.assert_not_called()
                else:
                    result=await land_and_confirm(drone,confirmation_seconds=0,stable_seconds=0)
                    self.assertEqual(result['landed_state_before_disarm'],1)
                    self.assertEqual(result['landed_state'],0);drone.disarm.assert_called_once()

    async def test_collision_or_deadline_after_inference_prevents_flight_command(self):
        from src.mission.flight import guarded_execute
        from src.mission.manager import MissionManager,MissionTarget
        state={'position':[0,0,-4],'orientation':[0,0,0,1],'velocity':[0,0,0]}
        for events,now,reason in (([{'time_stamp':11}],1,'collision'),([],10,'timeout')):
            manager=MissionManager(max_duration=10)
            manager.select(MissionTarget('surface',(2,0,-2.5),'depth'))
            manager.start(state,now=0)
            with patch('src.mission.flight.execute_action',new=AsyncMock()) as execute:
                result=await guarded_execute(Mock(),manager,state,events,10,{},now=now)
                execute.assert_not_called();self.assertIsNone(result)
                self.assertEqual(manager.reason,reason)

    async def test_selected_roof_is_approached_from_above_and_ground_targets_are_not(self):
        from src.mission.flight import approach_altitude,climb_to
        from src.integration.projectairsim_action_adapter import flight_limits
        limits=flight_limits()
        # Roof 16 m up, vehicle at 4 m and 40 m away: rise to 6 m above the roof.
        self.assertAlmostEqual(approach_altitude(-4.,-16.,-2.5,limits,40.),-22.)
        # Still far away: keep flying low, the climb comes within 45 m.
        self.assertIsNone(approach_altitude(-4.,-16.,-2.5,limits,96.))
        self.assertAlmostEqual(approach_altitude(-4.,-16.,-2.5,limits,45.),-22.)
        # Ground or platform already below the vehicle: no climb.
        self.assertIsNone(approach_altitude(-1.3,0.,0.,limits,5.));self.assertIsNone(approach_altitude(-4.,-2.5,-2.5,limits,5.))
        # Already above the roof; the flight ceiling caps the climb; zero switches the rule off.
        self.assertIsNone(approach_altitude(-22.,-16.,-2.5,limits,10.))
        self.assertAlmostEqual(approach_altitude(-4.,-30.,-2.5,limits,10.),-32.5)
        self.assertIsNone(approach_altitude(-4.,-16.,-2.5,flight_limits({'target_clearance_m':0.}),10.))
        with self.assertRaises(ValueError):flight_limits({'target_clearance_m':-1.})
        with self.assertRaises(ValueError):flight_limits({'approach_distance_m':0.})
        drone=Mock();calls=[]
        async def accepted():return True
        async def climb(*arguments,**keywords):
            calls.append((arguments,keywords));return accepted()
        drone.move_by_velocity_z_async=climb;drone.hover_async=AsyncMock(side_effect=lambda:accepted())
        drone.get_ground_truth_kinematics.return_value={'pose':{'position':dict(x=0,y=0,z=-4.)}}
        result=await climb_to(drone,-22.,2.)
        self.assertEqual(calls[0][0],(0.,0.,-22.));self.assertAlmostEqual(calls[0][1]['duration'],10.)
        self.assertEqual(result,{'from_z':-4.,'to_z':-22.})

    def test_collision_on_the_last_budgeted_step_is_a_collision_not_a_step_limit(self):
        from src.mission.manager import MissionManager,MissionTarget
        def last_step(collided):
            manager=MissionManager(max_steps=2)
            manager.select(MissionTarget('Blue cone',(50,0,-11.),'box'))
            pose=lambda x:{'position':[x,0,-2.],'orientation':[0,0,0,1]}
            manager.start(pose(0),now=0);manager.record_step(pose(0),pose(5),False)
            manager.record_step(pose(5),pose(10),False,collided);return manager
        self.assertEqual(last_step(False).reason,'max_steps')
        hit=last_step(True)
        self.assertEqual((hit.status,hit.reason,hit.steps),('FAILED','collision',2))
        # A stop decided in the same step does not turn a collision into a landing.
        manager=MissionManager();manager.select(MissionTarget('Blue cone',(50,0,-11.),'box'))
        manager.start({'position':[0,0,-2.],'orientation':[0,0,0,1]},now=0)
        manager.record_step({'position':[0,0,-2.]},{'position':[5,0,-2.]},True,True)
        self.assertEqual(manager.reason,'collision');self.assertIsNone(manager.stop)

    def test_stop_record_says_whether_the_vehicle_rests_on_the_selected_surface(self):
        from src.mission.manager import MissionManager,MissionTarget
        def landed_at(z):
            manager=MissionManager()
            manager.select(MissionTarget('Gray block top',(10,0,-16.),'depth'))
            start={'position':[0,0,-22.],'orientation':[0,0,0,1]};manager.start(start,now=0)
            manager.record_step(start,{'position':[8,0,-22.],'orientation':[0,0,0,1]},True)
            manager.finish_stop(landed=True,touchdown_z=z);return manager
        roof=landed_at(-16.25);street=landed_at(-.2)
        self.assertTrue(roof.stop['on_target_surface']);self.assertFalse(street.stop['on_target_surface'])
        self.assertAlmostEqual(street.stop['below_target_surface_m'],15.8)
        # The success rule is unchanged: both stopped 2 m away horizontally.
        self.assertEqual([roof.status,street.status],['SUCCESS','SUCCESS'])
        self.assertEqual(roof.snapshot()['target']['noun'],None)

    async def test_high_landing_descends_first_and_low_landing_does_not(self):
        from src.mission.flight import land_and_confirm
        for z,expected in ((-13.,True),(-2.9,False)):
            drone=Mock();calls=[]
            async def accepted():return True
            async def descend(*arguments,**keywords):
                calls.append((arguments,keywords));return accepted()
            drone.move_by_velocity_z_async=descend
            drone.hover_async=AsyncMock(side_effect=lambda:accepted())
            drone.land_async=AsyncMock(side_effect=lambda **keywords:accepted())
            drone.get_landed_state.side_effect=[0,0]
            drone.get_ground_truth_kinematics.return_value={'pose':{'position':dict(x=0,y=0,z=z)},
                'twist':{'linear':dict(x=0,y=0,z=0),'angular':dict(x=0,y=0,z=0)}}
            with patch('src.mission.flight.asyncio.sleep',new=AsyncMock()):
                result=await land_and_confirm(drone,surface_z=-1.)
            self.assertEqual(bool(calls),expected)
            if expected:
                self.assertAlmostEqual(calls[0][0][2],-2.5);self.assertEqual(result['pre_descent']['surface_z'],-1.)
            else:self.assertIsNone(result['pre_descent'])

    async def test_landing_does_not_disarm_an_airborne_drone(self):
        from src.mission.flight import land_and_confirm
        drone=Mock();response=AsyncMock(return_value=False)
        drone.land_async=AsyncMock(return_value=response())
        drone.get_landed_state.return_value=1
        with self.assertRaises(RuntimeError):await land_and_confirm(drone,confirmation_seconds=0)
        drone.disarm.assert_not_called()

    async def test_false_return_can_only_be_accepted_if_actual_landed(self):
        from src.mission.flight import land_and_confirm
        drone=Mock();response=AsyncMock(return_value=False)
        drone.land_async=AsyncMock(return_value=response())
        drone.get_landed_state.return_value=0
        with patch('src.mission.flight.asyncio.sleep',new=AsyncMock()):
            result=await land_and_confirm(drone,confirmation_seconds=0)
        self.assertFalse(result['land_return']);self.assertEqual(result['landed_state'],0)
        drone.disarm.assert_called_once()


if __name__=='__main__':unittest.main()
