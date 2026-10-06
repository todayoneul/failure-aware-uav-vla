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


class MissionManagerTests(unittest.TestCase):
    def state(self, x=0, y=0, z=-4, speed=0):
        return {'position':[x,y,z], 'orientation':[0,0,0,1], 'velocity':[speed,0,0]}

    def test_selection_navigation_success_and_reset(self):
        from src.mission.manager import MissionManager, MissionTarget
        manager=MissionManager()
        self.assertEqual(manager.status,'IDLE')
        manager.select(MissionTarget('surface',(2,0,-2.5),'depth frame'))
        self.assertEqual(manager.status,'TARGET_SELECTED')
        manager.start('GO_TO',self.state(),now=0)
        self.assertEqual(manager.status,'NAVIGATING')
        manager.observe(self.state(x=1.7),now=1)
        self.assertEqual(manager.status,'SUCCESS')
        manager.fail('cleanup issue')
        self.assertEqual(manager.status,'SUCCESS')
        manager.reset();self.assertEqual(manager.status,'IDLE');self.assertIsNone(manager.target)

    def test_hover_needs_low_speed_altitude_and_stable_hold(self):
        from src.mission.manager import MissionManager,MissionTarget
        manager=MissionManager(hover_seconds=1)
        manager.select(MissionTarget('surface',(2,0,-2.5),'depth frame'))
        manager.start('GO_TO_AND_HOVER',self.state(),now=0)
        manager.observe(self.state(x=1.8,speed=1),now=1)
        self.assertEqual(manager.status,'HOVERING')
        manager.observe(self.state(x=1.8),now=2)
        self.assertNotEqual(manager.status,'SUCCESS')
        manager.observe(self.state(x=1.8),now=3.1)
        self.assertEqual(manager.status,'SUCCESS')

    def test_hover_does_not_freeze_policy_at_wrong_altitude(self):
        from src.mission.manager import MissionManager,MissionTarget
        manager=MissionManager();manager.select(MissionTarget('surface',(2,0,-2.5),'depth frame'))
        manager.start('GO_TO_AND_HOVER',self.state(),now=0)
        manager.observe(self.state(x=1.8,z=-5.5),now=1)
        self.assertEqual(manager.status,'NAVIGATING')
        manager.observe(self.state(x=1.8,z=-4),now=2)
        self.assertEqual(manager.status,'HOVERING')
        manager.observe(self.state(x=1.3,z=-4),now=2.1)
        self.assertEqual(manager.status,'NAVIGATING')

    def test_land_requires_arrival_and_landed_state_not_cleanup(self):
        from src.mission.manager import MissionManager,MissionTarget
        manager=MissionManager()
        manager.select(MissionTarget('surface',(2,0,-2.5),'depth frame'))
        manager.start('GO_TO_AND_LAND',self.state(),now=0)
        manager.observe(self.state(x=1.8),now=1)
        self.assertEqual(manager.status,'LANDING')
        manager.confirm_landing(self.state(x=1.8,z=-2.7),landed=False)
        self.assertNotEqual(manager.status,'SUCCESS')
        manager.confirm_landing(self.state(x=1.8,z=-2.7),landed=True)
        self.assertEqual(manager.status,'SUCCESS')
        failed=MissionManager();failed.select(MissionTarget('surface',(2,0,-2.5),'depth frame'))
        failed.start('GO_TO',self.state(),now=0);failed.fail('collision')
        failed.confirm_landing(self.state(x=1.8,z=-2.7),landed=True)
        self.assertEqual(failed.status,'FAILED')

    def test_timeout_abort_and_distance(self):
        from src.mission.manager import MissionManager,MissionTarget
        manager=MissionManager(max_duration=10)
        manager.select(MissionTarget('surface',(3,4,-2.5),'depth frame'))
        manager.start('GO_TO',self.state(),now=0)
        self.assertEqual(manager.metrics(self.state())['horizontal_m'],5)
        manager.observe(self.state(),now=10)
        self.assertEqual(manager.status,'FAILED');self.assertEqual(manager.reason,'timeout')
        manager.reset();manager.select(MissionTarget('surface',(2,0,-2.5),'depth frame'))
        manager.start('GO_TO',self.state(),now=0);manager.abort()
        self.assertEqual(manager.status,'ABORTED')


class MissionControlTests(unittest.TestCase):
    def test_map_mouse_coordinates_use_original_rgb_resolution(self):
        from scripts.mission_viewer import map_pixel,render_canvas,MAP_RECT
        meta={'width':640,'height':360}
        left,top,width,height=MAP_RECT
        self.assertEqual(map_pixel(left+width//2,top+height//2,meta),[320,180])
        self.assertIsNone(map_pixel(left-1,top+height//2,meta))
        telemetry={'mission':{'state':'TARGET_SELECTED','type':None,'errors':None,
                             'target':{'surface_position':[1,2,-2.5]}},'phase':'Select target'}
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

    def test_landing_signal_requires_arrival_and_fallback_is_explicit(self):
        from src.mission.flight import landing_source
        self.assertEqual(landing_source({'output':'LAND'},'NAVIGATING'),None)
        self.assertEqual(landing_source({'output':'LAND'},'LANDING'),'AeroVLA LAND signal at target tolerance')
        self.assertEqual(landing_source({'output':'<LAND>'},'LANDING'),'AeroVLA LAND signal at target tolerance')
        self.assertEqual(landing_source({'output':'00 49 49','stop':True},'LANDING'),'AeroVLA stop signal at target tolerance')
        self.assertEqual(landing_source({'output':'00 49 49'},'LANDING'),'Mission Manager fallback after target tolerance')

    def test_selection_and_mission_keys_do_not_change_blur_state(self):
        from src.mission.control import default_mission_control,request_selection,mission_key
        control=default_mission_control()
        selected=request_selection(control,4,(300,250))
        self.assertEqual(selected['mission_request']['frame_id'],4)
        self.assertEqual(selected['revision'],0)
        started=mission_key(selected,ord('g'))
        self.assertEqual(started['mission_request']['type'],'GO_TO')
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
            manager.start('GO_TO',state,now=0)
            with patch('src.mission.flight.execute_action',new=AsyncMock()) as execute:
                result=await guarded_execute(Mock(),manager,state,-2.5,events,10,{},now=now)
                execute.assert_not_called();self.assertIsNone(result)
                self.assertEqual(manager.reason,reason)

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
