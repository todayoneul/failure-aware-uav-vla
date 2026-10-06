"""Failure-path integration tests with a fake simulator and no model loading."""
import argparse
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch
import numpy as np


class FakeDrone:
    def __init__(self,hover_failure=False,connection_loss=False):
        self.flying=False;self.commanded=False;self.post_command_reads=0
        self.hover_failure=hover_failure;self.connection_loss=connection_loss
        self.events=[];self.robot_info={'collision_info':'collision'}
        self.sensors={'Chase':{'scene_camera':'chase'}}

    def get_ground_truth_kinematics(self):
        if self.commanded and self.connection_loss:
            self.post_command_reads+=1
            if self.post_command_reads>1:raise ConnectionError('simulator disconnected')
        return {'pose':{'position':{'x':0,'y':0,'z':-4 if self.flying else -2.5},
                        'orientation':dict(x=0,y=0,z=0,w=1)},
                'twist':{'linear':dict(x=0,y=0,z=0)},'time_stamp':10}

    def get_landed_state(self):return 1 if self.flying else 0
    def enable_api_control(self):self.events.append('enable')
    def disable_api_control(self):self.events.append('disable')
    def arm(self):self.events.append('arm')
    def disarm(self):self.events.append('disarm')

    async def takeoff_async(self,**kwargs):
        async def command():self.flying=True;return True
        return asyncio.create_task(command())

    async def hover_async(self):
        async def command():
            if self.hover_failure:raise RuntimeError('hover failed after lift')
            return True
        return asyncio.create_task(command())

    async def land_async(self,**kwargs):
        async def command():self.events.append('land');self.flying=False;return True
        return asyncio.create_task(command())

    def get_images(self,*_):return {0:{'encoding':'BGR','time_stamp':10}}

    async def move_by_velocity_z_async(self,vx,vy,z,**kwargs):
        async def command():self.events.append(('climb',z));return True
        return asyncio.create_task(command())


class MissionRunnerFailureTests(unittest.IsolatedAsyncioTestCase):
    async def run_fake(self,drone,directory,invalid_action=False,guard_stop=False,model_stop=False,
                       surface=None,words=None,prompt_mode=None):
        from src.mission import runner
        root=Path(directory);(root/'configs').mkdir()
        (root/'configs/mission_limits.json').write_text(json.dumps({'max_duration':300}))
        output=root/'outputs';output.mkdir()
        frame=np.zeros((256,256,3),dtype=np.uint8)
        session=Mock();session.telemetry={}
        session.poll_control.return_value={'mission_requests':[
            {'id':1,'request':{'action':'select','frame_id':42,'pixel':[320,180]}},
            {'id':2,'request':{'action':'start'}}],**({'prompt_mode':prompt_mode} if prompt_mode else {})}
        session.inject.return_value=([frame,frame],{})
        overview=Mock();overview.select.return_value=(surface or [2,0,-2.5],'captured depth frame 42')
        overview.landmark_at.return_value=None;overview.landmarks=[];overview.describe.return_value=words
        overview.capture.return_value=(frame,{})
        action={'fwd':.5,'down':0,'yaw':0,'stop':False,'bins':[10,49,49],'output':'10 49 49'}
        from src.integration.projectairsim_observation_adapter import make_prompt
        self.prompts=[]
        self.poses_at_inference=[]
        def infer(front,down,state,goal,instruction,direction_hint=True,freeform=None,**_):
            self.prompts.append(make_prompt(state,goal,instruction,direction_hint,freeform))
            self.poses_at_inference.append('after climb' if any(isinstance(e,tuple) for e in drone.events) else 'no climb')
            return {'parsed_action':None if invalid_action else action,'prompt':self.prompts[-1],
                    'parse_error':'invalid model action' if invalid_action else None}
        model=Mock();model.infer.side_effect=infer
        # The policy's own LAND; the observer then quits so the completed run can be inspected.
        if model_stop:action.update(fwd=0.,stop=True,bins=None,output='09 49 48 LAND')
        survives=model_stop or invalid_action
        if survives:
            requests=session.poll_control.return_value;polls=[]
            def poll():
                polls.append(1)
                if len(polls)>40:raise runner.DemoQuit('test complete')
                return requests
            session.poll_control=Mock(side_effect=poll)
        async def execute(*_):drone.commanded=True;return {'move':True,'hover':True}
        async def guarded(*args,**kwargs):
            if guard_stop:
                args[1].fail('collision');drone.commanded=True;drone.connection_loss=True;drone.post_command_reads=1
                return None
            from src.mission.flight import guarded_execute
            return await guarded_execute(*args,**kwargs)
        with patch.object(runner,'ROOT',root),patch.object(runner,'OUT',output), \
             patch.object(runner,'BlurDemoSession',return_value=session), \
             patch.object(runner,'ProjectAirSimClient'),patch.object(runner,'World'), \
             patch.object(runner,'Drone',return_value=drone), \
             patch.object(runner,'OverviewScene',return_value=overview), \
             patch.object(runner,'prepare_mission_config',return_value=root), \
             patch.object(runner,'publish_image'),patch.object(runner.cv2,'imread',return_value=frame), \
             patch.object(runner,'unpack_image',return_value=frame), \
             patch.object(runner,'guarded_execute',side_effect=guarded), \
             patch.object(runner,'matched_camera_images',return_value=(frame,np.ones((256,256))*2,
                {'width':256,'height':256,'fov_deg':90,'position':[0,0,-4],'orientation':[0,0,0,1],'time_stamp':10})), \
             patch.dict(sys.modules,{'src.integration.aerovla_int4_loader':SimpleNamespace(AeroVLAInt4=lambda *_:model)}), \
             patch('src.mission.flight.execute_action',side_effect=execute), \
             patch('src.mission.flight.asyncio.sleep',new=AsyncMock()),patch('builtins.print'):
            if survives:await runner.main(argparse.Namespace(steps=4,host='fake'))
            else:
                with self.assertRaises(SystemExit):
                    await runner.main(argparse.Namespace(steps=4,host='fake'))
        return json.loads((output/'mission-results.json').read_text())

    async def test_invalid_model_decision_keeps_grounding_and_visibility_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result=await self.run_fake(FakeDrone(),directory,invalid_action=True)
            events=[json.loads(line) for line in (Path(directory)/'outputs/mission-decisions.jsonl').read_text().splitlines()]
        mission=result['missions'][0]
        self.assertEqual(mission['vla_steps'],1);self.assertEqual(mission['executed_steps'],0)
        self.assertIn('grounding',events[-1]);self.assertIn('target_visibility',events[-1])
        self.assertEqual(events[-1]['decision_status'],'INVALID_ACTION')
        # Garbage model text fails the mission but not the session.
        self.assertEqual(mission['state'],'FAILED');self.assertEqual(mission['reason'],'invalid_action')
        self.assertEqual(result['program_status'],'STOPPED');self.assertEqual(len(result['missions']),1)

    async def test_guard_stopped_decision_is_logged_without_a_navigation_step(self):
        with tempfile.TemporaryDirectory() as directory:
            result=await self.run_fake(FakeDrone(),directory,guard_stop=True)
            events=[json.loads(line) for line in (Path(directory)/'outputs/mission-decisions.jsonl').read_text().splitlines()]
        mission=result['missions'][0]
        self.assertEqual(mission['vla_steps'],1);self.assertEqual(mission['executed_steps'],0)
        self.assertEqual(events[-1]['decision_status'],'SKIPPED_FAILED')

    async def test_model_land_ends_the_mission_lands_and_is_scored_by_radius(self):
        drone=FakeDrone()
        with tempfile.TemporaryDirectory() as directory:
            result=await self.run_fake(drone,directory,model_stop=True)
            events=[json.loads(line) for line in (Path(directory)/'outputs/mission-decisions.jsonl').read_text().splitlines()]
        self.assertEqual(result['program_status'],'STOPPED');self.assertEqual(len(result['missions']),1)
        mission=result['missions'][0]
        self.assertEqual(mission['state'],'SUCCESS');self.assertEqual(mission['executed_steps'],1)
        self.assertEqual(mission['stop']['step'],1);self.assertTrue(mission['stop']['landed'])
        self.assertAlmostEqual(mission['stop']['distance_m'],2)
        self.assertEqual(events[-1]['decision_status'],'EXECUTED')
        self.assertIn('land',drone.events);self.assertLess(drone.events.index('land'),drone.events.index('disarm'))

    async def test_selected_roof_starts_from_above_with_its_words_in_the_instruction(self):
        drone=FakeDrone()
        words={'object':'TemplateCube_Rounded_96','noun':'the gray block','description':'The target is the top of a gray block.'}
        with tempfile.TemporaryDirectory() as directory:
            result=await self.run_fake(drone,directory,model_stop=True,surface=[2,0,-16.],words=words,prompt_mode='instruction')
        mission=result['missions'][0]
        # 2 m from the roof point, so the climb to 6 m above the 16 m roof comes before the first decision.
        self.assertIn(('climb',-22.),drone.events);self.assertLess(drone.events.index(('climb',-22.)),drone.events.index('land'))
        self.assertEqual(self.poses_at_inference[0],'after climb')
        self.assertEqual(mission['target']['name'],'Gray block top (2.00, 0.00)');self.assertEqual(mission['target']['noun'],'the gray block')
        self.assertEqual(mission['prompt_mode'],'instruction');self.assertFalse(mission['direction_hint'])
        self.assertEqual(self.prompts,['<image>\nLand on top of the gray block. Fly around, find it with your camera, '
                                       'then fly straight to it and land.\nAction: '])
        # The fake vehicle comes to rest at the launch platform, far below the roof it was sent to.
        self.assertFalse(mission['stop']['on_target_surface']);self.assertAlmostEqual(mission['stop']['below_target_surface_m'],13.5)

    async def test_distant_roof_is_flown_to_at_the_takeoff_height_first(self):
        drone=FakeDrone()
        words={'object':'TemplateCube_Rounded_96','noun':'the gray block','description':'The target is the top of a gray block.'}
        with tempfile.TemporaryDirectory() as directory:
            await self.run_fake(drone,directory,model_stop=True,surface=[96,0,-16.],words=words)
        # 96 m away: no climb above the launch point; the model flies (here: stops) at the takeoff height.
        self.assertNotIn(('climb',-22.),drone.events)
        self.assertEqual(self.poses_at_inference,['no climb'])

    async def test_ground_click_keeps_the_takeoff_height_and_the_generic_prompt(self):
        drone=FakeDrone()
        with tempfile.TemporaryDirectory() as directory:
            result=await self.run_fake(drone,directory,model_stop=True)
        self.assertFalse(any(isinstance(event,tuple) and event[0]=='climb' and event[1]<-4 for event in drone.events))
        self.assertEqual(result['missions'][0]['target']['name'],'Visible surface (2.00, 0.00)')
        self.assertEqual(self.prompts,['<image>\nFly straight ahead and find the target. Find the selected target location.\nAction: '])

    async def test_partial_takeoff_failure_lands_before_disarming(self):
        drone=FakeDrone(hover_failure=True)
        with tempfile.TemporaryDirectory() as directory:await self.run_fake(drone,directory)
        self.assertIn('land',drone.events)
        self.assertLess(drone.events.index('land'),drone.events.index('disarm'))

    async def test_connection_loss_retains_failed_mission_and_last_observation(self):
        drone=FakeDrone(connection_loss=True)
        with tempfile.TemporaryDirectory() as directory:result=await self.run_fake(drone,directory)
        self.assertEqual(result['program_status'],'FAIL')
        self.assertEqual(len(result['missions']),1)
        mission=result['missions'][0]
        self.assertEqual(mission['state'],'FAILED');self.assertEqual(mission['vla_steps'],1)
        self.assertEqual(mission['final_observation_status'],'last_validated_before_exit')
        self.assertEqual(mission['final_distance_m'],2)


if __name__=='__main__':unittest.main()
