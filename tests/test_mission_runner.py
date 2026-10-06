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


class MissionRunnerFailureTests(unittest.IsolatedAsyncioTestCase):
    async def run_fake(self,drone,directory,invalid_action=False,guard_stop=False):
        from src.mission import runner
        root=Path(directory);(root/'configs').mkdir()
        (root/'configs/mission_limits.json').write_text(json.dumps({'max_duration':300}))
        (root/'configs/mission_capabilities.json').write_text(json.dumps({'go_to_live_verified':False,'hover_live_verified':False}))
        output=root/'outputs';output.mkdir()
        frame=np.zeros((256,256,3),dtype=np.uint8)
        session=Mock();session.telemetry={'capabilities':{'GO_TO':True}}
        session.poll_control.return_value={'mission_requests':[
            {'id':1,'request':{'action':'select','frame_id':42,'pixel':[320,180]}},
            {'id':2,'request':{'action':'start','type':'GO_TO'}}]}
        session.inject.return_value=([frame,frame],{})
        overview=Mock();overview.select.return_value=([2,0,-2.5],'captured depth frame 42')
        overview.capture.return_value=(frame,{})
        action={'fwd':.5,'down':0,'yaw':0,'stop':False,'bins':[10,49,49],'output':'10 49 49'}
        model=Mock();model.infer.return_value={'parsed_action':action,'parse_error':None,
            'prompt':'<image>\nFly straight ahead and find the target. Find the selected target location.\nAction: '}
        if invalid_action:model.infer.return_value.update(parsed_action=None,parse_error='invalid model action')
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

    async def test_guard_stopped_decision_is_logged_without_a_navigation_step(self):
        with tempfile.TemporaryDirectory() as directory:
            result=await self.run_fake(FakeDrone(),directory,guard_stop=True)
            events=[json.loads(line) for line in (Path(directory)/'outputs/mission-decisions.jsonl').read_text().splitlines()]
        mission=result['missions'][0]
        self.assertEqual(mission['vla_steps'],1);self.assertEqual(mission['executed_steps'],0)
        self.assertEqual(events[-1]['decision_status'],'SKIPPED_FAILED')

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
