import copy
import math
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation


def box(name,center,size):
    return {'name':name,'bbox':{'center':dict(zip('xyz',center)),'size':dict(zip('xyz',size))}}


class FullMapTests(unittest.TestCase):
    def test_altitude_envelope_has_one_definition_and_never_raises_near_its_edge(self):
        from src.integration.projectairsim_action_adapter import convert_action
        action={'fwd':5,'down':0,'yaw':0,'stop':False}
        # 0.75-0.78 m clearance used to pass the monitor but crash the converter.
        for clearance in (.74,.76,.79,.8,30.,30.04):
            command=convert_action(action,{'position':[0,0,-clearance],'orientation':[0,0,0,1]},0)
            self.assertLessEqual(command['target_z'],-.8+1e-9);self.assertGreaterEqual(command['target_z'],-30-1e-9)
    def test_spatial_outlier_does_not_expand_finite_map(self):
        from src.mission.bounds import scene_bounds
        records=[box('Block'+str(i),[i*10,0,-2],[5,5,4]) for i in range(6)]
        records.append(box('FarOutlier',[1000,0,0],[5,5,4]))
        result=scene_bounds(records)
        self.assertLess(result['maximum'][0],100)
        self.assertIn('spatial outlier',result['excluded'][0]['reason'])

    def test_map_controls_preserve_mission_and_blur_and_full_map_resets_pan(self):
        from src.mission.control import default_mission_control,mission_key
        state=default_mission_control();moved=state
        for key in 'cw++ad':moved=mission_key(moved,ord(key))
        self.assertEqual(moved['mission_request_id'],state['mission_request_id'])
        self.assertEqual(moved['revision'],state['revision'])
        reset=mission_key(moved,ord('f'))
        self.assertEqual(reset['overview_pan'],[0,0]);self.assertEqual(reset['overview_zoom'],1)
    def test_altitude_maximum_is_configurable_without_changing_default(self):
        from src.integration.projectairsim_action_adapter import convert_action
        action={'fwd':.5,'down':-.3,'yaw':0,'stop':False}
        state={'position':[0,0,-7.95],'orientation':[0,0,0,1]}
        self.assertAlmostEqual(convert_action(action,state,0)['target_z'],-8.25)
        result=convert_action(action,state,0,limits={'minimum_clearance_m':.8,'maximum_clearance_m':8.})
        self.assertEqual(result['target_z'],-8);self.assertTrue(result['altitude_clamped'])
    def test_union_filters_giant_floor_helpers_and_nonfinite_bounds(self):
        from src.mission.bounds import scene_bounds
        records=[box('BlockA',[0,0,-3],[10,10,6]),box('BlockB',[100,50,-10],[10,10,20]),
                 box('Ground',[0,0,0],[40000,40000,2]),box('SunSky',[0,0,0],[10,10,10]),
                 box('Invalid',[float('nan'),0,0],[1,1,1])]
        result=scene_bounds(records)
        self.assertEqual(result['minimum'],[-5,-5,-20])
        self.assertEqual(result['maximum'],[105,55,0])
        self.assertEqual(len(result['excluded']),3)

    def test_camera_fit_contains_all_union_corners_in_both_views(self):
        from src.mission.bounds import fit_camera
        from src.mission.geometry import world_to_pixel
        b={'minimum':[-10,-70,-25],'maximum':[125,80,1]}
        for view in ('top','elevated'):
            pose=fit_camera(b,view)
            meta={'width':640,'height':360,'fov_deg':80,'position':pose['position'],
                  'orientation':Rotation.from_euler('xyz',pose['rpy'],degrees=True).as_quat()}
            for x in (-10,125):
                for y in (-70,80):
                    for z in (-25,1):
                        u,v=world_to_pixel([x,y,z],meta)
                        self.assertTrue(0<=u<640 and 0<=v<360,(view,u,v))

    def test_full_map_depth_beyond_old_100m_limit_is_valid(self):
        from src.mission.geometry import pixel_to_world
        meta={'width':640,'height':360,'fov_deg':80,'position':[0,0,-150],
              'orientation':Rotation.from_euler('xyz',[0,-90,0],degrees=True).as_quat()}
        self.assertAlmostEqual(pixel_to_world((320,180),150,meta)[2],0)
        with self.assertRaises(ValueError):pixel_to_world((320,180),65504,meta)


class TargetInspectorTests(unittest.TestCase):
    def test_mismatched_depth_timestamp_or_pose_cannot_classify_visibility(self):
        from src.mission.grounding import matched_camera_images
        rgb=dict(time_stamp=1,pos_x=0,pos_y=0,pos_z=0,rot_x=0,rot_y=0,rot_z=0,rot_w=1)
        for key,value in (('time_stamp',2),('pos_x',.1)):
            depth=dict(rgb);depth[key]=value
            with patch('projectairsim.utils.unpack_image') as unpack:
                with self.assertRaisesRegex(RuntimeError,'mismatch'):matched_camera_images({0:rgb,1:depth})
                unpack.assert_not_called()

    def test_unknown_depth_is_not_reported_as_confirmed_invisible(self):
        from src.mission.grounding import target_visibility
        result=target_visibility([10,0,0],self.meta(),np.full((256,256),float('nan')))
        self.assertIsNone(result['visible']);self.assertIsNone(result['observed_depth_m'])
        result=target_visibility([10,0,0],self.meta(),np.full((256,256),12))
        self.assertEqual(result['status'],'DEPTH MISMATCH');self.assertIsNone(result['visible'])
    def test_elevated_zoom_skips_behind_camera_markers_without_crashing(self):
        from scripts.mission_viewer import render_map
        from src.mission.bounds import fit_camera
        b={'minimum':[-40.1,-122.5,-26],'maximum':[120,117.5,2.5]}
        rig=fit_camera(b,'elevated',zoom=4.768)
        meta={'width':640,'height':360,'fov_deg':80,'position':rig['position'],
              'orientation':Rotation.from_euler('xyz',rig['rpy'],degrees=True).as_quat()}
        frame=np.zeros((360,640,3),dtype=np.uint8)
        telemetry={'mission':{'target':{'surface_position':[-35,112.5,-11]}},
                   'trajectory':[[-35,112.5,-11],[40,-2,-10]]}
        shown=render_map(frame,{'camera':meta},telemetry)
        self.assertEqual(shown.shape,frame.shape)
    def meta(self):
        return {'width':256,'height':256,'fov_deg':90,'position':[0,0,0],'orientation':[0,0,0,1]}

    def test_visibility_front_and_depth_comparison(self):
        from src.mission.grounding import target_visibility
        depth=np.full((256,256),10,dtype=np.float32)
        self.assertEqual(target_visibility([10,0,0],self.meta(),depth)['status'],'VISIBLE')
        self.assertEqual(target_visibility([12,0,0],self.meta(),depth)['status'],'OCCLUDED')
        self.assertEqual(target_visibility([-1,0,0],self.meta(),depth)['status'],'BEHIND CAMERA')
        self.assertEqual(target_visibility([1,3,0],self.meta(),depth)['status'],'OUT OF FOV')
        depth[128,128]=float('nan')
        self.assertEqual(target_visibility([10,0,0],self.meta(),depth)['status'],'UNKNOWN DEPTH')

    def test_grounding_uses_same_semantic_direction_and_prompt_as_model(self):
        from src.mission.grounding import grounding_report
        from src.integration.projectairsim_observation_adapter import make_prompt
        state={'position':[0,0,-4],'orientation':[0,0,0,1],'velocity':[0,0,0]}
        instruction='The target is 0 degrees from you. Find the selected target location. Please control the drone.'
        report=grounding_report(state,[5,5,-4],[5,5,-1],instruction)
        self.assertEqual(report['semantic_direction'],'forward-right')
        self.assertAlmostEqual(report['body_bearing_deg'],45)
        self.assertEqual(report['prompt'],make_prompt(state,[5,5,-4],instruction))
        self.assertFalse(any(report['vla_receives'].values()))
        self.assertEqual(report['mode'],'DIRECTION HINT + GENERIC DESCRIPTION')
        visual=grounding_report(state,[5,5,-4],[5,5,-1],instruction,direction_hint=False,landmark='blue_cone')
        self.assertEqual(visual['mode'],'NO HINT + LANDMARK DESCRIPTION')
        self.assertNotIn('forward-right',visual['prompt'])
        self.assertEqual(visual['prompt'],make_prompt(state,[5,5,-4],instruction,direction_hint=False))
        # The direction is still reported for the evaluator even when it is withheld from the model.
        self.assertEqual(visual['semantic_direction'],'forward-right')

    def test_marker_overlay_never_mutates_model_input(self):
        from src.mission.grounding import target_overlay
        raw=np.zeros((256,256,3),dtype=np.uint8);saved=raw.copy()
        debug=target_overlay(raw,{'pixel':[128,128],'in_fov':True,'status':'VISIBLE'})
        np.testing.assert_array_equal(raw,saved)
        self.assertFalse(np.array_equal(debug,raw))


if __name__=='__main__':unittest.main()
