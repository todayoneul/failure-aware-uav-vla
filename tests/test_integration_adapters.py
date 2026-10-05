import math
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class AdapterTests(unittest.TestCase):
    def test_mosaic_rgb_and_front_down_order(self):
        from src.integration.projectairsim_observation_adapter import make_mosaic
        # BGR red front and blue down exercise the actual color conversion.
        front = np.full((256,256,3), [0,0,255], dtype=np.uint8)
        down = np.full((256,256,3), [255,0,0], dtype=np.uint8)
        im = make_mosaic(front, down)
        self.assertEqual(im.size, (224,448))
        self.assertEqual(im.getpixel((100,100)), (255,0,0))
        self.assertEqual(im.getpixel((100,300)), (0,0,255))

    def test_ned_yaw_and_quaternion_order(self):
        from src.integration.projectairsim_observation_adapter import adapt_state, semantic_direction
        s = adapt_state({'pose': {'position': {'x':0,'y':0,'z':-2},
                        'orientation': {'w':math.sqrt(.5),'x':0,'y':0,'z':math.sqrt(.5)}},
                        'twist':{'linear':{'x':0,'y':0,'z':0}}})
        self.assertEqual(s['orientation'], [0,0,math.sqrt(.5),math.sqrt(.5)])
        self.assertEqual(semantic_direction(s, [0,3,-2]), 'straight ahead ')
        self.assertEqual(semantic_direction(s, [-3,0,-2]), 'to your right ')

    def test_bins_land_and_invalid_output(self):
        from src.integration.projectairsim_action_adapter import parse_action
        a = parse_action('prompt Action: 98 49 98</s>')
        self.assertEqual(a['fwd'], 5)
        self.assertAlmostEqual(a['down'], 0)
        self.assertAlmostEqual(a['yaw'], 1.1)
        self.assertFalse(a['stop'])
        self.assertTrue(parse_action('Action: 0 49 49</s>')['stop'])
        self.assertTrue(parse_action('Action: <LAND>')['stop'])
        for bad in ('Action: none', 'Action: 1 2', 'Action: -1 3 4', 'Action: 100 49 49'):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError): parse_action(bad)

    def test_distance_mapping_and_altitude_clipping(self):
        from src.integration.projectairsim_action_adapter import convert_action
        state={'position':[0,0,-4], 'orientation':[0,0,math.sqrt(.5),math.sqrt(.5)]}
        converted=convert_action({'fwd':5,'down':5,'yaw':0,'stop':False}, state, ground_z=-2.7)
        self.assertAlmostEqual(converted['displacement_ned'][0], 0, places=7)
        self.assertAlmostEqual(converted['displacement_ned'][1], .5)
        self.assertAlmostEqual(converted['displacement_ned'][2], .3)
        self.assertEqual(converted['duration_sec'], 1)
        # At the minimum clearance, a positive-down output must never descend.
        state['position'][2]=-3.5
        a=convert_action({'fwd':0,'down':5,'yaw':1.1,'stop':False},state,ground_z=-2.7)
        self.assertAlmostEqual(a['displacement_ned'][2],0)
        self.assertAlmostEqual(a['yaw_delta_rad'], math.radians(15))

    def test_out_of_envelope_state_is_rejected_without_large_vertical_command(self):
        from src.integration.projectairsim_action_adapter import convert_action
        action={'fwd':.5,'down':0,'yaw':0,'stop':False}
        # An automatic jump back into the envelope would violate the .3m cap.
        for z in (-2.7,-8.,float('nan')):
            with self.subTest(z=z):
                with self.assertRaises(ValueError):
                    convert_action(action,{'position':[0,0,z],'orientation':[0,0,0,1]},ground_z=-2.7)

if __name__=='__main__': unittest.main()
