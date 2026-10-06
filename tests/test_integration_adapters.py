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

    def test_model_displacement_is_flown_at_full_scale_with_altitude_clamp(self):
        from src.integration.projectairsim_action_adapter import convert_action
        state={'position':[0,0,-10], 'orientation':[0,0,math.sqrt(.5),math.sqrt(.5)]}
        converted=convert_action({'fwd':5,'down':2,'yaw':.1,'stop':False}, state, ground_z=-2.7)
        self.assertEqual(converted['mode'],'translate')
        self.assertAlmostEqual(math.hypot(*converted['displacement_ned'][:2]),5)
        self.assertAlmostEqual(converted['displacement_ned'][2],2)
        self.assertAlmostEqual(converted['yaw_delta_rad'],.1)
        np.testing.assert_allclose(converted['target_position'],
                                   np.add(state['position'],converted['displacement_ned']))
        self.assertFalse(converted['altitude_clamped'])
        # At the minimum clearance, a positive-down output must never descend.
        state['position'][2]=-3.5
        low=convert_action({'fwd':5,'down':5,'yaw':0,'stop':False},state,ground_z=-2.7)
        self.assertAlmostEqual(low['displacement_ned'][2],0);self.assertTrue(low['altitude_clamped'])
        self.assertAlmostEqual(math.hypot(*low['displacement_ned'][:2]),5)

    def test_large_turn_rotates_in_place_like_upstream(self):
        from src.integration.projectairsim_action_adapter import convert_action
        state={'position':[1,2,-10],'orientation':[0,0,0,1]}
        turn=convert_action({'fwd':4.5,'down':-2.55,'yaw':-1.1,'stop':False},state,ground_z=-2.7)
        self.assertEqual(turn['mode'],'turn')
        self.assertAlmostEqual(turn['yaw_delta_rad'],-1.1)  # The full model yaw, not a 15 degree clip.
        self.assertEqual(turn['displacement_ned'][:2],[0.,0.])
        self.assertAlmostEqual(turn['displacement_ned'][2],-2.55)
        small=convert_action({'fwd':4.5,'down':0,'yaw':-.24,'stop':False},state,ground_z=-2.7)
        self.assertEqual(small['mode'],'translate')
        stop=convert_action({'fwd':0,'down':0,'yaw':0,'stop':True},state,ground_z=-2.7)
        self.assertEqual(stop['mode'],'stop');self.assertEqual(stop['displacement_ned'],[0.,0.,0.])

    def test_out_of_envelope_state_is_recovered_by_a_bounded_clamp_not_an_exception(self):
        from src.integration.projectairsim_action_adapter import convert_action,PLATFORM_DEMO_LIMITS
        action={'fwd':.5,'down':0,'yaw':0,'stop':False}
        below=convert_action(action,{'position':[0,0,-2.7],'orientation':[0,0,0,1]},ground_z=-2.7)
        self.assertAlmostEqual(below['target_z'],-3.5);self.assertTrue(below['altitude_clamped'])
        # Far outside, one step never exceeds the model's own vertical range.
        above=convert_action(action,{'position':[0,0,-60.],'orientation':[0,0,0,1]},ground_z=-2.7)
        self.assertAlmostEqual(above['displacement_ned'][2],5)
        scaled=convert_action({'fwd':5,'down':0,'yaw':0,'stop':False},{'position':[0,0,-4],'orientation':[0,0,0,1]},
                              ground_z=-2.7,limits=PLATFORM_DEMO_LIMITS)
        self.assertAlmostEqual(scaled['displacement_ned'][0],.5)
        with self.assertRaises(ValueError):
            convert_action(action,{'position':[0,0,float('nan')],'orientation':[0,0,0,1]},ground_z=-2.7)
        with self.assertRaises(ValueError):
            convert_action(action,{'position':[0,0,-4],'orientation':[0,0,0,1]},ground_z=-2.7,limits={'displacement_scale':0})

    def test_action_grammar_admits_exactly_the_action_format(self):
        from src.integration.projectairsim_action_adapter import allowed_action_tokens,parse_action
        digits=list(range(10));space,land,end=10,[11,12],13
        text={**{d:str(d) for d in digits},space:' ',11:' L',12:'AND',end:''}
        def allowed(sequence):return allowed_action_tokens(sequence,digits,space,land,end)
        self.assertEqual(allowed([]),digits);self.assertEqual(allowed([4]),digits)
        # Bins stop at 98: after a 9 only 0-8 may follow.
        self.assertEqual(allowed([9]),digits[:9]);self.assertEqual(allowed([9,7,space,9]),digits[:9])
        self.assertEqual(allowed([9,7]),[space]);self.assertEqual(allowed([9,7,space,4,9]),[space])
        self.assertEqual(allowed([9,7,space,4,9,space,4,9]),[end,land[0]])
        self.assertEqual(allowed([9,7,space,4,9,space,4,9,land[0]]),[land[1]])
        self.assertEqual(allowed([9,7,space,4,9,space,4,9,land[0],land[1]]),[end])
        # Whatever is chosen inside the grammar parses: taking the first or the last allowed token each time.
        for pick,output,stop in ((0,'00 00 00',False),(-1,'98 98 98 LAND',True)):
            sequence=[]
            while not sequence or sequence[-1]!=end:sequence.append(allowed(sequence)[pick])
            action=parse_action('Action: '+''.join(text[token] for token in sequence))
            self.assertEqual((action['output'],action['stop']),(output,stop))

    def test_freeform_instruction_replaces_the_whole_template(self):
        from src.integration.projectairsim_observation_adapter import make_prompt
        state={'position':[0,0,-4],'orientation':[0,0,0,1]}
        self.assertEqual(make_prompt(state,[5,5,-4],'no delimiters needed',freeform=' Land on top of the gray block. '),
                         '<image>\nLand on top of the gray block.\nAction: ')
        with self.assertRaises(ValueError):make_prompt(state,[5,5,-4],'no delimiters needed')

    def test_prompt_without_direction_hint_keeps_upstream_template(self):
        from src.integration.projectairsim_observation_adapter import make_prompt
        state={'position':[0,0,-4],'orientation':[0,0,0,1]}
        instruction='The target is 0 degrees from you. The target is a large blue cone. Please control the drone.'
        self.assertEqual(make_prompt(state,[5,5,-4],instruction),
                         '<image>\nFly forward-right and find the target. The target is a large blue cone.\nAction: ')
        self.assertEqual(make_prompt(state,[5,5,-4],instruction,direction_hint=False),
                         '<image>\nFly and find the target. The target is a large blue cone.\nAction: ')

class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    class Drone:
        def __init__(self,z=-10.):
            self.calls=[];self.z=z
        def get_ground_truth_kinematics(self):return {'pose':{'position':{'x':0,'y':0,'z':self.z}}}
        def _task(self,name,arguments,keywords):
            self.calls.append((name,arguments,keywords))
            async def done():return True
            return done()
        async def hover_async(self):return self._task('hover',(),{})
        async def move_by_velocity_z_async(self,*arguments,**keywords):return self._task('velocity_z',arguments,keywords)
        async def move_to_position_async(self,*arguments,**keywords):return self._task('position',arguments,keywords)
        async def move_by_velocity_async(self,*arguments,**keywords):raise AssertionError('Plain velocity moves lose altitude')

    async def run_action(self,action,z=-10.,drone_z=None):
        from src.integration.projectairsim_action_adapter import convert_action,execute_action
        drone=self.Drone(z if drone_z is None else drone_z)
        command=convert_action(action,{'position':[0,0,z],'orientation':[0,0,0,1]},ground_z=-2.7)
        return drone,command,await execute_action(drone,command)

    async def test_translation_uses_position_control_at_the_commanded_altitude(self):
        drone,command,returns=await self.run_action({'fwd':4.9,'down':0,'yaw':0,'stop':False})
        name,arguments,keywords=drone.calls[0]
        self.assertEqual(name,'position');np.testing.assert_allclose(arguments[:3],[4.9,0,-10])
        self.assertEqual(arguments[3],1.);self.assertFalse(keywords['yaw_is_rate'])
        self.assertEqual([call[0] for call in drone.calls],['position','hover','hover'])
        self.assertTrue(returns['hover'])

    async def test_vertical_residual_after_translation_is_trimmed(self):
        # The vehicle is still 0.5 m short of the commanded climb when the path move ends.
        drone,command,returns=await self.run_action({'fwd':4.9,'down':-2.5,'yaw':0,'stop':False},drone_z=-12.)
        self.assertEqual([call[0] for call in drone.calls],['position','hover','velocity_z','hover'])
        self.assertAlmostEqual(drone.calls[2][1][2],-12.5);self.assertIn('trim',returns)

    async def test_large_turn_is_one_altitude_hold_command_without_translation(self):
        drone,command,returns=await self.run_action({'fwd':4.9,'down':-2.55,'yaw':-1.1,'stop':False})
        name,arguments,keywords=drone.calls[0]
        self.assertEqual(name,'velocity_z');self.assertEqual(arguments[:2],(0.,0.))
        self.assertAlmostEqual(arguments[2],-12.55);self.assertAlmostEqual(keywords['yaw'],-1.1)
        self.assertGreaterEqual(keywords['duration'],1.9)

    async def test_sub_metre_move_and_stop(self):
        drone,command,returns=await self.run_action({'fwd':.5,'down':0,'yaw':0,'stop':False})
        self.assertEqual(drone.calls[0][0],'velocity_z');self.assertEqual(drone.calls[0][2]['duration'],1.)
        drone,command,returns=await self.run_action({'fwd':0,'down':0,'yaw':0,'stop':True})
        self.assertEqual([call[0] for call in drone.calls],['hover'])


if __name__=='__main__': unittest.main()
