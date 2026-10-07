"""Protocol consistency and scoring tables; no simulator or model."""
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class EvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((ROOT/'configs/evaluation_protocol.json').read_text(encoding='utf-8'))
        self.landmarks=json.loads((ROOT/'configs/landmarks.json').read_text(encoding='utf-8'))

    def test_trials_reference_defined_starts_targets_and_prompts(self):
        from scripts.evaluate_model import PROMPTS
        identifiers=[trial['id'] for trial in self.protocol['trials']]
        self.assertEqual(len(identifiers),len(set(identifiers)))
        landmark_ids={item['id'] for item in self.landmarks}
        for target in self.protocol['targets'].values():
            self.assertTrue(('landmark' in target)!=('position' in target))
            if 'landmark' in target:self.assertIn(target['landmark'],landmark_ids)
        for trial in self.protocol['trials']:
            self.assertIn(trial['start'],self.protocol['starts'])
            self.assertIn(trial['target'],self.protocol['targets'])
            self.assertIn(trial.get('hint_target',trial['target']),self.protocol['targets'])
            self.assertIn(trial['prompt'],PROMPTS)
            self.assertTrue(1<=trial['max_steps']<=60)
            # A description can only be evaluated for a target that has one.
            target=self.protocol['targets'][trial['target']]
            if PROMPTS[trial['prompt']][1]:self.assertTrue('landmark' in target or 'description' in target)
            if PROMPTS[trial['prompt']][0]=='instruction':self.assertTrue('landmark' in target or 'noun' in target)
            self.assertIn(trial.get('approach'),(None,'above'))
        self.assertEqual(self.protocol['success_radii_m'],sorted(self.protocol['success_radii_m'],reverse=True))

    def test_target_description_follows_the_prompt_condition(self):
        from scripts.evaluate_model import resolve_target
        from src.mission.landmarks import instruction_for,GENERIC_DESCRIPTION
        scene=[{'id':'blue_cone','name':'Blue cone','description':'The target is a large blue cone.',
                'position':[91.4,-35.4,-11.],'objects':['Cone_5']}]
        described=resolve_target('blue_cone',self.protocol,scene,True)
        generic=resolve_target('blue_cone',self.protocol,scene,False)
        self.assertEqual(described.position,generic.position)
        self.assertIn('large blue cone',instruction_for(described))
        self.assertIn(GENERIC_DESCRIPTION,instruction_for(generic))
        coordinate=resolve_target('far',self.protocol,scene,False)
        self.assertEqual(list(coordinate.position),self.protocol['targets']['far']['position'])
        self.assertIsNone(coordinate.landmark)
        # A coordinate target may carry its own words; they follow the prompt condition like a landmark's.
        roof=resolve_target('roof',self.protocol,scene,True)
        self.assertIn('top of a gray block',instruction_for(roof));self.assertEqual(roof.noun,'the gray block')
        self.assertIn(GENERIC_DESCRIPTION,instruction_for(resolve_target('roof',self.protocol,scene,False)))

    def test_instruction_condition_sends_a_sentence_without_direction_or_template(self):
        from scripts.evaluate_model import PROMPTS,resolve_target
        from src.mission.landmarks import prompt_arguments
        from src.integration.projectairsim_observation_adapter import make_prompt
        scene=[{'id':'blue_cone','name':'Blue cone','description':'The target is a large blue cone.','noun':'the large blue cone',
                'position':[91.4,-35.4,-11.],'objects':['Cone_5']}]
        mode,described=PROMPTS['instruction']
        target=resolve_target('blue_cone',self.protocol,scene,described)
        hint,freeform=prompt_arguments(target,mode)
        self.assertFalse(hint)
        state={'position':[55,0,-3],'orientation':[0,0,0,1]}
        prompt=make_prompt(state,list(target.position),'unused',hint,freeform)
        self.assertEqual(prompt,'<image>\nLand on top of the large blue cone. Fly around, find it with your camera, '
                                'then fly straight to it and land.\nAction: ')
        for word in ('forward','left','right','ahead','find the target'):self.assertNotIn(word,prompt.replace('straight to it',''))
        self.assertEqual(prompt_arguments(target,'hint'),(True,None));self.assertEqual(prompt_arguments(target,'description'),(False,None))
        with self.assertRaises(ValueError):prompt_arguments(target,'coordinates')


class MotionProfileTests(unittest.TestCase):
    def test_stationary_share_separates_stop_and_go_from_continuous_flight(self):
        from scripts.evaluate_model import motion_profile
        second=1_000_000_000
        # 1 m/s for two seconds, then two seconds standing, sampled every 0.5 s.
        poses=[(int(i*second/2),min(i*.5,2.),0.) for i in range(9)]
        profile=motion_profile(poses,0,4*second)
        self.assertAlmostEqual(profile['navigation_sim_s'],4.);self.assertAlmostEqual(profile['mean_speed_mps'],.5)
        self.assertAlmostEqual(profile['stationary_fraction'],.5)
        self.assertAlmostEqual(motion_profile(poses,0,2*second)['stationary_fraction'],0.)
        self.assertIsNone(motion_profile(poses,10*second,11*second))


class EvaluationSummaryTests(unittest.TestCase):
    def trial(self,identifier,stop,minimum,reason=None,prompt='hint+landmark',group='prompt'):
        return {'id':identifier,'group':group,'start':'facing','target':'blue_cone','prompt_condition':prompt,
                'initial_distance_m':50.,'executed_steps':8,'minimum_distance_m':minimum,'reason':reason,'state':'FAILED',
                'stop':{'step':8,'distance_m':stop} if stop is not None else None,'actions':[['96 49 49',5]]}

    def test_success_needs_a_model_stop_inside_the_radius(self):
        from scripts.summarize_model_evaluation import within,outcome,group_table,trial_table
        landed=self.trial('a',12.,9.);passed_by=self.trial('b',None,3.,'max_steps');crashed=self.trial('c',None,6.,'collision')
        self.assertTrue(within(landed,20));self.assertFalse(within(landed,10))
        # Flying past the target without stopping is an oracle success only.
        self.assertFalse(within(passed_by,20))
        self.assertEqual(outcome(landed),'LAND @ step 8');self.assertEqual(outcome(crashed),'collision')
        table=group_table([landed,passed_by,crashed],[20.,10.,5.],lambda t:t['prompt_condition'],'Prompt')
        row=table.splitlines()[-1]
        self.assertEqual([cell.strip() for cell in row.strip('|').split('|')],
                         ['hint+landmark','3','1/3','1/3','0/3','0/3','3/3','1/3','12.0 m'])
        self.assertIn('| a | facing -> blue_cone | hint+landmark | 50.0 m | 8 | LAND @ step 8 | 12.0 m | 9.0 m | O | X | X |',
                      trial_table([landed],[20.,10.,5.]))

    def test_roof_runs_say_where_the_vehicle_came_down_and_decoder_changes_are_counted(self):
        from scripts.summarize_model_evaluation import run_text,decoder_table
        def roof(stop,**surface):
            return {**self.trial('r',stop,stop,group='roof'),'stop':{'step':8,'distance_m':stop,**surface}}
        self.assertEqual(run_text(roof(2.2,on_target_surface=True,below_target_surface_m=-.2),20.),'**stop 2.2 m, on the surface**')
        self.assertEqual(run_text(roof(25.1,on_target_surface=False,below_target_surface_m=14.8),20.),'stop 25.1 m, landed 15 m below')
        self.assertEqual(run_text(roof(7.1,on_target_surface=False,below_target_surface_m=-10.2),20.),'**stop 7.1 m, landed 10 m above**')
        # A landmark is an object to reach, not a surface to land on.
        self.assertEqual(run_text({**self.trial('c',6.4,4.4),'stop':{'step':8,'distance_m':6.4,'on_target_surface':False,
                                                                    'below_target_surface_m':9.8}},20.),'**stop 6.4 m**')
        free={**self.trial('a',None,9.,'invalid_action'),'steps':[{'raw_output':'9x 49 49'}]}
        fixed={**self.trial('b',12.,9.),'decoder':'grammar','steps':[{'decoder':{'intervened':True}},{'decoder':{'intervened':False}},{}]}
        rows=decoder_table([free,fixed]).splitlines()
        self.assertEqual(rows[2],'| free | 1 | 1 | - | 1 |');self.assertEqual(rows[3],'| grammar | 1 | 3 | 1 | 0 |')


if __name__=='__main__':unittest.main()
