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
            if PROMPTS[trial['prompt']][1]:self.assertIn('landmark',self.protocol['targets'][trial['target']])
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


if __name__=='__main__':unittest.main()
