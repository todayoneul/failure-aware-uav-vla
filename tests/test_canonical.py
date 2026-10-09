"""Canonical clean baseline: the landing finalizer, the three readings of a landing, the start sets and the gates.

Nothing here starts a simulator or loads a model.
"""
import inspect
import json
import math
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.visual_search.episodes import load_config
from src.visual_search.maps import load_map,load_landing,object_position
from src.visual_search.finalizer import LandingFinalizer,landing_requested,STATES
from src.visual_search.generalization import instruction_for_id
from src.visual_search import canonical
from src.aerovla_oft.spec import load_config as load_oft_config

CONFIG=load_config();OFT=load_oft_config();LANDING=load_landing();RULE=LANDING['finalizer'];CANON=CONFIG['canonical']
LAND='Find the blue landing pad and land on it.';APPROACH='Approach the blue landing pad.'


def fly(finalizer,decisions):
    """Feed decisions (height m, contact, commanded descent) half a second apart; returns the states."""
    states=[]
    for index,(height,contact,down) in enumerate(decisions):states.append(finalizer.update((0.,0.,-height),index*.5,contact,down))
    return states


DESCENT=[(8.,False,0.),(7.5,False,.5),(7.,False,.5),(6.5,False,.25)]


class FinalizerTests(unittest.TestCase):
    def test_only_a_sentence_that_asks_for_a_landing_arms_it(self):
        scene=load_map('field')
        for phrase in CONFIG['gen_v3']['instructions']['land']['train']+CONFIG['gen_v3']['instructions']['land']['held_out']:
            self.assertTrue(landing_requested(instruction_for_id(CONFIG,scene,'blue_pad',phrase)),phrase)
        for phrase in CONFIG['gen_v3']['instructions']['approach']['train']+CONFIG['gen_v3']['instructions']['approach']['held_out']:
            # "the blue landing pad" is in every one of these; the noun does not ask for a landing.
            self.assertFalse(landing_requested(instruction_for_id(CONFIG,scene,'blue_pad',phrase)),phrase)
        self.assertTrue(LandingFinalizer(RULE,LAND).active);self.assertFalse(LandingFinalizer(RULE,APPROACH).active)
        self.assertFalse(LandingFinalizer(RULE,LAND,enabled=False).active)

    def test_an_approach_mission_never_triggers_it(self):
        finalizer=LandingFinalizer(RULE,'Find and approach the blue landing pad.')
        states=fly(finalizer,DESCENT+[(2.5,True,.25)]+[(2.5,False,0.)]*8)
        self.assertEqual(set(states),{'FLYING'});self.assertFalse(finalizer.latched);self.assertEqual(finalizer.command([.3,.1,0.]),[.3,.1,0.])
        self.assertFalse(finalizer.report()['finalizer_triggered'])

    def test_it_needs_a_physical_contact_made_while_descending(self):
        # Hanging still just above a surface, however long: no contact, no landing.
        hover=LandingFinalizer(RULE,LAND);states=fly(hover,DESCENT+[(2.8,False,.1)]*3+[(2.8,False,0.)]*10)
        self.assertNotIn('CONTACT_CANDIDATE',states);self.assertFalse(hover.latched)
        # A contact in level flight is not a touchdown either.
        level=LandingFinalizer(RULE,LAND);states=fly(level,[(8.,False,0.)]*3+[(8.,True,0.)]+[(8.,False,0.)]*6)
        self.assertEqual(set(states),{'FLYING'});self.assertFalse(level.latched)

    def test_before_touchdown_nothing_is_suppressed_or_disarmed(self):
        finalizer=LandingFinalizer(RULE,LAND);states=fly(finalizer,DESCENT)
        self.assertEqual(states[-1],'LANDING_DESCENT');self.assertFalse(finalizer.latched)
        self.assertEqual(finalizer.command([0.,.25,0.]),[0.,.25,0.])
        report=finalizer.report();self.assertFalse(report['touchdown_detected'] or report['stable_contact_detected'] or report['disarm_triggered'])
        # The decision of the contact itself and the two after it are not enough.
        states=fly(LandingFinalizer(RULE,LAND),DESCENT+[(2.5,True,.25),(2.5,False,0.),(2.5,False,0.)])
        self.assertEqual(states[-3:],['CONTACT_CANDIDATE']*3)

    def test_a_stable_contact_latches_landed_and_then_passes_no_motion_on(self):
        finalizer=LandingFinalizer(RULE,LAND);states=fly(finalizer,DESCENT+[(2.5,True,.25)]+[(2.5,False,0.)]*RULE['stable_decisions'])
        self.assertEqual(states[-1],'LANDED_LATCHED');self.assertEqual(finalizer.path,['FLYING','LANDING_DESCENT','CONTACT_CANDIDATE','STABLE_CONTACT','LANDED_LATCHED'])
        self.assertTrue(finalizer.latched);self.assertEqual(finalizer.command([.4,-.5,.2]),[0.,0.,0.])
        # Whatever happens next, it stays latched.
        self.assertEqual(fly(finalizer,[(3.5,False,-.5),(2.5,True,.5)]),['LANDED_LATCHED']*2)
        report=finalizer.report();self.assertTrue(report['touchdown_detected'] and report['stable_contact_detected'] and report['finalizer_triggered'])
        self.assertAlmostEqual(report['stable_duration_s'],RULE['stable_decisions']*.5);self.assertAlmostEqual(report['touchdown_velocity_mps'],1.)
        self.assertFalse(report['disarm_triggered']);finalizer.disarmed(True)
        self.assertEqual(finalizer.state,'DISARMED');self.assertTrue(finalizer.report()['disarm_triggered']);self.assertEqual(finalizer.command([1.,0.,0.]),[0.,0.,0.])
        self.assertEqual(set(finalizer.path),set(STATES))

    def test_moving_on_the_surface_or_leaving_it_is_not_a_landing(self):
        sliding=LandingFinalizer(RULE,LAND);states=[]
        for index,(x,contact) in enumerate([(0.,False),(0.,False),(0.,False),(0.,True),(.3,False),(.6,False),(.9,False),(1.2,False)]):
            states.append(sliding.update((x,0.,-2.5 if index>=3 else -8.+index),index*.5,contact,.5 if index<3 else 0.))
        self.assertEqual(states[-1],'CONTACT_CANDIDATE');self.assertFalse(sliding.latched)
        bounced=LandingFinalizer(RULE,LAND);states=fly(bounced,DESCENT+[(2.5,True,.25),(2.5,False,0.),(2.9,False,-.3),(2.9,False,0.)])
        self.assertNotIn('LANDED_LATCHED',states);self.assertIn(states[-1],('LANDING_DESCENT','FLYING'));self.assertIsNone(bounced.contact)
        # Down again and still: this time it latches.
        states=fly_from(bounced,len(DESCENT)+4,[(2.6,False,.25),(2.5,True,.25)]+[(2.5,False,0.)]*RULE['stable_decisions'])
        self.assertEqual(states[-1],'LANDED_LATCHED')

    def test_it_is_never_told_what_the_vehicle_stands_on(self):
        names=set(inspect.signature(LandingFinalizer.update).parameters)|set(inspect.signature(LandingFinalizer.__init__).parameters)
        self.assertEqual(names,{'self','settings','instruction','enabled','position','stamp_s','contact_event','executed_down_m'})
        source=inspect.getsource(LandingFinalizer)
        for word in ('target','pad','bearing','distance','object_name'):self.assertNotIn(word,source,word)
        # The thresholds sit between standing on a pad and coming down onto one, as measured in the teacher's landings.
        self.assertLess(.029,RULE['max_vertical_mps']);self.assertLess(RULE['max_vertical_mps'],.5/2);self.assertGreaterEqual(RULE['stable_decisions'],3)


def fly_from(finalizer,start,decisions):
    return [finalizer.update((0.,0.,-height),(start+index)*.5,contact,down) for index,(height,contact,down) in enumerate(decisions)]


def step(distance,height=7.,landed=False,over=False,forward=0.,x=0.,epoch=0.):
    return {'distance_m':distance,'front_seen':True,'down_seen':False,'action':[forward,0.,0.],'height_m':height,'over':over,'landed':landed,'bearing_deg':0.,
            'position':[x,0.,-height],'velocity':[0.,0.,.4],'epoch':epoch}


class LandingReadingTests(unittest.TestCase):
    """Touchdown, stable physical landing, strict zero-action and the system's result are four separate things."""
    @classmethod
    def setUpClass(cls):
        try:
            from scripts import visual_search
        except ImportError as error:raise unittest.SkipTest(f'simulator client not installed here: {error}')
        cls.runner=visual_search
        cls.flight=[step(30.,forward=1.,epoch=0.),step(10.,forward=1.,epoch=.5),step(2.,over=True,epoch=1.),step(1.,over=True,height=4.,epoch=1.5)]+\
                   [step(1.,over=True,height=2.5,landed=True,epoch=2.+index*.5) for index in range(4)]
        cls.down={'object':'blue_pad','offset_m':[.6,-.8],'position':[0,0,0],'time_stamp':1}
        cls.latched={'active':True,'state':'DISARMED','path':list(STATES),'touchdown_detected':True,'stable_contact_detected':True,'finalizer_triggered':True,
                     'disarm_triggered':True,'touchdown_velocity_mps':.4,'horizontal_velocity_mps':0.,'stable_duration_s':1.5}

    def read(self,steps=None,stopped=False,touchdown='own',finalizer='latched',hit=None,task='land',reason='landed'):
        episode={'id':'e','target':'blue_pad','task':task}
        return self.runner.summarise(episode,steps or self.flight,CONFIG,reason,stopped,{'red_pad':42.},LANDING,self.down if touchdown=='own' else touchdown,hit,
                                     self.latched if finalizer=='latched' else finalizer)

    def test_the_system_lands_without_the_policys_zero_action_and_the_strict_reading_says_so(self):
        result=self.read()
        self.assertTrue(result['touchdown_success'] and result['stable_physical_landing'] and result['system_land_success'] and result['success'])
        self.assertFalse(result['strict_policy_zero_action'] or result['strict_land_success'] or result['land_success'])
        both=self.read(stopped=True);self.assertTrue(both['system_land_success'] and both['strict_policy_zero_action'] and both['strict_land_success'])

    def test_the_named_pad_is_judged_by_the_evaluator_not_by_the_finalizer(self):
        # The finalizer latched and disarmed exactly as above, on the other pad: a stable contact, and a failed mission.
        wrong=self.read(touchdown=dict(self.down,object='red_pad'))
        self.assertTrue(wrong['finalizer']['finalizer_triggered']);self.assertEqual(wrong['landed_on'],'red_pad')
        self.assertFalse(wrong['touchdown_success'] or wrong['stable_physical_landing'] or wrong['system_land_success'] or wrong['success']);self.assertTrue(wrong['wrong_target'])
        outside=self.read(touchdown=dict(self.down,offset_m=[6.8,0.]));self.assertFalse(outside['touchdown_success'] or outside['success'])
        self.assertFalse(self.read(hit='FieldBarn')['success'])

    def test_touching_down_is_not_yet_a_stable_landing_and_a_stable_landing_needs_the_finalizer(self):
        brief=self.flight[:5]+[step(1.,over=True,height=3.2,epoch=2.5)]
        result=self.read(steps=brief,finalizer=dict(self.latched,finalizer_triggered=False,disarm_triggered=False,stable_contact_detected=False,state='LANDING_DESCENT'))
        self.assertTrue(result['touchdown_success']);self.assertFalse(result['stable_physical_landing'] or result['system_land_success'])
        rested=self.read(finalizer=dict(self.latched,finalizer_triggered=False,disarm_triggered=False))
        self.assertTrue(rested['stable_physical_landing']);self.assertFalse(rested['system_land_success'])
        self.assertFalse(self.read(finalizer=dict(self.latched,disarm_triggered=False))['system_land_success'])
        fast=[dict(item,velocity=[0.,0.,.95]) for item in self.flight];self.assertFalse(self.read(steps=fast)['touchdown_success'])

    def test_runs_from_before_the_finalizer_are_read_as_they_were(self):
        from scripts.freeze_gen_v3 import read
        # No finalizer in the run: the mission's result is the strict rule, as in Gen-v3.
        old=self.read(finalizer=None);self.assertNotIn('system_land_success',old);self.assertFalse(old['success'])
        self.assertTrue(self.read(finalizer=None,stopped=True)['success'])
        self.assertFalse(self.read(finalizer=dict(self.latched,active=False))['success'])
        # (Three decisions flying toward the pad in view, so that the flight counts as having started its approach.)
        flight=[step(34.-index,forward=1.,epoch=-2.+index*.5) for index in range(4)]+self.flight
        rows=[dict(item,position=[-3080.+index,2940.,-8.]) for index,item in enumerate(flight)]
        legacy={'id':'e','target':'blue_pad','map':'field','layout':'a','task':'land','stopped':False,'success':False,'reason':'landed_no_stop','landed':True,
                'landed_on':'blue_pad','selected':'blue_pad','touchdown':{'horizontal_error_m':1.,'vertical_speed_mps':.4,'horizontal_speed_mps':0.,'inside_region':True,'soft':True}}
        record=read(legacy,rows,CONFIG['success_radius_m'],LANDING,{},{})
        self.assertEqual((record['failure'],record['mechanism'],record['finalizer_active'],record['system_landing']),('LANDING','no stop after touchdown',False,None))
        modern=read(dict(legacy,success=True,reason='landed',finalizer=self.latched,touchdown_success=True,stable_physical_landing=True,system_land_success=True,
                         strict_policy_zero_action=False),rows,CONFIG['success_radius_m'],LANDING,{},{})
        self.assertEqual((modern['failure'],modern['system_landing'],modern['strict_policy_zero_action'],modern['terminal_as_asked']),('',True,False,True))
        unlatched=read(dict(legacy,finalizer=dict(self.latched,finalizer_triggered=False),touchdown_success=True,stable_physical_landing=True,system_land_success=False),
                       rows,CONFIG['success_radius_m'],LANDING,{},{})
        self.assertEqual((unlatched['failure'],unlatched['mechanism']),('FINALIZER','did not latch'))
        bounced=read(dict(legacy,finalizer=self.latched,touchdown_success=True,stable_physical_landing=False,system_land_success=False),rows,CONFIG['success_radius_m'],LANDING,{},{})
        self.assertEqual(bounced['failure'],'PHYSICAL_LANDING')


class CanonicalSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets={name:json.loads(path.read_text(encoding='utf-8')) for name,path in canonical.FILES.items()}
        cls.scenes={name:load_map(name) for name in ('field','lot')}

    def test_both_files_are_reproduced_exactly_by_their_generator(self):
        for name,data in self.sets.items():
            self.assertEqual(json.loads(json.dumps(canonical.describe(CONFIG,name,canonical.build(CONFIG,OFT,name)))),data,name)

    def test_every_start_is_inside_the_canonical_range_and_the_bands_are_even(self):
        self.assertEqual(CANON['max_start_m'],44.);self.assertEqual(CANON['mission'],LAND)
        for name,size in (('validation',36),('test',48)):
            episodes=self.sets[name]['episodes'];self.assertEqual(len(episodes),size);self.assertEqual(len({e['id'] for e in episodes}),size)
            for band,(low,high) in CANON['bands_m'].items():
                group=[e for e in episodes if e['band']==band];self.assertEqual(len(group),size//3)
                # A start of a band is in the band; its twins may stand anywhere up to 44 m from what they name.
                self.assertTrue(all(low<=e['start_distance_m']<=high for e in group if 'twin_of' not in e))
            self.assertTrue(all(0<e['start_distance_m']<=CANON['max_start_m'] for e in episodes))
            centre=lambda e:object_position(self.scenes[e['map']],e['layout'],e['target'])
            self.assertTrue(all(abs(math.dist(e['start_xy'],centre(e))-e['start_distance_m'])<.02 for e in episodes))

    def test_seeds_layouts_and_starts_are_disjoint_from_everything_earlier(self):
        low,high=CONFIG['gen_v3']['seeds']['test'];sealed=json.loads(canonical.SEALED.read_text(encoding='utf-8'))
        sealed_seeds={e['seed'] for episodes in sealed['sets'].values() for e in episodes};used=set()
        for name,data in self.sets.items():
            first,last=CANON['seeds'][name];seeds={e['seed'] for e in data['episodes']}
            self.assertTrue(all(first<=seed<=last for seed in seeds));self.assertFalse(seeds&sealed_seeds);self.assertFalse(seeds&used);used|=seeds
            points=canonical.earlier_points(name)
            for episode in data['episodes']:
                self.assertIn(episode['layout'],CANON[f'{name}_layouts'][episode['map']])
                near=min(math.dist(episode['start_xy'],xy) for where,xy in points if where==episode['map'])
                self.assertGreaterEqual(near,CANON['separation_m'])
        # Validation flies layouts that were trained on; the test flies layouts that never were; neither touches the sealed scene.
        for scene,layouts in CANON['validation_layouts'].items():self.assertTrue(set(layouts)<=set(self.scenes[scene]['train_layouts']))
        for scene,layouts in CANON['test_layouts'].items():
            self.assertFalse(set(layouts)&set(self.scenes[scene]['train_layouts']));self.assertFalse(set(layouts)&set(CONFIG['gen_v3']['train_layouts'][scene]))
            self.assertNotIn('e',layouts)
        self.assertFalse(any(e['map']=='depot' for data in self.sets.values() for e in data['episodes']))

    def test_the_test_holds_every_kind_of_start_the_mission_is_meant_to_cover(self):
        episodes=self.sets['test']['episodes'];landings=[e for e in episodes if e['task']=='land']
        self.assertEqual((len(landings),sum(e['target']=='blue_pad' for e in landings),sum(e['target']=='red_pad' for e in landings)),(36,30,6))
        self.assertTrue(all(e['instruction']==instruction_for_id(CONFIG,self.scenes[e['map']],e['target'],'land') for e in landings))
        self.assertTrue(all(e['instruction']==LAND for e in landings if e['target']=='blue_pad'))
        in_view=[e for e in landings if e['plan']['acquired_tick']==0];self.assertGreater(len(in_view),8);self.assertGreater(len(landings)-len(in_view),8)
        neighbours={e['neighbour'] for e in episodes if e['role'] in ('pair','pair2')}
        self.assertIn('red_pad',neighbours);self.assertTrue(neighbours&{'blue_cube','blue_cylinder'})
        by_id={e['id']:e for e in episodes}
        for twin in [e for e in episodes if 'twin_of' in e]:
            base=by_id[twin['twin_of']];self.assertEqual((twin['start_xy'],twin['start_yaw_deg'],twin['start_height_m']),(base['start_xy'],base['start_yaw_deg'],base['start_height_m']))
            if twin['role']=='swap':
                self.assertEqual(twin['layout'],self.scenes[twin['map']]['swaps'][base['layout']]);self.assertEqual((twin['task'],twin['target']),('land','blue_pad'))
                self.assertEqual(object_position(self.scenes[twin['map']],twin['layout'],'blue_pad'),object_position(self.scenes[base['map']],base['layout'],base['neighbour']))
            elif twin['role']=='query':self.assertEqual((twin['task'],twin['target'],twin['layout']),('approach',base['neighbour'],base['layout']))
            else:self.assertEqual((twin['role'],twin['task'],twin['instruction']),('approach','approach',APPROACH))
        self.assertEqual({role:sum(e['role']==role for e in episodes) for role in ('swap','query','approach')},{'swap':6,'query':6,'approach':6})

    def test_the_smoke_starts_are_ten_validation_starts(self):
        episodes=self.sets['validation']['episodes'];ids=canonical.smoke_ids(CONFIG,episodes);by_id={e['id']:e for e in episodes}
        self.assertEqual(len(set(ids)),CANON['gates']['smoke']['episodes']);self.assertTrue(set(ids)<=set(by_id))
        self.assertEqual(sorted(by_id[i]['band'] for i in ids),['far']*3+['mid']*4+['near']*3)
        self.assertEqual(sum(by_id[i]['task']=='approach' for i in ids),3);self.assertFalse(set(ids)&{e['id'] for e in self.sets['test']['episodes']})

    def test_the_gates_ask_what_the_gen_v3_gates_asked(self):
        from scripts.canonical_baseline import checks,measures
        before=CONFIG['gen_v3']['gates'];now=CANON['gates']
        for key in ('success_rate_min','landing_success_rate_min','grounded_rate_min','terminal_as_asked_rate_min','wrong_target_max','collisions_max','errors_max','episodes_min'):
            self.assertEqual(now['representative'][key],before['representative'][key],key)
        for key in ('episodes','success_min','landing_success_min','collisions_max','errors_max'):self.assertEqual(now['smoke'][key],before['smoke'][key],key)
        item=lambda **changes:dict({'episode':'x','task':'land','band':'mid','success':True,'acquired':True,'grounded':True,'terminal_as_asked':True,'wrong_target':False,'collision':False},**changes)
        good=[item() for _ in range(27)]+[item(task='approach') for _ in range(9)]
        _,lines=checks('representative',now['representative'],good,[],CONFIG);self.assertTrue(all(value>=limit if how=='>=' else value<=limit for value,how,limit in lines.values()))
        bad=good[:28]+[item(success=False,terminal_as_asked=False) for _ in range(8)]
        _,lines=checks('representative',now['representative'],bad,[],CONFIG);value,how,limit=lines['success rate'];self.assertLess(value,limit)
        self.assertEqual(measures(good)['landing_success'],27)

    def test_a_latched_landing_the_stable_reading_missed_is_listed_and_not_recounted(self):
        try:
            from scripts.canonical_baseline import standing,disagreements,measures
        except ImportError as error:raise unittest.SkipTest(f'report dependencies not installed here: {error}')
        # The vehicle does not move after the contact, the simulator's reported vertical speed has not settled, and the
        # reading that asks for a settled speed counts only the last decision as standing.
        rest=lambda speed,flag,state:{'position':[5.,6.,-3.732],'velocity':[0.,0.,speed],'landed':flag,'finalizer':state}
        rows=[{'position':[5.,6.,-3.9],'velocity':[0.,0.,.6],'landed':False,'finalizer':'LANDING_DESCENT'}]+\
             [rest(.095,False,'CONTACT_CANDIDATE'),rest(.074,False,'CONTACT_CANDIDATE'),rest(.052,False,'CONTACT_CANDIDATE'),rest(.041,True,'LANDED_LATCHED')]
        shown=standing(rows)
        self.assertEqual((shown['decisions_in_contact'],shown['height_span_m'],shown['horizontal_span_m'],shown['decisions_read_as_standing']),(4,0.,0.,1))
        self.assertEqual(shown['reported_vertical_mps'],[.095,.074,.052,.041])
        self.assertIsNone(standing(rows[:2]))
        item=lambda **changes:dict({'episode':'a','task':'land','success':False,'acquired':True,'grounded':True,'terminal_as_asked':True,'wrong_target':False,'collision':False,
                                    'touchdown_success':True,'finalizer_triggered':True,'disarm_triggered':True,'stable_physical_landing':False,'strict_policy_zero_action':True},**changes)
        items=[item(),item(episode='b',stable_physical_landing=True,success=True),item(episode='c',touchdown_success=False),item(episode='d',task='approach')]
        listed=disagreements(items,{name:rows for name in 'abcd'})
        self.assertEqual([row['episode'] for row in listed],['a']);self.assertEqual(listed[0]['decisions_read_as_standing'],1)
        # Listing it changes nothing that is counted.
        self.assertEqual(measures(items)['landing_success'],1)


if __name__=='__main__':unittest.main()
