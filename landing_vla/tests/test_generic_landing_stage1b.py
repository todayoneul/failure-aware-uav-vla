"""Phase B, stage 1b (stop-safe generic landing): the stop rule the teacher is measured with, the teacher, the audit, the
plans and the gates. The flights are the canonical episode loop over a fake simulator; no model is loaded."""
import argparse
import asyncio
import itertools
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from tests.test_generic_landing import flight,setting,view

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'configs/generic_landing_stage1b.json'


def stage1b():
    from src.visual_search import generic_landing as stages
    return stages.load(STAGE)


def rule():
    stage=stage1b();return {key:stage['stop_safety'][key] for key in ('safe_ratio','near_stop_ratio')}


def steps(*labels,task='land'):
    """Decisions of one flight as they are recorded, from (action, state, landed) triples."""
    return [{'teacher_action':list(action),'teacher_state':state,'landed':landed,'distance_m':10.} for action,state,landed in labels]


class StopRuleTests(unittest.TestCase):
    def test_the_audit_and_the_teacher_are_measured_with_the_students_own_stop(self):
        import scripts.visual_search as canonical
        from src.aerovla_oft import spec
        from src.landing import failures,stop_safety
        # One function: the flight loop's count and the audit; the policy's own stop flag is the same call on the first action.
        for module in (canonical,stop_safety,failures):self.assertIs(module.is_stop,spec.is_stop)
        model=(ROOT/'src/aerovla_oft/model.py').read_text(encoding='utf-8')
        self.assertIn("from .spec import denormalize_action, is_stop",model);self.assertIn("'stop':is_stop(chunk[0],self.config)",model)
        source=(ROOT/'scripts/visual_search.py').read_text(encoding='utf-8');self.assertIn("chunk=inference['chunk'];action=chunk[0]",source)
        self.assertIn("halted=is_stop(action,oft) and not perturbed",source);self.assertIn("stop_count=stop_count+1 if halted else 0",source)
        self.assertIn("if stop_count>=config['stop_ticks']",source)
        config,oft,_,_=setting();self.assertEqual((config['stop_ticks'],oft['stop_threshold'],oft['chunk_size'],oft['execute_horizon']),(4,.08,4,1))
        limits=stop_safety.stop_limits(oft)
        for got,want in zip(limits,(.08,.08,.0336)):self.assertAlmostEqual(got,want)

    def test_the_stop_ratio_is_one_at_the_edge_of_the_stop_rule(self):
        from src.aerovla_oft.spec import is_stop
        from src.landing.stop_safety import stop_ratio
        _,oft,_,_=setting()
        for action,ratio in (([.15,0.,0.],1.875),([0.,.25,0.],3.125),([0.,-.5,0.],6.25),([0.,0.,.21],6.25),([.35,0.,0.],4.375),([.2,0.,.01],2.5),([0.,0.,0.],0.)):
            self.assertAlmostEqual(stop_ratio(action,oft),ratio,places=6)
        # At most one is a stop, anything more is not, on every axis and with every sign.
        for forward,down,yaw in itertools.product((0.,.04,.079,.081,.3),(-.3,-.081,-.079,0.,.079,.081),(-.05,-.0335,0.,.0335,.034)):
            action=[forward,down,yaw];self.assertEqual(is_stop(action,oft),stop_ratio(action,oft)<=1.,action)
        self.assertFalse(is_stop([.081,0.,0.],oft));self.assertLess(stop_ratio([.081,0.,0.],oft),rule()['safe_ratio'])

    def test_the_margin_is_the_pad_teachers_own_and_was_fixed_with_its_reason(self):
        safety=stage1b()['stop_safety'];_,oft,_,_=setting()
        from src.landing.stop_safety import stop_ratio
        from src.visual_search.maps import load_landing
        self.assertEqual((safety['safe_ratio'],safety['near_stop_ratio']),(3.,2.))
        # The canonical final descent is the pad teacher's smallest label before a touchdown; stage 1's creep is below the margin.
        self.assertGreaterEqual(stop_ratio([0.,load_landing()['teacher']['final_descent_m'],0.],oft),safety['safe_ratio'])
        self.assertLess(stop_ratio([load_landing()['teacher']['min_forward_m'],0.,0.],oft),safety['near_stop_ratio'])


class TeacherTests(unittest.TestCase):
    def teacher(self,target,kind='stop_safe'):
        from src.landing.teacher import TEACHERS
        config,oft,scene,surfaces=setting();self.oft=oft
        return TEACHERS[kind](config,oft,surface=surfaces[target])

    def grid(self,top):
        for height,distance,bearing,seen in itertools.product((3.,5.,top-.5,top+.4,top+1.2,top+2.5,top+3.3,top+3.6,top+5.,13.9),(.5,1.5,1.8,2.5,3.4,3.6,5.,7.,9.,11.9,12.1,18.,24.9,30.,44.),
                                                              (-170.,-40.,-25.,-8.,-2.,0.,1.,5.,12.,29.,31.,90.),((True,False,False),(True,True,False),(False,True,True),(False,False,True),(True,True,True),(False,False,False))):
            # A vehicle is over an object only when it is higher than its top and near its middle.
            if seen[2] and (height<top or distance>5.5):continue
            if height<top and distance<4.5:continue
            yield view(height_m=height,distance_m=distance,bearing_deg=bearing,visible=seen[0],below=seen[1],over=seen[2],surface_m=top)

    def test_no_label_of_a_landing_before_the_touchdown_is_a_stop_or_near_one(self):
        from src.aerovla_oft.spec import is_stop
        from src.landing.stop_safety import stop_ratio
        safe=rule()['safe_ratio']
        for target,top in (('blue_cube',8.),('green_cylinder',9.),('blue_pad',2.5),('red_cube',8.)):
            teacher=self.teacher(target);count=0
            for item in self.grid(top):
                action,state=teacher.act(dict(item));count+=1
                self.assertFalse(is_stop(action,self.oft),(target,item,action,state));self.assertGreaterEqual(stop_ratio(action,self.oft),safe,(target,item,action,state))
                self.assertNotIn(state,('landed','stop'))
            self.assertGreater(count,2000)

    def test_the_teacher_of_stage_1_is_kept_and_its_slow_entry_is_what_the_margin_catches(self):
        from src.landing.stop_safety import stop_ratio
        old=self.teacher('blue_cube','surface');new=self.teacher('blue_cube')
        over=view(distance_m=2.,height_m=11.5,visible=False,below=True,over=True,bearing_deg=0.)
        action,state=old.act(dict(over));self.assertEqual((action,state),([.15,0.,0.],'final'));self.assertAlmostEqual(stop_ratio(action,self.oft),1.875)
        self.assertLess(stop_ratio(action,self.oft),rule()['safe_ratio'])
        action,state=new.act(dict(over));self.assertEqual(state,'final');self.assertAlmostEqual(action[0],.35);self.assertGreaterEqual(stop_ratio(action,self.oft),4.)
        # Stage 1 stands still while it climbs near the object; stage 1b does so only until it is clear of the top.
        low=view(distance_m=10.,height_m=9.5);self.assertEqual(old.act(dict(low))[0][:2],[0.,-.5])
        action,state=new.act(dict(low));self.assertEqual(state,'climb');self.assertAlmostEqual(action[0],.5);self.assertEqual(action[1],-.5)

    def test_a_cube_is_climbed_to_in_place_until_the_vehicle_is_clear_of_its_top_and_then_while_moving_in(self):
        from src.landing.stop_safety import stop_ratio
        teacher=self.teacher('blue_cube');numbers=stage1b()['teacher']
        for height in (3.,6.,8.,8.9):
            action,state=teacher.act(view(distance_m=10.,height_m=height));self.assertEqual((state,action[0],action[1]),('climb',0.,-.5))
            self.assertGreaterEqual(stop_ratio(action,self.oft),6.)
        # Clear of the top (1 m over it): up and in, both far from a stop; the level flight that follows keeps the forward motion.
        for height in (8.+numbers['clear_above_top_m'],10.,11.):
            action,state=teacher.act(view(distance_m=10.,height_m=height));self.assertEqual(state,'climb');self.assertAlmostEqual(action[0],.5);self.assertLessEqual(action[1],-.25)
        action,state=teacher.act(view(distance_m=10.,height_m=11.5));self.assertIn(state,('final','approach'));self.assertEqual(action[1],0.);self.assertAlmostEqual(action[0],.5)
        # Far from the object it climbs on the way, as in stage 1; an approach never climbs toward the top.
        action,state=teacher.act(view(distance_m=20.,height_m=5.));self.assertEqual(state,'climb');self.assertGreater(action[0],.8);self.assertEqual(action[1],-.5)
        self.assertNotEqual(teacher.act(view(distance_m=20.,height_m=5.,task='approach'))[1],'climb')

    def test_over_a_cylinder_the_vehicle_flies_in_at_the_entry_speed_and_descends_at_the_canonical_radius(self):
        from src.landing.stop_safety import stop_ratio
        from src.visual_search.maps import load_landing
        teacher=self.teacher('green_cylinder');entry=stage1b()['teacher']['entry_forward_m'];land=load_landing()['teacher']
        over=lambda distance,height=12.6,bearing=0.:teacher.act(view(distance_m=distance,height_m=height,bearing_deg=bearing,visible=False,below=True,over=True,surface_m=9.))
        for distance in (1.8,2.5,3.5,5.,6.9):
            action,state=over(distance);self.assertEqual(state,'final');self.assertAlmostEqual(action[0],entry);self.assertEqual(action[1:],[0.,0.])
            self.assertGreaterEqual(stop_ratio(action,self.oft),4.3)
        # Slower from far out, never slower than the entry speed; off to a side it turns first, which is far from a stop too.
        self.assertAlmostEqual(over(10.)[0][0],.5);self.assertAlmostEqual(over(8.)[0][0],.4)
        action,state=over(3.,bearing=40.);self.assertEqual((state,action[0]),('align',0.));self.assertAlmostEqual(abs(action[2]),.21)
        # The descent: at the canonical radius, at the canonical rates, and it goes on low over the top a little away from the middle.
        self.assertEqual(over(land['align_radius_m']-.05),([0.,.5,0.],'descend'));self.assertEqual(over(1.,11.2),([0.,.25,0.],'descend'))
        self.assertEqual(over(3.4,11.9)[1],'descend');self.assertNotEqual(over(3.4,12.6)[1],'descend')
        for action,_ in (over(1.),over(1.,11.2)):self.assertGreaterEqual(stop_ratio(action,self.oft),rule()['safe_ratio'])

    def test_the_stop_of_a_landing_comes_only_after_the_touchdown_and_an_approach_stops_in_the_air(self):
        from src.aerovla_oft.spec import is_stop
        teacher=self.teacher('blue_cube')
        action,state=teacher.act(view(distance_m=.5,height_m=8.,landed=True,over=True));self.assertEqual((action,state),([0.,0.,0.],'landed'))
        action,state=teacher.act(view(distance_m=.5,height_m=8.2,landed=False,over=True,visible=False,below=True));self.assertEqual(state,'descend');self.assertFalse(is_stop(action,self.oft))
        # The approach twin: the canonical stop in the air at the arrival distance, with the object in view.
        action,state=teacher.act(view(distance_m=11.5,height_m=6.,task='approach'));self.assertEqual((action,state),([0.,0.,0.],'stop'))
        self.assertNotEqual(teacher.act(view(distance_m=11.5,height_m=6.,task='land'))[1],'stop')

    def test_on_a_pad_from_the_earlier_heights_and_in_every_approach_the_labels_are_the_canonical_ones(self):
        from src.visual_search.generalization import SearchTeacher
        config,oft,scene,surfaces=setting();old=SearchTeacher(config,oft)
        for target,top in (('blue_pad',2.5),('blue_cube',8.),('green_cylinder',9.)):
            new=self.teacher(target)
            for height,distance,bearing,seen in itertools.product((5.,6.,9.,12.,14.),(.5,1.5,3.,6.,12.,20.,30.,44.),(-120.,-30.,-5.,0.,10.,35.,170.),
                                                                  ((True,False,False),(False,True,True),(False,False,False),(True,True,True))):
                for task in ('land','approach'):
                    if target!='blue_pad' and task=='land' and any(seen):continue
                    item=view(height_m=height,distance_m=distance,bearing_deg=bearing,visible=seen[0],below=seen[1],over=seen[2],surface_m=top,task=task)
                    self.assertEqual(new.act(dict(item)),old.act(dict(item)),(target,item))

    def test_the_dry_run_lands_inside_the_usable_region_with_every_label_clear_of_a_stop(self):
        from src.landing import stop_safety
        from src.landing.teacher import rollout
        from src.visual_search.maps import object_position
        config,oft,scene,surfaces=setting()
        for target,height in (('blue_cube',6.),('blue_cube',3.),('green_cylinder',6.),('blue_cylinder',12.),('blue_pad',6.),('red_pad',3.)):
            centre=object_position(scene,'a',target)
            episode={'start_xy':[centre[0]-30.,centre[1]+5.],'start_yaw_deg':-40.,'start_height_m':height,'target':target,'task':'land','strategy':'right','max_ticks':320}
            trace=[];result=rollout(config,oft,scene,'a',episode,surfaces[target],kind='stop_safe',trace=trace)
            self.assertEqual((result['reason'],result['landed']),('teacher_stop',True),target);self.assertTrue(surfaces[target].inside([result['final_distance_m'],0.]),target)
            one=stop_safety.audit_episode(trace,'land',oft,rule(),config['stop_ticks'])
            self.assertEqual((one['nonterminal_stop'],one['below_safe'],one['stop_windows'],one['near_stop_windows'],one['terminal_not_a_stop']),(0,0,0,0,0),target)
            self.assertGreater(one['terminal'],0)
            # Stage 1's teacher from the same start, for a small top: the labels the margin is there for.
            trace=[];rollout(config,oft,scene,'a',episode,surfaces[target],kind='surface',trace=trace);old=stop_safety.audit_episode(trace,'land',oft,rule(),config['stop_ticks'])
            self.assertEqual(old['below_safe']>0,surfaces[target].usable_reach_m<6.,target)


class AuditTests(unittest.TestCase):
    def audit(self,episodes):
        from src.landing import stop_safety
        config,oft,_,_=setting();return stop_safety.audit(episodes,oft,rule(),config['stop_ticks'])

    def episode(self,labels,task='land',kind='cube',identifier='x'):
        return {'id':identifier,'task':task,'kind':kind,'start_distance_m':25.,'steps':steps(*labels)}

    def good_landing(self):
        return [([1.,0.,0.],'approach',False)]*6+[([.5,-.5,0.],'climb',False)]*4+[([.35,0.,0.],'final',False)]*8+[([0.,.5,0.],'descend',False)]*3+[([0.,.25,0.],'descend',False)]*5+[([0.,0.,0.],'landed',True)]*4

    def test_a_landing_whose_labels_keep_the_margin_passes_and_its_stop_after_the_touchdown_is_allowed(self):
        result=self.audit([self.episode(self.good_landing())]);self.assertTrue(result['passed'])
        land=result['by']['task']['LAND'];self.assertEqual((land['nonterminal'],land['terminal'],land['below_safe'],land['stop_windows']),(26,4,0,0))
        self.assertEqual(land['stop_ratio']['min'],3.125);self.assertIn('LAND SURFACE_APPROACH',result['by']['state']);self.assertIn('cube LAND',result['by']['kind'])

    def test_the_creep_of_stage_1_is_caught_as_below_the_margin_and_as_runs_near_a_stop(self):
        labels=self.good_landing();labels[10:18]=[([.15,0.,0.],'final',False)]*8
        result=self.audit([self.episode(labels)]);self.assertFalse(result['passed'])
        land=result['by']['task']['LAND'];self.assertEqual((land['nonterminal_stop'],land['below_safe'],land['stop_windows'],land['near_stop_windows']),(0,8,0,5))
        self.assertEqual(result['by']['state']['LAND SURFACE_APPROACH']['near_stop'],8);self.assertEqual(result['examples'][0]['examples'][0]['ratio'],1.875)
        self.assertEqual([check['passed'] for check in result['checks']],[True,False,True,True,True])

    def test_a_stop_before_the_touchdown_and_four_of_them_in_a_row_are_rejected(self):
        labels=self.good_landing();labels[12]=([.02,0.,0.],'final',False)
        result=self.audit([self.episode(labels)]);land=result['by']['task']['LAND'];self.assertFalse(result['passed'])
        self.assertEqual((land['nonterminal_stop'],land['stop_windows']),(1,0))
        labels=self.good_landing();labels[12:16]=[([0.,0.,0.],'final',False)]*4
        result=self.audit([self.episode(labels)]);land=result['by']['task']['LAND'];self.assertEqual((land['nonterminal_stop'],land['stop_windows']),(4,1))
        self.assertEqual([check['passed'] for check in result['checks']][:3],[False,False,False])
        # Five in a row are two runs of four; a forced move between them breaks the run and is no label.
        labels[12:17]=[([0.,0.,0.],'final',False)]*5;self.assertEqual(self.audit([self.episode(labels)])['by']['task']['LAND']['stop_windows'],2)
        episode=self.episode(labels);episode['steps'][14]['perturbed']=True
        land=self.audit([episode])['by']['task']['LAND'];self.assertEqual((land['stop_windows'],land['forced'],land['nonterminal_stop']),(0,1,4))

    def test_a_zero_label_while_the_vehicle_is_in_the_air_is_not_the_end_of_a_landing(self):
        # The teacher says `landed` but the vehicle does not stand on anything: a stop before the touchdown.
        labels=self.good_landing()[:-4]+[([0.,0.,0.],'landed',False)]*4
        land=self.audit([self.episode(labels)])['by']['task']['LAND'];self.assertEqual((land['terminal'],land['nonterminal_stop'],land['stop_windows']),(0,4,1))
        # And a label that is not a stop while the vehicle stands on the surface is counted at the end, not before it.
        labels=self.good_landing();labels[-1]=([0.,.25,0.],'descend',True)
        result=self.audit([self.episode(labels)]);self.assertEqual(result['total']['terminal_not_a_stop'],1);self.assertTrue(result['passed'])

    def test_an_approach_stops_in_the_air_and_its_slowing_down_is_counted_not_judged(self):
        approach=[([1.,0.,0.],'approach',False)]*6+[([.4,0.,0.],'approach',False),([.2,0.,0.],'approach',False)]+[([0.,0.,0.],'stop',False)]*4
        result=self.audit([self.episode(approach,task='approach')]);self.assertTrue(result['passed'])
        mine=result['by']['task']['APPROACH'];self.assertEqual((mine['nonterminal'],mine['terminal'],mine['below_safe'],mine['nonterminal_stop']),(8,4,1,0))
        self.assertEqual(mine['stop_ratio']['min'],2.5)
        # A zero before the teacher's own stop is not the end of an approach.
        approach[3]=([0.,0.,0.],'approach',False);self.assertFalse(self.audit([self.episode(approach,task='approach')])['passed'])

    def dataset(self,root,labels):
        folder=root/'episodes';folder.mkdir(parents=True)
        for index,(kind,task,rows) in enumerate(labels):
            summary={'id':f'e{index}','task':task,'target':f'blue_{kind}','surface_kind':kind,'start_distance_m':15.+10*index}
            (folder/f'e{index}.json').write_text(json.dumps({'summary':summary,'traj_rel_dir':f'field/e{index}','steps':steps(*rows)}))

    def run_tool(self,*arguments):
        import scripts.audit_generic_landing_teacher as tool
        with patch('sys.argv',['audit',*arguments]),patch('builtins.print'):
            with self.assertRaises(SystemExit) as ended:tool.main()
        return ended.exception.code

    def test_the_tool_fails_a_recorded_dataset_with_unsafe_labels_and_passes_a_corrected_one(self):
        approach=[([1.,0.,0.],'approach',False)]*5+[([.2,0.,0.],'approach',False)]+[([0.,0.,0.],'stop',False)]*4
        unsafe=self.good_landing();unsafe[10:18]=[([.15,0.,0.],'final',False)]*8
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.dataset(root/'old',[('cube','land',unsafe),('pad','land',self.good_landing()),('cube','approach',approach)])
            self.dataset(root/'new',[('cube','land',self.good_landing()),('cylinder','land',self.good_landing()),('pad','land',self.good_landing()),('cube','approach',approach)])
            self.assertEqual(self.run_tool('dataset',str(root/'old'),'--output',str(root/'old.json')),1)
            self.assertEqual(self.run_tool('dataset',str(root/'new'),'--output',str(root/'new.json')),0)
            old=json.loads((root/'old.json').read_text());new=json.loads((root/'new.json').read_text())
            self.assertEqual((old['passed'],old['episodes_that_break_the_rule'],old['by']['kind']['cube LAND']['below_safe'],old['by']['kind']['pad LAND']['below_safe']),(False,1,8,0))
            self.assertEqual((new['passed'],new['episodes_that_break_the_rule'],new['total']['episodes']),(True,0,4))
            self.assertEqual(new['rule']['safe_ratio'],3.);self.assertEqual(new['rule']['stop_ticks'],4)
            # Side by side: what the correction changed.
            import scripts.audit_generic_landing_teacher as tool
            rows={row['metric']:row for row in tool.compare(old,new)['rows']}
            self.assertEqual((rows['below the margin']['old'],rows['below the margin']['new']),(8,0))
            self.assertEqual((rows['smallest stop ratio']['old'],rows['smallest stop ratio']['new']),(1.875,3.125))
            self.assertEqual(rows['runs near a stop before the touchdown']['old'],5)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.visual_search import generic_landing as stages
        cls.stages=stages;cls.stage=stages.load(STAGE);cls.first=stages.load()
        cls.sets={name:json.loads(path.read_text(encoding='utf-8'))['episodes'] for name,path in stages.files_of(cls.stage).items()}
        cls.plan=json.loads(stages.train_plan_of(cls.stage).read_text(encoding='utf-8'))

    def count(self,episodes,**fields):
        return sum(all(episode.get(key)==value for key,value in fields.items()) for episode in episodes)

    def test_only_the_teacher_and_what_follows_from_a_new_run_differ_from_stage_1(self):
        same=('start_checkpoint','scenes','layouts','separation_m','targets','bands_m','height_m','instructions','max_ticks')
        for key in same:
            mine=self.stage[key];theirs=self.first[key]
            if key=='layouts':mine={k:mine[k] for k in ('train','evaluation')};theirs={k:theirs[k] for k in ('train','evaluation')}
            self.assertEqual(mine,theirs,key)
        for key in ('visible_deg','invisible_deg'):self.assertEqual(self.stage['first_view'][key],self.first['first_view'][key])
        for key in ('land_per_kind','approach_twins_per_kind','val_land_per_kind','val_approach_twins_per_kind'):self.assertEqual(self.stage['training'][key],self.first['training'][key])
        for key in ('updates','learning_rate','grounding_learning_rate','batch_size','gradient_accumulation','share','eval_every','eval_samples','balance','boost','select_band','val_file','candidates'):
            self.assertEqual(self.stage['fine_tune'][key],self.first['fine_tune'][key],key)
        # No gate is lower than stage 1's, and the pilot's lines are stage 1's.
        self.assertEqual(self.stage['validation']['gates'],self.first['validation']['gates']);self.assertEqual(self.stage['validation']['smoke'],self.first['validation']['smoke'])
        self.assertEqual(self.stage['pilot']['pass'],self.first['pilot']['pass'])
        self.assertEqual(self.stage['dataset']['replay'],self.first['dataset']['replay'])
        # Stage 1's episodes are not in the data, its checkpoint is not continued, and the teacher is the new one.
        self.assertIn(self.first['dataset']['added'],self.stage['dataset']['not_used']);self.assertNotIn(self.first['dataset']['added'],self.stage['dataset']['replay'])
        self.assertNotEqual(self.stage['dataset']['added'],self.first['dataset']['added']);self.assertNotEqual(self.stage['run'],self.first['run'])
        self.assertEqual((self.stages.teacher_of(self.stage),self.stages.teacher_of(self.first)),('stop_safe','surface'))

    def test_the_sets_have_the_make_up_of_stage_1_from_new_seeds_and_new_starts(self):
        train=self.plan['train'];validation=self.sets['validation'];pilot=self.sets['pilot']
        self.assertEqual((len(train),len(self.plan['val']),len(validation),len(pilot),len(self.sets['regression'])),(210,24,36,15,36))
        for kind,targets in self.stage['targets'].items():
            self.assertEqual((self.count(train,surface_kind=kind,task='land'),self.count(train,surface_kind=kind,task='approach')),(54,16))
            self.assertEqual((self.count(validation,surface_kind=kind,task='land'),self.count(validation,surface_kind=kind,task='approach')),(10,2))
            self.assertEqual((self.count(pilot,surface_kind=kind,task='land'),self.count(pilot,surface_kind=kind,task='approach')),(4,1))
            self.assertEqual(self.count(validation,surface_kind=kind,task='land',initially_visible=True),5)
            for target in targets:self.assertEqual(self.count(train,target=target,task='land'),27)
        self.assertEqual(self.count(train,initially_visible=True,task='land'),81)
        low,high=self.stage['height_m'];everything=train+self.plan['val']+validation+pilot
        self.assertTrue(all(low<=e['start_height_m']<=high and 10.-.01<=e['start_distance_m']<=44.01 for e in everything))
        self.assertTrue(all(e['start_distance_m']<=44.01 for e in self.sets['regression']))
        # New seeds, new names, and no start of stage 1 among the starts.
        used=lambda stage:{e['seed'] for name,path in self.stages.files_of(stage).items() for e in json.loads(path.read_text(encoding='utf-8'))['episodes']}| \
                          {e['seed'] for split in ('train','val') for e in json.loads(self.stages.train_plan_of(stage).read_text(encoding='utf-8'))[split]}
        mine,theirs=used(self.stage),used(self.first);self.assertFalse(mine&theirs);self.assertGreaterEqual(min(mine),50000);self.assertLess(max(theirs),50000)
        spots=lambda stage:{(e['map'],tuple(e['start_xy'])) for name,path in self.stages.files_of(stage).items() for e in json.loads(path.read_text(encoding='utf-8'))['episodes']}| \
                           {(e['map'],tuple(e['start_xy'])) for split in ('train','val') for e in json.loads(self.stages.train_plan_of(stage).read_text(encoding='utf-8'))[split]}
        self.assertFalse(spots(self.stage)&spots(self.first))
        self.assertTrue(all(e['id'].startswith('g1b') for e in everything+self.sets['regression']))

    def test_the_plans_are_reproducible_separate_and_in_the_right_layouts(self):
        import math
        import scripts.generic_landing as tool
        from src.visual_search.episodes import load_config
        from src.aerovla_oft.spec import load_config as load_oft
        self.assertEqual(tool.check(load_config(),load_oft(),self.stage),[])
        recorded={e['layout'] for split in ('train','val') for e in self.plan[split]};flown={e['layout'] for name in self.sets for e in self.sets[name]}
        self.assertTrue(recorded<=set(self.stage['layouts']['train']));self.assertTrue(flown<=set(self.stage['layouts']['evaluation']));self.assertFalse(recorded&flown)
        self.assertFalse((recorded|flown)&{'e','g','h','i','j'})
        # The two exceptions are the ones written down: the recorded starts may be near stage 1's recorded starts, the regression
        # starts near stage 1's regression starts; every check start of the pilot and the validation keeps away from everything.
        self.assertEqual(self.stage['training']['may_be_near'],['configs/generic_landing/train_plan.json'])
        self.assertEqual(self.stage['regression']['may_be_near'],['configs/generic_landing/regression.json'])
        earlier=self.stages.earlier_points(*self.stages.previous_files(self.stage))
        for episode in self.sets['pilot']+self.sets['validation']:
            self.assertFalse(any(where==episode['map'] and math.dist(episode['start_xy'],xy)<self.stage['separation_m'] for where,xy in earlier),episode['id'])

    def test_every_planned_landing_is_flown_by_the_stage_1b_teacher_in_its_dry_run(self):
        from src.landing.surface import SurfaceSet
        from src.landing.teacher import rollout
        from src.visual_search.maps import load_map
        config,oft,_,_=setting();scenes={}
        # A sample of every set: the plan's own record of the dry run is what the stage 1b teacher gives.
        for episode in self.plan['train'][::17]+self.sets['validation'][::5]+self.sets['pilot'][::4]:
            scene=scenes.setdefault(episode['map'],load_map(episode['map']))
            result=rollout(config,oft,scene,episode['layout'],episode,SurfaceSet(scene,episode['layout'])[episode['target']],kind='stop_safe')
            self.assertEqual((result['reason'],result['landed'],result['ticks'],result['states']),('teacher_stop',episode['task']=='land',episode['plan']['ticks'],episode['plan']['states']),episode['id'])


class GateTests(unittest.TestCase):
    def setUp(self):
        import scripts.generic_landing as tool
        self.tool=tool;self.stage=stage1b()

    def air_stop(self,kind):
        record=flight(kind,'land',False);record['surface_landing']['contact_from_above']=False;return record

    def test_the_candidate_is_chosen_by_the_order_of_stage_1b(self):
        def run(cube,cylinder,pad,air=0,wrong=0,hit=0):
            records=[flight('cube','land',i<cube) for i in range(4)]+[flight('cylinder','land',i<cylinder) for i in range(4)]+[flight('pad','land',i<pad) for i in range(4)]
            records+=[flight(kind,'approach',True) for kind in ('pad','cube','cylinder')]
            failed=[record for record in records if record['episode']['task']=='land' and not record['surface_landing']['success']]
            # A failed landing that is not a stop in the air: a touchdown that was no good landing.
            for record in failed[air:]:record['surface_landing']['contact_from_above']=True
            for index in range(wrong):records[index]['summary']['wrong_target']=True
            for index in range(hit):records[4+index]['summary']['hit']='Wall'
            return records
        self.assertEqual(self.stage['pilot']['order'][:3],['landings on cubes and cylinders (more)','flights that stop by themselves in the air during a landing (fewer)','landings on pads (more)'])
        # More landings on cubes and cylinders; then fewer stops in the air; then pads; then wrong objects; collisions; the loss; the earlier update.
        runs={'baseline':run(0,0,4,air=8),'1000':run(3,3,4,air=2),'1500':run(3,3,3,air=1),'2000':run(3,3,4,air=2,wrong=1),'2500':run(4,3,2,air=3)}
        rows,order,checks=self.tool.judge_pilot(self.stage,runs,{'1000':.07,'1500':.06,'2000':.065,'2500':.06})
        self.assertEqual(order,['2500','1500','1000','2000']);self.assertNotIn('baseline',order)
        by={row['candidate']:row for row in rows};self.assertEqual((by['1000']['air_self_stops'],by['1500']['air_self_stops'],by['baseline']['air_self_stops']),(2,1,8))
        self.assertEqual((by['2500']['cube_landings'],by['2500']['cylinder_landings'],by['2500']['pad_landings']),(4,3,2))
        self.assertEqual([row[0] for row in checks if not row[3]],['landings on pads'])
        rows,order,checks=self.tool.judge_pilot(self.stage,{'1000':run(3,3,4,air=1),'2000':run(3,3,4,air=1)},{'1000':.07,'2000':.06});self.assertEqual(order[0],'2000')
        self.assertTrue(all(row[3] for row in checks))

    def repeat(self,first,second,kind):
        make=lambda results:[flight(kind,'land',good,identifier=f'{kind}-{index}') for index,good in enumerate(results)]
        return make(first),make(second)

    def test_a_kind_of_surface_is_stable_only_if_two_flights_of_every_start_say_so(self):
        good=[True]*10
        def judge(cube_first,cube_second,cylinder_first=good,cylinder_second=good):
            a,b=self.repeat(cube_first,cube_second,'cube');c,d=self.repeat(cylinder_first,cylinder_second,'cylinder')
            table,checks=self.tool.judge_repeat(self.stage,a+c+[flight('pad','land',True) for _ in range(10)],b+d);return table,[row[0] for row in checks if not row[3]]
        self.assertEqual(judge(good,good)[1],[])
        # 8 and 8 of ten with one start failing twice: 16 of 20, passed, exactly at the limits.
        first=[False,False]+[True]*8;second=[False,True]+[True]*7+[False]
        table,failed=judge(first,second);self.assertEqual(failed,[]);self.assertEqual((table['cube']['aggregate'],table['cube']['neither'],table['cube']['one_of_two'],table['cube']['both']),(16,1,2,7))
        # Stage 1's cubes: 6 then 5 of ten, three starts failing twice.
        first=[True,False,False,True,False,True,True,True,True,False];second=[True,False,False,False,False,False,True,True,True,True]
        table,failed=judge(first,second);self.assertEqual((table['cube']['first_flights'],table['cube']['second_flights'],table['cube']['neither']),(6,5,3))
        self.assertEqual(failed,['cube landings, first flights','cube landings, both flights','cube starts that failed both times'])
        # A first flight of 8 that the second flight does not hold up, two starts that never land, and a start not flown twice.
        self.assertEqual(judge([False,False]+[True]*8,[True]*3+[False]*5+[True]*2)[1],['cube landings, both flights'])
        self.assertEqual(judge(good,good,[False,False]+[True]*8,[False,False]+[True]*8)[1],['cylinder starts that failed both times'])
        a,b=self.repeat(good,good[:9],'cube');c,d=self.repeat(good,good,'cylinder')
        self.assertIn('cube starts flown twice',[row[0] for row in self.tool.judge_repeat(self.stage,a+c,b+d)[1] if not row[3]])

    def test_the_regression_allows_no_collision_more_and_no_touchdown_in_an_approach(self):
        def run(success,wrong=0,hit=0,touched=0):
            records=[flight('pad','land',index<success) for index in range(27)]+[flight('pad','approach',index>=touched,landed=index<touched) for index in range(9)]
            for index in range(wrong):records[index]['summary']['wrong_target']=True
            for index in range(hit):records[index]['summary']['hit']='Hedge'
            return records
        failed=lambda model,baseline:[row[0] for row in self.tool.judge_regression(self.stage,model,baseline)[1] if not row[3]]
        self.assertEqual(failed(run(25),run(27)),[]);self.assertEqual(failed(run(24),run(27)),['mission success','landings'])
        self.assertEqual(failed(run(27,hit=1),run(27)),['collisions']);self.assertEqual(failed(run(27,hit=1),run(27,hit=1)),[])
        self.assertEqual(failed(run(27,wrong=2),run(27)),['flights at a wrong object']);self.assertEqual(failed(run(27,wrong=1),run(27)),[])
        self.assertEqual(failed(run(27,touched=1),run(27)),['touchdowns in an approach'])


class FailureTests(unittest.TestCase):
    def setUp(self):
        from src.landing import failures
        from src.visual_search.maps import load_landing
        self.failures=failures;self.config,self.oft,_,_=setting();self.landing=load_landing()
        self.cube={'landable':True,'top_m':8.,'usable_half_extent_m':[3.6,3.6]};self.pad={'landable':True,'top_m':2.5,'usable_half_extent_m':[6.6,6.6]}

    def step(self,distance,height,over=False,front=True,down=False,action=(0.,0.,0.)):
        return {'distance_m':distance,'height_m':height,'over':over,'front_seen':front,'down_seen':down,'action':list(action),'bearing_deg':0.,'yaw_rad':0.,'finalizer':'FLYING'}

    def test_the_phase_of_a_decision_is_read_from_where_the_vehicle_is(self):
        phase=lambda *arguments,surface=None,**more:self.failures.phase_of(self.step(*arguments,**more),surface or self.cube,self.landing,self.config)
        self.assertEqual(phase(30.,6.),'SURFACE_APPROACH');self.assertEqual(phase(30.,6.,front=False),'SEARCH')
        # Stage 1's six stops: two below the fly-in height short of the object, three at height or over it, one over the middle.
        self.assertEqual(phase(10.3,9.7),'CLIMB');self.assertEqual(phase(7.8,9.7,surface={'landable':True,'top_m':9.,'usable_radius_m':3.6}),'CLIMB')
        self.assertEqual(phase(8.2,12.),'ALIGNMENT');self.assertEqual(phase(6.6,16.5),'ALIGNMENT');self.assertEqual(phase(4.,10.7,over=True),'ALIGNMENT')
        self.assertEqual(phase(1.8,11.8,over=True,surface={'landable':True,'top_m':9.,'usable_radius_m':3.6}),'DESCENT');self.assertEqual(phase(1.,11.5,over=True),'DESCENT')
        # A pad is flown in at 2.5 m over its top.
        self.assertEqual(phase(9.,3.5,surface=self.pad),'CLIMB');self.assertEqual(phase(9.,5.,surface=self.pad),'ALIGNMENT')

    def record(self,**changes):
        reading={'success':False,'collision':None,'contact_from_above':False,'correct_object':False,'wrong_object':False,'wrong_object_landing':False,'inside_region':False,
                 'acceptable_touchdown':False,'target_surface':self.cube}
        summary={'reason':'model_stop','stopped':True,'wrong_target':False,'hit':None,'height_range_m':[5.,12.]}
        for key,value in changes.items():(summary if key in summary else reading)[key]=value
        return {'id':'x','episode':{'task':'land','surface_kind':'cube','start_distance_m':20.,'start_height_m':5.},'summary':summary,'surface_landing':reading}

    def test_a_failed_landing_is_named_by_how_it_ended(self):
        end=lambda **changes:self.failures.end_of(self.record(**changes))
        self.assertEqual(end(),'AIR_SELF_STOP');self.assertEqual(end(reason='collision',collision='BlueCube'),'COLLISION');self.assertEqual(end(reason='max_steps',stopped=False),'TIMEOUT')
        self.assertEqual(end(wrong_target=True),'TARGET_SELECTION');self.assertEqual(end(contact_from_above=True,correct_object=True),'TOUCHDOWN')
        self.assertEqual(end(contact_from_above=True,correct_object=True,inside_region=True,acceptable_touchdown=True),'STABILITY')

    def test_a_stop_in_the_air_is_logged_with_its_last_decisions(self):
        path=[self.step(20.,6.,action=(1.,0.,0.)),self.step(10.5,7.,action=(0.,-.5,0.)),self.step(10.3,9.,action=(0.,-.3,0.)),self.step(10.3,9.5,action=(0.,-.07,0.)),
              self.step(10.3,9.6,action=(.01,-.05,0.)),self.step(10.3,9.7,action=(0.,-.02,0.)),self.step(10.3,9.7,action=(0.,-.01,0.))]
        record=self.record();result=self.failures.failures([record,dict(self.record(),id='y')],{'x':path,'y':path},self.landing,self.config,self.oft)
        self.assertEqual((result['failed_landings'],result['by_end']['AIR_SELF_STOP'],result['air_self_stops_by_phase']['CLIMB'],result['by_kind']['cube']),(2,2,2,{'failed':2,'air_self_stop':2}))
        log=result['flights'][0]['self_stop'];self.assertEqual((log['height_above_top_m'],log['target_distance_m'],log['front_sees_target'],log['down_sees_target']),(1.7,10.3,True,False))
        self.assertEqual(len(log['last_actions']),4);self.assertTrue(all(ratio<=1. for ratio in log['last_stop_ratios']));self.assertEqual(log['phases_of_the_last_decisions'],['CLIMB']*4)
        self.assertEqual((log['start_distance_m'],log['start_height_m'],log['finalizer_state']),(20.,5.,'FLYING'))
        # How far the flown actions were from a stop, by phase, and how many flights came how close to four in a row.
        margins=self.failures.output_margins([record],{'x':path},self.landing,self.config,self.oft)
        self.assertEqual(margins['flights'],{'flights':1,'with_a_stop_decision':1,'with_two_in_a_row':1,'with_three_in_a_row':1,'ended_by_four':1})
        self.assertEqual((margins['by_phase']['CLIMB']['n'],margins['by_phase']['CLIMB']['stops'],margins['by_phase']['SURFACE_APPROACH']['n']),(6,4,1))
        over=self.failures.ceiling([self.record(height_range_m=[5.,16.5]),self.record()],14.);self.assertEqual((over['over'],over['highest_m'],over['flights']),(1,16.5,2))


class RecordingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):asyncio.get_running_loop().slow_callback_duration=60.

    async def test_the_stage_1b_teacher_is_recorded_and_the_recorded_labels_keep_the_rule(self):
        import projectairsim
        import scripts.surface_flights as worker
        import scripts.visual_search as canonical
        from src.landing import stop_safety
        from src.visual_search import generic_landing as stages
        from src.visual_search.generalization import SearchTeacher
        from src.visual_search.maps import load_map
        from tests.fake_airsim import FakeSim,FakeClient,world_class,drone_class
        stage=stages.load(STAGE);plan=json.loads(stages.train_plan_of(stage).read_text(encoding='utf-8'))['train']
        pick=lambda **fields:next(e for e in plan if all(e.get(k)==v for k,v in fields.items()))
        cube=next(e for e in plan if e['surface_kind']=='cube' and e['task']=='land' and e['plan']['climbed_m']>=3.)
        cylinder=pick(surface_kind='cylinder',task='land',initially_visible=False);twin=next(e for e in plan if e['task']=='approach' and e['surface_kind']=='cube')
        chosen=[cube,cylinder,twin]
        async def no_wait(_):return None
        clock=types.SimpleNamespace(sleep=no_wait,wait_for=asyncio.wait_for);config,oft,_,_=setting()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'plan.json';file.write_text(json.dumps({'train':chosen}))
            for teacher in ('stop_safe','surface'):
                data=root/teacher
                for episode in chosen:
                    sim=FakeSim(load_map(episode['map']),episode['layout'])
                    arguments=argparse.Namespace(host='fake',ports=[1,2],plan=str(file),set='train',output=str(root/f'run_{teacher}'),policy='teacher',checkpoint=None,model_name=None,
                                                 finalizer='off',record=str(data),only=[episode['id']],skip=0,limit=0,again=False,real_time_minimum=0.,teacher=teacher)
                    with patch.object(projectairsim,'ProjectAirSimClient',lambda **_:FakeClient(sim)),patch.object(canonical,'World',world_class(sim)), \
                         patch.object(canonical,'Drone',drone_class(sim)),patch.object(canonical,'matched_camera_images',sim.camera), \
                         patch.object(canonical,'asyncio',clock),patch('builtins.print'):
                        with self.assertRaises(SystemExit) as ended:await worker.fly(arguments)
                    self.assertEqual(ended.exception.code,0,(teacher,episode['id']));self.assertNotIn('land_async',sim.calls)
                self.assertIs(canonical.SearchTeacher,SearchTeacher)
            results=json.loads((root/'run_stop_safe/results.json').read_text());self.assertEqual(results['teacher'],'stop_safe')
            new=stop_safety.audit(stop_safety.recorded(root/'stop_safe'),oft,rule(),config['stop_ticks'])
            old=stop_safety.audit(stop_safety.recorded(root/'surface'),oft,rule(),config['stop_ticks'])
        # The same three starts: the stage 1b teacher's recorded labels keep the rule, stage 1's do not.
        self.assertTrue(new['passed']);self.assertEqual((new['total']['episodes'],new['by']['task']['LAND']['below_safe'],new['by']['task']['LAND']['near_stop_windows']),(3,0,0))
        self.assertGreaterEqual(new['by']['task']['LAND']['stop_ratio']['min'],3.);self.assertGreater(new['by']['task']['LAND']['terminal'],0)
        self.assertFalse(old['passed']);self.assertGreater(old['by']['task']['LAND']['below_safe'],0);self.assertLess(old['by']['task']['LAND']['stop_ratio']['min'],2.)
        self.assertEqual(new['by']['task']['APPROACH']['nonterminal_stop'],0);self.assertGreater(new['by']['task']['APPROACH']['terminal'],0)


if __name__=='__main__':unittest.main()
