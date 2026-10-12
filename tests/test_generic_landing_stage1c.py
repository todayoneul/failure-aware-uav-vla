"""Phase B, stage 1c (cube descent recovery and same-colour landing grounding): the teacher's descent rule, the forced
moves, the same-colour starts and their twins, what the data gives a student, the plans and the gates. The flights are the
canonical episode loop over a fake simulator; no model is loaded."""
import argparse
import asyncio
import itertools
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from tests.test_generic_landing import flight,setting,view

ROOT=Path(__file__).resolve().parents[1]


def stage1c():
    from src.visual_search import generic_landing_1c as plans
    return plans.load()


def sets():
    from src.visual_search import generic_landing_1c as plans
    stage=plans.load();out={name:json.loads(path.read_text(encoding='utf-8'))['episodes'] for name,path in plans.files_of(stage).items()}
    out['train_plan']=json.loads(plans.train_plan_of(stage).read_text(encoding='utf-8'));return stage,out


class DescentRuleTests(unittest.TestCase):
    def teacher(self,target,kind='recovery'):
        from src.landing.teacher import TEACHERS
        config,oft,scene,surfaces=setting();self.oft=oft;self.surface=surfaces[target]
        return TEACHERS[kind](config,oft,surface=surfaces[target])

    def over(self,teacher,distance,height,top=8.,bearing=0.,on=True):
        return teacher.act(view(distance_m=distance,height_m=height,bearing_deg=bearing,visible=False,below=True,over=on,surface_m=top))

    def test_the_abort_radius_is_half_way_between_the_align_radius_and_the_end_of_the_usable_top(self):
        from src.landing.teacher import abort_radius
        from src.visual_search.maps import load_landing
        rule=load_landing()['teacher'];self.teacher('blue_cube')
        self.assertAlmostEqual(abort_radius(rule,self.surface),2.675);self.assertEqual(rule['commit_radius_m'],3.5)
        text=stage1c()['teacher']['descent']['abort_radius'];self.assertIn('2.675',text)
        # A pad keeps the canonical radius; a cube and a cylinder get the smaller one.
        self.assertEqual(self.teacher('blue_pad').landing['commit_radius_m'],3.5)
        for target in ('blue_cube','red_cube','green_cylinder','blue_cylinder'):self.assertAlmostEqual(self.teacher(target).landing['commit_radius_m'],2.675)

    def test_a_descent_starts_only_near_the_middle_and_goes_on_only_within_the_abort_radius(self):
        new=self.teacher('blue_cube');old=self.teacher('blue_cube','stop_safe')
        # Centred with the height: down. Not centred at that height: no descent, in toward the middle (premature descent blocked).
        self.assertEqual(self.over(new,1.5,11.5),([0.,.5,0.],'descend'));self.assertEqual(self.over(new,1.5,10.2),([0.,.25,0.],'descend'))
        for distance in (2.,2.6,3.4,4.5):
            action,state=self.over(new,distance,11.5,on=distance<4.);self.assertEqual(state,'final');self.assertAlmostEqual(action[0],.35);self.assertEqual(action[1],0.)
        # Low over the top and a little off the middle: the descent goes on (no climb back for a small error).
        for distance in (1.8,2.2,2.6):self.assertEqual(self.over(new,distance,10.)[1],'descend');self.assertEqual(self.over(new,distance,8.6)[0],[0.,.25,0.])
        # Low and farther out than the abort radius: stage 1b went on descending up to 3.5 m; now up and in again.
        for distance in (2.8,3.2,3.45):
            self.assertEqual(self.over(old,distance,10.)[1],'descend')
            action,state=self.over(new,distance,10.);self.assertEqual(state,'climb');self.assertLess(action[1],0.);self.assertAlmostEqual(action[0],.35)
        # Under-climbed, over the edge: up in place until clear of the top, then up and in; never down.
        action,state=self.over(new,3.9,8.5);self.assertEqual((state,action[0],action[1]),('climb',0.,-.5))
        action,state=self.over(new,4.5,8.6,on=False);self.assertEqual((state,action[0],action[1]),('climb',0.,-.5))
        action,state=self.over(new,3.9,9.5);self.assertEqual(state,'climb');self.assertAlmostEqual(action[0],.35);self.assertEqual(action[1],-.5)
        # Past the middle at height (edge, corner): turn to the middle first.
        action,state=self.over(new,3.6,11.6,bearing=80.);self.assertEqual((state,action[0]),('align',0.));self.assertAlmostEqual(action[2],.21)
        # Touchdown: the stop, and only then.
        self.assertEqual(new.act(view(distance_m=2.,height_m=8.,landed=True,over=True)),([0.,0.,0.],'landed'))

    def test_a_pad_and_everything_but_the_descent_radius_are_stage_1b(self):
        for target,top in (('blue_pad',2.5),('red_pad',2.5)):
            new=self.teacher(target);old=self.teacher(target,'stop_safe')
            for height,distance,bearing,seen,task in itertools.product((3.,4.,5.,6.,9.,12.),(.5,1.5,2.,3.,3.4,5.,9.,12.,20.,40.),(-90.,-20.,0.,8.,35.),((True,False,False),(False,True,True),(False,False,False)),('land','approach')):
                item=view(height_m=height,distance_m=distance,bearing_deg=bearing,visible=seen[0],below=seen[1],over=seen[2],surface_m=top,task=task)
                self.assertEqual(new.act(dict(item)),old.act(dict(item)),(target,item))
        new=self.teacher('blue_cube');old=self.teacher('blue_cube','stop_safe');differ=0
        for height,distance,bearing,seen in itertools.product((5.,8.5,9.5,10.,10.9,11.5,12.5),(.5,1.5,2.,2.6,2.7,3.,3.4,3.6,5.,9.,12.,20.,40.),(-90.,-20.,0.,8.,35.),((True,False,False),(False,True,True),(False,False,False))):
            if seen[2] and distance>5.5:continue
            item=view(height_m=height,distance_m=distance,bearing_deg=bearing,visible=seen[0],below=seen[1],over=seen[2],surface_m=8.)
            if new.act(dict(item))!=old.act(dict(item)):
                # The two differ only low over the top between the two radii.
                differ+=1;self.assertTrue(seen[2] and 2.675<distance<=3.5 and height-8.<=3.,item)
        self.assertGreater(differ,0)

    def test_every_label_before_the_touchdown_still_keeps_the_margin(self):
        from src.aerovla_oft.spec import is_stop
        from src.landing.stop_safety import stop_ratio
        safe=stage1c()['stop_safety']['safe_ratio'];self.assertEqual(safe,3.)
        for target,top in (('blue_cube',8.),('green_cylinder',9.),('blue_pad',2.5)):
            teacher=self.teacher(target);count=0
            for height,distance,bearing,seen in itertools.product((3.,5.,top-.5,top+.4,top+1.2,top+2.,top+2.9,top+3.3,top+3.6,top+5.),(.5,1.5,1.8,2.5,2.7,3.,3.4,3.6,5.,7.,9.,11.9,12.1,18.,24.9,30.,44.),
                                                                  (-170.,-40.,-8.,0.,5.,12.,29.,31.,90.),((True,False,False),(True,True,False),(False,True,True),(False,False,True),(False,False,False))):
                if seen[2] and (height<top or distance>5.5):continue
                if height<top and distance<4.5:continue
                action,state=teacher.act(view(height_m=height,distance_m=distance,bearing_deg=bearing,visible=seen[0],below=seen[1],over=seen[2],surface_m=top));count+=1
                self.assertFalse(is_stop(action,self.oft),(target,action,state));self.assertGreaterEqual(stop_ratio(action,self.oft),safe,(target,height,distance,bearing,seen,action,state))
            self.assertGreater(count,1500)

    def test_without_the_named_object_in_view_the_search_knows_nothing_of_where_it_is(self):
        from src.visual_search.generalization import SearchTeacher
        config,oft,scene,surfaces=setting();old=SearchTeacher(config,oft)
        for target in ('blue_pad','blue_cube','red_cube'):
            new=self.teacher(target);answers=set()
            for bearing,distance,height in itertools.product((-170.,-60.,0.,45.,120.),(5.,20.,44.),(4.,9.,12.)):
                item=view(bearing_deg=bearing,distance_m=distance,height_m=height,visible=False,below=False,over=False)
                answer=new.act(dict(item));self.assertEqual(answer,old.act(dict(item)));answers.add((tuple(answer[0]),answer[1]))
            self.assertEqual(answers,{((0.,0.,.21),'search')})


class ForcedMoveTests(unittest.TestCase):
    def pushes(self,*items):
        from src.landing.teacher import RecoveryPushes
        return RecoveryPushes({'pushes':list(items)})

    def at(self,distance,above,bearing=0.,top=8.):return {'bearing_deg':bearing,'distance_m':distance,'height_m':top+above,'surface_m':top}

    def test_each_forced_move_is_due_at_a_place_and_ends_at_its_state(self):
        # Under-climb: only while the vehicle is in the band on its way up; then straight in without climbing, to `until_m`.
        push=self.pushes({'when':'under_climb','within_m':25.,'rise_m':.8,'band_m':.6,'until_m':4.,'forward_m':.5,'ticks':60})
        self.assertIsNone(push.step('climb',self.at(11.,.3),.21));self.assertIsNone(push.step('final',self.at(11.,3.5),.21));self.assertFalse(push.pending())
        self.assertEqual(push.step('climb',self.at(11.,1.),.21),[.5,0.,0.]);self.assertTrue(push.pending());self.assertEqual(push.step('climb',self.at(6.,1.2,bearing=9.),.21)[1],0.)
        self.assertIsNone(push.step('climb',self.at(3.9,1.2),.21));self.assertFalse(push.pending());self.assertEqual(push.fired[0]['when'],'under_climb')
        # Edge: the middle is kept to one side until it is abeam.
        push=self.pushes({'when':'edge','from_m':9.,'min_above_m':1.,'miss_m':3.5,'side':1,'forward_m':.5,'ticks':40})
        self.assertIsNone(push.step('final',self.at(9.5,3.5),.21));self.assertIsNone(push.step('climb',self.at(8.5,.5),.21))
        action=push.step('final',self.at(8.5,3.5),.21);self.assertLess(action[2],0.);self.assertEqual(action[1],0.);self.assertGreater(action[0],0.)
        self.assertIsNone(push.step('final',self.at(3.7,3.5,bearing=80.),.21))
        # Premature: down to the floor over the top, slowly on.
        push=self.pushes({'when':'premature','from_m':5.5,'floor_m':1.5,'forward_m':.2,'down_m':.5,'ticks':12})
        self.assertIsNone(push.step('final',self.at(7.,3.5),.21));self.assertEqual(push.step('final',self.at(5.4,3.5),.21),[.2,.5,0.])
        self.assertAlmostEqual(push.step('final',self.at(5.2,1.7),.21)[1],.25);self.assertIsNone(push.step('climb',self.at(5.,1.5),.21))
        # Stall: lifted during the descent, for its decisions; due only while the teacher is descending.
        push=self.pushes({'when':'stall','below_m':2.,'up_m':.4,'forward_m':.1,'ticks':2})
        self.assertIsNone(push.step('final',self.at(1.5,1.8),.21));self.assertIsNone(push.step('descend',self.at(1.5,2.4),.21))
        self.assertEqual(push.step('descend',self.at(1.5,1.9),.21),[.1,-.4,0.]);self.assertEqual(push.step('descend',self.at(1.5,2.3),.21),[.1,-.4,0.])
        self.assertIsNone(push.step('descend',self.at(1.5,2.7),.21))

    def test_a_push_that_is_not_yet_due_does_not_keep_a_flight_going_and_canonical_pushes_are_untouched(self):
        from src.visual_search.generalization import Pushes
        push=self.pushes({'when':'stall','below_m':2.,'up_m':.4,'ticks':2});self.assertTrue(push.forced);self.assertFalse(push.pending())
        item={'when':'landed','ticks':2,'up_m':.4};mine=self.pushes(item);theirs=Pushes({'pushes':[item]});self.assertFalse(mine.forced);self.assertEqual(mine.pending(),theirs.pending())
        for state,place in (('descend',self.at(1.,1.)),('landed',self.at(1.,0.)),('landed',self.at(1.,0.)),('landed',self.at(1.,.4))):
            self.assertEqual(mine.step(state,place,.21),theirs.step(state,place,.21))
        self.assertFalse(self.pushes().forced);self.assertFalse(self.pushes().pending())

    def test_the_dry_run_flies_every_forced_state_and_the_teacher_lands_with_labels_clear_of_a_stop(self):
        from src.landing import stop_safety
        from src.landing.teacher import rollout
        from src.visual_search.maps import object_position
        config,oft,scene,surfaces=setting();rule={'safe_ratio':3.,'near_stop_ratio':2.}
        forced={'under_climb':[{'when':'under_climb','within_m':25.,'rise_m':.8,'band_m':.6,'until_m':4.5,'forward_m':.5,'ticks':60}],
                'edge':[{'when':'edge','from_m':9.,'min_above_m':1.,'miss_m':3.5,'side':-1,'forward_m':.5,'ticks':40}],
                'premature':[{'when':'premature','from_m':5.5,'floor_m':1.5,'forward_m':.2,'down_m':.5,'ticks':12}],
                'stall':[{'when':'stall','below_m':2.,'up_m':.4,'forward_m':.2,'ticks':3},{'when':'stall','below_m':1.,'up_m':.5,'forward_m':0.,'ticks':2}]}
        for target in ('blue_cube','green_cylinder'):
            centre=object_position(scene,'a',target)
            for name,pushes in forced.items():
                episode={'start_xy':[centre[0]-14.,centre[1]-3.],'start_yaw_deg':10.,'start_height_m':3.5,'target':target,'task':'land','strategy':'right','max_ticks':320,'pushes':pushes}
                trace=[];result=rollout(config,oft,scene,'a',episode,surfaces[target],kind='recovery',trace=trace)
                self.assertEqual((result['reason'],result['landed'],result['forced_left']),('teacher_stop',True,0),(target,name));self.assertEqual(len(result['forced']),len(pushes))
                self.assertTrue(surfaces[target].inside([result['final_distance_m'],0.]));self.assertGreater(result['forced_ticks'],0)
                one=stop_safety.audit_episode(trace,'land',oft,rule,config['stop_ticks'])
                self.assertEqual((one['nonterminal_stop'],one['below_safe'],one['stop_windows'],one['near_stop_windows']),(0,0,0,0),(target,name));self.assertEqual(one['forced'],result['forced_ticks'])
                let_go=[after for before,after in zip(trace,trace[1:]) if before['perturbed'] and not after['perturbed']]
                self.assertTrue(let_go and all(row['seen'] for row in let_go))
                # After a stall the label is the descent again, at once; after the other three it is first up or to the middle.
                first=let_go[0]['teacher_state'];self.assertEqual(first,'descend') if name=='stall' else self.assertIn(first,('climb','final','align'))
        # A plain episode is flown exactly as before.
        plain={'start_xy':[centre[0]-14.,centre[1]-3.],'start_yaw_deg':10.,'start_height_m':3.5,'target':'green_cylinder','task':'land','strategy':'right','max_ticks':320}
        self.assertEqual(rollout(config,oft,scene,'a',plain,surfaces['green_cylinder'],kind='recovery')['forced_ticks'],0)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.visual_search import generic_landing as stages
        cls.stage,cls.sets=sets();cls.train=cls.sets['train_plan']['train'];cls.val=cls.sets['train_plan']['val'];cls.before=stages.load(ROOT/'configs/generic_landing_stage1b.json')

    def count(self,episodes,**fields):
        return sum(all(episode.get(key)==value for key,value in fields.items()) for episode in episodes)

    def test_the_recorded_episodes_are_the_two_corrections_and_nothing_else(self):
        self.assertEqual((len(self.train),len(self.val)),(152,16))
        forced=[e for e in self.train if e.get('recovery')];same=[e for e in self.train if e.get('distractor')];self.assertEqual((len(forced),len(same)),(80,72))
        for state in ('under_climb','edge','premature','stall'):
            self.assertEqual(self.count(forced,recovery=state,surface_kind='cube'),16);self.assertEqual(self.count(forced,recovery=state,surface_kind='cylinder'),2)
        self.assertEqual((self.count(forced,recovery='edge',surface_kind='pad'),self.count(forced,recovery='stall',surface_kind='pad'),self.count(forced,surface_kind='pad')),(4,4,8))
        # Every forced landing came to pass in the teacher's dry run and let go with the object in view; none repeats a start of stage 1b's checks.
        for episode in forced:
            self.assertEqual(len(episode['plan']['forced']),len(episode['pushes']),episode['id']);self.assertEqual(len(episode['plan']['let_go']),len(episode['pushes']))
            self.assertTrue(all(item['when']==episode['recovery'] for item in episode['pushes']));self.assertEqual(episode['task'],'land')
        # Same colour: the plan's make-up, sixteen twin pairs.
        pairs=lambda target,other,twin:sum(e['target']==target and e['distractor']==other and bool(e.get('twin_of'))==twin for e in same)
        self.assertEqual((pairs('blue_pad','blue_cube',False),pairs('blue_pad','blue_cone',False),pairs('blue_cube','blue_pad',False),pairs('blue_cube','blue_pad',True)),(20,12,6,10))
        self.assertEqual((pairs('red_pad','red_cube',False),pairs('red_cube','red_pad',False),pairs('red_cube','red_pad',True)),(12,6,6))
        self.assertTrue(all(e['task']=='land' and e['start_distance_m']<=50. for e in same));self.assertFalse(any(e['start_distance_m']>44.01 for e in same if not e.get('twin_of')))

    def test_a_same_colour_start_shows_the_other_object_first_and_its_twin_names_that_object(self):
        from src.landing.surface import SurfaceSet
        from src.landing.teacher import rollout
        from src.mission import start as starts
        from src.visual_search import generic_landing_1c as plans
        from src.visual_search.maps import MapGeometry,load_map
        config,oft,_,_=setting();scenes={};catalogue=load_map('field')['objects']
        everything=[e for e in self.train+self.val+self.sets['same_color']+self.sets['pilot']+self.sets['teacher_pilot'] if e.get('distractor')];by={e['id']:e for e in everything+self.sets['same_color_twins']}
        self.assertGreater(len(everything),120)
        for episode in everything[::3]+self.sets['same_color_twins']:
            scene=scenes.setdefault(episode['map'],load_map(episode['map']));geometry=MapGeometry(scene,episode['layout']);target,other=episode['target'],episode['distractor']
            # One colour, two shapes; both stand in the layout.
            self.assertEqual(catalogue[target]['attributes']['color'],catalogue[other]['attributes']['color']);self.assertNotEqual(target,other)
            start=starts.StartState.from_episode(episode,config['cruise_height_m']);seen=starts.view_of(config,geometry,target,start)
            if episode.get('twin_of'):
                first=by[episode['twin_of']];self.assertTrue(seen['initially_visible'])
                self.assertEqual((episode['start_xy'],episode['start_yaw_deg'],episode['start_height_m'],episode['map'],episode['layout']),(first['start_xy'],first['start_yaw_deg'],first['start_height_m'],first['map'],first['layout']))
                self.assertEqual((episode['target'],episode['distractor']),(first['distractor'],first['target']));self.assertNotEqual(episode['instruction'],first['instruction'])
                self.assertIn(catalogue[target]['noun'],episode['instruction']);self.assertNotIn(catalogue[other]['noun'],episode['instruction'])
            else:
                self.assertFalse(seen['initially_visible']);self.assertTrue(seen['line_of_sight']);self.assertIn(other,[item['name'] for item in seen['others_in_view']])
                self.assertTrue(episode['distractor_first']);self.assertLessEqual(episode['distractor_distance_m'],50.01)
            # The teacher lands on the named object, and its track never goes near the other one.
            trace=[];plan=rollout(config,oft,scene,episode['layout'],episode,SurfaceSet(scene,episode['layout'])[target],kind='recovery',trace=trace)
            self.assertEqual((plan['reason'],plan['landed']),('teacher_stop',True),episode['id']);self.assertFalse(plans.near_other(trace,geometry,target,other,config['arrive_distance_m']),episode['id'])
            # Until a camera shows the named object the label is the blind search, whatever else is in view.
            blind=[row for row in trace if not row['seen']];self.assertTrue(all(row['teacher_action'] in ([0.,0.,.21],[0.,-.5,0.]) for row in blind),episode['id'])
            if not episode.get('twin_of'):self.assertGreater(len(blind),0)

    def test_the_check_sets_are_what_the_configuration_says(self):
        gate_a=self.sets['validation'];gate_b=self.sets['same_color'];pilot=self.sets['pilot'];teacher=self.sets['teacher_pilot']
        self.assertEqual((len(gate_a),len(gate_b),len(self.sets['same_color_twins']),len(pilot),len(teacher),len(self.sets['regression'])),(36,24,11,24,32,36))
        for kind in ('pad','cube','cylinder'):
            self.assertEqual((self.count(gate_a,surface_kind=kind,task='land'),self.count(gate_a,surface_kind=kind,task='approach')),(10,2))
        self.assertFalse(any(e.get('pushes') or e.get('distractor') for e in gate_a))
        rows=collections_of(gate_b);self.assertEqual(rows,{('blue_pad','blue_cube'):6,('blue_pad','blue_cone'):4,('blue_cube','blue_pad'):4,('red_pad','red_cube'):5,('red_cube','red_pad'):5})
        self.assertTrue(all(e['distractor_first'] and not e.get('twin_of') and e['task']=='land' for e in gate_b))
        self.assertEqual((self.count(pilot,surface_kind='cube',task='land'),self.count(pilot,surface_kind='pad',task='land'),self.count(pilot,surface_kind='cylinder',task='land')),(10,10,4))
        self.assertEqual(sum(bool(e.get('recovery')) for e in pilot),8);self.assertEqual(sum(bool(e.get('distractor')) for e in pilot),8)
        self.assertEqual((sum(bool(e.get('recovery')) for e in teacher),sum(bool(e.get('distractor')) for e in teacher),sum(bool(e.get('twin_of')) for e in teacher)),(16,16,4))
        # The twelve starts of gate B flown twice: every cone start and the nearest of the rest.
        from src.visual_search import generic_landing_1c as plans
        again=plans.repeat_ids(self.stage,gate_b);self.assertEqual(len(again),12);self.assertTrue({e['id'] for e in gate_b if e['distractor']=='blue_cone'}<=set(again))

    def test_the_gates_are_the_plans_numbers_and_gate_a_is_stage_1s(self):
        first=json.loads((ROOT/'configs/generic_landing.json').read_text(encoding='utf-8'))
        self.assertEqual(self.stage['validation']['gates'],first['validation']['gates']);self.assertEqual(self.stage['validation']['smoke'],first['validation']['smoke'])
        repeat=self.stage['validation']['repeat'];self.assertEqual((repeat['aggregate_min'],repeat['first_flight_min'],repeat['both_failed_max']),({'cube':16,'cylinder':18},8,1))
        gates=self.stage['same_color']['gates']
        self.assertEqual((gates['of'],gates['correct_completion_min'],gates['wrong_target_landings_max'],gates['collisions_with_the_other_max'],gates['collisions_max'],gates['wrong_terminal_max'],gates['rejection_min']),(24,22,0,0,0,0,22))
        self.assertGreaterEqual(gates['rejection_min']/gates['of'],.9)
        regression=self.stage['regression'];self.assertEqual((regression['collisions_max'],regression['approach_touchdowns_max'],regression['success_below_baseline_max'],regression['landings_below_baseline_max']),(0,0,2,1))
        self.assertEqual(self.stage['stop_safety']['safe_ratio'],self.before['stop_safety']['safe_ratio'])

    def test_the_training_continues_stage_1b_with_its_safe_data_and_none_of_stage_1s(self):
        tune=self.stage['fine_tune'];data=self.stage['dataset']
        self.assertEqual(self.stage['start_checkpoint'],self.before['run']+'/update-2000');self.assertEqual(tune['learning_rate'],self.before['fine_tune']['learning_rate']/2)
        self.assertEqual(tune['grounding_learning_rate'],tune['learning_rate']);self.assertEqual((tune['updates'],tune['candidates']),(2000,[500,1000,1500,2000]))
        self.assertEqual(tune['share'],{'generic_landing_1c_added':.35,'generic_landing_1b_added':.35});self.assertEqual(data['stop_safe'],self.before['dataset']['added'])
        self.assertEqual(data['not_used'],['generic_landing_added']);self.assertNotIn('generic_landing_added',data['replay']+[data['stop_safe'],data['added']])
        self.assertEqual(data['replay'],self.before['dataset']['replay'])
        for key in ('batch_size','gradient_accumulation','eval_every','eval_samples','balance','boost','select_band','val_file'):self.assertEqual(tune[key],self.before['fine_tune'][key],key)
        # The checkpoint it starts from is named by its fingerprint, not as generic_landing_v1.
        self.assertEqual(len(self.stage['start_checkpoint_sha256']),4);self.assertNotIn('generic_landing_v1',self.stage['start_checkpoint'])

    def test_the_plans_are_reproducible_and_no_start_or_seed_is_shared(self):
        import scripts.generic_landing_1c as tool
        from src.visual_search import generic_landing as stages
        from src.visual_search.episodes import load_config
        from src.aerovla_oft.spec import load_config as load_oft
        self.assertEqual(tool.check(load_config(),load_oft(),self.stage),[])
        mine=[e for name in ('validation','same_color','pilot','teacher_pilot','regression') for e in self.sets[name]]+self.train+self.val
        seeds={e['seed'] for e in mine};self.assertGreaterEqual(min(seeds),60000)
        earlier=set();spots=set()
        for path in (ROOT/'configs').rglob('*.json'):
            if path.parent.name=='generic_landing_stage1c':continue
            def walk(item):
                if isinstance(item,dict):
                    if 'start_xy' in item and item.get('map'):spots.add((item['map'],tuple(item['start_xy'])))
                    if isinstance(item.get('seed'),int):earlier.add(item['seed'])
                    for value in item.values():walk(value)
                elif isinstance(item,list):
                    for value in item:walk(value)
            walk(json.loads(path.read_text(encoding='utf-8')))
        self.assertFalse(seeds&{seed for seed in earlier if seed>=1000});self.assertFalse({(e['map'],tuple(e['start_xy'])) for e in mine}&spots)
        # Sets of this stage share no seed: recorded, loss, pilots, the two gates, the regression.
        ranges=self.stage['seeds'];spans=sorted(ranges.values());self.assertTrue(all(a[1]<b[0] for a,b in zip(spans,spans[1:])))
        inside=lambda episodes,name:all(ranges[name][0]<=e['seed']<=ranges[name][1] for e in episodes)
        for name in ('validation','same_color','same_color_twins','pilot','teacher_pilot','regression'):self.assertTrue(inside(self.sets[name],'same_color' if name=='same_color_twins' else name),name)
        for split,episodes in (('train',self.train),('val',self.val)):
            self.assertTrue(inside([e for e in episodes if e.get('recovery')],f'{split}_recovery'));self.assertTrue(inside([e for e in episodes if e.get('distractor')],f'{split}_same_color'))
        recorded={e['layout'] for e in self.train+self.val};flown={e['layout'] for name in ('validation','same_color','same_color_twins','pilot','teacher_pilot','regression') for e in self.sets[name]}
        self.assertTrue(recorded<=set(self.stage['layouts']['train']));self.assertTrue(flown<=set(self.stage['layouts']['evaluation']));self.assertFalse((recorded|flown)&{'e','g','h','i','j'})


def collections_of(episodes):
    out={}
    for episode in episodes:out[(episode['target'],episode['distractor'])]=out.get((episode['target'],episode['distractor']),0)+1
    return out


class InputTests(unittest.TestCase):
    def dataset(self,root,episodes):
        folder=root/'episodes';folder.mkdir(parents=True)
        for episode in episodes:
            step={'teacher_action':[1.,0.,0.],'teacher_state':'approach','landed':False,'distance_m':20.,'bearing_deg':3.,'position':[0.,0.,0.],'front_seen':True,'down_seen':False,'over':False}
            (folder/f'{episode["id"]}.json').write_text(json.dumps({'summary':episode,'traj_rel_dir':f'{episode["map"]}/{episode["id"]}','steps':[step]}))

    def test_the_data_gives_a_student_two_pictures_and_a_sentence_from_the_catalogue(self):
        import scripts.generic_landing_1c as tool
        from src.visual_search.episodes import load_config
        stage,everything=sets();train=everything['train_plan']['train'];twin=next(e for e in train if e.get('twin_of'));first=next(e for e in train if e['id']==twin['twin_of'])
        forced=next(e for e in train if e.get('recovery'));config=load_config()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.dataset(root/'good',[first,twin,forced]);result=tool.inputs_audit(root/'good',stage,config)
            self.assertTrue(result['passed']);self.assertEqual((result['episodes'],result['sentences'],result['twins'],result['twins_with_their_landing'],result['proprio_enabled']),(3,len({first['instruction'],twin['instruction'],forced['instruction']}),1,1,False))
            self.assertEqual(result['sentences_not_from_the_catalogue_or_with_a_number_or_an_id'],[]);self.assertIn(first['instruction'],result['examples']);self.assertNotEqual(first['instruction'],twin['instruction'])
            # A sentence that carries a distance, a bearing or an object's id is not one the catalogue builds.
            for number,text in enumerate(('Find the blue landing pad 31 m ahead and land on it.','Find blue_pad and land on it.','Find the blue landing pad to the left and land on it.')):
                self.dataset(root/f'leak{number}',[dict(first,instruction=text)]);self.assertFalse(tool.inputs_audit(root/f'leak{number}',stage,config)['passed'],text)
            # A twin that does not share its start, or names the same object, does not fit.
            self.dataset(root/'moved',[first,dict(twin,start_xy=[twin['start_xy'][0]+1.,twin['start_xy'][1]])]);self.assertEqual(tool.inputs_audit(root/'moved',stage,config)['twins_that_do_not_fit'],[twin['id']])
            self.dataset(root/'same',[first,dict(twin,target=first['target'])]);self.assertFalse(tool.inputs_audit(root/'same',stage,config)['passed'])

    def test_the_model_is_given_the_mosaic_and_the_sentence_and_nothing_of_the_target(self):
        source=(ROOT/'scripts/visual_search.py').read_text(encoding='utf-8')
        self.assertIn("inputs=policy_inputs(observation['frames']['front'],observation['frames']['down'],episode['instruction'],",source)
        self.assertIn("inference=policy.infer(inputs['front'],inputs['down'],inputs['instruction'],inputs['proprio'])",source)
        builder=(ROOT/'scripts/build_visual_search_dataset.py').read_text(encoding='utf-8')
        # The builder keeps the target, its bearing and its distance as analysis fields under `meta`; the label and the sentence are what is trained on.
        self.assertIn("'instruction':'<image>\\n'+summary['instruction'],",builder);self.assertIn("'meta':{'episode_id':summary['id']",builder)
        config=json.loads((ROOT/'configs/aerovla_oft.json').read_text(encoding='utf-8'));self.assertFalse(config['proprio']['enabled'])
        self.assertFalse(any(word in field for field in config['proprio']['fields'] for word in ('target','bearing','distance','position')))


class ReadingTests(unittest.TestCase):
    def setUp(self):
        from src.landing import failures
        self.failures=failures;self.config,self.oft,_,_=setting()

    def record(self,**changes):
        reading={'success':False,'collision':None,'contact_from_above':False,'landed_on':None,'inside_region':False,'horizontal_error_m':None,'target_surface':{'landable':True,'top_m':8.,'usable_half_extent_m':[3.6,3.6]}}
        summary={'reason':'model_stop','stopped':True,'hit':None,'wrong_target':False,'height_range_m':[5.,12.]}
        for key,value in changes.items():(summary if key in summary else reading)[key]=value
        return {'id':'x','episode':{'task':'land','surface_kind':'cube','target':'blue_pad','distractor':'blue_cone','start_distance_m':25.},'summary':summary,'surface_landing':reading}

    def step(self,distance,height,down=0.,forward=0.,over=False,position=(0.,0.),front=True):
        return {'distance_m':distance,'height_m':height,'over':over,'front_seen':front,'down_seen':over,'action':[forward,down,0.],'bearing_deg':0.,'yaw_rad':0.,'position':list(position)+[0.],'landed':False,'on_surface':False}

    def test_the_descent_profile_reads_where_a_landing_began_to_come_down_and_how_often_it_changed_its_mind(self):
        from src.visual_search.maps import load_landing
        landing=load_landing();profile=lambda steps,**changes:self.failures.descent_profile(self.record(**changes),steps,landing,self.config,self.oft)
        good=[self.step(20.,6.,forward=1.),self.step(11.,9.,down=-.5),self.step(9.,11.5,forward=.5),self.step(1.6,11.5,down=.5,over=True),self.step(1.2,10.,down=.25,over=True),self.step(1.,8.3,down=.25,over=True)]
        row=profile(good,success=True,landed_on='blue_pad',inside_region=True,horizontal_error_m=1.0,reason='landed')
        self.assertEqual((row['descent_start_distance_m'],row['descent_start_height_above_top_m'],row['sign_flips'],row['entry_height_above_top_m'],row['highest_above_top_m']),(1.6,3.5,0,3.5,3.5))
        # Stage 1b's three: low in over the edge; down before the middle; up and down in turn over the middle.
        low=[self.step(12.,8.6,forward=.4),self.step(4.2,8.6,down=.3,over=False),self.step(3.9,8.4,down=.27,over=True)]
        hang=[self.step(2.3,10.7,down=.34,over=True)]+[self.step(1.6,9.6,down=sign*.2,over=True) for sign in (1,-1,1,1,-1,-1,1,-1)]
        rows=[profile(low,landed_on='blue_pad',inside_region=False,horizontal_error_m=3.8,reason='landed'),profile([self.step(11.,10.8,forward=.45),self.step(5.6,10.7,down=.24),self.step(5.3,8.4,down=.03,over=True)]),
              profile(hang,reason='max_steps',stopped=False),row]
        self.assertEqual((rows[0]['entry_height_above_top_m'],rows[0]['descent_start_distance_m'],rows[1]['descent_start_distance_m'],rows[2]['sign_flips'],rows[2]['climbs_after_the_descent_began']),(.4,4.2,5.6,5,4))
        counts=self.failures.cube_counts(rows,stage1c()['teacher'],3.6)
        # The flight that came down before the middle was 0.4 m over the top when it got over the object: counted as low in, too.
        self.assertEqual(counts,{'landings':4,'landed':1,'insufficient_climb':2,'premature_descent':2,'edge_touchdown':1,'descent_oscillation':1,'air_self_stop':1,'timeout':1})

    def test_a_same_colour_flight_is_read_against_the_other_object(self):
        other=(100.,0.);path=lambda *points:[self.step(distance,6.,position=position) for distance,position in points]
        read=lambda steps,**changes:self.failures.same_color_reading(self.record(**changes),steps,other,self.config['arrive_distance_m'])
        # Past the other object at a distance and on to the named one.
        row=read(path((30.,(60.,20.)),(15.,(60.,35.)),(1.,(60.,50.))),success=True,landed_on='blue_pad',reason='landed',contact_from_above=True)
        self.assertEqual((row['approached_the_other'],row['rejected_the_other'],row['wrong_terminal'],row['nearest_to_the_other_m']),(False,True,False,44.7))
        # To the cone and into it: stage 1b's collision.
        row=read(path((30.,(60.,20.)),(45.,(90.,5.)),(52.,(99.5,0.))),collision='SceneBlueCone',reason='collision',stopped=False)
        self.assertEqual((row['approached_the_other'],row['collision_with_the_other'],row['wrong_terminal'],row['rejected_the_other']),(True,True,True,False))
        # Onto the other object; a stop in the air beside it; a near pass on the way to a landing on the named one.
        row=read(path((30.,(60.,20.)),(50.,(99.,1.))),landed_on='blue_cone',reason='landed',contact_from_above=True);self.assertEqual((row['wrong_target_landing'],row['wrong_terminal']),(True,True))
        row=read(path((30.,(60.,20.)),(48.,(92.,3.))));self.assertEqual((row['stopped_at_the_other'],row['wrong_terminal'],row['rejected_the_other']),(True,True,False))
        row=read(path((30.,(60.,20.)),(20.,(90.,2.)),(1.,(70.,30.))),success=True,landed_on='blue_pad',reason='landed',contact_from_above=True)
        self.assertEqual((row['approached_the_other'],row['wrong_terminal'],row['rejected_the_other'],row['first_approach_of_the_other_at_decision']),(True,False,False,1))
        # A collision with the named object's own side is a collision, not a wrong object.
        row=read(path((30.,(60.,20.)),(4.,(60.,46.))),collision='BluePad',reason='collision',stopped=False);self.assertEqual((bool(row['collision']),row['collision_with_the_other'],row['wrong_terminal']),(True,False,False))
        self.assertEqual((self.failures.plain('SceneBlueCone'),self.failures.plain('blue_cone'),self.failures.plain('BlueCube')),('bluecone','bluecone','bluecube'))


class GateTests(unittest.TestCase):
    def setUp(self):
        import scripts.generic_landing_1c as tool
        self.tool=tool;self.stage,self.sets=sets()
        from src.visual_search.episodes import load_config
        self.config=load_config()

    def same_flights(self,outcomes=None,episodes=None):
        """Gate B's starts as flown: `outcomes` maps an index to landed / wrong / cone / stop / near / fail."""
        from src.visual_search.maps import MapGeometry,load_map
        scenes={};records=[];steps={};outcomes=outcomes or {}
        for index,episode in enumerate(episodes or self.sets['same_color']):
            scene=scenes.setdefault(episode['map'],load_map(episode['map']));objects=MapGeometry(scene,episode['layout']).objects;there=objects[episode['distractor']]['centre'];here=objects[episode['target']]['centre']
            outcome=outcomes.get(index,'landed');start=episode['start_xy'];record=flight(episode['surface_kind'],'land',outcome in ('landed','near'),target=episode['target'],identifier=episode['id'])
            record['episode']=dict(episode);gap=math.dist(there[:2],here[:2])
            place=lambda xy,distance:{'distance_m':distance,'height_m':6.,'over':False,'front_seen':True,'down_seen':False,'action':[.5,0.,0.],'bearing_deg':0.,'yaw_rad':0.,'position':[xy[0],xy[1],0.],'landed':False}
            path=[place(start,episode['start_distance_m'])]
            if outcome in ('wrong','cone','stop','near'):path.append(place([there[0]+1.,there[1]],gap))
            if outcome in ('landed','near'):path.append(place([here[0]+1.,here[1]],1.))
            if outcome=='wrong':record['surface_landing'].update(landed_on=episode['distractor'],contact_from_above=True);record['summary']['reason']='landed'
            if outcome=='cone':record['surface_landing'].update(collision='Scene'+''.join(part.title() for part in episode['distractor'].split('_')),contact_from_above=False);record['summary'].update(reason='collision',hit='x')
            if outcome in ('stop','fail'):record['surface_landing']['contact_from_above']=False
            if outcome=='fail':path.append(place([here[0]+9.,here[1]],9.))
            records.append(record);steps[episode['id']]=path
        return records,steps

    def failed(self,outcomes):
        records,steps=self.same_flights(outcomes);return [row[0] for row in self.tool.judge_b(self.stage,self.config,records,steps)[2] if not row[3]]

    def test_gate_b_asks_for_the_named_object_and_for_no_end_at_the_other_one(self):
        records,steps=self.same_flights();numbers,rows,checks=self.tool.judge_b(self.stage,self.config,records,steps)
        self.assertTrue(all(row[3] for row in checks));self.assertEqual((numbers['flights'],numbers['landed_on_the_named_object'],numbers['rejected_the_other'],numbers['collisions']),(24,24,24,0))
        self.assertEqual(numbers['by_pair']['blue_pad past blue_cone'],'4 / 4')
        # Two landings missed in the air away from the other object: still passed (22 of 24). Three: not.
        self.assertEqual(self.failed({0:'fail',1:'fail'}),[]);self.assertEqual(self.failed({0:'fail',1:'fail',2:'fail'}),['landings on the named object'])
        # One landing on the other object, one collision with it, one stop beside it: each fails the gate by itself.
        self.assertEqual(self.failed({0:'wrong'}),['landings on a wrong object','flights that ended at the wrong object'])
        self.assertEqual(self.failed({7:'cone'}),['collisions with the other object','collisions','flights that ended at the wrong object'])
        self.assertEqual(self.failed({3:'stop'}),['flights that ended at the wrong object'])
        # Going to the other object first and then landing on the named one is no rejection: three of those fail the 90 %.
        self.assertEqual(self.failed({0:'near',1:'near'}),[]);self.assertEqual(self.failed({0:'near',1:'near',2:'near'}),['the other object, first in view, rejected'])

    def test_gate_bs_second_flights_allow_one_start_that_never_lands_and_no_collision(self):
        from src.visual_search import generic_landing_1c as plans
        ids=set(plans.repeat_ids(self.stage,self.sets['same_color']));again=[e for e in self.sets['same_color'] if e['id'] in ids];index={e['id']:i for i,e in enumerate(self.sets['same_color'])}
        first,first_steps=self.same_flights({index[again[0]['id']]:'fail',index[again[1]['id']]:'fail'})
        def judge(outcomes):
            second,second_steps=self.same_flights(outcomes,episodes=again);table,checks=self.tool.judge_repeat_b(self.stage,self.config,first,second,first_steps,second_steps)
            return table,[row[0] for row in checks if not row[3]]
        table,failed=judge({});self.assertEqual((failed,table['starts'],table['both'],table['one_of_two'],table['neither']),([],12,10,2,0))
        self.assertEqual(judge({0:'fail'})[1],[]);self.assertEqual(judge({0:'fail',1:'fail'})[1],['starts that failed both times'])
        self.assertEqual(judge({5:'cone'})[1],['collisions in the second flights'])

    def gate_a(self,cube,cylinder,identifier):
        records=[]
        for kind,results in (('cube',cube),('cylinder',cylinder)):records+=[flight(kind,'land',good,identifier=f'{kind}-{index}') for index,good in enumerate(results)]
        return records

    def test_the_second_flights_of_gate_a_hold_cubes_to_sixteen_and_cylinders_to_eighteen_of_twenty(self):
        good=[True]*10;judge=lambda a,b,c=good,d=good:[row[0] for row in self.tool.judge_repeat_a(self.stage,self.gate_a(a,c,'1'),self.gate_a(b,d,'2'))[1] if not row[3]]
        self.assertEqual(judge(good,good),[]);self.assertEqual(judge([False,False]+[True]*8,[False]+[True]*8+[False]),[])
        self.assertEqual(judge([False]*3+[True]*7,[True]*10),['cube landings, first flights']);self.assertEqual(judge([False,False]+[True]*8,[True]*3+[False]*5+[True]*2),['cube landings, both flights'])
        # A cylinder at 8 and 9 of ten is 17: enough for a cube, not for a cylinder.
        self.assertEqual(judge(good,good,[False,False]+[True]*8,[True]*9+[False]),['cylinder landings, both flights'])
        self.assertEqual(judge(good,good,[False,False]+[True]*8,[False,False]+[True]*8),['cylinder landings, both flights','cylinder starts that failed both times'])

    def test_the_regression_allows_no_collision_at_all(self):
        def run(success,wrong=0,hit=0,touched=0):
            records=[flight('pad','land',index<success) for index in range(27)]+[flight('pad','approach',index>=touched,landed=index<touched) for index in range(9)]
            for index in range(wrong):records[index]['summary']['wrong_target']=True
            for index in range(hit):records[index]['summary']['hit']='Hedge'
            return records
        failed=lambda model,baseline:[row[0] for row in self.tool.judge_regression(self.stage,model,baseline)[1] if not row[3]]
        self.assertEqual(failed(run(26),run(27)),[]);self.assertEqual(failed(run(25),run(27)),['pad landings']);self.assertEqual(failed(run(24),run(27)),['mission success','pad landings'])
        # Stage 1b's regression: 33 against 34 with two collisions. A collision fails whatever the baseline did.
        self.assertEqual(failed(run(27,hit=1),run(27,hit=1)),['collisions']);self.assertEqual(failed(run(27,hit=2),run(27)),['collisions'])
        self.assertEqual(failed(run(27,wrong=2),run(27)),['flights at a wrong object']);self.assertEqual(failed(run(27,touched=1),run(27)),['touchdowns in an approach'])

    def pilot(self,recovered,cube_same,pad,cylinder,collisions=0,wrong=0,air=0):
        records=[];states=('under_climb','edge','premature','stall')
        for index in range(8):
            record=flight('cube','land',index<recovered);record['episode']['recovery']=states[index//2];records.append(record)
        for index in range(8):
            kind='cube' if index>=6 else 'pad';good=(index-6<cube_same) if kind=='cube' else index<pad-4;record=flight(kind,'land',good);record['episode']['distractor']='blue_cone';records.append(record)
        records+=[flight('pad','land',True) for _ in range(4)]+[flight('cylinder','land',index<cylinder) for index in range(4)]
        same=[record for record in records if record['episode'].get('distractor')]
        for record in same[:collisions]:record['summary']['hit']='SceneBlueCone';record['surface_landing'].update(success=False,collision='SceneBlueCone')
        for record in same[collisions:collisions+wrong]:record['surface_landing'].update(success=False,landed_on='blue_cone',contact_from_above=True)
        failed=[record for record in records if not record['surface_landing']['success'] and not record['surface_landing'].get('landed_on') and not record['summary'].get('hit')]
        for record in failed:record['surface_landing']['contact_from_above']=True
        for record in failed[:air]:record['surface_landing']['contact_from_above']=False
        return records

    def test_the_candidate_is_chosen_by_the_order_of_stage_1c_and_a_flagged_one_goes_behind(self):
        order=self.stage['pilot']['order'];self.assertTrue(order[0].startswith('collisions in the same-colour starts'));self.assertTrue(order[1].startswith('landings on a wrong object'));self.assertTrue(order[2].startswith('forced cube landings recovered'))
        runs={'stage1b':self.pilot(4,1,8,4,collisions=2,wrong=1),'500':self.pilot(6,2,10,4,collisions=1),'1000':self.pilot(6,2,10,4,wrong=1),'1500':self.pilot(6,2,10,4),'2000':self.pilot(7,2,10,2)}
        rows,ranked,checks=self.tool.judge_pilot(self.stage,runs,{'500':.06,'1000':.06,'1500':.06,'2000':.05})
        by={row['candidate']:row for row in rows}
        self.assertEqual((by['stage1b']['same_color_collisions'],by['stage1b']['wrong_target_landings'],by['stage1b']['recovered'],by['1500']['cube_landings'],by['1500']['pad_landings'],by['1500']['cylinder_landings']),(2,1,4,8,10,4))
        # 2000 recovers one more but lost two cylinders against stage 1b: behind. 500 collided: behind. 1500 before 1000 (no wrong object).
        self.assertEqual(by['2000']['put_behind'],['cylinder landings below stage 1b']);self.assertEqual(by['500']['put_behind'],['a collision in a same-colour start'])
        self.assertEqual(ranked[:2],['1500','1000']);self.assertEqual(set(ranked[2:]),{'500','2000'});self.assertNotIn('stage1b',ranked);self.assertTrue(all(row[3] for row in checks))
        # The pilot's lines: no collision in a same-colour start, whatever else is good.
        rows,ranked,checks=self.tool.judge_pilot(self.stage,{'stage1b':self.pilot(4,1,8,4),'1000':self.pilot(8,2,10,4,collisions=1)},{})
        self.assertEqual([row[0] for row in checks if not row[3]],['collisions in the same-colour starts'])
        rows,ranked,checks=self.tool.judge_pilot(self.stage,{'stage1b':self.pilot(4,1,8,4),'1000':self.pilot(3,2,10,4,air=3)},{});self.assertIn('forced landings recovered below stage 1b',rows[1]['put_behind'])
        self.assertEqual([row[0] for row in checks if not row[3]],['landings on cubes'])


class RecordingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):asyncio.get_running_loop().slow_callback_duration=60.

    async def test_forced_landings_and_a_same_colour_pair_are_recorded_and_the_records_keep_the_rules(self):
        import projectairsim
        import scripts.generic_landing_1c as tool
        import scripts.surface_flights as worker
        import scripts.visual_search as canonical
        from src.landing import stop_safety
        from src.visual_search.episodes import load_config
        from src.visual_search.generalization import SearchTeacher,Pushes
        from src.visual_search.maps import load_map
        from tests.fake_airsim import FakeSim,FakeClient,world_class,drone_class
        stage,everything=sets();plan=everything['train_plan']['train'];config=load_config();_,oft,_,_=setting()
        pick=lambda **fields:next(e for e in plan if all(e.get(k)==v for k,v in fields.items()))
        twin=next(e for e in plan if e.get('twin_of'));first=next(e for e in plan if e['id']==twin['twin_of'])
        chosen=[pick(recovery='premature',surface_kind='cube'),pick(recovery='stall',surface_kind='cube'),pick(recovery='edge',surface_kind='cube'),first,twin]
        async def no_wait(_):return None
        clock=types.SimpleNamespace(sleep=no_wait,wait_for=asyncio.wait_for)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'plan.json';file.write_text(json.dumps({'train':chosen}));data=root/'data'
            for episode in chosen:
                sim=FakeSim(load_map(episode['map']),episode['layout'])
                arguments=argparse.Namespace(host='fake',ports=[1,2],plan=str(file),set='train',output=str(root/'run'),policy='teacher',checkpoint=None,model_name=None,
                                             finalizer='off',record=str(data),only=[episode['id']],skip=0,limit=0,again=False,real_time_minimum=0.,teacher='recovery')
                with patch.object(projectairsim,'ProjectAirSimClient',lambda **_:FakeClient(sim)),patch.object(canonical,'World',world_class(sim)), \
                     patch.object(canonical,'Drone',drone_class(sim)),patch.object(canonical,'matched_camera_images',sim.camera), \
                     patch.object(canonical,'asyncio',clock),patch('builtins.print'):
                    with self.assertRaises(SystemExit) as ended:await worker.fly(arguments)
                self.assertEqual(ended.exception.code,0,episode['id']);self.assertNotIn('land_async',sim.calls)
                # The loop is handed its own teacher and its own pushes back.
                self.assertIs(canonical.SearchTeacher,SearchTeacher);self.assertIs(canonical.Pushes,Pushes)
            results={item['id']:item for item in json.loads((root/'run/results.json').read_text())['episodes']};self.assertFalse((data/'rejected').exists())
            for episode in chosen[:3]:
                item=results[episode['id']];self.assertTrue(item['recorded']);self.assertTrue(item['forced']['all_came_to_pass_and_seen']);self.assertEqual(item['forced']['came_to_pass'],len(episode['pushes']))
                record=json.loads((data/'episodes'/f'{episode["id"]}.json').read_text());pushed=[step for step in record['steps'] if step.get('perturbed')]
                self.assertGreater(len(pushed),0);self.assertTrue(all(step['teacher_state']=='push' for step in pushed));self.assertEqual(record['summary']['landed_on'],episode['target'])
            self.assertNotIn('forced',results[first['id']]);self.assertEqual((results[first['id']]['summary']['landed_on'],results[twin['id']]['summary']['landed_on']),(first['target'],twin['target']))
            labels=stop_safety.audit(stop_safety.recorded(data),oft,{'safe_ratio':3.,'near_stop_ratio':2.},config['stop_ticks']);sentences=tool.inputs_audit(data,stage,config)
        self.assertTrue(labels['passed']);self.assertEqual((labels['total']['episodes'],labels['by']['task']['LAND']['below_safe'],labels['by']['task']['LAND']['nonterminal_stop']),(5,0,0))
        self.assertGreater(labels['total']['forced'],0);self.assertTrue(sentences['passed']);self.assertEqual((sentences['twins'],sentences['twins_with_their_landing'],sentences['sentences']),(1,1,len({e['instruction'] for e in chosen})))


if __name__=='__main__':unittest.main()
