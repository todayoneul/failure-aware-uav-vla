"""Phase B, stage 1d (final descent continuation): the teacher's rule of place over a small top, its hysteresis, the three
forced moves of the last metres, the plans, what is replayed and what is not, what the data gives a student, and the
pilot, the qualification and the regression line. No simulator, no model."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from tests.test_generic_landing import flight,setting,view

ROOT=Path(__file__).resolve().parents[1]


def stage1d():
    from src.visual_search import generic_landing_1d as plans
    return plans.load()


def sets():
    from src.visual_search import generic_landing_1d as plans
    stage=plans.load();out={name:json.loads(path.read_text(encoding='utf-8'))['episodes'] for name,path in plans.files_of(stage).items()}
    out['train_plan']=json.loads(plans.train_plan_of(stage).read_text(encoding='utf-8'));return stage,out


class TeacherStateTests(unittest.TestCase):
    def teacher(self,target,kind='final_descent'):
        from src.landing.teacher import TEACHERS
        config,oft,scene,surfaces=setting();self.oft=oft;self.surface=surfaces[target]
        return TEACHERS[kind](config,oft,surface=surfaces[target])

    def at(self,teacher,distance,above,top=8.,bearing=0.,on=None,landed=False):
        seen=view(distance_m=distance,height_m=top+above,bearing_deg=bearing,visible=False,below=True,over=distance<4. if on is None else on,surface_m=top,landed=landed)
        action,state=teacher.act(seen)
        from src.landing.teacher import descent_phase
        return action,state,descent_phase(state,seen,teacher)

    def test_the_two_radii_are_the_canonical_align_radius_and_the_usable_top_less_the_drift(self):
        from src.landing.teacher import hard_recovery_radius
        rule=stage1d()['teacher'];teacher=self.teacher('blue_cube')
        self.assertEqual(rule['drift_margin_m'],.3);self.assertAlmostEqual(hard_recovery_radius(self.surface,rule['drift_margin_m']),3.3)
        self.assertEqual((teacher.align,round(teacher.hard,3)),(1.75,3.3));self.assertEqual(rule['thresholds']['A_descent_safe_m'],1.75);self.assertEqual(rule['thresholds']['B_hard_recovery_m']['cube'],3.3)
        for target in ('red_cube','green_cylinder','blue_cylinder'):self.assertAlmostEqual(self.teacher(target).landing['commit_radius_m'],3.3)
        # A pad keeps the canonical radius; stage 1c's teacher keeps its own.
        self.assertEqual(self.teacher('blue_pad').landing['commit_radius_m'],3.5);self.assertAlmostEqual(self.teacher('blue_cube','recovery').landing['commit_radius_m'],2.675)
        self.assertLess(teacher.align,teacher.hard);self.assertLess(teacher.hard,self.surface.usable_reach_m)

    def test_low_over_the_top_and_nearly_still_the_label_is_down_wherever_the_middle_is(self):
        """D1 CENTERED_LOW_HOVER and D2 CENTERED_DESCENT_DECAY: the teacher has no memory, so a vehicle that hovers and one
        whose descent died away get the same label: down, at the rate of every earlier descent."""
        from src.landing.stop_safety import stop_ratio
        teacher=self.teacher('blue_cube')
        for distance in (.2,1.,1.7,1.8,2.2,2.76,3.0,3.29):
            for above in (.2,.3,.6,1.07,1.4,2.,2.5):
                for bearing in (0.,60.,118.,-170.):
                    action,state,phase=self.at(teacher,distance,above,bearing=bearing)
                    self.assertEqual((action,state),([0.,.25,0.],'descend'),(distance,above,bearing));self.assertEqual(phase,'DESCENT_COMMIT' if distance<=1.75 else 'DESCENT_CONTINUE')
                    self.assertGreaterEqual(stop_ratio(action,self.oft),3.)
        # The same call again and again gives the same label: nothing dies away.
        self.assertEqual({tuple(self.at(teacher,2.76,1.07,bearing=118.)[0]) for _ in range(8)},{(0.,.25,0.)})
        # Where stage 1c's later checkpoints stopped (2.6-2.8 m out, 0.2-1.4 m over the top) stage 1c's teacher said up.
        old=self.teacher('blue_cube','recovery');self.assertEqual(self.at(old,2.76,1.07,bearing=118.)[1],'climb');self.assertLess(self.at(old,2.76,1.07,bearing=118.)[0][1],0.)

    def test_low_beyond_the_hard_radius_or_beside_the_top_there_is_no_descent(self):
        """D3 EDGE_LOW and D4 EDGE_AFTER_PARTIAL_DESCENT: up, in place only beside the top and lower than the clearance."""
        teacher=self.teacher('blue_cube')
        for distance,on in ((3.4,True),(3.9,True),(4.6,False),(5.4,False)):
            for above in (.3,.8,1.5,2.3):
                for bearing in (0.,75.,170.):
                    action,state,phase=self.at(teacher,distance,above,bearing=bearing,on=on)
                    self.assertEqual((state,phase),('climb','RECOVER_CLEARANCE'),(distance,above,bearing));self.assertLess(action[1],0.);self.assertLessEqual(action[1],-.25)
                    # Turning to the middle whenever it is not ahead; moving in only when it is ahead and the way is clear.
                    self.assertEqual(action[2]!=0.,bearing!=0.)
                    clear=on or above>=1.
                    self.assertEqual(action[0]>0.,bearing==0. and clear,(distance,above,bearing,on))
        # Beside the top and lower than the clearance over it: up where it is (there is a side to fly into).
        self.assertEqual(self.at(teacher,4.6,.5,on=False)[0][:2],[0.,-.5])
        # Over the top itself there is no side: up and in, also right over the surface. Stage 1c held still there.
        self.assertAlmostEqual(self.at(teacher,3.6,.5,on=True)[0][0],.35);self.assertEqual(self.at(self.teacher('blue_cube','recovery'),3.6,.5,on=True)[0][0],0.)
        # The place of stage 1c's other stop: 5.4 m out, beside the top, 2 m over its level, the middle abeam.
        action,state,phase=self.at(teacher,5.44,2.1,bearing=75.,on=False);self.assertEqual(state,'climb');self.assertEqual(action[0],0.);self.assertLess(action[1],0.);self.assertGreater(action[2],0.)

    def test_after_the_climb_the_vehicle_goes_to_the_middle_and_only_there_down_again(self):
        teacher=self.teacher('blue_cube')
        for distance in (4.5,3.4,2.5,1.9):
            action,state,phase=self.at(teacher,distance,3.5);self.assertEqual((state,phase),('final','RECENTER'),distance);self.assertAlmostEqual(action[0],.35);self.assertEqual(action[1],0.)
        self.assertEqual(self.at(teacher,1.7,3.5)[1:],('descend','DESCENT_COMMIT'));self.assertEqual(self.at(teacher,1.7,3.5)[0],[0.,.5,0.])
        # Turning on the spot at the height it flies in at is a label far from a stop, too.
        action,state,phase=self.at(teacher,3.8,3.5,bearing=160.);self.assertEqual((state,phase),('align','RECENTER'));self.assertEqual(action[0],0.);self.assertAlmostEqual(abs(action[2]),.21)

    def test_after_the_touchdown_the_label_is_the_stop_and_before_it_never(self):
        from src.landing.stop_safety import stop_ratio
        teacher=self.teacher('blue_cube');self.assertEqual(self.at(teacher,2.5,0.,landed=True),([0.,0.,0.],'landed','TOUCHDOWN'))
        for distance in (0.,1.,2.,3.,3.5,4.2,5.5,8.):
            for above in (.2,.6,1.,1.6,2.4,3.,3.5):
                for bearing in (0.,40.,95.,180.):
                    action=self.at(teacher,distance,above,bearing=bearing,on=distance<4.)[0];self.assertGreaterEqual(stop_ratio(action,self.oft),3.,(distance,above,bearing,action))

    def test_a_pad_and_everything_outside_the_last_metres_are_stage_1cs(self):
        new=self.teacher('blue_pad');old=self.teacher('blue_pad','recovery')
        for distance in (1.,3.,3.6,6.,15.,40.):
            for above in (.5,2.,3.5,6.):
                for bearing in (0.,30.,120.):
                    seen=view(distance_m=distance,height_m=2.5+above,bearing_deg=bearing,visible=True,over=distance<7.,surface_m=2.5);self.assertEqual(new.act(seen),old.act(seen),(distance,above,bearing))
        new=self.teacher('blue_cube');old=self.teacher('blue_cube','recovery')
        for distance in (9.,15.,30.):
            for height in (4.,8.5,11.5,13.):
                seen=view(distance_m=distance,height_m=height,bearing_deg=10.,visible=True,over=False,surface_m=8.);self.assertEqual(new.act(seen),old.act(seen),(distance,height))
        # Without the named object in view the teacher knows nothing of where it is.
        blind=view(distance_m=3.,height_m=9.,visible=False,below=False,over=False,surface_m=8.);self.assertEqual(new.act(blind)[1],'search')


class HysteresisTests(unittest.TestCase):
    def setUp(self):
        from src.landing.teacher import TEACHERS
        config,oft,scene,surfaces=setting();self.teacher=TEACHERS['final_descent'](config,oft,surface=surfaces['blue_cube'])

    def state(self,distance,above,bearing=0.):
        return self.teacher.act(view(distance_m=distance,height_m=8.+above,bearing_deg=bearing,visible=False,below=True,over=distance<4.,surface_m=8.))[1]

    def test_between_the_radii_the_label_follows_the_height_and_never_flips_with_a_small_drift(self):
        # Flying in (higher than the commit height): not down before A, however little is missing.
        for distance in (1.76,1.9,2.5,3.2):self.assertNotEqual(self.state(distance,3.5),'descend',distance)
        self.assertEqual(self.state(1.75,3.5),'descend')
        # Low: down all the way out to B. A little beyond A is no reason to climb.
        steps=[1.7+.05*index for index in range(36)];states=[self.state(distance,1.2) for distance in steps]
        self.assertEqual(set(states[:steps.index(next(d for d in steps if d>3.3))]),{'descend'});self.assertNotIn('climb',[self.state(distance,1.2) for distance in (1.76,1.8,2.0,2.675,2.7,3.0,3.3)])
        # A vehicle that drifts back and forth inside the band keeps one label.
        self.assertEqual({self.state(2.4+.3*math.sin(index),1.+.2*math.cos(index)) for index in range(40)},{'descend'})
        # Beyond B: recovery, and it stays recovery until the vehicle is back inside B.
        for distance in (3.31,3.4,3.9):self.assertEqual(self.state(distance,1.2),'climb',distance)
        self.assertEqual([self.state(distance,1.2) for distance in (3.6,3.45,3.35,3.25)],['climb','climb','climb','descend'])

    def test_the_band_is_wider_than_the_drift_of_a_descent(self):
        rule=stage1d()['teacher'];self.assertGreaterEqual(self.teacher.hard-self.teacher.align,5*rule['drift_margin_m'])


class ForcedMoveTests(unittest.TestCase):
    def pushes(self,*items):
        from src.landing.teacher import RecoveryPushes
        return RecoveryPushes({'pushes':list(items)})

    def at(self,distance,above,bearing=0.,top=8.):return {'bearing_deg':bearing,'distance_m':distance,'height_m':top+above,'surface_m':top}

    def test_each_forced_move_of_the_last_metres_is_due_at_a_place_and_ends_at_its_state(self):
        hover={'when':'low_hover','within_m':4.,'hold_m':1.,'hold_ticks':3,'band_m':.15,'gain':.2,'down_m':.4,'ticks':60};push=self.pushes(hover)
        self.assertTrue(push.forced);self.assertIsNone(push.step('final',self.at(6.,3.5),.21));self.assertFalse(push.pending())
        # Down toward the height, more slowly the nearer; held nearly still inside the band; then let go.
        self.assertEqual(push.step('final',self.at(3.,3.5),.21),[0.,.4,0.]);self.assertAlmostEqual(push.step('descend',self.at(3.,1.6),.21)[1],.12);self.assertTrue(push.pending())
        self.assertEqual([push.step('descend',self.at(3.,1.1),.21) for _ in range(3)],[[0.,0.,0.]]*3);self.assertIsNone(push.step('descend',self.at(3.,1.05),.21));self.assertFalse(push.pending())
        # A vehicle that sank below the height is brought up to it, not left there.
        push=self.pushes(hover);self.assertLess(push.step('descend',self.at(2.,.5),.21)[1],0.)
        self.assertNotIn('held',hover)     # the episode's own item is not written to
        decay={'when':'descent_decay','within_m':4.,'from_m':2.,'down_m':.3,'start_m':.2,'ratio':.5,'decay_ticks':4,'ticks':40};push=self.pushes(decay)
        self.assertEqual(push.step('final',self.at(2.,3.5),.21),[0.,.3,0.])
        values=[push.step('descend',self.at(2.,2.-.2*index),.21)[1] for index in range(4)];self.assertEqual([round(value,4) for value in values],[.2,.1,.05,.025])
        self.assertIsNone(push.step('descend',self.at(2.,1.2),.21))
        # It ends before the vehicle stands on the top.
        push=self.pushes(decay);push.step('final',self.at(2.,1.9),.21);self.assertIsNone(push.step('descend',self.at(2.,.05),.21))
        drift={'when':'edge_drift','within_m':3.,'from_m':1.5,'down_m':.3,'out_m':3.8,'forward_m':.3,'ticks':50};push=self.pushes(drift)
        self.assertEqual(push.step('descend',self.at(.8,3.4,bearing=80.),.21),[0.,.3,0.])
        action=push.step('descend',self.at(.8,1.4,bearing=95.),.21);self.assertEqual(action[1:],[0.,0.]);self.assertAlmostEqual(action[0],.3)
        self.assertAlmostEqual(push.step('descend',self.at(3.2,1.4,bearing=170.),.21)[0],.3*.6/1.5);self.assertIsNone(push.step('climb',self.at(3.85,1.4,bearing=175.),.21))

    def test_stage_1cs_moves_and_the_canonical_ones_are_untouched(self):
        from src.landing.teacher import FORCED,FINAL
        from src.visual_search.generalization import Pushes
        self.assertEqual(FORCED,('under_climb','edge','premature','stall'));self.assertEqual(FINAL,('low_hover','descent_decay','edge_drift'))
        push=self.pushes({'when':'stall','below_m':2.,'up_m':.4,'forward_m':.1,'ticks':2})
        self.assertIsNone(push.step('final',self.at(1.5,1.8),.21));self.assertEqual(push.step('descend',self.at(1.5,1.9),.21),[.1,-.4,0.])
        item={'when':'landed','ticks':2,'up_m':.4};mine=self.pushes(item);theirs=Pushes({'pushes':[item]});self.assertFalse(mine.forced)
        for state,place in (('descend',self.at(1.,1.)),('landed',self.at(1.,0.)),('landed',self.at(1.,0.))):self.assertEqual(mine.step(state,place,.21),theirs.step(state,place,.21))

    def test_the_dry_run_flies_every_state_and_the_teacher_lands_with_labels_clear_of_a_stop(self):
        from src.landing import stop_safety
        from src.landing.teacher import rollout
        from src.visual_search import generic_landing_1d as plans
        from src.visual_search.maps import object_position
        import random
        config,oft,scene,surfaces=setting();rule={'safe_ratio':3.,'near_stop_ratio':2.};stage=plans.load()
        for target in ('blue_cube','green_cylinder'):
            centre=object_position(scene,'a',target)
            for state in plans.STATES:
                # As the planner does: a draw of the pushes is taken only if a camera shows the object where each lets go.
                for attempt in range(12):
                    pushes=plans.pushes_of(stage,state,random.Random(f'{target}{state}{attempt}'),surfaces[target],None);self.assertEqual([item['when'] for item in pushes][0],'edge');self.assertEqual(len(pushes),2)
                    episode={'start_xy':[centre[0]-14.,centre[1]-3.],'start_yaw_deg':10.,'start_height_m':3.5,'target':target,'task':'land','strategy':'right','max_ticks':320,'pushes':pushes}
                    trace=[];result=rollout(config,oft,scene,'a',episode,surfaces[target],kind='final_descent',trace=trace)
                    if all(after['seen'] for before,after in zip(trace,trace[1:]) if before['perturbed'] and not after['perturbed']):break
                self.assertEqual((result['reason'],result['landed'],result['forced_left']),('teacher_stop',True,0),(target,state));self.assertEqual(len(result['forced']),2)
                self.assertTrue(surfaces[target].inside([result['final_distance_m'],0.]))
                one=stop_safety.audit_episode(trace,'land',oft,rule,config['stop_ticks'])
                self.assertEqual((one['nonterminal_stop'],one['below_safe'],one['stop_windows'],one['near_stop_windows']),(0,0,0,0),(target,state));self.assertEqual(one['forced'],result['forced_ticks'])
                let_go=[after for before,after in zip(trace,trace[1:]) if before['perturbed'] and not after['perturbed']]
                self.assertEqual(len(let_go),2);self.assertTrue(all(row['seen'] for row in let_go));last=let_go[-1];above=last['height_m']-surfaces[target].top_m
                self.assertGreaterEqual(above,stage['final_descent']['let_go_above_top_min_m']);self.assertLessEqual(above,2.6)
                # Inside the part where a descent goes on the label is down at once and stays down; outside it, up first.
                tail=[row['teacher_state'] for row in trace[trace.index(last):] if not row['landed']]
                if state in ('low_hover','descent_decay'):self.assertEqual(set(tail),{'descend'},(target,state));self.assertLessEqual(last['distance_m'],3.3)
                else:
                    self.assertEqual(tail[0],'climb',(target,state));self.assertGreater(last['distance_m'],3.3);self.assertEqual(tail[-1],'descend')
                    # Up once, to the middle, down once: the vertical label changes its sign a single time.
                    signs=[1 if row['teacher_action'][1]>0 else -1 for row in trace[trace.index(last):] if not row['landed'] and row['teacher_action'][1]!=0]
                    self.assertEqual(sum(a!=b for a,b in zip(signs,signs[1:])),1)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage,cls.sets=sets();cls.train=cls.sets['train_plan']['train'];cls.val=cls.sets['train_plan']['val']

    def count(self,episodes,**fields):return sum(all(episode.get(key)==value for key,value in fields.items()) for episode in episodes)

    def test_the_recorded_episodes_are_the_four_states_mostly_on_cubes_and_no_same_colour_start(self):
        self.assertEqual(len(self.train),80);self.assertEqual(len(self.val),8)
        self.assertEqual([self.count(self.train,recovery=state) for state in ('low_hover','descent_decay','edge_low','edge_drift')],[24,20,20,16])
        self.assertEqual([self.count(self.train,surface_kind=kind) for kind in ('cube','cylinder','pad')],[68,6,6])
        self.assertTrue(all(episode['task']=='land' and not episode.get('distractor') and not episode.get('twin_of') for episode in self.train+self.val))
        for episode in self.train+self.val:
            first,second=episode['pushes'];self.assertEqual(first['when'],'edge');self.assertEqual(second['when'],{'low_hover':'low_hover','edge_low':'low_hover','descent_decay':'descent_decay','edge_drift':'edge_drift'}[episode['recovery']])
            rule=self.stage['final_descent']['forced'][episode['recovery']];self.assertTrue(rule['edge']['miss_m'][0]<=first['miss_m']<=rule['edge']['miss_m'][1])
            self.assertEqual(len(episode['plan']['let_go']),2);self.assertGreaterEqual(episode['plan']['let_go'][-1]['above_top_m'],.15);self.assertIn(episode['layout'],self.stage['layouts']['train'])
        # The states inside the part where a descent goes on are set inside it, the edge states outside it.
        inside=self.stage['final_descent']['forced'];self.assertLess(inside['low_hover']['edge']['miss_m'][1],3.3);self.assertLess(inside['descent_decay']['edge']['miss_m'][1],3.3)
        self.assertGreater(inside['edge_low']['edge']['miss_m'][0],3.3);self.assertGreater(inside['edge_drift']['out_m'][0],3.5)

    def test_the_check_sets_are_what_the_configuration_says(self):
        pilot,qualification,teacher=self.sets['pilot'],self.sets['qualification'],self.sets['teacher_pilot']
        forced=[e for e in pilot if e.get('recovery')];same=[e for e in pilot if e.get('distractor')];plain=[e for e in pilot if not e.get('recovery') and not e.get('distractor')]
        self.assertEqual((len(pilot),len(forced),len(same),len(plain)),(24,12,6,6));self.assertTrue(all(e['surface_kind']=='cube' for e in forced))
        self.assertEqual([self.count(forced,recovery=state) for state in ('low_hover','descent_decay','edge_low','edge_drift')],[3,3,3,3])
        self.assertEqual(sorted((e['target'],e['distractor']) for e in same),sorted([('blue_pad','blue_cube')]*2+[('blue_pad','blue_cone')]*2+[('blue_cube','blue_pad'),('red_pad','red_cube')]))
        self.assertEqual([self.count(plain,surface_kind=kind) for kind in ('pad','cylinder')],[3,3])
        forced=[e for e in qualification if e.get('recovery')];same=[e for e in qualification if e.get('distractor')];plain=[e for e in qualification if not e.get('recovery') and not e.get('distractor')]
        self.assertEqual((len(qualification),len(forced),len(same)),(20,4,4));self.assertEqual(sorted(e['recovery'] for e in forced),sorted(('low_hover','descent_decay','edge_low','edge_drift')))
        self.assertEqual([self.count(plain,surface_kind=kind) for kind in ('cube','pad','cylinder')],[6,3,3])
        self.assertEqual(len(teacher),20);self.assertEqual([self.count(teacher,recovery=state) for state in ('low_hover','descent_decay','edge_low','edge_drift')],[5,5,5,5])
        self.assertTrue(all(e['surface_kind']=='cube' for e in teacher))
        for episode in pilot+qualification+teacher:self.assertIn(episode['layout'],self.stage['layouts']['evaluation']);self.assertEqual(episode['task'],'land')

    def test_the_plans_are_reproducible_and_keep_away_from_stage_1cs_sealed_sets(self):
        import scripts.generic_landing_1d as tool
        from src.visual_search import generic_landing_1c as before, generic_landing_1d as plans
        from src.visual_search.episodes import load_config
        from src.aerovla_oft.spec import load_config as load_oft
        self.assertEqual(tool.check(load_config(),load_oft(),self.stage),[])
        gates=plans.gates_of(self.stage);self.assertEqual(self.stage['gates']['config'],'configs/generic_landing_stage1c.json');sealed=[];spent=[]
        for name,path in before.files_of(gates).items():(spent if name=='teacher_pilot' else sealed).extend(json.loads(path.read_text(encoding='utf-8'))['episodes'])
        self.assertEqual(len(sealed),36+36+24+11+24);self.assertIn('configs/generic_landing_stage1c/teacher_pilot.json',self.stage['checks']['may_be_near'])
        self.assertFalse(any(name in path for path in self.stage['checks']['may_be_near'] for name in ('stage1c/validation','stage1c/same_color','stage1c/regression','stage1c/pilot')))
        mine=self.train+self.val+self.sets['pilot']+self.sets['qualification']+self.sets['teacher_pilot']
        near=lambda episode,others:any(other['map']==episode['map'] and math.dist(other['start_xy'],episode['start_xy'])<6. for other in others)
        # Nothing of stage 1d starts near a start of the sealed gates, the regression set or stage 1c's pilot; a recorded start
        # keeps away from stage 1c's teacher pilot too (a check start need not: only the teacher flew those).
        for episode in mine:self.assertFalse(near(episode,sealed),episode['id'])
        for episode in self.train+self.val:self.assertFalse(near(episode,spent),episode['id'])
        seeds=[episode['seed'] for episode in mine];self.assertEqual(len(seeds),len(set(seeds)));self.assertTrue(all(80000<=seed<=83999 for seed in seeds))
        self.assertFalse({episode['seed'] for episode in sealed}&set(seeds))

    def test_the_sealed_sets_of_stage_1c_are_the_files_stage_1c_wrote(self):
        """Gate A, gate B, the twins and the regression set are flown from stage 1c's files; their content is pinned here."""
        pinned=self.stage['gates']['sha256']
        for name,digest in pinned.items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)
        self.assertEqual(sorted(Path(name).name for name in pinned),['regression.json','same_color.json','same_color_twins.json','validation.json'])
        from src.visual_search import generic_landing_1c as before
        old=before.load();self.assertEqual(old['validation']['gates']['cube_land_min'],8);self.assertEqual(old['same_color']['gates']['wrong_target_landings_max'],0);self.assertEqual(old['regression']['collisions_max'],0)

    def test_the_schedule_is_narrower_than_stage_1cs_and_replays_its_data(self):
        tune=self.stage['fine_tune'];self.assertEqual((tune['updates'],tune['learning_rate'],tune['grounding_learning_rate'],tune['candidates']),(1500,1.25e-5,1.25e-5,[500,1000,1500]))
        self.assertEqual(tune['share'],{'generic_landing_1d_added':.25,'generic_landing_1c_added':.30,'generic_landing_1b_added':.25});self.assertAlmostEqual(1-sum(tune['share'].values()),.20)
        self.assertEqual(self.stage['dataset']['not_used'],['generic_landing_added']);self.assertEqual(self.stage['start_checkpoint'],'outputs/aerovla_oft/checkpoints/generic_landing_stage1c_run/update-2000')
        self.assertEqual(self.stage['stop_safety']['safe_ratio'],3.);self.assertEqual(self.stage['teacher']['descent_action']['last_2_5_m'],[0.,.25,0.])
        rule=self.stage['pilot']['pass'];self.assertEqual((rule['same_color_collisions_max'],rule['wrong_target_landings_max'],rule['critical_min'],rule['critical_air_self_stops_max'],rule['pad_landings_min'],rule['cylinder_landings_min']),(0,0,10,1,3,2))
        rule=self.stage['qualification']['pass'];self.assertEqual((rule['cube_landings_min'],rule['same_color_landings_min'],rule['pad_landings_min'],rule['cylinder_landings_min'],rule['air_self_stops_max'],rule['wrong_target_max'],rule['collisions_max']),(8,4,3,2,1,0,0))


class ReplayTests(unittest.TestCase):
    def dataset(self,root,name,episodes):
        folder=Path(root)/name/'episodes';folder.mkdir(parents=True)
        for summary,steps in episodes:(folder/f'{summary["id"]}.json').write_text(json.dumps({'summary':summary,'steps':steps}),encoding='utf-8')

    def step(self,index,distance,above,state,down,over=True,**more):
        return {'step':index,'distance_m':distance,'height_m':8.+above,'teacher_state':state,'teacher_action':[0.,down,0.],'over':over,'img_name':f'{index}.jpg',**more}

    def test_a_label_the_stage_1d_teacher_would_not_give_is_left_out_and_nothing_else(self):
        import scripts.generic_landing_1d as tool
        from src.visual_search.episodes import load_config
        from src.visual_search.maps import load_landing
        stage=stage1d();base={'map':'field','layout':'a','task':'land'}
        with tempfile.TemporaryDirectory() as root:
            steps=[self.step(0,2.9,1.5,'climb',-.5),self.step(1,2.6,1.5,'descend',.25),self.step(2,3.4,1.5,'climb',-.5),self.step(3,2.9,3.4,'climb',-.25),self.step(4,1.7,1.,'descend',.25),
                   self.step(5,2.9,1.5,'climb',-.5,perturbed=True),self.step(6,3.0,2.0,'climb',-.5),self.step(7,3.0,2.0,'climb',-.5,over=False)]
            self.dataset(root,'c',[({'id':'cube',**base,'target':'blue_cube'},steps),({'id':'pad',**base,'target':'blue_pad'},[self.step(0,2.9,1.5,'climb',-.5)]),
                                   ({'id':'approach',**base,'task':'approach','target':'blue_cube'},[self.step(0,2.9,1.5,'climb',-.5)])])
            old=tool.DATA;tool.DATA=Path(root)
            try:found=tool.left_out(stage,load_config(),load_landing(),['c'])
            finally:tool.DATA=old
        # Over a cube, low, between A and B, and a climb: steps 0 and 6. Not: a descent, beyond B, higher than the commit height, inside A, a forced move, beside the top, a pad, an approach.
        self.assertEqual(found['c'],{'decisions':2,'episodes':1,'steps':{'cube':[0,6]}})

    def test_the_builder_takes_a_left_out_decision_out_of_every_window_it_is_part_of(self):
        from scripts.build_visual_search_dataset import episode_samples
        steps=[{'step':index,'img_name':f'{index}.jpg','teacher_action':[0.,.25,0.],'teacher_state':'descend','action':[0.,.25,0.],'proprio':[0.]*4,'front_seen':False,'front_in_fov':False,
                'down_seen':True,'bearing_deg':90.,'distance_m':2.8,'height_m':9.} for index in range(10)]
        record={'summary':{'id':'e','task':'land','instruction':'x','target':'blue_cube','split':'train','case':'mid_edge'},'traj_rel_dir':'x','steps':steps}
        whole=episode_samples(record,4);steps[5]['left_out']=True;less=episode_samples(record,4)
        self.assertEqual(len(whole)-len(less),4);self.assertTrue(all(sample['meta']['step']<2 or sample['meta']['step']>5 for sample in less))
        # What a sample holds is what it held before: the picture, the sentence, the label; nothing of the forced state.
        self.assertEqual(set(less[0]),{'traj_rel_dir','img_name','instruction','label','is_last_step','is_penultimate','chunk','proprio','meta'});self.assertEqual(less[0]['instruction'],'<image>\nx')
        self.assertFalse({'recovery','pushes','state','left_out'}&set(less[0]['meta']))

    def test_the_merged_data_has_none_of_stage_1_and_all_of_stage_1cs_same_colour_episodes(self):
        import scripts.generic_landing_1d as tool
        stage=stage1d();data=stage['dataset']
        with tempfile.TemporaryDirectory() as root:
            same=[({'id':f's{index}','distractor':'blue_cone',**({'twin_of':'s0'} if index==1 else {})},[{'step':0}]) for index in range(3)]
            self.dataset(root,data['corrective'],same+[({'id':'r0','recovery':'edge'},[{'step':0}])])
            def merged(sources):
                folder=Path(root)/'merged';folder.mkdir(exist_ok=True);(folder/'summary.json').write_text(json.dumps({'episodes':sum(sources.values()),'samples':9,'sources':sources}),encoding='utf-8');return folder
            old=tool.DATA;tool.DATA=Path(root)
            try:
                good=tool.replay_audit(stage,merged({data['added']:80,data['corrective']:4,data['stop_safe']:231,'generalization_v1':500}))
                self.assertTrue(good['passed']);self.assertEqual((good['stage_1_generic_landing_episodes'],good['stage_1c_same_colour_episodes_recorded'],good['stage_1c_twins_recorded']),(0,3,1))
                self.assertFalse(tool.replay_audit(stage,merged({data['added']:80,data['corrective']:4,data['stop_safe']:231,'generic_landing_added':233}))['passed'])
                self.assertFalse(tool.replay_audit(stage,merged({data['added']:80,data['corrective']:3,data['stop_safe']:231}))['passed'])     # an episode of stage 1c is missing
                self.assertFalse(tool.replay_audit(stage,merged({data['added']:80,data['corrective']:4}))['passed'])                             # stage 1b is missing
                self.assertFalse(tool.replay_audit(stage,merged({data['corrective']:4,data['stop_safe']:231}))['passed'])                        # stage 1d is missing
            finally:tool.DATA=old


class InputTests(unittest.TestCase):
    def test_a_planned_episode_gives_the_student_a_catalogue_sentence_and_nothing_of_the_state(self):
        from src.visual_search.episodes import load_config
        from src.visual_search.generalization import instruction_for_id
        from src.visual_search.maps import load_map
        stage,every=sets();config=load_config();scenes={name:load_map(name) for name in stage['scenes']};episodes=every['train_plan']['train']+every['train_plan']['val']+every['pilot']+every['qualification']
        words=('hover','decay','edge','drift','centre','center','offset','height','metre','meter','distance','bearing','D1','D2','D3','D4')
        for episode in episodes:
            text=episode['instruction'];self.assertEqual(text,instruction_for_id(config,scenes[episode['map']],episode['target'],'land'))
            self.assertFalse(any(letter.isdigit() for letter in text));self.assertFalse(any(word.lower() in text.lower() for word in words),text);self.assertNotIn(episode['target'],text)
        self.assertLessEqual(len({episode['instruction'] for episode in episodes}),6)

    def test_the_model_is_given_the_mosaic_and_the_sentence_and_nothing_else(self):
        source=(ROOT/'scripts/visual_search.py').read_text(encoding='utf-8')
        self.assertIn("inputs=policy_inputs(observation['frames']['front'],observation['frames']['down'],episode['instruction'],",source)
        self.assertIn("inference=policy.infer(inputs['front'],inputs['down'],inputs['instruction'],inputs['proprio'])",source)
        config=json.loads((ROOT/'configs/aerovla_oft.json').read_text(encoding='utf-8'));self.assertFalse(config['proprio']['enabled'])
        self.assertFalse(any(word in field for field in config['proprio']['fields'] for word in ('target','bearing','distance','position','offset','surface')))
        # The forced moves of an episode are read by the flight loop's pushes; the policy is called with the two pictures and the sentence.
        self.assertNotIn("episode['pushes']",source[source.index("inputs=policy_inputs("):source.index("inference=policy.infer(")])


class GateTests(unittest.TestCase):
    def setUp(self):
        import scripts.generic_landing_1d as tool
        self.tool=tool;self.stage=stage1d()

    def pilot(self,critical=12,same=6,pad=3,cylinder=3,collisions=0,wrong=0,air=0,high=0):
        """The pilot's twenty-four flights: `critical` of the forced twelve landed, and so on; `air` of the failed forced ones stopped in the air."""
        records=[];states=('low_hover','descent_decay','edge_low','edge_drift')
        for index in range(12):
            record=flight('cube','land',index<critical);record['episode']['recovery']=states[index%4];record['forced']={'all_came_to_pass_and_seen':True,'last_let_go_in_the_air':True};records.append(record)
        targets=[('pad','blue_pad')]*4+[('cube','blue_cube'),('pad','red_pad')]
        for index,(kind,target) in enumerate(targets):
            record=flight(kind,'land',index<same,target=target);record['episode']['distractor']='blue_cone';records.append(record)
        records+=[flight('pad','land',index<pad) for index in range(3)]+[flight('cylinder','land',index<cylinder) for index in range(3)]
        chosen=[record for record in records if record['episode'].get('distractor')]
        for record in chosen[:collisions]:record['summary']['hit']='SceneBlueCone';record['surface_landing'].update(success=False,collision='SceneBlueCone',landed_on=None)
        for record in chosen[collisions:collisions+wrong]:record['surface_landing'].update(success=False,landed_on='blue_cone',contact_from_above=True)
        failed=[record for record in records if record['episode'].get('recovery') and not record['surface_landing']['success']]
        for index,record in enumerate(failed):record['surface_landing']['contact_from_above']=index>=air
        for index,record in enumerate(records):record['summary']['height_range_m']=[3.,20. if index<high else 12.]
        return records

    def test_only_a_candidate_that_meets_every_line_is_eligible_and_the_order_ranks_the_eligible(self):
        order=self.stage['pilot']['order'];self.assertTrue(order[0].startswith('collisions in the same-colour starts'));self.assertTrue(order[1].startswith('landings on a wrong object'))
        self.assertTrue(order[2].startswith('forced cube landings'));self.assertTrue(order[3].startswith('flights that stop by themselves'))
        runs={'stage1c':self.pilot(critical=9,air=3),'500':self.pilot(critical=12,wrong=1),'1000':self.pilot(critical=11),'1500':self.pilot(critical=12,high=3)}
        rows,eligible,ranked=self.tool.judge_pilot(self.stage,runs,{'500':.06,'1000':.06,'1500':.05},14.);by={row['candidate']:row for row in rows}
        self.assertEqual((by['stage1c']['critical'],by['stage1c']['critical_air_self_stops'],by['1000']['cube_landings'],by['1000']['cube_of'],by['1000']['pad_landings'],by['1000']['pad_of_all']),(9,3,12,13,3,8))
        # 500 landed on a wrong object once: not eligible, however many forced landings it made. 1500 before 1000: more forced landings.
        self.assertEqual(eligible,['1500','1000']);self.assertFalse(by['500']['eligible']);self.assertEqual([row[0] for row in by['500']['checks'] if not row[3]],['landings on a wrong object'])
        self.assertEqual(ranked[-1],'500');self.assertNotIn('stage1c',ranked)
        # The lines one by one.
        failed=lambda **changes:[row[0] for row in self.tool.judge_pilot(self.stage,{'stage1c':self.pilot(critical=9),'500':self.pilot(**changes)},{},14.)[0][1]['checks'] if not row[3]]
        self.assertEqual(failed(),[]);self.assertEqual(failed(critical=10,air=1),[]);self.assertEqual(failed(critical=9),['forced cube landings that landed'])
        self.assertEqual(failed(critical=10,air=2),['stops in the air among the forced cube landings']);self.assertEqual(failed(collisions=1),['collisions in the same-colour starts'])
        self.assertEqual(failed(pad=2),['pad landings']);self.assertEqual(failed(cylinder=2),[]);self.assertEqual(failed(cylinder=1),['cylinder landings'])
        # Not worse than stage 1c's checkpoint on the same forced starts, even above the line of ten.
        rows,eligible,ranked=self.tool.judge_pilot(self.stage,{'stage1c':self.pilot(critical=12),'500':self.pilot(critical=11)},{},14.)
        self.assertEqual(eligible,[]);self.assertEqual([row[0] for row in rows[1]['checks'] if not row[3]],["forced cube landings against stage 1c's checkpoint on the same starts"])
        # Ties are broken down the order: fewer stops in the air, then fewer flights above the ceiling, then the loss, then the earlier update.
        runs={'stage1c':self.pilot(critical=9),'500':self.pilot(critical=11,air=1),'1000':self.pilot(critical=11),'1500':self.pilot(critical=11,high=2)}
        self.assertEqual(self.tool.judge_pilot(self.stage,runs,{},14.)[1],['1000','1500','500'])
        self.assertEqual(self.tool.judge_pilot(self.stage,{'stage1c':self.pilot(critical=9),'500':self.pilot(),'1000':self.pilot()},{'500':.07,'1000':.06},14.)[1],['1000','500'])

    def qualification(self,cube=10,same=4,pad=3,cylinder=3,air=0,collisions=0,wrong=0):
        records=[];states=('low_hover','descent_decay','edge_low','edge_drift')
        for index in range(10):
            record=flight('cube','land',index<cube)
            if index<4:record['episode']['recovery']=states[index]
            records.append(record)
        for index,target in enumerate(('blue_pad','blue_pad','blue_cube','red_pad')):
            record=flight('cube' if 'cube' in target else 'pad','land',index<same,target=target);record['episode']['distractor']='blue_cone';records.append(record)
        records+=[flight('pad','land',index<pad) for index in range(3)]+[flight('cylinder','land',index<cylinder) for index in range(3)]
        failed=[record for record in records if not record['surface_landing']['success']]
        for index,record in enumerate(failed):record['surface_landing']['contact_from_above']=index>=air
        for record in records[10:10+collisions]:record['summary']['hit']='SceneBlueCone';record['surface_landing'].update(success=False,collision='SceneBlueCone')
        for record in records[14:14+wrong]:record['summary']['wrong_target']=True
        for record in records:record['summary']['height_range_m']=[3.,12.]
        return records

    def test_the_qualification_holds_ten_cube_landings_to_eight_and_allows_one_stop_in_the_air(self):
        failed=lambda **changes:[row[0] for row in self.tool.judge_qualification(self.stage,self.qualification(**changes),14.)[1] if not row[3]]
        self.assertEqual(failed(),[]);self.assertEqual(failed(cube=8),[]);self.assertEqual(failed(cube=8,air=1),[]);self.assertEqual(failed(cube=7),['cube landings (four forced, six plain)'])
        self.assertEqual(failed(cube=8,air=2),['flights that stop by themselves in the air']);self.assertEqual(failed(same=3),['same-colour starts landed on the named object'])
        self.assertEqual(failed(pad=2),['pad landings']);self.assertEqual(failed(cylinder=2),[]);self.assertEqual(failed(cylinder=1),['cylinder landings'])
        self.assertIn('collisions',failed(collisions=1));self.assertIn('collisions in the same-colour starts',failed(collisions=1));self.assertEqual(failed(wrong=1),['flights at a wrong object'])
        # The same-colour cube start is not one of the ten cube landings.
        row,checks=self.tool.judge_qualification(self.stage,self.qualification(),14.);self.assertEqual(checks[1][1],10);self.assertEqual(row['cube_of'],11)

    def test_the_regression_is_stage_1cs_and_one_landing_on_another_object_fails_it(self):
        def run(success,wrong_landing=0,hit=0):
            records=[flight('pad','land',index<success) for index in range(27)]+[flight('pad','approach',True) for index in range(9)]
            for index in range(hit):records[index]['summary']['hit']='SceneBlueCone'
            for record in records[success:success+wrong_landing]:record['surface_landing'].update(landed_on='blue_cube',contact_from_above=True)
            return records
        failed=lambda model,baseline:[row[0] for row in self.tool.judge_regression(self.stage,model,baseline)[1] if not row[3]]
        self.assertEqual(failed(run(27),run(27)),[]);self.assertEqual(failed(run(26),run(27)),[]);self.assertEqual(failed(run(25),run(27)),['pad landings'])
        self.assertEqual(failed(run(27,hit=1),run(27)),['collisions']);self.assertEqual(failed(run(26,wrong_landing=1),run(27)),['landings on another object than the named one'])

    def test_the_decision_needs_every_part_and_says_what_was_not_flown(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as root:
            names=('audit','inputs','replay','pilot','qualification','a','repeat-a','b','repeat-b','regression');paths={}
            for name in names:paths[name]=Path(root)/f'{name}.json';paths[name].write_text(json.dumps({'passed':True}),encoding='utf-8')
            def run(skip=(),fail=()):
                for name in fail:paths[name].write_text(json.dumps({'passed':False}),encoding='utf-8')
                arguments=[item for name in names if name not in skip for item in (f'--{name}',str(paths[name]))];out=Path(root)/'decision.json'
                code=subprocess.run([sys.executable,str(ROOT/'scripts/generic_landing_1d.py'),'decision',*arguments,'--output',str(out)],capture_output=True,text=True,cwd=ROOT).returncode
                return code,json.loads(out.read_text(encoding='utf-8'))
            code,result=run();self.assertEqual((code,result['stage_1d']),(0,'PASSED'))
            code,result=run(skip=('a','repeat-a','b','repeat-b','regression'));self.assertEqual((code,result['stage_1d']),(1,'NOT PASSED'));self.assertEqual(result['parts']['gate A (generic landing)'],'not flown')
            code,result=run(fail=('qualification',));self.assertEqual((code,result['stage_1d']),(1,'NOT PASSED'));self.assertIs(result['parts']['the qualification'],False)


class ReadingTests(unittest.TestCase):
    def test_the_last_metres_are_read_from_the_policys_own_decisions(self):
        from src.landing import failures as reading
        from src.visual_search.maps import load_landing
        config,oft,scene,surfaces=setting()
        step=lambda distance,above,down,forced=False,over=True:{'distance_m':distance,'height_m':8.+above,'action':[0.,down,0.],'over':over,'perturbed':forced,'landed':False,'on_surface':False,
                                                                 'bearing_deg':100.,'front_seen':False,'down_seen':True,'position':[0.,0.,0.]}
        record=flight('cube','land',False);record['surface_landing'].update(target_surface={'top_m':8.,'landable':True},contact_from_above=False,horizontal_error_m=None);record['summary']['reason']='model_stop'
        steps=[step(9.,3.5,0.),step(2.8,3.5,.4,forced=True),step(2.8,2.6,.4,forced=True),step(2.8,1.8,.3),step(2.8,1.5,.1),step(2.8,1.3,.02),step(2.8,1.2,0.),step(2.8,1.15,0.),step(2.8,1.1,0.),step(2.8,1.1,0.)]
        row=reading.final_descent_profile(record,steps,load_landing(),config,oft)
        self.assertEqual(row['let_go'],{'distance_m':2.8,'above_top_m':1.8,'over':True});self.assertEqual(row['low_band']['decisions'],7)
        self.assertEqual((row['low_band']['down_below_the_stop_band'],row['low_band']['stop_decisions'],row['low_band']['climbing']),(5,5,0))
        self.assertEqual((row['end'],row['ended_by_its_own_stop_in_the_band']),('AIR_SELF_STOP',True));self.assertEqual(row['end_place']['above_top_m'],1.1)
        self.assertEqual(row['descent_start_distance_m'],2.8);self.assertEqual(row['descent_start_height_above_top_m'],1.8)      # the forced descent is not the policy's


if __name__=='__main__':unittest.main()
