"""Gen-v3: scenes with distractors, the landing teacher, the fresh test set, the training plan and the landing score.

Nothing here starts a simulator or loads a model.
"""
import copy
import json
import math
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.visual_search.episodes import load_config,policy_inputs
from src.visual_search.maps import load_map,MapGeometry,scene_objects,structures,owner_of,lighting,load_landing,object_position
from src.visual_search.generalization import (SearchTeacher,geometric_view,make_start,rollout,variant,related,distractor_ok,others_in_view,
                                              instruction_for_id)
from src.visual_search import gen_v3
from src.aerovla_oft.spec import load_config as load_oft_config,is_stop

CONFIG=load_config();OFT=load_oft_config();LANDING=load_landing();V3=CONFIG['gen_v3']
TEST_FILE=ROOT/'configs/gen_v3_test_spawns.json'
SCENES={name:load_map(name) for name in ('field','lot','depot','blocks','yard')}


def view(**changes):
    base={'bearing_deg':0.,'distance_m':30.,'visible':True,'below':False,'over':False,'ahead_m':60.,'height_m':7.,'surface_m':2.5,'task':'land','landed':False}
    base.update(changes);return base


class SceneTests(unittest.TestCase):
    def test_objects_come_from_one_catalogue_and_differ_in_exactly_the_attributes_they_claim(self):
        objects=SCENES['field']['objects']
        self.assertEqual(objects['blue_pad']['color'],objects['blue_cube']['color']);self.assertEqual(objects['blue_pad']['color'],objects['blue_cylinder']['color'])
        self.assertEqual(objects['blue_pad']['size_m'],objects['red_pad']['size_m']);self.assertNotEqual(objects['blue_pad']['color'],objects['red_pad']['color'])
        self.assertTrue(related(SCENES['field'],'blue_pad','red_pad','same_shape'));self.assertFalse(related(SCENES['field'],'blue_pad','red_pad','same_color'))
        self.assertTrue(related(SCENES['field'],'blue_pad','blue_cube','same_color'));self.assertFalse(related(SCENES['field'],'blue_pad','blue_cube','same_shape'))
        self.assertFalse(related(SCENES['field'],'blue_pad','blue_pad','any'));self.assertTrue(related(SCENES['field'],'blue_pad','orange_ball',['orange_ball']))
        self.assertEqual(LANDING['pad']['size_m'],objects['blue_pad']['size_m']);self.assertEqual(LANDING['pad']['top_surface_m'],objects['blue_pad']['size_m'][2])
        self.assertEqual(LANDING['touchdown']['landing_region_half_width_m'],objects['blue_pad']['size_m'][0]/2-LANDING['vehicle']['half_span_m'])

    def test_a_scene_written_around_its_centre_is_moved_as_a_whole(self):
        field=SCENES['field'];raw=json.loads((ROOT/'configs/maps/field.json').read_text(encoding='utf-8'))
        ox,oy=raw['origin']
        self.assertEqual(field['sites']['S1'],[raw['sites']['S1'][0]+ox,raw['sites']['S1'][1]+oy])
        self.assertEqual(object_position(field,'a','blue_pad'),field['sites']['S1'])
        self.assertEqual(field['area']['x'],[raw['area']['x'][0]+ox,raw['area']['x'][1]+ox])
        self.assertTrue(all(abs(item['position'][0]-ox)<300 for item in field['structures']))

    def test_objects_stand_apart_and_clear_of_the_scenery(self):
        for name in ('field','lot','depot'):
            scene=SCENES[name]
            for layout in scene['layouts']:
                geometry=MapGeometry(scene,layout);centres=[item['centre'] for item in geometry.objects.values()]
                # Two success radii never overlap, so the object a vehicle stops at is unambiguous.
                self.assertGreaterEqual(min(math.dist(a,b) for i,a in enumerate(centres) for b in centres[i+1:]),2*CONFIG['success_radius_m']+10)
                scenery=[box for box in geometry.boxes if box['name'] not in geometry.objects]
                self.assertGreaterEqual(min(geometry._gap(box,*centre) for box in scenery for centre in centres),20.)

    def test_every_object_moves_between_layouts_and_the_test_layout_uses_a_place_training_never_did(self):
        for name in ('field','lot'):
            scene=SCENES[name];train=scene['train_layouts']
            for target in scene['train_objects']:
                self.assertGreaterEqual(len({tuple(object_position(scene,layout,target)) for layout in train}),3,target)
            held=scene['held_out_layouts'][0]
            self.assertNotIn(tuple(object_position(scene,held,'blue_pad')),{tuple(object_position(scene,layout,'blue_pad')) for layout in train})
        for a,b in SCENES['depot']['swaps'].items():
            first,second=SCENES['depot']['layouts'][a],SCENES['depot']['layouts'][b]
            moved=[target for target in first if first[target]!=second[target]]
            self.assertEqual(len(moved),2);self.assertIn('blue_pad',moved);self.assertEqual(first[moved[0]],second[moved[1]])

    def test_training_scenes_never_show_the_held_out_object_and_the_test_scene_is_never_trained_in(self):
        for name in ('field','lot'):self.assertTrue(all('yellow_pyramid' not in places for places in SCENES[name]['layouts'].values()))
        self.assertTrue(all('yellow_pyramid' not in SCENES['blocks']['layouts'][layout] for layout in V3['train_layouts']['blocks']))
        self.assertEqual(SCENES['depot']['train_layouts'],[]);self.assertNotIn('depot',V3['train_layouts']);self.assertNotIn('yard',V3['train_layouts'])
        self.assertTrue(all('yellow_pyramid' in places for places in SCENES['depot']['layouts'].values()))
        self.assertEqual({tuple(object_position(SCENES['blocks'],layout,'blue_pad')) for layout in ('c','d')},{(52.,-84.),(52.,84.)})

    def test_a_pad_is_seen_by_its_top_and_carries_its_marking(self):
        scene=SCENES['field'];geometry=MapGeometry(scene,'a');centre=geometry.objects['blue_pad']['centre']
        points=geometry.sight_points('blue_pad')
        self.assertTrue(all(point[2]==2.5 and geometry.over(point[0],point[1],'blue_pad') for point in points));self.assertEqual(points[1][:2],centre)
        self.assertTrue(geometry.over(centre[0]+6.9,centre[1]-6.9,'blue_pad'));self.assertFalse(geometry.over(centre[0]+7.1,centre[1],'blue_pad'))
        self.assertEqual(len(MapGeometry(scene,'a').sight_points('blue_cube')),4)
        items=scene_objects(scene,'a');marks=[item for item in items if item['name'].startswith('BluePadMark')]
        self.assertEqual(len(marks),3)
        for mark in marks:
            self.assertEqual(mark['lift_m'],2.5);self.assertTrue(geometry.over(mark['position'][0]+mark['size_m'][0]/2,mark['position'][1]+mark['size_m'][1]/2,'blue_pad'))
            self.assertEqual(owner_of(scene,'a',mark['name']),'blue_pad')
        self.assertEqual(owner_of(scene,'a','BluePad'),'blue_pad');self.assertEqual(owner_of(scene,'a','RedPad'),'red_pad');self.assertIsNone(owner_of(scene,'a','FieldBarn'))
        self.assertEqual(len({item['name'] for item in items}),len(items))

    def test_lighting_and_ground_change_with_the_layout(self):
        field=SCENES['field']
        self.assertNotEqual(lighting(field,'a'),lighting(field,'c'));self.assertEqual(lighting(field,'a'),lighting(field,'e'))
        ground=lambda layout:[item['name'] for item in structures(field,layout) if not item.get('obstacle',True)]
        self.assertEqual(len(ground('a')),1);self.assertNotEqual(ground('a'),ground('c'))
        times={lighting(SCENES[name],layout)['time_of_day'] for name in ('field','lot') for layout in SCENES[name]['train_layouts']}
        self.assertEqual(len(times),4);self.assertNotIn(lighting(SCENES['depot'],'c1')['time_of_day'],times|{SCENES['yard']['lighting']['time_of_day']})


class LandingTeacherTests(unittest.TestCase):
    def setUp(self):self.teacher=SearchTeacher(CONFIG,OFT,'right',14.,LANDING)

    def test_the_sentence_decides_how_the_flight_ends(self):
        near=dict(distance_m=11.,visible=True)
        action,state=self.teacher.act(view(task='approach',**near));self.assertEqual((action,state),([0.,0.,0.],'stop'))
        action,state=self.teacher.act(view(task='land',**near));self.assertEqual(state,'final');self.assertGreater(action[0],.5);self.assertEqual(action[1],0.)

    def test_it_descends_only_over_the_middle_and_stops_only_after_touchdown(self):
        action,state=self.teacher.act(view(distance_m=1.,over=True,below=True,height_m=8.));self.assertEqual((action,state),([0.,.5,0.],'descend'))
        action,state=self.teacher.act(view(distance_m=1.,over=True,below=True,height_m=4.));self.assertEqual((action,state),([0.,LANDING['teacher']['final_descent_m'],0.],'descend'))
        self.assertFalse(is_stop(action,OFT))
        action,state=self.teacher.act(view(distance_m=4.,over=True,below=True,height_m=8.));self.assertEqual(action[1],0.);self.assertGreater(action[0],0.)
        action,state=self.teacher.act(view(distance_m=1.,over=True,height_m=2.5,landed=True));self.assertEqual((action,state),([0.,0.,0.],'landed'))
        self.assertTrue(is_stop(action,OFT))

    def test_it_eases_in_over_the_pad_and_turns_before_it_flies(self):
        far=self.teacher.act(view(distance_m=30.))[0][0];close=self.teacher.act(view(distance_m=3.,over=True))[0][0]
        self.assertEqual(far,1.);self.assertAlmostEqual(close,3./LANDING['teacher']['ease_m'])
        action,state=self.teacher.act(view(distance_m=5.,bearing_deg=120.,over=True));self.assertEqual(state,'align');self.assertEqual(action[0],0.);self.assertGreater(action[2],0.)

    def test_nothing_about_searching_changes_with_the_task(self):
        for changes in (dict(visible=False),dict(visible=False,ahead_m=8.),dict(visible=False,ahead_m=8.,height_m=13.9)):
            self.assertEqual(self.teacher.act(view(task='land',**changes)),self.teacher.act(view(task='approach',**changes)))
        # Out of both views and not over the pad: no descent, whatever the distance.
        self.assertEqual(self.teacher.act(view(visible=False,distance_m=1.))[1],'search')

    def test_coming_in_high_it_loses_height_on_the_way(self):
        action,_=self.teacher.act(view(distance_m=20.,height_m=13.));self.assertEqual(action[1],LANDING['teacher']['glide_down_m']);self.assertEqual(action[0],1.)
        self.assertEqual(self.teacher.act(view(distance_m=40.,height_m=13.))[0][1],0.);self.assertEqual(self.teacher.act(view(distance_m=20.,height_m=8.))[0][1],0.)

    def test_the_dry_run_lands_on_the_pad_or_stops_short_of_it(self):
        scene=SCENES['field'];spec=gen_v3.spec_of(CONFIG,'land_visible')
        land=make_start(CONFIG,OFT,scene,'a','blue_pad','land_visible','v3',1,'t',spec=spec,task='land',instruction_id='land',max_ticks=V3['max_ticks'])
        self.assertEqual(land['task'],'land');self.assertEqual(land['instruction'],'Find the blue landing pad and land on it.')
        self.assertIn('descend',land['plan']['states']);self.assertEqual(land['plan']['states']['landed'],CONFIG['stop_ticks'])
        plan=rollout(CONFIG,OFT,MapGeometry(scene,'a'),land,14.);self.assertTrue(plan['landed']);self.assertLess(plan['final_distance_m'],LANDING['teacher']['align_radius_m'])
        approach=variant(CONFIG,OFT,scene,land,'twin',task='approach',instruction_id='approach')
        self.assertEqual((approach['start_xy'],approach['start_yaw_deg'],approach['start_height_m']),(land['start_xy'],land['start_yaw_deg'],land['start_height_m']))
        self.assertEqual(approach['instruction'],'Approach the blue landing pad.');self.assertNotIn('descend',approach['plan']['states'])
        stopped=rollout(CONFIG,OFT,MapGeometry(scene,'a'),approach,14.)
        self.assertFalse(stopped['landed']);self.assertAlmostEqual(stopped['final_distance_m'],CONFIG['arrive_distance_m'],delta=1.5)

    def test_a_twin_changes_only_what_it_is_asked_to(self):
        scene=SCENES['depot'];base=make_start(CONFIG,OFT,scene,'c1','blue_pad','pair_in_view','v3',8,'t',spec=gen_v3.spec_of(CONFIG,'pair_in_view'),instruction_id='find',
                                              max_ticks=V3['max_ticks'])
        other=variant(CONFIG,OFT,scene,base,'other',target='red_pad')
        if other:
            self.assertEqual(other['start_xy'],base['start_xy']);self.assertEqual(other['instruction'],'Find the red landing pad.')
            centre=MapGeometry(scene,'c1').objects['red_pad']['centre']
            self.assertAlmostEqual(other['start_distance_m'],math.dist(centre,base['start_xy']),places=1)
        self.assertIsNone(variant(CONFIG,OFT,scene,dict(base,max_ticks=3),'short'))

    def test_distractor_rules_read_the_first_view(self):
        scene=SCENES['field'];shown=[{'name':'red_pad','bearing_deg':-8.,'distance_m':40.},{'name':'blue_cube','bearing_deg':30.,'distance_m':120.}]
        self.assertTrue(distractor_ok({'relation':'same_shape','within_m':90.},scene,'blue_pad',35.,shown))
        self.assertFalse(distractor_ok({'relation':'same_color','within_m':90.},scene,'blue_pad',35.,shown))
        self.assertTrue(distractor_ok({'relation':'same_color'},scene,'blue_pad',35.,shown))
        self.assertTrue(distractor_ok({'centred_deg':12.},scene,'blue_pad',35.,shown));self.assertFalse(distractor_ok({'centred_deg':5.},scene,'blue_pad',35.,shown))
        self.assertTrue(distractor_ok({'max_gap_m':15.},scene,'blue_pad',35.,shown));self.assertFalse(distractor_ok({'max_gap_m':15.},scene,'blue_pad',60.,shown))
        self.assertTrue(distractor_ok({'nearer_by_m':8.},scene,'blue_pad',60.,shown));self.assertFalse(distractor_ok({'nearer_by_m':8.},scene,'blue_pad',45.,shown))
        geometry=MapGeometry(scene,'a');centre=geometry.objects['blue_pad']['centre'];partner=geometry.objects['red_pad']['centre']
        x,y=(centre[0]+partner[0])/2+35.,(centre[1]+partner[1])/2
        names={item['name'] for item in others_in_view(CONFIG,geometry,'blue_pad',x,y,7.,180.,40.)}
        self.assertIn('red_pad',names);self.assertNotIn('blue_pad',names)

    def test_sentences_name_the_object_and_carry_no_direction(self):
        scene=SCENES['field']
        for group in V3['instructions'].values():
            for phrase in group['train']+group['held_out']:
                sentence=instruction_for_id(CONFIG,scene,'blue_pad',phrase);self.assertIn('blue',sentence);self.assertIn('landing pad',sentence)
                policy_inputs(None,None,sentence)
        self.assertEqual(instruction_for_id(CONFIG,scene,'blue_pad','find_long_land'),'Find the blue rectangular landing pad and land on it.')
        self.assertEqual(instruction_for_id(CONFIG,scene,'blue_cube','find_long_land'),'Find the blue cube and land on it.')
        for task in V3['instructions'].values():self.assertFalse(set(task['train'])&set(task['held_out']))
        # The words the earlier prompt test held out stay held out.
        self.assertEqual(set(CONFIG['generalization']['instructions']['held_out']),set(V3['instructions']['approach']['held_out']))


class TestSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=json.loads(TEST_FILE.read_text(encoding='utf-8'));cls.sets=cls.data['sets']

    def test_the_test_file_is_reproduced_exactly_by_its_generator(self):
        self.assertEqual(json.loads(json.dumps(gen_v3.build_test(CONFIG,OFT))),self.data)

    def test_sets_have_the_planned_size_fresh_seeds_and_the_right_scene(self):
        self.assertEqual({name:len(episodes) for name,episodes in self.sets.items()},
                         {'Q1':8,'Q2':8,'Q3':8,'Q4':16,'Q5':16,'Q6':18,'Q7':18,'Q8':24,'Q8R':6,'Q10':18,'QS':16})
        low,high=V3['seeds']['test'];earlier=[CONFIG['generalization']['seeds'][name] for name in ('test','train','val')]+[CONFIG['generalization']['long_range']['seeds']]
        self.assertTrue(all(top<low for _,top in earlier))
        for name,episodes in self.sets.items():
            self.assertEqual(len({e['id'] for e in episodes}),len(episodes))
            for episode in episodes:
                self.assertTrue(low<=episode['seed']<=high);self.assertEqual(episode['max_ticks'],V3['max_ticks']);self.assertEqual(episode['set'],name)
                if name=='QS':self.assertIn((episode['map'],episode['layout']),(('field','e'),('lot','e')))
                else:self.assertEqual(episode['map'],'depot');self.assertFalse(episode['scene_seen_in_training'])
        self.assertEqual(self.data['canonical'],'Find the blue landing pad and land on it.')
        self.assertTrue(all(e['instruction']==self.data['canonical'] and e['task']=='land' and e['target']=='blue_pad' for e in self.sets['Q8']))

    def test_distractor_sets_show_the_named_distractor_and_their_swaps_keep_the_pose(self):
        within=V3['distractor_within_m']
        for name,partners in (('Q2',{'red_pad'}),('Q3',{'blue_cylinder','blue_cube'})):
            for episode in self.sets[name]:
                self.assertIn(episode['distractor'],partners)
                self.assertTrue(any(o['name']==episode['distractor'] and o['distance_m']<=within for o in episode['others_in_view']),episode['id'])
        bases={e['id']:e for e in self.sets['Q2']+self.sets['Q3']}
        for episode in self.sets['Q4']:
            base=bases[episode['swap_of']]
            self.assertEqual((episode['start_xy'],episode['start_yaw_deg'],episode['start_height_m'],episode['instruction']),
                             (base['start_xy'],base['start_yaw_deg'],base['start_height_m'],base['instruction']))
            self.assertEqual(episode['layout'],SCENES['depot']['swaps'][base['layout']])
            # The pad now stands where the distractor stood.
            self.assertEqual(object_position(SCENES['depot'],episode['layout'],'blue_pad'),object_position(SCENES['depot'],base['layout'],base['distractor']))

    def test_query_and_task_swaps_change_only_the_sentence(self):
        for pose in range(4):
            group=[e for e in self.sets['Q5'] if e['query_pose']==pose];self.assertEqual(len(group),4)
            self.assertEqual(len({(tuple(e['start_xy']),e['start_yaw_deg'],e['start_height_m'],e['layout']) for e in group}),1)
            targets=[e['target'] for e in group];self.assertEqual(len(set(targets)),4);self.assertEqual(targets[:2],['blue_pad','red_pad'])
            self.assertTrue(related(SCENES['depot'],'blue_pad',targets[2],'same_color'))
            self.assertFalse(related(SCENES['depot'],'blue_pad',targets[3],'same_color') or related(SCENES['depot'],'blue_pad',targets[3],'same_shape'))
        for pose in range(6):
            group=[e for e in self.sets['Q6'] if e['terminal_pose']==pose]
            self.assertEqual(len({(tuple(e['start_xy']),e['start_yaw_deg'],e['layout'],e['target']) for e in group}),1)
            self.assertEqual([(e['task'],e['instruction_id']) for e in group],[('land','land'),('approach','approach'),('approach','find_and_approach')])
            self.assertIn('descend',group[0]['plan']['states']);self.assertNotIn('descend',group[1]['plan']['states'])

    def test_far_and_landing_sets_cover_their_ranges_and_unseen_words(self):
        for band,(low,high) in V3['far_ranges_m'].items():
            group=[e for e in self.sets['Q7'] if e['range']==band];self.assertEqual(len(group),6)
            self.assertTrue(all(low<=e['start_distance_m']<=high and e['task']=='approach' for e in group))
            self.assertEqual(sum(e['target']=='blue_pad' for e in group),3);self.assertEqual(sum(e['plan']['acquired_tick']==0 for e in group),3)
        for band in ('near','medium','far'):
            group=[e for e in self.sets['Q8'] if e['range']==band];self.assertEqual(len(group),8);low,high=V3['ranges_m'][band]
            self.assertTrue(all(low-2.<=e['start_distance_m']<=high for e in group))
        unseen=set(V3['instructions']['land']['held_out']);bases={e['id'] for e in self.sets['Q8']}
        self.assertEqual({e['instruction_id'] for e in self.sets['Q10']},unseen);self.assertTrue(all(e['base'] in bases for e in self.sets['Q10']))
        self.assertEqual(sorted(e['range'] for e in self.sets['Q8R']),['far','far','medium','medium','near','near'])
        self.assertTrue(all(e['target']=='red_pad' and e['task']=='land' for e in self.sets['Q8R']))


class TrainingPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import plan_generalization as planner
        cls.planner=planner;cls.test=json.loads(TEST_FILE.read_text(encoding='utf-8'))
        cls.held_out=json.loads(planner.TEST_FILE.read_text(encoding='utf-8'));cls.long_range=json.loads(planner.LONG_FILE.read_text(encoding='utf-8'))
        # A recipe with two episodes of every kind is planned by the same code in a few seconds.
        cls.config=copy.deepcopy(CONFIG)
        for entry in cls.config['gen_v3']['kinds'].values():
            if 'count' in entry:entry['count']=[2,1]
        cls.points=gen_v3.test_points(cls.test,cls.held_out,cls.long_range)
        cls.plans={split:gen_v3.plan_training(cls.config,OFT,split,cls.points,cls.held_out) for split in ('train','val')}

    def test_the_plan_is_clear_of_everything_kept_for_testing(self):
        self.assertEqual(self.planner.check_v3(self.config,self.test,self.plans,self.held_out,self.long_range),[])
        train=self.plans['train'];self.assertGreaterEqual(len(train),2*sum('count' in e for e in V3['kinds'].values()))
        self.assertTrue(all(e['map'] in V3['train_layouts'] and e['layout'] in V3['train_layouts'][e['map']] for e in train))
        self.assertTrue(all(e['task']==V3['kinds'][e['kind'].replace('_approach','')]['task'] or e['kind'].endswith('_approach') for e in train))
        self.assertTrue(all(e['target'] in ('blue_pad','red_pad') for e in train if e['map']=='blocks' or e['task']=='land'))
        self.assertFalse({e['seed'] for e in train}&{e['seed'] for e in self.plans['val']})

    def test_a_landing_start_can_be_flown_again_as_an_approach(self):
        twins=[e for e in self.plans['train'] if e.get('twin_of')];by_id={e['id']:e for e in self.plans['train']}
        self.assertTrue(twins)
        for twin in twins:
            base=by_id[twin['twin_of']]
            self.assertEqual((twin['start_xy'],twin['start_yaw_deg'],twin['target'],twin['layout']),(base['start_xy'],base['start_yaw_deg'],base['target'],base['layout']))
            self.assertEqual((base['task'],twin['task']),('land','approach'));self.assertNotEqual(twin['instruction'],base['instruction'])

    def test_the_check_names_every_kind_of_violation(self):
        episode=copy.deepcopy(self.plans['train'][0]);check=lambda item:self.planner.check_v3(self.config,self.test,{'train':[item]},self.held_out,self.long_range)
        self.assertEqual(check(episode),[])
        self.assertIn('not a training layout',check(dict(episode,map='depot',layout='c1'))[0])
        self.assertIn('held-out words',' '.join(check(dict(episode,instruction_id='locate_land'))))
        self.assertIn('seed outside',' '.join(check(dict(episode,seed=8000))))
        near=next(e for e in self.test['sets']['QS'] if e['map']=='field')
        self.assertIn('evaluation start',' '.join(check(dict(episode,map='field',layout='a',start_xy=[near['start_xy'][0]+1.,near['start_xy'][1]]))))
        wedge=gen_v3.site_wedge(self.held_out,SCENES['blocks'],'c','blue_pad')
        self.assertEqual(wedge,self.held_out['wedges']['blocks']['a']['red_cube'])
        inside=dict(episode,map='blocks',layout='c',target='blue_pad',start_xy=[500.,500.],start_direction_deg=wedge[0]+1.)
        self.assertIn('held-out wedge',' '.join(check(inside)))

    def test_no_object_is_tied_to_one_sentence(self):
        full=gen_v3.plan_training(CONFIG,OFT,'val',self.points,self.held_out)
        words={}
        for episode in full:words.setdefault((episode['task'],episode['target']),set()).add(episode['instruction_id'])
        self.assertEqual(words[('land','blue_pad')],set(V3['instructions']['land']['train']))
        self.assertEqual(words[('approach','blue_pad')],set(V3['instructions']['approach']['train']))


if __name__=='__main__':unittest.main()
