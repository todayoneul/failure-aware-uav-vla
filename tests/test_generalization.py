"""Generalisation phase: maps, start planning, the teacher, held-out separation, dataset and training guards.

Nothing here starts a simulator or loads a model.
"""
import json
import math
import random
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.visual_search.episodes import load_config,policy_inputs
from src.visual_search.maps import load_map,MapGeometry,scene_objects,structures
from src.visual_search.shapes import mesh,KINDS as SHAPES
from src.visual_search.generalization import (SearchTeacher,geometric_view,make_start,rollout,spec_for,kind_table,sector_of,exclusion,plan_targeted,
                                              in_wedge,plan_split,STATELESS_STRATEGIES,MEMORY_STRATEGIES,KINDS)
from src.aerovla_oft.spec import load_config as load_oft_config

CONFIG=load_config();OFT=load_oft_config();TEST_FILE=ROOT/'configs/generalization_test_spawns.json'


def toy_map():
    """An object at (50, 0) with a 9 m wall across the way at x = 20."""
    return {'id':'toy','name':'Toy','ground_z':-1.,'area':{'x':[-40.,100.],'y':[-60.,60.]},'altitude':{'start_m':[5.,9.],'ceiling_m':14.},
            'boxes':[],'structures':[{'name':'Wall','shape':'box','size_m':[3.,40.,9.],'position':[20.,0.],'color':[.6,.6,.6]}],
            'objects':{'red_cube':{'noun':'the red cube','object':'RedCube','shape':'box','size_m':[8.,8.,8.],'color':[.8,0.,0.]}},
            'layouts':{'a':{'red_cube':[50.,0.]}},'train_objects':['red_cube'],'train_layouts':['a']}


class MapTests(unittest.TestCase):
    def test_maps_load_and_every_layout_places_defined_objects(self):
        for name in ('blocks','yard'):
            config=load_map(name)
            self.assertEqual(config['id'],name)
            for layout,places in config['layouts'].items():
                geometry=MapGeometry(config,layout)
                self.assertEqual(set(geometry.objects),set(places))
                for target,item in geometry.objects.items():
                    self.assertTrue(geometry.inside_area(*item['centre']),f'{name}/{layout}/{target}')
                    # Room to stand off at the stop distance on at least one side.
                    self.assertGreater(geometry.nearest(*item['centre'],ignore=(target,))[0],10.)

    def test_the_pilot_layout_is_the_untouched_map_and_held_out_objects_stay_out_of_training_layouts(self):
        blocks=load_map('blocks')
        self.assertEqual(structures(blocks,'pilot'),[]);self.assertEqual(scene_objects(blocks,'pilot'),[])
        self.assertEqual(len(structures(blocks,'a')),4)
        for layout in blocks['train_layouts']:
            self.assertFalse(set(blocks['held_out_objects'])&set(blocks['layouts'][layout]))
        self.assertEqual(blocks['layouts']['a']['red_cube'],blocks['layouts']['b']['green_cylinder'])
        self.assertEqual(load_map('yard')['train_layouts'],[])

    def test_native_objects_come_from_the_simulator_audit(self):
        blocks=load_map('blocks');audit=json.loads((ROOT/'configs/maps/blocks.obstacles.json').read_text())
        self.assertEqual(MapGeometry(blocks,'pilot').objects['blue_cone']['centre'],audit['objects']['Cone_5']['center'])
        self.assertGreater(len(audit['boxes']),50);self.assertEqual(max(box['top_m'] for box in audit['boxes']),25.)

    def test_a_wall_hides_the_object_until_the_viewer_is_above_it(self):
        geometry=MapGeometry(toy_map(),'a')
        self.assertFalse(geometry.sees(10.,0.,6.,'red_cube'));self.assertTrue(geometry.sees(10.,0.,13.,'red_cube'))
        self.assertTrue(geometry.sees(30.,0.,6.,'red_cube'));self.assertTrue(geometry.sees(10.,40.,6.,'red_cube'))
        self.assertEqual(geometry.blocking((10.,0.,6.),(50.,0.,4.)),'Wall')

    def test_free_distance_ahead_counts_only_what_reaches_up_to_the_vehicle(self):
        geometry=MapGeometry(toy_map(),'a')
        self.assertAlmostEqual(geometry.ahead(10.,0.,0.,6.,60.,2.5,ignore=('red_cube',)),8.5)
        self.assertEqual(geometry.ahead(10.,0.,0.,12.,60.,2.5,ignore=('red_cube',)),60.)
        self.assertEqual(geometry.ahead(10.,0.,math.pi,6.,60.,2.5,ignore=('red_cube',)),60.)
        self.assertFalse(geometry.corridor_clear((10.,0.),(50.,0.),6.,3.5,2.5,ignore=('red_cube',)))
        self.assertTrue(geometry.corridor_clear((10.,0.),(50.,0.),12.,3.5,2.5,ignore=('red_cube',)))

    def test_meshes_are_valid_glb_with_the_requested_bounding_box(self):
        for kind in SHAPES:
            data=mesh(kind,[8.,6.,9.] if kind=='box' else [8.,8.,9.],(.5,.1,.1))
            magic,version,length=struct.unpack('<4sII',data[:12]);self.assertEqual((magic,version,length),(b'glTF',2,len(data)))
            size,tag=struct.unpack('<I4s',data[12:20]);self.assertEqual(tag,b'JSON')
            document=json.loads(data[20:20+size]);bounds=document['accessors'][0]
            extent=[high-low for low,high in zip(bounds['min'],bounds['max'])]
            # glTF is Y-up: x, height, y.
            self.assertAlmostEqual(extent[1],9.,places=4);self.assertAlmostEqual(extent[0],8.,places=4)
            self.assertEqual(document['materials'][0]['pbrMetallicRoughness']['baseColorFactor'][:3],[.5,.1,.1])
        with self.assertRaises(ValueError):mesh('teapot',[1.,1.,1.],(0.,0.,0.))


class TeacherTests(unittest.TestCase):
    def view(self,**changes):
        base={'bearing_deg':90.,'distance_m':40.,'visible':False,'below':False,'ahead_m':60.,'height_m':6.};base.update(changes);return base

    def test_rules(self):
        teacher=SearchTeacher(CONFIG,OFT,'right',14.);yaw=OFT['action_bounds']['yaw_rad'][1]
        self.assertEqual(teacher.act(self.view()),([0.,0.,yaw],'search'))
        self.assertEqual(teacher.act(self.view(ahead_m=8.)),([0.,-.5,0.],'climb'))
        # At the ceiling a blocked view is swept past instead.
        self.assertEqual(teacher.act(self.view(ahead_m=8.,height_m=14.))[1],'search')
        action,state=teacher.act(self.view(visible=True,bearing_deg=5.))
        self.assertEqual(state,'approach');self.assertGreater(action[0],.5);self.assertAlmostEqual(action[2],math.radians(5.))
        self.assertEqual(teacher.act(self.view(visible=True,bearing_deg=40.))[1],'align')
        self.assertEqual(teacher.act(self.view(visible=True,bearing_deg=0.,ahead_m=4.))[1],'climb')
        self.assertEqual(teacher.act(self.view(visible=True,distance_m=11.)),([0.,0.,0.],'stop'))
        self.assertEqual(teacher.act(self.view(below=True,distance_m=5.))[1],'stop')

    def test_it_never_stops_or_advances_on_a_target_it_cannot_see(self):
        teacher=SearchTeacher(CONFIG,OFT,'right',14.)
        for distance in (3.,11.,30.):
            action,state=teacher.act(self.view(distance_m=distance))
            self.assertEqual(state,'search');self.assertEqual(action[0],0.)

    def test_the_training_strategy_is_a_function_of_the_view_alone(self):
        rng=random.Random(0)
        def views(count):
            return [self.view(bearing_deg=rng.uniform(-180,180),distance_m=rng.uniform(5,70),visible=rng.random()<.4,below=rng.random()<.1,
                              ahead_m=rng.choice((4.,10.,60.)),height_m=rng.uniform(5,14)) for _ in range(count)]
        for strategy in STATELESS_STRATEGIES:
            used=SearchTeacher(CONFIG,OFT,strategy,14.)
            for view in views(200):used.act(view)
            for view in views(200):
                self.assertEqual(used.act(view),SearchTeacher(CONFIG,OFT,strategy,14.).act(view))

    def test_the_other_strategies_depend_on_what_came_before(self):
        hidden=self.view()
        left=SearchTeacher(CONFIG,OFT,'left',14.);self.assertLess(left.act(hidden)[0][2],0)
        scan=SearchTeacher(CONFIG,OFT,'scan',14.);signs=[math.copysign(1,scan.act(hidden)[0][2]) for _ in range(40)]
        self.assertEqual(signs[0],1.);self.assertIn(-1.,signs);self.assertLess(signs.index(-1.),8)
        recall=SearchTeacher(CONFIG,OFT,'last_seen',14.);recall.act(self.view(visible=True,bearing_deg=-20.))
        self.assertLess(recall.act(hidden)[0][2],0)
        counted=SearchTeacher(CONFIG,OFT,'sweep_climb',14.);states=[counted.act(hidden)[1] for _ in range(40)]
        self.assertEqual(states[0],'search');self.assertIn('climb',states)
        self.assertEqual(set(MEMORY_STRATEGIES),{'left','scan','last_seen','sweep_climb'})
        with self.assertRaises(ValueError):SearchTeacher(CONFIG,OFT,'random',14.)

    def test_geometric_view_matches_the_map(self):
        geometry=MapGeometry(toy_map(),'a')
        behind_wall=geometric_view(CONFIG,geometry,'red_cube',10.,0.,6.,0.)
        self.assertFalse(behind_wall['visible']);self.assertAlmostEqual(behind_wall['ahead_m'],8.5);self.assertAlmostEqual(behind_wall['bearing_deg'],0.)
        self.assertTrue(geometric_view(CONFIG,geometry,'red_cube',10.,0.,13.,0.)['visible'])
        turned=geometric_view(CONFIG,geometry,'red_cube',30.,0.,6.,math.radians(-90.))
        self.assertFalse(turned['visible']);self.assertAlmostEqual(turned['bearing_deg'],90.)
        self.assertTrue(geometric_view(CONFIG,geometry,'red_cube',47.,0.,13.,0.)['below'])


class PlanningTests(unittest.TestCase):
    def test_kinds_and_objects_are_balanced_over_consecutive_seeds(self):
        table=kind_table(CONFIG);weights=CONFIG['generalization']['kinds']
        self.assertEqual({kind:table.count(kind) for kind in weights},weights);self.assertEqual(set(weights),set(KINDS))
        targets=['a','b','c','d'];specs=[spec_for(CONFIG,seed,targets) for seed in range(1000,1000+len(table)*len(targets))]
        for target in targets:
            mine=[spec['kind'] for spec in specs if spec['target']==target]
            self.assertEqual({kind:mine.count(kind) for kind in weights},weights)
        self.assertEqual(spec_for(CONFIG,1234,targets),spec_for(CONFIG,1234,targets))

    def test_each_kind_of_start_is_what_it_says_and_is_reproducible(self):
        config=toy_map();geometry=MapGeometry(config,'a')
        for kind in KINDS:
            episode=make_start(CONFIG,OFT,config,'a','red_cube',kind,'medium',7,'train')
            self.assertIsNotNone(episode,kind);self.assertEqual(episode,make_start(CONFIG,OFT,config,'a','red_cube',kind,'medium',7,'train'))
            x,y=episode['start_xy'];bearing=abs(episode['target_bearing_deg'])
            self.assertEqual(episode['strategy'],'right');self.assertEqual(episode['sector'],sector_of(episode['target_bearing_deg']))
            self.assertTrue(24<=episode['start_distance_m']<=44);self.assertTrue(5<=episode['start_height_m']<=9)
            self.assertEqual(geometry.sees(x,y,episode['start_height_m'],'red_cube'),kind!='altitude')
            if kind in ('visible','reacquire'):self.assertLessEqual(bearing,10.)
            if kind=='peripheral':self.assertTrue(25<=bearing<=38)
            if kind=='search':self.assertGreaterEqual(bearing,60.)
            self.assertEqual('kick' in episode,kind=='reacquire')
            self.assertEqual(episode['plan']['climbed_m']>=1.5,kind=='altitude')
            # The sentence is the only thing a policy is told, and it says nothing about where.
            policy_inputs(None,None,episode['instruction']);self.assertIn('the red cube',episode['instruction'])
            for forbidden in ('xy','bearing','distance','direction'):self.assertNotIn(forbidden,episode['instruction'].lower())

    def test_the_teacher_reaches_the_target_in_the_dry_run(self):
        config=toy_map();geometry=MapGeometry(config,'a')
        episode=make_start(CONFIG,OFT,config,'a','red_cube','altitude','medium',3,'train')
        plan=rollout(CONFIG,OFT,geometry,episode,14.)
        self.assertEqual(plan['reason'],'teacher_stop');self.assertLessEqual(plan['final_distance_m'],CONFIG['success_radius_m'])
        self.assertGreaterEqual(plan['clearance_m'],CONFIG['generalization']['rollout_clearance_m']);self.assertIn('climb',plan['states'])
        # A start whose way to the target stays blocked up to the ceiling is not an episode.
        tall=toy_map();tall['structures'][0]['size_m']=[3.,40.,30.]
        self.assertIsNone(make_start(CONFIG,OFT,tall,'a','red_cube','altitude','medium',3,'train'))

    def test_exclusion_keeps_training_starts_out_of_wedges_and_away_from_held_out_starts(self):
        held={'sets':{'G1':[{'map':'toy','start_xy':[80.,0.]}]},'wedges':{'toy':{'a':{'red_cube':[330.,30.]}}}}
        excluded=exclusion(held,'toy','a','red_cube',6.)
        self.assertTrue(excluded(80.,3.,90.));self.assertFalse(excluded(80.,7.,90.));self.assertTrue(excluded(0.,0.,350.));self.assertTrue(excluded(0.,0.,10.))
        self.assertTrue(in_wedge(0.,[330.,30.]));self.assertFalse(in_wedge(180.,[330.,30.]));self.assertTrue(in_wedge(250.,[240.,300.]))
        for seed in range(12):
            episode=make_start(CONFIG,OFT,toy_map(),'a','red_cube','visible','medium',seed,'train',excluded=excluded)
            self.assertFalse(in_wedge(episode['start_direction_deg'],[330.,30.]))
            self.assertGreaterEqual(math.hypot(episode['start_xy'][0]-80.,episode['start_xy'][1]),6.)


class HeldOutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.held=json.loads(TEST_FILE.read_text(encoding='utf-8'))

    def test_the_held_out_file_is_reproduced_exactly_by_its_generator(self):
        from scripts.plan_generalization import build_test
        self.assertEqual(json.loads(json.dumps(build_test(CONFIG,OFT))),self.held)

    def test_levels_cover_what_they_claim(self):
        sets=self.held['sets'];blocks=load_map('blocks');yard=load_map('yard')
        self.assertEqual({name:len(items) for name,items in sets.items()},{'G1':32,'G2':12,'G3':20,'G4':6,'P':24,'S':16})
        self.assertEqual({e['target'] for e in sets['G1']},set(blocks['train_objects']));self.assertEqual({e['layout'] for e in sets['G1']},set(blocks['train_layouts']))
        self.assertEqual({e['kind'] for e in sets['G1']},set(KINDS))
        self.assertEqual(sum(e['in_wedge'] for e in sets['G1']),16)
        self.assertEqual({e['target'] for e in sets['G2']},set(blocks['held_out_objects']));self.assertTrue(all(e['map']=='blocks' for e in sets['G2']))
        self.assertTrue(all(e['map']=='yard' for e in sets['G3']+sets['G4']));self.assertEqual({e['target'] for e in sets['G3']},set(yard['train_objects']))
        self.assertEqual({e['target'] for e in sets['G4']},set(yard['held_out_objects']))
        self.assertEqual({e['instruction_id'] for e in sets['P']},set(CONFIG['generalization']['instructions']['held_out']))
        self.assertFalse({e['instruction_id'] for e in sets['G1']}&{e['instruction_id'] for e in sets['P']})
        low,high=CONFIG['generalization']['seeds']['test']
        self.assertTrue(all(low<=e['seed']<=high for items in sets.values() for e in items))
        ids=[e['id'] for items in sets.values() for e in items];self.assertEqual(len(ids),len(set(ids)))

    def test_the_side_probe_turns_the_vehicle_on_the_spot(self):
        by_target={}
        for episode in self.held['sets']['S']:by_target.setdefault(episode['target'],[]).append(episode)
        for target,group in by_target.items():
            self.assertEqual(len({tuple(e['start_xy']) for e in group}),1);self.assertEqual(len({e['start_height_m'] for e in group}),1)
            self.assertEqual(sorted(e['target_bearing_deg'] for e in group),[-135.,-90.,90.,135.])

    def test_wedges_belong_to_places_not_objects(self):
        wedges=self.held['wedges']['blocks'];blocks=load_map('blocks')
        self.assertEqual(wedges['a']['blue_cone'],wedges['b']['blue_cone']);self.assertEqual(wedges['a']['red_cube'],wedges['b']['green_cylinder'])
        for layout in blocks['train_layouts']:
            for target,(low,high) in wedges[layout].items():self.assertEqual(high-low,self.held['wedge_deg'])

    def test_planned_training_starts_are_clear_of_every_held_out_start(self):
        from scripts.plan_generalization import check
        blocks=load_map('blocks');seeds=CONFIG['generalization']['seeds']
        plans={'train':plan_split(CONFIG,OFT,blocks,'train',range(seeds['train'][0],seeds['train'][0]+24),self.held),
               'val':plan_split(CONFIG,OFT,blocks,'val',range(seeds['val'][0],seeds['val'][0]+8),self.held)}
        self.assertEqual(check(CONFIG,self.held,plans),[])
        self.assertEqual({e['target'] for e in plans['train']},set(blocks['train_objects']))
        self.assertTrue(all(e['instruction_id'] in CONFIG['generalization']['instructions']['train'] for e in plans['train']+plans['val']))
        self.assertFalse({e['seed'] for e in plans['train']}&{e['seed'] for e in plans['val']})
        # The check has teeth: a start copied from the held-out file is reported, and so is a seed from the test range.
        intruder=dict(plans['train'][0],start_xy=self.held['sets']['G1'][0]['start_xy'])
        self.assertTrue(any('held-out' in problem for problem in check(CONFIG,self.held,{'train':[intruder],'val':[]})))
        self.assertTrue(any('seed' in problem for problem in check(CONFIG,self.held,{'train':[dict(plans['train'][0],seed=5)],'val':[]})))
        wedge=self.held['wedges']['blocks'][plans['train'][0]['layout']][plans['train'][0]['target']]
        self.assertTrue(any('wedge' in problem for problem in check(CONFIG,self.held,{'train':[dict(plans['train'][0],start_direction_deg=wedge[0]+1.)],'val':[]})))


class DatasetTests(unittest.TestCase):
    def record(self,identifier,split,seed,strategy='right',climbed=0.,kick=None):
        summary={'id':identifier,'case':'search','kind':'search','target':'red_cube','reason':'teacher_stop','instruction':'Find the red cube.','split':split,
                 'seed':seed,'map':'blocks','layout':'a','strategy':strategy,'initial_distance_m':30.,'target_bearing_deg':90.,'start_height_m':6.4,
                 'start_xy':[float(seed),0.],'initially_seen':False,'climbed_m':climbed}
        if kick:summary['kick']={'after_ticks':3,'degrees':kick}
        return {'traj_rel_dir':f'blocks/{identifier}','summary':summary,
                'steps':[{'step':i,'img_name':f'{i:06d}.png','action':[0.,0.,.21] if i<3 else [.5,0.,0.],'teacher_action':[0.,0.,.21] if i<3 else [.5,0.,0.],
                          'perturbed':False,'proprio':[0.]*5,'front_seen':i>=3,'front_in_fov':i>=3,'down_seen':False,'bearing_deg':90.-30*i if i<3 else 0.,
                          'distance_m':30.-i,'height_m':6.4,'teacher_state':'search' if i<3 else 'approach'} for i in range(6)]}

    def write(self,root,records):
        (root/'episodes').mkdir(parents=True)
        for record in records:(root/'episodes'/f'{record["summary"]["id"]}.json').write_text(json.dumps(record))

    def test_planned_episodes_are_split_by_seed_range_and_summarised(self):
        from scripts.build_visual_search_dataset import build
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.write(root,[self.record('train-1000-x','train',1000),self.record('train-1001-x','train',1001,climbed=5.),
                             self.record('train-1002-x','train',1002,strategy='left',kick=-90.),self.record('val-5000-x','val',5000)])
            splits,summary=build(root,OFT,.2,0)
        self.assertEqual({s['meta']['episode_id'] for s in splits['val']},{'val-5000-x'});self.assertEqual(len(splits['train']),18)
        self.assertEqual(summary['split_by'],'seed range of the plan');self.assertEqual(summary['seeds'],{'train':[1000,1002],'val':[5000,5000]})
        self.assertFalse(any(summary['leaks'].values()));self.assertEqual(summary['labels_outside_action_bounds'],0)
        for key in ('episodes','samples','maps','objects','start_distance_m','start_bearing_sector','target_visible','search_strategy_episodes','search_episodes'):
            self.assertIn(key,summary)
        self.assertEqual(summary['search_episodes'],{'yaw_only':3,'yaw_and_altitude':1});self.assertEqual(summary['search_strategy_episodes'],{'left':1,'right':3})
        self.assertEqual(summary['memory_strategy_samples'],6);self.assertEqual(summary['forced_turns'],{'left':1})
        self.assertEqual(splits['train'][0]['meta']['strategy'],'right');self.assertEqual(splits['train'][0]['meta']['seed'],1000)

    def test_shared_trajectories_or_starts_between_the_two_sides_are_reported(self):
        from scripts.build_visual_search_dataset import build
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);twin=self.record('val-5000-x','val',5000);twin['summary']['start_xy']=[1000.,0.];twin['traj_rel_dir']='blocks/train-1000-x'
            self.write(root,[self.record('train-1000-x','train',1000),twin])
            leaks=build(root,OFT,.2,0)[1]['leaks']
        self.assertEqual(leaks['folders'],['blocks/train-1000-x']);self.assertEqual(leaks['starts'],1)

    def test_mirrored_sweep_changes_only_the_turns_of_a_sweep(self):
        from scripts.mix_sweep_direction import mirrored,mix
        from scripts.build_visual_search_dataset import episode_samples
        samples=episode_samples(self.record('train-1000-x','train',1000),4);flipped=mirrored(samples[0],.21)
        self.assertEqual([step[2] for step in flipped['chunk']],[-.21,-.21,-.21,0.]);self.assertEqual(flipped['label']['yaw'],-.21)
        self.assertEqual([step[0] for step in flipped['chunk']],[step[0] for step in samples[0]['chunk']])
        self.assertEqual(samples[0]['chunk'][0][2],.21)
        everything=mix(samples,1.,.21)
        self.assertEqual([s['label']['yaw'] for s in everything],[-.21,-.21,-.21,0.,0.,0.])
        self.assertEqual(mix(samples,0.,.21),samples)


class TrainingGuardTests(unittest.TestCase):
    def test_an_existing_checkpoint_is_never_overwritten(self):
        from scripts.train_aerovla_oft import train
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'manifest.json').write_text('{}')
            with self.assertRaises(SystemExit) as raised:train(types.SimpleNamespace(output=folder))
        self.assertIn('never overwritten',str(raised.exception))

    def test_validation_subset_is_fixed_and_keeps_every_teacher_state(self):
        from scripts.train_aerovla_oft import stratified
        samples=[{'meta':{'teacher_state':state},'index':i} for i,state in enumerate(['approach']*400+['search']*60+['climb']*9+['stop']*30)]
        chosen=stratified(samples,100)
        self.assertEqual(chosen,stratified(samples,100))
        counts={state:sum(s['meta']['teacher_state']==state for s in chosen) for state in ('approach','search','climb','stop')}
        self.assertGreaterEqual(counts['climb'],8);self.assertGreater(counts['approach'],counts['search']);self.assertLessEqual(len(chosen),100)
        self.assertEqual(stratified(samples,None),samples)

    def test_bounded_head_stays_inside_the_action_range(self):
        import torch
        from src.aerovla_oft.model import ActionHead,ACTION_DIM
        torch.manual_seed(0);states=torch.randn(3,4*ACTION_DIM,32)*50
        self.assertLessEqual(ActionHead(32,16,2,4,bounded=True)(states).abs().max().item(),1.)
        self.assertGreater(ActionHead(32,16,2,4)(states).abs().max().item(),1.)

    def test_plan_episodes_can_be_filtered_moved_to_another_layout_and_reworded_for_the_pilot(self):
        from scripts.visual_search import planned_episodes
        def options(**changes):
            base=dict(plan=str(TEST_FILE),set='G1',only=None,strategy=None,layout=None,pilot_verbs=False,named=None,skip=0,limit=0);base.update(changes)
            return types.SimpleNamespace(**base)
        everything=planned_episodes(options(),CONFIG);self.assertEqual(len(everything),32)
        self.assertEqual(len(planned_episodes(options(only=['blue_cone','orange_ball']),CONFIG)),16)
        self.assertEqual([e['id'] for e in planned_episodes(options(skip=4,limit=3),CONFIG)],[e['id'] for e in everything[4:7]])
        moved=planned_episodes(options(layout='pilot',pilot_verbs=True,only=['blue_cone']),CONFIG)
        self.assertTrue(all(e['layout']=='pilot' and e['planned_layout'] in ('a','b') for e in moved))
        for episode in moved:
            verb='Approach' if episode['kind'] in ('visible','peripheral') else 'Find'
            self.assertEqual(episode['instruction'],f'{verb} the blue cone.')
        # The start state itself is never changed by these options.
        self.assertEqual([e['start_xy'] for e in moved],[e['start_xy'] for e in everything if e['target']=='blue_cone'])
        self.assertEqual({e['strategy'] for e in planned_episodes(options(strategy='scan',limit=2),CONFIG)},{'scan'})
        # The control names another object of the same scene and leaves the scored object alone.
        told=planned_episodes(options(set='G2',named='red_cube'),CONFIG)
        self.assertTrue(all(e['target']=='yellow_pyramid' and 'the red cube' in e['instruction'] and e['id'].endswith('-told-red_cube') for e in told))
        for episode in planned_episodes(options(set='G3',named='another'),CONFIG):
            self.assertNotEqual(episode['named_object'],episode['target'])
            self.assertIn(load_map('yard')['objects'][episode['named_object']]['noun'],episode['instruction'])

    def test_launchers_expose_plans_ports_and_levels(self):
        launcher=(ROOT/'scripts/run_visual_search.ps1').read_bytes().decode()
        for text in ('[string]$Plan','[string]$Set','$TopicsPort','-topicsport=','[string]$Layout',"[ValidateSet('step','continuous')][string]$Flight"):
            self.assertIn(text,launcher)
        demo=(ROOT/'scripts/run_visual_search_demo.ps1').read_bytes().decode()
        self.assertIn("[ValidateSet('','G1','G2','G3','G4','P','S')][string]$Level",demo);self.assertIn('configs/generalization_test_spawns.json',demo)


class ReportTests(unittest.TestCase):
    def episode(self,**changes):
        base={'success':False,'reason':'max_steps','stopped':False,'acquired_step':4,'minimum_distance_m':30.,'initial_distance_m':40.,
              'stopped_at_other':[],'initially_seen':False,'kind':'search','target_bearing_deg':-90.,'first_turn':'right'}
        base.update(changes);return base

    def test_failures_are_named_by_what_happened(self):
        from scripts.summarize_generalization import failure
        self.assertEqual(failure(self.episode(success=True),15.),'success')
        self.assertEqual(failure(self.episode(reason='collision'),15.),'collision')
        self.assertEqual(failure(self.episode(stopped=True,reason='model_stop',stopped_at_other=['orange_ball']),15.),'stopped at another object')
        self.assertEqual(failure(self.episode(stopped=True,reason='model_stop'),15.),'stopped away from the target')
        self.assertEqual(failure(self.episode(acquired_step=None),15.),'target never came into view')
        self.assertEqual(failure(self.episode(minimum_distance_m=12.),15.),'reached the target but did not stop')
        self.assertEqual(failure(self.episode(minimum_distance_m=38.),15.),'target in view but no approach')
        self.assertEqual(failure(self.episode(),15.),'approached but did not arrive')

    def test_first_turns_are_grouped_by_the_side_the_target_was_on(self):
        from scripts.summarize_generalization import turns
        table=turns([self.episode(),self.episode(target_bearing_deg=100.,first_turn='right'),self.episode(target_bearing_deg=170.,first_turn=None,acquired_step=None),
                     self.episode(side='behind_left',first_turn='left'),self.episode(initially_seen=True)])
        self.assertEqual(table['left'],{'right':1,'left':0,'none':0,'acquired':1,'n':1});self.assertEqual(table['right']['right'],1)
        self.assertEqual(table['behind'],{'right':0,'left':0,'none':1,'acquired':0,'n':1});self.assertEqual(table['behind_left']['left'],1)

    def test_observer_shows_the_generalisation_context(self):
        import hashlib
        from scripts.visual_search_viewer import render
        front=np.zeros((256,256,3),dtype=np.uint8);digest={name:hashlib.sha256(front.tobytes()).hexdigest() for name in ('front','down')}
        step={'step':3,'front_seen':True,'down_seen':False,'front_pixel':[100.,120.],'down_pixel':None,'distance_m':31.2,'bearing_deg':-12.,
              'action':[.5,0.,.1],'input_sha256':digest,'chunk':[[.5,0.,.1]]*4}
        live={'mode':'VISUAL SEARCH','model':'oft','model_name':'OFT-Generalization-v1','map':'Yard','target':'the red cube','level':'G3 - unseen map',
              'episode':{'id':'g3-0122','instruction':'Find the red cube.'},'step':step,'success_radius_m':15.,'failure':'NORMAL','raw_sha256':dict(digest)}
        shown=render(live,front,front);plain=render({k:v for k,v in live.items() if k not in ('model_name','map','target','level')},front,front)
        self.assertEqual(shown.shape,(760,1180,3));self.assertFalse(np.array_equal(shown[:80],plain[:80]))


class TargetedPlanTests(unittest.TestCase):
    """Gen-v2: extra episodes about what to do once the target is found."""
    def test_a_targeted_kind_keeps_its_base_rules_and_its_own_ranges(self):
        config=toy_map();geometry=MapGeometry(config,'a')
        zone=make_start(CONFIG,OFT,config,'a','red_cube','stop_zone','near',5,'train',spec={'base':'visible','distance_m':[9.,16.],'bearing_deg':[0.,25.],'height_m':[5.,14.]})
        self.assertEqual((zone['kind'],zone['base']),('stop_zone','visible'));self.assertTrue(9<=zone['start_distance_m']<=16)
        self.assertLessEqual(abs(zone['target_bearing_deg']),25.);self.assertTrue(geometry.sees(*zone['start_xy'],zone['start_height_m'],'red_cube'))
        self.assertEqual(zone['plan']['climbed_m'],0.);self.assertIn('stop',zone['plan']['states'])
        high=make_start(CONFIG,OFT,config,'a','red_cube','high_approach','medium',5,'train',spec={'base':'visible','height_m':[9.,14.]})
        self.assertTrue(9<=high['start_height_m']<=14);self.assertTrue(24<=high['start_distance_m']<=44)
        climb=make_start(CONFIG,OFT,config,'a','red_cube','climb_to_view','medium',5,'train',spec={'base':'altitude'})
        self.assertGreaterEqual(climb['plan']['climbed_m'],1.5);self.assertFalse(geometry.sees(*climb['start_xy'],climb['start_height_m'],'red_cube'))
        # Without a spec nothing about the earlier kinds changes: no extra field, same band as asked.
        plain=make_start(CONFIG,OFT,config,'a','red_cube','visible','medium',5,'train')
        self.assertNotIn('base',plain);self.assertEqual(plain['band'],'medium')

    def test_the_recipe_is_planned_with_its_own_seeds_clear_of_the_held_out_starts(self):
        from scripts.plan_generalization import check
        held=json.loads(TEST_FILE.read_text(encoding='utf-8'));blocks=load_map('blocks');recipe=json.loads(json.dumps(CONFIG['generalization']['targeted_v2']))
        self.assertEqual({spec['base'] for spec in recipe['kinds'].values()},{'altitude','visible','peripheral','search'})
        for spec in recipe['kinds'].values():spec['count']=[4,1]
        plans={split:plan_targeted(CONFIG,OFT,blocks,split,recipe,held) for split in ('train','val')}
        self.assertEqual(check(CONFIG,held,plans),[])
        self.assertEqual({e['kind'] for e in plans['train']},set(recipe['kinds']));self.assertEqual(len(plans['train']),4*len(recipe['kinds']))
        self.assertGreaterEqual(min(e['seed'] for e in plans['train']),recipe['seeds']['train'])
        self.assertGreaterEqual(min(e['seed'] for e in plans['val']),recipe['seeds']['val'])
        seeds=[e['seed'] for split in plans.values() for e in split];self.assertEqual(len(seeds),len(set(seeds)))
        for kind in recipe['kinds']:
            sides=[e['target_bearing_deg']>0 for e in plans['train'] if e['kind']==kind];self.assertEqual(sum(sides),2,kind)
        self.assertEqual(plans,{split:plan_targeted(CONFIG,OFT,blocks,split,recipe,held) for split in ('train','val')})

    def test_merging_datasets_refuses_anything_recorded_twice(self):
        from scripts.build_visual_search_dataset import load_records,collisions,build_records
        records=DatasetTests()
        with tempfile.TemporaryDirectory() as folder:
            first=Path(folder)/'first';second=Path(folder)/'second'
            records.write(first,[records.record('train-1000-x','train',1000),records.record('val-5000-x','val',5000)])
            records.write(second,[records.record('train-2000-x','train',2000,climbed=5.)])
            merged=load_records(first,'../first/')+load_records(second,'../second/')
            self.assertFalse(any(collisions(merged).values()))
            splits,summary=build_records(merged,OFT,.2,0)
            self.assertEqual(summary['sources'],{'first':2,'second':1});self.assertEqual(summary['episodes'],3)
            self.assertEqual({s['traj_rel_dir'].split('/')[1] for s in splits['train']},{'first','second'})
            again=collisions(merged+load_records(second,'../second/'))
            self.assertEqual(again['episode_ids'],['train-2000-x']);self.assertTrue(again['seeds'] and again['folders'] and again['starts'])


class FreezeTests(unittest.TestCase):
    def rows(self,seen,heights=None,position=(0.,0.)):
        return [{'front_seen':flag,'down_seen':False,'height_m':(heights or [6.]*len(seen))[i],'position':[position[0],position[1],-6.]} for i,flag in enumerate(seen)]

    def episode(self,**changes):
        base={'success':False,'reason':'max_steps','stopped':False,'minimum_distance_m':30.,'initial_distance_m':40.,'final_distance_m':30.,
              'initially_seen':False,'stopped_at_other':[],'target':'blue_cone','map':'blocks','layout':'pilot'}
        base.update(changes);return base

    def test_every_failure_gets_exactly_one_category(self):
        from scripts.freeze_generalization_results import classify,CATEGORIES
        seen=self.rows([False,True,True]);cache={}
        self.assertIsNone(classify(self.episode(success=True),seen,15.,cache))
        self.assertEqual(classify(self.episode(reason='collision'),seen,15.,cache),'F7')
        self.assertEqual(classify(self.episode(),self.rows([False,False]),15.,cache),'F1')
        self.assertEqual(classify(self.episode(stopped=True,reason='model_stop',stopped_at_other=['orange_ball']),seen,15.,cache),'F2')
        self.assertEqual(classify(self.episode(),seen,15.,cache),'F3')
        self.assertEqual(classify(self.episode(stopped=True,reason='model_stop',minimum_distance_m=16.,final_distance_m=16.),seen,15.,cache),'F4')
        self.assertEqual(classify(self.episode(minimum_distance_m=12.,final_distance_m=25.),seen,15.,cache),'F5')
        self.assertEqual(classify(self.episode(minimum_distance_m=12.),self.rows([False,True,True],[6.,6.,12.]),15.,cache),'F6')
        self.assertEqual(classify(self.episode(stopped=True,minimum_distance_m=16.),seen,15.,cache,base_success=True),'F8')
        self.assertEqual(classify(self.episode(reason='invalid_action'),seen,15.,cache),'F10')
        self.assertEqual(set(CATEGORIES),{f'F{i}' for i in range(1,11)})

    def test_stages_separate_finding_from_reaching_and_stopping(self):
        from scripts.freeze_generalization_results import stages,climb_after_sight
        found=stages(self.episode(minimum_distance_m=18.,stopped=True),self.rows([False,False,True]))
        self.assertEqual(found,{'started_hidden':True,'acquired':True,'approached':True,'stopped':True,'success':False})
        self.assertIsNone(stages(self.episode(initially_seen=True),self.rows([True,True]))['acquired'])
        self.assertFalse(stages(self.episode(),self.rows([False,False]))['acquired'])
        self.assertEqual(climb_after_sight(self.rows([False,True,True,True],[6.,9.,14.,12.])),5.)
        self.assertEqual(climb_after_sight(self.rows([False,False],[6.,20.])),0.)


class LongRangeTests(unittest.TestCase):
    """The fresh held-out starts by start distance, and how a flown episode of them is read."""
    @classmethod
    def setUpClass(cls):
        from scripts import plan_generalization as planner
        cls.planner=planner;cls.held_out=json.loads(TEST_FILE.read_text(encoding='utf-8'))
        cls.data=json.loads(planner.LONG_FILE.read_text(encoding='utf-8'));cls.settings=CONFIG['generalization']['long_range']

    def test_the_long_range_file_is_reproduced_exactly_by_its_generator(self):
        self.assertEqual(self.planner.build_long(CONFIG,OFT,self.held_out),self.data)

    def test_every_band_holds_both_scenes_both_kinds_of_start_and_unseen_objects(self):
        for name,(low,high) in self.settings['ranges_m'].items():
            episodes=self.data['sets'][name];self.assertEqual(len(episodes),12)
            self.assertTrue(all(low<=e['start_distance_m']<=high and e['range']==name for e in episodes))
            self.assertEqual({e['map'] for e in episodes},{'blocks','yard'})
            self.assertEqual(sorted(e['kind'] for e in episodes),['search']*6+['visible']*6)
            self.assertEqual(sum(not e['object_seen_in_training'] for e in episodes),4)
            self.assertTrue(all((e['target']=='yellow_pyramid')!=e['object_seen_in_training'] for e in episodes))
            # In view at the start means the teacher's dry run sees it at once; a search start does not.
            self.assertTrue(all((e['plan']['acquired_tick']==0)==(e['kind']=='visible') for e in episodes))

    def test_seeds_are_fresh_and_starts_are_clear_of_every_earlier_start(self):
        low,high=self.settings['seeds'];seeds=CONFIG['generalization']['seeds'];episodes=[e for es in self.data['sets'].values() for e in es]
        self.assertTrue(all(low<=e['seed']<=high for e in episodes));self.assertEqual(len({e['seed'] for e in episodes}),len(episodes))
        self.assertTrue(all(high_<low for _,high_ in seeds.values()))
        earlier=self.planner.earlier_starts(self.held_out);self.assertGreater(len(earlier),500)
        for episode in episodes:
            for _,_,where,xy in earlier:
                if where==episode['map']:self.assertGreaterEqual(math.hypot(episode['start_xy'][0]-xy[0],episode['start_xy'][1]-xy[1]),self.settings['separation_m'])
        # The farthest band lies beyond anything a training episode started from.
        self.assertLess(self.data['training_start_distance_max_m'],self.settings['ranges_m']['L3'][0])
        self.assertTrue(all(e['nearest_training_start_m'] is None for e in episodes if e['map']=='yard'))

    def rows(self,seen,distances,stamps=None):
        return [{'front_seen':flag,'down_seen':False,'distance_m':distance,'height_m':6.,'position':[0.,0.,-7.],'action':[.5,0.,0.],'epoch':float(index)}
                for index,(flag,distance) in enumerate(zip(seen,distances))]

    def test_an_episode_is_read_in_stages_and_a_failure_gets_one_type(self):
        from scripts.freeze_long_range import measure,failure_type,TYPES
        base={'target':'blue_cone','map':'blocks','layout':'a','stopped':False,'reason':'max_steps','success':False,'object_seen_in_training':True}
        def read(seen,distances,**changes):
            episode=dict(base,**changes);measured=measure(episode,self.rows(seen,distances),15.,{})
            return measured,failure_type(episode,measured,15.)
        measured,kind=read([False,False,False],[60.,60.,60.])
        self.assertEqual((measured['search_success'],measured['first_acquisition_step'],measured['timeout'],kind),(False,None,True,'LONG_SEARCH_FAILURE'))
        measured,kind=read([False,True,True],[60.,55.,40.])
        self.assertEqual((measured['search_success'],measured['first_acquisition_step'],measured['approach_success'],kind),(True,1,False,'LONG_APPROACH_FAILURE'))
        measured,kind=read([True,True,True],[60.,30.,18.],stopped=True,reason='model_stop')
        self.assertEqual((measured['initially_visible'],measured['entered_20m'],measured['entered_15m'],measured['self_stop'],kind),(True,True,False,True,'LONG_STOP_FAILURE'))
        measured,kind=read([True,True],[60.,12.],stopped=True,reason='model_stop',success=True)
        self.assertEqual((measured['task_success'],measured['entered_15m'],kind),(True,True,None))
        self.assertEqual(read([True,True],[60.,12.],reason='collision')[1],'COLLISION')
        self.assertEqual(read([True,True],[60.,50.],stopped=True,reason='model_stop',stopped_at_other=['orange_ball'],object_seen_in_training=False)[1],
                         'UNSEEN_OBJECT_GROUNDING_FAILURE')
        self.assertEqual(read([True,True],[60.,50.],stopped=True,reason='model_stop',stopped_at_other=['orange_ball'])[1],'LONG_APPROACH_FAILURE')
        self.assertEqual(len(TYPES),5)
        # `In view` is the simulator's word: a policy that keeps turning past the target is told apart from one that never had it in view.
        from scripts.freeze_long_range import mechanism
        measured,kind=read([True,False,True,False,True],[60.]*5)
        self.assertEqual((measured['times_came_into_view'],measured['decisions_in_view'],kind,mechanism(measured)),(3,3,'LONG_APPROACH_FAILURE','passed_over'))
        self.assertEqual(mechanism(read([True,True],[60.,50.],stopped=True,reason='model_stop')[0]),'early_stop')

    def test_tallies_count_stages_over_all_episodes_and_acquisition_over_hidden_starts(self):
        from scripts.freeze_long_range import tally
        item=lambda **changes:dict({'initially_visible':False,'ever_acquired':True,'first_acquisition_step':10,'entered_20m':True,'entered_15m':True,
                                    'self_stop':True,'task_success':True,'collision':False,'timeout':False},**changes)
        row=tally([item(),item(initially_visible=True,first_acquisition_step=0),item(ever_acquired=False,first_acquisition_step=None,entered_20m=False,
                                                                                  entered_15m=False,self_stop=False,task_success=False,timeout=True)])
        self.assertEqual((row['episodes'],row['acquired'],row['started_hidden'],row['hidden_acquired'],row['entered_20m'],row['success'],row['timeout']),(3,2,2,1,2,2,1))
        self.assertEqual(row['median_steps_to_acquire'],10)


if __name__=='__main__':unittest.main()
