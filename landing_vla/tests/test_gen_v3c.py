"""Gen-v3c: the hard-negative episodes, the layouts that keep training, pilot, validation and test apart, the fresh
validation set, the gates, and the pieces of the fine-tune that can be checked without a model."""
import json
import math
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.visual_search.episodes import load_config
from src.visual_search.maps import load_map,object_position,MapGeometry,lighting,structures
from src.visual_search.generalization import related,rollout,make_start
from src.visual_search import canonical,gen_v3c
from src.aerovla_oft.spec import load_config as load_oft_config

CONFIG=load_config();OFT=load_oft_config();V3C=CONFIG['gen_v3c'];HARD=V3C['hard_negative']
SCENES={name:load_map(name) for name in gen_v3c.SCENES}
GROUPS={'train':V3C['train_layouts'],'pilot':V3C['pilot_layouts'],'validation':V3C['validation_layouts'],'test':CONFIG['canonical']['test_layouts'],
        'earlier':CONFIG['gen_v3']['train_layouts']}


def pairs(scene,layout,reach):
    """Neighbouring sites of a layout with what stands on them: {(site, object, site, object)}."""
    raw={site:name for name,site in json.loads((ROOT/f'configs/maps/{scene["id"]}.json').read_text(encoding='utf-8'))['layouts'][layout].items()}
    sites=scene['sites']
    return {(a,raw[a],b,raw[b]) for a in sites for b in sites if a<b and math.dist(sites[a],sites[b])<=reach}


class LayoutTests(unittest.TestCase):
    def test_the_four_uses_have_their_own_layouts(self):
        for name in gen_v3c.SCENES:
            groups=[set(GROUPS[key][name]) for key in ('train','pilot','validation','test','earlier')]+[set(SCENES[name]['held_out_layouts'])-set(GROUPS['test'][name])]
            for index,first in enumerate(groups):
                for second in groups[index+1:]:self.assertFalse(first&second,name)
            self.assertTrue(all(layout in SCENES[name]['layouts'] for group in groups for layout in group))
            # The scene's own lists are as they were: the new layouts are neither its training layouts nor its sealed ones.
            self.assertEqual(SCENES[name]['train_layouts'],['a','b','c','d']);self.assertEqual(SCENES[name]['held_out_layouts'],['e','g','h','i','j'])

    def test_no_training_layout_repeats_a_neighbouring_pair_of_a_test_layout_and_validation_repeats_none_at_all(self):
        reach=V3C['reach_m']
        for name,scene in SCENES.items():
            def of(key,extra=()):return set().union(*[pairs(scene,layout,reach) for layout in list(GROUPS[key][name])+list(extra)])
            sealed=of('test',('e',))
            self.assertFalse(of('train')&sealed,name)
            for other in (of('train'),of('earlier'),of('pilot'),sealed):self.assertFalse(of('validation')&other,name)
            # A related pair of the pilot layouts is never one the model was trained beside.
            both=lambda items:{item for item in items if any(related(scene,item[1],item[3],kind) for kind in ('same_color','same_shape'))}
            self.assertFalse(both(of('pilot'))&both(of('train')|of('earlier')),name)

    def test_exchanged_layouts_differ_only_in_their_pairs_and_share_ground_and_light(self):
        for name,scene in SCENES.items():
            for layout in [l for key in ('train','pilot','validation') for l in GROUPS[key][name]]:
                other=scene['swaps'][layout];self.assertEqual(scene['swaps'][other],layout)
                moved=[target for target in scene['layouts'][layout] if scene['layouts'][layout][target]!=scene['layouts'][other][target]]
                self.assertEqual(len(moved),6,(name,layout))
                for target in moved:
                    partner=next(t for t in moved if scene['layouts'][other][t]==scene['layouts'][layout][target])
                    self.assertEqual(scene['layouts'][other][target],scene['layouts'][layout][partner])
                    self.assertIn(sorted((scene['layouts'][layout][target],scene['layouts'][layout][partner])),[sorted(pair) for pair in scene['pairs']])
                self.assertEqual(lighting(scene,layout),lighting(scene,other))
                ground=lambda which:[item['name'] for item in structures(scene,which) if not item.get('obstacle',True)]
                self.assertEqual(len(ground(layout)),1);self.assertEqual(ground(layout),ground(other))
                self.assertNotIn('yellow_pyramid',scene['layouts'][layout])

    def test_the_training_layouts_offer_every_pair_the_rows_name(self):
        for kind in ('color_first','shape_first','lost'):
            for target,other in HARD['kinds'][kind]['rows']:
                places=[(name,layout) for name in gen_v3c.SCENES for layout in V3C['train_layouts'][name]
                        if other in gen_v3c.neighbours(SCENES[name],layout,target,'related',V3C['reach_m'])]
                self.assertTrue(places,(kind,target,other))
                if kind!='lost':self.assertEqual(gen_v3c.relation_of(SCENES['field'],target,other),HARD['kinds'][kind]['relation'])


class HardNegativePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads(gen_v3c.TRAIN_PLAN.read_text(encoding='utf-8'));cls.all=cls.plan['train']+cls.plan['val']
        cls.by_id={episode['id']:episode for episode in cls.all}

    def test_the_tracked_plan_is_reproduced_exactly_by_its_generator(self):
        train=gen_v3c.build_training(CONFIG,OFT,'train');self.assertEqual(train,self.plan['train'])
        self.assertEqual(gen_v3c.build_training(CONFIG,OFT,'val',train),self.plan['val'])

    def test_the_counts_are_the_configured_ones_and_the_set_is_small(self):
        for column,split in enumerate(('train','val')):
            for kind,entry in HARD['kinds'].items():self.assertEqual(sum(e['kind']==kind for e in self.plan[split]),entry['count'][column],(split,kind))
        self.assertTrue(80<=len(self.plan['train'])<=120);self.assertTrue(len(self.all)<=120)

    def test_nothing_recorded_touches_an_evaluation_set(self):
        from scripts.gen_v3c import check_training
        self.assertEqual(check_training(CONFIG,{'train':self.plan['train'],'val':self.plan['val']}),[])
        points={}
        for path in (canonical.FILES['validation'],canonical.FILES['test'],gen_v3c.FILES['validation'],gen_v3c.FILES['pilot']):
            for episode in json.loads(path.read_text(encoding='utf-8'))['episodes']:points.setdefault(episode['map'],[]).append(episode['start_xy'])
        for episodes in json.loads(canonical.SEALED.read_text(encoding='utf-8'))['sets'].values():
            for episode in episodes:points.setdefault(episode['map'],[]).append(episode['start_xy'])
        for episode in self.all:
            self.assertIn(episode['layout'],V3C['train_layouts'][episode['map']])
            self.assertGreaterEqual(min(math.dist(episode['start_xy'],xy) for xy in points[episode['map']]),V3C['separation_m'],episode['id'])
            self.assertLessEqual(episode['start_distance_m'],V3C['max_start_m']);self.assertNotIn(episode['map'],('depot','yard','blocks'))
        # ... and a problem is reported when one does.
        moved=dict(self.plan['train'][0],layout=V3C['validation_layouts'][self.plan['train'][0]['map']][0])
        self.assertTrue(check_training(CONFIG,{'train':[moved],'val':[]}))
        near=dict(self.plan['train'][0],start_xy=points[self.plan['train'][0]['map']][0])
        self.assertTrue(any('earlier start' in line for line in check_training(CONFIG,{'train':[near],'val':[]})))

    def test_a_first_episode_shows_the_related_object_near_the_middle_and_not_the_named_one(self):
        rule=HARD['first_view'];view=CONFIG['generalization']['fov_half_deg']
        for episode in self.all:
            if episode['kind'] not in ('color_first','shape_first'):continue
            shown={item['name']:item for item in episode['others_in_view']};other=shown[episode['distractor']]
            self.assertLessEqual(abs(other['bearing_deg']),rule['centred_deg']);self.assertLessEqual(other['distance_m'],rule['within_m'])
            self.assertEqual(episode['relation'],HARD['kinds'][episode['kind']]['relation'])
            # No part of the named object is in the first view, and the teacher does not have it for at least a decision.
            half=math.degrees(math.atan(SCENES[episode['map']]['objects'][episode['target']]['size_m'][0]/2/episode['start_distance_m']))
            self.assertGreater(abs(episode['target_bearing_deg'])-half,view);self.assertGreater(episode['plan']['acquired_tick'],0)
            self.assertEqual(episode['plan']['climbed_m'],0.);self.assertIn('search',episode['plan']['states'])

    def test_a_lost_episode_begins_its_approach_and_is_turned_to_the_related_object(self):
        rule=HARD['lost'];step=math.degrees(OFT['action_bounds']['yaw_rad'][1]);sides=set()
        for episode in self.all:
            if episode['kind']!='lost':continue
            kick=episode['kick'];lost=episode['lost'];sides.add(kick['degrees']>0)
            self.assertEqual(episode['plan']['acquired_tick'],0);self.assertTrue(rule['after_ticks'][0]<=kick['after_ticks']<=rule['after_ticks'][1])
            self.assertGreaterEqual(abs(kick['degrees']),rule['min_turn_deg']-rule['centred_deg'])
            self.assertEqual(lost['turn_ends_tick'],kick['after_ticks']+math.ceil(abs(kick['degrees'])/step))
            self.assertGreater(lost['target_again_tick'],lost['turn_ends_tick']+1);self.assertLessEqual(abs(lost['distractor_bearing_deg']),rule['centred_deg']+1.)
            self.assertLessEqual(lost['distractor_distance_m'],rule['within_m']);self.assertEqual(episode['plan']['states']['kick'],lost['turn_ends_tick']-kick['after_ticks'])
            # The dry run agrees: after the forced turn the teacher has no target and turns on until it has.
            scene=SCENES[episode['map']];trace=[];rollout(CONFIG,OFT,MapGeometry(scene,episode['layout']),episode,scene['altitude']['ceiling_m'],trace=trace)
            after=trace[lost['turn_ends_tick']];self.assertFalse(after['visible']);self.assertEqual(after['state'],'search')
            self.assertTrue(all(row['state']=='kick' for row in trace[kick['after_ticks']:lost['turn_ends_tick']]))
            self.assertTrue(trace[lost['target_again_tick']]['visible'])
        self.assertEqual(sides,{True,False})

    def test_a_swap_twin_and_a_query_twin_keep_the_pose_and_change_one_thing(self):
        used=[]
        for twin in self.all:
            if twin['kind'] not in ('swap','query'):continue
            base=self.by_id[twin['twin_of']];used.append(base['id']);scene=SCENES[base['map']]
            self.assertEqual((twin['start_xy'],twin['start_yaw_deg'],twin['start_height_m'],twin['seed'],twin['split']),
                             (base['start_xy'],base['start_yaw_deg'],base['start_height_m'],base['seed'],base['split']))
            self.assertIn(base['kind'],('color_first','shape_first'));self.assertLessEqual(twin['start_distance_m'],HARD['twin_max_start_m'])
            self.assertEqual(twin['plan']['acquired_tick'],0)
            if twin['kind']=='swap':
                self.assertEqual(twin['layout'],scene['swaps'][base['layout']]);self.assertEqual((twin['target'],twin['instruction']),(base['target'],base['instruction']))
                self.assertEqual(object_position(scene,twin['layout'],twin['target']),object_position(scene,base['layout'],base['distractor']))
            else:
                self.assertEqual((twin['layout'],twin['target']),(base['layout'],base['distractor']));self.assertNotEqual(twin['instruction'],base['instruction'])
                self.assertIn(scene['objects'][twin['target']]['noun'],twin['instruction'])
        self.assertEqual(len(used),len(set(used)))

    def test_colours_shapes_bands_tasks_and_sides_are_spread(self):
        train=self.plan['train'];scene=SCENES['field'];of=lambda episode,key:scene['objects'][episode['target']]['attributes'][key];total=len(train)
        colours={colour:sum(of(e,'color')==colour for e in train) for colour in ('blue','red')};self.assertEqual(sum(colours.values()),total)
        self.assertGreaterEqual(colours['red'],.25*total)
        shapes={shape:sum(of(e,'shape')==shape for e in train) for shape in ('pad','cube','cylinder','cone')}
        self.assertEqual(sum(shapes.values()),total);self.assertGreaterEqual(min(shapes.values()),10);self.assertLessEqual(shapes['pad'],.5*total)
        self.assertLess(sum(e['target']=='blue_pad' for e in train),.35*total)
        bands={band:sum(e['band']==band for e in train) for band in V3C['bands_m']};self.assertGreaterEqual(bands['far'],.35*total);self.assertGreaterEqual(min(bands.values()),15)
        for episode in train:
            low,high=V3C['bands_m'][episode['band']]
            if episode['kind'] in ('color_first','shape_first','lost'):self.assertTrue(low<=episode['start_distance_m']<=high,episode['id'])
        self.assertTrue(all(e['task']=='approach' for e in train if e['target'] not in ('blue_pad','red_pad')))
        pads=[e for e in train if e['target'] in ('blue_pad','red_pad')];self.assertTrue(.5<=sum(e['task']=='land' for e in pads)/len(pads)<=.85)
        maps={name:sum(e['map']==name for e in train) for name in gen_v3c.SCENES};self.assertGreaterEqual(min(maps.values()),.35*total)
        unseen={phrase for task in CONFIG['gen_v3']['instructions'].values() for phrase in task['held_out']};self.assertFalse({e['instruction_id'] for e in self.all}&unseen)
        first=[e for e in train if e['kind'] in ('color_first','shape_first')]
        self.assertTrue(all(sum((e['target_bearing_deg']>0)==side for e in first)>=.3*len(first) for side in (True,False)))
        # Both kinds of relation, and every pair the canonical validation run got wrong, in both directions.
        named={(e['target'],e['distractor']) for e in train if e.get('distractor')}
        for pair in (('blue_pad','blue_cube'),('red_pad','red_cube'),('blue_pad','red_pad'),('blue_cube','blue_pad'),('red_cube','red_pad'),('red_pad','blue_pad')):self.assertIn(pair,named)


class EvaluationSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validation=json.loads(gen_v3c.FILES['validation'].read_text(encoding='utf-8'));cls.pilot=json.loads(gen_v3c.FILES['pilot'].read_text(encoding='utf-8'))

    def test_both_files_are_reproduced_exactly_by_their_generator(self):
        self.assertEqual(gen_v3c.build_validation(CONFIG,OFT),self.validation['episodes'])
        self.assertEqual(gen_v3c.build_pilot(CONFIG,OFT),self.pilot['episodes'])

    def test_the_validation_set_has_the_canonical_make_up_in_its_own_layouts(self):
        episodes=self.validation['episodes'];self.assertEqual(len(episodes),36);self.assertEqual(self.validation['evaluator'],'canonical_evaluator_v2')
        low,high=V3C['seeds']['validation']
        for band,span in V3C['bands_m'].items():
            mine=[e for e in episodes if e['band']==band];self.assertEqual(len(mine),12)
            self.assertEqual(sorted(e['role'] for e in mine),sorted(['visible','search','edge','past_color','past_shape','pair','swap','query','red_past','red_visible','approach','approach']))
            self.assertEqual(sum(e['task']=='land' for e in mine),9)
            for e in mine:
                if 'twin_of' not in e:self.assertTrue(span[0]<=e['start_distance_m']<=span[1],e['id'])
        for e in episodes:
            self.assertLessEqual(e['start_distance_m'],V3C['max_start_m']);self.assertIn(e['layout'],V3C['validation_layouts'][e['map']]);self.assertTrue(low<=e['seed']<=high)
            self.assertNotIn('kick',e)
        self.assertEqual({(e['map'],e['layout']) for e in episodes},{(name,layout) for name in gen_v3c.SCENES for layout in V3C['validation_layouts'][name]})
        self.assertEqual(sum(e['instruction']==CONFIG['canonical']['mission'] for e in episodes),21)
        self.assertTrue(all(.4<=sum(e['map']==name for e in episodes)/36<=.6 for name in gen_v3c.SCENES))

    def test_the_validation_set_asks_about_target_selection_in_every_band(self):
        within=V3C['validation']['past_within_m'];by_id={e['id']:e for e in self.validation['episodes']}
        for band in V3C['bands_m']:
            mine={e['role']:e for e in self.validation['episodes'] if e['band']==band and e['role']!='approach'}
            for role,relation in (('past_color','same_color'),('past_shape','same_shape')):
                e=mine[role];shown={item['name']:item for item in e['others_in_view']}
                self.assertEqual(e['relation'],relation);self.assertIn(e['neighbour'],shown);self.assertLessEqual(shown[e['neighbour']]['distance_m'],within)
                self.assertGreater(e['plan']['acquired_tick'],0);self.assertEqual(e['target'],'blue_pad')
            self.assertIn(mine['red_past']['neighbour'],{item['name'] for item in mine['red_past']['others_in_view']});self.assertEqual(mine['red_past']['target'],'red_pad')
            pair,swap,query=mine['pair'],mine['swap'],mine['query'];scene=SCENES[pair['map']]
            self.assertEqual((swap['twin_of'],query['twin_of']),(pair['id'],pair['id']));self.assertEqual(swap['layout'],scene['swaps'][pair['layout']])
            self.assertEqual(object_position(scene,swap['layout'],'blue_pad'),object_position(scene,pair['layout'],pair['neighbour']))
            self.assertEqual((query['target'],query['task'],query['instruction_id']),(pair['neighbour'],'approach','find'))
            self.assertEqual((swap['start_xy'],query['start_xy']),(pair['start_xy'],pair['start_xy']))
        relations=[e['relation'] for e in self.validation['episodes'] if e['role']=='red_past'];self.assertEqual(sorted(relations),['same_color','same_color','same_shape'])
        neighbours={e['neighbour'] for e in self.validation['episodes'] if e['role'] in ('past_color','pair')};self.assertTrue({'blue_cube','blue_cone','red_pad'}<=neighbours)

    def test_every_evaluation_start_is_away_from_every_earlier_start(self):
        earlier=canonical.earlier_points('test')+[(e['map'],e['start_xy']) for e in json.loads(canonical.FILES['test'].read_text(encoding='utf-8'))['episodes']]
        validation=[(e['map'],e['start_xy']) for e in self.validation['episodes']]
        for episodes,others in ((self.validation['episodes'],earlier),(self.pilot['episodes'],earlier+validation)):
            for e in episodes:
                self.assertGreaterEqual(min(math.dist(e['start_xy'],xy) for where,xy in others if where==e['map']),V3C['separation_m'],e['id'])

    def test_the_pilot_is_the_twelve_flights_it_says(self):
        episodes=self.pilot['episodes'];self.assertEqual(len(episodes),V3C['pilot']['episodes']);low,high=V3C['seeds']['pilot']
        self.assertEqual([e['role'] for e in episodes],['color_first']*4+['shape_first']*2+['lost']*2+['swap']*2+['search']*2)
        for e in episodes:
            self.assertIn(e['layout'],V3C['pilot_layouts'][e['map']]);self.assertTrue(low<=e['seed']<=high);self.assertLessEqual(e['start_distance_m'],V3C['max_start_m'])
            if e['role'].endswith('_first'):
                shown={item['name']:item for item in e['others_in_view']};self.assertLessEqual(abs(shown[e['distractor']]['bearing_deg']),HARD['first_view']['centred_deg'])
                self.assertGreater(e['plan']['acquired_tick'],0)
            if e['role']=='search':
                scene=SCENES[e['map']];self.assertFalse([item for item in e['others_in_view'] if gen_v3c.relation_of(scene,e['target'],item['name']) and item['distance_m']<=60])
            if e['role']=='lost':self.assertIn('kick',e)
            if e['role']=='swap':self.assertEqual(e['plan']['acquired_tick'],0)

    def test_the_smoke_starts_and_the_regression_starts_are_fixed(self):
        from scripts.gen_v3c import regression_ids
        chosen=canonical.smoke_ids(CONFIG,self.validation['episodes']);by_id={e['id']:e for e in self.validation['episodes']}
        self.assertEqual(len(set(chosen)),10);self.assertEqual(sorted(by_id[i]['band'] for i in chosen),['far']*3+['mid']*4+['near']*3)
        self.assertEqual(sum(by_id[i]['task']=='land' for i in chosen),7)
        again=regression_ids(CONFIG);self.assertEqual(again,regression_ids(CONFIG))
        self.assertEqual({name:len(ids) for name,ids in again.items()},{'G1':8,'G3':8,'P':8,'L1':3,'L2':3,'L3':3})
        self.assertTrue(all(len(set(ids))==len(ids) for ids in again.values()))


class GateAndFineTuneTests(unittest.TestCase):
    def test_the_gates_are_the_canonical_ones_and_the_fine_tune_is_a_small_correction(self):
        before=CONFIG['canonical']['gates'];now=V3C['gates']
        for name in ('smoke','representative'):
            self.assertEqual({k:v for k,v in now[name].items() if k!='what'},{k:v for k,v in before[name].items() if k!='what'})
        tune=V3C['fine_tune'];self.assertEqual((tune['updates'],tune['learning_rate'],tune['hard_negative_share'],tune['batch_size'],tune['gradient_accumulation']),(2000,5e-5,.5,1,4))
        self.assertTrue(1000<=tune['updates']<=3000);self.assertLess(tune['learning_rate'],OFT['train']['learning_rate'])
        self.assertEqual(tune['boost'],{'small_visible':2,'selection':3})
        self.assertEqual(V3C['start_checkpoint'],CONFIG['canonical']['checkpoint']);self.assertNotEqual(V3C['checkpoint'],V3C['start_checkpoint'])
        self.assertEqual((OFT['chunk_size'],OFT['proprio']['enabled'],OFT['head'].get('output','linear')),(4,False,'linear'))
        self.assertEqual(V3C['max_start_m'],CONFIG['canonical']['max_start_m']);self.assertEqual(V3C['evaluator'],'canonical_evaluator_v2')

    def test_the_gate_lines(self):
        from scripts.gen_v3c import checks
        item=lambda **changes:dict({'episode':'x','task':'land','band':'mid','success':True,'acquired':True,'grounded':True,'terminal_as_asked':True,'wrong_target':False,'collision':False},**changes)
        passed=lambda lines:all(value>=limit if how=='>=' else value<=limit for value,how,limit in lines.values())
        twelve=[item() for _ in range(9)]+[item(success=False) for _ in range(3)]
        self.assertTrue(passed(checks('pilot',CONFIG,twelve,[])[1]))
        self.assertFalse(passed(checks('pilot',CONFIG,twelve[:8]+[item(success=False)]*4,[])[1]))
        two_wrong=[item() for _ in range(10)]+[item(success=False,wrong_target=True) for _ in range(2)]
        self.assertFalse(passed(checks('pilot',CONFIG,two_wrong,[])[1]))
        self.assertFalse(passed(checks('pilot',CONFIG,twelve,['e'])[1]))
        good=[item() for _ in range(27)]+[item(task='approach') for _ in range(9)]
        self.assertTrue(passed(checks('representative',CONFIG,good,[])[1]))
        three_wrong=good[:33]+[item(success=False,wrong_target=True) for _ in range(3)]
        _,lines=checks('test',CONFIG,three_wrong,[]);self.assertFalse(passed(lines));self.assertEqual(lines['flights that ended at a wrong object'],(3,'<=',2))
        self.assertGreaterEqual(lines['success rate'][0],lines['success rate'][2])

    def test_a_named_object_lost_on_the_way_is_found_in_the_log(self):
        from scripts.gen_v3c import lost_and_found
        row=lambda distance,seen:{'distance_m':distance,'front_seen':seen,'down_seen':False};stage={'approach_m':20.}
        steady=[row(40.-index,True) for index in range(15)];self.assertEqual(lost_and_found(steady,stage),(False,False))
        lost=[row(40.,True),row(39.,True)]+[row(38.,False)]*4;self.assertEqual(lost_and_found(lost,stage),(True,False))
        self.assertEqual(lost_and_found(lost+[row(37.,True)],stage),(True,True))
        blink=[row(40.,True),row(39.,False),row(38.,False),row(37.,True)];self.assertEqual(lost_and_found(blink,stage),(False,False))
        # Out of view from the start is a search, not a loss; and below the approach distance the front view loses a pad anyway.
        self.assertEqual(lost_and_found([row(40.,False)]*6,stage),(False,False))
        self.assertEqual(lost_and_found([row(21.,True),row(19.,False),row(18.,False),row(17.,False),row(16.,False)],stage),(False,False))

    def test_the_validation_mix_is_half_new_and_half_old(self):
        import tempfile
        from scripts.gen_v3c import val_mix
        sample=lambda folder,state,index:{'traj_rel_dir':f'../{folder}/e{index}','meta':{'teacher_state':state}}
        new=[sample('added','search',i) for i in range(30)]+[sample('added','approach',i) for i in range(30,50)]
        old=[sample('first','approach',i) for i in range(600)]+[sample('first','search',i) for i in range(600,900)]+[sample('second','stop',i) for i in range(900,1000)]
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'val.json').write_text(json.dumps(old[:500]+new+old[500:]));result=val_mix(folder,'added')
            mixed=json.loads((Path(folder)/'val_mix.json').read_text())
        self.assertEqual((result['added_frames'],result['earlier_frames']),(50,50));self.assertEqual(len(mixed),100)
        self.assertEqual(sum(s['traj_rel_dir'].startswith('../added/') for s in mixed),50)
        self.assertEqual(result['earlier_by_state'],{'approach':30,'search':15,'stop':5})

    def test_a_share_of_the_draws_goes_to_the_named_source(self):
        try:
            from scripts.train_aerovla_oft import apply_shares
        except ImportError as error:raise unittest.SkipTest(f'training dependencies not installed here: {error}')
        samples=[{'traj_rel_dir':'../old/a'}]*900+[{'traj_rel_dir':'../added/b'}]*100
        weights,frames=apply_shares(samples,[1.]*600+[3.]*300+[1.]*50+[2.]*50,{'added':.5})
        self.assertEqual(frames,{'added':100,'None':900})
        self.assertAlmostEqual(sum(weights[900:])/sum(weights),.5);self.assertAlmostEqual(weights[600]/weights[0],3.);self.assertAlmostEqual(weights[950]/weights[900],2.)
        self.assertRaises(ValueError,apply_shares,samples,[1.]*1000,{'missing':.5});self.assertRaises(ValueError,apply_shares,samples,[1.]*1000,{'added':1.})

    def test_the_frames_the_choice_is_made_on(self):
        try:
            from scripts.train_aerovla_oft import BOOSTS
        except ImportError as error:raise unittest.SkipTest(f'training dependencies not installed here: {error}')
        chosen=BOOSTS['selection'];meta=lambda case,state,step=20:{'case':case,'teacher_state':state,'step':step}
        for case in ('color_first','shape_first','lost'):
            self.assertTrue(chosen(meta(case,'search')));self.assertFalse(chosen(meta(case,'approach')) or chosen(meta(case,'descend')))
        for case in ('swap','query'):
            self.assertTrue(chosen(meta(case,'approach',0)) and chosen(meta(case,'align',7)));self.assertFalse(chosen(meta(case,'approach',8)))
        # Nothing of the earlier data is drawn more often by it.
        for case in ('search_past','land_past_same_color','query_search-query','land_visible','reacquire','search'):self.assertFalse(chosen(meta(case,'search',0)))

    def test_a_layout_swapped_twin_is_not_a_repeat_for_the_merge_and_a_true_repeat_still_is(self):
        from scripts.build_visual_search_dataset import collisions
        record=lambda name,layout,**fields:{'traj_rel_dir':name,'summary':dict({'id':name,'map':'field','seed':7,'target':'blue_pad','task':'land','split':'train',
                                                                             'start_xy':[1.,2.],'layout':layout},**fields)}
        base=record('a','k');twin=record('a-swapped','l',twin_of='a')
        self.assertFalse(any(collisions([base,twin]).values()))
        self.assertTrue(collisions([base,record('b','k',twin_of='a')])['starts'])

    def test_the_dry_run_can_give_its_trace(self):
        scene=SCENES['field'];episode=make_start(CONFIG,OFT,scene,'k','blue_pad','visible','near',5,'trace',task='land',max_ticks=V3C['max_ticks'],spec={'base':'visible','distance_m':[20.,30.]})
        trace=[];plan=rollout(CONFIG,OFT,MapGeometry(scene,'k'),episode,scene['altitude']['ceiling_m'],trace=trace)
        self.assertEqual(len(trace),plan['ticks']);self.assertEqual((trace[0]['x'],trace[0]['y']),tuple(episode['start_xy']));self.assertTrue(trace[0]['visible'])
        self.assertEqual(plan,rollout(CONFIG,OFT,MapGeometry(scene,'k'),episode,scene['altitude']['ceiling_m']))


if __name__=='__main__':unittest.main()
