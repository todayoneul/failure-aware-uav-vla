"""Phase B, stage 1 (generic landing): the teacher, the plans, the recording and the fixed rules. The flights are the
canonical episode loop over a fake simulator; no model is loaded."""
import argparse
import asyncio
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def setting():
    from src.landing.surface import SurfaceSet
    from src.visual_search.episodes import load_config
    from src.visual_search.maps import load_map
    from src.aerovla_oft.spec import load_config as load_oft
    scene=load_map('field');return load_config(),load_oft(),scene,SurfaceSet(scene,'a')


def view(**changes):
    base={'bearing_deg':5.,'distance_m':30.,'visible':True,'below':False,'ahead_m':60.,'height_m':6.,'over':False,'surface_m':8.,'task':'land','landed':False}
    base.update(changes);return base


class TeacherTests(unittest.TestCase):
    def teachers(self,target):
        from src.landing.teacher import SurfaceTeacher
        from src.visual_search.generalization import SearchTeacher
        config,oft,scene,surfaces=setting()
        return SurfaceTeacher(config,oft,surface=surfaces[target]),SearchTeacher(config,oft)

    def test_on_a_pad_from_the_earlier_heights_the_labels_are_the_canonical_ones(self):
        new,old=self.teachers('blue_pad')
        for height in (5.,5.5,6.,7.5,9.,12.,14.):
            for distance in (.5,1.5,3.,6.,12.,20.,30.,44.):
                for bearing in (-120.,-30.,-5.,0.,10.,35.,170.):
                    for seen in ((True,False,False),(False,True,True),(False,False,False),(True,True,True)):
                        for task in ('land','approach'):
                            item=view(height_m=height,distance_m=distance,bearing_deg=bearing,visible=seen[0],below=seen[1],over=seen[2],surface_m=2.5,task=task)
                            self.assertEqual(new.act(dict(item)),old.act(dict(item)),item)
        # Lower than the earlier data ever started, near the pad: up to the height the canonical descent slows at, then as before.
        action,state=new.act(view(height_m=3.,distance_m=20.,surface_m=2.5));self.assertEqual(state,'climb');self.assertLess(action[1],0.);self.assertGreater(action[0],0.)
        self.assertEqual(new.act(view(height_m=3.,distance_m=30.,surface_m=2.5)),old.act(view(height_m=3.,distance_m=30.,surface_m=2.5)))

    def test_toward_a_top_above_it_the_teacher_climbs_and_waits_until_it_can_fly_over(self):
        new,old=self.teachers('blue_cube')
        far=view(distance_m=30.);self.assertEqual(new.act(dict(far)),old.act(dict(far)))
        action,state=new.act(view(distance_m=20.));self.assertEqual(state,'climb');self.assertEqual(action[1],-.5);self.assertGreater(action[0],0.)
        action,state=new.act(view(distance_m=10.));self.assertEqual((state,action[0],action[1]),('climb',0.,-.5))
        # High enough (3.5 m over an 8 m top): on, slowly; the canonical teacher would fly the last metres at full speed.
        action,state=new.act(view(distance_m=10.,height_m=11.5));self.assertIn(state,('final','approach'));self.assertAlmostEqual(action[0],.5);self.assertEqual(action[1],0.)
        self.assertGreater(old.act(view(distance_m=10.,height_m=11.5))[0][0],.8)
        # Below the ceiling only, and never for an approach.
        self.assertNotEqual(new.act(view(distance_m=20.,height_m=13.9,surface_m=12.))[1],'climb')
        approach=view(distance_m=20.,task='approach');self.assertEqual(new.act(dict(approach)),old.act(dict(approach)))

    def test_over_a_small_top_the_descent_starts_over_its_middle(self):
        new,old=self.teachers('blue_cube')
        over=lambda distance,height=11.5:new.act(view(distance_m=distance,height_m=height,visible=False,below=True,over=True))
        self.assertEqual(over(.9)[1],'descend');self.assertNotEqual(over(1.5)[1],'descend');self.assertNotEqual(over(3.4)[1],'descend')
        # The canonical radii are a pad's: 3.4 m from the middle of a cube is 0.2 m from the edge of its usable region.
        self.assertEqual(old.act(view(distance_m=3.4,height_m=10.5,visible=False,below=True,over=True))[1],'descend')
        # Low over it and near the middle the descent goes on (a vehicle drifting a metre is not told to turn back).
        self.assertEqual(over(1.8,10.9)[1],'descend');self.assertNotEqual(over(2.2,10.9)[1],'descend')
        self.assertEqual(over(.5,11.5)[0],[0.,.5,0.]);self.assertEqual(over(.5,10.)[0],[0.,.25,0.])
        self.assertEqual(new.act(view(distance_m=.5,height_m=8.,landed=True,over=True)),([0.,0.,0.],'landed'))

    def test_while_the_object_is_not_in_view_nothing_the_teacher_does_depends_on_where_it_is(self):
        for target in ('blue_cube','blue_pad','green_cylinder'):
            new,old=self.teachers(target);answers=set()
            for bearing in (-170.,-60.,0.,45.,120.):
                for distance in (5.,20.,44.,90.):
                    for task in ('land','approach'):
                        item=view(bearing_deg=bearing,distance_m=distance,visible=False,below=False,over=False,task=task)
                        answer=new.act(dict(item));self.assertEqual(answer,old.act(dict(item)));answers.add((tuple(answer[0]),answer[1]))
            self.assertEqual(len(answers),1)

    def test_the_dry_run_lands_near_the_middle_of_a_cube_and_of_a_cylinder(self):
        from src.landing.teacher import rollout
        from src.visual_search.maps import object_position
        config,oft,scene,surfaces=setting()
        for target,height in (('blue_cube',6.),('blue_cube',3.),('green_cylinder',6.),('blue_cylinder',12.),('blue_pad',6.)):
            centre=object_position(scene,'a',target)
            episode={'start_xy':[centre[0]-30.,centre[1]+5.],'start_yaw_deg':-40.,'start_height_m':height,'target':target,'task':'land','strategy':'right','max_ticks':320}
            result=rollout(config,oft,scene,'a',episode,surfaces[target])
            self.assertEqual((result['reason'],result['landed']),('teacher_stop',True),target)
            self.assertTrue(surfaces[target].inside([result['final_distance_m'],0.]),target)
            if surfaces[target].usable_reach_m<6.:self.assertLess(result['final_distance_m'],1.)
            twin=dict(episode,task='approach');result=rollout(config,oft,scene,'a',twin,surfaces[target])
            self.assertEqual((result['reason'],result['landed']),('teacher_stop',False));self.assertLess(abs(result['final_distance_m']-12.),1.)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.visual_search import generic_landing as stage1
        cls.stage=stage1.load();cls.sets={name:json.loads(path.read_text(encoding='utf-8'))['episodes'] for name,path in stage1.FILES.items()}
        cls.plan=json.loads(stage1.TRAIN_PLAN.read_text(encoding='utf-8'))

    def count(self,episodes,**fields):
        return sum(all(episode.get(key)==value for key,value in fields.items()) for episode in episodes)

    def test_the_recorded_episodes_are_what_the_configuration_says(self):
        train=self.plan['train'];training=self.stage['training']
        self.assertEqual(len(train),3*(training['land_per_kind']+training['approach_twins_per_kind']));self.assertEqual(len(train),210)
        for kind,targets in self.stage['targets'].items():
            self.assertEqual(self.count(train,surface_kind=kind,task='land'),54);self.assertEqual(self.count(train,surface_kind=kind,task='approach'),16)
            for target in targets:self.assertEqual(self.count(train,target=target,task='land'),27)
        for band,(low,high) in self.stage['bands_m'].items():
            mine=[e for e in train if e['band']==band and e['task']=='land'];self.assertEqual(len(mine),54)
            self.assertTrue(all(low-.01<=e['start_distance_m']<=high+.01 for e in mine))
        self.assertEqual(self.count(train,initially_visible=True,task='land'),81)
        low,high=self.stage['height_m'];self.assertTrue(all(low<=e['start_height_m']<=high for e in train))
        # Most cube and cylinder landings start lower than the top and so hold a climb; every twin shares its landing's start.
        tall=[e for e in train if e['task']=='land' and e['surface_kind']!='pad'];self.assertGreater(sum(e['plan']['climbed_m']>=1. for e in tall),.6*len(tall))
        by={e['id']:e for e in train}
        for twin in (e for e in train if e['task']=='approach'):
            first=by[twin['twin_of']];self.assertEqual((twin['start_xy'],twin['start_yaw_deg'],twin['start_height_m'],twin['target']),(first['start_xy'],first['start_yaw_deg'],first['start_height_m'],first['target']))
            self.assertTrue(twin['instruction'].startswith('Approach the '));self.assertTrue(first['instruction'].endswith(' and land on it.'))
        self.assertEqual(len(self.plan['val']),24)

    def test_the_check_sets_are_what_the_configuration_says(self):
        gates=self.stage['validation']['gates'];validation=self.sets['validation'];pilot=self.sets['pilot']
        self.assertEqual((len(validation),gates['of'],self.count(validation,task='land'),gates['land_of']),(36,36,30,30))
        for kind in self.stage['targets']:
            self.assertEqual((self.count(validation,surface_kind=kind,task='land'),self.count(validation,surface_kind=kind,task='approach')),(10,2))
            self.assertEqual((self.count(pilot,surface_kind=kind,task='land'),self.count(pilot,surface_kind=kind,task='approach')),(4,1))
            self.assertEqual(self.count(validation,surface_kind=kind,task='land',initially_visible=True),5)
        self.assertEqual(len(pilot),15);self.assertEqual(len(self.sets['regression']),36)
        self.assertTrue(all(e['start_distance_m']<=44.01 for name in self.sets for e in self.sets[name]))
        # The numbers of the gates are the plan's: 30 of 36; landings 80 %; pads 90 %, cubes and cylinders 75 % (of ten: 9, 8, 8).
        self.assertEqual((gates['success_min'],gates['land_system_success_min'],gates['pad_land_min'],gates['cube_land_min'],gates['cylinder_land_min']),(30,24,9,8,8))
        self.assertEqual((gates['wrong_target_max'],gates['collisions_max'],gates['approach_touchdowns_max']),(2,1,0))

    def test_the_plans_are_reproducible_separate_and_in_the_right_layouts(self):
        import scripts.generic_landing as tool
        from src.visual_search.episodes import load_config
        from src.aerovla_oft.spec import load_config as load_oft
        self.assertEqual(tool.check(load_config(),load_oft(),self.stage),[])
        recorded={e['layout'] for split in ('train','val') for e in self.plan[split]};flown={e['layout'] for name in self.sets for e in self.sets[name]}
        self.assertTrue(recorded<=set(self.stage['layouts']['train']));self.assertTrue(flown<=set(self.stage['layouts']['evaluation']));self.assertFalse(recorded&flown)
        # Neither the canonical test's layouts nor the sealed test's are touched, and no canonical test start is among the starts.
        self.assertFalse((recorded|flown)&{'e','g','h','i','j'})
        test={(e['map'],tuple(e['start_xy'])) for e in json.loads((ROOT/'configs/gen_v3_canonical_44m_test.json').read_text(encoding='utf-8'))['episodes']}
        mine={(e['map'],tuple(e['start_xy'])) for name in self.sets for e in self.sets[name]}|{(e['map'],tuple(e['start_xy'])) for s in ('train','val') for e in self.plan[s]}
        self.assertFalse(test&mine)

    def test_the_training_schedule_is_fixed_and_keeps_the_baselines_architecture(self):
        tune=self.stage['fine_tune'];base=json.loads((ROOT/'outputs/generalization/grounding_film_frozen.json').read_text(encoding='utf-8'))
        self.assertEqual(self.stage['start_checkpoint'],'outputs/aerovla_oft/checkpoints/grounding_film');self.assertIn(self.stage['start_checkpoint'],base['paths'])
        self.assertEqual((tune['updates'],tune['candidates']),(2500,[1000,1500,2000,2500]))
        self.assertTrue(all(update%tune['eval_every']==0 for update in tune['candidates']));self.assertTrue(.4<=tune['share']<=.5)
        source=(ROOT/'scripts/train_aerovla_oft.py').read_text(encoding='utf-8');self.assertIn("--save-at",source);self.assertIn("output/f'update-{name}'",source)
        self.assertEqual(len(self.stage['pilot']['order']),6);self.assertTrue(self.stage['pilot']['order'][0].startswith('landings on cubes and cylinders'))


def flight(kind,task,success,wrong=False,hit=None,landed=False,inside=True,target=None,identifier=None):
    """One flown episode as scripts/surface_flights.py records it, reduced to what the gates read."""
    target=target or {'pad':'blue_pad','cube':'blue_cube','cylinder':'green_cylinder'}[kind]
    return {'id':identifier or f'{kind}-{task}-{id(object())}','flown':True,'episode':{'task':task,'target':target,'surface_kind':kind},
            'summary':{'wrong_target':wrong,'hit':hit,'reason':'collision' if hit else 'landed' if success and task=='land' else 'model_stop','landed':landed,
                       'stopped':True,'final_distance_m':3.,'steps':80},
            'surface_landing':{'success':success,'correct_object':success or not wrong,'inside_region':inside,'contact_from_above':landed or (success and task=='land'),
                               'landed_on':target if success and task=='land' else None,'collision':hit}}


class GateTests(unittest.TestCase):
    def setUp(self):
        import scripts.generic_landing as tool
        from src.visual_search import generic_landing as stage1
        self.tool=tool;self.stage=stage1.load()

    def validation(self,pad=10,cube=10,cylinder=10,approach=6,**changes):
        records=[]
        for kind,good in (('pad',pad),('cube',cube),('cylinder',cylinder)):
            records+=[flight(kind,'land',index<good) for index in range(10)]+[flight(kind,'approach',index<approach//3) for index in range(2)]
        return records

    def test_the_validation_gate_counts_what_the_plan_asks_for(self):
        numbers,checks=self.tool.judge_validation(self.stage,'validation',self.validation())
        self.assertTrue(all(row[3] for row in checks));self.assertEqual((numbers['success'],numbers['land_success'],numbers['kinds']['cube']['land_success']),(36,30,10))
        # Exactly at the limits: pads 9, cubes 8, cylinders 8 make 25 landings and 31 missions.
        numbers,checks=self.tool.judge_validation(self.stage,'validation',self.validation(9,8,8));self.assertTrue(all(row[3] for row in checks))
        for short in ({'pad':8},{'cube':7},{'cylinder':7}):
            numbers,checks=self.tool.judge_validation(self.stage,'validation',self.validation(**{**{'pad':10,'cube':10,'cylinder':10},**short}))
            self.assertEqual([row[0] for row in checks if not row[3]],[f'{next(iter(short))} landings'])
        # Thirty missions are needed whatever the kinds say, and an approach that touches down fails the gate by itself.
        records=self.validation(9,8,8,approach=0);numbers,checks=self.tool.judge_validation(self.stage,'validation',records)
        self.assertEqual([row[0] for row in checks if not row[3]],['mission success'])
        records=self.validation();records[10]=flight('pad','approach',False,landed=True)
        self.assertIn('touchdowns in an approach',[row[0] for row in self.tool.judge_validation(self.stage,'validation',records)[1] if not row[3]])
        records=self.validation();records[0]=flight('pad','land',False,wrong=True);records[1]=flight('pad','land',False,wrong=True);records[12]=flight('cube','land',False,wrong=True)
        self.assertIn('flights at a wrong object',[row[0] for row in self.tool.judge_validation(self.stage,'validation',records)[1] if not row[3]])
        records=self.validation();records[0]=flight('pad','land',False,hit='Hedge');records[12]=flight('cube','land',False,hit='BlueCube')
        self.assertIn('collisions',[row[0] for row in self.tool.judge_validation(self.stage,'validation',records)[1] if not row[3]])

    def test_the_candidate_is_chosen_by_the_fixed_order_and_must_meet_the_pilots_lines(self):
        def run(cube,cylinder,pad,wrong=0,hit=0):
            records=[flight('cube','land',i<cube) for i in range(4)]+[flight('cylinder','land',i<cylinder) for i in range(4)]+[flight('pad','land',i<pad) for i in range(4)]
            records+=[flight(kind,'approach',True) for kind in ('pad','cube','cylinder')]
            for index in range(wrong):records[index]['summary']['wrong_target']=True
            for index in range(hit):records[4+index]['summary']['hit']='Wall'
            return records
        runs={'baseline':run(0,0,4),'1000':run(3,2,4),'1500':run(3,3,4,wrong=1),'2000':run(3,3,3),'2500':run(3,3,3)}
        rows,order,checks=self.tool.judge_pilot(self.stage,runs,{'1000':.07,'1500':.06,'2000':.065,'2500':.06})
        # More landings on cubes and cylinders first; then fewer wrong objects; then pads; then collisions; then the validation loss; then the earlier update.
        self.assertEqual(order,['2500','2000','1500','1000']);self.assertNotIn('baseline',order);self.assertTrue(all(row[3] for row in checks))
        rows,order,checks=self.tool.judge_pilot(self.stage,{'1000':run(2,2,4),'2000':run(2,1,4)},{})
        self.assertEqual(order[0],'1000');self.assertEqual([row[0] for row in checks if not row[3]],['landings on cubes and cylinders'])

    def test_the_regression_compares_with_the_baseline_and_allows_the_spread_of_two_flights(self):
        def run(success,wrong=0,hit=0):
            records=[flight('pad','land',index<success) for index in range(27)]+[flight('pad','approach',True) for _ in range(9)]
            for index in range(wrong):records[index]['summary']['wrong_target']=True
            for index in range(hit):records[index]['summary']['hit']='Hedge'
            return records
        self.assertTrue(all(row[3] for row in self.tool.judge_regression(self.stage,run(24),run(26))[1]))
        self.assertFalse(all(row[3] for row in self.tool.judge_regression(self.stage,run(23),run(26))[1]))
        self.assertFalse(all(row[3] for row in self.tool.judge_regression(self.stage,run(26,wrong=3),run(26,wrong=1))[1]))
        self.assertTrue(all(row[3] for row in self.tool.judge_regression(self.stage,run(26,wrong=2,hit=1),run(26,wrong=1))[1]))


class RecordingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):asyncio.get_running_loop().slow_callback_duration=60.

    async def test_the_teacher_is_recorded_landing_on_a_cube_and_a_cylinder_and_approaching_one(self):
        import projectairsim
        import scripts.surface_flights as worker
        import scripts.visual_search as canonical
        from scripts.build_visual_search_dataset import build_records,load_records
        from src.aerovla_oft.spec import load_config as load_oft
        from src.visual_search import generic_landing as stage1
        from src.visual_search.generalization import SearchTeacher
        from src.visual_search.maps import load_map
        from tests.fake_airsim import FakeSim,FakeClient,world_class,drone_class
        plan=json.loads(stage1.TRAIN_PLAN.read_text(encoding='utf-8'))['train']
        pick=lambda **fields:next(e for e in plan if all(e.get(k)==v for k,v in fields.items()))
        cube=pick(surface_kind='cube',task='land',initially_visible=True);cylinder=pick(surface_kind='cylinder',task='land',initially_visible=False)
        twin=next(e for e in plan if e['task']=='approach' and e['surface_kind']=='cube');chosen=[cube,cylinder,twin]
        async def no_wait(_):return None
        clock=types.SimpleNamespace(sleep=no_wait,wait_for=asyncio.wait_for)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'plan.json';file.write_text(json.dumps({'train':chosen}));data=root/'data'
            for episode in chosen:
                sim=FakeSim(load_map(episode['map']),episode['layout'])
                arguments=argparse.Namespace(host='fake',ports=[1,2],plan=str(file),set='train',output=str(root/'run'),policy='teacher',checkpoint=None,model_name=None,
                                             finalizer='off',record=str(data),only=[episode['id']],skip=0,limit=0,again=False,real_time_minimum=0.)
                with patch.object(projectairsim,'ProjectAirSimClient',lambda **_:FakeClient(sim)),patch.object(canonical,'World',world_class(sim)), \
                     patch.object(canonical,'Drone',drone_class(sim)),patch.object(canonical,'matched_camera_images',sim.camera), \
                     patch.object(canonical,'asyncio',clock),patch('builtins.print'):
                    with self.assertRaises(SystemExit) as ended:await worker.fly(arguments)
                self.assertEqual(ended.exception.code,0,episode['id']);self.assertNotIn('land_async',sim.calls)
            # The loop is handed its own teacher class back.
            self.assertIs(canonical.SearchTeacher,SearchTeacher)
            records=load_records(data);self.assertEqual({r['summary']['id'] for r in records},{e['id'] for e in chosen});self.assertFalse((data/'rejected').exists())
            by={r['summary']['id']:r for r in records}
            for episode in (cube,cylinder):
                record=by[episode['id']];summary=record['summary'];states=[s['teacher_state'] for s in record['steps']]
                self.assertEqual((summary['reason'],summary['landed_on'],summary['land_success']),('teacher_stop',episode['target'],True))
                self.assertTrue(summary['surface_landing']['success']);self.assertLess(summary['surface_landing']['horizontal_error_m'],1.5)
                self.assertIn('descend',states);self.assertEqual(states[-1],'landed');self.assertEqual(record['traj_rel_dir'],f'{episode["map"]}/{episode["id"]}')
                if episode['start_height_m']<episode['plan']['climbed_m']+episode['start_height_m']-.5:self.assertIn('climb',states)
                frames=sorted((data/record['traj_rel_dir']/'frontcamera').glob('*.png'));self.assertEqual(len(frames),len(record['steps']))
            approach=by[twin['id']]['summary'];self.assertEqual((approach['reason'],approach['landed'],approach['success']),('teacher_stop',False,True))
            # The earlier builder takes them as they are, and every label is an action the head can give.
            built=build_records(records,load_oft(),.1,0)
        summary=next(item for item in (built if isinstance(built,tuple) else (built,)) if isinstance(item,dict) and 'labels_outside_action_bounds' in item)
        self.assertEqual((summary['labels_outside_action_bounds'],summary['dropped_episodes'],summary['episodes']),(0,{},3))


if __name__=='__main__':unittest.main()
