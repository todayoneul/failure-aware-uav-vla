"""Visual-search inputs, the teacher, the dataset format and the AeroVLA-OFT conventions; no simulator, no weights."""
import inspect
import json
import math
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def state(yaw_deg=0.,position=(0.,0.,-6.),velocity=(0.,0.,0.)):
    half=math.radians(yaw_deg)/2
    return {'position':list(position),'orientation':[0.,0.,math.sin(half),math.cos(half)],'velocity':list(velocity)}


class VisualInputTests(unittest.TestCase):
    def setUp(self):
        from src.visual_search.episodes import load_config
        self.config=load_config()

    def test_policy_inputs_have_no_way_to_carry_target_geometry(self):
        from src.visual_search.episodes import policy_inputs
        self.assertEqual(list(inspect.signature(policy_inputs).parameters),['front','down','instruction','proprio'])
        frame=np.zeros((256,256,3),dtype=np.uint8)
        inputs=policy_inputs(frame,frame,'Find the blue cone.')
        self.assertEqual(sorted(inputs),['down','front','instruction','proprio'])
        self.assertIs(inputs['front'],frame)
        for leaked in ('Fly forward-left and find the target.','The target is 31.5 meters away.','Go to 91.4, -35.4.','It is to your right.'):
            with self.assertRaises(ValueError):policy_inputs(frame,frame,leaked)

    def test_every_instruction_and_prompt_is_free_of_direction_and_coordinates(self):
        from src.visual_search.episodes import instruction_for,DIRECTION_WORDS
        from src.integration.projectairsim_observation_adapter import make_prompt
        from src.aerovla_oft.model import oft_prompt
        for case in self.config['cases']:
            for target in self.config['targets']:
                text=instruction_for(self.config,case,target)
                # Baseline AeroVLA and AeroVLA-OFT are given the same sentence in the same wrapper.
                for prompt in (make_prompt(None,None,'',False,text),oft_prompt(text)):
                    self.assertEqual(prompt,f'<image>\n{text}\nAction: ')
                    self.assertFalse(any(word in prompt for word in DIRECTION_WORDS))
                    self.assertFalse(any(character.isdigit() for character in prompt))
        self.assertEqual(instruction_for(self.config,'A','blue_cone'),'Approach the blue cone.')
        self.assertEqual(instruction_for(self.config,'C','orange_ball'),'Find the orange ball.')

    def test_start_pose_puts_the_target_where_the_case_says(self):
        from src.visual_search.episodes import make_episode,relative_target
        centre=[91.4,-35.4,-6.]
        for case,(low,high) in (('A',(0,8)),('B',(28,40)),('C',(100,180))):
            sides=set()
            for seed in range(12):
                episode=make_episode(self.config,case,'blue_cone',centre,seed)
                bearing,distance,_=relative_target(state(episode['start_yaw_deg'],(*episode['start_xy'],-6.)),centre)
                self.assertAlmostEqual(bearing,episode['target_bearing_deg'],places=4)
                self.assertTrue(low-1e-6<=abs(bearing)<=high+1e-6);self.assertTrue(28<=distance<=45)
                sides.add(bearing>0)
                self.assertEqual(episode,make_episode(self.config,case,'blue_cone',centre,seed))
            if case!='A':self.assertEqual(sides,{True,False})
        self.assertIn('kick',make_episode(self.config,'D','blue_cone',centre,0))
        # Language test: one shared start, the cone to the left and the ball to the right, both inside the front view.
        cone=make_episode(self.config,'L','blue_cone',centre,0);ball=make_episode(self.config,'L','orange_ball',[91.2,32.1,-6.],0)
        self.assertEqual(cone['start_xy'],ball['start_xy'])
        self.assertTrue(-45<cone['target_bearing_deg']<-25);self.assertTrue(25<ball['target_bearing_deg']<45)
        self.assertNotEqual(cone['instruction'],ball['instruction'])


class TeacherTests(unittest.TestCase):
    def setUp(self):
        from src.visual_search.episodes import load_config,Teacher
        from src.aerovla_oft.spec import load_config as load_oft
        self.config=load_config();self.oft=load_oft();self.teacher=Teacher(self.config,self.oft)

    def test_seen_target_is_turned_to_and_approached_and_arrival_stops(self):
        from src.aerovla_oft.spec import is_stop
        action,mode=self.teacher.act(35.,30.,True)
        self.assertEqual(mode,'align');self.assertAlmostEqual(action[2],.21);self.assertEqual(action[0],0.)
        action,mode=self.teacher.act(-10.,30.,True)
        self.assertEqual(mode,'approach');self.assertLess(action[2],0);self.assertAlmostEqual(action[0],1-10/30)
        action,mode=self.teacher.act(0.,13.,True)
        self.assertLess(action[0],.3)
        action,mode=self.teacher.act(0.,11.9,True)
        self.assertEqual((action,mode),([0.,0.,0.],'stop'));self.assertTrue(is_stop(action,self.oft))

    def test_unseen_target_is_swept_one_way_then_a_climb_is_tried(self):
        actions=[self.teacher.act(170.,30.,False) for _ in range(40)]
        modes=[mode for _,mode in actions]
        self.assertEqual(set(modes[:30]),{'search'})
        self.assertTrue(all(action[2]==.21 and action[0]==0. for action,mode in actions if mode=='search'))
        climbs=[action for action,mode in actions if mode=='climb']
        self.assertTrue(climbs);self.assertAlmostEqual(-sum(action[1] for action in climbs),3.)
        # Left of the vehicle or right of it, an unseen target is searched the same way: the student cannot know.
        self.assertEqual(self.teacher.act(-170.,30.,False)[0][2],.21)


class OFTConventionTests(unittest.TestCase):
    def setUp(self):
        from src.aerovla_oft.spec import load_config
        self.config=load_config()

    def test_actions_scale_to_unit_range_and_back_with_clipping(self):
        from src.aerovla_oft.spec import normalize_action,denormalize_action
        self.assertEqual(normalize_action([0.,0.,0.],self.config),[-1.,0.,0.])
        self.assertEqual(normalize_action([1.,.5,.21],self.config),[1.,1.,1.])
        self.assertEqual(normalize_action([9.,-9.,-9.],self.config),[1.,-1.,-1.])
        for action in ([.37,-.2,.1],[0.,.5,-.21]):
            np.testing.assert_allclose(denormalize_action(normalize_action(action,self.config),self.config),action,atol=1e-9)
        np.testing.assert_allclose(denormalize_action([5.,-5.,0.],self.config),[1.,-.5,0.])

    def test_chunk_labels_keep_order_and_repeat_the_last_action_past_the_end(self):
        from src.aerovla_oft.spec import chunk_labels
        actions=[[float(i),0.,0.] for i in range(5)]
        chunks=chunk_labels(actions,4)
        self.assertEqual([step[0] for step in chunks[0]],[0.,1.,2.,3.])
        self.assertEqual([step[0] for step in chunks[3]],[3.,4.,4.,4.])
        self.assertEqual([step[0] for step in chunks[4]],[4.,4.,4.,4.])
        self.assertEqual(len(chunks),5);self.assertEqual(chunk_labels([],4),[])

    def test_config_is_a_short_chunk_executed_one_action_at_a_time(self):
        from src.aerovla_oft.spec import load_config,is_stop
        self.assertEqual((self.config['chunk_size'],self.config['execute_horizon']),(4,1))
        self.assertFalse(self.config['film'])
        self.assertTrue(is_stop([.05,.0,.01],self.config));self.assertFalse(is_stop([.3,0.,0.],self.config))
        self.assertFalse(is_stop([0.,0.,.1],self.config))
        broken=json.loads((ROOT/'configs/aerovla_oft.json').read_text());broken['execute_horizon']=5
        path=ROOT/'outputs/_test_oft_config.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(broken))
        try:
            with self.assertRaises(ValueError):load_config(path)
        finally:path.unlink()

    def test_proprio_is_vehicle_motion_only(self):
        from src.aerovla_oft.spec import load_config,proprio_vector,PROPRIO_FIELDS
        self.assertFalse(any(word in field for field in PROPRIO_FIELDS for word in ('target','bearing','distance','visible','position')))
        vector=proprio_vector(state(90.,(50.,-20.,-6.),(0.,2.,.5)),0.,self.config,.25)
        # Heading east at 2 m/s: all of it is forward speed; where the vehicle is does not enter.
        np.testing.assert_allclose(vector,[1.,0.,.5,6/30,.5],atol=1e-9)
        self.assertEqual(vector,proprio_vector(state(90.,(-5.,70.,-6.),(0.,2.,.5)),0.,self.config,.25))
        leaking=json.loads((ROOT/'configs/aerovla_oft.json').read_text());leaking['proprio']['fields'][0]='target_bearing'
        path=ROOT/'outputs/_test_oft_config.json';path.write_text(json.dumps(leaking))
        try:
            with self.assertRaises(ValueError):load_config(path)
        finally:path.unlink()

    def test_head_maps_action_token_states_to_a_chunk_and_survives_a_reload(self):
        import torch
        from src.aerovla_oft.model import ActionHead,ProprioProjector,ACTION_DIM
        torch.manual_seed(0)
        head=ActionHead(llm_dim=32,hidden_dim=16,blocks=2,chunk_size=4)
        states=torch.randn(2,4*ACTION_DIM,32);chunk=head(states)
        self.assertEqual(tuple(chunk.shape),(2,4,3))
        # Chunk step k is computed from the k-th group of action-token states only.
        changed=states.clone();changed[:,ACTION_DIM:2*ACTION_DIM]+=1.
        different=(head(changed)-chunk).abs().sum(dim=(0,2))
        self.assertGreater(different[1].item(),0);self.assertEqual(different[[0,2,3]].sum().item(),0)
        copy=ActionHead(32,16,2,4);copy.load_state_dict(head.state_dict())
        torch.testing.assert_close(copy(states),chunk)
        self.assertEqual(tuple(ProprioProjector(5,32)(torch.zeros(2,5)).shape),(2,32))


class DatasetTests(unittest.TestCase):
    def record(self,identifier,steps=8,perturbed=()):
        return {'traj_rel_dir':f'blocks/{identifier}','summary':{'id':identifier,'case':'C','target':'blue_cone','reason':'teacher_stop',
                                                              'instruction':'Find the blue cone.'},
                'steps':[{'step':i,'img_name':f'{i:06d}.png','action':[99.,99.,99.] if i in perturbed else [float(i),0.,0.],
                          'teacher_action':[float(i),0.,0.],'perturbed':i in perturbed,'proprio':[0.]*5,'front_seen':i>2,'front_in_fov':i>1,
                          'down_seen':False,'bearing_deg':10.,'distance_m':30.,'teacher_state':'search' if i<=2 else 'approach'} for i in range(steps)]}

    def test_samples_follow_the_aerovla_schema_with_a_future_chunk(self):
        from scripts.build_visual_search_dataset import episode_samples
        samples=episode_samples(self.record('C-blue_cone-0'),4)
        self.assertEqual(len(samples),8);first=samples[0]
        for key in ('traj_rel_dir','img_name','instruction','label','is_last_step','is_penultimate'):self.assertIn(key,first)
        self.assertEqual(first['instruction'],'<image>\nFind the blue cone.')
        self.assertEqual(first['label'],{'fwd':0.,'down':0.,'yaw':0.});self.assertEqual([s[0] for s in first['chunk']],[0.,1.,2.,3.])
        self.assertEqual([s[0] for s in samples[-1]['chunk']],[7.]*4)
        self.assertTrue(samples[-1]['is_last_step']);self.assertTrue(samples[-2]['is_penultimate'])
        self.assertEqual(first['meta']['teacher_state'],'search');self.assertFalse(first['meta']['target_visible'])

    def test_forced_turns_never_appear_in_a_label(self):
        from scripts.build_visual_search_dataset import episode_samples
        samples=episode_samples(self.record('D-blue_cone-0',steps=12,perturbed=(5,6)),4)
        self.assertEqual([sample['meta']['step'] for sample in samples],[0,1,7,8,9,10,11])
        self.assertFalse(any(99. in step for sample in samples for step in sample['chunk']))

    def test_split_is_by_episode(self):
        from src.aerovla_oft.spec import split_episodes
        identifiers=[f'ep{i}' for i in range(10) for _ in range(30)]
        train,validation=split_episodes(identifiers,.2,seed=0)
        self.assertEqual((len(train),len(validation)),(8,2));self.assertFalse(set(train)&set(validation))
        self.assertEqual(split_episodes(identifiers,.2,seed=0),(train,validation))


class SummaryTests(unittest.TestCase):
    def test_search_run_is_scored_by_where_it_stopped_and_what_it_found(self):
        from scripts.visual_search import summarise
        from src.visual_search.episodes import load_config
        config=load_config()
        def step(i,bearing,distance,seen,yaw):
            return {'epoch':i*.5,'bearing_deg':bearing,'distance_m':distance,'front_seen':seen,'action':[0.,0.,yaw],'height_m':6.,
                    'normalised':[-1.,0.,yaw/.21]}
        steps=[step(0,120.,30.,False,.21),step(1,100.,30.,False,.21),step(2,10.,30.,True,.05),step(3,0.,11.,True,0.)]
        result=summarise({'id':'C-x-0'},steps,config,'model_stop',True)
        self.assertTrue(result['success']);self.assertEqual(result['acquired_step'],2);self.assertTrue(result['first_turn_toward_target'])
        self.assertFalse(result['initially_seen']);self.assertTrue(result['centred'])
        self.assertAlmostEqual(result['yaw_swept_deg'],math.degrees(.47),places=3)
        self.assertGreater(result['action_jump'],0)
        self.assertFalse(summarise({'id':'C-x-0'},steps[:2],config,'max_steps',False)['success'])
        self.assertFalse(summarise({'id':'C-x-0'},steps,config,'collision',True)['success'])

    def test_three_policies_are_selectable(self):
        text=(ROOT/'scripts/visual_search.py').read_text(encoding='utf-8')
        self.assertIn("choices=('baseline','teacher','oft')",text)
        launcher=(ROOT/'scripts/run_visual_search.ps1').read_bytes().decode()
        self.assertIn("[ValidateSet('baseline','teacher','oft')][string]$Policy",launcher)


class ViewerTests(unittest.TestCase):
    def live(self,front,down,chunk=True):
        import hashlib
        digest={name:hashlib.sha256(frame.tobytes()).hexdigest() for name,frame in (('front',front),('down',down))}
        step={'step':3,'front_seen':True,'down_seen':False,'front_pixel':[100.,120.],'down_pixel':None,'distance_m':31.2,'bearing_deg':-12.,
              'action':[.5,0.,.1],'input_sha256':digest,'inference_ms':180.}
        if chunk:step['chunk']=[[.5,0.,.1],[.6,0.,.05],[.7,0.,0.],[.7,0.,0.]]
        return {'mode':'VISUAL SEARCH','model':'oft','episode':{'id':'C-blue_cone-0','instruction':'Find the blue cone.'},'step':step,
                'success_radius_m':15.,'failure':'NORMAL','raw_sha256':dict(digest)}

    def test_marker_is_drawn_on_a_copy_and_the_model_frame_is_verified_unchanged(self):
        from scripts.visual_search_viewer import marked,render,input_matches
        front=np.random.default_rng(1).integers(0,256,(256,256,3),dtype=np.uint8);down=front[::-1].copy()
        before=front.copy();shown=marked(front,[100.,120.],True)
        np.testing.assert_array_equal(front,before);self.assertFalse(np.array_equal(shown,front))
        np.testing.assert_array_equal(marked(front,None,False),front)
        live=self.live(front,down)
        self.assertTrue(input_matches(live,front,down));self.assertIsNone(input_matches(live,None,down))
        # A frame that carries a marker is not what the policy was given.
        self.assertFalse(input_matches(live,shown,down))
        for chunk in (True,False):
            canvas=render(self.live(front,down,chunk),front,down,{'episode':'C-blue_cone-0','reason':'model_stop','stopped':True,'final_distance_m':10.4})
            self.assertEqual(canvas.shape,(760,1180,3))
        np.testing.assert_array_equal(front,before)


if __name__=='__main__':unittest.main()
