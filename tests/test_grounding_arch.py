"""Language-vision grounding architectures: the pilot set every model flies twice, the lines an architecture has to hold,
and the two modules themselves (identity at the start, what they may read, what is trained)."""
import json
import math
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.visual_search.episodes import load_config
from src.visual_search.maps import load_map
from src.visual_search import canonical,gen_v3c,grounding_arch
from src.aerovla_oft.spec import load_config as load_oft_config

CONFIG=load_config();OFT=load_oft_config();ARCH=CONFIG['grounding_architecture'];V3C=CONFIG['gen_v3c']
SCENES={name:load_map(name) for name in grounding_arch.SCENES}


class PilotSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot=json.loads(grounding_arch.FILES['pilot'].read_text(encoding='utf-8'));cls.episodes=cls.pilot['episodes']

    def test_the_file_is_reproduced_exactly_by_its_generator(self):
        self.assertEqual(grounding_arch.build_pilot(CONFIG,OFT),self.episodes)

    def test_it_is_sixteen_starts_of_the_four_groups_each_flown_twice(self):
        self.assertEqual(len(self.episodes),16);self.assertEqual(self.pilot['repeats'],2);self.assertEqual(ARCH['pilot']['repeats'],2)
        self.assertEqual([e['role'] for e in self.episodes],['same_color']*6+['same_shape']*3+['first_view']*4+['lost']*3)
        self.assertEqual(len({e['id'] for e in self.episodes}),16);low,high=ARCH['seeds']['pilot']
        for e in self.episodes:
            self.assertTrue(low<=e['seed']<=high);self.assertLessEqual(e['start_distance_m'],V3C['max_start_m'])
            self.assertIn(e['layout'],ARCH['layouts'][e['map']]);self.assertEqual(e['relation'],gen_v3c.relation_of(SCENES[e['map']],e['target'],e['distractor']))
            self.assertEqual(e['task'],'land' if e['target'] in ('blue_pad','red_pad') else 'approach')
        self.assertEqual({e['band'] for e in self.episodes},set(V3C['bands_m']))
        self.assertTrue({('blue_pad','blue_cube'),('blue_pad','blue_cone'),('red_pad','red_cube'),('blue_pad','red_pad'),('blue_cube','red_cube')}<={(e['target'],e['distractor']) for e in self.episodes})
        self.assertEqual(sum(e['relation']=='same_color' for e in self.episodes),11);self.assertEqual(sum(e['relation']=='same_shape' for e in self.episodes),5)

    def test_the_related_object_is_where_the_group_says(self):
        zone=ARCH['pilot']['hard_zone'];view=CONFIG['generalization']['fov_half_deg']
        for e in self.episodes:
            if e['role']=='lost':
                self.assertIn('kick',e);self.assertEqual(e['plan']['acquired_tick'],0);self.assertEqual(e['kick']['degrees']>0,e['where']>0)
                self.assertLessEqual(abs(e['lost']['distractor_bearing_deg']),V3C['hard_negative']['lost']['centred_deg']+1.);continue
            self.assertNotIn('kick',e);self.assertGreater(e['plan']['acquired_tick'],0)
            half=math.degrees(math.atan(SCENES[e['map']]['objects'][e['target']]['size_m'][0]/2/e['start_distance_m']))
            self.assertGreater(abs(e['target_bearing_deg'])-half,view)
            seen=next(item for item in e['others_in_view'] if item['name']==e['distractor'])
            if e['where']=='right':self.assertTrue(zone['right_deg'][0]<=seen['bearing_deg']<=zone['right_deg'][1] and seen['distance_m']<=zone['within_m'],e['id'])
            if e['where']=='left':self.assertTrue(-zone['right_deg'][1]<=seen['bearing_deg']<=-zone['right_deg'][0],e['id'])
            if e['where']=='centre':self.assertTrue(abs(seen['bearing_deg'])<=zone['centre_deg'] and seen['distance_m']<=zone['centre_within_m'],e['id'])
        hard=[e for e in self.episodes if e['where'] in ('right','centre')];self.assertEqual(len(hard),9)
        self.assertTrue(all(e['role'] in ('same_color','same_shape') for e in hard));self.assertEqual({e['kick']['degrees']>0 for e in self.episodes if 'kick' in e},{True,False})

    def test_no_start_is_near_any_earlier_start_and_none_is_in_a_layout_that_was_recorded_in(self):
        earlier=grounding_arch.earlier_points();recorded={name:set(V3C['train_layouts'][name])|set(CONFIG['gen_v3']['train_layouts'][name]) for name in grounding_arch.SCENES}
        sealed={name:set(CONFIG['canonical']['test_layouts'][name])|{'e'} for name in grounding_arch.SCENES}
        for index,e in enumerate(self.episodes):
            self.assertGreaterEqual(min(math.dist(e['start_xy'],xy) for where,xy in earlier if where==e['map']),ARCH['separation_m'],e['id'])
            self.assertFalse(e['layout'] in recorded[e['map']] or e['layout'] in sealed[e['map']])
            for other in self.episodes[:index]:
                if other['map']==e['map']:self.assertGreaterEqual(math.dist(e['start_xy'],other['start_xy']),ARCH['separation_m'])
        # The earlier points do hold the sets that must stay apart.
        for path in (canonical.FILES['validation'],canonical.FILES['test'],gen_v3c.FILES['validation'],gen_v3c.FILES['pilot']):
            first=json.loads(path.read_text(encoding='utf-8'))['episodes'][0];self.assertIn((first['map'],first['start_xy']),earlier)
        plan=json.loads(gen_v3c.TRAIN_PLAN.read_text(encoding='utf-8'));self.assertIn((plan['train'][0]['map'],plan['train'][0]['start_xy']),earlier)

    def test_the_layouts_are_the_ones_nothing_was_recorded_in(self):
        for name in grounding_arch.SCENES:
            self.assertEqual(set(ARCH['layouts'][name]),set(V3C['pilot_layouts'][name])|set(V3C['validation_layouts'][name]))
            self.assertFalse(set(ARCH['layouts'][name])&(set(V3C['train_layouts'][name])|set(CONFIG['gen_v3']['train_layouts'][name])|set(CONFIG['canonical']['test_layouts'][name])))


class PilotLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.grounding_arch import pilot_lines
        cls.lines=staticmethod(pilot_lines);cls.rule=ARCH['pilot']['pass']

    def metrics(self,**changes):
        return dict({'flights':32,'errors':0,'success':20,'correct_target':22,'wrong_target':9,'collisions':1,'acquired':24,'terminal_as_asked':22,'approach_touchdowns':0,
                     'landing_rate':.95,'starts_never_correct':3},**changes)

    def passed(self,model,baseline=None,regression=None):
        lines=self.lines(model,baseline or self.metrics(),self.rule,regression)
        return all(value>=limit if how=='>=' else value<=limit for value,how,limit in lines.values()),lines

    def test_a_clear_drop_with_nothing_lost_holds_every_line(self):
        good=self.metrics(success=29,correct_target=30,wrong_target=1,collisions=0,acquired=31,terminal_as_asked=31,starts_never_correct=0)
        ok,lines=self.passed(good);self.assertTrue(ok);self.assertNotIn('regression starts: successes',lines)

    def test_the_wrong_target_lines(self):
        good=self.metrics(success=29,correct_target=30,wrong_target=1,collisions=0,acquired=31,terminal_as_asked=31,starts_never_correct=0)
        self.assertFalse(self.passed(dict(good,wrong_target=3))[0])
        # Two or fewer is not enough when the baseline was already there: the drop has to be more than a flight, and at least a half.
        self.assertFalse(self.passed(dict(good,wrong_target=2),self.metrics(wrong_target=3))[0]);self.assertTrue(self.passed(dict(good,wrong_target=1),self.metrics(wrong_target=3))[0])
        self.assertFalse(self.passed(dict(good,wrong_target=2),self.metrics(wrong_target=2))[0]);self.assertFalse(self.passed(dict(good,wrong_target=0),self.metrics(wrong_target=1))[0])
        self.assertTrue(self.passed(dict(good,wrong_target=2),self.metrics(wrong_target=4))[0])
        # More successes with as many wrong objects is not a pass.
        self.assertFalse(self.passed(dict(good,success=32,wrong_target=9))[0])
        self.assertFalse(self.passed(dict(good,starts_never_correct=2))[0])

    def test_nothing_else_may_get_worse(self):
        good=self.metrics(success=29,correct_target=30,wrong_target=1,collisions=0,acquired=31,terminal_as_asked=31,starts_never_correct=0)
        for change in ({'collisions':2},{'terminal_as_asked':21},{'approach_touchdowns':1},{'success':19},{'landing_rate':.85},{'acquired':22},{'errors':1},{'flights':31}):
            self.assertFalse(self.passed(dict(good,**change))[0],change)
        self.assertTrue(self.passed(dict(good,collisions=1,acquired=23,terminal_as_asked=22,success=20))[0])

    def test_the_regression_line_is_part_of_the_verdict_once_flown(self):
        good=self.metrics(success=29,correct_target=30,wrong_target=1,collisions=0,acquired=31,terminal_as_asked=31,starts_never_correct=0)
        flown=lambda success,collisions=0:{'groups':{'all':{'Gen-v2':{'success':30,'collisions':0},'Gen-v3c':{'success':28,'collisions':0},'FiLM':{'success':success,'collisions':collisions}}}}
        self.assertTrue(self.passed(good,regression=flown(26))[0]);self.assertFalse(self.passed(good,regression=flown(25))[0]);self.assertFalse(self.passed(good,regression=flown(28,1))[0])

    def test_the_pass_rule_and_the_schedule_are_the_ones_written_down(self):
        self.assertEqual((self.rule['wrong_target_max'],self.rule['wrong_target_drop_min'],self.rule['wrong_target_halved'],self.rule['starts_never_correct_max']),(2,2,True,1))
        training=ARCH['training'];tune=V3C['fine_tune']
        for key in ('updates','learning_rate','batch_size','gradient_accumulation','eval_every','eval_samples','hard_negative_share','boost','balance','select_band'):
            self.assertEqual(training[key],tune[key],key)
        self.assertEqual(training['grounding_learning_rate'],10*training['learning_rate']);self.assertEqual(ARCH['start_checkpoint'],V3C['checkpoint'])
        film,cross=ARCH['experiments']['film']['module'],ARCH['experiments']['cross_attention']['module']
        self.assertEqual((film['type'],cross['type']),('film','cross_attention'));self.assertEqual(film['position'],cross['position'])
        self.assertEqual(len({ARCH['experiments']['film']['checkpoint'],ARCH['experiments']['cross_attention']['checkpoint'],ARCH['control']['checkpoint'],ARCH['start_checkpoint']}),4)
        self.assertEqual(ARCH['selection']['order'][:2],['passes the pilot','wrong_target'])


class ModuleTests(unittest.TestCase):
    """The two grounding modules on small tensors: no base model is loaded."""
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from src.aerovla_oft import model
        except ImportError as error:raise unittest.SkipTest(f'model dependencies not installed here: {error}')
        cls.torch=torch;cls.model=model;torch.manual_seed(0)
        cls.visual=torch.randn(2,12,10);cls.language=torch.randn(2,5,8);cls.mask=torch.tensor([[1.,1.,1.,0.,0.],[1.,1.,1.,1.,1.]])

    def film(self):return self.model.VisualFiLM(8,10,6)
    def attention(self):return self.model.LanguageVisionAttention(8,10,8,2)

    def test_both_are_the_identity_until_trained(self):
        for module in (self.film(),self.attention()):
            self.assertTrue(self.torch.equal(module(self.visual,self.language,self.mask),self.visual))
            # ... and the first gradient step can move them: the zeroed last layer receives a gradient.
            module(self.visual,self.language,self.mask).square().sum().backward();last=module.net[-1] if hasattr(module,'net') else module.out
            self.assertGreater(float(last.weight.grad.abs().sum()),0.)

    def test_film_changes_every_patch_in_the_same_way_and_only_by_the_sentence(self):
        torch=self.torch;module=self.film();torch.nn.init.normal_(module.net[-1].weight,std=.5);torch.nn.init.normal_(module.net[-1].bias,std=.5)
        out=module(self.visual,self.language,self.mask);self.assertFalse(torch.allclose(out,self.visual))
        # One gamma and one beta per channel: two patches of one sample differ after it exactly as gamma times their difference.
        delta=module.net((self.language*self.mask.unsqueeze(-1)).sum(1)/self.mask.sum(1,keepdim=True));gamma=1+delta[:,:10]
        self.assertTrue(torch.allclose(out[:,0]-out[:,1],gamma*(self.visual[:,0]-self.visual[:,1]),atol=1e-5))
        # Padding is not read; another sentence gives another modulation; the image does not change gamma or beta.
        noisy=self.language.clone();noisy[0,3:]=99.;self.assertTrue(torch.allclose(module(self.visual,noisy,self.mask),out,atol=1e-6))
        self.assertFalse(torch.allclose(module(self.visual,self.language.flip(0),self.mask.flip(0))[0],out[0],atol=1e-4))
        other=torch.randn_like(self.visual);self.assertTrue(torch.allclose(module(other,self.language,self.mask)-other*gamma.unsqueeze(1),out-self.visual*gamma.unsqueeze(1),atol=1e-5))

    def test_attention_reads_the_patches_with_the_words_and_writes_back_where_it_read(self):
        torch=self.torch;module=self.attention();torch.nn.init.normal_(module.out.weight,std=.5)
        weights,_=module.attention(self.visual,self.language,self.mask)
        self.assertEqual(tuple(weights.shape),(2,2,5,12));self.assertTrue(torch.allclose(weights[1].sum(-1),torch.ones(2,5),atol=1e-5))
        self.assertTrue(torch.equal(weights[0,:,3:],torch.zeros(2,2,12)))
        out=module(self.visual,self.language,self.mask);self.assertFalse(torch.allclose(out,self.visual))
        noisy=self.language.clone();noisy[0,3:]=99.;self.assertTrue(torch.allclose(module(self.visual,noisy,self.mask),out,atol=1e-5))
        # Unlike FiLM the change is not the same for every patch, and it depends on the image as well as on the sentence.
        change=(out-self.visual)[1];self.assertGreater(float((change-change.mean(0,keepdim=True)).abs().max()),1e-3)
        # A patch that no word attends to is left as it was.
        with torch.no_grad():
            module.query.weight.zero_();module.query.bias.zero_();module.key.weight.zero_();module.key.bias.zero_();module.key.bias[0]=0.  # uniform attention
            uniform=module(self.visual,self.language,self.mask)
        self.assertGreater(float((uniform-self.visual).abs().min()),0.)
        sharp=self.attention();torch.nn.init.normal_(sharp.out.weight,std=.5)
        with torch.no_grad():
            # Every word looks at patch 0 only: its key is far along the queries' direction, the others are not.
            sharp.query.weight.zero_();sharp.query.bias.fill_(1.);sharp.key.weight.zero_();sharp.key.bias.zero_()
        visual=self.visual.clone();hook=sharp.key.register_forward_hook(lambda _,__,output:output+torch.cat([torch.full((2,1,8),40.),torch.zeros(2,11,8)],dim=1))
        moved=(sharp(visual,self.language,self.mask)-visual).abs().sum(-1);hook.remove()
        self.assertGreater(float(moved[:,0].min()),1e-3);self.assertLess(float(moved[:,1:].max()),1e-6)

    def test_the_module_named_by_a_checkpoint(self):
        build=self.model.grounding_module
        self.assertIsNone(build(None,8,10));self.assertIsNone(build({'type':'none'},8,10))
        film=build(ARCH['experiments']['film']['module'],4096,2176);cross=build(ARCH['experiments']['cross_attention']['module'],4096,2176)
        self.assertIsInstance(film,self.model.VisualFiLM);self.assertIsInstance(cross,self.model.LanguageVisionAttention)
        count=lambda module:sum(parameter.numel() for parameter in module.parameters())
        self.assertEqual(count(film),2*4096+4096*512+512+512*2*2176+2*2176);self.assertEqual(count(cross),2*4096+2*2176+4096*256+256+2*(2176*256+256)+256*2176+2176)
        self.assertLess(max(count(film),count(cross)),5_000_000)
        self.assertRaises(ValueError,build,{'type':'gate'},8,10);self.assertRaises(ValueError,build,{'type':'film','position':'after_projector','hidden_dim':4},8,10)
        self.assertRaises(ValueError,self.model.LanguageVisionAttention,8,10,9,2)

    def test_neither_module_is_given_anything_but_the_patches_and_the_sentence(self):
        import inspect
        for module in (self.model.VisualFiLM,self.model.LanguageVisionAttention):
            self.assertEqual(list(inspect.signature(module.forward).parameters),['self','visual','language','mask'])
        source=inspect.getsource(self.model.AeroVLAOFT.patch_tokens)+inspect.getsource(self.model.AeroVLAOFT.language_tokens)
        for word in ('target','distance','bearing','visible','position','privileged'):self.assertNotIn(word,source)
        # The sentence alone is embedded, and the vision encoder is not trained.
        self.assertIn('add_special_tokens=False',source);self.assertIn('torch.no_grad():features=self.core.vision_backbone',source)


if __name__=='__main__':unittest.main()
