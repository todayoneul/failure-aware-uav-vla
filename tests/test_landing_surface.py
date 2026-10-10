"""Landing surfaces: what may be landed on, the contact reading, and the evaluator's reading of flights whose outcome is
known. The flights are the canonical episode loop over a fake simulator, flown by a scripted pilot; no model is loaded."""
import argparse
import asyncio
import copy
import inspect
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'configs/experiments/controlled_landings.json'


def scene():
    from src.mission import semantic
    return semantic.load_demo()


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        from src.landing.surface import SurfaceSet
        self.scene=scene();self.set=SurfaceSet(self.scene,'mission')

    def test_a_pad_is_a_rectangle_and_its_region_is_the_canonical_one(self):
        from src.visual_search.maps import load_landing
        landing=load_landing();pad=self.set['blue_pad'];half=landing['touchdown']['landing_region_half_width_m']
        self.assertEqual((pad.surface_type,pad.outline,pad.landable),('rectangle','rectangle',True))
        self.assertEqual(pad.usable_half_extent_m,(half,half));self.assertEqual(pad.half_extent_m,(7.,7.))
        # The canonical test of a contact (the larger of the two offsets against the half-width), point for point.
        for offset in ([0.,0.],[6.6,6.6],[-6.6,3.],[6.61,0.],[0.,-6.7],[5.,6.59],[7.,7.]):
            self.assertEqual(pad.inside(offset),max(abs(value) for value in offset)<=half,offset)
        self.assertEqual((pad.contact_names,pad.contact_prefixes),(('BluePad',),('BluePadMark',)))

    def test_a_cube_is_a_rectangle_and_a_cylinder_is_a_circle(self):
        cube=self.set['blue_cube'];cylinder=self.set['green_cylinder']
        self.assertEqual((cube.surface_type,cube.landable,cube.half_extent_m,cube.usable_half_extent_m),('rectangle',True,(4.,4.),(3.6,3.6)))
        self.assertTrue(cube.inside([3.6,-3.6]));self.assertFalse(cube.inside([3.7,0.]));self.assertTrue(cube.on_top([3.9,3.9]));self.assertFalse(cube.on_top([4.1,0.]))
        self.assertEqual((cylinder.surface_type,cylinder.landable,cylinder.radius_m,cylinder.usable_radius_m),('circle',True,4.,3.6))
        self.assertTrue(cylinder.inside([3.6,0.]));self.assertTrue(cylinder.inside([0.,-3.59]))
        # The corner of the square around a disc is not on the disc: a circle is judged by its radius.
        self.assertFalse(cylinder.inside([2.6,2.6]));self.assertTrue(cylinder.on_top([2.8,2.8]));self.assertFalse(cylinder.on_top([2.9,2.9]))
        self.assertEqual((cube.usable_reach_m,cylinder.usable_reach_m),(3.6,3.6))

    def test_the_margin_is_the_vehicles_half_span_and_no_number_is_kept_in_the_rule_file(self):
        from src.landing.surface import load_rules
        from src.visual_search.maps import load_landing
        landing=load_landing()
        for surface in self.set.surfaces.values():self.assertEqual(surface.landing_margin_m,landing['vehicle']['half_span_m'])
        self.assertEqual(self.set.minimum,landing['teacher']['align_radius_m'])
        text=json.dumps({key:value for key,value in load_rules().items() if key!='description'})
        self.assertNotRegex(text,r'"[a-z_]+": ?\d');self.assertIn('vehicle.half_span_m',text);self.assertIn('teacher.align_radius_m',text)

    def test_a_turned_rectangle_is_judged_along_its_own_sides(self):
        from src.landing.surface import SurfaceSet
        turned=copy.deepcopy(self.scene);turned['objects']['blue_cube']=dict(turned['objects']['blue_cube'],yaw_deg=45.)
        cube=SurfaceSet(turned,'mission')['blue_cube'];straight=self.set['blue_cube']
        self.assertEqual(cube.yaw_deg,45.)
        # Along the world's diagonal is along a side of the turned cube; along the world's axis is toward its corner.
        self.assertTrue(cube.inside([2.5,2.5]));self.assertTrue(straight.inside([2.5,2.5]))
        self.assertFalse(cube.on_top([3.,3.]));self.assertTrue(straight.inside([3.,3.]))
        self.assertTrue(cube.inside([3.5,0.]));self.assertTrue(cube.on_top([5.,0.]));self.assertFalse(straight.on_top([5.,0.]))
        local=cube.local([1.,1.]);self.assertAlmostEqual(local[0],math.sqrt(2));self.assertAlmostEqual(local[1],0.)

    def test_the_height_of_a_surface_is_the_objects_own(self):
        ground=self.scene['ground_z']
        for name,top in (('blue_pad',2.5),('blue_cube',8.),('green_cylinder',9.)):
            surface=self.set[name];self.assertEqual((surface.top_m,surface.surface_z),(top,ground-top))
            # From above: at the top, or within the canonical tolerance below it. Lower is a side.
            self.assertTrue(surface.from_above(top-.2,.3));self.assertFalse(surface.from_above(top-.4,.3))
        self.assertIsNotNone(self.set.touched('BlueCube',8.));self.assertIsNone(self.set.touched('BlueCube',6.))
        self.assertEqual(self.set.touched('BluePadMarkGrid',2.5).object_id,'blue_pad');self.assertIsNone(self.set.touched('TemplateCube_Rounded_1',20.))

    def test_a_sphere_a_cone_and_a_pyramid_have_no_surface(self):
        from src.landing.surface import SurfaceSet,refusal,NO_SURFACE
        from src.visual_search.maps import load_map
        for name in ('orange_ball','blue_cone'):
            surface=self.set[name];self.assertEqual((surface.surface_type,surface.landable,surface.outline),('none',False,None))
            self.assertTrue(surface.reason);self.assertFalse(surface.inside([0.,0.]));self.assertFalse(surface.on_top([0.,0.]))
            self.assertEqual(refusal(surface),NO_SURFACE);self.assertIsNone(self.set.touched(self.scene['objects'][name]['native'],surface.top_m))
        self.assertEqual(NO_SURFACE,'Selected object has no valid landing surface. Use APPROACH.')
        self.assertIsNone(refusal(self.set['blue_cube']));self.assertIsNone(refusal(self.set['green_cylinder']))
        yard=load_map('yard');self.assertFalse(SurfaceSet(yard,'a_new')['yellow_pyramid'].landable)

    def test_a_top_too_small_for_a_descent_is_not_offered(self):
        from src.landing.surface import SurfaceSet
        small=copy.deepcopy(self.scene);small['objects']['blue_cube']=dict(small['objects']['blue_cube'],size_m=[4.,4.,8.])
        cube=SurfaceSet(small,'mission')['blue_cube']
        self.assertEqual((cube.surface_type,cube.landable),('rectangle',False));self.assertIn('too small',cube.reason);self.assertFalse(cube.inside([0.,0.]))

    def test_a_cap_makes_a_surface_of_its_plate_and_of_nothing_else(self):
        from src.landing.surface import SurfaceSet
        capped=SurfaceSet(self.scene,'mission_cap');ball=capped['orange_ball']
        self.assertEqual((ball.surface_type,ball.outline,ball.landable,ball.radius_m,ball.top_m),('cap','circle',True,3.,8.7+1.2))
        self.assertAlmostEqual(ball.usable_radius_m,2.6);self.assertEqual(ball.contact_names,('OrangeBallCap',))
        self.assertEqual(capped.owner('OrangeBallCap'),'orange_ball');self.assertIs(capped.touched('OrangeBallCap',9.9),ball)
        # The sphere under the plate is still a sphere: touching it is a collision wherever and from wherever.
        self.assertIsNone(capped.touched('OrangeBall',9.7));self.assertIsNone(capped.touched('OrangeBallCap',8.))
        # Without the plate (the default layout) the same object has no surface.
        self.assertEqual(self.set['orange_ball'].surface_type,'none')
        missing=copy.deepcopy(self.scene);missing['landing_caps']['mission_cap']['orange_ball']='NoSuchPlate'
        with self.assertRaisesRegex(ValueError,'not one of its structures'):SurfaceSet(missing,'mission_cap')

    def test_a_rule_that_changed_a_pads_region_is_refused(self):
        from src.landing.surface import SurfaceSet
        from src.visual_search.maps import load_landing
        landing=load_landing();landing['vehicle']['half_span_m']=.5
        with self.assertRaisesRegex(ValueError,'canonical region'):SurfaceSet(self.scene,'mission',landing)

    def test_every_map_of_the_evaluation_is_read_and_its_pads_are_unchanged(self):
        from src.landing.surface import SurfaceSet
        from src.visual_search.maps import load_map,MapGeometry
        for name in ('blocks','field','lot','yard','depot'):
            config=load_map(name)
            for layout in config['layouts']:
                surfaces=SurfaceSet(config,layout);geometry=MapGeometry(config,layout)
                for target,item in geometry.objects.items():
                    if item['landable']:self.assertEqual((surfaces[target].landable,surfaces[target].usable_half_extent_m),(True,(6.6,6.6)),(name,layout,target))


def bare(kind,map_config,layout,events):
    """An environment with nothing but what the reading of contacts uses."""
    from src.landing.surface import SurfaceSet
    from src.visual_search.maps import MapGeometry,load_landing
    env=object.__new__(kind);env.map_config=map_config;env.layout=layout;env.geometry=MapGeometry(map_config,layout);env.landing=load_landing()
    env.collisions=[];env.scanned=0;env.flight_stamp=100;env.hit=None;env.touchdown=None;env.contact_event=False;env.ground_z=map_config['ground_z']-.19
    env.surfaces=SurfaceSet(map_config,layout);env.feed=list(events);return env


def event(name,height,stamp=200,x=1.,y=2.,ground=-1.19):
    return {'time_stamp':stamp,'object_name':name,'position':{'x':x,'y':y,'z':ground-height}}


class ContactTests(unittest.TestCase):
    """The reading of the collision topic, general against canonical, over the same events."""
    def both(self,map_config,layout,events):
        import scripts.visual_search as canonical
        from src.landing.env import SurfaceContacts
        General=type('General',(SurfaceContacts,canonical.SearchEnv),{});results=[]
        for kind in (canonical.SearchEnv,General):
            env=bare(kind,map_config,layout,events);seen=[]
            for item in env.feed:
                # One event per call, then a call with nothing new, as the loop makes them.
                env.collisions.append(item);seen.append((copy.deepcopy(env.contacts()),env.contact_event))
                seen.append((copy.deepcopy(env.contacts()),env.contact_event))
            results.append(seen)
        return results

    def test_where_only_pads_can_be_landed_on_the_two_readings_are_the_same_event_for_event(self):
        from src.visual_search.maps import load_map
        blocks=load_map('blocks')
        streams=[[event('BluePad',2.5,x=50.,y=-80.)],[event('BluePadMark0',2.56,x=53.,y=-84.)],[event('BluePadMarkGrid',2.52)],
                 [event('BluePad',.8)],[event('BluePad',2.15)],[event('BluePad',2.25)],
                 [event('BluePad',2.5,stamp=50)],[event('BluePad',2.5,stamp=100)],
                 [event('BluePad',2.5,x=50.,y=-80.),event('RedPad',2.5,stamp=300,x=50.,y=80.)],
                 [event('BluePad',2.5),event('TemplateCube_Rounded_12',2.5,stamp=300)],
                 [event('TemplateCube_Rounded_12',9.),event('BluePad',2.5,stamp=300)],
                 [event('Cone_5',10.)],[event('OrangeBall',9.7)],[event('',3.)],[{'time_stamp':200,'position':{'x':0.,'y':0.,'z':-3.}}]]
        for stream in streams:
            canonical,general=self.both(blocks,'c',stream);self.assertEqual(canonical,general,stream)
        # And the streams did say something: a touchdown, a side, a stale event, a collision.
        self.assertEqual(self.both(blocks,'c',streams[0])[1][0][0][0]['object'],'blue_pad')
        self.assertEqual(self.both(blocks,'c',streams[3])[1][0][0],(None,'BluePad'));self.assertEqual(self.both(blocks,'c',streams[6])[1][0],((None,None),False))

    def test_the_top_of_a_cube_is_a_touchdown_and_its_side_is_a_collision(self):
        demo=scene()
        canonical,general=self.both(demo,'mission',[event('BlueCube',8.,x=3.,y=65.)])
        # The canonical reading knows pads only: a cube touched anywhere is a collision there.
        self.assertEqual(canonical[0][0],(None,'BlueCube'))
        touchdown,hit=general[0][0];self.assertIsNone(hit);self.assertEqual(touchdown['object'],'blue_cube')
        self.assertEqual(touchdown['offset_m'],[1.,-1.]);self.assertEqual(touchdown['position'][:2],[3.,65.])
        for height in (6.,7.6):
            self.assertEqual(self.both(demo,'mission',[event('BlueCube',height)])[1][0][0],(None,'BlueCube'))
        self.assertEqual(self.both(demo,'mission',[event('GreenCylinder',9.,x=55.,y=30.)])[1][0][0][0]['object'],'green_cylinder')
        self.assertEqual(self.both(demo,'mission',[event('OrangeBall',9.7)])[1][0][0],(None,'OrangeBall'))
        # On a pad the general reading of this layout is the canonical one.
        pads=[event('BluePad',2.5,x=12.,y=90.)];self.assertEqual(*self.both(demo,'mission',pads))

    def test_a_cap_is_a_touchdown_and_the_sphere_under_it_is_a_collision(self):
        demo=scene()
        touchdown,hit=self.both(demo,'mission_cap',[event('OrangeBallCap',9.9,x=92.15,y=32.1)])[1][0][0]
        self.assertIsNone(hit);self.assertEqual(touchdown['object'],'orange_ball');self.assertAlmostEqual(touchdown['offset_m'][0],1.)
        self.assertEqual(self.both(demo,'mission_cap',[event('OrangeBall',7.4)])[1][0][0],(None,'OrangeBall'))


class FinalizerTests(unittest.TestCase):
    def test_the_finalizer_is_told_nothing_of_a_target_or_a_surface(self):
        from src.visual_search.finalizer import LandingFinalizer
        self.assertEqual(list(inspect.signature(LandingFinalizer.__init__).parameters),['self','settings','instruction','enabled'])
        self.assertEqual(list(inspect.signature(LandingFinalizer.update).parameters),['self','position','stamp_s','contact_event','executed_down_m'])
        source=(ROOT/'src/visual_search/finalizer.py').read_text(encoding='utf-8').split('"""',2)[2]
        for word in ('target','radius','centre','offset','SurfaceSet','landable'):self.assertNotIn(word,source,word)
        # What this work adds reads contacts and summaries; it never constructs, feeds or patches a finalizer.
        for name in ('src/landing/env.py','src/landing/evaluator.py','src/landing/surface.py','src/mission/start.py'):
            code=(ROOT/name).read_text(encoding='utf-8').split('"""',2)[2]
            self.assertNotIn('LandingFinalizer',code,name);self.assertNotIn('finalizer.',code,name)


class Controlled:
    """One flight of the canonical loop over the fake simulator, flown by the scripted pilot through the mission's own parts."""
    def __init__(self,directory,layout='mission',launch=None):
        from tests.test_grounding_mission import Flight
        self.flight=Flight(directory,launch=launch,layout=layout)

    async def fly(self,told,task,mode,aim,offset=(0.,0.),top=None):
        from src.landing.controlled import ControlledPilot,aim_point
        from src.landing.surface import SurfaceSet
        flight=self.flight;surfaces=SurfaceSet(flight.scene,flight.scene['mission']['layout'])
        point,height=aim_point({'aim':aim,'offset_m':list(offset),'top_m':top or 0.},surfaces)
        pilot=ControlledPilot(flight.env,flight.oft,flight.env.landing,point,height,mode);self.pilot=pilot
        await flight.fly(told,task,lambda sim:(lambda index:pilot.act()[0]))
        return flight.summary,flight.reading


class EvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):asyncio.get_running_loop().slow_callback_duration=60.

    async def run_one(self,*arguments,layout='mission',**keywords):
        with tempfile.TemporaryDirectory() as directory:
            self.controlled=Controlled(directory,layout);return await self.controlled.fly(*arguments,**keywords)

    async def test_the_top_of_the_named_cube_is_a_landing(self):
        summary,reading=await self.run_one('blue_cube','land','land','blue_cube')
        self.assertTrue(reading['success']);self.assertEqual((reading['landed_on'],reading['target_surface']['surface']),('blue_cube','rectangle'))
        self.assertTrue(all(reading[key] for key in ('correct_object','on_top','inside_region','acceptable_touchdown','physical_landing','finalizer_latched',
                                                     'touchdown_success','stable_physical_landing','system_land_success')))
        self.assertIsNone(reading['collision']);self.assertEqual(summary['finalizer']['state'],'DISARMED');self.assertEqual(summary['reason'],'landed')
        self.assertLess(reading['horizontal_error_m'],.2);self.assertEqual(self.controlled.flight.text,'Find the blue cube and land on it.')
        # The vehicle passed the height of the cube's top on its way up, at 1 m/s. The canonical reading takes that decision for
        # the touchdown; this one takes the first decision at that height after the contact, when it was coming down at 0.5 m/s.
        self.assertLess(reading['canonical_touchdown_decision'],10);self.assertAlmostEqual(abs(reading['canonical_vertical_speed_mps']),1.)
        self.assertGreater(reading['touchdown_decision'],30);self.assertAlmostEqual(reading['vertical_speed_mps'],.5)
        self.assertFalse(summary['success']);self.assertFalse(reading['agrees_with_canonical'])
        # The vehicle stands 8 m above the ground, on the cube.
        self.assertAlmostEqual(self.controlled.flight.sim.ground-.19-self.controlled.flight.sim.z,8.,places=6)

    async def test_the_side_of_the_cube_is_a_collision(self):
        summary,reading=await self.run_one('blue_cube','land','ram','blue_cube')
        self.assertEqual((summary['reason'],reading['collision']),('collision','BlueCube'))
        self.assertFalse(reading['success']);self.assertFalse(reading['contact_from_above']);self.assertIsNone(reading['landed_on'])
        self.assertFalse(reading['physical_landing']);self.assertFalse(reading['finalizer_latched'])

    async def test_the_top_of_the_other_cube_is_a_physical_landing_and_a_failed_mission(self):
        from src.landing import evaluator
        summary,reading=await self.run_one('blue_cube','land','land','red_cube')
        self.assertEqual(self.controlled.flight.text,'Find the blue cube and land on it.')
        self.assertEqual((reading['landed_on'],reading['wrong_object'],reading['correct_object']),('red_cube',True,False))
        # The finalizer was not told which cube was meant: it latched and switched the motors off.
        self.assertTrue(reading['physical_landing']);self.assertTrue(reading['finalizer_latched']);self.assertEqual(summary['finalizer']['state'],'DISARMED')
        self.assertTrue(reading['wrong_object_landing']);self.assertFalse(reading['success']);self.assertTrue(summary['wrong_target'])
        self.assertIn('red_cube',evaluator.words(reading,summary));self.assertIn('physical landing',evaluator.words(reading,summary))

    async def test_the_middle_of_the_cylinder_is_a_landing_and_its_edge_is_not(self):
        summary,reading=await self.run_one('green_cylinder','land','land','green_cylinder')
        self.assertTrue(reading['success']);self.assertEqual((reading['landed_on'],reading['target_surface']['surface']),('green_cylinder','circle'))
        summary,reading=await self.run_one('green_cylinder','land','land','green_cylinder',offset=(3.8,0.))
        self.assertEqual(reading['landed_on'],'green_cylinder');self.assertTrue(reading['on_top']);self.assertFalse(reading['inside_region'])
        self.assertFalse(reading['success']);self.assertTrue(reading['physical_landing']);self.assertTrue(reading['finalizer_latched'])
        # The region written for a pad would have let this contact through: it is a 6.6 m square.
        self.assertTrue(reading['canonical_inside_region'])
        self.assertAlmostEqual(reading['horizontal_error_m'],3.8,delta=.15)

    async def test_hovering_over_the_cube_is_not_a_landing(self):
        from src.landing import evaluator
        summary,reading=await self.run_one('blue_cube','land','hover','blue_cube')
        self.assertEqual((summary['reason'],summary['stopped']),('model_stop',True));self.assertFalse(reading['success'])
        self.assertFalse(reading['contact_from_above']);self.assertIsNone(reading['collision']);self.assertGreater(reading['decisions_over_usable_region'],0)
        self.assertIn('in the air over',evaluator.words(reading,summary))

    async def test_a_sphere_is_refused_for_a_landing_and_touching_its_top_is_a_collision(self):
        from src.mission import semantic
        with tempfile.TemporaryDirectory() as directory:
            ball=Controlled(directory).flight.by['orange_ball']
        self.assertEqual(semantic.refusal(ball,'land'),'Selected object has no valid landing surface. Use APPROACH.')
        summary,reading=await self.run_one('orange_ball','approach','land','orange_ball')
        self.assertEqual(reading['collision'],'OrangeBall');self.assertFalse(reading['success']);self.assertFalse(reading['contact_from_above'])

    async def test_on_a_pad_the_reading_is_the_canonical_one(self):
        for offset,inside in (((0.,0.),True),((6.4,0.),True),((6.8,0.),False)):
            summary,reading=await self.run_one('blue_pad','land','land','blue_pad',offset=offset)
            self.assertEqual(reading['landed_on'],'blue_pad')
            self.assertEqual((reading['inside_region'],reading['canonical_inside_region']),(inside,inside),offset)
            self.assertEqual((reading['success'],summary['success']),(inside,inside),offset);self.assertTrue(reading['agrees_with_canonical'])
            self.assertEqual((reading['touchdown_decision'],reading['vertical_speed_mps']),(reading['canonical_touchdown_decision'],reading['canonical_vertical_speed_mps']))
            self.assertEqual((reading['touchdown_success'],reading['stable_physical_landing'],reading['system_land_success'],reading['physical_landing']),
                             (summary['touchdown_success'],summary['stable_physical_landing'],summary['system_land_success'],summary['physical_landing']))

    async def test_an_approach_is_read_by_the_canonical_evaluator_alone(self):
        # 12 m west of the ball's centre at the height the flight started at: a stop in the air beside it.
        summary,reading=await self.run_one('orange_ball','approach','hover',[79.15,32.1],top=3.)
        self.assertTrue(summary['success']);self.assertTrue(reading['success']);self.assertTrue(reading['agrees_with_canonical'])
        self.assertNotIn('finalizer',summary);self.assertFalse(reading['finalizer_active']);self.assertNotIn('system_land_success',reading)
        self.assertEqual(self.controlled.flight.sim.calls[-1],'hover');self.assertNotIn('disarm',self.controlled.flight.sim.calls)
        # Coming down on a cube during an approach to it is a touchdown, and so a failed approach: as on a pad.
        summary,reading=await self.run_one('blue_cube','approach','land','blue_cube')
        self.assertTrue(summary['landed']);self.assertFalse(summary['success']);self.assertFalse(reading['success']);self.assertNotIn('disarm',self.controlled.flight.sim.calls)

    async def test_a_cap_is_landed_on_and_the_sphere_beside_it_is_hit(self):
        summary,reading=await self.run_one('orange_ball','land','land','orange_ball',layout='mission_cap')
        self.assertTrue(reading['success']);self.assertEqual((reading['landed_on'],reading['target_surface']['surface']),('orange_ball','cap'))
        self.assertAlmostEqual(self.controlled.flight.sim.ground-.19-self.controlled.flight.sim.z,9.9,places=6)
        summary,reading=await self.run_one('orange_ball','land','land','orange_ball',offset=(-4.2,0.),layout='mission_cap')
        self.assertEqual(reading['collision'],'OrangeBall');self.assertFalse(reading['success']);self.assertFalse(reading['contact_from_above'])


class RecordedLandingTests(unittest.TestCase):
    def test_two_recorded_landings_on_a_pad_are_read_as_the_canonical_evaluator_reads_them(self):
        import scripts.visual_search as canonical
        from src.landing import evaluator
        from src.landing.surface import SurfaceSet
        from src.visual_search.episodes import load_config
        from src.visual_search.maps import load_landing,load_map
        config=load_config();landing=load_landing();surfaces=SurfaceSet(load_map('blocks'),'c',landing)
        episodes=json.loads((ROOT/'tests/data/noisy_velocity_landings.json').read_text(encoding='utf-8'))['episodes'];self.assertEqual(len(episodes),2)
        for item in episodes:
            episode={'id':item['id'],'target':item['target'],'task':'land'}
            summary=canonical.summarise(episode,item['steps'],config,item['reason'],item['stopped'],item['other_objects_m'],landing,item['touchdown'],None,item['finalizer'])
            reading=evaluator.read(summary,surfaces,landing,item['steps'])
            self.assertTrue(summary['success'] and reading['success'] and reading['agrees_with_canonical'],item['id'])
            for key in ('touchdown_success','stable_physical_landing','system_land_success','physical_landing'):self.assertEqual(reading[key],summary[key],(item['id'],key))
            self.assertEqual((reading['inside_region'],reading['touchdown_decision'],reading['vertical_speed_mps'],reading['landed_on']),
                             (summary['touchdown']['inside_region'],summary['touchdown']['step'],summary['touchdown']['vertical_speed_mps'],summary['landed_on']),item['id'])

    def test_the_stay_on_a_surface_is_found_from_the_contact_not_from_the_height_alone(self):
        from src.landing.evaluator import touchdown_decision
        def flight(heights,marked):
            return [{'position':[0.,0.,-height],'on_surface':index in marked} for index,height in enumerate(heights)]
        # Climbing past 8 m (decision 2), over the cube, down onto it at decision 7; the report of the contact arrives one decision late.
        heights=[6.,7.,8.,9.,11.,10.,9.,8.,8.,8.]
        self.assertEqual(touchdown_decision(flight(heights,{8,9}),{'step':2},.05),7)
        self.assertEqual(touchdown_decision(flight(heights,{7,8,9}),{'step':2},.05),7)
        # A pad: nothing was ever at that height before, and the canonical decision is the one of the stay.
        pad=[6.,5.,4.,3.,2.56,2.5,2.5,2.5]
        self.assertEqual(touchdown_decision(flight(pad,{6,7}),{'step':5},.05),5);self.assertEqual(touchdown_decision(flight(pad,{5,6,7}),{'step':5},.05),5)
        # Within the tolerance of the resting height while still coming down: the canonical decision, measured from the contact itself, is kept.
        self.assertEqual(touchdown_decision(flight([3.,2.549,2.5,2.5],{2,3}),{'step':2},.05),2)
        self.assertIsNone(touchdown_decision(flight(pad,set()),{'step':5},.05))


class ControlledPlanTests(unittest.IsolatedAsyncioTestCase):
    """The plan the simulator is flown with, flown here over the fake simulator by the same worker."""
    async def asyncSetUp(self):asyncio.get_running_loop().slow_callback_duration=60.

    def test_the_plan_says_what_each_flight_has_to_show_and_every_start_may_be_flown(self):
        from src.landing.controlled import MODES
        from src.mission import start as starts
        from src.visual_search.episodes import load_config
        plan=json.loads(PLAN.read_text(encoding='utf-8'));demo=scene();config=load_config()
        ids=[episode['id'] for episode in plan['episodes']];self.assertEqual(len(ids),len(set(ids)))
        for episode in plan['episodes']:
            layout=episode.get('layout','mission')
            state=starts.StartState.from_episode(episode,config['cruise_height_m']);self.assertTrue(starts.validate(state,demo,layout,config)['valid'],episode['id'])
            if episode.get('expect_refusal'):continue
            self.assertIn(episode['pilot']['mode'],MODES);self.assertTrue(episode['expect'],episode['id']);self.assertTrue(episode['why'])
        for needed in ('pad-land','cube-land','cylinder-land','cube-side','cube-wrong','cylinder-edge','cube-hover','sphere-land-refused','cap-land'):self.assertIn(needed,ids)
        self.assertEqual(sum(episode['id'].startswith('start-') for episode in plan['episodes']),3)

    async def test_every_controlled_flight_is_read_as_the_plan_expects(self):
        import projectairsim
        import scripts.surface_flights as worker
        import scripts.visual_search as canonical
        from src.mission.start import wrap_deg
        from tests.fake_airsim import FakeSim,FakeClient,world_class,drone_class
        plan=json.loads(PLAN.read_text(encoding='utf-8'));demo=scene()
        async def no_wait(_):return None
        clock=types.SimpleNamespace(sleep=no_wait,wait_for=asyncio.wait_for)
        with tempfile.TemporaryDirectory() as directory:
            for layout in ('mission','mission_cap'):
                only=[episode['id'] for episode in plan['episodes'] if episode.get('layout','mission')==layout];sim=FakeSim(demo,layout)
                arguments=argparse.Namespace(host='fake',ports=[1,2],plan=str(PLAN),output=directory,policy='controlled',checkpoint=None,only=only,limit=0,again=False)
                with patch.object(projectairsim,'ProjectAirSimClient',lambda **_:FakeClient(sim)),patch.object(canonical,'World',world_class(sim)), \
                     patch.object(canonical,'Drone',drone_class(sim)),patch.object(canonical,'matched_camera_images',sim.camera), \
                     patch.object(canonical,'asyncio',clock),patch('builtins.print'):
                    with self.assertRaises(SystemExit) as ended:await worker.fly(arguments)
                self.assertEqual(ended.exception.code,0,layout);self.assertNotIn('land_async',sim.calls)
            results=json.loads((Path(directory)/'results.json').read_text(encoding='utf-8'))
        records={item['id']:item for item in results['episodes']};self.assertEqual(set(records),{episode['id'] for episode in plan['episodes']})
        for record in records.values():self.assertTrue(record['passed'],(record['id'],record['checks']))
        self.assertFalse(records['sphere-land-refused']['flown']);self.assertEqual(records['sphere-land-refused']['refusal'],'Selected object has no valid landing surface. Use APPROACH.')
        # Every flown episode started where its plan said: the place, the heading and the height, on the ground.
        for episode in plan['episodes']:
            record=records[episode['id']]
            if not record['flown']:continue
            self.assertEqual((record['start_reached']['x'],record['start_reached']['y']),tuple(episode['start_xy']))
            self.assertAlmostEqual(wrap_deg(record['start_reached']['yaw_deg']-episode['start_yaw_deg']),0.,places=6)
            self.assertAlmostEqual(record['start_reached']['height_m'],episode['start_height_m'],places=6);self.assertTrue(record['spawn']['on_ground'])
        self.assertEqual(records['cube-land']['pilot_phases'],['climb','transit','descend','rest'])
        self.assertEqual(records['cube-wrong']['surface_landing']['landed_on'],'red_cube')


if __name__=='__main__':unittest.main()
