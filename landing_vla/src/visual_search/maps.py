"""Map definitions for visual search: what stands where, and the geometry questions asked about it.

A map is a JSON file under configs/maps/: the simulator scene it runs in, its fixed obstacles, the
objects that can be named as targets, where they stand in each layout, the area a vehicle may start
in and the altitudes it may use. Nothing here talks to the simulator or to a model; the planner, the
teacher and the scorer all read the same boxes.

Geometry is 2.5-D: every obstacle is an axis-aligned footprint that is solid from the ground up to
its top, which is conservative for the few blocks that hang in the air.
"""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MAPS=ROOT/'configs/maps'
TARGETS=ROOT/'configs/targets'
SIGHT_FRACTIONS=(.25,.5,.75,.9)
# A pad is seen by its top surface: its centre (second, as the reference point) and a point toward each edge.
PAD_SIGHT=((1,0),(0,0),(-1,0),(0,1),(0,-1))


def load_landing():
    """The landing rule (configs/targets/landing_pads.json)."""
    return json.loads((TARGETS/'landing_pads.json').read_text(encoding='utf-8'))


def load_map(name):
    """Read configs/maps/<name>.json (or a path) and attach its obstacle list."""
    path=Path(name) if str(name).endswith('.json') else MAPS/f'{name}.json'
    config=json.loads(path.read_text(encoding='utf-8'))
    boxes=[]
    if config.get('obstacles'):
        audit=json.loads((path.parent/config['obstacles']).read_text(encoding='utf-8'))
        boxes=[{'name':b['name'],'min':b['min'],'max':b['max'],'top_m':b['top_m']} for b in audit['boxes']]
        config['native_objects']=audit.get('objects',{})
    config['boxes']=boxes
    if config.get('catalogue'):
        # Shared object definitions; a map's own entry for an object wins.
        catalogue=json.loads((path.parent/config['catalogue']).read_text(encoding='utf-8'))
        config['objects']={**catalogue['objects'],**config.get('objects',{})};config['mark']=catalogue['mark']
    if 'origin' in config:
        # A scene written around its own centre is moved to where it stands in the simulator.
        ox,oy=config.pop('origin');shift=lambda point:[point[0]+ox,point[1]+oy]
        config['area']={'x':[value+ox for value in config['area']['x']],'y':[value+oy for value in config['area']['y']]}
        config['sites']={name:shift(point) for name,point in config.get('sites',{}).items()}
        for item in config.get('structures',[]):item['position']=shift(item['position'])
    for layout,places in config['layouts'].items():
        for target in places:
            if target not in config['objects']:raise ValueError(f'Layout {layout!r} places an undefined object: {target}')
    return config


def footprint(name,position,size_m):
    x,y=position[:2];sx,sy,height=size_m
    return {'name':name,'min':[x-sx/2,y-sy/2],'max':[x+sx/2,y+sy/2],'top_m':height}


def object_position(config,layout,target):
    """Ground-plane centre of one object in one layout."""
    place=config['layouts'][layout][target]
    if place=='native':
        return list(config['native_objects'][config['objects'][target]['native']]['center'])
    if isinstance(place,str):return list(config['sites'][place])
    return list(place)


def lighting(config,layout):
    """The lighting of one layout: its own if the map gives one, otherwise the map's."""
    return config.get('layout_lighting',{}).get(layout,config.get('lighting'))


def mark_parts(config,target,centre):
    """The pieces of the H on top of a landable object: name, centre, size."""
    spec=config['objects'][target];mark=config['mark'];sx,sy,top=spec['size_m'];parts=[]
    for index,(dx,dy,length_x,length_y) in enumerate(((0.,-mark['bar_offset']*sy,mark['bar_length']*sx,mark['bar_width']*sy),
                                                      (0.,mark['bar_offset']*sy,mark['bar_length']*sx,mark['bar_width']*sy),
                                                      (0.,0.,mark['bar_width']*sx,2*mark['bar_offset']*sy-mark['bar_width']*sy))):
        parts.append({'name':f'{spec["object"]}Mark{index}','position':[centre[0]+dx,centre[1]+dy],
                      'size_m':[round(length_x,3),round(length_y,3),mark['thickness_m']],'shape':'box','color':spec['mark_color'],'lift_m':top})
    grid=mark.get('grid')
    if grid:
        # Fine lines over the whole top: from a few decimetres up they are thin and many, on the surface one fills the view.
        parts.append({'name':f'{spec["object"]}MarkGrid','position':list(centre),'size_m':[sx,sy,grid['thickness_m']],'shape':'grid','color':spec['mark_color'],
                      'lift_m':top,'detail':{'spacing_m':grid['spacing_m'],'line_m':grid['line_m']}})
    return parts


class MapGeometry:
    """The obstacles of one map in one layout, objects included, with heights above the ground."""
    def __init__(self,config,layout):
        self.config=config;self.layout=layout;self.objects={}
        self.boxes=[dict(box) for box in config['boxes']]
        for item in structures(config,layout):
            if item.get('obstacle',True):self.boxes.append(footprint(item['name'],item['position'],item['size_m']))
        for target in config['layouts'][layout]:
            centre=object_position(config,layout,target);size=config['objects'][target]['size_m']
            self.objects[target]={'centre':centre,'size_m':size,'landable':bool(config['objects'][target].get('landable'))}
            self.boxes.append(footprint(target,centre,size))

    def sight_points(self,target):
        """Points up the middle of the object, as heights above the ground; the top one is near its tip."""
        item=self.objects[target]
        if item['landable']:
            (x,y),(sx,sy,top)=item['centre'],item['size_m']
            return [[x+ox*sx*.35,y+oy*sy*.35,top] for ox,oy in PAD_SIGHT]
        return [[item['centre'][0],item['centre'][1],item['size_m'][2]*fraction] for fraction in SIGHT_FRACTIONS]

    def over(self,x,y,target):
        """True when a point is inside the target's footprint."""
        item=self.objects[target]
        return abs(x-item['centre'][0])<=item['size_m'][0]/2 and abs(y-item['centre'][1])<=item['size_m'][1]/2

    @staticmethod
    def _gap(box,x,y):
        dx=max(box['min'][0]-x,0.,x-box['max'][0]);dy=max(box['min'][1]-y,0.,y-box['max'][1])
        return math.hypot(dx,dy)

    def nearest(self,x,y,above_m=0.,ignore=()):
        """Distance to the closest obstacle that rises above `above_m`, and its name."""
        best=(float('inf'),None)
        for box in self.boxes:
            if box['name'] in ignore or box['top_m']<=above_m:continue
            gap=self._gap(box,x,y)
            if gap<best[0]:best=(gap,box['name'])
        return best

    def inside_area(self,x,y):
        area=self.config['area']
        return area['x'][0]<=x<=area['x'][1] and area['y'][0]<=y<=area['y'][1]

    def blocking(self,start,end,ignore=()):
        """Name of the first obstacle that the straight line between two (x, y, height) points passes through."""
        best=(float('inf'),None)
        for box in self.boxes:
            if box['name'] in ignore:continue
            low,high=0.,1.
            for axis,(a,b) in enumerate(((box['min'][0],box['max'][0]),(box['min'][1],box['max'][1]),(-1e9,box['top_m']))):
                delta=end[axis]-start[axis]
                if abs(delta)<1e-9:
                    if not a<=start[axis]<=b:low=2.;break
                    continue
                t0,t1=(a-start[axis])/delta,(b-start[axis])/delta
                low=max(low,min(t0,t1));high=min(high,max(t0,t1))
            if low<=high and low<best[0]:best=(low,box['name'])
        return best[1]

    def sees(self,x,y,height_m,target):
        """True when at least one sight point of the target has a free line to the viewer."""
        return any(self.blocking((x,y,height_m),point,ignore=(target,)) is None for point in self.sight_points(target))

    def ahead(self,x,y,yaw_rad,height_m,reach_m,margin_m=1.,spread_deg=15.,ignore=()):
        """Free distance in front of a vehicle: the nearest obstacle whose top is less than `margin_m`
        below it, on three rays (straight ahead and to either side)."""
        best=reach_m
        for offset in (-spread_deg,0.,spread_deg):
            heading=yaw_rad+math.radians(offset);dx,dy=math.cos(heading),math.sin(heading)
            for box in self.boxes:
                if box['name'] in ignore or box['top_m']<=height_m-margin_m:continue
                low,high=0.,best
                for origin,delta,a,b in ((x,dx,box['min'][0],box['max'][0]),(y,dy,box['min'][1],box['max'][1])):
                    if abs(delta)<1e-9:
                        if not a<=origin<=b:low=float('inf');break
                        continue
                    t0,t1=(a-origin)/delta,(b-origin)/delta
                    low=max(low,min(t0,t1));high=min(high,max(t0,t1))
                if low<=high:best=min(best,max(0.,low))
        return best

    def corridor_clear(self,start_xy,end_xy,height_m,half_width_m,clearance_m,ignore=()):
        """No obstacle taller than `height - clearance` within `half_width` of the straight ground track."""
        length=math.hypot(end_xy[0]-start_xy[0],end_xy[1]-start_xy[1]);steps=max(1,int(length))
        for index in range(steps+1):
            t=index/steps;x=start_xy[0]+(end_xy[0]-start_xy[0])*t;y=start_xy[1]+(end_xy[1]-start_xy[1])*t
            if self.nearest(x,y,height_m-clearance_m,ignore)[0]<half_width_m:return False
        return True


def structures(config,layout):
    """Added scenery of one layout; a structure without a `layouts` list belongs to all of them."""
    return [item for item in config.get('structures',[]) if layout in item.get('layouts',[layout])]


def scene_objects(config,layout):
    """What the simulator has to add for one layout: structures first, then the objects that are not native.

    Each entry has a name, a centre position [x, y] on the ground, a size, and either a mesh
    (shape + colour) or the name of an asset packaged with the simulator."""
    items=[]
    for item in structures(config,layout):
        items.append({'name':item['name'],'position':item['position'],'size_m':item['size_m'],
                      'shape':item['shape'],'color':item['color'],'lift_m':item.get('lift_m',0.)})
    for target,place in config['layouts'][layout].items():
        if place=='native':continue
        centre=object_position(config,layout,target)
        spec=config['objects'][target];entry={'name':spec['object'],'position':centre,'size_m':spec['size_m'],'lift_m':0.}
        if 'asset' in spec:entry.update(asset=spec['asset'],scale=spec['scale'])
        else:entry.update(shape=spec['shape'],color=spec['color'])
        items.append(entry)
        if spec.get('landable'):items+=mark_parts(config,target,centre)
    return items


def owner_of(config,layout,name):
    """Which object of the layout a simulator object name belongs to (a pad's marking belongs to the pad), or None."""
    for target in config['layouts'][layout]:
        spec=config['objects'][target];own=spec.get('object') or spec.get('native')
        if name==own or (spec.get('landable') and name.startswith(f'{own}Mark')):return target
    return None
