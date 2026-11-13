"""Landing surfaces: where on an object a vehicle may land. No simulator, no model.

The canonical evaluation knows one landing surface, the top of a pad, and its rule is written for a pad
(configs/targets/landing_pads.json: a square region of half-width 6.6 m). Here the same rule is stated for any flat top:
the surface is read from the object's own geometry, and the usable region is that top shrunk by the vehicle's half-span.
For a pad this gives the canonical region exactly, and `SurfaceSet` refuses to exist if it does not.

    rectangle   the flat top of a pad, a cube or a box
    circle      the flat top of an upright cylinder
    cap         a flat plate standing on an object that has no flat top of its own
    none        nothing to land on (a sphere, a cone, a pyramid)

Everything here is for the mission interface, the evaluator and the log. A policy is given two frames and a sentence; the
landing finalizer is given a contact report and the vehicle's own positions. Neither is told what a surface is.
"""
import json
import math
from dataclasses import dataclass,replace
from pathlib import Path
from typing import Optional,Tuple
from src.visual_search.maps import MapGeometry,load_landing,owner_of,structures

ROOT=Path(__file__).resolve().parents[2]
RULES=ROOT/'configs/landing/surfaces.json'
TYPES=('rectangle','circle','cap','none')
NO_SURFACE='Selected object has no valid landing surface. Use APPROACH.'


def load_rules(path=None):
    return json.loads(Path(path or RULES).read_text(encoding='utf-8'))


@dataclass(frozen=True)
class LandingSurface:
    """One object's landing surface. Lengths in metres; `top_m` is the surface's height above the ground and `surface_z` the
    same in the simulator's frame (z down). `outline` says how the region is bounded: a cap has the outline of its plate."""
    object_id:str
    shape:str
    surface_type:str
    landable:bool
    centre_xy:Tuple[float,float]
    top_m:float
    surface_z:float
    outline:Optional[str]=None
    half_extent_m:Optional[Tuple[float,float]]=None
    radius_m:Optional[float]=None
    yaw_deg:float=0.
    landing_margin_m:float=0.
    contact_names:Tuple[str,...]=()
    contact_prefixes:Tuple[str,...]=()
    reason:Optional[str]=None

    @property
    def usable_half_extent_m(self):
        """Half-width and half-depth of the region the vehicle's centre may stand in (a rectangle), or None."""
        if self.outline!='rectangle':return None
        return tuple(max(0.,value-self.landing_margin_m) for value in self.half_extent_m)

    @property
    def usable_radius_m(self):
        if self.outline!='circle':return None
        return max(0.,self.radius_m-self.landing_margin_m)

    @property
    def usable_reach_m(self):
        """How far from its middle the usable region reaches in every direction."""
        if self.outline=='rectangle':return min(self.usable_half_extent_m)
        return self.usable_radius_m if self.outline=='circle' else 0.

    def local(self,offset):
        """An offset from the surface's centre (world x, y) in the surface's own axes: an object turned about the vertical
        is judged along its own sides."""
        turn=math.radians(self.yaw_deg);dx,dy=offset[0],offset[1]
        return dx*math.cos(turn)+dy*math.sin(turn),-dx*math.sin(turn)+dy*math.cos(turn)

    def _within(self,offset,shrink):
        if self.outline=='rectangle':
            u,v=self.local(offset);return abs(u)<=self.half_extent_m[0]-shrink and abs(v)<=self.half_extent_m[1]-shrink
        if self.outline=='circle':return math.hypot(offset[0],offset[1])<=self.radius_m-shrink
        return False

    def on_top(self,offset):
        """Is a point at this offset from the centre over the flat top at all?"""
        return self._within(offset,0.)

    def inside(self,offset):
        """Is it inside the usable region: the whole vehicle on the surface?"""
        return bool(self.landable and self._within(offset,self.landing_margin_m))

    def offset_of(self,x,y):
        return [x-self.centre_xy[0],y-self.centre_xy[1]]

    def from_above(self,height_m,tolerance_m):
        """Was a contact at this height above the ground made on the top, not on a side?"""
        return height_m>=self.top_m-tolerance_m

    def owns_contact(self,name):
        """Is a contact with this simulator object a contact with the surface itself?"""
        return name in self.contact_names or any(name.startswith(prefix) for prefix in self.contact_prefixes)

    def describe(self):
        """For the display and the log."""
        item={'object_id':self.object_id,'shape':self.shape,'surface':self.surface_type,'landable':self.landable,'outline':self.outline,
              'centre_xy':list(self.centre_xy),'top_m':self.top_m,'surface_z':self.surface_z,'yaw_deg':self.yaw_deg,
              'landing_margin_m':self.landing_margin_m,'reason':self.reason}
        if self.outline=='rectangle':item.update(half_extent_m=list(self.half_extent_m),usable_half_extent_m=list(self.usable_half_extent_m))
        if self.outline=='circle':item.update(radius_m=self.radius_m,usable_radius_m=self.usable_radius_m)
        return item


class SurfaceSet:
    """The landing surface of every object of one layout of one map, and which object a simulator object belongs to."""
    def __init__(self,map_config,layout,landing=None,rules=None):
        self.map_config=map_config;self.layout=layout;self.landing=landing or load_landing();self.rules=rules or load_rules()
        self.margin=self.landing['vehicle']['half_span_m'];self.minimum=self.landing['teacher']['align_radius_m']
        self.geometry=MapGeometry(map_config,layout);ground=map_config['ground_z']
        caps=(map_config.get('landing_caps') or {}).get(layout,{});added={item['name']:item for item in structures(map_config,layout)}
        self.cap_owner={};self.surfaces={}
        for name,item in self.geometry.objects.items():
            spec=map_config['objects'][name];shape=spec.get('attributes',{}).get('shape') or spec.get('shape') or 'unknown'
            own=spec.get('object') or spec.get('native');rule=self.rules['shapes'].get(shape,{'surface':'none','why':'no flat top is declared for this shape'})
            fields=dict(object_id=name,shape=shape,landing_margin_m=self.margin,yaw_deg=float(spec.get('yaw_deg',0.)))
            if name in caps:
                if caps[name] not in added:raise ValueError(f'Layout {layout!r} gives {name} a landing cap that is not one of its structures: {caps[name]}')
                plate=added[caps[name]];top=plate.get('lift_m',0.)+plate['size_m'][2];self.cap_owner[plate['name']]=name
                round_plate=plate['shape']=='cylinder'
                fields.update(surface_type='cap',centre_xy=tuple(plate['position'][:2]),top_m=top,surface_z=ground-top,contact_names=(plate['name'],),
                              outline='circle' if round_plate else 'rectangle',radius_m=plate['size_m'][0]/2 if round_plate else None,
                              half_extent_m=None if round_plate else (plate['size_m'][0]/2,plate['size_m'][1]/2))
            else:
                kind=rule['surface'];top=item['size_m'][2];sx,sy=item['size_m'][:2]
                fields.update(surface_type=kind,centre_xy=tuple(item['centre']),top_m=top,surface_z=ground-top)
                if kind=='rectangle':fields.update(outline='rectangle',half_extent_m=(sx/2,sy/2))
                elif kind=='circle':fields.update(outline='circle',radius_m=min(sx,sy)/2)
                elif kind!='none':raise ValueError(f'Unknown surface type for shape {shape!r}: {kind!r}')
                if kind!='none':
                    # A pad's marking lies on its top and belongs to it, as in the canonical rule.
                    fields.update(contact_names=(own,),contact_prefixes=(f'{own}Mark',) if item['landable'] else ())
            surface=LandingSurface(landable=False,**fields)
            if surface.surface_type=='none':reason=rule.get('why','no flat top')
            elif surface.usable_reach_m<self.minimum:
                reason=f'flat top too small: the usable region reaches {surface.usable_reach_m:.2f} m from its middle, less than the {self.minimum:.2f} m a descent may start at'
            else:reason=None
            surface=replace(surface,landable=reason is None,reason=reason)
            if item['landable']:
                # The canonical rule for a pad is the reference; the general rule has to say the same thing about it.
                canonical=self.landing['touchdown']['landing_region_half_width_m']
                same=surface.landable and surface.outline=='rectangle' and surface.yaw_deg==0. and all(abs(value-canonical)<1e-9 for value in surface.usable_half_extent_m)
                if not same or abs(surface.top_m-item['size_m'][2])>1e-9:
                    raise ValueError(f'The general landing rule does not give the canonical region for {name}: {surface.describe()}')
            self.surfaces[name]=surface

    def __getitem__(self,name):return self.surfaces[name]
    def __contains__(self,name):return name in self.surfaces
    def get(self,name):return self.surfaces.get(name)
    def items(self):return self.surfaces.items()

    def owner(self,name):
        """Which object of the layout a simulator object belongs to: the object itself, a pad's marking, or a landing cap."""
        return self.cap_owner.get(name) or owner_of(self.map_config,self.layout,name)

    def touched(self,name,height_m):
        """The surface a contact was a touchdown on, or None: a contact with a landable surface's own top, made from above.
        `height_m` is the height of the contact above the ground. Anything else that is touched is a collision."""
        owner=self.owner(name);surface=self.surfaces.get(owner)
        if surface is None or not surface.landable or not surface.owns_contact(name):return None
        return surface if surface.from_above(height_m,self.landing['touchdown']['from_above_tolerance_m']) else None

    def describe(self):
        return {name:surface.describe() for name,surface in self.surfaces.items()}


def refusal(surface):
    """Why a landing on this object may not be asked for, or None."""
    return None if surface is not None and surface.landable else NO_SURFACE
