"""Named scene landmarks; positions come from simulator bounding boxes, never guessed coordinates."""
import colorsys
import json
from pathlib import Path
import numpy as np

GENERIC_DESCRIPTION='Find the selected target location.'
GENERIC_NOUN='the selected target location'
# `instruction` prompt mode: what to find and what to do with it, with no direction word.
INSTRUCTION_TEMPLATE='Land on top of {noun}. Fly around, find it with your camera, then fly straight to it and land.'
PROMPT_MODES=('hint','description','instruction')
FOOTPRINT_MARGIN_M=1.
# Object kinds are read from simulator object names; anything else stays undescribed.
KINDS=(('Cone','cone'),('Cylinder','cylinder'),('Ball','ball'),('Cube','block'))


def instruction_for(target):
    # Keep the exact delimiters expected by the upstream wrapper; only the text
    # between them (the object description) reaches the model.
    description=(target.description if target is not None else None) or GENERIC_DESCRIPTION
    return f'The target is 0 degrees from you. {description} Please control the drone.'


def command_for(target):
    """Free-form instruction for the `instruction` prompt mode."""
    return INSTRUCTION_TEMPLATE.format(noun=(target.noun if target is not None else None) or GENERIC_NOUN)


def prompt_arguments(target,mode):
    """How one prompt mode reaches the model: (direction_hint, freeform text or None)."""
    if mode not in PROMPT_MODES:raise ValueError(f'Unknown prompt mode: {mode!r}')
    return mode=='hint',command_for(target) if mode=='instruction' else None


def color_name(bgr):
    """Coarse colour word for one rendered pixel; None when it is too dark to tell."""
    blue,green,red=(float(v)/255 for v in bgr)
    hue,saturation,value=colorsys.rgb_to_hsv(red,green,blue)
    if value<.2:return None
    if saturation<.25:return 'white' if value>.85 else 'gray'
    degrees=hue*360
    for limit,name in ((20,'red'),(45,'orange'),(70,'yellow'),(170,'green'),(260,'blue'),(340,'purple')):
        if degrees<limit:return name
    return 'red'


def describe_surface(records,point,bgr=None):
    """Words for the object under a selected point: its kind from the simulator name, its colour
    from the clicked map pixel. None for the ground or an object of unknown kind."""
    best=None
    for record in records:
        kind=next((word for key,word in KINDS if key in record.get('name','')),None)
        try:
            box=record['bbox'];center=[float(box['center'][k]) for k in 'xyz'];size=[float(box['size'][k]) for k in 'xyz']
        except (KeyError,TypeError,ValueError):continue
        if kind is None or not all(np.isfinite(center+size)) or min(size)<=0:continue
        if any(abs(point[i]-center[i])>size[i]/2+FOOTPRINT_MARGIN_M for i in (0,1)):continue
        # NED: the top of the box is its smallest z. The selected surface is the top of the highest object here.
        gap=abs(point[2]-(center[2]-size[2]/2))
        if gap<=1. and (best is None or gap<best[0]):best=(gap,record['name'],kind)
    if best is None:return None
    color=color_name(bgr) if bgr is not None else None
    phrase=f'{color} {best[2]}' if color else best[2]
    article='an' if phrase[0] in 'aeiou' else 'a'
    return {'object':best[1],'noun':f'the {phrase}','description':f'The target is the top of {article} {phrase}.'}


def load_landmarks(path,records):
    """Resolve configured landmarks against audited world-axis boxes; skip those absent from the scene."""
    boxes={}
    for record in records:
        try:
            box=record['bbox'];center=np.array([box['center'][k] for k in 'xyz'],dtype=float)
            size=np.array([box['size'][k] for k in 'xyz'],dtype=float)
        except (KeyError,TypeError,ValueError):continue
        if np.isfinite(center).all() and np.isfinite(size).all() and (size>0).all():
            boxes[record['name']]=(center-size/2,center+size/2)
    landmarks=[]
    for entry in json.loads(Path(path).read_text(encoding='utf-8')):
        if not all(name in boxes for name in entry['objects']):continue
        low=np.min([boxes[name][0] for name in entry['objects']],axis=0)
        high=np.max([boxes[name][1] for name in entry['objects']],axis=0)
        center=(low+high)/2
        # NED: the smallest z is the top of the object.
        landmarks.append({'id':entry['id'],'name':entry['name'],'description':entry['description'],'noun':entry.get('noun'),
                          'position':[float(center[0]),float(center[1]),float(low[2])],
                          'minimum':low.tolist(),'maximum':high.tolist(),'objects':list(entry['objects']),
                          # A wall is seen from above as part of a roof; such a landmark is chosen by key only.
                          'click':entry.get('click',True)})
    return landmarks


def landmark_at(landmarks,point):
    for landmark in landmarks:
        if landmark.get('click',True) and all(landmark['minimum'][i]-FOOTPRINT_MARGIN_M<=point[i]<=landmark['maximum'][i]+FOOTPRINT_MARGIN_M for i in (0,1)):
            return landmark
    return None
