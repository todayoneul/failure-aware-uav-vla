"""Named scene landmarks; positions come from simulator bounding boxes, never guessed coordinates."""
import json
from pathlib import Path
import numpy as np

GENERIC_DESCRIPTION='Find the selected target location.'
FOOTPRINT_MARGIN_M=1.


def instruction_for(target):
    # Keep the exact delimiters expected by the upstream wrapper; only the text
    # between them (the object description) reaches the model.
    description=(target.description if target is not None else None) or GENERIC_DESCRIPTION
    return f'The target is 0 degrees from you. {description} Please control the drone.'


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
        landmarks.append({'id':entry['id'],'name':entry['name'],'description':entry['description'],
                          'position':[float(center[0]),float(center[1]),float(low[2])],
                          'minimum':low.tolist(),'maximum':high.tolist(),'objects':list(entry['objects'])})
    return landmarks


def landmark_at(landmarks,point):
    for landmark in landmarks:
        if all(landmark['minimum'][i]-FOOTPRINT_MARGIN_M<=point[i]<=landmark['maximum'][i]+FOOTPRINT_MARGIN_M for i in (0,1)):
            return landmark
    return None
