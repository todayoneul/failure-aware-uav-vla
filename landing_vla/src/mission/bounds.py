"""Filter audited world-axis boxes and fit an observation camera to their union."""
import math
import re
import numpy as np

HELPER=re.compile(r'sky|light|fog|weather|camera|reflection|volume|drone|player|hud|manager|gamestate|gamesession|gamemode|worldinfo|navdata|menu|unrealscene|blocksmap',re.I)


def scene_bounds(records):
    valid=[];excluded=[]
    for record in records:
        reason=None
        try:
            box=record['bbox'];center=np.array([box['center'][k] for k in 'xyz'],dtype=float)
            size=np.array([box['size'][k] for k in 'xyz'],dtype=float)
            if HELPER.search(record['name']):reason='non-playable helper actor'
            elif not np.isfinite(center).all() or not np.isfinite(size).all():reason='nonfinite box'
            elif (size<.001).any():reason='empty/degenerate box'
            elif max(size)>500:reason='oversized box (>500m), including unbounded floor'
            elif max(abs(center))>5000:reason='extreme coordinate (>5000m)'
        except (KeyError,TypeError,ValueError):reason='box query failed or malformed'
        if reason:excluded.append({'name':record['name'],'reason':reason})
        else:valid.append((record,center,size))
    if len(valid)>=5:
        centers=np.array([item[1][:2] for item in valid]);median=np.median(centers,axis=0)
        radii=np.linalg.norm(centers-median,axis=1);radius=np.median(radii)
        cutoff=max(100.,radius+6*np.median(abs(radii-radius)),4*np.median([max(x[2]) for x in valid]))
        kept=[]
        for item,distance in zip(valid,radii):
            if distance>cutoff:excluded.append({'name':item[0]['name'],'reason':f'spatial outlier ({distance:.2f}m > {cutoff:.2f}m)'})
            else:kept.append(item)
        valid=kept
    if not valid:raise ValueError('No finite playable geometry; full map cannot be inferred')
    minimum=np.min([c-s/2 for _,c,s in valid],axis=0)
    maximum=np.max([c+s/2 for _,c,s in valid],axis=0)
    return {'minimum':minimum.tolist(),'maximum':maximum.tolist(),
            'included':[item[0]['name'] for item in valid],'excluded':excluded,
            'scope':'finite Blocks geometry union; oversized Ground is not the playable extent'}


def fit_camera(bounds,view='top',zoom=1.,pan=(0,0),width=640,height=360,fov=80):
    low=np.array(bounds['minimum']);high=np.array(bounds['maximum']);extent=high-low
    center=(low+high)/2
    center[:2]+=np.asarray(pan)*max(extent[:2])*.05/zoom
    half_h=math.radians(fov)/2;half_v=math.atan(math.tan(half_h)*height/width)
    if view=='top':
        # With pitch -90, screen horizontal is world Y and vertical is world X.
        distance=1.12*max(extent[1]/(2*math.tan(half_h)),extent[0]/(2*math.tan(half_v)))/zoom
        position=[center[0],center[1],low[2]-distance];rpy=[0,-90,0]
    else:
        radius=np.linalg.norm(extent)/2
        distance=1.12*radius/math.sin(min(half_h,half_v))/zoom
        direction=np.array([.5,-.5,math.sqrt(.5)])
        position=(center-direction*distance).tolist();rpy=[0,-45,-45]
    return {'position':list(map(float,position)),'rpy':rpy,'zoom':zoom,'view':view}
