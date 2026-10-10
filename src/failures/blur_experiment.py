"""Gaussian blur as an experimental condition: the fixed severities, the failure between the camera and the policy, the
order of the flights, what is measured about an image, and how the flights are compared. No simulator and no model.

Nothing here is given to a policy. The policy receives the two frames (blurred or not) and its sentence; the name of the
condition, the kernel, the sigma and every number computed from an image stay with the experiment's log.
"""
import itertools
import json
import math
import random
from pathlib import Path
from .gaussian_blur import GaussianBlurFailure,SEVERITIES

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/failures/gaussian_blur_characterization.json'
CAMERAS=('front','down')
SEVERITY_ORDER=('clean','low','medium','high')
# The canonical classifier's stage (scripts/freeze_gen_v3.classify) -> the class reported for this experiment.
CLASSES={'COLLISION':'B9_COLLISION','SEARCH':'B1_SEARCH_FAILURE','WRONG_TARGET':'B8_WRONG_TARGET','GROUNDING':'B2_GROUNDING_FAILURE',
         'APPROACH':'B3_APPROACH_FAILURE','ALIGNMENT':'B4_ALIGNMENT_FAILURE','PHYSICAL_LANDING':'B7_STABLE_LANDING_FAILURE',
         'FINALIZER':'B7_STABLE_LANDING_FAILURE','LANDING':'B7_STABLE_LANDING_FAILURE'}
ALL_CLASSES=('B1_SEARCH_FAILURE','B2_GROUNDING_FAILURE','B3_APPROACH_FAILURE','B4_ALIGNMENT_FAILURE','B5_DESCENT_FAILURE','B6_TOUCHDOWN_FAILURE',
             'B7_STABLE_LANDING_FAILURE','B8_WRONG_TARGET','B9_COLLISION','B10_TIMEOUT','B11_RUNTIME_ERROR','B12_OTHER')


def load(path=None):
    """The experiment's configuration. Its severities must be the ones the interactive demo's keys apply."""
    config=json.loads(Path(path or CONFIG).read_text(encoding='utf-8'))
    if tuple(config['conditions'])!=SEVERITY_ORDER:raise ValueError('conditions must be clean, low, medium, high, in that order')
    if config['conditions']['clean'] is not None:raise ValueError('clean is the camera frame itself')
    for name in SEVERITY_ORDER[1:]:
        spec=config['conditions'][name]
        if SEVERITIES[name]!=(spec['kernel'],spec['sigma']):raise ValueError(f'{name}: the experiment and the demo disagree on the blur')
    if tuple(config['cameras'])!=CAMERAS:raise ValueError('this experiment blurs Front and Down alike')
    return config


class Injector:
    """The failure for one condition, applied to what the camera gave before anything prepares it for the policy.

    Clean hands the camera's own arrays on, the same objects. A blurred condition gives new arrays, Front and Down with the
    same kernel and sigma. It has no state and no randomness: the same frame always gives the same result."""
    def __init__(self,name):
        if name not in SEVERITY_ORDER:raise ValueError(f'Unknown condition: {name!r}')
        self.name=name;self.failure=None if name=='clean' else GaussianBlurFailure(enabled=True,severity=name)
        self.kernel,self.sigma=(None,None) if name=='clean' else SEVERITIES[name]

    def apply(self,frames):
        if self.failure is None:return frames
        if set(frames)!=set(CAMERAS):raise ValueError('expected the Front and the Down frame')
        return {camera:self.failure.apply(frames[camera]) for camera in frames}


def quality(frame):
    """Sharpness numbers of one frame, for the analysis only: variance of the Laplacian, mean gradient magnitude, and the
    mean standard deviation of the grey levels in 16 x 16 blocks."""
    import cv2
    import numpy as np
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    height,width=gray.shape;blocks=gray[:height//16*16,:width//16*16].reshape(height//16,16,width//16,16).swapaxes(1,2).reshape(-1,256)
    return {'laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_64F).var()),'gradient_magnitude':float(np.hypot(gx,gy).mean()),
            'local_contrast':float(blocks.std(axis=1).mean())}


# --- the order of the flights ---------------------------------------------------------------------------------------------
def flight_order(config,start_ids,phase='main'):
    """Every (start, condition) in the order it is flown: start by start, the four conditions of a start in a fixed,
    balanced order.

    Main run: each of the 24 orders of four conditions is given to as many starts as possible (twice each for 48), so every
    condition is flown first, second, third and last equally often. Repeat run: whole Latin squares of orders, so the same
    holds for a multiple of four starts. Which start gets which order is drawn from the phase's seed."""
    names=list(config['conditions']);seed=config['order']['seed'] if phase=='main' else config['repeat']['seed'];rng=random.Random(f'{seed}|{phase}')
    if phase=='main':
        orders=[list(order) for order in itertools.permutations(names)]*math.ceil(len(start_ids)/24)
    else:
        # The rotations of one order form a Latin square; the six squares are the orders that begin with the first condition.
        squares=[[list(base[shift:]+base[:shift]) for shift in range(len(names))] for base in ([names[0],*rest] for rest in itertools.permutations(names[1:]))]
        rng.shuffle(squares);orders=[order for square in squares for order in square]
        orders=orders[:math.ceil(len(start_ids)/4)*4]
    orders=orders[:len(start_ids)] if phase!='main' else orders
    rng.shuffle(orders)
    return [(start,condition) for start,order in zip(start_ids,orders) for condition in order]


def positions(flights,conditions):
    """How often each condition is flown first, second, third and last of its start."""
    table={name:[0]*len(conditions) for name in conditions};seen={}
    for start,condition in flights:
        table[condition][seen.get(start,0)]+=1;seen[start]=seen.get(start,0)+1
    return table


# --- the repeat subset -------------------------------------------------------------------------------------------------
def start_facts(episode,scene,within_m=60.):
    """What a canonical start holds, from its plan alone (no flight is looked at). The same reading as the canonical tables'
    (scripts/gen_v3c.selection): the object beside the named one, and the related objects the first view shows within
    `within_m` when the named one is outside it."""
    from src.visual_search.gen_v3c import relation_of
    target=episode['target'];beside=episode.get('distractor') or episode.get('neighbour');hidden=episode['plan']['acquired_tick']!=0
    shown={relation_of(scene,target,item['name']) for item in episode.get('others_in_view',[]) if item['distance_m']<=within_m}-{None}
    relation=relation_of(scene,target,beside) if beside else None;first=shown if hidden else set()
    return {'same_color':relation=='same_color' or 'same_color' in first,'same_shape':relation=='same_shape' or 'same_shape' in first,'beside':relation,
            'first_view':sorted(first),'initially_invisible':hidden,'band':episode['band'],'task':episode['task'],'role':episode.get('role')}


def repeat_subset(config,episodes,scenes):
    """The starts flown a second time, chosen from the plan before any flight: for each wanted kind in turn, the first
    starts of a seeded shuffle that are of that kind and not yet chosen."""
    rule=config['repeat'];rng=random.Random(f'{rule["selection_seed"]}|subset');order=[episode['id'] for episode in episodes];rng.shuffle(order)
    by_id={episode['id']:episode for episode in episodes};facts={name:start_facts(by_id[name],scenes[by_id[name]['map']]) for name in order}
    tests={'same_color':lambda f:f['same_color'] and not f['same_shape'],'same_shape':lambda f:f['same_shape'] and not f['same_color'],
           'initially_invisible':lambda f:f['initially_invisible'],'far':lambda f:f['band']=='far',
           'landing_heavy':lambda f:f['task']=='land' and f['band']=='near' and not f['initially_invisible'],
           'easy_control':lambda f:f['task']=='approach' and f['band']=='near' and not f['initially_invisible']}
    chosen=[]
    for kind,count in rule['composition'].items():
        picked=[name for name in order if name not in [item['id'] for item in chosen] and tests[kind](facts[name])][:count]
        if len(picked)<count:raise ValueError(f'not enough starts of kind {kind}')
        chosen+=[{'id':name,'chosen_as':kind,**facts[name]} for name in picked]
    return chosen


# --- reading and comparing flights ---------------------------------------------------------------------------------------
def failure_class(record):
    """One class for a failed flight: the first stage it did not pass, as the canonical classifier reads it."""
    stage,how=record.get('failure'),record.get('mechanism','')
    if not stage:return ''
    if stage=='STOP':
        # An approach that came within reach of the object and did not end as an approach should.
        return 'B10_TIMEOUT' if how=='did not stop' else 'B3_APPROACH_FAILURE' if how=='stopped outside the radius' else 'B12_OTHER'
    if stage=='DESCENT':return 'B6_TOUCHDOWN_FAILURE' if record.get('descent_started') else 'B5_DESCENT_FAILURE'
    return CLASSES.get(stage,'B12_OTHER')


def cell(record):
    """A flight in one word for the table of starts."""
    if record is None:return 'E_RUNTIME'
    if record['success']:return 'S'
    return 'F_'+failure_class(record).split('_',1)[1].replace('_FAILURE','')


def paired(before,after):
    """Starts by what happened in two conditions: kept, lost, gained, failed in both; and the exact McNemar p-value of the
    two discordant counts (two-sided binomial, p = 0.5)."""
    counts={'S_S':0,'S_F':0,'F_S':0,'F_F':0}
    for start in before:
        if start in after:counts[('S' if before[start] else 'F')+'_'+('S' if after[start] else 'F')]+=1
    lost,gained=counts['S_F'],counts['F_S'];n=lost+gained
    p=1. if n==0 else min(1.,2*sum(math.comb(n,k) for k in range(0,min(lost,gained)+1))/2**n)
    return {**counts,'mcnemar_exact_p':p}


def onset(successes):
    """The lowest severity at which a start failed, and whether a higher severity then succeeded again.

    `successes`: condition -> bool, for one start. 'CLEAN' means the clean flight itself failed, so nothing is attributed."""
    order=[name for name in SEVERITY_ORDER if name in successes];first=next((name for name in order if not successes[name]),None)
    later=order[order.index(first)+1:] if first else []
    return {'onset':'NONE' if first is None else first.upper(),'non_monotonic':any(successes[name] for name in later)}


def retention(clean,value):
    """Change against clean in percentage points, and the share of the clean success rate that is kept."""
    return {'drop_points':round(100*(value-clean),1),'retention':round(value/clean,3) if clean else None}


def auroc(lower,higher):
    """How well a number separates two groups of frames: the probability that one drawn from `higher` exceeds one from `lower`."""
    values=sorted([(value,0) for value in lower]+[(value,1) for value in higher]);rank_sum=0.;index=0
    while index<len(values):
        end=index
        while end<len(values) and values[end][0]==values[index][0]:end+=1
        rank_sum+=sum(flag for _,flag in values[index:end])*(index+end+1)/2;index=end
    return (rank_sum-len(higher)*(len(higher)+1)/2)/(len(lower)*len(higher)) if lower and higher else None
