"""Where a human flight starts, and how it is written down. No simulator, no window.

A flight starts at a random place of a recording layout (never one of the layouts the checks are flown in), within one of
the distance bands from the named object (`distance_bands`: near, mid, far), facing a random way, and at least
`separation_from_check_starts_m` from every start that is flown for a check anywhere in the repository. The pilot is told
the sentence and nothing of where the object is.

Every recording layout also has a variant `<layout>_cap` with a flat round plate on the orange ball, which makes the ball a
landing surface (`with_caps`). The variants are built in memory: the map files are part of the frozen baseline's
fingerprints and are not changed.

A flight is written as the privileged teacher's flights are (scripts/visual_search.run_episode with --record): the two
pictures of every decision as PNG under ROOT/<scene>/<id>/, and the episode under ROOT/episodes/ (kept) or ROOT/rejected/
(not kept). Each decision also carries what the simulator knew (distance, bearing, what each camera showed), for analysis
only: the dataset builder keeps it under `meta`, and no model reads it. `teacher_action` is the pilot's action in the
policy's units (src/human_pilot/controls.TickLabel); `teacher_state` is `human`, or `stop` / `landed` for the decisions
after the pilot is done; `excluded` names why a decision is in no label window.
"""
import copy
import datetime
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/human_pilot.json'
CAP='_cap'      # a recording layout with the plate on the orange ball


def load(path=None):
    return json.loads(Path(path or CONFIG).read_text(encoding='utf-8'))


def check_starts(map_id):
    """Every start flown for a check, of one scene: each plan and set under configs/ but the plans of recorded episodes."""
    found=[]
    def walk(item,default):
        if isinstance(item,dict):
            if 'start_xy' in item and item.get('map',default)==map_id:found.append([float(v) for v in item['start_xy']])
            for value in item.values():walk(value,item.get('map',default))
        elif isinstance(item,list):
            for value in item:walk(value,default)
    for path in sorted((ROOT/'configs').rglob('*.json')):
        if path.name=='train_plan.json':continue
        try:data=json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,UnicodeDecodeError):continue
        walk(data,data.get('map') if isinstance(data,dict) else None)
    return found


def recording_layouts(settings):
    """Every layout a human flight may be flown in: the recording layouts and their variants with the ball's plate."""
    return list(settings['layouts'])+[layout+CAP for layout in settings['layouts']]


def with_caps(map_config,settings):
    """A copy of a map with a variant `<layout>_cap` of every recording layout: the same objects, the same scenery and
    lighting, and a flat round plate standing on the orange ball (configs/landing/surfaces.json, `cap`). The plate is
    placed as the interactive map's plate is (configs/mission/grounding_film_demo.json): it rests on the ball all round,
    its top a little above the ball's."""
    cap=settings['ball_cap'];config=copy.deepcopy(map_config);ball=cap['object']
    radius=config['objects'][ball]['size_m'][0]/2;plate=cap['plate_radius_m'];thickness=cap['thickness_m']
    # How far below the ball's top its surface lies at the plate's edge: the plate's underside stands there.
    below=radius-math.sqrt(radius**2-plate**2);lift=cap['ball_top_m'][config['id']]-below
    from src.visual_search.maps import object_position
    caps=config.setdefault('landing_caps',{});config.setdefault('structures',[])
    for layout in settings['layouts']:
        if ball not in config['layouts'][layout]:continue
        name=layout+CAP;config['layouts'][name]=dict(config['layouts'][layout])
        for item in config['structures']:
            if 'layouts' in item and layout in item['layouts']:item['layouts']=list(item['layouts'])+[name]
        if layout in config.get('layout_lighting',{}):config['layout_lighting'][name]=config['layout_lighting'][layout]
        plate_name=f'BallCap_{layout}'
        config['structures'].append({'name':plate_name,'shape':'cylinder','size_m':[2*plate,2*plate,thickness],'position':list(object_position(config,layout,ball)),
                                     'color':cap['color'],'lift_m':round(lift,3),'layouts':[name],'obstacle':False})
        caps[name]={ball:plate_name}
    return config


def targets_of(map_config,layout):
    """The objects of a layout, in the map's order, with their noun and whether a landing on them can be asked for."""
    from src.landing.surface import SurfaceSet
    surfaces=SurfaceSet(map_config,layout)
    return [{'name':name,'noun':map_config['objects'][name]['noun'],'landable':bool(surfaces[name].landable)} for name in map_config['layouts'][layout]]


def draw(settings,config,map_config,layout,target,task,seed,avoid=None,stamp=None,band=None):
    """One start for a flight: the canonical environment's episode fields, the sentence, and an id. Refuses a layout the
    checks are flown in. `band` names the distance band (the configuration's default when None)."""
    from src.mission.start import plan_starts
    if layout not in recording_layouts(settings):raise ValueError(f'layout {layout!r} is not a recording layout {recording_layouts(settings)}')
    band=band or settings['default_band']
    if map_config['id'] not in settings['scenes']:raise ValueError(f'scene {map_config["id"]!r} is not a recording scene {settings["scenes"]}')
    avoid=check_starts(map_config['id']) if avoid is None else avoid
    episode=plan_starts(config,map_config,layout,target,task,1,seed,settings['distance_bands'][band],yaw='random',avoid=avoid,
                        separation_m=settings['separation_from_check_starts_m'],instruction_id=settings['instructions'][task],
                        max_ticks=settings['max_ticks'],map_name=map_config['id'])[0]
    stamp=stamp or datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    episode.update(id=f'hp-{stamp}-{target}-{task}',kind='human',case='human',split='train',pilot='human',seed=seed,distance_band=band)
    return episode


def label_rows(steps):
    """How many decisions of a flight are labels, and why the others are not."""
    out={'decisions':len(steps),'labels':0,'pause':0,'below_safe':0,'backward':0,'over_bound':0,'model':0,'handover':0,'terminal':0}
    for step in steps:
        if step.get('excluded'):out[step['excluded']]+=1
        elif step.get('teacher_state') in ('stop','landed'):out['terminal']+=1
        else:out['labels']+=1
    return out


class Recorder:
    """The frames and the episode of one human flight."""
    def __init__(self,root,episode):
        import cv2
        self.cv2=cv2;self.root=Path(root);self.episode=episode;self.rel=f'{episode["map"]}/{episode["id"]}'
        self.folder=self.root/self.rel
        if self.folder.exists():shutil.rmtree(self.folder)
        for name in ('frontcamera','downcamera'):(self.folder/name).mkdir(parents=True,exist_ok=True)

    def frames(self,index,front,down):
        """The two pictures of decision `index`, exactly as a policy is given them; returns the image name."""
        name=f'{index:06d}.png'
        self.cv2.imwrite(str(self.folder/'frontcamera'/name),front);self.cv2.imwrite(str(self.folder/'downcamera'/name),down)
        return name

    def finish(self,summary,steps,kept,keep_rejected_frames=True):
        """Writes the episode (episodes/ when kept, rejected/ otherwise); returns its path."""
        folder=self.root/('episodes' if kept else 'rejected');folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'{self.episode["id"]}.json'
        path.write_text(json.dumps({'summary':summary,'traj_rel_dir':self.rel,'steps':steps}),encoding='utf-8')
        if not kept and not keep_rejected_frames and self.folder.exists():shutil.rmtree(self.folder)
        return path

    def discard(self):
        if self.folder.exists():shutil.rmtree(self.folder)


def precision_m(surface,landing=None):
    """How near the middle of its surface a landing has to come to be data: the canonical teacher's align radius (an
    eighth of a pad's width: where its descent starts), and never more than half of how far the surface's usable region
    reaches from its middle. Pad, cube and cylinder: 1.75 m; the plate on the ball: 1.3 m."""
    from src.visual_search.maps import load_landing
    landing=landing or load_landing()
    reach=surface['usable_radius_m'] if surface.get('outline')=='circle' else min(surface['usable_half_extent_m'])
    return min(landing['teacher']['align_radius_m'],reach/2)


def precise(reading,landing=None):
    """Did the landing come down within `precision_m` of the middle of the named surface?"""
    surface=reading.get('target_surface') or {};error=reading.get('horizontal_error_m')
    return bool(surface.get('landable') and error is not None and error<=precision_m(surface,landing)+1e-9)


def kept(reading,summary,factor,settings):
    """A human flight is data only if it did what the sentence asked by the surface evaluator's reading, the pilot said it
    was done (or landed; in takeover mode, accepted the model's own stop), nothing was hit, and the simulator kept up with real time. A landing is read by the surface
    evaluator's own touchdown (the named surface, inside its region, soft, then standing still), not by the canonical rule:
    that one reads the touchdown from the simulator's reported vertical speed, which can settle only after a hard contact
    (a hard landing counted) or not within the four decisions after a soft one (a soft landing missed). A landing must
    also be precise (`precise`): the surfaces are large, and data that lands anywhere on them teaches a coarse landing."""
    done=(reading.get('stable_physical_landing') and precise(reading)) if summary.get('task')=='land' else reading.get('success')
    return bool(done and summary['reason'] in ('human_stop','model_stop') and not summary.get('hit') and (factor is None or factor>=settings['real_time_minimum']))


def digest(frame):return hashlib.sha256(frame.tobytes()).hexdigest()
