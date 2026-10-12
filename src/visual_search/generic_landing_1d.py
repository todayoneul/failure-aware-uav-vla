"""Phase B, stage 1d: the starts of the final-descent data and of its checks. No simulator, no torch.

One thing is corrected (configs/generic_landing_stage1d.json): what the policy does in the last two metres over a top.
Every recorded episode is a landing that is pushed (src/landing/teacher.RecoveryPushes: flown, never a label) into one of
four states and then left to the stage's teacher:

  low_hover       D1  low over the top, inside the part of it where a descent goes on, nearly still
  descent_decay   D2  the same place, reached by a descent that dies away
  edge_low        D3  low over the edge or a corner of the top, or beside it: outside that part
  edge_drift      D4  low after part of a descent and carried on, level, out of that part

A push of stage 1d is two forced moves: an `edge` move of stage 1c that sets how far from the middle the vehicle passes,
and the move that makes the state. Both are due at places, so a tested policy is pushed as the teacher is.

The sets, in the order they are drawn (each keeps away from the ones before it and from every earlier start):

  pilot           twenty-four starts on which stage 1c's checkpoint and the candidates are compared
  qualification   twenty starts the chosen candidate flies before the sealed gates of stage 1c are opened
  teacher_pilot   twenty forced cube landings the teacher flies before anything is recorded
  train_plan      the recorded episodes

No same-colour episode is recorded in this stage; the same-colour starts of the pilot and the qualification only watch
what stage 1c taught. The gates and the regression set are stage 1c's files, untouched.
"""
import json
import random
from pathlib import Path
from src.landing.surface import SurfaceSet
from src.mission import start as starts
from .maps import load_landing
from . import generic_landing as stages
from . import generic_landing_1c as before

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/generic_landing_stage1d.json'
SETS=('pilot','qualification','teacher_pilot')
STATES=('low_hover','descent_decay','edge_low','edge_drift')
NAMES={'low_hover':'D1 CENTERED_LOW_HOVER','descent_decay':'D2 CENTERED_DESCENT_DECAY','edge_low':'D3 EDGE_LOW','edge_drift':'D4 EDGE_AFTER_PARTIAL_DESCENT'}


def load(path=None):
    stage=stages.load(path or CONFIG);stage.setdefault('file','configs/generic_landing_stage1d.json' if path is None else str(path));return stage


def files_of(stage):return {name:stages.folder_of(stage)/f'{name}.json' for name in SETS}
def train_plan_of(stage):return stages.train_plan_of(stage)
def gates_of(stage):return before.load(stage['gates']['config'])


def pushes_of(stage,category,rng,surface,landing):
    """The two forced moves of one episode of a state, drawn inside the ranges the stage fixes."""
    rule=stage['final_descent']['forced'][category];edge=rule['edge'];pick=lambda source,key:round(rng.uniform(*source[key]),2);count=lambda key:rng.randint(*rule[key])
    miss=pick(edge,'miss_m')
    first={'when':'edge','from_m':pick(edge,'from_m'),'min_above_m':edge['min_above_m'],'miss_m':miss,'side':rng.choice((-1,1)),'forward_m':edge['forward_m'],'ticks':edge['ticks']}
    within=round(miss+rule['within_beyond_miss_m'],2)
    if category in ('low_hover','edge_low'):
        second={'when':'low_hover','within_m':within,'hold_m':pick(rule,'hold_m'),'hold_ticks':count('hold_ticks'),'band_m':rule['band_m'],'gain':rule['gain'],'down_m':rule['down_m'],'ticks':rule['ticks']}
    elif category=='descent_decay':
        second={'when':'descent_decay','within_m':within,'from_m':pick(rule,'from_m'),'down_m':rule['down_m'],'start_m':pick(rule,'start_m'),'ratio':pick(rule,'ratio'),'decay_ticks':count('decay_ticks'),
                'ticks':rule['ticks']}
    else:
        second={'when':'edge_drift','within_m':within,'from_m':pick(rule,'from_m'),'down_m':rule['down_m'],'out_m':pick(rule,'out_m'),'forward_m':rule['forward_m'],'ticks':rule['ticks']}
    return [first,second]


def draw_final(config,oft,stage,scene,layout,target,category,band,visible,seed,split,name,excluded):
    """A landing start whose teacher flight is pushed into `category` and lands, or None. As a forced start of stage 1c,
    and the last push must let go in the air: a vehicle that stands on the top has nothing left to do."""
    episode=before.draw_recovery(config,oft,stage,scene,layout,target,category,band,visible,seed,split,name,excluded,make=pushes_of)
    if episode is None or len(episode['plan']['let_go'])!=2 or episode['plan']['let_go'][-1]['above_top_m']<stage['final_descent']['let_go_above_top_min_m']:return None
    episode['state']=NAMES[category];return episode


class Builder(before.Builder):
    def recovery(self,split,prefix,layouts,target,category,band,visible,attempts=400):
        for attempt in range(attempts):
            name=self.stage['scenes'][(self.counter+attempt)%len(self.stage['scenes'])];layout=layouts[((self.counter+attempt)//len(self.stage['scenes']))%len(layouts)]
            episode=draw_final(self.config,self.oft,self.stage,self.scenes[name],layout,target,category,band,visible,self.seed,split,
                               f'{prefix}-{{seed:05d}}-{target}-{band}_{category}',self.excluded(name));self.seed+=1
            if episode:episode['set']=split;return self.keep(episode)
        raise RuntimeError(f'No {split} start: {target} {category} {band}')


def plain(builder,split,prefix,layouts,kind,rows):
    """Plain landing starts of one kind of surface (stage 1's kind): [[band, first view holds the object]]."""
    stage=builder.stage;episodes=[]
    for index,(band,visible) in enumerate(rows):episodes+=builder.landing(split,prefix,layouts,stage['targets'][kind][index%2],band,bool(visible))
    return episodes


def build_pilot(builder):
    """Twenty-four starts: twelve forced cube landings (three for each state), six same-colour landings, three pad and
    three cylinder landings."""
    stage=builder.stage;layouts=stage['layouts']['evaluation'];prefix=stage['prefix']['pilot'];rule=stage['pilot']
    episodes=before.build_recovery(builder,'pilot',prefix,layouts,{'cube':{state:rule['critical_per_state'] for state in STATES}})
    episodes+=before.build_same_color(builder,'pilot',prefix,layouts,rule['same_color'])
    for kind in ('pad','cylinder'):episodes+=plain(builder,'pilot',prefix,layouts,kind,rule['plain_rows'])
    return episodes


def build_qualification(builder):
    """Twenty starts: ten cube landings (one forced for each state, six plain), four same-colour landings, three pad and
    three cylinder landings."""
    stage=builder.stage;layouts=stage['layouts']['evaluation'];prefix=stage['prefix']['qualification'];rule=stage['qualification']
    episodes=before.build_recovery(builder,'qualification',prefix,layouts,{'cube':{state:rule['forced_per_state'] for state in STATES}})
    episodes+=plain(builder,'qualification',prefix,layouts,'cube',rule['cube_rows'])
    episodes+=before.build_same_color(builder,'qualification',prefix,layouts,rule['same_color'])
    for kind in ('pad','cylinder'):episodes+=plain(builder,'qualification',prefix,layouts,kind,rule['plain_rows'])
    return episodes


def build_teacher_pilot(builder):
    stage=builder.stage
    return before.build_recovery(builder,'teacher_pilot',stage['prefix']['teacher_pilot'],stage['layouts']['evaluation'],{'cube':{state:stage['teacher_pilot']['per_state'] for state in STATES}})


def earlier_sets(stage,strict=False):
    """The files of stage 1c's sets a start of this stage keeps away from: all of them for a recorded start (`strict`), all
    but the ones named in `checks.may_be_near` for a check start."""
    near={(ROOT/path).resolve() for path in stage['checks'].get('may_be_near',[])}
    return [path for path in before.files_of(gates_of(stage)).values() if strict or Path(path).resolve() not in near]


def strict_points(stage):
    """Every earlier start a check start of this stage keeps away from: what the checkpoints of this chain were trained on,
    every earlier check start but the ones named in `checks.may_be_near`, and stage 1c's sets (its sealed gates, its
    regression set, its pilot)."""
    points=before.loose_points(stage,'checks')
    for path in earlier_sets(stage):
        if Path(path).exists():points+=[(e['map'],e['start_xy']) for e in json.loads(Path(path).read_text(encoding='utf-8'))['episodes']]
    return points


def build_all(config,oft,stage):
    """Every set of the stage. Checks first, then the recorded episodes, which keep away from every start that is flown
    for a check anywhere in the repository (stage 1c's sealed gates among them) and from this stage's own checks."""
    sets={};strict=strict_points(stage);checks=Builder(config,oft,stage,list(strict),stage['seeds']['pilot'][0])
    sets['pilot']=build_pilot(checks)
    checks.seed=stage['seeds']['qualification'][0];sets['qualification']=build_qualification(checks)
    checks.seed=stage['seeds']['teacher_pilot'][0];sets['teacher_pilot']=build_teacher_pilot(checks)
    own=checks.points[len(strict):];plan={};flown=before.check_points(stage)+own
    for split in ('train','val'):
        builder=Builder(config,oft,stage,flown+[point for other in plan.values() for point in before.firsts(other)],stage['seeds'][f'{split}_final'][0])
        plan[split]=before.build_recovery(builder,split,stage['prefix'][f'{split}_final'],stage['layouts']['train'],stage['training']['final' if split=='train' else 'val_final'])
    sets['train_plan']=plan
    return sets


def describe(stage,which,episodes):
    text={'pilot':'twenty-four starts flown once by stage 1c\'s checkpoint (update 2000) and by each candidate: twelve forced cube landings (three for each state), six same-colour landings, three '
                  'pad and three cylinder landings; stage 1b\'s checkpoint flies the twelve forced ones, for the comparison of the descent',
          'qualification':'twenty starts the chosen candidate flies once before the sealed gates of stage 1c are opened: ten cube landings (one forced for each state and six plain), four '
                          'same-colour landings, three pad and three cylinder landings',
          'teacher_pilot':'twenty forced cube landings (five for each state) the teacher flies before anything is recorded'}
    return {'description':f'Generic landing, stage 1d, {which}: {text[which]}. Written before any episode of stage 1d was recorded ({stage["file"]}). Flown in layouts of the two '
                          'training scenes that no episode was recorded in; seeds and starts are disjoint from every earlier plan and set.',
            'seeds':stage['seeds'][which],'bands_m':stage['bands_m'],'episodes':episodes}
