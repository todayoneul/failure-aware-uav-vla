"""How far a teacher's labels are from the student's stop, measured with the student's own stop rule. No simulator, no torch.

What ends a flight (scripts/visual_search.run_episode): the action that is flown at a decision is a stop when
`src/aerovla_oft/spec.is_stop` says so (every axis within `stop_threshold` of its range: forward 0.08 m, down 0.08 m,
yaw 0.0336 rad), and `stop_ticks` (4) such decisions in a row end the flight. The policy predicts a chunk of four actions
and one is flown per decision, so the four are the first actions of four inferences in a row; the teacher's own flights
are ended by the same lines. Nothing here restates that rule: `is_stop` is imported and called.

`stop_ratio` says how far an action is from that rule: the largest of its axes, each as a multiple of the axis's stop
limit. 1 or less is a stop. A landing has no stop in the air, so every label before the touchdown has to keep a margin
(`safe_ratio`, fixed in the stage's configuration before anything is recorded): a student whose output is a little smaller
than its label is otherwise read as stopping. An approach ends with a stop in the air; its labels slow down toward that
stop on purpose and are counted, not held to the margin.

A decision is terminal when the flight is over at it: in a landing the vehicle stands on a surface (`landed`), in an
approach the teacher's label is its stop (`teacher_state` is `stop`). A forced move (`perturbed`) is no label and breaks
a run of decisions.
"""
import collections
from src.aerovla_oft.spec import is_stop

# The canonical teacher's state names (src/visual_search/generalization.SearchTeacher) as the phases of a mission.
PHASES={'search':'SEARCH','approach':'APPROACH','final':'SURFACE_APPROACH','climb':'CLIMB','align':'ALIGN','descend':'DESCENT','landed':'TERMINAL','stop':'TERMINAL'}
BANDS=((10.,20.,'10-20 m'),(20.,32.,'20-32 m'),(32.,44.01,'32-44 m'))


def stop_limits(oft):
    """Per axis, the largest size `is_stop` still calls a stop."""
    return [oft['stop_threshold']*(high-low) for low,high in oft['action_bounds'].values()]


def stop_ratio(action,oft):
    return max(abs(value)/limit for value,limit in zip(action,stop_limits(oft)))


def terminal(step,task):
    return bool(step.get('landed')) if task=='land' else step.get('teacher_state')=='stop'


def label_of(step):
    return step.get('teacher_action',step.get('action'))


def band_of(distance):
    return next((name for low,high,name in BANDS if low<=distance<high),'other')


def percentile(values,share):
    if not values:return None
    ordered=sorted(values);return ordered[min(len(ordered)-1,int(share*len(ordered)))]


def spread(values):
    return None if not values else {'n':len(values),'min':round(min(values),3),'p5':round(percentile(values,.05),3),'median':round(percentile(values,.5),3),
                                    'p95':round(percentile(values,.95),3)}


def audit_episode(steps,task,oft,rule,window):
    """One flight's labels against the rule. `rule`: safe_ratio and near_stop_ratio. Returns counts, the ratio of every label
    that is not terminal (with its state), and where the rule is broken."""
    rows=[];stop_run=near_run=0;out={'actions':0,'terminal':0,'nonterminal':0,'forced':0,'nonterminal_stop':0,'below_safe':0,'stop_windows':0,'near_stop_windows':0,
                                    'terminal_not_a_stop':0,'broken':[]}
    for index,step in enumerate(steps):
        if step.get('perturbed'):out['forced']+=1;stop_run=near_run=0;continue
        action=label_of(step);out['actions']+=1
        if terminal(step,task):
            out['terminal']+=1;out['terminal_not_a_stop']+=not is_stop(action,oft);stop_run=near_run=0;continue
        ratio=stop_ratio(action,oft);stopped=is_stop(action,oft);state=step.get('teacher_state','?')
        out['nonterminal']+=1;out['nonterminal_stop']+=stopped;out['below_safe']+=ratio<rule['safe_ratio']
        rows.append({'step':index,'state':state,'ratio':ratio,'stop':stopped,'distance_m':step.get('distance_m')})
        stop_run=stop_run+1 if stopped else 0;near_run=near_run+1 if ratio<rule['near_stop_ratio'] else 0
        # Every run of `window` decisions that would end the flight, and every run that comes close to it.
        out['stop_windows']+=stop_run>=window;out['near_stop_windows']+=near_run>=window
        if ratio<rule['safe_ratio'] and len(out['broken'])<5:out['broken'].append({'step':index,'state':state,'action':[round(value,4) for value in action],'ratio':round(ratio,3)})
    out['rows']=rows;return out


def audit(episodes,oft,rule,window):
    """Every episode's labels against the rule, summed and broken down. `episodes`: dicts with id, task, kind (pad, cube,
    cylinder), start_distance_m and steps. A landing passes with no label before the touchdown that is a stop, none below the
    safe ratio and no run of `window` stops; an approach with no stop before the teacher's own and no such run."""
    keys=('actions','terminal','nonterminal','forced','nonterminal_stop','below_safe','stop_windows','near_stop_windows','terminal_not_a_stop')
    total=collections.Counter();groups=collections.defaultdict(collections.Counter);ratios=collections.defaultdict(list);broken=[];failed=[]
    for episode in episodes:
        task=episode['task'];one=audit_episode(episode['steps'],task,oft,rule,window);kind=episode.get('kind') or 'other'
        names=[('task',task.upper()),('kind',f'{kind} {task.upper()}'),('band',f'{task.upper()} {band_of(episode.get("start_distance_m",-1.))}')]
        counts={key:one[key] for key in keys};counts['episodes']=1;total.update(counts)
        for name in names:groups[name].update(counts)
        for row in one['rows']:
            phase=PHASES.get(row['state'],row['state'].upper())
            for name in names+[('state',f'{task.upper()} {phase}')]:ratios[name].append(row['ratio'])
            groups[('state',f'{task.upper()} {phase}')].update(nonterminal=1,below_safe=row['ratio']<rule['safe_ratio'],near_stop=row['ratio']<rule['near_stop_ratio'],
                                                               nonterminal_stop=row['stop'])
        bad=one['nonterminal_stop'] or one['stop_windows'] or (task=='land' and one['below_safe'])
        if bad:
            failed.append(episode['id'])
            if len(broken)<20:broken.append({'id':episode['id'],'task':task,'kind':kind,'nonterminal_stop':one['nonterminal_stop'],'below_safe':one['below_safe'],
                                            'stop_windows':one['stop_windows'],'examples':one['broken']})
    table={}
    for (group,name),counts in sorted(groups.items()):
        table.setdefault(group,{})[name]={**dict(counts),'stop_ratio':spread(ratios[(group,name)])}
    land=groups[('task','LAND')];approach=groups[('task','APPROACH')]
    checks=[('landing labels before the touchdown that are a stop',land['nonterminal_stop'],0),
            (f'landing labels before the touchdown below {rule["safe_ratio"]} x the stop rule',land['below_safe'],0),
            (f'runs of {window} stops before the touchdown of a landing',land['stop_windows'],0),
            ("approach labels that are a stop before the teacher's own stop",approach['nonterminal_stop'],0),
            (f'runs of {window} stops before the stop of an approach',approach['stop_windows'],0)]
    return {'rule':{**rule,'stop_ticks':window,'stop_limits':dict(zip(oft['action_bounds'],stop_limits(oft)))},'total':dict(total),'by':table,
            'checks':[{'what':what,'count':int(count),'allowed':allowed,'passed':count<=allowed} for what,count,allowed in checks],
            'passed':all(count<=allowed for _,count,allowed in checks),'episodes_that_break_the_rule':len(failed),'examples':broken}


def recorded(root):
    """The episodes of a recorded dataset folder (ROOT/episodes/*.json) in the form `audit` reads."""
    import json
    from pathlib import Path
    episodes=[]
    for path in sorted((Path(root)/'episodes').glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'));summary=record['summary'];target=summary['target']
        kind=summary.get('surface_kind') or ('pad' if target.endswith('_pad') else 'cube' if target.endswith('_cube') else 'cylinder' if target.endswith('_cylinder') else 'other')
        episodes.append({'id':summary['id'],'task':summary.get('task','approach'),'kind':kind,'start_distance_m':summary.get('start_distance_m',summary.get('initial_distance_m',-1.)),
                         'steps':record['steps']})
    return episodes
