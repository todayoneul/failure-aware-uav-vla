"""Markdown tables and a trajectory figure from evaluate_model.py results; no simulator or model."""
import argparse
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mission.geometry import world_to_pixel

COLORS={'blue_cone':(200,150,20),'orange_ball':(20,120,240),'colored_wall':(60,170,60),
        'near':(170,60,170),'medium':(170,60,170),'far':(170,60,170),'roof':(60,60,60)}
PANELS=[('harness','Earlier Near / Medium / Far coordinates (hint + generic text)'),
        ('prompt:hint+generic','Hint + generic text'),('prompt:hint+landmark|repeat','Hint + landmark text'),
        ('prompt:landmark-only','Landmark text only (no hint)'),
        ('conflict','Text names one landmark, hint points at the other'),
        ('fixed_hint','Hint always "straight ahead" + landmark text'),
        ('far|altitude','Long range (spawn start) and 10 m start altitude'),
        ('aligned','Start heading aligned with the target')]


def load(paths):
    trials=[];first=None
    for path in paths:
        data=json.loads((Path(path)/'results.json').read_text(encoding='utf-8'))
        first=first or (Path(path),data)
        for trial in data['trials']:
            # Where the vehicle actually ended, whatever target the trial was scored against.
            x,y=trial['final_position'][:2]
            name,distance=min(((item['id'],math.hypot(item['position'][0]-x,item['position'][1]-y)) for item in data['landmarks']),
                              key=lambda pair:pair[1])
            trial['nearest_landmark']={'id':name,'distance_m':distance}
            trials.append(trial)
    return first,trials


def stop_kind(trial):
    # The policy stops either with the LAND text or with an all-zero action; upstream treats both as its stop.
    steps=trial.get('steps') or []
    return 'LAND' if not steps or 'LAND' in steps[-1]['raw_output'] else 'zero action'


def outcome(trial):
    if trial['stop']:return f'{stop_kind(trial)} @ step {trial["stop"]["step"]}'
    return (trial['reason'] or trial['state']).split(' ')[0]


def within(trial,radius):return bool(trial['stop']) and trial['stop']['distance_m']<=radius


def trial_table(trials,radii):
    head='| Trial | Start -> target | Prompt | Initial | Steps | Outcome | Stop dist. | Min dist. | '+' | '.join(f'<={r:g} m' for r in radii)+' |'
    lines=[head,'|---|---|---|---:|---:|---|---:|---:|'+':---:|'*len(radii)]
    for t in trials:
        target=t['target']+(f' (hint: {t["hint_target"]})' if t.get('hint_target') else '')
        stop=f'{t["stop"]["distance_m"]:.1f} m' if t['stop'] else '-'
        lines.append(f'| {t["id"]} | {t["start"]} -> {target} | {t["prompt_condition"]} | {t["initial_distance_m"]:.1f} m | '
                     f'{t["executed_steps"]} | {outcome(t)} | {stop} | {t["minimum_distance_m"]:.1f} m | '
                     +' | '.join('O' if within(t,r) else 'X' for r in radii)+' |')
    return '\n'.join(lines)


def group_table(trials,radii,key,label):
    groups={}
    for t in trials:groups.setdefault(key(t),[]).append(t)
    head=f'| {label} | n | Model stop | '+' | '.join(f'SR<={r:g} m' for r in radii)+f' | Oracle<={radii[0]:g} m | Collision | Mean stop dist. |'
    lines=[head,'|---|---:|---:|'+'---:|'*len(radii)+'---:|---:|---:|']
    for name,items in groups.items():
        stops=[t['stop']['distance_m'] for t in items if t['stop']]
        lines.append(f'| {name} | {len(items)} | {len(stops)}/{len(items)} | '
                     +' | '.join(f'{sum(within(t,r) for t in items)}/{len(items)}' for r in radii)
                     +f' | {sum(t["minimum_distance_m"]<=radii[0] for t in items)}/{len(items)} | '
                     f'{sum((t["reason"] or "")=="collision" for t in items)}/{len(items)} | '
                     +(f'{np.mean(stops):.1f} m' if stops else '-')+' |')
    return '\n'.join(lines)


def cell_table(trials,radius):
    """Every run of the fixed-start prompt grid: stop distance (bold inside the radius) or how it ended."""
    prompts=['hint+generic','hint+landmark','landmark-only'];targets=[]
    for t in trials:
        if t['target'] not in targets:targets.append(t['target'])
    def short(t):
        if t['stop']:
            text=f'stop {t["stop"]["distance_m"]:.0f} m'
            return f'**{text}**' if within(t,radius) else text
        return (t['reason'] or t['state']).split(' ')[0]+f' (min {t["minimum_distance_m"]:.0f} m)'
    lines=['| Target (initial distance) | '+' | '.join(prompts)+' |','|---|'+'---|'*len(prompts)]
    for target in targets:
        items=[t for t in trials if t['target']==target]
        lines.append(f'| {target} ({items[0]["initial_distance_m"]:.0f} m) | '
                     +' | '.join('<br>'.join(short(t) for t in items if t['prompt_condition']==prompt) or '-' for prompt in prompts)+' |')
    return '\n'.join(lines)


def run_text(trial,radius):
    if trial['stop']:
        text=f'stop {trial["stop"]["distance_m"]:.1f} m'
        if trial['group']=='roof' and 'on_target_surface' in trial['stop']:
            # A roof target is only reached when the vehicle also comes down on that surface.
            gap=trial['stop']['below_target_surface_m']
            text+=', on the surface' if trial['stop']['on_target_surface'] else f', landed {abs(gap):.0f} m {"below" if gap>0 else "above"}'
        return f'**{text}**' if within(trial,radius) else text
    return (trial['reason'] or trial['state']).split(' ')[0]+f' (min {trial["minimum_distance_m"]:.1f} m)'


def runs_table(trials,radius,where=False):
    """One row per trial definition, every run listed; bold = model stop inside the radius."""
    order=[];groups={}
    for trial in trials:
        key=trial.get('base_id',trial['id'])
        if key not in groups:order.append(key);groups[key]=[]
        groups[key].append(trial)
    head='| Trial | Start -> target | Prompt | Start height | Initial | Runs: distance to the scored target |'+(' Ended nearest |' if where else '')
    lines=[head,'|---|---|---|---:|---:|---|'+('---|' if where else '')]
    for key in order:
        items=groups[key];first=items[0]
        target=first['target']+(f', hint -> {first["hint_target"]}' if first.get('hint_target') else '')
        target+=', hint fixed ahead' if first.get('hint_mode')=='ahead' else ''
        row=(f'| {key} | {first["start"]} -> {target} | {first["prompt_condition"]} | {first.get("start_clearance_m",float("nan")):.0f} m | '
             f'{first["initial_distance_m"]:.0f} m | '+' · '.join(run_text(t,radius) for t in items)+' |')
        if where:row+=' '+' · '.join(f'{t["nearest_landmark"]["id"]} {t["nearest_landmark"]["distance_m"]:.0f} m' for t in items)+' |'
        lines.append(row)
    return '\n'.join(lines)


def decoder_table(trials):
    """How often the action grammar had to replace a token, per decoder setting."""
    lines=['| Decoder | Trials | Decisions | Grammar replaced a token | Ended as invalid_action |','|---|---:|---:|---:|---:|']
    for name in ('free','grammar'):
        items=[t for t in trials if t.get('decoder','free')==name]
        if not items:continue
        steps=[step for t in items for step in t.get('steps') or []]
        lines.append(f'| {name} | {len(items)} | {len(steps)} | '
                     f'{sum(bool((step.get("decoder") or {}).get("intervened")) for step in steps) if name=="grammar" else "-"} | '
                     f'{sum((t["reason"] or "")=="invalid_action" for t in items)} |')
    return '\n'.join(lines)


def action_table(trials):
    lines=['| Trial | Most frequent raw actions (count) |','|---|---|']
    for t in trials:
        lines.append(f'| {t["id"]} | '+', '.join(f'`{action}` x{count}' for action,count in t['actions'][:4])+' |')
    return '\n'.join(lines)


def draw(run,trials,radius,path):
    image=cv2.imread(str(run/'map.png'));meta=json.loads((run/'map.json').read_text())
    scale=1.6;base=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    def pixel(point):
        u,v=world_to_pixel([point[0],point[1],-1.],meta)
        return int(round(u*scale)),int(round(v*scale))
    def chosen(trial,selector):
        for part in selector.split('|'):
            group,_,prompt=part.partition(':')
            if trial['group']==group and (not prompt or trial['prompt_condition']==prompt):return True
        return False
    panels=[]
    for selector,title in PANELS:
        panel=base.copy();ends=[]
        for trial in (t for t in trials if chosen(t,selector)):
            color=COLORS.get(trial['target'],(200,200,200))
            center=pixel(trial['target_position'])
            edge=pixel([trial['target_position'][0]+radius,trial['target_position'][1]])
            cv2.circle(panel,center,int(math.hypot(edge[0]-center[0],edge[1]-center[1])),color,1,cv2.LINE_AA)
            cv2.drawMarker(panel,center,color,cv2.MARKER_TILTED_CROSS,12,2,cv2.LINE_AA)
            points=[pixel(p) for p in trial.get('trajectory') or []]
            for a,b in zip(points,points[1:]):cv2.line(panel,a,b,color,2,cv2.LINE_AA)
            if points:
                cv2.circle(panel,points[0],5,(255,255,255),-1,cv2.LINE_AA);cv2.circle(panel,points[0],5,(40,40,40),1,cv2.LINE_AA)
                end=points[-1]
                if trial['stop']:cv2.circle(panel,end,6,color,-1,cv2.LINE_AA);cv2.circle(panel,end,6,(255,255,255),1,cv2.LINE_AA)
                else:cv2.rectangle(panel,(end[0]-5,end[1]-5),(end[0]+5,end[1]+5),(40,40,230),2)
                ends.append(end)
        cv2.rectangle(panel,(0,0),(panel.shape[1],30),(245,243,240),-1)
        cv2.putText(panel,f'{title}  (n = {len(ends)})',(10,21),0,.58,(40,40,40),1,cv2.LINE_AA)
        panels.append(panel)
    rows=[np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)]
    figure=np.vstack(rows)
    legend=np.full((44,figure.shape[1],3),(245,243,240),dtype=np.uint8)
    cv2.putText(legend,f'line colour = scored target | white dot: start | X + ring: target and {radius:g} m radius | filled dot: model stop (LAND) | red square: ended without a stop | top = +X, right = +Y',
                (10,28),0,.55,(40,40,40),1,cv2.LINE_AA)
    cv2.imwrite(str(path),np.vstack([figure,legend]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+')
    parser.add_argument('--figure');parser.add_argument('--markdown')
    args=parser.parse_args()
    (run,data),trials=load(args.runs)
    radii=json.loads((ROOT/'configs/evaluation_protocol.json').read_text(encoding='utf-8'))['success_radii_m']
    grid=[t for t in trials if t['group'] in ('prompt','repeat')]
    def group(*names):return [t for t in trials if t['group'] in names]
    text='\n\n'.join(['### Prompt grid from the `facing` start',cell_table(grid,radii[0]) if grid else '-',
        '### Earlier coordinates',runs_table(group('harness'),radii[0]),
        '### Hint against text',runs_table(group('conflict','fixed_hint'),radii[0],where=True),
        '### Range, altitude and heading',runs_table(group('far','altitude','aligned'),radii[0]),
        '### Roof target',runs_table(group('roof'),radii[0]),
        '### Instruction only',runs_table(group('instruction'),radii[0],where=True),
        '### Decoder',decoder_table(trials),
        '### Per trial',trial_table(trials,radii),'### By prompt condition (landmark targets, both starts)',
        group_table([t for t in trials if t['group']!='harness' and not t.get('hint_target')],radii,lambda t:t['prompt_condition'],'Prompt'),
        '### By group',group_table(trials,radii,lambda t:t['group'],'Group'),'### Raw actions',action_table(trials)])
    if args.markdown:Path(args.markdown).write_text(text+'\n',encoding='utf-8')
    else:print(text)
    if args.figure:draw(run,trials,radii[0],Path(args.figure))


if __name__=='__main__':main()
