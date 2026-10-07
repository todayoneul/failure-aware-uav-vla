"""Tables for visual-search runs: one column per run directory, same rows for every policy."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

CASES={'A':'A visible, centred','B':'B visible, at the edge','C':'C not visible at the start','D':'D lost after a forced turn','L':'L both in view, sentence names one'}


def load(path):
    data=json.loads((Path(path)/'results.json').read_text())
    episodes=[episode for episode in data['episodes'] if not episode.get('error')]
    steps={}
    file=Path(path)/'steps.jsonl'
    if file.exists():
        for line in file.read_text().splitlines():
            row=json.loads(line);steps.setdefault(row['episode'],[]).append(row)
    return data['policy'],episodes,steps


def mean(values):
    values=[value for value in values if value is not None]
    return statistics.mean(values) if values else float('nan')


def case_rows(episodes):
    rows={}
    for case in CASES:
        items=[episode for episode in episodes if episode['case']==case]
        if not items:continue
        turned=[episode['first_turn_toward_target'] for episode in items if episode['first_turn_toward_target'] is not None]
        found=[episode for episode in items if not episode['initially_seen']]
        rows[case]={'n':len(items),'success':sum(episode['success'] for episode in items),'reached':sum(episode['reached'] for episode in items),
                    'stopped':sum(episode['stopped'] for episode in items),'collisions':sum(episode['reason']=='collision' for episode in items),
                    'mean_final_m':mean(episode['final_distance_m'] for episode in items),
                    'turned_toward':f'{sum(turned)}/{len(turned)}' if turned else '-',
                    'acquired':f'{sum(episode["acquired_step"] is not None for episode in found)}/{len(found)}' if found else '-',
                    'mean_yaw_swept_deg':mean(episode['yaw_swept_deg'] for episode in items),
                    'max_height_change_m':max(episode['height_range_m'][1]-episode['height_range_m'][0] for episode in items)}
    return rows


# Full range of each action axis, to compare policies whose actions differ in size.
RANGES={'baseline':(5.,10.,2.2),'oft':(1.,1.,.42),'teacher':(1.,1.,.42)}


def motion(policy,episodes,steps):
    """How the command and the vehicle change from one decision to the next."""
    span=RANGES[policy];change=[];turn=[];travel=[];speed=[];period=[];latency=[];clipped=0;stationary=moving=0
    for episode in episodes:
        rows=steps.get(episode['id'],[])
        for previous,current in zip(rows,rows[1:]):
            change.append(sum(abs(a-b)/r for a,b,r in zip(current['action'],previous['action'],span))/3)
            delta=current['yaw_rad']-previous['yaw_rad'];turn.append(abs(math.degrees(math.atan2(math.sin(delta),math.cos(delta)))))
            seconds=current['epoch']-previous['epoch'];period.append(seconds)
            metres=math.dist(current['position'][:2],previous['position'][:2]);travel.append(metres)
            # A decision that asked for forward motion but left the vehicle (nearly) where it was.
            if previous['action'][0]>.2*span[0]:
                moving+=1;stationary+=metres/max(seconds,1e-6)<.1
            speed.append(metres/max(seconds,1e-6))
        latency+=[row['inference_ms'] for row in rows if 'inference_ms' in row]
        clipped+=sum(any(abs(value)>1 for value in row['normalised']) for row in rows if 'normalised' in row)
    return {'decisions':sum(len(steps.get(episode['id'],[])) for episode in episodes),'seconds_per_decision':mean(period),
            'inference_ms':mean(latency),'action_change_fraction_of_range':mean(change),'vehicle_turn_per_decision_deg':mean(turn),
            'largest_turn_per_decision_deg':max(turn) if turn else float('nan'),'travel_per_decision_m':mean(travel),
            'mean_speed_mps':sum(travel)/max(sum(period),1e-6),'speed_std_mps':statistics.pstdev(speed) if speed else float('nan'),
            'predictions_outside_range':clipped,
            'forward_decisions_left_standing':f'{stationary}/{moving}'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+');parser.add_argument('--json')
    args=parser.parse_args();columns=[];output={}
    for run in args.runs:
        policy,episodes,steps=load(run);name=f'{policy} ({Path(run).name})'
        columns.append((name,case_rows(episodes),motion(policy,episodes,steps)));output[name]={'cases':columns[-1][1],'motion':columns[-1][2]}
    print('| Test | '+' | '.join(name for name,_,_ in columns)+' |');print('|---|'+'---|'*len(columns))
    for case,label in CASES.items():
        if not any(case in rows for _,rows,_ in columns):continue
        def cell(rows):
            if case not in rows:return '-'
            r=rows[case]
            return (f'{r["success"]}/{r["n"]} stopped within radius; reached {r["reached"]}/{r["n"]}; mean final {r["mean_final_m"]:.1f} m; '
                    f'turned toward {r["turned_toward"]}; acquired {r["acquired"]}; collisions {r["collisions"]}')
        print(f'| {label} | '+' | '.join(cell(rows) for _,rows,_ in columns)+' |')
    for key,label in (('mean_yaw_swept_deg','Mean yaw swept (deg)'),('max_height_change_m','Largest height change (m)')):
        for case in CASES:
            if any(case in rows for _,rows,_ in columns):
                print(f'| {label}, {case} | '+' | '.join(f'{rows[case][key]:.1f}' if case in rows else '-' for _,rows,_ in columns)+' |')
    for key,label in (('decisions','Decisions'),('seconds_per_decision','Seconds per decision'),('inference_ms','Inference (ms)'),
                      ('action_change_fraction_of_range','Mean action change between decisions (fraction of axis range)'),
                      ('vehicle_turn_per_decision_deg','Mean heading change per decision (deg)'),
                      ('largest_turn_per_decision_deg','Largest heading change in one decision (deg)'),
                      ('travel_per_decision_m','Mean travel per decision (m)'),('mean_speed_mps','Mean speed (m/s)'),
                      ('speed_std_mps','Speed spread between decisions (m/s)'),('predictions_outside_range','Predictions outside the action range'),
                      ('forward_decisions_left_standing','Forward decisions after which the vehicle had not moved')):
        print(f'| {label} | '+' | '.join(f'{m[key]:.2f}' if isinstance(m[key],float) else str(m[key]) for _,_,m in columns)+' |')
    if args.json:Path(args.json).write_text(json.dumps(output,indent=1))


if __name__=='__main__':main()
