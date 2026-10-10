"""Write the small record of a surface-flight run: what was asked, what was reached, what the evaluator read, what was checked.

  python scripts/surface_flights_summary.py RUN_FOLDER --name NAME [--output outputs/arbitrary_start/summary]

Reads results.json of scripts/surface_flights.py and writes NAME.json (one compact record per episode, no decisions) and
NAME.md (a table). The flights themselves stay in the run folder, which is not part of the repository.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.mission.start import StartState,reproduction

READ=('success','landed_on','collision','contact_from_above','correct_object','wrong_object','on_top','inside_region','acceptable_touchdown',
      'physical_landing','finalizer_latched','horizontal_error_m','vertical_speed_mps','touchdown_decision','canonical_touchdown_decision',
      'canonical_vertical_speed_mps','canonical_success','decisions_over_surface')


def compact(record):
    episode=record['episode'];item={'id':record['id'],'layout':episode['layout'],'target':episode['target'],'task':episode['task'],
                                   'instruction':episode.get('instruction'),'pilot':episode.get('pilot'),'why':episode.get('why'),
                                   'flown':record['flown'],'passed':record['passed'],'checks':record['checks']}
    if not record['flown']:return {**item,'refusal':record.get('refusal')}
    wanted=StartState(**record['start']);got=StartState(**record['start_reached']);summary=record['summary'];reading=record['surface_landing']
    surface=reading['target_surface']
    return {**item,'start':record['start'],'start_reached':record['start_reached'],'start_error':reproduction(wanted,got),'spawn':record['spawn'],
            'reason':summary['reason'],'stopped':summary['stopped'],'decisions':summary['steps'],'mean_decision_s':summary.get('mean_decision_s'),
            'finalizer_path':(summary.get('finalizer') or {}).get('path'),'surface':{key:surface.get(key) for key in ('surface','landable','top_m','usable_radius_m','usable_half_extent_m')},
            'read':{key:reading.get(key) for key in READ},'words':record.get('words'),'pilot_phases':record.get('pilot_phases')}


def table(name,results,records):
    flown=[item for item in records if item['flown']];passed=sum(item['passed'] for item in records)
    lines=[f'# {name}','',f'Plan `{Path(results["plan"]).name}`, policy `{results["policy"]}`, evaluator `{results["evaluator"]}`. '
           f'{len(records)} episodes, {len(flown)} flown, {passed} with every check passed.','',
           '| Episode | Sentence | Start asked (x, y, yaw, height) | Start error (xy m / yaw deg / height m) | What happened | Landed on | Distance from the middle | Inside region | Physical landing | Latched | Mission | Checks |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for item in records:
        if not item['flown']:
            lines.append(f'| {item["id"]} | (none: refused) | - | - | not flown: {item["refusal"]} | - | - | - | - | - | - | {"PASS" if item["passed"] else "FAIL"} |');continue
        start=item['start'];error=item['start_error'];read=item['read'];yes=lambda flag:'yes' if flag else 'no'
        happened=f'collision with {read["collision"]}' if read['collision'] else item['reason']
        failed=[row[0] for row in item['checks'] if not row[3]]
        lines.append(f'| {item["id"]} | {item["instruction"]} | {start["x"]:g}, {start["y"]:g}, {start["yaw_deg"]:g}, {start["height_m"]:g} | '
                     f'{error["xy_error_m"]:.4f} / {error["yaw_error_deg"]:.4f} / {error["height_error_m"]:+.3f} | {happened} ({item["decisions"]} decisions) | '
                     f'{read["landed_on"] or "-"} | {"-" if read["horizontal_error_m"] is None else format(read["horizontal_error_m"],".2f")+" m"} | '
                     f'{yes(read["inside_region"]) if read["landed_on"] else "-"} | {yes(read["physical_landing"])} | {yes(read["finalizer_latched"])} | '
                     f'{"SUCCESS" if read["success"] else "failed"} | {"PASS" if item["passed"] else "FAIL: "+", ".join(failed)} |')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run');parser.add_argument('--name',required=True);parser.add_argument('--output',default='outputs/arbitrary_start/summary')
    args=parser.parse_args();folder=Path(args.run) if Path(args.run).is_absolute() else ROOT/args.run
    results=json.loads((folder/'results.json').read_text(encoding='utf-8'));records=[compact(record) for record in results['episodes']]
    output=Path(args.output) if Path(args.output).is_absolute() else ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    verified=[json.loads(line) for line in (folder/'frozen_verification.jsonl').read_text(encoding='utf-8').splitlines()] if (folder/'frozen_verification.jsonl').exists() else []
    (output/f'{args.name}.json').write_text(json.dumps({'plan':Path(results['plan']).name,'policy':results['policy'],'evaluator':results['evaluator'],
                                                       'start_tolerance':results['start_tolerance'],
                                                       'frozen_verifications':[{key:item[key] for key in ('when','record','passed','utc')} for item in verified],
                                                       'episodes':records},indent=1)+'\n',encoding='utf-8',newline='\n')
    (output/f'{args.name}.md').write_text(table(args.name,results,records),encoding='utf-8',newline='\n')
    print(table(args.name,results,records))


if __name__=='__main__':main()
