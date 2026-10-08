"""Record, or check, the fingerprints of a finished model's files so later work cannot change them unnoticed.

  freeze_baseline.py write  NAME PATH [PATH ...]   hash every file under the paths into outputs/generalization/NAME_frozen.json
  freeze_baseline.py verify NAME                   recompute and compare; exits 1 on any difference

Paths that are not in Git (checkpoints, datasets, raw run folders) are hashed where they lie. No simulator, no model.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Files a run folder keeps rewriting or that say nothing about the result.
SKIP=('worker.log','worker-errors.log','live.json','live_front.png','live_down.png','live_result.json','reload_check.json')


def fingerprint(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1<<20),b''):digest.update(block)
    return {'bytes':path.stat().st_size,'sha256':digest.hexdigest()}


def collect(paths):
    files={}
    for item in paths:
        base=ROOT/item
        if not base.exists():raise SystemExit(f'{item} does not exist')
        for path in sorted([base] if base.is_file() else base.rglob('*')):
            if path.is_file() and path.name not in SKIP and 'mission_config' not in path.parts:
                files[path.relative_to(ROOT).as_posix()]=fingerprint(path)
    return files


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('write','verify'));parser.add_argument('name');parser.add_argument('paths',nargs='*')
    args=parser.parse_args();record=ROOT/f'outputs/generalization/{args.name}_frozen.json'
    if args.command=='write':
        if record.exists():raise SystemExit(f'{record} exists; a frozen record is not rewritten')
        files=collect(args.paths);record.write_text(json.dumps({'paths':args.paths,'files':files},indent=1)+'\n',encoding='utf-8',newline='\n')
        print(f'{len(files)} files, {sum(f["bytes"] for f in files.values())/2**20:.1f} MiB -> {record.relative_to(ROOT)}');return
    saved=json.loads(record.read_text(encoding='utf-8'));now=collect(saved['paths'])
    changed=sorted(name for name in saved['files'] if name in now and now[name]!=saved['files'][name])
    missing=sorted(set(saved['files'])-set(now));added=sorted(set(now)-set(saved['files']))
    print(json.dumps({'files':len(saved['files']),'changed':changed,'missing':missing,'added':added},indent=1))
    if changed or missing or added:sys.exit(1)


if __name__=='__main__':main()
