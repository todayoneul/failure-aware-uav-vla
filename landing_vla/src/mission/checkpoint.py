"""What can be said about a mission's checkpoint before a model is loaded: it exists, what it is, and whether it is the
frozen baseline with every fingerprint unchanged. No torch, no simulator; runs on the Windows side before the demo starts.
"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REQUIRED=('manifest.json','head.pt','oft/adapter_model.safetensors','oft/adapter_config.json')


def frozen_state(relative,name,root=ROOT):
    """Compare the files of a frozen record (outputs/generalization/NAME_frozen.json) with what is on disk now.

    The record holds the checkpoint and the source and configuration files the evaluation was flown with, so a change to
    any of them is reported, not only a change to the weights."""
    record=Path(root)/f'outputs/generalization/{name}_frozen.json'
    if not record.exists():return {'state':'no record','record':None}
    saved=json.loads(record.read_text(encoding='utf-8'))
    if relative not in saved['paths']:return {'state':'custom','record':record.relative_to(root).as_posix()}
    import scripts.freeze_baseline as freeze
    original=freeze.ROOT;freeze.ROOT=Path(root)
    try:now=freeze.collect(saved['paths'])
    except SystemExit as error:return {'state':'changed','record':record.relative_to(root).as_posix(),'missing':[str(error)]}
    finally:freeze.ROOT=original
    changed=sorted(item for item in saved['files'] if item in now and now[item]!=saved['files'][item])
    missing=sorted(set(saved['files'])-set(now));added=sorted(set(now)-set(saved['files']))
    return {'state':'changed' if changed or missing or added else 'verified','record':record.relative_to(root).as_posix(),
            'files':len(saved['files']),'changed':changed,'missing':missing,'added':added}


def inspect(checkpoint,frozen_name=None,root=ROOT):
    """Describe a checkpoint directory. `problems` stop a mission; `warnings` are shown and the mission may go on."""
    root=Path(root);path=Path(checkpoint);path=path if path.is_absolute() else root/path
    try:relative=path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:relative=path.as_posix()
    result={'checkpoint':relative,'exists':path.is_dir(),'problems':[],'warnings':[],'frozen':{'state':'not checked'}}
    if not result['exists']:result['problems'].append(f'No checkpoint directory at {relative}');return result
    absent=[name for name in REQUIRED if not (path/name).is_file()]
    if absent:result['problems'].append(f'Checkpoint is incomplete: missing {", ".join(absent)}');return result
    config=json.loads((path/'manifest.json').read_text(encoding='utf-8'))['config'];grounding=config.get('grounding') or {}
    result.update(name=config.get('name'),grounding=grounding.get('type','none'),grounding_position=grounding.get('position'),
                  chunk_size=config['chunk_size'],execute_horizon=config['execute_horizon'],tick_s=config['tick_s'],
                  proprio=bool(config['proprio']['enabled']),action_axes=list(config['action_bounds']))
    if frozen_name:
        result['frozen']=frozen_state(relative,frozen_name,root);state=result['frozen']['state']
        if state=='changed':
            result['problems'].append('The frozen baseline no longer matches its recorded fingerprints: '
                                      +', '.join(result['frozen'].get('changed',[])+result['frozen'].get('missing',[])+result['frozen'].get('added',[])))
        elif state!='verified':
            result['warnings'].append(f'{relative} is not the frozen baseline ({frozen_name}); this is a custom checkpoint and nothing about it is verified')
    if result['grounding']!='film':
        result['warnings'].append(f'This checkpoint has no FiLM module (grounding: {result["grounding"]})')
    return result
