"""Download only the two authorized, revision-pinned integration models."""
import hashlib
import json
import os
import time
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/integration'
CACHE = Path(os.environ.get('UAV_VLA_HOME', Path.home() / 'uav-vla-smoke')) / 'models/hf'
OUT.mkdir(parents=True, exist_ok=True)
result = {'started': time.time(), 'models': []}
for name, repo, patterns in (
    ('base', 'openvla/openvla-7b', ['*.json', '*.py', '*.safetensors', 'tokenizer.model']),
    ('adapter', 'XuPeng23/AerialVLA', ['aero_vla/adapter_config.json', 'aero_vla/adapter_model.safetensors']),
):
    meta = json.loads((ROOT / 'configs' / f'{name}-model.json').read_text(encoding='utf-8-sig'))
    print(f'DOWNLOAD {repo} revision={meta["sha"]}', flush=True)
    path = Path(snapshot_download(repo, revision=meta['sha'], allow_patterns=patterns,
                                  cache_dir=CACHE, max_workers=3))
    files = []
    for entry in meta['siblings']:
        file = path / entry['rfilename']
        if not file.is_file():
            continue
        assert file.stat().st_size == entry['size'], str(file)
        digest = hashlib.sha256(file.read_bytes()).hexdigest() if file.stat().st_size < 1024**2 else None
        # Stream large files; never allocate an entire shard in host RAM.
        if digest is None:
            h = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024**2), b''):
                    h.update(block)
            digest = h.hexdigest()
        if entry.get('lfs'):
            assert digest == entry['lfs']['sha256'], str(file)
        files.append({'name': entry['rfilename'], 'bytes': file.stat().st_size, 'sha256': digest})
    result['models'].append({'repo': repo, 'revision': meta['sha'], 'snapshot': str(path), 'files': files})
    (OUT/'model-downloads.json').write_text(json.dumps(result, indent=2))
    print(f'VERIFIED {repo} {sum(f["bytes"] for f in files)} bytes', flush=True)
result['finished'] = time.time()
(OUT/'model-downloads.json').write_text(json.dumps(result, indent=2))
print('DOWNLOADS_COMPLETE', flush=True)
