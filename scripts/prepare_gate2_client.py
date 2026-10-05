"""Apply AeroVLA's documented client encoding fix to this isolated env only."""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'outputs/compatibility'
spec = importlib.util.find_spec('airsim')
path = Path(spec.origin).parent / 'client.py'
original = path.read_text()
old = ", pack_encoding = 'utf-8', unpack_encoding = 'utf-8'"
assert original.count(old) == 1, 'Expected exactly one upstream encoding argument pair'
updated = original.replace(old, '')
root.joinpath('airsim-client-encoding.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(True), updated.splitlines(True), fromfile='airsim/client.py.original',
    tofile='airsim/client.py.smoke-test')))
root.joinpath('airsim-client-patch-manifest.json').write_text(json.dumps({
    'path': str(path), 'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
    'patched_sha256': hashlib.sha256(updated.encode()).hexdigest(),
    'reason': 'AeroVLA pinned troubleshooting: remove VehicleClient encoding kwargs',
}, indent=2))
path.write_text(updated)
print('Patched isolated Gate 2 AirSim client; original source checkouts untouched')
