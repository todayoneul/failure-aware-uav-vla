"""The policy in another process, for takeover mode: the checkpoint is loaded in WSL (CUDA), the pilot runs on Windows.

  python src/human_pilot/remote.py --checkpoint DIR --port 8995 --ready FILE      (in WSL: loads the checkpoint, then serves)

One request is one decision of the canonical loop: the two 256 x 256 pictures, the sentence and the motion vector (for a
checkpoint trained with one), exactly what scripts/visual_search.run_episode gives `policy.infer`; the answer is what
`AeroVLAOFT.infer` returns. Nothing else is sent: no target, no position. The pictures go as raw bytes and the answer
carries their sha256, so every decision is recorded with proof that the model was given the pictures that were saved.

When the checkpoint is loaded the server writes FILE (its port, a key for this session, what was loaded); the pilot
connects with that key. The server serves one pilot and ends when the pilot closes the connection or asks it to.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def sha256(data):return hashlib.sha256(data).hexdigest()


def ask(connection,front,down,instruction,proprio=None):
    """One decision from the server: the model's answer, after checking it was given these two pictures."""
    import numpy as np
    pictures=[np.ascontiguousarray(front),np.ascontiguousarray(down)]
    connection.send_bytes(json.dumps({'instruction':instruction,'proprio':proprio,
                                      'pictures':[{'shape':list(item.shape),'dtype':str(item.dtype)} for item in pictures]}).encode())
    for item in pictures:connection.send_bytes(item.tobytes())
    answer=json.loads(connection.recv_bytes())
    if 'error' in answer:raise RuntimeError('the policy server failed: '+answer['error'])
    sent={name:sha256(item.tobytes()) for name,item in zip(('front','down'),pictures)}
    if answer['input_sha256']!=sent:raise RuntimeError(f'the policy server was given other pictures: {answer["input_sha256"]} != {sent}')
    return answer


def serve(model,listener,log=print):
    """Answers one pilot's decisions until it closes the connection or sends {"quit": true}."""
    import numpy as np
    with listener.accept() as connection:
        log('pilot connected')
        while True:
            try:header=json.loads(connection.recv_bytes())
            except EOFError:log('pilot closed the connection');return
            if header.get('quit'):log('pilot asked to stop');return
            pictures=[np.frombuffer(connection.recv_bytes(),dtype=item['dtype']).reshape(item['shape']).copy() for item in header['pictures']]
            try:
                answer=model.infer(pictures[0],pictures[1],header['instruction'],header.get('proprio'))
                answer=dict(answer,input_sha256={name:sha256(item.tobytes()) for name,item in zip(('front','down'),pictures)})
            except Exception:answer={'error':traceback.format_exc()}
            connection.send_bytes(json.dumps(answer).encode())


class RemotePolicy:
    """The pilot's side: connects once the server has said the checkpoint is loaded. One decision at a time."""
    def __init__(self,ready_path):
        self.path=Path(ready_path);self.connection=None;self.info=None;self.lock=threading.Lock()

    def status(self):
        """'ready', 'loading', or the server's error."""
        if self.connection is not None:return 'ready'
        if not self.path.exists():return 'loading'
        info=json.loads(self.path.read_text(encoding='utf-8'))
        if info.get('error'):return info['error']
        from multiprocessing.connection import Client
        self.connection=Client(('127.0.0.1',info['port']),authkey=bytes.fromhex(info['key']));self.info=info
        return 'ready'

    def infer(self,inputs):
        with self.lock:return ask(self.connection,inputs['front'],inputs['down'],inputs['instruction'],inputs['proprio'])

    def close(self):
        if self.connection is None:return
        with self.lock:
            try:self.connection.send_bytes(json.dumps({'quit':True}).encode())
            except OSError:pass
            self.connection.close();self.connection=None


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--port',type=int,default=8995);parser.add_argument('--ready',required=True)
    args=parser.parse_args();ready=Path(args.ready);ready.parent.mkdir(parents=True,exist_ok=True)
    log=lambda text:print(time.strftime('%H:%M:%S'),text,flush=True)
    try:
        from multiprocessing.connection import Listener
        from scripts.visual_search import MANIFEST
        from src.aerovla_oft.model import AeroVLAOFT
        from src.mission.oft_runner import loaded_modules
        checkpoint=Path(args.checkpoint);saved=json.loads((checkpoint/'manifest.json').read_text(encoding='utf-8'))['config']
        log(f'loading {checkpoint}')
        # As the interactive missions load it (src/mission/oft_runner.Loader): the checkpoint must have brought its trained weights.
        model=AeroVLAOFT(MANIFEST,saved,checkpoint=str(checkpoint));info=loaded_modules(model,saved)
        key=os.urandom(16).hex();listener=Listener(('127.0.0.1',args.port),authkey=bytes.fromhex(key))
    except Exception:
        ready.write_text(json.dumps({'error':traceback.format_exc()}),encoding='utf-8');raise
    temporary=ready.with_suffix('.tmp')
    temporary.write_text(json.dumps({'port':args.port,'key':key,'checkpoint':str(checkpoint),'loaded':info,'pid':os.getpid()}),encoding='utf-8')
    temporary.replace(ready);log(f'ready on port {args.port}: {info["name"]}, grounding {info["grounding"]}, loaded in {info["load_seconds"]:.0f} s')
    try:serve(model,listener,log)
    finally:
        listener.close()
        try:ready.unlink()
        except OSError:pass


if __name__=='__main__':main()
