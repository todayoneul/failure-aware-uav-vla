"""Files for the Windows viewer, written off the flight loop; the viewer's keys, read off it.

The canonical decision loop has no time to spare (about 0.51 s per 0.5 s decision in the recorded evaluation flights), and
a file written from WSL to the Windows drive costs milliseconds. Everything the display needs is therefore queued and
written by one background thread (where a file is replaced, only its newest version); the same thread reads the control file. The
flight loop only hands data over and reads the last control state from memory.

The Gaussian-blur switch of the earlier demos is kept: off, a frame is passed on as the very same array.
"""
import hashlib
import json
import threading
import time
from pathlib import Path
from src.failures.control import read_control
from src.failures.gaussian_blur import GaussianBlurFailure,SEVERITIES
from src.integration.blur_demo_support import publish_image


def sha256(frame):return hashlib.sha256(frame.tobytes()).hexdigest()


class Feed:
    def __init__(self,output,period_s=.05):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True);self.period=period_s
        self.control=read_control(self.output/'control.json');self.control_warning=None
        self.failure=GaussianBlurFailure();self.last_chase=0.;self.chase_ready=False
        self.lock=threading.Lock();self.pending={};self.lines=[];self.wake=threading.Event();self.idle=threading.Event();self.idle.set()
        self.closed=False;self.errors=[]
        self.thread=threading.Thread(target=self._run,name='mission-feed',daemon=True);self.thread.start()

    # --- handed over by the flight loop: returns at once -------------------------------------------------------
    def _put(self,name,item):
        with self.lock:
            # A replaced file moves to the end of the queue, so a telemetry file never precedes the frames it names.
            self.pending.pop(name,None);self.pending[name]=item;self.idle.clear()
        self.wake.set()

    def json(self,name,data):
        """Replace a JSON file. Serialised here, so the caller may go on changing its own data."""
        self._put(name,('text',json.dumps(data)))

    def image(self,name,frame):
        """Replace a PNG. The array must not be changed afterwards; frames from the simulator never are."""
        self._put(name,('image',frame))

    def append(self,name,record):
        """Add one line to a JSON-lines file; lines keep their order."""
        with self.lock:self.lines.append((name,json.dumps(record)));self.idle.clear()
        self.wake.set()

    def fail(self,frames):
        """The frames the policy is to be given, and what was done to them. Blur off: the same arrays, untouched."""
        control=self.control;enabled=bool(control['enabled'])
        self.failure.enabled=enabled;self.failure.set_severity(control['severity'])
        used={name:self.failure.apply(frame) for name,frame in frames.items()} if enabled else frames
        kernel,sigma=SEVERITIES[control['severity']]
        return used,{'failure_enabled':enabled,'failure_type':'gaussian_blur' if enabled else 'normal','severity':control['severity'],
                     'revision':control['revision'],'kernel':kernel,'sigma':sigma}

    def chase_callback(self,_,message):
        """The simulator's chase camera, on the subscriber's own thread."""
        if time.monotonic()-self.last_chase<.25:return
        from projectairsim.utils import unpack_image
        frame=unpack_image(message)
        if frame.shape==(360,640,3) and message['encoding']=='BGR':
            publish_image(self.output/'chase_latest.png',frame);self.last_chase=time.monotonic();self.chase_ready=True

    # --- the background thread -------------------------------------------------------------------------------
    def _write(self,name,item):
        kind,data=item;path=self.output/name
        if kind=='image':publish_image(path,data)
        else:
            temporary=path.with_suffix('.publish.tmp');temporary.write_text(data,encoding='utf-8')
            for attempt in range(20):
                try:temporary.replace(path);break
                except PermissionError:
                    # A short reader lease on the Windows side can deny the replacement for a moment.
                    if attempt==19:raise
                    time.sleep(.01)

    def _run(self):
        last_poll=0.
        while True:
            self.wake.wait(self.period);self.wake.clear()
            with self.lock:pending,self.pending=self.pending,{};lines,self.lines=self.lines,[]
            try:
                for name,text in lines:
                    with (self.output/name).open('a',encoding='utf-8') as file:file.write(text+'\n')
                for name,item in pending.items():self._write(name,item)
            except Exception as error:self.errors.append(repr(error))
            with self.lock:
                if not self.pending and not self.lines:self.idle.set()
            if time.monotonic()-last_poll>=.1:
                last_poll=time.monotonic()
                try:self.control=read_control(self.output/'control.json');self.control_warning=None
                except (OSError,ValueError) as error:self.control_warning=str(error)
            if self.closed and self.idle.is_set():return

    def flush(self,timeout=5.):
        """Wait until everything handed over so far is on disk."""
        self.wake.set();return self.idle.wait(timeout)

    def close(self):
        self.flush();self.closed=True;self.wake.set();self.thread.join(timeout=5)
