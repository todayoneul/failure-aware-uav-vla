"""The load order on the 12 GB card (docs/demo1.md 6.5): the simulator first, the VLM after it.

CARLA reaches about 10.9 GB for a few seconds while it starts and loads the map, then settles near 5.4 GB. A VLM
already on the card would not fit under that peak, so:

    - a live VLM (next to the simulator) may load only inside `sim.session`, once the map is up and the card's used
      memory has stopped moving;
    - the simulator may not start or change map in a process that holds a VLM;
    - an offline VLM (saved frames, training) may load only while no simulator is running.

`sim.session` and `vlm.make` call these checks; a wrong order raises LoadOrderError before anything is loaded.
"""
import socket
import subprocess
import time

SIMULATOR_PORT = 2000
NEED_FREE_MIB = 4500.0     # the largest candidate with RF-DETR took 4.5 GB on top of the simulator (docs/demo1.md 6.5)

_simulator_ready = False
_vlms = set()


class LoadOrderError(RuntimeError):
    pass


def simulator_ready(ready=True):
    global _simulator_ready
    _simulator_ready = ready


def is_simulator_ready():
    return _simulator_ready


def simulator_listening(port=SIMULATOR_PORT, host="localhost"):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def gpu_memory_mib():
    """(used, total) MiB of the whole card, all processes."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    used, total = (float(v) for v in out.strip().splitlines()[0].split(","))
    return used, total


def wait_settled(read=gpu_memory_mib, need_free_mib=NEED_FREE_MIB, stable_mib=200.0, stable_s=2.0, timeout_s=60.0,
                 every_s=0.25, clock=time.monotonic, sleep=time.sleep):
    """Wait until the used memory has stayed within `stable_mib` for `stable_s` with `need_free_mib` free (a plateau
    at the map-load peak is steady too, so steadiness alone is not enough). Returns the used MiB."""
    start = clock()
    used, total = read()
    low = high = used
    since = start
    while True:
        now = clock()
        if now - since >= stable_s and total - high >= need_free_mib:
            return high
        if now - start >= timeout_s:
            raise LoadOrderError(f"GPU not ready within {timeout_s:.0f} s: {low:.0f}-{high:.0f} of {total:.0f} MiB "
                                 f"used, {need_free_mib:.0f} MiB must be free")
        sleep(every_s)
        used, total = read()
        low, high = min(low, used), max(high, used)
        if high - low > stable_mib:
            low = high = used
            since = clock()


def before_map_load():
    """Call before starting the simulator or loading a map."""
    if _vlms:
        raise LoadOrderError(f"a VLM is on the GPU ({', '.join(sorted(_vlms))}): the simulator's start and map load "
                             "reach about 10.9 GB, so start the simulator first and load the VLM after it")


def before_vlm_load(name, offline=False, listening=None, settle=None):
    """Call before loading a VLM. `offline`: it works on saved frames or trains, with no simulator running."""
    if offline:
        if (listening or simulator_listening)():
            raise LoadOrderError(f"{name}: offline load refused, a simulator is running (it may be loading a map)")
        return
    if not _simulator_ready:
        raise LoadOrderError(f"{name}: load a live VLM inside sim.session, after the map is up (or pass offline=True "
                             "for saved frames)")
    (settle or wait_settled)()


def vlm_loaded(name):
    _vlms.add(name)


def vlm_unloaded(name):
    _vlms.discard(name)


def reset():
    """Forget the state (tests)."""
    global _simulator_ready
    _simulator_ready = False
    _vlms.clear()
