"""Replay of recorded traffic (docs/demo1.md 4): the cars of a recording are spawned with physics off and put at
their recorded poses every tick, so every run and every configuration sees exactly the same traffic.

A recording is the .npz that `scripts/demo1_follow.py record` writes (per tick: pose and velocity of every car,
plus each car's box). Newer recordings also name each car's blueprint and colour; for older ones they are taken
from the trial in its spawn order (target, look-alikes, background), and the boxes are checked against it.
"""
import carla
import numpy as np

SPAWN_LIFT_M = 0.5   # spawn a little above the recorded pose; the first replayed tick puts the car exactly in place


def transform(pos, rot):
    return carla.Transform(carla.Location(float(pos[0]), float(pos[1]), float(pos[2])),
                           carla.Rotation(pitch=float(rot[0]), yaw=float(rot[1]), roll=float(rot[2])))


def blueprints(rec, trial):
    """[(blueprint id, colour or '')] for the recorded cars, in the recording's order."""
    if "type_id" in rec:
        return [(str(b), str(c)) for b, c in zip(rec["type_id"], rec["color"])]
    t = trial["target"]
    order = [(t["blueprint"], t["color"])] + [(t["blueprint"], c["color"]) for c in trial["lookalikes"]]
    order += [(b["blueprint"], b["color"] or "") for b in trial["background"]]
    if len(order) != len(rec["ids"]):
        raise ValueError(f"{trial['id']}: the recording has {len(rec['ids'])} cars but the trial spawns {len(order)}; "
                         "a car failed to spawn when it was recorded, so record it again")
    return order


class Replay:
    """The recorded cars of one trial, spawned with physics off. `apply(k)` puts them at their tick-k poses."""

    def __init__(self, client, world, rec, trial):
        self.client, self.rec = client, rec
        library = world.get_blueprint_library()
        cmds = []
        for i, (bp_id, color) in enumerate(blueprints(rec, trial)):
            bp = library.find(bp_id)
            if color and bp.has_attribute("color"):
                bp.set_attribute("color", color)
            bp.set_attribute("role_name", "replay")
            start = transform(rec["pos"][0, i] + np.array([0.0, 0.0, SPAWN_LIFT_M]), rec["rot"][0, i])
            cmds.append(carla.command.SpawnActor(bp, start).then(
                carla.command.SetSimulatePhysics(carla.command.FutureActor, False)))
        results = client.apply_batch_sync(cmds, True)
        errors = [r.error for r in results if r.error]
        if errors:
            raise RuntimeError(f"replay spawn failed for {len(errors)} cars: {errors[0]}")
        self.ids = [r.actor_id for r in results]
        actors = {a.id: a for a in world.get_actors(self.ids)}
        self.actors = [actors[i] for i in self.ids]
        extent = np.array([[a.bounding_box.extent.x, a.bounding_box.extent.y, a.bounding_box.extent.z]
                           for a in self.actors])
        if not np.allclose(extent, rec["box_extent"], atol=1e-3):
            raise ValueError("replayed cars do not have the recorded boxes: the blueprint order does not match")
        self.kind = [str(k) for k in rec["kind"]]

    def commands(self, k):
        return [carla.command.ApplyTransform(aid, transform(self.rec["pos"][k, i], self.rec["rot"][k, i]))
                for i, aid in enumerate(self.ids)]

    def apply(self, k, wait=True):
        """Put every car at its tick-k pose (before the tick that renders tick k). With `wait` the call returns once
        the simulator has applied the batch."""
        if wait:
            self.client.apply_batch_sync(self.commands(k), False)
        else:
            self.client.apply_batch(self.commands(k))

    def destroy(self):
        self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in self.ids], False)
