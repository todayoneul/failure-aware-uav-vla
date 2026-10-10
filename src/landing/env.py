"""The canonical evaluation environment with landing surfaces in place of the one rule for pads.

`SurfaceContacts` is mixed in before `scripts.visual_search.SearchEnv`. It replaces one method, the reading of the
collision topic: where the canonical environment asks "is this a pad, touched from above?", it asks the same of every
landable surface of the layout (src/landing/surface.py). For a layout whose only landable surfaces are pads the two
readings are the same event for event (tests/test_landing_surface.py flies both over one event stream).

Nothing else of the flight changes: the observation, the sentence, the policy call, the command and the finalizer are the
canonical loop's. What a surface is stays on this side: the finalizer is told that a contact was reported, as before, and
nothing about what was touched.
"""
from src.landing.surface import SurfaceSet
from src.mission.start import spawn_report


class SurfaceContacts:
    async def reset(self,episode):
        await super().reset(episode)
        self.surfaces=SurfaceSet(self.map_config,self.layout,self.landing)
        # Every height of the flight is counted from where the vehicle came to rest; a start that is not on the ground shifts them all.
        self.spawn=spawn_report(self.ground_z,self.map_config,self.landing)

    def contacts(self):
        """Read the collision events that arrived since the last call: a touchdown on a landing surface, or a collision.

        A landable surface (or a pad's marking, or a landing cap) touched while the vehicle is above its top is a touchdown;
        the first one is kept with the vehicle's place at that moment. Anything else touched is a collision: the side of a
        cube or a cylinder, an object without a flat top, the body under a cap."""
        events=self.collisions[self.scanned:];self.scanned+=len(events)
        # For the executor: was any contact reported since the last call? What was touched is not passed on to it.
        self.contact_event=any(event.get('time_stamp',0)>self.flight_stamp for event in events)
        for event in events:
            if event.get('time_stamp',0)<=self.flight_stamp or self.hit:continue
            # Heights are counted from where the vehicle rested at the start, so standing on a surface reads as its height.
            surface=self.surfaces.touched(event.get('object_name',''),self.ground_z-event['position']['z'])
            if surface is not None:
                if self.touchdown is None:
                    centre=surface.centre_xy
                    self.touchdown={'object':surface.object_id,'position':[event['position']['x'],event['position']['y'],event['position']['z']],
                                    'offset_m':[event['position']['x']-centre[0],event['position']['y']-centre[1]],'time_stamp':event['time_stamp']}
            else:self.hit=event.get('object_name') or 'unknown'
        return self.touchdown,self.hit
