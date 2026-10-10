"""The reading of a landing on any landing surface. No simulator, no model.

The canonical evaluator (scripts/visual_search.landing_summary) reads a landing in steps that are kept apart: which object
was touched, whether the contact was inside the landing region, whether it was soft, whether the vehicle then stood still,
and whether the finalizer latched and disarmed. One of those steps is written for a pad: the landing region is a square of
half-width 6.6 m. `read` goes through the same steps with the region of the surface that was touched (src/landing/surface.py)
and leaves every other step as the canonical summary recorded it. On a pad the two readings are the same.

    LAND succeeds when   the object touched is the one the sentence named
                   and   the contact was on a landing surface, from above (otherwise it is a collision, and none is recorded)
                   and   it was inside that surface's usable region
                   and   it was soft
                   and   the flight ended with the vehicle standing still on it
                   and   the finalizer latched and the motors were switched off
                   and   nothing else was hit

A physical landing and the mission are two questions. A vehicle standing still on the wrong cube has landed, and the
finalizer, which is never told which object was meant, has latched on it; the mission has failed.

One more thing in the canonical reading holds for a pad and not in general. It takes as the decision of the touchdown the
first decision of the whole flight at which the vehicle was at the height of the contact, and the speed just before that
decision as the speed of the touchdown. A pad's top is 2.5 m above the ground and no flight was ever at that height except
standing on a pad. The top of a cube is 8 m above the ground: a vehicle climbing over the cube passes that height on its
way up, long before it lands. Here the decision of the touchdown is the first one of the stay at that height during which
the contact was reported (`touchdown_decision` below), which for every recorded landing on a pad is the same decision.
"""
VERSION='generalized_surface_v1'


def touchdown_decision(steps,record,tolerance_m):
    """The decision at which the vehicle came to stand on the surface it touched, or None when the log does not show it.

    The loop marks a decision `on_surface` when a contact has been reported and the vehicle is at the contact's height. The
    report can arrive a decision after the vehicle is already standing, so the stay is followed back from the first marked
    decision for as long as the vehicle was at that height. A decision at that height earlier in the flight, with the
    vehicle somewhere else and on its way up or down, is not part of the stay."""
    marked=next((index for index,step in enumerate(steps) if step.get('on_surface')),None)
    if marked is None:return None
    level=steps[marked]['position'][2];first=marked
    while first>0 and abs(steps[first-1]['position'][2]-level)<=tolerance_m:first-=1
    # The canonical reading measured from the contact's own height, which the log does not keep. Where its decision lies in
    # this stay it is the exact one; where it lies before the stay it is the vehicle passing that height, and is not used.
    canonical=record.get('step')
    return canonical if canonical is not None and first<=canonical<=marked else first


def read(summary,surfaces,landing,steps=None):
    """The generalised reading of one flight from its canonical summary. `steps`, when given, adds how long the vehicle was
    over the named surface (for a flight that hovered over it and never came down)."""
    target=summary['target'];task=summary.get('task','approach');named=surfaces[target];record=summary.get('touchdown')
    hit=summary.get('hit');final=summary.get('finalizer') or {}
    reading={'evaluator':VERSION,'task':task,'target':target,'target_surface':named.describe(),'collision':hit,
             'contact_from_above':record is not None,'landed_on':None,'landed_surface':None,'correct_object':False,'wrong_object':False,
             'on_top':False,'inside_region':False,'acceptable_touchdown':False,
             'physical_landing':bool(summary.get('physical_landing')),'finalizer_active':bool(final.get('active')),
             'finalizer_latched':bool(final.get('active') and final.get('finalizer_triggered') and final.get('disarm_triggered') and not hit),
             'canonical_success':bool(summary.get('success'))}
    if record:
        touched=surfaces[record['object']];speed=record['vertical_speed_mps'];decision=record.get('step')
        first=touchdown_decision(steps,record,landing['evaluator']['surface_height_tolerance_m']) if steps else None
        # The speed of the touchdown is what the vehicle was doing at the last decision before it stood on the surface.
        if first is not None:speed=steps[max(0,first-1)].get('velocity',[0.,0.,0.])[2];decision=first
        reading.update(landed_on=record['object'],landed_surface=touched.describe(),correct_object=record['object']==target,
                       wrong_object=record['object']!=target,on_top=touched.on_top(record['offset_m']),inside_region=touched.inside(record['offset_m']),
                       touchdown_offset_m=list(record['offset_m']),horizontal_error_m=record['horizontal_error_m'],
                       touchdown_decision=decision,vertical_speed_mps=speed,
                       acceptable_touchdown=abs(speed)<=landing['touchdown']['max_vertical_speed_mps'],
                       canonical_inside_region=record['inside_region'],canonical_touchdown_decision=record.get('step'),
                       canonical_vertical_speed_mps=record['vertical_speed_mps'])
    if steps is not None:
        over=[named.offset_of(step['position'][0],step['position'][1]) for step in steps]
        reading['decisions_over_surface']=sum(named.on_top(offset) for offset in over)
        reading['decisions_over_usable_region']=sum(named.inside(offset) for offset in over)
    if task=='land':
        reading['touchdown_success']=bool(reading['correct_object'] and reading['inside_region'] and reading['acceptable_touchdown'])
        reading['stable_physical_landing']=bool(reading['touchdown_success'] and reading['physical_landing'] and not hit)
        reading['system_land_success']=bool(reading['stable_physical_landing'] and reading['finalizer_latched'])
        # A vehicle standing still on a landing surface that is not the named one: landed, latched, and a failed mission.
        reading['wrong_object_landing']=bool(reading['wrong_object'] and reading['physical_landing'])
        # Without the finalizer the canonical result is the strict rule (the policy's own stop); the region is still the surface's.
        reading['success']=reading['system_land_success'] if reading['finalizer_active'] else bool(summary.get('land_success') and reading['inside_region'])
    else:
        # An approach asks for a stop in the air beside the object. Nothing of that rule is about a surface.
        reading['success']=reading['canonical_success']
    reading['agrees_with_canonical']=reading['success']==reading['canonical_success']
    return reading


def words(reading,summary):
    """One sentence for the display: what the flight did about the landing it was asked for."""
    reason=summary.get('reason');surface=(reading.get('target_surface') or {}).get('surface')
    if reading['success']:return f'landed on the named object ({surface} surface); latched and disarmed'
    if reading['collision']:return f'COLLISION with {reading["collision"]}'
    if reading['wrong_object'] and reading['physical_landing']:return f'landed on {reading["landed_on"]}, not on the named object: a physical landing, a failed mission'
    if reading['wrong_object']:return f'touched down on {reading["landed_on"]}, not on the named object'
    if reading['contact_from_above'] and not reading['inside_region']:
        return f'touched the named object {reading["horizontal_error_m"]:.1f} m from its middle, outside its landing region'
    if reading['contact_from_above'] and not reading['acceptable_touchdown']:return 'touched the named object too fast for a landing'
    if reading['contact_from_above']:return 'touched the named object, but not a stable landing that the finalizer latched'
    if reason=='max_steps':return 'ran out of decisions before landing'
    if reading.get('decisions_over_surface'):return 'stopped in the air over the named object without landing on it'
    return 'stopped without landing on the named object'
