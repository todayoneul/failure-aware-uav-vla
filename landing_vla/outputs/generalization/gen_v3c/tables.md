
# OFT Gen-v3c, validation
{"flights": 36, "success": 33, "landings": 27, "landing_success": 24, "approaches": 9, "approach_success": 9, "grounded": 32, "in_view": 33, "terminal_as_asked": 34, "wrong_target": 3, "collisions": 0}

### Landings by start distance

| Band | Correct target | Approach | Alignment | Touchdown | Stable physical landing | System landing | Strict zero-action |
|---|---|---|---|---|---|---|---|
| near | 8/9 | 8/9 | 6/9 | 8/9 | 8/9 | 8/9 | 8/9 |
| mid | 7/9 | 7/9 | 7/9 | 7/9 | 7/9 | 7/9 | 6/9 |
| far | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| all | 24/27 | 24/27 | 22/27 | 24/27 | 24/27 | 24/27 | 23/27 |

### By kind of start

| Start | Success |
|---|---|
| visible | 3/3 |
| approach | 6/6 |
| search | 3/3 |
| red_visible | 3/3 |
| edge | 3/3 |
| past_color | 2/3 |
| past_shape | 2/3 |
| pair | 3/3 |
| swap | 2/3 |
| query | 3/3 |
| red_past | 3/3 |

### By band

| Band | All | Landing | Approach |
|---|---|---|---|
| near | 11/12 | 8/9 | 3/3 |
| mid | 10/12 | 7/9 | 3/3 |
| far | 12/12 | 9/9 | 3/3 |

### Target selection

| Start holds | Flights | Success | Ended at the named object | Ended at a wrong object |
|---|---|---|---|---|
| an object of the same colour and another shape beside the named one or in the first view | 11 | 9 | 9 | 2 |
| an object of the same shape and another colour beside the named one or in the first view | 7 | 6 | 6 | 1 |
| a related object in the first view, the named one outside it | 15 | 12 | 12 | 3 |
| ... of the same colour | 9 | 7 | 7 | 2 |
| ... of the same shape | 6 | 5 | 5 | 1 |
| the two objects exchanged (position swap) | 3 | 2 | 2 | 1 |
| the neighbour named instead (query) | 3 | 3 | 3 | 0 |
| the named object outside the first view | 21 | 18 | 18 | 3 |
| the named object in the first view | 15 | 15 | 15 | 0 |
| named object lost on the way without a forced turn | 1 | 1 | 1 | 0 |
| ... and seen again | 1 | 1 | 1 | 0 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|
| SEARCH | 3 | stopped by itself | v3c-22005-blue_pad-near_pair-swapped v3c-22011-blue_pad-mid_past_color v3c-22012-blue_pad-mid_past_shape |

Finalizer: {"landing_missions": 27, "contacts": 25, "triggered": 25, "disarmed": 25, "rescued_success": 1, "approach_missions": 9, "active_in_approach_missions": 0, "evaluator": ["canonical_evaluator_v2"], "legacy_reading_disagrees": []}
Touchdown: {"touchdowns_on_the_named_pad": 24, "horizontal_error_m": {"median": 2.56, "max": 3.89}, "vertical_speed_mps": {"median": 0.61, "max": 0.65}}

# OFT Gen-v3c, pilot
{"flights": 12, "success": 11, "landings": 9, "landing_success": 8, "approaches": 3, "approach_success": 3, "grounded": 11, "in_view": 11, "terminal_as_asked": 11, "wrong_target": 1, "collisions": 0}

### Landings by start distance

| Band | Correct target | Approach | Alignment | Touchdown | Stable physical landing | System landing | Strict zero-action |
|---|---|---|---|---|---|---|---|
| mid | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 1/3 |
| far | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| all | 8/9 | 8/9 | 8/9 | 8/9 | 8/9 | 8/9 | 7/9 |

### By kind of start

| Start | Success |
|---|---|
| color_first | 3/4 |
| shape_first | 2/2 |
| lost | 2/2 |
| swap | 2/2 |
| search | 2/2 |

### By band

| Band | All | Landing | Approach |
|---|---|---|---|
| near | - | - | - |
| mid | 4/5 | 2/3 | 2/2 |
| far | 7/7 | 6/6 | 1/1 |

### Target selection

| Start holds | Flights | Success | Ended at the named object | Ended at a wrong object |
|---|---|---|---|---|
| an object of the same colour and another shape beside the named one or in the first view | 6 | 5 | 5 | 1 |
| an object of the same shape and another colour beside the named one or in the first view | 2 | 2 | 2 | 0 |
| a related object in the first view, the named one outside it | 6 | 5 | 5 | 1 |
| ... of the same colour | 4 | 3 | 3 | 1 |
| ... of the same shape | 2 | 2 | 2 | 0 |
| the two objects exchanged (position swap) | 2 | 2 | 2 | 0 |
| the named object outside the first view | 8 | 7 | 7 | 1 |
| the named object in the first view | 4 | 4 | 4 | 0 |
| named object lost to a forced turn, a related object in view | 2 | 2 | 2 | 0 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|
| SEARCH | 1 | stopped by itself | pilot-21501-red_pad-color_first |

Finalizer: {"landing_missions": 9, "contacts": 8, "triggered": 8, "disarmed": 8, "rescued_success": 1, "approach_missions": 3, "active_in_approach_missions": 0, "evaluator": ["canonical_evaluator_v2"], "legacy_reading_disagrees": []}
Touchdown: {"touchdowns_on_the_named_pad": 8, "horizontal_error_m": {"median": 2.12, "max": 3.56}, "vertical_speed_mps": {"median": 0.62, "max": 0.64}}

# OFT Gen-v3, pilot
{"flights": 12, "success": 9, "landings": 9, "landing_success": 7, "approaches": 3, "approach_success": 2, "grounded": 9, "in_view": 9, "terminal_as_asked": 10, "wrong_target": 2, "collisions": 1}

### Landings by start distance

| Band | Correct target | Approach | Alignment | Touchdown | Stable physical landing | System landing | Strict zero-action |
|---|---|---|---|---|---|---|---|
| mid | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 |
| far | 5/6 | 5/6 | 5/6 | 5/6 | 5/6 | 5/6 | 5/6 |
| all | 7/9 | 7/9 | 7/9 | 7/9 | 7/9 | 7/9 | 7/9 |

### By kind of start

| Start | Success |
|---|---|
| color_first | 1/4 |
| shape_first | 2/2 |
| lost | 2/2 |
| swap | 2/2 |
| search | 2/2 |

### By band

| Band | All | Landing | Approach |
|---|---|---|---|
| near | - | - | - |
| mid | 3/5 | 2/3 | 1/2 |
| far | 6/7 | 5/6 | 1/1 |

### Target selection

| Start holds | Flights | Success | Ended at the named object | Ended at a wrong object |
|---|---|---|---|---|
| an object of the same colour and another shape beside the named one or in the first view | 6 | 3 | 3 | 2 |
| an object of the same shape and another colour beside the named one or in the first view | 2 | 2 | 2 | 0 |
| a related object in the first view, the named one outside it | 6 | 3 | 3 | 2 |
| ... of the same colour | 4 | 1 | 1 | 2 |
| ... of the same shape | 2 | 2 | 2 | 0 |
| the two objects exchanged (position swap) | 2 | 2 | 2 | 0 |
| the named object outside the first view | 8 | 5 | 5 | 2 |
| the named object in the first view | 4 | 4 | 4 | 0 |
| named object lost to a forced turn, a related object in view | 2 | 2 | 2 | 0 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|
| COLLISION | 1 | collision | pilot-21502-blue_pad-color_first |
| SEARCH | 2 | stopped by itself | pilot-21501-red_pad-color_first pilot-21503-blue_cube-color_first |

Finalizer: {"landing_missions": 9, "contacts": 7, "triggered": 7, "disarmed": 7, "rescued_success": 0, "approach_missions": 3, "active_in_approach_missions": 0, "evaluator": ["canonical_evaluator_v2"], "legacy_reading_disagrees": []}
Touchdown: {"touchdowns_on_the_named_pad": 7, "horizontal_error_m": {"median": 1.87, "max": 3.71}, "vertical_speed_mps": {"median": 0.61, "max": 0.63}}

# Teacher, validation
{"flights": 36, "success": 36, "landings": 27, "landing_success": 27, "approaches": 9, "approach_success": 9, "grounded": 36, "in_view": 36, "terminal_as_asked": 36, "wrong_target": 0, "collisions": 0}

### Landings by start distance

| Band | Correct target | Approach | Alignment | Touchdown | Stable physical landing | System landing | Strict zero-action |
|---|---|---|---|---|---|---|---|
| near | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| mid | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| far | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| all | 27/27 | 27/27 | 27/27 | 27/27 | 27/27 | 27/27 | 27/27 |

### By kind of start

| Start | Success |
|---|---|
| visible | 3/3 |
| approach | 6/6 |
| search | 3/3 |
| edge | 3/3 |
| past_color | 3/3 |
| past_shape | 3/3 |
| pair | 3/3 |
| swap | 3/3 |
| query | 3/3 |
| red_past | 3/3 |
| red_visible | 3/3 |

### By band

| Band | All | Landing | Approach |
|---|---|---|---|
| near | 12/12 | 9/9 | 3/3 |
| mid | 12/12 | 9/9 | 3/3 |
| far | 12/12 | 9/9 | 3/3 |

### Target selection

| Start holds | Flights | Success | Ended at the named object | Ended at a wrong object |
|---|---|---|---|---|
| an object of the same colour and another shape beside the named one or in the first view | 11 | 11 | 11 | 0 |
| an object of the same shape and another colour beside the named one or in the first view | 7 | 7 | 7 | 0 |
| a related object in the first view, the named one outside it | 15 | 15 | 15 | 0 |
| ... of the same colour | 9 | 9 | 9 | 0 |
| ... of the same shape | 6 | 6 | 6 | 0 |
| the two objects exchanged (position swap) | 3 | 3 | 3 | 0 |
| the neighbour named instead (query) | 3 | 3 | 3 | 0 |
| the named object outside the first view | 21 | 21 | 21 | 0 |
| the named object in the first view | 15 | 15 | 15 | 0 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|

Finalizer: {"landing_missions": 27, "contacts": 27, "triggered": 27, "disarmed": 27, "rescued_success": 0, "approach_missions": 9, "active_in_approach_missions": 0, "evaluator": ["canonical_evaluator_v2"], "legacy_reading_disagrees": []}
Touchdown: {"touchdowns_on_the_named_pad": 27, "horizontal_error_m": {"median": 0.65, "max": 1.08}, "vertical_speed_mps": {"median": 0.36, "max": 0.37}}

# Teacher, pilot
{"flights": 12, "success": 12, "landings": 9, "landing_success": 9, "approaches": 3, "approach_success": 3, "grounded": 12, "in_view": 12, "terminal_as_asked": 12, "wrong_target": 0, "collisions": 0}

### Landings by start distance

| Band | Correct target | Approach | Alignment | Touchdown | Stable physical landing | System landing | Strict zero-action |
|---|---|---|---|---|---|---|---|
| mid | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| far | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| all | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |

### By kind of start

| Start | Success |
|---|---|
| color_first | 4/4 |
| shape_first | 2/2 |
| lost | 2/2 |
| swap | 2/2 |
| search | 2/2 |

### By band

| Band | All | Landing | Approach |
|---|---|---|---|
| near | - | - | - |
| mid | 5/5 | 3/3 | 2/2 |
| far | 7/7 | 6/6 | 1/1 |

### Target selection

| Start holds | Flights | Success | Ended at the named object | Ended at a wrong object |
|---|---|---|---|---|
| an object of the same colour and another shape beside the named one or in the first view | 6 | 6 | 6 | 0 |
| an object of the same shape and another colour beside the named one or in the first view | 2 | 2 | 2 | 0 |
| a related object in the first view, the named one outside it | 6 | 6 | 6 | 0 |
| ... of the same colour | 4 | 4 | 4 | 0 |
| ... of the same shape | 2 | 2 | 2 | 0 |
| the two objects exchanged (position swap) | 2 | 2 | 2 | 0 |
| the named object outside the first view | 8 | 8 | 8 | 0 |
| the named object in the first view | 4 | 4 | 4 | 0 |
| named object lost to a forced turn, a related object in view | 2 | 2 | 2 | 0 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|

Finalizer: {"landing_missions": 9, "contacts": 9, "triggered": 9, "disarmed": 9, "rescued_success": 0, "approach_missions": 3, "active_in_approach_missions": 0, "evaluator": ["canonical_evaluator_v2"], "legacy_reading_disagrees": []}
Touchdown: {"touchdowns_on_the_named_pad": 9, "horizontal_error_m": {"median": 0.64, "max": 1.1}, "vertical_speed_mps": {"median": 0.36, "max": 0.36}}
