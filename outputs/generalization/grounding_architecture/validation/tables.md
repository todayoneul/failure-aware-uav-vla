
# OFT FiLM, validation
{"flights": 36, "success": 34, "landings": 27, "landing_success": 25, "approaches": 9, "approach_success": 9, "grounded": 34, "in_view": 34, "terminal_as_asked": 35, "wrong_target": 2, "collisions": 0}

### Landings by start distance

| Band | Correct target | Approach | Alignment | Touchdown | Stable physical landing | System landing | Strict zero-action |
|---|---|---|---|---|---|---|---|
| near | 8/9 | 8/9 | 8/9 | 8/9 | 8/9 | 8/9 | 6/9 |
| mid | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| far | 8/9 | 8/9 | 8/9 | 8/9 | 8/9 | 8/9 | 8/9 |
| all | 25/27 | 25/27 | 25/27 | 25/27 | 25/27 | 25/27 | 23/27 |

### By kind of start

| Start | Success |
|---|---|
| visible | 3/3 |
| approach | 6/6 |
| search | 3/3 |
| red_visible | 3/3 |
| edge | 3/3 |
| past_color | 3/3 |
| past_shape | 3/3 |
| pair | 3/3 |
| swap | 2/3 |
| query | 3/3 |
| red_past | 2/3 |

### By band

| Band | All | Landing | Approach |
|---|---|---|---|
| near | 11/12 | 8/9 | 3/3 |
| mid | 12/12 | 9/9 | 3/3 |
| far | 11/12 | 8/9 | 3/3 |

### Target selection

| Start holds | Flights | Success | Ended at the named object | Ended at a wrong object |
|---|---|---|---|---|
| an object of the same colour and another shape beside the named one or in the first view | 16 | 15 | 15 | 1 |
| an object of the same shape and another colour beside the named one or in the first view | 4 | 3 | 3 | 1 |
| a related object in the first view, the named one outside it | 17 | 15 | 15 | 2 |
| ... of the same colour | 13 | 12 | 12 | 1 |
| ... of the same shape | 4 | 3 | 3 | 1 |
| the two objects exchanged (position swap) | 3 | 2 | 2 | 1 |
| the neighbour named instead (query) | 3 | 3 | 3 | 0 |
| the named object outside the first view | 21 | 19 | 19 | 2 |
| the named object in the first view | 15 | 15 | 15 | 0 |
| named object lost on the way without a forced turn | 1 | 1 | 1 | 0 |
| ... and seen again | 1 | 1 | 1 | 0 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|
| SEARCH | 2 | stopped by itself | av-24172-blue_pad-near_pair-swapped av-24267-red_pad-far_red_past |

Finalizer: {"landing_missions": 27, "contacts": 26, "triggered": 26, "disarmed": 26, "rescued_success": 2, "approach_missions": 9, "active_in_approach_missions": 0, "evaluator": ["canonical_evaluator_v2"], "legacy_reading_disagrees": []}
Touchdown: {"touchdowns_on_the_named_pad": 25, "horizontal_error_m": {"median": 1.92, "max": 3.86}, "vertical_speed_mps": {"median": 0.6, "max": 0.62}}

# OFT FiLM, second_flight
{"flights": 17, "success": 13, "landings": 13, "landing_success": 10, "approaches": 4, "approach_success": 3, "grounded": 14, "in_view": 14, "terminal_as_asked": 16, "wrong_target": 3, "collisions": 0}

### Landings by start distance

| Band | Correct target | Approach | Alignment | Touchdown | Stable physical landing | System landing | Strict zero-action |
|---|---|---|---|---|---|---|---|
| near | 3/4 | 3/4 | 3/4 | 3/4 | 3/4 | 3/4 | 3/4 |
| mid | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| far | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 | 3/5 | 4/5 |
| all | 11/13 | 11/13 | 11/13 | 11/13 | 11/13 | 10/13 | 11/13 |

### By kind of start

| Start | Success |
|---|---|
| past_color | 2/3 |
| past_shape | 3/3 |
| swap | 2/3 |
| query | 2/3 |
| red_past | 2/3 |
| search | 1/1 |
| approach | 1/1 |

### By band

| Band | All | Landing | Approach |
|---|---|---|---|
| near | 3/5 | 3/4 | 0/1 |
| mid | 5/5 | 4/4 | 1/1 |
| far | 5/7 | 3/5 | 2/2 |

### Target selection

| Start holds | Flights | Success | Ended at the named object | Ended at a wrong object |
|---|---|---|---|---|
| an object of the same colour and another shape beside the named one or in the first view | 13 | 10 | 11 | 2 |
| an object of the same shape and another colour beside the named one or in the first view | 4 | 3 | 3 | 1 |
| a related object in the first view, the named one outside it | 17 | 13 | 14 | 3 |
| ... of the same colour | 13 | 10 | 11 | 2 |
| ... of the same shape | 4 | 3 | 3 | 1 |
| the two objects exchanged (position swap) | 3 | 2 | 2 | 1 |
| the neighbour named instead (query) | 3 | 2 | 2 | 1 |
| the named object outside the first view | 17 | 13 | 14 | 3 |
| named object lost on the way without a forced turn | 1 | 1 | 1 | 0 |
| ... and seen again | 1 | 1 | 1 | 0 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|
| SEARCH | 3 | stopped by itself | av-24172-blue_pad-near_pair-swapped av-24172-blue_pad-near_pair-find-blue_cone av-24267-red_pad-far_red_past |
| FINALIZER | 1 | did not disarm | av-24219-blue_pad-far_past_color |

Finalizer: {"landing_missions": 13, "contacts": 12, "triggered": 12, "disarmed": 11, "rescued_success": 0, "approach_missions": 4, "active_in_approach_missions": 0, "evaluator": ["canonical_evaluator_v2"], "legacy_reading_disagrees": ["av-24003-blue_pad-near_past_color"]}
Touchdown: {"touchdowns_on_the_named_pad": 11, "horizontal_error_m": {"median": 2.36, "max": 4.04}, "vertical_speed_mps": {"median": 0.6, "max": 0.62}}

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
| an object of the same colour and another shape beside the named one or in the first view | 16 | 16 | 16 | 0 |
| an object of the same shape and another colour beside the named one or in the first view | 4 | 4 | 4 | 0 |
| a related object in the first view, the named one outside it | 17 | 17 | 17 | 0 |
| ... of the same colour | 13 | 13 | 13 | 0 |
| ... of the same shape | 4 | 4 | 4 | 0 |
| the two objects exchanged (position swap) | 3 | 3 | 3 | 0 |
| the neighbour named instead (query) | 3 | 3 | 3 | 0 |
| the named object outside the first view | 21 | 21 | 21 | 0 |
| the named object in the first view | 15 | 15 | 15 | 0 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|

Finalizer: {"landing_missions": 27, "contacts": 27, "triggered": 27, "disarmed": 27, "rescued_success": 0, "approach_missions": 9, "active_in_approach_missions": 0, "evaluator": ["canonical_evaluator_v2"], "legacy_reading_disagrees": []}
Touchdown: {"touchdowns_on_the_named_pad": 27, "horizontal_error_m": {"median": 0.64, "max": 1.04}, "vertical_speed_mps": {"median": 0.36, "max": 0.39}}
