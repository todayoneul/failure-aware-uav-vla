
# OFT FiLM, test
{"flights": 48, "success": 47, "landings": 36, "landing_success": 36, "approaches": 12, "approach_success": 11, "grounded": 46, "in_view": 47, "terminal_as_asked": 48, "wrong_target": 1, "collisions": 0}

### Landings by start distance

| Band | Correct target | Approach | Alignment | Touchdown | Stable physical landing | System landing | Strict zero-action |
|---|---|---|---|---|---|---|---|
| near | 12/12 | 12/12 | 11/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| mid | 12/12 | 12/12 | 11/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| far | 12/12 | 12/12 | 9/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| all | 36/36 | 36/36 | 31/36 | 36/36 | 36/36 | 36/36 | 36/36 |

### By kind of start

| Start | Success |
|---|---|
| visible | 3/3 |
| approach | 6/6 |
| search | 3/3 |
| edge | 3/3 |
| pair | 3/3 |
| swap | 6/6 |
| query | 5/6 |
| past | 3/3 |
| visible2 | 3/3 |
| behind | 3/3 |
| pair2 | 3/3 |
| red_visible | 3/3 |
| red_search | 3/3 |

### By band

| Band | All | Landing | Approach |
|---|---|---|---|
| near | 15/16 | 12/12 | 3/4 |
| mid | 16/16 | 12/12 | 4/4 |
| far | 16/16 | 12/12 | 4/4 |

### Target selection

| Start holds | Flights | Success | Ended at the named object | Ended at a wrong object |
|---|---|---|---|---|
| an object of the same colour and another shape beside the named one or in the first view | 22 | 21 | 21 | 1 |
| an object of the same shape and another colour beside the named one or in the first view | 26 | 26 | 26 | 0 |
| a related object in the first view, the named one outside it | 17 | 16 | 16 | 1 |
| ... of the same colour | 8 | 7 | 7 | 1 |
| ... of the same shape | 9 | 9 | 9 | 0 |
| the two objects exchanged (position swap) | 6 | 6 | 6 | 0 |
| the neighbour named instead (query) | 6 | 5 | 5 | 1 |
| the named object outside the first view | 27 | 26 | 26 | 1 |
| the named object in the first view | 21 | 21 | 21 | 0 |
| named object lost on the way without a forced turn | 2 | 2 | 2 | 0 |
| ... and seen again | 2 | 2 | 2 | 0 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|
| SEARCH | 1 | stopped by itself | ct-9507-blue_pad-near_pair2-find-blue_cube |

Finalizer: {"landing_missions": 36, "contacts": 36, "triggered": 36, "disarmed": 36, "rescued_success": 0, "approach_missions": 12, "active_in_approach_missions": 0, "evaluator": ["canonical_evaluator_v2"], "legacy_reading_disagrees": []}
Touchdown: {"touchdowns_on_the_named_pad": 36, "horizontal_error_m": {"median": 2.76, "max": 4.69}, "vertical_speed_mps": {"median": 0.6, "max": 0.65}}

# Teacher, test
{"flights": 48, "success": 48, "landings": 36, "landing_success": 36, "approaches": 12, "approach_success": 12, "grounded": 47, "in_view": 48, "terminal_as_asked": 48, "wrong_target": 0, "collisions": 0}

### Landings by start distance

| Band | Correct target | Approach | Alignment | Touchdown | Stable physical landing | System landing | Strict zero-action |
|---|---|---|---|---|---|---|---|
| near | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| mid | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| far | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| all | 36/36 | 36/36 | 36/36 | 36/36 | 36/36 | 36/36 | 36/36 |

### By kind of start

| Start | Success |
|---|---|
| visible | 3/3 |
| approach | 6/6 |
| search | 3/3 |
| edge | 3/3 |
| pair | 3/3 |
| swap | 6/6 |
| query | 6/6 |
| past | 3/3 |
| visible2 | 3/3 |
| behind | 3/3 |
| pair2 | 3/3 |
| red_visible | 3/3 |
| red_search | 3/3 |

### By band

| Band | All | Landing | Approach |
|---|---|---|---|
| near | 16/16 | 12/12 | 4/4 |
| mid | 16/16 | 12/12 | 4/4 |
| far | 16/16 | 12/12 | 4/4 |

### Target selection

| Start holds | Flights | Success | Ended at the named object | Ended at a wrong object |
|---|---|---|---|---|
| an object of the same colour and another shape beside the named one or in the first view | 22 | 22 | 22 | 0 |
| an object of the same shape and another colour beside the named one or in the first view | 26 | 26 | 26 | 0 |
| a related object in the first view, the named one outside it | 17 | 17 | 17 | 0 |
| ... of the same colour | 8 | 8 | 8 | 0 |
| ... of the same shape | 9 | 9 | 9 | 0 |
| the two objects exchanged (position swap) | 6 | 6 | 6 | 0 |
| the neighbour named instead (query) | 6 | 6 | 6 | 0 |
| the named object outside the first view | 27 | 27 | 27 | 0 |
| the named object in the first view | 21 | 21 | 21 | 0 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|

Finalizer: {"landing_missions": 36, "contacts": 36, "triggered": 36, "disarmed": 36, "rescued_success": 0, "approach_missions": 12, "active_in_approach_missions": 0, "evaluator": ["legacy"], "legacy_reading_disagrees": []}
Touchdown: {"touchdowns_on_the_named_pad": 36, "horizontal_error_m": {"median": 0.71, "max": 1.16}, "vertical_speed_mps": {"median": 0.36, "max": 0.37}}
