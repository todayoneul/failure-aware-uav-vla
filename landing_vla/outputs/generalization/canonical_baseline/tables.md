
# OFT Gen-v3, validation
{"flights": 36, "success": 30, "landings": 27, "landing_success": 21, "approaches": 9, "approach_success": 9, "grounded": 32, "in_view": 34, "terminal_as_asked": 33, "wrong_target": 3, "collisions": 0}

### Landings by start distance

| Band | Correct target | Approach | Alignment | Touchdown | Stable physical landing | System landing | Strict zero-action |
|---|---|---|---|---|---|---|---|
| near | 8/9 | 8/9 | 7/9 | 8/9 | 8/9 | 8/9 | 7/9 |
| mid | 9/9 | 9/9 | 9/9 | 9/9 | 8/9 | 8/9 | 9/9 |
| far | 6/9 | 6/9 | 6/9 | 6/9 | 5/9 | 5/9 | 6/9 |
| all | 23/27 | 23/27 | 22/27 | 23/27 | 21/27 | 21/27 | 22/27 |

### By kind of start

| Start | Success |
|---|---|
| visible | 3/3 |
| approach | 6/6 |
| search | 2/3 |
| red_visible | 2/3 |
| edge | 3/3 |
| pair | 3/3 |
| swap | 2/3 |
| query | 3/3 |
| past | 2/3 |
| behind | 2/3 |
| red_search | 2/3 |

### By band

| Band | All | Landing | Approach |
|---|---|---|---|
| near | 11/12 | 8/9 | 3/3 |
| mid | 11/12 | 8/9 | 3/3 |
| far | 8/12 | 5/9 | 3/3 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|
| SEARCH | 2 | stopped by itself | cv-8504-blue_pad-near_pair-swapped cv-8522-blue_pad-far_past |
| GROUNDING | 1 | stopped by itself | cv-8523-blue_pad-far_behind |
| WRONG_TARGET | 1 | at red_cube | cv-8524-red_pad-far_red_visible |
| PHYSICAL_LANDING | 2 | did not stay on the pad | cv-8510-blue_pad-mid_search cv-8525-red_pad-far_red_search |

Touchdown: {"touchdowns_on_the_named_pad": 23, "horizontal_error_m": {"median": 2.4, "p90": 3.37, "max": 4.03}, "vertical_speed_mps": {"median": 0.58, "max": 0.62}, "stable_duration_s": {"median": 1.5460007190704346}}

### Latched and disarmed on the named pad, not counted as a stable landing

| Episode | Decisions in contact | Height span m | Horizontal span m | Reported vertical m/s | Read as standing | Policy zero action |
|---|---|---|---|---|---|---|
| cv-8510-blue_pad-mid_search | 4 | 0.0004 | 0.0 | [0.095, 0.074, 0.052, 0.041] | 1 | True |
| cv-8525-red_pad-far_red_search | 4 | 0.0002 | 0.0 | [0.085, 0.08, 0.08, 0.063] | 0 | True |

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
| pair | 3/3 |
| swap | 3/3 |
| query | 3/3 |
| past | 3/3 |
| behind | 3/3 |
| red_visible | 3/3 |
| red_search | 3/3 |

### By band

| Band | All | Landing | Approach |
|---|---|---|---|
| near | 12/12 | 9/9 | 3/3 |
| mid | 12/12 | 9/9 | 3/3 |
| far | 12/12 | 9/9 | 3/3 |

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|

Touchdown: {"touchdowns_on_the_named_pad": 27, "horizontal_error_m": {"median": 0.62, "p90": 0.94, "max": 0.98}, "vertical_speed_mps": {"median": 0.36, "max": 0.37}, "stable_duration_s": {"median": 1.505357265472412}}

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

### Failures by the first stage not passed

| Stage | Count | How | Episodes |
|---|---|---|---|

Touchdown: {"touchdowns_on_the_named_pad": 36, "horizontal_error_m": {"median": 0.7, "p90": 0.93, "max": 1.16}, "vertical_speed_mps": {"median": 0.36, "max": 0.37}, "stable_duration_s": {"median": 1.5027318000793457}}
