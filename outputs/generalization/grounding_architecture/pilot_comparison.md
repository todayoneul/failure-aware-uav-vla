
### Architecture pilot: every model over the same sixteen starts, twice

| Metric | Gen-v3c | FiLM | Control (no module, +2000) |
|---|---|---|---|
| Overall success | 28/32 | 32/32 | 30/32 |
| Ended at the named object | 28/32 | 32/32 | 30/32 |
| Wrong target | 4 | 0 | 2 |
| Same colour, another shape | 18/22 | 22/22 | 20/22 |
| Same shape, another colour | 10/10 | 10/10 | 10/10 |
| First-visible related object rejected | 22/26 | 26/26 | 24/26 |
| Temporary loss, named object reacquired | 6/6 | 6/6 | 6/6 |
| Starts correct 2/2 | 14 | 16 | 14 |
| Starts correct 1/2 | 0 | 0 | 2 |
| Starts correct 0/2 | 2 | 0 | 0 |
| Collision | 2 | 0 | 0 |
| End of flight as asked | 28/32 | 32/32 | 30/32 |
| Landings completed / went to the named pad | 22/22 | 26/26 | 25/25 |
| Inference per decision (ms) | 396 | 401 | 406 |
| Peak memory at inference (GiB) | 6.94 | 6.94 | 6.94 |
| Peak memory in training (GiB) | 9.82 | 9.88 | 9.82 |
| Added parameters | 0 | 4338432 | 0 |
| Trainable parameters | 54693891 | 59032323 | 54693891 |

### Flights that ended at the named object, by start (of 2)

| Start | Group | Named <- related | Where | Gen-v3c | FiLM | Control (no module, +2000) |
|---|---|---|---|---|---|---|
| ap-23001-blue_pad-same_color | same_color | blue_pad <- blue_cube | right | 2 | 2 | 2 |
| ap-23004-blue_pad-same_color | same_color | blue_pad <- blue_cube | right | 2 | 2 | 1 |
| ap-23013-blue_pad-same_color | same_color | blue_pad <- blue_cone | right | 2 | 2 | 2 |
| ap-23047-blue_pad-same_color | same_color | blue_pad <- blue_cone | centre | 2 | 2 | 2 |
| ap-23108-red_pad-same_color | same_color | red_pad <- red_cube | right | 0 | 2 | 2 |
| ap-23109-red_pad-same_color | same_color | red_pad <- red_cube | right | 0 | 2 | 2 |
| ap-23110-blue_pad-same_shape | same_shape | blue_pad <- red_pad | right | 2 | 2 | 2 |
| ap-23111-blue_pad-same_shape | same_shape | blue_pad <- red_pad | centre | 2 | 2 | 2 |
| ap-23113-blue_cube-same_shape | same_shape | blue_cube <- red_cube | right | 2 | 2 | 2 |
| ap-23118-blue_cube-first_view | first_view | blue_cube <- blue_pad | any | 2 | 2 | 2 |
| ap-23119-blue_cone-first_view | first_view | blue_cone <- blue_pad | any | 2 | 2 | 1 |
| ap-23124-blue_pad-first_view | first_view | blue_pad <- blue_cube | left | 2 | 2 | 2 |
| ap-23138-red_pad-first_view | first_view | red_pad <- blue_pad | left | 2 | 2 | 2 |
| ap-23143-blue_pad-lost | lost | blue_pad <- blue_cube | 1 | 2 | 2 | 2 |
| ap-23146-blue_pad-lost | lost | blue_pad <- blue_cone | -1 | 2 | 2 | 2 |
| ap-23209-blue_pad-lost | lost | blue_pad <- red_pad | 1 | 2 | 2 | 2 |
