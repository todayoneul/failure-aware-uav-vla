# controlled_run1

Plan `controlled_landings.json`, policy `controlled`, evaluator `generalized_surface_v1`. 15 episodes, 14 flown, 13 with every check passed.

| Episode | Sentence | Start asked (x, y, yaw, height) | Start error (xy m / yaw deg / height m) | What happened | Landed on | Distance from the middle | Inside region | Physical landing | Latched | Mission | Checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| start-near | Approach the blue landing pad. | 18, 72, 30, 4 | 0.0000 / 0.0000 / -0.046 | model_stop (4 decisions) | - | - | - | no | no | failed | PASS |
| start-mid | Approach the blue landing pad. | 36, 60, -160, 9 | 0.0000 / 0.0000 / -0.073 | model_stop (4 decisions) | - | - | - | no | no | failed | PASS |
| start-far | Approach the blue landing pad. | -20, 20, 45, 12 | 0.0000 / 0.0000 / -0.030 | model_stop (4 decisions) | - | - | - | no | no | failed | PASS |
| pad-land | Find the blue landing pad and land on it. | 24, 74, 0, 6 | 0.0000 / 0.0000 / -0.053 | landed (86 decisions) | blue_pad | 1.28 m | yes | yes | yes | SUCCESS | PASS |
| cube-land | Find the blue cube and land on it. | 20, 52, 200, 6 | 0.0000 / 0.0000 / -0.052 | landed (56 decisions) | blue_cube | 3.40 m | yes | yes | yes | SUCCESS | PASS |
| cylinder-land | Find the green cylinder and land on it. | 70, 32.1, 90, 6 | 0.0000 / 0.0000 / -0.053 | landed (70 decisions) | green_cylinder | 2.59 m | yes | yes | yes | SUCCESS | PASS |
| cube-side | Find the blue cube and land on it. | 22, 66, 0, 6 | 0.0000 / 0.0000 / -0.053 | collision with BlueCube (32 decisions) | - | - | - | no | no | failed | PASS |
| cube-wrong | Find the blue cube and land on it. | 40, 52, -90, 6 | 0.0000 / 0.0000 / -0.053 | landed (55 decisions) | red_cube | 3.40 m | yes | yes | yes | failed | PASS |
| cylinder-edge | Find the green cylinder and land on it. | 70, 20, 180, 6 | 0.0000 / 0.0000 / -0.054 | max_steps (240 decisions) | - | - | - | no | no | failed | FAIL: landed_on, correct_object, on_top |
| cube-hover | Find the blue cube and land on it. | 14, 48, 180, 6 | 0.0000 / 0.0000 / -0.053 | model_stop (99 decisions) | - | - | - | no | no | failed | PASS |
| sphere-land-refused | (none: refused) | - | - | not flown: Selected object has no valid landing surface. Use APPROACH. | - | - | - | - | - | - | PASS |
| sphere-approach | Approach the orange ball. | 70, 32.1, 90, 6 | 0.0000 / 0.0000 / -0.053 | model_stop (59 decisions) | - | - | - | no | no | SUCCESS | PASS |
| sphere-top | Approach the orange ball. | 70, 20, 180, 6 | 0.0000 / 0.0000 / -0.054 | collision with OrangeBall (61 decisions) | - | - | - | no | no | failed | PASS |
| cap-land | Find the orange ball and land on it. | 70, 32.1, 90, 6 | 0.0000 / 0.0000 / -0.053 | landed (98 decisions) | orange_ball | 2.29 m | yes | yes | yes | SUCCESS | PASS |
| cap-body | Find the orange ball and land on it. | 70, 32.1, 90, 6 | 0.0000 / 0.0000 / -0.053 | landed (94 decisions) | orange_ball | 1.92 m | yes | yes | yes | SUCCESS | FAIL: success, collision, contact_from_above, landed_on |
