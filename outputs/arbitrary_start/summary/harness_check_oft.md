# harness_check_oft

Plan `arbitrary_start_harness_check.json`, policy `oft`, evaluator `generalized_surface_v1`. 3 episodes, 3 flown, 3 with every check passed.

| Episode | Sentence | Start asked (x, y, yaw, height) | Start error (xy m / yaw deg / height m) | What happened | Landed on | Distance from the middle | Inside region | Physical landing | Latched | Mission | Checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| as-30100-000-blue_pad-land | Find the blue landing pad and land on it. | 33.2, 77.4, -45.24, 7.2 | 0.0000 / 0.0000 / -0.072 | landed (73 decisions) | blue_pad | 2.12 m | yes | yes | yes | SUCCESS | PASS |
| as-30100-001-blue_pad-land | Find the blue landing pad and land on it. | -9.62, 80.91, 40.88, 6.3 | 0.0000 / 0.0000 / -0.057 | landed (42 decisions) | blue_pad | 4.88 m | yes | yes | yes | SUCCESS | PASS |
| as-30100-002-blue_pad-land | Find the blue landing pad and land on it. | 23.83, 81.42, 149.07, 8.74 | 0.0000 / 0.0000 / -0.080 | landed (41 decisions) | blue_pad | 2.32 m | yes | yes | yes | SUCCESS | PASS |
