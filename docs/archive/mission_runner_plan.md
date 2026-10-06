# Interactive Mission Runner Implementation Plan

User-approved scope: interactive overview/target selection, AeroVLA-only navigation,
mission outcome evaluation, optional hover/landing phases, existing Gaussian Blur.
No new failure types, detector/recovery, training, map downloads or benchmark loops.

## Verified implementation basis

- Current prompt: `make_prompt` uses upstream delimiters and coarse body-frame direction.
- Target coordinates must replace the prior synthetic point. They are not numerical input tokens.
- `parse_action` recognizes LAND; the demo maps stop to hover. Landing fallback must be labelled.
- Local Project AirSim client exposes object poses/bounds, camera GetImages and SetPose.
- Actual Blocks probe returned RGB and depth-planar with matching timestamp/pose.
- Depth deprojection at two platform pixels agrees with `TemplateCube_Rounded_1` top z=-2.5m.

## Tasks and verification

- [x] Geometry: add Overview-only config, pixel/depth/pose deprojection and projection.
  Test pixel/world round trip, invalid depth and unchanged Front/Down/vehicle config.
- [x] Mission core: one target, state transitions, distance, bounded duration/steps,
  hover stability, landing acknowledgement, terminal outcome preserved during cleanup.
  Test idle/select/start/reset/abort, distance, timeout, collision and success conditions.
- [x] Control/UI: map click carries exact frame ID; G/H/L requests are separate from
  Blur B/1/2/3; R resets only idle/terminal missions; display pose/trajectory/goal/result.
  Test stale frame rejection, coordinate acceptance, blur/mission independence.
- [x] Runner: reuse NF4 loader, existing preprocessing/injector/action adapter;
  all navigation velocities originate in model output. Manager only evaluates and
  triggers hover/landing after arrival. One native NNG owner, Windows file-only UI.
- [x] Launcher: reuse tested PowerShell 5.1/7 quoting, file controls, worker identity
  and termination. No inline Python `-c`, no new dependencies.
- [x] Live gates in order: map/target; one GO_TO OFF; only if success, HOVER OFF;
  only if success, LAND OFF; only on a proven mission, optional Blur test.
  Keep failures and stop progression if a prerequisite mission fails.
- [x] Review, regression, representative screenshots, one usage/results document,
  preserve historical baseline and push tested commits to main.

## Limits to state explicitly

- Depth-selected point is a visible surface; sky/invalid/steep points are rejected.
- Hover goal z is the current flight level; high surfaces outside the existing safe
  altitude envelope are rejected. This is not terrain-aware motion planning.
- Input only contains coarse goal direction and existing template, not arbitrary
  language grounding. No manager steering correction to manufacture success.
- Goal cleanup landing cannot change FAILED/ABORTED into SUCCESS.
