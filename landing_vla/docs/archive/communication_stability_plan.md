# Communication Stability Validation Plan

Goal: Verify a stable live Project AirSim → AeroVLA → UAV movement path with the existing checkpoints.

Spec: User-requested Gates A–E: direct reconnect, bounded resource pressure, conditional Windows gateway, one live VLA action, then exactly ten decisions. Final measurements are recorded in [final_closed_loop_validation.md](final_closed_loop_validation.md).

This is the user-authorized feasibility probe. Implement experimental integration files only; preserve upstream and previous measurements. No new model/simulator downloads, training, or Failure-aware implementation.

- [x] Gate A: Model-free direct Windows-host NNG reconnect, 20 short-idle and 5 long-idle trials. Log TCP preflight, per-stage latency and full failures. Save Windows TCP/firewall/IP and WSL socket state. Native direct and prior reverse-relay paths must be distinguished.
- [x] Gate B: Only if A passes, allocate bounded RAM pressure and approximately7GiB GPU tensor pressure. Record physical available/commit/pagefile and stop increasing allocation with insufficient headroom.
- [x] Gate C: Only if direct client remains unstable, use a single official Windows NNG client and a minimal binary TCP gateway. Standard library TCP avoids adding ZeroMQ dependencies absent from both existing runtimes. If Windows inbound is blocked, Windows initiates the simple gateway link to WSL; simulator NNG stays entirely on Windows. Preserve image bytes without lossy JPEG or base64. Validate100 observations and20 bounded actions.
- [x] Gate D: Reuse the verified NF4 loader and exact observation/action preprocessing. Record one live step and require actual movement before E.
- [x] Gate E: Exactly10 decisions, internal simulator control/hover between decisions. Stop on collision, invalid state, timeout, OOM or simulator failure. Save distinct live frames/actions and resource measurements.
- [x] Add communication_stability_test.md, gateway_evaluation.md, final_closed_loop_validation.md. Check raw files, test results and process cleanup before reporting.

Review focus: stale NNG ownership after reload; stage latency vs execution duration; ambiguous quaternion/axis units; out-of-range altitude or nonfinite output; request replay after transport loss. Commands must never retry automatically after an uncertain execution result.


Results: A20/20 + idle5/5; B2/4GiB RAM and7GiB GPU passed,8/12/16GiB RAM skipped for commit headroom; C NOT REQUIRED; D PASS; E10/10 PASS. Decision FEASIBLE WITH LIMITATIONS. No additional feasibility episode after completion.
