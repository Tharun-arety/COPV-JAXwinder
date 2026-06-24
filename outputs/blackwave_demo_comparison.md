# Blackwave Dummy Demo Comparison

This report compares the sparse public-target profile against a fully filled illustrative dummy line configuration.

## Why This Exists
- The sparse public profile proves the gating logic is conservative when key production data is missing.
- The dummy profile proves the pipeline reacts when machine, process, cure, inspection, and qualification inputs are supplied.
- The dummy profile is synthetic. It demonstrates software behavior, not production readiness.

## Contract Fill Improvement
- Public-target line-config completeness: `0.4482758620689655`
- Dummy-demo line-config completeness: `1.0`
- Public-target declared blocker count: `25`
- Dummy-demo declared blocker count: `5`

## Phase Comparison
| Phase | Public-target status | Public blockers | Dummy-demo status | Dummy blockers |
| --- | --- | ---: | --- | ---: |
| `phase_01` | `implemented_with_external_gaps` | `26` | `implemented_with_external_gaps` | `5` |
| `phase_02` | `computed` | `0` | `computed` | `0` |
| `phase_03` | `demand_quantified_only` | `4` | `screened_against_limits` | `0` |
| `phase_04` | `relative_seed_screening_only` | `2` | `relative_seed_screening_only` | `0` |
| `phase_05` | `first_order_surrogate` | `2` | `first_order_surrogate` | `0` |
| `phase_06` | `blocked_by_input_gaps` | `3` | `input_stack_ready_for_coupling` | `0` |
| `phase_07` | `partial_traceability_only` | `3` | `traceability_scaffold_ready` | `0` |
| `phase_08` | `do_not_release` | `2` | `do_not_release` | `1` |

## Dummy Demo Highlights
- Phase 03 closes all machine-input blockers and computes `max_required_mandrel_rpm = 18.11738198607877` with `turning_radius_violation_count = 0`.
- Phase 04 runs against declared deposition windows and reports `friction_headroom = 0.0015188694000244085`.
- Phase 05 runs against declared acceptance limits with `max_gap_mm = 0.0` and `max_overlap_mm = 0.0003502861335995533`.
- Phase 06 becomes input-ready with `input_completeness_ratio = 1.0` and `cure_step_count = 3`.

## Release Decision
- Public-target release ready: `False`
- Dummy-demo release ready: `False`
- The dummy case still ends in `do_not_release` because the repo is intentionally conservative about surrogate phases such as discrete planning, deposition physics, as-built prediction, and qualification closure.
- That is the expected and correct behavior. The demo proves the phase stack works and that stronger input data materially changes the intermediate results.

## Output Locations
- `outputs/blackwave_target_execution/`
- `outputs/blackwave_dummy_demo_execution/`
