# Cure and Autofrettage Input Readiness

Status: `input_stack_ready_for_coupling`
Real production gate closed: `False`
Closure basis: The phase can only audit input readiness today; a coupled cure/residual-stress/autofrettage solver is not yet implemented in-repo.

## Objective
Prepare the coupled cured/autofrettaged process state that a production-grade structural release would require.

## How It Is Done
- The phase checks whether cure steps, autofrettage target pressure, hold logic, and liner yield information are defined in the production contract.
- Where data exists, it aggregates cure exposure and pressure ratios so the repo can measure process-state readiness before adding a full coupled solver.

## Why It Is Done This Way
- Without cure kinetics and liner plasticity data, any residual-stress number would be fabricated. This phase is intentionally readiness-gated rather than numerically overconfident.
- The completeness ratio makes the missing process inputs explicit so later engineering effort goes to the right bottleneck instead of the wrong code path.

## Inputs
- `cure_cycle_name`: `DUMMY_CURE_CYCLE_V1`
- `autofrettage_model`: `dummy_elastoplastic_model`

## Metrics
- `input_completeness_ratio`: `1.0`
- `cure_step_count`: `3`
- `cure_peak_temperature_c`: `120.0`
- `cure_total_hold_time_min`: `65.0`
- `autofrettage_target_pressure`: `10.5`
- `pressure_to_autofrettage_ratio`: `0.6523809523809524`
- `operating_pressure`: `6.85`
- `allowable_pressure_with_margin`: `9.713539964813132`

## Verification
- `cure_cycle_defined`: `True`
- Detail: `{'step_count': 3}`
- `autofrettage_target_defined`: `True`
- Detail: `{'target_pressure': 10.5}`
- `operating_pressure_below_allowable_pressure`: `True`
- Detail: `{'operating_pressure': 6.85, 'allowable_pressure_with_margin': 9.713539964813132}`

## Blockers
- None.

## Artifacts
- phase_06_cure_and_autofrettage_input_readiness.json
- phase_06_cure_and_autofrettage_input_readiness.md
