# Cure and Autofrettage Input Readiness

Status: `blocked_by_input_gaps`
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
- `cure_cycle_name`: `UNSPECIFIED_CURE`
- `autofrettage_model`: `None`

## Metrics
- `input_completeness_ratio`: `0.0`
- `cure_step_count`: `0`
- `cure_peak_temperature_c`: `None`
- `cure_total_hold_time_min`: `0.0`
- `autofrettage_target_pressure`: `None`
- `pressure_to_autofrettage_ratio`: `None`
- `operating_pressure`: `6.85`
- `allowable_pressure_with_margin`: `9.713539964813132`

## Verification
- `cure_cycle_defined`: `False`
- Detail: `{'step_count': 0}`
- `autofrettage_target_defined`: `False`
- Detail: `{'target_pressure': None}`
- `operating_pressure_below_allowable_pressure`: `True`
- Detail: `{'operating_pressure': 6.85, 'allowable_pressure_with_margin': 9.713539964813132}`

## Blockers
- Cure cycle steps are missing, so degree-of-cure and residual thermal history cannot be simulated.
- Autofrettage target pressure is missing, so the liner/plastic pre-stress state cannot be evaluated.
- Liner yield pressure is missing, so the autofrettage window cannot be validated.

## Artifacts
- phase_06_cure_and_autofrettage_input_readiness.json
- phase_06_cure_and_autofrettage_input_readiness.md
