# Discrete Course Planning

Status: `computed`
Real production gate closed: `True`
Closure basis: Explicit course objects were exported without planner warnings.

## Objective
Translate the continuous winding screening field into explicit helical course pairs and hoop ring objects.

## How It Is Done
- The continuous helical pass profile is quantized into balanced clockwise/counter-clockwise course pairs.
- Short inactive gaps are merged and short active segments are removed so the exported plan avoids unrealistic fragments.
- The hoop profile is quantized separately into ring bands on the cylindrical span.

## Why It Is Done This Way
- The current optimizer is still axisymmetric, so the planner discretizes downstream instead of pretending the continuous field is already machine-ready.
- Balanced handedness preserves the structural intent of the screening solution while making the plan closer to what a real winding schedule would execute.

## Inputs
- `sample_point_count`: `320`

## Metrics
- `total_course_pairs`: `176`
- `total_individual_courses`: `352`
- `total_hoop_rings`: `24`
- `total_cut_restart_events`: `0`
- `helical_pass_rmse`: `5.617333549722722e-16`
- `hoop_pass_rmse`: `0.00033119687606903526`
- `helical_activation_balance_cv`: `0.0`

## Verification
- `helical_course_pairs_exported`: `True`
- Detail: `176`
- `helical_quantization_error_finite`: `True`
- Detail: `5.617333549722722e-16`
- `hoop_quantization_error_finite`: `True`
- Detail: `0.00033119687606903526`

## Blockers
- None.

## Artifacts
- phase_02_discrete_course_planning.json
- phase_02_discrete_course_planning.md
