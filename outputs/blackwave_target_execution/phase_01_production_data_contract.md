# Production Data Contract

Status: `implemented_with_external_gaps`
Real production gate closed: `False`
Closure basis: The contract scaffold exists, but required machine/material/process inputs are still missing.

## Objective
Define a stable machine/material/process contract that every later production phase can consume.

## How It Is Done
- The optimizer result is normalized into a structured production program with explicit geometry, screening state, line configuration, inspection gates, and execution order.
- The line configuration is checked recursively for missing leaf fields so the repo can distinguish between missing engineering data and missing code.

## Why It Is Done This Way
- A production optimizer fails in practice when downstream tooling relies on implicit assumptions. The contract is explicit so machine, cure, and inspection phases all read the same object model.
- The completeness check is recursive because later phases depend on deeply nested data such as heater limits, NDI method, and qualification evidence.

## Inputs
- `required_sections`: `['geometry', 'line_config', 'screening_snapshot', 'layout_profiles', 'process_basis', 'discrete_course_plan', 'inspection_gates', 'execution_sequence']`
- `line_name`: `BLACKWAVE_PUBLIC_TARGET_PROFILE`

## Metrics
- `contract_section_presence`: `{'geometry': True, 'line_config': True, 'screening_snapshot': True, 'layout_profiles': True, 'process_basis': True, 'discrete_course_plan': True, 'inspection_gates': True, 'execution_sequence': True}`
- `line_config_leaf_count`: `58`
- `line_config_filled_leaf_count`: `26`
- `line_config_completeness_ratio`: `0.4482758620689655`
- `missing_field_count`: `32`
- `declared_blocker_count`: `25`

## Verification
- `required_program_sections_present`: `True`
- Detail: `{'geometry': True, 'line_config': True, 'screening_snapshot': True, 'layout_profiles': True, 'process_basis': True, 'discrete_course_plan': True, 'inspection_gates': True, 'execution_sequence': True}`
- `line_config_has_declared_fields`: `True`
- Detail: `{'total_leaves': 58, 'filled_leaves': 26}`

## Blockers
- Populate the towpreg storage temperature limit.
- Populate the towpreg maximum out-time limit.
- Populate the validated deposition temperature window for the towpreg.
- Populate the validated tow tension window for the selected material system.
- Populate the nominal deposition speed for the target winding machine.
- Populate the machine maximum head speed.
- Populate the machine maximum mandrel RPM.
- Populate the minimum steerable turning radius for the line.
- Populate the nominal heater setpoint for deposition.
- Populate the nominal compaction force for the head.
- Populate the cure cycle definition, including ramp and hold steps.
- Populate the autofrettage target pressure and hold definition.
- Populate the liner yield pressure or equivalent autofrettage qualification limit.
- Populate the allowable gap threshold from the quality plan.
- Populate the allowable overlap threshold from the quality plan.
- Populate the allowable wrinkle threshold from the quality plan.
- Populate the final NDI method required for release.
- Populate the coupon qualification dataset path or evidence reference.
- Populate the subcomponent qualification dataset path or evidence reference.
- Populate the vessel-level qualification dataset path or evidence reference.
- Replace continuous pass-density optimization variables with discrete course scheduling variables, and score cuts/restarts directly inside the optimizer rather than only in downstream planning.
- Add machine-axis inverse kinematics and NC/post-processor output for the target line.
- Add an as-built thickness/defect model so structural analysis runs on predicted manufactured state.
- Couple cure, residual stress, and autofrettage into the structural workflow before release.
- Correlate the model against coupon, subcomponent, and vessel qualification data.
- Recursive contract audit still finds 32 unfilled leaf fields after excluding optional notes.

## Artifacts
- phase_01_production_data_contract.json
- phase_01_production_data_contract.md
