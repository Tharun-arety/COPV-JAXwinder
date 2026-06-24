# Machine Kinematic Demand Screen

Status: `screened_against_limits`
Real production gate closed: `False`
Closure basis: The phase quantifies kinematic demand, but a machine-specific inverse-kinematics and NC/post-processor stack is still missing.

## Objective
Quantify the geometric and rotational demand that the discrete courses place on a real winding machine.

## How It Is Done
- Each discrete helical course is converted into 3D path-demand metrics: path length, mandrel rotation, estimated time, required RPM, and local turning radius.
- If machine limits are present, those demand metrics are checked against the declared hardware envelope.

## Why It Is Done This Way
- A real winder must satisfy path demand before any structural argument matters. The analysis is demand-first because the repo still lacks a machine-specific inverse-kinematics post-processor.
- Turning radius is derived from the actual exported 3D path instead of only from angle profiles so the report reflects what the discrete plan is really asking the hardware to do.

## Inputs
- `course_count`: `352`
- `machine_name`: `DUMMY_4AXIS_COPV_WINDER`

## Metrics
- `course_count`: `352`
- `max_path_length_mm`: `887.9640045889641`
- `mean_path_length_mm`: `887.9640045889639`
- `max_required_mandrel_rpm`: `18.11738198607877`
- `min_local_turning_radius_mm`: `10.559709025611943`
- `rpm_violation_count`: `0`
- `turning_radius_violation_count`: `0`
- `nominal_head_speed_mm_s`: `140.0`
- `max_head_speed_mm_s`: `220.0`

## Verification
- `courses_have_geometry`: `True`
- Detail: `{'course_count': 352}`
- `turning_radius_computed`: `True`
- Detail: `{'finite_radius_count': 352, 'course_count': 352}`
- `mandrel_rpm_within_limit`: `True`
- Detail: `{'rpm_violations': 0, 'max_mandrel_rpm_limit': 60.0}`
- `planned_head_speed_within_machine_limit`: `True`
- Detail: `{'nominal_head_speed_mm_s': 140.0, 'max_head_speed_mm_s': 220.0}`
- `turning_radius_within_limit`: `True`
- Detail: `{'radius_violations': 0, 'minimum_turning_radius_limit_mm': 10.0}`

## Blockers
- None.

## Artifacts
- phase_03_machine_kinematic_demand_screen.json
- phase_03_machine_kinematic_demand_screen.md
