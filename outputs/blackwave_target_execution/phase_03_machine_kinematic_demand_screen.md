# Machine Kinematic Demand Screen

Status: `demand_quantified_only`
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
- `machine_name`: `BLACKWAVE_PUBLIC_PLACEHOLDER_WINDER`

## Metrics
- `course_count`: `352`
- `max_path_length_mm`: `887.9640045889641`
- `mean_path_length_mm`: `887.9640045889639`
- `max_required_mandrel_rpm`: `None`
- `min_local_turning_radius_mm`: `10.559709025611943`
- `rpm_violation_count`: `0`
- `turning_radius_violation_count`: `0`
- `nominal_head_speed_mm_s`: `None`
- `max_head_speed_mm_s`: `None`

## Verification
- `courses_have_geometry`: `True`
- Detail: `{'course_count': 352}`
- `turning_radius_computed`: `True`
- Detail: `{'finite_radius_count': 352, 'course_count': 352}`
- `mandrel_rpm_within_limit`: `None`
- Detail: `{'rpm_violations': 0, 'max_mandrel_rpm_limit': None}`
- `planned_head_speed_within_machine_limit`: `None`
- Detail: `{'nominal_head_speed_mm_s': None, 'max_head_speed_mm_s': None}`
- `turning_radius_within_limit`: `None`
- Detail: `{'radius_violations': 0, 'minimum_turning_radius_limit_mm': None}`

## Blockers
- Machine nominal head speed is not defined, so time and RPM demand cannot be converted into a planned cycle-rate check.
- Machine maximum head speed is not defined, so kinematic demand can only be quantified, not accepted/rejected.
- Machine mandrel RPM limit is not defined, so rotational feasibility cannot be closed.
- Machine minimum turning radius is not defined, so steering feasibility cannot be closed.

## Artifacts
- phase_03_machine_kinematic_demand_screen.json
- phase_03_machine_kinematic_demand_screen.md
