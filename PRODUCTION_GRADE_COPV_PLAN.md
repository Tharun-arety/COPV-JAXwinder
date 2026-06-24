# Production-Grade COPV Execution Plan

This file replaces vague "future work" with an execution order for turning the current winding-first screening repo into a production-facing COPV optimizer.

## Non-Negotiable Reality

No code-only change can make this repo directly runnable on a real COPV line without machine-specific, material-specific, and qualification-specific calibration. A production release requires at least:

- the exact towpreg material system and its storage, out-time, deposition, and cure limits
- the real winding machine axis limits, head hardware limits, and controller semantics
- gap, overlap, wrinkle, and void acceptance thresholds from the intended quality plan
- structural correlation against as-built coupons, subcomponents, and full vessel tests

The goal of this plan is therefore:

1. convert the optimizer from a smooth screening field into a machine-program generator
2. couple that machine program to as-built manufacturing physics
3. gate the released result through inspection and qualification evidence

## Current Repo State

Today the repo is still a winding-first structural screen:

- the optimizer solves continuous angle and pass-density fields
- a downstream discrete course planner now exists, but it is not yet the optimizer's native design variable set
- the manufacturability screen is friction-only
- cure, autofrettage, residual stress, and machine kinematics are not yet in the optimization loop

## Execution Order

### Phase 1: Production Data Contract

Status: `COMPLETED`

Deliverables:

- define the production line configuration object: machine, head, material, inspection, cure
- export the current optimizer result as a production-program scaffold instead of only an Abaqus shell deck
- generate a readiness report that states the remaining blockers before line release

Why first:

- every later step needs a stable contract for machine limits and process limits
- without a program artifact, there is nothing to hand to path planning, inspection, or MES tooling

Execution in this workspace:

- `src/copv_opt/production.py`
- `generate_production_program_outputs.py`
- `generate_full_production_phase_outputs.py`
- `outputs/winding_first_production_program.json`
- `outputs/winding_first_production_readiness.md`
- `outputs/production_phase_execution/phase_01_production_data_contract.md`
- `outputs/production_phase_execution/phase_execution_index.md`

### Phase 2: Discrete Course Planner

Status: `EXECUTED AS DOWNSTREAM SURROGATE`

Deliverables:

- replace continuous pass-density controls with discrete helical and hoop course objects
- resolve add/drop, cut/restart, do-not-cross, and family sequencing rules
- add alternating handedness balancing and boss-zone course templates

Code impact:

- replace `helical_pass_ctrl` and `hoop_pass_ctrl` as the primary optimization variables
- introduce course scheduling and discrete manufacturability penalties

Release gate:

- exported program has explicit course count, execution order, and cut/restart locations

Executed in this workspace:

- `src/copv_opt/course_planner.py`
- `outputs/winding_first_discrete_course_plan.json`
- `outputs/winding_first_discrete_course_plan.md`
- `outputs/production_phase_execution/phase_02_discrete_course_planning.md`

### Phase 3: Machine Kinematics and NC-Level Path Planning

Status: `EXECUTED AS KINEMATIC DEMAND SCREEN`

Deliverables:

- solve inverse kinematics for the target winder or robot
- enforce axis stroke, rotation speed, acceleration, and head orientation limits
- compute machine-feasible start, stop, dwell, and transition motions

Code impact:

- add a machine post-processor instead of treating path geometry as sufficient
- make cycle time a real optimization objective

Release gate:

- every exported course is executable by the intended controller without manual rework

Executed in this workspace:

- `src/copv_opt/production_pipeline.py`
- `outputs/production_phase_execution/phase_03_machine_kinematics.json`
- `outputs/production_phase_execution/phase_03_machine_kinematics.md`
- current implementation quantifies path demand and checks declared limits, but it does not yet solve machine-specific inverse kinematics or emit NC/controller output

### Phase 4: Towpreg Deposition Physics

Status: `EXECUTED AS RELATIVE SETPOINT SCREEN`

Deliverables:

- model tension, heating, tack, compaction, and slip for the selected towpreg system
- calibrate minimum steering radius, allowable tension window, and deposition temperature window
- add tension taper logic for layer buildup and boss transitions

Code impact:

- replace the current friction-only screen with a deposition model
- add process setpoints as first-class optimization variables

Release gate:

- optimizer outputs machine setpoints, not just geometry fields

Executed in this workspace:

- `src/copv_opt/production_pipeline.py`
- `outputs/production_phase_execution/phase_04_towpreg_deposition_physics.json`
- `outputs/production_phase_execution/phase_04_towpreg_deposition_physics.md`
- current implementation produces relative tension/compaction demand seeds and carries forward the friction screen, but it does not yet contain a calibrated towpreg deposition model

### Phase 5: As-Built Thickness and Defect Model

Status: `EXECUTED AS FIRST-ORDER SURROGATE`

Deliverables:

- predict local thickness from discrete courses and real overlap logic
- predict or detect gap, overlap, wrinkle, tow-break, and resin-rich regions
- convert nominal paths into as-built laminate fields before structural solve

Code impact:

- add an as-built laminate builder between the program generator and FEA
- store nominal and as-built states separately

Release gate:

- structural analysis runs on predicted as-built laminate state, not the commanded ideal state

Executed in this workspace:

- `src/copv_opt/production_pipeline.py`
- `outputs/production_phase_execution/phase_05_as_built_laminate_surrogate.json`
- `outputs/production_phase_execution/phase_05_as_built_laminate_surrogate.md`
- current implementation converts the discrete schedule into a first-order thickness/gap/overlap surrogate, not a calibrated defect-growth model

### Phase 6: Cure, Residual Stress, and Autofrettage Coupling

Status: `EXECUTED AS INPUT READINESS GATE`

Deliverables:

- add cure cycle definition and degree-of-cure / thermal history simulation
- propagate cure-induced residual stress into the overwrap and liner
- add autofrettage and post-autofrettage residual state to the structural workflow

Code impact:

- split structural evaluation into at least: deposition state, cured state, autofrettaged state, burst state

Release gate:

- failure margins include process-induced residual state, not just pressure loading

Executed in this workspace:

- `src/copv_opt/production_pipeline.py`
- `outputs/production_phase_execution/phase_06_cure_and_autofrettage_coupling.json`
- `outputs/production_phase_execution/phase_06_cure_and_autofrettage_coupling.md`
- current implementation audits cure/autofrettage inputs and computes readiness metrics, but it does not yet solve the coupled cured/autofrettaged residual state

### Phase 7: Inspection and Digital Thread

Status: `EXECUTED AS DIGITAL THREAD SCAFFOLD`

Deliverables:

- ingest inline inspection for gaps, overlaps, and visible wrinkle defects
- connect process IDs, spool lots, cure records, and inspection outputs to each vessel program
- compare planned versus measured thickness and defect state

Code impact:

- add inspection ingestion and acceptance checks
- add persistent traceability artifacts for each optimized build

Release gate:

- released program includes measurable acceptance criteria and an as-built comparison route

Executed in this workspace:

- `src/copv_opt/production_pipeline.py`
- `outputs/production_phase_execution/phase_07_inspection_and_digital_thread.json`
- `outputs/production_phase_execution/phase_07_inspection_and_digital_thread.md`
- current implementation builds the traceability record and checks declared acceptance thresholds, but it does not yet ingest live line inspection or connect to a production MES/QMS

### Phase 8: Qualification and Production Optimization Loop

Status: `EXECUTED AS DO-NOT-RELEASE GATE`

Deliverables:

- correlate model predictions with coupon, ring, dome, and full-vessel tests
- tune penalties and process windows against real scrap, cycle time, and quality yield
- optimize mass, cycle time, inspection risk, and process robustness together

Code impact:

- add calibration datasets and regression harnesses
- move from single-objective structural screening to production trade-space optimization

Release gate:

- optimizer is trusted enough to release production candidates under engineering change control

Executed in this workspace:

- `src/copv_opt/production_pipeline.py`
- `outputs/production_phase_execution/phase_08_qualification_and_release_decision.json`
- `outputs/production_phase_execution/phase_08_qualification_and_release_decision.md`
- current implementation closes the loop by explicitly evaluating internal gate closure plus external qualification evidence; for the staged case it correctly returns `do_not_release`

## Immediate Blockers Outside the Repo

The following inputs must come from manufacturing and materials engineering before the optimizer can be called production-grade:

- selected towpreg product and datasheet-controlled limits
- actual winder or robot model and post-processor format
- heater and compaction hardware capability envelope
- cure recipe and allowable thermal deviations
- acceptance thresholds for gaps, overlaps, wrinkles, and voids
- liner material and autofrettage process specification
- qualification matrix for coupon-to-vessel correlation

## What Was Executed Now

- Added a production-program API that exports the staged winding result as a machine-program scaffold.
- Added a readiness report generator so the repo states what is still missing before production deployment.
- Added a discrete course planner that converts the staged continuous winding profiles into explicit helical course pairs and hoop ring bands.
- Added an executable phase pipeline that runs phases 1 through 8 for the staged winding-first case and emits per-phase JSON and markdown artifacts.
- Generated packaged outputs for the staged winding-first case so the transition is visible on disk.

## Next Recommended Implementation Step

Close the loop by moving the new discrete course planner into the optimizer itself. That means replacing continuous pass-density optimization variables with discrete course scheduling variables and adding cut/restart, balance, and cycle-time penalties directly to the objective. After that, the highest-value next closure is a machine-specific inverse-kinematics and NC/post-processor stack, because phases 3 through 8 remain honest surrogates until the courses are executable on the target line.
