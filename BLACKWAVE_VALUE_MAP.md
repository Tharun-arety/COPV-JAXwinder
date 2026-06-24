# Blackwave Value Map

This file explains how the current optimizer maps to Blackwave's public COPV positioning, where it can create immediate value, and which Blackwave-only inputs are still required before it becomes a real production release tool.

## Publicly Visible Blackwave Context

The current public website indicates the following:

- Blackwave positions itself as: `To End COPV Failure In Spaceflight.`
- Blackwave describes its offering as `high-performance COPVs for spaceflight` with a focus on `performance, reliability and cost efficiency`.
- The homepage product language includes `OFF-THE-SHELF SPACE TANKS` and `MADE IN GERMANY`.
- The public location section lists:
  - `Taufkirchen` as `HEADQUARTER` with `R&T, Engineering, Offices`
  - `Garching` as `PRODUCTION SITE` with `Production`
- The public careers view shows at least these roles:
  - `CNC-Maschinenbediener`
  - `Intern - Aerospace Engineering`
  - `Working Student - Business Development`

What the public website does not publish:

- winding machine limits
- tow tension and heater process windows
- cure cycle details
- autofrettage settings
- inspection thresholds
- qualification datasets

That means the repo should not pretend to know Blackwave's proprietary line data. Instead, it should make the missing data explicit and provide a clean handoff point for loading it.

## Why This Repo Is Useful To Blackwave

For production technicians:

- it turns the winding-first design result into an execution-oriented discrete course plan
- it shows whether the release packet is blocked by missing process data instead of hiding those gaps
- it creates readable markdown outputs instead of only raw arrays or plots

For manufacturing and process engineers:

- it quantifies the current winding demand with explicit course counts, kinematic demand, deposition seed profiles, and first-order gap/overlap surrogates
- it identifies which machine and process parameters must be provided before a real line-feasibility answer is possible
- it gives one common production contract instead of separate ad hoc spreadsheets

For quality and qualification teams:

- it makes inspection gates, cure/autofrettage readiness, and qualification evidence explicit
- it forces a conservative `do_not_release` result when key evidence is missing
- it creates a traceability scaffold that can later be linked to MES, NDI, and qualification evidence

For leadership and CTO-level review:

- it separates what is already executable in software from what still depends on Blackwave-only engineering data
- it provides a structured release-gate stack instead of a single optimistic optimization score
- it reduces the risk of treating a structural screening result like a production-ready manufacturing recipe

## Where The Optimizer Fits In Blackwave's Workflow

1. Start with the fixed-geometry winding optimization.
   This finds a lower-risk winding field under the current in-repo structural screen.

2. Convert that field into discrete courses.
   This produces an execution-oriented helical and hoop plan.

3. Load Blackwave production data into the line-config template.
   This is the point where Blackwave-specific machine, material, cure, inspection, and qualification data enters the workflow.

4. Run the phase-gated production pipeline.
   This evaluates whether the current design is only structurally interesting, or actually close to a line release decision.

5. Close the missing gates one by one.
   The output tells Blackwave exactly which data package is still preventing release.

## Blackwave Handoff Files In This Repo

- `blackwave_public_line_config_template.json`
  This mirrors `ProductionLineConfig` and is the intended Blackwave data handoff file.

- `blackwave_dummy_line_config.json`
  This is a synthetic, fully filled demo profile used to prove the pipeline reacts to supplied machine/process/inspection data.

- `generate_blackwave_target_outputs.py`
  This runs the full phase stack using the selected line-config JSON.

- `generate_blackwave_demo_comparison.py`
  This compares the sparse public-target profile against the dummy-filled profile and writes a markdown summary.

- `outputs/blackwave_target_execution/`
  This is where the Blackwave-targeted phase reports are written after execution.

- `outputs/blackwave_dummy_demo_execution/`
  This is where the illustrative dummy demo reports are written.

- `outputs/blackwave_demo_comparison.md`
  This is the quickest proof that the phase stack changes its results when line data is filled.

- `outputs/production_phase_execution/`
  This is the same phase logic executed on the generic staged case.

## What Blackwave Should Fill First

Highest-value fields to replace first in `blackwave_public_line_config_template.json`:

- `machine.nominal_head_speed_mm_s`
- `machine.max_head_speed_mm_s`
- `machine.max_mandrel_rpm`
- `machine.min_turning_radius_mm`
- `material.deposition_temperature_window_c`
- `material.allowable_tension_window_n`
- `heating_compaction.target_heater_setpoint_c`
- `heating_compaction.target_compaction_force_n`
- `inspection.max_gap_mm`
- `inspection.max_overlap_mm`
- `inspection.max_wrinkle_height_mm`
- `inspection.final_ndi_method`
- `cure.steps`
- `autofrettage.target_pressure`
- `autofrettage.liner_yield_pressure`
- `qualification.coupon_dataset_path`
- `qualification.subcomponent_dataset_path`
- `qualification.vessel_dataset_path`

## What This Repo Still Does Not Do

Even after the Blackwave template is filled, the repo is not yet a full production optimizer unless these deeper work packages are implemented:

- discrete course variables inside the optimizer, not only downstream after optimization
- machine-specific inverse kinematics and NC/post-processor output
- calibrated deposition physics for the actual Blackwave material system
- a higher-fidelity as-built defect model
- coupled cure, residual stress, and autofrettage simulation
- qualification correlation against Blackwave coupon, subcomponent, and vessel data

## Recommended Blackwave Adoption Path

1. Run the dummy demo first and read `outputs/blackwave_demo_comparison.md`.
2. Fill `blackwave_public_line_config_template.json` with the real line data that production and engineering are willing to share internally.
3. Run `python generate_blackwave_target_outputs.py --config blackwave_public_line_config_template.json --output-dir outputs/blackwave_target_execution`.
4. Review `outputs/blackwave_target_execution/phase_execution_index.md`.
5. Close the highest-impact blockers in this order: machine limits, deposition windows, inspection thresholds, cure/autofrettage, qualification evidence.
6. Only after that move the discrete course planner into the optimizer core and connect the repo to Blackwave's actual machine post-processor.
