# Blackwave Value Map

Maps this repo's outputs to production integration and identifies which data must come from the line before a real release decision is possible.

## Handoff files

| File | Purpose |
|---|---|
| `blackwave_public_line_config_template.json` | Fill with real machine, material, cure, inspection, and qualification data |
| `blackwave_dummy_line_config.json` | Synthetic fully-filled demo — proves the pipeline reacts to supplied data |
| `generate_blackwave_target_outputs.py` | Runs the full phase stack for any selected line-config JSON |
| `generate_blackwave_demo_comparison.py` | Compares sparse vs. filled line-config phase results |
| `outputs/blackwave_target_execution/` | Phase reports after running the public template |
| `outputs/blackwave_dummy_demo_execution/` | Phase reports for the synthetic demo |
| `outputs/blackwave_demo_comparison.md` | Quickest proof of pipeline sensitivity to filled data |

## Priority fields to fill first

Replace these in `blackwave_public_line_config_template.json` to unblock the most pipeline gates:

```
machine.max_head_speed_mm_s
machine.max_mandrel_rpm
machine.min_turning_radius_mm
material.allowable_tension_window_n
material.deposition_temperature_window_c
material.out_time_limit_hours
heating_compaction.target_heater_setpoint_c
heating_compaction.target_compaction_force_n
inspection.max_gap_mm
inspection.max_overlap_mm
inspection.max_wrinkle_height_mm
inspection.final_ndi_method
cure.steps
autofrettage.target_pressure
autofrettage.liner_yield_pressure
qualification.coupon_dataset_path
qualification.subcomponent_dataset_path
qualification.vessel_dataset_path
```

## What the pipeline does not yet do

Even with a fully filled line-config, the following require additional implementation work before this becomes a real production optimizer:

- Discrete course variables inside the optimizer (currently only downstream)
- Machine-specific inverse kinematics and NC/post-processor output
- Calibrated deposition physics for the actual material system
- Higher-fidelity as-built defect model
- Coupled cure, residual stress, and autofrettage simulation
- Structural correlation against coupon, subcomponent, and vessel test data

## Adoption sequence

1. Run the dummy demo and read `outputs/blackwave_demo_comparison.md`
2. Fill `blackwave_public_line_config_template.json` with real line data
3. Run `python generate_blackwave_target_outputs.py --config blackwave_public_line_config_template.json --output-dir outputs/blackwave_target_execution`
4. Review `outputs/blackwave_target_execution/phase_execution_index.md`
5. Close blockers in this order: machine limits → deposition windows → inspection thresholds → cure/autofrettage → qualification evidence
