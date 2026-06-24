# First-Order As-Built Laminate Surrogate

Status: `first_order_surrogate`
Real production gate closed: `False`
Closure basis: The current implementation builds a first-order as-built thickness surrogate, not a calibrated manufactured-state laminate model.

## Objective
Build a first-order as-built laminate state from the discrete course plan before running higher-fidelity process physics.

## How It Is Done
- The continuous nominal winding thickness profile is rescaled by the ratio between quantized discrete passes and continuous screening passes.
- That yields a first-order discrete thickness field, plus gap and overlap fields derived from the difference between nominal and quantized build.

## Why It Is Done This Way
- A real production optimizer must reason about the manufactured laminate, not just the commanded design field. This surrogate is the lightest honest bridge from discrete plan to as-built state.
- The implementation uses pass-ratio rescaling instead of pretending to know full resin flow or tow compaction mechanics, because those models are not yet calibrated in-repo.

## Inputs
- `base_thickness_mm`: `1.2`
- `sample_point_count`: `320`

## Metrics
- `thickness_rmse_mm`: `9.397498358398144e-05`
- `max_gap_mm`: `0.0`
- `max_overlap_mm`: `0.0003502861335995533`
- `discrete_to_nominal_mass_ratio`: `1.0000028875164932`
- `max_total_thickness_mm`: `23.225142003662896`

## Verification
- `total_thickness_nonnegative`: `True`
- Detail: `{'minimum_total_thickness_mm': 5.511580619646053}`
- `gap_within_limit`: `None`
- Detail: `{'max_gap_mm': 0.0, 'gap_limit_mm': None}`
- `overlap_within_limit`: `None`
- Detail: `{'max_overlap_mm': 0.0003502861335995533, 'overlap_limit_mm': None}`

## Blockers
- Gap acceptance threshold is missing, so the surrogate gap field cannot be accepted or rejected against production criteria.
- Overlap acceptance threshold is missing, so the surrogate overlap field cannot be accepted or rejected against production criteria.

## Artifacts
- phase_05_first_order_as_built_laminate_surrogate.json
- phase_05_first_order_as_built_laminate_surrogate.md
