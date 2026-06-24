# As-Built Laminate Surrogate

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
- `thickness_rmse_mm`: `0.012292443842731227`
- `max_gap_mm`: `0.04130345710790656`
- `max_overlap_mm`: `0.010297292611411435`
- `discrete_to_nominal_mass_ratio`: `0.9987670754140638`
- `max_total_thickness_mm`: `1.6512533234687994`

## Verification
- `total_thickness_nonnegative`: `True`
- Detail: `{'minimum_total_thickness_mm': 1.2610831119771104}`
- `gap_within_limit`: `True`
- Detail: `{'max_gap_mm': 0.04130345710790656, 'gap_limit_mm': 0.08}`
- `overlap_within_limit`: `True`
- Detail: `{'max_overlap_mm': 0.010297292611411435, 'overlap_limit_mm': 0.05}`

## Blockers
- None.

## Artifacts
- phase_05_as_built_laminate_surrogate.json
- phase_05_as_built_laminate_surrogate.md
