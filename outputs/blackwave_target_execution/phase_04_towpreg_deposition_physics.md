# Towpreg Deposition Physics

Status: `relative_seed_screening_only`
Real production gate closed: `False`
Closure basis: This phase outputs relative setpoint seeds only; the repo still lacks a calibrated towpreg heating, tack, and compaction model.

## Objective
Translate the discrete course plan into deposition demand metrics and relative process setpoint seeds.

## How It Is Done
- The discrete helical and hoop build profiles are combined into a local layer-build demand profile.
- Relative tension and compaction seed profiles are derived from that local build profile so thicker regions can be handled more conservatively.
- The staged friction screen is carried forward as the current slip-risk metric until calibrated towpreg deposition physics are added.

## Why It Is Done This Way
- The repo does not yet have calibrated towpreg tack, heating, and compaction physics, so this phase reports demand-side surrogates instead of invented release values.
- Using dimensionless seed profiles preserves the ordering of process demand without pretending that the current code knows your machine's exact tension or heater recipe.

## Inputs
- `material_name`: `BLACKWAVE_PUBLIC_PLACEHOLDER_MATERIAL`
- `sample_point_count`: `320`

## Metrics
- `peak_required_friction_coefficient`: `0.08160163462162018`
- `friction_headroom`: `0.06839836537837982`
- `max_nominal_layer_build`: `1.3644131659530103`
- `max_discrete_layer_build`: `1.25`
- `deposition_continuity_index`: `1.0`
- `relative_tension_seed_min`: `0.8`
- `relative_tension_seed_max`: `0.8`
- `relative_compaction_seed_min`: `1.0`
- `relative_compaction_seed_max`: `1.0`

## Verification
- `friction_headroom_positive`: `True`
- Detail: `{'headroom': 0.06839836537837982}`
- `quantization_error_within_helical_warning`: `True`
- Detail: `{'rmse': 0.05653648540623992, 'warning_limit': 0.1}`
- `relative_tension_seed_bounded`: `True`
- Detail: `{'min_seed': 0.8, 'max_seed': 0.8}`

## Blockers
- No validated tow tension window is defined, so deposition tension can only be expressed as a relative seed profile.
- No validated deposition temperature window is defined, so heater demand can only be reported qualitatively.

## Artifacts
- phase_04_towpreg_deposition_physics.json
- phase_04_towpreg_deposition_physics.md
