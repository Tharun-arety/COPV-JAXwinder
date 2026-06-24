# COPV-JAXwinder

JAX-based differentiable optimizer for composite overwrapped pressure vessel (COPV) winding process design. Pre-screens winding angle and pass-density combinations before full-fidelity ACP + Mechanical ply-by-ply verification.

![Optimization evolution](outputs/optimization_evolution.gif)

## What it does

- Optimizes helical and hoop winding controls on a fixed COPV geometry using a shell-element structural screen
- Converts the continuous winding field to a discrete course plan (helical pairs and hoop rings)
- Runs an 8-phase production-readiness pipeline from data contract to release gate
- Reports which production data is missing and blocks release accordingly — `do_not_release` is correct output when machine, material, cure, or qualification data are absent

## Shell element model

Structural screen uses CST triangle elements on a midsurface mesh of an ellipsoidal dome shell (`dome_height_ratio=0.7`). Curvature-coupled membrane strain and edge-level bending regularization handle the curved dome geometry. Replaces an earlier solid-tet model that locked in bending.

The optimizer runs L-BFGS over 18 continuous controls: 6 meridional stations × (helical angle, helical pass count, hoop pass count).

## Results

| | |
|---|---|
| Baseline FI (no overwrap) | 4021.3 |
| Optimized FI | 0.497 |
| Mass delta vs. baseline | +757% |
| Peak required friction | 0.1485 (allowable: 0.15) |
| Burst factor | 1.418 |
| Dominant failure mode | Matrix tension |

**Baseline FI:** A 4-ply axial-only base laminate under 6.85 MPa gives hoop stress ≈ 548 MPa >> YT = 70 MPa. The overwrap is the structure; the high baseline confirms correct membrane stress computation.

**Mass delta:** The optimizer uses the full production-scale pass envelope (up to 44 helical, 24 hoop passes at 0.3 mm tow) to drive FI to 0.497. Downstream machine and build-time constraints are where the mass budget is tightened — not this screen.

## Physical context

**Clairaut friction limit.** When a tow is commanded along a non-geodesic path, it slides. The 0.15 friction allowable is the approximate boundary between stable deposition and lateral tow slip in wet-winding practice. An angle profile that violates it cannot be physically executed regardless of structural model output.

**Continuous field ≠ producible program.** The optimizer returns a structurally preferred continuous angle field. Closing the gap to a machine-executable program — discrete pass counts, cut-and-restart locations, handedness balance — is the job of the discrete course planner downstream.

## Modules

| File | Purpose |
|---|---|
| `src/copv_opt/geometry.py` | Ellipsoidal dome shell. Arc-length parameterized meridian, midsurface mesh via gmsh. |
| `src/copv_opt/physics.py` | CST shell FEA. Assembles stiffness, solves CG, evaluates Hashin failure. |
| `src/copv_opt/config.py` | `GeometryConfig`, `MaterialConfig`, `FailureConfig`, `WindingOptimizationConfig` dataclasses. |
| `src/copv_opt/optimize.py` | L-BFGS winding optimizer. FPP patch and hybrid winding+patch optimizers (thesis work). |
| `src/copv_opt/course_planner.py` | Converts continuous winding field to discrete helical pairs and hoop rings. |
| `src/copv_opt/production.py` | `ProductionLineConfig` dataclass hierarchy and gap reporting. |
| `src/copv_opt/production_pipeline.py` | 8-phase production readiness pipeline. |
| `src/copv_opt/abaqus_exporter.py` | Writes Abaqus `.inp` deck from optimized result. |
| `src/copv_opt/visualize.py` | PyVista and matplotlib rendering. Optional import. |

## Running

```bash
pip install -e .

# Run in order — each script consumes the previous script's outputs
python generate_winding_verification_outputs.py
python generate_production_program_outputs.py
python generate_full_production_phase_outputs.py

# Line-config pipeline (fill template with real data first)
python generate_blackwave_target_outputs.py --config blackwave_public_line_config_template.json --output-dir outputs/blackwave_target_execution
python generate_blackwave_target_outputs.py --config blackwave_dummy_line_config.json --output-dir outputs/blackwave_dummy_demo_execution
python generate_blackwave_demo_comparison.py
```

## Tests

```bash
python -m pytest tests/ -v
```

| Test | What it checks | Requires |
|---|---|---|
| `test_shell_state_uses_triangle_elements` | Mesh produces triangle shell elements | Fresh mesh (auto-generated) |
| `test_constant_angle_shell_regression` | 42° winding clears FI < 0.75 at 8 mm band thickness | Fresh mesh |
| `test_packaged_summary_is_shell_feasible` | Committed summary passes shell constraints | `outputs/winding_first_summary.json` — skipped if absent |

First two tests build a fresh mesh in a temp directory and run the full FEA solve. Takes 2–5 minutes.

## Production pipeline integration

The pipeline requires real machine, material, cure, inspection, and qualification data. None of this is available from public sources, so the template uses null placeholders throughout. Fill `blackwave_public_line_config_template.json` with actual line data to get a meaningful release decision.

To see how the pipeline responds to filled data, run the synthetic demo first:

```bash
python generate_blackwave_target_outputs.py --config blackwave_dummy_line_config.json --output-dir outputs/blackwave_dummy_demo_execution
python generate_blackwave_demo_comparison.py
# Read: outputs/blackwave_demo_comparison.md
```

See `BLACKWAVE_VALUE_MAP.md` for handoff files and priority fields.

## Material

T700/E862 UD ply at 60% fiber volume from **NASA/TM-2013-216574, Table 2**.

| Property | Value | Unit |
|---|---|---|
| E₁₁ | 139 067 | MPa |
| E₂₂ = E₃₃ | 7 908 | MPa |
| G₁₂ = G₁₃ | 3 206 | MPa |
| G₂₃ | 2 275 | MPa |
| ν₁₂ = ν₁₃ | 0.257 | — |
| ν₂₃ | 0.30 | — |

Hashin allowables (XT=2200, XC=1400, YT=70, YC=220, S=120 MPa) are literature values for this material class. Replace with coupon-derived allowables before production use.

## Scope and limits

- Geometry is fixed during optimization (dome shape and cylinder length are parameters, not design variables)
- Discrete course variables are not inside the optimizer — discretization is a downstream step
- No liner model
- Cure, residual stress, and autofrettage are not coupled into the structural solve
- Shell-element Hashin screen is a screening method, not a certified analysis approach — ACP + Mechanical with ply-by-ply draping is required for design verification
- `do_not_release` is the correct output until real machine, material, cure, inspection, and qualification data replace the null placeholders

## Workflow position

```
This repo                         Downstream
──────────────────────────────    ──────────────────────────────────
Winding optimization (JAX)    →   Ansys ACP: ply-by-ply verification
Identifies angle sensitivity  →   Physical DOE: targeted experiments
Discrete course plan (JSON)   →   Machine post-processor: NC code
do_not_release gate + list    →   Production data fill: close gaps
```
