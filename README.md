# COPV-JAXwinder

A winding process optimizer and production readiness checker for composite overwrapped pressure vessels.

![Optimization evolution](outputs/optimization_evolution.gif)

## What this does

The winding design for a COPV — which fiber angle, how many passes, how dense — determines whether the vessel holds pressure, whether the fiber stays on the mandrel, and whether the build is ready to release. Getting that wrong in a physical trial is expensive.

This tool does three things before any physical trial runs:

1. **Finds the winding design with the lowest structural failure risk** for a given geometry and pressure, in minutes rather than the hours a full Ansys ACP setup takes. The result is a starting point for ACP verification, not a replacement for it.
2. **Converts that design to a discrete course plan** — the specific helical passes and hoop rings a winding machine would execute, not just a continuous angle field.
3. **Runs a production readiness checklist** across 8 phases (data contract → release gate) and names every missing input that blocks release. The output is always `do_not_release` until real machine, material, cure, inspection, and qualification data replaces the placeholders. That is intentional.

---

## Who reads what

| Role | Open this file | It answers |
|---|---|---|
| Production / quality engineer | `outputs/production_phase_execution/phase_execution_index.md` | Which phases are blocked and what data is missing |
| Simulation / structures engineer | `outputs/winding_first_summary.json`, winding plots in `outputs/` | Which winding angle and pass count to bring into ACP first |
| Manufacturing / process engineer | `outputs/winding_first_production_readiness.md` | Discrete course count, machine kinematic demand, missing process windows |
| Anyone evaluating the pipeline | `outputs/blackwave_demo_comparison.md` | How the phase results change when real line data is loaded |

---

## What a pipeline run looks like

The table below is real output from `outputs/blackwave_demo_comparison.md`. Left column is the pipeline with only public data (25 blockers). Right column is with a fully filled synthetic demo config (5 blockers). Both still end in `do_not_release` — because surrogate phases and absent qualification evidence correctly block release even when machine and process data are present.

| Phase | No line data (25 blockers) | Full demo data (5 blockers) |
|---|---|---|
| Data contract | gaps present | gaps present |
| Discrete course planning | ✓ computed | ✓ computed |
| Machine kinematics | demand only | ✓ screened against limits |
| Deposition physics | seed screen only | seed screen only |
| As-built surrogate | first-order only | first-order only |
| Cure / autofrettage | **blocked** — inputs absent | inputs ready |
| Inspection / digital thread | partial traceability | scaffold ready |
| Release decision | **do_not_release** | **do_not_release** |

Filling in the machine and process data closes phases 3, 6, and 7. The final release gate stays closed because qualification evidence (coupon, subcomponent, vessel data) is not present. That is the correct behaviour — the pipeline will not pretend that a structurally screened design is a qualified one.

To see this yourself:
```bash
pip install -e .
python generate_blackwave_target_outputs.py --config blackwave_dummy_line_config.json --output-dir outputs/blackwave_dummy_demo_execution
python generate_blackwave_demo_comparison.py
# Read: outputs/blackwave_demo_comparison.md
```

To load real line data, fill `blackwave_public_line_config_template.json` and re-run with that config. See `BLACKWAVE_VALUE_MAP.md` for priority fields.

---

## Where this sits in the workflow

```
This tool                         What comes after
──────────────────────────────    ──────────────────────────────────
Winding optimization (minutes)→   Ansys ACP: ply-by-ply verification
Narrows angle/pass candidates →   Physical DOE: fewer, better-aimed tests
Discrete course plan (JSON)   →   Machine post-processor: NC code
do_not_release gate + list    →   Production data fill: close the gaps
```

---

## Technical reference

The sections below are for people reading or modifying the code.

### Shell element model

Structural screen uses CST triangle elements on a midsurface mesh of an ellipsoidal dome shell (`dome_height_ratio=0.7`). Curvature-coupled membrane strain and edge-level bending regularization handle the curved dome geometry. Replaces an earlier solid-tet model that locked in bending.

The optimizer runs L-BFGS over 18 continuous controls: 6 meridional stations × (helical angle, helical pass count, hoop pass count).

### Results

| | |
|---|---|
| Baseline FI (no overwrap) | 4021.3 |
| Optimized FI | 0.497 |
| Mass delta vs. baseline | +757% |
| Peak required friction | 0.1485 (effective optimizer limit: 0.1275, hard allowable: 0.15) |
| Burst factor | 1.418 |
| Dominant failure mode | Matrix tension |

**Baseline FI:** A 4-ply axial-only base laminate under 6.85 MPa gives hoop stress ≈ 548 MPa >> YT = 70 MPa. The overwrap is the structure; the high baseline confirms correct membrane stress computation.

**Mass delta:** The optimizer uses the full production-scale pass envelope (up to 44 helical, 24 hoop passes at 0.3 mm tow) to drive FI to 0.497. At +757%, the total wall is approximately 10.3 mm giving t/R ≈ 0.103, which is at the conventional thin-shell validity boundary (t/R < 0.1). Results at this wall fraction should be verified with a thick-shell or solid-element model before use in structural sizing. Downstream machine and build-time constraints are where the mass budget is tightened — not this screen.

### Physical context

**Clairaut friction limit.** When a tow is commanded along a non-geodesic path, it slides. The 0.15 friction allowable is the approximate boundary between stable deposition and lateral tow slip in wet-winding practice. An angle profile that violates it cannot be physically executed regardless of structural model output. The optimizer targets an effective limit of 0.1275 (85% of allowable) to ensure actual headroom.

**Continuous field ≠ producible program.** The optimizer returns a structurally preferred continuous angle field. Closing the gap to a machine-executable program — discrete pass counts, cut-and-restart locations, handedness balance — is the job of the discrete course planner downstream.

### Modules

| File | Purpose |
|---|---|
| `src/copv_opt/geometry.py` | Ellipsoidal dome shell. Arc-length parameterized meridian, midsurface mesh via gmsh. |
| `src/copv_opt/physics.py` | CST shell FEA. Assembles stiffness, solves CG, evaluates Hashin failure. |
| `src/copv_opt/config.py` | `GeometryConfig`, `MaterialConfig`, `FrictionConfig`, `WindingOptimizationConfig` dataclasses. |
| `src/copv_opt/optimize.py` | L-BFGS winding optimizer. FPP patch and hybrid winding+patch optimizers (thesis work). |
| `src/copv_opt/course_planner.py` | Converts continuous winding field to discrete helical pairs and hoop rings. |
| `src/copv_opt/production.py` | `ProductionLineConfig` dataclass hierarchy and gap reporting. |
| `src/copv_opt/production_pipeline.py` | 8-phase production readiness pipeline. |
| `src/copv_opt/abaqus_exporter.py` | Writes Abaqus `.inp` deck from optimized result. |
| `src/copv_opt/visualize.py` | PyVista and matplotlib rendering. Optional import. |

### Running

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

### Tests

```bash
python -m pytest tests/ -v
```

| Test | What it checks | Requires |
|---|---|---|
| `test_matrix_tension_at_allowable` | Hashin matrix tension formula is correct at YT | No mesh |
| `test_fiber_tension_at_allowable` | Hashin fiber tension formula is correct at XT | No mesh |
| `test_friction_safety_factor_reduces_effective_limit` | Optimizer-facing friction limit < hard allowable | No mesh |
| `test_course_planner_warns_on_empty_layout` | Course planner raises named warning on missing keys | No mesh |
| `test_shell_state_uses_triangle_elements` | Mesh produces triangle shell elements | Fresh mesh (auto-generated) |
| `test_constant_angle_shell_regression` | 42° winding clears FI < 0.75 at 8 mm band thickness | Fresh mesh |
| `test_packaged_summary_is_shell_feasible` | Committed summary passes shell constraints | `outputs/winding_first_summary.json` — skipped if absent |

### Material

T700/E862 UD ply at 60% fiber volume from **NASA/TM-2013-216574, Table 2**.

| Property | Value | Unit |
|---|---|---|
| E₁₁ | 139 067 | MPa |
| E₂₂ = E₃₃ | 7 908 | MPa |
| G₁₂ = G₁₃ | 3 206 | MPa |
| G₂₃ | 2 275 | MPa |
| ν₁₂ = ν₁₃ | 0.257 | — |
| ν₂₃ | 0.30 | — |

Hashin allowables (XT=2200, XC=1400, YT=70, YC=220, S=120 MPa) are literature values. Replace with coupon-derived allowables before production use.

### Scope and limits

- Geometry is fixed during optimization (dome shape and cylinder length are parameters, not design variables)
- Discrete course variables are not inside the optimizer — discretization is a downstream step
- No liner model
- Cure, residual stress, and autofrettage are not coupled into the structural solve
- Shell-element Hashin screen is a screening method, not a certified analysis approach — ACP + Mechanical with ply-by-ply draping is required for design verification
- `do_not_release` is the correct output until real machine, material, cure, inspection, and qualification data replace the null placeholders
