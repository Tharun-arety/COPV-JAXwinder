# CLAUDE.md

## What This Project Is

A JAX-based differentiable pre-screening optimizer for composite overwrapped pressure vessel (COPV) winding process design. It is an **experimental research tool**, not a production optimizer. Its correct position in a workflow is upstream of high-fidelity FEA (Ansys ACP + Mechanical) and upstream of machine programming.

This is a staging repo. Once validated against physical burst data and calibrated with real material allowables, the core optimizer will be integrated into the production toolchain.

---

## Architecture

### Core modules (`src/copv_opt/`)

| File | Purpose |
|---|---|
| `geometry.py` | Ellipsoidal dome COPV shell geometry. Arc-length parameterized meridian profile. Builds midsurface shell mesh via gmsh. Key: `build_copv_shell(dome_height_ratio=0.7)`. |
| `physics.py` | Shell-element FEA. CST membrane elements with curvature-coupled normal DOF and edge-level bending regularization. `build_copv_fem_state()` assembles state; `make_solve_compliance()` returns JAX-jitted CG solver. |
| `config.py` | All dataclasses: `GeometryConfig`, `MaterialConfig`, `FailureConfig`, `FrictionConfig`, `WindingOptimizationConfig`. |
| `optimize.py` | Winding process forward model and L-BFGS loop. Also contains FPP patch and hybrid (winding+patch) optimizers from M.Sc. thesis work. |
| `course_planner.py` | Converts continuous winding field to discrete helical course pairs and hoop rings. |
| `production.py` | `ProductionLineConfig` dataclass hierarchy and gap reporting. |
| `production_pipeline.py` | 8-phase production readiness pipeline (data contract → qualification). |
| `abaqus_exporter.py` | Writes Abaqus `.inp` deck from optimized winding result. |
| `visualize.py` | PyVista and matplotlib rendering utilities. Optional import. |

### Key design decisions

**Shell elements, not solid tetrahedra.** The previous version used linear tets, which lock in bending and require excessive mesh density to resolve shell stress gradients. The current CST shell elements with curvature-coupled membrane response are physically correct for thin-walled pressure vessels.

**Ellipsoidal dome, not hemispherical.** `dome_height_ratio=0.7` approximates an isotensoidal dome for the boss-to-radius ratio r₀/R ≈ 0.1 used here. A hemisphere gives the wrong meridional stress distribution and Clairaut path geometry near the boss transition.

**Continuous optimization → discrete course planner, not the reverse.** The optimizer solves continuous winding controls (angle profile, pass density per meridional station). The course planner discretizes downstream. The structural result from the optimizer and the production program handed to the machine are two different designs. This is a known approximation that is acceptable for screening but must be resolved before production use.

**`do_not_release` by design.** The production pipeline always returns `do_not_release` until real machine, material, cure, inspection, and qualification data replace the null placeholders. This is the correct behavior. It is not a failure of the demo.

---

## Scripts

| Script | What it does |
|---|---|
| `generate_winding_verification_outputs.py` | Runs winding optimization, writes `winding_first_summary.json`, VTU files, layout JSON |
| `generate_production_program_outputs.py` | Converts winding result to production program and discrete course plan |
| `generate_full_production_phase_outputs.py` | Runs all 8 phases on the generic staged case |
| `generate_blackwave_target_outputs.py` | Runs all 8 phases on a selected line-config JSON |
| `generate_blackwave_demo_comparison.py` | Compares sparse vs. filled line-config phase results |
| `generate_readme_assets.py` | Regenerates plots and GIF for the README |

---

## Running The Optimizer

```bash
pip install -e .
python generate_winding_verification_outputs.py
python generate_production_program_outputs.py
python generate_full_production_phase_outputs.py
python generate_blackwave_target_outputs.py --config blackwave_dummy_line_config.json --output-dir outputs/blackwave_dummy_demo_execution
python generate_blackwave_demo_comparison.py
```

These must be run in order — each script consumes outputs from the previous one.

---

## Running Tests

```bash
pip install -e .
python -m pytest tests/ -v
```

| Test | What it checks | Requires |
|---|---|---|
| `test_shell_state_uses_triangle_elements` | Mesh produces triangle elements, not tets | Fresh mesh (auto-generated in test) |
| `test_constant_angle_shell_regression` | 42° constant-angle winding clears FI < 0.75 at 8 mm band thickness | Fresh mesh |
| `test_packaged_summary_is_shell_feasible` | Committed summary matches shell-model constraints | `outputs/winding_first_summary.json` — **skipped automatically if absent** |

The first two tests build a fresh mesh in a temp directory and run the full FEA solve. They take roughly 2–5 minutes depending on hardware.

---

## Material Data

T700/E862 UD ply elastic constants at 60% fiber volume from **NASA/TM-2013-216574, Table 2**.

| Property | Value | Unit |
|---|---|---|
| E₁₁ | 139 067 | MPa |
| E₂₂ = E₃₃ | 7 908 | MPa |
| G₁₂ = G₁₃ | 3 206 | MPa |
| G₂₃ | 2 275 | MPa |
| ν₁₂ = ν₁₃ | 0.257 | — |
| ν₂₃ | 0.30 | — |
| ρ | 1.58 × 10⁻⁹ | t/mm³ |

Hashin allowables (XT=2200, XC=1400, YT=70, YC=220, S=120 MPa) are literature values for this material class. **Replace with Blackwave coupon-derived allowables before any production use.**

---

## Output Artifact State

The committed artifacts in `outputs/` were generated by the previous solid-element model. They will differ from a fresh run with the current shell-element code. **Regenerate all outputs** using the scripts above before comparing numbers or running `test_packaged_summary_is_shell_feasible`.

---

## Blackwave Handoff Files

| File | Purpose |
|---|---|
| `blackwave_public_line_config_template.json` | Fill this with real Blackwave machine, material, cure, inspection, and qualification data |
| `blackwave_dummy_line_config.json` | Synthetic fully-filled demo to prove the pipeline reacts to supplied data |
| `BLACKWAVE_VALUE_MAP.md` | Maps the repo to Blackwave's public COPV mission and lists highest-value fields to fill first |

---

## Known Limitations

- Discrete course variables are not inside the optimizer — they are a downstream approximation.
- No liner model. COPVs have liners; this repo does not model them.
- No machine-specific post-processor. The course plan is JSON; no NC code is emitted.
- Cure, residual stress, and autofrettage are not coupled into the structural solve.
- Geometry is fixed during optimization (dome shape and cylinder length are parameters, not design variables).
- The shell-element Hashin screen is a screening method, not a certified analysis approach. ACP + Mechanical with ply-by-ply draping is required for design verification.

---

## Intended Workflow Position

```
This repo                              Downstream tools
──────────────────────────────────     ──────────────────────────────────────
Fast winding optimization (JAX)   →   Ansys ACP: ply-by-ply verification
Identifies angle sensitivity      →   Physical DOE: targeted experiments
Discrete course plan (JSON)       →   Machine post-processor: NC code
do_not_release gate + blockers    →   Production data fill: close the gaps
```

This tool does not replace ACP, the machine post-processor, or the qualification database. It feeds them.
