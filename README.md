# Blackwave COPV Optimizer Staging

This repository is a Blackwave-targeted staging optimizer for composite overwrapped pressure vessels. Its job is not to produce a pretty winding plot. Its job is to help Blackwave move from a structurally promising winding concept to a production decision that is explicit, reviewable, and hard to misread.

In plain terms:

- a production technician should be able to see what runs, what is missing, and what still blocks release
- a process or manufacturing engineer should be able to see machine demand, deposition demand, and inspection gates
- a structures engineer should be able to see what the winding optimization actually changed
- a CTO or program lead should be able to see whether the current data stack supports `release` or `do_not_release`

The repo is aligned to Blackwave's public positioning around COPVs for spaceflight, but it does not invent proprietary Blackwave process windows. Instead, it provides a clean way to plug those values in.

![Optimization evolution](outputs/optimization_evolution.gif)

## Physical Context And Research Motivation

The design choices in this optimizer are grounded in physical filament winding experience: configuring winding angle, fiber tension, mandrel speed, and feed rate for CFRP pressure shell fabrication, and correlating those process settings to structural properties through NDT (ultrasonic scanning) and mechanical testing.

Two observations from that physical work determined how this tool is structured:

**The Clairaut friction limit is a real production constraint, not a model parameter.** When a tow is commanded along a non-geodesic path, it slides. The friction penalty in this optimizer's objective is the computational form of the relationship between the commanded winding angle and the tow's physical stability on the mandrel surface. The 0.15 allowable friction coefficient is not a tuning choice — it is the approximate boundary between stable deposition and lateral tow slip in wet-winding practice. An angle profile that violates this limit is an angle profile the machine cannot physically execute, regardless of what the structural model says.

**The gap between an optimal winding field and a producible winding program is not small.** An optimizer that returns a structurally ideal continuous angle profile while ignoring discrete pass counts, cut-and-restart locations, and handedness balancing is useful only as a pre-screen. The discrete course planner exists because closing that gap is real engineering work, not an afterthought that can be handled by a post-processing script.

This tool reduces the winding parameter search space before physical experiments are run. It identifies which angle profiles and pass-density combinations are most structurally sensitive, so that physical DOE resources target the parameter region most likely to produce useful information rather than being spread across a uniform grid.

## What This Repo Does

Today this repo does four useful things:

1. It optimizes winding process controls on a fixed COPV geometry using the in-repo JAX shell-element structural screen.
2. It converts the accepted continuous winding field into a discrete execution-oriented course plan.
3. It runs an 8-phase production-readiness pipeline and writes readable markdown and JSON artifacts.
4. It refuses to act production-ready when machine, process, inspection, or qualification data are still missing.

That last point is the real value. A lot of engineering code gives a better-looking layup and quietly hides the missing production evidence. This repo does the opposite.

## Why This Matters To Blackwave

Blackwave's public messaging is about ending COPV failure in spaceflight and building high-performance COPVs for spaceflight. That makes this repo useful only if it improves reliability, production clarity, and release discipline, not just if it lowers one simulated stress number.

This repo creates value for Blackwave in three ways:

- it reduces ambiguity between structural screening and production release
- it gives production and engineering one shared data contract instead of disconnected assumptions
- it turns missing machine, cure, inspection, and qualification inputs into explicit blockers

If Blackwave fills the line data template in this repo with real internal values, the same pipeline immediately becomes a much better production-decision layer.

## What Each Role Gets

| Role | What this repo gives them |
| --- | --- |
| Production technician | A readable execution stack, explicit blockers, and phase reports that show what still needs to be defined before a build should be released |
| Manufacturing / process engineer | Discrete course objects, kinematic demand, deposition seed profiles, and a clear list of missing machine and process windows |
| Quality / NDI / qualification engineer | Inspection gates, traceability scaffolding, cure/autofrettage readiness checks, and a conservative release gate |
| Structures engineer | Winding-first structural improvement on a fixed geometry plus a bridge from nominal winding field to first-order as-built surrogate |
| CTO / program lead | A single place to see whether the repo is still a screening tool or is genuinely close to a controlled production decision |

## Blackwave-Specific Files

Open these first:

- `BLACKWAVE_VALUE_MAP.md`
- `blackwave_public_line_config_template.json`
- `blackwave_dummy_line_config.json`
- `generate_blackwave_target_outputs.py`
- `generate_blackwave_demo_comparison.py`
- `outputs/blackwave_target_execution/phase_execution_index.md`
- `outputs/blackwave_dummy_demo_execution/phase_execution_index.md`
- `outputs/blackwave_demo_comparison.md`

What they are for:

- `BLACKWAVE_VALUE_MAP.md` explains how the repo maps to Blackwave's public COPV mission and what internal data Blackwave should load first.
- `blackwave_public_line_config_template.json` is the Blackwave handoff file. It mirrors `ProductionLineConfig`.
- `blackwave_dummy_line_config.json` is a synthetic fully filled demo case used only to prove that the phase pipeline reacts to supplied data.
- `generate_blackwave_target_outputs.py` runs the full phase stack for any selected line-config JSON.
- `generate_blackwave_demo_comparison.py` compares the sparse public profile against the synthetic dummy profile.
- `outputs/blackwave_target_execution/` is the Blackwave-targeted report directory after execution.
- `outputs/blackwave_dummy_demo_execution/` is the illustrative fully filled demo run.

## Public Blackwave Context Used Here

The public Blackwave website currently indicates:

- the company positions itself around ending COPV failure in spaceflight
- the public product language includes high-performance COPVs for spaceflight
- the public site markets off-the-shelf space tanks made in Germany
- the public site separates headquarters and production-site functions
- the public careers view shows both production and aerospace-engineering roles

That is enough to justify a Blackwave-targeted optimizer structure. It is not enough to justify assuming Blackwave's real process windows, machine limits, or qualification data. Those must come from Blackwave and are therefore modeled as explicit inputs, not hidden assumptions.

## What It Optimizes Today

The current packaged verification is still fixed-geometry and winding-first. `generate_winding_verification_outputs.py` solves for `18` continuous winding controls: `6` meridional control stations times `3` process parameters per station.

| Optimized process parameter | Internal field | Physical meaning | Default bounds |
| --- | --- | --- | --- |
| Helical winding angle profile | `winding_angle_ctrl` | Fiber angle relative to the local meridian, interpolated from pole to pole | `12-58 deg` |
| Helical pass-count / deposition profile | `helical_pass_ctrl` | Relative helical tow deposition at each station; converted to added thickness using `tow_thickness = 0.3 mm`, `tow_width = 12 mm`, `winding_family_count = 8`, and local radius | `0-44` continuous passes |
| Hoop pass-count / deposition profile | `hoop_pass_ctrl` | Relative hoop tow deposition at each station; active mainly on the cylinder through a smooth hoop window | `0-24` continuous passes |

Important limit:

- these are continuous screening controls, not final machine-programmed whole passes
- the structural screen now runs on a midsurface triangular shell mesh with curvature-coupled membrane response and shell-bending regularization, not on volumetric tetrahedra

That is why the repo now includes a discrete course planner and a phase-gated production pipeline.

## What The Packaged Case Proves

On the packaged winding-first case in `outputs/winding_first_summary.json`:

- the unwound shell baseline fails at reported `FI = 4021.3`
- the optimized winding result reduces that to reported `FI = 0.497`
- mass increases by `+756.9%`
- required friction coefficient is `0.1485`, still below the allowable `mu_max = 0.15`
- burst factor: `1.418`

This is the current validated story of the branch:

- fixed geometry
- winding-only structural improvement
- internal JAX shell-element screen
- accepted design that clears the active Hashin limit for the packaged case

The current branch does not claim more than that.

**Why the baseline FI is so high:**

A baseline of 4021.3 is physically correct for a 4-ply laminate carrying only axial fiber orientation under 6.85 MPa internal pressure. The hoop stress at the cylinder midsurface is approximately p × R / t = 6.85 × 96 / 1.2 ≈ 548 MPa. The matrix transverse allowable is YT = 70 MPa. Without meaningful hoop or helical overwrap, the Hashin matrix tension index exceeds the allowable by more than an order of magnitude. No structural COPV is built with only axial base plies. The overwrap is the structure. The high baseline confirms the shell-element model is computing membrane stresses correctly.

**Why the mass delta is +757%:**

The optimization bounds were set to reflect realistic COPV overwrap levels: up to 44 helical passes and 24 hoop passes, versus the placeholder bounds of 0–2 passes used in earlier runs. With 0.3 mm tow thickness and a tow-width-to-circumference deposition factor, the optimizer builds enough overwrap to drive the Hashin FI well below the allowable. A +757% mass increase on the 1.2 mm base laminate corresponds to a total wall thickness of approximately 10.3 mm on a 100 mm radius vessel at the cylinder. This is a conservative structural result: the optimizer is permitted to use the full pass envelope and does so to achieve FI = 0.497 with a burst factor of 1.418. The discrete course planner and machine kinematic constraints, which exist downstream of this screen, will tighten the pass budget and are the correct place to apply build-time mass constraints.

Why the baseline number is so high:

- the packaged shell baseline is intentionally only the `4`-ply base laminate with no meaningful overwrap, so it is a deliberately underbuilt reference state rather than a production-intent vessel

## Production Pipeline Outputs

The repo now writes production-facing artifacts, not just optimization plots.

Main output groups:

- `outputs/winding_first_production_program.json`
- `outputs/winding_first_production_readiness.md`
- `outputs/winding_first_discrete_course_plan.json`
- `outputs/winding_first_discrete_course_plan.md`
- `outputs/production_phase_execution/`
- `outputs/blackwave_target_execution/`

The most important production-facing report is:

- `outputs/production_phase_execution/phase_execution_index.md`

That file answers one management-level question quickly:

- is this case actually release-ready, or are we still looking at a structurally interesting but production-incomplete result?

For the current staged case, the answer is still `do_not_release`, which is correct and intentional.

## Dummy Demo Results

To avoid overselling the repo, there is now a fully executed illustrative dummy case in addition to the sparse public-target profile.

What the comparison proves:

- line-config completeness increases from `0.4483` on the sparse public profile to `1.0` on the dummy profile
- declared contract blockers drop from `25` to `5`
- machine kinematics moves from `demand_quantified_only` to `screened_against_limits`
- cure/autofrettage moves from `blocked_by_input_gaps` to `input_stack_ready_for_coupling`
- inspection/digital thread moves from `partial_traceability_only` to `traceability_scaffold_ready`
- external qualification evidence flips from `False` to `True`

Most important point:

- the dummy case still ends in `do_not_release`

That is not a failure of the demo. It is the proof that the pipeline is behaving honestly. Filling the inputs improves the intermediate engineering gates, but the repo still refuses to pretend that surrogate phases are equivalent to a true production release.

Read the executed comparison here:

- `outputs/blackwave_demo_comparison.md`

## Current Phase Status

The repo can already execute the following decision stack:

1. Production data contract
2. Discrete course planning
3. Machine kinematics demand screen
4. Towpreg deposition demand screen
5. First-order as-built laminate surrogate
6. Cure and autofrettage input readiness
7. Inspection and digital thread scaffold
8. Qualification and release decision

What this means in practice:

- the software can already tell Blackwave what is missing
- the software cannot yet replace Blackwave's real line knowledge

## How Blackwave Should Use It

1. Run the illustrative demo first:

```bash
python generate_blackwave_target_outputs.py --config blackwave_dummy_line_config.json --output-dir outputs/blackwave_dummy_demo_execution
python generate_blackwave_demo_comparison.py
```

2. Review `outputs/blackwave_demo_comparison.md` so everyone can see how the pipeline reacts to filled data.
3. Open `blackwave_public_line_config_template.json`.
4. Replace placeholders and `null` values with real Blackwave machine, material, cure, inspection, and qualification data.
5. Run:

```bash
python generate_blackwave_target_outputs.py --config blackwave_public_line_config_template.json --output-dir outputs/blackwave_target_execution
```

6. Review:

- `outputs/blackwave_target_execution/phase_execution_index.md`
- `outputs/blackwave_target_execution/phase_01_production_data_contract.md`
- `outputs/blackwave_target_execution/phase_08_qualification_and_release_decision.md`

7. Close the blockers in the order they appear.

If the output still says `do_not_release`, the repo is doing its job.

## Quick Start API

```python
from copv_opt import (
    GeometryConfig,
    MaterialConfig,
    WindingOptimizationConfig,
    build_copv_fem_state,
    ensure_copv_mesh,
    export_result_to_abaqus,
    make_solve_compliance,
    run_winding_optimization,
)

geom = GeometryConfig(pressure=6.85)
material = MaterialConfig()
mesh = ensure_copv_mesh("outputs/copv_shell.step", "outputs/copv_shell.msh", geom)
state = build_copv_fem_state(mesh.nodes, mesh.elems, material, geom)
solve = make_solve_compliance(state)
winding = run_winding_optimization(
    state,
    material,
    WindingOptimizationConfig(
        max_winding_thickness=18.0,
        max_helical_pass_count=44.0,
        max_hoop_pass_count=24.0,
        helical_seed_pass_count=14.0,
        hoop_seed_pass_count=2.0,
    ),
    geom,
    solve,
)["result"]
export_result_to_abaqus(
    state,
    winding,
    geom,
    "outputs/optimized_copv_winding_first.inp",
    material=material,
)
```

## Run The Main Reports

```bash
pip install -e .
python generate_winding_verification_outputs.py
python generate_production_program_outputs.py
python generate_full_production_phase_outputs.py
python generate_blackwave_target_outputs.py --config blackwave_public_line_config_template.json --output-dir outputs/blackwave_target_execution
python generate_blackwave_target_outputs.py --config blackwave_dummy_line_config.json --output-dir outputs/blackwave_dummy_demo_execution
python generate_blackwave_demo_comparison.py
python generate_readme_assets.py
```

## Scope And Limits

- The committed case is a screening-grade verification run, not a certification workflow.
- The current structural check is Hashin-based screening, not a full certification burst workflow.
- The geometry is fixed during optimization.
- The optimizer still solves continuous winding controls and only later discretizes them.
- The machine post-processor is not yet Blackwave-machine-specific.
- Deposition physics is still a demand screen, not a calibrated Blackwave material process model.
- Cure, residual stress, and autofrettage are not yet fully coupled into the structural solve.
- Qualification closure still depends on real Blackwave coupon, subcomponent, and vessel evidence.
- The committed output artifacts in `outputs/` were generated by the previous solid-element model and will differ from a fresh run with the current shell-element code. Regenerate with `python generate_winding_verification_outputs.py` followed by the production pipeline scripts before comparing results.

## From JAX Screening To ACP Verification

This tool and Ansys ACP + Mechanical are not alternatives. They work in sequence.

The JAX optimizer is fast and differentiable. It can evaluate thousands of angle-thickness combinations in the time ACP takes to run one ply-by-ply draping analysis. Its role is to reduce the design space to a small number of candidates worth full-fidelity verification — not to replace the verification.

The intended sequence:

1. Run the JAX optimizer. Identify the angle profiles and pass-density combinations with the lowest failure index at the lowest mass.
2. Export the Abaqus `.inp` file from the best candidate.
3. Import into ACP as the layup definition. Add ply draping, discrete ply-by-ply Hashin analysis, boss region mesh refinement, and proper through-thickness integration.
4. Compare the ACP ply failure maps to the JAX screening result. Where they diverge significantly, the divergence tells you which modelling assumption matters most — and which physical experiment would resolve it.
5. Feed physical coupon and burst test data back into the qualification section of the line-config template to close the production gate.

**Thesis connection:** The FPP patch optimization code in `src/copv_opt/optimize.py` — `patch_forward()`, `run_patch_optimization()`, `accumulate_patch_stiffness()`, `hybrid_forward()` — is the computational output of M.Sc. thesis research in fiber patch placement optimization. The hybrid winding-plus-patch optimizer combines continuous helical/hoop winding with localized FPP reinforcement in a single differentiable objective. This is the longer-term research direction: once the base winding optimization is validated against physical burst data, FPP reinforcement at the boss transition becomes the next structural efficiency lever.

## Bottom Line

This repo is already useful to Blackwave if the goal is to make COPV production decisions more disciplined and less ambiguous.

It is not yet a final Blackwave line optimizer.

The dummy demo now proves the software stack is working: when the line config is filled, the intermediate engineering phases improve exactly where they should.

It becomes one only when Blackwave loads its real process data into the provided line-config template and closes the remaining production gates with actual machine, inspection, cure, and qualification evidence.
