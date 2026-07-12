# Professional COPV Studio Architecture

Date: 2026-07-01

## Purpose

COPV Studio must become a professional, domain-specific CAE and optimization tool for composite overwrapped pressure vessels. It is not a generic prototype viewer, and it is not a pile of scripts. It should behave like a focused ANSYS/Abaqus/HyperWorks-class workflow for one domain:

```text
COPV geometry + liner/boss + composite laminate + winding physics + optimization + simulation verification + production handoff
```

The product goal is to reduce the number of expensive high-fidelity simulations, physical trials, and manufacturing iterations needed before production qualification. It must not pretend to replace certification or physical qualification. Its job is to produce fewer, better, well-documented candidate designs for high-fidelity verification and testing.

## Product Principle

The application should have one professional shell:

```text
python -m copv_studio
```

Inside that shell, the user moves through a complete engineering workflow:

```text
Project -> Requirements -> Geometry -> Liner/Boss -> Materials -> Laminate -> Winding -> Optimization -> Simulation -> Results -> Verification -> Export
```

The current codebase has useful backend pieces, but too many separate front doors:

- static demo HTML
- live trame app
- CLI
- exported HTML viewer
- production phase scripts
- verification scripts

Those should become internal tools behind a single project-centered application.

## Non-Negotiable Scope

COPV Studio must support:

- real design inputs, not hardcoded demo values
- save/load project files
- parametric COPV CAD generation
- material and coupon allowable management
- CLT and ply-level mechanics
- liner and overwrap load-sharing
- winding path generation
- winding manufacturability checks
- optimization over geometry, laminate, winding, and course variables
- fast screening FEA
- high-fidelity solver export
- result visualization
- engineering report generation
- production readiness gating

## Optimization Is The Core, Not A Feature

The central value proposition is:

```text
Use optimization and fast physics to reduce the number of expensive ACP/Abaqus/Ansys/full-test iterations.
```

The optimization workflow must therefore be first-class in the UI and backend.

### Optimization Levels

#### Level 0: Fast Parametric Screening

Purpose:

- quickly reject bad designs
- estimate mass, burst factor, failure index, friction demand
- explore radius, length, pressure, wall thickness, angle, band thickness

Variables:

- cylinder length
- dome ratio
- wall thickness
- helical angle
- hoop thickness
- simple pass counts

Outputs:

- feasible/infeasible
- FI max
- reserve factor
- burst factor
- mass
- friction margin

#### Level 1: Continuous Winding Field Optimization

Purpose:

- optimize spatially varying winding angle and pass-density profiles
- reduce failure index and mass before discrete course planning

Variables:

- helical angle profile `alpha(s)`
- helical pass density profile
- hoop pass density profile
- thickness distribution

Constraints:

- Hashin reserve factor
- friction/slip bound
- thickness caps
- smoothness
- manufacturable angle bounds
- mass bound

Current repo status:

- partially implemented in `src/copv_opt/optimize.py`
- must be integrated into one professional workflow

#### Level 2: Discrete Course Optimization

Purpose:

- close the gap between continuous optimized fields and machine-executable winding

Variables:

- integer helical course pairs
- hoop ring count
- handedness order
- circumferential phase offsets
- start/stop regions
- cut/restart locations
- boss turnaround rules

Constraints:

- coverage
- gaps
- overlaps
- course balance
- minimum course length
- minimum restart spacing
- steering radius
- machine axis limits
- cycle time

Current repo status:

- downstream discrete planner exists
- optimizer does not yet optimize discrete courses directly

Required upgrade:

```text
continuous field -> discrete course schedule -> as-built laminate -> structural solve -> optimizer loop
```

#### Level 3: Manufacturing-Physics Optimization

Purpose:

- optimize not just structure, but manufacturable process robustness

Variables:

- tow tension setpoints
- deposition speed
- heater temperature
- compaction force
- tow width/spread model
- hoop/helical sequencing

Constraints:

- tack/slip margin
- compaction window
- temperature window
- out-time
- wrinkle risk
- band spreading
- gap/overlap acceptance
- machine speed/RPM limits

Current repo status:

- relative setpoint screen only
- no calibrated towpreg/prepreg/wet-winding physics yet

#### Level 4: Verification-Aware Optimization

Purpose:

- reduce expensive solver/test iterations by ranking candidates before ACP/Abaqus/Ansys

Workflow:

```text
generate candidate population
screen with fast shell model
filter manufacturability
select top candidates
export to Abaqus/ACP/Ansys
import or record high-fidelity results
update surrogate/correlation model
rerun optimization
```

Optimization objectives:

- minimize mass
- maximize burst factor
- maximize minimum reserve factor
- minimize friction/slip demand
- minimize gaps/overlaps
- minimize cycle time
- maximize robustness to material/process variation
- minimize expected number of failed high-fidelity checks

## Professional Application Architecture

Target structure:

```text
copv_studio/
  __main__.py
  app.py
  project/
    schema.py
    io.py
    validation.py
  ui/
    shell.py
    panels/
      requirements_panel.py
      geometry_panel.py
      liner_boss_panel.py
      materials_panel.py
      laminate_panel.py
      winding_panel.py
      optimization_panel.py
      mesh_panel.py
      solve_panel.py
      results_panel.py
      verification_panel.py
      export_panel.py
  core/
    requirements.py
    geometry_cad.py
    liner.py
    materials.py
    laminate.py
    winding_paths.py
    course_schedule.py
    as_built.py
    meshing.py
    solver_fast.py
    failure.py
    optimization.py
    verification.py
    report.py
    export.py
  data/
    materials_default.json
    standards_templates.json
  examples/
    type3_9l_300bar.copv.json
    type4_70mpa_h2.copv.json
```

Existing modules should be migrated into this structure rather than duplicated.

## Project File

A professional tool needs a single project file. Proposed extension:

```text
.copv.json
```

Minimum schema:

```json
{
  "project": {
    "name": "type3_9l_300bar",
    "units": "mm_MPa_N",
    "created_with": "COPV Studio"
  },
  "requirements": {
    "fluid": "air_or_nitrogen",
    "working_pressure_mpa": 30.0,
    "proof_pressure_mpa": 45.0,
    "burst_factor_required": 2.25,
    "cycle_life": null,
    "standard_basis": null
  },
  "geometry": {
    "vessel_type": "Type3",
    "outer_radius_mm": 100.0,
    "cylinder_length_mm": 220.0,
    "dome_profile": "ellipsoidal",
    "dome_height_ratio": 0.7,
    "opening_radius_mm": 10.0
  },
  "liner": {
    "enabled": true,
    "material": "AL6061-T6",
    "thickness_mm": 3.0,
    "model": "elastic_thin_wall"
  },
  "materials": {
    "composite": "T700_E862_default",
    "allowables_source": "literature_or_coupon"
  },
  "laminate": {
    "base_plies": [],
    "course_stack_source": "winding"
  },
  "winding": {
    "tow_width_mm": 12.0,
    "tow_thickness_mm": 0.3,
    "families": [],
    "course_schedule": []
  },
  "optimization": {
    "level": "continuous_winding",
    "objectives": ["mass", "failure_index", "friction"],
    "constraints": {}
  },
  "mesh": {},
  "load_cases": [],
  "results": {}
}
```

## UI Layout

The professional UI should look like an engineering tool:

```text
Top toolbar:
  New | Open | Save | Import CAD | Generate CAD | Mesh | Solve | Optimize | Export

Left model tree:
  Project
  Requirements
  Geometry
  Liner/Boss
  Materials
  Laminate
  Winding
  Optimization
  Mesh
  Load Cases
  Results
  Verification

Center viewport:
  CAD / mesh / winding / contour visualization

Right properties panel:
  editable properties for selected tree item

Bottom console:
  solve log, warnings, blockers, convergence history
```

## Core Modules

### Geometry CAD

Use CadQuery/OpenCascade for:

- parametric Type 3/4 vessel shell
- liner solid/surface
- boss opening
- dome profiles
- STEP export
- later STEP/IGES import

Initial geometry modes:

- ellipsoidal dome
- spherical dome
- custom meridian points
- imported axisymmetric meridian CSV

### Materials

Must distinguish:

- fiber
- resin
- cured lamina
- towpreg
- prepreg tape
- wet winding resin bath
- liner material
- boss material

Material records must include:

- elastic constants
- density
- allowables
- ply thickness
- tow width
- tow thickness
- fiber volume fraction
- storage/out-time
- cure cycle
- process windows

### Laminate

Must support:

- CLT `Q`, `Qbar`, `ABD`
- symmetric/balanced checks
- ply-by-ply stress recovery
- Hashin/Tsai-Wu/Puck-ready interfaces
- laminate table UI
- local element stack from winding courses

Current repo status:

- standalone CLT is implemented
- must be connected to the main shell solver and winding stack

### Winding

Must support:

- geodesic path generation
- non-geodesic path generation
- Clairaut relation checks
- winding angle convention
- helical families
- hoop bands
- boss turnaround
- course schedule
- band footprint mapping
- gaps/overlaps
- friction/slip demand

### Meshing

Must support:

- shell mesh
- boss refinement
- dome refinement
- mesh quality metrics
- convergence runs
- later solid/submodel mesh

### Solver

Solver layers:

1. analytical thin-cylinder checks
2. CLT cylinder section checks
3. fast shell FEA
4. exported high-fidelity Abaqus/Ansys/ACP deck
5. later progressive damage / fatigue / stress rupture modules

### Results

Result fields:

- displacement
- stress resultants
- ply stress
- Hashin modes
- reserve factor
- laminate thickness
- winding angle
- gap/overlap
- friction demand
- mass distribution
- optimization history

## Optimization Panel

The optimization UI must allow:

- select optimization level
- choose objectives
- choose constraints
- choose variables
- set bounds
- run optimization
- pause/stop
- view convergence
- inspect candidate table
- compare candidate designs
- promote selected candidate to project baseline

Candidate table columns:

```text
candidate_id
mass
FI_max
min_reserve_factor
burst_factor
mu_required
max_gap
max_overlap
cycle_time_estimate
course_count
status
```

## Candidate Reduction Workflow

The professional workflow should explicitly reduce checks:

```text
1. User defines broad design space.
2. Tool generates candidate designs.
3. Fast solver screens hundreds of candidates.
4. Manufacturing filters remove impossible candidates.
5. Optimizer refines top candidates.
6. Tool exports only top 3-10 candidates to high-fidelity verification.
7. User records/imports high-fidelity results.
8. Tool updates correlation and recommends next candidates.
```

This is the core business value.

## Release Gate Philosophy

The tool should have multiple gates:

```text
screening_pass
optimization_candidate
solver_export_ready
manufacturing_trial_ready
qualification_ready
production_release_ready
```

The current `do_not_release` behavior is correct, but too blunt. A professional tool needs more nuanced states:

- `screen_failed`
- `screen_passed_uncorrelated`
- `candidate_for_high_fidelity_verification`
- `candidate_for_manufacturing_trial`
- `blocked_missing_material_data`
- `blocked_missing_machine_data`
- `blocked_missing_qualification_evidence`
- `production_release_ready`

## Migration Plan From Current Repo

### Keep

- `src/copv_opt/clt.py`
- `src/copv_opt/liner.py`
- `src/copv_opt/physics.py`
- `src/copv_opt/geometry.py`
- `src/copv_opt/optimize.py`
- `src/copv_opt/course_planner.py`
- `src/copv_opt/production_pipeline.py`
- `src/copv_opt/abaqus_exporter.py`

### Refactor

- `app/studio_unified.py` becomes a temporary bridge, not the final app.
- `app/studio_app.py`, `app/main.py`, and `app/export_results.py` should be absorbed into one professional app shell.
- CLI scripts become batch-mode commands behind the project schema.

### Remove From Main User Path

- static demos should move to `examples/demo_static/`
- generated outputs should not define architecture
- one-off generation scripts should become commands or test fixtures

## Implementation Phases

### Phase 1: Project-Centered Studio Shell

Deliverables:

- single app entry point
- project file schema
- save/load
- model tree
- properties panel
- viewport
- solve log
- fast solve using existing engine
- optimization button using existing optimizer

Exit criteria:

- user can define a COPV project, save it, reload it, solve it, optimize it, and view results in one app

### Phase 2: Professional Geometry Module

Deliverables:

- CadQuery/OpenCascade geometry builder
- geometry tree
- dome profile options
- liner/boss parameterization
- STEP export
- mesh generation from project geometry

Exit criteria:

- geometry is no longer scattered between app sizing and backend shell builders

### Phase 3: Material + Laminate Workbench

Deliverables:

- material database UI
- coupon allowable import
- CLT table
- ply stack editor
- liner-overwrap CLT check
- laminate report

Exit criteria:

- user can inspect laminate mechanics before FEA

### Phase 4: Winding Workbench

Deliverables:

- winding path editor
- geodesic/non-geodesic checks
- helical/hoop families
- course schedule editor
- footprint/gap/overlap map
- winding visualization

Exit criteria:

- user can see the exact winding plan that creates the laminate

### Phase 5: Optimization Workbench

Deliverables:

- optimization setup UI
- objective/constraint selection
- continuous winding optimization
- discrete course candidate generation
- candidate table
- convergence plots
- candidate comparison

Exit criteria:

- user can reduce a broad design space to ranked candidates

### Phase 6: Verification + Export

Deliverables:

- Abaqus/Ansys/ACP export
- report generation
- high-fidelity result tracking
- verification matrix
- gate state management

Exit criteria:

- top candidates are ready for external high-fidelity verification

### Phase 7: Manufacturing Physics

Deliverables:

- towpreg/prepreg/wet-winding process models
- tension/heating/compaction windows
- machine kinematic model
- as-built laminate prediction
- inspection data ingestion

Exit criteria:

- tool can recommend manufacturing-trial candidates, not just structural candidates

## Immediate Next Engineering Task

Stop building new UI prototypes. Build the foundation:

```text
1. Create `copv_studio/project/schema.py`
2. Create a `.copv.json` project schema
3. Create `copv_studio/app.py` as the single app entry point
4. Move current solve/optimize calls behind a `ProjectRunner`
5. Build the first professional shell around the project model
```

Only after the project model exists should additional panels be built.

## Definition Of Done For A Professional First Version

The first professional version is done when a user can:

1. Open COPV Studio.
2. Create a project.
3. Define geometry, liner, material, laminate, winding, and load case.
4. Run a fast solve.
5. Run optimization.
6. Compare candidates.
7. Promote a candidate.
8. Export STEP, Abaqus deck, course plan, and report.
9. Save the project.
10. Reopen the project and reproduce the result.

That is the minimum bar for "proper tool."
