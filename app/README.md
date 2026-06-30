# Tank Configurator (Phase 0)

A product layer over the `copv_opt` engine. You state a **requirement** — volume,
pressure, envelope radius — and it returns an optimized, structurally screened
wound-tank design with an interactive 3D failure-index view.

It sits **upstream of a machine-programming tool** (e.g. TaniqWind Pro): it finds
the design point and screens it; it does not emit NC code, and it always returns
`do_not_release` until real production data is supplied.

## Install

```bash
pip install -e .[app]
```

This pulls in `trame`, `trame-vuetify`, and `trame-vtk` on top of the engine
dependencies. Use the Python environment that already has JAX working.

## Run the GUI

```bash
python -m app.main
```

Opens at `http://localhost:8080`. Set a requirement in the left drawer, pick a mode,
press **Run**:

- **Fast screen** — one constant-angle forward solve (seconds). Interactive.
- **Full optimization** — the L-BFGS winding optimizer (minutes). The real design point.

The viewport colours the shell by Hashin failure index. Results (FI max, burst
factor, friction demand, mass, the release gate) appear in the drawer.

## Run headless

Same pipeline, no browser — for scripting or a quick check:

```bash
python -m app.cli --volume 9 --pressure 300 --radius 100 --angle 42 --band 8
python -m app.cli --volume 9 --pressure 300 --radius 100 --optimize --json
```

## How it fits together

```
requirement (V, P, R)                app/sizing.py        -> GeometryConfig
        │
        ▼
mesh + FEA state (cached per geom)    app/engine.py        -> copv_opt solver
        │
        ├── fast_screen   (winding_forward_angle + Hashin)
        └── full_optimize (run_winding_optimization)
        │
        ▼
DesignResult + release gate           app/main.py / cli.py
        │
        ├── app/project.py      catalog of designed tanks (save/load/list)
        ├── app/geometry_io.py  STEP export · liner mass
        ├── app/course.py       discrete course plan · kinematic demand · NC CSV
        ├── app/solver_export.py Abaqus .inp · CalculiX run
        ├── app/calibration.py  coupon allowables -> re-screen -> gate delta
        └── app/report.py       standalone HTML design report
```

`burst factor = 1 / sqrt(FI_max)` — Hashin indices are quadratic in stress and
stress is linear in pressure, so this is the pressure multiple that drives the
failure index to 1.

## Full pipeline (CLI)

```bash
python -m app.cli --volume 9 --pressure 300 --radius 100 --optimize \
    --course --export-step out/tank.step --liner 3.0 \
    --export-abaqus out/tank.inp --report out/tank.html \
    --save-project "9L 300bar" --store out/catalog

python -m app.cli --list --store out/catalog          # browse the catalog
python -m app.cli ... --allowables coupons.json        # calibrate to coupon data
```

## What is real vs. scaffolded

Real and verified: requirement sizing, fast screen + full optimization, project
catalog, STEP export, first-order liner mass, discrete course plan, first-order
kinematic demand, machine-neutral NC CSV, Abaqus `.inp` export, CalculiX
auto-detect/run, coupon-allowables calibration, HTML report.

**General axisymmetric geometry** (`app/meridian.py`, `app/general_state.py`,
`app/meridian_mesh.py`): an arbitrary axisymmetric mandrel — any meridian profile,
not just the parametric COPV — can be meshed and screened (`engine.screen_profile`,
CLI `--profile rho_z.csv`). The general FEA path is **validated to ~1.6%** against
the analytic COPV on an identical mesh (`python -m app.validate_general`). Known
limitation: the self-meshed path under-resolves the polar-opening stress
concentration, so its absolute FI runs lower than the parametric boss-refined path
(~46% on the COPV case) — use it for arbitrary-shape exploration and relative
comparison; use the parametric path for absolute COPV screening.

Honest gaps (blocked on external input, not yet built):
- **CAD file import (STEP/IGES) of a non-axisymmetric mandrel** — the engine is
  axisymmetric; truly general 3-D mandrels need a different state builder.
- **Machine-specific NC post + true collision** — needs the actual machine
  definition; `MachineLimits` fields default to "not supplied".
- **Real coupon allowables and a CalculiX/Ansys binary** — the mechanisms are
  wired; the data and solver are external.

The `do_not_release` gate stays closed until those are supplied. That is correct.
