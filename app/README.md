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
requirement (V, P, R)                app/sizing.py     -> GeometryConfig
        │
        ▼
mesh + FEA state (cached per geom)    app/engine.py     -> copv_opt solver
        │
        ├── fast_screen   (winding_forward_angle + Hashin)
        └── full_optimize (run_winding_optimization)
        │
        ▼
DesignResult + release gate           app/main.py / cli.py
```

`burst factor = 1 / sqrt(FI_max)` — Hashin indices are quadratic in stress and
stress is linear in pressure, so this is the pressure multiple that drives the
failure index to 1.

## Scope

This is Phase 0: it proves the integration thesis — a spec-driven front door over
the differentiable engine with a real 3D viewport. It deliberately does **not** yet
include CAD import, machine kinematics/collision, an NC post-processor, or coupon
calibration. Those are later phases.
