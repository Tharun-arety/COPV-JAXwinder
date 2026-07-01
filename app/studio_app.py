"""COPV Studio Pro — real-engine stress-analysis desktop application (trame).

Unlike the browser demos (app/demo/*.html), this drives the ACTUAL JAX shell-element
solver on a real COPV design and post-processes the REAL result fields. It is the
tool a stress engineer would sit in front of:

* real geometry inputs (radii, lengths, boss, wall) and pressure
* editable coupon allowables (load your own -> the calibration that makes it a
  primary tool rather than a literature screen)
* real solve (constant-angle screen) and real optimization (L-BFGS over the winding)
* real post-processing: failure index, all four Hashin modes, reserve factor,
  deformation, thickness, winding angle — in an interactive 3D contour
* reserve-factor margins table and the honest release gate
* mesh-convergence verification
* one-click Abaqus/ACP deck export and an HTML report

It stays honest: the release gate reads do_not_release until coupon allowables,
ACP cross-validation, and burst-test correlation are supplied. Those are wired as
first-class steps, not hidden.

Run:
    pip install -e .[app]
    python -m app.studio_app          # http://localhost:8080
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import signal as _signal_mod
import threading as _threading

# gmsh.initialize() installs a SIGINT handler via signal.signal(), which raises
# "signal only works in main thread" when the solve runs in a worker thread. In a
# web server that Ctrl+C handler is irrelevant, so make signal.signal() a no-op off
# the main thread. Must be installed before any solve triggers gmsh.
_real_signal = _signal_mod.signal
def _thread_safe_signal(sig, handler):
    if _threading.current_thread() is _threading.main_thread():
        return _real_signal(sig, handler)
    return None
_signal_mod.signal = _thread_safe_signal

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import pyvista as pv
    from pyvista.trame.ui import plotter_ui
    from trame.app import asynchronous, get_server
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout
    from trame.widgets import vuetify3 as v3
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "COPV Studio Pro needs the app extras:  pip install -e .[app]\n"
        f"(missing: {exc.name})"
    ) from exc

from app.engine import fast_screen, full_optimize, mesh_convergence
from copv_opt.config import FailureConfig, GeometryConfig, MaterialAllowables, MaterialConfig

server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

pv.OFF_SCREEN = True
plotter = pv.Plotter(off_screen=True)
plotter.background_color = "white"
plotter.add_text("Define the design and press Solve", font_size=11, name="hint")

STORE: dict = {"result": None, "poly": None}


def _defaults():
    state.update({
        # geometry (real COPV design inputs, mm / MPa)
        "outer_radius": 100.0, "cyl_length": 220.0, "wall_thickness": 8.0,
        "opening_radius": 10.0, "dome_ratio": 0.7, "pressure": 6.85, "tank_type": "Type 3 (Al liner)",
        # material allowables (MPa) — editable; load coupon-derived values here
        "xt": 2200.0, "xc": 1400.0, "yt": 70.0, "yc": 220.0, "s": 120.0,
        # analysis
        "mode": "Constant-angle screen", "angle_deg": 42.0, "band_mm": 8.0,
        "running": False, "status": "Idle. Define the design and run an analysis.",
        "have_result": False,
        # results
        "field_options": ["Failure index (Hashin)"], "field": "Failure index (Hashin)",
        "r_fi_max": "—", "r_rf": "—", "r_defmax": "—", "r_burst": "—",
        "r_mu": "—", "r_mode": "—", "r_mass": "—", "decision": "—", "blockers": [],
        # verification
        "conv_text": "", "export_status": "",
    })


_defaults()


def _geom() -> GeometryConfig:
    return GeometryConfig(
        outer_radius=float(state.outer_radius), cylinder_length=float(state.cyl_length),
        thickness=float(state.wall_thickness), opening_radius=float(state.opening_radius),
        dome_height_ratio=float(state.dome_ratio), pressure=float(state.pressure),
    )


def _failure_cfg() -> FailureConfig:
    allow = MaterialAllowables(xt=float(state.xt), xc=float(state.xc),
                               yt=float(state.yt), yc=float(state.yc), s=float(state.s))
    return FailureConfig(allowables=allow, margin_of_safety=1.0)


def _compute(spec: dict):
    geom, material, failure = spec["geom"], MaterialConfig(), spec["failure"]
    if spec["mode"] == "Optimize winding":
        r = full_optimize(geom, material, failure_cfg=failure)
    else:
        r = fast_screen(geom, material, spec["angle"], spec["band"], failure_cfg=failure)
    # element-averaged deformation field for contouring
    dnode = np.asarray(r.disp_node)
    r.fields["Total deformation [mm]"] = dnode[r.elems].mean(axis=1)
    return r


def _build_poly(r):
    elems = np.asarray(r.elems, dtype=np.int64)
    faces = np.hstack([np.full((len(elems), 1), 3, dtype=np.int64), elems]).ravel()
    poly = pv.PolyData(np.asarray(r.nodes, dtype=np.float64), faces)
    for name, arr in r.fields.items():
        poly.cell_data[name] = np.asarray(arr, dtype=np.float64)
    return poly


def _draw(field: str):
    r, poly = STORE["result"], STORE["poly"]
    if poly is None or field not in poly.cell_data:
        return
    vals = np.asarray(poly.cell_data[field])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    plotter.clear()
    plotter.add_mesh(poly, scalars=field, clim=(lo, hi), cmap="turbo", show_edges=False,
                     scalar_bar_args={"title": field, "fmt": "%.3g", "n_labels": 6})
    plotter.view_isometric()
    plotter.reset_camera()
    ctrl.view_update()


def _apply(r):
    m = r.margins
    with state:
        state.field_options = list(r.fields.keys())
        state.field = "Failure index (Hashin)"
        state.r_fi_max = f"{r.fi_max:.3f}"
        state.r_rf = f"{m['min_reserve_factor']:.3f}"
        state.r_defmax = f"{m['max_deformation_mm']:.3f} mm"
        state.r_burst = f"{r.burst_factor:.3f}×"
        state.r_mu = "—" if r.mu_max_required is None else f"{r.mu_max_required:.3f} / {r.mu_allowable:.2f}"
        state.r_mode = m["critical_mode"]
        state.r_mass = f"{r.mass_metric:.4g}"
        state.decision = r.gate["decision"].upper()
        state.blockers = list(r.gate["blockers"])
        state.have_result = True


async def _run():
    spec = {"geom": _geom(), "failure": _failure_cfg(), "mode": state.mode,
            "angle": float(state.angle_deg), "band": float(state.band_mm)}
    with state:
        state.running = True
        state.status = ("Meshing + optimizing (full FEA, minutes)…" if spec["mode"] == "Optimize winding"
                        else "Meshing + solving…")
    loop = asyncio.get_event_loop()
    try:
        r = await loop.run_in_executor(None, _compute, spec)
    except Exception as exc:
        traceback.print_exc()
        with state:
            state.running = False; state.status = f"Error: {exc}"
        return
    STORE["result"] = r
    STORE["poly"] = _build_poly(r)
    _apply(r)
    _draw("Failure index (Hashin)")
    with state:
        state.running = False
        state.status = (f"Solved · FI_max {r.fi_max:.3f} · min RF {r.margins['min_reserve_factor']:.2f} · "
                        f"burst {r.burst_factor:.2f}× · gate {r.gate['decision']}")


@ctrl.set("run")
def run():
    if not state.running:
        asynchronous.create_task(_run())


@state.change("field")
def _on_field(field, **_):
    if STORE["poly"] is not None:
        _draw(field)


async def _run_convergence():
    with state:
        state.running = True; state.status = "Running mesh-convergence study (several solves)…"
    loop = asyncio.get_event_loop()
    try:
        rows = await loop.run_in_executor(
            None, mesh_convergence, _geom(), MaterialConfig(), float(state.angle_deg), float(state.band_mm))
    except Exception as exc:
        with state:
            state.running = False; state.status = f"Convergence error: {exc}"
        return
    lines = ["h_max   elems    FI_max   min RF"]
    for r in rows:
        lines.append(f"{r['mesh_hmax']:>5.0f}  {r['elements']:>6d}   {r['fi_max']:>6.3f}   {r['min_reserve_factor']:>5.2f}")
    with state:
        state.conv_text = "\n".join(lines)
        state.running = False; state.status = "Mesh-convergence study complete."


@ctrl.set("converge")
def converge():
    if not state.running:
        asynchronous.create_task(_run_convergence())


@ctrl.set("export_deck")
def export_deck():
    r = STORE["result"]
    if r is None or r.winding_result is None:
        state.export_status = "Optimize first — Abaqus/ACP export needs an optimized result."
        return
    from app.solver_export import export_abaqus
    out = Path("outputs") / "studio_export" / "copv_optimized.inp"
    export_abaqus(r.state, r.winding_result, r.geom, out, r.material, heading="COPV Studio Pro export")
    state.export_status = f"Abaqus/ACP deck written: {out}"


@ctrl.set("export_report")
def export_report():
    r = STORE["result"]
    if r is None:
        state.export_status = "Run an analysis first."
        return
    from app.report import write_html
    from app.sizing import SizingReport
    sizing = SizingReport(cylinder_length_mm=float(state.cyl_length), inner_radius_mm=r.geom.inner_radius,
                          design_pressure_mpa=float(state.pressure), dome_volume_litres=0.0,
                          cylinder_volume_litres=0.0, achieved_volume_litres=0.0,
                          slenderness_l_over_d=0.0)
    out = Path("outputs") / "studio_export" / "copv_report.html"
    write_html(out, "COPV Studio Pro design", r, sizing)
    state.export_status = f"Report written: {out}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _num(model, label):
    v3.VTextField(v_model=(model,), label=label, type="number", density="compact",
                  variant="outlined", hide_details=True, classes="mb-2")


def _metric(label, key):
    with v3.VRow(classes="ma-0"):
        with v3.VCol(cols=7, classes="pa-1 text-caption text-medium-emphasis"):
            v3.VLabel(label)
        with v3.VCol(cols=5, classes="pa-1 text-right font-weight-bold"):
            v3.VLabel(f"{{{{ {key} }}}}")


with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("COPV Studio Pro — stress analysis")
    with layout.toolbar:
        v3.VSpacer()
        v3.VProgressCircular(indeterminate=("running",), v_show=("running",), size=22, width=3, classes="mr-3")
        v3.VChip("{{ decision }}", v_show=("have_result",), color="warning", size="small", classes="mr-2")

    with layout.drawer as drawer:
        drawer.width = 380
        with v3.VContainer(classes="pa-3"):
            with v3.VExpansionPanels(multiple=True, model_value=([0, 3],)):
                # Geometry
                with v3.VExpansionPanel(title="Geometry (COPV design)"):
                    with v3.VExpansionPanelText():
                        _num("outer_radius", "Outer radius [mm]")
                        _num("cyl_length", "Cylinder length [mm]")
                        _num("wall_thickness", "Structural wall [mm]")
                        _num("opening_radius", "Boss opening radius [mm]")
                        _num("dome_ratio", "Dome height ratio")
                        _num("pressure", "Design pressure [MPa]")
                        v3.VSelect(v_model=("tank_type",), items=("['Type 3 (Al liner)', 'Type 4 (PE liner)']",),
                                   label="Tank type", density="compact", variant="outlined", hide_details=True)
                # Material
                with v3.VExpansionPanel(title="Material allowables [MPa]"):
                    with v3.VExpansionPanelText():
                        v3.VCardSubtitle("Load coupon-derived values to calibrate.", classes="px-0 pb-2 text-caption")
                        _num("xt", "XT — fibre tension")
                        _num("xc", "XC — fibre compression")
                        _num("yt", "YT — matrix tension")
                        _num("yc", "YC — matrix compression")
                        _num("s", "S — shear")
                # Analysis
                with v3.VExpansionPanel(title="Analysis"):
                    with v3.VExpansionPanelText():
                        v3.VSelect(v_model=("mode",), items=("['Optimize winding', 'Constant-angle screen']",),
                                   label="Mode", density="compact", variant="outlined", hide_details=True, classes="mb-2")
                        _num("angle_deg", "Screen angle [deg]")
                        _num("band_mm", "Screen band [mm]")
                        v3.VBtn("Solve", click=ctrl.run, color="primary", block=True,
                                loading=("running",), classes="mt-1 mb-2")
                        v3.VBtn("Mesh-convergence study", click=ctrl.converge, variant="tonal",
                                block=True, size="small")
                        v3.VCardText("{{ conv_text }}", v_show=("conv_text.length > 0",),
                                     classes="pa-2 mt-2 text-caption",
                                     style="white-space: pre; font-family: monospace; background: rgba(0,0,0,0.04); border-radius: 6px;")
                # Results
                with v3.VExpansionPanel(title="Results · margins"):
                    with v3.VExpansionPanelText():
                        v3.VSelect(v_model=("field",), items=("field_options",), label="Contour field",
                                   density="compact", variant="outlined", hide_details=True, classes="mb-3")
                        _metric("FI max (Hashin)", "r_fi_max")
                        _metric("Min reserve factor", "r_rf")
                        _metric("Critical mode", "r_mode")
                        _metric("Max deformation", "r_defmax")
                        _metric("Burst factor", "r_burst")
                        _metric("Friction μ req / allow", "r_mu")
                        _metric("Mass metric", "r_mass")
                        with v3.VAlert(v_show=("have_result",), type="warning", density="compact",
                                       variant="tonal", classes="mt-2"):
                            v3.VCardText("{{ decision }} — not certified until coupon allowables, ACP "
                                         "cross-validation, and burst correlation are supplied.",
                                         classes="pa-0 text-caption")
                # Export
                with v3.VExpansionPanel(title="Verification handoff · export"):
                    with v3.VExpansionPanelText():
                        v3.VBtn("Export Abaqus / ACP deck", click=ctrl.export_deck, variant="tonal",
                                block=True, size="small", classes="mb-2")
                        v3.VBtn("Export HTML report", click=ctrl.export_report, variant="tonal",
                                block=True, size="small")
                        v3.VCardText("{{ export_status }}", v_show=("export_status.length > 0",),
                                     classes="pa-0 pt-2 text-caption text-medium-emphasis")

    with layout.content:
        with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
            with v3.VCard(classes="ma-2 flex-grow-1 d-flex flex-column", style="height: calc(100vh - 100px);"):
                plotter_ui(plotter, mode="server", style="flex: 1 1 auto;")
            v3.VBanner("{{ status }}", density="compact", classes="ma-2", icon="mdi-information-outline")


def main():
    server.start()


if __name__ == "__main__":
    main()
