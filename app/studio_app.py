"""COPV Studio Pro — a workflow-driven stress-analysis application.

An Ansys-Mechanical-style workflow over the real JAX engine. The user defines the
tank (by capacity or dimensions) and steps through an outline:

    Geometry -> Materials -> Mesh -> Analysis settings -> Solution -> Results

Each step has a Details panel; the 3D graphics window updates as you progress
(shell -> mesh -> contour). Every solve is the real FEA + winding optimiser + the
element-level CLT fields. Stays do_not_release until coupon/ACP/burst data exists.

Run:
    pip install -e .[app]
    python -m app.studio_app          # http://localhost:8080
"""

from __future__ import annotations

import warnings

warnings.warn("app.studio_app (trame GUI) is deprecated; use `python -m app.server` "
              "(the workflow web app at http://localhost:8081).", DeprecationWarning, stacklevel=2)

import asyncio
import os
import signal as _signal_mod
import sys
import threading as _threading
import traceback
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# gmsh installs a SIGINT handler off the main thread; no-op it in worker threads.
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
    from trame.ui.vuetify3 import VAppLayout
    from trame.widgets import html, vuetify3 as v3
except ImportError as exc:  # pragma: no cover
    raise SystemExit("COPV Studio Pro needs:  pip install -e .[app]\n"
                     f"(missing: {exc.name})") from exc

from app.engine import build_state, fast_screen, full_optimize
from app.sizing import TankRequirement, geometry_from_requirement
from copv_opt.config import FailureConfig, GeometryConfig, MaterialAllowables, MaterialConfig

server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

pv.OFF_SCREEN = True
plotter = pv.Plotter(off_screen=True)
plotter.background_color = "#222831"
plotter.add_text("Define the tank, then step through the workflow", font_size=11, name="hint", color="#c9d2db")

STUDIO: dict = {"geom": None, "bundle": None, "result": None, "poly": None}
_DENSITY = {"Coarse": 36.0, "Medium": 28.0, "Fine": 18.0}
_MATERIAL = MaterialConfig()

state.update({
    "step": "geometry",
    # workflow status icons
    "s_geometry": "mdi-circle-outline", "s_materials": "mdi-circle-outline",
    "s_mesh": "mdi-circle-outline", "s_setup": "mdi-circle-outline",
    "s_solution": "mdi-circle-outline", "s_results": "mdi-circle-outline",
    # geometry inputs
    "define_by": "Capacity", "capacity_l": 9.0, "radius": 100.0, "length": 220.0,
    "thickness": 8.0, "opening": 10.0, "dome_ratio": 0.7, "tank_type": "Type 3 (Al liner)",
    "derived": "",
    # materials
    "xt": 2200.0, "xc": 1400.0, "yt": 70.0, "yc": 220.0, "s": 120.0,
    # mesh
    "density": "Medium", "mesh_info": "not generated",
    # setup
    "pressure": 6.85, "mode": "Constant-angle screen", "angle_deg": 42.0, "band_mm": 8.0,
    # solution / results
    "field_options": ["Failure index (Hashin)"], "field": "Failure index (Hashin)",
    "r_fi": "—", "r_rf": "—", "r_mode": "—", "r_def": "—", "r_burst": "—", "r_clt": "—",
    "decision": "—", "have_result": False,
    "running": False, "status": "Ready. Start at Geometry.",
})


# ---------------------------------------------------------------------------
# geometry / drawing
# ---------------------------------------------------------------------------
def _build_geom() -> GeometryConfig:
    hmax = _DENSITY.get(state.density, 28.0)
    common = dict(thickness=float(state.thickness), opening_radius=float(state.opening),
                  dome_height_ratio=float(state.dome_ratio), pressure=float(state.pressure))
    if state.define_by == "Capacity":
        req = TankRequirement(internal_volume_litres=float(state.capacity_l),
                              design_pressure_bar=float(state.pressure) * 10.0,
                              envelope_outer_radius_mm=float(state.radius),
                              wall_thickness_mm=float(state.thickness),
                              opening_radius_mm=float(state.opening),
                              dome_height_ratio=float(state.dome_ratio))
        geom, rep = geometry_from_requirement(req)
        geom = GeometryConfig(outer_radius=geom.outer_radius, cylinder_length=geom.cylinder_length, **common)
        derived = (f"cyl length {rep.cylinder_length_mm:.0f} mm · achieved {rep.achieved_volume_litres:.2f} L "
                   f"· L/D {rep.slenderness_l_over_d:.2f}")
    else:
        geom = GeometryConfig(outer_radius=float(state.radius), cylinder_length=float(state.length), **common)
        derived = f"outer Ø {2*geom.outer_radius:.0f} mm · cyl length {geom.cylinder_length:.0f} mm"
    geom.mesh_hmax = hmax
    geom.mesh_hmin = min(10.0, hmax)
    return geom, derived


def _failure_cfg() -> FailureConfig:
    return FailureConfig(allowables=MaterialAllowables(xt=float(state.xt), xc=float(state.xc),
                         yt=float(state.yt), yc=float(state.yc), s=float(state.s)), margin_of_safety=1.0)


def _poly():
    b = STUDIO["bundle"]
    elems = np.asarray(b["elems"], dtype=np.int64)
    faces = np.hstack([np.full((len(elems), 1), 3, dtype=np.int64), elems]).ravel()
    return pv.PolyData(np.asarray(b["nodes"], dtype=np.float64), faces)


def _draw_shell(edges: bool):
    plotter.clear()
    plotter.add_mesh(_poly(), color="#8fa3b0", show_edges=edges, edge_color="#39424c", line_width=1)
    plotter.view_isometric(); plotter.reset_camera(); ctrl.view_update()


def _draw_field(field: str):
    poly = STUDIO["poly"]
    if poly is None or field not in poly.cell_data:
        return
    vals = np.asarray(poly.cell_data[field]); lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        hi = lo + 1.0
    plotter.clear()
    plotter.add_mesh(poly, scalars=field, clim=(lo, hi), cmap="turbo", show_edges=False,
                     scalar_bar_args={"title": field, "color": "white", "title_font_size": 14,
                                      "label_font_size": 11, "fmt": "%.3g", "n_labels": 6, "vertical": True,
                                      "position_x": 0.02, "position_y": 0.22, "width": 0.05, "height": 0.55})
    plotter.view_isometric(); plotter.reset_camera(); ctrl.view_update()


# ---------------------------------------------------------------------------
# workflow steps (run heavy work off the event loop)
# ---------------------------------------------------------------------------
async def _do(status_msg, fn, done_key=None, step_next=None):
    with state:
        state.running = True; state.status = status_msg
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, fn)
    except Exception as exc:
        traceback.print_exc()
        with state:
            state.running = False; state.status = f"Error: {exc}"
        return None
    with state:
        state.running = False
        if done_key:
            setattr(state, done_key, "mdi-check-circle")
        if step_next:
            state.step = step_next
    return result


def _gen_geometry_blocking():
    geom, derived = _build_geom()
    bundle = build_state(geom, _MATERIAL)
    STUDIO["geom"] = geom; STUDIO["bundle"] = bundle
    return derived


@ctrl.set("gen_geometry")
def gen_geometry():
    if state.running:
        return
    async def run():
        derived = await _do("Building geometry (OpenCASCADE + gmsh)…", _gen_geometry_blocking,
                            done_key="s_geometry", step_next="materials")
        if derived is not None:
            _draw_shell(edges=False)
            with state:
                state.derived = derived
                state.mesh_info = f"{len(STUDIO['bundle']['elems'])} elements · {len(STUDIO['bundle']['nodes'])} nodes"
                state.status = "Geometry generated. Set materials, then mesh."
    asynchronous.create_task(run())


@ctrl.set("gen_mesh")
def gen_mesh():
    if state.running or STUDIO["bundle"] is None:
        state.status = "Generate geometry first."
        return
    async def run():
        derived = await _do("Meshing…", _gen_geometry_blocking, done_key="s_mesh", step_next="setup")
        if derived is not None:
            _draw_shell(edges=True)
            with state:
                state.mesh_info = f"{len(STUDIO['bundle']['elems'])} elements · {len(STUDIO['bundle']['nodes'])} nodes"
                state.status = f"Mesh generated ({state.mesh_info}). Configure the analysis."
    asynchronous.create_task(run())


def _solve_blocking():
    geom = _build_geom()[0]
    failure = _failure_cfg()
    if state.mode == "Optimize winding":
        r = full_optimize(geom, _MATERIAL, failure_cfg=failure)
    else:
        r = fast_screen(geom, _MATERIAL, float(state.angle_deg), float(state.band_mm), failure_cfg=failure)
    STUDIO["result"] = r
    elems = np.asarray(r.elems, dtype=np.int64)
    faces = np.hstack([np.full((len(elems), 1), 3, dtype=np.int64), elems]).ravel()
    poly = pv.PolyData(np.asarray(r.nodes, dtype=np.float64), faces)
    for name, arr in r.fields.items():
        poly.cell_data[name] = np.asarray(arr, dtype=np.float64)
    STUDIO["poly"] = poly
    return r


@ctrl.set("solve")
def solve():
    if state.running or STUDIO["bundle"] is None:
        state.status = "Generate the mesh first."
        return
    async def run():
        r = await _do("Solving (real FEA + winding + element CLT)…", _solve_blocking,
                      done_key="s_solution", step_next="results")
        if r is None:
            return
        _draw_field("Failure index (Hashin)")
        with state:
            m = r.margins
            state.field_options = list(r.fields.keys())
            state.field = "Failure index (Hashin)"
            state.r_fi = f"{r.fi_max:.3f}"; state.r_rf = f"{m['min_reserve_factor']:.2f}"
            state.r_mode = m["critical_mode"]; state.r_def = f"{m['max_deformation_mm']:.3f} mm"
            state.r_burst = f"{r.burst_factor:.2f}×"
            state.r_clt = f"{float(np.max(r.fields['CLT critical-ply FI'])):.3f}" if "CLT critical-ply FI" in r.fields else "—"
            state.decision = r.gate["decision"].upper(); state.have_result = True
            state.s_results = "mdi-check-circle"
            state.status = f"Solved · FI_max {r.fi_max:.3f} · min RF {m['min_reserve_factor']:.2f} · gate {r.gate['decision']}"
    asynchronous.create_task(run())


@state.change("field")
def _on_field(field, **_):
    if STUDIO["poly"] is not None and state.have_result:
        _draw_field(field)


@state.change("xt", "xc", "yt", "yc", "s")
def _mark_materials(**_):
    state.s_materials = "mdi-check-circle"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CSS = """
.v-application{font-family:'Segoe UI',system-ui,sans-serif}
.cae-bar{border-bottom:1px solid #3a414b !important}
.wf-drawer{border-right:1px solid #3a414b !important}
.wf-step{cursor:pointer;border-radius:6px;margin:1px 6px;padding:6px 10px;font-size:13px;display:flex;align-items:center;gap:9px}
.wf-step:hover{background:#2f3640}
.wf-step.on{background:#34506b;color:#fff}
.wf-grp{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#8b95a1;padding:10px 12px 4px}
.details{border-top:1px solid #3a414b;padding:10px 12px}
.details h4{font-size:12px;color:#9aa4b1;text-transform:uppercase;letter-spacing:.04em;margin:0 0 8px}
.cae-status{position:absolute;left:0;right:0;bottom:0;height:26px;display:flex;align-items:center;gap:16px;
  padding:0 14px;font-size:11.5px;color:#9aa4b1;background:#23272e;border-top:1px solid #3a414b;z-index:3}
.cae-status b{color:#dfe4ea}.cae-status .gate{margin-left:auto;color:#e0a23a;font-weight:600}
.cae-view{position:absolute;top:0;left:0;right:0;bottom:26px}
"""

STEPS = [("geometry", "Geometry", "s_geometry"), ("materials", "Materials", "s_materials"),
         ("mesh", "Mesh", "s_mesh"), ("setup", "Analysis settings", "s_setup"),
         ("solution", "Solution", "s_solution"), ("results", "Results", "s_results")]


def _num(model, label):
    v3.VTextField(v_model=(model,), label=label, type="number", density="compact",
                  variant="outlined", hide_details=True, classes="mb-2")


def _metric(label, key):
    with v3.VRow(classes="ma-0"):
        with v3.VCol(cols=7, classes="pa-1 text-caption text-medium-emphasis"):
            v3.VLabel(label)
        with v3.VCol(cols=5, classes="pa-1 text-right font-weight-bold"):
            v3.VLabel(f"{{{{ {key} }}}}")


with VAppLayout(server) as layout:
    html.Style(CSS)
    with v3.VAppBar(theme="dark", density="compact", flat=True, color="#23272e", elevation=0, classes="cae-bar"):
        v3.VIcon("mdi-hexagon-multiple", color="teal", classes="ml-4 mr-2")
        html.Span("COPV Studio Pro", style="font-size:14px;font-weight:600;color:#eef2f6")
        html.Span("workflow", classes="text-caption text-medium-emphasis ml-3")
        v3.VSpacer()
        v3.VProgressCircular(indeterminate=("running",), v_show=("running",), size=20, width=2, color="teal", classes="mr-3")
        v3.VChip("{{ decision }}", v_show=("have_result",), color="warning", size="small", variant="flat", classes="mr-4")

    with v3.VNavigationDrawer(theme="dark", permanent=True, width=340, color="#262b33", classes="wf-drawer"):
        html.Div("Outline", classes="wf-grp")
        for key, label, icon in STEPS:
            with html.Div(classes=("`wf-step ${step==='%s' ? 'on' : ''}`" % key,), click=f"step='{key}'"):
                v3.VIcon((icon,), size="18", color="teal")
                html.Span(label)

        # ---- Details panels (one per step) ----
        with html.Div(classes="details", v_show=("step==='geometry'",)):
            html.H4("Details — Geometry")
            v3.VSelect(v_model=("define_by",), items=("['Capacity', 'Dimensions']",), label="Define tank by",
                       density="compact", variant="outlined", hide_details=True, classes="mb-2")
            _num("capacity_l", "Capacity [litres]")
            _num("radius", "Outer radius [mm]")
            with html.Div(v_show=("define_by==='Dimensions'",)):
                _num("length", "Cylinder length [mm]")
            _num("thickness", "Wall / liner thickness [mm]")
            _num("opening", "Boss opening radius [mm]")
            _num("dome_ratio", "Dome height ratio")
            v3.VSelect(v_model=("tank_type",), items=("['Type 3 (Al liner)', 'Type 4 (PE liner)']",),
                       label="Tank type", density="compact", variant="outlined", hide_details=True, classes="mb-2")
            v3.VBtn("Generate geometry", click=ctrl.gen_geometry, color="teal", block=True, loading=("running",))
            v3.VCardText("{{ derived }}", v_show=("derived.length>0",), classes="pa-0 pt-2 text-caption text-medium-emphasis")

        with html.Div(classes="details", v_show=("step==='materials'",)):
            html.H4("Details — Composite allowables [MPa]")
            v3.VCardText("Load coupon-derived values to calibrate.", classes="pa-0 pb-2 text-caption")
            _num("xt", "XT — fibre tension"); _num("xc", "XC — fibre compression")
            _num("yt", "YT — matrix tension"); _num("yc", "YC — matrix compression"); _num("s", "S — shear")

        with html.Div(classes="details", v_show=("step==='mesh'",)):
            html.H4("Details — Mesh")
            v3.VSelect(v_model=("density",), items=("['Coarse', 'Medium', 'Fine']",), label="Element size",
                       density="compact", variant="outlined", hide_details=True, classes="mb-2")
            v3.VBtn("Generate mesh", click=ctrl.gen_mesh, color="teal", block=True, loading=("running",))
            v3.VCardText("{{ mesh_info }}", classes="pa-0 pt-2 text-caption text-medium-emphasis")

        with html.Div(classes="details", v_show=("step==='setup'",)):
            html.H4("Details — Analysis settings")
            _num("pressure", "Design pressure [MPa]")
            v3.VSelect(v_model=("mode",), items=("['Optimize winding', 'Constant-angle screen']",), label="Winding",
                       density="compact", variant="outlined", hide_details=True, classes="mb-2")
            with html.Div(v_show=("mode==='Constant-angle screen'",)):
                _num("angle_deg", "Winding angle [deg]"); _num("band_mm", "Band thickness [mm]")

        with html.Div(classes="details", v_show=("step==='solution'",)):
            html.H4("Details — Solution")
            v3.VBtn("Solve", click=ctrl.solve, color="teal", block=True, loading=("running",), prepend_icon="mdi-play")
            v3.VCardText("Runs the real FEA, winding optimizer, and element-level CLT.", classes="pa-0 pt-2 text-caption")

        with html.Div(classes="details", v_show=("step==='results'",)):
            html.H4("Details — Results")
            v3.VSelect(v_model=("field",), items=("field_options",), label="Contour field",
                       density="compact", variant="outlined", hide_details=True, classes="mb-3")
            _metric("FI max (smeared)", "r_fi"); _metric("CLT critical-ply FI", "r_clt")
            _metric("Min reserve factor", "r_rf"); _metric("Critical mode", "r_mode")
            _metric("Max deformation", "r_def"); _metric("Burst factor", "r_burst")
            with v3.VAlert(v_show=("have_result",), type="warning", density="compact", variant="tonal", classes="mt-2"):
                v3.VCardText("{{ decision }} — not certified until coupon, ACP, and burst data are supplied.",
                             classes="pa-0 text-caption")

    with v3.VMain(theme="dark"):
        with html.Div(style="position:relative;height:calc(100vh - 48px);width:100%;"):
            with html.Div(classes="cae-view"):
                _view = plotter_ui(plotter, mode="client", style="width:100%;height:100%;")
                ctrl.view_update = _view.update
                ctrl.view_reset_camera = _view.reset_camera
            with html.Div(classes="cae-status"):
                html.Span("Model: "); html.Span("{{ mesh_info }}", style="color:#dfe4ea;font-weight:600")
                html.Span("{{ status }}", classes="ml-4")
                html.Span("{{ decision }}", classes="gate", v_show=("have_result",))


def main():
    server.start()


if __name__ == "__main__":
    main()
