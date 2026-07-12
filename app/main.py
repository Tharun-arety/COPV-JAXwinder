"""Trame web application — the tank configurator GUI.

Left drawer: the requirement spec (volume, pressure, envelope) plus the analysis
mode. Main area: an interactive 3D viewport of the vessel coloured by Hashin
failure index. A Run button drives the engine off the event loop so the UI stays
responsive while JAX solves.

Run it:
    pip install -e .[app]
    python -m app.main            # opens http://localhost:8080

The heavy numeric work (mesh build, solve, optimize) runs in a thread-pool executor;
all VTK/PyVista mutation happens back on the main coroutine so the render pipeline is
only ever touched from one thread.
"""

from __future__ import annotations

import warnings

warnings.warn("app.main (trame configurator) is deprecated; use `python -m app.server` "
              "(the workflow web app at http://localhost:8081).", DeprecationWarning, stacklevel=2)

import asyncio
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import signal as _signal_mod
import threading as _threading

# gmsh installs a SIGINT handler via signal.signal() on init, which raises off the
# main thread; the solve runs in a worker thread. Make signal.signal() a no-op off
# the main thread (the Ctrl+C handler is irrelevant in a web server).
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
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "The configurator GUI needs the app extras. Install them with:\n"
        "    pip install -e .[app]\n"
        f"(missing: {exc.name})"
    ) from exc

from app.engine import DesignResult, fast_screen, full_optimize
from app.sizing import TankRequirement, geometry_from_requirement
from copv_opt.config import MaterialConfig

# ---------------------------------------------------------------------------
# Server + plotter
# ---------------------------------------------------------------------------
server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

pv.OFF_SCREEN = True
plotter = pv.Plotter(off_screen=True)
plotter.background_color = "white"
plotter.add_text("Set a requirement and press Run", font_size=11, name="hint")

_MATERIAL = MaterialConfig()

# default requirement
state.update(
    {
        "volume_l": 9.0,
        "pressure_bar": 300.0,
        "radius_mm": 100.0,
        "thickness_mm": 8.0,
        "opening_mm": 10.0,
        "dome_ratio": 0.7,
        "mode": "Fast screen",
        "angle_deg": 42.0,
        "band_mm": 8.0,
        "running": False,
        "status": "Idle — set a requirement and press Run.",
        "have_result": False,
        # result fields
        "r_fi_max": "—",
        "r_burst": "—",
        "r_mass": "—",
        "r_mu": "—",
        "r_disp": "—",
        "r_cyl_len": "—",
        "r_volume": "—",
        "r_decision": "—",
        "r_blockers": [],
    }
)


def _color_mesh(result: DesignResult) -> None:
    """Build a PolyData from the result and (re)draw it. Main thread only."""
    elems = result.elems.astype(np.int64)
    faces = np.hstack([np.full((len(elems), 1), 3, dtype=np.int64), elems]).ravel()
    mesh = pv.PolyData(result.nodes, faces)
    mesh.cell_data["Failure index"] = result.failure_index

    plotter.clear()
    clim = (0.0, max(1.0, float(result.fi_max)))
    plotter.add_mesh(
        mesh,
        scalars="Failure index",
        clim=clim,
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={"title": "Hashin FI", "fmt": "%.2f"},
    )
    plotter.view_isometric()
    plotter.reset_camera()
    ctrl.view_update()


def _apply_result(result: DesignResult, sizing) -> None:
    with state:
        state.r_fi_max = f"{result.fi_max:.3f}"
        state.r_burst = f"{result.burst_factor:.3f}"
        state.r_mass = f"{result.mass_metric:.3g}"
        state.r_mu = "—" if result.mu_max_required is None else f"{result.mu_max_required:.3f} / {result.mu_allowable:.2f}"
        state.r_disp = f"{result.disp_max:.3f} mm"
        state.r_cyl_len = f"{sizing.cylinder_length_mm:.1f} mm"
        state.r_volume = f"{sizing.achieved_volume_litres:.2f} L"
        state.r_decision = result.gate["decision"].upper()
        state.r_blockers = list(result.gate["blockers"])
        state.have_result = True


def _compute_blocking(spec: dict, mode: str):
    """Runs in a worker thread. Returns (result, sizing) or raises."""
    req = TankRequirement(
        internal_volume_litres=spec["volume_l"],
        design_pressure_bar=spec["pressure_bar"],
        envelope_outer_radius_mm=spec["radius_mm"],
        wall_thickness_mm=spec["thickness_mm"],
        opening_radius_mm=spec["opening_mm"],
        dome_height_ratio=spec["dome_ratio"],
    )
    geom, sizing = geometry_from_requirement(req)
    if mode == "Full optimization":
        result = full_optimize(geom, _MATERIAL)
    else:
        result = fast_screen(geom, _MATERIAL, spec["angle_deg"], spec["band_mm"])
    return result, sizing


async def _run_pipeline() -> None:
    mode = state.mode
    spec = {
        "volume_l": float(state.volume_l),
        "pressure_bar": float(state.pressure_bar),
        "radius_mm": float(state.radius_mm),
        "thickness_mm": float(state.thickness_mm),
        "opening_mm": float(state.opening_mm),
        "dome_ratio": float(state.dome_ratio),
        "angle_deg": float(state.angle_deg),
        "band_mm": float(state.band_mm),
    }
    with state:
        state.running = True
        state.status = (
            "Meshing + optimizing (this takes a few minutes)…"
            if mode == "Full optimization"
            else "Meshing + screening…"
        )

    loop = asyncio.get_event_loop()
    try:
        result, sizing = await loop.run_in_executor(None, _compute_blocking, spec, mode)
    except Exception as exc:  # surface a clean message to the UI
        traceback.print_exc()
        with state:
            state.running = False
            state.status = f"Error: {exc}"
        return

    _color_mesh(result)            # VTK mutation on the main coroutine
    _apply_result(result, sizing)
    with state:
        state.running = False
        state.status = (
            f"Done — {result.mode.replace('_', ' ')} · FI_max {result.fi_max:.3f} · "
            f"burst ×{result.burst_factor:.2f} · gate {result.gate['decision']}"
        )


@ctrl.set("run")
def run() -> None:
    if state.running:
        return
    asynchronous.create_task(_run_pipeline())


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _metric(label: str, value_key: str):
    with v3.VRow(classes="ma-0 align-center"):
        with v3.VCol(cols=7, classes="pa-1 text-caption text-medium-emphasis"):
            v3.VLabel(label)
        with v3.VCol(cols=5, classes="pa-1 text-right font-weight-bold"):
            html_text = v3.VLabel(f"{{{{ {value_key} }}}}")
    return html_text


with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("COPV Tank Configurator")

    with layout.toolbar:
        v3.VSpacer()
        v3.VProgressCircular(indeterminate=("running",), v_show=("running",), size=22, width=3, classes="mr-3")
        v3.VChip("{{ r_decision }}", v_show=("have_result",), color="warning", size="small", classes="mr-2")

    with layout.drawer as drawer:
        drawer.width = 340
        with v3.VContainer(classes="pa-3"):
            v3.VCardSubtitle("Requirement", classes="px-0 text-overline")
            v3.VTextField(v_model=("volume_l",), label="Internal volume [L]", type="number", density="compact", variant="outlined", classes="mb-2")
            v3.VTextField(v_model=("pressure_bar",), label="Design pressure [bar]", type="number", density="compact", variant="outlined", classes="mb-2")
            v3.VTextField(v_model=("radius_mm",), label="Envelope outer radius [mm]", type="number", density="compact", variant="outlined", classes="mb-2")

            v3.VCardSubtitle("Shell", classes="px-0 text-overline")
            v3.VTextField(v_model=("thickness_mm",), label="Wall / base thickness [mm]", type="number", density="compact", variant="outlined", classes="mb-2")
            v3.VTextField(v_model=("opening_mm",), label="Boss opening radius [mm]", type="number", density="compact", variant="outlined", classes="mb-2")
            v3.VTextField(v_model=("dome_ratio",), label="Dome height ratio", type="number", density="compact", variant="outlined", classes="mb-2")

            v3.VCardSubtitle("Analysis", classes="px-0 text-overline")
            v3.VSelect(
                v_model=("mode",),
                items=("['Fast screen', 'Full optimization']",),
                label="Mode",
                density="compact",
                variant="outlined",
                classes="mb-2",
            )
            v3.VTextField(
                v_model=("angle_deg",),
                label="Winding angle [deg]",
                type="number",
                density="compact",
                variant="outlined",
                classes="mb-2",
                v_show=("mode === 'Fast screen'",),
            )
            v3.VTextField(
                v_model=("band_mm",),
                label="Band thickness [mm]",
                type="number",
                density="compact",
                variant="outlined",
                classes="mb-2",
                v_show=("mode === 'Fast screen'",),
            )

            v3.VBtn(
                "Run",
                click=ctrl.run,
                color="primary",
                block=True,
                loading=("running",),
                disabled=("running",),
                classes="mt-1 mb-3",
            )

            v3.VDivider(classes="mb-2")
            v3.VCardSubtitle("Result", classes="px-0 text-overline")
            _metric("Cylinder length", "r_cyl_len")
            _metric("Achieved volume", "r_volume")
            _metric("FI max", "r_fi_max")
            _metric("Burst factor", "r_burst")
            _metric("Max displacement", "r_disp")
            _metric("Friction μ req / allow", "r_mu")
            _metric("Mass metric", "r_mass")

            with v3.VAlert(v_show=("have_result",), type="warning", density="compact", variant="tonal", classes="mt-2"):
                html_div = v3.VCardText("{{ r_decision }} — release blocked until production data is supplied", classes="pa-0 text-caption")

    with layout.content:
        with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
            with v3.VCard(classes="ma-2 flex-grow-1 d-flex flex-column", style="height: calc(100vh - 100px);"):
                plotter_ui(plotter, mode="server", style="flex: 1 1 auto;")
            v3.VBanner("{{ status }}", density="compact", classes="ma-2", icon="mdi-information-outline")


def main() -> None:
    server.start()


if __name__ == "__main__":
    main()
