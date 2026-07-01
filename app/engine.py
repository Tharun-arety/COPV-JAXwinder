"""Engine layer: build the FEA state once per geometry, then screen or optimize.

Two analysis paths share one mesh/state cache:

* ``fast_screen``    — a single constant-angle forward solve (seconds). Interactive.
* ``full_optimize``  — the L-BFGS winding optimizer (minutes). The real design point.

Both return a :class:`DesignResult` with a per-element failure-index field for 3D
colouring, a burst factor, a mass figure, and a release gate. The gate always holds
``do_not_release`` by design — see ``release_gate``.

Burst factor
------------
Hashin indices are quadratic in stress and stress is linear in internal pressure,
so the failure index scales as p^2. The pressure multiple that drives FI to 1 is
therefore ``1 / sqrt(FI_max)`` — one formula for both paths, no extra solves.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from copv_opt.config import (
    FailureConfig,
    FrictionConfig,
    GeometryConfig,
    MaterialConfig,
    WindingConfig,
    WindingOptimizationConfig,
)
from copv_opt.geometry import ensure_copv_mesh
from copv_opt.optimize import run_winding_optimization, winding_forward_angle
from copv_opt.physics import (
    build_copv_fem_state,
    evaluate_hashin_failure,
    make_solve_compliance,
    rotate_stiffness_field,
)


# ---------------------------------------------------------------------------
# Result + gate
# ---------------------------------------------------------------------------
@dataclass
class DesignResult:
    mode: str                       # "fast_screen" | "full_optimize"
    nodes: np.ndarray               # (M, 3)
    elems: np.ndarray               # (N, 3) triangle connectivity
    failure_index: np.ndarray       # (N,) per-element Hashin index
    fi_max: float
    burst_factor: float
    mass_metric: float
    mass_delta_percent: float | None        # vs base laminate, when known
    mu_max_required: float | None           # friction demand, full optimize only
    mu_allowable: float
    angle_deg: float | None                 # constant angle, fast screen only
    disp_max: float
    gate: dict[str, Any] = field(default_factory=dict)
    # Full post-processing fields a stress engineer reviews (per-element, host numpy).
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    disp_node: Any = None                   # (M,) per-node displacement magnitude [mm]
    margins: dict[str, Any] = field(default_factory=dict)
    # Engine handles for downstream phases (not serialized; in-process only).
    geom: Any = None
    material: Any = None
    state: Any = None                       # FEA state dict
    winding_result: Any = None              # hostified optimizer result (full_optimize only)
    layout: Any = None                      # winding process layout (full_optimize only)


def _reserve_factor(fi: np.ndarray) -> np.ndarray:
    """RF = 1/sqrt(FI): the factor the load can be scaled by before first-ply failure."""
    return 1.0 / np.sqrt(np.maximum(np.asarray(fi, dtype=np.float64), 1e-12))


def assemble_fields(
    failure_like: dict[str, Any],
    thickness: Any,
    winding_angle_deg: np.ndarray,
    displacement: Any,
    nnodes: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Build the reviewer-facing result fields + margins from a solved state.

    ``failure_like`` is either the evaluate_hashin_failure output (fast screen) or the
    optimizer result (full optimize) — both carry the same per-mode keys."""
    fi = np.asarray(failure_like["failure_index"], dtype=np.float64)
    fields = {
        "Failure index (Hashin)": fi,
        "Reserve factor": _reserve_factor(fi),
        "Fibre tension": np.asarray(failure_like["fiber_tension"], dtype=np.float64),
        "Fibre compression": np.asarray(failure_like["fiber_compression"], dtype=np.float64),
        "Matrix tension": np.asarray(failure_like["matrix_tension"], dtype=np.float64),
        "Matrix compression": np.asarray(failure_like["matrix_compression"], dtype=np.float64),
        "Laminate thickness [mm]": np.asarray(thickness, dtype=np.float64),
        "Winding angle [deg]": np.asarray(winding_angle_deg, dtype=np.float64),
    }
    disp = np.asarray(displacement, dtype=np.float64).reshape(nnodes, 3)
    dmag = np.linalg.norm(disp, axis=1)
    fi_max = float(np.max(fi))
    crit = int(np.argmax(fi))
    mode_names = ["Fibre tension", "Fibre compression", "Matrix tension", "Matrix compression"]
    mode_vals = [float(fields[m][crit]) for m in mode_names]
    margins = {
        "fi_max": fi_max,
        "min_reserve_factor": float(1.0 / np.sqrt(max(fi_max, 1e-12))),
        "critical_element": crit,
        "critical_mode": mode_names[int(np.argmax(mode_vals))],
        "max_deformation_mm": float(np.max(dmag)),
    }
    return fields, dmag, margins


def release_gate(fi_max: float, mu_max_required: float | None, mu_allowable: float) -> dict[str, Any]:
    """The honest gate. Structural/friction screens can pass; release cannot — the
    qualification, machine, cure and inspection data needed to clear it are not in
    a screening tool. ``do_not_release`` is the correct Phase-0 output."""
    hashin_ok = fi_max <= 1.0 + 1e-6
    friction_ok = mu_max_required is None or mu_max_required <= mu_allowable + 1e-6
    blockers: list[str] = []
    if not hashin_ok:
        blockers.append(f"Structural screen fails: FI_max = {fi_max:.3f} > 1.0")
    if not friction_ok:
        blockers.append(
            f"Winding non-geodesic demand mu = {mu_max_required:.3f} exceeds allowable {mu_allowable:.3f}"
        )
    blockers.extend(
        [
            "No coupon-derived allowables (using literature Hashin values)",
            "No machine kinematic / collision verification",
            "No cure, residual-stress or autofrettage coupling",
            "No inspection thresholds or qualification dataset",
        ]
    )
    return {
        "hashin_ok": hashin_ok,
        "friction_ok": friction_ok,
        "release_ready": False,        # always — by design
        "decision": "do_not_release",
        "blockers": blockers,
    }


def _burst_factor(fi_max: float) -> float:
    return float(1.0 / np.sqrt(max(fi_max, 1e-12)))


def hostify(tree: Any) -> Any:
    """Recursively pull JAX arrays back to host NumPy. Mirrors the verification
    scripts' hostify_tree so downstream visualize/course helpers get plain arrays."""
    if isinstance(tree, dict):
        return {k: hostify(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(hostify(v) for v in tree)
    if hasattr(tree, "__array__") and not isinstance(tree, np.ndarray):
        return np.asarray(tree)
    return tree


# ---------------------------------------------------------------------------
# Mesh / state cache (keyed on the geometry that defines the mesh)
# ---------------------------------------------------------------------------
_STATE_CACHE: dict[str, dict[str, Any]] = {}
_WORK_ROOT = Path(tempfile.gettempdir()) / "copv_configurator_meshes"


def _geometry_key(geom: GeometryConfig) -> str:
    fields = (
        geom.outer_radius,
        geom.cylinder_length,
        geom.thickness,
        geom.opening_radius,
        geom.dome_height_ratio,
        geom.mesh_hmin,
        geom.mesh_hmax,
        geom.boss_hmin,
        geom.boss_refine_radius,
    )
    digest = hashlib.sha1(repr(fields).encode("utf-8")).hexdigest()[:16]
    return digest


def build_state(geom: GeometryConfig, material: MaterialConfig) -> dict[str, Any]:
    """Mesh the shell and assemble the FEA state, caching by geometry. Pressure is
    NOT part of the key — it scales the load vector, not the mesh — but the state is
    rebuilt per geometry, which already carries the current pressure via build."""
    key = _geometry_key(geom)
    cached = _STATE_CACHE.get(key)
    if cached is not None and cached["pressure"] == geom.pressure:
        return cached

    work = _WORK_ROOT / key
    work.mkdir(parents=True, exist_ok=True)
    mesh = ensure_copv_mesh(
        work / "copv_shell.step",
        work / "copv_shell.msh",
        geom,
        remesh=cached is None,       # reuse mesh files across pressure-only changes
        rebuild_step=cached is None,
    )
    state = build_copv_fem_state(mesh.nodes, mesh.elems, material, geom)
    solve = make_solve_compliance(state)
    bundle = {
        "pressure": geom.pressure,
        "nodes": np.asarray(mesh.nodes, dtype=np.float64),
        "elems": np.asarray(mesh.elems, dtype=np.int64),
        "state": state,
        "solve": solve,
        "material": material,
        "geom": geom,
    }
    _STATE_CACHE[key] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Analysis paths
# ---------------------------------------------------------------------------
def fast_screen(
    geom: GeometryConfig,
    material: MaterialConfig,
    angle_deg: float,
    band_thickness: float,
    friction_cfg: FrictionConfig | None = None,
    failure_cfg: FailureConfig | None = None,
) -> DesignResult:
    """One constant-angle forward solve + Hashin evaluation. Seconds, interactive.

    ``failure_cfg`` carries the allowables — pass a calibrated FailureConfig (see
    app.calibration) to screen against coupon-derived strengths instead of the
    literature defaults."""
    friction = FrictionConfig() if friction_cfg is None else friction_cfg
    failure = FailureConfig(margin_of_safety=1.0) if failure_cfg is None else failure_cfg
    bundle = build_state(geom, material)
    state, solve = bundle["state"], bundle["solve"]

    res = winding_forward_angle(
        angle_deg, state, material, WindingConfig(band_thickness=band_thickness), geom, solve
    )

    # Re-form the smeared laminate stiffness exactly as winding_forward_angle did,
    # so the Hashin stress is consistent with the solved displacement.
    base = material.base_thickness
    c_base = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    c_rot = rotate_stiffness_field(state["c_mat"], res["fiber_dirs"], state["surface_normals"])
    total = res["thickness"]
    c_eff = (base * c_base + band_thickness * c_rot) / total[:, None, None, None, None]

    failure_metrics = evaluate_hashin_failure(state, res["displacement"], c_eff, res["fiber_dirs"], failure)
    fi = np.asarray(failure_metrics["failure_index"], dtype=np.float64)
    fi_max = float(np.max(fi))
    nnodes = len(bundle["nodes"])
    winding_angle = np.full(fi.shape, float(angle_deg))
    fields, dmag, margins = assemble_fields(failure_metrics, total, winding_angle, res["displacement"], nnodes)

    # element-level per-ply CLT (base 0 + constant-angle helical band, no hoop)
    from copv_opt.clt_fem import element_clt_fields
    ne = len(fi)
    fields.update(element_clt_fields(state, res["displacement"], winding_angle,
                                     np.ones(ne, bool), np.zeros(ne, bool), material, failure.allowables))

    return DesignResult(
        mode="fast_screen",
        nodes=bundle["nodes"],
        elems=bundle["elems"],
        failure_index=fi,
        fi_max=fi_max,
        burst_factor=_burst_factor(fi_max),
        mass_metric=float(np.asarray(res["mass_metric"])),
        mass_delta_percent=None,
        mu_max_required=None,
        mu_allowable=friction.mu_max,
        angle_deg=float(angle_deg),
        disp_max=float(np.max(dmag)),
        gate=release_gate(fi_max, None, friction.mu_max),
        fields=fields,
        disp_node=dmag,
        margins=margins,
        geom=geom,
        material=material,
        state=state,
    )


def mesh_convergence(
    geom: GeometryConfig,
    material: MaterialConfig,
    angle_deg: float,
    band_thickness: float,
    hmax_list: list[float] = (36.0, 28.0, 20.0, 14.0),
) -> list[dict[str, Any]]:
    """Run the constant-angle screen at several mesh densities and report FI_max.

    A stress engineer needs to see the result is mesh-converged before trusting it.
    Returns one row per density; the trend should flatten as the mesh refines."""
    rows: list[dict[str, Any]] = []
    for hmax in hmax_list:
        g = replace(geom, mesh_hmax=float(hmax), mesh_hmin=min(geom.mesh_hmin, float(hmax)))
        r = fast_screen(g, material, angle_deg, band_thickness)
        rows.append({"mesh_hmax": float(hmax), "elements": int(len(r.elems)), "fi_max": r.fi_max,
                     "min_reserve_factor": r.margins["min_reserve_factor"]})
    return rows


def screen_profile(
    profile,
    geom: GeometryConfig,
    material: MaterialConfig,
    angle_deg: float,
    band_thickness: float,
    work_dir: Any = None,
    friction_cfg: FrictionConfig | None = None,
    failure_cfg: FailureConfig | None = None,
) -> DesignResult:
    """Screen an arbitrary axisymmetric mandrel given by a MeridianProfile.

    Meshes the revolved meridian, builds a general FEA state (validated against the
    parametric COPV — see app/validate_general.py), and runs the same constant-angle
    forward solve + Hashin screen. ``geom`` supplies pressure and a representative
    radius for the winding-angle field.

    KNOWN LIMITATION: the *physics* is validated to ~1.6% against the analytic COPV on
    an identical mesh, but this self-meshed path under-resolves the polar-opening stress
    concentration — the revolved BSpline rounds the boss more than the analytic cap, so
    absolute FI runs lower (non-conservative) than the parametric boss-refined path
    (~46% lower on the COPV case). Use fast_screen/full_optimize for absolute COPV
    screening; use this path for arbitrary-shape exploration and relative comparison
    until the boss meshing / revolve fidelity is improved."""
    import tempfile

    from app.general_state import build_general_fem_state
    from app.meridian_mesh import mesh_meridian

    friction = FrictionConfig() if friction_cfg is None else friction_cfg
    failure = FailureConfig(margin_of_safety=1.0) if failure_cfg is None else failure_cfg
    work = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="copv_meridian_"))

    nodes, elems = mesh_meridian(profile, work, hmin=geom.mesh_hmin, hmax=geom.mesh_hmax)
    state = build_general_fem_state(nodes, elems, material, profile, geom.pressure, geom.support_tol)
    solve = make_solve_compliance(state)

    res = winding_forward_angle(
        angle_deg, state, material, WindingConfig(band_thickness=band_thickness), geom, solve
    )
    base = material.base_thickness
    c_base = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    c_rot = rotate_stiffness_field(state["c_mat"], res["fiber_dirs"], state["surface_normals"])
    total = res["thickness"]
    c_eff = (base * c_base + band_thickness * c_rot) / total[:, None, None, None, None]
    failure_metrics = evaluate_hashin_failure(state, res["displacement"], c_eff, res["fiber_dirs"], failure)
    fi = np.asarray(failure_metrics["failure_index"], dtype=np.float64)
    fi_max = float(np.max(fi))
    disp = np.asarray(res["displacement"], dtype=np.float64).reshape(-1, 3)

    return DesignResult(
        mode="profile_screen",
        nodes=np.asarray(nodes, dtype=np.float64),
        elems=np.asarray(elems, dtype=np.int64),
        failure_index=fi,
        fi_max=fi_max,
        burst_factor=_burst_factor(fi_max),
        mass_metric=float(np.asarray(res["mass_metric"])),
        mass_delta_percent=None,
        mu_max_required=None,
        mu_allowable=friction.mu_max,
        angle_deg=float(angle_deg),
        disp_max=float(np.max(np.linalg.norm(disp, axis=1))),
        gate=release_gate(fi_max, None, friction.mu_max),
        geom=geom,
        material=material,
        state=state,
    )


def full_optimize(
    geom: GeometryConfig,
    material: MaterialConfig,
    winding_cfg: WindingOptimizationConfig | None = None,
    failure_cfg: FailureConfig | None = None,
    friction_cfg: FrictionConfig | None = None,
) -> DesignResult:
    """The real design point: L-BFGS over angle + pass-count controls. Minutes."""
    failure = FailureConfig(margin_of_safety=1.0, penalty_weight=4000.0) if failure_cfg is None else failure_cfg
    friction = FrictionConfig() if friction_cfg is None else friction_cfg
    cfg = winding_cfg or WindingOptimizationConfig(
        min_angle_deg=12.0,
        max_angle_deg=58.0,
        max_winding_thickness=18.0,
        winding_seed_angle_deg=42.0,
        winding_seed_thickness=7.0,
        max_helical_pass_count=44.0,
        max_hoop_pass_count=24.0,
        helical_seed_pass_count=14.0,
        hoop_seed_pass_count=2.0,
        lbfgs_maxiter=100,
        lbfgs_tol=1e-6,
        history_size=12,
    )

    bundle = build_state(geom, material)
    state, solve = bundle["state"], bundle["solve"]

    run = run_winding_optimization(
        state, material, cfg, geom, solve, failure_config=failure, friction_config=friction
    )
    result = hostify(run["result"])
    fi = np.asarray(result["failure_index"], dtype=np.float64)
    fi_max = float(np.asarray(result["fi_max"]))
    mu = float(np.asarray(result["mu_max_required"]))
    disp = np.asarray(result["displacement"], dtype=np.float64).reshape(-1, 3)

    # Build the winding process layout now so course planning / NC export downstream
    # have it without re-running the optimizer. Imported lazily to avoid the heavy
    # visualize import for callers that only screen.
    from copv_opt.visualize import build_winding_process_layout_data

    layout = build_winding_process_layout_data(result, geom, family_count=8, sample_count=320)

    nnodes = len(bundle["nodes"])
    winding_angle = np.degrees(np.asarray(result["winding_angle_field"], dtype=np.float64))
    fields, dmag, margins = assemble_fields(result, result["thickness"], winding_angle, result["displacement"], nnodes)

    # element-level per-ply CLT: base 0 + helical +/-alpha + hoop, per element
    from copv_opt.clt_fem import element_clt_fields
    has_helical = np.asarray(result["helical_thickness_field"], dtype=np.float64) > 1e-6
    has_hoop = np.asarray(result["hoop_thickness_field"], dtype=np.float64) > 1e-6
    fields.update(element_clt_fields(state, result["displacement"], winding_angle,
                                     has_helical, has_hoop, material, failure.allowables))

    return DesignResult(
        mode="full_optimize",
        nodes=bundle["nodes"],
        elems=bundle["elems"],
        failure_index=fi,
        fi_max=fi_max,
        burst_factor=_burst_factor(fi_max),
        mass_metric=float(np.asarray(result["mass_metric"])),
        mass_delta_percent=None,
        mu_max_required=mu,
        mu_allowable=friction.mu_max,
        angle_deg=None,
        disp_max=float(np.max(dmag)),
        gate=release_gate(fi_max, mu, friction.mu_max),
        fields=fields,
        disp_node=dmag,
        margins=margins,
        geom=geom,
        material=material,
        state=state,
        winding_result=result,
        layout=layout,
    )
