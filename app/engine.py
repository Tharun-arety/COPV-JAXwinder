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
from dataclasses import dataclass, field
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
    # Engine handles for downstream phases (not serialized; in-process only).
    geom: Any = None
    material: Any = None
    state: Any = None                       # FEA state dict
    winding_result: Any = None              # hostified optimizer result (full_optimize only)
    layout: Any = None                      # winding process layout (full_optimize only)


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
    disp = np.asarray(res["displacement"], dtype=np.float64).reshape(-1, 3)

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
        disp_max=float(np.max(np.linalg.norm(disp, axis=1))),
        gate=release_gate(fi_max, mu, friction.mu_max),
        geom=geom,
        material=material,
        state=state,
        winding_result=result,
        layout=layout,
    )
