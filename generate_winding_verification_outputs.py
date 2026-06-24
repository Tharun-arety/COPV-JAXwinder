from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax import config as jax_config

jax_config.update("jax_enable_x64", False)

import jax
import jax.numpy as jnp

from verification_common import (
    FAILURE_MODE_NAMES,
    PROJECT_ROOT,
    hostify_tree,
    plot_hashin_burst,
    plot_manufacturing_constraints,
    plot_objective_influence,
    repo_rel,
)

SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.abaqus_exporter import export_result_to_abaqus
from copv_opt.config import FailureConfig, FrictionConfig, GeometryConfig, MaterialConfig, WindingOptimizationConfig
from copv_opt.geometry import ensure_copv_mesh
from copv_opt.optimize import run_winding_optimization
from copv_opt.physics import (
    baseline_response,
    build_copv_fem_state,
    evaluate_hashin_failure,
    make_solve_compliance,
    rotate_stiffness_field,
)
from copv_opt.visualize import (
    build_winding_process_layout_data,
    plot_winding_process_paths,
    save_explicit_manufacturing_layout_screenshot,
    save_layout_json,
    show_copv_mesh,
    write_vtu,
)


LEGACY_WINDING_KEYS = {
    "active_patch_count",
    "coverage_excess",
    "max_patch_thickness",
    "overlap_penalty",
    "patch_added_thickness",
    "patch_alpha",
    "patch_coverage",
    "patch_fiber_dirs",
    "patch_l1",
    "patch_phi",
    "patch_s",
    "patch_thickness",
    "patch_weights",
    "repulsion_penalty",
}


def build_critical_summary(state: dict[str, Any], failure_metrics: dict[str, Any]) -> dict[str, Any]:
    failure_index = np.asarray(failure_metrics["failure_index"], dtype=np.float64)
    fiber_tension = np.asarray(failure_metrics["fiber_tension"], dtype=np.float64)
    fiber_compression = np.asarray(failure_metrics["fiber_compression"], dtype=np.float64)
    matrix_tension = np.asarray(failure_metrics["matrix_tension"], dtype=np.float64)
    matrix_compression = np.asarray(failure_metrics["matrix_compression"], dtype=np.float64)
    mode_stack = np.stack([fiber_tension, fiber_compression, matrix_tension, matrix_compression], axis=0)
    critical_idx = int(np.argmax(failure_index))
    dominant_mode_idx = int(np.argmax(mode_stack[:, critical_idx]))
    return {
        "critical_element_index": critical_idx,
        "critical_s": float(np.asarray(state["s_coords"])[critical_idx]),
        "critical_phi_deg": float(np.degrees(np.asarray(state["phi_coords"])[critical_idx])),
        "critical_failure_index": float(failure_index[critical_idx]),
        "dominant_mode": FAILURE_MODE_NAMES[dominant_mode_idx],
    }


def strip_legacy_winding_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_legacy_winding_fields(item)
            for key, item in value.items()
            if key not in LEGACY_WINDING_KEYS
        }
    if isinstance(value, list):
        return [strip_legacy_winding_fields(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the winding-first verification outputs for the staged COPV package.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for the winding-first case artifacts.",
    )
    parser.add_argument(
        "--remesh",
        action="store_true",
        help="Rebuild the STEP/Gmsh model instead of reusing outputs/copv_shell.msh.",
    )
    parser.add_argument(
        "--skip-pyvista",
        action="store_true",
        help="Skip the optional off-screen PyVista layout screenshot even if PyVista is installed.",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=5,
        help="Capture optimization snapshots every N LBFGS iterations for README assets. Use 0 to disable.",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=6.85,
        help="Internal pressure for the winding-first verification case.",
    )
    parser.add_argument(
        "--mass-weight",
        type=float,
        default=1.0,
        help="Mass term weight in the winding-only objective.",
    )
    parser.add_argument(
        "--max-winding-thickness",
        type=float,
        default=18.0,
        help="Upper bound on added winding thickness in the optimization model.",
    )
    parser.add_argument(
        "--max-helical-pass-count",
        type=float,
        default=44.0,
        help="Upper bound on the helical pass-count control.",
    )
    parser.add_argument(
        "--max-hoop-pass-count",
        type=float,
        default=24.0,
        help="Upper bound on the hoop pass-count control.",
    )
    parser.add_argument(
        "--winding-seed-thickness",
        type=float,
        default=7.0,
        help="Warm-start added thickness used for the initial winding seed.",
    )
    parser.add_argument(
        "--helical-seed-pass-count",
        type=float,
        default=14.0,
        help="Warm-start helical pass count used for the initial winding seed.",
    )
    parser.add_argument(
        "--hoop-seed-pass-count",
        type=float,
        default=2.0,
        help="Warm-start hoop pass count used for the initial winding seed.",
    )
    parser.add_argument(
        "--lbfgs-maxiter",
        type=int,
        default=100,
        help="Maximum LBFGS iterations per continuation stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs.resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    geom = GeometryConfig(pressure=float(args.pressure))
    material = MaterialConfig()
    failure_cfg = FailureConfig(margin_of_safety=1.0, penalty_weight=4000.0)
    friction_cfg = FrictionConfig()
    winding_cfg = WindingOptimizationConfig(
        mass_weight=float(args.mass_weight),
        min_angle_deg=12.0,
        max_angle_deg=58.0,
        max_winding_thickness=float(args.max_winding_thickness),
        winding_seed_angle_deg=42.0,
        winding_seed_thickness=float(args.winding_seed_thickness),
        max_helical_pass_count=float(args.max_helical_pass_count),
        max_hoop_pass_count=float(args.max_hoop_pass_count),
        helical_seed_pass_count=float(args.helical_seed_pass_count),
        hoop_seed_pass_count=float(args.hoop_seed_pass_count),
        hoop_transition_length=18.0,
        angle_smoothness_weight=0.015,
        pass_smoothness_weight=0.02,
        thickness_cap_penalty_weight=30.0,
        lbfgs_maxiter=int(args.lbfgs_maxiter),
        lbfgs_tol=1e-6,
        history_size=12,
    )

    step_path = outputs_dir / "copv_shell.step"
    msh_path = outputs_dir / "copv_shell.msh"
    mesh = ensure_copv_mesh(
        step_path,
        msh_path,
        geom,
        remesh=args.remesh,
        rebuild_step=args.remesh,
    )

    state = build_copv_fem_state(mesh.nodes, mesh.elems, material, geom)
    solve_compliance = make_solve_compliance(state)
    baseline = hostify_tree(baseline_response(state, material, solve_compliance))
    c_base = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    baseline_failure = hostify_tree(
        evaluate_hashin_failure(state, baseline["displacement"], c_base, baseline["fiber_dirs"], failure_cfg)
    )
    baseline_critical = build_critical_summary(state, baseline_failure)
    gc.collect()

    fig = show_copv_mesh(
        mesh.nodes,
        np.asarray(state["outer_faces"]),
        geom,
        f"Winding-first case mesh: {len(mesh.nodes)} nodes / {len(mesh.elems)} shell triangles",
        outputs_dir / "winding_first_analysis_mesh.png",
    )
    plt.close(fig)

    winding_run = hostify_tree(
        run_winding_optimization(
            state,
            material,
            winding_cfg,
            geom,
            solve_compliance,
            failure_config=failure_cfg,
            friction_config=friction_cfg,
            snapshot_every=None if args.snapshot_every <= 0 else args.snapshot_every,
        )
    )
    winding_result = winding_run["result"]
    winding_history = [strip_legacy_winding_fields(entry) for entry in winding_run["history"]]
    jax.clear_caches()
    gc.collect()

    plot_objective_influence(
        baseline,
        winding_run["history"],
        winding_cfg,
        friction_cfg,
        outputs_dir / "winding_first_objective_influence.png",
        design_label="Winding-first",
    )
    winding_critical, burst_profile = plot_hashin_burst(
        state,
        winding_result,
        geom,
        failure_cfg,
        outputs_dir / "winding_first_hashin_burst.png",
        design_label="optimized winding",
    )

    nodes = mesh.nodes
    elems = mesh.elems
    base_u = np.asarray(baseline["displacement"]).reshape(len(nodes), 3)
    winding_u = np.asarray(winding_result["displacement"]).reshape(len(nodes), 3)

    baseline_vtu = write_vtu(
        outputs_dir / "copv_winding_base_failure.vtu",
        nodes,
        elems,
        base_u,
        np.asarray(baseline["thickness"]),
        np.asarray(baseline["density"]),
        np.asarray(baseline["fiber_dirs"]),
        np.asarray(baseline["coverage"]),
        extra_cell_data={
            "failure_index": np.asarray(baseline_failure["failure_index"]),
            "failure_with_margin": np.asarray(baseline_failure["failure_with_margin"]),
            "fiber_tension": np.asarray(baseline_failure["fiber_tension"]),
            "fiber_compression": np.asarray(baseline_failure["fiber_compression"]),
            "matrix_tension": np.asarray(baseline_failure["matrix_tension"]),
            "matrix_compression": np.asarray(baseline_failure["matrix_compression"]),
        },
    )
    winding_vtu = write_vtu(
        outputs_dir / "copv_winding_first.vtu",
        nodes,
        elems,
        winding_u,
        np.asarray(winding_result["thickness"]),
        np.asarray(winding_result["density"]),
        np.asarray(winding_result["fiber_dirs"]),
        np.asarray(winding_result["coverage"]),
        extra_cell_data={
            "failure_index": np.asarray(winding_result["failure_index"]),
            "failure_with_margin": np.asarray(winding_result["failure_with_margin"]),
            "fiber_tension": np.asarray(winding_result["fiber_tension"]),
            "fiber_compression": np.asarray(winding_result["fiber_compression"]),
            "matrix_tension": np.asarray(winding_result["matrix_tension"]),
            "matrix_compression": np.asarray(winding_result["matrix_compression"]),
            "winding_angle_deg": np.degrees(np.asarray(winding_result["winding_angle_field"])),
            "winding_added_thickness": np.asarray(winding_result["winding_thickness_field"]),
            "helical_added_thickness": np.asarray(winding_result["helical_thickness_field"]),
            "hoop_added_thickness": np.asarray(winding_result["hoop_thickness_field"]),
            "helical_pass_count": np.asarray(winding_result["helical_pass_field"]),
            "hoop_pass_count": np.asarray(winding_result["hoop_pass_field"]),
        },
    )

    fig, _ = plot_winding_process_paths(
        winding_result,
        geom,
        family_count=8,
        sample_count=320,
        save_path=outputs_dir / "winding_first_winding_paths.png",
    )
    plt.close(fig)

    winding_layout = build_winding_process_layout_data(
        winding_result,
        geom,
        family_count=8,
        sample_count=320,
    )
    winding_layout_path = save_layout_json(outputs_dir / "winding_first_layout.json", winding_layout)
    plot_manufacturing_constraints(
        winding_result,
        winding_layout,
        winding_cfg,
        friction_cfg,
        outputs_dir / "winding_first_manufacturing_constraints.png",
        design_label="Optimized",
    )

    pyvista_path: Path | None = None
    if not args.skip_pyvista:
        try:
            pyvista_path = save_explicit_manufacturing_layout_screenshot(
                baseline_vtu,
                outputs_dir / "pyvista_winding_first_layout.png",
                curve_points_list=[entry["points"] for entry in winding_layout["paths"]],
                curve_colors=[
                    "forestgreen" if entry["handedness"] == "clockwise" else "darkmagenta"
                    for entry in winding_layout["paths"]
                ],
                tow_radius=1.15,
                title="Optimized winding layout over COPV",
            )
        except Exception as exc:
            existing = outputs_dir / "pyvista_winding_first_layout.png"
            pyvista_path = existing if existing.exists() else None
            print(f"Skipped PyVista screenshot refresh: {exc}")
    else:
        pyvista_path = None

    abaqus_path = export_result_to_abaqus(
        state,
        winding_result,
        geom,
        outputs_dir / "optimized_copv_winding_first.inp",
        material=material,
        heading="Winding-first COPV layup exported from the staged JAX verification workflow",
    )

    snapshot_path: Path | None = None
    snapshots = winding_run.get("optimization_snapshots", [])
    if snapshots:
        snapshot_path = save_layout_json(
            outputs_dir / "winding_optimization_snapshots.json",
            {
                "case": "winding_first",
                "snapshot_every": int(args.snapshot_every),
                "geometry": {
                    "pressure": float(geom.pressure),
                    "outer_radius": float(geom.outer_radius),
                    "cylinder_length": float(geom.cylinder_length),
                    "thickness": float(geom.thickness),
                    "opening_radius": float(geom.opening_radius),
                    "boss_hmin": float(geom.boss_hmin),
                    "boss_refine_radius": float(geom.boss_refine_radius),
                },
                "winding_config": {
                    "winding_ctrl_count": int(winding_cfg.winding_ctrl_count),
                    "min_angle_deg": float(winding_cfg.min_angle_deg),
                    "max_angle_deg": float(winding_cfg.max_angle_deg),
                    "max_winding_thickness": float(winding_cfg.max_winding_thickness),
                    "tow_width": float(winding_cfg.tow_width),
                    "tow_thickness": float(winding_cfg.tow_thickness),
                    "winding_family_count": int(winding_cfg.winding_family_count),
                    "max_helical_pass_count": float(winding_cfg.max_helical_pass_count),
                    "max_hoop_pass_count": float(winding_cfg.max_hoop_pass_count),
                    "hoop_transition_length": float(winding_cfg.hoop_transition_length),
                    "mass_weight": float(winding_cfg.mass_weight),
                },
                "base_vtu": repo_rel(baseline_vtu),
                "snapshots": strip_legacy_winding_fields(snapshots),
            },
        )

    mass_ratio = float(np.asarray(winding_result["mass_metric"])) / float(np.asarray(baseline["mass_metric"]))
    summary = {
        "case": "winding_first",
        "description": "High-pressure shell-element winding-first verification using a production-scale towpreg thickness envelope.",
        "mesh": {
            "nodes": int(len(mesh.nodes)),
            "elements": int(len(mesh.elems)),
            "cell_type": "triangle_shell",
            "step": repo_rel(step_path),
            "msh": repo_rel(msh_path),
            "mesh_hmin": float(geom.mesh_hmin),
            "mesh_hmax": float(geom.mesh_hmax),
            "boss_hmin": float(geom.boss_hmin),
            "boss_refine_radius": float(geom.boss_refine_radius),
        },
        "geometry": {
            "pressure": float(geom.pressure),
            "outer_radius": float(geom.outer_radius),
            "cylinder_length": float(geom.cylinder_length),
            "thickness": float(geom.thickness),
            "opening_radius": float(geom.opening_radius),
        },
        "failure_config": {
            "margin_of_safety": float(failure_cfg.margin_of_safety),
            "penalty_weight": float(failure_cfg.penalty_weight),
            "allowables": vars(failure_cfg.allowables),
        },
        "friction_config": {
            "mu_max": float(friction_cfg.mu_max),
            "penalty_weight": float(friction_cfg.penalty_weight),
        },
        "winding_config": {
            "winding_ctrl_count": int(winding_cfg.winding_ctrl_count),
            "beta_schedule": [float(x) for x in winding_cfg.beta_schedule],
            "mass_weight": float(winding_cfg.mass_weight),
            "min_angle_deg": float(winding_cfg.min_angle_deg),
            "max_angle_deg": float(winding_cfg.max_angle_deg),
            "max_winding_thickness": float(winding_cfg.max_winding_thickness),
            "tow_width": float(winding_cfg.tow_width),
            "tow_thickness": float(winding_cfg.tow_thickness),
            "winding_family_count": int(winding_cfg.winding_family_count),
            "max_helical_pass_count": float(winding_cfg.max_helical_pass_count),
            "max_hoop_pass_count": float(winding_cfg.max_hoop_pass_count),
            "hoop_transition_length": float(winding_cfg.hoop_transition_length),
            "lbfgs_maxiter": int(winding_cfg.lbfgs_maxiter),
            "lbfgs_tol": float(winding_cfg.lbfgs_tol),
            "history_size": int(winding_cfg.history_size),
        },
        "baseline": {
            "strain_energy": float(np.asarray(baseline["compliance"])),
            "mass_metric": float(np.asarray(baseline["mass_metric"])),
            "fi_max": float(np.asarray(baseline_failure["fi_max"])),
            "fi_max_with_margin": float(np.max(np.asarray(baseline_failure["failure_with_margin"]))),
            "dominant_failure_mode": baseline_critical["dominant_mode"],
            "critical_surface_coordinate": {
                "s": baseline_critical["critical_s"],
                "phi_deg": baseline_critical["critical_phi_deg"],
            },
            "vtu": repo_rel(baseline_vtu),
        },
        "winding": {
            "history": winding_history,
            "objective": float(np.asarray(winding_result["objective"])),
            "strain_energy": float(np.asarray(winding_result["compliance"])),
            "mass_metric": float(np.asarray(winding_result["mass_metric"])),
            "strain_energy_ratio_vs_baseline": float(np.asarray(winding_result["compliance"])) / float(np.asarray(baseline["compliance"])),
            "mass_ratio_vs_baseline": mass_ratio,
            "mass_delta_percent_vs_baseline": 100.0 * (mass_ratio - 1.0),
            "fi_max": float(np.asarray(winding_result["fi_max"])),
            "fi_max_with_margin": float(np.max(np.asarray(winding_result["failure_with_margin"]))),
            "fi_reduction_vs_baseline": float(np.asarray(baseline_failure["fi_max"])) - float(np.asarray(winding_result["fi_max"])),
            "friction_mu_max_required": float(np.asarray(winding_result["mu_max_required"])),
            "friction_mu_allowable": float(friction_cfg.mu_max),
            "hashin_constraint_satisfied": bool(float(np.max(np.asarray(winding_result["failure_with_margin"]))) <= 1.0 + 1e-6),
            "friction_constraint_satisfied": bool(float(np.asarray(winding_result["mu_max_required"])) <= friction_cfg.mu_max + 1e-6),
            "burst_factor": float(burst_profile["burst_factor"]),
            "allowable_factor_with_margin": float(burst_profile["allowable_factor_with_margin"]),
            "allowable_pressure_with_margin": float(burst_profile["allowable_pressure_with_margin"]),
            "dominant_failure_mode": winding_critical["dominant_mode"],
            "critical_surface_coordinate": {
                "s": winding_critical["critical_s"],
                "phi_deg": winding_critical["critical_phi_deg"],
            },
            "control_angles_deg": np.degrees(np.asarray(winding_result["winding_angle_ctrl"])).tolist(),
            "control_thicknesses": np.asarray(winding_result["winding_thickness_ctrl"], dtype=np.float64).tolist(),
            "control_helical_pass_counts": np.asarray(winding_result["helical_pass_ctrl"], dtype=np.float64).tolist(),
            "control_hoop_pass_counts": np.asarray(winding_result["hoop_pass_ctrl"], dtype=np.float64).tolist(),
            "max_winding_thickness": float(np.max(np.asarray(winding_result["winding_thickness_field"]))),
            "max_helical_thickness": float(np.max(np.asarray(winding_result["helical_thickness_field"]))),
            "max_hoop_thickness": float(np.max(np.asarray(winding_result["hoop_thickness_field"]))),
            "mean_helical_pass_count": float(np.mean(np.asarray(winding_result["helical_pass_field"]))),
            "mean_hoop_pass_count": float(np.mean(np.asarray(winding_result["hoop_pass_field"]))),
            "vtu": repo_rel(winding_vtu),
            "abaqus_inp": repo_rel(abaqus_path),
        },
        "visualisations": {
            "mesh_plot": repo_rel(outputs_dir / "winding_first_analysis_mesh.png"),
            "objective_influence_plot": repo_rel(outputs_dir / "winding_first_objective_influence.png"),
            "hashin_burst_plot": repo_rel(outputs_dir / "winding_first_hashin_burst.png"),
            "winding_plot": repo_rel(outputs_dir / "winding_first_winding_paths.png"),
            "manufacturing_constraints_plot": repo_rel(outputs_dir / "winding_first_manufacturing_constraints.png"),
            "winding_layout_json": repo_rel(winding_layout_path),
            "pyvista_winding_explicit": None if pyvista_path is None else repo_rel(pyvista_path),
            "optimization_snapshots": None if snapshot_path is None else repo_rel(snapshot_path),
        },
    }

    summary_path = outputs_dir / "winding_first_summary.json"
    summary_path.unlink(missing_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(
        json.dumps(
            {
                "pressure": summary["geometry"]["pressure"],
                "baseline_fi_max": summary["baseline"]["fi_max"],
                "winding_fi_max": summary["winding"]["fi_max"],
                "mass_delta_percent": summary["winding"]["mass_delta_percent_vs_baseline"],
                "mu_max_required": summary["winding"]["friction_mu_max_required"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
