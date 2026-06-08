from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from generate_hybrid_verification_outputs import (
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
from copv_opt.config import FailureConfig, FrictionConfig, GeometryConfig, HybridConfig, MaterialConfig, PatchConfig
from copv_opt.geometry import ensure_copv_mesh
from copv_opt.optimize import active_patch_threshold, count_active_patches_np, run_hybrid_optimization
from copv_opt.physics import baseline_response, build_copv_fem_state, evaluate_hashin_failure, make_solve_compliance
from copv_opt.visualize import (
    build_hybrid_winding_layout_data,
    build_patch_layout_data,
    plot_hybrid_winding_paths,
    plot_patch_projection,
    save_layout_json,
    show_copv_mesh,
    write_vtu,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the AFP-needed verification outputs for the COPV package.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "afp_activation",
        help="Directory for the AFP-needed case artifacts.",
    )
    parser.add_argument(
        "--remesh",
        action="store_true",
        help="Rebuild the shared STEP/Gmsh model instead of reusing outputs/copv_shell.msh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs.resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    shared_outputs = PROJECT_ROOT / "outputs"
    step_path = shared_outputs / "copv_shell.step"
    msh_path = shared_outputs / "copv_shell.msh"

    geom = GeometryConfig(pressure=6.85)
    material = MaterialConfig()
    failure_cfg = FailureConfig(margin_of_safety=1.0, penalty_weight=900.0)
    friction_cfg = FrictionConfig()
    hybrid_cfg = HybridConfig(
        patch_count=4,
        patch_length=68.0,
        patch_width=24.0,
        patch_l1_weight=0.0005,
        max_winding_thickness=0.008,
        max_patch_thickness=3.0,
        patch_seed_thickness=1.5,
        winding_seed_thickness=0.008,
        min_angle_deg=40.0,
        max_angle_deg=44.0,
        winding_seed_angle_deg=42.0,
    )
    patch_cfg = PatchConfig(
        count=hybrid_cfg.patch_count,
        length=hybrid_cfg.patch_length,
        width=hybrid_cfg.patch_width,
    )

    mesh = ensure_copv_mesh(step_path, msh_path, geom, remesh=args.remesh)
    state = build_copv_fem_state(mesh.nodes, mesh.elems, material, geom)
    solve_compliance = make_solve_compliance(state)
    baseline = hostify_tree(baseline_response(state, material, solve_compliance))
    c_base = jnp.broadcast_to(state["c_mat"], (state["element_count"],) + state["c_mat"].shape)
    baseline_failure = hostify_tree(
        evaluate_hashin_failure(state, baseline["displacement"], c_base, baseline["fiber_dirs"], failure_cfg)
    )
    baseline_critical = build_critical_summary(state, baseline_failure)

    fig = show_copv_mesh(
        mesh.nodes,
        np.asarray(state["outer_faces"]),
        geom,
        f"AFP-needed case mesh: {len(mesh.nodes)} nodes / {len(mesh.elems)} tetra",
        outputs_dir / "copv_analysis_mesh.png",
    )
    del fig

    hybrid_run = hostify_tree(
        run_hybrid_optimization(
            state,
            material,
            hybrid_cfg,
            geom,
            solve_compliance,
            failure_config=failure_cfg,
            friction_config=friction_cfg,
        )
    )
    hybrid_result = hybrid_run["result"]
    jax.clear_caches()
    gc.collect()

    plot_objective_influence(
        baseline,
        hybrid_run["history"],
        hybrid_cfg,
        friction_cfg,
        outputs_dir / "afp_objective_influence.png",
    )
    hybrid_critical, burst_profile = plot_hashin_burst(
        state,
        hybrid_result,
        geom,
        failure_cfg,
        outputs_dir / "afp_hashin_burst.png",
    )

    nodes = mesh.nodes
    elems = mesh.elems
    base_u = np.asarray(baseline["displacement"]).reshape(len(nodes), 3)
    hybrid_u = np.asarray(hybrid_result["displacement"]).reshape(len(nodes), 3)

    baseline_vtu = write_vtu(
        outputs_dir / "copv_base_failure.vtu",
        nodes,
        elems,
        base_u,
        np.asarray(baseline["thickness"]),
        np.asarray(baseline["density"]),
        np.asarray(baseline["fiber_dirs"]),
        np.asarray(baseline["coverage"]),
        extra_cell_data={
            "failure_index": np.asarray(baseline_failure["failure_index"]),
            "fiber_tension": np.asarray(baseline_failure["fiber_tension"]),
            "matrix_tension": np.asarray(baseline_failure["matrix_tension"]),
        },
    )
    hybrid_vtu = write_vtu(
        outputs_dir / "copv_hybrid_afp_activation.vtu",
        nodes,
        elems,
        hybrid_u,
        np.asarray(hybrid_result["thickness"]),
        np.asarray(hybrid_result["density"]),
        np.asarray(hybrid_result["fiber_dirs"]),
        np.asarray(hybrid_result["coverage"]),
        extra_cell_data={
            "failure_index": np.asarray(hybrid_result["failure_index"]),
            "failure_with_margin": np.asarray(hybrid_result["failure_with_margin"]),
            "fiber_tension": np.asarray(hybrid_result["fiber_tension"]),
            "fiber_compression": np.asarray(hybrid_result["fiber_compression"]),
            "matrix_tension": np.asarray(hybrid_result["matrix_tension"]),
            "matrix_compression": np.asarray(hybrid_result["matrix_compression"]),
            "winding_angle_deg": np.degrees(np.asarray(hybrid_result["winding_angle_field"])),
            "winding_added_thickness": np.asarray(hybrid_result["winding_thickness_field"]),
            "patch_added_thickness": np.asarray(hybrid_result["patch_added_thickness"]),
        },
    )

    patch_state = {
        "s_coords": np.asarray(hybrid_result["patch_s"]),
        "phis": np.asarray(hybrid_result["patch_phi"]),
        "alphas": np.asarray(hybrid_result["patch_alpha"]),
    }
    patch_layout = build_patch_layout_data(patch_state, patch_cfg, geom)
    patch_layout_path = save_layout_json(outputs_dir / "afp_patch_layout.json", patch_layout)
    fig = plot_patch_projection(
        patch_state,
        patch_cfg,
        geom,
        outputs_dir / "afp_patch_projection.png",
    )
    del fig

    fig, _ = plot_hybrid_winding_paths(
        hybrid_result,
        geom,
        family_count=8,
        sample_count=280,
        save_path=outputs_dir / "afp_winding_paths.png",
    )
    del fig

    winding_layout = build_hybrid_winding_layout_data(
        hybrid_result,
        geom,
        family_count=8,
        sample_count=280,
    )
    winding_layout_path = save_layout_json(outputs_dir / "afp_winding_layout.json", winding_layout)
    plot_manufacturing_constraints(
        hybrid_result,
        winding_layout,
        hybrid_cfg,
        friction_cfg,
        outputs_dir / "afp_manufacturing_constraints.png",
    )

    abaqus_path = export_result_to_abaqus(
        state,
        hybrid_result,
        geom,
        outputs_dir / "optimized_copv_hybrid_afp.inp",
        material=material,
        heading="AFP-needed hybrid COPV case exported from the JAX verification workflow",
    )

    patch_thickness = np.asarray(hybrid_result["patch_thickness"], dtype=np.float64)
    active_count = count_active_patches_np(patch_thickness, hybrid_cfg)
    summary = {
        "case": "afp_activation",
        "description": "High-pressure constrained-winding verification where AFP is required to pull fi_max below 1.0.",
        "mesh": {
            "nodes": int(len(mesh.nodes)),
            "elements": int(len(mesh.elems)),
            "step": repo_rel(step_path),
            "msh": repo_rel(msh_path),
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
        "hybrid_config": {
            "patch_count": int(hybrid_cfg.patch_count),
            "patch_length": float(hybrid_cfg.patch_length),
            "patch_width": float(hybrid_cfg.patch_width),
            "patch_l1_weight": float(hybrid_cfg.patch_l1_weight),
            "max_winding_thickness": float(hybrid_cfg.max_winding_thickness),
            "max_patch_thickness": float(hybrid_cfg.max_patch_thickness),
            "min_angle_deg": float(hybrid_cfg.min_angle_deg),
            "max_angle_deg": float(hybrid_cfg.max_angle_deg),
            "active_patch_threshold": float(active_patch_threshold(hybrid_cfg)),
        },
        "baseline": {
            "strain_energy": float(np.asarray(baseline["compliance"])),
            "mass_metric": float(np.asarray(baseline["mass_metric"])),
            "fi_max": float(np.asarray(baseline_failure["fi_max"])),
            "dominant_failure_mode": baseline_critical["dominant_mode"],
            "critical_surface_coordinate": {
                "s": baseline_critical["critical_s"],
                "phi_deg": baseline_critical["critical_phi_deg"],
            },
            "vtu": repo_rel(baseline_vtu),
        },
        "hybrid": {
            "objective": float(np.asarray(hybrid_result["objective"])),
            "strain_energy": float(np.asarray(hybrid_result["compliance"])),
            "mass_metric": float(np.asarray(hybrid_result["mass_metric"])),
            "fi_max": float(np.asarray(hybrid_result["fi_max"])),
            "fi_reduction_vs_baseline": float(np.asarray(baseline_failure["fi_max"])) - float(np.asarray(hybrid_result["fi_max"])),
            "dominant_failure_mode": hybrid_critical["dominant_mode"],
            "critical_surface_coordinate": {
                "s": hybrid_critical["critical_s"],
                "phi_deg": hybrid_critical["critical_phi_deg"],
            },
            "active_patch_count": int(active_count),
            "patch_thicknesses": patch_thickness.tolist(),
            "patch_centers": [
                {
                    "s": float(s_coord),
                    "phi_deg": float(phi_deg),
                }
                for s_coord, phi_deg in zip(
                    np.asarray(hybrid_result["patch_s"], dtype=np.float64),
                    np.degrees(np.asarray(hybrid_result["patch_phi"], dtype=np.float64)),
                )
            ],
            "max_winding_thickness": float(np.max(np.asarray(hybrid_result["winding_thickness_field"]))),
            "friction_mu_max_required": float(np.asarray(hybrid_result["mu_max_required"])),
            "history": hybrid_run["history"],
            "vtu": repo_rel(hybrid_vtu),
            "abaqus_inp": repo_rel(abaqus_path),
        },
        "artifacts": {
            "mesh_plot": repo_rel(outputs_dir / "copv_analysis_mesh.png"),
            "objective_influence_plot": repo_rel(outputs_dir / "afp_objective_influence.png"),
            "hashin_burst_plot": repo_rel(outputs_dir / "afp_hashin_burst.png"),
            "patch_projection_plot": repo_rel(outputs_dir / "afp_patch_projection.png"),
            "manufacturing_constraints_plot": repo_rel(outputs_dir / "afp_manufacturing_constraints.png"),
            "winding_plot": repo_rel(outputs_dir / "afp_winding_paths.png"),
            "patch_layout_json": repo_rel(patch_layout_path),
            "winding_layout_json": repo_rel(winding_layout_path),
        },
        "burst_proxy": {
            "burst_factor": float(burst_profile["burst_factor"]),
            "allowable_factor_with_margin": float(burst_profile["allowable_factor_with_margin"]),
        },
    }
    summary_path = outputs_dir / "afp_case_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(
        json.dumps(
            {
                "pressure": summary["geometry"]["pressure"],
                "baseline_fi_max": summary["baseline"]["fi_max"],
                "hybrid_fi_max": summary["hybrid"]["fi_max"],
                "active_patch_count": summary["hybrid"]["active_patch_count"],
                "patch_thicknesses": summary["hybrid"]["patch_thicknesses"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
