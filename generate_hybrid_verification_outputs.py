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


FAILURE_MODE_NAMES = [
    "Fiber tension",
    "Fiber compression",
    "Matrix tension",
    "Matrix compression",
]


def hostify_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)) if isinstance(x, jax.Array) else x,
        tree,
    )


def binned_max_profile(s_coords: np.ndarray, values: np.ndarray, bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    s_coords = np.asarray(s_coords, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    edges = np.linspace(float(np.min(s_coords)), float(np.max(s_coords)), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.full((bins,), np.nan, dtype=np.float64)
    for idx in range(bins):
        if idx == bins - 1:
            mask = (s_coords >= edges[idx]) & (s_coords <= edges[idx + 1])
        else:
            mask = (s_coords >= edges[idx]) & (s_coords < edges[idx + 1])
        if np.any(mask):
            profile[idx] = np.max(values[mask])
    return centers, profile


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root containing pyproject.toml and src/")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.abaqus_exporter import export_result_to_abaqus
from copv_opt.config import FailureConfig, FrictionConfig, GeometryConfig, HybridConfig, MaterialConfig, PatchConfig
from copv_opt.geometry import ensure_copv_mesh
from copv_opt.optimize import run_hybrid_optimization
from copv_opt.physics import baseline_response, build_copv_fem_state, estimate_burst_pressure_profile, make_solve_compliance
from copv_opt.visualize import (
    build_hybrid_winding_layout_data,
    build_patch_layout_data,
    plot_hybrid_winding_paths,
    plot_patch_projection,
    save_explicit_manufacturing_layout_screenshot,
    save_layout_json,
    show_copv_mesh,
    write_vtu,
)


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def plot_objective_influence(
    baseline: dict[str, Any],
    history: list[dict[str, Any]],
    hybrid_cfg: HybridConfig,
    friction_cfg: FrictionConfig,
    output_path: Path,
) -> None:
    beta = [entry["beta"] for entry in history]
    baseline_compliance = float(np.asarray(baseline["compliance"]))
    baseline_mass = float(np.asarray(baseline["mass_metric"]))

    mass_contrib = [hybrid_cfg.mass_weight * entry["mass_metric"] / baseline_mass for entry in history]
    l1_contrib = [hybrid_cfg.patch_l1_weight * entry["patch_l1"] for entry in history]
    overlap_contrib = [hybrid_cfg.overlap_penalty_weight * entry["overlap_penalty"] for entry in history]
    repulsion_contrib = [hybrid_cfg.repulsion_penalty_weight * entry["repulsion_penalty"] for entry in history]
    failure_contrib = [entry["failure_penalty"] for entry in history]
    friction_contrib = [entry["friction_penalty"] for entry in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    ax = axes[0]
    ax.plot(beta, [entry["objective"] for entry in history], marker="o", color="crimson", label="Objective")
    ax.plot(
        beta,
        [entry["strain_energy"] / baseline_compliance for entry in history],
        marker="o",
        color="steelblue",
        label="Compliance / baseline",
    )
    ax.plot(
        beta,
        [entry["mass_metric"] / baseline_mass for entry in history],
        marker="o",
        color="darkorange",
        label="Mass / baseline",
    )
    ax.set_xlabel("beta")
    ax.set_ylabel("normalized metric")
    ax.set_title("Hybrid continuation overview")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(beta, mass_contrib, marker="o", color="darkorange", label="Mass term")
    ax.plot(beta, l1_contrib, marker="o", color="slateblue", label="Patch L1 term")
    ax.plot(beta, overlap_contrib, marker="o", color="royalblue", label="Overlap term")
    ax.plot(beta, repulsion_contrib, marker="o", color="teal", label="Repulsion term")
    ax.plot(beta, failure_contrib, marker="o", color="forestgreen", label="Hashin penalty")
    ax.plot(beta, friction_contrib, marker="o", color="darkmagenta", label="Friction penalty")
    ax.set_xlabel("beta")
    ax.set_ylabel("Objective contribution")
    ax.set_title("What drives the objective")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(beta, [entry["active_patch_count"] for entry in history], marker="o", color="steelblue", label="Active patch count")
    ax.plot(beta, [entry["max_patch_thickness"] for entry in history], marker="o", color="royalblue", label="Max patch thickness")
    ax.plot(
        beta,
        [entry["max_winding_thickness"] for entry in history],
        marker="o",
        color="seagreen",
        label="Max winding thickness",
    )
    ax.plot(beta, [entry["mu_max_required"] for entry in history], marker="o", color="darkmagenta", label="Required friction")
    ax.axhline(friction_cfg.mu_max, color="darkmagenta", linestyle="--", linewidth=1.1)
    ax.set_xlabel("beta")
    ax.set_ylabel("Constraint / design metric")
    ax.set_title("Manufacturing and sparsity response")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_hashin_burst(
    state: dict[str, Any],
    hybrid_result: dict[str, Any],
    geom: GeometryConfig,
    failure_cfg: FailureConfig,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_s = np.asarray(jax.device_get(state["s_coords"]))
    state_phi_deg = np.degrees(np.asarray(jax.device_get(state["phi_coords"])))

    failure_with_margin = np.asarray(hybrid_result["failure_with_margin"])
    failure_index = np.asarray(hybrid_result["failure_index"])
    fiber_tension = np.asarray(hybrid_result["fiber_tension"])
    fiber_compression = np.asarray(hybrid_result["fiber_compression"])
    matrix_tension = np.asarray(hybrid_result["matrix_tension"])
    matrix_compression = np.asarray(hybrid_result["matrix_compression"])
    mode_stack = np.stack([fiber_tension, fiber_compression, matrix_tension, matrix_compression], axis=0)

    critical_idx = int(np.argmax(failure_with_margin))
    dominant_mode_idx = int(np.argmax(mode_stack[:, critical_idx]))
    critical_summary = {
        "critical_element_index": critical_idx,
        "critical_s": float(state_s[critical_idx]),
        "critical_phi_deg": float(state_phi_deg[critical_idx]),
        "critical_failure_index": float(failure_index[critical_idx]),
        "critical_failure_with_margin": float(failure_with_margin[critical_idx]),
        "dominant_mode": FAILURE_MODE_NAMES[dominant_mode_idx],
    }

    burst_profile = estimate_burst_pressure_profile(
        np.asarray(hybrid_result["local_stress"]),
        failure_cfg.allowables,
        operating_pressure=float(geom.pressure),
        margin_of_safety=float(failure_cfg.margin_of_safety),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    sc = ax.scatter(state_s, state_phi_deg, c=failure_with_margin, s=18, cmap="magma", alpha=0.9)
    ax.scatter([critical_summary["critical_s"]], [critical_summary["critical_phi_deg"]], color="cyan", s=64, edgecolor="black")
    ax.set_xlabel("Meridional coordinate")
    ax.set_ylabel("Azimuth [deg]")
    ax.set_title("Hashin failure field on the COPV surface")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Failure index * margin")

    ax = axes[1]
    for label, values, color in [
        ("Fiber tension", fiber_tension, "crimson"),
        ("Fiber compression", fiber_compression, "royalblue"),
        ("Matrix tension", matrix_tension, "darkorange"),
        ("Matrix compression", matrix_compression, "forestgreen"),
    ]:
        centers, profile = binned_max_profile(state_s, values)
        ax.plot(centers, profile, linewidth=2, label=label, color=color)
    centers, profile = binned_max_profile(state_s, failure_with_margin)
    ax.plot(centers, profile, linewidth=2.2, linestyle="--", color="black", label="Max FI * margin")
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Meridional coordinate")
    ax.set_ylabel("Binned max failure metric")
    ax.set_title("Where the laminate is critical")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(burst_profile["pressure_factors"], burst_profile["failure_curve"], color="steelblue", linewidth=2, label="Max Hashin FI")
    ax.plot(
        burst_profile["pressure_factors"],
        burst_profile["failure_curve_with_margin"],
        color="darkmagenta",
        linewidth=2,
        label="Max Hashin FI * margin",
    )
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.1)
    ax.axvline(burst_profile["burst_factor"], color="steelblue", linestyle=":", linewidth=1.4)
    ax.axvline(burst_profile["allowable_factor_with_margin"], color="darkmagenta", linestyle=":", linewidth=1.4)
    ax.set_xlabel("Pressure scale factor")
    ax.set_ylabel("Failure metric")
    ax.set_title("Burst-pressure proxy from the hybrid stress field")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return critical_summary, burst_profile


def plot_manufacturing_constraints(
    hybrid_result: dict[str, Any],
    hybrid_winding_layout: dict[str, Any],
    hybrid_cfg: HybridConfig,
    friction_cfg: FrictionConfig,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    ax = axes[0]
    ax.plot(hybrid_winding_layout["sample_s"], hybrid_winding_layout["angle_profile_deg"], color="seagreen", linewidth=2, label="Angle profile")
    ax.scatter(
        hybrid_winding_layout["control_s"],
        hybrid_winding_layout["control_angle_deg"],
        color="black",
        s=28,
        zorder=3,
        label="Control points",
    )
    ax.set_xlabel("Meridional coordinate")
    ax.set_ylabel("Winding angle [deg]")
    ax.set_title("Hybrid winding angle field")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(
        hybrid_winding_layout["sample_s"],
        hybrid_winding_layout["thickness_profile"],
        color="royalblue",
        linewidth=2,
        label="Winding thickness field",
    )
    if hybrid_cfg.patch_count > 0:
        ax.scatter(
            np.asarray(hybrid_result["patch_s"]),
            np.asarray(hybrid_result["patch_thickness"]),
            color="steelblue",
            s=34,
            label="AFP patch thickness",
        )
    ax.set_xlabel("Meridional coordinate / patch center")
    ax.set_ylabel("Added thickness")
    ax.set_title("How winding and AFP share material")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(hybrid_winding_layout["sample_s"], hybrid_winding_layout["mu_required"], color="darkmagenta", linewidth=2, label="Required friction")
    ax.axhline(friction_cfg.mu_max, color="0.30", linestyle="--", linewidth=1.1, label="Allowable friction")
    ax.set_xlabel("Meridional coordinate")
    ax.set_ylabel("Required friction coefficient")
    ax.set_title("Manufacturability constraint on the winding field")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def build_summary(
    outputs_dir: Path,
    step_path: Path,
    msh_path: Path,
    mesh: Any,
    geom: GeometryConfig,
    material: MaterialConfig,
    failure_cfg: FailureConfig,
    friction_cfg: FrictionConfig,
    hybrid_cfg: HybridConfig,
    baseline: dict[str, Any],
    hybrid_run: dict[str, Any],
    hybrid_result: dict[str, Any],
    critical_summary: dict[str, Any],
    burst_profile: dict[str, Any],
    base_vtu: Path,
    hybrid_vtu: Path,
    abaqus_path: Path,
    hybrid_patch_layout_path: Path | None,
    hybrid_winding_layout_path: Path,
    pyvista_path: Path | None,
) -> dict[str, Any]:
    fi_max = float(np.asarray(hybrid_result["fi_max"]))
    fi_max_with_margin = float(np.max(np.asarray(hybrid_result["failure_with_margin"])))
    mu_max_required = float(np.asarray(hybrid_result["mu_max_required"]))
    patch_thickness = np.asarray(hybrid_result["patch_thickness"])
    active_patch_count = int(np.sum(patch_thickness > 0.05 * hybrid_cfg.max_patch_thickness))

    return {
        "jax_backend": jax.default_backend(),
        "mesh": {
            "nodes": int(len(mesh.nodes)),
            "elements": int(len(mesh.elems)),
            "step": repo_rel(step_path),
            "msh": repo_rel(msh_path),
            "mesh_hmin": float(geom.mesh_hmin),
            "mesh_hmax": float(geom.mesh_hmax),
        },
        "failure_config": {
            "allowables": vars(failure_cfg.allowables),
            "margin_of_safety": float(failure_cfg.margin_of_safety),
            "penalty_weight": float(failure_cfg.penalty_weight),
        },
        "friction_config": {
            "mu_max": float(friction_cfg.mu_max),
            "penalty_weight": float(friction_cfg.penalty_weight),
        },
        "hybrid_config": {
            "winding_ctrl_count": int(hybrid_cfg.winding_ctrl_count),
            "patch_count": int(hybrid_cfg.patch_count),
            "beta_schedule": [float(x) for x in hybrid_cfg.beta_schedule],
            "patch_l1_weight": float(hybrid_cfg.patch_l1_weight),
            "max_winding_thickness": float(hybrid_cfg.max_winding_thickness),
            "max_patch_thickness": float(hybrid_cfg.max_patch_thickness),
            "lbfgs_maxiter": int(hybrid_cfg.lbfgs_maxiter),
            "lbfgs_tol": float(hybrid_cfg.lbfgs_tol),
            "friction_cap_sample_count": int(hybrid_cfg.friction_cap_sample_count),
            "friction_cylinder_sample_count": int(hybrid_cfg.friction_cylinder_sample_count),
        },
        "material": {
            "e_xx": float(material.e_xx),
            "e_yy": float(material.e_yy),
            "e_zz": float(material.e_zz),
            "nu_xy": float(material.nu_xy),
            "nu_xz": float(material.nu_xz),
            "nu_yz": float(material.nu_yz),
            "g_xy": float(material.g_xy),
            "g_xz": float(material.g_xz),
            "g_yz": float(material.g_yz),
            "base_plies": int(material.base_plies),
            "ply_thickness": float(material.ply_thickness),
        },
        "baseline": {
            "strain_energy": float(np.asarray(baseline["compliance"])),
            "mass_metric": float(np.asarray(baseline["mass_metric"])),
            "vtu": repo_rel(base_vtu),
        },
        "hybrid": {
            "history": hybrid_run["history"],
            "objective": float(np.asarray(hybrid_result["objective"])),
            "strain_energy": float(np.asarray(hybrid_result["compliance"])),
            "mass_metric": float(np.asarray(hybrid_result["mass_metric"])),
            "strain_energy_ratio_vs_baseline": float(np.asarray(hybrid_result["compliance"])) / float(np.asarray(baseline["compliance"])),
            "mass_ratio_vs_baseline": float(np.asarray(hybrid_result["mass_metric"])) / float(np.asarray(baseline["mass_metric"])),
            "fi_max": fi_max,
            "fi_max_with_margin": fi_max_with_margin,
            "friction_mu_max_required": mu_max_required,
            "friction_mu_allowable": float(friction_cfg.mu_max),
            "hashin_constraint_satisfied": bool(fi_max_with_margin <= 1.0 + 1e-6),
            "friction_constraint_satisfied": bool(mu_max_required <= friction_cfg.mu_max + 1e-6),
            "burst_factor": float(burst_profile["burst_factor"]),
            "allowable_factor_with_margin": float(burst_profile["allowable_factor_with_margin"]),
            "allowable_pressure_with_margin": float(burst_profile["allowable_pressure_with_margin"]),
            "dominant_failure_mode": critical_summary["dominant_mode"],
            "critical_surface_coordinate": {
                "s": critical_summary["critical_s"],
                "phi_deg": critical_summary["critical_phi_deg"],
            },
            "active_patch_count": active_patch_count,
            "vtu": repo_rel(hybrid_vtu),
            "abaqus_inp": repo_rel(abaqus_path),
        },
        "visualisations": {
            "mesh_plot": repo_rel(outputs_dir / "copv_analysis_mesh.png"),
            "objective_influence_plot": repo_rel(outputs_dir / "hybrid_objective_influence.png"),
            "hashin_burst_plot": repo_rel(outputs_dir / "hybrid_hashin_burst.png"),
            "winding_plot": repo_rel(outputs_dir / "copv_hybrid_winding_paths.png"),
            "manufacturing_constraints_plot": repo_rel(outputs_dir / "hybrid_manufacturing_constraints.png"),
            "patch_projection": None if hybrid_patch_layout_path is None else repo_rel(outputs_dir / "copv_hybrid_patch_projection.png"),
            "patch_layout_json": None if hybrid_patch_layout_path is None else repo_rel(hybrid_patch_layout_path),
            "winding_layout_json": repo_rel(hybrid_winding_layout_path),
            "pyvista_hybrid_explicit": None if pyvista_path is None else repo_rel(pyvista_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate the packaged COPV hybrid verification outputs.")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs", help="Output directory to refresh.")
    parser.add_argument("--remesh", action="store_true", help="Rebuild the STEP mesh with gmsh instead of reusing outputs/copv_shell.msh.")
    parser.add_argument(
        "--skip-pyvista",
        action="store_true",
        help="Skip the optional off-screen PyVista layout screenshot even if PyVista is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs.resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    geom = GeometryConfig()
    material = MaterialConfig()
    failure_cfg = FailureConfig()
    friction_cfg = FrictionConfig()
    hybrid_cfg = HybridConfig()
    verification_patch_cfg = PatchConfig(
        count=hybrid_cfg.patch_count,
        length=hybrid_cfg.patch_length,
        width=hybrid_cfg.patch_width,
    )

    step_path = outputs_dir / "copv_shell.step"
    msh_path = outputs_dir / "copv_shell.msh"
    mesh = ensure_copv_mesh(step_path, msh_path, geom, remesh=args.remesh)

    state = build_copv_fem_state(mesh.nodes, mesh.elems, material, geom)
    solve_compliance = make_solve_compliance(state)
    baseline = hostify_tree(baseline_response(state, material, solve_compliance))
    gc.collect()

    fig = show_copv_mesh(
        mesh.nodes,
        np.asarray(state["outer_faces"]),
        geom,
        f"COPV analysis mesh: {len(mesh.nodes)} nodes / {len(mesh.elems)} tetra",
        outputs_dir / "copv_analysis_mesh.png",
    )
    plt.close(fig)

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
        outputs_dir / "hybrid_objective_influence.png",
    )
    critical_summary, burst_profile = plot_hashin_burst(
        state,
        hybrid_result,
        geom,
        failure_cfg,
        outputs_dir / "hybrid_hashin_burst.png",
    )

    nodes = mesh.nodes
    elems = mesh.elems
    base_u = np.asarray(baseline["displacement"]).reshape(len(nodes), 3)
    hybrid_u = np.asarray(hybrid_result["displacement"]).reshape(len(nodes), 3)

    base_vtu = write_vtu(
        outputs_dir / "copv_base.vtu",
        nodes,
        elems,
        base_u,
        np.asarray(baseline["thickness"]),
        np.asarray(baseline["density"]),
        np.asarray(baseline["fiber_dirs"]),
        np.asarray(baseline["coverage"]),
    )
    hybrid_vtu = write_vtu(
        outputs_dir / "copv_hybrid_jax.vtu",
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

    hybrid_patch_layout_path: Path | None = None
    hybrid_patch_layout = None
    if hybrid_cfg.patch_count > 0:
        hybrid_patch_state = {
            "s_coords": np.asarray(hybrid_result["patch_s"]),
            "phis": np.asarray(hybrid_result["patch_phi"]),
            "alphas": np.asarray(hybrid_result["patch_alpha"]),
        }
        hybrid_patch_layout = build_patch_layout_data(hybrid_patch_state, verification_patch_cfg, geom)
        hybrid_patch_layout_path = save_layout_json(outputs_dir / "hybrid_patch_layout.json", hybrid_patch_layout)
        fig = plot_patch_projection(
            hybrid_patch_state,
            verification_patch_cfg,
            geom,
            outputs_dir / "copv_hybrid_patch_projection.png",
        )
        plt.close(fig)

    fig, _ = plot_hybrid_winding_paths(
        hybrid_result,
        geom,
        family_count=8,
        sample_count=280,
        save_path=outputs_dir / "copv_hybrid_winding_paths.png",
    )
    plt.close(fig)

    hybrid_winding_layout = build_hybrid_winding_layout_data(
        hybrid_result,
        geom,
        family_count=8,
        sample_count=280,
    )
    hybrid_winding_layout_path = save_layout_json(outputs_dir / "hybrid_winding_layout.json", hybrid_winding_layout)
    plot_manufacturing_constraints(
        hybrid_result,
        hybrid_winding_layout,
        hybrid_cfg,
        friction_cfg,
        outputs_dir / "hybrid_manufacturing_constraints.png",
    )

    pyvista_path: Path | None = None
    if not args.skip_pyvista:
        try:
            pyvista_path = save_explicit_manufacturing_layout_screenshot(
                base_vtu,
                outputs_dir / "pyvista_hybrid_explicit_layout.png",
                curve_points_list=[entry["points"] for entry in hybrid_winding_layout["paths"]],
                curve_colors=[
                    "forestgreen" if entry["handedness"] == "clockwise" else "darkmagenta"
                    for entry in hybrid_winding_layout["paths"]
                ],
                patch_polygons=None if hybrid_patch_layout is None else [entry["corners"] for entry in hybrid_patch_layout["patches"]],
                tow_radius=1.15,
                title="Hybrid winding + AFP layout over COPV",
            )
        except Exception as exc:
            existing = outputs_dir / "pyvista_hybrid_explicit_layout.png"
            pyvista_path = existing if existing.exists() else None
            print(f"Skipped PyVista screenshot refresh: {exc}")
    else:
        existing = outputs_dir / "pyvista_hybrid_explicit_layout.png"
        pyvista_path = existing if existing.exists() else None

    abaqus_path = export_result_to_abaqus(
        state,
        hybrid_result,
        geom,
        outputs_dir / "optimized_copv_hybrid.inp",
        material=material,
        heading="Hybrid COPV layup exported from the scriptable JAX verification workflow",
    )

    summary = build_summary(
        outputs_dir=outputs_dir,
        step_path=step_path,
        msh_path=msh_path,
        mesh=mesh,
        geom=geom,
        material=material,
        failure_cfg=failure_cfg,
        friction_cfg=friction_cfg,
        hybrid_cfg=hybrid_cfg,
        baseline=baseline,
        hybrid_run=hybrid_run,
        hybrid_result=hybrid_result,
        critical_summary=critical_summary,
        burst_profile=burst_profile,
        base_vtu=base_vtu,
        hybrid_vtu=hybrid_vtu,
        abaqus_path=abaqus_path,
        hybrid_patch_layout_path=hybrid_patch_layout_path,
        hybrid_winding_layout_path=hybrid_winding_layout_path,
        pyvista_path=pyvista_path,
    )

    summary_path = outputs_dir / "hybrid_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(
        json.dumps(
            {
                "baseline_strain_energy": summary["baseline"]["strain_energy"],
                "hybrid_strain_energy": summary["hybrid"]["strain_energy"],
                "hybrid_objective": summary["hybrid"]["objective"],
                "fi_max_with_margin": summary["hybrid"]["fi_max_with_margin"],
                "mu_max_required": summary["hybrid"]["friction_mu_max_required"],
                "active_patch_count": summary["hybrid"]["active_patch_count"],
                "dominant_failure_mode": summary["hybrid"]["dominant_failure_mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
