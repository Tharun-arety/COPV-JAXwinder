from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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

from copv_opt.config import FailureConfig, FrictionConfig, GeometryConfig, WindingOptimizationConfig  # noqa: E402
from copv_opt.physics import estimate_burst_pressure_profile  # noqa: E402


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _prepare_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)


def plot_objective_influence(
    baseline: dict[str, Any],
    history: list[dict[str, Any]],
    winding_cfg: WindingOptimizationConfig,
    friction_cfg: FrictionConfig,
    output_path: Path,
    design_label: str = "Design",
) -> None:
    beta = [entry["beta"] for entry in history]
    baseline_compliance = float(np.asarray(baseline["compliance"]))
    baseline_mass = float(np.asarray(baseline["mass_metric"]))

    mass_contrib = [winding_cfg.mass_weight * entry["mass_metric"] / baseline_mass for entry in history]
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
    ax.set_title(f"{design_label} continuation overview")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(beta, mass_contrib, marker="o", color="darkorange", label="Mass term")
    ax.plot(beta, failure_contrib, marker="o", color="forestgreen", label="Hashin penalty")
    ax.plot(beta, friction_contrib, marker="o", color="darkmagenta", label="Friction penalty")
    ax.set_xlabel("beta")
    ax.set_ylabel("Objective contribution")
    ax.set_title("What drives the objective")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
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
    ax.set_title("Manufacturing and design response")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _prepare_output_path(output_path)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_hashin_burst(
    state: dict[str, Any],
    result: dict[str, Any],
    geom: GeometryConfig,
    failure_cfg: FailureConfig,
    output_path: Path,
    design_label: str = "design",
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_s = np.asarray(jax.device_get(state["s_coords"]))
    state_phi_deg = np.degrees(np.asarray(jax.device_get(state["phi_coords"])))

    failure_with_margin = np.asarray(result["failure_with_margin"])
    failure_index = np.asarray(result["failure_index"])
    fiber_tension = np.asarray(result["fiber_tension"])
    fiber_compression = np.asarray(result["fiber_compression"])
    matrix_tension = np.asarray(result["matrix_tension"])
    matrix_compression = np.asarray(result["matrix_compression"])
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
        np.asarray(result["local_stress"]),
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
    ax.set_title(f"Burst-pressure proxy from the {design_label} stress field")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _prepare_output_path(output_path)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return critical_summary, burst_profile


def plot_manufacturing_constraints(
    result: dict[str, Any],
    winding_layout: dict[str, Any],
    winding_cfg: WindingOptimizationConfig,
    friction_cfg: FrictionConfig,
    output_path: Path,
    design_label: str = "Design",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    ax = axes[0]
    ax.plot(winding_layout["sample_s"], winding_layout["angle_profile_deg"], color="seagreen", linewidth=2, label="Angle profile")
    ax.scatter(
        winding_layout["control_s"],
        winding_layout["control_angle_deg"],
        color="black",
        s=28,
        zorder=3,
        label="Control points",
    )
    ax.set_xlabel("Meridional coordinate")
    ax.set_ylabel("Winding angle [deg]")
    ax.set_title(f"{design_label} winding angle field")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    sample_s = np.asarray(winding_layout["sample_s"])
    thickness_profile = np.asarray(winding_layout["thickness_profile"])
    helical_thickness_profile = winding_layout.get("helical_thickness_profile")
    hoop_thickness_profile = winding_layout.get("hoop_thickness_profile")
    if helical_thickness_profile is not None:
        ax.plot(
            sample_s,
            np.asarray(helical_thickness_profile),
            color="seagreen",
            linewidth=1.6,
            label="Helical winding thickness",
        )
    if hoop_thickness_profile is not None:
        ax.plot(
            sample_s,
            np.asarray(hoop_thickness_profile),
            color="darkorange",
            linewidth=1.6,
            label="Hoop winding thickness",
        )
    ax.plot(sample_s, thickness_profile, color="royalblue", linewidth=2.4, label="Total winding thickness")
    ax.axhline(
        float(winding_cfg.max_winding_thickness),
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label="Configured thickness cap",
    )
    ax.set_xlabel("Meridional coordinate / control point")
    ax.set_ylabel("Added thickness")
    ax.set_title("How material is distributed along the winding")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(winding_layout["sample_s"], winding_layout["mu_required"], color="darkmagenta", linewidth=2, label="Required friction")
    ax.axhline(friction_cfg.mu_max, color="0.30", linestyle="--", linewidth=1.1, label="Allowable friction")
    ax.set_xlabel("Meridional coordinate")
    ax.set_ylabel("Required friction coefficient")
    ax.set_title("Manufacturability constraint on the winding field")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _prepare_output_path(output_path)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
