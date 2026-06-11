from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from PIL import Image


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

from copv_opt.config import GeometryConfig
from copv_opt.visualize import (
    build_winding_process_layout_data,
    render_explicit_manufacturing_layout_image,
    render_vtu_scalar_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README hero assets for the winding-first COPV workflow.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Root outputs directory for the README media.",
    )
    parser.add_argument(
        "--winding-summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "winding_first_summary.json",
        help="Winding-first case summary JSON.",
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        help="Optimization snapshot JSON. Defaults to the winding-first artifact referenced in the summary.",
    )
    parser.add_argument(
        "--gif-summary",
        type=Path,
        default=None,
        help="Optional summary JSON to use only for the optimization GIF. Defaults to the packaged winding-first summary.",
    )
    parser.add_argument(
        "--gif-snapshots",
        type=Path,
        default=None,
        help="Optional snapshot JSON to use only for the optimization GIF. Defaults to the GIF summary artifact reference.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def as_array(value, dtype=np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def figure_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return Image.fromarray(buffer[:, :, :3])


def snapshot_to_winding_layout(snapshot: dict, geom: GeometryConfig) -> dict:
    result_like = {
        "winding_angle_ctrl": as_array(snapshot["winding_angle_ctrl"]),
        "winding_s_ctrl": as_array(snapshot["winding_s_ctrl"]),
        "winding_thickness_ctrl": as_array(snapshot["winding_thickness_ctrl"]),
    }
    return build_winding_process_layout_data(
        result_like,
        geom,
        family_count=8,
        sample_count=320,
    )


def filter_snapshots(snapshots: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    prev: dict | None = None
    for snapshot in snapshots:
        if prev is None:
            filtered.append(snapshot)
            prev = snapshot
            continue
        same_failure = abs(float(snapshot["fi_max_with_margin"]) - float(prev["fi_max_with_margin"])) < 1e-3
        same_thickness = abs(float(snapshot["max_winding_thickness"]) - float(prev["max_winding_thickness"])) < 1e-4
        same_angles = np.allclose(
            as_array(snapshot["winding_angle_ctrl"]),
            as_array(prev["winding_angle_ctrl"]),
            atol=1e-3,
        )
        if same_failure and same_thickness and same_angles:
            continue
        filtered.append(snapshot)
        prev = snapshot
    return filtered


def load_vtu_cell_scalar(vtu_path: Path, scalar_name: str) -> np.ndarray:
    mesh = meshio.read(str(vtu_path))
    scalar_blocks = mesh.cell_data_dict.get(scalar_name, {})
    for cell_block in mesh.cells:
        if cell_block.type in scalar_blocks:
            return np.asarray(scalar_blocks[cell_block.type], dtype=np.float64)
    raise KeyError(f"Scalar field '{scalar_name}' was not found in {vtu_path}")


def load_vtu_cell_centers(vtu_path: Path) -> np.ndarray:
    mesh = meshio.read(str(vtu_path))
    points = np.asarray(mesh.points, dtype=np.float64)
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            elems = np.asarray(cell_block.data, dtype=np.int32)
            return points[elems].mean(axis=1)
    raise KeyError(f"No tetrahedral cells were found in {vtu_path}")


def scalar_color(value: float, clim: tuple[float, float] | list[float], cmap_name: str) -> tuple[float, float, float]:
    lo, hi = float(clim[0]), float(clim[1])
    span = max(hi - lo, 1e-12)
    norm = np.clip((float(value) - lo) / span, 0.0, 1.0)
    rgba = matplotlib.colormaps.get_cmap(cmap_name)(norm)
    return tuple(float(channel) for channel in rgba[:3])


def build_safety_state_field(
    safety_margin: np.ndarray,
    fail_span: float = 0.12,
    safe_floor: float = 0.82,
) -> np.ndarray:
    """Map failure severity to red and all safe cells to green for the GIF story panel."""
    margin = as_array(safety_margin)
    span = max(float(fail_span), 1e-12)
    failing_state = np.clip(margin / span, -1.0, 0.0)
    safe_state = float(safe_floor) + (1.0 - float(safe_floor)) * np.clip(margin / span, 0.0, 1.0)
    return np.where(margin < 0.0, failing_state, safe_state)


def interpolate_values(start, end, t: float) -> np.ndarray:
    return (1.0 - float(t)) * as_array(start) + float(t) * as_array(end)


def trim_curve_points(points: np.ndarray, progress: float) -> np.ndarray | None:
    pts = as_array(points)
    if progress <= 0.0 or len(pts) < 2:
        return None
    if progress >= 1.0:
        return pts

    scaled_index = float(progress) * float(len(pts) - 1)
    lower = int(np.floor(scaled_index))
    upper = min(lower + 1, len(pts) - 1)
    frac = scaled_index - float(lower)

    head = pts[: lower + 1].copy()
    tail_point = (1.0 - frac) * pts[lower] + frac * pts[upper]
    if len(head) == 0:
        head = pts[:1].copy()
    if np.linalg.norm(head[-1] - tail_point) > 1e-9:
        head = np.vstack([head, tail_point])
    if len(head) < 2:
        head = np.vstack([pts[0], tail_point])
    return head


def reveal_curve_points(
    curve_points_list: list[np.ndarray],
    curve_colors: list[str],
    progress: float,
) -> tuple[list[np.ndarray], list[str]]:
    revealed_points: list[np.ndarray] = []
    revealed_colors: list[str] = []
    for points, color in zip(curve_points_list, curve_colors):
        partial = trim_curve_points(points, progress)
        if partial is None:
            continue
        revealed_points.append(partial)
        revealed_colors.append(color)
    return revealed_points, revealed_colors


def format_iteration_label(value: float) -> str:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-6:
        return str(int(rounded))
    return f"{float(value):.1f}"


def build_frame_states(
    snapshots: list[dict],
    baseline_failure_with_margin: np.ndarray,
    baseline_fi: float,
    baseline_hold_frames: int = 4,
    warmstart_frames: int = 6,
    interp_steps: int = 3,
) -> list[dict]:
    first_snapshot = snapshots[0]
    first_iteration = float(first_snapshot["global_iteration"])
    first_fi = float(first_snapshot["fi_max_with_margin"])
    first_thickness = float(first_snapshot["max_winding_thickness"])
    zero_thickness_ctrl = np.zeros_like(as_array(first_snapshot["winding_thickness_ctrl"]))
    actual_iterations = [float(snapshot["global_iteration"]) for snapshot in snapshots]
    actual_fi = [float(snapshot["fi_max_with_margin"]) for snapshot in snapshots]
    actual_thickness = [float(snapshot["max_winding_thickness"]) for snapshot in snapshots]
    max_iteration = max(actual_iterations) if actual_iterations else 0.0

    frame_states: list[dict] = []
    baseline_total = max(int(baseline_hold_frames), 1)
    for _ in range(baseline_total):
        frame_states.append(
            {
                "phase": "baseline",
                "beta": float(first_snapshot["beta"]),
                "objective": float(first_snapshot["objective"]),
                "global_iteration": first_iteration,
                "fi_max_with_margin": baseline_fi,
                "max_winding_thickness": 0.0,
                "winding_s_ctrl": as_array(first_snapshot["winding_s_ctrl"]),
                "winding_angle_ctrl": as_array(first_snapshot["winding_angle_ctrl"]),
                "winding_thickness_ctrl": zero_thickness_ctrl,
                "failure_with_margin": as_array(baseline_failure_with_margin),
                "path_progress": 0.0,
                "mandrel_opacity": 0.26,
                "tow_radius": 0.55,
                "history_iterations": [],
                "history_fi": [],
                "history_thickness": [],
                "max_iteration": max_iteration,
            }
        )

    warmstart_total = max(int(warmstart_frames), 1)
    for idx in range(warmstart_total):
        progress = (idx + 1) / float(warmstart_total)
        eased = 1.0 - (1.0 - progress) ** 2
        is_seed_frame = idx == warmstart_total - 1
        history_iterations = [first_iteration] if is_seed_frame else []
        history_fi = [first_fi] if is_seed_frame else []
        history_thickness = [first_thickness] if is_seed_frame else []
        frame_states.append(
            {
                "phase": "seed" if is_seed_frame else "warmstart",
                "beta": float(first_snapshot["beta"]),
                "objective": float(first_snapshot["objective"]),
                "global_iteration": first_iteration,
                "fi_max_with_margin": float((1.0 - eased) * baseline_fi + eased * first_fi),
                "max_winding_thickness": float(eased * first_thickness),
                "winding_s_ctrl": as_array(first_snapshot["winding_s_ctrl"]),
                "winding_angle_ctrl": as_array(first_snapshot["winding_angle_ctrl"]),
                "winding_thickness_ctrl": interpolate_values(zero_thickness_ctrl, first_snapshot["winding_thickness_ctrl"], eased),
                "failure_with_margin": interpolate_values(baseline_failure_with_margin, first_snapshot["failure_with_margin"], eased),
                "path_progress": float(eased),
                "mandrel_opacity": float(0.26 - 0.14 * eased),
                "tow_radius": float(0.55 + 0.60 * eased),
                "history_iterations": history_iterations,
                "history_fi": history_fi,
                "history_thickness": history_thickness,
                "max_iteration": max_iteration,
            }
        )

    for index, start_snapshot in enumerate(snapshots[:-1]):
        end_snapshot = snapshots[index + 1]
        start_iteration = float(start_snapshot["global_iteration"])
        end_iteration = float(end_snapshot["global_iteration"])
        total_progress_start = start_iteration / max(max_iteration, 1.0)
        for step in range(1, interp_steps + 1):
            t = step / float(interp_steps)
            current_iteration = (1.0 - t) * start_iteration + t * end_iteration
            current_fi = float((1.0 - t) * float(start_snapshot["fi_max_with_margin"]) + t * float(end_snapshot["fi_max_with_margin"]))
            current_thickness = float(
                (1.0 - t) * float(start_snapshot["max_winding_thickness"]) + t * float(end_snapshot["max_winding_thickness"])
            )
            history_iterations = actual_iterations[: index + 1].copy()
            history_fi = actual_fi[: index + 1].copy()
            history_thickness = actual_thickness[: index + 1].copy()
            if current_iteration > history_iterations[-1] + 1e-6:
                history_iterations.append(current_iteration)
                history_fi.append(current_fi)
                history_thickness.append(current_thickness)
            else:
                history_iterations = actual_iterations[: index + 2].copy()
                history_fi = actual_fi[: index + 2].copy()
                history_thickness = actual_thickness[: index + 2].copy()

            total_progress = total_progress_start + t * (end_iteration - start_iteration) / max(max_iteration, 1.0)
            frame_states.append(
                {
                    "phase": "optimization",
                    "beta": float((1.0 - t) * float(start_snapshot["beta"]) + t * float(end_snapshot["beta"])),
                    "objective": float((1.0 - t) * float(start_snapshot["objective"]) + t * float(end_snapshot["objective"])),
                    "global_iteration": current_iteration,
                    "fi_max_with_margin": current_fi,
                    "max_winding_thickness": current_thickness,
                    "winding_s_ctrl": interpolate_values(start_snapshot["winding_s_ctrl"], end_snapshot["winding_s_ctrl"], t),
                    "winding_angle_ctrl": interpolate_values(start_snapshot["winding_angle_ctrl"], end_snapshot["winding_angle_ctrl"], t),
                    "winding_thickness_ctrl": interpolate_values(
                        start_snapshot["winding_thickness_ctrl"],
                        end_snapshot["winding_thickness_ctrl"],
                        t,
                    ),
                    "failure_with_margin": interpolate_values(start_snapshot["failure_with_margin"], end_snapshot["failure_with_margin"], t),
                    "path_progress": 1.0,
                    "mandrel_opacity": float(0.12 - 0.05 * min(max(total_progress, 0.0), 1.0)),
                    "tow_radius": 1.15,
                    "history_iterations": history_iterations,
                    "history_fi": history_fi,
                    "history_thickness": history_thickness,
                    "max_iteration": max_iteration,
                }
            )
    return frame_states


def build_optimization_gif(
    snapshots_payload: dict,
    winding_summary: dict,
    output_path: Path,
    story_note: str | None = None,
) -> Path:
    snapshots = filter_snapshots(snapshots_payload["snapshots"])
    if not snapshots:
        raise ValueError("No optimization snapshots were found.")

    geom_data = snapshots_payload["geometry"]
    geom = GeometryConfig(
        pressure=float(geom_data["pressure"]),
        outer_radius=float(geom_data["outer_radius"]),
        cylinder_length=float(geom_data["cylinder_length"]),
        thickness=float(geom_data["thickness"]),
        opening_radius=float(geom_data["opening_radius"]),
    )
    base_vtu = resolve_repo_path(snapshots_payload["base_vtu"])

    baseline_vtu = resolve_repo_path(winding_summary["baseline"]["vtu"])
    baseline_failure_with_margin = load_vtu_cell_scalar(baseline_vtu, "failure_with_margin")
    cell_centers = load_vtu_cell_centers(baseline_vtu)
    baseline_fi = float(winding_summary["baseline"]["fi_max_with_margin"])
    all_fi = [float(snapshot["fi_max_with_margin"]) for snapshot in snapshots]
    all_thickness = [float(snapshot["max_winding_thickness"]) for snapshot in snapshots]
    fi_hi = max(1.15, baseline_fi, max(all_fi) * 1.06)
    thickness_hi = max(0.05, 1.10 * max(all_thickness))
    safety_state_clim = (-1.0, 1.0)
    critical_phi_deg = float(winding_summary["baseline"]["critical_surface_coordinate"]["phi_deg"])
    critical_phi_rad = np.radians(critical_phi_deg)
    radial = np.asarray([np.cos(critical_phi_rad), np.sin(critical_phi_rad), 0.0], dtype=np.float64)
    tangent = np.asarray([-np.sin(critical_phi_rad), np.cos(critical_phi_rad), 0.0], dtype=np.float64)
    safety_camera = [
        tuple((450.0 * radial + 90.0 * tangent + np.asarray([0.0, 0.0, 250.0], dtype=np.float64)).tolist()),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    safety_zoom = 0.72
    frame_states = build_frame_states(snapshots, baseline_failure_with_margin, baseline_fi)

    frames: list[Image.Image] = []
    for frame_state in frame_states:
        phase = str(frame_state["phase"])
        winding_layout = snapshot_to_winding_layout(frame_state, geom)
        winding_paths = [entry["points"] for entry in winding_layout["paths"]]
        winding_colors = [
            "forestgreen" if entry["handedness"] == "clockwise" else "darkmagenta"
            for entry in winding_layout["paths"]
        ]
        safety_curve_colors = [
            "#FFF7ED" if entry["handedness"] == "clockwise" else "#0F172A"
            for entry in winding_layout["paths"]
        ]
        layout_paths, layout_colors = reveal_curve_points(winding_paths, winding_colors, frame_state["path_progress"])
        safety_paths, safety_colors = reveal_curve_points(winding_paths, safety_curve_colors, frame_state["path_progress"])
        layout_img = render_explicit_manufacturing_layout_image(
            base_vtu,
            curve_points_list=layout_paths if layout_paths else None,
            curve_colors=layout_colors if layout_paths else None,
            tow_radius=float(frame_state["tow_radius"]),
            mandrel_opacity=float(frame_state["mandrel_opacity"]),
            title="Winding material buildup",
        )
        safety_margin = 1.0 - as_array(frame_state["failure_with_margin"])
        safety_state = build_safety_state_field(safety_margin)
        has_failure = bool(np.any(safety_margin < 0.0))
        critical_idx = int(np.argmin(safety_margin))
        failure_img = render_vtu_scalar_image(
            base_vtu,
            scalar_field="safety_state",
            scalar_values=safety_state,
            cmap="RdYlGn",
            clim=safety_state_clim,
            show_edges=False,
            slice_model=False,
            surface_only=True,
            mesh_opacity=0.40,
            highlight_threshold=0.0 if has_failure else None,
            highlight_below=True,
            highlight_opacity=0.96,
            marker_points=[cell_centers[critical_idx]] if has_failure else None,
            marker_colors=[scalar_color(-1.0, safety_state_clim, "RdYlGn")] if has_failure else None,
            marker_radius=11.5,
            curve_points_list=safety_paths if safety_paths else None,
            curve_colors=safety_colors if safety_paths else None,
            curve_radius=max(0.14, 0.24 * float(frame_state["path_progress"])),
            scalar_bar_title="Failing cells red / safe cells green",
            camera_position=safety_camera,
            camera_zoom=safety_zoom,
        )

        fig = plt.figure(figsize=(14.6, 8.8))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.65], width_ratios=[0.82, 1.18], hspace=0.20, wspace=0.05)

        ax_profile = fig.add_subplot(grid[0, :])
        history_iterations = [float(value) for value in frame_state["history_iterations"]]
        history_fi = [float(value) for value in frame_state["history_fi"]]
        history_thickness = [float(value) for value in frame_state["history_thickness"]]
        has_history = bool(history_iterations)
        ax_profile.set_xlim(0.0, max(float(frame_state["max_iteration"]), 1.0))
        ax_profile.set_ylim(0.0, fi_hi)
        ax_profile.set_xlabel("Global iteration")
        ax_profile.set_ylabel("Peak FI x margin", color="#13293D")
        ax_profile.tick_params(axis="y", labelcolor="#13293D")
        ax_profile.grid(alpha=0.25)
        ax_profile.axhline(1.0, color="#15803D", linestyle="--", linewidth=1.5, alpha=0.85)
        ax_profile.axhline(baseline_fi, color="#DC2626", linestyle=":", linewidth=1.3, alpha=0.9)
        if has_history:
            ax_profile.plot(history_iterations, history_fi, color="#13293D", linewidth=2.6)
            ax_profile.scatter(history_iterations[-1], history_fi[-1], color="#13293D", s=28, zorder=3)
        else:
            if phase == "baseline":
                profile_note = "True baseline solve: bare vessel only. This state fails and is not an optimization iteration."
            else:
                profile_note = "Pre-optimization warm-start buildup toward the seeded iteration 0 state."
            ax_profile.text(
                0.5,
                0.53,
                profile_note,
                transform=ax_profile.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#475569",
            )

        ax_thickness = ax_profile.twinx()
        ax_thickness.set_ylim(0.0, thickness_hi)
        ax_thickness.set_ylabel("Max added winding thickness", color="#3E92CC")
        ax_thickness.tick_params(axis="y", labelcolor="#3E92CC")
        if has_history:
            ax_thickness.fill_between(history_iterations, 0.0, history_thickness, color="#3E92CC", alpha=0.18)
            ax_thickness.plot(history_iterations, history_thickness, color="#3E92CC", linewidth=1.7)
            ax_thickness.scatter(history_iterations[-1], history_thickness[-1], color="#3E92CC", s=24, zorder=3)

        if phase == "baseline":
            ax_profile.set_title(
                f"Bare vessel baseline at pressure {geom.pressure:.2f} | Peak FI x margin {baseline_fi:.3f}",
                fontsize=14,
            )
        elif phase == "warmstart":
            ax_profile.set_title(
                f"Warm-start seed buildup before iteration 0 | Peak FI x margin {float(frame_state['fi_max_with_margin']):.3f}",
                fontsize=14,
            )
        elif phase == "seed":
            ax_profile.set_title(
                f"Seeded winding at iteration 0 | Peak FI x margin {float(frame_state['fi_max_with_margin']):.3f}",
                fontsize=14,
            )
        elif has_history:
            ax_profile.set_title(
                f"Iteration {format_iteration_label(frame_state['global_iteration'])} | beta {float(frame_state['beta']):.1f} | peak FI x margin {float(frame_state['fi_max_with_margin']):.3f}",
                fontsize=14,
            )
        else:
            ax_profile.set_title("Baseline vessel before optimization", fontsize=14)
        if phase == "baseline":
            profile_footer = f"Failing reference state. Added winding thickness 0.000   Safety threshold is FI x margin = 1.000"
        elif phase == "warmstart":
            profile_footer = (
                f"Illustrative transition to the warm-start seed. Added winding thickness {float(frame_state['max_winding_thickness']):.3f}"
            )
        elif phase == "seed":
            profile_footer = (
                f"Seed used to start L-BFGS. It is still unsafe here. Added winding thickness {float(frame_state['max_winding_thickness']):.3f}"
            )
        else:
            profile_footer = (
                f"Objective {float(frame_state['objective']):.4f}   Current max added thickness {float(frame_state['max_winding_thickness']):.3f}"
            )
        ax_profile.text(
            0.01,
            0.04,
            profile_footer,
            transform=ax_profile.transAxes,
            fontsize=10,
            color="#3F3F46",
        )

        ax_layout = fig.add_subplot(grid[1, 0])
        ax_layout.imshow(layout_img)
        if phase == "baseline":
            layout_title = "Bare vessel before winding"
        elif phase in {"warmstart", "seed"}:
            layout_title = "Warm-start seed buildup"
        else:
            layout_title = "Winding material buildup"
        ax_layout.set_title(layout_title, fontsize=12, fontweight="bold")
        ax_layout.axis("off")

        ax_failure = fig.add_subplot(grid[1, 1])
        ax_failure.imshow(failure_img)
        if phase == "baseline":
            failure_title = "Baseline safety state (failing cells in red)"
        elif phase == "warmstart":
            failure_title = "Transition toward the seeded state"
        elif phase == "seed":
            failure_title = "Seeded state safety field"
        elif has_failure:
            failure_title = "Whole-vessel safety field (remaining failing cells marked)"
        else:
            failure_title = "Whole-vessel safety field (all cells safe)"
        ax_failure.set_title(failure_title, fontsize=12, fontweight="bold")
        ax_failure.axis("off")

        if story_note:
            fig.text(0.012, 0.012, story_note, fontsize=8.5, color="#64748B")

        fig.patch.set_facecolor("white")
        frames.append(figure_to_image(fig))
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=180,
        loop=0,
        optimize=False,
    )
    return output_path


def build_winding_comparison_matrix(winding_summary: dict, output_path: Path) -> Path:
    baseline_vtu = resolve_repo_path(winding_summary["baseline"]["vtu"])
    winding_vtu = resolve_repo_path(winding_summary["winding"]["vtu"])
    winding_layout = load_json(resolve_repo_path(winding_summary["visualisations"]["winding_layout_json"]))

    failure_clim = (
        0.0,
        max(
            1.2,
            float(winding_summary["baseline"]["fi_max"]),
            float(winding_summary["winding"]["fi_max"]),
        ),
    )
    baseline_img = render_vtu_scalar_image(
        baseline_vtu,
        scalar_field="failure_index",
        cmap="RdYlGn_r",
        clim=failure_clim,
        surface_only=True,
        mesh_opacity=0.42,
        highlight_threshold=1.0,
        highlight_below=False,
        highlight_opacity=0.96,
    )
    winding_img = render_vtu_scalar_image(
        winding_vtu,
        scalar_field="failure_index",
        cmap="RdYlGn_r",
        clim=failure_clim,
        surface_only=True,
        mesh_opacity=0.42,
        highlight_threshold=1.0,
        highlight_below=False,
        highlight_opacity=0.96,
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))

    axes[0].imshow(baseline_img)
    axes[0].set_title("1. Baseline Failure", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    axes[0].text(
        0.5,
        -0.08,
        f"Peak FI {float(winding_summary['baseline']['fi_max']):.3f}",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    sample_s = as_array(winding_layout["sample_s"])
    angle_profile_deg = as_array(winding_layout["angle_profile_deg"])
    control_s = as_array(winding_layout["control_s"])
    control_angle_deg = as_array(winding_layout["control_angle_deg"])
    thickness_profile = as_array(winding_layout["thickness_profile"])
    ax_mid = axes[1]
    ax_mid.plot(sample_s, angle_profile_deg, color="#13293D", linewidth=2.6)
    ax_mid.scatter(control_s, control_angle_deg, color="#13293D", s=24, zorder=3)
    ax_mid.set_xlabel("Meridional coordinate")
    ax_mid.set_ylabel("Winding angle [deg]", color="#13293D")
    ax_mid.tick_params(axis="y", labelcolor="#13293D")
    ax_mid.grid(alpha=0.25)
    ax_mid.set_title("2. Optimized Winding Field", fontsize=12, fontweight="bold")
    ax_mid2 = ax_mid.twinx()
    ax_mid2.fill_between(sample_s, 0.0, thickness_profile, color="#3E92CC", alpha=0.18)
    ax_mid2.plot(sample_s, thickness_profile, color="#3E92CC", linewidth=1.7)
    ax_mid2.set_ylabel("Added thickness", color="#3E92CC")
    ax_mid2.tick_params(axis="y", labelcolor="#3E92CC")
    ax_mid.text(
        0.02,
        0.03,
        f"Mass {float(winding_summary['winding']['mass_delta_percent_vs_baseline']):+.2f}% vs baseline",
        transform=ax_mid.transAxes,
        fontsize=10,
        color="#3F3F46",
    )

    axes[2].imshow(winding_img)
    axes[2].set_title("3. Winding-First Success", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    axes[2].text(
        0.5,
        -0.08,
        f"Peak FI {float(winding_summary['winding']['fi_max']):.3f}",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    fig.suptitle("Winding-First Optimization Resolves The Failing COPV", fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs.resolve()
    winding_summary = load_json(args.winding_summary.resolve())

    gif_summary_path = args.gif_summary.resolve() if args.gif_summary is not None else args.winding_summary.resolve()
    gif_summary = load_json(gif_summary_path)

    snapshots_path = args.snapshots.resolve() if args.snapshots is not None else None
    if snapshots_path is None and gif_summary_path == args.winding_summary.resolve():
        visuals = winding_summary.get("visualisations", {})
        snapshot_ref = visuals.get("optimization_snapshots")
        if snapshot_ref is None:
            raise FileNotFoundError("Winding summary does not reference optimization snapshots. Regenerate the winding outputs first.")
        snapshots_path = resolve_repo_path(snapshot_ref)

    gif_snapshots_path = args.gif_snapshots.resolve() if args.gif_snapshots is not None else snapshots_path
    if gif_snapshots_path is None:
        visuals = gif_summary.get("visualisations", {})
        snapshot_ref = visuals.get("optimization_snapshots")
        if snapshot_ref is None:
            raise FileNotFoundError("GIF summary does not reference optimization snapshots. Regenerate the selected winding outputs first.")
        gif_snapshots_path = resolve_repo_path(snapshot_ref)
    snapshots_payload = load_json(gif_snapshots_path)

    story_note = None
    main_pressure = float(winding_summary["geometry"]["pressure"])
    gif_pressure = float(gif_summary["geometry"]["pressure"])
    notes: list[str] = []
    if abs(main_pressure - gif_pressure) > 1e-9:
        notes.append(
            f"Hero GIF uses a higher-pressure stress-test case at {gif_pressure:.2f}; the main verification summary remains at {main_pressure:.2f}."
        )
    main_cfg = winding_summary.get("winding_config", {})
    gif_cfg = gif_summary.get("winding_config", {})
    envelope_fields = (
        "mass_weight",
        "max_winding_thickness",
        "max_helical_pass_count",
        "max_hoop_pass_count",
    )
    if any(abs(float(main_cfg.get(field, 0.0)) - float(gif_cfg.get(field, 0.0))) > 1e-9 for field in envelope_fields):
        notes.append("Hero GIF also uses a looser winding design envelope than the packaged verification case.")
    if notes:
        story_note = " ".join(notes)

    gif_path = build_optimization_gif(
        snapshots_payload,
        gif_summary,
        outputs_dir / "optimization_evolution.gif",
        story_note=story_note,
    )
    matrix_path = build_winding_comparison_matrix(
        winding_summary,
        outputs_dir / "winding_comparison_matrix.png",
    )

    print(f"Wrote {gif_path}")
    print(f"Wrote {matrix_path}")


if __name__ == "__main__":
    main()
