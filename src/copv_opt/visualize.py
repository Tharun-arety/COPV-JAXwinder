from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import meshio
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .config import GeometryConfig, IFPConfig, PatchConfig, WindingConfig
from .geometry import copv_meridional_metrics, copv_surface_from_sphi_np, project_to_copv_surface


def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("PyVista is required for interactive VTU viewing.") from exc
    return pv


def _prepare_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)


def _build_glass_mandrel(plotter, base_vtu_path: str | Path, color: str = "lightgray", opacity: float = 0.18):
    pv = _import_pyvista()
    base_mesh = pv.read(str(base_vtu_path))
    surface = base_mesh.extract_surface().triangulate()
    plotter.add_mesh(
        surface,
        color=color,
        opacity=opacity,
        show_edges=False,
        smooth_shading=True,
        specular=0.35,
        specular_power=20.0,
        ambient=0.15,
    )
    return surface


def _build_explicit_layout_plotter(
    base_vtu_path: str | Path,
    curve_points_list: list[np.ndarray] | None = None,
    patch_polygons: list[np.ndarray] | None = None,
    curve_color: str = "crimson",
    curve_colors: list[str] | None = None,
    tow_radius: float = 1.35,
    patch_color: str = "steelblue",
    patch_edge_color: str = "midnightblue",
    patch_opacity: float = 0.90,
    mandrel_opacity: float = 0.18,
    title: str = "Explicit Manufacturing Layout",
    off_screen: bool = False,
    window_size: tuple[int, int] = (1600, 1200),
):
    pv = _import_pyvista()
    plotter = pv.Plotter(title=title, off_screen=off_screen, window_size=window_size)
    plotter.set_background("white")
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass

    _build_glass_mandrel(plotter, base_vtu_path, opacity=mandrel_opacity)

    if curve_points_list is not None:
        for idx, points in enumerate(curve_points_list):
            pts = np.asarray(points, dtype=np.float64)
            if len(pts) < 2:
                continue
            color = curve_colors[idx] if curve_colors is not None and idx < len(curve_colors) else curve_color
            if len(pts) == 2:
                spline = pv.Line(pts[0], pts[1])
            else:
                spline = pv.Spline(pts, n_points=max(len(pts) * 4, len(pts)))
            tow = spline.tube(radius=float(tow_radius), capping=True)
            plotter.add_mesh(
                tow,
                color=color,
                smooth_shading=True,
                specular=0.30,
                specular_power=18.0,
            )

    if patch_polygons is not None:
        for polygon_points in patch_polygons:
            corners = np.asarray(polygon_points, dtype=np.float64)
            if len(corners) < 3:
                continue
            center = corners.mean(axis=0, keepdims=True)
            points = np.vstack([corners, center])
            if len(corners) == 4:
                faces = np.array(
                    [
                        3, 0, 1, 4,
                        3, 1, 2, 4,
                        3, 2, 3, 4,
                        3, 3, 0, 4,
                    ],
                    dtype=np.int64,
                )
            else:
                faces_list: list[int] = []
                center_idx = len(corners)
                for i in range(len(corners)):
                    faces_list.extend([3, i, (i + 1) % len(corners), center_idx])
                faces = np.asarray(faces_list, dtype=np.int64)
            patch_surface = pv.PolyData(points, faces=faces)
            plotter.add_mesh(
                patch_surface,
                color=patch_color,
                opacity=patch_opacity,
                smooth_shading=True,
                show_edges=False,
                specular=0.22,
                specular_power=12.0,
            )
            border = pv.MultipleLines(np.vstack([corners, corners[0]]))
            plotter.add_mesh(border, color=patch_edge_color, line_width=3.0)

    plotter.add_axes()
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.25)
    return plotter


def render_explicit_manufacturing_layout(
    base_vtu_path: str | Path,
    curve_points_list: list[np.ndarray] | None = None,
    patch_polygons: list[np.ndarray] | None = None,
    curve_color: str = "crimson",
    curve_colors: list[str] | None = None,
    tow_radius: float = 1.35,
    patch_color: str = "steelblue",
    patch_edge_color: str = "midnightblue",
    patch_opacity: float = 0.90,
    mandrel_opacity: float = 0.18,
    title: str = "Explicit Manufacturing Layout",
):
    """
    Renders explicit geometric curves and patches over a glass-like COPV mandrel.
    """
    plotter = _build_explicit_layout_plotter(
        base_vtu_path=base_vtu_path,
        curve_points_list=curve_points_list,
        patch_polygons=patch_polygons,
        curve_color=curve_color,
        curve_colors=curve_colors,
        tow_radius=tow_radius,
        patch_color=patch_color,
        patch_edge_color=patch_edge_color,
        patch_opacity=patch_opacity,
        mandrel_opacity=mandrel_opacity,
        title=title,
        off_screen=False,
    )
    plotter.show()


def save_explicit_manufacturing_layout_screenshot(
    base_vtu_path: str | Path,
    screenshot_path: str | Path,
    curve_points_list: list[np.ndarray] | None = None,
    patch_polygons: list[np.ndarray] | None = None,
    curve_color: str = "crimson",
    curve_colors: list[str] | None = None,
    tow_radius: float = 1.35,
    patch_color: str = "steelblue",
    patch_edge_color: str = "midnightblue",
    patch_opacity: float = 0.90,
    mandrel_opacity: float = 0.18,
    title: str = "Explicit Manufacturing Layout",
    window_size: tuple[int, int] = (1600, 1200),
) -> Path:
    """
    Saves an off-screen explicit manufacturing layout screenshot over a glass-like COPV.
    """
    pv = _import_pyvista()
    pv.OFF_SCREEN = True
    screenshot_path = Path(screenshot_path)
    plotter = _build_explicit_layout_plotter(
        base_vtu_path=base_vtu_path,
        curve_points_list=curve_points_list,
        patch_polygons=patch_polygons,
        curve_color=curve_color,
        curve_colors=curve_colors,
        tow_radius=tow_radius,
        patch_color=patch_color,
        patch_edge_color=patch_edge_color,
        patch_opacity=patch_opacity,
        mandrel_opacity=mandrel_opacity,
        title=title,
        off_screen=True,
        window_size=window_size,
    )
    _prepare_output_path(screenshot_path)
    plotter.show(screenshot=str(screenshot_path), auto_close=True)
    return screenshot_path


def render_explicit_manufacturing_layout_image(
    base_vtu_path: str | Path,
    curve_points_list: list[np.ndarray] | None = None,
    patch_polygons: list[np.ndarray] | None = None,
    curve_color: str = "crimson",
    curve_colors: list[str] | None = None,
    tow_radius: float = 1.35,
    patch_color: str = "steelblue",
    patch_edge_color: str = "midnightblue",
    patch_opacity: float = 0.90,
    mandrel_opacity: float = 0.18,
    title: str = "Explicit Manufacturing Layout",
    window_size: tuple[int, int] = (1600, 1200),
) -> np.ndarray:
    pv = _import_pyvista()
    pv.OFF_SCREEN = True
    plotter = _build_explicit_layout_plotter(
        base_vtu_path=base_vtu_path,
        curve_points_list=curve_points_list,
        patch_polygons=patch_polygons,
        curve_color=curve_color,
        curve_colors=curve_colors,
        tow_radius=tow_radius,
        patch_color=patch_color,
        patch_edge_color=patch_edge_color,
        patch_opacity=patch_opacity,
        mandrel_opacity=mandrel_opacity,
        title=title,
        off_screen=True,
        window_size=window_size,
    )
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return np.asarray(image)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def save_layout_json(path: str | Path, layout: dict[str, Any]) -> Path:
    path = Path(path)
    _prepare_output_path(path)
    path.write_text(json.dumps(_to_serializable(layout), indent=2), encoding="utf-8")
    return path


def load_layout_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_vtu_interactive(
    vtu_path: str | Path,
    scalar_field: str = "thickness",
    show_edges: bool = False,
    slice_model: bool = False,
):
    """
    Renders a 3D interactive view of the COPV from a VTU file.

    Args:
        vtu_path: Path to the generated .vtu file.
        scalar_field: The data array to map to colors.
        show_edges: Whether to draw the shell/volume cell edges.
        slice_model: If True, adds an interactive clipping plane.
    """
    pv = _import_pyvista()
    vtu_path = Path(vtu_path)
    if not vtu_path.exists():
        raise FileNotFoundError(f"Cannot find VTU file at {vtu_path}")

    mesh = pv.read(str(vtu_path))
    available_scalars = mesh.array_names
    if scalar_field not in available_scalars:
        print(f"Warning: '{scalar_field}' not found. Available fields: {available_scalars}")
        scalar_field = available_scalars[0] if available_scalars else None

    plotter = pv.Plotter(title=f"COPV Optimization - {vtu_path.name}")
    plotter.set_background("white")
    actor = plotter.add_mesh(
        mesh,
        scalars=scalar_field,
        cmap="plasma",
        show_edges=show_edges,
        edge_color="black",
        line_width=0.5,
        lighting=True,
        smooth_shading=True,
        clim=mesh.get_data_range(scalar_field) if scalar_field else None,
    )

    if slice_model:
        plotter.add_mesh_clip_plane(
            mesh,
            assign_to_axis="y",
            invert=False,
            scalars=scalar_field,
            cmap="plasma",
            show_edges=show_edges,
        )
        plotter.remove_actor(actor)

    if scalar_field:
        plotter.add_scalar_bar(
            title=scalar_field.capitalize().replace("_", " "),
            title_font_size=16,
            label_font_size=14,
            color="black",
            vertical=False,
            position_y=0.05,
        )
    plotter.add_axes()
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.2)
    print(f"Launching PyVista viewer for {scalar_field or 'mesh'}... close the window to continue.")
    plotter.show()


def save_vtu_screenshot(
    vtu_path: str | Path,
    screenshot_path: str | Path,
    scalar_field: str = "thickness",
    show_edges: bool = False,
    slice_model: bool = False,
    window_size: tuple[int, int] = (1600, 1200),
) -> Path:
    """
    Saves an off-screen PyVista screenshot for a VTU file.
    """
    pv = _import_pyvista()
    pv.OFF_SCREEN = True

    vtu_path = Path(vtu_path)
    screenshot_path = Path(screenshot_path)
    if not vtu_path.exists():
        raise FileNotFoundError(f"Cannot find VTU file at {vtu_path}")

    mesh = pv.read(str(vtu_path))
    available_scalars = mesh.array_names
    if scalar_field not in available_scalars:
        print(f"Warning: '{scalar_field}' not found. Available fields: {available_scalars}")
        scalar_field = available_scalars[0] if available_scalars else None

    plotter = pv.Plotter(off_screen=True, window_size=window_size, title=f"COPV Screenshot - {vtu_path.name}")
    plotter.set_background("white")

    if slice_model:
        try:
            mesh_to_show = mesh.clip(normal="y", origin=mesh.center)
        except Exception:
            mesh_to_show = mesh
    else:
        mesh_to_show = mesh

    plotter.add_mesh(
        mesh_to_show,
        scalars=scalar_field,
        cmap="plasma",
        show_edges=show_edges,
        edge_color="black",
        line_width=0.5,
        lighting=True,
        smooth_shading=True,
        clim=mesh_to_show.get_data_range(scalar_field) if scalar_field else None,
    )
    if scalar_field:
        plotter.add_scalar_bar(
            title=scalar_field.capitalize().replace("_", " "),
            title_font_size=16,
            label_font_size=14,
            color="black",
            vertical=False,
            position_y=0.05,
        )
    plotter.add_axes()
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.2)

    _prepare_output_path(screenshot_path)
    plotter.show(screenshot=str(screenshot_path), auto_close=True)
    return screenshot_path


def render_vtu_scalar_image(
    vtu_path: str | Path,
    scalar_field: str = "thickness",
    scalar_values: np.ndarray | None = None,
    cmap: str = "plasma",
    clim: tuple[float, float] | list[float] | None = None,
    show_edges: bool = False,
    slice_model: bool = False,
    clip_normal: str | tuple | list = "y",
    clip_origin: tuple[float, float, float] | list[float] | None = None,
    surface_only: bool = True,
    curve_points_list: list[np.ndarray] | None = None,
    curve_color: str = "white",
    curve_colors: list[str] | None = None,
    curve_radius: float = 1.15,
    scalar_bar_title: str | None = None,
    camera_position: str | tuple | list | None = None,
    camera_zoom: float = 1.15,
    window_size: tuple[int, int] = (1600, 1200),
    mesh_opacity: float = 1.0,
    highlight_threshold: float | None = None,
    highlight_below: bool = False,
    highlight_opacity: float = 0.95,
    marker_points: list[np.ndarray] | None = None,
    marker_colors: list[str] | None = None,
    marker_radius: float = 6.0,
) -> np.ndarray:
    pv = _import_pyvista()
    pv.OFF_SCREEN = True

    vtu_path = Path(vtu_path)
    if not vtu_path.exists():
        raise FileNotFoundError(f"Cannot find VTU file at {vtu_path}")

    mesh = pv.read(str(vtu_path))
    if scalar_values is not None:
        mesh.cell_data[scalar_field] = np.asarray(scalar_values, dtype=np.float64)

    available_scalars = mesh.array_names
    if scalar_field not in available_scalars:
        raise ValueError(f"Scalar field '{scalar_field}' not found in {vtu_path}")

    mesh_to_show = mesh
    if slice_model:
        try:
            mesh_to_show = mesh.clip(normal=clip_normal, origin=mesh.center if clip_origin is None else clip_origin)
        except Exception:
            mesh_to_show = mesh

    highlight_source = mesh_to_show
    if surface_only:
        try:
            surface = mesh_to_show.extract_surface().triangulate()
            if scalar_field in surface.array_names:
                mesh_to_show = surface
        except Exception:
            pass

    plotter = pv.Plotter(off_screen=True, window_size=window_size, title=f"COPV Render - {vtu_path.name}")
    plotter.set_background("white")
    plotter.add_mesh(
        mesh_to_show,
        scalars=scalar_field,
        cmap=cmap,
        show_edges=show_edges,
        edge_color="black",
        line_width=0.4,
        lighting=True,
        smooth_shading=True,
        clim=clim if clim is not None else mesh_to_show.get_data_range(scalar_field),
        opacity=float(mesh_opacity),
    )
    if highlight_threshold is not None:
        try:
            highlight_mesh = highlight_source.threshold(
                value=float(highlight_threshold),
                scalars=scalar_field,
                preference="cell",
                invert=bool(highlight_below),
            )
            if highlight_mesh.n_cells > 0:
                plotter.add_mesh(
                    highlight_mesh,
                    scalars=scalar_field,
                    cmap=cmap,
                    show_edges=False,
                    lighting=True,
                    smooth_shading=True,
                    clim=clim if clim is not None else highlight_source.get_data_range(scalar_field),
                    opacity=float(highlight_opacity),
                )
        except Exception:
            pass
    if curve_points_list is not None:
        for idx, points in enumerate(curve_points_list):
            pts = np.asarray(points, dtype=np.float64)
            if len(pts) < 2:
                continue
            color = curve_colors[idx] if curve_colors is not None and idx < len(curve_colors) else curve_color
            if len(pts) == 2:
                spline = pv.Line(pts[0], pts[1])
            else:
                spline = pv.Spline(pts, n_points=max(len(pts) * 4, len(pts)))
            tow = spline.tube(radius=float(curve_radius), capping=True)
            plotter.add_mesh(
                tow,
                color=color,
                smooth_shading=True,
                specular=0.35,
                specular_power=18.0,
            )
    if marker_points is not None:
        for idx, point in enumerate(marker_points):
            center = np.asarray(point, dtype=np.float64).reshape(3)
            color = marker_colors[idx] if marker_colors is not None and idx < len(marker_colors) else "crimson"
            marker = pv.Sphere(radius=float(marker_radius), center=center, theta_resolution=32, phi_resolution=32)
            plotter.add_mesh(
                marker,
                color=color,
                smooth_shading=True,
                specular=0.45,
                specular_power=24.0,
            )
    plotter.add_scalar_bar(
        title=scalar_field.capitalize().replace("_", " ") if scalar_bar_title is None else scalar_bar_title,
        title_font_size=16,
        label_font_size=13,
        color="black",
        vertical=False,
        position_y=0.05,
    )
    plotter.add_axes()
    plotter.camera_position = "iso" if camera_position is None else camera_position
    if camera_zoom != 1.0:
        plotter.camera.zoom(camera_zoom)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return np.asarray(image)


def compare_vtu_side_by_side(vtu_path_1: str | Path, vtu_path_2: str | Path, scalar_field: str = "displacement_norm"):
    """
    Renders two VTU files side-by-side with synchronized cameras.
    Great for comparing baseline vs optimized.
    """
    pv = _import_pyvista()
    mesh1 = pv.read(str(vtu_path_1))
    mesh2 = pv.read(str(vtu_path_2))

    plotter = pv.Plotter(shape=(1, 2), title="COPV Comparison")
    plotter.set_background("white")

    plotter.subplot(0, 0)
    plotter.add_text("Baseline", color="black", font_size=12)
    plotter.add_mesh(mesh1, scalars=scalar_field, cmap="viridis", show_edges=True)
    plotter.add_axes()

    plotter.subplot(0, 1)
    plotter.add_text("Optimized", color="black", font_size=12)
    plotter.add_mesh(mesh2, scalars=scalar_field, cmap="viridis", show_edges=True)
    plotter.add_axes()

    plotter.link_views()
    plotter.camera_position = "iso"
    plotter.show()


def save_vtu_comparison_screenshot(
    vtu_path_1: str | Path,
    vtu_path_2: str | Path,
    screenshot_path: str | Path,
    scalar_field: str = "displacement_norm",
    window_size: tuple[int, int] = (1800, 900),
) -> Path:
    """
    Saves an off-screen side-by-side comparison screenshot for two VTU files.
    """
    pv = _import_pyvista()
    pv.OFF_SCREEN = True

    mesh1 = pv.read(str(vtu_path_1))
    mesh2 = pv.read(str(vtu_path_2))
    screenshot_path = Path(screenshot_path)

    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=window_size, title="COPV Comparison")
    plotter.set_background("white")

    plotter.subplot(0, 0)
    plotter.add_text("Baseline", color="black", font_size=12)
    plotter.add_mesh(mesh1, scalars=scalar_field, cmap="viridis", show_edges=True)
    plotter.add_axes()

    plotter.subplot(0, 1)
    plotter.add_text("Optimized", color="black", font_size=12)
    plotter.add_mesh(mesh2, scalars=scalar_field, cmap="viridis", show_edges=True)
    plotter.add_axes()

    plotter.link_views()
    plotter.camera_position = "iso"
    _prepare_output_path(screenshot_path)
    plotter.show(screenshot=str(screenshot_path), auto_close=True)
    return screenshot_path


def set_copv_axes(ax, outer_radius: float, cylinder_length: float) -> None:
    radial_span = 1.08 * outer_radius
    axial_span = 1.04 * (0.5 * cylinder_length + outer_radius)
    ax.set_xlim(-radial_span, radial_span)
    ax.set_ylim(-radial_span, radial_span)
    ax.set_zlim(-axial_span, axial_span)
    try:
        ax.set_box_aspect((2.0 * radial_span, 2.0 * radial_span, 2.0 * axial_span))
    except AttributeError:
        pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_copv_surface(ax, geom: GeometryConfig, alpha: float = 0.12) -> None:
    _, _, total_len = copv_meridional_metrics(
        geom.outer_radius,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    s = np.linspace(0.0, total_len, 84)
    phi = np.linspace(0.0, 2.0 * np.pi, 96)
    s_grid, phi_grid = np.meshgrid(s, phi, indexing="ij")
    surf = copv_surface_from_sphi_np(
        geom.outer_radius,
        s_grid.reshape(-1),
        phi_grid.reshape(-1),
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    points = surf["points"].reshape(s_grid.shape + (3,))
    ax.plot_surface(
        points[..., 0],
        points[..., 1],
        points[..., 2],
        color="aliceblue",
        alpha=alpha,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )

    set_copv_axes(ax, geom.outer_radius, geom.cylinder_length)


def show_copv_mesh(
    nodes: np.ndarray,
    outer_faces: np.ndarray,
    geom: GeometryConfig,
    title: str,
    save_path: Path | None = None,
):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(
            nodes[outer_faces],
            alpha=0.10,
            edgecolor="steelblue",
            linewidth=0.12,
            facecolor="aliceblue",
        )
    )
    sc = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=nodes[:, 2], cmap="viridis", s=4, alpha=0.8)
    fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.05, label="z")
    ax.set_title(title)
    set_copv_axes(ax, geom.outer_radius, geom.cylinder_length)
    ax.view_init(20, 36)
    fig.tight_layout()
    if save_path is not None:
        _prepare_output_path(save_path)
        fig.savefig(save_path, dpi=115)
    return fig


def extract_patch_polygons(
    result: dict[str, Any],
    config: PatchConfig,
    geom: GeometryConfig,
    surface_radius: float | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    skin_radius = geom.outer_radius if surface_radius is None else surface_radius
    s_coords = np.asarray(result["s_coords"])
    phis = np.asarray(result["phis"])
    alphas = np.asarray(result["alphas"])
    surf = copv_surface_from_sphi_np(
        skin_radius,
        s_coords,
        phis,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    centers = surf["points"]
    e_s = surf["meridian_dirs"]
    e_phi = surf["hoop_dirs"]
    normals = surf["normals"]
    ca = np.cos(alphas)
    sa = np.sin(alphas)
    fiber_dirs = ca[:, None] * e_s + sa[:, None] * e_phi
    perp_dirs = -sa[:, None] * e_s + ca[:, None] * e_phi
    local_corners = np.array(
        [
            [-0.5 * config.length, -0.5 * config.width],
            [0.5 * config.length, -0.5 * config.width],
            [0.5 * config.length, 0.5 * config.width],
            [-0.5 * config.length, 0.5 * config.width],
        ],
        dtype=np.float64,
    )
    polygons: list[np.ndarray] = []
    for center, fiber_dir, perp_dir in zip(centers, fiber_dirs, perp_dirs):
        trial = center + local_corners[:, :1] * fiber_dir[None, :] + local_corners[:, 1:] * perp_dir[None, :]
        polygons.append(
            project_to_copv_surface(
                trial,
                skin_radius,
                geom.cylinder_length,
                geom.opening_radius,
                geom.dome_height_ratio,
            )
        )
    return polygons, centers, fiber_dirs, normals


def build_patch_layout_data(
    result: dict[str, Any],
    config: PatchConfig,
    geom: GeometryConfig,
    lift: float = 0.85,
) -> dict[str, Any]:
    polygons, centers, fiber_dirs, normals = extract_patch_polygons(
        result,
        config,
        geom,
        surface_radius=geom.outer_radius + lift,
    )
    patches = []
    for corners, center, fiber_dir, normal in zip(polygons, centers, fiber_dirs, normals):
        patches.append(
            {
                "corners": np.asarray(corners, dtype=np.float64),
                "center": np.asarray(center, dtype=np.float64),
                "fiber_dir": np.asarray(fiber_dir, dtype=np.float64),
                "normal": np.asarray(normal, dtype=np.float64),
            }
        )
    return {
        "layout_type": "patch",
        "lift": float(lift),
        "surface_radius": float(geom.outer_radius + lift),
        "patches": patches,
    }


def build_ifp_layout_data(
    result: dict[str, Any],
    config: IFPConfig,
    geom: GeometryConfig,
    lift: float = 0.85,
) -> dict[str, Any]:
    curve_s = np.asarray(result["curve_s"])
    curve_phi = np.asarray(result["curve_phi"])
    ctrl_s = np.asarray(result["ctrl_s"])
    ctrl_phi = np.asarray(result["ctrl_phi"])
    offsets = np.linspace(0.0, 2.0 * np.pi, config.family_count, endpoint=False)
    surface_radius = geom.outer_radius + lift

    curves = []
    for offset in offsets:
        curve = copv_surface_from_sphi_np(
            surface_radius,
            curve_s,
            np.mod(curve_phi + offset, 2.0 * np.pi),
            geom.cylinder_length,
            geom.opening_radius,
            geom.dome_height_ratio,
        )["points"]
        curves.append(np.asarray(curve, dtype=np.float64))

    control_curve = copv_surface_from_sphi_np(
        surface_radius,
        ctrl_s,
        ctrl_phi,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )["points"]
    return {
        "layout_type": "ifp",
        "lift": float(lift),
        "surface_radius": float(surface_radius),
        "tow_width": float(config.tow_width),
        "curves": curves,
        "control_curve": np.asarray(control_curve, dtype=np.float64),
    }


def plot_patch_projection(
    result: dict[str, Any],
    config: PatchConfig,
    geom: GeometryConfig,
    save_path: Path | None = None,
):
    polygons, centers, fiber_dirs, _ = extract_patch_polygons(result, config, geom)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_copv_surface(ax, geom, alpha=0.11)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(1, len(polygons))))
    collection = Poly3DCollection(polygons, facecolors=colors[: len(polygons)], edgecolors="black", linewidths=0.75, alpha=0.55)
    ax.add_collection3d(collection)
    for center, fiber_dir, color in zip(centers, fiber_dirs, colors):
        guide = np.vstack([center, center + 18.0 * fiber_dir])
        ax.plot(guide[:, 0], guide[:, 1], guide[:, 2], color=color, linewidth=1.7)
    ax.set_title("Patch footprints over the full COPV")
    ax.view_init(20, 36)
    fig.tight_layout()
    if save_path is not None:
        _prepare_output_path(save_path)
        fig.savefig(save_path, dpi=115)
    return fig


def plot_ifp_projection(
    result: dict[str, Any],
    config: IFPConfig,
    geom: GeometryConfig,
    save_path: Path | None = None,
):
    curve_s = np.asarray(result["curve_s"])
    curve_phi = np.asarray(result["curve_phi"])
    ctrl_s = np.asarray(result["ctrl_s"])
    ctrl_phi = np.asarray(result["ctrl_phi"])
    offsets = np.linspace(0.0, 2.0 * np.pi, config.family_count, endpoint=False)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_copv_surface(ax, geom, alpha=0.11)
    for idx, offset in enumerate(offsets):
        curve = copv_surface_from_sphi_np(
            geom.outer_radius,
            curve_s,
            np.mod(curve_phi + offset, 2.0 * np.pi),
            geom.cylinder_length,
            geom.opening_radius,
            geom.dome_height_ratio,
        )["points"]
        color = plt.cm.plasma(idx / max(config.family_count - 1, 1))
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], linewidth=2.0, alpha=0.88, color=color)
        ax.scatter(curve[::4, 0], curve[::4, 1], curve[::4, 2], s=12, color="black", alpha=0.65, depthshade=False)

    ctrl_curve = copv_surface_from_sphi_np(
        geom.outer_radius,
        ctrl_s,
        ctrl_phi,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )["points"]
    ax.plot(ctrl_curve[:, 0], ctrl_curve[:, 1], ctrl_curve[:, 2], linestyle="--", color="navy", linewidth=1.5)
    ax.scatter(ctrl_curve[:, 0], ctrl_curve[:, 1], ctrl_curve[:, 2], color="navy", s=36, depthshade=False)
    ax.set_title("IFP curve family over the full COPV")
    ax.view_init(20, 36)
    fig.tight_layout()
    if save_path is not None:
        _prepare_output_path(save_path)
        fig.savefig(save_path, dpi=115)
    return fig


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=np.float64)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    return out


def piecewise_linear_basis_np(targets: np.ndarray, control_points: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    control_points = np.asarray(control_points, dtype=np.float64).reshape(-1)
    if len(control_points) == 0:
        raise ValueError("control_points must not be empty")
    if len(control_points) == 1:
        return np.ones((len(targets), 1), dtype=np.float64)

    basis = np.zeros((len(targets), len(control_points)), dtype=np.float64)
    hi = np.searchsorted(control_points, targets, side="right")
    hi = np.clip(hi, 1, len(control_points) - 1)
    lo = hi - 1

    x0 = control_points[lo]
    x1 = control_points[hi]
    span = np.clip(x1 - x0, 1e-12, None)
    frac = np.clip((targets - x0) / span, 0.0, 1.0)
    basis[np.arange(len(targets)), lo] = 1.0 - frac
    basis[np.arange(len(targets)), hi] += frac
    basis[targets <= control_points[0], 0] = 1.0
    basis[targets >= control_points[-1], -1] = 1.0
    return basis


def build_constant_angle_winding_paths(
    angle_deg: float,
    geom: GeometryConfig,
    family_count: int = 8,
    sample_count: int = 260,
    surface_radius: float | None = None,
) -> dict[str, Any]:
    radius = geom.outer_radius if surface_radius is None else surface_radius
    alpha_cyl = np.radians(angle_deg)
    _, _, total_len = copv_meridional_metrics(
        radius,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    s = np.linspace(0.0, total_len, sample_count)
    surf = copv_surface_from_sphi_np(
        radius,
        s,
        np.zeros_like(s),
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    rho = surf["rho"].clip(min=geom.opening_radius + 4.0)
    clairaut_radius = radius * np.sin(alpha_cyl)
    alpha_profile = np.arcsin(np.clip(clairaut_radius / rho, -0.98, 0.98))
    dphi_ds = np.tan(alpha_profile) / rho
    base_phi = cumulative_trapezoid(dphi_ds, s)

    paths: list[tuple[str, np.ndarray]] = []
    family_half = max(family_count // 2, 1)
    for handedness, sign in (("clockwise", 1.0), ("counter_clockwise", -1.0)):
        for k in range(family_half):
            phi = np.mod(sign * base_phi + 2.0 * np.pi * k / family_half, 2.0 * np.pi)
            curve = copv_surface_from_sphi_np(
                radius,
                s,
                phi,
                geom.cylinder_length,
                geom.opening_radius,
                geom.dome_height_ratio,
            )["points"]
            paths.append((handedness, curve))
    return {
        "paths": paths,
        "clairaut_radius": clairaut_radius,
        "family_count": len(paths),
        "surface_radius": radius,
    }


def build_variable_angle_winding_paths(
    angle_ctrl: np.ndarray,
    control_s: np.ndarray,
    geom: GeometryConfig,
    family_count: int = 8,
    sample_count: int = 260,
    surface_radius: float | None = None,
    thickness_ctrl: np.ndarray | None = None,
) -> dict[str, Any]:
    radius = geom.outer_radius if surface_radius is None else surface_radius
    _, _, total_len = copv_meridional_metrics(
        radius,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    sample_s = np.linspace(0.0, total_len, sample_count)
    basis = piecewise_linear_basis_np(sample_s, control_s)
    angle_ctrl = np.asarray(angle_ctrl, dtype=np.float64).reshape(-1)
    angle_profile = basis @ angle_ctrl
    thickness_profile = None
    if thickness_ctrl is not None:
        thickness_profile = basis @ np.asarray(thickness_ctrl, dtype=np.float64).reshape(-1)

    surf = copv_surface_from_sphi_np(
        radius,
        sample_s,
        np.zeros_like(sample_s),
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    rho = np.asarray(surf["rho"], dtype=np.float64).clip(min=max(geom.opening_radius + 4.0, 1e-6))
    dphi_ds = np.tan(angle_profile) / rho
    base_phi = cumulative_trapezoid(dphi_ds, sample_s)

    mu_required = np.zeros_like(sample_s, dtype=np.float64)
    if len(sample_s) > 1:
        clairaut = rho * np.sin(angle_profile)
        ds = np.clip(np.diff(sample_s), 1e-12, None)
        rho_mid = 0.5 * (rho[1:] + rho[:-1])
        alpha_mid = 0.5 * (angle_profile[1:] + angle_profile[:-1])
        mu_seg = np.abs(np.diff(clairaut) / ds) / np.clip(np.abs(rho_mid * np.cos(alpha_mid)), 1e-6, None)
        mu_required[0] = mu_seg[0]
        mu_required[1:] = mu_seg

    paths: list[tuple[str, np.ndarray]] = []
    family_half = max(family_count // 2, 1)
    for handedness, sign in (("clockwise", 1.0), ("counter_clockwise", -1.0)):
        for k in range(family_half):
            phi = np.mod(sign * base_phi + 2.0 * np.pi * k / family_half, 2.0 * np.pi)
            curve = copv_surface_from_sphi_np(
                radius,
                sample_s,
                phi,
                geom.cylinder_length,
                geom.opening_radius,
                geom.dome_height_ratio,
            )["points"]
            paths.append((handedness, curve))

    return {
        "paths": paths,
        "family_count": len(paths),
        "surface_radius": radius,
        "sample_s": sample_s,
        "rho": rho,
        "control_s": np.asarray(control_s, dtype=np.float64),
        "control_angle_rad": angle_ctrl,
        "control_angle_deg": np.degrees(angle_ctrl),
        "angle_profile_rad": angle_profile,
        "angle_profile_deg": np.degrees(angle_profile),
        "thickness_profile": thickness_profile,
        "control_thickness": None if thickness_ctrl is None else np.asarray(thickness_ctrl, dtype=np.float64),
        "mu_required": mu_required,
    }


def build_winding_layout_data(
    angle_deg: float,
    geom: GeometryConfig,
    config: WindingConfig,
    lift: float = 0.85,
) -> dict[str, Any]:
    surface_radius = geom.outer_radius + lift
    winding = build_constant_angle_winding_paths(
        angle_deg,
        geom,
        family_count=config.family_count,
        sample_count=config.sample_count,
        surface_radius=surface_radius,
    )
    paths = []
    for handedness, points in winding["paths"]:
        paths.append(
            {
                "handedness": handedness,
                "points": np.asarray(points, dtype=np.float64),
            }
        )
    return {
        "layout_type": "winding",
        "lift": float(lift),
        "surface_radius": float(surface_radius),
        "angle_deg": float(angle_deg),
        "clairaut_radius": float(winding["clairaut_radius"]),
        "family_count": int(winding["family_count"]),
        "paths": paths,
    }


def build_hybrid_winding_layout_data(
    result: dict[str, Any],
    geom: GeometryConfig,
    family_count: int = 8,
    sample_count: int = 260,
    lift: float = 0.85,
) -> dict[str, Any]:
    surface_radius = geom.outer_radius + lift
    winding = build_variable_angle_winding_paths(
        np.asarray(result["winding_angle_ctrl"]),
        np.asarray(result["winding_s_ctrl"]),
        geom,
        family_count=family_count,
        sample_count=sample_count,
        surface_radius=surface_radius,
        thickness_ctrl=np.asarray(result["winding_thickness_ctrl"]),
    )
    basis = piecewise_linear_basis_np(np.asarray(winding["sample_s"], dtype=np.float64), np.asarray(winding["control_s"], dtype=np.float64))
    helical_thickness_profile = None
    if "helical_thickness_ctrl" in result:
        helical_thickness_profile = basis @ np.asarray(result["helical_thickness_ctrl"], dtype=np.float64)
    hoop_thickness_profile = None
    if "hoop_thickness_ctrl" in result:
        hoop_thickness_profile = basis @ np.asarray(result["hoop_thickness_ctrl"], dtype=np.float64)
    helical_pass_profile = None
    if "helical_pass_ctrl" in result:
        helical_pass_profile = basis @ np.asarray(result["helical_pass_ctrl"], dtype=np.float64)
    hoop_pass_profile = None
    if "hoop_pass_ctrl" in result:
        hoop_pass_profile = basis @ np.asarray(result["hoop_pass_ctrl"], dtype=np.float64)
    paths = []
    for handedness, points in winding["paths"]:
        paths.append(
            {
                "handedness": handedness,
                "points": np.asarray(points, dtype=np.float64),
            }
        )
    return {
        "layout_type": "winding_process",
        "lift": float(lift),
        "surface_radius": float(surface_radius),
        "family_count": int(winding["family_count"]),
        "control_s": np.asarray(winding["control_s"], dtype=np.float64),
        "control_angle_deg": np.asarray(winding["control_angle_deg"], dtype=np.float64),
        "control_thickness": None
        if winding["control_thickness"] is None
        else np.asarray(winding["control_thickness"], dtype=np.float64),
        "sample_s": np.asarray(winding["sample_s"], dtype=np.float64),
        "angle_profile_deg": np.asarray(winding["angle_profile_deg"], dtype=np.float64),
        "thickness_profile": None
        if winding["thickness_profile"] is None
        else np.asarray(winding["thickness_profile"], dtype=np.float64),
        "helical_thickness_profile": None
        if helical_thickness_profile is None
        else np.asarray(helical_thickness_profile, dtype=np.float64),
        "hoop_thickness_profile": None
        if hoop_thickness_profile is None
        else np.asarray(hoop_thickness_profile, dtype=np.float64),
        "helical_pass_profile": None
        if helical_pass_profile is None
        else np.asarray(helical_pass_profile, dtype=np.float64),
        "hoop_pass_profile": None
        if hoop_pass_profile is None
        else np.asarray(hoop_pass_profile, dtype=np.float64),
        "mu_required": np.asarray(winding["mu_required"], dtype=np.float64),
        "paths": paths,
    }


def build_winding_process_layout_data(
    result: dict[str, Any],
    geom: GeometryConfig,
    family_count: int = 8,
    sample_count: int = 260,
    lift: float = 0.85,
) -> dict[str, Any]:
    return build_hybrid_winding_layout_data(
        result,
        geom,
        family_count=family_count,
        sample_count=sample_count,
        lift=lift,
    )


def plot_winding_paths(
    angle_deg: float,
    geom: GeometryConfig,
    config: WindingConfig,
    save_path: Path | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    winding = build_constant_angle_winding_paths(angle_deg, geom, family_count=config.family_count, sample_count=config.sample_count)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_copv_surface(ax, geom, alpha=0.10)
    family_colors = {"clockwise": "forestgreen", "counter_clockwise": "darkmagenta"}
    for idx, (handedness, path) in enumerate(winding["paths"]):
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=family_colors[handedness], linewidth=1.35, alpha=0.8)
        if idx == 0:
            guide = path[:: max(1, len(path) // 18)]
            ax.scatter(guide[:, 0], guide[:, 1], guide[:, 2], color="black", s=12, depthshade=False)
    ax.set_title("Filament winding path family over the full COPV")
    ax.view_init(20, 36)
    fig.tight_layout()
    if save_path is not None:
        _prepare_output_path(save_path)
        fig.savefig(save_path, dpi=115)
    return fig, winding


def plot_hybrid_winding_paths(
    result: dict[str, Any],
    geom: GeometryConfig,
    family_count: int = 8,
    sample_count: int = 260,
    save_path: Path | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    winding = build_variable_angle_winding_paths(
        np.asarray(result["winding_angle_ctrl"]),
        np.asarray(result["winding_s_ctrl"]),
        geom,
        family_count=family_count,
        sample_count=sample_count,
        thickness_ctrl=np.asarray(result["winding_thickness_ctrl"]),
    )
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_copv_surface(ax, geom, alpha=0.10)
    family_colors = {"clockwise": "forestgreen", "counter_clockwise": "darkmagenta"}
    for idx, (handedness, path) in enumerate(winding["paths"]):
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=family_colors[handedness], linewidth=1.35, alpha=0.82)
        if idx == 0:
            guide = path[:: max(1, len(path) // 18)]
            ax.scatter(guide[:, 0], guide[:, 1], guide[:, 2], color="black", s=12, depthshade=False)
    ax.set_title("Variable-angle winding paths over the full COPV")
    ax.view_init(20, 36)
    fig.tight_layout()
    if save_path is not None:
        _prepare_output_path(save_path)
        fig.savefig(save_path, dpi=115)
    return fig, winding


def plot_winding_process_paths(
    result: dict[str, Any],
    geom: GeometryConfig,
    family_count: int = 8,
    sample_count: int = 260,
    save_path: Path | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    # Backward-compatible wrapper used by the winding-first staging scripts.
    return plot_hybrid_winding_paths(
        result,
        geom,
        family_count=family_count,
        sample_count=sample_count,
        save_path=save_path,
    )


def save_tradeoff_plot(path: Path, labels: list[str], masses: list[float], strain_energies: list[float]):
    _prepare_output_path(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["gray", "steelblue", "darkorange", "forestgreen", "crimson", "slateblue"]
    ax.scatter(masses, strain_energies, s=90, c=colors[: len(labels)])
    for label, mass, se in zip(labels, masses, strain_energies):
        ax.annotate(label, (mass, se), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Relative mass metric")
    ax.set_ylabel("Strain energy / compliance")
    ax.set_title("COPV manufacturing comparison")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def write_vtu(
    path: Path,
    nodes: np.ndarray,
    elems: np.ndarray,
    displacement: np.ndarray,
    thickness: np.ndarray,
    density: np.ndarray,
    fiber_dirs: np.ndarray,
    coverage: np.ndarray,
    extra_cell_data: dict[str, np.ndarray] | None = None,
) -> Path:
    _prepare_output_path(path)
    elems = np.asarray(elems, dtype=np.int32)
    if elems.ndim != 2 or elems.shape[1] not in (3, 4):
        raise ValueError("write_vtu expects triangle or tetrahedral connectivity")
    cell_type = "triangle" if elems.shape[1] == 3 else "tetra"
    disp_norm = np.linalg.norm(displacement, axis=1)
    cell_data = {
        "thickness": [np.asarray(thickness, dtype=np.float64)],
        "density": [np.asarray(density, dtype=np.float64)],
        "fiber_dir": [np.asarray(fiber_dirs, dtype=np.float64)],
        "coverage": [np.asarray(coverage, dtype=np.float64)],
    }
    if extra_cell_data is not None:
        for key, value in extra_cell_data.items():
            cell_data[str(key)] = [np.asarray(value, dtype=np.float64)]
    mesh = meshio.Mesh(
        points=np.asarray(nodes, dtype=np.float64),
        cells=[(cell_type, elems)],
        point_data={
            "displacement": np.asarray(displacement, dtype=np.float64),
            "displacement_norm": np.asarray(disp_norm, dtype=np.float64),
        },
        cell_data=cell_data,
    )
    mesh.write(path)
    return path
