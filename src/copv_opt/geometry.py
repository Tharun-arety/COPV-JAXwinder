from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gmsh
import jax.numpy as jnp
import meshio
import numpy as np

from .config import GeometryConfig


@dataclass
class MeshResult:
    step_path: Path
    msh_path: Path
    nodes: np.ndarray
    elems: np.ndarray


def normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / np.linalg.norm(x, axis=axis, keepdims=True).clip(min=eps)


def normalize_jax(x: jnp.ndarray, axis: int = -1, eps: float = 1e-8) -> jnp.ndarray:
    return x / jnp.linalg.norm(x, axis=axis, keepdims=True).clip(min=eps)


def build_copv_shell(
    step_path: Path,
    outer_radius: float = 100.0,
    cylinder_length: float = 220.0,
    thickness: float = 8.0,
    opening_radius: float = 10.0,
) -> Path:
    """Build a simple thick-shell COPV solid with polar openings."""
    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError("cadquery is required only when rebuilding the STEP geometry.") from exc

    inner_radius = outer_radius - thickness
    if inner_radius <= 0.0:
        raise ValueError("thickness must be smaller than outer_radius")

    half_cyl = 0.5 * cylinder_length
    outer = (
        cq.Workplane("XY")
        .cylinder(cylinder_length, outer_radius)
        .union(cq.Workplane("XY").sphere(outer_radius).translate((0.0, 0.0, half_cyl)))
        .union(cq.Workplane("XY").sphere(outer_radius).translate((0.0, 0.0, -half_cyl)))
    )
    inner = (
        cq.Workplane("XY")
        .cylinder(cylinder_length, inner_radius)
        .union(cq.Workplane("XY").sphere(inner_radius).translate((0.0, 0.0, half_cyl)))
        .union(cq.Workplane("XY").sphere(inner_radius).translate((0.0, 0.0, -half_cyl)))
    )
    shell = outer.cut(inner)

    top_cut = (
        cq.Workplane("XY")
        .cylinder(2.2 * outer_radius, opening_radius)
        .translate((0.0, 0.0, half_cyl + 0.55 * outer_radius))
    )
    bot_cut = (
        cq.Workplane("XY")
        .cylinder(2.2 * outer_radius, opening_radius)
        .translate((0.0, 0.0, -half_cyl - 0.55 * outer_radius))
    )
    shell = shell.cut(top_cut).cut(bot_cut)

    step_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(shell, str(step_path))
    return step_path


def read_msh(msh_path: Path, step_path: Path | None = None) -> MeshResult:
    mesh = meshio.read(str(msh_path))
    elems = mesh.cells_dict.get("tetra", mesh.cells_dict.get("tetra10"))
    if elems is None:
        raise ValueError(f"No tetra elements found in {msh_path}")
    return MeshResult(
        step_path=step_path if step_path is not None else msh_path.with_suffix(".step"),
        msh_path=msh_path,
        nodes=np.asarray(mesh.points, dtype=np.float64),
        elems=np.asarray(elems, dtype=np.int32),
    )


def mesh_step(
    step_path: Path,
    msh_path: Path,
    hmin: float = 12.0,
    hmax: float = 28.0,
) -> MeshResult:
    """Mesh a STEP file with gmsh and return the tetrahedral arrays."""
    msh_path.parent.mkdir(parents=True, exist_ok=True)

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(step_path.stem)
    gmsh.merge(str(step_path))
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmin)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(msh_path))
    gmsh.finalize()

    return read_msh(msh_path, step_path=step_path)


def ensure_copv_mesh(
    step_path: Path,
    msh_path: Path,
    geom: GeometryConfig,
    remesh: bool = False,
) -> MeshResult:
    if remesh or not step_path.exists():
        build_copv_shell(
            step_path,
            outer_radius=geom.outer_radius,
            cylinder_length=geom.cylinder_length,
            thickness=geom.thickness,
            opening_radius=geom.opening_radius,
        )
    if remesh or not msh_path.exists():
        return mesh_step(step_path, msh_path, hmin=geom.mesh_hmin, hmax=geom.mesh_hmax)
    return read_msh(msh_path, step_path=step_path)


def copv_meridional_metrics(radius: float, cylinder_length: float, opening_radius: float) -> tuple[float, float, float]:
    theta_open = float(np.arcsin(np.clip(opening_radius / radius, 0.0, 0.999999)))
    cap_len = radius * (np.pi / 2.0 - theta_open)
    total_len = 2.0 * cap_len + cylinder_length
    return theta_open, cap_len, total_len


def copv_surface_from_sphi_np(
    radius: float,
    s: np.ndarray,
    phi: np.ndarray,
    cylinder_length: float,
    opening_radius: float,
) -> dict[str, np.ndarray]:
    theta_open, cap_len, total_len = copv_meridional_metrics(radius, cylinder_length, opening_radius)
    half_cyl = 0.5 * cylinder_length
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    phi = np.mod(np.asarray(phi, dtype=np.float64).reshape(-1), 2.0 * np.pi)
    s = np.clip(s, 0.0, total_len)

    top = s <= cap_len
    cyl = (s > cap_len) & (s < cap_len + cylinder_length)
    bot = ~(top | cyl)

    points = np.zeros((len(s), 3), dtype=np.float64)
    e_s = np.zeros_like(points)
    e_phi = np.zeros_like(points)
    normals = np.zeros_like(points)
    rho = np.zeros((len(s),), dtype=np.float64)

    if top.any():
        theta = theta_open + s[top] / radius
        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi[top])
        cp = np.cos(phi[top])
        rho_seg = radius * st
        points[top] = np.stack([rho_seg * cp, rho_seg * sp, half_cyl + radius * ct], axis=-1)
        e_s[top] = np.stack([ct * cp, ct * sp, -st], axis=-1)
        e_phi[top] = np.stack([-sp, cp, np.zeros_like(st)], axis=-1)
        normals[top] = np.stack([st * cp, st * sp, ct], axis=-1)
        rho[top] = rho_seg

    if cyl.any():
        s_cyl = s[cyl] - cap_len
        sp = np.sin(phi[cyl])
        cp = np.cos(phi[cyl])
        points[cyl] = np.stack([radius * cp, radius * sp, half_cyl - s_cyl], axis=-1)
        e_s[cyl] = np.stack([np.zeros_like(sp), np.zeros_like(sp), -np.ones_like(sp)], axis=-1)
        e_phi[cyl] = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1)
        normals[cyl] = np.stack([cp, sp, np.zeros_like(sp)], axis=-1)
        rho[cyl] = radius

    if bot.any():
        s_bot = s[bot] - cap_len - cylinder_length
        theta = np.pi / 2.0 + s_bot / radius
        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi[bot])
        cp = np.cos(phi[bot])
        rho_seg = radius * st
        points[bot] = np.stack([rho_seg * cp, rho_seg * sp, -half_cyl + radius * ct], axis=-1)
        e_s[bot] = np.stack([ct * cp, ct * sp, -st], axis=-1)
        e_phi[bot] = np.stack([-sp, cp, np.zeros_like(st)], axis=-1)
        normals[bot] = np.stack([st * cp, st * sp, ct], axis=-1)
        rho[bot] = rho_seg

    return {
        "points": points,
        "meridian_dirs": normalize_np(e_s),
        "hoop_dirs": normalize_np(e_phi),
        "normals": normalize_np(normals),
        "rho": rho,
        "s": s,
        "phi": phi,
    }


def copv_surface_from_sphi_jax(
    radius: float,
    s: jnp.ndarray,
    phi: jnp.ndarray,
    cylinder_length: float,
    opening_radius: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    theta_open, cap_len, total_len = copv_meridional_metrics(radius, cylinder_length, opening_radius)
    half_cyl = 0.5 * cylinder_length
    s = jnp.clip(jnp.reshape(s, (-1,)), 0.0, total_len)
    phi = jnp.mod(jnp.reshape(phi, (-1,)), 2.0 * jnp.pi)

    top = s <= cap_len
    cyl = (s > cap_len) & (s < cap_len + cylinder_length)

    theta_top = theta_open + s / radius
    st_top = jnp.sin(theta_top)
    ct_top = jnp.cos(theta_top)
    sp = jnp.sin(phi)
    cp = jnp.cos(phi)
    rho_top = radius * st_top
    pt_top = jnp.stack([rho_top * cp, rho_top * sp, half_cyl + radius * ct_top], axis=-1)
    es_top = jnp.stack([ct_top * cp, ct_top * sp, -st_top], axis=-1)
    eph_top = jnp.stack([-sp, cp, jnp.zeros_like(st_top)], axis=-1)
    n_top = jnp.stack([st_top * cp, st_top * sp, ct_top], axis=-1)

    s_cyl = s - cap_len
    pt_cyl = jnp.stack([radius * cp, radius * sp, half_cyl - s_cyl], axis=-1)
    es_cyl = jnp.stack([jnp.zeros_like(sp), jnp.zeros_like(sp), -jnp.ones_like(sp)], axis=-1)
    eph_cyl = jnp.stack([-sp, cp, jnp.zeros_like(sp)], axis=-1)
    n_cyl = jnp.stack([cp, sp, jnp.zeros_like(sp)], axis=-1)

    s_bot = s - cap_len - cylinder_length
    theta_bot = jnp.pi / 2.0 + s_bot / radius
    st_bot = jnp.sin(theta_bot)
    ct_bot = jnp.cos(theta_bot)
    rho_bot = radius * st_bot
    pt_bot = jnp.stack([rho_bot * cp, rho_bot * sp, -half_cyl + radius * ct_bot], axis=-1)
    es_bot = jnp.stack([ct_bot * cp, ct_bot * sp, -st_bot], axis=-1)
    eph_bot = jnp.stack([-sp, cp, jnp.zeros_like(st_bot)], axis=-1)
    n_bot = jnp.stack([st_bot * cp, st_bot * sp, ct_bot], axis=-1)

    points = jnp.where(top[:, None], pt_top, jnp.where(cyl[:, None], pt_cyl, pt_bot))
    e_s = jnp.where(top[:, None], es_top, jnp.where(cyl[:, None], es_cyl, es_bot))
    e_phi = jnp.where(top[:, None], eph_top, jnp.where(cyl[:, None], eph_cyl, eph_bot))
    normals = jnp.where(top[:, None], n_top, jnp.where(cyl[:, None], n_cyl, n_bot))
    rho = jnp.where(top, rho_top, jnp.where(cyl, jnp.full_like(s, radius), rho_bot))

    return points, normalize_jax(e_s), normalize_jax(e_phi), normalize_jax(normals), rho


def copv_surface_projection_np(
    points: np.ndarray,
    radius: float,
    cylinder_length: float,
    opening_radius: float,
) -> dict[str, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    theta_open, cap_len, total_len = copv_meridional_metrics(radius, cylinder_length, opening_radius)
    half_cyl = 0.5 * cylinder_length
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    rho = np.sqrt(x**2 + y**2)
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    s = np.zeros((len(pts),), dtype=np.float64)

    top = z > half_cyl
    cyl = (z >= -half_cyl) & (z <= half_cyl)
    bot = z < -half_cyl

    if top.any():
        theta = np.arctan2(rho[top], z[top] - half_cyl)
        theta = np.clip(theta, theta_open, np.pi / 2.0)
        s[top] = radius * (theta - theta_open)
    if cyl.any():
        s[cyl] = cap_len + np.clip(half_cyl - z[cyl], 0.0, cylinder_length)
    if bot.any():
        theta = np.arctan2(rho[bot], z[bot] + half_cyl)
        theta = np.clip(theta, np.pi / 2.0, np.pi - theta_open)
        s[bot] = cap_len + cylinder_length + radius * (theta - np.pi / 2.0)

    s = np.clip(s, 0.0, total_len)
    return copv_surface_from_sphi_np(radius, s, phi, cylinder_length, opening_radius)


def project_to_copv_surface(points: np.ndarray, radius: float, cylinder_length: float, opening_radius: float) -> np.ndarray:
    return copv_surface_projection_np(points, radius, cylinder_length, opening_radius)["points"]


def copv_skin_distance(points: np.ndarray, radius: float, cylinder_length: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    rho = np.linalg.norm(pts[:, :2], axis=1)
    z = pts[:, 2]
    half_cyl = 0.5 * cylinder_length
    dist = np.zeros((len(pts),), dtype=np.float64)

    cyl = np.abs(z) <= half_cyl
    top = z > half_cyl
    bot = z < -half_cyl

    if cyl.any():
        dist[cyl] = np.abs(rho[cyl] - radius)
    if top.any():
        rel = pts[top] - np.array([0.0, 0.0, half_cyl], dtype=np.float64)
        dist[top] = np.abs(np.linalg.norm(rel, axis=1) - radius)
    if bot.any():
        rel = pts[bot] - np.array([0.0, 0.0, -half_cyl], dtype=np.float64)
        dist[bot] = np.abs(np.linalg.norm(rel, axis=1) - radius)
    return dist


def copv_normals_np(points: np.ndarray, radius: float, cylinder_length: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    rho = np.linalg.norm(pts[:, :2], axis=1).clip(min=1e-12)
    z = pts[:, 2]
    half_cyl = 0.5 * cylinder_length
    normals = np.zeros_like(pts)

    cyl = np.abs(z) <= half_cyl
    top = z > half_cyl
    bot = z < -half_cyl

    if cyl.any():
        normals[cyl, 0] = pts[cyl, 0] / rho[cyl]
        normals[cyl, 1] = pts[cyl, 1] / rho[cyl]
    if top.any():
        rel = pts[top] - np.array([0.0, 0.0, half_cyl], dtype=np.float64)
        normals[top] = normalize_np(rel)
    if bot.any():
        rel = pts[bot] - np.array([0.0, 0.0, -half_cyl], dtype=np.float64)
        normals[bot] = normalize_np(rel)
    return normals


def extract_boundary_faces(elems: np.ndarray) -> np.ndarray:
    faces = np.concatenate(
        [
            elems[:, [1, 2, 3]],
            elems[:, [0, 3, 2]],
            elems[:, [0, 1, 3]],
            elems[:, [0, 2, 1]],
        ],
        axis=0,
    )
    sorted_faces = np.sort(faces, axis=1)
    _, idx, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]]


def classify_copv_boundary_faces(
    nodes: np.ndarray,
    faces: np.ndarray,
    geom: GeometryConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri_pts = nodes[faces]
    centroids = tri_pts.mean(axis=1)
    rho = np.linalg.norm(centroids[:, :2], axis=1)
    opening_z = geom.half_cyl + np.sqrt(max(geom.inner_radius**2 - geom.opening_radius**2, 0.0))
    opening_mask = (rho <= geom.opening_radius + geom.support_tol) & (np.abs(centroids[:, 2]) >= opening_z - geom.support_tol)
    inner_dist = copv_skin_distance(centroids, geom.inner_radius, geom.cylinder_length)
    outer_dist = copv_skin_distance(centroids, geom.outer_radius, geom.cylinder_length)
    inner_mask = (~opening_mask) & (inner_dist <= outer_dist)
    outer_mask = (~opening_mask) & (outer_dist < inner_dist)
    return inner_mask, outer_mask, opening_mask
