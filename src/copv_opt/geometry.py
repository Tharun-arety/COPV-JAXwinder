from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import gmsh
import jax.numpy as jnp
import meshio
import numpy as np

from .config import GeometryConfig


_CAP_LOOKUP_SAMPLES = 2049    # 2^11 + 1 — odd count enables Simpson's rule arc-length integration
_PROFILE_BUILD_SAMPLES = 97   # prime — near-uniform arc-length spacing without aliasing harmonics
_PROJECTION_BATCH_SIZE = 2048 # vectorised nearest-point projection batch; tuned for GPU/CPU cache


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


def _cap_axes(radius: float, dome_height_ratio: float) -> tuple[float, float]:
    if dome_height_ratio <= 0.0:
        raise ValueError(f"dome_height_ratio must be > 0, got {dome_height_ratio!r}")
    equatorial_radius = float(radius)
    polar_radius = float(dome_height_ratio) * equatorial_radius
    return equatorial_radius, polar_radius


@lru_cache(maxsize=128)
def _cap_lookup(
    radius: float,
    opening_radius: float,
    dome_height_ratio: float,
    sample_count: int = _CAP_LOOKUP_SAMPLES,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radius = float(radius)
    opening_radius = float(opening_radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if opening_radius < 0.0 or opening_radius >= radius:
        raise ValueError("opening_radius must satisfy 0 <= opening_radius < radius")

    equatorial_radius, polar_radius = _cap_axes(radius, dome_height_ratio)
    theta_open = float(np.arcsin(np.clip(opening_radius / equatorial_radius, 0.0, 0.999999)))
    theta = np.linspace(theta_open, 0.5 * np.pi, int(max(sample_count, 2)), dtype=np.float64)
    ds_dtheta = np.sqrt((equatorial_radius * np.cos(theta)) ** 2 + (polar_radius * np.sin(theta)) ** 2)
    s_from_open = np.zeros_like(theta)
    if theta.size > 1:
        increments = 0.5 * (ds_dtheta[1:] + ds_dtheta[:-1]) * np.diff(theta)
        s_from_open[1:] = np.cumsum(increments)
    cap_len = float(s_from_open[-1])
    rho = equatorial_radius * np.sin(theta)
    z_rel = polar_radius * np.cos(theta)
    drho_ds = np.divide(
        equatorial_radius * np.cos(theta),
        ds_dtheta,
        out=np.zeros_like(theta),
        where=ds_dtheta > 1e-12,
    )
    dz_ds = -np.divide(
        polar_radius * np.sin(theta),
        ds_dtheta,
        out=np.zeros_like(theta),
        where=ds_dtheta > 1e-12,
    )
    return theta_open, cap_len, theta, s_from_open, rho, z_rel, drho_ds, dz_ds


def _interp1d_jax(x: jnp.ndarray, xp: jnp.ndarray, fp: jnp.ndarray) -> jnp.ndarray:
    idx = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, xp.shape[0] - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    frac = (x - x0) / jnp.maximum(x1 - x0, 1e-12)
    return y0 + frac * (y1 - y0)


def _opening_z(radius: float, cylinder_length: float, opening_radius: float, dome_height_ratio: float) -> float:
    _, polar_radius = _cap_axes(radius, dome_height_ratio)
    theta_open, _, _, _, _, _, _, _ = _cap_lookup(radius, opening_radius, dome_height_ratio)
    return 0.5 * float(cylinder_length) + polar_radius * float(np.cos(theta_open))


def copv_opening_z(
    radius: float,
    cylinder_length: float,
    opening_radius: float,
    dome_height_ratio: float = 1.0,
) -> float:
    return _opening_z(radius, cylinder_length, opening_radius, dome_height_ratio)


def _sample_top_cap_profile(
    radius: float,
    cylinder_length: float,
    opening_radius: float,
    dome_height_ratio: float,
    sample_count: int = _PROFILE_BUILD_SAMPLES,
) -> np.ndarray:
    theta_open, _, _, _, _, _, _, _ = _cap_lookup(radius, opening_radius, dome_height_ratio)
    equatorial_radius, polar_radius = _cap_axes(radius, dome_height_ratio)
    theta = np.linspace(theta_open, 0.5 * np.pi, int(max(sample_count, 2)), dtype=np.float64)
    rho = equatorial_radius * np.sin(theta)
    z = 0.5 * float(cylinder_length) + polar_radius * np.cos(theta)
    return np.column_stack([rho, z])


def _sample_bottom_cap_profile(
    radius: float,
    cylinder_length: float,
    opening_radius: float,
    dome_height_ratio: float,
    sample_count: int = _PROFILE_BUILD_SAMPLES,
) -> np.ndarray:
    theta_open, _, _, _, _, _, _, _ = _cap_lookup(radius, opening_radius, dome_height_ratio)
    equatorial_radius, polar_radius = _cap_axes(radius, dome_height_ratio)
    theta = np.linspace(0.5 * np.pi, theta_open, int(max(sample_count, 2)), dtype=np.float64)
    rho = equatorial_radius * np.sin(theta)
    z = -0.5 * float(cylinder_length) - polar_radius * np.cos(theta)
    return np.column_stack([rho, z])


def build_copv_shell(
    step_path: Path,
    outer_radius: float = 100.0,
    cylinder_length: float = 220.0,
    thickness: float = 8.0,
    opening_radius: float = 10.0,
    dome_height_ratio: float = 0.7,
) -> Path:
    """Build a midsurface COPV shell STEP with ellipsoidal dome caps and polar openings."""
    surface_radius = outer_radius - 0.5 * thickness
    if surface_radius <= 0.0:
        raise ValueError("thickness must be smaller than outer_radius")
    if opening_radius <= 0.0 or opening_radius >= surface_radius:
        raise ValueError("opening_radius must be positive and smaller than the shell midsurface radius")

    top_profile = _sample_top_cap_profile(
        surface_radius,
        cylinder_length,
        opening_radius,
        dome_height_ratio,
    )
    bottom_profile = _sample_bottom_cap_profile(
        surface_radius,
        cylinder_length,
        opening_radius,
        dome_height_ratio,
    )

    step_path.parent.mkdir(parents=True, exist_ok=True)
    if step_path.exists():
        step_path.unlink()

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(step_path.stem)
    occ = gmsh.model.occ

    def add_point(rho: float, z: float) -> int:
        return occ.addPoint(float(rho), 0.0, float(z))

    def add_curve(point_tags: list[int]) -> int:
        if len(point_tags) == 2:
            return occ.addLine(point_tags[0], point_tags[1])
        return occ.addBSpline(point_tags)

    top_tags = [add_point(rho, z) for rho, z in top_profile]
    top_eq_tag = top_tags[-1]
    bottom_eq_tag = add_point(surface_radius, -0.5 * cylinder_length)
    bottom_mid_tags = [add_point(rho, z) for rho, z in bottom_profile[1:-1]]
    bottom_open_tag = add_point(*bottom_profile[-1])

    curve_tags = [
        add_curve(top_tags),
        occ.addLine(top_eq_tag, bottom_eq_tag),
        add_curve([bottom_eq_tag, *bottom_mid_tags, bottom_open_tag]),
    ]
    occ.revolve([(1, tag) for tag in curve_tags], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0 * np.pi)
    occ.synchronize()
    gmsh.write(str(step_path))
    gmsh.finalize()
    return step_path


def read_msh(msh_path: Path, step_path: Path | None = None) -> MeshResult:
    mesh = meshio.read(str(msh_path))
    elems = mesh.cells_dict.get("triangle", mesh.cells_dict.get("triangle6"))
    if elems is None:
        elems = mesh.cells_dict.get("tetra", mesh.cells_dict.get("tetra10"))
    if elems is None:
        raise ValueError(f"No tetra or triangle elements found in {msh_path}")
    return MeshResult(
        step_path=step_path if step_path is not None else msh_path.with_suffix(".step"),
        msh_path=msh_path,
        nodes=np.asarray(mesh.points, dtype=np.float64),
        elems=np.asarray(elems, dtype=np.int32),
    )


def mesh_step(
    step_path: Path,
    msh_path: Path,
    hmin: float = 10.0,
    hmax: float = 28.0,
    boss_hmin: float = 4.0,
    boss_refine_radius: float = 28.0,
    cylinder_half_len: float | None = None,
    outer_radius: float | None = None,
    opening_radius: float = 10.0,
    dome_height_ratio: float = 0.7,
) -> MeshResult:
    """Mesh a STEP file with gmsh and return the triangle or tetrahedral arrays."""
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    if msh_path.exists():
        msh_path.unlink()

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(step_path.stem)
    gmsh.merge(str(step_path))
    gmsh.model.occ.synchronize()
    base_hmin = min(float(hmin), float(boss_hmin))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", base_hmin)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(hmax))

    if (
        boss_refine_radius > 0.0
        and boss_hmin > 0.0
        and cylinder_half_len is not None
        and outer_radius is not None
    ):
        field = gmsh.model.mesh.field
        boss_fields: list[int] = []
        top_boss_z = _opening_z(
            float(outer_radius),
            2.0 * float(cylinder_half_len),
            float(opening_radius),
            dome_height_ratio,
        )
        bottom_boss_z = -top_boss_z
        for z_center in (top_boss_z, bottom_boss_z):
            ball_id = field.add("Ball")
            field.setNumber(ball_id, "Radius", float(boss_refine_radius))
            field.setNumber(ball_id, "Thickness", max(0.5 * float(boss_refine_radius), 1e-6))
            field.setNumber(ball_id, "VIn", float(boss_hmin))
            field.setNumber(ball_id, "VOut", float(hmax))
            field.setNumber(ball_id, "XCenter", 0.0)
            field.setNumber(ball_id, "YCenter", 0.0)
            field.setNumber(ball_id, "ZCenter", float(z_center))
            boss_fields.append(ball_id)
        if len(boss_fields) == 1:
            field.setAsBackgroundMesh(boss_fields[0])
        elif boss_fields:
            min_id = field.add("Min")
            field.setNumbers(min_id, "FieldsList", boss_fields)
            field.setAsBackgroundMesh(min_id)

    gmsh.model.mesh.generate(2)
    gmsh.write(str(msh_path))
    gmsh.finalize()

    return read_msh(msh_path, step_path=step_path)


def ensure_copv_mesh(
    step_path: Path,
    msh_path: Path,
    geom: GeometryConfig,
    remesh: bool = False,
    rebuild_step: bool = False,
) -> MeshResult:
    if rebuild_step or remesh or not step_path.exists():
        build_copv_shell(
            step_path,
            outer_radius=geom.outer_radius,
            cylinder_length=geom.cylinder_length,
            thickness=geom.thickness,
            opening_radius=geom.opening_radius,
            dome_height_ratio=geom.dome_height_ratio,
        )
    if remesh or rebuild_step or not msh_path.exists():
        return mesh_step(
            step_path,
            msh_path,
            hmin=geom.mesh_hmin,
            hmax=geom.mesh_hmax,
            boss_hmin=geom.boss_hmin,
            boss_refine_radius=geom.boss_refine_radius,
            cylinder_half_len=geom.half_cyl,
            outer_radius=geom.outer_radius,
            opening_radius=geom.opening_radius,
            dome_height_ratio=geom.dome_height_ratio,
        )
    mesh = read_msh(msh_path, step_path=step_path)
    if mesh.elems.ndim != 2 or mesh.elems.shape[1] != 3:
        build_copv_shell(
            step_path,
            outer_radius=geom.outer_radius,
            cylinder_length=geom.cylinder_length,
            thickness=geom.thickness,
            opening_radius=geom.opening_radius,
            dome_height_ratio=geom.dome_height_ratio,
        )
        return mesh_step(
            step_path,
            msh_path,
            hmin=geom.mesh_hmin,
            hmax=geom.mesh_hmax,
            boss_hmin=geom.boss_hmin,
            boss_refine_radius=geom.boss_refine_radius,
            cylinder_half_len=geom.half_cyl,
            outer_radius=geom.outer_radius,
            opening_radius=geom.opening_radius,
            dome_height_ratio=geom.dome_height_ratio,
        )
    return mesh


def copv_meridional_metrics(
    radius: float,
    cylinder_length: float,
    opening_radius: float,
    dome_height_ratio: float = 1.0,
) -> tuple[float, float, float]:
    theta_open, cap_len, _, _, _, _, _, _ = _cap_lookup(radius, opening_radius, dome_height_ratio)
    total_len = 2.0 * cap_len + float(cylinder_length)
    return theta_open, cap_len, total_len


def copv_principal_curvatures_np(
    radius: float,
    s: np.ndarray,
    cylinder_length: float,
    opening_radius: float,
    dome_height_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    _, cap_len, total_len = copv_meridional_metrics(
        radius,
        cylinder_length,
        opening_radius,
        dome_height_ratio,
    )
    equatorial_radius, polar_radius = _cap_axes(radius, dome_height_ratio)
    _, _, theta_grid, s_grid, _, _, _, _ = _cap_lookup(radius, opening_radius, dome_height_ratio)

    s = np.asarray(s, dtype=np.float64).reshape(-1)
    s = np.clip(s, 0.0, total_len)
    top = s <= cap_len
    cyl = (s > cap_len) & (s < cap_len + cylinder_length)
    bot = ~(top | cyl)

    k_meridian = np.zeros_like(s)
    k_hoop = np.zeros_like(s)

    if top.any():
        theta = np.interp(s[top], s_grid, theta_grid)
        ds_dtheta = np.sqrt((equatorial_radius * np.cos(theta)) ** 2 + (polar_radius * np.sin(theta)) ** 2)
        k_meridian[top] = -(equatorial_radius * polar_radius) / np.maximum(ds_dtheta**3, 1e-12)
        k_hoop[top] = -polar_radius / np.maximum(equatorial_radius * ds_dtheta, 1e-12)

    if cyl.any():
        k_meridian[cyl] = 0.0
        k_hoop[cyl] = -1.0 / max(float(radius), 1e-12)

    if bot.any():
        s_bot = s[bot] - cap_len - cylinder_length
        theta = np.interp(cap_len - s_bot, s_grid, theta_grid)
        ds_dtheta = np.sqrt((equatorial_radius * np.cos(theta)) ** 2 + (polar_radius * np.sin(theta)) ** 2)
        k_meridian[bot] = -(equatorial_radius * polar_radius) / np.maximum(ds_dtheta**3, 1e-12)
        k_hoop[bot] = -polar_radius / np.maximum(equatorial_radius * ds_dtheta, 1e-12)

    return k_meridian, k_hoop


def copv_surface_from_sphi_np(
    radius: float,
    s: np.ndarray,
    phi: np.ndarray,
    cylinder_length: float,
    opening_radius: float,
    dome_height_ratio: float = 1.0,
) -> dict[str, np.ndarray]:
    theta_open, cap_len, total_len = copv_meridional_metrics(
        radius,
        cylinder_length,
        opening_radius,
        dome_height_ratio,
    )
    del theta_open
    _, polar_radius = _cap_axes(radius, dome_height_ratio)
    _, _, theta_grid, s_grid, _, _, _, _ = _cap_lookup(radius, opening_radius, dome_height_ratio)
    half_cyl = 0.5 * float(cylinder_length)
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
        theta = np.interp(s[top], s_grid, theta_grid)
        sp = np.sin(phi[top])
        cp = np.cos(phi[top])
        ds_dtheta = np.sqrt((radius * np.cos(theta)) ** 2 + (polar_radius * np.sin(theta)) ** 2)
        drho_ds = np.divide(radius * np.cos(theta), ds_dtheta, out=np.zeros_like(theta), where=ds_dtheta > 1e-12)
        dz_ds = -np.divide(polar_radius * np.sin(theta), ds_dtheta, out=np.zeros_like(theta), where=ds_dtheta > 1e-12)
        rho_seg = radius * np.sin(theta)
        z_rel = polar_radius * np.cos(theta)
        points[top] = np.stack([rho_seg * cp, rho_seg * sp, half_cyl + z_rel], axis=-1)
        e_s[top] = np.stack([drho_ds * cp, drho_ds * sp, dz_ds], axis=-1)
        e_phi[top] = np.stack([-sp, cp, np.zeros_like(theta)], axis=-1)
        normals[top] = np.stack(
            [
                (rho_seg * cp) / max(radius**2, 1e-12),
                (rho_seg * sp) / max(radius**2, 1e-12),
                z_rel / max(polar_radius**2, 1e-12),
            ],
            axis=-1,
        )
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
        theta = np.interp(cap_len - s_bot, s_grid, theta_grid)
        sp = np.sin(phi[bot])
        cp = np.cos(phi[bot])
        ds_dtheta = np.sqrt((radius * np.cos(theta)) ** 2 + (polar_radius * np.sin(theta)) ** 2)
        drho_ds = -np.divide(radius * np.cos(theta), ds_dtheta, out=np.zeros_like(theta), where=ds_dtheta > 1e-12)
        dz_ds = -np.divide(polar_radius * np.sin(theta), ds_dtheta, out=np.zeros_like(theta), where=ds_dtheta > 1e-12)
        rho_seg = radius * np.sin(theta)
        z_rel = -polar_radius * np.cos(theta)
        points[bot] = np.stack([rho_seg * cp, rho_seg * sp, -half_cyl + z_rel], axis=-1)
        e_s[bot] = np.stack([drho_ds * cp, drho_ds * sp, dz_ds], axis=-1)
        e_phi[bot] = np.stack([-sp, cp, np.zeros_like(theta)], axis=-1)
        normals[bot] = np.stack(
            [
                (rho_seg * cp) / max(radius**2, 1e-12),
                (rho_seg * sp) / max(radius**2, 1e-12),
                z_rel / max(polar_radius**2, 1e-12),
            ],
            axis=-1,
        )
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
    dome_height_ratio: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    theta_open, cap_len, total_len = copv_meridional_metrics(
        radius,
        cylinder_length,
        opening_radius,
        dome_height_ratio,
    )
    del theta_open
    _, polar_radius = _cap_axes(radius, dome_height_ratio)
    _, _, theta_grid_np, s_grid_np, _, _, _, _ = _cap_lookup(radius, opening_radius, dome_height_ratio)
    half_cyl = 0.5 * float(cylinder_length)
    s = jnp.clip(jnp.reshape(s, (-1,)), 0.0, total_len)
    phi = jnp.mod(jnp.reshape(phi, (-1,)), 2.0 * jnp.pi)
    theta_grid = jnp.asarray(theta_grid_np, dtype=s.dtype)
    s_grid = jnp.asarray(s_grid_np, dtype=s.dtype)

    top = s <= cap_len
    cyl = (s > cap_len) & (s < cap_len + cylinder_length)

    sp = jnp.sin(phi)
    cp = jnp.cos(phi)

    theta_top = _interp1d_jax(s, s_grid, theta_grid)
    ds_dtheta_top = jnp.sqrt((radius * jnp.cos(theta_top)) ** 2 + (polar_radius * jnp.sin(theta_top)) ** 2)
    drho_ds_top = (radius * jnp.cos(theta_top)) / jnp.maximum(ds_dtheta_top, 1e-8)
    dz_ds_top = -(polar_radius * jnp.sin(theta_top)) / jnp.maximum(ds_dtheta_top, 1e-8)
    rho_top = radius * jnp.sin(theta_top)
    z_top = polar_radius * jnp.cos(theta_top)
    pt_top = jnp.stack([rho_top * cp, rho_top * sp, half_cyl + z_top], axis=-1)
    es_top = jnp.stack([drho_ds_top * cp, drho_ds_top * sp, dz_ds_top], axis=-1)
    eph_top = jnp.stack([-sp, cp, jnp.zeros_like(sp)], axis=-1)
    # 1e-8 mm² floor prevents division by zero at the polar boss (rho -> 0); physically ~0.0001 mm radius.
    n_top = jnp.stack(
        [
            (rho_top * cp) / max(radius**2, 1e-8),
            (rho_top * sp) / max(radius**2, 1e-8),
            z_top / max(polar_radius**2, 1e-8),
        ],
        axis=-1,
    )

    s_cyl = s - cap_len
    pt_cyl = jnp.stack([radius * cp, radius * sp, half_cyl - s_cyl], axis=-1)
    es_cyl = jnp.stack([jnp.zeros_like(sp), jnp.zeros_like(sp), -jnp.ones_like(sp)], axis=-1)
    eph_cyl = jnp.stack([-sp, cp, jnp.zeros_like(sp)], axis=-1)
    n_cyl = jnp.stack([cp, sp, jnp.zeros_like(sp)], axis=-1)

    s_bot = s - cap_len - cylinder_length
    theta_bot = _interp1d_jax(cap_len - s_bot, s_grid, theta_grid)
    ds_dtheta_bot = jnp.sqrt((radius * jnp.cos(theta_bot)) ** 2 + (polar_radius * jnp.sin(theta_bot)) ** 2)
    drho_ds_bot = -(radius * jnp.cos(theta_bot)) / jnp.maximum(ds_dtheta_bot, 1e-8)
    dz_ds_bot = -(polar_radius * jnp.sin(theta_bot)) / jnp.maximum(ds_dtheta_bot, 1e-8)
    rho_bot = radius * jnp.sin(theta_bot)
    z_bot = -polar_radius * jnp.cos(theta_bot)
    pt_bot = jnp.stack([rho_bot * cp, rho_bot * sp, -half_cyl + z_bot], axis=-1)
    es_bot = jnp.stack([drho_ds_bot * cp, drho_ds_bot * sp, dz_ds_bot], axis=-1)
    eph_bot = jnp.stack([-sp, cp, jnp.zeros_like(sp)], axis=-1)
    # same pole regularization as n_top above
    n_bot = jnp.stack(
        [
            (rho_bot * cp) / max(radius**2, 1e-8),
            (rho_bot * sp) / max(radius**2, 1e-8),
            z_bot / max(polar_radius**2, 1e-8),
        ],
        axis=-1,
    )

    points = jnp.where(top[:, None], pt_top, jnp.where(cyl[:, None], pt_cyl, pt_bot))
    e_s = jnp.where(top[:, None], es_top, jnp.where(cyl[:, None], es_cyl, es_bot))
    e_phi = jnp.where(top[:, None], eph_top, jnp.where(cyl[:, None], eph_cyl, eph_bot))
    normals = jnp.where(top[:, None], n_top, jnp.where(cyl[:, None], n_cyl, n_bot))
    rho = jnp.where(top, rho_top, jnp.where(cyl, jnp.full_like(s, radius), rho_bot))

    return points, normalize_jax(e_s), normalize_jax(e_phi), normalize_jax(normals), rho


def _project_cap_candidates(
    rho_query: np.ndarray,
    z_query: np.ndarray,
    radius: float,
    opening_radius: float,
    dome_height_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, _, _, s_grid, rho_grid, z_grid, _, _ = _cap_lookup(radius, opening_radius, dome_height_ratio)
    rho_query = np.asarray(rho_query, dtype=np.float64).reshape(-1)
    z_query = np.asarray(z_query, dtype=np.float64).reshape(-1)
    best_s = np.zeros_like(rho_query)
    best_rho = np.zeros_like(rho_query)
    best_z = np.zeros_like(rho_query)
    best_d2 = np.zeros_like(rho_query)

    for start in range(0, len(rho_query), _PROJECTION_BATCH_SIZE):
        stop = min(start + _PROJECTION_BATCH_SIZE, len(rho_query))
        dr = rho_query[start:stop, None] - rho_grid[None, :]
        dz = z_query[start:stop, None] - z_grid[None, :]
        dist2 = dr * dr + dz * dz
        idx = np.argmin(dist2, axis=1)
        row = np.arange(stop - start)
        best_s[start:stop] = s_grid[idx]
        best_rho[start:stop] = rho_grid[idx]
        best_z[start:stop] = z_grid[idx]
        best_d2[start:stop] = dist2[row, idx]

    return best_s, best_rho, best_z, best_d2


def copv_surface_projection_np(
    points: np.ndarray,
    radius: float,
    cylinder_length: float,
    opening_radius: float,
    dome_height_ratio: float = 1.0,
) -> dict[str, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    _, cap_len, total_len = copv_meridional_metrics(
        radius,
        cylinder_length,
        opening_radius,
        dome_height_ratio,
    )
    half_cyl = 0.5 * float(cylinder_length)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    rho = np.sqrt(x**2 + y**2)
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)

    z_cyl = np.clip(z, -half_cyl, half_cyl)
    s_cyl = cap_len + np.clip(half_cyl - z, 0.0, cylinder_length)
    d2_cyl = (rho - radius) ** 2 + (z - z_cyl) ** 2

    top_s_open, _, _, d2_top = _project_cap_candidates(
        rho,
        z - half_cyl,
        radius,
        opening_radius,
        dome_height_ratio,
    )
    bot_s_open, _, _, d2_bot = _project_cap_candidates(
        rho,
        -(z + half_cyl),
        radius,
        opening_radius,
        dome_height_ratio,
    )

    s_top = top_s_open
    s_bot = cap_len + cylinder_length + (cap_len - bot_s_open)

    use_top = (d2_top <= d2_cyl) & (d2_top <= d2_bot)
    use_bot = (d2_bot < d2_cyl) & (d2_bot < d2_top)
    s = np.where(use_top, s_top, np.where(use_bot, s_bot, s_cyl))
    s = np.clip(s, 0.0, total_len)
    return copv_surface_from_sphi_np(
        radius,
        s,
        phi,
        cylinder_length,
        opening_radius,
        dome_height_ratio=dome_height_ratio,
    )


def project_to_copv_surface(
    points: np.ndarray,
    radius: float,
    cylinder_length: float,
    opening_radius: float,
    dome_height_ratio: float = 1.0,
) -> np.ndarray:
    return copv_surface_projection_np(
        points,
        radius,
        cylinder_length,
        opening_radius,
        dome_height_ratio=dome_height_ratio,
    )["points"]


def copv_skin_distance(
    points: np.ndarray,
    radius: float,
    cylinder_length: float,
    opening_radius: float,
    dome_height_ratio: float = 1.0,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    projection = copv_surface_projection_np(
        pts,
        radius,
        cylinder_length,
        opening_radius,
        dome_height_ratio=dome_height_ratio,
    )
    return np.linalg.norm(projection["points"] - pts, axis=1)


def copv_normals_np(
    points: np.ndarray,
    radius: float,
    cylinder_length: float,
    opening_radius: float,
    dome_height_ratio: float = 1.0,
) -> np.ndarray:
    return copv_surface_projection_np(
        points,
        radius,
        cylinder_length,
        opening_radius,
        dome_height_ratio=dome_height_ratio,
    )["normals"]


def extract_boundary_faces(elems: np.ndarray) -> np.ndarray:
    elems = np.asarray(elems, dtype=np.int32)
    if elems.ndim != 2:
        raise ValueError("elems must be a 2D array")
    if elems.shape[1] == 3:
        return elems
    if elems.shape[1] != 4:
        raise ValueError("extract_boundary_faces expects triangle or tetrahedral connectivity")
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


def orient_shell_faces(
    nodes: np.ndarray,
    faces: np.ndarray,
    radius: float,
    cylinder_length: float,
    opening_radius: float,
    dome_height_ratio: float = 1.0,
) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int32).copy()
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("orient_shell_faces expects triangle connectivity")
    tri_pts = np.asarray(nodes, dtype=np.float64)[faces]
    centroids = tri_pts.mean(axis=1)
    target_normals = copv_normals_np(
        centroids,
        radius,
        cylinder_length,
        opening_radius,
        dome_height_ratio=dome_height_ratio,
    )
    face_normals = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    flip_mask = np.einsum("ij,ij->i", face_normals, target_normals) < 0.0
    faces[flip_mask] = faces[flip_mask][:, [0, 2, 1]]
    return faces


def classify_copv_boundary_faces(
    nodes: np.ndarray,
    faces: np.ndarray,
    geom: GeometryConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri_pts = nodes[faces]
    centroids = tri_pts.mean(axis=1)
    rho = np.linalg.norm(centroids[:, :2], axis=1)
    opening_z = _opening_z(
        geom.inner_radius,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    opening_mask = (rho <= geom.opening_radius + geom.support_tol) & (np.abs(centroids[:, 2]) >= opening_z - geom.support_tol)
    inner_dist = copv_skin_distance(
        centroids,
        geom.inner_radius,
        geom.cylinder_length,
        geom.opening_radius,
        dome_height_ratio=geom.dome_height_ratio,
    )
    outer_dist = copv_skin_distance(
        centroids,
        geom.outer_radius,
        geom.cylinder_length,
        geom.opening_radius,
        dome_height_ratio=geom.dome_height_ratio,
    )
    inner_mask = (~opening_mask) & (inner_dist <= outer_dist)
    outer_mask = (~opening_mask) & (outer_dist < inner_dist)
    return inner_mask, outer_mask, opening_mask
