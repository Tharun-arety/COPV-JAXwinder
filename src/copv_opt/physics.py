from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .config import FailureConfig, FrictionConfig, GeometryConfig, MaterialAllowables, MaterialConfig
from .geometry import (
    classify_copv_boundary_faces,
    copv_normals_np,
    copv_surface_projection_np,
    extract_boundary_faces,
    normalize_jax,
)


VM = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def orthotropic_stiffness_matrix(material: MaterialConfig) -> np.ndarray:
    e1 = material.e_xx
    e2 = material.e_yy
    e3 = material.e_zz
    nu12 = material.nu_xy
    nu13 = material.nu_xz
    nu23 = material.nu_yz
    g23 = material.g_yz
    g13 = material.g_xz
    g12 = material.g_xy

    nu21 = nu12 * e2 / e1
    nu31 = nu13 * e3 / e1
    nu32 = nu23 * e3 / e2
    compliance = np.array(
        [
            [1.0 / e1, -nu21 / e2, -nu31 / e3, 0.0, 0.0, 0.0],
            [-nu12 / e1, 1.0 / e2, -nu32 / e3, 0.0, 0.0, 0.0],
            [-nu13 / e1, -nu23 / e2, 1.0 / e3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / g23, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / g13, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / g12],
        ],
        dtype=np.float64,
    )
    return np.linalg.inv(compliance)


def d6_to_c4(c6: np.ndarray) -> np.ndarray:
    c4 = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for a, (i, j) in enumerate(VM):
        for b, (k, l) in enumerate(VM):
            value = c6[a, b]
            c4[i, j, k, l] = value
            c4[j, i, k, l] = value
            c4[i, j, l, k] = value
            c4[j, i, l, k] = value
    return c4


def base_material_tensor(material: MaterialConfig) -> np.ndarray:
    return d6_to_c4(orthotropic_stiffness_matrix(material))


def build_pressure_forces(nodes: np.ndarray, faces: np.ndarray, geom: GeometryConfig) -> np.ndarray:
    tri_pts = nodes[faces]
    centroids = tri_pts.mean(axis=1)
    normals = copv_normals_np(centroids, geom.inner_radius, geom.cylinder_length)
    areas = 0.5 * np.linalg.norm(np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]), axis=1)
    face_forces = geom.pressure * areas[:, None] * normals
    forces = np.zeros((len(nodes), 3), dtype=np.float64)
    for idx in range(3):
        np.add.at(forces, faces[:, idx], face_forces / 3.0)
    return forces


def build_copv_fem_state(
    nodes: np.ndarray,
    elems: np.ndarray,
    material: MaterialConfig,
    geom: GeometryConfig,
) -> dict[str, Any]:
    nodes = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(elems, dtype=np.int32)

    ne = nodes[elems]
    p0 = ne[:, 0]
    jac = np.stack([ne[:, 1] - p0, ne[:, 2] - p0, ne[:, 3] - p0], axis=1)
    volumes = np.abs(np.linalg.det(jac)) / 6.0

    d_n = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    d_ndx = np.einsum("ij,ekj->eik", d_n, np.linalg.inv(jac))
    b = np.zeros((len(elems), 6, 12), dtype=np.float64)
    for i in range(4):
        b[:, 0, 3 * i] = d_ndx[:, i, 0]
        b[:, 1, 3 * i + 1] = d_ndx[:, i, 1]
        b[:, 2, 3 * i + 2] = d_ndx[:, i, 2]
        b[:, 3, 3 * i + 1] = d_ndx[:, i, 2]
        b[:, 3, 3 * i + 2] = d_ndx[:, i, 1]
        b[:, 4, 3 * i] = d_ndx[:, i, 2]
        b[:, 4, 3 * i + 2] = d_ndx[:, i, 0]
        b[:, 5, 3 * i] = d_ndx[:, i, 1]
        b[:, 5, 3 * i + 1] = d_ndx[:, i, 0]

    elem_dofs = (elems[:, :, None] * 3 + np.arange(3, dtype=np.int32)).reshape(len(elems), 12)
    n_dof = len(nodes) * 3

    rho_nodes = np.linalg.norm(nodes[:, :2], axis=1)
    opening_z = geom.half_cyl + np.sqrt(max(geom.inner_radius**2 - geom.opening_radius**2, 0.0))
    support_mask = (rho_nodes <= geom.opening_radius + geom.support_tol) & (np.abs(nodes[:, 2]) >= opening_z - geom.support_tol)
    fixed_mask = np.tile(support_mask[:, None], (1, 3)).reshape(-1)
    free_dofs = np.setdiff1d(np.arange(n_dof), np.where(fixed_mask)[0]).astype(np.int32)

    boundary_faces = extract_boundary_faces(elems)
    inner_mask, outer_mask, _ = classify_copv_boundary_faces(nodes, boundary_faces, geom)
    inner_faces = boundary_faces[inner_mask]
    outer_faces = boundary_faces[outer_mask]
    forces = build_pressure_forces(nodes, inner_faces, geom).reshape(-1)

    centroids = nodes[elems].mean(axis=1)
    surf = copv_surface_projection_np(centroids, geom.mid_radius, geom.cylinder_length, geom.opening_radius)

    return {
        "nodes_np": nodes,
        "elems_np": elems,
        "element_count": int(len(elems)),
        "outer_faces": outer_faces,
        "inner_faces": inner_faces,
        "support_mask": support_mask,
        "n_dof": int(n_dof),
        "elem_dofs": jnp.asarray(elem_dofs),
        "free_dofs": jnp.asarray(free_dofs),
        "volumes": jnp.asarray(volumes),
        "b": jnp.asarray(b),
        "forces_full": jnp.asarray(forces),
        "forces_free": jnp.asarray(forces[free_dofs]),
        "surface_points": jnp.asarray(surf["points"]),
        "surface_normals": jnp.asarray(surf["normals"]),
        "meridian_dirs": jnp.asarray(surf["meridian_dirs"]),
        "hoop_dirs": jnp.asarray(surf["hoop_dirs"]),
        "surface_rho": jnp.asarray(surf["rho"]),
        "s_coords": jnp.asarray(surf["s"]),
        "phi_coords": jnp.asarray(surf["phi"]),
        "c_mat": jnp.asarray(base_material_tensor(material)),
    }


def c4_to_d6(c_tensor: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [jnp.stack([c_tensor[:, i, j, k, l] for k, l in VM], axis=-1) for i, j in VM],
        axis=-2,
    )


def rotate_stiffness_field(c_mat: jnp.ndarray, fiber_dirs: jnp.ndarray, normals: jnp.ndarray) -> jnp.ndarray:
    e3 = normalize_jax(normals)
    e1 = fiber_dirs - jnp.sum(fiber_dirs * e3, axis=-1, keepdims=True) * e3
    e1 = normalize_jax(e1)
    e2 = normalize_jax(jnp.cross(e3, e1))
    r = jnp.stack([e1, e2, e3], axis=-1)
    return jnp.einsum("abcd,eia,ejb,ekc,eld->eijkl", c_mat, r, r, r, r)


def engineering_strain_to_tensor_field(strain: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.stack([strain[..., 0], 0.5 * strain[..., 5], 0.5 * strain[..., 4]], axis=-1),
            jnp.stack([0.5 * strain[..., 5], strain[..., 1], 0.5 * strain[..., 3]], axis=-1),
            jnp.stack([0.5 * strain[..., 4], 0.5 * strain[..., 3], strain[..., 2]], axis=-1),
        ],
        axis=-2,
    )


def stress_voigt_to_tensor_field(stress: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.stack([stress[..., 0], stress[..., 5], stress[..., 4]], axis=-1),
            jnp.stack([stress[..., 5], stress[..., 1], stress[..., 3]], axis=-1),
            jnp.stack([stress[..., 4], stress[..., 3], stress[..., 2]], axis=-1),
        ],
        axis=-2,
    )


def local_frame_from_fiber(fiber_dirs: jnp.ndarray, normals: jnp.ndarray) -> jnp.ndarray:
    e3 = normalize_jax(normals)
    ref_x = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0], dtype=fiber_dirs.dtype), fiber_dirs.shape)
    ref_y = jnp.broadcast_to(jnp.asarray([0.0, 1.0, 0.0], dtype=fiber_dirs.dtype), fiber_dirs.shape)
    ref = jnp.where(jnp.abs(e3[:, :1]) < 0.9, ref_x, ref_y)
    proj = fiber_dirs - jnp.sum(fiber_dirs * e3, axis=-1, keepdims=True) * e3
    proj_norm = jnp.linalg.norm(proj, axis=-1, keepdims=True)
    fallback = normalize_jax(jnp.cross(e3, ref))
    e1 = jnp.where(proj_norm > 1e-8, proj / proj_norm, fallback)
    e2 = normalize_jax(jnp.cross(e3, e1))
    return jnp.stack([e1, e2, e3], axis=-1)


def element_strain_stress(
    state: dict[str, Any],
    displacement: jnp.ndarray,
    c_eff: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    elem_dofs = state["elem_dofs"]
    b = state["b"]
    u_e = displacement[elem_dofs]
    strain = jnp.einsum("eij,ej->ei", b, u_e)
    stress = jnp.einsum("eij,ej->ei", c4_to_d6(c_eff), strain)
    return strain, stress


def hashin_failure_indices(local_stress: jnp.ndarray, allowables: MaterialAllowables) -> dict[str, jnp.ndarray]:
    sigma_11 = local_stress[..., 0, 0]
    # Transverse stress averaged over the isotropic 2-3 plane of the UD ply.
    # For membrane-dominated loading away from the boss, sigma_33 is typically
    # small and this collapses back toward the usual 2D Hashin sigma_22 term.
    # Near the boss under local triaxial constraint this remains a screening
    # approximation rather than a full 3D progressive-damage treatment.
    sigma_transverse = 0.5 * (local_stress[..., 1, 1] + local_stress[..., 2, 2])
    tau_12 = local_stress[..., 0, 1]
    tau_13 = local_stress[..., 0, 2]
    tau_23 = local_stress[..., 1, 2]
    shear_fiber = tau_12**2 + tau_13**2
    shear_matrix = tau_12**2 + tau_23**2

    xt = jnp.asarray(allowables.xt, dtype=local_stress.dtype)
    xc = jnp.asarray(allowables.xc, dtype=local_stress.dtype)
    yt = jnp.asarray(allowables.yt, dtype=local_stress.dtype)
    yc = jnp.asarray(allowables.yc, dtype=local_stress.dtype)
    s = jnp.asarray(allowables.s, dtype=local_stress.dtype)

    fiber_tension = jnp.where(sigma_11 >= 0.0, (sigma_11 / xt) ** 2 + shear_fiber / (s**2), 0.0)
    fiber_compression = jnp.where(sigma_11 < 0.0, (sigma_11 / xc) ** 2, 0.0)
    matrix_tension = jnp.where(sigma_transverse >= 0.0, (sigma_transverse / yt) ** 2 + shear_matrix / (s**2), 0.0)
    matrix_compression_raw = (
        (sigma_transverse / (2.0 * s)) ** 2
        + ((yc / (2.0 * s)) ** 2 - 1.0) * (sigma_transverse / yc)
        + shear_matrix / (s**2)
    )
    matrix_compression = jnp.where(sigma_transverse < 0.0, jnp.maximum(matrix_compression_raw, 0.0), 0.0)
    failure_index = jnp.maximum(
        jnp.maximum(fiber_tension, fiber_compression),
        jnp.maximum(matrix_tension, matrix_compression),
    )
    return {
        "fiber_tension": fiber_tension,
        "fiber_compression": fiber_compression,
        "matrix_tension": matrix_tension,
        "matrix_compression": matrix_compression,
        "failure_index": failure_index,
    }


def hashin_failure_indices_np(local_stress: np.ndarray, allowables: MaterialAllowables) -> dict[str, np.ndarray]:
    local_stress = np.asarray(local_stress, dtype=np.float64)
    sigma_11 = local_stress[..., 0, 0]
    sigma_transverse = 0.5 * (local_stress[..., 1, 1] + local_stress[..., 2, 2])
    tau_12 = local_stress[..., 0, 1]
    tau_13 = local_stress[..., 0, 2]
    tau_23 = local_stress[..., 1, 2]
    shear_fiber = tau_12**2 + tau_13**2
    shear_matrix = tau_12**2 + tau_23**2

    xt = float(allowables.xt)
    xc = float(allowables.xc)
    yt = float(allowables.yt)
    yc = float(allowables.yc)
    s = float(allowables.s)

    fiber_tension = np.where(sigma_11 >= 0.0, (sigma_11 / xt) ** 2 + shear_fiber / (s**2), 0.0)
    fiber_compression = np.where(sigma_11 < 0.0, (sigma_11 / xc) ** 2, 0.0)
    matrix_tension = np.where(sigma_transverse >= 0.0, (sigma_transverse / yt) ** 2 + shear_matrix / (s**2), 0.0)
    matrix_compression_raw = (
        (sigma_transverse / (2.0 * s)) ** 2
        + ((yc / (2.0 * s)) ** 2 - 1.0) * (sigma_transverse / yc)
        + shear_matrix / (s**2)
    )
    matrix_compression = np.where(sigma_transverse < 0.0, np.maximum(matrix_compression_raw, 0.0), 0.0)
    failure_index = np.maximum(
        np.maximum(fiber_tension, fiber_compression),
        np.maximum(matrix_tension, matrix_compression),
    )
    return {
        "fiber_tension": fiber_tension,
        "fiber_compression": fiber_compression,
        "matrix_tension": matrix_tension,
        "matrix_compression": matrix_compression,
        "failure_index": failure_index,
    }


def evaluate_hashin_failure(
    state: dict[str, Any],
    displacement: jnp.ndarray,
    c_eff: jnp.ndarray,
    fiber_dirs: jnp.ndarray,
    failure: FailureConfig,
) -> dict[str, jnp.ndarray]:
    strain_voigt, stress_voigt = element_strain_stress(state, displacement, c_eff)
    stress_tensor = stress_voigt_to_tensor_field(stress_voigt)
    strain_tensor = engineering_strain_to_tensor_field(strain_voigt)
    frame = local_frame_from_fiber(fiber_dirs, state["surface_normals"])
    local_stress = jnp.einsum("eai,eab,ebj->eij", frame, stress_tensor, frame)
    local_strain = jnp.einsum("eai,eab,ebj->eij", frame, strain_tensor, frame)
    metrics = hashin_failure_indices(local_stress, failure.allowables)
    failure_with_margin = metrics["failure_index"] * failure.margin_of_safety
    smooth_excess = jax.nn.softplus(failure.softplus_scale * (failure_with_margin - 1.0)) / failure.softplus_scale
    mean_penalty = jnp.mean(smooth_excess**2)
    flat_excess = jnp.reshape(smooth_excess, (-1,))
    tail_fraction = float(np.clip(failure.penalty_tail_fraction, 0.0, 1.0))
    tail_count = max(1, min(int(np.ceil(tail_fraction * max(int(flat_excess.shape[0]), 1))), int(flat_excess.shape[0])))
    tail_values = jax.lax.top_k(flat_excess, tail_count)[0]
    tail_penalty = jnp.mean(tail_values**2)
    worst_case_mix = float(np.clip(failure.penalty_worst_case_mix, 0.0, 1.0))
    penalty = failure.penalty_weight * ((1.0 - worst_case_mix) * mean_penalty + worst_case_mix * tail_penalty)
    return {
        **metrics,
        "strain_voigt": strain_voigt,
        "stress_voigt": stress_voigt,
        "local_stress": local_stress,
        "local_strain": local_strain,
        "failure_with_margin": failure_with_margin,
        "fi_max": jnp.max(metrics["failure_index"]),
        "fi_mean": jnp.mean(metrics["failure_index"]),
        "penalty": penalty,
        "failure_penalty_mean": mean_penalty,
        "failure_penalty_tail": tail_penalty,
    }


def estimate_burst_pressure_profile(
    local_stress: np.ndarray,
    allowables: MaterialAllowables,
    operating_pressure: float = 1.0,
    margin_of_safety: float = 1.0,
    scale_max: float = 8.0,
    num_points: int = 161,
    growth_factor: float = 2.0,
    max_expansions: int = 8,
) -> dict[str, np.ndarray | float]:
    local_stress = np.asarray(local_stress, dtype=np.float64)
    target_with_margin = 1.0 / max(float(margin_of_safety), 1e-12)
    target_burst = 1.0
    current_scale_max = max(float(scale_max), 1e-6)

    def _curve(max_scale: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        pressure_factors = np.linspace(0.0, max_scale, int(max(num_points, 2)))
        scaled = pressure_factors[:, None, None, None] * local_stress[None, ...]
        metrics = hashin_failure_indices_np(scaled, allowables)
        return pressure_factors, metrics

    pressure_factors, metrics = _curve(current_scale_max)
    failure_curve = np.max(metrics["failure_index"], axis=1)
    expansions = 0
    while failure_curve[-1] < max(target_with_margin, target_burst) and expansions < max_expansions:
        current_scale_max *= max(float(growth_factor), 1.1)
        pressure_factors, metrics = _curve(current_scale_max)
        failure_curve = np.max(metrics["failure_index"], axis=1)
        expansions += 1

    mode_curves = {
        "fiber_tension": np.max(metrics["fiber_tension"], axis=1),
        "fiber_compression": np.max(metrics["fiber_compression"], axis=1),
        "matrix_tension": np.max(metrics["matrix_tension"], axis=1),
        "matrix_compression": np.max(metrics["matrix_compression"], axis=1),
    }

    def _interpolate_crossing(curve: np.ndarray, target: float) -> float:
        idx = np.where(curve >= target)[0]
        if len(idx) == 0:
            return float(pressure_factors[-1])
        hi = int(idx[0])
        if hi == 0:
            return float(pressure_factors[0])
        lo = hi - 1
        x0 = float(pressure_factors[lo])
        x1 = float(pressure_factors[hi])
        y0 = float(curve[lo])
        y1 = float(curve[hi])
        if abs(y1 - y0) < 1e-12:
            return x1
        return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

    burst_factor = _interpolate_crossing(failure_curve, target_burst)
    allowable_factor_with_margin = _interpolate_crossing(failure_curve, target_with_margin)
    burst_index = int(np.argmin(np.abs(pressure_factors - burst_factor)))
    mode_at_burst = {
        "fiber_tension": float(mode_curves["fiber_tension"][burst_index]),
        "fiber_compression": float(mode_curves["fiber_compression"][burst_index]),
        "matrix_tension": float(mode_curves["matrix_tension"][burst_index]),
        "matrix_compression": float(mode_curves["matrix_compression"][burst_index]),
    }

    return {
        "pressure_factors": pressure_factors,
        "pressure_values": operating_pressure * pressure_factors,
        "failure_curve": failure_curve,
        "failure_curve_with_margin": failure_curve * float(margin_of_safety),
        "mode_curves": mode_curves,
        "burst_factor": float(burst_factor),
        "allowable_factor_with_margin": float(allowable_factor_with_margin),
        "burst_pressure": float(operating_pressure * burst_factor),
        "allowable_pressure_with_margin": float(operating_pressure * allowable_factor_with_margin),
        "mode_at_burst": mode_at_burst,
    }


def required_friction_coefficient(
    s_coords: jnp.ndarray,
    rho: jnp.ndarray,
    alpha: jnp.ndarray,
    regularization: float = 1e-6,
) -> jnp.ndarray:
    if s_coords.shape[0] < 2:
        return jnp.zeros((1,), dtype=alpha.dtype)
    # Deviation from Clairaut's theorem, rho * sin(alpha) = const, requires
    # lateral friction to hold the tow on the commanded path. A geodesic path
    # has zero required friction everywhere; values above mu_max are penalized
    # upstream before the winding layout is accepted for export.
    clairaut = rho * jnp.sin(alpha)
    ds = jnp.maximum(jnp.diff(s_coords), regularization)
    dclairaut_ds = jnp.diff(clairaut) / ds
    rho_mid = 0.5 * (rho[1:] + rho[:-1])
    alpha_mid = 0.5 * (alpha[1:] + alpha[:-1])
    denom = jnp.maximum(jnp.abs(rho_mid * jnp.cos(alpha_mid)), regularization)
    return jnp.abs(dclairaut_ds) / denom


def friction_penalty(
    s_coords: jnp.ndarray,
    rho: jnp.ndarray,
    alpha: jnp.ndarray,
    config: FrictionConfig,
) -> dict[str, jnp.ndarray]:
    mu_required = required_friction_coefficient(
        s_coords,
        rho,
        alpha,
        regularization=config.mu_regularization,
    )
    excess = jax.nn.relu(mu_required - config.mu_max)
    penalty = config.penalty_weight * jnp.mean(excess**2)
    return {
        "mu_required": mu_required,
        "mu_max_required": jnp.max(mu_required),
        "penalty": penalty,
    }


def make_solve_compliance(state: dict[str, Any], tol: float = 1e-6, maxiter: int = 2400):
    n_dof = state["n_dof"]
    elem_dofs = state["elem_dofs"]
    free_dofs = state["free_dofs"]
    volumes = state["volumes"]
    b = state["b"]
    forces_free = state["forces_free"]
    forces_full = state["forces_full"]

    @jax.jit
    def solve(c_eff: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        d = c4_to_d6(c_eff)

        def matvec(u_free: jnp.ndarray) -> jnp.ndarray:
            u_full = jnp.zeros((n_dof,), dtype=c_eff.dtype).at[free_dofs].set(u_free)
            u_e = u_full[elem_dofs]
            strain = jnp.einsum("eij,ej->ei", b, u_e)
            stress = jnp.einsum("eij,ej->ei", d, strain)
            fint_e = volumes[:, None] * jnp.einsum("eji,ej->ei", b, stress)
            fint_full = jnp.zeros((n_dof,), dtype=c_eff.dtype).at[elem_dofs.reshape(-1)].add(fint_e.reshape(-1))
            return fint_full[free_dofs]

        u_free, _ = jsp.sparse.linalg.cg(matvec, forces_free, tol=tol, maxiter=maxiter)
        u_full = jnp.zeros((n_dof,), dtype=c_eff.dtype).at[free_dofs].set(u_free)
        compliance = jnp.vdot(u_full, forces_full)
        return compliance, u_full

    return solve


def baseline_response(state: dict[str, Any], material: MaterialConfig, solve_compliance) -> dict[str, jnp.ndarray]:
    element_count = state["element_count"]
    base_thickness = material.base_thickness
    c_base = jnp.broadcast_to(state["c_mat"], (element_count,) + state["c_mat"].shape)
    compliance, displacement = solve_compliance(c_base)
    thickness = jnp.full((element_count,), base_thickness)
    density = jnp.ones((element_count,))
    coverage = jnp.zeros((element_count,))
    fiber_dirs = normalize_jax(state["meridian_dirs"] + 1e-8)
    mass_metric = jnp.sum(thickness * state["volumes"])
    return {
        "compliance": compliance,
        "displacement": displacement,
        "thickness": thickness,
        "density": density,
        "coverage": coverage,
        "fiber_dirs": fiber_dirs,
        "mass_metric": mass_metric,
    }
