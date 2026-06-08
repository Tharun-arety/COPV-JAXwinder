from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxopt
import numpy as np

from .config import FailureConfig, FrictionConfig, GeometryConfig, HybridConfig, IFPConfig, MaterialConfig, PatchConfig, WindingConfig
from .geometry import copv_meridional_metrics, copv_surface_from_sphi_jax, normalize_jax
from .physics import evaluate_hashin_failure, friction_penalty, rotate_stiffness_field


def safe_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(x / (1.0 - x))


def current_jax_real_dtype():
    return jnp.asarray(0.0).dtype


def safe_inverse_sigmoid(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = max(hi - lo, 1e-6)
    scaled = (np.asarray(x, dtype=np.float64) - lo) / span
    return safe_logit(np.clip(scaled, 1e-4, 1.0 - 1e-4))


def triangular_basis(targets: jnp.ndarray, control_points: jnp.ndarray) -> jnp.ndarray:
    control_points = jnp.reshape(control_points, (-1,))
    targets = jnp.reshape(targets, (-1,))
    if control_points.shape[0] == 1:
        return jnp.ones((targets.shape[0], 1), dtype=targets.dtype)
    spacing = jnp.maximum(control_points[1] - control_points[0], 1e-6)
    dist = jnp.abs(targets[:, None] - control_points[None, :])
    weights = jnp.maximum(1.0 - dist / spacing, 0.0)
    weight_sum = jnp.sum(weights, axis=1, keepdims=True)
    nearest = jax.nn.one_hot(jnp.argmin(dist, axis=1), control_points.shape[0], dtype=targets.dtype)
    return jnp.where(weight_sum > 1e-12, weights / weight_sum, nearest)


def rectangular_signed_distance(xi: jnp.ndarray, eta: jnp.ndarray, length: float, width: float) -> jnp.ndarray:
    dx = jnp.abs(xi) - 0.5 * length
    dy = jnp.abs(eta) - 0.5 * width
    outside = jnp.sqrt(jnp.maximum(dx, 0.0) ** 2 + jnp.maximum(dy, 0.0) ** 2 + 1e-12)
    inside = jnp.minimum(0.5 * length - jnp.abs(xi), 0.5 * width - jnp.abs(eta))
    return jnp.where((dx > 0.0) | (dy > 0.0), outside, -inside)


def bernstein_bezier(ctrl: jnp.ndarray, sample_count: int) -> jnp.ndarray:
    t = jnp.linspace(0.0, 1.0, sample_count)
    basis = jnp.stack(
        [
            (1.0 - t) ** 3,
            3.0 * t * (1.0 - t) ** 2,
            3.0 * t**2 * (1.0 - t),
            t**3,
        ],
        axis=-1,
    )
    return basis @ ctrl


def patch_overlap_penalty(weights: jnp.ndarray, stack_target: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    coverage = jnp.sum(weights, axis=0)
    excess = jnp.maximum(coverage - stack_target, 0.0)
    penalty = jnp.mean(excess**2)
    return penalty, coverage, excess


def patch_center_repulsion_penalty(
    s_coords: jnp.ndarray,
    phis: jnp.ndarray,
    config: PatchConfig,
    geom: GeometryConfig,
) -> jnp.ndarray:
    if config.count < 2:
        return jnp.asarray(0.0, dtype=s_coords.dtype)

    ds = s_coords[:, None] - s_coords[None, :]
    dphi = jnp.abs(phis[:, None] - phis[None, :])
    dphi = jnp.minimum(dphi, 2.0 * jnp.pi - dphi)
    arc = geom.mid_radius * dphi
    dist = jnp.sqrt(ds**2 + arc**2 + 1e-12)
    min_spacing = max(config.min_center_spacing_factor * np.hypot(config.length, config.width), 1e-6)
    mask = 1.0 - jnp.eye(config.count, dtype=s_coords.dtype)
    repulsion = jnp.exp(-((dist / min_spacing) ** 2)) * mask
    return jnp.sum(repulsion) / jnp.maximum(jnp.sum(mask), 1.0)


def patch_objective_from_result(
    result: dict[str, jnp.ndarray],
    config: PatchConfig,
    compliance_scale: jnp.ndarray,
) -> jnp.ndarray:
    compliance_term = result["compliance"] / jnp.maximum(compliance_scale, 1.0)
    return (
        compliance_term
        + config.overlap_penalty_weight * result["overlap_penalty"]
        + config.repulsion_penalty_weight * result["repulsion_penalty"]
    )


@jax.checkpoint
def patch_stiffness_contribution(
    weight_e: jnp.ndarray,
    fiber_dir: jnp.ndarray,
    c_mat: jnp.ndarray,
    surface_normals: jnp.ndarray,
) -> jnp.ndarray:
    """
    Build a single patch contribution without materializing the full
    (patch, element, 3, 3, 3, 3) rotated stiffness block.
    """
    c_rot = rotate_stiffness_field(
        c_mat,
        jnp.broadcast_to(fiber_dir, surface_normals.shape),
        surface_normals,
    )
    return weight_e[:, None, None, None, None] * c_rot


def accumulate_patch_stiffness(
    weights: jnp.ndarray,
    fiber_dirs: jnp.ndarray,
    c_mat: jnp.ndarray,
    surface_normals: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate patch stiffness contributions with a scan so the patch path
    scales linearly in memory with patch count.
    """
    element_count = weights.shape[1]
    init = jnp.zeros((element_count,) + c_mat.shape, dtype=c_mat.dtype)

    def body(acc: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, None]:
        weight_e, fiber_dir = inputs
        acc = acc + patch_stiffness_contribution(weight_e, fiber_dir, c_mat, surface_normals)
        return acc, None

    c_add, _ = jax.lax.scan(body, init, (weights, fiber_dirs), unroll=1)
    return c_add


def accumulate_weighted_patch_stiffness(
    weights: jnp.ndarray,
    thicknesses: jnp.ndarray,
    fiber_dirs: jnp.ndarray,
    c_mat: jnp.ndarray,
    surface_normals: jnp.ndarray,
) -> jnp.ndarray:
    element_count = weights.shape[1]
    init = jnp.zeros((element_count,) + c_mat.shape, dtype=c_mat.dtype)

    def body(acc: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, None]:
        weight_e, thickness, fiber_dir = inputs
        acc = acc + patch_stiffness_contribution(weight_e * thickness, fiber_dir, c_mat, surface_normals)
        return acc, None

    c_add, _ = jax.lax.scan(body, init, (weights, thicknesses, fiber_dirs), unroll=1)
    return c_add


def decode_patch_params(raw: jnp.ndarray, config: PatchConfig, geom: GeometryConfig) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    _, _, total_len = copv_meridional_metrics(geom.mid_radius, geom.cylinder_length, geom.opening_radius)
    margin = max(0.5 * np.hypot(config.length, config.width) + 4.0, 8.0)
    span = max(total_len - 2.0 * margin, 1e-6)
    s_u = jax.nn.sigmoid(raw[: config.count])
    phi_u = jax.nn.sigmoid(raw[config.count : 2 * config.count])
    alpha_u = jax.nn.sigmoid(raw[2 * config.count : 3 * config.count])
    s_coords = margin + span * s_u
    phis = 2.0 * jnp.pi * phi_u
    alphas = (alpha_u - 0.5) * jnp.pi
    return s_coords, phis, alphas


def patch_forward(
    raw: jnp.ndarray,
    state: dict[str, Any],
    material: MaterialConfig,
    config: PatchConfig,
    geom: GeometryConfig,
    beta: float,
    solve_compliance,
) -> dict[str, jnp.ndarray]:
    s_coords, phis, alphas = decode_patch_params(raw, config, geom)
    centers, e_s, e_phi, _, _ = copv_surface_from_sphi_jax(
        geom.mid_radius,
        s_coords,
        phis,
        geom.cylinder_length,
        geom.opening_radius,
    )
    ca = jnp.cos(alphas)
    sa = jnp.sin(alphas)
    fiber_dirs = ca[:, None] * e_s + sa[:, None] * e_phi
    perp_dirs = -sa[:, None] * e_s + ca[:, None] * e_phi
    delta = state["surface_points"][None] - centers[:, None]
    xi = jnp.einsum("pei,pi->pe", delta, fiber_dirs)
    eta = jnp.einsum("pei,pi->pe", delta, perp_dirs)
    signed = rectangular_signed_distance(xi, eta, config.length, config.width)
    weights = jax.nn.sigmoid(-beta * signed)
    overlap_penalty, coverage, coverage_excess = patch_overlap_penalty(weights, config.stack_target)
    repulsion_penalty = patch_center_repulsion_penalty(s_coords, phis, config, geom)

    base = material.base_thickness
    added = material.ply_thickness * jnp.sum(weights, axis=0)
    total = jnp.clip(base + added, a_min=base * material.density_floor)
    c_base = jnp.broadcast_to(state["c_mat"], (state["element_count"],) + state["c_mat"].shape)
    c_add = accumulate_patch_stiffness(weights, fiber_dirs, state["c_mat"], state["surface_normals"])
    c_eff = (
        base * c_base
        + material.ply_thickness * c_add
    ) / total[:, None, None, None, None]

    compliance, displacement = solve_compliance(c_eff)
    density = total / base
    fiber_combined = normalize_jax(jnp.einsum("pe,pi->ei", weights, fiber_dirs) + 1e-8)
    mass_metric = jnp.sum(total * state["volumes"])
    return {
        "compliance": compliance,
        "displacement": displacement,
        "weights": weights,
        "coverage": coverage,
        "thickness": total,
        "density": density,
        "fiber_dirs": fiber_combined,
        "s_coords": s_coords,
        "phis": phis,
        "alphas": alphas,
        "mass_metric": mass_metric,
        "overlap_penalty": overlap_penalty,
        "repulsion_penalty": repulsion_penalty,
        "coverage_excess": coverage_excess,
    }


def decode_ifp_ctrl(raw: jnp.ndarray, config: IFPConfig, geom: GeometryConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    _, _, total_len = copv_meridional_metrics(geom.mid_radius, geom.cylinder_length, geom.opening_radius)
    margin = 6.0
    span = max(total_len - 2.0 * margin, 1e-6)
    raw_ctrl = raw.reshape(config.ctrl_count, 2)
    s_ctrl = margin + span * jax.nn.sigmoid(raw_ctrl[:, 0])
    phi_ctrl = 2.0 * jnp.pi * jax.nn.sigmoid(raw_ctrl[:, 1])
    return s_ctrl, phi_ctrl


def ifp_forward(
    raw: jnp.ndarray,
    state: dict[str, Any],
    material: MaterialConfig,
    config: IFPConfig,
    geom: GeometryConfig,
    beta: float,
    solve_compliance,
) -> dict[str, jnp.ndarray]:
    s_ctrl, phi_ctrl = decode_ifp_ctrl(raw, config, geom)
    ctrl = jnp.stack([s_ctrl, phi_ctrl], axis=-1)
    curve = bernstein_bezier(ctrl, config.sample_count)
    curve_s = curve[:, 0]
    curve_phi = curve[:, 1]
    offsets = jnp.linspace(0.0, 2.0 * jnp.pi, config.family_count + 1)[:-1]

    def family_points(offset: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        phi = jnp.mod(curve_phi + offset, 2.0 * jnp.pi)
        pts, _, _, _, _ = copv_surface_from_sphi_jax(
            geom.mid_radius,
            curve_s,
            phi,
            geom.cylinder_length,
            geom.opening_radius,
        )
        tangent_seed = pts[1:2] - pts[:1]
        tangents = jnp.concatenate([tangent_seed, pts[1:] - pts[:-1]], axis=0)
        return pts, normalize_jax(tangents)

    family_pts, family_tangents = jax.vmap(family_points)(offsets)
    flat_pts = family_pts.reshape(-1, 3)
    flat_tangents = family_tangents.reshape(-1, 3)
    dists = jnp.linalg.norm(state["surface_points"][None] - flat_pts[:, None], axis=-1)
    d_min = -config.smooth_tau * jsp.special.logsumexp(-dists / config.smooth_tau, axis=0)
    weights_sample = jax.nn.softmax(-dists / config.smooth_tau, axis=0)
    fiber_dirs = normalize_jax(jnp.einsum("se,si->ei", weights_sample, flat_tangents))
    cover = jax.nn.sigmoid(beta * (0.5 * config.tow_width - d_min))

    c_rot = rotate_stiffness_field(state["c_mat"], fiber_dirs, state["surface_normals"])
    base = material.base_thickness
    total = jnp.clip(base + config.tow_thickness * cover, a_min=base * material.density_floor)
    c_base = jnp.broadcast_to(state["c_mat"], (state["element_count"],) + state["c_mat"].shape)
    c_eff = (
        base * c_base
        + (config.tow_thickness * cover)[:, None, None, None, None] * c_rot
    ) / total[:, None, None, None, None]

    compliance, displacement = solve_compliance(c_eff)
    density = total / base
    mass_metric = jnp.sum(total * state["volumes"])
    return {
        "compliance": compliance,
        "displacement": displacement,
        "coverage": cover,
        "thickness": total,
        "density": density,
        "fiber_dirs": fiber_dirs,
        "curve_s": curve_s,
        "curve_phi": curve_phi,
        "ctrl_s": s_ctrl,
        "ctrl_phi": phi_ctrl,
        "mass_metric": mass_metric,
    }


def winding_forward_angle(
    angle_deg: float,
    state: dict[str, Any],
    material: MaterialConfig,
    config: WindingConfig,
    geom: GeometryConfig,
    solve_compliance,
) -> dict[str, jnp.ndarray]:
    alpha_cyl = jnp.deg2rad(angle_deg)
    clairaut_radius = geom.mid_radius * jnp.sin(alpha_cyl)
    rho_floor = max(geom.opening_radius + 4.0, 0.18 * geom.mid_radius)
    rho = jnp.linalg.norm(state["surface_points"][:, :2], axis=-1).clip(min=rho_floor)
    alpha_profile = jnp.arcsin(jnp.clip(clairaut_radius / rho, -0.98, 0.98))
    fiber_dirs = jnp.cos(alpha_profile)[:, None] * state["meridian_dirs"] + jnp.sin(alpha_profile)[:, None] * state["hoop_dirs"]
    fiber_dirs = fiber_dirs - jnp.sum(fiber_dirs * state["surface_normals"], axis=-1, keepdims=True) * state["surface_normals"]
    fiber_dirs = normalize_jax(fiber_dirs)

    base = material.base_thickness
    total = jnp.clip(base + config.band_thickness * jnp.ones_like(rho), a_min=base * material.density_floor)
    c_rot = rotate_stiffness_field(state["c_mat"], fiber_dirs, state["surface_normals"])
    c_base = jnp.broadcast_to(state["c_mat"], (state["element_count"],) + state["c_mat"].shape)
    c_eff = (base * c_base + config.band_thickness * c_rot) / total[:, None, None, None, None]

    compliance, displacement = solve_compliance(c_eff)
    density = total / base
    coverage = jnp.ones_like(rho)
    mass_metric = jnp.sum(total * state["volumes"])
    return {
        "compliance": compliance,
        "displacement": displacement,
        "coverage": coverage,
        "thickness": total,
        "density": density,
        "fiber_dirs": fiber_dirs,
        "alpha_cyl": alpha_cyl,
        "clairaut_radius": clairaut_radius,
        "mass_metric": mass_metric,
    }


def hybrid_patch_overlap_penalty(
    weights: jnp.ndarray,
    thicknesses: jnp.ndarray,
    material: MaterialConfig,
    config: HybridConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    thickness_scale = thicknesses[:, None] / max(material.ply_thickness, 1e-6)
    coverage = jnp.sum(weights * thickness_scale, axis=0)
    excess = jnp.maximum(coverage - config.stack_target, 0.0)
    penalty = jnp.mean(excess**2)
    return penalty, coverage, excess


def hybrid_patch_center_repulsion_penalty(
    s_coords: jnp.ndarray,
    phis: jnp.ndarray,
    config: HybridConfig,
    geom: GeometryConfig,
) -> jnp.ndarray:
    if config.patch_count < 2:
        return jnp.asarray(0.0, dtype=s_coords.dtype)
    ds = s_coords[:, None] - s_coords[None, :]
    dphi = jnp.abs(phis[:, None] - phis[None, :])
    dphi = jnp.minimum(dphi, 2.0 * jnp.pi - dphi)
    arc = geom.mid_radius * dphi
    dist = jnp.sqrt(ds**2 + arc**2 + 1e-12)
    min_spacing = max(config.min_center_spacing_factor * np.hypot(config.patch_length, config.patch_width), 1e-6)
    mask = 1.0 - jnp.eye(config.patch_count, dtype=s_coords.dtype)
    repulsion = jnp.exp(-((dist / min_spacing) ** 2)) * mask
    return jnp.sum(repulsion) / jnp.maximum(jnp.sum(mask), 1.0)


def decode_hybrid_params(raw: jnp.ndarray, config: HybridConfig, geom: GeometryConfig) -> dict[str, jnp.ndarray]:
    winding_n = config.winding_ctrl_count
    patch_n = config.patch_count
    theta_open, cap_len, total_len = copv_meridional_metrics(geom.mid_radius, geom.cylinder_length, geom.opening_radius)
    del theta_open, cap_len
    margin = max(0.5 * np.hypot(config.patch_length, config.patch_width) + 4.0, 8.0)
    span = max(total_len - 2.0 * margin, 1e-6)

    angle_lo = np.deg2rad(config.min_angle_deg)
    angle_hi = np.deg2rad(config.max_angle_deg)
    angle_ctrl = angle_lo + (angle_hi - angle_lo) * jax.nn.sigmoid(raw[:winding_n])
    thickness_ctrl = config.max_winding_thickness * jax.nn.sigmoid(raw[winding_n : 2 * winding_n])
    s_ctrl = jnp.linspace(margin, total_len - margin, winding_n)

    patch_offset = 2 * winding_n
    if patch_n > 0:
        patch_s_raw = raw[patch_offset : patch_offset + patch_n]
        patch_phi_raw = raw[patch_offset + patch_n : patch_offset + 2 * patch_n]
        patch_alpha_raw = raw[patch_offset + 2 * patch_n : patch_offset + 3 * patch_n]
        patch_thickness_raw = raw[patch_offset + 3 * patch_n : patch_offset + 4 * patch_n]
        patch_s = margin + span * jax.nn.sigmoid(patch_s_raw)
        patch_phi = 2.0 * jnp.pi * jax.nn.sigmoid(patch_phi_raw)
        patch_alpha = (jax.nn.sigmoid(patch_alpha_raw) - 0.5) * jnp.pi
        patch_thickness = config.max_patch_thickness * jax.nn.sigmoid(patch_thickness_raw)
    else:
        patch_s = jnp.zeros((0,), dtype=raw.dtype)
        patch_phi = jnp.zeros((0,), dtype=raw.dtype)
        patch_alpha = jnp.zeros((0,), dtype=raw.dtype)
        patch_thickness = jnp.zeros((0,), dtype=raw.dtype)

    return {
        "winding_s_ctrl": s_ctrl,
        "winding_angle_ctrl": angle_ctrl,
        "winding_thickness_ctrl": thickness_ctrl,
        "patch_s": patch_s,
        "patch_phi": patch_phi,
        "patch_alpha": patch_alpha,
        "patch_thickness": patch_thickness,
    }


def hybrid_forward(
    raw: jnp.ndarray,
    state: dict[str, Any],
    material: MaterialConfig,
    config: HybridConfig,
    geom: GeometryConfig,
    solve_compliance,
    beta: float,
    failure_config: FailureConfig | None = None,
    friction_config: FrictionConfig | None = None,
) -> dict[str, jnp.ndarray]:
    failure = FailureConfig() if failure_config is None else failure_config
    friction = FrictionConfig() if friction_config is None else friction_config
    decoded = decode_hybrid_params(raw, config, geom)
    winding_s_ctrl = decoded["winding_s_ctrl"]
    winding_angle_ctrl = decoded["winding_angle_ctrl"]
    winding_thickness_ctrl = decoded["winding_thickness_ctrl"]
    patch_s = decoded["patch_s"]
    patch_phi = decoded["patch_phi"]
    patch_alpha = decoded["patch_alpha"]
    patch_thickness = decoded["patch_thickness"]

    winding_basis = triangular_basis(state["s_coords"], winding_s_ctrl)
    winding_angle = winding_basis @ winding_angle_ctrl
    winding_thickness = winding_basis @ winding_thickness_ctrl
    winding_dirs = (
        jnp.cos(winding_angle)[:, None] * state["meridian_dirs"]
        + jnp.sin(winding_angle)[:, None] * state["hoop_dirs"]
    )
    winding_dirs = winding_dirs - jnp.sum(winding_dirs * state["surface_normals"], axis=-1, keepdims=True) * state["surface_normals"]
    winding_dirs = normalize_jax(winding_dirs)

    _, _, _, _, rho_ctrl = copv_surface_from_sphi_jax(
        geom.mid_radius,
        winding_s_ctrl,
        jnp.zeros_like(winding_s_ctrl),
        geom.cylinder_length,
        geom.opening_radius,
    )
    friction_metrics = friction_penalty(winding_s_ctrl, rho_ctrl, winding_angle_ctrl, friction)

    if config.patch_count > 0:
        centers, e_s, e_phi, _, _ = copv_surface_from_sphi_jax(
            geom.mid_radius,
            patch_s,
            patch_phi,
            geom.cylinder_length,
            geom.opening_radius,
        )
        ca = jnp.cos(patch_alpha)
        sa = jnp.sin(patch_alpha)
        patch_dirs = ca[:, None] * e_s + sa[:, None] * e_phi
        perp_dirs = -sa[:, None] * e_s + ca[:, None] * e_phi
        delta = state["surface_points"][None] - centers[:, None]
        xi = jnp.einsum("pei,pi->pe", delta, patch_dirs)
        eta = jnp.einsum("pei,pi->pe", delta, perp_dirs)
        signed = rectangular_signed_distance(xi, eta, config.patch_length, config.patch_width)
        patch_weights = jax.nn.sigmoid(-beta * signed)
        overlap_penalty, patch_coverage, coverage_excess = hybrid_patch_overlap_penalty(
            patch_weights,
            patch_thickness,
            material,
            config,
        )
        repulsion_penalty = hybrid_patch_center_repulsion_penalty(patch_s, patch_phi, config, geom)
        patch_added_thickness = jnp.einsum("pe,p->e", patch_weights, patch_thickness)
        patch_stiffness = accumulate_weighted_patch_stiffness(
            patch_weights,
            patch_thickness,
            patch_dirs,
            state["c_mat"],
            state["surface_normals"],
        )
        patch_fiber_field = jnp.einsum("pe,p,pi->ei", patch_weights, patch_thickness, patch_dirs)
    else:
        patch_dirs = jnp.zeros((0, 3), dtype=raw.dtype)
        patch_weights = jnp.zeros((0, state["element_count"]), dtype=raw.dtype)
        patch_coverage = jnp.zeros((state["element_count"],), dtype=raw.dtype)
        coverage_excess = jnp.zeros((state["element_count"],), dtype=raw.dtype)
        overlap_penalty = jnp.asarray(0.0, dtype=raw.dtype)
        repulsion_penalty = jnp.asarray(0.0, dtype=raw.dtype)
        patch_added_thickness = jnp.zeros((state["element_count"],), dtype=raw.dtype)
        patch_stiffness = jnp.zeros((state["element_count"],) + state["c_mat"].shape, dtype=raw.dtype)
        patch_fiber_field = jnp.zeros((state["element_count"], 3), dtype=raw.dtype)

    base = material.base_thickness
    c_base = jnp.broadcast_to(state["c_mat"], (state["element_count"],) + state["c_mat"].shape)
    winding_stiffness = rotate_stiffness_field(state["c_mat"], winding_dirs, state["surface_normals"])
    total_thickness = jnp.clip(
        base + winding_thickness + patch_added_thickness,
        a_min=base * material.density_floor,
    )
    c_eff = (
        base * c_base
        + winding_thickness[:, None, None, None, None] * winding_stiffness
        + patch_stiffness
    ) / total_thickness[:, None, None, None, None]

    compliance, displacement = solve_compliance(c_eff)
    density = total_thickness / base
    coverage = (winding_thickness + patch_added_thickness) / max(material.ply_thickness, 1e-6)
    effective_fiber_dirs = normalize_jax(
        winding_thickness[:, None] * winding_dirs
        + patch_fiber_field
        + 1e-8 * state["meridian_dirs"]
    )
    failure_metrics = evaluate_hashin_failure(
        state,
        displacement,
        c_eff,
        effective_fiber_dirs,
        failure,
    )
    mass_metric = jnp.sum(total_thickness * state["volumes"])
    baseline_mass = base * jnp.sum(state["volumes"])
    patch_l1 = jnp.sum(patch_thickness) / max(material.ply_thickness, 1e-6)
    objective = (
        config.mass_weight * (mass_metric / jnp.maximum(baseline_mass, 1e-6))
        + config.patch_l1_weight * patch_l1
        + config.overlap_penalty_weight * overlap_penalty
        + config.repulsion_penalty_weight * repulsion_penalty
        + failure_metrics["penalty"]
        + friction_metrics["penalty"]
    )

    return {
        "objective": objective,
        "compliance": compliance,
        "displacement": displacement,
        "thickness": total_thickness,
        "density": density,
        "coverage": coverage,
        "fiber_dirs": effective_fiber_dirs,
        "c_eff": c_eff,
        "mass_metric": mass_metric,
        "winding_s_ctrl": winding_s_ctrl,
        "winding_angle_ctrl": winding_angle_ctrl,
        "winding_thickness_ctrl": winding_thickness_ctrl,
        "winding_angle_field": winding_angle,
        "winding_thickness_field": winding_thickness,
        "patch_s": patch_s,
        "patch_phi": patch_phi,
        "patch_alpha": patch_alpha,
        "patch_thickness": patch_thickness,
        "patch_added_thickness": patch_added_thickness,
        "patch_fiber_dirs": patch_dirs,
        "patch_weights": patch_weights,
        "patch_coverage": patch_coverage,
        "coverage_excess": coverage_excess,
        "patch_l1": patch_l1,
        "overlap_penalty": overlap_penalty,
        "repulsion_penalty": repulsion_penalty,
        "friction_penalty": friction_metrics["penalty"],
        "mu_required": friction_metrics["mu_required"],
        "mu_max_required": friction_metrics["mu_max_required"],
        "failure_penalty": failure_metrics["penalty"],
        "fiber_tension": failure_metrics["fiber_tension"],
        "fiber_compression": failure_metrics["fiber_compression"],
        "matrix_tension": failure_metrics["matrix_tension"],
        "matrix_compression": failure_metrics["matrix_compression"],
        "failure_index": failure_metrics["failure_index"],
        "failure_with_margin": failure_metrics["failure_with_margin"],
        "strain_voigt": failure_metrics["strain_voigt"],
        "stress_voigt": failure_metrics["stress_voigt"],
        "local_stress": failure_metrics["local_stress"],
        "local_strain": failure_metrics["local_strain"],
        "fi_max": failure_metrics["fi_max"],
        "fi_mean": failure_metrics["fi_mean"],
    }


def initial_patch_params(config: PatchConfig) -> jnp.ndarray:
    idx = np.arange(config.count, dtype=np.float64)
    s_seed = (idx + 0.5) / max(config.count, 1)
    phi_seed = np.mod(0.5 + idx * ((np.sqrt(5.0) - 1.0) / 2.0), 1.0)
    alpha_seed = np.mod(0.25 + idx * (1.0 / np.pi), 1.0)
    seed = np.stack([s_seed, phi_seed, alpha_seed], axis=-1)
    if config.init_jitter > 0.0:
        rng = np.random.RandomState(config.seed)
        jitter = rng.uniform(-config.init_jitter, config.init_jitter, size=seed.shape)
        seed = np.clip(seed + jitter, 1e-3, 1.0 - 1e-3)
    return jnp.asarray(
        np.concatenate([safe_logit(seed[:, 0]), safe_logit(seed[:, 1]), safe_logit(seed[:, 2])]),
        dtype=current_jax_real_dtype(),
    )


def initial_ifp_params(config: IFPConfig) -> jnp.ndarray:
    seed_points = np.asarray(config.seed_points, dtype=np.float64)
    if seed_points.shape != (config.ctrl_count, 2):
        raise ValueError("seed_points must match ctrl_count x 2")
    return jnp.asarray(safe_logit(seed_points).reshape(-1), dtype=current_jax_real_dtype())


def initial_hybrid_params(config: HybridConfig) -> jnp.ndarray:
    rng = np.random.RandomState(config.seed)
    winding_angle = np.full((config.winding_ctrl_count,), config.winding_seed_angle_deg, dtype=np.float64)
    winding_thickness = np.full((config.winding_ctrl_count,), config.winding_seed_thickness, dtype=np.float64)
    params = [
        safe_inverse_sigmoid(winding_angle, config.min_angle_deg, config.max_angle_deg),
        safe_inverse_sigmoid(winding_thickness, 0.0, config.max_winding_thickness),
    ]
    if config.patch_count > 0:
        idx = np.arange(config.patch_count, dtype=np.float64)
        patch_s = (idx + 0.5) / max(config.patch_count, 1)
        patch_phi = np.mod(0.5 + idx * ((np.sqrt(5.0) - 1.0) / 2.0), 1.0)
        patch_alpha = np.full((config.patch_count,), 0.5, dtype=np.float64)
        if config.init_jitter > 0.0:
            jitter = rng.uniform(-config.init_jitter, config.init_jitter, size=(config.patch_count, 3))
            patch_seed = np.stack([patch_s, patch_phi, patch_alpha], axis=-1)
            patch_seed = np.clip(patch_seed + jitter, 1e-3, 1.0 - 1e-3)
            patch_s, patch_phi, patch_alpha = patch_seed[:, 0], patch_seed[:, 1], patch_seed[:, 2]
        patch_thickness = np.full((config.patch_count,), config.patch_seed_thickness, dtype=np.float64)
        params.extend(
            [
                safe_logit(patch_s),
                safe_logit(patch_phi),
                safe_logit(patch_alpha),
                safe_inverse_sigmoid(patch_thickness, 0.0, config.max_patch_thickness),
            ]
        )
    return jnp.asarray(np.concatenate(params), dtype=current_jax_real_dtype())


def run_lbfgs(loss_fn, params0: jnp.ndarray, maxiter: int, tol: float, history_size: int) -> tuple[jnp.ndarray, dict[str, Any]]:
    solver = jaxopt.LBFGS(
        fun=loss_fn,
        maxiter=maxiter,
        tol=tol,
        history_size=history_size,
        implicit_diff=False,
        jit=True,
        verbose=False,
    )
    out = solver.run(params0)
    state = out.state
    info = {
        "iterations": int(np.asarray(jax.device_get(state.iter_num))),
        "loss": float(np.asarray(jax.device_get(state.value))),
        "error": float(np.asarray(jax.device_get(state.error))),
        "failed_linesearch": bool(np.asarray(jax.device_get(state.failed_linesearch))),
        "num_fun_eval": int(np.asarray(jax.device_get(state.num_fun_eval))),
        "num_grad_eval": int(np.asarray(jax.device_get(state.num_grad_eval))),
    }
    return out.params, info


def run_patch_optimization(
    state: dict[str, Any],
    material: MaterialConfig,
    config: PatchConfig,
    geom: GeometryConfig,
    solve_compliance,
    params0: jnp.ndarray | None = None,
) -> dict[str, Any]:
    params = initial_patch_params(config) if params0 is None else params0
    history: list[dict[str, Any]] = []
    result = None
    c_base = jnp.broadcast_to(state["c_mat"], (state["element_count"],) + state["c_mat"].shape)
    compliance_scale, _ = solve_compliance(c_base)
    for beta in config.beta_schedule:
        loss_fn = lambda raw, beta=beta: patch_objective_from_result(
            patch_forward(raw, state, material, config, geom, beta, solve_compliance),
            config,
            compliance_scale,
        )
        params, info = run_lbfgs(loss_fn, params, config.lbfgs_maxiter, config.lbfgs_tol, config.history_size)
        result = patch_forward(params, state, material, config, geom, beta, solve_compliance)
        objective = patch_objective_from_result(result, config, compliance_scale)
        history.append(
            {
                "beta": float(beta),
                "strain_energy": float(np.asarray(jax.device_get(result["compliance"]))),
                "mass_metric": float(np.asarray(jax.device_get(result["mass_metric"]))),
                "objective": float(np.asarray(jax.device_get(objective))),
                "overlap_penalty": float(np.asarray(jax.device_get(result["overlap_penalty"]))),
                "repulsion_penalty": float(np.asarray(jax.device_get(result["repulsion_penalty"]))),
                "max_coverage": float(np.asarray(jax.device_get(jnp.max(result["coverage"])))),
                "max_overlap_excess": float(np.asarray(jax.device_get(jnp.max(result["coverage_excess"])))),
                **info,
            }
        )
    if result is None:
        raise RuntimeError("Patch optimisation did not run")
    return {"params": params, "history": history, "result": result}


def run_ifp_optimization(
    state: dict[str, Any],
    material: MaterialConfig,
    config: IFPConfig,
    geom: GeometryConfig,
    solve_compliance,
    params0: jnp.ndarray | None = None,
) -> dict[str, Any]:
    params = initial_ifp_params(config) if params0 is None else params0
    history: list[dict[str, Any]] = []
    result = None
    for beta in config.beta_schedule:
        loss_fn = lambda raw, beta=beta: ifp_forward(raw, state, material, config, geom, beta, solve_compliance)["compliance"]
        params, info = run_lbfgs(loss_fn, params, config.lbfgs_maxiter, config.lbfgs_tol, config.history_size)
        result = ifp_forward(params, state, material, config, geom, beta, solve_compliance)
        history.append(
            {
                "beta": float(beta),
                "strain_energy": float(np.asarray(jax.device_get(result["compliance"]))),
                "mass_metric": float(np.asarray(jax.device_get(result["mass_metric"]))),
                **info,
            }
        )
    if result is None:
        raise RuntimeError("IFP optimisation did not run")
    return {"params": params, "history": history, "result": result}


def run_winding_angle_sweep(
    state: dict[str, Any],
    material: MaterialConfig,
    config: WindingConfig,
    geom: GeometryConfig,
    solve_compliance,
) -> dict[str, Any]:
    angle_grid = np.linspace(config.min_angle_deg, config.max_angle_deg, config.angle_count)
    history: list[dict[str, Any]] = []
    best_result = None
    best_angle_deg = None
    best_energy = float("inf")
    for angle_deg in angle_grid:
        result = winding_forward_angle(float(angle_deg), state, material, config, geom, solve_compliance)
        compliance = float(np.asarray(jax.device_get(result["compliance"])))
        mass_metric = float(np.asarray(jax.device_get(result["mass_metric"])))
        history.append(
            {
                "angle_deg": float(angle_deg),
                "strain_energy": compliance,
                "mass_metric": mass_metric,
            }
        )
        if np.isfinite(compliance) and compliance < best_energy:
            best_energy = compliance
            best_angle_deg = float(angle_deg)
            best_result = result
    if best_result is None or best_angle_deg is None:
        raise RuntimeError("Winding sweep did not produce a finite result")
    return {
        "angle_deg": best_angle_deg,
        "history": history,
        "result": best_result,
    }


def run_hybrid_optimization(
    state: dict[str, Any],
    material: MaterialConfig,
    config: HybridConfig,
    geom: GeometryConfig,
    solve_compliance,
    failure_config: FailureConfig | None = None,
    friction_config: FrictionConfig | None = None,
    params0: jnp.ndarray | None = None,
) -> dict[str, Any]:
    failure = FailureConfig() if failure_config is None else failure_config
    friction = FrictionConfig() if friction_config is None else friction_config
    params = initial_hybrid_params(config) if params0 is None else params0
    history: list[dict[str, Any]] = []
    result = None
    for beta in config.beta_schedule:
        loss_fn = lambda raw, beta=beta: hybrid_forward(
            raw,
            state,
            material,
            config,
            geom,
            solve_compliance,
            beta=beta,
            failure_config=failure,
            friction_config=friction,
        )["objective"]
        params, info = run_lbfgs(loss_fn, params, config.lbfgs_maxiter, config.lbfgs_tol, config.history_size)
        result = hybrid_forward(
            params,
            state,
            material,
            config,
            geom,
            solve_compliance,
            beta=beta,
            failure_config=failure,
            friction_config=friction,
        )
        history.append(
            {
                "beta": float(beta),
                "objective": float(np.asarray(jax.device_get(result["objective"]))),
                "strain_energy": float(np.asarray(jax.device_get(result["compliance"]))),
                "mass_metric": float(np.asarray(jax.device_get(result["mass_metric"]))),
                "fi_max": float(np.asarray(jax.device_get(result["fi_max"]))),
                "fi_mean": float(np.asarray(jax.device_get(result["fi_mean"]))),
                "mu_max_required": float(np.asarray(jax.device_get(result["mu_max_required"]))),
                "patch_l1": float(np.asarray(jax.device_get(result["patch_l1"]))),
                "overlap_penalty": float(np.asarray(jax.device_get(result["overlap_penalty"]))),
                "repulsion_penalty": float(np.asarray(jax.device_get(result["repulsion_penalty"]))),
                "failure_penalty": float(np.asarray(jax.device_get(result["failure_penalty"]))),
                "friction_penalty": float(np.asarray(jax.device_get(result["friction_penalty"]))),
                "active_patch_count": int(
                    np.asarray(jax.device_get(jnp.sum(result["patch_thickness"] > 0.05 * config.max_patch_thickness)))
                )
                if config.patch_count > 0
                else 0,
                "max_patch_thickness": float(
                    np.asarray(jax.device_get(jnp.max(result["patch_thickness"])))
                )
                if config.patch_count > 0
                else 0.0,
                "max_winding_thickness": float(np.asarray(jax.device_get(jnp.max(result["winding_thickness_field"])))),
                **info,
            }
        )
    if result is None:
        raise RuntimeError("Hybrid optimisation did not run")
    return {"params": params, "history": history, "result": result}
