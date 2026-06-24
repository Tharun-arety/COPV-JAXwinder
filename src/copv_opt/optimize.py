from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from scipy import optimize as scipy_optimize

try:
    # jaxopt L-BFGS is used when available; scipy.optimize.minimize is the fallback.
    # jaxopt is deprecated upstream — migrate to jax.scipy.optimize or optax when stable.
    import jaxopt
except ImportError:
    jaxopt = None

from .config import FailureConfig, FrictionConfig, GeometryConfig, HybridConfig, MaterialConfig
from .geometry import copv_meridional_metrics, copv_surface_from_sphi_jax, normalize_jax
from .physics import evaluate_hashin_failure, friction_penalty, required_friction_coefficient, rotate_stiffness_field

if TYPE_CHECKING:
    from .config import IFPConfig, PatchConfig, WindingConfig


def safe_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(x / (1.0 - x))


def current_jax_real_dtype():
    return jnp.asarray(0.0).dtype


def active_patch_threshold(config: PatchConfig | HybridConfig) -> float:
    # Treat sub-0.05 mm patches as effectively pruned in summaries/plots.
    return max(0.05, 0.02 * float(config.max_patch_thickness))


def count_active_patches_np(patch_thickness: np.ndarray, config: PatchConfig | HybridConfig) -> int:
    threshold = active_patch_threshold(config)
    return int(np.sum(np.asarray(patch_thickness, dtype=np.float64) > threshold))


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


def cosine_spaced_segment(start: jnp.ndarray, stop: jnp.ndarray, count: int) -> jnp.ndarray:
    if count <= 1:
        midpoint = 0.5 * (start + stop)
        return jnp.asarray([midpoint], dtype=jnp.asarray(start).dtype)
    samples = jnp.linspace(0.0, 1.0, count, dtype=jnp.asarray(start).dtype)
    return start + (stop - start) * 0.5 * (1.0 - jnp.cos(jnp.pi * samples))


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
    _, _, total_len = copv_meridional_metrics(
        geom.mid_radius,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
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
        geom.dome_height_ratio,
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
    total = jnp.clip(base + added, base * material.density_floor)
    c_base = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    c_add = accumulate_patch_stiffness(weights, fiber_dirs, state["c_mat"], state["surface_normals"])
    c_eff = (
        base * c_base
        + material.ply_thickness * c_add
    ) / total[:, None, None, None, None]

    compliance, displacement = solve_compliance(c_eff, total)
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
    _, _, total_len = copv_meridional_metrics(
        geom.mid_radius,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
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
            geom.dome_height_ratio,
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
    total = jnp.clip(base + config.tow_thickness * cover, base * material.density_floor)
    c_base = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    c_eff = (
        base * c_base
        + (config.tow_thickness * cover)[:, None, None, None, None] * c_rot
    ) / total[:, None, None, None, None]

    compliance, displacement = solve_compliance(c_eff, total)
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
    # Clairaut minimum reach for this cylindrical winding angle is
    # rho_min = R_mid * sin(alpha_cyl). The boss aperture lies inside that
    # exclusion zone for the shallow helical families used here, so the arcsin
    # argument is clipped before the boss region and those cells are left to
    # the hoop-pass field instead of forcing an unphysical helical path.
    rho_floor = max(geom.opening_radius + 4.0, 0.18 * geom.mid_radius)
    rho = jnp.linalg.norm(state["surface_points"][:, :2], axis=-1).clip(min=rho_floor)
    alpha_profile = jnp.arcsin(jnp.clip(clairaut_radius / rho, -0.98, 0.98))
    fiber_dirs = jnp.cos(alpha_profile)[:, None] * state["meridian_dirs"] + jnp.sin(alpha_profile)[:, None] * state["hoop_dirs"]
    fiber_dirs = fiber_dirs - jnp.sum(fiber_dirs * state["surface_normals"], axis=-1, keepdims=True) * state["surface_normals"]
    fiber_dirs = normalize_jax(fiber_dirs)

    base = material.base_thickness
    total = jnp.clip(base + config.band_thickness * jnp.ones_like(rho), base * material.density_floor)
    c_rot = rotate_stiffness_field(state["c_mat"], fiber_dirs, state["surface_normals"])
    c_base = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    c_eff = (base * c_base + config.band_thickness * c_rot) / total[:, None, None, None, None]

    compliance, displacement = solve_compliance(c_eff, total)
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
    theta_open, cap_len, total_len = copv_meridional_metrics(
        geom.mid_radius,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
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


def hybrid_friction_samples(
    winding_s_ctrl: jnp.ndarray,
    winding_angle_ctrl: jnp.ndarray,
    geom: GeometryConfig,
    config: HybridConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    _, cap_len, total_len = copv_meridional_metrics(
        geom.mid_radius,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    start_s = winding_s_ctrl[0]
    end_s = winding_s_ctrl[-1]
    top_stop = jnp.clip(cap_len, start_s, end_s)
    bottom_start = jnp.clip(cap_len + geom.cylinder_length, start_s, end_s)

    top_samples = cosine_spaced_segment(start_s, top_stop, config.friction_cap_sample_count)
    cylinder_count = max(int(config.friction_cylinder_sample_count), 2)
    cylinder_samples = jnp.linspace(top_stop, bottom_start, cylinder_count, dtype=winding_s_ctrl.dtype)
    bottom_samples = cosine_spaced_segment(bottom_start, end_s, config.friction_cap_sample_count)
    s_eval = jnp.concatenate([top_samples, cylinder_samples[1:-1], bottom_samples[1:]])

    winding_basis = triangular_basis(s_eval, winding_s_ctrl)
    winding_angle_eval = winding_basis @ winding_angle_ctrl
    _, _, _, _, rho_eval = copv_surface_from_sphi_jax(
        geom.mid_radius,
        s_eval,
        jnp.zeros_like(s_eval),
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    return s_eval, rho_eval, winding_angle_eval


def smooth_window(
    s_coords: jnp.ndarray,
    start: jnp.ndarray,
    stop: jnp.ndarray,
    transition: float,
    beta: float = 1.0,
) -> jnp.ndarray:
    width = max(float(transition), 1e-6) / max(float(beta), 1e-6)
    width_t = jnp.asarray(width, dtype=s_coords.dtype)
    return jax.nn.sigmoid((s_coords - start) / width_t) * jax.nn.sigmoid((stop - s_coords) / width_t)


def control_smoothness_penalty(values: jnp.ndarray, control_s: jnp.ndarray) -> jnp.ndarray:
    if values.shape[0] < 2:
        return jnp.asarray(0.0, dtype=values.dtype)
    span = jnp.maximum(control_s[-1] - control_s[0], 1e-6)
    normalized_ds = jnp.maximum(jnp.diff(control_s) / span, 1e-6)
    slope = jnp.diff(values) / normalized_ds
    return jnp.mean(slope**2)


def decode_winding_process_params(raw: jnp.ndarray, config: HybridConfig, geom: GeometryConfig) -> dict[str, jnp.ndarray]:
    winding_n = int(config.winding_ctrl_count)
    _, _, total_len = copv_meridional_metrics(
        geom.mid_radius,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    control_s = jnp.linspace(0.0, total_len, winding_n, dtype=raw.dtype)

    angle_lo = np.deg2rad(config.min_angle_deg)
    angle_hi = np.deg2rad(config.max_angle_deg)
    angle_ctrl = angle_lo + (angle_hi - angle_lo) * jax.nn.sigmoid(raw[:winding_n])

    helical_raw = raw[winding_n : 2 * winding_n]
    hoop_raw = raw[2 * winding_n : 3 * winding_n]
    helical_pass_ctrl = config.max_helical_pass_count * jax.nn.sigmoid(helical_raw)
    hoop_pass_ctrl = config.max_hoop_pass_count * jax.nn.sigmoid(hoop_raw)

    return {
        "winding_s_ctrl": control_s,
        "winding_angle_ctrl": angle_ctrl,
        "helical_pass_ctrl": helical_pass_ctrl,
        "hoop_pass_ctrl": hoop_pass_ctrl,
    }


def helical_deposition_factor(
    rho: jnp.ndarray,
    angle: jnp.ndarray,
    config: HybridConfig,
) -> jnp.ndarray:
    rho_safe = jnp.maximum(rho, 1e-6)
    cos_alpha = jnp.maximum(jnp.cos(angle), 1e-3)
    tow_width = max(float(config.tow_width), 1e-6)
    family_count = max(int(config.winding_family_count), 1)
    return (family_count * tow_width) / (2.0 * jnp.pi * rho_safe * cos_alpha)


def winding_process_friction_metrics(
    winding_s_ctrl: jnp.ndarray,
    winding_angle_ctrl: jnp.ndarray,
    helical_pass_ctrl: jnp.ndarray,
    geom: GeometryConfig,
    config: HybridConfig,
    friction: FrictionConfig,
) -> dict[str, jnp.ndarray]:
    s_eval, rho_eval, angle_eval = hybrid_friction_samples(winding_s_ctrl, winding_angle_ctrl, geom, config)
    mu_required = required_friction_coefficient(
        s_eval,
        rho_eval,
        angle_eval,
        regularization=friction.mu_regularization,
    )
    pass_basis = triangular_basis(s_eval, winding_s_ctrl)
    helical_pass_eval = pass_basis @ helical_pass_ctrl
    if helical_pass_eval.shape[0] > 1:
        active_pass_eval = 0.5 * (helical_pass_eval[1:] + helical_pass_eval[:-1])
    else:
        active_pass_eval = jnp.zeros((1,), dtype=helical_pass_eval.dtype)
    activation = jnp.clip(active_pass_eval / max(float(config.max_helical_pass_count), 1e-6), 0.0, 1.0)
    excess = jax.nn.relu(mu_required - friction.mu_max)
    penalty = friction.penalty_weight * jnp.mean((activation * excess) ** 2)
    mu_max_required = jnp.max(mu_required) if mu_required.shape[0] > 0 else jnp.asarray(0.0, dtype=angle_eval.dtype)
    return {
        "s_eval": s_eval,
        "rho_eval": rho_eval,
        "angle_eval": angle_eval,
        "mu_required": mu_required,
        "activation": activation,
        "mu_max_required": mu_max_required,
        "penalty": penalty,
    }


def winding_process_forward(
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
    decoded = decode_winding_process_params(raw, config, geom)
    winding_s_ctrl = decoded["winding_s_ctrl"]
    winding_angle_ctrl = decoded["winding_angle_ctrl"]
    helical_pass_ctrl = decoded["helical_pass_ctrl"]
    hoop_pass_ctrl = decoded["hoop_pass_ctrl"]

    winding_basis = triangular_basis(state["s_coords"], winding_s_ctrl)
    winding_angle = winding_basis @ winding_angle_ctrl
    helical_pass_field = winding_basis @ helical_pass_ctrl
    hoop_pass_field = winding_basis @ hoop_pass_ctrl

    # The helical field cannot geodesically reach all the way into the boss
    # aperture for the allowed angle range. rho_floor guards the Clairaut-based
    # arcsin evaluation in that exclusion zone; the boss rim then falls back to
    # the base laminate because the hoop window is concentrated on the cylinder.
    rho_floor = max(geom.opening_radius + 4.0, 0.18 * geom.mid_radius)
    rho_field = jnp.maximum(state["surface_rho"], rho_floor)
    helical_dirs = (
        jnp.cos(winding_angle)[:, None] * state["meridian_dirs"]
        + jnp.sin(winding_angle)[:, None] * state["hoop_dirs"]
    )
    helical_dirs = helical_dirs - jnp.sum(helical_dirs * state["surface_normals"], axis=-1, keepdims=True) * state["surface_normals"]
    helical_dirs = normalize_jax(helical_dirs)

    helical_factor = helical_deposition_factor(rho_field, winding_angle, config)
    helical_added_thickness = config.tow_thickness * helical_pass_field * helical_factor

    _, cap_len, _ = copv_meridional_metrics(
        geom.mid_radius,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    cylinder_start = jnp.asarray(cap_len, dtype=raw.dtype)
    cylinder_stop = jnp.asarray(cap_len + geom.cylinder_length, dtype=raw.dtype)
    hoop_window = smooth_window(
        state["s_coords"],
        cylinder_start,
        cylinder_stop,
        transition=config.hoop_transition_length,
        beta=beta,
    )
    hoop_added_thickness = config.tow_thickness * hoop_pass_field * hoop_window

    control_phi = jnp.zeros_like(winding_s_ctrl)
    _, _, _, _, rho_ctrl = copv_surface_from_sphi_jax(
        geom.mid_radius,
        winding_s_ctrl,
        control_phi,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    rho_ctrl = jnp.maximum(rho_ctrl, rho_floor)
    helical_factor_ctrl = helical_deposition_factor(rho_ctrl, winding_angle_ctrl, config)
    helical_thickness_ctrl = config.tow_thickness * helical_pass_ctrl * helical_factor_ctrl
    hoop_window_ctrl = smooth_window(
        winding_s_ctrl,
        cylinder_start,
        cylinder_stop,
        transition=config.hoop_transition_length,
        beta=beta,
    )
    hoop_thickness_ctrl = config.tow_thickness * hoop_pass_ctrl * hoop_window_ctrl

    friction_metrics = winding_process_friction_metrics(
        winding_s_ctrl,
        winding_angle_ctrl,
        helical_pass_ctrl,
        geom,
        config,
        friction,
    )

    base = material.base_thickness
    c_base = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    helical_stiffness = rotate_stiffness_field(state["c_mat"], helical_dirs, state["surface_normals"])
    hoop_stiffness = rotate_stiffness_field(state["c_mat"], state["hoop_dirs"], state["surface_normals"])

    winding_thickness = helical_added_thickness + hoop_added_thickness
    total_thickness = jnp.clip(base + winding_thickness, base * material.density_floor)
    c_eff = (
        base * c_base
        + helical_added_thickness[:, None, None, None, None] * helical_stiffness
        + hoop_added_thickness[:, None, None, None, None] * hoop_stiffness
    ) / total_thickness[:, None, None, None, None]

    compliance, displacement = solve_compliance(c_eff, total_thickness)
    density = total_thickness / base
    coverage = winding_thickness / max(float(config.tow_thickness), 1e-6)
    effective_fiber_dirs = normalize_jax(
        helical_added_thickness[:, None] * helical_dirs
        + hoop_added_thickness[:, None] * state["hoop_dirs"]
        + 1e-8 * state["meridian_dirs"]
    )
    winding_angle_field = jnp.arctan2(
        jnp.sum(effective_fiber_dirs * state["hoop_dirs"], axis=-1),
        jnp.sum(effective_fiber_dirs * state["meridian_dirs"], axis=-1),
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

    angle_smoothness = control_smoothness_penalty(winding_angle_ctrl, winding_s_ctrl)
    pass_smoothness = control_smoothness_penalty(helical_pass_ctrl, winding_s_ctrl) + control_smoothness_penalty(
        hoop_pass_ctrl,
        winding_s_ctrl,
    )
    thickness_excess = jax.nn.relu(winding_thickness - config.max_winding_thickness)
    thickness_cap_penalty = config.thickness_cap_penalty_weight * jnp.mean(thickness_excess**2)
    objective = (
        config.mass_weight * (mass_metric / jnp.maximum(baseline_mass, 1e-6))
        + config.angle_smoothness_weight * angle_smoothness
        + config.pass_smoothness_weight * pass_smoothness
        + thickness_cap_penalty
        + failure_metrics["penalty"]
        + friction_metrics["penalty"]
    )

    zero_patch_scalar = jnp.asarray(0.0, dtype=raw.dtype)
    zero_patch_vector = jnp.zeros((0,), dtype=raw.dtype)
    zero_patch_dirs = jnp.zeros((0, 3), dtype=raw.dtype)
    zero_patch_weights = jnp.zeros((0, state["element_count"]), dtype=raw.dtype)
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
        "winding_thickness_ctrl": helical_thickness_ctrl + hoop_thickness_ctrl,
        "winding_angle_field": winding_angle_field,
        "winding_thickness_field": winding_thickness,
        "winding_helical_angle_field": winding_angle,
        "helical_pass_ctrl": helical_pass_ctrl,
        "hoop_pass_ctrl": hoop_pass_ctrl,
        "helical_pass_field": helical_pass_field,
        "hoop_pass_field": hoop_pass_field,
        "helical_thickness_ctrl": helical_thickness_ctrl,
        "hoop_thickness_ctrl": hoop_thickness_ctrl,
        "helical_thickness_field": helical_added_thickness,
        "hoop_thickness_field": hoop_added_thickness,
        "hoop_window_field": hoop_window,
        "friction_s_eval": friction_metrics["s_eval"],
        "friction_angle_eval": friction_metrics["angle_eval"],
        "friction_mu_required": friction_metrics["mu_required"],
        "friction_activation": friction_metrics["activation"],
        "patch_s": zero_patch_vector,
        "patch_phi": zero_patch_vector,
        "patch_alpha": zero_patch_vector,
        "patch_thickness": zero_patch_vector,
        "patch_added_thickness": jnp.zeros((state["element_count"],), dtype=raw.dtype),
        "patch_fiber_dirs": zero_patch_dirs,
        "patch_weights": zero_patch_weights,
        "patch_coverage": jnp.zeros((state["element_count"],), dtype=raw.dtype),
        "coverage_excess": jnp.zeros((state["element_count"],), dtype=raw.dtype),
        "patch_l1": zero_patch_scalar,
        "overlap_penalty": zero_patch_scalar,
        "repulsion_penalty": zero_patch_scalar,
        "friction_penalty": friction_metrics["penalty"],
        "mu_required": friction_metrics["mu_required"],
        "mu_max_required": friction_metrics["mu_max_required"],
        "failure_penalty": failure_metrics["penalty"],
        "failure_penalty_mean": failure_metrics["failure_penalty_mean"],
        "failure_penalty_tail": failure_metrics["failure_penalty_tail"],
        "angle_smoothness_penalty": config.angle_smoothness_weight * angle_smoothness,
        "pass_smoothness_penalty": config.pass_smoothness_weight * pass_smoothness,
        "thickness_cap_penalty": thickness_cap_penalty,
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

    friction_s_eval, friction_rho_eval, friction_angle_eval = hybrid_friction_samples(
        winding_s_ctrl,
        winding_angle_ctrl,
        geom,
        config,
    )
    friction_metrics = friction_penalty(friction_s_eval, friction_rho_eval, friction_angle_eval, friction)

    if config.patch_count > 0:
        centers, e_s, e_phi, _, _ = copv_surface_from_sphi_jax(
            geom.mid_radius,
            patch_s,
            patch_phi,
            geom.cylinder_length,
            geom.opening_radius,
            geom.dome_height_ratio,
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
    c_base = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    winding_stiffness = rotate_stiffness_field(state["c_mat"], winding_dirs, state["surface_normals"])
    total_thickness = jnp.clip(base + winding_thickness + patch_added_thickness, base * material.density_floor)
    c_eff = (
        base * c_base
        + winding_thickness[:, None, None, None, None] * winding_stiffness
        + patch_stiffness
    ) / total_thickness[:, None, None, None, None]

    compliance, displacement = solve_compliance(c_eff, total_thickness)
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
        "friction_s_eval": friction_s_eval,
        "friction_angle_eval": friction_angle_eval,
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
        "failure_penalty_mean": failure_metrics["failure_penalty_mean"],
        "failure_penalty_tail": failure_metrics["failure_penalty_tail"],
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


def initial_winding_process_params(config: HybridConfig) -> jnp.ndarray:
    angle_seed = np.full((config.winding_ctrl_count,), config.winding_seed_angle_deg, dtype=np.float64)
    if config.max_helical_pass_count > 1e-8:
        helical_seed = np.full((config.winding_ctrl_count,), config.helical_seed_pass_count, dtype=np.float64)
        helical_seed = np.clip(helical_seed, 0.0, config.max_helical_pass_count)
        helical_raw = safe_inverse_sigmoid(helical_seed, 0.0, config.max_helical_pass_count)
    else:
        helical_raw = np.full((config.winding_ctrl_count,), -12.0, dtype=np.float64)
    if config.max_hoop_pass_count > 1e-8:
        hoop_seed = np.full((config.winding_ctrl_count,), config.hoop_seed_pass_count, dtype=np.float64)
        hoop_seed = np.clip(hoop_seed, 0.0, config.max_hoop_pass_count)
        hoop_raw = safe_inverse_sigmoid(hoop_seed, 0.0, config.max_hoop_pass_count)
    else:
        hoop_raw = np.full((config.winding_ctrl_count,), -12.0, dtype=np.float64)
    params = [
        safe_inverse_sigmoid(angle_seed, config.min_angle_deg, config.max_angle_deg),
        helical_raw,
        hoop_raw,
    ]
    return jnp.asarray(np.concatenate(params), dtype=current_jax_real_dtype())


def winding_result_metrics(
    result: dict[str, jnp.ndarray],
    config: HybridConfig,
    friction: FrictionConfig,
    tol: float = 1e-6,
) -> dict[str, float | bool]:
    fi_max_with_margin = float(np.asarray(jax.device_get(jnp.max(result["failure_with_margin"]))))
    mu_max_required = float(np.asarray(jax.device_get(result["mu_max_required"])))
    max_winding_thickness = float(np.asarray(jax.device_get(jnp.max(result["winding_thickness_field"]))))
    objective = float(np.asarray(jax.device_get(result["objective"])))
    mass_metric = float(np.asarray(jax.device_get(result["mass_metric"])))
    feasible = (
        fi_max_with_margin <= 1.0 + float(tol)
        and mu_max_required <= float(friction.mu_max) + float(tol)
        and max_winding_thickness <= float(config.max_winding_thickness) + float(tol)
    )
    return {
        "fi_max_with_margin": fi_max_with_margin,
        "mu_max_required": mu_max_required,
        "max_winding_thickness": max_winding_thickness,
        "objective": objective,
        "mass_metric": mass_metric,
        "feasible": feasible,
    }


def add_uniform_winding_pass_bias(
    params: jnp.ndarray,
    config: HybridConfig,
    bias: float,
) -> jnp.ndarray:
    winding_n = int(config.winding_ctrl_count)
    biased = np.asarray(jax.device_get(params), dtype=np.float64).copy()
    biased[winding_n : 3 * winding_n] += float(bias)
    return jnp.asarray(biased, dtype=params.dtype)


def restore_winding_stage_feasibility(
    params: jnp.ndarray,
    state: dict[str, Any],
    material: MaterialConfig,
    config: HybridConfig,
    geom: GeometryConfig,
    solve_compliance,
    beta: float,
    failure_config: FailureConfig,
    friction_config: FrictionConfig,
    initial_bias: float = 0.35,
    max_bias: float = 10.0,
    refine_steps: int = 8,
) -> dict[str, Any]:
    base_result = winding_process_forward(
        params,
        state,
        material,
        config,
        geom,
        solve_compliance,
        beta=beta,
        failure_config=failure_config,
        friction_config=friction_config,
    )
    base_metrics = winding_result_metrics(base_result, config, friction_config)
    if bool(base_metrics["feasible"]):
        return {
            "params": params,
            "result": base_result,
            "metrics": base_metrics,
            "restored": False,
            "applied_bias": 0.0,
        }

    lower_bias = 0.0
    upper_bias: float | None = None
    upper_params: jnp.ndarray | None = None
    upper_result: dict[str, jnp.ndarray] | None = None
    upper_metrics: dict[str, float | bool] | None = None
    bias = max(float(initial_bias), 1e-3)

    while bias <= float(max_bias) + 1e-9:
        trial_params = add_uniform_winding_pass_bias(params, config, bias)
        trial_result = winding_process_forward(
            trial_params,
            state,
            material,
            config,
            geom,
            solve_compliance,
            beta=beta,
            failure_config=failure_config,
            friction_config=friction_config,
        )
        trial_metrics = winding_result_metrics(trial_result, config, friction_config)
        if bool(trial_metrics["feasible"]):
            upper_bias = bias
            upper_params = trial_params
            upper_result = trial_result
            upper_metrics = trial_metrics
            break
        lower_bias = bias
        bias *= 2.0

    if upper_bias is None or upper_params is None or upper_result is None or upper_metrics is None:
        return {
            "params": params,
            "result": base_result,
            "metrics": base_metrics,
            "restored": False,
            "applied_bias": 0.0,
        }

    for _ in range(max(int(refine_steps), 0)):
        mid_bias = 0.5 * (lower_bias + upper_bias)
        trial_params = add_uniform_winding_pass_bias(params, config, mid_bias)
        trial_result = winding_process_forward(
            trial_params,
            state,
            material,
            config,
            geom,
            solve_compliance,
            beta=beta,
            failure_config=failure_config,
            friction_config=friction_config,
        )
        trial_metrics = winding_result_metrics(trial_result, config, friction_config)
        if bool(trial_metrics["feasible"]):
            upper_bias = mid_bias
            upper_params = trial_params
            upper_result = trial_result
            upper_metrics = trial_metrics
        else:
            lower_bias = mid_bias

    return {
        "params": upper_params,
        "result": upper_result,
        "metrics": upper_metrics,
        "restored": True,
        "applied_bias": float(upper_bias),
    }


def run_lbfgs(
    loss_fn,
    params0: jnp.ndarray,
    maxiter: int,
    tol: float,
    history_size: int,
    callback: Callable[[jnp.ndarray, dict[str, Any]], None] | None = None,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    def info_from_jaxopt_state(state) -> dict[str, Any]:
        return {
            "iterations": int(np.asarray(jax.device_get(state.iter_num))),
            "loss": float(np.asarray(jax.device_get(state.value))),
            "error": float(np.asarray(jax.device_get(state.error))),
            "failed_linesearch": bool(np.asarray(jax.device_get(state.failed_linesearch))),
            "num_fun_eval": int(np.asarray(jax.device_get(state.num_fun_eval))),
            "num_grad_eval": int(np.asarray(jax.device_get(state.num_grad_eval))),
        }

    if jaxopt is not None:
        solver = jaxopt.LBFGS(
            fun=loss_fn,
            maxiter=maxiter,
            tol=tol,
            history_size=history_size,
            implicit_diff=False,
            jit=True,
            verbose=False,
        )
        if callback is None:
            out = solver.run(params0)
            return out.params, info_from_jaxopt_state(out.state)

        params = params0
        state = solver.init_state(params0)
        while True:
            info = info_from_jaxopt_state(state)
            if info["iterations"] >= int(maxiter) or info["error"] <= float(tol) or info["failed_linesearch"]:
                break
            opt_step = solver.update(params, state)
            params, state = opt_step.params, opt_step.state
            callback(params, info_from_jaxopt_state(state))
        return params, info_from_jaxopt_state(state)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    fun_eval_count = 0
    grad_eval_count = 0

    def scipy_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal fun_eval_count, grad_eval_count
        params = jnp.asarray(x, dtype=params0.dtype)
        value, grad = loss_and_grad(params)
        value_np = float(np.asarray(jax.device_get(value)))
        grad_np = np.asarray(jax.device_get(grad), dtype=np.float64)
        fun_eval_count += 1
        grad_eval_count += 1
        return value_np, grad_np

    callback_iter = 0

    def scipy_callback(xk: np.ndarray) -> None:
        nonlocal callback_iter
        if callback is None:
            return
        callback_iter += 1
        params = jnp.asarray(xk, dtype=params0.dtype)
        value, grad = loss_and_grad(params)
        grad_norm = float(np.linalg.norm(np.asarray(jax.device_get(grad), dtype=np.float64), ord=np.inf))
        callback(
            params,
            {
                "iterations": callback_iter,
                "loss": float(np.asarray(jax.device_get(value))),
                "error": grad_norm,
                "failed_linesearch": False,
                "num_fun_eval": fun_eval_count,
                "num_grad_eval": grad_eval_count,
            },
        )

    result = scipy_optimize.minimize(
        scipy_objective,
        np.asarray(params0, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        callback=scipy_callback,
        options={
            "maxiter": int(maxiter),
            "ftol": float(tol),
            "gtol": float(tol),
            "maxcor": int(history_size),
        },
    )
    grad_norm = float(np.linalg.norm(np.asarray(result.jac, dtype=np.float64), ord=np.inf)) if result.jac is not None else 0.0
    info = {
        "iterations": int(result.nit),
        "loss": float(result.fun),
        "error": grad_norm,
        "failed_linesearch": (not bool(result.success)) and ("line search" in str(result.message).lower()),
        "num_fun_eval": int(result.nfev),
        "num_grad_eval": int(result.njev),
    }
    return jnp.asarray(result.x, dtype=params0.dtype), info


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
    c_base = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    compliance_scale, _ = solve_compliance(
        c_base,
        jnp.full((state["element_count"],), material.base_thickness, dtype=c_base.dtype),
    )
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
    snapshot_every: int | None = None,
) -> dict[str, Any]:
    failure = FailureConfig() if failure_config is None else failure_config
    friction = FrictionConfig() if friction_config is None else friction_config
    params = initial_hybrid_params(config) if params0 is None else params0
    history: list[dict[str, Any]] = []
    optimization_snapshots: list[dict[str, Any]] = []
    result = None

    def build_snapshot(
        current_result: dict[str, jnp.ndarray],
        beta: float,
        beta_index: int,
        stage_iteration: int,
        global_iteration: int,
        snapshot_kind: str,
        solver_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        patch_thickness = current_result["patch_thickness"]
        failure_with_margin = current_result["failure_with_margin"]
        snapshot = {
            "beta": float(beta),
            "beta_index": int(beta_index),
            "stage_iteration": int(stage_iteration),
            "global_iteration": int(global_iteration),
            "snapshot_kind": snapshot_kind,
            "objective": current_result["objective"],
            "strain_energy": current_result["compliance"],
            "mass_metric": current_result["mass_metric"],
            "fi_max": current_result["fi_max"],
            "fi_max_with_margin": jnp.max(failure_with_margin),
            "fi_mean": current_result["fi_mean"],
            "mu_max_required": current_result["mu_max_required"],
            "active_patch_count": count_active_patches_np(patch_thickness, config),
            "max_patch_thickness": jnp.max(patch_thickness) if config.patch_count > 0 else jnp.asarray(0.0, dtype=params.dtype),
            "max_winding_thickness": jnp.max(current_result["winding_thickness_field"]),
            "winding_s_ctrl": current_result["winding_s_ctrl"],
            "winding_angle_ctrl": current_result["winding_angle_ctrl"],
            "winding_thickness_ctrl": current_result["winding_thickness_ctrl"],
            "patch_s": current_result["patch_s"],
            "patch_phi": current_result["patch_phi"],
            "patch_alpha": current_result["patch_alpha"],
            "patch_thickness": patch_thickness,
            "failure_index": current_result["failure_index"],
            "failure_with_margin": failure_with_margin,
        }
        if solver_info is not None:
            snapshot["solver"] = {
                "iterations": int(solver_info["iterations"]),
                "loss": float(solver_info["loss"]),
                "error": float(solver_info["error"]),
                "failed_linesearch": bool(solver_info["failed_linesearch"]),
                "num_fun_eval": int(solver_info["num_fun_eval"]),
                "num_grad_eval": int(solver_info["num_grad_eval"]),
            }
        return snapshot

    total_iterations = 0
    if snapshot_every is not None and snapshot_every > 0 and len(config.beta_schedule) > 0:
        initial_beta = float(config.beta_schedule[0])
        initial_result = hybrid_forward(
            params,
            state,
            material,
            config,
            geom,
            solve_compliance,
            beta=initial_beta,
            failure_config=failure,
            friction_config=friction,
        )
        optimization_snapshots.append(
            build_snapshot(
                initial_result,
                beta=initial_beta,
                beta_index=0,
                stage_iteration=0,
                global_iteration=0,
                snapshot_kind="initial",
            )
        )

    for beta_index, beta in enumerate(config.beta_schedule):
        stage_last_iteration = 0

        def record_snapshot(current_params: jnp.ndarray, info: dict[str, Any]) -> None:
            nonlocal stage_last_iteration
            stage_last_iteration = int(info["iterations"])
            if snapshot_every is None or snapshot_every <= 0:
                return
            if stage_last_iteration <= 0 or stage_last_iteration % snapshot_every != 0:
                return
            current_result = hybrid_forward(
                current_params,
                state,
                material,
                config,
                geom,
                solve_compliance,
                beta=beta,
                failure_config=failure,
                friction_config=friction,
            )
            optimization_snapshots.append(
                build_snapshot(
                    current_result,
                    beta=float(beta),
                    beta_index=beta_index,
                    stage_iteration=stage_last_iteration,
                    global_iteration=total_iterations + stage_last_iteration,
                    snapshot_kind="iteration",
                    solver_info=info,
                )
            )

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
        params, info = run_lbfgs(
            loss_fn,
            params,
            config.lbfgs_maxiter,
            config.lbfgs_tol,
            config.history_size,
            callback=record_snapshot if snapshot_every is not None and snapshot_every > 0 else None,
        )
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
        stage_last_iteration = int(info["iterations"])
        if snapshot_every is not None and snapshot_every > 0:
            should_record_final = stage_last_iteration == 0 or stage_last_iteration % snapshot_every != 0
            if should_record_final:
                optimization_snapshots.append(
                    build_snapshot(
                        result,
                        beta=float(beta),
                        beta_index=beta_index,
                        stage_iteration=stage_last_iteration,
                        global_iteration=total_iterations + stage_last_iteration,
                        snapshot_kind="beta_final",
                        solver_info=info,
                    )
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
                    np.asarray(jax.device_get(jnp.sum(result["patch_thickness"] > active_patch_threshold(config))))
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
        total_iterations += stage_last_iteration
    if result is None:
        raise RuntimeError("Hybrid optimisation did not run")
    return {
        "params": params,
        "history": history,
        "result": result,
        "optimization_snapshots": optimization_snapshots,
    }


def run_winding_optimization(
    state: dict[str, Any],
    material: MaterialConfig,
    config: HybridConfig,
    geom: GeometryConfig,
    solve_compliance,
    failure_config: FailureConfig | None = None,
    friction_config: FrictionConfig | None = None,
    params0: jnp.ndarray | None = None,
    snapshot_every: int | None = None,
) -> dict[str, Any]:
    failure = FailureConfig() if failure_config is None else failure_config
    friction = FrictionConfig() if friction_config is None else friction_config
    params = initial_winding_process_params(config) if params0 is None else params0
    history: list[dict[str, Any]] = []
    optimization_snapshots: list[dict[str, Any]] = []
    result = None

    def build_snapshot(
        current_result: dict[str, jnp.ndarray],
        beta: float,
        beta_index: int,
        stage_iteration: int,
        global_iteration: int,
        snapshot_kind: str,
        solver_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        failure_with_margin = current_result["failure_with_margin"]
        snapshot = {
            "beta": float(beta),
            "beta_index": int(beta_index),
            "stage_iteration": int(stage_iteration),
            "global_iteration": int(global_iteration),
            "snapshot_kind": snapshot_kind,
            "objective": current_result["objective"],
            "strain_energy": current_result["compliance"],
            "mass_metric": current_result["mass_metric"],
            "fi_max": current_result["fi_max"],
            "fi_max_with_margin": jnp.max(failure_with_margin),
            "fi_mean": current_result["fi_mean"],
            "mu_max_required": current_result["mu_max_required"],
            "active_patch_count": 0,
            "max_patch_thickness": jnp.asarray(0.0, dtype=params.dtype),
            "max_winding_thickness": jnp.max(current_result["winding_thickness_field"]),
            "max_helical_thickness": jnp.max(current_result["helical_thickness_field"]),
            "max_hoop_thickness": jnp.max(current_result["hoop_thickness_field"]),
            "winding_s_ctrl": current_result["winding_s_ctrl"],
            "winding_angle_ctrl": current_result["winding_angle_ctrl"],
            "winding_thickness_ctrl": current_result["winding_thickness_ctrl"],
            "helical_pass_ctrl": current_result["helical_pass_ctrl"],
            "hoop_pass_ctrl": current_result["hoop_pass_ctrl"],
            "helical_thickness_ctrl": current_result["helical_thickness_ctrl"],
            "hoop_thickness_ctrl": current_result["hoop_thickness_ctrl"],
            "patch_s": current_result["patch_s"],
            "patch_phi": current_result["patch_phi"],
            "patch_alpha": current_result["patch_alpha"],
            "patch_thickness": current_result["patch_thickness"],
            "failure_index": current_result["failure_index"],
            "failure_with_margin": failure_with_margin,
        }
        if solver_info is not None:
            snapshot["solver"] = {
                "iterations": int(solver_info["iterations"]),
                "loss": float(solver_info["loss"]),
                "error": float(solver_info["error"]),
                "failed_linesearch": bool(solver_info["failed_linesearch"]),
                "num_fun_eval": int(solver_info["num_fun_eval"]),
                "num_grad_eval": int(solver_info["num_grad_eval"]),
            }
        return snapshot

    total_iterations = 0
    if snapshot_every is not None and snapshot_every > 0 and len(config.beta_schedule) > 0:
        initial_beta = float(config.beta_schedule[0])
        initial_result = winding_process_forward(
            params,
            state,
            material,
            config,
            geom,
            solve_compliance,
            beta=initial_beta,
            failure_config=failure,
            friction_config=friction,
        )
        optimization_snapshots.append(
            build_snapshot(
                initial_result,
                beta=initial_beta,
                beta_index=0,
                stage_iteration=0,
                global_iteration=0,
                snapshot_kind="initial",
            )
        )

    for beta_index, beta in enumerate(config.beta_schedule):
        stage_last_iteration = 0
        stage_restore_bias = 0.0
        stage_best_feasible_params: jnp.ndarray | None = None
        stage_best_feasible_result: dict[str, jnp.ndarray] | None = None
        stage_best_feasible_metrics: dict[str, float | bool] | None = None
        stage_best_feasible_iteration: int | None = None

        def consider_stage_feasible_candidate(
            candidate_params: jnp.ndarray,
            candidate_result: dict[str, jnp.ndarray],
            global_iteration: int,
        ) -> tuple[dict[str, float | bool], bool]:
            nonlocal stage_best_feasible_params, stage_best_feasible_result, stage_best_feasible_metrics, stage_best_feasible_iteration
            candidate_metrics = winding_result_metrics(candidate_result, config, friction)
            if not bool(candidate_metrics["feasible"]):
                return candidate_metrics, False
            replace = stage_best_feasible_metrics is None
            if not replace:
                best_objective = float(stage_best_feasible_metrics["objective"])
                best_mass = float(stage_best_feasible_metrics["mass_metric"])
                candidate_objective = float(candidate_metrics["objective"])
                candidate_mass = float(candidate_metrics["mass_metric"])
                replace = (
                    candidate_objective < best_objective - 1e-9
                    or (
                        abs(candidate_objective - best_objective) <= 1e-9
                        and candidate_mass < best_mass - 1e-9
                    )
                )
            if replace:
                stage_best_feasible_params = candidate_params
                stage_best_feasible_result = candidate_result
                stage_best_feasible_metrics = candidate_metrics
                stage_best_feasible_iteration = int(global_iteration)
            return candidate_metrics, True

        if beta_index > 0:
            restored_start = restore_winding_stage_feasibility(
                params,
                state,
                material,
                config,
                geom,
                solve_compliance,
                beta=float(beta),
                failure_config=failure,
                friction_config=friction,
            )
            if bool(restored_start["metrics"]["feasible"]):
                params = restored_start["params"]
                stage_restore_bias = float(restored_start["applied_bias"])
                consider_stage_feasible_candidate(params, restored_start["result"], total_iterations)

        def record_snapshot(current_params: jnp.ndarray, info: dict[str, Any]) -> None:
            nonlocal stage_last_iteration
            stage_last_iteration = int(info["iterations"])
            current_result = winding_process_forward(
                current_params,
                state,
                material,
                config,
                geom,
                solve_compliance,
                beta=beta,
                failure_config=failure,
                friction_config=friction,
            )
            current_iteration = total_iterations + stage_last_iteration
            current_metrics, current_feasible = consider_stage_feasible_candidate(
                current_params,
                current_result,
                current_iteration,
            )
            if snapshot_every is None or snapshot_every <= 0:
                return
            if stage_last_iteration <= 0 or stage_last_iteration % snapshot_every != 0:
                return
            snapshot_result = current_result
            snapshot_kind = "iteration"
            if stage_best_feasible_result is not None and not current_feasible:
                snapshot_result = stage_best_feasible_result
                snapshot_kind = "accepted_feasible"
            snapshot = build_snapshot(
                snapshot_result,
                beta=float(beta),
                beta_index=beta_index,
                stage_iteration=stage_last_iteration,
                global_iteration=current_iteration,
                snapshot_kind=snapshot_kind,
                solver_info=info,
            )
            if snapshot_result is not current_result:
                snapshot["reported_from_feasible_incumbent"] = True
                snapshot["accepted_source_global_iteration"] = int(stage_best_feasible_iteration)
                snapshot["raw_fi_max_with_margin"] = float(current_metrics["fi_max_with_margin"])
                snapshot["raw_max_winding_thickness"] = float(current_metrics["max_winding_thickness"])
            if stage_restore_bias > 0.0:
                snapshot["stage_restore_pass_bias"] = float(stage_restore_bias)
            optimization_snapshots.append(
                snapshot
            )

        loss_fn = lambda raw, beta=beta: winding_process_forward(
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
        params, info = run_lbfgs(
            loss_fn,
            params,
            config.lbfgs_maxiter,
            config.lbfgs_tol,
            config.history_size,
            callback=record_snapshot if snapshot_every is not None and snapshot_every > 0 else None,
        )
        result = winding_process_forward(
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
        stage_last_iteration = int(info["iterations"])
        raw_global_iteration = total_iterations + stage_last_iteration
        raw_metrics, raw_feasible = consider_stage_feasible_candidate(params, result, raw_global_iteration)
        if not raw_feasible and stage_best_feasible_result is None:
            restored_final = restore_winding_stage_feasibility(
                params,
                state,
                material,
                config,
                geom,
                solve_compliance,
                beta=float(beta),
                failure_config=failure,
                friction_config=friction,
            )
            if bool(restored_final["metrics"]["feasible"]):
                stage_restore_bias = max(stage_restore_bias, float(restored_final["applied_bias"]))
                consider_stage_feasible_candidate(restored_final["params"], restored_final["result"], raw_global_iteration)

        accepted_result = result
        accepted_params = params
        accepted_metrics = raw_metrics
        reported_from_incumbent = False
        if stage_best_feasible_result is not None and stage_best_feasible_params is not None and stage_best_feasible_metrics is not None:
            accepted_result = stage_best_feasible_result
            accepted_params = stage_best_feasible_params
            accepted_metrics = stage_best_feasible_metrics
            reported_from_incumbent = (
                stage_best_feasible_iteration is not None
                and (
                    int(stage_best_feasible_iteration) < int(raw_global_iteration)
                    or abs(float(accepted_metrics["objective"]) - float(raw_metrics["objective"])) > 1e-9
                    or abs(float(accepted_metrics["fi_max_with_margin"]) - float(raw_metrics["fi_max_with_margin"])) > 1e-9
                )
            )
        params = accepted_params
        result = accepted_result
        if snapshot_every is not None and snapshot_every > 0:
            should_record_final = (
                stage_last_iteration == 0
                or stage_last_iteration % snapshot_every != 0
                or reported_from_incumbent
            )
            if should_record_final:
                snapshot_kind = "beta_accepted" if reported_from_incumbent else "beta_final"
                snapshot = build_snapshot(
                    result,
                    beta=float(beta),
                    beta_index=beta_index,
                    stage_iteration=stage_last_iteration,
                    global_iteration=raw_global_iteration,
                    snapshot_kind=snapshot_kind,
                    solver_info=info,
                )
                if reported_from_incumbent:
                    snapshot["reported_from_feasible_incumbent"] = True
                    snapshot["accepted_source_global_iteration"] = int(stage_best_feasible_iteration)
                    snapshot["raw_fi_max_with_margin"] = float(raw_metrics["fi_max_with_margin"])
                    snapshot["raw_max_winding_thickness"] = float(raw_metrics["max_winding_thickness"])
                if stage_restore_bias > 0.0:
                    snapshot["stage_restore_pass_bias"] = float(stage_restore_bias)
                optimization_snapshots.append(snapshot)
        history.append(
            {
                "beta": float(beta),
                "objective": float(np.asarray(jax.device_get(result["objective"]))),
                "strain_energy": float(np.asarray(jax.device_get(result["compliance"]))),
                "mass_metric": float(np.asarray(jax.device_get(result["mass_metric"]))),
                "fi_max": float(np.asarray(jax.device_get(result["fi_max"]))),
                "fi_mean": float(np.asarray(jax.device_get(result["fi_mean"]))),
                "mu_max_required": float(np.asarray(jax.device_get(result["mu_max_required"]))),
                "max_winding_thickness": float(np.asarray(jax.device_get(jnp.max(result["winding_thickness_field"])))),
                "max_helical_thickness": float(np.asarray(jax.device_get(jnp.max(result["helical_thickness_field"])))),
                "max_hoop_thickness": float(np.asarray(jax.device_get(jnp.max(result["hoop_thickness_field"])))),
                "mean_helical_pass_count": float(np.asarray(jax.device_get(jnp.mean(result["helical_pass_field"])))),
                "mean_hoop_pass_count": float(np.asarray(jax.device_get(jnp.mean(result["hoop_pass_field"])))),
                "patch_l1": 0.0,
                "overlap_penalty": 0.0,
                "repulsion_penalty": 0.0,
                "failure_penalty": float(np.asarray(jax.device_get(result["failure_penalty"]))),
                "failure_penalty_mean": float(np.asarray(jax.device_get(result["failure_penalty_mean"]))),
                "failure_penalty_tail": float(np.asarray(jax.device_get(result["failure_penalty_tail"]))),
                "friction_penalty": float(np.asarray(jax.device_get(result["friction_penalty"]))),
                "angle_smoothness_penalty": float(np.asarray(jax.device_get(result["angle_smoothness_penalty"]))),
                "pass_smoothness_penalty": float(np.asarray(jax.device_get(result["pass_smoothness_penalty"]))),
                "thickness_cap_penalty": float(np.asarray(jax.device_get(result["thickness_cap_penalty"]))),
                "accepted_feasible_incumbent": bool(reported_from_incumbent),
                "accepted_source_global_iteration": int(stage_best_feasible_iteration) if stage_best_feasible_iteration is not None else int(raw_global_iteration),
                "stage_restore_pass_bias": float(stage_restore_bias),
                "raw_fi_max_with_margin": float(raw_metrics["fi_max_with_margin"]),
                "accepted_fi_max_with_margin": float(accepted_metrics["fi_max_with_margin"]),
                **info,
            }
        )
        total_iterations += stage_last_iteration

    if result is None:
        raise RuntimeError("Winding optimisation did not run")
    return {
        "params": params,
        "history": history,
        "result": result,
        "optimization_snapshots": optimization_snapshots,
    }
