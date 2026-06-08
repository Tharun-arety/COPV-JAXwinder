from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MaterialConfig:
    e_xx: float = 136101.0
    e_yy: float = 1442.0
    e_zz: float = 1442.0
    nu_xy: float = 0.954
    nu_xz: float = 0.01
    nu_yz: float = 0.01
    g_yz: float = 8685.0
    g_xz: float = 8685.0
    g_xy: float = 8685.0
    base_plies: int = 4
    ply_thickness: float = 0.3
    density_floor: float = 1e-3

    @property
    def base_thickness(self) -> float:
        return self.base_plies * self.ply_thickness


@dataclass
class MaterialAllowables:
    xt: float = 2200.0
    xc: float = 1400.0
    yt: float = 70.0
    yc: float = 220.0
    s: float = 120.0


@dataclass
class FailureConfig:
    allowables: MaterialAllowables = field(default_factory=MaterialAllowables)
    margin_of_safety: float = 1.5
    penalty_weight: float = 40.0
    softplus_scale: float = 12.0


@dataclass
class FrictionConfig:
    mu_max: float = 0.15
    penalty_weight: float = 10.0
    mu_regularization: float = 1e-6


@dataclass
class PatchConfig:
    count: int = 12
    length: float = 34.0
    width: float = 12.0
    beta_schedule: tuple[float, ...] = (2.0, 8.0, 16.0)
    lbfgs_maxiter: int = 12
    lbfgs_tol: float = 1e-3
    history_size: int = 8
    seed: int = 42
    stack_target: float = 1.0
    overlap_penalty_weight: float = 0.45
    repulsion_penalty_weight: float = 0.12
    min_center_spacing_factor: float = 0.85
    init_jitter: float = 0.025


@dataclass
class IFPConfig:
    ctrl_count: int = 4
    sample_count: int = 24
    family_count: int = 8
    tow_width: float = 12.0
    tow_thickness: float = 0.3
    beta_schedule: tuple[float, ...] = (2.0, 8.0, 16.0)
    smooth_tau: float = 6.0
    lbfgs_maxiter: int = 12
    lbfgs_tol: float = 1e-3
    history_size: int = 8
    seed_points: tuple[tuple[float, float], ...] = (
        (0.05, 0.05),
        (0.35, 0.18),
        (0.65, 0.42),
        (0.95, 0.68),
    )


@dataclass
class WindingConfig:
    family_count: int = 8
    sample_count: int = 260
    band_thickness: float = 0.3
    angle_count: int = 16
    min_angle_deg: float = 12.0
    max_angle_deg: float = 55.0


@dataclass
class HybridConfig:
    winding_ctrl_count: int = 6
    patch_count: int = 10
    patch_length: float = 34.0
    patch_width: float = 12.0
    beta_schedule: tuple[float, ...] = (2.0, 6.0, 12.0)
    lbfgs_maxiter: int = 14
    lbfgs_tol: float = 1e-3
    history_size: int = 10
    seed: int = 7
    init_jitter: float = 0.02
    stack_target: float = 1.0
    min_center_spacing_factor: float = 0.85
    overlap_penalty_weight: float = 0.18
    repulsion_penalty_weight: float = 0.06
    patch_l1_weight: float = 0.14
    mass_weight: float = 1.0
    min_angle_deg: float = 12.0
    max_angle_deg: float = 58.0
    max_winding_thickness: float = 1.2
    max_patch_thickness: float = 1.2
    winding_seed_angle_deg: float = 42.0
    winding_seed_thickness: float = 0.35
    patch_seed_thickness: float = 0.08


@dataclass
class GeometryConfig:
    outer_radius: float = 100.0
    cylinder_length: float = 220.0
    thickness: float = 8.0
    opening_radius: float = 10.0
    pressure: float = 1.0
    support_tol: float = 1.5
    mesh_hmin: float = 16.0
    mesh_hmax: float = 36.0

    @property
    def inner_radius(self) -> float:
        return self.outer_radius - self.thickness

    @property
    def mid_radius(self) -> float:
        return self.outer_radius - 0.5 * self.thickness

    @property
    def half_cyl(self) -> float:
        return 0.5 * self.cylinder_length
