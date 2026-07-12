"""Element-level Classical Laminate Theory on the solved shell.

Turns the smeared FEM solution into per-ply results on EVERY element: for each
element take the membrane strain from the FEM, express it in the local
meridian/hoop laminate frame, then evaluate each distinct ply orientation the
winding lays down there (base 0 deg, helical +/-alpha, hoop 90 deg) against that
strain. The worst ply gives the element's critical-ply failure index, reserve
factor, and which orientation fails.

Under the shell iso-strain (membrane) assumption every ply shares the element
strain, so a ply's stress is simply Q(theta):epsilon — no per-element ABD solve is
needed, which makes this a fast vectorised post-process over all elements.
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from copv_opt.clt import PlyMaterial, hashin_modes, reduced_stiffness
from copv_opt.config import MaterialAllowables, MaterialConfig
from copv_opt.physics import element_strain_stress, engineering_strain_to_tensor_field, rotate_stiffness_field


def _hashin(s11, s22, t12, a: MaterialAllowables):
    # single-source criterion: delegate to clt.hashin_modes so formulas cannot drift
    return hashin_modes(s11, s22, t12, a)["failure_index"]


def _ply_stress(eps0: np.ndarray, theta_deg: np.ndarray, Q: np.ndarray):
    """Stress in a ply at angle theta given laminate strain eps0=[em,eh,gamma]."""
    t = np.radians(theta_deg)
    m, n = np.cos(t), np.sin(t)
    ex, ey, g = eps0[:, 0], eps0[:, 1], eps0[:, 2]
    e1 = m * m * ex + n * n * ey + m * n * g               # engineering-strain transform
    e2 = n * n * ex + m * m * ey - m * n * g
    g12 = -2 * m * n * ex + 2 * m * n * ey + (m * m - n * n) * g
    s11 = Q[0, 0] * e1 + Q[0, 1] * e2
    s22 = Q[0, 1] * e1 + Q[1, 1] * e2
    t12 = Q[2, 2] * g12
    return s11, s22, t12


def element_clt_fields(
    state: dict,
    displacement,
    winding_angle_deg: np.ndarray,
    has_helical: np.ndarray,
    has_hoop: np.ndarray,
    material: MaterialConfig | None = None,
    allowables: MaterialAllowables | None = None,
) -> dict[str, np.ndarray]:
    """Per-element per-ply CLT fields on the solved vessel.

    Returns critical-ply failure index, per-ply reserve factor, and the critical
    ply orientation, one value per element."""
    material = MaterialConfig() if material is None else material
    allow = MaterialAllowables() if allowables is None else allowables
    Q = reduced_stiffness(PlyMaterial(e1=material.e_xx, e2=material.e_yy, g12=material.g_xy, nu12=material.nu_xy))

    # membrane strain (global) -> tensor -> meridian/hoop laminate frame. Strain is
    # independent of the stiffness passed here, so a nominal c_eff is fine.
    c_dummy = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    strain_voigt, _ = element_strain_stress(state, jnp.asarray(displacement), c_dummy)
    eps_t = np.asarray(engineering_strain_to_tensor_field(strain_voigt), dtype=np.float64)
    md = np.asarray(state["meridian_dirs"], dtype=np.float64)
    hd = np.asarray(state["hoop_dirs"], dtype=np.float64)
    e_m = np.einsum("ni,nij,nj->n", md, eps_t, md)
    e_h = np.einsum("ni,nij,nj->n", hd, eps_t, hd)
    g_mh = 2.0 * np.einsum("ni,nij,nj->n", md, eps_t, hd)   # engineering shear
    eps0 = np.stack([e_m, e_h, g_mh], axis=-1)

    n = len(e_m)
    wa = np.asarray(winding_angle_deg, dtype=np.float64)
    has_h = np.asarray(has_helical, dtype=bool)
    has_p = np.asarray(has_hoop, dtype=bool)
    fi = np.zeros(n)
    crit_angle = np.zeros(n)

    plies = [
        (np.ones(n, bool), np.zeros(n)),     # base axial ply (0 deg)
        (has_h, wa),                          # helical +alpha
        (has_h, -wa),                         # helical -alpha
        (has_p, np.full(n, 90.0)),            # hoop
    ]
    for present, theta in plies:
        s11, s22, t12 = _ply_stress(eps0, theta, Q)
        f = np.where(present, _hashin(s11, s22, t12, allow), 0.0)
        newmax = f > fi
        crit_angle = np.where(newmax, theta, crit_angle)
        fi = np.maximum(fi, f)

    rf = 1.0 / np.sqrt(np.maximum(fi, 1e-12))
    return {
        "CLT critical-ply FI": fi,
        "CLT per-ply reserve factor": np.minimum(rf, 99.0),
        "CLT critical ply angle [deg]": np.abs(crit_angle),
    }
