"""Classical Laminate Theory (CLT) — ply-by-ply laminate mechanics.

This is the fidelity step from "Hashin on a smeared laminate stress" to real
ply-by-ply analysis: build the laminate from its actual plies, recover the stress in
EACH ply in its own material coordinates, and evaluate failure per ply. It is what a
composites stress engineer (and Ansys ACP) actually does.

Pure NumPy, standard plane-stress CLT (Jones / Barbero convention):

* per-ply reduced stiffness Q (material coords), transformed to laminate coords Qbar
* laminate stiffness A (extensional), B (coupling), D (bending):
      A = Σ Qbar_k (z_k - z_{k-1})
      B = ½ Σ Qbar_k (z_k² - z_{k-1}²)
      D = ⅓ Σ Qbar_k (z_k³ - z_{k-1}³)
* constitutive [N; M] = [[A, B], [B, D]] [ε⁰; κ]
* per-ply strain ε(z) = ε⁰ + z κ, rotated to material coords, σ = Q ε
* per-ply Hashin (reuses the engine's failure convention)

All matrix algebra — a differentiable JAX port is a mechanical `np`→`jnp` swap.
Verified in app/verify_clt.py against analytic and invariant checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from copv_opt.config import MaterialAllowables


@dataclass
class PlyMaterial:
    """Orthotropic UD ply elastic constants (plane stress)."""
    e1: float = 139067.0
    e2: float = 7908.0
    g12: float = 3206.0
    nu12: float = 0.257


@dataclass
class Ply:
    angle_deg: float          # fibre orientation from the laminate x-axis
    thickness: float          # mm
    material: PlyMaterial = None

    def __post_init__(self):
        if self.material is None:
            self.material = PlyMaterial()


def reduced_stiffness(m: PlyMaterial) -> np.ndarray:
    """Plane-stress reduced stiffness Q (material coords), 3x3 [11,22,12/66]."""
    nu21 = m.nu12 * m.e2 / m.e1
    denom = 1.0 - m.nu12 * nu21
    q11 = m.e1 / denom
    q22 = m.e2 / denom
    q12 = m.nu12 * m.e2 / denom
    q66 = m.g12
    return np.array([[q11, q12, 0.0], [q12, q22, 0.0], [0.0, 0.0, q66]], dtype=np.float64)


def qbar(q: np.ndarray, angle_deg: float) -> np.ndarray:
    """Transform ply stiffness Q to laminate coords by fibre angle (closed form)."""
    t = np.radians(angle_deg)
    m, n = np.cos(t), np.sin(t)
    q11, q12, q22, q66 = q[0, 0], q[0, 1], q[1, 1], q[2, 2]
    m2, n2 = m * m, n * n
    m4, n4 = m2 * m2, n2 * n2
    qb11 = q11 * m4 + 2 * (q12 + 2 * q66) * m2 * n2 + q22 * n4
    qb22 = q11 * n4 + 2 * (q12 + 2 * q66) * m2 * n2 + q22 * m4
    qb12 = (q11 + q22 - 4 * q66) * m2 * n2 + q12 * (m4 + n4)
    qb66 = (q11 + q22 - 2 * q12 - 2 * q66) * m2 * n2 + q66 * (m4 + n4)
    qb16 = (q11 - q12 - 2 * q66) * m * m2 * n + (q12 - q22 + 2 * q66) * m * n * n2
    qb26 = (q11 - q12 - 2 * q66) * m * n * n2 + (q12 - q22 + 2 * q66) * m * m2 * n
    return np.array([[qb11, qb12, qb16], [qb12, qb22, qb26], [qb16, qb26, qb66]], dtype=np.float64)


def _strain_transform(angle_deg: float) -> np.ndarray:
    """Engineering-strain transform T_eps: [ε1,ε2,γ12] = T_eps [εx,εy,γxy]."""
    t = np.radians(angle_deg)
    m, n = np.cos(t), np.sin(t)
    return np.array([[m * m, n * n, m * n],
                     [n * n, m * m, -m * n],
                     [-2 * m * n, 2 * m * n, m * m - n * n]], dtype=np.float64)


def z_boundaries(plies: list[Ply]) -> np.ndarray:
    """Ply interface z-coordinates about the mid-plane, length N+1."""
    total = sum(p.thickness for p in plies)
    z = -0.5 * total
    out = [z]
    for p in plies:
        z += p.thickness
        out.append(z)
    return np.asarray(out, dtype=np.float64)


def abd_matrices(plies: list[Ply]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Laminate extensional (A), coupling (B), and bending (D) stiffness matrices."""
    z = z_boundaries(plies)
    A = np.zeros((3, 3)); B = np.zeros((3, 3)); D = np.zeros((3, 3))
    for k, p in enumerate(plies):
        qb = qbar(reduced_stiffness(p.material), p.angle_deg)
        zk, zk1 = z[k + 1], z[k]
        A += qb * (zk - zk1)
        B += 0.5 * qb * (zk ** 2 - zk1 ** 2)
        D += (1.0 / 3.0) * qb * (zk ** 3 - zk1 ** 3)
    return A, B, D


def solve_midplane(plies: list[Ply], N: np.ndarray, M: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Solve [N;M] = ABD [ε⁰;κ] for mid-plane strain ε⁰ and curvature κ.

    N = [Nx, Ny, Nxy] force resultants [N/mm]; M = moment resultants [N] (default 0)."""
    A, B, D = abd_matrices(plies)
    M = np.zeros(3) if M is None else np.asarray(M, dtype=np.float64)
    abd = np.block([[A, B], [B, D]])
    rhs = np.concatenate([np.asarray(N, dtype=np.float64), M])
    sol = np.linalg.solve(abd, rhs)
    return sol[:3], sol[3:]


def ply_stresses(plies: list[Ply], eps0: np.ndarray, kappa: np.ndarray) -> list[dict]:
    """Per-ply stress/strain in MATERIAL coords, evaluated at each ply mid-thickness."""
    z = z_boundaries(plies)
    out = []
    for k, p in enumerate(plies):
        zmid = 0.5 * (z[k] + z[k + 1])
        eps_lam = eps0 + zmid * kappa                       # [εx, εy, γxy] laminate
        eps_mat = _strain_transform(p.angle_deg) @ eps_lam  # [ε1, ε2, γ12] material
        q = reduced_stiffness(p.material)
        sig_mat = q @ eps_mat                                # [σ1, σ2, τ12] material
        out.append({
            "ply": k, "angle_deg": p.angle_deg, "z_mid": zmid,
            "sigma_11": float(sig_mat[0]), "sigma_22": float(sig_mat[1]), "tau_12": float(sig_mat[2]),
            "eps_11": float(eps_mat[0]), "eps_22": float(eps_mat[1]), "gamma_12": float(eps_mat[2]),
        })
    return out


def hashin_ply(sigma_11: float, sigma_22: float, tau_12: float,
               allow: MaterialAllowables) -> dict[str, float]:
    """Plane-stress Hashin failure indices for one ply (material coords)."""
    xt, xc, yt, yc, s = allow.xt, allow.xc, allow.yt, allow.yc, allow.s
    ft = (sigma_11 / xt) ** 2 + (tau_12 / s) ** 2 if sigma_11 >= 0 else 0.0
    fc = (sigma_11 / xc) ** 2 if sigma_11 < 0 else 0.0
    if sigma_22 >= 0:
        mt = (sigma_22 / yt) ** 2 + (tau_12 / s) ** 2
        mc = 0.0
    else:
        mt = 0.0
        mc = ((sigma_22 / (2 * s)) ** 2 + ((yc / (2 * s)) ** 2 - 1.0) * (sigma_22 / yc) + (tau_12 / s) ** 2)
    fi = max(ft, fc, mt, mc)
    modes = {"fiber_tension": ft, "fiber_compression": fc, "matrix_tension": mt, "matrix_compression": mc}
    dominant = max(modes, key=modes.get)
    return {**modes, "failure_index": fi, "reserve_factor": (1.0 / np.sqrt(fi)) if fi > 1e-12 else np.inf,
            "dominant_mode": dominant}


def analyse_laminate(plies: list[Ply], N: np.ndarray, M: np.ndarray | None = None,
                     allow: MaterialAllowables | None = None) -> dict:
    """Full ply-by-ply analysis under membrane (+ optional bending) load.

    Returns per-ply stress + Hashin, and the laminate-critical ply / reserve factor."""
    allow = MaterialAllowables() if allow is None else allow
    eps0, kappa = solve_midplane(plies, N, M)
    plies_out = ply_stresses(plies, eps0, kappa)
    for pr in plies_out:
        pr["hashin"] = hashin_ply(pr["sigma_11"], pr["sigma_22"], pr["tau_12"], allow)
    fis = [pr["hashin"]["failure_index"] for pr in plies_out]
    crit = int(np.argmax(fis))
    return {
        "eps0": eps0, "kappa": kappa, "plies": plies_out,
        "laminate_fi_max": float(fis[crit]),
        "min_reserve_factor": float(1.0 / np.sqrt(max(fis[crit], 1e-12))),
        "critical_ply": crit,
        "critical_mode": plies_out[crit]["hashin"]["dominant_mode"],
    }


# ---------------------------------------------------------------------------
# COPV cylinder layup builder
# ---------------------------------------------------------------------------
def copv_cylinder_layup(helical_angle_deg: float, n_helical_pairs: int, n_hoop: int,
                        ply_thickness: float = 0.3, base_axial_plies: int = 4,
                        material: PlyMaterial | None = None) -> list[Ply]:
    """A representative COPV cylinder wall, symmetric + balanced.

    Half-stack of axial base (0°) + balanced ±helical pairs + 90° hoops, then mirrored
    about the mid-plane. Symmetric (B≈0, no membrane–bending coupling) and balanced
    (each +α has a -α), as a real filament-wound design is."""
    material = PlyMaterial() if material is None else material
    half: list[Ply] = [Ply(0.0, ply_thickness, material) for _ in range(base_axial_plies)]
    for _ in range(n_helical_pairs):
        half.append(Ply(+helical_angle_deg, ply_thickness, material))
        half.append(Ply(-helical_angle_deg, ply_thickness, material))
    half += [Ply(90.0, ply_thickness, material) for _ in range(n_hoop)]
    return half + list(reversed(half))   # mirror -> symmetric


def cylinder_load_resultants(pressure: float, radius: float) -> np.ndarray:
    """Thin-wall closed-cylinder membrane force resultants under internal pressure.

    Hoop  N_theta = p r ;  axial N_x = p r / 2  [N/mm].  Returned as [Nx, Ny=Nhoop, Nxy]."""
    return np.array([0.5 * pressure * radius, pressure * radius, 0.0], dtype=np.float64)
