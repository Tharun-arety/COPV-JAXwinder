"""Fast unit tests for the ply-by-ply CLT engine — no mesh, no JAX."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.clt import (
    Ply, PlyMaterial, abd_matrices, analyse_laminate, copv_cylinder_layup,
    cylinder_load_resultants, reduced_stiffness, solve_midplane,
)

MAT = PlyMaterial()
T = 0.3


def test_single_ply_uniaxial_analytic():
    q = reduced_stiffness(MAT)
    eps0, kappa = solve_midplane([Ply(0.0, T, MAT)], [100.0, 0.0, 0.0])
    sig11 = q[0, 0] * eps0[0] + q[0, 1] * eps0[1]
    assert abs(sig11 - 100.0 / T) / (100.0 / T) < 1e-6
    assert np.allclose(kappa, 0.0, atol=1e-10)


def test_symmetric_laminate_B_zero():
    _, B, _ = abd_matrices([Ply(0.0, T, MAT), Ply(90.0, T, MAT), Ply(90.0, T, MAT), Ply(0.0, T, MAT)])
    assert np.max(np.abs(B)) < 1e-6


def test_cross_ply_A11_exact():
    q = reduced_stiffness(MAT)
    A, _, _ = abd_matrices([Ply(0.0, T, MAT), Ply(90.0, T, MAT)])
    assert abs(A[0, 0] - (q[0, 0] + q[1, 1]) * T) / ((q[0, 0] + q[1, 1]) * T) < 1e-6


def test_balanced_angle_ply_no_A16_A26():
    A, _, _ = abd_matrices([Ply(45.0, T, MAT), Ply(-45.0, T, MAT)])
    assert abs(A[0, 2]) < 1e-6 and abs(A[1, 2]) < 1e-6


def test_quasi_isotropic_inplane_isotropy():
    qi = [Ply(a, T, MAT) for a in (0, 45, -45, 90, 90, -45, 45, 0)]
    A, _, _ = abd_matrices(qi)
    assert abs(A[0, 0] - A[1, 1]) / A[0, 0] < 1e-6
    assert abs(A[0, 2]) < 1e-6 and abs(A[1, 2]) < 1e-6
    assert abs(A[2, 2] - 0.5 * (A[0, 0] - A[0, 1])) / A[2, 2] < 1e-6


def test_symmetric_copv_layup_membrane_only_and_per_ply_resolution():
    layup = copv_cylinder_layup(helical_angle_deg=25.0, n_helical_pairs=6, n_hoop=8)
    res = analyse_laminate(layup, cylinder_load_resultants(pressure=30.0, radius=100.0))
    assert np.allclose(res["kappa"], 0.0, atol=1e-9)          # symmetric -> no bending
    s11 = {round(p["angle_deg"]): p["sigma_11"] for p in res["plies"]}
    assert len({round(v, -1) for v in s11.values()}) > 1      # plies differ (not smeared)
    assert s11[90] == max(s11.values())                        # hoop fibres carry the most
