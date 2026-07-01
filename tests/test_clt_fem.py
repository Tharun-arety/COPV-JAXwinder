"""Unit tests for the element-level CLT math (pure, no mesh)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.clt import PlyMaterial, reduced_stiffness
from copv_opt.clt_fem import _hashin, _ply_stress
from copv_opt.config import MaterialAllowables, MaterialConfig

MAT = MaterialConfig()
Q = reduced_stiffness(PlyMaterial(e1=MAT.e_xx, e2=MAT.e_yy, g12=MAT.g_xy, nu12=MAT.nu_xy))


def test_ply_stress_zero_deg_uniaxial():
    # 0-deg ply, strain [e, 0, 0]: s11 = Q11 e, s22 = Q12 e, t12 = 0
    e = 1e-3
    eps0 = np.array([[e, 0.0, 0.0]])
    s11, s22, t12 = _ply_stress(eps0, np.array([0.0]), Q)
    assert abs(s11[0] - Q[0, 0] * e) < 1e-6
    assert abs(s22[0] - Q[0, 1] * e) < 1e-6
    assert abs(t12[0]) < 1e-9


def test_ply_stress_90_deg_fiber_along_y():
    # 90-deg ply: fiber strain e1 <- ey, transverse e2 <- ex
    ex, ey = 1e-3, 3e-3
    eps0 = np.array([[ex, ey, 0.0]])
    s11_90, s22_90, _ = _ply_stress(eps0, np.array([90.0]), Q)
    assert abs(s11_90[0] - (Q[0, 0] * ey + Q[0, 1] * ex)) < 1e-6
    assert abs(s22_90[0] - (Q[0, 1] * ey + Q[1, 1] * ex)) < 1e-6


def test_hashin_fiber_tension_at_allowable():
    a = MaterialAllowables()
    fi = _hashin(np.array([a.xt]), np.array([0.0]), np.array([0.0]), a)
    assert abs(fi[0] - 1.0) < 1e-6


def test_hashin_matrix_tension_at_allowable():
    a = MaterialAllowables()
    fi = _hashin(np.array([0.0]), np.array([a.yt]), np.array([0.0]), a)
    assert abs(fi[0] - 1.0) < 1e-6
