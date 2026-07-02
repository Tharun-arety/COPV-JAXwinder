"""Tests for the isotensoid dome contour and the non-geodesic path solver."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.winding import isotensoid_dome, nongeodesic_angle_profile, slippage_coefficient


# ---- isotensoid dome ----
def test_isotensoid_dome_shape():
    r, z = isotensoid_dome(100.0, 20.0)
    assert r[0] == 100.0 and abs(z[0]) < 1e-9                 # starts at the equator
    assert np.all(np.diff(r) <= 1e-6)                         # radius monotonically decreases
    assert np.all(np.diff(z) >= -1e-6)                        # height monotonically increases
    assert r[-1] <= 22.0 and r[-1] >= 20.0                    # closes near the polar opening
    assert z[-1] > 0.2 * 100.0                                # a real dome height


def test_isotensoid_starts_vertical():
    r, z = isotensoid_dome(100.0, 20.0, n_steps=400)
    # first step: tangent nearly axial (dz >> |dr|) at the equator
    assert (z[1] - z[0]) > 10.0 * abs(r[1] - r[0])


# ---- non-geodesic solver ----
def test_geodesic_limit_is_clairaut():
    # slippage = 0 must preserve r*sin(alpha) = const on a converging meridian
    r = np.linspace(100.0, 55.0, 60)
    s = np.linspace(0.0, 60.0, 60)
    a = nongeodesic_angle_profile(r, s, np.zeros_like(r), np.zeros_like(r), 30.0, slippage=0.0)
    C = [r[i] * math.sin(math.radians(a[i])) for i in range(len(a))]
    assert max(C) - min(C) < 1e-3                             # Clairaut invariant
    assert abs(C[0] - 100.0 * math.sin(math.radians(30.0))) < 1e-6


def test_cylinder_constant_then_steers():
    r = np.full(50, 100.0)
    s = np.linspace(0.0, 200.0, 50)
    kp = np.full(50, 1.0 / 100.0)
    km = np.zeros(50)
    flat = nongeodesic_angle_profile(r, s, km, kp, 30.0, slippage=0.0)
    assert max(flat) - min(flat) < 1e-6                       # geodesic: constant angle on a cylinder
    steered = nongeodesic_angle_profile(r, s, km, kp, 30.0, slippage=0.15)
    assert steered[-1] > steered[0] + 1.0                     # non-geodesic steers the angle


def test_slippage_coefficient_cylinder_formula():
    # cylinder: lambda = r*(dalpha/ds)/sin^2(alpha)
    lam = slippage_coefficient(100.0, drds=0.0, angle_deg=30.0, dalpha_ds=0.001,
                               k_meridian=0.0, k_parallel=0.01)
    assert abs(lam - 100.0 * 0.001 / math.sin(math.radians(30.0)) ** 2) < 1e-6
    # geodesic (no steering) -> zero slippage
    assert abs(slippage_coefficient(100.0, 0.0, 30.0, 0.0, 0.0, 0.01)) < 1e-9
