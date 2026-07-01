"""Fast unit tests for netting analysis — no mesh, no JAX."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.netting import (
    OPTIMAL_HELICAL_ANGLE_DEG, clairaut_angle_deg, netting_burst_pressure, netting_cylinder,
)

P, R, SIGMA = 6.85, 100.0, 2200.0


def test_optimal_angle_is_atan_sqrt2():
    assert abs(OPTIMAL_HELICAL_ANGLE_DEG - math.degrees(math.atan(math.sqrt(2)))) < 1e-9
    assert abs(OPTIMAL_HELICAL_ANGLE_DEG - 54.7356) < 1e-3
    opt = netting_cylinder(P, R, SIGMA)
    assert opt.t_hoop < 1e-9 and not opt.hoop_required


def test_total_thickness_invariant():
    expect = 1.5 * P * R / SIGMA
    for a in (15.0, 30.0, 45.0, 54.0, OPTIMAL_HELICAL_ANGLE_DEG):
        res = netting_cylinder(P, R, SIGMA, angle_deg=a)
        assert abs(res.t_total - expect) / expect < 1e-6


def test_clairaut_angle_from_opening():
    assert abs(clairaut_angle_deg(50.0, 100.0) - 30.0) < 1e-6
    assert abs(clairaut_angle_deg(R * math.sin(math.radians(54.7356)), R) - 54.7356) < 1e-3


def test_shallower_angle_needs_more_hoop():
    assert netting_cylinder(P, R, SIGMA, angle_deg=20).hoop_fraction > netting_cylinder(P, R, SIGMA, angle_deg=50).hoop_fraction


def test_sized_design_carries_design_pressure():
    res = netting_cylinder(P, R, SIGMA, angle_deg=30.0)
    assert abs(netting_burst_pressure(R, SIGMA, 30.0, res.t_helical, res.t_hoop) - P) / P < 1e-6
