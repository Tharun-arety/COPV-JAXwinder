"""Tests for the requirement -> geometry sizing front door (no mesh, no JAX)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.sizing import TankRequirement, geometry_from_requirement


def test_volume_round_trip_exact():
    req = TankRequirement(internal_volume_litres=9.0, design_pressure_bar=300.0,
                          envelope_outer_radius_mm=100.0)
    geom, rep = geometry_from_requirement(req)
    assert abs(rep.achieved_volume_litres - 9.0) < 1e-9
    assert geom.cylinder_length > 0
    assert abs(geom.pressure - 30.0) < 1e-12          # 300 bar -> 30 MPa


def test_infeasible_envelope_raises():
    # domes alone exceed the requested volume -> must refuse, not return negative length
    with pytest.raises(ValueError):
        geometry_from_requirement(TankRequirement(internal_volume_litres=0.5,
                                                  design_pressure_bar=100.0,
                                                  envelope_outer_radius_mm=200.0))


def test_wall_thicker_than_radius_raises():
    with pytest.raises(ValueError):
        geometry_from_requirement(TankRequirement(internal_volume_litres=5.0,
                                                  design_pressure_bar=100.0,
                                                  envelope_outer_radius_mm=10.0,
                                                  wall_thickness_mm=12.0))


def test_length_override_bypasses_volume_solve():
    req = TankRequirement(internal_volume_litres=9.0, design_pressure_bar=100.0,
                          envelope_outer_radius_mm=100.0, cylinder_length_override_mm=150.0)
    geom, _ = geometry_from_requirement(req)
    assert geom.cylinder_length == 150.0
