"""Fast unit tests for the filament-winding design module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.winding import (
    clairaut_constant, dome_thickness_ratio, geodesic_angle_deg, helical_coverage,
    layers_for_thickness, winding_pattern, winding_summary,
)


def test_geodesic_clairaut_invariant():
    # angle from the opening; the Clairaut constant equals the polar radius at every r
    r_open = 20.0
    for r in (100.0, 80.0, 50.0, 25.0):
        a = geodesic_angle_deg(r, r_open)
        assert abs(clairaut_constant(r, a) - r_open) < 1e-9
    # at r = opening the tow turns around (alpha -> 90)
    assert abs(geodesic_angle_deg(r_open, r_open) - 90.0) < 1e-6


def test_coverage_counts_and_pitch():
    cov = helical_coverage(diameter=200.0, angle_deg=54.7356, band_width=6.0)
    # n = pi D sin(alpha) / b
    expect = math.pi * 200.0 * math.sin(math.radians(54.7356)) / 6.0
    assert abs(cov.n_bands - math.ceil(expect)) == 0
    # pitch * n approx circumference
    assert abs(cov.circ_pitch_mm * cov.n_bands - math.pi * 200.0) < 1e-6
    assert abs(cov.coverage_angle_deg - 360.0 / cov.n_bands) < 1e-9


def test_pattern_closure_coprime():
    # pattern closes only when the advance is coprime with the band count
    n = helical_coverage(200.0, 40.0, 6.0).n_bands
    assert winding_pattern(200.0, 40.0, 6.0, pattern_number=1).closes  # 1 is coprime with anything
    if n % 2 == 0:
        assert not winding_pattern(200.0, 40.0, 6.0, pattern_number=2).closes


def test_dome_thickness_buildup():
    R, r_open = 100.0, 20.0
    assert abs(dome_thickness_ratio(R, R, r_open) - 1.0) < 1e-9          # unity at the cylinder
    assert dome_thickness_ratio(60.0, R, r_open) > 1.0                   # thicker toward the boss
    assert dome_thickness_ratio(40.0, R, r_open) > dome_thickness_ratio(60.0, R, r_open)


def test_layers_and_summary():
    assert layers_for_thickness(3.0, 0.3) == 10
    s = winding_summary(diameter=200.0, cylinder_length=220.0, opening_radius=20.0)
    assert 0 < s["helical_angle_deg"] < 20                              # low angle from a small boss
    assert s["bands_per_layer"] > 0 and s["circuits_per_layer"] == s["bands_per_layer"]
    assert abs(s["clairaut_constant_mm"] - 20.0) < 1e-6
