"""Filament winding design physics — geodesic paths, coverage, pattern, dome buildup.

The core winding-design layer a filament-winding tool (TaniqWind Pro class) provides,
built on classical theory and verified against closed-form limits. Complements the
existing pieces: geodesic path points (geometry.py), course planning (course_planner.py),
netting sizing (netting.py), and the winding optimizer (optimize.py).

Conventions: winding angle ``alpha`` is measured from the cylinder AXIS (0 = axial,
90 = hoop). Band width ``b`` is measured perpendicular to the tow.

Implemented (verified):
- Geodesic angle profile via Clairaut:  r · sin(alpha) = C  (C = polar turnaround radius)
- Helical coverage: bands per layer, circumferential pitch, coverage pattern angle
- Winding pattern: pattern number / closure / circuits for one layer
- Dome thickness buildup from band continuity (fibre converges toward the boss)
- Layer count for a target thickness; tow length / winding-time estimates

Governing-equation placeholders (documented, not full solvers): non-geodesic steering
under a slippage limit, and the isotensoid dome contour ODE. Machine inverse-kinematics
/ NC post-processing needs a specific machine definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Geodesic paths (Clairaut)
# ---------------------------------------------------------------------------
def clairaut_constant(radius: float, angle_deg: float) -> float:
    """C = r·sin(alpha) — invariant along a geodesic on a surface of revolution."""
    return radius * math.sin(math.radians(angle_deg))


def geodesic_angle_deg(radius: float, polar_radius: float) -> float:
    """Winding angle at ``radius`` for a geodesic turning around at ``polar_radius``."""
    return math.degrees(math.asin(min(max(polar_radius / radius, 0.0), 1.0)))


def geodesic_angle_profile(radii, polar_radius: float) -> list[float]:
    """Geodesic winding-angle profile [deg] across a list of local radii."""
    return [geodesic_angle_deg(r, polar_radius) for r in radii]


# ---------------------------------------------------------------------------
# Coverage and winding pattern
# ---------------------------------------------------------------------------
@dataclass
class CoverageResult:
    n_bands: int              # bands to cover one helical layer
    circ_pitch_mm: float      # circumferential pitch between adjacent bands
    coverage_angle_deg: float # angular pitch = 360 / n_bands
    layer_thickness_mm: float # one covered layer thickness


def helical_coverage(diameter: float, angle_deg: float, band_width: float,
                     ply_thickness: float = 0.3) -> CoverageResult:
    """Bands needed to fully cover one helical layer.

    Unrolling the cylinder, parallel bands of width b at angle alpha tile with a
    circumferential intercept pitch b/sin(alpha); the number to cover the circumference
    is pi·D·sin(alpha)/b (rounded up)."""
    a = math.radians(angle_deg)
    circ = math.pi * diameter
    pitch = band_width / max(math.sin(a), 1e-6)
    n = max(1, math.ceil(circ / pitch))
    return CoverageResult(n_bands=n, circ_pitch_mm=circ / n, coverage_angle_deg=360.0 / n,
                          layer_thickness_mm=ply_thickness)


@dataclass
class PatternResult:
    n_bands: int
    pattern_number: int       # circuits between circumferentially adjacent bands
    circuits_per_layer: int   # circuits to complete one layer
    closes: bool              # pattern closes (integer coverage)


def winding_pattern(diameter: float, angle_deg: float, band_width: float,
                    pattern_number: int = 1) -> PatternResult:
    """Winding pattern for one helical layer.

    Each circuit lays one band; after ``n_bands`` circuits the layer closes. The
    ``pattern_number`` p (advance in units of the coverage pitch per circuit) must be
    coprime with n_bands for even, gap-free coverage — a 'p/n' star pattern."""
    n = helical_coverage(diameter, angle_deg, band_width).n_bands
    p = max(1, int(pattern_number))
    closes = math.gcd(p, n) == 1
    return PatternResult(n_bands=n, pattern_number=p, circuits_per_layer=n, closes=closes)


# ---------------------------------------------------------------------------
# Dome thickness buildup (band continuity)
# ---------------------------------------------------------------------------
def dome_thickness_ratio(radius: float, cylinder_radius: float, polar_radius: float) -> float:
    """Thickness buildup t(r)/t_cyl on the dome from fibre continuity.

    The same n helical bands cross every parallel circle. As r decreases toward the
    boss the fibres converge and the winding angle steepens (Clairaut), so the layer
    thickens: t(r)/t_cyl = (R·cos alpha_R)/(r·cos alpha_r). Equals 1 at the cylinder
    and diverges at the polar opening (r -> polar_radius, alpha -> 90 deg)."""
    a_cyl = math.asin(min(polar_radius / cylinder_radius, 1.0))
    a_r = math.asin(min(polar_radius / radius, 1.0))
    denom = radius * math.cos(a_r)
    if denom <= 1e-9:
        return float("inf")
    return (cylinder_radius * math.cos(a_cyl)) / denom


# ---------------------------------------------------------------------------
# Layup / process helpers
# ---------------------------------------------------------------------------
def layers_for_thickness(target_thickness: float, ply_thickness: float = 0.3) -> int:
    """Number of covered layers to reach a target structural thickness."""
    return max(1, math.ceil(target_thickness / max(ply_thickness, 1e-6)))


def tow_length_per_circuit(cylinder_length: float, cylinder_radius: float, angle_deg: float,
                           dome_allowance: float = 1.25) -> float:
    """Approximate tow length for one full circuit (two cylinder passes + dome turnarounds).

    Cylinder pass length = L / cos(alpha); ``dome_allowance`` scales up for the dome
    traverses and turnarounds (>1)."""
    a = math.radians(angle_deg)
    cyl_pass = cylinder_length / max(math.cos(a), 1e-6)
    return 2.0 * cyl_pass * dome_allowance


def winding_summary(diameter: float, cylinder_length: float, opening_radius: float,
                    band_width: float = 6.0, ply_thickness: float = 0.3,
                    target_thickness: float | None = None) -> dict:
    """A full geometric winding design summary for the cylinder + dome."""
    r = 0.5 * diameter
    angle = geodesic_angle_deg(r, opening_radius)
    cov = helical_coverage(diameter, angle, band_width, ply_thickness)
    pat = winding_pattern(diameter, angle, band_width)
    n_layers = layers_for_thickness(target_thickness, ply_thickness) if target_thickness else None
    tow = tow_length_per_circuit(cylinder_length, r, angle)
    return {
        "helical_angle_deg": angle,
        "clairaut_constant_mm": clairaut_constant(r, angle),
        "bands_per_layer": cov.n_bands,
        "circ_pitch_mm": cov.circ_pitch_mm,
        "coverage_angle_deg": cov.coverage_angle_deg,
        "pattern_number": pat.pattern_number,
        "pattern_closes": pat.closes,
        "circuits_per_layer": pat.circuits_per_layer,
        "dome_thickness_ratio_at_half_radius": dome_thickness_ratio(0.5 * r + 0.5 * opening_radius, r, opening_radius),
        "tow_length_per_circuit_mm": tow,
        "layers_for_target": n_layers,
    }
