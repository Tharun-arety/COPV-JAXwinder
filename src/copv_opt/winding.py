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

import numpy as np


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


@dataclass
class PatternDesign:
    n_bands: int              # bands for 100% single-layer coverage
    circuits: int             # circuits to reach the requested coverage
    coverage_achieved: float  # actual coverage fraction (>=1 = full)
    pattern_number: int       # evenly-spreading advance, coprime with n_bands
    closes: bool


def _even_pattern_number(n: int) -> int:
    """A pattern advance coprime with n, near golden-ratio spacing for even coverage."""
    if n <= 2:
        return 1
    target = max(1, round(n * 0.381966))
    for d in range(0, n):
        for p in (target + d, target - d):
            if 1 <= p < n and math.gcd(p, n) == 1:
                return p
    return 1


def pattern_for_coverage(diameter: float, angle_deg: float, band_width: float,
                         target_coverage: float = 1.0) -> PatternDesign:
    """Find the winding pattern that achieves a requested tape coverage.

    Mirrors TaniqWind Pro's 'coverage path': enter a desired coverage (e.g. 2.0 = 200%)
    and get the circuit count + an evenly-spreading, gap-free p/n pattern. Coverage
    fraction achieved = circuits / n_bands."""
    cov = helical_coverage(diameter, angle_deg, band_width)
    n = cov.n_bands
    circuits = max(1, math.ceil(n * max(target_coverage, 1e-6)))
    achieved = circuits / n
    p = _even_pattern_number(n)
    return PatternDesign(n_bands=n, circuits=circuits, coverage_achieved=achieved,
                         pattern_number=p, closes=math.gcd(p, n) == 1)


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
# ---------------------------------------------------------------------------
# Isotensoid dome contour (netting + membrane equilibrium, geodesic winding)
# ---------------------------------------------------------------------------
def isotensoid_dome(equator_radius: float, opening_radius: float, n_steps: int = 800,
                    alpha_cap_deg: float = 85.0) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic-isotensoid dome meridian (r, z) from the equator to the polar opening.

    The dome shape where, under netting (fibres carry all load) with geodesic winding
    (sin alpha = r_open/r), every fibre is at equal tension. Derived from membrane
    equilibrium: with N_theta/N_phi = tan^2(alpha) and N_phi = p r/(2 sin phi), Laplace
    gives the meridional curvature 1/R1 = 2 cos(psi)(1 - tan^2(alpha)/2)/r, integrated in
    arc length s with the tangent angle psi (0 at the equator = vertical tangent):

        dr/ds = -sin(psi),  dz/ds = cos(psi),  dpsi/ds = 2 cos(psi)(1 - tan^2(alpha)/2)/r

    The angle is capped at ``alpha_cap_deg`` to keep the near-boss integration stable.
    Note the inflection at alpha = 54.74 deg (dpsi/ds = 0) — the isotensoid signature."""
    R, r0 = float(equator_radius), float(opening_radius)
    sa2_cap = math.sin(math.radians(alpha_cap_deg)) ** 2
    ds = 2.0 * R / n_steps

    def deriv(r, psi):
        sa2 = min((r0 / r) ** 2, sa2_cap) if r > 0 else sa2_cap
        tan2 = sa2 / max(1.0 - sa2, 1e-9)
        return -math.sin(psi), math.cos(psi), 2.0 * math.cos(psi) * (1.0 - tan2 / 2.0) / max(r, 1e-6)

    r, z, psi = R, 0.0, 0.0
    rs, zs = [r], [z]
    for _ in range(n_steps * 4):
        if r <= r0:
            break
        k1 = deriv(r, psi)
        k2 = deriv(r + 0.5 * ds * k1[0], psi + 0.5 * ds * k1[2])
        k3 = deriv(r + 0.5 * ds * k2[0], psi + 0.5 * ds * k2[2])
        k4 = deriv(r + ds * k3[0], psi + ds * k3[2])
        r += ds * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        z += ds * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        psi += ds * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
        rs.append(max(r, r0))
        zs.append(z)
    return np.asarray(rs), np.asarray(zs)


# ---------------------------------------------------------------------------
# Non-geodesic path solver (slippage-limited steering)
# ---------------------------------------------------------------------------
def slippage_coefficient(radius: float, drds: float, angle_deg: float, dalpha_ds: float,
                         k_meridian: float, k_parallel: float) -> float:
    """Slippage coefficient lambda = k_g/k_n for a tow at ``angle_deg`` (from the meridian).

    k_g is the Clairaut-deviation geodesic curvature (1/(r cos a))·d(r sin a)/ds — zero
    for a geodesic; k_n = k_meridian cos^2 a + k_parallel sin^2 a (Euler). Manufacturable
    when |lambda| <= friction/tack coefficient."""
    a = math.radians(angle_deg)
    dC_ds = math.sin(a) * drds + radius * math.cos(a) * dalpha_ds  # d(r sin a)/ds
    k_g = dC_ds / (radius * math.cos(a)) if abs(math.cos(a)) > 1e-6 else float("inf")
    k_n = k_meridian * math.cos(a) ** 2 + k_parallel * math.sin(a) ** 2
    return k_g / k_n if abs(k_n) > 1e-12 else float("inf")


def nongeodesic_angle_profile(radii, s_coords, k_meridian, k_parallel,
                              alpha0_deg: float, slippage: float) -> list[float]:
    """Integrate the winding-angle profile for a non-geodesic path at constant slippage.

    Integrates the Clairaut quantity C = r·sin(a):  dC/ds = lambda·r cos a·(k_m cos^2 a +
    k_p sin^2 a), then recovers a = asin(C/r). slippage = 0 keeps C constant exactly —
    the geodesic (Clairaut) limit. Returns alpha [deg]."""
    r = np.asarray(radii, float); s = np.asarray(s_coords, float)
    km = np.asarray(k_meridian, float); kp = np.asarray(k_parallel, float)
    C = r[0] * math.sin(math.radians(alpha0_deg))
    out = [alpha0_deg]
    for i in range(1, len(s)):
        ds = s[i] - s[i - 1]
        rr = max(r[i - 1], 1e-9)
        sa = min(max(C / rr, -1.0), 1.0)
        ca = math.sqrt(max(1.0 - sa * sa, 0.0))
        C += slippage * rr * ca * (km[i - 1] * ca * ca + kp[i - 1] * sa * sa) * ds
        out.append(math.degrees(math.asin(min(max(C / max(r[i], 1e-9), -1.0), 1.0))))
    return out


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
