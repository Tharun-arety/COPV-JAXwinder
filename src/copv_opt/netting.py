"""Netting analysis — classical fiber-only COPV sizing (the TaniqWind-Pro core).

Netting theory assumes the fibers carry ALL the load and the matrix carries none. For
a thin cylinder under internal pressure it gives closed-form laminate sizing and the
optimal winding angle — the first-principles design layer that sits upstream of the
FEM (physics.py) and the winding optimizer (optimize.py).

Cylinder membrane resultants: N_hoop = p·r, N_axial = p·r/2.
A balanced ±alpha helical layer of fiber-thickness t at fiber stress sigma carries
N_axial = sigma·t·cos^2(alpha), N_hoop = sigma·t·sin^2(alpha). Hoop (90 deg) layers
carry only N_hoop = sigma·t_hoop. Equilibrium gives:

    t_helical = p·r / (2·sigma·cos^2 alpha)
    t_hoop    = (p·r/sigma)·(1 - tan^2(alpha)/2)

so the total fiber thickness is p·r/sigma·(1/2 + 1) = 1.5·p·r/sigma — invariant with
angle while hoop layers are needed. Helical-only (t_hoop = 0) requires tan^2(alpha)=2,
i.e. alpha = atan(sqrt(2)) = 54.7356 deg (the classic netting angle). The helical angle
is otherwise set by the polar opening through the geodesic Clairaut relation
sin(alpha) = r_open / r.

Verified in app/verify_netting.py / tests/test_netting.py against these closed forms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Optimal helical-only winding angle for a closed cylinder: atan(sqrt(2)).
OPTIMAL_HELICAL_ANGLE_DEG = math.degrees(math.atan(math.sqrt(2.0)))  # 54.7356...


@dataclass
class NettingResult:
    angle_deg: float          # helical winding angle from the cylinder axis
    t_helical: float          # structural helical fibre thickness [mm]
    t_hoop: float             # structural hoop fibre thickness [mm]
    t_total: float            # total structural fibre thickness [mm]
    helical_fraction: float
    hoop_fraction: float
    hoop_required: bool       # False when alpha >= 54.74 deg (helical carries hoop alone)
    fiber_allowable: float    # fibre-direction stress used for sizing [MPa]


def clairaut_angle_deg(opening_radius: float, cylinder_radius: float) -> float:
    """Geodesic helical angle at the cylinder set by the polar opening: asin(r_open/r)."""
    return math.degrees(math.asin(min(max(opening_radius / cylinder_radius, 0.0), 1.0)))


def netting_cylinder(pressure: float, radius: float, fiber_allowable: float = 2200.0,
                     angle_deg: float | None = None, opening_radius: float | None = None) -> NettingResult:
    """Size the cylinder laminate by netting analysis.

    The helical angle comes from ``angle_deg`` if given, else from the polar opening via
    Clairaut (``opening_radius``), else the helical-only optimum (54.74 deg). Thicknesses
    are the structural fibre thicknesses (divide by fibre volume fraction for ply build)."""
    if angle_deg is None:
        angle_deg = (clairaut_angle_deg(opening_radius, radius)
                     if opening_radius is not None else OPTIMAL_HELICAL_ANGLE_DEG)
    a = math.radians(angle_deg)
    ca2 = math.cos(a) ** 2
    t_helical = pressure * radius / (2.0 * fiber_allowable * max(ca2, 1e-9))
    t_hoop_raw = (pressure * radius / fiber_allowable) * (1.0 - math.tan(a) ** 2 / 2.0)
    t_hoop = max(t_hoop_raw, 0.0)
    t_total = t_helical + t_hoop
    return NettingResult(
        angle_deg=angle_deg, t_helical=t_helical, t_hoop=t_hoop, t_total=t_total,
        helical_fraction=t_helical / t_total if t_total > 0 else 0.0,
        hoop_fraction=t_hoop / t_total if t_total > 0 else 0.0,
        hoop_required=t_hoop_raw > 1e-12, fiber_allowable=fiber_allowable,
    )


def netting_burst_pressure(radius: float, fiber_ultimate: float, angle_deg: float,
                           t_helical: float, t_hoop: float) -> float:
    """Netting burst pressure of a laminate: the pressure at which the fibre stress
    reaches ``fiber_ultimate`` (the binding of the axial and hoop capacities)."""
    a = math.radians(angle_deg)
    p_axial = 2.0 * fiber_ultimate * t_helical * math.cos(a) ** 2 / radius
    p_hoop = fiber_ultimate * (t_helical * math.sin(a) ** 2 + t_hoop) / radius
    return min(p_axial, p_hoop)


def netting_design(pressure: float, radius: float, opening_radius: float,
                   fiber_allowable: float = 2200.0, fiber_ultimate: float = 2500.0,
                   ply_thickness: float = 0.3, fiber_volume_fraction: float = 0.6) -> dict:
    """A full first-principles netting design point for a COPV cylinder.

    Angle from the boss opening (Clairaut), netting thicknesses, ply counts, burst
    pressure and burst factor. This is the kind of instant sizing TaniqWind Pro leads
    with, upstream of the FEM/optimization."""
    res = netting_cylinder(pressure, radius, fiber_allowable, opening_radius=opening_radius)
    # structural fibre thickness -> ply thickness via fibre volume fraction
    ply_t = ply_thickness * fiber_volume_fraction
    n_helical_pairs = max(1, round(res.t_helical / (2.0 * ply_t)))
    n_hoop = round(res.t_hoop / ply_t)
    burst = netting_burst_pressure(radius, fiber_ultimate, res.angle_deg, res.t_helical, res.t_hoop)
    return {
        "helical_angle_deg": res.angle_deg,
        "t_helical_mm": res.t_helical, "t_hoop_mm": res.t_hoop, "t_total_mm": res.t_total,
        "helical_pairs": n_helical_pairs, "hoop_rings": n_hoop,
        "burst_pressure_mpa": burst, "burst_factor": burst / pressure if pressure > 0 else float("inf"),
        "hoop_required": res.hoop_required,
    }
