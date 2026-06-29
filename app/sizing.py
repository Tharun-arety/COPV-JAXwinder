"""Requirement spec -> vessel geometry.

This is the front door of the configurator. A customer states *what the tank must
do* (hold V litres at P bar within a radius envelope); this module solves for the
geometry the engine needs (cylinder length, internal pressure in solver units).

Pure NumPy/math — no JAX, no mesh — so it imports fast and is unit-testable.

Volume model
------------
Internal volume = cylinder + two ellipsoidal dome caps, all on the *inner* radius
Ri = outer_radius - wall_thickness:

    V = pi * Ri^2 * L           (cylinder)
      + (4/3) * pi * Ri^3 * h   (two half-ellipsoids, polar semi-axis = h * Ri)

where h = dome_height_ratio. Given a target V we solve for L. If the dome caps
alone already exceed V, the envelope radius is too large for the requested volume
and we say so rather than returning a negative length.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from copv_opt.config import GeometryConfig

_LITRE_MM3 = 1.0e6   # 1 litre = 1e6 mm^3
_BAR_MPA = 0.1       # 1 bar = 0.1 MPa


@dataclass
class TankRequirement:
    """What the customer specifies. SI-friendly, human units (litres, bar, mm)."""

    internal_volume_litres: float
    design_pressure_bar: float
    envelope_outer_radius_mm: float
    wall_thickness_mm: float = 8.0          # structural base/liner thickness for the geometry shell
    opening_radius_mm: float = 10.0         # boss aperture radius
    dome_height_ratio: float = 0.7          # isotensoidal-ish ellipsoidal dome
    cylinder_length_override_mm: float | None = None  # bypass the volume solve if set


@dataclass
class SizingReport:
    """What the volume solve produced, for display back to the user."""

    cylinder_length_mm: float
    inner_radius_mm: float
    design_pressure_mpa: float
    dome_volume_litres: float
    cylinder_volume_litres: float
    achieved_volume_litres: float
    slenderness_l_over_d: float


def geometry_from_requirement(req: TankRequirement) -> tuple[GeometryConfig, SizingReport]:
    """Solve a TankRequirement into a (GeometryConfig, SizingReport).

    Raises ValueError when the requirement is geometrically infeasible — a too-thick
    wall, an opening larger than the bore, or an envelope so wide the domes alone
    overshoot the target volume.
    """
    r_out = float(req.envelope_outer_radius_mm)
    t = float(req.wall_thickness_mm)
    r_in = r_out - t
    if r_in <= 0.0:
        raise ValueError(f"wall_thickness ({t} mm) must be smaller than outer radius ({r_out} mm)")
    if r_in <= req.opening_radius_mm:
        raise ValueError(
            f"inner radius ({r_in:.1f} mm) must exceed the opening radius ({req.opening_radius_mm} mm)"
        )

    h = float(req.dome_height_ratio)
    if h <= 0.0:
        raise ValueError(f"dome_height_ratio must be > 0, got {h}")

    dome_volume_mm3 = (4.0 / 3.0) * math.pi * r_in**3 * h
    target_mm3 = float(req.internal_volume_litres) * _LITRE_MM3

    if req.cylinder_length_override_mm is not None:
        length = float(req.cylinder_length_override_mm)
        if length < 0.0:
            raise ValueError("cylinder_length_override_mm must be >= 0")
    else:
        cyl_volume_mm3 = target_mm3 - dome_volume_mm3
        if cyl_volume_mm3 < 0.0:
            dome_litres = dome_volume_mm3 / _LITRE_MM3
            raise ValueError(
                f"Envelope radius {r_out:.0f} mm is too large for {req.internal_volume_litres:.1f} L: "
                f"the dome caps alone hold {dome_litres:.2f} L. Reduce the radius or raise the volume."
            )
        length = cyl_volume_mm3 / (math.pi * r_in**2)

    pressure_mpa = float(req.design_pressure_bar) * _BAR_MPA

    geom = GeometryConfig(
        outer_radius=r_out,
        cylinder_length=length,
        thickness=t,
        opening_radius=float(req.opening_radius_mm),
        dome_height_ratio=h,
        pressure=pressure_mpa,
    )

    cyl_volume_mm3 = math.pi * r_in**2 * length
    achieved_mm3 = cyl_volume_mm3 + dome_volume_mm3
    overall_length = length + 2.0 * h * r_in
    report = SizingReport(
        cylinder_length_mm=length,
        inner_radius_mm=r_in,
        design_pressure_mpa=pressure_mpa,
        dome_volume_litres=dome_volume_mm3 / _LITRE_MM3,
        cylinder_volume_litres=cyl_volume_mm3 / _LITRE_MM3,
        achieved_volume_litres=achieved_mm3 / _LITRE_MM3,
        slenderness_l_over_d=overall_length / (2.0 * r_out),
    )
    return geom, report
