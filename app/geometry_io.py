"""Phase 2 — geometry input/output.

What is real here:
* ``export_step`` writes the configured vessel as a STEP solid (via the engine's
  OpenCASCADE-backed builder) for handoff to CAD or a winding tool.
* ``liner_mass`` does first-order geometric mass accounting for a metallic/polymer
  liner — surface area times thickness times density. It is NOT coupled into the
  structural solve; it is a mass/handoff figure only.

What is deliberately a stub:
* ``import_mandrel`` — the FEA state builder assumes the parametric ellipsoidal COPV
  (meridian arc-length, boss aperture, dome ratio). Accepting an arbitrary imported
  mandrel needs a generalized state builder, which is a later phase. We raise a clear
  error rather than silently meshing something the physics can't interpret.
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
from copv_opt.geometry import build_copv_shell


def export_step(geom: GeometryConfig, path: str | Path) -> Path:
    """Write the configured vessel shell as a STEP file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_copv_shell(
        path,
        outer_radius=geom.outer_radius,
        cylinder_length=geom.cylinder_length,
        thickness=geom.thickness,
        opening_radius=geom.opening_radius,
        dome_height_ratio=geom.dome_height_ratio,
    )
    return path


def _ellipsoid_cap_area(r_eq: float, h_polar: float) -> float:
    """Lateral surface area of one ellipsoidal dome cap (half-spheroid), by a
    numerically robust meridian revolution integral. r_eq = equatorial radius,
    h_polar = polar semi-axis (= dome_height_ratio * r_eq)."""
    n = 400
    # meridian from equator (t=0) to pole (t=pi/2): rho = r_eq cos t, z = h sin t
    ts = [0.5 * math.pi * i / n for i in range(n + 1)]
    area = 0.0
    for i in range(n):
        t0, t1 = ts[i], ts[i + 1]
        rho0, rho1 = r_eq * math.cos(t0), r_eq * math.cos(t1)
        z0, z1 = h_polar * math.sin(t0), h_polar * math.sin(t1)
        dl = math.hypot(rho1 - rho0, z1 - z0)
        area += math.pi * (rho0 + rho1) * dl   # frustum lateral area = pi (r0+r1) * slant
    return area


@dataclass
class LinerReport:
    liner_thickness_mm: float
    liner_density_t_per_mm3: float
    inner_surface_area_mm2: float
    liner_volume_mm3: float
    liner_mass_g: float


def liner_mass(
    geom: GeometryConfig,
    liner_thickness_mm: float,
    liner_density_t_per_mm3: float = 9.4e-10,   # ~ HDPE default (940 kg/m^3)
) -> LinerReport:
    """First-order liner mass from inner-surface area x thickness x density.

    Default density is a polymer (HDPE) liner; pass a metal density for a Type-III
    vessel. Mass is reported in grams. This figure feeds the mass budget; it does not
    enter the structural screen."""
    r_in = geom.inner_radius
    h_polar = geom.dome_height_ratio * r_in
    cyl_area = 2.0 * math.pi * r_in * geom.cylinder_length
    dome_area = 2.0 * _ellipsoid_cap_area(r_in, h_polar)   # two caps
    area = cyl_area + dome_area
    volume = area * liner_thickness_mm
    mass_t = volume * liner_density_t_per_mm3               # tonnes (MPa-mm-t system)
    return LinerReport(
        liner_thickness_mm=float(liner_thickness_mm),
        liner_density_t_per_mm3=float(liner_density_t_per_mm3),
        inner_surface_area_mm2=float(area),
        liner_volume_mm3=float(volume),
        liner_mass_g=float(mass_t * 1.0e6),                # 1 t = 1e6 g
    )


def import_mandrel(path: str | Path):  # pragma: no cover - intentional stub
    """Not yet supported. The structural engine assumes the parametric ellipsoidal
    COPV; arbitrary imported mandrels require a generalized FEA state builder."""
    raise NotImplementedError(
        "Arbitrary mandrel import is not supported in this phase. The FEA state "
        "builder is specialized to the parametric ellipsoidal COPV (meridian "
        "arc-length, boss aperture, dome ratio). Use the requirement-driven "
        "parametric builder, or extend build_copv_fem_state to accept a general "
        "axisymmetric profile before wiring import here."
    )
