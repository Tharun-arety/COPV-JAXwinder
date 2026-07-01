"""Metal / polymer liner model for Type 3 and Type 4 COPVs.

Adds the piece the design guide (sections 4.1, 8, simulation steps 8-9) calls for:
an isotropic structural liner that carries load, can be solved on its own ("solve the
aluminium tank first"), and shares load with the composite overwrap.

An isotropic liner is a special orthotropic ply (E1=E2=E, nu12=nu, G12=E/2(1+nu)), so
it drops straight into the verified CLT stack. Liner strength is judged by von Mises
against yield; the composite plies keep the Hashin criteria.

Verified in app/verify_liner.py against the thin-wall closed-form:
    hoop  = p r / t ,  axial = p r / 2t ,  von Mises = sqrt(3)/2 * p r / t .
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from copv_opt.clt import (
    Ply, PlyMaterial, hashin_ply, ply_stresses, solve_midplane,
)
from copv_opt.config import MaterialAllowables


@dataclass
class LinerMaterial:
    name: str
    E: float            # Young's modulus [MPa]
    nu: float           # Poisson ratio
    yield_strength: float  # 0.2% proof / yield [MPa]
    density: float = 2.70e-9  # [t/mm^3]


# Representative handbook values — replace with certified material data before use.
PRESETS: dict[str, LinerMaterial] = {
    "AL6061-T6": LinerMaterial("AL6061-T6", 68900.0, 0.33, 276.0, 2.70e-9),
    "AL7075-T6": LinerMaterial("AL7075-T6", 71700.0, 0.33, 503.0, 2.81e-9),
    "SS316": LinerMaterial("SS316", 193000.0, 0.30, 290.0, 8.00e-9),
    "Ti-6Al-4V": LinerMaterial("Ti-6Al-4V", 113800.0, 0.34, 880.0, 4.43e-9),
    "HDPE": LinerMaterial("HDPE", 1100.0, 0.42, 26.0, 0.95e-9),  # Type 4 polymer liner
}


def liner_ply_material(liner: LinerMaterial) -> PlyMaterial:
    """Isotropic liner expressed as an (isotropic) orthotropic ply for CLT."""
    g = liner.E / (2.0 * (1.0 + liner.nu))
    return PlyMaterial(e1=liner.E, e2=liner.E, g12=g, nu12=liner.nu)


def von_mises(s11: float, s22: float, t12: float) -> float:
    """Plane-stress von Mises equivalent stress."""
    return float(np.sqrt(s11 * s11 - s11 * s22 + s22 * s22 + 3.0 * t12 * t12))


def analyse_bare_liner(liner: LinerMaterial, thickness: float, N: np.ndarray) -> dict:
    """Solve the liner ALONE under membrane load — the metal tank on its own."""
    ply = Ply(0.0, thickness, liner_ply_material(liner))
    eps0, kappa = solve_midplane([ply], N)
    s = ply_stresses([ply], eps0, kappa)[0]
    vm = von_mises(s["sigma_11"], s["sigma_22"], s["tau_12"])
    return {
        "sigma_axial": s["sigma_11"], "sigma_hoop": s["sigma_22"], "von_mises": vm,
        "yield": liner.yield_strength, "yield_margin": liner.yield_strength / max(vm, 1e-9),
        "yields": vm > liner.yield_strength,
    }


def analyse_type3(liner: LinerMaterial, liner_thickness: float, overwrap_plies: list[Ply],
                  N: np.ndarray, allowables: MaterialAllowables | None = None) -> dict:
    """Liner + composite overwrap sharing membrane load (Type 3 / Type 4 section).

    Returns the liner von Mises + yield margin and the composite per-ply Hashin, so you
    can see the overwrap offload the liner."""
    allow = MaterialAllowables() if allowables is None else allowables
    liner_ply = Ply(0.0, liner_thickness, liner_ply_material(liner))
    plies = [liner_ply] + list(overwrap_plies)
    eps0, kappa = solve_midplane(plies, N)
    stresses = ply_stresses(plies, eps0, kappa)

    ls = stresses[0]
    vm = von_mises(ls["sigma_11"], ls["sigma_22"], ls["tau_12"])
    comp = [hashin_ply(p["sigma_11"], p["sigma_22"], p["tau_12"], allow) for p in stresses[1:]]
    comp_fi = max((c["failure_index"] for c in comp), default=0.0)
    return {
        "liner_von_mises": vm, "liner_yield": liner.yield_strength,
        "liner_yield_margin": liner.yield_strength / max(vm, 1e-9), "liner_yields": vm > liner.yield_strength,
        "composite_fi_max": comp_fi, "composite_min_reserve": (1.0 / np.sqrt(comp_fi)) if comp_fi > 1e-12 else float("inf"),
        "eps0": eps0, "n_plies": len(plies),
    }


def liner_yield_pressure(liner: LinerMaterial, radius: float, thickness: float) -> float:
    """Thin-wall closed-cylinder pressure at which the bare liner reaches yield [MPa].

    von Mises = sqrt(3)/2 * p r / t = yield  ->  p = 2 yield t / (sqrt(3) r)."""
    return 2.0 * liner.yield_strength * thickness / (np.sqrt(3.0) * radius)
