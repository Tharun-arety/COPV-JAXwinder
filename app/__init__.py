"""Phase 0 tank configurator — a thin product layer over the copv_opt engine.

The package turns a customer *requirement* (volume, pressure, envelope) into an
optimized, structurally screened wound-tank design, and renders the failure-index
field in an interactive 3D viewport. It sits upstream of a machine-programming
tool (e.g. TaniqWind Pro): it finds the design point, it does not emit NC code.

Modules
-------
sizing  : requirement spec -> GeometryConfig (pure, no JAX).
engine  : mesh build + fast screen / full optimization over the copv_opt solver.
cli     : headless runner (no browser) for scripting and verification.
main    : trame web application (the configurator GUI).
"""

from .sizing import TankRequirement, geometry_from_requirement

__all__ = ["TankRequirement", "geometry_from_requirement"]
