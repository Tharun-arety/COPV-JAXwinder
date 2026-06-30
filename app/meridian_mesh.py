"""Mesh an arbitrary axisymmetric meridian into a shell triangle mesh.

Revolves the meridian polyline (top opening -> bottom opening) about the z-axis with
gmsh/OpenCASCADE and meshes the resulting surface of revolution. Topologically this
is a shell with two polar openings — the same family the engine expects — but the
profile can be any axisymmetric shape, not just the parametric COPV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gmsh

from app.meridian import MeridianProfile
from copv_opt.geometry import read_msh


def build_meridian_step(profile: MeridianProfile, step_path: Path, n_spline: int = 80) -> Path:
    """Revolve the meridian into a STEP shell surface."""
    step_path = Path(step_path)
    step_path.parent.mkdir(parents=True, exist_ok=True)
    if step_path.exists():
        step_path.unlink()

    rho, z = profile.sample(n_spline)
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(step_path.stem)
    occ = gmsh.model.occ
    pts = [occ.addPoint(float(r), 0.0, float(zz)) for r, zz in zip(rho, z)]
    curve = occ.addBSpline(pts)
    occ.revolve([(1, curve)], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0 * np.pi)
    occ.synchronize()
    gmsh.write(str(step_path))
    gmsh.finalize()
    return step_path


def mesh_meridian(
    profile: MeridianProfile,
    work_dir: Path,
    hmin: float = 10.0,
    hmax: float = 28.0,
    n_spline: int = 80,
    boss_hmin: float = 4.0,
    boss_refine_radius: float = 28.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build + mesh the revolved meridian with refinement at both polar openings.

    The openings are stress concentrators; without local refinement the peak failure
    index is under-resolved (non-conservative). We place a refinement ball at each
    opening, matching the parametric COPV mesher's behaviour for a general profile."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    step_path = work_dir / "meridian_shell.step"
    msh_path = work_dir / "meridian_shell.msh"
    build_meridian_step(profile, step_path, n_spline=n_spline)

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(step_path.stem)
    gmsh.merge(str(step_path))
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(float(hmin), float(boss_hmin)))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(hmax))

    if boss_refine_radius > 0.0 and boss_hmin > 0.0:
        field = gmsh.model.mesh.field
        balls: list[int] = []
        for _, z_open in (profile.top_opening, profile.bottom_opening):
            ball = field.add("Ball")
            field.setNumber(ball, "Radius", float(boss_refine_radius))
            field.setNumber(ball, "Thickness", max(0.5 * float(boss_refine_radius), 1e-6))
            field.setNumber(ball, "VIn", float(boss_hmin))
            field.setNumber(ball, "VOut", float(hmax))
            field.setNumber(ball, "XCenter", 0.0)
            field.setNumber(ball, "YCenter", 0.0)
            field.setNumber(ball, "ZCenter", float(z_open))
            balls.append(ball)
        min_id = field.add("Min")
        field.setNumbers(min_id, "FieldsList", balls)
        field.setAsBackgroundMesh(min_id)

    gmsh.model.mesh.generate(2)
    gmsh.write(str(msh_path))
    gmsh.finalize()

    result = read_msh(msh_path, step_path=step_path)
    return np.asarray(result.nodes, dtype=np.float64), np.asarray(result.elems, dtype=np.int32)
