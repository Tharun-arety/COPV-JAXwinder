"""Validate the general axisymmetric path against the parametric COPV.

The general meridian geometry is only trustworthy if, given a profile that *is* the
parametric COPV, it reproduces the COPV engine's geometry and failure index. This
harness builds both states on the identical mesh and compares them.

    python -m app.validate_general
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.general_state import build_general_fem_state
from app.meridian import MeridianProfile
from copv_opt.config import FailureConfig, GeometryConfig, MaterialConfig
from copv_opt.geometry import ensure_copv_mesh
from copv_opt.physics import (
    baseline_response,
    build_copv_fem_state,
    evaluate_hashin_failure,
    make_solve_compliance,
    rotate_stiffness_field,
)


def _baseline_fi(state, material, geom) -> tuple[float, float]:
    solve = make_solve_compliance(state)
    base = baseline_response(state, material, solve)
    c_base = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    failure = evaluate_hashin_failure(
        state, base["displacement"], c_base, base["fiber_dirs"], FailureConfig(margin_of_safety=1.0)
    )
    fi_max = float(np.asarray(failure["fi_max"]))
    disp_max = float(np.max(np.linalg.norm(np.asarray(base["displacement"]).reshape(-1, 3), axis=1)))
    return fi_max, disp_max


def main() -> int:
    geom = GeometryConfig(pressure=6.85)
    material = MaterialConfig()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        mesh = ensure_copv_mesh(work / "s.step", work / "s.msh", geom, remesh=True, rebuild_step=True)

        ref_state = build_copv_fem_state(mesh.nodes, mesh.elems, material, geom)
        profile = MeridianProfile.from_parametric_copv(geom)
        gen_state = build_general_fem_state(
            mesh.nodes, mesh.elems, material, profile, geom.pressure, support_tol=geom.support_tol
        )

        # --- geometry agreement (per element) ---
        def _dirs(state, key):
            return np.asarray(state[key])

        def _cos(a, b):
            return np.abs(np.einsum("ij,ij->i", a, b))

        cos_normal = _cos(_dirs(ref_state, "surface_normals"), _dirs(gen_state, "surface_normals"))
        cos_merid = _cos(_dirs(ref_state, "meridian_dirs"), _dirs(gen_state, "meridian_dirs"))
        cos_hoop = _cos(_dirs(ref_state, "hoop_dirs"), _dirs(gen_state, "hoop_dirs"))
        n_support_ref = int(np.sum(ref_state["support_mask"]))
        n_support_gen = int(np.sum(gen_state["support_mask"]))

        print("=== General-vs-parametric geometry agreement ===")
        print(f"elements                : {ref_state['element_count']}")
        print(f"normal  |cos|  min/mean : {cos_normal.min():.4f} / {cos_normal.mean():.4f}")
        print(f"merid   |cos|  min/mean : {cos_merid.min():.4f} / {cos_merid.mean():.4f}")
        print(f"hoop    |cos|  min/mean : {cos_hoop.min():.4f} / {cos_hoop.mean():.4f}")
        print(f"support nodes ref/gen   : {n_support_ref} / {n_support_gen}")

        # --- failure index agreement ---
        ref_fi, ref_disp = _baseline_fi(ref_state, material, geom)
        gen_fi, gen_disp = _baseline_fi(gen_state, material, geom)
        fi_rel = abs(gen_fi - ref_fi) / max(ref_fi, 1e-9)
        disp_rel = abs(gen_disp - ref_disp) / max(ref_disp, 1e-9)
        print("\n=== Baseline failure index ===")
        print(f"FI_max   ref / gen      : {ref_fi:.3f} / {gen_fi:.3f}   (rel diff {fi_rel*100:.1f}%)")
        print(f"disp_max ref / gen      : {ref_disp:.4f} / {gen_disp:.4f} mm (rel diff {disp_rel*100:.1f}%)")

        # --- verdict ---
        ok_geom = cos_normal.mean() > 0.99 and cos_merid.mean() > 0.99 and cos_hoop.mean() > 0.999
        ok_fi = fi_rel < 0.10
        verdict = ok_geom and ok_fi
        print(f"\nVERDICT: {'PASS' if verdict else 'FAIL'} "
              f"(geometry {'ok' if ok_geom else 'off'}, FI within 10% {'ok' if ok_fi else 'off'})")
        return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
