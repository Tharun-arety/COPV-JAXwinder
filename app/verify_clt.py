"""Verify the CLT engine against analytic and invariant checks.

Fidelity has to be proven, not claimed. Each check below is either exact algebra or a
theorem of laminate theory, so a pass means the implementation is correct — no
transcribed textbook numbers to trust.

    python -m app.verify_clt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 can't print ≈/°/σ
except AttributeError:
    pass

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from copv_opt.clt import (
    Ply, PlyMaterial, abd_matrices, analyse_laminate, copv_cylinder_layup,
    cylinder_load_resultants, reduced_stiffness, solve_midplane,
)

MAT = PlyMaterial()
TOL = 1e-6


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def main() -> int:
    ok = True
    q = reduced_stiffness(MAT)
    print("=== CLT verification ===\n")

    # 1. Single 0-deg ply under uniaxial N_x: analytic stress and strain
    print("1. Single 0° ply, uniaxial N_x")
    t = 0.3
    Nx = 100.0  # N/mm
    plies = [Ply(0.0, t, MAT)]
    eps0, kappa = solve_midplane(plies, [Nx, 0.0, 0.0])
    sig11 = q[0, 0] * eps0[0] + q[0, 1] * eps0[1]
    ok &= check("sigma_11 = N_x / t", _rel(sig11, Nx / t) < TOL, f"{sig11:.3f} vs {Nx/t:.3f} MPa")
    ok &= check("eps_x = sigma11/E1 (uniaxial)", _rel(eps0[0], (Nx / t) / MAT.e1) < 1e-4)
    ok &= check("kappa = 0 (membrane only, symmetric)", np.allclose(kappa, 0.0, atol=1e-10))

    # 2. Symmetric laminate -> B = 0
    print("\n2. Symmetric laminate [0/90]s -> B = 0")
    sym = [Ply(0.0, t, MAT), Ply(90.0, t, MAT), Ply(90.0, t, MAT), Ply(0.0, t, MAT)]
    A, B, D = abd_matrices(sym)
    ok &= check("B ≈ 0", np.max(np.abs(B)) < 1e-6, f"max|B| = {np.max(np.abs(B)):.2e}")

    # 3. Cross-ply [0/90] A11 is hand-computable: Q11*t + Q22*t
    print("\n3. Cross-ply [0/90] A11 exact")
    A2, _, _ = abd_matrices([Ply(0.0, t, MAT), Ply(90.0, t, MAT)])
    expect = (q[0, 0] + q[1, 1]) * t
    ok &= check("A11 = (Q11+Q22)·t", _rel(A2[0, 0], expect) < TOL, f"{A2[0,0]:.1f} vs {expect:.1f}")

    # 4. Balanced angle-ply [+45/-45] -> A16 = A26 = 0
    print("\n4. Balanced angle-ply [+45/-45] -> A16 = A26 = 0")
    Ab, _, _ = abd_matrices([Ply(45.0, t, MAT), Ply(-45.0, t, MAT)])
    ok &= check("A16 ≈ 0", abs(Ab[0, 2]) < 1e-6, f"{Ab[0,2]:.2e}")
    ok &= check("A26 ≈ 0", abs(Ab[1, 2]) < 1e-6, f"{Ab[1,2]:.2e}")

    # 5. Quasi-isotropic [0/45/-45/90]s -> A11 = A22, A16 = A26 = 0
    print("\n5. Quasi-isotropic [0/±45/90]s -> in-plane isotropy of A")
    qi = [Ply(a, t, MAT) for a in (0, 45, -45, 90, 90, -45, 45, 0)]
    Aq, _, _ = abd_matrices(qi)
    ok &= check("A11 = A22", _rel(Aq[0, 0], Aq[1, 1]) < TOL, f"{Aq[0,0]:.1f} vs {Aq[1,1]:.1f}")
    ok &= check("A16 = A26 ≈ 0", abs(Aq[0, 2]) < 1e-6 and abs(Aq[1, 2]) < 1e-6)
    # quasi-iso shear relation: A66 = (A11 - A12)/2
    ok &= check("A66 = (A11 - A12)/2 (isotropy)", _rel(Aq[2, 2], 0.5 * (Aq[0, 0] - Aq[0, 1])) < 1e-6)

    # 6. Ply-by-ply resolves what a smeared model cannot:
    #    a COPV cylinder layup under pressure -> hoop plies and helical plies see
    #    different stress; report the per-ply spread.
    print("\n6. COPV cylinder layup under internal pressure — per-ply resolution")
    layup = copv_cylinder_layup(helical_angle_deg=25.0, n_helical_pairs=6, n_hoop=8)
    N = cylinder_load_resultants(pressure=30.0, radius=100.0)
    res = analyse_laminate(layup, N)
    fibre_dir_stress = {round(p["angle_deg"]): p["sigma_11"] for p in res["plies"]}
    print(f"     plies: {len(layup)} | critical ply {res['critical_ply']} ({res['critical_mode']}) "
          f"| FI_max {res['laminate_fi_max']:.3f} | min RF {res['min_reserve_factor']:.2f}")
    print(f"     σ11 by ply angle: " + ", ".join(f"{a}°:{s:.0f}MPa" for a, s in fibre_dir_stress.items()))
    ok &= check("symmetric layup -> curvature ≈ 0 under membrane load",
                np.allclose(res["kappa"], 0.0, atol=1e-9), f"max|κ| = {np.max(np.abs(res['kappa'])):.2e}")
    ok &= check("per-ply σ11 differs across orientations (real resolution, not smeared)",
                len(set(round(v, -1) for v in fibre_dir_stress.values())) > 1)
    ok &= check("hoop (90°) plies carry the most fibre-direction stress",
                fibre_dir_stress.get(90, 0) == max(fibre_dir_stress.values()),
                "fibres aligned with the dominant hoop load")

    print(f"\nVERDICT: {'ALL CHECKS PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
