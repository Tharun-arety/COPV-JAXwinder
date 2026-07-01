"""Verify the liner model against the guide's thin-wall closed-form, and show the
Type-3 liner->overwrap load-sharing story.

    python -m app.verify_liner
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from copv_opt.clt import copv_cylinder_layup, cylinder_load_resultants
from copv_opt.liner import PRESETS, analyse_bare_liner, analyse_type3, liner_yield_pressure

TOL = 1e-4


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def main() -> int:
    ok = True
    al = PRESETS["AL6061-T6"]
    r, t = 100.0, 3.0
    print("=== Liner model verification ===\n")

    # 1. Bare liner reproduces thin-wall closed-form
    print("1. Bare AL6061-T6 liner, thin-wall closed-form")
    p = 5.0
    N = cylinder_load_resultants(p, r)                     # [axial=pr/2, hoop=pr, 0]
    res = analyse_bare_liner(al, t, N)
    ok &= check("hoop = p·r/t", abs(res["sigma_hoop"] - p * r / t) / (p * r / t) < TOL,
                f"{res['sigma_hoop']:.2f} vs {p*r/t:.2f} MPa")
    ok &= check("axial = p·r/2t", abs(res["sigma_axial"] - p * r / (2 * t)) / (p * r / (2 * t)) < TOL,
                f"{res['sigma_axial']:.2f} vs {p*r/(2*t):.2f} MPa")
    vm_expected = np.sqrt(3) / 2 * p * r / t
    ok &= check("von Mises = √3/2·p·r/t", abs(res["von_mises"] - vm_expected) / vm_expected < TOL,
                f"{res['von_mises']:.2f} vs {vm_expected:.2f} MPa")

    # 2. Yield pressure: at p_yield the margin is exactly 1
    print("\n2. Bare liner yield pressure")
    py = liner_yield_pressure(al, r, t)
    resy = analyse_bare_liner(al, t, cylinder_load_resultants(py, r))
    ok &= check("margin = 1.0 at p_yield", abs(resy["yield_margin"] - 1.0) < 1e-4,
                f"p_yield = {py:.2f} MPa, margin {resy['yield_margin']:.4f}")

    # 3. The Type-3 story: bare aluminium tank fails, overwrap saves it
    print("\n3. Type-3: solve aluminium tank first, then add overwrap")
    p_service = 20.0
    bare = analyse_bare_liner(al, t, cylinder_load_resultants(p_service, r))
    print(f"   Bare 3 mm AL liner @ {p_service:.0f} MPa: von Mises {bare['von_mises']:.0f} MPa "
          f"vs yield {al.yield_strength:.0f} → {'YIELDS' if bare['yields'] else 'ok'} (margin {bare['yield_margin']:.2f})")
    ok &= check("bare aluminium tank yields at service pressure", bare["yields"])

    overwrap = copv_cylinder_layup(helical_angle_deg=25.0, n_helical_pairs=6, n_hoop=10, base_axial_plies=0)
    t3 = analyse_type3(al, t, overwrap, cylinder_load_resultants(p_service, r))
    print(f"   + composite overwrap ({t3['n_plies']-1} plies): liner von Mises {t3['liner_von_mises']:.0f} MPa "
          f"(margin {t3['liner_yield_margin']:.2f}), composite FI {t3['composite_fi_max']:.3f} "
          f"(reserve {t3['composite_min_reserve']:.2f})")
    ok &= check("overwrap offloads the liner below yield", not t3["liner_yields"] and t3["liner_yield_margin"] > 1.0,
                f"liner margin {bare['yield_margin']:.2f} → {t3['liner_yield_margin']:.2f}")
    ok &= check("composite carries the load within Hashin", t3["composite_fi_max"] <= 1.0,
                f"composite FI {t3['composite_fi_max']:.3f}")

    print(f"\nVERDICT: {'ALL CHECKS PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
