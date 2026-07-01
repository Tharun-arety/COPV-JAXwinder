"""Verify netting analysis against the classical closed-form results.

    python -m app.verify_netting
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from copv_opt.netting import (
    OPTIMAL_HELICAL_ANGLE_DEG, clairaut_angle_deg, netting_burst_pressure, netting_cylinder, netting_design,
)

TOL = 1e-6


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    return bool(cond)


def main() -> int:
    ok = True
    p, r, sigma = 6.85, 100.0, 2200.0
    print("=== Netting analysis verification ===\n")

    # 1. Optimal helical-only angle = atan(sqrt(2)) = 54.7356 deg
    print("1. Helical-only optimum")
    ok &= check("optimal angle = atan(sqrt2)", abs(OPTIMAL_HELICAL_ANGLE_DEG - 54.735610) < 1e-4,
                f"{OPTIMAL_HELICAL_ANGLE_DEG:.4f} deg")
    opt = netting_cylinder(p, r, sigma)  # defaults to optimal angle
    ok &= check("hoop thickness ~ 0 at the optimum", opt.t_hoop < 1e-9 and not opt.hoop_required,
                f"t_hoop = {opt.t_hoop:.3e}")

    # 2. Total fibre thickness = 1.5 p r / sigma, invariant with angle (<= optimum)
    print("\n2. Total fibre thickness invariant = 1.5 p r / sigma")
    expect = 1.5 * p * r / sigma
    ok &= check("t_total at optimum = 1.5 pr/sigma", abs(opt.t_total - expect) / expect < TOL,
                f"{opt.t_total:.4f} vs {expect:.4f} mm")
    for a in (15.0, 30.0, 45.0, 54.0):
        res = netting_cylinder(p, r, sigma, angle_deg=a)
        ok &= check(f"t_total invariant at {a:.0f} deg", abs(res.t_total - expect) / expect < TOL,
                    f"{res.t_total:.4f} (helical {res.t_helical:.4f} + hoop {res.t_hoop:.4f})")

    # 3. Helical angle set by the polar opening via Clairaut
    print("\n3. Clairaut angle from the polar opening")
    ok &= check("opening = r sin(54.74) -> 54.74 deg",
                abs(clairaut_angle_deg(r * math.sin(math.radians(54.7356)), r) - 54.7356) < 1e-3)
    ok &= check("opening = 50 on r=100 -> 30 deg", abs(clairaut_angle_deg(50.0, 100.0) - 30.0) < 1e-6,
                f"{clairaut_angle_deg(50.0,100.0):.3f} deg")

    # 4. Hoop fraction grows as the angle drops (more hoop, less helical)
    print("\n4. Layer split vs angle")
    low, high = netting_cylinder(p, r, sigma, angle_deg=20.0), netting_cylinder(p, r, sigma, angle_deg=50.0)
    ok &= check("shallower angle needs more hoop", low.hoop_fraction > high.hoop_fraction,
                f"20 deg hoop {low.hoop_fraction:.2f} > 50 deg hoop {high.hoop_fraction:.2f}")

    # 5. Burst: a laminate sized to sigma carries exactly p; ultimate scales it
    print("\n5. Netting burst pressure")
    res = netting_cylinder(p, r, sigma, angle_deg=30.0)
    p_at_allow = netting_burst_pressure(r, sigma, 30.0, res.t_helical, res.t_hoop)
    ok &= check("sized-to-allowable design carries exactly p", abs(p_at_allow - p) / p < 1e-6,
                f"{p_at_allow:.4f} vs {p:.4f} MPa")
    p_burst = netting_burst_pressure(r, 2500.0, 30.0, res.t_helical, res.t_hoop)
    ok &= check("burst scales with ultimate/allowable", abs(p_burst - p * 2500.0 / sigma) / (p * 2500.0 / sigma) < 1e-6,
                f"burst {p_burst:.3f} MPa (factor {p_burst/p:.3f})")

    # 6. Full design point (angle from boss)
    print("\n6. First-principles design point (r=100, opening=20, p=6.85)")
    d = netting_design(6.85, 100.0, opening_radius=20.0)
    print(f"     angle {d['helical_angle_deg']:.1f} deg · helical {d['t_helical_mm']:.3f} + hoop {d['t_hoop_mm']:.3f} mm"
          f" · {d['helical_pairs']} pairs + {d['hoop_rings']} hoops · burst x{d['burst_factor']:.2f}")
    ok &= check("boss opening 20 mm -> low helical angle, hoop required",
                d["helical_angle_deg"] < 15.0 and d["hoop_required"])

    print(f"\nVERDICT: {'ALL CHECKS PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
