"""End-to-end COPV design -> simulation -> optimization pipeline.

One command runs the whole toolchain the goal describes and prints a staged
engineering log:

  1. CAD + mesh          — OpenCASCADE geometry, gmsh shell mesh
  2. FEM assembly        — winding drives element-level material (rotated ply stiffness)
  3. Baseline solve      — unreinforced compliance
  4. Optimization        — L-BFGS winding design for best compliance
  5. Classical Laminate  — per-ply stresses/failure at the cylinder section (+ Type-3 liner)
  6. Showcase            — self-contained interactive 3D results viewer

Everything is a real solve, reusing the verified engine (geometry/physics/optimize),
CLT (clt.py) and liner (liner.py) modules.

    python -m app.pipeline --pressure 6.85
    python -m app.pipeline --pressure 30 --type3 --liner AL6061-T6 --liner-thickness 3
"""

from __future__ import annotations

import argparse
import os
import signal as _sig
import sys
import threading as _thr
import time
import webbrowser
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_rs = _sig.signal
_sig.signal = lambda s, h: (_rs(s, h) if _thr.current_thread() is _thr.main_thread() else None)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.engine import build_state, full_optimize
from app.export_results import write_results_viewer
from copv_opt.clt import analyse_laminate, copv_cylinder_layup, cylinder_load_resultants
from copv_opt.config import GeometryConfig, MaterialConfig
from copv_opt.liner import PRESETS, analyse_bare_liner, analyse_type3
from copv_opt.physics import baseline_response


def stage(n: int, title: str) -> None:
    print(f"\n[{n}/6] {title}")


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end COPV design/simulation/optimization pipeline.")
    ap.add_argument("--radius", type=float, default=100.0)
    ap.add_argument("--length", type=float, default=220.0)
    ap.add_argument("--thickness", type=float, default=8.0)
    ap.add_argument("--pressure", type=float, default=6.85, help="Design pressure [MPa]")
    ap.add_argument("--type3", action="store_true", help="Add a metal liner and report Type-3 load sharing")
    ap.add_argument("--liner", type=str, default="AL6061-T6", choices=list(PRESETS))
    ap.add_argument("--liner-thickness", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=Path("outputs") / "studio_export" / "pipeline_results.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    geom = GeometryConfig(outer_radius=args.radius, cylinder_length=args.length,
                          thickness=args.thickness, pressure=args.pressure)
    material = MaterialConfig()

    print("=" * 64)
    print(f" COPV pipeline · r={args.radius:g} L={args.length:g} t={args.thickness:g} mm · p={args.pressure:g} MPa"
          + (f" · Type 3 {args.liner}" if args.type3 else ""))
    print("=" * 64)

    stage(1, "CAD + mesh — OpenCASCADE geometry, gmsh shell mesh")
    t0 = time.time()
    bundle = build_state(geom, material)
    state = bundle["state"]
    print(f"      revolved COPV shell -> {len(bundle['nodes'])} nodes, {len(bundle['elems'])} triangular shell elements "
          f"({time.time()-t0:.0f}s)")

    stage(2, "FEM assembly — winding drives element-level material")
    print("      each element stiffness = base laminate + winding band, orthotropic ply")
    print("      tensor rotated into the local fibre direction on the shell surface")

    stage(3, "Baseline solve — unreinforced vessel")
    base = baseline_response(state, material, bundle["solve"])
    base_c = float(np.asarray(base["compliance"]))
    print(f"      baseline strain energy (compliance): {base_c:.4g}")

    stage(4, "Optimization — L-BFGS winding design for best compliance")
    t1 = time.time()
    opt = full_optimize(geom, material)
    opt_c = float(np.asarray(opt.winding_result["compliance"]))
    ang = np.asarray(opt.fields["Winding angle [deg]"], dtype=np.float64)
    print(f"      optimized in {time.time()-t1:.0f}s")
    print(f"      compliance {base_c:.4g} -> {opt_c:.4g}   ({100.0*(1.0-opt_c/base_c):.1f}% stiffer)")
    print(f"      FI_max {opt.fi_max:.3f} · min reserve {opt.margins['min_reserve_factor']:.2f} · burst {opt.burst_factor:.2f}x")
    print(f"      winding angle {ang.min():.0f}-{ang.max():.0f} deg along the vessel · friction mu {opt.mu_max_required:.3f}")

    stage(5, "Classical Laminate Theory — per-ply at the cylinder section")
    helical_t = float(np.mean(np.asarray(opt.winding_result["helical_thickness_field"])))
    hoop_t = float(np.mean(np.asarray(opt.winding_result["hoop_thickness_field"])))
    mean_angle = float(np.mean(ang))
    pairs = max(1, int(round(helical_t / (2 * 0.3))))
    n_hoop = max(1, int(round(hoop_t / 0.3)))
    overwrap = copv_cylinder_layup(mean_angle, pairs, n_hoop, base_axial_plies=0)
    N = cylinder_load_resultants(geom.pressure, geom.mid_radius)
    print(f"      cylinder layup from optimizer: +/-{mean_angle:.0f} helical x{pairs} pairs, hoop x{n_hoop} ({len(overwrap)} plies)")
    if args.type3:
        liner = PRESETS[args.liner]
        bare = analyse_bare_liner(liner, args.liner_thickness, N)
        t3 = analyse_type3(liner, args.liner_thickness, overwrap, N)
        print(f"      bare {liner.name} liner {args.liner_thickness:g}mm: von Mises {bare['von_mises']:.0f} vs yield "
              f"{liner.yield_strength:.0f} MPa -> {'YIELDS' if bare['yields'] else 'ok'} (margin {bare['yield_margin']:.2f})")
        print(f"      + overwrap: liner von Mises {t3['liner_von_mises']:.0f} MPa (margin {t3['liner_yield_margin']:.2f}), "
              f"composite per-ply FI {t3['composite_fi_max']:.3f} (reserve {t3['composite_min_reserve']:.2f})")
    else:
        res = analyse_laminate(overwrap, N)
        crit = res["plies"][res["critical_ply"]]
        print(f"      per-ply critical: ply {res['critical_ply']} @ {crit['angle_deg']:.0f} deg, mode {res['critical_mode']}, "
              f"FI {res['laminate_fi_max']:.3f}, min reserve {res['min_reserve_factor']:.2f}")

    stage(6, "Showcase — interactive 3D results viewer")
    out = write_results_viewer(opt, args.out)
    print(f"      wrote {out}")

    print("\n" + "=" * 64)
    print(" Pipeline complete: CAD -> mesh -> FEM -> winding material -> CLT -> optimized design")
    print("=" * 64)
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
