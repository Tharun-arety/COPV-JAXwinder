"""Headless runner for the configurator.

Runs the same spec -> geometry -> screen/optimize pipeline as the GUI, but prints a
text report instead of rendering. Useful for scripting, CI smoke tests, and proving
the engine wiring without a browser.

Examples
--------
    python -m app.cli --volume 9 --pressure 300 --radius 100 --angle 42 --band 8
    python -m app.cli --volume 9 --pressure 300 --radius 100 --optimize
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.engine import DesignResult, fast_screen, full_optimize
from app.sizing import TankRequirement, geometry_from_requirement
from copv_opt.config import MaterialConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless COPV tank configurator.")
    p.add_argument("--volume", type=float, required=True, help="Internal volume [litres]")
    p.add_argument("--pressure", type=float, required=True, help="Design pressure [bar]")
    p.add_argument("--radius", type=float, required=True, help="Envelope outer radius [mm]")
    p.add_argument("--thickness", type=float, default=8.0, help="Wall/base thickness [mm]")
    p.add_argument("--opening", type=float, default=10.0, help="Boss opening radius [mm]")
    p.add_argument("--dome-ratio", type=float, default=0.7, help="Dome height ratio")
    p.add_argument("--length", type=float, default=None, help="Override cylinder length [mm]")
    p.add_argument("--angle", type=float, default=42.0, help="Fast-screen winding angle [deg]")
    p.add_argument("--band", type=float, default=8.0, help="Fast-screen band thickness [mm]")
    p.add_argument("--optimize", action="store_true", help="Run the full L-BFGS optimizer instead of a fast screen")
    p.add_argument("--json", action="store_true", help="Emit a JSON report instead of text")
    return p.parse_args()


def _result_dict(result: DesignResult, sizing) -> dict:
    return {
        "mode": result.mode,
        "sizing": {
            "cylinder_length_mm": round(sizing.cylinder_length_mm, 2),
            "inner_radius_mm": round(sizing.inner_radius_mm, 2),
            "design_pressure_mpa": round(sizing.design_pressure_mpa, 4),
            "achieved_volume_litres": round(sizing.achieved_volume_litres, 3),
            "slenderness_l_over_d": round(sizing.slenderness_l_over_d, 3),
        },
        "result": {
            "fi_max": round(result.fi_max, 4),
            "burst_factor": round(result.burst_factor, 4),
            "mass_metric": round(result.mass_metric, 4),
            "mu_max_required": None if result.mu_max_required is None else round(result.mu_max_required, 4),
            "mu_allowable": result.mu_allowable,
            "angle_deg": result.angle_deg,
            "disp_max_mm": round(result.disp_max, 4),
        },
        "gate": result.gate,
    }


def _print_text(report: dict) -> None:
    s, r, g = report["sizing"], report["result"], report["gate"]
    print("\n=== COPV tank configurator ===")
    print(f"mode                 : {report['mode']}")
    print("\n-- geometry from requirement --")
    print(f"cylinder length      : {s['cylinder_length_mm']} mm")
    print(f"inner radius         : {s['inner_radius_mm']} mm")
    print(f"design pressure      : {s['design_pressure_mpa']} MPa")
    print(f"achieved volume      : {s['achieved_volume_litres']} L")
    print(f"slenderness (L/D)    : {s['slenderness_l_over_d']}")
    print("\n-- structural screen --")
    print(f"FI_max               : {r['fi_max']}")
    print(f"burst factor         : {r['burst_factor']}  (pressure multiple to FI=1)")
    print(f"max displacement     : {r['disp_max_mm']} mm")
    print(f"mass metric          : {r['mass_metric']}")
    if r["mu_max_required"] is not None:
        print(f"friction demand mu   : {r['mu_max_required']} (allowable {r['mu_allowable']})")
    if r["angle_deg"] is not None:
        print(f"constant angle       : {r['angle_deg']} deg")
    print("\n-- release gate --")
    print(f"hashin ok            : {g['hashin_ok']}")
    print(f"friction ok          : {g['friction_ok']}")
    print(f"decision             : {g['decision'].upper()}")
    print("blockers:")
    for b in g["blockers"]:
        print(f"  - {b}")
    print()


def main() -> None:
    args = _parse_args()
    req = TankRequirement(
        internal_volume_litres=args.volume,
        design_pressure_bar=args.pressure,
        envelope_outer_radius_mm=args.radius,
        wall_thickness_mm=args.thickness,
        opening_radius_mm=args.opening,
        dome_height_ratio=args.dome_ratio,
        cylinder_length_override_mm=args.length,
    )
    geom, sizing = geometry_from_requirement(req)
    material = MaterialConfig()

    t0 = time.time()
    if args.optimize:
        result = full_optimize(geom, material)
    else:
        result = fast_screen(geom, material, args.angle, args.band)
    elapsed = time.time() - t0

    report = _result_dict(result, sizing)
    report["elapsed_seconds"] = round(elapsed, 1)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text(report)
        print(f"(elapsed {elapsed:.1f}s)\n")


if __name__ == "__main__":
    main()
