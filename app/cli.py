"""Headless runner for the configurator — drives all phases without a browser.

Runs the spec -> geometry -> screen/optimize pipeline and, optionally, the
downstream phases: save to a project catalog, export STEP/Abaqus/NC, build the
discrete course plan + kinematic demand, calibrate against coupon allowables, run
CalculiX, and render an HTML report.

Examples
--------
    # fast screen
    python -m app.cli --volume 9 --pressure 300 --radius 100 --angle 42 --band 8

    # full optimize + course plan + Abaqus + report, saved to a catalog
    python -m app.cli --volume 9 --pressure 300 --radius 100 --optimize \
        --course --export-abaqus out/tank.inp --report out/tank.html \
        --save-project "9L 300bar" --store out/catalog

    # calibrate a fast screen against coupon allowables
    python -m app.cli --volume 9 --pressure 300 --radius 100 --allowables coupons.json

    # list a catalog
    python -m app.cli --list --store out/catalog
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
    p = argparse.ArgumentParser(description="Headless COPV tank configurator (all phases).")
    # catalog listing mode
    p.add_argument("--list", action="store_true", help="List projects in --store and exit")
    p.add_argument("--store", type=Path, default=None, help="Project catalog directory")

    # requirement
    p.add_argument("--volume", type=float, help="Internal volume [litres]")
    p.add_argument("--pressure", type=float, help="Design pressure [bar]")
    p.add_argument("--radius", type=float, help="Envelope outer radius [mm]")
    p.add_argument("--thickness", type=float, default=8.0, help="Wall/base thickness [mm]")
    p.add_argument("--opening", type=float, default=10.0, help="Boss opening radius [mm]")
    p.add_argument("--dome-ratio", type=float, default=0.7, help="Dome height ratio")
    p.add_argument("--length", type=float, default=None, help="Override cylinder length [mm]")

    # analysis
    p.add_argument("--angle", type=float, default=42.0, help="Fast-screen winding angle [deg]")
    p.add_argument("--band", type=float, default=8.0, help="Fast-screen band thickness [mm]")
    p.add_argument("--optimize", action="store_true", help="Full L-BFGS optimizer instead of a fast screen")
    p.add_argument("--lbfgs-maxiter", type=int, default=None, help="Override optimizer iterations (faster checks)")

    # phase outputs
    p.add_argument("--save-project", type=str, default=None, help="Save result to the catalog under this name")
    p.add_argument("--report", type=Path, default=None, help="Write an HTML design report to this path")
    p.add_argument("--export-step", type=Path, default=None, help="Write the vessel as a STEP solid")
    p.add_argument("--liner", type=float, default=None, help="Liner thickness [mm] -> report liner mass")
    p.add_argument("--course", action="store_true", help="Build discrete course plan + kinematic demand (needs --optimize)")
    p.add_argument("--export-nc", type=Path, default=None, help="Write a machine-neutral NC CSV (needs --course)")
    p.add_argument("--export-abaqus", type=Path, default=None, help="Write an Abaqus .inp deck (needs --optimize)")
    p.add_argument("--run-ccx", action="store_true", help="Run CalculiX on the exported deck if ccx is on PATH")
    p.add_argument("--allowables", type=Path, default=None, help="Coupon allowables (JSON/CSV) -> calibrated re-screen")

    p.add_argument("--json", action="store_true", help="Emit a JSON report instead of text")
    return p.parse_args()


def _require_requirement(args) -> None:
    missing = [n for n in ("volume", "pressure", "radius") if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"--{', --'.join(missing)} required (or use --list)")


def _run_listing(store_dir: Path) -> None:
    from app.project import ProjectStore

    rows = ProjectStore(store_dir).list()
    if not rows:
        print(f"(no projects in {store_dir})")
        return
    print(f"\nCatalog: {store_dir}  ({len(rows)} projects)")
    for r in rows:
        print(
            f"  {r['name']:<24} {str(r['volume_l']):>6}L {str(r['pressure_bar']):>6}bar  "
            f"FI={r['fi_max']}  burst={r['burst_factor']}  {r['decision']}"
        )
    print()


def main() -> None:
    args = _parse_args()

    if args.list:
        if args.store is None:
            raise SystemExit("--list needs --store DIR")
        _run_listing(args.store)
        return

    _require_requirement(args)
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

    report: dict[str, object] = {
        "sizing": {
            "cylinder_length_mm": round(sizing.cylinder_length_mm, 2),
            "inner_radius_mm": round(sizing.inner_radius_mm, 2),
            "design_pressure_mpa": round(sizing.design_pressure_mpa, 4),
            "achieved_volume_litres": round(sizing.achieved_volume_litres, 3),
            "slenderness_l_over_d": round(sizing.slenderness_l_over_d, 3),
        }
    }

    # ---- analysis ----
    t0 = time.time()
    if args.optimize:
        from copv_opt.config import WindingOptimizationConfig

        cfg = None
        if args.lbfgs_maxiter is not None:
            cfg = WindingOptimizationConfig(
                min_angle_deg=12.0, max_angle_deg=58.0, max_winding_thickness=18.0,
                winding_seed_angle_deg=42.0, winding_seed_thickness=7.0,
                max_helical_pass_count=44.0, max_hoop_pass_count=24.0,
                helical_seed_pass_count=14.0, hoop_seed_pass_count=2.0,
                lbfgs_maxiter=int(args.lbfgs_maxiter), lbfgs_tol=1e-6, history_size=12,
            )
        result = full_optimize(geom, material, winding_cfg=cfg)
    else:
        result = fast_screen(geom, material, args.angle, args.band)
    report["mode"] = result.mode
    report["result"] = {
        "fi_max": round(result.fi_max, 4),
        "burst_factor": round(result.burst_factor, 4),
        "mass_metric": round(result.mass_metric, 4),
        "mu_max_required": None if result.mu_max_required is None else round(result.mu_max_required, 4),
        "disp_max_mm": round(result.disp_max, 4),
        "angle_deg": result.angle_deg,
    }
    report["gate"] = result.gate

    # ---- Phase 2: STEP export + liner ----
    if args.export_step is not None:
        from app.geometry_io import export_step

        report["step"] = str(export_step(geom, args.export_step))
    if args.liner is not None:
        from app.geometry_io import liner_mass

        lr = liner_mass(geom, args.liner)
        report["liner"] = {"thickness_mm": lr.liner_thickness_mm, "mass_g": round(lr.liner_mass_g, 2)}

    # ---- Phase 3: course plan + kinematics + NC ----
    course_summary = None
    kinematics = None
    if args.course:
        if result.layout is None:
            print("warning: --course needs --optimize (no winding layout from a fast screen); skipping")
        else:
            from app.course import course_plan, export_nc_csv, kinematic_demand

            plan = course_plan(result.layout, geom)
            course_summary = plan.get("metrics", {})
            kinematics = kinematic_demand(plan, geom)
            report["course"] = {
                "total_course_pairs": course_summary.get("total_course_pairs"),
                "total_hoop_rings": course_summary.get("total_hoop_rings"),
                "peak_mandrel_rpm": round(kinematics["peak_mandrel_rpm"], 3),
                "screen_status": kinematics["screen_status"],
            }
            if args.export_nc is not None:
                report["nc_csv"] = str(export_nc_csv(plan, args.export_nc))

    # ---- Phase 4: Abaqus + CalculiX ----
    if args.export_abaqus is not None:
        if result.winding_result is None:
            print("warning: --export-abaqus needs --optimize; skipping")
        else:
            from app.solver_export import export_abaqus, run_calculix

            inp = export_abaqus(result.state, result.winding_result, geom, args.export_abaqus, material)
            report["abaqus_inp"] = str(inp)
            if args.run_ccx:
                run = run_calculix(inp)
                report["calculix"] = {"ran": run.ran, "returncode": run.returncode, "message": run.message}

    # ---- Phase 5: calibration ----
    if args.allowables is not None:
        from app.calibration import calibrate_screen, calibration_delta, load_allowables

        allow = load_allowables(args.allowables)
        calibrated = calibrate_screen(geom, material, args.angle, args.band, allow)
        report["calibration"] = calibration_delta(result, calibrated, allow)

    # ---- Phase 6: report + project save ----
    if args.report is not None:
        from app.report import write_html

        report["report_html"] = str(write_html(args.report, args.save_project or "tank", result, sizing, course_summary, kinematics))
    if args.save_project is not None:
        from app.project import Project, ProjectStore

        project = Project(name=args.save_project, requirement=req)
        project.record_result(result, sizing)
        store = ProjectStore(args.store or Path("catalog"))
        report["project_saved"] = str(store.save(project))

    report["elapsed_seconds"] = round(time.time() - t0, 1)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text(report)


def _print_text(report: dict) -> None:
    s, r, g = report["sizing"], report["result"], report["gate"]
    print("\n=== COPV tank configurator ===")
    print(f"mode                 : {report['mode']}")
    print("\n-- geometry from requirement --")
    print(f"cylinder length      : {s['cylinder_length_mm']} mm")
    print(f"design pressure      : {s['design_pressure_mpa']} MPa")
    print(f"achieved volume      : {s['achieved_volume_litres']} L")
    print("\n-- structural screen --")
    print(f"FI_max               : {r['fi_max']}")
    print(f"burst factor         : {r['burst_factor']}  (pressure multiple to FI=1)")
    print(f"max displacement     : {r['disp_max_mm']} mm")
    if r["mu_max_required"] is not None:
        print(f"friction demand mu   : {r['mu_max_required']}")
    print("\n-- release gate --")
    print(f"decision             : {g['decision'].upper()}")
    for b in g["blockers"]:
        print(f"  - {b}")
    for key, label in (
        ("liner", "liner"),
        ("course", "course/kinematics"),
        ("step", "STEP"),
        ("abaqus_inp", "Abaqus deck"),
        ("calculix", "CalculiX"),
        ("nc_csv", "NC CSV"),
        ("calibration", "calibration"),
        ("report_html", "HTML report"),
        ("project_saved", "project saved"),
    ):
        if key in report:
            print(f"\n-- {label} --")
            print(f"  {json.dumps(report[key]) if not isinstance(report[key], str) else report[key]}")
    print(f"\n(elapsed {report['elapsed_seconds']}s)\n")


if __name__ == "__main__":
    main()
