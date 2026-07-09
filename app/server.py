"""COPV Studio Pro — web backend for the Three.js workflow front-end.

Serves app/webapp/index.html and a small JSON API backed by the real engine, so the
browser renders reliably with Three.js while Python does the actual FEM / winding
optimization / element-level CLT.

    python -m app.server          # http://localhost:8081

Dependency-free (stdlib http.server). Engine calls are serialized with a lock
(gmsh/JAX are not concurrency-safe) and the gmsh SIGINT handler is no-op'd off the
main thread.
"""

from __future__ import annotations

import json
import math
import os
import signal as _sig
import sys
import threading as _thr
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_rs = _sig.signal
_sig.signal = lambda s, h: (_rs(s, h) if _thr.current_thread() is _thr.main_thread() else None)

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.engine import build_state, fast_screen, full_optimize
from app.sizing import TankRequirement, geometry_from_requirement
from copv_opt.config import FailureConfig, GeometryConfig, MaterialAllowables, MaterialConfig

WEBAPP = Path(__file__).resolve().parent / "webapp"
MATERIAL = MaterialConfig()
DENSITY = {"Coarse": 36.0, "Medium": 28.0, "Fine": 18.0}
CACHE: dict = {}
LOCK = _thr.Lock()


def _geom(p: dict):
    hmax = DENSITY.get(p.get("density", "Medium"), 28.0)
    common = dict(thickness=float(p["thickness"]), opening_radius=float(p["opening"]),
                  dome_height_ratio=float(p["dome_ratio"]), pressure=float(p.get("pressure", 6.85)))
    if p.get("define_by") == "Capacity":
        req = TankRequirement(internal_volume_litres=float(p["capacity"]),
                              design_pressure_bar=float(p.get("pressure", 6.85)) * 10.0,
                              envelope_outer_radius_mm=float(p["radius"]), wall_thickness_mm=float(p["thickness"]),
                              opening_radius_mm=float(p["opening"]), dome_height_ratio=float(p["dome_ratio"]))
        g, rep = geometry_from_requirement(req)
        geom = GeometryConfig(outer_radius=g.outer_radius, cylinder_length=g.cylinder_length, **common)
        derived = (f"cyl length {rep.cylinder_length_mm:.0f} mm · achieved {rep.achieved_volume_litres:.2f} L "
                   f"· L/D {rep.slenderness_l_over_d:.2f}")
    else:
        geom = GeometryConfig(outer_radius=float(p["radius"]), cylinder_length=float(p["length"]), **common)
        derived = f"outer Ø {2*geom.outer_radius:.0f} mm · cyl length {geom.cylinder_length:.0f} mm"
    geom.mesh_hmax = hmax
    geom.mesh_hmin = min(10.0, hmax)
    return geom, derived


def api_geometry(p: dict) -> dict:
    with LOCK:
        geom, derived = _geom(p)
        CACHE["geom"] = geom
        if p.get("dome") == "Isotensoid":
            import tempfile

            from app.engine import isotensoid_vessel_profile
            from app.meridian_mesh import mesh_meridian
            prof = isotensoid_vessel_profile(geom.mid_radius, geom.cylinder_length, geom.opening_radius)
            nodes, elems = mesh_meridian(prof, Path(tempfile.mkdtemp(prefix="iso_")),
                                         hmin=geom.mesh_hmin, hmax=geom.mesh_hmax)
            derived += " · isotensoid dome"
        else:
            bundle = build_state(geom, MATERIAL)
            nodes, elems = bundle["nodes"], bundle["elems"]
    nodes = np.asarray(nodes, np.float64); elems = np.asarray(elems, np.int64)
    return {"nodes": nodes.round(3).tolist(), "elems": elems.tolist(),
            "derived": derived, "nelem": int(len(elems)), "nnode": int(len(nodes)),
            "geom": {"outer_radius": geom.outer_radius, "mid_radius": geom.mid_radius,
                     "cyl_length": geom.cylinder_length, "thickness": geom.thickness,
                     "opening": geom.opening_radius, "dome_ratio": geom.dome_height_ratio}}


def _course_orientation(points: np.ndarray) -> np.ndarray:
    """Local winding angle [deg from axis] at every point of a course path.

    Orientation = angle between the path tangent and the meridian, computed from the
    hoop component of the tangent — the per-point tow orientation deliverable."""
    pts = np.asarray(points, dtype=np.float64)
    d = np.gradient(pts, axis=0)
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    hoop = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
    t_hoop = np.abs(np.einsum("ij,ij->i", d, hoop))
    t_norm = np.linalg.norm(d, axis=1) + 1e-12
    return np.degrees(np.arcsin(np.clip(t_hoop / t_norm, 0.0, 1.0)))


def _winding(r, geom) -> dict:
    """Actual discrete winding course plan (real helical courses + hoop rings) from the
    optimizer layout. Only available after Optimize winding."""
    if r.mode != "full_optimize" or r.winding_result is None:
        return {"available": False, "reason": "Run Optimize winding to generate the course plan."}
    try:
        from copv_opt.course_planner import DiscreteCoursePlanningConfig, build_discrete_winding_plan_from_layout
        from copv_opt.geometry import copv_surface_from_sphi_np
        from copv_opt.visualize import build_winding_process_layout_data
        layout = build_winding_process_layout_data(r.winding_result, geom, family_count=16, sample_count=400)
        plan = build_discrete_winding_plan_from_layout(layout, geom, DiscreteCoursePlanningConfig(emit_path_points=True))
        helical = []
        for c in plan["helical_courses"]:
            pts = c.get("path_points_mm")
            if pts is None:
                continue
            arr = np.asarray(pts, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] < 2:
                continue
            helical.append({"hand": c["handedness"], "points": arr.round(2).tolist()})
            if len(helical) >= 600:
                break
        hoops = []
        for ring in plan["hoop_rings"]:
            mid = 0.5 * (ring["start_s_mm"] + ring["stop_s_mm"])
            surf = copv_surface_from_sphi_np(geom.mid_radius, np.array([mid]), np.array([0.0]),
                                             geom.cylinder_length, geom.opening_radius,
                                             dome_height_ratio=geom.dome_height_ratio)
            pt = np.asarray(surf["points"])[0]
            hoops.append({"z": float(pt[2]), "radius": float(np.hypot(pt[0], pt[1]))})
        machine = None
        if helical:
            try:
                from copv_opt.machine import machine_program_from_path, program_summary
                machine = program_summary(machine_program_from_path(np.asarray(helical[0]["points"], dtype=np.float64)))
            except Exception:
                machine = None
        m = plan["metrics"]
        wa = np.asarray(r.fields.get("Winding angle [deg]", []), dtype=np.float64)
        # optimized tow schedule: decimated positions + per-point orientations per course
        sched_courses = []
        for i, c in enumerate(helical[:150]):
            pts = np.asarray(c["points"], dtype=np.float64)
            step = max(1, len(pts) // 40)
            ang = _course_orientation(pts)[::step]
            sched_courses.append({"id": i, "hand": c["hand"],
                                  "points": pts[::step].round(2).tolist(),
                                  "angle_deg": np.round(ang, 2).tolist()})
        schedule = {
            "layers": [
                {"type": "helical", "angle_deg_mean": float(np.mean(wa)) if wa.size else None,
                 "course_pairs": int(m["total_course_pairs"])},
                {"type": "hoop", "rings": int(m["total_hoop_rings"])},
            ],
            "geometry": {"outer_radius_mm": geom.outer_radius, "cylinder_length_mm": geom.cylinder_length,
                         "opening_radius_mm": geom.opening_radius, "pressure_mpa": geom.pressure},
            "courses": sched_courses,
        }
        return {"available": True, "helical": helical, "hoops": hoops, "machine": machine,
                "schedule": schedule,
                "n_pairs": int(m["total_course_pairs"]), "n_courses": int(m["total_individual_courses"]),
                "n_hoops": int(m["total_hoop_rings"]), "cut_restart": int(m["total_cut_restart_events"]),
                "angle_mean": float(np.mean(wa)) if wa.size else None,
                "angle_min": float(np.min(wa)) if wa.size else None,
                "angle_max": float(np.max(wa)) if wa.size else None,
                "warnings": list(plan.get("warnings", []))[:3]}
    except Exception as exc:
        traceback.print_exc()
        return {"available": False, "reason": f"course plan failed: {exc}"}


def api_solve(p: dict) -> dict:
    with LOCK:
        geom, _ = _geom(p)
        allow = MaterialAllowables(xt=float(p["xt"]), xc=float(p["xc"]), yt=float(p["yt"]),
                                   yc=float(p["yc"]), s=float(p["s"]))
        failure = FailureConfig(allowables=allow, margin_of_safety=1.0)
        if p.get("dome") == "Isotensoid":
            from app.engine import screen_isotensoid
            r = screen_isotensoid(geom, MATERIAL, float(p["angle"]), float(p["band"]), failure_cfg=failure)
        elif p.get("mode") == "Optimize winding":
            r = full_optimize(geom, MATERIAL, failure_cfg=failure)
        else:
            r = fast_screen(geom, MATERIAL, float(p["angle"]), float(p["band"]), failure_cfg=failure)
        winding = _winding(r, geom)
        design = _winding_design(geom, float(p.get("band", 6.0)))
    return {"nodes": np.asarray(r.nodes, np.float64).round(3).tolist(),
            "elems": np.asarray(r.elems, np.int64).tolist(),
            "fields": {k: np.asarray(v, np.float64).round(5).tolist() for k, v in r.fields.items()},
            "margins": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.margins.items()},
            "burst": float(r.burst_factor), "fi_max": float(r.fi_max),
            "mu": None if r.mu_max_required is None else float(r.mu_max_required),
            "decision": r.gate["decision"].upper(), "mode": r.mode,
            "winding": winding, "winding_design": design}


def _winding_design(geom, band_width: float) -> dict | None:
    """Classical winding-design summary (geodesic angle, coverage, pattern, dome buildup)."""
    try:
        from copv_opt.winding import pattern_for_coverage, winding_summary
        d = winding_summary(2.0 * geom.outer_radius, geom.cylinder_length, geom.opening_radius,
                            band_width=max(band_width, 1.0), target_thickness=geom.thickness)
        cov = pattern_for_coverage(2.0 * geom.outer_radius, d["helical_angle_deg"], max(band_width, 1.0), 1.0)
        d.update(coverage_100_circuits=cov.circuits, coverage_pattern_number=cov.pattern_number,
                 coverage_closes=cov.closes)
        return d
    except Exception:
        traceback.print_exc()
        return None


def api_layer_metrics(p: dict) -> dict:
    """Per-layer design metrics for a user-defined layer stack (the Layer Design panel).

    For each layer: coverage bands, p/n pattern closure, circuits for 100% coverage, and
    geodesic feasibility (a helical angle below asin(r_open/R) would turn around inside
    the boss opening — infeasible without pins)."""
    from copv_opt.winding import geodesic_angle_deg, helical_coverage, pattern_for_coverage
    geom, _ = _geom(p)
    D = 2.0 * geom.mid_radius
    a_min = geodesic_angle_deg(geom.mid_radius, geom.opening_radius)
    out = []
    for L in p.get("layers", []):
        band = max(float(L.get("band", 6.0)), 0.5)
        if L.get("type") == "hoop":
            n = max(1, math.ceil(geom.cylinder_length / band))
            out.append({"type": "hoop", "angle_deg": 90.0, "bands": n, "feasible": True,
                        "note": f"{n} circuits tile the cylinder"})
        else:
            a = float(L.get("angle", a_min))
            cov = helical_coverage(D, a, band)
            pat = pattern_for_coverage(D, a, band, 1.0)
            feasible = a >= a_min - 1e-6
            out.append({"type": "helical", "angle_deg": a, "bands": cov.n_bands,
                        "pattern": pat.pattern_number, "closes": bool(pat.closes),
                        "circuits": pat.circuits, "feasible": bool(feasible),
                        "note": "ok" if feasible else f"below geodesic minimum {a_min:.1f} deg — turnaround inside boss"})
    return {"min_geodesic_angle_deg": a_min, "layers": out}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):  # CORS preflight (page opened from file:// or an IDE preview)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            data = (WEBAPP / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/geometry":
                self._json(api_geometry(body))
            elif self.path == "/api/solve":
                self._json(api_solve(body))
            elif self.path == "/api/layer_metrics":
                self._json(api_layer_metrics(body))
            else:
                self._json({"error": "unknown endpoint"}, 404)
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": str(exc)}, 500)


def main():
    port = int(os.environ.get("COPV_PORT", "8081"))
    print(f"COPV Studio Pro web app -> http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
