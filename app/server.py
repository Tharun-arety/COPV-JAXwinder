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
        bundle = build_state(geom, MATERIAL)
        CACHE["geom"] = geom
    return {"nodes": np.asarray(bundle["nodes"], np.float64).round(3).tolist(),
            "elems": np.asarray(bundle["elems"], np.int64).tolist(),
            "derived": derived, "nelem": int(len(bundle["elems"])), "nnode": int(len(bundle["nodes"]))}


def api_solve(p: dict) -> dict:
    with LOCK:
        geom, _ = _geom(p)
        allow = MaterialAllowables(xt=float(p["xt"]), xc=float(p["xc"]), yt=float(p["yt"]),
                                   yc=float(p["yc"]), s=float(p["s"]))
        failure = FailureConfig(allowables=allow, margin_of_safety=1.0)
        if p.get("mode") == "Optimize winding":
            r = full_optimize(geom, MATERIAL, failure_cfg=failure)
        else:
            r = fast_screen(geom, MATERIAL, float(p["angle"]), float(p["band"]), failure_cfg=failure)
    return {"nodes": np.asarray(r.nodes, np.float64).round(3).tolist(),
            "elems": np.asarray(r.elems, np.int64).tolist(),
            "fields": {k: np.asarray(v, np.float64).round(5).tolist() for k, v in r.fields.items()},
            "margins": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.margins.items()},
            "burst": float(r.burst_factor), "fi_max": float(r.fi_max),
            "mu": None if r.mu_max_required is None else float(r.mu_max_required),
            "decision": r.gate["decision"].upper(), "mode": r.mode}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

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
