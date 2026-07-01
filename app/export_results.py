"""Run the real engine and emit a self-contained interactive results viewer.

This is the honest counterpart to app/demo/studio.html: instead of a cosmetic model
with black bands laid over a shape, it colours the ACTUAL solver mesh by the ACTUAL
computed fields — failure index, the four Hashin modes, reserve factor, deformation.
Real material physics in, real contours out.

The mesh + fields are embedded directly into a standalone HTML (no server, no fetch,
opens by double-click). Interactive 3D via Three.js.

    python -m app.export_results                 # default COPV, fast screen
    python -m app.export_results --optimize      # full winding optimization
"""

from __future__ import annotations

import argparse
import json
import os
import signal as _sig
import sys
import threading as _thr
import webbrowser
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_real_signal = _sig.signal
_sig.signal = lambda s, h: (_real_signal(s, h) if _thr.current_thread() is _thr.main_thread() else None)

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.engine import fast_screen, full_optimize
from copv_opt.config import FailureConfig, GeometryConfig, MaterialAllowables, MaterialConfig


VIEWER_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>COPV Studio — real results</title>
<style>
  html,body{margin:0;height:100%;background:#1b1f25;color:#dfe4ea;font-family:"Segoe UI",system-ui,sans-serif;overflow:hidden}
  #bar{height:44px;display:flex;align-items:center;gap:14px;padding:0 16px;background:#23272e;border-bottom:1px solid #3a414b}
  #bar b{color:#eef2f6;font-weight:600}
  #bar .tag{color:#8b95a1;font-size:12px}
  #wrap{position:absolute;top:44px;left:0;right:0;bottom:0}
  #gl{width:100%;height:100%;display:block}
  .panel{position:absolute;top:60px;left:14px;background:#1b212bee;border:1px solid #3a414b;border-radius:8px;padding:12px 14px;width:230px;font-size:12px}
  .panel h3{margin:0 0 8px;font-size:12px;font-weight:600;color:#eef2f6}
  select{width:100%;background:#262b33;color:#dfe4ea;border:1px solid #3a414b;border-radius:5px;padding:6px;font-size:12px;margin-bottom:10px}
  .row{display:flex;justify-content:space-between;margin:3px 0;color:#b8c2cc}
  .row b{color:#eef2f6;font-weight:600}
  .legend{position:absolute;bottom:16px;left:14px;display:flex;gap:8px;align-items:flex-end}
  .bar{width:16px;height:180px;border:1px solid #0006;border-radius:2px;
    background:linear-gradient(to top,#000084,#1414ff,#00b4ff,#28d76b,#d7e000,#ff7a00,#d60000,#7a0000)}
  .ticks{display:flex;flex-direction:column;justify-content:space-between;height:180px;font-size:10.5px;color:#cdd6df}
  .gate{position:absolute;bottom:16px;right:16px;background:#3a2a0e;border:1px solid #6b4f16;color:#e0a23a;
    border-radius:6px;padding:6px 12px;font-size:12px;font-weight:600}
</style></head><body>
<div id="bar"><b>COPV Studio — real results</b>
  <span class="tag">solver: CG shell · Hashin · __MODE__</span>
  <span class="tag">__NELEM__ elements</span></div>
<div id="wrap"><canvas id="gl"></canvas>
  <div class="panel">
    <h3>Result field</h3>
    <select id="field"></select>
    <div class="row"><span>FI max</span><b id="m_fi"></b></div>
    <div class="row"><span>Min reserve factor</span><b id="m_rf"></b></div>
    <div class="row"><span>Critical mode</span><b id="m_mode"></b></div>
    <div class="row"><span>Max deformation</span><b id="m_def"></b></div>
    <div class="row"><span>Burst factor</span><b id="m_burst"></b></div>
  </div>
  <div class="legend"><div class="bar"></div><div class="ticks" id="ticks"></div></div>
  <div class="gate">__DECISION__</div>
</div>
<script type="importmap">{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/" }}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const DATA = __DATA__;

function jet(t){ t=Math.max(0,Math.min(1,t));
  return [Math.max(0,Math.min(1,1.5-Math.abs(4*t-3))),Math.max(0,Math.min(1,1.5-Math.abs(4*t-2))),Math.max(0,Math.min(1,1.5-Math.abs(4*t-1)))]; }

const nodes=DATA.nodes, elems=DATA.elems, nElem=elems.length;
const positions=new Float32Array(nElem*9);
for(let e=0;e<nElem;e++){ const [i,j,k]=elems[e];
  positions.set([...nodes[i],...nodes[j],...nodes[k]], e*9); }
const geo=new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(positions,3));
const colors=new Float32Array(nElem*9);
geo.setAttribute('color', new THREE.BufferAttribute(colors,3));
geo.computeVertexNormals();
const mat=new THREE.MeshLambertMaterial({vertexColors:true, side:THREE.DoubleSide, flatShading:true});
const mesh=new THREE.Mesh(geo,mat);

// center + scale
const box=new THREE.Box3().setFromBufferAttribute(geo.attributes.position);
const c=box.getCenter(new THREE.Vector3()), size=box.getSize(new THREE.Vector3()).length();
mesh.position.sub(c);

const scene=new THREE.Scene();
scene.add(mesh);
scene.add(new THREE.AmbientLight(0xffffff,0.85));
const d=new THREE.DirectionalLight(0xffffff,0.5); d.position.set(1,1,1); scene.add(d);
const wrap=document.getElementById('wrap');
const camera=new THREE.PerspectiveCamera(40,1,size*0.01,size*10);
camera.position.set(size*0.7,size*0.4,size*0.7);
const renderer=new THREE.WebGLRenderer({canvas:document.getElementById('gl'),antialias:true});
renderer.setClearColor(0x222831); renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const controls=new OrbitControls(camera,renderer.domElement); controls.enableDamping=true; controls.autoRotate=true; controls.autoRotateSpeed=0.7;
function resize(){const w=wrap.clientWidth,h=wrap.clientHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();}
addEventListener('resize',resize); resize();
(function loop(){requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera);})();

function recolor(field){
  const vals=DATA.fields[field]; let lo=Infinity,hi=-Infinity;
  for(const v of vals){ if(v<lo)lo=v; if(v>hi)hi=v; } if(hi-lo<1e-9) hi=lo+1;
  for(let e=0;e<nElem;e++){ const [r,g,b]=jet((vals[e]-lo)/(hi-lo));
    for(let v=0;v<3;v++){ const o=(e*3+v)*3; colors[o]=r; colors[o+1]=g; colors[o+2]=b; } }
  geo.attributes.color.needsUpdate=true;
  const ticks=document.getElementById('ticks'); ticks.innerHTML='';
  for(let i=6;i>=0;i--){ const val=lo+(hi-lo)*i/6; const el=document.createElement('div'); el.textContent=val.toPrecision(3); ticks.appendChild(el); }
}
const sel=document.getElementById('field');
Object.keys(DATA.fields).forEach(f=>{ const o=document.createElement('option'); o.value=f; o.textContent=f; sel.appendChild(o); });
sel.value=DATA.default_field; sel.addEventListener('change',()=>recolor(sel.value)); recolor(DATA.default_field);

const M=DATA.margins;
document.getElementById('m_fi').textContent=M.fi_max.toFixed(3);
document.getElementById('m_rf').textContent=M.min_reserve_factor.toFixed(2);
document.getElementById('m_mode').textContent=M.critical_mode.replace('_',' ');
document.getElementById('m_def').textContent=M.max_deformation_mm.toFixed(3)+' mm';
document.getElementById('m_burst').textContent=DATA.burst_factor.toFixed(2)+'×';
</script></body></html>"""


def _parse():
    p = argparse.ArgumentParser(description="Export a real-engine interactive results viewer.")
    p.add_argument("--radius", type=float, default=100.0)
    p.add_argument("--length", type=float, default=220.0)
    p.add_argument("--thickness", type=float, default=8.0)
    p.add_argument("--pressure", type=float, default=6.85, help="Design pressure [MPa]")
    p.add_argument("--angle", type=float, default=42.0)
    p.add_argument("--band", type=float, default=8.0)
    p.add_argument("--optimize", action="store_true")
    p.add_argument("--out", type=Path, default=Path("outputs") / "studio_export" / "results_viewer.html")
    p.add_argument("--no-open", action="store_true")
    return p.parse_args()


def write_results_viewer(r, out: Path) -> Path:
    """Build the self-contained interactive results viewer from a DesignResult."""
    data = {
        "nodes": np.asarray(r.nodes, dtype=np.float64).round(4).tolist(),
        "elems": np.asarray(r.elems, dtype=np.int64).tolist(),
        "fields": {name: np.asarray(vals, dtype=np.float64).round(6).tolist()
                   for name, vals in r.fields.items()},
        "default_field": "Failure index (Hashin)",
        "margins": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in r.margins.items()},
        "burst_factor": float(r.burst_factor),
    }
    html = (VIEWER_TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__MODE__", r.mode.replace("_", " "))
            .replace("__NELEM__", str(len(r.elems)))
            .replace("__DECISION__", r.gate["decision"].upper()))
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    args = _parse()
    geom = GeometryConfig(outer_radius=args.radius, cylinder_length=args.length,
                          thickness=args.thickness, pressure=args.pressure)
    material = MaterialConfig()
    failure = FailureConfig(allowables=MaterialAllowables(), margin_of_safety=1.0)

    print("Solving (real FEA)…")
    if args.optimize:
        r = full_optimize(geom, material, failure_cfg=failure)
    else:
        r = fast_screen(geom, material, args.angle, args.band, failure_cfg=failure)

    out = write_results_viewer(r, args.out)
    print(f"FI_max {r.fi_max:.3f} · min RF {r.margins['min_reserve_factor']:.2f} · "
          f"critical {r.margins['critical_mode']} · burst {r.burst_factor:.2f}x")
    print(f"Wrote {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
