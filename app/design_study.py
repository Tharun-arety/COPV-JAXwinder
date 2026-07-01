"""Design study — the tool's core job, made visible and responsive.

Workflow a structural engineer follows:
1. Solve the UNWOUND vessel under pressure -> where does it fail?  (baseline, all red)
2. Run the winding optimiser -> it varies angle + thickness + pass density along the
   vessel to reduce failure.  (optimised, turns blue = safe)
3. Show the OPTIMISED winding-angle distribution -> which angles, where.
4. A quick constant-angle sweep shows that a *uniform* angle barely helps — the gain
   is in spatially varying the layup, which is the point.

Every state is a real FEA solve on a fixed, meaningful scale (failure index, 1.0 =
failure: red = fails, blue = safe), so results genuinely change with geometry /
pressure / material. Output is a self-contained interactive HTML.

    python -m app.design_study --pressure 6.85
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
_rs = _sig.signal
_sig.signal = lambda s, h: (_rs(s, h) if _thr.current_thread() is _thr.main_thread() else None)

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.engine import build_state, fast_screen, full_optimize
from copv_opt.config import FailureConfig, GeometryConfig, MaterialConfig
from copv_opt.physics import baseline_response, evaluate_hashin_failure, rotate_stiffness_field

FI_CLIM = 1.2  # 1.0 = first-ply failure


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>COPV design study</title>
<style>
 html,body{margin:0;height:100%;background:#1b1f25;color:#dfe4ea;font-family:"Segoe UI",system-ui,sans-serif;overflow:hidden}
 #bar{height:44px;display:flex;align-items:center;gap:14px;padding:0 16px;background:#23272e;border-bottom:1px solid #3a414b}
 #bar b{color:#eef2f6}#bar .tag{color:#8b95a1;font-size:12px}
 #main{position:absolute;top:44px;left:0;right:0;bottom:0;display:flex}
 #side{width:300px;background:#20252c;border-right:1px solid #3a414b;padding:14px;overflow:auto}
 #view{flex:1;position:relative}#gl{width:100%;height:100%;display:block}
 h3{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#8b95a1;margin:16px 0 8px}
 .big{font-size:12px;color:#c3ccd6;line-height:1.65}.big b{color:#eef2f6}
 .state{display:block;width:100%;text-align:left;background:#262b33;color:#dfe4ea;border:1px solid #3a414b;
   border-radius:6px;padding:8px 10px;margin:4px 0;cursor:pointer;font-size:12.5px}
 .state:hover{border-color:#1d9e75}.state.on{background:#1d3b31;border-color:#1d9e75;color:#fff}
 .state span{float:right;color:#9aa4b1}
 .legend{position:absolute;bottom:16px;left:16px;display:flex;gap:8px;align-items:flex-end}
 .lbar{width:16px;height:180px;border:1px solid #0006;border-radius:2px;
   background:linear-gradient(to top,#000084,#1414ff,#00b4ff,#28d76b,#d7e000,#ff7a00,#d60000)}
 .lt{display:flex;flex-direction:column;justify-content:space-between;height:180px;font-size:10.5px;color:#cdd6df}
 .lu{position:absolute;bottom:200px;left:16px;font-size:11px;color:#b8c2cc}
 .vtitle{position:absolute;top:12px;left:16px;font-size:14px;color:#eef2f6;text-shadow:0 1px 3px #000}
 .vtitle small{display:block;color:#b9c4d0;font-size:12px}
 svg text{fill:#b8c2cc}
</style></head><body>
<div id="bar"><b>COPV design study</b><span class="tag">real FEA · failure index (1.0 = failure)</span>
 <span class="tag">__NELEM__ elements · p=__PRESS__ MPa</span></div>
<div id="main">
 <div id="side">
  <h3>What the tool did</h3>
  <div class="big" id="summary"></div>
  <h3>Constant-angle check</h3>
  <svg id="chart" viewBox="0 0 270 165" style="width:100%"></svg>
  <div class="big" style="font-size:11px;color:#9aa4b1">A uniform angle barely helps — the optimiser wins by varying angle + thickness along the vessel.</div>
  <h3>View on the vessel</h3>
  <div id="states"></div>
 </div>
 <div id="view"><canvas id="gl"></canvas>
  <div class="vtitle" id="vtitle"></div>
  <div class="lu" id="lu"></div>
  <div class="legend"><div class="lbar"></div><div class="lt" id="lt"></div></div>
 </div>
</div>
<script type="importmap">{ "imports": {
 "three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
 "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/" }}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const D=__DATA__;
function jet(t){t=Math.max(0,Math.min(1,t));return [Math.max(0,Math.min(1,1.5-Math.abs(4*t-3))),Math.max(0,Math.min(1,1.5-Math.abs(4*t-2))),Math.max(0,Math.min(1,1.5-Math.abs(4*t-1)))];}
const nodes=D.nodes, elems=D.elems, nE=elems.length;
const positions=new Float32Array(nE*9);
for(let e=0;e<nE;e++){const[i,j,k]=elems[e];positions.set([...nodes[i],...nodes[j],...nodes[k]],e*9);}
const geo=new THREE.BufferGeometry(); geo.setAttribute('position',new THREE.BufferAttribute(positions,3));
const colors=new Float32Array(nE*9); geo.setAttribute('color',new THREE.BufferAttribute(colors,3)); geo.computeVertexNormals();
const mesh=new THREE.Mesh(geo,new THREE.MeshLambertMaterial({vertexColors:true,side:THREE.DoubleSide,flatShading:true}));
const box=new THREE.Box3().setFromBufferAttribute(geo.attributes.position);
const ctr=box.getCenter(new THREE.Vector3()), sz=box.getSize(new THREE.Vector3()).length(); mesh.position.sub(ctr);
const scene=new THREE.Scene(); scene.add(mesh); scene.add(new THREE.AmbientLight(0xffffff,0.85));
const dl=new THREE.DirectionalLight(0xffffff,0.5); dl.position.set(1,1,1); scene.add(dl);
const view=document.getElementById('view');
const cam=new THREE.PerspectiveCamera(40,1,sz*0.01,sz*10); cam.position.set(sz*0.7,sz*0.4,sz*0.7);
const rnd=new THREE.WebGLRenderer({canvas:document.getElementById('gl'),antialias:true}); rnd.setClearColor(0x222831); rnd.setPixelRatio(Math.min(devicePixelRatio,2));
const ctl=new OrbitControls(cam,rnd.domElement); ctl.enableDamping=true; ctl.autoRotate=true; ctl.autoRotateSpeed=0.6;
function rs(){const w=view.clientWidth,h=view.clientHeight; rnd.setSize(w,h,false); cam.aspect=w/h; cam.updateProjectionMatrix();}
addEventListener('resize',rs); rs();
(function loop(){requestAnimationFrame(loop); ctl.update(); rnd.render(scene,cam);})();

function show(key){
 const st=D.states[key], lo=st.clim[0], hi=st.clim[1], rng=(hi-lo)||1, v=st.values;
 for(let e=0;e<nE;e++){const[r,g,b]=jet((v[e]-lo)/rng); for(let q=0;q<3;q++){const o=(e*3+q)*3; colors[o]=r;colors[o+1]=g;colors[o+2]=b;}}
 geo.attributes.color.needsUpdate=true;
 let sub = st.kind==='fi'
   ? `FI max ${st.fi_max.toFixed(2)} · ${st.fi_max<=1?'SAFE':'FAILS'} · min reserve ${(1/Math.sqrt(Math.max(st.fi_max,1e-9))).toFixed(2)}`
   : `${st.unit} · range ${lo.toFixed(0)}–${hi.toFixed(0)}`;
 document.getElementById('vtitle').innerHTML=`<b>${st.label}</b><small>${sub}</small>`;
 document.getElementById('lu').textContent=st.unit;
 const lt=document.getElementById('lt'); lt.innerHTML='';
 for(let i=6;i>=0;i--){const d=document.createElement('div'); d.textContent=(lo+rng*i/6).toFixed(st.kind==='fi'?2:0); lt.appendChild(d);}
 document.querySelectorAll('.state').forEach(b=>b.classList.toggle('on',b.dataset.k===key));
}
const sc=document.getElementById('states');
D.order.forEach(key=>{const st=D.states[key]; const b=document.createElement('button'); b.className='state'; b.dataset.k=key;
 const tag = st.kind==='fi' ? `${st.fi_max<=1?'safe':'fails'} · FI ${st.fi_max.toFixed(2)}` : 'angles';
 b.innerHTML=`${st.label}<span>${tag}</span>`; b.onclick=()=>show(key); sc.appendChild(b);});
document.getElementById('summary').innerHTML =
 `Bare vessel fails at <b>FI ${D.baseline_fi_max.toFixed(0)}</b> (≫1) — red everywhere.<br>`+
 `Optimised winding → <b>FI ${D.opt_fi.toFixed(2)}</b> (${D.opt_fi<=1?'safe':'still failing'}), reserve <b>${(1/Math.sqrt(D.opt_fi)).toFixed(2)}</b>.<br>`+
 `Optimised angles span <b>${D.angle_lo}–${D.angle_hi}°</b> along the vessel.`;
(function(){
 const sw=D.sweep; if(!sw.length){return;}
 const xs=sw.map(p=>p[0]), ys=sw.map(p=>p[1]);
 const x0=34,x1=258,y0=135,y1=14, ax=Math.min(...xs)-3, bx=Math.max(...xs)+3, ymax=Math.max(1.25,...ys)*1.05;
 const X=a=>x0+(a-ax)/(bx-ax)*(x1-x0), Y=v=>y0-(v/ymax)*(y0-y1);
 let s=`<line x1="${x0}" y1="${Y(1).toFixed(1)}" x2="${x1}" y2="${Y(1).toFixed(1)}" stroke="#a35" stroke-dasharray="4 3"/>`;
 s+=`<text x="${x1}" y="${(Y(1)-3).toFixed(1)}" font-size="8" text-anchor="end" fill="#e88">failure (FI=1)</text>`;
 s+=`<polyline points="${sw.map(p=>X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join(' ')}" fill="none" stroke="#7fb0e0" stroke-width="1.6"/>`;
 sw.forEach(p=>{s+=`<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="2.6" fill="#7fb0e0"/>`;});
 s+=`<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y0}" stroke="#566270"/>`;
 s+=`<text x="145" y="157" font-size="8.5" text-anchor="middle">uniform helical angle (deg)</text>`;
 document.getElementById('chart').innerHTML=s;
})();
show(D.default);
</script></body></html>"""


def _fi_field(state, disp, fiber_dirs, failure):
    c = rotate_stiffness_field(state["c_mat"], fiber_dirs, state["surface_normals"])
    return np.asarray(evaluate_hashin_failure(state, disp, c, fiber_dirs, failure)["failure_index"], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="COPV design study: baseline -> optimised, with angle field.")
    ap.add_argument("--radius", type=float, default=100.0)
    ap.add_argument("--length", type=float, default=220.0)
    ap.add_argument("--thickness", type=float, default=8.0)
    ap.add_argument("--pressure", type=float, default=6.85)
    ap.add_argument("--band", type=float, default=8.0, help="Band for the constant-angle check")
    ap.add_argument("--angles", type=float, nargs="+", default=[15, 30, 45, 58])
    ap.add_argument("--out", type=Path, default=Path("outputs") / "studio_export" / "design_study.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    geom = GeometryConfig(outer_radius=args.radius, cylinder_length=args.length,
                          thickness=args.thickness, pressure=args.pressure)
    material, failure = MaterialConfig(), FailureConfig(margin_of_safety=1.0)
    bundle = build_state(geom, material)
    state = bundle["state"]

    def fi_state(label, fi, fi_max):
        return {"label": label, "kind": "fi", "values": np.minimum(np.asarray(fi), FI_CLIM * 3).round(5).tolist(),
                "clim": [0.0, FI_CLIM], "unit": "failure index", "fi_max": float(fi_max)}

    print("Solving bare vessel (baseline)…")
    base = baseline_response(state, material, bundle["solve"])
    base_fi = _fi_field(state, base["displacement"], base["fiber_dirs"], failure)
    states = {"baseline": fi_state("Bare vessel (no winding)", base_fi, float(np.max(base_fi)))}
    order = ["baseline"]

    print("Optimising winding (angle + thickness + pass density)…")
    opt = full_optimize(geom, material, failure_cfg=failure)
    opt_fi = np.asarray(opt.fields["Failure index (Hashin)"], dtype=np.float64)
    ang = np.asarray(opt.fields["Winding angle [deg]"], dtype=np.float64)
    states["optimised"] = fi_state("Optimised winding", opt_fi, opt.fi_max)
    a_lo, a_hi = float(np.min(ang)), float(np.max(ang))
    states["angle"] = {"label": "Optimised winding angle", "kind": "angle",
                       "values": ang.round(3).tolist(), "clim": [a_lo, a_hi], "unit": "winding angle [deg]", "fi_max": 0.0}
    order += ["optimised", "angle"]

    sweep = []
    for a in args.angles:
        print(f"Constant-angle check {a:.0f}°…")
        r = fast_screen(geom, material, float(a), args.band, failure_cfg=failure)
        sweep.append([float(a), float(r.fi_max)])
        key = f"a{int(a)}"
        states[key] = fi_state(f"Uniform {int(a)}°", np.asarray(r.fields["Failure index (Hashin)"]), r.fi_max)
        order.append(key)

    data = {
        "nodes": np.asarray(bundle["nodes"], dtype=np.float64).round(4).tolist(),
        "elems": np.asarray(bundle["elems"], dtype=np.int64).tolist(),
        "states": states, "order": order, "default": "optimised", "sweep": sweep,
        "baseline_fi_max": states["baseline"]["fi_max"], "opt_fi": float(opt.fi_max),
        "angle_lo": round(a_lo), "angle_hi": round(a_hi),
    }
    html = (TEMPLATE.replace("__DATA__", json.dumps(data))
            .replace("__NELEM__", str(len(bundle["elems"]))).replace("__PRESS__", f"{args.pressure:g}"))
    out = args.out.resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\nBaseline FI_max {states['baseline']['fi_max']:.0f}  ->  optimised FI_max {opt.fi_max:.2f} "
          f"(angles {round(a_lo)}–{round(a_hi)}°)")
    print(f"Wrote {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
