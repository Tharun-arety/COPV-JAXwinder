"""Unified COPV Studio: Studio-like UI backed by the real solver.

This is the single tool that bridges the gap between:
- app/demo/studio.html: polished but synthetic browser demo
- app.studio_app: real solver, but separate trame UI
- app.export_results: real results, but static after generation

Run:
    python -m app.studio_unified

Then open:
    http://localhost:8088
"""

from __future__ import annotations

import warnings

warnings.warn("app.studio_unified is deprecated; use `python -m app.server` "
              "(the workflow web app at http://localhost:8081).", DeprecationWarning, stacklevel=2)

import argparse
import json
import os
import signal as _signal_mod
import sys
import threading as _threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# gmsh may install a SIGINT handler. Solves run in HTTP worker threads, so avoid
# "signal only works in main thread" without changing main-thread behavior.
_real_signal = _signal_mod.signal


def _thread_safe_signal(sig, handler):
    if _threading.current_thread() is _threading.main_thread():
        return _real_signal(sig, handler)
    return None


_signal_mod.signal = _thread_safe_signal

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.engine import fast_screen, full_optimize
from copv_opt.config import FailureConfig, GeometryConfig, MaterialAllowables, MaterialConfig


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>COPV Studio - live solver</title>
<style>
  :root{
    --bar:#23272e; --bar2:#2b3038; --panel:#262b33; --line:#3a414b; --ink:#dfe4ea;
    --muted:#9aa4b1; --accent:#1d9e75; --accentd:#0f6e56; --sel:#34506b; --warn:#e0a23a;
    --bad:#e05757; --good:#38b77a;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;font-family:"Segoe UI",system-ui,sans-serif;color:var(--ink);
    background:#1b1f25;font-size:13px;overflow:hidden}
  #app{display:grid;grid-template-rows:30px 42px 1fr 26px;height:100vh}
  .titlebar{background:var(--bar);display:flex;align-items:center;gap:16px;padding:0 12px;border-bottom:1px solid var(--line)}
  .titlebar .brand{font-weight:600;color:#fff;display:flex;align-items:center;gap:8px}
  .titlebar .brand i{width:16px;height:16px;background:var(--accent);border-radius:4px;display:inline-block}
  .menu{display:flex;gap:14px;color:var(--muted)}
  .toolbar{background:var(--bar2);display:flex;align-items:center;gap:6px;padding:0 8px;border-bottom:1px solid var(--line);overflow-x:auto}
  .tbtn{display:flex;align-items:center;gap:6px;padding:6px 10px;border-radius:5px;color:var(--ink);cursor:pointer;
    border:1px solid transparent;font-size:12px;background:transparent;white-space:nowrap}
  .tbtn:hover{background:#333a44;border-color:var(--line)}
  .tbtn.on{background:var(--sel);border-color:#4a6e92;color:#fff}
  .tbtn.solve{background:var(--accentd);border-color:#15876a;color:#fff;font-weight:600;margin-left:4px}
  .tbtn.solve:hover{background:var(--accent)}
  .tbtn.danger{background:#6e2b2b;border-color:#8f3b3b}
  .sep{width:1px;height:22px;background:var(--line);margin:0 6px}
  .lbl{color:var(--muted);font-size:11px}
  .main{display:grid;grid-template-columns:300px 1fr 310px;min-height:0}
  .tree,.props{background:var(--panel);overflow:auto;padding:8px 0}
  .tree{border-right:1px solid var(--line)}
  .props{border-left:1px solid var(--line);padding:10px}
  .grp{padding:8px 12px 5px;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
  .node{display:flex;align-items:center;gap:7px;padding:4px 10px 4px 14px;color:var(--ink);font-size:12.5px}
  .node.ind{padding-left:30px}.node.ind2{padding-left:46px}
  .node.sel{background:var(--sel);color:#fff}
  .sw{width:9px;height:9px;border-radius:2px;border:1px solid #5a626d;margin-right:2px}
  .sw.on{background:var(--accent);border-color:var(--accent)}
  .node small{color:var(--muted);margin-left:auto;font-size:11px}
  .viewport{position:relative;background:radial-gradient(120% 120% at 50% 18%,#3a4658 0%,#222831 70%,#1a1f26 100%);overflow:hidden}
  #gl{width:100%;height:100%;display:block}
  .vp-title{position:absolute;top:10px;left:14px;color:#eef2f6;font-size:13px;line-height:1.5;text-shadow:0 1px 2px #000}
  .vp-title b{font-weight:600}.vp-title span{color:#b9c4d0;font-size:11.5px}
  .legend{position:absolute;top:70px;left:14px;width:132px;color:#eef2f6;text-shadow:0 1px 2px #000;font-size:11px}
  .legend .lt{font-size:11.5px;margin-bottom:6px;font-weight:600}
  .legend .barwrap{display:flex;gap:6px}
  .bar{width:16px;height:200px;border:1px solid #00000055;border-radius:2px;
    background:linear-gradient(to top,#000084,#1414ff,#00b4ff,#28d76b,#d7e000,#ff7a00,#d60000,#7a0000)}
  .ticks{display:flex;flex-direction:column;justify-content:space-between;height:200px;font-size:10.5px;color:#dfe6ee}
  .overlay{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:#11151bcc;
    color:#eef2f6;z-index:5;flex-direction:column;gap:14px}
  .overlay.show{display:flex}
  .spinner{width:34px;height:34px;border:3px solid #3a414b;border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .vp-hint{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);color:#eef2f6;background:#1b212bdd;
    border:1px solid #3a414b;border-radius:6px;padding:7px 14px;font-size:12px}
  .card{background:#1f252d;border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:10px}
  .card h3{margin:0 0 8px;font-size:12px;color:#eef2f6;text-transform:uppercase;letter-spacing:.05em}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
  label{display:block;color:var(--muted);font-size:11px;margin-bottom:3px}
  input,select{width:100%;background:#151a20;color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:6px;font-size:12px}
  .row{display:flex;justify-content:space-between;gap:10px;margin:4px 0;color:#c3ccd6}
  .row b{color:#fff;font-weight:600;text-align:right}
  .decision{border-radius:6px;padding:7px 9px;font-weight:600;text-align:center;margin-top:8px;background:#3a2a0e;color:var(--warn);border:1px solid #6b4f16}
  .decision.ok{background:#123524;color:var(--good);border-color:#246747}
  .blockers{margin:7px 0 0;padding-left:16px;color:#d3b27b;font-size:11.5px;line-height:1.4}
  .status{background:var(--bar);border-top:1px solid var(--line);display:flex;align-items:center;gap:18px;padding:0 12px;color:var(--muted);font-size:11.5px}
  .status .gate{margin-left:auto;color:var(--warn);font-weight:600}
  .status b{color:var(--ink);font-weight:600}
  .error{color:#ff9f9f;white-space:pre-wrap;font-size:11.5px}
</style>
</head>
<body>
<div id="app">
  <div class="titlebar">
    <div class="brand"><i></i> COPV Studio</div>
    <div class="menu"><span>File</span><span>View</span><span>Solve</span><span>Export</span><span>Help</span></div>
  </div>
  <div class="toolbar">
    <div class="tbtn on" id="fit">Fit</div>
    <div class="tbtn on" id="rotate">Rotate</div>
    <div class="sep"></div>
    <div class="tbtn on" id="showMesh">Overwrap mesh</div>
    <div class="tbtn on" id="showEdges">Edges</div>
    <div class="sep"></div>
    <span class="lbl">Result:</span>
    <select id="fieldSelect" style="width:220px"></select>
    <div class="sep"></div>
    <div class="tbtn solve" id="solveBtn">Solve</div>
    <div class="tbtn solve" id="optBtn">Optimize winding</div>
  </div>
  <div class="main">
    <div class="tree">
      <div class="grp">Outline</div>
      <div class="node"><span>v</span> Model <small id="treeType">Type 3/4 screen</small></div>
      <div class="node ind"><span class="sw on"></span> Geometry <small>parametric COPV</small></div>
      <div class="node ind"><span class="sw on"></span> Liner <small>not coupled in shell</small></div>
      <div class="node ind"><span class="sw on"></span> Composite overwrap</div>
      <div class="node ind2">Helical / hoop winding field</div>
      <div class="grp">Loads</div>
      <div class="node ind2">Internal pressure <small id="pressureLabel">6.85 MPa</small></div>
      <div class="node ind2">Boss supports <small>fixed support ring</small></div>
      <div class="grp">Results</div>
      <div class="node ind sel">Failure index</div>
      <div class="node ind">Reserve factor</div>
      <div class="node ind">Hashin modes</div>
      <div class="node ind">Winding angle</div>
      <div class="grp">Release</div>
      <div class="node ind">Gate <small id="treeGate">do_not_release</small></div>
    </div>
    <div class="viewport">
      <canvas id="gl"></canvas>
      <div class="vp-title"><b>Live COPV solver result</b><br><span id="subTitle">Set inputs and press Solve</span></div>
      <div class="legend">
        <div class="lt" id="legendTitle">Field</div>
        <div class="barwrap"><div class="bar"></div><div class="ticks" id="ticks"></div></div>
      </div>
      <div class="overlay" id="overlay"><div class="spinner"></div><div id="overlayText">Running solver...</div></div>
      <div class="vp-hint">Drag to orbit. Inputs on the right call the real Python/JAX engine.</div>
    </div>
    <div class="props">
      <div class="card">
        <h3>Geometry / Load</h3>
        <div class="grid">
          <div><label>Outer radius [mm]</label><input id="outer_radius" type="number" value="100"></div>
          <div><label>Cylinder length [mm]</label><input id="cyl_length" type="number" value="220"></div>
          <div><label>Wall thickness [mm]</label><input id="wall_thickness" type="number" value="8"></div>
          <div><label>Opening radius [mm]</label><input id="opening_radius" type="number" value="10"></div>
          <div><label>Dome ratio</label><input id="dome_ratio" type="number" value="0.7" step="0.05"></div>
          <div><label>Pressure [MPa]</label><input id="pressure" type="number" value="6.85" step="0.1"></div>
        </div>
      </div>
      <div class="card">
        <h3>Winding / Analysis</h3>
        <div class="grid">
          <div><label>Fast angle [deg]</label><input id="angle_deg" type="number" value="42"></div>
          <div><label>Band thickness [mm]</label><input id="band_mm" type="number" value="8"></div>
        </div>
      </div>
      <div class="card">
        <h3>Allowables [MPa]</h3>
        <div class="grid">
          <div><label>XT</label><input id="xt" type="number" value="2200"></div>
          <div><label>XC</label><input id="xc" type="number" value="1400"></div>
          <div><label>YT</label><input id="yt" type="number" value="70"></div>
          <div><label>YC</label><input id="yc" type="number" value="220"></div>
          <div><label>S</label><input id="s" type="number" value="120"></div>
        </div>
      </div>
      <div class="card">
        <h3>Result Summary</h3>
        <div class="row"><span>Mode</span><b id="r_mode">-</b></div>
        <div class="row"><span>Elements</span><b id="r_elems">-</b></div>
        <div class="row"><span>FI max</span><b id="r_fi">-</b></div>
        <div class="row"><span>Min reserve factor</span><b id="r_rf">-</b></div>
        <div class="row"><span>Critical mode</span><b id="r_crit">-</b></div>
        <div class="row"><span>Max deformation</span><b id="r_def">-</b></div>
        <div class="row"><span>Burst factor</span><b id="r_burst">-</b></div>
        <div class="row"><span>Friction</span><b id="r_mu">-</b></div>
        <div class="decision" id="decision">NO RESULT</div>
        <ul class="blockers" id="blockers"></ul>
        <pre class="error" id="error"></pre>
      </div>
    </div>
  </div>
  <div class="status">
    <span>Solver: <b>real Python/JAX shell engine</b></span>
    <span id="meshStatus">No result</span>
    <span class="gate" id="gateStatus">Gate: -</span>
  </div>
</div>
<script type="importmap">{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/" }}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let DATA=null, mesh=null, edges=null;
const canvas=document.getElementById('gl');
const scene=new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff,0.85));
const d=new THREE.DirectionalLight(0xffffff,0.55); d.position.set(1,1,1); scene.add(d);
const camera=new THREE.PerspectiveCamera(40,1,0.1,10000);
camera.position.set(320,190,320);
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setClearColor(0x222831);
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true; controls.autoRotate=true; controls.autoRotateSpeed=0.4;

function resize(){
  const vp=document.querySelector('.viewport');
  renderer.setSize(vp.clientWidth,vp.clientHeight,false);
  camera.aspect=vp.clientWidth/vp.clientHeight;
  camera.updateProjectionMatrix();
}
addEventListener('resize',resize); resize();
function loop(){requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera);} loop();

document.getElementById('rotate').onclick=()=>{controls.autoRotate=!controls.autoRotate; document.getElementById('rotate').classList.toggle('on', controls.autoRotate);};
document.getElementById('showMesh').onclick=()=>{if(mesh){mesh.visible=!mesh.visible; document.getElementById('showMesh').classList.toggle('on', mesh.visible);}};
document.getElementById('showEdges').onclick=()=>{if(edges){edges.visible=!edges.visible; document.getElementById('showEdges').classList.toggle('on', edges.visible);}};
document.getElementById('fit').onclick=fitCamera;
document.getElementById('solveBtn').onclick=()=>solve('screen');
document.getElementById('optBtn').onclick=()=>solve('optimize');
document.getElementById('fieldSelect').onchange=()=>recolor(document.getElementById('fieldSelect').value);

function jet(t){
  t=Math.max(0,Math.min(1,t));
  return [Math.max(0,Math.min(1,1.5-Math.abs(4*t-3))),
          Math.max(0,Math.min(1,1.5-Math.abs(4*t-2))),
          Math.max(0,Math.min(1,1.5-Math.abs(4*t-1)))];
}

function value(id){ return Number(document.getElementById(id).value); }
function payload(mode){
  return {
    mode,
    geometry:{
      outer_radius:value('outer_radius'),
      cylinder_length:value('cyl_length'),
      thickness:value('wall_thickness'),
      opening_radius:value('opening_radius'),
      dome_height_ratio:value('dome_ratio'),
      pressure:value('pressure')
    },
    winding:{ angle_deg:value('angle_deg'), band_mm:value('band_mm') },
    allowables:{ xt:value('xt'), xc:value('xc'), yt:value('yt'), yc:value('yc'), s:value('s') }
  };
}

async function solve(mode){
  const overlay=document.getElementById('overlay');
  document.getElementById('error').textContent='';
  document.getElementById('overlayText').textContent = mode==='optimize' ? 'Optimizing winding. This can take minutes...' : 'Meshing and solving...';
  overlay.classList.add('show');
  try{
    const res=await fetch('/api/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload(mode))});
    const obj=await res.json();
    if(!res.ok || obj.error){ throw new Error(obj.error || ('HTTP '+res.status)); }
    DATA=obj;
    buildMesh(obj);
    fillUi(obj);
  }catch(err){
    document.getElementById('error').textContent=String(err.stack || err);
  }finally{
    overlay.classList.remove('show');
  }
}

function buildMesh(data){
  if(mesh){ scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); mesh=null; }
  if(edges){ scene.remove(edges); edges.geometry.dispose(); edges.material.dispose(); edges=null; }
  const nodes=data.nodes, elems=data.elems, n=elems.length;
  const pos=new Float32Array(n*9);
  for(let e=0;e<n;e++){
    const tri=elems[e];
    pos.set([...nodes[tri[0]],...nodes[tri[1]],...nodes[tri[2]]], e*9);
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(new Float32Array(n*9),3));
  geo.computeVertexNormals();
  mesh=new THREE.Mesh(geo,new THREE.MeshLambertMaterial({vertexColors:true,side:THREE.DoubleSide,flatShading:true}));
  scene.add(mesh);
  edges=new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({color:0x0a0f14,transparent:true,opacity:0.22}));
  scene.add(edges);
  const fs=document.getElementById('fieldSelect');
  fs.innerHTML='';
  Object.keys(data.fields).forEach(f=>{const o=document.createElement('option'); o.value=f; o.textContent=f; fs.appendChild(o);});
  fs.value='Failure index (Hashin)';
  recolor(fs.value);
  fitCamera();
}

function fitCamera(){
  if(!mesh) return;
  const box=new THREE.Box3().setFromObject(mesh);
  const center=box.getCenter(new THREE.Vector3());
  const size=box.getSize(new THREE.Vector3()).length();
  controls.target.copy(center);
  camera.position.set(center.x+size*0.55, center.y+size*0.32, center.z+size*0.55);
  camera.near=size*0.001; camera.far=size*6; camera.updateProjectionMatrix();
}

function recolor(field){
  if(!DATA || !mesh) return;
  const vals=DATA.fields[field], colors=mesh.geometry.attributes.color.array;
  let lo=Infinity, hi=-Infinity;
  for(const v of vals){ if(v<lo)lo=v; if(v>hi)hi=v; }
  if(hi-lo<1e-12) hi=lo+1;
  for(let e=0;e<DATA.elems.length;e++){
    const [r,g,b]=jet((vals[e]-lo)/(hi-lo));
    for(let q=0;q<3;q++){ const o=(e*3+q)*3; colors[o]=r; colors[o+1]=g; colors[o+2]=b; }
  }
  mesh.geometry.attributes.color.needsUpdate=true;
  document.getElementById('legendTitle').textContent=field;
  const ticks=document.getElementById('ticks'); ticks.innerHTML='';
  for(let i=6;i>=0;i--){ const el=document.createElement('div'); el.textContent=(lo+(hi-lo)*i/6).toPrecision(3); ticks.appendChild(el); }
}

function fillUi(d){
  document.getElementById('subTitle').textContent=d.mode.replace('_',' ')+' - real solver fields';
  document.getElementById('pressureLabel').textContent=d.geometry.pressure+' MPa';
  document.getElementById('r_mode').textContent=d.mode;
  document.getElementById('r_elems').textContent=d.elems.length;
  document.getElementById('r_fi').textContent=d.summary.fi_max.toFixed(3);
  document.getElementById('r_rf').textContent=d.summary.min_reserve_factor.toFixed(3);
  document.getElementById('r_crit').textContent=d.summary.critical_mode;
  document.getElementById('r_def').textContent=d.summary.max_deformation_mm.toFixed(3)+' mm';
  document.getElementById('r_burst').textContent=d.summary.burst_factor.toFixed(3)+'x';
  document.getElementById('r_mu').textContent=d.summary.mu_max_required==null ? '-' : d.summary.mu_max_required.toFixed(3)+' / '+d.summary.mu_allowable.toFixed(2);
  const decision=document.getElementById('decision');
  decision.textContent=d.gate.decision.toUpperCase();
  decision.classList.toggle('ok', d.gate.release_ready);
  document.getElementById('treeGate').textContent=d.gate.decision;
  document.getElementById('gateStatus').textContent='Gate: '+d.gate.decision;
  document.getElementById('meshStatus').innerHTML=d.elems.length+' elements - '+d.nodes.length+' nodes';
  const bl=document.getElementById('blockers'); bl.innerHTML='';
  for(const b of d.gate.blockers){ const li=document.createElement('li'); li.textContent=b; bl.appendChild(li); }
}
</script>
</body>
</html>"""


def _float_mapping(data: dict[str, Any], defaults: dict[str, float]) -> dict[str, float]:
    return {key: float(data.get(key, default)) for key, default in defaults.items()}


def _solve(payload: dict[str, Any]) -> dict[str, Any]:
    geom_data = _float_mapping(
        payload.get("geometry", {}),
        {
            "outer_radius": 100.0,
            "cylinder_length": 220.0,
            "thickness": 8.0,
            "opening_radius": 10.0,
            "dome_height_ratio": 0.7,
            "pressure": 6.85,
        },
    )
    winding = _float_mapping(payload.get("winding", {}), {"angle_deg": 42.0, "band_mm": 8.0})
    allow = _float_mapping(
        payload.get("allowables", {}),
        {"xt": 2200.0, "xc": 1400.0, "yt": 70.0, "yc": 220.0, "s": 120.0},
    )

    geom = GeometryConfig(**geom_data)
    failure = FailureConfig(
        allowables=MaterialAllowables(**allow),
        margin_of_safety=1.0,
    )
    material = MaterialConfig()
    mode = str(payload.get("mode", "screen"))
    if mode == "optimize":
        result = full_optimize(geom, material, failure_cfg=failure)
    else:
        result = fast_screen(geom, material, winding["angle_deg"], winding["band_mm"], failure_cfg=failure)

    fields = {name: np.asarray(values, dtype=np.float64) for name, values in result.fields.items()}
    if result.disp_node is not None:
        disp_node = np.asarray(result.disp_node, dtype=np.float64)
        fields["Total deformation [mm]"] = disp_node[np.asarray(result.elems, dtype=np.int64)].mean(axis=1)

    response = {
        "mode": result.mode,
        "geometry": geom_data,
        "nodes": np.asarray(result.nodes, dtype=np.float64).round(5).tolist(),
        "elems": np.asarray(result.elems, dtype=np.int64).tolist(),
        "fields": {name: values.round(6).tolist() for name, values in fields.items()},
        "summary": {
            "fi_max": float(result.fi_max),
            "min_reserve_factor": float(result.margins.get("min_reserve_factor", result.burst_factor)),
            "critical_mode": str(result.margins.get("critical_mode", "unknown")),
            "max_deformation_mm": float(result.margins.get("max_deformation_mm", result.disp_max)),
            "burst_factor": float(result.burst_factor),
            "mu_max_required": None if result.mu_max_required is None else float(result.mu_max_required),
            "mu_allowable": float(result.mu_allowable),
            "mass_metric": float(result.mass_metric),
        },
        "gate": result.gate,
    }
    return response


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "COPVStudioUnified/0.1"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/health":
            self._send(200, b'{"ok":true}', "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/solve":
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            result = _solve(payload)
            body = json.dumps(result).encode("utf-8")
            self._send(200, body, "application/json")
        except Exception as exc:  # pragma: no cover - surfaced to browser
            traceback.print_exc()
            body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode("utf-8")
            self._send(500, body, "application/json")

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unified COPV Studio live solver UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), StudioHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"COPV Studio unified live tool running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping COPV Studio unified.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
