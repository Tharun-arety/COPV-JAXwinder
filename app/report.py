"""Phase 6 — design report.

Renders a self-contained HTML report for a designed tank: the requirement, the
geometry it sized to, the structural screen, the release gate with its blockers, and
(when present) the discrete course-plan summary. No external dependencies — the
report is a single static file suitable for emailing or archiving with a project.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _row(label: str, value: Any) -> str:
    return f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"


def _table(rows: list[str]) -> str:
    return "<table>" + "".join(rows) + "</table>"


def render_html(
    project_name: str,
    result,
    sizing,
    course_summary: dict[str, Any] | None = None,
    kinematics: dict[str, Any] | None = None,
) -> str:
    g = result.gate
    decision = g["decision"].upper()
    decision_color = "#b54708" if not g["release_ready"] else "#067647"

    req_rows = [
        _row("Cylinder length", f"{sizing.cylinder_length_mm:.1f} mm"),
        _row("Inner radius", f"{sizing.inner_radius_mm:.1f} mm"),
        _row("Design pressure", f"{sizing.design_pressure_mpa:.3f} MPa"),
        _row("Achieved volume", f"{sizing.achieved_volume_litres:.2f} L"),
        _row("Slenderness L/D", f"{sizing.slenderness_l_over_d:.2f}"),
    ]
    screen_rows = [
        _row("Analysis mode", result.mode.replace("_", " ")),
        _row("FI max (Hashin)", f"{result.fi_max:.3f}"),
        _row("Burst factor", f"{result.burst_factor:.3f}  (pressure multiple to FI=1)"),
        _row("Max displacement", f"{result.disp_max:.3f} mm"),
        _row("Mass metric", f"{result.mass_metric:.4g}"),
    ]
    if result.mu_max_required is not None:
        screen_rows.append(_row("Friction demand μ", f"{result.mu_max_required:.3f} (allowable {result.mu_allowable:.2f})"))
    if result.angle_deg is not None:
        screen_rows.append(_row("Constant winding angle", f"{result.angle_deg:.1f}°"))

    blockers = "".join(f"<li>{html.escape(b)}</li>" for b in g["blockers"])

    course_block = ""
    if course_summary:
        course_rows = [
            _row("Helical course pairs", course_summary.get("total_course_pairs", "—")),
            _row("Hoop rings", course_summary.get("total_hoop_rings", "—")),
            _row("Cut/restart events", course_summary.get("total_cut_restart_events", "—")),
        ]
        course_block = f"<h2>Discrete course plan</h2>{_table(course_rows)}"

    kin_block = ""
    if kinematics:
        kin_rows = [
            _row("Peak mandrel rpm", f"{kinematics.get('peak_mandrel_rpm', 0):.2f}"),
            _row("Overall length", f"{kinematics.get('overall_length_mm', 0):.1f} mm"),
            _row("Kinematic screen", kinematics.get("screen_status", "—")),
        ]
        kin_block = f"<h2>Machine kinematic demand</h2>{_table(kin_rows)}"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>COPV design report — {html.escape(project_name)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; color: #1a1a1a; padding: 0 1rem; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.2rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 1.8rem; border-bottom: 1px solid #e3e3e3; padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; }}
  th {{ width: 45%; color: #555; font-weight: 600; }}
  .gate {{ display: inline-block; padding: 0.3rem 0.8rem; border-radius: 6px; color: white;
           background: {decision_color}; font-weight: 700; margin-top: 0.5rem; }}
  ul {{ margin-top: 0.6rem; }}
  footer {{ margin-top: 2rem; color: #888; font-size: 0.8rem; }}
</style></head><body>
<h1>COPV design report</h1>
<div style="color:#666">Project: <strong>{html.escape(project_name)}</strong></div>
<div class="gate">{decision}</div>

<h2>Geometry from requirement</h2>{_table(req_rows)}
<h2>Structural screen</h2>{_table(screen_rows)}
{course_block}
{kin_block}
<h2>Release gate</h2>
<p>Release ready: <strong>{g['release_ready']}</strong> · Hashin ok: {g['hashin_ok']} · Friction ok: {g['friction_ok']}</p>
<p>Outstanding blockers:</p>
<ul>{blockers}</ul>

<footer>Screening result. Not a certified analysis. Verify with Ansys ACP + Mechanical
and real qualification data before production use.</footer>
</body></html>"""


def write_html(
    path: str | Path,
    project_name: str,
    result,
    sizing,
    course_summary: dict[str, Any] | None = None,
    kinematics: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_html(project_name, result, sizing, course_summary, kinematics),
        encoding="utf-8",
    )
    return path
