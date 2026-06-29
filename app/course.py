"""Phase 3 — winding course plan, kinematic demand, neutral NC export.

Real:
* ``course_plan`` discretizes the optimized continuous winding field into helical
  course pairs and hoop rings (engine's course planner).
* ``kinematic_demand`` derives first-order machine demand (mandrel revolutions,
  peak rpm at a given band speed, helical/hoop counts) and screens it against
  supplied :class:`MachineLimits`.
* ``export_nc_csv`` writes a machine-neutral operation sequence.

Honest gap:
* A machine-specific post-processor (Roth/Mikrosam/etc. NC dialect) and true
  collision/payout-eye kinematics need the actual machine definition. ``MachineLimits``
  fields default to ``None`` and the screen reports "limit not supplied" rather than
  inventing a bound.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from copv_opt.config import GeometryConfig
from copv_opt.course_planner import build_discrete_winding_plan_from_layout


def course_plan(layout: dict[str, Any], geom: GeometryConfig) -> dict[str, Any]:
    """Discrete course plan from a winding process layout. Requires a full_optimize
    result's layout (constant-angle fast screens carry none)."""
    if not layout:
        raise ValueError("course_plan needs a winding layout — run full_optimize first")
    return build_discrete_winding_plan_from_layout(layout, geom)


@dataclass
class MachineLimits:
    """Winding-machine envelope. None = not supplied (no bound applied)."""

    max_mandrel_rpm: float | None = None
    max_band_speed_mm_s: float | None = None
    min_turning_radius_mm: float | None = None
    max_carriage_travel_mm: float | None = None


def kinematic_demand(
    plan: dict[str, Any],
    geom: GeometryConfig,
    band_speed_mm_s: float = 500.0,
    limits: MachineLimits | None = None,
) -> dict[str, Any]:
    """First-order kinematic demand + limit screen.

    band_speed_mm_s is the tangential laydown speed; peak rpm follows from the
    cylinder circumference. Counts come from the discrete plan."""
    limits = limits or MachineLimits()
    metrics = dict(plan.get("metrics", {}))
    helical_pairs = int(metrics.get("total_course_pairs", 0))
    hoop_rings = int(metrics.get("total_hoop_rings", 0))
    cut_restarts = int(metrics.get("total_cut_restart_events", 0))

    circumference = 2.0 * math.pi * geom.mid_radius           # mm
    peak_rpm = band_speed_mm_s / circumference * 60.0          # rev/min on the cylinder
    overall_length = geom.cylinder_length + 2.0 * geom.dome_height_ratio * geom.inner_radius

    checks: list[dict[str, Any]] = []

    def _check(name: str, demand: float, limit: float | None, unit: str) -> None:
        if limit is None:
            checks.append({"name": name, "demand": demand, "limit": None, "unit": unit, "status": "limit_not_supplied"})
        else:
            checks.append(
                {
                    "name": name,
                    "demand": demand,
                    "limit": limit,
                    "unit": unit,
                    "status": "ok" if demand <= limit + 1e-9 else "exceeds_limit",
                }
            )

    _check("peak_mandrel_rpm", peak_rpm, limits.max_mandrel_rpm, "rev/min")
    _check("band_speed", band_speed_mm_s, limits.max_band_speed_mm_s, "mm/s")
    _check("turning_radius", geom.opening_radius, None if limits.min_turning_radius_mm is None else limits.min_turning_radius_mm, "mm")
    _check("carriage_travel", overall_length, limits.max_carriage_travel_mm, "mm")

    any_violation = any(c["status"] == "exceeds_limit" for c in checks)
    any_unsupplied = any(c["status"] == "limit_not_supplied" for c in checks)
    return {
        "helical_course_pairs": helical_pairs,
        "hoop_rings": hoop_rings,
        "cut_restart_events": cut_restarts,
        "cylinder_circumference_mm": circumference,
        "peak_mandrel_rpm": peak_rpm,
        "overall_length_mm": overall_length,
        "band_speed_mm_s": band_speed_mm_s,
        "checks": checks,
        "screen_status": "violations" if any_violation else ("incomplete_limits" if any_unsupplied else "within_limits"),
    }


def export_nc_csv(plan: dict[str, Any], path: str | Path) -> Path:
    """Write a machine-neutral operation sequence as CSV.

    Columns: step, operation, family/ring index, count, note. This is an exchange
    format, not a machine post — a real controller needs its own dialect."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sequence = plan.get("execution_sequence", [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "operation", "index", "detail"])
        if sequence:
            for i, op in enumerate(sequence, start=1):
                if isinstance(op, dict):
                    writer.writerow([i, op.get("type", "op"), op.get("index", ""), op.get("note", "")])
                else:
                    writer.writerow([i, str(op), "", ""])
        else:
            # Fall back to summarizing pairs and rings if no explicit sequence.
            metrics = dict(plan.get("metrics", {}))
            writer.writerow([1, "helical_course_pairs", "", metrics.get("total_course_pairs", 0)])
            writer.writerow([2, "hoop_rings", "", metrics.get("total_hoop_rings", 0)])
    return path
