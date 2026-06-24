from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.config import GeometryConfig
from copv_opt.course_planner import DiscreteCoursePlanningConfig
from copv_opt.production import ProductionLineConfig
from copv_opt.production_pipeline import export_full_production_phase_pipeline, run_full_production_phase_pipeline


def main() -> None:
    outputs_dir = ROOT / "outputs"
    phase_dir = outputs_dir / "production_phase_execution"
    summary = json.loads((outputs_dir / "winding_first_summary.json").read_text(encoding="utf-8"))
    layout = json.loads((outputs_dir / "winding_first_layout.json").read_text(encoding="utf-8"))

    geom = GeometryConfig(**summary["geometry"])
    line = ProductionLineConfig()
    planning = DiscreteCoursePlanningConfig()
    pipeline = run_full_production_phase_pipeline(
        layout=layout,
        summary=summary,
        geom=geom,
        line_config=line,
        planning_config=planning,
        artifact_output_dir=phase_dir,
    )
    exported = export_full_production_phase_pipeline(phase_dir, pipeline)

    print(f"Wrote {exported['index']}")
    print(f"Wrote {exported['program_snapshot']}")
    print(f"Wrote {exported['discrete_plan_snapshot']}")
    for phase_id, artifacts in exported["phase_artifacts"].items():
        print(f"{phase_id}: {artifacts['json']}")
        print(f"{phase_id}: {artifacts['md']}")


if __name__ == "__main__":
    main()
