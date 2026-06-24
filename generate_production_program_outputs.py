from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.config import GeometryConfig
from copv_opt.course_planner import (
    DiscreteCoursePlanningConfig,
    build_discrete_winding_plan_from_layout,
    export_discrete_winding_plan,
    save_discrete_winding_plan_markdown,
)
from copv_opt.production import (
    ProductionLineConfig,
    build_production_program_from_layout,
    export_production_program,
    save_production_readiness_markdown,
)


def main() -> None:
    outputs_dir = ROOT / "outputs"
    summary = json.loads((outputs_dir / "winding_first_summary.json").read_text(encoding="utf-8"))
    layout = json.loads((outputs_dir / "winding_first_layout.json").read_text(encoding="utf-8"))

    geom = GeometryConfig(**summary["geometry"])
    line = ProductionLineConfig()
    planning = DiscreteCoursePlanningConfig()
    program = build_production_program_from_layout(
        layout=layout,
        geom=geom,
        line_config=line,
        summary=summary,
        source_label="outputs/winding_first_layout.json",
        include_path_points=True,
        planning_config=planning,
    )
    discrete_plan = build_discrete_winding_plan_from_layout(layout=layout, geom=geom, config=planning)

    program_path = export_production_program(outputs_dir / "winding_first_production_program.json", program)
    report_path = save_production_readiness_markdown(outputs_dir / "winding_first_production_readiness.md", program)
    discrete_plan_path = export_discrete_winding_plan(outputs_dir / "winding_first_discrete_course_plan.json", discrete_plan)
    discrete_report_path = save_discrete_winding_plan_markdown(
        outputs_dir / "winding_first_discrete_course_plan.md",
        discrete_plan,
    )

    print(f"Wrote {program_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {discrete_plan_path}")
    print(f"Wrote {discrete_report_path}")


if __name__ == "__main__":
    main()
