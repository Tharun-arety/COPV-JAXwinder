from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.config import GeometryConfig
from copv_opt.course_planner import DiscreteCoursePlanningConfig
from copv_opt.production import production_line_config_from_mapping
from copv_opt.production_pipeline import export_full_production_phase_pipeline, run_full_production_phase_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full production phase pipeline for a given Blackwave line-config JSON.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "blackwave_public_line_config_template.json"),
        help="Path to the line-config JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "blackwave_target_execution"),
        help="Directory where the phase outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = ROOT / "outputs"
    target_dir = Path(args.output_dir)
    if not target_dir.is_absolute():
        target_dir = ROOT / target_dir
    summary = json.loads((outputs_dir / "winding_first_summary.json").read_text(encoding="utf-8"))
    layout = json.loads((outputs_dir / "winding_first_layout.json").read_text(encoding="utf-8"))
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    line_payload = json.loads(config_path.read_text(encoding="utf-8"))

    geom = GeometryConfig(**summary["geometry"])
    planning = DiscreteCoursePlanningConfig()
    line = production_line_config_from_mapping(line_payload)
    pipeline = run_full_production_phase_pipeline(
        layout=layout,
        summary=summary,
        geom=geom,
        line_config=line,
        planning_config=planning,
        artifact_output_dir=target_dir,
    )
    exported = export_full_production_phase_pipeline(target_dir, pipeline)

    print(f"Wrote {exported['index']}")
    print(f"Wrote {exported['program_snapshot']}")
    print(f"Wrote {exported['discrete_plan_snapshot']}")
    for phase_id, artifacts in exported["phase_artifacts"].items():
        print(f"{phase_id}: {artifacts['json']}")
        print(f"{phase_id}: {artifacts['md']}")


if __name__ == "__main__":
    main()
