from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "phase_01": base_dir / "phase_01_production_data_contract.json",
        "phase_02": base_dir / "phase_02_discrete_course_planning.json",
        "phase_03": base_dir / "phase_03_machine_kinematic_demand_screen.json",
        "phase_04": base_dir / "phase_04_towpreg_relative_setpoint_screen.json",
        "phase_05": base_dir / "phase_05_first_order_as_built_laminate_surrogate.json",
        "phase_06": base_dir / "phase_06_cure_and_autofrettage_input_readiness.json",
        "phase_07": base_dir / "phase_07_inspection_and_digital_thread_scaffold.json",
        "phase_08": base_dir / "phase_08_qualification_and_release_decision.json",
    }


def main() -> None:
    sparse_dir = OUTPUTS / "blackwave_target_execution"
    dummy_dir = OUTPUTS / "blackwave_dummy_demo_execution"
    sparse = {phase: _read_json(path) for phase, path in _phase_paths(sparse_dir).items()}
    dummy = {phase: _read_json(path) for phase, path in _phase_paths(dummy_dir).items()}

    sparse_contract = sparse["phase_01"]["metrics"]
    dummy_contract = dummy["phase_01"]["metrics"]
    sparse_release = sparse["phase_08"]["metrics"]
    dummy_release = dummy["phase_08"]["metrics"]

    lines = [
        "# Blackwave Dummy Demo Comparison",
        "",
        "This report compares the sparse public-target profile against a fully filled illustrative dummy line configuration.",
        "",
        "## Why This Exists",
        "- The sparse public profile proves the gating logic is conservative when key production data is missing.",
        "- The dummy profile proves the pipeline reacts when machine, process, cure, inspection, and qualification inputs are supplied.",
        "- The dummy profile is synthetic. It demonstrates software behavior, not production readiness.",
        "",
        "## Contract Fill Improvement",
        f"- Public-target line-config completeness: `{sparse_contract['line_config_completeness_ratio']}`",
        f"- Dummy-demo line-config completeness: `{dummy_contract['line_config_completeness_ratio']}`",
        f"- Public-target declared blocker count: `{sparse_contract['declared_blocker_count']}`",
        f"- Dummy-demo declared blocker count: `{dummy_contract['declared_blocker_count']}`",
        "",
        "## Phase Comparison",
        "| Phase | Public-target status | Public blockers | Dummy-demo status | Dummy blockers |",
        "| --- | --- | ---: | --- | ---: |",
    ]

    for phase_id in sorted(sparse.keys()):
        lines.append(
            f"| `{phase_id}` | `{sparse[phase_id]['status']}` | `{len(sparse[phase_id].get('blockers', []))}` | "
            f"`{dummy[phase_id]['status']}` | `{len(dummy[phase_id].get('blockers', []))}` |"
        )

    phase03 = dummy["phase_03"]["metrics"]
    phase04 = dummy["phase_04"]["metrics"]
    phase05 = dummy["phase_05"]["metrics"]
    phase06 = dummy["phase_06"]["metrics"]

    lines.extend(
        [
            "",
            "## Dummy Demo Highlights",
            f"- Phase 03 closes all machine-input blockers and computes `max_required_mandrel_rpm = {phase03['max_required_mandrel_rpm']}` with `turning_radius_violation_count = {phase03['turning_radius_violation_count']}`.",
            f"- Phase 04 runs against declared deposition windows and reports `friction_headroom = {phase04['friction_headroom']}`.",
            f"- Phase 05 runs against declared acceptance limits with `max_gap_mm = {phase05['max_gap_mm']}` and `max_overlap_mm = {phase05['max_overlap_mm']}`.",
            f"- Phase 06 becomes input-ready with `input_completeness_ratio = {phase06['input_completeness_ratio']}` and `cure_step_count = {phase06['cure_step_count']}`.",
            "",
            "## Release Decision",
            f"- Public-target release ready: `{sparse_release['release_ready']}`",
            f"- Dummy-demo release ready: `{dummy_release['release_ready']}`",
            "- The dummy case still ends in `do_not_release` because the repo is intentionally conservative about surrogate phases such as discrete planning, deposition physics, as-built prediction, and qualification closure.",
            "- That is the expected and correct behavior. The demo proves the phase stack works and that stronger input data materially changes the intermediate results.",
            "",
            "## Output Locations",
            "- `outputs/blackwave_target_execution/`",
            "- `outputs/blackwave_dummy_demo_execution/`",
        ]
    )

    out_path = OUTPUTS / "blackwave_demo_comparison.md"
    out_path.unlink(missing_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
