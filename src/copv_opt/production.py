from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np

from .config import GeometryConfig, HybridConfig
from .course_planner import DiscreteCoursePlanningConfig, build_discrete_winding_plan_from_layout


@dataclass
class TowpregMaterialConfig:
    material_name: str = "UNSPECIFIED_TOWPREG"
    tow_width_mm: float = 12.0
    tow_thickness_mm: float = 0.3
    storage_temperature_c: float | None = None
    max_out_time_h: float | None = None
    deposition_temperature_window_c: tuple[float, float] | None = None
    allowable_tension_window_n: tuple[float, float] | None = None
    cure_kinetics_reference: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class WindingMachineConfig:
    machine_name: str = "UNSPECIFIED_WINDER"
    axis_count: int = 4
    controller_family: str | None = None
    nominal_head_speed_mm_s: float | None = None
    max_head_speed_mm_s: float | None = None
    max_mandrel_rpm: float | None = None
    min_turning_radius_mm: float | None = None
    min_restart_spacing_mm: float | None = None
    supports_tow_cut_restart: bool = False
    supports_closed_loop_tension_control: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class HeatingCompactionConfig:
    heater_type: str = "UNSPECIFIED"
    target_heater_setpoint_c: float | None = None
    max_heater_setpoint_c: float | None = None
    target_compaction_force_n: float | None = None
    compaction_roller_width_mm: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class CureCycleStep:
    name: str
    target_temp_c: float | None = None
    ramp_rate_c_per_min: float | None = None
    hold_time_min: float | None = None


@dataclass
class CureCycleConfig:
    cycle_name: str = "UNSPECIFIED_CURE"
    steps: list[CureCycleStep] = field(default_factory=list)
    post_cure_required: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class AutofrettageConfig:
    target_pressure: float | None = None
    hold_time_min: float | None = None
    liner_yield_pressure: float | None = None
    residual_stress_model: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class InspectionConfig:
    inline_sensor_type: str = "UNSPECIFIED"
    max_gap_mm: float | None = None
    max_overlap_mm: float | None = None
    max_wrinkle_height_mm: float | None = None
    max_void_fraction_pct: float | None = None
    final_ndi_method: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class QualificationConfig:
    coupon_dataset_path: str | None = None
    subcomponent_dataset_path: str | None = None
    vessel_dataset_path: str | None = None
    required_coupon_tests: int | None = None
    required_subcomponent_tests: int | None = None
    required_vessel_tests: int | None = None
    accepted_coupon_tests: int = 0
    accepted_subcomponent_tests: int = 0
    accepted_vessel_tests: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class ProductionLineConfig:
    line_name: str = "UNSPECIFIED_COPV_LINE"
    qualification_standard: str | None = None
    material: TowpregMaterialConfig = field(default_factory=TowpregMaterialConfig)
    machine: WindingMachineConfig = field(default_factory=WindingMachineConfig)
    heating_compaction: HeatingCompactionConfig = field(default_factory=HeatingCompactionConfig)
    cure: CureCycleConfig = field(default_factory=CureCycleConfig)
    autofrettage: AutofrettageConfig = field(default_factory=AutofrettageConfig)
    inspection: InspectionConfig = field(default_factory=InspectionConfig)
    qualification: QualificationConfig = field(default_factory=QualificationConfig)
    notes: list[str] = field(default_factory=list)


def _meridional_metrics(radius: float, cylinder_length: float, opening_radius: float) -> tuple[float, float, float]:
    theta_open = float(np.arcsin(np.clip(opening_radius / radius, 0.0, 0.999999)))
    cap_len = radius * (np.pi / 2.0 - theta_open)
    total_len = 2.0 * cap_len + cylinder_length
    return theta_open, cap_len, total_len


def _optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return None
    return arr


def _profile_mean(sample_s: np.ndarray, profile: np.ndarray | None) -> float:
    if profile is None or profile.size == 0 or sample_s.size == 0:
        return 0.0
    span = max(float(sample_s[-1] - sample_s[0]), 1e-9)
    return float(np.trapezoid(profile, sample_s) / span)


def _path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _optional_pair_of_floats(value: Any, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    values = list(value)
    if len(values) != 2:
        raise ValueError(f"`{field_name}` must contain exactly two numeric values.")
    return float(values[0]), float(values[1])


def production_line_config_to_dict(line_config: ProductionLineConfig) -> dict[str, Any]:
    return _to_serializable(line_config)


def production_line_config_from_mapping(data: dict[str, Any] | None) -> ProductionLineConfig:
    payload = {} if data is None else dict(data)
    material = dict(payload.get("material", {}))
    machine = dict(payload.get("machine", {}))
    head = dict(payload.get("heating_compaction", {}))
    cure = dict(payload.get("cure", {}))
    autofrettage = dict(payload.get("autofrettage", {}))
    inspection = dict(payload.get("inspection", {}))
    qualification = dict(payload.get("qualification", {}))

    cure_steps = [
        CureCycleStep(
            name=str(step.get("name", f"STEP_{idx + 1:02d}")),
            target_temp_c=None if step.get("target_temp_c") is None else float(step["target_temp_c"]),
            ramp_rate_c_per_min=None
            if step.get("ramp_rate_c_per_min") is None
            else float(step["ramp_rate_c_per_min"]),
            hold_time_min=None if step.get("hold_time_min") is None else float(step["hold_time_min"]),
        )
        for idx, step in enumerate(cure.get("steps", []) or [])
    ]

    return ProductionLineConfig(
        line_name=str(payload.get("line_name", "UNSPECIFIED_COPV_LINE")),
        qualification_standard=payload.get("qualification_standard"),
        material=TowpregMaterialConfig(
            material_name=str(material.get("material_name", "UNSPECIFIED_TOWPREG")),
            tow_width_mm=float(material.get("tow_width_mm", 12.0)),
            tow_thickness_mm=float(material.get("tow_thickness_mm", 0.3)),
            storage_temperature_c=None
            if material.get("storage_temperature_c") is None
            else float(material["storage_temperature_c"]),
            max_out_time_h=None if material.get("max_out_time_h") is None else float(material["max_out_time_h"]),
            deposition_temperature_window_c=_optional_pair_of_floats(
                material.get("deposition_temperature_window_c"),
                "material.deposition_temperature_window_c",
            ),
            allowable_tension_window_n=_optional_pair_of_floats(
                material.get("allowable_tension_window_n"),
                "material.allowable_tension_window_n",
            ),
            cure_kinetics_reference=material.get("cure_kinetics_reference"),
            notes=_string_list(material.get("notes")),
        ),
        machine=WindingMachineConfig(
            machine_name=str(machine.get("machine_name", "UNSPECIFIED_WINDER")),
            axis_count=int(machine.get("axis_count", 4)),
            controller_family=machine.get("controller_family"),
            nominal_head_speed_mm_s=None
            if machine.get("nominal_head_speed_mm_s") is None
            else float(machine["nominal_head_speed_mm_s"]),
            max_head_speed_mm_s=None
            if machine.get("max_head_speed_mm_s") is None
            else float(machine["max_head_speed_mm_s"]),
            max_mandrel_rpm=None if machine.get("max_mandrel_rpm") is None else float(machine["max_mandrel_rpm"]),
            min_turning_radius_mm=None
            if machine.get("min_turning_radius_mm") is None
            else float(machine["min_turning_radius_mm"]),
            min_restart_spacing_mm=None
            if machine.get("min_restart_spacing_mm") is None
            else float(machine["min_restart_spacing_mm"]),
            supports_tow_cut_restart=bool(machine.get("supports_tow_cut_restart", False)),
            supports_closed_loop_tension_control=bool(machine.get("supports_closed_loop_tension_control", False)),
            notes=_string_list(machine.get("notes")),
        ),
        heating_compaction=HeatingCompactionConfig(
            heater_type=str(head.get("heater_type", "UNSPECIFIED")),
            target_heater_setpoint_c=None
            if head.get("target_heater_setpoint_c") is None
            else float(head["target_heater_setpoint_c"]),
            max_heater_setpoint_c=None
            if head.get("max_heater_setpoint_c") is None
            else float(head["max_heater_setpoint_c"]),
            target_compaction_force_n=None
            if head.get("target_compaction_force_n") is None
            else float(head["target_compaction_force_n"]),
            compaction_roller_width_mm=None
            if head.get("compaction_roller_width_mm") is None
            else float(head["compaction_roller_width_mm"]),
            notes=_string_list(head.get("notes")),
        ),
        cure=CureCycleConfig(
            cycle_name=str(cure.get("cycle_name", "UNSPECIFIED_CURE")),
            steps=cure_steps,
            post_cure_required=bool(cure.get("post_cure_required", False)),
            notes=_string_list(cure.get("notes")),
        ),
        autofrettage=AutofrettageConfig(
            target_pressure=None
            if autofrettage.get("target_pressure") is None
            else float(autofrettage["target_pressure"]),
            hold_time_min=None
            if autofrettage.get("hold_time_min") is None
            else float(autofrettage["hold_time_min"]),
            liner_yield_pressure=None
            if autofrettage.get("liner_yield_pressure") is None
            else float(autofrettage["liner_yield_pressure"]),
            residual_stress_model=autofrettage.get("residual_stress_model"),
            notes=_string_list(autofrettage.get("notes")),
        ),
        inspection=InspectionConfig(
            inline_sensor_type=str(inspection.get("inline_sensor_type", "UNSPECIFIED")),
            max_gap_mm=None if inspection.get("max_gap_mm") is None else float(inspection["max_gap_mm"]),
            max_overlap_mm=None
            if inspection.get("max_overlap_mm") is None
            else float(inspection["max_overlap_mm"]),
            max_wrinkle_height_mm=None
            if inspection.get("max_wrinkle_height_mm") is None
            else float(inspection["max_wrinkle_height_mm"]),
            max_void_fraction_pct=None
            if inspection.get("max_void_fraction_pct") is None
            else float(inspection["max_void_fraction_pct"]),
            final_ndi_method=inspection.get("final_ndi_method"),
            notes=_string_list(inspection.get("notes")),
        ),
        qualification=QualificationConfig(
            coupon_dataset_path=qualification.get("coupon_dataset_path"),
            subcomponent_dataset_path=qualification.get("subcomponent_dataset_path"),
            vessel_dataset_path=qualification.get("vessel_dataset_path"),
            required_coupon_tests=None
            if qualification.get("required_coupon_tests") is None
            else int(qualification["required_coupon_tests"]),
            required_subcomponent_tests=None
            if qualification.get("required_subcomponent_tests") is None
            else int(qualification["required_subcomponent_tests"]),
            required_vessel_tests=None
            if qualification.get("required_vessel_tests") is None
            else int(qualification["required_vessel_tests"]),
            accepted_coupon_tests=int(qualification.get("accepted_coupon_tests", 0)),
            accepted_subcomponent_tests=int(qualification.get("accepted_subcomponent_tests", 0)),
            accepted_vessel_tests=int(qualification.get("accepted_vessel_tests", 0)),
            notes=_string_list(qualification.get("notes")),
        ),
        notes=_string_list(payload.get("notes")),
    )


def build_production_gap_report(line_config: ProductionLineConfig) -> list[str]:
    blockers: list[str] = []

    material = line_config.material
    machine = line_config.machine
    head = line_config.heating_compaction
    cure = line_config.cure
    autofrettage = line_config.autofrettage
    inspection = line_config.inspection
    qualification = line_config.qualification

    if material.storage_temperature_c is None:
        blockers.append("Populate the towpreg storage temperature limit.")
    if material.max_out_time_h is None:
        blockers.append("Populate the towpreg maximum out-time limit.")
    if material.deposition_temperature_window_c is None:
        blockers.append("Populate the validated deposition temperature window for the towpreg.")
    if material.allowable_tension_window_n is None:
        blockers.append("Populate the validated tow tension window for the selected material system.")
    if machine.nominal_head_speed_mm_s is None:
        blockers.append("Populate the nominal deposition speed for the target winding machine.")
    if machine.max_head_speed_mm_s is None:
        blockers.append("Populate the machine maximum head speed.")
    if machine.max_mandrel_rpm is None:
        blockers.append("Populate the machine maximum mandrel RPM.")
    if machine.min_turning_radius_mm is None:
        blockers.append("Populate the minimum steerable turning radius for the line.")
    if head.target_heater_setpoint_c is None:
        blockers.append("Populate the nominal heater setpoint for deposition.")
    if head.target_compaction_force_n is None:
        blockers.append("Populate the nominal compaction force for the head.")
    if not cure.steps:
        blockers.append("Populate the cure cycle definition, including ramp and hold steps.")
    if autofrettage.target_pressure is None:
        blockers.append("Populate the autofrettage target pressure and hold definition.")
    if autofrettage.liner_yield_pressure is None:
        blockers.append("Populate the liner yield pressure or equivalent autofrettage qualification limit.")
    if inspection.max_gap_mm is None:
        blockers.append("Populate the allowable gap threshold from the quality plan.")
    if inspection.max_overlap_mm is None:
        blockers.append("Populate the allowable overlap threshold from the quality plan.")
    if inspection.max_wrinkle_height_mm is None:
        blockers.append("Populate the allowable wrinkle threshold from the quality plan.")
    if inspection.final_ndi_method is None:
        blockers.append("Populate the final NDI method required for release.")
    if qualification.coupon_dataset_path is None:
        blockers.append("Populate the coupon qualification dataset path or evidence reference.")
    if qualification.subcomponent_dataset_path is None:
        blockers.append("Populate the subcomponent qualification dataset path or evidence reference.")
    if qualification.vessel_dataset_path is None:
        blockers.append("Populate the vessel-level qualification dataset path or evidence reference.")

    blockers.extend(
        [
            "Replace continuous pass-density optimization variables with discrete course scheduling variables, and score cuts/restarts directly inside the optimizer rather than only in downstream planning.",
            "Add machine-axis inverse kinematics and NC/post-processor output for the target line.",
            "Add an as-built thickness/defect model so structural analysis runs on predicted manufactured state.",
            "Couple cure, residual stress, and autofrettage into the structural workflow before release.",
            "Correlate the model against coupon, subcomponent, and vessel qualification data.",
        ]
    )
    return blockers


def _screening_snapshot(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {}
    winding = summary.get("winding", {})
    return {
        "case": summary.get("case"),
        "pressure": summary.get("geometry", {}).get("pressure"),
        "fi_max": winding.get("fi_max"),
        "fi_max_with_margin": winding.get("fi_max_with_margin"),
        "mu_max_required": winding.get("friction_mu_max_required"),
        "mu_allowable": winding.get("friction_mu_allowable"),
        "mass_delta_percent_vs_baseline": winding.get("mass_delta_percent_vs_baseline"),
        "burst_factor": winding.get("burst_factor"),
        "allowable_pressure_with_margin": winding.get("allowable_pressure_with_margin"),
        "hashin_constraint_satisfied": winding.get("hashin_constraint_satisfied"),
        "friction_constraint_satisfied": winding.get("friction_constraint_satisfied"),
    }


def _minimal_summary_from_result(result: dict[str, Any], geom: GeometryConfig) -> dict[str, Any]:
    def _scalar(name: str) -> float | None:
        if name not in result:
            return None
        return float(np.asarray(result[name]))

    return {
        "case": "optimizer_result",
        "geometry": {
            "pressure": float(geom.pressure),
            "outer_radius": float(geom.outer_radius),
            "cylinder_length": float(geom.cylinder_length),
            "thickness": float(geom.thickness),
            "opening_radius": float(geom.opening_radius),
        },
        "winding": {
            "fi_max": _scalar("fi_max"),
            "fi_max_with_margin": None
            if "failure_with_margin" not in result
            else float(np.max(np.asarray(result["failure_with_margin"]))),
            "friction_mu_max_required": _scalar("mu_max_required"),
            "mass_metric": _scalar("mass_metric"),
            "max_winding_thickness": None
            if "winding_thickness_field" not in result
            else float(np.max(np.asarray(result["winding_thickness_field"]))),
        },
    }


def build_production_program_from_layout(
    layout: dict[str, Any],
    geom: GeometryConfig,
    line_config: ProductionLineConfig | None = None,
    summary: dict[str, Any] | None = None,
    source_label: str | None = None,
    include_path_points: bool = True,
    planning_config: DiscreteCoursePlanningConfig | None = None,
) -> dict[str, Any]:
    line = ProductionLineConfig() if line_config is None else line_config
    planner = DiscreteCoursePlanningConfig() if planning_config is None else planning_config
    planner.emit_path_points = bool(include_path_points)

    control_s = np.asarray(layout.get("control_s", []), dtype=np.float64)
    control_angle_deg = np.asarray(layout.get("control_angle_deg", []), dtype=np.float64)
    control_thickness = _optional_array(layout.get("control_thickness"))
    sample_s = np.asarray(layout.get("sample_s", []), dtype=np.float64)
    angle_profile_deg = np.asarray(layout.get("angle_profile_deg", []), dtype=np.float64)
    thickness_profile = _optional_array(layout.get("thickness_profile"))
    helical_pass_profile = _optional_array(layout.get("helical_pass_profile"))
    hoop_pass_profile = _optional_array(layout.get("hoop_pass_profile"))
    mu_required = _optional_array(layout.get("mu_required"))
    paths = list(layout.get("paths", []))

    _, cap_len, total_len = _meridional_metrics(geom.outer_radius, geom.cylinder_length, geom.opening_radius)
    cylinder_start = float(cap_len)
    cylinder_stop = float(cap_len + geom.cylinder_length)

    mean_helical_pass = _profile_mean(sample_s, helical_pass_profile)
    mean_hoop_pass = _profile_mean(sample_s, hoop_pass_profile)
    peak_helical_pass = 0.0 if helical_pass_profile is None else float(np.max(helical_pass_profile))
    peak_hoop_pass = 0.0 if hoop_pass_profile is None else float(np.max(hoop_pass_profile))
    screening = _screening_snapshot(summary)
    peak_mu_required = (
        float(screening["mu_max_required"])
        if screening.get("mu_max_required") is not None
        else (0.0 if mu_required is None else float(np.max(mu_required)))
    )

    blockers = build_production_gap_report(line)
    discrete_plan = build_discrete_winding_plan_from_layout(layout, geom, planner)
    helical_course_pairs = list(discrete_plan.get("helical_course_pairs", []))
    helical_courses = list(discrete_plan.get("helical_courses", []))
    hoop_rings = list(discrete_plan.get("hoop_rings", []))
    execution_sequence = list(discrete_plan.get("execution_sequence", []))
    execution_sequence.append({"step_type": "cure_cycle", "cycle_name": line.cure.cycle_name})
    execution_sequence.append({"step_type": "autofrettage", "note": "Autofrettage model not yet coupled in-repo."})
    execution_sequence.append({"step_type": "inspection_release", "method": line.inspection.final_ndi_method})

    return {
        "program_type": "copv_towpreg_production_scaffold",
        "program_status": "planning_scaffold_only",
        "source": {
            "label": source_label or "layout",
            "layout_type": layout.get("layout_type", "unknown"),
        },
        "geometry": {
            "outer_radius_mm": float(geom.outer_radius),
            "cylinder_length_mm": float(geom.cylinder_length),
            "wall_thickness_mm": float(geom.thickness),
            "opening_radius_mm": float(geom.opening_radius),
            "pressure": float(geom.pressure),
            "total_meridional_length_mm": float(total_len),
            "cylinder_start_s_mm": cylinder_start,
            "cylinder_stop_s_mm": cylinder_stop,
        },
        "screening_snapshot": screening,
        "line_config": _to_serializable(line),
        "planning_assumptions": [
            "This artifact converts the staged winding result into a production-program scaffold, not final NC/G-code.",
            "Continuous pass-density controls are translated into a deterministic discrete course plan, but the optimizer itself still solves continuous screening fields.",
            "Machine-axis inverse kinematics, cure-induced residual stress, and autofrettage coupling remain open work packages.",
        ],
        "process_basis": {
            "family_count": int(layout.get("family_count", len(paths))),
            "control_s_mm": control_s,
            "control_angle_deg": control_angle_deg,
            "control_thickness_mm": control_thickness,
            "mean_helical_pass_count": mean_helical_pass,
            "peak_helical_pass_count": peak_helical_pass,
            "mean_hoop_pass_count": mean_hoop_pass,
            "peak_hoop_pass_count": peak_hoop_pass,
            "peak_required_friction_coefficient": peak_mu_required,
        },
        "layout_profiles": {
            "sample_s_mm": sample_s,
            "angle_profile_deg": angle_profile_deg,
            "thickness_profile_mm": thickness_profile,
            "helical_pass_profile": helical_pass_profile,
            "hoop_pass_profile": hoop_pass_profile,
            "mu_required": mu_required,
        },
        "discrete_course_plan": discrete_plan,
        "discrete_planning_metrics": discrete_plan.get("metrics", {}),
        "discrete_planning_warnings": discrete_plan.get("warnings", []),
        "helical_course_pairs": helical_course_pairs,
        "helical_courses": helical_courses,
        "hoop_rings": hoop_rings,
        "cure_cycle": _to_serializable(line.cure.steps),
        "inspection_gates": [
            {
                "name": "hashin_screening_gate",
                "passed": screening.get("hashin_constraint_satisfied"),
                "value": screening.get("fi_max_with_margin"),
            },
            {
                "name": "friction_screening_gate",
                "passed": screening.get("friction_constraint_satisfied"),
                "value": screening.get("mu_max_required"),
                "allowable": screening.get("mu_allowable"),
            },
            {
                "name": "inline_gap_overlap_gate",
                "passed": None,
                "allowable_gap_mm": line.inspection.max_gap_mm,
                "allowable_overlap_mm": line.inspection.max_overlap_mm,
            },
            {
                "name": "wrinkle_gate",
                "passed": None,
                "allowable_wrinkle_height_mm": line.inspection.max_wrinkle_height_mm,
            },
        ],
        "blocking_requirements": blockers,
        "execution_sequence": execution_sequence,
    }


def build_production_program_from_result(
    result: dict[str, Any],
    geom: GeometryConfig,
    config: HybridConfig,
    line_config: ProductionLineConfig | None = None,
    include_path_points: bool = True,
    planning_config: DiscreteCoursePlanningConfig | None = None,
) -> dict[str, Any]:
    from .visualize import build_winding_process_layout_data

    layout = build_winding_process_layout_data(
        result,
        geom,
        family_count=int(config.winding_family_count),
    )
    summary = _minimal_summary_from_result(result, geom)
    return build_production_program_from_layout(
        layout=layout,
        geom=geom,
        line_config=line_config,
        summary=summary,
        source_label="optimizer_result",
        include_path_points=include_path_points,
        planning_config=planning_config,
    )


def export_production_program(path: str | Path, program: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.write_text(json.dumps(_to_serializable(program), indent=2), encoding="utf-8")
    return path


def render_production_readiness_markdown(program: dict[str, Any]) -> str:
    source = program.get("source", {})
    geometry = program.get("geometry", {})
    screening = program.get("screening_snapshot", {})
    blockers = list(program.get("blocking_requirements", []))
    sequence = list(program.get("execution_sequence", []))
    helical_courses = list(program.get("helical_courses", []))
    helical_course_pairs = list(program.get("helical_course_pairs", []))
    hoop_rings = list(program.get("hoop_rings", []))
    discrete_metrics = dict(program.get("discrete_planning_metrics", {}))
    helical_metrics = dict(discrete_metrics.get("helical", {}))
    discrete_warnings = list(program.get("discrete_planning_warnings", []))

    lines = [
        "# COPV Production Readiness Report",
        "",
        "## Status",
        f"- Program status: `{program.get('program_status', 'unknown')}`",
        f"- Source label: `{source.get('label', 'unknown')}`",
        f"- Layout type: `{source.get('layout_type', 'unknown')}`",
        "",
        "## Geometry",
        f"- Outer radius: `{geometry.get('outer_radius_mm')}` mm",
        f"- Cylinder length: `{geometry.get('cylinder_length_mm')}` mm",
        f"- Wall thickness: `{geometry.get('wall_thickness_mm')}` mm",
        f"- Opening radius: `{geometry.get('opening_radius_mm')}` mm",
        f"- Pressure case: `{geometry.get('pressure')}`",
        "",
        "## Screening Snapshot",
        f"- Case: `{screening.get('case', 'unknown')}`",
        f"- FI max with margin: `{screening.get('fi_max_with_margin')}`",
        f"- Peak required friction coefficient: `{screening.get('mu_max_required')}`",
        f"- Allowable friction coefficient: `{screening.get('mu_allowable')}`",
        f"- Mass delta vs baseline: `{screening.get('mass_delta_percent_vs_baseline')}` %",
        f"- Burst factor: `{screening.get('burst_factor')}`",
        "",
        "## Exported Program Basis",
        f"- Helical course pairs exported: `{len(helical_course_pairs)}`",
        f"- Individual helical courses exported: `{len(helical_courses)}`",
        f"- Hoop rings exported: `{len(hoop_rings)}`",
        f"- Discrete cut/restart events: `{discrete_metrics.get('total_cut_restart_events', 0)}`",
        f"- Discrete helical pass RMSE: `{helical_metrics.get('continuous_profile_rmse', 0.0)}`",
        f"- Discrete helical activation imbalance CV: `{helical_metrics.get('pair_activation_length_cv', 0.0)}`",
    ]

    if helical_course_pairs:
        first = helical_course_pairs[0]
        lines.extend(
            [
                f"- First helical course pair id: `{first.get('pair_id')}`",
                f"- First helical course pair length: `{first.get('segment_length_mm')}` mm",
            ]
        )

    lines.extend(
        [
            "",
            "## Blocking Requirements Before Real Production Release",
        ]
    )
    for blocker in blockers:
        lines.append(f"- [ ] {blocker}")

    lines.extend(
        [
            "",
            "## Discrete Planning Warnings",
        ]
    )
    if discrete_warnings:
        for warning in discrete_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Planned Execution Sequence",
        ]
    )
    for index, step in enumerate(sequence, start=1):
        if step.get("step_type") == "helical_pair":
            lines.append(
                f"{index}. Run pair `{step['pair_id']}` using `{step['clockwise_course_id']}` and `{step['counter_clockwise_course_id']}`."
            )
        elif step.get("step_type") == "hoop_ring":
            lines.append(f"{index}. Run hoop ring `{step['ring_id']}`.")
        elif step.get("step_type") == "cure_cycle":
            lines.append(f"{index}. Execute cure cycle `{step.get('cycle_name')}`.")
        elif step.get("step_type") == "autofrettage":
            lines.append(f"{index}. Perform autofrettage. Note: {step.get('note')}")
        elif step.get("step_type") == "inspection_release":
            method = step.get("method") if step.get("method") is not None else "UNSPECIFIED"
            lines.append(f"{index}. Run release inspection using `{method}`.")
        else:
            lines.append(f"{index}. {step}")

    lines.extend(
        [
            "",
            "## Notes",
            "- This report is generated from the current winding-first staging artifact.",
            "- It is a production-planning scaffold, not a qualified machine release package.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_production_readiness_markdown(path: str | Path, program: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.write_text(render_production_readiness_markdown(program), encoding="utf-8")
    return path
