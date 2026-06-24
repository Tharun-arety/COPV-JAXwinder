from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import json
import math
import os
import re

import numpy as np

from .config import GeometryConfig, MaterialConfig
from .course_planner import DiscreteCoursePlanningConfig, build_discrete_winding_plan_from_layout
from .production import ProductionLineConfig, build_production_gap_report, build_production_program_from_layout


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


def _optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return None
    return arr


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return value
    return {}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_empty_collection(prefix: str) -> bool:
    return prefix == "notes" or prefix.endswith(".notes")


def _collect_missing_paths(value: Any, prefix: str = "") -> list[str]:
    missing: list[str] = []
    if is_dataclass(value):
        return _collect_missing_paths(asdict(value), prefix)
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            missing.extend(_collect_missing_paths(child, child_prefix))
        return missing
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            if not _optional_empty_collection(prefix):
                missing.append(prefix or "root")
            return missing
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            missing.extend(_collect_missing_paths(child, child_prefix))
        return missing
    if value is None:
        missing.append(prefix or "root")
    return missing


def _leaf_completeness(value: Any, prefix: str = "") -> tuple[int, int]:
    if is_dataclass(value):
        return _leaf_completeness(asdict(value), prefix)
    if isinstance(value, dict):
        total = 0
        filled = 0
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            child_total, child_filled = _leaf_completeness(child, child_prefix)
            total += child_total
            filled += child_filled
        return total, filled
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            if _optional_empty_collection(prefix):
                return 0, 0
            return 1, 0
        total = 0
        filled = 0
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            child_total, child_filled = _leaf_completeness(child, child_prefix)
            total += child_total
            filled += child_filled
        return total, filled
    return 1, 0 if value is None else 1


def _path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _unwrap_phi_deg(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    phi = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    return np.degrees(phi)


def _min_turning_radius(points: np.ndarray) -> float | None:
    if points.shape[0] < 3:
        return None
    radii: list[float] = []
    for idx in range(1, points.shape[0] - 1):
        p0 = points[idx - 1]
        p1 = points[idx]
        p2 = points[idx + 1]
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)
        if a <= 1e-9 or b <= 1e-9 or c <= 1e-9:
            continue
        area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        if area2 <= 1e-12:
            continue
        radius = (a * b * c) / max(2.0 * area2, 1e-12)
        if math.isfinite(radius):
            radii.append(float(radius))
    if not radii:
        return None
    return float(min(radii))


def _artifact_exists(path: str | Path) -> bool:
    return Path(path).exists()


def _phase_artifact_basename(phase_id: str, title: str, suffix: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"{phase_id}_{slug}.{suffix}"


def _phase_result(
    phase_id: str,
    title: str,
    status: str,
    release_gate_closed: bool,
    closure_basis: str,
    objective: str,
    how_it_is_done: list[str],
    why_this_way: list[str],
    inputs: dict[str, Any],
    metrics: dict[str, Any],
    verification_checks: list[dict[str, Any]],
    blockers: list[str],
    artifacts: list[str],
) -> dict[str, Any]:
    return {
        "phase_id": phase_id,
        "title": title,
        "status": status,
        "release_gate_closed": release_gate_closed,
        "closure_basis": closure_basis,
        "objective": objective,
        "how_it_is_done": how_it_is_done,
        "why_this_way": why_this_way,
        "inputs": inputs,
        "metrics": metrics,
        "verification_checks": verification_checks,
        "blockers": blockers,
        "artifacts": artifacts,
    }


def execute_phase_01_data_contract(program: dict[str, Any], line_config: ProductionLineConfig) -> dict[str, Any]:
    required_sections = [
        "geometry",
        "line_config",
        "screening_snapshot",
        "layout_profiles",
        "process_basis",
        "discrete_course_plan",
        "inspection_gates",
        "execution_sequence",
    ]
    section_presence = {name: name in program for name in required_sections}
    total_leaves, filled_leaves = _leaf_completeness(line_config)
    completeness = 0.0 if total_leaves <= 0 else filled_leaves / total_leaves
    missing_paths = _collect_missing_paths(line_config)
    declared_blockers = build_production_gap_report(line_config)
    verification_checks = [
        {
            "name": "required_program_sections_present",
            "passed": all(section_presence.values()),
            "detail": section_presence,
        },
        {
            "name": "line_config_has_declared_fields",
            "passed": total_leaves > 0,
            "detail": {"total_leaves": total_leaves, "filled_leaves": filled_leaves},
        },
    ]
    status = "implemented_with_external_gaps" if declared_blockers or completeness < 1.0 else "complete"
    blockers = list(declared_blockers)
    if missing_paths:
        blockers.append(
            f"Recursive contract audit still finds {len(missing_paths)} unfilled leaf fields after excluding optional notes."
        )
    release_gate_closed = status == "complete"
    return _phase_result(
        phase_id="phase_01",
        title="Production Data Contract",
        status=status,
        release_gate_closed=release_gate_closed,
        closure_basis=(
            "All required contract fields are populated and the exported program exposes the needed sections."
            if release_gate_closed
            else "The contract scaffold exists, but required machine/material/process inputs are still missing."
        ),
        objective="Define a stable machine/material/process contract that every later production phase can consume.",
        how_it_is_done=[
            "The optimizer result is normalized into a structured production program with explicit geometry, screening state, line configuration, inspection gates, and execution order.",
            "The line configuration is checked recursively for missing leaf fields so the repo can distinguish between missing engineering data and missing code.",
        ],
        why_this_way=[
            "A production optimizer fails in practice when downstream tooling relies on implicit assumptions. The contract is explicit so machine, cure, and inspection phases all read the same object model.",
            "The completeness check is recursive because later phases depend on deeply nested data such as heater limits, NDI method, and qualification evidence.",
        ],
        inputs={
            "required_sections": required_sections,
            "line_name": line_config.line_name,
        },
        metrics={
            "contract_section_presence": section_presence,
            "line_config_leaf_count": total_leaves,
            "line_config_filled_leaf_count": filled_leaves,
            "line_config_completeness_ratio": completeness,
            "missing_field_count": len(missing_paths),
            "declared_blocker_count": len(declared_blockers),
        },
        verification_checks=verification_checks,
        blockers=blockers,
        artifacts=[],
    )


def execute_phase_02_discrete_planning(discrete_plan: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(discrete_plan.get("metrics", {}))
    helical_metrics = dict(metrics.get("helical", {}))
    hoop_metrics = dict(metrics.get("hoop", {}))
    warnings = list(discrete_plan.get("warnings", []))
    verification_checks = [
        {
            "name": "helical_course_pairs_exported",
            "passed": int(metrics.get("total_course_pairs", 0)) > 0,
            "detail": metrics.get("total_course_pairs", 0),
        },
        {
            "name": "helical_quantization_error_finite",
            "passed": math.isfinite(float(helical_metrics.get("continuous_profile_rmse", 0.0))),
            "detail": helical_metrics.get("continuous_profile_rmse", 0.0),
        },
        {
            "name": "hoop_quantization_error_finite",
            "passed": math.isfinite(float(hoop_metrics.get("continuous_profile_rmse", 0.0))),
            "detail": hoop_metrics.get("continuous_profile_rmse", 0.0),
        },
    ]
    status = "computed_with_warnings" if warnings else "computed"
    release_gate_closed = status == "computed"
    return _phase_result(
        phase_id="phase_02",
        title="Discrete Course Planning",
        status=status,
        release_gate_closed=release_gate_closed,
        closure_basis=(
            "Explicit course objects were exported without planner warnings."
            if release_gate_closed
            else "The discrete planner executed, but warnings show the current plan still needs engineering review before line release."
        ),
        objective="Translate the continuous winding screening field into explicit helical course pairs and hoop ring objects.",
        how_it_is_done=[
            "The continuous helical pass profile is quantized into balanced clockwise/counter-clockwise course pairs.",
            "Short inactive gaps are merged and short active segments are removed so the exported plan avoids unrealistic fragments.",
            "The hoop profile is quantized separately into ring bands on the cylindrical span.",
        ],
        why_this_way=[
            "The current optimizer is still axisymmetric, so the planner discretizes downstream instead of pretending the continuous field is already machine-ready.",
            "Balanced handedness preserves the structural intent of the screening solution while making the plan closer to what a real winding schedule would execute.",
        ],
        inputs={
            "sample_point_count": len(discrete_plan.get("sample_s_mm", [])),
        },
        metrics={
            "total_course_pairs": metrics.get("total_course_pairs", 0),
            "total_individual_courses": metrics.get("total_individual_courses", 0),
            "total_hoop_rings": metrics.get("total_hoop_rings", 0),
            "total_cut_restart_events": metrics.get("total_cut_restart_events", 0),
            "helical_pass_rmse": helical_metrics.get("continuous_profile_rmse", 0.0),
            "hoop_pass_rmse": hoop_metrics.get("continuous_profile_rmse", 0.0),
            "helical_activation_balance_cv": helical_metrics.get("pair_activation_length_cv", 0.0),
        },
        verification_checks=verification_checks,
        blockers=warnings,
        artifacts=[],
    )


def execute_phase_03_machine_kinematics(program: dict[str, Any]) -> dict[str, Any]:
    courses = list(program.get("helical_courses", []))
    line = _as_mapping(program.get("line_config"))
    machine = _as_mapping(line.get("machine"))
    nominal_speed = _safe_float(machine.get("nominal_head_speed_mm_s"))
    max_head_speed = _safe_float(machine.get("max_head_speed_mm_s"))
    max_mandrel_rpm = _safe_float(machine.get("max_mandrel_rpm"))
    min_turning_radius_limit = _safe_float(machine.get("min_turning_radius_mm"))
    planned_head_speed_violation = (
        nominal_speed is not None and max_head_speed is not None and nominal_speed > max_head_speed
    )

    course_metrics: list[dict[str, Any]] = []
    rpm_violations = 0
    radius_violations = 0
    finite_radii: list[float] = []
    finite_rpms: list[float] = []
    path_lengths: list[float] = []
    for course in courses:
        points = _optional_array(course.get("path_points_mm"))
        if points is None or points.shape[0] < 2:
            course_metrics.append({"course_id": course.get("course_id"), "valid_points": False})
            continue
        path_length_mm = _path_length(points)
        phi_deg = _unwrap_phi_deg(points)
        mandrel_rotation_deg = float(phi_deg[-1] - phi_deg[0]) if phi_deg.size > 0 else 0.0
        min_turning_radius_mm = _min_turning_radius(points)
        time_s = None if nominal_speed in (None, 0.0) else path_length_mm / nominal_speed
        required_mandrel_rpm = None
        if time_s is not None and time_s > 1e-9:
            required_mandrel_rpm = abs(mandrel_rotation_deg) / 360.0 / time_s * 60.0
            finite_rpms.append(float(required_mandrel_rpm))
        if min_turning_radius_mm is not None:
            finite_radii.append(float(min_turning_radius_mm))
        path_lengths.append(path_length_mm)
        if max_mandrel_rpm is not None and required_mandrel_rpm is not None and required_mandrel_rpm > max_mandrel_rpm:
            rpm_violations += 1
        if min_turning_radius_limit is not None and min_turning_radius_mm is not None and min_turning_radius_mm < min_turning_radius_limit:
            radius_violations += 1
        course_metrics.append(
            {
                "course_id": course.get("course_id"),
                "path_length_mm": path_length_mm,
                "mandrel_rotation_deg": mandrel_rotation_deg,
                "estimated_time_s": time_s,
                "required_mandrel_rpm": required_mandrel_rpm,
                "min_turning_radius_mm": min_turning_radius_mm,
            }
        )

    limit_ready = (
        nominal_speed is not None
        and max_head_speed is not None
        and max_mandrel_rpm is not None
        and min_turning_radius_limit is not None
    )
    status = "screened_against_limits" if limit_ready else "demand_quantified_only"
    blockers: list[str] = []
    if nominal_speed is None:
        blockers.append("Machine nominal head speed is not defined, so time and RPM demand cannot be converted into a planned cycle-rate check.")
    if max_head_speed is None:
        blockers.append("Machine maximum head speed is not defined, so kinematic demand can only be quantified, not accepted/rejected.")
    if max_mandrel_rpm is None:
        blockers.append("Machine mandrel RPM limit is not defined, so rotational feasibility cannot be closed.")
    if min_turning_radius_limit is None:
        blockers.append("Machine minimum turning radius is not defined, so steering feasibility cannot be closed.")
    if planned_head_speed_violation:
        blockers.append("The declared nominal head speed is above the machine maximum head speed.")
    metrics = {
        "course_count": len(courses),
        "max_path_length_mm": 0.0 if not path_lengths else float(max(path_lengths)),
        "mean_path_length_mm": 0.0 if not path_lengths else float(np.mean(path_lengths)),
        "max_required_mandrel_rpm": None if not finite_rpms else float(max(finite_rpms)),
        "min_local_turning_radius_mm": None if not finite_radii else float(min(finite_radii)),
        "rpm_violation_count": rpm_violations,
        "turning_radius_violation_count": radius_violations,
        "nominal_head_speed_mm_s": nominal_speed,
        "max_head_speed_mm_s": max_head_speed,
    }
    verification_checks = [
        {
            "name": "courses_have_geometry",
            "passed": all(item.get("valid_points", True) for item in course_metrics),
            "detail": {"course_count": len(courses)},
        },
        {
            "name": "turning_radius_computed",
            "passed": len(finite_radii) == len(courses) if courses else True,
            "detail": {"finite_radius_count": len(finite_radii), "course_count": len(courses)},
        },
        {
            "name": "mandrel_rpm_within_limit",
            "passed": None if max_mandrel_rpm is None else rpm_violations == 0,
            "detail": {"rpm_violations": rpm_violations, "max_mandrel_rpm_limit": max_mandrel_rpm},
        },
        {
            "name": "planned_head_speed_within_machine_limit",
            "passed": None if nominal_speed is None or max_head_speed is None else nominal_speed <= max_head_speed,
            "detail": {"nominal_head_speed_mm_s": nominal_speed, "max_head_speed_mm_s": max_head_speed},
        },
        {
            "name": "turning_radius_within_limit",
            "passed": None if min_turning_radius_limit is None else radius_violations == 0,
            "detail": {
                "radius_violations": radius_violations,
                "minimum_turning_radius_limit_mm": min_turning_radius_limit,
            },
        },
    ]
    return _phase_result(
        phase_id="phase_03",
        title="Machine Kinematic Demand Screen",
        status=status,
        release_gate_closed=False,
        closure_basis=(
            "The phase quantifies kinematic demand, but a machine-specific inverse-kinematics and NC/post-processor stack is still missing."
        ),
        objective="Quantify the geometric and rotational demand that the discrete courses place on a real winding machine.",
        how_it_is_done=[
            "Each discrete helical course is converted into 3D path-demand metrics: path length, mandrel rotation, estimated time, required RPM, and local turning radius.",
            "If machine limits are present, those demand metrics are checked against the declared hardware envelope.",
        ],
        why_this_way=[
            "A real winder must satisfy path demand before any structural argument matters. The analysis is demand-first because the repo still lacks a machine-specific inverse-kinematics post-processor.",
            "Turning radius is derived from the actual exported 3D path instead of only from angle profiles so the report reflects what the discrete plan is really asking the hardware to do.",
        ],
        inputs={
            "course_count": len(courses),
            "machine_name": machine.get("machine_name"),
        },
        metrics=metrics,
        verification_checks=verification_checks,
        blockers=blockers,
        artifacts=[],
    )


def execute_phase_04_towpreg_deposition(program: dict[str, Any]) -> dict[str, Any]:
    process_basis = dict(program.get("process_basis", {}))
    layout_profiles = dict(program.get("layout_profiles", {}))
    discrete_plan = dict(program.get("discrete_course_plan", {}))
    line = _as_mapping(program.get("line_config"))
    material = _as_mapping(line.get("material"))
    head = _as_mapping(line.get("heating_compaction"))
    screening = dict(program.get("screening_snapshot", {}))
    planner_config = _as_mapping(discrete_plan.get("planner_config"))
    sample_s = _optional_array(layout_profiles.get("sample_s_mm"))
    quantized_helical = _optional_array(discrete_plan.get("quantized_helical_pass_profile"))
    quantized_hoop = _optional_array(discrete_plan.get("quantized_hoop_pass_profile"))
    continuous_helical = _optional_array(layout_profiles.get("helical_pass_profile"))
    continuous_hoop = _optional_array(layout_profiles.get("hoop_pass_profile"))
    if sample_s is None:
        sample_s = np.zeros((0,), dtype=np.float64)
    if continuous_helical is None:
        continuous_helical = np.zeros_like(sample_s)
    if continuous_hoop is None:
        continuous_hoop = np.zeros_like(sample_s)
    if quantized_helical is None:
        quantized_helical = np.zeros_like(sample_s)
    if quantized_hoop is None:
        quantized_hoop = np.zeros_like(sample_s)

    continuous_build = continuous_helical + continuous_hoop
    quantized_build = quantized_helical + quantized_hoop
    max_build = float(np.max(quantized_build)) if quantized_build.size > 0 else 0.0
    relative_build = np.zeros_like(quantized_build) if max_build <= 1e-9 else quantized_build / max_build
    relative_tension_seed = np.clip(1.0 - 0.20 * relative_build, 0.8, 1.0)
    relative_compaction_seed = np.clip(0.85 + 0.15 * relative_build, 0.85, 1.0)

    friction_required = _safe_float(screening.get("mu_max_required"))
    friction_allowable = _safe_float(screening.get("mu_allowable"))
    friction_headroom = None
    if friction_required is not None and friction_allowable is not None:
        friction_headroom = friction_allowable - friction_required

    tension_window = material.get("allowable_tension_window_n")
    heater_window = material.get("deposition_temperature_window_c")
    nominal_tension_defined = tension_window is not None
    nominal_heater_defined = heater_window is not None or head.get("target_heater_setpoint_c") is not None

    continuity = 1.0
    total_courses = int(discrete_plan.get("metrics", {}).get("total_individual_courses", 0))
    cut_events = int(discrete_plan.get("metrics", {}).get("total_cut_restart_events", 0))
    if total_courses > 0:
        continuity = max(0.0, 1.0 - cut_events / max(2 * total_courses, 1))

    blockers: list[str] = []
    if not nominal_tension_defined:
        blockers.append("No validated tow tension window is defined, so deposition tension can only be expressed as a relative seed profile.")
    if not nominal_heater_defined:
        blockers.append("No validated deposition temperature window is defined, so heater demand can only be reported qualitatively.")

    verification_checks = [
        {
            "name": "friction_headroom_positive",
            "passed": None if friction_headroom is None else friction_headroom >= 0.0,
            "detail": {"headroom": friction_headroom},
        },
        {
            "name": "quantization_error_within_helical_warning",
            "passed": float(discrete_plan.get("metrics", {}).get("helical", {}).get("continuous_profile_rmse", 0.0))
            <= float(planner_config.get("helical_rmse_warning", 0.10)),
            "detail": {
                "rmse": discrete_plan.get("metrics", {}).get("helical", {}).get("continuous_profile_rmse", 0.0),
                "warning_limit": planner_config.get("helical_rmse_warning", 0.10),
            },
        },
        {
            "name": "relative_tension_seed_bounded",
            "passed": bool(np.all((relative_tension_seed >= 0.8) & (relative_tension_seed <= 1.0))) if relative_tension_seed.size else True,
            "detail": {
                "min_seed": None if relative_tension_seed.size == 0 else float(np.min(relative_tension_seed)),
                "max_seed": None if relative_tension_seed.size == 0 else float(np.max(relative_tension_seed)),
            },
        },
    ]
    status = "relative_seed_screening_only"
    return _phase_result(
        phase_id="phase_04",
        title="Towpreg Relative Setpoint Screen",
        status=status,
        release_gate_closed=False,
        closure_basis=(
            "This phase outputs relative setpoint seeds only; the repo still lacks a calibrated towpreg heating, tack, and compaction model."
        ),
        objective="Translate the discrete course plan into deposition demand metrics and relative process setpoint seeds.",
        how_it_is_done=[
            "The discrete helical and hoop build profiles are combined into a local layer-build demand profile.",
            "Relative tension and compaction seed profiles are derived from that local build profile so thicker regions can be handled more conservatively.",
            "The staged friction screen is carried forward as the current slip-risk metric until calibrated towpreg deposition physics are added.",
        ],
        why_this_way=[
            "The repo does not yet have calibrated towpreg tack, heating, and compaction physics, so this phase reports demand-side surrogates instead of invented release values.",
            "Using dimensionless seed profiles preserves the ordering of process demand without pretending that the current code knows your machine's exact tension or heater recipe.",
        ],
        inputs={
            "material_name": material.get("material_name"),
            "sample_point_count": len(sample_s),
        },
        metrics={
            "peak_required_friction_coefficient": friction_required,
            "friction_headroom": friction_headroom,
            "max_nominal_layer_build": 0.0 if continuous_build.size == 0 else float(np.max(continuous_build)),
            "max_discrete_layer_build": 0.0 if quantized_build.size == 0 else float(np.max(quantized_build)),
            "deposition_continuity_index": continuity,
            "relative_tension_seed_min": None if relative_tension_seed.size == 0 else float(np.min(relative_tension_seed)),
            "relative_tension_seed_max": None if relative_tension_seed.size == 0 else float(np.max(relative_tension_seed)),
            "relative_compaction_seed_min": None if relative_compaction_seed.size == 0 else float(np.min(relative_compaction_seed)),
            "relative_compaction_seed_max": None if relative_compaction_seed.size == 0 else float(np.max(relative_compaction_seed)),
        },
        verification_checks=verification_checks,
        blockers=blockers,
        artifacts=[],
    )


def execute_phase_05_as_built_surrogate(program: dict[str, Any], material: MaterialConfig | None = None) -> dict[str, Any]:
    material_cfg = MaterialConfig() if material is None else material
    layout_profiles = dict(program.get("layout_profiles", {}))
    discrete_plan = dict(program.get("discrete_course_plan", {}))
    line = _as_mapping(program.get("line_config"))
    inspection = _as_mapping(line.get("inspection"))

    sample_s = _optional_array(layout_profiles.get("sample_s_mm"))
    nominal_winding_thickness = _optional_array(layout_profiles.get("thickness_profile_mm"))
    continuous_helical = _optional_array(discrete_plan.get("continuous_helical_pass_profile"))
    continuous_hoop = _optional_array(discrete_plan.get("continuous_hoop_pass_profile"))
    quantized_helical = _optional_array(discrete_plan.get("quantized_helical_pass_profile"))
    quantized_hoop = _optional_array(discrete_plan.get("quantized_hoop_pass_profile"))

    if sample_s is None:
        sample_s = np.zeros((0,), dtype=np.float64)
    if nominal_winding_thickness is None:
        nominal_winding_thickness = np.zeros_like(sample_s)
    if continuous_helical is None:
        continuous_helical = np.zeros_like(sample_s)
    if continuous_hoop is None:
        continuous_hoop = np.zeros_like(sample_s)
    if quantized_helical is None:
        quantized_helical = np.zeros_like(sample_s)
    if quantized_hoop is None:
        quantized_hoop = np.zeros_like(sample_s)

    continuous_total_pass = continuous_helical + continuous_hoop
    quantized_total_pass = quantized_helical + quantized_hoop
    scale = np.zeros_like(nominal_winding_thickness)
    active = continuous_total_pass > 1e-9
    scale[active] = quantized_total_pass[active] / continuous_total_pass[active]
    discrete_winding_thickness = nominal_winding_thickness * scale
    gap_profile = np.maximum(nominal_winding_thickness - discrete_winding_thickness, 0.0)
    overlap_profile = np.maximum(discrete_winding_thickness - nominal_winding_thickness, 0.0)
    nominal_total_thickness = material_cfg.base_thickness + nominal_winding_thickness
    discrete_total_thickness = material_cfg.base_thickness + discrete_winding_thickness
    thickness_error = discrete_winding_thickness - nominal_winding_thickness
    rmse = float(np.sqrt(np.mean(thickness_error**2))) if thickness_error.size > 0 else 0.0
    gap_max = float(np.max(gap_profile)) if gap_profile.size > 0 else 0.0
    overlap_max = float(np.max(overlap_profile)) if overlap_profile.size > 0 else 0.0
    mass_ratio = 1.0
    if nominal_total_thickness.size > 0 and np.sum(nominal_total_thickness) > 1e-9:
        mass_ratio = float(np.sum(discrete_total_thickness) / np.sum(nominal_total_thickness))

    gap_limit = _safe_float(inspection.get("max_gap_mm"))
    overlap_limit = _safe_float(inspection.get("max_overlap_mm"))
    blockers: list[str] = []
    if gap_limit is None:
        blockers.append("Gap acceptance threshold is missing, so the surrogate gap field cannot be accepted or rejected against production criteria.")
    if overlap_limit is None:
        blockers.append("Overlap acceptance threshold is missing, so the surrogate overlap field cannot be accepted or rejected against production criteria.")

    verification_checks = [
        {
            "name": "total_thickness_nonnegative",
            "passed": bool(np.all(discrete_total_thickness >= 0.0)),
            "detail": {"minimum_total_thickness_mm": None if discrete_total_thickness.size == 0 else float(np.min(discrete_total_thickness))},
        },
        {
            "name": "gap_within_limit",
            "passed": None if gap_limit is None else gap_max <= gap_limit,
            "detail": {"max_gap_mm": gap_max, "gap_limit_mm": gap_limit},
        },
        {
            "name": "overlap_within_limit",
            "passed": None if overlap_limit is None else overlap_max <= overlap_limit,
            "detail": {"max_overlap_mm": overlap_max, "overlap_limit_mm": overlap_limit},
        },
    ]
    status = "first_order_surrogate"
    return _phase_result(
        phase_id="phase_05",
        title="First-Order As-Built Laminate Surrogate",
        status=status,
        release_gate_closed=False,
        closure_basis=(
            "The current implementation builds a first-order as-built thickness surrogate, not a calibrated manufactured-state laminate model."
        ),
        objective="Build a first-order as-built laminate state from the discrete course plan before running higher-fidelity process physics.",
        how_it_is_done=[
            "The continuous nominal winding thickness profile is rescaled by the ratio between quantized discrete passes and continuous screening passes.",
            "That yields a first-order discrete thickness field, plus gap and overlap fields derived from the difference between nominal and quantized build.",
        ],
        why_this_way=[
            "A real production optimizer must reason about the manufactured laminate, not just the commanded design field. This surrogate is the lightest honest bridge from discrete plan to as-built state.",
            "The implementation uses pass-ratio rescaling instead of pretending to know full resin flow or tow compaction mechanics, because those models are not yet calibrated in-repo.",
        ],
        inputs={
            "base_thickness_mm": material_cfg.base_thickness,
            "sample_point_count": len(sample_s),
        },
        metrics={
            "thickness_rmse_mm": rmse,
            "max_gap_mm": gap_max,
            "max_overlap_mm": overlap_max,
            "discrete_to_nominal_mass_ratio": mass_ratio,
            "max_total_thickness_mm": 0.0 if discrete_total_thickness.size == 0 else float(np.max(discrete_total_thickness)),
        },
        verification_checks=verification_checks,
        blockers=blockers,
        artifacts=[],
    )


def execute_phase_06_process_state(program: dict[str, Any]) -> dict[str, Any]:
    line = _as_mapping(program.get("line_config"))
    cure = _as_mapping(line.get("cure"))
    autofrettage = _as_mapping(line.get("autofrettage"))
    screening = dict(program.get("screening_snapshot", {}))

    cure_steps = list(cure.get("steps", []))
    hold_times = [step.get("hold_time_min") for step in cure_steps if isinstance(step, dict)]
    hold_times = [float(v) for v in hold_times if v is not None]
    temps = [step.get("target_temp_c") for step in cure_steps if isinstance(step, dict)]
    temps = [float(v) for v in temps if v is not None]

    target_pressure = _safe_float(autofrettage.get("target_pressure"))
    operating_pressure = _safe_float(screening.get("pressure"))
    allowable_pressure = _safe_float(screening.get("allowable_pressure_with_margin"))
    pressure_to_autofrettage_ratio = None
    if operating_pressure is not None and target_pressure not in (None, 0.0):
        pressure_to_autofrettage_ratio = operating_pressure / target_pressure

    total_required_inputs = 5
    filled_inputs = 0
    filled_inputs += 1 if cure_steps else 0
    filled_inputs += 1 if target_pressure is not None else 0
    filled_inputs += 1 if autofrettage.get("hold_time_min") is not None else 0
    filled_inputs += 1 if autofrettage.get("liner_yield_pressure") is not None else 0
    filled_inputs += 1 if cure.get("cycle_name") not in (None, "UNSPECIFIED_CURE") else 0
    completeness = filled_inputs / total_required_inputs

    blockers: list[str] = []
    if not cure_steps:
        blockers.append("Cure cycle steps are missing, so degree-of-cure and residual thermal history cannot be simulated.")
    if target_pressure is None:
        blockers.append("Autofrettage target pressure is missing, so the liner/plastic pre-stress state cannot be evaluated.")
    if autofrettage.get("liner_yield_pressure") is None:
        blockers.append("Liner yield pressure is missing, so the autofrettage window cannot be validated.")

    verification_checks = [
        {
            "name": "cure_cycle_defined",
            "passed": bool(cure_steps),
            "detail": {"step_count": len(cure_steps)},
        },
        {
            "name": "autofrettage_target_defined",
            "passed": target_pressure is not None,
            "detail": {"target_pressure": target_pressure},
        },
        {
            "name": "operating_pressure_below_allowable_pressure",
            "passed": None if operating_pressure is None or allowable_pressure is None else operating_pressure <= allowable_pressure,
            "detail": {"operating_pressure": operating_pressure, "allowable_pressure_with_margin": allowable_pressure},
        },
    ]
    status = "blocked_by_input_gaps" if blockers else "input_stack_ready_for_coupling"
    return _phase_result(
        phase_id="phase_06",
        title="Cure and Autofrettage Input Readiness",
        status=status,
        release_gate_closed=False,
        closure_basis=(
            "The phase can only audit input readiness today; a coupled cure/residual-stress/autofrettage solver is not yet implemented in-repo."
        ),
        objective="Prepare the coupled cured/autofrettaged process state that a production-grade structural release would require.",
        how_it_is_done=[
            "The phase checks whether cure steps, autofrettage target pressure, hold logic, and liner yield information are defined in the production contract.",
            "Where data exists, it aggregates cure exposure and pressure ratios so the repo can measure process-state readiness before adding a full coupled solver.",
        ],
        why_this_way=[
            "Without cure kinetics and liner plasticity data, any residual-stress number would be fabricated. This phase is intentionally readiness-gated rather than numerically overconfident.",
            "The completeness ratio makes the missing process inputs explicit so later engineering effort goes to the right bottleneck instead of the wrong code path.",
        ],
        inputs={
            "cure_cycle_name": cure.get("cycle_name"),
            "autofrettage_model": autofrettage.get("residual_stress_model"),
        },
        metrics={
            "input_completeness_ratio": completeness,
            "cure_step_count": len(cure_steps),
            "cure_peak_temperature_c": None if not temps else max(temps),
            "cure_total_hold_time_min": 0.0 if not hold_times else float(sum(hold_times)),
            "autofrettage_target_pressure": target_pressure,
            "pressure_to_autofrettage_ratio": pressure_to_autofrettage_ratio,
            "operating_pressure": operating_pressure,
            "allowable_pressure_with_margin": allowable_pressure,
        },
        verification_checks=verification_checks,
        blockers=blockers,
        artifacts=[],
    )


def execute_phase_07_inspection_and_traceability(
    program: dict[str, Any],
    as_built_phase: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    line = _as_mapping(program.get("line_config"))
    inspection = _as_mapping(line.get("inspection"))
    qualification = _as_mapping(line.get("qualification"))
    screening = dict(program.get("screening_snapshot", {}))
    discrete_plan_artifact = output_dir / _phase_artifact_basename("phase_02", "Discrete Course Planning", "json")
    kinematics_artifact = output_dir / _phase_artifact_basename("phase_03", "Machine Kinematic Demand Screen", "json")
    as_built_artifact = output_dir / _phase_artifact_basename(
        "phase_05",
        "First-Order As-Built Laminate Surrogate",
        "json",
    )
    traceability_record = {
        "program_type": program.get("program_type"),
        "line_name": line.get("line_name"),
        "material_name": dict(line.get("material", {})).get("material_name"),
        "source_label": dict(program.get("source", {})).get("label"),
        "discrete_plan_artifact": str(discrete_plan_artifact),
        "kinematics_artifact": str(kinematics_artifact),
        "as_built_artifact": str(as_built_artifact),
        "required_ndi_method": inspection.get("final_ndi_method"),
        "qualification_standard": line.get("qualification_standard"),
        "qualification_datasets": {
            "coupon": qualification.get("coupon_dataset_path"),
            "subcomponent": qualification.get("subcomponent_dataset_path"),
            "vessel": qualification.get("vessel_dataset_path"),
        },
        "screening_hashin_pass": screening.get("hashin_constraint_satisfied"),
        "screening_friction_pass": screening.get("friction_constraint_satisfied"),
    }
    metrics = dict(as_built_phase.get("metrics", {}))
    gap_limit = _safe_float(inspection.get("max_gap_mm"))
    overlap_limit = _safe_float(inspection.get("max_overlap_mm"))
    wrinkle_limit = _safe_float(inspection.get("max_wrinkle_height_mm"))
    blockers: list[str] = []
    if inspection.get("final_ndi_method") is None:
        blockers.append("Final NDI method is not defined, so the release packet cannot close the inspection loop.")
    if gap_limit is None or overlap_limit is None or wrinkle_limit is None:
        blockers.append("Inspection acceptance thresholds are incomplete, so as-built deviation fields cannot be judged against a release plan.")
    if qualification.get("coupon_dataset_path") is None:
        blockers.append("Coupon qualification evidence is not linked into the digital thread.")

    verification_checks = [
        {
            "name": "source_layout_artifact_present",
            "passed": _artifact_exists(traceability_record["source_label"]) if traceability_record["source_label"] else False,
            "detail": {"source_label": traceability_record["source_label"]},
        },
        {
            "name": "internal_traceability_paths_declared",
            "passed": all(
                bool(path)
                for path in [
                    traceability_record["discrete_plan_artifact"],
                    traceability_record["kinematics_artifact"],
                    traceability_record["as_built_artifact"],
                ]
            ),
            "detail": {
                "discrete_plan_artifact": traceability_record["discrete_plan_artifact"],
                "kinematics_artifact": traceability_record["kinematics_artifact"],
                "as_built_artifact": traceability_record["as_built_artifact"],
            },
        },
        {
            "name": "gap_threshold_defined",
            "passed": gap_limit is not None,
            "detail": {"gap_limit_mm": gap_limit},
        },
        {
            "name": "ndi_method_defined",
            "passed": inspection.get("final_ndi_method") is not None,
            "detail": {"final_ndi_method": inspection.get("final_ndi_method")},
        },
    ]
    status = "partial_traceability_only" if blockers else "traceability_scaffold_ready"
    return _phase_result(
        phase_id="phase_07",
        title="Inspection and Digital Thread Scaffold",
        status=status,
        release_gate_closed=False,
        closure_basis=(
            "The repo now emits a digital-thread scaffold, but it does not yet ingest live inspection data or integrate with a real quality/MES stack."
        ),
        objective="Connect the production program to quality gates, as-built deviations, and evidence references required for traceable release.",
        how_it_is_done=[
            "The phase builds a traceability record linking the source optimizer artifact, discrete plan, as-built surrogate, and declared inspection/qualification references.",
            "Inspection readiness is checked against the presence of NDI method and acceptance thresholds for gap, overlap, and wrinkle criteria.",
        ],
        why_this_way=[
            "A production optimizer is not just a geometry generator. It must produce a digital thread that later inspection and certification systems can consume.",
            "The traceability record is explicit so missing evidence is visible as missing data, not silently assumed to exist.",
        ],
        inputs={
            "line_name": line.get("line_name"),
            "qualification_standard": line.get("qualification_standard"),
        },
        metrics={
            "max_gap_mm": metrics.get("max_gap_mm"),
            "max_overlap_mm": metrics.get("max_overlap_mm"),
            "thickness_rmse_mm": metrics.get("thickness_rmse_mm"),
            "required_ndi_method": inspection.get("final_ndi_method"),
            "traceability_record_field_count": len(traceability_record),
            "source_layout_artifact": traceability_record["source_label"],
            "discrete_plan_artifact": traceability_record["discrete_plan_artifact"],
            "kinematics_artifact": traceability_record["kinematics_artifact"],
            "as_built_artifact": traceability_record["as_built_artifact"],
        },
        verification_checks=verification_checks,
        blockers=blockers,
        artifacts=[],
    )


def execute_phase_08_qualification_release(
    phases: dict[str, dict[str, Any]],
    program: dict[str, Any],
) -> dict[str, Any]:
    line = _as_mapping(program.get("line_config"))
    qualification = _as_mapping(line.get("qualification"))
    required_paths = [
        qualification.get("coupon_dataset_path"),
        qualification.get("subcomponent_dataset_path"),
        qualification.get("vessel_dataset_path"),
    ]
    required_paths = [path for path in required_paths if path is not None]
    external_evidence_ready = bool(required_paths) and all(_artifact_exists(path) for path in required_paths)
    internal_phase_status = {phase_id: phase.get("status") for phase_id, phase in phases.items()}
    internal_gate_closure = {
        phase_id: bool(phase.get("release_gate_closed"))
        for phase_id, phase in phases.items()
        if phase_id in {"phase_01", "phase_02", "phase_03", "phase_04", "phase_05", "phase_06", "phase_07"}
    }
    internal_ready = all(internal_gate_closure.values())
    release_ready = internal_ready and external_evidence_ready
    blockers: list[str] = []
    if not internal_ready:
        open_phases = [phase_id for phase_id, closed in internal_gate_closure.items() if not closed]
        blockers.append(f"Internal production release gates remain open for: {', '.join(open_phases)}.")
    if not external_evidence_ready:
        blockers.append("Qualification dataset references are missing or do not exist on disk, so production release evidence is incomplete.")
    verification_checks = [
        {
            "name": "internal_phase_stack_executed",
            "passed": True,
            "detail": internal_phase_status,
        },
        {
            "name": "internal_release_gates_closed",
            "passed": internal_ready,
            "detail": internal_gate_closure,
        },
        {
            "name": "external_qualification_evidence_present",
            "passed": external_evidence_ready,
            "detail": {"required_paths": required_paths},
        },
        {
            "name": "release_decision_ready",
            "passed": release_ready,
            "detail": {"release_ready": release_ready},
        },
    ]
    return _phase_result(
        phase_id="phase_08",
        title="Qualification and Release Decision",
        status="release_ready" if release_ready else "do_not_release",
        release_gate_closed=release_ready,
        closure_basis=(
            "All internal release gates are closed and the required qualification evidence exists."
            if release_ready
            else "The staged repo still lacks one or more closed internal production gates and/or the required external qualification evidence."
        ),
        objective="Decide whether the current production pipeline has enough engineering and qualification evidence to release a real COPV build.",
        how_it_is_done=[
            "Internal evidence is gathered from the executed phase stack: data contract, discrete planning, kinematics, deposition, as-built, process-state, and inspection.",
            "External evidence is checked through the declared qualification dataset references for coupon, subcomponent, and vessel-level proof.",
        ],
        why_this_way=[
            "A production release is an evidence problem as much as a modeling problem. This phase makes that release logic explicit instead of leaving it implicit in engineering judgment alone.",
            "The release decision is conservative by design: missing qualification evidence forces a `do_not_release` result even if the in-repo screening artifacts look good.",
        ],
        inputs={
            "qualification_standard": line.get("qualification_standard"),
            "required_qualification_paths": required_paths,
        },
        metrics={
            "internal_phase_count": len(phases),
            "closed_internal_phase_count": sum(1 for closed in internal_gate_closure.values() if closed),
            "internal_release_stack_ready": internal_ready,
            "external_evidence_ready": external_evidence_ready,
            "release_ready": release_ready,
        },
        verification_checks=verification_checks,
        blockers=blockers,
        artifacts=[],
    )


def render_phase_markdown(phase: dict[str, Any]) -> str:
    lines = [
        f"# {phase['title']}",
        "",
        f"Status: `{phase['status']}`",
        f"Real production gate closed: `{phase.get('release_gate_closed')}`",
        f"Closure basis: {phase.get('closure_basis')}",
        "",
        "## Objective",
        phase["objective"],
        "",
        "## How It Is Done",
    ]
    for item in phase.get("how_it_is_done", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Why It Is Done This Way"])
    for item in phase.get("why_this_way", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Inputs"])
    for key, value in phase.get("inputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Metrics"])
    for key, value in phase.get("metrics", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Verification"])
    for check in phase.get("verification_checks", []):
        lines.append(f"- `{check.get('name')}`: `{check.get('passed')}`")
        lines.append(f"- Detail: `{check.get('detail')}`")
    lines.extend(["", "## Blockers"])
    blockers = phase.get("blockers", [])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Artifacts"])
    for artifact in phase.get("artifacts", []):
        lines.append(f"- {artifact}")
    lines.append("")
    return "\n".join(lines)


def export_phase_artifacts(output_dir: Path, phase: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_payload = dict(phase)
    phase_payload["artifacts"] = [
        _phase_artifact_basename(phase["phase_id"], phase["title"], "json"),
        _phase_artifact_basename(phase["phase_id"], phase["title"], "md"),
    ]
    json_path = output_dir / phase_payload["artifacts"][0]
    md_path = output_dir / phase_payload["artifacts"][1]
    json_path.unlink(missing_ok=True)
    md_path.unlink(missing_ok=True)
    json_path.write_text(json.dumps(_to_serializable(phase_payload), indent=2), encoding="utf-8")
    md_path.write_text(render_phase_markdown(phase_payload), encoding="utf-8")
    return json_path, md_path


def render_index_markdown(
    phases: dict[str, dict[str, Any]],
    program: dict[str, Any] | None = None,
) -> str:
    program = {} if program is None else dict(program)
    source = dict(program.get("source", {}))
    line = _as_mapping(program.get("line_config"))
    lines = [
        "# Production Phase Execution Index",
        "",
        "This file records the executed production phase stack for the current target profile.",
        "",
        "## Context",
        f"- `line_name`: `{line.get('line_name', 'UNSPECIFIED_COPV_LINE')}`",
        f"- `source_label`: `{source.get('label', 'unknown')}`",
        "",
        "## Phase Status",
    ]
    for phase_id in sorted(phases.keys()):
        phase = phases[phase_id]
        lines.append(
            f"- `{phase_id}`: `{phase['title']}` -> `{phase['status']}` | gate closed: `{phase.get('release_gate_closed')}`"
        )
    lines.append("")
    lines.append("## Release Summary")
    phase8 = phases.get("phase_08", {})
    release_ready = dict(phase8.get("metrics", {})).get("release_ready")
    lines.append(f"- Release ready: `{release_ready}`")
    return "\n".join(lines) + "\n"


def run_full_production_phase_pipeline(
    layout: dict[str, Any],
    summary: dict[str, Any],
    geom: GeometryConfig,
    line_config: ProductionLineConfig | None = None,
    planning_config: DiscreteCoursePlanningConfig | None = None,
    material: MaterialConfig | None = None,
    artifact_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    line = ProductionLineConfig() if line_config is None else line_config
    planner = DiscreteCoursePlanningConfig() if planning_config is None else planning_config
    phase_output_dir = Path("outputs") if artifact_output_dir is None else Path(artifact_output_dir)
    program = build_production_program_from_layout(
        layout=layout,
        geom=geom,
        line_config=line,
        summary=summary,
        source_label="outputs/winding_first_layout.json",
        include_path_points=bool(planner.emit_path_points),
        planning_config=planner,
    )
    discrete_plan = program["discrete_course_plan"]

    phases: dict[str, dict[str, Any]] = {}
    phases["phase_01"] = execute_phase_01_data_contract(program, line)
    phases["phase_02"] = execute_phase_02_discrete_planning(discrete_plan)
    phases["phase_03"] = execute_phase_03_machine_kinematics(program)
    phases["phase_04"] = execute_phase_04_towpreg_deposition(program)
    phases["phase_05"] = execute_phase_05_as_built_surrogate(program, material=material)
    phases["phase_06"] = execute_phase_06_process_state(program)
    phases["phase_07"] = execute_phase_07_inspection_and_traceability(program, phases["phase_05"], phase_output_dir)
    phases["phase_08"] = execute_phase_08_qualification_release(phases, program)

    return {
        "program": program,
        "discrete_plan": discrete_plan,
        "phases": phases,
        "artifact_output_dir": str(phase_output_dir),
    }


def export_full_production_phase_pipeline(output_dir: str | Path, pipeline: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Any] = {"phase_artifacts": {}}
    for phase_id, phase in pipeline.get("phases", {}).items():
        json_path, md_path = export_phase_artifacts(output_dir, phase)
        exported["phase_artifacts"][phase_id] = {
            "json": str(json_path),
            "md": str(md_path),
        }
    index_path = output_dir / "phase_execution_index.md"
    index_path.unlink(missing_ok=True)
    index_path.write_text(
        render_index_markdown(
            pipeline.get("phases", {}),
            program=pipeline.get("program", {}),
        ),
        encoding="utf-8",
    )
    exported["index"] = str(index_path)
    program_path = output_dir / "production_program_snapshot.json"
    program_path.unlink(missing_ok=True)
    program_path.write_text(json.dumps(_to_serializable(pipeline.get("program", {})), indent=2), encoding="utf-8")
    discrete_path = output_dir / "discrete_course_plan_snapshot.json"
    discrete_path.unlink(missing_ok=True)
    discrete_path.write_text(json.dumps(_to_serializable(pipeline.get("discrete_plan", {})), indent=2), encoding="utf-8")
    exported["program_snapshot"] = str(program_path)
    exported["discrete_plan_snapshot"] = str(discrete_path)
    return exported
