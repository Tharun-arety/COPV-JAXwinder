from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np

from .config import GeometryConfig


@dataclass
class DiscreteCoursePlanningConfig:
    min_helical_course_length_mm: float = 80.0
    min_hoop_band_length_mm: float = 20.0
    merge_gap_length_mm: float = 16.0
    emit_path_points: bool = True
    circumferential_bias_warning_cv: float = 0.35
    helical_rmse_warning: float = 0.10
    hoop_rmse_warning: float = 0.15


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


def _segment_bounds(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=False)]


def _segment_length(sample_s: np.ndarray, start_idx: int, end_idx: int) -> float:
    if end_idx <= start_idx:
        return 0.0
    return float(sample_s[end_idx] - sample_s[start_idx])


def _merge_short_gaps(mask: np.ndarray, sample_s: np.ndarray, max_gap_length: float) -> np.ndarray:
    merged = np.asarray(mask, dtype=bool).copy()
    if merged.size == 0 or max_gap_length <= 0.0:
        return merged
    changed = True
    while changed:
        changed = False
        segments = _segment_bounds(merged)
        for left, right in zip(segments[:-1], segments[1:], strict=False):
            gap_length = float(sample_s[right[0]] - sample_s[left[1]])
            if gap_length <= max_gap_length:
                merged[left[1] : right[0] + 1] = True
                changed = True
                break
    return merged


def _remove_short_segments(mask: np.ndarray, sample_s: np.ndarray, min_length: float) -> np.ndarray:
    filtered = np.asarray(mask, dtype=bool).copy()
    if filtered.size == 0 or min_length <= 0.0:
        return filtered
    for start_idx, end_idx in _segment_bounds(filtered):
        if _segment_length(sample_s, start_idx, end_idx) < min_length:
            filtered[start_idx : end_idx + 1] = False
    return filtered


def _cleanup_mask(mask: np.ndarray, sample_s: np.ndarray, min_length: float, merge_gap_length: float) -> np.ndarray:
    cleaned = _merge_short_gaps(mask, sample_s, merge_gap_length)
    cleaned = _remove_short_segments(cleaned, sample_s, min_length)
    return cleaned


def _extract_segment_points(points: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(points[start_idx : end_idx + 1], dtype=np.float64)


def _build_path_lookup(paths: list[dict[str, Any]]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[str]]:
    warnings: list[str] = []
    clockwise: dict[int, np.ndarray] = {}
    counter_clockwise: dict[int, np.ndarray] = {}
    cw_index = 0
    ccw_index = 0
    for path in paths:
        handedness = str(path.get("handedness", "unknown"))
        points = np.asarray(path.get("points", []), dtype=np.float64)
        if handedness == "clockwise":
            cw_index += 1
            clockwise[cw_index] = points
        elif handedness == "counter_clockwise":
            ccw_index += 1
            counter_clockwise[ccw_index] = points
        else:
            warnings.append(f"Unknown handedness `{handedness}` in layout path export.")
    if cw_index != ccw_index:
        warnings.append(
            f"Handedness family mismatch in layout export: clockwise={cw_index}, counter_clockwise={ccw_index}."
        )
    return clockwise, counter_clockwise, warnings


def _build_helical_course_plan(
    layout: dict[str, Any],
    config: DiscreteCoursePlanningConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, dict[str, Any], list[str]]:
    warnings: list[str] = []
    sample_s = np.asarray(layout.get("sample_s", []), dtype=np.float64)
    if sample_s.size == 0:
        return [], [], np.zeros((0,), dtype=np.float64), {"continuous_profile_rmse": 0.0}, ["Layout sample_s is empty."]

    paths = list(layout.get("paths", []))
    family_count = int(layout.get("family_count", len(paths)))
    if family_count <= 0 or family_count % 2 != 0:
        warnings.append("Discrete helical planning requires an even family count; helical course plan omitted.")
        return [], [], np.zeros_like(sample_s), {"continuous_profile_rmse": 0.0}, warnings

    pair_count = family_count // 2
    clockwise, counter_clockwise, path_warnings = _build_path_lookup(paths)
    warnings.extend(path_warnings)
    max_pair_index = min(len(clockwise), len(counter_clockwise))
    if max_pair_index == 0:
        warnings.append("No balanced helical path pairs were found in the layout export.")
        return [], [], np.zeros_like(sample_s), {"continuous_profile_rmse": 0.0}, warnings

    continuous_profile = _optional_array(layout.get("helical_pass_profile"))
    if continuous_profile is None:
        warnings.append("Layout is missing helical_pass_profile; helical course plan omitted.")
        return [], [], np.zeros_like(sample_s), {"continuous_profile_rmse": 0.0}, warnings

    quantized_pair_count = np.rint(np.clip(continuous_profile, 0.0, None) * pair_count).astype(np.int32)
    max_pair_instances = int(np.max(quantized_pair_count)) if quantized_pair_count.size > 0 else 0
    quantized_profile = np.zeros_like(sample_s, dtype=np.float64)

    helical_course_pairs: list[dict[str, Any]] = []
    helical_courses: list[dict[str, Any]] = []
    pair_activation_length_mm = np.zeros((pair_count,), dtype=np.float64)
    pair_cut_restart_events = 0

    for pair_instance in range(max_pair_instances):
        pair_index = (pair_instance % pair_count) + 1
        if pair_index > max_pair_index:
            continue
        layer_index = (pair_instance // pair_count) + 1
        active_mask = quantized_pair_count >= (pair_instance + 1)
        active_mask = _cleanup_mask(
            active_mask,
            sample_s,
            min_length=float(config.min_helical_course_length_mm),
            merge_gap_length=float(config.merge_gap_length_mm),
        )
        if not np.any(active_mask):
            continue

        cw_points = clockwise[pair_index]
        ccw_points = counter_clockwise[pair_index]
        for segment_index, (start_idx, end_idx) in enumerate(_segment_bounds(active_mask), start=1):
            segment_length = _segment_length(sample_s, start_idx, end_idx)
            if segment_length <= 0.0:
                continue

            quantized_profile[start_idx : end_idx + 1] += 1.0 / pair_count
            pair_activation_length_mm[pair_index - 1] += segment_length

            start_at_boundary = start_idx == 0
            end_at_boundary = end_idx == sample_s.size - 1
            pair_cut_restart_events += (0 if start_at_boundary else 2) + (0 if end_at_boundary else 2)

            pair_id = f"HEL_PAIR_L{layer_index:02d}_P{pair_index:02d}_S{segment_index:02d}"
            pair_record = {
                "pair_id": pair_id,
                "layer_index": int(layer_index),
                "pair_index": int(pair_index),
                "segment_index": int(segment_index),
                "start_s_mm": float(sample_s[start_idx]),
                "stop_s_mm": float(sample_s[end_idx]),
                "segment_length_mm": segment_length,
                "pass_count_increment": float(1.0 / pair_count),
                "source_profile": "helical_pass_profile",
                "start_requires_cut": not start_at_boundary,
                "stop_requires_cut": not end_at_boundary,
                "clockwise_course_id": f"HEL_CW_L{layer_index:02d}_P{pair_index:02d}_S{segment_index:02d}",
                "counter_clockwise_course_id": f"HEL_CCW_L{layer_index:02d}_P{pair_index:02d}_S{segment_index:02d}",
            }
            helical_course_pairs.append(pair_record)

            for handedness, course_id, segment_points in (
                ("clockwise", pair_record["clockwise_course_id"], _extract_segment_points(cw_points, start_idx, end_idx)),
                (
                    "counter_clockwise",
                    pair_record["counter_clockwise_course_id"],
                    _extract_segment_points(ccw_points, start_idx, end_idx),
                ),
            ):
                helical_courses.append(
                    {
                        "course_id": course_id,
                        "pair_id": pair_id,
                        "family": "helical",
                        "handedness": handedness,
                        "layer_index": int(layer_index),
                        "pair_index": int(pair_index),
                        "segment_index": int(segment_index),
                        "start_s_mm": float(sample_s[start_idx]),
                        "stop_s_mm": float(sample_s[end_idx]),
                        "segment_length_mm": segment_length,
                        "start_requires_cut": not start_at_boundary,
                        "stop_requires_cut": not end_at_boundary,
                        "start_point_mm": None if segment_points.size == 0 else segment_points[0],
                        "end_point_mm": None if segment_points.size == 0 else segment_points[-1],
                        "path_points_mm": segment_points if config.emit_path_points else None,
                    }
                )

    activation_mean = float(np.mean(pair_activation_length_mm)) if pair_activation_length_mm.size > 0 else 0.0
    activation_std = float(np.std(pair_activation_length_mm)) if pair_activation_length_mm.size > 0 else 0.0
    activation_cv = 0.0 if activation_mean <= 1e-9 else activation_std / activation_mean
    rmse = float(np.sqrt(np.mean((quantized_profile - continuous_profile) ** 2))) if continuous_profile.size > 0 else 0.0

    if rmse > float(config.helical_rmse_warning):
        warnings.append(
            f"Discrete helical pass quantization RMSE is {rmse:.4f}, above the warning threshold {config.helical_rmse_warning:.4f}."
        )
    if activation_cv > float(config.circumferential_bias_warning_cv):
        warnings.append(
            f"Helical pair activation coefficient of variation is {activation_cv:.4f}; continuous-to-discrete conversion is still circumferentially biased."
        )
    if max_pair_instances % pair_count != 0:
        warnings.append(
            "The helical plan contains partial extra pass layers. This is a production-oriented discretization of an axisymmetric screening field, not a fully resolved circumferential laminate schedule."
        )

    metrics = {
        "continuous_profile_rmse": rmse,
        "pair_activation_length_mm": pair_activation_length_mm,
        "pair_activation_length_cv": activation_cv,
        "course_pair_count": len(helical_course_pairs),
        "individual_course_count": len(helical_courses),
        "cut_restart_event_count": int(pair_cut_restart_events),
    }
    return helical_course_pairs, helical_courses, quantized_profile, metrics, warnings


def _build_hoop_ring_plan(
    layout: dict[str, Any],
    geom: GeometryConfig,
    config: DiscreteCoursePlanningConfig,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any], list[str]]:
    warnings: list[str] = []
    sample_s = np.asarray(layout.get("sample_s", []), dtype=np.float64)
    if sample_s.size == 0:
        return [], np.zeros((0,), dtype=np.float64), {"continuous_profile_rmse": 0.0}, ["Layout sample_s is empty."]

    continuous_profile = _optional_array(layout.get("hoop_pass_profile"))
    if continuous_profile is None:
        warnings.append("Layout is missing hoop_pass_profile; hoop ring plan omitted.")
        return [], np.zeros_like(sample_s), {"continuous_profile_rmse": 0.0}, warnings

    quantized_ring_count = np.rint(np.clip(continuous_profile, 0.0, None)).astype(np.int32)
    quantized_profile = np.zeros_like(sample_s, dtype=np.float64)
    hoop_rings: list[dict[str, Any]] = []
    max_rings = int(np.max(quantized_ring_count)) if quantized_ring_count.size > 0 else 0
    circumference = float(2.0 * np.pi * geom.outer_radius)
    cut_restart_events = 0

    for ring_index in range(1, max_rings + 1):
        active_mask = quantized_ring_count >= ring_index
        active_mask = _cleanup_mask(
            active_mask,
            sample_s,
            min_length=float(config.min_hoop_band_length_mm),
            merge_gap_length=float(config.merge_gap_length_mm),
        )
        if not np.any(active_mask):
            continue
        for segment_index, (start_idx, end_idx) in enumerate(_segment_bounds(active_mask), start=1):
            segment_length = _segment_length(sample_s, start_idx, end_idx)
            if segment_length <= 0.0:
                continue
            quantized_profile[start_idx : end_idx + 1] += 1.0
            start_at_boundary = start_idx == 0
            end_at_boundary = end_idx == sample_s.size - 1
            cut_restart_events += (0 if start_at_boundary else 1) + (0 if end_at_boundary else 1)
            hoop_rings.append(
                {
                    "ring_id": f"HOOP_L{ring_index:02d}_S{segment_index:02d}",
                    "layer_index": int(ring_index),
                    "segment_index": int(segment_index),
                    "start_s_mm": float(sample_s[start_idx]),
                    "stop_s_mm": float(sample_s[end_idx]),
                    "segment_length_mm": segment_length,
                    "circumference_mm": circumference,
                    "start_requires_cut": not start_at_boundary,
                    "stop_requires_cut": not end_at_boundary,
                }
            )

    rmse = float(np.sqrt(np.mean((quantized_profile - continuous_profile) ** 2))) if continuous_profile.size > 0 else 0.0
    if rmse > float(config.hoop_rmse_warning):
        warnings.append(
            f"Discrete hoop ring quantization RMSE is {rmse:.4f}, above the warning threshold {config.hoop_rmse_warning:.4f}."
        )
    metrics = {
        "continuous_profile_rmse": rmse,
        "ring_count": len(hoop_rings),
        "cut_restart_event_count": int(cut_restart_events),
    }
    return hoop_rings, quantized_profile, metrics, warnings


def build_discrete_winding_plan_from_layout(
    layout: dict[str, Any],
    geom: GeometryConfig,
    config: DiscreteCoursePlanningConfig | None = None,
) -> dict[str, Any]:
    planner = DiscreteCoursePlanningConfig() if config is None else config
    sample_s = np.asarray(layout.get("sample_s", []), dtype=np.float64)
    helical_pairs, helical_courses, helical_quantized, helical_metrics, helical_warnings = _build_helical_course_plan(
        layout,
        planner,
    )
    hoop_rings, hoop_quantized, hoop_metrics, hoop_warnings = _build_hoop_ring_plan(layout, geom, planner)

    execution_sequence: list[dict[str, Any]] = []
    for pair in helical_pairs:
        execution_sequence.append(
            {
                "step_type": "helical_pair",
                "pair_id": pair["pair_id"],
                "clockwise_course_id": pair["clockwise_course_id"],
                "counter_clockwise_course_id": pair["counter_clockwise_course_id"],
            }
        )
    for ring in hoop_rings:
        execution_sequence.append({"step_type": "hoop_ring", "ring_id": ring["ring_id"]})

    helical_continuous = _optional_array(layout.get("helical_pass_profile"))
    if helical_continuous is None:
        helical_continuous = np.zeros_like(sample_s)
    hoop_continuous = _optional_array(layout.get("hoop_pass_profile"))
    if hoop_continuous is None:
        hoop_continuous = np.zeros_like(sample_s)

    return {
        "plan_type": "discrete_winding_course_plan",
        "planner_config": _to_serializable(planner),
        "sample_s_mm": sample_s,
        "continuous_helical_pass_profile": helical_continuous,
        "quantized_helical_pass_profile": helical_quantized,
        "continuous_hoop_pass_profile": hoop_continuous,
        "quantized_hoop_pass_profile": hoop_quantized,
        "helical_course_pairs": helical_pairs,
        "helical_courses": helical_courses,
        "hoop_rings": hoop_rings,
        "execution_sequence": execution_sequence,
        "metrics": {
            "helical": helical_metrics,
            "hoop": hoop_metrics,
            "total_course_pairs": len(helical_pairs),
            "total_individual_courses": len(helical_courses),
            "total_hoop_rings": len(hoop_rings),
            "total_cut_restart_events": int(
                helical_metrics.get("cut_restart_event_count", 0) + hoop_metrics.get("cut_restart_event_count", 0)
            ),
        },
        "warnings": helical_warnings + hoop_warnings,
    }


def export_discrete_winding_plan(path: str | Path, plan: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_serializable(plan), indent=2), encoding="utf-8")
    return path


def render_discrete_winding_plan_markdown(plan: dict[str, Any]) -> str:
    metrics = dict(plan.get("metrics", {}))
    helical_metrics = dict(metrics.get("helical", {}))
    hoop_metrics = dict(metrics.get("hoop", {}))
    warnings = list(plan.get("warnings", []))
    lines = [
        "# Discrete Winding Course Plan",
        "",
        "## Summary",
        f"- Helical course pairs: `{metrics.get('total_course_pairs', 0)}`",
        f"- Individual helical courses: `{metrics.get('total_individual_courses', 0)}`",
        f"- Hoop rings: `{metrics.get('total_hoop_rings', 0)}`",
        f"- Total cut/restart events: `{metrics.get('total_cut_restart_events', 0)}`",
        f"- Helical pass quantization RMSE: `{helical_metrics.get('continuous_profile_rmse', 0.0)}`",
        f"- Hoop pass quantization RMSE: `{hoop_metrics.get('continuous_profile_rmse', 0.0)}`",
        f"- Helical activation imbalance CV: `{helical_metrics.get('pair_activation_length_cv', 0.0)}`",
        "",
        "## Execution Sequence",
    ]
    for index, step in enumerate(plan.get("execution_sequence", []), start=1):
        if step.get("step_type") == "helical_pair":
            lines.append(
                f"{index}. Run pair `{step['pair_id']}` using `{step['clockwise_course_id']}` and `{step['counter_clockwise_course_id']}`."
            )
        elif step.get("step_type") == "hoop_ring":
            lines.append(f"{index}. Run hoop ring `{step['ring_id']}`.")
        else:
            lines.append(f"{index}. {step}")
    lines.extend(["", "## Warnings"])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def save_discrete_winding_plan_markdown(path: str | Path, plan: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_discrete_winding_plan_markdown(plan), encoding="utf-8")
    return path
