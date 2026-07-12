"""copv_opt — differentiable winding-first design screening for COPVs.

Public API is re-exported lazily (PEP 562): ``from copv_opt import GeometryConfig``
works as before, but importing the package — or a light pure-NumPy module such as
``copv_opt.clt`` / ``copv_opt.netting`` — no longer drags JAX, gmsh, PyVista, and
matplotlib in. Heavy dependencies load only when a symbol that needs them is first
accessed.
"""

from __future__ import annotations

import importlib

__version__ = "0.2.0"

# name -> submodule that provides it
_EXPORTS: dict[str, str] = {
    # abaqus_exporter
    "export_result_to_abaqus": "abaqus_exporter",
    "export_to_abaqus": "abaqus_exporter",
    # config
    "FailureConfig": "config",
    "FrictionConfig": "config",
    "GeometryConfig": "config",
    "MaterialAllowables": "config",
    "MaterialConfig": "config",
    "WindingOptimizationConfig": "config",
    "WindingConfig": "config",
    # geometry
    "MeshResult": "geometry",
    "build_copv_shell": "geometry",
    "ensure_copv_mesh": "geometry",
    "mesh_step": "geometry",
    "read_msh": "geometry",
    # optimize
    "run_winding_optimization": "optimize",
    "run_winding_angle_sweep": "optimize",
    "winding_forward_angle": "optimize",
    # physics
    "baseline_response": "physics",
    "build_copv_fem_state": "physics",
    "element_strain_stress": "physics",
    "estimate_burst_pressure_profile": "physics",
    "evaluate_hashin_failure": "physics",
    "friction_penalty": "physics",
    "hashin_failure_indices": "physics",
    "hashin_failure_indices_np": "physics",
    "make_solve_compliance": "physics",
    "required_friction_coefficient": "physics",
    # course_planner
    "DiscreteCoursePlanningConfig": "course_planner",
    "build_discrete_winding_plan_from_layout": "course_planner",
    "export_discrete_winding_plan": "course_planner",
    "render_discrete_winding_plan_markdown": "course_planner",
    "save_discrete_winding_plan_markdown": "course_planner",
    # production
    "AutofrettageConfig": "production",
    "CureCycleConfig": "production",
    "CureCycleStep": "production",
    "HeatingCompactionConfig": "production",
    "InspectionConfig": "production",
    "ProductionLineConfig": "production",
    "QualificationConfig": "production",
    "TowpregMaterialConfig": "production",
    "WindingMachineConfig": "production",
    "build_production_gap_report": "production",
    "production_line_config_from_mapping": "production",
    "production_line_config_to_dict": "production",
    "build_production_program_from_layout": "production",
    "build_production_program_from_result": "production",
    "export_production_program": "production",
    "render_production_readiness_markdown": "production",
    "save_production_readiness_markdown": "production",
    # production_pipeline
    "execute_phase_01_data_contract": "production_pipeline",
    "execute_phase_02_discrete_planning": "production_pipeline",
    "execute_phase_03_machine_kinematics": "production_pipeline",
    "execute_phase_04_towpreg_deposition": "production_pipeline",
    "execute_phase_05_as_built_surrogate": "production_pipeline",
    "execute_phase_06_process_state": "production_pipeline",
    "execute_phase_07_inspection_and_traceability": "production_pipeline",
    "execute_phase_08_qualification_release": "production_pipeline",
    "export_full_production_phase_pipeline": "production_pipeline",
    "render_index_markdown": "production_pipeline",
    "render_phase_markdown": "production_pipeline",
    "run_full_production_phase_pipeline": "production_pipeline",
    # visualize (optional dependency: pyvista/matplotlib — ImportError surfaces on access)
    "build_variable_angle_winding_paths": "visualize",
    "build_winding_process_layout_data": "visualize",
    "build_winding_layout_data": "visualize",
    "compare_vtu_side_by_side": "visualize",
    "load_layout_json": "visualize",
    "plot_winding_paths": "visualize",
    "plot_winding_process_paths": "visualize",
    "render_explicit_manufacturing_layout": "visualize",
    "render_explicit_manufacturing_layout_image": "visualize",
    "render_vtu_interactive": "visualize",
    "render_vtu_scalar_image": "visualize",
    "save_explicit_manufacturing_layout_screenshot": "visualize",
    "save_layout_json": "visualize",
    "save_vtu_comparison_screenshot": "visualize",
    "save_vtu_screenshot": "visualize",
    "write_vtu": "visualize",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'copv_opt' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
