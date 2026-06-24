from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from copv_opt.config import FailureConfig, FrictionConfig, GeometryConfig, MaterialAllowables, MaterialConfig, WindingConfig
from copv_opt.geometry import ensure_copv_mesh
from copv_opt.optimize import winding_forward_angle
from copv_opt.physics import build_copv_fem_state, evaluate_hashin_failure, hashin_failure_indices, make_solve_compliance, rotate_stiffness_field


class HashinAnalyticTests(unittest.TestCase):
    """Fast unit tests — no mesh, no JAX compilation overhead."""

    def test_matrix_tension_at_allowable(self) -> None:
        import jax.numpy as jnp
        # sigma_22 = sigma_33 = YT = 70 MPa, all other stresses zero.
        # sigma_transverse = 0.5*(70+70) = 70 → fi_matrix_tension = (70/70)^2 = 1.0 exactly.
        stress = jnp.zeros((3, 3))
        stress = stress.at[1, 1].set(70.0).at[2, 2].set(70.0)
        result = hashin_failure_indices(stress, MaterialAllowables())
        self.assertAlmostEqual(float(result["matrix_tension"]), 1.0, places=5)
        self.assertAlmostEqual(float(result["fiber_tension"]), 0.0, places=5)
        self.assertAlmostEqual(float(result["fiber_compression"]), 0.0, places=5)

    def test_fiber_tension_at_allowable(self) -> None:
        import jax.numpy as jnp
        # sigma_11 = XT = 2200 MPa → fi_fiber_tension = (2200/2200)^2 = 1.0 exactly.
        stress = jnp.zeros((3, 3))
        stress = stress.at[0, 0].set(2200.0)
        result = hashin_failure_indices(stress, MaterialAllowables())
        self.assertAlmostEqual(float(result["fiber_tension"]), 1.0, places=5)
        self.assertAlmostEqual(float(result["matrix_tension"]), 0.0, places=5)

    def test_friction_safety_factor_reduces_effective_limit(self) -> None:
        cfg = FrictionConfig(mu_max=0.15, friction_safety_factor=0.85)
        effective = cfg.mu_max * cfg.friction_safety_factor
        self.assertAlmostEqual(effective, 0.1275, places=4)
        self.assertLess(effective, cfg.mu_max)

    def test_course_planner_warns_on_empty_layout(self) -> None:
        from copv_opt.course_planner import build_discrete_winding_plan_from_layout
        result = build_discrete_winding_plan_from_layout({}, GeometryConfig())
        self.assertIn("warnings", result)
        self.assertGreater(len(result["warnings"]), 0)


class ShellRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geom = GeometryConfig(pressure=6.85)
        cls.material = MaterialConfig()
        cls._tempdir = tempfile.TemporaryDirectory()
        work = Path(cls._tempdir.name)
        cls.mesh = ensure_copv_mesh(
            work / "copv_shell.step",
            work / "copv_shell.msh",
            cls.geom,
            remesh=True,
            rebuild_step=True,
        )
        cls.state = build_copv_fem_state(cls.mesh.nodes, cls.mesh.elems, cls.material, cls.geom)
        cls.solve = make_solve_compliance(cls.state)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tempdir.cleanup()

    def test_shell_state_uses_triangle_elements(self) -> None:
        self.assertEqual(self.mesh.elems.shape[1], 3)
        self.assertEqual(self.state["cell_type"], "triangle")
        self.assertEqual(np.asarray(self.state["outer_faces"]).shape[1], 3)
        self.assertGreater(int(self.state["element_count"]), 0)

    def test_constant_angle_shell_regression(self) -> None:
        result = winding_forward_angle(
            42.0,
            self.state,
            self.material,
            WindingConfig(band_thickness=8.0),
            self.geom,
            type(self).solve,
        )
        c_eff = (
            self.material.base_thickness
            * rotate_stiffness_field(self.state["c_mat"], self.state["meridian_dirs"], self.state["surface_normals"])
            + 8.0 * rotate_stiffness_field(self.state["c_mat"], result["fiber_dirs"], self.state["surface_normals"])
        ) / result["thickness"][:, None, None, None, None]
        failure = evaluate_hashin_failure(
            self.state,
            result["displacement"],
            c_eff,
            result["fiber_dirs"],
            FailureConfig(margin_of_safety=1.0),
        )
        fi_max = float(np.asarray(failure["fi_max"]))
        disp_max = float(np.max(np.linalg.norm(np.asarray(result["displacement"]).reshape(-1, 3), axis=1)))
        # 0.75 threshold: geodesic 42-deg constant winding at 8 mm band thickness clears the
        # Hashin limit with ~25% headroom; regression catches FEA model regressions.
        self.assertLessEqual(fi_max, 0.75)
        self.assertLessEqual(disp_max, 1.5)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "outputs" / "winding_first_summary.json").exists(),
        "winding_first_summary.json not present — run generate_winding_verification_outputs.py first",
    )
    def test_packaged_summary_is_shell_feasible(self) -> None:
        summary_path = PROJECT_ROOT / "outputs" / "winding_first_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["mesh"]["cell_type"], "triangle_shell")
        self.assertTrue(summary["winding"]["hashin_constraint_satisfied"])
        self.assertTrue(summary["winding"]["friction_constraint_satisfied"])
        self.assertLessEqual(float(summary["winding"]["fi_max"]), 1.05)
        self.assertLessEqual(
            float(summary["winding"]["friction_mu_max_required"]),
            float(summary["winding"]["friction_mu_allowable"]) + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
