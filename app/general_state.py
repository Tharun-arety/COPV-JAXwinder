"""Build an FEA state from an arbitrary axisymmetric meridian.

Mirrors copv_opt.physics.build_copv_fem_state key-for-key, but takes surface
geometry from a :class:`app.meridian.MeridianProfile` instead of the analytic COPV
projection. The membrane/bending kinematics, material tensor, and the entire solver
are reused unchanged — only the surface geometry and boundary conditions are general.

The produced state dict is a drop-in for make_solve_compliance, baseline_response,
evaluate_hashin_failure, and the winding forward model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import jax.numpy as jnp

from app.meridian import MeridianProfile
from copv_opt.config import MaterialConfig
from copv_opt.physics import (
    _curvature_tensor_in_element_basis,
    _shell_bending_edges,
    _triangle_geometry,
    _triangle_membrane_kinematics,
    base_material_tensor,
)


def _orient_faces(nodes: np.ndarray, faces: np.ndarray, profile: MeridianProfile) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int32).copy()
    tri = nodes[faces]
    centroids = tri.mean(axis=1)
    target = profile.project(centroids)["normals"]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", fn, target) < 0.0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def _pressure_forces(nodes: np.ndarray, faces: np.ndarray, profile: MeridianProfile, pressure: float) -> np.ndarray:
    tri = nodes[faces]
    centroids = tri.mean(axis=1)
    normals = profile.project(centroids)["normals"]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    face_forces = pressure * areas[:, None] * normals
    forces = np.zeros((len(nodes), 3), dtype=np.float64)
    for idx in range(3):
        np.add.at(forces, faces[:, idx], face_forces / 3.0)
    return forces


def _opening_support_mask(nodes: np.ndarray, profile: MeridianProfile, support_tol: float) -> np.ndarray:
    """Pin nodes near either meridian opening (Euclidean in the (rho, z) plane)."""
    rho = np.hypot(nodes[:, 0], nodes[:, 1])
    z = nodes[:, 2]
    top_r, top_z = profile.top_opening
    bot_r, bot_z = profile.bottom_opening
    d_top = np.hypot(rho - top_r, z - top_z)
    d_bot = np.hypot(rho - bot_r, z - bot_z)
    return (d_top <= support_tol) | (d_bot <= support_tol)


def build_general_fem_state(
    nodes: np.ndarray,
    elems: np.ndarray,
    material: MaterialConfig,
    profile: MeridianProfile,
    pressure: float,
    support_tol: float = 1.5,
) -> dict:
    nodes = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(elems, dtype=np.int32)
    if elems.ndim != 2 or elems.shape[1] != 3:
        raise ValueError("build_general_fem_state expects a 2D triangular shell mesh")

    elems = _orient_faces(nodes, elems, profile)
    areas, element_basis, dndx, dndy = _triangle_geometry(nodes, elems)
    centroids = nodes[elems].mean(axis=1)
    surf = profile.project(centroids)
    k_meridian, k_hoop = profile.principal_curvatures(surf["s"])
    curvature_tensor = _curvature_tensor_in_element_basis(
        k_meridian, k_hoop, surf["meridian_dirs"], surf["hoop_dirs"], element_basis
    )
    b = _triangle_membrane_kinematics(element_basis, dndx, dndy, curvature_tensor)
    elem_dofs = (elems[:, :, None] * 3 + np.arange(3, dtype=np.int32)).reshape(len(elems), 9)
    n_dof = len(nodes) * 3

    support_mask = _opening_support_mask(nodes, profile, support_tol)
    fixed_mask = np.tile(support_mask[:, None], (1, 3)).reshape(-1)
    free_dofs = np.setdiff1d(np.arange(n_dof), np.where(fixed_mask)[0]).astype(np.int32)

    forces = _pressure_forces(nodes, elems, profile, pressure).reshape(-1)
    node_normals = profile.project(nodes)["normals"]
    bend_edges, bend_edge_lengths, bend_edge_owners = _shell_bending_edges(nodes, elems)

    return {
        "nodes_np": nodes,
        "elems_np": elems,
        "cell_type": "triangle",
        "element_count": int(len(elems)),
        "outer_faces": elems,
        "inner_faces": elems,
        "support_mask": support_mask,
        "n_dof": int(n_dof),
        "elem_dofs": jnp.asarray(elem_dofs),
        "free_dofs": jnp.asarray(free_dofs),
        "volumes": jnp.asarray(areas),
        "areas": jnp.asarray(areas),
        "b": jnp.asarray(b),
        "element_basis": jnp.asarray(element_basis),
        "node_normals": jnp.asarray(node_normals),
        "bend_edges": jnp.asarray(bend_edges),
        "bend_edge_lengths": jnp.asarray(bend_edge_lengths),
        "bend_edge_owners": jnp.asarray(bend_edge_owners),
        "forces_full": jnp.asarray(forces),
        "forces_free": jnp.asarray(forces[free_dofs]),
        "surface_points": jnp.asarray(surf["points"]),
        "surface_normals": jnp.asarray(surf["normals"]),
        "meridian_dirs": jnp.asarray(surf["meridian_dirs"]),
        "hoop_dirs": jnp.asarray(surf["hoop_dirs"]),
        "surface_rho": jnp.asarray(surf["rho"]),
        "s_coords": jnp.asarray(surf["s"]),
        "phi_coords": jnp.asarray(surf["phi"]),
        "c_mat": jnp.asarray(base_material_tensor(material)),
    }
