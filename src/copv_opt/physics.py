from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .config import FailureConfig, FrictionConfig, GeometryConfig, MaterialAllowables, MaterialConfig
from .geometry import (
    copv_opening_z,
    copv_normals_np,
    copv_principal_curvatures_np,
    copv_surface_projection_np,
    normalize_jax,
    orient_shell_faces,
)


VM = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
MEMBRANE_VOIGT = np.asarray([0, 1, 5], dtype=np.int32)
TRANSVERSE_VOIGT = np.asarray([2, 3, 4], dtype=np.int32)


def orthotropic_stiffness_matrix(material: MaterialConfig) -> np.ndarray:
    e1 = material.e_xx
    e2 = material.e_yy
    e3 = material.e_zz
    nu12 = material.nu_xy
    nu13 = material.nu_xz
    nu23 = material.nu_yz
    g23 = material.g_yz
    g13 = material.g_xz
    g12 = material.g_xy

    nu21 = nu12 * e2 / e1
    nu31 = nu13 * e3 / e1
    nu32 = nu23 * e3 / e2
    compliance = np.array(
        [
            [1.0 / e1, -nu21 / e2, -nu31 / e3, 0.0, 0.0, 0.0],
            [-nu12 / e1, 1.0 / e2, -nu32 / e3, 0.0, 0.0, 0.0],
            [-nu13 / e1, -nu23 / e2, 1.0 / e3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / g23, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / g13, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / g12],
        ],
        dtype=np.float64,
    )
    return np.linalg.inv(compliance)


def d6_to_c4(c6: np.ndarray) -> np.ndarray:
    c4 = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for a, (i, j) in enumerate(VM):
        for b, (k, l) in enumerate(VM):
            value = c6[a, b]
            c4[i, j, k, l] = value
            c4[j, i, k, l] = value
            c4[i, j, l, k] = value
            c4[j, i, l, k] = value
    return c4


def base_material_tensor(material: MaterialConfig) -> np.ndarray:
    return d6_to_c4(orthotropic_stiffness_matrix(material))


def c4_to_d6(c_tensor: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [jnp.stack([c_tensor[:, i, j, k, l] for k, l in VM], axis=-1) for i, j in VM],
        axis=-2,
    )


def engineering_strain_to_tensor_field(strain: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.stack([strain[..., 0], 0.5 * strain[..., 5], 0.5 * strain[..., 4]], axis=-1),
            jnp.stack([0.5 * strain[..., 5], strain[..., 1], 0.5 * strain[..., 3]], axis=-1),
            jnp.stack([0.5 * strain[..., 4], 0.5 * strain[..., 3], strain[..., 2]], axis=-1),
        ],
        axis=-2,
    )


def stress_voigt_to_tensor_field(stress: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.stack([stress[..., 0], stress[..., 5], stress[..., 4]], axis=-1),
            jnp.stack([stress[..., 5], stress[..., 1], stress[..., 3]], axis=-1),
            jnp.stack([stress[..., 4], stress[..., 3], stress[..., 2]], axis=-1),
        ],
        axis=-2,
    )


def tensor_to_engineering_voigt_field(tensor: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [
            tensor[..., 0, 0],
            tensor[..., 1, 1],
            tensor[..., 2, 2],
            2.0 * tensor[..., 1, 2],
            2.0 * tensor[..., 0, 2],
            2.0 * tensor[..., 0, 1],
        ],
        axis=-1,
    )


def tensor_to_stress_voigt_field(tensor: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [
            tensor[..., 0, 0],
            tensor[..., 1, 1],
            tensor[..., 2, 2],
            tensor[..., 1, 2],
            tensor[..., 0, 2],
            tensor[..., 0, 1],
        ],
        axis=-1,
    )


def rotate_stiffness_field(c_mat: jnp.ndarray, fiber_dirs: jnp.ndarray, normals: jnp.ndarray) -> jnp.ndarray:
    e3 = normalize_jax(normals)
    e1 = fiber_dirs - jnp.sum(fiber_dirs * e3, axis=-1, keepdims=True) * e3
    e1 = normalize_jax(e1)
    e2 = normalize_jax(jnp.cross(e3, e1))
    r = jnp.stack([e1, e2, e3], axis=-1)
    return jnp.einsum("abcd,eia,ejb,ekc,eld->eijkl", c_mat, r, r, r, r)


def local_frame_from_fiber(fiber_dirs: jnp.ndarray, normals: jnp.ndarray) -> jnp.ndarray:
    e3 = normalize_jax(normals)
    ref_x = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0], dtype=fiber_dirs.dtype), fiber_dirs.shape)
    ref_y = jnp.broadcast_to(jnp.asarray([0.0, 1.0, 0.0], dtype=fiber_dirs.dtype), fiber_dirs.shape)
    ref = jnp.where(jnp.abs(e3[:, :1]) < 0.9, ref_x, ref_y)
    proj = fiber_dirs - jnp.sum(fiber_dirs * e3, axis=-1, keepdims=True) * e3
    proj_norm = jnp.linalg.norm(proj, axis=-1, keepdims=True)
    fallback = normalize_jax(jnp.cross(e3, ref))
    e1 = jnp.where(proj_norm > 1e-8, proj / proj_norm, fallback)
    e2 = normalize_jax(jnp.cross(e3, e1))
    return jnp.stack([e1, e2, e3], axis=-1)


def _triangle_geometry(
    nodes: np.ndarray,
    elems: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tri = np.asarray(nodes, dtype=np.float64)[np.asarray(elems, dtype=np.int32)]
    edge_12 = tri[:, 1] - tri[:, 0]
    edge_13 = tri[:, 2] - tri[:, 0]

    raw_normals = np.cross(edge_12, edge_13)
    normal_norm = np.linalg.norm(raw_normals, axis=1, keepdims=True).clip(min=1e-12)
    normals = raw_normals / normal_norm
    t1 = edge_12 / np.linalg.norm(edge_12, axis=1, keepdims=True).clip(min=1e-12)
    t2 = np.cross(normals, t1)
    t2 = t2 / np.linalg.norm(t2, axis=1, keepdims=True).clip(min=1e-12)
    basis = np.stack([t1, t2, normals], axis=-1)

    x1 = np.zeros((len(elems),), dtype=np.float64)
    y1 = np.zeros((len(elems),), dtype=np.float64)
    x2 = np.linalg.norm(edge_12, axis=1)
    y2 = np.zeros((len(elems),), dtype=np.float64)
    x3 = np.einsum("ei,ei->e", edge_13, t1)
    y3 = np.einsum("ei,ei->e", edge_13, t2)

    two_area = x2 * y3 - x3 * y2
    areas = 0.5 * np.abs(two_area)
    if np.any(areas <= 1e-10):
        raise ValueError("Degenerate shell triangle detected during membrane kinematics assembly")

    dndx = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1) / two_area[:, None]
    dndy = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1) / two_area[:, None]

    return areas, basis, dndx, dndy


def _curvature_tensor_in_element_basis(
    k_meridian: np.ndarray,
    k_hoop: np.ndarray,
    meridian_dirs: np.ndarray,
    hoop_dirs: np.ndarray,
    element_basis: np.ndarray,
) -> np.ndarray:
    meridian_dirs = np.asarray(meridian_dirs, dtype=np.float64)
    hoop_dirs = np.asarray(hoop_dirs, dtype=np.float64)
    element_basis = np.asarray(element_basis, dtype=np.float64)
    t1 = element_basis[:, :, 0]
    t2 = element_basis[:, :, 1]
    transform = np.stack(
        [
            np.stack([np.einsum("ei,ei->e", meridian_dirs, t1), np.einsum("ei,ei->e", meridian_dirs, t2)], axis=-1),
            np.stack([np.einsum("ei,ei->e", hoop_dirs, t1), np.einsum("ei,ei->e", hoop_dirs, t2)], axis=-1),
        ],
        axis=1,
    )
    curvature_mh = np.zeros((len(k_meridian), 2, 2), dtype=np.float64)
    curvature_mh[:, 0, 0] = np.asarray(k_meridian, dtype=np.float64)
    curvature_mh[:, 1, 1] = np.asarray(k_hoop, dtype=np.float64)
    return np.einsum("eai,eab,ebj->eij", transform, curvature_mh, transform)


def _triangle_membrane_kinematics(
    basis: np.ndarray,
    dndx: np.ndarray,
    dndy: np.ndarray,
    curvature_tensor: np.ndarray,
) -> np.ndarray:
    element_count = len(dndx)
    b_local = np.zeros((element_count, 3, 9), dtype=np.float64)
    curvature_vec = np.stack(
        [
            curvature_tensor[:, 0, 0],
            curvature_tensor[:, 1, 1],
            2.0 * curvature_tensor[:, 0, 1],
        ],
        axis=-1,
    )
    for idx in range(3):
        dof = 3 * idx
        b_local[:, 0, dof] = dndx[:, idx]
        b_local[:, 1, dof + 1] = dndy[:, idx]
        b_local[:, 2, dof] = dndy[:, idx]
        b_local[:, 2, dof + 1] = dndx[:, idx]
        b_local[:, :, dof + 2] = -(1.0 / 3.0) * curvature_vec

    t_map = np.zeros((element_count, 9, 9), dtype=np.float64)
    local_transform = np.swapaxes(np.asarray(basis, dtype=np.float64), 1, 2)
    for idx in range(3):
        dof = 3 * idx
        t_map[:, dof : dof + 3, dof : dof + 3] = local_transform

    return np.einsum("eij,ejk->eik", b_local, t_map)


def _shell_bending_edges(
    nodes: np.ndarray,
    elems: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elems = np.asarray(elems, dtype=np.int32)
    edges = np.concatenate(
        [
            elems[:, [0, 1]],
            elems[:, [1, 2]],
            elems[:, [2, 0]],
        ],
        axis=0,
    )
    owners = np.repeat(np.arange(len(elems), dtype=np.int32), 3)
    sort_order = np.argsort(edges, axis=1)
    sorted_edges = np.take_along_axis(edges, sort_order, axis=1)
    unique_edges, inverse = np.unique(sorted_edges, axis=0, return_inverse=True)
    edge_owners = -np.ones((len(unique_edges), 2), dtype=np.int32)
    fill = np.zeros((len(unique_edges),), dtype=np.int32)
    for edge_idx, owner in zip(inverse, owners, strict=False):
        slot = fill[edge_idx]
        if slot < 2:
            edge_owners[edge_idx, slot] = owner
        fill[edge_idx] += 1
    edge_lengths = np.linalg.norm(nodes[unique_edges[:, 1]] - nodes[unique_edges[:, 0]], axis=1)
    return unique_edges, edge_lengths.astype(np.float64), edge_owners


def build_pressure_forces(nodes: np.ndarray, faces: np.ndarray, geom: GeometryConfig) -> np.ndarray:
    tri_pts = np.asarray(nodes, dtype=np.float64)[np.asarray(faces, dtype=np.int32)]
    centroids = tri_pts.mean(axis=1)
    normals = copv_normals_np(
        centroids,
        geom.mid_radius,
        geom.cylinder_length,
        geom.opening_radius,
        dome_height_ratio=geom.dome_height_ratio,
    )
    areas = 0.5 * np.linalg.norm(np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]), axis=1)
    face_forces = geom.pressure * areas[:, None] * normals
    forces = np.zeros((len(nodes), 3), dtype=np.float64)
    for idx in range(3):
        np.add.at(forces, faces[:, idx], face_forces / 3.0)
    return forces


def build_copv_fem_state(
    nodes: np.ndarray,
    elems: np.ndarray,
    material: MaterialConfig,
    geom: GeometryConfig,
) -> dict[str, Any]:
    nodes = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(elems, dtype=np.int32)
    if elems.ndim != 2 or elems.shape[1] != 3:
        raise ValueError("build_copv_fem_state expects a 2D triangular shell mesh")

    elems = orient_shell_faces(
        nodes,
        elems,
        geom.mid_radius,
        geom.cylinder_length,
        geom.opening_radius,
        dome_height_ratio=geom.dome_height_ratio,
    )
    areas, element_basis, dndx, dndy = _triangle_geometry(nodes, elems)
    centroids = nodes[elems].mean(axis=1)
    surf = copv_surface_projection_np(
        centroids,
        geom.mid_radius,
        geom.cylinder_length,
        geom.opening_radius,
        dome_height_ratio=geom.dome_height_ratio,
    )
    k_meridian, k_hoop = copv_principal_curvatures_np(
        geom.mid_radius,
        surf["s"],
        geom.cylinder_length,
        geom.opening_radius,
        dome_height_ratio=geom.dome_height_ratio,
    )
    curvature_tensor = _curvature_tensor_in_element_basis(
        k_meridian,
        k_hoop,
        surf["meridian_dirs"],
        surf["hoop_dirs"],
        element_basis,
    )
    b = _triangle_membrane_kinematics(element_basis, dndx, dndy, curvature_tensor)
    elem_dofs = (elems[:, :, None] * 3 + np.arange(3, dtype=np.int32)).reshape(len(elems), 9)
    n_dof = len(nodes) * 3

    rho_nodes = np.linalg.norm(nodes[:, :2], axis=1)
    opening_z = copv_opening_z(
        geom.mid_radius,
        geom.cylinder_length,
        geom.opening_radius,
        geom.dome_height_ratio,
    )
    support_mask = (rho_nodes <= geom.opening_radius + geom.support_tol) & (np.abs(nodes[:, 2]) >= opening_z - geom.support_tol)
    fixed_mask = np.tile(support_mask[:, None], (1, 3)).reshape(-1)
    free_dofs = np.setdiff1d(np.arange(n_dof), np.where(fixed_mask)[0]).astype(np.int32)

    forces = build_pressure_forces(nodes, elems, geom).reshape(-1)
    node_normals = copv_normals_np(
        nodes,
        geom.mid_radius,
        geom.cylinder_length,
        geom.opening_radius,
        dome_height_ratio=geom.dome_height_ratio,
    )
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


def _membrane_constitutive_matrices(
    c_eff: jnp.ndarray,
    element_basis: jnp.ndarray,
) -> jnp.ndarray:
    c_local = jnp.einsum(
        "eia,ejb,ekc,eld,eabcd->eijkl",
        element_basis,
        element_basis,
        element_basis,
        element_basis,
        c_eff,
    )
    d_local = c4_to_d6(c_local)
    membrane_idx = jnp.asarray(MEMBRANE_VOIGT)
    transverse_idx = jnp.asarray(TRANSVERSE_VOIGT)
    d_aa = jnp.take(jnp.take(d_local, membrane_idx, axis=1), membrane_idx, axis=2)
    d_ab = jnp.take(jnp.take(d_local, membrane_idx, axis=1), transverse_idx, axis=2)
    d_ba = jnp.take(jnp.take(d_local, transverse_idx, axis=1), membrane_idx, axis=2)
    d_bb = jnp.take(jnp.take(d_local, transverse_idx, axis=1), transverse_idx, axis=2)
    eye = jnp.broadcast_to(jnp.eye(3, dtype=c_eff.dtype), d_bb.shape)
    d_bb_inv = jnp.linalg.inv(d_bb + 1e-8 * eye)
    return d_aa - jnp.einsum("eij,ejk,ekl->eil", d_ab, d_bb_inv, d_ba)


def _membrane_strain_stress_local(
    state: dict[str, Any],
    displacement: jnp.ndarray,
    c_eff: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    elem_dofs = state["elem_dofs"]
    b = state["b"]
    u_e = displacement[elem_dofs]
    strain_local = jnp.einsum("eij,ej->ei", b, u_e)
    q_membrane = _membrane_constitutive_matrices(c_eff, state["element_basis"])
    stress_local = jnp.einsum("eij,ej->ei", q_membrane, strain_local)
    return strain_local, stress_local


def element_strain_stress(
    state: dict[str, Any],
    displacement: jnp.ndarray,
    c_eff: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    strain_local, stress_local = _membrane_strain_stress_local(state, displacement, c_eff)
    zero = jnp.zeros((state["element_count"],), dtype=displacement.dtype)

    strain_local_tensor = jnp.stack(
        [
            jnp.stack([strain_local[:, 0], 0.5 * strain_local[:, 2], zero], axis=-1),
            jnp.stack([0.5 * strain_local[:, 2], strain_local[:, 1], zero], axis=-1),
            jnp.stack([zero, zero, zero], axis=-1),
        ],
        axis=-2,
    )
    stress_local_tensor = jnp.stack(
        [
            jnp.stack([stress_local[:, 0], stress_local[:, 2], zero], axis=-1),
            jnp.stack([stress_local[:, 2], stress_local[:, 1], zero], axis=-1),
            jnp.stack([zero, zero, zero], axis=-1),
        ],
        axis=-2,
    )

    basis = state["element_basis"]
    strain_global = jnp.einsum("eia,eab,ejb->eij", basis, strain_local_tensor, basis)
    stress_global = jnp.einsum("eia,eab,ejb->eij", basis, stress_local_tensor, basis)
    return tensor_to_engineering_voigt_field(strain_global), tensor_to_stress_voigt_field(stress_global)


# Used in the autodiff optimization loop — must stay JAX-traceable.
def hashin_failure_indices(local_stress: jnp.ndarray, allowables: MaterialAllowables) -> dict[str, jnp.ndarray]:
    sigma_11 = local_stress[..., 0, 0]
    sigma_transverse = 0.5 * (local_stress[..., 1, 1] + local_stress[..., 2, 2])
    tau_12 = local_stress[..., 0, 1]
    tau_13 = local_stress[..., 0, 2]
    tau_23 = local_stress[..., 1, 2]
    shear_fiber = tau_12**2 + tau_13**2
    shear_matrix = tau_12**2 + tau_23**2

    xt = jnp.asarray(allowables.xt, dtype=local_stress.dtype)
    xc = jnp.asarray(allowables.xc, dtype=local_stress.dtype)
    yt = jnp.asarray(allowables.yt, dtype=local_stress.dtype)
    yc = jnp.asarray(allowables.yc, dtype=local_stress.dtype)
    s = jnp.asarray(allowables.s, dtype=local_stress.dtype)

    fiber_tension = jnp.where(sigma_11 >= 0.0, (sigma_11 / xt) ** 2 + shear_fiber / (s**2), 0.0)
    fiber_compression = jnp.where(sigma_11 < 0.0, (sigma_11 / xc) ** 2, 0.0)
    matrix_tension = jnp.where(sigma_transverse >= 0.0, (sigma_transverse / yt) ** 2 + shear_matrix / (s**2), 0.0)
    matrix_compression_raw = (
        (sigma_transverse / (2.0 * s)) ** 2
        + ((yc / (2.0 * s)) ** 2 - 1.0) * (sigma_transverse / yc)
        + shear_matrix / (s**2)
    )
    matrix_compression = jnp.where(sigma_transverse < 0.0, jnp.maximum(matrix_compression_raw, 0.0), 0.0)
    failure_index = jnp.maximum(
        jnp.maximum(fiber_tension, fiber_compression),
        jnp.maximum(matrix_tension, matrix_compression),
    )
    return {
        "fiber_tension": fiber_tension,
        "fiber_compression": fiber_compression,
        "matrix_tension": matrix_tension,
        "matrix_compression": matrix_compression,
        "failure_index": failure_index,
    }


# Used in post-processing and reporting — not on the autodiff path.
def hashin_failure_indices_np(local_stress: np.ndarray, allowables: MaterialAllowables) -> dict[str, np.ndarray]:
    local_stress = np.asarray(local_stress, dtype=np.float64)
    sigma_11 = local_stress[..., 0, 0]
    sigma_transverse = 0.5 * (local_stress[..., 1, 1] + local_stress[..., 2, 2])
    tau_12 = local_stress[..., 0, 1]
    tau_13 = local_stress[..., 0, 2]
    tau_23 = local_stress[..., 1, 2]
    shear_fiber = tau_12**2 + tau_13**2
    shear_matrix = tau_12**2 + tau_23**2

    xt = float(allowables.xt)
    xc = float(allowables.xc)
    yt = float(allowables.yt)
    yc = float(allowables.yc)
    s = float(allowables.s)

    fiber_tension = np.where(sigma_11 >= 0.0, (sigma_11 / xt) ** 2 + shear_fiber / (s**2), 0.0)
    fiber_compression = np.where(sigma_11 < 0.0, (sigma_11 / xc) ** 2, 0.0)
    matrix_tension = np.where(sigma_transverse >= 0.0, (sigma_transverse / yt) ** 2 + shear_matrix / (s**2), 0.0)
    matrix_compression_raw = (
        (sigma_transverse / (2.0 * s)) ** 2
        + ((yc / (2.0 * s)) ** 2 - 1.0) * (sigma_transverse / yc)
        + shear_matrix / (s**2)
    )
    matrix_compression = np.where(sigma_transverse < 0.0, np.maximum(matrix_compression_raw, 0.0), 0.0)
    failure_index = np.maximum(
        np.maximum(fiber_tension, fiber_compression),
        np.maximum(matrix_tension, matrix_compression),
    )
    return {
        "fiber_tension": fiber_tension,
        "fiber_compression": fiber_compression,
        "matrix_tension": matrix_tension,
        "matrix_compression": matrix_compression,
        "failure_index": failure_index,
    }


def evaluate_hashin_failure(
    state: dict[str, Any],
    displacement: jnp.ndarray,
    c_eff: jnp.ndarray,
    fiber_dirs: jnp.ndarray,
    failure: FailureConfig,
) -> dict[str, jnp.ndarray]:
    strain_voigt, stress_voigt = element_strain_stress(state, displacement, c_eff)
    stress_tensor = stress_voigt_to_tensor_field(stress_voigt)
    strain_tensor = engineering_strain_to_tensor_field(strain_voigt)
    frame = local_frame_from_fiber(fiber_dirs, state["surface_normals"])
    local_stress = jnp.einsum("eai,eab,ebj->eij", frame, stress_tensor, frame)
    local_strain = jnp.einsum("eai,eab,ebj->eij", frame, strain_tensor, frame)
    metrics = hashin_failure_indices(local_stress, failure.allowables)
    failure_with_margin = metrics["failure_index"] * failure.margin_of_safety
    smooth_excess = jax.nn.softplus(failure.softplus_scale * (failure_with_margin - 1.0)) / failure.softplus_scale
    mean_penalty = jnp.mean(smooth_excess**2)
    flat_excess = jnp.reshape(smooth_excess, (-1,))
    tail_fraction = float(np.clip(failure.penalty_tail_fraction, 0.0, 1.0))
    tail_count = max(1, min(int(np.ceil(tail_fraction * max(int(flat_excess.shape[0]), 1))), int(flat_excess.shape[0])))
    tail_values = jax.lax.top_k(flat_excess, tail_count)[0]
    tail_penalty = jnp.mean(tail_values**2)
    worst_case_mix = float(np.clip(failure.penalty_worst_case_mix, 0.0, 1.0))
    penalty = failure.penalty_weight * ((1.0 - worst_case_mix) * mean_penalty + worst_case_mix * tail_penalty)
    return {
        **metrics,
        "strain_voigt": strain_voigt,
        "stress_voigt": stress_voigt,
        "local_stress": local_stress,
        "local_strain": local_strain,
        "failure_with_margin": failure_with_margin,
        "fi_max": jnp.max(metrics["failure_index"]),
        "fi_mean": jnp.mean(metrics["failure_index"]),
        "penalty": penalty,
        "failure_penalty_mean": mean_penalty,
        "failure_penalty_tail": tail_penalty,
    }


def estimate_burst_pressure_profile(
    local_stress: np.ndarray,
    allowables: MaterialAllowables,
    operating_pressure: float = 1.0,
    margin_of_safety: float = 1.0,
    scale_max: float = 8.0,
    num_points: int = 161,
    growth_factor: float = 2.0,
    max_expansions: int = 8,
) -> dict[str, np.ndarray | float]:
    local_stress = np.asarray(local_stress, dtype=np.float64)
    target_with_margin = 1.0 / max(float(margin_of_safety), 1e-12)
    target_burst = 1.0
    current_scale_max = max(float(scale_max), 1e-6)

    def _curve(max_scale: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        pressure_factors = np.linspace(0.0, max_scale, int(max(num_points, 2)))
        scaled = pressure_factors[:, None, None, None] * local_stress[None, ...]
        metrics = hashin_failure_indices_np(scaled, allowables)
        return pressure_factors, metrics

    pressure_factors, metrics = _curve(current_scale_max)
    failure_curve = np.max(metrics["failure_index"], axis=1)
    expansions = 0
    while failure_curve[-1] < max(target_with_margin, target_burst) and expansions < max_expansions:
        current_scale_max *= max(float(growth_factor), 1.1)
        pressure_factors, metrics = _curve(current_scale_max)
        failure_curve = np.max(metrics["failure_index"], axis=1)
        expansions += 1

    mode_curves = {
        "fiber_tension": np.max(metrics["fiber_tension"], axis=1),
        "fiber_compression": np.max(metrics["fiber_compression"], axis=1),
        "matrix_tension": np.max(metrics["matrix_tension"], axis=1),
        "matrix_compression": np.max(metrics["matrix_compression"], axis=1),
    }

    def _interpolate_crossing(curve: np.ndarray, target: float) -> float:
        idx = np.where(curve >= target)[0]
        if len(idx) == 0:
            return float(pressure_factors[-1])
        hi = int(idx[0])
        if hi == 0:
            return float(pressure_factors[0])
        lo = hi - 1
        x0 = float(pressure_factors[lo])
        x1 = float(pressure_factors[hi])
        y0 = float(curve[lo])
        y1 = float(curve[hi])
        if abs(y1 - y0) < 1e-12:
            return x1
        return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

    burst_factor = _interpolate_crossing(failure_curve, target_burst)
    allowable_factor_with_margin = _interpolate_crossing(failure_curve, target_with_margin)
    burst_index = int(np.argmin(np.abs(pressure_factors - burst_factor)))
    mode_at_burst = {
        "fiber_tension": float(mode_curves["fiber_tension"][burst_index]),
        "fiber_compression": float(mode_curves["fiber_compression"][burst_index]),
        "matrix_tension": float(mode_curves["matrix_tension"][burst_index]),
        "matrix_compression": float(mode_curves["matrix_compression"][burst_index]),
    }

    return {
        "pressure_factors": pressure_factors,
        "pressure_values": operating_pressure * pressure_factors,
        "failure_curve": failure_curve,
        "failure_curve_with_margin": failure_curve * float(margin_of_safety),
        "mode_curves": mode_curves,
        "burst_factor": float(burst_factor),
        "allowable_factor_with_margin": float(allowable_factor_with_margin),
        "burst_pressure": float(operating_pressure * burst_factor),
        "allowable_pressure_with_margin": float(operating_pressure * allowable_factor_with_margin),
        "mode_at_burst": mode_at_burst,
    }


def required_friction_coefficient(
    s_coords: jnp.ndarray,
    rho: jnp.ndarray,
    alpha: jnp.ndarray,
    regularization: float = 1e-6,
) -> jnp.ndarray:
    if s_coords.shape[0] < 2:
        return jnp.zeros((1,), dtype=alpha.dtype)
    clairaut = rho * jnp.sin(alpha)
    ds = jnp.maximum(jnp.diff(s_coords), regularization)
    dclairaut_ds = jnp.diff(clairaut) / ds
    rho_mid = 0.5 * (rho[1:] + rho[:-1])
    alpha_mid = 0.5 * (alpha[1:] + alpha[:-1])
    denom = jnp.maximum(jnp.abs(rho_mid * jnp.cos(alpha_mid)), regularization)
    return jnp.abs(dclairaut_ds) / denom


def friction_penalty(
    s_coords: jnp.ndarray,
    rho: jnp.ndarray,
    alpha: jnp.ndarray,
    config: FrictionConfig,
) -> dict[str, jnp.ndarray]:
    mu_required = required_friction_coefficient(
        s_coords,
        rho,
        alpha,
        regularization=config.mu_regularization,
    )
    effective_mu_limit = config.mu_max * config.friction_safety_factor
    excess = jax.nn.relu(mu_required - effective_mu_limit)
    penalty = config.penalty_weight * jnp.mean(excess**2)
    return {
        "mu_required": mu_required,
        "mu_max_required": jnp.max(mu_required),
        "penalty": penalty,
    }


def make_solve_compliance(state: dict[str, Any], tol: float = 1e-6, maxiter: int = 2400, bending_scale: float = 1.0):
    n_dof = state["n_dof"]
    elem_dofs = state["elem_dofs"]
    free_dofs = state["free_dofs"]
    areas = state["areas"]
    b = state["b"]
    element_basis = state["element_basis"]
    node_normals = state["node_normals"]
    bend_edges = state["bend_edges"]
    bend_edge_lengths = state["bend_edge_lengths"]
    bend_edge_owners = state["bend_edge_owners"]
    forces_free = state["forces_free"]
    forces_full = state["forces_full"]
    @jax.jit
    def _solve(c_eff: jnp.ndarray, thickness: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        q_membrane = _membrane_constitutive_matrices(c_eff, element_basis)
        thickness_field = jnp.broadcast_to(jnp.reshape(thickness, (-1,)), areas.shape)
        bend_owner_a = bend_edge_owners[:, 0]
        bend_owner_b = bend_edge_owners[:, 1]
        owner_b_safe = jnp.maximum(bend_owner_b, 0)
        has_owner_b = bend_owner_b >= 0
        membrane_modulus = jnp.sqrt(jnp.maximum(q_membrane[:, 0, 0] * q_membrane[:, 1, 1], 1e-6))
        mod_a = membrane_modulus[bend_owner_a]
        mod_b = jnp.where(has_owner_b, membrane_modulus[owner_b_safe], mod_a)
        th_a = thickness_field[bend_owner_a]
        th_b = jnp.where(has_owner_b, thickness_field[owner_b_safe], th_a)
        edge_modulus = 0.5 * (mod_a + mod_b)
        edge_thickness = 0.5 * (th_a + th_b)
        edge_bending = bending_scale * edge_modulus * edge_thickness**3 / (12.0 * jnp.maximum(bend_edge_lengths**2, 1e-6))

        def matvec(u_free: jnp.ndarray) -> jnp.ndarray:
            u_full = jnp.zeros((n_dof,), dtype=c_eff.dtype).at[free_dofs].set(u_free)
            u_e = u_full[elem_dofs]
            strain = jnp.einsum("eij,ej->ei", b, u_e)
            stress = jnp.einsum("eij,ej->ei", q_membrane, strain)
            fint_e = (areas * thickness_field)[:, None] * jnp.einsum("eji,ej->ei", b, stress)
            fint_full = jnp.zeros((n_dof,), dtype=c_eff.dtype).at[elem_dofs.reshape(-1)].add(fint_e.reshape(-1))

            u_nodes = u_full.reshape((-1, 3))
            normal_disp = jnp.einsum("ni,ni->n", u_nodes, node_normals)
            edge_i = bend_edges[:, 0]
            edge_j = bend_edges[:, 1]
            edge_dw = normal_disp[edge_i] - normal_disp[edge_j]
            edge_force = edge_bending * edge_dw
            bending_full = jnp.zeros_like(u_nodes)
            bending_full = bending_full.at[edge_i].add(edge_force[:, None] * node_normals[edge_i])
            bending_full = bending_full.at[edge_j].add(-edge_force[:, None] * node_normals[edge_j])
            fint_full = fint_full + bending_full.reshape(-1)
            return fint_full[free_dofs]

        u_free, _ = jsp.sparse.linalg.cg(matvec, forces_free, tol=tol, maxiter=maxiter)
        u_full = jnp.zeros((n_dof,), dtype=c_eff.dtype).at[free_dofs].set(u_free)
        compliance = jnp.vdot(u_full, forces_full)
        return compliance, u_full

    def solve(c_eff: jnp.ndarray, thickness: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        if thickness is None:
            thickness = jnp.ones((int(state["element_count"]),), dtype=c_eff.dtype)
        return _solve(c_eff, thickness)

    return solve


def baseline_response(state: dict[str, Any], material: MaterialConfig, solve_compliance) -> dict[str, jnp.ndarray]:
    element_count = state["element_count"]
    base_thickness = material.base_thickness
    c_base = rotate_stiffness_field(state["c_mat"], state["meridian_dirs"], state["surface_normals"])
    thickness = jnp.full((element_count,), base_thickness)
    compliance, displacement = solve_compliance(c_base, thickness)
    density = jnp.ones((element_count,))
    coverage = jnp.zeros((element_count,))
    fiber_dirs = normalize_jax(state["meridian_dirs"] + 1e-8)
    mass_metric = jnp.sum(thickness * state["volumes"])
    return {
        "compliance": compliance,
        "displacement": displacement,
        "thickness": thickness,
        "density": density,
        "coverage": coverage,
        "fiber_dirs": fiber_dirs,
        "mass_metric": mass_metric,
    }
