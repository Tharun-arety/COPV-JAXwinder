from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import GeometryConfig, MaterialConfig
from .geometry import classify_copv_boundary_faces, copv_normals_np


def _boundary_faces_with_owners(elems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    elems = np.asarray(elems, dtype=np.int32)
    face_templates = np.array(
        [
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
        ],
        dtype=np.int32,
    )
    faces = elems[:, face_templates].reshape(-1, 3)
    face_owner = np.repeat(np.arange(len(elems), dtype=np.int32), 4)
    sorted_faces = np.sort(faces, axis=1)
    _, unique_idx, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
    boundary_idx = unique_idx[counts == 1]
    return faces[boundary_idx], face_owner[boundary_idx]


def _orient_outer_faces(nodes: np.ndarray, faces: np.ndarray, geom: GeometryConfig) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int32).copy()
    tri_pts = nodes[faces]
    centroids = tri_pts.mean(axis=1)
    target_normals = copv_normals_np(centroids, geom.outer_radius, geom.cylinder_length)
    face_normals = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    flip_mask = np.einsum("ij,ij->i", face_normals, target_normals) < 0.0
    faces[flip_mask] = faces[flip_mask][:, [0, 2, 1]]
    return faces


def _surface_shell_mesh(
    nodes: np.ndarray,
    elems: np.ndarray,
    geom: GeometryConfig,
) -> dict[str, np.ndarray]:
    boundary_faces, owners = _boundary_faces_with_owners(elems)
    _, outer_mask, _ = classify_copv_boundary_faces(nodes, boundary_faces, geom)
    outer_faces = _orient_outer_faces(nodes, boundary_faces[outer_mask], geom)
    outer_owners = owners[outer_mask]

    unique_nodes = np.unique(outer_faces.reshape(-1))
    node_lookup = {int(node_id): idx + 1 for idx, node_id in enumerate(unique_nodes)}
    shell_nodes = np.asarray(nodes[unique_nodes], dtype=np.float64)
    shell_faces = np.vectorize(node_lookup.get, otypes=[np.int32])(outer_faces)
    return {
        "node_ids": unique_nodes,
        "nodes": shell_nodes,
        "faces": shell_faces,
        "source_faces": outer_faces,
        "owners": outer_owners,
    }


def _normalize_np(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True).clip(min=eps)


def export_to_abaqus(
    output_path: str | Path,
    nodes: np.ndarray,
    elems: np.ndarray,
    thickness: np.ndarray,
    fiber_dirs: np.ndarray,
    geom: GeometryConfig,
    material: MaterialConfig | None = None,
    material_name: str = "CFRP",
    integration_points: int = 3,
    heading: str = "COPV Optimized Shell Export",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(elems, dtype=np.int32)
    thickness = np.asarray(thickness, dtype=np.float64).reshape(-1)
    fiber_dirs = np.asarray(fiber_dirs, dtype=np.float64)
    if len(thickness) != len(elems):
        raise ValueError("thickness must be defined per tetrahedral element")
    if fiber_dirs.shape != (len(elems), 3):
        raise ValueError("fiber_dirs must have shape (element_count, 3)")

    material = MaterialConfig() if material is None else material
    shell = _surface_shell_mesh(nodes, elems, geom)
    shell_nodes = shell["nodes"]
    shell_faces = shell["faces"]
    source_faces = shell["source_faces"]
    owners = shell["owners"]
    face_pts = nodes[source_faces]
    centroids = face_pts.mean(axis=1)
    normals = _normalize_np(np.cross(face_pts[:, 1] - face_pts[:, 0], face_pts[:, 2] - face_pts[:, 0]))

    owner_thickness = thickness[owners]
    owner_fibers = fiber_dirs[owners]
    owner_fibers = owner_fibers - np.sum(owner_fibers * normals, axis=1, keepdims=True) * normals
    fallback = _normalize_np(face_pts[:, 1] - face_pts[:, 0])
    fiber_norm = np.linalg.norm(owner_fibers, axis=1, keepdims=True)
    owner_fibers = np.where(fiber_norm > 1e-10, owner_fibers / fiber_norm.clip(min=1e-10), fallback)
    tangent_2 = _normalize_np(np.cross(normals, owner_fibers))

    with output_path.open("w", encoding="utf-8") as stream:
        stream.write("*HEADING\n")
        stream.write(f"{heading}\n")
        stream.write("** Exported from copv-optimizer-fw\n")
        stream.write("** Surface shell extracted from the outer boundary of the tetrahedral COPV mesh.\n")
        stream.write("*NODE\n")
        for node_idx, point in enumerate(shell_nodes, start=1):
            stream.write(f"{node_idx}, {point[0]:.8f}, {point[1]:.8f}, {point[2]:.8f}\n")

        stream.write("*ELEMENT, TYPE=S3\n")
        for elem_idx, face in enumerate(shell_faces, start=1):
            stream.write(f"{elem_idx}, {face[0]}, {face[1]}, {face[2]}\n")

        stream.write(f"*MATERIAL, NAME={material_name}\n")
        stream.write("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n")
        stream.write(
            f"{material.e_xx:.6f}, {material.e_yy:.6f}, {material.e_zz:.6f}, "
            f"{material.nu_xy:.6f}, {material.nu_xz:.6f}, {material.nu_yz:.6f}, "
            f"{material.g_xy:.6f}, {material.g_xz:.6f}, {material.g_yz:.6f}\n"
        )
        if getattr(material, "density", None) is not None:
            stream.write("*DENSITY\n")
            stream.write(f"{float(material.density):.12f}\n")

        for elem_idx, (center, e1, e2, t) in enumerate(
            zip(centroids, owner_fibers, tangent_2, owner_thickness),
            start=1,
        ):
            a_point = center + e1
            b_point = center + e2
            stream.write(f"** SOURCE_TET={int(owners[elem_idx - 1]) + 1}\n")
            stream.write(f"*ELSET, ELSET=ELEM_{elem_idx}\n")
            stream.write(f"{elem_idx}\n")
            stream.write(f"*ORIENTATION, NAME=ORI_{elem_idx}, SYSTEM=RECTANGULAR\n")
            stream.write(
                f"{a_point[0]:.8f}, {a_point[1]:.8f}, {a_point[2]:.8f}, "
                f"{b_point[0]:.8f}, {b_point[1]:.8f}, {b_point[2]:.8f}\n"
            )
            stream.write("3, 0.\n")
            stream.write(f"*SHELL SECTION, ELSET=ELEM_{elem_idx}, COMPOSITE, ORIENTATION=ORI_{elem_idx}\n")
            stream.write(f"{max(float(t), 1e-6):.8f}, {integration_points}, {material_name}, 0.\n")

    return output_path


def export_result_to_abaqus(
    state: dict[str, Any],
    result: dict[str, Any],
    geom: GeometryConfig,
    output_path: str | Path,
    material: MaterialConfig | None = None,
    material_name: str = "CFRP",
    integration_points: int = 3,
    heading: str = "COPV Optimized Shell Export",
) -> Path:
    return export_to_abaqus(
        output_path=output_path,
        nodes=np.asarray(state["nodes_np"]),
        elems=np.asarray(state["elems_np"]),
        thickness=np.asarray(result["thickness"]),
        fiber_dirs=np.asarray(result["fiber_dirs"]),
        geom=geom,
        material=material,
        material_name=material_name,
        integration_points=integration_points,
        heading=heading,
    )
