"""Winding machine post-processor — turn a surface path into machine axis motions.

Inverse kinematics for the classical filament-winding machine (rotating mandrel on the
Z axis, translating carriage, delivery eye), the CNC side of a TaniqWind-Pro-class
post-processor. A path of 3-D points on the surface of revolution becomes:

    mandrel_deg  — accumulated mandrel rotation (unwrapped azimuth)
    carriage_mm  — axial carriage position (z)
    eye_yaw_deg  — delivery-eye yaw ~ local winding angle from the axis
    radius_mm    — local surface radius (for eye stand-off and path reconstruction)

This is a real 3-axis (mandrel + carriage + eye-yaw) CNC post. A robotic post and
machine-specific NC dialects need the actual machine definition and are not included.
Verified by reconstructing the surface path from the axis motions (machine.py tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class WindingProgram:
    mandrel_deg: np.ndarray   # (N,) accumulated mandrel rotation [deg]
    carriage_mm: np.ndarray   # (N,) axial carriage position [mm]
    eye_yaw_deg: np.ndarray   # (N,) delivery-eye yaw ~ winding angle from axis [deg]
    radius_mm: np.ndarray     # (N,) local surface radius [mm]
    axes: int = 3


def machine_program_from_path(points, axes: int = 3) -> WindingProgram:
    """Post-process a surface path (Nx3, mandrel axis = Z) into machine axis motions."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.hypot(x, y)
    phi = np.unwrap(np.arctan2(y, x))               # continuous azimuth
    mandrel_deg = np.degrees(phi)                   # mandrel rotation = azimuth of the lay point
    carriage_mm = z.copy()
    # local winding angle from the path tangent: axial component vs hoop (r*dphi) component
    dz = np.gradient(z)
    hoop = r * np.gradient(phi)
    eye_yaw_deg = np.degrees(np.arctan2(np.abs(hoop), np.abs(dz) + 1e-12))
    return WindingProgram(mandrel_deg=mandrel_deg, carriage_mm=carriage_mm,
                          eye_yaw_deg=eye_yaw_deg, radius_mm=r, axes=int(axes))


def reconstruct_path(program: WindingProgram) -> np.ndarray:
    """Rebuild the surface path from the axis motions (used to verify the post-processor)."""
    az = np.radians(program.mandrel_deg)
    return np.column_stack([program.radius_mm * np.cos(az),
                            program.radius_mm * np.sin(az),
                            program.carriage_mm])


def export_program_csv(program: WindingProgram, path: str | Path) -> Path:
    """Write a machine-neutral axis motion table (step, mandrel, carriage, eye yaw)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["step,mandrel_deg,carriage_mm,eye_yaw_deg"]
    for i in range(len(program.mandrel_deg)):
        rows.append(f"{i},{program.mandrel_deg[i]:.4f},{program.carriage_mm[i]:.4f},{program.eye_yaw_deg[i]:.4f}")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def program_summary(program: WindingProgram) -> dict:
    """Headline machine-motion figures for one course/path."""
    return {
        "axes": program.axes,
        "mandrel_revolutions": float((program.mandrel_deg[-1] - program.mandrel_deg[0]) / 360.0),
        "carriage_stroke_mm": float(np.max(program.carriage_mm) - np.min(program.carriage_mm)),
        "eye_yaw_min_deg": float(np.min(program.eye_yaw_deg)),
        "eye_yaw_max_deg": float(np.max(program.eye_yaw_deg)),
        "points": int(len(program.mandrel_deg)),
    }
