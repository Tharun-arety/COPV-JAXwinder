"""General axisymmetric meridian geometry.

The validated engine specializes its surface geometry — normals, meridian/hoop
directions, arc-length ``s``, and the two principal curvatures — to the parametric
ellipsoidal COPV. Every one of those quantities is, in fact, defined for *any*
surface of revolution by its 1-D meridian profile ``(rho(s), z(s))``.

This module computes them for an arbitrary profile, producing the exact same dict
structure the engine's COPV projection returns, so a general mandrel can be screened
without touching the solver. It is validated by reproducing the parametric COPV (see
app/validate_general.py): feed it a profile sampled from the analytic COPV and the
projection, curvatures, and resulting failure index must match.

Conventions (matched to copv_surface_from_sphi_np / copv_principal_curvatures_np):
* ``s`` increases from the top opening, down through the cylinder, to the bottom
  opening; on an arc-length grid the meridian tangent is unit.
* ``meridian_dirs`` = e_s = (drho/ds cosφ, drho/ds sinφ, dz/ds).
* ``hoop_dirs``     = e_phi = (-sinφ, cosφ, 0).
* outward in-plane normal (n_rho, n_z) = (-dz/ds, drho/ds); on the cylinder this is
  the radial (cosφ, sinφ, 0).
* curvatures are negative for an outward-bulging surface: cylinder gives
  k_meridian = 0, k_hoop = -1/R.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from copv_opt.config import GeometryConfig
from copv_opt.geometry import copv_surface_from_sphi_np, copv_meridional_metrics

_GRID = 1601  # dense uniform-arclength resample for stable derivatives + projection


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


class MeridianProfile:
    """An axisymmetric shell defined by a meridian polyline ``(rho, z)``.

    Points are ordered from the top opening to the bottom opening. The profile is
    resampled onto a uniform arc-length grid for stable finite-difference geometry.
    """

    def __init__(self, rho: np.ndarray, z: np.ndarray):
        rho = np.asarray(rho, dtype=np.float64).reshape(-1)
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if rho.shape != z.shape or rho.size < 4:
            raise ValueError("meridian needs matching rho/z arrays with >= 4 points")
        if np.any(rho < 0.0):
            raise ValueError("meridian rho must be non-negative (axisymmetric)")

        # raw cumulative arc length, then resample uniformly
        seg = np.sqrt(np.diff(rho) ** 2 + np.diff(z) ** 2)
        s_raw = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s_raw[-1])
        if total <= 0.0:
            raise ValueError("degenerate meridian (zero length)")

        sg = np.linspace(0.0, total, _GRID)
        rg = np.interp(sg, s_raw, rho)
        zg = np.interp(sg, s_raw, z)

        # arc-length derivatives (unit tangent enforced)
        drho = np.gradient(rg, sg)
        dz = np.gradient(zg, sg)
        mag = np.maximum(np.sqrt(drho**2 + dz**2), 1e-12)
        drho /= mag
        dz /= mag
        d2rho = np.gradient(drho, sg)
        d2z = np.gradient(dz, sg)

        self.total_len = total
        self._sg, self._rg, self._zg = sg, rg, zg
        self._drho, self._dz = drho, dz
        self._d2rho, self._d2z = d2rho, d2z
        self.top_opening = (float(rg[0]), float(zg[0]))
        self.bottom_opening = (float(rg[-1]), float(zg[-1]))

    # -- factory -------------------------------------------------------------
    @classmethod
    def from_parametric_copv(cls, geom: GeometryConfig, samples: int = 1200) -> "MeridianProfile":
        """Sample the analytic COPV meridian (phi=0). Used for validation and as the
        bridge from the parametric front door to the general path."""
        _, _, total = copv_meridional_metrics(
            geom.mid_radius, geom.cylinder_length, geom.opening_radius, geom.dome_height_ratio
        )
        s = np.linspace(0.0, total, samples)
        surf = copv_surface_from_sphi_np(
            geom.mid_radius, s, np.zeros_like(s), geom.cylinder_length, geom.opening_radius,
            dome_height_ratio=geom.dome_height_ratio,
        )
        pts = surf["points"]
        rho = np.hypot(pts[:, 0], pts[:, 1])
        return cls(rho, pts[:, 2])

    @classmethod
    def from_points(cls, points: np.ndarray) -> "MeridianProfile":
        """Build from an (N,2) array of (rho, z) meridian points."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("expected an (N, 2) array of (rho, z) points")
        return cls(pts[:, 0], pts[:, 1])

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (rho, z) at ``n`` equal-arc-length stations along the meridian."""
        s = np.linspace(0.0, self.total_len, int(n))
        return np.interp(s, self._sg, self._rg), np.interp(s, self._sg, self._zg)

    # -- geometry at arc length ----------------------------------------------
    def _interp(self, s: np.ndarray):
        s = np.clip(np.asarray(s, dtype=np.float64).reshape(-1), 0.0, self.total_len)
        rho = np.interp(s, self._sg, self._rg)
        z = np.interp(s, self._sg, self._zg)
        drho = np.interp(s, self._sg, self._drho)
        dz = np.interp(s, self._sg, self._dz)
        d2rho = np.interp(s, self._sg, self._d2rho)
        d2z = np.interp(s, self._sg, self._d2z)
        return s, rho, z, drho, dz, d2rho, d2z

    def surface_from_sphi(self, s: np.ndarray, phi: np.ndarray) -> dict[str, np.ndarray]:
        s, rho, z, drho, dz, _, _ = self._interp(s)
        phi = np.mod(np.asarray(phi, dtype=np.float64).reshape(-1), 2.0 * np.pi)
        cp, sp = np.cos(phi), np.sin(phi)
        points = np.stack([rho * cp, rho * sp, z], axis=-1)
        e_s = np.stack([drho * cp, drho * sp, dz], axis=-1)
        e_phi = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1)
        n_rho, n_z = -dz, drho                      # outward in-plane normal
        normals = np.stack([n_rho * cp, n_rho * sp, n_z], axis=-1)
        return {
            "points": points,
            "meridian_dirs": _normalize_rows(e_s),
            "hoop_dirs": _normalize_rows(e_phi),
            "normals": _normalize_rows(normals),
            "rho": rho,
            "s": s,
            "phi": phi,
        }

    def principal_curvatures(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s, rho, _, drho, dz, d2rho, d2z = self._interp(s)
        # meridian curvature = signed curvature of the (rho, z) profile in arc-length
        # parameterization, oriented to match the engine's outward-normal convention
        # (validated against copv_principal_curvatures_np: cylinder -> 0, caps negative).
        k_meridian = drho * d2z - dz * d2rho
        n_rho = -dz                                  # radial component of outward normal
        k_hoop = -n_rho / np.maximum(rho, 1e-9)
        return k_meridian, k_hoop

    # -- projection ----------------------------------------------------------
    def project(self, points: np.ndarray, chunk: int = 4096) -> dict[str, np.ndarray]:
        """Nearest-meridian projection of 3-D points -> the surface dict + s/phi."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        rho = np.hypot(pts[:, 0], pts[:, 1])
        phi = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2.0 * np.pi)
        s_best = np.empty(len(pts), dtype=np.float64)
        rg, zg, sg = self._rg, self._zg, self._sg
        for i in range(0, len(pts), chunk):
            r = rho[i : i + chunk][:, None]
            z = pts[i : i + chunk, 2][:, None]
            d2 = (r - rg[None, :]) ** 2 + (z - zg[None, :]) ** 2
            s_best[i : i + chunk] = sg[np.argmin(d2, axis=1)]
        out = self.surface_from_sphi(s_best, phi)
        return out
