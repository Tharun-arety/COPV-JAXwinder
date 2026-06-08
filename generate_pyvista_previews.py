from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.visualize import load_layout_json, save_explicit_manufacturing_layout_screenshot


def main() -> None:
    outputs = ROOT / "outputs"
    base_vtu = outputs / "copv_base.vtu"
    patch_layout_path = outputs / "patch_layout.json"
    ifp_layout_path = outputs / "ifp_layout.json"
    winding_layout_path = outputs / "winding_layout.json"
    required = [base_vtu, patch_layout_path, ifp_layout_path, winding_layout_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files required for explicit previews: {missing}")

    patch_layout = load_layout_json(patch_layout_path)
    ifp_layout = load_layout_json(ifp_layout_path)
    winding_layout = load_layout_json(winding_layout_path)

    patch_polygons = [np.asarray(entry["corners"], dtype=np.float64) for entry in patch_layout["patches"]]
    ifp_curves = [np.asarray(points, dtype=np.float64) for points in ifp_layout["curves"]]
    winding_curves = [np.asarray(entry["points"], dtype=np.float64) for entry in winding_layout["paths"]]
    winding_curve_colors = [
        "forestgreen" if entry["handedness"] == "clockwise" else "darkmagenta"
        for entry in winding_layout["paths"]
    ]

    generated = [
        save_explicit_manufacturing_layout_screenshot(
            base_vtu,
            outputs / "pyvista_patch_explicit_layout.png",
            patch_polygons=patch_polygons,
            patch_color="steelblue",
            patch_edge_color="midnightblue",
            title="PyVista Explicit Patch Layout",
        ),
        save_explicit_manufacturing_layout_screenshot(
            base_vtu,
            outputs / "pyvista_ifp_explicit_layout.png",
            curve_points_list=ifp_curves,
            curve_color="crimson",
            tow_radius=max(1.15, 0.12 * float(ifp_layout["tow_width"])),
            title="PyVista Explicit IFP Layout",
        ),
        save_explicit_manufacturing_layout_screenshot(
            base_vtu,
            outputs / "pyvista_winding_explicit_layout.png",
            curve_points_list=winding_curves,
            curve_colors=winding_curve_colors,
            tow_radius=1.15,
            title="PyVista Explicit Winding Layout",
        ),
    ]

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
