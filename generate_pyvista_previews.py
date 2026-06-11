from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.visualize import load_layout_json, save_explicit_manufacturing_layout_screenshot


def main() -> None:
    outputs = ROOT / "outputs"
    base_vtu = outputs / "copv_winding_first.vtu"
    winding_layout_path = outputs / "winding_first_layout.json"
    required = [base_vtu, winding_layout_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files required for the winding-first PyVista preview: {missing}")

    winding_layout = load_layout_json(winding_layout_path)

    winding_curves = [entry["points"] for entry in winding_layout["paths"]]
    winding_curve_colors = [
        "forestgreen" if entry["handedness"] == "clockwise" else "darkmagenta"
        for entry in winding_layout["paths"]
    ]

    screenshot = save_explicit_manufacturing_layout_screenshot(
        base_vtu,
        outputs / "pyvista_winding_first_layout.png",
        curve_points_list=winding_curves,
        curve_colors=winding_curve_colors,
        tow_radius=1.15,
        title="PyVista explicit winding-first layout",
    )
    print(screenshot)


if __name__ == "__main__":
    main()
