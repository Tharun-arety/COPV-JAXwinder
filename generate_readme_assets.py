from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS = PROJECT_ROOT / "outputs"
SUMMARY_PATH = OUTPUTS / "summary.json"
README_FIGURE_PATH = OUTPUTS / "readme_optimization_summary.png"


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def normalized(series: list[dict], baseline: float) -> list[float]:
    return [entry["strain_energy"] / baseline for entry in series]


def main() -> None:
    summary = load_summary()

    baseline_energy = summary["baseline"]["strain_energy"]
    baseline_mass = summary["baseline"]["mass_metric"]
    patch_history = summary["patch_jax"]["history"]
    ifp_history = summary["ifp_jax"]["history"]
    winding_history = summary["winding_jax"]["history"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    patch_x = list(range(1, len(patch_history) + 1))
    ifp_x = list(range(1, len(ifp_history) + 1))
    winding_x = list(range(1, len(winding_history) + 1))

    ax.axhline(1.0, color="0.45", linestyle="--", linewidth=1.2, label="Baseline")
    ax.plot(
        patch_x,
        normalized(patch_history, baseline_energy),
        marker="o",
        color="steelblue",
        linewidth=2,
        label="Patch continuation",
    )
    ax.plot(
        ifp_x,
        normalized(ifp_history, baseline_energy),
        marker="o",
        color="darkorange",
        linewidth=2,
        label="IFP continuation",
    )
    ax.plot(
        winding_x,
        normalized(winding_history, baseline_energy),
        marker="o",
        color="forestgreen",
        linewidth=2,
        markersize=4,
        label="Winding angle sweep",
    )
    ax.set_xlabel("Optimization step / angle sample")
    ax.set_ylabel("Normalized compliance (baseline = 1.0)")
    ax.set_title("Optimization traces")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    points = [
        (
            "Base",
            summary["baseline"]["mass_metric"],
            summary["baseline"]["strain_energy"] / baseline_energy,
            "0.50",
        ),
        (
            "Patch",
            summary["patch_jax"]["mass_metric"],
            summary["patch_jax"]["strain_energy"] / baseline_energy,
            "steelblue",
        ),
        (
            "IFP",
            summary["ifp_jax"]["mass_metric"],
            summary["ifp_jax"]["strain_energy"] / baseline_energy,
            "darkorange",
        ),
        (
            "Winding",
            summary["winding_jax"]["mass_metric"],
            summary["winding_jax"]["strain_energy"] / baseline_energy,
            "forestgreen",
        ),
    ]
    for label, mass_metric, norm_energy, color in points:
        ax.scatter(mass_metric, norm_energy, s=140, color=color)
        mass_delta = (mass_metric / baseline_mass - 1.0) * 100.0
        if label == "Base":
            text = "Base"
        else:
            text = f"{label}\n{mass_delta:+.1f}% mass"
        ax.annotate(text, (mass_metric, norm_energy), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Relative mass metric")
    ax.set_ylabel("Normalized compliance")
    ax.set_title("Final stiffness / mass trade-off")
    ax.grid(alpha=0.3)

    fig.suptitle("COPV optimizer summary (CPU run, 899 nodes / 2696 tetrahedra)", fontsize=14)
    fig.tight_layout()
    fig.savefig(README_FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {README_FIGURE_PATH}")


if __name__ == "__main__":
    main()
