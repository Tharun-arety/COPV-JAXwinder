"""Phase 4 — external solver orchestration.

The screening solve is fast but approximate. Verification belongs in a trusted FEA
code. The product's job is to *orchestrate* that handoff, not to replace it.

Real:
* ``export_abaqus`` writes an Abaqus ``.inp`` deck from the optimized result.
* ``find_calculix`` / ``run_calculix`` detect and drive CalculiX (``ccx``), which
  reads a subset of the Abaqus deck format, when the binary is installed.

Honest gap:
* The CalculiX binary itself is not bundled. If ``ccx`` is absent, ``run_calculix``
  reports that clearly instead of failing obscurely. Production verification at
  Blackwave would target Ansys ACP + Mechanical.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from copv_opt.abaqus_exporter import export_result_to_abaqus
from copv_opt.config import GeometryConfig, MaterialConfig


def export_abaqus(
    state: dict[str, Any],
    winding_result: dict[str, Any],
    geom: GeometryConfig,
    path: str | Path,
    material: MaterialConfig | None = None,
    heading: str = "COPV configurator export",
) -> Path:
    """Write an Abaqus .inp deck from an optimized winding result."""
    if state is None or winding_result is None:
        raise ValueError("Abaqus export needs a full_optimize result (state + winding_result)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return export_result_to_abaqus(
        state,
        winding_result,
        geom,
        path,
        material=material or MaterialConfig(),
        heading=heading,
    )


def find_calculix() -> str | None:
    """Return the path to a CalculiX executable (ccx) if one is on PATH."""
    for name in ("ccx", "ccx_2.21", "ccx_2.20", "ccx_2.19", "CalculiX"):
        found = shutil.which(name)
        if found:
            return found
    return None


@dataclass
class SolverRun:
    ran: bool
    solver_path: str | None
    returncode: int | None
    message: str
    frd_path: str | None = None


def run_calculix(inp_path: str | Path, timeout_s: int = 1800) -> SolverRun:
    """Run CalculiX on an Abaqus-format deck if available.

    CalculiX expects the job name without the .inp suffix and writes <job>.frd."""
    inp_path = Path(inp_path)
    ccx = find_calculix()
    if ccx is None:
        return SolverRun(
            ran=False,
            solver_path=None,
            returncode=None,
            message=(
                "CalculiX (ccx) not found on PATH. Install CalculiX to run verification, "
                "or hand the .inp deck to Ansys/Abaqus. The deck was still written."
            ),
        )
    job = inp_path.with_suffix("")   # ccx adds .inp / .frd itself
    try:
        proc = subprocess.run(
            [ccx, job.name],
            cwd=str(inp_path.parent),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return SolverRun(ran=True, solver_path=ccx, returncode=None, message=f"CalculiX timed out after {timeout_s}s")
    frd = job.with_suffix(".frd")
    return SolverRun(
        ran=True,
        solver_path=ccx,
        returncode=proc.returncode,
        message=("CalculiX completed" if proc.returncode == 0 else f"CalculiX exited {proc.returncode}: {proc.stderr[-400:]}"),
        frd_path=str(frd) if frd.exists() else None,
    )
