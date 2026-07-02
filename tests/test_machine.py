"""Tests for coverage/pattern generation and the machine post-processor."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.machine import machine_program_from_path, program_summary, reconstruct_path
from copv_opt.winding import helical_coverage, pattern_for_coverage


# ---- coverage / pattern ----
def test_pattern_coverage_fraction():
    n = helical_coverage(200.0, 45.0, 6.0).n_bands
    one = pattern_for_coverage(200.0, 45.0, 6.0, target_coverage=1.0)
    two = pattern_for_coverage(200.0, 45.0, 6.0, target_coverage=2.0)
    assert one.circuits == n and abs(one.coverage_achieved - 1.0) < 1e-9
    assert two.circuits == 2 * n and abs(two.coverage_achieved - 2.0) < 1e-9


def test_pattern_number_is_coprime():
    d = pattern_for_coverage(200.0, 40.0, 6.0, target_coverage=1.0)
    assert math.gcd(d.pattern_number, d.n_bands) == 1 and d.closes


# ---- machine post-processor (verify by reconstructing the path) ----
def _helix(r=100.0, turns=3.0, length=200.0, n=400):
    phi = np.linspace(0.0, turns * 2 * math.pi, n)
    z = np.linspace(0.0, length, n)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def test_machine_program_reconstructs_path():
    path = _helix()
    prog = machine_program_from_path(path)
    recon = reconstruct_path(prog)
    assert np.max(np.linalg.norm(recon - path, axis=1)) < 1e-6


def test_machine_program_axes_sane():
    prog = machine_program_from_path(_helix(turns=3.0))
    s = program_summary(prog)
    assert abs(s["mandrel_revolutions"] - 3.0) < 1e-6      # 3 turns
    assert abs(s["carriage_stroke_mm"] - 200.0) < 1e-3     # full length
    # constant-pitch helix -> near-constant delivery yaw
    assert s["eye_yaw_max_deg"] - s["eye_yaw_min_deg"] < 5.0
