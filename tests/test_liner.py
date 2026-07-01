"""Fast unit tests for the Type 3/4 liner model — no mesh, no JAX."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copv_opt.clt import copv_cylinder_layup, cylinder_load_resultants
from copv_opt.liner import PRESETS, analyse_bare_liner, analyse_type3, liner_yield_pressure

AL = PRESETS["AL6061-T6"]
R, T = 100.0, 3.0


def test_bare_liner_thin_wall_closed_form():
    p = 5.0
    res = analyse_bare_liner(AL, T, cylinder_load_resultants(p, R))
    assert abs(res["sigma_hoop"] - p * R / T) / (p * R / T) < 1e-4
    assert abs(res["sigma_axial"] - p * R / (2 * T)) / (p * R / (2 * T)) < 1e-4
    assert abs(res["von_mises"] - np.sqrt(3) / 2 * p * R / T) / (np.sqrt(3) / 2 * p * R / T) < 1e-4


def test_yield_pressure_gives_unit_margin():
    py = liner_yield_pressure(AL, R, T)
    res = analyse_bare_liner(AL, T, cylinder_load_resultants(py, R))
    assert abs(res["yield_margin"] - 1.0) < 1e-4


def test_type3_overwrap_offloads_liner():
    p = 20.0
    bare = analyse_bare_liner(AL, T, cylinder_load_resultants(p, R))
    assert bare["yields"]                                  # bare aluminium tank fails
    overwrap = copv_cylinder_layup(25.0, n_helical_pairs=6, n_hoop=10, base_axial_plies=0)
    t3 = analyse_type3(AL, T, overwrap, cylinder_load_resultants(p, R))
    assert not t3["liner_yields"] and t3["liner_yield_margin"] > bare["yield_margin"]
    assert t3["composite_fi_max"] <= 1.0
