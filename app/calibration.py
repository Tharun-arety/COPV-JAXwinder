"""Phase 5 — material calibration.

This is what turns the honest ``do_not_release`` gate into a workflow rather than a
dead end. The screen ships with literature Hashin allowables. Supply coupon-derived
strengths and the tool re-screens against them and reports how the failure index and
gate move.

The *mechanism* is real and verified with synthetic numbers. The *data* — real
Blackwave coupon allowables — is the external input that is genuinely missing; that
is the gap calibration is built to close.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from copv_opt.config import FailureConfig, GeometryConfig, MaterialAllowables, MaterialConfig

_FIELDS = ("xt", "xc", "yt", "yc", "s")


def load_allowables(path: str | Path) -> MaterialAllowables:
    """Load coupon allowables from JSON ({"xt":..,...}) or a 2-column key,value CSV.

    Only the five Hashin strengths are read; missing keys keep their default."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")   # tolerate a UTF-8 BOM (Excel/PowerShell exports)
    values: dict[str, float] = {}
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
        for k in _FIELDS:
            if k in raw:
                values[k] = float(raw[k])
            elif k.upper() in raw:
                values[k] = float(raw[k.upper()])
    else:
        for row in csv.reader(text.splitlines()):
            if len(row) < 2:
                continue
            key = row[0].strip().lower()
            if key in _FIELDS:
                try:
                    values[key] = float(row[1])
                except ValueError:
                    continue
    if not values:
        raise ValueError(f"no Hashin allowables ({', '.join(_FIELDS)}) found in {path}")
    base = MaterialAllowables()
    return MaterialAllowables(**{k: values.get(k, getattr(base, k)) for k in _FIELDS})


def failure_config_from_allowables(allowables: MaterialAllowables, margin_of_safety: float = 1.0) -> FailureConfig:
    return FailureConfig(allowables=allowables, margin_of_safety=margin_of_safety)


def calibration_delta(default_result, calibrated_result, allowables: MaterialAllowables) -> dict[str, Any]:
    """Compare a default-allowables screen against a calibrated one."""
    return {
        "default": {
            "fi_max": float(default_result.fi_max),
            "burst_factor": float(default_result.burst_factor),
            "decision": default_result.gate["decision"],
            "hashin_ok": default_result.gate["hashin_ok"],
        },
        "calibrated": {
            "fi_max": float(calibrated_result.fi_max),
            "burst_factor": float(calibrated_result.burst_factor),
            "decision": calibrated_result.gate["decision"],
            "hashin_ok": calibrated_result.gate["hashin_ok"],
        },
        "fi_max_change": float(calibrated_result.fi_max - default_result.fi_max),
        "burst_change": float(calibrated_result.burst_factor - default_result.burst_factor),
        "allowables_used": asdict(allowables),
    }


def calibrate_screen(
    geom: GeometryConfig,
    material: MaterialConfig,
    angle_deg: float,
    band_thickness: float,
    allowables: MaterialAllowables,
):
    """Run a fast screen against calibrated allowables. Imported here to keep the
    engine import lazy for callers that only parse allowables."""
    from app.engine import fast_screen

    failure_cfg = failure_config_from_allowables(allowables)
    return fast_screen(geom, material, angle_deg, band_thickness, failure_cfg=failure_cfg)
