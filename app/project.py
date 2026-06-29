"""Phase 1 — project data model.

A *project* is one designed tank: the requirement that defined it, the geometry it
sized to, and a snapshot of the last screening/optimization result. It is the unit
of a catalog — Blackwave's "off-the-shelf in days" model is a library of these.

Projects serialize to a single JSON file (no heavy mesh arrays — those are
re-derived from the requirement on demand). A :class:`ProjectStore` is a directory
of such files.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.sizing import SizingReport, TankRequirement, geometry_from_requirement

SCHEMA_VERSION = 1


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "untitled"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Project:
    name: str
    requirement: TankRequirement
    notes: str = ""
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    sizing: dict[str, Any] | None = None         # SizingReport snapshot
    result_summary: dict[str, Any] | None = None  # design metrics + gate (no big arrays)

    # -- result snapshot -----------------------------------------------------
    def record_result(self, result, sizing: SizingReport) -> None:
        """Store a compact, serializable snapshot of a DesignResult + sizing."""
        self.sizing = asdict(sizing)
        self.result_summary = {
            "mode": result.mode,
            "fi_max": float(result.fi_max),
            "burst_factor": float(result.burst_factor),
            "mass_metric": float(result.mass_metric),
            "mu_max_required": None if result.mu_max_required is None else float(result.mu_max_required),
            "mu_allowable": float(result.mu_allowable),
            "angle_deg": None if result.angle_deg is None else float(result.angle_deg),
            "disp_max": float(result.disp_max),
            "gate": result.gate,
        }
        self.updated = _now()

    # -- (de)serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "requirement": asdict(self.requirement),
            "notes": self.notes,
            "created": self.created,
            "updated": self.updated,
            "sizing": self.sizing,
            "result_summary": self.result_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        version = data.get("schema_version", 1)
        if version > SCHEMA_VERSION:
            raise ValueError(f"project schema v{version} is newer than supported v{SCHEMA_VERSION}")
        return cls(
            name=data["name"],
            requirement=TankRequirement(**data["requirement"]),
            notes=data.get("notes", ""),
            created=data.get("created", _now()),
            updated=data.get("updated", _now()),
            sizing=data.get("sizing"),
            result_summary=data.get("result_summary"),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- convenience ---------------------------------------------------------
    def geometry(self):
        """Re-derive (GeometryConfig, SizingReport) from the stored requirement."""
        return geometry_from_requirement(self.requirement)


class ProjectStore:
    """A directory of project JSON files — the tank catalog."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{_slug(name)}.project.json"

    def save(self, project: Project) -> Path:
        return project.save(self._path(project.name))

    def load(self, name: str) -> Project:
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"no project named {name!r} in {self.root}")
        return Project.load(path)

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def list(self) -> list[dict[str, Any]]:
        """Lightweight catalog listing (name, dates, headline metrics)."""
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.project.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            summary = data.get("result_summary") or {}
            rows.append(
                {
                    "name": data.get("name", path.stem),
                    "updated": data.get("updated", ""),
                    "volume_l": data.get("requirement", {}).get("internal_volume_litres"),
                    "pressure_bar": data.get("requirement", {}).get("design_pressure_bar"),
                    "fi_max": summary.get("fi_max"),
                    "burst_factor": summary.get("burst_factor"),
                    "decision": (summary.get("gate") or {}).get("decision"),
                    "path": str(path),
                }
            )
        return rows
