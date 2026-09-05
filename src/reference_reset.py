"""Constructive certificates for assignment-induced reference resets.

The learning policy selects one local tracking template per agent.  Each template
is assigned a canonical normalized reference offset.  When the selected template
changes from ``source`` to ``target``, continuity of the physical state gives the
tracking-coordinate reset

    j = r_source - r_target + d_reset,

where ``d_reset`` is an independently bounded residual reset.  This module
constructs the finite reference image, enumerates every template pair, and
certifies a policy-uniform jump envelope before any policy trace is generated.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ABS_TOL = 5.0e-12


class ReferenceResetError(ValueError):
    """Raised when a reference-reset library cannot certify its jump envelope."""


@dataclass(frozen=True)
class ReferenceResetCertificate:
    """Finite-library certificate for one application."""

    campaign_id: str
    application: str
    state_dimension: int
    mode_names: tuple[str, ...]
    offsets: dict[str, np.ndarray]
    residual_reset_bound: float
    reference_diameter: float
    configured_jump_bound: float
    certified_jump_bound: float
    certification_margin: float
    maximizing_pairs: tuple[tuple[str, str], ...]
    tracking_config_path: str
    campaign_config_path: str
    assignment_map_verified: bool

    def jump_vector(self, source_mode: str, target_mode: str) -> np.ndarray:
        """Return the assignment-induced coordinate reset without residual error."""
        try:
            return self.offsets[source_mode] - self.offsets[target_mode]
        except KeyError as exc:
            raise ReferenceResetError(f"unknown template in jump pair: {exc.args[0]}") from exc

    def jump_norm(self, source_mode: str, target_mode: str) -> float:
        return float(np.linalg.norm(self.jump_vector(source_mode, target_mode)))

    def verify_jump(
        self,
        source_mode: str,
        target_mode: str,
        residual_reset: Iterable[float] | np.ndarray | None = None,
    ) -> float:
        """Return the norm of a concrete reset and reject envelope violations."""
        jump = self.jump_vector(source_mode, target_mode)
        if residual_reset is not None:
            residual = np.asarray(tuple(residual_reset), dtype=float)
            if residual.shape != (self.state_dimension,):
                raise ReferenceResetError(
                    f"residual reset has shape {residual.shape}, expected {(self.state_dimension,)}"
                )
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm > self.residual_reset_bound + ABS_TOL:
                raise ReferenceResetError(
                    f"residual reset norm {residual_norm:.17g} exceeds certified bound "
                    f"{self.residual_reset_bound:.17g}"
                )
            jump = jump + residual
        norm = float(np.linalg.norm(jump))
        if norm > self.certified_jump_bound + ABS_TOL:
            raise ReferenceResetError(
                f"reset norm {norm:.17g} exceeds certified envelope "
                f"{self.certified_jump_bound:.17g}"
            )
        return norm


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceResetError(f"cannot read JSON configuration {path}: {exc}") from exc


def _normalized_direction(values: Any, dimension: int) -> np.ndarray:
    direction = np.asarray(values, dtype=float)
    if direction.shape != (dimension,):
        raise ReferenceResetError(
            f"reference direction has shape {direction.shape}, expected {(dimension,)}"
        )
    if not np.all(np.isfinite(direction)):
        raise ReferenceResetError("reference direction contains a nonfinite value")
    norm = float(np.linalg.norm(direction))
    if norm <= np.finfo(float).eps:
        raise ReferenceResetError("reference direction must be nonzero")
    return direction / norm


def _mode_names(tracking_config: dict[str, Any]) -> tuple[str, ...]:
    templates = tracking_config.get("controller_templates")
    if not isinstance(templates, dict) or not templates:
        raise ReferenceResetError("tracking configuration has no controller_templates")
    return tuple(str(name) for name in templates)


def _state_dimension(tracking_config: dict[str, Any]) -> int:
    gains = tracking_config.get("axis_input_gains")
    if not isinstance(gains, list) or not gains:
        raise ReferenceResetError("tracking configuration has no axis_input_gains")
    return 2 * len(gains)


def _resolve_config_path(sim_root: Path, value: Any, field: str, campaign_id: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReferenceResetError(f"{campaign_id}: missing {field}")
    path = (sim_root / value).resolve()
    root = sim_root.resolve()
    if root != path and root not in path.parents:
        raise ReferenceResetError(f"{campaign_id}: {field} escapes simulation root")
    if not path.is_file():
        raise ReferenceResetError(f"{campaign_id}: missing {field} file {path}")
    return path


def validate_assignment_template_map(
    campaign_id: str, campaign_config: dict[str, Any], mode_names: tuple[str, ...]
) -> tuple[tuple[float, float], ...]:
    """Validate the single runtime assignment-to-template partition.

    The first template is the unassigned/fallback template and must be the
    singleton interval ``[0,0]``.  The remaining templates form an ordered,
    gap-free, nonoverlapping partition of ``[0,1]``.  Shared endpoints are
    assigned to the lower-demand template by :func:`mode_from_assignment_map`.
    """
    mapping = campaign_config.get("controller_template_map")
    if not isinstance(mapping, dict) or tuple(mapping) != mode_names:
        raise ReferenceResetError(
            f"{campaign_id}: controller_template_map must preserve exactly {list(mode_names)}"
        )
    intervals: list[tuple[float, float]] = []
    for mode in mode_names:
        interval = mapping[mode]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ReferenceResetError(f"{campaign_id}: invalid assignment interval for {mode}")
        lower, upper = (float(interval[0]), float(interval[1]))
        if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 <= lower <= upper <= 1.0):
            raise ReferenceResetError(f"{campaign_id}: invalid assignment interval for {mode}")
        intervals.append((lower, upper))

    if len(intervals) < 2:
        raise ReferenceResetError(f"{campaign_id}: at least one assigned template is required")
    if abs(intervals[0][0]) > ABS_TOL or abs(intervals[0][1]) > ABS_TOL:
        raise ReferenceResetError(
            f"{campaign_id}: the first template must be the singleton unassigned interval [0,0]"
        )
    assigned = intervals[1:]
    if abs(assigned[0][0]) > ABS_TOL or abs(assigned[-1][1] - 1.0) > ABS_TOL:
        raise ReferenceResetError(f"{campaign_id}: assigned intervals must cover [0,1]")
    previous_upper = assigned[0][0]
    for index, (lower, upper) in enumerate(assigned, start=1):
        if upper <= lower + ABS_TOL:
            raise ReferenceResetError(
                f"{campaign_id}: assigned interval for {mode_names[index]} must have positive width"
            )
        if abs(lower - previous_upper) > ABS_TOL:
            raise ReferenceResetError(
                f"{campaign_id}: assigned intervals must be ordered, gap-free, and nonoverlapping"
            )
        previous_upper = upper
    return tuple(intervals)


def mode_from_assignment_map(
    campaign_id: str,
    campaign_config: dict[str, Any],
    mode_names: tuple[str, ...],
    demand: float,
    assigned: bool,
) -> str:
    """Return the configured controller template for one normalized demand."""
    intervals = validate_assignment_template_map(campaign_id, campaign_config, mode_names)
    if not assigned:
        return mode_names[0]
    value = float(demand)
    if not math.isfinite(value) or value < -ABS_TOL or value > 1.0 + ABS_TOL:
        raise ReferenceResetError(f"{campaign_id}: normalized demand is outside [0,1]")
    value = min(1.0, max(0.0, value))
    for mode, (_, upper) in zip(mode_names[1:], intervals[1:]):
        if value <= upper:
            return mode
    raise ReferenceResetError(f"{campaign_id}: assignment map does not cover demand {value}")


def construct_reference_reset_certificate(
    sim_root: Path,
    campaign_id: str,
    entry: dict[str, Any],
) -> ReferenceResetCertificate:
    """Construct and verify one finite reference-reset certificate."""
    tracking_path = _resolve_config_path(
        sim_root, entry.get("tracking_model_config"), "tracking_model_config", campaign_id
    )
    campaign_path = _resolve_config_path(
        sim_root, entry.get("campaign_config"), "campaign_config", campaign_id
    )
    tracking = _read_json(tracking_path)
    campaign = _read_json(campaign_path)

    mode_names = _mode_names(tracking)
    validate_assignment_template_map(campaign_id, campaign, mode_names)
    dimension = _state_dimension(tracking)
    configured_jump_bound = float(tracking.get("jump_bound", math.nan))
    residual_bound = float(entry.get("residual_reset_bound", 0.0))
    if not math.isfinite(configured_jump_bound) or configured_jump_bound < 0.0:
        raise ReferenceResetError(f"{campaign_id}: invalid configured jump bound")
    if not math.isfinite(residual_bound) or residual_bound < 0.0:
        raise ReferenceResetError(f"{campaign_id}: invalid residual reset bound")
    if residual_bound > configured_jump_bound + ABS_TOL:
        raise ReferenceResetError(f"{campaign_id}: residual reset bound exceeds jump envelope")

    levels_raw = entry.get("mode_levels")
    if not isinstance(levels_raw, dict) or set(levels_raw) != set(mode_names):
        raise ReferenceResetError(
            f"{campaign_id}: mode_levels must contain exactly {list(mode_names)}"
        )
    levels = {name: float(levels_raw[name]) for name in mode_names}
    if any(not math.isfinite(value) for value in levels.values()):
        raise ReferenceResetError(f"{campaign_id}: nonfinite reference level")
    minimum_level = min(levels.values())
    maximum_level = max(levels.values())
    level_span = maximum_level - minimum_level
    if level_span <= 0.0:
        raise ReferenceResetError(f"{campaign_id}: reference levels must not be identical")

    direction = _normalized_direction(entry.get("reference_direction"), dimension)
    allocated_diameter = configured_jump_bound - residual_bound
    offsets = {
        name: ((levels[name] - minimum_level) / level_span) * allocated_diameter * direction
        for name in mode_names
    }

    pair_norms: dict[tuple[str, str], float] = {}
    for source, target in itertools.product(mode_names, repeat=2):
        pair_norms[(source, target)] = float(np.linalg.norm(offsets[source] - offsets[target]))
    diameter = max(pair_norms.values())
    maximizing_pairs = tuple(
        sorted(pair for pair, value in pair_norms.items() if abs(value - diameter) <= ABS_TOL)
    )
    certified = diameter + residual_bound
    margin = configured_jump_bound - certified
    if margin < -ABS_TOL:
        raise ReferenceResetError(
            f"{campaign_id}: constructed envelope {certified:.17g} exceeds configured "
            f"jump bound {configured_jump_bound:.17g}"
        )

    certificate = ReferenceResetCertificate(
        campaign_id=campaign_id,
        application=str(tracking.get("application", campaign_id)),
        state_dimension=dimension,
        mode_names=mode_names,
        offsets=offsets,
        residual_reset_bound=residual_bound,
        reference_diameter=diameter,
        configured_jump_bound=configured_jump_bound,
        certified_jump_bound=certified,
        certification_margin=margin,
        maximizing_pairs=maximizing_pairs,
        tracking_config_path=tracking_path.relative_to(sim_root.resolve()).as_posix(),
        campaign_config_path=campaign_path.relative_to(sim_root.resolve()).as_posix(),
        assignment_map_verified=True,
    )

    # Exhaustive finite-family verification.  This is deliberately performed
    # after construction so a future change in the library cannot bypass the
    # pairwise contract.
    for source, target in itertools.product(mode_names, repeat=2):
        certificate.verify_jump(source, target)
    return certificate


def load_reference_reset_certificates(
    sim_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, ReferenceResetCertificate]:
    root = Path(sim_root).resolve()
    path = Path(config_path).resolve() if config_path is not None else root / "configs" / "reference_reset_library.json"
    payload = _read_json(path)
    if payload.get("schema_version") != "reference-reset-library-v1":
        raise ReferenceResetError("unsupported reference-reset schema")
    applications = payload.get("applications")
    if not isinstance(applications, dict) or not applications:
        raise ReferenceResetError("reference-reset configuration has no applications")
    return {
        str(campaign_id): construct_reference_reset_certificate(root, str(campaign_id), entry)
        for campaign_id, entry in sorted(applications.items())
    }
