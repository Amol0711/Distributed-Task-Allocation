"""Typed scope contract for operational and fixed-trace return routes.

The contract prevents a certificate replay from being silently promoted to a
counterfactual target-policy execution.  A trajectory-coupled target route is
admissible only when both its measurable nominal return and its tracking-to-
return Lipschitz interface are declared before execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

_ALLOWED_ROUTE_STATUS = {
    "operational",
    "fixed_trace_recertification",
    "inactive_trajectory_interface",
}
_REQUIRED_ROUTE_STATUS = {
    "direct_operational": "operational",
    "support_fixed_trace": "fixed_trace_recertification",
    "prior_fixed_trace": "fixed_trace_recertification",
    "trajectory_interface_only": "inactive_trajectory_interface",
}


class ReturnModelScopeError(ValueError):
    """Raised when the return-model scope contract is inconsistent."""


@dataclass(frozen=True)
class RouteDeclaration:
    name: str
    status: str
    description: str


@dataclass(frozen=True)
class TrajectoryReturnInterface:
    enabled: bool
    nominal_return_specification: str | None
    tracking_to_return_lipschitz_specification: str | None


@dataclass(frozen=True)
class ReturnModelScope:
    schema_version: int
    target_execution_route: str
    routes: Mapping[str, RouteDeclaration]
    trajectory_return_interface: TrajectoryReturnInterface

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReturnModelScope":
        schema_version = int(raw.get("schema_version", -1))
        if schema_version != 1:
            raise ReturnModelScopeError("schema_version must equal 1")

        route_raw = raw.get("routes")
        if not isinstance(route_raw, Mapping):
            raise ReturnModelScopeError("routes must be a mapping")

        routes: dict[str, RouteDeclaration] = {}
        for name, item in route_raw.items():
            if not isinstance(item, Mapping):
                raise ReturnModelScopeError(f"route {name!r} must be a mapping")
            status = str(item.get("status", ""))
            description = str(item.get("description", "")).strip()
            if status not in _ALLOWED_ROUTE_STATUS:
                raise ReturnModelScopeError(f"route {name!r} has unsupported status {status!r}")
            if not description:
                raise ReturnModelScopeError(f"route {name!r} requires a nonempty description")
            routes[str(name)] = RouteDeclaration(str(name), status, description)

        for name, expected_status in _REQUIRED_ROUTE_STATUS.items():
            if name not in routes:
                raise ReturnModelScopeError(f"required route {name!r} is missing")
            if routes[name].status != expected_status:
                raise ReturnModelScopeError(
                    f"route {name!r} must have status {expected_status!r}"
                )

        target_execution_route = str(raw.get("target_execution_route", ""))
        if target_execution_route not in routes:
            raise ReturnModelScopeError("target_execution_route must name a declared route")
        if routes[target_execution_route].status != "operational":
            raise ReturnModelScopeError(
                "target_execution_route cannot be a fixed-trace or inactive trajectory route"
            )

        interface_raw = raw.get("trajectory_return_interface")
        if not isinstance(interface_raw, Mapping):
            raise ReturnModelScopeError("trajectory_return_interface must be a mapping")
        interface = TrajectoryReturnInterface(
            enabled=bool(interface_raw.get("enabled", False)),
            nominal_return_specification=_optional_text(
                interface_raw.get("nominal_return_specification")
            ),
            tracking_to_return_lipschitz_specification=_optional_text(
                interface_raw.get("tracking_to_return_lipschitz_specification")
            ),
        )
        declared = (
            interface.nominal_return_specification is not None,
            interface.tracking_to_return_lipschitz_specification is not None,
        )
        if interface.enabled and not all(declared):
            raise ReturnModelScopeError(
                "an enabled trajectory return requires both nominal-return and "
                "tracking-to-return Lipschitz specifications"
            )
        if not interface.enabled and any(declared):
            raise ReturnModelScopeError(
                "a disabled trajectory return must not contain a partial physical-return declaration"
            )
        return cls(
            schema_version=schema_version,
            target_execution_route=target_execution_route,
            routes=routes,
            trajectory_return_interface=interface,
        )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_return_model_scope(path: str | Path) -> ReturnModelScope:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReturnModelScopeError(f"cannot load return-model scope from {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ReturnModelScopeError("top-level scope document must be a mapping")
    return ReturnModelScope.from_dict(raw)
