from __future__ import annotations

import math
from typing import Any


RESOURCE_DIMENSIONS = (
    "active_evaluator_seconds",
    "exclusive_machine_seconds",
    "gpu_active_seconds",
)
_CAPS = {
    "active_evaluator_seconds": "max_active_evaluator_seconds",
    "exclusive_machine_seconds": "max_exclusive_machine_seconds",
    "gpu_active_seconds": "max_gpu_active_seconds",
}


class BudgetExhausted(ValueError):
    pass


def _nonnegative_number(value: Any, description: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{description} must be finite and nonnegative")
    return float(value)


def empty_usage() -> dict[str, float | int]:
    return {
        "calendar_seconds": 0.0,
        "active_evaluator_seconds": 0.0,
        "exclusive_machine_seconds": 0.0,
        "gpu_active_seconds": 0.0,
        "candidates_admitted": 0,
        "failed_attempts": 0,
    }


def validate_budget(contract: dict[str, Any]) -> None:
    if not isinstance(contract, dict) or set(contract) != {"total", "reserves"}:
        raise ValueError("budget must contain total and reserves")
    total = contract["total"]
    required = {
        "max_candidates_admitted",
        "max_calendar_seconds",
        "max_active_evaluator_seconds",
        "max_exclusive_machine_seconds",
        "max_gpu_active_seconds",
        "max_tokens",
        "max_monetary_usd",
    }
    if not isinstance(total, dict) or set(total) != required:
        raise ValueError("budget total fields are incomplete")
    if type(total["max_candidates_admitted"]) is not int or total[
        "max_candidates_admitted"
    ] < 1:
        raise ValueError("candidate budget must be positive")
    for name in required - {"max_candidates_admitted"}:
        _nonnegative_number(total[name], name)

    reserves = contract["reserves"]
    if not isinstance(reserves, list):
        raise ValueError("budget reserves must be a list")
    ids: set[str] = set()
    sums = {name: 0.0 for name in RESOURCE_DIMENSIONS}
    for reserve in reserves:
        if not isinstance(reserve, dict) or set(reserve) != {
            "tier_id",
            "invocations",
            "resources",
        }:
            raise ValueError("budget reserve fields are incomplete")
        tier_id = reserve["tier_id"]
        if not isinstance(tier_id, str) or not tier_id or tier_id in ids:
            raise ValueError("budget reserve tier_id is invalid or duplicated")
        ids.add(tier_id)
        if type(reserve["invocations"]) is not int or reserve["invocations"] < 1:
            raise ValueError("budget reserve invocation count must be positive")
        resources = reserve["resources"]
        if not isinstance(resources, dict) or not set(resources) <= set(
            RESOURCE_DIMENSIONS
        ):
            raise ValueError("budget reserve resources are invalid")
        for name in RESOURCE_DIMENSIONS:
            sums[name] += _nonnegative_number(resources.get(name, 0), name)
    for name, reserved in sums.items():
        if reserved > float(total[_CAPS[name]]):
            raise ValueError(f"{name} reserves exceed the total budget")


def _unspent_reserves(
    contract: dict[str, Any], active_tier: str
) -> dict[str, float]:
    result = {name: 0.0 for name in RESOURCE_DIMENSIONS}
    for reserve in contract["reserves"]:
        if reserve["tier_id"] == active_tier:
            continue
        for name in RESOURCE_DIMENSIONS:
            result[name] += float(reserve["resources"].get(name, 0.0))
    return result


def admit_tier(
    contract: dict[str, Any],
    usage: dict[str, float | int],
    tier_id: str,
    cost_limit: dict[str, Any],
) -> None:
    validate_budget(contract)
    if not isinstance(cost_limit, dict) or not set(cost_limit) <= set(
        RESOURCE_DIMENSIONS
    ):
        raise ValueError("tier cost limit is invalid")
    protected = _unspent_reserves(contract, tier_id)
    total = contract["total"]
    for name in RESOURCE_DIMENSIONS:
        requested = _nonnegative_number(cost_limit.get(name, 0.0), name)
        projected = float(usage.get(name, 0.0)) + requested + protected[name]
        if projected > float(total[_CAPS[name]]):
            raise BudgetExhausted(f"{name} budget is reserved for later tiers")


def charge_attempt(
    usage: dict[str, float | int], attempt: dict[str, Any]
) -> None:
    controller = attempt.get("controller")
    resources = attempt.get("resources")
    if not isinstance(controller, dict) or not isinstance(resources, dict):
        raise ValueError("attempt telemetry is incomplete")
    usage["active_evaluator_seconds"] = float(
        usage["active_evaluator_seconds"]
    ) + _nonnegative_number(
        controller.get("subprocess_wall_seconds"), "subprocess wall time"
    )
    usage["exclusive_machine_seconds"] = float(
        usage["exclusive_machine_seconds"]
    ) + _nonnegative_number(
        controller.get("exclusive_lease_seconds"), "exclusive lease time"
    )
    usage["gpu_active_seconds"] = float(usage["gpu_active_seconds"]) + (
        _nonnegative_number(
            resources.get("gpu_active_charge_seconds"), "GPU-active charge"
        )
    )
    if attempt.get("outcome") != "success":
        usage["failed_attempts"] = int(usage["failed_attempts"]) + 1

