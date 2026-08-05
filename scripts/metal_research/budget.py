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


def empty_usage() -> dict[str, Any]:
    return {
        "calendar_seconds": 0.0,
        "active_evaluator_seconds": 0.0,
        "exclusive_machine_seconds": 0.0,
        "gpu_active_seconds": 0.0,
        "gpu_active_validated_seconds": 0.0,
        "gpu_active_estimated_seconds": 0.0,
        "candidates_admitted": 0,
        "failed_attempts": 0,
        "reserve_invocations": {},
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
            "id",
            "invocations",
            "resources",
        }:
            raise ValueError("budget reserve fields are incomplete")
        reserve_id = reserve["id"]
        if not isinstance(reserve_id, str) or not reserve_id or reserve_id in ids:
            raise ValueError("budget reserve id is invalid or duplicated")
        ids.add(reserve_id)
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
    contract: dict[str, Any], usage: dict[str, Any], active_reserve: str | None
) -> dict[str, float]:
    result = {name: 0.0 for name in RESOURCE_DIMENSIONS}
    consumed = usage.get("reserve_invocations", {})
    if not isinstance(consumed, dict):
        raise ValueError("reserve invocation accounting is invalid")
    for reserve in contract["reserves"]:
        used = consumed.get(reserve["id"], 0)
        if type(used) is not int or used < 0:
            raise ValueError("reserve invocation accounting is invalid")
        remaining = max(0, reserve["invocations"] - used)
        if reserve["id"] == active_reserve:
            remaining = max(0, remaining - 1)
        fraction = remaining / reserve["invocations"]
        for name in RESOURCE_DIMENSIONS:
            result[name] += fraction * float(reserve["resources"].get(name, 0.0))
    return result


def admit_tier(
    contract: dict[str, Any],
    usage: dict[str, Any],
    cost_limit: dict[str, Any],
    reserve_id: str | None = None,
) -> None:
    validate_budget(contract)
    if not isinstance(cost_limit, dict) or not set(cost_limit) <= set(
        RESOURCE_DIMENSIONS
    ):
        raise ValueError("tier cost limit is invalid")
    reserve_ids = {reserve["id"] for reserve in contract["reserves"]}
    if reserve_id is not None and reserve_id not in reserve_ids:
        raise ValueError(f"unknown budget reserve: {reserve_id}")
    protected = _unspent_reserves(contract, usage, reserve_id)
    total = contract["total"]
    for name in RESOURCE_DIMENSIONS:
        requested = _nonnegative_number(cost_limit.get(name, 0.0), name)
        projected = float(usage.get(name, 0.0)) + requested + protected[name]
        if projected > float(total[_CAPS[name]]):
            raise BudgetExhausted(f"{name} budget is reserved for later tiers")


def charge_attempt(usage: dict[str, Any], attempt: dict[str, Any]) -> None:
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
    gpu_charge = _nonnegative_number(
        resources.get("gpu_active_charge_seconds"), "GPU-active charge"
    )
    usage["gpu_active_seconds"] = float(usage["gpu_active_seconds"]) + gpu_charge
    actual = resources.get("gpu_active_seconds")
    if actual is None:
        usage["gpu_active_estimated_seconds"] = float(
            usage["gpu_active_estimated_seconds"]
        ) + gpu_charge
    else:
        validated = _nonnegative_number(actual, "validated GPU-active time")
        if validated > gpu_charge:
            raise ValueError("validated GPU-active time exceeds its budget charge")
        usage["gpu_active_validated_seconds"] = float(
            usage["gpu_active_validated_seconds"]
        ) + validated
    if attempt.get("outcome") != "success":
        usage["failed_attempts"] = int(usage["failed_attempts"]) + 1
    reserve_id = attempt.get("budget_reserve")
    if reserve_id is not None:
        if not isinstance(reserve_id, str) or not reserve_id:
            raise ValueError("attempt budget reserve is invalid")
        invocations = usage["reserve_invocations"]
        invocations[reserve_id] = int(invocations.get(reserve_id, 0)) + 1
