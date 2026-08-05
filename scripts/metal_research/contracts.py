from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .budget import RESOURCE_DIMENSIONS, validate_budget
from .paired import validate_replication
from .versions import (
    GOAL_SCHEMA_VERSION,
    TEMPLATE_SCHEMA_VERSION,
    TIER_RESULT_SCHEMA,
)


_ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
_ROLES = {"correctness", "proxy", "representative", "holdout", "transfer"}
_RESULT_ADAPTERS = {"outer_remainder_v3", "metal_piop_v7"}


def _relative_file(root: Path, value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} path must be a nonempty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} path must stay within the repository")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"{description} path does not exist: {value}")
    return path


def _speedup(value: Any, description: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 1.0
    ):
        raise ValueError(f"{description} must be a finite speedup")
    return float(value)


def validate_goal_contract(contract: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "goal",
        "goal_prompt",
        "primary_metric",
        "timing_boundary",
        "continuation",
        "kernel_promotion",
        "kernel_overrides",
        "portfolio_acceptance",
        "transfer_validation",
        "orchestration",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        raise ValueError("goal contract fields are incomplete")
    if contract["schema_version"] != GOAL_SCHEMA_VERSION:
        raise ValueError("unsupported goal contract schema")
    metric = contract["primary_metric"]
    floor = _speedup(metric.get("minimum_accepted_speedup"), "portfolio floor")
    if floor < 5.0 or metric.get("direction") != "max":
        raise ValueError("the portfolio must maximize a speedup with a 5x floor")
    if metric.get("timed_span") != "jolt_prover::piop":
        raise ValueError("the portfolio timing boundary must be the PIOP span")

    promotion = contract["kernel_promotion"]
    if not isinstance(promotion, list) or not promotion:
        raise ValueError("kernel promotion policy is empty")
    for item in promotion:
        minimum = _speedup(
            item.get("minimum_hybrid_speedup"), "kernel promotion floor"
        )
        target = _speedup(item.get("target_hybrid_speedup"), "kernel target")
        if minimum < 5.0:
            raise ValueError("every kernel promotion floor must be at least 5x")
        if target < minimum:
            raise ValueError("kernel targets cannot be below their promotion floor")
    overrides = contract["kernel_overrides"]
    instruction_ra = overrides.get("instruction_ra_virtualization", {})
    if _speedup(
        instruction_ra.get("minimum_hybrid_speedup"),
        "Instruction RA promotion floor",
    ) < 7.0:
        raise ValueError("Instruction RA must retain its 7x promotion floor")

    acceptance = contract["portfolio_acceptance"]
    if (
        acceptance.get("pairs", 0) < 5
        or acceptance.get("require_alternating_orders") is not True
        or _speedup(
            acceptance.get("minimum_overall_speedup"), "acceptance floor"
        )
        < floor
        or _speedup(
            acceptance.get("minimum_order_stratum_speedup"),
            "order-stratum floor",
        )
        < floor
    ):
        raise ValueError("portfolio acceptance does not enforce five paired 5x results")
    transfer = contract["transfer_validation"]
    if (
        27 not in transfer.get("required_log_trace_sizes", [])
        or _speedup(transfer.get("minimum_speedup"), "transfer floor") < floor
    ):
        raise ValueError("transfer validation must retain the log-27 5x floor")
    continuation = contract["continuation"]
    if continuation.get("stop_at_minimum") is not False:
        raise ValueError("the goal must remain uncapped after reaching 5x")
    trigger = _speedup(
        continuation.get("analytical_headroom_trigger"),
        "analytical headroom trigger",
    )
    if trigger > floor:
        raise ValueError("the analytical headroom trigger cannot exceed the goal floor")
    minimum_gain = continuation.get("minimum_projected_relative_gain_after_target")
    if (
        isinstance(minimum_gain, bool)
        or not isinstance(minimum_gain, (int, float))
        or not math.isfinite(minimum_gain)
        or not 0 < minimum_gain < 1
    ):
        raise ValueError("the post-target continuation gain is invalid")
    orchestration = contract["orchestration"]
    if (
        orchestration.get("worktree_writer") != "root"
        or orchestration.get("evaluator_concurrency") != 1
        or orchestration.get("proposal_concurrency", 0) < 2
        or orchestration.get("promotion_queue", {}).get("owner") != "root"
    ):
        raise ValueError("the root agent must serialize writes and evaluator execution")


def _validate_tier(tier: dict[str, Any], goal_floor: float) -> None:
    tier_id = tier.get("id")
    role = tier.get("role")
    if not isinstance(tier_id, str) or _ID.fullmatch(tier_id) is None:
        raise ValueError("tier id is invalid")
    if role not in _ROLES:
        raise ValueError("tier role is invalid")
    applicable = tier.get("applicable")
    if applicable is False:
        if not isinstance(tier.get("reason"), str) or not tier["reason"]:
            raise ValueError("an inapplicable tier must explain why")
        return
    if applicable is not True:
        raise ValueError("tier applicability must be explicit")
    required = {
        "id",
        "role",
        "applicable",
        "evaluator",
        "replication",
        "promotion",
        "cost_limit",
    }
    if set(tier) != required:
        raise ValueError(f"applicable tier {tier_id} fields are incomplete")
    evaluator = tier["evaluator"]
    if (
        not isinstance(evaluator, dict)
        or not isinstance(evaluator.get("command"), list)
        or not evaluator["command"]
        or not all(isinstance(item, str) and item for item in evaluator["command"])
        or not isinstance(evaluator.get("result_adapter"), str)
        or evaluator["result_adapter"] not in _RESULT_ADAPTERS
        or not isinstance(evaluator.get("timeout_seconds"), (int, float))
        or evaluator["timeout_seconds"] <= 0
    ):
        raise ValueError(f"tier {tier_id} evaluator is invalid")
    validate_replication(tier["replication"], role)
    cost_limit = tier["cost_limit"]
    if not isinstance(cost_limit, dict) or set(cost_limit) != set(
        RESOURCE_DIMENSIONS
    ):
        raise ValueError(f"tier {tier_id} cost limit is incomplete")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in cost_limit.values()
    ):
        raise ValueError(f"tier {tier_id} cost limit is invalid")
    promotion = tier["promotion"]
    kind = promotion.get("kind")
    if role == "correctness" and kind != "all_guards":
        raise ValueError(f"tier {tier_id} correctness promotion is invalid")
    minimum_relative = promotion.get("minimum_relative_improvement")
    noise_multiplier = promotion.get("noise_multiplier")
    if role in {"proxy", "representative"} and (
        kind != "relative_improvement"
        or isinstance(minimum_relative, bool)
        or not isinstance(minimum_relative, (int, float))
        or not 0 < float(minimum_relative) < 1
        or isinstance(noise_multiplier, bool)
        or not isinstance(noise_multiplier, (int, float))
        or float(noise_multiplier) < 1
    ):
        raise ValueError(f"tier {tier_id} relative promotion is invalid")
    acceptance_kind = {
        "holdout": "portfolio_acceptance",
        "transfer": "transfer_acceptance",
    }
    if role in acceptance_kind and kind != acceptance_kind[role]:
        raise ValueError(f"tier {tier_id} acceptance promotion is invalid")
    if role in acceptance_kind and (
        not isinstance(promotion.get("local_kernel"), str)
        or not promotion["local_kernel"]
        or not isinstance(promotion.get("local_metric"), str)
        or not promotion["local_metric"]
        or _speedup(
            promotion.get("minimum_local_speedup"), "local-kernel floor"
        )
        < goal_floor
    ):
        raise ValueError(f"tier {tier_id} local-kernel promotion is invalid")
    minimum = promotion.get("minimum_accepted_speedup")
    if role in {"representative", "holdout", "transfer"} and (
        minimum is None or _speedup(minimum, "tier acceptance floor") < goal_floor
    ):
        raise ValueError(f"tier {tier_id} must retain the 5x acceptance floor")
    if role in acceptance_kind and (
        type(promotion.get("log_n")) is not int or promotion["log_n"] < 26
    ):
        raise ValueError(f"tier {tier_id} target scale is invalid")


def validate_template(template: dict[str, Any], root: Path) -> None:
    required = {
        "schema_version",
        "slot_id",
        "kernel",
        "goal",
        "hypothesis",
        "metric",
        "portfolio_contract",
        "registry_contract",
        "scope",
        "search_space",
        "baseline_params",
        "budget",
        "evaluation",
        "collaboration",
    }
    if not isinstance(template, dict) or set(template) != required:
        raise ValueError("template fields are incomplete")
    if template["schema_version"] != TEMPLATE_SCHEMA_VERSION:
        raise ValueError("unsupported template schema")

    registry_path = _relative_file(root, template["registry_contract"], "registry")
    registry = json.loads(registry_path.read_text())
    slots = {slot["id"] for slot in registry.get("slots", [])}
    if template["slot_id"] not in slots:
        raise ValueError("template does not bind a canonical registry slot")
    goal_path = _relative_file(root, template["portfolio_contract"], "portfolio")
    goal = json.loads(goal_path.read_text())
    validate_goal_contract(goal)
    floor = float(goal["primary_metric"]["minimum_accepted_speedup"])

    metric = template["metric"]
    if metric.get("direction") != "max" or metric.get("unit") != "x":
        raise ValueError("v2 kernel searches must maximize a speedup")
    search_space = template["search_space"]
    baseline = template["baseline_params"]
    if not isinstance(search_space, dict) or set(baseline) != set(search_space):
        raise ValueError("baseline parameters must close the search space")
    for name, values in search_space.items():
        if not isinstance(values, list) or not values or baseline[name] not in values:
            raise ValueError(f"baseline parameter {name} is outside the search space")

    scope = template["scope"]
    if not isinstance(scope, dict) or set(scope) != {"editable", "frozen"}:
        raise ValueError("template scope is invalid")
    editable = set(scope["editable"])
    frozen = set(scope["frozen"])
    if not editable or editable & frozen:
        raise ValueError("template editable and frozen scopes are invalid")
    for path in editable | frozen:
        _relative_file(root, path, "scope")
    if template["portfolio_contract"] not in frozen:
        raise ValueError("the goal contract must be frozen")
    if template["registry_contract"] not in frozen:
        raise ValueError("the kernel registry must be frozen")

    validate_budget(template["budget"])
    evaluation = template["evaluation"]
    if (
        not isinstance(evaluation, dict)
        or set(evaluation) != {"tier_result_schema", "tiers"}
        or evaluation["tier_result_schema"] != TIER_RESULT_SCHEMA
    ):
        raise ValueError("evaluation contract is invalid")
    tiers = evaluation["tiers"]
    if not isinstance(tiers, list) or not tiers:
        raise ValueError("evaluation tiers are empty")
    ids: set[str] = set()
    roles: set[str] = set()
    ordered_roles: list[str] = []
    for tier in tiers:
        _validate_tier(tier, floor)
        if tier["id"] in ids:
            raise ValueError("evaluation tier ids are duplicated")
        ids.add(tier["id"])
        if tier.get("applicable") is True:
            roles.add(tier["role"])
            ordered_roles.append(tier["role"])
    if not {"representative", "holdout", "transfer"} <= roles:
        raise ValueError("evaluation requires representative, holdout, and transfer tiers")
    rank = {
        "correctness": 0,
        "proxy": 1,
        "representative": 2,
        "holdout": 3,
        "transfer": 4,
    }
    if (
        ordered_roles != sorted(ordered_roles, key=rank.__getitem__)
        or ordered_roles.count("representative") != 1
        or ordered_roles.count("holdout") != 1
        or ordered_roles.count("transfer") != 1
    ):
        raise ValueError("evaluation tiers are not in canonical stage order")
    executable = {
        tier["role"]: tier for tier in tiers if tier.get("applicable") is True
    }
    holdout = executable["holdout"]
    portfolio = goal["portfolio_acceptance"]
    if (
        holdout["promotion"]["log_n"] != portfolio["log_n"]
        or holdout["replication"]["included_pairs"] != portfolio["pairs"]
    ):
        raise ValueError("holdout tier does not match portfolio acceptance")
    transfer = executable["transfer"]
    transfer_contract = goal["transfer_validation"]
    if (
        transfer["promotion"]["log_n"]
        not in transfer_contract["required_log_trace_sizes"]
        or transfer["replication"]["included_pairs"]
        != transfer_contract["pairs"]
        or float(transfer["promotion"]["minimum_accepted_speedup"])
        < float(transfer_contract["minimum_speedup"])
    ):
        raise ValueError("transfer tier does not match transfer acceptance")
    reserves = {reserve["id"] for reserve in template["budget"]["reserves"]}
    if not {
        "representative_revalidation",
        "piop_holdout",
        "piop_transfer",
    } <= reserves:
        raise ValueError("production validation reserves are incomplete")
    collaboration = template["collaboration"]
    if collaboration != {
        "proposal_agents": 3,
        "worktree_writer": "root",
        "evaluator_owner": "root",
        "evaluator_concurrency": 1,
    }:
        raise ValueError("template collaboration policy is invalid")
