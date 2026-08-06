from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .artifacts import (
    runtime_artifact_controller_paths,
    runtime_artifact_result_adapter,
    runtime_artifact_result_adapters,
    validate_runtime_artifact_contract,
)
from .attempt import unsafe_environment_name
from .budget import RESOURCE_DIMENSIONS, validate_budget
from .paired import validate_replication
from .versions import (
    GOAL_SCHEMA_VERSION,
    TEMPLATE_SCHEMA_VERSION,
    TIER_RESULT_SCHEMA,
)


_ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
_ROLES = {"correctness", "proxy", "representative", "holdout", "transfer"}
_RESULT_ADAPTERS = {
    "outer_remainder_screen_v1",
    "outer_remainder_successor_v1",
    "outer_remainder_v3",
    "metal_piop_v7",
} | runtime_artifact_result_adapters()


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


def _relative_path(value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} path must be a nonempty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} path must stay within the repository")
    return relative


def _speedup(value: Any, description: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 1.0
    ):
        raise ValueError(f"{description} must be a finite speedup")
    return float(value)


def _positive(value: Any, description: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{description} must be finite and positive")
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
        or _speedup(
            transfer.get("kernel_minimum_speedup"), "kernel transfer floor"
        )
        < floor
        or _speedup(
            transfer.get("portfolio_minimum_speedup"),
            "portfolio transfer floor",
        )
        < floor
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
    if not isinstance(evaluator, dict):
        raise ValueError(f"tier {tier_id} evaluator is invalid")
    evaluator_env = evaluator.get("env", {})
    if (
        not isinstance(evaluator.get("command"), list)
        or not evaluator["command"]
        or not all(isinstance(item, str) and item for item in evaluator["command"])
        or not isinstance(evaluator.get("result_adapter"), str)
        or evaluator["result_adapter"] not in _RESULT_ADAPTERS
        or not isinstance(evaluator.get("timeout_seconds"), (int, float))
        or evaluator["timeout_seconds"] <= 0
        or not isinstance(evaluator_env, dict)
        or not all(
            isinstance(name, str)
            and name
            and not name.startswith("JOLT_AUTORESEARCH_")
            and not unsafe_environment_name(name)
            and isinstance(value, str)
            for name, value in evaluator_env.items()
        )
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
    successor_screen = (
        evaluator["result_adapter"] == "outer_remainder_successor_v2"
    )
    if role == "correctness" and kind != "all_guards":
        raise ValueError(f"tier {tier_id} correctness promotion is invalid")
    minimum_relative = promotion.get("minimum_relative_improvement")
    noise_multiplier = promotion.get("noise_multiplier")
    if successor_screen:
        values = (
            promotion.get("clear_loss_ratio"),
            promotion.get("minimum_uncertainty"),
            promotion.get("maximum_calibration_absolute_log_bias"),
            promotion.get("maximum_screen_relative_mad"),
        )
        bias = promotion.get("maximum_calibration_absolute_log_bias")
        uncertainty = promotion.get("minimum_uncertainty")
        clear_loss_ratio = promotion.get("clear_loss_ratio")
        if (
            role != "proxy"
            or kind != "successor_screen"
            or tier["replication"]["excluded_warmup_pairs"] != 1
            or type(promotion.get("inconclusive_retry_limit")) is not int
            or promotion["inconclusive_retry_limit"] not in {0, 1}
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < float(value) < 1
                for value in values
            )
            or float(bias) > float(uncertainty)
            or float(clear_loss_ratio) * math.exp(float(uncertainty)) >= 1.0
        ):
            raise ValueError(f"tier {tier_id} successor promotion is invalid")
    elif role in {"proxy", "representative"} and (
        kind != "relative_improvement"
        or isinstance(minimum_relative, bool)
        or not isinstance(minimum_relative, (int, float))
        or not 0 < float(minimum_relative) < 1
        or isinstance(noise_multiplier, bool)
        or not isinstance(noise_multiplier, (int, float))
        or float(noise_multiplier) < 1
        or isinstance(promotion.get("maximum_relative_mad"), bool)
        or not isinstance(
            promotion.get("maximum_relative_mad"), (int, float)
        )
        or not 0 < float(promotion["maximum_relative_mad"]) < 1
        or isinstance(
            promotion.get("maximum_order_stratum_log_skew"), bool
        )
        or not isinstance(
            promotion.get("maximum_order_stratum_log_skew"), (int, float)
        )
        or not math.isfinite(promotion["maximum_order_stratum_log_skew"])
        or not 0 < float(promotion["maximum_order_stratum_log_skew"]) < 1
    ):
        raise ValueError(f"tier {tier_id} relative promotion is invalid")
    acceptance_kind = {
        "holdout": "kernel_piop_holdout",
        "transfer": "kernel_transfer",
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
    if role == "representative" and _speedup(
        promotion.get("minimum_accepted_speedup"), "tier acceptance floor"
    ) < goal_floor:
        raise ValueError(f"tier {tier_id} must retain the 5x acceptance floor")
    if role == "representative":
        _positive(
            promotion.get("maximum_treatment_ms"),
            "representative latency bar",
        )
    if role in acceptance_kind:
        portfolio_floor = _speedup(
            promotion.get("minimum_portfolio_speedup"),
            "kernel-validation portfolio floor",
        )
        if portfolio_floor >= goal_floor:
            raise ValueError(
                f"tier {tier_id} conflates kernel validation with portfolio acceptance"
            )
        _positive(
            promotion.get("maximum_local_treatment_ms"),
            "kernel-validation latency bar",
        )
    if role in acceptance_kind and (
        type(promotion.get("log_n")) is not int or promotion["log_n"] < 26
    ):
        raise ValueError(f"tier {tier_id} target scale is invalid")


def _validate_sealed_binaries(
    template: dict[str, Any],
    root: Path,
    tiers: list[dict[str, Any]],
    frozen_paths: set[str],
) -> None:
    contracts = template.get("sealed_binaries", {})
    if not isinstance(contracts, dict):
        raise ValueError("sealed binary contracts must be an object")
    applicable = {
        tier["id"]: tier for tier in tiers if tier.get("applicable") is True
    }
    consumers: set[str] = set()
    tokens = {
        binary_id: f"{{sealed_binary:{binary_id}}}" for binary_id in contracts
    }
    if any(_ID.fullmatch(binary_id) is None for binary_id in contracts):
        raise ValueError("sealed binary id is invalid")

    for binary_id, contract in contracts.items():
        fields = {
            "build",
            "source_paths",
            "consumer_tiers",
            "result_fingerprint",
        }
        if not isinstance(contract, dict) or set(contract) != fields:
            raise ValueError(f"sealed binary {binary_id} contract is invalid")
        build = contract["build"]
        if not isinstance(build, dict) or set(build) != {
            "command",
            "output_path",
            "timeout_seconds",
        }:
            raise ValueError(f"sealed binary {binary_id} build is invalid")
        command = build["command"]
        timeout = build["timeout_seconds"]
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(argument, str) and argument for argument in command)
            or any("{sealed_binary:" in argument for argument in command)
            or isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(f"sealed binary {binary_id} build is invalid")
        _relative_path(build["output_path"], f"sealed binary {binary_id} output")

        source_paths = contract["source_paths"]
        if (
            not isinstance(source_paths, list)
            or not source_paths
            or not all(isinstance(path, str) and path for path in source_paths)
            or len(source_paths) != len(set(source_paths))
        ):
            raise ValueError(f"sealed binary {binary_id} sources are invalid")
        for source_path in source_paths:
            _relative_file(root, source_path, f"sealed binary {binary_id} source")
            if source_path not in frozen_paths:
                raise ValueError(
                    f"sealed binary {binary_id} source is outside the frozen closure"
                )

        consumer_tiers = contract["consumer_tiers"]
        if (
            not isinstance(consumer_tiers, list)
            or not consumer_tiers
            or not all(
                isinstance(tier_id, str) and tier_id
                for tier_id in consumer_tiers
            )
            or len(consumer_tiers) != len(set(consumer_tiers))
            or any(tier_id not in applicable for tier_id in consumer_tiers)
            or consumers.intersection(consumer_tiers)
        ):
            raise ValueError(f"sealed binary {binary_id} consumers are invalid")
        consumers.update(consumer_tiers)
        if contract["result_fingerprint"] != [
            "fingerprint",
            "runner_binary_sha256",
        ]:
            raise ValueError(f"sealed binary {binary_id} fingerprint is invalid")

        token = tokens[binary_id]
        for tier_id, tier in applicable.items():
            tier_command = tier["evaluator"]["command"]
            occurrences = tier_command.count(token)
            if tier_id in consumer_tiers and (
                occurrences != 1
                or tier_command[0] != token
                or tier["evaluator"]["result_adapter"]
                != "outer_remainder_successor_v2"
            ):
                raise ValueError(
                    f"sealed binary {binary_id} consumer is not a direct v2 evaluator"
                )
            if tier_id not in consumer_tiers and occurrences:
                raise ValueError(
                    f"sealed binary {binary_id} token appears in a nonconsumer tier"
                )

    allowed_tokens = set(tokens.values())
    for tier in applicable.values():
        evaluator = tier["evaluator"]
        for argument in evaluator["command"]:
            if "{sealed_binary:" in argument and argument not in allowed_tokens:
                raise ValueError("sealed binary command token is invalid")
        reserved_values = list(evaluator.get("env", {}).items())
        reserved_values.extend(
            (binding.get("parameter"), binding.get("flag"))
            for binding in evaluator.get("parameter_bindings", [])
            if isinstance(binding, dict)
        )
        if any(
            "{sealed_binary:" in value
            for pair in reserved_values
            for value in pair
            if isinstance(value, str)
        ):
            raise ValueError(
                "sealed binary tokens are restricted to whole command arguments"
            )
    v2_tiers = {
        tier_id
        for tier_id, tier in applicable.items()
        if tier["evaluator"]["result_adapter"]
        == "outer_remainder_successor_v2"
    }
    if v2_tiers != consumers:
        raise ValueError("every successor v2 tier must consume one sealed binary")


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
    optional = {"runtime_artifact", "sealed_binaries"}
    if (
        not isinstance(template, dict)
        or not required <= set(template)
        or set(template) - required - optional
    ):
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
    if (
        not isinstance(search_space, dict)
        or set(baseline) != set(search_space)
        or any(
            not isinstance(name, str)
            or not name
            or name.startswith("JOLT_AUTORESEARCH_")
            or unsafe_environment_name(name)
            for name in search_space
        )
    ):
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

    runtime_artifact = template.get("runtime_artifact")
    runtime_artifact_adapter = None
    if runtime_artifact is not None:
        validate_runtime_artifact_contract(
            root, runtime_artifact, editable, search_space, baseline
        )
        runtime_artifact_adapter = runtime_artifact_result_adapter(
            runtime_artifact
        )
        if not set(runtime_artifact_controller_paths(runtime_artifact)) <= frozen:
            raise ValueError("runtime artifact controller must be frozen")

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
    runtime_artifact_tiers = [
        tier
        for tier in tiers
        if tier.get("applicable") is True
        and tier["evaluator"]["result_adapter"]
        in runtime_artifact_result_adapters()
    ]
    if any(
        tier.get("applicable") is True
        and tier["evaluator"]["result_adapter"]
        == "outer_remainder_successor_v1"
        for tier in tiers
    ):
        raise ValueError("schema-2 templates cannot execute successor v1")
    if runtime_artifact is not None:
        artifact_tiers = [
            tier
            for tier in tiers
            if tier.get("id") == runtime_artifact["tier_id"]
            and tier.get("applicable") is True
            and tier.get("role") == "proxy"
        ]
        if len(artifact_tiers) != 1:
            raise ValueError("runtime artifact must bind one executable proxy tier")
        if (
            artifact_tiers[0]["evaluator"]["result_adapter"]
            != runtime_artifact_adapter
        ):
            raise ValueError("runtime artifact proxy must use its sealed adapter")
        if runtime_artifact_tiers != artifact_tiers:
            raise ValueError("successor adapter is not bound to runtime artifacts")
    elif runtime_artifact_tiers:
        raise ValueError("successor adapter requires runtime artifacts")
    _validate_sealed_binaries(template, root, tiers, frozen)
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
        or float(transfer["promotion"]["minimum_local_speedup"])
        < float(transfer_contract["kernel_minimum_speedup"])
    ):
        raise ValueError("transfer tier does not match transfer acceptance")
    reserves = {
        reserve["id"]: reserve for reserve in template["budget"]["reserves"]
    }
    required_reserves = {
        "representative_revalidation",
        "piop_holdout",
        "piop_transfer",
    }
    if not required_reserves <= set(reserves):
        raise ValueError("production validation reserves are incomplete")
    validation_tiers = {
        "representative_revalidation": executable["representative"],
        "piop_holdout": holdout,
        "piop_transfer": transfer,
    }
    for reserve_id, tier in validation_tiers.items():
        reserve = reserves[reserve_id]
        invocations = reserve["invocations"]
        if invocations < 2:
            raise ValueError(f"{reserve_id} must protect one retry")
        for resource, cost_limit in tier["cost_limit"].items():
            if float(reserve["resources"].get(resource, 0.0)) < (
                invocations * float(cost_limit)
            ):
                raise ValueError(f"{reserve_id} retry resources are not protected")
    collaboration = template["collaboration"]
    if collaboration != {
        "proposal_agents": 3,
        "worktree_writer": "root",
        "evaluator_owner": "root",
        "evaluator_concurrency": 1,
    }:
        raise ValueError("template collaboration policy is invalid")
