from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .artifacts import (
    canonical_json,
    runtime_artifact_controller_paths,
    runtime_artifact_result_adapter,
    runtime_artifact_result_adapters,
    sha256,
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
    "metal_piop_v10",
} | runtime_artifact_result_adapters()
PHASE_CHECKPOINT_FIELDS = {
    "materialize_gpu_active_ms": "materialize",
    "first_bind_gpu_active_ms": "first_bind",
    "dense_rounds_gpu_active_ms": "dense_rounds",
    "openings_gpu_active_ms": "openings",
}
ITERATION_PROFILE_CONTROLLER_PATHS = (
    "scripts/metal_autoresearch.py",
    "scripts/metal_research/artifacts.py",
    "scripts/metal_research/attempt.py",
    "scripts/metal_research/binaries.py",
    "scripts/metal_research/contracts.py",
    "scripts/metal_research/iteration_profile.py",
    "scripts/metal_research/paired.py",
    "scripts/metal_research/results.py",
    "scripts/metal_research/runner.py",
    "scripts/metal_research/versions.py",
)
ITERATION_PROFILE_SOURCE_PATHS = (
    "crates/jolt-kernels/src/metal/solinas/fp128.metal",
    "crates/jolt-kernels/src/metal/solinas/simd_reduce.metal",
    "crates/jolt-kernels/src/metal/solinas/spartan_outer_common.metal",
    "crates/jolt-kernels/src/metal/solinas/outer_remainder/shader.metal",
    "crates/jolt-kernels/src/metal/solinas/outer_remainder/opening_padded_56.metal",
)
ITERATION_PROFILE_SOLINAS_OFFSET = 0xFFFF_A7F7


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


def phase_checkpoint_record(
    phase: dict[str, Any],
    result: dict[str, Any],
    candidates_admitted: int,
) -> dict[str, Any]:
    checkpoint = phase["checkpoint"]
    due = candidates_admitted >= int(checkpoint["after_candidates"])
    record: dict[str, Any] = {
        "phase_id": phase["id"],
        "after_candidates": checkpoint["after_candidates"],
        "due": due,
        "passed": None,
        "metrics": [],
    }
    if not due:
        return record
    if result.get("fingerprint", {}).get("log_n") != checkpoint["scale_log_n"]:
        raise ValueError("phase checkpoint result used the wrong trace scale")
    timings = result.get("telemetry", {}).get("candidate_phase_gpu_active_ns")
    if not isinstance(timings, dict):
        raise ValueError("phase checkpoint telemetry is missing")
    passed = True
    metrics = []
    for contract in checkpoint["metrics"]:
        field = PHASE_CHECKPOINT_FIELDS.get(contract["name"])
        value = timings.get(field) if field is not None else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("phase checkpoint telemetry is invalid")
        observed_ms = float(value) / 1_000_000.0
        threshold = float(contract["threshold"])
        metric_passed = (
            observed_ms <= threshold
            if contract["comparison"] == "lte"
            else observed_ms >= threshold
        )
        metrics.append(
            {
                "name": contract["name"],
                "comparison": contract["comparison"],
                "threshold": threshold,
                "observed_ms": observed_ms,
                "passed": metric_passed,
            }
        )
        passed = passed and metric_passed
    record["metrics"] = metrics
    record["passed"] = passed
    return record


def iteration_profile_evaluator_fingerprint(
    profile: dict[str, Any], root: Path
) -> dict[str, str]:
    path = _relative_file(
        root, profile["evidence_path"], "iteration profile evidence"
    )
    payload = path.read_bytes()
    if sha256(payload) != _hex_digest(
        profile["evidence_sha256"], "iteration evidence digest"
    ):
        raise ValueError("iteration profile evidence digest does not match")
    try:
        evidence = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("iteration profile evidence is invalid JSON") from error
    evaluator = evidence.get("evaluator") if isinstance(evidence, dict) else None
    if not isinstance(evaluator, dict):
        raise ValueError("iteration profile evaluator is incomplete")
    result = {
        "runner_binary_sha256": _hex_digest(
            evaluator.get("runner_binary_sha256"), "iteration runner digest"
        )
    }
    source_digest = evaluator.get("runner_source_sha256")
    if source_digest is not None:
        result["runner_source_sha256"] = _hex_digest(
            source_digest, "iteration runner source digest"
        )
    return result


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
    if role == "representative":
        if _speedup(
            promotion.get("minimum_accepted_speedup"), "tier acceptance floor"
        ) < goal_floor:
            raise ValueError(f"tier {tier_id} must retain the 5x acceptance floor")
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

    runtime_artifact = template.get("runtime_artifact")
    if runtime_artifact is not None:
        runtime_tier = runtime_artifact["tier_id"]
        runtime_binaries = [
            contract
            for contract in contracts.values()
            if runtime_tier in contract["consumer_tiers"]
        ]
        if len(runtime_binaries) != 1:
            raise ValueError("runtime artifact must bind one sealed binary")
        runtime_binary = runtime_binaries[0]
        command = runtime_binary["build"]["command"]
        cargo_features: set[str] = set()
        for index, argument in enumerate(command):
            if argument == "--features" and index + 1 < len(command):
                cargo_features.update(command[index + 1].split(","))
            elif argument.startswith("--features="):
                cargo_features.update(argument.removeprefix("--features=").split(","))
        if (
            runtime_artifact["source_path"] in runtime_binary["source_paths"]
            or "metal-runtime-artifact-only" not in cargo_features
        ):
            raise ValueError(
                "sealed proxy must compile in runtime-artifact-only mode"
            )


def _validate_mechanism_phase(
    phase: Any, budget: dict[str, Any]
) -> None:
    required = {
        "id",
        "hypothesis",
        "analytical_ceiling",
        "timebox",
        "checkpoint",
        "success",
        "kill_or_redesign",
    }
    optional = {"candidate_params"}
    if (
        not isinstance(phase, dict)
        or not required <= set(phase)
        or set(phase) - required - optional
    ):
        raise ValueError("mechanism phase fields are incomplete")
    if not isinstance(phase["id"], str) or _ID.fullmatch(phase["id"]) is None:
        raise ValueError("mechanism phase id is invalid")
    for field in ("hypothesis", "kill_or_redesign"):
        if not isinstance(phase[field], str) or not phase[field].strip():
            raise ValueError(f"mechanism phase {field} is invalid")

    ceiling = phase["analytical_ceiling"]
    if not isinstance(ceiling, dict) or set(ceiling) != {
        "control_ms",
        "parent_member_ms",
        "best_case_member_ms",
        "best_case_speedup",
        "basis",
    }:
        raise ValueError("mechanism phase analytical ceiling is invalid")
    control_ms = _positive(ceiling["control_ms"], "phase control latency")
    parent_ms = _positive(ceiling["parent_member_ms"], "phase parent latency")
    best_ms = _positive(ceiling["best_case_member_ms"], "phase best-case latency")
    best_speedup = _speedup(ceiling["best_case_speedup"], "phase best-case speedup")
    if (
        best_ms >= parent_ms
        or not isinstance(ceiling["basis"], str)
        or not ceiling["basis"].strip()
        or not math.isclose(best_speedup, control_ms / best_ms, rel_tol=1e-6)
    ):
        raise ValueError("mechanism phase analytical ceiling is inconsistent")

    timebox = phase["timebox"]
    if not isinstance(timebox, dict) or set(timebox) != {
        "max_search_calendar_seconds",
        "max_candidates_admitted",
    }:
        raise ValueError("phase timebox is invalid")
    max_candidates = timebox["max_candidates_admitted"]
    max_seconds = timebox["max_search_calendar_seconds"]
    total = budget["total"]
    if (
        type(max_candidates) is not int
        or max_candidates <= 0
        or max_candidates > total["max_candidates_admitted"]
        or isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
        or max_seconds > total["max_calendar_seconds"]
    ):
        raise ValueError("phase timebox is invalid")

    checkpoint = phase["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "after_candidates",
        "scale_log_n",
        "metrics",
    }:
        raise ValueError("mechanism phase checkpoint is invalid")
    if (
        type(checkpoint["after_candidates"]) is not int
        or not 1 <= checkpoint["after_candidates"] <= max_candidates
        or type(checkpoint["scale_log_n"]) is not int
        or not 4 <= checkpoint["scale_log_n"] <= 30
        or not isinstance(checkpoint["metrics"], list)
        or not checkpoint["metrics"]
    ):
        raise ValueError("mechanism phase checkpoint is invalid")
    metric_names: set[str] = set()
    for metric in checkpoint["metrics"]:
        if (
            not isinstance(metric, dict)
            or set(metric) != {"name", "comparison", "threshold"}
            or not isinstance(metric["name"], str)
            or metric["name"] not in PHASE_CHECKPOINT_FIELDS
            or metric["name"] in metric_names
            or metric["comparison"] not in {"lte", "gte"}
        ):
            raise ValueError("mechanism phase checkpoint metric is invalid")
        _positive(metric["threshold"], "phase checkpoint threshold")
        metric_names.add(metric["name"])

    success = phase["success"]
    if not isinstance(success, dict) or set(success) != {
        "maximum_member_ms",
        "minimum_relative_improvement",
    }:
        raise ValueError("mechanism phase success gate is invalid")
    maximum_member_ms = _positive(
        success["maximum_member_ms"], "phase success latency"
    )
    relative = success["minimum_relative_improvement"]
    if (
        maximum_member_ms >= parent_ms
        or isinstance(relative, bool)
        or not isinstance(relative, (int, float))
        or not math.isfinite(relative)
        or not 0 < relative < 1
    ):
        raise ValueError("mechanism phase success gate is invalid")


def _hex_digest(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} is invalid")
    return value


def _profile_compile_context(
    context: Any, closure: dict[str, Any], role: str
) -> int:
    fields = {
        "source_assembly_ns",
        "library_compile_ns",
        "source_bytes",
        "assembled_source_sha256",
        "pipeline_set_ns",
        "pipeline_set_total_ns",
    }
    if not isinstance(context, dict) or set(context) != fields:
        raise ValueError("iteration compilation context is incomplete")
    integer_fields = ("source_assembly_ns", "library_compile_ns", "source_bytes")
    if any(type(context[field]) is not int or context[field] <= 0 for field in integer_fields):
        raise ValueError("iteration compilation context is invalid")
    _hex_digest(context["assembled_source_sha256"], "assembled source digest")
    pipeline = context["pipeline_set_ns"]
    if (
        not isinstance(pipeline, list)
        or len(pipeline) != 5
        or any(type(value) is not int or value <= 0 for value in pipeline)
        or type(context["pipeline_set_total_ns"]) is not int
        or context["pipeline_set_total_ns"] != sum(pipeline)
        or context["source_bytes"] != closure[f"{role}_assembled_source_bytes"]
        or context["assembled_source_sha256"]
        != closure[f"{role}_assembled_source_sha256"]
    ):
        raise ValueError("iteration compilation context does not match the closure")
    return int(context["library_compile_ns"])


def _validate_profile_cycle(
    name: str,
    cycle: Any,
    closure: dict[str, Any],
    maximum_overhead: float,
    root: Path,
    frozen: set[str],
    proxy: dict[str, Any],
    phase: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    fields = {
        "controller_wall_ns",
        "subprocess_wall_ns",
        "parse_validate_checkpoint_ns",
        "controller_overhead_ns",
        "raw_result_path",
        "raw_result_sha256",
        "result_bytes",
        "successor_speedup",
        "gpu_active_total_ns",
        "output_sha256",
        "candidate_phase_gpu_active_ns",
        "compilation",
        "guards",
        "checkpoint",
    }
    if not isinstance(cycle, dict) or set(cycle) != fields:
        raise ValueError(f"iteration {name} fields are incomplete")
    positive_integers = (
        "controller_wall_ns",
        "subprocess_wall_ns",
        "parse_validate_checkpoint_ns",
        "controller_overhead_ns",
        "result_bytes",
        "gpu_active_total_ns",
    )
    if any(type(cycle[field]) is not int or cycle[field] <= 0 for field in positive_integers):
        raise ValueError(f"iteration {name} timing is invalid")
    if (
        cycle["controller_wall_ns"]
        != cycle["subprocess_wall_ns"] + cycle["controller_overhead_ns"]
        or cycle["parse_validate_checkpoint_ns"] > cycle["controller_overhead_ns"]
        or cycle["controller_overhead_ns"] / cycle["controller_wall_ns"]
        > maximum_overhead
    ):
        raise ValueError(f"iteration {name} overhead exceeds its measured contract")
    raw_path = cycle["raw_result_path"]
    path = _relative_file(root, raw_path, "raw iteration result")
    if raw_path not in frozen:
        raise ValueError("raw iteration result must be frozen")
    raw_payload = path.read_bytes()
    if (
        len(raw_payload) != cycle["result_bytes"]
        or sha256(raw_payload)
        != _hex_digest(cycle["raw_result_sha256"], "raw iteration result digest")
    ):
        raise ValueError(f"iteration {name} raw result digest does not match")
    try:
        raw_result = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"iteration {name} raw result is invalid JSON") from error
    from .results import adapt_result, validate_tier_result

    normalized, _ = adapt_result(proxy, raw_result, "OuterRemainder")
    validate_tier_result(normalized, proxy)
    _hex_digest(cycle["output_sha256"], "iteration output digest")
    speedup = cycle["successor_speedup"]
    if (
        isinstance(speedup, bool)
        or not isinstance(speedup, (int, float))
        or not math.isfinite(speedup)
        or not 0.95 <= float(speedup) <= 1.05
    ):
        raise ValueError(f"iteration {name} inert successor is not equivalent")

    guards = cycle["guards"]
    expected_guards = {
        "all_exact",
        "correctness_exact",
        "gpu_timestamps_exact",
        "metal_phase_schedule_exact",
        "resident_row_handle_lifecycle_exact",
        "runtime_artifacts_exact",
        "target_scale",
    }
    if (
        not isinstance(guards, dict)
        or set(guards) != expected_guards
        or any(value is not True for value in guards.values())
    ):
        raise ValueError(f"iteration {name} exact guards are invalid")
    if raw_result.get("guards") != guards:
        raise ValueError(f"iteration {name} guards do not match the raw result")

    phases = cycle["candidate_phase_gpu_active_ns"]
    phase_fields = {"materialize", "first_bind", "dense_rounds", "openings"}
    if (
        not isinstance(phases, dict)
        or set(phases) != phase_fields
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in phases.values()
        )
    ):
        raise ValueError(f"iteration {name} phase evidence is invalid")
    if normalized.get("telemetry", {}).get("candidate_phase_gpu_active_ns") != phases:
        raise ValueError(f"iteration {name} phases do not match the raw result")
    expected_checkpoint = phase_checkpoint_record(
        phase,
        normalized,
        int(phase["checkpoint"]["after_candidates"]),
    )
    if cycle["checkpoint"] != expected_checkpoint:
        raise ValueError(
            f"iteration {name} checkpoint is not derived from phase evidence"
        )

    compilation = cycle["compilation"]
    if (
        not isinstance(compilation, dict)
        or set(compilation) != {"context_order", "parent", "candidate"}
        or compilation["context_order"] != ["parent", "candidate"]
    ):
        raise ValueError(f"iteration {name} compilation evidence is invalid")
    if raw_result.get("telemetry", {}).get("compilation") != compilation:
        raise ValueError(f"iteration {name} compilation does not match the raw result")
    if (
        not math.isclose(
            float(normalized["primary"]["value"]),
            float(speedup),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or raw_result.get("resources", {}).get("gpu_active_total_ns")
        != cycle["gpu_active_total_ns"]
        or raw_result.get("samples", [{}])[0]
        .get("candidate", {})
        .get("output_sha256")
        != cycle["output_sha256"]
    ):
        raise ValueError(f"iteration {name} summary does not match the raw result")
    _profile_compile_context(compilation["parent"], closure, "parent")
    return (
        _profile_compile_context(compilation["candidate"], closure, "candidate"),
        raw_result,
    )


def _validate_iteration_profile(
    profile: Any,
    root: Path,
    editable: set[str],
    frozen: set[str],
    proxy: dict[str, Any],
    phase: dict[str, Any],
    verify_editable_sources: bool,
    verify_live_sources: bool,
) -> None:
    required = {
        "profile_base_revision",
        "evidence_path",
        "evidence_sha256",
        "minimum_valid_proxy_cycles_per_hour",
        "maximum_controller_overhead_fraction",
    }
    if not isinstance(profile, dict) or set(profile) != required:
        raise ValueError("iteration profile fields are incomplete")
    revision = profile["profile_base_revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("iteration profile revision is invalid")
    evidence_path = profile["evidence_path"]
    path = _relative_file(root, evidence_path, "iteration profile evidence")
    if evidence_path not in frozen:
        raise ValueError("iteration profile evidence must be frozen")
    encoded = path.read_bytes()
    if sha256(encoded) != _hex_digest(
        profile["evidence_sha256"], "iteration evidence digest"
    ):
        raise ValueError("iteration profile evidence digest does not match")
    try:
        evidence = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("iteration profile evidence is invalid JSON") from error
    schema = evidence.get("schema") if isinstance(evidence, dict) else None
    schema_version = (
        evidence.get("schema_version") if isinstance(evidence, dict) else None
    )
    is_v2 = schema == "outer_remainder_iteration_profile_v2" and schema_version == 2
    is_v3 = schema == "outer_remainder_iteration_profile_v3" and schema_version == 3
    evidence_fields = {
        "schema",
        "schema_version",
        "created_at",
        "profile_base_revision",
        "machine",
        "evaluator",
        "minimal_closure",
        "cold_cycle",
        "warm_cycle",
    }
    if is_v3:
        evidence_fields.add("controller_sources")
    if (
        not isinstance(evidence, dict)
        or set(evidence) != evidence_fields
        or not (is_v2 or is_v3)
        or evidence["profile_base_revision"] != revision
        or not isinstance(evidence["created_at"], str)
        or not evidence["created_at"]
    ):
        raise ValueError("iteration profile evidence contract is invalid")

    if is_v3:
        controller_sources = evidence["controller_sources"]
        if (
            not isinstance(controller_sources, list)
            or [record.get("path") for record in controller_sources]
            != list(ITERATION_PROFILE_CONTROLLER_PATHS)
            or not set(ITERATION_PROFILE_CONTROLLER_PATHS) <= frozen
        ):
            raise ValueError("iteration profile controller sources are invalid")
        for record in controller_sources:
            if not isinstance(record, dict) or set(record) != {
                "path",
                "bytes",
                "sha256",
            }:
                raise ValueError("iteration profile controller source is invalid")
            if type(record["bytes"]) is not int or record["bytes"] <= 0:
                raise ValueError("iteration profile controller source is invalid")
            digest = _hex_digest(
                record["sha256"], "iteration controller source digest"
            )
            if not verify_live_sources:
                continue
            payload = _relative_file(
                root, record["path"], "iteration profile controller source"
            ).read_bytes()
            if record["bytes"] != len(payload) or digest != sha256(payload):
                raise ValueError(
                    "iteration profile controller changed since profiling"
                )

    machine = evidence["machine"]
    machine_fields = {
        "device_name",
        "os_product_version",
        "os_build_version",
        "macos_sdk_version",
        "rustc_release",
        "rustc_commit_hash",
        "rustc_host",
        "llvm_version",
        "cargo_release",
    }
    if (
        not isinstance(machine, dict)
        or set(machine) != machine_fields
        or any(not isinstance(value, str) or not value for value in machine.values())
        or machine["device_name"] != "Apple M4 Max"
        or machine["rustc_host"] != "aarch64-apple-darwin"
    ):
        raise ValueError("iteration profile machine is invalid")

    evaluator = evidence["evaluator"]
    evaluator_fields = {
        "result_adapter",
        "runner_binary_sha256",
        "log_n",
        "pairs",
        "excluded_warmup_pairs",
        "rayon_threads",
        "binding_plan",
        "cpu_tail_elements",
        "trace_cutoff_elements",
        "parent_artifact_sha256",
        "candidate_artifact_sha256",
        "parent_outer_source_sha256",
        "candidate_outer_source_sha256",
    }
    if is_v3:
        evaluator_fields.add("runner_source_sha256")
    if not isinstance(evaluator, dict) or set(evaluator) != evaluator_fields:
        raise ValueError("iteration profile evaluator is incomplete")
    _hex_digest(evaluator["runner_binary_sha256"], "iteration runner digest")
    if is_v3:
        _hex_digest(
            evaluator["runner_source_sha256"],
            "iteration runner source digest",
        )
    _hex_digest(evaluator["parent_artifact_sha256"], "parent artifact digest")
    _hex_digest(evaluator["candidate_artifact_sha256"], "candidate artifact digest")
    _hex_digest(evaluator["parent_outer_source_sha256"], "parent source digest")
    _hex_digest(evaluator["candidate_outer_source_sha256"], "candidate source digest")
    env = proxy["evaluator"]["env"]
    if (
        evaluator["result_adapter"] != proxy["evaluator"]["result_adapter"]
        or evaluator["log_n"] != proxy["promotion"]["log_n"]
        or evaluator["pairs"] != proxy["replication"]["included_pairs"]
        or evaluator["excluded_warmup_pairs"]
        != proxy["replication"]["excluded_warmup_pairs"]
        or evaluator["rayon_threads"] != int(env["RAYON_NUM_THREADS"])
        or evaluator["binding_plan"] != "b_only_v1"
        or evaluator["cpu_tail_elements"] <= 0
        or evaluator["trace_cutoff_elements"] <= 0
    ):
        raise ValueError("iteration profile does not match the proxy tier")

    closure = evidence["minimal_closure"]
    closure_fields = {
        "dependency_model",
        "source_fragments",
        "parent_assembled_source_bytes",
        "parent_assembled_source_sha256",
        "candidate_assembled_source_bytes",
        "candidate_assembled_source_sha256",
        "candidate_source_suffix",
    }
    if is_v3:
        closure_fields.add("solinas_offset")
    if (
        not isinstance(closure, dict)
        or set(closure) != closure_fields
        or closure["dependency_model"] != "outer_only_shader_closure_v1"
    ):
        raise ValueError("minimal closure evidence is invalid")
    if is_v3 and (
        type(closure["solinas_offset"]) is not int
        or closure["solinas_offset"] != ITERATION_PROFILE_SOLINAS_OFFSET
    ):
        raise ValueError("minimal closure Solinas offset is invalid")
    expected_paths = list(ITERATION_PROFILE_SOURCE_PATHS)
    fragments = closure["source_fragments"]
    if not isinstance(fragments, list) or [item.get("path") for item in fragments] != expected_paths:
        raise ValueError("minimal closure source fragments are invalid")
    if expected_paths[-1] not in editable or any(
        path not in frozen for path in expected_paths[:-1]
    ):
        raise ValueError("minimal closure source scopes are invalid")
    fragment_payloads: dict[str, bytes] = {}
    for fragment in fragments:
        if not isinstance(fragment, dict) or set(fragment) != {"path", "bytes", "sha256"}:
            raise ValueError("minimal closure source fragment is invalid")
        if type(fragment["bytes"]) is not int or fragment["bytes"] <= 0:
            raise ValueError("minimal closure source fragment size is invalid")
        fragment_digest = _hex_digest(
            fragment["sha256"], "minimal closure source digest"
        )
        if not verify_live_sources or (
            fragment["path"] in editable and not verify_editable_sources
        ):
            continue
        source = _relative_file(root, fragment["path"], "minimal closure source")
        payload = source.read_bytes()
        fragment_payloads[fragment["path"]] = payload
        if (
            fragment["bytes"] != len(payload)
            or fragment_digest != sha256(payload)
        ):
            raise ValueError("minimal closure source changed since profiling")
    suffix = closure["candidate_source_suffix"]
    if (
        not isinstance(suffix, str)
        or not suffix.startswith("\n// iteration profile ")
        or evaluator["parent_outer_source_sha256"]
        != fragments[-1]["sha256"]
    ):
        raise ValueError("minimal closure candidate nonce is invalid")
    if expected_paths[-1] in fragment_payloads:
        candidate_source = fragment_payloads[expected_paths[-1]] + suffix.encode()
        if evaluator["candidate_outer_source_sha256"] != sha256(candidate_source):
            raise ValueError("minimal closure candidate nonce is invalid")
        if is_v3:
            offset = closure["solinas_offset"]
            prefix = f"#define SOLINAS_OFFSET {offset}u\n".encode()
            parent_source = prefix + b"\n".join(
                fragment_payloads[path] for path in expected_paths
            )
            candidate_assembled = prefix + b"\n".join(
                [
                    *(fragment_payloads[path] for path in expected_paths[:-1]),
                    candidate_source,
                ]
            )
            for role, assembled in (
                ("parent", parent_source),
                ("candidate", candidate_assembled),
            ):
                if (
                    closure[f"{role}_assembled_source_bytes"] != len(assembled)
                    or closure[f"{role}_assembled_source_sha256"]
                    != sha256(assembled)
                ):
                    raise ValueError(
                        "minimal closure assembled source does not match its fragments"
                    )
    for field in ("parent_assembled_source_bytes", "candidate_assembled_source_bytes"):
        if type(closure[field]) is not int or closure[field] <= 0:
            raise ValueError("minimal closure assembled source size is invalid")
    for field in ("parent_assembled_source_sha256", "candidate_assembled_source_sha256"):
        _hex_digest(closure[field], "minimal closure assembled source digest")

    maximum_overhead = profile["maximum_controller_overhead_fraction"]
    if (
        isinstance(maximum_overhead, bool)
        or not isinstance(maximum_overhead, (int, float))
        or not math.isfinite(maximum_overhead)
        or not 0 < maximum_overhead <= 0.01
    ):
        raise ValueError("iteration overhead threshold is invalid")
    cold_compile, cold_raw = _validate_profile_cycle(
        "cold cycle",
        evidence["cold_cycle"],
        closure,
        float(maximum_overhead),
        root,
        frozen,
        proxy,
        phase,
    )
    warm_compile, warm_raw = _validate_profile_cycle(
        "warm cycle",
        evidence["warm_cycle"],
        closure,
        float(maximum_overhead),
        root,
        frozen,
        proxy,
        phase,
    )
    fingerprints = [cold_raw.get("fingerprint"), warm_raw.get("fingerprint")]
    telemetry = [cold_raw.get("telemetry"), warm_raw.get("telemetry")]
    if (
        evidence["cold_cycle"]["output_sha256"]
        != evidence["warm_cycle"]["output_sha256"]
        or cold_compile <= 10 * warm_compile
        or not all(isinstance(value, dict) for value in fingerprints + telemetry)
        or fingerprints[0] != fingerprints[1]
        or fingerprints[0].get("fixture") != "resident-outer-remainder-v2"
        or fingerprints[0].get("log_n") != evaluator["log_n"]
        or fingerprints[0].get("pairs") != evaluator["pairs"]
        or fingerprints[0].get("excluded_warmup_pairs")
        != evaluator["excluded_warmup_pairs"]
        or fingerprints[0].get("runner_binary_sha256")
        != evaluator["runner_binary_sha256"]
        or fingerprints[0].get("parent_artifact_sha256")
        != evaluator["parent_artifact_sha256"]
        or fingerprints[0].get("candidate_artifact_sha256")
        != evaluator["candidate_artifact_sha256"]
        or any(record.get("device_name") != machine["device_name"] for record in telemetry)
        or any(record.get("parent_binding_plan") != evaluator["binding_plan"] for record in telemetry)
        or any(record.get("candidate_binding_plan") != evaluator["binding_plan"] for record in telemetry)
        or any(
            record.get("parent_source_sha256")
            != evaluator["parent_outer_source_sha256"]
            for record in telemetry
        )
        or any(
            record.get("candidate_source_sha256")
            != evaluator["candidate_outer_source_sha256"]
            for record in telemetry
        )
    ):
        raise ValueError("iteration cold and warm classifications are invalid")
    throughput = _positive(
        profile["minimum_valid_proxy_cycles_per_hour"],
        "valid proxy cycles per hour target",
    )
    measured_cold = 3_600_000_000_000.0 / float(
        evidence["cold_cycle"]["controller_wall_ns"]
    )
    if throughput > measured_cold:
        raise ValueError("iteration throughput target exceeds cold evidence")


def _validate_search_policy(
    policy: Any,
    collaboration: dict[str, Any],
    search_space: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    required = {
        "regime",
        "proposal_batch_size",
        "proposal_queue_capacity",
        "selection_order",
        "duplicate_key",
        "diversity_key",
        "exploration_reserve_candidates",
        "direct_to_representative_lane",
        "proxy_calibration",
    }
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("search policy fields are incomplete")
    if policy["regime"] != "hill_climb":
        raise ValueError("fresh kernel phases use a hill-climb search parent")
    if (
        type(policy["proposal_batch_size"]) is not int
        or policy["proposal_batch_size"] != collaboration["proposal_agents"]
        or type(policy["proposal_queue_capacity"]) is not int
        or policy["proposal_queue_capacity"] < policy["proposal_batch_size"]
        or type(policy["exploration_reserve_candidates"]) is not int
        or policy["exploration_reserve_candidates"] < 0
        or policy["direct_to_representative_lane"] is not True
    ):
        raise ValueError("search policy capacity is invalid")
    if policy["selection_order"] != [
        "guard_feasibility",
        "expected_information_per_cost",
        "predicted_member_latency",
        "complexity",
    ]:
        raise ValueError("search policy ordering is invalid")
    for field in ("duplicate_key", "diversity_key"):
        if not isinstance(policy[field], str) or not policy[field]:
            raise ValueError(f"search policy {field} is invalid")

    calibration = policy["proxy_calibration"]
    if not isinstance(calibration, dict) or set(calibration) != {
        "sentinels",
        "rank_metric",
        "minimum_rank_agreement",
        "material_effect_threshold",
        "maximum_material_inversions",
        "audit_every_candidates",
        "on_material_misranking",
    }:
        raise ValueError("proxy calibration policy is incomplete")
    if (
        calibration["rank_metric"] != "kendall_tau_b"
        or calibration["on_material_misranking"]
        != "disable_and_require_phase_change"
        or type(calibration["maximum_material_inversions"]) is not int
        or calibration["maximum_material_inversions"] < 0
        or type(calibration["audit_every_candidates"]) is not int
        or calibration["audit_every_candidates"] < 1
    ):
        raise ValueError("proxy calibration policy is invalid")
    agreement = calibration["minimum_rank_agreement"]
    material = calibration["material_effect_threshold"]
    if (
        isinstance(agreement, bool)
        or not isinstance(agreement, (int, float))
        or not math.isfinite(agreement)
        or not -1 <= agreement <= 1
        or isinstance(material, bool)
        or not isinstance(material, (int, float))
        or not math.isfinite(material)
        or not 0 < material < 1
    ):
        raise ValueError("proxy calibration thresholds are invalid")
    sentinels = calibration["sentinels"]
    if not isinstance(sentinels, list) or len(sentinels) < 3:
        raise ValueError("proxy calibration requires three fixed sentinels")
    ids: set[str] = set()
    parameter_sets: set[bytes] = set()
    baseline_key = canonical_json({name: str(value) for name, value in baseline.items()})
    for sentinel in sentinels:
        if (
            not isinstance(sentinel, dict)
            or set(sentinel) != {"id", "params"}
            or not isinstance(sentinel["id"], str)
            or _ID.fullmatch(sentinel["id"]) is None
            or sentinel["id"] in ids
            or not isinstance(sentinel["params"], dict)
            or set(sentinel["params"]) != set(search_space)
        ):
            raise ValueError("proxy calibration sentinel is invalid")
        ids.add(sentinel["id"])
        params = {name: str(value) for name, value in sentinel["params"].items()}
        if any(
            params[name] not in {str(value) for value in search_space[name]}
            for name in search_space
        ):
            raise ValueError("proxy calibration sentinel leaves the search space")
        key = canonical_json(params)
        if key == baseline_key or key in parameter_sets:
            raise ValueError("proxy calibration sentinels must be unique and nonbaseline")
        parameter_sets.add(key)


def validate_template(
    template: dict[str, Any],
    root: Path,
    *,
    verify_editable_profile_sources: bool = True,
    verify_iteration_profile: bool = True,
    verify_iteration_profile_sources: bool = True,
) -> None:
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
        "mechanism_phase",
        "iteration_profile",
        "search_policy",
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
    profile_path = Path(template["iteration_profile"]["evidence_path"])
    refresh_outputs = {
        profile_path.as_posix(),
        profile_path.with_name(f"{profile_path.stem}.cold.raw.json").as_posix(),
        profile_path.with_name(f"{profile_path.stem}.warm.raw.json").as_posix(),
    }
    for path in editable | frozen:
        if not verify_iteration_profile and path in refresh_outputs:
            _relative_path(path, "iteration profile refresh output")
        else:
            _relative_file(root, path, "scope")
    if template["portfolio_contract"] not in frozen:
        raise ValueError("the goal contract must be frozen")
    if template["registry_contract"] not in frozen:
        raise ValueError("the kernel registry must be frozen")

    validate_budget(template["budget"])
    _validate_mechanism_phase(template["mechanism_phase"], template["budget"])
    candidate_params = template["mechanism_phase"].get("candidate_params")
    if candidate_params is not None:
        if (
            not isinstance(candidate_params, dict)
            or set(candidate_params) != set(search_space)
            or any(
                candidate_params[name] not in search_space[name]
                for name in search_space
            )
            or candidate_params == baseline
        ):
            raise ValueError("mechanism phase fixed candidate is invalid")
    _validate_search_policy(
        template["search_policy"],
        template["collaboration"],
        search_space,
        baseline,
    )

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
    proxy = executable.get("proxy")
    if (
        proxy is None
        or proxy["promotion"].get("log_n")
        != template["mechanism_phase"]["checkpoint"]["scale_log_n"]
    ):
        raise ValueError("phase checkpoint scale must match the proxy tier")
    if verify_iteration_profile:
        _validate_iteration_profile(
            template["iteration_profile"],
            root,
            editable,
            frozen,
            proxy,
            template["mechanism_phase"],
            verify_editable_profile_sources,
            verify_iteration_profile_sources,
        )
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
    protected_calendar = 0.0
    for reserve_id, tier in validation_tiers.items():
        reserve = reserves[reserve_id]
        invocations = reserve["invocations"]
        protected_calendar += invocations * float(
            tier["evaluator"]["timeout_seconds"]
        )
        if invocations < 2:
            raise ValueError(f"{reserve_id} must protect one retry")
        for resource, cost_limit in tier["cost_limit"].items():
            if float(reserve["resources"].get(resource, 0.0)) < (
                invocations * float(cost_limit)
            ):
                raise ValueError(f"{reserve_id} retry resources are not protected")
    if (
        float(
            template["mechanism_phase"]["timebox"][
                "max_search_calendar_seconds"
            ]
        )
        + protected_calendar
        > float(template["budget"]["total"]["max_calendar_seconds"])
    ):
        raise ValueError("phase and production calendar reserves are overcommitted")
    collaboration = template["collaboration"]
    if collaboration != {
        "proposal_agents": 3,
        "worktree_writer": "root",
        "evaluator_owner": "root",
        "evaluator_concurrency": 1,
    }:
        raise ValueError("template collaboration policy is invalid")
