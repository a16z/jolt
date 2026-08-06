from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .attempt import (
    EvaluatorLeaseTimeout,
    evaluator_lease,
    run_attempt,
    sanitized_parent_environment,
    stop_recorded_process_group,
)
from .artifacts import (
    materialize_runtime_artifact_context,
    validate_runtime_artifact_output,
    verify_runtime_artifact_context,
)
from .binaries import (
    declared_source_sha256,
    materialize_sealed_binary,
    prepare_sealed_binary_from_output,
    seal_sealed_binary_store,
    sealed_binary_token,
    verify_sealed_binary_contract,
    verify_sealed_binary_record,
    verify_sealed_binary_store,
)
from .budget import BudgetExhausted, admit_tier, charge_attempt, empty_usage
from .contracts import (
    iteration_profile_evaluator_fingerprint,
    phase_checkpoint_record,
    validate_goal_contract,
    validate_template,
)
from .paired import paired_summary
from .results import adapt_result, validate_tier_result
from .versions import EVENT_SCHEMA_VERSION, RUN_SCHEMA_VERSION


def _legacy() -> Any:
    try:
        from scripts import metal_autoresearch
    except ModuleNotFoundError:
        import metal_autoresearch

    return metal_autoresearch


def _registry() -> Any:
    try:
        from scripts import metal_kernel_registry
    except ModuleNotFoundError:
        import metal_kernel_registry

    return metal_kernel_registry


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def append_event(path: Path, event: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(descriptor, canonical_json(event) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_state(run_dir: Path, state: dict[str, Any]) -> None:
    payload = dict(state)
    payload.pop("state_sha256", None)
    payload["state_sha256"] = sha256(canonical_json(payload))
    encoded = canonical_json(payload)
    state_path = run_dir / "state.json"
    legacy_path = run_dir / "run.json"
    if not state_path.exists():
        if not legacy_path.is_file():
            state_path = legacy_path
        else:
            legacy = read_json(legacy_path)
            if legacy.get("record_kind") != "metal_autoresearch_run_contract_v1":
                state_path = legacy_path
    temporary = state_path.with_name(f".{state_path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(state_path)
    state["state_sha256"] = payload["state_sha256"]


_RUN_CONTRACT_FIELDS = (
    "created_at",
    "template_path",
    "template_sha256",
    "template",
    "goal",
    "registry_binding",
    "base_revision",
    "fingerprint",
)


def _write_run_contract(run_dir: Path, state: dict[str, Any]) -> None:
    path = run_dir / "run.json"
    if path.exists():
        raise ValueError("immutable run contract already exists")
    contract = {
        "schema_version": RUN_SCHEMA_VERSION,
        "record_kind": "metal_autoresearch_run_contract_v1",
        **{field: state[field] for field in _RUN_CONTRACT_FIELDS},
    }
    contract["run_sha256"] = sha256(canonical_json(contract))
    path.write_bytes(canonical_json(contract))


def _load_run_contract(run_dir: Path) -> dict[str, Any]:
    contract = read_json(run_dir / "run.json")
    claimed = contract.get("run_sha256")
    payload = dict(contract)
    payload.pop("run_sha256", None)
    if (
        contract.get("schema_version") != RUN_SCHEMA_VERSION
        or contract.get("record_kind") != "metal_autoresearch_run_contract_v1"
        or not isinstance(claimed, str)
        or claimed != sha256(canonical_json(payload))
    ):
        raise ValueError("immutable run contract digest does not match")
    return contract


def write_inflight(run_dir: Path, inflight: dict[str, Any]) -> None:
    temporary = run_dir / ".inflight.json.tmp"
    temporary.write_bytes(canonical_json(inflight))
    temporary.replace(run_dir / "inflight.json")


def load_state(
    run_dir: Path, *, verify_sealed_binaries: bool = True
) -> dict[str, Any]:
    state_path = run_dir / "state.json"
    contract = _load_run_contract(run_dir) if state_path.is_file() else None
    encoded = (state_path if state_path.is_file() else run_dir / "run.json").read_bytes()
    state = json.loads(encoded)
    if not isinstance(state, dict) or state.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported run state schema")
    claimed = state.get("state_sha256")
    if claimed is None:
        legacy_digest = run_dir / "run.sha256"
        if not legacy_digest.is_file() or legacy_digest.read_text().strip() != sha256(
            encoded
        ):
            raise ValueError("run state digest does not match")
    else:
        payload = dict(state)
        payload.pop("state_sha256")
        if claimed != sha256(canonical_json(payload)):
            raise ValueError("run state digest does not match")
    if contract is not None and any(
        canonical_json(state.get(field)) != canonical_json(contract.get(field))
        for field in _RUN_CONTRACT_FIELDS
    ):
        raise ValueError("mutable state disagrees with the immutable run contract")
    state["usage"] = _derived_usage(run_dir)
    parent = state.get("accepted_parent")
    if parent is not None:
        snapshot = run_dir / "snapshots" / parent["snapshot"]
        observed = _legacy().path_digest(
            snapshot, state["template"]["scope"]["editable"]
        )
        if observed != parent.get("snapshot_sha256"):
            raise ValueError("accepted parent snapshot digest does not match")
    if verify_sealed_binaries:
        _verify_state_sealed_binaries(run_dir, state)
    return state


def _verify_state_sealed_binaries(
    run_dir: Path, state: dict[str, Any]
) -> None:
    contracts = state.get("template", {}).get("sealed_binaries", {})
    records = state.get("sealed_binaries", {})
    if not isinstance(contracts, dict) or not isinstance(records, dict):
        raise ValueError("sealed binary run state is invalid")
    if state.get("status") == "sealed_binary_invalid":
        if not set(records) <= set(contracts):
            raise ValueError("sealed binary run state does not match the template")
        return
    sealing = state.get("status") in {
        "sealing_binaries",
        "sealing_binaries_retryable",
    }
    if (sealing and not set(records) <= set(contracts)) or (
        not sealing and set(records) != set(contracts)
    ):
        raise ValueError("sealed binary run state does not match the template")
    if not contracts:
        return
    verify_sealed_binary_store(
        run_dir, require_nonwritable=not sealing
    )
    for binary_id, record in records.items():
        verify_sealed_binary_record(run_dir, binary_id, record)


def _derived_usage(run_dir: Path) -> dict[str, float | int]:
    usage = empty_usage()
    seen_evaluations: set[str] = set()
    ledgers = (
        (
            "binary",
            run_dir / "binary-events.jsonl",
            {"sealed_binary_built", "sealed_binary_build_recovered"},
        ),
        (
            "tier",
            run_dir / "tier-events.jsonl",
            {"tier_evaluated", "tier_recovered"},
        ),
    )
    for ledger_name, path, allowed_events in ledgers:
        for event in _events(path):
            evaluation_id = event.get("evaluation_id")
            attempt = event.get("attempt")
            if (
                event.get("event") not in allowed_events
                or not isinstance(evaluation_id, str)
                or not isinstance(attempt, dict)
            ):
                raise ValueError(
                    f"{ledger_name} ledger contains an invalid attempt record"
                )
            if evaluation_id in seen_evaluations:
                raise ValueError("attempt ledgers contain a duplicate evaluation")
            seen_evaluations.add(evaluation_id)
            charge_attempt(usage, attempt)
    admitted = {
        event.get("candidate_id")
        for event in _events(run_dir / "candidate-events.jsonl")
        if event.get("event") == "candidate_admitted"
    }
    if None in admitted:
        raise ValueError("candidate ledger contains an invalid admission")
    usage["candidates_admitted"] = len(admitted)
    return usage


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("template path must stay within the repository") from error


def _tier_by_role(template: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [
        tier
        for tier in template["evaluation"]["tiers"]
        if tier["role"] == role and tier.get("applicable") is True
    ]
    if len(matches) != 1:
        raise ValueError(f"exactly one executable {role} tier is required")
    return matches[0]


def _tier_by_id(template: dict[str, Any], tier_id: str) -> dict[str, Any]:
    matches = [
        tier
        for tier in template["evaluation"]["tiers"]
        if tier["id"] == tier_id and tier.get("applicable") is True
    ]
    if len(matches) != 1:
        raise ValueError(f"exactly one executable {tier_id} tier is required")
    return matches[0]


def _search_tiers(template: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        tier
        for tier in template["evaluation"]["tiers"]
        if tier.get("applicable") is True
        and tier["role"] not in {"holdout", "transfer"}
    ]


def _params(template: dict[str, Any], overrides: list[str]) -> dict[str, str]:
    result = {
        str(name): str(value) for name, value in template["baseline_params"].items()
    }
    for override in overrides:
        if "=" not in override:
            raise ValueError("parameter overrides must use NAME=VALUE")
        name, value = override.split("=", 1)
        result[name] = value
    return _validate_params(template, result)


def _validate_params(
    template: dict[str, Any], params: dict[str, str]
) -> dict[str, str]:
    unknown = sorted(set(params) - set(template["search_space"]))
    if unknown:
        raise ValueError(f"parameters are outside the search space: {unknown}")
    if set(params) != set(template["search_space"]):
        raise ValueError("parameters do not close the search space")
    for name, value in params.items():
        allowed = {str(item) for item in template["search_space"][name]}
        if value not in allowed:
            raise ValueError(f"{name}={value} is outside the search space")
    return params


def _calendar_seconds(state: dict[str, Any]) -> float:
    created = datetime.fromisoformat(state["created_at"].replace("Z", "+00:00"))
    return max(0.0, (datetime.now(timezone.utc) - created).total_seconds())


def _refresh_calendar(state: dict[str, Any]) -> None:
    state["usage"]["calendar_seconds"] = _calendar_seconds(state)


def _phase_calendar_seconds(state: dict[str, Any]) -> float:
    value = state.get("search_started_at", state["created_at"])
    if value is None:
        return 0.0
    started = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _require_calendar_budget(state: dict[str, Any]) -> None:
    _refresh_calendar(state)
    maximum = float(state["template"]["budget"]["total"]["max_calendar_seconds"])
    if float(state["usage"]["calendar_seconds"]) > maximum:
        raise BudgetExhausted("calendar budget is exhausted")


def _require_search_phase_budget(state: dict[str, Any]) -> None:
    _require_calendar_budget(state)
    timebox = state["template"]["mechanism_phase"]["timebox"]
    if int(state["usage"]["candidates_admitted"]) >= int(
        timebox["max_candidates_admitted"]
    ):
        raise BudgetExhausted("phase candidate budget is exhausted")
    if _phase_calendar_seconds(state) > float(timebox["max_search_calendar_seconds"]):
        raise BudgetExhausted("phase calendar budget is exhausted")


def _search_phase_queue_budget(
    state: dict[str, Any], evaluator_timeout: float
) -> float:
    total_queue_budget = _queue_budget_for_timeout(state, evaluator_timeout)
    phase_maximum = float(
        state["template"]["mechanism_phase"]["timebox"][
            "max_search_calendar_seconds"
        ]
    )
    phase_remaining = phase_maximum - _phase_calendar_seconds(state)
    if evaluator_timeout > phase_remaining:
        raise BudgetExhausted("phase calendar budget cannot contain this evaluator")
    return min(total_queue_budget, phase_remaining - evaluator_timeout)


def _queue_budget(state: dict[str, Any], tier: dict[str, Any]) -> float:
    return _queue_budget_for_timeout(
        state, float(tier["evaluator"]["timeout_seconds"])
    )


def _queue_budget_for_timeout(
    state: dict[str, Any],
    evaluator_timeout: float,
    active_reserve: Optional[str] = None,
) -> float:
    _require_calendar_budget(state)
    maximum = float(state["template"]["budget"]["total"]["max_calendar_seconds"])
    remaining = (
        maximum
        - float(state["usage"]["calendar_seconds"])
        - _protected_calendar_seconds(state, active_reserve)
    )
    if evaluator_timeout > remaining:
        raise BudgetExhausted(
            "calendar budget is reserved for production validation"
        )
    return remaining - evaluator_timeout


_VALIDATION_RESERVE_ROLES = {
    "representative_revalidation": "representative",
    "piop_holdout": "holdout",
    "piop_transfer": "transfer",
}


def _protected_calendar_seconds(
    state: dict[str, Any], active_reserve: Optional[str]
) -> float:
    template = state["template"]
    consumed = state["usage"].get("reserve_invocations", {})
    if not isinstance(consumed, dict):
        raise ValueError("reserve invocation accounting is invalid")
    protected = 0.0
    for reserve in template["budget"]["reserves"]:
        reserve_id = reserve["id"]
        role = _VALIDATION_RESERVE_ROLES.get(reserve_id)
        if role is None:
            continue
        used = consumed.get(reserve_id, 0)
        if type(used) is not int or used < 0:
            raise ValueError("reserve invocation accounting is invalid")
        remaining = max(0, int(reserve["invocations"]) - used)
        if reserve_id == active_reserve:
            remaining = max(0, remaining - 1)
        timeout = float(
            _tier_by_role(template, role)["evaluator"]["timeout_seconds"]
        )
        protected += remaining * timeout
    return protected


def _scope_fingerprint(root: Path, template: dict[str, Any]) -> dict[str, str]:
    legacy = _legacy()
    return {
        "editable_paths_sha256": legacy.path_digest(
            root, template["scope"]["editable"]
        ),
        "frozen_paths_sha256": legacy.path_digest(root, template["scope"]["frozen"]),
        "outside_editable_worktree_sha256": legacy.outside_editable_worktree_digest(
            root, template["scope"]["editable"]
        ),
    }


def _assert_frozen(root: Path, state: dict[str, Any]) -> None:
    legacy = _legacy()
    template = state["template"]
    if (
        legacy.path_digest(root, template["scope"]["frozen"])
        != state["fingerprint"]["frozen_paths_sha256"]
    ):
        raise ValueError("a frozen path changed; start a new run")
    if (
        legacy.outside_editable_worktree_digest(root, template["scope"]["editable"])
        != state["fingerprint"]["outside_editable_worktree_sha256"]
    ):
        raise ValueError("a path outside the editable scope changed; start a new run")


def _validate_live_state(root: Path, state: dict[str, Any]) -> None:
    template = state["template"]
    if sha256(canonical_json(template)) != state["template_sha256"]:
        raise ValueError("sealed template digest does not match the run state")
    # The live editable source is the candidate; its baseline bytes are sealed in the run.
    validate_template(
        template, root, verify_editable_profile_sources=False
    )
    live_template = read_json(root / state["template_path"])
    if canonical_json(live_template) != canonical_json(template):
        raise ValueError("live template no longer matches the sealed run template")
    goal = read_json(root / template["portfolio_contract"])
    if canonical_json(goal) != canonical_json(state["goal"]):
        raise ValueError("live goal no longer matches the sealed run goal")
    registry = _registry().read_registry(root / template["registry_contract"])
    _registry().validate_registry(root, registry)
    binding = _registry().resolve_template_binding(
        root, registry, root / state["template_path"]
    )
    if binding != state["registry_binding"]:
        raise ValueError("live registry binding no longer matches the run")


def _required_guards(
    state: dict[str, Any], tier: dict[str, Any], result: dict[str, Any]
) -> tuple[bool, str]:
    required = tier["promotion"].get("required_guards")
    if required is None and tier["role"] == "holdout":
        required = state["goal"]["portfolio_acceptance"]["required_guards"]
    if required is None:
        required = ["all_exact"]
    guards = result.get("guards", {})
    failed = [name for name in required if guards.get(name) is not True]
    return (not failed, "all guards passed" if not failed else f"failed guards: {failed}")


def _tier_record(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["replication"]["summary"]
    treatment_ns = statistics.median(
        float(pair["arms"]["treatment"]["primary_ns"])
        for pair in result["replication"]["pairs"]
    )
    return {
        "metric": float(result["primary"]["value"]),
        "relative_mad": float(summary["mad"]) / float(summary["median"]),
        "paired_summary": summary,
        "treatment_median_ns": treatment_ns,
    }


def _phase_checkpoint(
    state: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    phase = state["template"]["mechanism_phase"]
    return phase_checkpoint_record(
        phase,
        result,
        int(state["usage"]["candidates_admitted"]),
    )


def _phase_checkpoint_state(state: dict[str, Any]) -> dict[str, Any]:
    phase_id = state["template"]["mechanism_phase"]["id"]
    checkpoint = state.get("phase_checkpoint")
    if checkpoint is None:
        checkpoint = {
            "phase_id": phase_id,
            "status": "pending",
            "candidate_id": None,
            "evaluation_id": None,
            "tier_result_sha256": None,
            "record": None,
        }
        state["phase_checkpoint"] = checkpoint
    expected = {
        "phase_id",
        "status",
        "candidate_id",
        "evaluation_id",
        "tier_result_sha256",
        "record",
    }
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != expected
        or checkpoint["phase_id"] != phase_id
        or checkpoint["status"] not in {"pending", "passed", "failed"}
    ):
        raise ValueError("mechanism-phase checkpoint state is invalid")
    terminal_values = (
        checkpoint["candidate_id"],
        checkpoint["evaluation_id"],
        checkpoint["tier_result_sha256"],
        checkpoint["record"],
    )
    if checkpoint["status"] == "pending":
        if any(value is not None for value in terminal_values):
            raise ValueError("pending mechanism-phase checkpoint is not empty")
    elif (
        any(value is None for value in terminal_values)
        or not isinstance(checkpoint["record"], dict)
        or checkpoint["record"].get("passed")
        is not (checkpoint["status"] == "passed")
    ):
        raise ValueError("terminal mechanism-phase checkpoint is incomplete")
    return checkpoint


def _seal_phase_checkpoint(
    state: dict[str, Any],
    record: dict[str, Any],
    candidate_id: str,
    evaluation_id: str,
    tier_result_sha256: Any,
) -> None:
    checkpoint = _phase_checkpoint_state(state)
    if checkpoint["status"] != "pending":
        raise ValueError("mechanism-phase checkpoint is already terminal")
    digest = tier_result_sha256
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or record.get("due") is not True
        or type(record.get("passed")) is not bool
    ):
        raise ValueError("mechanism-phase checkpoint cannot be sealed")
    checkpoint.update(
        {
            "status": "passed" if record["passed"] else "failed",
            "candidate_id": candidate_id,
            "evaluation_id": evaluation_id,
            "tier_result_sha256": digest,
            "record": record,
        }
    )


def _verify_sealed_phase_checkpoint(
    run_dir: Path, state: dict[str, Any]
) -> dict[str, Any]:
    checkpoint = _phase_checkpoint_state(state)
    if checkpoint["status"] == "pending":
        raise ValueError("mechanism-phase checkpoint is not terminal")
    events = [
        event
        for event in _events(run_dir / "tier-events.jsonl")
        if event.get("evaluation_id") == checkpoint["evaluation_id"]
    ]
    path = (
        run_dir
        / "evaluations"
        / checkpoint["evaluation_id"]
        / "tier-result.json"
    )
    if (
        len(events) != 1
        or events[0].get("tier_id")
        != _tier_by_role(state["template"], "proxy")["id"]
        or events[0].get("attempt", {}).get("tier_result_sha256")
        != checkpoint["tier_result_sha256"]
        or not path.is_file()
    ):
        raise ValueError("mechanism-phase checkpoint result is not sealed")
    payload = path.read_bytes()
    if sha256(payload) != checkpoint["tier_result_sha256"]:
        raise ValueError("mechanism-phase checkpoint result digest does not match")
    result = json.loads(payload)
    replay_state = dict(state)
    replay_state["usage"] = dict(state["usage"])
    replay_state["usage"]["candidates_admitted"] = max(
        int(state["usage"]["candidates_admitted"]),
        int(checkpoint["record"]["after_candidates"]),
    )
    if canonical_json(_phase_checkpoint(replay_state, result)) != canonical_json(
        checkpoint["record"]
    ):
        raise ValueError("mechanism-phase checkpoint cannot be reproduced")
    return checkpoint


def _phase_success(
    state: dict[str, Any], result: dict[str, Any], parent: dict[str, Any]
) -> tuple[bool, str]:
    success = state["template"]["mechanism_phase"]["success"]
    latency_ms = _treatment_median_ms(result)
    if latency_ms > float(success["maximum_member_ms"]):
        return False, "candidate exceeds the mechanism-phase latency ceiling"
    required = float(parent["metric"]) * (
        1.0 + float(success["minimum_relative_improvement"])
    )
    if float(result["primary"]["value"]) < required:
        return False, "candidate misses the mechanism-phase improvement gate"
    return True, "candidate clears the mechanism-phase success gate"


def _proxy_calibration_decision(
    policy: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(records) < 3:
        raise ValueError("proxy calibration requires at least three sentinels")
    ids: set[str] = set()
    values = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "id",
            "proxy_score",
            "representative_score",
        }:
            raise ValueError("proxy calibration record is invalid")
        if not isinstance(record["id"], str) or record["id"] in ids:
            raise ValueError("proxy calibration sentinel id is invalid")
        ids.add(record["id"])
        proxy_score = record["proxy_score"]
        representative_score = record["representative_score"]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in (proxy_score, representative_score)
        ):
            raise ValueError("proxy calibration score is invalid")
        values.append((float(proxy_score), float(representative_score)))

    concordant = 0
    discordant = 0
    proxy_ties = 0
    representative_ties = 0
    material_inversions = 0
    material_threshold = float(policy["material_effect_threshold"])
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            proxy_delta = values[left][0] - values[right][0]
            representative_delta = values[left][1] - values[right][1]
            if proxy_delta == 0:
                proxy_ties += 1
            if representative_delta == 0:
                representative_ties += 1
            if proxy_delta * representative_delta > 0:
                concordant += 1
            elif proxy_delta * representative_delta < 0:
                discordant += 1
                low = min(values[left][1], values[right][1])
                high = max(values[left][1], values[right][1])
                if high / low - 1.0 >= material_threshold:
                    material_inversions += 1
    denominator = math.sqrt(
        (concordant + discordant + proxy_ties)
        * (concordant + discordant + representative_ties)
    )
    tau_b = (
        (concordant - discordant) / denominator if denominator > 0 else 0.0
    )
    enabled = (
        tau_b >= float(policy["minimum_rank_agreement"])
        and material_inversions <= int(policy["maximum_material_inversions"])
    )
    return {
        "status": "enabled" if enabled else "disabled",
        "kendall_tau_b": tau_b,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "proxy_ties": proxy_ties,
        "representative_ties": representative_ties,
        "material_inversions": material_inversions,
        "sentinels": records,
    }


def _result_relative_mad(result: dict[str, Any]) -> float:
    summary = result["replication"]["summary"]
    return float(summary["mad"]) / float(summary["median"])


def _relative_stability(
    tier: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    median = float(summary["median"])
    relative_mad = float(summary["mad"]) / median
    order_stratum_log_skew = max(
        abs(math.log(float(summary[field]) / median))
        for field in ("control_first_median", "treatment_first_median")
    )
    maximum_relative_mad = float(tier["promotion"]["maximum_relative_mad"])
    maximum_order_skew = float(
        tier["promotion"]["maximum_order_stratum_log_skew"]
    )
    return {
        "kind": "relative_stability_v1",
        "relative_mad": relative_mad,
        "order_stratum_log_skew": order_stratum_log_skew,
        "limits": {
            "maximum_relative_mad": maximum_relative_mad,
            "maximum_order_stratum_log_skew": maximum_order_skew,
        },
        "admitted": relative_mad <= maximum_relative_mad
        and order_stratum_log_skew <= maximum_order_skew,
    }


def _successor_wall_summary(
    tier: dict[str, Any], result: dict[str, Any]
) -> dict[str, float]:
    samples = result.get("payload", {}).get("samples", [])
    gpu_pairs = result["replication"]["pairs"]
    if not isinstance(samples, list) or len(samples) != len(gpu_pairs):
        raise ValueError("successor wall evidence is incomplete")
    wall_pairs = []
    for gpu_pair, sample in zip(gpu_pairs, samples):
        parent_wall = sample.get("parent", {}).get("wall_ns")
        candidate_wall = sample.get("candidate", {}).get("wall_ns")
        if (
            isinstance(parent_wall, bool)
            or not isinstance(parent_wall, (int, float))
            or parent_wall <= 0
            or isinstance(candidate_wall, bool)
            or not isinstance(candidate_wall, (int, float))
            or candidate_wall <= 0
        ):
            raise ValueError("successor wall evidence is invalid")
        wall_pairs.append(
            {
                "index": gpu_pair["index"],
                "input_id": gpu_pair["input_id"],
                "order": gpu_pair["order"],
                "arms": {
                    "control": {"primary_ns": parent_wall},
                    "treatment": {"primary_ns": candidate_wall},
                },
                "effect": float(parent_wall) / float(candidate_wall),
                "guards": gpu_pair["guards"],
            }
        )
    return paired_summary(wall_pairs, tier["replication"])


def _axis_calibration(summary: dict[str, Any]) -> dict[str, float]:
    median = float(summary["median"])
    control_first = float(summary["control_first_median"])
    treatment_first = float(summary["treatment_first_median"])
    return {
        "median": median,
        "control_first_median": control_first,
        "treatment_first_median": treatment_first,
        "relative_mad": float(summary["mad"]) / median,
        "maximum_absolute_log_bias": max(
            abs(math.log(median)),
            abs(math.log(control_first)),
            abs(math.log(treatment_first)),
        ),
    }


def _exact_string_identity(left: Any, right: Any) -> bool:
    return isinstance(left, str) and bool(left) and left == right


def _successor_calibration(
    tier: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    promotion = tier["promotion"]
    payload = result.get("payload", {})
    fingerprint = result.get("fingerprint", {})
    telemetry = payload.get("telemetry", {})
    gpu = _axis_calibration(result["replication"]["summary"])
    wall = _axis_calibration(_successor_wall_summary(tier, result))
    maximum_noise = float(promotion["maximum_screen_relative_mad"])
    maximum_bias = float(
        promotion["maximum_calibration_absolute_log_bias"]
    )
    identities = {
        "v2_contract": result.get("result_contract")
        == "outer_remainder_successor_v2",
        "artifact": _exact_string_identity(
            fingerprint.get("parent_artifact_sha256"),
            fingerprint.get("candidate_artifact_sha256"),
        ),
        "source": _exact_string_identity(
            telemetry.get("parent_source_sha256"),
            telemetry.get("candidate_source_sha256"),
        ),
        "binding_plan": _exact_string_identity(
            telemetry.get("parent_binding_plan"),
            telemetry.get("candidate_binding_plan"),
        ),
    }
    admitted = (
        all(identities.values())
        and all(
            axis["relative_mad"] <= maximum_noise
            and axis["maximum_absolute_log_bias"] <= maximum_bias
            for axis in (gpu, wall)
        )
    )
    return {
        "kind": "successor_a_a_v1",
        "identities": identities,
        "gpu": gpu,
        "wall": wall,
        "limits": {
            "maximum_relative_mad": maximum_noise,
            "maximum_absolute_log_bias": maximum_bias,
        },
        "admitted": admitted,
    }


def _baseline_admission(
    tier: dict[str, Any], result: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    promotion = tier["promotion"]
    if promotion["kind"] == "successor_screen":
        diagnostics = _successor_calibration(tier, result)
        if not all(diagnostics["identities"].values()):
            return False, "successor calibration is not exact A/A", diagnostics
        if any(
            axis["relative_mad"]
            > diagnostics["limits"]["maximum_relative_mad"]
            for axis in (diagnostics["gpu"], diagnostics["wall"])
        ):
            return False, "successor A/A calibration is unstable", diagnostics
        if any(
            axis["maximum_absolute_log_bias"]
            > diagnostics["limits"]["maximum_absolute_log_bias"]
            for axis in (diagnostics["gpu"], diagnostics["wall"])
        ):
            return False, "successor A/A calibration is biased", diagnostics
        return (
            True,
            "successor A/A calibration admitted the machine",
            diagnostics,
        )
    if promotion["kind"] == "all_guards":
        return (
            True,
            "correctness baseline guards passed",
            {"kind": "guard_baseline_v1", "admitted": True},
        )
    diagnostics = _relative_stability(
        tier, result["replication"]["summary"]
    )
    if not diagnostics["admitted"]:
        return False, f"{tier['id']} baseline is unstable", diagnostics
    return True, "baseline stability gate passed", diagnostics


def _promotion_pass(
    tier: dict[str, Any],
    result: dict[str, Any],
    parent: dict[str, Any],
) -> tuple[bool, str]:
    kind = tier["promotion"]["kind"]
    if kind == "all_guards":
        return True, "all correctness guards passed"
    if kind == "successor_screen":
        metric = float(result["primary"]["value"])
        summary = result["replication"]["summary"]
        wall_summary = _successor_wall_summary(tier, result)
        current_noise = max(
            float(summary["mad"]) / metric,
            float(wall_summary["mad"]) / float(wall_summary["median"]),
        )
        calibration = parent.get("calibration")
        if (
            not isinstance(calibration, dict)
            or calibration.get("admitted") is not True
        ):
            raise ValueError("successor parent has no admitted A/A calibration")
        calibration_noise = max(
            float(calibration["gpu"]["relative_mad"]),
            float(calibration["wall"]["relative_mad"]),
        )
        uncertainty = max(
            float(tier["promotion"]["minimum_uncertainty"]),
            calibration_noise,
            current_noise,
        )
        pairs = result["replication"]["pairs"]
        every_active_pair_loses = all(float(pair["effect"]) < 1.0 for pair in pairs)
        wall_fails_both_orders = all(
            float(wall_summary[field]) <= 1.0
            for field in ("control_first_median", "treatment_first_median")
        )
        optimistic = metric * math.exp(uncertainty)
        clear_loss = (
            optimistic <= float(tier["promotion"]["clear_loss_ratio"])
            and every_active_pair_loses
            and wall_fails_both_orders
        )
        if clear_loss:
            return False, "successor is a calibrated clear loss"
        return True, "successor screen is non-losing or inconclusive"
    if kind != "relative_improvement":
        raise ValueError(f"unsupported search-tier promotion kind: {kind}")
    metric = float(result["primary"]["value"])
    current_stability = _relative_stability(
        tier, result["replication"]["summary"]
    )
    if not current_stability["admitted"]:
        return False, "result exceeds the representative stability gate"
    parent_stability = _relative_stability(tier, parent["paired_summary"])
    if not parent_stability["admitted"]:
        raise ValueError("accepted parent no longer passes machine admission")
    current_noise = max(
        float(current_stability["relative_mad"]),
        float(current_stability["order_stratum_log_skew"]),
    )
    parent_noise = max(
        float(parent_stability["relative_mad"]),
        float(parent_stability["order_stratum_log_skew"]),
    )
    threshold = max(
        float(tier["promotion"]["minimum_relative_improvement"]),
        float(tier["promotion"]["noise_multiplier"])
        * max(parent_noise, current_noise),
    )
    if metric >= float(parent["metric"]) * (1.0 + threshold):
        return True, "improves beyond the noise-qualified threshold"
    return False, "does not improve beyond the noise-qualified threshold"


def _successor_screen_disposition(
    tier: dict[str, Any],
    result: dict[str, Any],
    parent: dict[str, Any],
    *,
    retry_available: bool,
) -> tuple[str, str]:
    promoted, reason = _promotion_pass(tier, result, parent)
    if not promoted:
        return "discard", reason
    gpu_summary = result["replication"]["summary"]
    wall_summary = _successor_wall_summary(tier, result)
    gpu_noise = _result_relative_mad(result)
    wall_noise = float(wall_summary["mad"]) / float(wall_summary["median"])
    noise_limit = float(tier["promotion"]["maximum_screen_relative_mad"])
    if max(gpu_noise, wall_noise) <= noise_limit:
        return "advance", reason

    calibration = parent["calibration"]
    stable_uncertainty = max(
        float(tier["promotion"]["minimum_uncertainty"]),
        float(calibration["gpu"]["relative_mad"]),
        float(calibration["wall"]["relative_mad"]),
    )
    potential_clear_loss = (
        float(gpu_summary["median"]) * math.exp(stable_uncertainty)
        <= float(tier["promotion"]["clear_loss_ratio"])
    )
    stable_contradiction = any(
        noise <= noise_limit
        and any(
            float(summary[field]) >= 1.0
            for field in (
                "median",
                "control_first_median",
                "treatment_first_median",
            )
        )
        for summary, noise in (
            (gpu_summary, gpu_noise),
            (wall_summary, wall_noise),
        )
    )
    if potential_clear_loss and not stable_contradiction and retry_available:
        return "retry", "successor screen is unstable and inconclusive"
    if potential_clear_loss and not stable_contradiction:
        return "advance", "successor remained inconclusive after its bounded retry"
    return "advance", "successor has no stable clear-loss signal"


def _validate_successor_compilation_telemetry(
    output: dict[str, Any], pair_count: int
) -> None:
    telemetry = output["telemetry"]["compilation"]
    if (
        not isinstance(telemetry, dict)
        or set(telemetry) != {"context_order", "parent", "candidate"}
        or telemetry["context_order"] != ["parent", "candidate"]
    ):
        raise ValueError("OuterRemainder compilation telemetry is invalid")
    warmup = output.get("excluded_warmup")
    samples = output.get("samples")
    if (
        not isinstance(warmup, dict)
        or not isinstance(samples, list)
        or len(samples) != pair_count
    ):
        raise ValueError("OuterRemainder compilation telemetry is incomplete")
    records = [warmup, *samples]
    fields = {
        "source_assembly_ns",
        "library_compile_ns",
        "source_bytes",
        "assembled_source_sha256",
        "pipeline_set_ns",
        "pipeline_set_total_ns",
    }
    for role in ("parent", "candidate"):
        context = telemetry[role]
        if not isinstance(context, dict) or set(context) != fields:
            raise ValueError("OuterRemainder compilation context is invalid")
        assembly_ns = context["source_assembly_ns"]
        library_ns = context["library_compile_ns"]
        source_bytes = context["source_bytes"]
        digest = context["assembled_source_sha256"]
        pipeline_ns = context["pipeline_set_ns"]
        total_ns = context["pipeline_set_total_ns"]
        if (
            type(assembly_ns) is not int
            or assembly_ns < 0
            or type(library_ns) is not int
            or library_ns <= 0
            or type(source_bytes) is not int
            or source_bytes <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(pipeline_ns, list)
            or len(pipeline_ns) != pair_count + 1
            or any(type(value) is not int or value < 0 for value in pipeline_ns)
            or type(total_ns) is not int
            or total_ns != sum(pipeline_ns)
        ):
            raise ValueError("OuterRemainder compilation context is invalid")
        for record, compile_ns in zip(records, pipeline_ns):
            arm = record.get(role) if isinstance(record, dict) else None
            if (
                not isinstance(arm, dict)
                or type(arm.get("setup_wall_ns")) is not int
                or compile_ns > arm["setup_wall_ns"]
            ):
                raise ValueError(
                    "OuterRemainder pipeline compilation exceeds setup wall"
                )


def _validate_closed_result(
    root: Path,
    tier: dict[str, Any],
    output: dict[str, Any],
    params: dict[str, str],
) -> None:
    adapter = tier["evaluator"]["result_adapter"]
    if adapter == "outer_remainder_successor_v1":
        fingerprint = output.get("fingerprint")
        if not isinstance(fingerprint, dict):
            raise ValueError("OuterRemainder successor result is not closed")
        pairs = tier["replication"]["included_pairs"]
        first_order = tier["replication"]["first_order"]
        raw_first = (
            ["parent", "candidate"]
            if first_order == ["control", "treatment"]
            else ["candidate", "parent"]
        )
        orders = [
            raw_first if pair % 2 == 0 else list(reversed(raw_first))
            for pair in range(pairs)
        ]
        fields = {
            "fixture",
            "log_n",
            "pairs",
            "excluded_warmup_pairs",
            "orders",
            "parent_artifact_sha256",
            "candidate_artifact_sha256",
        }
        digests = (
            fingerprint.get("parent_artifact_sha256"),
            fingerprint.get("candidate_artifact_sha256"),
        )
        if (
            output.get("schema") != "outer_remainder_successor_v1"
            or output.get("schema_version") != 1
            or output.get("kernel") != "OuterRemainder"
            or output.get("all_exact") is not True
            or set(fingerprint) != fields
            or fingerprint.get("fixture") != "resident-outer-remainder-v1"
            or fingerprint.get("log_n") != tier["promotion"].get("log_n")
            or fingerprint.get("pairs") != pairs
            or fingerprint.get("excluded_warmup_pairs")
            != tier["replication"]["excluded_warmup_pairs"]
            or fingerprint.get("orders") != orders
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
                for digest in digests
            )
        ):
            raise ValueError("OuterRemainder successor result is not closed")
        return
    if adapter == "outer_remainder_successor_v2":
        fingerprint = output.get("fingerprint")
        pairs = tier["replication"]["included_pairs"]
        first_order = tier["replication"]["first_order"]
        raw_first = (
            ["parent", "candidate"]
            if first_order == ["control", "treatment"]
            else ["candidate", "parent"]
        )
        orders = [
            raw_first if pair % 2 == 0 else list(reversed(raw_first))
            for pair in range(pairs)
        ]
        fingerprint_fields = {
            "fixture",
            "log_n",
            "pairs",
            "excluded_warmup_pairs",
            "orders",
            "parent_artifact_sha256",
            "candidate_artifact_sha256",
            "runner_binary_sha256",
        }
        guard_fields = {
            "all_exact",
            "correctness_exact",
            "target_scale",
            "runtime_artifacts_exact",
            "resident_row_handle_lifecycle_exact",
            "metal_phase_schedule_exact",
            "gpu_timestamps_exact",
        }
        telemetry_fields = {
            "device_name",
            "device_registry_shared",
            "cycles",
            "parent_binding_plan",
            "candidate_binding_plan",
            "parent_source_sha256",
            "candidate_source_sha256",
            "production_last_owner_release_deferred",
            "compilation",
        }
        digests = (
            fingerprint.get("parent_artifact_sha256")
            if isinstance(fingerprint, dict)
            else None,
            fingerprint.get("candidate_artifact_sha256")
            if isinstance(fingerprint, dict)
            else None,
            fingerprint.get("runner_binary_sha256")
            if isinstance(fingerprint, dict)
            else None,
        )
        guards = output.get("guards")
        telemetry = output.get("telemetry")
        if (
            set(output)
            != {
                "schema",
                "schema_version",
                "kernel",
                "fingerprint",
                "metrics",
                "excluded_warmup",
                "samples",
                "guards",
                "all_exact",
                "resources",
                "telemetry",
            }
            or output.get("schema") != "outer_remainder_successor_v2"
            or output.get("schema_version") != 2
            or output.get("kernel") != "OuterRemainder"
            or output.get("all_exact") is not True
            or not isinstance(fingerprint, dict)
            or set(fingerprint) != fingerprint_fields
            or fingerprint.get("fixture") != "resident-outer-remainder-v2"
            or fingerprint.get("log_n") != tier["promotion"].get("log_n")
            or fingerprint.get("pairs") != pairs
            or fingerprint.get("excluded_warmup_pairs")
            != tier["replication"]["excluded_warmup_pairs"]
            or fingerprint.get("orders") != orders
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
                for digest in digests
            )
            or not isinstance(guards, dict)
            or set(guards) != guard_fields
            or any(value is not True for value in guards.values())
            or not isinstance(telemetry, dict)
            or set(telemetry) != telemetry_fields
            or not isinstance(telemetry.get("device_name"), str)
            or not telemetry["device_name"]
            or telemetry.get("device_registry_shared") is not True
            or telemetry.get("cycles") != 1 << fingerprint["log_n"]
            or telemetry.get("production_last_owner_release_deferred") is not True
        ):
            raise ValueError("OuterRemainder successor v2 result is not closed")
        _validate_successor_compilation_telemetry(output, pairs)
        return
    if adapter == "metal_piop_v7":
        if tier["promotion"].get("local_kernel") != "OuterRemainder":
            raise ValueError("schema-2 PIOP closure is not implemented for this kernel")
        legacy = _legacy()
        legacy_template = read_json(
            root / "crates/jolt-kernels/autoresearch/outer_remainder.template.json"
        )
        gate = legacy_template["final_validation"]["production_gate"]
        gate["minimum_local_speedup"] = tier["promotion"][
            "minimum_local_speedup"
        ]
        gate["minimum_log_n"] = tier["promotion"]["log_n"]
        gate["minimum_pairs"] = tier["replication"]["included_pairs"]
        gate["required_guards"] = tier["promotion"]["required_guards"]
        legacy.validate_production_result(
            legacy_template,
            output,
            legacy.git_head(root),
            params,
            legacy.git_worktree_clean(root),
        )
        return
    if adapter == "outer_remainder_screen_v1":
        fingerprint = output.get("fingerprint", {})
        if not isinstance(fingerprint, dict):
            raise ValueError("OuterRemainder screen result is not closed")
        fingerprint_fields = {
            "fixture",
            "log_n",
            "trace_elements",
            "trace_rows",
            "pairs",
            "excluded_warmup_pairs",
            "orders",
            "rayon_threads",
            "materialize_threads",
            "transition_threads",
            "output_threads",
            "cutoff_log2",
            "trace_cutoff_log2",
            "storage_initialization",
            "member_span",
            "rounds",
            "output_claims",
            "source_sha256",
            "binary_sha256",
        }
        parameter_fields = {
            "JOLT_METAL_OUTER_REMAINDER_CUTOFF_LOG2": "cutoff_log2",
            "JOLT_METAL_OUTER_REMAINDER_MATERIALIZE_THREADS": (
                "materialize_threads"
            ),
            "JOLT_METAL_OUTER_REMAINDER_OUTPUT_THREADS": "output_threads",
            "JOLT_METAL_OUTER_REMAINDER_TRACE_CUTOFF_LOG2": (
                "trace_cutoff_log2"
            ),
            "JOLT_METAL_OUTER_REMAINDER_TRANSITION_THREADS": (
                "transition_threads"
            ),
        }
        digests = (
            fingerprint.get("source_sha256"),
            fingerprint.get("binary_sha256"),
        )
        log_n = tier["promotion"]["log_n"]
        pairs = tier["replication"]["included_pairs"]
        orders = [
            ["optimized", "metal"] if pair % 2 == 0 else ["metal", "optimized"]
            for pair in range(pairs)
        ]
        if (
            output.get("schema") != "outer_remainder_screen_v1"
            or output.get("schema_version") != 1
            or output.get("kernel") != "OuterRemainder"
            or output.get("all_exact") is not True
            or set(fingerprint) != fingerprint_fields
            or fingerprint.get("fixture") != "real-fibonacci-akita-proof"
            or fingerprint.get("log_n") != log_n
            or fingerprint.get("trace_elements") != 1 << log_n
            or type(fingerprint.get("trace_rows")) is not int
            or not 0 < fingerprint["trace_rows"] <= 1 << log_n
            or fingerprint.get("pairs") != pairs
            or fingerprint.get("excluded_warmup_pairs")
            != tier["replication"]["excluded_warmup_pairs"]
            or fingerprint.get("orders") != orders
            or fingerprint.get("rayon_threads") != 16
            or fingerprint.get("storage_initialization") != "full"
            or fingerprint.get("member_span")
            != "OuterRemainder::complete_member"
            or fingerprint.get("rounds") != log_n + 1
            or fingerprint.get("output_claims") != 35
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in digests
            )
            or any(
                str(fingerprint.get(field)) != params[name]
                for name, field in parameter_fields.items()
            )
        ):
            raise ValueError("OuterRemainder screen result is not closed")
        return
    if adapter != "outer_remainder_v3":
        return
    legacy_template = read_json(
        root / "crates/jolt-kernels/autoresearch/outer_remainder.template.json"
    )
    _legacy().validate_local_result_contract(legacy_template, output, params)


def cost_limit_overages(
    tier: dict[str, Any], attempt: dict[str, Any]
) -> list[str]:
    observed = {
        "active_evaluator_seconds": attempt["controller"][
            "subprocess_wall_seconds"
        ],
        "exclusive_machine_seconds": attempt["controller"][
            "exclusive_lease_seconds"
        ],
        "gpu_active_seconds": attempt["resources"]["gpu_active_charge_seconds"],
    }
    return [
        name
        for name, value in observed.items()
        if float(value) > float(tier["cost_limit"][name]) + 1e-9
    ]


def _arm_process_tracking(
    run_dir: Path, evaluation_id: str
) -> dict[str, str]:
    inflight_path = run_dir / "inflight.json"
    inflight = read_json(inflight_path)
    if inflight.get("evaluation_id") != evaluation_id:
        raise ValueError("inflight evaluation does not match the evaluator launch")
    relative_identity = (
        Path("evaluations") / evaluation_id / "process-identity.json"
    )
    tracking = {
        "evaluation_id": evaluation_id,
        "launch_token": secrets.token_hex(32),
        "identity_path": relative_identity.as_posix(),
    }
    inflight["process_tracking"] = tracking
    write_inflight(run_dir, inflight)
    return {
        **tracking,
        "identity_path": str((run_dir / relative_identity).resolve()),
    }


def _runtime_artifact_contract(
    template: dict[str, Any], tier_id: str
) -> Optional[dict[str, Any]]:
    contract = template.get("runtime_artifact")
    if not isinstance(contract, dict) or contract.get("tier_id") != tier_id:
        return None
    return contract


def _runtime_artifact_context(
    root: Path,
    run_dir: Path,
    state: dict[str, Any],
    tier: dict[str, Any],
    params: dict[str, str],
) -> tuple[dict[str, str], Optional[dict[str, Any]]]:
    contract = _runtime_artifact_contract(state["template"], tier["id"])
    if contract is None:
        return {}, None
    accepted = state.get("accepted_parent")
    if accepted is None:
        parent_snapshot = "baseline"
        parent_params = state["template"]["baseline_params"]
        candidate_root = run_dir / "snapshots" / "baseline"
    else:
        parent_snapshot = accepted["snapshot"]
        parent_params = accepted["params"]
        candidate_root = root
    return materialize_runtime_artifact_context(
        run_dir,
        contract,
        run_dir / "snapshots" / parent_snapshot,
        parent_params,
        candidate_root,
        params,
    )


def _sealed_binary_ids_for_tier(
    template: dict[str, Any], tier_id: str
) -> list[str]:
    return [
        binary_id
        for binary_id, contract in template.get("sealed_binaries", {}).items()
        if tier_id in contract["consumer_tiers"]
    ]


def _sealed_binary_context(
    root: Path,
    run_dir: Path,
    state: dict[str, Any],
    tier: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], Optional[dict[str, Any]]]:
    evaluator = tier["evaluator"]
    binary_ids = _sealed_binary_ids_for_tier(state["template"], tier["id"])
    if not binary_ids:
        return evaluator, {}, None
    if len(binary_ids) != 1:
        raise ValueError("a tier must consume exactly one sealed binary")
    binary_id = binary_ids[0]
    record = state.get("sealed_binaries", {}).get(binary_id)
    if record is None:
        raise ValueError(f"sealed binary {binary_id} is missing from run state")
    runner_path = verify_sealed_binary_contract(
        root,
        run_dir,
        binary_id,
        state["template"]["sealed_binaries"][binary_id],
        record,
    )
    token = sealed_binary_token(binary_id)
    command = [
        str(runner_path.resolve()) if argument == token else argument
        for argument in evaluator["command"]
    ]
    if command == evaluator["command"] or token in command:
        raise ValueError(f"sealed binary {binary_id} command token is unresolved")
    context = {binary_id: record}
    return (
        {**evaluator, "command": command},
        {
            "JOLT_AUTORESEARCH_RUNNER_SHA256": record["manifest"][
                "binary_sha256"
            ]
        },
        context,
    )


def _publish_evaluation_contexts(
    run_dir: Path,
    evaluation_id: str,
    execution_context: Optional[dict[str, Any]],
    sealed_binary_context: Optional[dict[str, Any]],
) -> None:
    if execution_context is None and sealed_binary_context is None:
        return
    inflight = read_json(run_dir / "inflight.json")
    if inflight.get("evaluation_id") != evaluation_id:
        raise ValueError("inflight evaluation does not match its execution contexts")
    if execution_context is not None:
        inflight["execution_context"] = execution_context
    if sealed_binary_context is not None:
        inflight["sealed_binary_context"] = sealed_binary_context
    write_inflight(run_dir, inflight)


def _verify_sealed_binary_context(
    run_dir: Path,
    state: dict[str, Any],
    tier: dict[str, Any],
    context: Optional[dict[str, Any]],
) -> None:
    expected_ids = set(_sealed_binary_ids_for_tier(state["template"], tier["id"]))
    if context is None:
        if expected_ids:
            raise ValueError("required sealed binary context is missing")
        return
    if not isinstance(context, dict) or set(context) != expected_ids:
        raise ValueError("sealed binary context does not match its consumer tier")
    for binary_id, record in context.items():
        if canonical_json(record) != canonical_json(
            state.get("sealed_binaries", {}).get(binary_id)
        ):
            raise ValueError(f"sealed binary {binary_id} context changed")
        verify_sealed_binary_record(run_dir, binary_id, record)


def _validate_sealed_binary_fingerprint(
    output: dict[str, Any],
    state: dict[str, Any],
    tier: dict[str, Any],
    context: Optional[dict[str, Any]],
) -> None:
    if context is None:
        return
    for binary_id, record in context.items():
        path = state["template"]["sealed_binaries"][binary_id][
            "result_fingerprint"
        ]
        observed: Any = output
        for field in path:
            if not isinstance(observed, dict) or field not in observed:
                raise ValueError("sealed binary result fingerprint is missing")
            observed = observed[field]
        if observed != record["manifest"]["binary_sha256"]:
            raise ValueError(
                "sealed binary result fingerprint does not match the controller"
            )


def _seal_attempt_artifacts(
    run_dir: Path,
    expected_kind: Optional[str],
    context: Optional[dict[str, Any]],
    attempt: dict[str, Any],
) -> bool:
    try:
        verify_runtime_artifact_context(run_dir, expected_kind, context)
    except (OSError, ValueError) as error:
        attempt["evaluator_outcome"] = attempt["outcome"]
        attempt["outcome"] = "artifact_changed"
        attempt["error"] = str(error)
        return False
    return True


def _seal_attempt_binaries(
    run_dir: Path,
    state: dict[str, Any],
    tier: dict[str, Any],
    context: Optional[dict[str, Any]],
    attempt: dict[str, Any],
) -> bool:
    try:
        _verify_sealed_binary_context(run_dir, state, tier, context)
    except (OSError, ValueError) as error:
        attempt["evaluator_outcome"] = attempt["outcome"]
        attempt["outcome"] = "binary_changed"
        attempt["error"] = str(error)
        return False
    return True


def _artifact_rejected_attempt(
    tier: dict[str, Any],
    error: Exception,
    context: Optional[dict[str, Any]],
    sealed_binary_context: Optional[dict[str, Any]],
    command: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tier_id": tier["id"],
        "outcome": "artifact_rejected",
        "error": str(error),
        "command": command or list(tier["evaluator"]["command"]),
        "started_at": utc_now(),
        "controller": {
            "queue_wait_seconds": 0.0,
            "exclusive_lease_seconds": 0.0,
            "subprocess_wall_seconds": 0.0,
        },
        "resources": {
            "gpu_active_seconds": 0.0,
            "gpu_active_charge_seconds": 0.0,
            "gpu_active_charge_kind": "validated",
        },
        "result_sha256": None,
        "process_tracking": None,
        "execution_context": context,
        "sealed_binary_context": sealed_binary_context,
    }


def execute_tier(
    root: Path,
    run_dir: Path,
    state: dict[str, Any],
    tier: dict[str, Any],
    params: dict[str, str],
    evaluation_id: str,
    *,
    expected_editable_digest: Optional[str] = None,
    budget_reserve: Optional[str] = None,
    search_phase: bool = False,
    retain_inflight_on_budget_exhausted: bool = False,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    try:
        queue_budget = (
            _search_phase_queue_budget(
                state, float(tier["evaluator"]["timeout_seconds"])
            )
            if search_phase
            else _queue_budget_for_timeout(
                state,
                float(tier["evaluator"]["timeout_seconds"]),
                budget_reserve,
            )
        )
        admit_tier(
            state["template"]["budget"],
            state["usage"],
            tier["cost_limit"],
            budget_reserve,
        )
    except BudgetExhausted:
        inflight_path = run_dir / "inflight.json"
        if inflight_path.is_file() and not retain_inflight_on_budget_exhausted:
            inflight = read_json(inflight_path)
            if inflight.get("evaluation_id") == evaluation_id:
                inflight_path.unlink()
        raise
    evaluation_dir = run_dir / "evaluations" / evaluation_id
    context_record = None
    sealed_binary_context = None
    runtime_artifact = _runtime_artifact_contract(
        state["template"], tier["id"]
    )
    runtime_artifact_kind = (
        runtime_artifact["kind"] if runtime_artifact is not None else None
    )
    resolved_evaluator = tier["evaluator"]
    contexts_admitted = False
    try:
        context_env, context_record = _runtime_artifact_context(
            root, run_dir, state, tier, params
        )
        (
            resolved_evaluator,
            binary_env,
            sealed_binary_context,
        ) = _sealed_binary_context(root, run_dir, state, tier)
        context_env.update(binary_env)
        _publish_evaluation_contexts(
            run_dir,
            evaluation_id,
            context_record,
            sealed_binary_context,
        )
        contexts_admitted = True
    except (OSError, ValueError) as error:
        evaluation_dir.mkdir(parents=True, exist_ok=False)
        attempt = _artifact_rejected_attempt(
            tier,
            error,
            context_record,
            sealed_binary_context,
            list(resolved_evaluator["command"]),
        )
        output = None
    else:
        process_tracking = _arm_process_tracking(run_dir, evaluation_id)

        def verify_admitted_contexts() -> None:
            verify_runtime_artifact_context(
                run_dir, runtime_artifact_kind, context_record
            )
            _verify_sealed_binary_context(
                run_dir, state, tier, sealed_binary_context
            )

        attempt, output = run_attempt(
            root,
            resolved_evaluator,
            params,
            evaluation_dir,
            tier["id"],
            queue_timeout_seconds=queue_budget,
            process_tracking=process_tracking,
            context_env=context_env,
            context_record=context_record,
            sealed_binary_context=sealed_binary_context,
            prelaunch_check=verify_admitted_contexts,
        )
    if contexts_admitted:
        if not _seal_attempt_artifacts(
            run_dir, runtime_artifact_kind, context_record, attempt
        ):
            output = None
        if not _seal_attempt_binaries(
            run_dir, state, tier, sealed_binary_context, attempt
        ):
            state["status"] = "sealed_binary_invalid"
            write_state(run_dir, state)
            output = None
    result = None
    if attempt["outcome"] == "success" and output is not None:
        try:
            validate_runtime_artifact_output(
                output, runtime_artifact_kind, context_record
            )
            _validate_sealed_binary_fingerprint(
                output, state, tier, sealed_binary_context
            )
            _validate_closed_result(root, tier, output, params)
            result, resource_charge = adapt_result(
                tier, output, state["template"]["kernel"]
            )
            validate_tier_result(result, tier)
            attempt["resources"] = resource_charge
        except (KeyError, TypeError, ValueError) as error:
            attempt["outcome"] = "invalid_result"
            attempt["error"] = str(error)
            result = None
    if (
        expected_editable_digest is not None
        and _legacy().path_digest(root, state["template"]["scope"]["editable"])
        != expected_editable_digest
    ):
        attempt["outcome"] = "invalid_result"
        attempt["error"] = "editable source changed during evaluation"
        result = None
    overages = cost_limit_overages(tier, attempt)
    if overages:
        attempt["outcome"] = "cost_limit_exceeded"
        attempt["error"] = f"tier exceeded declared cost limits: {overages}"
        result = None
    if result is not None:
        result_bytes = canonical_json(result)
        (evaluation_dir / "tier-result.json").write_bytes(result_bytes)
        attempt["tier_result_sha256"] = sha256(result_bytes)
    attempt["budget_reserve"] = budget_reserve
    charge_attempt(state["usage"], attempt)
    _refresh_calendar(state)
    (evaluation_dir / "attempt.json").write_bytes(canonical_json(attempt))
    event = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event": "tier_evaluated",
        "evaluation_id": evaluation_id,
        "tier_id": tier["id"],
        "params": params,
        "attempt": attempt,
        "primary": result["primary"] if result is not None else None,
        "paired_summary": (
            result["replication"]["summary"] if result is not None else None
        ),
        "recorded_at": utc_now(),
    }
    append_event(run_dir / "tier-events.jsonl", event)
    return event, result


def _initialize_run_files(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "evaluations").mkdir()
    (run_dir / "snapshots").mkdir()
    (run_dir / "artifacts").mkdir()
    (run_dir / "binaries").mkdir()
    for name in (
        "baseline-events.jsonl",
        "binary-events.jsonl",
        "candidate-events.jsonl",
        "tier-events.jsonl",
        "decision-events.jsonl",
        "kernel-validations.jsonl",
        "proxy-events.jsonl",
    ):
        (run_dir / name).touch()


def _binary_build_evaluation_id(run_dir: Path, binary_id: str) -> str:
    prefix = f"binary-build-{binary_id}"
    prior = list((run_dir / "evaluations").glob(f"{prefix}*"))
    if not prior:
        return prefix
    return f"{prefix}-retry-{len(prior) + 1:03d}"


def _binary_build_cost_limit(contract: dict[str, Any]) -> dict[str, float]:
    timeout = float(contract["build"]["timeout_seconds"])
    return {
        "active_evaluator_seconds": timeout,
        "exclusive_machine_seconds": timeout,
        "gpu_active_seconds": 0.0,
    }


def _record_binary_build_gpu_usage(attempt: dict[str, Any]) -> None:
    attempt["resources"] = {
        "gpu_active_seconds": 0.0,
        "gpu_active_charge_seconds": 0.0,
        "gpu_active_charge_kind": "sealed_binary_build_no_gpu",
    }


def _validate_profiled_binary(
    root: Path, state: dict[str, Any]
) -> None:
    profile = state["template"].get("iteration_profile")
    if not isinstance(profile, dict):
        return
    proxies = [
        tier
        for tier in state["template"]["evaluation"]["tiers"]
        if tier.get("applicable") is True and tier.get("role") == "proxy"
    ]
    if len(proxies) != 1:
        raise ValueError("iteration profile requires exactly one proxy tier")
    binary_ids = _sealed_binary_ids_for_tier(
        state["template"], proxies[0]["id"]
    )
    if len(binary_ids) != 1:
        raise ValueError("iteration profile requires one sealed proxy evaluator")
    record = state["sealed_binaries"].get(binary_ids[0])
    if not isinstance(record, dict):
        raise ValueError("profiled sealed evaluator is missing")
    expected = iteration_profile_evaluator_fingerprint(profile, root)
    manifest = record.get("manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("binary_sha256")
        != expected["runner_binary_sha256"]
        or (
            "runner_source_sha256" in expected
            and manifest.get("source_sha256")
            != expected["runner_source_sha256"]
        )
    ):
        raise ValueError(
            "sealed proxy evaluator does not match its iteration profile"
        )


def _continue_binary_sealing(
    root: Path, run_dir: Path, state: dict[str, Any]
) -> dict[str, Any]:
    contracts = state["template"].get("sealed_binaries", {})
    records = state["sealed_binaries"]
    for binary_id in sorted(contracts):
        if binary_id in records:
            verify_sealed_binary_contract(
                root, run_dir, binary_id, contracts[binary_id], records[binary_id]
            )
            continue
        contract = contracts[binary_id]
        cost_limit = _binary_build_cost_limit(contract)
        admit_tier(state["template"]["budget"], state["usage"], cost_limit)
        queue_budget = _queue_budget_for_timeout(
            state, float(contract["build"]["timeout_seconds"])
        )
        source_sha256 = declared_source_sha256(root, contract["source_paths"])
        build_environment_sha256 = sha256(
            canonical_json(sanitized_parent_environment())
        )
        evaluation_id = _binary_build_evaluation_id(run_dir, binary_id)
        inflight = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "kind": "sealed_binary_build",
            "evaluation_id": evaluation_id,
            "binary_id": binary_id,
            "source_sha256": source_sha256,
            "build_environment_sha256": build_environment_sha256,
            "started_at": utc_now(),
        }
        write_inflight(run_dir, inflight)
        tracking = _arm_process_tracking(run_dir, evaluation_id)
        build = contract["build"]

        def source_unchanged() -> None:
            if declared_source_sha256(root, contract["source_paths"]) != source_sha256:
                raise ValueError("sealed binary sources changed before build launch")

        attempt, _ = run_attempt(
            root,
            {
                "command": build["command"],
                "env": {},
                "timeout_seconds": build["timeout_seconds"],
            },
            {},
            run_dir / "evaluations" / evaluation_id,
            f"sealed_binary:{binary_id}",
            process_tracking=tracking,
            parse_result=False,
            prelaunch_check=source_unchanged,
            queue_timeout_seconds=queue_budget,
        )
        _record_binary_build_gpu_usage(attempt)
        record = None
        if attempt["outcome"] == "success":
            try:
                prepared = prepare_sealed_binary_from_output(
                    root,
                    binary_id,
                    contract,
                    source_sha256,
                    build_environment_sha256,
                )
                record = materialize_sealed_binary(run_dir, prepared)
                verify_sealed_binary_contract(
                    root, run_dir, binary_id, contract, record
                )
            except (OSError, ValueError) as error:
                attempt["outcome"] = "invalid_binary_output"
                attempt["error"] = str(error)
        attempt["sealed_binary"] = record
        attempt_path = run_dir / "evaluations" / evaluation_id / "attempt.json"
        attempt_path.write_bytes(canonical_json(attempt))
        append_event(
            run_dir / "binary-events.jsonl",
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event": "sealed_binary_built",
                "evaluation_id": evaluation_id,
                "binary_id": binary_id,
                "attempt": attempt,
                "recorded_at": utc_now(),
            },
        )
        charge_attempt(state["usage"], attempt)
        _refresh_calendar(state)
        if record is None:
            state["status"] = "sealing_binaries_retryable"
            write_state(run_dir, state)
            (run_dir / "inflight.json").unlink()
            raise ValueError(
                f"sealed binary {binary_id} build failed: {attempt['error']}"
            )
        records[binary_id] = record
        write_state(run_dir, state)
        (run_dir / "inflight.json").unlink()

    try:
        _validate_profiled_binary(root, state)
    except ValueError:
        state["status"] = "sealed_binary_invalid"
        write_state(run_dir, state)
        raise
    seal_sealed_binary_store(run_dir)
    state["status"] = "initializing"
    write_state(run_dir, state)
    return _continue_initialization(root, run_dir, state)


def validate_template_file(root: Path, template_path: Path) -> dict[str, Any]:
    root = root.resolve()
    template_path = template_path.resolve()
    template = read_json(template_path)
    validate_template(template, root)
    registry_path = root / template["registry_contract"]
    registry = _registry().read_registry(registry_path)
    _registry().validate_registry(root, registry)
    binding = _registry().resolve_template_binding(root, registry, template_path)
    if binding["slot_id"] != template["slot_id"]:
        raise ValueError("template registry slot does not match its registered binding")
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "template": _relative(root, template_path),
        "slot_id": template["slot_id"],
        "kernel": template["kernel"],
        "lifecycle": binding["lifecycle"],
        "fresh_init_eligible": binding["fresh_init_eligible"],
        "binding": binding,
        "valid": True,
    }


def init_run(root: Path, template_path: Path, run_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    template_path = template_path.resolve()
    template = read_json(template_path)
    validate_template(template, root)
    registry_path = root / template["registry_contract"]
    registry = _registry().read_registry(registry_path)
    _registry().validate_registry(root, registry)
    binding = _registry().resolve_template_binding(root, registry, template_path)
    if binding["slot_id"] != template["slot_id"]:
        raise ValueError("template registry slot does not match its registered binding")
    if not binding["fresh_init_eligible"]:
        raise ValueError("template is existing-run-only and cannot initialize a fresh run")

    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = run_dir.parent / (
        f".{run_dir.name}.initializing-{secrets.token_hex(8)}"
    )
    _initialize_run_files(staging)
    legacy = _legacy()
    try:
        baseline_snapshot = staging / "snapshots" / "baseline"
        legacy.snapshot_paths(
            root, template["scope"]["editable"], baseline_snapshot
        )
        snapshot_sha256 = legacy.path_digest(
            baseline_snapshot, template["scope"]["editable"]
        )
        fingerprint = _scope_fingerprint(root, template)
        if snapshot_sha256 != fingerprint["editable_paths_sha256"]:
            raise ValueError("baseline snapshot changed during initialization")
        state = {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": "sealing_binaries",
            "created_at": utc_now(),
            "template_path": _relative(root, template_path),
            "template_sha256": sha256(canonical_json(template)),
            "template": template,
            "goal": read_json(root / template["portfolio_contract"]),
            "registry_binding": binding,
            "base_revision": legacy.git_head(root),
            "fingerprint": fingerprint,
            "usage": empty_usage(),
            "accepted_parent": None,
            "sealed_binaries": {},
            "search_started_at": None,
            "phase_checkpoint": {
                "phase_id": template["mechanism_phase"]["id"],
                "status": "pending",
                "candidate_id": None,
                "evaluation_id": None,
                "tier_result_sha256": None,
                "record": None,
            },
            "proxy": {
                "phase_id": template["mechanism_phase"]["id"],
                "status": "pending_calibration",
                "reason": "fixed sentinels have not been calibrated",
                "calibration": None,
                "audits_completed": 0,
            },
        }
        _write_run_contract(staging, state)
        write_state(staging, state)
        append_event(
            staging / "proxy-events.jsonl",
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event": "proxy_pending_calibration",
                "phase_id": template["mechanism_phase"]["id"],
                "reason": state["proxy"]["reason"],
                "recorded_at": utc_now(),
            },
        )
        staging.rename(run_dir)
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return _continue_binary_sealing(root, run_dir, state)


def _accepted_baseline_records(
    run_dir: Path, template: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    tiers = {tier["id"]: tier for tier in _search_tiers(template)}
    tier_events = {
        event["evaluation_id"]: event
        for event in _events(run_dir / "tier-events.jsonl")
    }
    for event in _events(run_dir / "baseline-events.jsonl"):
        if event.get("event") != "baseline_accepted":
            continue
        tier_id = event.get("tier_id")
        if (
            not isinstance(tier_id, str)
            or tier_id not in tiers
            or tier_id in records
        ):
            raise ValueError("baseline ledger contains duplicate accepted tiers")
        evaluation_id = event.get("evaluation_id")
        tier_event = tier_events.get(evaluation_id, {})
        digest = event.get("tier_result_sha256")
        result_path = run_dir / "evaluations" / str(evaluation_id) / "tier-result.json"
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or tier_event.get("attempt", {}).get("outcome") != "success"
            or tier_event.get("attempt", {}).get("tier_result_sha256") != digest
            or not result_path.is_file()
            or sha256(result_path.read_bytes()) != digest
        ):
            raise ValueError("accepted baseline tier has no matching sealed result")
        sealed = read_json(result_path)
        validate_tier_result(sealed, tiers[tier_id])
        admitted, _, admission = _baseline_admission(tiers[tier_id], sealed)
        if not admitted:
            raise ValueError("accepted baseline no longer passes machine admission")
        record = _tier_record(sealed)
        if tiers[tier_id]["promotion"]["kind"] == "successor_screen":
            record["calibration"] = admission
        observed_treatment = event.get("treatment_median_ns")
        if (
            tier_event.get("tier_id") != tier_id
            or canonical_json(event.get("primary"))
            != canonical_json(sealed["primary"])
            or canonical_json(event.get("paired_summary"))
            != canonical_json(record["paired_summary"])
            or canonical_json(event.get("admission"))
            != canonical_json(admission)
            or isinstance(observed_treatment, bool)
            or not isinstance(observed_treatment, (int, float))
            or float(observed_treatment)
            != record["treatment_median_ns"]
        ):
            raise ValueError("accepted baseline tier disagrees with its sealed result")
        records[tier_id] = record
    return records


def _baseline_evaluation_id(run_dir: Path, tier_id: str) -> str:
    prefix = f"baseline-{tier_id}"
    prior = list((run_dir / "evaluations").glob(f"{prefix}*"))
    if not prior:
        return prefix
    return f"{prefix}-retry-{len(prior) + 1:03d}"


def _continue_initialization(
    root: Path, run_dir: Path, state: dict[str, Any]
) -> dict[str, Any]:
    template = state["template"]
    fingerprint = state["fingerprint"]
    legacy = _legacy()
    params = _params(template, [])
    tier_records = _accepted_baseline_records(run_dir, template)
    for tier in _search_tiers(template):
        if tier["id"] in tier_records:
            continue
        evaluation_id = _baseline_evaluation_id(run_dir, tier["id"])
        inflight = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "kind": "baseline",
            "evaluation_id": evaluation_id,
            "tier_id": tier["id"],
            "params": params,
            "started_at": utc_now(),
        }
        write_inflight(run_dir, inflight)
        event, result = execute_tier(
            root,
            run_dir,
            state,
            tier,
            params,
            inflight["evaluation_id"],
            expected_editable_digest=fingerprint["editable_paths_sha256"],
        )
        write_state(run_dir, state)
        if result is None:
            if state["status"] != "sealed_binary_invalid":
                state["status"] = "initialization_retryable"
            write_state(run_dir, state)
            (run_dir / "inflight.json").unlink()
            raise ValueError(
                f"{tier['id']} baseline evaluator failed: {event['attempt']['error']}"
            )
        passed, reason = _required_guards(state, tier, result)
        if not passed:
            state["status"] = "initialization_retryable"
            write_state(run_dir, state)
            (run_dir / "inflight.json").unlink()
            raise ValueError(f"{tier['id']} baseline evaluator is invalid: {reason}")
        admitted, reason, admission = _baseline_admission(tier, result)
        if not admitted:
            append_event(
                run_dir / "baseline-events.jsonl",
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "event": "baseline_rejected",
                    "evaluation_id": inflight["evaluation_id"],
                    "tier_id": tier["id"],
                    "reason": reason,
                    "admission": admission,
                    "tier_result_sha256": event["attempt"].get(
                        "tier_result_sha256"
                    ),
                    "recorded_at": utc_now(),
                },
            )
            state["status"] = "initialization_retryable"
            write_state(run_dir, state)
            (run_dir / "inflight.json").unlink()
            raise ValueError(f"{tier['id']} baseline was not admitted: {reason}")
        record = _tier_record(result)
        if tier["promotion"]["kind"] == "successor_screen":
            record["calibration"] = admission
        tier_records[tier["id"]] = record
        append_event(
            run_dir / "baseline-events.jsonl",
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event": "baseline_accepted",
                "evaluation_id": inflight["evaluation_id"],
                "tier_id": tier["id"],
                "primary": result["primary"],
                "paired_summary": record["paired_summary"],
                "treatment_median_ns": record["treatment_median_ns"],
                "admission": admission,
                "tier_result_sha256": event["attempt"]["tier_result_sha256"],
                "recorded_at": utc_now(),
            },
        )
    if (
        legacy.path_digest(root, template["scope"]["editable"])
        != fingerprint["editable_paths_sha256"]
    ):
        raise ValueError("editable source changed during baseline evaluation")
    _assert_frozen(root, state)
    representative = tier_records["representative"]
    state["accepted_parent"] = {
        "id": "baseline",
        "params": params,
        "metric": representative["metric"],
        "relative_mad": representative["relative_mad"],
        "paired_summary": representative["paired_summary"],
        "tiers": tier_records,
        "snapshot": "baseline",
        "editable_paths_sha256": fingerprint["editable_paths_sha256"],
        "snapshot_sha256": legacy.path_digest(
            run_dir / "snapshots" / "baseline", template["scope"]["editable"]
        ),
    }
    state["status"] = "active"
    _refresh_calendar(state)
    write_state(run_dir, state)
    (run_dir / "inflight.json").unlink(missing_ok=True)
    return state


def resume_initialization(root: Path, run_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    state = load_state(run_dir)
    _validate_live_state(root, state)
    if state["status"] not in {
        "sealing_binaries",
        "sealing_binaries_retryable",
        "initializing",
        "initialization_retryable",
    }:
        raise ValueError("run initialization is not retryable")
    if state["accepted_parent"] is not None:
        raise ValueError("retryable initialization already has an accepted parent")
    if (run_dir / "inflight.json").exists():
        raise ValueError("an interrupted evaluation must be recovered first")
    _assert_frozen(root, state)
    if (
        _legacy().path_digest(root, state["template"]["scope"]["editable"])
        != state["fingerprint"]["editable_paths_sha256"]
    ):
        raise ValueError("editable source changed during interrupted initialization")
    if state["status"] in {
        "sealing_binaries",
        "sealing_binaries_retryable",
    }:
        state["status"] = "sealing_binaries"
        write_state(run_dir, state)
        return _continue_binary_sealing(root, run_dir, state)
    if state["status"] != "initializing":
        state["status"] = "initializing"
        write_state(run_dir, state)
    return _continue_initialization(root, run_dir, state)


def _candidate_id(state: dict[str, Any]) -> str:
    return f"candidate-{int(state['usage']['candidates_admitted']) + 1:03d}"


def _validation_id(run_dir: Path) -> str:
    index = 1
    while any((run_dir / "evaluations").glob(f"validation-{index:03d}-*")):
        index += 1
    return f"validation-{index:03d}"


def _accepted_validation(
    run_dir: Path,
    state: dict[str, Any],
    role: str,
    parent_id: str,
    revision: str,
) -> dict[str, Any]:
    matches = [
        event
        for event in _events(run_dir / "kernel-validations.jsonl")
        if event.get("role") == role
        and event.get("accepted_parent") == parent_id
        and event.get("revision") == revision
        and event.get("status") == "accepted"
    ]
    if not matches:
        raise ValueError(f"run status has no accepted {role} evidence")
    if len(matches) > 1:
        raise ValueError(f"run has duplicate accepted {role} evidence")
    record = matches[0]
    _sealed_validation_result(run_dir, state, record)
    return record


def _validation_for_evaluation(
    run_dir: Path, evaluation_id: str
) -> Optional[dict[str, Any]]:
    matches = [
        event
        for event in _events(run_dir / "kernel-validations.jsonl")
        if event.get("event") == "kernel_validated"
        and event.get("evaluation_id") == evaluation_id
    ]
    if len(matches) > 1:
        raise ValueError("validation ledger contains duplicate evaluation records")
    return matches[0] if matches else None


def _sealed_validation_result(
    run_dir: Path,
    state: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    evaluation_id = record.get("evaluation_id")
    tier_id = record.get("tier_id")
    digest = record.get("tier_result_sha256")
    tier_events = [
        event
        for event in _events(run_dir / "tier-events.jsonl")
        if event.get("evaluation_id") == evaluation_id
    ]
    tiers = [
        tier
        for tier in state["template"]["evaluation"]["tiers"]
        if tier.get("id") == tier_id
    ]
    role = record.get("role")
    expected_tier_role = (
        "representative" if role == "representative_revalidation" else role
    )
    parent = state.get("accepted_parent")
    result_path = run_dir / "evaluations" / str(evaluation_id) / "tier-result.json"
    try:
        if (
            record.get("schema_version") != EVENT_SCHEMA_VERSION
            or record.get("event") != "kernel_validated"
            or record.get("status") not in {"accepted", "rejected"}
            or not isinstance(evaluation_id, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or len(tier_events) != 1
            or len(tiers) != 1
            or not isinstance(parent, dict)
            or record.get("accepted_parent") != parent.get("id")
            or canonical_json(tier_events[0].get("params"))
            != canonical_json(parent.get("params"))
            or tiers[0].get("role") != expected_tier_role
            or tier_events[0].get("event") != "tier_evaluated"
            or tier_events[0].get("tier_id") != tier_id
            or tier_events[0].get("attempt", {}).get("outcome") != "success"
            or tier_events[0].get("attempt", {}).get("tier_result_sha256")
            != digest
            or not result_path.is_file()
        ):
            raise ValueError
        result_bytes = result_path.read_bytes()
        if sha256(result_bytes) != digest:
            raise ValueError
        result = read_json(result_path)
        validate_tier_result(result, tiers[0])
        if (
            canonical_json(record.get("primary"))
            != canonical_json(result.get("primary"))
            or canonical_json(record.get("paired_summary"))
            != canonical_json(result.get("replication", {}).get("summary"))
            or canonical_json(record.get("local"))
            != canonical_json(result.get("local"))
            or canonical_json(tier_events[0].get("primary"))
            != canonical_json(result.get("primary"))
            or canonical_json(tier_events[0].get("paired_summary"))
            != canonical_json(result.get("replication", {}).get("summary"))
        ):
            raise ValueError
        if role in {"holdout", "transfer"}:
            accepted, _ = _acceptance_result(state, tiers[0], result, None)
            if (record.get("status") == "accepted") is not accepted:
                raise ValueError
        elif role == "representative_revalidation":
            accepted, _ = _required_guards(state, tiers[0], result)
            accepted = accepted and _result_clears_floor(
                result,
                float(tiers[0]["promotion"]["minimum_accepted_speedup"]),
            )
            accepted = accepted and _treatment_median_ms(result) <= float(
                tiers[0]["promotion"]["maximum_treatment_ms"]
            )
            if (record.get("status") == "accepted") is not accepted:
                raise ValueError
        if record.get("role") in {"holdout", "transfer"}:
            expected_floor = _result_clears_floor(
                result,
                float(
                    state["goal"]["primary_metric"]["minimum_accepted_speedup"]
                ),
            )
            if record.get("portfolio_floor_met") is not expected_floor:
                raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "kernel validation has no matching sealed tier result"
        ) from error
    return result


def _conservative_result_speedup(result: dict[str, Any]) -> float:
    summary = result["replication"]["summary"]
    return min(
        float(result["primary"]["value"]),
        float(summary["control_first_median"]),
        float(summary["treatment_first_median"]),
    )


def _proxy_calibration_evaluation_id(
    run_dir: Path, sentinel_id: str, tier_id: str
) -> str:
    prefix = f"proxy-calibration-{sentinel_id}-{tier_id}"
    prior = list((run_dir / "evaluations").glob(f"{prefix}*"))
    return prefix if not prior else f"{prefix}-retry-{len(prior) + 1:03d}"


def _finish_proxy_calibration(
    run_dir: Path,
    state: dict[str, Any],
    calibration: dict[str, Any],
    evaluation_ids: list[str],
) -> dict[str, Any]:
    reason = (
        "fixed sentinels preserve representative ranking"
        if calibration["status"] == "enabled"
        else "fixed sentinels materially misrank representative performance"
    )
    event = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event": (
            "proxy_calibrated"
            if calibration["status"] == "enabled"
            else "proxy_disabled"
        ),
        "phase_id": state["template"]["mechanism_phase"]["id"],
        "status": calibration["status"],
        "reason": reason,
        "evaluation_ids": evaluation_ids,
        "calibration": calibration,
        "recorded_at": utc_now(),
    }
    append_event(run_dir / "proxy-events.jsonl", event)
    _apply_proxy_calibration_event(state, event)
    write_state(run_dir, state)
    return event


def _apply_proxy_calibration_event(
    state: dict[str, Any], event: dict[str, Any]
) -> None:
    phase_id = state["template"]["mechanism_phase"]["id"]
    calibration = event.get("calibration")
    policy = state["template"]["search_policy"]["proxy_calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("proxy calibration event has no calibration evidence")
    expected = _proxy_calibration_decision(policy, calibration.get("sentinels", []))
    status = expected["status"]
    expected_event = "proxy_calibrated" if status == "enabled" else "proxy_disabled"
    expected_reason = (
        "fixed sentinels preserve representative ranking"
        if status == "enabled"
        else "fixed sentinels materially misrank representative performance"
    )
    evaluation_ids = event.get("evaluation_ids")
    if (
        event.get("event") != expected_event
        or event.get("phase_id") != phase_id
        or event.get("status") != status
        or event.get("reason") != expected_reason
        or canonical_json(calibration) != canonical_json(expected)
        or not isinstance(evaluation_ids, list)
        or len(evaluation_ids) != 2 * len(policy["sentinels"])
        or len(set(evaluation_ids)) != len(evaluation_ids)
        or not isinstance(event.get("recorded_at"), str)
    ):
        raise ValueError("proxy calibration terminal event is invalid")
    state["proxy"]["status"] = status
    state["proxy"]["reason"] = expected_reason
    state["proxy"]["calibration"] = calibration
    if state.get("search_started_at") is None:
        state["search_started_at"] = event["recorded_at"]


def _sealed_proxy_calibration_result(
    run_dir: Path,
    sentinel_id: str,
    tier: dict[str, Any],
    params: dict[str, str],
) -> Optional[tuple[str, dict[str, Any]]]:
    prefix = f"proxy-calibration-{sentinel_id}-{tier['id']}"
    matches = []
    for event in _events(run_dir / "tier-events.jsonl"):
        evaluation_id = event.get("evaluation_id")
        attempt = event.get("attempt")
        if (
            event.get("event") != "tier_evaluated"
            or not isinstance(evaluation_id, str)
            or not (
                evaluation_id == prefix
                or evaluation_id.startswith(f"{prefix}-retry-")
            )
            or event.get("tier_id") != tier["id"]
            or event.get("params") != params
            or not isinstance(attempt, dict)
            or attempt.get("outcome") != "success"
        ):
            continue
        path = run_dir / "evaluations" / evaluation_id / "tier-result.json"
        if not path.is_file():
            raise ValueError("proxy calibration result file is missing")
        payload = path.read_bytes()
        if attempt.get("tier_result_sha256") != sha256(payload):
            raise ValueError("proxy calibration result digest does not match")
        result = json.loads(payload)
        validate_tier_result(result, tier)
        matches.append((evaluation_id, result))
    if not matches:
        return None
    digests = {sha256(canonical_json(result)) for _, result in matches}
    if len(digests) != 1:
        raise ValueError("proxy calibration has conflicting sealed retries")
    return matches[0]


def calibrate_proxy(
    root: Path, run_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    state = load_state(run_dir)
    _validate_live_state(root, state)
    _assert_frozen(root, state)
    if state["status"] != "active":
        raise ValueError("proxy calibration requires an active run")
    if (run_dir / "inflight.json").exists():
        raise ValueError("an interrupted evaluation must be recovered first")
    proxy = state.get("proxy")
    if not isinstance(proxy, dict) or proxy.get("phase_id") != state["template"][
        "mechanism_phase"
    ]["id"]:
        raise ValueError("proxy state does not match the mechanism phase")
    prior = [
        event
        for event in _events(run_dir / "proxy-events.jsonl")
        if event.get("event") in {"proxy_calibrated", "proxy_disabled"}
        and event.get("phase_id") == state["template"]["mechanism_phase"]["id"]
        and "calibration" in event
    ]
    if len(prior) > 1:
        raise ValueError("proxy calibration ledger has duplicate terminal events")
    if prior:
        _apply_proxy_calibration_event(state, prior[0])
        write_state(run_dir, state)
        return prior[0], state
    if proxy.get("calibration") is not None:
        raise ValueError("proxy calibration state has no terminal ledger event")
    if proxy.get("status") != "pending_calibration":
        raise ValueError("proxy calibration cannot repair a disabled live phase")

    template = state["template"]
    policy = template["search_policy"]["proxy_calibration"]
    proxy_tier = _tier_by_role(template, "proxy")
    representative = _tier_by_role(template, "representative")
    editable_digest = state["fingerprint"]["editable_paths_sha256"]
    records = []
    evaluation_ids = []
    for sentinel in policy["sentinels"]:
        params = _validate_params(template, dict(sentinel["params"]))
        scores = {}
        for tier, score_name in (
            (proxy_tier, "proxy_score"),
            (representative, "representative_score"),
        ):
            sealed = _sealed_proxy_calibration_result(
                run_dir, sentinel["id"], tier, params
            )
            if sealed is not None:
                evaluation_id, result = sealed
                passed, reason = _required_guards(state, tier, result)
                if not passed:
                    raise ValueError(
                        f"proxy sentinel {sentinel['id']} is invalid: {reason}"
                    )
                evaluation_ids.append(evaluation_id)
                scores[score_name] = float(result["primary"]["value"])
                continue
            evaluation_id = _proxy_calibration_evaluation_id(
                run_dir, sentinel["id"], tier["id"]
            )
            inflight = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "kind": "proxy_calibration",
                "phase_id": template["mechanism_phase"]["id"],
                "sentinel_id": sentinel["id"],
                "evaluation_id": evaluation_id,
                "tier_id": tier["id"],
                "params": params,
                "editable_paths_sha256": editable_digest,
                "started_at": utc_now(),
            }
            write_inflight(run_dir, inflight)
            event, result = execute_tier(
                root,
                run_dir,
                state,
                tier,
                params,
                evaluation_id,
                expected_editable_digest=editable_digest,
            )
            evaluation_ids.append(evaluation_id)
            write_state(run_dir, state)
            (run_dir / "inflight.json").unlink(missing_ok=True)
            if result is None:
                raise ValueError(
                    f"proxy sentinel {sentinel['id']} failed: "
                    f"{event['attempt']['error']}"
                )
            passed, reason = _required_guards(state, tier, result)
            if not passed:
                raise ValueError(
                    f"proxy sentinel {sentinel['id']} is invalid: {reason}"
                )
            scores[score_name] = float(result["primary"]["value"])
        records.append({"id": sentinel["id"], **scores})

    calibration = _proxy_calibration_decision(policy, records)
    event = _finish_proxy_calibration(
        run_dir, state, calibration, evaluation_ids
    )
    return event, state


def trial(
    root: Path,
    run_dir: Path,
    overrides: list[str],
    summary: str,
    *,
    direct_to_representative: bool = False,
    direct_reason: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    state = load_state(run_dir)
    _validate_live_state(root, state)
    if state["status"] != "active":
        raise ValueError("run is not active")
    if (run_dir / "inflight.json").exists():
        raise ValueError("an interrupted evaluation must be recovered first")
    _assert_frozen(root, state)
    requested_direct = direct_to_representative
    if requested_direct != bool(direct_reason and direct_reason.strip()):
        raise ValueError(
            "the direct representative lane requires both its flag and a reason"
        )
    if (
        requested_direct
        and state["template"]["search_policy"]["direct_to_representative_lane"]
        is not True
    ):
        raise ValueError("the run contract does not admit a direct representative lane")
    proxy = state.get("proxy", {"status": "enabled", "reason": "legacy run"})
    if not isinstance(proxy, dict) or proxy.get("status") not in {
        "pending_calibration",
        "enabled",
        "disabled",
    }:
        raise ValueError("proxy state is invalid")
    if not requested_direct and proxy["status"] == "pending_calibration":
        raise ValueError(
            "proxy ranking is not calibrated; calibrate it or use the direct lane"
        )
    if not requested_direct and proxy["status"] == "disabled":
        direct_to_representative = True
        direct_reason = str(proxy["reason"])
    checkpoint_state = _phase_checkpoint_state(state)
    checkpoint_contract = state["template"]["mechanism_phase"]["checkpoint"]
    checkpoint_probe = (
        checkpoint_state["status"] == "pending"
        and int(state["usage"]["candidates_admitted"]) + 1
        >= int(checkpoint_contract["after_candidates"])
    )
    lane = {
        "requested": requested_direct,
        "effective": (
            "representative_direct" if direct_to_representative else "proxy_first"
        ),
        "reason": direct_reason if direct_to_representative else None,
        "checkpoint_probe": checkpoint_probe,
    }
    audit_every = int(
        state["template"]["search_policy"]["proxy_calibration"][
            "audit_every_candidates"
        ]
    )
    params = dict(state["accepted_parent"]["params"])
    for override in overrides:
        if "=" not in override:
            raise ValueError("parameter overrides must use NAME=VALUE")
        name, value = override.split("=", 1)
        params[name] = value
    params = _validate_params(state["template"], params)
    if state.get("search_started_at") is None:
        state["search_started_at"] = utc_now()
        write_state(run_dir, state)
    _require_search_phase_budget(state)
    candidate_id = _candidate_id(state)
    editable_digest = _legacy().path_digest(
        root, state["template"]["scope"]["editable"]
    )
    state["usage"]["candidates_admitted"] = int(
        state["usage"]["candidates_admitted"]
    ) + 1
    proxy_audit_due = (
        not direct_to_representative
        and proxy["status"] == "enabled"
        and int(state["usage"]["candidates_admitted"]) % audit_every == 0
    )
    write_inflight(
        run_dir,
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "kind": "candidate_pending",
            "candidate_id": candidate_id,
            "phase_id": state["template"]["mechanism_phase"]["id"],
            "lane": lane,
            "evaluation_id": candidate_id,
            "params": params,
            "editable_paths_sha256": editable_digest,
            "started_at": utc_now(),
        },
    )
    append_event(
        run_dir / "candidate-events.jsonl",
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event": "candidate_admitted",
            "candidate_id": candidate_id,
            "phase_id": state["template"]["mechanism_phase"]["id"],
            "lane": lane,
            "summary": summary,
            "params": params,
            "editable_paths_sha256": editable_digest,
            "recorded_at": utc_now(),
        },
    )
    write_state(run_dir, state)
    parent = state["accepted_parent"]
    verdict = "keep"
    reason = "candidate cleared every search tier"
    tier_results: dict[str, dict[str, Any]] = {}
    evaluation_ids: list[str] = []
    screen_attempts: list[dict[str, Any]] = []
    representative_result = None
    phase_checkpoint = None
    phase_exhausted = False
    screen_rejected = False
    proxy_audit = None
    proxy_misranked = False
    search_tiers = _search_tiers(state["template"])
    if direct_to_representative:
        trial_tiers = []
        if checkpoint_probe:
            trial_tiers.append(_tier_by_role(state["template"], "proxy"))
        trial_tiers.append(_tier_by_role(state["template"], "representative"))
    else:
        trial_tiers = search_tiers
    if checkpoint_state["status"] != "pending":
        phase_checkpoint = checkpoint_state["record"]
    for tier in trial_tiers:
        attempt_index = 0
        while True:
            base_evaluation_id = f"{candidate_id}-{tier['id']}"
            evaluation_id = (
                base_evaluation_id
                if attempt_index == 0
                else f"{base_evaluation_id}-retry-{attempt_index + 1:03d}"
            )
            inflight = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "kind": "candidate",
                "candidate_id": candidate_id,
                "phase_id": state["template"]["mechanism_phase"]["id"],
                "lane": lane,
                "evaluation_id": evaluation_id,
                "tier_id": tier["id"],
                "params": params,
                "editable_paths_sha256": editable_digest,
                "started_at": utc_now(),
            }
            write_inflight(run_dir, inflight)
            try:
                event, result = execute_tier(
                    root,
                    run_dir,
                    state,
                    tier,
                    params,
                    evaluation_id,
                    expected_editable_digest=editable_digest,
                    search_phase=True,
                    retain_inflight_on_budget_exhausted=True,
                )
            except BudgetExhausted as error:
                verdict = "invalid"
                reason = str(error)
                break
            evaluation_ids.append(evaluation_id)
            write_state(run_dir, state)
            _assert_frozen(root, state)
            if result is None:
                verdict = "invalid"
                reason = (
                    event["attempt"]["error"] or "evaluator result is invalid"
                )
                break
            passed, reason = _required_guards(state, tier, result)
            if not passed:
                verdict = "invalid"
                break
            record = _tier_record(result)
            tier_results[tier["id"]] = record
            if tier["promotion"]["kind"] == "successor_screen":
                live_checkpoint = _phase_checkpoint_state(state)
                if live_checkpoint["status"] == "pending":
                    phase_checkpoint = _phase_checkpoint(state, result)
                    if phase_checkpoint["due"]:
                        _seal_phase_checkpoint(
                            state,
                            phase_checkpoint,
                            candidate_id,
                            evaluation_id,
                            event["attempt"].get("tier_result_sha256"),
                        )
                        write_state(run_dir, state)
                else:
                    phase_checkpoint = live_checkpoint["record"]
                if phase_checkpoint["due"] and not phase_checkpoint["passed"]:
                    verdict = "discard"
                    reason = "candidate missed the mechanism-phase checkpoint"
                    phase_exhausted = True
                    state["status"] = "phase_exhausted"
                    write_state(run_dir, state)
                    break
                if direct_to_representative:
                    reason = "checkpoint probe passed; proxy ranking was bypassed"
                    break
                retry_limit = int(
                    tier["promotion"]["inconclusive_retry_limit"]
                )
                disposition, reason = _successor_screen_disposition(
                    tier,
                    result,
                    parent["tiers"][tier["id"]],
                    retry_available=attempt_index < retry_limit,
                )
                wall_summary = _successor_wall_summary(tier, result)
                screen_attempts.append(
                    {
                        "evaluation_id": evaluation_id,
                        "disposition": disposition,
                        "reason": reason,
                        "gpu_metric": record["metric"],
                        "gpu_relative_mad": record["relative_mad"],
                        "wall_metric": wall_summary["median"],
                        "wall_relative_mad": float(wall_summary["mad"])
                        / float(wall_summary["median"]),
                    }
                )
                if disposition == "retry":
                    attempt_index += 1
                    continue
                if disposition == "discard":
                    if proxy_audit_due:
                        screen_rejected = True
                        reason = "proxy rejection advanced to its scheduled representative audit"
                    else:
                        verdict = "discard"
            else:
                promoted, reason = _promotion_pass(
                    tier, result, parent["tiers"][tier["id"]]
                )
                if promoted and tier["role"] == "representative":
                    promoted, reason = _phase_success(state, result, parent)
                if (
                    tier["role"] == "representative"
                    and proxy_audit_due
                    and screen_rejected
                ):
                    proxy_misranked = promoted
                    proxy_audit = {
                        "candidate_id": candidate_id,
                        "proxy_disposition": "discard",
                        "representative_promoted": promoted,
                        "ranking_ok": not promoted,
                    }
                    proxy["audits_completed"] = int(
                        proxy.get("audits_completed", 0)
                    ) + 1
                if not promoted:
                    verdict = "discard"
                elif tier["role"] == "representative":
                    representative_result = result
            break
        if verdict != "keep":
            break

    decision_tier_results = dict(tier_results)
    if verdict == "keep" and representative_result is not None:
        decision_tier_results = {**parent["tiers"], **tier_results}
        for tier in search_tiers:
            if tier["promotion"]["kind"] == "successor_screen":
                decision_tier_results[tier["id"]] = parent["tiers"][tier["id"]]

    decision = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event": "candidate_decided",
        "candidate_id": candidate_id,
        "phase_id": state["template"]["mechanism_phase"]["id"],
        "lane": lane,
        "evaluation_ids": evaluation_ids,
        "parent_id": parent["id"],
        "summary": summary,
        "params": params,
        "verdict": verdict,
        "reason": reason,
        "primary": (
            representative_result["primary"]
            if representative_result is not None
            else None
        ),
        "paired_summary": (
            representative_result["replication"]["summary"]
            if representative_result is not None
            else None
        ),
        "tier_results": decision_tier_results,
        "screen_attempts": screen_attempts,
        "phase_checkpoint": phase_checkpoint,
        "proxy_audit": proxy_audit,
        "recorded_at": utc_now(),
    }
    append_event(run_dir / "decision-events.jsonl", decision)
    if proxy_audit is not None:
        append_event(
            run_dir / "proxy-events.jsonl",
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event": "proxy_audited",
                "phase_id": state["template"]["mechanism_phase"]["id"],
                **proxy_audit,
                "recorded_at": utc_now(),
            },
        )
    if verdict == "keep" and representative_result is not None:
        snapshot = run_dir / "snapshots" / candidate_id
        _legacy().snapshot_paths(
            root, state["template"]["scope"]["editable"], snapshot
        )
        snapshot_digest = _legacy().path_digest(
            snapshot, state["template"]["scope"]["editable"]
        )
        if snapshot_digest != editable_digest:
            raise ValueError("accepted candidate snapshot does not match its source")
        paired = representative_result["replication"]["summary"]
        state["accepted_parent"] = {
            "id": candidate_id,
            "lane": lane,
            "params": params,
            "metric": float(representative_result["primary"]["value"]),
            "relative_mad": float(paired["mad"]) / float(paired["median"]),
            "paired_summary": paired,
            "tiers": decision["tier_results"],
            "snapshot": candidate_id,
            "editable_paths_sha256": editable_digest,
            "snapshot_sha256": snapshot_digest,
        }
        if (
            (direct_to_representative or proxy_misranked)
            and proxy.get("status") in {"enabled", "pending_calibration"}
        ):
            proxy["status"] = "disabled"
            proxy["reason"] = (
                "a representative audit found a material proxy false negative"
                if proxy_misranked
                else "an accepted direct candidate invalidated proxy ranking"
            )
            append_event(
                run_dir / "proxy-events.jsonl",
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "event": "proxy_disabled",
                    "phase_id": state["template"]["mechanism_phase"]["id"],
                    "candidate_id": candidate_id,
                    "reason": proxy["reason"],
                    "recorded_at": utc_now(),
                },
            )
        state["status"] = "active"
    else:
        _legacy().restore_snapshot(
            root,
            state["template"]["scope"]["editable"],
            run_dir / "snapshots" / parent["snapshot"],
        )
        if phase_exhausted:
            state["status"] = "phase_exhausted"
    _refresh_calendar(state)
    write_state(run_dir, state)
    (run_dir / "inflight.json").unlink(missing_ok=True)
    return decision, state


def _local_acceptance(state: dict[str, Any], tier: dict[str, Any]) -> None:
    parent = state["accepted_parent"]
    floor = float(tier["promotion"]["minimum_accepted_speedup"])
    summary = parent["paired_summary"]
    if any(
        float(value) < floor
        for value in (
            parent["metric"],
            summary["control_first_median"],
            summary["treatment_first_median"],
        )
    ):
        raise ValueError("accepted parent has not cleared the 5x local gate")
    latency_ms = (
        float(parent["tiers"][tier["id"]]["treatment_median_ns"]) / 1_000_000.0
    )
    if latency_ms > float(tier["promotion"]["maximum_treatment_ms"]):
        raise ValueError("accepted parent has not cleared the calibrated latency bar")


def _result_clears_floor(result: dict[str, Any], floor: float) -> bool:
    summary = result["replication"]["summary"]
    return all(
        float(value) >= floor
        for value in (
            result["primary"]["value"],
            summary["control_first_median"],
            summary["treatment_first_median"],
        )
    )


def _treatment_median_ms(result: dict[str, Any], local: bool = False) -> float:
    source = result["local"] if local else result
    return statistics.median(
        float(pair["arms"]["treatment"]["primary_ns"])
        for pair in source["replication"]["pairs"]
    ) / 1_000_000.0


def _acceptance_result(
    state: dict[str, Any],
    tier: dict[str, Any],
    result: Optional[dict[str, Any]],
    error: Optional[str],
) -> tuple[bool, str]:
    if result is None:
        return False, error or f"{tier['role']} evaluator is invalid"
    passed, reason = _required_guards(state, tier, result)
    floor = float(tier["promotion"]["minimum_portfolio_speedup"])
    if passed and not _result_clears_floor(result, floor):
        return (
            False,
            f"PIOP {tier['role']} fell below its kernel-validation portfolio floor",
        )
    local = result.get("local", {})
    local_summary = local.get("replication", {}).get("summary", {})
    local_values = (
        local.get("primary", {}).get("value"),
        local_summary.get("control_first_median"),
        local_summary.get("treatment_first_median"),
    )
    if passed and any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) < float(tier["promotion"]["minimum_local_speedup"])
        for value in local_values
    ):
        return False, f"PIOP {tier['role']} did not retain the 5x local-kernel gate"
    if passed and _treatment_median_ms(result, local=True) > float(
        tier["promotion"]["maximum_local_treatment_ms"]
    ):
        return False, f"PIOP {tier['role']} exceeded the calibrated local latency bar"
    return passed, reason


def _run_acceptance_tier(
    root: Path,
    run_dir: Path,
    state: dict[str, Any],
    tier: dict[str, Any],
    accepted: dict[str, Any],
    revision: str,
    validation_id: str,
    reserve_id: str,
    accepted_status: str,
    rejected_status: str,
    invalid_status: Optional[str] = None,
) -> tuple[dict[str, Any], bool, str]:
    evaluation_id = f"{validation_id}-{tier['role']}"
    inflight = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "kind": tier["role"],
        "evaluation_id": evaluation_id,
        "tier_id": tier["id"],
        "params": accepted["params"],
        "budget_reserve": reserve_id,
        "started_at": utc_now(),
    }
    write_inflight(run_dir, inflight)
    event, result = execute_tier(
        root,
        run_dir,
        state,
        tier,
        accepted["params"],
        evaluation_id,
        expected_editable_digest=accepted["editable_paths_sha256"],
        budget_reserve=reserve_id,
    )
    _assert_frozen(root, state)
    passed, reason = _acceptance_result(
        state, tier, result, event["attempt"]["error"]
    )
    record = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event": "kernel_validated",
        "evaluation_id": evaluation_id,
        "tier_id": tier["id"],
        "role": tier["role"],
        "accepted_parent": accepted["id"],
        "revision": revision,
        "status": (
            "accepted" if passed else "invalid" if result is None else "rejected"
        ),
        "reason": reason,
        "primary": result["primary"] if result is not None else None,
        "paired_summary": (
            result["replication"]["summary"] if result is not None else None
        ),
        "local": result.get("local") if result is not None else None,
        "tier_result_sha256": event["attempt"].get("tier_result_sha256"),
        "portfolio_floor_met": (
            result is not None
            and _result_clears_floor(
                result,
                float(state["goal"]["primary_metric"]["minimum_accepted_speedup"]),
            )
        ),
        "recorded_at": utc_now(),
    }
    append_event(run_dir / "kernel-validations.jsonl", record)
    if passed:
        state["status"] = accepted_status
    elif result is None and invalid_status is not None:
        state["status"] = invalid_status
    else:
        state["status"] = rejected_status
    _refresh_calendar(state)
    write_state(run_dir, state)
    return record, passed, reason


def validate_production(
    root: Path, run_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    state = load_state(run_dir)
    _validate_live_state(root, state)
    if state["status"] not in {
        "active",
        "holdout_retryable",
        "kernel_accepted",
        "kernel_transferred",
    }:
        raise ValueError("run is not active")
    if (run_dir / "inflight.json").exists():
        raise ValueError("an interrupted evaluation must be recovered first")
    _assert_frozen(root, state)
    tier = _tier_by_role(state["template"], "holdout")
    transfer = _tier_by_role(state["template"], "transfer")
    representative = _tier_by_role(state["template"], "representative")
    _local_acceptance(state, representative)
    legacy = _legacy()
    if not legacy.git_worktree_clean(root):
        raise ValueError("production validation requires a clean worktree")
    live_revision = legacy.git_head(root)
    legacy.validate_production_revision_scope(
        root,
        state["base_revision"],
        live_revision,
        state["template"]["scope"]["editable"],
    )
    accepted = state["accepted_parent"]
    if legacy.path_digest(
        root, state["template"]["scope"]["editable"]
    ) != accepted["editable_paths_sha256"]:
        raise ValueError("live source does not match the accepted parent")
    if state["status"] == "kernel_transferred":
        return (
            _accepted_validation(
                run_dir, state, "transfer", accepted["id"], live_revision
            ),
            state,
        )
    validation_id = _validation_id(run_dir)
    if state["status"] == "active":
        revalidation_id = f"{validation_id}-representative"
        revalidation_inflight = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "kind": "revalidation",
            "evaluation_id": revalidation_id,
            "tier_id": representative["id"],
            "params": accepted["params"],
            "budget_reserve": "representative_revalidation",
            "started_at": utc_now(),
        }
        write_inflight(run_dir, revalidation_inflight)
        revalidation_event, revalidation = execute_tier(
            root,
            run_dir,
            state,
            representative,
            accepted["params"],
            revalidation_id,
            expected_editable_digest=accepted["editable_paths_sha256"],
            budget_reserve="representative_revalidation",
        )
        _assert_frozen(root, state)
        write_state(run_dir, state)
        revalidation_passed = revalidation is not None
        revalidation_reason = (
            revalidation_event["attempt"]["error"]
            or "representative revalidation is invalid"
        )
        if revalidation is not None:
            revalidation_passed, revalidation_reason = _required_guards(
                state, representative, revalidation
            )
            if revalidation_passed and not _result_clears_floor(
                revalidation,
                float(representative["promotion"]["minimum_accepted_speedup"]),
            ):
                revalidation_passed = False
                revalidation_reason = (
                    "fresh representative result did not clear the 5x local gate"
                )
            if revalidation_passed and _treatment_median_ms(
                revalidation
            ) > float(representative["promotion"]["maximum_treatment_ms"]):
                revalidation_passed = False
                revalidation_reason = (
                    "fresh representative result exceeded the calibrated latency bar"
                )
        if not revalidation_passed:
            record = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event": "kernel_validated",
                "evaluation_id": revalidation_id,
                "tier_id": representative["id"],
                "role": "representative_revalidation",
                "accepted_parent": accepted["id"],
                "revision": live_revision,
                "status": "rejected",
                "reason": revalidation_reason,
                "primary": (
                    revalidation["primary"] if revalidation is not None else None
                ),
                "paired_summary": (
                    revalidation["replication"]["summary"]
                    if revalidation is not None
                    else None
                ),
                "local": None,
                "recorded_at": utc_now(),
            }
            append_event(run_dir / "kernel-validations.jsonl", record)
            _refresh_calendar(state)
            write_state(run_dir, state)
            (run_dir / "inflight.json").unlink()
            raise ValueError(revalidation_reason)

        append_event(
            run_dir / "kernel-validations.jsonl",
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event": "kernel_validated",
                "evaluation_id": revalidation_id,
                "tier_id": representative["id"],
                "role": "representative_revalidation",
                "accepted_parent": accepted["id"],
                "revision": live_revision,
                "status": "accepted",
                "reason": revalidation_reason,
                "primary": revalidation["primary"],
                "paired_summary": revalidation["replication"]["summary"],
                "local": None,
                "tier_result_sha256": revalidation_event["attempt"].get(
                    "tier_result_sha256"
                ),
                "recorded_at": utc_now(),
            },
        )
    elif state["status"] == "holdout_retryable":
        _accepted_validation(
            run_dir,
            state,
            "representative_revalidation",
            accepted["id"],
            live_revision,
        )

    if state["status"] in {"active", "holdout_retryable"}:
        _, passed, reason = _run_acceptance_tier(
            root,
            run_dir,
            state,
            tier,
            accepted,
            live_revision,
            validation_id,
            "piop_holdout",
            "kernel_accepted",
            "holdout_rejected",
            "holdout_retryable",
        )
        if not passed:
            (run_dir / "inflight.json").unlink()
            raise ValueError(reason)
    else:
        _accepted_validation(
            run_dir, state, "holdout", accepted["id"], live_revision
        )
    transfer_record, transferred, transfer_reason = _run_acceptance_tier(
        root,
        run_dir,
        state,
        transfer,
        accepted,
        live_revision,
        validation_id,
        "piop_transfer",
        "kernel_transferred",
        "kernel_accepted",
    )
    (run_dir / "inflight.json").unlink()
    if not transferred:
        raise ValueError(transfer_reason)
    return transfer_record, state


def _restore_with_quarantine(root: Path, run_dir: Path, state: dict[str, Any]) -> Optional[Path]:
    parent = state["accepted_parent"]
    if parent is None:
        return None
    editable = state["template"]["scope"]["editable"]
    quarantine = None
    if _legacy().path_digest(root, editable) != parent["editable_paths_sha256"]:
        quarantine = (
            run_dir
            / "quarantine"
            / utc_now().replace(":", "-")
        )
        _legacy().snapshot_paths(root, editable, quarantine)
    _legacy().restore_snapshot(
        root, editable, run_dir / "snapshots" / parent["snapshot"]
    )
    return quarantine


def _recover_committed_candidate(
    root: Path,
    run_dir: Path,
    state: dict[str, Any],
    inflight: dict[str, Any],
) -> tuple[bool, Optional[Path]]:
    candidate_id = inflight.get("candidate_id")
    decisions = [
        event
        for event in _events(run_dir / "decision-events.jsonl")
        if event.get("event") == "candidate_decided"
        and event.get("candidate_id") == candidate_id
    ]
    if len(decisions) > 1:
        raise ValueError("candidate ledger contains duplicate final decisions")
    if not decisions:
        return False, _restore_with_quarantine(root, run_dir, state)

    decision = decisions[0]
    if decision.get("verdict") != "keep":
        checkpoint = _phase_checkpoint_state(state)
        if checkpoint["status"] == "failed":
            checkpoint = _verify_sealed_phase_checkpoint(run_dir, state)
            if canonical_json(decision.get("phase_checkpoint")) != canonical_json(
                checkpoint["record"]
            ):
                raise ValueError("candidate decision checkpoint is not sealed by state")
            state["status"] = "phase_exhausted"
        return False, _restore_with_quarantine(root, run_dir, state)

    current_parent = state["accepted_parent"]
    if current_parent.get("id") == candidate_id:
        if (
            decision.get("params") != current_parent.get("params")
            or decision.get("tier_results") != current_parent.get("tiers")
            or inflight.get("editable_paths_sha256")
            != current_parent.get("editable_paths_sha256")
        ):
            raise ValueError("committed candidate decision disagrees with live state")
        _legacy().restore_snapshot(
            root,
            state["template"]["scope"]["editable"],
            run_dir / "snapshots" / current_parent["snapshot"],
        )
        state["status"] = "active"
        return True, None

    accepted_tiers = decision.get("tier_results")
    expected_tier_ids = {
        tier["id"] for tier in _search_tiers(state["template"])
    }
    if (
        not isinstance(accepted_tiers, dict)
        or set(accepted_tiers) != expected_tier_ids
    ):
        return False, _restore_with_quarantine(root, run_dir, state)
    for tier in _search_tiers(state["template"]):
        if tier["promotion"]["kind"] == "successor_screen" and canonical_json(
            accepted_tiers[tier["id"]]
        ) != canonical_json(state["accepted_parent"]["tiers"][tier["id"]]):
            return False, _restore_with_quarantine(root, run_dir, state)
    checkpoint = _verify_sealed_phase_checkpoint(run_dir, state)
    if (
        checkpoint["status"] != "passed"
        or canonical_json(decision.get("phase_checkpoint"))
        != canonical_json(checkpoint["record"])
    ):
        raise ValueError("kept candidate has no sealed phase checkpoint")

    evaluation_ids = decision.get("evaluation_ids")
    if (
        not isinstance(evaluation_ids, list)
        or len(evaluation_ids) != len(set(evaluation_ids))
    ):
        raise ValueError("candidate decision evaluation ids are invalid")
    representative = _tier_by_role(state["template"], "representative")
    representative_events = [
        event
        for event in _events(run_dir / "tier-events.jsonl")
        if event.get("evaluation_id") in evaluation_ids
        and event.get("tier_id") == representative["id"]
    ]
    if len(representative_events) != 1:
        raise ValueError("candidate decision has no unique representative result")
    representative_event = representative_events[0]
    representative_id = representative_event["evaluation_id"]
    result_path = (
        run_dir / "evaluations" / representative_id / "tier-result.json"
    )
    attempt = representative_event.get("attempt")
    if (
        representative_event.get("event") != "tier_evaluated"
        or representative_event.get("params") != decision.get("params")
        or not isinstance(attempt, dict)
        or attempt.get("outcome") != "success"
        or not result_path.is_file()
    ):
        raise ValueError("candidate representative event is invalid")
    result_payload = result_path.read_bytes()
    if attempt.get("tier_result_sha256") != sha256(result_payload):
        raise ValueError("candidate representative result digest does not match")
    representative_result = json.loads(result_payload)
    validate_tier_result(representative_result, representative)
    guards_passed, _ = _required_guards(
        state, representative, representative_result
    )
    promoted, _ = _promotion_pass(
        representative,
        representative_result,
        current_parent["tiers"][representative["id"]],
    )
    phase_passed, _ = _phase_success(
        state, representative_result, current_parent
    )
    if (
        not guards_passed
        or not promoted
        or not phase_passed
        or canonical_json(decision.get("primary"))
        != canonical_json(representative_result["primary"])
        or canonical_json(decision.get("paired_summary"))
        != canonical_json(representative_result["replication"]["summary"])
        or canonical_json(accepted_tiers[representative["id"]])
        != canonical_json(_tier_record(representative_result))
    ):
        raise ValueError("candidate decision is not supported by representative evidence")
    editable = state["template"]["scope"]["editable"]
    expected = inflight.get("editable_paths_sha256")
    snapshot = run_dir / "snapshots" / str(candidate_id)
    if not snapshot.exists():
        if not isinstance(expected, str) or _legacy().path_digest(root, editable) != expected:
            return False, _restore_with_quarantine(root, run_dir, state)
        _legacy().snapshot_paths(root, editable, snapshot)
    observed = _legacy().path_digest(snapshot, editable)
    if not isinstance(expected, str) or observed != expected:
        return False, _restore_with_quarantine(root, run_dir, state)
    paired = representative_result["replication"]["summary"]
    state["accepted_parent"] = {
        "id": candidate_id,
        "lane": decision.get("lane"),
        "params": decision["params"],
        "metric": float(representative_result["primary"]["value"]),
        "relative_mad": float(paired["mad"]) / float(paired["median"]),
        "paired_summary": paired,
        "tiers": accepted_tiers,
        "snapshot": candidate_id,
        "editable_paths_sha256": expected,
        "snapshot_sha256": observed,
    }
    lane = decision.get("lane")
    audit = decision.get("proxy_audit")
    proxy = state.get("proxy")
    if (
        isinstance(proxy, dict)
        and proxy.get("status") in {"enabled", "pending_calibration"}
        and (
            (isinstance(lane, dict) and lane.get("effective") == "representative_direct")
            or (isinstance(audit, dict) and audit.get("ranking_ok") is False)
        )
    ):
        proxy["status"] = "disabled"
        proxy["reason"] = "recovered candidate invalidated proxy ranking"
    state["status"] = "active"
    _legacy().restore_snapshot(root, editable, snapshot)
    return True, None


def _recorded_process_identity(
    run_dir: Path, inflight: dict[str, Any]
) -> Optional[dict[str, Any]]:
    tracking = inflight.get("process_tracking")
    if tracking is None:
        return None
    required = {"evaluation_id", "launch_token", "identity_path"}
    if not isinstance(tracking, dict) or set(tracking) != required:
        raise ValueError("inflight process tracking is invalid")
    if (
        tracking["evaluation_id"] != inflight.get("evaluation_id")
        or not isinstance(tracking["launch_token"], str)
        or not tracking["launch_token"]
    ):
        raise ValueError("inflight process tracking does not match the evaluation")
    relative = Path(tracking["identity_path"])
    expected = (
        Path("evaluations")
        / str(inflight["evaluation_id"])
        / "process-identity.json"
    )
    if relative.is_absolute() or ".." in relative.parts or relative != expected:
        raise ValueError("inflight process identity path is invalid")
    path = run_dir / relative
    deadline = time.monotonic() + 2.0
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not path.is_file():
        try:
            with evaluator_lease(
                {
                    "kind": "recovery_probe",
                    "evaluation_id": inflight["evaluation_id"],
                },
                timeout_seconds=0.0,
            ):
                return None
        except EvaluatorLeaseTimeout as error:
            raise RuntimeError(
                "tracked evaluator holds the lease but has not published its identity"
            ) from error
    identity = read_json(path)
    if (
        identity.get("schema_version") != 1
        or identity.get("evaluation_id") != tracking["evaluation_id"]
        or identity.get("launch_token") != tracking["launch_token"]
    ):
        raise ValueError("recorded evaluator identity does not match inflight state")
    return identity


def recover(root: Path, run_dir: Path) -> dict[str, Any]:
    inflight_path = run_dir / "inflight.json"
    if not inflight_path.is_file():
        raise ValueError("there is no interrupted evaluation")
    inflight = read_json(inflight_path)
    identity = _recorded_process_identity(run_dir, inflight)
    if identity is not None:
        stop_recorded_process_group(identity)
    with evaluator_lease(
        {
            "kind": "recovery",
            "evaluation_id": inflight["evaluation_id"],
        },
        timeout_seconds=30.0,
    ):
        return _recover_under_lease(root, run_dir)


def _recovery_execution_context_error(
    run_dir: Path,
    state: dict[str, Any],
    inflight: dict[str, Any],
    attempt: Optional[dict[str, Any]],
) -> Optional[str]:
    context = inflight.get("execution_context")
    runtime_artifact = _runtime_artifact_contract(
        state["template"], inflight["tier_id"]
    )
    context_required = runtime_artifact is not None
    if context_required and context is None:
        return "required runtime artifact context was not published"
    try:
        verify_runtime_artifact_context(
            run_dir,
            runtime_artifact["kind"] if runtime_artifact is not None else None,
            context,
        )
    except (OSError, ValueError) as error:
        return str(error)
    if attempt is not None and canonical_json(
        attempt.get("execution_context")
    ) != canonical_json(context):
        return "attempt and inflight artifact contexts differ"
    return None


def _recovery_sealed_binary_context_error(
    run_dir: Path,
    state: dict[str, Any],
    inflight: dict[str, Any],
    attempt: Optional[dict[str, Any]],
) -> Optional[str]:
    tier = _tier_by_id(state["template"], inflight["tier_id"])
    context = inflight.get("sealed_binary_context")
    expected_ids = _sealed_binary_ids_for_tier(state["template"], tier["id"])
    if context is None and expected_ids and inflight.get("process_tracking") is None:
        try:
            for binary_id in expected_ids:
                verify_sealed_binary_record(
                    run_dir, binary_id, state["sealed_binaries"][binary_id]
                )
        except (KeyError, OSError, ValueError) as error:
            return str(error)
        return None
    try:
        _verify_sealed_binary_context(run_dir, state, tier, context)
    except (OSError, ValueError) as error:
        return str(error)
    if attempt is not None and canonical_json(
        attempt.get("sealed_binary_context")
    ) != canonical_json(context):
        return "attempt and inflight sealed binary contexts differ"
    return None


def _recover_binary_build(
    root: Path,
    run_dir: Path,
    state: dict[str, Any],
    inflight: dict[str, Any],
) -> dict[str, Any]:
    binary_id = inflight.get("binary_id")
    contracts = state["template"].get("sealed_binaries", {})
    if not isinstance(binary_id, str) or binary_id not in contracts:
        raise ValueError("interrupted sealed binary build is invalid")
    evaluation_id = inflight["evaluation_id"]
    evaluation_dir = run_dir / "evaluations" / inflight["evaluation_id"]
    attempt_path = evaluation_dir / "attempt.json"

    terminal_events = [
        event
        for event in _events(run_dir / "binary-events.jsonl")
        if event.get("evaluation_id") == evaluation_id
    ]
    if len(terminal_events) > 1:
        raise ValueError("binary ledger contains a duplicate evaluation")
    if terminal_events:
        terminal = terminal_events[0]
        if (
            terminal.get("event")
            not in {"sealed_binary_built", "sealed_binary_build_recovered"}
            or terminal.get("binary_id") != binary_id
            or not isinstance(terminal.get("attempt"), dict)
            or not attempt_path.is_file()
            or canonical_json(read_json(attempt_path))
            != canonical_json(terminal["attempt"])
        ):
            raise ValueError("sealed binary terminal event is invalid")
        attempt = terminal["attempt"]
        record = attempt.get("sealed_binary")
        integrity_error = None
        if (attempt.get("outcome") == "success") != (record is not None):
            integrity_error = "sealed binary outcome and record disagree"
        elif record is not None:
            try:
                verify_sealed_binary_contract(
                    root, run_dir, binary_id, contracts[binary_id], record
                )
                prior = state.get("sealed_binaries", {}).get(binary_id)
                if prior is not None and canonical_json(prior) != canonical_json(record):
                    raise ValueError("sealed binary state and terminal event disagree")
            except (OSError, ValueError) as error:
                integrity_error = str(error)
        if integrity_error is None and record is not None:
            state["sealed_binaries"][binary_id] = record
        state["status"] = (
            "sealed_binary_invalid"
            if integrity_error is not None
            else "sealing_binaries_retryable"
        )
        write_state(run_dir, state)
        (run_dir / "inflight.json").unlink()
        return state

    if attempt_path.is_file():
        attempt = read_json(attempt_path)
        if "sealed_binary" not in attempt:
            attempt["evaluator_outcome"] = attempt.get("outcome")
            attempt["outcome"] = "interrupted"
            attempt["error"] = (
                "controller interrupted before sealing the binary build"
            )
            attempt["sealed_binary"] = None
    else:
        started = datetime.fromisoformat(
            inflight["started_at"].replace("Z", "+00:00")
        )
        elapsed = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
        attempt = {
            "schema_version": 1,
            "tier_id": f"sealed_binary:{binary_id}",
            "outcome": "interrupted",
            "error": "sealed binary build was interrupted before an attempt was sealed",
            "command": contracts[binary_id]["build"]["command"],
            "started_at": inflight["started_at"],
            "controller": {
                "queue_wait_seconds": 0.0,
                "exclusive_lease_seconds": elapsed,
                "subprocess_wall_seconds": elapsed,
            },
            "result_sha256": None,
            "process_tracking": inflight.get("process_tracking"),
            "sealed_binary": None,
        }
        evaluation_dir.mkdir(parents=True, exist_ok=True)
    _record_binary_build_gpu_usage(attempt)
    record = attempt.get("sealed_binary")
    integrity_error = None
    if record is not None:
        try:
            verify_sealed_binary_contract(
                root, run_dir, binary_id, contracts[binary_id], record
            )
            prior = state.get("sealed_binaries", {}).get(binary_id)
            if prior is not None and canonical_json(prior) != canonical_json(record):
                raise ValueError("sealed binary state and recovered attempt disagree")
        except (OSError, ValueError) as error:
            integrity_error = str(error)
            message = attempt.get("error")
            attempt["error"] = (
                f"{message}; {integrity_error}" if message else integrity_error
            )
    if integrity_error is None and record is not None:
        if attempt.get("outcome") != "success":
            integrity_error = "sealed binary outcome and recovered record disagree"
        else:
            state["sealed_binaries"][binary_id] = record
    elif record is None and attempt.get("outcome") == "success":
        attempt["evaluator_outcome"] = "success"
        attempt["outcome"] = "interrupted"
        attempt["error"] = "controller interrupted before sealing the binary output"
    attempt_path.write_bytes(canonical_json(attempt))
    append_event(
        run_dir / "binary-events.jsonl",
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event": "sealed_binary_build_recovered",
            "evaluation_id": inflight["evaluation_id"],
            "binary_id": binary_id,
            "attempt": attempt,
            "recorded_at": utc_now(),
        },
    )
    charge_attempt(state["usage"], attempt)
    _refresh_calendar(state)
    state["status"] = (
        "sealed_binary_invalid"
        if integrity_error is not None
        else "sealing_binaries_retryable"
    )
    write_state(run_dir, state)
    (run_dir / "inflight.json").unlink()
    return state


def _recover_pending_candidate(
    root: Path,
    run_dir: Path,
    state: dict[str, Any],
    inflight: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = inflight.get("candidate_id")
    phase_id = state["template"]["mechanism_phase"]["id"]
    if (
        not isinstance(candidate_id, str)
        or inflight.get("evaluation_id") != candidate_id
        or inflight.get("phase_id") != phase_id
        or not isinstance(inflight.get("params"), dict)
        or not isinstance(inflight.get("editable_paths_sha256"), str)
    ):
        raise ValueError("pending candidate transaction is invalid")
    quarantine = _restore_with_quarantine(root, run_dir, state)
    checkpoint = _phase_checkpoint_state(state)
    state["status"] = (
        "phase_exhausted" if checkpoint["status"] == "failed" else "active"
    )
    append_event(
        run_dir / "decision-events.jsonl",
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event": "pending_candidate_recovered",
            "candidate_id": candidate_id,
            "phase_id": phase_id,
            "quarantine": str(quarantine) if quarantine is not None else None,
            "recorded_at": utc_now(),
        },
    )
    _refresh_calendar(state)
    write_state(run_dir, state)
    (run_dir / "inflight.json").unlink()
    return state


def _recover_under_lease(root: Path, run_dir: Path) -> dict[str, Any]:
    state = load_state(run_dir, verify_sealed_binaries=False)
    inflight_path = run_dir / "inflight.json"
    if not inflight_path.is_file():
        raise ValueError("there is no interrupted evaluation")
    inflight = read_json(inflight_path)
    if inflight.get("kind") == "sealed_binary_build":
        return _recover_binary_build(root, run_dir, state, inflight)
    if inflight.get("kind") == "candidate_pending":
        return _recover_pending_candidate(root, run_dir, state, inflight)
    evaluation_dir = run_dir / "evaluations" / inflight["evaluation_id"]
    attempt_path = evaluation_dir / "attempt.json"
    inflight_context = inflight.get("execution_context")
    inflight_binary_context = inflight.get("sealed_binary_context")
    sealed_attempt = read_json(attempt_path) if attempt_path.is_file() else None
    context_error = _recovery_execution_context_error(
        run_dir, state, inflight, sealed_attempt
    )
    binary_context_error = _recovery_sealed_binary_context_error(
        run_dir, state, inflight, sealed_attempt
    )
    if sealed_attempt is not None:
        attempt = sealed_attempt
        charged_ids = {
            event.get("evaluation_id")
            for event in _events(run_dir / "tier-events.jsonl")
        }
        if (
            inflight["evaluation_id"] in charged_ids
            and (context_error is not None or binary_context_error is not None)
        ):
            errors = "; ".join(
                error
                for error in (context_error, binary_context_error)
                if error is not None
            )
            raise ValueError(
                f"sealed evaluation context is invalid: {errors}"
            )
        if inflight["evaluation_id"] not in charged_ids:
            attempt["evaluator_outcome"] = attempt.get("outcome")
            attempt["outcome"] = "interrupted"
            attempt["error"] = "controller interrupted before sealing the tier ledger"
            attempt["execution_context"] = inflight_context
            attempt["artifact_context_valid"] = context_error is None
            attempt["sealed_binary_context"] = inflight_binary_context
            attempt["binary_context_valid"] = binary_context_error is None
            if context_error is not None:
                attempt["error"] += f"; {context_error}"
            if binary_context_error is not None:
                attempt["error"] += f"; {binary_context_error}"
            attempt["budget_reserve"] = inflight.get("budget_reserve")
            charge_attempt(state["usage"], attempt)
            append_event(
                run_dir / "tier-events.jsonl",
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "event": "tier_recovered",
                    "evaluation_id": inflight["evaluation_id"],
                    "tier_id": inflight["tier_id"],
                    "attempt": attempt,
                    "recorded_at": utc_now(),
                },
            )
    else:
        started = datetime.fromisoformat(
            inflight["started_at"].replace("Z", "+00:00")
        )
        elapsed = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
        attempt = {
            "schema_version": 1,
            "tier_id": inflight["tier_id"],
            "outcome": "interrupted",
            "error": "evaluation was interrupted before an attempt record was sealed",
            "command": [],
            "started_at": inflight["started_at"],
            "controller": {
                "queue_wait_seconds": 0.0,
                "exclusive_lease_seconds": elapsed,
                "subprocess_wall_seconds": elapsed,
            },
            "resources": {
                "gpu_active_seconds": None,
                "gpu_active_charge_seconds": elapsed,
                "gpu_active_charge_kind": "conservative_wall_upper_bound",
            },
            "result_sha256": None,
            "execution_context": inflight_context,
            "artifact_context_valid": context_error is None,
            "sealed_binary_context": inflight_binary_context,
            "binary_context_valid": binary_context_error is None,
            "budget_reserve": inflight.get("budget_reserve"),
        }
        if context_error is not None:
            attempt["error"] += f"; {context_error}"
        if binary_context_error is not None:
            attempt["error"] += f"; {binary_context_error}"
        charge_attempt(state["usage"], attempt)
        append_event(
            run_dir / "tier-events.jsonl",
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event": "tier_recovered",
                "evaluation_id": inflight["evaluation_id"],
                "tier_id": inflight["tier_id"],
                "attempt": attempt,
                "recorded_at": utc_now(),
            },
        )
    recovered_keep = False
    quarantine = None
    if inflight["kind"] == "candidate" and state["accepted_parent"] is not None:
        recovered_keep, quarantine = _recover_committed_candidate(
            root, run_dir, state, inflight
        )
    elif state["accepted_parent"] is not None:
        quarantine = _restore_with_quarantine(root, run_dir, state)
        kind = inflight["kind"]
        if kind == "revalidation":
            validation = _validation_for_evaluation(
                run_dir, inflight["evaluation_id"]
            )
            if validation is not None and validation.get("role") != (
                "representative_revalidation"
            ):
                raise ValueError(
                    "validation ledger role does not match interrupted evaluation"
                )
            validation_status = (
                validation.get("status") if validation is not None else None
            )
            if validation_status == "accepted":
                _sealed_validation_result(run_dir, state, validation)
                state["status"] = "holdout_retryable"
            elif validation_status in {None, "invalid", "rejected"}:
                state["status"] = "active"
            else:
                raise ValueError("validation ledger has an invalid terminal status")
        elif kind in {"holdout", "transfer"}:
            validation = _validation_for_evaluation(
                run_dir, inflight["evaluation_id"]
            )
            if validation is not None and validation.get("role") != kind:
                raise ValueError(
                    "validation ledger role does not match interrupted evaluation"
                )
            validation_status = (
                validation.get("status") if validation is not None else None
            )
            if validation_status in {"accepted", "rejected"}:
                _sealed_validation_result(run_dir, state, validation)
            elif validation_status not in {None, "invalid"}:
                raise ValueError("validation ledger has an invalid terminal status")
            if kind == "holdout":
                state["status"] = {
                    "accepted": "kernel_accepted",
                    "rejected": "holdout_rejected",
                    "invalid": "holdout_retryable",
                    None: "holdout_retryable",
                }[validation_status]
            else:
                state["status"] = (
                    "kernel_transferred"
                    if validation_status == "accepted"
                    else "kernel_accepted"
                )
    elif inflight["kind"] == "baseline":
        state["status"] = "initialization_retryable"
    if binary_context_error is not None:
        state["status"] = "sealed_binary_invalid"
    append_event(
        run_dir / "decision-events.jsonl",
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event": "interrupted_evaluation_recovered",
            "evaluation_id": inflight["evaluation_id"],
            "recovered_committed_keep": recovered_keep,
            "quarantine": str(quarantine) if quarantine is not None else None,
            "recorded_at": utc_now(),
        },
    )
    _refresh_calendar(state)
    write_state(run_dir, state)
    inflight_path.unlink()
    return state


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def status(run_dir: Path) -> dict[str, Any]:
    state = load_state(run_dir)
    _refresh_calendar(state)
    budget = state["template"]["budget"]
    total = budget["total"]
    usage = state["usage"]
    available = {
        "calendar_seconds": max(
            0.0, float(total["max_calendar_seconds"]) - float(usage["calendar_seconds"])
        ),
        "active_evaluator_seconds": max(
            0.0,
            float(total["max_active_evaluator_seconds"])
            - float(usage["active_evaluator_seconds"]),
        ),
        "exclusive_machine_seconds": max(
            0.0,
            float(total["max_exclusive_machine_seconds"])
            - float(usage["exclusive_machine_seconds"]),
        ),
        "gpu_active_seconds": max(
            0.0,
            float(total["max_gpu_active_seconds"])
            - float(usage["gpu_active_seconds"]),
        ),
        "candidates": max(
            0,
            int(total["max_candidates_admitted"])
            - int(usage["candidates_admitted"]),
        ),
    }
    return {
        "status": state["status"],
        "slot_id": state["registry_binding"]["slot_id"],
        "accepted_parent": state["accepted_parent"],
        "usage": usage,
        "reserves": budget["reserves"],
        "available": available,
        "inflight": (run_dir / "inflight.json").is_file(),
    }


def candidate_context(run_dir: Path) -> dict[str, Any]:
    state = load_state(run_dir)
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "slot_id": state["registry_binding"]["slot_id"],
        "base_revision": state["base_revision"],
        "accepted_parent": state["accepted_parent"],
        "editable": state["template"]["scope"]["editable"],
        "search_space": state["template"]["search_space"],
        "budget": status(run_dir),
    }


def parse_goal_candidate(value: str) -> dict[str, Any]:
    parts = value.rsplit(":", 2)
    if len(parts) != 3 or not parts[0]:
        raise ValueError("goal candidates use KERNEL:CURRENT_PIOP_SHARE:LOCAL_SPEEDUP")
    return {
        "kernel": parts[0],
        "current_piop_share": float(parts[1]),
        "conservative_local_speedup": float(parts[2]),
    }


def goal_decision(
    contract: dict[str, Any],
    current_speedup: float,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_goal_contract(contract)
    if not math.isfinite(current_speedup) or current_speedup <= 0:
        raise ValueError("current PIOP speedup must be finite and positive")
    floor = float(contract["primary_metric"]["minimum_accepted_speedup"])
    minimum_gain = float(
        contract["continuation"]["minimum_projected_relative_gain_after_target"]
    )
    headroom_trigger = float(contract["continuation"]["analytical_headroom_trigger"])
    projected_time = 1.0
    total_share = 0.0
    ranked = []
    for candidate in candidates:
        kernel = str(candidate["kernel"])
        share = float(candidate["current_piop_share"])
        local_speedup = float(candidate["conservative_local_speedup"])
        if not math.isfinite(share) or not 0 <= share <= 1:
            raise ValueError(f"{kernel} has an invalid PIOP share")
        if not math.isfinite(local_speedup) or local_speedup < 1:
            raise ValueError(f"{kernel} has an invalid conservative local speedup")
        total_share += share
        saved = share * (1.0 - 1.0 / local_speedup)
        projected_time -= saved
        ranked.append(
            {
                **candidate,
                "projected_time_fraction_saved": saved,
            }
        )
    if total_share > 1.0 + 1e-12:
        raise ValueError("candidate PIOP shares overlap or sum above one")
    projected_speedup = current_speedup / projected_time
    projected_gain = projected_speedup / current_speedup - 1.0
    floor_met = current_speedup >= floor
    clear_headroom = any(
        float(candidate["conservative_local_speedup"]) > headroom_trigger
        for candidate in ranked
    )
    should_continue = not floor_met or projected_gain >= minimum_gain or clear_headroom
    ranked.sort(key=lambda item: item["projected_time_fraction_saved"], reverse=True)
    return {
        "continue": should_continue,
        "floor_met": floor_met,
        "current_piop_speedup": current_speedup,
        "minimum_accepted_speedup": floor,
        "projected_piop_speedup": projected_speedup,
        "projected_relative_gain": projected_gain,
        "minimum_projected_relative_gain_after_target": minimum_gain,
        "analytical_headroom_trigger": headroom_trigger,
        "clear_headroom": clear_headroom,
        "next_kernel": ranked[0]["kernel"] if ranked else None,
        "candidates": ranked,
    }


def record_goal_decision(
    root: Path,
    run_dir: Path,
    candidates: list[dict[str, Any]],
    shares_disjoint: bool,
) -> dict[str, Any]:
    root = root.resolve()
    state = load_state(run_dir)
    _validate_live_state(root, state)
    _assert_frozen(root, state)
    if state["status"] != "kernel_transferred":
        raise ValueError("goal decisions require a transferred kernel run")
    if (run_dir / "inflight.json").exists():
        raise ValueError("an interrupted evaluation must be recovered first")
    if candidates and not shares_disjoint:
        raise ValueError("portfolio candidates require disjoint share attestation")
    accepted = state["accepted_parent"]
    revision = _legacy().git_head(root)
    holdout = _accepted_validation(
        run_dir, state, "holdout", accepted["id"], revision
    )
    transfer = _accepted_validation(
        run_dir, state, "transfer", accepted["id"], revision
    )
    holdout_result = _sealed_validation_result(run_dir, state, holdout)
    transfer_result = _sealed_validation_result(run_dir, state, transfer)
    current_speedup = min(
        _conservative_result_speedup(holdout_result),
        _conservative_result_speedup(transfer_result),
    )
    decision = goal_decision(state["goal"], current_speedup, candidates)
    evidence = {
        "accepted_parent": accepted["id"],
        "snapshot_sha256": accepted["snapshot_sha256"],
        "revision": revision,
        "goal_sha256": sha256(canonical_json(state["goal"])),
        "template_sha256": state["template_sha256"],
        "holdout_evaluation_id": holdout["evaluation_id"],
        "holdout_result_sha256": holdout.get("tier_result_sha256"),
        "transfer_evaluation_id": transfer["evaluation_id"],
        "transfer_result_sha256": transfer.get("tier_result_sha256"),
    }
    decision_key = sha256(
        canonical_json(
            {
                "evidence": evidence,
                "shares_disjoint": shares_disjoint,
                "candidates": candidates,
            }
        )
    )
    prior = [
        event
        for event in _events(run_dir / "decision-events.jsonl")
        if event.get("event") == "portfolio_goal_decided"
    ]
    for event in prior:
        if event.get("decision_key") == decision_key:
            return event["decision"]
    decision_id = f"goal-decision-{len(prior) + 1:03d}"
    event = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event": "portfolio_goal_decided",
        "decision_id": decision_id,
        "supersedes": prior[-1]["decision_id"] if prior else None,
        "decision_key": decision_key,
        "evidence": evidence,
        "shares_disjoint": shares_disjoint,
        "candidates": candidates,
        "decision": decision,
        "successor": {
            "required": decision["continue"],
            "kernel": decision["next_kernel"],
            "run_id": (
                f"successor-{decision_id}-{decision_key[:12]}"
                if decision["continue"]
                else None
            ),
        },
        "recorded_at": utc_now(),
    }
    append_event(run_dir / "decision-events.jsonl", event)
    return decision


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run versioned Metal kernel research")
    result.add_argument("--root", default=Path(__file__).resolve().parents[2])
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("template")
    init.add_argument("run_dir")
    validate = commands.add_parser("validate-template")
    validate.add_argument("template")
    profile = commands.add_parser("profile-iteration")
    profile.add_argument("template")
    profile.add_argument("output_prefix")
    resume = commands.add_parser("resume-init")
    resume.add_argument("run_dir")
    calibrate = commands.add_parser("calibrate-proxy")
    calibrate.add_argument("run_dir")
    trial_parser = commands.add_parser("trial")
    trial_parser.add_argument("run_dir")
    trial_parser.add_argument("--param", action="append", default=[])
    trial_parser.add_argument("--summary", required=True)
    trial_parser.add_argument("--direct-to-representative", action="store_true")
    trial_parser.add_argument("--direct-reason")
    context = commands.add_parser("candidate-context")
    context.add_argument("run_dir")
    state = commands.add_parser("status")
    state.add_argument("run_dir")
    production = commands.add_parser("validate-production")
    production.add_argument("run_dir")
    recovery = commands.add_parser("recover")
    recovery.add_argument("run_dir")
    goal_prompt = commands.add_parser("goal-prompt")
    goal_prompt.add_argument("contract")
    goal = commands.add_parser("goal-decision")
    goal.add_argument("contract")
    goal.add_argument("--run-dir", required=True)
    goal.add_argument("--candidate", action="append", default=[])
    goal.add_argument("--shares-disjoint", action="store_true")
    return result


def main(argv: Optional[list[str]] = None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.root).resolve()
    try:
        if args.command == "init":
            if not _legacy().git_worktree_clean(root):
                raise ValueError(
                    "v2 initialization requires a clean worktree"
                )
            value = init_run(root, Path(args.template), Path(args.run_dir).resolve())
        elif args.command == "validate-template":
            value = validate_template_file(root, Path(args.template))
        elif args.command == "profile-iteration":
            from .iteration_profile import generate_iteration_profile

            template = read_json(Path(args.template))
            validate_template(
                template,
                root,
                verify_iteration_profile=False,
            )
            value = generate_iteration_profile(
                root,
                template,
                Path(args.output_prefix),
                _validate_closed_result,
            )
        elif args.command == "resume-init":
            value = resume_initialization(root, Path(args.run_dir).resolve())
        elif args.command == "calibrate-proxy":
            value, _ = calibrate_proxy(root, Path(args.run_dir).resolve())
        elif args.command == "trial":
            value, _ = trial(
                root,
                Path(args.run_dir).resolve(),
                args.param,
                args.summary,
                direct_to_representative=args.direct_to_representative,
                direct_reason=args.direct_reason,
            )
        elif args.command == "candidate-context":
            value = candidate_context(Path(args.run_dir).resolve())
        elif args.command == "status":
            value = status(Path(args.run_dir).resolve())
        elif args.command == "validate-production":
            value, _ = validate_production(root, Path(args.run_dir).resolve())
        elif args.command == "recover":
            value = recover(root, Path(args.run_dir).resolve())
        elif args.command == "goal-prompt":
            contract = read_json(Path(args.contract))
            validate_goal_contract(contract)
            print(f"/goal {contract['goal_prompt']}")
            return 0
        else:
            contract = read_json(Path(args.contract))
            state = load_state(Path(args.run_dir).resolve())
            if canonical_json(contract) != canonical_json(state["goal"]):
                raise ValueError("goal decision contract does not match the sealed run")
            candidates = [parse_goal_candidate(value) for value in args.candidate]
            if candidates and not args.shares_disjoint:
                raise ValueError("portfolio candidates require --shares-disjoint")
            value = record_goal_decision(
                root,
                Path(args.run_dir).resolve(),
                candidates,
                args.shares_disjoint,
            )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except (
        BudgetExhausted,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
