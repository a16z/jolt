from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .attempt import run_attempt
from .budget import BudgetExhausted, admit_tier, charge_attempt, empty_usage
from .contracts import validate_goal_contract, validate_template
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
    temporary = run_dir / ".run.json.tmp"
    temporary.write_bytes(encoded)
    temporary.replace(run_dir / "run.json")
    state["state_sha256"] = payload["state_sha256"]


def write_inflight(run_dir: Path, inflight: dict[str, Any]) -> None:
    temporary = run_dir / ".inflight.json.tmp"
    temporary.write_bytes(canonical_json(inflight))
    temporary.replace(run_dir / "inflight.json")


def load_state(run_dir: Path) -> dict[str, Any]:
    encoded = (run_dir / "run.json").read_bytes()
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
    state["usage"] = _derived_usage(run_dir)
    parent = state.get("accepted_parent")
    if parent is not None:
        snapshot = run_dir / "snapshots" / parent["snapshot"]
        observed = _legacy().path_digest(
            snapshot, state["template"]["scope"]["editable"]
        )
        if observed != parent.get("snapshot_sha256"):
            raise ValueError("accepted parent snapshot digest does not match")
    return state


def _derived_usage(run_dir: Path) -> dict[str, float | int]:
    usage = empty_usage()
    seen_evaluations: set[str] = set()
    for event in _events(run_dir / "tier-events.jsonl"):
        evaluation_id = event.get("evaluation_id")
        attempt = event.get("attempt")
        if not isinstance(evaluation_id, str) or not isinstance(attempt, dict):
            raise ValueError("tier ledger contains an invalid attempt record")
        if evaluation_id in seen_evaluations:
            raise ValueError("tier ledger contains a duplicate evaluation")
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


def _require_calendar_budget(state: dict[str, Any]) -> None:
    _refresh_calendar(state)
    maximum = float(state["template"]["budget"]["total"]["max_calendar_seconds"])
    if float(state["usage"]["calendar_seconds"]) > maximum:
        raise BudgetExhausted("calendar budget is exhausted")


def _queue_budget(state: dict[str, Any], tier: dict[str, Any]) -> float:
    _require_calendar_budget(state)
    maximum = float(state["template"]["budget"]["total"]["max_calendar_seconds"])
    remaining = maximum - float(state["usage"]["calendar_seconds"])
    evaluator_timeout = float(tier["evaluator"]["timeout_seconds"])
    if evaluator_timeout > remaining:
        raise BudgetExhausted("calendar budget cannot contain this evaluator")
    return remaining - evaluator_timeout


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
    validate_template(template, root)
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
    return {
        "metric": float(result["primary"]["value"]),
        "relative_mad": float(summary["mad"]) / float(summary["median"]),
        "paired_summary": summary,
    }


def _promotion_pass(
    tier: dict[str, Any],
    result: dict[str, Any],
    parent: dict[str, Any],
) -> tuple[bool, str]:
    kind = tier["promotion"]["kind"]
    if kind == "all_guards":
        return True, "all correctness guards passed"
    if kind != "relative_improvement":
        raise ValueError(f"unsupported search-tier promotion kind: {kind}")
    metric = float(result["primary"]["value"])
    current_noise = float(result["replication"]["summary"]["mad"]) / metric
    threshold = max(
        float(tier["promotion"]["minimum_relative_improvement"]),
        float(tier["promotion"]["noise_multiplier"])
        * max(float(parent["relative_mad"]), current_noise),
    )
    if metric >= float(parent["metric"]) * (1.0 + threshold):
        return True, "improves beyond the noise-qualified threshold"
    return False, "does not improve beyond the noise-qualified threshold"


def _validate_closed_result(
    root: Path,
    tier: dict[str, Any],
    output: dict[str, Any],
    params: dict[str, str],
) -> None:
    adapter = tier["evaluator"]["result_adapter"]
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
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    queue_budget = _queue_budget(state, tier)
    admit_tier(
        state["template"]["budget"],
        state["usage"],
        tier["cost_limit"],
        budget_reserve,
    )
    evaluation_dir = run_dir / "evaluations" / evaluation_id
    attempt, output = run_attempt(
        root,
        tier["evaluator"],
        params,
        evaluation_dir,
        tier["id"],
        queue_timeout_seconds=queue_budget,
    )
    result = None
    if attempt["outcome"] == "success" and output is not None:
        try:
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
    for name in (
        "baseline-events.jsonl",
        "candidate-events.jsonl",
        "tier-events.jsonl",
        "decision-events.jsonl",
        "portfolio-validations.jsonl",
    ):
        (run_dir / name).touch()


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

    _initialize_run_files(run_dir)
    legacy = _legacy()
    legacy.snapshot_paths(
        root, template["scope"]["editable"], run_dir / "snapshots" / "baseline"
    )
    fingerprint = _scope_fingerprint(root, template)
    state = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "initializing",
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
    }
    write_state(run_dir, state)
    params = _params(template, [])
    tier_records = {}
    for tier in _search_tiers(template):
        inflight = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "kind": "baseline",
            "evaluation_id": f"baseline-{tier['id']}",
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
            state["status"] = "initialization_failed"
            write_state(run_dir, state)
            (run_dir / "inflight.json").unlink()
            raise ValueError(
                f"{tier['id']} baseline evaluator failed: {event['attempt']['error']}"
            )
        passed, reason = _required_guards(state, tier, result)
        if not passed:
            state["status"] = "initialization_failed"
            write_state(run_dir, state)
            (run_dir / "inflight.json").unlink()
            raise ValueError(f"{tier['id']} baseline evaluator is invalid: {reason}")
        record = _tier_record(result)
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
    (run_dir / "inflight.json").unlink()
    return state


def _candidate_id(state: dict[str, Any]) -> str:
    return f"candidate-{int(state['usage']['candidates_admitted']) + 1:03d}"


def _validation_id(run_dir: Path) -> str:
    index = 1
    while any((run_dir / "evaluations").glob(f"validation-{index:03d}-*")):
        index += 1
    return f"validation-{index:03d}"


def _accepted_validation(
    run_dir: Path,
    role: str,
    parent_id: str,
    revision: str,
) -> dict[str, Any]:
    matches = [
        event
        for event in _events(run_dir / "portfolio-validations.jsonl")
        if event.get("role") == role
        and event.get("accepted_parent") == parent_id
        and event.get("revision") == revision
        and event.get("status") == "accepted"
    ]
    if not matches:
        raise ValueError(f"run status has no accepted {role} evidence")
    return matches[-1]


def trial(
    root: Path,
    run_dir: Path,
    overrides: list[str],
    summary: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    state = load_state(run_dir)
    _validate_live_state(root, state)
    if state["status"] not in {"active", "portfolio_accepted", "transferred"}:
        raise ValueError("run is not active")
    if (run_dir / "inflight.json").exists():
        raise ValueError("an interrupted evaluation must be recovered first")
    _assert_frozen(root, state)
    _require_calendar_budget(state)
    maximum = state["template"]["budget"]["total"]["max_candidates_admitted"]
    if int(state["usage"]["candidates_admitted"]) >= int(maximum):
        raise BudgetExhausted("candidate budget is exhausted")
    params = dict(state["accepted_parent"]["params"])
    for override in overrides:
        if "=" not in override:
            raise ValueError("parameter overrides must use NAME=VALUE")
        name, value = override.split("=", 1)
        params[name] = value
    params = _validate_params(state["template"], params)
    candidate_id = _candidate_id(state)
    editable_digest = _legacy().path_digest(
        root, state["template"]["scope"]["editable"]
    )
    state["usage"]["candidates_admitted"] = int(
        state["usage"]["candidates_admitted"]
    ) + 1
    append_event(
        run_dir / "candidate-events.jsonl",
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event": "candidate_admitted",
            "candidate_id": candidate_id,
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
    evaluation_ids = []
    representative_result = None
    for tier in _search_tiers(state["template"]):
        evaluation_id = f"{candidate_id}-{tier['id']}"
        evaluation_ids.append(evaluation_id)
        inflight = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "kind": "candidate",
            "candidate_id": candidate_id,
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
        write_state(run_dir, state)
        _assert_frozen(root, state)
        if result is None:
            verdict = "invalid"
            reason = event["attempt"]["error"] or "evaluator result is invalid"
            break
        passed, reason = _required_guards(state, tier, result)
        if not passed:
            verdict = "invalid"
            break
        record = _tier_record(result)
        tier_results[tier["id"]] = record
        promoted, reason = _promotion_pass(
            tier, result, parent["tiers"][tier["id"]]
        )
        if not promoted:
            verdict = "discard"
            break
        if tier["role"] == "representative":
            representative_result = result

    decision = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event": "candidate_decided",
        "candidate_id": candidate_id,
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
        "tier_results": tier_results,
        "recorded_at": utc_now(),
    }
    append_event(run_dir / "decision-events.jsonl", decision)
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
            "params": params,
            "metric": float(representative_result["primary"]["value"]),
            "relative_mad": float(paired["mad"]) / float(paired["median"]),
            "paired_summary": paired,
            "tiers": tier_results,
            "snapshot": candidate_id,
            "editable_paths_sha256": editable_digest,
            "snapshot_sha256": snapshot_digest,
        }
        state["status"] = "active"
    else:
        _legacy().restore_snapshot(
            root,
            state["template"]["scope"]["editable"],
            run_dir / "snapshots" / parent["snapshot"],
        )
    _refresh_calendar(state)
    write_state(run_dir, state)
    (run_dir / "inflight.json").unlink()
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


def _acceptance_result(
    state: dict[str, Any],
    tier: dict[str, Any],
    result: Optional[dict[str, Any]],
    error: Optional[str],
) -> tuple[bool, str]:
    if result is None:
        return False, error or f"{tier['role']} evaluator is invalid"
    passed, reason = _required_guards(state, tier, result)
    floor = float(tier["promotion"]["minimum_accepted_speedup"])
    if passed and not _result_clears_floor(result, floor):
        return False, f"PIOP {tier['role']} did not clear 5x in both order strata"
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
        "event": "portfolio_validated",
        "evaluation_id": evaluation_id,
        "tier_id": tier["id"],
        "role": tier["role"],
        "accepted_parent": accepted["id"],
        "revision": revision,
        "status": "accepted" if passed else "rejected",
        "reason": reason,
        "primary": result["primary"] if result is not None else None,
        "paired_summary": (
            result["replication"]["summary"] if result is not None else None
        ),
        "local": result.get("local") if result is not None else None,
        "recorded_at": utc_now(),
    }
    append_event(run_dir / "portfolio-validations.jsonl", record)
    state["status"] = accepted_status if passed else rejected_status
    _refresh_calendar(state)
    write_state(run_dir, state)
    return record, passed, reason


def validate_production(
    root: Path, run_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    state = load_state(run_dir)
    _validate_live_state(root, state)
    if state["status"] not in {"active", "portfolio_accepted", "transferred"}:
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
    if state["status"] == "transferred":
        return (
            _accepted_validation(
                run_dir, "transfer", accepted["id"], live_revision
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
        if not revalidation_passed:
            record = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event": "portfolio_validated",
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
            append_event(run_dir / "portfolio-validations.jsonl", record)
            _refresh_calendar(state)
            write_state(run_dir, state)
            (run_dir / "inflight.json").unlink()
            raise ValueError(revalidation_reason)

        append_event(
            run_dir / "portfolio-validations.jsonl",
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event": "portfolio_validated",
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
                "recorded_at": utc_now(),
            },
        )

        _, passed, reason = _run_acceptance_tier(
            root,
            run_dir,
            state,
            tier,
            accepted,
            live_revision,
            validation_id,
            "piop_holdout",
            "portfolio_accepted",
            "active",
        )
        if not passed:
            (run_dir / "inflight.json").unlink()
            raise ValueError(reason)
    else:
        _accepted_validation(run_dir, "holdout", accepted["id"], live_revision)
    transfer_record, transferred, transfer_reason = _run_acceptance_tier(
        root,
        run_dir,
        state,
        transfer,
        accepted,
        live_revision,
        validation_id,
        "piop_transfer",
        "transferred",
        "portfolio_accepted",
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
    if not decisions or decisions[0].get("verdict") != "keep":
        return False, _restore_with_quarantine(root, run_dir, state)

    decision = decisions[0]
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
    paired = decision["paired_summary"]
    state["accepted_parent"] = {
        "id": candidate_id,
        "params": decision["params"],
        "metric": float(decision["primary"]["value"]),
        "relative_mad": float(paired["mad"]) / float(paired["median"]),
        "paired_summary": paired,
        "tiers": decision["tier_results"],
        "snapshot": candidate_id,
        "editable_paths_sha256": expected,
        "snapshot_sha256": observed,
    }
    state["status"] = "active"
    _legacy().restore_snapshot(root, editable, snapshot)
    return True, None


def recover(root: Path, run_dir: Path) -> dict[str, Any]:
    state = load_state(run_dir)
    inflight_path = run_dir / "inflight.json"
    if not inflight_path.is_file():
        raise ValueError("there is no interrupted evaluation")
    inflight = read_json(inflight_path)
    evaluation_dir = run_dir / "evaluations" / inflight["evaluation_id"]
    attempt_path = evaluation_dir / "attempt.json"
    if attempt_path.is_file():
        attempt = read_json(attempt_path)
        charged_ids = {
            event.get("evaluation_id")
            for event in _events(run_dir / "tier-events.jsonl")
        }
        if inflight["evaluation_id"] not in charged_ids:
            attempt["evaluator_outcome"] = attempt.get("outcome")
            attempt["outcome"] = "interrupted"
            attempt["error"] = "controller interrupted before sealing the tier ledger"
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
            "budget_reserve": inflight.get("budget_reserve"),
        }
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
    elif inflight["kind"] == "baseline":
        state["status"] = "initialization_failed"
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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run versioned Metal kernel research")
    result.add_argument("--root", default=Path(__file__).resolve().parents[2])
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("template")
    init.add_argument("run_dir")
    trial_parser = commands.add_parser("trial")
    trial_parser.add_argument("run_dir")
    trial_parser.add_argument("--param", action="append", default=[])
    trial_parser.add_argument("--summary", required=True)
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
    goal.add_argument("--current-speedup", required=True, type=float)
    goal.add_argument("--candidate", action="append", default=[])
    goal.add_argument("--shares-disjoint", action="store_true")
    return result


def main(argv: Optional[list[str]] = None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.root).resolve()
    try:
        if args.command == "init":
            value = init_run(root, Path(args.template), Path(args.run_dir).resolve())
        elif args.command == "trial":
            value, _ = trial(
                root,
                Path(args.run_dir).resolve(),
                args.param,
                args.summary,
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
            candidates = [parse_goal_candidate(value) for value in args.candidate]
            if candidates and not args.shares_disjoint:
                raise ValueError("portfolio candidates require --shares-disjoint")
            value = goal_decision(contract, args.current_speedup, candidates)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except (BudgetExhausted, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
