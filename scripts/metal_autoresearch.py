#!/usr/bin/env python3
"""Run bounded Metal kernel experiments with snapshots and a durable ledger."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = 1
VERDICTS = {"keep", "discard", "crash", "invalid"}
CANDIDATE_STATUSES = {"queued", "accepted_parent", "promoted", "rejected"}
EVALUATOR_LOCK_PATH = Path("/private/tmp/jolt-metal-autoresearch-evaluator.lock")
EVALUATOR_LOCK_HELD_ENV = "JOLT_METAL_EVAL_LOCK_HELD"
CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
COMMON_PRODUCTION_GUARDS = frozenset(
    {
        "cpu_proofs_verified",
        "metal_proofs_verified",
        "target_scale",
        "production_contract",
        "local_kernel_attributed",
        "local_kernel_metal_backend_exercised",
        "stable_source",
        "stable_binary",
    }
)
PRODUCTION_LOCAL_KERNELS = {
    "InstructionRaVirtualization": {
        "metric": "instruction_ra_speedup",
        "paired_metric": "paired_instruction_ra_speedups",
        "parameters": frozenset(
            {
                "JOLT_METAL_INSTRUCTION_RA_MATERIALIZE_WIDTH",
                "JOLT_METAL_INSTRUCTION_RA_REUSE_INVERSE",
            }
        ),
        "required_guards": COMMON_PRODUCTION_GUARDS,
    },
    "BytecodeReadRafCycle": {
        "metric": "bytecode_read_raf_cycle_speedup",
        "paired_metric": "paired_bytecode_read_raf_cycle_speedups",
        "parameters": frozenset(
            {
                "JOLT_METAL_BYTECODE_MESSAGE_THREADS",
                "JOLT_METAL_BYTECODE_TRANSITION_THREADS",
                "JOLT_METAL_BYTECODE_MAX_THREADGROUPS",
                "JOLT_METAL_BYTECODE_CUTOFF_LOG2",
                "JOLT_METAL_BYTECODE_TRACE_CUTOFF_LOG2",
            }
        ),
        "required_guards": COMMON_PRODUCTION_GUARDS
        | {
            "bytecode_q10_cpu_control",
            "bytecode_metal_backend_exercised",
            "bytecode_working_set_admitted",
            "bytecode_readback_exact",
            "bytecode_local_gate",
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def expand_paths(root: Path, paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for relative in paths:
        path = root / relative
        if path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"contract path does not exist: {relative}")
    return sorted(set(files))


def path_digest(root: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in expand_paths(root, paths):
        relative = path.relative_to(root)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def path_is_in_scope(relative: str, scope: list[str]) -> bool:
    path = Path(relative)
    return any(path == Path(item) or Path(item) in path.parents for item in scope)


def outside_editable_worktree_digest(root: Path, editable: list[str]) -> str:
    tracked = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", "HEAD", "--"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    changed = {
        os.fsdecode(raw)
        for raw in [*tracked.split(b"\0"), *untracked.split(b"\0")]
        if raw
    }
    digest = hashlib.sha256()
    for relative in sorted(changed):
        if path_is_in_scope(relative, editable):
            continue
        path = root / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode())
        elif path.is_file():
            digest.update(f"mode:{path.stat().st_mode & 0o777:o}\0".encode())
            digest.update(path.read_bytes())
        else:
            digest.update(b"missing")
        digest.update(b"\0")
    return digest.hexdigest()


@contextmanager
def evaluator_lock(owner: dict[str, Any]):
    """Serialize every controller-launched compile and GPU/CPU evaluator."""
    inherited_token = os.environ.get(EVALUATOR_LOCK_HELD_ENV)
    if inherited_token:
        try:
            record = read_json(EVALUATOR_LOCK_PATH)
        except (OSError, ValueError, json.JSONDecodeError):
            record = {}
        if secrets.compare_digest(str(record.get("token", "")), inherited_token):
            yield
            return
    descriptor = os.open(EVALUATOR_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    previous_marker = os.environ.get(EVALUATOR_LOCK_HELD_ENV)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        token = secrets.token_hex(32)
        os.environ[EVALUATOR_LOCK_HELD_ENV] = token
        os.ftruncate(descriptor, 0)
        lock_record = {
            **owner,
            "pid": os.getpid(),
            "locked_at": utc_now(),
            "token": token,
        }
        os.write(descriptor, canonical_json(lock_record))
        os.fsync(descriptor)
        yield
    finally:
        if previous_marker is None:
            os.environ.pop(EVALUATOR_LOCK_HELD_ENV, None)
        else:
            os.environ[EVALUATOR_LOCK_HELD_ENV] = previous_marker
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def snapshot_paths(root: Path, paths: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for source in expand_paths(root, paths):
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def restore_snapshot(root: Path, paths: list[str], snapshot: Path) -> None:
    for target in expand_paths(root, paths):
        source = snapshot / target.relative_to(root)
        if not source.is_file():
            raise ValueError(f"snapshot is missing {target.relative_to(root)}")
        shutil.copy2(source, target)


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_worktree_clean(root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return not result.stdout


def parse_schema_result(stdout: str, schema_version: int) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema_version") == schema_version:
            return value
    raise ValueError(
        f"evaluator stdout contains no schema-version {schema_version} JSON object"
    )


def parse_unique_schema_result(stdout: str, schema_version: int) -> dict[str, Any]:
    matches = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema_version") == schema_version:
            matches.append(value)
    if len(matches) != 1:
        raise ValueError(
            f"evaluator stdout must contain exactly one schema-version {schema_version} JSON object"
        )
    return matches[0]


def parse_result(stdout: str) -> dict[str, Any]:
    return parse_schema_result(stdout, SCHEMA_VERSION)


def validate_template(template: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "kernel",
        "goal",
        "hypothesis",
        "metric",
        "portfolio_contract",
        "guards",
        "evaluator",
        "scope",
        "budget",
        "search_space",
        "baseline_repeats",
        "candidate_repeats",
        "stopping_conditions",
        "final_validation",
    }
    missing = sorted(required - template.keys())
    if missing:
        raise ValueError(f"template is missing fields: {missing}")
    if template["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported template schema")
    if template["metric"]["direction"] not in {"min", "max"}:
        raise ValueError("metric direction must be min or max")
    if template["baseline_repeats"] < 3:
        raise ValueError("baseline_repeats must be at least three")
    if template["baseline_repeats"] % 2 == 0:
        raise ValueError("baseline_repeats must be odd")
    candidate_repeats = template["candidate_repeats"]
    if candidate_repeats < 1 or candidate_repeats % 2 == 0:
        raise ValueError("candidate_repeats must be a positive odd integer")
    if template["budget"]["max_trials"] < 1:
        raise ValueError("max_trials must be positive")
    editable = set(template["scope"]["editable"])
    frozen = set(template["scope"]["frozen"])
    overlap = sorted(editable & frozen)
    if overlap:
        raise ValueError(f"paths cannot be editable and frozen: {overlap}")
    if template["portfolio_contract"] not in frozen:
        raise ValueError("the portfolio contract must be in the frozen path set")
    search_space = template["search_space"]
    for combination in template.get("invalid_parameter_combinations", []):
        if not isinstance(combination, dict) or not combination:
            raise ValueError("invalid parameter combinations must be non-empty objects")
        unknown = sorted(set(combination) - set(search_space))
        if unknown:
            raise ValueError(f"invalid parameter combination has unknown fields: {unknown}")
        for name, value in combination.items():
            if str(value) not in {str(item) for item in search_space[name]}:
                raise ValueError(f"invalid parameter combination uses unsupported {name}")
    evaluator_paths = set(
        template["evaluator"].get("frozen_paths", template["scope"]["frozen"])
    )
    if not evaluator_paths or not evaluator_paths <= frozen:
        raise ValueError("evaluator frozen_paths must be a subset of scope.frozen")
    collaboration = template.get("collaboration")
    if collaboration is not None:
        if collaboration.get("promotion_owner") != "root":
            raise ValueError("the root controller must own candidate promotion")
        if collaboration.get("evaluator_lock") != str(EVALUATOR_LOCK_PATH):
            raise ValueError("all Metal evaluators must share the global lock")
        if collaboration.get("local_acceptance_status") != "accepted_parent":
            raise ValueError("local winners must remain accepted parents until production validation")
    if template["metric"].get("role") == "search_proxy":
        gate = template["final_validation"].get("production_gate", {})
        if gate.get("metric") is None or float(gate.get("minimum_local_speedup", 0.0)) <= 1.0:
            raise ValueError("search proxies require a production local-speedup gate")
        if int(gate.get("minimum_pairs", 0)) < 5:
            raise ValueError("production promotion requires at least five paired observations")
        if int(gate.get("minimum_log_n", 0)) < 1:
            raise ValueError("production promotion requires a target trace scale")
        if not gate.get("workload"):
            raise ValueError("production promotion requires a fixed workload")
        if gate.get("require_alternating_orders") is not True:
            raise ValueError("production promotion requires alternating backend orders")
        if gate.get("require_clean_worktree") is not True:
            raise ValueError("production promotion requires a clean source worktree")
        evaluator = gate.get("evaluator", {})
        if not isinstance(evaluator.get("command"), list) or not evaluator["command"]:
            raise ValueError("production promotion requires an executable evaluator command")
        if int(evaluator.get("timeout_seconds", 0)) < 1:
            raise ValueError("production evaluator timeout must be positive")
        result_schema = int(evaluator.get("schema_version", 4))
        if result_schema not in {4, 5}:
            raise ValueError("production evaluator schema must be 4 or 5")
        local_kernel = gate.get("local_kernel")
        if local_kernel is not None and result_schema != 5:
            raise ValueError("named local-kernel production gates require schema 5")
        if result_schema == 5 and local_kernel not in PRODUCTION_LOCAL_KERNELS:
            raise ValueError("schema-5 production gates require a known local kernel")
        if local_kernel is not None:
            descriptor = PRODUCTION_LOCAL_KERNELS.get(local_kernel)
            if descriptor is None:
                raise ValueError("production gate names an unknown local kernel")
            if gate.get("metric") != descriptor["metric"]:
                raise ValueError("production scalar metric does not match the local kernel")
            if gate.get("paired_metric") != descriptor["paired_metric"]:
                raise ValueError("production paired metric does not match the local kernel")
            missing_guards = sorted(
                descriptor["required_guards"] - set(gate.get("required_guards", []))
            )
            if missing_guards:
                raise ValueError(
                    f"production gate omits mandatory local-kernel guards: {missing_guards}"
                )
        elif not isinstance(
            gate.get("paired_metric", "paired_instruction_ra_speedups"), str
        ):
            raise ValueError("production promotion requires a paired local metric")

        bindings = evaluator.get("parameter_bindings")
        if bindings is not None:
            if not isinstance(bindings, list):
                raise ValueError("production parameter bindings must be a list")
            expected_fingerprint = gate.get("expected_fingerprint", {})
            if not isinstance(expected_fingerprint, dict):
                raise ValueError("expected production fingerprint must be an object")
            binding_parameters = [binding.get("parameter") for binding in bindings]
            fingerprint_parameters = [
                specification.get("parameter")
                for specification in expected_fingerprint.values()
                if isinstance(specification, dict)
            ]
            if len(binding_parameters) != len(set(binding_parameters)):
                raise ValueError("production parameter bindings must be unique")
            if len(fingerprint_parameters) != len(expected_fingerprint) or len(
                fingerprint_parameters
            ) != len(set(fingerprint_parameters)):
                raise ValueError("production fingerprint parameters must be unique")
            if set(binding_parameters) != set(fingerprint_parameters):
                raise ValueError(
                    "production parameter bindings and fingerprint parameters must match"
                )
            if local_kernel is not None and set(binding_parameters) != descriptor["parameters"]:
                raise ValueError(
                    "production parameter bindings do not cover the local-kernel contract"
                )

            flags = []
            environment_names = []
            for binding in bindings:
                parameter = binding.get("parameter")
                if parameter not in search_space or parameter not in template.get(
                    "baseline_params", {}
                ):
                    raise ValueError(
                        "production parameter binding must name a baseline search parameter"
                    )
                destination = binding.get("destination")
                if destination == "argument":
                    flag = binding.get("flag")
                    value_format = binding.get("value_format")
                    if (
                        not isinstance(flag, str)
                        or not flag.startswith("--")
                        or not isinstance(value_format, str)
                        or value_format.count("{}") != 1
                    ):
                        raise ValueError(
                            "argument bindings require a safe flag and one-value format"
                        )
                    if flag in {"--mode", "--local-kernel"}:
                        raise ValueError("production bindings cannot override reserved flags")
                    try:
                        rendered = value_format.format("value")
                    except (IndexError, KeyError, ValueError) as error:
                        raise ValueError("invalid production argument value_format") from error
                    if not rendered or any(character.isspace() for character in rendered):
                        raise ValueError("production argument values must be one token")
                    flags.append(flag)
                elif destination == "boolean_flag":
                    flag = binding.get("flag")
                    if (
                        not isinstance(flag, str)
                        or not flag.startswith("--")
                        or str(binding.get("true_value")) != "1"
                        or {str(value) for value in search_space[parameter]} - {"0", "1"}
                    ):
                        raise ValueError("invalid production Boolean flag binding")
                    if flag in {"--mode", "--local-kernel"}:
                        raise ValueError("production bindings cannot override reserved flags")
                    flags.append(flag)
                elif destination == "environment":
                    name = binding.get("name")
                    if (
                        not isinstance(name, str)
                        or not name.startswith("JOLT_METAL_")
                        or name == EVALUATOR_LOCK_HELD_ENV
                        or name in evaluator.get("env", {})
                    ):
                        raise ValueError("production environment bindings require JOLT_METAL_ names")
                    environment_names.append(name)
                else:
                    raise ValueError("unknown production parameter binding destination")
            if len(flags) != len(set(flags)) or any(
                flag in evaluator["command"] for flag in flags
            ):
                raise ValueError("production argument flags must be unique and unbound")
            if len(environment_names) != len(set(environment_names)):
                raise ValueError("production environment names must be unique")
            for specification in expected_fingerprint.values():
                if specification.get("type") not in {"int", "bool01", "str"}:
                    raise ValueError("unsupported production fingerprint conversion")
        elif local_kernel is not None and descriptor["parameters"]:
            raise ValueError("local-kernel production gates require parameter bindings")
        if any(flag in evaluator["command"] for flag in ("--mode", "--local-kernel")):
            raise ValueError("production evaluator command contains a reserved controller flag")
        if EVALUATOR_LOCK_HELD_ENV in evaluator.get("env", {}):
            raise ValueError("production evaluator environment cannot override the lock token")


def validate_goal_contract(contract: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "goal",
        "goal_prompt",
        "primary_metric",
        "timing_boundary",
        "continuation",
        "kernel_promotion",
        "phase_budget",
        "validation",
    }
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError(f"goal contract is missing fields: {missing}")
    if contract["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported goal contract schema")
    metric = contract["primary_metric"]
    if metric["direction"] != "max" or metric["timed_span"] != "jolt_prover::piop":
        raise ValueError("the portfolio metric must maximize the PIOP span speedup")
    floor = float(metric["minimum_accepted_speedup"])
    if not math.isfinite(floor) or floor <= 1.0:
        raise ValueError("the portfolio speedup floor must exceed one")
    continuation = contract["continuation"]
    if continuation["stop_at_minimum"] is not False:
        raise ValueError("the portfolio must not stop solely because it reaches the floor")
    minimum_gain = float(continuation["minimum_projected_relative_gain"])
    if not 0.0 < minimum_gain < 1.0:
        raise ValueError("the portfolio continuation gain must be between zero and one")
    local_stretch_floor = float(continuation.get("clear_local_speedup_to_pursue", floor))
    if not math.isfinite(local_stretch_floor) or local_stretch_floor < floor:
        raise ValueError("the clear local stretch floor must be at least the portfolio floor")
    promotion_queue = contract.get("orchestration", {}).get("promotion_queue", {})
    if promotion_queue.get("owner") != "root":
        raise ValueError("the root controller must own the promotion queue")
    if promotion_queue.get("global_lock") != str(EVALUATOR_LOCK_PATH):
        raise ValueError("the promotion queue must use the shared evaluator lock")
    orchestration = contract.get("orchestration", {})
    if orchestration.get("goal_decision_requires_disjoint_share_attestation") is not True:
        raise ValueError("portfolio projections require disjoint-share attestation")
    if int(contract["validation"].get("interleaved_pairs", 0)) < 5:
        raise ValueError("portfolio acceptance requires at least five interleaved pairs")


def validate_params(config: dict[str, Any], params: dict[str, str]) -> None:
    search_space = config["search_space"]
    unknown = sorted(set(params) - set(search_space))
    if unknown:
        raise ValueError(f"parameters are outside the search space: {unknown}")
    for name, value in params.items():
        allowed = {str(item) for item in search_space[name]}
        if value not in allowed:
            raise ValueError(f"{name}={value} is not one of {sorted(allowed)}")
    effective = {
        **{str(name): str(value) for name, value in config.get("baseline_params", {}).items()},
        **params,
    }
    for combination in config.get("invalid_parameter_combinations", []):
        if all(effective.get(name) == str(value) for name, value in combination.items()):
            rendered = ", ".join(f"{name}={value}" for name, value in combination.items())
            raise ValueError(f"invalid parameter combination: {rendered}")


def run_evaluator(
    root: Path,
    config: dict[str, Any],
    params: dict[str, str],
    log_dir: Path,
    label: str,
    remaining_seconds: Optional[float] = None,
) -> tuple[dict[str, Any], float]:
    command = config["evaluator"]["command"]
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in config["evaluator"].get("env", {}).items()})
    environment.update(params)
    environment["JOLT_AUTORESEARCH_EVAL_DIR"] = str(log_dir / f"{label}.artifacts")
    timeout = float(config["evaluator"]["timeout_seconds"])
    if remaining_seconds is not None:
        timeout = min(timeout, remaining_seconds)
    if timeout <= 0.0:
        raise ValueError("evaluator phase wall-clock budget exhausted")
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as error:
        (log_dir / f"{label}.stdout").write_text(error.stdout or "")
        (log_dir / f"{label}.stderr").write_text(error.stderr or "")
        raise ValueError("evaluator timed out") from error
    elapsed = time.monotonic() - started
    (log_dir / f"{label}.stdout").write_text(result.stdout)
    (log_dir / f"{label}.stderr").write_text(result.stderr)
    if result.returncode != 0:
        raise ValueError(f"evaluator exited with status {result.returncode}")
    output = parse_result(result.stdout)
    if output.get("kernel") != config["kernel"]:
        raise ValueError("evaluator returned the wrong kernel")
    metric = output.get("metrics", {}).get(config["metric"]["name"])
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric):
        raise ValueError("evaluator returned a non-finite primary metric")
    return output, elapsed


def guards_pass(config: dict[str, Any], output: dict[str, Any]) -> tuple[bool, str]:
    guards = output.get("guards")
    if not isinstance(guards, dict):
        return False, "evaluator returned no guard object"
    failed = [name for name in config["guards"]["required_true"] if guards.get(name) is not True]
    if failed:
        return False, f"failed guards: {failed}"
    return True, "all guards passed"


def expected_fingerprint_value(specification: dict[str, Any], value: str) -> Any:
    conversion = specification["type"]
    if conversion == "int":
        return int(value)
    if conversion == "bool01":
        if value not in {"0", "1"}:
            raise ValueError("bool01 production parameters must be zero or one")
        return value == "1"
    if conversion == "str":
        return value
    raise ValueError(f"unknown fingerprint conversion {conversion}")


def validate_production_result(
    config: dict[str, Any],
    result: dict[str, Any],
    expected_revision: str,
    expected_params: dict[str, str],
    current_worktree_clean: bool,
) -> dict[str, Any]:
    gate = config["final_validation"].get("production_gate", {})
    result_schema = int(gate.get("evaluator", {}).get("schema_version", 4))
    if result.get("schema_version") != result_schema or result.get("kernel") != "akita_piop":
        raise ValueError(
            f"production validation requires a schema-{result_schema} Akita PIOP result"
        )
    local_kernel = gate.get("local_kernel")
    if local_kernel is not None:
        descriptor = PRODUCTION_LOCAL_KERNELS[local_kernel]
        if result.get("local_kernel") != local_kernel or result.get("local_metric") != {
            "metric": descriptor["metric"],
            "paired_metric": descriptor["paired_metric"],
        }:
            raise ValueError("production result local-kernel descriptor does not match the gate")
        run_class = result.get("run_class")
        if run_class != {"mode": "production", "acceptance_eligible": True}:
            raise ValueError("production result was not emitted under the production contract")
    guards = result.get("guards", {})
    if not isinstance(guards, dict):
        raise ValueError("production result has no guard object")
    failed = [name for name in gate["required_guards"] if guards.get(name) is not True]
    if failed:
        raise ValueError(f"production result failed guards: {failed}")
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("production result has no metric object")
    metric_name = gate["metric"]
    metric = metrics.get(metric_name)
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric):
        raise ValueError("production result has no finite local-speedup metric")
    if metric < float(gate["minimum_local_speedup"]):
        raise ValueError("production result does not clear the local-speedup gate")
    pairs = metrics.get("paired_speedups")
    if not isinstance(pairs, list) or len(pairs) < int(gate["minimum_pairs"]):
        raise ValueError("production result has too few paired observations")
    if local_kernel is not None and len(pairs) != int(gate["minimum_pairs"]):
        raise ValueError("production result must contain exactly the contracted pair count")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in pairs
    ):
        raise ValueError("production result has invalid paired PIOP speedups")
    paired_metric = gate.get("paired_metric", "paired_instruction_ra_speedups")
    local_pairs = metrics.get(paired_metric)
    if not isinstance(local_pairs, list) or len(local_pairs) != len(pairs):
        raise ValueError("production result has incomplete local paired observations")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in local_pairs
    ):
        raise ValueError("production result has invalid local paired speedups")
    if not math.isclose(float(metric), statistics.median(local_pairs), rel_tol=1e-12):
        raise ValueError("production local-speedup summary disagrees with its pairs")
    if local_kernel == "BytecodeReadRafCycle":
        decision = metrics.get("bytecode_read_raf_cycle_decision")
        if not isinstance(decision, dict) or decision.get("clears") is not True:
            raise ValueError("production Bytecode result did not clear its fixed local decision")
        decision_speedup = decision.get("median_speedup")
        if not math.isclose(
            float(decision_speedup)
            if isinstance(decision_speedup, (int, float))
            and not isinstance(decision_speedup, bool)
            else math.nan,
            float(metric),
            rel_tol=1e-12,
        ):
            raise ValueError("production Bytecode decision disagrees with its scalar metric")
        pair_records = result.get("pairs")
        if not isinstance(pair_records, list) or len(pair_records) != len(local_pairs):
            raise ValueError("production Bytecode result has incomplete raw pair records")
        for index, (record, local_speedup) in enumerate(zip(pair_records, local_pairs)):
            expected_order = (
                ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
            )
            if not isinstance(record, dict) or record.get("order") != expected_order:
                raise ValueError("production Bytecode raw pair order is invalid")
            arms = record.get("arms", {})
            try:
                cpu_member = arms["optimized"]["bytecode"]["member_ns"]
                metal_member = arms["metal"]["bytecode"]["member_ns"]
            except (KeyError, TypeError) as error:
                raise ValueError("production Bytecode raw pair is incomplete") from error
            if (
                isinstance(cpu_member, bool)
                or not isinstance(cpu_member, int)
                or cpu_member <= 0
                or isinstance(metal_member, bool)
                or not isinstance(metal_member, int)
                or metal_member <= 0
                or not math.isclose(
                    float(local_speedup), cpu_member / metal_member, rel_tol=1e-9
                )
            ):
                raise ValueError("production Bytecode raw pair disagrees with its speedup")
    piop_speedup = metrics.get("piop_speedup")
    if (
        isinstance(piop_speedup, bool)
        or not isinstance(piop_speedup, (int, float))
        or not math.isfinite(piop_speedup)
        or not math.isclose(float(piop_speedup), statistics.median(pairs), rel_tol=1e-12)
    ):
        raise ValueError("production PIOP summary disagrees with its pairs")
    fingerprint = result.get("fingerprint", {})
    if not isinstance(fingerprint, dict):
        raise ValueError("production result has no fingerprint object")
    if fingerprint.get("git_revision") != expected_revision:
        raise ValueError("production result revision does not match the accepted source")
    if local_kernel is not None and fingerprint.get("local_kernel") != local_kernel:
        raise ValueError("production fingerprint used the wrong local kernel")
    if gate.get("require_clean_worktree") and (
        fingerprint.get("worktree_dirty") is not False or not current_worktree_clean
    ):
        raise ValueError("production promotion requires clean result and current worktrees")
    if fingerprint.get("workload") != gate["workload"]:
        raise ValueError("production result used the wrong workload")
    log_n = fingerprint.get("log_n")
    if isinstance(log_n, bool) or not isinstance(log_n, int) or log_n < int(gate["minimum_log_n"]):
        raise ValueError("production result used a sub-target trace scale")
    if fingerprint.get("span") != "jolt_prover::piop":
        raise ValueError("production result used the wrong timed span")
    orders = fingerprint.get("orders")
    expected_orders = [
        ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
        for index in range(len(pairs))
    ]
    if gate.get("require_alternating_orders") and orders != expected_orders:
        raise ValueError("production result did not alternate backend order")
    for name, specification in gate.get("expected_fingerprint", {}).items():
        parameter = specification["parameter"]
        if parameter not in expected_params:
            raise ValueError(f"accepted parameters are missing {parameter}")
        expected = expected_fingerprint_value(specification, expected_params[parameter])
        if fingerprint.get(name) != expected:
            raise ValueError(f"production fingerprint does not match {parameter}")
    return {
        "metric": metric_name,
        "paired_metric": paired_metric,
        "metric_value": float(metric),
        "minimum_local_speedup": float(gate["minimum_local_speedup"]),
        "pairs": len(pairs),
        "piop_speedup": float(piop_speedup),
    }


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_path = run_dir / "run.json"
    config = read_json(run_path)
    expected = (run_dir / "run.sha256").read_text().strip()
    if sha256(run_path.read_bytes()) != expected:
        raise ValueError("run.json changed after initialization")
    events: list[dict[str, Any]] = []
    accepted = {"baseline"}
    seen = {"baseline"}
    for number, line in enumerate((run_dir / "events.jsonl").read_text().splitlines(), 1):
        if not line:
            raise ValueError(f"events.jsonl:{number}: blank record")
        event = json.loads(line)
        if event.get("index") != number or event.get("verdict") not in VERDICTS:
            raise ValueError(f"events.jsonl:{number}: invalid event")
        if event.get("trial_id") in seen or event.get("parent_id") not in accepted:
            raise ValueError(f"events.jsonl:{number}: invalid lineage")
        seen.add(event["trial_id"])
        if event["verdict"] == "keep":
            accepted.add(event["trial_id"])
        events.append(event)
    return config, events


def accepted_parent(config: dict[str, Any], events: list[dict[str, Any]]) -> tuple[str, float]:
    parent_id = "baseline"
    value = float(config["baseline"]["metric_median"])
    for event in events:
        if event["verdict"] == "keep":
            parent_id = event["trial_id"]
            value = float(event["metric_value"])
    return parent_id, value


def accepted_parent_params(
    config: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, str]:
    params = {str(name): str(value) for name, value in config["baseline"]["params"].items()}
    for event in events:
        if event["verdict"] == "keep":
            params.update({str(name): str(value) for name, value in event["params"].items()})
    return params


def candidate_context(
    run_dir: Path, config: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, str]:
    parent_id, _ = accepted_parent(config, events)
    return {
        "run_sha256": (run_dir / "run.sha256").read_text().strip(),
        "base_revision": config["base_revision"],
        "parent_id": parent_id,
        "frozen_paths_sha256": config["fingerprint"]["frozen_paths_sha256"],
        "parent_editable_paths_sha256": path_digest(
            run_dir / "snapshots" / parent_id, config["scope"]["editable"]
        ),
        "parent_params_sha256": sha256(
            canonical_json(accepted_parent_params(config, events))
        ),
        "evaluator_contract_sha256": config["fingerprint"][
            "evaluator_contract_sha256"
        ],
        "evaluator_paths_sha256": config["fingerprint"]["evaluator_paths_sha256"],
        "outside_editable_worktree_sha256": config["fingerprint"][
            "outside_editable_worktree_sha256"
        ],
    }


def validate_candidate_manifest(
    manifest: dict[str, Any], expected: dict[str, str]
) -> None:
    required = {
        "schema_version",
        "candidate_id",
        "producer",
        "summary",
        "candidate_editable_paths_sha256",
        "analysis_sha256",
        "patch_sha256",
        *expected.keys(),
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"candidate manifest is missing fields: {missing}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported candidate manifest schema")
    if CANDIDATE_ID.fullmatch(str(manifest["candidate_id"])) is None:
        raise ValueError("candidate_id contains unsafe characters")
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"candidate has stale {field}")
    for field in (
        "candidate_editable_paths_sha256",
        "analysis_sha256",
        "patch_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(manifest[field])) is None:
            raise ValueError(f"candidate {field} must be SHA-256")


def median_and_relative_mad(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one measurement is required")
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    relative_mad = statistics.median(deviations) / abs(median) if median else 0.0
    return median, relative_mad


def goal_decision(
    contract: dict[str, Any],
    current_piop_speedup: float,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    floor = float(contract["primary_metric"]["minimum_accepted_speedup"])
    minimum_gain = float(contract["continuation"]["minimum_projected_relative_gain"])
    local_stretch_floor = float(
        contract["continuation"].get("clear_local_speedup_to_pursue", floor)
    )
    if not math.isfinite(current_piop_speedup) or current_piop_speedup <= 0.0:
        raise ValueError("current PIOP speedup must be finite and positive")
    if not math.isfinite(floor) or floor <= 1.0:
        raise ValueError("the accepted PIOP speedup floor must exceed one")
    if not math.isfinite(minimum_gain) or not 0.0 < minimum_gain < 1.0:
        raise ValueError("the projected continuation gain must be between zero and one")
    if not math.isfinite(local_stretch_floor) or local_stretch_floor < floor:
        raise ValueError("the clear local stretch floor must be at least the portfolio floor")

    total_share = 0.0
    projected_time = 1.0
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        kernel = str(candidate["kernel"])
        share = float(candidate["current_piop_share"])
        local_speedup = float(candidate["conservative_local_speedup"])
        if not math.isfinite(share) or not 0.0 <= share <= 1.0:
            raise ValueError(f"{kernel} has an invalid current PIOP share")
        if not math.isfinite(local_speedup) or local_speedup < 1.0:
            raise ValueError(f"{kernel} has an invalid conservative local speedup")
        total_share += share
        projected_time -= share * (1.0 - 1.0 / local_speedup)
        ranked.append(
            {
                "kernel": kernel,
                "current_piop_share": share,
                "conservative_local_speedup": local_speedup,
                "projected_time_fraction_saved": share * (1.0 - 1.0 / local_speedup),
            }
        )
    if total_share > 1.0 + 1e-12:
        raise ValueError("candidate PIOP shares overlap or sum above one")

    projected_speedup = current_piop_speedup / projected_time
    projected_gain = projected_speedup / current_piop_speedup - 1.0
    floor_met = current_piop_speedup >= floor
    clear_local_stretch = any(
        candidate["conservative_local_speedup"] > local_stretch_floor for candidate in ranked
    )
    should_continue = not floor_met or projected_gain >= minimum_gain or clear_local_stretch
    ranked.sort(key=lambda candidate: candidate["projected_time_fraction_saved"], reverse=True)
    return {
        "continue": should_continue,
        "floor_met": floor_met,
        "current_piop_speedup": current_piop_speedup,
        "minimum_accepted_speedup": floor,
        "projected_piop_speedup": projected_speedup,
        "projected_relative_gain": projected_gain,
        "minimum_projected_relative_gain": minimum_gain,
        "clear_local_speedup_to_pursue": local_stretch_floor,
        "clear_local_stretch": clear_local_stretch,
        "next_kernel": ranked[0]["kernel"] if ranked else None,
        "candidates": ranked,
        "reason": (
            "the minimum PIOP speedup has not been reached"
            if not floor_met
            else "conservative residual headroom clears the continuation threshold"
            if projected_gain >= minimum_gain
            else "a conservative local speedup exceeds the uncapped stretch floor"
            if clear_local_stretch
            else "the floor is met and conservative residual headroom is below the threshold"
        ),
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


def append_event(path: Path, event: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(descriptor, (json.dumps(event, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def record_candidate_event(
    run_dir: Path,
    candidate_id: str,
    status: str,
    reason: str,
    manifest_sha256: str,
) -> None:
    if status not in CANDIDATE_STATUSES:
        raise ValueError(f"invalid candidate status: {status}")
    append_event(
        run_dir / "candidate-events.jsonl",
        {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "status": status,
            "reason": reason,
            "manifest_sha256": manifest_sha256,
            "recorded_at": utc_now(),
        },
    )


def candidate_status_recorded(run_dir: Path, candidate_id: str, status: str) -> bool:
    path = run_dir / "candidate-events.jsonl"
    if not path.exists():
        return False
    return any(
        record.get("candidate_id") == candidate_id and record.get("status") == status
        for record in (json.loads(line) for line in path.read_text().splitlines())
    )


def production_rejection_record(run_dir: Path) -> Optional[dict[str, Any]]:
    ledger = run_dir / "production-validations.jsonl"
    if not ledger.exists():
        return None
    for line in ledger.read_text().splitlines():
        record = json.loads(line)
        if record.get("status") == "rejected":
            return record
    return None


def command_init(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    template = read_json(Path(args.template))
    validate_template(template)
    goal_contract = read_json(root / template["portfolio_contract"])
    validate_goal_contract(goal_contract)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = run_dir / "logs"
    logs.mkdir()
    snapshots = run_dir / "snapshots"
    snapshots.mkdir()
    snapshot_paths(root, template["scope"]["editable"], snapshots / "baseline")
    initial_editable_digest = path_digest(root, template["scope"]["editable"])
    initial_frozen_digest = path_digest(root, template["scope"]["frozen"])
    initial_evaluator_digest = path_digest(
        root,
        template["evaluator"].get("frozen_paths", template["scope"]["frozen"]),
    )
    initial_outside_editable_digest = outside_editable_worktree_digest(
        root, template["scope"]["editable"]
    )

    baseline_params = {str(k): str(v) for k, v in template.get("baseline_params", {}).items()}
    validate_params(template, baseline_params)
    measurements = []
    elapsed_total = 0.0
    gpu_seconds = 0.0
    for index in range(template["baseline_repeats"]):
        remaining_seconds = float(template["budget"]["max_seconds"]) - elapsed_total
        output, elapsed = run_evaluator(
            root,
            template,
            baseline_params,
            logs,
            f"baseline-{index + 1:02d}",
            remaining_seconds,
        )
        passed, reason = guards_pass(template, output)
        if not passed:
            raise ValueError(f"baseline {index + 1} is invalid: {reason}")
        measurements.append(float(output["metrics"][template["metric"]["name"]]))
        elapsed_total += elapsed
        gpu_seconds += float(output.get("resources", {}).get("gpu_seconds", 0.0))
        if gpu_seconds > float(template["budget"]["max_gpu_seconds"]):
            raise ValueError("baseline GPU budget exhausted")

    if path_digest(root, template["scope"]["frozen"]) != initial_frozen_digest:
        raise ValueError("a frozen path changed during baseline evaluation")
    if path_digest(root, template["scope"]["editable"]) != initial_editable_digest:
        raise ValueError("an editable path changed during baseline evaluation")
    if path_digest(
        root,
        template["evaluator"].get("frozen_paths", template["scope"]["frozen"]),
    ) != initial_evaluator_digest:
        raise ValueError("an evaluator path changed during baseline evaluation")
    if outside_editable_worktree_digest(
        root, template["scope"]["editable"]
    ) != initial_outside_editable_digest:
        raise ValueError("a path outside the editable scope changed during baseline evaluation")

    median, relative_mad = median_and_relative_mad(measurements)
    config = dict(template)
    config["portfolio"] = goal_contract
    config["created_at"] = utc_now()
    config["base_revision"] = git_head(root)
    config["controller"] = {
        "path": "scripts/metal_autoresearch.py",
        "version": SCHEMA_VERSION,
        "mode": "foreground source and parameter search",
    }
    config["fingerprint"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "frozen_paths_sha256": initial_frozen_digest,
        "editable_paths_sha256": initial_editable_digest,
        "portfolio_contract_sha256": sha256(canonical_json(goal_contract)),
        "evaluator_contract_sha256": sha256(canonical_json(config["evaluator"])),
        "evaluator_paths_sha256": initial_evaluator_digest,
        "outside_editable_worktree_sha256": initial_outside_editable_digest,
    }
    config["baseline"] = {
        "params": baseline_params,
        "measurements": measurements,
        "metric_median": median,
        "relative_mad": relative_mad,
        "elapsed_seconds": elapsed_total,
        "gpu_seconds": gpu_seconds,
    }
    config["metric"]["promotion_relative_threshold"] = max(
        float(config["metric"]["minimum_relative_improvement"]),
        3.0 * relative_mad,
    )
    config["fingerprint"]["evaluator"] = output.get("fingerprint", {})
    encoded = canonical_json(config)
    (run_dir / "run.json").write_bytes(encoded)
    (run_dir / "run.sha256").write_text(sha256(encoded) + "\n")
    (run_dir / "events.jsonl").touch()
    (run_dir / "candidate-events.jsonl").touch()
    (run_dir / "production-validations.jsonl").touch()
    print(json.dumps({"run_dir": str(run_dir), "baseline": config["baseline"]}, sort_keys=True))
    return 0


def command_candidate_context(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    config, events = load_run(run_dir)
    print(json.dumps(candidate_context(run_dir, config, events), indent=2, sort_keys=True))
    return 0


def command_trial(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve()
    config, events = load_run(run_dir)
    if (run_dir / "production-rejected.json").exists() or production_rejection_record(run_dir):
        raise ValueError("the production gate rejected this phase; start a new phase")
    candidate = None
    candidate_manifest_sha256 = None
    live_revision = git_head(root)
    if config.get("collaboration") is not None:
        if args.candidate_manifest is None:
            raise ValueError("collaborative runs require --candidate-manifest")
        manifest_path = Path(args.candidate_manifest).resolve()
        manifest_bytes = manifest_path.read_bytes()
        candidate = json.loads(manifest_bytes)
        if not isinstance(candidate, dict):
            raise ValueError("candidate manifest must contain a JSON object")
        candidate_manifest_sha256 = sha256(manifest_bytes)
        candidate_id = str(candidate.get("candidate_id", "invalid"))
        try:
            if live_revision != config["base_revision"]:
                raise ValueError("run phase base revision no longer matches live HEAD")
            if root == manifest_path or root in manifest_path.parents:
                raise ValueError("candidate artifacts must be outside the shared worktree")
            expected = candidate_context(run_dir, config, events)
            expected.update(
                frozen_paths_sha256=path_digest(root, config["scope"]["frozen"]),
                evaluator_paths_sha256=path_digest(
                    root,
                    config["evaluator"].get(
                        "frozen_paths", config["scope"]["frozen"]
                    ),
                ),
                candidate_editable_paths_sha256=path_digest(
                    root, config["scope"]["editable"]
                ),
                outside_editable_worktree_sha256=outside_editable_worktree_digest(
                    root, config["scope"]["editable"]
                ),
            )
            validate_candidate_manifest(candidate, expected)
            candidate_ledger = run_dir / "candidate-events.jsonl"
            if candidate_ledger.exists() and any(
                json.loads(line).get("candidate_id") == candidate["candidate_id"]
                for line in candidate_ledger.read_text().splitlines()
            ):
                raise ValueError("candidate_id was already admitted in this run")
            artifacts = {"analysis.md": "analysis_sha256", "candidate.patch": "patch_sha256"}
            for relative, field in artifacts.items():
                artifact = manifest_path.parent / relative
                if not artifact.is_file() or sha256(artifact.read_bytes()) != candidate[field]:
                    raise ValueError(f"candidate artifact hash mismatch: {relative}")
        except (OSError, ValueError) as error:
            record_candidate_event(
                run_dir,
                candidate_id,
                "rejected",
                str(error),
                candidate_manifest_sha256,
            )
            raise
    elif args.summary is None:
        raise ValueError("non-collaborative trials require --summary")
    if live_revision != config["base_revision"]:
        raise ValueError("run phase base revision no longer matches live HEAD")
    if path_digest(root, config["scope"]["frozen"]) != config["fingerprint"]["frozen_paths_sha256"]:
        raise ValueError("a frozen path changed; start a new run phase")
    if outside_editable_worktree_digest(
        root, config["scope"]["editable"]
    ) != config["fingerprint"]["outside_editable_worktree_sha256"]:
        raise ValueError("a path outside the editable scope changed; start a new run phase")
    inflight = run_dir / "inflight.json"
    if inflight.exists():
        raise ValueError("an interrupted trial needs `recover` before another trial")
    if len(events) >= config["budget"]["max_trials"]:
        raise ValueError("trial budget exhausted")
    elapsed_used = float(config["baseline"]["elapsed_seconds"]) + sum(
        float(event["elapsed_seconds"]) for event in events
    )
    if elapsed_used >= config["budget"]["max_seconds"]:
        raise ValueError("wall-clock budget exhausted")
    gpu_used = float(config["baseline"]["gpu_seconds"]) + sum(
        float(event["resources"].get("gpu_seconds", 0.0)) for event in events
    )
    if gpu_used >= config["budget"]["max_gpu_seconds"]:
        raise ValueError("GPU budget exhausted")

    parameter_overrides = dict(item.split("=", 1) for item in args.param)
    params = accepted_parent_params(config, events)
    params.update(parameter_overrides)
    validate_params(config, params)
    index = len(events) + 1
    trial_id = f"trial-{index:03d}"
    parent_id, parent_metric = accepted_parent(config, events)
    started_at = utc_now()
    candidate_revision = path_digest(root, config["scope"]["editable"])
    inflight.write_bytes(
        canonical_json(
            {
                "trial_id": trial_id,
                "parent_id": parent_id,
                "candidate_revision": candidate_revision,
                "candidate_id": candidate.get("candidate_id") if candidate else None,
                "candidate_manifest_sha256": candidate_manifest_sha256,
                "params": params,
                "started_at": started_at,
            }
        )
    )
    if candidate is not None:
        record_candidate_event(
            run_dir,
            candidate["candidate_id"],
            "queued",
            "root admitted candidate for serialized evaluation",
            candidate_manifest_sha256,
        )
    elapsed = 0.0
    gpu_seconds = 0.0
    measurements = []
    combined_guards = {name: True for name in config["guards"]["required_true"]}
    try:
        for repeat in range(config.get("candidate_repeats", 1)):
            remaining_seconds = float(config["budget"]["max_seconds"]) - elapsed_used - elapsed
            output, repetition_elapsed = run_evaluator(
                root,
                config,
                params,
                run_dir / "logs",
                f"{trial_id}-{repeat + 1:02d}",
                remaining_seconds,
            )
            elapsed += repetition_elapsed
            gpu_seconds += float(output.get("resources", {}).get("gpu_seconds", 0.0))
            if gpu_used + gpu_seconds > float(config["budget"]["max_gpu_seconds"]):
                raise ValueError("candidate GPU budget exhausted")
            measurements.append(float(output["metrics"][config["metric"]["name"]]))
            passed, reason = guards_pass(config, output)
            for name in combined_guards:
                combined_guards[name] = combined_guards[name] and output["guards"].get(name) is True
            if not passed:
                break
        if path_digest(root, config["scope"]["editable"]) != candidate_revision:
            raise ValueError("editable source changed during candidate evaluation")
        if outside_editable_worktree_digest(
            root, config["scope"]["editable"]
        ) != config["fingerprint"]["outside_editable_worktree_sha256"]:
            raise ValueError("a path outside the editable scope changed during evaluation")
        metric_value = statistics.median(measurements)
        if not passed:
            verdict = "invalid"
        else:
            delta = config["metric"]["promotion_relative_threshold"]
            if config["metric"]["direction"] == "max":
                kept = metric_value >= parent_metric * (1.0 + delta)
            else:
                kept = metric_value <= parent_metric * (1.0 - delta)
            verdict = "keep" if kept else "discard"
            reason = (
                "improves beyond the contract threshold"
                if kept
                else "does not clear the contract threshold"
            )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        metric_value = None
        verdict = "crash"
        reason = str(error)

    event = {
        "schema_version": SCHEMA_VERSION,
        "index": index,
        "trial_id": trial_id,
        "parent_id": parent_id,
        "candidate_revision": sha256(
            canonical_json({"source": candidate_revision, "params": params})
        ),
        "proposal_summary": candidate["summary"] if candidate else args.summary,
        "candidate_id": candidate.get("candidate_id") if candidate else None,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "params": params,
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "metric_value": metric_value,
        "measurements": measurements,
        "guards": combined_guards,
        "resources": {"gpu_seconds": gpu_seconds},
        "verdict": verdict,
        "reason": reason,
    }
    if verdict == "keep":
        snapshot_paths(
            root,
            config["scope"]["editable"],
            run_dir / "snapshots" / trial_id,
        )
    else:
        restore_snapshot(
            root,
            config["scope"]["editable"],
            run_dir / "snapshots" / parent_id,
        )
    append_event(run_dir / "events.jsonl", event)
    if candidate is not None:
        record_candidate_event(
            run_dir,
            candidate["candidate_id"],
            "accepted_parent" if verdict == "keep" else "rejected",
            reason,
            candidate_manifest_sha256,
        )
    inflight.unlink()
    print(json.dumps(event, sort_keys=True))
    return 0 if verdict in {"keep", "discard"} else 2


def command_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    config, events = load_run(run_dir)
    parent_id, metric = accepted_parent(config, events)
    validations = (run_dir / "production-validations.jsonl")
    validation_count = len(validations.read_text().splitlines()) if validations.exists() else 0
    summary = {
        "kernel": config["kernel"],
        "trials": len(events),
        "remaining_trials": config["budget"]["max_trials"] - len(events),
        "accepted_parent": parent_id,
        "accepted_metric": metric,
        "accepted_params": accepted_parent_params(config, events),
        "production_validations": validation_count,
        "production_rejected": (run_dir / "production-rejected.json").exists()
        or production_rejection_record(run_dir) is not None,
        "portfolio_minimum_speedup": config.get("portfolio", {})
        .get("primary_metric", {})
        .get("minimum_accepted_speedup"),
        "portfolio_stops_at_minimum": config.get("portfolio", {})
        .get("continuation", {})
        .get("stop_at_minimum"),
        "inflight": (Path(args.run_dir).resolve() / "inflight.json").exists(),
        "verdicts": {name: sum(event["verdict"] == name for event in events) for name in sorted(VERDICTS)},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def production_parent_event(
    events: list[dict[str, Any]], parent_id: str
) -> Optional[dict[str, Any]]:
    return next((event for event in events if event["trial_id"] == parent_id), None)


def repair_candidate_promotion(
    run_dir: Path,
    events: list[dict[str, Any]],
    parent_id: str,
) -> None:
    parent_event = production_parent_event(events, parent_id)
    if parent_event is None or parent_event.get("candidate_id") is None:
        return
    candidate_id = parent_event["candidate_id"]
    if candidate_status_recorded(run_dir, candidate_id, "promoted"):
        return
    record_candidate_event(
        run_dir,
        candidate_id,
        "promoted",
        "production relation cleared the executable paired-validation gate",
        parent_event["candidate_manifest_sha256"],
    )


def finalize_production_rejection(
    root: Path,
    run_dir: Path,
    config: dict[str, Any],
    events: list[dict[str, Any]],
    rejection: dict[str, Any],
) -> None:
    parent_id = str(rejection["parent_id"])
    parent_event = production_parent_event(events, parent_id)
    restored_parent = None
    if parent_event is not None:
        restored_parent = parent_event["parent_id"]
        restore_snapshot(
            root,
            config["scope"]["editable"],
            run_dir / "snapshots" / restored_parent,
        )
        candidate_id = parent_event.get("candidate_id")
        if candidate_id is not None and not candidate_status_recorded(
            run_dir, candidate_id, "rejected"
        ):
            record_candidate_event(
                run_dir,
                candidate_id,
                "rejected",
                f"production gate failed: {rejection['reason']}",
                parent_event["candidate_manifest_sha256"],
            )
    marker = {**rejection, "restored_parent": restored_parent}
    (run_dir / "production-rejected.json").write_bytes(canonical_json(marker))


def run_production_evaluator(
    root: Path,
    run_dir: Path,
    config: dict[str, Any],
    params: dict[str, str],
) -> tuple[dict[str, Any], bytes, Path]:
    gate = config["final_validation"]["production_gate"]
    evaluator = gate["evaluator"]
    command = [str(item) for item in evaluator["command"]]
    local_kernel = gate.get("local_kernel")
    if local_kernel is not None:
        command.extend(["--mode", "production", "--local-kernel", local_kernel])
    inherited = os.environ.copy()
    lock_token = inherited.get(EVALUATOR_LOCK_HELD_ENV)
    environment = {
        name: value
        for name, value in inherited.items()
        if not name.startswith("JOLT_METAL_")
        and not name.startswith("JOLT_AUTORESEARCH_")
    }
    if lock_token is not None:
        environment[EVALUATOR_LOCK_HELD_ENV] = lock_token
    environment.update(
        {str(name): str(value) for name, value in evaluator.get("env", {}).items()}
    )
    bindings = evaluator.get("parameter_bindings")
    if bindings is None:
        fingerprint = gate.get("expected_fingerprint", {})
        width = fingerprint.get("instruction_ra_materialize_width")
        reuse = fingerprint.get("instruction_ra_reuse_inverse")
        if width is not None and reuse is not None:
            command.extend(
                ["--instruction-ra-materialize-width", params[width["parameter"]]]
            )
            reuse_value = params[reuse["parameter"]]
            if reuse_value not in {"0", "1"}:
                raise ValueError("legacy production reuse parameter must be zero or one")
            if reuse_value == "1":
                command.append("--instruction-ra-reuse-inverse")
        elif width is not None or reuse is not None:
            raise ValueError("legacy production fingerprint must bind both Instruction RA flags")
    else:
        for binding in bindings:
            value = params[binding["parameter"]]
            destination = binding["destination"]
            if destination == "argument":
                command.extend(
                    [binding["flag"], str(binding["value_format"]).format(value)]
                )
            elif destination == "boolean_flag":
                if value == str(binding["true_value"]):
                    command.append(binding["flag"])
            elif destination == "environment":
                environment[binding["name"]] = value

    attempts = run_dir / "production-attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / utc_now().replace(":", "-")
    attempt.mkdir()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            timeout=int(evaluator["timeout_seconds"]),
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
        (attempt / "stdout.log").write_text(stdout)
        (attempt / "stderr.log").write_text(stderr)
        raise ValueError("production evaluator timed out") from error
    (attempt / "stdout.log").write_text(completed.stdout)
    (attempt / "stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise ValueError(f"production evaluator exited with status {completed.returncode}")
    result_schema = int(evaluator.get("schema_version", 4))
    result = (
        parse_unique_schema_result(completed.stdout, result_schema)
        if result_schema == 5
        else parse_schema_result(completed.stdout, result_schema)
    )
    result_bytes = canonical_json(result)
    (attempt / "result.json").write_bytes(result_bytes)
    return result, result_bytes, attempt


def command_validate_production(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve()
    config, events = load_run(run_dir)
    parent_id, _ = accepted_parent(config, events)
    params = accepted_parent_params(config, events)
    ledger = run_dir / "production-validations.jsonl"
    if not ledger.exists():
        ledger.touch()
    prior = [json.loads(line) for line in ledger.read_text().splitlines()]
    successful = next(
        (
            record
            for record in prior
            if record.get("parent_id") == parent_id and record.get("status") == "promoted"
        ),
        None,
    )
    if successful is not None:
        repair_candidate_promotion(run_dir, events, parent_id)
        print(json.dumps(successful, sort_keys=True))
        return 0
    rejected = next(
        (
            record
            for record in prior
            if record.get("parent_id") == parent_id and record.get("status") == "rejected"
        ),
        None,
    )
    if rejected is not None:
        finalize_production_rejection(root, run_dir, config, events, rejected)
        raise ValueError("the production gate already rejected this phase")

    accepted_snapshot = run_dir / "snapshots" / parent_id
    editable = config["scope"]["editable"]
    if path_digest(root, editable) != path_digest(accepted_snapshot, editable):
        raise ValueError("live editable source does not match the accepted parent snapshot")
    if path_digest(root, config["scope"]["frozen"]) != config["fingerprint"][
        "frozen_paths_sha256"
    ]:
        raise ValueError("a frozen path changed after phase initialization")
    if not git_worktree_clean(root):
        raise ValueError("production evaluation requires a clean source worktree")
    expected_revision = git_head(root)
    result, result_bytes, attempt = run_production_evaluator(root, run_dir, config, params)
    try:
        evidence = validate_production_result(
            config,
            result,
            expected_revision,
            params,
            git_worktree_clean(root),
        )
        if git_head(root) != expected_revision:
            raise ValueError("source revision changed during production evaluation")
        if path_digest(root, editable) != path_digest(accepted_snapshot, editable):
            raise ValueError("accepted source changed during production evaluation")
        if path_digest(root, config["scope"]["frozen"]) != config["fingerprint"][
            "frozen_paths_sha256"
        ]:
            raise ValueError("a frozen path changed during production evaluation")
    except ValueError as error:
        rejection = {
            "schema_version": SCHEMA_VERSION,
            "status": "rejected",
            "parent_id": parent_id,
            "result_sha256": sha256(result_bytes),
            "attempt": str(attempt),
            "reason": str(error),
            "recorded_at": utc_now(),
        }
        append_event(ledger, rejection)
        finalize_production_rejection(root, run_dir, config, events, rejection)
        raise ValueError(f"production gate rejected the accepted parent: {error}") from error

    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "promoted",
        "parent_id": parent_id,
        "result_sha256": sha256(result_bytes),
        "attempt": str(attempt),
        "recorded_at": utc_now(),
        **evidence,
    }
    append_event(ledger, record)
    repair_candidate_promotion(run_dir, events, parent_id)
    print(json.dumps(record, sort_keys=True))
    return 0


def command_recover(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve()
    config, events = load_run(run_dir)
    inflight = run_dir / "inflight.json"
    if not inflight.exists():
        raise ValueError("there is no interrupted trial")
    interrupted = read_json(inflight)
    committed = next(
        (event for event in events if event["trial_id"] == interrupted["trial_id"]),
        None,
    )
    parent_id, _ = accepted_parent(config, events)
    quarantine = run_dir / "quarantine" / utc_now().replace(":", "-")
    snapshot_paths(root, config["scope"]["editable"], quarantine)
    orphan = run_dir / "snapshots" / interrupted["trial_id"]
    if committed is None and orphan.exists():
        shutil.move(orphan, quarantine / "orphan-accepted-snapshot")
    restore_snapshot(
        root,
        config["scope"]["editable"],
        run_dir / "snapshots" / parent_id,
    )
    candidate_id = interrupted.get("candidate_id")
    if candidate_id is not None and not candidate_status_recorded(
        run_dir, candidate_id, "queued"
    ):
        record_candidate_event(
            run_dir,
            candidate_id,
            "queued",
            "recovered an interrupted admission before its queue ledger write",
            interrupted.get("candidate_manifest_sha256", ""),
        )
    if committed is not None and candidate_id is not None:
        status = "accepted_parent" if committed["verdict"] == "keep" else "rejected"
        if not candidate_status_recorded(run_dir, candidate_id, status):
            record_candidate_event(
                run_dir,
                candidate_id,
                status,
                "recovered a committed trial whose final ledger write was interrupted",
                interrupted.get("candidate_manifest_sha256", ""),
            )
    elif candidate_id is not None and not candidate_status_recorded(
        run_dir, candidate_id, "rejected"
    ):
        record_candidate_event(
            run_dir,
            candidate_id,
            "rejected",
            "interrupted evaluation recovered to the accepted parent",
            interrupted.get("candidate_manifest_sha256", ""),
        )
    inflight.unlink()
    print(
        json.dumps(
            {
                "committed": committed is not None,
                "restored": parent_id,
                "quarantine": str(quarantine),
            },
            sort_keys=True,
        )
    )
    return 0


def command_goal_decision(args: argparse.Namespace) -> int:
    contract = read_json(Path(args.contract))
    validate_goal_contract(contract)
    candidates = [parse_goal_candidate(value) for value in args.candidate]
    if candidates and not args.shares_disjoint:
        raise ValueError("portfolio candidates require --shares-disjoint attestation")
    decision = goal_decision(contract, args.current_speedup, candidates)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def command_goal_prompt(args: argparse.Namespace) -> int:
    contract = read_json(Path(args.contract))
    validate_goal_contract(contract)
    print(f"/goal {contract['goal_prompt']}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", default=Path(__file__).resolve().parents[1])
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("template")
    init.add_argument("run_dir")
    init.set_defaults(handler=command_init)
    context = commands.add_parser("candidate-context")
    context.add_argument("run_dir")
    context.set_defaults(handler=command_candidate_context)
    trial = commands.add_parser("trial")
    trial.add_argument("run_dir")
    trial.add_argument("--candidate-manifest")
    trial.add_argument("--param", action="append", default=[])
    trial.add_argument("--summary")
    trial.set_defaults(handler=command_trial)
    status = commands.add_parser("status")
    status.add_argument("run_dir")
    status.set_defaults(handler=command_status)
    production = commands.add_parser("validate-production")
    production.add_argument("run_dir")
    production.set_defaults(handler=command_validate_production)
    recover = commands.add_parser("recover")
    recover.add_argument("run_dir")
    recover.set_defaults(handler=command_recover)
    goal = commands.add_parser("goal-decision")
    goal.add_argument("contract")
    goal.add_argument("--current-speedup", type=float, required=True)
    goal.add_argument("--candidate", action="append", default=[])
    goal.add_argument("--shares-disjoint", action="store_true")
    goal.set_defaults(handler=command_goal_decision)
    goal_prompt = commands.add_parser("goal-prompt")
    goal_prompt.add_argument("contract")
    goal_prompt.set_defaults(handler=command_goal_prompt)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command in {"init", "trial", "recover", "validate-production"}:
            with evaluator_lock({"controller_command": args.command}):
                return args.handler(args)
        return args.handler(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
