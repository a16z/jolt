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
CURRENT_PIOP_RESULT_SCHEMA = 7
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
    "InstructionInput": {
        "metric": "instruction_input_kernel_service_speedup",
        "paired_metric": "paired_instruction_input_kernel_service_speedups",
        "parameters": frozenset(
            {
                "JOLT_METAL_INSTRUCTION_INPUT_NATIVE_MESSAGE_THREADS",
                "JOLT_METAL_INSTRUCTION_INPUT_NATIVE_TRANSITION_THREADS",
                "JOLT_METAL_INSTRUCTION_INPUT_DENSE_TRANSITION_THREADS",
                "JOLT_METAL_INSTRUCTION_INPUT_CUTOFF_LOG2",
                "JOLT_METAL_INSTRUCTION_INPUT_TRACE_CUTOFF_LOG2",
            }
        ),
        "required_guards": COMMON_PRODUCTION_GUARDS
        | {
            "instruction_input_cpu_control",
            "instruction_input_cpu_rows_reused",
            "instruction_input_metal_backend_exercised",
            "instruction_input_resident_rows_reused",
            "instruction_input_working_set_admitted",
            "instruction_input_readback_exact",
            "instruction_input_host_readback_preallocated_outside_piop",
            "instruction_input_no_round_device_buffer_allocations",
            "instruction_input_local_gate",
        },
        "schema6_required_guards": {
            "instruction_input_minimal_initialization_exact",
            "instruction_input_storage_buffers_stable",
            "instruction_input_native_primer_exact_and_protocol_inert",
        },
        "schema7_required_guards": {
            "instruction_input_compact_rows_direct_and_stable",
        },
    },
}
LOCAL_RESULT_CONTRACTS = {
    "bytecode_read_raf_cycle_v1",
    "instruction_input_v2",
    "instruction_input_v3",
    "instruction_input_v4",
}
LOCAL_RESULT_SCHEMA_VERSIONS = {
    "bytecode_read_raf_cycle_v1": 1,
    "instruction_input_v2": 2,
    "instruction_input_v3": 3,
    "instruction_input_v4": 4,
}
BYTECODE_LOCAL_FINGERPRINT_PARAMETERS = {
    "message_threads": "JOLT_METAL_BYTECODE_MESSAGE_THREADS",
    "transition_threads": "JOLT_METAL_BYTECODE_TRANSITION_THREADS",
    "max_threadgroups": "JOLT_METAL_BYTECODE_MAX_THREADGROUPS",
    "cutoff_log2": "JOLT_METAL_BYTECODE_CUTOFF_LOG2",
    "trace_cutoff_log2": "JOLT_METAL_BYTECODE_TRACE_CUTOFF_LOG2",
}
BYTECODE_LOCAL_FINGERPRINT_ENV = {
    "log_n": "JOLT_METAL_EVAL_LOG_N",
    "repeats": "JOLT_METAL_EVAL_REPEATS",
    "seed": "JOLT_METAL_EVAL_SEED",
}
INSTRUCTION_INPUT_LOCAL_FINGERPRINT_PARAMETERS = {
    "native_message_threads": "JOLT_METAL_INSTRUCTION_INPUT_NATIVE_MESSAGE_THREADS",
    "native_transition_threads": "JOLT_METAL_INSTRUCTION_INPUT_NATIVE_TRANSITION_THREADS",
    "dense_transition_threads": "JOLT_METAL_INSTRUCTION_INPUT_DENSE_TRANSITION_THREADS",
    "cutoff_log2": "JOLT_METAL_INSTRUCTION_INPUT_CUTOFF_LOG2",
    "trace_cutoff_log2": "JOLT_METAL_INSTRUCTION_INPUT_TRACE_CUTOFF_LOG2",
}
INSTRUCTION_INPUT_LOCAL_FINGERPRINT_ENV = {
    "log_n": "JOLT_METAL_EVAL_LOG_N",
    "validation_log_n": "JOLT_METAL_EVAL_VALIDATE_LOG_N",
    "repeats": "JOLT_METAL_EVAL_REPEATS",
    "seed": "JOLT_METAL_EVAL_SEED",
}
INSTRUCTION_INPUT_V3_FINGERPRINT_ENV = {
    "frozen_cpu_reference_ns": "JOLT_METAL_EVAL_CPU_REFERENCE_NS",
}
INSTRUCTION_INPUT_V3_CPU_REFERENCE_NS = 814_395_125
INSTRUCTION_INPUT_V3_CPU_REFERENCE_PROVENANCE = (
    "median of 25 CPU ns samples from immutable instruction-input-a2-2f87d8b6a8 at "
    "2f87d8b6a81f1bb253c27795badc7da7baa3d0d8; compact-JSON sample SHA256 "
    "59f9946b7d1a3c05d3094528e853d2228ae5ec0d94a5dae2c63d5713a560a966"
)
INSTRUCTION_INPUT_ARCHITECTURE_EVIDENCE_GUARDS = frozenset(
    {
        "bytecode_command_buffers_completed",
        "bytecode_local_gate",
        "bytecode_metal_backend_exercised",
        "bytecode_q10_cpu_control",
        "bytecode_readback_exact",
        "bytecode_working_set_admitted",
        "cpu_proofs_verified",
        "instruction_input_cpu_control",
        "instruction_input_cpu_rows_reused",
        "instruction_input_host_readback_preallocated_outside_piop",
        "instruction_input_local_gate",
        "instruction_input_metal_backend_exercised",
        "instruction_input_minimal_initialization_exact",
        "instruction_input_native_primer_completed_before_join",
        "instruction_input_native_primer_exact_and_protocol_inert",
        "instruction_input_no_round_device_buffer_allocations",
        "instruction_input_readback_exact",
        "instruction_input_resident_rows_reused",
        "instruction_input_storage_buffers_stable",
        "instruction_input_working_set_admitted",
        "local_kernel_attributed",
        "local_kernel_metal_backend_exercised",
        "metal_proofs_verified",
        "production_contract",
        "stable_binary",
        "stable_source",
        "target_scale",
        "unique_backend_witness_prepare_span",
        "unique_piop_span",
    }
)


def instruction_input_sequence_storage_bytes(log_n: int) -> int:
    rows = 1 << log_n
    e_out = 1 << (log_n // 2)
    e_in = (rows // 2) // e_out
    elements = (
        8 * (rows // 2)
        + 8 * (rows // 4)
        + e_in
        + e_out
        + 2 * 3 * e_out
    )
    return 16 * elements


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


def git_changed_paths(root: Path, base_revision: str, revision: str) -> set[str]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            f"{base_revision}..{revision}",
            "--",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {os.fsdecode(raw) for raw in result.stdout.split(b"\0") if raw}


def validate_production_revision_scope(
    root: Path,
    base_revision: str,
    revision: str,
    editable: list[str],
) -> None:
    outside = sorted(
        path
        for path in git_changed_paths(root, base_revision, revision)
        if not path_is_in_scope(path, editable)
    )
    if outside:
        raise ValueError(
            f"production revision changed paths outside the editable scope: {outside}"
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


def validate_instruction_input_architecture_evidence(
    evidence: dict[str, Any], template: dict[str, Any]
) -> None:
    top_fields = {
        "schema",
        "schema_version",
        "status",
        "recorded_at",
        "candidate",
        "launch",
        "samples",
        "decision",
        "guards",
        "artifact",
        "reason",
        "disposition",
    }
    if set(evidence) != top_fields:
        raise ValueError("architecture baseline evidence fields are incomplete")
    if (
        evidence["schema"] != "metal_piop_rejection_evidence_v1"
        or type(evidence["schema_version"]) is not int
        or evidence["schema_version"] != 1
        or evidence["status"] != "rejected"
        or not isinstance(evidence["recorded_at"], str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", evidence["recorded_at"])
        is None
    ):
        raise ValueError("architecture baseline evidence has an invalid schema")

    candidate = evidence["candidate"]
    if not isinstance(candidate, dict) or set(candidate) != {
        "git_revision",
        "worktree_dirty",
        "worktree_state_sha256",
        "binary_sha256",
    }:
        raise ValueError("architecture baseline candidate record is incomplete")
    if (
        type(candidate["git_revision"]) is not str
        or re.fullmatch(r"[0-9a-f]{40}", candidate["git_revision"]) is None
        or candidate["worktree_dirty"] is not False
        or any(
            type(candidate[name]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", candidate[name]) is None
            for name in ("worktree_state_sha256", "binary_sha256")
        )
    ):
        raise ValueError("architecture baseline candidate record is invalid")

    launch = evidence["launch"]
    launch_fields = {
        "mode",
        "acceptance_eligible",
        "workload",
        "log_n",
        "pairs",
        "orders",
        "instruction_input",
    }
    if not isinstance(launch, dict) or set(launch) != launch_fields:
        raise ValueError("architecture baseline launch record is incomplete")
    pairs = launch["pairs"]
    orders = launch["orders"]
    expected_orders = [
        ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
        for index in range(pairs if type(pairs) is int and pairs > 0 else 0)
    ]
    if (
        launch["mode"] != "production"
        or launch["acceptance_eligible"] is not True
        or not isinstance(launch["workload"], str)
        or not launch["workload"]
        or type(launch["log_n"]) is not int
        or launch["log_n"] < 26
        or type(pairs) is not int
        or pairs < 5
        or pairs % 2 == 0
        or orders != expected_orders
    ):
        raise ValueError("architecture baseline launch record is invalid")
    instruction_input = launch["instruction_input"]
    instruction_input_fields = {
        "cutoff_log2",
        "trace_cutoff_log2",
        "native_message_threads",
        "native_transition_threads",
        "dense_transition_threads",
        "storage_initialization",
        "native_primer",
    }
    if not isinstance(instruction_input, dict) or set(instruction_input) != instruction_input_fields:
        raise ValueError("architecture baseline InstructionInput launch is incomplete")
    if (
        any(
            type(instruction_input[name]) is not int or instruction_input[name] <= 0
            for name in (
                "cutoff_log2",
                "trace_cutoff_log2",
                "native_message_threads",
                "native_transition_threads",
                "dense_transition_threads",
            )
        )
        or instruction_input["cutoff_log2"] > launch["log_n"]
        or instruction_input["trace_cutoff_log2"] > launch["log_n"]
        or instruction_input["storage_initialization"] != "minimal"
        or instruction_input["native_primer"] != "async"
    ):
        raise ValueError("architecture baseline InstructionInput launch is invalid")

    samples = evidence["samples"]
    sample_fields = {
        "pair",
        "order",
        "cpu_service_ns",
        "metal_service_ns",
        "service_speedup",
        "cpu_piop_ns",
        "metal_piop_ns",
        "piop_speedup",
        "cpu_piop_plus_prepare_ns",
        "metal_piop_plus_prepare_ns",
        "piop_plus_prepare_speedup",
    }
    if not isinstance(samples, list) or len(samples) != pairs:
        raise ValueError("architecture baseline samples are incomplete")
    timing_fields = {
        "service_speedup": ("cpu_service_ns", "metal_service_ns"),
        "piop_speedup": ("cpu_piop_ns", "metal_piop_ns"),
        "piop_plus_prepare_speedup": (
            "cpu_piop_plus_prepare_ns",
            "metal_piop_plus_prepare_ns",
        ),
    }
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != sample_fields:
            raise ValueError("architecture baseline sample fields are incomplete")
        if (
            type(sample["pair"]) is not int
            or sample["pair"] != index + 1
            or sample["order"] != orders[index]
        ):
            raise ValueError("architecture baseline sample order is invalid")
        for cpu_name, metal_name in timing_fields.values():
            if (
                type(sample[cpu_name]) is not int
                or sample[cpu_name] <= 0
                or type(sample[metal_name]) is not int
                or sample[metal_name] <= 0
            ):
                raise ValueError("architecture baseline sample timing is invalid")
        if (
            sample["cpu_service_ns"] > sample["cpu_piop_ns"]
            or sample["metal_service_ns"] > sample["metal_piop_ns"]
            or sample["cpu_piop_plus_prepare_ns"] <= sample["cpu_piop_ns"]
            or sample["metal_piop_plus_prepare_ns"] <= sample["metal_piop_ns"]
        ):
            raise ValueError("architecture baseline sample timing relationship is invalid")
        for ratio_name, (cpu_name, metal_name) in timing_fields.items():
            ratio = sample[ratio_name]
            expected_ratio = sample[cpu_name] / sample[metal_name]
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(ratio)
                or not math.isclose(
                    float(ratio), expected_ratio, rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise ValueError("architecture baseline sample ratio is invalid")

    decision = evidence["decision"]
    if not isinstance(decision, dict):
        raise ValueError("architecture baseline decision is incomplete")
    minimum_speedup = decision.get("minimum_speedup")
    minimum_pairs = decision.get("minimum_pairs")
    if (
        isinstance(minimum_speedup, bool)
        or not isinstance(minimum_speedup, (int, float))
        or not math.isfinite(minimum_speedup)
        or minimum_speedup <= 1.0
        or type(minimum_pairs) is not int
        or minimum_pairs < 1
    ):
        raise ValueError("architecture baseline decision threshold is invalid")
    cpu_service = [sample["cpu_service_ns"] for sample in samples]
    metal_service = [sample["metal_service_ns"] for sample in samples]
    service_speedups = [cpu / metal for cpu, metal in zip(cpu_service, metal_service)]
    improvements = [1.0 - metal / cpu for cpu, metal in zip(cpu_service, metal_service)]
    speedup_median = statistics.median(service_speedups)
    improvement_median = statistics.median(improvements)
    improvement_mad = statistics.median(
        abs(value - improvement_median) for value in improvements
    )
    cpu_median = statistics.median(cpu_service)
    metal_median = statistics.median(metal_service)
    optimized_first = [
        speedup
        for sample, speedup in zip(samples, service_speedups)
        if sample["order"] == ["optimized", "metal"]
    ]
    metal_first = [
        speedup
        for sample, speedup in zip(samples, service_speedups)
        if sample["order"] == ["metal", "optimized"]
    ]
    optimized_first_median = statistics.median(optimized_first)
    metal_first_median = statistics.median(metal_first)
    clears_speedup = speedup_median >= minimum_speedup
    clears_order_strata = (
        optimized_first_median >= minimum_speedup
        and metal_first_median >= minimum_speedup
    )
    clears_noise = improvement_median > 3.0 * improvement_mad
    clears_fraction = improvement_median >= 1.0 - 1.0 / minimum_speedup
    expected_decision = {
        "minimum_speedup": float(minimum_speedup),
        "minimum_pairs": minimum_pairs,
        "median_speedup": speedup_median,
        "optimized_first_median_speedup": optimized_first_median,
        "metal_first_median_speedup": metal_first_median,
        "cpu_service_ms_median": cpu_median / 1e6,
        "cpu_service_ms_mad": statistics.median(
            abs(value - cpu_median) for value in cpu_service
        )
        / 1e6,
        "metal_service_ms_median": metal_median / 1e6,
        "metal_service_ms_mad": statistics.median(
            abs(value - metal_median) for value in metal_service
        )
        / 1e6,
        "median_fractional_improvement": improvement_median,
        "mad_fractional_improvement": improvement_mad,
        "clears_speedup": clears_speedup,
        "clears_order_strata": clears_order_strata,
        "clears_noise": clears_noise,
        "clears_fractional_improvement": clears_fraction,
        "clears": len(samples) >= minimum_pairs
        and clears_speedup
        and clears_order_strata
        and clears_noise
        and clears_fraction
        and metal_median < cpu_median,
        "piop_speedup": statistics.median(
            sample["cpu_piop_ns"] / sample["metal_piop_ns"] for sample in samples
        ),
        "piop_plus_prepare_speedup": statistics.median(
            sample["cpu_piop_plus_prepare_ns"]
            / sample["metal_piop_plus_prepare_ns"]
            for sample in samples
        ),
    }
    if not decisions_match(decision, expected_decision):
        raise ValueError("architecture baseline decision disagrees with its samples")
    if decision["clears"] is not False:
        raise ValueError("architecture baseline must record a failed local gate")

    guards = evidence["guards"]
    if (
        not isinstance(guards, dict)
        or set(guards) != INSTRUCTION_INPUT_ARCHITECTURE_EVIDENCE_GUARDS
        or any(type(value) is not bool for value in guards.values())
        or any(
            value is not True
            for name, value in guards.items()
            if name != "instruction_input_local_gate"
        )
        or guards["instruction_input_local_gate"] is not False
    ):
        raise ValueError("architecture baseline guards are invalid")

    artifact = evidence["artifact"]
    if not isinstance(artifact, dict) or set(artifact) != {
        "result_path",
        "result_sha256",
        "source_schema_version",
    }:
        raise ValueError("architecture baseline artifact is incomplete")
    result_path = Path(artifact["result_path"]) if type(artifact["result_path"]) is str else None
    if (
        result_path is None
        or result_path.is_absolute()
        or ".." in result_path.parts
        or result_path.name != "result.json"
        or type(artifact["result_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", artifact["result_sha256"]) is None
        or type(artifact["source_schema_version"]) is not int
        or artifact["source_schema_version"] != 6
        or evidence["reason"] != "the fixed InstructionInput 4x local gate failed"
        or evidence["disposition"]
        != "retained only as the compact-row phase infrastructure baseline"
    ):
        raise ValueError("architecture baseline artifact is invalid")

    gate = template["final_validation"]["production_gate"]
    command = gate["evaluator"]["command"]
    try:
        if any(command.count(flag) != 1 for flag in ("--workload", "--log-n", "--repeats")):
            raise ValueError("architecture command bindings must be unique")
        command_workload = command[command.index("--workload") + 1]
        command_log_n = int(command[command.index("--log-n") + 1])
        command_repeats = int(command[command.index("--repeats") + 1])
        baseline = template["baseline_params"]
        expected_instruction_input = {
            "native_message_threads": int(
                baseline["JOLT_METAL_INSTRUCTION_INPUT_NATIVE_MESSAGE_THREADS"]
            ),
            "native_transition_threads": int(
                baseline["JOLT_METAL_INSTRUCTION_INPUT_NATIVE_TRANSITION_THREADS"]
            ),
            "dense_transition_threads": int(
                baseline["JOLT_METAL_INSTRUCTION_INPUT_DENSE_TRANSITION_THREADS"]
            ),
            "cutoff_log2": int(
                baseline["JOLT_METAL_INSTRUCTION_INPUT_CUTOFF_LOG2"]
            ),
            "trace_cutoff_log2": int(
                baseline["JOLT_METAL_INSTRUCTION_INPUT_TRACE_CUTOFF_LOG2"]
            ),
        }
    except (KeyError, ValueError, IndexError) as error:
        raise ValueError(
            "InstructionInput template cannot bind its architecture baseline"
        ) from error
    observed_instruction_input = {
        name: instruction_input[name] for name in expected_instruction_input
    }
    primary_log_n = template["final_validation"]["primary_log_n"]
    if (
        launch["workload"] != command_workload
        or launch["workload"] != gate["workload"]
        or launch["log_n"] != command_log_n
        or launch["log_n"] != primary_log_n
        or launch["log_n"] != gate["minimum_log_n"]
        or pairs != command_repeats
        or pairs != gate["minimum_pairs"]
        or minimum_pairs != gate["minimum_pairs"]
        or not math.isclose(
            float(minimum_speedup),
            float(gate["minimum_local_speedup"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or observed_instruction_input != expected_instruction_input
    ):
        raise ValueError("architecture baseline diverges from the template contract")


def validate_template(template: dict[str, Any], root: Optional[Path] = None) -> None:
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
        "baseline_params",
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
    architecture_phase = template.get("architecture_phase")
    if template.get("kernel") == "instruction_input" and architecture_phase is None:
        raise ValueError("InstructionInput template requires an architecture baseline")
    if architecture_phase is not None:
        architecture_fields = {
            "baseline_evidence",
            "baseline_evidence_sha256",
            "compact_row_bytes",
            "residual_row_bytes",
            "stage1_combined_row_bytes",
            "status",
        }
        if not isinstance(architecture_phase, dict) or set(architecture_phase) != architecture_fields:
            raise ValueError("architecture_phase must be an object")
        baseline_evidence = architecture_phase.get("baseline_evidence")
        if not isinstance(baseline_evidence, str) or not baseline_evidence:
            raise ValueError("architecture_phase must name its baseline evidence")
        if baseline_evidence not in frozen:
            raise ValueError("architecture baseline evidence must be frozen")
        relative_evidence_path = Path(baseline_evidence)
        if relative_evidence_path.is_absolute() or ".." in relative_evidence_path.parts:
            raise ValueError("architecture baseline evidence path must stay within the root")
        validation_root = (root or Path(__file__).resolve().parents[1]).resolve()
        evidence_path = (validation_root / relative_evidence_path).resolve()
        if validation_root not in evidence_path.parents:
            raise ValueError("architecture baseline evidence path must stay within the root")
        if not evidence_path.is_file():
            raise ValueError("architecture baseline evidence does not exist")
        evidence_digest = architecture_phase["baseline_evidence_sha256"]
        if (
            not isinstance(evidence_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", evidence_digest) is None
            or sha256(evidence_path.read_bytes()) != evidence_digest
        ):
            raise ValueError("architecture baseline evidence digest does not match")
        if (
            architecture_phase["compact_row_bytes"] != 48
            or architecture_phase["residual_row_bytes"] != 112
            or architecture_phase["stage1_combined_row_bytes"] != 160
            or architecture_phase["compact_row_bytes"]
            + architecture_phase["residual_row_bytes"]
            != architecture_phase["stage1_combined_row_bytes"]
            or architecture_phase["status"] != "frozen before shader search"
        ):
            raise ValueError("architecture phase geometry is invalid")
        evidence = read_json(evidence_path)
        validate_instruction_input_architecture_evidence(evidence, template)
    search_space = template["search_space"]
    if set(template["baseline_params"]) != set(search_space):
        raise ValueError("baseline parameters must close the search space")
    if any(not isinstance(values, list) or not values for values in search_space.values()):
        raise ValueError("every search parameter must have at least one allowed value")
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
    evaluator = template["evaluator"]
    if EVALUATOR_LOCK_HELD_ENV in evaluator.get("env", {}):
        raise ValueError("the local evaluator environment cannot override the lock token")
    result_contract = evaluator.get("result_contract")
    result_schema_version = evaluator.get("result_schema_version", SCHEMA_VERSION)
    if type(result_schema_version) is not int or result_schema_version < 1:
        raise ValueError("the local evaluator result schema version must be positive")
    if (
        template["kernel"] == "bytecode_read_raf_cycle"
        and result_contract != "bytecode_read_raf_cycle_v1"
    ):
        raise ValueError("the Bytecode evaluator requires its closed result contract")
    if (
        template["kernel"] == "instruction_input"
        and result_contract != "instruction_input_v4"
    ):
        raise ValueError("the InstructionInput evaluator requires its closed result contract")
    if result_contract is not None and result_contract not in LOCAL_RESULT_CONTRACTS:
        raise ValueError("the local evaluator names an unknown result contract")
    if (
        result_contract is not None
        and result_schema_version != LOCAL_RESULT_SCHEMA_VERSIONS[result_contract]
    ):
        raise ValueError("the local evaluator result schema version mismatches its contract")
    if result_contract == "bytecode_read_raf_cycle_v1":
        if template["kernel"] != "bytecode_read_raf_cycle":
            raise ValueError("the Bytecode result contract requires the Bytecode kernel")
        missing_env = sorted(
            set(BYTECODE_LOCAL_FINGERPRINT_ENV.values()) - set(evaluator.get("env", {}))
        )
        if missing_env:
            raise ValueError(
                f"the Bytecode result contract is missing evaluator environment: {missing_env}"
            )
        required_params = set(BYTECODE_LOCAL_FINGERPRINT_PARAMETERS.values())
        if required_params - set(template["search_space"]) or required_params - set(
            template.get("baseline_params", {})
        ):
            raise ValueError(
                "the Bytecode result contract requires every launch parameter in the search space and baseline"
            )
    if result_contract in {
        "instruction_input_v2",
        "instruction_input_v3",
        "instruction_input_v4",
    }:
        if template["kernel"] != "instruction_input":
            raise ValueError(
                "the InstructionInput result contract requires the InstructionInput kernel"
            )
        required_env = set(INSTRUCTION_INPUT_LOCAL_FINGERPRINT_ENV.values())
        if result_contract in {"instruction_input_v3", "instruction_input_v4"}:
            required_env.update(INSTRUCTION_INPUT_V3_FINGERPRINT_ENV.values())
        missing_env = sorted(required_env - set(evaluator.get("env", {})))
        if missing_env:
            raise ValueError(
                "the InstructionInput result contract is missing evaluator environment: "
                f"{missing_env}"
            )
        try:
            evaluator_repeats = int(evaluator["env"]["JOLT_METAL_EVAL_REPEATS"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "the InstructionInput evaluator requires an integer repeat count"
            ) from error
        if evaluator_repeats < 5 or evaluator_repeats % 2 == 0:
            raise ValueError(
                "the InstructionInput evaluator requires at least five odd paired repeats"
            )
        if result_contract in {"instruction_input_v3", "instruction_input_v4"}:
            try:
                cpu_reference_ns = int(
                    evaluator["env"]["JOLT_METAL_EVAL_CPU_REFERENCE_NS"]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "the InstructionInput evaluator requires an integer CPU reference"
                ) from error
            if cpu_reference_ns <= 0:
                raise ValueError(
                    "the InstructionInput evaluator CPU reference must be positive"
                )
            if cpu_reference_ns != INSTRUCTION_INPUT_V3_CPU_REFERENCE_NS:
                raise ValueError(
                    "the InstructionInput evaluator CPU reference must match the frozen a2 baseline"
                )
            metric = template.get("metric", {})
            if metric.get("name") != "frozen_cpu_reference_ratio":
                raise ValueError(
                    "the InstructionInput evaluator requires its drift-free primary metric"
                )
            if (
                metric.get("role") != "search_proxy"
                or metric.get("target") is not None
                or metric.get("unit") != "normalized_ratio"
            ):
                raise ValueError(
                    "the InstructionInput evaluator metric must remain a relative-only search proxy"
                )
        required_params = set(INSTRUCTION_INPUT_LOCAL_FINGERPRINT_PARAMETERS.values())
        if required_params - set(template["search_space"]) or required_params - set(
            template.get("baseline_params", {})
        ):
            raise ValueError(
                "the InstructionInput result contract requires every launch parameter in the search space and baseline"
            )
        final_validation = template.get("final_validation")
        if not isinstance(final_validation, dict) or set(final_validation) != {
            "primary_log_n",
            "production_gate",
        }:
            raise ValueError(
                "the InstructionInput final-validation contract contains inert checks"
            )
        if final_validation["primary_log_n"] != int(
            evaluator["env"]["JOLT_METAL_EVAL_LOG_N"]
        ):
            raise ValueError(
                "the InstructionInput final-validation contract targets the wrong scale"
            )
        if template.get("scope", {}).get("editable") != [
            "crates/jolt-kernels/src/metal/solinas/instruction_input.metal"
        ]:
            raise ValueError(
                "the InstructionInput search scope must remain shader-only"
            )
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
        if result_schema not in {4, 5, 6, 7}:
            raise ValueError("production evaluator schema must be 4, 5, 6, or 7")
        local_kernel = gate.get("local_kernel")
        if local_kernel is not None and result_schema not in {5, 6, 7}:
            raise ValueError("named local-kernel production gates require schema 5, 6, or 7")
        if result_schema in {5, 6, 7} and local_kernel not in PRODUCTION_LOCAL_KERNELS:
            raise ValueError(
                "schema-5/6/7 production gates require a known local kernel"
            )
        if local_kernel is not None:
            descriptor = PRODUCTION_LOCAL_KERNELS.get(local_kernel)
            if descriptor is None:
                raise ValueError("production gate names an unknown local kernel")
            if gate.get("metric") != descriptor["metric"]:
                raise ValueError("production scalar metric does not match the local kernel")
            if gate.get("paired_metric") != descriptor["paired_metric"]:
                raise ValueError("production paired metric does not match the local kernel")
            required_guards = set(descriptor["required_guards"])
            if result_schema >= 6:
                required_guards.update(descriptor.get("schema6_required_guards", set()))
            if result_schema >= 7:
                required_guards.update(descriptor.get("schema7_required_guards", set()))
            missing_guards = sorted(
                required_guards - set(gate.get("required_guards", []))
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


def validate_new_run_template(template: dict[str, Any]) -> None:
    if template.get("kernel") != "instruction_input":
        return
    if not isinstance(template.get("architecture_phase"), dict):
        raise ValueError("new InstructionInput runs require an architecture baseline")
    result_schema = template["final_validation"]["production_gate"]["evaluator"].get(
        "schema_version", 4
    )
    if result_schema != CURRENT_PIOP_RESULT_SCHEMA:
        raise ValueError(
            "new InstructionInput runs require the current production result schema"
        )


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
    result_schema_version = config["evaluator"].get(
        "result_schema_version", SCHEMA_VERSION
    )
    if type(result_schema_version) is not int or result_schema_version < 1:
        raise ValueError("evaluator result schema version is invalid")
    output = parse_unique_schema_result(result.stdout, result_schema_version)
    if output.get("kernel") != config["kernel"]:
        raise ValueError("evaluator returned the wrong kernel")
    validate_local_result_contract(config, output, params)
    metric = output.get("metrics", {}).get(config["metric"]["name"])
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric):
        raise ValueError("evaluator returned a non-finite primary metric")
    return output, elapsed


def positive_integer_samples(
    metrics: dict[str, Any], name: str, repeats: int
) -> list[int]:
    samples = metrics.get(name)
    if (
        not isinstance(samples, list)
        or len(samples) != repeats
        or any(type(value) is not int or value <= 0 for value in samples)
    ):
        raise ValueError(f"Bytecode evaluator {name} samples are invalid")
    return samples


def validate_bytecode_phase_sample(
    sample: Any, log_n: int, cutoff_log2: int
) -> None:
    prefix = "MetalBytecodeReadRafCycle::"
    expected_counts = {
        f"{prefix}allocation_plan": 1,
        f"{prefix}cpu_tail": cutoff_log2,
        f"{prefix}dense_round": log_n - cutoff_log2 - 1,
        f"{prefix}first_bind": 1,
        f"{prefix}first_message": 1,
        f"{prefix}prepare": 1,
        f"{prefix}readback": 1,
    }
    if not isinstance(sample, dict) or set(sample) != {
        "counts",
        "allocation",
        "readback",
    }:
        raise ValueError("Bytecode evaluator phase schedule is incomplete")
    if sample.get("counts") != expected_counts:
        raise ValueError("Bytecode evaluator phase schedule is invalid")

    allocation = sample.get("allocation")
    allocation_fields = {
        "current_device_bytes",
        "device_buffers",
        "planned_device_bytes",
        "recommended_device_bytes",
    }
    if not isinstance(allocation, dict) or set(allocation) != allocation_fields:
        raise ValueError("Bytecode evaluator phase schedule has no allocation plan")
    if any(type(allocation[name]) is not int or allocation[name] < 0 for name in allocation_fields):
        raise ValueError("Bytecode evaluator phase schedule has invalid allocation values")
    if (
        allocation["device_buffers"] != 17
        or allocation["planned_device_bytes"] <= 0
        or allocation["recommended_device_bytes"] <= 0
        or allocation["current_device_bytes"]
        + allocation["planned_device_bytes"]
        > allocation["recommended_device_bytes"]
    ):
        raise ValueError("Bytecode evaluator phase schedule violates device admission")

    expected_readback = {"bytes": 5 * (1 << cutoff_log2) * 16}
    if sample.get("readback") != expected_readback:
        raise ValueError("Bytecode evaluator phase schedule has the wrong readback")


def validate_instruction_input_local_result(
    config: dict[str, Any], output: dict[str, Any], params: dict[str, str]
) -> None:
    result_contract = config["evaluator"].get("result_contract")
    if result_contract not in {
        "instruction_input_v2",
        "instruction_input_v3",
        "instruction_input_v4",
    }:
        raise ValueError("InstructionInput evaluator has an unknown result contract")
    has_frozen_cpu_reference = result_contract in {
        "instruction_input_v3",
        "instruction_input_v4",
    }
    compact_row_contract = result_contract == "instruction_input_v4"
    expected_schema = result_contract
    expected_schema_version = LOCAL_RESULT_SCHEMA_VERSIONS[result_contract]
    top_fields = {
        "schema",
        "schema_version",
        "kernel",
        "metrics",
        "timings",
        "guards",
        "resources",
        "workload",
        "pipelines",
        "fingerprint",
    }
    if (
        set(output) != top_fields
        or output.get("schema") != expected_schema
        or output.get("schema_version") != expected_schema_version
        or output.get("kernel") != "instruction_input"
    ):
        raise ValueError("InstructionInput evaluator result violates its top-level schema")

    environment = config["evaluator"].get("env", {})
    try:
        expected = {
            field: int(environment[name])
            for field, name in INSTRUCTION_INPUT_LOCAL_FINGERPRINT_ENV.items()
        }
        if has_frozen_cpu_reference:
            expected.update(
                {
                    field: int(environment[name])
                    for field, name in INSTRUCTION_INPUT_V3_FINGERPRINT_ENV.items()
                }
            )
        expected.update(
            {
                field: int(params[name])
                for field, name in INSTRUCTION_INPUT_LOCAL_FINGERPRINT_PARAMETERS.items()
            }
        )
    except KeyError as error:
        raise ValueError(
            f"InstructionInput evaluator parameters are missing {error.args[0]}"
        ) from error
    if (
        has_frozen_cpu_reference
        and expected["frozen_cpu_reference_ns"]
        != INSTRUCTION_INPUT_V3_CPU_REFERENCE_NS
    ):
        raise ValueError("InstructionInput evaluator CPU reference is not the frozen a2 value")
    repeats = expected["repeats"]
    if repeats < 5 or repeats % 2 == 0:
        raise ValueError("InstructionInput evaluator requires at least five odd repeats")
    rows = 1 << expected["log_n"]
    cutoff = 1 << expected["cutoff_log2"]
    fingerprint = output["fingerprint"]
    fingerprint_fields = {
        "device",
        "max_buffer_length",
        "recommended_max_working_set_size",
        "current_allocated_size",
        "cpu_threads",
        "log_n",
        "validation_log_n",
        "repeats",
        "seed",
        "cutoff_log2",
        "trace_cutoff_log2",
        "native_message_threads",
        "native_transition_threads",
        "dense_transition_threads",
        "arm_schedule",
        "process_model",
        "warmup_tape_index",
        "validation_full_sequence_metal_runs",
        "residency_warmup_runs",
        "timed_full_sequence_metal_runs",
        "evaluator_full_sequence_metal_runs",
        "protocol_seeds",
        "protocol_transcript_states",
    }
    if has_frozen_cpu_reference:
        fingerprint_fields.add("frozen_cpu_reference_ns")
    if not isinstance(fingerprint, dict) or set(fingerprint) != fingerprint_fields:
        raise ValueError("InstructionInput evaluator fingerprint is incomplete")
    for name, value in expected.items():
        if type(fingerprint.get(name)) is not int or fingerprint[name] != value:
            raise ValueError(f"InstructionInput evaluator fingerprint does not match {name}")
    if fingerprint["arm_schedule"] != [
        "cpu_batch",
        "excluded_full_metal_warmup",
        "metal_timed_batch",
    ]:
        raise ValueError("InstructionInput evaluator has the wrong phased schedule")
    if (
        fingerprint["process_model"] != "single_process_steady_state_search_proxy"
        or type(fingerprint["warmup_tape_index"]) is not int
        or fingerprint["warmup_tape_index"] != 0
        or type(fingerprint["validation_full_sequence_metal_runs"]) is not int
        or fingerprint["validation_full_sequence_metal_runs"] != 1
        or type(fingerprint["residency_warmup_runs"]) is not int
        or fingerprint["residency_warmup_runs"] != 1
        or type(fingerprint["timed_full_sequence_metal_runs"]) is not int
        or fingerprint["timed_full_sequence_metal_runs"] != repeats
        or type(fingerprint["evaluator_full_sequence_metal_runs"]) is not int
        or fingerprint["evaluator_full_sequence_metal_runs"] != repeats + 2
    ):
        raise ValueError("InstructionInput evaluator warmup fingerprint is invalid")
    protocol_seeds = [
        expected["seed"] ^ ((0x9E3779B97F4A7C15 * (index + 1)) & ((1 << 64) - 1))
        for index in range(repeats)
    ]
    if (
        fingerprint["protocol_seeds"] != protocol_seeds
        or len(set(protocol_seeds)) != repeats
    ):
        raise ValueError("InstructionInput evaluator protocol tapes are invalid")
    protocol_transcript_states = fingerprint["protocol_transcript_states"]
    if (
        not isinstance(protocol_transcript_states, list)
        or len(protocol_transcript_states) != repeats
        or any(
            not isinstance(state, list)
            or len(state) != 32
            or any(type(byte) is not int or not 0 <= byte <= 255 for byte in state)
            for state in protocol_transcript_states
        )
        or len({tuple(state) for state in protocol_transcript_states}) != repeats
    ):
        raise ValueError("InstructionInput evaluator transcript tapes are not distinct")
    if expected["trace_cutoff_log2"] > expected["log_n"]:
        raise ValueError("InstructionInput trace cutoff does not admit the target")
    if (
        not isinstance(fingerprint["device"], str)
        or not fingerprint["device"]
        or any(
            type(fingerprint[name]) is not int or fingerprint[name] <= 0
            for name in (
                "max_buffer_length",
                "recommended_max_working_set_size",
                "cpu_threads",
            )
        )
        or type(fingerprint["current_allocated_size"]) is not int
        or fingerprint["current_allocated_size"] < 0
    ):
        raise ValueError("InstructionInput evaluator machine fingerprint is invalid")
    phase_fingerprint = config.get("fingerprint", {}).get("evaluator")
    if has_frozen_cpu_reference and isinstance(phase_fingerprint, dict):
        for name in (
            "device",
            "cpu_threads",
            "max_buffer_length",
            "recommended_max_working_set_size",
        ):
            if fingerprint[name] != phase_fingerprint.get(name):
                raise ValueError(
                    f"InstructionInput evaluator phase machine diverged at {name}"
                )

    workload = output["workload"]
    workload_fields = {
        "log_n",
        "rows",
        "validation_log_n",
        "tables",
        "samples_per_round",
        "descriptor_fields_returned_by_gpu",
        "cpu_native_row_bytes",
        "cutoff_log2",
        "cutoff_elements",
        "trace_cutoff_log2",
        "trace_cutoff_elements",
        "native_message_threads",
        "native_transition_threads",
        "dense_transition_threads",
        "host_fiat_shamir",
        "primary_timing",
        "workload_preparation_in_primary_metric",
        "sequence_preparation_in_primary_metric",
        "resident_source_materialization_in_primary_metric",
        "residency_warmup_in_primary_metric",
        "residency_warmup_reuses_first_protocol_tape",
        "residency_warmup_runs",
        "host_readback_allocation_in_primary_metric",
        "protocol_tape_preparation_in_primary_metric",
        "protocol_tapes_per_process",
        "protocol_tape_derivation",
        "cpu_trials_run_while_resident_metal_sequence_is_allocated",
        "cpu_trials_run_before_resident_source_materialization",
        "cpu_control",
        "metal_control",
    }
    resident_row_width_field = (
        "resident_compact_row_bytes"
        if compact_row_contract
        else "resident_stage1_row_bytes"
    )
    workload_fields.add(resident_row_width_field)
    if has_frozen_cpu_reference:
        workload_fields.update(
            {
                "primary_metric",
                "frozen_cpu_reference_ns",
                "frozen_cpu_reference_provenance",
                "live_cpu_controls_in_primary_metric",
            }
        )
    if not isinstance(workload, dict) or set(workload) != workload_fields:
        raise ValueError("InstructionInput evaluator workload contract is incomplete")
    expected_workload = {
        "log_n": expected["log_n"],
        "rows": rows,
        "validation_log_n": expected["validation_log_n"],
        "tables": 8,
        "samples_per_round": 4,
        "descriptor_fields_returned_by_gpu": 3,
        "cpu_native_row_bytes": 48,
        resident_row_width_field: 48 if compact_row_contract else 160,
        "cutoff_log2": expected["cutoff_log2"],
        "cutoff_elements": cutoff,
        "trace_cutoff_log2": expected["trace_cutoff_log2"],
        "trace_cutoff_elements": 1 << expected["trace_cutoff_log2"],
        "native_message_threads": expected["native_message_threads"],
        "native_transition_threads": expected["native_transition_threads"],
        "dense_transition_threads": expected["dense_transition_threads"],
        "host_fiat_shamir": True,
        "primary_timing": "after one excluded full-sequence residency warmup: resident sequence reset plus Metal rounds, host Fiat-Shamir, one dense readback, and exact four-sample CPU tail",
        "workload_preparation_in_primary_metric": False,
        "sequence_preparation_in_primary_metric": False,
        "resident_source_materialization_in_primary_metric": False,
        "residency_warmup_in_primary_metric": False,
        "residency_warmup_reuses_first_protocol_tape": True,
        "residency_warmup_runs": 1,
        "host_readback_allocation_in_primary_metric": False,
        "protocol_tape_preparation_in_primary_metric": False,
        "protocol_tapes_per_process": repeats,
        "protocol_tape_derivation": "base_seed xor ((repeat + 1) * 0x9e3779b97f4a7c15 modulo 2^64)",
        "cpu_trials_run_while_resident_metal_sequence_is_allocated": False,
        "cpu_trials_run_before_resident_source_materialization": True,
        "cpu_control": "standalone row-stride and arithmetic mirror of OptimizedInstructionInputKernel",
        "metal_control": (
            "public InstructionInputSequence over resident compact InstructionInputRow storage"
            if compact_row_contract
            else "public InstructionInputSequence over resident SpartanOuterUniskipRow storage"
        ),
    }
    if has_frozen_cpu_reference:
        expected_workload.update(
            {
                "primary_metric": "timed complete-member throughput normalized by a frozen CPU reference",
                "frozen_cpu_reference_ns": expected["frozen_cpu_reference_ns"],
                "frozen_cpu_reference_provenance": INSTRUCTION_INPUT_V3_CPU_REFERENCE_PROVENANCE,
                "live_cpu_controls_in_primary_metric": False,
            }
        )
    if workload != expected_workload:
        raise ValueError("InstructionInput evaluator workload fingerprint diverged")

    metrics = output["metrics"]
    metric_fields = {
        "hybrid_speedup",
        "resident_speedup",
        "paired_hybrid_speedups",
        "paired_resident_speedups",
        "cpu_ns_samples",
        "hybrid_ns_samples",
        "resident_ns_samples",
        "cpu_million_rows_per_second",
        "hybrid_million_rows_per_second",
    }
    if has_frozen_cpu_reference:
        metric_fields.update(
            {
                "frozen_cpu_reference_ratio",
                "paired_frozen_cpu_reference_ratios",
            }
        )
    if not isinstance(metrics, dict) or set(metrics) != metric_fields:
        raise ValueError("InstructionInput evaluator metric record is incomplete")

    def integer_samples(record: dict[str, Any], name: str, allow_zero: bool = False) -> list[int]:
        values = record.get(name)
        minimum = 0 if allow_zero else 1
        if (
            not isinstance(values, list)
            or len(values) != repeats
            or any(type(value) is not int or value < minimum for value in values)
        ):
            raise ValueError(f"InstructionInput evaluator {name} samples are invalid")
        return values

    def integer_value(record: dict[str, Any], name: str, allow_zero: bool = False) -> int:
        value = record.get(name)
        minimum = 0 if allow_zero else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"InstructionInput evaluator {name} is invalid")
        return value

    cpu_samples = integer_samples(metrics, "cpu_ns_samples")
    hybrid_samples = integer_samples(metrics, "hybrid_ns_samples")
    resident_samples = integer_samples(metrics, "resident_ns_samples")
    paired = metrics["paired_hybrid_speedups"]
    resident_paired = metrics["paired_resident_speedups"]
    recomputed = [cpu / metal for cpu, metal in zip(cpu_samples, hybrid_samples)]
    recomputed_resident = [
        cpu / metal for cpu, metal in zip(cpu_samples, resident_samples)
    ]
    paired_records = [
        ("paired_hybrid_speedups", paired, recomputed),
        ("paired_resident_speedups", resident_paired, recomputed_resident),
    ]
    recomputed_reference: list[float] = []
    if has_frozen_cpu_reference:
        recomputed_reference = [
            expected["frozen_cpu_reference_ns"] / metal for metal in hybrid_samples
        ]
        paired_records.append(
            (
                "paired_frozen_cpu_reference_ratios",
                metrics["paired_frozen_cpu_reference_ratios"],
                recomputed_reference,
            )
        )
    for name, actual, wanted in paired_records:
        if (
            not isinstance(actual, list)
            or len(actual) != repeats
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not math.isclose(float(value), expected_value, rel_tol=1e-12)
                for value, expected_value in zip(actual, wanted)
            )
        ):
            raise ValueError(f"InstructionInput evaluator {name} are invalid")
    scalar_records = [
        ("hybrid_speedup", statistics.median(recomputed)),
        ("resident_speedup", statistics.median(recomputed_resident)),
        (
            "cpu_million_rows_per_second",
            rows / (statistics.median(cpu_samples) / 1e9) / 1e6,
        ),
        (
            "hybrid_million_rows_per_second",
            rows / (statistics.median(hybrid_samples) / 1e9) / 1e6,
        ),
    ]
    if has_frozen_cpu_reference:
        scalar_records.append(
            (
                "frozen_cpu_reference_ratio",
                statistics.median(recomputed_reference),
            )
        )
    for name, wanted in scalar_records:
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not math.isclose(float(value), wanted, rel_tol=1e-12)
        ):
            raise ValueError(f"InstructionInput evaluator {name} is invalid")

    timings = output["timings"]
    timing_fields = {
        "workload_and_protocol_preparation_seconds",
        "resident_source_sequence_upload_and_storage_preparation_seconds",
        "cpu_median_seconds",
        "hybrid_median_seconds",
        "resident_median_seconds",
        "sequence_reset_median_seconds",
        "gpu_dispatch_wall_median_seconds",
        "host_round_median_seconds",
        "readback_median_seconds",
        "cpu_tail_median_seconds",
        "timed_gpu_active_total_seconds",
        "evaluator_gpu_active_total_seconds",
        "validation_gpu_active_ns",
        "residency_warmup_wall_ns",
        "residency_warmup_resident_ns",
        "residency_warmup_reset_ns",
        "residency_warmup_gpu_dispatch_wall_ns",
        "residency_warmup_host_round_ns",
        "residency_warmup_readback_ns",
        "residency_warmup_cpu_tail_ns",
        "residency_warmup_gpu_active_ns",
        "residency_warmup_to_timed_gpu_active_ratio",
        "sequence_reset_ns_samples",
        "gpu_dispatch_wall_ns_samples",
        "host_round_ns_samples",
        "readback_ns_samples",
        "cpu_tail_ns_samples",
        "gpu_active_ns_samples",
        "repeats",
    }
    if not isinstance(timings, dict) or set(timings) != timing_fields:
        raise ValueError("InstructionInput evaluator timing record is incomplete")
    if timings["repeats"] != repeats:
        raise ValueError("InstructionInput evaluator repeat count diverged")
    reset_samples = integer_samples(timings, "sequence_reset_ns_samples", allow_zero=True)
    component_samples = {
        "gpu_dispatch_wall_median_seconds": integer_samples(
            timings, "gpu_dispatch_wall_ns_samples"
        ),
        "host_round_median_seconds": integer_samples(timings, "host_round_ns_samples"),
        "readback_median_seconds": integer_samples(timings, "readback_ns_samples"),
        "cpu_tail_median_seconds": integer_samples(timings, "cpu_tail_ns_samples"),
    }
    gpu_active_samples = integer_samples(timings, "gpu_active_ns_samples")
    gpu_wall_samples = integer_samples(timings, "gpu_dispatch_wall_ns_samples")
    if any(
        not 0 < active <= gpu_wall <= hybrid
        for active, gpu_wall, hybrid in zip(
            gpu_active_samples, gpu_wall_samples, hybrid_samples
        )
    ):
        raise ValueError("InstructionInput evaluator GPU timing is not reconciled")
    if any(
        resident != hybrid - reset or hybrid <= reset
        for resident, hybrid, reset in zip(
            resident_samples, hybrid_samples, reset_samples
        )
    ):
        raise ValueError("InstructionInput evaluator resident timing is not reconciled")
    if any(
        sum(samples[index] for samples in component_samples.values())
        > resident_samples[index]
        for index in range(repeats)
    ):
        raise ValueError("InstructionInput evaluator component timings exceed resident wall")
    median_samples = {
        "cpu_median_seconds": cpu_samples,
        "hybrid_median_seconds": hybrid_samples,
        "resident_median_seconds": resident_samples,
        "sequence_reset_median_seconds": reset_samples,
        **component_samples,
    }
    for name, samples in median_samples.items():
        value = timings[name]
        wanted = statistics.median(samples) / 1e9
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not math.isclose(float(value), wanted, rel_tol=1e-12, abs_tol=1e-15)
        ):
            raise ValueError(f"InstructionInput evaluator {name} is invalid")
    for name in (
        "workload_and_protocol_preparation_seconds",
        "resident_source_sequence_upload_and_storage_preparation_seconds",
    ):
        value = timings[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"InstructionInput evaluator {name} is invalid")
    gpu_active_total = timings["timed_gpu_active_total_seconds"]
    if (
        isinstance(gpu_active_total, bool)
        or not isinstance(gpu_active_total, (int, float))
        or not math.isclose(
            float(gpu_active_total), sum(gpu_active_samples) / 1e9, rel_tol=1e-12
        )
    ):
        raise ValueError("InstructionInput evaluator GPU-active total is invalid")
    validation_gpu_active = integer_value(timings, "validation_gpu_active_ns")
    warmup_wall = integer_value(timings, "residency_warmup_wall_ns")
    warmup_resident = integer_value(timings, "residency_warmup_resident_ns")
    warmup_reset = integer_value(
        timings, "residency_warmup_reset_ns", allow_zero=True
    )
    warmup_gpu_wall = integer_value(
        timings, "residency_warmup_gpu_dispatch_wall_ns"
    )
    warmup_host = integer_value(timings, "residency_warmup_host_round_ns")
    warmup_readback = integer_value(timings, "residency_warmup_readback_ns")
    warmup_tail = integer_value(timings, "residency_warmup_cpu_tail_ns")
    warmup_gpu_active = integer_value(timings, "residency_warmup_gpu_active_ns")
    if not (
        warmup_wall == warmup_reset + warmup_resident
        and warmup_gpu_wall + warmup_host + warmup_readback + warmup_tail
        <= warmup_resident
        and warmup_gpu_active <= warmup_gpu_wall
    ):
        raise ValueError("InstructionInput evaluator residency warmup timing is invalid")
    warmup_ratio = timings["residency_warmup_to_timed_gpu_active_ratio"]
    wanted_warmup_ratio = warmup_gpu_active / statistics.median(gpu_active_samples)
    if (
        isinstance(warmup_ratio, bool)
        or not isinstance(warmup_ratio, (int, float))
        or not math.isfinite(warmup_ratio)
        or not math.isclose(float(warmup_ratio), wanted_warmup_ratio, rel_tol=1e-12)
    ):
        raise ValueError("InstructionInput evaluator residency warmup ratio is invalid")
    evaluator_gpu_active_total = timings["evaluator_gpu_active_total_seconds"]
    expected_evaluator_gpu_active_total = (
        validation_gpu_active + warmup_gpu_active + sum(gpu_active_samples)
    ) / 1e9
    if (
        isinstance(evaluator_gpu_active_total, bool)
        or not isinstance(evaluator_gpu_active_total, (int, float))
        or not math.isclose(
            float(evaluator_gpu_active_total),
            expected_evaluator_gpu_active_total,
            rel_tol=1e-12,
        )
    ):
        raise ValueError("InstructionInput evaluator total GPU-active time is invalid")

    guards = output["guards"]
    guard_fields = {
        "exact_four_sample_q_evals",
        "exact_round_polynomials",
        "exact_host_fiat_shamir_challenges",
        "exact_round_schedule",
        "exact_cutoff_tables",
        "exact_final_eight_claims",
        "exact_final_sumcheck_claim",
        "exact_transcript_state",
        "exact_derived_eq_cycle",
        "exact_final_relation",
        "actual_optimized_cpu_validation_parity",
        "protocol_retarget_reuses_cpu_rows",
        "production_trace_cutoff_admits_target",
        "raw_timing_relations",
        "resident_rows_stable_across_reset",
        "static_device_buffer_identities_stable",
        "exactly_one_dense_readback",
        "host_readback_preallocated_before_primary_timer",
        "distinct_protocol_tapes",
        "round_device_buffer_allocations_zero",
        "host_fiat_shamir",
        "cpu_tail_uses_exact_four_samples",
        "exactly_one_excluded_residency_warmup",
        "all_exact",
    }
    if (
        not isinstance(guards, dict)
        or set(guards) != guard_fields
        or any(guards[name] is not True for name in guard_fields)
    ):
        raise ValueError("InstructionInput evaluator correctness guard failed")

    resources = output["resources"]
    resource_fields = {
        "gpu_seconds",
        "cpu_native_rows_bytes",
        "sequence_owned_working_storage_bytes",
        "cpu_phase_persistent_modeled_bytes",
        "cpu_first_dense_table_bytes",
        "cpu_bind_scratch_capacity_bytes",
        "cpu_trial_peak_modeled_bytes",
        "metal_phase_persistent_modeled_bytes",
        "hybrid_readback_plus_tail_table_capacity_bytes",
        "hybrid_cpu_tail_bind_scratch_capacity_bytes",
        "metal_warmup_and_trial_peak_modeled_bytes",
        "sequence_setup_peak_modeled_bytes",
        "evaluator_peak_modeled_bytes",
        "resident_source_host_copy_bytes_dropped_before_metal_trials",
        "setup_peak_increment_from_resident_source_copy_bytes",
        "cutoff_readback_bytes",
        "unified_memory_no_per_round_row_upload",
        "sequence_owned_storage_includes_dense_ping_pong_weights_and_reductions",
    }
    resident_rows_resource_field = (
        "resident_compact_rows_bytes"
        if compact_row_contract
        else "resident_stage1_rows_bytes"
    )
    resource_fields.add(resident_rows_resource_field)
    if not isinstance(resources, dict) or set(resources) != resource_fields:
        raise ValueError("InstructionInput evaluator resource record is incomplete")
    gpu_seconds = resources["gpu_seconds"]
    if (
        isinstance(gpu_seconds, bool)
        or not isinstance(gpu_seconds, (int, float))
        or not math.isfinite(gpu_seconds)
        or not math.isclose(
            float(gpu_seconds),
            expected_evaluator_gpu_active_total,
            rel_tol=1e-12,
        )
    ):
        raise ValueError("InstructionInput evaluator GPU resource timing is invalid")
    cpu_rows_bytes = workload["cpu_native_row_bytes"] * rows
    resident_rows_bytes = workload[resident_row_width_field] * rows
    resident_source_rows_bytes = 160 * rows if compact_row_contract else resident_rows_bytes
    sequence_bytes = resources["sequence_owned_working_storage_bytes"]
    if (
        type(sequence_bytes) is not int
        or sequence_bytes != instruction_input_sequence_storage_bytes(expected["log_n"])
    ):
        raise ValueError("InstructionInput evaluator sequence storage is invalid")
    cpu_first_dense_bytes = 8 * (rows // 2) * 16
    cpu_bind_scratch_bytes = (rows // 4) * 16
    hybrid_tail_bytes = 2 * 8 * cutoff * 16
    hybrid_tail_bind_scratch_bytes = (cutoff // 2) * 16
    metal_persistent_bytes = cpu_rows_bytes + resident_rows_bytes + sequence_bytes
    cpu_peak_bytes = cpu_rows_bytes + cpu_first_dense_bytes + cpu_bind_scratch_bytes
    metal_peak_bytes = (
        metal_persistent_bytes + hybrid_tail_bytes + hybrid_tail_bind_scratch_bytes
    )
    setup_peak_bytes = metal_persistent_bytes + resident_source_rows_bytes
    expected_resources = {
        "cpu_native_rows_bytes": cpu_rows_bytes,
        resident_rows_resource_field: resident_rows_bytes,
        "cpu_phase_persistent_modeled_bytes": cpu_rows_bytes,
        "cpu_first_dense_table_bytes": cpu_first_dense_bytes,
        "cpu_bind_scratch_capacity_bytes": cpu_bind_scratch_bytes,
        "cpu_trial_peak_modeled_bytes": cpu_peak_bytes,
        "metal_phase_persistent_modeled_bytes": metal_persistent_bytes,
        "hybrid_readback_plus_tail_table_capacity_bytes": hybrid_tail_bytes,
        "hybrid_cpu_tail_bind_scratch_capacity_bytes": hybrid_tail_bind_scratch_bytes,
        "metal_warmup_and_trial_peak_modeled_bytes": metal_peak_bytes,
        "sequence_setup_peak_modeled_bytes": setup_peak_bytes,
        "evaluator_peak_modeled_bytes": max(
            cpu_peak_bytes, metal_peak_bytes, setup_peak_bytes
        ),
        "resident_source_host_copy_bytes_dropped_before_metal_trials": resident_source_rows_bytes,
        "setup_peak_increment_from_resident_source_copy_bytes": resident_source_rows_bytes,
        "cutoff_readback_bytes": 8 * cutoff * 16,
    }
    for name, wanted in expected_resources.items():
        if type(resources[name]) is not int or resources[name] != wanted:
            raise ValueError(f"InstructionInput evaluator resource {name} is invalid")
    if (
        resources["unified_memory_no_per_round_row_upload"] is not True
        or resources[
            "sequence_owned_storage_includes_dense_ping_pong_weights_and_reductions"
        ]
        is not True
    ):
        raise ValueError("InstructionInput evaluator peak resource accounting is invalid")

    pipelines = output["pipelines"]
    pipeline_fields = {
        "native_message_execution_width",
        "native_message_max_threads",
        "native_transition_execution_width",
        "native_transition_max_threads",
        "dense_transition_execution_width",
        "dense_transition_max_threads",
    }
    if not isinstance(pipelines, dict) or set(pipelines) != pipeline_fields:
        raise ValueError("InstructionInput evaluator pipeline record is incomplete")
    for prefix, selected in (
        ("native_message", expected["native_message_threads"]),
        ("native_transition", expected["native_transition_threads"]),
        ("dense_transition", expected["dense_transition_threads"]),
    ):
        if (
            pipelines[f"{prefix}_execution_width"] != 32
            or type(pipelines[f"{prefix}_max_threads"]) is not int
            or pipelines[f"{prefix}_max_threads"] < selected
            or selected % 32 != 0
        ):
            raise ValueError(f"InstructionInput evaluator {prefix} pipeline is invalid")


def validate_local_result_contract(
    config: dict[str, Any], output: dict[str, Any], params: dict[str, str]
) -> None:
    contract = config["evaluator"].get("result_contract")
    if contract is None:
        return
    if contract in {
        "instruction_input_v2",
        "instruction_input_v3",
        "instruction_input_v4",
    }:
        validate_instruction_input_local_result(config, output, params)
        return
    if contract != "bytecode_read_raf_cycle_v1":
        raise ValueError("unknown local evaluator result contract")

    fingerprint = output.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("Bytecode evaluator result has no fingerprint")
    environment = config["evaluator"].get("env", {})
    expected = {
        field: int(environment[name])
        for field, name in BYTECODE_LOCAL_FINGERPRINT_ENV.items()
    }
    try:
        expected.update(
            {
                field: int(params[name])
                for field, name in BYTECODE_LOCAL_FINGERPRINT_PARAMETERS.items()
            }
        )
    except KeyError as error:
        raise ValueError(
            f"Bytecode evaluator parameters are missing {error.args[0]}"
        ) from error
    expected.update(
        trace_elements=1 << expected["log_n"],
        cutoff_elements=1 << expected["cutoff_log2"],
        trace_cutoff_elements=1 << expected["trace_cutoff_log2"],
    )
    for name, value in expected.items():
        if type(fingerprint.get(name)) is not int or fingerprint[name] != value:
            raise ValueError(f"Bytecode evaluator fingerprint does not match {name}")

    repeats = expected["repeats"]
    orders = [
        ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
        for index in range(repeats)
    ]
    if fingerprint.get("orders") != orders:
        raise ValueError("Bytecode evaluator fingerprint has the wrong backend orders")
    fixed_fingerprint = {
        "cpu_algebra": "q10",
        "entry_bytecode_index": 1,
        "fixture": "address-diverse TraceBackend in a full 8192-row program and padded cycle domain",
        "fixture_program_rows": 1 << 13,
        "fixture_trace_rows": min(1 << expected["log_n"], (1 << 13) - 1),
        "covers_high_ra_chunk": True,
        "fused_inc_fixture": "mixed rd and RAM signed deltas",
        "relation_variant": "full-program",
        "initial_claim": "independent direct cycle-domain sum",
        "primary_metric_includes_host_fs": True,
    }
    for name, value in fixed_fingerprint.items():
        if type(fingerprint.get(name)) is not type(value) or fingerprint[name] != value:
            raise ValueError(
                f"Bytecode evaluator fingerprint violates the {name} algorithm contract"
            )

    metrics = output.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Bytecode evaluator result has no metrics")
    paired = metrics.get("paired_speedups")
    cpu_rounds = metrics.get("cpu_round_ns_samples")
    metal_rounds = metrics.get("metal_round_ns_samples")
    for name, samples in (
        ("paired_speedups", paired),
        ("cpu_round_ns_samples", cpu_rounds),
        ("metal_round_ns_samples", metal_rounds),
        ("phase_samples", output.get("phase_samples")),
    ):
        if not isinstance(samples, list) or len(samples) != repeats:
            raise ValueError(
                f"Bytecode evaluator result has the wrong number of {name}"
            )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in paired
    ):
        raise ValueError("Bytecode evaluator paired speedups are invalid")
    if any(
        not isinstance(rounds, list)
        or len(rounds) != expected["log_n"]
        or any(type(value) is not int or value <= 0 for value in rounds)
        for rounds in [*cpu_rounds, *metal_rounds]
    ):
        raise ValueError("Bytecode evaluator round samples are invalid")

    component_names = (
        "cpu_member_ns_samples",
        "metal_member_ns_samples",
        "cpu_no_resident_member_ns_samples",
        "cpu_core_ns_samples",
        "metal_core_ns_samples",
        "cpu_prepare_ns_samples",
        "metal_prepare_ns_samples",
        "cpu_host_fs_ns_samples",
        "metal_host_fs_ns_samples",
    )
    timing = {
        name: positive_integer_samples(metrics, name, repeats)
        for name in component_names
    }
    cpu_samples = timing["cpu_member_ns_samples"]
    metal_samples = timing["metal_member_ns_samples"]
    cpu_controls = timing["cpu_no_resident_member_ns_samples"]
    for arm in ("cpu", "metal"):
        member = timing[f"{arm}_member_ns_samples"]
        prepare = timing[f"{arm}_prepare_ns_samples"]
        core = timing[f"{arm}_core_ns_samples"]
        host_fs = timing[f"{arm}_host_fs_ns_samples"]
        rounds = cpu_rounds if arm == "cpu" else metal_rounds
        if any(
            core_ns <= sum(round_samples)
            for core_ns, round_samples in zip(core, rounds)
        ):
            raise ValueError("Bytecode evaluator core timing has no finish/output residual")
        if any(
            total != prepare_ns + core_ns + host_fs_ns
            for total, prepare_ns, core_ns, host_fs_ns in zip(
                member, prepare, core, host_fs
            )
        ):
            raise ValueError("Bytecode evaluator member timing is not fully reconciled")

    recomputed = [cpu / metal for cpu, metal in zip(cpu_samples, metal_samples)]
    if any(
        not math.isclose(float(actual), expected_value, rel_tol=1e-12)
        for actual, expected_value in zip(paired, recomputed)
    ):
        raise ValueError("Bytecode evaluator paired speedups disagree with member samples")
    primary = metrics.get(config["metric"]["name"])
    if (
        isinstance(primary, bool)
        or not isinstance(primary, (int, float))
        or not math.isfinite(primary)
        or not math.isclose(float(primary), statistics.median(recomputed), rel_tol=1e-12)
    ):
        raise ValueError("Bytecode evaluator primary metric disagrees with its pairs")

    recomputed_mad = statistics.median(
        abs(value - statistics.median(recomputed)) for value in recomputed
    )
    reported_mad = metrics.get("paired_speedup_mad")
    if (
        isinstance(reported_mad, bool)
        or not isinstance(reported_mad, (int, float))
        or not math.isclose(float(reported_mad), recomputed_mad, rel_tol=1e-12)
    ):
        raise ValueError("Bytecode evaluator paired-speedup dispersion is invalid")

    kernel_only = [
        (cpu - cpu_fs) / (metal - metal_fs)
        for cpu, cpu_fs, metal, metal_fs in zip(
            cpu_samples,
            timing["cpu_host_fs_ns_samples"],
            metal_samples,
            timing["metal_host_fs_ns_samples"],
        )
    ]
    reported_kernel_pairs = metrics.get("kernel_only_paired_speedups")
    if (
        not isinstance(reported_kernel_pairs, list)
        or len(reported_kernel_pairs) != repeats
        or any(
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isclose(float(actual), expected_value, rel_tol=1e-12)
            for actual, expected_value in zip(reported_kernel_pairs, kernel_only)
        )
    ):
        raise ValueError("Bytecode evaluator kernel-only pairs are invalid")
    reported_kernel_median = metrics.get("kernel_only_hybrid_speedup")
    if (
        isinstance(reported_kernel_median, bool)
        or not isinstance(reported_kernel_median, (int, float))
        or not math.isclose(
            float(reported_kernel_median), statistics.median(kernel_only), rel_tol=1e-12
        )
    ):
        raise ValueError("Bytecode evaluator kernel-only median is invalid")

    for name, samples in (("cpu", cpu_samples), ("metal", metal_samples)):
        reported_ms = metrics.get(f"{name}_member_ms_median")
        expected_ms = statistics.median(samples) / 1e6
        if (
            isinstance(reported_ms, bool)
            or not isinstance(reported_ms, (int, float))
            or not math.isclose(float(reported_ms), expected_ms, rel_tol=1e-12)
        ):
            raise ValueError(f"Bytecode evaluator {name} member median is invalid")

    cpu_control_median = statistics.median(cpu_controls)
    paired_cpu_median = statistics.median(cpu_samples)
    denominator_ratio = max(cpu_control_median, paired_cpu_median) / min(
        cpu_control_median, paired_cpu_median
    )
    reported_ratio = metrics.get("cpu_denominator_ratio")
    guards = output.get("guards")
    stable = denominator_ratio <= 1.10
    if (
        isinstance(reported_ratio, bool)
        or not isinstance(reported_ratio, (int, float))
        or not math.isclose(float(reported_ratio), denominator_ratio, rel_tol=1e-12)
        or not isinstance(guards, dict)
        or guards.get("cpu_denominator_stable") is not stable
    ):
        raise ValueError("Bytecode evaluator CPU denominator claim is invalid")

    for sample in output["phase_samples"]:
        validate_bytecode_phase_sample(sample, expected["log_n"], expected["cutoff_log2"])
    if (
        guards.get("metal_backend_exercised") is not True
        or guards.get("exact_metal_schedule") is not True
    ):
        raise ValueError("Bytecode evaluator phase schedule guard is invalid")

    resources = output.get("resources")
    resource_fields = {
        "gpu_seconds",
        "metal_hybrid_wall_seconds",
        "input_claim_precompute_ns",
        "resident_upload_ns",
        "resident_row_bytes",
    }
    if not isinstance(resources, dict) or set(resources) != resource_fields:
        raise ValueError("Bytecode evaluator resource record is incomplete")
    metal_seconds = sum(metal_samples) / 1e9
    if any(
        isinstance(resources[name], bool)
        or not isinstance(resources[name], (int, float))
        or not math.isfinite(resources[name])
        or not math.isclose(float(resources[name]), metal_seconds, rel_tol=1e-12)
        for name in ("gpu_seconds", "metal_hybrid_wall_seconds")
    ):
        raise ValueError("Bytecode evaluator resource timing is invalid")
    if (
        type(resources["input_claim_precompute_ns"]) is not int
        or resources["input_claim_precompute_ns"] <= 0
        or type(resources["resident_upload_ns"]) is not int
        or resources["resident_upload_ns"] <= 0
        or resources["resident_row_bytes"] != 40 * expected["trace_elements"]
    ):
        raise ValueError("Bytecode evaluator resident-row fingerprint is invalid")


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


def validate_production_bytecode_member(
    member: Any, backend: str, log_n: int, cutoff_log2: int
) -> int:
    fields = {
        "prepare_ns",
        "rounds_ns",
        "rounds_total_ns",
        "finish_ns",
        "output_claims_ns",
        "member_ns",
        "outer_counts",
        "metal_counts",
        "resource_observation",
    }
    if not isinstance(member, dict) or set(member) != fields:
        raise ValueError("production Bytecode member record is incomplete")
    scalar_names = (
        "prepare_ns",
        "rounds_total_ns",
        "finish_ns",
        "output_claims_ns",
        "member_ns",
    )
    if any(type(member[name]) is not int or member[name] <= 0 for name in scalar_names):
        raise ValueError("production Bytecode member timing is invalid")
    rounds = member["rounds_ns"]
    if (
        not isinstance(rounds, list)
        or len(rounds) != log_n
        or any(type(value) is not int or value <= 0 for value in rounds)
        or member["rounds_total_ns"] != sum(rounds)
        or member["member_ns"]
        != member["prepare_ns"]
        + member["rounds_total_ns"]
        + member["finish_ns"]
        + member["output_claims_ns"]
    ):
        raise ValueError("production Bytecode member timing is not reconciled")
    if member["outer_counts"] != {
        "prepare": 1,
        "prove_round": log_n,
        "finish_rounds": 1,
        "output_claims": 1,
    }:
        raise ValueError("production Bytecode outer schedule is invalid")

    expected_metal_counts = {
        "prepare": 0,
        "allocation_plan": 0,
        "first_message": 0,
        "first_bind": 0,
        "dense_round": 0,
        "readback": 0,
        "cpu_tail": 0,
        "invalid_round": 0,
    }
    if backend == "metal":
        expected_metal_counts.update(
            {
                "prepare": 1,
                "allocation_plan": 1,
                "first_message": 1,
                "first_bind": 1,
                "dense_round": log_n - cutoff_log2 - 1,
                "readback": 1,
                "cpu_tail": cutoff_log2,
            }
        )
    if member["metal_counts"] != expected_metal_counts:
        raise ValueError("production Bytecode Metal schedule is invalid")

    observation = member["resource_observation"]
    if backend == "optimized":
        if observation is not None:
            raise ValueError("production optimized Bytecode arm has Metal resources")
    else:
        if not isinstance(observation, dict) or set(observation) != {
            "allocation",
            "readback_bytes",
        }:
            raise ValueError("production Bytecode Metal resource record is incomplete")
        allocation = observation["allocation"]
        allocation_fields = {
            "current_device_bytes",
            "device_buffers",
            "planned_device_bytes",
            "recommended_device_bytes",
        }
        if not isinstance(allocation, dict) or set(allocation) != allocation_fields:
            raise ValueError("production Bytecode Metal allocation record is incomplete")
        if any(
            type(allocation[name]) is not int or allocation[name] < 0
            for name in allocation_fields
        ):
            raise ValueError("production Bytecode Metal allocation values are invalid")
        if (
            allocation["device_buffers"] != 17
            or allocation["planned_device_bytes"] <= 0
            or allocation["recommended_device_bytes"] <= 0
            or allocation["current_device_bytes"]
            + allocation["planned_device_bytes"]
            > allocation["recommended_device_bytes"]
            or type(observation["readback_bytes"]) is not int
            or observation["readback_bytes"] != 5 * (1 << cutoff_log2) * 16
        ):
            raise ValueError("production Bytecode Metal resource accounting is invalid")
    return member["member_ns"]


def validate_production_instruction_input_row_lifecycle(
    lifecycle: Any, backend: str, log_n: int, result_schema: int
) -> None:
    common_fields = {
        "kind",
        "rows",
        "row_bytes",
        "prepare_storage_id",
        "stage3_storage_id",
    }
    expected_fields = common_fields
    if backend == "metal":
        expected_fields |= {"stage1_storage_id"}
        if result_schema >= 7:
            expected_fields |= {"residual_storage_id", "row_production"}
    if not isinstance(lifecycle, dict) or set(lifecycle) != expected_fields:
        raise ValueError(
            "production InstructionInput row lifecycle record is incomplete"
        )
    integer_fields = expected_fields - {"kind", "row_production"}
    if any(
        type(lifecycle[name]) is not int or lifecycle[name] <= 0
        for name in integer_fields
    ):
        raise ValueError("production InstructionInput row lifecycle is invalid")
    if backend == "optimized":
        valid = (
            lifecycle["kind"] == "optimized_cpu"
            and lifecycle["rows"] == 1 << log_n
            and lifecycle["row_bytes"] == 48
            and lifecycle["prepare_storage_id"] == lifecycle["stage3_storage_id"]
        )
    elif result_schema < 7:
        valid = (
            lifecycle["kind"] == "metal_resident"
            and lifecycle["rows"] == 1 << log_n
            and lifecycle["row_bytes"] == 160
            and lifecycle["prepare_storage_id"]
            == lifecycle["stage1_storage_id"]
            == lifecycle["stage3_storage_id"]
        )
    else:
        row_count = 1 << log_n
        valid = (
            lifecycle["kind"] == "metal_compact_resident"
            and lifecycle["rows"] == row_count
            and lifecycle["row_bytes"] == 48
            and lifecycle["prepare_storage_id"]
            == lifecycle["stage1_storage_id"]
            == lifecycle["stage3_storage_id"]
            and lifecycle["residual_storage_id"]
            != lifecycle["prepare_storage_id"]
            and lifecycle["row_production"]
            == {
                "source_kind": "owned_random_access",
                "witness_row_extractions": row_count,
                "residual_rows_written": row_count,
                "compact_rows_written": row_count,
                "compact_row_bytes": 48,
                "residual_row_bytes": 112,
                "compact_allocations": 1,
                "residual_allocations": 1,
                "full_row_allocations": 0,
                "full_domain_copy_bytes": 0,
                "full_domain_copy_dispatches": 0,
                "host_repack_rows": 0,
            }
        )
    if not valid:
        raise ValueError("production InstructionInput row lifecycle is invalid")


def validate_production_instruction_input_member(
    member: Any,
    backend: str,
    log_n: int,
    cutoff_log2: int,
    result_schema: int,
) -> int:
    fields = {
        "prepare_ns",
        "rounds_ns",
        "rounds_total_ns",
        "finish_ns",
        "output_claims_ns",
        "member_ns",
        "outer_counts",
        "metal_counts",
        "resource_observation",
    }
    if result_schema >= 6:
        fields |= {"prefetch_submit_ns", "service_ns"}
    if not isinstance(member, dict) or set(member) != fields:
        raise ValueError("production InstructionInput member record is incomplete")
    scalar_names = (
        "prepare_ns",
        "rounds_total_ns",
        "finish_ns",
        "output_claims_ns",
        "member_ns",
    )
    if any(type(member[name]) is not int or member[name] <= 0 for name in scalar_names):
        raise ValueError("production InstructionInput member timing is invalid")
    if result_schema >= 6 and (
        type(member["prefetch_submit_ns"]) is not int
        or member["prefetch_submit_ns"] < 0
        or type(member["service_ns"]) is not int
        or member["service_ns"] <= 0
    ):
        raise ValueError("production InstructionInput service timing is invalid")
    rounds = member["rounds_ns"]
    if (
        not isinstance(rounds, list)
        or len(rounds) != log_n
        or any(type(value) is not int or value <= 0 for value in rounds)
        or member["rounds_total_ns"] != sum(rounds)
        or member["member_ns"]
        != member["prepare_ns"]
        + member["rounds_total_ns"]
        + member["finish_ns"]
        + member["output_claims_ns"]
        or (
            result_schema >= 6
            and member["service_ns"]
            != member["member_ns"] + member["prefetch_submit_ns"]
        )
    ):
        raise ValueError("production InstructionInput member timing is not reconciled")
    if member["outer_counts"] != {
        "prepare": 1,
        "prove_round": log_n,
        "finish_rounds": 1,
        "output_claims": 1,
    }:
        raise ValueError("production InstructionInput outer schedule is invalid")

    expected_metal_counts = {
        "storage_prepare": 0,
        "allocation_plan": 0,
        "prepare": 0,
        "first_message": 0,
        "first_bind": 0,
        "dense_round": 0,
        "readback": 0,
        "cpu_tail": 0,
    }
    if result_schema >= 6:
        expected_metal_counts.update(
            {
                "storage_initialize": 0,
                "storage_initialize_complete": 0,
                "native_primer_submit": 0,
                "native_primer_join": 0,
                "native_primer_complete": 0,
            }
        )
    if backend == "metal":
        expected_metal_counts.update(
            {
                "storage_prepare": 1,
                "allocation_plan": 1,
                "prepare": 1,
                "first_message": 1,
                "first_bind": 1,
                "dense_round": log_n - cutoff_log2 - 1,
                "readback": 1,
                "cpu_tail": cutoff_log2,
            }
        )
        if result_schema >= 6:
            expected_metal_counts.update(
                {
                    "storage_initialize": 1,
                    "storage_initialize_complete": 1,
                    "native_primer_submit": 1,
                    "native_primer_join": 1,
                    "native_primer_complete": 1,
                }
            )
    if member["metal_counts"] != expected_metal_counts:
        raise ValueError("production InstructionInput Metal schedule is invalid")

    observation = member["resource_observation"]
    if backend == "optimized":
        if observation is not None:
            raise ValueError("production optimized InstructionInput arm has Metal resources")
        if result_schema >= 6 and (
            member["prefetch_submit_ns"] != 0
            or member["service_ns"] != member["member_ns"]
        ):
            raise ValueError(
                "production optimized InstructionInput has Metal prefetch service"
            )
    else:
        observation_fields = {
            "allocation",
            "host_tail_bytes",
            "readback_bytes",
            "resident_rows_reused",
            "round_device_buffer_allocations",
        }
        if result_schema >= 6:
            observation_fields |= {"storage_initialization", "native_primer"}
        if not isinstance(observation, dict) or set(observation) != observation_fields:
            raise ValueError(
                "production InstructionInput Metal resource record is incomplete"
            )
        allocation = observation["allocation"]
        allocation_fields = {
            "current_device_bytes",
            "device_buffers",
            "planned_device_bytes",
            "recommended_device_bytes",
        }
        if not isinstance(allocation, dict) or set(allocation) != allocation_fields:
            raise ValueError(
                "production InstructionInput Metal allocation record is incomplete"
            )
        if any(
            type(allocation[name]) is not int or allocation[name] < 0
            for name in allocation_fields
        ):
            raise ValueError(
                "production InstructionInput Metal allocation values are invalid"
            )
        expected_tail_bytes = 8 * (1 << cutoff_log2) * 16
        expected_sequence_bytes = instruction_input_sequence_storage_bytes(log_n)
        expected_resident_row_bytes = 160 * (1 << log_n)
        if (
            allocation["device_buffers"] != 6
            or allocation["planned_device_bytes"] != expected_sequence_bytes
            or allocation["current_device_bytes"] < expected_resident_row_bytes
            or allocation["recommended_device_bytes"] <= 0
            or allocation["current_device_bytes"]
            + allocation["planned_device_bytes"]
            > allocation["recommended_device_bytes"]
            or type(observation["host_tail_bytes"]) is not int
            or observation["host_tail_bytes"] != expected_tail_bytes
            or type(observation["readback_bytes"]) is not int
            or observation["readback_bytes"] != expected_tail_bytes
            or observation["resident_rows_reused"] is not True
            or type(observation["round_device_buffer_allocations"]) is not int
            or observation["round_device_buffer_allocations"] != 0
        ):
            raise ValueError(
                "production InstructionInput Metal resource accounting is invalid"
            )
        if result_schema >= 6:
            initialization = observation["storage_initialization"]
            initialization_fields = {
                "mode",
                "device_buffers",
                "bytes",
                "protocol_dispatches",
                "buffer_identities",
                "gpu_active_ns",
                "wall_ns",
            }
            if (
                not isinstance(initialization, dict)
                or set(initialization) != initialization_fields
            ):
                raise ValueError(
                    "production InstructionInput initialization record is incomplete"
                )
            initialization_ids = initialization["buffer_identities"]
            if (
                initialization["mode"] != "minimal"
                or initialization["device_buffers"] != 6
                or initialization["bytes"] != 96
                or initialization["protocol_dispatches"] != 0
                or not isinstance(initialization_ids, list)
                or len(initialization_ids) != 6
                or any(type(value) is not int or value <= 0 for value in initialization_ids)
                or len(set(initialization_ids)) != 6
                or type(initialization["gpu_active_ns"]) is not int
                or initialization["gpu_active_ns"] <= 0
                or type(initialization["wall_ns"]) is not int
                or initialization["wall_ns"] <= 0
                or initialization["gpu_active_ns"] > initialization["wall_ns"]
            ):
                raise ValueError(
                    "production InstructionInput minimal initialization is invalid"
                )

            primer = observation["native_primer"]
            primer_fields = {
                "source_elements",
                "e_in_elements",
                "e_out_elements",
                "resident_rows_storage_id",
                "storage_buffer_identities",
                "command_committed",
                "protocol_state_advanced",
                "timings",
                "completed_before_join",
                "command_completed",
                "produced_zero",
            }
            if not isinstance(primer, dict) or set(primer) != primer_fields:
                raise ValueError(
                    "production InstructionInput native primer record is incomplete"
                )
            timings = primer["timings"]
            timing_fields = {
                "submit_wall_ns",
                "submit_span_wall_ns",
                "overlap_wall_ns",
                "join_wall_ns",
                "lifecycle_wall_ns",
                "gpu_active_ns",
            }
            if not isinstance(timings, dict) or set(timings) != timing_fields:
                raise ValueError(
                    "production InstructionInput native primer timing is incomplete"
                )
            if (
                primer["source_elements"] != 64
                or primer["e_in_elements"] != 1
                or primer["e_out_elements"] != 32
                or type(primer["resident_rows_storage_id"]) is not int
                or primer["resident_rows_storage_id"] <= 0
                or primer["storage_buffer_identities"] != initialization_ids
                or primer["command_committed"] is not True
                or primer["protocol_state_advanced"] is not False
                or primer["command_completed"] is not True
                or primer["produced_zero"] is not True
                or type(primer["completed_before_join"]) is not bool
                or any(type(timings[name]) is not int or timings[name] <= 0 for name in timing_fields)
                or timings["submit_wall_ns"]
                + timings["overlap_wall_ns"]
                + timings["join_wall_ns"]
                > timings["lifecycle_wall_ns"]
                or timings["gpu_active_ns"] > timings["lifecycle_wall_ns"]
                or timings["submit_wall_ns"] > timings["submit_span_wall_ns"] + 1
                or member["prefetch_submit_ns"] != timings["submit_span_wall_ns"]
                or timings["join_wall_ns"] > member["rounds_ns"][0]
            ):
                raise ValueError(
                    "production InstructionInput native primer record is invalid"
                )
    return member["service_ns"] if result_schema >= 6 else member["member_ns"]


def recompute_local_member_decision(
    pair_records: list[dict[str, Any]],
    cpu: list[int],
    metal: list[int],
    minimum_speedup: float,
    minimum_pairs: int,
) -> tuple[list[float], dict[str, Any]]:
    speedups = [cpu_ns / metal_ns for cpu_ns, metal_ns in zip(cpu, metal)]
    improvements = [
        1.0 - metal_ns / cpu_ns for cpu_ns, metal_ns in zip(cpu, metal)
    ]
    speedup_median = statistics.median(speedups)
    improvement_median = statistics.median(improvements)
    improvement_mad = statistics.median(
        abs(value - improvement_median) for value in improvements
    )
    cpu_median = statistics.median(cpu)
    metal_median = statistics.median(metal)
    optimized_first = [
        speedup
        for pair, speedup in zip(pair_records, speedups)
        if pair.get("order") == ["optimized", "metal"]
    ]
    metal_first = [
        speedup
        for pair, speedup in zip(pair_records, speedups)
        if pair.get("order") == ["metal", "optimized"]
    ]
    optimized_first_median = (
        statistics.median(optimized_first) if optimized_first else None
    )
    metal_first_median = statistics.median(metal_first) if metal_first else None
    enough_pairs = len(pair_records) >= minimum_pairs
    clears_speedup = speedup_median >= minimum_speedup
    clears_fraction = improvement_median >= 1.0 - 1.0 / minimum_speedup
    clears_noise = improvement_median > 3.0 * improvement_mad
    lower_median = metal_median < cpu_median
    clears_order_strata = (
        optimized_first_median is not None
        and metal_first_median is not None
        and optimized_first_median >= minimum_speedup
        and metal_first_median >= minimum_speedup
    )
    return improvements, {
        "minimum_speedup": minimum_speedup,
        "minimum_pairs": minimum_pairs,
        "median_speedup": speedup_median,
        "median_fractional_improvement": improvement_median,
        "mad_fractional_improvement": improvement_mad,
        "cpu_member_ms_median": cpu_median / 1e6,
        "cpu_member_ms_mad": statistics.median(
            abs(value - cpu_median) for value in cpu
        )
        / 1e6,
        "metal_member_ms_median": metal_median / 1e6,
        "metal_member_ms_mad": statistics.median(
            abs(value - metal_median) for value in metal
        )
        / 1e6,
        "enough_pairs": enough_pairs,
        "clears_speedup": clears_speedup,
        "clears_fractional_improvement": clears_fraction,
        "clears_noise": clears_noise,
        "lower_metal_median": lower_median,
        "optimized_first_median_speedup": optimized_first_median,
        "metal_first_median_speedup": metal_first_median,
        "clears_order_strata": clears_order_strata,
        "clears": enough_pairs
        and clears_speedup
        and clears_fraction
        and clears_noise
        and lower_median
        and clears_order_strata,
    }


def decisions_match(actual: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    for name, wanted in expected.items():
        got = actual[name]
        if isinstance(wanted, bool):
            if got is not wanted:
                return False
        elif isinstance(wanted, int):
            if type(got) is not int or got != wanted:
                return False
        elif wanted is None:
            if got is not None:
                return False
        elif (
            isinstance(got, bool)
            or not isinstance(got, (int, float))
            or not math.isfinite(got)
            or not math.isclose(float(got), wanted, rel_tol=1e-9, abs_tol=1e-9)
        ):
            return False
    return True


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
    pair_records = None
    if result_schema >= 5:
        pair_records = result.get("pairs")
        if not isinstance(pair_records, list) or len(pair_records) != len(pairs):
            raise ValueError("production result has incomplete raw PIOP pair records")
        raw_cpu_piop = []
        raw_metal_piop = []
        raw_cpu_prepare = []
        raw_metal_prepare = []
        for index, (record, reported_speedup) in enumerate(zip(pair_records, pairs)):
            expected_order = (
                ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
            )
            if (
                not isinstance(record, dict)
                or set(record) != {"index", "order", "arms"}
                or record.get("index") != index + 1
                or record.get("order") != expected_order
            ):
                raise ValueError("production raw PIOP pairs do not alternate correctly")
            arms = record.get("arms")
            if not isinstance(arms, dict) or set(arms) != {"optimized", "metal"}:
                raise ValueError("production raw PIOP pair has invalid arms")
            try:
                cpu_piop = arms["optimized"]["piop_ns"]
                metal_piop = arms["metal"]["piop_ns"]
            except (KeyError, TypeError) as error:
                raise ValueError("production raw PIOP pair is incomplete") from error
            if (
                type(cpu_piop) is not int
                or cpu_piop <= 0
                or type(metal_piop) is not int
                or metal_piop <= 0
                or not math.isclose(
                    float(reported_speedup), cpu_piop / metal_piop, rel_tol=1e-9
                )
            ):
                raise ValueError("production raw PIOP pair disagrees with its speedup")
            raw_cpu_piop.append(cpu_piop)
            raw_metal_piop.append(metal_piop)
            if result_schema >= 6:
                try:
                    cpu_prepare = arms["optimized"]["backend_witness_prepare_ns"]
                    metal_prepare = arms["metal"]["backend_witness_prepare_ns"]
                except (KeyError, TypeError) as error:
                    raise ValueError(
                        "production raw backend preparation pair is incomplete"
                    ) from error
                if (
                    type(cpu_prepare) is not int
                    or cpu_prepare <= 0
                    or type(metal_prepare) is not int
                    or metal_prepare <= 0
                ):
                    raise ValueError(
                        "production raw backend preparation timing is invalid"
                    )
                raw_cpu_prepare.append(cpu_prepare)
                raw_metal_prepare.append(metal_prepare)
        for name, raw_samples in (
            ("cpu_piop_ms_samples", raw_cpu_piop),
            ("metal_piop_ms_samples", raw_metal_piop),
        ):
            reported_samples = metrics.get(name)
            if (
                not isinstance(reported_samples, list)
                or len(reported_samples) != len(raw_samples)
                or any(
                    isinstance(reported, bool)
                    or not isinstance(reported, (int, float))
                    or not math.isfinite(reported)
                    or not math.isclose(
                        float(reported) * 1e6,
                        raw,
                        rel_tol=1e-12,
                        abs_tol=0.500001,
                    )
                    for reported, raw in zip(reported_samples, raw_samples)
                )
            ):
                raise ValueError("production PIOP sample summary is invalid")
        if result_schema >= 6:
            for name, raw_samples in (
                ("cpu_backend_witness_prepare_ms_samples", raw_cpu_prepare),
                ("metal_backend_witness_prepare_ms_samples", raw_metal_prepare),
            ):
                reported_samples = metrics.get(name)
                if (
                    not isinstance(reported_samples, list)
                    or len(reported_samples) != len(raw_samples)
                    or any(
                        isinstance(reported, bool)
                        or not isinstance(reported, (int, float))
                        or not math.isfinite(reported)
                        or not math.isclose(
                            float(reported) * 1e6,
                            raw,
                            rel_tol=1e-12,
                            abs_tol=0.500001,
                        )
                        for reported, raw in zip(reported_samples, raw_samples)
                    )
                ):
                    raise ValueError(
                        "production backend preparation sample summary is invalid"
                    )
            for name, raw_samples in (
                ("cpu_piop_ms", raw_cpu_piop),
                ("metal_piop_ms", raw_metal_piop),
                ("cpu_backend_witness_prepare_ms", raw_cpu_prepare),
                ("metal_backend_witness_prepare_ms", raw_metal_prepare),
            ):
                reported = metrics.get(name)
                expected = statistics.median(raw_samples) / 1e6
                if (
                    isinstance(reported, bool)
                    or not isinstance(reported, (int, float))
                    or not math.isfinite(reported)
                    or not math.isclose(
                        float(reported), expected, rel_tol=1e-12, abs_tol=0.500001e-6
                    )
                ):
                    raise ValueError(
                        "production PIOP/backend preparation median summary is invalid"
                    )
            paired_with_prepare = [
                (cpu_piop + cpu_prepare) / (metal_piop + metal_prepare)
                for cpu_piop, metal_piop, cpu_prepare, metal_prepare in zip(
                    raw_cpu_piop,
                    raw_metal_piop,
                    raw_cpu_prepare,
                    raw_metal_prepare,
                )
            ]
            reported_pairs_with_prepare = metrics.get(
                "paired_speedups_with_backend_witness_prepare"
            )
            reported_prepare_speedup = metrics.get(
                "piop_plus_backend_witness_prepare_speedup"
            )
            if (
                not isinstance(reported_pairs_with_prepare, list)
                or len(reported_pairs_with_prepare) != len(paired_with_prepare)
                or any(
                    isinstance(reported, bool)
                    or not isinstance(reported, (int, float))
                    or not math.isclose(
                        float(reported), expected, rel_tol=1e-12
                    )
                    for reported, expected in zip(
                        reported_pairs_with_prepare, paired_with_prepare
                    )
                )
                or isinstance(reported_prepare_speedup, bool)
                or not isinstance(reported_prepare_speedup, (int, float))
                or not math.isclose(
                    float(reported_prepare_speedup),
                    statistics.median(paired_with_prepare),
                    rel_tol=1e-12,
                )
            ):
                raise ValueError(
                    "production PIOP plus backend preparation summary is invalid"
                )
        resources = result.get("resources")
        reported_metal_seconds = (
            resources.get("metal_piop_seconds") if isinstance(resources, dict) else None
        )
        raw_metal_seconds = sum(raw_metal_piop) / 1e9
        if (
            isinstance(reported_metal_seconds, bool)
            or not isinstance(reported_metal_seconds, (int, float))
            or not math.isfinite(reported_metal_seconds)
            or not math.isclose(
                float(reported_metal_seconds),
                raw_metal_seconds,
                rel_tol=1e-9,
                abs_tol=len(raw_metal_piop) / 2e9 + 1e-12,
            )
        ):
            raise ValueError("production PIOP resource summary is invalid")
    optimized_first_median = None
    metal_first_median = None
    if local_kernel == "BytecodeReadRafCycle":
        bytecode_fingerprint = result.get("fingerprint")
        if not isinstance(bytecode_fingerprint, dict):
            raise ValueError("production Bytecode result has no fingerprint")
        log_n = bytecode_fingerprint.get("log_n")
        cutoff_log2 = bytecode_fingerprint.get("bytecode_metal_cutoff_log2")
        if (
            type(log_n) is not int
            or type(cutoff_log2) is not int
            or not 1 <= cutoff_log2 <= log_n - 2
        ):
            raise ValueError("production Bytecode result has invalid cycle geometry")
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
        if pair_records is None or len(pair_records) != len(local_pairs):
            raise ValueError("production Bytecode result has incomplete raw pair records")
        optimized_first_speedups = []
        metal_first_speedups = []
        for index, (record, local_speedup) in enumerate(zip(pair_records, local_pairs)):
            expected_order = (
                ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
            )
            if not isinstance(record, dict) or record.get("order") != expected_order:
                raise ValueError("production Bytecode raw pair order is invalid")
            arms = record.get("arms", {})
            try:
                cpu_record = arms["optimized"]["bytecode"]
                metal_record = arms["metal"]["bytecode"]
            except (KeyError, TypeError) as error:
                raise ValueError("production Bytecode raw pair is incomplete") from error
            cpu_member = validate_production_bytecode_member(
                cpu_record, "optimized", log_n, cutoff_log2
            )
            metal_member = validate_production_bytecode_member(
                metal_record, "metal", log_n, cutoff_log2
            )
            if (
                not math.isclose(
                    float(local_speedup), cpu_member / metal_member, rel_tol=1e-9
                )
            ):
                raise ValueError("production Bytecode raw pair disagrees with its speedup")
            (
                optimized_first_speedups
                if expected_order == ["optimized", "metal"]
                else metal_first_speedups
            ).append(float(local_speedup))
        optimized_first_median = statistics.median(optimized_first_speedups)
        metal_first_median = statistics.median(metal_first_speedups)
        minimum_speedup = float(gate["minimum_local_speedup"])
        if (
            optimized_first_median < minimum_speedup
            or metal_first_median < minimum_speedup
        ):
            raise ValueError("production Bytecode order stratum does not clear the local gate")
        decision_values = {
            "minimum_speedup": minimum_speedup,
            "median_speedup": float(metric),
            "optimized_first_median_speedup": optimized_first_median,
            "metal_first_median_speedup": metal_first_median,
        }
        for name, expected in decision_values.items():
            actual = decision.get(name)
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isclose(float(actual), expected, rel_tol=1e-12)
            ):
                raise ValueError(
                    f"production Bytecode decision disagrees with recomputed {name}"
                )
        if (
            decision.get("minimum_pairs") != int(gate["minimum_pairs"])
            or decision.get("clears_order_strata") is not True
        ):
            raise ValueError("production Bytecode decision has an invalid order-strata claim")
    if local_kernel == "InstructionInput":
        instruction_input_fingerprint = result.get("fingerprint")
        if not isinstance(instruction_input_fingerprint, dict):
            raise ValueError("production InstructionInput result has no fingerprint")
        log_n = instruction_input_fingerprint.get("log_n")
        cutoff_log2 = instruction_input_fingerprint.get(
            "instruction_input_metal_cutoff_log2"
        )
        if (
            type(log_n) is not int
            or type(cutoff_log2) is not int
            or not 1 <= cutoff_log2 <= log_n - 2
        ):
            raise ValueError("production InstructionInput result has invalid cycle geometry")
        if result_schema >= 6 and (
            instruction_input_fingerprint.get(
                "instruction_input_storage_initialization"
            )
            != "minimal"
            or instruction_input_fingerprint.get("instruction_input_native_primer")
            != "async"
        ):
            raise ValueError(
                "production InstructionInput result used the wrong startup controls"
            )
        decision = metrics.get("instruction_input_kernel_service_decision")
        if not isinstance(decision, dict) or decision.get("clears") is not True:
            raise ValueError(
                "production InstructionInput result did not clear its fixed local decision"
            )
        decision_speedup = decision.get("median_speedup")
        if not math.isclose(
            float(decision_speedup)
            if isinstance(decision_speedup, (int, float))
            and not isinstance(decision_speedup, bool)
            else math.nan,
            float(metric),
            rel_tol=1e-12,
        ):
            raise ValueError(
                "production InstructionInput decision disagrees with its scalar metric"
            )
        if pair_records is None or len(pair_records) != len(local_pairs):
            raise ValueError(
                "production InstructionInput result has incomplete raw pair records"
            )
        raw_cpu_members = []
        raw_metal_members = []
        for index, (record, local_speedup) in enumerate(zip(pair_records, local_pairs)):
            expected_order = (
                ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
            )
            if not isinstance(record, dict) or record.get("order") != expected_order:
                raise ValueError("production InstructionInput raw pair order is invalid")
            arms = record.get("arms", {})
            try:
                cpu_record = arms["optimized"]["instruction_input"]
                metal_record = arms["metal"]["instruction_input"]
                cpu_row_lifecycle = arms["optimized"][
                    "instruction_input_row_lifecycle"
                ]
                metal_row_lifecycle = arms["metal"][
                    "instruction_input_row_lifecycle"
                ]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    "production InstructionInput raw pair is incomplete"
                ) from error
            cpu_member = validate_production_instruction_input_member(
                cpu_record, "optimized", log_n, cutoff_log2, result_schema
            )
            metal_member = validate_production_instruction_input_member(
                metal_record, "metal", log_n, cutoff_log2, result_schema
            )
            validate_production_instruction_input_row_lifecycle(
                cpu_row_lifecycle, "optimized", log_n, result_schema
            )
            validate_production_instruction_input_row_lifecycle(
                metal_row_lifecycle, "metal", log_n, result_schema
            )
            if result_schema >= 6 and metal_record["resource_observation"][
                "native_primer"
            ]["resident_rows_storage_id"] != metal_row_lifecycle["stage3_storage_id"]:
                raise ValueError(
                    "production InstructionInput primer used the wrong resident rows"
                )
            rounding_slack_ns = log_n + 4
            if (
                cpu_member
                > arms["optimized"]["piop_ns"] + rounding_slack_ns
                or metal_member > arms["metal"]["piop_ns"] + rounding_slack_ns
            ):
                raise ValueError(
                    "production InstructionInput service timing exceeds its PIOP span"
                )
            if result_schema >= 6:
                metal_resources = metal_record["resource_observation"]
                if (
                    metal_resources["storage_initialization"]["wall_ns"]
                    > arms["metal"]["backend_witness_prepare_ns"] + 1
                    or metal_resources["native_primer"]["timings"][
                        "lifecycle_wall_ns"
                    ]
                    > arms["metal"]["piop_ns"] + rounding_slack_ns
                ):
                    raise ValueError(
                        "production InstructionInput startup timing exceeds its enclosing span"
                    )
            if not math.isclose(
                float(local_speedup), cpu_member / metal_member, rel_tol=1e-9
            ):
                raise ValueError(
                    "production InstructionInput raw pair disagrees with its speedup"
                )
            raw_cpu_members.append(cpu_member)
            raw_metal_members.append(metal_member)
        for name, raw_samples in (
            ("cpu_instruction_input_kernel_service_ms_samples", raw_cpu_members),
            ("metal_instruction_input_kernel_service_ms_samples", raw_metal_members),
        ):
            reported_samples = metrics.get(name)
            if (
                not isinstance(reported_samples, list)
                or len(reported_samples) != len(raw_samples)
                or any(
                    isinstance(reported, bool)
                    or not isinstance(reported, (int, float))
                    or not math.isfinite(reported)
                    or not math.isclose(
                        float(reported) * 1e6,
                        raw,
                        rel_tol=1e-12,
                        abs_tol=0.500001,
                    )
                    for reported, raw in zip(reported_samples, raw_samples)
                )
            ):
                raise ValueError("production InstructionInput sample summary is invalid")
        minimum_speedup = float(gate["minimum_local_speedup"])
        recomputed_improvements, recomputed_decision = recompute_local_member_decision(
            pair_records,
            raw_cpu_members,
            raw_metal_members,
            minimum_speedup,
            int(gate["minimum_pairs"]),
        )
        optimized_first_median = float(
            recomputed_decision["optimized_first_median_speedup"]
        )
        metal_first_median = float(
            recomputed_decision["metal_first_median_speedup"]
        )
        reported_improvements = metrics.get(
            "paired_instruction_input_kernel_service_fractional_improvements"
        )
        if (
            not isinstance(reported_improvements, list)
            or len(reported_improvements) != len(recomputed_improvements)
            or any(
                isinstance(reported, bool)
                or not isinstance(reported, (int, float))
                or not math.isfinite(reported)
                or not math.isclose(
                    float(reported), expected, rel_tol=1e-9, abs_tol=1e-12
                )
                for reported, expected in zip(
                    reported_improvements, recomputed_improvements
                )
            )
        ):
            raise ValueError(
                "production InstructionInput fractional improvements disagree with raw pairs"
            )
        if recomputed_decision["clears_order_strata"] is not True:
            raise ValueError(
                "production InstructionInput order stratum does not clear the local gate"
            )
        if recomputed_decision["clears"] is not True:
            raise ValueError(
                "production InstructionInput raw pairs do not clear the fixed local decision"
            )
        if not decisions_match(decision, recomputed_decision):
            raise ValueError(
                "production InstructionInput decision disagrees with recomputed raw-pair decision"
            )
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
        **(
            {
                "optimized_first_median_speedup": optimized_first_median,
                "metal_first_median_speedup": metal_first_median,
            }
            if local_kernel in {"BytecodeReadRafCycle", "InstructionInput"}
            else {}
        ),
    }


def validate_loaded_event(
    run_dir: Path,
    config: dict[str, Any],
    event: dict[str, Any],
    number: int,
    expected_parent: str,
) -> None:
    required = {
        "schema_version",
        "index",
        "trial_id",
        "parent_id",
        "candidate_revision",
        "proposal_summary",
        "candidate_id",
        "candidate_manifest_sha256",
        "params",
        "started_at",
        "elapsed_seconds",
        "metric_value",
        "measurements",
        "guards",
        "resources",
        "verdict",
        "reason",
    }
    if set(event) != required:
        raise ValueError(f"events.jsonl:{number}: event schema is not closed")
    if (
        event.get("schema_version") != SCHEMA_VERSION
        or event.get("index") != number
        or event.get("trial_id") != f"trial-{number:03d}"
        or event.get("parent_id") != expected_parent
        or event.get("verdict") not in VERDICTS
    ):
        raise ValueError(f"events.jsonl:{number}: invalid event identity or lineage")
    params = event.get("params")
    if (
        not isinstance(params, dict)
        or set(params) != set(config["search_space"])
        or any(type(name) is not str or type(value) is not str for name, value in params.items())
    ):
        raise ValueError(f"events.jsonl:{number}: invalid parameter record")
    validate_params(config, params)
    candidate_revision = event.get("candidate_revision")
    if (
        not isinstance(candidate_revision, str)
        or re.fullmatch(r"[0-9a-f]{64}", candidate_revision) is None
    ):
        raise ValueError(f"events.jsonl:{number}: invalid candidate revision")
    if event["verdict"] == "keep":
        snapshot = run_dir / "snapshots" / event["trial_id"]
        if not snapshot.is_dir():
            raise ValueError(f"events.jsonl:{number}: accepted snapshot is missing")
        source = path_digest(snapshot, config["scope"]["editable"])
        expected_revision = sha256(canonical_json({"source": source, "params": params}))
        if candidate_revision != expected_revision:
            raise ValueError(
                f"events.jsonl:{number}: accepted snapshot does not match its revision"
            )


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_path = run_dir / "run.json"
    config = read_json(run_path)
    expected = (run_dir / "run.sha256").read_text().strip()
    if sha256(run_path.read_bytes()) != expected:
        raise ValueError("run.json changed after initialization")
    strict_events = (
        isinstance(config.get("search_space"), dict)
        and isinstance(config.get("scope", {}).get("editable"), list)
        and isinstance(config.get("fingerprint", {}).get("editable_paths_sha256"), str)
    )
    if strict_events:
        baseline_params = {
            str(name): str(value)
            for name, value in config["baseline"]["params"].items()
        }
        if set(baseline_params) != set(config["search_space"]):
            raise ValueError("baseline parameters do not close the search space")
        validate_params(config, baseline_params)
        baseline_snapshot = run_dir / "snapshots" / "baseline"
        if path_digest(baseline_snapshot, config["scope"]["editable"]) != config[
            "fingerprint"
        ]["editable_paths_sha256"]:
            raise ValueError("baseline snapshot changed after initialization")

    events: list[dict[str, Any]] = []
    current_parent = "baseline"
    seen = {"baseline"}
    for number, line in enumerate((run_dir / "events.jsonl").read_text().splitlines(), 1):
        if not line:
            raise ValueError(f"events.jsonl:{number}: blank record")
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError(f"events.jsonl:{number}: invalid event")
        if strict_events:
            validate_loaded_event(run_dir, config, event, number, current_parent)
        elif event.get("index") != number or event.get("verdict") not in VERDICTS:
            raise ValueError(f"events.jsonl:{number}: invalid event")
        if event.get("trial_id") in seen or event.get("parent_id") != current_parent:
            raise ValueError(f"events.jsonl:{number}: invalid lineage")
        seen.add(event["trial_id"])
        if event["verdict"] == "keep":
            current_parent = event["trial_id"]
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


def validate_accepted_parent_for_production(
    config: dict[str, Any], metric_value: float
) -> None:
    if config.get("evaluator", {}).get("result_contract") != "instruction_input_v2":
        return
    minimum = float(
        config["final_validation"]["production_gate"]["minimum_local_speedup"]
    )
    if not math.isfinite(metric_value) or metric_value < minimum:
        raise ValueError(
            "the accepted local parent does not clear the full-protocol search gate"
        )


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
    validate_template(template, root)
    validate_new_run_template(template)
    goal_contract = read_json(root / template["portfolio_contract"])
    validate_goal_contract(goal_contract)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = run_dir / "logs"
    logs.mkdir()
    snapshots = run_dir / "snapshots"
    snapshots.mkdir()
    snapshot_paths(root, template["scope"]["editable"], snapshots / "baseline")
    initial_editable_digest = path_digest(
        snapshots / "baseline", template["scope"]["editable"]
    )
    if path_digest(root, template["scope"]["editable"]) != initial_editable_digest:
        raise ValueError("baseline snapshot changed during initialization")
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
        accepted_snapshot = run_dir / "snapshots" / trial_id
        snapshot_paths(
            root,
            config["scope"]["editable"],
            accepted_snapshot,
        )
        if path_digest(accepted_snapshot, config["scope"]["editable"]) != candidate_revision:
            quarantine = run_dir / "quarantine" / utc_now().replace(":", "-")
            quarantine.mkdir(parents=True)
            shutil.move(accepted_snapshot, quarantine / "orphan-accepted-snapshot")
            restore_snapshot(
                root,
                config["scope"]["editable"],
                run_dir / "snapshots" / parent_id,
            )
            verdict = "crash"
            metric_value = None
            reason = "editable source changed while snapshotting the accepted candidate"
            event.update(verdict=verdict, metric_value=metric_value, reason=reason)
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
    result = parse_unique_schema_result(completed.stdout, result_schema)
    result_bytes = canonical_json(result)
    (attempt / "result.json").write_bytes(result_bytes)
    return result, result_bytes, attempt


def validate_cached_production_promotion(
    root: Path,
    run_dir: Path,
    config: dict[str, Any],
    params: dict[str, str],
    parent_id: str,
    record: dict[str, Any],
    expected_revision: str,
) -> dict[str, Any]:
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("status") != "promoted"
        or record.get("parent_id") != parent_id
        or not isinstance(record.get("result_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", record["result_sha256"]) is None
        or not isinstance(record.get("attempt"), str)
        or not isinstance(record.get("recorded_at"), str)
    ):
        raise ValueError("cached production promotion record is incomplete")
    attempt = Path(record["attempt"]).resolve()
    attempts = (run_dir / "production-attempts").resolve()
    if attempt.parent != attempts:
        raise ValueError("cached production promotion names an invalid attempt")
    result_path = attempt / "result.json"
    if not result_path.is_file():
        raise ValueError("cached production promotion has no result artifact")
    result_bytes = result_path.read_bytes()
    if sha256(result_bytes) != record["result_sha256"]:
        raise ValueError("cached production promotion result hash changed")
    try:
        result = json.loads(result_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("cached production promotion result is not JSON") from error
    if canonical_json(result) != result_bytes:
        raise ValueError("cached production promotion result is not canonical")
    evidence = validate_production_result(
        config,
        result,
        expected_revision,
        params,
        git_worktree_clean(root),
    )
    expected_fields = {
        "schema_version",
        "status",
        "parent_id",
        "result_sha256",
        "attempt",
        "recorded_at",
        *evidence,
    }
    if set(record) != expected_fields or any(
        record.get(name) != value for name, value in evidence.items()
    ):
        raise ValueError("cached production promotion evidence changed")
    return evidence


def command_validate_production(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve()
    config, events = load_run(run_dir)
    parent_id, parent_metric = accepted_parent(config, events)
    validate_accepted_parent_for_production(config, parent_metric)
    params = accepted_parent_params(config, events)
    validate_params(config, params)
    ledger = run_dir / "production-validations.jsonl"
    if not ledger.exists():
        ledger.touch()
    prior = [json.loads(line) for line in ledger.read_text().splitlines()]
    successful_records = [
        record
        for record in prior
        if record.get("parent_id") == parent_id and record.get("status") == "promoted"
    ]
    if len(successful_records) > 1:
        raise ValueError("cached production promotion ledger has duplicate successes")
    successful = successful_records[0] if successful_records else None
    if successful is not None and not {
        "schema_version",
        "status",
        "parent_id",
        "result_sha256",
        "attempt",
        "recorded_at",
    } <= set(successful):
        raise ValueError(
            "cached production promotion record is incomplete"
        )
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
    validate_production_revision_scope(
        root,
        config["base_revision"],
        expected_revision,
        editable,
    )
    if successful is not None:
        validate_cached_production_promotion(
            root,
            run_dir,
            config,
            params,
            parent_id,
            successful,
            expected_revision,
        )
        repair_candidate_promotion(run_dir, events, parent_id)
        print(json.dumps(successful, sort_keys=True))
        return 0
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
