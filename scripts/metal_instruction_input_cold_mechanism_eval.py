#!/usr/bin/env python3
"""Isolate cold InstructionInput scratch binding and pipeline submission costs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from metal_autoresearch import evaluator_lock
except ModuleNotFoundError:
    from scripts.metal_autoresearch import evaluator_lock


SCHEMA = "instruction_input_cold_mechanism_v1"
SCHEMA_VERSION = 1
ARMS = ("lazy", "minimal", "compute_control", "native_primer")
STORAGE_MODE = {
    "lazy": "lazy",
    "minimal": "minimal",
    "compute_control": "minimal",
    "native_primer": "minimal",
}
INITIALIZATION_BYTES = {"lazy": 0, "minimal": 96}
INITIALIZATION_BUFFERS = {"lazy": 0, "minimal": 6}
LOG_N = 26
CUTOFF_LOG2 = 16
BLOCKS = 4
RAW_MEMBER_MIN_NS = 180_000_000
RAW_ROUND_0_NONACTIVE_MIN_NS = 80_000_000
MINIMAL_RATIO_MAX = 0.85
MINIMAL_MEMBER_MAX_NS = 260_000_000
CONTROL_RATIO_MAX = 0.75
CONTROL_MEMBER_MAX_NS = 160_000_000
CONTROL_WAIT_REDUCTION_MIN_NS = 75_000_000
CONTROL_ALIGNMENT_MAX_NS = 20_000_000
INACTIVE_EFFECT_NS = 20_000_000
GPU_RATIO_TOLERANCE = 0.10
RELEVANT_PATHS = (
    "crates/jolt-kernels/Cargo.toml",
    "crates/jolt-kernels/examples/metal-instruction-input-cold-mechanism-eval.rs",
    "crates/jolt-kernels/examples/support/instruction_input.rs",
    "crates/jolt-kernels/src/metal/solinas/fp128.metal",
    "crates/jolt-kernels/src/metal/solinas/instruction_input.metal",
    "crates/jolt-kernels/src/metal/solinas/instruction_input.rs",
    "crates/jolt-kernels/src/metal/solinas/mod.rs",
    "crates/jolt-kernels/src/metal/solinas/probes.metal",
    "crates/jolt-kernels/src/metal/solinas/spartan_outer_uniskip.metal",
    "crates/jolt-kernels/src/optimized/instruction_input.rs",
    "scripts/metal_instruction_input_cold_mechanism_eval.py",
)
TOP_LEVEL_FIELDS = {
    "schema",
    "schema_version",
    "kernel",
    "arm",
    "metrics",
    "timings",
    "guards",
    "all_exact",
    "resources",
    "workload",
    "fingerprint",
}
METRIC_FIELDS = {
    "member_wall_ns",
    "round_0_nonactive_ns",
    "control_plus_member_ns",
}
TIMING_FIELDS = {
    "cpu_control_ns",
    "sequence_preparation_ns",
    "storage_initialization_wall_ns",
    "storage_initialization_gpu_active_ns",
    "control_wall_ns",
    "control_gpu_active_ns",
    "member_wall_ns",
    "gpu_dispatch_wall_ns",
    "gpu_active_ns",
    "host_round_ns",
    "readback_ns",
    "cpu_tail_ns",
    "round_0_gpu_command_wall_ns",
    "round_0_gpu_command_active_ns",
    "round_0_nonactive_ns",
    "first_three_gpu_command_wall_ns",
    "first_three_gpu_command_active_ns",
    "later_gpu_command_wall_ns",
    "later_gpu_command_active_ns",
    "gpu_command_wall_ns",
    "gpu_command_active_ns",
}
GUARD_FIELDS = {
    "exact_four_sample_q_evals",
    "exact_round_polynomials",
    "exact_host_fiat_shamir_challenges",
    "exact_round_schedule",
    "exact_final_eight_claims",
    "exact_final_sumcheck_claim",
    "exact_transcript_state",
    "exact_derived_eq_cycle",
    "exact_final_relation",
    "storage_initialization_mode_exact",
    "storage_initialization_bytes_exact",
    "storage_initialization_buffer_count_exact",
    "storage_initialization_completed_before_member",
    "storage_initialization_timestamps_exact",
    "control_command_presence_exact",
    "control_command_timestamps_valid",
    "native_primer_geometry_exact",
    "static_device_buffer_identities_stable",
    "static_device_buffer_identities_distinct",
    "resident_rows_stable",
    "exactly_one_dense_readback",
    "readback_bytes_exact",
    "round_device_buffer_allocations_zero",
    "gpu_command_count_exact",
    "gpu_wall_reconciled",
    "gpu_active_reconciled",
    "gpu_command_timestamps_valid",
    "host_fiat_shamir",
    "no_excluded_target_warmup",
    "one_first_use_target_sequence",
}
RESOURCE_FIELDS = {
    "sequence_owned_storage_bytes",
    "storage_initialization_bytes",
    "storage_initialization_device_buffers",
    "storage_buffer_identities",
    "resident_row_identity",
    "primer_source_elements",
    "primer_e_in_elements",
    "primer_e_out_elements",
    "primer_resident_row_identity",
    "primer_storage_buffer_identities",
    "cutoff_readback_bytes",
    "persistent_device_buffers",
    "round_device_buffer_allocations",
}
WORKLOAD_FIELDS = {
    "log_n",
    "rows",
    "cutoff_log2",
    "cutoff_elements",
    "tables",
    "host_fiat_shamir",
    "target_sequences",
    "excluded_target_warmups",
    "cpu_control_before_sequence_preparation",
    "storage_initialization_outside_member_timer",
    "control_outside_member_timer",
}
FINGERPRINT_FIELDS = {
    "device",
    "max_buffer_length",
    "recommended_max_working_set_size",
    "cpu_threads",
    "seed",
    "log_n",
    "cutoff_log2",
    "native_message_threads",
    "native_transition_threads",
    "dense_transition_threads",
    "storage_initialization",
    "control",
    "gpu_command_count",
    "process_model",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(root: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("cold-mechanism evaluator requires a clean worktree")
    digest = hashlib.sha256()
    for relative in RELEVANT_PATHS:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing cold-mechanism evaluator source: {relative}")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"revision": revision, "relevant_paths_sha256": digest.hexdigest()}


def default_artifact_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    return (
        root
        / "benchmark-runs"
        / "metal-autoresearch"
        / f"instruction-input-cold-mechanism-{stamp}"
    )


def median(values: list[float | int]) -> float:
    return float(statistics.median(values))


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def validate_result(result: dict[str, Any], arm: str, seed: int) -> None:
    if arm not in ARMS or set(result) != TOP_LEVEL_FIELDS:
        raise ValueError("cold-mechanism evaluator violates its top-level schema")
    if (
        result["schema"] != SCHEMA
        or result["schema_version"] != SCHEMA_VERSION
        or result["kernel"] != "instruction_input"
        or result["arm"] != arm
        or result["all_exact"] is not True
    ):
        raise ValueError("cold-mechanism evaluator identity or exactness is invalid")
    metrics = result["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != METRIC_FIELDS:
        raise ValueError("cold-mechanism evaluator metric schema is invalid")
    guards = result["guards"]
    if (
        not isinstance(guards, dict)
        or set(guards) != GUARD_FIELDS
        or any(value is not True for value in guards.values())
    ):
        raise ValueError("cold-mechanism evaluator guard failed")

    timings = result["timings"]
    if not isinstance(timings, dict) or set(timings) != TIMING_FIELDS:
        raise ValueError("cold-mechanism evaluator timing schema is invalid")
    zero_allowed = {
        "storage_initialization_wall_ns",
        "storage_initialization_gpu_active_ns",
        "control_wall_ns",
        "control_gpu_active_ns",
    }
    scalar_names = TIMING_FIELDS - {
        "gpu_command_wall_ns",
        "gpu_command_active_ns",
    }
    for name in scalar_names:
        if name in zero_allowed:
            nonnegative_int(timings[name], name)
        else:
            positive_int(timings[name], name)
    command_wall = timings["gpu_command_wall_ns"]
    command_active = timings["gpu_command_active_ns"]
    if (
        not isinstance(command_wall, list)
        or not isinstance(command_active, list)
        or len(command_wall) != LOG_N - CUTOFF_LOG2 + 1
        or len(command_active) != len(command_wall)
        or any(
            positive_int(value, "GPU command timing") <= 0
            for value in command_wall + command_active
        )
        or any(active > wall for active, wall in zip(command_active, command_wall))
    ):
        raise ValueError("cold-mechanism GPU command timings are invalid")
    if (
        sum(command_wall) != timings["gpu_dispatch_wall_ns"]
        or sum(command_active) != timings["gpu_active_ns"]
        or command_wall[0] != timings["round_0_gpu_command_wall_ns"]
        or command_active[0] != timings["round_0_gpu_command_active_ns"]
        or command_wall[0] - command_active[0] != timings["round_0_nonactive_ns"]
        or sum(command_wall[:3]) != timings["first_three_gpu_command_wall_ns"]
        or sum(command_active[:3]) != timings["first_three_gpu_command_active_ns"]
        or sum(command_wall[3:]) != timings["later_gpu_command_wall_ns"]
        or sum(command_active[3:]) != timings["later_gpu_command_active_ns"]
    ):
        raise ValueError("cold-mechanism GPU timings do not reconcile")
    accounted = (
        timings["gpu_dispatch_wall_ns"]
        + timings["host_round_ns"]
        + timings["readback_ns"]
        + timings["cpu_tail_ns"]
    )
    if accounted > timings["member_wall_ns"]:
        raise ValueError("cold-mechanism member timing is under-accounted")
    if (
        metrics["member_wall_ns"] != timings["member_wall_ns"]
        or metrics["round_0_nonactive_ns"] != timings["round_0_nonactive_ns"]
        or metrics["control_plus_member_ns"]
        != timings["control_wall_ns"] + timings["member_wall_ns"]
    ):
        raise ValueError("cold-mechanism metric aliases are invalid")

    mode = STORAGE_MODE[arm]
    expects_control = arm in {"compute_control", "native_primer"}
    if expects_control and not (
        timings["control_wall_ns"] > 0
        and 0 < timings["control_gpu_active_ns"] <= timings["control_wall_ns"]
    ):
        raise ValueError("cold-mechanism control timing is invalid")
    if not expects_control and (
        timings["control_wall_ns"] != 0
        or timings["control_gpu_active_ns"] != 0
    ):
        raise ValueError("cold-mechanism inactive control timing is invalid")
    if mode == "lazy":
        if (
            timings["storage_initialization_gpu_active_ns"] != 0
            or timings["storage_initialization_wall_ns"] < 0
        ):
            raise ValueError("lazy storage unexpectedly executed a GPU initialization")
    elif not (
        0
        < timings["storage_initialization_gpu_active_ns"]
        <= timings["storage_initialization_wall_ns"]
    ):
        raise ValueError("minimal storage initialization timing is invalid")

    resources = result["resources"]
    if not isinstance(resources, dict) or set(resources) != RESOURCE_FIELDS:
        raise ValueError("cold-mechanism resource schema is invalid")
    if (
        resources.get("sequence_owned_storage_bytes") != 6_443_433_984
        or resources.get("storage_initialization_bytes")
        != INITIALIZATION_BYTES[mode]
        or resources.get("storage_initialization_device_buffers")
        != INITIALIZATION_BUFFERS[mode]
        or resources.get("persistent_device_buffers") != 6
        or resources.get("round_device_buffer_allocations") != 0
        or resources.get("cutoff_readback_bytes")
        != 8 * (1 << CUTOFF_LOG2) * 16
    ):
        raise ValueError("cold-mechanism resource accounting is invalid")
    identities = resources.get("storage_buffer_identities")
    resident_identity = resources.get("resident_row_identity")
    if (
        not isinstance(identities, list)
        or len(identities) != 6
        or len(set(identities)) != 6
        or positive_int(resident_identity, "resident row identity") <= 0
    ):
        raise ValueError("cold-mechanism target identities are invalid")
    primer_identities = resources.get("primer_storage_buffer_identities")
    if not isinstance(primer_identities, list) or len(primer_identities) != 6:
        raise ValueError("cold-mechanism primer identity schema is invalid")
    if arm == "native_primer":
        if (
            resources.get("primer_source_elements") != 64
            or resources.get("primer_e_in_elements") != 1
            or resources.get("primer_e_out_elements") != 32
            or resources.get("primer_resident_row_identity") != resident_identity
            or primer_identities != identities
        ):
            raise ValueError("cold-mechanism native primer geometry is invalid")
    elif (
        resources.get("primer_source_elements") != 0
        or resources.get("primer_e_in_elements") != 0
        or resources.get("primer_e_out_elements") != 0
        or resources.get("primer_resident_row_identity") != 0
        or primer_identities != [0] * 6
    ):
        raise ValueError("cold-mechanism inactive primer accounting is invalid")

    workload = result["workload"]
    if (
        not isinstance(workload, dict)
        or set(workload) != WORKLOAD_FIELDS
        or workload.get("log_n") != LOG_N
        or workload.get("rows") != 1 << LOG_N
        or workload.get("cutoff_log2") != CUTOFF_LOG2
        or workload.get("cutoff_elements") != 1 << CUTOFF_LOG2
        or workload.get("tables") != 8
        or workload.get("host_fiat_shamir") is not True
        or workload.get("target_sequences") != 1
        or workload.get("excluded_target_warmups") != 0
        or workload.get("cpu_control_before_sequence_preparation") is not True
        or workload.get("storage_initialization_outside_member_timer") is not True
        or workload.get("control_outside_member_timer") is not True
    ):
        raise ValueError("cold-mechanism workload contract is invalid")
    fingerprint = result["fingerprint"]
    if (
        not isinstance(fingerprint, dict)
        or set(fingerprint) != FINGERPRINT_FIELDS
        or not isinstance(fingerprint.get("device"), str)
        or not fingerprint.get("device")
        or positive_int(fingerprint.get("max_buffer_length"), "max buffer length") <= 0
        or positive_int(
            fingerprint.get("recommended_max_working_set_size"),
            "recommended working set",
        )
        <= 0
        or positive_int(fingerprint.get("cpu_threads"), "CPU threads") <= 0
        or fingerprint.get("seed") != seed
        or fingerprint.get("log_n") != LOG_N
        or fingerprint.get("cutoff_log2") != CUTOFF_LOG2
        or fingerprint.get("native_message_threads") != 256
        or fingerprint.get("native_transition_threads") != 128
        or fingerprint.get("dense_transition_threads") != 128
        or fingerprint.get("storage_initialization") != mode
        or fingerprint.get("control") != arm
        or fingerprint.get("gpu_command_count") != LOG_N - CUTOFF_LOG2 + 1
        or fingerprint.get("process_model")
        != "one_cold_target_sequence_per_process"
    ):
        raise ValueError("cold-mechanism fingerprint is invalid")


def run_arm(
    root: Path,
    binary: Path,
    artifact_dir: Path,
    block_index: int,
    arm: str,
    seed: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "JOLT_METAL_EVAL_LOG_N": str(LOG_N),
            "JOLT_METAL_EVAL_SEED": str(seed),
            "JOLT_METAL_INSTRUCTION_INPUT_CUTOFF_LOG2": str(CUTOFF_LOG2),
            "JOLT_METAL_INSTRUCTION_INPUT_NATIVE_MESSAGE_THREADS": "256",
            "JOLT_METAL_INSTRUCTION_INPUT_NATIVE_TRANSITION_THREADS": "128",
            "JOLT_METAL_INSTRUCTION_INPUT_DENSE_TRANSITION_THREADS": "128",
            "JOLT_METAL_INSTRUCTION_INPUT_COLD_ARM": arm,
        }
    )
    result = subprocess.run(
        [str(binary)],
        cwd=root,
        env=env,
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
    )
    label = f"block-{block_index:02d}-{arm}"
    (artifact_dir / f"{label}.stdout").write_text(result.stdout)
    (artifact_dir / f"{label}.stderr").write_text(result.stderr)
    if result.returncode != 0:
        raise ValueError(f"{arm} evaluator exited with status {result.returncode}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{arm} evaluator must emit exactly one JSON object")
    parsed = json.loads(lines[0])
    if not isinstance(parsed, dict):
        raise ValueError(f"{arm} evaluator did not emit a JSON object")
    validate_result(parsed, arm, seed)
    return parsed


def arm_values(blocks: list[dict[str, Any]], arm: str, field: str) -> list[int]:
    return [block[arm]["timings"][field] for block in blocks]


def contrast(
    blocks: list[dict[str, Any]], control: str, treated: str
) -> dict[str, Any]:
    control_member = arm_values(blocks, control, "member_wall_ns")
    treated_member = arm_values(blocks, treated, "member_wall_ns")
    control_wait = arm_values(blocks, control, "round_0_nonactive_ns")
    treated_wait = arm_values(blocks, treated, "round_0_nonactive_ns")
    member_effects = [
        control_ns - treated_ns
        for control_ns, treated_ns in zip(control_member, treated_member)
    ]
    wait_effects = [
        control_ns - treated_ns
        for control_ns, treated_ns in zip(control_wait, treated_wait)
    ]
    ratios = [
        treated_ns / control_ns
        for control_ns, treated_ns in zip(control_member, treated_member)
    ]
    member_effect = median(member_effects)
    wait_effect = median(wait_effects)
    effect_mad = median([abs(value - member_effect) for value in member_effects])
    paired_alignment = [
        abs(member - wait)
        for member, wait in zip(member_effects, wait_effects)
    ]
    active_ratios = [
        treated_ns / control_ns
        for control_ns, treated_ns in zip(
            arm_values(blocks, control, "gpu_active_ns"),
            arm_values(blocks, treated, "gpu_active_ns"),
        )
    ]
    later_ratios = [
        treated_ns / control_ns
        for control_ns, treated_ns in zip(
            arm_values(blocks, control, "later_gpu_command_wall_ns"),
            arm_values(blocks, treated, "later_gpu_command_wall_ns"),
        )
    ]
    treated_median = median(treated_member)
    criteria = {
        "at_least_three_of_four_faster": sum(value > 0 for value in member_effects) >= 3,
        "at_least_three_of_four_round_0_waits_lower": sum(value > 0 for value in wait_effects)
        >= 3,
        "median_ratio_at_most_0_75": median(ratios) <= CONTROL_RATIO_MAX,
        "treated_member_at_most_160_ms": treated_median <= CONTROL_MEMBER_MAX_NS,
        "round_0_nonactive_reduction_at_least_75_ms": (
            wait_effect >= CONTROL_WAIT_REDUCTION_MIN_NS
        ),
        "three_of_four_member_and_wait_reductions_align_within_20_ms": sum(
            value <= CONTROL_ALIGNMENT_MAX_NS for value in paired_alignment
        )
        >= 3,
        "member_reduction_exceeds_three_paired_mads": member_effect > 3 * effect_mad,
        "three_of_four_gpu_active_ratios_within_10_percent": sum(
            abs(value - 1.0) <= GPU_RATIO_TOLERANCE for value in active_ratios
        )
        >= 3,
        "three_of_four_later_wall_ratios_within_10_percent": sum(
            abs(value - 1.0) <= GPU_RATIO_TOLERANCE for value in later_ratios
        )
        >= 3,
    }
    return {
        "control": control,
        "treated": treated,
        "control_member_ns": control_member,
        "treated_member_ns": treated_member,
        "paired_member_effect_ns": member_effects,
        "paired_member_ratios": ratios,
        "median_member_effect_ns": member_effect,
        "median_wait_effect_ns": wait_effect,
        "paired_member_wait_alignment_ns": paired_alignment,
        "paired_effect_mad_ns": effect_mad,
        "treated_member_median_ns": treated_median,
        "treated_over_control_median": median(ratios),
        "paired_gpu_active_ratios": active_ratios,
        "paired_later_wall_ratios": later_ratios,
        "criteria": criteria,
        "clears": all(criteria.values()),
    }


def inactive(comparison: dict[str, Any]) -> bool:
    return (
        sum(
            abs(value) <= INACTIVE_EFFECT_NS
            for value in comparison["paired_member_effect_ns"]
        )
        >= 3
        and sum(0.8 <= value <= 1.2 for value in comparison["paired_member_ratios"])
        >= 3
    )


def hardware_fingerprint(result: dict[str, Any]) -> dict[str, Any]:
    fingerprint = result["fingerprint"]
    return {
        name: fingerprint[name]
        for name in (
            "device",
            "max_buffer_length",
            "recommended_max_working_set_size",
            "cpu_threads",
            "native_message_threads",
            "native_transition_threads",
            "dense_transition_threads",
        )
    }


def summarize(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    lazy_member = arm_values(blocks, "lazy", "member_wall_ns")
    minimal_member = arm_values(blocks, "minimal", "member_wall_ns")
    lazy_wait = arm_values(blocks, "lazy", "round_0_nonactive_ns")
    minimal_ratios = [
        minimal_ns / lazy_ns
        for lazy_ns, minimal_ns in zip(lazy_member, minimal_member)
    ]
    minimal_effects = [
        lazy_ns - minimal_ns
        for lazy_ns, minimal_ns in zip(lazy_member, minimal_member)
    ]
    minimal_effect = median(minimal_effects)
    minimal_effect_mad = median(
        [abs(value - minimal_effect) for value in minimal_effects]
    )
    minimal_active_ratios = [
        minimal_ns / lazy_ns
        for lazy_ns, minimal_ns in zip(
            arm_values(blocks, "lazy", "gpu_active_ns"),
            arm_values(blocks, "minimal", "gpu_active_ns"),
        )
    ]
    minimal_later_ratios = [
        minimal_ns / lazy_ns
        for lazy_ns, minimal_ns in zip(
            arm_values(blocks, "lazy", "later_gpu_command_wall_ns"),
            arm_values(blocks, "minimal", "later_gpu_command_wall_ns"),
        )
    ]
    raw_phenomenon = {
        "at_least_three_of_four_lazy_members_at_least_180_ms": sum(
            value >= RAW_MEMBER_MIN_NS for value in lazy_member
        )
        >= 3,
        "at_least_three_of_four_lazy_round_0_waits_at_least_80_ms": sum(
            value >= RAW_ROUND_0_NONACTIVE_MIN_NS for value in lazy_wait
        )
        >= 3,
    }
    minimal_criteria = {
        "at_least_three_of_four_faster": sum(value > 0 for value in minimal_effects) >= 3,
        "median_ratio_at_most_0_85": median(minimal_ratios) <= MINIMAL_RATIO_MAX,
        "minimal_member_at_most_260_ms": median(minimal_member)
        <= MINIMAL_MEMBER_MAX_NS,
        "member_reduction_exceeds_three_paired_mads": minimal_effect
        > 3 * minimal_effect_mad,
        "three_of_four_gpu_active_ratios_within_10_percent": sum(
            abs(value - 1.0) <= GPU_RATIO_TOLERANCE
            for value in minimal_active_ratios
        )
        >= 3,
        "three_of_four_later_wall_ratios_within_10_percent": sum(
            abs(value - 1.0) <= GPU_RATIO_TOLERANCE
            for value in minimal_later_ratios
        )
        >= 3,
    }
    compute = contrast(blocks, "minimal", "compute_control")
    primer = contrast(blocks, "minimal", "native_primer")
    primer_over_compute = contrast(blocks, "compute_control", "native_primer")

    phenomenon_reproduced = all(raw_phenomenon.values())
    if not phenomenon_reproduced:
        mechanism = "none"
        reason = "the lazy arm did not reproduce the predeclared cold phenomenon"
    elif compute["clears"] and primer["clears"] and inactive(primer_over_compute):
        mechanism = "general_compute_startup"
        reason = "generic compute and the native primer cleared equally; target-specific work was inactive"
    elif primer["clears"] and not compute["clears"]:
        mechanism = "native_pipeline_or_row_binding"
        reason = "only the exact native pipeline and row-prefix primer cleared the cold delay"
    elif compute["clears"] and not primer["clears"]:
        mechanism = "compute_control_only"
        reason = "only the unrelated compute command cleared; broad attribution is not established"
    elif compute["clears"] and primer["clears"]:
        mechanism = "shared_or_mixed"
        reason = "both controls cleared, but their incremental contrast remained material"
    else:
        mechanism = "none"
        reason = "neither predispatch control cleared its frozen mechanism gate"

    arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        members = arm_values(blocks, arm, "member_wall_ns")
        controls = arm_values(blocks, arm, "control_wall_ns")
        preparations = arm_values(blocks, arm, "sequence_preparation_ns")
        cpu_controls = arm_values(blocks, arm, "cpu_control_ns")
        arm_summary[arm] = {
            "member_ns": members,
            "member_median_ns": median(members),
            "cpu_control_median_ns": median(cpu_controls),
            "paired_cpu_over_member_speedups": [
                cpu / member for cpu, member in zip(cpu_controls, members)
            ],
            "paired_cpu_over_control_plus_member_speedups": [
                cpu / (control + member)
                for cpu, control, member in zip(cpu_controls, controls, members)
            ],
            "round_0_nonactive_median_ns": median(
                arm_values(blocks, arm, "round_0_nonactive_ns")
            ),
            "control_median_ns": median(controls),
            "control_plus_member_median_ns": median(
                [control + member for control, member in zip(controls, members)]
            ),
            "prepare_control_member_median_ns": median(
                [
                    prepare + control + member
                    for prepare, control, member in zip(
                        preparations, controls, members
                    )
                ]
            ),
        }
    return {
        "decision": mechanism,
        "reason": reason,
        "acceptance_eligible": False,
        "phenomenon_reproduced": phenomenon_reproduced,
        "raw_phenomenon": raw_phenomenon,
        "minimal_storage_selected": phenomenon_reproduced
        and all(minimal_criteria.values()),
        "minimal_storage_criteria": minimal_criteria,
        "minimal_over_lazy_ratios": minimal_ratios,
        "minimal_effect_ns": minimal_effects,
        "minimal_effect_mad_ns": minimal_effect_mad,
        "minimal_gpu_active_ratios": minimal_active_ratios,
        "minimal_later_wall_ratios": minimal_later_ratios,
        "compute_control": compute,
        "native_primer": primer,
        "native_primer_over_compute": primer_over_compute,
        "arms": arm_summary,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    result.add_argument("--artifact-dir", type=Path)
    result.add_argument("--timeout-seconds", type=int, default=1800)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.timeout_seconds < 1:
        print("error: timeout must be positive", file=sys.stderr)
        return 2
    root = args.root.resolve()
    artifact_dir = (
        args.artifact_dir.resolve()
        if args.artifact_dir is not None
        else default_artifact_dir(root)
    )
    orders = [
        list(ARMS[index:] + ARMS[:index])
        for index in range(BLOCKS)
    ]
    try:
        artifact_dir.mkdir(parents=True, exist_ok=False)
        source = source_fingerprint(root)
        contract = {
            "schema": "instruction_input_cold_mechanism_experiment_v1",
            "created_at": utc_now(),
            "acceptance_eligible": False,
            "question": "which first-use mechanism causes cold InstructionInput latency?",
            "arms": {
                "lazy": "no GPU initialization or control",
                "minimal": "one 96-byte blit binding all six scratch resources",
                "compute_control": "minimal plus one unrelated no-op threadgroup",
                "native_primer": "minimal plus exact native-message and reduction PSOs on 64 resident rows",
            },
            "orders": orders,
            "blocks": BLOCKS,
            "log_n": LOG_N,
            "cutoff_log2": CUTOFF_LOG2,
            "thresholds": {
                "raw_member_min_ns": RAW_MEMBER_MIN_NS,
                "raw_round_0_nonactive_min_ns": RAW_ROUND_0_NONACTIVE_MIN_NS,
                "minimal_ratio_max": MINIMAL_RATIO_MAX,
                "minimal_member_max_ns": MINIMAL_MEMBER_MAX_NS,
                "control_ratio_max": CONTROL_RATIO_MAX,
                "control_member_max_ns": CONTROL_MEMBER_MAX_NS,
                "control_wait_reduction_min_ns": CONTROL_WAIT_REDUCTION_MIN_NS,
                "control_alignment_max_ns": CONTROL_ALIGNMENT_MAX_NS,
                "inactive_effect_ns": INACTIVE_EFFECT_NS,
                "gpu_ratio_tolerance": GPU_RATIO_TOLERANCE,
                "paired_improvements_required": 3,
                "noise_multiplier": 3,
            },
            "primary_metric": "target member wall; control and preparation are reported separately",
            "holdout": "any selected mechanism requires a new production CPU-vs-Metal holdout",
            "source": source,
        }
        (artifact_dir / "run-contract.json").write_bytes(canonical_json(contract))
        build_command = [
            "cargo",
            "build",
            "--release",
            "-q",
            "-p",
            "jolt-kernels",
            "--features",
            "metal,parallel",
            "--example",
            "metal-instruction-input-cold-mechanism-eval",
        ]
        built = subprocess.run(
            build_command,
            cwd=root,
            timeout=args.timeout_seconds,
            capture_output=True,
            text=True,
        )
        (artifact_dir / "build.stdout").write_text(built.stdout)
        (artifact_dir / "build.stderr").write_text(built.stderr)
        if built.returncode != 0:
            raise ValueError(f"evaluator build exited with status {built.returncode}")
        binary = (
            root
            / "target"
            / "release"
            / "examples"
            / "metal-instruction-input-cold-mechanism-eval"
        )
        if not binary.is_file():
            raise ValueError("evaluator build did not produce the expected binary")
        binary_sha256 = file_sha256(binary)
        if source_fingerprint(root) != source:
            raise ValueError("source changed during the evaluator build")

        blocks: list[dict[str, Any]] = []
        started = time.monotonic()
        for index, order in enumerate(orders, 1):
            seed = 0x9E37_79B9 ^ (index * 0x85EB_CA6B)
            results: dict[str, Any] = {}
            for arm in order:
                results[arm] = run_arm(
                    root,
                    binary,
                    artifact_dir,
                    index,
                    arm,
                    seed,
                    args.timeout_seconds,
                )
            blocks.append({"index": index, "order": order, "seed": seed, **results})
        hardware = hardware_fingerprint(blocks[0][ARMS[0]])
        if any(
            hardware_fingerprint(block[arm]) != hardware
            for block in blocks
            for arm in ARMS
        ):
            raise ValueError("hardware fingerprint changed during the blocked evaluator")
        if source_fingerprint(root) != source:
            raise ValueError("source changed during the blocked evaluator")
        if file_sha256(binary) != binary_sha256:
            raise ValueError("evaluator binary changed during the blocked evaluator")
        summary = summarize(blocks)
        output = {
            "schema": "instruction_input_cold_mechanism_blocks_v1",
            "schema_version": 1,
            "run_class": "diagnostic",
            "acceptance_eligible": False,
            "source": source,
            "binary_sha256": binary_sha256,
            "contract_sha256": file_sha256(artifact_dir / "run-contract.json"),
            "hardware": hardware,
            "elapsed_seconds": time.monotonic() - started,
            "summary": summary,
            "blocks": blocks,
            "artifacts": str(artifact_dir),
        }
        (artifact_dir / "result.json").write_bytes(canonical_json(output))
        print(json.dumps(output, sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    with evaluator_lock(
        {"direct_evaluator": "metal_instruction_input_cold_mechanism_eval"}
    ):
        raise SystemExit(main())
