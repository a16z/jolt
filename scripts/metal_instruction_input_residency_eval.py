#!/usr/bin/env python3
"""Compare cold minimal and full InstructionInput storage initialization."""

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


SCHEMA = "instruction_input_residency_v1"
SCHEMA_VERSION = 1
LOG_N = 26
CUTOFF_LOG2 = 16
PAIRS = 3
STEADY_REFERENCE_NS = 106_168_000
FULL_MEMBER_LIMIT_NS = 160_000_000
FULL_RATIO_LIMIT = 0.70
GAP_CLOSED_MINIMUM = 0.80
EARLY_SAVINGS_SHARE_MINIMUM = 0.80
GPU_ACTIVE_RATIO_TOLERANCE = 0.10
INITIALIZATION_BYTES = {"minimal": 96, "full": 6_443_433_984}
RELEVANT_PATHS = (
    "crates/jolt-kernels/Cargo.toml",
    "crates/jolt-kernels/examples/metal-instruction-input-residency-eval.rs",
    "crates/jolt-kernels/examples/support/instruction_input.rs",
    "crates/jolt-kernels/src/metal/instruction_input.rs",
    "crates/jolt-kernels/src/metal/solinas/instruction_input.metal",
    "crates/jolt-kernels/src/metal/solinas/instruction_input.rs",
    "crates/jolt-kernels/src/optimized/instruction_input.rs",
    "scripts/metal_instruction_input_residency_eval.py",
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
TIMING_FIELDS = {
    "cpu_control_ns",
    "sequence_preparation_ns",
    "storage_initialization_wall_ns",
    "storage_initialization_gpu_active_ns",
    "member_wall_ns",
    "gpu_dispatch_wall_ns",
    "gpu_active_ns",
    "host_round_ns",
    "readback_ns",
    "cpu_tail_ns",
    "first_three_gpu_command_wall_ns",
    "first_three_gpu_command_active_ns",
    "later_gpu_command_wall_ns",
    "later_gpu_command_active_ns",
    "gpu_command_wall_ns",
    "gpu_command_active_ns",
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
    digest = hashlib.sha256()
    for relative in RELEVANT_PATHS:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing residency evaluator source: {relative}")
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
    return root / "benchmark-runs" / "metal-autoresearch" / f"instruction-input-residency-{stamp}"


def median(values: list[float | int]) -> float:
    return float(statistics.median(values))


def relative_mad(values: list[float]) -> float:
    center = median(values)
    if center == 0.0:
        return 0.0
    return median([abs(value - center) for value in values]) / abs(center)


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_result(result: dict[str, Any], arm: str, seed: int) -> None:
    if set(result) != TOP_LEVEL_FIELDS:
        raise ValueError("residency evaluator result violates its top-level schema")
    if (
        result["schema"] != SCHEMA
        or result["schema_version"] != SCHEMA_VERSION
        or result["kernel"] != "instruction_input"
        or result["arm"] != arm
        or result["all_exact"] is not True
    ):
        raise ValueError("residency evaluator identity or exactness is invalid")
    guards = result["guards"]
    if not isinstance(guards, dict) or not guards or any(value is not True for value in guards.values()):
        raise ValueError("residency evaluator guard failed")
    timings = result["timings"]
    if not isinstance(timings, dict) or set(timings) != TIMING_FIELDS:
        raise ValueError("residency evaluator timing schema is invalid")
    scalar_names = TIMING_FIELDS - {"gpu_command_wall_ns", "gpu_command_active_ns"}
    for name in scalar_names:
        positive_int(timings[name], name)
    command_wall = timings["gpu_command_wall_ns"]
    command_active = timings["gpu_command_active_ns"]
    if (
        not isinstance(command_wall, list)
        or not isinstance(command_active, list)
        or len(command_wall) != LOG_N - CUTOFF_LOG2 + 1
        or len(command_active) != len(command_wall)
        or any(positive_int(value, "GPU command timing") <= 0 for value in command_wall + command_active)
        or any(active > wall for active, wall in zip(command_active, command_wall))
    ):
        raise ValueError("residency evaluator GPU command timings are invalid")
    if (
        sum(command_wall) != timings["gpu_dispatch_wall_ns"]
        or sum(command_active) != timings["gpu_active_ns"]
        or sum(command_wall[:3]) != timings["first_three_gpu_command_wall_ns"]
        or sum(command_active[:3]) != timings["first_three_gpu_command_active_ns"]
        or sum(command_wall[3:]) != timings["later_gpu_command_wall_ns"]
        or sum(command_active[3:]) != timings["later_gpu_command_active_ns"]
    ):
        raise ValueError("residency evaluator GPU timings do not reconcile")
    accounted = (
        timings["gpu_dispatch_wall_ns"]
        + timings["host_round_ns"]
        + timings["readback_ns"]
        + timings["cpu_tail_ns"]
    )
    if accounted > timings["member_wall_ns"]:
        raise ValueError("residency evaluator member timing is under-accounted")
    resources = result["resources"]
    if (
        resources.get("sequence_owned_storage_bytes") != INITIALIZATION_BYTES["full"]
        or resources.get("storage_initialization_bytes") != INITIALIZATION_BYTES[arm]
        or resources.get("storage_initialization_device_buffers") != 6
        or resources.get("persistent_device_buffers") != 6
        or resources.get("round_device_buffer_allocations") != 0
        or resources.get("cutoff_readback_bytes") != 8 * (1 << CUTOFF_LOG2) * 16
    ):
        raise ValueError("residency evaluator resource accounting is invalid")
    identities = resources.get("storage_buffer_identities")
    if not isinstance(identities, list) or len(identities) != 6 or len(set(identities)) != 6:
        raise ValueError("residency evaluator buffer identities are invalid")
    workload = result["workload"]
    if (
        workload.get("log_n") != LOG_N
        or workload.get("cutoff_log2") != CUTOFF_LOG2
        or workload.get("target_sequences") != 1
        or workload.get("excluded_target_warmups") != 0
        or workload.get("storage_initialization_outside_member_timer") is not True
    ):
        raise ValueError("residency evaluator workload contract is invalid")
    fingerprint = result["fingerprint"]
    if (
        fingerprint.get("seed") != seed
        or fingerprint.get("log_n") != LOG_N
        or fingerprint.get("cutoff_log2") != CUTOFF_LOG2
        or fingerprint.get("storage_initialization") != arm
        or fingerprint.get("process_model") != "one_cold_target_sequence_per_process"
    ):
        raise ValueError("residency evaluator fingerprint is invalid")


def run_arm(
    root: Path,
    binary: Path,
    artifact_dir: Path,
    pair_index: int,
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
            "JOLT_METAL_INSTRUCTION_INPUT_STORAGE_INITIALIZATION": arm,
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
    label = f"pair-{pair_index:02d}-{arm}"
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


def summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    minimal = [pair["minimal"] for pair in pairs]
    full = [pair["full"] for pair in pairs]
    minimal_member = [item["timings"]["member_wall_ns"] for item in minimal]
    full_member = [item["timings"]["member_wall_ns"] for item in full]
    paired_ratios = [full_ns / minimal_ns for full_ns, minimal_ns in zip(full_member, minimal_member)]
    paired_improvements = [1.0 - ratio for ratio in paired_ratios]
    minimal_median = median(minimal_member)
    full_median = median(full_member)
    minimal_early = median([item["timings"]["first_three_gpu_command_wall_ns"] for item in minimal])
    full_early = median([item["timings"]["first_three_gpu_command_wall_ns"] for item in full])
    total_savings = minimal_median - full_median
    early_savings = minimal_early - full_early
    early_savings_share = early_savings / total_savings if total_savings > 0 else 0.0
    gap_closed = (
        (minimal_median - full_median) / (minimal_median - STEADY_REFERENCE_NS)
        if minimal_median > STEADY_REFERENCE_NS
        else 0.0
    )
    minimal_active = median([item["timings"]["gpu_active_ns"] for item in minimal])
    full_active = median([item["timings"]["gpu_active_ns"] for item in full])
    active_ratio = full_active / minimal_active
    minimal_wait = median(
        [item["timings"]["gpu_dispatch_wall_ns"] - item["timings"]["gpu_active_ns"] for item in minimal]
    )
    full_wait = median(
        [item["timings"]["gpu_dispatch_wall_ns"] - item["timings"]["gpu_active_ns"] for item in full]
    )
    full_criteria = {
        "all_three_full_faster": all(full_ns < minimal_ns for full_ns, minimal_ns in zip(full_member, minimal_member)),
        "median_full_over_minimal_at_most_0_70": median(paired_ratios) <= FULL_RATIO_LIMIT,
        "gap_closed_at_least_0_80": gap_closed >= GAP_CLOSED_MINIMUM,
        "full_member_at_most_160_ms": full_median <= FULL_MEMBER_LIMIT_NS,
        "early_savings_share_at_least_0_80": early_savings_share >= EARLY_SAVINGS_SHARE_MINIMUM,
        "gpu_active_within_10_percent": abs(active_ratio - 1.0) <= GPU_ACTIVE_RATIO_TOLERANCE,
        "wait_time_collapsed": full_wait < minimal_wait,
    }
    noise_gate = max(0.03, 3.0 * relative_mad(paired_improvements))
    both_fast = minimal_median <= FULL_MEMBER_LIMIT_NS and full_median <= FULL_MEMBER_LIMIT_NS
    full_gain = median(paired_improvements)
    if all(full_criteria.values()):
        decision = "full"
        reason = "full touch cleared every predeclared scratch-residency criterion"
    elif both_fast and full_gain <= noise_gate:
        decision = "minimal"
        reason = "GPU wake was sufficient and full touch added no noise-qualified gain"
    else:
        decision = "none"
        reason = "neither initialization mode cleared its predeclared mechanism gate"
    minimal_prepare_plus_member = [
        item["timings"]["sequence_preparation_ns"] + item["timings"]["member_wall_ns"]
        for item in minimal
    ]
    full_prepare_plus_member = [
        item["timings"]["sequence_preparation_ns"] + item["timings"]["member_wall_ns"]
        for item in full
    ]
    return {
        "decision": decision,
        "reason": reason,
        "acceptance_eligible": False,
        "minimal_member_ns": minimal_member,
        "full_member_ns": full_member,
        "minimal_member_median_ns": minimal_median,
        "full_member_median_ns": full_median,
        "paired_full_over_minimal": paired_ratios,
        "paired_full_fractional_improvements": paired_improvements,
        "paired_ratio_median": median(paired_ratios),
        "paired_improvement_median": full_gain,
        "paired_improvement_relative_mad": relative_mad(paired_improvements),
        "noise_gate": noise_gate,
        "gap_closed": gap_closed,
        "early_savings_share": early_savings_share,
        "minimal_gpu_active_median_ns": minimal_active,
        "full_gpu_active_median_ns": full_active,
        "full_over_minimal_gpu_active": active_ratio,
        "minimal_wait_median_ns": minimal_wait,
        "full_wait_median_ns": full_wait,
        "full_criteria": full_criteria,
        "minimal_prepare_plus_member_median_ns": median(minimal_prepare_plus_member),
        "full_prepare_plus_member_median_ns": median(full_prepare_plus_member),
        "prepare_plus_member_full_over_minimal": median(
            [full_ns / minimal_ns for full_ns, minimal_ns in zip(full_prepare_plus_member, minimal_prepare_plus_member)]
        ),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
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
        args.artifact_dir.resolve() if args.artifact_dir is not None else default_artifact_dir(root)
    )
    try:
        artifact_dir.mkdir(parents=True, exist_ok=False)
        source = source_fingerprint(root)
        contract = {
            "schema": "instruction_input_residency_experiment_v1",
            "created_at": utc_now(),
            "acceptance_eligible": False,
            "question": "does full scratch residency, rather than GPU wake alone, explain cold InstructionInput latency?",
            "arms": {"minimal": INITIALIZATION_BYTES["minimal"], "full": INITIALIZATION_BYTES["full"]},
            "orders": [["minimal", "full"], ["full", "minimal"], ["minimal", "full"]],
            "pairs": PAIRS,
            "log_n": LOG_N,
            "cutoff_log2": CUTOFF_LOG2,
            "steady_reference_ns": STEADY_REFERENCE_NS,
            "full_criteria": {
                "all_three_full_faster": True,
                "median_full_over_minimal_max": FULL_RATIO_LIMIT,
                "gap_closed_min": GAP_CLOSED_MINIMUM,
                "full_member_ns_max": FULL_MEMBER_LIMIT_NS,
                "early_savings_share_min": EARLY_SAVINGS_SHARE_MINIMUM,
                "gpu_active_ratio_tolerance": GPU_ACTIVE_RATIO_TOLERANCE,
                "wait_time_must_collapse": True,
            },
            "holdout": "a selected mode requires a new five-pair production CPU-vs-Metal gate",
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
            "metal-instruction-input-residency-eval",
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
        binary = root / "target" / "release" / "examples" / "metal-instruction-input-residency-eval"
        if not binary.is_file():
            raise ValueError("evaluator build did not produce the expected binary")
        binary_sha256 = file_sha256(binary)
        if source_fingerprint(root) != source:
            raise ValueError("source changed during the evaluator build")
        orders = contract["orders"]
        pairs: list[dict[str, Any]] = []
        started = time.monotonic()
        for index, order in enumerate(orders, 1):
            seed = 0x9E37_79B9 ^ (index * 0x85EB_CA6B)
            arms: dict[str, Any] = {}
            for arm in order:
                arms[arm] = run_arm(
                    root,
                    binary,
                    artifact_dir,
                    index,
                    arm,
                    seed,
                    args.timeout_seconds,
                )
            pairs.append({"index": index, "order": order, "seed": seed, **arms})
        if source_fingerprint(root) != source:
            raise ValueError("source changed during the paired evaluator")
        if file_sha256(binary) != binary_sha256:
            raise ValueError("evaluator binary changed during the paired evaluator")
        summary = summarize(pairs)
        output = {
            "schema": "instruction_input_residency_pairs_v1",
            "schema_version": 1,
            "run_class": "diagnostic",
            "acceptance_eligible": False,
            "source": source,
            "binary_sha256": binary_sha256,
            "contract_sha256": file_sha256(artifact_dir / "run-contract.json"),
            "elapsed_seconds": time.monotonic() - started,
            "summary": summary,
            "pairs": pairs,
            "artifacts": str(artifact_dir),
        }
        (artifact_dir / "result.json").write_bytes(canonical_json(output))
        print(json.dumps(output, sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    with evaluator_lock({"direct_evaluator": "metal_instruction_input_residency_eval"}):
        raise SystemExit(main())
