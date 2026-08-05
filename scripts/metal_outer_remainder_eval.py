#!/usr/bin/env python3
"""Run and score the fixed log-26 Spartan OuterRemainder evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from metal_autoresearch import evaluator_lock
except ModuleNotFoundError:
    from scripts.metal_autoresearch import evaluator_lock


SCHEMA = "outer_remainder_v3"
SCHEMA_VERSION = 3
RUNNER_SCHEMA = "outer_remainder_runner_v2"
FEATURES = "metal,prover-fixtures"
EXAMPLE = "metal-outer-remainder-eval"
LOG_N = 26
PAIRS = 5
ROUNDS = LOG_N + 1
OUTPUT_CLAIMS = 35
FIELD_BYTES = 16
COMPACT_ROW_BYTES = 48
RESIDUAL_ROW_BYTES = 112
STORAGE_BUFFERS = 9
STORAGE_BYTES = 4_300_079_856
DENSE_STORAGE_BYTES = 4 * (1 << 30)
REMAINING_SEQUENCE_STORAGE_BYTES = STORAGE_BYTES - DENSE_STORAGE_BYTES
MAXIMUM_STORAGE_BUFFER_BYTES = 2 * (1 << 30)
MIN_SPEEDUP = 4.0
RAYON_THREADS = 16
TRACE_EPSILON_US = 1e-6

ARM = "OuterRemainderEval::arm"
MEMBER = "OuterRemainder::complete_member"
PREPARE = "OuterRemainder::prepare"
PROVE_ROUND = "OuterRemainder::prove_round"
SUMCHECK_ROUND = "sumcheck_round"
HOST_FS = "sumcheck_host_fiat_shamir"
FINISH = "OuterRemainder::finish_rounds"
OUTPUT = "OuterRemainder::output_claims"
CPU_OUTPUT_WALK = "SpartanOuter::claimed_input_walk"

METAL_PREPARE = "MetalOuterRemainder::prepare"
METAL_SEQUENCE_PREPARE = "MetalOuterRemainder::sequence_prepare"
METAL_ALLOCATION_PLAN = "MetalOuterRemainder::allocation_plan"
METAL_FIRST_MESSAGE = "MetalOuterRemainder::first_message"
METAL_FIRST_BIND = "MetalOuterRemainder::first_bind"
METAL_DENSE_ROUND = "MetalOuterRemainder::dense_round"
METAL_READBACK = "MetalOuterRemainder::readback"
METAL_CPU_TAIL = "MetalOuterRemainder::cpu_tail"
METAL_OUTPUT = "MetalOuterRemainder::output_claims"
METAL_INVALID_ROUND = "MetalOuterRemainder::invalid_round"
METAL_ROW_HANDOFF = "MetalOuterRemainder::row_handoff"
METAL_ROW_RELEASE = "MetalOuterRemainder::row_release"
METAL_STORAGE_PREPARE = "MetalOuterRemainder::storage_prepare"
METAL_STORAGE_INITIALIZE = "MetalOuterRemainder::storage_initialize"

ROW_PREPARE = "MetalInstructionInput::compact_rows_prepare"
ROW_STAGE1_HANDOFF = "MetalInstructionInput::compact_rows_stage1_handoff"
INSTRUCTION_INPUT_PREPARE = "MetalInstructionInput::prepare"

SOURCE_PATHS = (
    "crates/jolt-kernels/src/metal/instruction_read_raf.rs",
    "crates/jolt-prover/examples/metal-outer-remainder-eval.rs",
    "crates/jolt-prover/src/stages/stage1.rs",
    "crates/jolt-kernels/src/metal/spartan_outer.rs",
    "crates/jolt-kernels/src/metal/solinas/fp128.metal",
    "crates/jolt-kernels/src/metal/solinas/mod.rs",
    "crates/jolt-kernels/src/metal/solinas/outer_remainder.rs",
    "crates/jolt-kernels/src/metal/solinas/outer_remainder.metal",
    "crates/jolt-kernels/src/optimized/spartan_outer.rs",
    "crates/jolt-prover/Cargo.toml",
    "scripts/metal_outer_remainder_eval.py",
)


@dataclass(frozen=True)
class Span:
    name: str
    start_us: float
    end_us: float
    args: dict[str, Any]
    pid: Any
    tid: Any

    @property
    def duration_us(self) -> float:
        return self.end_us - self.start_us

    def contains(self, other: "Span") -> bool:
        return (
            self.start_us - TRACE_EPSILON_US <= other.start_us
            and other.end_us <= self.end_us + TRACE_EPSILON_US
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def path_digest(root: Path, paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"fixed evaluator source is missing: {relative}")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def merged_args(begin: Any, end: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(begin, dict):
        result.update(begin)
    if isinstance(end, dict):
        result.update(end)
    return result


def parse_spans(events: list[dict[str, Any]]) -> list[Span]:
    stacks: dict[tuple[Any, Any, str], list[tuple[float, dict[str, Any]]]] = {}
    spans: list[Span] = []
    for event in events:
        name = event.get("name")
        phase = event.get("ph")
        if not isinstance(name, str) or phase not in {"B", "E", "X"}:
            continue
        try:
            timestamp = float(event["ts"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name} has an invalid timestamp") from error
        if not math.isfinite(timestamp):
            raise ValueError(f"{name} has a non-finite timestamp")
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        pid, tid = event.get("pid"), event.get("tid")
        key = (pid, tid, name)
        if phase == "B":
            stacks.setdefault(key, []).append((timestamp, args))
        elif phase == "E":
            starts = stacks.get(key)
            if not starts:
                raise ValueError(f"{name} has an unmatched end event")
            start, begin_args = starts.pop()
            if timestamp <= start:
                raise ValueError(f"{name} has a non-positive duration")
            spans.append(Span(name, start, timestamp, merged_args(begin_args, args), pid, tid))
        else:
            try:
                duration = float(event["dur"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{name} has an invalid complete-event duration") from error
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"{name} has a non-positive duration")
            spans.append(Span(name, timestamp, timestamp + duration, dict(args), pid, tid))
    if any(starts for starts in stacks.values()):
        raise ValueError("trace has an unmatched begin event")
    return sorted(spans, key=lambda span: (span.start_us, -span.end_us))


def descendants(spans: list[Span], parent: Span, name: str | None = None) -> list[Span]:
    return [
        span
        for span in spans
        if span is not parent and parent.contains(span) and (name is None or span.name == name)
    ]


def unique(items: list[Span], description: str) -> Span:
    if len(items) != 1:
        raise ValueError(f"expected exactly one {description}, found {len(items)}")
    return items[0]


def trace_string(value: Any, description: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{description} must be a trace string")
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        if isinstance(decoded, str):
            return decoded
    return value


def trace_int(
    value: Any,
    description: str,
    *,
    positive: bool = False,
    allow_negative: bool = False,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be an integer") from error
    if str(parsed) != str(value) and not isinstance(value, int):
        raise ValueError(f"{description} is not a canonical integer")
    if (parsed < 0 and not allow_negative) or (positive and parsed == 0):
        raise ValueError(f"{description} is out of range")
    return parsed


def trace_bool(value: Any, description: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{description} must be true or false")


def arg_int(span: Span, name: str, *, positive: bool = False) -> int:
    if name not in span.args:
        raise ValueError(f"{span.name} is missing {name}")
    return trace_int(span.args[name], f"{span.name}.{name}", positive=positive)


def arg_bool(span: Span, name: str) -> bool:
    if name not in span.args:
        raise ValueError(f"{span.name} is missing {name}")
    return trace_bool(span.args[name], f"{span.name}.{name}")


def arg_string(span: Span, name: str) -> str:
    if name not in span.args:
        raise ValueError(f"{span.name} is missing {name}")
    return trace_string(span.args[name], f"{span.name}.{name}")


def nested(child: Span, parent: Span) -> bool:
    return parent.contains(child)


def median(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("metric samples must be finite and non-empty")
    return float(statistics.median(values))


def mad(values: list[float]) -> float:
    center = median(values)
    return median([abs(value - center) for value in values])


def parse_outer_remainder_member(
    spans: list[Span],
    arm: Span,
    backend: str,
    cutoff_log2: int,
    trace_cutoff_log2: int,
) -> dict[str, Any]:
    inside = descendants(spans, arm)
    member = unique(descendants(inside, arm, MEMBER), f"{backend} complete member")
    member_spans = descendants(inside, member)
    prepare = unique(descendants(member_spans, member, PREPARE), f"{backend} prepare")
    finish = unique(descendants(member_spans, member, FINISH), f"{backend} finish")
    output = unique(descendants(member_spans, member, OUTPUT), f"{backend} output")
    rounds = descendants(member_spans, member, SUMCHECK_ROUND)
    prove_rounds = descendants(member_spans, member, PROVE_ROUND)
    host_fs = descendants(member_spans, member, HOST_FS)
    rounds.sort(key=lambda span: span.start_us)
    prove_rounds.sort(key=lambda span: span.start_us)
    host_fs.sort(key=lambda span: span.start_us)

    round_indices = [arg_int(span, "round") for span in rounds]
    topology = (
        round_indices == list(range(ROUNDS))
        and len(prove_rounds) == ROUNDS
        and len(host_fs) == ROUNDS
        and all(
            nested(prove_rounds[index], rounds[index])
            and nested(host_fs[index], rounds[index])
            and prove_rounds[index].end_us <= host_fs[index].start_us
            for index in range(ROUNDS)
        )
        and prepare.end_us <= rounds[0].start_us + TRACE_EPSILON_US
        and rounds[-1].end_us <= finish.start_us + TRACE_EPSILON_US
        and finish.end_us <= output.start_us + TRACE_EPSILON_US
    )
    high_level_us = (
        prepare.duration_us
        + sum(span.duration_us for span in rounds)
        + finish.duration_us
        + output.duration_us
    )
    unattributed_us = member.duration_us - high_level_us
    reconciliation_limit_us = max(2_000.0, member.duration_us * 0.02)
    reconciled = -1.0 <= unattributed_us <= reconciliation_limit_us

    result: dict[str, Any] = {
        "member_ns": round(member.duration_us * 1000),
        "component_ns": {
            "prepare": round(prepare.duration_us * 1000),
            "rounds": round(sum(span.duration_us for span in rounds) * 1000),
            "kernel_prove_rounds": round(
                sum(span.duration_us for span in prove_rounds) * 1000
            ),
            "host_fiat_shamir": round(sum(span.duration_us for span in host_fs) * 1000),
            "finish_rounds": round(finish.duration_us * 1000),
            "output_claims": round(output.duration_us * 1000),
            "unattributed": round(unattributed_us * 1000),
        },
        "topology_exact": topology,
        "component_timings_reconciled": reconciled,
        "round_indices": round_indices,
    }

    if backend == "optimized":
        cpu_walks = descendants(member_spans, output, CPU_OUTPUT_WALK)
        result.update(
            cpu_output_walk_count=len(cpu_walks),
            cpu_output_walk_ns=round(sum(span.duration_us for span in cpu_walks) * 1000),
            metal_backend_spans=sorted(
                span.name for span in member_spans if span.name.startswith("MetalOuterRemainder::")
            ),
        )
        return result

    cutoff = 1 << cutoff_log2
    dense_count = ROUNDS - cutoff_log2 - 1
    gpu_last_round = ROUNDS - cutoff_log2
    metal_prepare = unique(descendants(member_spans, prepare, METAL_PREPARE), "Metal prepare")
    sequence = unique(
        descendants(member_spans, metal_prepare, METAL_SEQUENCE_PREPARE),
        "Metal sequence prepare",
    )
    allocation = unique(
        descendants(member_spans, metal_prepare, METAL_ALLOCATION_PLAN),
        "Metal allocation plan",
    )
    first_message = unique(
        descendants(member_spans, metal_prepare, METAL_FIRST_MESSAGE),
        "Metal first message",
    )
    first_bind = unique(descendants(member_spans, rounds[1], METAL_FIRST_BIND), "first bind")
    dense = descendants(member_spans, member, METAL_DENSE_ROUND)
    dense.sort(key=lambda span: span.start_us)
    readback = unique(
        descendants(member_spans, rounds[gpu_last_round + 1], METAL_READBACK),
        "table readback",
    )
    cpu_tail = descendants(member_spans, member, METAL_CPU_TAIL)
    cpu_tail.sort(key=lambda span: span.start_us)
    metal_output = unique(descendants(member_spans, output, METAL_OUTPUT), "Metal outputs")
    row_handoff = unique(descendants(member_spans, metal_prepare, METAL_ROW_HANDOFF), "row handoff")
    row_release = unique(descendants(member_spans, output, METAL_ROW_RELEASE), "row release")

    dense_rounds = [
        index
        for index, round_span in enumerate(rounds)
        for phase in dense
        if nested(phase, round_span)
    ]
    tail_rounds = [
        index
        for index, round_span in enumerate(rounds)
        for phase in cpu_tail
        if nested(phase, round_span)
    ]
    tail_finish_count = sum(nested(phase, finish) for phase in cpu_tail)
    phase_schedule_exact = (
        dense_rounds == list(range(2, gpu_last_round + 1))
        and tail_rounds == list(range(gpu_last_round + 1, ROUNDS))
        and tail_finish_count == 1
        and len(dense) == dense_count
        and len(cpu_tail) == cutoff_log2
        and not descendants(member_spans, member, METAL_INVALID_ROUND)
    )

    gpu_phases = [first_message, first_bind, *dense, metal_output]
    gpu_timing_exact = all(
        (wall := arg_int(span, "dispatch_wall_ns", positive=True))
        >= (active := arg_int(span, "gpu_active_ns", positive=True))
        and active > 0
        for span in gpu_phases
    )
    rows = 1 << LOG_N
    sequence_storage_ids = [
        arg_int(sequence, f"storage_buffer_{index}", positive=True)
        for index in range(STORAGE_BUFFERS)
    ]
    sequence_exact = (
        arg_int(sequence, "resident_rows") == rows
        and arg_int(sequence, "rounds") == ROUNDS
        and arg_int(sequence, "cutoff_elements") == cutoff
        and arg_int(sequence, "trace_cutoff_elements") == 1 << trace_cutoff_log2
        and arg_int(sequence, "row_upload_bytes") == 0
        and arg_int(sequence, "full_domain_copy_dispatches") == 0
        and arg_int(sequence, "sequence_device_buffer_allocations") == 0
        and arg_int(sequence, "round_device_buffer_allocations") == 0
        and arg_bool(sequence, "storage_reused")
        and arg_string(sequence, "storage_initialization_mode") == "full"
        and arg_int(sequence, "planned_device_bytes") == STORAGE_BYTES
        and arg_int(sequence, "preinitialized_device_bytes") == STORAGE_BYTES
        and arg_int(sequence, "initialization_bytes") == STORAGE_BYTES
        and arg_int(sequence, "attached_owned_bytes") == STORAGE_BYTES
        and len(set(sequence_storage_ids)) == STORAGE_BUFFERS
    )
    expected_table_elements = 2 * cutoff
    readback_exact = (
        arg_int(readback, "readbacks") == 1
        and arg_int(readback, "elements") == expected_table_elements
        and arg_int(readback, "bytes") == FIELD_BYTES * expected_table_elements
        and arg_int(metal_output, "readbacks") == 1
        and arg_int(metal_output, "output_elements") == OUTPUT_CLAIMS
        and arg_int(metal_output, "readback_bytes") == FIELD_BYTES * OUTPUT_CLAIMS
        and arg_int(metal_output, "row_upload_bytes") == 0
    )
    resident_bytes = rows * (COMPACT_ROW_BYTES + RESIDUAL_ROW_BYTES)
    allocation_exact = (
        arg_bool(allocation, "admitted")
        and arg_bool(allocation, "storage_reused")
        and arg_int(allocation, "existing_resident_bytes") == resident_bytes
        and arg_int(allocation, "preallocated_device_bytes") == STORAGE_BYTES
        and arg_int(allocation, "additional_working_set_bytes") == 0
        and arg_int(allocation, "current_device_bytes")
        >= arg_int(allocation, "existing_resident_bytes") + STORAGE_BYTES
        and arg_int(allocation, "current_device_bytes")
        <= arg_int(allocation, "recommended_max_working_set_bytes", positive=True)
    )
    identity_exact = (
        arg_int(sequence, "compact_rows_storage_id", positive=True)
        == arg_int(row_handoff, "compact_rows_storage_id", positive=True)
        == arg_int(row_release, "compact_rows_storage_id", positive=True)
        and arg_int(sequence, "residual_rows_storage_id", positive=True)
        == arg_int(row_handoff, "residual_rows_storage_id", positive=True)
        == arg_int(row_release, "residual_rows_storage_id", positive=True)
        and arg_int(sequence, "device_registry_id", positive=True)
        == arg_int(row_handoff, "device_registry_id", positive=True)
        == arg_int(row_release, "device_registry_id", positive=True)
    )
    phase_chronology_exact = (
        allocation.end_us <= row_handoff.start_us + TRACE_EPSILON_US
        and row_handoff.end_us <= sequence.start_us + TRACE_EPSILON_US
        and sequence.end_us <= first_message.start_us + TRACE_EPSILON_US
    )

    result.update(
        {
            "metal_phase_ns": {
                name: [round(span.duration_us * 1000) for span in phase_spans]
                for name, phase_spans in {
                    "prepare": [metal_prepare],
                    "sequence_prepare": [sequence],
                    "allocation_plan": [allocation],
                    "first_message": [first_message],
                    "first_bind": [first_bind],
                    "dense_round": dense,
                    "readback": [readback],
                    "cpu_tail": cpu_tail,
                    "output_claims": [metal_output],
                }.items()
            },
            "metal_dispatch_wall_ns": {
                name: [arg_int(span, "dispatch_wall_ns", positive=True) for span in phase_spans]
                for name, phase_spans in {
                    "first_message": [first_message],
                    "first_bind": [first_bind],
                    "dense_round": dense,
                    "output_claims": [metal_output],
                }.items()
            },
            "metal_gpu_active_ns": {
                name: [arg_int(span, "gpu_active_ns", positive=True) for span in phase_spans]
                for name, phase_spans in {
                    "first_message": [first_message],
                    "first_bind": [first_bind],
                    "dense_round": dense,
                    "output_claims": [metal_output],
                }.items()
            },
            "phase_schedule_exact": phase_schedule_exact,
            "gpu_timing_exact": gpu_timing_exact,
            "sequence_geometry_exact": sequence_exact,
            "readback_exact": readback_exact,
            "allocation_plan_admitted": allocation_exact,
            "row_identity_exact": identity_exact,
            "phase_chronology_exact": phase_chronology_exact,
            "cpu_output_walk_count": len(descendants(member_spans, output, CPU_OUTPUT_WALK)),
            "lifecycle": {
                "handoff": row_handoff.args,
                "sequence": sequence.args,
                "release": row_release.args,
                "storage_buffer_ids": sequence_storage_ids,
            },
        }
    )
    return result


def parse_outer_remainder_storage(
    spans: list[Span], arm: Span, member: Span
) -> dict[str, Any]:
    inside = descendants(spans, arm)
    storage = unique(
        descendants(inside, arm, METAL_STORAGE_PREPARE),
        "outer-remainder storage preparation",
    )
    initialization = unique(
        descendants(inside, storage, METAL_STORAGE_INITIALIZE),
        "outer-remainder storage initialization",
    )
    storage_ids = [
        arg_int(storage, f"buffer_{index}", positive=True)
        for index in range(STORAGE_BUFFERS)
    ]
    initialization_ids = [
        arg_int(initialization, f"buffer_{index}", positive=True)
        for index in range(STORAGE_BUFFERS)
    ]
    initialization_wall_ns = arg_int(
        storage, "initialization_wall_ns", positive=True
    )
    initialization_gpu_active_ns = arg_int(
        storage, "initialization_gpu_active_ns", positive=True
    )
    current_bytes = arg_int(storage, "current_device_bytes")
    recommended_bytes = arg_int(
        storage, "recommended_max_working_set_bytes", positive=True
    )
    exact = (
        arg_int(storage, "cycles") == 1 << LOG_N
        and arg_int(storage, "planned_device_bytes") == STORAGE_BYTES
        and arg_int(storage, "maximum_buffer_bytes")
        == MAXIMUM_STORAGE_BUFFER_BYTES
        and current_bytes + STORAGE_BYTES <= recommended_bytes
        and arg_string(storage, "initialization_mode") == "full"
        and arg_bool(storage, "admitted")
        and arg_bool(storage, "initialized")
        and arg_string(storage, "fallback_reason") == "none"
        and arg_int(storage, "device_buffers") == STORAGE_BUFFERS
        and arg_int(storage, "initialization_bytes") == STORAGE_BYTES
        and initialization_gpu_active_ns <= initialization_wall_ns
        and initialization_wall_ns <= round(storage.duration_us * 1000)
        and arg_string(initialization, "mode") == "full"
        and arg_int(initialization, "device_buffers") == STORAGE_BUFFERS
        and arg_int(initialization, "bytes") == STORAGE_BYTES
        and arg_int(initialization, "protocol_dispatches") == 0
        and storage_ids == initialization_ids
        and len(set(storage_ids)) == STORAGE_BUFFERS
    )
    outside_member = storage.end_us <= member.start_us + TRACE_EPSILON_US
    return {
        "storage_prepare_ns": round(storage.duration_us * 1000),
        "storage_initialization_ns": round(initialization.duration_us * 1000),
        "storage_initialization_wall_ns": initialization_wall_ns,
        "storage_initialization_gpu_active_ns": initialization_gpu_active_ns,
        "storage_exact": exact,
        "storage_outside_member": outside_member,
        "buffer_ids": storage_ids,
        "args": storage.args,
    }


def arm_key(span: Span) -> tuple[int, str]:
    if "sample_index" not in span.args:
        raise ValueError(f"{ARM} is missing sample_index")
    sample = trace_int(
        span.args["sample_index"], f"{ARM}.sample_index", allow_negative=True
    )
    backend = trace_string(span.args.get("backend"), f"{ARM}.backend")
    if backend not in {"optimized", "metal"}:
        raise ValueError(f"unknown evaluator backend {backend!r}")
    return sample, backend


def parse_outer_remainder_result(
    events: list[dict[str, Any]],
    runner: dict[str, Any],
    *,
    source_sha256: str,
    binary_sha256: str,
    artifact_dir: str,
) -> dict[str, Any]:
    if runner.get("schema") != RUNNER_SCHEMA or runner.get("schema_version") != 2:
        raise ValueError("runner output has the wrong schema")
    if runner.get("log_n") != LOG_N or runner.get("pairs") != PAIRS:
        raise ValueError("runner output violates the frozen scale or pair count")
    if runner.get("rayon_threads") != RAYON_THREADS:
        raise ValueError("runner did not pin Rayon to 16 threads")
    expected_orders = [
        ["optimized", "metal"] if pair % 2 == 0 else ["metal", "optimized"]
        for pair in range(PAIRS)
    ]
    if runner.get("orders") != expected_orders:
        raise ValueError("runner output has the wrong alternating order")

    runner_samples = runner.get("samples")
    if not isinstance(runner_samples, list) or len(runner_samples) != PAIRS:
        raise ValueError("runner output must contain five timed pairs")
    warmup = runner.get("warmup")
    if not isinstance(warmup, dict) or warmup.get("excluded_warmup") is not True:
        raise ValueError("runner output must contain one excluded warmup pair")
    all_runner_pairs = [warmup, *runner_samples]
    expected_runner_pairs = [
        (-1, True, ["optimized", "metal"]),
        *[
            (pair, False, expected_orders[pair])
            for pair in range(PAIRS)
        ],
    ]
    if any(
        sample.get("pair") != pair
        or sample.get("excluded_warmup") is not excluded
        or sample.get("order") != order
        for sample, (pair, excluded, order) in zip(
            all_runner_pairs, expected_runner_pairs
        )
    ):
        raise ValueError("runner samples do not match their frozen pair strata")
    correctness_exact = all(
        pair.get("proofs_exact") is True
        and pair.get("optimized", {}).get("proof_verified") is True
        and pair.get("metal", {}).get("proof_verified") is True
        for pair in all_runner_pairs
    )

    spans = parse_spans(events)
    arms = [span for span in spans if span.name == ARM]
    if len(arms) != 2 * (PAIRS + 1):
        raise ValueError("trace must contain one warmup pair and five timed pairs")
    keyed: dict[tuple[int, str], Span] = {}
    for arm in arms:
        key = arm_key(arm)
        if key in keyed:
            raise ValueError(f"duplicate evaluator arm {key}")
        keyed[key] = arm
    expected_keys = {
        (pair, backend)
        for pair in [-1, *range(PAIRS)]
        for backend in ("optimized", "metal")
    }
    if set(keyed) != expected_keys:
        raise ValueError("trace evaluator arms do not match the frozen sample set")

    parameters = runner.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("runner output has no parameter fingerprint")
    integer_parameter_names = {
        "materialize_threads",
        "transition_threads",
        "output_threads",
        "cutoff_log2",
        "trace_cutoff_log2",
    }
    if set(parameters) != integer_parameter_names | {"storage_initialization"}:
        raise ValueError("runner output has the wrong parameter fingerprint")
    parameters = {
        name: trace_int(parameters[name], name, positive=True)
        for name in sorted(integer_parameter_names)
    }
    storage_initialization = trace_string(
        runner["parameters"]["storage_initialization"], "storage_initialization"
    )
    if storage_initialization != "full":
        raise ValueError("frozen evaluator requires full storage initialization")
    parameters["storage_initialization"] = storage_initialization
    cutoff_log2 = parameters["cutoff_log2"]
    trace_cutoff_log2 = parameters["trace_cutoff_log2"]
    if trace_cutoff_log2 > LOG_N or cutoff_log2 < 2 or cutoff_log2 >= ROUNDS - 1:
        raise ValueError("runner cutoffs are outside the frozen evaluator domain")

    trace_rows = trace_int(runner.get("trace_rows"), "trace_rows", positive=True)
    padded_trace_rows = trace_int(
        runner.get("padded_trace_rows"), "padded_trace_rows", positive=True
    )
    actual_arm_order_exact = True
    for pair, _, order in expected_runner_pairs:
        first = keyed[(pair, order[0])]
        second = keyed[(pair, order[1])]
        actual_arm_order_exact &= (
            arg_int(first, "order_position") == 0
            and arg_int(second, "order_position") == 1
            and arg_bool(first, "excluded_warmup") == (pair == -1)
            and arg_bool(second, "excluded_warmup") == (pair == -1)
            and arg_int(first, "trace_rows") == trace_rows
            and arg_int(second, "trace_rows") == trace_rows
            and arg_int(first, "padded_trace_rows") == padded_trace_rows
            and arg_int(second, "padded_trace_rows") == padded_trace_rows
            and first.end_us <= second.start_us + TRACE_EPSILON_US
        )

    parsed: dict[int, dict[str, Any]] = {}
    for pair in [-1, *range(PAIRS)]:
        parsed[pair] = {
            backend: parse_outer_remainder_member(
                spans,
                keyed[(pair, backend)],
                backend,
                cutoff_log2,
                trace_cutoff_log2,
            )
            for backend in ("optimized", "metal")
        }
        metal_arm = keyed[(pair, "metal")]
        metal_member = unique(
            descendants(descendants(spans, metal_arm), metal_arm, MEMBER),
            "Metal member",
        )
        storage = parse_outer_remainder_storage(spans, metal_arm, metal_member)
        parsed[pair]["metal"]["storage"] = storage
        parsed[pair]["metal"]["cold_inclusive_ns"] = (
            parsed[pair]["metal"]["member_ns"] + storage["storage_prepare_ns"]
        )

    rows = 1 << LOG_N
    lifecycle_guards: list[bool] = []
    for pair in [-1, *range(PAIRS)]:
        metal_arm = keyed[(pair, "metal")]
        inside = descendants(spans, metal_arm)
        production = unique(descendants(inside, metal_arm, ROW_PREPARE), "resident row production")
        stage1 = unique(descendants(inside, metal_arm, ROW_STAGE1_HANDOFF), "stage-1 row handoff")
        instruction_input = unique(
            descendants(inside, metal_arm, INSTRUCTION_INPUT_PREPARE),
            "downstream InstructionInput prepare",
        )
        member = unique(descendants(inside, metal_arm, MEMBER), "Metal member")
        storage_prepare = unique(
            descendants(inside, metal_arm, METAL_STORAGE_PREPARE),
            "outer-remainder storage preparation",
        )
        handoff_args = parsed[pair]["metal"]["lifecycle"]["handoff"]
        sequence_args = parsed[pair]["metal"]["lifecycle"]["sequence"]
        release_args = parsed[pair]["metal"]["lifecycle"]["release"]
        storage_ids = parsed[pair]["metal"]["storage"]["buffer_ids"]
        sequence_storage_ids = parsed[pair]["metal"]["lifecycle"][
            "storage_buffer_ids"
        ]
        compact_ids = [
            arg_int(production, "compact_rows_storage_id", positive=True),
            arg_int(stage1, "compact_rows_storage_id", positive=True),
            trace_int(sequence_args.get("compact_rows_storage_id"), "sequence.compact_rows_storage_id", positive=True),
            trace_int(handoff_args.get("compact_rows_storage_id"), "row_handoff.compact_rows_storage_id", positive=True),
            trace_int(release_args.get("compact_rows_storage_id"), "row_release.compact_rows_storage_id", positive=True),
            arg_int(instruction_input, "resident_rows_storage_id", positive=True),
        ]
        residual_ids = [
            arg_int(production, "residual_rows_storage_id", positive=True),
            arg_int(stage1, "residual_rows_storage_id", positive=True),
            trace_int(sequence_args.get("residual_rows_storage_id"), "sequence.residual_rows_storage_id", positive=True),
            trace_int(handoff_args.get("residual_rows_storage_id"), "row_handoff.residual_rows_storage_id", positive=True),
            trace_int(release_args.get("residual_rows_storage_id"), "row_release.residual_rows_storage_id", positive=True),
        ]
        row_counts = [
            arg_int(production, "resident_rows"),
            arg_int(stage1, "resident_rows"),
            trace_int(sequence_args.get("resident_rows"), "sequence.resident_rows"),
            trace_int(handoff_args.get("resident_rows"), "row_handoff.resident_rows"),
            trace_int(release_args.get("resident_rows"), "row_release.resident_rows"),
            arg_int(instruction_input, "resident_rows"),
        ]
        lifecycle_guards.append(
            len(set(compact_ids)) == 1
            and len(set(residual_ids)) == 1
            and row_counts == [rows] * 6
            and trace_int(handoff_args.get("row_upload_bytes"), "row_handoff.row_upload_bytes") == 0
            and trace_int(handoff_args.get("device_allocations"), "row_handoff.device_allocations") == 0
            and trace_int(release_args.get("row_upload_bytes"), "row_release.row_upload_bytes") == 0
            and trace_int(release_args.get("device_allocations"), "row_release.device_allocations") == 0
            and trace_int(release_args.get("residual_row_bytes"), "row_release.residual_row_bytes")
            == rows * RESIDUAL_ROW_BYTES
            and trace_int(
                release_args.get("remaining_sequence_storage_bytes"),
                "row_release.remaining_sequence_storage_bytes",
            )
            == REMAINING_SEQUENCE_STORAGE_BYTES
            and trace_int(
                release_args.get("compact_release_bytes"),
                "row_release.compact_release_bytes",
            )
            == 0
            and trace_int(
                release_args.get("released_owned_bytes"),
                "row_release.released_owned_bytes",
            )
            == rows * RESIDUAL_ROW_BYTES + REMAINING_SEQUENCE_STORAGE_BYTES
            and trace_bool(
                release_args.get("release_completed"),
                "row_release.release_completed",
            )
            and trace_bool(release_args.get("residual_released"), "row_release.residual_released")
            and trace_bool(release_args.get("compact_retained"), "row_release.compact_retained")
            and trace_int(handoff_args.get("device_registry_id"), "row_handoff.device_registry_id", positive=True)
            == trace_int(sequence_args.get("device_registry_id"), "sequence.device_registry_id", positive=True)
            == trace_int(release_args.get("device_registry_id"), "row_release.device_registry_id", positive=True)
            and storage_ids == sequence_storage_ids
            and production.end_us <= storage_prepare.start_us + TRACE_EPSILON_US
            and storage_prepare.end_us <= stage1.start_us + TRACE_EPSILON_US
            and stage1.end_us <= member.start_us + TRACE_EPSILON_US
            and member.end_us <= instruction_input.start_us
        )

    timed_samples = [parsed[pair] for pair in range(PAIRS)]
    cpu_ns = [float(sample["optimized"]["member_ns"]) for sample in timed_samples]
    metal_ns = [float(sample["metal"]["member_ns"]) for sample in timed_samples]
    storage_prepare_ns = [
        float(sample["metal"]["storage"]["storage_prepare_ns"])
        for sample in timed_samples
    ]
    cold_inclusive_ns = [
        float(sample["metal"]["cold_inclusive_ns"]) for sample in timed_samples
    ]
    paired_speedups = [cpu / gpu for cpu, gpu in zip(cpu_ns, metal_ns)]
    cold_inclusive_speedups = [
        cpu / gpu for cpu, gpu in zip(cpu_ns, cold_inclusive_ns)
    ]
    improvements = [1.0 - gpu / cpu for cpu, gpu in zip(cpu_ns, metal_ns)]
    cpu_first = [paired_speedups[pair] for pair in range(PAIRS) if pair % 2 == 0]
    metal_first = [paired_speedups[pair] for pair in range(PAIRS) if pair % 2 == 1]
    median_speedup = median(paired_speedups)
    median_improvement = median(improvements)
    improvement_mad = mad(improvements)

    guards = {
        "correctness_exact": correctness_exact,
        "sample_cardinality_exact": len(timed_samples) == PAIRS,
        "alternating_orders_exact": runner.get("orders") == expected_orders,
        "actual_arm_order_exact": actual_arm_order_exact,
        "rayon_threads_pinned": runner.get("rayon_threads") == RAYON_THREADS,
        "target_scale": runner.get("padded_trace_rows") == rows,
        "round_topology_exact": all(
            sample[backend]["topology_exact"]
            for sample in parsed.values()
            for backend in ("optimized", "metal")
        ),
        "component_timings_reconciled": all(
            sample[backend]["component_timings_reconciled"]
            for sample in parsed.values()
            for backend in ("optimized", "metal")
        ),
        "optimized_cpu_output_walk_exact": all(
            sample["optimized"]["cpu_output_walk_count"] == 1
            for sample in parsed.values()
        ),
        "metal_output_replaces_cpu_walk": all(
            sample["metal"]["cpu_output_walk_count"] == 0 for sample in parsed.values()
        ),
        "metal_phase_schedule_exact": all(
            sample["metal"]["phase_schedule_exact"] for sample in parsed.values()
        ),
        "metal_gpu_timing_exact": all(
            sample["metal"]["gpu_timing_exact"] for sample in parsed.values()
        ),
        "metal_sequence_geometry_exact": all(
            sample["metal"]["sequence_geometry_exact"] for sample in parsed.values()
        ),
        "metal_phase_chronology_exact": all(
            sample["metal"]["phase_chronology_exact"] for sample in parsed.values()
        ),
        "metal_row_identity_exact": all(
            sample["metal"]["row_identity_exact"] for sample in parsed.values()
        ),
        "metal_readback_exact": all(
            sample["metal"]["readback_exact"] for sample in parsed.values()
        ),
        "metal_working_set_admitted": all(
            sample["metal"]["allocation_plan_admitted"] for sample in parsed.values()
        ),
        "metal_storage_preparation_exact": all(
            sample["metal"]["storage"]["storage_exact"]
            for sample in parsed.values()
        ),
        "metal_storage_preparation_outside_member": all(
            sample["metal"]["storage"]["storage_outside_member"]
            for sample in parsed.values()
        ),
        "resident_row_lifecycle_exact": all(lifecycle_guards),
        "optimized_has_no_metal_member_spans": all(
            not sample["optimized"]["metal_backend_spans"] for sample in parsed.values()
        ),
        "member_durations_positive": all(value > 0 for value in [*cpu_ns, *metal_ns]),
        "speedups_finite_positive": all(
            math.isfinite(value) and value > 0 for value in paired_speedups
        ),
    }
    all_exact = all(guards.values())
    guards["all_exact"] = all_exact
    local_gate = (
        all_exact
        and median_speedup >= MIN_SPEEDUP
        and median(cpu_first) >= MIN_SPEEDUP
        and median(metal_first) >= MIN_SPEEDUP
        and median(metal_ns) < median(cpu_ns)
        and median_improvement > 3.0 * improvement_mad
    )

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kernel": "OuterRemainder",
        "workload": "fibonacci",
        "fingerprint": {
            "fixture": runner.get("fixture"),
            "log_n": LOG_N,
            "trace_elements": rows,
            "trace_rows": runner.get("trace_rows"),
            "pairs": PAIRS,
            "excluded_warmup_pairs": 1,
            "orders": expected_orders,
            "rayon_threads": RAYON_THREADS,
            **parameters,
            "member_span": MEMBER,
            "rounds": ROUNDS,
            "output_claims": OUTPUT_CLAIMS,
            "source_sha256": source_sha256,
            "binary_sha256": binary_sha256,
        },
        "metrics": {
            "hybrid_speedup": median_speedup,
            "cpu_member_ns_samples": cpu_ns,
            "metal_member_ns_samples": metal_ns,
            "paired_speedups": paired_speedups,
            "storage_prepare_ns_samples": storage_prepare_ns,
            "metal_cold_inclusive_ns_samples": cold_inclusive_ns,
            "cold_inclusive_speedups": cold_inclusive_speedups,
            "median_cpu_member_ns": median(cpu_ns),
            "median_metal_member_ns": median(metal_ns),
            "median_paired_speedup": median_speedup,
            "median_storage_prepare_ns": median(storage_prepare_ns),
            "median_metal_cold_inclusive_ns": median(cold_inclusive_ns),
            "median_cold_inclusive_speedup": median(cold_inclusive_speedups),
            "cpu_first_median_speedup": median(cpu_first),
            "metal_first_median_speedup": median(metal_first),
            "median_fractional_improvement": median_improvement,
            "fractional_improvement_mad": improvement_mad,
        },
        "samples": [
            {"pair": pair, "order": expected_orders[pair], **parsed[pair]}
            for pair in range(PAIRS)
        ],
        "excluded_warmup": parsed[-1],
        "guards": guards,
        "all_exact": all_exact,
        "resources": {
            "compact_row_bytes": COMPACT_ROW_BYTES,
            "residual_row_bytes": RESIDUAL_ROW_BYTES,
            "resident_row_bytes": rows * (COMPACT_ROW_BYTES + RESIDUAL_ROW_BYTES),
            "outer_remainder_storage_bytes": STORAGE_BYTES,
            "maximum_storage_buffer_bytes": MAXIMUM_STORAGE_BUFFER_BYTES,
            "table_readback_bytes": FIELD_BYTES * 2 * (1 << cutoff_log2),
            "output_readback_bytes": FIELD_BYTES * OUTPUT_CLAIMS,
        },
        "promotion": {
            "eligible": local_gate,
            "minimum_speedup": MIN_SPEEDUP,
            "production_holdout_required": True,
            "continue_above_floor": True,
        },
        "oracle_limits": [
            "Chrome span wall time is authoritative; gpu_active_ns is implementation telemetry",
            "cold_inclusive adds only OuterRemainder storage preparation to the resident member",
            "full-proof equality assumes deterministic clear Akita proving",
            "promotion still requires a separate five-pair production PIOP holdout",
        ],
        "artifacts": artifact_dir,
    }


def parse_runner_stdout(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("runner must emit exactly one non-empty stdout line")
    parsed = json.loads(lines[0])
    if not isinstance(parsed, dict):
        raise ValueError("runner output must be a JSON object")
    return parsed


def load_events(path: Path) -> list[dict[str, Any]]:
    parsed = json.loads(path.read_text())
    if not isinstance(parsed, list) or not all(isinstance(event, dict) for event in parsed):
        raise ValueError("trace must be a JSON event array")
    return parsed


def parser() -> argparse.ArgumentParser:
    def env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--log-n", type=int, default=env_int("JOLT_METAL_EVAL_LOG_N", LOG_N)
    )
    result.add_argument(
        "--pairs", type=int, default=env_int("JOLT_METAL_EVAL_REPEATS", PAIRS)
    )
    result.add_argument(
        "--materialize-threads",
        type=int,
        choices=[128, 256, 512],
        default=env_int("JOLT_METAL_OUTER_REMAINDER_MATERIALIZE_THREADS", 256),
    )
    result.add_argument(
        "--transition-threads",
        type=int,
        choices=[64, 128, 256, 512],
        default=env_int("JOLT_METAL_OUTER_REMAINDER_TRANSITION_THREADS", 128),
    )
    result.add_argument(
        "--output-threads",
        type=int,
        choices=[128, 256, 512],
        default=env_int("JOLT_METAL_OUTER_REMAINDER_OUTPUT_THREADS", 256),
    )
    result.add_argument(
        "--cutoff-log2",
        type=int,
        choices=[14, 15, 16, 17, 18],
        default=env_int("JOLT_METAL_OUTER_REMAINDER_CUTOFF_LOG2", 16),
    )
    result.add_argument(
        "--trace-cutoff-log2",
        type=int,
        choices=[18],
        default=env_int("JOLT_METAL_OUTER_REMAINDER_TRACE_CUTOFF_LOG2", 18),
    )
    result.add_argument("--timeout-seconds", type=int, default=7200)
    result.add_argument("--artifact-dir", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.log_n != LOG_N or args.pairs != PAIRS:
        raise ValueError("outer_remainder_v3 is frozen at log_n=26 and five pairs")
    root = Path(__file__).resolve().parents[1]
    artifact_dir = args.artifact_dir or (
        root
        / "benchmark-runs"
        / "metal-outer-remainder-eval"
        / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    trace_path = artifact_dir / "trace.json"
    binary = root / "target" / "release" / "examples" / EXAMPLE

    source_before = path_digest(root, SOURCE_PATHS)
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-q",
            "-p",
            "jolt-prover",
            "--features",
            FEATURES,
            "--example",
            EXAMPLE,
        ],
        cwd=root,
        check=True,
        timeout=args.timeout_seconds,
    )
    if not binary.is_file():
        raise ValueError(f"evaluator binary is missing: {binary}")
    binary_sha256 = sha256_bytes(binary.read_bytes())

    command = [
        str(binary),
        "--log-n",
        str(args.log_n),
        "--pairs",
        str(args.pairs),
        "--trace-path",
        str(trace_path),
        "--materialize-threads",
        str(args.materialize_threads),
        "--transition-threads",
        str(args.transition_threads),
        "--output-threads",
        str(args.output_threads),
        "--cutoff-log2",
        str(args.cutoff_log2),
        "--trace-cutoff-log2",
        str(args.trace_cutoff_log2),
    ]
    environment = os.environ.copy()
    environment.update(RAYON_NUM_THREADS=str(RAYON_THREADS), RUST_LOG="warn")
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
    )
    (artifact_dir / "stdout.log").write_text(completed.stdout)
    (artifact_dir / "stderr.log").write_text(completed.stderr)
    (artifact_dir / "command.json").write_text(
        json.dumps({"command": command, "environment": {"RAYON_NUM_THREADS": "16"}}, sort_keys=True)
        + "\n"
    )
    source_after = path_digest(root, SOURCE_PATHS)
    if source_before != source_after:
        raise ValueError("fixed evaluator sources changed during the run")
    binary_sha256_after = sha256_bytes(binary.read_bytes())
    if binary_sha256 != binary_sha256_after:
        raise ValueError("fixed evaluator binary changed during the run")

    output = parse_outer_remainder_result(
        load_events(trace_path),
        parse_runner_stdout(completed.stdout),
        source_sha256=source_after,
        binary_sha256=binary_sha256,
        artifact_dir=str(artifact_dir),
    )
    output["run"] = {
        "created_at": utc_now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "command": command,
    }
    encoded = json.dumps(output, sort_keys=True)
    (artifact_dir / "result.json").write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    with evaluator_lock({"direct_evaluator": SCHEMA}):
        try:
            raise SystemExit(main())
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(2) from error
