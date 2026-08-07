#!/usr/bin/env python3
"""Measure optimized-CPU and Metal-hybrid Akita PIOP wall time."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from metal_autoresearch import evaluator_lock
except ModuleNotFoundError:
    from scripts.metal_autoresearch import evaluator_lock


SCHEMA_VERSION = 10
FEATURES = "metal,prover-fixtures"
PIOP_SPAN = "jolt_prover::piop"
BACKEND_WITNESS_PREP_SPAN = "jolt_prover::backend_witness_prepare"
SPARTAN_SHIFT_PREPARE_SPAN = "SpartanShift::prepare"
BYTECODE_KERNEL = "BytecodeReadRafCycle"
BYTECODE_COMPONENTS = ("prepare", "prove_round", "finish_rounds", "output_claims")
BYTECODE_METAL_PHASES = (
    "prepare",
    "allocation_plan",
    "first_message",
    "first_bind",
    "dense_round",
    "readback",
    "cpu_tail",
    "invalid_round",
)
BYTECODE_MIN_SPEEDUP = 5.0
INSTRUCTION_INPUT_KERNEL = "InstructionInput"
INSTRUCTION_INPUT_COMPONENTS = ("prepare", "prove_round", "finish_rounds", "output_claims")
INSTRUCTION_INPUT_METAL_PHASES = (
    "storage_prepare",
    "allocation_plan",
    "storage_initialize",
    "storage_initialize_complete",
    "outer_residual_transfer",
    "native_primer_submit",
    "prepare",
    "native_primer_join",
    "native_primer_complete",
    "first_message",
    "first_bind",
    "dense_round",
    "readback",
    "cpu_tail",
)
INSTRUCTION_INPUT_MIN_SPEEDUP = 5.0
INSTRUCTION_READ_RAF_KERNEL = "InstructionReadRaf"
INSTRUCTION_READ_RAF_MIN_SPEEDUP = 5.0
BOOLEANITY_ADDRESS_KERNEL = "BooleanityAddressPhase"
BOOLEANITY_ADDRESS_COMPONENTS = (
    "prepare",
    "prove_round",
    "finish_rounds",
    "output_claims",
)
BOOLEANITY_ADDRESS_METAL_PHASES = (
    "prepare",
    "sequence_prepare",
    "allocation_plan",
    "dispatch",
    "readback",
)
BOOLEANITY_ADDRESS_MIN_SPEEDUP = 5.0
HAMMING_WEIGHT_KERNEL = "HammingWeightClaimReduction"
HAMMING_WEIGHT_METAL_PHASES = BOOLEANITY_ADDRESS_METAL_PHASES
HAMMING_WEIGHT_MIN_SPEEDUP = 5.0
BOOLEANITY_HAMMING_MIN_SPEEDUP = 5.0
OUTER_REMAINDER_KERNEL = "OuterRemainder"
OUTER_REMAINDER_COMPLETE_MEMBER = "OuterRemainder::complete_member"
OUTER_REMAINDER_METAL_PHASES = (
    "storage_prepare",
    "storage_initialize",
    "storage_initialize_complete",
    "prepare",
    "allocation_plan",
    "row_handoff",
    "sequence_prepare",
    "first_message",
    "first_bind",
    "dense_round",
    "readback",
    "cpu_tail",
    "output_claims",
    "row_release",
    "product_uniskip_carrier_park",
)
OUTER_REMAINDER_MIN_SPEEDUP = 5.0
OUTER_REMAINDER_A_LOOKUP_FIELDS = 202
PRODUCT_UNISKIP_KERNEL = "SpartanProductUniskip"
METAL_PRODUCT_UNISKIP_STANDALONE = "MetalProductUniskip::prepare"
METAL_PRODUCT_UNISKIP_CARRIER = "MetalProductUniskip::outer_opening_carrier"
PRODUCT_REMAINDER_KERNEL = "ProductRemainder"
PRODUCT_REMAINDER_MIN_SPEEDUP = 5.0
OUTER_PRODUCT_FAMILY_MIN_SPEEDUP = 5.0
INSTRUCTION_CLAIM_KERNEL = "InstructionClaimReduction"
INSTRUCTION_CLAIM_MIN_SPEEDUP = 5.0
OPTIMIZED_HAMMING_WEIGHT_ROW_SOURCE = (
    "OptimizedHammingWeightClaimReduction::row_source"
)
SUMCHECK_ROUND_SPAN = "sumcheck_round"
SUMCHECK_HOST_FIAT_SHAMIR_SPAN = "sumcheck_host_fiat_shamir"
OPTIMIZED_BOOLEANITY_ADDRESS_ROW_SOURCE = "OptimizedBooleanityAddress::row_source"
PRODUCTION_RAYON_THREADS = 16
METAL_BOOLEANITY_ROWS_STAGE5_PREPARE = "MetalBooleanityRows::stage5_prepare"
METAL_BOOLEANITY_ROWS_STAGE6A_USE = "MetalBooleanityRows::stage6a_address_use"
METAL_BOOLEANITY_ROWS_STAGE6B_USE = "MetalBooleanityRows::stage6b_cycle_use"
METAL_BOOLEANITY_ROWS_STAGE6B_RETAIN = (
    "MetalBooleanityRows::stage6b_retain_for_stage7"
)
METAL_BOOLEANITY_ROWS_STAGE7_HAMMING_USE = (
    "MetalBooleanityRows::stage7_hamming_use"
)
METAL_HAMMING_HOT_STAGE6B_RETAIN = "MetalHammingHotRows::stage6b_retain_for_stage7"
METAL_HAMMING_HOT_STAGE7_USE = "MetalHammingHotRows::stage7_terminal_use"
OPTIMIZED_INSTRUCTION_INPUT_ROWS_PREPARE = "OptimizedInstructionInput::rows_prepare"
OPTIMIZED_INSTRUCTION_INPUT_ROWS_STAGE3_USE = (
    "OptimizedInstructionInput::rows_stage3_use"
)
METAL_INSTRUCTION_INPUT_ROWS_PREPARE = (
    "MetalInstructionInput::compact_rows_prepare"
)
METAL_INSTRUCTION_INPUT_ROWS_STAGE1_HANDOFF = (
    "MetalInstructionInput::compact_rows_stage1_handoff"
)
METAL_OUTER_REMAINDER_ROW_RELEASE = "MetalOuterRemainder::row_release"
PRODUCTION_PAIRS = 5
LOCAL_KERNELS = {
    INSTRUCTION_READ_RAF_KERNEL: {
        "name": INSTRUCTION_READ_RAF_KERNEL,
        "metric": "instruction_read_raf_speedup",
        "paired_metric": "paired_instruction_read_raf_speedups",
        "backend_prefix": "MetalInstructionReadRaf::",
    },
    "OuterRemainder": {
        "name": OUTER_REMAINDER_KERNEL,
        "metric": "outer_remainder_speedup",
        "paired_metric": "paired_outer_remainder_speedups",
        "backend_prefix": "MetalOuterRemainder::",
    },
    "BytecodeReadRafCycle": {
        "name": BYTECODE_KERNEL,
        "metric": "bytecode_read_raf_cycle_speedup",
        "paired_metric": "paired_bytecode_read_raf_cycle_speedups",
        "backend_prefix": "MetalBytecodeReadRafCycle::",
    },
    "InstructionRaVirtualization": {
        "name": "InstructionRaVirtualization",
        "metric": "instruction_ra_speedup",
        "paired_metric": "paired_instruction_ra_speedups",
        "backend_prefix": "MetalInstructionRaVirtualization::",
    },
    "InstructionInput": {
        "name": INSTRUCTION_INPUT_KERNEL,
        "metric": "instruction_input_kernel_service_speedup",
        "paired_metric": "paired_instruction_input_kernel_service_speedups",
        "backend_prefix": "MetalInstructionInput::",
    },
    "BooleanityAddressPhase": {
        "name": BOOLEANITY_ADDRESS_KERNEL,
        "metric": "booleanity_address_phase_speedup",
        "paired_metric": "paired_booleanity_address_phase_speedups",
        "backend_prefix": "MetalBooleanityAddressPhase::",
    },
    "HammingWeightClaimReduction": {
        "name": HAMMING_WEIGHT_KERNEL,
        "metric": "hamming_weight_claim_reduction_speedup",
        "paired_metric": "paired_hamming_weight_claim_reduction_speedups",
        "backend_prefix": "MetalHammingWeightClaimReduction::",
    },
    "ProductRemainder": {
        "name": PRODUCT_REMAINDER_KERNEL,
        "metric": "product_remainder_speedup",
        "paired_metric": "paired_product_remainder_speedups",
        "backend_prefix": "MetalProductRemainder::",
    },
    "InstructionClaimReduction": {
        "name": INSTRUCTION_CLAIM_KERNEL,
        "metric": "instruction_claim_reduction_critical_path_speedup",
        "paired_metric": "paired_instruction_claim_reduction_critical_path_speedups",
        "backend_prefix": "MetalInstructionClaimReduction::",
    },
}


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


def instruction_input_sequence_auxiliary_storage_bytes(log_n: int) -> int:
    rows = 1 << log_n
    return instruction_input_sequence_storage_bytes(log_n) - 96 * rows


def booleanity_address_sequence_storage_bytes(
    log_n: int, inner_log2: int, selectors_per_tile: int
) -> int:
    if log_n < 0 or inner_log2 < 0 or inner_log2 > log_n:
        raise ValueError("invalid Booleanity address storage geometry")
    rows = 1 << log_n
    e_in = 1 << inner_log2
    if e_in > rows or selectors_per_tile < 1:
        raise ValueError("invalid Booleanity address storage geometry")
    e_out = rows // e_in
    selector_bytes = 29 * 8
    weight_bytes = 16 * (e_in + e_out)
    partial_bytes = 16 * e_out * selectors_per_tile * 256
    output_bytes = 16 * 29 * 256
    return selector_bytes + weight_bytes + partial_bytes + output_bytes


BYTECODE_CONFIG_RE = re.compile(
    r"^BYTECODE_CYCLE_CONFIG requested=(?P<requested>\S+) "
    r"effective=(?P<effective>\S+) log_t=(?P<log_t>\d+) "
    r"log_k=(?P<log_k>\d+) chunk_bits=(?P<chunk_bits>\d+) "
    r"num_ra=(?P<num_ra>\d+) degree=(?P<degree>\d+)$"
)
BYTECODE_METAL_CONFIG_RE = re.compile(
    r"^BYTECODE_METAL_CONFIG backend=metal cpu_tail=(?P<cpu_tail>\S+) "
    r"trace_cutoff=(?P<trace_cutoff>\d+) cutoff=(?P<cutoff>\d+) "
    r"message_threads=(?P<message_threads>\d+) "
    r"transition_threads=(?P<transition_threads>\d+) "
    r"max_threadgroups=(?P<max_threadgroups>\d+)$"
)
INSTRUCTION_INPUT_METAL_CONFIG_RE = re.compile(
    r"^INSTRUCTION_INPUT_METAL_CONFIG backend=metal "
    r"trace_cutoff=(?P<trace_cutoff>\d+) cutoff=(?P<cutoff>\d+) "
    r"native_message_threads=(?P<native_message_threads>\d+) "
    r"native_transition_threads=(?P<native_transition_threads>\d+) "
    r"dense_transition_threads=(?P<dense_transition_threads>\d+) "
    r"storage_initialization=(?P<storage_initialization>\S+) "
    r"dense_storage_mode=(?P<dense_storage_mode>\S+) "
    r"native_primer=(?P<native_primer>\S+)$"
)
BOOLEANITY_ADDRESS_METAL_CONFIG_RE = re.compile(
    r"^BOOLEANITY_ADDRESS_METAL_CONFIG backend=metal "
    r"trace_cutoff=(?P<trace_cutoff>\d+) "
    r"inner_log2=(?P<inner_log2>\d+) "
    r"selectors_per_tile=(?P<selectors_per_tile>\d+) "
    r"tile_threads=(?P<tile_threads>\d+) "
    r"finalize_threads=(?P<finalize_threads>\d+)$"
)
BOOLEANITY_ADDRESS_METAL_IMPLEMENTATION_RE = re.compile(
    r"^BOOLEANITY_ADDRESS_METAL_IMPLEMENTATION value=(?P<implementation>accepted|packed-hot)$"
)
HAMMING_WEIGHT_METAL_CONFIG_RE = re.compile(
    r"^HAMMING_WEIGHT_METAL_CONFIG backend=metal "
    r"trace_cutoff=(?P<trace_cutoff>\d+) "
    r"inner_log2=(?P<inner_log2>\d+) "
    r"selectors_per_tile=(?P<selectors_per_tile>\d+) "
    r"tile_threads=(?P<tile_threads>\d+) "
    r"finalize_threads=(?P<finalize_threads>\d+)$"
)
HAMMING_WEIGHT_METAL_IMPLEMENTATION_RE = re.compile(
    r"^HAMMING_WEIGHT_METAL_IMPLEMENTATION value=(?P<implementation>accepted-rows|retained-hot)$"
)
OUTER_REMAINDER_METAL_CONFIG_RE = re.compile(
    r"^OUTER_REMAINDER_METAL_CONFIG backend=metal "
    r"trace_cutoff=(?P<trace_cutoff>\d+) cutoff=(?P<cutoff>\d+) "
    r"materialize_threads=(?P<materialize_threads>\d+) "
    r"transition_threads=(?P<transition_threads>\d+) "
    r"output_threads=(?P<output_threads>\d+) "
    r"max_threadgroups=(?P<max_threadgroups>\d+) "
    r"binding_plan=(?P<binding_plan>\S+) "
    r"storage_initialization=(?P<storage_initialization>\S+) "
    r"product_uniskip_carrier=(?P<product_uniskip_carrier>true|false)$"
)
PIOP_EXECUTION_CONFIG_RE = re.compile(
    r"^PIOP_EXECUTION_CONFIG rayon_threads=(?P<rayon_threads>\d+)$"
)
MAX_RSS_RE = re.compile(r"^\s*(?P<bytes>\d+)\s+maximum resident set size\s*$", re.MULTILINE)


def span_intervals_us(
    events: list[dict[str, Any]], selected_name: Optional[str] = None
) -> list[tuple[str, float, float]]:
    stacks: dict[tuple[Any, Any, str], list[float]] = {}
    intervals: list[tuple[str, float, float]] = []
    for event in events:
        name = event.get("name")
        if not isinstance(name, str) or (selected_name is not None and name != selected_name):
            continue
        phase = event.get("ph")
        if phase == "X":
            start = float(event["ts"])
            intervals.append((name, start, start + float(event["dur"])))
            continue
        key = (event.get("pid"), event.get("tid"), name)
        if phase == "B":
            stacks.setdefault(key, []).append(float(event["ts"]))
        elif phase == "E":
            starts = stacks.get(key)
            if not starts:
                raise ValueError(f"trace has an unmatched end event for {name}")
            intervals.append((name, starts.pop(), float(event["ts"])))
    if any(starts for starts in stacks.values()):
        raise ValueError("trace has an unmatched begin event")
    return intervals


def strict_named_intervals(
    events: list[dict[str, Any]], names: set[str]
) -> dict[str, list[tuple[float, float]]]:
    intervals = {name: [] for name in names}
    stacks: dict[tuple[Any, Any, str], list[float]] = {}
    for event in events:
        name = event.get("name")
        if name not in names:
            continue
        try:
            timestamp = float(event["ts"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name} has an invalid timestamp") from error
        if not math.isfinite(timestamp):
            raise ValueError(f"{name} has a non-finite timestamp")
        phase = event.get("ph")
        if phase == "X":
            try:
                duration = float(event["dur"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{name} has an invalid duration") from error
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError(f"{name} has a non-positive duration")
            intervals[name].append((timestamp, timestamp + duration))
        elif phase == "B":
            stacks.setdefault((event.get("pid"), event.get("tid"), name), []).append(
                timestamp
            )
        elif phase == "E":
            starts = stacks.get((event.get("pid"), event.get("tid"), name))
            if not starts:
                raise ValueError(f"{name} has an unmatched end event")
            start = starts.pop()
            if timestamp <= start:
                raise ValueError(f"{name} has a non-positive duration")
            intervals[name].append((start, timestamp))
        else:
            raise ValueError(f"{name} has unsupported trace phase {phase!r}")
    if any(starts for starts in stacks.values()):
        raise ValueError("trace has an unmatched begin event")
    return intervals


def unique_named_span_duration_us(events: list[dict[str, Any]], name: str) -> float:
    intervals = span_intervals_us(events, name)
    durations = [end - start for _, start, end in intervals]
    if len(durations) != 1 or not math.isfinite(durations[0]) or durations[0] <= 0.0:
        raise ValueError(f"trace must contain exactly one positive {name} span")
    return durations[0]


def unique_span_duration_us(events: list[dict[str, Any]]) -> float:
    return unique_named_span_duration_us(events, PIOP_SPAN)


def load_trace_events(path: Path) -> list[dict[str, Any]]:
    events = json.loads(path.read_text())
    if not isinstance(events, list):
        raise ValueError(f"{path} must contain a trace event array")
    return events


def union_duration_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def trace_attribution(events: list[dict[str, Any]]) -> dict[str, Any]:
    piop_intervals = span_intervals_us(events, PIOP_SPAN)
    if len(piop_intervals) != 1:
        raise ValueError("trace must contain exactly one PIOP span for attribution")
    _, piop_start, piop_end = piop_intervals[0]
    piop_us = piop_end - piop_start
    intervals_by_name: dict[str, list[tuple[float, float]]] = {}
    for name, start, end in span_intervals_us(events):
        if start >= piop_start and end <= piop_end:
            intervals_by_name.setdefault(name, []).append((start, end))
    durations = {
        name: union_duration_us(intervals) for name, intervals in intervals_by_name.items()
    }

    stages = {
        name: duration / 1000.0
        for name, duration in durations.items()
        if name.startswith("prove_stage")
    }
    kernel_durations: dict[str, float] = {}
    suffixes = (
        "::prepare",
        "::first_round_poly",
        "::prove_round",
        "::finish_rounds",
        "::output_claims",
    )
    for name, duration in durations.items():
        suffix = next((suffix for suffix in suffixes if name.endswith(suffix)), None)
        if suffix is not None:
            kernel = name[: -len(suffix)]
            kernel_durations[kernel] = kernel_durations.get(kernel, 0.0) + duration
    kernels = [
        {
            "kernel": kernel,
            "wall_ms": duration / 1000.0,
            "piop_share": duration / piop_us,
        }
        for kernel, duration in kernel_durations.items()
    ]
    kernels.sort(key=lambda item: item["wall_ms"], reverse=True)
    backend_spans = [
        {
            "span": name,
            "wall_ms": union_duration_us(intervals) / 1000.0,
            "piop_share": union_duration_us(intervals) / piop_us,
            "occurrences": len(intervals),
        }
        for name, intervals in intervals_by_name.items()
        if name.startswith("Metal")
    ]
    backend_spans.sort(key=lambda item: item["wall_ms"], reverse=True)
    return {
        "piop_ms": piop_us / 1000.0,
        "stage_ms": stages,
        "kernels": kernels,
        "backend_spans": backend_spans,
    }


def kernel_wall_us(attribution: dict[str, Any], kernel: str) -> float:
    matches = [
        float(item["wall_ms"]) * 1000.0
        for item in attribution.get("kernels", [])
        if item.get("kernel") == kernel
    ]
    if len(matches) != 1 or not math.isfinite(matches[0]) or matches[0] <= 0.0:
        raise ValueError(f"attribution must contain one positive {kernel} seam")
    return matches[0]


def positive_span_count(events: list[dict[str, Any]], name: str) -> int:
    intervals = span_intervals_us(events, name)
    if any(not math.isfinite(end - start) or end <= start for _, start, end in intervals):
        raise ValueError(f"trace contains a non-positive {name} span")
    return len(intervals)


def validate_bytecode_stdout(
    stdout: str,
    backend: str,
    log_n: int,
    message_threads: int = 256,
    transition_threads: int = 128,
    max_threadgroups: int = 8192,
    cutoff_log2: int = 16,
    trace_cutoff_log2: int = 18,
) -> dict[str, Any]:
    configs = [
        match
        for line in stdout.splitlines()
        if (match := BYTECODE_CONFIG_RE.fullmatch(line)) is not None
    ]
    if len(configs) != 1:
        raise ValueError("evaluator must emit exactly one Bytecode cycle config record")
    parsed_config = configs[0].groupdict()
    config = {
        "requested": parsed_config["requested"],
        "effective": parsed_config["effective"],
        **{
            name: int(parsed_config[name])
            for name in ("log_t", "log_k", "chunk_bits", "num_ra", "degree")
        },
    }
    expected = {
        "requested": "q10",
        "log_t": log_n,
        "log_k": 13,
    }
    if log_n >= 25:
        expected.update(
            {
                "effective": "q10",
                "chunk_bits": 8,
                "num_ra": 2,
                "degree": 4,
            }
        )
    for name, value in expected.items():
        if config[name] != value:
            raise ValueError(
                f"Bytecode cycle config {name}={config[name]}, expected {value}"
            )

    proof_record = f"PROOF_VERIFIED backend={backend} value=true"
    if stdout.splitlines().count(proof_record) != 1:
        raise ValueError(f"evaluator must emit exactly one `{proof_record}` record")

    metal_configs = [
        match
        for line in stdout.splitlines()
        if (match := BYTECODE_METAL_CONFIG_RE.fullmatch(line)) is not None
    ]
    if backend == "metal":
        if len(metal_configs) != 1:
            raise ValueError("Metal evaluator must emit exactly one Bytecode Metal config")
        raw_metal_config = metal_configs[0].groupdict()
        metal_config: Optional[dict[str, Any]] = {
            "cpu_tail": raw_metal_config["cpu_tail"],
            "trace_cutoff": int(raw_metal_config["trace_cutoff"]),
            "cutoff": int(raw_metal_config["cutoff"]),
            "message_threads": int(raw_metal_config["message_threads"]),
            "transition_threads": int(raw_metal_config["transition_threads"]),
            "max_threadgroups": int(raw_metal_config["max_threadgroups"]),
        }
        if metal_config != {
            "cpu_tail": "q10",
            "trace_cutoff": 1 << trace_cutoff_log2,
            "cutoff": 1 << cutoff_log2,
            "message_threads": message_threads,
            "transition_threads": transition_threads,
            "max_threadgroups": max_threadgroups,
        }:
            raise ValueError(f"unexpected Bytecode Metal config: {metal_config}")
    elif metal_configs:
        raise ValueError("optimized evaluator emitted a Bytecode Metal config")
    else:
        metal_config = None
    return {"relation": config, "metal_runtime": metal_config}


def validate_instruction_input_stdout(
    stdout: str,
    backend: str,
    native_message_threads: int = 256,
    native_transition_threads: int = 128,
    dense_transition_threads: int = 128,
    cutoff_log2: int = 16,
    trace_cutoff_log2: int = 25,
    borrow_outer_residual: bool = False,
) -> Optional[dict[str, Any]]:
    configs = [
        match
        for line in stdout.splitlines()
        if (match := INSTRUCTION_INPUT_METAL_CONFIG_RE.fullmatch(line)) is not None
    ]
    if backend == "optimized":
        if configs:
            raise ValueError("optimized evaluator emitted an InstructionInput Metal config")
        return None
    if len(configs) != 1:
        raise ValueError(
            "Metal evaluator must emit exactly one InstructionInput Metal config"
        )
    raw_config = configs[0].groupdict()
    config = {
        "trace_cutoff": int(raw_config["trace_cutoff"]),
        "cutoff": int(raw_config["cutoff"]),
        "native_message_threads": int(raw_config["native_message_threads"]),
        "native_transition_threads": int(raw_config["native_transition_threads"]),
        "dense_transition_threads": int(raw_config["dense_transition_threads"]),
        "storage_initialization": raw_config["storage_initialization"],
        "dense_storage_mode": raw_config["dense_storage_mode"],
        "native_primer": raw_config["native_primer"],
    }
    expected = {
        "trace_cutoff": 1 << trace_cutoff_log2,
        "cutoff": 1 << cutoff_log2,
        "native_message_threads": native_message_threads,
        "native_transition_threads": native_transition_threads,
        "dense_transition_threads": dense_transition_threads,
        "storage_initialization": "minimal",
        "dense_storage_mode": "OuterResidual" if borrow_outer_residual else "Owned",
        "native_primer": "async",
    }
    if config != expected:
        raise ValueError(f"unexpected InstructionInput Metal config: {config}")
    return config


def validate_booleanity_address_stdout(
    stdout: str,
    backend: str,
    inner_log2: int = 15,
    selectors_per_tile: int = 6,
    tile_threads: int = 512,
    finalize_threads: int = 1024,
    trace_cutoff_log2: int = 18,
    implementation: str = "accepted",
) -> Optional[dict[str, Any]]:
    configs = [
        match
        for line in stdout.splitlines()
        if (match := BOOLEANITY_ADDRESS_METAL_CONFIG_RE.fullmatch(line)) is not None
    ]
    implementations = [
        match
        for line in stdout.splitlines()
        if (match := BOOLEANITY_ADDRESS_METAL_IMPLEMENTATION_RE.fullmatch(line))
        is not None
    ]
    if backend == "optimized":
        if configs or implementations:
            raise ValueError(
                "optimized evaluator emitted a Booleanity address Metal config"
            )
        return None
    if len(configs) != 1 or len(implementations) != 1:
        raise ValueError(
            "Metal evaluator must emit exactly one Booleanity address Metal config"
        )
    config = {name: int(value) for name, value in configs[0].groupdict().items()}
    config["implementation"] = implementations[0].group("implementation")
    expected = {
        "trace_cutoff": 1 << trace_cutoff_log2,
        "inner_log2": inner_log2,
        "selectors_per_tile": selectors_per_tile,
        "tile_threads": tile_threads,
        "finalize_threads": finalize_threads,
        "implementation": implementation,
    }
    if config != expected:
        raise ValueError(f"unexpected Booleanity address Metal config: {config}")
    return config


def validate_hamming_weight_stdout(
    stdout: str,
    backend: str,
    inner_log2: int = 15,
    selectors_per_tile: int = 6,
    tile_threads: int = 512,
    finalize_threads: int = 1024,
    trace_cutoff_log2: int = 18,
    implementation: str = "accepted-rows",
) -> Optional[dict[str, Any]]:
    configs = [
        match
        for line in stdout.splitlines()
        if (match := HAMMING_WEIGHT_METAL_CONFIG_RE.fullmatch(line)) is not None
    ]
    implementations = [
        match
        for line in stdout.splitlines()
        if (match := HAMMING_WEIGHT_METAL_IMPLEMENTATION_RE.fullmatch(line)) is not None
    ]
    if backend == "optimized":
        if configs or implementations:
            raise ValueError("optimized evaluator emitted a Hamming-weight Metal config")
        return None
    if len(configs) != 1 or len(implementations) != 1:
        raise ValueError(
            "Metal evaluator must emit exactly one Hamming-weight Metal config"
        )
    config = {name: int(value) for name, value in configs[0].groupdict().items()}
    config["implementation"] = implementations[0].group("implementation")
    expected = {
        "trace_cutoff": 1 << trace_cutoff_log2,
        "inner_log2": inner_log2,
        "selectors_per_tile": selectors_per_tile,
        "tile_threads": tile_threads,
        "finalize_threads": finalize_threads,
        "implementation": implementation,
    }
    if config != expected:
        raise ValueError(f"unexpected Hamming-weight Metal config: {config}")
    return config


def validate_outer_remainder_stdout(
    stdout: str,
    backend: str,
    materialize_threads: int = 256,
    transition_threads: int = 128,
    output_threads: int = 256,
    cutoff_log2: int = 16,
    trace_cutoff_log2: int = 18,
    binding_plan: str = "b_only_v1",
    product_uniskip_carrier: bool = False,
) -> Optional[dict[str, Any]]:
    configs = [
        match
        for line in stdout.splitlines()
        if (match := OUTER_REMAINDER_METAL_CONFIG_RE.fullmatch(line)) is not None
    ]
    if backend == "optimized":
        if configs:
            raise ValueError("optimized evaluator emitted an OuterRemainder Metal config")
        return None
    if len(configs) != 1:
        raise ValueError("Metal evaluator must emit exactly one OuterRemainder config")
    raw = configs[0].groupdict()
    config = {
        **{
            name: int(raw[name])
            for name in (
                "trace_cutoff",
                "cutoff",
                "materialize_threads",
                "transition_threads",
                "output_threads",
                "max_threadgroups",
            )
        },
        "storage_initialization": raw["storage_initialization"],
        "binding_plan": raw["binding_plan"],
        "product_uniskip_carrier": raw["product_uniskip_carrier"] == "true",
    }
    expected = {
        "trace_cutoff": 1 << trace_cutoff_log2,
        "cutoff": 1 << cutoff_log2,
        "materialize_threads": materialize_threads,
        "transition_threads": transition_threads,
        "output_threads": output_threads,
        "max_threadgroups": 8192,
        "storage_initialization": "full",
        "binding_plan": binding_plan,
        "product_uniskip_carrier": product_uniskip_carrier,
    }
    if config != expected:
        raise ValueError(f"unexpected OuterRemainder Metal config: {config}")
    return config


def validate_piop_execution_stdout(stdout: str) -> dict[str, int]:
    configs = [
        match
        for line in stdout.splitlines()
        if (match := PIOP_EXECUTION_CONFIG_RE.fullmatch(line)) is not None
    ]
    if len(configs) != 1:
        raise ValueError("evaluator must emit exactly one PIOP execution config")
    rayon_threads = int(configs[0].group("rayon_threads"))
    if rayon_threads != PRODUCTION_RAYON_THREADS:
        raise ValueError(
            f"unexpected production Rayon width: {rayon_threads}"
        )
    return {"rayon_threads": rayon_threads}


def interval_duration_us(interval: tuple[float, float]) -> float:
    return interval[1] - interval[0]


def require_contained(
    child: tuple[float, float], parent: tuple[float, float], description: str
) -> None:
    if child[0] < parent[0] or child[1] > parent[1]:
        raise ValueError(f"{description} is not contained in its outer member span")


def unique_span_args(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    records = [
        event
        for event in events
        if event.get("name") == name and event.get("ph") in {"B", "X"}
    ]
    if len(records) != 1 or not isinstance(records[0].get("args"), dict):
        raise ValueError(f"trace must contain one argument record for {name}")
    record = records[0]
    args = dict(record["args"])
    if record.get("ph") == "B":
        end_records = [
            event
            for event in events
            if event.get("name") == name
            and event.get("ph") == "E"
            and event.get("pid") == record.get("pid")
            and event.get("tid") == record.get("tid")
        ]
        if len(end_records) != 1 or not isinstance(end_records[0].get("args"), dict):
            raise ValueError(f"trace must contain one ending argument record for {name}")
        for field, value in end_records[0]["args"].items():
            if field in args and args[field] != value:
                raise ValueError(f"trace span {name} changed argument {field}")
            args[field] = value
    return args


def bytecode_member_breakdown(
    events: list[dict[str, Any]], backend: str, log_n: int, cutoff_log2: int = 16
) -> dict[str, Any]:
    outer_names = {f"{BYTECODE_KERNEL}::{component}" for component in BYTECODE_COMPONENTS}
    inner_names = {
        f"Metal{BYTECODE_KERNEL}::{phase}" for phase in BYTECODE_METAL_PHASES
    }
    intervals = strict_named_intervals(events, outer_names | inner_names | {PIOP_SPAN})
    if len(intervals[PIOP_SPAN]) != 1:
        raise ValueError("trace must contain exactly one positive PIOP span")
    piop = intervals[PIOP_SPAN][0]
    by_component = {
        component: sorted(intervals[f"{BYTECODE_KERNEL}::{component}"])
        for component in BYTECODE_COMPONENTS
    }
    expected_outer_counts = {
        "prepare": 1,
        "prove_round": log_n,
        "finish_rounds": 1,
        "output_claims": 1,
    }
    outer_counts = {
        component: len(component_intervals)
        for component, component_intervals in by_component.items()
    }
    if outer_counts != expected_outer_counts:
        raise ValueError(
            f"Bytecode member span counts {outer_counts}, expected {expected_outer_counts}"
        )

    prepare = by_component["prepare"][0]
    rounds = by_component["prove_round"]
    finish = by_component["finish_rounds"][0]
    output = by_component["output_claims"][0]
    ordered = [prepare, *rounds, finish, output]
    if any(start < piop[0] or end > piop[1] for start, end in ordered):
        raise ValueError("a Bytecode member span lies outside PIOP")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Bytecode member spans overlap or appear out of order")

    inner = {
        phase: sorted(intervals[f"Metal{BYTECODE_KERNEL}::{phase}"])
        for phase in BYTECODE_METAL_PHASES
    }
    inner_counts = {phase: len(values) for phase, values in inner.items()}
    if backend == "optimized":
        if any(inner_counts.values()):
            raise ValueError("optimized trace unexpectedly contains Bytecode Metal spans")
    else:
        expected_inner_counts = {
            "prepare": 1,
            "allocation_plan": 1,
            "first_message": 1,
            "first_bind": 1,
            "dense_round": log_n - cutoff_log2 - 1,
            "readback": 1,
            "cpu_tail": cutoff_log2,
            "invalid_round": 0,
        }
        if inner_counts != expected_inner_counts:
            raise ValueError(
                f"Bytecode Metal span counts {inner_counts}, expected {expected_inner_counts}"
            )
        dense_count = expected_inner_counts["dense_round"]
        handoff_round = 2 + dense_count
        require_contained(inner["prepare"][0], prepare, "Metal Bytecode prepare")
        require_contained(
            inner["allocation_plan"][0], prepare, "Metal Bytecode allocation plan"
        )
        require_contained(inner["first_message"][0], rounds[0], "first Metal message")
        require_contained(inner["first_bind"][0], rounds[1], "first Metal bind")
        for index, interval in enumerate(inner["dense_round"]):
            require_contained(interval, rounds[index + 2], "dense Metal round")
        require_contained(inner["readback"][0], rounds[handoff_round], "Metal readback")
        for interval, outer_round in zip(inner["cpu_tail"][:-1], rounds[handoff_round:]):
            require_contained(interval, outer_round, "Bytecode CPU-tail round")
        require_contained(inner["cpu_tail"][-1], finish, "Bytecode CPU-tail finish")

        allocation = unique_span_args(
            events, f"Metal{BYTECODE_KERNEL}::allocation_plan"
        )
        expected_allocation_fields = {
            "device_buffers",
            "planned_device_bytes",
            "current_device_bytes",
            "recommended_device_bytes",
        }
        if set(allocation) != expected_allocation_fields:
            raise ValueError("Bytecode allocation plan has unexpected fields")
        allocation = {name: int(value) for name, value in allocation.items()}
        if allocation["device_buffers"] != 17 or allocation["planned_device_bytes"] <= 0:
            raise ValueError("Bytecode allocation plan has invalid buffer accounting")
        if (
            allocation["current_device_bytes"] + allocation["planned_device_bytes"]
            > allocation["recommended_device_bytes"]
        ):
            raise ValueError("Bytecode allocation plan exceeds the admitted working set")
        readback = unique_span_args(events, f"Metal{BYTECODE_KERNEL}::readback")
        if readback != {"bytes": str(5 * (1 << cutoff_log2) * 16)}:
            raise ValueError("Bytecode readback does not cover exactly five cutoff tables")

    round_durations = [interval_duration_us(interval) for interval in rounds]
    components = {
        "prepare_us": interval_duration_us(prepare),
        "rounds_us": round_durations,
        "rounds_total_us": sum(round_durations),
        "finish_us": interval_duration_us(finish),
        "output_claims_us": interval_duration_us(output),
    }
    components["member_us"] = (
        components["prepare_us"]
        + components["rounds_total_us"]
        + components["finish_us"]
        + components["output_claims_us"]
    )
    scalar_components = [value for value in components.values() if not isinstance(value, list)]
    if any(not math.isfinite(value) or value <= 0.0 for value in scalar_components):
        raise ValueError("trace contains a non-positive Bytecode member duration")
    return {
        "components": components,
        "outer_counts": outer_counts,
        "metal_counts": inner_counts,
        "resource_observation": (
            {"allocation": allocation, "readback_bytes": int(readback["bytes"])}
            if backend == "metal"
            else None
        ),
    }


def trace_boolean(value: Any) -> Optional[bool]:
    if value is True or value == "true":
        return True
    if value is False or value == "false":
        return False
    return None


def trace_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"InstructionInput trace has invalid {field}")
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"InstructionInput trace has invalid {field}") from error
        if not isinstance(decoded, str):
            raise ValueError(f"InstructionInput trace has invalid {field}")
        return decoded
    return value


def positive_trace_integer(value: Any, field: str) -> int:
    if type(value) is int:
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdecimal():
        parsed = int(value)
    else:
        raise ValueError(f"InstructionInput trace has invalid {field}")
    if parsed <= 0:
        raise ValueError(f"InstructionInput trace has invalid {field}")
    return parsed


def nonnegative_trace_integer(value: Any, field: str) -> int:
    if type(value) is int:
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdecimal():
        parsed = int(value)
    else:
        raise ValueError(f"InstructionInput trace has invalid {field}")
    if parsed < 0:
        raise ValueError(f"InstructionInput trace has invalid {field}")
    return parsed


def booleanity_trace_integer(
    value: Any, field: str, *, allow_zero: bool = False
) -> int:
    if type(value) is int:
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdecimal():
        parsed = int(value)
    else:
        raise ValueError(f"Booleanity address trace has invalid {field}")
    if parsed < 0 or (parsed == 0 and not allow_zero):
        raise ValueError(f"Booleanity address trace has invalid {field}")
    return parsed


def exact_span_args(
    events: list[dict[str, Any]], name: str, fields: set[str]
) -> dict[str, Any]:
    args = unique_span_args(events, name)
    if set(args) != fields:
        raise ValueError(f"{name} has unexpected argument fields")
    return args


def required_span_args(
    events: list[dict[str, Any]], name: str, fields: set[str]
) -> dict[str, Any]:
    args = unique_span_args(events, name)
    missing = fields - args.keys()
    if missing:
        raise ValueError(f"{name} is missing argument fields: {sorted(missing)}")
    return {field: args[field] for field in fields}


def instruction_claim_trace_integer(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if type(value) is int:
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdecimal():
        parsed = int(value)
    else:
        raise ValueError(f"Instruction Claim trace has invalid {field}")
    if parsed < 0 or (parsed == 0 and not allow_zero):
        raise ValueError(f"Instruction Claim trace has invalid {field}")
    return parsed


def instruction_claim_observation(
    events: list[dict[str, Any]], backend: str, log_n: int
) -> dict[str, Any]:
    if backend not in {"optimized", "metal"}:
        raise ValueError(f"unsupported Instruction Claim backend {backend!r}")
    if log_n < 1:
        raise ValueError("Instruction Claim requires a non-empty trace")

    outer_counts = {
        component: positive_span_count(
            events, f"{INSTRUCTION_CLAIM_KERNEL}::{component}"
        )
        for component in ("prepare", "prove_round", "finish_rounds", "output_claims")
    }
    if outer_counts != {
        "prepare": 1,
        "prove_round": log_n,
        "finish_rounds": 1,
        "output_claims": 1,
    }:
        raise ValueError("Instruction Claim member span counts do not match the trace")

    phases = (
        "first_message_submit",
        "prepare",
        "first_message_join",
        "bind_and_message",
        "output_claims",
    )
    expected_names = {
        f"Metal{INSTRUCTION_CLAIM_KERNEL}::{phase}" for phase in phases
    }
    observed_names = {
        name
        for name, _, _ in span_intervals_us(events)
        if name.startswith(f"Metal{INSTRUCTION_CLAIM_KERNEL}::")
    }
    unknown_names = observed_names - expected_names
    if unknown_names:
        raise ValueError(
            f"Instruction Claim trace contains unknown Metal phases: {sorted(unknown_names)}"
        )
    metal_counts = {
        phase: positive_span_count(
            events, f"Metal{INSTRUCTION_CLAIM_KERNEL}::{phase}"
        )
        for phase in phases
    }
    if backend == "optimized":
        if any(metal_counts.values()):
            raise ValueError("optimized Instruction Claim trace exercised Metal")
        return {
            "outer_counts": outer_counts,
            "metal_counts": metal_counts,
            "resource_observation": None,
        }
    if metal_counts != {
        "first_message_submit": 1,
        "prepare": 1,
        "first_message_join": 1,
        "bind_and_message": log_n - 1,
        "output_claims": 1,
    }:
        raise ValueError("Instruction Claim Metal phase counts are incomplete")

    prefix = f"Metal{INSTRUCTION_CLAIM_KERNEL}"
    submit = exact_span_args(
        events,
        f"{prefix}::first_message_submit",
        {
            "command_committed",
            "lookup_rows_storage_id",
            "resident_rows_storage_id",
            "submit_wall_ns",
        },
    )
    prepare = exact_span_args(
        events,
        f"{prefix}::prepare",
        {
            "cycles",
            "lookup_rows_storage_id",
            "resident_rows_storage_id",
            "round_device_buffer_allocations",
            "rounds",
            "row_upload_bytes",
            "workspace_bytes",
        },
    )
    join = exact_span_args(
        events,
        f"{prefix}::first_message_join",
        {
            "command_completed",
            "completed_before_join",
            "gpu_active_ns",
            "join_wall_ns",
            "lifecycle_wall_ns",
            "overlap_wall_ns",
            "resident_rows_storage_id",
            "submit_wall_ns",
        },
    )
    output = exact_span_args(
        events,
        f"{prefix}::output_claims",
        {
            "dispatch_wall_ns",
            "gpu_active_ns",
            "resident_rows_storage_id",
            "row_upload_bytes",
        },
    )
    producer = required_span_args(
        events,
        f"Metal{PRODUCT_REMAINDER_KERNEL}::prepare",
        {"resident_rows_storage_id", "row_upload_bytes"},
    )

    resident_id = instruction_claim_trace_integer(
        submit["resident_rows_storage_id"], "resident row storage ID"
    )
    lookup_id = instruction_claim_trace_integer(
        submit["lookup_rows_storage_id"], "lookup row storage ID"
    )
    submit_wall_ns = instruction_claim_trace_integer(
        submit["submit_wall_ns"], "submit wall time"
    )
    overlap_wall_ns = instruction_claim_trace_integer(
        join["overlap_wall_ns"], "overlap wall time", allow_zero=True
    )
    join_wall_ns = instruction_claim_trace_integer(
        join["join_wall_ns"], "join wall time"
    )
    lifecycle_wall_ns = instruction_claim_trace_integer(
        join["lifecycle_wall_ns"], "lifecycle wall time"
    )
    initial_gpu_active_ns = instruction_claim_trace_integer(
        join["gpu_active_ns"], "initial-message GPU time"
    )
    if (
        trace_boolean(submit["command_committed"]) is not True
        or trace_boolean(join["command_completed"]) is not True
        or trace_boolean(join["completed_before_join"]) is None
        or instruction_claim_trace_integer(prepare["cycles"], "cycle count")
        != 1 << log_n
        or instruction_claim_trace_integer(prepare["rounds"], "round count")
        != log_n
        or instruction_claim_trace_integer(
            prepare["round_device_buffer_allocations"],
            "round device-buffer allocation count",
            allow_zero=True,
        )
        != 0
        or instruction_claim_trace_integer(
            prepare["row_upload_bytes"], "prepare row uploads", allow_zero=True
        )
        != 0
        or instruction_claim_trace_integer(
            output["row_upload_bytes"], "output row uploads", allow_zero=True
        )
        != 0
        or instruction_claim_trace_integer(
            prepare["workspace_bytes"], "workspace size"
        )
        <= 0
        or lookup_id == resident_id
        or instruction_claim_trace_integer(
            producer["resident_rows_storage_id"], "producer resident storage ID"
        )
        != resident_id
        or instruction_claim_trace_integer(
            producer["row_upload_bytes"], "producer row uploads", allow_zero=True
        )
        != 0
        or any(
            instruction_claim_trace_integer(record[field], field) != expected
            for record, field, expected in (
                (prepare, "resident_rows_storage_id", resident_id),
                (prepare, "lookup_rows_storage_id", lookup_id),
                (join, "resident_rows_storage_id", resident_id),
                (output, "resident_rows_storage_id", resident_id),
                (join, "submit_wall_ns", submit_wall_ns),
            )
        )
        or initial_gpu_active_ns > lifecycle_wall_ns
        or abs(
            lifecycle_wall_ns
            - (submit_wall_ns + overlap_wall_ns + join_wall_ns)
        )
        > 100_000
    ):
        raise ValueError("Instruction Claim Metal lifecycle is inconsistent")

    bind_records = [
        event.get("args")
        for event in events
        if event.get("name") == f"{prefix}::bind_and_message"
        and event.get("ph") in {"E", "X"}
    ]
    if len(bind_records) != log_n - 1 or any(
        not isinstance(record, dict) for record in bind_records
    ):
        raise ValueError("Instruction Claim bind records are incomplete")
    bind_gpu_active_ns = 0
    bind_dispatch_wall_ns = 0
    observed_rounds = set()
    for record in bind_records:
        assert isinstance(record, dict)
        if set(record) != {
            "dispatch_wall_ns",
            "gpu_active_ns",
            "resident_rows_storage_id",
            "round",
            "source_elements",
        }:
            raise ValueError("Instruction Claim bind record fields are incomplete")
        round_index = instruction_claim_trace_integer(record["round"], "bind round")
        source_elements = instruction_claim_trace_integer(
            record["source_elements"], "bind source size"
        )
        dispatch_wall_ns = instruction_claim_trace_integer(
            record["dispatch_wall_ns"], "bind dispatch wall time"
        )
        gpu_active_ns = instruction_claim_trace_integer(
            record["gpu_active_ns"], "bind GPU time"
        )
        if (
            round_index >= log_n
            or source_elements != 1 << (log_n - round_index + 1)
            or instruction_claim_trace_integer(
                record["resident_rows_storage_id"], "bind resident storage ID"
            )
            != resident_id
            or gpu_active_ns > dispatch_wall_ns
        ):
            raise ValueError("Instruction Claim bind schedule is inconsistent")
        observed_rounds.add(round_index)
        bind_dispatch_wall_ns += dispatch_wall_ns
        bind_gpu_active_ns += gpu_active_ns
    if observed_rounds != set(range(1, log_n)):
        raise ValueError("Instruction Claim bind rounds are not exact")

    output_dispatch_wall_ns = instruction_claim_trace_integer(
        output["dispatch_wall_ns"], "output dispatch wall time"
    )
    output_gpu_active_ns = instruction_claim_trace_integer(
        output["gpu_active_ns"], "output GPU time"
    )
    if output_gpu_active_ns > output_dispatch_wall_ns:
        raise ValueError("Instruction Claim output GPU time exceeds dispatch time")
    return {
        "outer_counts": outer_counts,
        "metal_counts": metal_counts,
        "resource_observation": {
            "resident_rows_storage_id": resident_id,
            "lookup_rows_storage_id": lookup_id,
            "producer_rows_storage_id": resident_id,
            "command_committed": True,
            "command_completed": True,
            "completed_before_join": trace_boolean(join["completed_before_join"]),
            "submit_wall_ns": submit_wall_ns,
            "overlap_wall_ns": overlap_wall_ns,
            "join_wall_ns": join_wall_ns,
            "lifecycle_wall_ns": lifecycle_wall_ns,
            "initial_gpu_active_ns": initial_gpu_active_ns,
            "bind_dispatches": len(bind_records),
            "bind_dispatch_wall_ns": bind_dispatch_wall_ns,
            "bind_gpu_active_ns": bind_gpu_active_ns,
            "output_dispatch_wall_ns": output_dispatch_wall_ns,
            "output_gpu_active_ns": output_gpu_active_ns,
            "row_upload_bytes": 0,
            "round_device_buffer_allocations": 0,
        },
    }


def outer_remainder_storage_geometry(
    log_n: int, product_uniskip_carrier: bool = False
) -> dict[str, Any]:
    rows = 1 << log_n
    weight_capacity = 1 << ((log_n + 1) // 2)
    threadgroups = min(8192, weight_capacity)
    opening_outputs = 37 if product_uniskip_carrier else 35
    elements = [
        2 * rows,
        2 * rows,
        weight_capacity,
        weight_capacity,
        OUTER_REMAINDER_A_LOOKUP_FIELDS,
        2 * threadgroups,
        2,
        opening_outputs * threadgroups,
        opening_outputs,
    ]
    return {
        "element_counts": elements,
        "owned_bytes": 16 * sum(elements),
        "maximum_buffer_bytes": 16 * max(elements),
    }


def outer_remainder_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    cutoff_log2: int = 16,
    trace_cutoff_log2: int = 18,
    product_uniskip_carrier: bool = False,
) -> dict[str, Any]:
    if backend not in {"optimized", "metal"}:
        raise ValueError(f"unsupported OuterRemainder backend {backend!r}")
    if not 1 <= cutoff_log2 < log_n or not 1 <= trace_cutoff_log2 <= log_n:
        raise ValueError("invalid OuterRemainder evaluator geometry")
    metal_names = {
        f"MetalOuterRemainder::{phase}" for phase in OUTER_REMAINDER_METAL_PHASES
    }
    unknown = {
        name
        for event in events
        if isinstance((name := event.get("name")), str)
        and name.startswith("MetalOuterRemainder::")
        and name not in metal_names
    }
    if unknown:
        raise ValueError(
            f"OuterRemainder trace contains unknown Metal phases: {sorted(unknown)}"
        )
    intervals = strict_named_intervals(
        events,
        metal_names
        | {
            PIOP_SPAN,
            OUTER_REMAINDER_COMPLETE_MEMBER,
            SUMCHECK_ROUND_SPAN,
            SUMCHECK_HOST_FIAT_SHAMIR_SPAN,
        },
    )
    if len(intervals[PIOP_SPAN]) != 1 or len(
        intervals[OUTER_REMAINDER_COMPLETE_MEMBER]
    ) != 1:
        raise ValueError("trace must contain one PIOP and OuterRemainder member span")
    piop = intervals[PIOP_SPAN][0]
    member = intervals[OUTER_REMAINDER_COMPLETE_MEMBER][0]
    require_contained(member, piop, "OuterRemainder complete member")
    rounds = [
        interval
        for interval in intervals[SUMCHECK_ROUND_SPAN]
        if member[0] <= interval[0] and interval[1] <= member[1]
    ]
    host_fiat_shamir = [
        interval
        for interval in intervals[SUMCHECK_HOST_FIAT_SHAMIR_SPAN]
        if member[0] <= interval[0] and interval[1] <= member[1]
    ]
    if len(rounds) != log_n + 1 or len(host_fiat_shamir) != log_n + 1:
        raise ValueError("OuterRemainder round or host Fiat-Shamir topology is incomplete")
    if any(
        len(
            [
                fiat_shamir
                for fiat_shamir in host_fiat_shamir
                if round_interval[0] <= fiat_shamir[0]
                and fiat_shamir[1] <= round_interval[1]
            ]
        )
        != 1
        for round_interval in rounds
    ):
        raise ValueError("OuterRemainder rounds do not each own one Fiat-Shamir span")

    metal_counts = {
        phase: len(intervals[f"MetalOuterRemainder::{phase}"])
        for phase in OUTER_REMAINDER_METAL_PHASES
    }
    if backend == "optimized":
        if any(metal_counts.values()):
            raise ValueError("optimized trace unexpectedly contains OuterRemainder Metal spans")
        resource_observation = None
        row_lifecycle = None
        product_uniskip_carrier_observation = None
    else:
        expected_counts = {
            "storage_prepare": 1,
            "storage_initialize": 1,
            "storage_initialize_complete": 1,
            "prepare": 1,
            "allocation_plan": 1,
            "row_handoff": 1,
            "sequence_prepare": 1,
            "first_message": 1,
            "first_bind": 1,
            "dense_round": log_n - cutoff_log2,
            "readback": 1,
            "cpu_tail": cutoff_log2,
            "output_claims": 1,
            "row_release": 1,
            "product_uniskip_carrier_park": int(product_uniskip_carrier),
        }
        if metal_counts != expected_counts:
            raise ValueError(
                f"OuterRemainder Metal span counts {metal_counts}, expected {expected_counts}"
            )
        setup_phases = {
            "storage_prepare",
            "storage_initialize",
            "storage_initialize_complete",
        }
        for phase in OUTER_REMAINDER_METAL_PHASES:
            if phase not in setup_phases:
                for interval in intervals[f"MetalOuterRemainder::{phase}"]:
                    require_contained(interval, member, f"OuterRemainder {phase}")

        rows = 1 << log_n
        geometry = outer_remainder_storage_geometry(log_n, product_uniskip_carrier)
        storage_fields = {
            "cycles",
            "planned_device_bytes",
            "maximum_buffer_bytes",
            "current_device_bytes",
            "recommended_max_working_set_bytes",
            "initialization_mode",
            "admitted",
            "initialized",
            "fallback_reason",
            "device_buffers",
            "initialization_bytes",
            "initialization_wall_ns",
            "initialization_gpu_active_ns",
            *{f"buffer_{index}" for index in range(9)},
        }
        storage_raw = required_span_args(
            events, "MetalOuterRemainder::storage_prepare", storage_fields
        )
        storage = {
            field: nonnegative_trace_integer(storage_raw[field], field)
            for field in storage_fields
            if field
            not in {"initialization_mode", "admitted", "initialized", "fallback_reason"}
        }
        storage.update(
            {
                "initialization_mode": trace_string(
                    storage_raw["initialization_mode"], "initialization_mode"
                ),
                "admitted": trace_boolean(storage_raw["admitted"]),
                "initialized": trace_boolean(storage_raw["initialized"]),
                "fallback_reason": trace_string(
                    storage_raw["fallback_reason"], "fallback_reason"
                ),
            }
        )
        buffer_ids = [storage[f"buffer_{index}"] for index in range(9)]
        initialization_fields = {
            "mode",
            "device_buffers",
            "bytes",
            "protocol_dispatches",
            *{f"buffer_{index}" for index in range(9)},
        }
        initialization_raw = exact_span_args(
            events,
            "MetalOuterRemainder::storage_initialize",
            initialization_fields,
        )
        initialization = {
            field: nonnegative_trace_integer(initialization_raw[field], field)
            for field in initialization_fields
            if field != "mode"
        }
        initialization["mode"] = trace_string(initialization_raw["mode"], "mode")
        completion_raw = exact_span_args(
            events,
            "MetalOuterRemainder::storage_initialize_complete",
            {"mode", "command_completed", "bytes", "wall_ns", "gpu_active_ns"},
        )
        completion = {
            "mode": trace_string(completion_raw["mode"], "mode"),
            "command_completed": trace_boolean(completion_raw["command_completed"]),
            **{
                field: nonnegative_trace_integer(completion_raw[field], field)
                for field in ("bytes", "wall_ns", "gpu_active_ns")
            },
        }
        initialization_ids = [
            initialization[f"buffer_{index}"] for index in range(9)
        ]
        if (
            storage["cycles"] != rows
            or storage["planned_device_bytes"] != geometry["owned_bytes"]
            or storage["maximum_buffer_bytes"] != geometry["maximum_buffer_bytes"]
            or storage["admitted"] is not True
            or storage["initialized"] is not True
            or storage["fallback_reason"] != "none"
            or storage["initialization_mode"] != "full"
            or storage["device_buffers"] != 9
            or storage["initialization_bytes"] != geometry["owned_bytes"]
            or storage["current_device_bytes"] + storage["planned_device_bytes"]
            > storage["recommended_max_working_set_bytes"]
            or any(identity <= 0 for identity in buffer_ids)
            or len(set(buffer_ids)) != 9
            or initialization["mode"] != "full"
            or initialization["device_buffers"] != 9
            or initialization["bytes"] != geometry["owned_bytes"]
            or initialization["protocol_dispatches"] != 0
            or initialization_ids != buffer_ids
            or completion["mode"] != "full"
            or completion["command_completed"] is not True
            or completion["bytes"] != geometry["owned_bytes"]
            or completion["wall_ns"] <= 0
            or completion["gpu_active_ns"] <= 0
            or completion["gpu_active_ns"] > completion["wall_ns"]
            or storage["initialization_wall_ns"] != completion["wall_ns"]
            or storage["initialization_gpu_active_ns"] != completion["gpu_active_ns"]
        ):
            raise ValueError("OuterRemainder storage preparation is inconsistent")
        storage["initialization"] = {
            "mode": initialization["mode"],
            "device_buffers": initialization["device_buffers"],
            "bytes": initialization["bytes"],
            "protocol_dispatches": initialization["protocol_dispatches"],
            "buffer_identities": initialization_ids,
            "command_completed": completion["command_completed"],
            "wall_ns": completion["wall_ns"],
            "gpu_active_ns": completion["gpu_active_ns"],
        }

        handoff_fields = {
            "compact_rows_storage_id",
            "residual_rows_storage_id",
            "device_registry_id",
            "resident_rows",
            "row_upload_bytes",
            "device_allocations",
        }
        handoff = {
            field: nonnegative_trace_integer(value, field)
            for field, value in required_span_args(
                events, "MetalOuterRemainder::row_handoff", handoff_fields
            ).items()
        }
        sequence_fields = {
            "resident_rows",
            "rounds",
            "cutoff_elements",
            "trace_cutoff_elements",
            "planned_device_bytes",
            "compact_rows_storage_id",
            "residual_rows_storage_id",
            "device_registry_id",
            "storage_reused",
            "storage_initialization_mode",
            "preinitialized_device_bytes",
            "initialization_bytes",
            "attached_owned_bytes",
            "row_upload_bytes",
            "full_domain_copy_dispatches",
            "sequence_device_buffer_allocations",
            "round_device_buffer_allocations",
            *{f"storage_buffer_{index}" for index in range(9)},
        }
        sequence_raw = required_span_args(
            events, "MetalOuterRemainder::sequence_prepare", sequence_fields
        )
        sequence = {
            field: nonnegative_trace_integer(sequence_raw[field], field)
            for field in sequence_fields
            if field not in {"storage_reused", "storage_initialization_mode"}
        }
        sequence["storage_reused"] = trace_boolean(sequence_raw["storage_reused"])
        sequence["storage_initialization_mode"] = trace_string(
            sequence_raw["storage_initialization_mode"],
            "storage_initialization_mode",
        )
        sequence_buffers = [
            sequence[f"storage_buffer_{index}"] for index in range(9)
        ]
        if (
            handoff["resident_rows"] != rows
            or handoff["row_upload_bytes"] != 0
            or handoff["device_allocations"] != 0
            or min(
                handoff["compact_rows_storage_id"],
                handoff["residual_rows_storage_id"],
                handoff["device_registry_id"],
            )
            <= 0
            or sequence["resident_rows"] != rows
            or sequence["rounds"] != log_n + 1
            or sequence["cutoff_elements"] != 1 << cutoff_log2
            or sequence["trace_cutoff_elements"] != 1 << trace_cutoff_log2
            or sequence["planned_device_bytes"] != geometry["owned_bytes"]
            or sequence["storage_reused"] is not True
            or sequence["storage_initialization_mode"] != "full"
            or sequence["preinitialized_device_bytes"] != geometry["owned_bytes"]
            or sequence["initialization_bytes"] != geometry["owned_bytes"]
            or sequence["attached_owned_bytes"] != geometry["owned_bytes"]
            or any(
                sequence[field] != 0
                for field in (
                    "row_upload_bytes",
                    "full_domain_copy_dispatches",
                    "sequence_device_buffer_allocations",
                    "round_device_buffer_allocations",
                )
            )
            or sequence_buffers != buffer_ids
            or any(
                sequence[field] != handoff[field]
                for field in (
                    "compact_rows_storage_id",
                    "residual_rows_storage_id",
                    "device_registry_id",
                )
            )
        ):
            raise ValueError("OuterRemainder resident sequence is inconsistent")

        readback = {
            field: nonnegative_trace_integer(value, field)
            for field, value in required_span_args(
                events,
                "MetalOuterRemainder::readback",
                {"readbacks", "elements", "bytes"},
            ).items()
        }
        output_raw = required_span_args(
            events,
            "MetalOuterRemainder::output_claims",
            {
                "dispatch_wall_ns",
                "gpu_active_ns",
                "readbacks",
                "output_elements",
                "readback_bytes",
                "row_upload_bytes",
            },
        )
        output = {
            field: nonnegative_trace_integer(value, field)
            for field, value in output_raw.items()
        }
        if readback != {
            "readbacks": 1,
            "elements": 2 * (1 << cutoff_log2),
            "bytes": 2 * (1 << cutoff_log2) * 16,
        } or (
            output["dispatch_wall_ns"] <= 0
            or output["gpu_active_ns"] <= 0
            or output["readbacks"] != 1
            or output["output_elements"]
            != (37 if product_uniskip_carrier else 35)
            or output["readback_bytes"]
            != (37 if product_uniskip_carrier else 35) * 16
            or output["row_upload_bytes"] != 0
        ):
            raise ValueError("OuterRemainder readback accounting is inconsistent")

        product_uniskip_carrier_observation = None
        if product_uniskip_carrier:
            carrier_fields = {
                "rows",
                "source_rows_storage_id",
                "endpoint_elements",
            }
            carrier_raw = exact_span_args(
                events,
                "MetalOuterRemainder::product_uniskip_carrier_park",
                carrier_fields,
            )
            product_uniskip_carrier_observation = {
                field: nonnegative_trace_integer(carrier_raw[field], field)
                for field in carrier_fields
            }
            if product_uniskip_carrier_observation != {
                "rows": rows,
                "source_rows_storage_id": handoff["compact_rows_storage_id"],
                "endpoint_elements": 2,
            }:
                raise ValueError("OuterRemainder Product uni-skip carrier is inconsistent")

        release_fields = handoff_fields | {
            "residual_row_bytes",
            "remaining_sequence_storage_bytes",
            "compact_release_bytes",
            "deferred_owned_bytes",
            "release_mode",
            "cleanup_scope",
            "ownership_transfer_completed",
            "physical_release_completed",
            "residual_released",
            "residual_deferred",
            "compact_retained",
        }
        release_raw = required_span_args(
            events, "MetalOuterRemainder::row_release", release_fields
        )
        release = {
            field: nonnegative_trace_integer(release_raw[field], field)
            for field in release_fields
            if field
            not in {
                "release_mode",
                "cleanup_scope",
                "ownership_transfer_completed",
                "physical_release_completed",
                "residual_released",
                "residual_deferred",
                "compact_retained",
            }
        }
        for field in (
            "ownership_transfer_completed",
            "physical_release_completed",
            "residual_released",
            "residual_deferred",
            "compact_retained",
        ):
            release[field] = trace_boolean(release_raw[field])
        release["release_mode"] = trace_string(release_raw["release_mode"], "release_mode")
        release["cleanup_scope"] = trace_string(release_raw["cleanup_scope"], "cleanup_scope")
        if (
            any(release[field] != handoff[field] for field in handoff_fields)
            or release["release_mode"] != "proof_session_deferred"
            or release["cleanup_scope"] != "proof_session"
            or release["ownership_transfer_completed"] is not True
            or release["physical_release_completed"] is not False
            or release["residual_released"] is not False
            or release["residual_deferred"] is not True
            or release["compact_retained"] is not True
        ):
            raise ValueError("OuterRemainder resident row lifetime transfer is inconsistent")
        resource_observation = {
            "storage": storage,
            "sequence": sequence,
            "readback": readback,
            "output": output,
        }
        row_lifecycle = {"handoff": handoff, "release": release}

    member_us = interval_duration_us(member)
    if not math.isfinite(member_us) or member_us <= 0.0:
        raise ValueError("OuterRemainder complete-member duration is invalid")
    return {
        "components": {"member_us": member_us},
        "outer_counts": {
            "complete_member": 1,
            "sumcheck_round": len(rounds),
            "host_fiat_shamir": len(host_fiat_shamir),
        },
        "metal_counts": metal_counts,
        "resource_observation": resource_observation,
        "row_lifecycle": row_lifecycle,
        "product_uniskip_carrier": product_uniskip_carrier_observation,
    }


def product_uniskip_observation(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    product_uniskip_carrier: bool,
) -> dict[str, Any]:
    if backend not in {"optimized", "metal"}:
        raise ValueError(f"unsupported Product uni-skip backend {backend!r}")
    names = {
        PIOP_SPAN,
        f"{PRODUCT_UNISKIP_KERNEL}::prepare",
        f"{PRODUCT_UNISKIP_KERNEL}::first_round_poly",
        METAL_PRODUCT_UNISKIP_STANDALONE,
        METAL_PRODUCT_UNISKIP_CARRIER,
    }
    unknown = {
        name
        for event in events
        if isinstance((name := event.get("name")), str)
        and name.startswith("MetalProductUniskip::")
        and name not in names
    }
    if unknown:
        raise ValueError(
            f"Product uni-skip trace contains unknown Metal phases: {sorted(unknown)}"
        )
    intervals = strict_named_intervals(events, names)
    if len(intervals[PIOP_SPAN]) != 1:
        raise ValueError("trace must contain one PIOP span for Product uni-skip")
    piop = intervals[PIOP_SPAN][0]
    prepare = intervals[f"{PRODUCT_UNISKIP_KERNEL}::prepare"]
    first_round = intervals[f"{PRODUCT_UNISKIP_KERNEL}::first_round_poly"]
    expected_seams = 2 if backend == "optimized" else 1
    if len(prepare) != expected_seams or len(first_round) != expected_seams:
        raise ValueError("Product uni-skip seam topology is incomplete")
    for interval in [*prepare, *first_round]:
        require_contained(interval, piop, "Product uni-skip seam")
    if max(interval[1] for interval in prepare) > min(
        interval[0] for interval in first_round
    ):
        raise ValueError("Product uni-skip seams overlap or appear out of order")

    standalone = intervals[METAL_PRODUCT_UNISKIP_STANDALONE]
    carrier = intervals[METAL_PRODUCT_UNISKIP_CARRIER]
    expected_paths = (0, 0) if backend == "optimized" else (
        (0, 1) if product_uniskip_carrier else (1, 0)
    )
    if (len(standalone), len(carrier)) != expected_paths:
        raise ValueError("Product uni-skip Metal execution path is inconsistent")
    for interval in [*standalone, *carrier]:
        if not any(
            parent[0] <= interval[0] and interval[1] <= parent[1]
            for parent in prepare
        ):
            raise ValueError("Product uni-skip Metal execution lies outside prepare")

    rows = 1 << log_n
    resource_observation = None
    if standalone:
        fields = {
            "cycles",
            "resident_rows_storage_id",
            "row_upload_bytes",
            "round_device_buffer_allocations",
            "dispatch_wall_ns",
            "gpu_active_ns",
        }
        raw = exact_span_args(events, METAL_PRODUCT_UNISKIP_STANDALONE, fields)
        observation = {
            field: nonnegative_trace_integer(raw[field], field) for field in fields
        }
        if (
            observation["cycles"] != rows
            or observation["resident_rows_storage_id"] <= 0
            or observation["row_upload_bytes"] != 0
            or observation["round_device_buffer_allocations"] != 0
            or observation["dispatch_wall_ns"] <= 0
            or observation["gpu_active_ns"] <= 0
            or observation["gpu_active_ns"] > observation["dispatch_wall_ns"]
        ):
            raise ValueError("standalone Product uni-skip accounting is inconsistent")
        resource_observation = {
            "path": "standalone",
            "dispatches": 1,
            "command_buffers": 1,
            "readback_bytes": 2 * 16,
            **observation,
        }
    elif carrier:
        fields = {
            "cycles",
            "source_rows_storage_id",
            "product_rows_storage_id",
            "row_upload_bytes",
            "dispatches",
            "command_buffers",
            "readback_bytes",
        }
        raw = exact_span_args(events, METAL_PRODUCT_UNISKIP_CARRIER, fields)
        observation = {
            field: nonnegative_trace_integer(raw[field], field) for field in fields
        }
        if (
            observation["cycles"] != rows
            or observation["source_rows_storage_id"] <= 0
            or observation["product_rows_storage_id"] <= 0
            or any(
                observation[field] != 0
                for field in (
                    "row_upload_bytes",
                    "dispatches",
                    "command_buffers",
                    "readback_bytes",
                )
            )
        ):
            raise ValueError("carried Product uni-skip accounting is inconsistent")
        resource_observation = {"path": "outer_opening_carrier", **observation}

    member_us = union_duration_us([*prepare, *first_round])
    if not math.isfinite(member_us) or member_us <= 0.0:
        raise ValueError("Product uni-skip member duration is invalid")
    return {
        "components": {"member_us": member_us},
        "seam_counts": {
            "prepare": len(prepare),
            "first_round_poly": len(first_round),
        },
        "metal_counts": {
            "standalone": len(standalone),
            "carrier": len(carrier),
        },
        "resource_observation": resource_observation,
    }


def booleanity_address_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    inner_log2: int = 15,
    selectors_per_tile: int = 6,
    tile_threads: int = 512,
    finalize_threads: int = 1024,
    *,
    kernel: str = BOOLEANITY_ADDRESS_KERNEL,
    row_source_span: str = OPTIMIZED_BOOLEANITY_ADDRESS_ROW_SOURCE,
    require_hamming_lifecycle: bool = False,
) -> dict[str, Any]:
    if backend not in {"optimized", "metal"}:
        raise ValueError(f"unsupported Booleanity address backend {backend!r}")
    if (
        log_n < 0
        or inner_log2 < 0
        or inner_log2 > min(log_n, 16)
        or not 1 <= selectors_per_tile <= 6
        or tile_threads <= 0
        or tile_threads % 32 != 0
        or finalize_threads < 256
        or finalize_threads % 256 != 0
    ):
        raise ValueError("invalid Booleanity address evaluator geometry")
    outer_names = {
        f"{kernel}::{component}"
        for component in BOOLEANITY_ADDRESS_COMPONENTS
    }
    inner_names = {
        f"Metal{kernel}::{phase}"
        for phase in BOOLEANITY_ADDRESS_METAL_PHASES
    }
    lifecycle_names = {
        METAL_BOOLEANITY_ROWS_STAGE5_PREPARE,
        METAL_BOOLEANITY_ROWS_STAGE6A_USE,
        METAL_BOOLEANITY_ROWS_STAGE6B_USE,
    }
    hamming_lifecycle_names = {
        METAL_BOOLEANITY_ROWS_STAGE6B_RETAIN,
        METAL_BOOLEANITY_ROWS_STAGE7_HAMMING_USE,
    }
    allowed_metal_names = inner_names | lifecycle_names | hamming_lifecycle_names
    unknown_metal_names = {
        name
        for event in events
        if isinstance((name := event.get("name")), str)
        and (
            name.startswith(f"Metal{kernel}::")
            or name.startswith("MetalBooleanityRows::")
        )
        and name not in allowed_metal_names
    }
    if unknown_metal_names:
        raise ValueError(
            "Booleanity address trace contains unknown Metal phases: "
            f"{sorted(unknown_metal_names)}"
        )
    parent_names = {"InstructionReadRaf::prepare", "Booleanity::prepare"}
    if require_hamming_lifecycle:
        parent_names.add(f"{BOOLEANITY_ADDRESS_KERNEL}::prepare")
    intervals = strict_named_intervals(
        events,
        outer_names
        | inner_names
        | lifecycle_names
        | hamming_lifecycle_names
        | parent_names
        | {
            PIOP_SPAN,
            SUMCHECK_ROUND_SPAN,
            SUMCHECK_HOST_FIAT_SHAMIR_SPAN,
            row_source_span,
        },
    )
    if len(intervals[PIOP_SPAN]) != 1:
        raise ValueError("trace must contain exactly one positive PIOP span")
    piop = intervals[PIOP_SPAN][0]
    by_component = {
        component: sorted(intervals[f"{kernel}::{component}"])
        for component in BOOLEANITY_ADDRESS_COMPONENTS
    }
    expected_outer_counts = {
        "prepare": 1,
        "prove_round": 8,
        "finish_rounds": 1,
        "output_claims": 1,
    }
    outer_counts = {
        component: len(component_intervals)
        for component, component_intervals in by_component.items()
    }
    if outer_counts != expected_outer_counts:
        raise ValueError(
            "Booleanity address member span counts "
            f"{outer_counts}, expected {expected_outer_counts}"
        )

    prepare = by_component["prepare"][0]
    rounds = by_component["prove_round"]
    finish = by_component["finish_rounds"][0]
    output = by_component["output_claims"][0]
    ordered = [prepare, *rounds, finish, output]
    if any(start < piop[0] or end > piop[1] for start, end in ordered):
        raise ValueError("a Booleanity address member span lies outside PIOP")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError(
            "Booleanity address member spans overlap or appear out of order"
        )

    host_fiat_shamir = []
    for member_round in rounds:
        containing_rounds = [
            interval
            for interval in intervals[SUMCHECK_ROUND_SPAN]
            if interval[0] <= member_round[0] and member_round[1] <= interval[1]
        ]
        if len(containing_rounds) != 1:
            raise ValueError(
                "a Booleanity address round lacks one enclosing sumcheck round"
            )
        sumcheck_round = containing_rounds[0]
        round_fiat_shamir = [
            interval
            for interval in intervals[SUMCHECK_HOST_FIAT_SHAMIR_SPAN]
            if sumcheck_round[0] <= interval[0] and interval[1] <= sumcheck_round[1]
        ]
        if len(round_fiat_shamir) != 1:
            raise ValueError(
                "a Booleanity address round lacks one host Fiat-Shamir span"
            )
        fiat_shamir = round_fiat_shamir[0]
        if member_round[1] > fiat_shamir[0]:
            raise ValueError(
                "Booleanity address host Fiat-Shamir precedes its round polynomial"
            )
        host_fiat_shamir.append(fiat_shamir)
    if len(set(host_fiat_shamir)) != len(rounds):
        raise ValueError("Booleanity address rounds reuse a host Fiat-Shamir span")

    inner = {
        phase: sorted(intervals[f"Metal{kernel}::{phase}"])
        for phase in BOOLEANITY_ADDRESS_METAL_PHASES
    }
    inner_counts = {phase: len(values) for phase, values in inner.items()}
    row_lifecycle = None
    resource_observation = None
    row_source_intervals = intervals[row_source_span]
    if backend == "optimized":
        if any(inner_counts.values()) or any(
            intervals[name] for name in lifecycle_names | hamming_lifecycle_names
        ):
            raise ValueError(
                "optimized trace unexpectedly contains Booleanity address Metal spans"
            )
        if len(row_source_intervals) != 1:
            raise ValueError(
                "optimized Booleanity address trace must contain one row-source span"
            )
        require_contained(
            row_source_intervals[0], prepare, "optimized Booleanity address row source"
        )
    else:
        if row_source_intervals:
            raise ValueError(
                "Metal Booleanity address trace contains an optimized row-source span"
            )
        expected_inner_counts = {phase: 1 for phase in BOOLEANITY_ADDRESS_METAL_PHASES}
        if inner_counts != expected_inner_counts:
            raise ValueError(
                "Booleanity address Metal span counts "
                f"{inner_counts}, expected {expected_inner_counts}"
            )
        if any(len(intervals[name]) != 1 for name in lifecycle_names):
            raise ValueError("Booleanity address resident-row lifecycle is incomplete")
        hamming_lifecycle_counts = {
            name: len(intervals[name]) for name in hamming_lifecycle_names
        }
        allowed_hamming_lifecycle_counts = (
            {
                METAL_BOOLEANITY_ROWS_STAGE6B_RETAIN: 0,
                METAL_BOOLEANITY_ROWS_STAGE7_HAMMING_USE: 0,
            },
            {
                METAL_BOOLEANITY_ROWS_STAGE6B_RETAIN: 1,
                METAL_BOOLEANITY_ROWS_STAGE7_HAMMING_USE: 1,
            },
        )
        if (
            require_hamming_lifecycle
            and hamming_lifecycle_counts != allowed_hamming_lifecycle_counts[1]
        ) or (
            not require_hamming_lifecycle
            and hamming_lifecycle_counts not in allowed_hamming_lifecycle_counts
        ):
            raise ValueError("Booleanity Hamming resident-row lifecycle is incomplete")
        if len(intervals["InstructionReadRaf::prepare"]) != 1 or len(
            intervals["Booleanity::prepare"]
        ) != 1 or (
            require_hamming_lifecycle
            and len(intervals[f"{BOOLEANITY_ADDRESS_KERNEL}::prepare"]) != 1
        ):
            raise ValueError("Booleanity address resident-row lifecycle parents are incomplete")

        metal_prepare = inner["prepare"][0]
        sequence_interval = inner["sequence_prepare"][0]
        allocation_interval = inner["allocation_plan"][0]
        dispatch_interval = inner["dispatch"][0]
        readback_interval = inner["readback"][0]
        require_contained(metal_prepare, prepare, "Metal Booleanity address prepare")
        require_contained(
            sequence_interval, metal_prepare, "Booleanity address sequence preparation"
        )
        require_contained(
            allocation_interval,
            sequence_interval,
            "Booleanity address allocation plan",
        )
        require_contained(
            dispatch_interval, metal_prepare, "Booleanity address dispatch"
        )
        require_contained(
            readback_interval, metal_prepare, "Booleanity address readback"
        )
        if sequence_interval[1] > dispatch_interval[0] or dispatch_interval[1] > readback_interval[0]:
            raise ValueError(
                "Booleanity address sequence, dispatch, and readback overlap or are out of order"
            )

        stage5_interval = intervals[METAL_BOOLEANITY_ROWS_STAGE5_PREPARE][0]
        stage6a_interval = intervals[METAL_BOOLEANITY_ROWS_STAGE6A_USE][0]
        stage6b_interval = intervals[METAL_BOOLEANITY_ROWS_STAGE6B_USE][0]
        require_contained(
            stage5_interval,
            intervals["InstructionReadRaf::prepare"][0],
            "Booleanity stage-5 resident-row preparation",
        )
        stage6a_parent = (
            intervals[f"{BOOLEANITY_ADDRESS_KERNEL}::prepare"][0]
            if require_hamming_lifecycle
            else prepare
        )
        require_contained(
            stage6a_interval, stage6a_parent, "Booleanity stage-6a resident-row use"
        )
        require_contained(
            stage6b_interval,
            intervals["Booleanity::prepare"][0],
            "Booleanity stage-6b resident-row use",
        )
        if not (
            stage5_interval[1] <= stage6a_interval[0]
            and stage6a_interval[1] <= stage6b_interval[0]
        ):
            raise ValueError("Booleanity address resident-row lifecycle is out of order")
        if require_hamming_lifecycle:
            stage6b_retain_interval = intervals[
                METAL_BOOLEANITY_ROWS_STAGE6B_RETAIN
            ][0]
            stage7_interval = intervals[METAL_BOOLEANITY_ROWS_STAGE7_HAMMING_USE][0]
            require_contained(
                stage6b_retain_interval,
                intervals["Booleanity::prepare"][0],
                "Booleanity stage-6b Hamming retention",
            )
            require_contained(
                stage7_interval,
                metal_prepare,
                "Hamming stage-7 resident-row terminal use",
            )
            if not (
                stage6b_interval[1] <= stage6b_retain_interval[0]
                and stage6b_retain_interval[1] <= stage7_interval[0]
                and sequence_interval[1] <= stage7_interval[0]
            ):
                raise ValueError("Hamming resident-row lifecycle is out of order")
            require_contained(
                dispatch_interval,
                stage7_interval,
                "Hamming stage-7 resident-row dispatch",
            )
            require_contained(
                readback_interval,
                stage7_interval,
                "Hamming stage-7 resident-row readback",
            )

        lifecycle_fields = {
            "resident_rows_storage_id",
            "resident_rows",
            "resident_row_bytes",
            "device_registry_id",
            "row_allocations",
            "row_upload_bytes",
        }
        lifecycle_args = {
            "stage5": exact_span_args(
                events, METAL_BOOLEANITY_ROWS_STAGE5_PREPARE, lifecycle_fields
            ),
            "stage6a": exact_span_args(
                events, METAL_BOOLEANITY_ROWS_STAGE6A_USE, lifecycle_fields
            ),
            "stage6b": exact_span_args(
                events, METAL_BOOLEANITY_ROWS_STAGE6B_USE, lifecycle_fields
            ),
        }
        if require_hamming_lifecycle:
            lifecycle_args["stage6b_retain"] = exact_span_args(
                events,
                METAL_BOOLEANITY_ROWS_STAGE6B_RETAIN,
                lifecycle_fields,
            )
            lifecycle_args["stage7"] = exact_span_args(
                events,
                METAL_BOOLEANITY_ROWS_STAGE7_HAMMING_USE,
                lifecycle_fields
                | {"terminal_consumer", "terminal_carry_removed"},
            )
        parsed_lifecycle = {
            stage: {
                field: booleanity_trace_integer(value, f"{stage} {field}", allow_zero=True)
                for field, value in args.items()
                if field in lifecycle_fields
            }
            for stage, args in lifecycle_args.items()
        }
        lifecycle_stages = tuple(parsed_lifecycle)
        rows = 1 << log_n
        storage_ids = [
            parsed_lifecycle[stage]["resident_rows_storage_id"]
            for stage in lifecycle_stages
        ]
        registries = [
            parsed_lifecycle[stage]["device_registry_id"]
            for stage in lifecycle_stages
        ]
        terminal_exact = not require_hamming_lifecycle or (
            trace_boolean(lifecycle_args["stage7"]["terminal_consumer"]) is True
            and trace_boolean(lifecycle_args["stage7"]["terminal_carry_removed"])
            is True
        )
        if (
            any(storage_id <= 0 for storage_id in storage_ids)
            or len(set(storage_ids)) != 1
            or any(registry <= 0 for registry in registries)
            or len(set(registries)) != 1
            or any(
                parsed_lifecycle[stage]["resident_rows"] != rows
                or parsed_lifecycle[stage]["resident_row_bytes"] != 40
                for stage in lifecycle_stages
            )
            or parsed_lifecycle["stage5"]["row_allocations"] != 1
            or parsed_lifecycle["stage5"]["row_upload_bytes"] != rows * 40
            or any(
                parsed_lifecycle[stage]["row_allocations"] != 0
                or parsed_lifecycle[stage]["row_upload_bytes"] != 0
                for stage in lifecycle_stages
                if stage != "stage5"
            )
            or not terminal_exact
        ):
            raise ValueError("Booleanity address resident-row lifecycle is inconsistent")
        row_lifecycle = {
            "kind": (
                "metal_hamming_resident"
                if require_hamming_lifecycle
                else "metal_booleanity_resident"
            ),
            "rows": rows,
            "row_bytes": 40,
            "device_registry_id": registries[0],
            "stage5_storage_id": storage_ids[0],
            "stage6a_storage_id": storage_ids[1],
            "stage6b_storage_id": storage_ids[2],
            **{
                stage: {
                    "row_allocations": parsed_lifecycle[stage]["row_allocations"],
                    "row_upload_bytes": parsed_lifecycle[stage]["row_upload_bytes"],
                }
                for stage in lifecycle_stages
            },
        }
        if require_hamming_lifecycle:
            row_lifecycle.update(
                {
                    "stage6b_retain_storage_id": storage_ids[3],
                    "stage7_storage_id": storage_ids[4],
                    "terminal_consumer": True,
                    "terminal_carry_removed": True,
                }
            )

        sequence_fields = {
            "resident_rows_storage_id",
            "resident_rows",
            "resident_row_bytes",
            "row_upload_bytes",
            "polys",
            "k",
            "e_in_elements",
            "e_out_elements",
            "requested_inner_log2",
            "effective_inner_log2",
            "requested_selectors_per_tile",
            "effective_selectors_per_tile",
            "requested_tile_threads",
            "effective_tile_threads",
            "requested_finalize_threads",
            "effective_finalize_threads",
            "selector_tiles",
            "production_specialized",
        }
        sequence_args = exact_span_args(
            events,
            f"Metal{kernel}::sequence_prepare",
            sequence_fields,
        )
        production_specialized = trace_boolean(sequence_args["production_specialized"])
        if production_specialized is None:
            raise ValueError(
                "Booleanity address sequence has invalid production_specialized"
            )
        sequence = {
            field: booleanity_trace_integer(value, f"sequence {field}", allow_zero=True)
            for field, value in sequence_args.items()
            if field != "production_specialized"
        }
        sequence["production_specialized"] = production_specialized
        selector_tiles = 29 // selectors_per_tile + (29 % selectors_per_tile != 0)
        expected_sequence = {
            "resident_rows_storage_id": storage_ids[0],
            "resident_rows": rows,
            "resident_row_bytes": 40,
            "row_upload_bytes": 0,
            "polys": 29,
            "k": 256,
            "e_in_elements": 1 << inner_log2,
            "e_out_elements": 1 << (log_n - inner_log2),
            "requested_inner_log2": inner_log2,
            "effective_inner_log2": inner_log2,
            "requested_selectors_per_tile": selectors_per_tile,
            "effective_selectors_per_tile": selectors_per_tile,
            "requested_tile_threads": tile_threads,
            "effective_tile_threads": tile_threads,
            "requested_finalize_threads": finalize_threads,
            "effective_finalize_threads": finalize_threads,
            "selector_tiles": selector_tiles,
            "production_specialized": selectors_per_tile in {3, 6},
        }
        if sequence != expected_sequence:
            raise ValueError(
                f"Booleanity address sequence geometry is inconsistent: {sequence}"
            )

        allocation_fields = {
            "device_buffers",
            "planned_device_bytes",
            "current_device_bytes",
            "recommended_device_bytes",
        }
        allocation_args = exact_span_args(
            events,
            f"Metal{kernel}::allocation_plan",
            allocation_fields,
        )
        allocation = {
            field: booleanity_trace_integer(value, field, allow_zero=True)
            for field, value in allocation_args.items()
        }
        expected_planned_bytes = booleanity_address_sequence_storage_bytes(
            log_n, inner_log2, selectors_per_tile
        )
        if (
            allocation["device_buffers"] != 5
            or allocation["planned_device_bytes"] != expected_planned_bytes
            or allocation["current_device_bytes"] < rows * 40
            or allocation["current_device_bytes"] + allocation["planned_device_bytes"]
            > allocation["recommended_device_bytes"]
        ):
            raise ValueError("Booleanity address allocation plan has invalid buffer accounting")

        dispatch_fields = {
            "command_buffers",
            "tile_dispatches",
            "finalize_dispatches",
            "command_completed",
            "gpu_active_ns",
            "resident_rows_storage_id",
        }
        dispatch_args = exact_span_args(
            events,
            f"Metal{kernel}::dispatch",
            dispatch_fields,
        )
        command_completed = trace_boolean(dispatch_args["command_completed"])
        if command_completed is not True:
            raise ValueError("Booleanity address command did not complete")
        dispatch = {
            field: booleanity_trace_integer(value, f"dispatch {field}", allow_zero=True)
            for field, value in dispatch_args.items()
            if field != "command_completed"
        }
        dispatch["command_completed"] = command_completed
        if (
            dispatch["command_buffers"] != 1
            or dispatch["tile_dispatches"] != selector_tiles
            or dispatch["finalize_dispatches"] != selector_tiles
            or dispatch["gpu_active_ns"] <= 0
            or dispatch["resident_rows_storage_id"] != storage_ids[0]
        ):
            raise ValueError("Booleanity address dispatch accounting is inconsistent")

        readback_fields = {"elements", "bytes", "readbacks"}
        readback_args = exact_span_args(
            events,
            f"Metal{kernel}::readback",
            readback_fields,
        )
        readback = {
            field: booleanity_trace_integer(value, f"readback {field}", allow_zero=True)
            for field, value in readback_args.items()
        }
        if readback != {"elements": 29 * 256, "bytes": 29 * 256 * 16, "readbacks": 1}:
            raise ValueError("Booleanity address readback accounting is inconsistent")
        resource_observation = {
            "sequence": sequence,
            "allocation": allocation,
            "dispatch": dispatch,
            "readback": readback,
        }

    round_durations = [interval_duration_us(interval) for interval in rounds]
    host_fiat_shamir_durations = [
        interval_duration_us(interval) for interval in host_fiat_shamir
    ]
    prepare_us = interval_duration_us(prepare)
    row_source_us = (
        interval_duration_us(row_source_intervals[0]) if row_source_intervals else 0.0
    )
    normalized_prepare_us = prepare_us - row_source_us
    components = {
        "prepare_us": prepare_us,
        "row_source_us": row_source_us,
        "normalized_prepare_us": normalized_prepare_us,
        "rounds_us": round_durations,
        "rounds_total_us": sum(round_durations),
        "host_fiat_shamir_us": host_fiat_shamir_durations,
        "host_fiat_shamir_total_us": sum(host_fiat_shamir_durations),
        "finish_us": interval_duration_us(finish),
        "output_claims_us": interval_duration_us(output),
    }
    components["member_us"] = (
        components["prepare_us"]
        + components["rounds_total_us"]
        + components["host_fiat_shamir_total_us"]
        + components["finish_us"]
        + components["output_claims_us"]
    )
    components["normalized_member_us"] = (
        components["normalized_prepare_us"]
        + components["rounds_total_us"]
        + components["host_fiat_shamir_total_us"]
        + components["finish_us"]
        + components["output_claims_us"]
    )
    positive_components = (
        "prepare_us",
        "normalized_prepare_us",
        "rounds_total_us",
        "host_fiat_shamir_total_us",
        "finish_us",
        "output_claims_us",
        "member_us",
        "normalized_member_us",
    )
    if any(
        not math.isfinite(components[name]) or components[name] <= 0.0
        for name in positive_components
    ) or not math.isfinite(row_source_us) or row_source_us < 0.0:
        raise ValueError("trace contains a non-positive Booleanity address member duration")
    return {
        "components": components,
        "outer_counts": outer_counts,
        "metal_counts": inner_counts,
        "resource_observation": resource_observation,
        "row_lifecycle": row_lifecycle,
    }


def hamming_weight_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    inner_log2: int = 15,
    selectors_per_tile: int = 6,
    tile_threads: int = 512,
    finalize_threads: int = 1024,
) -> dict[str, Any]:
    return booleanity_address_member_breakdown(
        events,
        backend,
        log_n,
        inner_log2,
        selectors_per_tile,
        tile_threads,
        finalize_threads,
        kernel=HAMMING_WEIGHT_KERNEL,
        row_source_span=OPTIMIZED_HAMMING_WEIGHT_ROW_SOURCE,
        require_hamming_lifecycle=True,
    )


def booleanity_outer_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    kernel: str,
    row_source_span: str,
) -> dict[str, Any]:
    outer_names = {
        f"{kernel}::{component}" for component in BOOLEANITY_ADDRESS_COMPONENTS
    }
    intervals = strict_named_intervals(
        events,
        outer_names
        | {
            PIOP_SPAN,
            SUMCHECK_ROUND_SPAN,
            SUMCHECK_HOST_FIAT_SHAMIR_SPAN,
            row_source_span,
        },
    )
    if len(intervals[PIOP_SPAN]) != 1:
        raise ValueError("trace must contain exactly one positive PIOP span")
    piop = intervals[PIOP_SPAN][0]
    by_component = {
        component: sorted(intervals[f"{kernel}::{component}"])
        for component in BOOLEANITY_ADDRESS_COMPONENTS
    }
    outer_counts = {
        component: len(component_intervals)
        for component, component_intervals in by_component.items()
    }
    expected_outer_counts = {
        "prepare": 1,
        "prove_round": 8,
        "finish_rounds": 1,
        "output_claims": 1,
    }
    if outer_counts != expected_outer_counts:
        raise ValueError(
            f"{kernel} member span counts {outer_counts}, expected {expected_outer_counts}"
        )

    prepare = by_component["prepare"][0]
    rounds = by_component["prove_round"]
    finish = by_component["finish_rounds"][0]
    output = by_component["output_claims"][0]
    ordered = [prepare, *rounds, finish, output]
    if any(start < piop[0] or end > piop[1] for start, end in ordered):
        raise ValueError(f"a {kernel} member span lies outside PIOP")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"{kernel} member spans overlap or appear out of order")

    host_fiat_shamir = []
    for member_round in rounds:
        containing_rounds = [
            interval
            for interval in intervals[SUMCHECK_ROUND_SPAN]
            if interval[0] <= member_round[0] and member_round[1] <= interval[1]
        ]
        if len(containing_rounds) != 1:
            raise ValueError(f"a {kernel} round lacks one enclosing sumcheck round")
        sumcheck_round = containing_rounds[0]
        round_fiat_shamir = [
            interval
            for interval in intervals[SUMCHECK_HOST_FIAT_SHAMIR_SPAN]
            if sumcheck_round[0] <= interval[0] and interval[1] <= sumcheck_round[1]
        ]
        if len(round_fiat_shamir) != 1:
            raise ValueError(f"a {kernel} round lacks one host Fiat-Shamir span")
        fiat_shamir = round_fiat_shamir[0]
        if member_round[1] > fiat_shamir[0]:
            raise ValueError(f"{kernel} host Fiat-Shamir precedes its round polynomial")
        host_fiat_shamir.append(fiat_shamir)
    if len(set(host_fiat_shamir)) != len(rounds):
        raise ValueError(f"{kernel} rounds reuse a host Fiat-Shamir span")

    row_sources = intervals[row_source_span]
    if backend == "optimized":
        if len(row_sources) != 1:
            raise ValueError(f"optimized {kernel} must contain one row-source span")
        require_contained(row_sources[0], prepare, f"optimized {kernel} row source")
    elif backend == "metal":
        if row_sources:
            raise ValueError(f"Metal {kernel} contains an optimized row-source span")
    else:
        raise ValueError(f"unsupported {kernel} backend {backend!r}")

    round_durations = [interval_duration_us(interval) for interval in rounds]
    host_fiat_shamir_durations = [
        interval_duration_us(interval) for interval in host_fiat_shamir
    ]
    prepare_us = interval_duration_us(prepare)
    row_source_us = interval_duration_us(row_sources[0]) if row_sources else 0.0
    components = {
        "prepare_us": prepare_us,
        "row_source_us": row_source_us,
        "normalized_prepare_us": prepare_us - row_source_us,
        "rounds_us": round_durations,
        "rounds_total_us": sum(round_durations),
        "host_fiat_shamir_us": host_fiat_shamir_durations,
        "host_fiat_shamir_total_us": sum(host_fiat_shamir_durations),
        "finish_us": interval_duration_us(finish),
        "output_claims_us": interval_duration_us(output),
    }
    components["member_us"] = (
        components["prepare_us"]
        + components["rounds_total_us"]
        + components["host_fiat_shamir_total_us"]
        + components["finish_us"]
        + components["output_claims_us"]
    )
    components["normalized_member_us"] = components["member_us"] - row_source_us
    positive = (
        "prepare_us",
        "normalized_prepare_us",
        "rounds_total_us",
        "host_fiat_shamir_total_us",
        "finish_us",
        "output_claims_us",
        "member_us",
        "normalized_member_us",
    )
    if any(
        not math.isfinite(components[name]) or components[name] <= 0.0
        for name in positive
    ) or not math.isfinite(row_source_us) or row_source_us < 0.0:
        raise ValueError(f"trace contains a non-positive {kernel} duration")
    return {
        "components": components,
        "outer_counts": outer_counts,
        "prepare_interval": prepare,
    }


def retained_hamming_lifecycle(
    events: list[dict[str, Any]], log_n: int
) -> dict[str, Any]:
    raw_names = {
        METAL_BOOLEANITY_ROWS_STAGE5_PREPARE,
        METAL_BOOLEANITY_ROWS_STAGE6A_USE,
        METAL_BOOLEANITY_ROWS_STAGE6B_USE,
    }
    hot_names = {METAL_HAMMING_HOT_STAGE6B_RETAIN, METAL_HAMMING_HOT_STAGE7_USE}
    parent_names = {
        "InstructionReadRaf::prepare",
        f"{BOOLEANITY_ADDRESS_KERNEL}::prepare",
        "Booleanity::prepare",
        f"{HAMMING_WEIGHT_KERNEL}::prepare",
    }
    intervals = strict_named_intervals(events, raw_names | hot_names | parent_names)
    if any(len(intervals[name]) != 1 for name in raw_names | hot_names | parent_names):
        raise ValueError("retained Hamming lifecycle is incomplete")

    raw_intervals = {
        "stage5": intervals[METAL_BOOLEANITY_ROWS_STAGE5_PREPARE][0],
        "stage6a": intervals[METAL_BOOLEANITY_ROWS_STAGE6A_USE][0],
        "stage6b": intervals[METAL_BOOLEANITY_ROWS_STAGE6B_USE][0],
    }
    hot_intervals = {
        "stage6b_retain": intervals[METAL_HAMMING_HOT_STAGE6B_RETAIN][0],
        "stage7": intervals[METAL_HAMMING_HOT_STAGE7_USE][0],
    }
    require_contained(
        raw_intervals["stage5"],
        intervals["InstructionReadRaf::prepare"][0],
        "retained Hamming stage-5 source",
    )
    require_contained(
        raw_intervals["stage6a"],
        intervals[f"{BOOLEANITY_ADDRESS_KERNEL}::prepare"][0],
        "retained Hamming stage-6a source use",
    )
    require_contained(
        raw_intervals["stage6b"],
        intervals["Booleanity::prepare"][0],
        "retained Hamming stage-6b source use",
    )
    require_contained(
        hot_intervals["stage6b_retain"],
        intervals["Booleanity::prepare"][0],
        "retained Hamming projection retention",
    )
    require_contained(
        hot_intervals["stage7"],
        intervals[f"{HAMMING_WEIGHT_KERNEL}::prepare"][0],
        "retained Hamming terminal projection use",
    )
    ordered = [
        raw_intervals["stage5"],
        raw_intervals["stage6a"],
        hot_intervals["stage6b_retain"],
        raw_intervals["stage6b"],
        hot_intervals["stage7"],
    ]
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("retained Hamming lifecycle is out of order")

    raw_fields = {
        "resident_rows_storage_id",
        "resident_rows",
        "resident_row_bytes",
        "device_registry_id",
        "row_allocations",
        "row_upload_bytes",
    }
    raw_args = {
        "stage5": exact_span_args(
            events, METAL_BOOLEANITY_ROWS_STAGE5_PREPARE, raw_fields
        ),
        "stage6a": exact_span_args(
            events, METAL_BOOLEANITY_ROWS_STAGE6A_USE, raw_fields
        ),
        "stage6b": exact_span_args(
            events, METAL_BOOLEANITY_ROWS_STAGE6B_USE, raw_fields
        ),
    }
    raw = {
        stage: {
            field: booleanity_trace_integer(value, f"{stage} {field}", allow_zero=True)
            for field, value in args.items()
        }
        for stage, args in raw_args.items()
    }
    rows = 1 << log_n
    source_ids = [raw[stage]["resident_rows_storage_id"] for stage in raw]
    registries = [raw[stage]["device_registry_id"] for stage in raw]
    if (
        any(identity <= 0 for identity in source_ids)
        or len(set(source_ids)) != 1
        or any(registry <= 0 for registry in registries)
        or len(set(registries)) != 1
        or any(
            raw[stage]["resident_rows"] != rows
            or raw[stage]["resident_row_bytes"] != 40
            for stage in raw
        )
        or raw["stage5"]["row_allocations"] != 1
        or raw["stage5"]["row_upload_bytes"] != rows * 40
        or any(
            raw[stage]["row_allocations"] != 0
            or raw[stage]["row_upload_bytes"] != 0
            for stage in ("stage6a", "stage6b")
        )
    ):
        raise ValueError("retained Hamming source-row lifecycle is inconsistent")

    retain_fields = {
        "hot_rows_storage_id",
        "source_rows_storage_id",
        "hot_rows",
        "hot_row_bytes",
        "device_registry_id",
        "row_allocations",
        "row_upload_bytes",
    }
    terminal_fields = {
        "hot_rows_storage_id",
        "source_rows_storage_id",
        "hot_rows",
        "hot_row_bytes",
        "device_registry_id",
        "row_allocations",
        "row_upload_bytes",
        "terminal_consumer",
        "terminal_carry_removed",
    }
    retain_args = exact_span_args(
        events, METAL_HAMMING_HOT_STAGE6B_RETAIN, retain_fields
    )
    terminal_args = exact_span_args(events, METAL_HAMMING_HOT_STAGE7_USE, terminal_fields)
    retain = {
        field: booleanity_trace_integer(value, f"retained {field}", allow_zero=True)
        for field, value in retain_args.items()
    }
    terminal = {
        field: booleanity_trace_integer(value, f"terminal {field}", allow_zero=True)
        for field, value in terminal_args.items()
        if field not in {"terminal_consumer", "terminal_carry_removed"}
    }
    terminal_consumer = trace_boolean(terminal_args["terminal_consumer"])
    terminal_carry_removed = trace_boolean(
        terminal_args["terminal_carry_removed"]
    )
    if (
        retain["hot_rows_storage_id"] <= 0
        or retain["hot_rows_storage_id"] != terminal["hot_rows_storage_id"]
        or retain["source_rows_storage_id"] != source_ids[0]
        or terminal["source_rows_storage_id"] != source_ids[0]
        or retain["hot_rows"] != rows
        or terminal["hot_rows"] != rows
        or retain["hot_row_bytes"] != 29
        or terminal["hot_row_bytes"] != 29
        or retain["device_registry_id"] != registries[0]
        or terminal["device_registry_id"] != registries[0]
        or retain["row_allocations"] != 0
        or retain["row_upload_bytes"] != 0
        or terminal["row_allocations"] != 0
        or terminal["row_upload_bytes"] != 0
        or terminal_consumer is not True
        or terminal_carry_removed is not True
    ):
        raise ValueError("retained Hamming projection lifecycle is inconsistent")

    return {
        "kind": "metal_hamming_hot",
        "rows": rows,
        "source_row_bytes": 40,
        "hot_row_bytes": 29,
        "device_registry_id": registries[0],
        "source_rows_storage_id": source_ids[0],
        "hot_rows_storage_id": retain["hot_rows_storage_id"],
        "stage5": {
            "row_allocations": raw["stage5"]["row_allocations"],
            "row_upload_bytes": raw["stage5"]["row_upload_bytes"],
        },
        "stage6a": {
            "row_allocations": raw["stage6a"]["row_allocations"],
            "row_upload_bytes": raw["stage6a"]["row_upload_bytes"],
        },
        "stage6b": {
            "row_allocations": raw["stage6b"]["row_allocations"],
            "row_upload_bytes": raw["stage6b"]["row_upload_bytes"],
        },
        "stage6b_retain": {
            "row_allocations": retain["row_allocations"],
            "row_upload_bytes": retain["row_upload_bytes"],
        },
        "stage7": {
            "row_allocations": terminal["row_allocations"],
            "row_upload_bytes": terminal["row_upload_bytes"],
            "terminal_consumer": True,
            "terminal_carry_removed": True,
        },
    }


def packed_hot_booleanity_address_member_breakdown(
    events: list[dict[str, Any]], backend: str, log_n: int
) -> dict[str, Any]:
    if backend != "metal":
        raise ValueError("packed-hot Booleanity parsing requires the Metal arm")
    outer = booleanity_outer_member_breakdown(
        events,
        backend,
        BOOLEANITY_ADDRESS_KERNEL,
        OPTIMIZED_BOOLEANITY_ADDRESS_ROW_SOURCE,
    )
    lifecycle = retained_hamming_lifecycle(events, log_n)
    names = {
        "prepare": "MetalBooleanityAddressPhase::prepare",
        "sequence": "MetalBooleanityAddressPhase::packed_hot_sequence",
        "dispatch": "MetalBooleanityAddressPhase::packed_hot_dispatch",
        "readback": "MetalBooleanityAddressPhase::packed_hot_readback",
    }
    intervals = strict_named_intervals(events, set(names.values()))
    if any(len(intervals[name]) != 1 for name in names.values()):
        raise ValueError("packed-hot Booleanity trace is incomplete")
    prepare = intervals[names["prepare"]][0]
    sequence_interval = intervals[names["sequence"]][0]
    dispatch_interval = intervals[names["dispatch"]][0]
    readback_interval = intervals[names["readback"]][0]
    require_contained(prepare, outer["prepare_interval"], "packed-hot prepare")
    for interval, description in (
        (sequence_interval, "packed-hot sequence"),
        (dispatch_interval, "packed-hot dispatch"),
        (readback_interval, "packed-hot readback"),
    ):
        require_contained(interval, prepare, description)
    if sequence_interval[1] > dispatch_interval[0] or dispatch_interval[1] > readback_interval[0]:
        raise ValueError("packed-hot sequence, dispatch, and readback are out of order")

    sequence_fields = {
        "resident_rows_storage_id",
        "hot_rows_storage_id",
        "rows",
        "resident_row_bytes",
        "hot_bytes",
        "validity_bytes",
        "e_in_fields",
        "e_out_fields",
        "partial_fields",
        "output_fields",
        "owned_bytes",
        "current_device_bytes",
        "recommended_device_bytes",
        "command_buffers",
        "dispatches",
        "readbacks",
    }
    sequence = {
        field: booleanity_trace_integer(value, f"packed-hot {field}", allow_zero=True)
        for field, value in exact_span_args(
            events, names["sequence"], sequence_fields
        ).items()
    }
    rows = 1 << log_n
    e_in = 1 << 15
    e_out = rows // e_in
    partial_fields = e_out * 29 * 256
    output_fields = 29 * 256
    owned_bytes = (
        29 * rows
        + rows
        + 16 * (e_in + e_out + partial_fields + output_fields)
    )
    expected_sequence = {
        "resident_rows_storage_id": lifecycle["source_rows_storage_id"],
        "hot_rows_storage_id": lifecycle["hot_rows_storage_id"],
        "rows": rows,
        "resident_row_bytes": 40,
        "hot_bytes": 29 * rows,
        "validity_bytes": rows,
        "e_in_fields": e_in,
        "e_out_fields": e_out,
        "partial_fields": partial_fields,
        "output_fields": output_fields,
        "owned_bytes": owned_bytes,
        "command_buffers": 1,
        "dispatches": 3,
        "readbacks": 1,
    }
    observed_sequence = {
        field: value
        for field, value in sequence.items()
        if field not in {"current_device_bytes", "recommended_device_bytes"}
    }
    if (
        observed_sequence != expected_sequence
        or sequence["current_device_bytes"] + owned_bytes
        > sequence["recommended_device_bytes"]
    ):
        raise ValueError(f"packed-hot sequence geometry is inconsistent: {sequence}")

    dispatch_fields = {
        "command_buffers",
        "dispatches",
        "command_completed",
        "gpu_active_ns",
        "resident_rows_storage_id",
        "hot_rows_storage_id",
    }
    dispatch_args = exact_span_args(events, names["dispatch"], dispatch_fields)
    dispatch = {
        field: booleanity_trace_integer(value, f"packed-hot {field}", allow_zero=True)
        for field, value in dispatch_args.items()
        if field != "command_completed"
    }
    dispatch["command_completed"] = trace_boolean(dispatch_args["command_completed"])
    if (
        dispatch["command_buffers"] != 1
        or dispatch["dispatches"] != 3
        or dispatch["command_completed"] is not True
        or dispatch["gpu_active_ns"] <= 0
        or dispatch["resident_rows_storage_id"] != lifecycle["source_rows_storage_id"]
        or dispatch["hot_rows_storage_id"] != lifecycle["hot_rows_storage_id"]
    ):
        raise ValueError("packed-hot dispatch accounting is inconsistent")

    readback = {
        field: booleanity_trace_integer(value, f"packed-hot readback {field}")
        for field, value in exact_span_args(
            events, names["readback"], {"elements", "bytes", "readbacks"}
        ).items()
    }
    expected_readback = {
        "elements": output_fields,
        "bytes": output_fields * 16,
        "readbacks": 1,
    }
    if readback != expected_readback:
        raise ValueError("packed-hot readback accounting is inconsistent")
    return {
        "components": outer["components"],
        "outer_counts": outer["outer_counts"],
        "metal_counts": {phase: 1 for phase in names},
        "resource_observation": {
            "implementation": "packed-hot",
            "sequence": sequence,
            "dispatch": dispatch,
            "readback": readback,
        },
        "row_lifecycle": lifecycle,
    }


def retained_hot_hamming_weight_member_breakdown(
    events: list[dict[str, Any]], backend: str, log_n: int
) -> dict[str, Any]:
    if backend != "metal":
        raise ValueError("retained-hot Hamming parsing requires the Metal arm")
    outer = booleanity_outer_member_breakdown(
        events,
        backend,
        HAMMING_WEIGHT_KERNEL,
        OPTIMIZED_HAMMING_WEIGHT_ROW_SOURCE,
    )
    lifecycle = retained_hamming_lifecycle(events, log_n)
    names = {
        "sequence": "MetalHammingWeightClaimReduction::retained_sequence",
        "dispatch": "MetalHammingWeightClaimReduction::retained_dispatch",
        "readback": "MetalHammingWeightClaimReduction::retained_readback",
    }
    intervals = strict_named_intervals(events, set(names.values()))
    if any(len(intervals[name]) != 1 for name in names.values()):
        raise ValueError("retained-hot Hamming trace is incomplete")
    for name in names.values():
        require_contained(
            intervals[name][0], outer["prepare_interval"], f"retained-hot {name}"
        )
    if (
        intervals[names["sequence"]][0][1] > intervals[names["dispatch"]][0][0]
        or intervals[names["dispatch"]][0][1] > intervals[names["readback"]][0][0]
    ):
        raise ValueError("retained-hot sequence, dispatch, and readback are out of order")

    sequence_fields = {
        "hot_rows_storage_id",
        "source_rows_storage_id",
        "rows",
        "hot_bytes",
        "e_in_fields",
        "e_out_fields",
        "partial_fields",
        "output_fields",
        "owned_bytes",
        "current_device_bytes",
        "recommended_device_bytes",
        "command_buffers",
        "encoders",
        "dispatches",
        "tile_threadgroups",
        "finalize_threadgroups",
        "readbacks",
    }
    sequence = {
        field: booleanity_trace_integer(value, f"retained-hot {field}", allow_zero=True)
        for field, value in exact_span_args(
            events, names["sequence"], sequence_fields
        ).items()
    }
    rows = 1 << log_n
    e_in = 1 << 15
    e_out = rows // e_in
    partial_fields = e_out * 6 * 256
    output_fields = 29 * 256
    owned_bytes = 16 * (e_in + e_out + partial_fields + output_fields)
    expected_sequence = {
        "hot_rows_storage_id": lifecycle["hot_rows_storage_id"],
        "source_rows_storage_id": lifecycle["source_rows_storage_id"],
        "rows": rows,
        "hot_bytes": 29 * rows,
        "e_in_fields": e_in,
        "e_out_fields": e_out,
        "partial_fields": partial_fields,
        "output_fields": output_fields,
        "owned_bytes": owned_bytes,
        "command_buffers": 1,
        "encoders": 10,
        "dispatches": 10,
        "tile_threadgroups": e_out * 5,
        "finalize_threadgroups": 29,
        "readbacks": 1,
    }
    observed_sequence = {
        field: value
        for field, value in sequence.items()
        if field not in {"current_device_bytes", "recommended_device_bytes"}
    }
    if (
        observed_sequence != expected_sequence
        or sequence["current_device_bytes"] + owned_bytes
        > sequence["recommended_device_bytes"]
    ):
        raise ValueError(f"retained-hot sequence geometry is inconsistent: {sequence}")

    dispatch_fields = {
        "command_buffers",
        "tile_dispatches",
        "finalize_dispatches",
        "command_completed",
        "gpu_active_ns",
        "hot_rows_storage_id",
    }
    dispatch_args = exact_span_args(events, names["dispatch"], dispatch_fields)
    dispatch = {
        field: booleanity_trace_integer(value, f"retained-hot {field}", allow_zero=True)
        for field, value in dispatch_args.items()
        if field != "command_completed"
    }
    dispatch["command_completed"] = trace_boolean(dispatch_args["command_completed"])
    if (
        dispatch["command_buffers"] != 1
        or dispatch["tile_dispatches"] != 5
        or dispatch["finalize_dispatches"] != 5
        or dispatch["command_completed"] is not True
        or dispatch["gpu_active_ns"] <= 0
        or dispatch["hot_rows_storage_id"] != lifecycle["hot_rows_storage_id"]
    ):
        raise ValueError("retained-hot dispatch accounting is inconsistent")

    readback = {
        field: booleanity_trace_integer(value, f"retained-hot readback {field}")
        for field, value in exact_span_args(
            events, names["readback"], {"elements", "bytes", "readbacks"}
        ).items()
    }
    expected_readback = {
        "elements": output_fields,
        "bytes": output_fields * 16,
        "readbacks": 1,
    }
    if readback != expected_readback:
        raise ValueError("retained-hot readback accounting is inconsistent")
    return {
        "components": outer["components"],
        "outer_counts": outer["outer_counts"],
        "metal_counts": {phase: 1 for phase in names},
        "resource_observation": {
            "implementation": "retained-hot",
            "sequence": sequence,
            "dispatch": dispatch,
            "readback": readback,
        },
        "row_lifecycle": lifecycle,
    }


def instruction_input_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    cutoff_log2: int = 16,
    borrow_outer_residual: bool = False,
) -> dict[str, Any]:
    outer_names = {
        f"{INSTRUCTION_INPUT_KERNEL}::{component}"
        for component in INSTRUCTION_INPUT_COMPONENTS
    }
    inner_names = {
        f"Metal{INSTRUCTION_INPUT_KERNEL}::{phase}"
        for phase in INSTRUCTION_INPUT_METAL_PHASES
    }
    lifecycle_names = {
        OPTIMIZED_INSTRUCTION_INPUT_ROWS_PREPARE,
        OPTIMIZED_INSTRUCTION_INPUT_ROWS_STAGE3_USE,
        METAL_INSTRUCTION_INPUT_ROWS_PREPARE,
        METAL_INSTRUCTION_INPUT_ROWS_STAGE1_HANDOFF,
    }
    allowed_metal_names = inner_names | {
        METAL_INSTRUCTION_INPUT_ROWS_PREPARE,
        METAL_INSTRUCTION_INPUT_ROWS_STAGE1_HANDOFF,
    }
    unknown_metal_names = {
        name
        for event in events
        if isinstance((name := event.get("name")), str)
        and name.startswith(f"Metal{INSTRUCTION_INPUT_KERNEL}::")
        and name not in allowed_metal_names
    }
    if unknown_metal_names:
        raise ValueError(
            f"InstructionInput trace contains unknown Metal phases: {sorted(unknown_metal_names)}"
        )
    intervals = strict_named_intervals(
        events,
        outer_names
        | inner_names
        | lifecycle_names
        | {
            PIOP_SPAN,
            BACKEND_WITNESS_PREP_SPAN,
            SPARTAN_SHIFT_PREPARE_SPAN,
            METAL_OUTER_REMAINDER_ROW_RELEASE,
        },
    )
    if len(intervals[PIOP_SPAN]) != 1:
        raise ValueError("trace must contain exactly one positive PIOP span")
    if len(intervals[BACKEND_WITNESS_PREP_SPAN]) != 1:
        raise ValueError(
            "trace must contain exactly one positive backend witness preparation span"
        )
    piop = intervals[PIOP_SPAN][0]
    backend_prepare = intervals[BACKEND_WITNESS_PREP_SPAN][0]
    by_component = {
        component: sorted(intervals[f"{INSTRUCTION_INPUT_KERNEL}::{component}"])
        for component in INSTRUCTION_INPUT_COMPONENTS
    }
    expected_outer_counts = {
        "prepare": 1,
        "prove_round": log_n,
        "finish_rounds": 1,
        "output_claims": 1,
    }
    outer_counts = {
        component: len(component_intervals)
        for component, component_intervals in by_component.items()
    }
    if outer_counts != expected_outer_counts:
        raise ValueError(
            f"InstructionInput member span counts {outer_counts}, expected {expected_outer_counts}"
        )

    prepare = by_component["prepare"][0]
    rounds = by_component["prove_round"]
    finish = by_component["finish_rounds"][0]
    output = by_component["output_claims"][0]
    ordered = [prepare, *rounds, finish, output]
    if any(start < piop[0] or end > piop[1] for start, end in ordered):
        raise ValueError("an InstructionInput member span lies outside PIOP")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("InstructionInput member spans overlap or appear out of order")

    inner = {
        phase: sorted(intervals[f"Metal{INSTRUCTION_INPUT_KERNEL}::{phase}"])
        for phase in INSTRUCTION_INPUT_METAL_PHASES
    }
    inner_counts = {phase: len(values) for phase, values in inner.items()}
    resource_observation = None
    row_lifecycle = None
    prefetch_submit_us = 0.0
    if backend == "optimized":
        if any(inner_counts.values()):
            raise ValueError(
                "optimized trace unexpectedly contains InstructionInput Metal spans"
            )
        if (
            len(intervals[OPTIMIZED_INSTRUCTION_INPUT_ROWS_PREPARE]) != 1
            or len(intervals[OPTIMIZED_INSTRUCTION_INPUT_ROWS_STAGE3_USE]) != 1
            or intervals[METAL_INSTRUCTION_INPUT_ROWS_PREPARE]
            or intervals[METAL_INSTRUCTION_INPUT_ROWS_STAGE1_HANDOFF]
        ):
            raise ValueError("optimized InstructionInput row lifecycle is incomplete")
        rows_prepare = intervals[OPTIMIZED_INSTRUCTION_INPUT_ROWS_PREPARE][0]
        rows_stage3 = intervals[OPTIMIZED_INSTRUCTION_INPUT_ROWS_STAGE3_USE][0]
        if (
            rows_prepare[0] < backend_prepare[0]
            or rows_prepare[1] > backend_prepare[1]
            or rows_prepare[1] > piop[0]
        ):
            raise ValueError(
                "optimized InstructionInput rows were not prepared before PIOP"
            )
        require_contained(
            rows_stage3, prepare, "optimized InstructionInput prepared-row use"
        )
        prepare_args = unique_span_args(
            events, OPTIMIZED_INSTRUCTION_INPUT_ROWS_PREPARE
        )
        stage3_args = unique_span_args(
            events, OPTIMIZED_INSTRUCTION_INPUT_ROWS_STAGE3_USE
        )
        if set(prepare_args) != {
            "cpu_rows_storage_id",
            "cpu_rows",
            "cpu_row_bytes",
        } or set(stage3_args) != {"cpu_rows_storage_id", "cpu_rows"}:
            raise ValueError("optimized InstructionInput row lifecycle is incomplete")
        prepare_storage_id = positive_trace_integer(
            prepare_args["cpu_rows_storage_id"], "CPU prepare storage ID"
        )
        stage3_storage_id = positive_trace_integer(
            stage3_args["cpu_rows_storage_id"], "CPU stage-3 storage ID"
        )
        cpu_rows = positive_trace_integer(prepare_args["cpu_rows"], "CPU row count")
        stage3_rows = positive_trace_integer(
            stage3_args["cpu_rows"], "CPU stage-3 row count"
        )
        cpu_row_bytes = positive_trace_integer(
            prepare_args["cpu_row_bytes"], "CPU row width"
        )
        if (
            cpu_rows != 1 << log_n
            or stage3_rows != cpu_rows
            or cpu_row_bytes != 48
            or stage3_storage_id != prepare_storage_id
        ):
            raise ValueError("optimized InstructionInput row lifecycle is invalid")
        row_lifecycle = {
            "kind": "optimized_cpu",
            "rows": cpu_rows,
            "row_bytes": cpu_row_bytes,
            "prepare_storage_id": prepare_storage_id,
            "stage3_storage_id": stage3_storage_id,
        }
    else:
        if (
            intervals[OPTIMIZED_INSTRUCTION_INPUT_ROWS_PREPARE]
            or intervals[OPTIMIZED_INSTRUCTION_INPUT_ROWS_STAGE3_USE]
            or len(intervals[METAL_INSTRUCTION_INPUT_ROWS_PREPARE]) != 1
            or len(intervals[METAL_INSTRUCTION_INPUT_ROWS_STAGE1_HANDOFF]) != 1
        ):
            raise ValueError("Metal InstructionInput row lifecycle is incomplete")
        expected_inner_counts = {
            "storage_prepare": 1,
            "allocation_plan": 1,
            "storage_initialize": 1,
            "storage_initialize_complete": 1,
            "outer_residual_transfer": 1 if borrow_outer_residual else 0,
            "native_primer_submit": 1,
            "prepare": 1,
            "native_primer_join": 1,
            "native_primer_complete": 1,
            "first_message": 1,
            "first_bind": 1,
            "dense_round": log_n - cutoff_log2 - 1,
            "readback": 1,
            "cpu_tail": cutoff_log2,
        }
        if inner_counts != expected_inner_counts:
            raise ValueError(
                f"InstructionInput Metal span counts {inner_counts}, expected {expected_inner_counts}"
            )
        storage_prepare = inner["storage_prepare"][0]
        allocation_plan = inner["allocation_plan"][0]
        storage_initialize = inner["storage_initialize"][0]
        storage_initialize_complete = inner["storage_initialize_complete"][0]
        outer_residual_transfer = (
            inner["outer_residual_transfer"][0]
            if borrow_outer_residual
            else None
        )
        primer_submit = inner["native_primer_submit"][0]
        primer_submit_us = interval_duration_us(primer_submit)
        primer_join = inner["native_primer_join"][0]
        primer_complete = inner["native_primer_complete"][0]
        rows_prepare = intervals[METAL_INSTRUCTION_INPUT_ROWS_PREPARE][0]
        rows_stage1_handoff = intervals[
            METAL_INSTRUCTION_INPUT_ROWS_STAGE1_HANDOFF
        ][0]
        shift_prepares = intervals[SPARTAN_SHIFT_PREPARE_SPAN]
        if len(shift_prepares) != 1:
            raise ValueError("Metal trace must contain exactly one SpartanShift prepare span")
        shift_prepare = shift_prepares[0]
        if (
            storage_prepare[0] < backend_prepare[0]
            or storage_prepare[1] > backend_prepare[1]
            or storage_prepare[1] > piop[0]
        ):
            raise ValueError(
                "InstructionInput storage is not contained in backend witness preparation"
            )
        require_contained(
            allocation_plan, storage_prepare, "InstructionInput allocation plan"
        )
        require_contained(
            storage_initialize,
            allocation_plan,
            "InstructionInput minimal storage initialization",
        )
        require_contained(
            storage_initialize_complete,
            storage_initialize,
            "InstructionInput storage initialization completion",
        )
        if (
            rows_prepare[0] < backend_prepare[0]
            or rows_prepare[1] > backend_prepare[1]
            or rows_prepare[1] > storage_prepare[0]
        ):
            raise ValueError(
                "Metal InstructionInput compact rows were not directly prepared before PIOP"
            )
        if (
            rows_stage1_handoff[0] < piop[0]
            or rows_stage1_handoff[1] > prepare[0]
        ):
            raise ValueError(
                "Metal InstructionInput stage-1 compact handoff is outside its lifecycle"
            )
        require_contained(primer_submit, piop, "InstructionInput native primer submit")
        require_contained(shift_prepare, piop, "SpartanShift preparation")
        if (
            rows_stage1_handoff[1] > primer_submit[0]
            or primer_submit[1] > shift_prepare[0]
            or shift_prepare[1] > prepare[0]
        ):
            raise ValueError(
                "InstructionInput native primer was not submitted before stage-3 Shift preparation"
            )
        if outer_residual_transfer is not None:
            outer_releases = intervals[METAL_OUTER_REMAINDER_ROW_RELEASE]
            if len(outer_releases) != 1:
                raise ValueError(
                    "borrowed InstructionInput requires one completed Outer row release"
                )
            outer_release = outer_releases[0]
            require_contained(
                outer_residual_transfer,
                piop,
                "InstructionInput Outer residual transfer",
            )
            if (
                outer_release[1] > outer_residual_transfer[0]
                or outer_residual_transfer[1] > primer_submit[0]
            ):
                raise ValueError(
                    "InstructionInput Outer residual transfer is outside its ownership seam"
                )
        require_contained(inner["prepare"][0], prepare, "Metal InstructionInput prepare")
        require_contained(primer_join, rounds[0], "InstructionInput native primer join")
        require_contained(
            primer_complete, rounds[0], "InstructionInput native primer completion"
        )
        if (
            primer_join[1] > primer_complete[0]
            or primer_complete[1] > inner["first_message"][0][0]
        ):
            raise ValueError(
                "InstructionInput native primer did not complete before the first message"
            )
        require_contained(
            inner["first_message"][0], rounds[0], "first InstructionInput Metal message"
        )
        require_contained(
            inner["first_bind"][0], rounds[1], "first InstructionInput Metal bind"
        )
        for index, interval in enumerate(inner["dense_round"]):
            require_contained(interval, rounds[index + 2], "dense InstructionInput round")
        handoff_round = 2 + expected_inner_counts["dense_round"]
        require_contained(
            inner["readback"][0], rounds[handoff_round], "InstructionInput readback"
        )
        if inner["readback"][0][1] > inner["cpu_tail"][0][0]:
            raise ValueError("InstructionInput readback overlaps CPU-tail algebra")
        for interval, outer_round in zip(
            inner["cpu_tail"][:-1], rounds[handoff_round:]
        ):
            require_contained(interval, outer_round, "InstructionInput CPU-tail round")
        require_contained(
            inner["cpu_tail"][-1], finish, "InstructionInput CPU-tail finish"
        )

        storage = unique_span_args(
            events, f"Metal{INSTRUCTION_INPUT_KERNEL}::storage_prepare"
        )
        if set(storage) != {
            "trace_elements",
            "cutoff_elements",
            "dense_storage_mode",
            "host_tail_bytes",
            "resident_rows_storage_id",
            "resident_rows",
            "resident_row_bytes",
        }:
            raise ValueError("InstructionInput storage preparation has unexpected fields")
        dense_storage_mode = trace_string(
            storage.pop("dense_storage_mode"), "InstructionInput dense storage mode"
        )
        storage = {
            name: positive_trace_integer(value, name)
            for name, value in storage.items()
        }
        host_tail_bytes = 8 * (1 << cutoff_log2) * 16
        if (
            storage["trace_elements"] != 1 << log_n
            or storage["cutoff_elements"] != 1 << cutoff_log2
            or storage["host_tail_bytes"] != host_tail_bytes
            or storage["resident_rows"] != 1 << log_n
            or storage["resident_row_bytes"] != 48
            or dense_storage_mode
            != ("OuterResidual" if borrow_outer_residual else "Owned")
        ):
            raise ValueError("InstructionInput storage preparation has invalid geometry")

        allocation = unique_span_args(
            events, f"Metal{INSTRUCTION_INPUT_KERNEL}::allocation_plan"
        )
        allocation_fields = {
            "device_buffers",
            "planned_device_bytes",
            "owned_device_bytes",
            "reused_device_bytes",
            "borrowed_outer_residual",
            "current_device_bytes",
            "recommended_device_bytes",
        }
        if set(allocation) != allocation_fields:
            raise ValueError("InstructionInput allocation plan has unexpected fields")
        allocation_borrowed = trace_boolean(
            allocation.pop("borrowed_outer_residual")
        )
        allocation = {
            name: nonnegative_trace_integer(value, name)
            for name, value in allocation.items()
        }
        expected_sequence_bytes = instruction_input_sequence_storage_bytes(log_n)
        expected_owned_bytes = (
            instruction_input_sequence_auxiliary_storage_bytes(log_n)
            if borrow_outer_residual
            else expected_sequence_bytes
        )
        expected_reused_bytes = expected_sequence_bytes - expected_owned_bytes
        expected_resident_row_bytes = 160 * (1 << log_n)
        if (
            allocation["device_buffers"] != (4 if borrow_outer_residual else 6)
            or allocation["planned_device_bytes"] != expected_sequence_bytes
            or allocation["owned_device_bytes"] != expected_owned_bytes
            or allocation["reused_device_bytes"] != expected_reused_bytes
            or allocation_borrowed is not borrow_outer_residual
            or allocation["current_device_bytes"] < expected_resident_row_bytes
        ):
            raise ValueError(
                "InstructionInput allocation plan has invalid buffer accounting"
            )
        if (
            allocation["current_device_bytes"] + allocation["owned_device_bytes"]
            > allocation["recommended_device_bytes"]
        ):
            raise ValueError(
                "InstructionInput allocation plan exceeds the admitted working set"
            )

        storage_initialization_args = unique_span_args(
            events, f"Metal{INSTRUCTION_INPUT_KERNEL}::storage_initialize"
        )
        initialization_fields = {
            "mode",
            "device_buffers",
            "bytes",
            "protocol_dispatches",
            *(f"buffer_{index}" for index in range(6)),
        }
        if set(storage_initialization_args) != initialization_fields:
            raise ValueError(
                "InstructionInput storage initialization has unexpected fields"
            )
        initialization_buffer_ids = [
            positive_trace_integer(
                storage_initialization_args[f"buffer_{index}"],
                f"storage initialization buffer {index}",
            )
            for index in range(6)
        ]
        storage_initialization = {
            "mode": trace_string(
                storage_initialization_args["mode"], "storage initialization mode"
            ),
            "device_buffers": nonnegative_trace_integer(
                storage_initialization_args["device_buffers"],
                "storage initialization device buffers",
            ),
            "bytes": nonnegative_trace_integer(
                storage_initialization_args["bytes"],
                "storage initialization bytes",
            ),
            "protocol_dispatches": nonnegative_trace_integer(
                storage_initialization_args["protocol_dispatches"],
                "storage initialization protocol dispatches",
            ),
            "buffer_identities": initialization_buffer_ids,
        }
        if (
            storage_initialization["mode"] != "minimal"
            or storage_initialization["device_buffers"]
            != (4 if borrow_outer_residual else 6)
            or storage_initialization["bytes"]
            != (64 if borrow_outer_residual else 96)
            or storage_initialization["protocol_dispatches"] != 0
            or len(set(initialization_buffer_ids))
            != (5 if borrow_outer_residual else 6)
            or (
                borrow_outer_residual
                and initialization_buffer_ids[0] != initialization_buffer_ids[1]
            )
        ):
            raise ValueError(
                "InstructionInput storage initialization is not the exact minimal control"
            )
        storage_initialization_complete_args = unique_span_args(
            events,
            f"Metal{INSTRUCTION_INPUT_KERNEL}::storage_initialize_complete",
        )
        if set(storage_initialization_complete_args) != {
            "mode",
            "command_completed",
            "gpu_active_ns",
        }:
            raise ValueError(
                "InstructionInput storage initialization completion has unexpected fields"
            )
        initialization_gpu_active_ns = positive_trace_integer(
            storage_initialization_complete_args["gpu_active_ns"],
            "storage initialization GPU active time",
        )
        if (
            trace_string(
                storage_initialization_complete_args["mode"],
                "storage initialization completion mode",
            )
            != "minimal"
            or trace_boolean(
                storage_initialization_complete_args["command_completed"]
            )
            is not True
        ):
            raise ValueError(
                "InstructionInput minimal storage initialization did not complete"
            )
        initialization_wall_ns = round(
            interval_duration_us(storage_initialize) * 1000.0
        )
        if initialization_gpu_active_ns > initialization_wall_ns:
            raise ValueError(
                "InstructionInput storage initialization GPU time exceeds wall time"
            )
        storage_initialization["gpu_active_ns"] = initialization_gpu_active_ns
        storage_initialization["wall_ns"] = initialization_wall_ns

        primer_resource_fields = {
            "source_elements",
            "e_in_elements",
            "e_out_elements",
            "resident_rows_storage_id",
            *(f"storage_buffer_{index}" for index in range(6)),
        }

        def primer_resources(
            args: dict[str, Any], expected_extra: set[str], phase: str
        ) -> dict[str, Any]:
            if set(args) != primer_resource_fields | expected_extra:
                raise ValueError(
                    f"InstructionInput native primer {phase} has unexpected fields"
                )
            resources = {
                "source_elements": positive_trace_integer(
                    args["source_elements"], "primer source elements"
                ),
                "e_in_elements": positive_trace_integer(
                    args["e_in_elements"], "primer e_in elements"
                ),
                "e_out_elements": positive_trace_integer(
                    args["e_out_elements"], "primer e_out elements"
                ),
                "resident_rows_storage_id": positive_trace_integer(
                    args["resident_rows_storage_id"], "primer resident row identity"
                ),
                "storage_buffer_identities": [
                    positive_trace_integer(
                        args[f"storage_buffer_{index}"],
                        f"primer storage buffer {index}",
                    )
                    for index in range(6)
                ],
            }
            if (
                resources["source_elements"] != 64
                or resources["e_in_elements"] != 1
                or resources["e_out_elements"] != 32
            ):
                raise ValueError(
                    f"InstructionInput native primer {phase} has invalid geometry"
                )
            return resources

        primer_submit_args = unique_span_args(
            events, f"Metal{INSTRUCTION_INPUT_KERNEL}::native_primer_submit"
        )
        primer_submit = primer_resources(
            primer_submit_args,
            {"command_committed", "protocol_state_advanced"},
            "submit",
        )
        if (
            trace_boolean(primer_submit_args["command_committed"]) is not True
            or trace_boolean(primer_submit_args["protocol_state_advanced"])
            is not False
        ):
            raise ValueError(
                "InstructionInput native primer submission is not protocol-inert"
            )
        primer_submit["command_committed"] = True
        primer_submit["protocol_state_advanced"] = False
        primer_join_args = unique_span_args(
            events, f"Metal{INSTRUCTION_INPUT_KERNEL}::native_primer_join"
        )
        primer_join_resources = primer_resources(primer_join_args, set(), "join")
        primer_complete_args = unique_span_args(
            events, f"Metal{INSTRUCTION_INPUT_KERNEL}::native_primer_complete"
        )
        primer_complete = primer_resources(
            primer_complete_args,
            {
                "command_completed",
                "produced_zero",
                "protocol_state_advanced",
                "completed_before_join",
                "submit_wall_ns",
                "overlap_wall_ns",
                "join_wall_ns",
                "lifecycle_wall_ns",
                "gpu_active_ns",
            },
            "completion",
        )
        primer_timings = {
            "submit_wall_ns": positive_trace_integer(
                primer_complete_args["submit_wall_ns"], "primer submit wall time"
            ),
            "overlap_wall_ns": positive_trace_integer(
                primer_complete_args["overlap_wall_ns"], "primer overlap time"
            ),
            "join_wall_ns": positive_trace_integer(
                primer_complete_args["join_wall_ns"], "primer join wall time"
            ),
            "lifecycle_wall_ns": positive_trace_integer(
                primer_complete_args["lifecycle_wall_ns"], "primer lifecycle wall time"
            ),
            "gpu_active_ns": positive_trace_integer(
                primer_complete_args["gpu_active_ns"], "primer GPU active time"
            ),
            "submit_span_wall_ns": round(primer_submit_us * 1000.0),
        }
        completed_before_join = trace_boolean(
            primer_complete_args["completed_before_join"]
        )
        if (
            trace_boolean(primer_complete_args["command_completed"]) is not True
            or trace_boolean(primer_complete_args["produced_zero"]) is not True
            or trace_boolean(primer_complete_args["protocol_state_advanced"])
            is not False
            or completed_before_join is None
            or primer_timings["submit_wall_ns"]
            + primer_timings["overlap_wall_ns"]
            + primer_timings["join_wall_ns"]
            > primer_timings["lifecycle_wall_ns"]
            or primer_timings["gpu_active_ns"]
            > primer_timings["lifecycle_wall_ns"]
            or primer_timings["submit_span_wall_ns"] <= 0
            or primer_timings["submit_wall_ns"]
            > primer_timings["submit_span_wall_ns"] + 1
        ):
            raise ValueError(
                "InstructionInput native primer completion is inconsistent"
            )
        primer_submit["timings"] = primer_timings
        primer_submit["completed_before_join"] = completed_before_join
        primer_submit["command_completed"] = True
        primer_submit["produced_zero"] = True
        prefetch_submit_us = primer_submit_us

        prepare_args = unique_span_args(
            events, f"Metal{INSTRUCTION_INPUT_KERNEL}::prepare"
        )
        if set(prepare_args) != {
            "resident_rows_reused",
            "round_device_buffer_allocations",
            "resident_rows_storage_id",
            "resident_rows",
            "storage_initialization",
            "storage_initialization_bytes",
            "native_primer",
            "dense_a_offset_bytes",
            "dense_a_length_bytes",
            "dense_b_offset_bytes",
            "dense_b_length_bytes",
            *(f"storage_buffer_{index}" for index in range(6)),
        }:
            raise ValueError("InstructionInput Metal prepare has unexpected fields")
        resident_rows_reused = trace_boolean(prepare_args["resident_rows_reused"])
        round_allocations = nonnegative_trace_integer(
            prepare_args["round_device_buffer_allocations"],
            "round device buffer allocations",
        )
        stage3_storage_id = positive_trace_integer(
            prepare_args["resident_rows_storage_id"], "Metal stage-3 storage ID"
        )
        stage3_rows = positive_trace_integer(
            prepare_args["resident_rows"], "Metal stage-3 row count"
        )
        prepare_buffer_ids = [
            positive_trace_integer(
                prepare_args[f"storage_buffer_{index}"],
                f"Metal stage-3 storage buffer {index}",
            )
            for index in range(6)
        ]
        dense_ranges = {
            name: nonnegative_trace_integer(prepare_args[name], name)
            for name in (
                "dense_a_offset_bytes",
                "dense_a_length_bytes",
                "dense_b_offset_bytes",
                "dense_b_length_bytes",
            )
        }
        dense_a_bytes = 64 * (1 << log_n)
        dense_b_bytes = 32 * (1 << log_n)
        if (
            trace_string(
                prepare_args["storage_initialization"],
                "Metal stage-3 storage initialization mode",
            )
            != "minimal"
            or nonnegative_trace_integer(
                prepare_args["storage_initialization_bytes"],
                "Metal stage-3 storage initialization bytes",
            )
            != (64 if borrow_outer_residual else 96)
            or trace_string(
                prepare_args["native_primer"], "Metal stage-3 native primer mode"
            )
            != "async"
            or dense_ranges
            != {
                "dense_a_offset_bytes": 0,
                "dense_a_length_bytes": dense_a_bytes,
                "dense_b_offset_bytes": dense_a_bytes if borrow_outer_residual else 0,
                "dense_b_length_bytes": dense_b_bytes,
            }
        ):
            raise ValueError(
                "InstructionInput Metal prepare did not preserve the selected startup controls"
            )
        row_production_args = unique_span_args(
            events, METAL_INSTRUCTION_INPUT_ROWS_PREPARE
        )
        if set(row_production_args) != {
            "source_kind",
            "witness_row_extractions",
            "residual_rows_written",
            "compact_rows_written",
            "compact_row_bytes",
            "residual_row_bytes",
            "compact_allocations",
            "residual_allocations",
            "full_row_allocations",
            "full_domain_copy_bytes",
            "full_domain_copy_dispatches",
            "host_repack_rows",
            "compact_rows_storage_id",
            "residual_rows_storage_id",
            "resident_rows",
            "explicit_rows",
        }:
            raise ValueError("Metal InstructionInput row lifecycle is incomplete")
        row_count = 1 << log_n
        production_storage_id = positive_trace_integer(
            row_production_args["compact_rows_storage_id"],
            "Metal compact-row production storage ID",
        )
        residual_storage_id = positive_trace_integer(
            row_production_args["residual_rows_storage_id"],
            "Metal residual-row production storage ID",
        )
        row_production = {
            "source_kind": trace_string(
                row_production_args["source_kind"],
                "Metal witness-row production source",
            ),
            "witness_row_extractions": positive_trace_integer(
                row_production_args["witness_row_extractions"],
                "Metal witness row extractions",
            ),
            "residual_rows_written": positive_trace_integer(
                row_production_args["residual_rows_written"],
                "Metal residual rows written",
            ),
            "compact_rows_written": positive_trace_integer(
                row_production_args["compact_rows_written"],
                "Metal compact rows written",
            ),
            "compact_row_bytes": positive_trace_integer(
                row_production_args["compact_row_bytes"],
                "Metal compact row width",
            ),
            "residual_row_bytes": positive_trace_integer(
                row_production_args["residual_row_bytes"],
                "Metal residual row width",
            ),
            "compact_allocations": positive_trace_integer(
                row_production_args["compact_allocations"],
                "Metal compact row allocations",
            ),
            "residual_allocations": positive_trace_integer(
                row_production_args["residual_allocations"],
                "Metal residual row allocations",
            ),
            "full_row_allocations": nonnegative_trace_integer(
                row_production_args["full_row_allocations"],
                "Metal full-row allocations",
            ),
            "full_domain_copy_bytes": nonnegative_trace_integer(
                row_production_args["full_domain_copy_bytes"],
                "Metal compact full-domain copy bytes",
            ),
            "full_domain_copy_dispatches": nonnegative_trace_integer(
                row_production_args["full_domain_copy_dispatches"],
                "Metal compact full-domain copy dispatches",
            ),
            "host_repack_rows": nonnegative_trace_integer(
                row_production_args["host_repack_rows"],
                "Metal compact host repack rows",
            ),
        }
        production_rows = positive_trace_integer(
            row_production_args["resident_rows"], "Metal production row count"
        )
        production_explicit_rows = nonnegative_trace_integer(
            row_production_args["explicit_rows"], "Metal production explicit row count"
        )
        stage1_args = unique_span_args(
            events, METAL_INSTRUCTION_INPUT_ROWS_STAGE1_HANDOFF
        )
        if set(stage1_args) != {
            "compact_rows_storage_id",
            "residual_rows_storage_id",
            "resident_rows",
            "explicit_rows",
            "compact_row_bytes",
            "residual_row_bytes",
            "full_domain_copy_bytes",
            "full_domain_copy_dispatches",
            "host_repack_rows",
        }:
            raise ValueError("Metal InstructionInput stage-1 handoff is incomplete")
        stage1_storage_id = positive_trace_integer(
            stage1_args["compact_rows_storage_id"], "Metal stage-1 compact storage ID"
        )
        stage1_residual_storage_id = positive_trace_integer(
            stage1_args["residual_rows_storage_id"],
            "Metal stage-1 residual storage ID",
        )
        stage1_rows = positive_trace_integer(
            stage1_args["resident_rows"], "Metal stage-1 compact row count"
        )
        stage1_explicit_rows = nonnegative_trace_integer(
            stage1_args["explicit_rows"], "Metal stage-1 explicit row count"
        )
        stage1_row_bytes = positive_trace_integer(
            stage1_args["compact_row_bytes"], "Metal stage-1 compact row width"
        )
        stage1_residual_row_bytes = positive_trace_integer(
            stage1_args["residual_row_bytes"],
            "Metal stage-1 residual row width",
        )
        stage1_copy_bytes = nonnegative_trace_integer(
            stage1_args["full_domain_copy_bytes"],
            "Metal stage-1 full-domain copy bytes",
        )
        stage1_copy_dispatches = nonnegative_trace_integer(
            stage1_args["full_domain_copy_dispatches"],
            "Metal stage-1 full-domain copy dispatches",
        )
        stage1_host_repack_rows = nonnegative_trace_integer(
            stage1_args["host_repack_rows"], "Metal stage-1 host repack rows"
        )
        prepare_storage_id = storage["resident_rows_storage_id"]
        outer_transfer = None
        if outer_residual_transfer is not None:
            transfer_args = unique_span_args(
                events, f"Metal{INSTRUCTION_INPUT_KERNEL}::outer_residual_transfer"
            )
            if set(transfer_args) != {
                "resident_rows",
                "outer_residual_generation",
                "compact_rows_storage_id",
                "residual_rows_storage_id",
                "device_registry_id",
                "outer_sequence_owned_bytes",
                "outer_sequence_consumed",
                "compact_rows_transferred",
                "residual_rows_transferred",
            }:
                raise ValueError(
                    "InstructionInput Outer residual transfer has unexpected fields"
                )
            outer_transfer = {
                "resident_rows": positive_trace_integer(
                    transfer_args["resident_rows"], "Outer transfer row count"
                ),
                "outer_residual_generation": positive_trace_integer(
                    transfer_args["outer_residual_generation"],
                    "Outer residual generation",
                ),
                "compact_rows_storage_id": positive_trace_integer(
                    transfer_args["compact_rows_storage_id"],
                    "Outer transfer compact storage ID",
                ),
                "residual_rows_storage_id": positive_trace_integer(
                    transfer_args["residual_rows_storage_id"],
                    "Outer transfer residual storage ID",
                ),
                "device_registry_id": positive_trace_integer(
                    transfer_args["device_registry_id"],
                    "Outer transfer device registry ID",
                ),
                "outer_sequence_owned_bytes": positive_trace_integer(
                    transfer_args["outer_sequence_owned_bytes"],
                    "Outer transfer sequence-owned bytes",
                ),
                "outer_sequence_consumed": trace_boolean(
                    transfer_args["outer_sequence_consumed"]
                ),
                "compact_rows_transferred": trace_boolean(
                    transfer_args["compact_rows_transferred"]
                ),
                "residual_rows_transferred": trace_boolean(
                    transfer_args["residual_rows_transferred"]
                ),
            }
            outer_release_args = unique_span_args(
                events, METAL_OUTER_REMAINDER_ROW_RELEASE
            )
            if (
                outer_transfer["resident_rows"] != row_count
                or outer_transfer["compact_rows_storage_id"]
                != production_storage_id
                or outer_transfer["residual_rows_storage_id"]
                != residual_storage_id
                or outer_transfer["device_registry_id"]
                != positive_trace_integer(
                    outer_release_args["device_registry_id"],
                    "Outer release device registry ID",
                )
                or outer_transfer["outer_sequence_owned_bytes"]
                != positive_trace_integer(
                    outer_release_args["remaining_sequence_storage_bytes"],
                    "Outer release sequence-owned bytes",
                )
                or outer_transfer["outer_sequence_consumed"] is not True
                or outer_transfer["compact_rows_transferred"] is not True
                or outer_transfer["residual_rows_transferred"] is not True
            ):
                raise ValueError(
                    "InstructionInput Outer residual ownership transfer is incomplete"
                )
        primer_resource_records = [
            primer_submit,
            primer_join_resources,
            primer_complete,
        ]
        if (
            resident_rows_reused is not True
            or round_allocations != 0
            or production_rows != row_count
            or stage1_rows != row_count
            or production_explicit_rows > row_count
            or stage1_explicit_rows != production_explicit_rows
            or stage3_rows != stage1_rows
            or row_production
            != {
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
            or stage1_row_bytes != 48
            or stage1_residual_row_bytes != 112
            or stage1_copy_bytes != 0
            or stage1_copy_dispatches != 0
            or stage1_host_repack_rows != 0
            or residual_storage_id == production_storage_id
            or stage1_residual_storage_id != residual_storage_id
            or production_storage_id != prepare_storage_id
            or stage1_storage_id != prepare_storage_id
            or stage3_storage_id != prepare_storage_id
            or storage["resident_row_bytes"] != 48
            or (
                borrow_outer_residual
                and (
                    initialization_buffer_ids[0] != residual_storage_id
                    or initialization_buffer_ids[1] != residual_storage_id
                    or any(
                        buffer_id == residual_storage_id
                        for buffer_id in initialization_buffer_ids[2:]
                    )
                    or dense_ranges["dense_b_offset_bytes"]
                    + dense_ranges["dense_b_length_bytes"]
                    >= 112 * row_count
                )
            )
            or any(
                record["resident_rows_storage_id"] != prepare_storage_id
                or record["storage_buffer_identities"]
                != initialization_buffer_ids
                for record in primer_resource_records
            )
            or prepare_buffer_ids != initialization_buffer_ids
        ):
            raise ValueError(
                "InstructionInput Metal row lifecycle did not preserve residency"
            )
        row_lifecycle = {
            "kind": "metal_compact_resident",
            "rows": stage1_rows,
            "explicit_rows": stage1_explicit_rows,
            "row_bytes": 48,
            "prepare_storage_id": prepare_storage_id,
            "stage1_storage_id": stage1_storage_id,
            "stage3_storage_id": stage3_storage_id,
            "residual_storage_id": residual_storage_id,
            "row_production": row_production,
            "outer_residual_transfer": outer_transfer,
        }

        readback = unique_span_args(
            events, f"Metal{INSTRUCTION_INPUT_KERNEL}::readback"
        )
        if readback != {"bytes": str(host_tail_bytes)}:
            raise ValueError(
                "InstructionInput readback does not cover exactly eight cutoff tables"
            )
        resource_observation = {
            "allocation": allocation,
            "borrowed_outer_residual": allocation_borrowed,
            "dense_ranges": dense_ranges,
            "outer_residual_transfer": outer_transfer,
            "storage_initialization": storage_initialization,
            "native_primer": primer_submit,
            "host_tail_bytes": storage["host_tail_bytes"],
            "readback_bytes": int(readback["bytes"]),
            "resident_rows_reused": resident_rows_reused,
            "round_device_buffer_allocations": round_allocations,
        }

    round_durations = [interval_duration_us(interval) for interval in rounds]
    components = {
        "prepare_us": interval_duration_us(prepare),
        "rounds_us": round_durations,
        "rounds_total_us": sum(round_durations),
        "finish_us": interval_duration_us(finish),
        "output_claims_us": interval_duration_us(output),
    }
    components["member_us"] = (
        components["prepare_us"]
        + components["rounds_total_us"]
        + components["finish_us"]
        + components["output_claims_us"]
    )
    scalar_components = [value for value in components.values() if not isinstance(value, list)]
    if any(not math.isfinite(value) or value <= 0.0 for value in scalar_components):
        raise ValueError("trace contains a non-positive InstructionInput member duration")
    components["prefetch_submit_us"] = prefetch_submit_us
    components["service_us"] = components["member_us"] + prefetch_submit_us
    return {
        "components": components,
        "outer_counts": outer_counts,
        "metal_counts": inner_counts,
        "resource_observation": resource_observation,
        "row_lifecycle": row_lifecycle,
    }


def local_member_decision(
    pairs: list[dict[str, Any]],
    cpu: list[float],
    metal: list[float],
    minimum_speedup: float,
) -> tuple[list[float], list[float], dict[str, Any]]:
    speedups = [cpu_us / metal_us for cpu_us, metal_us in zip(cpu, metal)]
    improvements = [1.0 - metal_us / cpu_us for cpu_us, metal_us in zip(cpu, metal)]
    speedup_median = statistics.median(speedups)
    improvement_median = statistics.median(improvements)
    improvement_mad = statistics.median(
        abs(value - improvement_median) for value in improvements
    )
    cpu_median = statistics.median(cpu)
    metal_median = statistics.median(metal)
    optimized_first = [
        speedup
        for pair, speedup in zip(pairs, speedups)
        if pair.get("order") == ["optimized", "metal"]
    ]
    metal_first = [
        speedup
        for pair, speedup in zip(pairs, speedups)
        if pair.get("order") == ["metal", "optimized"]
    ]
    optimized_first_median = (
        statistics.median(optimized_first) if optimized_first else None
    )
    metal_first_median = statistics.median(metal_first) if metal_first else None
    enough_pairs = len(pairs) >= PRODUCTION_PAIRS
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
    decision = {
        "minimum_speedup": minimum_speedup,
        "minimum_pairs": PRODUCTION_PAIRS,
        "median_speedup": speedup_median,
        "median_fractional_improvement": improvement_median,
        "mad_fractional_improvement": improvement_mad,
        "cpu_member_ms_median": cpu_median / 1000.0,
        "cpu_member_ms_mad": statistics.median(
            abs(value - cpu_median) for value in cpu
        )
        / 1000.0,
        "metal_member_ms_median": metal_median / 1000.0,
        "metal_member_ms_mad": statistics.median(
            abs(value - metal_median) for value in metal
        )
        / 1000.0,
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
    return speedups, improvements, decision


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("at least one CPU/Metal pair is required")
    hamming_fields = {
        "cpu_hamming_weight_us",
        "metal_hamming_weight_us",
        "cpu_hamming_weight_service_us",
        "metal_hamming_weight_service_us",
    }
    for index, pair in enumerate(pairs, 1):
        missing = hamming_fields - pair.keys()
        if missing:
            raise ValueError(
                f"pair {index} is missing Hamming-weight timing fields: {sorted(missing)}"
            )
    cpu = [float(pair["cpu_us"]) for pair in pairs]
    metal = [float(pair["metal_us"]) for pair in pairs]
    cpu_prepare = [float(pair["cpu_prepare_us"]) for pair in pairs]
    metal_prepare = [float(pair["metal_prepare_us"]) for pair in pairs]
    cpu_instruction_ra = [float(pair["cpu_instruction_ra_us"]) for pair in pairs]
    metal_instruction_ra = [float(pair["metal_instruction_ra_us"]) for pair in pairs]
    cpu_bytecode = [float(pair["cpu_bytecode_us"]) for pair in pairs]
    metal_bytecode = [float(pair["metal_bytecode_us"]) for pair in pairs]
    cpu_instruction_input = [float(pair["cpu_instruction_input_us"]) for pair in pairs]
    metal_instruction_input = [
        float(pair["metal_instruction_input_us"]) for pair in pairs
    ]
    cpu_instruction_read_raf = [
        float(pair.get("cpu_instruction_read_raf_us", pair["cpu_hamming_weight_us"]))
        for pair in pairs
    ]
    metal_instruction_read_raf = [
        float(pair.get("metal_instruction_read_raf_us", pair["metal_hamming_weight_us"]))
        for pair in pairs
    ]
    cpu_booleanity_address = [
        float(pair["cpu_booleanity_address_us"]) for pair in pairs
    ]
    metal_booleanity_address = [
        float(pair["metal_booleanity_address_us"]) for pair in pairs
    ]
    cpu_booleanity_address_service = [
        float(
            pair.get(
                "cpu_booleanity_address_service_us",
                pair["cpu_booleanity_address_us"],
            )
        )
        for pair in pairs
    ]
    metal_booleanity_address_service = [
        float(
            pair.get(
                "metal_booleanity_address_service_us",
                pair["metal_booleanity_address_us"],
            )
        )
        for pair in pairs
    ]
    cpu_hamming_weight = [
        float(pair["cpu_hamming_weight_us"]) for pair in pairs
    ]
    metal_hamming_weight = [
        float(pair["metal_hamming_weight_us"]) for pair in pairs
    ]
    cpu_hamming_weight_service = [
        float(pair["cpu_hamming_weight_service_us"]) for pair in pairs
    ]
    metal_hamming_weight_service = [
        float(pair["metal_hamming_weight_service_us"]) for pair in pairs
    ]
    cpu_outer_remainder = [
        float(pair.get("cpu_outer_remainder_us", pair["cpu_hamming_weight_us"]))
        for pair in pairs
    ]
    metal_outer_remainder = [
        float(pair.get("metal_outer_remainder_us", pair["metal_hamming_weight_us"]))
        for pair in pairs
    ]
    cpu_product_uniskip = [
        float(
            pair.get(
                "cpu_product_uniskip_us",
                pair.get("cpu_outer_remainder_us", pair["cpu_hamming_weight_us"]),
            )
        )
        for pair in pairs
    ]
    metal_product_uniskip = [
        float(
            pair.get(
                "metal_product_uniskip_us",
                pair.get("metal_outer_remainder_us", pair["metal_hamming_weight_us"]),
            )
        )
        for pair in pairs
    ]
    cpu_product_remainder = [
        float(
            pair.get(
                "cpu_product_remainder_us",
                pair.get("cpu_outer_remainder_us", pair["cpu_hamming_weight_us"]),
            )
        )
        for pair in pairs
    ]
    metal_product_remainder = [
        float(
            pair.get(
                "metal_product_remainder_us",
                pair.get("metal_outer_remainder_us", pair["metal_hamming_weight_us"]),
            )
        )
        for pair in pairs
    ]
    cpu_outer_product_family = [
        outer + uniskip + product
        for outer, uniskip, product in zip(
            cpu_outer_remainder,
            cpu_product_uniskip,
            cpu_product_remainder,
        )
    ]
    metal_outer_product_family = [
        outer + uniskip + product
        for outer, uniskip, product in zip(
            metal_outer_remainder,
            metal_product_uniskip,
            metal_product_remainder,
        )
    ]
    cpu_instruction_claim = [
        float(pair.get("cpu_instruction_claim_us", pair["cpu_hamming_weight_us"]))
        for pair in pairs
    ]
    metal_instruction_claim = [
        float(pair.get("metal_instruction_claim_us", pair["metal_hamming_weight_us"]))
        for pair in pairs
    ]
    metal_instruction_claim_isolated_service = [
        float(
            pair.get(
                "metal_instruction_claim_isolated_service_us",
                pair.get("metal_instruction_claim_us", pair["metal_hamming_weight_us"]),
            )
        )
        for pair in pairs
    ]
    cpu_product_instruction_claim = [
        product + instruction
        for product, instruction in zip(
            cpu_product_remainder, cpu_instruction_claim
        )
    ]
    metal_product_instruction_claim = [
        product + instruction
        for product, instruction in zip(
            metal_product_remainder, metal_instruction_claim
        )
    ]
    if any(not math.isfinite(value) or value <= 0.0 for value in cpu + metal):
        raise ValueError("PIOP durations must be finite and positive")
    if any(not math.isfinite(value) or value < 0.0 for value in cpu_prepare + metal_prepare):
        raise ValueError("backend witness preparation durations must be finite and non-negative")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in cpu_instruction_ra
        + metal_instruction_ra
        + cpu_bytecode
        + metal_bytecode
        + cpu_instruction_input
        + metal_instruction_input
        + cpu_instruction_read_raf
        + metal_instruction_read_raf
        + cpu_booleanity_address
        + metal_booleanity_address
        + cpu_booleanity_address_service
        + metal_booleanity_address_service
        + cpu_hamming_weight
        + metal_hamming_weight
        + cpu_hamming_weight_service
        + metal_hamming_weight_service
        + cpu_outer_remainder
        + metal_outer_remainder
        + cpu_product_uniskip
        + metal_product_uniskip
        + cpu_product_remainder
        + metal_product_remainder
        + cpu_outer_product_family
        + metal_outer_product_family
        + cpu_instruction_claim
        + metal_instruction_claim
        + metal_instruction_claim_isolated_service
        + cpu_product_instruction_claim
        + metal_product_instruction_claim
    ):
        raise ValueError("kernel durations must be finite and positive")
    paired_speedups = [cpu_us / metal_us for cpu_us, metal_us in zip(cpu, metal)]
    instruction_ra_speedups = [
        cpu_us / metal_us for cpu_us, metal_us in zip(cpu_instruction_ra, metal_instruction_ra)
    ]
    bytecode_speedups = [
        cpu_us / metal_us for cpu_us, metal_us in zip(cpu_bytecode, metal_bytecode)
    ]
    bytecode_improvements = [
        1.0 - metal_us / cpu_us
        for cpu_us, metal_us in zip(cpu_bytecode, metal_bytecode)
    ]
    instruction_input_speedups, instruction_input_improvements, instruction_input_decision = (
        local_member_decision(
            pairs,
            cpu_instruction_input,
            metal_instruction_input,
            INSTRUCTION_INPUT_MIN_SPEEDUP,
        )
    )
    (
        instruction_read_raf_speedups,
        instruction_read_raf_improvements,
        instruction_read_raf_decision,
    ) = local_member_decision(
        pairs,
        cpu_instruction_read_raf,
        metal_instruction_read_raf,
        INSTRUCTION_READ_RAF_MIN_SPEEDUP,
    )
    (
        booleanity_address_speedups,
        booleanity_address_improvements,
        booleanity_address_decision,
    ) = local_member_decision(
        pairs,
        cpu_booleanity_address,
        metal_booleanity_address,
        BOOLEANITY_ADDRESS_MIN_SPEEDUP,
    )
    booleanity_address_service_speedups = [
        cpu_us / metal_us
        for cpu_us, metal_us in zip(
            cpu_booleanity_address_service, metal_booleanity_address_service
        )
    ]
    hamming_weight_speedups, hamming_weight_improvements, hamming_weight_decision = (
        local_member_decision(
            pairs,
            cpu_hamming_weight,
            metal_hamming_weight,
            HAMMING_WEIGHT_MIN_SPEEDUP,
        )
    )
    hamming_weight_service_speedups = [
        cpu_us / metal_us
        for cpu_us, metal_us in zip(
            cpu_hamming_weight_service, metal_hamming_weight_service
        )
    ]
    cpu_booleanity_hamming = [
        booleanity + hamming
        for booleanity, hamming in zip(
            cpu_booleanity_address, cpu_hamming_weight
        )
    ]
    metal_booleanity_hamming = [
        booleanity + hamming
        for booleanity, hamming in zip(
            metal_booleanity_address, metal_hamming_weight
        )
    ]
    (
        booleanity_hamming_speedups,
        booleanity_hamming_improvements,
        booleanity_hamming_decision,
    ) = local_member_decision(
        pairs,
        cpu_booleanity_hamming,
        metal_booleanity_hamming,
        BOOLEANITY_HAMMING_MIN_SPEEDUP,
    )
    (
        outer_remainder_speedups,
        outer_remainder_improvements,
        outer_remainder_decision,
    ) = local_member_decision(
        pairs,
        cpu_outer_remainder,
        metal_outer_remainder,
        OUTER_REMAINDER_MIN_SPEEDUP,
    )
    (
        product_remainder_speedups,
        product_remainder_improvements,
        product_remainder_decision,
    ) = local_member_decision(
        pairs,
        cpu_product_remainder,
        metal_product_remainder,
        PRODUCT_REMAINDER_MIN_SPEEDUP,
    )
    (
        outer_product_family_speedups,
        outer_product_family_improvements,
        outer_product_family_decision,
    ) = local_member_decision(
        pairs,
        cpu_outer_product_family,
        metal_outer_product_family,
        OUTER_PRODUCT_FAMILY_MIN_SPEEDUP,
    )
    (
        instruction_claim_speedups,
        instruction_claim_improvements,
        instruction_claim_decision,
    ) = local_member_decision(
        pairs,
        cpu_instruction_claim,
        metal_instruction_claim,
        INSTRUCTION_CLAIM_MIN_SPEEDUP,
    )
    instruction_claim_isolated_service_speedups = [
        cpu_us / metal_us
        for cpu_us, metal_us in zip(
            cpu_instruction_claim, metal_instruction_claim_isolated_service
        )
    ]
    (
        product_instruction_claim_speedups,
        product_instruction_claim_improvements,
        product_instruction_claim_decision,
    ) = local_member_decision(
        pairs,
        cpu_product_instruction_claim,
        metal_product_instruction_claim,
        INSTRUCTION_CLAIM_MIN_SPEEDUP,
    )
    paired_with_prepare = [
        (cpu_us + cpu_prepare_us) / (metal_us + metal_prepare_us)
        for cpu_us, metal_us, cpu_prepare_us, metal_prepare_us in zip(
            cpu, metal, cpu_prepare, metal_prepare
        )
    ]
    bytecode_speedup_median = statistics.median(bytecode_speedups)
    bytecode_improvement_median = statistics.median(bytecode_improvements)
    bytecode_improvement_mad = statistics.median(
        abs(value - bytecode_improvement_median) for value in bytecode_improvements
    )
    cpu_bytecode_median = statistics.median(cpu_bytecode)
    metal_bytecode_median = statistics.median(metal_bytecode)
    cpu_bytecode_mad = statistics.median(
        abs(value - cpu_bytecode_median) for value in cpu_bytecode
    )
    metal_bytecode_mad = statistics.median(
        abs(value - metal_bytecode_median) for value in metal_bytecode
    )
    enough_pairs = len(pairs) >= PRODUCTION_PAIRS
    clears_speedup = bytecode_speedup_median >= BYTECODE_MIN_SPEEDUP
    clears_fraction = bytecode_improvement_median >= 1.0 - 1.0 / BYTECODE_MIN_SPEEDUP
    clears_noise = bytecode_improvement_median > 3.0 * bytecode_improvement_mad
    lower_median = metal_bytecode_median < cpu_bytecode_median
    optimized_first_speedups = [
        speedup
        for pair, speedup in zip(pairs, bytecode_speedups)
        if pair.get("order") == ["optimized", "metal"]
    ]
    metal_first_speedups = [
        speedup
        for pair, speedup in zip(pairs, bytecode_speedups)
        if pair.get("order") == ["metal", "optimized"]
    ]
    optimized_first_median = (
        statistics.median(optimized_first_speedups) if optimized_first_speedups else None
    )
    metal_first_median = (
        statistics.median(metal_first_speedups) if metal_first_speedups else None
    )
    clears_order_strata = (
        optimized_first_median is not None
        and metal_first_median is not None
        and optimized_first_median >= BYTECODE_MIN_SPEEDUP
        and metal_first_median >= BYTECODE_MIN_SPEEDUP
    )
    return {
        "piop_speedup": statistics.median(paired_speedups),
        "instruction_ra_speedup": statistics.median(instruction_ra_speedups),
        "bytecode_read_raf_cycle_speedup": bytecode_speedup_median,
        "instruction_input_kernel_service_speedup": statistics.median(
            instruction_input_speedups
        ),
        "instruction_read_raf_speedup": statistics.median(
            instruction_read_raf_speedups
        ),
        "booleanity_address_phase_speedup": statistics.median(
            booleanity_address_speedups
        ),
        "booleanity_address_phase_service_speedup": statistics.median(
            booleanity_address_service_speedups
        ),
        "hamming_weight_claim_reduction_speedup": statistics.median(
            hamming_weight_speedups
        ),
        "hamming_weight_claim_reduction_service_speedup": statistics.median(
            hamming_weight_service_speedups
        ),
        "booleanity_hamming_family_speedup": statistics.median(
            booleanity_hamming_speedups
        ),
        "outer_remainder_speedup": statistics.median(outer_remainder_speedups),
        "product_uniskip_speedup": statistics.median(
            cpu_us / metal_us
            for cpu_us, metal_us in zip(
                cpu_product_uniskip, metal_product_uniskip
            )
        ),
        "product_remainder_speedup": statistics.median(product_remainder_speedups),
        "outer_product_family_speedup": statistics.median(
            outer_product_family_speedups
        ),
        "instruction_claim_reduction_critical_path_speedup": statistics.median(
            instruction_claim_speedups
        ),
        "instruction_claim_reduction_isolated_service_speedup": statistics.median(
            instruction_claim_isolated_service_speedups
        ),
        "product_instruction_claim_family_speedup": statistics.median(
            product_instruction_claim_speedups
        ),
        "piop_plus_backend_witness_prepare_speedup": statistics.median(paired_with_prepare),
        "cpu_piop_ms": statistics.median(cpu) / 1000.0,
        "metal_piop_ms": statistics.median(metal) / 1000.0,
        "cpu_backend_witness_prepare_ms": statistics.median(cpu_prepare) / 1000.0,
        "metal_backend_witness_prepare_ms": statistics.median(metal_prepare) / 1000.0,
        "paired_speedups": paired_speedups,
        "paired_instruction_ra_speedups": instruction_ra_speedups,
        "paired_bytecode_read_raf_cycle_speedups": bytecode_speedups,
        "paired_bytecode_read_raf_cycle_fractional_improvements": bytecode_improvements,
        "paired_instruction_input_kernel_service_speedups": instruction_input_speedups,
        "paired_instruction_input_kernel_service_fractional_improvements": instruction_input_improvements,
        "paired_instruction_read_raf_speedups": instruction_read_raf_speedups,
        "paired_instruction_read_raf_fractional_improvements": instruction_read_raf_improvements,
        "paired_booleanity_address_phase_speedups": booleanity_address_speedups,
        "paired_booleanity_address_phase_fractional_improvements": booleanity_address_improvements,
        "paired_booleanity_address_phase_service_speedups": booleanity_address_service_speedups,
        "paired_hamming_weight_claim_reduction_speedups": hamming_weight_speedups,
        "paired_hamming_weight_claim_reduction_fractional_improvements": hamming_weight_improvements,
        "paired_hamming_weight_claim_reduction_service_speedups": hamming_weight_service_speedups,
        "paired_booleanity_hamming_family_speedups": booleanity_hamming_speedups,
        "paired_booleanity_hamming_family_fractional_improvements": (
            booleanity_hamming_improvements
        ),
        "paired_outer_remainder_speedups": outer_remainder_speedups,
        "paired_outer_remainder_fractional_improvements": outer_remainder_improvements,
        "paired_product_uniskip_speedups": [
            cpu_us / metal_us
            for cpu_us, metal_us in zip(
                cpu_product_uniskip, metal_product_uniskip
            )
        ],
        "paired_product_remainder_speedups": product_remainder_speedups,
        "paired_product_remainder_fractional_improvements": product_remainder_improvements,
        "paired_outer_product_family_speedups": outer_product_family_speedups,
        "paired_outer_product_family_fractional_improvements": (
            outer_product_family_improvements
        ),
        "paired_instruction_claim_reduction_critical_path_speedups": instruction_claim_speedups,
        "paired_instruction_claim_reduction_critical_path_fractional_improvements": (
            instruction_claim_improvements
        ),
        "paired_instruction_claim_reduction_isolated_service_speedups": (
            instruction_claim_isolated_service_speedups
        ),
        "paired_product_instruction_claim_family_speedups": product_instruction_claim_speedups,
        "paired_product_instruction_claim_family_fractional_improvements": (
            product_instruction_claim_improvements
        ),
        "paired_speedups_with_backend_witness_prepare": paired_with_prepare,
        "cpu_piop_ms_samples": [value / 1000.0 for value in cpu],
        "metal_piop_ms_samples": [value / 1000.0 for value in metal],
        "cpu_backend_witness_prepare_ms_samples": [value / 1000.0 for value in cpu_prepare],
        "metal_backend_witness_prepare_ms_samples": [value / 1000.0 for value in metal_prepare],
        "cpu_bytecode_read_raf_cycle_ms_samples": [
            value / 1000.0 for value in cpu_bytecode
        ],
        "metal_bytecode_read_raf_cycle_ms_samples": [
            value / 1000.0 for value in metal_bytecode
        ],
        "cpu_instruction_input_kernel_service_ms_samples": [
            value / 1000.0 for value in cpu_instruction_input
        ],
        "metal_instruction_input_kernel_service_ms_samples": [
            value / 1000.0 for value in metal_instruction_input
        ],
        "cpu_instruction_read_raf_ms_samples": [
            value / 1000.0 for value in cpu_instruction_read_raf
        ],
        "metal_instruction_read_raf_ms_samples": [
            value / 1000.0 for value in metal_instruction_read_raf
        ],
        "cpu_booleanity_address_phase_ms_samples": [
            value / 1000.0 for value in cpu_booleanity_address
        ],
        "metal_booleanity_address_phase_ms_samples": [
            value / 1000.0 for value in metal_booleanity_address
        ],
        "cpu_booleanity_address_phase_service_ms_samples": [
            value / 1000.0 for value in cpu_booleanity_address_service
        ],
        "metal_booleanity_address_phase_service_ms_samples": [
            value / 1000.0 for value in metal_booleanity_address_service
        ],
        "cpu_hamming_weight_claim_reduction_ms_samples": [
            value / 1000.0 for value in cpu_hamming_weight
        ],
        "metal_hamming_weight_claim_reduction_ms_samples": [
            value / 1000.0 for value in metal_hamming_weight
        ],
        "cpu_hamming_weight_claim_reduction_service_ms_samples": [
            value / 1000.0 for value in cpu_hamming_weight_service
        ],
        "metal_hamming_weight_claim_reduction_service_ms_samples": [
            value / 1000.0 for value in metal_hamming_weight_service
        ],
        "cpu_booleanity_hamming_family_ms_samples": [
            value / 1000.0 for value in cpu_booleanity_hamming
        ],
        "metal_booleanity_hamming_family_ms_samples": [
            value / 1000.0 for value in metal_booleanity_hamming
        ],
        "cpu_outer_remainder_ms_samples": [
            value / 1000.0 for value in cpu_outer_remainder
        ],
        "metal_outer_remainder_ms_samples": [
            value / 1000.0 for value in metal_outer_remainder
        ],
        "cpu_product_uniskip_ms_samples": [
            value / 1000.0 for value in cpu_product_uniskip
        ],
        "metal_product_uniskip_ms_samples": [
            value / 1000.0 for value in metal_product_uniskip
        ],
        "cpu_product_remainder_ms_samples": [
            value / 1000.0 for value in cpu_product_remainder
        ],
        "metal_product_remainder_ms_samples": [
            value / 1000.0 for value in metal_product_remainder
        ],
        "cpu_outer_product_family_ms_samples": [
            value / 1000.0 for value in cpu_outer_product_family
        ],
        "metal_outer_product_family_ms_samples": [
            value / 1000.0 for value in metal_outer_product_family
        ],
        "cpu_instruction_claim_reduction_critical_path_ms_samples": [
            value / 1000.0 for value in cpu_instruction_claim
        ],
        "metal_instruction_claim_reduction_critical_path_ms_samples": [
            value / 1000.0 for value in metal_instruction_claim
        ],
        "metal_instruction_claim_reduction_isolated_service_ms_samples": [
            value / 1000.0 for value in metal_instruction_claim_isolated_service
        ],
        "cpu_product_instruction_claim_family_ms_samples": [
            value / 1000.0 for value in cpu_product_instruction_claim
        ],
        "metal_product_instruction_claim_family_ms_samples": [
            value / 1000.0 for value in metal_product_instruction_claim
        ],
        "instruction_input_kernel_service_decision": instruction_input_decision,
        "instruction_read_raf_decision": instruction_read_raf_decision,
        "booleanity_address_phase_decision": booleanity_address_decision,
        "hamming_weight_claim_reduction_decision": hamming_weight_decision,
        "booleanity_hamming_family_decision": booleanity_hamming_decision,
        "outer_remainder_decision": outer_remainder_decision,
        "product_remainder_decision": product_remainder_decision,
        "outer_product_family_decision": outer_product_family_decision,
        "instruction_claim_reduction_critical_path_decision": instruction_claim_decision,
        "product_instruction_claim_family_decision": product_instruction_claim_decision,
        "bytecode_read_raf_cycle_decision": {
            "minimum_speedup": BYTECODE_MIN_SPEEDUP,
            "minimum_pairs": PRODUCTION_PAIRS,
            "median_speedup": bytecode_speedup_median,
            "median_fractional_improvement": bytecode_improvement_median,
            "mad_fractional_improvement": bytecode_improvement_mad,
            "cpu_member_ms_median": cpu_bytecode_median / 1000.0,
            "cpu_member_ms_mad": cpu_bytecode_mad / 1000.0,
            "metal_member_ms_median": metal_bytecode_median / 1000.0,
            "metal_member_ms_mad": metal_bytecode_mad / 1000.0,
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
        },
    }


def microseconds_to_nanoseconds(value: float) -> int:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("duration must be finite and positive")
    return round(value * 1000.0)


def member_record(
    member: dict[str, Any], *, include_prefetch: bool = False
) -> dict[str, Any]:
    components = member["components"]
    rounds_ns = [microseconds_to_nanoseconds(value) for value in components["rounds_us"]]
    prepare_ns = microseconds_to_nanoseconds(components["prepare_us"])
    finish_ns = microseconds_to_nanoseconds(components["finish_us"])
    output_claims_ns = microseconds_to_nanoseconds(components["output_claims_us"])
    rounds_total_ns = sum(rounds_ns)
    prefetch_submit_ns = round(float(components.get("prefetch_submit_us", 0.0)) * 1000.0)
    host_fiat_shamir_us = components.get("host_fiat_shamir_us")
    host_fiat_shamir_ns = (
        [microseconds_to_nanoseconds(value) for value in host_fiat_shamir_us]
        if host_fiat_shamir_us is not None
        else []
    )
    host_fiat_shamir_total_ns = sum(host_fiat_shamir_ns)
    member_ns = (
        prepare_ns
        + rounds_total_ns
        + host_fiat_shamir_total_ns
        + finish_ns
        + output_claims_ns
    )
    record = {
        "prepare_ns": prepare_ns,
        "rounds_ns": rounds_ns,
        "rounds_total_ns": rounds_total_ns,
        "finish_ns": finish_ns,
        "output_claims_ns": output_claims_ns,
        "member_ns": member_ns,
        "outer_counts": member["outer_counts"],
        "metal_counts": member["metal_counts"],
        "resource_observation": member["resource_observation"],
    }
    if host_fiat_shamir_us is not None:
        row_source_us = float(components["row_source_us"])
        if not math.isfinite(row_source_us) or row_source_us < 0.0:
            raise ValueError("row-source duration must be finite and non-negative")
        row_source_ns = round(row_source_us * 1000.0)
        normalized_prepare_ns = prepare_ns - row_source_ns
        normalized_member_ns = member_ns - row_source_ns
        if normalized_prepare_ns <= 0 or normalized_member_ns <= 0:
            raise ValueError("normalized member duration must be positive")
        record["host_fiat_shamir_ns"] = host_fiat_shamir_ns
        record["host_fiat_shamir_total_ns"] = host_fiat_shamir_total_ns
        record["row_source_ns"] = row_source_ns
        record["normalized_prepare_ns"] = normalized_prepare_ns
        record["normalized_member_ns"] = normalized_member_ns
    if include_prefetch:
        record["prefetch_submit_ns"] = prefetch_submit_ns
        record["service_ns"] = member_ns + prefetch_submit_ns
    return record


def outer_remainder_member_record(member: dict[str, Any]) -> dict[str, Any]:
    member_ns = microseconds_to_nanoseconds(member["components"]["member_us"])
    return {
        "member_ns": member_ns,
        "outer_counts": member["outer_counts"],
        "metal_counts": member["metal_counts"],
        "resource_observation": member["resource_observation"],
        "row_lifecycle": member["row_lifecycle"],
    }


def local_kernel_primary_us(result: dict[str, Any], kernel: str) -> float:
    if kernel == BYTECODE_KERNEL:
        value = result["bytecode_member"]["components"]["member_us"]
    elif kernel == "InstructionRaVirtualization":
        value = kernel_wall_us(result["attribution"], kernel)
    elif kernel == INSTRUCTION_INPUT_KERNEL:
        value = result["instruction_input_member"]["components"]["service_us"]
    elif kernel == BOOLEANITY_ADDRESS_KERNEL:
        value = member_record(result["booleanity_address_member"])[
            "normalized_member_ns"
        ] / 1000.0
    elif kernel == HAMMING_WEIGHT_KERNEL:
        value = member_record(result["hamming_weight_member"])[
            "normalized_member_ns"
        ] / 1000.0
    elif kernel == OUTER_REMAINDER_KERNEL:
        value = result["outer_remainder_member"]["components"]["member_us"]
    elif kernel == PRODUCT_REMAINDER_KERNEL:
        value = kernel_wall_us(result["attribution"], kernel)
    elif kernel == INSTRUCTION_CLAIM_KERNEL:
        value = kernel_wall_us(result["attribution"], kernel)
    elif kernel == INSTRUCTION_READ_RAF_KERNEL:
        value = kernel_wall_us(result["attribution"], kernel)
    else:
        raise ValueError(f"unsupported local kernel {kernel}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{kernel} local member duration is invalid")
    return value


def trace_path(root: Path, workload: str, log_n: int, backend: str) -> Path:
    name = workload.replace("-", "_")
    return root / "benchmark-runs" / "perfetto_traces" / f"akita_{name}_{log_n}_{backend}.json"


def run_backend(
    root: Path,
    binary: Path,
    artifact_dir: Path,
    workload: str,
    log_n: int,
    backend: str,
    instruction_ra_materialize_width: int,
    instruction_ra_reuse_inverse: bool,
    bytecode_message_threads: int,
    bytecode_transition_threads: int,
    bytecode_max_threadgroups: int,
    bytecode_cutoff_log2: int,
    bytecode_trace_cutoff_log2: int,
    bytecode_address_implementation: str,
    bytecode_address_outer_tiles: int,
    bytecode_address_trace_cutoff_log2: int,
    instruction_input_native_message_threads: int,
    instruction_input_native_transition_threads: int,
    instruction_input_dense_transition_threads: int,
    instruction_input_cutoff_log2: int,
    instruction_input_trace_cutoff_log2: int,
    instruction_input_borrow_outer_residual: bool,
    booleanity_address_inner_log2: int,
    booleanity_address_selectors_per_tile: int,
    booleanity_address_tile_threads: int,
    booleanity_address_finalize_threads: int,
    booleanity_address_trace_cutoff_log2: int,
    booleanity_address_implementation: str,
    hamming_weight_inner_log2: int,
    hamming_weight_selectors_per_tile: int,
    hamming_weight_tile_threads: int,
    hamming_weight_finalize_threads: int,
    hamming_weight_trace_cutoff_log2: int,
    hamming_weight_implementation: str,
    outer_remainder_materialize_threads: int,
    outer_remainder_transition_threads: int,
    outer_remainder_output_threads: int,
    outer_remainder_cutoff_log2: int,
    outer_remainder_trace_cutoff_log2: int,
    outer_remainder_binding_plan: str,
    product_uniskip_outer_carrier: bool,
    pair_index: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    benchmark_command = [
        str(binary),
        "--name",
        workload,
        "--scale",
        str(log_n),
        "--format",
        "chrome",
        "--backend",
        backend,
        "--bytecode-cycle-algebra",
        "q10",
        "--bytecode-metal-message-threads",
        str(bytecode_message_threads),
        "--bytecode-metal-transition-threads",
        str(bytecode_transition_threads),
        "--bytecode-metal-max-threadgroups",
        str(bytecode_max_threadgroups),
        "--bytecode-metal-cutoff-log2",
        str(bytecode_cutoff_log2),
        "--bytecode-metal-trace-cutoff-log2",
        str(bytecode_trace_cutoff_log2),
        "--bytecode-address-metal-implementation",
        bytecode_address_implementation,
        "--bytecode-address-metal-outer-tiles",
        str(bytecode_address_outer_tiles),
        "--bytecode-address-metal-trace-cutoff-log2",
        str(bytecode_address_trace_cutoff_log2),
        "--instruction-input-metal-native-message-threads",
        str(instruction_input_native_message_threads),
        "--instruction-input-metal-native-transition-threads",
        str(instruction_input_native_transition_threads),
        "--instruction-input-metal-dense-transition-threads",
        str(instruction_input_dense_transition_threads),
        "--instruction-input-metal-cutoff-log2",
        str(instruction_input_cutoff_log2),
        "--instruction-input-metal-trace-cutoff-log2",
        str(instruction_input_trace_cutoff_log2),
        "--booleanity-address-metal-inner-log2",
        str(booleanity_address_inner_log2),
        "--booleanity-address-metal-selectors-per-tile",
        str(booleanity_address_selectors_per_tile),
        "--booleanity-address-metal-tile-threads",
        str(booleanity_address_tile_threads),
        "--booleanity-address-metal-finalize-threads",
        str(booleanity_address_finalize_threads),
        "--booleanity-address-metal-trace-cutoff-log2",
        str(booleanity_address_trace_cutoff_log2),
        "--booleanity-address-metal-implementation",
        booleanity_address_implementation,
        "--hamming-weight-metal-inner-log2",
        str(hamming_weight_inner_log2),
        "--hamming-weight-metal-selectors-per-tile",
        str(hamming_weight_selectors_per_tile),
        "--hamming-weight-metal-tile-threads",
        str(hamming_weight_tile_threads),
        "--hamming-weight-metal-finalize-threads",
        str(hamming_weight_finalize_threads),
        "--hamming-weight-metal-trace-cutoff-log2",
        str(hamming_weight_trace_cutoff_log2),
        "--hamming-weight-metal-implementation",
        hamming_weight_implementation,
        "--outer-remainder-metal-materialize-threads",
        str(outer_remainder_materialize_threads),
        "--outer-remainder-metal-transition-threads",
        str(outer_remainder_transition_threads),
        "--outer-remainder-metal-output-threads",
        str(outer_remainder_output_threads),
        "--outer-remainder-metal-cutoff-log2",
        str(outer_remainder_cutoff_log2),
        "--outer-remainder-metal-trace-cutoff-log2",
        str(outer_remainder_trace_cutoff_log2),
        "--outer-remainder-metal-binding-plan",
        outer_remainder_binding_plan,
    ]
    if product_uniskip_outer_carrier:
        benchmark_command.append("--product-uniskip-outer-carrier")
    if instruction_input_borrow_outer_residual:
        benchmark_command.append("--instruction-input-metal-borrow-outer-residual")
    if backend == "metal":
        benchmark_command.extend(
            [
                "--instruction-ra-materialize-width",
                f"w{instruction_ra_materialize_width}",
            ]
        )
        if instruction_ra_reuse_inverse:
            benchmark_command.append("--instruction-ra-reuse-inverse")
    command = ["/usr/bin/time", "-l", *benchmark_command]
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(PRODUCTION_RAYON_THREADS)
    started_ns = time.time_ns()
    result = subprocess.run(
        command,
        cwd=root,
        env=environment,
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
    )
    label = f"pair-{pair_index:02d}-{backend}"
    (artifact_dir / f"{label}.stdout").write_text(result.stdout)
    (artifact_dir / f"{label}.stderr").write_text(result.stderr)
    if result.returncode != 0:
        raise ValueError(f"{backend} evaluator exited with status {result.returncode}")
    max_rss = parse_max_rss(result.stderr)
    execution_config = validate_piop_execution_stdout(result.stdout)
    bytecode_config = validate_bytecode_stdout(
        result.stdout,
        backend,
        log_n,
        bytecode_message_threads,
        bytecode_transition_threads,
        bytecode_max_threadgroups,
        bytecode_cutoff_log2,
        bytecode_trace_cutoff_log2,
    )
    instruction_input_config = validate_instruction_input_stdout(
        result.stdout,
        backend,
        instruction_input_native_message_threads,
        instruction_input_native_transition_threads,
        instruction_input_dense_transition_threads,
        instruction_input_cutoff_log2,
        instruction_input_trace_cutoff_log2,
        instruction_input_borrow_outer_residual,
    )
    booleanity_address_config = validate_booleanity_address_stdout(
        result.stdout,
        backend,
        booleanity_address_inner_log2,
        booleanity_address_selectors_per_tile,
        booleanity_address_tile_threads,
        booleanity_address_finalize_threads,
        booleanity_address_trace_cutoff_log2,
        booleanity_address_implementation,
    )
    hamming_weight_config = validate_hamming_weight_stdout(
        result.stdout,
        backend,
        hamming_weight_inner_log2,
        hamming_weight_selectors_per_tile,
        hamming_weight_tile_threads,
        hamming_weight_finalize_threads,
        hamming_weight_trace_cutoff_log2,
        hamming_weight_implementation,
    )
    outer_remainder_config = validate_outer_remainder_stdout(
        result.stdout,
        backend,
        outer_remainder_materialize_threads,
        outer_remainder_transition_threads,
        outer_remainder_output_threads,
        outer_remainder_cutoff_log2,
        outer_remainder_trace_cutoff_log2,
        outer_remainder_binding_plan,
        product_uniskip_outer_carrier,
    )
    source = trace_path(root, workload, log_n, backend)
    if not source.is_file() or source.stat().st_mtime_ns <= started_ns:
        raise ValueError(f"{backend} evaluator did not emit a fresh trace")
    source_stat = source.stat()
    source_sha256 = file_sha256(source)
    destination = artifact_dir / f"{label}.trace.json"
    shutil.copy2(source, destination)
    final_source_stat = source.stat()
    if (
        final_source_stat.st_mtime_ns != source_stat.st_mtime_ns
        or final_source_stat.st_size != source_stat.st_size
        or file_sha256(destination) != source_sha256
    ):
        raise ValueError(f"{backend} trace changed while it was captured")
    events = load_trace_events(destination)
    bytecode_member = bytecode_member_breakdown(
        events, backend, log_n, bytecode_cutoff_log2
    )
    instruction_input_member = instruction_input_member_breakdown(
        events,
        backend,
        log_n,
        instruction_input_cutoff_log2,
        instruction_input_borrow_outer_residual,
    )
    if backend == "metal" and booleanity_address_implementation == "packed-hot":
        booleanity_address_member = packed_hot_booleanity_address_member_breakdown(
            events, backend, log_n
        )
    else:
        booleanity_address_member = booleanity_address_member_breakdown(
            events,
            backend,
            log_n,
            booleanity_address_inner_log2,
            booleanity_address_selectors_per_tile,
            booleanity_address_tile_threads,
            booleanity_address_finalize_threads,
        )
    if backend == "metal" and hamming_weight_implementation == "retained-hot":
        hamming_weight_member = retained_hot_hamming_weight_member_breakdown(
            events, backend, log_n
        )
    else:
        hamming_weight_member = hamming_weight_member_breakdown(
            events,
            backend,
            log_n,
            hamming_weight_inner_log2,
            hamming_weight_selectors_per_tile,
            hamming_weight_tile_threads,
            hamming_weight_finalize_threads,
        )
    outer_remainder_member = outer_remainder_member_breakdown(
        events,
        backend,
        log_n,
        outer_remainder_cutoff_log2,
        outer_remainder_trace_cutoff_log2,
        product_uniskip_outer_carrier,
    )
    product_uniskip = product_uniskip_observation(
        events,
        backend,
        log_n,
        product_uniskip_outer_carrier,
    )
    instruction_claim = instruction_claim_observation(events, backend, log_n)
    if backend == "metal" and product_uniskip_outer_carrier:
        carrier = product_uniskip["resource_observation"]
        outer_carrier = outer_remainder_member["product_uniskip_carrier"]
        producer = instruction_claim["resource_observation"]
        if (
            carrier is None
            or outer_carrier is None
            or producer is None
            or carrier["source_rows_storage_id"]
            != outer_carrier["source_rows_storage_id"]
            or carrier["product_rows_storage_id"]
            != producer["producer_rows_storage_id"]
        ):
            raise ValueError("Product uni-skip carrier provenance is inconsistent")
    attribution = trace_attribution(events)
    for name, member in (
        (BYTECODE_KERNEL, bytecode_member),
        (INSTRUCTION_INPUT_KERNEL, instruction_input_member),
        (BOOLEANITY_ADDRESS_KERNEL, booleanity_address_member),
        (HAMMING_WEIGHT_KERNEL, hamming_weight_member),
    ):
        attributed_us = kernel_wall_us(attribution, name)
        if name in {BOOLEANITY_ADDRESS_KERNEL, HAMMING_WEIGHT_KERNEL}:
            attributed_us += float(
                member["components"]["host_fiat_shamir_total_us"]
            )
        if not math.isclose(
            attributed_us,
            float(member["components"]["member_us"]),
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{name} member parser disagrees with trace attribution")
    complete_member_us = float(outer_remainder_member["components"]["member_us"])
    if complete_member_us > unique_span_duration_us(events):
        raise ValueError("OuterRemainder member exceeds the PIOP span")
    stdout_path = artifact_dir / f"{label}.stdout"
    stderr_path = artifact_dir / f"{label}.stderr"
    return {
        "piop_us": unique_span_duration_us(events),
        "backend_witness_prepare_us": unique_named_span_duration_us(
            events, BACKEND_WITNESS_PREP_SPAN
        ),
        "attribution": attribution,
        "bytecode_config": bytecode_config,
        "bytecode_member": bytecode_member,
        "instruction_input_config": instruction_input_config,
        "instruction_input_member": instruction_input_member,
        "booleanity_address_config": booleanity_address_config,
        "booleanity_address_member": booleanity_address_member,
        "hamming_weight_config": hamming_weight_config,
        "hamming_weight_member": hamming_weight_member,
        "outer_remainder_config": outer_remainder_config,
        "outer_remainder_member": outer_remainder_member,
        "product_uniskip": product_uniskip,
        "instruction_claim": instruction_claim,
        "execution_config": execution_config,
        "command": command,
        "max_rss_bytes": max_rss,
        "artifacts": {
            "stdout": {"path": stdout_path.name, "sha256": file_sha256(stdout_path)},
            "stderr": {"path": stderr_path.name, "sha256": file_sha256(stderr_path)},
            "trace": {"path": destination.name, "sha256": source_sha256},
        },
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_max_rss(stderr: str) -> int:
    matches = list(MAX_RSS_RE.finditer(stderr))
    if len(matches) != 1:
        raise ValueError("evaluator must emit exactly one maximum-RSS record")
    value = int(matches[0].group("bytes"))
    if value <= 0:
        raise ValueError("evaluator maximum RSS must be positive")
    return value


def canonical_build_command() -> list[str]:
    return [
        "cargo",
        "build",
        "--release",
        "--quiet",
        "-p",
        "jolt-prover",
        "--example",
        "modular_benchmark",
        "--features",
        FEATURES,
    ]


def build_binary(
    root: Path, artifact_dir: Path, timeout_seconds: int
) -> tuple[Path, list[str]]:
    if os.environ.get("CARGO_TARGET_DIR"):
        raise ValueError("canonical evaluator forbids CARGO_TARGET_DIR")
    command = canonical_build_command()
    completed = subprocess.run(
        command,
        cwd=root,
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "build.stdout").write_text(completed.stdout)
    (artifact_dir / "build.stderr").write_text(completed.stderr)
    if completed.returncode != 0:
        raise ValueError(f"evaluator build exited with status {completed.returncode}")
    binary = root / "target" / "release" / "examples" / "modular_benchmark"
    if not binary.is_file():
        raise ValueError("evaluator binary is missing")
    return binary, command


def default_artifact_dir(root: Path) -> Path:
    configured = os.environ.get("JOLT_AUTORESEARCH_EVAL_DIR")
    if configured:
        return Path(configured).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return root / "benchmark-runs" / "metal-piop-eval" / timestamp


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def worktree_state_digest(
    tracked_diff: bytes, untracked_files: list[tuple[bytes, bytes]]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"tracked\0")
    digest.update(tracked_diff)
    for path, contents in sorted(untracked_files):
        digest.update(b"untracked\0")
        digest.update(path)
        digest.update(b"\0")
        digest.update(contents)
    return digest.hexdigest()


def source_fingerprint(root: Path) -> dict[str, Any]:
    tracked = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    untracked_output = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    untracked = []
    for raw_path in filter(None, untracked_output.split(b"\0")):
        path = root / os.fsdecode(raw_path)
        contents = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        untracked.append((raw_path, contents))
    return {
        "git_revision": git_head(root),
        "worktree_dirty": bool(tracked or untracked),
        "worktree_state_sha256": worktree_state_digest(tracked, untracked),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", default=Path(__file__).resolve().parents[1])
    result.add_argument("--mode", choices=["diagnostic", "production"], default="diagnostic")
    result.add_argument(
        "--local-kernel", choices=sorted(LOCAL_KERNELS), default=BYTECODE_KERNEL
    )
    result.add_argument(
        "--workload",
        choices=["fibonacci", "sha2-chain", "sha3-chain", "btreemap"],
        default="fibonacci",
    )
    result.add_argument("--log-n", type=int, default=26)
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--timeout-seconds", type=int, default=7200)
    result.add_argument(
        "--instruction-ra-materialize-width",
        type=int,
        choices=[16, 32, 64, 128, 256, 512],
        default=16,
    )
    result.add_argument("--instruction-ra-reuse-inverse", action="store_true")
    result.add_argument(
        "--bytecode-metal-message-threads",
        type=int,
        choices=[32, 64, 128, 256, 512, 1024],
        default=256,
    )
    result.add_argument(
        "--bytecode-metal-transition-threads",
        type=int,
        choices=[32, 64, 128, 256, 512, 1024],
        default=128,
    )
    result.add_argument(
        "--bytecode-metal-max-threadgroups",
        type=int,
        choices=[1024, 2048, 4096, 8192],
        default=8192,
    )
    result.add_argument(
        "--bytecode-metal-cutoff-log2",
        type=int,
        choices=[12, 14, 16, 18, 20],
        default=16,
    )
    result.add_argument(
        "--bytecode-metal-trace-cutoff-log2",
        type=int,
        choices=[18, 20, 22, 24, 25],
        default=18,
    )
    result.add_argument(
        "--bytecode-address-metal-implementation",
        choices=["cpu", "csr-shadow", "address-major-shadow"],
        default="cpu",
    )
    result.add_argument(
        "--bytecode-address-metal-outer-tiles",
        type=int,
        choices=[1, 2, 4, 8, 16, 32],
        default=8,
    )
    result.add_argument(
        "--bytecode-address-metal-trace-cutoff-log2",
        type=int,
        choices=[18, 20, 22, 24, 25, 26, 27, 28],
        default=26,
    )
    result.add_argument(
        "--instruction-input-metal-native-message-threads",
        type=int,
        choices=[32, 64, 128, 256, 512, 1024],
        default=256,
    )
    result.add_argument(
        "--instruction-input-metal-native-transition-threads",
        type=int,
        choices=[32, 64, 128, 256, 512, 1024],
        default=128,
    )
    result.add_argument(
        "--instruction-input-metal-dense-transition-threads",
        type=int,
        choices=[32, 64, 128, 256, 512, 1024],
        default=128,
    )
    result.add_argument(
        "--instruction-input-metal-cutoff-log2",
        type=int,
        choices=[12, 14, 15, 16, 17, 18, 20],
        default=16,
    )
    result.add_argument(
        "--instruction-input-metal-trace-cutoff-log2",
        type=int,
        choices=[24, 25, 26, 27, 28],
        default=25,
    )
    result.add_argument(
        "--instruction-input-metal-borrow-outer-residual", action="store_true"
    )
    result.add_argument(
        "--booleanity-address-metal-inner-log2",
        type=int,
        choices=[12, 13, 14, 15, 16],
        default=15,
    )
    result.add_argument(
        "--booleanity-address-metal-selectors-per-tile",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=6,
    )
    result.add_argument(
        "--booleanity-address-metal-tile-threads",
        type=int,
        choices=[32, 64, 128, 256, 512, 1024],
        default=512,
    )
    result.add_argument(
        "--booleanity-address-metal-finalize-threads",
        type=int,
        choices=[256, 512, 768, 1024],
        default=1024,
    )
    result.add_argument(
        "--booleanity-address-metal-trace-cutoff-log2",
        type=int,
        choices=[18, 20, 22, 24, 25, 26, 27, 28],
        default=18,
    )
    result.add_argument(
        "--booleanity-address-metal-implementation",
        choices=["accepted", "packed-hot"],
        default="accepted",
    )
    result.add_argument(
        "--hamming-weight-metal-inner-log2",
        type=int,
        choices=[12, 13, 14, 15, 16],
        default=15,
    )
    result.add_argument(
        "--hamming-weight-metal-selectors-per-tile",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=6,
    )
    result.add_argument(
        "--hamming-weight-metal-tile-threads",
        type=int,
        choices=[32, 64, 128, 256, 512, 1024],
        default=512,
    )
    result.add_argument(
        "--hamming-weight-metal-finalize-threads",
        type=int,
        choices=[256, 512, 768, 1024],
        default=1024,
    )
    result.add_argument(
        "--hamming-weight-metal-trace-cutoff-log2",
        type=int,
        choices=[18, 20, 22, 24, 25, 26, 27, 28],
        default=18,
    )
    result.add_argument(
        "--hamming-weight-metal-implementation",
        choices=["accepted-rows", "retained-hot"],
        default="accepted-rows",
    )
    result.add_argument(
        "--outer-remainder-metal-materialize-threads",
        type=int,
        choices=[128, 256, 512],
        default=256,
    )
    result.add_argument(
        "--outer-remainder-metal-transition-threads",
        type=int,
        choices=[64, 128, 256],
        default=128,
    )
    result.add_argument(
        "--outer-remainder-metal-output-threads",
        type=int,
        choices=[128, 256, 512],
        default=256,
    )
    result.add_argument(
        "--outer-remainder-metal-cutoff-log2",
        type=int,
        choices=[14, 15, 16, 17, 18],
        default=16,
    )
    result.add_argument(
        "--outer-remainder-metal-trace-cutoff-log2",
        type=int,
        choices=[18, 20, 22, 24, 25, 26, 27, 28],
        default=18,
    )
    result.add_argument(
        "--outer-remainder-metal-binding-plan",
        choices=["b_only_v1", "b_only_padded_56_v1"],
        default="b_only_v1",
    )
    result.add_argument("--product-uniskip-outer-carrier", action="store_true")
    result.add_argument("--trace", type=Path)
    return result


def validate_run_class(mode: str, workload: str, log_n: int, repeats: int) -> None:
    if mode == "production" and (
        workload != "fibonacci" or log_n not in {26, 27} or repeats != PRODUCTION_PAIRS
    ):
        raise ValueError(
            "production mode requires Fibonacci, log-n 26 or 27, and five pairs"
        )


def main() -> int:
    args = parser().parse_args()
    if args.trace is not None:
        try:
            print(json.dumps(trace_attribution(load_trace_events(args.trace)), indent=2))
            return 0
        except (OSError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    if args.log_n < 1 or args.repeats < 1 or args.timeout_seconds < 1:
        print("error: log-n, repeats, and timeout must be positive", file=sys.stderr)
        return 2
    if args.log_n < 25:
        print("error: Metal PIOP evaluation requires log-n at least 25", file=sys.stderr)
        return 2
    if args.bytecode_metal_cutoff_log2 > args.log_n - 1:
        print("error: Bytecode Metal cutoff must not exceed half the trace", file=sys.stderr)
        return 2
    if args.bytecode_metal_trace_cutoff_log2 > args.log_n:
        print("error: Bytecode Metal trace cutoff disables the measured backend", file=sys.stderr)
        return 2
    if (
        args.bytecode_address_metal_implementation != "cpu"
        and args.bytecode_address_metal_trace_cutoff_log2 > args.log_n
    ):
        print(
            "error: Bytecode address Metal trace cutoff disables the measured backend",
            file=sys.stderr,
        )
        return 2
    if args.instruction_input_metal_cutoff_log2 > args.log_n - 1:
        print("error: InstructionInput cutoff must not exceed half the trace", file=sys.stderr)
        return 2
    if args.instruction_input_metal_trace_cutoff_log2 > args.log_n:
        print(
            "error: InstructionInput trace cutoff disables the measured backend",
            file=sys.stderr,
        )
        return 2
    if args.booleanity_address_metal_inner_log2 > args.log_n:
        print("error: Booleanity address inner split exceeds the trace", file=sys.stderr)
        return 2
    if args.booleanity_address_metal_trace_cutoff_log2 > args.log_n:
        print(
            "error: Booleanity address trace cutoff disables the measured backend",
            file=sys.stderr,
        )
        return 2
    if args.hamming_weight_metal_inner_log2 > args.log_n:
        print("error: Hamming-weight inner split exceeds the trace", file=sys.stderr)
        return 2
    if args.hamming_weight_metal_trace_cutoff_log2 > args.log_n:
        print(
            "error: Hamming-weight trace cutoff disables the measured backend",
            file=sys.stderr,
        )
        return 2
    if (
        args.hamming_weight_metal_implementation == "retained-hot"
        and args.booleanity_address_metal_implementation != "packed-hot"
    ):
        print(
            "error: retained Hamming requires the packed-hot Booleanity producer",
            file=sys.stderr,
        )
        return 2
    if args.outer_remainder_metal_cutoff_log2 > args.log_n - 1:
        print("error: OuterRemainder cutoff must not exceed half the trace", file=sys.stderr)
        return 2
    if args.outer_remainder_metal_trace_cutoff_log2 > args.log_n:
        print(
            "error: OuterRemainder trace cutoff disables the measured backend",
            file=sys.stderr,
        )
        return 2
    if args.instruction_ra_reuse_inverse and args.instruction_ra_materialize_width == 16:
        print("error: width-16 Instruction RA cannot reuse the inverse", file=sys.stderr)
        return 2
    try:
        validate_run_class(args.mode, args.workload, args.log_n, args.repeats)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    root = Path(args.root).resolve()
    local_kernel = LOCAL_KERNELS[args.local_kernel]
    artifact_dir = default_artifact_dir(root)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    source = source_fingerprint(root)
    pairs = []
    pair_records = []
    orders = []
    attributions = []
    try:
        if args.mode == "production" and source["worktree_dirty"]:
            raise ValueError("production evaluation requires a clean source worktree")
        binary, build_command = build_binary(root, artifact_dir, args.timeout_seconds)
        if source_fingerprint(root) != source:
            raise ValueError("source worktree changed during the evaluator build")
        binary_sha256 = file_sha256(binary)
        for index in range(args.repeats):
            order = ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
            orders.append(order)
            results: dict[str, dict[str, Any]] = {}
            for backend in order:
                results[backend] = run_backend(
                    root,
                    binary,
                    artifact_dir,
                    args.workload,
                    args.log_n,
                    backend,
                    args.instruction_ra_materialize_width,
                    args.instruction_ra_reuse_inverse,
                    args.bytecode_metal_message_threads,
                    args.bytecode_metal_transition_threads,
                    args.bytecode_metal_max_threadgroups,
                    args.bytecode_metal_cutoff_log2,
                    args.bytecode_metal_trace_cutoff_log2,
                    args.bytecode_address_metal_implementation,
                    args.bytecode_address_metal_outer_tiles,
                    args.bytecode_address_metal_trace_cutoff_log2,
                    args.instruction_input_metal_native_message_threads,
                    args.instruction_input_metal_native_transition_threads,
                    args.instruction_input_metal_dense_transition_threads,
                    args.instruction_input_metal_cutoff_log2,
                    args.instruction_input_metal_trace_cutoff_log2,
                    args.instruction_input_metal_borrow_outer_residual,
                    args.booleanity_address_metal_inner_log2,
                    args.booleanity_address_metal_selectors_per_tile,
                    args.booleanity_address_metal_tile_threads,
                    args.booleanity_address_metal_finalize_threads,
                    args.booleanity_address_metal_trace_cutoff_log2,
                    args.booleanity_address_metal_implementation,
                    args.hamming_weight_metal_inner_log2,
                    args.hamming_weight_metal_selectors_per_tile,
                    args.hamming_weight_metal_tile_threads,
                    args.hamming_weight_metal_finalize_threads,
                    args.hamming_weight_metal_trace_cutoff_log2,
                    args.hamming_weight_metal_implementation,
                    args.outer_remainder_metal_materialize_threads,
                    args.outer_remainder_metal_transition_threads,
                    args.outer_remainder_metal_output_threads,
                    args.outer_remainder_metal_cutoff_log2,
                    args.outer_remainder_metal_trace_cutoff_log2,
                    args.outer_remainder_metal_binding_plan,
                    args.product_uniskip_outer_carrier,
                    index + 1,
                    args.timeout_seconds,
                )
            booleanity_address_records = {
                backend: member_record(results[backend]["booleanity_address_member"])
                for backend in ("optimized", "metal")
            }
            hamming_weight_records = {
                backend: member_record(results[backend]["hamming_weight_member"])
                for backend in ("optimized", "metal")
            }
            outer_remainder_records = {
                backend: outer_remainder_member_record(
                    results[backend]["outer_remainder_member"]
                )
                for backend in ("optimized", "metal")
            }
            metal_instruction_claim_resources = results["metal"][
                "instruction_claim"
            ]["resource_observation"]
            if metal_instruction_claim_resources is None:
                raise ValueError("Metal Instruction Claim observation is missing")
            cpu_instruction_claim_us = kernel_wall_us(
                results["optimized"]["attribution"], INSTRUCTION_CLAIM_KERNEL
            )
            metal_instruction_claim_us = kernel_wall_us(
                results["metal"]["attribution"], INSTRUCTION_CLAIM_KERNEL
            )
            pairs.append(
                {
                    "order": order,
                    "cpu_us": results["optimized"]["piop_us"],
                    "metal_us": results["metal"]["piop_us"],
                    "cpu_prepare_us": results["optimized"]["backend_witness_prepare_us"],
                    "metal_prepare_us": results["metal"]["backend_witness_prepare_us"],
                    "cpu_instruction_ra_us": kernel_wall_us(
                        results["optimized"]["attribution"], "InstructionRaVirtualization"
                    ),
                    "metal_instruction_ra_us": kernel_wall_us(
                        results["metal"]["attribution"], "InstructionRaVirtualization"
                    ),
                    "cpu_bytecode_us": float(
                        results["optimized"]["bytecode_member"]["components"]["member_us"]
                    ),
                    "metal_bytecode_us": float(
                        results["metal"]["bytecode_member"]["components"]["member_us"]
                    ),
                    "cpu_instruction_input_us": float(
                        results["optimized"]["instruction_input_member"]["components"][
                            "service_us"
                        ]
                    ),
                    "metal_instruction_input_us": float(
                        results["metal"]["instruction_input_member"]["components"][
                            "service_us"
                        ]
                    ),
                    "cpu_instruction_read_raf_us": kernel_wall_us(
                        results["optimized"]["attribution"],
                        INSTRUCTION_READ_RAF_KERNEL,
                    ),
                    "metal_instruction_read_raf_us": kernel_wall_us(
                        results["metal"]["attribution"],
                        INSTRUCTION_READ_RAF_KERNEL,
                    ),
                    "cpu_booleanity_address_us": float(
                        booleanity_address_records["optimized"][
                            "normalized_member_ns"
                        ]
                    )
                    / 1000.0,
                    "metal_booleanity_address_us": float(
                        booleanity_address_records["metal"]["normalized_member_ns"]
                    )
                    / 1000.0,
                    "cpu_booleanity_address_service_us": float(
                        booleanity_address_records["optimized"]["member_ns"]
                    )
                    / 1000.0,
                    "metal_booleanity_address_service_us": float(
                        booleanity_address_records["metal"]["member_ns"]
                    )
                    / 1000.0,
                    "cpu_hamming_weight_us": float(
                        hamming_weight_records["optimized"]["normalized_member_ns"]
                    )
                    / 1000.0,
                    "metal_hamming_weight_us": float(
                        hamming_weight_records["metal"]["normalized_member_ns"]
                    )
                    / 1000.0,
                    "cpu_hamming_weight_service_us": float(
                        hamming_weight_records["optimized"]["member_ns"]
                    )
                    / 1000.0,
                    "metal_hamming_weight_service_us": float(
                        hamming_weight_records["metal"]["member_ns"]
                    )
                    / 1000.0,
                    "cpu_outer_remainder_us": float(
                        outer_remainder_records["optimized"]["member_ns"]
                    )
                    / 1000.0,
                    "metal_outer_remainder_us": float(
                        outer_remainder_records["metal"]["member_ns"]
                    )
                    / 1000.0,
                    "cpu_product_uniskip_us": float(
                        results["optimized"]["product_uniskip"]["components"][
                            "member_us"
                        ]
                    ),
                    "metal_product_uniskip_us": float(
                        results["metal"]["product_uniskip"]["components"][
                            "member_us"
                        ]
                    ),
                    "cpu_product_remainder_us": kernel_wall_us(
                        results["optimized"]["attribution"], PRODUCT_REMAINDER_KERNEL
                    ),
                    "metal_product_remainder_us": kernel_wall_us(
                        results["metal"]["attribution"], PRODUCT_REMAINDER_KERNEL
                    ),
                    "cpu_instruction_claim_us": cpu_instruction_claim_us,
                    "metal_instruction_claim_us": metal_instruction_claim_us,
                    "metal_instruction_claim_isolated_service_us": (
                        metal_instruction_claim_us
                        + float(metal_instruction_claim_resources["overlap_wall_ns"])
                        / 1000.0
                    ),
                }
            )
            attributions.append(
                {
                    "optimized": results["optimized"]["attribution"],
                    "metal": results["metal"]["attribution"],
                    "bytecode": {
                        "optimized_config": results["optimized"]["bytecode_config"],
                        "metal_config": results["metal"]["bytecode_config"],
                        "optimized_member": results["optimized"]["bytecode_member"],
                        "metal_member": results["metal"]["bytecode_member"],
                    },
                    "instruction_input": {
                        "optimized_config": results["optimized"][
                            "instruction_input_config"
                        ],
                        "metal_config": results["metal"]["instruction_input_config"],
                        "optimized_member": results["optimized"][
                            "instruction_input_member"
                        ],
                        "metal_member": results["metal"]["instruction_input_member"],
                    },
                    "booleanity_address": {
                        "optimized_config": results["optimized"][
                            "booleanity_address_config"
                        ],
                        "metal_config": results["metal"]["booleanity_address_config"],
                        "optimized_member": results["optimized"][
                            "booleanity_address_member"
                        ],
                        "metal_member": results["metal"]["booleanity_address_member"],
                    },
                    "hamming_weight": {
                        "optimized_config": results["optimized"][
                            "hamming_weight_config"
                        ],
                        "metal_config": results["metal"]["hamming_weight_config"],
                        "optimized_member": results["optimized"][
                            "hamming_weight_member"
                        ],
                        "metal_member": results["metal"]["hamming_weight_member"],
                    },
                    "outer_remainder": {
                        "optimized_config": results["optimized"][
                            "outer_remainder_config"
                        ],
                        "metal_config": results["metal"]["outer_remainder_config"],
                        "optimized_member": results["optimized"][
                            "outer_remainder_member"
                        ],
                        "metal_member": results["metal"]["outer_remainder_member"],
                    },
                    "product_uniskip": {
                        "optimized": results["optimized"]["product_uniskip"],
                        "metal": results["metal"]["product_uniskip"],
                    },
                    "instruction_claim": {
                        "optimized": results["optimized"]["instruction_claim"],
                        "metal": results["metal"]["instruction_claim"],
                    },
                    "execution": {
                        "optimized": results["optimized"]["execution_config"],
                        "metal": results["metal"]["execution_config"],
                    },
                    "commands": {
                        "optimized": results["optimized"]["command"],
                        "metal": results["metal"]["command"],
                    },
                }
            )
            pair_records.append(
                {
                    "index": index + 1,
                    "order": order,
                    "arms": {
                        backend: {
                            "piop_ns": microseconds_to_nanoseconds(
                                results[backend]["piop_us"]
                            ),
                            "backend_witness_prepare_ns": microseconds_to_nanoseconds(
                                results[backend]["backend_witness_prepare_us"]
                            ),
                            "max_rss_bytes": results[backend]["max_rss_bytes"],
                            "bytecode": member_record(
                                results[backend]["bytecode_member"]
                            ),
                            "instruction_input": member_record(
                                results[backend]["instruction_input_member"],
                                include_prefetch=True,
                            ),
                            "instruction_input_row_lifecycle": results[backend][
                                "instruction_input_member"
                            ]["row_lifecycle"],
                            "booleanity_address": booleanity_address_records[backend],
                            "booleanity_address_row_lifecycle": results[backend][
                                "booleanity_address_member"
                            ]["row_lifecycle"],
                            "hamming_weight": hamming_weight_records[backend],
                            "hamming_weight_row_lifecycle": results[backend][
                                "hamming_weight_member"
                            ]["row_lifecycle"],
                            "outer_remainder": outer_remainder_records[backend],
                            "outer_remainder_row_lifecycle": results[backend][
                                "outer_remainder_member"
                            ]["row_lifecycle"],
                            "product_uniskip": results[backend]["product_uniskip"],
                            "product_uniskip_ns": microseconds_to_nanoseconds(
                                results[backend]["product_uniskip"]["components"][
                                    "member_us"
                                ]
                            ),
                            "instruction_claim": results[backend][
                                "instruction_claim"
                            ],
                            "product_remainder_ns": microseconds_to_nanoseconds(
                                kernel_wall_us(
                                    results[backend]["attribution"],
                                    PRODUCT_REMAINDER_KERNEL,
                                )
                            ),
                            "instruction_claim_isolated_service_ns": microseconds_to_nanoseconds(
                                kernel_wall_us(
                                    results[backend]["attribution"],
                                    INSTRUCTION_CLAIM_KERNEL,
                                )
                                + (
                                    float(
                                        results[backend]["instruction_claim"][
                                            "resource_observation"
                                        ]["overlap_wall_ns"]
                                    )
                                    / 1000.0
                                    if results[backend]["instruction_claim"][
                                        "resource_observation"
                                    ]
                                    is not None
                                    else 0.0
                                )
                            ),
                            "local": {
                                "kernel": local_kernel["name"],
                                "primary_ns": microseconds_to_nanoseconds(
                                    local_kernel_primary_us(
                                        results[backend], local_kernel["name"]
                                    )
                                ),
                            },
                            "config": results[backend]["bytecode_config"],
                            "command": results[backend]["command"],
                            "artifacts": results[backend]["artifacts"],
                        }
                        for backend in ("optimized", "metal")
                    },
                }
            )
        metrics = summarize_pairs(pairs)
        if source_fingerprint(root) != source:
            raise ValueError("source worktree changed during the paired evaluation")
        if file_sha256(binary) != binary_sha256:
            raise ValueError("evaluator binary changed during the paired evaluation")
        output = {
            "schema_version": SCHEMA_VERSION,
            "kernel": "akita_piop",
            "local_kernel": local_kernel["name"],
            "local_metric": {
                "metric": local_kernel["metric"],
                "paired_metric": local_kernel["paired_metric"],
            },
            "run_class": {
                "mode": args.mode,
                "acceptance_eligible": args.mode == "production",
            },
            "metrics": metrics,
            "pairs": pair_records,
            "attribution_samples": attributions,
            "guards": {
                "cpu_proofs_verified": True,
                "metal_proofs_verified": True,
                "unique_piop_span": True,
                "unique_backend_witness_prepare_span": True,
                "rayon_threads_pinned": all(
                    sample["execution"]
                    == {
                        "optimized": {"rayon_threads": PRODUCTION_RAYON_THREADS},
                        "metal": {"rayon_threads": PRODUCTION_RAYON_THREADS},
                    }
                    for sample in attributions
                ),
                "target_scale": args.log_n in {26, 27},
                "bytecode_q10_cpu_control": True,
                "bytecode_metal_backend_exercised": all(
                    sample["bytecode"]["metal_member"]["metal_counts"]["first_message"]
                    == 1
                    for sample in attributions
                ),
                "bytecode_working_set_admitted": all(
                    sample["bytecode"]["metal_member"]["resource_observation"]
                    is not None
                    for sample in attributions
                ),
                "bytecode_readback_exact": all(
                    sample["bytecode"]["metal_member"]["resource_observation"][
                        "readback_bytes"
                    ]
                    == 5 * (1 << args.bytecode_metal_cutoff_log2) * 16
                    for sample in attributions
                ),
                "bytecode_command_buffers_completed": all(
                    2
                    + sample["bytecode"]["metal_member"]["metal_counts"][
                        "dense_round"
                    ]
                    == args.log_n - args.bytecode_metal_cutoff_log2 + 1
                    for sample in attributions
                ),
                "local_kernel_attributed": all(
                    all(
                        (
                            sample["outer_remainder"][f"{backend}_member"][
                                "components"
                            ]["member_us"]
                            > 0.0
                            if local_kernel["name"] == OUTER_REMAINDER_KERNEL
                            else any(
                                kernel["kernel"] == local_kernel["name"]
                                for kernel in sample[backend]["kernels"]
                            )
                        )
                        for backend in ("optimized", "metal")
                    )
                    for sample in attributions
                ),
                "local_kernel_metal_backend_exercised": all(
                    any(
                        span["span"].startswith(local_kernel["backend_prefix"])
                        for span in sample["metal"]["backend_spans"]
                    )
                    for sample in attributions
                ),
                "stable_source": True,
                "stable_binary": True,
                "production_contract": args.mode == "production",
                "bytecode_local_gate": metrics["bytecode_read_raf_cycle_decision"][
                    "clears"
                ],
                "instruction_read_raf_local_gate": metrics[
                    "instruction_read_raf_decision"
                ]["clears"],
                "instruction_input_cpu_control": all(
                    not any(
                        sample["instruction_input"]["optimized_member"]["metal_counts"].values()
                    )
                    for sample in attributions
                ),
                "instruction_input_cpu_rows_reused": all(
                    sample["instruction_input"]["optimized_member"]["row_lifecycle"][
                        "kind"
                    ]
                    == "optimized_cpu"
                    and sample["instruction_input"]["optimized_member"][
                        "row_lifecycle"
                    ]["rows"]
                    == 1 << args.log_n
                    and sample["instruction_input"]["optimized_member"][
                        "row_lifecycle"
                    ]["row_bytes"]
                    == 48
                    and sample["instruction_input"]["optimized_member"][
                        "row_lifecycle"
                    ]["prepare_storage_id"]
                    == sample["instruction_input"]["optimized_member"][
                        "row_lifecycle"
                    ]["stage3_storage_id"]
                    for sample in attributions
                ),
                "instruction_input_metal_backend_exercised": all(
                    sample["instruction_input"]["metal_member"]["metal_counts"][
                        "first_message"
                    ]
                    == 1
                    for sample in attributions
                ),
                "instruction_input_resident_rows_reused": all(
                    sample["instruction_input"]["metal_member"]["resource_observation"][
                        "resident_rows_reused"
                    ]
                    is True
                    and sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "kind"
                    ]
                    == "metal_compact_resident"
                    and sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "rows"
                    ]
                    == 1 << args.log_n
                    and sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "row_bytes"
                    ]
                    == 48
                    and sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "prepare_storage_id"
                    ]
                    == sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "stage1_storage_id"
                    ]
                    == sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "stage3_storage_id"
                    ]
                    and sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "residual_storage_id"
                    ]
                    != sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "prepare_storage_id"
                    ]
                    for sample in attributions
                ),
                "instruction_input_compact_rows_direct_and_stable": all(
                    sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "row_production"
                    ]
                    == {
                        "source_kind": "owned_random_access",
                        "witness_row_extractions": 1 << args.log_n,
                        "residual_rows_written": 1 << args.log_n,
                        "compact_rows_written": 1 << args.log_n,
                        "compact_row_bytes": 48,
                        "residual_row_bytes": 112,
                        "compact_allocations": 1,
                        "residual_allocations": 1,
                        "full_row_allocations": 0,
                        "full_domain_copy_bytes": 0,
                        "full_domain_copy_dispatches": 0,
                        "host_repack_rows": 0,
                    }
                    for sample in attributions
                ),
                "instruction_input_working_set_admitted": all(
                    sample["instruction_input"]["metal_member"]["resource_observation"]
                    is not None
                    for sample in attributions
                ),
                "instruction_input_readback_exact": all(
                    sample["instruction_input"]["metal_member"]["resource_observation"][
                        "readback_bytes"
                    ]
                    == 8 * (1 << args.instruction_input_metal_cutoff_log2) * 16
                    for sample in attributions
                ),
                "instruction_input_host_readback_preallocated_outside_piop": all(
                    sample["instruction_input"]["metal_member"]["resource_observation"][
                        "host_tail_bytes"
                    ]
                    == 8 * (1 << args.instruction_input_metal_cutoff_log2) * 16
                    for sample in attributions
                ),
                "instruction_input_no_round_device_buffer_allocations": all(
                    sample["instruction_input"]["metal_member"]["resource_observation"][
                        "round_device_buffer_allocations"
                    ]
                    == 0
                    for sample in attributions
                ),
                "instruction_input_minimal_initialization_exact": all(
                    sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["storage_initialization"]["mode"]
                    == "minimal"
                    and sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["storage_initialization"]["bytes"]
                    == (
                        64
                        if args.instruction_input_metal_borrow_outer_residual
                        else 96
                    )
                    and sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["storage_initialization"]["device_buffers"]
                    == (
                        4
                        if args.instruction_input_metal_borrow_outer_residual
                        else 6
                    )
                    for sample in attributions
                ),
                "instruction_input_storage_buffers_stable": all(
                    len(
                        set(
                            sample["instruction_input"]["metal_member"][
                                "resource_observation"
                            ]["storage_initialization"]["buffer_identities"]
                        )
                    )
                    == (
                        5
                        if args.instruction_input_metal_borrow_outer_residual
                        else 6
                    )
                    and sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["storage_initialization"]["buffer_identities"]
                    == sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["native_primer"]["storage_buffer_identities"]
                    for sample in attributions
                ),
                "instruction_input_borrowed_dense_arena_exact": all(
                    (
                        sample["instruction_input"]["metal_member"][
                            "resource_observation"
                        ]["borrowed_outer_residual"]
                        is args.instruction_input_metal_borrow_outer_residual
                        and sample["instruction_input"]["metal_member"][
                            "resource_observation"
                        ]["allocation"]["owned_device_bytes"]
                        == (
                            instruction_input_sequence_auxiliary_storage_bytes(
                                args.log_n
                            )
                            if args.instruction_input_metal_borrow_outer_residual
                            else instruction_input_sequence_storage_bytes(args.log_n)
                        )
                        and sample["instruction_input"]["metal_member"][
                            "resource_observation"
                        ]["allocation"]["reused_device_bytes"]
                        == (
                            96 * (1 << args.log_n)
                            if args.instruction_input_metal_borrow_outer_residual
                            else 0
                        )
                        and (
                            sample["instruction_input"]["metal_member"][
                                "resource_observation"
                            ]["outer_residual_transfer"]
                            is not None
                        )
                        is args.instruction_input_metal_borrow_outer_residual
                        and (
                            not args.instruction_input_metal_borrow_outer_residual
                            or (
                                sample["instruction_input"]["metal_member"][
                                    "resource_observation"
                                ]["storage_initialization"]["buffer_identities"][:2]
                                == [
                                    sample["instruction_input"]["metal_member"][
                                        "row_lifecycle"
                                    ]["residual_storage_id"]
                                ]
                                * 2
                                and sample["instruction_input"]["metal_member"][
                                    "resource_observation"
                                ]["dense_ranges"]
                                == {
                                    "dense_a_offset_bytes": 0,
                                    "dense_a_length_bytes": 64 * (1 << args.log_n),
                                    "dense_b_offset_bytes": 64 * (1 << args.log_n),
                                    "dense_b_length_bytes": 32 * (1 << args.log_n),
                                }
                            )
                        )
                    )
                    for sample in attributions
                ),
                "instruction_input_native_primer_exact_and_protocol_inert": all(
                    sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["native_primer"]["source_elements"]
                    == 64
                    and sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["native_primer"]["e_in_elements"]
                    == 1
                    and sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["native_primer"]["e_out_elements"]
                    == 32
                    and sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["native_primer"]["command_committed"]
                    is True
                    and sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["native_primer"]["command_completed"]
                    is True
                    and sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["native_primer"]["produced_zero"]
                    is True
                    and sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["native_primer"]["protocol_state_advanced"]
                    is False
                    for sample in attributions
                ),
                "instruction_input_native_primer_completed_before_join": all(
                    sample["instruction_input"]["metal_member"][
                        "resource_observation"
                    ]["native_primer"]["completed_before_join"]
                    is True
                    for sample in attributions
                ),
                "instruction_input_local_gate": metrics[
                    "instruction_input_kernel_service_decision"
                ]["clears"],
                "booleanity_address_cpu_control": all(
                    not any(
                        sample["booleanity_address"]["optimized_member"][
                            "metal_counts"
                        ].values()
                    )
                    and sample["booleanity_address"]["optimized_member"][
                        "row_lifecycle"
                    ]
                    is None
                    for sample in attributions
                ),
                "booleanity_address_cpu_row_source_attributed": all(
                    sample["booleanity_address"]["optimized_member"]["components"][
                        "row_source_us"
                    ]
                    > 0.0
                    and sample["booleanity_address"]["optimized_member"]["components"][
                        "normalized_member_us"
                    ]
                    < sample["booleanity_address"]["optimized_member"]["components"][
                        "member_us"
                    ]
                    and sample["booleanity_address"]["metal_member"]["components"][
                        "row_source_us"
                    ]
                    == 0.0
                    and sample["booleanity_address"]["metal_member"]["components"][
                        "normalized_member_us"
                    ]
                    == sample["booleanity_address"]["metal_member"]["components"][
                        "member_us"
                    ]
                    for sample in attributions
                ),
                "booleanity_address_metal_backend_exercised": all(
                    sample["booleanity_address"]["metal_member"]["metal_counts"]
                    == (
                        {
                            "prepare": 1,
                            "sequence": 1,
                            "dispatch": 1,
                            "readback": 1,
                        }
                        if args.booleanity_address_metal_implementation == "packed-hot"
                        else {
                            "prepare": 1,
                            "sequence_prepare": 1,
                            "allocation_plan": 1,
                            "dispatch": 1,
                            "readback": 1,
                        }
                    )
                    for sample in attributions
                ),
                "booleanity_address_resident_rows_reused": all(
                    (lifecycle := sample["booleanity_address"]["metal_member"]["row_lifecycle"])
                    is not None
                    and (
                        lifecycle["source_rows_storage_id"] > 0
                        and lifecycle["stage6a"]
                        == {"row_allocations": 0, "row_upload_bytes": 0}
                        and lifecycle["stage6b"]
                        == {"row_allocations": 0, "row_upload_bytes": 0}
                        if args.booleanity_address_metal_implementation == "packed-hot"
                        else lifecycle["stage5_storage_id"]
                        == lifecycle["stage6a_storage_id"]
                        == lifecycle["stage6b_storage_id"]
                        and lifecycle["stage6a"]
                        == {"row_allocations": 0, "row_upload_bytes": 0}
                        and lifecycle["stage6b"]
                        == {"row_allocations": 0, "row_upload_bytes": 0}
                    )
                    for sample in attributions
                ),
                "booleanity_address_working_set_admitted": all(
                    (
                        observation := sample["booleanity_address"]["metal_member"][
                            "resource_observation"
                        ]
                    )
                    is not None
                    and (
                        observation["sequence"]["current_device_bytes"]
                        + observation["sequence"]["owned_bytes"]
                        <= observation["sequence"]["recommended_device_bytes"]
                        if args.booleanity_address_metal_implementation == "packed-hot"
                        else observation["allocation"]["current_device_bytes"]
                        + observation["allocation"]["planned_device_bytes"]
                        <= observation["allocation"]["recommended_device_bytes"]
                    )
                    for sample in attributions
                ),
                "booleanity_address_readback_exact": all(
                    sample["booleanity_address"]["metal_member"][
                        "resource_observation"
                    ]["readback"]
                    == {"elements": 29 * 256, "bytes": 29 * 256 * 16, "readbacks": 1}
                    for sample in attributions
                ),
                "booleanity_address_dispatch_exact": all(
                    sample["booleanity_address"]["metal_member"][
                        "resource_observation"
                    ]["dispatch"]["command_buffers"]
                    == 1
                    and (
                        sample["booleanity_address"]["metal_member"][
                            "resource_observation"
                        ]["dispatch"]["dispatches"]
                        == 3
                        if args.booleanity_address_metal_implementation == "packed-hot"
                        else sample["booleanity_address"]["metal_member"][
                            "resource_observation"
                        ]["dispatch"]["tile_dispatches"]
                        == (29 + args.booleanity_address_metal_selectors_per_tile - 1)
                        // args.booleanity_address_metal_selectors_per_tile
                        and sample["booleanity_address"]["metal_member"][
                            "resource_observation"
                        ]["dispatch"]["finalize_dispatches"]
                        == (29 + args.booleanity_address_metal_selectors_per_tile - 1)
                        // args.booleanity_address_metal_selectors_per_tile
                    )
                    for sample in attributions
                ),
                "booleanity_address_command_completed": all(
                    sample["booleanity_address"]["metal_member"][
                        "resource_observation"
                    ]["dispatch"]["command_completed"]
                    is True
                    for sample in attributions
                ),
                "booleanity_address_local_gate": metrics[
                    "booleanity_address_phase_decision"
                ]["clears"],
                "hamming_weight_cpu_control": all(
                    not any(
                        sample["hamming_weight"]["optimized_member"][
                            "metal_counts"
                        ].values()
                    )
                    and sample["hamming_weight"]["optimized_member"]["row_lifecycle"]
                    is None
                    for sample in attributions
                ),
                "hamming_weight_cpu_row_source_attributed": all(
                    sample["hamming_weight"]["optimized_member"]["components"][
                        "row_source_us"
                    ]
                    > 0.0
                    and sample["hamming_weight"]["optimized_member"]["components"][
                        "normalized_member_us"
                    ]
                    < sample["hamming_weight"]["optimized_member"]["components"][
                        "member_us"
                    ]
                    and sample["hamming_weight"]["metal_member"]["components"][
                        "row_source_us"
                    ]
                    == 0.0
                    and sample["hamming_weight"]["metal_member"]["components"][
                        "normalized_member_us"
                    ]
                    == sample["hamming_weight"]["metal_member"]["components"][
                        "member_us"
                    ]
                    for sample in attributions
                ),
                "hamming_weight_metal_backend_exercised": all(
                    sample["hamming_weight"]["metal_member"]["metal_counts"]
                    == (
                        {"sequence": 1, "dispatch": 1, "readback": 1}
                        if args.hamming_weight_metal_implementation == "retained-hot"
                        else {
                            "prepare": 1,
                            "sequence_prepare": 1,
                            "allocation_plan": 1,
                            "dispatch": 1,
                            "readback": 1,
                        }
                    )
                    for sample in attributions
                ),
                "hamming_weight_resident_rows_reused": all(
                    (lifecycle := sample["hamming_weight"]["metal_member"]["row_lifecycle"])
                    is not None
                    and (
                        lifecycle["kind"] == "metal_hamming_hot"
                        and lifecycle["source_rows_storage_id"] > 0
                        and lifecycle["hot_rows_storage_id"] > 0
                        and all(
                            lifecycle[stage]
                            == {"row_allocations": 0, "row_upload_bytes": 0}
                            for stage in ("stage6a", "stage6b", "stage6b_retain")
                        )
                        and lifecycle["stage7"]["row_allocations"] == 0
                        and lifecycle["stage7"]["row_upload_bytes"] == 0
                        if args.hamming_weight_metal_implementation == "retained-hot"
                        else lifecycle["kind"] == "metal_hamming_resident"
                        and lifecycle["stage5_storage_id"]
                        == lifecycle["stage6a_storage_id"]
                        == lifecycle["stage6b_storage_id"]
                        == lifecycle["stage6b_retain_storage_id"]
                        == lifecycle["stage7_storage_id"]
                        and all(
                            lifecycle[stage]
                            == {"row_allocations": 0, "row_upload_bytes": 0}
                            for stage in (
                                "stage6a",
                                "stage6b",
                                "stage6b_retain",
                                "stage7",
                            )
                        )
                    )
                    for sample in attributions
                ),
                "hamming_weight_zero_row_upload": all(
                    (
                        sample["hamming_weight"]["metal_member"]["row_lifecycle"][
                            "stage6b_retain"
                        ]["row_upload_bytes"]
                        == 0
                        and sample["hamming_weight"]["metal_member"]["row_lifecycle"][
                            "stage7"
                        ]["row_upload_bytes"]
                        == 0
                        if args.hamming_weight_metal_implementation == "retained-hot"
                        else sample["hamming_weight"]["metal_member"][
                            "resource_observation"
                        ]["sequence"]["row_upload_bytes"]
                        == 0
                        and sample["hamming_weight"]["metal_member"]["row_lifecycle"][
                            "stage7"
                        ]["row_upload_bytes"]
                        == 0
                    )
                    for sample in attributions
                ),
                "hamming_weight_terminal_carry_removed": all(
                    (
                        sample["hamming_weight"]["metal_member"]["row_lifecycle"][
                            "stage7"
                        ]["terminal_consumer"]
                        is True
                        and sample["hamming_weight"]["metal_member"]["row_lifecycle"][
                            "stage7"
                        ]["terminal_carry_removed"]
                        is True
                        if args.hamming_weight_metal_implementation == "retained-hot"
                        else sample["hamming_weight"]["metal_member"]["row_lifecycle"][
                            "terminal_consumer"
                        ]
                        is True
                        and sample["hamming_weight"]["metal_member"]["row_lifecycle"][
                            "terminal_carry_removed"
                        ]
                        is True
                    )
                    for sample in attributions
                ),
                "hamming_weight_k256_schedule_exact": all(
                    (
                        sample["hamming_weight"]["metal_member"][
                            "resource_observation"
                        ]["sequence"]["output_fields"]
                        == 29 * 256
                        and sample["hamming_weight"]["metal_member"][
                            "resource_observation"
                        ]["sequence"]["encoders"]
                        == 10
                        if args.hamming_weight_metal_implementation == "retained-hot"
                        else sample["hamming_weight"]["metal_member"][
                            "resource_observation"
                        ]["sequence"]["polys"]
                        == 29
                        and sample["hamming_weight"]["metal_member"][
                            "resource_observation"
                        ]["sequence"]["k"]
                        == 256
                    )
                    for sample in attributions
                ),
                "hamming_weight_working_set_admitted": all(
                    (
                        observation := sample["hamming_weight"]["metal_member"][
                            "resource_observation"
                        ]
                    )
                    is not None
                    and (
                        observation["sequence"]["current_device_bytes"]
                        + observation["sequence"]["owned_bytes"]
                        <= observation["sequence"]["recommended_device_bytes"]
                        if args.hamming_weight_metal_implementation == "retained-hot"
                        else observation["allocation"]["current_device_bytes"]
                        + observation["allocation"]["planned_device_bytes"]
                        <= observation["allocation"]["recommended_device_bytes"]
                    )
                    for sample in attributions
                ),
                "hamming_weight_readback_exact": all(
                    sample["hamming_weight"]["metal_member"]["resource_observation"][
                        "readback"
                    ]
                    == {"elements": 29 * 256, "bytes": 29 * 256 * 16, "readbacks": 1}
                    for sample in attributions
                ),
                "hamming_weight_dispatch_exact": all(
                    sample["hamming_weight"]["metal_member"]["resource_observation"][
                        "dispatch"
                    ]["command_buffers"]
                    == 1
                    and sample["hamming_weight"]["metal_member"][
                        "resource_observation"
                    ]["dispatch"]["tile_dispatches"]
                    == (
                        5
                        if args.hamming_weight_metal_implementation == "retained-hot"
                        else (29 + args.hamming_weight_metal_selectors_per_tile - 1)
                        // args.hamming_weight_metal_selectors_per_tile
                    )
                    and sample["hamming_weight"]["metal_member"][
                        "resource_observation"
                    ]["dispatch"]["finalize_dispatches"]
                    == (
                        5
                        if args.hamming_weight_metal_implementation == "retained-hot"
                        else (29 + args.hamming_weight_metal_selectors_per_tile - 1)
                        // args.hamming_weight_metal_selectors_per_tile
                    )
                    for sample in attributions
                ),
                "hamming_weight_command_completed": all(
                    sample["hamming_weight"]["metal_member"]["resource_observation"][
                        "dispatch"
                    ]["command_completed"]
                    is True
                    for sample in attributions
                ),
                "hamming_weight_local_gate": metrics[
                    "hamming_weight_claim_reduction_decision"
                ]["clears"],
                "booleanity_hamming_family_local_gate": metrics[
                    "booleanity_hamming_family_decision"
                ]["clears"],
                "product_remainder_local_gate": metrics[
                    "product_remainder_decision"
                ]["clears"],
                "product_uniskip_cpu_control": all(
                    sample["product_uniskip"]["optimized"]["seam_counts"]
                    == {"prepare": 2, "first_round_poly": 2}
                    and sample["product_uniskip"]["optimized"]["metal_counts"]
                    == {"standalone": 0, "carrier": 0}
                    and sample["product_uniskip"]["optimized"][
                        "resource_observation"
                    ]
                    is None
                    for sample in attributions
                ),
                "product_uniskip_execution_path_exact": all(
                    sample["product_uniskip"]["metal"]["seam_counts"]
                    == {"prepare": 1, "first_round_poly": 1}
                    and sample["product_uniskip"]["metal"]["metal_counts"]
                    == (
                        {"standalone": 0, "carrier": 1}
                        if args.product_uniskip_outer_carrier
                        else {"standalone": 1, "carrier": 0}
                    )
                    for sample in attributions
                ),
                "product_uniskip_carrier_provenance_exact": all(
                    (
                        sample["product_uniskip"]["metal"][
                            "resource_observation"
                        ]["source_rows_storage_id"]
                        == sample["outer_remainder"]["metal_member"][
                            "product_uniskip_carrier"
                        ]["source_rows_storage_id"]
                        and sample["product_uniskip"]["metal"][
                            "resource_observation"
                        ]["product_rows_storage_id"]
                        == sample["instruction_claim"]["metal"][
                            "resource_observation"
                        ]["producer_rows_storage_id"]
                        if args.product_uniskip_outer_carrier
                        else sample["product_uniskip"]["metal"][
                            "resource_observation"
                        ]["path"]
                        == "standalone"
                    )
                    for sample in attributions
                ),
                "outer_product_family_local_gate": metrics[
                    "outer_product_family_decision"
                ]["clears"],
                "instruction_claim_cpu_control": all(
                    sample["instruction_claim"]["optimized"][
                        "resource_observation"
                    ]
                    is None
                    and not any(
                        sample["instruction_claim"]["optimized"][
                            "metal_counts"
                        ].values()
                    )
                    for sample in attributions
                ),
                "instruction_claim_metal_backend_exercised": all(
                    sample["instruction_claim"]["metal"]["metal_counts"]
                    == {
                        "first_message_submit": 1,
                        "prepare": 1,
                        "first_message_join": 1,
                        "bind_and_message": args.log_n - 1,
                        "output_claims": 1,
                    }
                    for sample in attributions
                ),
                "instruction_claim_async_lifecycle_exact": all(
                    (observation := sample["instruction_claim"]["metal"][
                        "resource_observation"
                    ])
                    is not None
                    and observation["command_committed"] is True
                    and observation["command_completed"] is True
                    and observation["submit_wall_ns"] > 0
                    and observation["overlap_wall_ns"] > 0
                    and observation["join_wall_ns"] > 0
                    and observation["initial_gpu_active_ns"] > 0
                    and observation["lifecycle_wall_ns"] > 0
                    for sample in attributions
                ),
                "instruction_claim_product_rows_reused": all(
                    (observation := sample["instruction_claim"]["metal"][
                        "resource_observation"
                    ])
                    is not None
                    and observation["resident_rows_storage_id"]
                    == observation["producer_rows_storage_id"]
                    and observation["lookup_rows_storage_id"]
                    != observation["resident_rows_storage_id"]
                    and observation["row_upload_bytes"] == 0
                    and observation["round_device_buffer_allocations"] == 0
                    for sample in attributions
                ),
                "instruction_claim_local_gate": metrics[
                    "instruction_claim_reduction_critical_path_decision"
                ]["clears"],
                "product_instruction_claim_family_local_gate": metrics[
                    "product_instruction_claim_family_decision"
                ]["clears"],
                "outer_remainder_cpu_control": all(
                    sample["outer_remainder"]["optimized_config"] is None
                    and not any(
                        sample["outer_remainder"]["optimized_member"][
                            "metal_counts"
                        ].values()
                    )
                    and sample["outer_remainder"]["optimized_member"][
                        "resource_observation"
                    ]
                    is None
                    for sample in attributions
                ),
                "outer_remainder_metal_backend_exercised": all(
                    sample["outer_remainder"]["metal_member"]["metal_counts"][
                        "first_message"
                    ]
                    == 1
                    and sample["outer_remainder"]["metal_member"]["metal_counts"][
                        "output_claims"
                    ]
                    == 1
                    for sample in attributions
                ),
                "outer_remainder_round_topology_exact": all(
                    all(
                        sample["outer_remainder"][f"{backend}_member"][
                            "outer_counts"
                        ]
                        == {
                            "complete_member": 1,
                            "sumcheck_round": args.log_n + 1,
                            "host_fiat_shamir": args.log_n + 1,
                        }
                        for backend in ("optimized", "metal")
                    )
                    for sample in attributions
                ),
                "outer_remainder_resident_lifecycle_exact": all(
                    sample["outer_remainder"]["metal_member"]["row_lifecycle"]
                    is not None
                    for sample in attributions
                ),
                "outer_remainder_working_set_admitted": all(
                    sample["outer_remainder"]["metal_member"][
                        "resource_observation"
                    ]["storage"]["admitted"]
                    is True
                    for sample in attributions
                ),
                "outer_remainder_readback_exact": all(
                    sample["outer_remainder"]["metal_member"][
                        "resource_observation"
                    ]["readback"]
                    == {
                        "readbacks": 1,
                        "elements": 2 * (1 << args.outer_remainder_metal_cutoff_log2),
                        "bytes": 2
                        * (1 << args.outer_remainder_metal_cutoff_log2)
                        * 16,
                    }
                    for sample in attributions
                ),
                "outer_remainder_opening_output_exact": all(
                    sample["outer_remainder"]["metal_member"][
                        "resource_observation"
                    ]["output"]["readbacks"]
                    == 1
                    and sample["outer_remainder"]["metal_member"][
                        "resource_observation"
                    ]["output"]["output_elements"]
                    == (37 if args.product_uniskip_outer_carrier else 35)
                    and sample["outer_remainder"]["metal_member"][
                        "resource_observation"
                    ]["output"]["readback_bytes"]
                    == (37 if args.product_uniskip_outer_carrier else 35) * 16
                    and sample["outer_remainder"]["metal_member"][
                        "resource_observation"
                    ]["output"]["row_upload_bytes"]
                    == 0
                    for sample in attributions
                ),
                "outer_remainder_zero_member_allocations": all(
                    sample["outer_remainder"]["metal_member"][
                        "resource_observation"
                    ]["sequence"]["sequence_device_buffer_allocations"]
                    == 0
                    and sample["outer_remainder"]["metal_member"][
                        "resource_observation"
                    ]["sequence"]["round_device_buffer_allocations"]
                    == 0
                    for sample in attributions
                ),
                "outer_remainder_local_gate": metrics[
                    "outer_remainder_decision"
                ]["clears"],
            },
            "resources": {
                "metal_piop_seconds": sum(pair["metal_us"] for pair in pairs)
                / 1_000_000.0,
                "optimized_max_rss_bytes": [
                    pair["arms"]["optimized"]["max_rss_bytes"] for pair in pair_records
                ],
                "metal_max_rss_bytes": [
                    pair["arms"]["metal"]["max_rss_bytes"] for pair in pair_records
                ],
            },
            "fingerprint": {
                **source,
                "binary_sha256": binary_sha256,
                "build_command": build_command,
                "machine": platform.machine(),
                "platform": platform.platform(),
                "workload": args.workload,
                "log_n": args.log_n,
                "local_kernel": local_kernel["name"],
                "rayon_threads": PRODUCTION_RAYON_THREADS,
                "instruction_ra_materialize_width": args.instruction_ra_materialize_width,
                "instruction_ra_reuse_inverse": args.instruction_ra_reuse_inverse,
                "bytecode_metal_message_threads": args.bytecode_metal_message_threads,
                "bytecode_metal_transition_threads": args.bytecode_metal_transition_threads,
                "bytecode_metal_max_threadgroups": args.bytecode_metal_max_threadgroups,
                "bytecode_metal_cutoff_log2": args.bytecode_metal_cutoff_log2,
                "bytecode_metal_cutoff_elements": 1 << args.bytecode_metal_cutoff_log2,
                "bytecode_metal_trace_cutoff_log2": args.bytecode_metal_trace_cutoff_log2,
                "bytecode_metal_trace_cutoff_elements": 1
                << args.bytecode_metal_trace_cutoff_log2,
                "bytecode_cpu_tail_algebra": "q10",
                "instruction_input_metal_native_message_threads": args.instruction_input_metal_native_message_threads,
                "instruction_input_metal_native_transition_threads": args.instruction_input_metal_native_transition_threads,
                "instruction_input_metal_dense_transition_threads": args.instruction_input_metal_dense_transition_threads,
                "instruction_input_metal_cutoff_log2": args.instruction_input_metal_cutoff_log2,
                "instruction_input_metal_cutoff_elements": 1
                << args.instruction_input_metal_cutoff_log2,
                "instruction_input_metal_trace_cutoff_log2": args.instruction_input_metal_trace_cutoff_log2,
                "instruction_input_metal_trace_cutoff_elements": 1
                << args.instruction_input_metal_trace_cutoff_log2,
                "instruction_input_metal_borrow_outer_residual": args.instruction_input_metal_borrow_outer_residual,
                "instruction_input_storage_initialization": "minimal",
                "instruction_input_native_primer": "async",
                "booleanity_address_metal_inner_log2": args.booleanity_address_metal_inner_log2,
                "booleanity_address_metal_selectors_per_tile": args.booleanity_address_metal_selectors_per_tile,
                "booleanity_address_metal_tile_threads": args.booleanity_address_metal_tile_threads,
                "booleanity_address_metal_finalize_threads": args.booleanity_address_metal_finalize_threads,
                "booleanity_address_metal_trace_cutoff_log2": args.booleanity_address_metal_trace_cutoff_log2,
                "booleanity_address_metal_trace_cutoff_elements": 1
                << args.booleanity_address_metal_trace_cutoff_log2,
                "booleanity_address_metal_implementation": args.booleanity_address_metal_implementation,
                "hamming_weight_metal_inner_log2": args.hamming_weight_metal_inner_log2,
                "hamming_weight_metal_selectors_per_tile": args.hamming_weight_metal_selectors_per_tile,
                "hamming_weight_metal_tile_threads": args.hamming_weight_metal_tile_threads,
                "hamming_weight_metal_finalize_threads": args.hamming_weight_metal_finalize_threads,
                "hamming_weight_metal_trace_cutoff_log2": args.hamming_weight_metal_trace_cutoff_log2,
                "hamming_weight_metal_trace_cutoff_elements": 1
                << args.hamming_weight_metal_trace_cutoff_log2,
                "hamming_weight_metal_implementation": args.hamming_weight_metal_implementation,
                "outer_remainder_metal_materialize_threads": args.outer_remainder_metal_materialize_threads,
                "outer_remainder_metal_transition_threads": args.outer_remainder_metal_transition_threads,
                "outer_remainder_metal_output_threads": args.outer_remainder_metal_output_threads,
                "outer_remainder_metal_cutoff_log2": args.outer_remainder_metal_cutoff_log2,
                "outer_remainder_metal_cutoff_elements": 1
                << args.outer_remainder_metal_cutoff_log2,
                "outer_remainder_metal_trace_cutoff_log2": args.outer_remainder_metal_trace_cutoff_log2,
                "outer_remainder_metal_trace_cutoff_elements": 1
                << args.outer_remainder_metal_trace_cutoff_log2,
                "outer_remainder_metal_binding_plan": args.outer_remainder_metal_binding_plan,
                "product_uniskip_outer_carrier": args.product_uniskip_outer_carrier,
                "span": PIOP_SPAN,
                "orders": orders,
            },
            "artifacts": str(artifact_dir),
        }
        encoded = json.dumps(output, sort_keys=True)
        (artifact_dir / "result.json").write_text(encoded + "\n")
        print(encoded)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    with evaluator_lock({"direct_evaluator": "metal_piop_eval"}):
        raise SystemExit(main())
