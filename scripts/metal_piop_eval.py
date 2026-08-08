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


SCHEMA_VERSION = 17
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
BYTECODE_ADDRESS_KERNEL = "BytecodeReadRafAddressPhase"
BYTECODE_ADDRESS_COMPONENTS = (
    "prepare",
    "prove_round",
    "finish_rounds",
    "output_claims",
)
BYTECODE_ADDRESS_MIN_SPEEDUP = 5.0
METAL_BYTECODE_ADDRESS_ROUTE = "MetalBytecodeReadRafAddress::route"
METAL_BYTECODE_ADDRESS_CARRIER_PUBLISH = (
    "MetalBytecodeReadRafAddress::carrier_publish"
)
METAL_BYTECODE_ADDRESS_FUSED_TOPOLOGY_PREPARE = (
    "MetalBytecodeReadRafAddress::fused_topology_prepare"
)
METAL_BYTECODE_ADDRESS_FUSED_CARRIER_PUBLISH = (
    "MetalBytecodeReadRafAddress::fused_carrier_publish"
)
METAL_BYTECODE_ADDRESS_FUSED_ROUTE = "address_major_fused_stage1_grouped_v1"
BYTECODE_ADDRESS_MAX_ADMITTED_DESCRIPTORS_PER_CHUNK = 512
BYTECODE_ADDRESS_MAX_ADMITTED_PIVOTS_PER_CHUNK = 15
METAL_BYTECODE_ADDRESS_PREPARE = (
    "MetalBytecodeReadRafAddress::address_major_prepare"
)
METAL_BYTECODE_ADDRESS_JOIN = "MetalBytecodeReadRafAddress::address_major_join"
METAL_BYTECODE_ADDRESS_COMPLETE = (
    "MetalBytecodeReadRafAddress::address_major_complete"
)
METAL_BYTECODE_ADDRESS_SHADOW_PREPARE = (
    "MetalBytecodeReadRafAddress::shadow_prepare"
)
METAL_BYTECODE_ADDRESS_SHADOW_JOIN = "MetalBytecodeReadRafAddress::shadow_join"
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
    "registers_claim_alias_publish",
)
INSTRUCTION_INPUT_MIN_SPEEDUP = 5.0
INSTRUCTION_READ_RAF_KERNEL = "InstructionReadRaf"
INSTRUCTION_READ_RAF_MIN_SPEEDUP = 5.0
INSTRUCTION_READ_RAF_COMPONENTS = (
    "prepare",
    "prove_round",
    "finish_rounds",
    "output_claims",
)
INSTRUCTION_READ_RAF_STAGE1_SOURCE = (
    "MetalInstructionReadRaf::stage1_source_publish"
)
INSTRUCTION_READ_RAF_STAGE1_SCATTER = (
    "MetalInstructionReadRaf::stage1_grouped_scatter"
)
INSTRUCTION_READ_RAF_STAGE1_SEQUENCE = (
    "MetalInstructionReadRaf::stage1_grouped_sequence_prepare"
)
INSTRUCTION_READ_RAF_METAL_PHASES = (
    "address_round",
    "resident_first_message",
    "resident_handoff",
    "resident_round",
    "readback",
)
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
    "registers_claim_carrier_park",
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
REGISTERS_CLAIM_KERNEL = "RegistersClaimReduction"
REGISTERS_CLAIM_MIN_SPEEDUP = 5.0
RAM_READ_WRITE_KERNEL = "RamReadWriteChecking"
RAM_READ_WRITE_MIN_SPEEDUP = 5.0
RAM_HAMMING_KERNEL = "RamHammingBooleanity"
RAM_HAMMING_MIN_SPEEDUP = 5.0
RAM_CYCLE_FAMILY_KERNEL = "RamCycleFamily"
RAM_RAF_EVALUATION_KERNEL = "RamRafEvaluation"
RAM_VAL_CHECK_KERNEL = "RamValCheck"
RAM_RA_CLAIM_REDUCTION_KERNEL = "RamRaClaimReduction"
RAM_RA_VIRTUALIZATION_KERNEL = "RamRaVirtualization"
RAM_CYCLE_FAMILY_MIN_SPEEDUP = 5.0
RAM_CYCLE_FAMILY_LOG_K = 13
RAM_CYCLE_FAMILY_CHARGE_MODEL = "six_raw_members_plus_witness_prepare_once_v1"
RAM_CYCLE_FAMILY_MEMBERS = (
    ("raf_evaluation", RAM_RAF_EVALUATION_KERNEL),
    ("read_write", RAM_READ_WRITE_KERNEL),
    ("val_check", RAM_VAL_CHECK_KERNEL),
    ("ra_claim_reduction", RAM_RA_CLAIM_REDUCTION_KERNEL),
    ("hamming_booleanity", RAM_HAMMING_KERNEL),
    ("ra_virtualization", RAM_RA_VIRTUALIZATION_KERNEL),
)
RAM_CYCLE_FAMILY_PAIR_MEMBERS = (
    ("raf_evaluation", "ram_raf_evaluation"),
    ("read_write", "ram_read_write"),
    ("val_check", "ram_val_check"),
    ("ra_claim_reduction", "ram_ra_claim_reduction"),
    ("hamming_booleanity", "ram_hamming"),
    ("ra_virtualization", "ram_ra_virtualization"),
)
METAL_RAM_CYCLE_FAMILY_WITNESS_PREPARE = (
    "MetalRamCycleFamily::witness_prepare"
)
METAL_RAM_CYCLE_FAMILY_OWNER = "MetalRamCycleFamily::owner_prepare"
METAL_RAM_CYCLE_FAMILY_TERMINAL_TAKE = "MetalRamCycleFamily::terminal_take"
METAL_RAM_VAL_CHECK_ROUTE = "MetalRamValCheck::route"
METAL_RAM_RA_CLAIM_REDUCTION_ROUTE = "MetalRamRaClaimReduction::route"
METAL_RAM_RA_VIRTUALIZATION_ROUTE = "MetalRamRaVirtualization::route"
RAM_CYCLE_FAMILY_SCHEMA_VERSION = 3
RAM_HAMMING_PRODUCT_CAP = 1_000_000
OPTIMIZED_HAMMING_WEIGHT_ROW_SOURCE = (
    "OptimizedHammingWeightClaimReduction::row_source"
)
SUMCHECK_ROUND_SPAN = "sumcheck_round"
SUMCHECK_HOST_FIAT_SHAMIR_SPAN = "sumcheck_host_fiat_shamir"
OPTIMIZED_BOOLEANITY_ADDRESS_ROW_SOURCE = "OptimizedBooleanityAddress::row_source"
PRODUCTION_RAYON_THREADS = 16
PIOP_MIN_SPEEDUP = 5.0
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
    BYTECODE_ADDRESS_KERNEL: {
        "name": BYTECODE_ADDRESS_KERNEL,
        "metric": "bytecode_read_raf_address_speedup",
        "paired_metric": "paired_bytecode_read_raf_address_speedups",
        "backend_prefix": "MetalBytecodeReadRafAddress::",
    },
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
    "RegistersClaimReduction": {
        "name": REGISTERS_CLAIM_KERNEL,
        "metric": "registers_claim_reduction_member_speedup",
        "paired_metric": "paired_registers_claim_reduction_member_speedups",
        "backend_prefix": "MetalRegistersClaimReduction::",
    },
    "RamCycleFamily": {
        "name": RAM_CYCLE_FAMILY_KERNEL,
        "metric": "ram_cycle_family_speedup",
        "paired_metric": "paired_ram_cycle_family_speedups",
        "backend_prefix": "MetalRamCycleFamily::",
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
BYTECODE_ADDRESS_METAL_CONFIG_RE = re.compile(
    r"^BYTECODE_ADDRESS_METAL_CONFIG backend=metal "
    r"implementation=(?P<implementation>cpu|csr-shadow|address-major-shadow|address-major) "
    r"trace_cutoff=(?P<trace_cutoff>\d+) "
    r"outer_tiles=(?P<outer_tiles>\d+)$"
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
INSTRUCTION_READ_RAF_METAL_CONFIG_RE = re.compile(
    r"^INSTRUCTION_READ_RAF_METAL_CONFIG backend=metal "
    r"address_cutoff=(?P<address_cutoff>\d+) cutoff=(?P<cutoff>\d+) "
    r"stage1_scatter_threads=(?P<stage1_scatter_threads>\d+)$"
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
    r"product_uniskip_carrier=(?P<product_uniskip_carrier>true|false) "
    r"registers_claim_carrier=(?P<registers_claim_carrier>true|false)$"
)
REGISTERS_CLAIM_METAL_CONFIG_RE = re.compile(
    r"^REGISTERS_CLAIM_METAL_CONFIG backend=metal "
    r"implementation=(?P<implementation>cpu|direct-hybrid|outer-carrier-alias-hybrid) "
    r"trace_cutoff=(?P<trace_cutoff>\d+)$"
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


def validate_bytecode_address_stdout(
    stdout: str,
    backend: str,
    implementation: str,
    outer_tiles: int,
    trace_cutoff_log2: int,
) -> Optional[dict[str, Any]]:
    configs = [
        match
        for line in stdout.splitlines()
        if (match := BYTECODE_ADDRESS_METAL_CONFIG_RE.fullmatch(line)) is not None
    ]
    if backend == "optimized":
        if configs:
            raise ValueError(
                "optimized evaluator emitted a Bytecode address Metal config"
            )
        return None
    if len(configs) != 1:
        raise ValueError(
            "Metal evaluator must emit exactly one Bytecode address Metal config"
        )
    raw = configs[0].groupdict()
    config = {
        "implementation": raw["implementation"],
        "trace_cutoff": int(raw["trace_cutoff"]),
        "outer_tiles": int(raw["outer_tiles"]),
    }
    expected = {
        "implementation": implementation,
        "trace_cutoff": 1 << trace_cutoff_log2,
        "outer_tiles": outer_tiles,
    }
    if config != expected:
        raise ValueError(f"unexpected Bytecode address Metal config: {config}")
    return config


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


def validate_instruction_read_raf_stdout(
    stdout: str,
    backend: str,
    scatter_threads: int,
    address_cutoff_log2: int = 25,
    cutoff_log2: int = 16,
) -> Optional[dict[str, int]]:
    configs = [
        match
        for line in stdout.splitlines()
        if (match := INSTRUCTION_READ_RAF_METAL_CONFIG_RE.fullmatch(line))
        is not None
    ]
    if backend == "optimized":
        if configs:
            raise ValueError(
                "optimized evaluator emitted an InstructionReadRaf Metal config"
            )
        return None
    if len(configs) != 1:
        raise ValueError(
            "Metal evaluator must emit exactly one InstructionReadRaf Metal config"
        )
    config = {name: int(value) for name, value in configs[0].groupdict().items()}
    expected = {
        "address_cutoff": 1 << address_cutoff_log2,
        "cutoff": 1 << cutoff_log2,
        "stage1_scatter_threads": scatter_threads,
    }
    if config != expected:
        raise ValueError(f"unexpected InstructionReadRaf Metal config: {config}")
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
    registers_claim_carrier: bool = False,
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
        "registers_claim_carrier": raw["registers_claim_carrier"] == "true",
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
        "registers_claim_carrier": registers_claim_carrier,
    }
    if config != expected:
        raise ValueError(f"unexpected OuterRemainder Metal config: {config}")
    return config


def validate_registers_claim_stdout(
    stdout: str,
    backend: str,
    implementation: str,
    trace_cutoff_log2: int,
) -> Optional[dict[str, Any]]:
    configs = [
        match
        for line in stdout.splitlines()
        if (match := REGISTERS_CLAIM_METAL_CONFIG_RE.fullmatch(line)) is not None
    ]
    if backend == "optimized":
        if configs:
            raise ValueError("optimized evaluator emitted a RegistersClaim Metal config")
        return None
    if len(configs) != 1:
        raise ValueError("Metal evaluator must emit exactly one RegistersClaim config")
    raw = configs[0].groupdict()
    config = {
        "implementation": raw["implementation"],
        "trace_cutoff": int(raw["trace_cutoff"]),
    }
    expected = {
        "implementation": implementation,
        "trace_cutoff": 1 << trace_cutoff_log2,
    }
    if config != expected:
        raise ValueError(f"unexpected RegistersClaim Metal config: {config}")
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


def bytecode_address_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    implementation: str = "cpu",
    outer_tiles: int = 8,
    trace_cutoff_log2: int = 26,
    stage1_source: Optional[dict[str, Any]] = None,
    stage1_scatter: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    implementations = {"cpu", "csr-shadow", "address-major-shadow", "address-major"}
    if backend not in {"optimized", "metal"}:
        raise ValueError(f"unsupported Bytecode address backend {backend!r}")
    if implementation not in implementations:
        raise ValueError(f"unsupported Bytecode address implementation {implementation!r}")
    if log_n < 15 or outer_tiles <= 0 or trace_cutoff_log2 < 0:
        raise ValueError("invalid Bytecode address evaluator geometry")

    outer_names = {
        f"{BYTECODE_ADDRESS_KERNEL}::{component}"
        for component in BYTECODE_ADDRESS_COMPONENTS
    }
    metal_names = {
        METAL_BYTECODE_ADDRESS_ROUTE,
        METAL_BYTECODE_ADDRESS_CARRIER_PUBLISH,
        METAL_BYTECODE_ADDRESS_FUSED_TOPOLOGY_PREPARE,
        METAL_BYTECODE_ADDRESS_FUSED_CARRIER_PUBLISH,
        METAL_BYTECODE_ADDRESS_PREPARE,
        METAL_BYTECODE_ADDRESS_JOIN,
        METAL_BYTECODE_ADDRESS_COMPLETE,
        METAL_BYTECODE_ADDRESS_SHADOW_PREPARE,
        METAL_BYTECODE_ADDRESS_SHADOW_JOIN,
    }
    supporting_names = {
        PIOP_SPAN,
        BACKEND_WITNESS_PREP_SPAN,
        INSTRUCTION_READ_RAF_STAGE1_SCATTER,
        "InstructionReadRaf::prepare",
    }
    intervals = strict_named_intervals(
        events, outer_names | metal_names | supporting_names
    )
    if len(intervals[PIOP_SPAN]) != 1 or len(
        intervals[BACKEND_WITNESS_PREP_SPAN]
    ) != 1:
        raise ValueError(
            "Bytecode address requires one PIOP and backend witness-prepare span"
        )
    piop = intervals[PIOP_SPAN][0]
    witness_prepare = intervals[BACKEND_WITNESS_PREP_SPAN][0]
    if witness_prepare[1] > piop[0]:
        raise ValueError("Bytecode address witness preparation overlaps PIOP")

    by_component = {
        component: sorted(intervals[f"{BYTECODE_ADDRESS_KERNEL}::{component}"])
        for component in BYTECODE_ADDRESS_COMPONENTS
    }
    outer_counts = {
        component: len(component_intervals)
        for component, component_intervals in by_component.items()
    }
    expected_outer_counts = {
        "prepare": 1,
        "prove_round": 13,
        "finish_rounds": 1,
        "output_claims": 1,
    }
    if outer_counts != expected_outer_counts:
        raise ValueError(
            f"Bytecode address member span counts {outer_counts}, "
            f"expected {expected_outer_counts}"
        )
    prepare = by_component["prepare"][0]
    rounds = by_component["prove_round"]
    finish = by_component["finish_rounds"][0]
    output = by_component["output_claims"][0]
    ordered = [prepare, *rounds, finish, output]
    if any(start < piop[0] or end > piop[1] for start, end in ordered):
        raise ValueError("a Bytecode address member span lies outside PIOP")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Bytecode address member spans overlap or are out of order")

    components = {
        "prepare_us": interval_duration_us(prepare),
        "rounds_us": [interval_duration_us(interval) for interval in rounds],
        "finish_us": interval_duration_us(finish),
        "output_claims_us": interval_duration_us(output),
    }
    components["rounds_total_us"] = sum(components["rounds_us"])
    components["member_us"] = (
        components["prepare_us"]
        + components["rounds_total_us"]
        + components["finish_us"]
        + components["output_claims_us"]
    )
    if any(
        not math.isfinite(float(components[field]))
        or float(components[field]) <= 0.0
        for field in (
            "prepare_us",
            "rounds_total_us",
            "finish_us",
            "output_claims_us",
            "member_us",
        )
    ):
        raise ValueError("Bytecode address has a non-positive member duration")

    all_span_names = {name for name, _, _ in span_intervals_us(events)}
    atom_names = {
        name
        for name in all_span_names
        if name.startswith("Metal")
        and "bytecode" in name.lower()
        and "address" in name.lower()
        and "atom" in name.lower()
    }
    if atom_names:
        raise ValueError(
            f"Bytecode address trace exercised a forbidden Atom path: {sorted(atom_names)}"
        )
    observed_metal_names = {
        name
        for name in all_span_names
        if name.startswith("MetalBytecodeReadRafAddress::")
    }
    unknown_metal_names = observed_metal_names - metal_names
    if unknown_metal_names:
        raise ValueError(
            "Bytecode address trace contains legacy or unknown Metal phases: "
            f"{sorted(unknown_metal_names)}"
        )
    metal_counts = {
        name.rsplit("::", 1)[-1]: len(intervals[name]) for name in metal_names
    }
    if backend == "optimized":
        if any(metal_counts.values()):
            raise ValueError("Bytecode address CPU control unexpectedly exercised Metal")
        return {
            "components": components,
            "outer_counts": outer_counts,
            "metal_counts": metal_counts,
            "route_observation": None,
            "topology_observation": None,
            "resource_observation": None,
        }

    expected_counts = {
        "route": 1,
        "carrier_publish": 0,
        "fused_topology_prepare": 0,
        "fused_carrier_publish": 0,
        "address_major_prepare": 0,
        "address_major_join": 0,
        "address_major_complete": 0,
        "shadow_prepare": 0,
        "shadow_join": 0,
    }
    fused_producer = False
    if implementation == "address-major":
        legacy_counts = {
            **expected_counts,
            "carrier_publish": 1,
            "address_major_prepare": 1,
            "address_major_join": 1,
            "address_major_complete": 1,
        }
        fused_counts = {
            **expected_counts,
            "fused_topology_prepare": 1,
            "fused_carrier_publish": 1,
            "address_major_prepare": 1,
            "address_major_join": 1,
            "address_major_complete": 1,
        }
        if metal_counts == fused_counts:
            expected_counts = fused_counts
            fused_producer = True
        else:
            expected_counts = legacy_counts
    elif implementation == "address-major-shadow":
        expected_counts.update({"shadow_prepare": 1, "address_major_join": 1})
    elif implementation == "csr-shadow":
        expected_counts.update({"shadow_prepare": 1, "shadow_join": 1})
    elif (
        implementation == "cpu"
        and metal_counts["fused_topology_prepare"] == 1
    ):
        expected_counts["fused_topology_prepare"] = 1
    if metal_counts != expected_counts:
        raise ValueError(
            f"Bytecode address Metal span counts {metal_counts}, expected {expected_counts}"
        )

    route_fields = {"cycles", "requested", "realized_route", "fallback_reason"}
    raw_route = exact_span_args(events, METAL_BYTECODE_ADDRESS_ROUTE, route_fields)
    route = {
        "cycles": nonnegative_trace_integer(raw_route["cycles"], "cycles"),
        "requested": trace_string(raw_route["requested"], "requested"),
        "realized_route": trace_string(
            raw_route["realized_route"], "realized_route"
        ),
        "fallback_reason": trace_string(
            raw_route["fallback_reason"], "fallback_reason"
        ),
    }
    rows = 1 << log_n
    requested = implementation.replace("-", "_")
    expected_route = {
        "cycles": rows,
        "requested": requested,
        "realized_route": (
            METAL_BYTECODE_ADDRESS_FUSED_ROUTE
            if fused_producer
            else "address_major"
            if implementation == "address-major"
            else "cpu"
        ),
        "fallback_reason": (
            "none"
            if implementation == "address-major"
            else "configured_cpu"
            if implementation == "cpu"
            else "shadow_only"
        ),
    }
    if log_n < trace_cutoff_log2 and implementation != "cpu":
        expected_route.update(
            {"realized_route": "cpu", "fallback_reason": "trace_cutoff"}
        )
    if route != expected_route:
        raise ValueError(
            f"Bytecode address route is not the configured fail-closed route: {route}"
        )
    route_interval = intervals[METAL_BYTECODE_ADDRESS_ROUTE][0]
    require_contained(route_interval, prepare, "Bytecode address route")

    topology_observation = None
    if metal_counts["fused_topology_prepare"] == 1:
        topology_fields = {
            "enabled",
            "physical_rows",
            "chunk_rows",
            "chunks",
            "descriptors",
            "descriptor_elements",
            "descriptor_bytes",
            "descriptor_storage_id",
            "pivots",
            "pivot_elements",
            "pivot_bytes",
            "pivot_storage_id",
            "chunk_offset_elements",
            "chunk_offset_bytes",
            "chunk_offset_storage_id",
            "work_items",
            "work_item_elements",
            "work_item_bytes",
            "work_item_storage_id",
            "address_offset_elements",
            "address_offset_bytes",
            "address_offset_storage_id",
            "max_descriptors_per_chunk",
            "max_pivots_per_chunk",
            "first_push_pc",
            "source_generation",
            "source_completion_serial",
            "source_rows_storage_id",
            "source_claim_storage_id",
            "topology_completion_serial",
            "shared_source_row_scans",
            "additional_source_row_scans",
            "extra_source_scans",
            "source_windows",
            "member_upload_bytes",
            "complete_overwrite",
            "covered_rows",
        }
        raw_topology = exact_span_args(
            events,
            METAL_BYTECODE_ADDRESS_FUSED_TOPOLOGY_PREPARE,
            topology_fields,
        )
        topology_observation = {
            field: nonnegative_trace_integer(value, f"fused topology {field}")
            for field, value in raw_topology.items()
            if field not in {"enabled", "complete_overwrite"}
        }
        topology_observation["enabled"] = trace_boolean(raw_topology["enabled"])
        topology_observation["complete_overwrite"] = trace_boolean(
            raw_topology["complete_overwrite"]
        )
        topology_interval = intervals[
            METAL_BYTECODE_ADDRESS_FUSED_TOPOLOGY_PREPARE
        ][0]
        require_contained(
            topology_interval,
            witness_prepare,
            "fused Bytecode address topology preparation",
        )
        topology_observation["wall_us"] = interval_duration_us(topology_interval)
        if topology_observation["wall_us"] <= 0.0:
            raise ValueError("fused Bytecode address topology timing is invalid")
        if stage1_source is None:
            raise ValueError(
                "fused Bytecode address topology is missing its Stage1 source"
            )
        physical_rows = topology_observation["physical_rows"]
        source_match = {
            "physical_rows": stage1_source["explicit_rows"],
            "source_generation": stage1_source["source_generation"],
            "source_completion_serial": stage1_source["completion_serial"],
            "source_rows_storage_id": stage1_source["row_allocation_identity"],
            "source_claim_storage_id": stage1_source["claim_allocation_identity"],
            "source_windows": stage1_source["source_windows"],
        }
        common_topology = {
            "chunk_rows": 4096,
            "shared_source_row_scans": 1,
            "additional_source_row_scans": 0,
            "extra_source_scans": 0,
            "member_upload_bytes": 0,
        }
        if (
            physical_rows <= 0
            or physical_rows > rows
            or any(
                topology_observation[field] != value
                for field, value in {**source_match, **common_topology}.items()
            )
        ):
            raise ValueError(
                "fused Bytecode address topology source or traffic is invalid"
            )
        if fused_producer:
            chunks = (physical_rows + 4095) // 4096
            descriptors = topology_observation["descriptors"]
            pivots = topology_observation["pivots"]
            work_items = topology_observation["work_items"]
            max_descriptors = topology_observation["max_descriptors_per_chunk"]
            max_pivots = topology_observation["max_pivots_per_chunk"]
            topology_ids = [
                topology_observation[field]
                for field in (
                    "descriptor_storage_id",
                    "pivot_storage_id",
                    "chunk_offset_storage_id",
                    "work_item_storage_id",
                    "address_offset_storage_id",
                )
            ]
            expected_topology = {
                "enabled": True,
                "chunks": chunks,
                "descriptor_elements": descriptors + chunks,
                "descriptor_bytes": 8 * (descriptors + chunks),
                "pivot_elements": pivots + 1,
                "pivot_bytes": 2 * (pivots + 1),
                "chunk_offset_elements": 2 * chunks,
                "chunk_offset_bytes": 8 * chunks,
                "work_item_elements": work_items,
                "work_item_bytes": 8 * work_items,
                "address_offset_elements": (1 << 13) + 1,
                "address_offset_bytes": 4 * ((1 << 13) + 1),
                "complete_overwrite": True,
                "covered_rows": physical_rows,
            }
            if (
                descriptors < chunks
                or descriptors
                > BYTECODE_ADDRESS_MAX_ADMITTED_DESCRIPTORS_PER_CHUNK * chunks
                or work_items < chunks
                or work_items > physical_rows
                or not (
                    1
                    <= max_descriptors
                    <= BYTECODE_ADDRESS_MAX_ADMITTED_DESCRIPTORS_PER_CHUNK
                )
                or max_descriptors > descriptors
                or pivots
                > BYTECODE_ADDRESS_MAX_ADMITTED_PIVOTS_PER_CHUNK * chunks
                or max_pivots > BYTECODE_ADDRESS_MAX_ADMITTED_PIVOTS_PER_CHUNK
                or max_pivots > pivots
                or topology_observation["first_push_pc"] >= 1 << 13
                or topology_observation["topology_completion_serial"] <= 0
                or any(identity <= 0 for identity in topology_ids)
                or len(set(topology_ids)) != len(topology_ids)
                or set(topology_ids)
                & {
                    topology_observation["source_rows_storage_id"],
                    topology_observation["source_claim_storage_id"],
                }
                or any(
                    topology_observation[field] != value
                    for field, value in expected_topology.items()
                )
            ):
                raise ValueError(
                    "fused Bytecode address topology receipt is invalid"
                )
        elif implementation == "cpu":
            zero_topology_fields = {
                "chunks",
                "descriptors",
                "descriptor_elements",
                "descriptor_bytes",
                "descriptor_storage_id",
                "pivots",
                "pivot_elements",
                "pivot_bytes",
                "pivot_storage_id",
                "chunk_offset_elements",
                "chunk_offset_bytes",
                "chunk_offset_storage_id",
                "work_items",
                "work_item_elements",
                "work_item_bytes",
                "work_item_storage_id",
                "address_offset_elements",
                "address_offset_bytes",
                "address_offset_storage_id",
                "max_descriptors_per_chunk",
                "max_pivots_per_chunk",
                "first_push_pc",
                "topology_completion_serial",
                "covered_rows",
            }
            if (
                topology_observation["enabled"] is not False
                or topology_observation["complete_overwrite"] is not False
                or any(topology_observation[field] != 0 for field in zero_topology_fields)
            ):
                raise ValueError(
                    "disabled Bytecode address topology control is not inert"
                )
        else:
            raise ValueError(
                "fused Bytecode address topology appeared on an unsupported route"
            )

    if implementation == "cpu":
        return {
            "components": components,
            "outer_counts": outer_counts,
            "metal_counts": metal_counts,
            "route_observation": route,
            "topology_observation": topology_observation,
            "resource_observation": None,
        }
    if log_n < trace_cutoff_log2:
        raise ValueError("Bytecode address configured route fell back at the trace cutoff")

    inner_prepare = intervals[
        METAL_BYTECODE_ADDRESS_PREPARE
        if implementation == "address-major"
        else METAL_BYTECODE_ADDRESS_SHADOW_PREPARE
    ][0]
    require_contained(inner_prepare, route_interval, "Bytecode address inner prepare")
    if implementation == "csr-shadow":
        inner_join = intervals[METAL_BYTECODE_ADDRESS_SHADOW_JOIN][0]
    else:
        inner_join = intervals[METAL_BYTECODE_ADDRESS_JOIN][0]
    require_contained(inner_join, route_interval, "Bytecode address join")
    if inner_prepare[1] > inner_join[0]:
        raise ValueError("Bytecode address prepare and join overlap or are out of order")

    if implementation != "address-major":
        return {
            "components": components,
            "outer_counts": outer_counts,
            "metal_counts": metal_counts,
            "route_observation": route,
            "topology_observation": topology_observation,
            "resource_observation": None,
        }

    publish_name = (
        METAL_BYTECODE_ADDRESS_FUSED_CARRIER_PUBLISH
        if fused_producer
        else METAL_BYTECODE_ADDRESS_CARRIER_PUBLISH
    )
    publish_interval = intervals[publish_name][0]
    complete_interval = intervals[METAL_BYTECODE_ADDRESS_COMPLETE][0]
    if fused_producer:
        if len(intervals["InstructionReadRaf::prepare"]) != 1 or len(
            intervals[INSTRUCTION_READ_RAF_STAGE1_SCATTER]
        ) != 1:
            raise ValueError(
                "fused Bytecode address requires one InstructionReadRaf scatter"
            )
        require_contained(
            publish_interval,
            intervals["InstructionReadRaf::prepare"][0],
            "fused Bytecode address carrier publication",
        )
        if intervals[INSTRUCTION_READ_RAF_STAGE1_SCATTER][0][1] > publish_interval[0]:
            raise ValueError(
                "fused Bytecode address carrier was published before scatter completion"
            )
    else:
        require_contained(
            publish_interval,
            witness_prepare,
            "Bytecode address carrier publication",
        )
    require_contained(complete_interval, inner_join, "Bytecode address completion")
    if not fused_producer and publish_interval[1] > piop[0]:
        raise ValueError("Bytecode address carrier publication extends into PIOP")

    legacy_publish_fields = {
        "cycles",
        "physical_rows",
        "work_items",
        "source_generation",
        "source_completion_serial",
        "source_rows_storage_id",
        "source_claim_storage_id",
        "source_device_registry_id",
        "source_windows",
        "carrier_completion_serial",
        "carrier_occurrence_storage_id",
        "carrier_occurrence_bytes",
        "carrier_magnitude_storage_id",
        "carrier_magnitude_bytes",
        "carrier_work_item_storage_id",
        "carrier_work_item_bytes",
        "carrier_address_offset_storage_id",
        "carrier_address_offset_bytes",
        "carrier_resident_bytes",
        "carrier_allocations",
        "producer_persistent_write_bytes",
        "producer_logical_movement_bytes",
        "producer_topology_read_bytes",
        "shared_source_row_scans",
        "additional_source_row_scans",
        "member_source_upload_bytes",
        "complete_overwrite",
        "covered_rows",
    }
    fused_publish_fields = {
        "route",
        "cycles",
        "physical_rows",
        "work_items",
        "source_generation",
        "source_completion_serial",
        "source_rows_storage_id",
        "source_claim_storage_id",
        "source_device_registry_id",
        "source_windows",
        "carrier_completion_serial",
        "carrier_occurrence_storage_id",
        "carrier_occurrence_bytes",
        "carrier_magnitude_storage_id",
        "carrier_magnitude_bytes",
        "carrier_work_item_storage_id",
        "carrier_work_item_bytes",
        "carrier_address_offset_storage_id",
        "carrier_address_offset_bytes",
        "bytecode_descriptor_storage_id",
        "bytecode_descriptor_bytes",
        "bytecode_pivot_storage_id",
        "bytecode_pivot_bytes",
        "bytecode_chunk_offset_storage_id",
        "bytecode_chunk_offset_bytes",
        "carrier_resident_bytes",
        "carrier_buffers",
        "scatter_output_allocations",
        "producer_persistent_write_bytes",
        "producer_logical_movement_bytes",
        "producer_topology_read_bytes",
        "complete_overwrite",
        "covered_rows",
        "shared_source_row_scans",
        "additional_source_row_scans",
        "member_upload_bytes",
        "command_buffers",
        "waits",
        "encoders",
        "dispatches",
        "released",
    }
    publish_fields = fused_publish_fields if fused_producer else legacy_publish_fields
    raw_publish = exact_span_args(
        events, publish_name, publish_fields
    )
    publish = {
        field: nonnegative_trace_integer(value, f"carrier publication {field}")
        for field, value in raw_publish.items()
        if field not in {"route", "complete_overwrite", "released"}
    }
    if fused_producer:
        publish["route"] = trace_string(raw_publish["route"], "carrier route")
    publish["complete_overwrite"] = trace_boolean(
        raw_publish["complete_overwrite"]
    )
    if fused_producer:
        publish["released"] = trace_boolean(raw_publish["released"])
    physical_rows = publish["physical_rows"]
    work_items = publish["work_items"]
    addresses = 1 << 13
    address_offset_bytes = 4 * (addresses + 1)
    carrier_resident_bytes = (
        10 * physical_rows + 8 * work_items + address_offset_bytes
    )
    producer_logical_movement_bytes = (
        30 * physical_rows + 16 * work_items + address_offset_bytes
    )
    expected_publish = {
        "cycles": rows,
        "source_windows": rows,
        "carrier_occurrence_bytes": 2 * physical_rows,
        "carrier_magnitude_bytes": 8 * physical_rows,
        "carrier_work_item_bytes": 8 * work_items,
        "carrier_address_offset_bytes": address_offset_bytes,
        "shared_source_row_scans": 1,
        "additional_source_row_scans": 0,
        "complete_overwrite": True,
        "covered_rows": physical_rows,
    }
    if fused_producer:
        topology_read_bytes = (
            publish["bytecode_descriptor_bytes"]
            + publish["bytecode_pivot_bytes"]
            + publish["bytecode_chunk_offset_bytes"]
        )
        expected_publish.update(
            {
                "route": METAL_BYTECODE_ADDRESS_FUSED_ROUTE,
                "carrier_resident_bytes": carrier_resident_bytes,
                "carrier_buffers": 4,
                "scatter_output_allocations": 2,
                "producer_persistent_write_bytes": 10 * physical_rows,
                "producer_logical_movement_bytes": 10 * physical_rows
                + topology_read_bytes,
                "producer_topology_read_bytes": topology_read_bytes,
                "member_upload_bytes": 0,
                "command_buffers": 1,
                "waits": 1,
                "encoders": 1,
                "dispatches": 1,
                "released": False,
            }
        )
    else:
        expected_publish.update(
            {
                "carrier_resident_bytes": carrier_resident_bytes,
                "carrier_allocations": 4,
                "producer_persistent_write_bytes": carrier_resident_bytes,
                "producer_logical_movement_bytes": producer_logical_movement_bytes,
                "producer_topology_read_bytes": 0,
                "member_source_upload_bytes": 0,
            }
        )
    if (
        physical_rows <= 0
        or physical_rows > rows
        or work_items < (physical_rows + 4095) // 4096
        or work_items > physical_rows
        or publish["carrier_completion_serial"] <= 0
        or any(publish[field] != value for field, value in expected_publish.items())
    ):
        raise ValueError(
            f"Bytecode address carrier publication ledger is invalid: {publish}"
        )
    source_ids = [
        publish["source_rows_storage_id"],
        publish["source_claim_storage_id"],
    ]
    carrier_ids = [
        publish["carrier_occurrence_storage_id"],
        publish["carrier_magnitude_storage_id"],
        publish["carrier_work_item_storage_id"],
        publish["carrier_address_offset_storage_id"],
    ]
    topology_ids = (
        [
            publish["bytecode_descriptor_storage_id"],
            publish["bytecode_pivot_storage_id"],
            publish["bytecode_chunk_offset_storage_id"],
        ]
        if fused_producer
        else []
    )
    if (
        any(identity <= 0 for identity in source_ids + carrier_ids + topology_ids)
        or len(set(source_ids)) != len(source_ids)
        or len(set(carrier_ids)) != len(carrier_ids)
        or len(set(topology_ids)) != len(topology_ids)
        or set(source_ids) & set(carrier_ids + topology_ids)
        or set(carrier_ids) & set(topology_ids)
        or publish["source_generation"] <= 0
        or publish["source_completion_serial"] <= 0
        or publish["source_device_registry_id"] <= 0
    ):
        raise ValueError("Bytecode address carrier publication provenance is invalid")
    if stage1_source is None:
        raise ValueError("Bytecode address AddressMajor route is missing its Stage1 source")
    if stage1_source.get("explicit_rows") != physical_rows:
        raise ValueError(
            "Bytecode address carrier physical rows do not match the Stage1 projection"
        )
    source_match = {
        "source_generation": stage1_source["source_generation"],
        "source_completion_serial": stage1_source["completion_serial"],
        "source_rows_storage_id": stage1_source["row_allocation_identity"],
        "source_claim_storage_id": stage1_source["claim_allocation_identity"],
        "source_device_registry_id": stage1_source["device_registry_id"],
        "source_windows": stage1_source["source_windows"],
    }
    if any(publish[field] != value for field, value in source_match.items()):
        raise ValueError("Bytecode address carrier does not match its Stage1 source")
    if fused_producer:
        if stage1_scatter is None or stage1_scatter.get("bytecode_fused") is not True:
            raise ValueError(
                "fused Bytecode address carrier is missing its InstructionReadRaf scatter"
            )
        scatter_match = {
            "physical_rows": stage1_scatter["bytecode_physical_rows"],
            "work_items": stage1_scatter["bytecode_work_items"],
            "carrier_occurrence_storage_id": stage1_scatter[
                "bytecode_occurrence_storage_id"
            ],
            "carrier_occurrence_bytes": stage1_scatter[
                "bytecode_occurrence_bytes"
            ],
            "carrier_magnitude_storage_id": stage1_scatter[
                "bytecode_magnitude_storage_id"
            ],
            "carrier_magnitude_bytes": stage1_scatter["bytecode_magnitude_bytes"],
            "carrier_work_item_storage_id": stage1_scatter[
                "bytecode_work_item_storage_id"
            ],
            "carrier_work_item_bytes": stage1_scatter["bytecode_work_item_bytes"],
            "carrier_address_offset_storage_id": stage1_scatter[
                "bytecode_address_offset_storage_id"
            ],
            "carrier_address_offset_bytes": stage1_scatter[
                "bytecode_address_offset_bytes"
            ],
            "bytecode_descriptor_storage_id": stage1_scatter[
                "bytecode_descriptor_storage_id"
            ],
            "bytecode_descriptor_bytes": stage1_scatter[
                "bytecode_descriptor_bytes"
            ],
            "bytecode_pivot_storage_id": stage1_scatter[
                "bytecode_pivot_storage_id"
            ],
            "bytecode_pivot_bytes": stage1_scatter["bytecode_pivot_bytes"],
            "bytecode_chunk_offset_storage_id": stage1_scatter[
                "bytecode_chunk_offset_storage_id"
            ],
            "bytecode_chunk_offset_bytes": stage1_scatter[
                "bytecode_chunk_offset_bytes"
            ],
        }
        if (
            any(publish[field] != value for field, value in scatter_match.items())
            or publish["command_buffers"] != stage1_scatter["command_buffers"]
            or publish["waits"] != stage1_scatter["waits"]
            or publish["encoders"] != stage1_scatter["encoders"]
            or publish["dispatches"] != stage1_scatter["dispatches"]
            or publish["bytecode_descriptor_bytes"] <= 0
            or publish["bytecode_chunk_offset_bytes"] <= 0
        ):
            raise ValueError(
                "fused Bytecode address carrier receipt does not match its scatter"
            )
        if topology_observation is None or topology_observation["enabled"] is not True:
            raise ValueError(
                "fused Bytecode address carrier is missing its topology receipt"
            )
        topology_scatter_match = {
            "physical_rows": "bytecode_physical_rows",
            "descriptor_elements": "bytecode_descriptor_elements",
            "descriptor_bytes": "bytecode_descriptor_bytes",
            "descriptor_storage_id": "bytecode_descriptor_storage_id",
            "pivot_elements": "bytecode_pivot_elements",
            "pivot_bytes": "bytecode_pivot_bytes",
            "pivot_storage_id": "bytecode_pivot_storage_id",
            "chunk_offset_elements": "bytecode_chunk_offset_elements",
            "chunk_offset_bytes": "bytecode_chunk_offset_bytes",
            "chunk_offset_storage_id": "bytecode_chunk_offset_storage_id",
            "work_items": "bytecode_work_items",
            "work_item_bytes": "bytecode_work_item_bytes",
            "work_item_storage_id": "bytecode_work_item_storage_id",
            "address_offset_elements": "bytecode_address_offset_elements",
            "address_offset_bytes": "bytecode_address_offset_bytes",
            "address_offset_storage_id": "bytecode_address_offset_storage_id",
            "max_descriptors_per_chunk": "bytecode_max_descriptors_per_chunk",
            "max_pivots_per_chunk": "bytecode_max_pivots_per_chunk",
        }
        if any(
            topology_observation[topology_field]
            != stage1_scatter[scatter_field]
            for topology_field, scatter_field in topology_scatter_match.items()
        ):
            raise ValueError(
                "fused Bytecode address topology does not match its scatter"
            )
        publish["carrier_resident_bytes"] = carrier_resident_bytes
        publish["topology_resident_bytes"] = (
            publish["bytecode_descriptor_bytes"]
            + publish["bytecode_pivot_bytes"]
            + publish["bytecode_chunk_offset_bytes"]
        )

    complete_fields = {
        "cycles",
        "addresses",
        "stages",
        "physical_rows",
        "work_items",
        "requested",
        "realized_route",
        "fallback_reason",
        "source_generation",
        "source_completion_serial",
        "source_rows_storage_id",
        "source_rows_bytes",
        "source_claim_storage_id",
        "source_device_registry_id",
        "carrier_completion_serial",
        "carrier_occurrence_storage_id",
        "carrier_occurrence_bytes",
        "carrier_magnitude_storage_id",
        "carrier_magnitude_bytes",
        "carrier_work_item_storage_id",
        "carrier_work_item_bytes",
        "carrier_address_offset_storage_id",
        "carrier_address_offset_bytes",
        "carrier_resident_bytes",
        "producer_persistent_write_bytes",
        "producer_logical_movement_bytes",
        "producer_topology_read_bytes",
        "member_carrier_owned_bytes",
        "member_source_scans",
        "member_source_upload_bytes",
        "equality_bytes",
        "padding_bytes",
        "partial_bytes",
        "output_readback_bytes",
        "member_owned_bytes",
        "command_buffers",
        "waits",
        "worker_dispatches",
        "worker_variant",
        "worker_simd_width",
        "worker_threads",
        "worker_items_per_threadgroup",
        "worker_threadgroups",
        "worker_tail_slots",
        "worker_dynamic_threadgroup_bytes",
        "worker_static_threadgroup_bytes",
        "worker_threadgroup_bytes",
        "reducer_dispatches",
        "reducer_threads",
        "reducer_threadgroups",
        "reducer_static_threadgroup_bytes",
        "output_fields",
        "submit_ns",
        "overlap_ns",
        "join_ns",
        "resident_wall_ns",
        "gpu_active_ns",
        "completed_before_join",
        "complete_overwrite",
        "carrier_released",
    }
    fused_complete_fields = {
        "bytecode_descriptor_storage_id",
        "bytecode_descriptor_bytes",
        "bytecode_pivot_storage_id",
        "bytecode_pivot_bytes",
        "bytecode_chunk_offset_storage_id",
        "bytecode_chunk_offset_bytes",
        "topology_publication_bytes",
    }
    if fused_producer:
        complete_fields |= fused_complete_fields
    raw_complete = exact_span_args(
        events, METAL_BYTECODE_ADDRESS_COMPLETE, complete_fields
    )
    string_fields = {
        "requested",
        "realized_route",
        "fallback_reason",
        "worker_variant",
    }
    boolean_fields = {
        "completed_before_join",
        "complete_overwrite",
        "carrier_released",
    }
    complete = {
        field: nonnegative_trace_integer(value, f"address completion {field}")
        for field, value in raw_complete.items()
        if field not in string_fields | boolean_fields
    }
    complete.update(
        {
            field: trace_string(raw_complete[field], field)
            for field in string_fields
        }
    )
    complete.update(
        {field: trace_boolean(raw_complete[field]) for field in boolean_fields}
    )
    stages = 9
    inner = 1 << 15
    outer = rows // inner
    equality_bytes = 16 * stages * (inner + outer)
    padding_bytes = 5 * 16
    partial_bytes = 16 * stages * work_items
    output_bytes = 16 * stages * addresses
    producer_topology_read_bytes = (
        publish.get("topology_resident_bytes", 0) if fused_producer else 0
    )
    producer_persistent_write_bytes = (
        10 * physical_rows if fused_producer else carrier_resident_bytes
    )
    complete_producer_logical_movement_bytes = (
        producer_persistent_write_bytes + producer_topology_read_bytes
        if fused_producer
        else producer_logical_movement_bytes
    )
    topology_publication_bytes = (
        producer_topology_read_bytes
        + publish["carrier_work_item_bytes"]
        + publish["carrier_address_offset_bytes"]
        if fused_producer
        else 0
    )
    expected_complete = {
        "cycles": rows,
        "addresses": addresses,
        "stages": stages,
        "physical_rows": physical_rows,
        "work_items": work_items,
        "requested": "address_major",
        "realized_route": (
            METAL_BYTECODE_ADDRESS_FUSED_ROUTE
            if fused_producer
            else "address_major"
        ),
        "fallback_reason": "none",
        "source_generation": publish["source_generation"],
        "source_completion_serial": publish["source_completion_serial"],
        "source_rows_storage_id": publish["source_rows_storage_id"],
        "source_rows_bytes": 40 * rows,
        "source_claim_storage_id": publish["source_claim_storage_id"],
        "source_device_registry_id": publish["source_device_registry_id"],
        "carrier_completion_serial": publish["carrier_completion_serial"],
        "carrier_occurrence_storage_id": publish[
            "carrier_occurrence_storage_id"
        ],
        "carrier_occurrence_bytes": 2 * physical_rows,
        "carrier_magnitude_storage_id": publish["carrier_magnitude_storage_id"],
        "carrier_magnitude_bytes": 8 * physical_rows,
        "carrier_work_item_storage_id": publish[
            "carrier_work_item_storage_id"
        ],
        "carrier_work_item_bytes": 8 * work_items,
        "carrier_address_offset_storage_id": publish[
            "carrier_address_offset_storage_id"
        ],
        "carrier_address_offset_bytes": address_offset_bytes,
        "carrier_resident_bytes": carrier_resident_bytes,
        "producer_persistent_write_bytes": producer_persistent_write_bytes,
        "producer_logical_movement_bytes": complete_producer_logical_movement_bytes,
        "producer_topology_read_bytes": producer_topology_read_bytes,
        "member_carrier_owned_bytes": 0,
        "member_source_scans": 0,
        "member_source_upload_bytes": 0,
        "equality_bytes": equality_bytes,
        "padding_bytes": padding_bytes,
        "partial_bytes": partial_bytes,
        "output_readback_bytes": output_bytes,
        "member_owned_bytes": (
            equality_bytes + padding_bytes + partial_bytes + output_bytes
        ),
        "command_buffers": 1,
        "waits": 1,
        "worker_dispatches": 1,
        "worker_variant": "packed4_halfwidth_v1",
        "worker_simd_width": 32,
        "worker_threads": 128,
        "worker_items_per_threadgroup": 4,
        "worker_threadgroups": (work_items + 3) // 4,
        "worker_tail_slots": (4 - work_items % 4) % 4,
        "worker_dynamic_threadgroup_bytes": 0,
        "worker_static_threadgroup_bytes": 0,
        "worker_threadgroup_bytes": 0,
        "reducer_dispatches": 1,
        "reducer_threads": 256,
        "reducer_threadgroups": (stages * addresses + 255) // 256,
        "reducer_static_threadgroup_bytes": 0,
        "output_fields": stages * addresses,
        "submit_ns": 0,
        "overlap_ns": 0,
        "completed_before_join": False,
        "complete_overwrite": True,
        "carrier_released": True,
    }
    if fused_producer:
        expected_complete.update(
            {
                "bytecode_descriptor_storage_id": publish[
                    "bytecode_descriptor_storage_id"
                ],
                "bytecode_descriptor_bytes": publish["bytecode_descriptor_bytes"],
                "bytecode_pivot_storage_id": publish["bytecode_pivot_storage_id"],
                "bytecode_pivot_bytes": publish["bytecode_pivot_bytes"],
                "bytecode_chunk_offset_storage_id": publish[
                    "bytecode_chunk_offset_storage_id"
                ],
                "bytecode_chunk_offset_bytes": publish[
                    "bytecode_chunk_offset_bytes"
                ],
                "topology_publication_bytes": topology_publication_bytes,
            }
        )
    if any(complete[field] != value for field, value in expected_complete.items()):
        raise ValueError(
            f"Bytecode address AddressMajor completion ledger is invalid: {complete}"
        )
    if (
        complete["resident_wall_ns"] <= 0
        or complete["join_ns"] != complete["resident_wall_ns"]
        or complete["gpu_active_ns"] <= 0
        or complete["gpu_active_ns"] > complete["resident_wall_ns"]
    ):
        raise ValueError("Bytecode address AddressMajor timing ledger is invalid")

    return {
        "components": components,
        "outer_counts": outer_counts,
        "metal_counts": metal_counts,
        "route_observation": route,
        "topology_observation": topology_observation,
        "resource_observation": {
            "producer_kind": (
                "fused_stage1_grouped_v1" if fused_producer else "legacy_sparse_v1"
            ),
            "fused_topology_prepare": topology_observation,
            "carrier_publish": publish,
            "address_major_complete": complete,
        },
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


def ram_sparse_member_components(
    events: list[dict[str, Any]], kernel: str, rounds: int
) -> dict[str, Any]:
    components = ("prepare", "prove_round", "finish_rounds", "output_claims")
    names = {f"{kernel}::{component}" for component in components}
    intervals = strict_named_intervals(
        events,
        names | {PIOP_SPAN, SUMCHECK_ROUND_SPAN, SUMCHECK_HOST_FIAT_SHAMIR_SPAN},
    )
    if len(intervals[PIOP_SPAN]) != 1:
        raise ValueError(f"{kernel} requires exactly one PIOP span")
    piop = intervals[PIOP_SPAN][0]
    by_component = {
        component: sorted(intervals[f"{kernel}::{component}"])
        for component in components
    }
    counts = {
        component: len(component_intervals)
        for component, component_intervals in by_component.items()
    }
    expected_counts = {
        "prepare": 1,
        "prove_round": rounds,
        "finish_rounds": 1,
        "output_claims": 1,
    }
    if counts != expected_counts:
        raise ValueError(f"{kernel} member span counts {counts}, expected {expected_counts}")

    prepare = by_component["prepare"][0]
    prove_rounds = by_component["prove_round"]
    finish = by_component["finish_rounds"][0]
    output = by_component["output_claims"][0]
    ordered = [prepare, *prove_rounds, finish, output]
    if any(start < piop[0] or end > piop[1] for start, end in ordered):
        raise ValueError(f"a {kernel} member span lies outside PIOP")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"{kernel} member spans overlap or are out of order")

    fiat_shamir_intervals = []
    for member_round in prove_rounds:
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
        fiat_shamir_intervals.append(fiat_shamir)
    if len(set(fiat_shamir_intervals)) != rounds:
        raise ValueError(f"{kernel} rounds reuse a host Fiat-Shamir span")

    prepare_us = interval_duration_us(prepare)
    round_us = [interval_duration_us(interval) for interval in prove_rounds]
    finish_us = interval_duration_us(finish)
    output_us = interval_duration_us(output)
    member_us = prepare_us + sum(round_us) + finish_us + output_us
    host_fiat_shamir_us = [
        interval_duration_us(interval) for interval in fiat_shamir_intervals
    ]
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in [prepare_us, *round_us, finish_us, output_us, member_us]
    ):
        raise ValueError(f"{kernel} has a non-positive member duration")
    return {
        "components": {
            "prepare_us": prepare_us,
            "rounds_us": round_us,
            "rounds_total_us": sum(round_us),
            "host_fiat_shamir_us": host_fiat_shamir_us,
            "host_fiat_shamir_total_us": sum(host_fiat_shamir_us),
            "finish_us": finish_us,
            "output_claims_us": output_us,
            "member_us": member_us,
        },
        "outer_counts": counts,
        "intervals": {
            "piop": piop,
            "prepare": prepare,
            "prove_rounds": prove_rounds,
            "finish": finish,
            "output_claims": output,
            "canonical": ordered,
            "host_fiat_shamir": fiat_shamir_intervals,
        },
    }


def ram_cycle_family_owner_observation(
    events: list[dict[str, Any]], backend: str, log_n: int
) -> Optional[dict[str, Any]]:
    if backend not in {"optimized", "metal"}:
        raise ValueError(f"unsupported RAM cycle-family backend {backend!r}")
    count = positive_span_count(events, METAL_RAM_CYCLE_FAMILY_OWNER)
    if backend == "optimized":
        if count != 0:
            raise ValueError("optimized RAM route unexpectedly built a Metal owner")
        return None
    if count != 1:
        raise ValueError("Metal RAM route must build exactly one cycle-family owner")

    fields = {
        "enabled",
        "schema_version",
        "source_kind",
        "source_generation",
        "source_fingerprint",
        "log_t",
        "log_k",
        "cycles",
        "address_domain",
        "access_records",
        "increment_records",
        "hamming_exact",
        "retained_records",
        "final_memory_elements",
        "record_bytes",
        "final_memory_bytes",
        "read_write_topology_nodes",
        "block_topology_nodes",
        "topology_bytes",
        "owner_bytes",
        "source_rows",
        "source_collection_performed",
        "shared_source_row_scans",
        "additional_source_row_scans",
        "member_upload_bytes",
        "complete_publication",
    }
    raw = exact_span_args(events, METAL_RAM_CYCLE_FAMILY_OWNER, fields)
    strings = {"source_kind": trace_string(raw["source_kind"], "source_kind")}
    booleans = {
        field: trace_boolean(raw[field])
        for field in (
            "enabled",
            "hamming_exact",
            "source_collection_performed",
            "complete_publication",
        )
    }
    integers = {
        field: nonnegative_trace_integer(raw[field], f"RAM owner {field}")
        for field in fields - strings.keys() - booleans.keys()
    }
    observation = {**strings, **booleans, **integers}
    cycles = 1 << log_n
    access_records = observation["access_records"]
    increment_records = observation["increment_records"]
    address_domain = observation["address_domain"]
    if (
        observation["enabled"] is not True
        or observation["schema_version"] != RAM_CYCLE_FAMILY_SCHEMA_VERSION
        or observation["source_kind"] != "ram_access_tape_v1"
        or observation["source_generation"] <= 0
        or observation["source_fingerprint"] <= 0
        or observation["log_t"] != log_n
        or observation["cycles"] != cycles
        or observation["source_rows"] != cycles
        or observation["log_k"] <= 0
        or address_domain != 1 << observation["log_k"]
        or observation["final_memory_elements"] != address_domain
        or access_records <= 0
        or observation["retained_records"] != access_records
        or observation["hamming_exact"] is not True
        or observation["record_bytes"] != 24 * (access_records + increment_records)
        or observation["final_memory_bytes"] != 8 * address_domain
        or observation["read_write_topology_nodes"] <= 0
        or observation["block_topology_nodes"] < access_records
        or observation["topology_bytes"] <= 0
        or observation["owner_bytes"]
        != observation["record_bytes"]
        + observation["final_memory_bytes"]
        + observation["topology_bytes"]
        or observation["source_collection_performed"] is not True
        or observation["shared_source_row_scans"] != 1
        or observation["additional_source_row_scans"] != 0
        or observation["member_upload_bytes"] != 0
        or observation["complete_publication"] is not True
    ):
        raise ValueError("RAM cycle-family owner receipt is inconsistent")
    observation["wall_us"] = unique_named_span_duration_us(
        events, METAL_RAM_CYCLE_FAMILY_OWNER
    )
    observation["interval"] = span_intervals_us(
        events, METAL_RAM_CYCLE_FAMILY_OWNER
    )[0][1:]
    lifecycle = strict_named_intervals(
        events, {BACKEND_WITNESS_PREP_SPAN, PIOP_SPAN}
    )
    if len(lifecycle[BACKEND_WITNESS_PREP_SPAN]) != 1:
        raise ValueError(
            "Metal RAM route requires exactly one backend witness preparation span"
        )
    if len(lifecycle[PIOP_SPAN]) != 1:
        raise ValueError("Metal RAM route requires exactly one PIOP span")
    backend_witness_prepare = lifecycle[BACKEND_WITNESS_PREP_SPAN][0]
    piop = lifecycle[PIOP_SPAN][0]
    if (
        observation["interval"][0] < backend_witness_prepare[0]
        or observation["interval"][1] > backend_witness_prepare[1]
    ):
        raise ValueError(
            "RAM owner construction is not contained in backend witness preparation"
        )
    if backend_witness_prepare[1] > piop[0]:
        raise ValueError("RAM owner construction must complete before PIOP")
    observation["backend_witness_prepare_interval"] = backend_witness_prepare
    observation["piop_interval"] = piop
    observation["published_before_piop"] = True
    return observation


def ram_cycle_family_witness_observation(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    owner: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if backend not in {"optimized", "metal"}:
        raise ValueError(f"unsupported RAM cycle-family backend {backend!r}")
    count = positive_span_count(events, METAL_RAM_CYCLE_FAMILY_WITNESS_PREPARE)
    if backend == "optimized":
        if count != 0 or owner is not None:
            raise ValueError("optimized RAM route unexpectedly prepared Metal witness state")
        return None
    if owner is None or count != 1:
        raise ValueError("Metal RAM route requires exactly one full witness preparation")

    fields = {
        "schema_version",
        "requested",
        "selected",
        "fallback_reason",
        "log_t",
        "log_k",
        "cycles",
        "address_domain",
        "source_generation",
        "source_fingerprint",
        "source_collection_performed",
        "witness_source_scans",
        "additional_witness_source_scans",
        "address_validation_passes",
        "address_rows",
        "address_plane_storage_id",
        "address_plane_device_registry_id",
        "address_plane_bytes",
        "address_plane_upload_bytes",
        "address_plane_allocations",
        "owner_published",
        "address_plane_published",
        "complete_publication",
    }
    raw = exact_span_args(events, METAL_RAM_CYCLE_FAMILY_WITNESS_PREPARE, fields)
    string_fields = {"requested", "selected", "fallback_reason"}
    boolean_fields = {
        "source_collection_performed",
        "owner_published",
        "address_plane_published",
        "complete_publication",
    }
    observation = {
        **{
            field: trace_string(raw[field], field)
            for field in string_fields
        },
        **{
            field: trace_boolean(raw[field])
            for field in boolean_fields
        },
        **{
            field: nonnegative_trace_integer(raw[field], f"RAM witness {field}")
            for field in fields - string_fields - boolean_fields
        },
    }
    cycles = 1 << log_n
    address_bytes = 4 * cycles
    if (
        observation["schema_version"] != RAM_CYCLE_FAMILY_SCHEMA_VERSION
        or observation["requested"] != "host_sparse_v1"
        or observation["selected"] != "host_sparse_v1"
        or observation["fallback_reason"] != "none"
        or observation["log_t"] != log_n
        or observation["log_k"] != RAM_CYCLE_FAMILY_LOG_K
        or observation["cycles"] != cycles
        or observation["address_domain"] != 1 << RAM_CYCLE_FAMILY_LOG_K
        or observation["source_generation"] != owner["source_generation"]
        or observation["source_fingerprint"] != owner["source_fingerprint"]
        or observation["source_collection_performed"] is not True
        or observation["source_collection_performed"]
        != owner["source_collection_performed"]
        or observation["witness_source_scans"] != 1
        or observation["witness_source_scans"]
        != owner["shared_source_row_scans"]
        or observation["additional_witness_source_scans"] != 0
        or observation["address_validation_passes"] != 3
        or observation["address_rows"] != cycles
        or observation["address_plane_storage_id"] <= 0
        or observation["address_plane_device_registry_id"] <= 0
        or observation["address_plane_bytes"] != address_bytes
        or observation["address_plane_upload_bytes"] != address_bytes
        or observation["address_plane_allocations"] != 1
        or observation["owner_published"] is not True
        or observation["address_plane_published"] is not True
        or observation["complete_publication"] is not True
    ):
        raise ValueError("RAM cycle-family witness receipt is inconsistent")

    observation["wall_us"] = unique_named_span_duration_us(
        events, METAL_RAM_CYCLE_FAMILY_WITNESS_PREPARE
    )
    observation["interval"] = span_intervals_us(
        events, METAL_RAM_CYCLE_FAMILY_WITNESS_PREPARE
    )[0][1:]
    require_contained(
        owner["interval"], observation["interval"], "RAM cycle-family owner"
    )
    require_contained(
        observation["interval"],
        owner["backend_witness_prepare_interval"],
        "RAM cycle-family witness preparation",
    )
    if observation["interval"][1] > owner["piop_interval"][0]:
        raise ValueError("RAM witness preparation must complete before PIOP")
    return observation


def ram_cycle_family_terminal_take_observation(
    events: list[dict[str, Any]],
    backend: str,
    owner: Optional[dict[str, Any]],
    hamming_sparse_prepare: Optional[tuple[float, float]],
) -> Optional[dict[str, Any]]:
    count = positive_span_count(events, METAL_RAM_CYCLE_FAMILY_TERMINAL_TAKE)
    if backend == "optimized":
        if count != 0:
            raise ValueError("optimized RAM route unexpectedly took a Metal owner")
        return None
    if backend != "metal" or owner is None or hamming_sparse_prepare is None:
        raise ValueError("RAM terminal take lacks its Metal owner lifecycle")
    if count != 1:
        raise ValueError("Metal RAM route must take exactly one cycle-family owner")

    fields = {
        "source_generation",
        "source_fingerprint",
        "selected",
        "fallback_reason",
        "session_owner_removed",
        "columns_removed",
    }
    raw = exact_span_args(events, METAL_RAM_CYCLE_FAMILY_TERMINAL_TAKE, fields)
    observation = {
        "source_generation": positive_trace_integer(
            raw["source_generation"], "RAM terminal source_generation"
        ),
        "source_fingerprint": positive_trace_integer(
            raw["source_fingerprint"], "RAM terminal source_fingerprint"
        ),
        "selected": trace_string(raw["selected"], "selected"),
        "fallback_reason": trace_string(raw["fallback_reason"], "fallback_reason"),
        "session_owner_removed": trace_boolean(raw["session_owner_removed"]),
        "columns_removed": trace_boolean(raw["columns_removed"]),
    }
    if observation != {
        "source_generation": owner["source_generation"],
        "source_fingerprint": owner["source_fingerprint"],
        "selected": "host_sparse_v1",
        "fallback_reason": "none",
        "session_owner_removed": True,
        "columns_removed": True,
    }:
        raise ValueError("RAM terminal-take receipt is inconsistent")
    interval = span_intervals_us(events, METAL_RAM_CYCLE_FAMILY_TERMINAL_TAKE)[0][1:]
    if interval[0] < hamming_sparse_prepare[1]:
        raise ValueError("RAM terminal take precedes Hamming sparse preparation")
    enclosing_name = "RamRaVirtualization::prepare"
    enclosing = strict_named_intervals(events, {enclosing_name})[enclosing_name]
    if len(enclosing) != 1:
        raise ValueError("RAM terminal take requires one RA virtualization preparation")
    require_contained(interval, enclosing[0], "RAM terminal take")
    observation["interval"] = interval
    observation["enclosing_prepare"] = enclosing[0]
    return observation


def ram_read_write_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    owner: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    base = ram_sparse_member_components(events, RAM_READ_WRITE_KERNEL, log_n + 13)
    phases = ("route", "sparse_prepare", "sparse_complete", "sparse_derived_validate")
    names = {f"MetalRamReadWrite::{phase}" for phase in phases}
    unknown = {
        name
        for name, _, _ in span_intervals_us(events)
        if name.startswith("MetalRamReadWrite::") and name not in names
    }
    if unknown:
        raise ValueError(f"RAM read-write trace contains unknown Metal phases: {sorted(unknown)}")
    counts = {
        phase: positive_span_count(events, f"MetalRamReadWrite::{phase}")
        for phase in phases
    }
    if owner is None:
        owner = ram_cycle_family_owner_observation(events, backend, log_n)
    if backend == "optimized":
        if any(counts.values()) or owner is not None:
            raise ValueError("optimized RAM read-write exercised Metal sparse work")
        base["components"]["owner_prepare_us"] = 0.0
        base["components"]["charged_member_us"] = base["components"]["member_us"]
        return {
            "components": base["components"],
            "outer_counts": base["outer_counts"],
            "metal_counts": counts,
            "resource_observation": None,
            "intervals": base["intervals"],
        }
    if backend != "metal" or owner is None or any(count != 1 for count in counts.values()):
        raise ValueError("Metal RAM read-write sparse route is incomplete")

    intervals = {
        phase: span_intervals_us(events, f"MetalRamReadWrite::{phase}")[0][1:]
        for phase in phases
    }
    outer = base["intervals"]
    if owner["interval"][1] > outer["prepare"][0]:
        raise ValueError("RAM read-write starts before its shared owner is published")
    require_contained(intervals["sparse_prepare"], outer["prepare"], "RAM read-write prepare")
    require_contained(intervals["route"], intervals["sparse_prepare"], "RAM read-write route")
    require_contained(intervals["sparse_complete"], outer["output_claims"], "RAM read-write output")
    derived = intervals["sparse_derived_validate"]
    if derived[0] < outer["finish"][1] or derived[1] > outer["output_claims"][0]:
        raise ValueError("RAM read-write derived validation is out of order")

    route_fields = {
        "cycles",
        "log_t",
        "log_k",
        "requested",
        "selected",
        "fallback_reason",
        "source_generation",
        "source_fingerprint",
    }
    prepare_fields = {
        "selected",
        "source_generation",
        "source_fingerprint",
        "log_t",
        "log_k",
        "rounds",
        "access_records",
        "increment_records",
        "owner_bytes",
        "cycle_cutoff",
        "additional_source_row_scans",
        "member_upload_bytes",
        "gpu_dispatches",
        "command_buffers",
        "waits",
        "readbacks",
    }
    complete_fields = {
        "selected",
        "source_generation",
        "source_fingerprint",
        "output_claims_valid",
    }
    derived_fields = {"source_generation", "source_fingerprint", "derived_claim_valid"}
    route_raw = exact_span_args(events, "MetalRamReadWrite::route", route_fields)
    prepare_raw = exact_span_args(events, "MetalRamReadWrite::sparse_prepare", prepare_fields)
    complete_raw = exact_span_args(events, "MetalRamReadWrite::sparse_complete", complete_fields)
    derived_raw = exact_span_args(
        events, "MetalRamReadWrite::sparse_derived_validate", derived_fields
    )
    route = {
        "requested": trace_string(route_raw["requested"], "requested"),
        "selected": trace_string(route_raw["selected"], "selected"),
        "fallback_reason": trace_string(route_raw["fallback_reason"], "fallback_reason"),
        **{
            field: nonnegative_trace_integer(route_raw[field], f"RAM read-write {field}")
            for field in route_fields - {"requested", "selected", "fallback_reason"}
        },
    }
    prepare = {
        "selected": trace_string(prepare_raw["selected"], "selected"),
        **{
            field: nonnegative_trace_integer(prepare_raw[field], f"RAM read-write {field}")
            for field in prepare_fields - {"selected"}
        },
    }
    complete = {
        "selected": trace_string(complete_raw["selected"], "selected"),
        "source_generation": positive_trace_integer(
            complete_raw["source_generation"], "RAM read-write source_generation"
        ),
        "source_fingerprint": positive_trace_integer(
            complete_raw["source_fingerprint"], "RAM read-write source_fingerprint"
        ),
        "output_claims_valid": trace_boolean(complete_raw["output_claims_valid"]),
    }
    derived_observation = {
        "source_generation": positive_trace_integer(
            derived_raw["source_generation"], "RAM read-write source_generation"
        ),
        "source_fingerprint": positive_trace_integer(
            derived_raw["source_fingerprint"], "RAM read-write source_fingerprint"
        ),
        "derived_claim_valid": trace_boolean(derived_raw["derived_claim_valid"]),
    }
    expected_source = (owner["source_generation"], owner["source_fingerprint"])
    source_pairs = [
        (route["source_generation"], route["source_fingerprint"]),
        (prepare["source_generation"], prepare["source_fingerprint"]),
        (complete["source_generation"], complete["source_fingerprint"]),
        (
            derived_observation["source_generation"],
            derived_observation["source_fingerprint"],
        ),
    ]
    if (
        route != {
            "cycles": 1 << log_n,
            "log_t": log_n,
            "log_k": owner["log_k"],
            "requested": "host_sparse_v1",
            "selected": "host_sparse_v1",
            "fallback_reason": "none",
            "source_generation": owner["source_generation"],
            "source_fingerprint": owner["source_fingerprint"],
        }
        or any(pair != expected_source for pair in source_pairs)
        or prepare["selected"] != "host_sparse_v1"
        or prepare["log_t"] != log_n
        or prepare["log_k"] != owner["log_k"]
        or prepare["rounds"] != log_n + owner["log_k"]
        or prepare["access_records"] != owner["access_records"]
        or prepare["increment_records"] != owner["increment_records"]
        or prepare["owner_bytes"] != owner["owner_bytes"]
        or any(
            prepare[field] != 0
            for field in (
                "cycle_cutoff",
                "additional_source_row_scans",
                "member_upload_bytes",
                "gpu_dispatches",
                "command_buffers",
                "waits",
                "readbacks",
            )
        )
        or complete["selected"] != "host_sparse_v1"
        or complete["output_claims_valid"] is not True
        or derived_observation["derived_claim_valid"] is not True
    ):
        raise ValueError("RAM read-write sparse receipt is inconsistent")
    base["components"]["owner_prepare_us"] = owner["wall_us"]
    base["components"]["charged_member_us"] = (
        base["components"]["member_us"] + owner["wall_us"]
    )
    return {
        "components": base["components"],
        "outer_counts": base["outer_counts"],
        "metal_counts": counts,
        "resource_observation": {
            "owner": owner,
            "route": route,
            "prepare": prepare,
            "complete": complete,
            "derived": derived_observation,
        },
        "intervals": base["intervals"],
    }


def ram_hamming_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    owner: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    base = ram_sparse_member_components(events, RAM_HAMMING_KERNEL, log_n)
    phases = ("route", "sparse_prepare", "sparse_complete", "sparse_derived_validate")
    names = {f"MetalRamHammingBooleanity::{phase}" for phase in phases}
    unknown = {
        name
        for name, _, _ in span_intervals_us(events)
        if name.startswith("MetalRamHammingBooleanity::") and name not in names
    }
    if unknown:
        raise ValueError(f"RAM Hamming trace contains unknown Metal phases: {sorted(unknown)}")
    counts = {
        phase: positive_span_count(events, f"MetalRamHammingBooleanity::{phase}")
        for phase in phases
    }
    if owner is None:
        owner = ram_cycle_family_owner_observation(events, backend, log_n)
    if backend == "optimized":
        if any(counts.values()) or owner is not None:
            raise ValueError("optimized RAM Hamming exercised Metal sparse work")
        ram_cycle_family_terminal_take_observation(events, backend, owner, None)
        base["components"]["owner_prepare_us"] = 0.0
        base["components"]["charged_member_us"] = base["components"]["member_us"]
        return {
            "components": base["components"],
            "outer_counts": base["outer_counts"],
            "metal_counts": counts,
            "resource_observation": None,
            "intervals": base["intervals"],
        }
    if backend != "metal" or owner is None or any(count != 1 for count in counts.values()):
        raise ValueError("Metal RAM Hamming sparse route is incomplete")

    intervals = {
        phase: span_intervals_us(events, f"MetalRamHammingBooleanity::{phase}")[0][1:]
        for phase in phases
    }
    outer = base["intervals"]
    if owner["interval"][1] > outer["prepare"][0]:
        raise ValueError("RAM Hamming starts before its shared owner is published")
    require_contained(intervals["sparse_prepare"], outer["prepare"], "RAM Hamming prepare")
    terminal_take = ram_cycle_family_terminal_take_observation(
        events, backend, owner, intervals["sparse_prepare"]
    )
    require_contained(intervals["route"], intervals["sparse_prepare"], "RAM Hamming route")
    require_contained(intervals["sparse_complete"], outer["output_claims"], "RAM Hamming output")
    derived = intervals["sparse_derived_validate"]
    if derived[0] < outer["finish"][1] or derived[1] > outer["output_claims"][0]:
        raise ValueError("RAM Hamming derived validation is out of order")

    route_fields = {
        "cycles",
        "requested",
        "selected",
        "fallback_reason",
        "source_generation",
        "source_fingerprint",
    }
    prepare_fields = {
        "selected",
        "fallback_reason",
        "source_generation",
        "source_fingerprint",
        "log_t",
        "access_leaves",
        "parent_nodes",
        "middle_nodes",
        "rounds",
        "estimated_products",
        "product_cap",
        "topology_builds",
        "topology_bytes",
        "member_heap_bytes_including_topology",
        "non_topology_heap_bytes",
        "additional_source_row_scans",
        "dense_h_elements",
        "member_upload_bytes",
        "gpu_dispatches",
        "command_buffers",
        "waits",
        "readbacks",
        "complete_plan",
    }
    complete_fields = {
        "selected",
        "source_generation",
        "source_fingerprint",
        "access_leaves",
        "parent_nodes",
        "middle_nodes",
        "estimated_products",
        "topology_bytes",
        "member_heap_bytes_including_topology",
        "non_topology_heap_bytes",
        "terminal_ready",
        "output_claim_emitted",
    }
    derived_fields = {"source_generation", "source_fingerprint", "derived_claim_valid"}
    route_raw = exact_span_args(events, "MetalRamHammingBooleanity::route", route_fields)
    prepare_raw = exact_span_args(
        events, "MetalRamHammingBooleanity::sparse_prepare", prepare_fields
    )
    complete_raw = exact_span_args(
        events, "MetalRamHammingBooleanity::sparse_complete", complete_fields
    )
    derived_raw = exact_span_args(
        events, "MetalRamHammingBooleanity::sparse_derived_validate", derived_fields
    )
    route = {
        "requested": trace_string(route_raw["requested"], "requested"),
        "selected": trace_string(route_raw["selected"], "selected"),
        "fallback_reason": trace_string(route_raw["fallback_reason"], "fallback_reason"),
        **{
            field: nonnegative_trace_integer(route_raw[field], f"RAM Hamming {field}")
            for field in route_fields - {"requested", "selected", "fallback_reason"}
        },
    }
    prepare = {
        "selected": trace_string(prepare_raw["selected"], "selected"),
        "fallback_reason": trace_string(prepare_raw["fallback_reason"], "fallback_reason"),
        "complete_plan": trace_boolean(prepare_raw["complete_plan"]),
        **{
            field: nonnegative_trace_integer(prepare_raw[field], f"RAM Hamming {field}")
            for field in prepare_fields - {"selected", "fallback_reason", "complete_plan"}
        },
    }
    complete = {
        "selected": trace_string(complete_raw["selected"], "selected"),
        "terminal_ready": trace_boolean(complete_raw["terminal_ready"]),
        "output_claim_emitted": trace_boolean(complete_raw["output_claim_emitted"]),
        **{
            field: nonnegative_trace_integer(complete_raw[field], f"RAM Hamming {field}")
            for field in complete_fields - {"selected", "terminal_ready", "output_claim_emitted"}
        },
    }
    derived_observation = {
        "source_generation": positive_trace_integer(
            derived_raw["source_generation"], "RAM Hamming source_generation"
        ),
        "source_fingerprint": positive_trace_integer(
            derived_raw["source_fingerprint"], "RAM Hamming source_fingerprint"
        ),
        "derived_claim_valid": trace_boolean(derived_raw["derived_claim_valid"]),
    }
    parent_nodes = prepare["parent_nodes"]
    middle_nodes = prepare["middle_nodes"]
    estimated_products = 7 * parent_nodes + middle_nodes + 10 * log_n
    expected_source = (owner["source_generation"], owner["source_fingerprint"])
    source_pairs = [
        (route["source_generation"], route["source_fingerprint"]),
        (prepare["source_generation"], prepare["source_fingerprint"]),
        (complete["source_generation"], complete["source_fingerprint"]),
        (
            derived_observation["source_generation"],
            derived_observation["source_fingerprint"],
        ),
    ]
    geometry_fields = (
        "access_leaves",
        "parent_nodes",
        "middle_nodes",
        "estimated_products",
        "topology_bytes",
        "member_heap_bytes_including_topology",
        "non_topology_heap_bytes",
    )
    if (
        parent_nodes <= 0
        or middle_nodes < 0
        or middle_nodes + 1 != parent_nodes
        or route != {
            "cycles": 1 << log_n,
            "requested": "host_sparse_v1",
            "selected": "host_sparse_v1",
            "fallback_reason": "none",
            "source_generation": owner["source_generation"],
            "source_fingerprint": owner["source_fingerprint"],
        }
        or any(pair != expected_source for pair in source_pairs)
        or prepare["selected"] != "host_sparse_v1"
        or prepare["fallback_reason"] != "none"
        or prepare["log_t"] != log_n
        or prepare["rounds"] != log_n
        or prepare["access_leaves"] != owner["access_records"]
        or prepare["parent_nodes"] != parent_nodes
        or prepare["middle_nodes"] != middle_nodes
        or prepare["estimated_products"] != estimated_products
        or prepare["product_cap"] != RAM_HAMMING_PRODUCT_CAP
        or prepare["estimated_products"] > prepare["product_cap"]
        or prepare["topology_builds"] != 1
        or prepare["topology_bytes"] <= 0
        or prepare["member_heap_bytes_including_topology"]
        != prepare["topology_bytes"] + prepare["non_topology_heap_bytes"]
        or any(
            prepare[field] != 0
            for field in (
                "additional_source_row_scans",
                "dense_h_elements",
                "member_upload_bytes",
                "gpu_dispatches",
                "command_buffers",
                "waits",
                "readbacks",
            )
        )
        or prepare["complete_plan"] is not True
        or complete["selected"] != "host_sparse_v1"
        or any(complete[field] != prepare[field] for field in geometry_fields)
        or complete["terminal_ready"] is not True
        or complete["output_claim_emitted"] is not True
        or derived_observation["derived_claim_valid"] is not True
    ):
        raise ValueError("RAM Hamming sparse receipt is inconsistent")
    base["components"]["owner_prepare_us"] = owner["wall_us"]
    base["components"]["charged_member_us"] = (
        base["components"]["member_us"] + owner["wall_us"]
    )
    return {
        "components": base["components"],
        "outer_counts": base["outer_counts"],
        "metal_counts": counts,
        "resource_observation": {
            "owner": owner,
            "route": route,
            "prepare": prepare,
            "complete": complete,
            "derived": derived_observation,
            "terminal_take": terminal_take,
        },
        "intervals": base["intervals"],
    }


def ram_host_sparse_route_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    owner: Optional[dict[str, Any]],
    *,
    kernel: str,
    route_span: str,
    product_capped: bool,
) -> dict[str, Any]:
    base = ram_sparse_member_components(events, kernel, log_n)
    route_prefix = route_span.rsplit("::", 1)[0] + "::"
    unknown = {
        name
        for name, _, _ in span_intervals_us(events)
        if name.startswith(route_prefix) and name != route_span
    }
    if unknown:
        raise ValueError(f"{kernel} trace contains unknown Metal phases: {sorted(unknown)}")
    route_count = positive_span_count(events, route_span)
    if backend == "optimized":
        if route_count != 0 or owner is not None:
            raise ValueError(f"optimized {kernel} unexpectedly exercised Metal sparse work")
        return {
            "components": base["components"],
            "outer_counts": base["outer_counts"],
            "metal_counts": {"route": 0},
            "resource_observation": None,
            "intervals": base["intervals"],
        }
    if backend != "metal" or owner is None or route_count != 1:
        raise ValueError(f"Metal {kernel} sparse route is incomplete")

    fields = {
        "cycles",
        "log_t",
        "log_k",
        "requested",
        "selected",
        "fallback_reason",
        "source_generation",
        "source_fingerprint",
        "access_records",
        "increment_records",
        "additional_source_row_scans",
        "member_upload_bytes",
        "complete_sequence",
    }
    if product_capped:
        fields |= {"estimated_products", "product_cap"}
    raw = exact_span_args(events, route_span, fields)
    string_fields = {"requested", "selected", "fallback_reason"}
    observation = {
        **{field: trace_string(raw[field], field) for field in string_fields},
        "complete_sequence": trace_boolean(raw["complete_sequence"]),
        **{
            field: nonnegative_trace_integer(raw[field], f"{kernel} {field}")
            for field in fields - string_fields - {"complete_sequence"}
        },
    }
    if (
        observation["cycles"] != 1 << log_n
        or observation["log_t"] != log_n
        or observation["log_k"] != owner["log_k"]
        or observation["requested"] != "host_sparse_v1"
        or observation["selected"] != "host_sparse_v1"
        or observation["fallback_reason"] != "none"
        or observation["source_generation"] != owner["source_generation"]
        or observation["source_fingerprint"] != owner["source_fingerprint"]
        or observation["access_records"] != owner["access_records"]
        or observation["increment_records"] != owner["increment_records"]
        or observation["additional_source_row_scans"] != 0
        or observation["member_upload_bytes"] != 0
        or observation["complete_sequence"] is not True
        or (
            product_capped
            and (
                observation["estimated_products"] <= 0
                or observation["estimated_products"] > RAM_HAMMING_PRODUCT_CAP
                or observation["product_cap"] != RAM_HAMMING_PRODUCT_CAP
            )
        )
    ):
        raise ValueError(f"{kernel} sparse route receipt is inconsistent")
    route_interval = span_intervals_us(events, route_span)[0][1:]
    require_contained(route_interval, base["intervals"]["prepare"], f"{kernel} route")
    if owner["interval"][1] > base["intervals"]["prepare"][0]:
        raise ValueError(f"{kernel} starts before its shared owner is published")
    return {
        "components": base["components"],
        "outer_counts": base["outer_counts"],
        "metal_counts": {"route": 1},
        "resource_observation": {"route": observation},
        "intervals": base["intervals"],
    }


def ram_raf_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    witness: Optional[dict[str, Any]],
    read_write_prepare: tuple[float, float],
) -> dict[str, Any]:
    base = ram_sparse_member_components(
        events, RAM_RAF_EVALUATION_KERNEL, RAM_CYCLE_FAMILY_LOG_K
    )
    phases = ("submit", "join")
    names = {f"MetalRamRafEvaluation::{phase}" for phase in phases}
    unknown = {
        name
        for name, _, _ in span_intervals_us(events)
        if name.startswith("MetalRamRafEvaluation::") and name not in names
    }
    if unknown:
        raise ValueError(f"RAM RAF trace contains unknown Metal phases: {sorted(unknown)}")
    counts = {
        phase: positive_span_count(events, f"MetalRamRafEvaluation::{phase}")
        for phase in phases
    }
    if backend == "optimized":
        if any(counts.values()) or witness is not None:
            raise ValueError("optimized RAM RAF unexpectedly exercised Metal work")
        return {
            "components": base["components"],
            "outer_counts": base["outer_counts"],
            "metal_counts": counts,
            "resource_observation": None,
            "intervals": base["intervals"],
        }
    if backend != "metal" or witness is None or any(count != 1 for count in counts.values()):
        raise ValueError("Metal RAM RAF route is incomplete")

    submit_fields = {"cycles", "resident_address_bytes", "address_storage_id"}
    submit_raw = exact_span_args(
        events, "MetalRamRafEvaluation::submit", submit_fields
    )
    submit = {
        field: nonnegative_trace_integer(submit_raw[field], f"RAM RAF {field}")
        for field in submit_fields
    }
    if submit != {
        "cycles": 1 << log_n,
        "resident_address_bytes": 4 * (1 << log_n),
        "address_storage_id": witness["address_plane_storage_id"],
    }:
        raise ValueError("RAM RAF submit receipt is inconsistent")
    submit_interval = span_intervals_us(events, "MetalRamRafEvaluation::submit")[0][1:]
    join_interval = span_intervals_us(events, "MetalRamRafEvaluation::join")[0][1:]
    require_contained(submit_interval, read_write_prepare, "RAM RAF submit")
    containing_rounds = [
        interval
        for interval in base["intervals"]["prove_rounds"]
        if interval[0] <= join_interval[0] and join_interval[1] <= interval[1]
    ]
    if len(containing_rounds) != 1:
        raise ValueError("RAM RAF join lacks one enclosing RAF round")
    if witness["interval"][1] > submit_interval[0]:
        raise ValueError("RAM RAF submit precedes address-plane publication")
    return {
        "components": base["components"],
        "outer_counts": base["outer_counts"],
        "metal_counts": counts,
        "resource_observation": {
            "submit": submit,
            "submit_interval": submit_interval,
            "join_interval": join_interval,
        },
        "intervals": base["intervals"],
    }


def ram_cycle_family_breakdown(
    events: list[dict[str, Any]], backend: str, log_n: int
) -> dict[str, Any]:
    owner = ram_cycle_family_owner_observation(events, backend, log_n)
    witness = ram_cycle_family_witness_observation(
        events, backend, log_n, owner
    )
    read_write = ram_read_write_member_breakdown(
        events, backend, log_n, owner
    )
    raf = ram_raf_member_breakdown(
        events,
        backend,
        log_n,
        witness,
        read_write["intervals"]["prepare"],
    )
    val_check = ram_host_sparse_route_member_breakdown(
        events,
        backend,
        log_n,
        owner,
        kernel=RAM_VAL_CHECK_KERNEL,
        route_span=METAL_RAM_VAL_CHECK_ROUTE,
        product_capped=False,
    )
    ra_claim = ram_host_sparse_route_member_breakdown(
        events,
        backend,
        log_n,
        owner,
        kernel=RAM_RA_CLAIM_REDUCTION_KERNEL,
        route_span=METAL_RAM_RA_CLAIM_REDUCTION_ROUTE,
        product_capped=True,
    )
    hamming = ram_hamming_member_breakdown(events, backend, log_n, owner)
    ra_virtualization = ram_host_sparse_route_member_breakdown(
        events,
        backend,
        log_n,
        owner,
        kernel=RAM_RA_VIRTUALIZATION_KERNEL,
        route_span=METAL_RAM_RA_VIRTUALIZATION_ROUTE,
        product_capped=True,
    )
    members = {
        "raf_evaluation": raf,
        "read_write": read_write,
        "val_check": val_check,
        "ra_claim_reduction": ra_claim,
        "hamming_booleanity": hamming,
        "ra_virtualization": ra_virtualization,
    }
    canonical_intervals = sorted(
        (
            interval[0],
            interval[1],
            member_name,
        )
        for member_name, member in members.items()
        for interval in member["intervals"]["canonical"]
    )
    for left, right in zip(canonical_intervals, canonical_intervals[1:]):
        if left[1] > right[0]:
            raise ValueError(
                f"RAM family canonical spans overlap between {left[2]} and {right[2]}"
            )
    if witness is not None and any(
        witness["interval"][1] > interval[0]
        for interval in canonical_intervals
    ):
        raise ValueError("RAM family member work starts before witness publication")

    terminal = (
        hamming["resource_observation"]["terminal_take"]
        if hamming["resource_observation"] is not None
        else None
    )
    if backend == "metal":
        if terminal is None:
            raise ValueError("Metal RAM family lacks terminal ownership receipt")
        route_interval = span_intervals_us(
            events, METAL_RAM_RA_VIRTUALIZATION_ROUTE
        )[0][1:]
        require_contained(
            terminal["interval"], route_interval, "RAM cycle-family terminal take"
        )

    raw_member_us = sum(
        float(member["components"]["member_us"])
        for member in members.values()
    )
    witness_prepare_us = float(witness["wall_us"]) if witness is not None else 0.0
    for member in members.values():
        member.pop("intervals")
    return {
        "components": {
            "raw_member_us": raw_member_us,
            "witness_prepare_us": witness_prepare_us,
            "owner_prepare_us": float(owner["wall_us"]) if owner is not None else 0.0,
            "charged_member_us": raw_member_us + witness_prepare_us,
            "producer_charge_count": 1 if witness is not None else 0,
        },
        "members": members,
        "witness_prepare": witness,
        "owner": owner,
        "terminal_take": terminal,
        "canonical_span_count": len(canonical_intervals),
        "canonical_nonoverlap": True,
    }


def outer_remainder_storage_geometry(
    log_n: int,
    product_uniskip_carrier: bool = False,
    registers_claim_carrier: bool = False,
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
    initialization_bytes = 16 * sum(elements)
    carrier = None
    if registers_claim_carrier:
        prefix_elements = 1 << ((log_n + 1) // 2)
        suffix_elements = rows // prefix_elements
        blocks = min(suffix_elements, 256)
        carrier = {
            "prefix_elements": prefix_elements,
            "suffix_elements": suffix_elements,
            "blocks": blocks,
            "partial_bytes": 48 * blocks * prefix_elements,
            "component_bytes": 48 * prefix_elements,
            "rd_bytes": 8 * rows,
        }
        carrier["owned_bytes"] = (
            carrier["partial_bytes"]
            + carrier["component_bytes"]
            + carrier["rd_bytes"]
        )
    carrier_owned_bytes = 0 if carrier is None else carrier["owned_bytes"]
    carrier_maximum_bytes = 0 if carrier is None else max(
        carrier["partial_bytes"], carrier["component_bytes"], carrier["rd_bytes"]
    )
    return {
        "element_counts": elements,
        "initialization_bytes": initialization_bytes,
        "owned_bytes": initialization_bytes + carrier_owned_bytes,
        "maximum_buffer_bytes": max(16 * max(elements), carrier_maximum_bytes),
        "registers_claim_carrier": carrier,
    }


def outer_remainder_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    cutoff_log2: int = 16,
    trace_cutoff_log2: int = 18,
    product_uniskip_carrier: bool = False,
    registers_claim_carrier: bool = False,
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
        registers_claim_carrier_observation = None
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
            "registers_claim_carrier_park": int(registers_claim_carrier),
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
        geometry = outer_remainder_storage_geometry(
            log_n, product_uniskip_carrier, registers_claim_carrier
        )
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
            or storage["initialization_bytes"] != geometry["initialization_bytes"]
            or storage["current_device_bytes"] + storage["planned_device_bytes"]
            > storage["recommended_max_working_set_bytes"]
            or any(identity <= 0 for identity in buffer_ids)
            or len(set(buffer_ids)) != 9
            or initialization["mode"] != "full"
            or initialization["device_buffers"] != 9
            or initialization["bytes"] != geometry["initialization_bytes"]
            or initialization["protocol_dispatches"] != 0
            or initialization_ids != buffer_ids
            or completion["mode"] != "full"
            or completion["command_completed"] is not True
            or completion["bytes"] != geometry["initialization_bytes"]
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
            or sequence["preinitialized_device_bytes"] != geometry["initialization_bytes"]
            or sequence["initialization_bytes"] != geometry["initialization_bytes"]
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

        registers_claim_carrier_observation = None
        if registers_claim_carrier:
            carrier_fields = {
                "rows",
                "explicit_rows",
                "prefix_elements",
                "suffix_elements",
                "blocks",
                "device_registry_id",
                "source_generation",
                "completion_serial",
                "source_compact_storage_id",
                "source_residual_storage_id",
                "partial_storage_id",
                "component_storage_id",
                "rd_storage_id",
                "partial_bytes",
                "component_bytes",
                "component_host_read_bytes",
                "rd_bytes",
                "scratch_release_bytes",
                "retained_rd_bytes",
                "source_allocations",
                "row_scans",
                "carrier_dispatches",
                "command_buffers",
                "waits",
                "uploads",
                "prezero_dispatches",
                "complete_overwrite",
                "stage1_carry_parks",
            }
            carrier_raw = exact_span_args(
                events,
                "MetalOuterRemainder::registers_claim_carrier_park",
                carrier_fields,
            )
            registers_claim_carrier_observation = {
                field: nonnegative_trace_integer(carrier_raw[field], field)
                for field in carrier_fields - {"complete_overwrite"}
            }
            registers_claim_carrier_observation["complete_overwrite"] = trace_boolean(
                carrier_raw["complete_overwrite"]
            )
            expected_carrier = geometry["registers_claim_carrier"]
            assert expected_carrier is not None
            identities = [
                registers_claim_carrier_observation[field]
                for field in (
                    "source_compact_storage_id",
                    "source_residual_storage_id",
                    "partial_storage_id",
                    "component_storage_id",
                    "rd_storage_id",
                )
            ]
            if (
                registers_claim_carrier_observation["rows"] != rows
                or registers_claim_carrier_observation["explicit_rows"] > rows
                or registers_claim_carrier_observation["prefix_elements"]
                != expected_carrier["prefix_elements"]
                or registers_claim_carrier_observation["suffix_elements"]
                != expected_carrier["suffix_elements"]
                or registers_claim_carrier_observation["blocks"]
                != expected_carrier["blocks"]
                or registers_claim_carrier_observation["device_registry_id"]
                != handoff["device_registry_id"]
                or registers_claim_carrier_observation["source_generation"] <= 0
                or registers_claim_carrier_observation["completion_serial"] <= 0
                or registers_claim_carrier_observation["source_compact_storage_id"]
                != handoff["compact_rows_storage_id"]
                or registers_claim_carrier_observation["source_residual_storage_id"]
                != handoff["residual_rows_storage_id"]
                or any(identity <= 0 for identity in identities)
                or len(set(identities)) != len(identities)
                or registers_claim_carrier_observation["partial_bytes"]
                != expected_carrier["partial_bytes"]
                or registers_claim_carrier_observation["component_bytes"]
                != expected_carrier["component_bytes"]
                or registers_claim_carrier_observation["component_host_read_bytes"]
                != expected_carrier["component_bytes"]
                or registers_claim_carrier_observation["rd_bytes"]
                != expected_carrier["rd_bytes"]
                or registers_claim_carrier_observation["scratch_release_bytes"]
                != expected_carrier["partial_bytes"] + expected_carrier["component_bytes"]
                or registers_claim_carrier_observation["retained_rd_bytes"]
                != expected_carrier["rd_bytes"]
                or registers_claim_carrier_observation["source_allocations"] != 3
                or registers_claim_carrier_observation["row_scans"] != 2
                or registers_claim_carrier_observation["carrier_dispatches"] != 3
                or registers_claim_carrier_observation["command_buffers"] != 1
                or registers_claim_carrier_observation["waits"] != 1
                or registers_claim_carrier_observation["uploads"] != 0
                or registers_claim_carrier_observation["prezero_dispatches"] != 0
                or registers_claim_carrier_observation["complete_overwrite"] is not True
                or registers_claim_carrier_observation["stage1_carry_parks"] != 1
            ):
                raise ValueError("OuterRemainder RegistersClaim carrier is inconsistent")

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
            "registers_claim_carrier": registers_claim_carrier_observation,
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
        "registers_claim_carrier": registers_claim_carrier_observation,
    }


def registers_claim_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    implementation: str,
    outer_carrier: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if backend not in {"optimized", "metal"}:
        raise ValueError(f"unsupported RegistersClaim backend {backend!r}")
    component_names = {
        f"{REGISTERS_CLAIM_KERNEL}::{component}"
        for component in ("prepare", "prove_round", "finish_rounds", "output_claims")
    }
    metal_names = {
        "MetalRegistersClaimReduction::route",
        "MetalRegistersClaimReduction::prepare",
        "MetalRegistersClaimReduction::midpoint_projection",
        "MetalInstructionInput::registers_claim_alias_publish",
    }
    unknown = {
        name
        for event in events
        if isinstance((name := event.get("name")), str)
        and name.startswith("MetalRegistersClaimReduction::")
        and name not in metal_names
    }
    if unknown:
        raise ValueError(
            f"RegistersClaim trace contains unknown Metal phases: {sorted(unknown)}"
        )
    intervals = strict_named_intervals(
        events, component_names | metal_names | {PIOP_SPAN, SUMCHECK_ROUND_SPAN}
    )
    if len(intervals[PIOP_SPAN]) != 1:
        raise ValueError("trace must contain one PIOP span for RegistersClaim")
    piop = intervals[PIOP_SPAN][0]
    by_component = {
        component: sorted(intervals[f"{REGISTERS_CLAIM_KERNEL}::{component}"])
        for component in ("prepare", "prove_round", "finish_rounds", "output_claims")
    }
    outer_counts = {name: len(values) for name, values in by_component.items()}
    expected_outer_counts = {
        "prepare": 1,
        "prove_round": log_n,
        "finish_rounds": 1,
        "output_claims": 1,
    }
    if outer_counts != expected_outer_counts:
        raise ValueError(
            f"RegistersClaim member span counts {outer_counts}, expected {expected_outer_counts}"
        )
    prepare = by_component["prepare"][0]
    rounds = by_component["prove_round"]
    finish = by_component["finish_rounds"][0]
    output = by_component["output_claims"][0]
    ordered = [prepare, *rounds, finish, output]
    if any(start < piop[0] or end > piop[1] for start, end in ordered):
        raise ValueError("a RegistersClaim member span lies outside PIOP")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("RegistersClaim member spans overlap or appear out of order")

    metal_counts = {name: len(intervals[name]) for name in metal_names}
    resource_observation = None
    if backend == "optimized":
        if any(metal_counts.values()) or outer_carrier is not None:
            raise ValueError("optimized RegistersClaim unexpectedly contains Metal state")
    elif implementation != "outer-carrier-alias-hybrid":
        if any(metal_counts.values()) or outer_carrier is not None:
            raise ValueError("CPU RegistersClaim unexpectedly contains Metal state")
    else:
        expected_metal_counts = {name: 1 for name in metal_names}
        if metal_counts != expected_metal_counts or outer_carrier is None:
            raise ValueError("RegistersClaim Metal carrier lifecycle is incomplete")
        route_interval = intervals["MetalRegistersClaimReduction::route"][0]
        prepare_interval = intervals["MetalRegistersClaimReduction::prepare"][0]
        midpoint_interval = intervals[
            "MetalRegistersClaimReduction::midpoint_projection"
        ][0]
        alias_interval = intervals[
            "MetalInstructionInput::registers_claim_alias_publish"
        ][0]
        require_contained(route_interval, prepare, "RegistersClaim route")
        require_contained(prepare_interval, prepare, "RegistersClaim Metal prepare")
        prefix_vars = (log_n + 1) // 2
        require_contained(
            midpoint_interval,
            rounds[prefix_vars],
            "RegistersClaim midpoint projection",
        )
        if alias_interval[1] > midpoint_interval[0]:
            raise ValueError("RegistersClaim alias publication did not precede its midpoint")
        enclosing_alias_rounds = [
            interval
            for interval in intervals[SUMCHECK_ROUND_SPAN]
            if interval[0] <= alias_interval[0] and alias_interval[1] <= interval[1]
        ]
        enclosing_midpoint_rounds = [
            interval
            for interval in intervals[SUMCHECK_ROUND_SPAN]
            if interval[0] <= midpoint_interval[0]
            and midpoint_interval[1] <= interval[1]
        ]
        if (
            len(enclosing_alias_rounds) != 1
            or enclosing_alias_rounds != enclosing_midpoint_rounds
        ):
            raise ValueError("RegistersClaim alias and midpoint cross a Fiat-Shamir seam")

        route_fields = {
            "cycles",
            "requested",
            "stage1_carry_present",
            "alias_receiver_present",
            "realized_route",
            "fallback_reason",
        }
        route_raw = exact_span_args(
            events, "MetalRegistersClaimReduction::route", route_fields
        )
        route = {
            "cycles": nonnegative_trace_integer(route_raw["cycles"], "cycles"),
            "requested": trace_string(route_raw["requested"], "requested"),
            "stage1_carry_present": trace_boolean(route_raw["stage1_carry_present"]),
            "alias_receiver_present": trace_boolean(route_raw["alias_receiver_present"]),
            "realized_route": trace_string(
                route_raw["realized_route"], "realized_route"
            ),
            "fallback_reason": trace_string(
                route_raw["fallback_reason"], "fallback_reason"
            ),
        }
        prepare_fields = {
            "cycles",
            "requested",
            "realized_route",
            "fallback_reason",
            "resident_bytes",
            "source_allocations",
            "source_upload_bytes",
            "source_host_write_bytes",
            "source_generation",
            "source_compact_storage_id",
            "source_rd_storage_id",
            "alias_generation",
        }
        prepare_raw = exact_span_args(
            events, "MetalRegistersClaimReduction::prepare", prepare_fields
        )
        prepare_receipt = {
            field: nonnegative_trace_integer(prepare_raw[field], field)
            for field in prepare_fields
            - {"requested", "realized_route", "fallback_reason"}
        }
        prepare_receipt.update(
            {
                field: trace_string(prepare_raw[field], field)
                for field in ("requested", "realized_route", "fallback_reason")
            }
        )
        alias_fields = {
            "rows",
            "source_compact_storage_id",
            "alias_generation",
            "prefix_challenges",
            "table_0",
            "table_1",
            "host_table_copies",
            "snapshot_host_bytes",
            "publishes",
        }
        alias_raw = exact_span_args(
            events,
            "MetalInstructionInput::registers_claim_alias_publish",
            alias_fields,
        )
        alias = {
            field: nonnegative_trace_integer(alias_raw[field], field)
            for field in alias_fields
        }
        midpoint_fields = {
            "source",
            "round",
            "rows",
            "source_generation",
            "device_registry_id",
            "source_compact_storage_id",
            "source_rd_storage_id",
            "alias_generation",
            "rd_source_bytes",
            "eq_upload_bytes",
            "readback_bytes",
            "device_allocations",
            "dispatches",
            "command_buffers",
            "waits",
            "alias_takes",
            "useful_half_width_terms",
            "gpu_active_ns",
            "resident_wall_ns",
        }
        midpoint_raw = exact_span_args(
            events,
            "MetalRegistersClaimReduction::midpoint_projection",
            midpoint_fields,
        )
        midpoint = {
            field: nonnegative_trace_integer(midpoint_raw[field], field)
            for field in midpoint_fields - {"source"}
        }
        midpoint["source"] = trace_string(midpoint_raw["source"], "source")

        rows = 1 << log_n
        prefix_elements = 1 << prefix_vars
        suffix_elements = rows // prefix_elements
        if (
            route
            != {
                "cycles": rows,
                "requested": "outer_carrier_alias_hybrid",
                "stage1_carry_present": True,
                "alias_receiver_present": True,
                "realized_route": "outer_carrier_alias_hybrid",
                "fallback_reason": "none",
            }
            or prepare_receipt["cycles"] != rows
            or prepare_receipt["requested"] != "outer_carrier_alias_hybrid"
            or prepare_receipt["realized_route"] != "outer_carrier_alias_hybrid"
            or prepare_receipt["fallback_reason"] != "none"
            or prepare_receipt["resident_bytes"] != 8 * rows
            or any(
                prepare_receipt[field] != 0
                for field in (
                    "source_allocations",
                    "source_upload_bytes",
                    "source_host_write_bytes",
                )
            )
            or min(
                prepare_receipt["source_generation"],
                prepare_receipt["source_compact_storage_id"],
                prepare_receipt["source_rd_storage_id"],
                prepare_receipt["alias_generation"],
            )
            <= 0
            or prepare_receipt["source_generation"]
            != outer_carrier["source_generation"]
            or prepare_receipt["source_compact_storage_id"]
            != outer_carrier["source_compact_storage_id"]
            or prepare_receipt["source_rd_storage_id"]
            != outer_carrier["rd_storage_id"]
            or alias
            != {
                "rows": rows,
                "source_compact_storage_id": prepare_receipt[
                    "source_compact_storage_id"
                ],
                "alias_generation": prepare_receipt["alias_generation"],
                "prefix_challenges": prefix_vars,
                "table_0": 1,
                "table_1": 5,
                "host_table_copies": 2,
                "snapshot_host_bytes": 32 * suffix_elements,
                "publishes": 1,
            }
            or midpoint["source"] != "outer_carrier_alias"
            or midpoint["round"] != prefix_vars
            or midpoint["rows"] != rows
            or midpoint["source_generation"] != prepare_receipt["source_generation"]
            or midpoint["device_registry_id"] != outer_carrier["device_registry_id"]
            or midpoint["source_compact_storage_id"]
            != prepare_receipt["source_compact_storage_id"]
            or midpoint["source_rd_storage_id"]
            != prepare_receipt["source_rd_storage_id"]
            or midpoint["alias_generation"] != prepare_receipt["alias_generation"]
            or midpoint["rd_source_bytes"] != 8 * rows
            or midpoint["eq_upload_bytes"] != 16 * prefix_elements
            or midpoint["readback_bytes"] != 16 * suffix_elements
            or midpoint["device_allocations"] != 2
            or midpoint["dispatches"] != 1
            or midpoint["command_buffers"] != 1
            or midpoint["waits"] != 1
            or midpoint["alias_takes"] != 1
            or midpoint["useful_half_width_terms"] != rows
            or midpoint["gpu_active_ns"] <= 0
            or midpoint["resident_wall_ns"] < midpoint["gpu_active_ns"]
        ):
            raise ValueError("RegistersClaim Metal lifecycle is inconsistent")
        resource_observation = {
            "route": route,
            "prepare": prepare_receipt,
            "alias_publish": alias,
            "midpoint": midpoint,
            "outer_carrier": outer_carrier,
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
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in components.values()
        if not isinstance(value, list)
    ):
        raise ValueError("RegistersClaim member duration is invalid")
    return {
        "components": components,
        "outer_counts": outer_counts,
        "metal_counts": metal_counts,
        "resource_observation": resource_observation,
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


def instruction_read_raf_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    cutoff_log2: int = 16,
    scatter_threads: int = 256,
    expect_fused_bytecode_address: bool = False,
) -> dict[str, Any]:
    if backend not in {"optimized", "metal"}:
        raise ValueError(f"unsupported InstructionReadRaf backend {backend!r}")
    if log_n < 1 or not 0 < cutoff_log2 < log_n:
        raise ValueError("invalid InstructionReadRaf evaluator geometry")
    if backend == "metal" and log_n < 25:
        raise ValueError("Stage1Grouped InstructionReadRaf requires log-n at least 25")
    if scatter_threads not in {128, 256, 512, 1024}:
        raise ValueError("invalid InstructionReadRaf scatter width")

    outer_names = {
        f"{INSTRUCTION_READ_RAF_KERNEL}::{component}"
        for component in INSTRUCTION_READ_RAF_COMPONENTS
    }
    metal_names = {
        f"Metal{INSTRUCTION_READ_RAF_KERNEL}::{phase}"
        for phase in INSTRUCTION_READ_RAF_METAL_PHASES
    } | {
        INSTRUCTION_READ_RAF_STAGE1_SOURCE,
        INSTRUCTION_READ_RAF_STAGE1_SCATTER,
        INSTRUCTION_READ_RAF_STAGE1_SEQUENCE,
    }
    supporting_names = {
        PIOP_SPAN,
        BACKEND_WITNESS_PREP_SPAN,
        METAL_INSTRUCTION_INPUT_ROWS_PREPARE,
        "MetalSpartanDense::witness_prepare",
    }
    intervals = strict_named_intervals(
        events, outer_names | metal_names | supporting_names
    )
    if len(intervals[PIOP_SPAN]) != 1 or len(
        intervals[BACKEND_WITNESS_PREP_SPAN]
    ) != 1:
        raise ValueError("InstructionReadRaf requires one PIOP and witness-prepare span")
    piop = intervals[PIOP_SPAN][0]
    witness_prepare = intervals[BACKEND_WITNESS_PREP_SPAN][0]
    if witness_prepare[1] > piop[0]:
        raise ValueError("InstructionReadRaf witness preparation overlaps PIOP")

    by_component = {
        component: sorted(intervals[f"{INSTRUCTION_READ_RAF_KERNEL}::{component}"])
        for component in INSTRUCTION_READ_RAF_COMPONENTS
    }
    outer_counts = {
        component: len(component_intervals)
        for component, component_intervals in by_component.items()
    }
    expected_outer_counts = {
        "prepare": 1,
        "prove_round": 128 + log_n,
        "finish_rounds": 1,
        "output_claims": 1,
    }
    if outer_counts != expected_outer_counts:
        raise ValueError(
            f"InstructionReadRaf member span counts {outer_counts}, "
            f"expected {expected_outer_counts}"
        )
    prepare = by_component["prepare"][0]
    rounds = by_component["prove_round"]
    finish = by_component["finish_rounds"][0]
    output = by_component["output_claims"][0]
    ordered = [prepare, *rounds, finish, output]
    if any(start < piop[0] or end > piop[1] for start, end in ordered):
        raise ValueError("an InstructionReadRaf member span lies outside PIOP")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("InstructionReadRaf member spans overlap or are out of order")

    components = {
        "prepare_us": interval_duration_us(prepare),
        "rounds_us": [interval_duration_us(interval) for interval in rounds],
        "finish_us": interval_duration_us(finish),
        "output_claims_us": interval_duration_us(output),
    }
    components["rounds_total_us"] = sum(components["rounds_us"])
    components["member_us"] = (
        components["prepare_us"]
        + components["rounds_total_us"]
        + components["finish_us"]
        + components["output_claims_us"]
    )
    if any(
        not math.isfinite(float(components[field]))
        or float(components[field]) <= 0.0
        for field in (
            "prepare_us",
            "rounds_total_us",
            "finish_us",
            "output_claims_us",
            "member_us",
        )
    ):
        raise ValueError("InstructionReadRaf has a non-positive member duration")

    observed_metal_names = {
        name
        for name, _, _ in span_intervals_us(events)
        if name.startswith(f"Metal{INSTRUCTION_READ_RAF_KERNEL}::")
    }
    unknown_metal_names = observed_metal_names - metal_names
    if unknown_metal_names:
        raise ValueError(
            "InstructionReadRaf trace contains legacy or unknown Metal phases: "
            f"{sorted(unknown_metal_names)}"
        )
    metal_counts = {
        name.rsplit("::", 1)[-1]: len(intervals[name]) for name in metal_names
    }
    stage1_expected = backend == "metal"
    if not stage1_expected:
        if any(metal_counts.values()):
            raise ValueError("InstructionReadRaf CPU route unexpectedly exercised Metal")
        if intervals[METAL_INSTRUCTION_INPUT_ROWS_PREPARE] or intervals[
            "MetalSpartanDense::witness_prepare"
        ]:
            raise ValueError("InstructionReadRaf CPU route published a Stage1 owner")
        return {
            "components": components,
            "outer_counts": outer_counts,
            "metal_counts": metal_counts,
            "source_observation": None,
            "scatter_observation": None,
            "scatter_wall_us": None,
            "fused_bytecode_observation": None,
            "stage1_projection": None,
            "resource_observation": None,
        }

    expected_metal_counts = {
        "stage1_source_publish": 1,
        "stage1_grouped_scatter": 1,
        "stage1_grouped_sequence_prepare": 1,
        "address_round": 129,
        "resident_first_message": 1,
        "resident_handoff": 1,
        "resident_round": log_n - cutoff_log2 - 1,
        "readback": 1,
    }
    if metal_counts != expected_metal_counts:
        raise ValueError(
            f"InstructionReadRaf Metal span counts {metal_counts}, "
            f"expected {expected_metal_counts}"
        )
    if any(
        start < piop[0] or end > piop[1]
        for name in metal_names - {INSTRUCTION_READ_RAF_STAGE1_SOURCE}
        for start, end in intervals[name]
    ):
        raise ValueError("an InstructionReadRaf Metal phase lies outside PIOP")

    source_interval = intervals[INSTRUCTION_READ_RAF_STAGE1_SOURCE][0]
    scatter_interval = intervals[INSTRUCTION_READ_RAF_STAGE1_SCATTER][0]
    sequence_interval = intervals[INSTRUCTION_READ_RAF_STAGE1_SEQUENCE][0]
    require_contained(
        source_interval,
        witness_prepare,
        "InstructionReadRaf Stage1 source publication",
    )
    require_contained(
        scatter_interval, prepare, "InstructionReadRaf Stage1 grouped scatter"
    )
    require_contained(
        sequence_interval, prepare, "InstructionReadRaf resident sequence preparation"
    )
    if source_interval[1] > piop[0] or scatter_interval[1] > sequence_interval[0]:
        raise ValueError("InstructionReadRaf Stage1 phases are out of order")

    phase_rounds = (
        (
            "address_round",
            sorted(intervals["MetalInstructionReadRaf::address_round"]),
            rounds[:129],
        ),
        (
            "resident_first_message",
            intervals["MetalInstructionReadRaf::resident_first_message"],
            rounds[128:129],
        ),
        (
            "resident_handoff",
            intervals["MetalInstructionReadRaf::resident_handoff"],
            rounds[129:130],
        ),
        (
            "resident_round",
            sorted(intervals["MetalInstructionReadRaf::resident_round"]),
            rounds[130 : 130 + log_n - cutoff_log2 - 1],
        ),
    )
    for phase, inner_intervals, outer_intervals in phase_rounds:
        if len(inner_intervals) != len(outer_intervals):
            raise ValueError(f"InstructionReadRaf {phase} round mapping is incomplete")
        for index, (inner, outer) in enumerate(zip(inner_intervals, outer_intervals)):
            require_contained(
                inner,
                outer,
                f"InstructionReadRaf {phase} round {index}",
            )
    final_address = sorted(intervals["MetalInstructionReadRaf::address_round"])[-1]
    resident_first_message = intervals[
        "MetalInstructionReadRaf::resident_first_message"
    ][0]
    if final_address[1] > resident_first_message[0]:
        raise ValueError(
            "InstructionReadRaf final address round overlaps its resident first message"
        )
    resident_rounds = log_n - cutoff_log2 - 1
    require_contained(
        intervals["MetalInstructionReadRaf::readback"][0],
        rounds[130 + resident_rounds],
        "InstructionReadRaf readback",
    )

    source_fields = {
        "rows",
        "row_bytes",
        "claim_bytes",
        "resident_device_bytes",
        "count_chunks",
        "count_bytes",
        "host_row_write_bytes",
        "host_claim_write_bytes",
        "host_count_update_bytes",
        "row_allocation_identity",
        "claim_allocation_identity",
        "count_allocation_identity",
        "device_registry_id",
        "source_generation",
        "completion_serial",
        "count_order",
        "publication_kind",
        "complete_overwrite",
        "source_windows",
        "member_upload_bytes",
        "projection_dispatches",
    }
    source_args = exact_span_args(
        events, INSTRUCTION_READ_RAF_STAGE1_SOURCE, source_fields
    )
    source = {
        field: nonnegative_trace_integer(value, f"Stage1 source {field}")
        for field, value in source_args.items()
        if field
        not in {"count_order", "publication_kind", "complete_overwrite"}
    }
    source["count_order"] = trace_string(source_args["count_order"], "count_order")
    source["publication_kind"] = trace_string(
        source_args["publication_kind"], "publication_kind"
    )
    source["complete_overwrite"] = trace_boolean(
        source_args["complete_overwrite"]
    )
    rows = 1 << log_n
    chunks = rows // 4096
    expected_source = {
        "rows": rows,
        "row_bytes": 40 * rows,
        "claim_bytes": rows,
        "resident_device_bytes": 41 * rows,
        "count_chunks": chunks,
        "count_bytes": 328 * chunks,
        "host_row_write_bytes": 40 * rows,
        "host_claim_write_bytes": rows,
        "host_count_update_bytes": 4 * rows,
        "count_order": "table_major_then_none_v1",
        "publication_kind": "host_fill_v1",
        "complete_overwrite": True,
        "source_windows": rows,
        "member_upload_bytes": 0,
        "projection_dispatches": 0,
    }
    if any(source[field] != value for field, value in expected_source.items()):
        raise ValueError(f"InstructionReadRaf Stage1 source ledger is invalid: {source}")
    source_id_fields = (
        "row_allocation_identity",
        "claim_allocation_identity",
        "count_allocation_identity",
    )
    source_ids = [source[field] for field in source_id_fields]
    if (
        any(identity <= 0 for identity in source_ids)
        or len(set(source_ids)) != len(source_ids)
        or source["device_registry_id"] <= 0
        or source["source_generation"] <= 0
        or source["completion_serial"] <= 0
    ):
        raise ValueError("InstructionReadRaf Stage1 source provenance is invalid")

    compact_fields = {
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
        "resident_rows",
        "explicit_rows",
        "compact_rows_storage_id",
        "residual_rows_storage_id",
    }
    compact_args = exact_span_args(
        events, METAL_INSTRUCTION_INPUT_ROWS_PREPARE, compact_fields
    )
    compact = {
        field: nonnegative_trace_integer(value, f"Stage1 compact rows {field}")
        for field, value in compact_args.items()
        if field != "source_kind"
    }
    compact["source_kind"] = trace_string(
        compact_args["source_kind"], "compact source_kind"
    )
    if len(intervals[METAL_INSTRUCTION_INPUT_ROWS_PREPARE]) != 1:
        raise ValueError("InstructionReadRaf requires one compact-row producer")
    require_contained(
        intervals[METAL_INSTRUCTION_INPUT_ROWS_PREPARE][0],
        witness_prepare,
        "InstructionReadRaf compact-row producer",
    )
    compact_expected = {
        "source_kind": "owned_random_access",
        "witness_row_extractions": rows,
        "residual_rows_written": rows,
        "compact_rows_written": rows,
        "compact_row_bytes": 48,
        "residual_row_bytes": 112,
        "compact_allocations": 1,
        "residual_allocations": 1,
        "full_row_allocations": 0,
        "full_domain_copy_bytes": 0,
        "full_domain_copy_dispatches": 0,
        "host_repack_rows": 0,
        "resident_rows": rows,
    }
    if any(compact[field] != value for field, value in compact_expected.items()):
        raise ValueError("InstructionReadRaf compact-row production is inconsistent")
    if (
        compact["explicit_rows"] > rows
        or compact["compact_rows_storage_id"] <= 0
        or compact["residual_rows_storage_id"] <= 0
        or compact["compact_rows_storage_id"]
        == compact["residual_rows_storage_id"]
    ):
        raise ValueError("InstructionReadRaf compact-row storage provenance is invalid")

    witness_fields = {
        "cycles",
        "source",
        "admitted",
        "fallback_reason",
        "native_register_contract_bytes",
        "owner_generation",
        "shift_late_copy_dispatches",
        "shift_resident_bytes",
        "shift_row_extractions",
    }
    witness_args = exact_span_args(
        events, "MetalSpartanDense::witness_prepare", witness_fields
    )
    if len(intervals["MetalSpartanDense::witness_prepare"]) != 1:
        raise ValueError("InstructionReadRaf requires one Stage1 projection")
    require_contained(
        intervals["MetalSpartanDense::witness_prepare"][0],
        witness_prepare,
        "InstructionReadRaf Stage1 projection",
    )
    stage1_projection = {
        field: nonnegative_trace_integer(value, f"Stage1 projection {field}")
        for field, value in witness_args.items()
        if field not in {"source", "admitted", "fallback_reason"}
    }
    stage1_projection["source"] = trace_string(witness_args["source"], "source")
    stage1_projection["admitted"] = trace_boolean(witness_args["admitted"])
    stage1_projection["fallback_reason"] = trace_string(
        witness_args["fallback_reason"], "fallback_reason"
    )
    if (
        stage1_projection["cycles"] != rows
        or stage1_projection["source"] != "stage1_single_projection"
        or stage1_projection["admitted"] is not True
        or stage1_projection["fallback_reason"] != "none"
        or stage1_projection["owner_generation"] <= 0
        or stage1_projection["shift_row_extractions"] != rows
        or stage1_projection["shift_late_copy_dispatches"] != 0
        or stage1_projection["native_register_contract_bytes"] <= 0
        or stage1_projection["shift_resident_bytes"] <= 0
    ):
        raise ValueError("InstructionReadRaf Stage1 projection is inconsistent")

    scatter_fields = {
        "rows",
        "preparation_wall_ns",
        "command_wall_ns",
        "gpu_active_ns",
        "status_readback_bytes",
        "packed_rows_bytes",
        "lookups_bytes",
        "inverse_bytes",
        "weights_bytes",
        "packed_rows_identity",
        "lookups_identity",
        "inverse_identity",
        "weights_identity",
        "source_generation",
        "source_completion_serial",
        "source_row_allocation_identity",
        "source_claim_allocation_identity",
        "source_count_allocation_identity",
        "source_count_chunks",
        "source_count_bytes",
        "source_count_order",
        "source_device_registry_id",
        "scatter_completion_serial",
        "e_in_length",
        "e_out_length",
        "command_buffers",
        "encoders",
        "waits",
        "dispatches",
        "threadgroups",
        "threads_per_threadgroup",
        "dynamic_threadgroup_bytes",
        "static_threadgroup_bytes",
        "source_copy_bytes",
        "full_plane_readback_bytes",
        "complete_overwrite",
        "additional_allocation_bytes",
    }
    fused_scatter_fields = {
        "bytecode_fused",
        "bytecode_physical_rows",
        "bytecode_descriptor_elements",
        "bytecode_descriptor_bytes",
        "bytecode_descriptor_storage_id",
        "bytecode_pivot_elements",
        "bytecode_pivot_bytes",
        "bytecode_pivot_storage_id",
        "bytecode_chunk_offset_elements",
        "bytecode_chunk_offset_bytes",
        "bytecode_chunk_offset_storage_id",
        "bytecode_work_items",
        "bytecode_work_item_bytes",
        "bytecode_work_item_storage_id",
        "bytecode_address_offset_elements",
        "bytecode_address_offset_bytes",
        "bytecode_address_offset_storage_id",
        "bytecode_occurrence_bytes",
        "bytecode_occurrence_storage_id",
        "bytecode_magnitude_bytes",
        "bytecode_magnitude_storage_id",
        "bytecode_max_descriptors_per_chunk",
        "bytecode_max_admitted_descriptors_per_chunk",
        "bytecode_max_pivots_per_chunk",
        "bytecode_max_admitted_pivots_per_chunk",
        "bytecode_dynamic_threadgroup_bytes",
        "bytecode_threadgroup_memory_limit_bytes",
        "shared_source_row_scans",
        "additional_source_row_scans",
        "member_upload_bytes",
    }
    scatter_args = unique_span_args(events, INSTRUCTION_READ_RAF_STAGE1_SCATTER)
    expected_scatter_fields = scatter_fields | (
        fused_scatter_fields if expect_fused_bytecode_address else set()
    )
    if set(scatter_args) != expected_scatter_fields:
        raise ValueError(
            f"{INSTRUCTION_READ_RAF_STAGE1_SCATTER} has unexpected argument fields"
        )
    scatter = {
        field: nonnegative_trace_integer(value, f"Stage1 scatter {field}")
        for field, value in scatter_args.items()
        if field
        not in {"source_count_order", "complete_overwrite", "bytecode_fused"}
    }
    scatter["source_count_order"] = trace_string(
        scatter_args["source_count_order"], "source_count_order"
    )
    scatter["complete_overwrite"] = trace_boolean(
        scatter_args["complete_overwrite"]
    )
    if expect_fused_bytecode_address:
        scatter["bytecode_fused"] = trace_boolean(
            scatter_args["bytecode_fused"]
        )
    e_out = 1 << (log_n // 2)
    e_in = rows // e_out
    expected_additional_bytes = (
        37 * rows
        + 328 * chunks
        + 332
        + 16 * (e_in + e_out)
        + 4
        + 88
    )
    if expect_fused_bytecode_address:
        expected_additional_bytes += 10 * compact["explicit_rows"]
    expected_scatter = {
        "rows": rows,
        "status_readback_bytes": 4,
        "packed_rows_bytes": rows,
        "lookups_bytes": 16 * rows,
        "inverse_bytes": 4 * rows,
        "weights_bytes": 16 * rows,
        "source_generation": source["source_generation"],
        "source_completion_serial": source["completion_serial"],
        "source_row_allocation_identity": source["row_allocation_identity"],
        "source_claim_allocation_identity": source["claim_allocation_identity"],
        "source_count_allocation_identity": source["count_allocation_identity"],
        "source_count_chunks": source["count_chunks"],
        "source_count_bytes": source["count_bytes"],
        "source_count_order": source["count_order"],
        "source_device_registry_id": source["device_registry_id"],
        "e_in_length": e_in,
        "e_out_length": e_out,
        "command_buffers": 1,
        "encoders": 1,
        "waits": 1,
        "dispatches": 1,
        "threadgroups": chunks,
        "threads_per_threadgroup": scatter_threads,
        "dynamic_threadgroup_bytes": 328,
        "static_threadgroup_bytes": 0,
        "source_copy_bytes": 0,
        "full_plane_readback_bytes": 0,
        "complete_overwrite": True,
        "additional_allocation_bytes": expected_additional_bytes,
    }
    if expect_fused_bytecode_address:
        expected_scatter["dynamic_threadgroup_bytes"] = scatter[
            "bytecode_dynamic_threadgroup_bytes"
        ]
    if any(scatter[field] != value for field, value in expected_scatter.items()):
        raise ValueError(f"InstructionReadRaf Stage1 scatter ledger is invalid: {scatter}")
    output_ids = [
        scatter[field]
        for field in (
            "packed_rows_identity",
            "lookups_identity",
            "inverse_identity",
            "weights_identity",
        )
    ]
    if (
        any(identity <= 0 for identity in output_ids)
        or len(set(output_ids)) != len(output_ids)
        or set(output_ids) & set(source_ids)
        or scatter["scatter_completion_serial"] <= 0
        or scatter["preparation_wall_ns"] <= 0
        or scatter["command_wall_ns"] <= 0
        or scatter["gpu_active_ns"] <= 0
        or interval_duration_us(scatter_interval) <= 0.0
        or scatter["gpu_active_ns"] > scatter["command_wall_ns"]
        or (scatter["preparation_wall_ns"] + scatter["command_wall_ns"])
        > int(components["prepare_us"] * 1000.0)
    ):
        raise ValueError("InstructionReadRaf Stage1 scatter execution is invalid")

    fused_bytecode = None
    if expect_fused_bytecode_address:
        physical_rows = compact["explicit_rows"]
        work_items = scatter["bytecode_work_items"]
        descriptor_elements = scatter["bytecode_descriptor_elements"]
        pivot_elements = scatter["bytecode_pivot_elements"]
        bytecode_chunks = (physical_rows + 4095) // 4096
        chunk_offset_elements = 2 * bytecode_chunks
        address_offset_elements = (1 << 13) + 1
        max_descriptors = scatter["bytecode_max_descriptors_per_chunk"]
        max_admitted_descriptors = scatter[
            "bytecode_max_admitted_descriptors_per_chunk"
        ]
        max_pivots = scatter["bytecode_max_pivots_per_chunk"]
        max_admitted_pivots = scatter[
            "bytecode_max_admitted_pivots_per_chunk"
        ]
        real_descriptors = descriptor_elements - bytecode_chunks
        real_pivots = pivot_elements - 1
        dynamic_threadgroup_bytes = (
            328 + 8 * (max_descriptors + 1) + 2 * max(max_pivots, 1)
        )
        expected_fused = {
            "bytecode_fused": True,
            "bytecode_physical_rows": physical_rows,
            "bytecode_descriptor_bytes": 8 * descriptor_elements,
            "bytecode_pivot_bytes": 2 * pivot_elements,
            "bytecode_chunk_offset_elements": chunk_offset_elements,
            "bytecode_chunk_offset_bytes": 4 * chunk_offset_elements,
            "bytecode_work_item_bytes": 8 * work_items,
            "bytecode_address_offset_elements": address_offset_elements,
            "bytecode_address_offset_bytes": 4 * address_offset_elements,
            "bytecode_occurrence_bytes": 2 * physical_rows,
            "bytecode_magnitude_bytes": 8 * physical_rows,
            "shared_source_row_scans": 1,
            "additional_source_row_scans": 0,
            "member_upload_bytes": 0,
        }
        fused_ids = [
            scatter[field]
            for field in (
                "bytecode_descriptor_storage_id",
                "bytecode_pivot_storage_id",
                "bytecode_chunk_offset_storage_id",
                "bytecode_work_item_storage_id",
                "bytecode_address_offset_storage_id",
                "bytecode_occurrence_storage_id",
                "bytecode_magnitude_storage_id",
            )
        ]
        if (
            physical_rows <= 0
            or physical_rows > rows
            or real_descriptors < bytecode_chunks
            or real_descriptors
            > BYTECODE_ADDRESS_MAX_ADMITTED_DESCRIPTORS_PER_CHUNK
            * bytecode_chunks
            or real_pivots < 0
            or real_pivots
            > BYTECODE_ADDRESS_MAX_ADMITTED_PIVOTS_PER_CHUNK * bytecode_chunks
            or work_items < bytecode_chunks
            or work_items > physical_rows
            or not 1 <= max_descriptors <= max_admitted_descriptors
            or max_admitted_descriptors
            != BYTECODE_ADDRESS_MAX_ADMITTED_DESCRIPTORS_PER_CHUNK
            or max_descriptors > real_descriptors
            or max_admitted_pivots
            != BYTECODE_ADDRESS_MAX_ADMITTED_PIVOTS_PER_CHUNK
            or max_pivots > max_admitted_pivots
            or max_pivots > real_pivots
            or scatter["bytecode_dynamic_threadgroup_bytes"]
            != dynamic_threadgroup_bytes
            or scatter["bytecode_threadgroup_memory_limit_bytes"]
            < dynamic_threadgroup_bytes + scatter["static_threadgroup_bytes"]
            or any(scatter[field] != value for field, value in expected_fused.items())
            or any(identity <= 0 for identity in fused_ids)
            or len(set(fused_ids)) != len(fused_ids)
            or set(fused_ids) & set(source_ids + output_ids)
        ):
            raise ValueError(
                f"InstructionReadRaf fused Bytecode scatter ledger is invalid: {scatter}"
            )
        fused_bytecode = {field: scatter[field] for field in fused_scatter_fields}

    stage1_projection_observation = {
        "compact_rows": compact,
        "witness": stage1_projection,
    }
    return {
        "components": components,
        "outer_counts": outer_counts,
        "metal_counts": metal_counts,
        "source_observation": source,
        "scatter_observation": scatter,
        "scatter_wall_us": interval_duration_us(scatter_interval),
        "fused_bytecode_observation": fused_bytecode,
        "stage1_projection": stage1_projection_observation,
        "resource_observation": {
            "source": source,
            "scatter": scatter,
            "fused_bytecode": fused_bytecode,
            "stage1_projection": stage1_projection_observation,
        },
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
    stage1_source: Optional[dict[str, Any]] = None,
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
        stage5_provenance_fields = {
            "source_kind",
            "source_generation",
            "source_completion_serial",
            "source_claim_allocation_identity",
        }
        stage5_fields = lifecycle_fields | (
            stage5_provenance_fields if stage1_source is not None else set()
        )
        lifecycle_args = {
            "stage5": exact_span_args(
                events, METAL_BOOLEANITY_ROWS_STAGE5_PREPARE, stage5_fields
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
        if stage1_source is None:
            stage5_source_exact = (
                parsed_lifecycle["stage5"]["row_allocations"] == 1
                and parsed_lifecycle["stage5"]["row_upload_bytes"] == rows * 40
            )
            stage5_source = {
                "source_kind": "member_upload_v1",
                "source_generation": 0,
                "source_completion_serial": 0,
                "source_claim_allocation_identity": 0,
            }
        else:
            stage5_source = {
                "source_kind": trace_string(
                    lifecycle_args["stage5"]["source_kind"], "source_kind"
                ),
                **{
                    field: booleanity_trace_integer(
                        lifecycle_args["stage5"][field],
                        f"stage5 {field}",
                        allow_zero=True,
                    )
                    for field in stage5_provenance_fields - {"source_kind"}
                },
            }
            stage5_source_exact = (
                parsed_lifecycle["stage5"]["row_allocations"] == 0
                and parsed_lifecycle["stage5"]["row_upload_bytes"] == 0
                and stage5_source
                == {
                    "source_kind": "stage1_owner_v1",
                    "source_generation": stage1_source["source_generation"],
                    "source_completion_serial": stage1_source[
                        "completion_serial"
                    ],
                    "source_claim_allocation_identity": stage1_source[
                        "claim_allocation_identity"
                    ],
                }
                and storage_ids[0] == stage1_source["row_allocation_identity"]
                and registries[0] == stage1_source["device_registry_id"]
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
            or not stage5_source_exact
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
            **(stage5_source if stage1_source is not None else {}),
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
    stage1_source: Optional[dict[str, Any]] = None,
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
        stage1_source=stage1_source,
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
    events: list[dict[str, Any]],
    log_n: int,
    stage1_source: Optional[dict[str, Any]] = None,
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
    stage5_provenance_fields = {
        "source_kind",
        "source_generation",
        "source_completion_serial",
        "source_claim_allocation_identity",
    }
    stage5_fields = raw_fields | (
        stage5_provenance_fields if stage1_source is not None else set()
    )
    raw_args = {
        "stage5": exact_span_args(
            events, METAL_BOOLEANITY_ROWS_STAGE5_PREPARE, stage5_fields
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
            if field in raw_fields
        }
        for stage, args in raw_args.items()
    }
    rows = 1 << log_n
    source_ids = [raw[stage]["resident_rows_storage_id"] for stage in raw]
    registries = [raw[stage]["device_registry_id"] for stage in raw]
    if stage1_source is None:
        stage5_source_exact = (
            raw["stage5"]["row_allocations"] == 1
            and raw["stage5"]["row_upload_bytes"] == rows * 40
        )
        stage5_source = {
            "source_kind": "member_upload_v1",
            "source_generation": 0,
            "source_completion_serial": 0,
            "source_claim_allocation_identity": 0,
        }
    else:
        stage5_source = {
            "source_kind": trace_string(
                raw_args["stage5"]["source_kind"], "source_kind"
            ),
            **{
                field: booleanity_trace_integer(
                    raw_args["stage5"][field], f"stage5 {field}", allow_zero=True
                )
                for field in stage5_provenance_fields - {"source_kind"}
            },
        }
        stage5_source_exact = (
            raw["stage5"]["row_allocations"] == 0
            and raw["stage5"]["row_upload_bytes"] == 0
            and stage5_source
            == {
                "source_kind": "stage1_owner_v1",
                "source_generation": stage1_source["source_generation"],
                "source_completion_serial": stage1_source["completion_serial"],
                "source_claim_allocation_identity": stage1_source[
                    "claim_allocation_identity"
                ],
            }
            and source_ids[0] == stage1_source["row_allocation_identity"]
            and registries[0] == stage1_source["device_registry_id"]
        )
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
        or not stage5_source_exact
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
        **(stage5_source if stage1_source is not None else {}),
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
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    stage1_source: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if backend != "metal":
        raise ValueError("packed-hot Booleanity parsing requires the Metal arm")
    outer = booleanity_outer_member_breakdown(
        events,
        backend,
        BOOLEANITY_ADDRESS_KERNEL,
        OPTIMIZED_BOOLEANITY_ADDRESS_ROW_SOURCE,
    )
    lifecycle = retained_hamming_lifecycle(events, log_n, stage1_source)
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
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    stage1_source: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if backend != "metal":
        raise ValueError("retained-hot Hamming parsing requires the Metal arm")
    outer = booleanity_outer_member_breakdown(
        events,
        backend,
        HAMMING_WEIGHT_KERNEL,
        OPTIMIZED_HAMMING_WEIGHT_ROW_SOURCE,
    )
    lifecycle = retained_hamming_lifecycle(events, log_n, stage1_source)
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
    registers_claim_alias: bool = False,
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
            "registers_claim_alias_publish": int(registers_claim_alias),
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
        if registers_claim_alias:
            require_contained(
                inner["registers_claim_alias_publish"][0],
                rounds[(log_n + 1) // 2],
                "InstructionInput RegistersClaim alias publication",
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
    required_member_fields = {
        "cpu_instruction_read_raf_us",
        "metal_instruction_read_raf_us",
        "cpu_registers_claim_us",
        "metal_registers_claim_us",
        "cpu_hamming_weight_us",
        "metal_hamming_weight_us",
        "cpu_hamming_weight_service_us",
        "metal_hamming_weight_service_us",
    }
    for index, pair in enumerate(pairs, 1):
        missing = required_member_fields - pair.keys()
        if missing:
            raise ValueError(
                f"pair {index} is missing required member timing fields: {sorted(missing)}"
            )
    bytecode_address_fields = {
        "cpu_bytecode_address_us",
        "metal_bytecode_address_us",
    }
    bytecode_address_presence = []
    for index, pair in enumerate(pairs, 1):
        observed = bytecode_address_fields & pair.keys()
        if observed and observed != bytecode_address_fields:
            raise ValueError(
                f"pair {index} has an incomplete Bytecode address timing record"
            )
        bytecode_address_presence.append(observed == bytecode_address_fields)
    if any(bytecode_address_presence) and not all(bytecode_address_presence):
        raise ValueError("Bytecode address timing records must cover every pair")
    has_bytecode_address = all(bytecode_address_presence)
    producer_control_fields = {
        "metal_bytecode_address_control_prepare_us",
        "producer_order",
    }
    producer_control_presence = []
    for index, pair in enumerate(pairs, 1):
        observed = producer_control_fields & pair.keys()
        if observed and observed != producer_control_fields:
            raise ValueError(
                f"pair {index} has an incomplete Bytecode address producer control"
            )
        producer_control_presence.append(observed == producer_control_fields)
    if any(producer_control_presence) and not all(producer_control_presence):
        raise ValueError("Bytecode address producer controls must cover every pair")
    has_bytecode_address_producer_control = all(producer_control_presence)
    fused_producer_fields = {
        "metal_bytecode_address_stage1_topology_us",
        "metal_bytecode_address_control_stage1_topology_us",
        "metal_bytecode_address_irraf_scatter_us",
        "metal_bytecode_address_control_irraf_scatter_us",
    }
    fused_producer_presence = []
    for index, pair in enumerate(pairs, 1):
        observed = fused_producer_fields & pair.keys()
        if observed and observed != fused_producer_fields:
            raise ValueError(
                f"pair {index} has an incomplete fused Bytecode address producer record"
            )
        fused_producer_presence.append(observed == fused_producer_fields)
    if any(fused_producer_presence) and not all(fused_producer_presence):
        raise ValueError("fused Bytecode address producer records must cover every pair")
    has_fused_bytecode_address_producer = all(fused_producer_presence)
    if (
        has_fused_bytecode_address_producer
        and not has_bytecode_address_producer_control
    ):
        raise ValueError("fused Bytecode address producer requires paired controls")
    cpu = [float(pair["cpu_us"]) for pair in pairs]
    metal = [float(pair["metal_us"]) for pair in pairs]
    cpu_prepare = [float(pair["cpu_prepare_us"]) for pair in pairs]
    metal_prepare = [float(pair["metal_prepare_us"]) for pair in pairs]
    cpu_instruction_ra = [float(pair["cpu_instruction_ra_us"]) for pair in pairs]
    metal_instruction_ra = [float(pair["metal_instruction_ra_us"]) for pair in pairs]
    cpu_bytecode = [float(pair["cpu_bytecode_us"]) for pair in pairs]
    metal_bytecode = [float(pair["metal_bytecode_us"]) for pair in pairs]
    cpu_bytecode_address = (
        [float(pair["cpu_bytecode_address_us"]) for pair in pairs]
        if has_bytecode_address
        else []
    )
    metal_bytecode_address = (
        [float(pair["metal_bytecode_address_us"]) for pair in pairs]
        if has_bytecode_address
        else []
    )
    bytecode_address_prepare_deltas = (
        [metal_us - cpu_us for cpu_us, metal_us in zip(cpu_prepare, metal_prepare)]
        if has_bytecode_address
        else []
    )
    bytecode_address_control_prepare = (
        [
            float(pair["metal_bytecode_address_control_prepare_us"])
            for pair in pairs
        ]
        if has_bytecode_address_producer_control
        else []
    )
    bytecode_address_target_control_deltas = (
        [
            target_us - control_us
            for target_us, control_us in zip(
                metal_prepare, bytecode_address_control_prepare
            )
        ]
        if has_bytecode_address_producer_control
        else bytecode_address_prepare_deltas
    )
    bytecode_address_stage1_topology = (
        [
            float(pair["metal_bytecode_address_stage1_topology_us"])
            for pair in pairs
        ]
        if has_fused_bytecode_address_producer
        else []
    )
    bytecode_address_control_stage1_topology = (
        [
            float(pair["metal_bytecode_address_control_stage1_topology_us"])
            for pair in pairs
        ]
        if has_fused_bytecode_address_producer
        else []
    )
    bytecode_address_irraf_scatter = (
        [
            float(pair["metal_bytecode_address_irraf_scatter_us"])
            for pair in pairs
        ]
        if has_fused_bytecode_address_producer
        else []
    )
    bytecode_address_control_irraf_scatter = (
        [
            float(pair["metal_bytecode_address_control_irraf_scatter_us"])
            for pair in pairs
        ]
        if has_fused_bytecode_address_producer
        else []
    )
    bytecode_address_stage1_topology_deltas = [
        target_us - control_us
        for target_us, control_us in zip(
            bytecode_address_stage1_topology,
            bytecode_address_control_stage1_topology,
        )
    ]
    bytecode_address_irraf_scatter_deltas = [
        target_us - control_us
        for target_us, control_us in zip(
            bytecode_address_irraf_scatter,
            bytecode_address_control_irraf_scatter,
        )
    ]
    bytecode_address_signed_producer_deltas = (
        [
            stage1_delta_us + scatter_delta_us
            for stage1_delta_us, scatter_delta_us in zip(
                bytecode_address_stage1_topology_deltas,
                bytecode_address_irraf_scatter_deltas,
            )
        ]
        if has_fused_bytecode_address_producer
        else bytecode_address_target_control_deltas
    )
    bytecode_address_charged_producer_deltas = [
        max(0.0, delta) for delta in bytecode_address_signed_producer_deltas
    ]
    charged_metal_address = [
        member_us + producer_delta_us
        for member_us, producer_delta_us in zip(
            metal_bytecode_address, bytecode_address_charged_producer_deltas
        )
    ]
    cpu_instruction_input = [float(pair["cpu_instruction_input_us"]) for pair in pairs]
    metal_instruction_input = [
        float(pair["metal_instruction_input_us"]) for pair in pairs
    ]
    cpu_registers_claim = [
        float(pair["cpu_registers_claim_us"]) for pair in pairs
    ]
    metal_registers_claim = [
        float(pair["metal_registers_claim_us"]) for pair in pairs
    ]
    cpu_instruction_read_raf = [
        float(pair["cpu_instruction_read_raf_us"])
        for pair in pairs
    ]
    metal_instruction_read_raf = [
        float(pair["metal_instruction_read_raf_us"])
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
    ram_member_fields = {
        f"{backend}_{pair_prefix}_us"
        for _, pair_prefix in RAM_CYCLE_FAMILY_PAIR_MEMBERS
        for backend in ("cpu", "metal")
    }
    ram_fields = ram_member_fields | {
        "metal_ram_read_write_charged_us",
        "metal_ram_hamming_charged_us",
        "metal_ram_cycle_family_owner_us",
        "metal_ram_cycle_family_witness_prepare_us",
    }
    ram_presence = []
    for index, pair in enumerate(pairs, 1):
        observed_ram_fields = {
            key
            for key in pair
            if key.startswith("cpu_ram_") or key.startswith("metal_ram_")
        }
        if observed_ram_fields and observed_ram_fields != ram_fields:
            raise ValueError(
                f"pair {index} must contain the exact full RAM cycle-family key set"
            )
        ram_presence.append(observed_ram_fields == ram_fields)
    if any(ram_presence) and not all(ram_presence):
        raise ValueError("RAM cycle-family timing records must cover every pair")
    has_ram_cycle_family = all(ram_presence)
    if has_ram_cycle_family:
        ram_member_samples = {
            member_name: {
                backend: [
                    float(pair[f"{backend}_{pair_prefix}_us"])
                    for pair in pairs
                ]
                for backend in ("cpu", "metal")
            }
            for member_name, pair_prefix in RAM_CYCLE_FAMILY_PAIR_MEMBERS
        }
        cpu_ram_read_write = ram_member_samples["read_write"]["cpu"]
        metal_ram_read_write = ram_member_samples["read_write"]["metal"]
        metal_ram_read_write_charged = [
            float(pair["metal_ram_read_write_charged_us"]) for pair in pairs
        ]
        cpu_ram_hamming = ram_member_samples["hamming_booleanity"]["cpu"]
        metal_ram_hamming = ram_member_samples["hamming_booleanity"]["metal"]
        metal_ram_hamming_charged = [
            float(pair["metal_ram_hamming_charged_us"]) for pair in pairs
        ]
        metal_ram_cycle_family_owner = [
            float(pair["metal_ram_cycle_family_owner_us"]) for pair in pairs
        ]
        metal_ram_cycle_family_witness_prepare = [
            float(pair["metal_ram_cycle_family_witness_prepare_us"])
            for pair in pairs
        ]
        for (
            raw_read_write,
            charged_read_write,
            raw_hamming,
            charged_hamming,
            owner,
            witness_prepare,
        ) in zip(
            metal_ram_read_write,
            metal_ram_read_write_charged,
            metal_ram_hamming,
            metal_ram_hamming_charged,
            metal_ram_cycle_family_owner,
            metal_ram_cycle_family_witness_prepare,
        ):
            if owner <= 0.0 or witness_prepare <= 0.0 or owner > witness_prepare:
                raise ValueError("RAM cycle-family producer durations are inconsistent")
            if not math.isclose(
                charged_read_write,
                raw_read_write + owner,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError("RAM read-write charged timing must include the owner once")
            if not math.isclose(
                charged_hamming,
                raw_hamming + owner,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError("RAM Hamming charged timing must include the owner once")
        cpu_ram_cycle_family = [
            sum(ram_member_samples[name]["cpu"][index] for name, _ in RAM_CYCLE_FAMILY_PAIR_MEMBERS)
            for index in range(len(pairs))
        ]
        metal_ram_cycle_family_raw = [
            sum(ram_member_samples[name]["metal"][index] for name, _ in RAM_CYCLE_FAMILY_PAIR_MEMBERS)
            for index in range(len(pairs))
        ]
        metal_ram_cycle_family = [
            raw + witness_prepare
            for raw, witness_prepare in zip(
                metal_ram_cycle_family_raw,
                metal_ram_cycle_family_witness_prepare,
            )
        ]
    else:
        ram_member_samples = {
            member_name: {"cpu": [], "metal": []}
            for member_name, _ in RAM_CYCLE_FAMILY_PAIR_MEMBERS
        }
        cpu_ram_read_write = []
        metal_ram_read_write = []
        metal_ram_read_write_charged = []
        cpu_ram_hamming = []
        metal_ram_hamming = []
        metal_ram_hamming_charged = []
        metal_ram_cycle_family_owner = []
        metal_ram_cycle_family_witness_prepare = []
        cpu_ram_cycle_family = []
        metal_ram_cycle_family_raw = []
        metal_ram_cycle_family = []
    ram_member_duration_samples = [
        value
        for member_name, _ in RAM_CYCLE_FAMILY_PAIR_MEMBERS
        for backend in ("cpu", "metal")
        for value in ram_member_samples[member_name][backend]
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
    cpu_outer_product_instruction_input_registers_family = [
        family + instruction_input + registers
        for family, instruction_input, registers in zip(
            cpu_outer_product_family,
            cpu_instruction_input,
            cpu_registers_claim,
        )
    ]
    metal_outer_product_instruction_input_registers_family = [
        family + instruction_input + registers
        for family, instruction_input, registers in zip(
            metal_outer_product_family,
            metal_instruction_input,
            metal_registers_claim,
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
        not math.isfinite(value) or value < 0.0
        for value in bytecode_address_control_prepare
        + bytecode_address_stage1_topology
        + bytecode_address_control_stage1_topology
        + bytecode_address_irraf_scatter
        + bytecode_address_control_irraf_scatter
    ):
        raise ValueError("Bytecode address control preparation durations are invalid")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in cpu_instruction_ra
        + metal_instruction_ra
        + cpu_bytecode
        + metal_bytecode
        + cpu_bytecode_address
        + metal_bytecode_address
        + charged_metal_address
        + cpu_instruction_input
        + metal_instruction_input
        + cpu_registers_claim
        + metal_registers_claim
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
        + ram_member_duration_samples
        + metal_ram_read_write_charged
        + metal_ram_hamming_charged
        + metal_ram_cycle_family_owner
        + metal_ram_cycle_family_witness_prepare
        + cpu_ram_cycle_family
        + metal_ram_cycle_family_raw
        + metal_ram_cycle_family
        + cpu_outer_remainder
        + metal_outer_remainder
        + cpu_product_uniskip
        + metal_product_uniskip
        + cpu_product_remainder
        + metal_product_remainder
        + cpu_outer_product_family
        + metal_outer_product_family
        + cpu_outer_product_instruction_input_registers_family
        + metal_outer_product_instruction_input_registers_family
        + cpu_instruction_claim
        + metal_instruction_claim
        + metal_instruction_claim_isolated_service
        + cpu_product_instruction_claim
        + metal_product_instruction_claim
    ):
        raise ValueError("kernel durations must be finite and positive")
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
    if has_bytecode_address:
        (
            bytecode_address_speedups,
            bytecode_address_improvements,
            bytecode_address_decision,
        ) = local_member_decision(
            pairs,
            cpu_bytecode_address,
            metal_bytecode_address,
            BYTECODE_ADDRESS_MIN_SPEEDUP,
        )
        (
            bytecode_address_charged_speedups,
            bytecode_address_charged_improvements,
            bytecode_address_charged_decision,
        ) = local_member_decision(
            pairs,
            cpu_bytecode_address,
            charged_metal_address,
            BYTECODE_ADDRESS_MIN_SPEEDUP,
        )
        target_first_speedups = [
            speedup
            for pair, speedup in zip(pairs, bytecode_address_charged_speedups)
            if pair.get("producer_order") == ["target", "control"]
        ]
        control_first_speedups = [
            speedup
            for pair, speedup in zip(pairs, bytecode_address_charged_speedups)
            if pair.get("producer_order") == ["control", "target"]
        ]
        target_first_median = (
            statistics.median(target_first_speedups) if target_first_speedups else None
        )
        control_first_median = (
            statistics.median(control_first_speedups) if control_first_speedups else None
        )
        signed_producer_delta_median = statistics.median(
            bytecode_address_signed_producer_deltas
        )
        signed_producer_delta_mad = statistics.median(
            abs(value - signed_producer_delta_median)
            for value in bytecode_address_signed_producer_deltas
        )
        target_first_producer_deltas = [
            delta
            for pair, delta in zip(pairs, bytecode_address_signed_producer_deltas)
            if pair.get("producer_order") == ["target", "control"]
        ]
        control_first_producer_deltas = [
            delta
            for pair, delta in zip(pairs, bytecode_address_signed_producer_deltas)
            if pair.get("producer_order") == ["control", "target"]
        ]
        clears_producer_order_strata = (
            has_bytecode_address_producer_control
            and target_first_median is not None
            and control_first_median is not None
            and target_first_median >= BYTECODE_ADDRESS_MIN_SPEEDUP
            and control_first_median >= BYTECODE_ADDRESS_MIN_SPEEDUP
        )
        bytecode_address_charged_decision.update(
            {
                "producer_control_present": has_bytecode_address_producer_control,
                "charge_model": (
                    "stage1_topology_plus_irraf_scatter_v1"
                    if has_fused_bytecode_address_producer
                    else "backend_witness_prepare_v1"
                ),
                "target_first_median_speedup": target_first_median,
                "control_first_median_speedup": control_first_median,
                "signed_producer_delta_ms_median": signed_producer_delta_median
                / 1000.0,
                "signed_producer_delta_ms_mad": signed_producer_delta_mad / 1000.0,
                "target_first_producer_sample_count": len(
                    target_first_producer_deltas
                ),
                "control_first_producer_sample_count": len(
                    control_first_producer_deltas
                ),
                "target_first_signed_producer_delta_ms_median": (
                    statistics.median(target_first_producer_deltas) / 1000.0
                    if target_first_producer_deltas
                    else None
                ),
                "control_first_signed_producer_delta_ms_median": (
                    statistics.median(control_first_producer_deltas) / 1000.0
                    if control_first_producer_deltas
                    else None
                ),
                "clears_producer_order_strata": clears_producer_order_strata,
                "clears": bytecode_address_charged_decision["clears"]
                and clears_producer_order_strata,
            }
        )
    instruction_input_speedups, instruction_input_improvements, instruction_input_decision = (
        local_member_decision(
            pairs,
            cpu_instruction_input,
            metal_instruction_input,
            INSTRUCTION_INPUT_MIN_SPEEDUP,
        )
    )
    (
        registers_claim_speedups,
        registers_claim_improvements,
        registers_claim_decision,
    ) = local_member_decision(
        pairs,
        cpu_registers_claim,
        metal_registers_claim,
        REGISTERS_CLAIM_MIN_SPEEDUP,
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
    if has_ram_cycle_family:
        ram_member_decisions = {
            member_name: local_member_decision(
                pairs,
                samples["cpu"],
                samples["metal"],
                RAM_CYCLE_FAMILY_MIN_SPEEDUP,
            )
            for member_name, samples in ram_member_samples.items()
        }
        (
            ram_read_write_speedups,
            ram_read_write_improvements,
            ram_read_write_decision,
        ) = ram_member_decisions["read_write"]
        (
            ram_read_write_charged_speedups,
            ram_read_write_charged_improvements,
            ram_read_write_charged_decision,
        ) = local_member_decision(
            pairs,
            cpu_ram_read_write,
            metal_ram_read_write_charged,
            RAM_READ_WRITE_MIN_SPEEDUP,
        )
        (
            ram_hamming_speedups,
            ram_hamming_improvements,
            ram_hamming_decision,
        ) = ram_member_decisions["hamming_booleanity"]
        (
            ram_hamming_charged_speedups,
            ram_hamming_charged_improvements,
            ram_hamming_charged_decision,
        ) = local_member_decision(
            pairs,
            cpu_ram_hamming,
            metal_ram_hamming_charged,
            RAM_HAMMING_MIN_SPEEDUP,
        )
        (
            ram_cycle_family_speedups,
            ram_cycle_family_improvements,
            ram_cycle_family_decision,
        ) = local_member_decision(
            pairs,
            cpu_ram_cycle_family,
            metal_ram_cycle_family,
            RAM_CYCLE_FAMILY_MIN_SPEEDUP,
        )
    else:
        ram_member_decisions = {
            member_name: ([], [], None)
            for member_name, _ in RAM_CYCLE_FAMILY_PAIR_MEMBERS
        }
        ram_read_write_speedups = []
        ram_read_write_improvements = []
        ram_read_write_decision = None
        ram_read_write_charged_speedups = []
        ram_read_write_charged_improvements = []
        ram_read_write_charged_decision = None
        ram_hamming_speedups = []
        ram_hamming_improvements = []
        ram_hamming_decision = None
        ram_hamming_charged_speedups = []
        ram_hamming_charged_improvements = []
        ram_hamming_charged_decision = None
        ram_cycle_family_speedups = []
        ram_cycle_family_improvements = []
        ram_cycle_family_decision = None
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
        outer_product_instruction_input_registers_family_speedups,
        outer_product_instruction_input_registers_family_improvements,
        outer_product_instruction_input_registers_family_decision,
    ) = local_member_decision(
        pairs,
        cpu_outer_product_instruction_input_registers_family,
        metal_outer_product_instruction_input_registers_family,
        REGISTERS_CLAIM_MIN_SPEEDUP,
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
    piop_speedups, piop_improvements, piop_decision = local_member_decision(
        pairs,
        cpu,
        metal,
        PIOP_MIN_SPEEDUP,
    )
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
    metrics = {
        "piop_speedup": statistics.median(piop_speedups),
        "instruction_ra_speedup": statistics.median(instruction_ra_speedups),
        "bytecode_read_raf_cycle_speedup": bytecode_speedup_median,
        "instruction_input_kernel_service_speedup": statistics.median(
            instruction_input_speedups
        ),
        "registers_claim_reduction_member_speedup": statistics.median(
            registers_claim_speedups
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
        "ram_read_write_speedup": (
            statistics.median(ram_read_write_speedups)
            if has_ram_cycle_family
            else None
        ),
        "ram_read_write_charged_speedup": (
            statistics.median(ram_read_write_charged_speedups)
            if has_ram_cycle_family
            else None
        ),
        "ram_read_write_standalone_charged_speedup": (
            statistics.median(ram_read_write_charged_speedups)
            if has_ram_cycle_family
            else None
        ),
        "ram_hamming_booleanity_speedup": (
            statistics.median(ram_hamming_speedups)
            if has_ram_cycle_family
            else None
        ),
        "ram_hamming_booleanity_charged_speedup": (
            statistics.median(ram_hamming_charged_speedups)
            if has_ram_cycle_family
            else None
        ),
        "ram_hamming_booleanity_standalone_charged_speedup": (
            statistics.median(ram_hamming_charged_speedups)
            if has_ram_cycle_family
            else None
        ),
        "ram_raf_evaluation_speedup": (
            statistics.median(ram_member_decisions["raf_evaluation"][0])
            if has_ram_cycle_family
            else None
        ),
        "ram_val_check_speedup": (
            statistics.median(ram_member_decisions["val_check"][0])
            if has_ram_cycle_family
            else None
        ),
        "ram_ra_claim_reduction_speedup": (
            statistics.median(ram_member_decisions["ra_claim_reduction"][0])
            if has_ram_cycle_family
            else None
        ),
        "ram_ra_virtualization_speedup": (
            statistics.median(ram_member_decisions["ra_virtualization"][0])
            if has_ram_cycle_family
            else None
        ),
        "ram_cycle_family_speedup": (
            statistics.median(ram_cycle_family_speedups)
            if has_ram_cycle_family
            else None
        ),
        "ram_standalone_charged_metrics_additive": False,
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
        "outer_product_instruction_input_registers_family_speedup": statistics.median(
            outer_product_instruction_input_registers_family_speedups
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
        "paired_speedups": piop_speedups,
        "paired_instruction_ra_speedups": instruction_ra_speedups,
        "paired_bytecode_read_raf_cycle_speedups": bytecode_speedups,
        "paired_bytecode_read_raf_cycle_fractional_improvements": bytecode_improvements,
        "paired_instruction_input_kernel_service_speedups": instruction_input_speedups,
        "paired_instruction_input_kernel_service_fractional_improvements": instruction_input_improvements,
        "paired_registers_claim_reduction_member_speedups": registers_claim_speedups,
        "paired_registers_claim_reduction_member_fractional_improvements": (
            registers_claim_improvements
        ),
        "paired_instruction_read_raf_speedups": instruction_read_raf_speedups,
        "paired_instruction_read_raf_fractional_improvements": instruction_read_raf_improvements,
        "paired_booleanity_address_phase_speedups": booleanity_address_speedups,
        "paired_booleanity_address_phase_fractional_improvements": booleanity_address_improvements,
        "paired_booleanity_address_phase_service_speedups": booleanity_address_service_speedups,
        "paired_hamming_weight_claim_reduction_speedups": hamming_weight_speedups,
        "paired_hamming_weight_claim_reduction_fractional_improvements": hamming_weight_improvements,
        "paired_hamming_weight_claim_reduction_service_speedups": hamming_weight_service_speedups,
        "paired_ram_read_write_speedups": ram_read_write_speedups,
        "paired_ram_read_write_fractional_improvements": ram_read_write_improvements,
        "paired_ram_read_write_charged_speedups": ram_read_write_charged_speedups,
        "paired_ram_read_write_standalone_charged_speedups": (
            ram_read_write_charged_speedups
        ),
        "paired_ram_read_write_charged_fractional_improvements": (
            ram_read_write_charged_improvements
        ),
        "paired_ram_hamming_booleanity_speedups": ram_hamming_speedups,
        "paired_ram_hamming_booleanity_fractional_improvements": ram_hamming_improvements,
        "paired_ram_hamming_booleanity_charged_speedups": ram_hamming_charged_speedups,
        "paired_ram_hamming_booleanity_standalone_charged_speedups": (
            ram_hamming_charged_speedups
        ),
        "paired_ram_hamming_booleanity_charged_fractional_improvements": (
            ram_hamming_charged_improvements
        ),
        "paired_ram_raf_evaluation_speedups": ram_member_decisions[
            "raf_evaluation"
        ][0],
        "paired_ram_raf_evaluation_fractional_improvements": ram_member_decisions[
            "raf_evaluation"
        ][1],
        "paired_ram_val_check_speedups": ram_member_decisions["val_check"][0],
        "paired_ram_val_check_fractional_improvements": ram_member_decisions[
            "val_check"
        ][1],
        "paired_ram_ra_claim_reduction_speedups": ram_member_decisions[
            "ra_claim_reduction"
        ][0],
        "paired_ram_ra_claim_reduction_fractional_improvements": ram_member_decisions[
            "ra_claim_reduction"
        ][1],
        "paired_ram_ra_virtualization_speedups": ram_member_decisions[
            "ra_virtualization"
        ][0],
        "paired_ram_ra_virtualization_fractional_improvements": ram_member_decisions[
            "ra_virtualization"
        ][1],
        "paired_ram_cycle_family_speedups": ram_cycle_family_speedups,
        "paired_ram_cycle_family_fractional_improvements": (
            ram_cycle_family_improvements
        ),
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
        "paired_outer_product_instruction_input_registers_family_speedups": (
            outer_product_instruction_input_registers_family_speedups
        ),
        "paired_outer_product_instruction_input_registers_family_fractional_improvements": (
            outer_product_instruction_input_registers_family_improvements
        ),
        "paired_piop_fractional_improvements": piop_improvements,
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
        "cpu_registers_claim_reduction_member_ms_samples": [
            value / 1000.0 for value in cpu_registers_claim
        ],
        "metal_registers_claim_reduction_member_ms_samples": [
            value / 1000.0 for value in metal_registers_claim
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
        "cpu_ram_read_write_ms_samples": [
            value / 1000.0 for value in cpu_ram_read_write
        ],
        "metal_ram_read_write_ms_samples": [
            value / 1000.0 for value in metal_ram_read_write
        ],
        "metal_ram_read_write_charged_ms_samples": [
            value / 1000.0 for value in metal_ram_read_write_charged
        ],
        "metal_ram_read_write_standalone_charged_ms_samples": [
            value / 1000.0 for value in metal_ram_read_write_charged
        ],
        "metal_ram_cycle_family_owner_ms_samples": [
            value / 1000.0 for value in metal_ram_cycle_family_owner
        ],
        "metal_ram_cycle_family_witness_prepare_ms_samples": [
            value / 1000.0 for value in metal_ram_cycle_family_witness_prepare
        ],
        "cpu_ram_raf_evaluation_ms_samples": [
            value / 1000.0
            for value in ram_member_samples["raf_evaluation"]["cpu"]
        ],
        "metal_ram_raf_evaluation_ms_samples": [
            value / 1000.0
            for value in ram_member_samples["raf_evaluation"]["metal"]
        ],
        "cpu_ram_val_check_ms_samples": [
            value / 1000.0 for value in ram_member_samples["val_check"]["cpu"]
        ],
        "metal_ram_val_check_ms_samples": [
            value / 1000.0 for value in ram_member_samples["val_check"]["metal"]
        ],
        "cpu_ram_ra_claim_reduction_ms_samples": [
            value / 1000.0
            for value in ram_member_samples["ra_claim_reduction"]["cpu"]
        ],
        "metal_ram_ra_claim_reduction_ms_samples": [
            value / 1000.0
            for value in ram_member_samples["ra_claim_reduction"]["metal"]
        ],
        "cpu_ram_hamming_booleanity_ms_samples": [
            value / 1000.0 for value in cpu_ram_hamming
        ],
        "metal_ram_hamming_booleanity_ms_samples": [
            value / 1000.0 for value in metal_ram_hamming
        ],
        "metal_ram_hamming_booleanity_charged_ms_samples": [
            value / 1000.0 for value in metal_ram_hamming_charged
        ],
        "metal_ram_hamming_booleanity_standalone_charged_ms_samples": [
            value / 1000.0 for value in metal_ram_hamming_charged
        ],
        "cpu_ram_ra_virtualization_ms_samples": [
            value / 1000.0
            for value in ram_member_samples["ra_virtualization"]["cpu"]
        ],
        "metal_ram_ra_virtualization_ms_samples": [
            value / 1000.0
            for value in ram_member_samples["ra_virtualization"]["metal"]
        ],
        "cpu_ram_cycle_family_ms_samples": [
            value / 1000.0 for value in cpu_ram_cycle_family
        ],
        "metal_ram_cycle_family_raw_ms_samples": [
            value / 1000.0 for value in metal_ram_cycle_family_raw
        ],
        "metal_ram_cycle_family_ms_samples": [
            value / 1000.0 for value in metal_ram_cycle_family
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
        "cpu_outer_product_instruction_input_registers_family_ms_samples": [
            value / 1000.0
            for value in cpu_outer_product_instruction_input_registers_family
        ],
        "metal_outer_product_instruction_input_registers_family_ms_samples": [
            value / 1000.0
            for value in metal_outer_product_instruction_input_registers_family
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
        "registers_claim_reduction_member_decision": registers_claim_decision,
        "instruction_read_raf_decision": instruction_read_raf_decision,
        "booleanity_address_phase_decision": booleanity_address_decision,
        "hamming_weight_claim_reduction_decision": hamming_weight_decision,
        "ram_read_write_decision": ram_read_write_decision,
        "ram_read_write_charged_decision": ram_read_write_charged_decision,
        "ram_read_write_standalone_charged_decision": (
            ram_read_write_charged_decision
        ),
        "ram_hamming_booleanity_decision": ram_hamming_decision,
        "ram_hamming_booleanity_charged_decision": ram_hamming_charged_decision,
        "ram_hamming_booleanity_standalone_charged_decision": (
            ram_hamming_charged_decision
        ),
        "ram_raf_evaluation_decision": ram_member_decisions["raf_evaluation"][2],
        "ram_val_check_decision": ram_member_decisions["val_check"][2],
        "ram_ra_claim_reduction_decision": ram_member_decisions[
            "ra_claim_reduction"
        ][2],
        "ram_ra_virtualization_decision": ram_member_decisions[
            "ra_virtualization"
        ][2],
        "ram_cycle_family_decision": ram_cycle_family_decision,
        "booleanity_hamming_family_decision": booleanity_hamming_decision,
        "outer_remainder_decision": outer_remainder_decision,
        "product_remainder_decision": product_remainder_decision,
        "outer_product_family_decision": outer_product_family_decision,
        "outer_product_instruction_input_registers_family_decision": (
            outer_product_instruction_input_registers_family_decision
        ),
        "piop_decision": piop_decision,
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
    if has_bytecode_address:
        metrics.update(
            {
                "bytecode_read_raf_address_speedup": statistics.median(
                    bytecode_address_speedups
                ),
                "paired_bytecode_read_raf_address_speedups": (
                    bytecode_address_speedups
                ),
                "paired_bytecode_read_raf_address_fractional_improvements": (
                    bytecode_address_improvements
                ),
                "cpu_bytecode_read_raf_address_ms_samples": [
                    value / 1000.0 for value in cpu_bytecode_address
                ],
                "metal_bytecode_read_raf_address_ms_samples": [
                    value / 1000.0 for value in metal_bytecode_address
                ],
                "bytecode_read_raf_address_decision": bytecode_address_decision,
                "bytecode_read_raf_address_charged_speedup": statistics.median(
                    bytecode_address_charged_speedups
                ),
                "paired_bytecode_read_raf_address_charged_speedups": (
                    bytecode_address_charged_speedups
                ),
                "paired_bytecode_read_raf_address_charged_fractional_improvements": (
                    bytecode_address_charged_improvements
                ),
                "bytecode_read_raf_address_backend_witness_prepare_delta_ms_samples": [
                    value / 1000.0 for value in bytecode_address_prepare_deltas
                ],
                "metal_bytecode_read_raf_address_control_prepare_ms_samples": [
                    value / 1000.0 for value in bytecode_address_control_prepare
                ],
                "bytecode_read_raf_address_target_control_prepare_delta_ms_samples": [
                    value / 1000.0
                    for value in bytecode_address_target_control_deltas
                ],
                "bytecode_read_raf_address_charged_producer_delta_ms_samples": [
                    value / 1000.0
                    for value in bytecode_address_charged_producer_deltas
                ],
                "charged_metal_address_ms_samples": [
                    value / 1000.0 for value in charged_metal_address
                ],
                "bytecode_read_raf_address_charged_decision": (
                    bytecode_address_charged_decision
                ),
            }
        )
        if has_fused_bytecode_address_producer:
            metrics.update(
                {
                    "metal_bytecode_read_raf_address_stage1_topology_ms_samples": [
                        value / 1000.0 for value in bytecode_address_stage1_topology
                    ],
                    "metal_bytecode_read_raf_address_control_stage1_topology_ms_samples": [
                        value / 1000.0
                        for value in bytecode_address_control_stage1_topology
                    ],
                    "metal_bytecode_read_raf_address_irraf_scatter_ms_samples": [
                        value / 1000.0 for value in bytecode_address_irraf_scatter
                    ],
                    "metal_bytecode_read_raf_address_control_irraf_scatter_ms_samples": [
                        value / 1000.0
                        for value in bytecode_address_control_irraf_scatter
                    ],
                    "bytecode_read_raf_address_stage1_topology_delta_ms_samples": [
                        value / 1000.0
                        for value in bytecode_address_stage1_topology_deltas
                    ],
                    "bytecode_read_raf_address_irraf_scatter_delta_ms_samples": [
                        value / 1000.0
                        for value in bytecode_address_irraf_scatter_deltas
                    ],
                    "bytecode_read_raf_address_signed_fused_producer_delta_ms_samples": [
                        value / 1000.0
                        for value in bytecode_address_signed_producer_deltas
                    ],
                }
            )
    return metrics


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
    if "topology_observation" in member:
        record["topology_observation"] = member["topology_observation"]
    if "scatter_wall_us" in member:
        scatter_wall_us = member["scatter_wall_us"]
        record["scatter_wall_ns"] = (
            round(float(scatter_wall_us) * 1000.0)
            if scatter_wall_us is not None
            else None
        )
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
    elif kernel == BYTECODE_ADDRESS_KERNEL:
        value = result["bytecode_address_member"]["components"]["member_us"]
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
    elif kernel == REGISTERS_CLAIM_KERNEL:
        value = result["registers_claim_member"]["components"]["member_us"]
    elif kernel == INSTRUCTION_READ_RAF_KERNEL:
        value = result["instruction_read_raf_member"]["components"]["member_us"]
    elif kernel == RAM_CYCLE_FAMILY_KERNEL:
        value = result["ram_cycle_family"]["components"]["charged_member_us"]
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
    instruction_read_raf_scatter_threads: int,
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
    registers_claim_implementation: str,
    registers_claim_trace_cutoff_log2: int,
    pair_index: int,
    timeout_seconds: int,
    artifact_label: Optional[str] = None,
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
        "--instruction-read-raf-metal-scatter-threads",
        str(instruction_read_raf_scatter_threads),
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
        "--registers-claim-metal-implementation",
        registers_claim_implementation,
        "--registers-claim-metal-trace-cutoff-log2",
        str(registers_claim_trace_cutoff_log2),
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
    arm_label = artifact_label or backend
    if re.fullmatch(r"[a-z0-9-]+", arm_label) is None:
        raise ValueError(f"invalid evaluator arm label {arm_label!r}")
    label = f"pair-{pair_index:02d}-{arm_label}"
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
    bytecode_address_config = validate_bytecode_address_stdout(
        result.stdout,
        backend,
        bytecode_address_implementation,
        bytecode_address_outer_tiles,
        bytecode_address_trace_cutoff_log2,
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
    instruction_read_raf_config = validate_instruction_read_raf_stdout(
        result.stdout,
        backend,
        instruction_read_raf_scatter_threads,
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
        registers_claim_implementation == "outer-carrier-alias-hybrid",
    )
    registers_claim_config = validate_registers_claim_stdout(
        result.stdout,
        backend,
        registers_claim_implementation,
        registers_claim_trace_cutoff_log2,
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
        registers_claim_implementation == "outer-carrier-alias-hybrid",
    )
    instruction_read_raf_member = instruction_read_raf_member_breakdown(
        events,
        backend,
        log_n,
        scatter_threads=instruction_read_raf_scatter_threads,
        expect_fused_bytecode_address=(
            backend == "metal"
            and bytecode_address_implementation == "address-major"
            and log_n >= bytecode_address_trace_cutoff_log2
        ),
    )
    stage1_source = instruction_read_raf_member["source_observation"]
    if stage1_source is not None:
        stage1_source = {
            **stage1_source,
            "explicit_rows": instruction_read_raf_member["stage1_projection"][
                "compact_rows"
            ]["explicit_rows"],
        }
    bytecode_address_member = bytecode_address_member_breakdown(
        events,
        backend,
        log_n,
        bytecode_address_implementation,
        bytecode_address_outer_tiles,
        bytecode_address_trace_cutoff_log2,
        stage1_source,
        instruction_read_raf_member["scatter_observation"],
    )
    if backend == "metal" and booleanity_address_implementation == "packed-hot":
        booleanity_address_member = packed_hot_booleanity_address_member_breakdown(
            events, backend, log_n, stage1_source
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
            stage1_source=stage1_source,
        )
    if backend == "metal" and hamming_weight_implementation == "retained-hot":
        hamming_weight_member = retained_hot_hamming_weight_member_breakdown(
            events, backend, log_n, stage1_source
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
            stage1_source,
        )
    outer_remainder_member = outer_remainder_member_breakdown(
        events,
        backend,
        log_n,
        outer_remainder_cutoff_log2,
        outer_remainder_trace_cutoff_log2,
        product_uniskip_outer_carrier,
        registers_claim_implementation == "outer-carrier-alias-hybrid",
    )
    registers_claim_member = registers_claim_member_breakdown(
        events,
        backend,
        log_n,
        registers_claim_implementation,
        outer_remainder_member["registers_claim_carrier"],
    )
    product_uniskip = product_uniskip_observation(
        events,
        backend,
        log_n,
        product_uniskip_outer_carrier,
    )
    instruction_claim = instruction_claim_observation(events, backend, log_n)
    ram_cycle_family = ram_cycle_family_breakdown(events, backend, log_n)
    ram_members = ram_cycle_family["members"]
    ram_read_write_member = ram_members["read_write"]
    ram_hamming_member = ram_members["hamming_booleanity"]
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
        (BYTECODE_ADDRESS_KERNEL, bytecode_address_member),
        (INSTRUCTION_INPUT_KERNEL, instruction_input_member),
        (INSTRUCTION_READ_RAF_KERNEL, instruction_read_raf_member),
        (BOOLEANITY_ADDRESS_KERNEL, booleanity_address_member),
        (HAMMING_WEIGHT_KERNEL, hamming_weight_member),
        (REGISTERS_CLAIM_KERNEL, registers_claim_member),
        *(
            (kernel, ram_members[member_name])
            for member_name, kernel in RAM_CYCLE_FAMILY_MEMBERS
        ),
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
    backend_witness_prepare_us = unique_named_span_duration_us(
        events, BACKEND_WITNESS_PREP_SPAN
    )
    bytecode_address_member["backend_witness_prepare_us"] = (
        backend_witness_prepare_us
    )
    return {
        "piop_us": unique_span_duration_us(events),
        "backend_witness_prepare_us": backend_witness_prepare_us,
        "attribution": attribution,
        "bytecode_config": bytecode_config,
        "bytecode_member": bytecode_member,
        "bytecode_address_config": bytecode_address_config,
        "bytecode_address_member": bytecode_address_member,
        "instruction_input_config": instruction_input_config,
        "instruction_input_member": instruction_input_member,
        "instruction_read_raf_config": instruction_read_raf_config,
        "instruction_read_raf_member": instruction_read_raf_member,
        "booleanity_address_config": booleanity_address_config,
        "booleanity_address_member": booleanity_address_member,
        "hamming_weight_config": hamming_weight_config,
        "hamming_weight_member": hamming_weight_member,
        "outer_remainder_config": outer_remainder_config,
        "outer_remainder_member": outer_remainder_member,
        "registers_claim_config": registers_claim_config,
        "registers_claim_member": registers_claim_member,
        "product_uniskip": product_uniskip,
        "instruction_claim": instruction_claim,
        "ram_cycle_family": ram_cycle_family,
        "ram_cycle_family_owner": ram_cycle_family["owner"],
        "ram_raf_evaluation_member": ram_members["raf_evaluation"],
        "ram_read_write_member": ram_read_write_member,
        "ram_val_check_member": ram_members["val_check"],
        "ram_ra_claim_reduction_member": ram_members["ra_claim_reduction"],
        "ram_hamming_member": ram_hamming_member,
        "ram_ra_virtualization_member": ram_members["ra_virtualization"],
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
        "--instruction-read-raf-metal-scatter-threads",
        type=int,
        choices=[128, 256, 512, 1024],
        default=256,
    )
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
        choices=["cpu", "csr-shadow", "address-major-shadow", "address-major"],
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
    result.add_argument(
        "--registers-claim-metal-implementation",
        choices=["cpu", "outer-carrier-alias-hybrid"],
        default="outer-carrier-alias-hybrid",
    )
    result.add_argument(
        "--registers-claim-metal-trace-cutoff-log2",
        type=int,
        choices=[25, 26, 27, 28],
        default=25,
    )
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
    if args.registers_claim_metal_trace_cutoff_log2 > args.log_n:
        print(
            "error: RegistersClaim Metal trace cutoff disables the measured backend",
            file=sys.stderr,
        )
        return 2
    if args.registers_claim_metal_implementation == "outer-carrier-alias-hybrid":
        if (
            args.outer_remainder_metal_trace_cutoff_log2
            > args.registers_claim_metal_trace_cutoff_log2
            or args.instruction_input_metal_trace_cutoff_log2
            > args.registers_claim_metal_trace_cutoff_log2
        ):
            print(
                "error: RegistersClaim carrier producers must activate no later than the consumer",
                file=sys.stderr,
            )
            return 2
        if args.instruction_input_metal_cutoff_log2 < 1 + args.log_n // 2:
            print(
                "error: RegistersClaim alias requires InstructionInput host tables before the midpoint",
                file=sys.stderr,
            )
            return 2
        if args.outer_remainder_metal_binding_plan != "b_only_v1":
            print(
                "error: RegistersClaim carrier promotion is frozen to b_only_v1",
                file=sys.stderr,
            )
            return 2
    if (
        args.local_kernel == REGISTERS_CLAIM_KERNEL
        and args.registers_claim_metal_implementation
        != "outer-carrier-alias-hybrid"
    ):
        print(
            "error: RegistersClaim local evaluation requires the carrier-alias route",
            file=sys.stderr,
        )
        return 2
    if (
        args.local_kernel == BYTECODE_ADDRESS_KERNEL
        and args.bytecode_address_metal_implementation != "address-major"
    ):
        print(
            "error: Bytecode address local evaluation requires the AddressMajor route",
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
    producer_orders = []
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
            producer_control_enabled = (
                args.bytecode_address_metal_implementation == "address-major"
            )
            producer_order = (
                ["target", "control"] if index % 2 == 0 else ["control", "target"]
            )
            if producer_control_enabled:
                producer_orders.append(producer_order)
                execution_order = (
                    ["optimized", "metal", "metal_address_control"]
                    if index % 2 == 0
                    else ["metal_address_control", "metal", "optimized"]
                )
            else:
                execution_order = order
            results: dict[str, dict[str, Any]] = {}
            for arm in execution_order:
                backend = "optimized" if arm == "optimized" else "metal"
                address_implementation = (
                    "cpu"
                    if arm == "metal_address_control"
                    else args.bytecode_address_metal_implementation
                )
                results[arm] = run_backend(
                    root,
                    binary,
                    artifact_dir,
                    args.workload,
                    args.log_n,
                    backend,
                    args.instruction_ra_materialize_width,
                    args.instruction_ra_reuse_inverse,
                    args.instruction_read_raf_metal_scatter_threads,
                    args.bytecode_metal_message_threads,
                    args.bytecode_metal_transition_threads,
                    args.bytecode_metal_max_threadgroups,
                    args.bytecode_metal_cutoff_log2,
                    args.bytecode_metal_trace_cutoff_log2,
                    address_implementation,
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
                    args.registers_claim_metal_implementation,
                    args.registers_claim_metal_trace_cutoff_log2,
                    index + 1,
                    args.timeout_seconds,
                    arm.replace("_", "-"),
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
            cpu_bytecode_address_us = float(
                results["optimized"]["bytecode_address_member"]["components"][
                    "member_us"
                ]
            )
            metal_bytecode_address_us = float(
                results["metal"]["bytecode_address_member"]["components"]["member_us"]
            )
            bytecode_address_prepare_delta_us = (
                float(results["metal"]["backend_witness_prepare_us"])
                - float(results["optimized"]["backend_witness_prepare_us"])
            )
            bytecode_address_control_prepare_us = (
                float(results["metal_address_control"]["backend_witness_prepare_us"])
                if producer_control_enabled
                else None
            )
            bytecode_address_target_control_delta_us = (
                float(results["metal"]["backend_witness_prepare_us"])
                - bytecode_address_control_prepare_us
                if bytecode_address_control_prepare_us is not None
                else bytecode_address_prepare_delta_us
            )
            bytecode_address_stage1_topology_us = None
            bytecode_address_control_stage1_topology_us = None
            bytecode_address_irraf_scatter_us = None
            bytecode_address_control_irraf_scatter_us = None
            if producer_control_enabled:
                target_address = results["metal"]["bytecode_address_member"]
                control_address = results["metal_address_control"][
                    "bytecode_address_member"
                ]
                target_topology = target_address["topology_observation"]
                control_topology = control_address["topology_observation"]
                target_resources = target_address["resource_observation"]
                target_scatter = results["metal"]["instruction_read_raf_member"]
                control_scatter = results["metal_address_control"][
                    "instruction_read_raf_member"
                ]
                if (
                    target_topology is None
                    or target_topology["enabled"] is not True
                    or control_topology is None
                    or control_topology["enabled"] is not False
                    or target_resources is None
                    or target_resources.get("producer_kind")
                    != "fused_stage1_grouped_v1"
                    or target_scatter["fused_bytecode_observation"] is None
                    or control_scatter["fused_bytecode_observation"] is not None
                    or target_scatter["scatter_wall_us"] is None
                    or control_scatter["scatter_wall_us"] is None
                ):
                    raise ValueError(
                        "fused Bytecode address producer/control evidence is incomplete"
                    )
                bytecode_address_stage1_topology_us = float(
                    target_topology["wall_us"]
                )
                bytecode_address_control_stage1_topology_us = float(
                    control_topology["wall_us"]
                )
                bytecode_address_irraf_scatter_us = float(
                    target_scatter["scatter_wall_us"]
                )
                bytecode_address_control_irraf_scatter_us = float(
                    control_scatter["scatter_wall_us"]
                )
                bytecode_address_signed_producer_delta_us = (
                    bytecode_address_stage1_topology_us
                    - bytecode_address_control_stage1_topology_us
                    + bytecode_address_irraf_scatter_us
                    - bytecode_address_control_irraf_scatter_us
                )
            else:
                bytecode_address_signed_producer_delta_us = (
                    bytecode_address_target_control_delta_us
                )
            bytecode_address_charged_producer_delta_us = max(
                0.0, bytecode_address_signed_producer_delta_us
            )
            charged_metal_address_us = (
                metal_bytecode_address_us
                + bytecode_address_charged_producer_delta_us
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
                    "cpu_bytecode_address_us": cpu_bytecode_address_us,
                    "metal_bytecode_address_us": metal_bytecode_address_us,
                    "bytecode_address_backend_witness_prepare_delta_us": (
                        bytecode_address_prepare_delta_us
                    ),
                    **(
                        {
                            "metal_bytecode_address_control_prepare_us": (
                                bytecode_address_control_prepare_us
                            ),
                            "bytecode_address_target_control_prepare_delta_us": (
                                bytecode_address_target_control_delta_us
                            ),
                            "metal_bytecode_address_stage1_topology_us": (
                                bytecode_address_stage1_topology_us
                            ),
                            "metal_bytecode_address_control_stage1_topology_us": (
                                bytecode_address_control_stage1_topology_us
                            ),
                            "metal_bytecode_address_irraf_scatter_us": (
                                bytecode_address_irraf_scatter_us
                            ),
                            "metal_bytecode_address_control_irraf_scatter_us": (
                                bytecode_address_control_irraf_scatter_us
                            ),
                            "producer_order": producer_order,
                        }
                        if producer_control_enabled
                        else {}
                    ),
                    "bytecode_address_charged_producer_delta_us": (
                        bytecode_address_charged_producer_delta_us
                    ),
                    "charged_metal_address_us": charged_metal_address_us,
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
                    "cpu_registers_claim_us": float(
                        results["optimized"]["registers_claim_member"]["components"][
                            "member_us"
                        ]
                    ),
                    "metal_registers_claim_us": float(
                        results["metal"]["registers_claim_member"]["components"][
                            "member_us"
                        ]
                    ),
                    "cpu_instruction_read_raf_us": float(
                        results["optimized"]["instruction_read_raf_member"][
                            "components"
                        ]["member_us"]
                    ),
                    "metal_instruction_read_raf_us": float(
                        results["metal"]["instruction_read_raf_member"][
                            "components"
                        ]["member_us"]
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
                    "cpu_ram_raf_evaluation_us": float(
                        results["optimized"]["ram_raf_evaluation_member"][
                            "components"
                        ]["member_us"]
                    ),
                    "metal_ram_raf_evaluation_us": float(
                        results["metal"]["ram_raf_evaluation_member"]["components"][
                            "member_us"
                        ]
                    ),
                    "cpu_ram_read_write_us": float(
                        results["optimized"]["ram_read_write_member"]["components"][
                            "member_us"
                        ]
                    ),
                    "metal_ram_read_write_us": float(
                        results["metal"]["ram_read_write_member"]["components"][
                            "member_us"
                        ]
                    ),
                    "metal_ram_read_write_charged_us": float(
                        results["metal"]["ram_read_write_member"]["components"][
                            "charged_member_us"
                        ]
                    ),
                    "cpu_ram_val_check_us": float(
                        results["optimized"]["ram_val_check_member"]["components"][
                            "member_us"
                        ]
                    ),
                    "metal_ram_val_check_us": float(
                        results["metal"]["ram_val_check_member"]["components"][
                            "member_us"
                        ]
                    ),
                    "cpu_ram_ra_claim_reduction_us": float(
                        results["optimized"]["ram_ra_claim_reduction_member"]
                        ["components"]["member_us"]
                    ),
                    "metal_ram_ra_claim_reduction_us": float(
                        results["metal"]["ram_ra_claim_reduction_member"][
                            "components"
                        ]["member_us"]
                    ),
                    "cpu_ram_hamming_us": float(
                        results["optimized"]["ram_hamming_member"]["components"][
                            "member_us"
                        ]
                    ),
                    "metal_ram_hamming_us": float(
                        results["metal"]["ram_hamming_member"]["components"][
                            "member_us"
                        ]
                    ),
                    "metal_ram_hamming_charged_us": float(
                        results["metal"]["ram_hamming_member"]["components"][
                            "charged_member_us"
                        ]
                    ),
                    "cpu_ram_ra_virtualization_us": float(
                        results["optimized"]["ram_ra_virtualization_member"]
                        ["components"]["member_us"]
                    ),
                    "metal_ram_ra_virtualization_us": float(
                        results["metal"]["ram_ra_virtualization_member"][
                            "components"
                        ]["member_us"]
                    ),
                    "metal_ram_cycle_family_owner_us": float(
                        results["metal"]["ram_cycle_family_owner"]["wall_us"]
                    ),
                    "metal_ram_cycle_family_witness_prepare_us": float(
                        results["metal"]["ram_cycle_family"]["components"]
                        ["witness_prepare_us"]
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
                    "bytecode_address": {
                        "optimized_config": results["optimized"][
                            "bytecode_address_config"
                        ],
                        "metal_config": results["metal"]["bytecode_address_config"],
                        "optimized_member": results["optimized"][
                            "bytecode_address_member"
                        ],
                        "metal_member": results["metal"][
                            "bytecode_address_member"
                        ],
                        "producer_order": producer_order
                        if producer_control_enabled
                        else None,
                        "control_config": results["metal_address_control"][
                            "bytecode_address_config"
                        ]
                        if producer_control_enabled
                        else None,
                        "control_member": results["metal_address_control"][
                            "bytecode_address_member"
                        ]
                        if producer_control_enabled
                        else None,
                        "control_stage1_source": results["metal_address_control"][
                            "instruction_read_raf_member"
                        ]["source_observation"]
                        if producer_control_enabled
                        else None,
                        "control_stage1_projection": results[
                            "metal_address_control"
                        ]["instruction_read_raf_member"]["stage1_projection"]
                        if producer_control_enabled
                        else None,
                        "control_stage1_scatter": results[
                            "metal_address_control"
                        ]["instruction_read_raf_member"]["scatter_observation"]
                        if producer_control_enabled
                        else None,
                        "control_backend_witness_prepare_us": results[
                            "metal_address_control"
                        ]["backend_witness_prepare_us"]
                        if producer_control_enabled
                        else None,
                        "control_max_rss_bytes": results["metal_address_control"][
                            "max_rss_bytes"
                        ]
                        if producer_control_enabled
                        else None,
                        "control_command": results["metal_address_control"]["command"]
                        if producer_control_enabled
                        else None,
                        "control_artifacts": results["metal_address_control"]["artifacts"]
                        if producer_control_enabled
                        else None,
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
                    "registers_claim": {
                        "optimized_config": results["optimized"][
                            "registers_claim_config"
                        ],
                        "metal_config": results["metal"]["registers_claim_config"],
                        "optimized_member": results["optimized"][
                            "registers_claim_member"
                        ],
                        "metal_member": results["metal"]["registers_claim_member"],
                    },
                    "instruction_read_raf": {
                        "optimized_config": results["optimized"][
                            "instruction_read_raf_config"
                        ],
                        "metal_config": results["metal"][
                            "instruction_read_raf_config"
                        ],
                        "optimized_member": results["optimized"][
                            "instruction_read_raf_member"
                        ],
                        "metal_member": results["metal"][
                            "instruction_read_raf_member"
                        ],
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
                    "ram_cycle_family": {
                        "optimized": results["optimized"]["ram_cycle_family"],
                        "metal": results["metal"]["ram_cycle_family"],
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
                    "bytecode_address_charge": {
                        "backend_witness_prepare_delta_ns": round(
                            bytecode_address_prepare_delta_us * 1000.0
                        ),
                        "target_control_prepare_delta_ns": round(
                            bytecode_address_target_control_delta_us * 1000.0
                        ),
                        "stage1_topology_target_ns": (
                            round(bytecode_address_stage1_topology_us * 1000.0)
                            if bytecode_address_stage1_topology_us is not None
                            else None
                        ),
                        "stage1_topology_control_ns": (
                            round(
                                bytecode_address_control_stage1_topology_us * 1000.0
                            )
                            if bytecode_address_control_stage1_topology_us is not None
                            else None
                        ),
                        "instruction_read_raf_scatter_target_ns": (
                            round(bytecode_address_irraf_scatter_us * 1000.0)
                            if bytecode_address_irraf_scatter_us is not None
                            else None
                        ),
                        "instruction_read_raf_scatter_control_ns": (
                            round(
                                bytecode_address_control_irraf_scatter_us * 1000.0
                            )
                            if bytecode_address_control_irraf_scatter_us is not None
                            else None
                        ),
                        "signed_fused_producer_delta_ns": round(
                            bytecode_address_signed_producer_delta_us * 1000.0
                        ),
                        "control_backend_witness_prepare_ns": (
                            microseconds_to_nanoseconds(
                                bytecode_address_control_prepare_us
                            )
                            if bytecode_address_control_prepare_us is not None
                            else None
                        ),
                        "producer_order": producer_order
                        if producer_control_enabled
                        else None,
                        "charged_producer_delta_ns": round(
                            bytecode_address_charged_producer_delta_us * 1000.0
                        ),
                        "charged_metal_address_ns": microseconds_to_nanoseconds(
                            charged_metal_address_us
                        ),
                    },
                    "ram_cycle_family_charge": {
                        "model": RAM_CYCLE_FAMILY_CHARGE_MODEL,
                        "cpu_raw_members_ns": microseconds_to_nanoseconds(
                            results["optimized"]["ram_cycle_family"]["components"][
                                "raw_member_us"
                            ]
                        ),
                        "metal_raw_members_ns": microseconds_to_nanoseconds(
                            results["metal"]["ram_cycle_family"]["components"][
                                "raw_member_us"
                            ]
                        ),
                        "witness_prepare_ns": microseconds_to_nanoseconds(
                            results["metal"]["ram_cycle_family"]["components"][
                                "witness_prepare_us"
                            ]
                        ),
                        "owner_prepare_ns": microseconds_to_nanoseconds(
                            results["metal"]["ram_cycle_family"]["components"][
                                "owner_prepare_us"
                            ]
                        ),
                        "charged_metal_ns": microseconds_to_nanoseconds(
                            results["metal"]["ram_cycle_family"]["components"][
                                "charged_member_us"
                            ]
                        ),
                        "producer_charge_count": results["metal"][
                            "ram_cycle_family"
                        ]["components"]["producer_charge_count"],
                    },
                    "arms": {
                        **{
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
                            "bytecode_address": {
                                **member_record(
                                    results[backend]["bytecode_address_member"]
                                ),
                                "backend_witness_prepare_ns": (
                                    microseconds_to_nanoseconds(
                                        results[backend][
                                            "backend_witness_prepare_us"
                                        ]
                                    )
                                ),
                            },
                            "instruction_input": member_record(
                                results[backend]["instruction_input_member"],
                                include_prefetch=True,
                            ),
                            "instruction_input_row_lifecycle": results[backend][
                                "instruction_input_member"
                            ]["row_lifecycle"],
                            "registers_claim": member_record(
                                results[backend]["registers_claim_member"]
                            ),
                            "instruction_read_raf": member_record(
                                results[backend]["instruction_read_raf_member"]
                            ),
                            "instruction_read_raf_resources": {
                                "source": results[backend][
                                    "instruction_read_raf_member"
                                ]["source_observation"],
                                "scatter": results[backend][
                                    "instruction_read_raf_member"
                                ]["scatter_observation"],
                                "stage1_projection": results[backend][
                                    "instruction_read_raf_member"
                                ]["stage1_projection"],
                            },
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
                            "ram_cycle_family": results[backend][
                                "ram_cycle_family"
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
                            "bytecode_address_config": results[backend][
                                "bytecode_address_config"
                            ],
                            "command": results[backend]["command"],
                            "artifacts": results[backend]["artifacts"],
                            }
                            for backend in ("optimized", "metal")
                        },
                        **(
                            {
                                "metal_address_control": {
                                    "piop_ns": microseconds_to_nanoseconds(
                                        results["metal_address_control"]["piop_us"]
                                    ),
                                    "backend_witness_prepare_ns": (
                                        microseconds_to_nanoseconds(
                                            results["metal_address_control"][
                                                "backend_witness_prepare_us"
                                            ]
                                        )
                                    ),
                                    "max_rss_bytes": results[
                                        "metal_address_control"
                                    ]["max_rss_bytes"],
                                    "bytecode_address": member_record(
                                        results["metal_address_control"][
                                            "bytecode_address_member"
                                        ]
                                    ),
                                    "instruction_read_raf_resources": {
                                        "source": results["metal_address_control"][
                                            "instruction_read_raf_member"
                                        ]["source_observation"],
                                        "scatter": results["metal_address_control"][
                                            "instruction_read_raf_member"
                                        ]["scatter_observation"],
                                        "stage1_projection": results[
                                            "metal_address_control"
                                        ]["instruction_read_raf_member"][
                                            "stage1_projection"
                                        ],
                                    },
                                    "bytecode_address_config": results[
                                        "metal_address_control"
                                    ]["bytecode_address_config"],
                                    "command": results["metal_address_control"][
                                        "command"
                                    ],
                                    "artifacts": results["metal_address_control"][
                                        "artifacts"
                                    ],
                                }
                            }
                            if producer_control_enabled
                            else {}
                        ),
                    },
                }
            )
        metrics = summarize_pairs(pairs)
        bytecode_address_sparse_receipt_exact = producer_control_enabled
        bytecode_address_sparse_carrier_borrowed = producer_control_enabled
        bytecode_address_sparse_member_storage_exact = producer_control_enabled
        bytecode_address_sparse_dispatch_geometry_exact = producer_control_enabled
        bytecode_address_producer_control_exact = producer_control_enabled
        bytecode_address_fused_receipt_exact = producer_control_enabled
        bytecode_address_fused_no_extra_work = producer_control_enabled
        for sample in attributions:
            address_sample = sample["bytecode_address"]
            target_resources = address_sample["metal_member"]["resource_observation"]
            target_projection = sample["instruction_read_raf"]["metal_member"][
                "stage1_projection"
            ]
            target_source = sample["instruction_read_raf"]["metal_member"][
                "source_observation"
            ]
            if target_resources is None or target_source is None:
                bytecode_address_sparse_receipt_exact = False
                bytecode_address_sparse_carrier_borrowed = False
                bytecode_address_sparse_member_storage_exact = False
                bytecode_address_sparse_dispatch_geometry_exact = False
                bytecode_address_fused_receipt_exact = False
                bytecode_address_fused_no_extra_work = False
                continue
            publish = target_resources["carrier_publish"]
            complete = target_resources["address_major_complete"]
            topology = target_resources["fused_topology_prepare"]
            scatter = sample["instruction_read_raf"]["metal_member"][
                "scatter_observation"
            ]
            fused_scatter = sample["instruction_read_raf"]["metal_member"][
                "fused_bytecode_observation"
            ]
            physical_rows = publish["physical_rows"]
            work_items = publish["work_items"]
            offset_bytes = 4 * ((1 << 13) + 1)
            topology_bytes = (
                publish.get("bytecode_descriptor_bytes", 0)
                + publish.get("bytecode_pivot_bytes", 0)
                + publish.get("bytecode_chunk_offset_bytes", 0)
            )
            bytecode_address_sparse_receipt_exact &= (
                target_resources.get("producer_kind") == "fused_stage1_grouped_v1"
                and physical_rows
                == target_projection["compact_rows"]["explicit_rows"]
                and (physical_rows + 4095) // 4096 <= work_items <= physical_rows
                and publish["carrier_resident_bytes"]
                == 10 * physical_rows + 8 * work_items + offset_bytes
                and complete["producer_persistent_write_bytes"] == 10 * physical_rows
                and complete["producer_topology_read_bytes"] == topology_bytes
                and complete["producer_logical_movement_bytes"]
                == 10 * physical_rows + topology_bytes
            )
            bytecode_address_fused_receipt_exact &= (
                topology is not None
                and topology["enabled"] is True
                and fused_scatter is not None
                and publish["physical_rows"] == topology["physical_rows"]
                and publish["work_items"] == topology["work_items"]
                and all(
                    publish[publish_field] == topology[topology_field]
                    for publish_field, topology_field in (
                        ("bytecode_descriptor_storage_id", "descriptor_storage_id"),
                        ("bytecode_descriptor_bytes", "descriptor_bytes"),
                        ("bytecode_pivot_storage_id", "pivot_storage_id"),
                        ("bytecode_pivot_bytes", "pivot_bytes"),
                        ("bytecode_chunk_offset_storage_id", "chunk_offset_storage_id"),
                        ("bytecode_chunk_offset_bytes", "chunk_offset_bytes"),
                        ("carrier_work_item_storage_id", "work_item_storage_id"),
                        ("carrier_work_item_bytes", "work_item_bytes"),
                        ("carrier_address_offset_storage_id", "address_offset_storage_id"),
                        ("carrier_address_offset_bytes", "address_offset_bytes"),
                    )
                )
            )
            bytecode_address_fused_no_extra_work &= (
                topology is not None
                and topology["shared_source_row_scans"] == 1
                and topology["additional_source_row_scans"] == 0
                and topology["extra_source_scans"] == 0
                and topology["member_upload_bytes"] == 0
                and scatter is not None
                and scatter["command_buffers"] == 1
                and scatter["waits"] == 1
                and scatter["encoders"] == 1
                and scatter["dispatches"] == 1
                and scatter["source_copy_bytes"] == 0
                and scatter["full_plane_readback_bytes"] == 0
                and fused_scatter is not None
                and fused_scatter["additional_source_row_scans"] == 0
                and fused_scatter["member_upload_bytes"] == 0
                and publish["shared_source_row_scans"] == 1
                and publish["additional_source_row_scans"] == 0
                and publish["member_upload_bytes"] == 0
                and publish["command_buffers"] == 1
                and publish["waits"] == 1
                and publish["encoders"] == 1
                and publish["dispatches"] == 1
                and complete["command_buffers"] == 1
                and complete["waits"] == 1
                and complete["member_source_scans"] == 0
                and complete["member_source_upload_bytes"] == 0
            )
            bytecode_address_sparse_carrier_borrowed &= (
                complete["carrier_completion_serial"]
                == publish["carrier_completion_serial"]
                and all(
                    complete[field] == publish[field]
                    for field in (
                        "carrier_occurrence_storage_id",
                        "carrier_magnitude_storage_id",
                        "carrier_work_item_storage_id",
                        "carrier_address_offset_storage_id",
                    )
                )
                and complete["member_carrier_owned_bytes"] == 0
                and complete["member_source_scans"] == 0
                and complete["member_source_upload_bytes"] == 0
                and complete["carrier_released"] is True
            )
            equality_bytes = 16 * 9 * ((1 << 15) + (1 << args.log_n) // (1 << 15))
            partial_bytes = 16 * 9 * work_items
            output_bytes = 16 * 9 * (1 << 13)
            bytecode_address_sparse_member_storage_exact &= (
                complete["equality_bytes"] == equality_bytes
                and complete["padding_bytes"] == 5 * 16
                and complete["partial_bytes"] == partial_bytes
                and complete["output_readback_bytes"] == output_bytes
                and complete["member_owned_bytes"]
                == equality_bytes + 5 * 16 + partial_bytes + output_bytes
            )
            bytecode_address_sparse_dispatch_geometry_exact &= (
                complete["worker_dispatches"] == 1
                and complete["worker_variant"] == "packed4_halfwidth_v1"
                and complete["worker_simd_width"] == 32
                and complete["worker_threads"] == 128
                and complete["worker_items_per_threadgroup"] == 4
                and complete["worker_threadgroups"] == (work_items + 3) // 4
                and complete["worker_tail_slots"] == (4 - work_items % 4) % 4
                and complete["worker_dynamic_threadgroup_bytes"] == 0
                and complete["worker_static_threadgroup_bytes"] == 0
                and complete["worker_threadgroup_bytes"] == 0
                and complete["reducer_dispatches"] == 1
                and complete["reducer_threads"] == 256
                and complete["reducer_threadgroups"] == 288
                and complete["reducer_static_threadgroup_bytes"] == 0
            )

            control_member = address_sample["control_member"]
            control_source = address_sample["control_stage1_source"]
            control_projection = address_sample["control_stage1_projection"]
            control_scatter = address_sample["control_stage1_scatter"]
            if (
                control_member is None
                or control_source is None
                or control_projection is None
                or control_scatter is None
            ):
                bytecode_address_producer_control_exact = False
                bytecode_address_fused_no_extra_work = False
                continue
            source_geometry_fields = {
                "rows",
                "row_bytes",
                "claim_bytes",
                "resident_device_bytes",
                "count_chunks",
                "count_bytes",
                "host_row_write_bytes",
                "host_claim_write_bytes",
                "host_count_update_bytes",
                "count_order",
                "publication_kind",
                "complete_overwrite",
                "source_windows",
                "member_upload_bytes",
                "projection_dispatches",
            }
            compact_geometry_fields = {
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
                "resident_rows",
                "explicit_rows",
            }
            witness_geometry_fields = {
                "cycles",
                "source",
                "admitted",
                "fallback_reason",
                "native_register_contract_bytes",
                "shift_late_copy_dispatches",
                "shift_resident_bytes",
                "shift_row_extractions",
            }
            target_command = list(sample["commands"]["metal"])
            control_command = list(address_sample["control_command"] or [])
            for command in (target_command, control_command):
                try:
                    implementation_index = command.index(
                        "--bytecode-address-metal-implementation"
                    )
                    command[implementation_index + 1] = "<implementation>"
                except (ValueError, IndexError):
                    bytecode_address_producer_control_exact = False
            bytecode_address_producer_control_exact &= (
                address_sample["control_config"]
                == {
                    "implementation": "cpu",
                    "trace_cutoff": 1
                    << args.bytecode_address_metal_trace_cutoff_log2,
                    "outer_tiles": args.bytecode_address_metal_outer_tiles,
                }
                and control_member["route_observation"]
                == {
                    "cycles": 1 << args.log_n,
                    "requested": "cpu",
                    "realized_route": "cpu",
                    "fallback_reason": "configured_cpu",
                }
                and control_member["resource_observation"] is None
                and control_member["topology_observation"] is not None
                and control_member["topology_observation"]["enabled"] is False
                and control_member["metal_counts"]
                == {
                    "route": 1,
                    "carrier_publish": 0,
                    "fused_topology_prepare": 1,
                    "fused_carrier_publish": 0,
                    "address_major_prepare": 0,
                    "address_major_join": 0,
                    "address_major_complete": 0,
                    "shadow_prepare": 0,
                    "shadow_join": 0,
                }
                and all(
                    target_source[field] == control_source[field]
                    for field in source_geometry_fields
                )
                and all(
                    target_projection["compact_rows"][field]
                    == control_projection["compact_rows"][field]
                    for field in compact_geometry_fields
                )
                and all(
                    target_projection["witness"][field]
                    == control_projection["witness"][field]
                    for field in witness_geometry_fields
                )
                and target_command == control_command
            )
            bytecode_address_fused_no_extra_work &= (
                control_member["topology_observation"] is not None
                and control_member["topology_observation"][
                    "additional_source_row_scans"
                ]
                == 0
                and control_member["topology_observation"]["extra_source_scans"] == 0
                and control_member["topology_observation"]["member_upload_bytes"] == 0
                and control_scatter["command_buffers"] == 1
                and control_scatter["waits"] == 1
                and control_scatter["encoders"] == 1
                and control_scatter["dispatches"] == 1
                and control_scatter["source_copy_bytes"] == 0
                and control_scatter["full_plane_readback_bytes"] == 0
                and "bytecode_fused" not in control_scatter
            )
        expected_producer_orders = [
            ["target", "control"] if index % 2 == 0 else ["control", "target"]
            for index in range(args.repeats)
        ]
        bytecode_address_producer_control_ordered = (
            producer_control_enabled and producer_orders == expected_producer_orders
        )
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
                "moved_work_control_lower_metal_total": all(
                    pair["metal_us"] + pair["metal_prepare_us"]
                    < pair["cpu_us"] + pair["cpu_prepare_us"]
                    for pair in pairs
                ),
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
                            (
                                sample["ram_cycle_family"][backend]["components"][
                                    "charged_member_us"
                                ]
                                > 0.0
                                if local_kernel["name"] == RAM_CYCLE_FAMILY_KERNEL
                                else sample["outer_remainder"][f"{backend}_member"][
                                    "components"
                                ]["member_us"]
                                > 0.0
                            )
                            if local_kernel["name"]
                            in {OUTER_REMAINDER_KERNEL, RAM_CYCLE_FAMILY_KERNEL}
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
                "ram_cycle_family_witness_prepare_exact": all(
                    sample["ram_cycle_family"]["optimized"]["witness_prepare"]
                    is None
                    and sample["ram_cycle_family"]["metal"]["witness_prepare"]
                    is not None
                    and sample["ram_cycle_family"]["metal"]["witness_prepare"][
                        "address_plane_bytes"
                    ]
                    == 4 * (1 << args.log_n)
                    and sample["ram_cycle_family"]["metal"]["witness_prepare"][
                        "address_plane_upload_bytes"
                    ]
                    == 4 * (1 << args.log_n)
                    and sample["ram_cycle_family"]["metal"]["components"][
                        "producer_charge_count"
                    ]
                    == 1
                    for sample in attributions
                ),
                "ram_cycle_family_owner_exact": all(
                    sample["ram_cycle_family"]["optimized"]["owner"] is None
                    and sample["ram_cycle_family"]["metal"]["owner"] is not None
                    and sample["ram_cycle_family"]["metal"]["owner"][
                        "complete_publication"
                    ]
                    is True
                    and sample["ram_cycle_family"]["metal"]["owner"][
                        "source_collection_performed"
                    ]
                    is True
                    and sample["ram_cycle_family"]["metal"]["owner"][
                        "additional_source_row_scans"
                    ]
                    == 0
                    and sample["ram_cycle_family"]["metal"]["owner"][
                        "member_upload_bytes"
                    ]
                    == 0
                    for sample in attributions
                ),
                "ram_cycle_family_terminal_take_exact": all(
                    sample["ram_cycle_family"]["optimized"]["terminal_take"] is None
                    and sample["ram_cycle_family"]["metal"]["terminal_take"][
                        "session_owner_removed"
                    ]
                    is True
                    and sample["ram_cycle_family"]["metal"]["terminal_take"][
                        "columns_removed"
                    ]
                    is True
                    for sample in attributions
                ),
                "ram_cycle_family_routes_exact": all(
                    all(
                        member["resource_observation"] is None
                        for member in sample["ram_cycle_family"]["optimized"][
                            "members"
                        ].values()
                    )
                    and all(
                        member["resource_observation"] is not None
                        for member in sample["ram_cycle_family"]["metal"][
                            "members"
                        ].values()
                    )
                    and sample["ram_cycle_family"]["metal"][
                        "canonical_nonoverlap"
                    ]
                    is True
                    for sample in attributions
                ),
                "ram_cycle_family_raw_members_gate": all(
                    metrics[name]["clears"]
                    for name in (
                        "ram_raf_evaluation_decision",
                        "ram_read_write_decision",
                        "ram_val_check_decision",
                        "ram_ra_claim_reduction_decision",
                        "ram_hamming_booleanity_decision",
                        "ram_ra_virtualization_decision",
                    )
                ),
                "ram_cycle_family_gate": metrics[
                    "ram_cycle_family_decision"
                ]["clears"],
                "stable_source": True,
                "stable_binary": True,
                "production_contract": args.mode == "production",
                "bytecode_local_gate": metrics["bytecode_read_raf_cycle_decision"][
                    "clears"
                ],
                "bytecode_address_raw_local_gate": metrics[
                    "bytecode_read_raf_address_decision"
                ]["clears"],
                "bytecode_address_charged_local_gate": metrics[
                    "bytecode_read_raf_address_charged_decision"
                ]["clears"],
                "bytecode_address_local_gate": (
                    metrics["bytecode_read_raf_address_decision"]["clears"]
                    and metrics["bytecode_read_raf_address_charged_decision"][
                        "clears"
                    ]
                    and bytecode_address_sparse_receipt_exact
                    and bytecode_address_sparse_carrier_borrowed
                    and bytecode_address_sparse_member_storage_exact
                    and bytecode_address_sparse_dispatch_geometry_exact
                    and bytecode_address_fused_receipt_exact
                    and bytecode_address_fused_no_extra_work
                    and bytecode_address_producer_control_exact
                    and bytecode_address_producer_control_ordered
                ),
                "bytecode_address_cpu_control": all(
                    sample["bytecode_address"]["optimized_config"] is None
                    and not any(
                        sample["bytecode_address"]["optimized_member"][
                            "metal_counts"
                        ].values()
                    )
                    and sample["bytecode_address"]["optimized_member"][
                        "resource_observation"
                    ]
                    is None
                    for sample in attributions
                ),
                "bytecode_address_configured_route_exact": all(
                    sample["bytecode_address"]["metal_config"]
                    == {
                        "implementation": args.bytecode_address_metal_implementation,
                        "trace_cutoff": 1
                        << args.bytecode_address_metal_trace_cutoff_log2,
                        "outer_tiles": args.bytecode_address_metal_outer_tiles,
                    }
                    and sample["bytecode_address"]["metal_member"][
                        "route_observation"
                    ]
                    is not None
                    for sample in attributions
                ),
                "bytecode_address_address_major_anti_fallback": (
                    args.bytecode_address_metal_implementation != "address-major"
                    or all(
                        sample["bytecode_address"]["metal_member"][
                            "route_observation"
                        ]
                        == {
                            "cycles": 1 << args.log_n,
                            "requested": "address_major",
                            "realized_route": METAL_BYTECODE_ADDRESS_FUSED_ROUTE,
                            "fallback_reason": "none",
                        }
                        and sample["bytecode_address"]["metal_member"][
                            "metal_counts"
                        ]["address_major_complete"]
                        == 1
                        and sample["bytecode_address"]["metal_member"][
                            "resource_observation"
                        ]
                        is not None
                        for sample in attributions
                    )
                ),
                "bytecode_address_sparse_receipt_exact": (
                    bytecode_address_sparse_receipt_exact
                ),
                "bytecode_address_sparse_carrier_borrowed": (
                    bytecode_address_sparse_carrier_borrowed
                ),
                "bytecode_address_sparse_member_storage_exact": (
                    bytecode_address_sparse_member_storage_exact
                ),
                "bytecode_address_sparse_dispatch_geometry_exact": (
                    bytecode_address_sparse_dispatch_geometry_exact
                ),
                "bytecode_address_fused_receipt_exact": (
                    bytecode_address_fused_receipt_exact
                ),
                "bytecode_address_fused_no_extra_work": (
                    bytecode_address_fused_no_extra_work
                ),
                "bytecode_address_producer_control_exact": (
                    bytecode_address_producer_control_exact
                ),
                "bytecode_address_producer_control_ordered": (
                    bytecode_address_producer_control_ordered
                ),
                "instruction_read_raf_local_gate": metrics[
                    "instruction_read_raf_decision"
                ]["clears"],
                "instruction_read_raf_cpu_control": all(
                    sample["instruction_read_raf"]["optimized_member"][
                        "source_observation"
                    ]
                    is None
                    and sample["instruction_read_raf"]["optimized_member"][
                        "scatter_observation"
                    ]
                    is None
                    and not any(
                        sample["instruction_read_raf"]["optimized_member"][
                            "metal_counts"
                        ].values()
                    )
                    for sample in attributions
                ),
                "instruction_read_raf_stage1_route_exact": all(
                    sample["instruction_read_raf"]["metal_member"]["outer_counts"]
                    == {
                        "prepare": 1,
                        "prove_round": 128 + args.log_n,
                        "finish_rounds": 1,
                        "output_claims": 1,
                    }
                    and sample["instruction_read_raf"]["metal_member"][
                        "metal_counts"
                    ]
                    == {
                        "stage1_source_publish": 1,
                        "stage1_grouped_scatter": 1,
                        "stage1_grouped_sequence_prepare": 1,
                        "address_round": 129,
                        "resident_first_message": 1,
                        "resident_handoff": 1,
                        "resident_round": args.log_n - 17,
                        "readback": 1,
                    }
                    for sample in attributions
                ),
                "instruction_read_raf_source_provenance_exact": all(
                    (source_observation := sample["instruction_read_raf"][
                        "metal_member"
                    ]["source_observation"])
                    is not None
                    and source_observation["rows"] == 1 << args.log_n
                    and source_observation["resident_device_bytes"]
                    == 41 * (1 << args.log_n)
                    and source_observation["member_upload_bytes"] == 0
                    and source_observation["projection_dispatches"] == 0
                    and source_observation["complete_overwrite"] is True
                    for sample in attributions
                ),
                "instruction_read_raf_scatter_exact": all(
                    (scatter_observation := sample["instruction_read_raf"][
                        "metal_member"
                    ]["scatter_observation"])
                    is not None
                    and scatter_observation["threads_per_threadgroup"]
                    == args.instruction_read_raf_metal_scatter_threads
                    and scatter_observation["source_copy_bytes"] == 0
                    and scatter_observation["full_plane_readback_bytes"] == 0
                    and scatter_observation["status_readback_bytes"] == 4
                    and scatter_observation["complete_overwrite"] is True
                    for sample in attributions
                ),
                "instruction_read_raf_scatter_within_roof_target": all(
                    sample["instruction_read_raf"]["metal_member"][
                        "scatter_observation"
                    ]["gpu_active_ns"]
                    <= round(7_250_000 * (2 ** (args.log_n - 25)))
                    for sample in attributions
                ),
                "instruction_read_raf_stage5_owner_reused": all(
                    (source_observation := sample["instruction_read_raf"][
                        "metal_member"
                    ]["source_observation"])
                    is not None
                    and all(
                        (lifecycle := sample[family]["metal_member"][
                            "row_lifecycle"
                        ])["source_kind"]
                        == "stage1_owner_v1"
                        and lifecycle["source_generation"]
                        == source_observation["source_generation"]
                        and lifecycle["source_completion_serial"]
                        == source_observation["completion_serial"]
                        and lifecycle["source_claim_allocation_identity"]
                        == source_observation["claim_allocation_identity"]
                        and lifecycle["device_registry_id"]
                        == source_observation["device_registry_id"]
                        and (
                            lifecycle.get(
                                "stage5_storage_id",
                                lifecycle.get("source_rows_storage_id"),
                            )
                            == source_observation["row_allocation_identity"]
                        )
                        and lifecycle["stage5"]["row_allocations"] == 0
                        and lifecycle["stage5"]["row_upload_bytes"] == 0
                        for family in ("booleanity_address", "hamming_weight")
                    )
                    for sample in attributions
                ),
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
                "registers_claim_cpu_control": all(
                    sample["registers_claim"]["optimized_member"][
                        "resource_observation"
                    ]
                    is None
                    and not any(
                        sample["registers_claim"]["optimized_member"][
                            "metal_counts"
                        ].values()
                    )
                    for sample in attributions
                ),
                "registers_claim_carrier_alias_route_exact": all(
                    (
                        observation := sample["registers_claim"]["metal_member"][
                            "resource_observation"
                        ]
                    )
                    is not None
                    and observation["route"]
                    == {
                        "cycles": 1 << args.log_n,
                        "requested": "outer_carrier_alias_hybrid",
                        "stage1_carry_present": True,
                        "alias_receiver_present": True,
                        "realized_route": "outer_carrier_alias_hybrid",
                        "fallback_reason": "none",
                    }
                    and observation["prepare"]["source_allocations"] == 0
                    and observation["prepare"]["source_upload_bytes"] == 0
                    and observation["prepare"]["source_host_write_bytes"] == 0
                    and observation["midpoint"]["alias_takes"] == 1
                    and observation["midpoint"]["useful_half_width_terms"]
                    == 1 << args.log_n
                    for sample in attributions
                ),
                "registers_claim_shared_provenance_exact": all(
                    (
                        observation := sample["registers_claim"]["metal_member"][
                            "resource_observation"
                        ]
                    )
                    is not None
                    and observation["prepare"]["source_generation"]
                    == observation["outer_carrier"]["source_generation"]
                    and observation["prepare"]["source_compact_storage_id"]
                    == observation["outer_carrier"]["source_compact_storage_id"]
                    == observation["alias_publish"]["source_compact_storage_id"]
                    and observation["prepare"]["source_rd_storage_id"]
                    == observation["outer_carrier"]["rd_storage_id"]
                    == observation["midpoint"]["source_rd_storage_id"]
                    and observation["prepare"]["alias_generation"]
                    == observation["alias_publish"]["alias_generation"]
                    == observation["midpoint"]["alias_generation"]
                    for sample in attributions
                ),
                "registers_claim_local_gate": metrics[
                    "registers_claim_reduction_member_decision"
                ]["clears"],
                "registers_claim_charged_family_gate": metrics[
                    "outer_product_instruction_input_registers_family_decision"
                ]["clears"],
                "piop_gate": metrics["piop_decision"]["clears"],
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
                "instruction_read_raf_stage1": [
                    sample["instruction_read_raf"]["metal_member"][
                        "resource_observation"
                    ]
                    for sample in attributions
                ],
                "registers_claim": [
                    sample["registers_claim"]["metal_member"][
                        "resource_observation"
                    ]
                    for sample in attributions
                ],
                "bytecode_address": [
                    sample["bytecode_address"]["metal_member"][
                        "resource_observation"
                    ]
                    for sample in attributions
                ],
                "ram_cycle_family": [
                    sample["ram_cycle_family"] for sample in attributions
                ],
                "optimized_max_rss_bytes": [
                    pair["arms"]["optimized"]["max_rss_bytes"] for pair in pair_records
                ],
                "metal_max_rss_bytes": [
                    pair["arms"]["metal"]["max_rss_bytes"] for pair in pair_records
                ],
                "metal_address_control_max_rss_bytes": [
                    sample["bytecode_address"]["control_max_rss_bytes"]
                    for sample in attributions
                    if sample["bytecode_address"]["control_max_rss_bytes"] is not None
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
                "instruction_read_raf_metal_scatter_threads": args.instruction_read_raf_metal_scatter_threads,
                "bytecode_metal_message_threads": args.bytecode_metal_message_threads,
                "bytecode_metal_transition_threads": args.bytecode_metal_transition_threads,
                "bytecode_metal_max_threadgroups": args.bytecode_metal_max_threadgroups,
                "bytecode_metal_cutoff_log2": args.bytecode_metal_cutoff_log2,
                "bytecode_metal_cutoff_elements": 1 << args.bytecode_metal_cutoff_log2,
                "bytecode_metal_trace_cutoff_log2": args.bytecode_metal_trace_cutoff_log2,
                "bytecode_metal_trace_cutoff_elements": 1
                << args.bytecode_metal_trace_cutoff_log2,
                "bytecode_cpu_tail_algebra": "q10",
                "bytecode_address_metal_implementation": (
                    args.bytecode_address_metal_implementation
                ),
                "bytecode_address_metal_outer_tiles": (
                    args.bytecode_address_metal_outer_tiles
                ),
                "bytecode_address_metal_trace_cutoff_log2": (
                    args.bytecode_address_metal_trace_cutoff_log2
                ),
                "bytecode_address_metal_trace_cutoff_elements": 1
                << args.bytecode_address_metal_trace_cutoff_log2,
                "bytecode_address_worker_variant": "packed4_halfwidth_v1",
                "bytecode_address_worker_simd_width": 32,
                "bytecode_address_worker_threads": 128,
                "bytecode_address_worker_items_per_threadgroup": 4,
                "bytecode_address_worker_threadgroup_bytes": 0,
                "bytecode_address_reducer_threads": 256,
                "bytecode_address_producer_route": METAL_BYTECODE_ADDRESS_FUSED_ROUTE,
                "bytecode_address_producer_charge_model": (
                    "stage1_topology_plus_irraf_scatter_v1"
                ),
                "bytecode_address_max_admitted_descriptors_per_chunk": (
                    BYTECODE_ADDRESS_MAX_ADMITTED_DESCRIPTORS_PER_CHUNK
                ),
                "bytecode_address_max_admitted_pivots_per_chunk": (
                    BYTECODE_ADDRESS_MAX_ADMITTED_PIVOTS_PER_CHUNK
                ),
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
                "registers_claim_metal_implementation": (
                    args.registers_claim_metal_implementation
                ),
                "registers_claim_metal_trace_cutoff_log2": (
                    args.registers_claim_metal_trace_cutoff_log2
                ),
                "registers_claim_metal_trace_cutoff_elements": 1
                << args.registers_claim_metal_trace_cutoff_log2,
                "registers_claim_outer_carrier": (
                    args.registers_claim_metal_implementation
                    == "outer-carrier-alias-hybrid"
                ),
                "registers_claim_required_instruction_input_cutoff_log2": 1
                + args.log_n // 2,
                "ram_cycle_family_schema_version": RAM_CYCLE_FAMILY_SCHEMA_VERSION,
                "ram_cycle_family_source_kind": "ram_access_tape_v1",
                "ram_hamming_sparse_product_cap": RAM_HAMMING_PRODUCT_CAP,
                "ram_hamming_charge_model": "raw_member_plus_owner_prepare_v1",
                "ram_read_write_charge_model": "raw_member_plus_owner_prepare_v1",
                "ram_cycle_family_charge_model": RAM_CYCLE_FAMILY_CHARGE_MODEL,
                "ram_cycle_family_members": [
                    kernel for _, kernel in RAM_CYCLE_FAMILY_MEMBERS
                ],
                "ram_standalone_charged_metrics_additive": False,
                "registers_claim_carrier_owned_bytes": (
                    outer_remainder_storage_geometry(
                        args.log_n,
                        args.product_uniskip_outer_carrier,
                        True,
                    )["registers_claim_carrier"]["owned_bytes"]
                    if args.registers_claim_metal_implementation
                    == "outer-carrier-alias-hybrid"
                    else 0
                ),
                "span": PIOP_SPAN,
                "orders": orders,
                "bytecode_address_producer_orders": producer_orders,
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
