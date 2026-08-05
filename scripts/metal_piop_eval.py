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


SCHEMA_VERSION = 5
FEATURES = "metal,prover-fixtures"
PIOP_SPAN = "jolt_prover::piop"
BACKEND_WITNESS_PREP_SPAN = "jolt_prover::backend_witness_prepare"
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
BYTECODE_MIN_SPEEDUP = 4.0
INSTRUCTION_INPUT_KERNEL = "InstructionInput"
INSTRUCTION_INPUT_COMPONENTS = ("prepare", "prove_round", "finish_rounds", "output_claims")
INSTRUCTION_INPUT_METAL_PHASES = (
    "storage_prepare",
    "allocation_plan",
    "prepare",
    "first_message",
    "first_bind",
    "dense_round",
    "readback",
    "cpu_tail",
)
INSTRUCTION_INPUT_MIN_SPEEDUP = 4.0
OPTIMIZED_INSTRUCTION_INPUT_ROWS_PREPARE = "OptimizedInstructionInput::rows_prepare"
OPTIMIZED_INSTRUCTION_INPUT_ROWS_STAGE3_USE = (
    "OptimizedInstructionInput::rows_stage3_use"
)
METAL_INSTRUCTION_INPUT_ROWS_STAGE1_USE = (
    "MetalInstructionInput::resident_rows_stage1_use"
)
PRODUCTION_PAIRS = 5
LOCAL_KERNELS = {
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
    r"dense_transition_threads=(?P<dense_transition_threads>\d+)$"
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
) -> Optional[dict[str, int]]:
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
    config = {name: int(value) for name, value in configs[0].groupdict().items()}
    expected = {
        "trace_cutoff": 1 << trace_cutoff_log2,
        "cutoff": 1 << cutoff_log2,
        "native_message_threads": native_message_threads,
        "native_transition_threads": native_transition_threads,
        "dense_transition_threads": dense_transition_threads,
    }
    if config != expected:
        raise ValueError(f"unexpected InstructionInput Metal config: {config}")
    return config


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
    return records[0]["args"]


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


def instruction_input_member_breakdown(
    events: list[dict[str, Any]],
    backend: str,
    log_n: int,
    cutoff_log2: int = 16,
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
        METAL_INSTRUCTION_INPUT_ROWS_STAGE1_USE,
    }
    intervals = strict_named_intervals(
        events,
        outer_names
        | inner_names
        | lifecycle_names
        | {PIOP_SPAN, BACKEND_WITNESS_PREP_SPAN},
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
    if backend == "optimized":
        if any(inner_counts.values()):
            raise ValueError(
                "optimized trace unexpectedly contains InstructionInput Metal spans"
            )
        if (
            len(intervals[OPTIMIZED_INSTRUCTION_INPUT_ROWS_PREPARE]) != 1
            or len(intervals[OPTIMIZED_INSTRUCTION_INPUT_ROWS_STAGE3_USE]) != 1
            or intervals[METAL_INSTRUCTION_INPUT_ROWS_STAGE1_USE]
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
            or len(intervals[METAL_INSTRUCTION_INPUT_ROWS_STAGE1_USE]) != 1
        ):
            raise ValueError("Metal InstructionInput row lifecycle is incomplete")
        expected_inner_counts = {
            "storage_prepare": 1,
            "allocation_plan": 1,
            "prepare": 1,
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
        rows_stage1 = intervals[METAL_INSTRUCTION_INPUT_ROWS_STAGE1_USE][0]
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
        if rows_stage1[0] < piop[0] or rows_stage1[1] > prepare[0]:
            raise ValueError(
                "Metal InstructionInput stage-1 row use is outside its lifecycle"
            )
        require_contained(inner["prepare"][0], prepare, "Metal InstructionInput prepare")
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
            "host_tail_bytes",
            "resident_rows_storage_id",
            "resident_rows",
            "resident_row_bytes",
        }:
            raise ValueError("InstructionInput storage preparation has unexpected fields")
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
            or storage["resident_row_bytes"] != 160
        ):
            raise ValueError("InstructionInput storage preparation has invalid geometry")

        allocation = unique_span_args(
            events, f"Metal{INSTRUCTION_INPUT_KERNEL}::allocation_plan"
        )
        allocation_fields = {
            "device_buffers",
            "planned_device_bytes",
            "current_device_bytes",
            "recommended_device_bytes",
        }
        if set(allocation) != allocation_fields:
            raise ValueError("InstructionInput allocation plan has unexpected fields")
        allocation = {
            name: nonnegative_trace_integer(value, name)
            for name, value in allocation.items()
        }
        expected_sequence_bytes = instruction_input_sequence_storage_bytes(log_n)
        expected_resident_row_bytes = 160 * (1 << log_n)
        if (
            allocation["device_buffers"] != 6
            or allocation["planned_device_bytes"] != expected_sequence_bytes
            or allocation["current_device_bytes"] < expected_resident_row_bytes
        ):
            raise ValueError(
                "InstructionInput allocation plan has invalid buffer accounting"
            )
        if (
            allocation["current_device_bytes"] + allocation["planned_device_bytes"]
            > allocation["recommended_device_bytes"]
        ):
            raise ValueError(
                "InstructionInput allocation plan exceeds the admitted working set"
            )

        prepare_args = unique_span_args(
            events, f"Metal{INSTRUCTION_INPUT_KERNEL}::prepare"
        )
        if set(prepare_args) != {
            "resident_rows_reused",
            "round_device_buffer_allocations",
            "resident_rows_storage_id",
            "resident_rows",
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
        stage1_args = unique_span_args(
            events, METAL_INSTRUCTION_INPUT_ROWS_STAGE1_USE
        )
        if set(stage1_args) != {"resident_rows_storage_id", "resident_rows"}:
            raise ValueError("Metal InstructionInput row lifecycle is incomplete")
        stage1_storage_id = positive_trace_integer(
            stage1_args["resident_rows_storage_id"], "Metal stage-1 storage ID"
        )
        stage1_rows = positive_trace_integer(
            stage1_args["resident_rows"], "Metal stage-1 row count"
        )
        prepare_storage_id = storage["resident_rows_storage_id"]
        if (
            resident_rows_reused is not True
            or round_allocations != 0
            or stage1_rows != 1 << log_n
            or stage3_rows != stage1_rows
            or stage1_storage_id != prepare_storage_id
            or stage3_storage_id != prepare_storage_id
        ):
            raise ValueError(
                "InstructionInput Metal row lifecycle did not preserve residency"
            )
        row_lifecycle = {
            "kind": "metal_resident",
            "rows": stage1_rows,
            "row_bytes": storage["resident_row_bytes"],
            "prepare_storage_id": prepare_storage_id,
            "stage1_storage_id": stage1_storage_id,
            "stage3_storage_id": stage3_storage_id,
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
        "instruction_input_kernel_service_decision": instruction_input_decision,
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


def member_record(member: dict[str, Any]) -> dict[str, Any]:
    components = member["components"]
    rounds_ns = [microseconds_to_nanoseconds(value) for value in components["rounds_us"]]
    prepare_ns = microseconds_to_nanoseconds(components["prepare_us"])
    finish_ns = microseconds_to_nanoseconds(components["finish_us"])
    output_claims_ns = microseconds_to_nanoseconds(components["output_claims_us"])
    rounds_total_ns = sum(rounds_ns)
    return {
        "prepare_ns": prepare_ns,
        "rounds_ns": rounds_ns,
        "rounds_total_ns": rounds_total_ns,
        "finish_ns": finish_ns,
        "output_claims_ns": output_claims_ns,
        "member_ns": prepare_ns + rounds_total_ns + finish_ns + output_claims_ns,
        "outer_counts": member["outer_counts"],
        "metal_counts": member["metal_counts"],
        "resource_observation": member["resource_observation"],
    }


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
    instruction_input_native_message_threads: int,
    instruction_input_native_transition_threads: int,
    instruction_input_dense_transition_threads: int,
    instruction_input_cutoff_log2: int,
    instruction_input_trace_cutoff_log2: int,
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
    ]
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
    started_ns = time.time_ns()
    result = subprocess.run(
        command,
        cwd=root,
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
        events, backend, log_n, instruction_input_cutoff_log2
    )
    attribution = trace_attribution(events)
    for name, member in (
        (BYTECODE_KERNEL, bytecode_member),
        (INSTRUCTION_INPUT_KERNEL, instruction_input_member),
    ):
        if not math.isclose(
            kernel_wall_us(attribution, name),
            float(member["components"]["member_us"]),
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{name} member parser disagrees with trace attribution")
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
    result.add_argument("--trace", type=Path)
    return result


def validate_run_class(mode: str, workload: str, log_n: int, repeats: int) -> None:
    if mode == "production" and (
        workload != "fibonacci" or log_n != 26 or repeats != PRODUCTION_PAIRS
    ):
        raise ValueError(
            "production mode requires Fibonacci, log-n 26, and five pairs"
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
    if args.instruction_input_metal_cutoff_log2 > args.log_n - 1:
        print("error: InstructionInput cutoff must not exceed half the trace", file=sys.stderr)
        return 2
    if args.instruction_input_metal_trace_cutoff_log2 > args.log_n:
        print(
            "error: InstructionInput trace cutoff disables the measured backend",
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
                    args.instruction_input_metal_native_message_threads,
                    args.instruction_input_metal_native_transition_threads,
                    args.instruction_input_metal_dense_transition_threads,
                    args.instruction_input_metal_cutoff_log2,
                    args.instruction_input_metal_trace_cutoff_log2,
                    index + 1,
                    args.timeout_seconds,
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
                            "member_us"
                        ]
                    ),
                    "metal_instruction_input_us": float(
                        results["metal"]["instruction_input_member"]["components"][
                            "member_us"
                        ]
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
                                results[backend]["instruction_input_member"]
                            ),
                            "instruction_input_row_lifecycle": results[backend][
                                "instruction_input_member"
                            ]["row_lifecycle"],
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
                "target_scale": args.log_n == 26,
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
                        any(
                            kernel["kernel"] == local_kernel["name"]
                            for kernel in sample[backend]["kernels"]
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
                    == "metal_resident"
                    and sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "rows"
                    ]
                    == 1 << args.log_n
                    and sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "row_bytes"
                    ]
                    == 160
                    and sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "prepare_storage_id"
                    ]
                    == sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "stage1_storage_id"
                    ]
                    == sample["instruction_input"]["metal_member"]["row_lifecycle"][
                        "stage3_storage_id"
                    ]
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
                "instruction_input_local_gate": metrics[
                    "instruction_input_kernel_service_decision"
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
