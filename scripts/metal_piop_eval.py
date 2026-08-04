#!/usr/bin/env python3
"""Measure optimized-CPU and Metal-hybrid Akita PIOP wall time."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
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


SCHEMA_VERSION = 4
PIOP_SPAN = "jolt_prover::piop"
BACKEND_WITNESS_PREP_SPAN = "jolt_prover::backend_witness_prepare"


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


def summarize_pairs(pairs: list[dict[str, float]]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("at least one CPU/Metal pair is required")
    cpu = [float(pair["cpu_us"]) for pair in pairs]
    metal = [float(pair["metal_us"]) for pair in pairs]
    cpu_prepare = [float(pair["cpu_prepare_us"]) for pair in pairs]
    metal_prepare = [float(pair["metal_prepare_us"]) for pair in pairs]
    cpu_instruction_ra = [float(pair["cpu_instruction_ra_us"]) for pair in pairs]
    metal_instruction_ra = [float(pair["metal_instruction_ra_us"]) for pair in pairs]
    if any(not math.isfinite(value) or value <= 0.0 for value in cpu + metal):
        raise ValueError("PIOP durations must be finite and positive")
    if any(not math.isfinite(value) or value < 0.0 for value in cpu_prepare + metal_prepare):
        raise ValueError("backend witness preparation durations must be finite and non-negative")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in cpu_instruction_ra + metal_instruction_ra
    ):
        raise ValueError("Instruction RA durations must be finite and positive")
    paired_speedups = [cpu_us / metal_us for cpu_us, metal_us in zip(cpu, metal)]
    instruction_ra_speedups = [
        cpu_us / metal_us for cpu_us, metal_us in zip(cpu_instruction_ra, metal_instruction_ra)
    ]
    paired_with_prepare = [
        (cpu_us + cpu_prepare_us) / (metal_us + metal_prepare_us)
        for cpu_us, metal_us, cpu_prepare_us, metal_prepare_us in zip(
            cpu, metal, cpu_prepare, metal_prepare
        )
    ]
    return {
        "piop_speedup": statistics.median(paired_speedups),
        "instruction_ra_speedup": statistics.median(instruction_ra_speedups),
        "piop_plus_backend_witness_prepare_speedup": statistics.median(paired_with_prepare),
        "cpu_piop_ms": statistics.median(cpu) / 1000.0,
        "metal_piop_ms": statistics.median(metal) / 1000.0,
        "cpu_backend_witness_prepare_ms": statistics.median(cpu_prepare) / 1000.0,
        "metal_backend_witness_prepare_ms": statistics.median(metal_prepare) / 1000.0,
        "paired_speedups": paired_speedups,
        "paired_instruction_ra_speedups": instruction_ra_speedups,
        "paired_speedups_with_backend_witness_prepare": paired_with_prepare,
        "cpu_piop_ms_samples": [value / 1000.0 for value in cpu],
        "metal_piop_ms_samples": [value / 1000.0 for value in metal],
        "cpu_backend_witness_prepare_ms_samples": [value / 1000.0 for value in cpu_prepare],
        "metal_backend_witness_prepare_ms_samples": [value / 1000.0 for value in metal_prepare],
    }


def trace_path(root: Path, workload: str, log_n: int, backend: str) -> Path:
    name = workload.replace("-", "_")
    return root / "benchmark-runs" / "perfetto_traces" / f"akita_{name}_{log_n}_{backend}.json"


def run_backend(
    root: Path,
    artifact_dir: Path,
    workload: str,
    log_n: int,
    backend: str,
    instruction_ra_materialize_width: int,
    instruction_ra_reuse_inverse: bool,
    pair_index: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "-p",
        "jolt-prover",
        "--example",
        "modular_benchmark",
        "--features",
        "metal,prover-fixtures",
        "--",
        "--name",
        workload,
        "--scale",
        str(log_n),
        "--format",
        "chrome",
        "--backend",
        backend,
    ]
    if backend == "metal":
        command.extend(
            [
                "--instruction-ra-materialize-width",
                f"w{instruction_ra_materialize_width}",
            ]
        )
        if instruction_ra_reuse_inverse:
            command.append("--instruction-ra-reuse-inverse")
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
    source = trace_path(root, workload, log_n, backend)
    if not source.is_file() or source.stat().st_mtime_ns < started_ns:
        raise ValueError(f"{backend} evaluator did not emit a fresh trace")
    destination = artifact_dir / f"{label}.trace.json"
    shutil.copy2(source, destination)
    events = load_trace_events(destination)
    return {
        "piop_us": unique_span_duration_us(events),
        "backend_witness_prepare_us": unique_named_span_duration_us(
            events, BACKEND_WITNESS_PREP_SPAN
        ),
        "attribution": trace_attribution(events),
    }


def default_artifact_dir(root: Path) -> Path:
    configured = os.environ.get("JOLT_AUTORESEARCH_EVAL_DIR")
    if configured:
        return Path(configured).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
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
    result.add_argument("--trace", type=Path)
    return result


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
    if args.instruction_ra_reuse_inverse and args.instruction_ra_materialize_width == 16:
        print("error: width-16 Instruction RA cannot reuse the inverse", file=sys.stderr)
        return 2
    root = Path(args.root).resolve()
    source = source_fingerprint(root)
    artifact_dir = default_artifact_dir(root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    orders = []
    attributions = []
    try:
        for index in range(args.repeats):
            order = ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
            orders.append(order)
            results: dict[str, dict[str, Any]] = {}
            for backend in order:
                results[backend] = run_backend(
                    root,
                    artifact_dir,
                    args.workload,
                    args.log_n,
                    backend,
                    args.instruction_ra_materialize_width,
                    args.instruction_ra_reuse_inverse,
                    index + 1,
                    args.timeout_seconds,
                )
            pairs.append(
                {
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
                }
            )
            attributions.append(
                {
                    "optimized": results["optimized"]["attribution"],
                    "metal": results["metal"]["attribution"],
                }
            )
        metrics = summarize_pairs(pairs)
        if source_fingerprint(root) != source:
            raise ValueError("source worktree changed during the paired evaluation")
        output = {
            "schema_version": SCHEMA_VERSION,
            "kernel": "akita_piop",
            "metrics": metrics,
            "attribution_samples": attributions,
            "guards": {
                "cpu_proofs_verified": True,
                "metal_proofs_verified": True,
                "unique_piop_span": True,
                "unique_backend_witness_prepare_span": True,
                "target_scale": args.log_n >= 26,
            },
            "resources": {
                "gpu_seconds": sum(pair["metal_us"] for pair in pairs) / 1_000_000.0,
            },
            "fingerprint": {
                **source,
                "machine": platform.machine(),
                "platform": platform.platform(),
                "workload": args.workload,
                "log_n": args.log_n,
                "instruction_ra_materialize_width": args.instruction_ra_materialize_width,
                "instruction_ra_reuse_inverse": args.instruction_ra_reuse_inverse,
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
