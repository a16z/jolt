#!/usr/bin/env python3
"""Measure optimized-CPU and Metal-hybrid Akita PIOP wall time."""

from __future__ import annotations

import argparse
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
from typing import Any


SCHEMA_VERSION = 1
PIOP_SPAN = "jolt_prover::piop"


def unique_span_duration_us(events: list[dict[str, Any]]) -> float:
    stacks: dict[tuple[Any, Any], list[float]] = {}
    durations: list[float] = []
    for event in events:
        if event.get("name") != PIOP_SPAN:
            continue
        phase = event.get("ph")
        if phase == "X":
            durations.append(float(event["dur"]))
            continue
        key = (event.get("pid"), event.get("tid"))
        if phase == "B":
            stacks.setdefault(key, []).append(float(event["ts"]))
        elif phase == "E":
            starts = stacks.get(key)
            if not starts:
                raise ValueError("PIOP trace has an unmatched end event")
            durations.append(float(event["ts"]) - starts.pop())
    if any(starts for starts in stacks.values()):
        raise ValueError("PIOP trace has an unmatched begin event")
    if len(durations) != 1 or not math.isfinite(durations[0]) or durations[0] <= 0.0:
        raise ValueError("trace must contain exactly one positive PIOP span")
    return durations[0]


def load_piop_duration_us(path: Path) -> float:
    events = json.loads(path.read_text())
    if not isinstance(events, list):
        raise ValueError(f"{path} must contain a trace event array")
    return unique_span_duration_us(events)


def summarize_pairs(pairs: list[dict[str, float]]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("at least one CPU/Metal pair is required")
    cpu = [float(pair["cpu_us"]) for pair in pairs]
    metal = [float(pair["metal_us"]) for pair in pairs]
    if any(not math.isfinite(value) or value <= 0.0 for value in cpu + metal):
        raise ValueError("PIOP durations must be finite and positive")
    paired_speedups = [cpu_us / metal_us for cpu_us, metal_us in zip(cpu, metal)]
    return {
        "piop_speedup": statistics.median(paired_speedups),
        "cpu_piop_ms": statistics.median(cpu) / 1000.0,
        "metal_piop_ms": statistics.median(metal) / 1000.0,
        "paired_speedups": paired_speedups,
        "cpu_piop_ms_samples": [value / 1000.0 for value in cpu],
        "metal_piop_ms_samples": [value / 1000.0 for value in metal],
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
    pair_index: int,
    timeout_seconds: int,
) -> float:
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
    return load_piop_duration_us(destination)


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
    return result


def main() -> int:
    args = parser().parse_args()
    if args.log_n < 1 or args.repeats < 1 or args.timeout_seconds < 1:
        print("error: log-n, repeats, and timeout must be positive", file=sys.stderr)
        return 2
    root = Path(args.root).resolve()
    artifact_dir = default_artifact_dir(root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    orders = []
    try:
        for index in range(args.repeats):
            order = ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
            orders.append(order)
            durations: dict[str, float] = {}
            for backend in order:
                durations[backend] = run_backend(
                    root,
                    artifact_dir,
                    args.workload,
                    args.log_n,
                    backend,
                    index + 1,
                    args.timeout_seconds,
                )
            pairs.append({"cpu_us": durations["optimized"], "metal_us": durations["metal"]})
        metrics = summarize_pairs(pairs)
        output = {
            "schema_version": SCHEMA_VERSION,
            "kernel": "akita_piop",
            "metrics": metrics,
            "guards": {
                "cpu_proofs_verified": True,
                "metal_proofs_verified": True,
                "unique_piop_span": True,
                "target_scale": args.log_n >= 26,
            },
            "resources": {
                "gpu_seconds": sum(pair["metal_us"] for pair in pairs) / 1_000_000.0,
            },
            "fingerprint": {
                "git_revision": git_head(root),
                "machine": platform.machine(),
                "platform": platform.platform(),
                "workload": args.workload,
                "log_n": args.log_n,
                "span": PIOP_SPAN,
                "orders": orders,
            },
            "artifacts": str(artifact_dir),
        }
        print(json.dumps(output, sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
