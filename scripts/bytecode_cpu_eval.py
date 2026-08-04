#!/usr/bin/env python3
"""Freeze and select the optimized CPU algebra for Akita BytecodeReadRafCycle."""

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
    from metal_piop_eval import source_fingerprint, trace_path
except ModuleNotFoundError:
    from scripts.metal_autoresearch import evaluator_lock
    from scripts.metal_piop_eval import source_fingerprint, trace_path


SCHEMA_VERSION = 1
EVALUATOR = "bytecode_cpu_eval"
KERNEL = "BytecodeReadRafCycle"
WORKLOAD = "fibonacci"
SMOKE_LOG_N = 22
TARGET_LOG_N = 26
FEATURES = "akita,prover-fixtures"
PIOP_SPAN = "jolt_prover::piop"
BACKEND_PREP_SPAN = "jolt_prover::backend_witness_prepare"
ARMS = ("generic", "q10", "q10-accum")
ORDERS = (
    ("generic", "q10", "q10-accum"),
    ("q10", "q10-accum", "generic"),
    ("q10-accum", "generic", "q10"),
    ("generic", "q10-accum", "q10"),
    ("q10", "generic", "q10-accum"),
)
COMPONENTS = ("prepare", "prove_round", "finish_rounds", "output_claims")
DURATION_COMPONENTS = ("prepare", "rounds_total", "finish", "output_claims")
OBSERVATION_GUARDS = (
    "exit_zero",
    "proof_verified",
    "unique_config",
    "effective_geometry",
    "fresh_trace",
    "span_cardinality",
    "piop_containment",
    "ordered_nonoverlap",
    "finite_positive",
    "stable_source",
    "stable_binary",
)
MIN_Q10_VS_GENERIC = 0.05
MIN_Q10_ACCUM_VS_Q10 = 0.03
DEFAULT_TIMEOUT_SECONDS = 7200
CONFIG_RE = re.compile(
    r"^BYTECODE_CYCLE_CONFIG "
    r"requested=(?P<requested>\S+) effective=(?P<effective>\S+) "
    r"log_t=(?P<log_t>\d+) log_k=(?P<log_k>\d+) "
    r"chunk_bits=(?P<chunk_bits>\d+) num_ra=(?P<num_ra>\d+) "
    r"degree=(?P<degree>\d+)$",
    re.MULTILINE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value))
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("ab") as output:
        output.write(canonical_json(value))
        output.flush()
        os.fsync(output.fileno())


def median_mad(values: list[float]) -> tuple[float, float]:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("timings must be finite and positive")
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return median, mad


def parse_config(stdout: str) -> dict[str, Any]:
    matches = list(CONFIG_RE.finditer(stdout))
    if len(matches) != 1:
        raise ValueError("benchmark output must contain exactly one bytecode config record")
    fields = matches[0].groupdict()
    return {
        "requested": fields["requested"],
        "effective": fields["effective"],
        **{
            name: int(fields[name])
            for name in ("log_t", "log_k", "chunk_bits", "num_ra", "degree")
        },
    }


def strict_named_intervals(
    events: list[dict[str, Any]], names: set[str]
) -> dict[str, list[tuple[float, float]]]:
    intervals = {name: [] for name in names}
    stacks: dict[tuple[Any, Any, str], list[float]] = {}
    for event in events:
        name = event.get("name")
        if name not in names:
            continue
        phase = event.get("ph")
        try:
            timestamp = float(event["ts"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name} has an invalid timestamp") from error
        if not math.isfinite(timestamp):
            raise ValueError(f"{name} has a non-finite timestamp")
        if phase == "X":
            try:
                duration = float(event["dur"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{name} has an invalid duration") from error
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError(f"{name} has a non-positive duration")
            intervals[name].append((timestamp, timestamp + duration))
        elif phase == "B":
            key = (event.get("pid"), event.get("tid"), name)
            stacks.setdefault(key, []).append(timestamp)
        elif phase == "E":
            key = (event.get("pid"), event.get("tid"), name)
            starts = stacks.get(key)
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


def member_breakdown(events: list[dict[str, Any]], rounds: int) -> dict[str, Any]:
    member_names = {f"{KERNEL}::{component}" for component in COMPONENTS}
    names = member_names | {PIOP_SPAN, BACKEND_PREP_SPAN}
    intervals = strict_named_intervals(events, names)
    if len(intervals[PIOP_SPAN]) != 1:
        raise ValueError("trace must contain exactly one positive PIOP span")
    if len(intervals[BACKEND_PREP_SPAN]) != 1:
        raise ValueError("trace must contain exactly one positive backend-prepare span")
    piop_start, piop_end = intervals[PIOP_SPAN][0]

    expected_counts = {
        "prepare": 1,
        "prove_round": rounds,
        "finish_rounds": 1,
        "output_claims": 1,
    }
    by_component = {
        component: sorted(intervals[f"{KERNEL}::{component}"])
        for component in COMPONENTS
    }
    actual_counts = {component: len(values) for component, values in by_component.items()}
    if actual_counts != expected_counts:
        raise ValueError("trace has unexpected bytecode member span counts")
    member_intervals = [interval for values in by_component.values() for interval in values]
    if any(start < piop_start or end > piop_end for start, end in member_intervals):
        raise ValueError("a bytecode member span lies outside PIOP")

    prepare = by_component["prepare"][0]
    round_intervals = by_component["prove_round"]
    finish = by_component["finish_rounds"][0]
    output = by_component["output_claims"][0]
    ordered = [prepare, *round_intervals, finish, output]
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("bytecode member spans overlap or appear out of order")

    duration = lambda interval: interval[1] - interval[0]
    round_durations = [duration(interval) for interval in round_intervals]
    durations = {
        "prepare": duration(prepare),
        "rounds": round_durations,
        "rounds_total": sum(round_durations),
        "finish": duration(finish),
        "output_claims": duration(output),
        "piop": piop_end - piop_start,
        "backend_witness_prepare": duration(intervals[BACKEND_PREP_SPAN][0]),
    }
    durations["member"] = (
        durations["prepare"]
        + durations["rounds_total"]
        + durations["finish"]
        + durations["output_claims"]
    )
    scalar_durations = [value for key, value in durations.items() if key != "rounds"]
    if any(not math.isfinite(value) or value <= 0.0 for value in scalar_durations):
        raise ValueError("trace contains a non-positive member duration")
    return {"durations_us": durations, "occurrences": actual_counts}


def fractional_improvements(
    observations: list[dict[str, Any]], baseline: str, candidate: str
) -> list[float]:
    return [
        (float(block[baseline]["member_wall_us"]) - float(block[candidate]["member_wall_us"]))
        / float(block[baseline]["member_wall_us"])
        for block in observations
    ]


def comparison(
    observations: list[dict[str, Any]], baseline: str, candidate: str, minimum: float
) -> dict[str, Any]:
    improvements = fractional_improvements(observations, baseline, candidate)
    speedups = [
        float(block[baseline]["member_wall_us"]) / float(block[candidate]["member_wall_us"])
        for block in observations
    ]
    median = statistics.median(improvements)
    mad = statistics.median(abs(value - median) for value in improvements)
    baseline_median = statistics.median(
        float(block[baseline]["member_wall_us"]) for block in observations
    )
    candidate_median = statistics.median(
        float(block[candidate]["member_wall_us"]) for block in observations
    )
    clears_minimum = median >= minimum
    clears_noise = median > 3.0 * mad
    lower_median = candidate_median < baseline_median
    return {
        "baseline": baseline,
        "candidate": candidate,
        "paired_fractional_improvements": improvements,
        "paired_speedups": speedups,
        "median_fractional_improvement": median,
        "mad_fractional_improvement": mad,
        "minimum_fractional_improvement": minimum,
        "clears_minimum": clears_minimum,
        "clears_noise": clears_noise,
        "lower_unpaired_median": lower_median,
        "clears": clears_minimum and clears_noise and lower_median,
    }


def summarize(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if len(observations) != len(ORDERS):
        raise ValueError("the verdict requires all five target blocks")
    arms = {}
    for arm in ARMS:
        member = [float(block[arm]["member_wall_us"]) for block in observations]
        piop = [float(block[arm]["piop_us"]) for block in observations]
        member_median, member_mad = median_mad(member)
        piop_median, piop_mad = median_mad(piop)
        arms[arm] = {
            "member_wall_ms_samples": [value / 1000.0 for value in member],
            "member_wall_ms_median": member_median / 1000.0,
            "member_wall_ms_mad": member_mad / 1000.0,
            "member_wall_relative_mad": member_mad / member_median,
            "piop_ms_samples": [value / 1000.0 for value in piop],
            "piop_ms_median": piop_median / 1000.0,
            "piop_ms_mad": piop_mad / 1000.0,
        }
    q10 = comparison(observations, "generic", "q10", MIN_Q10_VS_GENERIC)
    q10_accum = comparison(observations, "generic", "q10-accum", MIN_Q10_VS_GENERIC)
    accum_over_q10 = comparison(
        observations, "q10", "q10-accum", MIN_Q10_ACCUM_VS_Q10
    )
    if q10_accum["clears"] and accum_over_q10["clears"]:
        selected = "q10-accum"
    elif q10["clears"]:
        selected = "q10"
    else:
        selected = "generic"
    return {
        "arms": arms,
        "comparisons": {
            "q10_vs_generic": q10,
            "q10_accum_vs_generic": q10_accum,
            "q10_accum_vs_q10": accum_over_q10,
        },
        "selected": selected,
    }


def component_summary(observations: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in ARMS:
        arm_observations = [observation for observation in observations if observation["arm"] == arm]
        if not arm_observations:
            continue
        summary = {}
        for component in DURATION_COMPONENTS:
            samples = [
                float(observation["durations_us"][component])
                for observation in arm_observations
            ]
            median, mad = median_mad(samples)
            summary[f"{component}_ms_samples"] = [sample / 1000.0 for sample in samples]
            summary[f"{component}_ms_median"] = median / 1000.0
            summary[f"{component}_ms_mad"] = mad / 1000.0
        result[arm] = summary
    return result


def schedule() -> list[dict[str, Any]]:
    cells = [
        {
            "id": f"smoke-p{position:02d}-{arm}",
            "phase": "smoke",
            "block": None,
            "position": position,
            "arm": arm,
            "scale": SMOKE_LOG_N,
            "excluded_from_verdict": True,
        }
        for position, arm in enumerate(ARMS, start=1)
    ]
    for block, order in enumerate(ORDERS, start=1):
        cells.extend(
            {
                "id": f"primary-b{block:02d}-p{position:02d}-{arm}",
                "phase": "primary",
                "block": block,
                "position": position,
                "arm": arm,
                "scale": TARGET_LOG_N,
                "excluded_from_verdict": False,
            }
            for position, arm in enumerate(order, start=1)
        )
    return cells


def run_contract(timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    return {
        "question": (
            "Which exact optimized CPU algebra minimizes BytecodeReadRafCycle member wall "
            "time at the production 2^26 Fibonacci geometry?"
        ),
        "hypothesis": (
            "Recovering each quadratic factor grid from three products reduces the target "
            "round sequence enough for Q10, with Q10Accum winning only if deferred terminal "
            "reductions add at least three percent."
        ),
        "primary_metric": {
            "name": "member_wall_us",
            "direction": "minimize",
            "unit": "microseconds",
            "formula": "prepare + sum(prove_round) + finish_rounds + output_claims",
            "parser": "strict Chrome trace interval reconstruction",
        },
        "guard_metrics": [
            "proof verifies exactly once",
            "requested and effective geometry match",
            "trace is fresh with exact positive ordered span cardinality",
            "source and evaluator binary remain unchanged",
        ],
        "minimum_effects": {
            "q10_vs_generic": MIN_Q10_VS_GENERIC,
            "q10_accum_vs_generic": MIN_Q10_VS_GENERIC,
            "q10_accum_vs_q10": MIN_Q10_ACCUM_VS_Q10,
            "noise_gate": "paired median fractional improvement > 3 * paired MAD",
        },
        "workload": {
            "name": WORKLOAD,
            "smoke_log_t": SMOKE_LOG_N,
            "target_log_t": TARGET_LOG_N,
            "backend": "optimized",
            "features": FEATURES.split(","),
        },
        "schedule": schedule(),
        "budget": {
            "builds": 1,
            "observations": len(schedule()),
            "target_blocks": len(ORDERS),
            "timeout_seconds_per_process": timeout_seconds,
            "external_service_cost": 0,
        },
        "selection_policy": (
            "Select Q10Accum only when it clears Generic and its extra minimum over Q10; "
            "otherwise select a clearing Q10, else retain Generic."
        ),
        "falsifying_outcome": (
            "Neither candidate clears its fixed effect and paired-noise gates while all "
            "correctness and provenance guards pass."
        ),
        "target_evidence_stage": "revalidated",
        "editable_paths": [],
        "stopping_conditions": [
            "all scheduled observations complete",
            "any failed correctness or provenance guard",
            "any timeout or nonzero exit",
            "source, binary, manifest, or evaluator drift",
            "interruption inside a three-arm block",
        ],
    }


def expected_config(cell: dict[str, Any]) -> dict[str, Any]:
    if cell["phase"] == "smoke":
        return {
            "requested": cell["arm"],
            "effective": "generic",
            "log_t": SMOKE_LOG_N,
            "log_k": 13,
            "chunk_bits": 4,
            "num_ra": 4,
            "degree": 6,
        }
    return {
        "requested": cell["arm"],
        "effective": cell["arm"],
        "log_t": TARGET_LOG_N,
        "log_k": 13,
        "chunk_bits": 8,
        "num_ra": 2,
        "degree": 4,
    }


def stat_tuple(path: Path) -> Optional[tuple[int, int, int]]:
    if not path.exists():
        return None
    value = path.stat()
    return value.st_ino, value.st_size, value.st_mtime_ns


def command_for(binary: Path, cell: dict[str, Any]) -> list[str]:
    return [
        str(binary),
        "--name",
        WORKLOAD,
        "--scale",
        str(cell["scale"]),
        "--format",
        "chrome",
        "--backend",
        "optimized",
        "--bytecode-cycle-algebra",
        cell["arm"],
    ]


def artifact_record(run_dir: Path, path: Path) -> dict[str, str]:
    return {"path": os.path.relpath(path, run_dir), "sha256": file_sha256(path)}


def run_observation(
    root: Path,
    run_dir: Path,
    binary: Path,
    cell: dict[str, Any],
    source: dict[str, Any],
    threads: int,
    timeout_seconds: int,
    attempt: int,
) -> dict[str, Any]:
    label = f"{cell['id']}-attempt-{attempt:02d}"
    stdout_path = run_dir / f"{label}.stdout"
    stderr_path = run_dir / f"{label}.stderr"
    trace_artifact = run_dir / f"{label}.trace.json"
    source_trace = trace_path(root, WORKLOAD, cell["scale"], "optimized")
    before_trace = stat_tuple(source_trace)
    command = command_for(binary, cell)
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(threads)
    started_ns = time.time_ns()
    started_at = utc_now()
    record = {
        **cell,
        "attempt": attempt,
        "status": "invalid",
        "command": command,
        "started_ns": started_ns,
        "started_at": started_at,
        "finished_ns": None,
        "finished_at": None,
        "exit_code": None,
        "config": None,
        "durations_us": None,
        "guards": {name: False for name in OBSERVATION_GUARDS},
        "artifacts": {},
        "error": None,
    }
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        record["exit_code"] = completed.returncode
        if completed.returncode != 0:
            raise ValueError(f"evaluator exited with status {completed.returncode}")
        record["guards"]["exit_zero"] = True

        if proof_completion_count(completed.stdout, cell) != 1:
            raise ValueError("benchmark output has no unique post-verification completion line")
        record["guards"]["proof_verified"] = True
        config = parse_config(completed.stdout)
        record["config"] = config
        record["guards"]["unique_config"] = True
        if config != expected_config(cell):
            raise ValueError("benchmark reported the wrong effective algebra or geometry")
        record["guards"]["effective_geometry"] = True

        after_trace = stat_tuple(source_trace)
        if (
            after_trace is None
            or after_trace == before_trace
            or source_trace.stat().st_mtime_ns < started_ns
            or not source_trace.is_file()
        ):
            raise ValueError("benchmark did not emit a fresh regular trace")
        shutil.copy2(source_trace, trace_artifact)
        record["guards"]["fresh_trace"] = True
        events = json.loads(trace_artifact.read_text())
        if not isinstance(events, list):
            raise ValueError("trace root must be an event array")
        breakdown = member_breakdown(events, cell["scale"])
        record["durations_us"] = breakdown["durations_us"]
        record["span_occurrences"] = breakdown["occurrences"]
        record["guards"]["span_cardinality"] = True
        record["guards"]["piop_containment"] = True
        record["guards"]["ordered_nonoverlap"] = True
        record["guards"]["finite_positive"] = True

        expected_source = {
            key: value for key, value in source.items() if key != "binary_sha256"
        }
        if source_fingerprint(root) != expected_source:
            raise ValueError("source worktree changed during observation")
        record["guards"]["stable_source"] = True
        if file_sha256(binary) != source["binary_sha256"]:
            raise ValueError("evaluator binary changed during observation")
        record["guards"]["stable_binary"] = True
        record["status"] = "valid"
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        stdout_path.write_text(stdout.decode() if isinstance(stdout, bytes) else stdout)
        stderr_path.write_text(stderr.decode() if isinstance(stderr, bytes) else stderr)
        record["error"] = "evaluator timed out"
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if not stdout_path.exists():
            stdout_path.write_text("")
        if not stderr_path.exists():
            stderr_path.write_text("")
        record["error"] = str(error)
    finally:
        record["finished_ns"] = time.time_ns()
        record["finished_at"] = utc_now()
        record["artifacts"]["stdout"] = artifact_record(run_dir, stdout_path)
        record["artifacts"]["stderr"] = artifact_record(run_dir, stderr_path)
        if trace_artifact.exists():
            record["artifacts"]["trace"] = artifact_record(run_dir, trace_artifact)
    return record


def cpu_model() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return platform.processor()


def command_output(command: list[str], root: Path) -> str:
    return subprocess.run(
        command, cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


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
    root: Path, run_dir: Path, timeout_seconds: int
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
    (run_dir / "build.stdout").write_text(completed.stdout)
    (run_dir / "build.stderr").write_text(completed.stderr)
    if completed.returncode != 0:
        raise ValueError(f"canonical evaluator build exited with status {completed.returncode}")
    binary = root / "target" / "release" / "examples" / "modular_benchmark"
    if not binary.is_file():
        raise ValueError("canonical evaluator binary is missing")
    return binary, command


def default_run_dir(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return root / "benchmark-runs" / "bytecode-cpu-eval" / timestamp


def load_observations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def proof_completion_count(stdout: str, cell: dict[str, Any]) -> int:
    return len(
        re.findall(
            rf"^modular Akita {WORKLOAD} \(2\^{cell['scale']}, optimized\): Prover completed .+$",
            stdout,
            re.MULTILINE,
        )
    )


def resolved_artifact(run_dir: Path, artifact: dict[str, str]) -> Path:
    path = (run_dir / artifact["path"]).resolve()
    if run_dir != path and run_dir not in path.parents:
        raise ValueError("observation artifact escapes its run directory")
    if not path.is_file() or file_sha256(path) != artifact["sha256"]:
        raise ValueError("observation artifact is missing or has the wrong hash")
    return path


def validate_completed_observation(
    run_dir: Path, binary: Path, cell: dict[str, Any], observation: dict[str, Any]
) -> None:
    for key in (
        "id",
        "phase",
        "block",
        "position",
        "arm",
        "scale",
        "excluded_from_verdict",
    ):
        if observation.get(key) != cell[key]:
            raise ValueError("completed observation does not match the frozen schedule")
    if observation.get("status") != "valid" or observation.get("error") is not None:
        raise ValueError("run contains an invalid observation")
    guards = observation.get("guards")
    if (
        not isinstance(guards, dict)
        or set(guards) != set(OBSERVATION_GUARDS)
        or not all(guards.values())
    ):
        raise ValueError("completed observation has a failed guard")
    if observation.get("exit_code") != 0:
        raise ValueError("completed observation did not exit successfully")
    if observation.get("command") != command_for(binary, cell):
        raise ValueError("completed observation used the wrong command")
    if observation.get("config") != expected_config(cell):
        raise ValueError("completed observation has the wrong geometry")
    artifacts = observation.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"stdout", "stderr", "trace"}:
        raise ValueError("completed observation has an incomplete artifact set")
    stdout = resolved_artifact(run_dir, artifacts["stdout"]).read_text()
    _ = resolved_artifact(run_dir, artifacts["stderr"])
    trace = resolved_artifact(run_dir, artifacts["trace"])
    if proof_completion_count(stdout, cell) != 1 or parse_config(stdout) != expected_config(cell):
        raise ValueError("completed observation stdout does not prove its recorded run")
    events = json.loads(trace.read_text())
    if not isinstance(events, list):
        raise ValueError("completed observation trace root is not an event array")
    breakdown = member_breakdown(events, cell["scale"])
    if observation.get("durations_us") != breakdown["durations_us"]:
        raise ValueError("completed observation durations disagree with its trace")
    if observation.get("span_occurrences") != breakdown["occurrences"]:
        raise ValueError("completed observation span counts disagree with its trace")


def validate_resume_prefix(
    run_dir: Path,
    binary: Path,
    observations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    frozen_schedule = schedule()
    allowed_lengths = {0, len(ARMS)} | {
        len(ARMS) + len(ARMS) * blocks for blocks in range(1, len(ORDERS) + 1)
    }
    if len(observations) not in allowed_lengths:
        raise ValueError("resume is allowed only at a completed smoke or target-block boundary")
    completed = {}
    for cell, observation in zip(frozen_schedule, observations):
        if not isinstance(observation, dict):
            raise ValueError("observation ledger contains a non-object record")
        validate_completed_observation(run_dir, binary, cell, observation)
        completed[cell["id"]] = observation
    if len(observations) < len(frozen_schedule):
        next_id = frozen_schedule[len(observations)]["id"]
        if list(run_dir.glob(f"{next_id}-attempt-*")):
            raise ValueError("run contains orphaned artifacts for an incomplete observation")
    return completed


def binary_is_running(binary: Path) -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", str(binary)], capture_output=True, text=True
        )
    except OSError as error:
        raise ValueError("could not inspect evaluator process state") from error
    if result.returncode == 0:
        return bool(result.stdout.strip())
    if result.returncode == 1:
        return False
    raise ValueError("could not inspect evaluator process state")


def run_identity(
    root: Path,
    run_dir: Path,
    binary: Path,
    initial_source: dict[str, Any],
    threads: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    source = {**initial_source, "binary_sha256": file_sha256(binary)}
    fixed_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("CARGO", "RUST", "RAYON"))
    }
    fixed_environment["RAYON_NUM_THREADS"] = str(threads)
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator": EVALUATOR,
        "run_dir": str(run_dir),
        "source": source,
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        "cargo_lock_sha256": file_sha256(root / "Cargo.lock"),
        "binary": os.path.relpath(binary, root),
        "build_command": canonical_build_command(),
        "command_template": command_for(binary, schedule()[-1]),
        "rustc_vv": command_output(["rustc", "-Vv"], root),
        "cargo_version": command_output(["cargo", "-V"], root),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "rayon_threads": threads,
        "environment": fixed_environment,
        "timeout_seconds": timeout_seconds,
        "schedule": schedule(),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", default=Path(__file__).resolve().parents[1])
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.timeout_seconds < 1:
        print("error: timeout must be positive", file=sys.stderr)
        return 2
    root = Path(args.root).resolve()
    run_dir = (args.run_dir or default_run_dir(root)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    observations_path = run_dir / "observations.jsonl"
    try:
        initial_source = source_fingerprint(root)
        if initial_source["worktree_dirty"]:
            raise ValueError("canonical CPU denominator requires a clean source worktree")
        threads = os.cpu_count() or 1
        manifest_path = run_dir / "run.json"
        digest_path = run_dir / "run.sha256"
        binary = root / "target" / "release" / "examples" / "modular_benchmark"
        contract = run_contract(args.timeout_seconds)
        if manifest_path.exists():
            manifest_bytes = manifest_path.read_bytes()
            if hashlib.sha256(manifest_bytes).hexdigest() != digest_path.read_text().strip():
                raise ValueError("run manifest changed after initialization")
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict) or not isinstance(manifest.get("identity"), dict):
                raise ValueError("run manifest is not an object with an identity")
            if manifest.get("contract") != contract:
                raise ValueError("resume contract does not match the frozen run")
            if not binary.is_file():
                raise ValueError("resume binary is missing; start a new canonical run")
            identity = run_identity(
                root, run_dir, binary, initial_source, threads, args.timeout_seconds
            )
            if manifest["identity"] != identity:
                raise ValueError("resume identity does not match the frozen run")
        else:
            if any(run_dir.iterdir()):
                raise ValueError("new run directory must be empty")
            binary, _ = build_binary(root, run_dir, args.timeout_seconds)
            if source_fingerprint(root) != initial_source:
                raise ValueError("source worktree changed during evaluator build")
            identity = run_identity(
                root, run_dir, binary, initial_source, threads, args.timeout_seconds
            )
            manifest = {"contract": contract, "identity": identity, "started_at": utc_now()}
            manifest_bytes = canonical_json(manifest)
            manifest_path.write_bytes(manifest_bytes)
            digest_path.write_text(hashlib.sha256(manifest_bytes).hexdigest() + "\n")
            observations_path.touch()

        source = identity["source"]
        observations = load_observations(observations_path)
        completed = validate_resume_prefix(run_dir, binary, observations)
        if binary_is_running(binary):
            raise ValueError("another canonical evaluator binary is still running")

        for cell in schedule():
            if cell["id"] in completed:
                continue
            if source_fingerprint(root) != initial_source:
                raise ValueError("source worktree changed before observation")
            if file_sha256(binary) != source["binary_sha256"]:
                raise ValueError("evaluator binary changed before observation")
            attempts = len(list(run_dir.glob(f"{cell['id']}-attempt-*.stdout"))) + 1
            observation = run_observation(
                root,
                run_dir,
                binary,
                cell,
                source,
                threads,
                args.timeout_seconds,
                attempts,
            )
            append_jsonl(observations_path, observation)
            if observation["status"] != "valid":
                raise ValueError(f"observation {cell['id']} failed: {observation['error']}")
            completed[cell["id"]] = observation

        ordered = [completed[cell["id"]] for cell in schedule()]
        validate_resume_prefix(run_dir, binary, ordered)
        primary = ordered[len(ARMS) :]
        blocks = []
        for block in range(1, len(ORDERS) + 1):
            block_observations = {
                observation["arm"]: {
                    "member_wall_us": observation["durations_us"]["member"],
                    "piop_us": observation["durations_us"]["piop"],
                }
                for observation in primary
                if observation["block"] == block
            }
            if set(block_observations) != set(ARMS):
                raise ValueError("target block is incomplete")
            blocks.append(block_observations)
        metrics = summarize(blocks)
        metrics["components"] = component_summary(primary)
        selected = metrics["selected"]
        result = {
            "schema_version": SCHEMA_VERSION,
            "evaluator": EVALUATOR,
            "status": "complete",
            "evidence_stage": "revalidated",
            "target": {
                "workload": WORKLOAD,
                "log_t": TARGET_LOG_N,
                "trace_elements": 1 << TARGET_LOG_N,
                "backend": "optimized",
                "features": FEATURES.split(","),
                "profile": "release",
                "log_k": 13,
                "chunk_bits": 8,
                "chunk_k": 256,
                "num_ra": 2,
                "degree": 4,
            },
            "metric_boundary": {
                "formula": "prepare + sum(prove_round) + finish_rounds + output_claims",
                "host_fiat_shamir_included": False,
                "derived_points_and_final_batch_check_included": False,
            },
            "metrics": metrics,
            "verdict": {
                "selected_arm": selected,
                "q10": "promote" if selected == "q10" else "reject",
                "q10_accum": "promote" if selected == "q10-accum" else "reject",
            },
            "observations": ordered,
            "provenance": {**identity, "started_at": manifest["started_at"], "finished_at": utc_now()},
            "guards": {
                "all_18_proofs_verified": len(ordered) == len(schedule())
                and all(observation["guards"]["proof_verified"] for observation in ordered),
                "all_18_traces_valid": len(ordered) == len(schedule())
                and all(
                    observation["guards"][guard]
                    for observation in ordered
                    for guard in (
                        "fresh_trace",
                        "span_cardinality",
                        "piop_containment",
                        "ordered_nonoverlap",
                        "finite_positive",
                    )
                ),
                "source_clean_and_stable": source_fingerprint(root) == initial_source,
                "binary_stable": file_sha256(binary) == source["binary_sha256"],
            },
            "artifacts": {
                "directory": os.path.relpath(run_dir, root),
                "manifest": os.path.relpath(manifest_path, root),
                "observations_jsonl": os.path.relpath(observations_path, root),
            },
        }
        if not all(result["guards"].values()):
            raise ValueError("final provenance guards failed")
        write_atomic(run_dir / "result.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    with evaluator_lock({"direct_evaluator": EVALUATOR}):
        raise SystemExit(main())
