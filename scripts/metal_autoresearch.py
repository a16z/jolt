#!/usr/bin/env python3
"""Run bounded Metal kernel experiments with snapshots and a durable ledger."""

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
from typing import Any


SCHEMA_VERSION = 1
VERDICTS = {"keep", "discard", "crash", "invalid"}


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


def parse_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema_version") == SCHEMA_VERSION:
            return value
    raise ValueError("evaluator stdout contains no schema-version 1 JSON object")


def validate_template(template: dict[str, Any]) -> None:
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
    candidate_repeats = template["candidate_repeats"]
    if candidate_repeats < 1 or candidate_repeats % 2 == 0:
        raise ValueError("candidate_repeats must be a positive odd integer")
    if template["baseline_repeats"] % candidate_repeats != 0:
        raise ValueError("baseline_repeats must be divisible by candidate_repeats")
    if template["budget"]["max_trials"] < 1:
        raise ValueError("max_trials must be positive")
    editable = set(template["scope"]["editable"])
    frozen = set(template["scope"]["frozen"])
    overlap = sorted(editable & frozen)
    if overlap:
        raise ValueError(f"paths cannot be editable and frozen: {overlap}")
    if template["portfolio_contract"] not in frozen:
        raise ValueError("the portfolio contract must be in the frozen path set")


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


def validate_params(config: dict[str, Any], params: dict[str, str]) -> None:
    search_space = config["search_space"]
    unknown = sorted(set(params) - set(search_space))
    if unknown:
        raise ValueError(f"parameters are outside the search space: {unknown}")
    for name, value in params.items():
        allowed = {str(item) for item in search_space[name]}
        if value not in allowed:
            raise ValueError(f"{name}={value} is not one of {sorted(allowed)}")


def run_evaluator(
    root: Path,
    config: dict[str, Any],
    params: dict[str, str],
    log_dir: Path,
    label: str,
) -> tuple[dict[str, Any], float]:
    command = config["evaluator"]["command"]
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in config["evaluator"].get("env", {}).items()})
    environment.update(params)
    environment["JOLT_AUTORESEARCH_EVAL_DIR"] = str(log_dir / f"{label}.artifacts")
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            timeout=config["evaluator"]["timeout_seconds"],
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
    output = parse_result(result.stdout)
    if output.get("kernel") != config["kernel"]:
        raise ValueError("evaluator returned the wrong kernel")
    metric = output.get("metrics", {}).get(config["metric"]["name"])
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric):
        raise ValueError("evaluator returned a non-finite primary metric")
    return output, elapsed


def guards_pass(config: dict[str, Any], output: dict[str, Any]) -> tuple[bool, str]:
    guards = output.get("guards")
    if not isinstance(guards, dict):
        return False, "evaluator returned no guard object"
    failed = [name for name in config["guards"]["required_true"] if guards.get(name) is not True]
    if failed:
        return False, f"failed guards: {failed}"
    return True, "all guards passed"


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_path = run_dir / "run.json"
    config = read_json(run_path)
    expected = (run_dir / "run.sha256").read_text().strip()
    if sha256(run_path.read_bytes()) != expected:
        raise ValueError("run.json changed after initialization")
    events: list[dict[str, Any]] = []
    accepted = {"baseline"}
    seen = {"baseline"}
    for number, line in enumerate((run_dir / "events.jsonl").read_text().splitlines(), 1):
        if not line:
            raise ValueError(f"events.jsonl:{number}: blank record")
        event = json.loads(line)
        if event.get("index") != number or event.get("verdict") not in VERDICTS:
            raise ValueError(f"events.jsonl:{number}: invalid event")
        if event.get("trial_id") in seen or event.get("parent_id") not in accepted:
            raise ValueError(f"events.jsonl:{number}: invalid lineage")
        seen.add(event["trial_id"])
        if event["verdict"] == "keep":
            accepted.add(event["trial_id"])
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


def grouped_medians(values: list[float], group_size: int) -> list[float]:
    if not values or group_size < 1 or len(values) % group_size != 0:
        raise ValueError("measurements do not form complete comparison groups")
    return [
        statistics.median(values[start : start + group_size])
        for start in range(0, len(values), group_size)
    ]


def goal_decision(
    contract: dict[str, Any],
    current_piop_speedup: float,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    floor = float(contract["primary_metric"]["minimum_accepted_speedup"])
    minimum_gain = float(contract["continuation"]["minimum_projected_relative_gain"])
    if not math.isfinite(current_piop_speedup) or current_piop_speedup <= 0.0:
        raise ValueError("current PIOP speedup must be finite and positive")
    if not math.isfinite(floor) or floor <= 1.0:
        raise ValueError("the accepted PIOP speedup floor must exceed one")
    if not math.isfinite(minimum_gain) or not 0.0 < minimum_gain < 1.0:
        raise ValueError("the projected continuation gain must be between zero and one")

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
    should_continue = not floor_met or projected_gain >= minimum_gain
    ranked.sort(key=lambda candidate: candidate["projected_time_fraction_saved"], reverse=True)
    return {
        "continue": should_continue,
        "floor_met": floor_met,
        "current_piop_speedup": current_piop_speedup,
        "minimum_accepted_speedup": floor,
        "projected_piop_speedup": projected_speedup,
        "projected_relative_gain": projected_gain,
        "minimum_projected_relative_gain": minimum_gain,
        "next_kernel": ranked[0]["kernel"] if ranked else None,
        "candidates": ranked,
        "reason": (
            "the minimum PIOP speedup has not been reached"
            if not floor_met
            else "conservative residual headroom clears the continuation threshold"
            if should_continue
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


def command_init(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    template = read_json(Path(args.template))
    validate_template(template)
    goal_contract = read_json(root / template["portfolio_contract"])
    validate_goal_contract(goal_contract)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = run_dir / "logs"
    logs.mkdir()
    snapshots = run_dir / "snapshots"
    snapshots.mkdir()
    snapshot_paths(root, template["scope"]["editable"], snapshots / "baseline")

    baseline_params = {str(k): str(v) for k, v in template.get("baseline_params", {}).items()}
    validate_params(template, baseline_params)
    measurements = []
    elapsed_total = 0.0
    gpu_seconds = 0.0
    for index in range(template["baseline_repeats"]):
        output, elapsed = run_evaluator(
            root, template, baseline_params, logs, f"baseline-{index + 1:02d}"
        )
        passed, reason = guards_pass(template, output)
        if not passed:
            raise ValueError(f"baseline {index + 1} is invalid: {reason}")
        measurements.append(float(output["metrics"][template["metric"]["name"]]))
        elapsed_total += elapsed
        gpu_seconds += float(output.get("resources", {}).get("gpu_seconds", 0.0))

    comparison_measurements = grouped_medians(measurements, template["candidate_repeats"])
    median = statistics.median(comparison_measurements)
    deviations = [abs(value - median) for value in comparison_measurements]
    relative_mad = statistics.median(deviations) / abs(median) if median else 0.0
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
        "frozen_paths_sha256": path_digest(root, config["scope"]["frozen"]),
        "editable_paths_sha256": path_digest(root, config["scope"]["editable"]),
        "portfolio_contract_sha256": sha256(canonical_json(goal_contract)),
    }
    config["baseline"] = {
        "params": baseline_params,
        "measurements": measurements,
        "comparison_measurements": comparison_measurements,
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
    print(json.dumps({"run_dir": str(run_dir), "baseline": config["baseline"]}, sort_keys=True))
    return 0


def command_trial(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve()
    config, events = load_run(run_dir)
    if path_digest(root, config["scope"]["frozen"]) != config["fingerprint"]["frozen_paths_sha256"]:
        raise ValueError("a frozen path changed; start a new run phase")
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

    params = dict(item.split("=", 1) for item in args.param)
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
                "params": params,
                "started_at": started_at,
            }
        )
    )
    elapsed = 0.0
    gpu_seconds = 0.0
    measurements = []
    combined_guards = {name: True for name in config["guards"]["required_true"]}
    try:
        for repeat in range(config.get("candidate_repeats", 1)):
            output, repetition_elapsed = run_evaluator(
                root,
                config,
                params,
                run_dir / "logs",
                f"{trial_id}-{repeat + 1:02d}",
            )
            elapsed += repetition_elapsed
            gpu_seconds += float(output.get("resources", {}).get("gpu_seconds", 0.0))
            measurements.append(float(output["metrics"][config["metric"]["name"]]))
            passed, reason = guards_pass(config, output)
            for name in combined_guards:
                combined_guards[name] = combined_guards[name] and output["guards"].get(name) is True
            if not passed:
                break
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
        "proposal_summary": args.summary,
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
    append_event(run_dir / "events.jsonl", event)
    if verdict == "keep":
        snapshot_paths(
            root,
            config["scope"]["editable"],
            run_dir / "snapshots" / trial_id,
        )
    else:
        restore_snapshot(
            root,
            config["scope"]["editable"],
            run_dir / "snapshots" / parent_id,
        )
    inflight.unlink()
    print(json.dumps(event, sort_keys=True))
    return 0 if verdict in {"keep", "discard"} else 2


def command_status(args: argparse.Namespace) -> int:
    config, events = load_run(Path(args.run_dir).resolve())
    parent_id, metric = accepted_parent(config, events)
    summary = {
        "kernel": config["kernel"],
        "trials": len(events),
        "remaining_trials": config["budget"]["max_trials"] - len(events),
        "accepted_parent": parent_id,
        "accepted_metric": metric,
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


def command_recover(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve()
    config, events = load_run(run_dir)
    inflight = run_dir / "inflight.json"
    if not inflight.exists():
        raise ValueError("there is no interrupted trial")
    parent_id, _ = accepted_parent(config, events)
    quarantine = run_dir / "quarantine" / utc_now().replace(":", "-")
    snapshot_paths(root, config["scope"]["editable"], quarantine)
    restore_snapshot(
        root,
        config["scope"]["editable"],
        run_dir / "snapshots" / parent_id,
    )
    inflight.unlink()
    print(json.dumps({"restored": parent_id, "quarantine": str(quarantine)}, sort_keys=True))
    return 0


def command_goal_decision(args: argparse.Namespace) -> int:
    contract = read_json(Path(args.contract))
    validate_goal_contract(contract)
    candidates = [parse_goal_candidate(value) for value in args.candidate]
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
    trial = commands.add_parser("trial")
    trial.add_argument("run_dir")
    trial.add_argument("--param", action="append", default=[])
    trial.add_argument("--summary", required=True)
    trial.set_defaults(handler=command_trial)
    status = commands.add_parser("status")
    status.add_argument("run_dir")
    status.set_defaults(handler=command_status)
    recover = commands.add_parser("recover")
    recover.add_argument("run_dir")
    recover.set_defaults(handler=command_recover)
    goal = commands.add_parser("goal-decision")
    goal.add_argument("contract")
    goal.add_argument("--current-speedup", type=float, required=True)
    goal.add_argument("--candidate", action="append", default=[])
    goal.set_defaults(handler=command_goal_decision)
    goal_prompt = commands.add_parser("goal-prompt")
    goal_prompt.add_argument("contract")
    goal_prompt.set_defaults(handler=command_goal_prompt)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
