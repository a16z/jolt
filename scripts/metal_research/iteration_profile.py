from __future__ import annotations

import json
import os
import secrets
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import (
    canonical_json,
    materialize_outer_artifact,
    outer_dispatch_from_params,
    sha256,
    validate_runtime_artifact_output,
)
from .attempt import run_attempt, sanitized_parent_environment
from .binaries import (
    declared_source_sha256,
    materialize_sealed_binary,
    prepare_sealed_binary_from_output,
    seal_sealed_binary_store,
    sealed_binary_token,
    verify_sealed_binary_contract,
)
from .contracts import (
    ITERATION_PROFILE_CONTROLLER_PATHS,
    ITERATION_PROFILE_SOURCE_PATHS,
    phase_checkpoint_record,
)
from .results import adapt_result, validate_tier_result


ValidateOutput = Callable[
    [Path, dict[str, Any], dict[str, Any], dict[str, str]], None
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_text(root: Path, command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=root,
        env=sanitized_parent_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ValueError(f"profiling fingerprint command failed: {command}")
    return completed.stdout.strip()


def _machine_record(root: Path, device_name: str) -> dict[str, str]:
    rustc = _run_text(root, ["rustc", "-vV"])
    rust_fields = {}
    for line in rustc.splitlines():
        if ": " in line:
            name, value = line.split(": ", 1)
            rust_fields[name] = value
    cargo = _run_text(root, ["cargo", "-V"]).split()
    if len(cargo) < 2:
        raise ValueError("cargo version fingerprint is invalid")
    required_rust = {"release", "commit-hash", "host", "LLVM version"}
    if not required_rust <= set(rust_fields):
        raise ValueError("rustc version fingerprint is incomplete")
    return {
        "device_name": device_name,
        "os_product_version": _run_text(root, ["sw_vers", "-productVersion"]),
        "os_build_version": _run_text(root, ["sw_vers", "-buildVersion"]),
        "macos_sdk_version": _run_text(
            root, ["xcrun", "--sdk", "macosx", "--show-sdk-version"]
        ),
        "rustc_release": rust_fields["release"],
        "rustc_commit_hash": rust_fields["commit-hash"],
        "rustc_host": rust_fields["host"],
        "llvm_version": rust_fields["LLVM version"],
        "cargo_release": cargo[1],
    }


def _source_records(root: Path, paths: tuple[str, ...]) -> list[dict[str, Any]]:
    records = []
    for relative in paths:
        payload = (root / relative).read_bytes()
        records.append(
            {"path": relative, "bytes": len(payload), "sha256": sha256(payload)}
        )
    return records


def _outer_closure(
    root: Path, candidate_suffix: str, offset: int
) -> dict[str, Any]:
    fragments = _source_records(root, ITERATION_PROFILE_SOURCE_PATHS)
    payloads = [(root / record["path"]).read_bytes() for record in fragments]
    candidate = payloads[-1] + candidate_suffix.encode()
    prefix = f"#define SOLINAS_OFFSET {offset}u\n".encode()
    parent_assembled = prefix + b"\n".join(payloads)
    candidate_assembled = prefix + b"\n".join([*payloads[:-1], candidate])
    return {
        "dependency_model": "outer_only_shader_closure_v1",
        "source_fragments": fragments,
        "solinas_offset": offset,
        "parent_assembled_source_bytes": len(parent_assembled),
        "parent_assembled_source_sha256": sha256(parent_assembled),
        "candidate_assembled_source_bytes": len(candidate_assembled),
        "candidate_assembled_source_sha256": sha256(candidate_assembled),
        "candidate_source_suffix": candidate_suffix,
    }


def _proxy_tier(template: dict[str, Any]) -> dict[str, Any]:
    tiers = [
        tier
        for tier in template["evaluation"]["tiers"]
        if tier.get("applicable") is True and tier.get("role") == "proxy"
    ]
    if len(tiers) != 1:
        raise ValueError("iteration profiling requires exactly one proxy tier")
    return tiers[0]


def _profiled_binary(
    template: dict[str, Any], tier_id: str
) -> tuple[str, dict[str, Any]]:
    matches = [
        (binary_id, contract)
        for binary_id, contract in template["sealed_binaries"].items()
        if tier_id in contract["consumer_tiers"]
    ]
    if len(matches) != 1:
        raise ValueError("iteration profiling requires one sealed proxy evaluator")
    return matches[0]


def _build_runner(
    root: Path,
    run_dir: Path,
    binary_id: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    build = contract["build"]
    source_digest = declared_source_sha256(root, contract["source_paths"])
    environment = sanitized_parent_environment()
    completed = subprocess.run(
        build["command"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(build["timeout_seconds"]),
    )
    if completed.returncode != 0:
        raise ValueError(
            f"profile evaluator build failed with status {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    environment_digest = sha256(canonical_json(environment))
    prepared = prepare_sealed_binary_from_output(
        root,
        binary_id,
        contract,
        source_digest,
        environment_digest,
    )
    record = materialize_sealed_binary(run_dir, prepared)
    seal_sealed_binary_store(run_dir)
    runner = verify_sealed_binary_contract(
        root, run_dir, binary_id, contract, record
    )
    return {"record": record, "runner": runner}


def _profile_cycle(
    root: Path,
    run_dir: Path,
    cycle_name: str,
    tier: dict[str, Any],
    params: dict[str, str],
    phase: dict[str, Any],
    parent: dict[str, Any],
    candidate: dict[str, Any],
    runner: Path,
    runner_digest: str,
    binary_id: str,
    binary_record: dict[str, Any],
    validate_output: ValidateOutput,
) -> tuple[dict[str, Any], bytes]:
    token = sealed_binary_token(binary_id)
    command = [
        str(runner) if argument == token else argument
        for argument in tier["evaluator"]["command"]
    ]
    if command == tier["evaluator"]["command"] or token in command:
        raise ValueError("profile evaluator token was not resolved")
    evaluator = {**tier["evaluator"], "command": command}
    context = {"kind": "outer_msl_v1", "parent": parent, "candidate": candidate}
    context_env = {
        "JOLT_AUTORESEARCH_PARENT_ARTIFACT": str(
            (run_dir / parent["artifact_path"]).resolve()
        ),
        "JOLT_AUTORESEARCH_CANDIDATE_ARTIFACT": str(
            (run_dir / candidate["artifact_path"]).resolve()
        ),
        "JOLT_AUTORESEARCH_RUNNER_SHA256": runner_digest,
    }
    started = time.monotonic_ns()
    attempt, output = run_attempt(
        root,
        evaluator,
        params,
        run_dir / "evaluations" / cycle_name,
        f"iteration-profile:{cycle_name}",
        queue_timeout_seconds=float(evaluator["timeout_seconds"]),
        context_env=context_env,
        context_record=context,
        sealed_binary_context={binary_id: binary_record},
    )
    if attempt["outcome"] != "success" or output is None:
        raise ValueError(
            f"profile {cycle_name} evaluator failed: {attempt.get('error')}"
        )
    parse_started = time.monotonic_ns()
    validate_runtime_artifact_output(output, "outer_msl_v1", context)
    fingerprint = output.get("fingerprint", {})
    if fingerprint.get("runner_binary_sha256") != runner_digest:
        raise ValueError("profile result does not identify the sealed evaluator")
    validate_output(root, tier, output, params)
    normalized, _ = adapt_result(tier, output, "OuterRemainder")
    validate_tier_result(normalized, tier)
    checkpoint = phase_checkpoint_record(
        phase, normalized, int(phase["checkpoint"]["after_candidates"])
    )
    parse_ns = max(1, time.monotonic_ns() - parse_started)
    elapsed_ns = max(1, time.monotonic_ns() - started)
    subprocess_ns = max(
        1, round(float(attempt["controller"]["subprocess_wall_seconds"]) * 1e9)
    )
    overhead_ns = max(parse_ns, elapsed_ns - subprocess_ns, 1)
    raw = canonical_json(output)
    candidate_phases = normalized["telemetry"]["candidate_phase_gpu_active_ns"]
    return (
        {
            "controller_wall_ns": subprocess_ns + overhead_ns,
            "subprocess_wall_ns": subprocess_ns,
            "parse_validate_checkpoint_ns": parse_ns,
            "controller_overhead_ns": overhead_ns,
            "raw_result_path": None,
            "raw_result_sha256": sha256(raw),
            "result_bytes": len(raw),
            "successor_speedup": normalized["primary"]["value"],
            "gpu_active_total_ns": output["resources"]["gpu_active_total_ns"],
            "output_sha256": output["samples"][0]["candidate"]["output_sha256"],
            "candidate_phase_gpu_active_ns": candidate_phases,
            "compilation": output["telemetry"]["compilation"],
            "guards": output["guards"],
            "checkpoint": checkpoint,
        },
        raw,
    )


def _repository_output(root: Path, output_prefix: Path) -> tuple[Path, str]:
    absolute = output_prefix if output_prefix.is_absolute() else root / output_prefix
    absolute = absolute.resolve()
    if root != absolute.parent and root not in absolute.parent.parents:
        raise ValueError("iteration profile output must stay within the repository")
    return absolute, absolute.relative_to(root).as_posix()


def _publish_bundle(payloads: list[tuple[Path, bytes]]) -> None:
    if any(path.exists() for path, _ in payloads):
        raise ValueError("iteration profile output already exists")
    temporary: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for path, payload in payloads:
            path.parent.mkdir(parents=True, exist_ok=True)
            staging = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
            descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("iteration profile write made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            temporary.append((staging, path))
        for staging, path in temporary:
            staging.replace(path)
            published.append(path)
    except BaseException:
        for staging, _ in temporary:
            if staging.exists():
                staging.unlink()
        for path in published:
            path.unlink()
        raise


def generate_iteration_profile(
    root: Path,
    template: dict[str, Any],
    output_prefix: Path,
    validate_output: ValidateOutput,
) -> dict[str, Any]:
    root = root.resolve()
    prefix, relative_prefix = _repository_output(root, output_prefix)
    summary_path = Path(f"{prefix}.json")
    cold_path = prefix.with_name(f"{prefix.name}.cold.raw.json")
    warm_path = prefix.with_name(f"{prefix.name}.warm.raw.json")
    tier = _proxy_tier(template)
    binary_id, binary_contract = _profiled_binary(template, tier["id"])
    params = {str(name): str(value) for name, value in template["baseline_params"].items()}
    phase = template["mechanism_phase"]
    suffix = (
        f"\n// iteration profile {phase['id']} {secrets.token_hex(8)}\n"
    )
    closure = _outer_closure(root, suffix, 275)

    with tempfile.TemporaryDirectory(prefix="jolt-metal-iteration-profile-") as directory:
        run_dir = Path(directory)
        (run_dir / "artifacts").mkdir()
        (run_dir / "binaries").mkdir()
        (run_dir / "evaluations").mkdir()
        candidate_source = run_dir / "candidate.metal"
        outer_source = root / ITERATION_PROFILE_SOURCE_PATHS[-1]
        candidate_source.write_bytes(outer_source.read_bytes() + suffix.encode())
        dispatch = outer_dispatch_from_params(params)
        binding_plan = params["JOLT_METAL_OUTER_REMAINDER_BINDING_PLAN"]
        parent = materialize_outer_artifact(
            run_dir, outer_source, binding_plan, dispatch
        )
        candidate = materialize_outer_artifact(
            run_dir, candidate_source, binding_plan, dispatch
        )
        built = _build_runner(root, run_dir, binary_id, binary_contract)
        binary_record = built["record"]
        runner_digest = binary_record["manifest"]["binary_sha256"]
        cold, cold_raw = _profile_cycle(
            root,
            run_dir,
            "cold",
            tier,
            params,
            phase,
            parent,
            candidate,
            built["runner"],
            runner_digest,
            binary_id,
            binary_record,
            validate_output,
        )
        warm, warm_raw = _profile_cycle(
            root,
            run_dir,
            "warm",
            tier,
            params,
            phase,
            parent,
            candidate,
            built["runner"],
            runner_digest,
            binary_id,
            binary_record,
            validate_output,
        )

    for cycle in (cold, warm):
        overhead = cycle["controller_overhead_ns"] / cycle["controller_wall_ns"]
        if overhead > float(template["iteration_profile"]["maximum_controller_overhead_fraction"]):
            raise ValueError("iteration profile controller overhead exceeds its contract")
    cycles_per_hour = 3_600_000_000_000 / cold["controller_wall_ns"]
    minimum_cycles = float(
        template["iteration_profile"]["minimum_valid_proxy_cycles_per_hour"]
    )
    if cycles_per_hour < minimum_cycles:
        raise ValueError("iteration profile misses the valid-cycles-per-hour contract")
    cold_compile = cold["compilation"]["candidate"]["library_compile_ns"]
    warm_compile = warm["compilation"]["candidate"]["library_compile_ns"]
    if cold_compile <= 10 * warm_compile:
        raise ValueError("iteration profile did not observe a distinct cold compile")

    cold_relative = f"{relative_prefix}.cold.raw.json"
    warm_relative = f"{relative_prefix}.warm.raw.json"
    cold["raw_result_path"] = cold_relative
    warm["raw_result_path"] = warm_relative
    fingerprint = json.loads(cold_raw)["fingerprint"]
    telemetry = json.loads(cold_raw)["telemetry"]
    evidence = {
        "schema": "outer_remainder_iteration_profile_v3",
        "schema_version": 3,
        "created_at": _utc_now(),
        "profile_base_revision": _run_text(root, ["git", "rev-parse", "HEAD"]),
        "machine": _machine_record(root, telemetry["device_name"]),
        "controller_sources": _source_records(
            root, ITERATION_PROFILE_CONTROLLER_PATHS
        ),
        "evaluator": {
            "result_adapter": tier["evaluator"]["result_adapter"],
            "runner_binary_sha256": runner_digest,
            "runner_source_sha256": binary_record["manifest"]["source_sha256"],
            "log_n": fingerprint["log_n"],
            "pairs": fingerprint["pairs"],
            "excluded_warmup_pairs": fingerprint["excluded_warmup_pairs"],
            "rayon_threads": int(tier["evaluator"]["env"]["RAYON_NUM_THREADS"]),
            "binding_plan": binding_plan,
            "cpu_tail_elements": dispatch["cpu_tail_elements"],
            "trace_cutoff_elements": dispatch["trace_cutoff_elements"],
            "parent_artifact_sha256": parent["artifact_sha256"],
            "candidate_artifact_sha256": candidate["artifact_sha256"],
            "parent_outer_source_sha256": parent["manifest"]["outer_source_sha256"],
            "candidate_outer_source_sha256": candidate["manifest"]["outer_source_sha256"],
        },
        "minimal_closure": closure,
        "cold_cycle": cold,
        "warm_cycle": warm,
    }
    evidence_payload = json.dumps(evidence, indent=2).encode() + b"\n"
    _publish_bundle(
        [
            (cold_path, cold_raw),
            (warm_path, warm_raw),
            (summary_path, evidence_payload),
        ]
    )
    profile = {
        "profile_base_revision": evidence["profile_base_revision"],
        "evidence_path": f"{relative_prefix}.json",
        "evidence_sha256": sha256(evidence_payload),
        "minimum_valid_proxy_cycles_per_hour": minimum_cycles,
        "maximum_controller_overhead_fraction": float(
            template["iteration_profile"]["maximum_controller_overhead_fraction"]
        ),
    }
    return {
        "iteration_profile": profile,
        "cold_cycles_per_hour": cycles_per_hour,
        "cold_controller_seconds": cold["controller_wall_ns"] / 1e9,
        "warm_controller_seconds": warm["controller_wall_ns"] / 1e9,
        "evidence": str(summary_path),
    }
