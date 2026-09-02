#!/usr/bin/env python3
"""Validate and report the registered scalar Fp64 proof builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "proofs/hol-light/fp64-certified-builds.json"
CERTIFICATION_SCOPES = {
    "complete-inspection-symbol",
    "checked-wrapper-proved-inner-sequence",
}
WRAPPER_POLICIES = {"exact", "darwin-frame"}
POST_RETURN_POLICIES = {"none", "nop-padding-only", "int3-padding-only"}
RUNNER_PROFILE_IDS = {
    "aarch64": ["baseline"],
    "x86_64": ["baseline", "bmi2-mul"],
}
MATRIX_CONTRACT_ENV = "JOLT_FP64_PROOF_MATRIX_CONTRACT"
RELEASE_PROFILE_FIELDS = {
    "opt_level",
    "lto",
    "codegen_units",
    "debug",
    "debug_assertions",
    "overflow_checks",
    "split_debuginfo",
    "strip",
    "rpath",
    "incremental",
    "panic",
}


def load_matrix(path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        matrix = json.load(source)
    if not isinstance(matrix, dict):
        raise ValueError("Fp64 proof matrix root must be an object")
    return matrix


def matrix_contract_id(matrix: dict[str, Any]) -> str:
    encoded = json.dumps(matrix, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def target_by_id(matrix: dict[str, Any], target_id: str) -> dict[str, Any]:
    matches = [target for target in matrix["targets"] if target["id"] == target_id]
    if len(matches) != 1:
        raise ValueError(f"Fp64 proof target id {target_id!r} is not registered exactly once")
    return matches[0]


def target_by_triple(matrix: dict[str, Any], triple: str) -> dict[str, Any]:
    matches = [
        target for target in matrix["targets"] if target["target_triple"] == triple
    ]
    if len(matches) != 1:
        raise ValueError(f"Fp64 proof target triple {triple!r} is not registered exactly once")
    return matches[0]


def profile_by_id(target: dict[str, Any], profile_id: str) -> dict[str, Any]:
    matches = [profile for profile in target["profiles"] if profile["id"] == profile_id]
    if len(matches) != 1:
        raise ValueError(
            f"Fp64 profile {profile_id!r} is not registered exactly once for {target['id']}"
        )
    return matches[0]


def validate_matrix(matrix: dict[str, Any], repo_root: Path = REPO_ROOT) -> None:
    if matrix.get("schema_version") != 1:
        raise ValueError("unsupported Fp64 proof matrix schema")
    targets = matrix.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Fp64 proof matrix must contain targets")

    ids = [target.get("id") for target in targets]
    triples = [target.get("target_triple") for target in targets]
    if len(ids) != len(set(ids)):
        raise ValueError("Fp64 proof matrix contains duplicate target ids")
    if len(triples) != len(set(triples)):
        raise ValueError("Fp64 proof matrix contains duplicate target triples")

    required_target_fields = {
        "id",
        "target_triple",
        "architecture",
        "vendor",
        "target_os",
        "target_env",
        "endian",
        "pointer_width",
        "object_format_marker",
        "certification_scope",
        "wrapper_policy",
        "post_return_policy",
        "ci_required",
        "profiles",
    }
    required_operation_fields = {
        "name",
        "expected_sequence",
        "object_source",
        "correctness_source",
        "body_theorem",
        "subroutine_theorem",
    }
    required_profile_fields = {
        "id",
        "rustflags",
        "target_features",
        "required_cpu_features",
        "operations",
    }
    for target in targets:
        missing = required_target_fields - target.keys()
        if missing:
            raise ValueError(f"{target.get('id', '<unknown>')} lacks {sorted(missing)}")
        if target["architecture"] not in {"aarch64", "x86_64"}:
            raise ValueError(f"unsupported architecture in {target['id']}")
        if target["endian"] != "little" or target["pointer_width"] != 64:
            raise ValueError(f"{target['id']} is not a little-endian 64-bit build")
        if target["certification_scope"] not in CERTIFICATION_SCOPES:
            raise ValueError(f"unknown certification scope in {target['id']}")
        if target["wrapper_policy"] not in WRAPPER_POLICIES:
            raise ValueError(f"unknown wrapper policy in {target['id']}")
        if target["post_return_policy"] not in POST_RETURN_POLICIES:
            raise ValueError(f"unknown post-return policy in {target['id']}")
        if target["ci_required"] and target["certification_scope"] != "complete-inspection-symbol":
            raise ValueError(f"CI-required target {target['id']} is not a complete matrix entry")
        if (
            target["certification_scope"] == "complete-inspection-symbol"
            and not target["ci_required"]
        ):
            raise ValueError(f"complete target {target['id']} must run in CI")

        profile_ids = [profile.get("id") for profile in target["profiles"]]
        expected_profile_ids = RUNNER_PROFILE_IDS[target["architecture"]]
        if profile_ids != expected_profile_ids:
            raise ValueError(
                f"{target['id']} profiles must be exactly {expected_profile_ids}"
            )
        for profile in target["profiles"]:
            missing = required_profile_fields - profile.keys()
            if missing:
                raise ValueError(
                    f"{target['id']} profile {profile.get('id')} lacks {sorted(missing)}"
                )
            if not isinstance(profile.get("target_features"), list):
                raise ValueError(
                    f"{target['id']} profile {profile.get('id')} has no target feature set"
                )
            if not profile.get("operations"):
                raise ValueError(f"{target['id']} profile {profile.get('id')} has no operations")
            operation_names = [operation.get("name") for operation in profile["operations"]]
            if len(operation_names) != len(set(operation_names)):
                raise ValueError(f"duplicate operation in {target['id']} profile {profile['id']}")
            for operation in profile["operations"]:
                missing = required_operation_fields - operation.keys()
                if missing:
                    raise ValueError(
                        f"{target['id']} {profile['id']} operation lacks {sorted(missing)}"
                    )
                if operation["name"] not in {"add", "sub", "mul"}:
                    raise ValueError(f"unknown operation in {target['id']}")
                for source_field in ("object_source", "correctness_source"):
                    source = repo_root / "proofs/hol-light" / operation[source_field]
                    if not source.is_file():
                        raise ValueError(f"registered proof source does not exist: {source}")
                correctness = (
                    repo_root / "proofs/hol-light" / operation["correctness_source"]
                ).read_text(encoding="utf-8")
                for theorem_field in ("body_theorem", "subroutine_theorem"):
                    if operation[theorem_field] not in correctness:
                        raise ValueError(
                            f"{operation[theorem_field]} is absent from "
                            f"{operation['correctness_source']}"
                        )
        baseline = profile_by_id(target, "baseline")
        if baseline["rustflags"] or baseline["required_cpu_features"]:
            raise ValueError(f"{target['id']} baseline cannot require optional features")
        if {operation["name"] for operation in baseline["operations"]} != {
            "add",
            "sub",
            "mul",
        }:
            raise ValueError(f"{target['id']} baseline must cover add, sub, and mul")
        if target["architecture"] == "x86_64":
            bmi2 = profile_by_id(target, "bmi2-mul")
            if (
                bmi2["rustflags"] != ["-C", "target-feature=+bmi2"]
                or bmi2["required_cpu_features"] != ["bmi2"]
                or [operation["name"] for operation in bmi2["operations"]] != ["mul"]
            ):
                raise ValueError(
                    f"{target['id']} bmi2-mul profile does not match runner support"
                )

    with (repo_root / "Cargo.toml").open("rb") as source:
        cargo_toml = tomllib.load(source)
    actual_release = cargo_toml.get("profile", {}).get("release", {})
    expected_release = matrix["build_contract"]["release_profile"]
    if set(expected_release) != RELEASE_PROFILE_FIELDS:
        raise ValueError(
            "Fp64 release profile fields must be exactly "
            f"{sorted(RELEASE_PROFILE_FIELDS)}"
        )
    cargo_field_names = {
        "opt_level": "opt-level",
        "lto": "lto",
        "codegen_units": "codegen-units",
        "debug": "debug",
        "debug_assertions": "debug-assertions",
        "overflow_checks": "overflow-checks",
        "split_debuginfo": "split-debuginfo",
        "strip": "strip",
        "rpath": "rpath",
        "incremental": "incremental",
        "panic": "panic",
    }
    for field, cargo_field in cargo_field_names.items():
        if (
            cargo_field in actual_release
            and actual_release[cargo_field] != expected_release[field]
        ):
            raise ValueError(
                f"Cargo release profile {cargo_field} does not match the Fp64 matrix"
            )

    workflow = (repo_root / ".github/workflows/fp64-formal-verification.yml").read_text(
        encoding="utf-8"
    )
    for required_path in (
        "proofs/hol-light/fp64-certified-builds.json",
        "scripts/fp64_certified_matrix.py",
        "scripts/tests/test_fp64_certified_matrix.py",
    ):
        if workflow.count(required_path) < 2:
            raise ValueError(f"Fp64 workflow does not watch {required_path}")
    for target in targets:
        occurrences = workflow.count(f"--matrix-entry {target['id']}")
        expected_occurrences = 1 if target["ci_required"] else 0
        if occurrences != expected_occurrences:
            raise ValueError(
                f"Fp64 workflow matrix entry count for {target['id']} is {occurrences}, "
                f"expected {expected_occurrences}"
            )


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_rustc_verbose(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    first, *lines = text.splitlines()
    result["version"] = first
    for line in lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            result[key.replace("-", "_")] = value
    return result


def parse_cargo_version(text: str) -> dict[str, str]:
    parts = text.split()
    if len(parts) < 3 or not parts[2].startswith("("):
        raise ValueError(f"unexpected cargo version output: {text}")
    return {"version": parts[1], "commit": parts[2].removeprefix("(")}


def codegen_overrides(matrix: dict[str, Any], environment: dict[str, str]) -> dict[str, str]:
    contract = matrix["build_contract"]
    forbidden: dict[str, str] = {}
    for name in contract["forbidden_environment_exact"]:
        if value := environment.get(name):
            forbidden[name] = value
    for name, value in environment.items():
        if not value:
            continue
        if any(name.startswith(prefix) for prefix in contract["forbidden_environment_prefixes"]):
            forbidden[name] = value
        if name.startswith("CARGO_TARGET_") and any(
            name.endswith(suffix)
            for suffix in contract["forbidden_target_environment_suffixes"]
        ):
            forbidden[name] = value
    return forbidden


def validate_toolchain(
    matrix: dict[str, Any], rustc_command: str = "rustc", cargo_command: str = "cargo"
) -> dict[str, Any]:
    rustc = parse_rustc_verbose(command_output([rustc_command, "-Vv"]))
    cargo = parse_cargo_version(command_output([cargo_command, "-V"]))
    expected = matrix["toolchain"]
    actual_toolchain = {
        "rustc_release": rustc.get("release"),
        "rustc_commit_hash": rustc.get("commit_hash"),
        "llvm_version": rustc.get("LLVM version"),
        "cargo_release": cargo["version"],
        "cargo_commit_hash": cargo["commit"],
    }
    mismatches = {
        name: (expected[name], actual)
        for name, actual in actual_toolchain.items()
        if expected[name] != actual
    }
    if mismatches:
        raise ValueError(f"toolchain is outside the Fp64 proof matrix: {mismatches}")
    return {"rustc": rustc, "cargo": cargo}


def validate_environment(matrix: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    toolchain = validate_toolchain(matrix)
    if overrides := codegen_overrides(matrix, dict(os.environ)):
        raise ValueError(f"ambient code-generation overrides are not certified: {overrides}")
    return {
        **toolchain,
        "target": target["target_triple"],
    }


def validate_build_arguments(
    matrix: dict[str, Any],
    args: argparse.Namespace,
    environment: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = target_by_triple(matrix, args.target_triple)
    actual_fields: dict[str, Any] = {
        "architecture": args.architecture,
        "vendor": args.vendor,
        "target_os": args.target_os,
        "target_env": args.target_env,
        "endian": args.endian,
        "pointer_width": args.pointer_width,
    }
    mismatches = {
        name: (target[name], actual)
        for name, actual in actual_fields.items()
        if target[name] != actual
    }
    if mismatches:
        raise ValueError(f"Cargo target disagrees with the Fp64 matrix: {mismatches}")

    actual_features = {feature for feature in args.target_features.split(",") if feature}
    matches = [
        profile
        for profile in target["profiles"]
        if set(profile["target_features"]) == actual_features
    ]
    if len(matches) != 1:
        raise ValueError(
            f"no registered feature profile for {target['id']}: {sorted(actual_features)}"
        )
    profile = matches[0]

    build_environment = dict(os.environ) if environment is None else environment
    if build_environment.get(MATRIX_CONTRACT_ENV) != matrix_contract_id(matrix):
        raise ValueError(
            "Fp64 proof linkage must be invoked through check-fp64.sh with the "
            "current matrix contract"
        )
    encoded_rustflags = build_environment.get("CARGO_ENCODED_RUSTFLAGS", "")
    actual_rustflags = encoded_rustflags.split("\x1f") if encoded_rustflags else []
    if actual_rustflags != profile["rustflags"]:
        raise ValueError(
            f"Rust flags differ from registered profile {profile['id']}: "
            f"{actual_rustflags}"
        )
    overrides = codegen_overrides(matrix, build_environment)
    overrides.pop("CARGO_ENCODED_RUSTFLAGS", None)
    if overrides:
        raise ValueError(
            f"ambient code-generation overrides are not certified: {overrides}"
        )

    contract = matrix["build_contract"]
    expected_debug = contract["release_profile"]["debug"] != 0
    if (
        args.profile != contract["cargo_profile"]
        or args.opt_level != str(contract["release_profile"]["opt_level"])
        or args.debug != str(expected_debug).lower()
    ):
        raise ValueError(
            f"Fp64 proof linkage requires the registered {contract['cargo_profile']} profile"
        )
    validate_toolchain(matrix, rustc_command=args.rustc)
    return target, profile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repository: Path, *arguments: str) -> str:
    return command_output(["git", "-C", str(repository), *arguments])


def assembler_identity() -> dict[str, str]:
    command = shlex.split(os.environ.get("CC", "cc"))
    try:
        version = command_output([*command, "--version"]).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        version = "unavailable"
    return {"command": " ".join(command), "version": version}


def artifact_arguments(values: list[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("artifact arguments must use label=path")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not label or not path.is_file():
            raise ValueError(f"invalid evidence artifact: {value}")
        if label in artifacts:
            raise ValueError(f"duplicate evidence artifact label: {label}")
        artifacts[label] = path
    return artifacts


def write_evidence(args: argparse.Namespace, matrix: dict[str, Any]) -> None:
    target = target_by_id(matrix, args.target_id)
    selected_profiles = [profile_by_id(target, profile_id) for profile_id in args.profile]
    expected_profile_ids = [profile["id"] for profile in target["profiles"]]
    if args.profile != expected_profile_ids:
        raise ValueError(
            f"evidence profiles must be exactly {expected_profile_ids}, got {args.profile}"
        )
    environment = validate_environment(matrix, target)
    hol_light = Path(args.hol_light_dir)
    s2n_bignum = Path(args.s2n_bignum_dir)
    actual_pins = {
        "hol_light_commit": git_value(hol_light, "rev-parse", "HEAD"),
        "s2n_bignum_commit": git_value(s2n_bignum, "rev-parse", "HEAD"),
    }
    if actual_pins != matrix["proof_libraries"]:
        raise ValueError(
            f"proof library revisions differ from the matrix: {actual_pins}"
        )

    repo_root = Path(args.repo_root)
    artifacts = artifact_arguments(args.artifact)
    required_labels = {
        "add_object",
        "add_proof_object",
        "sub_object",
        "sub_proof_object",
        "mul_object",
        "mul_proof_object",
        "production_witness",
        "proof_log",
    }
    if any(profile["id"] == "bmi2-mul" for profile in selected_profiles):
        required_labels.update(
            {
                "mul_bmi2_object",
                "mul_bmi2_proof_object",
                "bmi2_production_witness",
            }
        )
    if set(artifacts) != required_labels:
        raise ValueError(
            f"evidence artifacts must be exactly {sorted(required_labels)}, "
            f"got {sorted(artifacts)}"
        )

    checker = repo_root / "scripts/check_fp64_proof_artifacts.py"
    checker_command = [
        sys.executable,
        str(checker),
        "--matrix",
        str(args.matrix),
        "--target-id",
        target["id"],
        "--add-object",
        str(artifacts["add_object"]),
        "--sub-object",
        str(artifacts["sub_object"]),
        "--mul-object",
        str(artifacts["mul_object"]),
        "--production-witness",
        str(artifacts["production_witness"]),
    ]
    if "mul_bmi2_object" in artifacts:
        checker_command.extend(
            [
                "--mul-bmi2-object",
                str(artifacts["mul_bmi2_object"]),
                "--bmi2-production-witness",
                str(artifacts["bmi2_production_witness"]),
            ]
        )
    byte_check_output = command_output(checker_command)

    theorem_names = [
        operation["subroutine_theorem"]
        for profile in selected_profiles
        for operation in profile["operations"]
    ]
    theorem_names.append("JOLT_FP64_PRIME")
    proof_log = artifacts["proof_log"].read_text(encoding="utf-8")
    missing_theorems = [
        theorem for theorem in theorem_names if f"val {theorem} : thm" not in proof_log
    ]
    if missing_theorems:
        raise ValueError(f"proof log lacks theorem markers: {missing_theorems}")

    ci_environment = {
        name: os.environ[name]
        for name in (
            "GITHUB_ACTIONS",
            "GITHUB_REPOSITORY",
            "GITHUB_WORKFLOW",
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "GITHUB_SHA",
        )
        if os.environ.get(name)
    }
    evidence = {
        "schema_version": 1,
        "matrix_entry_id": target["id"],
        "certification_scope": target["certification_scope"],
        "matrix_sha256": sha256(Path(args.matrix)),
        "matrix_contract_id": matrix_contract_id(matrix),
        "record": {
            "authentication": "none",
            "interpretation": (
                "This is an unauthenticated run record. Trusted CI provenance or "
                "independent replay is required to establish that the recorded run "
                "occurred."
            ),
            "reported_ci_environment": ci_environment,
        },
        "source": {
            "jolt_commit": git_value(repo_root, "rev-parse", "HEAD"),
            "worktree_clean": not bool(
                git_value(repo_root, "status", "--porcelain")
            ),
            "cargo_lock_sha256": sha256(repo_root / "Cargo.lock"),
            "cargo_toml_sha256": sha256(repo_root / "Cargo.toml"),
        },
        "build": {
            "target": target,
            "selected_profiles": args.profile,
            "contract": matrix["build_contract"],
            "environment": environment,
            "assembler": assembler_identity(),
            "recorded_overrides": {
                name: value
                for name, value in os.environ.items()
                if value and name in {"CC", "CFLAGS", "LDFLAGS"}
            },
        },
        "artifacts": {
            label: {"file": path.name, "sha256": sha256(path)}
            for label, path in sorted(artifacts.items())
        },
        "proof": {
            "isolated_workspace_requested": args.clean,
            "exact_byte_comparison_completed": True,
            "exact_byte_comparison_output": byte_check_output,
            "wrapper_policy": target["wrapper_policy"],
            "required_theorem_markers_present": True,
            "proof_libraries": actual_pins,
            "theorems": theorem_names,
            "proof_log_sha256": sha256(artifacts["proof_log"]),
        },
        "downstream_binary": {
            "status": "not established by this run record",
            "reason": (
                "The run record covers one compiled inspection witness, not every "
                "inlined caller or a reachable release binary."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote unauthenticated Fp64 proof run record to {output}")


def nested_value(target: dict[str, Any], field: str) -> Any:
    value: Any = target
    for component in field.split("."):
        value = value[component]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate")
    subparsers.add_parser("contract-id")

    build_field = subparsers.add_parser("build-field")
    build_field.add_argument("--field", required=True)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--target-triple", required=True)

    get = subparsers.add_parser("get")
    get.add_argument("--target-id", required=True)
    get.add_argument("--field", required=True)

    profile_ids = subparsers.add_parser("profile-ids")
    profile_ids.add_argument("--target-id", required=True)

    operation = subparsers.add_parser("operation")
    operation.add_argument("--target-id", required=True)
    operation.add_argument("--profile", required=True)
    operation.add_argument("--operation", required=True)
    operation.add_argument("--field", required=True)

    check = subparsers.add_parser("check-environment")
    check.add_argument("--target-id", required=True)

    build = subparsers.add_parser("validate-build")
    build.add_argument("--target-triple", required=True)
    build.add_argument("--architecture", required=True)
    build.add_argument("--vendor", required=True)
    build.add_argument("--target-os", required=True)
    build.add_argument("--target-env", required=True)
    build.add_argument("--endian", required=True)
    build.add_argument("--pointer-width", type=int, required=True)
    build.add_argument("--target-features", required=True)
    build.add_argument("--profile", required=True)
    build.add_argument("--opt-level", required=True)
    build.add_argument("--debug", required=True)
    build.add_argument("--rustc", required=True)

    evidence = subparsers.add_parser("evidence")
    evidence.add_argument("--target-id", required=True)
    evidence.add_argument("--repo-root", required=True)
    evidence.add_argument("--hol-light-dir", required=True)
    evidence.add_argument("--s2n-bignum-dir", required=True)
    evidence.add_argument("--output", required=True)
    evidence.add_argument("--artifact", action="append", default=[])
    evidence.add_argument("--profile", action="append", default=[])
    evidence.add_argument("--clean", action="store_true")

    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    validate_matrix(matrix)

    if args.command == "validate":
        print(f"Validated {len(matrix['targets'])} registered Fp64 proof targets.")
    elif args.command == "contract-id":
        print(matrix_contract_id(matrix))
    elif args.command == "build-field":
        value = nested_value(matrix["build_contract"], args.field)
        print(json.dumps(value) if isinstance(value, (dict, list, bool)) else value)
    elif args.command == "resolve":
        print(target_by_triple(matrix, args.target_triple)["id"])
    elif args.command == "get":
        value = nested_value(target_by_id(matrix, args.target_id), args.field)
        print(json.dumps(value) if isinstance(value, (dict, list, bool)) else value)
    elif args.command == "profile-ids":
        target = target_by_id(matrix, args.target_id)
        for profile in target["profiles"]:
            print(profile["id"])
    elif args.command == "operation":
        target = target_by_id(matrix, args.target_id)
        profile = profile_by_id(target, args.profile)
        matches = [
            entry for entry in profile["operations"] if entry["name"] == args.operation
        ]
        if len(matches) != 1:
            raise ValueError(
                f"operation {args.operation!r} is not unique in {target['id']} {profile['id']}"
            )
        value = nested_value(matches[0], args.field)
        print(json.dumps(value) if isinstance(value, (dict, list, bool)) else value)
    elif args.command == "check-environment":
        target = target_by_id(matrix, args.target_id)
        validate_environment(matrix, target)
        print(f"Toolchain and environment match {target['id']}.")
    elif args.command == "validate-build":
        target, profile = validate_build_arguments(matrix, args)
        print(target["id"])
        print(profile["id"])
        print(target["certification_scope"])
    else:
        write_evidence(args, matrix)


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
