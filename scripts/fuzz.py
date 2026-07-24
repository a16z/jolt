#!/usr/bin/env python3
"""Discover and operate every cargo-fuzz workspace in the repository."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PINNED_CARGO_FUZZ = "cargo-fuzz 0.13.2"
PINNED_NIGHTLY = "nightly-2026-07-20"
REQUIRED_TOOLCHAIN_COMPONENTS = frozenset(("llvm-tools-preview", "rust-src"))
PROFILES = ("pr", "daily", "weekly")
FOCUS_VALUES = frozenset(("soundness", "correctness", "defensive"))

POLICY_TABLE = re.compile(
    r"(?ms)^\[package\.metadata\.jolt-fuzz\.targets\.(?P<name>[A-Za-z0-9_-]+)\]\s*$"
    r"(?P<body>.*?)(?=^\[|\Z)"
)


class FuzzConfigurationError(RuntimeError):
    """The checked-in fuzz configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class FuzzTarget:
    name: str
    focus: str
    cargo_features: tuple[str, ...]
    pr_seconds: int
    daily_seconds: int
    weekly_seconds: int

    def profile_seconds(self, profile: str) -> int:
        return int(getattr(self, f"{profile}_seconds"))


@dataclass(frozen=True)
class FuzzWorkspace:
    name: str
    directory: Path
    targets: tuple[FuzzTarget, ...]

    def relative_directory(self, root: Path) -> str:
        return self.directory.relative_to(root).as_posix()

    def target_names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.targets)

    def profile_runtime(self, profile: str) -> int:
        return sum(target.profile_seconds(profile) for target in self.targets)


def format_seconds(total: int) -> str:
    minutes, seconds = divmod(total, 60)
    if minutes and seconds:
        return f"{minutes}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


def repository_root() -> Path:
    override = os.environ.get("JOLT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def parse_target_policy(
    manifest_name: str, target_name: str, body: str
) -> FuzzTarget:
    focus_match = re.search(r'(?m)^focus\s*=\s*"([^"]+)"\s*$', body)
    focus = focus_match.group(1) if focus_match else None
    if focus not in FOCUS_VALUES:
        choices = ", ".join(sorted(FOCUS_VALUES))
        raise FuzzConfigurationError(
            f"{manifest_name} target {target_name!r} focus must be one of: {choices}"
        )
    feature_match = re.search(r"(?m)^cargo-features\s*=\s*(.+?)\s*$", body)
    cargo_features: tuple[str, ...] = ()
    if feature_match is not None:
        try:
            parsed_features = json.loads(feature_match.group(1))
        except json.JSONDecodeError as error:
            raise FuzzConfigurationError(
                f"{manifest_name} target {target_name!r} cargo-features must be "
                "an array of feature strings"
            ) from error
        if (
            not isinstance(parsed_features, list)
            or any(
                not isinstance(feature, str)
                or not feature
                or re.fullmatch(r"[A-Za-z0-9_./:+-]+", feature) is None
                for feature in parsed_features
            )
            or len(parsed_features) != len(set(parsed_features))
        ):
            raise FuzzConfigurationError(
                f"{manifest_name} target {target_name!r} cargo-features must be "
                "a unique array of non-empty feature strings"
            )
        cargo_features = tuple(parsed_features)
    budgets = {}
    for key in ("pr-seconds", "daily-seconds", "weekly-seconds"):
        value_match = re.search(rf"(?m)^{key}\s*=\s*(-?\d+)\s*$", body)
        if value_match is None:
            raise FuzzConfigurationError(
                f"{manifest_name} target {target_name!r} is missing or has an "
                f"invalid {key} (expected an integer)"
            )
        value = int(value_match.group(1))
        if value <= 0:
            raise FuzzConfigurationError(
                f"{manifest_name} target {target_name!r} {key} must be positive"
            )
        budgets[key] = value
    return FuzzTarget(
        name=target_name,
        focus=focus,
        cargo_features=cargo_features,
        pr_seconds=budgets["pr-seconds"],
        daily_seconds=budgets["daily-seconds"],
        weekly_seconds=budgets["weekly-seconds"],
    )


def discover_workspaces(root: Path) -> tuple[FuzzWorkspace, ...]:
    manifest_paths = sorted(
        set(root.glob("crates/*/fuzz/Cargo.toml")) | set(root.glob("*/fuzz/Cargo.toml"))
    )
    workspaces = []
    for manifest_path in manifest_paths:
        manifest = manifest_path.read_text()
        metadata = re.search(
            r"(?ms)^\[package\.metadata\]\s*$"
            r"(?P<body>.*?)(?=^\[|\Z)",
            manifest,
        )
        if metadata is None or not re.search(
            r"(?m)^cargo-fuzz\s*=\s*true\s*$", metadata.group("body")
        ):
            raise FuzzConfigurationError(
                f"{manifest_path.relative_to(root)} must set "
                "[package.metadata].cargo-fuzz = true"
            )

        manifest_name = manifest_path.relative_to(root).as_posix()
        policies = {}
        for policy in POLICY_TABLE.finditer(manifest):
            policy_name = policy.group("name")
            if policy_name in policies:
                raise FuzzConfigurationError(
                    f"{manifest_name} has duplicate policy entries for "
                    f"target {policy_name!r}"
                )
            policies[policy_name] = policy.group("body")

        target_names = []
        for target in re.split(r"(?m)^\[\[bin\]\]\s*$", manifest)[1:]:
            name_match = re.search(r'(?m)^name\s*=\s*"([^"]+)"\s*$', target)
            path_match = re.search(r'(?m)^path\s*=\s*"([^"]+)"\s*$', target)
            if name_match is None or path_match is None:
                raise FuzzConfigurationError(
                    f"{manifest_name} has a [[bin]] without "
                    "string name and path fields"
                )
            name = name_match.group(1)
            path = path_match.group(1)
            target_path = manifest_path.parent / path
            if not target_path.is_file():
                raise FuzzConfigurationError(
                    f"fuzz target {name!r} points to missing "
                    f"{target_path.relative_to(root)}"
                )
            target_names.append(name)

        if not target_names:
            raise FuzzConfigurationError(f"{manifest_name} has no fuzz targets")
        if len(target_names) != len(set(target_names)):
            raise FuzzConfigurationError(
                f"{manifest_name} has duplicate target names"
            )

        targets = []
        for name in target_names:
            body = policies.pop(name, None)
            if body is None:
                raise FuzzConfigurationError(
                    f"{manifest_name} target {name!r} has no "
                    f"[package.metadata.jolt-fuzz.targets.{name}] policy"
                )
            targets.append(parse_target_policy(manifest_name, name, body))
        if policies:
            unknown = ", ".join(sorted(policies))
            raise FuzzConfigurationError(
                f"{manifest_name} has policy entries for unknown targets: {unknown}"
            )

        workspaces.append(
            FuzzWorkspace(
                name=manifest_path.parent.parent.name,
                directory=manifest_path.parent,
                targets=tuple(sorted(targets, key=lambda target: target.name)),
            )
        )

    if not workspaces:
        raise FuzzConfigurationError(
            "no cargo-fuzz workspaces found under crates/ or the repository root"
        )
    names = [workspace.name for workspace in workspaces]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise FuzzConfigurationError(
            "duplicate fuzz workspace names: " + ", ".join(duplicates)
        )
    return tuple(sorted(workspaces, key=lambda workspace: workspace.name))


def select_workspaces(
    workspaces: Sequence[FuzzWorkspace], name: str | None
) -> tuple[FuzzWorkspace, ...]:
    if name is None:
        return tuple(workspaces)
    selected = tuple(workspace for workspace in workspaces if workspace.name == name)
    if not selected:
        choices = ", ".join(workspace.name for workspace in workspaces)
        raise FuzzConfigurationError(
            f"unknown fuzz workspace {name!r}; expected one of: {choices}"
        )
    return selected


def check_workspace(root: Path, workspace: FuzzWorkspace, resolve: bool) -> None:
    toolchain_path = workspace.directory / "rust-toolchain.toml"
    if not toolchain_path.is_file():
        raise FuzzConfigurationError(
            f"{workspace.relative_directory(root)} is missing rust-toolchain.toml"
        )
    channel_match = re.search(
        r'(?m)^channel\s*=\s*"([^"]+)"\s*$', toolchain_path.read_text()
    )
    channel = channel_match.group(1) if channel_match else None
    if channel != PINNED_NIGHTLY:
        raise FuzzConfigurationError(
            f"{toolchain_path.relative_to(root)} pins {channel!r}; "
            f"expected {PINNED_NIGHTLY!r}"
        )
    components_match = re.search(
        r"(?m)^components\s*=\s*\[(?P<components>[^\]]*)\]\s*$",
        toolchain_path.read_text(),
    )
    components = (
        set(re.findall(r'"([^"]+)"', components_match.group("components")))
        if components_match
        else set()
    )
    missing_components = REQUIRED_TOOLCHAIN_COMPONENTS - components
    if missing_components:
        missing = ", ".join(sorted(missing_components))
        raise FuzzConfigurationError(
            f"{toolchain_path.relative_to(root)} is missing components: {missing}"
        )

    lockfile = workspace.directory / "Cargo.lock"
    if not lockfile.is_file():
        raise FuzzConfigurationError(
            f"{workspace.relative_directory(root)} is missing Cargo.lock"
        )

    for target in workspace.targets:
        seed_directory = workspace.directory / "seeds" / target.name
        if not seed_directory.is_dir() or not any(
            path.is_file() and not path.name.startswith(".")
            for path in seed_directory.iterdir()
        ):
            raise FuzzConfigurationError(
                f"{workspace.name}/{target.name} has no checked-in seed inputs"
            )

    if resolve:
        run_command(
            ["cargo", "metadata", "--locked", "--format-version=1", "--no-deps"],
            cwd=workspace.directory,
            quiet=True,
        )


def check_cargo_fuzz_version() -> None:
    try:
        result = subprocess.run(
            ["cargo", "fuzz", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise FuzzConfigurationError(
            "cargo-fuzz is not installed; install it with "
            f"`cargo install cargo-fuzz --version {PINNED_CARGO_FUZZ.split()[-1]} "
            "--locked`"
        ) from error
    version = result.stdout.strip()
    if result.returncode != 0:
        raise FuzzConfigurationError(
            "cargo-fuzz is not installed; install it with "
            f"`cargo install cargo-fuzz --version {PINNED_CARGO_FUZZ.split()[-1]} --locked`"
        )
    if version != PINNED_CARGO_FUZZ:
        raise FuzzConfigurationError(
            f"installed {version!r}; expected {PINNED_CARGO_FUZZ!r}"
        )


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    quiet: bool = False,
) -> int:
    if not quiet:
        rendered = shlex.join(command)
        print(f"[{cwd}] $ {rendered}", flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.DEVNULL if quiet else None,
    )
    return result.returncode


def seed_and_regression_files(workspace: FuzzWorkspace, target: str) -> tuple[Path, ...]:
    files = []
    for directory_name in ("seeds", "regressions"):
        directory = workspace.directory / directory_name / target
        if directory.is_dir():
            files.extend(
                path
                for path in sorted(directory.iterdir())
                if path.is_file() and not path.name.startswith(".")
            )
    return tuple(files)


def corpus_directories(workspace: FuzzWorkspace, target: str) -> tuple[Path, ...]:
    directories = []
    for directory_name in ("corpus", "seeds", "regressions"):
        directory = workspace.directory / directory_name / target
        if directory.is_dir() and any(
            path.is_file() and not path.name.startswith(".")
            for path in directory.iterdir()
        ):
            directories.append(directory)
    return tuple(directories)


def selected_targets(
    workspace: FuzzWorkspace, requested_target: str | None
) -> tuple[FuzzTarget, ...]:
    if requested_target is None:
        return workspace.targets
    for target in workspace.targets:
        if target.name == requested_target:
            return (target,)
    choices = ", ".join(workspace.target_names())
    raise FuzzConfigurationError(
        f"{workspace.name} has no target {requested_target!r}; expected: {choices}"
    )


def cargo_feature_args(target: FuzzTarget) -> list[str]:
    if not target.cargo_features:
        return []
    return ["--features", ",".join(target.cargo_features)]


def cargo_target_args(args: argparse.Namespace) -> list[str]:
    if args.target_triple is None:
        return []
    return ["--target", args.target_triple]


def run_for_targets(
    root: Path,
    workspaces: Sequence[FuzzWorkspace],
    args: argparse.Namespace,
) -> None:
    check_cargo_fuzz_version()
    failures = []

    for workspace in workspaces:
        check_workspace(root, workspace, resolve=True)
        for target in selected_targets(workspace, args.target):
            sanitizer = ["--sanitizer", args.sanitizer]
            target_triple = cargo_target_args(args)
            features = cargo_feature_args(target)
            seconds = None
            if args.command == "replay":
                files = seed_and_regression_files(workspace, target.name)
                if not files:
                    raise FuzzConfigurationError(
                        f"{workspace.name}/{target.name} has no seeds or regressions"
                    )
                command = [
                    "cargo",
                    "fuzz",
                    "run",
                    *sanitizer,
                    *target_triple,
                    *features,
                    target.name,
                ]
                command.extend(str(path) for path in files)
            elif args.command == "reproduce":
                command = [
                    "cargo",
                    "fuzz",
                    "run",
                    *sanitizer,
                    *target_triple,
                    *features,
                    target.name,
                    str(args.input),
                ]
            elif args.command == "tmin":
                command = [
                    "cargo",
                    "fuzz",
                    "tmin",
                    *sanitizer,
                    *target_triple,
                    *features,
                    target.name,
                    str(args.input),
                ]
            elif args.command == "run":
                seconds = (
                    args.seconds
                    if args.seconds is not None
                    else target.profile_seconds(args.profile)
                )
                corpus = workspace.directory / "corpus" / target.name
                artifacts = workspace.directory / "artifacts" / target.name
                corpus.mkdir(parents=True, exist_ok=True)
                artifacts.mkdir(parents=True, exist_ok=True)
                command = [
                    "cargo",
                    "fuzz",
                    "run",
                    *sanitizer,
                    *target_triple,
                    *features,
                    target.name,
                    str(corpus),
                ]
                command.extend(
                    str(path)
                    for path in corpus_directories(workspace, target.name)
                    if path != corpus
                )
                command.extend(
                    [
                        "--",
                        f"-max_total_time={seconds}",
                        f"-max_len={args.max_len}",
                        f"-timeout={args.timeout}",
                        f"-rss_limit_mb={args.rss_limit_mb}",
                        f"-artifact_prefix={artifacts}{os.sep}",
                        "-print_final_stats=1",
                        "-use_value_profile=1",
                    ]
                )
            elif args.command == "cmin":
                corpus = workspace.directory / "corpus" / target.name
                if not corpus.is_dir() or not any(corpus.iterdir()):
                    print(f"Skipping empty corpus for {workspace.name}/{target.name}")
                    continue
                command = [
                    "cargo",
                    "fuzz",
                    "cmin",
                    *sanitizer,
                    *target_triple,
                    *features,
                    target.name,
                    str(corpus),
                ]
            elif args.command == "coverage":
                directories = corpus_directories(workspace, target.name)
                if not directories:
                    raise FuzzConfigurationError(
                        f"{workspace.name}/{target.name} has no corpus inputs"
                    )
                command = [
                    "cargo",
                    "fuzz",
                    "coverage",
                    *target_triple,
                    *features,
                    target.name,
                ]
                command.extend(str(path) for path in directories)
            else:
                raise AssertionError(f"unsupported target command {args.command}")

            started = time.monotonic()
            status = run_command(command, cwd=workspace.directory)
            elapsed = time.monotonic() - started
            if args.command == "run":
                print(
                    f"[stats] {workspace.name}/{target.name}: exit={status} "
                    f"focus={target.focus} budget={seconds}s "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if status != 0:
                failures.append(f"{workspace.name}/{target.name} (exit {status})")
                if args.command in ("run", "replay"):
                    artifacts = workspace.directory / "artifacts" / target.name
                    print(
                        f"replay a failing input with: python3 scripts/fuzz.py "
                        f"--workspace {workspace.name} --target {target.name} "
                        f"reproduce <input> (run artifacts land in {artifacts})",
                        file=sys.stderr,
                    )

    if failures:
        joined = "\n  - ".join(failures)
        raise RuntimeError(f"fuzz commands failed:\n  - {joined}")


def print_inventory(
    root: Path, workspaces: Sequence[FuzzWorkspace], output_format: str
) -> None:
    if output_format == "json":
        payload = [
            {
                "workspace": workspace.name,
                "directory": workspace.relative_directory(root),
                "target_count": len(workspace.targets),
                **{
                    f"{profile}_seconds": workspace.profile_runtime(profile)
                    for profile in PROFILES
                },
                "targets": [
                    {
                        "name": target.name,
                        "focus": target.focus,
                        "cargo_features": list(target.cargo_features),
                        **{
                            f"{profile}_seconds": target.profile_seconds(profile)
                            for profile in PROFILES
                        },
                    }
                    for target in workspace.targets
                ],
            }
            for workspace in workspaces
        ]
        print(json.dumps(payload, separators=(",", ":")))
        return
    if output_format == "github":
        include = [
            {
                "workspace": workspace.name,
                "directory": workspace.relative_directory(root),
                "target_count": len(workspace.targets),
                **{
                    f"{profile}_seconds": workspace.profile_runtime(profile)
                    for profile in PROFILES
                },
            }
            for workspace in workspaces
        ]
        print(json.dumps({"include": include}, separators=(",", ":")))
        return

    total = 0
    for workspace in workspaces:
        runtimes = ", ".join(
            f"{profile} {format_seconds(workspace.profile_runtime(profile))}"
            for profile in PROFILES
        )
        print(f"{workspace.name} ({len(workspace.targets)} targets; {runtimes})")
        for target in workspace.targets:
            budgets = ", ".join(
                f"{profile} {format_seconds(target.profile_seconds(profile))}"
                for profile in PROFILES
            )
            print(f"  {target.name} [{target.focus}]: {budgets}")
        total += len(workspace.targets)
    print(f"{len(workspaces)} workspaces, {total} targets")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        help="operate on one crate name instead of every fuzz workspace",
    )
    parser.add_argument("--target", help="operate on one target in the selected workspace")
    parser.add_argument(
        "--sanitizer",
        choices=("address", "none"),
        default="address",
        help="sanitizer for target builds; macOS local builds of the "
        "arkworks-dependent workspaces fail to link under ASan (see FUZZING.md)",
    )
    parser.add_argument(
        "--target-triple",
        help="pass a cargo-fuzz --target triple; CI pins Linux ASan builds to "
        "x86_64-unknown-linux-gnu so cargo-fuzz does not select musl",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="list discovered fuzz targets")
    inventory.add_argument(
        "--format",
        choices=("table", "json", "github"),
        default="table",
        dest="output_format",
    )

    check = subparsers.add_parser("check", help="validate fuzz workspace configuration")
    check.add_argument(
        "--resolve",
        action="store_true",
        help="also require every committed lockfile to resolve unchanged",
    )
    check.add_argument(
        "--compile",
        action="store_true",
        help="also type-check every fuzz target with `cargo check --locked`",
    )
    subparsers.add_parser("build", help="build every selected fuzz target")
    subparsers.add_parser("replay", help="replay checked-in seeds and regressions")
    reproduce = subparsers.add_parser("reproduce", help="replay one artifact")
    reproduce.add_argument("input", type=Path)
    tmin = subparsers.add_parser("tmin", help="minimize one failing artifact in place")
    tmin.add_argument("input", type=Path)

    run = subparsers.add_parser("run", help="run coverage-guided fuzzing")
    run.add_argument(
        "--profile",
        choices=PROFILES,
        help="use each target's manifest-declared budget for this profile",
    )
    run.add_argument(
        "--seconds",
        type=int,
        help="override every selected target's budget with one duration",
    )
    run.add_argument("--max-len", type=int, default=4096)
    run.add_argument("--timeout", type=int, default=30)
    run.add_argument("--rss-limit-mb", type=int, default=4096)

    subparsers.add_parser("cmin", help="minimize mutable corpora")
    subparsers.add_parser("coverage", help="generate coverage data from all corpora")
    return parser


def positive(value: int, label: str) -> None:
    if value <= 0:
        raise FuzzConfigurationError(f"{label} must be positive")


def main(argv: Iterable[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    root = repository_root()

    try:
        workspaces = select_workspaces(discover_workspaces(root), args.workspace)
        if args.target is not None and len(workspaces) != 1:
            raise FuzzConfigurationError("--target requires --workspace")
        if args.target is not None and args.command in ("inventory", "check"):
            raise FuzzConfigurationError(
                f"--target is not supported by the {args.command} command"
            )
        if args.command in ("reproduce", "tmin"):
            if len(workspaces) != 1 or args.target is None:
                raise FuzzConfigurationError(
                    f"{args.command} requires --workspace and --target"
                )
            args.input = args.input.resolve()
            if not args.input.is_file():
                raise FuzzConfigurationError(
                    f"reproducer does not exist or is not a file: {args.input}"
                )

        if args.command == "inventory":
            print_inventory(root, workspaces, args.output_format)
        elif args.command == "check":
            for workspace in workspaces:
                check_workspace(root, workspace, resolve=args.resolve)
                runtimes = ", ".join(
                    f"{profile} {format_seconds(workspace.profile_runtime(profile))}"
                    for profile in PROFILES
                )
                print(f"{workspace.name}: {runtimes}")
            if args.compile:
                # `cargo metadata` never compiles a target, so configuration
                # validation alone cannot catch API bitrot in target sources.
                failures = []
                for workspace in workspaces:
                    for target in workspace.targets:
                        status = run_command(
                            [
                                "cargo",
                                "check",
                                "--locked",
                                "--quiet",
                                "--bin",
                                target.name,
                                *cargo_feature_args(target),
                            ],
                            cwd=workspace.directory,
                        )
                        if status != 0:
                            failures.append(
                                f"{workspace.name}/{target.name} (exit {status})"
                            )
                if failures:
                    raise RuntimeError(
                        "fuzz workspaces failed to compile: " + ", ".join(failures)
                    )
            print(
                f"Validated {len(workspaces)} fuzz workspaces with "
                f"{sum(len(workspace.targets) for workspace in workspaces)} targets"
            )
        elif args.command == "build":
            check_cargo_fuzz_version()
            failures = []
            for workspace in workspaces:
                check_workspace(root, workspace, resolve=True)
                for target in selected_targets(workspace, args.target):
                    command = [
                        "cargo",
                        "fuzz",
                        "build",
                        "--sanitizer",
                        args.sanitizer,
                        *cargo_target_args(args),
                        *cargo_feature_args(target),
                        target.name,
                    ]
                    status = run_command(command, cwd=workspace.directory)
                    if status != 0:
                        failures.append(
                            f"{workspace.name}/{target.name} (exit {status})"
                        )
            if failures:
                raise RuntimeError("fuzz builds failed: " + ", ".join(failures))
        else:
            if args.command == "run":
                if (args.profile is None) == (args.seconds is None):
                    raise FuzzConfigurationError(
                        "run requires exactly one of --profile or --seconds"
                    )
                if args.seconds is not None:
                    positive(args.seconds, "seconds")
                positive(args.max_len, "max-len")
                positive(args.timeout, "timeout")
                positive(args.rss_limit_mb, "rss-limit-mb")
            run_for_targets(root, workspaces, args)
    except (FuzzConfigurationError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
