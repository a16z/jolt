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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PINNED_CARGO_FUZZ = "cargo-fuzz 0.13.2"
PINNED_NIGHTLY = "nightly-2026-07-20"
REQUIRED_TOOLCHAIN_COMPONENTS = frozenset(("llvm-tools-preview", "rust-src"))


class FuzzConfigurationError(RuntimeError):
    """The checked-in fuzz configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class FuzzWorkspace:
    name: str
    directory: Path
    targets: tuple[str, ...]

    def relative_directory(self, root: Path) -> str:
        return self.directory.relative_to(root).as_posix()


def repository_root() -> Path:
    override = os.environ.get("JOLT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


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

        target_names = []
        for target in re.split(r"(?m)^\[\[bin\]\]\s*$", manifest)[1:]:
            name_match = re.search(r'(?m)^name\s*=\s*"([^"]+)"\s*$', target)
            path_match = re.search(r'(?m)^path\s*=\s*"([^"]+)"\s*$', target)
            if name_match is None or path_match is None:
                raise FuzzConfigurationError(
                    f"{manifest_path.relative_to(root)} has a [[bin]] without "
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
            raise FuzzConfigurationError(
                f"{manifest_path.relative_to(root)} has no fuzz targets"
            )
        if len(target_names) != len(set(target_names)):
            raise FuzzConfigurationError(
                f"{manifest_path.relative_to(root)} has duplicate target names"
            )

        workspaces.append(
            FuzzWorkspace(
                name=manifest_path.parent.parent.name,
                directory=manifest_path.parent,
                targets=tuple(sorted(target_names)),
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
        seed_directory = workspace.directory / "seeds" / target
        if not seed_directory.is_dir() or not any(
            path.is_file() and not path.name.startswith(".")
            for path in seed_directory.iterdir()
        ):
            raise FuzzConfigurationError(
                f"{workspace.name}/{target} has no checked-in seed inputs"
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


def target_names(
    workspace: FuzzWorkspace, requested_target: str | None
) -> tuple[str, ...]:
    if requested_target is None:
        return workspace.targets
    if requested_target not in workspace.targets:
        choices = ", ".join(workspace.targets)
        raise FuzzConfigurationError(
            f"{workspace.name} has no target {requested_target!r}; expected: {choices}"
        )
    return (requested_target,)


def run_for_targets(
    root: Path,
    workspaces: Sequence[FuzzWorkspace],
    args: argparse.Namespace,
) -> None:
    check_cargo_fuzz_version()
    failures = []

    for workspace in workspaces:
        check_workspace(root, workspace, resolve=True)
        for target in target_names(workspace, args.target):
            if args.command == "replay":
                files = seed_and_regression_files(workspace, target)
                if not files:
                    raise FuzzConfigurationError(
                        f"{workspace.name}/{target} has no seeds or regressions"
                    )
                command = ["cargo", "fuzz", "run", target]
                command.extend(str(path) for path in files)
            elif args.command == "reproduce":
                command = ["cargo", "fuzz", "run", target, str(args.input)]
            elif args.command == "tmin":
                command = ["cargo", "fuzz", "tmin", target, str(args.input)]
            elif args.command == "run":
                corpus = workspace.directory / "corpus" / target
                artifacts = workspace.directory / "artifacts" / target
                corpus.mkdir(parents=True, exist_ok=True)
                artifacts.mkdir(parents=True, exist_ok=True)
                command = ["cargo", "fuzz", "run", target, str(corpus)]
                command.extend(
                    str(path)
                    for path in corpus_directories(workspace, target)
                    if path != corpus
                )
                command.extend(
                    [
                        "--",
                        f"-max_total_time={args.seconds}",
                        f"-max_len={args.max_len}",
                        f"-timeout={args.timeout}",
                        f"-rss_limit_mb={args.rss_limit_mb}",
                        f"-artifact_prefix={artifacts}{os.sep}",
                        "-print_final_stats=1",
                        "-use_value_profile=1",
                    ]
                )
            elif args.command == "cmin":
                corpus = workspace.directory / "corpus" / target
                if not corpus.is_dir() or not any(corpus.iterdir()):
                    print(f"Skipping empty corpus for {workspace.name}/{target}")
                    continue
                command = ["cargo", "fuzz", "cmin", target, str(corpus)]
            elif args.command == "coverage":
                directories = corpus_directories(workspace, target)
                if not directories:
                    raise FuzzConfigurationError(
                        f"{workspace.name}/{target} has no corpus inputs"
                    )
                command = ["cargo", "fuzz", "coverage", target]
                command.extend(str(path) for path in directories)
            else:
                raise AssertionError(f"unsupported target command {args.command}")

            status = run_command(command, cwd=workspace.directory)
            if status != 0:
                failures.append(f"{workspace.name}/{target} (exit {status})")
                if args.command in ("run", "replay"):
                    artifacts = workspace.directory / "artifacts" / target
                    print(
                        f"replay a failing input with: python3 scripts/fuzz.py "
                        f"--workspace {workspace.name} --target {target} "
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
                "targets": list(workspace.targets),
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
            }
            for workspace in workspaces
        ]
        print(json.dumps({"include": include}, separators=(",", ":")))
        return

    total = 0
    for workspace in workspaces:
        targets = ", ".join(workspace.targets)
        print(f"{workspace.name} ({len(workspace.targets)}): {targets}")
        total += len(workspace.targets)
    print(f"{len(workspaces)} workspaces, {total} targets")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        help="operate on one crate name instead of every fuzz workspace",
    )
    parser.add_argument("--target", help="operate on one target in the selected workspace")
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
    subparsers.add_parser("build", help="build every selected fuzz target")
    subparsers.add_parser("replay", help="replay checked-in seeds and regressions")
    reproduce = subparsers.add_parser("reproduce", help="replay one artifact")
    reproduce.add_argument("input", type=Path)
    tmin = subparsers.add_parser("tmin", help="minimize one failing artifact in place")
    tmin.add_argument("input", type=Path)

    run = subparsers.add_parser("run", help="run coverage-guided fuzzing")
    run.add_argument("--seconds", type=int, default=30)
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
            print(
                f"Validated {len(workspaces)} fuzz workspaces with "
                f"{sum(len(workspace.targets) for workspace in workspaces)} targets"
            )
        elif args.command == "build":
            check_cargo_fuzz_version()
            failures = []
            for workspace in workspaces:
                check_workspace(root, workspace, resolve=True)
                command = ["cargo", "fuzz", "build"]
                if args.target is not None:
                    target_names(workspace, args.target)
                    command.append(args.target)
                status = run_command(command, cwd=workspace.directory)
                if status != 0:
                    failures.append(f"{workspace.name} (exit {status})")
            if failures:
                raise RuntimeError("fuzz builds failed: " + ", ".join(failures))
        else:
            if args.command == "run":
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
