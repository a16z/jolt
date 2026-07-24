from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "fuzz.py"
SPEC = importlib.util.spec_from_file_location("jolt_fuzz", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FUZZ = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FUZZ
SPEC.loader.exec_module(FUZZ)


class FuzzInventoryTests(unittest.TestCase):
    DEFAULT_POLICY = (
        'focus = "correctness"\n'
        "pr-seconds = 30\n"
        "daily-seconds = 600\n"
        "weekly-seconds = 900\n"
    )

    def make_workspace(
        self,
        root: Path,
        crate: str = "sample",
        targets: tuple[str, ...] = ("alpha", "beta"),
        top_level: bool = False,
        policies: dict[str, str] | None = None,
    ) -> Path:
        base = root / crate if top_level else root / "crates" / crate
        fuzz_dir = base / "fuzz"
        target_dir = fuzz_dir / "fuzz_targets"
        target_dir.mkdir(parents=True)
        if policies is None:
            policies = {target: self.DEFAULT_POLICY for target in targets}
        policy_blocks = "\n".join(
            f"[package.metadata.jolt-fuzz.targets.{name}]\n{body}"
            for name, body in policies.items()
        )
        bins = "\n".join(
            (
                "[[bin]]\n"
                f'name = "{target}"\n'
                f'path = "fuzz_targets/{target}.rs"\n'
                "test = false\n"
                "doc = false\n"
                "bench = false\n"
            )
            for target in targets
        )
        (fuzz_dir / "Cargo.toml").write_text(
            "[package]\n"
            f'name = "{crate}-fuzz"\n'
            'version = "0.0.0"\n'
            'edition = "2021"\n'
            "\n"
            "[package.metadata]\n"
            "cargo-fuzz = true\n"
            "\n"
            f"{policy_blocks}\n"
            "\n"
            f"{bins}"
        )
        for target in targets:
            (target_dir / f"{target}.rs").write_text("#![no_main]\n")
        return fuzz_dir

    def make_validated_workspace(
        self,
        root: Path,
        crate: str = "sample",
        targets: tuple[str, ...] = ("alpha", "beta"),
    ) -> Path:
        fuzz_dir = self.make_workspace(root, crate, targets)
        (fuzz_dir / "rust-toolchain.toml").write_text(
            "[toolchain]\n"
            f'channel = "{FUZZ.PINNED_NIGHTLY}"\n'
            'components = ["llvm-tools-preview", "rust-src"]\n'
        )
        (fuzz_dir / "Cargo.lock").write_text("# generated\n")
        for target in targets:
            seed_dir = fuzz_dir / "seeds" / target
            seed_dir.mkdir(parents=True)
            (seed_dir / "seed").write_text("seed")
        return fuzz_dir

    def test_discovers_every_manifest_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(root, "zeta", ("second", "first"))
            self.make_workspace(root, "alpha", ("only",))

            workspaces = FUZZ.discover_workspaces(root)

            self.assertEqual([workspace.name for workspace in workspaces], ["alpha", "zeta"])
            self.assertEqual(workspaces[1].target_names(), ("first", "second"))

    def test_discovers_top_level_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(root, "nested", ("one",))
            self.make_workspace(root, "floating", ("two",), top_level=True)

            workspaces = FUZZ.discover_workspaces(root)

            self.assertEqual(
                [workspace.name for workspace in workspaces],
                ["floating", "nested"],
            )
            self.assertEqual(workspaces[0].relative_directory(root), "floating/fuzz")

    def test_rejects_duplicate_workspace_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(root, "sample", ("one",))
            self.make_workspace(root, "sample", ("two",), top_level=True)

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, "duplicate fuzz workspace names"
            ):
                FUZZ.discover_workspaces(root)

    def test_parses_focus_and_profile_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(
                root,
                targets=("alpha",),
                policies={
                    "alpha": (
                        'focus = "soundness"\n'
                        "pr-seconds = 90\n"
                        "daily-seconds = 1800\n"
                        "weekly-seconds = 5400\n"
                    )
                },
            )

            workspace = FUZZ.discover_workspaces(root)[0]
            target = workspace.targets[0]

            self.assertEqual(target.focus, "soundness")
            self.assertEqual(target.profile_seconds("pr"), 90)
            self.assertEqual(target.profile_seconds("daily"), 1800)
            self.assertEqual(target.profile_seconds("weekly"), 5400)
            self.assertEqual(workspace.profile_runtime("weekly"), 5400)

    def test_parses_per_target_cargo_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(
                root,
                targets=("alpha",),
                policies={
                    "alpha": (
                        'focus = "soundness"\n'
                        'cargo-features = ["zk", "fuzzing"]\n'
                        "pr-seconds = 30\n"
                        "daily-seconds = 600\n"
                        "weekly-seconds = 900\n"
                    )
                },
            )

            target = FUZZ.discover_workspaces(root)[0].targets[0]

            self.assertEqual(target.cargo_features, ("zk", "fuzzing"))

    def test_rejects_invalid_or_duplicate_cargo_features(self) -> None:
        invalid_values = ('["zk", "zk"]', '[""]', '"zk"', '["zk", 7]')
        for value in invalid_values:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_workspace(
                    root,
                    targets=("alpha",),
                    policies={
                        "alpha": (
                            'focus = "soundness"\n'
                            f"cargo-features = {value}\n"
                            "pr-seconds = 30\n"
                            "daily-seconds = 600\n"
                            "weekly-seconds = 900\n"
                        )
                    },
                )

                with self.assertRaisesRegex(
                    FUZZ.FuzzConfigurationError, "cargo-features"
                ):
                    FUZZ.discover_workspaces(root)

    def test_rejects_target_without_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(root, targets=("alpha",), policies={})

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, "has no .*jolt-fuzz.targets.alpha"
            ):
                FUZZ.discover_workspaces(root)

    def test_rejects_policy_for_unknown_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(
                root,
                targets=("alpha",),
                policies={
                    "alpha": self.DEFAULT_POLICY,
                    "ghost": self.DEFAULT_POLICY,
                },
            )

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, "unknown targets: ghost"
            ):
                FUZZ.discover_workspaces(root)

    def test_rejects_invalid_focus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(
                root,
                targets=("alpha",),
                policies={
                    "alpha": (
                        'focus = "important"\n'
                        "pr-seconds = 30\n"
                        "daily-seconds = 600\n"
                        "weekly-seconds = 900\n"
                    )
                },
            )

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, "focus must be one of"
            ):
                FUZZ.discover_workspaces(root)

    def test_rejects_missing_or_invalid_budget(self) -> None:
        for budget_line in ("", 'daily-seconds = "fast"\n', "daily-seconds = 3.5\n"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_workspace(
                    root,
                    targets=("alpha",),
                    policies={
                        "alpha": (
                            'focus = "defensive"\n'
                            "pr-seconds = 30\n"
                            f"{budget_line}"
                            "weekly-seconds = 900\n"
                        )
                    },
                )

                with self.assertRaisesRegex(
                    FUZZ.FuzzConfigurationError, "missing or has an invalid"
                ):
                    FUZZ.discover_workspaces(root)

    def test_rejects_non_positive_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(
                root,
                targets=("alpha",),
                policies={
                    "alpha": (
                        'focus = "defensive"\n'
                        "pr-seconds = 0\n"
                        "daily-seconds = 600\n"
                        "weekly-seconds = 900\n"
                    )
                },
            )

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, "pr-seconds must be positive"
            ):
                FUZZ.discover_workspaces(root)

    def test_check_reports_workspace_profile_runtimes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_validated_workspace(root, targets=("alpha", "beta"))
            output = StringIO()

            with (
                mock.patch.object(FUZZ, "repository_root", return_value=root),
                redirect_stdout(output),
            ):
                status = FUZZ.main(("check",))

            self.assertEqual(status, 0)
            self.assertIn("sample: pr 1m, daily 20m, weekly 30m", output.getvalue())

    def test_check_compile_type_checks_every_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.make_validated_workspace(root, "alpha", targets=("one",))
            self.make_validated_workspace(root, "beta", targets=("two",))
            commands = []

            with (
                mock.patch.object(FUZZ, "repository_root", return_value=root),
                mock.patch.object(
                    FUZZ,
                    "run_command",
                    side_effect=lambda command, **_: commands.append(command) or 0,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(FUZZ.main(("check",)), 0)
                self.assertEqual(commands, [])
                self.assertEqual(FUZZ.main(("check", "--compile")), 0)

            self.assertEqual(
                commands,
                [
                    ["cargo", "check", "--locked", "--quiet", "--bin", "one"],
                    ["cargo", "check", "--locked", "--quiet", "--bin", "two"],
                ],
            )

    def test_check_compile_reports_every_failing_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.make_validated_workspace(root, "alpha", targets=("one",))
            self.make_validated_workspace(root, "beta", targets=("two",))
            errors = StringIO()

            with (
                mock.patch.object(FUZZ, "repository_root", return_value=root),
                mock.patch.object(FUZZ, "run_command", return_value=101),
                redirect_stdout(StringIO()),
                redirect_stderr(errors),
            ):
                status = FUZZ.main(("check", "--compile"))

            self.assertEqual(status, 1)
            self.assertIn("alpha/one (exit 101)", errors.getvalue())
            self.assertIn("beta/two (exit 101)", errors.getvalue())

    def test_run_uses_manifest_budgets_and_seconds_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.make_validated_workspace(root, targets=("alpha",))
            commands = []

            with (
                mock.patch.object(FUZZ, "repository_root", return_value=root),
                mock.patch.object(FUZZ, "check_cargo_fuzz_version"),
                mock.patch.object(FUZZ, "check_workspace"),
                mock.patch.object(
                    FUZZ,
                    "run_command",
                    side_effect=lambda command, **_: commands.append(command) or 0,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(FUZZ.main(("run", "--profile", "daily")), 0)
                self.assertEqual(FUZZ.main(("run", "--seconds", "7")), 0)

            self.assertIn("-max_total_time=600", commands[0])
            self.assertIn("-max_total_time=7", commands[1])

    def test_target_features_reach_check_build_and_run_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.make_validated_workspace(root, targets=("alpha",))
            manifest = root / "crates" / "sample" / "fuzz" / "Cargo.toml"
            manifest.write_text(
                manifest.read_text().replace(
                    'focus = "correctness"\n',
                    'focus = "correctness"\ncargo-features = ["zk", "fuzzing"]\n',
                )
            )
            commands = []

            with (
                mock.patch.object(FUZZ, "repository_root", return_value=root),
                mock.patch.object(FUZZ, "check_cargo_fuzz_version"),
                mock.patch.object(FUZZ, "check_workspace"),
                mock.patch.object(
                    FUZZ,
                    "run_command",
                    side_effect=lambda command, **_: commands.append(command) or 0,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(FUZZ.main(("check", "--compile")), 0)
                self.assertEqual(FUZZ.main(("build",)), 0)
                self.assertEqual(FUZZ.main(("run", "--seconds", "1")), 0)

            for command in commands:
                self.assertIn("--features", command)
                self.assertEqual(command[command.index("--features") + 1], "zk,fuzzing")

    def test_target_triple_reaches_cargo_fuzz_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fuzz_dir = self.make_validated_workspace(root, targets=("alpha",))
            corpus_dir = fuzz_dir / "corpus" / "alpha"
            corpus_dir.mkdir(parents=True)
            (corpus_dir / "input").write_text("corpus")
            commands = []
            target_triple = "x86_64-unknown-linux-gnu"

            with (
                mock.patch.object(FUZZ, "repository_root", return_value=root),
                mock.patch.object(FUZZ, "check_cargo_fuzz_version"),
                mock.patch.object(FUZZ, "check_workspace"),
                mock.patch.object(
                    FUZZ,
                    "run_command",
                    side_effect=lambda command, **_: commands.append(command) or 0,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(
                    FUZZ.main(("--target-triple", target_triple, "build")), 0
                )
                self.assertEqual(
                    FUZZ.main(("--target-triple", target_triple, "replay")), 0
                )
                self.assertEqual(
                    FUZZ.main(
                        ("--target-triple", target_triple, "run", "--seconds", "1")
                    ),
                    0,
                )
                self.assertEqual(
                    FUZZ.main(("--target-triple", target_triple, "cmin")), 0
                )
                self.assertEqual(
                    FUZZ.main(("--target-triple", target_triple, "coverage")), 0
                )

            for command in commands:
                self.assertIn("--target", command)
                self.assertEqual(command[command.index("--target") + 1], target_triple)

    def test_run_requires_exactly_one_of_profile_or_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_validated_workspace(root, targets=("alpha",))

            with (
                mock.patch.object(FUZZ, "repository_root", return_value=root),
                redirect_stderr(StringIO()),
            ):
                self.assertEqual(FUZZ.main(("run",)), 1)
                self.assertEqual(
                    FUZZ.main(("run", "--profile", "pr", "--seconds", "5")), 1
                )

    def test_rejects_missing_target_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fuzz_dir = self.make_workspace(root)
            (fuzz_dir / "fuzz_targets" / "beta.rs").unlink()

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, "points to missing"
            ):
                FUZZ.discover_workspaces(root)

    def test_rejects_duplicate_target_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(root, targets=("duplicate", "duplicate"))

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, "duplicate target names"
            ):
                FUZZ.discover_workspaces(root)

    def test_github_matrix_contains_every_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(root, "beta", ("two", "three"))
            self.make_workspace(root, "alpha", ("one",))
            output = StringIO()

            with redirect_stdout(output):
                FUZZ.print_inventory(
                    root, FUZZ.discover_workspaces(root), output_format="github"
                )

            matrix = json.loads(output.getvalue())
            self.assertEqual(
                [entry["workspace"] for entry in matrix["include"]],
                ["alpha", "beta"],
            )
            self.assertEqual(matrix["include"][1]["target_count"], 2)

    def test_select_workspace_rejects_unknown_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(root)
            workspaces = FUZZ.discover_workspaces(root)

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, "unknown fuzz workspace"
            ):
                FUZZ.select_workspaces(workspaces, "missing")

    def test_collects_seed_and_regression_files_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fuzz_dir = self.make_workspace(root, targets=("alpha",))
            seed_dir = fuzz_dir / "seeds" / "alpha"
            regression_dir = fuzz_dir / "regressions" / "alpha"
            seed_dir.mkdir(parents=True)
            regression_dir.mkdir(parents=True)
            (seed_dir / "z-seed").write_text("z")
            (seed_dir / "a-seed").write_text("a")
            (regression_dir / "crash-1").write_text("regression")
            workspace = FUZZ.discover_workspaces(root)[0]

            files = FUZZ.seed_and_regression_files(workspace, "alpha")

            self.assertEqual(
                [path.name for path in files],
                ["a-seed", "z-seed", "crash-1"],
            )

    def test_validation_rejects_unpinned_toolchain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fuzz_dir = self.make_validated_workspace(root, targets=("alpha",))
            (fuzz_dir / "rust-toolchain.toml").write_text(
                "[toolchain]\n"
                'channel = "nightly"\n'
                'components = ["llvm-tools-preview", "rust-src"]\n'
            )
            workspace = FUZZ.discover_workspaces(root)[0]

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, f"expected '{FUZZ.PINNED_NIGHTLY}'"
            ):
                FUZZ.check_workspace(root, workspace, resolve=False)

    def test_validation_rejects_missing_lockfile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fuzz_dir = self.make_validated_workspace(root, targets=("alpha",))
            (fuzz_dir / "Cargo.lock").unlink()
            workspace = FUZZ.discover_workspaces(root)[0]

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, "missing Cargo.lock"
            ):
                FUZZ.check_workspace(root, workspace, resolve=False)

    def test_validation_rejects_missing_toolchain_component(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fuzz_dir = self.make_validated_workspace(root, targets=("alpha",))
            (fuzz_dir / "rust-toolchain.toml").write_text(
                "[toolchain]\n"
                f'channel = "{FUZZ.PINNED_NIGHTLY}"\n'
                'components = ["rust-src"]\n'
            )
            workspace = FUZZ.discover_workspaces(root)[0]

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, "llvm-tools-preview"
            ):
                FUZZ.check_workspace(root, workspace, resolve=False)

    def test_validation_rejects_missing_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fuzz_dir = self.make_validated_workspace(root, targets=("alpha",))
            (fuzz_dir / "seeds" / "alpha" / "seed").unlink()
            workspace = FUZZ.discover_workspaces(root)[0]

            with self.assertRaisesRegex(
                FUZZ.FuzzConfigurationError, "has no checked-in seed"
            ):
                FUZZ.check_workspace(root, workspace, resolve=False)

    def test_reproduce_and_tmin_invoke_cargo_fuzz_on_the_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.make_validated_workspace(root, targets=("alpha",))
            artifact = root / "crash-1"
            artifact.write_text("input")
            commands = []

            with (
                mock.patch.object(FUZZ, "repository_root", return_value=root),
                mock.patch.object(FUZZ, "check_cargo_fuzz_version"),
                mock.patch.object(FUZZ, "check_workspace"),
                mock.patch.object(
                    FUZZ,
                    "run_command",
                    side_effect=lambda command, **_: commands.append(command) or 0,
                ),
            ):
                for command in ("reproduce", "tmin"):
                    status = FUZZ.main(
                        (
                            "--workspace",
                            "sample",
                            "--target",
                            "alpha",
                            command,
                            str(artifact),
                        )
                    )
                    self.assertEqual(status, 0)

            self.assertEqual(
                commands,
                [
                    ["cargo", "fuzz", "run", "--sanitizer", "address", "alpha", str(artifact)],
                    ["cargo", "fuzz", "tmin", "--sanitizer", "address", "alpha", str(artifact)],
                ],
            )

    def test_sanitizer_override_reaches_cargo_fuzz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.make_validated_workspace(root, targets=("alpha",))
            commands = []

            with (
                mock.patch.object(FUZZ, "repository_root", return_value=root),
                mock.patch.object(FUZZ, "check_cargo_fuzz_version"),
                mock.patch.object(FUZZ, "check_workspace"),
                mock.patch.object(
                    FUZZ,
                    "run_command",
                    side_effect=lambda command, **_: commands.append(command) or 0,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(
                    FUZZ.main(("--sanitizer", "none", "run", "--seconds", "5")), 0
                )

            self.assertEqual(commands[0][:5], ["cargo", "fuzz", "run", "--sanitizer", "none"])

    def test_reproducer_commands_require_workspace_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_workspace(root, targets=("alpha",))

            with (
                mock.patch.object(FUZZ, "repository_root", return_value=root),
                redirect_stderr(StringIO()),
            ):
                status = FUZZ.main(("reproduce", "missing-artifact"))

            self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
