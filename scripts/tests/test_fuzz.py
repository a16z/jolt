import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "fuzz.py"
SPEC = importlib.util.spec_from_file_location("jolt_fuzz", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FUZZ = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FUZZ
SPEC.loader.exec_module(FUZZ)


class FuzzInventoryTests(unittest.TestCase):
    def make_workspace(
        self,
        root: Path,
        crate: str = "sample",
        targets: tuple[str, ...] = ("alpha", "beta"),
        top_level: bool = False,
    ) -> Path:
        base = root / crate if top_level else root / "crates" / crate
        fuzz_dir = base / "fuzz"
        target_dir = fuzz_dir / "fuzz_targets"
        target_dir.mkdir(parents=True)
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
            self.assertEqual(workspaces[1].targets, ("first", "second"))

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


if __name__ == "__main__":
    unittest.main()
