import copy
import os
import tempfile
import unittest
from pathlib import Path

from scripts.metal_research import binaries


class SealedBinaryTests(unittest.TestCase):
    def contract(self) -> dict[str, object]:
        return {
            "build": {
                "command": ["cargo", "build", "--example", "runner"],
                "output_path": "target/runner",
                "timeout_seconds": 30,
            },
            "source_paths": ["src/main.rs"],
            "consumer_tiers": ["screen"],
            "result_fingerprint": ["fingerprint", "runner_binary_sha256"],
        }

    def prepare(self, root: Path) -> dict[str, object]:
        source = root / "src/main.rs"
        output = root / "target/runner"
        source.parent.mkdir(parents=True)
        output.parent.mkdir(parents=True)
        source.write_text("fn main() {}\n")
        output.write_bytes(b"sealed evaluator")
        output.chmod(0o755)
        return binaries.prepare_sealed_binary_from_output(
            root,
            "outer_remainder_eval",
            self.contract(),
            binaries.declared_source_sha256(root, ["src/main.rs"]),
            "0" * 64,
        )

    def test_materializes_and_verifies_content_addressed_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            run_dir = Path(directory) / "run"
            root.mkdir()
            (run_dir / "binaries").mkdir(parents=True)
            prepared = self.prepare(root)

            record = binaries.materialize_sealed_binary(run_dir, prepared)
            runner = binaries.verify_sealed_binary_record(
                run_dir, "outer_remainder_eval", record
            )

            self.assertEqual(runner.read_bytes(), b"sealed evaluator")
            self.assertEqual(record["manifest"]["binary_bytes"], 16)
            self.assertTrue(os.access(runner, os.X_OK))
            self.assertFalse(runner.stat().st_mode & 0o222)
            self.assertFalse(runner.parent.stat().st_mode & 0o222)

    def test_rejects_manifest_binary_and_directory_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            run_dir = Path(directory) / "run"
            root.mkdir()
            (run_dir / "binaries").mkdir(parents=True)
            record = binaries.materialize_sealed_binary(
                run_dir, self.prepare(root)
            )
            artifact = run_dir / record["artifact_path"]
            artifact.chmod(0o755)
            runner = artifact / "runner"
            runner.chmod(0o755)
            runner.write_bytes(b"changed evaluator")
            runner.chmod(0o555)
            artifact.chmod(0o555)

            with self.assertRaisesRegex(ValueError, "manifest"):
                binaries.verify_sealed_binary_record(
                    run_dir, "outer_remainder_eval", record
                )

            artifact.chmod(0o755)
            (artifact / "extra").write_text("unexpected")
            artifact.chmod(0o555)
            with self.assertRaisesRegex(ValueError, "unexpected files"):
                binaries.verify_sealed_binary(artifact)

    def test_record_cannot_alias_another_binary_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            run_dir = Path(directory) / "run"
            root.mkdir()
            (run_dir / "binaries").mkdir(parents=True)
            record = binaries.materialize_sealed_binary(
                run_dir, self.prepare(root)
            )

            forged = copy.deepcopy(record)
            forged["manifest"]["id"] = "other"
            with self.assertRaisesRegex(ValueError, "changed"):
                binaries.verify_sealed_binary_record(
                    run_dir, "outer_remainder_eval", forged
                )

    def test_rejects_symlinked_build_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/main.rs"
            real = root / "target/real"
            output = root / "target/runner"
            source.parent.mkdir()
            real.parent.mkdir()
            source.write_text("fn main() {}\n")
            real.write_bytes(b"evaluator")
            real.chmod(0o755)
            output.symlink_to(real)
            with self.assertRaisesRegex(ValueError, "regular file"):
                binaries.prepare_sealed_binary_from_output(
                    root,
                    "outer_remainder_eval",
                    self.contract(),
                    binaries.declared_source_sha256(root, ["src/main.rs"]),
                    "0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
