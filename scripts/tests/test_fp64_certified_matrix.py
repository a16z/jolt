import copy
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "fp64_certified_matrix.py"
SPEC = importlib.util.spec_from_file_location("fp64_certified_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


class Fp64CertifiedMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = MATRIX.load_matrix()

    @staticmethod
    def linux_x86_args(**overrides: object) -> object:
        values = {
            "target_triple": "x86_64-unknown-linux-gnu",
            "architecture": "x86_64",
            "vendor": "unknown",
            "target_os": "linux",
            "target_env": "gnu",
            "endian": "little",
            "pointer_width": 64,
            "target_features": "fxsr,sse,sse2",
            "profile": "release",
            "opt_level": "3",
            "debug": "true",
            "rustc": "rustc",
        }
        values.update(overrides)
        return type("Args", (), values)()

    def build_environment(self, **overrides: str) -> dict[str, str]:
        values = {
            MATRIX.MATRIX_CONTRACT_ENV: MATRIX.matrix_contract_id(self.matrix),
        }
        values.update(overrides)
        return values

    def test_checked_in_matrix_is_valid(self) -> None:
        MATRIX.validate_matrix(self.matrix)

    def test_unknown_target_triple_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MATRIX.target_by_triple(self.matrix, "x86_64-pc-windows-msvc")

    def test_registered_targets_are_full_build_identities(self) -> None:
        expected = {
            "aarch64-apple-darwin",
            "aarch64-unknown-linux-gnu",
            "x86_64-unknown-linux-gnu",
            "x86_64-apple-darwin-inspection-only",
        }
        self.assertEqual(
            {target["id"] for target in self.matrix["targets"]},
            expected,
        )

    def test_only_complete_targets_are_required_in_ci(self) -> None:
        for target in self.matrix["targets"]:
            if target["ci_required"]:
                self.assertEqual(
                    target["certification_scope"],
                    "complete-inspection-symbol",
                )
        darwin_x86 = MATRIX.target_by_id(
            self.matrix, "x86_64-apple-darwin-inspection-only"
        )
        self.assertFalse(darwin_x86["ci_required"])
        self.assertEqual(darwin_x86["wrapper_policy"], "darwin-frame")

    def test_bmi2_profile_is_multiplication_only(self) -> None:
        linux_x86 = MATRIX.target_by_id(
            self.matrix, "x86_64-unknown-linux-gnu"
        )
        bmi2 = MATRIX.profile_by_id(linux_x86, "bmi2-mul")
        self.assertEqual(
            [operation["name"] for operation in bmi2["operations"]],
            ["mul"],
        )

    def test_target_feature_mismatch_is_rejected(self) -> None:
        args = self.linux_x86_args(target_features="avx2,fxsr,sse,sse2")
        with self.assertRaisesRegex(ValueError, "no registered feature profile"):
            MATRIX.validate_build_arguments(self.matrix, args)

    def test_build_script_rejects_profile_environment_overrides(self) -> None:
        args = self.linux_x86_args()
        environment = self.build_environment(CARGO_PROFILE_RELEASE_LTO="off")
        with self.assertRaisesRegex(ValueError, "ambient code-generation overrides"):
            MATRIX.validate_build_arguments(self.matrix, args, environment)

    def test_build_script_requires_profile_rustflags(self) -> None:
        args = self.linux_x86_args(
            target_features="bmi2,fxsr,sse,sse2",
        )
        with self.assertRaisesRegex(ValueError, "Rust flags differ"):
            MATRIX.validate_build_arguments(
                self.matrix,
                args,
                self.build_environment(),
            )

    def test_build_script_requires_current_runner_contract(self) -> None:
        args = self.linux_x86_args()
        with self.assertRaisesRegex(ValueError, "check-fp64.sh"):
            MATRIX.validate_build_arguments(self.matrix, args, {})

    def test_codegen_overrides_are_rejected(self) -> None:
        environment = {
            "RUSTFLAGS": "-C target-cpu=native",
            "CARGO_PROFILE_RELEASE_LTO": "off",
            "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS": "-C target-feature=+avx2",
        }
        self.assertEqual(
            MATRIX.codegen_overrides(self.matrix, environment),
            environment,
        )

    def test_duplicate_artifact_labels_are_rejected(self) -> None:
        current = Path(__file__)
        with self.assertRaisesRegex(ValueError, "duplicate evidence artifact"):
            MATRIX.artifact_arguments([f"log={current}", f"log={current}"])

    def test_duplicate_target_id_is_rejected(self) -> None:
        duplicate = copy.deepcopy(self.matrix)
        duplicate["targets"].append(copy.deepcopy(duplicate["targets"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate target ids"):
            MATRIX.validate_matrix(duplicate)

    def test_checked_in_release_profile_drift_is_rejected(self) -> None:
        drifted = copy.deepcopy(self.matrix)
        drifted["build_contract"]["release_profile"]["lto"] = "off"
        with self.assertRaisesRegex(ValueError, "lto"):
            MATRIX.validate_matrix(drifted)

    def test_complete_target_cannot_skip_ci(self) -> None:
        drifted = copy.deepcopy(self.matrix)
        target = MATRIX.target_by_id(drifted, "aarch64-unknown-linux-gnu")
        target["ci_required"] = False
        with self.assertRaisesRegex(ValueError, "must run in CI"):
            MATRIX.validate_matrix(drifted)

    def test_unimplemented_profile_is_rejected(self) -> None:
        drifted = copy.deepcopy(self.matrix)
        target = MATRIX.target_by_id(drifted, "x86_64-unknown-linux-gnu")
        extra = copy.deepcopy(MATRIX.profile_by_id(target, "baseline"))
        extra["id"] = "unbuilt-avx2"
        target["profiles"].append(extra)
        with self.assertRaisesRegex(ValueError, "profiles must be exactly"):
            MATRIX.validate_matrix(drifted)

if __name__ == "__main__":
    unittest.main()
