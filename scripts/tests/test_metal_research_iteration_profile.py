import tempfile
import unittest
from pathlib import Path

from scripts.metal_research import iteration_profile


ROOT = Path(__file__).resolve().parents[2]


class IterationProfileTests(unittest.TestCase):
    def test_outer_closure_is_reconstructed_from_the_exact_fragments(self) -> None:
        suffix = "\n// iteration profile test nonce\n"
        offset = iteration_profile.ITERATION_PROFILE_SOLINAS_OFFSET
        closure = iteration_profile._outer_closure(ROOT, suffix, offset)
        payloads = [
            (ROOT / record["path"]).read_bytes()
            for record in closure["source_fragments"]
        ]
        prefix = f"#define SOLINAS_OFFSET {offset}u\n".encode()
        parent = prefix + b"\n".join(payloads)
        candidate = prefix + b"\n".join(
            [*payloads[:-1], payloads[-1] + suffix.encode()]
        )

        self.assertEqual(closure["parent_assembled_source_bytes"], len(parent))
        self.assertEqual(
            closure["parent_assembled_source_sha256"],
            iteration_profile.sha256(parent),
        )
        self.assertEqual(
            closure["parent_assembled_source_sha256"],
            "749e79bc85bdcf0338834edb5fc756fe3326c1463fa65dae241def4a257f95e1",
        )
        self.assertEqual(
            closure["candidate_assembled_source_sha256"],
            iteration_profile.sha256(candidate),
        )

    def test_bundle_publish_refuses_to_replace_prior_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_bytes(b"retained")

            with self.assertRaisesRegex(ValueError, "already exists"):
                iteration_profile._publish_bundle(
                    [(first, b"new"), (second, b"new")]
                )

            self.assertEqual(first.read_bytes(), b"retained")
            self.assertFalse(second.exists())

    def test_profile_output_cannot_escape_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()

            with self.assertRaisesRegex(ValueError, "within the repository"):
                iteration_profile._repository_output(
                    root, root.parent / "outside" / "profile"
                )


if __name__ == "__main__":
    unittest.main()
