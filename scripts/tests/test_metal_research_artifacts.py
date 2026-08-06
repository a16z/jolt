import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.metal_research.artifacts import (
    MAX_OUTER_SOURCE_BYTES,
    materialize_runtime_artifact_context,
    materialize_outer_artifact,
    outer_dispatch_from_params,
    runtime_artifact_result_adapter,
    validate_runtime_artifact_output,
    validate_runtime_artifact_contract,
    verify_runtime_artifact_context,
    verify_outer_artifact,
)


PARAMS = {
    "JOLT_METAL_OUTER_REMAINDER_MATERIALIZE_THREADS": "256",
    "JOLT_METAL_OUTER_REMAINDER_TRANSITION_THREADS": "128",
    "JOLT_METAL_OUTER_REMAINDER_OUTPUT_THREADS": "256",
    "JOLT_METAL_OUTER_REMAINDER_CUTOFF_LOG2": "16",
    "JOLT_METAL_OUTER_REMAINDER_TRACE_CUTOFF_LOG2": "18",
}
DISPATCH = outer_dispatch_from_params(PARAMS)


def prepare_store(root: Path) -> None:
    (root / "artifacts").mkdir()


class OuterArtifactTests(unittest.TestCase):
    def test_runtime_handler_dispatch_rejects_unknown_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = {
                "runtime_artifact": {
                    "kind": "unknown_v1",
                    "tier_id": "screen",
                }
            }
            with self.assertRaisesRegex(ValueError, "unsupported"):
                validate_runtime_artifact_contract(
                    root, template["runtime_artifact"], set(), {}, {}
                )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                runtime_artifact_result_adapter(template["runtime_artifact"])
            with self.assertRaisesRegex(ValueError, "unsupported"):
                materialize_runtime_artifact_context(
                    root,
                    template["runtime_artifact"],
                    root,
                    {},
                    root,
                    {},
                )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                verify_runtime_artifact_context(
                    root, "unknown_v1", {"kind": "unknown_v1"}
                )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                validate_runtime_artifact_output(
                    {}, "unknown_v1", {"kind": "unknown_v1"}
                )
            with self.assertRaisesRegex(ValueError, "context is missing"):
                verify_runtime_artifact_context(
                    root, "outer_msl_v1", None
                )
            with self.assertRaisesRegex(ValueError, "context is missing"):
                validate_runtime_artifact_output(
                    {}, "outer_msl_v1", None
                )
            verify_runtime_artifact_context(root, None, None)
            validate_runtime_artifact_output({}, None, None)

    def test_materializes_and_reuses_a_content_addressed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_store(root)
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")

            first = materialize_outer_artifact(
                root, source, "b_only_v1", DISPATCH
            )
            second = materialize_outer_artifact(
                root, source, "b_only_v1", DISPATCH
            )

            self.assertEqual(first, second)
            self.assertEqual(
                Path(first["artifact_path"]).name, first["artifact_sha256"]
            )
            self.assertEqual(
                first["manifest"]["binding_plan"], "b_only_v1"
            )

    def test_padded_56_plan_is_content_addressed_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_store(root)
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")

            legacy = materialize_outer_artifact(
                root, source, "b_only_v1", DISPATCH
            )
            padded = materialize_outer_artifact(
                root, source, "b_only_padded_56_v1", DISPATCH
            )

            self.assertNotEqual(
                legacy["artifact_sha256"], padded["artifact_sha256"]
            )
            self.assertEqual(
                padded["manifest"]["binding_plan"],
                "b_only_padded_56_v1",
            )
            self.assertEqual(
                padded["manifest"]["schema"], "jolt_outer_artifact_v2"
            )
            self.assertEqual(padded["manifest"]["schema_version"], 2)
            self.assertEqual(
                padded["manifest"]["opening_layout"],
                {
                    "row_stride_words": 21,
                    "source_row_words": 20,
                    "tile_rows": 56,
                    "uses_shard_sums": False,
                },
            )
            self.assertEqual(
                padded["manifest"]["required_entrypoints"]["opening"],
                "solinas_outer_remainder_opening_tiles_padded_56",
            )
            self.assertNotEqual(
                legacy["manifest"]["opening_layout_sha256"],
                padded["manifest"]["opening_layout_sha256"],
            )
            self.assertNotEqual(
                legacy["manifest"]["outer_abi_sha256"],
                padded["manifest"]["outer_abi_sha256"],
            )

            artifact_dir = root / padded["artifact_path"]
            manifest_path = artifact_dir / "manifest.json"
            manifest = copy.deepcopy(padded["manifest"])
            manifest["opening_layout"]["tile_rows"] = 55
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_outer_artifact(artifact_dir)

    def test_runtime_output_is_bound_to_manifest_telemetry_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_store(root)
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")
            artifact = materialize_outer_artifact(
                root, source, "b_only_v1", DISPATCH
            )
            context = {
                "kind": "outer_msl_v1",
                "parent": artifact,
                "candidate": copy.deepcopy(artifact),
            }
            manifest = artifact["manifest"]
            arm = {"tail_elements": DISPATCH["cpu_tail_elements"]}
            output = {
                "schema": "outer_remainder_successor_v2",
                "fingerprint": {
                    "parent_artifact_sha256": artifact["artifact_sha256"],
                    "candidate_artifact_sha256": artifact[
                        "artifact_sha256"
                    ],
                },
                "telemetry": {
                    "parent_binding_plan": manifest["binding_plan"],
                    "candidate_binding_plan": manifest["binding_plan"],
                    "parent_source_sha256": manifest[
                        "outer_source_sha256"
                    ],
                    "candidate_source_sha256": manifest[
                        "outer_source_sha256"
                    ],
                },
                "excluded_warmup": {
                    "parent": copy.deepcopy(arm),
                    "candidate": copy.deepcopy(arm),
                },
                "samples": [],
            }

            validate_runtime_artifact_output(
                output, "outer_msl_v1", context
            )
            for field in (
                "candidate_binding_plan",
                "candidate_source_sha256",
            ):
                tampered = copy.deepcopy(output)
                tampered["telemetry"][field] = "tampered"
                with self.assertRaisesRegex(ValueError, "telemetry"):
                    validate_runtime_artifact_output(
                        tampered, "outer_msl_v1", context
                    )

    def test_runtime_output_accepts_a_closed_legacy_to_padded_plan_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_store(root)
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")
            parent = materialize_outer_artifact(
                root, source, "b_only_v1", DISPATCH
            )
            candidate = materialize_outer_artifact(
                root, source, "b_only_padded_56_v1", DISPATCH
            )
            context = {
                "kind": "outer_msl_v1",
                "parent": parent,
                "candidate": candidate,
            }
            arm = {"tail_elements": DISPATCH["cpu_tail_elements"]}
            output = {
                "schema": "outer_remainder_successor_v2",
                "fingerprint": {
                    "parent_artifact_sha256": parent["artifact_sha256"],
                    "candidate_artifact_sha256": candidate["artifact_sha256"],
                },
                "telemetry": {
                    "parent_binding_plan": "b_only_v1",
                    "candidate_binding_plan": "b_only_padded_56_v1",
                    "parent_source_sha256": parent["manifest"][
                        "outer_source_sha256"
                    ],
                    "candidate_source_sha256": candidate["manifest"][
                        "outer_source_sha256"
                    ],
                },
                "excluded_warmup": {
                    "parent": copy.deepcopy(arm),
                    "candidate": copy.deepcopy(arm),
                },
                "samples": [],
            }

            validate_runtime_artifact_output(output, "outer_msl_v1", context)
            tampered = copy.deepcopy(output)
            tampered["excluded_warmup"]["candidate"]["tail_elements"] //= 2
            with self.assertRaisesRegex(ValueError, "CPU tail"):
                validate_runtime_artifact_output(
                    tampered, "outer_msl_v1", context
                )

    def test_rejects_the_retired_split_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_store(root)
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")

            with self.assertRaisesRegex(ValueError, "unsupported"):
                materialize_outer_artifact(
                    root, source, "split_ab_v1", DISPATCH
                )

    def test_rejects_symlink_nul_oversize_and_unknown_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_store(root)
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")
            symlink = root / "candidate-link.metal"
            symlink.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "regular file"):
                materialize_outer_artifact(
                    root, symlink, "b_only_v1", DISPATCH
                )

            source.write_bytes(b"kernel\0void candidate() {}")
            with self.assertRaisesRegex(ValueError, "NUL"):
                materialize_outer_artifact(
                    root, source, "b_only_v1", DISPATCH
                )

            source.write_bytes(b"x" * (MAX_OUTER_SOURCE_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "size"):
                materialize_outer_artifact(
                    root, source, "b_only_v1", DISPATCH
                )

            source.write_text("kernel void candidate() {}")
            with self.assertRaisesRegex(ValueError, "unsupported"):
                materialize_outer_artifact(
                    root, source, "arbitrary_host_plan", DISPATCH
                )

            source.write_text('# include "/tmp/unsealed.metal"')
            with self.assertRaisesRegex(
                ValueError, "external source|canonical source form"
            ):
                materialize_outer_artifact(
                    root, source, "b_only_v1", DISPATCH
                )

            for external in (
                '#inc\\\nlude "/tmp/unsealed.metal"',
                '#inc/**/lude "/tmp/unsealed.metal"',
                '#if __has_include("/tmp/unsealed.metal")',
                '%:include "/tmp/unsealed.metal"',
                '??=include "/tmp/unsealed.metal"',
                '\ufeff#include "/tmp/unsealed.metal"',
                'kernel void candidate() {}\r#include "/tmp/unsealed.metal"',
                (
                    'constant char *a = "/*";\n'
                    '#include <unsealed.metal>\n'
                    'constant char *b = "*/";'
                ),
                "// /*\n#include <unsealed.metal>\n// */",
                "#inc/\\\n*comment*/lude <unsealed.metal>",
            ):
                source.write_text(external)
                with self.assertRaisesRegex(
                    ValueError,
                    "external source|trigraphs|canonical source form",
                ):
                    materialize_outer_artifact(
                        root, source, "b_only_v1", DISPATCH
                    )

    def test_verifier_rejects_manifest_and_path_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_store(root)
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")
            record = materialize_outer_artifact(
                root, source, "b_only_v1", DISPATCH
            )
            artifact_dir = root / record["artifact_path"]
            manifest_path = artifact_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["binding_plan_sha256"] = "0" * 64
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_outer_artifact(artifact_dir)

            manifest_path.write_bytes(
                json.dumps(
                    record["manifest"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            renamed = artifact_dir.with_name("0" * 64)
            artifact_dir.rename(renamed)
            with self.assertRaisesRegex(ValueError, "directory does not match"):
                verify_outer_artifact(renamed)

    def test_verifier_rejects_noncanonical_manifest_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_store(root)
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")
            record = materialize_outer_artifact(
                root, source, "b_only_v1", DISPATCH
            )
            artifact_dir = root / record["artifact_path"]
            manifest_path = artifact_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(record["manifest"], sort_keys=True, indent=2)
            )

            with self.assertRaisesRegex(ValueError, "not canonical"):
                verify_outer_artifact(artifact_dir)

    def test_verifier_rejects_artifact_directory_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_store(root)
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")
            record = materialize_outer_artifact(
                root, source, "b_only_v1", DISPATCH
            )
            link = root / "artifact-link"
            os.symlink(root / record["artifact_path"], link)

            with self.assertRaisesRegex(ValueError, "must be a directory"):
                verify_outer_artifact(link)

    def test_materializer_rejects_a_symlinked_artifact_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            external = root / "external"
            run_dir.mkdir()
            external.mkdir()
            (run_dir / "artifacts").symlink_to(
                external, target_is_directory=True
            )
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")

            with self.assertRaisesRegex(ValueError, "artifact store"):
                materialize_outer_artifact(
                    run_dir, source, "b_only_v1", DISPATCH
                )

    def test_materializer_does_not_recreate_a_deleted_artifact_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")

            with self.assertRaisesRegex(ValueError, "cannot be read"):
                materialize_outer_artifact(
                    root, source, "b_only_v1", DISPATCH
                )

    def test_every_dispatch_parameter_changes_the_artifact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_store(root)
            source = root / "candidate.metal"
            source.write_text("kernel void candidate() {}")
            baseline = materialize_outer_artifact(
                root, source, "b_only_v1", DISPATCH
            )
            changes = {
                "JOLT_METAL_OUTER_REMAINDER_MATERIALIZE_THREADS": "512",
                "JOLT_METAL_OUTER_REMAINDER_TRANSITION_THREADS": "256",
                "JOLT_METAL_OUTER_REMAINDER_OUTPUT_THREADS": "512",
                "JOLT_METAL_OUTER_REMAINDER_CUTOFF_LOG2": "17",
                "JOLT_METAL_OUTER_REMAINDER_TRACE_CUTOFF_LOG2": "19",
            }

            for parameter, value in changes.items():
                with self.subTest(parameter=parameter):
                    changed = dict(PARAMS)
                    changed[parameter] = value
                    candidate = materialize_outer_artifact(
                        root,
                        source,
                        "b_only_v1",
                        outer_dispatch_from_params(changed),
                    )
                    self.assertNotEqual(
                        baseline["artifact_sha256"],
                        candidate["artifact_sha256"],
                    )


if __name__ == "__main__":
    unittest.main()
