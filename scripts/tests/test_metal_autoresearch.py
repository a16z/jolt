import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "metal_autoresearch.py"
SPEC = importlib.util.spec_from_file_location("metal_autoresearch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
metal_autoresearch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metal_autoresearch)


class MetalAutoresearchTests(unittest.TestCase):
    def test_result_parser_uses_last_schema_record(self) -> None:
        output = "compile noise\n{\"schema_version\": 1, \"metrics\": {\"x\": 2}}\n"
        self.assertEqual(metal_autoresearch.parse_result(output)["metrics"]["x"], 2)

    def test_snapshot_restores_discarded_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "shader.metal"
            source.write_text("baseline")
            snapshot = root / "snapshots" / "baseline"
            metal_autoresearch.snapshot_paths(root, ["shader.metal"], snapshot)
            source.write_text("candidate")
            metal_autoresearch.restore_snapshot(root, ["shader.metal"], snapshot)
            self.assertEqual(source.read_text(), "baseline")

    def test_run_digest_rejects_contract_edits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = {"baseline": {"metric_median": 1.0}}
            encoded = metal_autoresearch.canonical_json(config)
            (run_dir / "run.json").write_bytes(encoded)
            (run_dir / "run.sha256").write_text(metal_autoresearch.sha256(encoded) + "\n")
            (run_dir / "events.jsonl").write_text("")
            metal_autoresearch.load_run(run_dir)
            (run_dir / "run.json").write_text(json.dumps({"baseline": {"metric_median": 2.0}}))
            with self.assertRaisesRegex(ValueError, "changed after initialization"):
                metal_autoresearch.load_run(run_dir)

    def test_grouped_medians_match_candidate_repetition_shape(self) -> None:
        values = [1.0, 10.0, 2.0, 4.0, 3.0, 5.0]
        self.assertEqual(metal_autoresearch.grouped_medians(values, 3), [2.0, 4.0])

    def test_append_event_writes_one_durable_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "events.jsonl"
            ledger.touch()
            event = {"index": 1, "trial_id": "trial-001"}
            metal_autoresearch.append_event(ledger, event)
            self.assertEqual(ledger.read_text().splitlines(), [json.dumps(event, sort_keys=True)])


if __name__ == "__main__":
    unittest.main()
