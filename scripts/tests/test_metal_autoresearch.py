import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "metal_autoresearch.py"
ROOT = SCRIPT.parents[1]
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

    def test_goal_continues_below_floor_without_headroom_estimate(self) -> None:
        contract = {
            "primary_metric": {"minimum_accepted_speedup": 4.0},
            "continuation": {"minimum_projected_relative_gain": 0.05},
        }
        decision = metal_autoresearch.goal_decision(contract, 3.9, [])
        self.assertTrue(decision["continue"])
        self.assertFalse(decision["floor_met"])

    def test_goal_continues_past_floor_when_clear_headroom_remains(self) -> None:
        contract = {
            "primary_metric": {"minimum_accepted_speedup": 4.0},
            "continuation": {"minimum_projected_relative_gain": 0.05},
        }
        candidates = [
            {
                "kernel": "Booleanity",
                "current_piop_share": 0.20,
                "conservative_local_speedup": 4.0,
            }
        ]
        decision = metal_autoresearch.goal_decision(contract, 4.1, candidates)
        self.assertTrue(decision["continue"])
        self.assertTrue(decision["floor_met"])
        self.assertAlmostEqual(decision["projected_piop_speedup"], 4.1 / 0.85)

    def test_goal_can_stop_past_floor_when_remaining_gain_is_marginal(self) -> None:
        contract = {
            "primary_metric": {"minimum_accepted_speedup": 4.0},
            "continuation": {"minimum_projected_relative_gain": 0.05},
        }
        candidates = [
            {
                "kernel": "small_tail",
                "current_piop_share": 0.01,
                "conservative_local_speedup": 2.0,
            }
        ]
        decision = metal_autoresearch.goal_decision(contract, 4.2, candidates)
        self.assertFalse(decision["continue"])
        self.assertTrue(decision["floor_met"])

    def test_repository_goal_contract_is_valid_and_uncapped(self) -> None:
        contract = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/piop_goal.json"
        )
        metal_autoresearch.validate_goal_contract(contract)
        self.assertEqual(contract["primary_metric"]["minimum_accepted_speedup"], 4.0)
        self.assertFalse(contract["continuation"]["stop_at_minimum"])


if __name__ == "__main__":
    unittest.main()
