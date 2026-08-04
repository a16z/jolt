import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "metal_piop_eval.py"
SPEC = importlib.util.spec_from_file_location("metal_piop_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
metal_piop_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metal_piop_eval)


class MetalPiopEvalTests(unittest.TestCase):
    def test_worktree_digest_binds_untracked_paths_and_contents(self) -> None:
        first = metal_piop_eval.worktree_state_digest(
            b"diff", [(b"b.rs", b"two"), (b"a.rs", b"one")]
        )
        reordered = metal_piop_eval.worktree_state_digest(
            b"diff", [(b"a.rs", b"one"), (b"b.rs", b"two")]
        )
        changed = metal_piop_eval.worktree_state_digest(
            b"diff", [(b"a.rs", b"changed"), (b"b.rs", b"two")]
        )
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)

    def test_extracts_one_complete_piop_span(self) -> None:
        events = [
            {"name": "jolt_prover::piop", "ph": "B", "pid": 1, "tid": 0, "ts": 10.0},
            {"name": "nested", "ph": "B", "pid": 1, "tid": 0, "ts": 11.0},
            {"name": "nested", "ph": "E", "pid": 1, "tid": 0, "ts": 12.0},
            {"name": "jolt_prover::piop", "ph": "E", "pid": 1, "tid": 0, "ts": 15.5},
        ]
        self.assertEqual(metal_piop_eval.unique_span_duration_us(events), 5.5)

    def test_rejects_missing_or_ambiguous_piop_spans(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            metal_piop_eval.unique_span_duration_us([])
        events = [
            {
                "name": "jolt_prover::piop",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 10.0,
                "dur": 2.0,
            },
            {
                "name": "jolt_prover::piop",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 20.0,
                "dur": 3.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            metal_piop_eval.unique_span_duration_us(events)

    def test_speedup_is_median_of_interleaved_pairs(self) -> None:
        metrics = metal_piop_eval.summarize_pairs(
            [
                {
                    "cpu_us": 100.0,
                    "metal_us": 20.0,
                    "cpu_prepare_us": 0.0,
                    "metal_prepare_us": 5.0,
                    "cpu_instruction_ra_us": 50.0,
                    "metal_instruction_ra_us": 10.0,
                },
                {
                    "cpu_us": 120.0,
                    "metal_us": 30.0,
                    "cpu_prepare_us": 0.0,
                    "metal_prepare_us": 10.0,
                    "cpu_instruction_ra_us": 60.0,
                    "metal_instruction_ra_us": 15.0,
                },
            ]
        )
        self.assertEqual(metrics["paired_speedups"], [5.0, 4.0])
        self.assertEqual(metrics["paired_instruction_ra_speedups"], [5.0, 4.0])
        self.assertEqual(metrics["instruction_ra_speedup"], 4.5)
        self.assertEqual(metrics["piop_speedup"], 4.5)
        self.assertEqual(metrics["paired_speedups_with_backend_witness_prepare"], [4.0, 3.0])
        self.assertEqual(metrics["piop_plus_backend_witness_prepare_speedup"], 3.5)

    def test_attribution_sums_kernel_seams_inside_piop_only(self) -> None:
        events = [
            {"name": "outside::prepare", "ph": "B", "pid": 1, "tid": 0, "ts": 0.0},
            {"name": "outside::prepare", "ph": "E", "pid": 1, "tid": 0, "ts": 1.0},
            {"name": "jolt_prover::piop", "ph": "B", "pid": 1, "tid": 0, "ts": 10.0},
            {"name": "prove_stage1", "ph": "B", "pid": 1, "tid": 0, "ts": 11.0},
            {"name": "Booleanity::prepare", "ph": "B", "pid": 1, "tid": 0, "ts": 12.0},
            {"name": "Booleanity::prepare", "ph": "B", "pid": 1, "tid": 0, "ts": 12.2},
            {"name": "Booleanity::prepare", "ph": "E", "pid": 1, "tid": 0, "ts": 12.8},
            {"name": "Booleanity::prepare", "ph": "E", "pid": 1, "tid": 0, "ts": 13.0},
            {"name": "Booleanity::prove_round", "ph": "B", "pid": 1, "tid": 0, "ts": 14.0},
            {
                "name": "MetalBooleanity::resident_round",
                "ph": "B",
                "pid": 1,
                "tid": 0,
                "ts": 14.5,
            },
            {
                "name": "MetalBooleanity::resident_round",
                "ph": "E",
                "pid": 1,
                "tid": 0,
                "ts": 15.5,
            },
            {"name": "Booleanity::prove_round", "ph": "E", "pid": 1, "tid": 0, "ts": 16.0},
            {"name": "Booleanity::output_claims", "ph": "B", "pid": 1, "tid": 0, "ts": 16.0},
            {"name": "Booleanity::output_claims", "ph": "E", "pid": 1, "tid": 0, "ts": 16.5},
            {"name": "prove_stage1", "ph": "E", "pid": 1, "tid": 0, "ts": 17.0},
            {"name": "jolt_prover::piop", "ph": "E", "pid": 1, "tid": 0, "ts": 20.0},
        ]
        attribution = metal_piop_eval.trace_attribution(events)
        self.assertEqual(attribution["stage_ms"], {"prove_stage1": 0.006})
        self.assertEqual(attribution["kernels"][0]["kernel"], "Booleanity")
        self.assertEqual(attribution["kernels"][0]["wall_ms"], 0.0035)
        self.assertEqual(attribution["kernels"][0]["piop_share"], 0.35)
        self.assertEqual(
            attribution["backend_spans"],
            [
                {
                    "span": "MetalBooleanity::resident_round",
                    "wall_ms": 0.001,
                    "piop_share": 0.1,
                    "occurrences": 1,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
