import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "bytecode_cpu_eval.py"
SPEC = importlib.util.spec_from_file_location("bytecode_cpu_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bytecode_cpu_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bytecode_cpu_eval)


class BytecodeCpuEvalTests(unittest.TestCase):
    def member_events(self) -> list[dict[str, object]]:
        return [
            {
                "name": "jolt_prover::backend_witness_prepare",
                "ph": "X",
                "ts": 2.0,
                "dur": 1.0,
            },
            {"name": "jolt_prover::piop", "ph": "B", "pid": 1, "tid": 0, "ts": 10.0},
            {
                "name": "BytecodeReadRafCycle::prepare",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 11.0,
                "dur": 2.0,
            },
            {
                "name": "BytecodeReadRafCycle::prove_round",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 14.0,
                "dur": 3.0,
            },
            {
                "name": "BytecodeReadRafCycle::finish_rounds",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 18.0,
                "dur": 1.0,
            },
            {
                "name": "BytecodeReadRafCycle::output_claims",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 20.0,
                "dur": 0.5,
            },
            {"name": "jolt_prover::piop", "ph": "E", "pid": 1, "tid": 0, "ts": 25.0},
        ]

    def valid_observation(
        self, run_dir: Path, binary: Path, cell: dict[str, object]
    ) -> dict[str, object]:
        config = bytecode_cpu_eval.expected_config(cell)
        stdout = run_dir / f"{cell['id']}.stdout"
        stderr = run_dir / f"{cell['id']}.stderr"
        trace = run_dir / f"{cell['id']}.trace.json"
        stdout.write_text(
            "BYTECODE_CYCLE_CONFIG "
            + " ".join(f"{key}={value}" for key, value in config.items())
            + "\n"
            + f"modular Akita fibonacci (2^{cell['scale']}, optimized): "
            "Prover completed in 1s\n"
        )
        stderr.write_text("")
        events = [
            {
                "name": "jolt_prover::backend_witness_prepare",
                "ph": "X",
                "ts": 2.0,
                "dur": 1.0,
            },
            {"name": "jolt_prover::piop", "ph": "B", "pid": 1, "tid": 0, "ts": 10.0},
            {
                "name": "BytecodeReadRafCycle::prepare",
                "ph": "X",
                "ts": 11.0,
                "dur": 1.0,
            },
        ]
        events.extend(
            {
                "name": "BytecodeReadRafCycle::prove_round",
                "ph": "X",
                "ts": 13.0 + 2.0 * round_index,
                "dur": 1.0,
            }
            for round_index in range(int(cell["scale"]))
        )
        finish_start = 14.0 + 2.0 * int(cell["scale"])
        events.extend(
            [
                {
                    "name": "BytecodeReadRafCycle::finish_rounds",
                    "ph": "X",
                    "ts": finish_start,
                    "dur": 1.0,
                },
                {
                    "name": "BytecodeReadRafCycle::output_claims",
                    "ph": "X",
                    "ts": finish_start + 2.0,
                    "dur": 1.0,
                },
                {
                    "name": "jolt_prover::piop",
                    "ph": "E",
                    "pid": 1,
                    "tid": 0,
                    "ts": finish_start + 4.0,
                },
            ]
        )
        trace.write_text(json.dumps(events))
        breakdown = bytecode_cpu_eval.member_breakdown(events, int(cell["scale"]))
        return {
            **cell,
            "attempt": 1,
            "status": "valid",
            "exit_code": 0,
            "command": bytecode_cpu_eval.command_for(binary, cell),
            "config": config,
            "durations_us": breakdown["durations_us"],
            "span_occurrences": breakdown["occurrences"],
            "guards": {name: True for name in bytecode_cpu_eval.OBSERVATION_GUARDS},
            "artifacts": {
                name: bytecode_cpu_eval.artifact_record(run_dir, path)
                for name, path in (("stdout", stdout), ("stderr", stderr), ("trace", trace))
            },
            "error": None,
        }

    def test_parses_exact_geometry_record(self) -> None:
        config = bytecode_cpu_eval.parse_config(
            "noise\nBYTECODE_CYCLE_CONFIG requested=q10 effective=q10 "
            "log_t=26 log_k=13 chunk_bits=8 num_ra=2 degree=4\n"
        )
        self.assertEqual(
            config,
            {
                "requested": "q10",
                "effective": "q10",
                "log_t": 26,
                "log_k": 13,
                "chunk_bits": 8,
                "num_ra": 2,
                "degree": 4,
            },
        )

    def test_rejects_duplicate_geometry_records(self) -> None:
        line = (
            "BYTECODE_CYCLE_CONFIG requested=q10 effective=q10 "
            "log_t=26 log_k=13 chunk_bits=8 num_ra=2 degree=4\n"
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            bytecode_cpu_eval.parse_config(line + line)

    def test_frozen_smoke_exercises_generic_fallback(self) -> None:
        smoke = bytecode_cpu_eval.schedule()[1]
        self.assertEqual(
            bytecode_cpu_eval.expected_config(smoke),
            {
                "requested": "q10",
                "effective": "generic",
                "log_t": 22,
                "log_k": 13,
                "chunk_bits": 4,
                "num_ra": 4,
                "degree": 6,
            },
        )

    def test_member_breakdown_sums_ordered_positive_spans(self) -> None:
        result = bytecode_cpu_eval.member_breakdown(self.member_events(), 1)
        self.assertEqual(result["durations_us"]["member"], 6.5)
        self.assertEqual(result["durations_us"]["rounds"], [3.0])
        self.assertEqual(
            result["occurrences"],
            {"prepare": 1, "prove_round": 1, "finish_rounds": 1, "output_claims": 1},
        )

    def test_member_breakdown_rejects_an_outside_target_span(self) -> None:
        events = self.member_events()
        events.append(
            {"name": "BytecodeReadRafCycle::prepare", "ph": "X", "ts": 0.0, "dur": 1.0}
        )
        with self.assertRaisesRegex(ValueError, "span counts"):
            bytecode_cpu_eval.member_breakdown(events, 1)

    def test_member_breakdown_rejects_overlap(self) -> None:
        events = self.member_events()
        events[3]["ts"] = 12.0
        with self.assertRaisesRegex(ValueError, "overlap"):
            bytecode_cpu_eval.member_breakdown(events, 1)

    def test_verdict_prefers_accumulator_only_for_a_clear_extra_gain(self) -> None:
        observations = []
        for generic, q10, accum in [
            (100.0, 80.0, 70.0),
            (102.0, 81.0, 71.0),
            (98.0, 79.0, 69.0),
            (101.0, 80.5, 70.5),
            (99.0, 79.5, 69.5),
        ]:
            observations.append(
                {
                    "generic": {"member_wall_us": generic, "piop_us": 1000.0},
                    "q10": {"member_wall_us": q10, "piop_us": 1000.0},
                    "q10-accum": {"member_wall_us": accum, "piop_us": 1000.0},
                }
            )
        self.assertEqual(bytecode_cpu_eval.summarize(observations)["selected"], "q10-accum")

    def test_verdict_uses_simpler_q10_when_accumulator_delta_is_small(self) -> None:
        observations = []
        for generic, q10, accum in [
            (100.0, 80.0, 79.0),
            (102.0, 81.0, 80.0),
            (98.0, 79.0, 78.0),
            (101.0, 80.5, 79.5),
            (99.0, 79.5, 78.5),
        ]:
            observations.append(
                {
                    "generic": {"member_wall_us": generic, "piop_us": 1000.0},
                    "q10": {"member_wall_us": q10, "piop_us": 1000.0},
                    "q10-accum": {"member_wall_us": accum, "piop_us": 1000.0},
                }
            )
        self.assertEqual(bytecode_cpu_eval.summarize(observations)["selected"], "q10")

    def test_verdict_does_not_waive_accumulator_delta_when_q10_misses(self) -> None:
        observations = []
        for generic, q10, accum in [
            (100.0, 96.0, 94.0),
            (102.0, 97.9, 95.9),
            (98.0, 94.1, 92.1),
            (101.0, 97.0, 95.0),
            (99.0, 95.0, 93.0),
        ]:
            observations.append(
                {
                    "generic": {"member_wall_us": generic, "piop_us": 1000.0},
                    "q10": {"member_wall_us": q10, "piop_us": 1000.0},
                    "q10-accum": {"member_wall_us": accum, "piop_us": 1000.0},
                }
            )
        self.assertEqual(bytecode_cpu_eval.summarize(observations)["selected"], "generic")

    def test_verdict_rejects_an_incomplete_schedule(self) -> None:
        block = {
            arm: {"member_wall_us": 100.0, "piop_us": 1000.0}
            for arm in bytecode_cpu_eval.ARMS
        }
        with self.assertRaisesRegex(ValueError, "five target blocks"):
            bytecode_cpu_eval.summarize([block])

    def test_resume_requires_a_complete_block_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "block boundary"):
            bytecode_cpu_eval.validate_resume_prefix(
                Path("/tmp"), Path("/tmp/binary"), [{"status": "valid"}]
            )

    def test_resume_reconstructs_a_completed_observation_from_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory).resolve()
            binary = run_dir / "benchmark"
            binary.write_bytes(b"binary")
            cell = bytecode_cpu_eval.schedule()[0]
            observation = self.valid_observation(run_dir, binary, cell)
            bytecode_cpu_eval.validate_completed_observation(
                run_dir, binary, cell, observation
            )

            (run_dir / observation["artifacts"]["trace"]["path"]).write_text("[]")
            with self.assertRaisesRegex(ValueError, "wrong hash"):
                bytecode_cpu_eval.validate_completed_observation(
                    run_dir, binary, cell, observation
                )

    def test_resume_accepts_complete_smoke_and_rejects_next_cell_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory).resolve()
            binary = run_dir / "benchmark"
            binary.write_bytes(b"binary")
            cells = bytecode_cpu_eval.schedule()[:3]
            observations = [
                self.valid_observation(run_dir, binary, cell) for cell in cells
            ]
            completed = bytecode_cpu_eval.validate_resume_prefix(
                run_dir, binary, observations
            )
            self.assertEqual(set(completed), {cell["id"] for cell in cells})

            next_id = bytecode_cpu_eval.schedule()[3]["id"]
            (run_dir / f"{next_id}-attempt-01.stdout").write_text("")
            with self.assertRaisesRegex(ValueError, "orphaned artifacts"):
                bytecode_cpu_eval.validate_resume_prefix(run_dir, binary, observations)

    def test_contract_freezes_question_thresholds_and_schedule(self) -> None:
        contract = bytecode_cpu_eval.run_contract()
        self.assertEqual(contract["primary_metric"]["name"], "member_wall_us")
        self.assertEqual(contract["primary_metric"]["direction"], "minimize")
        self.assertEqual(contract["minimum_effects"]["q10_vs_generic"], 0.05)
        self.assertEqual(contract["minimum_effects"]["q10_accum_vs_q10"], 0.03)
        self.assertEqual(contract["budget"]["observations"], 18)
        self.assertEqual(contract["schedule"], bytecode_cpu_eval.schedule())

    def test_component_summary_reports_each_measured_member_phase(self) -> None:
        observations = [
            {
                "arm": "generic",
                "durations_us": {
                    "prepare": 10.0,
                    "rounds_total": 20.0,
                    "finish": 1.0,
                    "output_claims": 2.0,
                },
            },
            {
                "arm": "generic",
                "durations_us": {
                    "prepare": 14.0,
                    "rounds_total": 22.0,
                    "finish": 3.0,
                    "output_claims": 4.0,
                },
            },
        ]
        summary = bytecode_cpu_eval.component_summary(observations)
        self.assertEqual(summary["generic"]["prepare_ms_median"], 0.012)
        self.assertEqual(summary["generic"]["rounds_total_ms_median"], 0.021)

    def test_process_probe_fails_closed_on_pgrep_error(self) -> None:
        result = SimpleNamespace(returncode=2, stdout="", stderr="permission denied")
        with patch.object(bytecode_cpu_eval.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "process state"):
                bytecode_cpu_eval.binary_is_running(Path("/tmp/benchmark"))


if __name__ == "__main__":
    unittest.main()
