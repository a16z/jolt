import importlib.util
import unittest
from pathlib import Path
from typing import Optional


SCRIPT = Path(__file__).parents[1] / "metal_piop_eval.py"
SPEC = importlib.util.spec_from_file_location("metal_piop_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
metal_piop_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metal_piop_eval)


def complete_bytecode_trace(log_n: int, backend: str) -> list[dict[str, object]]:
    def event(
        name: str,
        timestamp: float,
        duration: float,
        args: Optional[dict[str, str]] = None,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "name": name,
            "ph": "X",
            "pid": 1,
            "tid": 0,
            "ts": timestamp,
            "dur": duration,
        }
        if args is not None:
            record["args"] = args
        return record

    events = [event("jolt_prover::piop", 0.0, 10_000.0)]
    events.append(event("BytecodeReadRafCycle::prepare", 100.0, 50.0))
    round_starts = [200.0 + 100.0 * index for index in range(log_n)]
    events.extend(
        event("BytecodeReadRafCycle::prove_round", timestamp, 80.0)
        for timestamp in round_starts
    )
    finish_start = 200.0 + 100.0 * log_n
    events.append(event("BytecodeReadRafCycle::finish_rounds", finish_start, 80.0))
    events.append(
        event("BytecodeReadRafCycle::output_claims", finish_start + 100.0, 40.0)
    )
    if backend == "metal":
        dense_count = log_n - 17
        handoff_round = 2 + dense_count
        events.extend(
            [
                event("MetalBytecodeReadRafCycle::prepare", 110.0, 20.0),
                event(
                    "MetalBytecodeReadRafCycle::allocation_plan",
                    112.0,
                    10.0,
                    {
                        "device_buffers": "17",
                        "planned_device_bytes": "1000",
                        "current_device_bytes": "100",
                        "recommended_device_bytes": "2000",
                    },
                ),
                event(
                    "MetalBytecodeReadRafCycle::first_message",
                    round_starts[0] + 10.0,
                    20.0,
                ),
                event(
                    "MetalBytecodeReadRafCycle::first_bind",
                    round_starts[1] + 10.0,
                    20.0,
                ),
            ]
        )
        events.extend(
            event(
                "MetalBytecodeReadRafCycle::dense_round",
                round_starts[index + 2] + 10.0,
                20.0,
            )
            for index in range(dense_count)
        )
        events.append(
            event(
                "MetalBytecodeReadRafCycle::readback",
                round_starts[handoff_round] + 5.0,
                5.0,
                {"bytes": str(5 * (1 << 16) * 16)},
            )
        )
        events.extend(
            event(
                "MetalBytecodeReadRafCycle::cpu_tail",
                timestamp + 20.0,
                20.0,
            )
            for timestamp in round_starts[handoff_round:]
        )
        events.append(
            event("MetalBytecodeReadRafCycle::cpu_tail", finish_start + 10.0, 20.0)
        )
    return events


def complete_instruction_input_trace(
    log_n: int, backend: str, cutoff_log2: int = 16
) -> list[dict[str, object]]:
    def event(
        name: str,
        timestamp: float,
        duration: float,
        args: Optional[dict[str, object]] = None,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "name": name,
            "ph": "X",
            "pid": 1,
            "tid": 0,
            "ts": timestamp,
            "dur": duration,
        }
        if args is not None:
            record["args"] = args
        return record

    events = [
        event("jolt_prover::backend_witness_prepare", 0.0, 900.0),
        event("jolt_prover::piop", 1_000.0, 9_000.0),
        event("InstructionInput::prepare", 1_100.0, 50.0),
    ]
    if backend == "optimized":
        events.extend(
            [
                event(
                    "OptimizedInstructionInput::rows_prepare",
                    100.0,
                    400.0,
                    {
                        "cpu_rows_storage_id": "101",
                        "cpu_rows": str(1 << log_n),
                        "cpu_row_bytes": "48",
                    },
                ),
                event(
                    "OptimizedInstructionInput::rows_stage3_use",
                    1_110.0,
                    20.0,
                    {
                        "cpu_rows_storage_id": "101",
                        "cpu_rows": str(1 << log_n),
                    },
                ),
            ]
        )
    round_starts = [1_200.0 + 100.0 * index for index in range(log_n)]
    events.extend(
        event("InstructionInput::prove_round", timestamp, 80.0)
        for timestamp in round_starts
    )
    finish_start = 1_200.0 + 100.0 * log_n
    events.append(event("InstructionInput::finish_rounds", finish_start, 80.0))
    events.append(event("InstructionInput::output_claims", finish_start + 100.0, 40.0))
    if backend == "metal":
        dense_count = log_n - cutoff_log2 - 1
        handoff_round = 2 + dense_count
        host_tail_bytes = 8 * (1 << cutoff_log2) * 16
        sequence_bytes = metal_piop_eval.instruction_input_sequence_storage_bytes(log_n)
        resident_row_bytes = 160 * (1 << log_n)
        events.extend(
            [
                event(
                    "MetalInstructionInput::compact_rows_prepare",
                    50.0,
                    20.0,
                    {
                        "source_kind": "owned_random_access",
                        "witness_row_extractions": str(1 << log_n),
                        "residual_rows_written": str(1 << log_n),
                        "compact_rows_written": str(1 << log_n),
                        "compact_row_bytes": "48",
                        "residual_row_bytes": "112",
                        "compact_allocations": "1",
                        "residual_allocations": "1",
                        "full_row_allocations": "0",
                        "full_domain_copy_bytes": "0",
                        "full_domain_copy_dispatches": "0",
                        "host_repack_rows": "0",
                        "compact_rows_storage_id": "202",
                        "residual_rows_storage_id": "203",
                        "resident_rows": str(1 << log_n),
                    },
                ),
                event(
                    "MetalInstructionInput::storage_prepare",
                    100.0,
                    400.0,
                    {
                        "trace_elements": str(1 << log_n),
                        "cutoff_elements": str(1 << cutoff_log2),
                        "host_tail_bytes": str(host_tail_bytes),
                        "resident_rows_storage_id": "202",
                        "resident_rows": str(1 << log_n),
                        "resident_row_bytes": "48",
                    },
                ),
                event(
                    "MetalInstructionInput::allocation_plan",
                    200.0,
                    200.0,
                    {
                        "device_buffers": "6",
                        "planned_device_bytes": str(sequence_bytes),
                        "current_device_bytes": str(resident_row_bytes),
                        "recommended_device_bytes": str(
                            resident_row_bytes + sequence_bytes
                        ),
                    },
                ),
                event(
                    "MetalInstructionInput::storage_initialize",
                    220.0,
                    100.0,
                    {
                        "mode": "minimal",
                        "device_buffers": "6",
                        "bytes": "96",
                        "protocol_dispatches": "0",
                        **{
                            f"buffer_{index}": str(301 + index)
                            for index in range(6)
                        },
                    },
                ),
                event(
                    "MetalInstructionInput::storage_initialize_complete",
                    300.0,
                    1.0,
                    {
                        "mode": "minimal",
                        "command_completed": "true",
                        "gpu_active_ns": "50000",
                    },
                ),
                event(
                    "MetalInstructionInput::compact_rows_stage1_handoff",
                    1_020.0,
                    30.0,
                    {
                        "compact_rows_storage_id": "202",
                        "residual_rows_storage_id": "203",
                        "resident_rows": str(1 << log_n),
                        "compact_row_bytes": "48",
                        "residual_row_bytes": "112",
                        "full_domain_copy_bytes": "0",
                        "full_domain_copy_dispatches": "0",
                        "host_repack_rows": "0",
                    },
                ),
                event(
                    "MetalInstructionInput::native_primer_submit",
                    1_060.0,
                    10.0,
                    {
                        "source_elements": "64",
                        "e_in_elements": "1",
                        "e_out_elements": "32",
                        "resident_rows_storage_id": "202",
                        **{
                            f"storage_buffer_{index}": str(301 + index)
                            for index in range(6)
                        },
                        "command_committed": "true",
                        "protocol_state_advanced": "false",
                    },
                ),
                event("SpartanShift::prepare", 1_080.0, 10.0),
                event(
                    "MetalInstructionInput::prepare",
                    1_110.0,
                    20.0,
                    {
                        "resident_rows_reused": "true",
                        "round_device_buffer_allocations": "0",
                        "resident_rows_storage_id": "202",
                        "resident_rows": str(1 << log_n),
                        "storage_initialization": "minimal",
                        "storage_initialization_bytes": "96",
                        "native_primer": "async",
                        **{
                            f"storage_buffer_{index}": str(301 + index)
                            for index in range(6)
                        },
                    },
                ),
                event(
                    "MetalInstructionInput::native_primer_join",
                    round_starts[0] + 5.0,
                    2.0,
                    {
                        "source_elements": "64",
                        "e_in_elements": "1",
                        "e_out_elements": "32",
                        "resident_rows_storage_id": "202",
                        **{
                            f"storage_buffer_{index}": str(301 + index)
                            for index in range(6)
                        },
                    },
                ),
                event(
                    "MetalInstructionInput::native_primer_complete",
                    round_starts[0] + 8.0,
                    1.0,
                    {
                        "source_elements": "64",
                        "e_in_elements": "1",
                        "e_out_elements": "32",
                        "resident_rows_storage_id": "202",
                        **{
                            f"storage_buffer_{index}": str(301 + index)
                            for index in range(6)
                        },
                        "command_completed": "true",
                        "produced_zero": "true",
                        "protocol_state_advanced": "false",
                        "completed_before_join": "true",
                        "submit_wall_ns": "5000",
                        "overlap_wall_ns": "140000",
                        "join_wall_ns": "2000",
                        "lifecycle_wall_ns": "147000",
                        "gpu_active_ns": "100000",
                    },
                ),
                event(
                    "MetalInstructionInput::first_message",
                    round_starts[0] + 10.0,
                    20.0,
                ),
                event(
                    "MetalInstructionInput::first_bind",
                    round_starts[1] + 10.0,
                    20.0,
                ),
            ]
        )
        events.extend(
            event(
                "MetalInstructionInput::dense_round",
                round_starts[index + 2] + 10.0,
                20.0,
            )
            for index in range(dense_count)
        )
        events.append(
            event(
                "MetalInstructionInput::readback",
                round_starts[handoff_round] + 5.0,
                5.0,
                {"bytes": str(host_tail_bytes)},
            )
        )
        events.extend(
            event(
                "MetalInstructionInput::cpu_tail",
                round_starts[index] + 20.0,
                20.0,
            )
            for index in range(handoff_round, log_n)
        )
        events.append(
            event("MetalInstructionInput::cpu_tail", finish_start + 20.0, 20.0)
        )
    return events


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

    def test_span_args_include_fields_recorded_after_entry(self) -> None:
        events = [
            {
                "name": "resident_rows",
                "ph": "B",
                "pid": 1,
                "tid": 2,
                "ts": 10.0,
                "args": {"rows": "16"},
            },
            {
                "name": "resident_rows",
                "ph": "E",
                "pid": 1,
                "tid": 2,
                "ts": 11.0,
                "args": {"rows": "16", "storage_id": "202"},
            },
        ]
        self.assertEqual(
            metal_piop_eval.unique_span_args(events, "resident_rows"),
            {"rows": "16", "storage_id": "202"},
        )

        events[1]["args"]["rows"] = "17"
        with self.assertRaisesRegex(ValueError, "changed argument rows"):
            metal_piop_eval.unique_span_args(events, "resident_rows")

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
                    "cpu_bytecode_us": 40.0,
                    "metal_bytecode_us": 10.0,
                    "cpu_instruction_input_us": 80.0,
                    "metal_instruction_input_us": 16.0,
                },
                {
                    "cpu_us": 120.0,
                    "metal_us": 30.0,
                    "cpu_prepare_us": 0.0,
                    "metal_prepare_us": 10.0,
                    "cpu_instruction_ra_us": 60.0,
                    "metal_instruction_ra_us": 15.0,
                    "cpu_bytecode_us": 45.0,
                    "metal_bytecode_us": 15.0,
                    "cpu_instruction_input_us": 90.0,
                    "metal_instruction_input_us": 30.0,
                },
            ]
        )
        self.assertEqual(metrics["paired_speedups"], [5.0, 4.0])
        self.assertEqual(metrics["paired_instruction_ra_speedups"], [5.0, 4.0])
        self.assertEqual(metrics["instruction_ra_speedup"], 4.5)
        self.assertEqual(metrics["paired_bytecode_read_raf_cycle_speedups"], [4.0, 3.0])
        self.assertEqual(metrics["bytecode_read_raf_cycle_speedup"], 3.5)
        self.assertEqual(
            metrics["paired_instruction_input_kernel_service_speedups"],
            [5.0, 3.0],
        )
        self.assertEqual(metrics["instruction_input_kernel_service_speedup"], 4.0)
        self.assertFalse(metrics["bytecode_read_raf_cycle_decision"]["enough_pairs"])
        self.assertEqual(metrics["piop_speedup"], 4.5)
        self.assertEqual(metrics["paired_speedups_with_backend_witness_prepare"], [4.0, 3.0])
        self.assertEqual(metrics["piop_plus_backend_witness_prepare_speedup"], 3.5)

    def test_validates_target_bytecode_records(self) -> None:
        optimized = "\n".join(
            [
                "BYTECODE_CYCLE_CONFIG requested=q10 effective=q10 log_t=26 log_k=13 chunk_bits=8 num_ra=2 degree=4",
                "PROOF_VERIFIED backend=optimized value=true",
            ]
        )
        metal = "\n".join(
            [
                "BYTECODE_CYCLE_CONFIG requested=q10 effective=q10 log_t=26 log_k=13 chunk_bits=8 num_ra=2 degree=4",
                "BYTECODE_METAL_CONFIG backend=metal cpu_tail=q10 trace_cutoff=262144 cutoff=65536 message_threads=256 transition_threads=128 max_threadgroups=8192",
                "PROOF_VERIFIED backend=metal value=true",
            ]
        )
        self.assertEqual(
            metal_piop_eval.validate_bytecode_stdout(optimized, "optimized", 26)[
                "relation"
            ]["degree"],
            4,
        )
        self.assertEqual(
            metal_piop_eval.validate_bytecode_stdout(metal, "metal", 26)["relation"][
                "num_ra"
            ],
            2,
        )
        self.assertEqual(
            metal_piop_eval.validate_bytecode_stdout(metal, "metal", 26)[
                "metal_runtime"
            ]["cutoff"],
            1 << 16,
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            metal_piop_eval.validate_bytecode_stdout(metal + "\n" + metal, "metal", 26)

    def test_validates_instruction_input_runtime_record(self) -> None:
        record = "INSTRUCTION_INPUT_METAL_CONFIG backend=metal trace_cutoff=33554432 cutoff=65536 native_message_threads=256 native_transition_threads=128 dense_transition_threads=128 storage_initialization=minimal native_primer=async"
        self.assertIsNone(
            metal_piop_eval.validate_instruction_input_stdout("", "optimized")
        )
        observed = metal_piop_eval.validate_instruction_input_stdout(record, "metal")
        assert observed is not None
        self.assertEqual(observed["storage_initialization"], "minimal")
        self.assertEqual(observed["native_primer"], "async")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            metal_piop_eval.validate_instruction_input_stdout(
                record + "\n" + record, "metal"
            )

    def test_requires_exact_target_bytecode_member_and_metal_spans(self) -> None:
        observed = metal_piop_eval.bytecode_member_breakdown(
            complete_bytecode_trace(26, "metal"), "metal", 26
        )
        self.assertEqual(observed["outer_counts"]["prove_round"], 26)
        self.assertEqual(observed["metal_counts"]["dense_round"], 9)
        self.assertEqual(observed["metal_counts"]["cpu_tail"], 16)
        scale_25 = metal_piop_eval.bytecode_member_breakdown(
            complete_bytecode_trace(25, "metal"), "metal", 25
        )
        self.assertEqual(scale_25["metal_counts"]["dense_round"], 8)

        with self.assertRaisesRegex(ValueError, "unexpectedly contains"):
            metal_piop_eval.bytecode_member_breakdown(
                complete_bytecode_trace(26, "metal"), "optimized", 26
            )

    def test_requires_exact_instruction_input_lifecycle_and_resources(self) -> None:
        optimized = metal_piop_eval.instruction_input_member_breakdown(
            complete_instruction_input_trace(26, "optimized"), "optimized", 26, 16
        )
        self.assertEqual(
            optimized["row_lifecycle"],
            {
                "kind": "optimized_cpu",
                "rows": 1 << 26,
                "row_bytes": 48,
                "prepare_storage_id": 101,
                "stage3_storage_id": 101,
            },
        )
        observed = metal_piop_eval.instruction_input_member_breakdown(
            complete_instruction_input_trace(26, "metal"), "metal", 26, 16
        )
        self.assertEqual(observed["outer_counts"]["prove_round"], 26)
        self.assertEqual(observed["metal_counts"]["dense_round"], 9)
        self.assertEqual(observed["metal_counts"]["cpu_tail"], 16)
        self.assertEqual(observed["components"]["prefetch_submit_us"], 10.0)
        self.assertEqual(
            observed["components"]["service_us"],
            observed["components"]["member_us"] + 10.0,
        )
        self.assertEqual(
            observed["resource_observation"]["native_primer"]["timings"][
                "submit_span_wall_ns"
            ],
            10_000,
        )
        self.assertEqual(
            observed["resource_observation"]["host_tail_bytes"],
            8 * (1 << 16) * 16,
        )
        self.assertTrue(observed["resource_observation"]["resident_rows_reused"])
        self.assertEqual(
            observed["resource_observation"]["round_device_buffer_allocations"], 0
        )
        self.assertEqual(
            observed["row_lifecycle"],
            {
                "kind": "metal_compact_resident",
                "rows": 1 << 26,
                "row_bytes": 48,
                "prepare_storage_id": 202,
                "stage1_storage_id": 202,
                "stage3_storage_id": 202,
                "residual_storage_id": 203,
                "row_production": {
                    "source_kind": "owned_random_access",
                    "witness_row_extractions": 1 << 26,
                    "residual_rows_written": 1 << 26,
                    "compact_rows_written": 1 << 26,
                    "compact_row_bytes": 48,
                    "residual_row_bytes": 112,
                    "compact_allocations": 1,
                    "residual_allocations": 1,
                    "full_row_allocations": 0,
                    "full_domain_copy_bytes": 0,
                    "full_domain_copy_dispatches": 0,
                    "host_repack_rows": 0,
                },
            },
        )

    def test_schema_seven_service_fields_are_instruction_input_only(self) -> None:
        bytecode = metal_piop_eval.bytecode_member_breakdown(
            complete_bytecode_trace(26, "metal"), "metal", 26
        )
        self.assertNotIn("service_ns", metal_piop_eval.member_record(bytecode))

        instruction_input = metal_piop_eval.instruction_input_member_breakdown(
            complete_instruction_input_trace(26, "metal"), "metal", 26, 16
        )
        record = metal_piop_eval.member_record(
            instruction_input, include_prefetch=True
        )
        self.assertEqual(record["prefetch_submit_ns"], 10_000)
        self.assertEqual(
            record["service_ns"], record["member_ns"] + record["prefetch_submit_ns"]
        )

        with self.assertRaisesRegex(ValueError, "unexpectedly contains"):
            metal_piop_eval.instruction_input_member_breakdown(
                complete_instruction_input_trace(26, "metal"), "optimized", 26, 16
            )

    def test_rejects_instruction_input_resource_or_containment_drift(self) -> None:
        wrong_buffers = complete_instruction_input_trace(26, "metal")
        allocation = next(
            event
            for event in wrong_buffers
            if event["name"] == "MetalInstructionInput::allocation_plan"
        )
        allocation["args"]["device_buffers"] = "5"
        with self.assertRaisesRegex(ValueError, "buffer accounting"):
            metal_piop_eval.instruction_input_member_breakdown(
                wrong_buffers, "metal", 26, 16
            )

        boolean_allocation = complete_instruction_input_trace(26, "metal")
        allocation = next(
            event
            for event in boolean_allocation
            if event["name"] == "MetalInstructionInput::allocation_plan"
        )
        allocation["args"]["current_device_bytes"] = True
        with self.assertRaisesRegex(ValueError, "invalid current_device_bytes"):
            metal_piop_eval.instruction_input_member_breakdown(
                boolean_allocation, "metal", 26, 16
            )

        wrong_plan = complete_instruction_input_trace(26, "metal")
        allocation = next(
            event
            for event in wrong_plan
            if event["name"] == "MetalInstructionInput::allocation_plan"
        )
        allocation["args"]["planned_device_bytes"] = str(
            metal_piop_eval.instruction_input_sequence_storage_bytes(26) - 16
        )
        with self.assertRaisesRegex(ValueError, "buffer accounting"):
            metal_piop_eval.instruction_input_member_breakdown(
                wrong_plan, "metal", 26, 16
            )

        missing_rows = complete_instruction_input_trace(26, "metal")
        allocation = next(
            event
            for event in missing_rows
            if event["name"] == "MetalInstructionInput::allocation_plan"
        )
        allocation["args"]["current_device_bytes"] = str(160 * (1 << 26) - 1)
        with self.assertRaisesRegex(ValueError, "buffer accounting"):
            metal_piop_eval.instruction_input_member_breakdown(
                missing_rows, "metal", 26, 16
            )

        inside_piop = complete_instruction_input_trace(26, "metal")
        storage = next(
            event
            for event in inside_piop
            if event["name"] == "MetalInstructionInput::storage_prepare"
        )
        storage["ts"] = 1_010.0
        with self.assertRaisesRegex(ValueError, "backend witness preparation"):
            metal_piop_eval.instruction_input_member_breakdown(
                inside_piop, "metal", 26, 16
            )

        for backend, span, field in (
            (
                "optimized",
                "OptimizedInstructionInput::rows_stage3_use",
                "cpu_rows_storage_id",
            ),
            (
                "metal",
                "MetalInstructionInput::compact_rows_stage1_handoff",
                "compact_rows_storage_id",
            ),
            (
                "metal",
                "MetalInstructionInput::prepare",
                "resident_rows_storage_id",
            ),
        ):
            with self.subTest(backend=backend, span=span):
                mismatched = complete_instruction_input_trace(26, backend)
                lifecycle = next(event for event in mismatched if event["name"] == span)
                lifecycle["args"][field] = "303"
                with self.assertRaisesRegex(ValueError, "row lifecycle"):
                    metal_piop_eval.instruction_input_member_breakdown(
                        mismatched, backend, 26, 16
                    )

        copied_rows = complete_instruction_input_trace(26, "metal")
        production = next(
            event
            for event in copied_rows
            if event["name"] == "MetalInstructionInput::compact_rows_prepare"
        )
        production["args"]["full_domain_copy_bytes"] = "48"
        with self.assertRaisesRegex(ValueError, "row lifecycle"):
            metal_piop_eval.instruction_input_member_breakdown(
                copied_rows, "metal", 26, 16
            )

        repacked_rows = complete_instruction_input_trace(26, "metal")
        production = next(
            event
            for event in repacked_rows
            if event["name"] == "MetalInstructionInput::compact_rows_prepare"
        )
        production["args"]["source_kind"] = "retained_host_repack"
        production["args"]["host_repack_rows"] = str(1 << 26)
        with self.assertRaisesRegex(ValueError, "row lifecycle"):
            metal_piop_eval.instruction_input_member_breakdown(
                repacked_rows, "metal", 26, 16
            )

        aliased_residual = complete_instruction_input_trace(26, "metal")
        production = next(
            event
            for event in aliased_residual
            if event["name"] == "MetalInstructionInput::compact_rows_prepare"
        )
        production["args"]["residual_rows_storage_id"] = "202"
        with self.assertRaisesRegex(ValueError, "row lifecycle"):
            metal_piop_eval.instruction_input_member_breakdown(
                aliased_residual, "metal", 26, 16
            )

        changed_residual = complete_instruction_input_trace(26, "metal")
        stage1 = next(
            event
            for event in changed_residual
            if event["name"]
            == "MetalInstructionInput::compact_rows_stage1_handoff"
        )
        stage1["args"]["residual_rows_storage_id"] = "204"
        with self.assertRaisesRegex(ValueError, "row lifecycle"):
            metal_piop_eval.instruction_input_member_breakdown(
                changed_residual, "metal", 26, 16
            )

    def test_rejects_instruction_input_primer_contract_drift(self) -> None:
        unknown = complete_instruction_input_trace(26, "metal")
        unknown.append(
            {
                "name": "MetalInstructionInput::unknown_startup_phase",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 1_075.0,
                "dur": 1.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown Metal phases"):
            metal_piop_eval.instruction_input_member_breakdown(
                unknown, "metal", 26, 16
            )

        missing = complete_instruction_input_trace(26, "metal")
        missing.remove(
            next(
                event
                for event in missing
                if event["name"] == "MetalInstructionInput::native_primer_join"
            )
        )
        with self.assertRaisesRegex(ValueError, "span counts"):
            metal_piop_eval.instruction_input_member_breakdown(
                missing, "metal", 26, 16
            )

        wrong_geometry = complete_instruction_input_trace(26, "metal")
        submit = next(
            event
            for event in wrong_geometry
            if event["name"] == "MetalInstructionInput::native_primer_submit"
        )
        submit["args"]["source_elements"] = "32"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "invalid geometry"):
            metal_piop_eval.instruction_input_member_breakdown(
                wrong_geometry, "metal", 26, 16
            )

        advanced = complete_instruction_input_trace(26, "metal")
        completion = next(
            event
            for event in advanced
            if event["name"] == "MetalInstructionInput::native_primer_complete"
        )
        completion["args"]["protocol_state_advanced"] = "true"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "completion is inconsistent"):
            metal_piop_eval.instruction_input_member_breakdown(
                advanced, "metal", 26, 16
            )

        mismatched_buffer = complete_instruction_input_trace(26, "metal")
        join = next(
            event
            for event in mismatched_buffer
            if event["name"] == "MetalInstructionInput::native_primer_join"
        )
        join["args"]["storage_buffer_0"] = "999"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "row lifecycle"):
            metal_piop_eval.instruction_input_member_breakdown(
                mismatched_buffer, "metal", 26, 16
            )

        invalid_timing = complete_instruction_input_trace(26, "metal")
        completion = next(
            event
            for event in invalid_timing
            if event["name"] == "MetalInstructionInput::native_primer_complete"
        )
        completion["args"]["lifecycle_wall_ns"] = "1000"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "completion is inconsistent"):
            metal_piop_eval.instruction_input_member_breakdown(
                invalid_timing, "metal", 26, 16
            )

        undercharged_submit = complete_instruction_input_trace(26, "metal")
        completion = next(
            event
            for event in undercharged_submit
            if event["name"] == "MetalInstructionInput::native_primer_complete"
        )
        completion["args"]["submit_wall_ns"] = "11000"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "completion is inconsistent"):
            metal_piop_eval.instruction_input_member_breakdown(
                undercharged_submit, "metal", 26, 16
            )

        late_submit = complete_instruction_input_trace(26, "metal")
        submit = next(
            event
            for event in late_submit
            if event["name"] == "MetalInstructionInput::native_primer_submit"
        )
        submit["ts"] = 1_085.0
        with self.assertRaisesRegex(ValueError, "before stage-3 Shift preparation"):
            metal_piop_eval.instruction_input_member_breakdown(
                late_submit, "metal", 26, 16
            )

    def test_rejects_incomplete_or_misattributed_instruction_input_cpu_tail(
        self,
    ) -> None:
        missing = complete_instruction_input_trace(26, "metal")
        tails = [
            event
            for event in missing
            if event["name"] == "MetalInstructionInput::cpu_tail"
        ]
        missing.remove(tails[2])
        with self.assertRaisesRegex(ValueError, "span counts"):
            metal_piop_eval.instruction_input_member_breakdown(
                missing, "metal", 26, 16
            )

        wrong_round = complete_instruction_input_trace(26, "metal")
        tails = [
            event
            for event in wrong_round
            if event["name"] == "MetalInstructionInput::cpu_tail"
        ]
        tails[2]["ts"] = float(tails[1]["ts"]) + 1.0
        with self.assertRaisesRegex(ValueError, "not contained"):
            metal_piop_eval.instruction_input_member_breakdown(
                wrong_round, "metal", 26, 16
            )

        wrong_finish = complete_instruction_input_trace(26, "metal")
        tails = [
            event
            for event in wrong_finish
            if event["name"] == "MetalInstructionInput::cpu_tail"
        ]
        tails[-1]["ts"] = 3_890.0
        with self.assertRaisesRegex(ValueError, "not contained"):
            metal_piop_eval.instruction_input_member_breakdown(
                wrong_finish, "metal", 26, 16
            )

        overlap = complete_instruction_input_trace(26, "metal")
        readback = next(
            event
            for event in overlap
            if event["name"] == "MetalInstructionInput::readback"
        )
        readback["dur"] = 30.0
        with self.assertRaisesRegex(ValueError, "overlaps CPU-tail"):
            metal_piop_eval.instruction_input_member_breakdown(
                overlap, "metal", 26, 16
            )

    def test_rejects_incomplete_or_misattributed_bytecode_member(self) -> None:
        missing_round = complete_bytecode_trace(26, "metal")
        missing_round.remove(
            next(
                event
                for event in missing_round
                if event["name"] == "BytecodeReadRafCycle::prove_round"
            )
        )
        with self.assertRaisesRegex(ValueError, "member span counts"):
            metal_piop_eval.bytecode_member_breakdown(missing_round, "metal", 26)

        outside = complete_bytecode_trace(26, "metal")
        first_message = next(
            event
            for event in outside
            if event["name"] == "MetalBytecodeReadRafCycle::first_message"
        )
        first_message["ts"] = 9_900.0
        with self.assertRaisesRegex(ValueError, "not contained"):
            metal_piop_eval.bytecode_member_breakdown(outside, "metal", 26)

    def test_five_stable_pairs_clear_the_fixed_bytecode_gate(self) -> None:
        pair = {
            "cpu_us": 1_000.0,
            "metal_us": 500.0,
            "cpu_prepare_us": 1.0,
            "metal_prepare_us": 1.0,
            "cpu_instruction_ra_us": 700.0,
            "metal_instruction_ra_us": 100.0,
            "cpu_bytecode_us": 1_000.0,
            "metal_bytecode_us": 200.0,
            "cpu_instruction_input_us": 1_000.0,
            "metal_instruction_input_us": 200.0,
        }
        pairs = [
            {
                **pair,
                "order": ["optimized", "metal"]
                if index % 2 == 0
                else ["metal", "optimized"],
            }
            for index in range(5)
        ]
        decision = metal_piop_eval.summarize_pairs(pairs)[
            "bytecode_read_raf_cycle_decision"
        ]
        self.assertTrue(decision["clears"])
        self.assertEqual(decision["median_speedup"], 5.0)
        self.assertEqual(decision["optimized_first_median_speedup"], 5.0)
        self.assertEqual(decision["metal_first_median_speedup"], 5.0)
        self.assertTrue(decision["clears_order_strata"])

    def test_bytecode_gate_rejects_a_slow_order_stratum(self) -> None:
        pairs = []
        for index in range(5):
            metal_first = index % 2 == 1
            pairs.append(
                {
                    "order": ["metal", "optimized"]
                    if metal_first
                    else ["optimized", "metal"],
                    "cpu_us": 1_000.0,
                    "metal_us": 500.0,
                    "cpu_prepare_us": 1.0,
                    "metal_prepare_us": 1.0,
                    "cpu_instruction_ra_us": 700.0,
                    "metal_instruction_ra_us": 100.0,
                    "cpu_bytecode_us": 1_000.0,
                    "metal_bytecode_us": 10_000.0 if metal_first else 200.0,
                    "cpu_instruction_input_us": 1_000.0,
                    "metal_instruction_input_us": 200.0,
                }
            )
        decision = metal_piop_eval.summarize_pairs(pairs)[
            "bytecode_read_raf_cycle_decision"
        ]
        self.assertEqual(decision["median_speedup"], 5.0)
        self.assertEqual(decision["optimized_first_median_speedup"], 5.0)
        self.assertEqual(decision["metal_first_median_speedup"], 0.1)
        self.assertFalse(decision["clears_order_strata"])
        self.assertFalse(decision["clears"])

    def test_five_stable_pairs_clear_instruction_input_gate(self) -> None:
        pairs = [
            {
                "order": ["optimized", "metal"]
                if index % 2 == 0
                else ["metal", "optimized"],
                "cpu_us": 1_000.0,
                "metal_us": 500.0,
                "cpu_prepare_us": 1.0,
                "metal_prepare_us": 1.0,
                "cpu_instruction_ra_us": 700.0,
                "metal_instruction_ra_us": 100.0,
                "cpu_bytecode_us": 1_000.0,
                "metal_bytecode_us": 200.0,
                "cpu_instruction_input_us": 800.0,
                "metal_instruction_input_us": 160.0,
            }
            for index in range(5)
        ]
        decision = metal_piop_eval.summarize_pairs(pairs)[
            "instruction_input_kernel_service_decision"
        ]
        self.assertTrue(decision["clears"])
        self.assertEqual(decision["median_speedup"], 5.0)

    def test_production_run_class_is_exact(self) -> None:
        metal_piop_eval.validate_run_class("diagnostic", "btreemap", 25, 1)
        metal_piop_eval.validate_run_class("production", "fibonacci", 26, 5)
        for workload, log_n, repeats in [
            ("btreemap", 26, 5),
            ("fibonacci", 25, 5),
            ("fibonacci", 26, 1),
        ]:
            with self.assertRaisesRegex(ValueError, "requires Fibonacci"):
                metal_piop_eval.validate_run_class(
                    "production", workload, log_n, repeats
                )

    def test_parses_one_positive_maximum_rss_record(self) -> None:
        stderr = "       123456789  maximum resident set size\n"
        self.assertEqual(metal_piop_eval.parse_max_rss(stderr), 123_456_789)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            metal_piop_eval.parse_max_rss("")

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
