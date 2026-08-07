from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "metal_outer_remainder_eval.py"
sys.path.insert(0, str(Path.cwd()))
SPEC = importlib.util.spec_from_file_location("metal_outer_remainder_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EVAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVAL
SPEC.loader.exec_module(EVAL)


def complete(
    name: str, start: float, duration: float, args: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "ph": "X",
        "ts": start,
        "dur": duration,
        "pid": 1,
        "tid": 1,
        "args": args or {},
    }


def gpu_args(wall: int = 1_000, active: int = 800) -> dict[str, str]:
    return {"dispatch_wall_ns": str(wall), "gpu_active_ns": str(active)}


def arm_events(
    pair: int,
    backend: str,
    base: float,
    member_duration: float,
    order_position: int,
    *,
    wrong_compact_release: bool = False,
    wrong_storage_id: bool = False,
    omit_storage_initialization: bool = False,
    storage_inside_member: bool = False,
    nonzero_member_allocation: bool = False,
    omit_last_fs: bool = False,
) -> list[dict[str, Any]]:
    rows = 1 << EVAL.LOG_N
    compact_id = 10_000 + 10 * (pair + 1)
    residual_id = compact_id + 1
    device_id = compact_id + 2
    member_start = base + 100.0
    member_end = member_start + member_duration
    arm_end = member_end + 200.0
    events = [
        complete(
            EVAL.ARM,
            base,
            arm_end - base,
            {
                "backend": backend,
                "sample_index": str(pair),
                "pair": str(pair),
                "order_position": str(order_position),
                "excluded_warmup": "true" if pair == -1 else "false",
                "trace_rows": str(rows - 100),
                "padded_trace_rows": str(rows),
            },
        )
    ]
    if backend == "metal":
        storage_ids = [20_000 + 100 * (pair + 1) + index for index in range(9)]
        storage_start = member_start + 2 if storage_inside_member else base + 25
        events.extend(
            [
                complete(
                    EVAL.ROW_PREPARE,
                    base + 10,
                    10,
                    {
                        "compact_rows_storage_id": str(compact_id),
                        "residual_rows_storage_id": str(residual_id),
                        "resident_rows": str(rows),
                    },
                ),
                complete(
                    EVAL.METAL_STORAGE_PREPARE,
                    storage_start,
                    10,
                    {
                        "cycles": str(rows),
                        "planned_device_bytes": str(EVAL.STORAGE_BYTES),
                        "maximum_buffer_bytes": str(
                            EVAL.MAXIMUM_STORAGE_BUFFER_BYTES
                        ),
                        "current_device_bytes": str(rows * 160),
                        "recommended_max_working_set_bytes": str(32 * (1 << 30)),
                        "initialization_mode": "full",
                        "admitted": "true",
                        "initialized": "true",
                        "fallback_reason": "none",
                        "device_buffers": str(EVAL.STORAGE_BUFFERS),
                        "initialization_bytes": str(EVAL.STORAGE_BYTES),
                        "initialization_wall_ns": "8000",
                        "initialization_gpu_active_ns": "6000",
                        **{
                            f"buffer_{index}": str(identity)
                            for index, identity in enumerate(storage_ids)
                        },
                    },
                ),
                complete(
                    EVAL.ROW_STAGE1_HANDOFF,
                    base + 40,
                    20,
                    {
                        "compact_rows_storage_id": str(compact_id),
                        "residual_rows_storage_id": str(residual_id),
                        "resident_rows": str(rows),
                    },
                ),
                complete(
                    EVAL.INSTRUCTION_INPUT_PREPARE,
                    member_end + 40,
                    20,
                    {
                        "resident_rows_storage_id": str(compact_id),
                        "resident_rows": str(rows),
                    },
                ),
            ]
        )
        if not omit_storage_initialization:
            events.append(
                complete(
                    EVAL.METAL_STORAGE_INITIALIZE,
                    storage_start + 1,
                    8,
                    {
                        "mode": "full",
                        "device_buffers": str(EVAL.STORAGE_BUFFERS),
                        "bytes": str(EVAL.STORAGE_BYTES),
                        "protocol_dispatches": "0",
                        **{
                            f"buffer_{index}": str(identity)
                            for index, identity in enumerate(storage_ids)
                        },
                    },
                )
            )

    events.append(complete(EVAL.MEMBER, member_start, member_duration))
    prepare_duration = member_duration * (0.50 if backend == "optimized" else 0.30)
    rounds_duration = member_duration * (0.28 if backend == "optimized" else 0.50)
    finish_duration = member_duration * 0.01
    output_duration = member_duration * (0.20 if backend == "optimized" else 0.18)
    prepare_start = member_start + member_duration * 0.005
    rounds_start = prepare_start + prepare_duration
    round_duration = rounds_duration / EVAL.ROUNDS
    finish_start = rounds_start + rounds_duration
    output_start = finish_start + finish_duration
    events.extend(
        [
            complete(EVAL.PREPARE, prepare_start, prepare_duration),
            complete(EVAL.FINISH, finish_start, finish_duration),
            complete(EVAL.OUTPUT, output_start, output_duration),
        ]
    )

    round_spans: list[tuple[float, float]] = []
    for round_index in range(EVAL.ROUNDS):
        start = rounds_start + round_index * round_duration
        round_spans.append((start, round_duration))
        events.append(
            complete(EVAL.SUMCHECK_ROUND, start, round_duration, {"round": str(round_index)})
        )
        events.append(complete(EVAL.PROVE_ROUND, start + 1, round_duration * 0.78))
        if not (omit_last_fs and round_index == EVAL.ROUNDS - 1):
            events.append(
                complete(EVAL.HOST_FS, start + round_duration * 0.82, round_duration * 0.10)
            )

    if backend == "optimized":
        events.append(
            complete(
                EVAL.CPU_OUTPUT_WALK,
                output_start + 1,
                output_duration - 2,
            )
        )
        return events

    metal_prepare_start = prepare_start + 1
    metal_prepare_duration = prepare_duration - 2
    events.extend(
        [
            complete(EVAL.METAL_PREPARE, metal_prepare_start, metal_prepare_duration),
            complete(
                EVAL.METAL_ALLOCATION_PLAN,
                metal_prepare_start + 1,
                5,
                {
                    "admitted": "true",
                    "storage_reused": "true",
                    "existing_resident_bytes": str(rows * 160),
                    "preallocated_device_bytes": str(EVAL.STORAGE_BYTES),
                    "additional_working_set_bytes": str(
                        1 if nonzero_member_allocation else 0
                    ),
                    "current_device_bytes": str(rows * 160 + EVAL.STORAGE_BYTES),
                    "recommended_max_working_set_bytes": str(32 * (1 << 30)),
                },
            ),
            complete(
                EVAL.METAL_ROW_HANDOFF,
                metal_prepare_start + 10,
                5,
                {
                    "compact_rows_storage_id": str(compact_id),
                    "residual_rows_storage_id": str(residual_id),
                    "device_registry_id": str(device_id),
                    "resident_rows": str(rows),
                    "row_upload_bytes": "0",
                    "device_allocations": "0",
                },
            ),
            complete(
                EVAL.METAL_SEQUENCE_PREPARE,
                metal_prepare_start + 20,
                100,
                {
                    "resident_rows": str(rows),
                    "rounds": str(EVAL.ROUNDS),
                    "cutoff_elements": str(1 << 16),
                    "trace_cutoff_elements": str(1 << 18),
                    "planned_device_bytes": str(EVAL.STORAGE_BYTES),
                    "compact_rows_storage_id": str(compact_id),
                    "residual_rows_storage_id": str(residual_id),
                    "device_registry_id": str(device_id),
                    "storage_reused": "true",
                    "storage_initialization_mode": "full",
                    "preinitialized_device_bytes": str(EVAL.STORAGE_BYTES),
                    "initialization_bytes": str(EVAL.STORAGE_BYTES),
                    "attached_owned_bytes": str(EVAL.STORAGE_BYTES),
                    **{
                        f"storage_buffer_{index}": str(
                            identity + (1 if wrong_storage_id and index == 0 else 0)
                        )
                        for index, identity in enumerate(storage_ids)
                    },
                    "row_upload_bytes": "0",
                    "full_domain_copy_dispatches": "0",
                    "sequence_device_buffer_allocations": "0",
                    "round_device_buffer_allocations": "0",
                },
            ),
            complete(
                EVAL.METAL_FIRST_MESSAGE,
                metal_prepare_start + metal_prepare_duration * 0.10,
                metal_prepare_duration * 0.80,
                gpu_args(),
            ),
        ]
    )

    first_start, first_duration = round_spans[1]
    events.append(
        complete(EVAL.METAL_FIRST_BIND, first_start + 2, first_duration * 0.70, gpu_args())
    )
    gpu_last_round = EVAL.ROUNDS - 16
    for round_index in range(2, gpu_last_round + 1):
        start, duration = round_spans[round_index]
        events.append(
            complete(EVAL.METAL_DENSE_ROUND, start + 2, duration * 0.70, gpu_args())
        )
    readback_round = gpu_last_round + 1
    readback_start, readback_round_duration = round_spans[readback_round]
    events.append(
        complete(
            EVAL.METAL_READBACK,
            readback_start + readback_round_duration * 0.72,
            readback_round_duration * 0.05,
            {
                "readbacks": "1",
                "elements": str(2 * (1 << 16)),
                "bytes": str(EVAL.FIELD_BYTES * 2 * (1 << 16)),
            },
        )
    )
    for round_index in range(readback_round, EVAL.ROUNDS):
        start, duration = round_spans[round_index]
        events.append(
            complete(
                EVAL.METAL_CPU_TAIL,
                start + (duration * 0.10 if round_index == readback_round else 2),
                duration * 0.70,
            )
        )
    events.append(complete(EVAL.METAL_CPU_TAIL, finish_start + 1, finish_duration - 2))

    events.extend(
        [
            complete(
                EVAL.METAL_OUTPUT,
                output_start + 1,
                output_duration * 0.80,
                {
                    **gpu_args(),
                    "readbacks": "1",
                    "output_elements": str(EVAL.OUTPUT_CLAIMS),
                    "readback_bytes": str(EVAL.FIELD_BYTES * EVAL.OUTPUT_CLAIMS),
                    "row_upload_bytes": "0",
                },
            ),
            complete(
                EVAL.METAL_ROW_RELEASE,
                output_start + output_duration * 0.85,
                output_duration * 0.10,
                {
                    "compact_rows_storage_id": str(
                        compact_id + 99 if wrong_compact_release else compact_id
                    ),
                    "residual_rows_storage_id": str(residual_id),
                    "device_registry_id": str(device_id),
                    "resident_rows": str(rows),
                    "row_upload_bytes": "0",
                    "device_allocations": "0",
                    "residual_row_bytes": str(rows * EVAL.RESIDUAL_ROW_BYTES),
                    "remaining_sequence_storage_bytes": str(
                        EVAL.REMAINING_SEQUENCE_STORAGE_BYTES
                    ),
                    "compact_release_bytes": "0",
                    "released_owned_bytes": str(
                        rows * EVAL.RESIDUAL_ROW_BYTES
                        + EVAL.REMAINING_SEQUENCE_STORAGE_BYTES
                    ),
                    "release_completed": "true",
                    "residual_released": "true",
                    "compact_retained": "true",
                },
            ),
        ]
    )
    return events


def fixture(
    *,
    wrong_compact_release: bool = False,
    wrong_storage_id: bool = False,
    omit_storage_initialization: bool = False,
    storage_inside_member: bool = False,
    nonzero_member_allocation: bool = False,
    omit_last_fs: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    orders = [
        ["optimized", "metal"] if pair % 2 == 0 else ["metal", "optimized"]
        for pair in range(EVAL.PAIRS)
    ]
    events: list[dict[str, Any]] = []
    runner_pairs = []
    all_pairs = [-1, *range(EVAL.PAIRS)]
    for sequence, pair in enumerate(all_pairs):
        order = ["optimized", "metal"] if pair == -1 else orders[pair]
        base = float(sequence * 3_000_000)
        offsets = {order[0]: 0.0, order[1]: 1_200_000.0}
        events.extend(
            arm_events(
                pair,
                "optimized",
                base + offsets["optimized"],
                900_000.0,
                order.index("optimized"),
            )
        )
        events.extend(
            arm_events(
                pair,
                "metal",
                base + offsets["metal"],
                200_000.0,
                order.index("metal"),
                wrong_compact_release=wrong_compact_release,
                wrong_storage_id=wrong_storage_id,
                omit_storage_initialization=omit_storage_initialization,
                storage_inside_member=storage_inside_member,
                nonzero_member_allocation=nonzero_member_allocation,
                omit_last_fs=omit_last_fs,
            )
        )
        runner_pairs.append(
            {
                "pair": pair,
                "excluded_warmup": pair == -1,
                "order": order,
                "optimized": {"full_prove_ns": 1, "proof_verified": True},
                "metal": {"full_prove_ns": 1, "proof_verified": True},
                "proofs_exact": True,
            }
        )
    runner = {
        "schema": EVAL.RUNNER_SCHEMA,
        "schema_version": 2,
        "fixture": "real-fibonacci-akita-proof",
        "log_n": EVAL.LOG_N,
        "trace_rows": (1 << EVAL.LOG_N) - 100,
        "padded_trace_rows": 1 << EVAL.LOG_N,
        "pairs": EVAL.PAIRS,
        "excluded_warmup_pairs": 1,
        "rayon_threads": EVAL.RAYON_THREADS,
        "orders": orders,
        "parameters": {
            "materialize_threads": 256,
            "transition_threads": 128,
            "output_threads": 256,
            "cutoff_log2": 16,
            "trace_cutoff_log2": 18,
            "binding_plan": "b_only_v1",
            "storage_initialization": "full",
        },
        "warmup": runner_pairs[0],
        "samples": runner_pairs[1:],
    }
    return events, runner


class OuterRemainderEvaluatorTests(unittest.TestCase):
    def parse(self, events: list[dict[str, Any]], runner: dict[str, Any]) -> dict[str, Any]:
        return EVAL.parse_outer_remainder_result(
            events,
            runner,
            source_sha256="a" * 64,
            binary_sha256="b" * 64,
            artifact_dir="test",
        )

    def test_happy_path_is_exact_and_clears_local_gate(self) -> None:
        result = self.parse(*fixture())
        self.assertEqual(result["schema"], EVAL.SCHEMA)
        self.assertTrue(result["all_exact"])
        self.assertTrue(result["promotion"]["eligible"])
        self.assertAlmostEqual(result["metrics"]["median_paired_speedup"], 4.5)
        self.assertLess(
            result["metrics"]["median_cold_inclusive_speedup"],
            result["metrics"]["median_paired_speedup"],
        )
        self.assertEqual(result["resources"]["output_readback_bytes"], 560)
        self.assertEqual(
            result["resources"]["metal_full_prove_ns_samples"], [1] * 6
        )
        self.assertEqual(result["resources"]["gpu_seconds"], 6e-9)

    def test_log_25_three_pair_screen_uses_the_same_exact_parser(self) -> None:
        with mock.patch.multiple(
            EVAL,
            SCHEMA="outer_remainder_screen_v1",
            SCHEMA_VERSION=1,
            LOG_N=25,
            PAIRS=3,
            ROUNDS=26,
            STORAGE_BYTES=2_152_599_280,
            DENSE_STORAGE_BYTES=2 * (1 << 30),
            REMAINING_SEQUENCE_STORAGE_BYTES=5_115_632,
            MAXIMUM_STORAGE_BUFFER_BYTES=1 << 30,
        ):
            result = self.parse(*fixture())

        self.assertEqual(result["schema"], "outer_remainder_screen_v1")
        self.assertEqual(result["fingerprint"]["log_n"], 25)
        self.assertEqual(result["fingerprint"]["pairs"], 3)
        self.assertEqual(len(result["samples"]), 3)
        self.assertTrue(result["all_exact"])
        self.assertEqual(
            result["resources"]["outer_remainder_storage_bytes"],
            2_152_599_280,
        )

    def test_gpu_accounting_includes_warmup_and_all_timed_metal_arms(self) -> None:
        events, runner = fixture()
        expected = [11, 22, 33, 44, 55, 66]
        for pair, duration in zip(
            [runner["warmup"], *runner["samples"]], expected
        ):
            pair["metal"]["full_prove_ns"] = duration

        result = self.parse(events, runner)

        self.assertEqual(
            result["resources"]["metal_full_prove_ns_samples"], expected
        )
        self.assertEqual(
            result["resources"]["gpu_seconds"], sum(expected) / 1e9
        )

    def test_every_metal_full_prove_duration_must_be_positive(self) -> None:
        for index in range(EVAL.PAIRS + 1):
            with self.subTest(index=index):
                events, runner = fixture()
                [runner["warmup"], *runner["samples"]][index]["metal"][
                    "full_prove_ns"
                ] = 0
                with self.assertRaisesRegex(ValueError, "full-prove durations"):
                    self.parse(events, runner)

    def test_artifact_directory_prefers_explicit_then_controller_environment(
        self,
    ) -> None:
        root = Path("root")
        explicit = Path("explicit")
        configured = Path("configured")
        with mock.patch.dict(
            os.environ,
            {"JOLT_AUTORESEARCH_EVAL_DIR": str(configured)},
            clear=False,
        ):
            self.assertEqual(EVAL.resolve_artifact_dir(root, explicit), explicit)
            self.assertEqual(
                EVAL.resolve_artifact_dir(root, None), configured.resolve()
            )

    def test_proof_mismatch_fails_correctness_and_promotion(self) -> None:
        events, runner = fixture()
        runner["samples"][2]["proofs_exact"] = False
        result = self.parse(events, runner)
        self.assertFalse(result["guards"]["correctness_exact"])
        self.assertFalse(result["all_exact"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_missing_fiat_shamir_span_fails_topology(self) -> None:
        result = self.parse(*fixture(omit_last_fs=True))
        self.assertFalse(result["guards"]["round_topology_exact"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_changed_compact_identity_fails_lifecycle(self) -> None:
        result = self.parse(*fixture(wrong_compact_release=True))
        self.assertFalse(result["guards"]["resident_row_lifecycle_exact"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_trace_arm_order_must_match_the_reported_stratum(self) -> None:
        events, runner = fixture()
        arm = next(
            event
            for event in events
            if event["name"] == EVAL.ARM
            and event["args"]["sample_index"] == "0"
            and event["args"]["backend"] == "metal"
        )
        arm["args"]["order_position"] = "0"
        result = self.parse(events, runner)
        self.assertFalse(result["guards"]["actual_arm_order_exact"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_sequence_trace_cutoff_must_match_the_runner(self) -> None:
        events, runner = fixture()
        runner["parameters"]["trace_cutoff_log2"] = 19
        result = self.parse(events, runner)
        self.assertFalse(result["guards"]["metal_sequence_geometry_exact"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_sequence_identity_is_part_of_the_lifecycle_chain(self) -> None:
        events, runner = fixture()
        sequence = next(
            event
            for event in events
            if event["name"] == EVAL.METAL_SEQUENCE_PREPARE and event["ph"] == "X"
        )
        sequence["args"]["compact_rows_storage_id"] = "999"
        result = self.parse(events, runner)
        self.assertFalse(result["guards"]["metal_row_identity_exact"])
        self.assertFalse(result["guards"]["resident_row_lifecycle_exact"])

    def test_metal_prepare_subphases_must_be_sequential(self) -> None:
        events, runner = fixture()
        sequence = next(
            event
            for event in events
            if event["name"] == EVAL.METAL_SEQUENCE_PREPARE and event["ph"] == "X"
        )
        sequence["ts"] -= 20
        result = self.parse(events, runner)
        self.assertFalse(result["guards"]["metal_phase_chronology_exact"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_storage_preparation_must_be_outside_the_member(self) -> None:
        result = self.parse(*fixture(storage_inside_member=True))
        self.assertFalse(
            result["guards"]["metal_storage_preparation_outside_member"]
        )
        self.assertFalse(result["promotion"]["eligible"])

    def test_storage_identity_must_survive_member_handoff(self) -> None:
        result = self.parse(*fixture(wrong_storage_id=True))
        self.assertFalse(result["guards"]["metal_sequence_geometry_exact"])
        self.assertFalse(result["guards"]["resident_row_lifecycle_exact"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_attached_storage_byte_count_is_checked(self) -> None:
        events, runner = fixture()
        sequence = next(
            event
            for event in events
            if event["name"] == EVAL.METAL_SEQUENCE_PREPARE and event["ph"] == "X"
        )
        sequence["args"]["attached_owned_bytes"] = "1"
        result = self.parse(events, runner)
        self.assertFalse(result["guards"]["metal_sequence_geometry_exact"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_release_owned_byte_count_is_checked(self) -> None:
        events, runner = fixture()
        release = next(
            event
            for event in events
            if event["name"] == EVAL.METAL_ROW_RELEASE and event["ph"] == "X"
        )
        release["args"]["released_owned_bytes"] = "1"
        result = self.parse(events, runner)
        self.assertFalse(result["guards"]["resident_row_lifecycle_exact"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_incomplete_release_is_rejected(self) -> None:
        events, runner = fixture()
        release = next(
            event
            for event in events
            if event["name"] == EVAL.METAL_ROW_RELEASE and event["ph"] == "X"
        )
        release["args"]["release_completed"] = "false"
        result = self.parse(events, runner)
        self.assertFalse(result["guards"]["resident_row_lifecycle_exact"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_storage_initialization_span_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "storage initialization"):
            self.parse(*fixture(omit_storage_initialization=True))

    def test_member_must_allocate_no_device_storage(self) -> None:
        result = self.parse(*fixture(nonzero_member_allocation=True))
        self.assertFalse(result["guards"]["metal_working_set_admitted"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_stale_runner_schema_is_rejected(self) -> None:
        events, runner = fixture()
        runner["schema"] = "outer_remainder_runner_v1"
        runner["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "wrong schema"):
            self.parse(events, runner)

    def test_cutoff_one_is_rejected_before_round_indexing(self) -> None:
        events, runner = fixture()
        runner["parameters"]["cutoff_log2"] = 1
        with self.assertRaisesRegex(ValueError, "cutoffs"):
            self.parse(events, runner)

    def test_unmatched_trace_event_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unmatched begin"):
            EVAL.parse_spans(
                [{"name": "bad", "ph": "B", "ts": 1, "pid": 1, "tid": 1}]
            )


if __name__ == "__main__":
    unittest.main()
