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


def complete_booleanity_address_trace(
    log_n: int,
    backend: str,
    inner_log2: int = 15,
    selectors_per_tile: int = 6,
    tile_threads: int = 512,
    finalize_threads: int = 1024,
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
        event("jolt_prover::piop", 0.0, 10_000.0),
        event("InstructionReadRaf::prepare", 90.0, 60.0),
        event("BooleanityAddressPhase::prepare", 1_000.0, 600.0),
    ]
    round_starts = [1_700.0 + 100.0 * index for index in range(8)]
    for timestamp in round_starts:
        events.extend(
            [
                event("sumcheck_round", timestamp - 5.0, 95.0),
                event("BooleanityAddressPhase::prove_round", timestamp, 80.0),
                event("sumcheck_host_fiat_shamir", timestamp + 85.0, 5.0),
            ]
        )
    events.extend(
        [
            event("BooleanityAddressPhase::finish_rounds", 2_500.0, 80.0),
            event("BooleanityAddressPhase::output_claims", 2_600.0, 40.0),
            event("Booleanity::prepare", 2_700.0, 100.0),
        ]
    )
    if backend == "optimized":
        events.append(
            event("OptimizedBooleanityAddress::row_source", 1_050.0, 300.0)
        )
    if backend == "metal":
        rows = 1 << log_n
        row_bytes = 40
        polys = 29
        k = 256
        e_in = 1 << inner_log2
        e_out = rows // e_in
        selector_tiles = (polys + selectors_per_tile - 1) // selectors_per_tile
        planned_bytes = metal_piop_eval.booleanity_address_sequence_storage_bytes(
            log_n, inner_log2, selectors_per_tile
        )
        events.extend(
            [
                event(
                    "MetalBooleanityRows::stage5_prepare",
                    100.0,
                    20.0,
                    {
                        "resident_rows_storage_id": "401",
                        "resident_rows": str(rows),
                        "resident_row_bytes": str(row_bytes),
                        "device_registry_id": "17",
                        "row_allocations": "1",
                        "row_upload_bytes": str(rows * row_bytes),
                    },
                ),
                event(
                    "MetalBooleanityRows::stage6a_address_use",
                    1_002.0,
                    2.0,
                    {
                        "resident_rows_storage_id": "401",
                        "resident_rows": str(rows),
                        "resident_row_bytes": str(row_bytes),
                        "device_registry_id": "17",
                        "row_allocations": "0",
                        "row_upload_bytes": "0",
                    },
                ),
                event("MetalBooleanityAddressPhase::prepare", 1_005.0, 580.0),
                event(
                    "MetalBooleanityAddressPhase::sequence_prepare",
                    1_010.0,
                    190.0,
                    {
                        "resident_rows_storage_id": "401",
                        "resident_rows": str(rows),
                        "resident_row_bytes": str(row_bytes),
                        "row_upload_bytes": "0",
                        "polys": str(polys),
                        "k": str(k),
                        "e_in_elements": str(e_in),
                        "e_out_elements": str(e_out),
                        "requested_inner_log2": str(inner_log2),
                        "effective_inner_log2": str(inner_log2),
                        "requested_selectors_per_tile": str(selectors_per_tile),
                        "effective_selectors_per_tile": str(selectors_per_tile),
                        "requested_tile_threads": str(tile_threads),
                        "effective_tile_threads": str(tile_threads),
                        "requested_finalize_threads": str(finalize_threads),
                        "effective_finalize_threads": str(finalize_threads),
                        "selector_tiles": str(selector_tiles),
                        "production_specialized": str(
                            selectors_per_tile in {3, 6}
                        ).lower(),
                    },
                ),
                event(
                    "MetalBooleanityAddressPhase::allocation_plan",
                    1_020.0,
                    80.0,
                    {
                        "device_buffers": "5",
                        "planned_device_bytes": str(planned_bytes),
                        "current_device_bytes": str(rows * row_bytes),
                        "recommended_device_bytes": str(rows * row_bytes + planned_bytes),
                    },
                ),
                event(
                    "MetalBooleanityAddressPhase::dispatch",
                    1_210.0,
                    190.0,
                    {
                        "command_buffers": "1",
                        "tile_dispatches": str(selector_tiles),
                        "finalize_dispatches": str(selector_tiles),
                        "command_completed": "true",
                        "gpu_active_ns": "150000",
                        "resident_rows_storage_id": "401",
                    },
                ),
                event(
                    "MetalBooleanityAddressPhase::readback",
                    1_410.0,
                    90.0,
                    {
                        "elements": str(polys * k),
                        "bytes": str(polys * k * 16),
                        "readbacks": "1",
                    },
                ),
                event(
                    "MetalBooleanityRows::stage6b_cycle_use",
                    2_710.0,
                    20.0,
                    {
                        "resident_rows_storage_id": "401",
                        "resident_rows": str(rows),
                        "resident_row_bytes": str(row_bytes),
                        "device_registry_id": "17",
                        "row_allocations": "0",
                        "row_upload_bytes": "0",
                    },
                ),
            ]
        )
    return events


def complete_hamming_weight_trace(
    log_n: int,
    backend: str,
    inner_log2: int = 15,
    selectors_per_tile: int = 6,
    tile_threads: int = 512,
    finalize_threads: int = 1024,
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
        event("jolt_prover::piop", 0.0, 10_000.0),
        event("InstructionReadRaf::prepare", 90.0, 60.0),
        event("BooleanityAddressPhase::prepare", 1_000.0, 600.0),
        event("Booleanity::prepare", 1_700.0, 200.0),
        event("HammingWeightClaimReduction::prepare", 2_000.0, 600.0),
    ]
    round_starts = [2_700.0 + 100.0 * index for index in range(8)]
    for timestamp in round_starts:
        events.extend(
            [
                event("sumcheck_round", timestamp - 5.0, 95.0),
                event("HammingWeightClaimReduction::prove_round", timestamp, 80.0),
                event("sumcheck_host_fiat_shamir", timestamp + 85.0, 5.0),
            ]
        )
    events.extend(
        [
            event("HammingWeightClaimReduction::finish_rounds", 3_500.0, 80.0),
            event("HammingWeightClaimReduction::output_claims", 3_600.0, 40.0),
        ]
    )
    if backend == "optimized":
        events.append(
            event("OptimizedHammingWeightClaimReduction::row_source", 2_050.0, 300.0)
        )
        return events

    rows = 1 << log_n
    row_bytes = 40
    polys = 29
    k = 256
    e_in = 1 << inner_log2
    e_out = rows // e_in
    selector_tiles = (polys + selectors_per_tile - 1) // selectors_per_tile
    planned_bytes = metal_piop_eval.booleanity_address_sequence_storage_bytes(
        log_n, inner_log2, selectors_per_tile
    )
    lifecycle_args = {
        "resident_rows_storage_id": "401",
        "resident_rows": str(rows),
        "resident_row_bytes": str(row_bytes),
        "device_registry_id": "17",
        "row_allocations": "0",
        "row_upload_bytes": "0",
    }
    events.extend(
        [
            event(
                "MetalBooleanityRows::stage5_prepare",
                100.0,
                20.0,
                {
                    **lifecycle_args,
                    "row_allocations": "1",
                    "row_upload_bytes": str(rows * row_bytes),
                },
            ),
            event(
                "MetalBooleanityRows::stage6a_address_use",
                1_002.0,
                2.0,
                {**lifecycle_args},
            ),
            event(
                "MetalBooleanityRows::stage6b_cycle_use",
                1_710.0,
                20.0,
                {**lifecycle_args},
            ),
            event(
                "MetalBooleanityRows::stage6b_retain_for_stage7",
                1_740.0,
                20.0,
                {**lifecycle_args},
            ),
            event(
                "MetalBooleanityRows::stage7_hamming_use",
                2_202.0,
                300.0,
                {
                    **lifecycle_args,
                    "terminal_consumer": "true",
                    "terminal_carry_removed": "true",
                },
            ),
            event("MetalHammingWeightClaimReduction::prepare", 2_005.0, 580.0),
            event(
                "MetalHammingWeightClaimReduction::sequence_prepare",
                2_010.0,
                190.0,
                {
                    "resident_rows_storage_id": "401",
                    "resident_rows": str(rows),
                    "resident_row_bytes": str(row_bytes),
                    "row_upload_bytes": "0",
                    "polys": str(polys),
                    "k": str(k),
                    "e_in_elements": str(e_in),
                    "e_out_elements": str(e_out),
                    "requested_inner_log2": str(inner_log2),
                    "effective_inner_log2": str(inner_log2),
                    "requested_selectors_per_tile": str(selectors_per_tile),
                    "effective_selectors_per_tile": str(selectors_per_tile),
                    "requested_tile_threads": str(tile_threads),
                    "effective_tile_threads": str(tile_threads),
                    "requested_finalize_threads": str(finalize_threads),
                    "effective_finalize_threads": str(finalize_threads),
                    "selector_tiles": str(selector_tiles),
                    "production_specialized": str(
                        selectors_per_tile in {3, 6}
                    ).lower(),
                },
            ),
            event(
                "MetalHammingWeightClaimReduction::allocation_plan",
                2_020.0,
                80.0,
                {
                    "device_buffers": "5",
                    "planned_device_bytes": str(planned_bytes),
                    "current_device_bytes": str(rows * row_bytes),
                    "recommended_device_bytes": str(rows * row_bytes + planned_bytes),
                },
            ),
            event(
                "MetalHammingWeightClaimReduction::dispatch",
                2_210.0,
                190.0,
                {
                    "command_buffers": "1",
                    "tile_dispatches": str(selector_tiles),
                    "finalize_dispatches": str(selector_tiles),
                    "command_completed": "true",
                    "gpu_active_ns": "150000",
                    "resident_rows_storage_id": "401",
                },
            ),
            event(
                "MetalHammingWeightClaimReduction::readback",
                2_410.0,
                90.0,
                {
                    "elements": str(polys * k),
                    "bytes": str(polys * k * 16),
                    "readbacks": "1",
                },
            ),
        ]
    )
    return events


def complete_retained_hamming_trace(log_n: int) -> list[dict[str, object]]:
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

    rows = 1 << log_n
    e_in = 1 << 15
    e_out = rows // e_in
    output_fields = 29 * 256
    producer_partial_fields = e_out * 29 * 256
    producer_owned_bytes = 29 * rows + rows + 16 * (
        e_in + e_out + producer_partial_fields + output_fields
    )
    hamming_partial_fields = e_out * 6 * 256
    hamming_owned_bytes = 16 * (
        e_in + e_out + hamming_partial_fields + output_fields
    )
    current_device_bytes = rows * 40
    source = {
        "resident_rows_storage_id": "401",
        "resident_rows": str(rows),
        "resident_row_bytes": "40",
        "device_registry_id": "17",
        "row_allocations": "0",
        "row_upload_bytes": "0",
    }
    events = [
        event("jolt_prover::piop", 0.0, 10_000.0),
        event("InstructionReadRaf::prepare", 90.0, 100.0),
        event(
            "MetalBooleanityRows::stage5_prepare",
            110.0,
            20.0,
            {
                **source,
                "row_allocations": "1",
                "row_upload_bytes": str(rows * 40),
            },
        ),
        event("BooleanityAddressPhase::prepare", 1_000.0, 600.0),
        event(
            "MetalBooleanityRows::stage6a_address_use",
            1_002.0,
            2.0,
            source,
        ),
        event("MetalBooleanityAddressPhase::prepare", 1_005.0, 500.0),
        event(
            "MetalBooleanityAddressPhase::packed_hot_sequence",
            1_010.0,
            10.0,
            {
                "resident_rows_storage_id": "401",
                "hot_rows_storage_id": "501",
                "rows": str(rows),
                "resident_row_bytes": "40",
                "hot_bytes": str(29 * rows),
                "validity_bytes": str(rows),
                "e_in_fields": str(e_in),
                "e_out_fields": str(e_out),
                "partial_fields": str(producer_partial_fields),
                "output_fields": str(output_fields),
                "owned_bytes": str(producer_owned_bytes),
                "current_device_bytes": str(current_device_bytes),
                "recommended_device_bytes": str(
                    current_device_bytes + producer_owned_bytes
                ),
                "command_buffers": "1",
                "dispatches": "3",
                "readbacks": "1",
            },
        ),
        event(
            "MetalBooleanityAddressPhase::packed_hot_dispatch",
            1_030.0,
            170.0,
            {
                "command_buffers": "1",
                "dispatches": "3",
                "command_completed": "true",
                "gpu_active_ns": "150000",
                "resident_rows_storage_id": "401",
                "hot_rows_storage_id": "501",
            },
        ),
        event(
            "MetalBooleanityAddressPhase::packed_hot_readback",
            1_210.0,
            90.0,
            {
                "elements": str(output_fields),
                "bytes": str(output_fields * 16),
                "readbacks": "1",
            },
        ),
        event("Booleanity::prepare", 2_700.0, 200.0),
        event(
            "MetalHammingHotRows::stage6b_retain_for_stage7",
            2_750.0,
            20.0,
            {
                "hot_rows_storage_id": "501",
                "source_rows_storage_id": "401",
                "hot_rows": str(rows),
                "hot_row_bytes": "29",
                "device_registry_id": "17",
                "row_allocations": "0",
                "row_upload_bytes": "0",
            },
        ),
        event(
            "MetalBooleanityRows::stage6b_cycle_use",
            2_800.0,
            20.0,
            source,
        ),
        event("HammingWeightClaimReduction::prepare", 3_000.0, 600.0),
        event(
            "MetalHammingHotRows::stage7_terminal_use",
            3_005.0,
            400.0,
            {
                "hot_rows_storage_id": "501",
                "source_rows_storage_id": "401",
                "hot_rows": str(rows),
                "hot_row_bytes": "29",
                "device_registry_id": "17",
                "row_allocations": "0",
                "row_upload_bytes": "0",
                "terminal_consumer": "true",
                "terminal_carry_removed": "true",
            },
        ),
        event(
            "MetalHammingWeightClaimReduction::retained_sequence",
            3_010.0,
            10.0,
            {
                "hot_rows_storage_id": "501",
                "source_rows_storage_id": "401",
                "rows": str(rows),
                "hot_bytes": str(29 * rows),
                "e_in_fields": str(e_in),
                "e_out_fields": str(e_out),
                "partial_fields": str(hamming_partial_fields),
                "output_fields": str(output_fields),
                "owned_bytes": str(hamming_owned_bytes),
                "current_device_bytes": str(current_device_bytes),
                "recommended_device_bytes": str(
                    current_device_bytes + hamming_owned_bytes
                ),
                "command_buffers": "1",
                "encoders": "10",
                "dispatches": "10",
                "tile_threadgroups": str(e_out * 5),
                "finalize_threadgroups": "29",
                "readbacks": "1",
            },
        ),
        event(
            "MetalHammingWeightClaimReduction::retained_dispatch",
            3_030.0,
            120.0,
            {
                "command_buffers": "1",
                "tile_dispatches": "5",
                "finalize_dispatches": "5",
                "command_completed": "true",
                "gpu_active_ns": "100000",
                "hot_rows_storage_id": "501",
            },
        ),
        event(
            "MetalHammingWeightClaimReduction::retained_readback",
            3_160.0,
            40.0,
            {
                "elements": str(output_fields),
                "bytes": str(output_fields * 16),
                "readbacks": "1",
            },
        ),
    ]
    for index in range(8):
        address_round = 1_700.0 + 100.0 * index
        hamming_round = 3_700.0 + 100.0 * index
        events.extend(
            [
                event("sumcheck_round", address_round - 5.0, 95.0),
                event("BooleanityAddressPhase::prove_round", address_round, 80.0),
                event("sumcheck_host_fiat_shamir", address_round + 85.0, 5.0),
                event("sumcheck_round", hamming_round - 5.0, 95.0),
                event("HammingWeightClaimReduction::prove_round", hamming_round, 80.0),
                event("sumcheck_host_fiat_shamir", hamming_round + 85.0, 5.0),
            ]
        )
    events.extend(
        [
            event("BooleanityAddressPhase::finish_rounds", 2_500.0, 80.0),
            event("BooleanityAddressPhase::output_claims", 2_600.0, 40.0),
            event("HammingWeightClaimReduction::finish_rounds", 4_500.0, 80.0),
            event("HammingWeightClaimReduction::output_claims", 4_600.0, 40.0),
        ]
    )
    return events


def complete_outer_remainder_trace(
    log_n: int,
    backend: str,
    cutoff_log2: int = 16,
    product_uniskip_carrier: bool = False,
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
        event("jolt_prover::piop", 0.0, 20_000.0),
        event("OuterRemainder::complete_member", 100.0, 10_000.0),
    ]
    round_starts = [200.0 + 300.0 * index for index in range(log_n + 1)]
    for timestamp in round_starts:
        events.append(event("sumcheck_round", timestamp, 240.0))
        events.append(event("sumcheck_host_fiat_shamir", timestamp + 180.0, 20.0))
    if backend == "optimized":
        return events

    rows = 1 << log_n
    geometry = metal_piop_eval.outer_remainder_storage_geometry(
        log_n, product_uniskip_carrier
    )
    ids = list(range(1_001, 1_010))
    storage_args: dict[str, object] = {
        "cycles": rows,
        "planned_device_bytes": geometry["owned_bytes"],
        "maximum_buffer_bytes": geometry["maximum_buffer_bytes"],
        "current_device_bytes": 1_000,
        "recommended_max_working_set_bytes": geometry["owned_bytes"] + 2_000,
        "initialization_mode": "full",
        "admitted": True,
        "initialized": True,
        "fallback_reason": "none",
        "device_buffers": 9,
        "initialization_bytes": geometry["owned_bytes"],
        "initialization_wall_ns": 100,
        "initialization_gpu_active_ns": 80,
        **{f"buffer_{index}": identity for index, identity in enumerate(ids)},
    }
    handoff = {
        "compact_rows_storage_id": 201,
        "residual_rows_storage_id": 202,
        "device_registry_id": 203,
        "resident_rows": rows,
        "row_upload_bytes": 0,
        "device_allocations": 0,
    }
    sequence = {
        "resident_rows": rows,
        "rounds": log_n + 1,
        "cutoff_elements": 1 << cutoff_log2,
        "trace_cutoff_elements": 1 << 18,
        "planned_device_bytes": geometry["owned_bytes"],
        "compact_rows_storage_id": 201,
        "residual_rows_storage_id": 202,
        "device_registry_id": 203,
        "storage_reused": True,
        "storage_initialization_mode": "full",
        "preinitialized_device_bytes": geometry["owned_bytes"],
        "initialization_bytes": geometry["owned_bytes"],
        "attached_owned_bytes": geometry["owned_bytes"],
        "row_upload_bytes": 0,
        "full_domain_copy_dispatches": 0,
        "sequence_device_buffer_allocations": 0,
        "round_device_buffer_allocations": 0,
        **{
            f"storage_buffer_{index}": identity
            for index, identity in enumerate(ids)
        },
    }
    events.extend(
        [
            event("MetalOuterRemainder::storage_prepare", -200.0, 100.0, storage_args),
            event(
                "MetalOuterRemainder::storage_initialize",
                -190.0,
                40.0,
                {
                    "mode": "full",
                    "device_buffers": 9,
                    "bytes": geometry["owned_bytes"],
                    "protocol_dispatches": 0,
                    **{
                        f"buffer_{index}": identity
                        for index, identity in enumerate(ids)
                    },
                },
            ),
            event(
                "MetalOuterRemainder::storage_initialize_complete",
                -145.0,
                5.0,
                {
                    "mode": "full",
                    "command_completed": True,
                    "bytes": geometry["owned_bytes"],
                    "wall_ns": 100,
                    "gpu_active_ns": 80,
                },
            ),
            event("MetalOuterRemainder::prepare", 110.0, 80.0),
            event("MetalOuterRemainder::allocation_plan", 115.0, 10.0),
            event("MetalOuterRemainder::row_handoff", 125.0, 10.0, handoff),
            event("MetalOuterRemainder::sequence_prepare", 135.0, 20.0, sequence),
            event("MetalOuterRemainder::first_message", 160.0, 20.0),
            event("MetalOuterRemainder::first_bind", round_starts[1] + 20.0, 20.0),
        ]
    )
    events.extend(
        event("MetalOuterRemainder::dense_round", round_starts[round] + 20.0, 20.0)
        for round in range(2, log_n - cutoff_log2 + 2)
    )
    handoff_round = log_n - cutoff_log2 + 2
    events.append(
        event(
            "MetalOuterRemainder::readback",
            round_starts[handoff_round] + 10.0,
            5.0,
            {
                "readbacks": 1,
                "elements": 2 * (1 << cutoff_log2),
                "bytes": 2 * (1 << cutoff_log2) * 16,
            },
        )
    )
    events.extend(
        event("MetalOuterRemainder::cpu_tail", round_starts[round] + 20.0, 20.0)
        for round in range(handoff_round, log_n + 1)
    )
    terminal = round_starts[-1] + 250.0
    events.append(event("MetalOuterRemainder::cpu_tail", terminal, 20.0))
    output_elements = 37 if product_uniskip_carrier else 35
    output_args = {
        "dispatch_wall_ns": 100,
        "gpu_active_ns": 80,
        "readbacks": 1,
        "output_elements": output_elements,
        "readback_bytes": output_elements * 16,
        "row_upload_bytes": 0,
    }
    release = {
        **handoff,
        "residual_row_bytes": rows * 112,
        "remaining_sequence_storage_bytes": geometry["owned_bytes"],
        "compact_release_bytes": 0,
        "deferred_owned_bytes": geometry["owned_bytes"] + rows * 112,
        "release_mode": "proof_session_deferred",
        "cleanup_scope": "proof_session",
        "ownership_transfer_completed": True,
        "physical_release_completed": False,
        "residual_released": False,
        "residual_deferred": True,
        "compact_retained": True,
    }
    events.append(event("MetalOuterRemainder::output_claims", terminal + 30.0, 80.0, output_args))
    events.append(event("MetalOuterRemainder::row_release", terminal + 115.0, 20.0, release))
    if product_uniskip_carrier:
        events.append(
            event(
                "MetalOuterRemainder::product_uniskip_carrier_park",
                terminal + 140.0,
                5.0,
                {
                    "rows": rows,
                    "source_rows_storage_id": handoff["compact_rows_storage_id"],
                    "endpoint_elements": 2,
                },
            )
        )
    return events


def complete_product_uniskip_trace(
    log_n: int, backend: str, product_uniskip_carrier: bool
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
        event("jolt_prover::piop", 0.0, 1_000.0),
        event("SpartanProductUniskip::prepare", 100.0, 80.0),
        event("SpartanProductUniskip::first_round_poly", 200.0, 20.0),
    ]
    if backend == "optimized":
        events.extend(
            [
                event("SpartanProductUniskip::prepare", 105.0, 70.0),
                event("SpartanProductUniskip::first_round_poly", 205.0, 10.0),
            ]
        )
        return events
    if product_uniskip_carrier:
        events.append(
            event(
                "MetalProductUniskip::outer_opening_carrier",
                110.0,
                10.0,
                {
                    "cycles": 1 << log_n,
                    "source_rows_storage_id": 201,
                    "product_rows_storage_id": 401,
                    "row_upload_bytes": 0,
                    "dispatches": 0,
                    "command_buffers": 0,
                    "readback_bytes": 0,
                },
            )
        )
    else:
        events.append(
            event(
                "MetalProductUniskip::prepare",
                110.0,
                50.0,
                {
                    "cycles": 1 << log_n,
                    "resident_rows_storage_id": 401,
                    "row_upload_bytes": 0,
                    "round_device_buffer_allocations": 0,
                    "dispatch_wall_ns": 40_000,
                    "gpu_active_ns": 30_000,
                },
            )
        )
    return events


def complete_instruction_claim_trace(
    log_n: int, backend: str
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
        event("jolt_prover::piop", 0.0, 20_000.0),
        event("InstructionClaimReduction::prepare", 100.0, 20.0),
        *[
            event("InstructionClaimReduction::prove_round", 200.0 + round_index, 1.0)
            for round_index in range(log_n)
        ],
        event("InstructionClaimReduction::finish_rounds", 300.0, 1.0),
        event("InstructionClaimReduction::output_claims", 310.0, 2.0),
    ]
    if backend == "optimized":
        return events

    resident_id = 401
    lookup_id = 402
    submit_wall_ns = 40_000
    overlap_wall_ns = 46_000_000
    join_wall_ns = 6_000_000
    lifecycle_wall_ns = submit_wall_ns + overlap_wall_ns + join_wall_ns
    events.extend(
        [
            event(
                "MetalProductRemainder::prepare",
                50.0,
                10.0,
                {
                    "resident_rows_storage_id": str(resident_id),
                    "row_upload_bytes": "0",
                },
            ),
            event(
                "MetalInstructionClaimReduction::first_message_submit",
                101.0,
                1.0,
                {
                    "command_committed": "true",
                    "lookup_rows_storage_id": str(lookup_id),
                    "resident_rows_storage_id": str(resident_id),
                    "submit_wall_ns": str(submit_wall_ns),
                },
            ),
            event(
                "MetalInstructionClaimReduction::prepare",
                102.0,
                1.0,
                {
                    "cycles": str(1 << log_n),
                    "lookup_rows_storage_id": str(lookup_id),
                    "resident_rows_storage_id": str(resident_id),
                    "round_device_buffer_allocations": "0",
                    "rounds": str(log_n),
                    "row_upload_bytes": "0",
                    "workspace_bytes": "1024",
                },
            ),
            event(
                "MetalInstructionClaimReduction::first_message_join",
                200.0,
                2.0,
                {
                    "command_completed": "true",
                    "completed_before_join": "false",
                    "gpu_active_ns": "12000000",
                    "join_wall_ns": str(join_wall_ns),
                    "lifecycle_wall_ns": str(lifecycle_wall_ns),
                    "overlap_wall_ns": str(overlap_wall_ns),
                    "resident_rows_storage_id": str(resident_id),
                    "submit_wall_ns": str(submit_wall_ns),
                },
            ),
            *[
                event(
                    "MetalInstructionClaimReduction::bind_and_message",
                    210.0 + round_index,
                    1.0,
                    {
                        "dispatch_wall_ns": "1000",
                        "gpu_active_ns": "800",
                        "resident_rows_storage_id": str(resident_id),
                        "round": str(round_index),
                        "source_elements": str(
                            1 << (log_n - round_index + 1)
                        ),
                    },
                )
                for round_index in range(1, log_n)
            ],
            event(
                "MetalInstructionClaimReduction::output_claims",
                320.0,
                2.0,
                {
                    "dispatch_wall_ns": "2000",
                    "gpu_active_ns": "1500",
                    "resident_rows_storage_id": str(resident_id),
                    "row_upload_bytes": "0",
                },
            ),
        ]
    )
    return events


class MetalPiopEvalTests(unittest.TestCase):
    def test_instruction_claim_observation_proves_async_residency(self) -> None:
        optimized = metal_piop_eval.instruction_claim_observation(
            complete_instruction_claim_trace(26, "optimized"), "optimized", 26
        )
        self.assertIsNone(optimized["resource_observation"])

        observed = metal_piop_eval.instruction_claim_observation(
            complete_instruction_claim_trace(26, "metal"), "metal", 26
        )
        resources = observed["resource_observation"]
        self.assertEqual(resources["overlap_wall_ns"], 46_000_000)
        self.assertEqual(resources["bind_dispatches"], 25)
        self.assertEqual(
            resources["resident_rows_storage_id"],
            resources["producer_rows_storage_id"],
        )
        self.assertEqual(resources["row_upload_bytes"], 0)

    def test_instruction_claim_observation_rejects_producer_identity_drift(self) -> None:
        events = complete_instruction_claim_trace(26, "metal")
        producer = next(
            event
            for event in events
            if event["name"] == "MetalProductRemainder::prepare"
        )
        producer["args"]["resident_rows_storage_id"] = "999"
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            metal_piop_eval.instruction_claim_observation(events, "metal", 26)

    def test_outer_remainder_complete_member_is_the_local_timing_boundary(self) -> None:
        optimized = metal_piop_eval.outer_remainder_member_breakdown(
            complete_outer_remainder_trace(26, "optimized"), "optimized", 26
        )
        metal = metal_piop_eval.outer_remainder_member_breakdown(
            complete_outer_remainder_trace(26, "metal"), "metal", 26
        )

        self.assertEqual(optimized["components"]["member_us"], 10_000.0)
        self.assertEqual(metal["components"]["member_us"], 10_000.0)
        self.assertEqual(metal["metal_counts"]["dense_round"], 10)
        self.assertEqual(metal["metal_counts"]["cpu_tail"], 16)
        self.assertEqual(metal["outer_counts"]["host_fiat_shamir"], 27)

        carrier = metal_piop_eval.outer_remainder_member_breakdown(
            complete_outer_remainder_trace(
                26, "metal", product_uniskip_carrier=True
            ),
            "metal",
            26,
            product_uniskip_carrier=True,
        )
        self.assertEqual(
            carrier["resource_observation"]["output"]["output_elements"], 37
        )
        self.assertEqual(
            carrier["product_uniskip_carrier"],
            {
                "rows": 1 << 26,
                "source_rows_storage_id": 201,
                "endpoint_elements": 2,
            },
        )

    def test_product_uniskip_requires_exactly_one_execution_path(self) -> None:
        optimized = metal_piop_eval.product_uniskip_observation(
            complete_product_uniskip_trace(26, "optimized", False),
            "optimized",
            26,
            False,
        )
        self.assertEqual(optimized["metal_counts"], {"standalone": 0, "carrier": 0})

        standalone = metal_piop_eval.product_uniskip_observation(
            complete_product_uniskip_trace(26, "metal", False),
            "metal",
            26,
            False,
        )
        self.assertEqual(standalone["metal_counts"], {"standalone": 1, "carrier": 0})
        self.assertEqual(standalone["resource_observation"]["dispatches"], 1)

        carrier = metal_piop_eval.product_uniskip_observation(
            complete_product_uniskip_trace(26, "metal", True),
            "metal",
            26,
            True,
        )
        self.assertEqual(carrier["metal_counts"], {"standalone": 0, "carrier": 1})
        self.assertEqual(carrier["resource_observation"]["dispatches"], 0)

        events = complete_product_uniskip_trace(26, "metal", True)
        events.extend(
            event
            for event in complete_product_uniskip_trace(26, "metal", False)
            if event["name"] == "MetalProductUniskip::prepare"
        )
        with self.assertRaisesRegex(ValueError, "execution path"):
            metal_piop_eval.product_uniskip_observation(
                events, "metal", 26, True
            )

    def test_outer_remainder_rejects_storage_identity_drift(self) -> None:
        events = complete_outer_remainder_trace(26, "metal")
        sequence = next(
            event
            for event in events
            if event["name"] == "MetalOuterRemainder::sequence_prepare"
        )
        sequence["args"]["storage_buffer_8"] = 9999
        with self.assertRaisesRegex(ValueError, "resident sequence"):
            metal_piop_eval.outer_remainder_member_breakdown(
                events, "metal", 26
            )

    def test_validates_outer_remainder_runtime_config(self) -> None:
        stdout = (
            "OUTER_REMAINDER_METAL_CONFIG backend=metal trace_cutoff=262144 "
            "cutoff=65536 materialize_threads=256 transition_threads=128 "
            "output_threads=256 max_threadgroups=8192 binding_plan=b_only_v1 "
            "storage_initialization=full product_uniskip_carrier=false"
        )
        config = metal_piop_eval.validate_outer_remainder_stdout(
            stdout, "metal"
        )
        self.assertEqual(config["cutoff"], 1 << 16)
        self.assertEqual(config["binding_plan"], "b_only_v1")
        self.assertFalse(config["product_uniskip_carrier"])
        self.assertIsNone(
            metal_piop_eval.validate_outer_remainder_stdout("", "optimized")
        )

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
        pairs = [
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
                    "cpu_booleanity_address_us": 60.0,
                    "metal_booleanity_address_us": 10.0,
                    "cpu_hamming_weight_us": 45.0,
                    "metal_hamming_weight_us": 9.0,
                    "cpu_hamming_weight_service_us": 50.0,
                    "metal_hamming_weight_service_us": 10.0,
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
                    "cpu_booleanity_address_us": 70.0,
                    "metal_booleanity_address_us": 14.0,
                    "cpu_hamming_weight_us": 36.0,
                    "metal_hamming_weight_us": 12.0,
                    "cpu_hamming_weight_service_us": 42.0,
                    "metal_hamming_weight_service_us": 14.0,
                },
            ]
        metrics = metal_piop_eval.summarize_pairs(pairs)
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
        self.assertEqual(metrics["paired_booleanity_address_phase_speedups"], [6.0, 5.0])
        self.assertEqual(metrics["booleanity_address_phase_speedup"], 5.5)
        self.assertEqual(
            metrics["paired_hamming_weight_claim_reduction_speedups"], [5.0, 3.0]
        )
        self.assertEqual(metrics["hamming_weight_claim_reduction_speedup"], 4.0)
        self.assertFalse(metrics["bytecode_read_raf_cycle_decision"]["enough_pairs"])
        self.assertEqual(metrics["piop_speedup"], 4.5)
        self.assertEqual(metrics["paired_speedups_with_backend_witness_prepare"], [4.0, 3.0])
        self.assertEqual(metrics["piop_plus_backend_witness_prepare_speedup"], 3.5)

        incomplete = [dict(pair) for pair in pairs]
        incomplete[0].pop("cpu_hamming_weight_us")
        with self.assertRaisesRegex(ValueError, "missing Hamming-weight timing"):
            metal_piop_eval.summarize_pairs(incomplete)

    def test_instruction_claim_gate_requires_critical_path_and_family(self) -> None:
        pair = {
            "cpu_us": 2_000.0,
            "metal_us": 500.0,
            "cpu_prepare_us": 1.0,
            "metal_prepare_us": 1.0,
            "cpu_instruction_ra_us": 700.0,
            "metal_instruction_ra_us": 100.0,
            "cpu_bytecode_us": 1_000.0,
            "metal_bytecode_us": 200.0,
            "cpu_instruction_input_us": 800.0,
            "metal_instruction_input_us": 160.0,
            "cpu_booleanity_address_us": 1_000.0,
            "metal_booleanity_address_us": 200.0,
            "cpu_hamming_weight_us": 900.0,
            "metal_hamming_weight_us": 180.0,
            "cpu_hamming_weight_service_us": 990.0,
            "metal_hamming_weight_service_us": 180.0,
            "cpu_outer_remainder_us": 600.0,
            "metal_outer_remainder_us": 120.0,
            "cpu_product_uniskip_us": 200.0,
            "metal_product_uniskip_us": 10.0,
            "cpu_product_remainder_us": 400.0,
            "metal_product_remainder_us": 40.0,
            "cpu_instruction_claim_us": 300.0,
            "metal_instruction_claim_us": 30.0,
            "metal_instruction_claim_isolated_service_us": 80.0,
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
        metrics = metal_piop_eval.summarize_pairs(pairs)
        self.assertEqual(
            metrics["instruction_claim_reduction_critical_path_speedup"], 10.0
        )
        self.assertEqual(
            metrics["instruction_claim_reduction_isolated_service_speedup"], 3.75
        )
        self.assertEqual(metrics["product_instruction_claim_family_speedup"], 10.0)
        self.assertAlmostEqual(
            metrics["outer_product_family_speedup"], 1200.0 / 170.0
        )
        self.assertTrue(
            metrics["instruction_claim_reduction_critical_path_decision"]["clears"]
        )
        self.assertTrue(
            metrics["product_instruction_claim_family_decision"]["clears"]
        )
        self.assertTrue(metrics["outer_product_family_decision"]["clears"])

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

    def test_validates_booleanity_address_runtime_record(self) -> None:
        record = "\n".join(
            [
                "BOOLEANITY_ADDRESS_METAL_CONFIG backend=metal trace_cutoff=262144 inner_log2=15 selectors_per_tile=6 tile_threads=512 finalize_threads=1024",
                "BOOLEANITY_ADDRESS_METAL_IMPLEMENTATION value=accepted",
            ]
        )
        self.assertIsNone(
            metal_piop_eval.validate_booleanity_address_stdout("", "optimized")
        )
        observed = metal_piop_eval.validate_booleanity_address_stdout(record, "metal")
        assert observed is not None
        self.assertEqual(observed["inner_log2"], 15)
        self.assertEqual(observed["selectors_per_tile"], 6)
        self.assertEqual(observed["implementation"], "accepted")
        packed = record.replace("value=accepted", "value=packed-hot")
        self.assertEqual(
            metal_piop_eval.validate_booleanity_address_stdout(
                packed, "metal", implementation="packed-hot"
            )["implementation"],
            "packed-hot",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            metal_piop_eval.validate_booleanity_address_stdout(
                record + "\n" + record, "metal"
            )

    def test_validates_hamming_weight_runtime_record(self) -> None:
        record = "\n".join(
            [
                "HAMMING_WEIGHT_METAL_CONFIG backend=metal trace_cutoff=262144 inner_log2=15 selectors_per_tile=6 tile_threads=512 finalize_threads=1024",
                "HAMMING_WEIGHT_METAL_IMPLEMENTATION value=accepted-rows",
            ]
        )
        self.assertIsNone(
            metal_piop_eval.validate_hamming_weight_stdout("", "optimized")
        )
        observed = metal_piop_eval.validate_hamming_weight_stdout(record, "metal")
        assert observed is not None
        self.assertEqual(observed["inner_log2"], 15)
        self.assertEqual(observed["selectors_per_tile"], 6)
        self.assertEqual(observed["implementation"], "accepted-rows")
        retained = record.replace("value=accepted-rows", "value=retained-hot")
        self.assertEqual(
            metal_piop_eval.validate_hamming_weight_stdout(
                retained, "metal", implementation="retained-hot"
            )["implementation"],
            "retained-hot",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            metal_piop_eval.validate_hamming_weight_stdout(
                record + "\n" + record, "metal"
            )

    def test_requires_pinned_production_rayon_width(self) -> None:
        record = "PIOP_EXECUTION_CONFIG rayon_threads=16"
        self.assertEqual(
            metal_piop_eval.validate_piop_execution_stdout(record),
            {"rayon_threads": 16},
        )
        with self.assertRaisesRegex(ValueError, "Rayon width"):
            metal_piop_eval.validate_piop_execution_stdout(
                "PIOP_EXECUTION_CONFIG rayon_threads=15"
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            metal_piop_eval.validate_piop_execution_stdout("")

    def test_requires_exact_booleanity_address_member_resources_and_lifecycle(
        self,
    ) -> None:
        optimized = metal_piop_eval.booleanity_address_member_breakdown(
            complete_booleanity_address_trace(26, "optimized"),
            "optimized",
            26,
            15,
            6,
            512,
            1024,
        )
        self.assertIsNone(optimized["row_lifecycle"])
        self.assertTrue(not any(optimized["metal_counts"].values()))
        self.assertEqual(optimized["components"]["row_source_us"], 300.0)
        self.assertEqual(optimized["components"]["normalized_prepare_us"], 300.0)
        self.assertEqual(
            optimized["components"]["normalized_member_us"],
            optimized["components"]["member_us"] - 300.0,
        )

        observed = metal_piop_eval.booleanity_address_member_breakdown(
            complete_booleanity_address_trace(26, "metal"),
            "metal",
            26,
            15,
            6,
            512,
            1024,
        )
        self.assertEqual(observed["outer_counts"]["prove_round"], 8)
        self.assertEqual(
            observed["components"]["host_fiat_shamir_us"], [5.0] * 8
        )
        self.assertEqual(observed["components"]["host_fiat_shamir_total_us"], 40.0)
        self.assertEqual(observed["components"]["row_source_us"], 0.0)
        self.assertEqual(
            observed["components"]["normalized_member_us"],
            observed["components"]["member_us"],
        )
        self.assertEqual(
            observed["metal_counts"],
            {
                "prepare": 1,
                "sequence_prepare": 1,
                "allocation_plan": 1,
                "dispatch": 1,
                "readback": 1,
            },
        )
        self.assertEqual(
            observed["resource_observation"]["allocation"]["planned_device_bytes"],
            51_007_720,
        )
        self.assertEqual(
            observed["resource_observation"]["readback"],
            {"elements": 7_424, "bytes": 118_784, "readbacks": 1},
        )
        self.assertEqual(
            observed["row_lifecycle"],
            {
                "kind": "metal_booleanity_resident",
                "rows": 1 << 26,
                "row_bytes": 40,
                "device_registry_id": 17,
                "stage5_storage_id": 401,
                "stage6a_storage_id": 401,
                "stage6b_storage_id": 401,
                "stage5": {
                    "row_allocations": 1,
                    "row_upload_bytes": 40 * (1 << 26),
                },
                "stage6a": {"row_allocations": 0, "row_upload_bytes": 0},
                "stage6b": {"row_allocations": 0, "row_upload_bytes": 0},
            },
        )

        tuned = metal_piop_eval.booleanity_address_member_breakdown(
            complete_booleanity_address_trace(25, "metal", 14, 4, 256, 512),
            "metal",
            25,
            14,
            4,
            256,
            512,
        )
        self.assertFalse(
            tuned["resource_observation"]["sequence"]["production_specialized"]
        )
        self.assertEqual(
            tuned["resource_observation"]["allocation"]["planned_device_bytes"],
            metal_piop_eval.booleanity_address_sequence_storage_bytes(25, 14, 4),
        )

    def test_requires_exact_hamming_weight_member_resources_and_terminal_lifecycle(
        self,
    ) -> None:
        optimized = metal_piop_eval.hamming_weight_member_breakdown(
            complete_hamming_weight_trace(26, "optimized"),
            "optimized",
            26,
        )
        self.assertIsNone(optimized["row_lifecycle"])
        self.assertEqual(optimized["components"]["row_source_us"], 300.0)
        self.assertEqual(
            optimized["components"]["normalized_member_us"],
            optimized["components"]["member_us"] - 300.0,
        )

        observed = metal_piop_eval.hamming_weight_member_breakdown(
            complete_hamming_weight_trace(26, "metal"),
            "metal",
            26,
        )
        self.assertEqual(observed["components"]["row_source_us"], 0.0)
        self.assertEqual(
            observed["components"]["normalized_member_us"],
            observed["components"]["member_us"],
        )
        self.assertEqual(
            observed["metal_counts"],
            {
                "prepare": 1,
                "sequence_prepare": 1,
                "allocation_plan": 1,
                "dispatch": 1,
                "readback": 1,
            },
        )
        self.assertEqual(
            observed["resource_observation"]["readback"],
            {"elements": 7_424, "bytes": 118_784, "readbacks": 1},
        )
        self.assertEqual(
            observed["row_lifecycle"],
            {
                "kind": "metal_hamming_resident",
                "rows": 1 << 26,
                "row_bytes": 40,
                "device_registry_id": 17,
                "stage5_storage_id": 401,
                "stage6a_storage_id": 401,
                "stage6b_storage_id": 401,
                "stage6b_retain_storage_id": 401,
                "stage7_storage_id": 401,
                "stage5": {
                    "row_allocations": 1,
                    "row_upload_bytes": 40 * (1 << 26),
                },
                "stage6a": {"row_allocations": 0, "row_upload_bytes": 0},
                "stage6b": {"row_allocations": 0, "row_upload_bytes": 0},
                "stage6b_retain": {
                    "row_allocations": 0,
                    "row_upload_bytes": 0,
                },
                "stage7": {"row_allocations": 0, "row_upload_bytes": 0},
                "terminal_consumer": True,
                "terminal_carry_removed": True,
            },
        )

    def test_requires_exact_packed_hot_retained_hamming_contract(self) -> None:
        events = complete_retained_hamming_trace(26)
        producer = metal_piop_eval.packed_hot_booleanity_address_member_breakdown(
            events, "metal", 26
        )
        consumer = metal_piop_eval.retained_hot_hamming_weight_member_breakdown(
            events, "metal", 26
        )

        self.assertEqual(
            producer["metal_counts"],
            {"prepare": 1, "sequence": 1, "dispatch": 1, "readback": 1},
        )
        self.assertEqual(
            consumer["metal_counts"],
            {"sequence": 1, "dispatch": 1, "readback": 1},
        )
        self.assertEqual(
            producer["resource_observation"]["sequence"]["owned_bytes"],
            2_257_211_392,
        )
        self.assertEqual(
            consumer["resource_observation"]["sequence"]["owned_bytes"],
            51_007_488,
        )
        self.assertEqual(
            consumer["resource_observation"]["dispatch"]["tile_dispatches"], 5
        )
        self.assertEqual(producer["row_lifecycle"], consumer["row_lifecycle"])
        self.assertEqual(producer["row_lifecycle"]["kind"], "metal_hamming_hot")
        self.assertEqual(producer["row_lifecycle"]["source_rows_storage_id"], 401)
        self.assertEqual(producer["row_lifecycle"]["hot_rows_storage_id"], 501)
        self.assertTrue(
            producer["row_lifecycle"]["stage7"]["terminal_carry_removed"]
        )

        missing_terminal = [dict(event) for event in events]
        missing_terminal[:] = [
            event
            for event in missing_terminal
            if event.get("name") != "MetalHammingHotRows::stage7_terminal_use"
        ]
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            metal_piop_eval.retained_hot_hamming_weight_member_breakdown(
                missing_terminal, "metal", 26
            )

        wrong_geometry = [
            {**event, "args": dict(event["args"])} if "args" in event else dict(event)
            for event in events
        ]
        sequence = next(
            event
            for event in wrong_geometry
            if event.get("name")
            == "MetalHammingWeightClaimReduction::retained_sequence"
        )
        sequence["args"]["partial_fields"] = "1"
        with self.assertRaisesRegex(ValueError, "geometry"):
            metal_piop_eval.retained_hot_hamming_weight_member_breakdown(
                wrong_geometry, "metal", 26
            )

    def test_rejects_hamming_weight_trace_contract_drift(self) -> None:
        def named(
            events: list[dict[str, object]], name: str
        ) -> dict[str, object]:
            return next(event for event in events if event.get("name") == name)

        def remove_named(events: list[dict[str, object]], name: str) -> None:
            events[:] = [event for event in events if event.get("name") != name]

        optimized = complete_hamming_weight_trace(26, "optimized")
        remove_named(
            optimized, "OptimizedHammingWeightClaimReduction::row_source"
        )
        with self.assertRaisesRegex(ValueError, "row-source"):
            metal_piop_eval.hamming_weight_member_breakdown(
                optimized, "optimized", 26
            )

        cases = (
            (
                "missing retain",
                lambda events: remove_named(
                    events, "MetalBooleanityRows::stage6b_retain_for_stage7"
                ),
                "lifecycle",
            ),
            (
                "wrong terminal storage",
                lambda events: named(
                    events, "MetalBooleanityRows::stage7_hamming_use"
                )["args"].__setitem__("resident_rows_storage_id", "402"),
                "lifecycle",
            ),
            (
                "terminal upload",
                lambda events: named(
                    events, "MetalBooleanityRows::stage7_hamming_use"
                )["args"].__setitem__("row_upload_bytes", "1"),
                "lifecycle",
            ),
            (
                "terminal carry retained",
                lambda events: named(
                    events, "MetalBooleanityRows::stage7_hamming_use"
                )["args"].__setitem__("terminal_carry_removed", "false"),
                "lifecycle",
            ),
            (
                "terminal use before sequence completes",
                lambda events: named(
                    events, "MetalBooleanityRows::stage7_hamming_use"
                ).__setitem__("ts", 2_100.0),
                "out of order",
            ),
            (
                "terminal use starts after dispatch",
                lambda events: named(
                    events, "MetalBooleanityRows::stage7_hamming_use"
                ).__setitem__("ts", 2_220.0),
                "dispatch",
            ),
            (
                "terminal use ends before readback",
                lambda events: named(
                    events, "MetalBooleanityRows::stage7_hamming_use"
                ).__setitem__("dur", 200.0),
                "readback",
            ),
            (
                "command incomplete",
                lambda events: named(
                    events, "MetalHammingWeightClaimReduction::dispatch"
                )["args"].__setitem__("command_completed", "false"),
                "command",
            ),
            (
                "wrong dispatch count",
                lambda events: named(
                    events, "MetalHammingWeightClaimReduction::dispatch"
                )["args"].__setitem__("tile_dispatches", "4"),
                "dispatch",
            ),
            (
                "wrong readback",
                lambda events: named(
                    events, "MetalHammingWeightClaimReduction::readback"
                )["args"].__setitem__("bytes", "1"),
                "readback",
            ),
            (
                "unknown phase",
                lambda events: events.append(
                    {
                        "name": "MetalHammingWeightClaimReduction::hidden_copy",
                        "ph": "X",
                        "pid": 1,
                        "tid": 0,
                        "ts": 2_100.0,
                        "dur": 1.0,
                    }
                ),
                "unknown Metal phases",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name):
                events = complete_hamming_weight_trace(26, "metal")
                mutate(events)
                with self.assertRaisesRegex(ValueError, message):
                    metal_piop_eval.hamming_weight_member_breakdown(
                        events, "metal", 26
                    )

        metal_with_cpu_source = complete_hamming_weight_trace(26, "metal")
        metal_with_cpu_source.append(
            {
                "name": "OptimizedHammingWeightClaimReduction::row_source",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 2_050.0,
                "dur": 1.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "optimized row-source"):
            metal_piop_eval.hamming_weight_member_breakdown(
                metal_with_cpu_source, "metal", 26
            )

    def test_rejects_booleanity_address_trace_contract_drift(self) -> None:
        cases = [
            (
                "unknown Metal phase",
                "unknown Metal phases",
                lambda events: events.append(
                    {
                        "name": "MetalBooleanityAddressPhase::hidden_copy",
                        "ph": "X",
                        "pid": 1,
                        "tid": 0,
                        "ts": 1_100.0,
                        "dur": 1.0,
                    }
                ),
            ),
            (
                "mismatched lifecycle identity",
                "row lifecycle",
                lambda events: next(
                    event
                    for event in events
                    if event["name"] == "MetalBooleanityRows::stage6a_address_use"
                )["args"].update({"resident_rows_storage_id": "402"}),
            ),
            (
                "hidden stage-6a upload",
                "row lifecycle",
                lambda events: next(
                    event
                    for event in events
                    if event["name"] == "MetalBooleanityRows::stage6a_address_use"
                )["args"].update({"row_upload_bytes": "40"}),
            ),
            (
                "wrong allocation",
                "buffer accounting",
                lambda events: next(
                    event
                    for event in events
                    if event["name"]
                    == "MetalBooleanityAddressPhase::allocation_plan"
                )["args"].update({"planned_device_bytes": "1"}),
            ),
            (
                "wrong dispatch count",
                "dispatch accounting",
                lambda events: next(
                    event
                    for event in events
                    if event["name"] == "MetalBooleanityAddressPhase::dispatch"
                )["args"].update({"tile_dispatches": "4"}),
            ),
            (
                "incomplete command",
                "command did not complete",
                lambda events: next(
                    event
                    for event in events
                    if event["name"] == "MetalBooleanityAddressPhase::dispatch"
                )["args"].update({"command_completed": "false"}),
            ),
            (
                "wrong readback",
                "readback accounting",
                lambda events: next(
                    event
                    for event in events
                    if event["name"] == "MetalBooleanityAddressPhase::readback"
                )["args"].update({"bytes": "16"}),
            ),
            (
                "dispatch outside prepare",
                "not contained",
                lambda events: next(
                    event
                    for event in events
                    if event["name"] == "MetalBooleanityAddressPhase::dispatch"
                ).update({"ts": 1_590.0}),
            ),
        ]
        for label, error, mutate in cases:
            with self.subTest(label=label):
                events = complete_booleanity_address_trace(26, "metal")
                mutate(events)
                with self.assertRaisesRegex(ValueError, error):
                    metal_piop_eval.booleanity_address_member_breakdown(
                        events, "metal", 26, 15, 6, 512, 1024
                    )

    def test_rejects_incomplete_booleanity_address_member(self) -> None:
        missing_row_source = complete_booleanity_address_trace(26, "optimized")
        missing_row_source.remove(
            next(
                event
                for event in missing_row_source
                if event["name"] == "OptimizedBooleanityAddress::row_source"
            )
        )
        with self.assertRaisesRegex(ValueError, "row-source span"):
            metal_piop_eval.booleanity_address_member_breakdown(
                missing_row_source, "optimized", 26, 15, 6, 512, 1024
            )

        missing_round = complete_booleanity_address_trace(26, "metal")
        missing_round.remove(
            next(
                event
                for event in missing_round
                if event["name"] == "BooleanityAddressPhase::prove_round"
            )
        )
        with self.assertRaisesRegex(ValueError, "member span counts"):
            metal_piop_eval.booleanity_address_member_breakdown(
                missing_round, "metal", 26, 15, 6, 512, 1024
            )

        missing_fiat_shamir = complete_booleanity_address_trace(26, "metal")
        missing_fiat_shamir.remove(
            next(
                event
                for event in missing_fiat_shamir
                if event["name"] == "sumcheck_host_fiat_shamir"
            )
        )
        with self.assertRaisesRegex(ValueError, "host Fiat-Shamir"):
            metal_piop_eval.booleanity_address_member_breakdown(
                missing_fiat_shamir, "metal", 26, 15, 6, 512, 1024
            )

        missing_lifecycle = complete_booleanity_address_trace(26, "metal")
        missing_lifecycle.remove(
            next(
                event
                for event in missing_lifecycle
                if event["name"] == "MetalBooleanityRows::stage6b_cycle_use"
            )
        )
        with self.assertRaisesRegex(ValueError, "row lifecycle"):
            metal_piop_eval.booleanity_address_member_breakdown(
                missing_lifecycle, "metal", 26, 15, 6, 512, 1024
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

        booleanity_address = metal_piop_eval.booleanity_address_member_breakdown(
            complete_booleanity_address_trace(26, "metal"),
            "metal",
            26,
            15,
            6,
            512,
            1024,
        )
        booleanity_record = metal_piop_eval.member_record(booleanity_address)
        self.assertNotIn("service_ns", booleanity_record)
        self.assertEqual(booleanity_record["host_fiat_shamir_ns"], [5_000] * 8)
        self.assertEqual(booleanity_record["host_fiat_shamir_total_ns"], 40_000)
        self.assertEqual(
            booleanity_record["member_ns"],
            booleanity_record["prepare_ns"]
            + booleanity_record["rounds_total_ns"]
            + booleanity_record["host_fiat_shamir_total_ns"]
            + booleanity_record["finish_ns"]
            + booleanity_record["output_claims_ns"],
        )
        self.assertEqual(booleanity_record["row_source_ns"], 0)
        self.assertEqual(
            booleanity_record["normalized_member_ns"],
            booleanity_record["member_ns"],
        )
        optimized_booleanity = metal_piop_eval.booleanity_address_member_breakdown(
            complete_booleanity_address_trace(26, "optimized"),
            "optimized",
            26,
            15,
            6,
            512,
            1024,
        )
        optimized_record = metal_piop_eval.member_record(optimized_booleanity)
        self.assertEqual(optimized_record["row_source_ns"], 300_000)
        self.assertEqual(
            optimized_record["normalized_member_ns"],
            optimized_record["member_ns"] - optimized_record["row_source_ns"],
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
            "cpu_booleanity_address_us": 1_000.0,
            "metal_booleanity_address_us": 200.0,
            "cpu_hamming_weight_us": 900.0,
            "metal_hamming_weight_us": 180.0,
            "cpu_hamming_weight_service_us": 990.0,
            "metal_hamming_weight_service_us": 180.0,
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
        booleanity_decision = metal_piop_eval.summarize_pairs(pairs)[
            "booleanity_address_phase_decision"
        ]
        self.assertTrue(booleanity_decision["clears"])
        self.assertEqual(booleanity_decision["median_speedup"], 5.0)
        family_decision = metal_piop_eval.summarize_pairs(pairs)[
            "booleanity_hamming_family_decision"
        ]
        self.assertTrue(family_decision["clears"])
        self.assertEqual(family_decision["median_speedup"], 5.0)

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
                    "cpu_booleanity_address_us": 1_000.0,
                    "metal_booleanity_address_us": 200.0,
                    "cpu_hamming_weight_us": 900.0,
                    "metal_hamming_weight_us": 180.0,
                    "cpu_hamming_weight_service_us": 990.0,
                    "metal_hamming_weight_service_us": 180.0,
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
                "cpu_booleanity_address_us": 1_000.0,
                "metal_booleanity_address_us": 200.0,
                "cpu_hamming_weight_us": 900.0,
                "metal_hamming_weight_us": 180.0,
                "cpu_hamming_weight_service_us": 990.0,
                "metal_hamming_weight_service_us": 180.0,
            }
            for index in range(5)
        ]
        decision = metal_piop_eval.summarize_pairs(pairs)[
            "instruction_input_kernel_service_decision"
        ]
        self.assertTrue(decision["clears"])
        self.assertEqual(decision["median_speedup"], 5.0)

    def test_booleanity_address_gate_requires_both_order_strata(self) -> None:
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
                    "metal_bytecode_us": 200.0,
                    "cpu_instruction_input_us": 800.0,
                    "metal_instruction_input_us": 160.0,
                    "cpu_booleanity_address_us": 1_000.0,
                    "metal_booleanity_address_us": 10_000.0
                    if metal_first
                    else 200.0,
                    "cpu_hamming_weight_us": 900.0,
                    "metal_hamming_weight_us": 180.0,
                    "cpu_hamming_weight_service_us": 990.0,
                    "metal_hamming_weight_service_us": 180.0,
                }
            )
        decision = metal_piop_eval.summarize_pairs(pairs)[
            "booleanity_address_phase_decision"
        ]
        self.assertEqual(decision["median_speedup"], 5.0)
        self.assertEqual(decision["optimized_first_median_speedup"], 5.0)
        self.assertEqual(decision["metal_first_median_speedup"], 0.1)
        self.assertFalse(decision["clears_order_strata"])
        self.assertFalse(decision["clears"])

    def test_hamming_weight_gate_requires_both_order_strata(self) -> None:
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
                    "metal_bytecode_us": 200.0,
                    "cpu_instruction_input_us": 800.0,
                    "metal_instruction_input_us": 160.0,
                    "cpu_booleanity_address_us": 1_000.0,
                    "metal_booleanity_address_us": 200.0,
                    "cpu_hamming_weight_us": 1_000.0,
                    "metal_hamming_weight_us": 10_000.0
                    if metal_first
                    else 200.0,
                    "cpu_hamming_weight_service_us": 1_100.0,
                    "metal_hamming_weight_service_us": 10_000.0
                    if metal_first
                    else 200.0,
                }
            )
        decision = metal_piop_eval.summarize_pairs(pairs)[
            "hamming_weight_claim_reduction_decision"
        ]
        self.assertEqual(decision["median_speedup"], 5.0)
        self.assertEqual(decision["optimized_first_median_speedup"], 5.0)
        self.assertEqual(decision["metal_first_median_speedup"], 0.1)
        self.assertFalse(decision["clears_order_strata"])
        self.assertFalse(decision["clears"])
        family_decision = metal_piop_eval.summarize_pairs(pairs)[
            "booleanity_hamming_family_decision"
        ]
        self.assertFalse(family_decision["clears_order_strata"])
        self.assertFalse(family_decision["clears"])

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
