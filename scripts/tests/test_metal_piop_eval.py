import copy
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


def bytecode_address_stage1_source(log_n: int) -> dict[str, object]:
    return {
        "source_generation": 7,
        "completion_serial": 11,
        "row_allocation_identity": 101,
        "claim_allocation_identity": 102,
        "device_registry_id": 55,
        "source_windows": 1 << log_n,
        "explicit_rows": (1 << log_n) - 123,
    }


def complete_bytecode_address_trace(
    log_n: int, backend: str, outer_tiles: int = 8
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

    rows = 1 << log_n
    addresses = 1 << 13
    stages = 9
    inner = 1 << 15
    outer = rows // inner
    physical_rows = rows - 123
    work_items = (physical_rows + 4095) // 4096 + 17
    address_offset_bytes = 4 * (addresses + 1)
    carrier_resident_bytes = (
        10 * physical_rows + 8 * work_items + address_offset_bytes
    )
    producer_logical_movement_bytes = (
        30 * physical_rows + 16 * work_items + address_offset_bytes
    )
    equality_bytes = 16 * stages * (inner + outer)
    padding_bytes = 5 * 16
    partial_bytes = 16 * stages * work_items
    output_bytes = 16 * stages * addresses
    events = [
        event("jolt_prover::backend_witness_prepare", 0.0, 80.0),
        event("jolt_prover::piop", 100.0, 5_000.0),
        event("BytecodeReadRafAddressPhase::prepare", 200.0, 500.0),
    ]
    round_starts = [800.0 + 100.0 * index for index in range(13)]
    events.extend(
        event("BytecodeReadRafAddressPhase::prove_round", timestamp, 80.0)
        for timestamp in round_starts
    )
    events.extend(
        [
            event("BytecodeReadRafAddressPhase::finish_rounds", 2_100.0, 80.0),
            event("BytecodeReadRafAddressPhase::output_claims", 2_200.0, 40.0),
        ]
    )
    if backend == "metal":
        source = bytecode_address_stage1_source(log_n)
        publish = {
            "cycles": str(rows),
            "physical_rows": str(physical_rows),
            "work_items": str(work_items),
            "source_generation": str(source["source_generation"]),
            "source_completion_serial": str(source["completion_serial"]),
            "source_rows_storage_id": str(source["row_allocation_identity"]),
            "source_claim_storage_id": str(source["claim_allocation_identity"]),
            "source_device_registry_id": str(source["device_registry_id"]),
            "source_windows": str(rows),
            "carrier_completion_serial": "11",
            "carrier_occurrence_storage_id": "201",
            "carrier_occurrence_bytes": str(2 * physical_rows),
            "carrier_magnitude_storage_id": "202",
            "carrier_magnitude_bytes": str(8 * physical_rows),
            "carrier_work_item_storage_id": "203",
            "carrier_work_item_bytes": str(8 * work_items),
            "carrier_address_offset_storage_id": "204",
            "carrier_address_offset_bytes": str(address_offset_bytes),
            "carrier_resident_bytes": str(carrier_resident_bytes),
            "carrier_allocations": "4",
            "producer_persistent_write_bytes": str(carrier_resident_bytes),
            "producer_logical_movement_bytes": str(
                producer_logical_movement_bytes
            ),
            "producer_topology_read_bytes": "0",
            "shared_source_row_scans": "1",
            "additional_source_row_scans": "0",
            "member_source_upload_bytes": "0",
            "complete_overwrite": "true",
            "covered_rows": str(physical_rows),
        }
        complete = {
            "cycles": str(rows),
            "addresses": str(addresses),
            "stages": str(stages),
            "physical_rows": str(physical_rows),
            "work_items": str(work_items),
            "requested": "address_major",
            "realized_route": "address_major",
            "fallback_reason": "none",
            "source_generation": str(source["source_generation"]),
            "source_completion_serial": str(source["completion_serial"]),
            "source_rows_storage_id": str(source["row_allocation_identity"]),
            "source_rows_bytes": str(40 * rows),
            "source_claim_storage_id": str(source["claim_allocation_identity"]),
            "source_device_registry_id": str(source["device_registry_id"]),
            "carrier_completion_serial": "11",
            "carrier_occurrence_storage_id": "201",
            "carrier_occurrence_bytes": str(2 * physical_rows),
            "carrier_magnitude_storage_id": "202",
            "carrier_magnitude_bytes": str(8 * physical_rows),
            "carrier_work_item_storage_id": "203",
            "carrier_work_item_bytes": str(8 * work_items),
            "carrier_address_offset_storage_id": "204",
            "carrier_address_offset_bytes": str(address_offset_bytes),
            "carrier_resident_bytes": str(carrier_resident_bytes),
            "producer_persistent_write_bytes": str(carrier_resident_bytes),
            "producer_logical_movement_bytes": str(
                producer_logical_movement_bytes
            ),
            "producer_topology_read_bytes": "0",
            "member_carrier_owned_bytes": "0",
            "member_source_scans": "0",
            "member_source_upload_bytes": "0",
            "equality_bytes": str(equality_bytes),
            "padding_bytes": str(padding_bytes),
            "partial_bytes": str(partial_bytes),
            "output_readback_bytes": str(output_bytes),
            "member_owned_bytes": str(
                equality_bytes + padding_bytes + partial_bytes + output_bytes
            ),
            "command_buffers": "1",
            "waits": "1",
            "worker_dispatches": "1",
            "worker_variant": "packed4_halfwidth_v1",
            "worker_simd_width": "32",
            "worker_threads": "128",
            "worker_items_per_threadgroup": "4",
            "worker_threadgroups": str((work_items + 3) // 4),
            "worker_tail_slots": str((4 - work_items % 4) % 4),
            "worker_dynamic_threadgroup_bytes": "0",
            "worker_static_threadgroup_bytes": "0",
            "worker_threadgroup_bytes": "0",
            "reducer_dispatches": "1",
            "reducer_threads": "256",
            "reducer_threadgroups": str((stages * addresses + 255) // 256),
            "reducer_static_threadgroup_bytes": "0",
            "output_fields": str(stages * addresses),
            "submit_ns": "0",
            "overlap_ns": "0",
            "join_ns": "600",
            "resident_wall_ns": "600",
            "gpu_active_ns": "500",
            "completed_before_join": "false",
            "complete_overwrite": "true",
            "carrier_released": "true",
        }
        events.extend(
            [
                event(
                    "MetalBytecodeReadRafAddress::carrier_publish",
                    10.0,
                    1.0,
                    publish,
                ),
                event(
                    "MetalBytecodeReadRafAddress::route",
                    210.0,
                    480.0,
                    {
                        "cycles": str(rows),
                        "requested": "address_major",
                        "realized_route": "address_major",
                        "fallback_reason": "none",
                    },
                ),
                event(
                    "MetalBytecodeReadRafAddress::address_major_prepare",
                    220.0,
                    80.0,
                ),
                event(
                    "MetalBytecodeReadRafAddress::address_major_join",
                    320.0,
                    350.0,
                ),
                event(
                    "MetalBytecodeReadRafAddress::address_major_complete",
                    400.0,
                    1.0,
                    complete,
                ),
            ]
        )
    return events


def fused_bytecode_stage1_scatter(log_n: int) -> dict[str, object]:
    rows = 1 << log_n
    physical_rows = rows - 123
    chunks = (physical_rows + 4095) // 4096
    descriptors = 3 * chunks
    descriptor_elements = descriptors + chunks
    pivots = 2 * chunks
    pivot_elements = pivots + 1
    work_items = chunks + 17
    return {
        "bytecode_fused": True,
        "bytecode_physical_rows": physical_rows,
        "bytecode_descriptor_elements": descriptor_elements,
        "bytecode_descriptor_bytes": 8 * descriptor_elements,
        "bytecode_descriptor_storage_id": 801,
        "bytecode_pivot_elements": pivot_elements,
        "bytecode_pivot_bytes": 2 * pivot_elements,
        "bytecode_pivot_storage_id": 802,
        "bytecode_chunk_offset_elements": 2 * chunks,
        "bytecode_chunk_offset_bytes": 8 * chunks,
        "bytecode_chunk_offset_storage_id": 803,
        "bytecode_work_items": work_items,
        "bytecode_work_item_bytes": 8 * work_items,
        "bytecode_work_item_storage_id": 804,
        "bytecode_address_offset_elements": (1 << 13) + 1,
        "bytecode_address_offset_bytes": 4 * ((1 << 13) + 1),
        "bytecode_address_offset_storage_id": 805,
        "bytecode_occurrence_bytes": 2 * physical_rows,
        "bytecode_occurrence_storage_id": 806,
        "bytecode_magnitude_bytes": 8 * physical_rows,
        "bytecode_magnitude_storage_id": 807,
        "bytecode_max_descriptors_per_chunk": 504,
        "bytecode_max_admitted_descriptors_per_chunk": 512,
        "bytecode_max_pivots_per_chunk": 11,
        "bytecode_max_admitted_pivots_per_chunk": 15,
        "bytecode_dynamic_threadgroup_bytes": 4390,
        "bytecode_threadgroup_memory_limit_bytes": 32768,
        "shared_source_row_scans": 1,
        "additional_source_row_scans": 0,
        "member_upload_bytes": 0,
        "command_buffers": 1,
        "waits": 1,
        "encoders": 1,
        "dispatches": 1,
    }


def fused_bytecode_topology_args(log_n: int, enabled: bool) -> dict[str, object]:
    rows = 1 << log_n
    physical_rows = rows - 123
    source = bytecode_address_stage1_source(log_n)
    common = {
        "enabled": "true" if enabled else "false",
        "physical_rows": str(physical_rows),
        "chunk_rows": "4096",
        "source_generation": str(source["source_generation"]),
        "source_completion_serial": str(source["completion_serial"]),
        "source_rows_storage_id": str(source["row_allocation_identity"]),
        "source_claim_storage_id": str(source["claim_allocation_identity"]),
        "shared_source_row_scans": "1",
        "additional_source_row_scans": "0",
        "extra_source_scans": "0",
        "source_windows": str(rows),
        "member_upload_bytes": "0",
    }
    if not enabled:
        return {
            **common,
            **{
                field: "0"
                for field in (
                    "chunks",
                    "descriptors",
                    "descriptor_elements",
                    "descriptor_bytes",
                    "descriptor_storage_id",
                    "pivots",
                    "pivot_elements",
                    "pivot_bytes",
                    "pivot_storage_id",
                    "chunk_offset_elements",
                    "chunk_offset_bytes",
                    "chunk_offset_storage_id",
                    "work_items",
                    "work_item_elements",
                    "work_item_bytes",
                    "work_item_storage_id",
                    "address_offset_elements",
                    "address_offset_bytes",
                    "address_offset_storage_id",
                    "max_descriptors_per_chunk",
                    "max_pivots_per_chunk",
                    "first_push_pc",
                    "topology_completion_serial",
                    "covered_rows",
                )
            },
            "complete_overwrite": "false",
        }
    scatter = fused_bytecode_stage1_scatter(log_n)
    chunks = (physical_rows + 4095) // 4096
    descriptors = 3 * chunks
    pivots = 2 * chunks
    work_items = chunks + 17
    return {
        **common,
        "chunks": str(chunks),
        "descriptors": str(descriptors),
        "descriptor_elements": str(scatter["bytecode_descriptor_elements"]),
        "descriptor_bytes": str(scatter["bytecode_descriptor_bytes"]),
        "descriptor_storage_id": str(scatter["bytecode_descriptor_storage_id"]),
        "pivots": str(pivots),
        "pivot_elements": str(scatter["bytecode_pivot_elements"]),
        "pivot_bytes": str(scatter["bytecode_pivot_bytes"]),
        "pivot_storage_id": str(scatter["bytecode_pivot_storage_id"]),
        "chunk_offset_elements": str(scatter["bytecode_chunk_offset_elements"]),
        "chunk_offset_bytes": str(scatter["bytecode_chunk_offset_bytes"]),
        "chunk_offset_storage_id": str(
            scatter["bytecode_chunk_offset_storage_id"]
        ),
        "work_items": str(work_items),
        "work_item_elements": str(work_items),
        "work_item_bytes": str(scatter["bytecode_work_item_bytes"]),
        "work_item_storage_id": str(scatter["bytecode_work_item_storage_id"]),
        "address_offset_elements": str(scatter["bytecode_address_offset_elements"]),
        "address_offset_bytes": str(scatter["bytecode_address_offset_bytes"]),
        "address_offset_storage_id": str(
            scatter["bytecode_address_offset_storage_id"]
        ),
        "max_descriptors_per_chunk": "504",
        "max_pivots_per_chunk": "11",
        "first_push_pc": "4096",
        "topology_completion_serial": "21",
        "complete_overwrite": "true",
        "covered_rows": str(physical_rows),
    }


def complete_fused_bytecode_address_trace(
    log_n: int, *, control: bool = False
) -> list[dict[str, object]]:
    def event(
        name: str,
        timestamp: float,
        duration: float,
        args: dict[str, object],
    ) -> dict[str, object]:
        return {
            "name": name,
            "ph": "X",
            "pid": 1,
            "tid": 0,
            "ts": timestamp,
            "dur": duration,
            "args": args,
        }

    if control:
        events = complete_bytecode_address_trace(log_n, "optimized")
        events.extend(
            [
                event(
                    "MetalBytecodeReadRafAddress::fused_topology_prepare",
                    10.0,
                    5.0,
                    fused_bytecode_topology_args(log_n, False),
                ),
                event(
                    "MetalBytecodeReadRafAddress::route",
                    210.0,
                    480.0,
                    {
                        "cycles": str(1 << log_n),
                        "requested": "cpu",
                        "realized_route": "cpu",
                        "fallback_reason": "configured_cpu",
                    },
                ),
            ]
        )
        return events

    events = complete_bytecode_address_trace(log_n, "metal")
    legacy_publish = next(
        trace_event
        for trace_event in events
        if trace_event["name"] == "MetalBytecodeReadRafAddress::carrier_publish"
    )
    events.remove(legacy_publish)
    route = next(
        trace_event
        for trace_event in events
        if trace_event["name"] == "MetalBytecodeReadRafAddress::route"
    )
    route["args"]["realized_route"] = "address_major_fused_stage1_grouped_v1"
    complete = next(
        trace_event
        for trace_event in events
        if trace_event["name"]
        == "MetalBytecodeReadRafAddress::address_major_complete"
    )["args"]
    scatter = fused_bytecode_stage1_scatter(log_n)
    topology = fused_bytecode_topology_args(log_n, True)
    physical_rows = int(scatter["bytecode_physical_rows"])
    work_items = int(scatter["bytecode_work_items"])
    topology_bytes = (
        int(scatter["bytecode_descriptor_bytes"])
        + int(scatter["bytecode_pivot_bytes"])
        + int(scatter["bytecode_chunk_offset_bytes"])
    )
    carrier_resident_bytes = (
        10 * physical_rows
        + int(scatter["bytecode_work_item_bytes"])
        + int(scatter["bytecode_address_offset_bytes"])
    )
    source = bytecode_address_stage1_source(log_n)
    publish = {
        "route": "address_major_fused_stage1_grouped_v1",
        "cycles": str(1 << log_n),
        "physical_rows": str(physical_rows),
        "work_items": str(work_items),
        "source_generation": str(source["source_generation"]),
        "source_completion_serial": str(source["completion_serial"]),
        "source_rows_storage_id": str(source["row_allocation_identity"]),
        "source_claim_storage_id": str(source["claim_allocation_identity"]),
        "source_device_registry_id": str(source["device_registry_id"]),
        "source_windows": str(1 << log_n),
        "carrier_completion_serial": "22",
        "carrier_occurrence_storage_id": str(
            scatter["bytecode_occurrence_storage_id"]
        ),
        "carrier_occurrence_bytes": str(scatter["bytecode_occurrence_bytes"]),
        "carrier_magnitude_storage_id": str(
            scatter["bytecode_magnitude_storage_id"]
        ),
        "carrier_magnitude_bytes": str(scatter["bytecode_magnitude_bytes"]),
        "carrier_work_item_storage_id": str(
            scatter["bytecode_work_item_storage_id"]
        ),
        "carrier_work_item_bytes": str(scatter["bytecode_work_item_bytes"]),
        "carrier_address_offset_storage_id": str(
            scatter["bytecode_address_offset_storage_id"]
        ),
        "carrier_address_offset_bytes": str(
            scatter["bytecode_address_offset_bytes"]
        ),
        "bytecode_descriptor_storage_id": str(
            scatter["bytecode_descriptor_storage_id"]
        ),
        "bytecode_descriptor_bytes": str(scatter["bytecode_descriptor_bytes"]),
        "bytecode_pivot_storage_id": str(scatter["bytecode_pivot_storage_id"]),
        "bytecode_pivot_bytes": str(scatter["bytecode_pivot_bytes"]),
        "bytecode_chunk_offset_storage_id": str(
            scatter["bytecode_chunk_offset_storage_id"]
        ),
        "bytecode_chunk_offset_bytes": str(
            scatter["bytecode_chunk_offset_bytes"]
        ),
        "carrier_resident_bytes": str(carrier_resident_bytes),
        "carrier_buffers": "4",
        "scatter_output_allocations": "2",
        "producer_persistent_write_bytes": str(10 * physical_rows),
        "producer_logical_movement_bytes": str(10 * physical_rows + topology_bytes),
        "producer_topology_read_bytes": str(topology_bytes),
        "complete_overwrite": "true",
        "covered_rows": str(physical_rows),
        "shared_source_row_scans": "1",
        "additional_source_row_scans": "0",
        "member_upload_bytes": "0",
        "command_buffers": "1",
        "waits": "1",
        "encoders": "1",
        "dispatches": "1",
        "released": "false",
    }
    complete.update(
        {
            "realized_route": "address_major_fused_stage1_grouped_v1",
            "carrier_completion_serial": "22",
            "carrier_occurrence_storage_id": str(
                scatter["bytecode_occurrence_storage_id"]
            ),
            "carrier_occurrence_bytes": str(scatter["bytecode_occurrence_bytes"]),
            "carrier_magnitude_storage_id": str(
                scatter["bytecode_magnitude_storage_id"]
            ),
            "carrier_magnitude_bytes": str(scatter["bytecode_magnitude_bytes"]),
            "carrier_work_item_storage_id": str(
                scatter["bytecode_work_item_storage_id"]
            ),
            "carrier_work_item_bytes": str(scatter["bytecode_work_item_bytes"]),
            "carrier_address_offset_storage_id": str(
                scatter["bytecode_address_offset_storage_id"]
            ),
            "carrier_address_offset_bytes": str(
                scatter["bytecode_address_offset_bytes"]
            ),
            "carrier_resident_bytes": str(carrier_resident_bytes),
            "bytecode_descriptor_storage_id": str(
                scatter["bytecode_descriptor_storage_id"]
            ),
            "bytecode_descriptor_bytes": str(scatter["bytecode_descriptor_bytes"]),
            "bytecode_pivot_storage_id": str(
                scatter["bytecode_pivot_storage_id"]
            ),
            "bytecode_pivot_bytes": str(scatter["bytecode_pivot_bytes"]),
            "bytecode_chunk_offset_storage_id": str(
                scatter["bytecode_chunk_offset_storage_id"]
            ),
            "bytecode_chunk_offset_bytes": str(
                scatter["bytecode_chunk_offset_bytes"]
            ),
            "topology_publication_bytes": str(
                topology_bytes
                + int(scatter["bytecode_work_item_bytes"])
                + int(scatter["bytecode_address_offset_bytes"])
            ),
            "producer_persistent_write_bytes": str(10 * physical_rows),
            "producer_logical_movement_bytes": str(
                10 * physical_rows + topology_bytes
            ),
            "producer_topology_read_bytes": str(topology_bytes),
        }
    )
    events.extend(
        [
            event(
                "MetalBytecodeReadRafAddress::fused_topology_prepare",
                10.0,
                20.0,
                topology,
            ),
            event("InstructionReadRaf::prepare", 105.0, 80.0, {}),
            event(
                "MetalInstructionReadRaf::stage1_grouped_scatter",
                115.0,
                30.0,
                {},
            ),
            event(
                "MetalBytecodeReadRafAddress::fused_carrier_publish",
                150.0,
                1.0,
                publish,
            ),
        ]
    )
    return events


def complete_instruction_input_trace(
    log_n: int,
    backend: str,
    cutoff_log2: int = 16,
    borrow_outer_residual: bool = False,
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
        auxiliary_bytes = (
            metal_piop_eval.instruction_input_sequence_auxiliary_storage_bytes(log_n)
        )
        resident_row_bytes = 160 * (1 << log_n)
        storage_buffer_ids = (
            [203, 203, 301, 302, 303, 304]
            if borrow_outer_residual
            else list(range(301, 307))
        )
        owned_bytes = auxiliary_bytes if borrow_outer_residual else sequence_bytes
        reused_bytes = sequence_bytes - owned_bytes
        initialized_buffers = 4 if borrow_outer_residual else 6
        initialized_bytes = 64 if borrow_outer_residual else 96
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
                        "explicit_rows": str(1 << log_n),
                    },
                ),
                event(
                    "MetalInstructionInput::storage_prepare",
                    100.0,
                    400.0,
                    {
                        "trace_elements": str(1 << log_n),
                        "cutoff_elements": str(1 << cutoff_log2),
                        "dense_storage_mode": (
                            "OuterResidual" if borrow_outer_residual else "Owned"
                        ),
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
                        "device_buffers": str(initialized_buffers),
                        "planned_device_bytes": str(sequence_bytes),
                        "owned_device_bytes": str(owned_bytes),
                        "reused_device_bytes": str(reused_bytes),
                        "borrowed_outer_residual": str(
                            borrow_outer_residual
                        ).lower(),
                        "current_device_bytes": str(resident_row_bytes),
                        "recommended_device_bytes": str(
                            resident_row_bytes + owned_bytes
                        ),
                    },
                ),
                event(
                    "MetalInstructionInput::storage_initialize",
                    220.0,
                    100.0,
                    {
                        "mode": "minimal",
                        "device_buffers": str(initialized_buffers),
                        "bytes": str(initialized_bytes),
                        "protocol_dispatches": "0",
                        **{
                            f"buffer_{index}": str(storage_buffer_ids[index])
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
                        "explicit_rows": str(1 << log_n),
                        "compact_row_bytes": "48",
                        "residual_row_bytes": "112",
                        "full_domain_copy_bytes": "0",
                        "full_domain_copy_dispatches": "0",
                        "host_repack_rows": "0",
                    },
                ),
                event(
                    "MetalOuterRemainder::row_release",
                    1_051.0,
                    3.0,
                    {
                        "device_registry_id": "1",
                        "remaining_sequence_storage_bytes": "5242880",
                    },
                ),
                *(
                    [
                        event(
                            "MetalInstructionInput::outer_residual_transfer",
                            1_055.0,
                            2.0,
                            {
                                "resident_rows": str(1 << log_n),
                                "outer_residual_generation": "7",
                                "compact_rows_storage_id": "202",
                                "residual_rows_storage_id": "203",
                                "device_registry_id": "1",
                                "outer_sequence_owned_bytes": "5242880",
                                "outer_sequence_consumed": "true",
                                "compact_rows_transferred": "true",
                                "residual_rows_transferred": "true",
                            },
                        )
                    ]
                    if borrow_outer_residual
                    else []
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
                            f"storage_buffer_{index}": str(storage_buffer_ids[index])
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
                        "storage_initialization_bytes": str(initialized_bytes),
                        "native_primer": "async",
                        "dense_a_offset_bytes": "0",
                        "dense_a_length_bytes": str(64 * (1 << log_n)),
                        "dense_b_offset_bytes": str(
                            64 * (1 << log_n) if borrow_outer_residual else 0
                        ),
                        "dense_b_length_bytes": str(32 * (1 << log_n)),
                        **{
                            f"storage_buffer_{index}": str(storage_buffer_ids[index])
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
                            f"storage_buffer_{index}": str(storage_buffer_ids[index])
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
                            f"storage_buffer_{index}": str(storage_buffer_ids[index])
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


def complete_instruction_read_raf_trace(
    log_n: int,
    backend: str,
    scatter_threads: int = 512,
    fused_bytecode: bool = False,
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

    rounds = 128 + log_n
    round_starts = [1_400.0 + 20.0 * index for index in range(rounds)]
    finish_start = 1_400.0 + 20.0 * rounds
    events = [
        event("jolt_prover::backend_witness_prepare", 0.0, 900.0),
        event("jolt_prover::piop", 1_000.0, 5_000.0),
        event("InstructionReadRaf::prepare", 1_100.0, 250.0),
        *(
            event("InstructionReadRaf::prove_round", timestamp, 10.0)
            for timestamp in round_starts
        ),
        event("InstructionReadRaf::finish_rounds", finish_start, 10.0),
        event("InstructionReadRaf::output_claims", finish_start + 20.0, 10.0),
    ]
    if backend == "optimized":
        return events

    rows = 1 << log_n
    chunks = rows // 4096
    e_out = 1 << (log_n // 2)
    e_in = rows // e_out
    source = {
        "rows": str(rows),
        "row_bytes": str(40 * rows),
        "claim_bytes": str(rows),
        "resident_device_bytes": str(41 * rows),
        "count_chunks": str(chunks),
        "count_bytes": str(328 * chunks),
        "host_row_write_bytes": str(40 * rows),
        "host_claim_write_bytes": str(rows),
        "host_count_update_bytes": str(4 * rows),
        "row_allocation_identity": "501",
        "claim_allocation_identity": "502",
        "count_allocation_identity": "503",
        "device_registry_id": "17",
        "source_generation": "7",
        "completion_serial": "9",
        "count_order": "table_major_then_none_v1",
        "publication_kind": "host_fill_v1",
        "complete_overwrite": "true",
        "source_windows": str(rows),
        "member_upload_bytes": "0",
        "projection_dispatches": "0",
    }
    compact = {
        "source_kind": "owned_random_access",
        "witness_row_extractions": str(rows),
        "residual_rows_written": str(rows),
        "compact_rows_written": str(rows),
        "compact_row_bytes": "48",
        "residual_row_bytes": "112",
        "compact_allocations": "1",
        "residual_allocations": "1",
        "full_row_allocations": "0",
        "full_domain_copy_bytes": "0",
        "full_domain_copy_dispatches": "0",
        "host_repack_rows": "0",
        "resident_rows": str(rows),
        "explicit_rows": str(rows - 1),
        "compact_rows_storage_id": "601",
        "residual_rows_storage_id": "602",
    }
    witness = {
        "cycles": str(rows),
        "source": "stage1_single_projection",
        "admitted": "true",
        "fallback_reason": "none",
        "native_register_contract_bytes": str(24 * rows),
        "owner_generation": "19",
        "shift_late_copy_dispatches": "0",
        "shift_resident_bytes": str(16 * rows),
        "shift_row_extractions": str(rows),
    }
    additional = 37 * rows + 328 * chunks + 332 + 16 * (e_in + e_out) + 4 + 88
    scatter = {
        "rows": str(rows),
        "preparation_wall_ns": "10000",
        "command_wall_ns": "60000",
        "gpu_active_ns": "50000",
        "status_readback_bytes": "4",
        "packed_rows_bytes": str(rows),
        "lookups_bytes": str(16 * rows),
        "inverse_bytes": str(4 * rows),
        "weights_bytes": str(16 * rows),
        "packed_rows_identity": "701",
        "lookups_identity": "702",
        "inverse_identity": "703",
        "weights_identity": "704",
        "source_generation": "7",
        "source_completion_serial": "9",
        "source_row_allocation_identity": "501",
        "source_claim_allocation_identity": "502",
        "source_count_allocation_identity": "503",
        "source_count_chunks": str(chunks),
        "source_count_bytes": str(328 * chunks),
        "source_count_order": "table_major_then_none_v1",
        "source_device_registry_id": "17",
        "scatter_completion_serial": "11",
        "e_in_length": str(e_in),
        "e_out_length": str(e_out),
        "command_buffers": "1",
        "encoders": "1",
        "waits": "1",
        "dispatches": "1",
        "threadgroups": str(chunks),
        "threads_per_threadgroup": str(scatter_threads),
        "dynamic_threadgroup_bytes": "328",
        "static_threadgroup_bytes": "0",
        "source_copy_bytes": "0",
        "full_plane_readback_bytes": "0",
        "complete_overwrite": "true",
        "additional_allocation_bytes": str(additional),
    }
    if fused_bytecode:
        physical_rows = rows - 1
        bytecode_chunks = (physical_rows + 4095) // 4096
        descriptor_elements = 4 * bytecode_chunks
        pivot_elements = 2 * bytecode_chunks + 1
        work_items = bytecode_chunks + 17
        scatter["additional_allocation_bytes"] = str(additional + 10 * physical_rows)
        scatter["dynamic_threadgroup_bytes"] = "4390"
        scatter.update(
            {
                "bytecode_fused": "true",
                "bytecode_physical_rows": str(physical_rows),
                "bytecode_descriptor_elements": str(descriptor_elements),
                "bytecode_descriptor_bytes": str(8 * descriptor_elements),
                "bytecode_descriptor_storage_id": "801",
                "bytecode_pivot_elements": str(pivot_elements),
                "bytecode_pivot_bytes": str(2 * pivot_elements),
                "bytecode_pivot_storage_id": "802",
                "bytecode_chunk_offset_elements": str(2 * bytecode_chunks),
                "bytecode_chunk_offset_bytes": str(8 * bytecode_chunks),
                "bytecode_chunk_offset_storage_id": "803",
                "bytecode_work_items": str(work_items),
                "bytecode_work_item_bytes": str(8 * work_items),
                "bytecode_work_item_storage_id": "804",
                "bytecode_address_offset_elements": str((1 << 13) + 1),
                "bytecode_address_offset_bytes": str(4 * ((1 << 13) + 1)),
                "bytecode_address_offset_storage_id": "805",
                "bytecode_occurrence_bytes": str(2 * physical_rows),
                "bytecode_occurrence_storage_id": "806",
                "bytecode_magnitude_bytes": str(8 * physical_rows),
                "bytecode_magnitude_storage_id": "807",
                "bytecode_max_descriptors_per_chunk": "504",
                "bytecode_max_admitted_descriptors_per_chunk": "512",
                "bytecode_max_pivots_per_chunk": "11",
                "bytecode_max_admitted_pivots_per_chunk": "15",
                "bytecode_dynamic_threadgroup_bytes": "4390",
                "bytecode_threadgroup_memory_limit_bytes": "32768",
                "shared_source_row_scans": "1",
                "additional_source_row_scans": "0",
                "member_upload_bytes": "0",
            }
        )
    events.extend(
        [
            event("MetalSpartanDense::witness_prepare", 50.0, 700.0, witness),
            event("MetalInstructionInput::compact_rows_prepare", 100.0, 500.0, compact),
            event(
                "MetalInstructionReadRaf::stage1_source_publish",
                800.0,
                1.0,
                source,
            ),
            event(
                "MetalInstructionReadRaf::stage1_grouped_scatter",
                1_150.0,
                1.0,
                scatter,
            ),
            event(
                "MetalInstructionReadRaf::stage1_grouped_sequence_prepare",
                1_160.0,
                1.0,
            ),
            *(
                event("MetalInstructionReadRaf::address_round", timestamp + 1.0, 1.0)
                for timestamp in round_starts[:129]
            ),
            event(
                "MetalInstructionReadRaf::resident_first_message",
                round_starts[128] + 3.0,
                1.0,
            ),
            event(
                "MetalInstructionReadRaf::resident_handoff",
                round_starts[129] + 1.0,
                1.0,
            ),
            *(
                event("MetalInstructionReadRaf::resident_round", timestamp + 1.0, 1.0)
                for timestamp in round_starts[130 : 130 + log_n - 17]
            ),
            event(
                "MetalInstructionReadRaf::readback",
                round_starts[130 + log_n - 17] + 1.0,
                1.0,
            ),
        ]
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


def complete_registers_claim_trace(
    log_n: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
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
    prefix_vars = (log_n + 1) // 2
    prefix_elements = 1 << prefix_vars
    suffix_elements = rows // prefix_elements
    outer_carrier = {
        "source_generation": 17,
        "source_compact_storage_id": 101,
        "rd_storage_id": 103,
        "device_registry_id": 107,
    }
    round_starts = [200.0 + 100.0 * index for index in range(log_n)]
    midpoint_start = round_starts[prefix_vars]
    events = [
        event("jolt_prover::piop", 0.0, 10_000.0),
        event("RegistersClaimReduction::prepare", 100.0, 50.0),
        *[
            event("RegistersClaimReduction::prove_round", timestamp, 80.0)
            for timestamp in round_starts
        ],
        event(
            "RegistersClaimReduction::finish_rounds",
            200.0 + 100.0 * log_n,
            50.0,
        ),
        event(
            "RegistersClaimReduction::output_claims",
            300.0 + 100.0 * log_n,
            40.0,
        ),
        event(
            "MetalRegistersClaimReduction::route",
            105.0,
            5.0,
            {
                "cycles": rows,
                "requested": "outer_carrier_alias_hybrid",
                "stage1_carry_present": True,
                "alias_receiver_present": True,
                "realized_route": "outer_carrier_alias_hybrid",
                "fallback_reason": "none",
            },
        ),
        event(
            "MetalRegistersClaimReduction::prepare",
            112.0,
            20.0,
            {
                "cycles": rows,
                "requested": "outer_carrier_alias_hybrid",
                "realized_route": "outer_carrier_alias_hybrid",
                "fallback_reason": "none",
                "resident_bytes": 8 * rows,
                "source_allocations": 0,
                "source_upload_bytes": 0,
                "source_host_write_bytes": 0,
                "source_generation": outer_carrier["source_generation"],
                "source_compact_storage_id": outer_carrier[
                    "source_compact_storage_id"
                ],
                "source_rd_storage_id": outer_carrier["rd_storage_id"],
                "alias_generation": 19,
            },
        ),
        event(
            "MetalInstructionInput::registers_claim_alias_publish",
            midpoint_start + 5.0,
            5.0,
            {
                "rows": rows,
                "source_compact_storage_id": outer_carrier[
                    "source_compact_storage_id"
                ],
                "alias_generation": 19,
                "prefix_challenges": prefix_vars,
                "table_0": 1,
                "table_1": 5,
                "host_table_copies": 2,
                "snapshot_host_bytes": 32 * suffix_elements,
                "publishes": 1,
            },
        ),
        event(
            "MetalRegistersClaimReduction::midpoint_projection",
            midpoint_start + 20.0,
            20.0,
            {
                "source": "outer_carrier_alias",
                "round": prefix_vars,
                "rows": rows,
                "source_generation": outer_carrier["source_generation"],
                "device_registry_id": outer_carrier["device_registry_id"],
                "source_compact_storage_id": outer_carrier[
                    "source_compact_storage_id"
                ],
                "source_rd_storage_id": outer_carrier["rd_storage_id"],
                "alias_generation": 19,
                "rd_source_bytes": 8 * rows,
                "eq_upload_bytes": 16 * prefix_elements,
                "readback_bytes": 16 * suffix_elements,
                "device_allocations": 2,
                "dispatches": 1,
                "command_buffers": 1,
                "waits": 1,
                "alias_takes": 1,
                "useful_half_width_terms": rows,
                "gpu_active_ns": 10_000,
                "resident_wall_ns": 20_000,
            },
        ),
        event("sumcheck_round", midpoint_start, 80.0),
    ]
    return events, outer_carrier


def complete_ram_sparse_trace(log_n: int, backend: str) -> list[dict[str, object]]:
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
        event("jolt_prover::backend_witness_prepare", 0.0, 80.0),
        event("jolt_prover::piop", 100.0, 10_000.0),
    ]

    def member(kernel: str, rounds: int, start: float) -> tuple[float, list[float]]:
        events.append(event(f"{kernel}::prepare", start, 80.0))
        round_starts = [start + 100.0 + 30.0 * index for index in range(rounds)]
        for round_start in round_starts:
            events.extend(
                [
                    event("sumcheck_round", round_start - 2.0, 22.0),
                    event(f"{kernel}::prove_round", round_start, 6.0),
                    event("sumcheck_host_fiat_shamir", round_start + 10.0, 2.0),
                ]
            )
        finish = round_starts[-1] + 25.0
        events.append(event(f"{kernel}::finish_rounds", finish, 5.0))
        output = finish + 10.0
        events.append(event(f"{kernel}::output_claims", output, 8.0))
        return output + 20.0, [finish, output]

    rw_start = 200.0
    rw_end, rw_tail = member("RamReadWriteChecking", log_n + 13, rw_start)
    raf_start = rw_end + 40.0
    raf_end, _ = member("RamRafEvaluation", 13, raf_start)
    val_start = raf_end + 40.0
    val_end, _ = member("RamValCheck", log_n, val_start)
    ra_claim_start = val_end + 40.0
    ra_claim_end, _ = member("RamRaClaimReduction", log_n, ra_claim_start)
    hamming_start = ra_claim_end + 40.0
    hamming_end, hamming_tail = member(
        "RamHammingBooleanity", log_n, hamming_start
    )
    ra_virtualization_start = hamming_end + 40.0
    member("RamRaVirtualization", log_n, ra_virtualization_start)
    if backend == "optimized":
        return events

    generation = 7
    fingerprint = 9
    address_domain = 1 << 13
    access_records = 4
    increment_records = 2
    record_bytes = 24 * (access_records + increment_records)
    final_memory_bytes = 8 * address_domain
    topology_bytes = 1_000
    owner_bytes = record_bytes + final_memory_bytes + topology_bytes
    owner = {
        "enabled": "true",
        "schema_version": "3",
        "source_kind": "ram_access_tape_v1",
        "source_generation": str(generation),
        "source_fingerprint": str(fingerprint),
        "log_t": str(log_n),
        "log_k": "13",
        "cycles": str(1 << log_n),
        "address_domain": str(address_domain),
        "access_records": str(access_records),
        "increment_records": str(increment_records),
        "hamming_exact": "true",
        "retained_records": str(access_records),
        "final_memory_elements": str(address_domain),
        "record_bytes": str(record_bytes),
        "final_memory_bytes": str(final_memory_bytes),
        "read_write_topology_nodes": "20",
        "block_topology_nodes": "11",
        "topology_bytes": str(topology_bytes),
        "owner_bytes": str(owner_bytes),
        "source_rows": str(1 << log_n),
        "source_collection_performed": "true",
        "shared_source_row_scans": "1",
        "additional_source_row_scans": "0",
        "member_upload_bytes": "0",
        "complete_publication": "true",
    }
    address_plane_bytes = 4 * (1 << log_n)
    witness_prepare = {
        "schema_version": "3",
        "requested": "host_sparse_v1",
        "selected": "host_sparse_v1",
        "fallback_reason": "none",
        "log_t": str(log_n),
        "log_k": "13",
        "cycles": str(1 << log_n),
        "address_domain": str(address_domain),
        "source_generation": str(generation),
        "source_fingerprint": str(fingerprint),
        "source_collection_performed": "true",
        "witness_source_scans": "1",
        "additional_witness_source_scans": "0",
        "address_validation_passes": "3",
        "address_rows": str(1 << log_n),
        "address_plane_storage_id": "101",
        "address_plane_device_registry_id": "202",
        "address_plane_bytes": str(address_plane_bytes),
        "address_plane_upload_bytes": str(address_plane_bytes),
        "address_plane_allocations": "1",
        "owner_published": "true",
        "address_plane_published": "true",
        "complete_publication": "true",
    }
    events.extend(
        [
            event(
                "MetalRamCycleFamily::witness_prepare",
                10.0,
                60.0,
                witness_prepare,
            ),
            event("MetalRamCycleFamily::owner_prepare", 20.0, 20.0, owner),
            event(
                "MetalRamRafEvaluation::submit",
                rw_start + 10.0,
                4.0,
                {
                    "cycles": str(1 << log_n),
                    "resident_address_bytes": str(address_plane_bytes),
                    "address_storage_id": "101",
                },
            ),
            event(
                "MetalRamRafEvaluation::join",
                raf_start + 105.0,
                1.0,
            ),
            event(
                "MetalRamReadWrite::sparse_prepare",
                rw_start + 5.0,
                35.0,
                {
                    "selected": "host_sparse_v1",
                    "source_generation": str(generation),
                    "source_fingerprint": str(fingerprint),
                    "log_t": str(log_n),
                    "log_k": "13",
                    "rounds": str(log_n + 13),
                    "access_records": str(access_records),
                    "increment_records": str(increment_records),
                    "owner_bytes": str(owner_bytes),
                    "cycle_cutoff": "0",
                    "additional_source_row_scans": "0",
                    "member_upload_bytes": "0",
                    "gpu_dispatches": "0",
                    "command_buffers": "0",
                    "waits": "0",
                    "readbacks": "0",
                },
            ),
            event(
                "MetalRamReadWrite::route",
                rw_start + 15.0,
                2.0,
                {
                    "cycles": str(1 << log_n),
                    "log_t": str(log_n),
                    "log_k": "13",
                    "requested": "host_sparse_v1",
                    "selected": "host_sparse_v1",
                    "fallback_reason": "none",
                    "source_generation": str(generation),
                    "source_fingerprint": str(fingerprint),
                },
            ),
            event(
                "MetalRamReadWrite::sparse_derived_validate",
                rw_tail[0] + 6.0,
                1.0,
                {
                    "source_generation": str(generation),
                    "source_fingerprint": str(fingerprint),
                    "derived_claim_valid": "true",
                },
            ),
            event(
                "MetalRamReadWrite::sparse_complete",
                rw_tail[1] + 1.0,
                1.0,
                {
                    "selected": "host_sparse_v1",
                    "source_generation": str(generation),
                    "source_fingerprint": str(fingerprint),
                    "output_claims_valid": "true",
                },
            ),
        ]
    )
    common_route = {
        "cycles": str(1 << log_n),
        "log_t": str(log_n),
        "log_k": "13",
        "requested": "host_sparse_v1",
        "selected": "host_sparse_v1",
        "fallback_reason": "none",
        "source_generation": str(generation),
        "source_fingerprint": str(fingerprint),
        "access_records": str(access_records),
        "increment_records": str(increment_records),
        "additional_source_row_scans": "0",
        "member_upload_bytes": "0",
        "complete_sequence": "true",
    }
    product_route = {
        **common_route,
        "estimated_products": "100",
        "product_cap": "1000000",
    }
    events.extend(
        [
            event(
                "MetalRamValCheck::route",
                val_start + 10.0,
                10.0,
                dict(common_route),
            ),
            event(
                "MetalRamRaClaimReduction::route",
                ra_claim_start + 10.0,
                10.0,
                dict(product_route),
            ),
            event(
                "MetalRamRaVirtualization::route",
                ra_virtualization_start + 10.0,
                40.0,
                dict(product_route),
            ),
        ]
    )
    parent_nodes = 7
    middle_nodes = 6
    estimated_products = 7 * parent_nodes + middle_nodes + 10 * log_n
    hamming_prepare = {
        "selected": "host_sparse_v1",
        "fallback_reason": "none",
        "source_generation": str(generation),
        "source_fingerprint": str(fingerprint),
        "log_t": str(log_n),
        "access_leaves": str(access_records),
        "parent_nodes": str(parent_nodes),
        "middle_nodes": str(middle_nodes),
        "rounds": str(log_n),
        "estimated_products": str(estimated_products),
        "product_cap": "1000000",
        "topology_builds": "1",
        "topology_bytes": "192",
        "member_heap_bytes_including_topology": "300",
        "non_topology_heap_bytes": "108",
        "additional_source_row_scans": "0",
        "dense_h_elements": "0",
        "member_upload_bytes": "0",
        "gpu_dispatches": "0",
        "command_buffers": "0",
        "waits": "0",
        "readbacks": "0",
        "complete_plan": "true",
    }
    hamming_complete = {
        key: value
        for key, value in hamming_prepare.items()
        if key
        in {
            "selected",
            "source_generation",
            "source_fingerprint",
            "access_leaves",
            "parent_nodes",
            "middle_nodes",
            "estimated_products",
            "topology_bytes",
            "member_heap_bytes_including_topology",
            "non_topology_heap_bytes",
        }
    }
    hamming_complete.update(
        {"terminal_ready": "true", "output_claim_emitted": "true"}
    )
    events.extend(
        [
            event(
                "MetalRamHammingBooleanity::sparse_prepare",
                hamming_start + 5.0,
                50.0,
                hamming_prepare,
            ),
            event(
                "MetalRamHammingBooleanity::route",
                hamming_start + 20.0,
                2.0,
                {
                    "cycles": str(1 << log_n),
                    "requested": "host_sparse_v1",
                    "selected": "host_sparse_v1",
                    "fallback_reason": "none",
                    "source_generation": str(generation),
                    "source_fingerprint": str(fingerprint),
                },
            ),
            event(
                "MetalRamCycleFamily::terminal_take",
                ra_virtualization_start + 20.0,
                2.0,
                {
                    "source_generation": str(generation),
                    "source_fingerprint": str(fingerprint),
                    "selected": "host_sparse_v1",
                    "fallback_reason": "none",
                    "session_owner_removed": "true",
                    "columns_removed": "true",
                },
            ),
            event(
                "MetalRamHammingBooleanity::sparse_derived_validate",
                hamming_tail[0] + 6.0,
                1.0,
                {
                    "source_generation": str(generation),
                    "source_fingerprint": str(fingerprint),
                    "derived_claim_valid": "true",
                },
            ),
            event(
                "MetalRamHammingBooleanity::sparse_complete",
                hamming_tail[1] + 1.0,
                1.0,
                hamming_complete,
            ),
        ]
    )
    return events


class MetalPiopEvalTests(unittest.TestCase):
    def test_ram_cycle_family_is_the_primary_local_kernel(self) -> None:
        self.assertEqual(metal_piop_eval.SCHEMA_VERSION, 17)
        self.assertEqual(
            metal_piop_eval.RAM_CYCLE_FAMILY_CHARGE_MODEL,
            "six_raw_members_plus_witness_prepare_once_v1",
        )
        self.assertEqual(
            metal_piop_eval.LOCAL_KERNELS["RamCycleFamily"]["metric"],
            "ram_cycle_family_speedup",
        )

    def test_ram_sparse_owner_lifecycle_and_standalone_charges_are_exact(self) -> None:
        log_n = 4
        optimized_events = complete_ram_sparse_trace(log_n, "optimized")
        optimized_family = metal_piop_eval.ram_cycle_family_breakdown(
            optimized_events, "optimized", log_n
        )
        optimized_owner = metal_piop_eval.ram_cycle_family_owner_observation(
            optimized_events, "optimized", log_n
        )
        optimized_read_write = metal_piop_eval.ram_read_write_member_breakdown(
            optimized_events, "optimized", log_n, optimized_owner
        )
        optimized_hamming = metal_piop_eval.ram_hamming_member_breakdown(
            optimized_events, "optimized", log_n, optimized_owner
        )
        self.assertIsNone(optimized_owner)
        self.assertEqual(optimized_family["components"]["producer_charge_count"], 0)
        self.assertEqual(
            optimized_family["components"]["charged_member_us"],
            optimized_family["components"]["raw_member_us"],
        )
        self.assertEqual(
            optimized_read_write["components"]["charged_member_us"],
            optimized_read_write["components"]["member_us"],
        )
        self.assertEqual(
            optimized_hamming["components"]["charged_member_us"],
            optimized_hamming["components"]["member_us"],
        )

        metal_events = complete_ram_sparse_trace(log_n, "metal")
        family = metal_piop_eval.ram_cycle_family_breakdown(
            metal_events, "metal", log_n
        )
        owner = metal_piop_eval.ram_cycle_family_owner_observation(
            metal_events, "metal", log_n
        )
        assert owner is not None
        read_write = metal_piop_eval.ram_read_write_member_breakdown(
            metal_events, "metal", log_n, owner
        )
        hamming = metal_piop_eval.ram_hamming_member_breakdown(
            metal_events, "metal", log_n, owner
        )
        self.assertEqual(owner["wall_us"], 20.0)
        self.assertEqual(owner["backend_witness_prepare_interval"], (0.0, 80.0))
        self.assertEqual(owner["piop_interval"], (100.0, 10_100.0))
        self.assertTrue(owner["source_collection_performed"])
        self.assertEqual(family["components"]["witness_prepare_us"], 60.0)
        self.assertEqual(family["components"]["producer_charge_count"], 1)
        self.assertEqual(
            family["components"]["charged_member_us"],
            family["components"]["raw_member_us"] + 60.0,
        )
        self.assertTrue(family["canonical_nonoverlap"])
        self.assertEqual(family["canonical_span_count"], 64)
        self.assertEqual(
            family["members"]["raf_evaluation"]["resource_observation"][
                "submit"
            ]["address_storage_id"],
            family["witness_prepare"]["address_plane_storage_id"],
        )
        self.assertEqual(read_write["components"]["member_us"], 195.0)
        self.assertEqual(read_write["components"]["host_fiat_shamir_total_us"], 34.0)
        self.assertEqual(read_write["components"]["charged_member_us"], 215.0)
        self.assertEqual(hamming["components"]["member_us"], 117.0)
        self.assertEqual(hamming["components"]["host_fiat_shamir_total_us"], 8.0)
        self.assertEqual(hamming["components"]["charged_member_us"], 137.0)
        terminal = hamming["resource_observation"]["terminal_take"]
        self.assertEqual(terminal["source_generation"], owner["source_generation"])
        self.assertEqual(terminal["source_fingerprint"], owner["source_fingerprint"])
        self.assertTrue(terminal["session_owner_removed"])
        self.assertTrue(terminal["columns_removed"])

    def test_ram_sparse_parser_rejects_lifecycle_fallback_and_receipt_mutations(
        self,
    ) -> None:
        log_n = 4

        owner_in_read_write_prepare = complete_ram_sparse_trace(log_n, "metal")
        next(
            event
            for event in owner_in_read_write_prepare
            if event["name"] == "MetalRamCycleFamily::owner_prepare"
        )["ts"] = 205.0
        with self.assertRaisesRegex(ValueError, "backend witness preparation"):
            metal_piop_eval.ram_cycle_family_owner_observation(
                owner_in_read_write_prepare, "metal", log_n
            )

        false_collection = complete_ram_sparse_trace(log_n, "metal")
        next(
            event
            for event in false_collection
            if event["name"] == "MetalRamCycleFamily::owner_prepare"
        )["args"]["source_collection_performed"] = "false"
        with self.assertRaisesRegex(ValueError, "owner receipt"):
            metal_piop_eval.ram_cycle_family_owner_observation(
                false_collection, "metal", log_n
            )

        duplicate_owner = complete_ram_sparse_trace(log_n, "metal")
        owner_event = next(
            event
            for event in duplicate_owner
            if event["name"] == "MetalRamCycleFamily::owner_prepare"
        )
        duplicate_owner.append(copy.deepcopy(owner_event))
        with self.assertRaisesRegex(ValueError, "exactly one cycle-family owner"):
            metal_piop_eval.ram_cycle_family_owner_observation(
                duplicate_owner, "metal", log_n
            )

        fallback = complete_ram_sparse_trace(log_n, "metal")
        next(
            event
            for event in fallback
            if event["name"] == "MetalRamReadWrite::route"
        )["args"]["selected"] = "optimized_cpu"
        fallback_owner = metal_piop_eval.ram_cycle_family_owner_observation(
            fallback, "metal", log_n
        )
        with self.assertRaisesRegex(ValueError, "read-write sparse receipt"):
            metal_piop_eval.ram_read_write_member_breakdown(
                fallback, "metal", log_n, fallback_owner
            )

        malformed_hamming = complete_ram_sparse_trace(log_n, "metal")
        for event in malformed_hamming:
            if event["name"] in {
                "MetalRamHammingBooleanity::sparse_prepare",
                "MetalRamHammingBooleanity::sparse_complete",
            }:
                event["args"]["middle_nodes"] = "5"
                event["args"]["estimated_products"] = "94"
        malformed_owner = metal_piop_eval.ram_cycle_family_owner_observation(
            malformed_hamming, "metal", log_n
        )
        with self.assertRaisesRegex(ValueError, "Hamming sparse receipt"):
            metal_piop_eval.ram_hamming_member_breakdown(
                malformed_hamming, "metal", log_n, malformed_owner
            )

        missing_enclosing = complete_ram_sparse_trace(log_n, "metal")
        missing_enclosing.remove(
            next(
                event
                for event in missing_enclosing
                if event["name"] == "RamRaVirtualization::prepare"
            )
        )
        missing_owner = metal_piop_eval.ram_cycle_family_owner_observation(
            missing_enclosing, "metal", log_n
        )
        with self.assertRaisesRegex(ValueError, "one RA virtualization preparation"):
            metal_piop_eval.ram_hamming_member_breakdown(
                missing_enclosing, "metal", log_n, missing_owner
            )

        mismatched_take = complete_ram_sparse_trace(log_n, "metal")
        next(
            event
            for event in mismatched_take
            if event["name"] == "MetalRamCycleFamily::terminal_take"
        )["args"]["source_generation"] = "8"
        mismatched_owner = metal_piop_eval.ram_cycle_family_owner_observation(
            mismatched_take, "metal", log_n
        )
        with self.assertRaisesRegex(ValueError, "terminal-take receipt"):
            metal_piop_eval.ram_hamming_member_breakdown(
                mismatched_take, "metal", log_n, mismatched_owner
            )

        retained_columns = complete_ram_sparse_trace(log_n, "metal")
        next(
            event
            for event in retained_columns
            if event["name"] == "MetalRamCycleFamily::terminal_take"
        )["args"]["columns_removed"] = "false"
        retained_owner = metal_piop_eval.ram_cycle_family_owner_observation(
            retained_columns, "metal", log_n
        )
        with self.assertRaisesRegex(ValueError, "terminal-take receipt"):
            metal_piop_eval.ram_hamming_member_breakdown(
                retained_columns, "metal", log_n, retained_owner
            )

        duplicate_take = complete_ram_sparse_trace(log_n, "metal")
        terminal = next(
            event
            for event in duplicate_take
            if event["name"] == "MetalRamCycleFamily::terminal_take"
        )
        duplicate_take.append(copy.deepcopy(terminal))
        duplicate_owner = metal_piop_eval.ram_cycle_family_owner_observation(
            duplicate_take, "metal", log_n
        )
        with self.assertRaisesRegex(ValueError, "exactly one cycle-family owner"):
            metal_piop_eval.ram_hamming_member_breakdown(
                duplicate_take, "metal", log_n, duplicate_owner
            )

    def test_ram_cycle_family_rejects_witness_route_and_raf_mutations(self) -> None:
        log_n = 4
        mutations = (
            (
                "MetalRamCycleFamily::witness_prepare",
                "address_plane_allocations",
                "0",
                "witness receipt",
            ),
            (
                "MetalRamCycleFamily::witness_prepare",
                "address_plane_upload_bytes",
                "63",
                "witness receipt",
            ),
            (
                "MetalRamCycleFamily::witness_prepare",
                "address_validation_passes",
                "0",
                "witness receipt",
            ),
            (
                "MetalRamValCheck::route",
                "selected",
                "optimized_cpu",
                "RamValCheck sparse route receipt",
            ),
            (
                "MetalRamRaClaimReduction::route",
                "source_generation",
                "8",
                "RamRaClaimReduction sparse route receipt",
            ),
            (
                "MetalRamRaVirtualization::route",
                "product_cap",
                "999999",
                "RamRaVirtualization sparse route receipt",
            ),
            (
                "MetalRamRafEvaluation::submit",
                "address_storage_id",
                "102",
                "RAF submit receipt",
            ),
        )
        for span, field, value, error in mutations:
            with self.subTest(span=span, field=field):
                events = complete_ram_sparse_trace(log_n, "metal")
                next(event for event in events if event["name"] == span)["args"][
                    field
                ] = value
                with self.assertRaisesRegex(ValueError, error):
                    metal_piop_eval.ram_cycle_family_breakdown(
                        events, "metal", log_n
                    )

        missing_join = complete_ram_sparse_trace(log_n, "metal")
        missing_join.remove(
            next(
                event
                for event in missing_join
                if event["name"] == "MetalRamRafEvaluation::join"
            )
        )
        with self.assertRaisesRegex(ValueError, "RAF route is incomplete"):
            metal_piop_eval.ram_cycle_family_breakdown(
                missing_join, "metal", log_n
            )

        shadow_fallback = complete_ram_sparse_trace(log_n, "metal")
        shadow_fallback.append(
            {
                "name": "MetalRamValCheck::shadow_submit",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 1_000.0,
                "dur": 1.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown Metal phases"):
            metal_piop_eval.ram_cycle_family_breakdown(
                shadow_fallback, "metal", log_n
            )

        overlapping = complete_ram_sparse_trace(log_n, "metal")
        raf_prepare = next(
            event
            for event in overlapping
            if event["name"] == "RamRafEvaluation::prepare"
        )
        next(
            event
            for event in overlapping
            if event["name"] == "RamValCheck::prepare"
        )["ts"] = raf_prepare["ts"]
        next(
            event
            for event in overlapping
            if event["name"] == "MetalRamValCheck::route"
        )["ts"] = raf_prepare["ts"] + 10.0
        with self.assertRaisesRegex(ValueError, "canonical spans overlap"):
            metal_piop_eval.ram_cycle_family_breakdown(
                overlapping, "metal", log_n
            )

    def test_ram_local_kernel_primary_uses_the_full_charged_family(self) -> None:
        result = {
            "ram_cycle_family": {
                "components": {"raw_member_us": 200.0, "charged_member_us": 260.0}
            },
        }
        self.assertEqual(
            metal_piop_eval.local_kernel_primary_us(result, "RamCycleFamily"),
            260.0,
        )
        self.assertNotIn("RamReadWriteChecking", metal_piop_eval.LOCAL_KERNELS)
        self.assertNotIn("RamHammingBooleanity", metal_piop_eval.LOCAL_KERNELS)

    def test_ram_sparse_summary_charges_owner_once_and_rejects_partial_records(
        self,
    ) -> None:
        base = {
            "cpu_us": 2_000.0,
            "metal_us": 400.0,
            "cpu_prepare_us": 10.0,
            "metal_prepare_us": 10.0,
            "cpu_instruction_ra_us": 500.0,
            "metal_instruction_ra_us": 100.0,
            "cpu_bytecode_us": 500.0,
            "metal_bytecode_us": 100.0,
            "cpu_instruction_input_us": 500.0,
            "metal_instruction_input_us": 100.0,
            "cpu_registers_claim_us": 500.0,
            "metal_registers_claim_us": 100.0,
            "cpu_instruction_read_raf_us": 500.0,
            "metal_instruction_read_raf_us": 100.0,
            "cpu_booleanity_address_us": 500.0,
            "metal_booleanity_address_us": 100.0,
            "cpu_hamming_weight_us": 500.0,
            "metal_hamming_weight_us": 100.0,
            "cpu_hamming_weight_service_us": 500.0,
            "metal_hamming_weight_service_us": 100.0,
            "cpu_ram_read_write_us": 600.0,
            "metal_ram_read_write_us": 80.0,
            "metal_ram_read_write_charged_us": 120.0,
            "cpu_ram_raf_evaluation_us": 600.0,
            "metal_ram_raf_evaluation_us": 80.0,
            "cpu_ram_val_check_us": 600.0,
            "metal_ram_val_check_us": 80.0,
            "cpu_ram_ra_claim_reduction_us": 600.0,
            "metal_ram_ra_claim_reduction_us": 80.0,
            "cpu_ram_hamming_us": 600.0,
            "metal_ram_hamming_us": 80.0,
            "metal_ram_hamming_charged_us": 120.0,
            "cpu_ram_ra_virtualization_us": 600.0,
            "metal_ram_ra_virtualization_us": 80.0,
            "metal_ram_cycle_family_owner_us": 40.0,
            "metal_ram_cycle_family_witness_prepare_us": 120.0,
        }
        pairs = [
            {
                **base,
                "order": ["optimized", "metal"]
                if index % 2 == 0
                else ["metal", "optimized"],
            }
            for index in range(5)
        ]
        metrics = metal_piop_eval.summarize_pairs(pairs)
        self.assertEqual(metrics["ram_read_write_speedup"], 7.5)
        self.assertEqual(
            metrics["ram_read_write_standalone_charged_speedup"], 5.0
        )
        self.assertEqual(
            metrics["ram_hamming_booleanity_standalone_charged_speedup"], 5.0
        )
        self.assertEqual(metrics["ram_cycle_family_speedup"], 6.0)
        self.assertEqual(metrics["metal_ram_cycle_family_raw_ms_samples"], [0.48] * 5)
        self.assertEqual(metrics["metal_ram_cycle_family_ms_samples"], [0.6] * 5)
        self.assertEqual(
            metrics["metal_ram_cycle_family_owner_ms_samples"], [0.04] * 5
        )
        self.assertEqual(
            metrics["metal_ram_cycle_family_witness_prepare_ms_samples"],
            [0.12] * 5,
        )
        self.assertFalse(metrics["ram_standalone_charged_metrics_additive"])
        self.assertTrue(metrics["ram_read_write_standalone_charged_decision"]["clears"])
        self.assertTrue(metrics["ram_cycle_family_decision"]["clears"])
        for decision in (
            "ram_raf_evaluation_decision",
            "ram_read_write_decision",
            "ram_val_check_decision",
            "ram_ra_claim_reduction_decision",
            "ram_hamming_booleanity_decision",
            "ram_ra_virtualization_decision",
        ):
            self.assertTrue(metrics[decision]["clears"])

        partial = [dict(pair) for pair in pairs]
        for pair in partial:
            pair.pop("metal_ram_cycle_family_witness_prepare_us")
        with self.assertRaisesRegex(ValueError, "exact full RAM"):
            metal_piop_eval.summarize_pairs(partial)

        double_charged = [dict(pair) for pair in pairs]
        double_charged[0]["metal_ram_hamming_charged_us"] = 160.0
        with self.assertRaisesRegex(ValueError, "include the owner once"):
            metal_piop_eval.summarize_pairs(double_charged)

        without_ram = [
            {
                key: value
                for key, value in pair.items()
                if "_ram_" not in key
            }
            for pair in pairs
        ]
        no_ram_metrics = metal_piop_eval.summarize_pairs(without_ram)
        self.assertIsNone(no_ram_metrics["ram_read_write_speedup"])
        self.assertIsNone(no_ram_metrics["ram_cycle_family_speedup"])
        self.assertEqual(no_ram_metrics["paired_ram_read_write_speedups"], [])

    def test_outer_registers_carrier_is_complete_overwrite_storage(self) -> None:
        geometry = metal_piop_eval.outer_remainder_storage_geometry(26, True, True)
        carrier = geometry["registers_claim_carrier"]
        self.assertEqual(carrier["partial_bytes"], 100_663_296)
        self.assertEqual(carrier["component_bytes"], 393_216)
        self.assertEqual(carrier["rd_bytes"], 536_870_912)
        self.assertEqual(carrier["owned_bytes"], 637_927_424)
        self.assertEqual(
            geometry["owned_bytes"] - geometry["initialization_bytes"],
            carrier["owned_bytes"],
        )

    def test_instruction_input_alias_publication_is_route_scoped(self) -> None:
        events = complete_instruction_input_trace(26, "metal")
        rounds = [
            event
            for event in events
            if event["name"] == "InstructionInput::prove_round"
        ]
        midpoint = rounds[(26 + 1) // 2]
        events.append(
            {
                "name": "MetalInstructionInput::registers_claim_alias_publish",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": float(midpoint["ts"]) + 1.0,
                "dur": 1.0,
            }
        )
        observed = metal_piop_eval.instruction_input_member_breakdown(
            events,
            "metal",
            26,
            16,
            False,
            True,
        )
        self.assertEqual(
            observed["metal_counts"]["registers_claim_alias_publish"], 1
        )
        with self.assertRaisesRegex(ValueError, "span counts"):
            metal_piop_eval.instruction_input_member_breakdown(
                events,
                "metal",
                26,
                16,
                False,
                False,
            )

    def test_registers_claim_parser_requires_the_carrier_alias_route(self) -> None:
        events, carrier = complete_registers_claim_trace(26)
        observed = metal_piop_eval.registers_claim_member_breakdown(
            events,
            "metal",
            26,
            "outer-carrier-alias-hybrid",
            carrier,
        )
        resources = observed["resource_observation"]
        self.assertEqual(resources["route"]["fallback_reason"], "none")
        self.assertEqual(resources["prepare"]["source_upload_bytes"], 0)
        self.assertEqual(resources["midpoint"]["useful_half_width_terms"], 1 << 26)

        drifted = copy.deepcopy(events)
        route = next(
            event
            for event in drifted
            if event["name"] == "MetalRegistersClaimReduction::route"
        )
        route["args"]["fallback_reason"] = "missing_stage1_carry"
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            metal_piop_eval.registers_claim_member_breakdown(
                drifted,
                "metal",
                26,
                "outer-carrier-alias-hybrid",
                carrier,
            )

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
            "storage_initialization=full product_uniskip_carrier=false "
            "registers_claim_carrier=false"
        )
        config = metal_piop_eval.validate_outer_remainder_stdout(
            stdout, "metal"
        )
        self.assertEqual(config["cutoff"], 1 << 16)
        self.assertEqual(config["binding_plan"], "b_only_v1")
        self.assertFalse(config["product_uniskip_carrier"])
        self.assertFalse(config["registers_claim_carrier"])
        self.assertIsNone(
            metal_piop_eval.validate_outer_remainder_stdout("", "optimized")
        )

    def test_validates_registers_claim_runtime_config(self) -> None:
        stdout = (
            "REGISTERS_CLAIM_METAL_CONFIG backend=metal "
            "implementation=outer-carrier-alias-hybrid trace_cutoff=33554432"
        )
        config = metal_piop_eval.validate_registers_claim_stdout(
            stdout,
            "metal",
            "outer-carrier-alias-hybrid",
            25,
        )
        self.assertEqual(config["trace_cutoff"], 1 << 25)
        self.assertIsNone(
            metal_piop_eval.validate_registers_claim_stdout(
                "",
                "optimized",
                "outer-carrier-alias-hybrid",
                25,
            )
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
                    "cpu_registers_claim_us": 80.0,
                    "metal_registers_claim_us": 16.0,
                    "cpu_instruction_read_raf_us": 80.0,
                    "metal_instruction_read_raf_us": 16.0,
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
                    "cpu_registers_claim_us": 90.0,
                    "metal_registers_claim_us": 30.0,
                    "cpu_instruction_read_raf_us": 90.0,
                    "metal_instruction_read_raf_us": 30.0,
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
        with self.assertRaisesRegex(ValueError, "missing required member timing"):
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
            "cpu_registers_claim_us": 800.0,
            "metal_registers_claim_us": 160.0,
            "cpu_instruction_read_raf_us": 800.0,
            "metal_instruction_read_raf_us": 160.0,
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

    def test_bytecode_address_charged_gate_rejects_moved_work(self) -> None:
        base = {
            "cpu_us": 10_000.0,
            "metal_us": 2_000.0,
            "cpu_prepare_us": 900.0,
            "metal_prepare_us": 1_400.0,
            "cpu_instruction_ra_us": 700.0,
            "metal_instruction_ra_us": 100.0,
            "cpu_bytecode_us": 1_000.0,
            "metal_bytecode_us": 200.0,
            "cpu_bytecode_address_us": 6_000.0,
            "metal_bytecode_address_us": 1_000.0,
            "metal_bytecode_address_control_prepare_us": 900.0,
            "cpu_instruction_input_us": 800.0,
            "metal_instruction_input_us": 160.0,
            "cpu_registers_claim_us": 800.0,
            "metal_registers_claim_us": 160.0,
            "cpu_instruction_read_raf_us": 800.0,
            "metal_instruction_read_raf_us": 160.0,
            "cpu_booleanity_address_us": 1_000.0,
            "metal_booleanity_address_us": 200.0,
            "cpu_hamming_weight_us": 900.0,
            "metal_hamming_weight_us": 180.0,
            "cpu_hamming_weight_service_us": 990.0,
            "metal_hamming_weight_service_us": 180.0,
        }
        pairs = [
            {
                **base,
                "order": ["optimized", "metal"]
                if index % 2 == 0
                else ["metal", "optimized"],
                "producer_order": ["target", "control"]
                if index % 2 == 0
                else ["control", "target"],
            }
            for index in range(5)
        ]
        metrics = metal_piop_eval.summarize_pairs(pairs)
        self.assertEqual(metrics["bytecode_read_raf_address_speedup"], 6.0)
        self.assertEqual(
            metrics["paired_bytecode_read_raf_address_speedups"], [6.0] * 5
        )
        self.assertEqual(
            metrics["cpu_bytecode_read_raf_address_ms_samples"], [6.0] * 5
        )
        self.assertEqual(
            metrics["metal_bytecode_read_raf_address_ms_samples"], [1.0] * 5
        )
        self.assertTrue(metrics["bytecode_read_raf_address_decision"]["clears"])
        self.assertEqual(metrics["bytecode_read_raf_address_charged_speedup"], 4.0)
        self.assertEqual(
            metrics[
                "bytecode_read_raf_address_backend_witness_prepare_delta_ms_samples"
            ],
            [0.5] * 5,
        )
        self.assertEqual(
            metrics[
                "bytecode_read_raf_address_charged_producer_delta_ms_samples"
            ],
            [0.5] * 5,
        )
        self.assertEqual(
            metrics[
                "bytecode_read_raf_address_target_control_prepare_delta_ms_samples"
            ],
            [0.5] * 5,
        )
        self.assertEqual(
            metrics["metal_bytecode_read_raf_address_control_prepare_ms_samples"],
            [0.9] * 5,
        )
        self.assertEqual(
            metrics["charged_metal_address_ms_samples"],
            [1.5] * 5,
        )
        self.assertFalse(
            metrics["bytecode_read_raf_address_charged_decision"]["clears"]
        )

        changed_optimized = copy.deepcopy(pairs)
        for index, pair in enumerate(changed_optimized):
            pair["cpu_prepare_us"] = 1_000.0 if index % 2 == 0 else 1_500.0
        unchanged_charge = metal_piop_eval.summarize_pairs(changed_optimized)
        self.assertEqual(
            unchanged_charge[
                "bytecode_read_raf_address_backend_witness_prepare_delta_ms_samples"
            ],
            [0.4, -0.1, 0.4, -0.1, 0.4],
        )
        self.assertEqual(
            unchanged_charge[
                "bytecode_read_raf_address_charged_producer_delta_ms_samples"
            ],
            [0.5] * 5,
        )
        self.assertEqual(
            unchanged_charge["charged_metal_address_ms_samples"],
            [1.5] * 5,
        )

        nonpositive_control_delta = copy.deepcopy(pairs)
        for index, pair in enumerate(nonpositive_control_delta):
            pair["metal_bytecode_address_control_prepare_us"] = (
                1_400.0 if index % 2 == 0 else 1_500.0
            )
        uncharged = metal_piop_eval.summarize_pairs(nonpositive_control_delta)
        self.assertEqual(
            uncharged[
                "bytecode_read_raf_address_charged_producer_delta_ms_samples"
            ],
            [0.0] * 5,
        )
        self.assertEqual(uncharged["charged_metal_address_ms_samples"], [1.0] * 5)
        self.assertEqual(
            uncharged["paired_bytecode_read_raf_address_charged_speedups"],
            [6.0] * 5,
        )
        self.assertTrue(
            uncharged["bytecode_read_raf_address_charged_decision"]["clears"]
        )

        incomplete = copy.deepcopy(pairs)
        incomplete[0].pop("metal_bytecode_address_us")
        with self.assertRaisesRegex(ValueError, "incomplete Bytecode address"):
            metal_piop_eval.summarize_pairs(incomplete)

        incomplete_control = copy.deepcopy(pairs)
        incomplete_control[0].pop("producer_order")
        with self.assertRaisesRegex(ValueError, "incomplete Bytecode address producer"):
            metal_piop_eval.summarize_pairs(incomplete_control)

    def test_bytecode_address_fused_charge_sums_signed_stage1_and_scatter_deltas(
        self,
    ) -> None:
        base = {
            "cpu_us": 10_000.0,
            "metal_us": 2_000.0,
            "cpu_prepare_us": 900.0,
            "metal_prepare_us": 1_400.0,
            "cpu_instruction_ra_us": 700.0,
            "metal_instruction_ra_us": 100.0,
            "cpu_bytecode_us": 1_000.0,
            "metal_bytecode_us": 200.0,
            "cpu_bytecode_address_us": 6_000.0,
            "metal_bytecode_address_us": 1_000.0,
            "metal_bytecode_address_control_prepare_us": 900.0,
            "metal_bytecode_address_stage1_topology_us": 300.0,
            "metal_bytecode_address_control_stage1_topology_us": 100.0,
            "metal_bytecode_address_irraf_scatter_us": 700.0,
            "metal_bytecode_address_control_irraf_scatter_us": 500.0,
            "cpu_instruction_input_us": 800.0,
            "metal_instruction_input_us": 160.0,
            "cpu_registers_claim_us": 800.0,
            "metal_registers_claim_us": 160.0,
            "cpu_instruction_read_raf_us": 800.0,
            "metal_instruction_read_raf_us": 160.0,
            "cpu_booleanity_address_us": 1_000.0,
            "metal_booleanity_address_us": 200.0,
            "cpu_hamming_weight_us": 900.0,
            "metal_hamming_weight_us": 180.0,
            "cpu_hamming_weight_service_us": 990.0,
            "metal_hamming_weight_service_us": 180.0,
        }
        pairs = [
            {
                **base,
                "order": ["optimized", "metal"]
                if index % 2 == 0
                else ["metal", "optimized"],
                "producer_order": ["target", "control"]
                if index % 2 == 0
                else ["control", "target"],
            }
            for index in range(5)
        ]

        metrics = metal_piop_eval.summarize_pairs(pairs)
        self.assertEqual(
            metrics["bytecode_read_raf_address_stage1_topology_delta_ms_samples"],
            [0.2] * 5,
        )
        self.assertEqual(
            metrics["bytecode_read_raf_address_irraf_scatter_delta_ms_samples"],
            [0.2] * 5,
        )
        self.assertEqual(
            metrics["bytecode_read_raf_address_signed_fused_producer_delta_ms_samples"],
            [0.4] * 5,
        )
        self.assertEqual(
            metrics["bytecode_read_raf_address_charged_producer_delta_ms_samples"],
            [0.4] * 5,
        )
        self.assertEqual(metrics["charged_metal_address_ms_samples"], [1.4] * 5)
        self.assertFalse(
            metrics["bytecode_read_raf_address_charged_decision"]["clears"]
        )

        signed = copy.deepcopy(pairs)
        for pair in signed:
            pair["metal_bytecode_address_stage1_topology_us"] = 100.0
            pair["metal_bytecode_address_control_stage1_topology_us"] = 400.0
        uncharged = metal_piop_eval.summarize_pairs(signed)
        self.assertEqual(
            uncharged[
                "bytecode_read_raf_address_signed_fused_producer_delta_ms_samples"
            ],
            [-0.1] * 5,
        )
        self.assertEqual(
            uncharged[
                "bytecode_read_raf_address_charged_producer_delta_ms_samples"
            ],
            [0.0] * 5,
        )
        self.assertEqual(uncharged["charged_metal_address_ms_samples"], [1.0] * 5)

        incomplete = copy.deepcopy(pairs)
        incomplete[0].pop("metal_bytecode_address_control_irraf_scatter_us")
        with self.assertRaisesRegex(ValueError, "incomplete fused Bytecode address"):
            metal_piop_eval.summarize_pairs(incomplete)

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

    def test_validates_bytecode_address_runtime_record(self) -> None:
        record = (
            "BYTECODE_ADDRESS_METAL_CONFIG backend=metal "
            "implementation=address-major trace_cutoff=67108864 outer_tiles=8"
        )
        self.assertIsNone(
            metal_piop_eval.validate_bytecode_address_stdout(
                "", "optimized", "address-major", 8, 26
            )
        )
        self.assertEqual(
            metal_piop_eval.validate_bytecode_address_stdout(
                record, "metal", "address-major", 8, 26
            ),
            {
                "implementation": "address-major",
                "trace_cutoff": 1 << 26,
                "outer_tiles": 8,
            },
        )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            metal_piop_eval.validate_bytecode_address_stdout(
                record, "metal", "address-major", 4, 26
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            metal_piop_eval.validate_bytecode_address_stdout(
                record + "\n" + record, "metal", "address-major", 8, 26
            )

    def test_validates_instruction_input_runtime_record(self) -> None:
        record = "INSTRUCTION_INPUT_METAL_CONFIG backend=metal trace_cutoff=33554432 cutoff=65536 native_message_threads=256 native_transition_threads=128 dense_transition_threads=128 storage_initialization=minimal dense_storage_mode=Owned native_primer=async"
        self.assertIsNone(
            metal_piop_eval.validate_instruction_input_stdout("", "optimized")
        )
        observed = metal_piop_eval.validate_instruction_input_stdout(record, "metal")
        assert observed is not None
        self.assertEqual(observed["storage_initialization"], "minimal")
        self.assertEqual(observed["dense_storage_mode"], "Owned")
        self.assertEqual(observed["native_primer"], "async")
        borrowed = record.replace("dense_storage_mode=Owned", "dense_storage_mode=OuterResidual")
        self.assertIsNotNone(
            metal_piop_eval.validate_instruction_input_stdout(
                borrowed, "metal", borrow_outer_residual=True
            )
        )
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
        borrowed = metal_piop_eval.instruction_input_member_breakdown(
            complete_instruction_input_trace(26, "metal", 16, True),
            "metal",
            26,
            16,
            True,
        )
        borrowed_resources = borrowed["resource_observation"]
        self.assertTrue(borrowed_resources["borrowed_outer_residual"])
        self.assertEqual(
            borrowed_resources["allocation"]["owned_device_bytes"],
            metal_piop_eval.instruction_input_sequence_auxiliary_storage_bytes(26),
        )
        self.assertEqual(
            borrowed_resources["allocation"]["reused_device_bytes"],
            96 * (1 << 26),
        )
        self.assertEqual(
            borrowed_resources["storage_initialization"]["buffer_identities"][:2],
            [203, 203],
        )
        self.assertEqual(
            borrowed_resources["outer_residual_transfer"],
            {
                "resident_rows": 1 << 26,
                "outer_residual_generation": 7,
                "compact_rows_storage_id": 202,
                "residual_rows_storage_id": 203,
                "device_registry_id": 1,
                "outer_sequence_owned_bytes": 5_242_880,
                "outer_sequence_consumed": True,
                "compact_rows_transferred": True,
                "residual_rows_transferred": True,
            },
        )
        self.assertEqual(
            borrowed_resources["dense_ranges"],
            {
                "dense_a_offset_bytes": 0,
                "dense_a_length_bytes": 64 * (1 << 26),
                "dense_b_offset_bytes": 64 * (1 << 26),
                "dense_b_length_bytes": 32 * (1 << 26),
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

    def test_requires_exact_bytecode_address_major_route_and_resources(self) -> None:
        optimized = metal_piop_eval.bytecode_address_member_breakdown(
            complete_bytecode_address_trace(26, "optimized"),
            "optimized",
            26,
            "address-major",
            8,
            26,
        )
        self.assertFalse(any(optimized["metal_counts"].values()))
        self.assertIsNone(optimized["resource_observation"])

        observed = metal_piop_eval.bytecode_address_member_breakdown(
            complete_bytecode_address_trace(26, "metal"),
            "metal",
            26,
            "address-major",
            8,
            26,
            bytecode_address_stage1_source(26),
        )
        self.assertEqual(observed["outer_counts"]["prove_round"], 13)
        self.assertEqual(observed["metal_counts"]["address_major_complete"], 1)
        self.assertEqual(observed["components"]["member_us"], 1_660.0)
        resources = observed["resource_observation"]
        assert resources is not None
        self.assertEqual(
            resources["carrier_publish"]["producer_persistent_write_bytes"],
            resources["carrier_publish"]["carrier_resident_bytes"],
        )
        self.assertEqual(
            resources["carrier_publish"]["carrier_occurrence_bytes"],
            2 * resources["carrier_publish"]["physical_rows"],
        )
        self.assertEqual(
            resources["address_major_complete"]["member_carrier_owned_bytes"], 0
        )
        self.assertEqual(resources["address_major_complete"]["member_source_scans"], 0)

        observed_log27 = metal_piop_eval.bytecode_address_member_breakdown(
            complete_bytecode_address_trace(27, "metal"),
            "metal",
            27,
            "address-major",
            8,
            26,
            bytecode_address_stage1_source(27),
        )
        log27_resources = observed_log27["resource_observation"]
        assert log27_resources is not None
        self.assertEqual(
            log27_resources["address_major_complete"]["source_rows_bytes"],
            40 * (1 << 27),
        )
        self.assertEqual(
            log27_resources["address_major_complete"]["carrier_resident_bytes"],
            log27_resources["carrier_publish"]["carrier_resident_bytes"],
        )

    def test_bytecode_address_cpu_route_remains_a_diagnostic_choice(self) -> None:
        events = complete_bytecode_address_trace(26, "optimized")
        events.append(
            {
                "name": "MetalBytecodeReadRafAddress::route",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 210.0,
                "dur": 480.0,
                "args": {
                    "cycles": str(1 << 26),
                    "requested": "cpu",
                    "realized_route": "cpu",
                    "fallback_reason": "configured_cpu",
                },
            }
        )
        observed = metal_piop_eval.bytecode_address_member_breakdown(
            events, "metal", 26, "cpu", 8, 26
        )
        self.assertEqual(observed["metal_counts"]["route"], 1)
        self.assertEqual(observed["route_observation"]["realized_route"], "cpu")
        self.assertIsNone(observed["resource_observation"])

    def test_bytecode_address_fused_stage1_grouped_route_is_exact(self) -> None:
        log_n = 26
        source = bytecode_address_stage1_source(log_n)
        scatter = fused_bytecode_stage1_scatter(log_n)
        observed = metal_piop_eval.bytecode_address_member_breakdown(
            complete_fused_bytecode_address_trace(log_n),
            "metal",
            log_n,
            "address-major",
            8,
            26,
            source,
            scatter,
        )
        self.assertEqual(
            observed["route_observation"]["realized_route"],
            "address_major_fused_stage1_grouped_v1",
        )
        topology = observed["topology_observation"]
        self.assertIsNotNone(topology)
        assert topology is not None
        self.assertTrue(topology["enabled"])
        self.assertEqual(
            topology["descriptor_elements"],
            topology["descriptors"] + topology["chunks"],
        )
        self.assertEqual(topology["max_descriptors_per_chunk"], 504)
        self.assertEqual(topology["max_pivots_per_chunk"], 11)
        resources = observed["resource_observation"]
        self.assertIsNotNone(resources)
        assert resources is not None
        self.assertEqual(resources["producer_kind"], "fused_stage1_grouped_v1")
        publish = resources["carrier_publish"]
        complete = resources["address_major_complete"]
        self.assertEqual(
            complete["producer_persistent_write_bytes"],
            10 * publish["physical_rows"],
        )
        self.assertEqual(
            complete["producer_logical_movement_bytes"],
            complete["producer_persistent_write_bytes"]
            + complete["producer_topology_read_bytes"],
        )
        self.assertEqual(publish["additional_source_row_scans"], 0)
        self.assertEqual(publish["member_upload_bytes"], 0)

        control = metal_piop_eval.bytecode_address_member_breakdown(
            complete_fused_bytecode_address_trace(log_n, control=True),
            "metal",
            log_n,
            "cpu",
            8,
            26,
            source,
        )
        self.assertIsNone(control["resource_observation"])
        self.assertIsNotNone(control["topology_observation"])
        self.assertFalse(control["topology_observation"]["enabled"])
        self.assertEqual(control["topology_observation"]["covered_rows"], 0)

    def test_bytecode_address_fused_route_mutations_fail_closed(self) -> None:
        log_n = 26
        source = bytecode_address_stage1_source(log_n)
        scatter = fused_bytecode_stage1_scatter(log_n)

        missing_topology = complete_fused_bytecode_address_trace(log_n)
        missing_topology.remove(
            next(
                trace_event
                for trace_event in missing_topology
                if trace_event["name"]
                == "MetalBytecodeReadRafAddress::fused_topology_prepare"
            )
        )
        bad_sentinel = complete_fused_bytecode_address_trace(log_n)
        topology = next(
            trace_event
            for trace_event in bad_sentinel
            if trace_event["name"]
            == "MetalBytecodeReadRafAddress::fused_topology_prepare"
        )
        topology["args"]["descriptor_elements"] = str(
            int(topology["args"]["descriptor_elements"]) - 1
        )
        hidden_upload = complete_fused_bytecode_address_trace(log_n)
        publish = next(
            trace_event
            for trace_event in hidden_upload
            if trace_event["name"]
            == "MetalBytecodeReadRafAddress::fused_carrier_publish"
        )
        publish["args"]["member_upload_bytes"] = "1"
        incomplete_publish = complete_fused_bytecode_address_trace(log_n)
        publish = next(
            trace_event
            for trace_event in incomplete_publish
            if trace_event["name"]
            == "MetalBytecodeReadRafAddress::fused_carrier_publish"
        )
        publish["args"].pop("carrier_buffers")
        mismatched_receipt = complete_fused_bytecode_address_trace(log_n)
        topology = next(
            trace_event
            for trace_event in mismatched_receipt
            if trace_event["name"]
            == "MetalBytecodeReadRafAddress::fused_topology_prepare"
        )
        topology["args"]["work_item_storage_id"] = "999"

        for label, events, message in (
            ("missing topology", missing_topology, "span counts"),
            ("sentinel accounting", bad_sentinel, "topology receipt"),
            ("hidden upload", hidden_upload, "publication ledger"),
            ("incomplete publication", incomplete_publish, "unexpected argument fields"),
            ("receipt mismatch", mismatched_receipt, "does not match its scatter"),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    metal_piop_eval.bytecode_address_member_breakdown(
                        events,
                        "metal",
                        log_n,
                        "address-major",
                        8,
                        26,
                        source,
                        scatter,
                    )

        non_inert_control = complete_fused_bytecode_address_trace(log_n, control=True)
        control_topology = next(
            trace_event
            for trace_event in non_inert_control
            if trace_event["name"]
            == "MetalBytecodeReadRafAddress::fused_topology_prepare"
        )
        control_topology["args"]["work_items"] = "1"
        with self.assertRaisesRegex(ValueError, "not inert"):
            metal_piop_eval.bytecode_address_member_breakdown(
                non_inert_control,
                "metal",
                log_n,
                "cpu",
                8,
                26,
                source,
            )

    def test_bytecode_address_major_mutations_fail_closed(self) -> None:
        source = bytecode_address_stage1_source(26)

        wrong_route = complete_bytecode_address_trace(26, "metal")
        route = next(
            event
            for event in wrong_route
            if event["name"] == "MetalBytecodeReadRafAddress::route"
        )
        route["args"]["realized_route"] = "cpu"

        missing_complete = complete_bytecode_address_trace(26, "metal")
        missing_complete.remove(
            next(
                event
                for event in missing_complete
                if event["name"]
                == "MetalBytecodeReadRafAddress::address_major_complete"
            )
        )

        wrong_carrier = complete_bytecode_address_trace(26, "metal")
        complete = next(
            event
            for event in wrong_carrier
            if event["name"]
            == "MetalBytecodeReadRafAddress::address_major_complete"
        )
        complete["args"]["carrier_occurrence_storage_id"] = "999"

        wrong_source = complete_bytecode_address_trace(26, "metal")
        publish = next(
            event
            for event in wrong_source
            if event["name"] == "MetalBytecodeReadRafAddress::carrier_publish"
        )
        publish["args"]["source_generation"] = "8"

        hidden_scan = complete_bytecode_address_trace(26, "metal")
        complete = next(
            event
            for event in hidden_scan
            if event["name"]
            == "MetalBytecodeReadRafAddress::address_major_complete"
        )
        complete["args"]["member_source_scans"] = "1"

        wrong_movement = complete_bytecode_address_trace(26, "metal")
        publish = next(
            event
            for event in wrong_movement
            if event["name"] == "MetalBytecodeReadRafAddress::carrier_publish"
        )
        publish["args"]["producer_logical_movement_bytes"] = str(
            int(publish["args"]["producer_logical_movement_bytes"]) + 1
        )

        atom_path = complete_bytecode_address_trace(26, "metal")
        atom_path.append(
            {
                "name": "MetalBytecodeReadRafAddress::atom_prepare",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 230.0,
                "dur": 1.0,
            }
        )

        impossible_work_items = complete_bytecode_address_trace(26, "metal")
        impossible_count = 1
        for trace_event in impossible_work_items:
            if trace_event["name"] == "MetalBytecodeReadRafAddress::carrier_publish":
                args = trace_event["args"]
                physical_rows = int(args["physical_rows"])
                offset_bytes = int(args["carrier_address_offset_bytes"])
                resident_bytes = 10 * physical_rows + 8 * impossible_count + offset_bytes
                args.update(
                    {
                        "work_items": str(impossible_count),
                        "carrier_work_item_bytes": str(8 * impossible_count),
                        "carrier_resident_bytes": str(resident_bytes),
                        "producer_persistent_write_bytes": str(resident_bytes),
                        "producer_logical_movement_bytes": str(
                            30 * physical_rows + 16 * impossible_count + offset_bytes
                        ),
                    }
                )
            elif trace_event["name"] == "MetalBytecodeReadRafAddress::address_major_complete":
                args = trace_event["args"]
                physical_rows = int(args["physical_rows"])
                offset_bytes = int(args["carrier_address_offset_bytes"])
                resident_bytes = 10 * physical_rows + 8 * impossible_count + offset_bytes
                equality_bytes = int(args["equality_bytes"])
                padding_bytes = int(args["padding_bytes"])
                output_bytes = int(args["output_readback_bytes"])
                partial_bytes = 16 * 9 * impossible_count
                args.update(
                    {
                        "work_items": str(impossible_count),
                        "carrier_work_item_bytes": str(8 * impossible_count),
                        "carrier_resident_bytes": str(resident_bytes),
                        "producer_persistent_write_bytes": str(resident_bytes),
                        "producer_logical_movement_bytes": str(
                            30 * physical_rows + 16 * impossible_count + offset_bytes
                        ),
                        "partial_bytes": str(partial_bytes),
                        "member_owned_bytes": str(
                            equality_bytes + padding_bytes + partial_bytes + output_bytes
                        ),
                    }
                )

        for label, events, message in (
            ("fallback", wrong_route, "fail-closed route"),
            ("completion", missing_complete, "span counts"),
            ("carrier identity", wrong_carrier, "completion ledger"),
            ("source provenance", wrong_source, "Stage1 source"),
            ("hidden scan", hidden_scan, "completion ledger"),
            ("logical movement", wrong_movement, "publication ledger"),
            ("atom path", atom_path, "forbidden Atom path"),
            ("impossible work-item count", impossible_work_items, "publication ledger"),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    metal_piop_eval.bytecode_address_member_breakdown(
                        events,
                        "metal",
                        26,
                        "address-major",
                        8,
                        26,
                        source,
                    )

        geometry_mutations = {
            "worker_variant": "legacy",
            "worker_simd_width": "16",
            "worker_threads": "256",
            "worker_items_per_threadgroup": "1",
            "worker_threadgroups": "1",
            "worker_tail_slots": "0",
            "worker_dynamic_threadgroup_bytes": "1",
            "worker_static_threadgroup_bytes": "1",
            "worker_threadgroup_bytes": "1",
            "reducer_threads": "128",
            "reducer_threadgroups": "1",
            "reducer_static_threadgroup_bytes": "1",
        }
        for field, value in geometry_mutations.items():
            events = complete_bytecode_address_trace(26, "metal")
            complete = next(
                event
                for event in events
                if event["name"]
                == "MetalBytecodeReadRafAddress::address_major_complete"
            )
            complete["args"][field] = value
            with self.subTest(geometry_field=field):
                with self.assertRaisesRegex(ValueError, "completion ledger"):
                    metal_piop_eval.bytecode_address_member_breakdown(
                        events,
                        "metal",
                        26,
                        "address-major",
                        8,
                        26,
                        source,
                    )

        wrong_projection_source = {**source, "explicit_rows": int(source["explicit_rows"]) - 1}
        with self.assertRaisesRegex(ValueError, "Stage1 projection"):
            metal_piop_eval.bytecode_address_member_breakdown(
                complete_bytecode_address_trace(26, "metal"),
                "metal",
                26,
                "address-major",
                8,
                26,
                wrong_projection_source,
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
                "explicit_rows": 1 << 26,
                "row_bytes": 48,
                "prepare_storage_id": 202,
                "stage1_storage_id": 202,
                "stage3_storage_id": 202,
                "residual_storage_id": 203,
                "outer_residual_transfer": None,
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
            "cpu_registers_claim_us": 1_000.0,
            "metal_registers_claim_us": 200.0,
            "cpu_instruction_read_raf_us": 1_000.0,
            "metal_instruction_read_raf_us": 200.0,
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
                    "cpu_registers_claim_us": 1_000.0,
                    "metal_registers_claim_us": 200.0,
                    "cpu_instruction_read_raf_us": 1_000.0,
                    "metal_instruction_read_raf_us": 200.0,
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
                "cpu_registers_claim_us": 800.0,
                "metal_registers_claim_us": 160.0,
                "cpu_instruction_read_raf_us": 800.0,
                "metal_instruction_read_raf_us": 160.0,
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
                    "cpu_registers_claim_us": 800.0,
                    "metal_registers_claim_us": 160.0,
                    "cpu_instruction_read_raf_us": 800.0,
                    "metal_instruction_read_raf_us": 160.0,
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
                    "cpu_registers_claim_us": 800.0,
                    "metal_registers_claim_us": 160.0,
                    "cpu_instruction_read_raf_us": 800.0,
                    "metal_instruction_read_raf_us": 160.0,
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

    def test_instruction_read_raf_is_a_local_kernel(self) -> None:
        self.assertEqual(
            metal_piop_eval.LOCAL_KERNELS["InstructionReadRaf"],
            {
                "name": "InstructionReadRaf",
                "metric": "instruction_read_raf_speedup",
                "paired_metric": "paired_instruction_read_raf_speedups",
                "backend_prefix": "MetalInstructionReadRaf::",
            },
        )
        result = {
            "attribution": {
                "kernels": [
                    {
                        "kernel": "InstructionReadRaf",
                        "wall_ms": 12.5,
                    }
                ]
            },
            "instruction_read_raf_member": {
                "components": {"member_us": 12_500.0}
            },
        }
        self.assertEqual(
            metal_piop_eval.local_kernel_primary_us(result, "InstructionReadRaf"),
            12_500.0,
        )

    def test_bytecode_address_phase_is_a_local_kernel(self) -> None:
        self.assertEqual(
            metal_piop_eval.LOCAL_KERNELS["BytecodeReadRafAddressPhase"],
            {
                "name": "BytecodeReadRafAddressPhase",
                "metric": "bytecode_read_raf_address_speedup",
                "paired_metric": "paired_bytecode_read_raf_address_speedups",
                "backend_prefix": "MetalBytecodeReadRafAddress::",
            },
        )
        result = {
            "bytecode_address_member": {
                "components": {"member_us": 1_250.0}
            }
        }
        self.assertEqual(
            metal_piop_eval.local_kernel_primary_us(
                result, "BytecodeReadRafAddressPhase"
            ),
            1_250.0,
        )

    def test_instruction_read_raf_stage1_route_is_exact(self) -> None:
        optimized = metal_piop_eval.instruction_read_raf_member_breakdown(
            complete_instruction_read_raf_trace(25, "optimized"),
            "optimized",
            25,
            scatter_threads=512,
        )
        self.assertIsNone(optimized["resource_observation"])
        self.assertFalse(any(optimized["metal_counts"].values()))

        events = complete_instruction_read_raf_trace(25, "metal", 512)
        observed = metal_piop_eval.instruction_read_raf_member_breakdown(
            events, "metal", 25, scatter_threads=512
        )
        self.assertEqual(observed["outer_counts"]["prove_round"], 153)
        self.assertEqual(observed["metal_counts"]["address_round"], 129)
        self.assertEqual(
            observed["source_observation"]["resident_device_bytes"],
            41 * (1 << 25),
        )
        self.assertEqual(
            observed["scatter_observation"]["additional_allocation_bytes"],
            1_244_397_992,
        )
        self.assertEqual(
            observed["scatter_observation"]["threads_per_threadgroup"], 512
        )

        mutated = copy.deepcopy(events)
        source = next(
            event
            for event in mutated
            if event["name"]
            == "MetalInstructionReadRaf::stage1_source_publish"
        )
        source["args"]["member_upload_bytes"] = "1"
        with self.assertRaisesRegex(ValueError, "source ledger"):
            metal_piop_eval.instruction_read_raf_member_breakdown(
                mutated, "metal", 25, scatter_threads=512
            )

        mutated = copy.deepcopy(events)
        scatter = next(
            event
            for event in mutated
            if event["name"]
            == "MetalInstructionReadRaf::stage1_grouped_scatter"
        )
        scatter["args"]["source_generation"] = "8"
        with self.assertRaisesRegex(ValueError, "scatter ledger"):
            metal_piop_eval.instruction_read_raf_member_breakdown(
                mutated, "metal", 25, scatter_threads=512
            )

        mutated = copy.deepcopy(events)
        mutated.append(
            {
                "name": "MetalInstructionReadRaf::sequence_prepare",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 1_170.0,
                "dur": 1.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "legacy or unknown"):
            metal_piop_eval.instruction_read_raf_member_breakdown(
                mutated, "metal", 25, scatter_threads=512
            )

        mutated = copy.deepcopy(events)
        resident_first = next(
            event
            for event in mutated
            if event["name"]
            == "MetalInstructionReadRaf::resident_first_message"
        )
        resident_first["ts"] += 20.0
        with self.assertRaisesRegex(ValueError, "resident_first_message round"):
            metal_piop_eval.instruction_read_raf_member_breakdown(
                mutated, "metal", 25, scatter_threads=512
            )

        with self.assertRaisesRegex(ValueError, "requires log-n at least 25"):
            metal_piop_eval.instruction_read_raf_member_breakdown(
                complete_instruction_read_raf_trace(24, "optimized"),
                "metal",
                24,
                scatter_threads=512,
            )

    def test_instruction_read_raf_fused_bytecode_scatter_is_exact(self) -> None:
        events = complete_instruction_read_raf_trace(
            25, "metal", 512, fused_bytecode=True
        )
        observed = metal_piop_eval.instruction_read_raf_member_breakdown(
            events,
            "metal",
            25,
            scatter_threads=512,
            expect_fused_bytecode_address=True,
        )
        fused = observed["fused_bytecode_observation"]
        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertTrue(fused["bytecode_fused"])
        self.assertEqual(fused["bytecode_occurrence_bytes"], 2 * ((1 << 25) - 1))
        self.assertEqual(fused["bytecode_magnitude_bytes"], 8 * ((1 << 25) - 1))
        self.assertEqual(fused["bytecode_max_descriptors_per_chunk"], 504)
        self.assertEqual(
            fused["bytecode_max_admitted_descriptors_per_chunk"], 512
        )
        self.assertEqual(fused["bytecode_max_pivots_per_chunk"], 11)
        self.assertEqual(fused["bytecode_max_admitted_pivots_per_chunk"], 15)
        self.assertEqual(fused["bytecode_dynamic_threadgroup_bytes"], 4390)
        self.assertEqual(fused["additional_source_row_scans"], 0)
        self.assertEqual(fused["member_upload_bytes"], 0)

        missing = copy.deepcopy(events)
        scatter = next(
            event
            for event in missing
            if event["name"] == "MetalInstructionReadRaf::stage1_grouped_scatter"
        )
        scatter["args"].pop("bytecode_occurrence_storage_id")
        with self.assertRaisesRegex(ValueError, "unexpected argument fields"):
            metal_piop_eval.instruction_read_raf_member_breakdown(
                missing,
                "metal",
                25,
                scatter_threads=512,
                expect_fused_bytecode_address=True,
            )

        for field, value in (
            ("bytecode_max_admitted_descriptors_per_chunk", "511"),
            ("bytecode_max_admitted_pivots_per_chunk", "14"),
            ("bytecode_threadgroup_memory_limit_bytes", "1"),
        ):
            invalid_admission = copy.deepcopy(events)
            scatter = next(
                event
                for event in invalid_admission
                if event["name"]
                == "MetalInstructionReadRaf::stage1_grouped_scatter"
            )
            scatter["args"][field] = value
            with self.subTest(admission_field=field):
                with self.assertRaisesRegex(ValueError, "fused Bytecode scatter ledger"):
                    metal_piop_eval.instruction_read_raf_member_breakdown(
                        invalid_admission,
                        "metal",
                        25,
                        scatter_threads=512,
                        expect_fused_bytecode_address=True,
                    )

        unexpected = complete_instruction_read_raf_trace(
            25, "metal", 512, fused_bytecode=True
        )
        with self.assertRaisesRegex(ValueError, "unexpected argument fields"):
            metal_piop_eval.instruction_read_raf_member_breakdown(
                unexpected, "metal", 25, scatter_threads=512
            )

    def test_instruction_read_raf_stage1_rows_are_reused_by_booleanity(self) -> None:
        source = metal_piop_eval.instruction_read_raf_member_breakdown(
            complete_instruction_read_raf_trace(26, "metal", 512),
            "metal",
            26,
            scatter_threads=512,
        )["source_observation"]
        self.assertIsNotNone(source)

        for events, parser in (
            (
                complete_booleanity_address_trace(26, "metal"),
                lambda trace: metal_piop_eval.booleanity_address_member_breakdown(
                    trace, "metal", 26, stage1_source=source
                ),
            ),
            (
                complete_hamming_weight_trace(26, "metal"),
                lambda trace: metal_piop_eval.hamming_weight_member_breakdown(
                    trace, "metal", 26, stage1_source=source
                ),
            ),
        ):
            for event in events:
                args = event.get("args")
                if not isinstance(args, dict):
                    continue
                if "resident_rows_storage_id" in args:
                    args["resident_rows_storage_id"] = str(
                        source["row_allocation_identity"]
                    )
                if event["name"] == "MetalBooleanityRows::stage5_prepare":
                    args.update(
                        {
                            "row_allocations": "0",
                            "row_upload_bytes": "0",
                            "source_kind": "stage1_owner_v1",
                            "source_generation": str(source["source_generation"]),
                            "source_completion_serial": str(
                                source["completion_serial"]
                            ),
                            "source_claim_allocation_identity": str(
                                source["claim_allocation_identity"]
                            ),
                        }
                    )
            lifecycle = parser(events)["row_lifecycle"]
            self.assertEqual(lifecycle["source_kind"], "stage1_owner_v1")
            self.assertEqual(lifecycle["stage5"]["row_allocations"], 0)
            self.assertEqual(lifecycle["stage5"]["row_upload_bytes"], 0)

            broken = copy.deepcopy(events)
            stage5 = next(
                event
                for event in broken
                if event["name"] == "MetalBooleanityRows::stage5_prepare"
            )
            stage5["args"]["source_generation"] = "8"
            with self.assertRaisesRegex(ValueError, "lifecycle"):
                parser(broken)

    def test_validates_instruction_read_raf_runtime_config(self) -> None:
        stdout = (
            "INSTRUCTION_READ_RAF_METAL_CONFIG backend=metal "
            "address_cutoff=33554432 cutoff=65536 stage1_scatter_threads=512"
        )
        self.assertEqual(
            metal_piop_eval.validate_instruction_read_raf_stdout(
                stdout, "metal", 512
            ),
            {
                "address_cutoff": 1 << 25,
                "cutoff": 1 << 16,
                "stage1_scatter_threads": 512,
            },
        )
        self.assertIsNone(
            metal_piop_eval.validate_instruction_read_raf_stdout(
                "", "optimized", 512
            )
        )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            metal_piop_eval.validate_instruction_read_raf_stdout(
                stdout, "metal", 256
            )

    def test_instruction_read_raf_summary_uses_complete_member_wall(self) -> None:
        base = {
            "cpu_us": 20_000.0,
            "metal_us": 4_000.0,
            "cpu_prepare_us": 10.0,
            "metal_prepare_us": 20.0,
            "cpu_instruction_ra_us": 700.0,
            "metal_instruction_ra_us": 100.0,
            "cpu_bytecode_us": 1_000.0,
            "metal_bytecode_us": 200.0,
            "cpu_instruction_input_us": 800.0,
            "metal_instruction_input_us": 160.0,
            "cpu_registers_claim_us": 800.0,
            "metal_registers_claim_us": 160.0,
            "cpu_instruction_read_raf_us": 3_750.0,
            "metal_instruction_read_raf_us": 750.0,
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
                **base,
                "order": ["optimized", "metal"]
                if index % 2 == 0
                else ["metal", "optimized"],
            }
            for index in range(5)
        ]
        metrics = metal_piop_eval.summarize_pairs(pairs)
        self.assertEqual(metrics["instruction_read_raf_speedup"], 5.0)
        self.assertEqual(metrics["paired_instruction_read_raf_speedups"], [5.0] * 5)
        self.assertEqual(
            metrics["cpu_instruction_read_raf_ms_samples"], [3.75] * 5
        )
        self.assertEqual(
            metrics["metal_instruction_read_raf_ms_samples"], [0.75] * 5
        )
        self.assertTrue(metrics["instruction_read_raf_decision"]["clears"])


if __name__ == "__main__":
    unittest.main()
