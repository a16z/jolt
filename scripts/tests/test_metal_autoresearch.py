import copy
import importlib.util
import io
import json
import os
import statistics
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "metal_autoresearch.py"
ROOT = SCRIPT.parents[1]
SPEC = importlib.util.spec_from_file_location("metal_autoresearch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
metal_autoresearch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metal_autoresearch)


class MetalAutoresearchTests(unittest.TestCase):
    def write_recovery_run(
        self,
        root: Path,
        run_dir: Path,
        events: list[dict[str, object]],
    ) -> None:
        config = {
            "scope": {"editable": ["editable.txt"]},
            "baseline": {"metric_median": 1.0, "params": {}},
        }
        encoded = metal_autoresearch.canonical_json(config)
        run_dir.mkdir()
        (run_dir / "run.json").write_bytes(encoded)
        (run_dir / "run.sha256").write_text(metal_autoresearch.sha256(encoded) + "\n")
        (run_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
        )
        (run_dir / "candidate-events.jsonl").touch()
        (run_dir / "snapshots").mkdir()
        metal_autoresearch.snapshot_paths(
            root, ["editable.txt"], run_dir / "snapshots" / "baseline"
        )

    def bytecode_local_contract_fixture(
        self,
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        config = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/bytecode_read_raf_cycle.template.json"
        )
        params = {
            name: str(value) for name, value in config["baseline_params"].items()
        }
        log_n = 26
        cutoff_log2 = 16
        repeats = 3
        cpu_prepare = [100, 110, 120]
        cpu_core = [340, 340, 340]
        cpu_host_fs = [10, 10, 10]
        metal_prepare = [10, 10, 10]
        metal_core = [80, 80, 80]
        metal_host_fs = [10, 10, 10]
        cpu = [
            prepare + core + host_fs
            for prepare, core, host_fs in zip(
                cpu_prepare, cpu_core, cpu_host_fs
            )
        ]
        metal = [
            prepare + core + host_fs
            for prepare, core, host_fs in zip(
                metal_prepare, metal_core, metal_host_fs
            )
        ]
        paired = [
            cpu_value / metal_value
            for cpu_value, metal_value in zip(cpu, metal)
        ]
        kernel_only = [
            (cpu_value - cpu_fs) / (metal_value - metal_fs)
            for cpu_value, cpu_fs, metal_value, metal_fs in zip(
                cpu, cpu_host_fs, metal, metal_host_fs
            )
        ]
        cpu_controls = [455, 465, 475]
        cpu_denominator_ratio = statistics.median(cpu_controls) / statistics.median(
            cpu
        )
        phase = {
            "counts": {
                "MetalBytecodeReadRafCycle::allocation_plan": 1,
                "MetalBytecodeReadRafCycle::cpu_tail": cutoff_log2,
                "MetalBytecodeReadRafCycle::dense_round": log_n - cutoff_log2 - 1,
                "MetalBytecodeReadRafCycle::first_bind": 1,
                "MetalBytecodeReadRafCycle::first_message": 1,
                "MetalBytecodeReadRafCycle::prepare": 1,
                "MetalBytecodeReadRafCycle::readback": 1,
            },
            "allocation": {
                "current_device_bytes": 100,
                "device_buffers": 17,
                "planned_device_bytes": 200,
                "recommended_device_bytes": 300,
            },
            "readback": {"bytes": 5 * (1 << cutoff_log2) * 16},
        }
        output = {
            "schema_version": 1,
            "kernel": "bytecode_read_raf_cycle",
            "guards": {
                "cpu_denominator_stable": True,
                "metal_backend_exercised": True,
                "exact_metal_schedule": True,
            },
            "metrics": {
                "hybrid_speedup": statistics.median(paired),
                "paired_speedups": paired,
                "paired_speedup_mad": statistics.median(
                    [abs(value - statistics.median(paired)) for value in paired]
                ),
                "kernel_only_hybrid_speedup": statistics.median(kernel_only),
                "kernel_only_paired_speedups": kernel_only,
                "cpu_member_ms_median": statistics.median(cpu) / 1e6,
                "metal_member_ms_median": statistics.median(metal) / 1e6,
                "cpu_member_ns_samples": cpu,
                "metal_member_ns_samples": metal,
                "cpu_no_resident_member_ns_samples": cpu_controls,
                "cpu_denominator_ratio": cpu_denominator_ratio,
                "cpu_core_ns_samples": cpu_core,
                "metal_core_ns_samples": metal_core,
                "cpu_round_ns_samples": [[1] * log_n for _ in range(repeats)],
                "metal_round_ns_samples": [[1] * log_n for _ in range(repeats)],
                "cpu_prepare_ns_samples": cpu_prepare,
                "metal_prepare_ns_samples": metal_prepare,
                "cpu_host_fs_ns_samples": cpu_host_fs,
                "metal_host_fs_ns_samples": metal_host_fs,
            },
            "phase_samples": [copy.deepcopy(phase) for _ in range(repeats)],
            "resources": {
                "gpu_seconds": sum(metal) / 1e9,
                "metal_hybrid_wall_seconds": sum(metal) / 1e9,
                "input_claim_precompute_ns": 1,
                "resident_upload_ns": 1,
                "resident_row_bytes": 40 * (1 << log_n),
            },
            "fingerprint": {
                "cpu_algebra": "q10",
                "entry_bytecode_index": 1,
                "log_n": log_n,
                "trace_elements": 1 << log_n,
                "seed": 1,
                "repeats": repeats,
                "message_threads": 256,
                "transition_threads": 128,
                "max_threadgroups": 8192,
                "cutoff_log2": cutoff_log2,
                "cutoff_elements": 1 << cutoff_log2,
                "trace_cutoff_log2": 18,
                "trace_cutoff_elements": 1 << 18,
                "orders": [
                    ["optimized", "metal"],
                    ["metal", "optimized"],
                    ["optimized", "metal"],
                ],
                "fixture": "address-diverse TraceBackend in a full 8192-row program and padded cycle domain",
                "fixture_program_rows": 1 << 13,
                "fixture_trace_rows": (1 << 13) - 1,
                "covers_high_ra_chunk": True,
                "fused_inc_fixture": "mixed rd and RAM signed deltas",
                "relation_variant": "full-program",
                "initial_claim": "independent direct cycle-domain sum",
                "primary_metric_includes_host_fs": True,
            },
        }
        return config, params, output

    def instruction_input_local_contract_fixture(
        self,
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        config = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        params = {
            name: str(value) for name, value in config["baseline_params"].items()
        }
        log_n = int(config["evaluator"]["env"]["JOLT_METAL_EVAL_LOG_N"])
        validation_log_n = int(
            config["evaluator"]["env"]["JOLT_METAL_EVAL_VALIDATE_LOG_N"]
        )
        repeats = int(config["evaluator"]["env"]["JOLT_METAL_EVAL_REPEATS"])
        seed = int(config["evaluator"]["env"]["JOLT_METAL_EVAL_SEED"])
        cutoff_log2 = int(params["JOLT_METAL_INSTRUCTION_INPUT_CUTOFF_LOG2"])
        trace_cutoff_log2 = int(
            params["JOLT_METAL_INSTRUCTION_INPUT_TRACE_CUTOFF_LOG2"]
        )
        rows = 1 << log_n
        cutoff = 1 << cutoff_log2
        cpu_samples = [500 + 100 * index for index in range(repeats)]
        hybrid_samples = [100 + 20 * index for index in range(repeats)]
        reset_samples = [10 + 10 * index for index in range(repeats)]
        resident_samples = [
            hybrid - reset
            for hybrid, reset in zip(hybrid_samples, reset_samples)
        ]
        gpu_wall_samples = [50 + 10 * index for index in range(repeats)]
        gpu_active_samples = [40 + 10 * index for index in range(repeats)]
        host_samples = [10 + index for index in range(repeats)]
        readback_samples = [5 + index for index in range(repeats)]
        tail_samples = [10 + index for index in range(repeats)]
        paired = [
            cpu / hybrid for cpu, hybrid in zip(cpu_samples, hybrid_samples)
        ]
        resident_paired = [
            cpu / resident for cpu, resident in zip(cpu_samples, resident_samples)
        ]
        sequence_bytes = metal_autoresearch.instruction_input_sequence_storage_bytes(
            log_n
        )
        cpu_rows_bytes = 48 * rows
        resident_rows_bytes = 160 * rows
        persistent_bytes = cpu_rows_bytes + resident_rows_bytes + sequence_bytes
        cpu_first_dense_bytes = 8 * (rows // 2) * 16
        hybrid_tail_bytes = 2 * 8 * cutoff * 16
        protocol_seeds = [
            seed ^ ((0x9E3779B97F4A7C15 * (index + 1)) & ((1 << 64) - 1))
            for index in range(repeats)
        ]
        guards = {
            name: True
            for name in (
                "exact_four_sample_q_evals",
                "exact_round_polynomials",
                "exact_host_fiat_shamir_challenges",
                "exact_round_schedule",
                "exact_cutoff_tables",
                "exact_final_eight_claims",
                "exact_final_sumcheck_claim",
                "exact_transcript_state",
                "exact_derived_eq_cycle",
                "exact_final_relation",
                "actual_optimized_cpu_validation_parity",
                "protocol_retarget_reuses_cpu_rows",
                "production_trace_cutoff_admits_target",
                "raw_timing_relations",
                "resident_rows_stable_across_reset",
                "static_device_buffer_identities_stable",
                "exactly_one_dense_readback",
                "host_readback_preallocated_before_primary_timer",
                "distinct_protocol_tapes",
                "round_device_buffer_allocations_zero",
                "host_fiat_shamir",
                "cpu_tail_uses_exact_four_samples",
                "all_exact",
            )
        }
        output = {
            "schema": "instruction_input_v1",
            "schema_version": 1,
            "kernel": "instruction_input",
            "metrics": {
                "hybrid_speedup": statistics.median(paired),
                "resident_speedup": statistics.median(resident_paired),
                "paired_hybrid_speedups": paired,
                "paired_resident_speedups": resident_paired,
                "cpu_ns_samples": cpu_samples,
                "hybrid_ns_samples": hybrid_samples,
                "resident_ns_samples": resident_samples,
                "cpu_million_rows_per_second": rows
                / (statistics.median(cpu_samples) / 1e9)
                / 1e6,
                "hybrid_million_rows_per_second": rows
                / (statistics.median(hybrid_samples) / 1e9)
                / 1e6,
            },
            "timings": {
                "workload_and_source_preparation_seconds": 1.0,
                "sequence_upload_and_storage_preparation_seconds": 2.0,
                "cpu_median_seconds": statistics.median(cpu_samples) / 1e9,
                "hybrid_median_seconds": statistics.median(hybrid_samples) / 1e9,
                "resident_median_seconds": statistics.median(resident_samples) / 1e9,
                "sequence_reset_median_seconds": statistics.median(reset_samples)
                / 1e9,
                "gpu_dispatch_wall_median_seconds": statistics.median(
                    gpu_wall_samples
                )
                / 1e9,
                "host_round_median_seconds": statistics.median(host_samples) / 1e9,
                "readback_median_seconds": statistics.median(readback_samples) / 1e9,
                "cpu_tail_median_seconds": statistics.median(tail_samples) / 1e9,
                "gpu_active_total_seconds": sum(gpu_active_samples) / 1e9,
                "sequence_reset_ns_samples": reset_samples,
                "gpu_dispatch_wall_ns_samples": gpu_wall_samples,
                "host_round_ns_samples": host_samples,
                "readback_ns_samples": readback_samples,
                "cpu_tail_ns_samples": tail_samples,
                "gpu_active_ns_samples": gpu_active_samples,
                "repeats": repeats,
            },
            "guards": guards,
            "resources": {
                "gpu_seconds": sum(gpu_active_samples) / 1e9,
                "cpu_native_rows_bytes": cpu_rows_bytes,
                "resident_stage1_rows_bytes": resident_rows_bytes,
                "sequence_owned_working_storage_bytes": sequence_bytes,
                "persistent_modeled_bytes_during_primary_trials": persistent_bytes,
                "cpu_first_dense_table_bytes": cpu_first_dense_bytes,
                "cpu_trial_peak_modeled_bytes": persistent_bytes
                + cpu_first_dense_bytes,
                "hybrid_readback_plus_tail_table_capacity_bytes": hybrid_tail_bytes,
                "hybrid_trial_peak_modeled_bytes": persistent_bytes
                + hybrid_tail_bytes,
                "resident_source_host_copy_bytes_dropped_before_primary_trials": resident_rows_bytes,
                "setup_peak_increment_from_resident_source_copy_bytes": resident_rows_bytes,
                "cutoff_readback_bytes": 8 * cutoff * 16,
                "unified_memory_no_per_round_row_upload": True,
                "sequence_owned_storage_includes_dense_ping_pong_weights_and_reductions": True,
            },
            "workload": {
                "log_n": log_n,
                "rows": rows,
                "validation_log_n": validation_log_n,
                "tables": 8,
                "samples_per_round": 4,
                "descriptor_fields_returned_by_gpu": 3,
                "cpu_native_row_bytes": 48,
                "resident_stage1_row_bytes": 160,
                "cutoff_log2": cutoff_log2,
                "cutoff_elements": cutoff,
                "trace_cutoff_log2": trace_cutoff_log2,
                "trace_cutoff_elements": 1 << trace_cutoff_log2,
                "native_message_threads": int(
                    params["JOLT_METAL_INSTRUCTION_INPUT_NATIVE_MESSAGE_THREADS"]
                ),
                "native_transition_threads": int(
                    params["JOLT_METAL_INSTRUCTION_INPUT_NATIVE_TRANSITION_THREADS"]
                ),
                "dense_transition_threads": int(
                    params["JOLT_METAL_INSTRUCTION_INPUT_DENSE_TRANSITION_THREADS"]
                ),
                "host_fiat_shamir": True,
                "primary_timing": "resident sequence reset plus Metal rounds, host Fiat-Shamir, one dense readback, and exact four-sample CPU tail",
                "workload_preparation_in_primary_metric": False,
                "sequence_preparation_in_primary_metric": False,
                "host_readback_allocation_in_primary_metric": False,
                "protocol_tape_preparation_in_primary_metric": False,
                "protocol_tapes_per_process": repeats,
                "protocol_tape_derivation": "base_seed xor ((repeat + 1) * 0x9e3779b97f4a7c15 modulo 2^64)",
                "cpu_trials_run_while_resident_metal_sequence_is_allocated": True,
                "cpu_control": "standalone row-stride and arithmetic mirror of OptimizedInstructionInputKernel",
                "metal_control": "public InstructionInputSequence over resident SpartanOuterUniskipRow storage",
            },
            "pipelines": {
                "native_message_execution_width": 32,
                "native_message_max_threads": 1024,
                "native_transition_execution_width": 32,
                "native_transition_max_threads": 1024,
                "dense_transition_execution_width": 32,
                "dense_transition_max_threads": 1024,
            },
            "fingerprint": {
                "device": "fixture Metal device",
                "max_buffer_length": 1 << 36,
                "recommended_max_working_set_size": 1 << 40,
                "current_allocated_size": 0,
                "cpu_threads": 8,
                "log_n": log_n,
                "validation_log_n": validation_log_n,
                "repeats": repeats,
                "seed": seed,
                "cutoff_log2": cutoff_log2,
                "trace_cutoff_log2": trace_cutoff_log2,
                "native_message_threads": int(
                    params["JOLT_METAL_INSTRUCTION_INPUT_NATIVE_MESSAGE_THREADS"]
                ),
                "native_transition_threads": int(
                    params["JOLT_METAL_INSTRUCTION_INPUT_NATIVE_TRANSITION_THREADS"]
                ),
                "dense_transition_threads": int(
                    params["JOLT_METAL_INSTRUCTION_INPUT_DENSE_TRANSITION_THREADS"]
                ),
                "orders": [
                    ["cpu", "metal"] if index % 2 == 0 else ["metal", "cpu"]
                    for index in range(repeats)
                ],
                "protocol_seeds": protocol_seeds,
                "protocol_transcript_states": [
                    [index + 1] * 32 for index in range(repeats)
                ],
            },
        }
        return config, params, output

    def production_bytecode_member_fixture(
        self, backend: str, member_ns: int, log_n: int = 26, cutoff_log2: int = 16
    ) -> dict[str, object]:
        prepare_ns = 1
        rounds_ns = [1] * log_n
        finish_ns = 1
        output_claims_ns = member_ns - prepare_ns - sum(rounds_ns) - finish_ns
        self.assertGreater(output_claims_ns, 0)
        metal_phases = {
            "prepare": 0,
            "allocation_plan": 0,
            "first_message": 0,
            "first_bind": 0,
            "dense_round": 0,
            "readback": 0,
            "cpu_tail": 0,
            "invalid_round": 0,
        }
        resource = None
        if backend == "metal":
            metal_phases.update(
                {
                    "prepare": 1,
                    "allocation_plan": 1,
                    "first_message": 1,
                    "first_bind": 1,
                    "dense_round": log_n - cutoff_log2 - 1,
                    "readback": 1,
                    "cpu_tail": cutoff_log2,
                }
            )
            resource = {
                "allocation": {
                    "current_device_bytes": 100,
                    "device_buffers": 17,
                    "planned_device_bytes": 200,
                    "recommended_device_bytes": 300,
                },
                "readback_bytes": 5 * (1 << cutoff_log2) * 16,
            }
        return {
            "prepare_ns": prepare_ns,
            "rounds_ns": rounds_ns,
            "rounds_total_ns": sum(rounds_ns),
            "finish_ns": finish_ns,
            "output_claims_ns": output_claims_ns,
            "member_ns": member_ns,
            "outer_counts": {
                "prepare": 1,
                "prove_round": log_n,
                "finish_rounds": 1,
                "output_claims": 1,
            },
            "metal_counts": metal_phases,
            "resource_observation": resource,
        }

    def production_instruction_input_member_fixture(
        self, backend: str, member_ns: int, log_n: int = 26, cutoff_log2: int = 16
    ) -> dict[str, object]:
        prepare_ns = 1
        rounds_ns = [1] * log_n
        finish_ns = 1
        output_claims_ns = member_ns - prepare_ns - sum(rounds_ns) - finish_ns
        self.assertGreater(output_claims_ns, 0)
        metal_phases = {
            "storage_prepare": 0,
            "allocation_plan": 0,
            "prepare": 0,
            "first_message": 0,
            "first_bind": 0,
            "dense_round": 0,
            "readback": 0,
            "cpu_tail": 0,
        }
        resource = None
        if backend == "metal":
            metal_phases.update(
                {
                    "storage_prepare": 1,
                    "allocation_plan": 1,
                    "prepare": 1,
                    "first_message": 1,
                    "first_bind": 1,
                    "dense_round": log_n - cutoff_log2 - 1,
                    "readback": 1,
                    "cpu_tail": cutoff_log2,
                }
            )
            resource = {
                "allocation": {
                    "current_device_bytes": 160 * (1 << log_n),
                    "device_buffers": 6,
                    "planned_device_bytes": metal_autoresearch.instruction_input_sequence_storage_bytes(
                        log_n
                    ),
                    "recommended_device_bytes": 160
                    * (1 << log_n)
                    + metal_autoresearch.instruction_input_sequence_storage_bytes(
                        log_n
                    ),
                },
                "host_tail_bytes": 8 * (1 << cutoff_log2) * 16,
                "resident_rows_reused": True,
                "round_device_buffer_allocations": 0,
                "readback_bytes": 8 * (1 << cutoff_log2) * 16,
            }
        return {
            "prepare_ns": prepare_ns,
            "rounds_ns": rounds_ns,
            "rounds_total_ns": sum(rounds_ns),
            "finish_ns": finish_ns,
            "output_claims_ns": output_claims_ns,
            "member_ns": member_ns,
            "outer_counts": {
                "prepare": 1,
                "prove_round": log_n,
                "finish_rounds": 1,
                "output_claims": 1,
            },
            "metal_counts": metal_phases,
            "resource_observation": resource,
        }

    def production_instruction_input_row_lifecycle_fixture(
        self, backend: str, log_n: int = 26
    ) -> dict[str, object]:
        if backend == "optimized":
            return {
                "kind": "optimized_cpu",
                "rows": 1 << log_n,
                "row_bytes": 48,
                "prepare_storage_id": 101,
                "stage3_storage_id": 101,
            }
        return {
            "kind": "metal_resident",
            "rows": 1 << log_n,
            "row_bytes": 160,
            "prepare_storage_id": 202,
            "stage1_storage_id": 202,
            "stage3_storage_id": 202,
        }

    def production_instruction_input_result_fixture(
        self,
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        config = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        params = {
            name: str(value) for name, value in config["baseline_params"].items()
        }
        gate = config["final_validation"]["production_gate"]
        pairs = int(gate["minimum_pairs"])
        log_n = 26
        cutoff_log2 = int(
            params["JOLT_METAL_INSTRUCTION_INPUT_CUTOFF_LOG2"]
        )
        cpu_piop_ns = 200
        metal_piop_ns = 100
        cpu_member_ns = 500
        metal_member_ns = 100
        local_speedup = cpu_member_ns / metal_member_ns
        orders = [
            ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
            for index in range(pairs)
        ]
        result = {
            "schema_version": 5,
            "kernel": "akita_piop",
            "local_kernel": "InstructionInput",
            "local_metric": {
                "metric": "instruction_input_kernel_service_speedup",
                "paired_metric": "paired_instruction_input_kernel_service_speedups",
            },
            "run_class": {"mode": "production", "acceptance_eligible": True},
            "guards": {name: True for name in gate["required_guards"]},
            "metrics": {
                "instruction_input_kernel_service_speedup": local_speedup,
                "piop_speedup": cpu_piop_ns / metal_piop_ns,
                "paired_speedups": [cpu_piop_ns / metal_piop_ns] * pairs,
                "cpu_piop_ms_samples": [cpu_piop_ns / 1e6] * pairs,
                "metal_piop_ms_samples": [metal_piop_ns / 1e6] * pairs,
                "paired_instruction_input_kernel_service_speedups": [local_speedup]
                * pairs,
                "cpu_instruction_input_kernel_service_ms_samples": [cpu_member_ns / 1e6]
                * pairs,
                "metal_instruction_input_kernel_service_ms_samples": [metal_member_ns / 1e6]
                * pairs,
                "instruction_input_kernel_service_decision": {
                    "clears": True,
                    "minimum_speedup": float(gate["minimum_local_speedup"]),
                    "minimum_pairs": pairs,
                    "median_speedup": local_speedup,
                    "optimized_first_median_speedup": local_speedup,
                    "metal_first_median_speedup": local_speedup,
                    "clears_order_strata": True,
                },
            },
            "pairs": [
                {
                    "index": index + 1,
                    "order": order,
                    "arms": {
                        "optimized": {
                            "piop_ns": cpu_piop_ns,
                            "instruction_input": self.production_instruction_input_member_fixture(
                                "optimized", cpu_member_ns, log_n, cutoff_log2
                            ),
                            "instruction_input_row_lifecycle": self.production_instruction_input_row_lifecycle_fixture(
                                "optimized", log_n
                            ),
                        },
                        "metal": {
                            "piop_ns": metal_piop_ns,
                            "instruction_input": self.production_instruction_input_member_fixture(
                                "metal", metal_member_ns, log_n, cutoff_log2
                            ),
                            "instruction_input_row_lifecycle": self.production_instruction_input_row_lifecycle_fixture(
                                "metal", log_n
                            ),
                        },
                    },
                }
                for index, order in enumerate(orders)
            ],
            "resources": {"metal_piop_seconds": pairs * metal_piop_ns / 1e9},
            "fingerprint": {
                "git_revision": "abc",
                "worktree_dirty": False,
                "local_kernel": "InstructionInput",
                "log_n": log_n,
                "instruction_input_metal_native_message_threads": int(
                    params["JOLT_METAL_INSTRUCTION_INPUT_NATIVE_MESSAGE_THREADS"]
                ),
                "instruction_input_metal_native_transition_threads": int(
                    params["JOLT_METAL_INSTRUCTION_INPUT_NATIVE_TRANSITION_THREADS"]
                ),
                "instruction_input_metal_dense_transition_threads": int(
                    params["JOLT_METAL_INSTRUCTION_INPUT_DENSE_TRANSITION_THREADS"]
                ),
                "instruction_input_metal_cutoff_log2": cutoff_log2,
                "instruction_input_metal_trace_cutoff_log2": int(
                    params["JOLT_METAL_INSTRUCTION_INPUT_TRACE_CUTOFF_LOG2"]
                ),
                "orders": orders,
                "span": "jolt_prover::piop",
                "workload": "fibonacci",
            },
        }
        return config, params, result

    def test_schema_five_parser_requires_one_result_record(self) -> None:
        record = '{"schema_version": 5, "kernel": "akita_piop"}'
        self.assertEqual(
            metal_autoresearch.parse_unique_schema_result(record, 5)["kernel"],
            "akita_piop",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            metal_autoresearch.parse_unique_schema_result(f"{record}\n{record}", 5)

    def test_local_evaluator_requires_one_result_record(self) -> None:
        config = {
            "kernel": "test",
            "metric": {"name": "score"},
            "evaluator": {"command": ["unused"], "timeout_seconds": 30},
        }
        record = json.dumps(
            {
                "schema_version": 1,
                "kernel": "test",
                "metrics": {"score": 1.0},
            }
        )
        completed = SimpleNamespace(
            returncode=0,
            stdout=f"{record}\n{record}\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                metal_autoresearch.subprocess, "run", return_value=completed
            ):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    metal_autoresearch.run_evaluator(
                        Path(directory), config, {}, Path(directory), "duplicate"
                    )

    def test_local_evaluator_scrubs_ambient_metal_environment(self) -> None:
        config = {
            "kernel": "test",
            "metric": {"name": "score"},
            "evaluator": {
                "command": ["unused"],
                "timeout_seconds": 30,
                "env": {"JOLT_METAL_DECLARED": "contract"},
            },
        }

        def completed_run(*_args: object, **kwargs: object) -> SimpleNamespace:
            environment = kwargs["env"]
            output = {
                "schema_version": 1,
                "kernel": "test",
                "metrics": {"score": 1.0},
                "ambient_visible": "JOLT_METAL_UNDECLARED" in environment,
                "ambient_autoresearch_visible": "JOLT_AUTORESEARCH_UNDECLARED"
                in environment,
                "declared": environment.get("JOLT_METAL_DECLARED"),
                "parameter": environment.get("JOLT_METAL_PARAMETER"),
            }
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(output) + "\n",
                stderr="",
            )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {
                    "JOLT_METAL_UNDECLARED": "forged",
                    "JOLT_AUTORESEARCH_UNDECLARED": "forged",
                },
                clear=False,
            ):
                with mock.patch.object(
                    metal_autoresearch.subprocess, "run", side_effect=completed_run
                ):
                    output, _ = metal_autoresearch.run_evaluator(
                        Path(directory),
                        config,
                        {"JOLT_METAL_PARAMETER": "candidate"},
                        Path(directory),
                        "environment",
                    )
        self.assertFalse(output["ambient_visible"])
        self.assertFalse(output["ambient_autoresearch_visible"])
        self.assertEqual(output["declared"], "contract")
        self.assertEqual(output["parameter"], "candidate")

    def test_bytecode_local_result_rejects_fingerprint_drift(self) -> None:
        config = {
            "kernel": "bytecode_read_raf_cycle",
            "metric": {"name": "hybrid_speedup"},
            "evaluator": {
                "command": ["unused"],
                "timeout_seconds": 30,
                "env": {
                    "JOLT_METAL_EVAL_LOG_N": "26",
                    "JOLT_METAL_EVAL_REPEATS": "3",
                    "JOLT_METAL_EVAL_SEED": "1",
                },
                "result_contract": "bytecode_read_raf_cycle_v1",
            },
        }
        params = {
            "JOLT_METAL_BYTECODE_MESSAGE_THREADS": "256",
            "JOLT_METAL_BYTECODE_TRANSITION_THREADS": "128",
            "JOLT_METAL_BYTECODE_MAX_THREADGROUPS": "8192",
            "JOLT_METAL_BYTECODE_CUTOFF_LOG2": "16",
            "JOLT_METAL_BYTECODE_TRACE_CUTOFF_LOG2": "18",
        }
        output = {
            "schema_version": 1,
            "kernel": "bytecode_read_raf_cycle",
            "metrics": {
                "hybrid_speedup": 4.5,
                "paired_speedups": [4.5, 4.5, 4.5],
            },
            "fingerprint": {
                "log_n": 25,
                "trace_elements": 1 << 25,
                "seed": 1,
                "repeats": 3,
                "message_threads": 256,
                "transition_threads": 128,
                "max_threadgroups": 8192,
                "cutoff_log2": 16,
                "cutoff_elements": 1 << 16,
                "trace_cutoff_log2": 18,
                "trace_cutoff_elements": 1 << 18,
                "orders": [
                    ["optimized", "metal"],
                    ["metal", "optimized"],
                    ["optimized", "metal"],
                ],
                "relation_variant": "full-program",
                "initial_claim": "independent direct cycle-domain sum",
                "primary_metric_includes_host_fs": True,
            },
        }
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(output) + "\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                metal_autoresearch.subprocess, "run", return_value=completed
            ):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    metal_autoresearch.run_evaluator(
                        Path(directory), config, params, Path(directory), "fingerprint"
                    )

    def test_bytecode_local_result_accepts_closed_contract(self) -> None:
        config, params, output = self.bytecode_local_contract_fixture()
        metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_bytecode_local_result_rejects_unsubstantiated_phase_schedule(
        self,
    ) -> None:
        config, params, output = self.bytecode_local_contract_fixture()
        output["phase_samples"][0]["counts"][
            "MetalBytecodeReadRafCycle::dense_round"
        ] = 0

        with self.assertRaisesRegex(ValueError, "phase schedule"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_bytecode_local_result_rejects_unreconciled_member_timing(self) -> None:
        config, params, output = self.bytecode_local_contract_fixture()
        output["metrics"]["metal_host_fs_ns_samples"][0] += 1

        with self.assertRaisesRegex(ValueError, "member timing"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_bytecode_local_result_recomputes_cpu_denominator(self) -> None:
        config, params, output = self.bytecode_local_contract_fixture()
        output["metrics"]["cpu_no_resident_member_ns_samples"] = [900, 900, 900]

        with self.assertRaisesRegex(ValueError, "denominator"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_bytecode_local_result_requires_positive_core_residual(self) -> None:
        config, params, output = self.bytecode_local_contract_fixture()
        output["metrics"]["cpu_core_ns_samples"][0] = 26

        with self.assertRaisesRegex(ValueError, "core timing"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_bytecode_local_result_reconciles_reported_resources(self) -> None:
        config, params, output = self.bytecode_local_contract_fixture()
        output["resources"]["gpu_seconds"] *= 2

        with self.assertRaisesRegex(ValueError, "resource"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_bytecode_template_requires_its_closed_result_contract(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/bytecode_read_raf_cycle.template.json"
        )
        del template["evaluator"]["result_contract"]

        with self.assertRaisesRegex(ValueError, "result contract"):
            metal_autoresearch.validate_template(template)

    def test_instruction_input_local_result_accepts_closed_contract(self) -> None:
        config, params, output = self.instruction_input_local_contract_fixture()
        metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_instruction_input_local_result_rejects_schema_extensions(self) -> None:
        config, params, output = self.instruction_input_local_contract_fixture()
        output["undeclared"] = True
        with self.assertRaisesRegex(ValueError, "top-level schema"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

        config, params, output = self.instruction_input_local_contract_fixture()
        output["metrics"]["undeclared"] = 1
        with self.assertRaisesRegex(ValueError, "metric record"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_instruction_input_local_result_recomputes_reported_values(self) -> None:
        mutations = (
            (
                "raw CPU sample",
                lambda output: output["metrics"]["cpu_ns_samples"].__setitem__(
                    0, 501
                ),
                "paired_hybrid_speedups",
            ),
            (
                "paired speedup",
                lambda output: output["metrics"]["paired_hybrid_speedups"].__setitem__(
                    0, 99.0
                ),
                "paired_hybrid_speedups",
            ),
            (
                "median",
                lambda output: output["metrics"].__setitem__("hybrid_speedup", 99.0),
                "hybrid_speedup",
            ),
            (
                "timing median",
                lambda output: output["timings"].__setitem__(
                    "gpu_dispatch_wall_median_seconds", 99.0
                ),
                "gpu_dispatch_wall_median_seconds",
            ),
            (
                "GPU total",
                lambda output: output["timings"].__setitem__(
                    "gpu_active_total_seconds", 99.0
                ),
                "GPU-active total",
            ),
            (
                "GPU resource",
                lambda output: output["resources"].__setitem__(
                    "gpu_seconds", 99.0
                ),
                "GPU resource timing",
            ),
            (
                "resource total",
                lambda output: output["resources"].__setitem__(
                    "persistent_modeled_bytes_during_primary_trials", 1
                ),
                "resource",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                config, params, output = self.instruction_input_local_contract_fixture()
                mutate(output)
                with self.assertRaisesRegex(ValueError, message):
                    metal_autoresearch.validate_local_result_contract(
                        config, output, params
                    )

    def test_instruction_input_local_result_rejects_protocol_tampering(self) -> None:
        config, params, output = self.instruction_input_local_contract_fixture()
        output["fingerprint"]["repeats"] = 3
        with self.assertRaisesRegex(ValueError, "fingerprint does not match repeats"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

        config, params, output = self.instruction_input_local_contract_fixture()
        output["fingerprint"]["protocol_seeds"][1] = output["fingerprint"][
            "protocol_seeds"
        ][0]
        with self.assertRaisesRegex(ValueError, "protocol tapes"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

        config, params, output = self.instruction_input_local_contract_fixture()
        output["fingerprint"]["protocol_transcript_states"][1] = output[
            "fingerprint"
        ]["protocol_transcript_states"][0]
        with self.assertRaisesRegex(ValueError, "transcript tapes"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

        config, params, output = self.instruction_input_local_contract_fixture()
        output["guards"]["distinct_protocol_tapes"] = False
        with self.assertRaisesRegex(ValueError, "correctness guard"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_instruction_input_local_result_enforces_gpu_time_budget(self) -> None:
        for name, active, gpu_wall, hybrid, message in (
            ("zero active", 0, 50, 100, "samples"),
            ("active exceeds wall", 51, 50, 100, "GPU timing"),
            ("wall exceeds hybrid", 50, 101, 100, "GPU timing"),
        ):
            with self.subTest(name=name):
                config, params, output = self.instruction_input_local_contract_fixture()
                output["timings"]["gpu_active_ns_samples"][0] = active
                output["timings"]["gpu_dispatch_wall_ns_samples"][0] = gpu_wall
                output["metrics"]["hybrid_ns_samples"][0] = hybrid
                with self.assertRaisesRegex(ValueError, message):
                    metal_autoresearch.validate_local_result_contract(
                        config, output, params
                    )

        config, params, output = self.instruction_input_local_contract_fixture()
        output["timings"]["host_round_ns_samples"][0] = 100
        with self.assertRaisesRegex(ValueError, "component timings"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_instruction_input_template_requires_precise_allocation_guard(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        precise_guard = "instruction_input_no_round_device_buffer_allocations"
        old_guard = "instruction_input_no_round_allocations"
        descriptor_guards = metal_autoresearch.PRODUCTION_LOCAL_KERNELS[
            "InstructionInput"
        ]["required_guards"]
        self.assertIn(precise_guard, descriptor_guards)
        self.assertNotIn(old_guard, descriptor_guards)
        metal_autoresearch.validate_template(template)

        template["final_validation"]["production_gate"]["required_guards"].remove(
            precise_guard
        )
        template["final_validation"]["production_gate"]["required_guards"].append(
            old_guard
        )
        with self.assertRaisesRegex(ValueError, "mandatory local-kernel guards"):
            metal_autoresearch.validate_template(template)

    def test_instruction_input_template_rejects_inert_validation_claims(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        template["final_validation"]["fresh_processes"] = 2
        with self.assertRaisesRegex(ValueError, "inert checks"):
            metal_autoresearch.validate_template(template)

    def test_instruction_input_template_requires_shader_only_scope(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        template["scope"]["editable"].append(
            "crates/jolt-kernels/src/metal/solinas/instruction_ra_sequence.metal"
        )
        with self.assertRaisesRegex(ValueError, "shader-only"):
            metal_autoresearch.validate_template(template)

        template = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        template["evaluator"]["env"]["JOLT_METAL_EVAL_REPEATS"] = "3"
        with self.assertRaisesRegex(ValueError, "at least five odd"):
            metal_autoresearch.validate_template(template)

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

    def test_strict_run_loads_a_genuine_kept_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            source = root / "editable.txt"
            source.write_text("baseline")
            run_dir = temporary / "run"
            run_dir.mkdir()
            snapshots = run_dir / "snapshots"
            snapshots.mkdir()
            metal_autoresearch.snapshot_paths(
                root, ["editable.txt"], snapshots / "baseline"
            )
            baseline_digest = metal_autoresearch.path_digest(
                snapshots / "baseline", ["editable.txt"]
            )
            source.write_text("accepted")
            metal_autoresearch.snapshot_paths(
                root, ["editable.txt"], snapshots / "trial-001"
            )
            params = {"threads": "256"}
            source_digest = metal_autoresearch.path_digest(
                snapshots / "trial-001", ["editable.txt"]
            )
            event = {
                "schema_version": 1,
                "index": 1,
                "trial_id": "trial-001",
                "parent_id": "baseline",
                "candidate_revision": metal_autoresearch.sha256(
                    metal_autoresearch.canonical_json(
                        {"source": source_digest, "params": params}
                    )
                ),
                "proposal_summary": "allowed candidate",
                "candidate_id": None,
                "candidate_manifest_sha256": None,
                "params": params,
                "started_at": "2026-08-04T00:00:00Z",
                "elapsed_seconds": 1.0,
                "metric_value": 2.0,
                "measurements": [2.0],
                "guards": {},
                "resources": {"gpu_seconds": 0.0},
                "verdict": "keep",
                "reason": "improves",
            }
            config = {
                "baseline": {
                    "metric_median": 1.0,
                    "params": {"threads": "128"},
                },
                "search_space": {"threads": ["128", "256"]},
                "scope": {"editable": ["editable.txt"]},
                "fingerprint": {"editable_paths_sha256": baseline_digest},
            }
            encoded = metal_autoresearch.canonical_json(config)
            (run_dir / "run.json").write_bytes(encoded)
            (run_dir / "run.sha256").write_text(
                metal_autoresearch.sha256(encoded) + "\n"
            )
            (run_dir / "events.jsonl").write_text(json.dumps(event) + "\n")

            _, events = metal_autoresearch.load_run(run_dir)
            self.assertEqual(events, [event])

    def test_init_rejects_an_edit_racing_the_baseline_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            source = root / "editable.txt"
            source.write_text("baseline")
            run_dir = temporary / "run"
            template = {
                "scope": {"editable": ["editable.txt"], "frozen": []},
                "evaluator": {"frozen_paths": []},
                "portfolio_contract": "portfolio.json",
                "baseline_params": {},
                "search_space": {},
                "baseline_repeats": 3,
                "budget": {"max_seconds": 30, "max_gpu_seconds": 30},
                "metric": {"name": "score", "minimum_relative_improvement": 0.01},
                "guards": {"required_true": []},
            }
            original_snapshot = metal_autoresearch.snapshot_paths

            def racing_snapshot(
                snapshot_root: Path, paths: list[str], destination: Path
            ) -> None:
                original_snapshot(snapshot_root, paths, destination)
                if destination.name == "baseline":
                    source.write_text("raced")

            with mock.patch.object(
                metal_autoresearch,
                "read_json",
                side_effect=[template, {}],
            ), mock.patch.object(
                metal_autoresearch, "validate_template"
            ), mock.patch.object(
                metal_autoresearch, "validate_goal_contract"
            ), mock.patch.object(
                metal_autoresearch,
                "outside_editable_worktree_digest",
                return_value="outside",
            ), mock.patch.object(
                metal_autoresearch,
                "snapshot_paths",
                side_effect=racing_snapshot,
            ), mock.patch.object(
                metal_autoresearch,
                "run_evaluator",
                side_effect=AssertionError("baseline evaluator launched"),
            ) as evaluator:
                with self.assertRaisesRegex(ValueError, "baseline snapshot changed"):
                    metal_autoresearch.command_init(
                        SimpleNamespace(
                            root=root,
                            template=temporary / "template.json",
                            run_dir=run_dir,
                        )
                    )
                evaluator.assert_not_called()

    def test_trial_rejects_an_edit_racing_the_kept_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            source = root / "editable.txt"
            source.write_text("baseline")
            run_dir = temporary / "run"
            run_dir.mkdir()
            (run_dir / "logs").mkdir()
            snapshots = run_dir / "snapshots"
            snapshots.mkdir()
            metal_autoresearch.snapshot_paths(
                root, ["editable.txt"], snapshots / "baseline"
            )
            baseline_digest = metal_autoresearch.path_digest(
                snapshots / "baseline", ["editable.txt"]
            )
            config = {
                "base_revision": "base",
                "baseline": {
                    "metric_median": 1.0,
                    "params": {},
                    "elapsed_seconds": 0.0,
                    "gpu_seconds": 0.0,
                },
                "search_space": {},
                "scope": {"editable": ["editable.txt"], "frozen": []},
                "fingerprint": {
                    "editable_paths_sha256": baseline_digest,
                    "frozen_paths_sha256": metal_autoresearch.path_digest(root, []),
                    "outside_editable_worktree_sha256": "outside",
                },
                "budget": {
                    "max_trials": 2,
                    "max_seconds": 30,
                    "max_gpu_seconds": 30,
                },
                "candidate_repeats": 1,
                "metric": {
                    "name": "score",
                    "direction": "max",
                    "promotion_relative_threshold": 0.01,
                },
                "guards": {"required_true": []},
            }
            encoded = metal_autoresearch.canonical_json(config)
            (run_dir / "run.json").write_bytes(encoded)
            (run_dir / "run.sha256").write_text(
                metal_autoresearch.sha256(encoded) + "\n"
            )
            (run_dir / "events.jsonl").touch()
            (run_dir / "candidate-events.jsonl").touch()
            source.write_text("candidate")
            original_snapshot = metal_autoresearch.snapshot_paths

            def racing_snapshot(
                snapshot_root: Path, paths: list[str], destination: Path
            ) -> None:
                if destination.name == "trial-001":
                    source.write_text("raced")
                original_snapshot(snapshot_root, paths, destination)

            output = {
                "metrics": {"score": 2.0},
                "resources": {"gpu_seconds": 0.0},
                "guards": {},
            }
            with mock.patch.object(
                metal_autoresearch, "git_head", return_value="base"
            ), mock.patch.object(
                metal_autoresearch,
                "outside_editable_worktree_digest",
                return_value="outside",
            ), mock.patch.object(
                metal_autoresearch, "run_evaluator", return_value=(output, 0.1)
            ), mock.patch.object(
                metal_autoresearch,
                "snapshot_paths",
                side_effect=racing_snapshot,
            ), redirect_stdout(io.StringIO()):
                status = metal_autoresearch.command_trial(
                    SimpleNamespace(
                        root=root,
                        run_dir=run_dir,
                        candidate_manifest=None,
                        summary="candidate",
                        param=[],
                    )
                )

            self.assertEqual(status, 2)
            _, events = metal_autoresearch.load_run(run_dir)
            self.assertEqual(events[0]["verdict"], "crash")
            self.assertEqual(source.read_text(), "baseline")

    def test_baseline_noise_uses_independent_evaluator_medians(self) -> None:
        median, relative_mad = metal_autoresearch.median_and_relative_mad(
            [2.938, 3.005, 3.035]
        )
        self.assertEqual(median, 3.005)
        self.assertGreater(relative_mad, 0.0)

    def test_accepted_parameters_follow_kept_lineage(self) -> None:
        config = {"baseline": {"params": {"width": "16", "threads": "128"}}}
        events = [
            {"verdict": "keep", "params": {"width": "32", "threads": "128"}},
            {"verdict": "discard", "params": {"width": "64", "threads": "128"}},
            {"verdict": "keep", "params": {"width": "32", "threads": "256"}},
        ]
        self.assertEqual(
            metal_autoresearch.accepted_parent_params(config, events),
            {"width": "32", "threads": "256"},
        )

    def test_append_event_writes_one_durable_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "events.jsonl"
            ledger.touch()
            event = {"index": 1, "trial_id": "trial-001"}
            metal_autoresearch.append_event(ledger, event)
            self.assertEqual(ledger.read_text().splitlines(), [json.dumps(event, sort_keys=True)])

    def test_evaluator_lock_is_reentrant_only_with_the_live_token(self) -> None:
        marker = metal_autoresearch.EVALUATOR_LOCK_HELD_ENV
        previous = os.environ.get(marker)
        previous_path = metal_autoresearch.EVALUATOR_LOCK_PATH
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "evaluator.lock"
            metal_autoresearch.EVALUATOR_LOCK_PATH = lock_path
            os.environ.pop(marker, None)
            try:
                with metal_autoresearch.evaluator_lock({"test": "outer"}):
                    token = os.environ[marker]
                    self.assertNotEqual(token, "1")
                    with metal_autoresearch.evaluator_lock({"test": "inner"}):
                        self.assertEqual(os.environ[marker], token)
                self.assertEqual(lock_path.read_text(), "")
            finally:
                metal_autoresearch.EVALUATOR_LOCK_PATH = previous_path
                if previous is None:
                    os.environ.pop(marker, None)
                else:
                    os.environ[marker] = previous

    def test_outside_editable_digest_ignores_only_declared_candidate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.mkdir(root / ".git")
            editable = ["editable.rs"]
            with mock.patch.object(
                metal_autoresearch.subprocess,
                "run",
                side_effect=[
                    SimpleNamespace(stdout=b"editable.rs\0frozen.rs\0"),
                    SimpleNamespace(stdout=b"new.txt\0"),
                ],
            ):
                (root / "editable.rs").write_text("candidate")
                (root / "frozen.rs").write_text("frozen")
                (root / "new.txt").write_text("new")
                first = metal_autoresearch.outside_editable_worktree_digest(root, editable)
            (root / "editable.rs").write_text("different candidate")
            with mock.patch.object(
                metal_autoresearch.subprocess,
                "run",
                side_effect=[
                    SimpleNamespace(stdout=b"editable.rs\0frozen.rs\0"),
                    SimpleNamespace(stdout=b"new.txt\0"),
                ],
            ):
                second = metal_autoresearch.outside_editable_worktree_digest(root, editable)
            self.assertEqual(first, second)

    def test_candidate_manifest_rejects_a_stale_parent(self) -> None:
        expected = {
            "run_sha256": "run",
            "base_revision": "base",
            "parent_id": "baseline",
            "frozen_paths_sha256": "frozen",
            "parent_editable_paths_sha256": "parent",
            "parent_params_sha256": "params",
            "evaluator_contract_sha256": "contract",
            "evaluator_paths_sha256": "evaluator",
        }
        manifest = {
            "schema_version": 1,
            "candidate_id": "candidate-001",
            "producer": "/root/kernel-agent",
            "summary": "fuse two passes",
            "candidate_editable_paths_sha256": "a" * 64,
            "analysis_sha256": "b" * 64,
            "patch_sha256": "c" * 64,
            **expected,
        }
        metal_autoresearch.validate_candidate_manifest(manifest, expected)
        manifest["parent_id"] = "trial-999"
        with self.assertRaisesRegex(ValueError, "stale parent_id"):
            metal_autoresearch.validate_candidate_manifest(manifest, expected)

    def test_recovery_discards_an_uncommitted_candidate_and_orphan_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            source = root / "editable.txt"
            source.write_text("baseline")
            run_dir = temporary / "run"
            self.write_recovery_run(root, run_dir, [])
            source.write_text("candidate")
            metal_autoresearch.snapshot_paths(
                root, ["editable.txt"], run_dir / "snapshots" / "trial-001"
            )
            (run_dir / "inflight.json").write_bytes(
                metal_autoresearch.canonical_json(
                    {
                        "trial_id": "trial-001",
                        "candidate_id": "candidate-001",
                        "candidate_manifest_sha256": "manifest",
                    }
                )
            )

            with redirect_stdout(io.StringIO()):
                metal_autoresearch.command_recover(
                    SimpleNamespace(root=root, run_dir=run_dir)
                )

            self.assertEqual(source.read_text(), "baseline")
            self.assertFalse((run_dir / "inflight.json").exists())
            quarantines = list((run_dir / "quarantine").iterdir())
            self.assertEqual(len(quarantines), 1)
            orphan = quarantines[0] / "orphan-accepted-snapshot" / "editable.txt"
            self.assertEqual(orphan.read_text(), "candidate")
            statuses = [
                json.loads(line)["status"]
                for line in (run_dir / "candidate-events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(statuses, ["queued", "rejected"])

    def test_recovery_repairs_a_committed_keep_candidate_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            source = root / "editable.txt"
            source.write_text("baseline")
            event = {
                "index": 1,
                "trial_id": "trial-001",
                "parent_id": "baseline",
                "verdict": "keep",
                "metric_value": 2.0,
                "params": {},
            }
            run_dir = temporary / "run"
            self.write_recovery_run(root, run_dir, [event])
            source.write_text("accepted")
            metal_autoresearch.snapshot_paths(
                root, ["editable.txt"], run_dir / "snapshots" / "trial-001"
            )
            (run_dir / "candidate-events.jsonl").write_text(
                json.dumps(
                    {"candidate_id": "candidate-001", "status": "queued"},
                    sort_keys=True,
                )
                + "\n"
            )
            source.write_text("interrupted-post-commit")
            (run_dir / "inflight.json").write_bytes(
                metal_autoresearch.canonical_json(
                    {
                        "trial_id": "trial-001",
                        "candidate_id": "candidate-001",
                        "candidate_manifest_sha256": "manifest",
                    }
                )
            )

            with redirect_stdout(io.StringIO()):
                metal_autoresearch.command_recover(
                    SimpleNamespace(root=root, run_dir=run_dir)
                )

            self.assertEqual(source.read_text(), "accepted")
            statuses = [
                json.loads(line)["status"]
                for line in (run_dir / "candidate-events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(statuses, ["queued", "accepted_parent"])

    def test_production_gate_requires_clean_five_pair_local_result(self) -> None:
        config = {
            "final_validation": {
                "production_gate": {
                    "metric": "instruction_ra_speedup",
                    "minimum_local_speedup": 7.0,
                    "minimum_log_n": 26,
                    "minimum_pairs": 5,
                    "require_alternating_orders": True,
                    "require_clean_worktree": True,
                    "workload": "fibonacci",
                    "required_guards": ["cpu_proofs_verified", "metal_proofs_verified"],
                    "expected_fingerprint": {
                        "instruction_ra_materialize_width": {
                            "parameter": "width",
                            "type": "int",
                        },
                        "instruction_ra_reuse_inverse": {
                            "parameter": "reuse",
                            "type": "bool01",
                        },
                    },
                }
            }
        }
        result = {
            "schema_version": 4,
            "kernel": "akita_piop",
            "guards": {"cpu_proofs_verified": True, "metal_proofs_verified": True},
            "metrics": {
                "instruction_ra_speedup": 7.5,
                "piop_speedup": 2.0,
                "paired_speedups": [2.0] * 5,
                "paired_instruction_ra_speedups": [7.5] * 5,
            },
            "fingerprint": {
                "git_revision": "abc",
                "worktree_dirty": False,
                "instruction_ra_materialize_width": 16,
                "instruction_ra_reuse_inverse": False,
                "log_n": 26,
                "orders": [
                    ["optimized", "metal"],
                    ["metal", "optimized"],
                    ["optimized", "metal"],
                    ["metal", "optimized"],
                    ["optimized", "metal"],
                ],
                "span": "jolt_prover::piop",
                "workload": "fibonacci",
            },
        }
        evidence = metal_autoresearch.validate_production_result(
            config, result, "abc", {"width": "16", "reuse": "0"}, True
        )
        self.assertEqual(evidence["pairs"], 5)
        result["fingerprint"]["worktree_dirty"] = True
        with self.assertRaisesRegex(ValueError, "clean"):
            metal_autoresearch.validate_production_result(
                config, result, "abc", {"width": "16", "reuse": "0"}, True
            )

    def test_production_gate_rejects_non_alternating_orders(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/instruction_ra_virtualization.template.json"
        )
        result = {
            "schema_version": 5,
            "kernel": "akita_piop",
            "local_kernel": "InstructionRaVirtualization",
            "local_metric": {
                "metric": "instruction_ra_speedup",
                "paired_metric": "paired_instruction_ra_speedups",
            },
            "run_class": {"mode": "production", "acceptance_eligible": True},
            "guards": {
                "cpu_proofs_verified": True,
                "metal_proofs_verified": True,
                "target_scale": True,
                "production_contract": True,
                "local_kernel_attributed": True,
                "local_kernel_metal_backend_exercised": True,
                "stable_source": True,
                "stable_binary": True,
            },
            "metrics": {
                "instruction_ra_speedup": 7.5,
                "piop_speedup": 2.0,
                "paired_speedups": [2.0] * 5,
                "paired_instruction_ra_speedups": [7.5] * 5,
            },
            "pairs": [
                {
                    "index": index + 1,
                    "order": ["optimized", "metal"],
                    "arms": {
                        "optimized": {"piop_ns": 200},
                        "metal": {"piop_ns": 100},
                    },
                }
                for index in range(5)
            ],
            "fingerprint": {
                "git_revision": "abc",
                "worktree_dirty": False,
                "instruction_ra_materialize_width": 16,
                "instruction_ra_reuse_inverse": False,
                "local_kernel": "InstructionRaVirtualization",
                "log_n": 26,
                "orders": [["optimized", "metal"]] * 5,
                "span": "jolt_prover::piop",
                "workload": "fibonacci",
            },
        }
        with self.assertRaisesRegex(ValueError, "alternate"):
            metal_autoresearch.validate_production_result(
                template,
                result,
                "abc",
                template["baseline_params"],
                True,
            )

    def test_production_evaluator_is_launched_from_the_frozen_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            run_dir = temporary / "run"
            run_dir.mkdir()
            result = {"schema_version": 4, "kernel": "akita_piop"}
            config = {
                "final_validation": {
                    "production_gate": {
                        "evaluator": {
                            "command": [
                                sys.executable,
                                "-c",
                                f"import json; print(json.dumps({result!r}))",
                            ],
                            "timeout_seconds": 30,
                        },
                        "expected_fingerprint": {
                            "instruction_ra_materialize_width": {
                                "parameter": "width"
                            },
                            "instruction_ra_reuse_inverse": {"parameter": "reuse"},
                        },
                    }
                }
            }
            parsed, encoded, attempt = metal_autoresearch.run_production_evaluator(
                root,
                run_dir,
                config,
                {"width": "16", "reuse": "0"},
            )
            self.assertEqual(parsed, result)
            self.assertEqual(json.loads(encoded), result)
            self.assertTrue((attempt / "result.json").is_file())

    def test_production_gate_supports_schema_five_and_named_local_pairs(self) -> None:
        config = {
            "final_validation": {
                "production_gate": {
                    "evaluator": {"schema_version": 5},
                    "local_kernel": "BytecodeReadRafCycle",
                    "metric": "bytecode_read_raf_cycle_speedup",
                    "paired_metric": "paired_bytecode_read_raf_cycle_speedups",
                    "minimum_local_speedup": 4.0,
                    "minimum_log_n": 26,
                    "minimum_pairs": 5,
                    "require_alternating_orders": True,
                    "require_clean_worktree": True,
                    "workload": "fibonacci",
                    "required_guards": ["bytecode_metal_backend_exercised"],
                    "expected_fingerprint": {},
                }
            }
        }
        result = {
            "schema_version": 5,
            "kernel": "akita_piop",
            "local_kernel": "BytecodeReadRafCycle",
            "local_metric": {
                "metric": "bytecode_read_raf_cycle_speedup",
                "paired_metric": "paired_bytecode_read_raf_cycle_speedups",
            },
            "run_class": {"mode": "production", "acceptance_eligible": True},
            "guards": {"bytecode_metal_backend_exercised": True},
            "metrics": {
                "bytecode_read_raf_cycle_speedup": 4.5,
                "piop_speedup": 2.0,
                "paired_speedups": [2.0] * 5,
                "cpu_piop_ms_samples": [200 / 1e6] * 5,
                "metal_piop_ms_samples": [100 / 1e6] * 5,
                "paired_bytecode_read_raf_cycle_speedups": [4.5] * 5,
                "bytecode_read_raf_cycle_decision": {
                    "clears": True,
                    "minimum_speedup": 4.0,
                    "minimum_pairs": 5,
                    "median_speedup": 4.5,
                    "optimized_first_median_speedup": 4.5,
                    "metal_first_median_speedup": 4.5,
                    "clears_order_strata": True,
                },
            },
            "pairs": [
                {
                    "index": index + 1,
                    "order": ["optimized", "metal"]
                    if index % 2 == 0
                    else ["metal", "optimized"],
                    "arms": {
                        "optimized": {
                            "piop_ns": 200,
                            "bytecode": self.production_bytecode_member_fixture(
                                "optimized", 450
                            ),
                        },
                        "metal": {
                            "piop_ns": 100,
                            "bytecode": self.production_bytecode_member_fixture(
                                "metal", 100
                            ),
                        },
                    },
                }
                for index in range(5)
            ],
            "resources": {"metal_piop_seconds": 5 * 100 / 1e9},
            "fingerprint": {
                "git_revision": "abc",
                "worktree_dirty": False,
                "local_kernel": "BytecodeReadRafCycle",
                "log_n": 26,
                "bytecode_metal_cutoff_log2": 16,
                "orders": [
                    ["optimized", "metal"],
                    ["metal", "optimized"],
                    ["optimized", "metal"],
                    ["metal", "optimized"],
                    ["optimized", "metal"],
                ],
                "span": "jolt_prover::piop",
                "workload": "fibonacci",
            },
        }
        evidence = metal_autoresearch.validate_production_result(
            config, result, "abc", {}, True
        )
        self.assertEqual(
            evidence["paired_metric"], "paired_bytecode_read_raf_cycle_speedups"
        )
        inconsistent = copy.deepcopy(result)
        inconsistent["pairs"][0]["arms"]["optimized"]["piop_ns"] = 300
        with self.assertRaisesRegex(ValueError, "raw PIOP"):
            metal_autoresearch.validate_production_result(
                config, inconsistent, "abc", {}, True
            )
        inconsistent = copy.deepcopy(result)
        inconsistent["pairs"][0]["arms"]["optimized"]["bytecode"][
            "rounds_total_ns"
        ] += 1
        with self.assertRaisesRegex(ValueError, "member timing"):
            metal_autoresearch.validate_production_result(
                config, inconsistent, "abc", {}, True
            )
        inconsistent = copy.deepcopy(result)
        inconsistent["pairs"][0]["arms"]["metal"]["bytecode"]["metal_counts"][
            "dense_round"
        ] -= 1
        with self.assertRaisesRegex(ValueError, "Metal schedule"):
            metal_autoresearch.validate_production_result(
                config, inconsistent, "abc", {}, True
            )
        inconsistent = copy.deepcopy(result)
        inconsistent["metrics"]["cpu_piop_ms_samples"][0] *= 2
        with self.assertRaisesRegex(ValueError, "PIOP sample"):
            metal_autoresearch.validate_production_result(
                config, inconsistent, "abc", {}, True
            )

    def test_production_bytecode_gate_recomputes_each_order_stratum(self) -> None:
        config = {
            "final_validation": {
                "production_gate": {
                    "evaluator": {"schema_version": 5},
                    "local_kernel": "BytecodeReadRafCycle",
                    "metric": "bytecode_read_raf_cycle_speedup",
                    "paired_metric": "paired_bytecode_read_raf_cycle_speedups",
                    "minimum_local_speedup": 4.0,
                    "minimum_log_n": 26,
                    "minimum_pairs": 5,
                    "require_alternating_orders": True,
                    "require_clean_worktree": True,
                    "workload": "fibonacci",
                    "required_guards": ["bytecode_metal_backend_exercised"],
                    "expected_fingerprint": {},
                }
            }
        }
        local_speedups = [5.0, 3.0, 5.0, 3.0, 5.0]
        result = {
            "schema_version": 5,
            "kernel": "akita_piop",
            "local_kernel": "BytecodeReadRafCycle",
            "local_metric": {
                "metric": "bytecode_read_raf_cycle_speedup",
                "paired_metric": "paired_bytecode_read_raf_cycle_speedups",
            },
            "run_class": {"mode": "production", "acceptance_eligible": True},
            "guards": {"bytecode_metal_backend_exercised": True},
            "metrics": {
                "bytecode_read_raf_cycle_speedup": 5.0,
                "piop_speedup": 2.0,
                "paired_speedups": [2.0] * 5,
                "cpu_piop_ms_samples": [200 / 1e6] * 5,
                "metal_piop_ms_samples": [100 / 1e6] * 5,
                "paired_bytecode_read_raf_cycle_speedups": local_speedups,
                "bytecode_read_raf_cycle_decision": {
                    "clears": True,
                    "minimum_speedup": 4.0,
                    "minimum_pairs": 5,
                    "median_speedup": 5.0,
                    "optimized_first_median_speedup": 5.0,
                    "metal_first_median_speedup": 3.0,
                    "clears_order_strata": True,
                },
            },
            "pairs": [
                {
                    "index": index + 1,
                    "order": ["optimized", "metal"]
                    if index % 2 == 0
                    else ["metal", "optimized"],
                    "arms": {
                        "optimized": {
                            "piop_ns": 200,
                            "bytecode": self.production_bytecode_member_fixture(
                                "optimized", round(speedup * 100)
                            ),
                        },
                        "metal": {
                            "piop_ns": 100,
                            "bytecode": self.production_bytecode_member_fixture(
                                "metal", 100
                            ),
                        },
                    },
                }
                for index, speedup in enumerate(local_speedups)
            ],
            "resources": {"metal_piop_seconds": 5 * 100 / 1e9},
            "fingerprint": {
                "git_revision": "abc",
                "worktree_dirty": False,
                "local_kernel": "BytecodeReadRafCycle",
                "log_n": 26,
                "bytecode_metal_cutoff_log2": 16,
                "orders": [
                    ["optimized", "metal"]
                    if index % 2 == 0
                    else ["metal", "optimized"]
                    for index in range(5)
                ],
                "span": "jolt_prover::piop",
                "workload": "fibonacci",
            },
        }
        with self.assertRaisesRegex(ValueError, "order stratum"):
            metal_autoresearch.validate_production_result(
                config, result, "abc", {}, True
            )

    def test_production_instruction_input_gate_validates_raw_members(self) -> None:
        config, params, result = self.production_instruction_input_result_fixture()
        evidence = metal_autoresearch.validate_production_result(
            config, result, "abc", params, True
        )
        self.assertEqual(
            evidence["paired_metric"],
            "paired_instruction_input_kernel_service_speedups",
        )
        self.assertEqual(evidence["optimized_first_median_speedup"], 5.0)
        self.assertEqual(evidence["metal_first_median_speedup"], 5.0)

        mutations = (
            (
                "closed member schema",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ].__setitem__("undeclared", True),
                "member record",
            ),
            (
                "round total",
                lambda value: value["pairs"][0]["arms"]["optimized"][
                    "instruction_input"
                ].__setitem__("rounds_total_ns", 27),
                "member timing",
            ),
            (
                "dense schedule",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ]["metal_counts"].__setitem__("dense_round", 8),
                "Metal schedule",
            ),
            (
                "CPU-tail schedule",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ]["metal_counts"].__setitem__("cpu_tail", 1),
                "Metal schedule",
            ),
            (
                "round allocation",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ]["resource_observation"].__setitem__(
                    "round_device_buffer_allocations", 1
                ),
                "resource accounting",
            ),
            (
                "round allocation type",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ]["resource_observation"].__setitem__(
                    "round_device_buffer_allocations", False
                ),
                "resource accounting",
            ),
            (
                "readback",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ]["resource_observation"].__setitem__("readback_bytes", 1),
                "resource accounting",
            ),
            (
                "preallocated host tail",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ]["resource_observation"].__setitem__("host_tail_bytes", 1),
                "resource accounting",
            ),
            (
                "exact sequence bytes",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ]["resource_observation"]["allocation"].__setitem__(
                    "planned_device_bytes",
                    metal_autoresearch.instruction_input_sequence_storage_bytes(26)
                    - 16,
                ),
                "resource accounting",
            ),
            (
                "resident rows in current allocation",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ]["resource_observation"]["allocation"].__setitem__(
                    "current_device_bytes", 160 * (1 << 26) - 1
                ),
                "resource accounting",
            ),
            (
                "CPU row lifecycle",
                lambda value: value["pairs"][0]["arms"]["optimized"][
                    "instruction_input_row_lifecycle"
                ].__setitem__("stage3_storage_id", 303),
                "row lifecycle",
            ),
            (
                "Metal row lifecycle",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input_row_lifecycle"
                ].__setitem__("stage1_storage_id", 303),
                "row lifecycle",
            ),
            (
                "row lifecycle boolean ID",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input_row_lifecycle"
                ].__setitem__("stage1_storage_id", True),
                "row lifecycle",
            ),
            (
                "closed row lifecycle schema",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input_row_lifecycle"
                ].__setitem__("undeclared", True),
                "row lifecycle record",
            ),
            (
                "raw speedup",
                lambda value: value["metrics"][
                    "paired_instruction_input_kernel_service_speedups"
                ].__setitem__(0, 4.5),
                "raw pair",
            ),
            (
                "pair order",
                lambda value: value["pairs"][0].__setitem__(
                    "order", ["metal", "optimized"]
                ),
                "alternate correctly",
            ),
            (
                "sample summary",
                lambda value: value["metrics"][
                    "cpu_instruction_input_kernel_service_ms_samples"
                ].__setitem__(0, 1.0),
                "sample summary",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                tampered = copy.deepcopy(result)
                mutate(tampered)
                with self.assertRaisesRegex(ValueError, message):
                    metal_autoresearch.validate_production_result(
                        config, tampered, "abc", params, True
                    )

    def test_production_requires_full_protocol_local_parent_bar(self) -> None:
        config, _, _ = self.production_instruction_input_result_fixture()
        metal_autoresearch.validate_accepted_parent_for_production(config, 4.0)
        with self.assertRaisesRegex(ValueError, "full-protocol search gate"):
            metal_autoresearch.validate_accepted_parent_for_production(config, 3.99)

    def test_production_instruction_input_gate_requires_precise_guard(self) -> None:
        config, params, result = self.production_instruction_input_result_fixture()
        del result["guards"][
            "instruction_input_no_round_device_buffer_allocations"
        ]
        result["guards"]["instruction_input_no_round_allocations"] = True
        with self.assertRaisesRegex(ValueError, "failed guards"):
            metal_autoresearch.validate_production_result(
                config, result, "abc", params, True
            )

        config, params, result = self.production_instruction_input_result_fixture()
        del result["guards"]["instruction_input_cpu_rows_reused"]
        with self.assertRaisesRegex(ValueError, "failed guards"):
            metal_autoresearch.validate_production_result(
                config, result, "abc", params, True
            )

    def test_production_instruction_input_gate_recomputes_order_strata(self) -> None:
        config, params, result = self.production_instruction_input_result_fixture()
        speedups = [5.0, 3.0, 5.0, 3.0, 5.0]
        result["metrics"]["paired_instruction_input_kernel_service_speedups"] = speedups
        result["metrics"]["cpu_instruction_input_kernel_service_ms_samples"] = [
            speedup * 100 / 1e6 for speedup in speedups
        ]
        result["metrics"]["instruction_input_kernel_service_decision"].update(
            {
                "median_speedup": 5.0,
                "optimized_first_median_speedup": 5.0,
                "metal_first_median_speedup": 3.0,
                "clears_order_strata": True,
                "clears": True,
            }
        )
        for pair, speedup in zip(result["pairs"], speedups):
            member = pair["arms"]["optimized"]["instruction_input"]
            member_ns = round(speedup * 100)
            member["member_ns"] = member_ns
            member["output_claims_ns"] = (
                member_ns
                - member["prepare_ns"]
                - member["rounds_total_ns"]
                - member["finish_ns"]
            )
        with self.assertRaisesRegex(ValueError, "order stratum"):
            metal_autoresearch.validate_production_result(
                config, result, "abc", params, True
            )

    def test_production_revision_rejects_commits_outside_editable_scope(self) -> None:
        with mock.patch.object(
            metal_autoresearch,
            "git_changed_paths",
            return_value={"editable/kernel.metal", "scripts/evaluator.py"},
        ):
            with self.assertRaisesRegex(ValueError, "outside the editable scope"):
                metal_autoresearch.validate_production_revision_scope(
                    Path("/repo"), "base", "candidate", ["editable"]
                )

    def test_production_scope_is_checked_before_evaluator_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            (root / "editable.txt").write_text("accepted")
            run_dir = temporary / "run"
            run_dir.mkdir()
            config = {
                "base_revision": "base",
                "baseline": {"metric_median": 1.0, "params": {}},
                "search_space": {},
                "scope": {"editable": ["editable.txt"], "frozen": []},
                "fingerprint": {
                    "frozen_paths_sha256": metal_autoresearch.path_digest(root, [])
                },
            }
            encoded = metal_autoresearch.canonical_json(config)
            (run_dir / "run.json").write_bytes(encoded)
            (run_dir / "run.sha256").write_text(
                metal_autoresearch.sha256(encoded) + "\n"
            )
            (run_dir / "events.jsonl").touch()
            (run_dir / "production-validations.jsonl").touch()
            metal_autoresearch.snapshot_paths(
                root, ["editable.txt"], run_dir / "snapshots" / "baseline"
            )
            with mock.patch.object(
                metal_autoresearch, "git_worktree_clean", return_value=True
            ), mock.patch.object(
                metal_autoresearch, "git_head", return_value="candidate"
            ), mock.patch.object(
                metal_autoresearch,
                "validate_production_revision_scope",
                side_effect=ValueError("outside the editable scope"),
            ), mock.patch.object(
                metal_autoresearch, "run_production_evaluator"
            ) as evaluator:
                with self.assertRaisesRegex(ValueError, "outside the editable scope"):
                    metal_autoresearch.command_validate_production(
                        SimpleNamespace(root=root, run_dir=run_dir)
                    )
                evaluator.assert_not_called()

    def test_production_rejects_out_of_space_kept_parameters_before_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            editable = root / "editable.txt"
            editable.write_text("baseline")
            run_dir = temporary / "run"
            run_dir.mkdir()
            snapshots = run_dir / "snapshots"
            snapshots.mkdir()
            metal_autoresearch.snapshot_paths(
                root, ["editable.txt"], snapshots / "baseline"
            )
            baseline_digest = metal_autoresearch.path_digest(
                snapshots / "baseline", ["editable.txt"]
            )
            editable.write_text("accepted")
            metal_autoresearch.snapshot_paths(
                root, ["editable.txt"], snapshots / "trial-001"
            )
            params = {"threads": "1024"}
            source_digest = metal_autoresearch.path_digest(
                snapshots / "trial-001", ["editable.txt"]
            )
            event = {
                "schema_version": 1,
                "index": 1,
                "trial_id": "trial-001",
                "parent_id": "baseline",
                "candidate_revision": metal_autoresearch.sha256(
                    metal_autoresearch.canonical_json(
                        {"source": source_digest, "params": params}
                    )
                ),
                "proposal_summary": "forged unmeasured configuration",
                "candidate_id": None,
                "candidate_manifest_sha256": None,
                "params": params,
                "started_at": "2026-08-04T00:00:00Z",
                "elapsed_seconds": 1.0,
                "metric_value": 8.0,
                "measurements": [8.0],
                "guards": {},
                "resources": {"gpu_seconds": 0.0},
                "verdict": "keep",
                "reason": "forged",
            }
            config = {
                "base_revision": "base",
                "baseline_params": {"threads": "128"},
                "baseline": {
                    "metric_median": 1.0,
                    "params": {"threads": "128"},
                },
                "search_space": {"threads": ["128", "256"]},
                "scope": {"editable": ["editable.txt"], "frozen": []},
                "guards": {"required_true": []},
                "fingerprint": {
                    "editable_paths_sha256": baseline_digest,
                    "frozen_paths_sha256": metal_autoresearch.path_digest(root, []),
                },
            }
            encoded = metal_autoresearch.canonical_json(config)
            (run_dir / "run.json").write_bytes(encoded)
            (run_dir / "run.sha256").write_text(
                metal_autoresearch.sha256(encoded) + "\n"
            )
            (run_dir / "events.jsonl").write_text(json.dumps(event) + "\n")
            (run_dir / "production-validations.jsonl").touch()
            with mock.patch.object(
                metal_autoresearch, "git_worktree_clean", return_value=True
            ), mock.patch.object(
                metal_autoresearch, "git_head", return_value="candidate"
            ), mock.patch.object(
                metal_autoresearch, "validate_production_revision_scope"
            ), mock.patch.object(
                metal_autoresearch,
                "run_production_evaluator",
                side_effect=AssertionError("production evaluator launched"),
            ) as evaluator:
                with self.assertRaisesRegex(ValueError, "not one of"):
                    metal_autoresearch.command_validate_production(
                        SimpleNamespace(root=root, run_dir=run_dir)
                    )
                evaluator.assert_not_called()

    def test_cached_production_promotion_is_not_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            run_dir = temporary / "run"
            config = {
                "baseline": {"metric_median": 1.0, "params": {}},
                "search_space": {},
            }
            encoded = metal_autoresearch.canonical_json(config)
            run_dir.mkdir()
            (run_dir / "run.json").write_bytes(encoded)
            (run_dir / "run.sha256").write_text(
                metal_autoresearch.sha256(encoded) + "\n"
            )
            (run_dir / "events.jsonl").touch()
            (run_dir / "production-validations.jsonl").write_text(
                json.dumps(
                    {
                        "parent_id": "baseline",
                        "status": "promoted",
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "cached production promotion"):
                metal_autoresearch.command_validate_production(
                    SimpleNamespace(root=root, run_dir=run_dir)
                )

    def test_valid_cached_production_promotion_repairs_without_rerunning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            (root / "editable.txt").write_text("accepted")
            run_dir = temporary / "run"
            run_dir.mkdir()
            snapshots = run_dir / "snapshots"
            snapshots.mkdir()
            metal_autoresearch.snapshot_paths(
                root, ["editable.txt"], snapshots / "baseline"
            )
            editable_digest = metal_autoresearch.path_digest(
                snapshots / "baseline", ["editable.txt"]
            )
            config = {
                "base_revision": "base",
                "baseline": {"metric_median": 1.0, "params": {}},
                "search_space": {},
                "scope": {"editable": ["editable.txt"], "frozen": []},
                "fingerprint": {
                    "editable_paths_sha256": editable_digest,
                    "frozen_paths_sha256": metal_autoresearch.path_digest(root, []),
                },
                "final_validation": {
                    "production_gate": {
                        "evaluator": {"schema_version": 4},
                        "metric": "instruction_ra_speedup",
                        "minimum_local_speedup": 7.0,
                        "minimum_log_n": 26,
                        "minimum_pairs": 5,
                        "require_alternating_orders": True,
                        "require_clean_worktree": True,
                        "workload": "fibonacci",
                        "required_guards": [
                            "cpu_proofs_verified",
                            "metal_proofs_verified",
                        ],
                        "expected_fingerprint": {},
                    }
                },
            }
            result = {
                "schema_version": 4,
                "kernel": "akita_piop",
                "guards": {
                    "cpu_proofs_verified": True,
                    "metal_proofs_verified": True,
                },
                "metrics": {
                    "instruction_ra_speedup": 7.5,
                    "piop_speedup": 2.0,
                    "paired_speedups": [2.0] * 5,
                    "paired_instruction_ra_speedups": [7.5] * 5,
                },
                "fingerprint": {
                    "git_revision": "candidate",
                    "worktree_dirty": False,
                    "log_n": 26,
                    "orders": [
                        ["optimized", "metal"],
                        ["metal", "optimized"],
                        ["optimized", "metal"],
                        ["metal", "optimized"],
                        ["optimized", "metal"],
                    ],
                    "span": "jolt_prover::piop",
                    "workload": "fibonacci",
                },
            }
            evidence = metal_autoresearch.validate_production_result(
                config, result, "candidate", {}, True
            )
            attempt = run_dir / "production-attempts" / "attempt-001"
            attempt.mkdir(parents=True)
            result_bytes = metal_autoresearch.canonical_json(result)
            (attempt / "result.json").write_bytes(result_bytes)
            record = {
                "schema_version": 1,
                "status": "promoted",
                "parent_id": "baseline",
                "result_sha256": metal_autoresearch.sha256(result_bytes),
                "attempt": str(attempt),
                "recorded_at": "2026-08-04T00:00:00Z",
                **evidence,
            }
            encoded = metal_autoresearch.canonical_json(config)
            (run_dir / "run.json").write_bytes(encoded)
            (run_dir / "run.sha256").write_text(
                metal_autoresearch.sha256(encoded) + "\n"
            )
            (run_dir / "events.jsonl").touch()
            (run_dir / "production-validations.jsonl").write_text(
                json.dumps(record) + "\n"
            )
            with mock.patch.object(
                metal_autoresearch, "git_worktree_clean", return_value=True
            ), mock.patch.object(
                metal_autoresearch, "git_head", return_value="candidate"
            ), mock.patch.object(
                metal_autoresearch, "validate_production_revision_scope"
            ), mock.patch.object(
                metal_autoresearch, "run_production_evaluator"
            ) as evaluator, redirect_stdout(io.StringIO()):
                self.assertEqual(
                    metal_autoresearch.command_validate_production(
                        SimpleNamespace(root=root, run_dir=run_dir)
                    ),
                    0,
                )
                evaluator.assert_not_called()

    def test_cached_promotion_repairs_the_candidate_status_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "candidate-events.jsonl").touch()
            events = [
                {
                    "trial_id": "trial-001",
                    "candidate_id": "candidate-001",
                    "candidate_manifest_sha256": "a" * 64,
                }
            ]

            metal_autoresearch.repair_candidate_promotion(
                run_dir, events, "trial-001"
            )
            metal_autoresearch.repair_candidate_promotion(
                run_dir, events, "trial-001"
            )

            records = [
                json.loads(line)
                for line in (run_dir / "candidate-events.jsonl").read_text().splitlines()
            ]
            self.assertEqual([record["status"] for record in records], ["promoted"])

    def test_production_evaluator_applies_generic_parameter_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            run_dir = temporary / "run"
            run_dir.mkdir()
            program = (
                "import json,os,sys;"
                "print(json.dumps({'schema_version':5,'kernel':'akita_piop',"
                "'argv':sys.argv[1:],'threads':os.environ['JOLT_METAL_KERNEL_THREADS']}))"
            )
            config = {
                "final_validation": {
                    "production_gate": {
                        "local_kernel": "BytecodeReadRafCycle",
                        "evaluator": {
                            "command": [sys.executable, "-c", program],
                            "schema_version": 5,
                            "timeout_seconds": 30,
                            "parameter_bindings": [
                                {
                                    "parameter": "width",
                                    "destination": "argument",
                                    "flag": "--width",
                                    "value_format": "w{}",
                                },
                                {
                                    "parameter": "reuse",
                                    "destination": "boolean_flag",
                                    "flag": "--reuse",
                                    "true_value": "1",
                                },
                                {
                                    "parameter": "threads",
                                    "destination": "environment",
                                    "name": "JOLT_METAL_KERNEL_THREADS",
                                },
                            ],
                        },
                        "expected_fingerprint": {},
                    }
                }
            }
            parsed, _, _ = metal_autoresearch.run_production_evaluator(
                root,
                run_dir,
                config,
                {"width": "32", "reuse": "1", "threads": "128"},
            )
            self.assertEqual(
                parsed["argv"],
                [
                    "--mode",
                    "production",
                    "--local-kernel",
                    "BytecodeReadRafCycle",
                    "--width",
                    "w32",
                    "--reuse",
                ],
            )
            self.assertEqual(parsed["threads"], "128")

    def test_production_rejection_replays_rollback_after_a_split_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            source = root / "editable.txt"
            source.write_text("baseline")
            event = {
                "index": 1,
                "trial_id": "trial-001",
                "parent_id": "baseline",
                "verdict": "keep",
                "metric_value": 2.0,
                "params": {},
                "candidate_id": "candidate-001",
                "candidate_manifest_sha256": "manifest",
            }
            run_dir = temporary / "run"
            self.write_recovery_run(root, run_dir, [event])
            source.write_text("candidate")
            metal_autoresearch.snapshot_paths(
                root, ["editable.txt"], run_dir / "snapshots" / "trial-001"
            )
            (run_dir / "candidate-events.jsonl").write_text(
                json.dumps(
                    {"candidate_id": "candidate-001", "status": "accepted_parent"}
                )
                + "\n"
            )
            rejection = {
                "schema_version": 1,
                "status": "rejected",
                "parent_id": "trial-001",
                "reason": "below local bar",
            }

            metal_autoresearch.finalize_production_rejection(
                root,
                run_dir,
                {"scope": {"editable": ["editable.txt"]}},
                [event],
                rejection,
            )

            self.assertEqual(source.read_text(), "baseline")
            marker = json.loads((run_dir / "production-rejected.json").read_text())
            self.assertEqual(marker["restored_parent"], "baseline")
            statuses = [
                json.loads(line)["status"]
                for line in (run_dir / "candidate-events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(statuses, ["accepted_parent", "rejected"])

    def test_instruction_ra_template_prunes_capacity_only_schedules(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/instruction_ra_virtualization.template.json"
        )
        metal_autoresearch.validate_template(template)
        metal_autoresearch.validate_params(template, template["baseline_params"])
        self.assertEqual(
            template["search_space"]["JOLT_METAL_INSTRUCTION_RA_MATERIALIZE_WIDTH"],
            [16],
        )
        self.assertEqual(
            template["search_space"]["JOLT_METAL_INSTRUCTION_RA_REUSE_INVERSE"],
            [0],
        )

    def test_schema_five_template_closes_local_metric_and_parameter_bindings(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/instruction_ra_virtualization.template.json"
        )
        wrong_metric = copy.deepcopy(template)
        wrong_metric["final_validation"]["production_gate"]["paired_metric"] = (
            "paired_bytecode_read_raf_cycle_speedups"
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            metal_autoresearch.validate_template(wrong_metric)

        missing_fingerprint = copy.deepcopy(template)
        missing_fingerprint["final_validation"]["production_gate"][
            "expected_fingerprint"
        ].pop("instruction_ra_reuse_inverse")
        with self.assertRaisesRegex(ValueError, "must match"):
            metal_autoresearch.validate_template(missing_fingerprint)

        unknown_schema = copy.deepcopy(template)
        unknown_schema["final_validation"]["production_gate"]["evaluator"][
            "schema_version"
        ] = 6
        with self.assertRaisesRegex(ValueError, "must be 4 or 5"):
            metal_autoresearch.validate_template(unknown_schema)

        bytecode = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/bytecode_read_raf_cycle.template.json"
        )
        metal_autoresearch.validate_template(bytecode)
        metal_autoresearch.validate_params(bytecode, bytecode["baseline_params"])
        bindings = bytecode["final_validation"]["production_gate"]["evaluator"][
            "parameter_bindings"
        ]
        self.assertEqual(
            {binding["parameter"] for binding in bindings},
            metal_autoresearch.PRODUCTION_LOCAL_KERNELS["BytecodeReadRafCycle"][
                "parameters"
            ],
        )

        missing_guard = copy.deepcopy(bytecode)
        missing_guard["final_validation"]["production_gate"]["required_guards"].remove(
            "bytecode_readback_exact"
        )
        with self.assertRaisesRegex(ValueError, "omits mandatory"):
            metal_autoresearch.validate_template(missing_guard)

        reserved_flag = copy.deepcopy(bytecode)
        reserved_flag["final_validation"]["production_gate"]["evaluator"][
            "command"
        ].extend(["--mode", "diagnostic"])
        with self.assertRaisesRegex(ValueError, "reserved controller flag"):
            metal_autoresearch.validate_template(reserved_flag)

        lock_override = copy.deepcopy(bytecode)
        lock_override["final_validation"]["production_gate"]["evaluator"]["env"] = {
            metal_autoresearch.EVALUATOR_LOCK_HELD_ENV: "forged"
        }
        with self.assertRaisesRegex(ValueError, "cannot override the lock token"):
            metal_autoresearch.validate_template(lock_override)

        with self.assertRaisesRegex(ValueError, "is not one of"):
            metal_autoresearch.validate_params(
                template,
                {
                    "JOLT_METAL_INSTRUCTION_RA_MATERIALIZE_WIDTH": "32",
                },
            )

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

    def test_goal_pursues_clear_local_stretch_past_portfolio_floor(self) -> None:
        contract = {
            "primary_metric": {"minimum_accepted_speedup": 4.0},
            "continuation": {
                "minimum_projected_relative_gain": 0.05,
                "clear_local_speedup_to_pursue": 4.0,
            },
        }
        candidates = [
            {
                "kernel": "small_but_fast",
                "current_piop_share": 0.01,
                "conservative_local_speedup": 4.1,
            }
        ]
        decision = metal_autoresearch.goal_decision(contract, 4.2, candidates)
        self.assertTrue(decision["continue"])
        self.assertTrue(decision["clear_local_stretch"])

    def test_repository_goal_contract_is_valid_and_uncapped(self) -> None:
        contract = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/piop_goal.json"
        )
        metal_autoresearch.validate_goal_contract(contract)
        self.assertEqual(contract["primary_metric"]["minimum_accepted_speedup"], 4.0)
        self.assertFalse(contract["continuation"]["stop_at_minimum"])
        self.assertEqual(contract["continuation"]["clear_local_speedup_to_pursue"], 4.0)
        self.assertEqual(contract["orchestration"]["promotion_queue"]["owner"], "root")
        self.assertEqual(
            contract["orchestration"]["promotion_queue"]["global_lock"],
            str(metal_autoresearch.EVALUATOR_LOCK_PATH),
        )
        self.assertEqual(contract["validation"]["interleaved_pairs"], 5)


if __name__ == "__main__":
    unittest.main()
