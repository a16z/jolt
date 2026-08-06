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

    def outer_remainder_local_contract_fixture(
        self,
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        config = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/outer_remainder.template.json"
        )
        params = {
            name: str(value) for name, value in config["baseline_params"].items()
        }
        fixture_path = ROOT / "scripts/tests/test_metal_outer_remainder_eval.py"
        fixture_spec = importlib.util.spec_from_file_location(
            "outer_remainder_autoresearch_fixture", fixture_path
        )
        assert fixture_spec is not None and fixture_spec.loader is not None
        fixture_module = importlib.util.module_from_spec(fixture_spec)
        sys.modules[fixture_spec.name] = fixture_module
        fixture_spec.loader.exec_module(fixture_module)
        events, runner = fixture_module.fixture()
        output = fixture_module.EVAL.parse_outer_remainder_result(
            events,
            runner,
            source_sha256="a" * 64,
            binary_sha256="b" * 64,
            artifact_dir="test-artifacts",
        )
        output["run"] = {
            "created_at": "2026-08-05T00:00:00+00:00",
            "host": "test-host",
            "platform": "test-platform",
            "command": ["python3", "scripts/metal_outer_remainder_eval.py"],
        }
        return config, params, output

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

    def booleanity_address_local_contract_fixture(
        self,
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        config = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/booleanity_address.template.json"
        )
        params = {
            name: str(value) for name, value in config["baseline_params"].items()
        }
        log_n = int(config["evaluator"]["env"]["JOLT_METAL_EVAL_LOG_N"])
        repeats = int(config["evaluator"]["env"]["JOLT_METAL_EVAL_REPEATS"])
        seed = int(config["evaluator"]["env"]["JOLT_METAL_EVAL_SEED"])
        cpu_threads = int(config["evaluator"]["env"]["RAYON_NUM_THREADS"])
        inner_log2 = int(params["JOLT_METAL_BOOLEANITY_ADDRESS_INNER_LOG2"])
        selectors_per_tile = int(
            params["JOLT_METAL_BOOLEANITY_ADDRESS_SELECTORS_PER_TILE"]
        )
        tile_threads = int(
            params["JOLT_METAL_BOOLEANITY_ADDRESS_TILE_THREADS"]
        )
        finalize_threads = int(
            params["JOLT_METAL_BOOLEANITY_ADDRESS_FINALIZE_THREADS"]
        )
        trace_cutoff_log2 = int(
            params["JOLT_METAL_BOOLEANITY_ADDRESS_TRACE_CUTOFF_LOG2"]
        )
        rows = 1 << log_n
        e_in = 1 << inner_log2
        e_out = rows // e_in
        selector_tiles = (29 + selectors_per_tile - 1) // selectors_per_tile
        cpu_samples = [500 + 100 * index for index in range(repeats)]
        metal_samples = [100 + 20 * index for index in range(repeats)]
        cpu_prepare = [200] * repeats
        cpu_host = [100] * repeats
        cpu_unattributed = [
            total - prepare - host
            for total, prepare, host in zip(cpu_samples, cpu_prepare, cpu_host)
        ]
        metal_prepare = [10] * repeats
        metal_dispatch = [40 + 10 * index for index in range(repeats)]
        metal_gpu = [30 + 10 * index for index in range(repeats)]
        metal_readback = [5] * repeats
        metal_host = [10] * repeats
        metal_unattributed = [
            total - prepare - dispatch - readback - host
            for total, prepare, dispatch, readback, host in zip(
                metal_samples,
                metal_prepare,
                metal_dispatch,
                metal_readback,
                metal_host,
            )
        ]
        paired = [cpu / metal for cpu, metal in zip(cpu_samples, metal_samples)]
        speedup = statistics.median(paired)
        speedup_mad = statistics.median(abs(value - speedup) for value in paired)
        selector_bytes = 29 * 8
        e_in_bytes = e_in * 16
        e_out_bytes = e_out * 16
        partial_bytes = e_out * selectors_per_tile * 256 * 16
        output_bytes = 29 * 256 * 16
        address_bytes = (
            selector_bytes
            + e_in_bytes
            + e_out_bytes
            + partial_bytes
            + output_bytes
        )
        orders = [
            ["optimized", "metal"]
            if index % 2 == 0
            else ["metal", "optimized"]
            for index in range(repeats)
        ]
        guard_names = {
            "reference_mass_lengths_exact",
            "timed_mass_lengths_exact",
            "expected_mass_count_exact",
            "exact_masses",
            "exact_four_sample_q_evals",
            "exact_round_polynomials",
            "exact_host_fiat_shamir_challenges",
            "exact_final_claim",
            "exact_output_claims",
            "exact_final_relations",
            "exact_transcript_state",
            "timed_samples_match_reference",
            "correctness_exact",
            "sample_cardinality_exact",
            "alternating_orders_exact",
            "cpu_component_timings_reconciled",
            "metal_component_timings_reconciled",
            "warmup_cpu_component_timings_reconciled",
            "warmup_metal_component_timings_reconciled",
            "gpu_active_nested_in_dispatch_wall",
            "member_durations_positive",
            "speedups_finite_positive",
            "resident_rows_reused",
            "resident_rows_stable_for_cycle_handoff",
            "metal_shape_stable_across_samples",
            "row_count_exact",
            "polynomial_count_exact",
            "production_selector_schedule_exact",
            "selector_tile_width_exact",
            "selector_tile_count_exact",
            "e_in_size_exact",
            "e_out_size_exact",
            "output_size_exact",
            "partial_size_exact",
            "production_specialization_exact",
            "requested_effective_tile_threads_exact",
            "requested_effective_finalize_threads_exact",
            "tile_pipeline_simd_width_exact",
            "finalize_pipeline_simd_width_exact",
            "tile_pipeline_thread_limit_admits_dispatch",
            "finalize_pipeline_thread_limit_admits_dispatch",
            "tile_threadgroup_memory_admitted",
            "finalize_threadgroup_memory_admitted",
            "static_device_buffers_stable",
            "static_device_buffers_distinct",
            "buffer_lengths_admitted",
            "working_set_admitted",
            "solinas_offset_exact",
            "field_and_row_sizes_exact",
            "one_execute_timed_call_per_member",
            "single_command_completion_contract",
            "single_result_readback_contract",
            "no_per_row_contribution_buffer_contract",
            "host_fiat_shamir",
            "production_trace_cutoff_admits_target",
            "all_exact",
        }
        output = {
            "schema": "booleanity_address_v1",
            "schema_version": 1,
            "kernel": "booleanity_address",
            "workload": {
                "log_n": log_n,
                "rows": rows,
                "selectors": 29,
                "k": 256,
                "address_rounds": 8,
                "row_bytes": 40,
                "seed": seed,
                "repeats": repeats,
                "orders": orders,
                "resident_rows_prepared_once_outside_members": True,
                "cpu_row_construction_outside_members": True,
                "excluded_warmup_pairs": 1,
                "cpu_member_contract": "parallel tensor-equality pushforward mirror plus host address rounds",
                "metal_member_contract": "weight preparation and upload plus one command encode/submit/wait plus one result readback plus host address rounds",
                "gpu_active_accounting": "nested in metal dispatch wall; never added to member components",
            },
            "fingerprint": {
                "trace_cutoff_log2": trace_cutoff_log2,
                "trace_cutoff_elements": 1 << trace_cutoff_log2,
                "inner_log2": inner_log2,
                "selectors_per_tile": selectors_per_tile,
                "tile_threads": tile_threads,
                "finalize_threads": finalize_threads,
                "effective_selector_tiles": selector_tiles,
                "effective_tile_threads": tile_threads,
                "effective_finalize_threads": finalize_threads,
                "production_specialized": selectors_per_tile in {3, 6},
                "accumulator_words": 5,
                "resident_row_identity": 101,
                "cpu_threads": cpu_threads,
                "cpu_control": "standalone parallel TensorEqTable/AkitaAccumulator mirror",
                "host_round_oracle": "shared deterministic evaluator implementation",
            },
            "metrics": {
                "hybrid_speedup": speedup,
                "ratio_of_member_medians": statistics.median(cpu_samples)
                / statistics.median(metal_samples),
                "paired_speedups": paired,
                "paired_speedup_mad": speedup_mad,
                "cpu_member_ns_samples": cpu_samples,
                "metal_member_ns_samples": metal_samples,
                "minimum_promotion_speedup": 4.0,
            },
            "timings": {
                "cpu_member_median_ns": statistics.median(cpu_samples),
                "cpu_prepare_median_ns": statistics.median(cpu_prepare),
                "cpu_host_rounds_median_ns": statistics.median(cpu_host),
                "cpu_unattributed_median_ns": statistics.median(cpu_unattributed),
                "metal_member_median_ns": statistics.median(metal_samples),
                "metal_prepare_median_ns": statistics.median(metal_prepare),
                "metal_dispatch_wall_median_ns": statistics.median(metal_dispatch),
                "metal_gpu_active_median_ns": statistics.median(metal_gpu),
                "metal_readback_median_ns": statistics.median(metal_readback),
                "metal_host_rounds_median_ns": statistics.median(metal_host),
                "metal_unattributed_median_ns": statistics.median(
                    metal_unattributed
                ),
                "cpu_prepare_ns_samples": cpu_prepare,
                "cpu_host_rounds_ns_samples": cpu_host,
                "cpu_unattributed_ns_samples": cpu_unattributed,
                "metal_prepare_ns_samples": metal_prepare,
                "metal_dispatch_wall_ns_samples": metal_dispatch,
                "metal_gpu_active_ns_samples": metal_gpu,
                "metal_readback_ns_samples": metal_readback,
                "metal_host_rounds_ns_samples": metal_host,
                "metal_unattributed_ns_samples": metal_unattributed,
                "exclusive_component_accounting": [
                    "prepare",
                    "dispatch_wall",
                    "readback",
                    "host_rounds",
                    "unattributed",
                ],
                "excluded_warmup": {
                    "cpu_member_ns": 500,
                    "cpu_prepare_ns": 200,
                    "cpu_host_rounds_ns": 100,
                    "cpu_unattributed_ns": 200,
                    "metal_member_ns": 100,
                    "metal_prepare_ns": 10,
                    "metal_dispatch_wall_ns": 40,
                    "metal_gpu_active_ns": 30,
                    "metal_readback_ns": 5,
                    "metal_host_rounds_ns": 10,
                    "metal_unattributed_ns": 35,
                },
            },
            "guards": {name: True for name in guard_names},
            "all_exact": True,
            "resources": {
                "device": {
                    "name": "fixture",
                    "max_buffer_length": 1 << 40,
                    "max_threadgroup_memory_length": 32_768,
                    "recommended_max_working_set_size": rows * 40
                    + address_bytes
                    + 1,
                    "current_allocated_size": rows * 40,
                    "offset": 0xFFFF_A7F7,
                },
                "device_allocated_before_reference_bytes": rows * 40,
                "resident_row_bytes": rows * 40,
                "selector_bytes": selector_bytes,
                "e_in_bytes": e_in_bytes,
                "e_out_bytes": e_out_bytes,
                "partial_owned_bytes": partial_bytes,
                "partial_expected_bytes": partial_bytes,
                "address_owned_device_bytes": address_bytes,
                "result_readback_bytes": output_bytes,
                "static_device_buffer_count": 5,
                "static_device_buffer_identities": [201, 202, 203, 204, 205],
                "gpu_active_total_ns": sum(metal_gpu),
                "gpu_seconds": (30 + sum(metal_gpu)) / 1e9,
            },
            "pipelines": {
                "tile": {
                    "thread_execution_width": 32,
                    "max_total_threads_per_threadgroup": 1024,
                    "static_threadgroup_bytes": 0,
                    "dynamic_threadgroup_bytes": selectors_per_tile * 256 * 5 * 4,
                    "total_threadgroup_bytes": selectors_per_tile * 256 * 5 * 4,
                    "effective_threads_per_threadgroup": tile_threads,
                },
                "finalize": {
                    "thread_execution_width": 32,
                    "max_total_threads_per_threadgroup": 1024,
                    "static_threadgroup_bytes": 0,
                    "dynamic_threadgroup_bytes": finalize_threads * 16,
                    "total_threadgroup_bytes": finalize_threads * 16,
                    "effective_threads_per_threadgroup": finalize_threads,
                },
            },
            "promotion": {
                "minimum_log_n": 26,
                "minimum_pairs": 5,
                "minimum_speedup": 4.0,
                "scale_eligible": True,
                "pair_count_eligible": True,
                "speedup_eligible": True,
                "local_eligible": True,
                "production_piop_holdout_required": True,
            },
            "oracle_limits": {
                "cpu_denominator_is_production_kernel": False,
                "cpu_denominator_scope": "standalone optimized pushforward mirror plus the same host-round routine",
                "host_rounds_are_independently_implemented": False,
                "mass_oracle_independent_of_metal_shader": True,
                "command_and_readback_counts_are_runtime_counters": False,
                "requires_production_piop_holdout": True,
            },
        }
        return config, params, output

    def hamming_weight_local_contract_fixture(
        self,
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        _, _, output = self.booleanity_address_local_contract_fixture()
        config = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/hamming_weight_claim_reduction.template.json"
        )
        params = {
            name: str(value) for name, value in config["baseline_params"].items()
        }
        rows = output["workload"]["rows"]
        opportunities = rows * 29
        nonzero = opportunities - rows

        output["schema"] = "hamming_weight_claim_reduction_v1"
        output["kernel"] = "hamming_weight_claim_reduction"
        workload = output["workload"]
        workload["hamming_address_rounds"] = workload.pop("address_rounds")
        workload.update(
            {
                "selector_row_opportunities": opportunities,
                "nonzero_recentered_contributions": nonzero,
                "resident_row_upload_bytes_inside_metal_member": 0,
                "cpu_member_contract": "optimized shared-row tensor-equality pushforward mirror, bucket-zero recentering, W/baseline construction, and eight host Fiat-Shamir rounds",
                "metal_member_contract": "cycle-equality preparation and upload over resident rows, one command encode/submit/wait, one result readback, bucket-zero recentering, and the same W/baseline and eight host Fiat-Shamir rounds",
            }
        )
        output["fingerprint"].update(
            {
                "cpu_control": "standalone parallel optimized TensorEqTable/AkitaAccumulator pushforward mirror",
                "host_round_oracle": "identical deterministic W/baseline and host-round implementation",
            }
        )
        metrics = output["metrics"]
        cpu_median = statistics.median(metrics["cpu_member_ns_samples"])
        metal_median = statistics.median(metrics["metal_member_ns_samples"])
        metrics.update(
            {
                "selector_row_opportunities": opportunities,
                "nonzero_recentered_contributions": nonzero,
                "cpu_selector_row_opportunities_per_second": opportunities
                * 1e9
                / cpu_median,
                "metal_selector_row_opportunities_per_second": opportunities
                * 1e9
                / metal_median,
                "cpu_nonzero_recentered_contributions_per_second": nonzero
                * 1e9
                / cpu_median,
                "metal_nonzero_recentered_contributions_per_second": nonzero
                * 1e9
                / metal_median,
            }
        )
        guards = output["guards"]
        guards["exact_skipped_q_evals"] = guards.pop(
            "exact_four_sample_q_evals"
        )
        guards["resident_rows_stable_for_stage7_handoff"] = guards.pop(
            "resident_rows_stable_for_cycle_handoff"
        )
        guards.update(
            {
                "recentered_bucket_zero_exact": True,
                "nonzero_recentered_values_present": True,
                "nonzero_contribution_count_admitted": True,
                "output_claim_count_exact": True,
            }
        )
        resources = output["resources"]
        resources["hamming_owned_device_bytes"] = resources.pop(
            "address_owned_device_bytes"
        )
        output["oracle_limits"]["cpu_denominator_scope"] = (
            "standalone optimized shared-row pushforward mirror plus identical W/baseline and host rounds"
        )
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
        cpu_reference_ns = int(
            config["evaluator"]["env"]["JOLT_METAL_EVAL_CPU_REFERENCE_NS"]
        )
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
        warmup_wall = 220
        warmup_reset = 20
        warmup_gpu_wall = 180
        warmup_gpu_active = 160
        warmup_host = 5
        warmup_readback = 5
        warmup_tail = 5
        validation_gpu_active = 30
        host_samples = [10 + index for index in range(repeats)]
        readback_samples = [5 + index for index in range(repeats)]
        tail_samples = [10 + index for index in range(repeats)]
        paired = [
            cpu / hybrid for cpu, hybrid in zip(cpu_samples, hybrid_samples)
        ]
        resident_paired = [
            cpu / resident for cpu, resident in zip(cpu_samples, resident_samples)
        ]
        reference_paired = [
            cpu_reference_ns / hybrid for hybrid in hybrid_samples
        ]
        sequence_bytes = metal_autoresearch.instruction_input_sequence_storage_bytes(
            log_n
        )
        cpu_rows_bytes = 48 * rows
        resident_rows_bytes = 48 * rows
        resident_source_rows_bytes = 160 * rows
        persistent_bytes = cpu_rows_bytes + resident_rows_bytes + sequence_bytes
        cpu_first_dense_bytes = 8 * (rows // 2) * 16
        cpu_bind_scratch_bytes = (rows // 4) * 16
        hybrid_tail_bytes = 2 * 8 * cutoff * 16
        hybrid_tail_bind_scratch_bytes = (cutoff // 2) * 16
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
                "exactly_one_excluded_residency_warmup",
                "all_exact",
            )
        }
        output = {
            "schema": "instruction_input_v4",
            "schema_version": 4,
            "kernel": "instruction_input",
            "metrics": {
                "hybrid_speedup": statistics.median(paired),
                "resident_speedup": statistics.median(resident_paired),
                "frozen_cpu_reference_ratio": statistics.median(reference_paired),
                "paired_hybrid_speedups": paired,
                "paired_resident_speedups": resident_paired,
                "paired_frozen_cpu_reference_ratios": reference_paired,
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
                "workload_and_protocol_preparation_seconds": 1.0,
                "resident_source_sequence_upload_and_storage_preparation_seconds": 2.0,
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
                "timed_gpu_active_total_seconds": sum(gpu_active_samples) / 1e9,
                "evaluator_gpu_active_total_seconds": (
                    validation_gpu_active
                    + warmup_gpu_active
                    + sum(gpu_active_samples)
                )
                / 1e9,
                "validation_gpu_active_ns": validation_gpu_active,
                "residency_warmup_wall_ns": warmup_wall,
                "residency_warmup_resident_ns": warmup_wall - warmup_reset,
                "residency_warmup_reset_ns": warmup_reset,
                "residency_warmup_gpu_dispatch_wall_ns": warmup_gpu_wall,
                "residency_warmup_host_round_ns": warmup_host,
                "residency_warmup_readback_ns": warmup_readback,
                "residency_warmup_cpu_tail_ns": warmup_tail,
                "residency_warmup_gpu_active_ns": warmup_gpu_active,
                "residency_warmup_to_timed_gpu_active_ratio": warmup_gpu_active
                / statistics.median(gpu_active_samples),
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
                "gpu_seconds": (
                    validation_gpu_active
                    + sum(gpu_active_samples)
                    + warmup_gpu_active
                )
                / 1e9,
                "cpu_native_rows_bytes": cpu_rows_bytes,
                "resident_compact_rows_bytes": resident_rows_bytes,
                "sequence_owned_working_storage_bytes": sequence_bytes,
                "cpu_phase_persistent_modeled_bytes": cpu_rows_bytes,
                "cpu_first_dense_table_bytes": cpu_first_dense_bytes,
                "cpu_bind_scratch_capacity_bytes": cpu_bind_scratch_bytes,
                "cpu_trial_peak_modeled_bytes": cpu_rows_bytes
                + cpu_first_dense_bytes
                + cpu_bind_scratch_bytes,
                "metal_phase_persistent_modeled_bytes": persistent_bytes,
                "hybrid_readback_plus_tail_table_capacity_bytes": hybrid_tail_bytes,
                "hybrid_cpu_tail_bind_scratch_capacity_bytes": hybrid_tail_bind_scratch_bytes,
                "metal_warmup_and_trial_peak_modeled_bytes": persistent_bytes
                + hybrid_tail_bytes
                + hybrid_tail_bind_scratch_bytes,
                "sequence_setup_peak_modeled_bytes": persistent_bytes
                + resident_source_rows_bytes,
                "evaluator_peak_modeled_bytes": max(
                    cpu_rows_bytes + cpu_first_dense_bytes + cpu_bind_scratch_bytes,
                    persistent_bytes
                    + hybrid_tail_bytes
                    + hybrid_tail_bind_scratch_bytes,
                    persistent_bytes + resident_source_rows_bytes,
                ),
                "resident_source_host_copy_bytes_dropped_before_metal_trials": resident_source_rows_bytes,
                "setup_peak_increment_from_resident_source_copy_bytes": resident_source_rows_bytes,
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
                "resident_compact_row_bytes": 48,
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
                "primary_timing": "after one excluded full-sequence residency warmup: resident sequence reset plus Metal rounds, host Fiat-Shamir, one dense readback, and exact four-sample CPU tail",
                "primary_metric": "timed complete-member throughput normalized by a frozen CPU reference",
                "frozen_cpu_reference_ns": cpu_reference_ns,
                "frozen_cpu_reference_provenance": metal_autoresearch.INSTRUCTION_INPUT_V3_CPU_REFERENCE_PROVENANCE,
                "live_cpu_controls_in_primary_metric": False,
                "workload_preparation_in_primary_metric": False,
                "sequence_preparation_in_primary_metric": False,
                "resident_source_materialization_in_primary_metric": False,
                "residency_warmup_in_primary_metric": False,
                "residency_warmup_reuses_first_protocol_tape": True,
                "residency_warmup_runs": 1,
                "host_readback_allocation_in_primary_metric": False,
                "protocol_tape_preparation_in_primary_metric": False,
                "protocol_tapes_per_process": repeats,
                "protocol_tape_derivation": "base_seed xor ((repeat + 1) * 0x9e3779b97f4a7c15 modulo 2^64)",
                "cpu_trials_run_while_resident_metal_sequence_is_allocated": False,
                "cpu_trials_run_before_resident_source_materialization": True,
                "cpu_control": "standalone row-stride and arithmetic mirror of OptimizedInstructionInputKernel",
                "metal_control": "public InstructionInputSequence over resident compact InstructionInputRow storage",
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
                "frozen_cpu_reference_ns": cpu_reference_ns,
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
                "arm_schedule": [
                    "cpu_batch",
                    "excluded_full_metal_warmup",
                    "metal_timed_batch",
                ],
                "process_model": "single_process_steady_state_search_proxy",
                "warmup_tape_index": 0,
                "validation_full_sequence_metal_runs": 1,
                "residency_warmup_runs": 1,
                "timed_full_sequence_metal_runs": repeats,
                "evaluator_full_sequence_metal_runs": repeats + 2,
                "protocol_seeds": protocol_seeds,
                "protocol_transcript_states": [
                    [index + 1] * 32 for index in range(repeats)
                ],
            },
        }
        return config, params, output

    def instruction_input_v3_local_contract_fixture(
        self,
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        config, params, output = self.instruction_input_local_contract_fixture()
        config["evaluator"]["result_contract"] = "instruction_input_v3"
        config["evaluator"]["result_schema_version"] = 3
        output["schema"] = "instruction_input_v3"
        output["schema_version"] = 3
        rows = output["workload"]["rows"]
        cutoff = output["workload"]["cutoff_elements"]
        cpu_rows_bytes = 48 * rows
        resident_rows_bytes = 160 * rows
        sequence_bytes = output["resources"]["sequence_owned_working_storage_bytes"]
        persistent_bytes = cpu_rows_bytes + resident_rows_bytes + sequence_bytes
        cpu_peak_bytes = output["resources"]["cpu_trial_peak_modeled_bytes"]
        hybrid_tail_bytes = output["resources"][
            "hybrid_readback_plus_tail_table_capacity_bytes"
        ]
        hybrid_scratch_bytes = output["resources"][
            "hybrid_cpu_tail_bind_scratch_capacity_bytes"
        ]
        del output["workload"]["resident_compact_row_bytes"]
        output["workload"]["resident_stage1_row_bytes"] = 160
        output["workload"]["metal_control"] = (
            "public InstructionInputSequence over resident SpartanOuterUniskipRow storage"
        )
        del output["resources"]["resident_compact_rows_bytes"]
        output["resources"]["resident_stage1_rows_bytes"] = resident_rows_bytes
        output["resources"]["metal_phase_persistent_modeled_bytes"] = persistent_bytes
        output["resources"]["metal_warmup_and_trial_peak_modeled_bytes"] = (
            persistent_bytes + hybrid_tail_bytes + hybrid_scratch_bytes
        )
        output["resources"]["sequence_setup_peak_modeled_bytes"] = (
            persistent_bytes + resident_rows_bytes
        )
        output["resources"]["evaluator_peak_modeled_bytes"] = max(
            cpu_peak_bytes,
            persistent_bytes + hybrid_tail_bytes + hybrid_scratch_bytes,
            persistent_bytes + resident_rows_bytes,
        )
        output["resources"][
            "resident_source_host_copy_bytes_dropped_before_metal_trials"
        ] = resident_rows_bytes
        output["resources"][
            "setup_peak_increment_from_resident_source_copy_bytes"
        ] = resident_rows_bytes
        self.assertEqual(output["resources"]["cutoff_readback_bytes"], 8 * cutoff * 16)
        return config, params, output

    def instruction_input_v2_local_contract_fixture(
        self,
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        config, params, output = self.instruction_input_v3_local_contract_fixture()
        config["evaluator"]["result_contract"] = "instruction_input_v2"
        config["evaluator"]["result_schema_version"] = 2
        config["metric"]["name"] = "hybrid_speedup"
        del config["evaluator"]["env"]["JOLT_METAL_EVAL_CPU_REFERENCE_NS"]
        output["schema"] = "instruction_input_v2"
        output["schema_version"] = 2
        del output["metrics"]["frozen_cpu_reference_ratio"]
        del output["metrics"]["paired_frozen_cpu_reference_ratios"]
        for name in (
            "primary_metric",
            "frozen_cpu_reference_ns",
            "frozen_cpu_reference_provenance",
            "live_cpu_controls_in_primary_metric",
        ):
            del output["workload"][name]
        del output["fingerprint"]["frozen_cpu_reference_ns"]
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
        self, backend: str, service_ns: int, log_n: int = 26, cutoff_log2: int = 16
    ) -> dict[str, object]:
        prefetch_submit_ns = 1 if backend == "metal" else 0
        member_ns = service_ns - prefetch_submit_ns
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
            "storage_initialize": 0,
            "storage_initialize_complete": 0,
            "native_primer_submit": 0,
            "native_primer_join": 0,
            "native_primer_complete": 0,
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
                    "storage_initialize": 1,
                    "storage_initialize_complete": 1,
                    "native_primer_submit": 1,
                    "native_primer_join": 1,
                    "native_primer_complete": 1,
                }
            )
            buffer_identities = [301, 302, 303, 304, 305, 306]
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
                "storage_initialization": {
                    "mode": "minimal",
                    "device_buffers": 6,
                    "bytes": 96,
                    "protocol_dispatches": 0,
                    "buffer_identities": buffer_identities,
                    "gpu_active_ns": 1,
                    "wall_ns": 10,
                },
                "native_primer": {
                    "source_elements": 64,
                    "e_in_elements": 1,
                    "e_out_elements": 32,
                    "resident_rows_storage_id": 202,
                    "storage_buffer_identities": buffer_identities,
                    "command_committed": True,
                    "protocol_state_advanced": False,
                    "timings": {
                        "submit_wall_ns": 1,
                        "submit_span_wall_ns": 1,
                        "overlap_wall_ns": 100,
                        "join_wall_ns": 1,
                        "lifecycle_wall_ns": 102,
                        "gpu_active_ns": 1,
                    },
                    "completed_before_join": True,
                    "command_completed": True,
                    "produced_zero": True,
                },
            }
        return {
            "prepare_ns": prepare_ns,
            "rounds_ns": rounds_ns,
            "rounds_total_ns": sum(rounds_ns),
            "finish_ns": finish_ns,
            "output_claims_ns": output_claims_ns,
            "member_ns": member_ns,
            "prefetch_submit_ns": prefetch_submit_ns,
            "service_ns": service_ns,
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
            "kind": "metal_compact_resident",
            "rows": 1 << log_n,
            "row_bytes": 48,
            "prepare_storage_id": 202,
            "stage1_storage_id": 202,
            "stage3_storage_id": 202,
            "residual_storage_id": 203,
            "row_production": {
                "source_kind": "owned_random_access",
                "witness_row_extractions": 1 << log_n,
                "residual_rows_written": 1 << log_n,
                "compact_rows_written": 1 << log_n,
                "compact_row_bytes": 48,
                "residual_row_bytes": 112,
                "compact_allocations": 1,
                "residual_allocations": 1,
                "full_row_allocations": 0,
                "full_domain_copy_bytes": 0,
                "full_domain_copy_dispatches": 0,
                "host_repack_rows": 0,
            },
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
        cpu_piop_ns = 1_000
        metal_piop_ns = 200
        cpu_prepare_ns = 10
        metal_prepare_ns = 20
        cpu_member_ns = 500
        metal_member_ns = 100
        local_speedup = cpu_member_ns / metal_member_ns
        orders = [
            ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
            for index in range(pairs)
        ]
        result = {
            "schema_version": gate["evaluator"]["schema_version"],
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
                "cpu_piop_ms": cpu_piop_ns / 1e6,
                "metal_piop_ms": metal_piop_ns / 1e6,
                "cpu_backend_witness_prepare_ms": cpu_prepare_ns / 1e6,
                "metal_backend_witness_prepare_ms": metal_prepare_ns / 1e6,
                "paired_speedups": [cpu_piop_ns / metal_piop_ns] * pairs,
                "cpu_piop_ms_samples": [cpu_piop_ns / 1e6] * pairs,
                "metal_piop_ms_samples": [metal_piop_ns / 1e6] * pairs,
                "cpu_backend_witness_prepare_ms_samples": [
                    cpu_prepare_ns / 1e6
                ]
                * pairs,
                "metal_backend_witness_prepare_ms_samples": [
                    metal_prepare_ns / 1e6
                ]
                * pairs,
                "paired_speedups_with_backend_witness_prepare": [
                    (cpu_piop_ns + cpu_prepare_ns)
                    / (metal_piop_ns + metal_prepare_ns)
                ]
                * pairs,
                "piop_plus_backend_witness_prepare_speedup": (
                    cpu_piop_ns + cpu_prepare_ns
                )
                / (metal_piop_ns + metal_prepare_ns),
                "paired_instruction_input_kernel_service_speedups": [local_speedup]
                * pairs,
                "paired_instruction_input_kernel_service_fractional_improvements": [
                    1.0 - metal_member_ns / cpu_member_ns
                ]
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
                    "median_fractional_improvement": 1.0
                    - metal_member_ns / cpu_member_ns,
                    "mad_fractional_improvement": 0.0,
                    "cpu_member_ms_median": cpu_member_ns / 1e6,
                    "cpu_member_ms_mad": 0.0,
                    "metal_member_ms_median": metal_member_ns / 1e6,
                    "metal_member_ms_mad": 0.0,
                    "enough_pairs": True,
                    "clears_speedup": True,
                    "clears_fractional_improvement": True,
                    "clears_noise": True,
                    "lower_metal_median": True,
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
                            "backend_witness_prepare_ns": cpu_prepare_ns,
                            "instruction_input": self.production_instruction_input_member_fixture(
                                "optimized", cpu_member_ns, log_n, cutoff_log2
                            ),
                            "instruction_input_row_lifecycle": self.production_instruction_input_row_lifecycle_fixture(
                                "optimized", log_n
                            ),
                        },
                        "metal": {
                            "piop_ns": metal_piop_ns,
                            "backend_witness_prepare_ns": metal_prepare_ns,
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
                "instruction_input_storage_initialization": "minimal",
                "instruction_input_native_primer": "async",
                "orders": orders,
                "span": "jolt_prover::piop",
                "workload": "fibonacci",
            },
        }
        return config, params, result

    def production_booleanity_address_member_fixture(
        self,
        backend: str,
        member_ns: int,
        params: dict[str, str],
        log_n: int = 26,
    ) -> tuple[dict[str, object], object]:
        prepare_ns = 100 if backend == "optimized" else 50
        row_source_ns = 40 if backend == "optimized" else 0
        service_member_ns = member_ns + row_source_ns
        rounds_ns = [1] * 8
        host_fiat_shamir_ns = [1] * 8
        finish_ns = 1
        output_claims_ns = (
            service_member_ns
            - prepare_ns
            - sum(rounds_ns)
            - sum(host_fiat_shamir_ns)
            - finish_ns
        )
        self.assertGreater(output_claims_ns, 0)
        metal_counts = {
            "prepare": 0,
            "sequence_prepare": 0,
            "allocation_plan": 0,
            "dispatch": 0,
            "readback": 0,
        }
        resource = None
        lifecycle = None
        if backend == "metal":
            metal_counts = {name: 1 for name in metal_counts}
            rows = 1 << log_n
            inner_log2 = int(
                params["JOLT_METAL_BOOLEANITY_ADDRESS_INNER_LOG2"]
            )
            selectors_per_tile = int(
                params["JOLT_METAL_BOOLEANITY_ADDRESS_SELECTORS_PER_TILE"]
            )
            tile_threads = int(
                params["JOLT_METAL_BOOLEANITY_ADDRESS_TILE_THREADS"]
            )
            finalize_threads = int(
                params["JOLT_METAL_BOOLEANITY_ADDRESS_FINALIZE_THREADS"]
            )
            e_in = 1 << inner_log2
            e_out = rows // e_in
            selector_tiles = (29 + selectors_per_tile - 1) // selectors_per_tile
            storage_id = 401
            current_bytes = rows * 40
            planned_bytes = (
                metal_autoresearch.booleanity_address_sequence_storage_bytes(
                    log_n, inner_log2, selectors_per_tile
                )
            )
            resource = {
                "sequence": {
                    "resident_rows_storage_id": storage_id,
                    "resident_rows": rows,
                    "resident_row_bytes": 40,
                    "row_upload_bytes": 0,
                    "polys": 29,
                    "k": 256,
                    "e_in_elements": e_in,
                    "e_out_elements": e_out,
                    "requested_inner_log2": inner_log2,
                    "effective_inner_log2": inner_log2,
                    "requested_selectors_per_tile": selectors_per_tile,
                    "effective_selectors_per_tile": selectors_per_tile,
                    "requested_tile_threads": tile_threads,
                    "effective_tile_threads": tile_threads,
                    "requested_finalize_threads": finalize_threads,
                    "effective_finalize_threads": finalize_threads,
                    "selector_tiles": selector_tiles,
                    "production_specialized": selectors_per_tile in {3, 6},
                },
                "allocation": {
                    "device_buffers": 5,
                    "planned_device_bytes": planned_bytes,
                    "current_device_bytes": current_bytes,
                    "recommended_device_bytes": current_bytes + planned_bytes + 1,
                },
                "dispatch": {
                    "command_buffers": 1,
                    "tile_dispatches": selector_tiles,
                    "finalize_dispatches": selector_tiles,
                    "command_completed": True,
                    "gpu_active_ns": 40,
                    "resident_rows_storage_id": storage_id,
                },
                "readback": {
                    "elements": 29 * 256,
                    "bytes": 29 * 256 * 16,
                    "readbacks": 1,
                },
            }
            lifecycle = {
                "kind": "metal_booleanity_resident",
                "rows": rows,
                "row_bytes": 40,
                "device_registry_id": 7,
                "stage5_storage_id": storage_id,
                "stage6a_storage_id": storage_id,
                "stage6b_storage_id": storage_id,
                "stage5": {
                    "row_allocations": 1,
                    "row_upload_bytes": rows * 40,
                },
                "stage6a": {"row_allocations": 0, "row_upload_bytes": 0},
                "stage6b": {"row_allocations": 0, "row_upload_bytes": 0},
            }
        member = {
            "prepare_ns": prepare_ns,
            "rounds_ns": rounds_ns,
            "rounds_total_ns": sum(rounds_ns),
            "host_fiat_shamir_ns": host_fiat_shamir_ns,
            "host_fiat_shamir_total_ns": sum(host_fiat_shamir_ns),
            "row_source_ns": row_source_ns,
            "normalized_prepare_ns": prepare_ns - row_source_ns,
            "normalized_member_ns": member_ns,
            "finish_ns": finish_ns,
            "output_claims_ns": output_claims_ns,
            "member_ns": service_member_ns,
            "outer_counts": {
                "prepare": 1,
                "prove_round": 8,
                "finish_rounds": 1,
                "output_claims": 1,
            },
            "metal_counts": metal_counts,
            "resource_observation": resource,
        }
        return member, lifecycle

    def production_booleanity_address_result_fixture(
        self,
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        config = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/booleanity_address.template.json"
        )
        params = {
            name: str(value) for name, value in config["baseline_params"].items()
        }
        gate = config["final_validation"]["production_gate"]
        pair_count = int(gate["minimum_pairs"])
        cpu_piop_ns = 1_000
        metal_piop_ns = 200
        cpu_prepare_ns = 10
        metal_prepare_ns = 20
        cpu_member_ns = 500
        metal_member_ns = 100
        local_speedup = cpu_member_ns / metal_member_ns
        orders = [
            ["optimized", "metal"]
            if index % 2 == 0
            else ["metal", "optimized"]
            for index in range(pair_count)
        ]
        pair_records = []
        for index, order in enumerate(orders):
            cpu_member, cpu_lifecycle = (
                self.production_booleanity_address_member_fixture(
                    "optimized", cpu_member_ns, params
                )
            )
            metal_member, metal_lifecycle = (
                self.production_booleanity_address_member_fixture(
                    "metal", metal_member_ns, params
                )
            )
            pair_records.append(
                {
                    "index": index + 1,
                    "order": order,
                    "arms": {
                        "optimized": {
                            "piop_ns": cpu_piop_ns,
                            "backend_witness_prepare_ns": cpu_prepare_ns,
                            "booleanity_address": cpu_member,
                            "booleanity_address_row_lifecycle": cpu_lifecycle,
                        },
                        "metal": {
                            "piop_ns": metal_piop_ns,
                            "backend_witness_prepare_ns": metal_prepare_ns,
                            "booleanity_address": metal_member,
                            "booleanity_address_row_lifecycle": metal_lifecycle,
                        },
                    },
                }
            )
        _, decision = metal_autoresearch.recompute_local_member_decision(
            pair_records,
            [cpu_member_ns] * pair_count,
            [metal_member_ns] * pair_count,
            float(gate["minimum_local_speedup"]),
            pair_count,
        )
        result = {
            "schema_version": 7,
            "kernel": "akita_piop",
            "local_kernel": "BooleanityAddressPhase",
            "local_metric": {
                "metric": "booleanity_address_phase_speedup",
                "paired_metric": "paired_booleanity_address_phase_speedups",
            },
            "run_class": {"mode": "production", "acceptance_eligible": True},
            "guards": {name: True for name in gate["required_guards"]},
            "metrics": {
                "booleanity_address_phase_speedup": local_speedup,
                "booleanity_address_phase_service_speedup": (
                    cpu_member_ns + 40
                )
                / metal_member_ns,
                "piop_speedup": cpu_piop_ns / metal_piop_ns,
                "piop_plus_backend_witness_prepare_speedup": (
                    cpu_piop_ns + cpu_prepare_ns
                )
                / (metal_piop_ns + metal_prepare_ns),
                "cpu_piop_ms": cpu_piop_ns / 1e6,
                "metal_piop_ms": metal_piop_ns / 1e6,
                "cpu_backend_witness_prepare_ms": cpu_prepare_ns / 1e6,
                "metal_backend_witness_prepare_ms": metal_prepare_ns / 1e6,
                "paired_speedups": [cpu_piop_ns / metal_piop_ns] * pair_count,
                "paired_speedups_with_backend_witness_prepare": [
                    (cpu_piop_ns + cpu_prepare_ns)
                    / (metal_piop_ns + metal_prepare_ns)
                ]
                * pair_count,
                "paired_booleanity_address_phase_speedups": [local_speedup]
                * pair_count,
                "paired_booleanity_address_phase_service_speedups": [
                    (cpu_member_ns + 40) / metal_member_ns
                ]
                * pair_count,
                "paired_booleanity_address_phase_fractional_improvements": [
                    1.0 - metal_member_ns / cpu_member_ns
                ]
                * pair_count,
                "cpu_piop_ms_samples": [cpu_piop_ns / 1e6] * pair_count,
                "metal_piop_ms_samples": [metal_piop_ns / 1e6] * pair_count,
                "cpu_backend_witness_prepare_ms_samples": [
                    cpu_prepare_ns / 1e6
                ]
                * pair_count,
                "metal_backend_witness_prepare_ms_samples": [
                    metal_prepare_ns / 1e6
                ]
                * pair_count,
                "cpu_booleanity_address_phase_ms_samples": [
                    cpu_member_ns / 1e6
                ]
                * pair_count,
                "metal_booleanity_address_phase_ms_samples": [
                    metal_member_ns / 1e6
                ]
                * pair_count,
                "cpu_booleanity_address_phase_service_ms_samples": [
                    (cpu_member_ns + 40) / 1e6
                ]
                * pair_count,
                "metal_booleanity_address_phase_service_ms_samples": [
                    metal_member_ns / 1e6
                ]
                * pair_count,
                "booleanity_address_phase_decision": decision,
            },
            "pairs": pair_records,
            "resources": {
                "metal_piop_seconds": pair_count * metal_piop_ns / 1e9
            },
            "fingerprint": {
                "git_revision": "abc",
                "worktree_dirty": False,
                "local_kernel": "BooleanityAddressPhase",
                "log_n": 26,
                "cpu_threads": int(config["evaluator"]["env"]["RAYON_NUM_THREADS"]),
                "orders": orders,
                "span": "jolt_prover::piop",
                "workload": "fibonacci",
                "booleanity_address_metal_inner_log2": int(
                    params["JOLT_METAL_BOOLEANITY_ADDRESS_INNER_LOG2"]
                ),
                "booleanity_address_metal_selectors_per_tile": int(
                    params["JOLT_METAL_BOOLEANITY_ADDRESS_SELECTORS_PER_TILE"]
                ),
                "booleanity_address_metal_tile_threads": int(
                    params["JOLT_METAL_BOOLEANITY_ADDRESS_TILE_THREADS"]
                ),
                "booleanity_address_metal_finalize_threads": int(
                    params["JOLT_METAL_BOOLEANITY_ADDRESS_FINALIZE_THREADS"]
                ),
                "booleanity_address_metal_trace_cutoff_log2": int(
                    params["JOLT_METAL_BOOLEANITY_ADDRESS_TRACE_CUTOFF_LOG2"]
                ),
            },
        }
        return config, params, result

    def production_hamming_weight_member_fixture(
        self,
        backend: str,
        member_ns: int,
        params: dict[str, str],
        log_n: int = 26,
    ) -> tuple[dict[str, object], object]:
        booleanity_params = {
            "JOLT_METAL_BOOLEANITY_ADDRESS_INNER_LOG2": params[
                "JOLT_METAL_HAMMING_WEIGHT_INNER_LOG2"
            ],
            "JOLT_METAL_BOOLEANITY_ADDRESS_SELECTORS_PER_TILE": params[
                "JOLT_METAL_HAMMING_WEIGHT_SELECTORS_PER_TILE"
            ],
            "JOLT_METAL_BOOLEANITY_ADDRESS_TILE_THREADS": params[
                "JOLT_METAL_HAMMING_WEIGHT_TILE_THREADS"
            ],
            "JOLT_METAL_BOOLEANITY_ADDRESS_FINALIZE_THREADS": params[
                "JOLT_METAL_HAMMING_WEIGHT_FINALIZE_THREADS"
            ],
        }
        member, lifecycle = self.production_booleanity_address_member_fixture(
            backend, member_ns, booleanity_params, log_n
        )
        if backend == "metal":
            assert isinstance(lifecycle, dict)
            storage_id = lifecycle["stage5_storage_id"]
            lifecycle.update(
                {
                    "kind": "metal_hamming_resident",
                    "stage6b_retain_storage_id": storage_id,
                    "stage7_storage_id": storage_id,
                    "stage6b_retain": {
                        "row_allocations": 0,
                        "row_upload_bytes": 0,
                    },
                    "stage7": {"row_allocations": 0, "row_upload_bytes": 0},
                    "terminal_consumer": True,
                    "terminal_carry_removed": True,
                }
            )
        return member, lifecycle

    def production_hamming_weight_result_fixture(
        self,
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        config = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/hamming_weight_claim_reduction.template.json"
        )
        params = {
            name: str(value) for name, value in config["baseline_params"].items()
        }
        gate = config["final_validation"]["production_gate"]
        _, _, booleanity_result = self.production_booleanity_address_result_fixture()
        result = copy.deepcopy(booleanity_result)
        result["local_kernel"] = "HammingWeightClaimReduction"
        result["local_metric"] = {
            "metric": "hamming_weight_claim_reduction_speedup",
            "paired_metric": "paired_hamming_weight_claim_reduction_speedups",
        }
        result["guards"] = {name: True for name in gate["required_guards"]}
        metric_names = {
            "booleanity_address_phase_speedup": "hamming_weight_claim_reduction_speedup",
            "booleanity_address_phase_service_speedup": "hamming_weight_claim_reduction_service_speedup",
            "paired_booleanity_address_phase_speedups": "paired_hamming_weight_claim_reduction_speedups",
            "paired_booleanity_address_phase_service_speedups": "paired_hamming_weight_claim_reduction_service_speedups",
            "paired_booleanity_address_phase_fractional_improvements": "paired_hamming_weight_claim_reduction_fractional_improvements",
            "cpu_booleanity_address_phase_ms_samples": "cpu_hamming_weight_claim_reduction_ms_samples",
            "metal_booleanity_address_phase_ms_samples": "metal_hamming_weight_claim_reduction_ms_samples",
            "cpu_booleanity_address_phase_service_ms_samples": "cpu_hamming_weight_claim_reduction_service_ms_samples",
            "metal_booleanity_address_phase_service_ms_samples": "metal_hamming_weight_claim_reduction_service_ms_samples",
            "booleanity_address_phase_decision": "hamming_weight_claim_reduction_decision",
        }
        result["metrics"] = {
            metric_names.get(name, name): value
            for name, value in result["metrics"].items()
        }
        for pair in result["pairs"]:
            for backend, member_ns in (("optimized", 500), ("metal", 100)):
                arm = pair["arms"][backend]
                arm.pop("booleanity_address")
                arm.pop("booleanity_address_row_lifecycle")
                member, lifecycle = self.production_hamming_weight_member_fixture(
                    backend, member_ns, params
                )
                arm["hamming_weight"] = member
                arm["hamming_weight_row_lifecycle"] = lifecycle
        fingerprint = result["fingerprint"]
        fingerprint["local_kernel"] = "HammingWeightClaimReduction"
        for name in tuple(fingerprint):
            if name.startswith("booleanity_address_metal_"):
                fingerprint.pop(name)
        fingerprint.update(
            {
                "hamming_weight_metal_inner_log2": int(
                    params["JOLT_METAL_HAMMING_WEIGHT_INNER_LOG2"]
                ),
                "hamming_weight_metal_selectors_per_tile": int(
                    params["JOLT_METAL_HAMMING_WEIGHT_SELECTORS_PER_TILE"]
                ),
                "hamming_weight_metal_tile_threads": int(
                    params["JOLT_METAL_HAMMING_WEIGHT_TILE_THREADS"]
                ),
                "hamming_weight_metal_finalize_threads": int(
                    params["JOLT_METAL_HAMMING_WEIGHT_FINALIZE_THREADS"]
                ),
                "hamming_weight_metal_trace_cutoff_log2": int(
                    params["JOLT_METAL_HAMMING_WEIGHT_TRACE_CUTOFF_LOG2"]
                ),
            }
        )
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

    def test_booleanity_address_local_result_accepts_closed_contract(self) -> None:
        config, params, output = self.booleanity_address_local_contract_fixture()
        metal_autoresearch.validate_local_result_contract(config, output, params)
        passed, reason = metal_autoresearch.guards_pass(config, output)
        self.assertTrue(passed, reason)

    def test_booleanity_address_local_result_recomputes_raw_evidence(self) -> None:
        config, params, output = self.booleanity_address_local_contract_fixture()
        mutations = (
            (
                "paired speedup",
                lambda value: value["metrics"]["paired_speedups"].__setitem__(
                    0, 4.0
                ),
                "paired speedups",
            ),
            (
                "component overlap",
                lambda value: value["timings"][
                    "metal_dispatch_wall_ns_samples"
                ].__setitem__(0, 41),
                "not reconciled",
            ),
            (
                "GPU outside wall",
                lambda value: value["timings"][
                    "metal_gpu_active_ns_samples"
                ].__setitem__(0, 41),
                "not nested",
            ),
            (
                "partial bytes",
                lambda value: value["resources"].__setitem__(
                    "partial_owned_bytes", 1
                ),
                "resource geometry",
            ),
            (
                "threadgroup bytes",
                lambda value: value["pipelines"]["tile"].__setitem__(
                    "total_threadgroup_bytes", 1
                ),
                "tile pipeline",
            ),
            (
                "fingerprint parameter",
                lambda value: value["fingerprint"].__setitem__(
                    "selectors_per_tile", 5
                ),
                "fingerprint diverged",
            ),
            (
                "extra schema field",
                lambda value: value.__setitem__("unsupported", True),
                "contract is incomplete",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                tampered = copy.deepcopy(output)
                mutate(tampered)
                with self.assertRaisesRegex(ValueError, message):
                    metal_autoresearch.validate_local_result_contract(
                        config, tampered, params
                    )

    def test_booleanity_address_local_result_binds_all_five_parameters(self) -> None:
        config, params, output = self.booleanity_address_local_contract_fixture()
        for parameter in (
            "JOLT_METAL_BOOLEANITY_ADDRESS_INNER_LOG2",
            "JOLT_METAL_BOOLEANITY_ADDRESS_SELECTORS_PER_TILE",
            "JOLT_METAL_BOOLEANITY_ADDRESS_TILE_THREADS",
            "JOLT_METAL_BOOLEANITY_ADDRESS_FINALIZE_THREADS",
            "JOLT_METAL_BOOLEANITY_ADDRESS_TRACE_CUTOFF_LOG2",
        ):
            with self.subTest(parameter=parameter):
                tampered_params = dict(params)
                tampered_params[parameter] = str(int(tampered_params[parameter]) + 1)
                with self.assertRaisesRegex(ValueError, "fingerprint|geometry"):
                    metal_autoresearch.validate_local_result_contract(
                        config, output, tampered_params
                    )

    def test_booleanity_address_exact_sub_floor_result_remains_searchable(self) -> None:
        config, params, output = self.booleanity_address_local_contract_fixture()
        metal_samples = output["metrics"]["metal_member_ns_samples"]
        cpu_samples = [3 * value for value in metal_samples]
        cpu_prepare = output["timings"]["cpu_prepare_ns_samples"]
        cpu_host = output["timings"]["cpu_host_rounds_ns_samples"]
        cpu_unattributed = [
            total - prepare - host
            for total, prepare, host in zip(cpu_samples, cpu_prepare, cpu_host)
        ]
        output["metrics"].update(
            {
                "hybrid_speedup": 3.0,
                "ratio_of_member_medians": 3.0,
                "paired_speedups": [3.0] * len(metal_samples),
                "paired_speedup_mad": 0.0,
                "cpu_member_ns_samples": cpu_samples,
            }
        )
        output["timings"].update(
            {
                "cpu_member_median_ns": statistics.median(cpu_samples),
                "cpu_unattributed_median_ns": statistics.median(cpu_unattributed),
                "cpu_unattributed_ns_samples": cpu_unattributed,
            }
        )
        output["promotion"].update(
            {"speedup_eligible": False, "local_eligible": False}
        )

        metal_autoresearch.validate_local_result_contract(config, output, params)
        passed, reason = metal_autoresearch.guards_pass(config, output)
        self.assertTrue(passed, reason)
        self.assertTrue(output["all_exact"])

    def test_outer_remainder_template_and_closed_local_result(self) -> None:
        config, params, output = self.outer_remainder_local_contract_fixture()

        metal_autoresearch.validate_template(config, ROOT)
        metal_autoresearch.validate_new_run_template(config)
        metal_autoresearch.validate_local_result_contract(config, output, params)
        passed, reason = metal_autoresearch.guards_pass(config, output)
        self.assertTrue(passed, reason)
        observations = metal_autoresearch.evaluator_metric_observations(
            config, output
        )
        self.assertEqual(observations, output["metrics"]["paired_speedups"])
        self.assertEqual(
            statistics.median(observations), output["metrics"]["hybrid_speedup"]
        )

    def test_outer_remainder_internal_replication_is_closed(self) -> None:
        config, _, _ = self.outer_remainder_local_contract_fixture()
        mutations = (
            (
                "pair count",
                lambda value: value["evaluator"]["replication"].__setitem__(
                    "pairs", 3
                ),
                "internal evaluator replication|internal paired replication",
            ),
            (
                "controller baseline repeats",
                lambda value: value.__setitem__("baseline_repeats", 3),
                "exactly once",
            ),
            (
                "controller candidate repeats",
                lambda value: value.__setitem__("candidate_repeats", 3),
                "exactly once",
            ),
            (
                "missing replication",
                lambda value: value["evaluator"].pop("replication"),
                "baseline_repeats",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                tampered = copy.deepcopy(config)
                mutate(tampered)
                with self.assertRaisesRegex(ValueError, message):
                    metal_autoresearch.validate_template(tampered, ROOT)

    def test_outer_remainder_local_result_recomputes_raw_evidence(self) -> None:
        config, params, output = self.outer_remainder_local_contract_fixture()
        mutations = (
            (
                "paired speedup",
                lambda value: value["metrics"]["paired_speedups"].__setitem__(
                    0, 4.0
                ),
                "paired_speedups is inconsistent",
            ),
            (
                "raw member sample",
                lambda value: value["samples"][0]["optimized"].__setitem__(
                    "member_ns", 1
                ),
                "raw samples",
            ),
            (
                "fingerprint parameter",
                lambda value: value["fingerprint"].__setitem__(
                    "materialize_threads", 512
                ),
                "fingerprint does not match",
            ),
            (
                "source hash",
                lambda value: value["fingerprint"].__setitem__(
                    "source_sha256", "not-a-digest"
                ),
                "fingerprint diverged",
            ),
            (
                "GPU seconds",
                lambda value: value["resources"].__setitem__("gpu_seconds", 1.0),
                "resource accounting",
            ),
            (
                "full proof sample",
                lambda value: value["resources"][
                    "metal_full_prove_ns_samples"
                ].__setitem__(0, 0),
                "resource accounting",
            ),
            (
                "promotion eligibility",
                lambda value: value["promotion"].__setitem__("eligible", False),
                "promotion record",
            ),
            (
                "exactness guard",
                lambda value: value["guards"].__setitem__(
                    "round_topology_exact", False
                ),
                "exactness guards",
            ),
            (
                "extra schema field",
                lambda value: value.__setitem__("unsupported", True),
                "contract is incomplete",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                tampered = copy.deepcopy(output)
                mutate(tampered)
                with self.assertRaisesRegex(ValueError, message):
                    metal_autoresearch.validate_local_result_contract(
                        config, tampered, params
                    )

    def test_outer_remainder_local_result_binds_all_five_parameters(self) -> None:
        config, params, output = self.outer_remainder_local_contract_fixture()
        replacements = {
            "JOLT_METAL_OUTER_REMAINDER_MATERIALIZE_THREADS": "512",
            "JOLT_METAL_OUTER_REMAINDER_TRANSITION_THREADS": "256",
            "JOLT_METAL_OUTER_REMAINDER_OUTPUT_THREADS": "512",
            "JOLT_METAL_OUTER_REMAINDER_CUTOFF_LOG2": "17",
            "JOLT_METAL_OUTER_REMAINDER_TRACE_CUTOFF_LOG2": "19",
        }
        for parameter, replacement in replacements.items():
            with self.subTest(parameter=parameter):
                tampered_params = {**params, parameter: replacement}
                with self.assertRaisesRegex(ValueError, "fingerprint|geometry"):
                    metal_autoresearch.validate_local_result_contract(
                        config, output, tampered_params
                    )

    def test_outer_remainder_evaluator_artifact_is_controller_owned(self) -> None:
        config, params, output = self.outer_remainder_local_contract_fixture()
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            artifact_dir = log_dir / "candidate.artifacts"
            artifact_dir.mkdir()
            output["artifacts"] = str(artifact_dir)
            (artifact_dir / "result.json").write_text(json.dumps(output))
            completed = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(output) + "\n",
                stderr="",
            )
            with mock.patch.object(
                metal_autoresearch.subprocess, "run", return_value=completed
            ):
                observed, _ = metal_autoresearch.run_evaluator(
                    ROOT, config, params, log_dir, "candidate"
                )
            self.assertEqual(observed, output)

            output["artifacts"] = str(log_dir / "wrong")
            completed.stdout = json.dumps(output) + "\n"
            with mock.patch.object(
                metal_autoresearch.subprocess, "run", return_value=completed
            ):
                with self.assertRaisesRegex(ValueError, "wrong artifact directory"):
                    metal_autoresearch.run_evaluator(
                        ROOT, config, params, log_dir, "candidate"
                    )

    def test_outer_remainder_production_requires_equal_input_local_bar(self) -> None:
        config, _, _ = self.outer_remainder_local_contract_fixture()
        metal_autoresearch.validate_accepted_parent_for_production(config, 4.0)
        with self.assertRaisesRegex(ValueError, "full-protocol search gate"):
            metal_autoresearch.validate_accepted_parent_for_production(config, 3.99)

    def test_outer_remainder_production_member_is_closed_from_raw_evidence(self) -> None:
        from scripts.tests.test_metal_piop_eval import (
            complete_outer_remainder_trace,
            metal_piop_eval,
        )

        member = metal_piop_eval.outer_remainder_member_record(
            metal_piop_eval.outer_remainder_member_breakdown(
                complete_outer_remainder_trace(26, "metal"), "metal", 26
            )
        )
        observed = metal_autoresearch.validate_production_outer_remainder_member(
            member, "metal", 26, 16, 18
        )
        self.assertEqual(observed, member["member_ns"])

        member["resource_observation"]["sequence"][
            "round_device_buffer_allocations"
        ] = 1
        with self.assertRaisesRegex(ValueError, "storage evidence"):
            metal_autoresearch.validate_production_outer_remainder_member(
                member, "metal", 26, 16, 18
            )

    def test_hamming_weight_template_and_closed_local_result(self) -> None:
        config, params, output = self.hamming_weight_local_contract_fixture()
        metal_autoresearch.validate_template(config, ROOT)
        metal_autoresearch.validate_local_result_contract(config, output, params)
        passed, reason = metal_autoresearch.guards_pass(config, output)
        self.assertTrue(passed, reason)

    def test_hamming_weight_local_result_recomputes_specific_evidence(self) -> None:
        config, params, output = self.hamming_weight_local_contract_fixture()
        mutations = (
            (
                "throughput rate",
                lambda value: value["metrics"].__setitem__(
                    "metal_nonzero_recentered_contributions_per_second", 1.0
                ),
                "contributions_per_second",
            ),
            (
                "contribution count",
                lambda value: value["metrics"].__setitem__(
                    "nonzero_recentered_contributions", 1
                ),
                "contribution accounting",
            ),
            (
                "recenter guard",
                lambda value: value["guards"].__setitem__(
                    "recentered_bucket_zero_exact", False
                ),
                "protocol guards",
            ),
            (
                "owned bytes",
                lambda value: value["resources"].__setitem__(
                    "hamming_owned_device_bytes", 1
                ),
                "resource geometry",
            ),
            (
                "extra schema field",
                lambda value: value.__setitem__("unsupported", True),
                "contract is incomplete",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                tampered = copy.deepcopy(output)
                mutate(tampered)
                with self.assertRaisesRegex(ValueError, message):
                    metal_autoresearch.validate_local_result_contract(
                        config, tampered, params
                    )

    def test_hamming_weight_local_result_binds_all_five_parameters(self) -> None:
        config, params, output = self.hamming_weight_local_contract_fixture()
        for parameter in (
            "JOLT_METAL_HAMMING_WEIGHT_INNER_LOG2",
            "JOLT_METAL_HAMMING_WEIGHT_SELECTORS_PER_TILE",
            "JOLT_METAL_HAMMING_WEIGHT_TILE_THREADS",
            "JOLT_METAL_HAMMING_WEIGHT_FINALIZE_THREADS",
            "JOLT_METAL_HAMMING_WEIGHT_TRACE_CUTOFF_LOG2",
        ):
            with self.subTest(parameter=parameter):
                tampered_params = dict(params)
                tampered_params[parameter] = str(int(tampered_params[parameter]) + 1)
                with self.assertRaisesRegex(ValueError, "fingerprint|geometry"):
                    metal_autoresearch.validate_local_result_contract(
                        config, output, tampered_params
                    )

    def test_hamming_weight_exact_sub_floor_result_remains_searchable(self) -> None:
        config, params, output = self.hamming_weight_local_contract_fixture()
        metrics = output["metrics"]
        metal_samples = metrics["metal_member_ns_samples"]
        cpu_samples = [3 * value for value in metal_samples]
        cpu_prepare = output["timings"]["cpu_prepare_ns_samples"]
        cpu_host = output["timings"]["cpu_host_rounds_ns_samples"]
        cpu_unattributed = [
            total - prepare - host
            for total, prepare, host in zip(cpu_samples, cpu_prepare, cpu_host)
        ]
        metrics.update(
            {
                "hybrid_speedup": 3.0,
                "ratio_of_member_medians": 3.0,
                "paired_speedups": [3.0] * len(metal_samples),
                "paired_speedup_mad": 0.0,
                "cpu_member_ns_samples": cpu_samples,
                "cpu_selector_row_opportunities_per_second": metrics[
                    "selector_row_opportunities"
                ]
                * 1e9
                / statistics.median(cpu_samples),
                "cpu_nonzero_recentered_contributions_per_second": metrics[
                    "nonzero_recentered_contributions"
                ]
                * 1e9
                / statistics.median(cpu_samples),
            }
        )
        output["timings"].update(
            {
                "cpu_member_median_ns": statistics.median(cpu_samples),
                "cpu_unattributed_median_ns": statistics.median(cpu_unattributed),
                "cpu_unattributed_ns_samples": cpu_unattributed,
            }
        )
        output["promotion"].update(
            {"speedup_eligible": False, "local_eligible": False}
        )

        metal_autoresearch.validate_local_result_contract(config, output, params)
        passed, reason = metal_autoresearch.guards_pass(config, output)
        self.assertTrue(passed, reason)

    def test_instruction_input_local_result_accepts_closed_contract(self) -> None:
        config, params, output = self.instruction_input_local_contract_fixture()
        metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_instruction_input_run_evaluator_accepts_schema_three(self) -> None:
        config, params, output = self.instruction_input_v3_local_contract_fixture()
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(output) + "\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with mock.patch.object(
                metal_autoresearch.subprocess, "run", return_value=completed
            ):
                parsed, _ = metal_autoresearch.run_evaluator(
                    path, config, params, path, "instruction-input-v3"
                )
        self.assertEqual(parsed["schema_version"], 3)

    def test_instruction_input_template_pins_schema_four(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        template["evaluator"]["result_schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema version mismatches"):
            metal_autoresearch.validate_template(template)

    def test_instruction_input_template_freezes_architecture_evidence(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        evidence = template["architecture_phase"]["baseline_evidence"]
        template["scope"]["frozen"].remove(evidence)
        with self.assertRaisesRegex(ValueError, "baseline evidence must be frozen"):
            metal_autoresearch.validate_template(template)

    def test_new_instruction_input_run_requires_architecture_evidence(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        del template["architecture_phase"]

        with self.assertRaisesRegex(ValueError, "architecture baseline"):
            metal_autoresearch.validate_new_run_template(template)

    def test_instruction_input_architecture_evidence_recomputes_ratios(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        source = ROOT / template["architecture_phase"]["baseline_evidence"]
        evidence = metal_autoresearch.read_json(source)
        evidence["samples"][0]["service_speedup"] += 0.25
        encoded = metal_autoresearch.canonical_json(evidence)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "architecture-evidence.json"
            path.write_bytes(encoded)
            old_path = template["architecture_phase"]["baseline_evidence"]
            template["architecture_phase"]["baseline_evidence"] = path.name
            template["architecture_phase"]["baseline_evidence_sha256"] = (
                metal_autoresearch.sha256(encoded)
            )
            template["scope"]["frozen"].remove(old_path)
            template["scope"]["frozen"].append(path.name)

            with self.assertRaisesRegex(ValueError, "sample ratio"):
                metal_autoresearch.validate_template(template, root)

    def test_instruction_input_architecture_evidence_binds_fixed_bar(self) -> None:
        original = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        source = ROOT / original["architecture_phase"]["baseline_evidence"]

        for threshold, clears, message in (
            (3.0, True, "failed local gate"),
            (5.0, False, "template contract"),
        ):
            with self.subTest(threshold=threshold):
                template = copy.deepcopy(original)
                evidence = metal_autoresearch.read_json(source)
                evidence["decision"]["minimum_speedup"] = threshold
                if clears:
                    evidence["decision"]["clears_speedup"] = True
                    evidence["decision"]["clears_order_strata"] = True
                    evidence["decision"]["clears_fractional_improvement"] = True
                    evidence["decision"]["clears"] = True
                    evidence["guards"]["instruction_input_local_gate"] = True
                encoded = metal_autoresearch.canonical_json(evidence)

                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path = root / "architecture-evidence.json"
                    path.write_bytes(encoded)
                    old_path = template["architecture_phase"]["baseline_evidence"]
                    template["architecture_phase"]["baseline_evidence"] = path.name
                    template["architecture_phase"]["baseline_evidence_sha256"] = (
                        metal_autoresearch.sha256(encoded)
                    )
                    template["scope"]["frozen"].remove(old_path)
                    template["scope"]["frozen"].append(path.name)

                    with self.assertRaisesRegex(ValueError, message):
                        metal_autoresearch.validate_template(template, root)

    def test_instruction_input_architecture_evidence_requires_other_guards(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        source = ROOT / template["architecture_phase"]["baseline_evidence"]
        evidence = metal_autoresearch.read_json(source)
        evidence["guards"]["instruction_input_readback_exact"] = False
        encoded = metal_autoresearch.canonical_json(evidence)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "architecture-evidence.json"
            path.write_bytes(encoded)
            old_path = template["architecture_phase"]["baseline_evidence"]
            template["architecture_phase"]["baseline_evidence"] = path.name
            template["architecture_phase"]["baseline_evidence_sha256"] = (
                metal_autoresearch.sha256(encoded)
            )
            template["scope"]["frozen"].remove(old_path)
            template["scope"]["frozen"].append(path.name)

            with self.assertRaisesRegex(ValueError, "guards are invalid"):
                metal_autoresearch.validate_template(template, root)

    def test_instruction_input_architecture_evidence_binds_command_contract(self) -> None:
        original = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        source = ROOT / original["architecture_phase"]["baseline_evidence"]

        workload = copy.deepcopy(original)
        workload_command = workload["final_validation"]["production_gate"][
            "evaluator"
        ]["command"]
        workload_command[workload_command.index("--workload") + 1] = "unrelated"
        evidence = metal_autoresearch.read_json(source)
        evidence["launch"]["workload"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "template contract"):
            metal_autoresearch.validate_instruction_input_architecture_evidence(
                evidence, workload
            )

        repeats = copy.deepcopy(original)
        repeats_command = repeats["final_validation"]["production_gate"]["evaluator"][
            "command"
        ]
        repeats_command[repeats_command.index("--repeats") + 1] = "7"
        evidence = metal_autoresearch.read_json(source)
        with self.assertRaisesRegex(ValueError, "template contract"):
            metal_autoresearch.validate_instruction_input_architecture_evidence(
                evidence, repeats
            )

    def test_instruction_input_architecture_evidence_rejects_json_type_aliases(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        source = ROOT / template["architecture_phase"]["baseline_evidence"]

        revision = metal_autoresearch.read_json(source)
        revision["candidate"]["git_revision"] = int("1" * 40)
        with self.assertRaisesRegex(ValueError, "candidate record is invalid"):
            metal_autoresearch.validate_instruction_input_architecture_evidence(
                revision, template
            )

        pair = metal_autoresearch.read_json(source)
        pair["samples"][0]["pair"] = True
        with self.assertRaisesRegex(ValueError, "sample order is invalid"):
            metal_autoresearch.validate_instruction_input_architecture_evidence(
                pair, template
            )

    def test_instruction_input_architecture_evidence_requires_nested_timings(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        source = ROOT / template["architecture_phase"]["baseline_evidence"]

        for field, value_field in (
            ("cpu_service_ns", "cpu_piop_ns"),
            ("cpu_piop_plus_prepare_ns", "cpu_piop_ns"),
        ):
            with self.subTest(field=field):
                evidence = metal_autoresearch.read_json(source)
                sample = evidence["samples"][0]
                sample[field] = (
                    sample[value_field] + 1
                    if field == "cpu_service_ns"
                    else sample[value_field] - 1
                )
                with self.assertRaisesRegex(ValueError, "timing relationship"):
                    metal_autoresearch.validate_instruction_input_architecture_evidence(
                        evidence, template
                    )

    def test_instruction_input_local_result_keeps_v2_history_readable(self) -> None:
        config, params, output = self.instruction_input_v2_local_contract_fixture()
        metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_instruction_input_local_result_keeps_v3_history_readable(self) -> None:
        config, params, output = self.instruction_input_v3_local_contract_fixture()
        metal_autoresearch.validate_local_result_contract(config, output, params)

    def test_instruction_input_run_evaluator_keeps_schema_two_readable(self) -> None:
        config, params, output = self.instruction_input_v2_local_contract_fixture()
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(output) + "\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with mock.patch.object(
                metal_autoresearch.subprocess, "run", return_value=completed
            ):
                parsed, _ = metal_autoresearch.run_evaluator(
                    path, config, params, path, "instruction-input-v2"
                )
        self.assertEqual(parsed["schema_version"], 2)

    def test_instruction_input_v3_reference_derives_from_a2_samples(self) -> None:
        samples = [
            861_240_375,
            778_143_459,
            777_036_000,
            772_410_208,
            774_448_458,
            871_609_708,
            775_459_791,
            770_859_125,
            771_844_000,
            799_733_000,
            815_094_584,
            789_479_208,
            791_994_542,
            794_477_583,
            801_751_375,
            841_780_209,
            820_155_667,
            814_395_125,
            825_222_917,
            849_530_667,
            865_644_958,
            842_593_708,
            854_701_250,
            846_923_666,
            851_209_375,
        ]
        encoded = json.dumps(samples, separators=(",", ":")).encode()
        self.assertEqual(
            statistics.median(samples),
            metal_autoresearch.INSTRUCTION_INPUT_V3_CPU_REFERENCE_NS,
        )
        self.assertEqual(
            metal_autoresearch.sha256(encoded),
            "59f9946b7d1a3c05d3094528e853d2228ae5ec0d94a5dae2c63d5713a560a966",
        )

    def test_instruction_input_v4_primary_ignores_live_cpu_drift(self) -> None:
        config, params, output = self.instruction_input_local_contract_fixture()
        primary = output["metrics"]["frozen_cpu_reference_ratio"]
        cpu_samples = [5_000 + 1_000 * index for index in range(5)]
        hybrid_samples = output["metrics"]["hybrid_ns_samples"]
        resident_samples = output["metrics"]["resident_ns_samples"]
        paired = [
            cpu / hybrid for cpu, hybrid in zip(cpu_samples, hybrid_samples)
        ]
        resident_paired = [
            cpu / resident for cpu, resident in zip(cpu_samples, resident_samples)
        ]
        output["metrics"]["cpu_ns_samples"] = cpu_samples
        output["metrics"]["paired_hybrid_speedups"] = paired
        output["metrics"]["paired_resident_speedups"] = resident_paired
        output["metrics"]["hybrid_speedup"] = statistics.median(paired)
        output["metrics"]["resident_speedup"] = statistics.median(resident_paired)
        output["metrics"]["cpu_million_rows_per_second"] = (
            output["workload"]["rows"]
            / (statistics.median(cpu_samples) / 1e9)
            / 1e6
        )
        output["timings"]["cpu_median_seconds"] = (
            statistics.median(cpu_samples) / 1e9
        )

        metal_autoresearch.validate_local_result_contract(config, output, params)
        self.assertEqual(output["metrics"]["frozen_cpu_reference_ratio"], primary)

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
                "frozen reference sample",
                lambda output: output["metrics"][
                    "paired_frozen_cpu_reference_ratios"
                ].__setitem__(0, 99.0),
                "paired_frozen_cpu_reference_ratios",
            ),
            (
                "frozen reference median",
                lambda output: output["metrics"].__setitem__(
                    "frozen_cpu_reference_ratio", 99.0
                ),
                "frozen_cpu_reference_ratio",
            ),
            (
                "frozen reference fingerprint",
                lambda output: output["fingerprint"].__setitem__(
                    "frozen_cpu_reference_ns", 1
                ),
                "fingerprint does not match frozen_cpu_reference_ns",
            ),
            (
                "frozen reference workload",
                lambda output: output["workload"].__setitem__(
                    "frozen_cpu_reference_provenance", "changed"
                ),
                "workload fingerprint diverged",
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
                    "timed_gpu_active_total_seconds", 99.0
                ),
                "GPU-active total",
            ),
            (
                "residency warmup",
                lambda output: output["timings"].__setitem__(
                    "residency_warmup_gpu_active_ns", 999
                ),
                "residency warmup timing",
            ),
            (
                "evaluator GPU total",
                lambda output: output["timings"].__setitem__(
                    "evaluator_gpu_active_total_seconds", 99.0
                ),
                "total GPU-active time",
            ),
            (
                "GPU resource",
                lambda output: output["resources"].__setitem__(
                    "gpu_seconds", 99.0
                ),
                "GPU resource timing",
            ),
            (
                "GPU resource omits warmup",
                lambda output: output["resources"].__setitem__(
                    "gpu_seconds",
                    output["timings"]["timed_gpu_active_total_seconds"],
                ),
                "GPU resource timing",
            ),
            (
                "resource total",
                lambda output: output["resources"].__setitem__(
                    "metal_phase_persistent_modeled_bytes", 1
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
        output["fingerprint"]["arm_schedule"] = ["metal", "cpu"]
        with self.assertRaisesRegex(ValueError, "phased schedule"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

        config, params, output = self.instruction_input_local_contract_fixture()
        output["fingerprint"]["residency_warmup_runs"] = 0
        with self.assertRaisesRegex(ValueError, "warmup fingerprint"):
            metal_autoresearch.validate_local_result_contract(config, output, params)

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

    def test_instruction_input_v4_pins_phase_machine(self) -> None:
        config, params, output = self.instruction_input_local_contract_fixture()
        config["fingerprint"] = {"evaluator": copy.deepcopy(output["fingerprint"])}
        output["fingerprint"]["device"] = "different Metal device"
        with self.assertRaisesRegex(ValueError, "phase machine diverged at device"):
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

        template = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        template["evaluator"]["env"]["JOLT_METAL_EVAL_CPU_REFERENCE_NS"] = "1"
        with self.assertRaisesRegex(ValueError, "frozen a2 baseline"):
            metal_autoresearch.validate_template(template)

        template = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        template["metric"]["target"] = 5.0
        template["metric"]["unit"] = "x"
        with self.assertRaisesRegex(ValueError, "relative-only search proxy"):
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

    def test_schema_one_init_rejects_before_creating_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "root"
            root.mkdir()
            run_dir = temporary / "run"
            template = {"schema_version": 1}
            with mock.patch.object(
                metal_autoresearch,
                "read_json",
                return_value=template,
            ), mock.patch.object(
                metal_autoresearch, "validate_template"
            ), mock.patch.object(
                metal_autoresearch,
                "snapshot_paths",
            ) as snapshot:
                with self.assertRaisesRegex(ValueError, "existing-run-only"):
                    metal_autoresearch.command_init(
                        SimpleNamespace(
                            root=root,
                            template=temporary / "template.json",
                            run_dir=run_dir,
                        )
                    )
                snapshot.assert_not_called()
                self.assertFalse(run_dir.exists())

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
        template["final_validation"]["production_gate"]["evaluator"][
            "schema_version"
        ] = 5
        result = {
            "schema_version": template["final_validation"]["production_gate"][
                "evaluator"
            ]["schema_version"],
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
                "compact row projection",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input_row_lifecycle"
                ]["row_production"].__setitem__("full_domain_copy_bytes", 48),
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
            (
                "charged submit span",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ]["resource_observation"]["native_primer"]["timings"].__setitem__(
                    "submit_span_wall_ns", 2
                ),
                "native primer record",
            ),
            (
                "join exceeds round zero",
                lambda value: (
                    value["pairs"][0]["arms"]["metal"]["instruction_input"][
                        "resource_observation"
                    ]["native_primer"]["timings"].__setitem__("join_wall_ns", 2),
                    value["pairs"][0]["arms"]["metal"]["instruction_input"][
                        "resource_observation"
                    ]["native_primer"]["timings"].__setitem__(
                        "lifecycle_wall_ns", 103
                    ),
                ),
                "native primer record",
            ),
            (
                "initialization exceeds preparation",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ]["resource_observation"]["storage_initialization"].__setitem__(
                    "wall_ns", 22
                ),
                "startup timing",
            ),
            (
                "primer lifecycle exceeds PIOP",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "instruction_input"
                ]["resource_observation"]["native_primer"]["timings"].__setitem__(
                    "lifecycle_wall_ns", 1_000
                ),
                "startup timing",
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

        impossible_service = copy.deepcopy(result)
        impossible_service["pairs"][0]["arms"]["optimized"]["piop_ns"] = 400
        impossible_service["metrics"]["paired_speedups"][0] = 2.0
        impossible_service["metrics"]["cpu_piop_ms_samples"][0] = 400 / 1e6
        impossible_service["metrics"][
            "paired_speedups_with_backend_witness_prepare"
        ][0] = (400 + 10) / (200 + 20)
        with self.assertRaisesRegex(ValueError, "service timing exceeds"):
            metal_autoresearch.validate_production_result(
                config, impossible_service, "abc", params, True
            )

        wrong_prepare_median = copy.deepcopy(result)
        wrong_prepare_median["metrics"]["metal_backend_witness_prepare_ms"] = 1.0
        with self.assertRaisesRegex(ValueError, "median summary"):
            metal_autoresearch.validate_production_result(
                config, wrong_prepare_median, "abc", params, True
            )

    def test_schema_five_instruction_input_result_remains_readable(self) -> None:
        config, params, result = self.production_instruction_input_result_fixture()
        gate = config["final_validation"]["production_gate"]
        gate["evaluator"]["schema_version"] = 5
        for guard in (
            "instruction_input_minimal_initialization_exact",
            "instruction_input_storage_buffers_stable",
            "instruction_input_native_primer_exact_and_protocol_inert",
            "instruction_input_compact_rows_direct_and_stable",
        ):
            gate["required_guards"].remove(guard)
        result["schema_version"] = 5
        result["fingerprint"].pop("instruction_input_storage_initialization")
        result["fingerprint"].pop("instruction_input_native_primer")
        for pair in result["pairs"]:
            pair["arms"]["metal"]["instruction_input_row_lifecycle"] = {
                "kind": "metal_resident",
                "rows": 1 << 26,
                "row_bytes": 160,
                "prepare_storage_id": 202,
                "stage1_storage_id": 202,
                "stage3_storage_id": 202,
            }
            for backend in ("optimized", "metal"):
                member = pair["arms"][backend]["instruction_input"]
                if backend == "metal":
                    member["member_ns"] += member["prefetch_submit_ns"]
                    member["output_claims_ns"] += member["prefetch_submit_ns"]
                member.pop("prefetch_submit_ns")
                member.pop("service_ns")
                for phase in (
                    "storage_initialize",
                    "storage_initialize_complete",
                    "native_primer_submit",
                    "native_primer_join",
                    "native_primer_complete",
                ):
                    member["metal_counts"].pop(phase)
                if backend == "metal":
                    member["resource_observation"].pop("storage_initialization")
                    member["resource_observation"].pop("native_primer")
        evidence = metal_autoresearch.validate_production_result(
            config, result, "abc", params, True
        )
        self.assertEqual(evidence["pairs"], 5)

    def test_schema_six_instruction_input_result_remains_readable(self) -> None:
        config, params, result = self.production_instruction_input_result_fixture()
        gate = config["final_validation"]["production_gate"]
        gate["evaluator"]["schema_version"] = 6
        gate["required_guards"].remove(
            "instruction_input_compact_rows_direct_and_stable"
        )
        result["schema_version"] = 6
        for pair in result["pairs"]:
            pair["arms"]["metal"]["instruction_input_row_lifecycle"] = {
                "kind": "metal_resident",
                "rows": 1 << 26,
                "row_bytes": 160,
                "prepare_storage_id": 202,
                "stage1_storage_id": 202,
                "stage3_storage_id": 202,
            }
        evidence = metal_autoresearch.validate_production_result(
            config, result, "abc", params, True
        )
        self.assertEqual(evidence["pairs"], 5)

    def test_new_instruction_input_run_requires_current_production_schema(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT / "crates/jolt-kernels/autoresearch/instruction_input.template.json"
        )
        metal_autoresearch.validate_new_run_template(template)
        template["final_validation"]["production_gate"]["evaluator"][
            "schema_version"
        ] = 5
        with self.assertRaisesRegex(ValueError, "current production result schema"):
            metal_autoresearch.validate_new_run_template(template)

    def test_production_v3_defers_absolute_bar_to_actual_pairs(self) -> None:
        config, _, _ = self.production_instruction_input_result_fixture()
        metal_autoresearch.validate_accepted_parent_for_production(config, 0.01)

        config["evaluator"]["result_contract"] = "instruction_input_v2"
        metal_autoresearch.validate_accepted_parent_for_production(config, 4.0)
        with self.assertRaisesRegex(ValueError, "full-protocol search gate"):
            metal_autoresearch.validate_accepted_parent_for_production(config, 3.99)

    def test_booleanity_production_requires_equal_input_local_bar(self) -> None:
        config, _, _ = self.production_booleanity_address_result_fixture()
        metal_autoresearch.validate_accepted_parent_for_production(config, 4.0)
        with self.assertRaisesRegex(ValueError, "full-protocol search gate"):
            metal_autoresearch.validate_accepted_parent_for_production(config, 3.99)

    def test_hamming_weight_production_requires_equal_input_local_bar(self) -> None:
        config, _, _ = self.hamming_weight_local_contract_fixture()
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
        result["metrics"][
            "paired_instruction_input_kernel_service_fractional_improvements"
        ] = [1.0 - 1.0 / speedup for speedup in speedups]
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
            member["service_ns"] = member_ns + member["prefetch_submit_ns"]
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

    def test_production_instruction_input_gate_recomputes_full_decision(self) -> None:
        for field, value in (
            ("median_fractional_improvement", 0.79),
            ("mad_fractional_improvement", 0.01),
            ("cpu_member_ms_median", 99.0),
            ("clears_noise", False),
            ("clears_fractional_improvement", False),
        ):
            with self.subTest(field=field):
                config, params, result = (
                    self.production_instruction_input_result_fixture()
                )
                result["metrics"][
                    "instruction_input_kernel_service_decision"
                ][field] = value
                with self.assertRaisesRegex(ValueError, "raw-pair decision"):
                    metal_autoresearch.validate_production_result(
                        config, result, "abc", params, True
                    )

        config, params, result = self.production_instruction_input_result_fixture()
        result["metrics"][
            "paired_instruction_input_kernel_service_fractional_improvements"
        ][0] = 0.5
        with self.assertRaisesRegex(ValueError, "fractional improvements"):
            metal_autoresearch.validate_production_result(
                config, result, "abc", params, True
            )

    def test_production_booleanity_address_gate_recomputes_raw_members(self) -> None:
        config, params, result = self.production_booleanity_address_result_fixture()
        evidence = metal_autoresearch.validate_production_result(
            config, result, "abc", params, True
        )
        self.assertEqual(evidence["metric"], "booleanity_address_phase_speedup")
        self.assertEqual(evidence["metric_value"], 5.0)
        self.assertEqual(evidence["optimized_first_median_speedup"], 5.0)
        self.assertEqual(evidence["metal_first_median_speedup"], 5.0)

        mutations = (
            (
                "readback bytes",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "booleanity_address"
                ]["resource_observation"]["readback"].__setitem__("bytes", 1),
                "readback",
            ),
            (
                "second command buffer",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "booleanity_address"
                ]["resource_observation"]["dispatch"].__setitem__(
                    "command_buffers", 2
                ),
                "dispatch",
            ),
            (
                "row allocation in stage6a",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "booleanity_address_row_lifecycle"
                ]["stage6a"].__setitem__("row_allocations", 1),
                "row lifecycle",
            ),
            (
                "unbound selector width",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "booleanity_address"
                ]["resource_observation"]["sequence"].__setitem__(
                    "requested_selectors_per_tile", 5
                ),
                "sequence",
            ),
            (
                "reported local sample",
                lambda value: value["metrics"][
                    "metal_booleanity_address_phase_ms_samples"
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

    def test_production_hamming_weight_gate_recomputes_terminal_raw_members(
        self,
    ) -> None:
        config, params, result = self.production_hamming_weight_result_fixture()
        evidence = metal_autoresearch.validate_production_result(
            config, result, "abc", params, True
        )
        self.assertEqual(evidence["metric"], "hamming_weight_claim_reduction_speedup")
        self.assertEqual(evidence["metric_value"], 5.0)
        self.assertEqual(evidence["optimized_first_median_speedup"], 5.0)
        self.assertEqual(evidence["metal_first_median_speedup"], 5.0)

        mutations = (
            (
                "readback bytes",
                lambda value: value["pairs"][0]["arms"]["metal"]["hamming_weight"][
                    "resource_observation"
                ]["readback"].__setitem__("bytes", 1),
                "readback",
            ),
            (
                "wrong K",
                lambda value: value["pairs"][0]["arms"]["metal"]["hamming_weight"][
                    "resource_observation"
                ]["sequence"].__setitem__("k", 16),
                "sequence",
            ),
            (
                "wrong terminal storage",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "hamming_weight_row_lifecycle"
                ].__setitem__("stage7_storage_id", 402),
                "row lifecycle",
            ),
            (
                "terminal upload",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "hamming_weight_row_lifecycle"
                ]["stage7"].__setitem__("row_upload_bytes", 1),
                "row lifecycle",
            ),
            (
                "terminal carry retained",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "hamming_weight_row_lifecycle"
                ].__setitem__("terminal_carry_removed", False),
                "row lifecycle",
            ),
            (
                "extra lifecycle field",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "hamming_weight_row_lifecycle"
                ].__setitem__("uncontracted", True),
                "lifecycle record is incomplete",
            ),
            (
                "extra member field",
                lambda value: value["pairs"][0]["arms"]["metal"][
                    "hamming_weight"
                ].__setitem__("uncontracted", True),
                "member record is incomplete",
            ),
            (
                "reported local sample",
                lambda value: value["metrics"][
                    "metal_hamming_weight_claim_reduction_ms_samples"
                ].__setitem__(0, 1.0),
                "sample summary",
            ),
            (
                "reported service sample",
                lambda value: value["metrics"][
                    "cpu_hamming_weight_claim_reduction_service_ms_samples"
                ].__setitem__(0, 1.0),
                "service sample summary",
            ),
            (
                "reported improvement",
                lambda value: value["metrics"][
                    "paired_hamming_weight_claim_reduction_fractional_improvements"
                ].__setitem__(0, 0.5),
                "fractional improvements",
            ),
            (
                "forged decision",
                lambda value: value["metrics"][
                    "hamming_weight_claim_reduction_decision"
                ].__setitem__("clears_noise", False),
                "raw-pair decision",
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

    def test_production_hamming_weight_gate_requires_both_order_strata(self) -> None:
        config, params, result = self.production_hamming_weight_result_fixture()
        speedups = [5.0, 3.0, 5.0, 3.0, 5.0]
        metrics = result["metrics"]
        metrics["paired_hamming_weight_claim_reduction_speedups"] = speedups
        metrics["paired_hamming_weight_claim_reduction_fractional_improvements"] = [
            1.0 - 1.0 / speedup for speedup in speedups
        ]
        metrics["cpu_hamming_weight_claim_reduction_ms_samples"] = [
            speedup * 100 / 1e6 for speedup in speedups
        ]
        metrics["cpu_hamming_weight_claim_reduction_service_ms_samples"] = [
            (speedup * 100 + 40) / 1e6 for speedup in speedups
        ]
        service_speedups = [(speedup * 100 + 40) / 100 for speedup in speedups]
        metrics["paired_hamming_weight_claim_reduction_service_speedups"] = (
            service_speedups
        )
        metrics["hamming_weight_claim_reduction_service_speedup"] = statistics.median(
            service_speedups
        )
        metrics["hamming_weight_claim_reduction_decision"].update(
            {
                "median_speedup": 5.0,
                "optimized_first_median_speedup": 5.0,
                "metal_first_median_speedup": 3.0,
                "clears_order_strata": True,
                "clears": True,
            }
        )
        for pair, speedup in zip(result["pairs"], speedups):
            member = pair["arms"]["optimized"]["hamming_weight"]
            normalized_member_ns = round(speedup * 100)
            member["normalized_member_ns"] = normalized_member_ns
            member["member_ns"] = normalized_member_ns + member["row_source_ns"]
            member["output_claims_ns"] = (
                member["member_ns"]
                - member["prepare_ns"]
                - member["rounds_total_ns"]
                - member["host_fiat_shamir_total_ns"]
                - member["finish_ns"]
            )
        with self.assertRaisesRegex(ValueError, "order stratum"):
            metal_autoresearch.validate_production_result(
                config, result, "abc", params, True
            )

    def test_booleanity_address_template_closes_schema_scope_and_bindings(self) -> None:
        template = metal_autoresearch.read_json(
            ROOT
            / "crates/jolt-kernels/autoresearch/booleanity_address.template.json"
        )
        metal_autoresearch.validate_template(template)
        metal_autoresearch.validate_new_run_template(template)
        metal_autoresearch.validate_params(template, template["baseline_params"])
        self.assertEqual(
            template["scope"]["editable"],
            ["crates/jolt-kernels/src/metal/solinas/booleanity_address.metal"],
        )
        gate = template["final_validation"]["production_gate"]
        self.assertEqual(gate["evaluator"]["schema_version"], 7)
        self.assertEqual(
            {
                binding["parameter"]
                for binding in gate["evaluator"]["parameter_bindings"]
            },
            metal_autoresearch.PRODUCTION_LOCAL_KERNELS[
                "BooleanityAddressPhase"
            ]["parameters"],
        )

        missing_guard = copy.deepcopy(template)
        missing_guard["final_validation"]["production_gate"][
            "required_guards"
        ].remove("booleanity_address_readback_exact")
        with self.assertRaisesRegex(ValueError, "omits mandatory"):
            metal_autoresearch.validate_template(missing_guard)

        widened_scope = copy.deepcopy(template)
        widened_scope["scope"]["editable"].append(
            "crates/jolt-kernels/src/lib.rs"
        )
        with self.assertRaisesRegex(ValueError, "shader-only"):
            metal_autoresearch.validate_template(widened_scope)

        stale_schema = copy.deepcopy(template)
        stale_schema["final_validation"]["production_gate"]["evaluator"][
            "schema_version"
        ] = 6
        with self.assertRaisesRegex(ValueError, "current production result schema"):
            metal_autoresearch.validate_new_run_template(stale_schema)

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

    def test_production_template_closes_local_metric_and_parameter_bindings(self) -> None:
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
        ] = 8
        with self.assertRaisesRegex(ValueError, "must be 4, 5, 6, or 7"):
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
