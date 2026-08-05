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
