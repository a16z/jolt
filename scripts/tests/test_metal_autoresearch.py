import copy
import importlib.util
import io
import json
import os
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

    def test_result_parser_uses_last_schema_record(self) -> None:
        output = "compile noise\n{\"schema_version\": 1, \"metrics\": {\"x\": 2}}\n"
        self.assertEqual(metal_autoresearch.parse_result(output)["metrics"]["x"], 2)

    def test_schema_five_parser_requires_one_result_record(self) -> None:
        record = '{"schema_version": 5, "kernel": "akita_piop"}'
        self.assertEqual(
            metal_autoresearch.parse_unique_schema_result(record, 5)["kernel"],
            "akita_piop",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            metal_autoresearch.parse_unique_schema_result(f"{record}\n{record}", 5)

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
                "paired_bytecode_read_raf_cycle_speedups": [4.5] * 5,
                "bytecode_read_raf_cycle_decision": {
                    "clears": True,
                    "median_speedup": 4.5,
                },
            },
            "pairs": [
                {
                    "index": index + 1,
                    "order": ["optimized", "metal"]
                    if index % 2 == 0
                    else ["metal", "optimized"],
                    "arms": {
                        "optimized": {"bytecode": {"member_ns": 450}},
                        "metal": {"bytecode": {"member_ns": 100}},
                    },
                }
                for index in range(5)
            ],
            "fingerprint": {
                "git_revision": "abc",
                "worktree_dirty": False,
                "local_kernel": "BytecodeReadRafCycle",
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
            config, result, "abc", {}, True
        )
        self.assertEqual(
            evidence["paired_metric"], "paired_bytecode_read_raf_cycle_speedups"
        )

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
