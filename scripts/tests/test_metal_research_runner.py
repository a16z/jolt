import copy
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest import mock

from scripts.metal_research.attempt import EvaluatorLeaseTimeout, run_attempt
from scripts.metal_research import attempt as attempt_runtime
from scripts.metal_research import process_wrapper
from scripts.metal_research.paired import paired_summary
from scripts.metal_research.results import adapt_result, validate_tier_result
from scripts.metal_research import runner
from scripts import metal_autoresearch


ROOT = Path(__file__).resolve().parents[2]


def evaluator() -> dict[str, object]:
    return {
        "command": ["unused"],
        "env": {"JOLT_METAL_DECLARED": "yes"},
        "result_adapter": "test",
        "timeout_seconds": 30,
    }


@contextmanager
def fake_lease(
    _owner: dict[str, object], _timeout_seconds: object = None
):
    telemetry = {
        "queue_wait_seconds": 2.0,
        "exclusive_lease_seconds": 0.0,
        "lock_fd": 99,
    }
    yield telemetry
    telemetry["exclusive_lease_seconds"] = 11.0


def descriptor(warmups: int = 0, pairs: int = 5) -> dict[str, object]:
    return {
        "mode": "internal_paired",
        "included_pairs": pairs,
        "excluded_warmup_pairs": warmups,
        "order_policy": "alternating",
        "first_order": ["control", "treatment"],
        "minimum_pairs_per_order_stratum": 2 if pairs >= 5 else 1,
        "input_policy": {
            "within_pair": "identical",
            "across_pairs": "same_fixture",
        },
        "effect": "control_over_treatment",
        "aggregate": "median_of_pair_effects",
    }


class AttemptTests(unittest.TestCase):
    def test_tracked_attempt_fails_closed_without_inherited_lease_fd(self) -> None:
        @contextmanager
        def lease_without_fd(*_args: object, **_kwargs: object):
            yield {
                "queue_wait_seconds": 0.0,
                "exclusive_lease_seconds": 0.0,
            }

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.metal_research.attempt.evaluator_lease", lease_without_fd
        ), mock.patch(
            "scripts.metal_research.attempt.subprocess.Popen"
        ) as popen:
            evaluation_dir = Path(directory) / "evaluation"
            observed, output = run_attempt(
                Path(directory),
                evaluator(),
                {},
                evaluation_dir,
                "representative",
                process_tracking={
                    "evaluation_id": "evaluation-001",
                    "launch_token": "launch-token",
                    "identity_path": str(
                        evaluation_dir / "process-identity.json"
                    ),
                },
            )

        self.assertEqual(observed["outcome"], "launch_error")
        self.assertIsNone(output)
        popen.assert_not_called()

    def test_tracked_attempt_runs_end_to_end(self) -> None:
        tracked_evaluator = {
            "command": [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'ok': True}))",
            ],
            "result_adapter": "test",
            "timeout_seconds": 30,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root / "scripts/metal_research"
            wrapper.mkdir(parents=True)
            source = ROOT / "scripts/metal_research/process_wrapper.py"
            (wrapper / "process_wrapper.py").write_bytes(source.read_bytes())
            evaluation_dir = root / "evaluation"
            tracking = {
                "evaluation_id": "evaluation-001",
                "launch_token": "launch-token",
                "identity_path": str(evaluation_dir / "process-identity.json"),
            }

            observed, output = run_attempt(
                root,
                tracked_evaluator,
                {},
                evaluation_dir,
                "representative",
                process_tracking=tracking,
            )

            identity = json.loads(
                (evaluation_dir / "process-identity.json").read_text()
            )

        self.assertEqual(observed["outcome"], "success")
        self.assertEqual(output, {"ok": True})
        self.assertEqual(identity["launch_token"], "launch-token")

    def test_recovery_does_not_signal_a_reused_pid_after_lease_release(self) -> None:
        identity = {
            "evaluation_id": "evaluation-001",
            "launch_token": "launch-token",
            "pid": 321,
            "pgid": 321,
        }
        with mock.patch.object(
            attempt_runtime, "_recorded_process_owns_lease", return_value=False
        ), mock.patch.object(attempt_runtime.os, "getpgid") as getpgid:
            attempt_runtime.stop_recorded_process_group(identity)

        getpgid.assert_not_called()

    def test_recovery_signals_only_the_matching_tracked_process_group(self) -> None:
        identity = {
            "evaluation_id": "evaluation-001",
            "launch_token": "launch-token",
            "pid": 321,
            "pgid": 321,
        }
        command = (
            "python3 scripts/metal_research/process_wrapper.py "
            "--launch-token launch-token"
        )
        with mock.patch.object(
            attempt_runtime, "_recorded_process_owns_lease", return_value=True
        ), mock.patch.object(
            attempt_runtime, "_recorded_process_command", return_value=command
        ), mock.patch.object(
            attempt_runtime, "_process_group_exists", return_value=False
        ), mock.patch.object(
            attempt_runtime.os, "getpgid", return_value=321
        ), mock.patch.object(attempt_runtime.os, "killpg") as killpg:
            attempt_runtime.stop_recorded_process_group(identity)

        killpg.assert_called_once_with(321, signal.SIGTERM)

    def test_process_wrapper_publishes_identity_before_exec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            identity_path = Path(directory) / "process-identity.json"

            def execute(*_args: object, **_kwargs: object) -> int:
                identity = json.loads(identity_path.read_text())
                self.assertEqual(identity["evaluation_id"], "evaluation-001")
                self.assertEqual(identity["launch_token"], "launch-token")
                return 7

            with mock.patch.object(
                process_wrapper.subprocess, "call", side_effect=execute
            ):
                returncode = process_wrapper.main(
                    [
                        "--identity-path",
                        str(identity_path),
                        "--evaluation-id",
                        "evaluation-001",
                        "--launch-token",
                        "launch-token",
                        "--",
                        "unused",
                    ]
                )

            self.assertEqual(returncode, 7)

    def test_tracked_attempt_inherits_lease_and_uses_wrapper(self) -> None:
        process = SimpleNamespace(
            returncode=0,
            pid=321,
            communicate=mock.Mock(return_value=(json.dumps({"ok": True}) + "\n", "")),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.metal_research.attempt.evaluator_lease", fake_lease
        ), mock.patch(
            "scripts.metal_research.attempt.subprocess.Popen", return_value=process
        ) as popen, mock.patch(
            "scripts.metal_research.attempt.time.monotonic",
            side_effect=[10.0, 11.0],
        ):
            evaluation_dir = Path(directory) / "evaluation"
            tracking = {
                "evaluation_id": "candidate-001-representative",
                "launch_token": "unique-launch-token",
                "identity_path": str(evaluation_dir / "process-identity.json"),
            }
            attempt, output = run_attempt(
                Path(directory),
                evaluator(),
                {},
                evaluation_dir,
                "representative",
                process_tracking=tracking,
            )

        command = popen.call_args.args[0]
        self.assertTrue(
            any(
                item.endswith("scripts/metal_research/process_wrapper.py")
                for item in command
            )
        )
        self.assertEqual(popen.call_args.kwargs["pass_fds"], (99,))
        self.assertEqual(attempt["command"], ["unused"])
        self.assertEqual(output, {"ok": True})

    def test_tier_cost_limit_checks_every_accounting_dimension(self) -> None:
        tier = {
            "cost_limit": {
                "active_evaluator_seconds": 2.0,
                "exclusive_machine_seconds": 3.0,
                "gpu_active_seconds": 4.0,
            }
        }
        attempt = {
            "controller": {
                "subprocess_wall_seconds": 2.1,
                "exclusive_lease_seconds": 3.1,
            },
            "resources": {"gpu_active_charge_seconds": 4.1},
        }
        self.assertEqual(
            runner.cost_limit_overages(tier, attempt),
            [
                "active_evaluator_seconds",
                "exclusive_machine_seconds",
                "gpu_active_seconds",
            ],
        )

    def test_nonzero_attempt_keeps_wall_and_lease_time(self) -> None:
        process = SimpleNamespace(
            returncode=7,
            pid=123,
            communicate=mock.Mock(return_value=("partial\n", "bad\n")),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.metal_research.attempt.evaluator_lease", fake_lease
        ), mock.patch(
            "scripts.metal_research.attempt.subprocess.Popen", return_value=process
        ), mock.patch(
            "scripts.metal_research.attempt.time.monotonic",
            side_effect=[10.0, 13.5],
        ):
            attempt, output = run_attempt(
                Path(directory),
                evaluator(),
                {},
                Path(directory) / "evaluation",
                "representative",
            )

        self.assertIsNone(output)
        self.assertEqual(attempt["outcome"], "nonzero_exit")
        self.assertEqual(attempt["controller"]["subprocess_wall_seconds"], 3.5)
        self.assertEqual(attempt["controller"]["exclusive_lease_seconds"], 11.0)
        self.assertEqual(
            attempt["resources"]["gpu_active_charge_kind"],
            "conservative_wall_upper_bound",
        )
        self.assertEqual(attempt["resources"]["gpu_active_charge_seconds"], 3.5)

    def test_timeout_keeps_partial_logs_and_wall_time(self) -> None:
        process = SimpleNamespace(
            returncode=None,
            pid=123,
            communicate=mock.Mock(
                side_effect=[
                    subprocess.TimeoutExpired(["unused"], 30),
                    ("partial-out", "partial-err"),
                ]
            ),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.metal_research.attempt.evaluator_lease", fake_lease
        ), mock.patch(
            "scripts.metal_research.attempt.subprocess.Popen", return_value=process
        ), mock.patch(
            "scripts.metal_research.attempt.os.killpg"
        ) as killpg, mock.patch(
            "scripts.metal_research.attempt.time.monotonic",
            side_effect=[20.0, 50.0],
        ):
            evaluation_dir = Path(directory) / "evaluation"
            attempt, output = run_attempt(
                Path(directory), evaluator(), {}, evaluation_dir, "representative"
            )

            self.assertEqual((evaluation_dir / "stdout").read_text(), "partial-out")
            self.assertEqual((evaluation_dir / "stderr").read_text(), "partial-err")
        self.assertIsNone(output)
        self.assertEqual(attempt["outcome"], "timeout")
        self.assertEqual(attempt["controller"]["subprocess_wall_seconds"], 30.0)
        killpg.assert_called_once_with(123, signal.SIGTERM)

    def test_cancellation_drains_the_process_group_before_releasing_lease(self) -> None:
        process = SimpleNamespace(
            returncode=None,
            pid=456,
            communicate=mock.Mock(
                side_effect=[KeyboardInterrupt(), ("partial-out", "partial-err")]
            ),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.metal_research.attempt.evaluator_lease", fake_lease
        ), mock.patch(
            "scripts.metal_research.attempt.subprocess.Popen", return_value=process
        ), mock.patch(
            "scripts.metal_research.attempt.os.killpg"
        ) as killpg:
            with self.assertRaises(KeyboardInterrupt):
                run_attempt(
                    Path(directory),
                    evaluator(),
                    {},
                    Path(directory) / "evaluation",
                    "representative",
                )

        killpg.assert_called_once_with(456, signal.SIGTERM)

    def test_lease_timeout_is_a_charged_failed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.metal_research.attempt.evaluator_lease",
            side_effect=EvaluatorLeaseTimeout(7.0),
        ):
            attempt, output = run_attempt(
                Path(directory),
                evaluator(),
                {},
                Path(directory) / "evaluation",
                "representative",
                queue_timeout_seconds=7.0,
            )

        self.assertIsNone(output)
        self.assertEqual(attempt["outcome"], "lease_timeout")
        self.assertEqual(attempt["controller"]["queue_wait_seconds"], 7.0)

    def test_successful_attempt_scrubs_ambient_metal_state(self) -> None:
        def process(*_args: object, **kwargs: object) -> SimpleNamespace:
            environment = kwargs["env"]
            output = {
                "schema_version": 1,
                "ambient": environment.get("JOLT_METAL_AMBIENT"),
                "declared": environment.get("JOLT_METAL_DECLARED"),
                "parameter": environment.get("JOLT_METAL_PARAMETER"),
            }
            return SimpleNamespace(
                returncode=0,
                pid=123,
                communicate=mock.Mock(return_value=(json.dumps(output) + "\n", "")),
            )

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            "os.environ", {"JOLT_METAL_AMBIENT": "forged"}, clear=False
        ), mock.patch(
            "scripts.metal_research.attempt.evaluator_lease", fake_lease
        ), mock.patch(
            "scripts.metal_research.attempt.subprocess.Popen", side_effect=process
        ), mock.patch(
            "scripts.metal_research.attempt.time.monotonic",
            side_effect=[10.0, 11.0],
        ):
            attempt, output = run_attempt(
                Path(directory),
                evaluator(),
                {"JOLT_METAL_PARAMETER": "candidate"},
                Path(directory) / "evaluation",
                "representative",
            )

        self.assertEqual(attempt["outcome"], "success")
        self.assertIsNone(output["ambient"])
        self.assertEqual(output["declared"], "yes")
        self.assertEqual(output["parameter"], "candidate")


class ResultAdapterTests(unittest.TestCase):
    def test_outer_screen_adapter_recomputes_three_raw_pairs(self) -> None:
        orders = [
            ["optimized", "metal"],
            ["metal", "optimized"],
            ["optimized", "metal"],
        ]
        samples = [
            {
                "pair": index,
                "order": order,
                "optimized": {"member_ns": 400},
                "metal": {"member_ns": 100},
            }
            for index, order in enumerate(orders)
        ]
        output = {
            "schema": "outer_remainder_screen_v1",
            "schema_version": 1,
            "kernel": "OuterRemainder",
            "fingerprint": {
                "fixture": "fixed",
                "log_n": 25,
                "pairs": 3,
            },
            "metrics": {"hybrid_speedup": 4.0},
            "samples": samples,
            "excluded_warmup": {},
            "guards": {"all_exact": True},
            "all_exact": True,
            "resources": {"gpu_seconds": 0.25},
        }
        tier = {
            "id": "screen",
            "role": "proxy",
            "replication": descriptor(pairs=3, warmups=1),
            "evaluator": {"result_adapter": "outer_remainder_screen_v1"},
        }

        result, charge = adapt_result(tier, output, "outer_remainder")
        validate_tier_result(result, tier)

        self.assertEqual(result["primary"]["value"], 4.0)
        self.assertEqual(charge["gpu_active_charge_seconds"], 0.25)

    def test_outer_remainder_adapter_recomputes_raw_pairs(self) -> None:
        orders = [
            ["optimized", "metal"] if index % 2 == 0 else ["metal", "optimized"]
            for index in range(5)
        ]
        samples = [
            {
                "pair": index,
                "order": orders[index],
                "optimized": {"member_ns": 400 + 20 * index},
                "metal": {"member_ns": 100 + 5 * index},
            }
            for index in range(5)
        ]
        output = {
            "schema": "outer_remainder_v3",
            "schema_version": 3,
            "kernel": "OuterRemainder",
            "fingerprint": {
                "fixture": "fixed",
                "orders": orders,
                "source_sha256": "a" * 64,
                "binary_sha256": "b" * 64,
            },
            "metrics": {
                "hybrid_speedup": 4.0,
                "paired_speedups": [4.0] * 5,
            },
            "samples": samples,
            "excluded_warmup": {
                "optimized": {"member_ns": 400},
                "metal": {"member_ns": 100},
            },
            "guards": {"all_exact": True},
            "all_exact": True,
            "resources": {"gpu_seconds": 0.5},
        }
        tier = {
            "id": "representative",
            "role": "representative",
            "replication": descriptor(warmups=1),
            "evaluator": {"result_adapter": "outer_remainder_v3"},
        }

        result, charge = adapt_result(tier, output, "outer_remainder")
        validate_tier_result(result, tier)

        self.assertEqual(result["primary"]["value"], 4.0)
        self.assertEqual(result["replication"]["summary"]["median"], 4.0)
        self.assertEqual(charge["gpu_active_charge_seconds"], 0.5)
        self.assertEqual(charge["gpu_active_charge_kind"], "treatment_wall_upper_bound")

    def test_piop_adapter_recomputes_overall_and_order_strata(self) -> None:
        pairs = []
        for index in range(5):
            order = (
                ["optimized", "metal"]
                if index % 2 == 0
                else ["metal", "optimized"]
            )
            pairs.append(
                {
                    "index": index + 1,
                    "order": order,
                    "arms": {
                        "optimized": {
                            "piop_ns": 500,
                            "local": {
                                "kernel": "OuterRemainder",
                                "primary_ns": 250,
                            },
                        },
                        "metal": {
                            "piop_ns": 100,
                            "local": {
                                "kernel": "OuterRemainder",
                                "primary_ns": 50,
                            },
                        },
                    },
                }
            )
        output = {
            "schema_version": 7,
            "kernel": "akita_piop",
            "local_kernel": "OuterRemainder",
            "local_metric": {
                "metric": "outer_remainder_speedup",
                "paired_metric": "paired_outer_remainder_speedups",
            },
            "run_class": {"mode": "production", "acceptance_eligible": True},
            "metrics": {
                "piop_speedup": 5.0,
                "paired_speedups": [5.0] * 5,
                "outer_remainder_speedup": 5.0,
                "paired_outer_remainder_speedups": [5.0] * 5,
            },
            "pairs": pairs,
            "guards": {
                "cpu_proofs_verified": True,
                "metal_proofs_verified": True,
                "production_contract": True,
                "stable_source": True,
                "stable_binary": True,
                "target_scale": True,
            },
            "resources": {"metal_piop_seconds": 0.0000005},
            "fingerprint": {
                "workload": "fibonacci",
                "log_n": 26,
                "source_sha256": "a" * 64,
                "binary_sha256": "b" * 64,
            },
        }
        tier = {
            "id": "piop_holdout",
            "role": "holdout",
            "replication": descriptor(),
            "evaluator": {"result_adapter": "metal_piop_v7"},
            "promotion": {
                "local_kernel": "OuterRemainder",
                "local_metric": "outer_remainder_speedup",
                "log_n": 26,
            },
        }

        result, _ = adapt_result(tier, output, "outer_remainder")
        observed = validate_tier_result(result, tier)

        self.assertEqual(observed, paired_summary(result["replication"]["pairs"], descriptor()))
        self.assertEqual(observed["control_first_median"], 5.0)
        self.assertEqual(observed["treatment_first_median"], 5.0)

    def test_piop_closure_delegates_to_the_full_schema_seven_validator(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        tier = next(
            item
            for item in template["evaluation"]["tiers"]
            if item.get("role") == "holdout"
        )
        params = {name: str(value) for name, value in template["baseline_params"].items()}
        output: dict[str, object] = {}
        with mock.patch.object(
            metal_autoresearch, "validate_production_result"
        ) as validate, mock.patch.object(
            metal_autoresearch, "git_head", return_value="revision"
        ), mock.patch.object(
            metal_autoresearch, "git_worktree_clean", return_value=True
        ):
            runner._validate_closed_result(ROOT, tier, output, params)

        gate = validate.call_args.args[0]["final_validation"]["production_gate"]
        self.assertEqual(gate["minimum_local_speedup"], 5.0)
        self.assertEqual(gate["minimum_log_n"], 26)
        self.assertIn("outer_remainder_readback_exact", gate["required_guards"])
        self.assertEqual(validate.call_args.args[1:], (output, "revision", params, True))

    def test_outer_screen_closure_binds_scale_pairs_and_parameters(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        tier = next(
            item
            for item in template["evaluation"]["tiers"]
            if item.get("role") == "proxy"
        )
        params = {name: str(value) for name, value in template["baseline_params"].items()}
        output = {
            "schema": "outer_remainder_screen_v1",
            "schema_version": 1,
            "kernel": "OuterRemainder",
            "all_exact": True,
            "fingerprint": {
                "fixture": "real-fibonacci-akita-proof",
                "log_n": 25,
                "trace_elements": 1 << 25,
                "trace_rows": (1 << 25) - 1,
                "pairs": 3,
                "excluded_warmup_pairs": 1,
                "orders": [
                    ["optimized", "metal"],
                    ["metal", "optimized"],
                    ["optimized", "metal"],
                ],
                "rayon_threads": 16,
                "materialize_threads": 256,
                "transition_threads": 128,
                "output_threads": 256,
                "cutoff_log2": 16,
                "trace_cutoff_log2": 18,
                "storage_initialization": "full",
                "member_span": "OuterRemainder::complete_member",
                "rounds": 26,
                "output_claims": 35,
                "source_sha256": "a" * 64,
                "binary_sha256": "b" * 64,
            },
        }

        runner._validate_closed_result(ROOT, tier, output, params)

        for field, bad_value in (
            ("log_n", 26),
            ("pairs", 5),
            ("rounds", 27),
            ("materialize_threads", 128),
            ("source_sha256", "not-a-digest"),
        ):
            with self.subTest(field=field):
                invalid = copy.deepcopy(output)
                invalid["fingerprint"][field] = bad_value
                with self.assertRaisesRegex(ValueError, "screen result is not closed"):
                    runner._validate_closed_result(ROOT, tier, invalid, params)


class RunnerIntegrationTests(unittest.TestCase):
    def outer_output(self) -> dict[str, object]:
        from scripts.tests.test_metal_autoresearch import MetalAutoresearchTests

        return MetalAutoresearchTests().outer_remainder_local_contract_fixture()[2]

    def screen_output(
        self, output: dict[str, object], params: dict[str, str]
    ) -> dict[str, object]:
        result = copy.deepcopy(output)
        result["schema"] = "outer_remainder_screen_v1"
        result["schema_version"] = 1
        result["samples"] = result["samples"][:3]
        fingerprint = result["fingerprint"]
        fingerprint.update(
            {
                "log_n": 25,
                "trace_elements": 1 << 25,
                "trace_rows": (1 << 25) - 100,
                "pairs": 3,
                "orders": [sample["order"] for sample in result["samples"]],
                "rounds": 26,
                "materialize_threads": int(
                    params["JOLT_METAL_OUTER_REMAINDER_MATERIALIZE_THREADS"]
                ),
                "transition_threads": int(
                    params["JOLT_METAL_OUTER_REMAINDER_TRANSITION_THREADS"]
                ),
                "output_threads": int(
                    params["JOLT_METAL_OUTER_REMAINDER_OUTPUT_THREADS"]
                ),
                "cutoff_log2": int(
                    params["JOLT_METAL_OUTER_REMAINDER_CUTOFF_LOG2"]
                ),
                "trace_cutoff_log2": int(
                    params["JOLT_METAL_OUTER_REMAINDER_TRACE_CUTOFF_LOG2"]
                ),
            }
        )
        resources = result["resources"]
        resources["resident_row_bytes"] = (1 << 25) * 160
        resources["outer_remainder_storage_bytes"] = 2_152_596_208
        resources["maximum_storage_buffer_bytes"] = 1 << 30
        resources["metal_full_prove_ns_samples"] = resources[
            "metal_full_prove_ns_samples"
        ][:4]
        resources["gpu_seconds"] = (
            sum(resources["metal_full_prove_ns_samples"]) / 1e9
        )
        return result

    def successful_attempt(
        self, output: dict[str, object], launches: list[str]
    ):
        def attempt(
            _root: Path,
            _evaluator: dict[str, object],
            _params: dict[str, str],
            evaluation_dir: Path,
            tier_id: str,
            **_kwargs: object,
        ) -> tuple[dict[str, object], dict[str, object]]:
            evaluation_dir.mkdir(parents=True, exist_ok=False)
            launches.append(tier_id)
            tier_output = (
                self.screen_output(output, _params)
                if tier_id == "screen"
                else output
            )
            return (
                {
                    "schema_version": 1,
                    "tier_id": tier_id,
                    "outcome": "success",
                    "error": None,
                    "command": ["mocked"],
                    "started_at": "2026-08-05T00:00:00Z",
                    "controller": {
                        "queue_wait_seconds": 0.0,
                        "exclusive_lease_seconds": 1.0,
                        "subprocess_wall_seconds": 1.0,
                    },
                    "resources": {
                        "gpu_active_seconds": None,
                        "gpu_active_charge_seconds": 1.0,
                        "gpu_active_charge_kind": "conservative_wall_upper_bound",
                    },
                    "result_sha256": "a" * 64,
                },
                tier_output,
            )

        return attempt

    def tier_result(
        self,
        tier: dict[str, object],
        speedup: float = 5.0,
        local_speedup: Optional[float] = None,
    ) -> dict[str, object]:
        replication = tier["replication"]
        raw_pairs = [
            {
                "index": index,
                "input_id": "fixed",
                "order": (
                    ["control", "treatment"]
                    if index % 2 == 0
                    else ["treatment", "control"]
                ),
                "arms": {
                    "control": {"primary_ns": 500.0},
                    "treatment": {"primary_ns": 500.0 / speedup},
                },
                "effect": speedup,
                "guards": {"exact": True},
            }
            for index in range(replication["included_pairs"])
        ]
        summary = paired_summary(raw_pairs, replication)
        required = tier["promotion"].get("required_guards", ["all_exact"])
        result: dict[str, object] = {
            "schema": "metal_autoresearch_tier_result_v1",
            "schema_version": 1,
            "tier_id": tier["id"],
            "kernel": "OuterRemainder",
            "result_contract": "mocked",
            "primary": {
                "name": "speedup",
                "unit": "x",
                "direction": "max",
                "value": speedup,
            },
            "replication": {
                "mode": "internal_paired",
                "warmups": [
                    {}
                    for _ in range(replication["excluded_warmup_pairs"])
                ],
                "pairs": raw_pairs,
                "summary": summary,
            },
            "guards": {name: True for name in required},
            "telemetry": {},
            "fingerprint": {},
            "payload": {},
        }
        if tier["role"] in {"holdout", "transfer"}:
            local_speedup = speedup if local_speedup is None else local_speedup
            local_pairs = copy.deepcopy(raw_pairs)
            for pair in local_pairs:
                pair["arms"]["treatment"]["primary_ns"] = 500.0 / local_speedup
                pair["effect"] = local_speedup
            result["local"] = {
                "kernel": "OuterRemainder",
                "primary": {"value": local_speedup},
                "replication": {
                    "pairs": local_pairs,
                    "summary": paired_summary(local_pairs, replication),
                },
            }
        return result

    def sealed_tier_executor(
        self,
        run_dir: Path,
        speedups: dict[str, float],
        local_speedup: float = 5.2,
    ):
        def execute(
            _root: Path,
            _run_dir: Path,
            _state: dict[str, object],
            tier: dict[str, object],
            _params: dict[str, str],
            evaluation_id: str,
            **kwargs: object,
        ) -> tuple[dict[str, object], dict[str, object]]:
            role = str(tier["role"])
            result = self.tier_result(
                tier,
                speedup=speedups[role],
                local_speedup=local_speedup,
            )
            return self.seal_tier_evaluation(
                run_dir,
                tier,
                _params,
                evaluation_id,
                result,
                budget_reserve=kwargs.get("budget_reserve"),
            )

        return execute

    def seal_tier_evaluation(
        self,
        run_dir: Path,
        tier: dict[str, object],
        params: dict[str, str],
        evaluation_id: str,
        result: Optional[dict[str, object]],
        *,
        outcome: str = "success",
        error: Optional[str] = None,
        budget_reserve: object = None,
    ) -> tuple[dict[str, object], Optional[dict[str, object]]]:
        evaluation_dir = run_dir / "evaluations" / evaluation_id
        evaluation_dir.mkdir(parents=True, exist_ok=False)
        digest = None
        if result is not None:
            result_bytes = runner.canonical_json(result)
            digest = runner.sha256(result_bytes)
            (evaluation_dir / "tier-result.json").write_bytes(result_bytes)
        attempt = {
            "schema_version": 1,
            "tier_id": tier["id"],
            "outcome": outcome,
            "error": error,
            "command": ["mocked-sealed-evaluator"],
            "started_at": "2026-08-05T00:00:00Z",
            "controller": {
                "queue_wait_seconds": 0.0,
                "exclusive_lease_seconds": 1.0,
                "subprocess_wall_seconds": 1.0,
            },
            "resources": {
                "gpu_active_seconds": None,
                "gpu_active_charge_seconds": 1.0,
                "gpu_active_charge_kind": "conservative_wall_upper_bound",
            },
            "result_sha256": "a" * 64,
            "tier_result_sha256": digest,
            "budget_reserve": budget_reserve,
        }
        (evaluation_dir / "attempt.json").write_bytes(runner.canonical_json(attempt))
        event = {
            "schema_version": 2,
            "event": "tier_evaluated",
            "evaluation_id": evaluation_id,
            "tier_id": tier["id"],
            "params": params,
            "attempt": attempt,
            "primary": result["primary"] if result is not None else None,
            "paired_summary": (
                result["replication"]["summary"]
                if result is not None
                else None
            ),
            "recorded_at": runner.utc_now(),
        }
        runner.append_event(run_dir / "tier-events.jsonl", event)
        return event, result

    def test_kernel_holdout_accepts_before_the_portfolio_reaches_five_x(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        tier = copy.deepcopy(
            next(
                item
                for item in template["evaluation"]["tiers"]
                if item.get("role") == "holdout"
            )
        )
        tier["promotion"].pop("minimum_accepted_speedup", None)
        tier["promotion"]["minimum_portfolio_speedup"] = 2.5
        result = self.tier_result(tier, speedup=3.0, local_speedup=5.2)

        passed, reason = runner._acceptance_result(
            {"goal": template}, tier, result, None
        )

        self.assertTrue(passed, reason)

    def test_kernel_holdout_still_rejects_a_subfloor_local_kernel(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        tier = copy.deepcopy(
            next(
                item
                for item in template["evaluation"]["tiers"]
                if item.get("role") == "holdout"
            )
        )
        tier["promotion"].pop("minimum_accepted_speedup", None)
        tier["promotion"]["minimum_portfolio_speedup"] = 2.5
        result = self.tier_result(tier, speedup=3.0, local_speedup=4.9)

        passed, reason = runner._acceptance_result(
            {"goal": template}, tier, result, None
        )

        self.assertFalse(passed)
        self.assertIn("local-kernel", reason)

    def test_kernel_holdout_rejects_a_slow_result_above_the_speedup_floor(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        tier = copy.deepcopy(
            next(
                item
                for item in template["evaluation"]["tiers"]
                if item.get("role") == "holdout"
            )
        )
        tier["promotion"]["maximum_local_treatment_ms"] = 170.5
        result = self.tier_result(tier, speedup=3.0, local_speedup=5.2)
        for pair in result["local"]["replication"]["pairs"]:
            pair["arms"]["control"]["primary_ns"] = 936_000_000.0
            pair["arms"]["treatment"]["primary_ns"] = 180_000_000.0

        passed, reason = runner._acceptance_result(
            {"goal": template}, tier, result, None
        )

        self.assertFalse(passed)
        self.assertIn("latency", reason)

    def test_init_and_trial_launch_one_internally_paired_process_each(self) -> None:
        output = self.outer_output()
        launches: list[str] = []

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, launches),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            state["usage"] = runner.empty_usage()
            runner.write_state(run_dir, state)
            reconstructed = runner.load_state(run_dir)
            self.assertEqual(reconstructed["usage"]["active_evaluator_seconds"], 2.0)
            decision, state = runner.trial(
                ROOT, run_dir, [], "repeat the baseline candidate"
            )

            self.assertEqual(launches, ["screen", "representative", "screen"])
            self.assertEqual(state["accepted_parent"]["id"], "baseline")
            self.assertEqual(decision["verdict"], "discard")
            self.assertEqual(state["usage"]["candidates_admitted"], 1)
            self.assertEqual(
                len((run_dir / "tier-events.jsonl").read_text().splitlines()), 3
            )

    def test_recovery_conservatively_charges_an_unsealed_attempt(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            state = runner.load_state(run_dir)
            (run_dir / "inflight.json").write_bytes(
                runner.canonical_json(
                    {
                        "schema_version": 2,
                        "kind": "candidate",
                        "candidate_id": "candidate-001",
                        "evaluation_id": "candidate-001-representative",
                        "tier_id": "representative",
                        "params": state["accepted_parent"]["params"],
                        "editable_paths_sha256": state["fingerprint"][
                            "editable_paths_sha256"
                        ],
                        "started_at": runner.utc_now(),
                    }
                )
            )

            recovered = runner.recover(ROOT, run_dir)

            self.assertEqual(recovered["usage"]["failed_attempts"], 1)
            self.assertGreaterEqual(
                recovered["usage"]["gpu_active_estimated_seconds"], 0.0
            )
            self.assertFalse((run_dir / "inflight.json").exists())

    def test_interrupted_holdout_recovery_never_reopens_tuning(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            runner.write_inflight(
                run_dir,
                {
                    "schema_version": 2,
                    "kind": "holdout",
                    "evaluation_id": "validation-001-holdout",
                    "tier_id": "piop_holdout",
                    "params": state["accepted_parent"]["params"],
                    "started_at": runner.utc_now(),
                },
            )

            @contextmanager
            def recovery_lease(*_args: object, **_kwargs: object):
                yield {}

            with mock.patch.object(
                runner, "evaluator_lease", side_effect=recovery_lease
            ):
                recovered = runner.recover(ROOT, run_dir)

            self.assertEqual(recovered["status"], "holdout_retryable")
            with mock.patch.object(runner, "_validate_live_state"):
                with self.assertRaisesRegex(ValueError, "not active"):
                    runner.trial(ROOT, run_dir, [], "holdout is exposed")

    def test_interrupted_holdout_recovery_honors_sealed_acceptance(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            tier = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "holdout"
            )
            result = self.tier_result(tier, speedup=3.0, local_speedup=5.2)
            evaluation_id = "validation-001-holdout"
            event, _ = self.seal_tier_evaluation(
                run_dir,
                tier,
                state["accepted_parent"]["params"],
                evaluation_id,
                result,
                budget_reserve="piop_holdout",
            )
            runner.append_event(
                run_dir / "kernel-validations.jsonl",
                {
                    "schema_version": 2,
                    "event": "kernel_validated",
                    "evaluation_id": evaluation_id,
                    "tier_id": tier["id"],
                    "role": "holdout",
                    "accepted_parent": state["accepted_parent"]["id"],
                    "revision": metal_autoresearch.git_head(ROOT),
                    "status": "accepted",
                    "reason": "all guards passed",
                    "primary": result["primary"],
                    "paired_summary": result["replication"]["summary"],
                    "local": result["local"],
                    "tier_result_sha256": event["attempt"][
                        "tier_result_sha256"
                    ],
                    "portfolio_floor_met": False,
                    "recorded_at": runner.utc_now(),
                },
            )
            runner.write_inflight(
                run_dir,
                {
                    "schema_version": 2,
                    "kind": "holdout",
                    "evaluation_id": evaluation_id,
                    "tier_id": tier["id"],
                    "params": state["accepted_parent"]["params"],
                    "started_at": runner.utc_now(),
                },
            )

            @contextmanager
            def recovery_lease(*_args: object, **_kwargs: object):
                yield {}

            with mock.patch.object(
                runner, "evaluator_lease", side_effect=recovery_lease
            ):
                recovered = runner.recover(ROOT, run_dir)

            self.assertEqual(recovered["status"], "kernel_accepted")

    def test_interrupted_accepted_revalidation_resumes_at_holdout(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            tier = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "representative"
            )
            result = self.tier_result(tier, speedup=5.1)
            evaluation_id = "validation-001-representative"
            event, _ = self.seal_tier_evaluation(
                run_dir,
                tier,
                state["accepted_parent"]["params"],
                evaluation_id,
                result,
                budget_reserve="representative_revalidation",
            )
            runner.append_event(
                run_dir / "kernel-validations.jsonl",
                {
                    "schema_version": 2,
                    "event": "kernel_validated",
                    "evaluation_id": evaluation_id,
                    "tier_id": tier["id"],
                    "role": "representative_revalidation",
                    "accepted_parent": state["accepted_parent"]["id"],
                    "revision": metal_autoresearch.git_head(ROOT),
                    "status": "accepted",
                    "reason": "all guards passed",
                    "primary": result["primary"],
                    "paired_summary": result["replication"]["summary"],
                    "local": None,
                    "tier_result_sha256": event["attempt"][
                        "tier_result_sha256"
                    ],
                    "recorded_at": runner.utc_now(),
                },
            )
            runner.write_inflight(
                run_dir,
                {
                    "schema_version": 2,
                    "kind": "revalidation",
                    "evaluation_id": evaluation_id,
                    "tier_id": tier["id"],
                    "params": state["accepted_parent"]["params"],
                    "started_at": runner.utc_now(),
                },
            )

            @contextmanager
            def recovery_lease(*_args: object, **_kwargs: object):
                yield {}

            with mock.patch.object(
                runner, "evaluator_lease", side_effect=recovery_lease
            ):
                recovered = runner.recover(ROOT, run_dir)

            self.assertEqual(recovered["status"], "holdout_retryable")

    def test_interrupted_initialization_is_retryable(self) -> None:
        output = self.outer_output()
        launches: list[str] = []
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, launches),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            state["status"] = "initializing"
            state["accepted_parent"] = None
            runner.write_state(run_dir, state)
            runner.write_inflight(
                run_dir,
                {
                    "schema_version": 2,
                    "kind": "baseline",
                    "evaluation_id": "baseline-representative-retry-002",
                    "tier_id": "representative",
                    "params": state["template"]["baseline_params"],
                    "started_at": runner.utc_now(),
                },
            )

            @contextmanager
            def recovery_lease(*_args: object, **_kwargs: object):
                yield {}

            with mock.patch.object(
                runner, "evaluator_lease", side_effect=recovery_lease
            ):
                recovered = runner.recover(ROOT, run_dir)

            self.assertEqual(recovered["status"], "initialization_retryable")
            resumed = runner.resume_initialization(ROOT, run_dir)
            self.assertEqual(resumed["status"], "active")
            self.assertEqual(launches, ["screen", "representative"])

    def test_resume_initialization_relaunches_unaccepted_tier_with_new_id(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            state["status"] = "initialization_retryable"
            state["accepted_parent"] = None
            runner.write_state(run_dir, state)
            (run_dir / "baseline-events.jsonl").write_text("")
            relaunched: list[str] = []

            def attempt(
                _root: Path,
                _evaluator: dict[str, object],
                _params: dict[str, str],
                evaluation_dir: Path,
                tier_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                relaunched.append(evaluation_dir.name)
                return self.successful_attempt(output, [])(
                    _root,
                    _evaluator,
                    _params,
                    evaluation_dir,
                    tier_id,
                )

            with mock.patch.object(runner, "run_attempt", side_effect=attempt):
                resumed = runner.resume_initialization(ROOT, run_dir)

            self.assertEqual(resumed["status"], "active")
            self.assertEqual(
                relaunched,
                [
                    "baseline-screen-retry-002",
                    "baseline-representative-retry-002",
                ],
            )

    def test_resume_initialization_rejects_unsealed_completed_tier(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            state["status"] = "initialization_retryable"
            state["accepted_parent"] = None
            runner.write_state(run_dir, state)
            result_path = (
                run_dir
                / "evaluations"
                / "baseline-representative"
                / "tier-result.json"
            )
            result_path.write_text("{}")

            with self.assertRaisesRegex(ValueError, "matching sealed result"):
                runner.resume_initialization(ROOT, run_dir)

    def test_run_bound_goal_decision_is_append_only(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "representative"
            )
            accepted = self.tier_result(representative)
            state["accepted_parent"]["metric"] = 5.0
            state["accepted_parent"]["paired_summary"] = accepted["replication"][
                "summary"
            ]
            state["accepted_parent"]["tiers"]["representative"][
                "treatment_median_ns"
            ] = 100.0
            runner.write_state(run_dir, state)
            candidate = {
                "kernel": "registers_read_write",
                "current_piop_share": 0.13,
                "conservative_local_speedup": 5.0,
            }

            with mock.patch.object(
                metal_autoresearch, "git_worktree_clean", return_value=True
            ), mock.patch.object(
                metal_autoresearch, "validate_production_revision_scope"
            ), mock.patch.object(
                runner,
                "execute_tier",
                side_effect=self.sealed_tier_executor(
                    run_dir,
                    {"representative": 5.0, "holdout": 3.0, "transfer": 3.0},
                ),
            ), mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(runner, "_assert_frozen"):
                runner.validate_production(ROOT, run_dir)
                decision = runner.record_goal_decision(
                    ROOT, run_dir, [candidate], True
                )
                repeated = runner.record_goal_decision(
                    ROOT, run_dir, [candidate], True
                )

            events = [
                event
                for event in runner._events(run_dir / "decision-events.jsonl")
                if event.get("event") == "portfolio_goal_decided"
            ]
            self.assertTrue(decision["continue"])
            self.assertEqual(decision, repeated)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["decision"], decision)

    def test_goal_floor_requires_both_log_26_and_log_27_portfolio_evidence(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "representative"
            )
            accepted = self.tier_result(representative)
            state["accepted_parent"]["metric"] = 5.0
            state["accepted_parent"]["paired_summary"] = accepted["replication"][
                "summary"
            ]
            state["accepted_parent"]["tiers"]["representative"][
                "treatment_median_ns"
            ] = 100.0
            runner.write_state(run_dir, state)
            with mock.patch.object(
                metal_autoresearch, "git_worktree_clean", return_value=True
            ), mock.patch.object(
                metal_autoresearch, "validate_production_revision_scope"
            ), mock.patch.object(
                runner,
                "execute_tier",
                side_effect=self.sealed_tier_executor(
                    run_dir,
                    {
                        "representative": 5.1,
                        "holdout": 5.1,
                        "transfer": 4.9,
                    },
                ),
            ), mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(runner, "_assert_frozen"):
                runner.validate_production(ROOT, run_dir)
                decision = runner.record_goal_decision(ROOT, run_dir, [], False)

            self.assertTrue(decision["continue"])
            self.assertFalse(decision["floor_met"])

    def test_goal_decision_rejects_tampered_validation_evidence(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "representative"
            )
            accepted = self.tier_result(representative)
            state["accepted_parent"]["metric"] = 5.0
            state["accepted_parent"]["paired_summary"] = accepted["replication"][
                "summary"
            ]
            state["accepted_parent"]["tiers"]["representative"][
                "treatment_median_ns"
            ] = 100.0
            runner.write_state(run_dir, state)
            with mock.patch.object(
                metal_autoresearch, "git_worktree_clean", return_value=True
            ), mock.patch.object(
                metal_autoresearch, "validate_production_revision_scope"
            ), mock.patch.object(
                runner,
                "execute_tier",
                side_effect=self.sealed_tier_executor(
                    run_dir,
                    {
                        "representative": 5.1,
                        "holdout": 5.1,
                        "transfer": 5.1,
                    },
                ),
            ), mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(runner, "_assert_frozen"):
                runner.validate_production(ROOT, run_dir)
                holdout = next(
                    event
                    for event in runner._events(
                        run_dir / "kernel-validations.jsonl"
                    )
                    if event.get("role") == "holdout"
                )
                tier_events_path = run_dir / "tier-events.jsonl"
                original_tier_events = tier_events_path.read_bytes()
                tier_events = runner._events(tier_events_path)
                next(
                    event
                    for event in tier_events
                    if event.get("evaluation_id") == holdout["evaluation_id"]
                )["params"] = {"replayed": "configuration"}
                tier_events_path.write_bytes(
                    b"".join(
                        runner.canonical_json(event) + b"\n"
                        for event in tier_events
                    )
                )
                with self.assertRaisesRegex(ValueError, "sealed"):
                    runner.record_goal_decision(ROOT, run_dir, [], False)

                tier_events_path.write_bytes(original_tier_events)
                result_path = (
                    run_dir
                    / "evaluations"
                    / holdout["evaluation_id"]
                    / "tier-result.json"
                )
                result_path.write_text("{}")

                with self.assertRaisesRegex(ValueError, "sealed"):
                    runner.record_goal_decision(ROOT, run_dir, [], False)

    def test_invalid_holdout_retries_without_revalidation_or_tuning(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "representative"
            )
            accepted = self.tier_result(representative)
            state["accepted_parent"]["metric"] = 5.0
            state["accepted_parent"]["paired_summary"] = accepted["replication"][
                "summary"
            ]
            state["accepted_parent"]["tiers"]["representative"][
                "treatment_median_ns"
            ] = 100.0
            runner.write_state(run_dir, state)
            launches: list[str] = []
            holdout_attempts = 0

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                params: dict[str, str],
                evaluation_id: str,
                **kwargs: object,
            ) -> tuple[dict[str, object], Optional[dict[str, object]]]:
                nonlocal holdout_attempts
                role = str(tier["role"])
                launches.append(role)
                if role == "holdout":
                    holdout_attempts += 1
                    if holdout_attempts == 1:
                        return self.seal_tier_evaluation(
                            run_dir,
                            tier,
                            params,
                            evaluation_id,
                            None,
                            outcome="timeout",
                            error="timeout",
                            budget_reserve=kwargs.get("budget_reserve"),
                        )
                return self.seal_tier_evaluation(
                    run_dir,
                    tier,
                    params,
                    evaluation_id,
                    self.tier_result(tier),
                    budget_reserve=kwargs.get("budget_reserve"),
                )

            with mock.patch.object(
                metal_autoresearch, "git_worktree_clean", return_value=True
            ), mock.patch.object(
                metal_autoresearch, "validate_production_revision_scope"
            ), mock.patch.object(runner, "execute_tier", side_effect=execute):
                with self.assertRaisesRegex(ValueError, "timeout"):
                    runner.validate_production(ROOT, run_dir)
                self.assertEqual(
                    runner.load_state(run_dir)["status"], "holdout_retryable"
                )
                with mock.patch.object(runner, "_validate_live_state"):
                    with self.assertRaisesRegex(ValueError, "not active"):
                        runner.trial(ROOT, run_dir, [], "must not tune on holdout")
                _, final_state = runner.validate_production(ROOT, run_dir)

            self.assertEqual(
                launches, ["representative", "holdout", "holdout", "transfer"]
            )
            self.assertEqual(final_state["status"], "kernel_transferred")

    def test_recovery_drains_the_recorded_process_before_releasing_inflight(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            state = runner.load_state(run_dir)
            identity_path = (
                run_dir
                / "evaluations"
                / "candidate-001-representative"
                / "process-identity.json"
            )
            identity_path.parent.mkdir()
            identity_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "evaluation_id": "candidate-001-representative",
                        "launch_token": "orphan-token",
                        "pid": 4321,
                        "pgid": 4321,
                    }
                )
            )
            runner.write_inflight(
                run_dir,
                {
                    "schema_version": 2,
                    "kind": "candidate",
                    "candidate_id": "candidate-001",
                    "evaluation_id": "candidate-001-representative",
                    "tier_id": "representative",
                    "params": state["accepted_parent"]["params"],
                    "editable_paths_sha256": state["fingerprint"][
                        "editable_paths_sha256"
                    ],
                    "started_at": runner.utc_now(),
                    "process_tracking": {
                        "evaluation_id": "candidate-001-representative",
                        "launch_token": "orphan-token",
                        "identity_path": str(identity_path.relative_to(run_dir)),
                    },
                },
            )
            order: list[str] = []

            @contextmanager
            def recovery_lease(*_args: object, **_kwargs: object):
                order.append("lease")
                yield {}

            def stop(identity: dict[str, object]) -> None:
                self.assertTrue((run_dir / "inflight.json").exists())
                self.assertEqual(identity["launch_token"], "orphan-token")
                order.append("stop")

            with mock.patch.object(
                runner, "stop_recorded_process_group", side_effect=stop, create=True
            ), mock.patch.object(
                runner, "evaluator_lease", side_effect=recovery_lease, create=True
            ):
                runner.recover(ROOT, run_dir)

            self.assertEqual(order, ["stop", "lease"])
            self.assertFalse((run_dir / "inflight.json").exists())

    def test_proxy_rejection_skips_the_representative_tier(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            screen = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("id") == "screen"
            )
            screen_output = self.screen_output(
                output,
                state["accepted_parent"]["params"],
            )
            screen_result, _ = adapt_result(
                screen, screen_output, "outer_remainder"
            )
            launches: list[str] = []

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                params: dict[str, str],
                evaluation_id: str,
                **kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                launches.append(str(tier["id"]))
                return (
                    {"attempt": {"error": None, "outcome": "success"}},
                    screen_result,
                )

            with mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(runner, "execute_tier", side_effect=execute):
                decision, _ = runner.trial(
                    ROOT, run_dir, [], "candidate rejected by the proxy"
                )

            self.assertEqual(launches, ["screen"])
            self.assertEqual(decision["verdict"], "discard")

    def test_production_runs_revalidation_holdout_transfer_once(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "representative"
            )
            accepted = self.tier_result(representative)
            state["accepted_parent"]["metric"] = 5.0
            state["accepted_parent"]["paired_summary"] = accepted["replication"][
                "summary"
            ]
            state["accepted_parent"]["tiers"]["representative"][
                "treatment_median_ns"
            ] = 100.0
            runner.write_state(run_dir, state)
            launches: list[str] = []

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                params: dict[str, str],
                evaluation_id: str,
                **kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                launches.append(str(tier["role"]))
                return self.seal_tier_evaluation(
                    run_dir,
                    tier,
                    params,
                    evaluation_id,
                    self.tier_result(tier),
                    budget_reserve=kwargs.get("budget_reserve"),
                )

            with mock.patch.object(
                metal_autoresearch, "git_worktree_clean", return_value=True
            ), mock.patch.object(
                metal_autoresearch, "validate_production_revision_scope"
            ), mock.patch.object(runner, "execute_tier", side_effect=execute):
                record, state = runner.validate_production(ROOT, run_dir)
                repeated, repeated_state = runner.validate_production(ROOT, run_dir)

            self.assertEqual(launches, ["representative", "holdout", "transfer"])
            self.assertEqual(record["role"], "transfer")
            self.assertEqual(repeated, record)
            self.assertEqual(state["status"], "kernel_transferred")
            self.assertEqual(repeated_state["status"], "kernel_transferred")

    def test_transfer_retry_does_not_rerun_accepted_holdout(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "representative"
            )
            accepted = self.tier_result(representative)
            state["accepted_parent"]["metric"] = 5.0
            state["accepted_parent"]["paired_summary"] = accepted["replication"][
                "summary"
            ]
            state["accepted_parent"]["tiers"]["representative"][
                "treatment_median_ns"
            ] = 100.0
            runner.write_state(run_dir, state)
            launches: list[str] = []
            transfer_attempts = 0

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                params: dict[str, str],
                evaluation_id: str,
                **kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                nonlocal transfer_attempts
                role = str(tier["role"])
                launches.append(role)
                speedup = 5.0
                if role == "transfer":
                    transfer_attempts += 1
                    if transfer_attempts == 1:
                        speedup = 4.9
                return self.seal_tier_evaluation(
                    run_dir,
                    tier,
                    params,
                    evaluation_id,
                    self.tier_result(tier, speedup),
                    budget_reserve=kwargs.get("budget_reserve"),
                )

            with mock.patch.object(
                metal_autoresearch, "git_worktree_clean", return_value=True
            ), mock.patch.object(
                metal_autoresearch, "validate_production_revision_scope"
            ), mock.patch.object(runner, "execute_tier", side_effect=execute):
                with self.assertRaisesRegex(ValueError, "local-kernel"):
                    runner.validate_production(ROOT, run_dir)
                self.assertEqual(
                    runner.load_state(run_dir)["status"], "kernel_accepted"
                )
                _, state = runner.validate_production(ROOT, run_dir)

            self.assertEqual(
                launches, ["representative", "holdout", "transfer", "transfer"]
            )
            self.assertEqual(state["status"], "kernel_transferred")

    def test_failed_holdout_seals_the_run_against_further_tuning(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "representative"
            )
            accepted = self.tier_result(representative)
            state["accepted_parent"]["metric"] = 5.0
            state["accepted_parent"]["paired_summary"] = accepted["replication"][
                "summary"
            ]
            state["accepted_parent"]["tiers"]["representative"][
                "treatment_median_ns"
            ] = 100.0
            runner.write_state(run_dir, state)

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                _params: dict[str, str],
                _evaluation_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                if tier["role"] == "holdout":
                    result = self.tier_result(
                        tier, speedup=2.4, local_speedup=5.2
                    )
                else:
                    result = self.tier_result(tier)
                return ({"attempt": {"error": None, "outcome": "success"}}, result)

            with mock.patch.object(
                metal_autoresearch, "git_worktree_clean", return_value=True
            ), mock.patch.object(
                metal_autoresearch, "validate_production_revision_scope"
            ), mock.patch.object(runner, "execute_tier", side_effect=execute):
                with self.assertRaisesRegex(ValueError, "portfolio floor"):
                    runner.validate_production(ROOT, run_dir)

            self.assertEqual(runner.load_state(run_dir)["status"], "holdout_rejected")
            with mock.patch.object(runner, "_validate_live_state"):
                with self.assertRaisesRegex(ValueError, "not active"):
                    runner.trial(ROOT, run_dir, [], "must not tune on holdout")

    def test_state_digest_is_committed_inside_the_atomic_state_file(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            raw = json.loads((run_dir / "run.json").read_text())
            self.assertIn("state_sha256", raw)
            self.assertFalse((run_dir / "run.sha256").exists())
            raw["status"] = "forged"
            (run_dir / "run.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "state digest"):
                runner.load_state(run_dir)

    def test_recovered_kept_candidate_returns_to_unvalidated_active_state(self) -> None:
        output = self.outer_output()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(output, []),
        ):
            run_dir = Path(directory) / "run"
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                run_dir,
            )
            state["status"] = "kernel_transferred"
            runner.write_state(run_dir, state)
            candidate_id = "candidate-001"
            editable_digest = metal_autoresearch.path_digest(
                ROOT, state["template"]["scope"]["editable"]
            )
            runner.append_event(
                run_dir / "decision-events.jsonl",
                {
                    "event": "candidate_decided",
                    "candidate_id": candidate_id,
                    "verdict": "keep",
                    "params": state["accepted_parent"]["params"],
                    "primary": {"value": 5.1},
                    "paired_summary": state["accepted_parent"]["paired_summary"],
                    "tier_results": state["accepted_parent"]["tiers"],
                },
            )
            recovered, _ = runner._recover_committed_candidate(
                ROOT,
                run_dir,
                state,
                {
                    "candidate_id": candidate_id,
                    "editable_paths_sha256": editable_digest,
                },
            )

            self.assertTrue(recovered)
            self.assertEqual(state["accepted_parent"]["id"], candidate_id)
            self.assertEqual(state["status"], "active")


class DispatchAndGoalTests(unittest.TestCase):
    def test_v2_cli_init_rejects_a_dirty_worktree(self) -> None:
        args = SimpleNamespace(
            command="init",
            root=ROOT,
            template=ROOT
            / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
            run_dir=ROOT / "unused-test-run",
        )
        parser = mock.Mock()
        parser.parse_args.return_value = args
        legacy = mock.Mock()
        legacy.git_worktree_clean.return_value = False
        with mock.patch.object(runner, "parser", return_value=parser), mock.patch.object(
            runner, "_legacy", return_value=legacy
        ), mock.patch.object(runner, "init_run", return_value={}) as initialize:
            exit_code = runner.main([])

        self.assertEqual(exit_code, 2)
        initialize.assert_not_called()

    def test_v2_template_can_be_validated_without_initializing_a_run(self) -> None:
        template = (
            ROOT
            / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
        )

        result = runner.validate_template_file(ROOT, template)

        self.assertTrue(result["valid"])
        self.assertEqual(result["slot_id"], "spartan_outer_remainder")

    def test_canonical_controller_dispatches_only_schema_two_contracts(self) -> None:
        self.assertTrue(
            metal_autoresearch.command_uses_v2(
                SimpleNamespace(
                    command="init",
                    template=ROOT
                    / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                )
            )
        )
        self.assertFalse(
            metal_autoresearch.command_uses_v2(
                SimpleNamespace(
                    command="init",
                    template=ROOT
                    / "crates/jolt-kernels/autoresearch/outer_remainder.template.json",
                )
            )
        )

    def test_goal_continues_below_five_and_pursues_clear_headroom(self) -> None:
        contract = json.loads(
            (
                ROOT / "crates/jolt-kernels/autoresearch/piop_goal.v2.json"
            ).read_text()
        )
        below = runner.goal_decision(contract, 4.99, [])
        self.assertTrue(below["continue"])
        self.assertFalse(below["floor_met"])

        above = runner.goal_decision(
            contract,
            5.1,
            [
                {
                    "kernel": "outer_remainder",
                    "current_piop_share": 0.01,
                    "conservative_local_speedup": 5.5,
                }
            ],
        )
        self.assertTrue(above["continue"])
        self.assertTrue(above["clear_headroom"])


if __name__ == "__main__":
    unittest.main()
