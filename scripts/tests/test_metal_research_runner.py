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
from scripts.metal_research import artifacts as runtime_artifacts
from scripts.metal_research import binaries as binary_artifacts
from scripts.metal_research import process_wrapper
from scripts.metal_research.contracts import validate_template
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
                "dyld": environment.get("DYLD_INSERT_LIBRARIES"),
                "python": environment.get("PYTHONPATH"),
                "declared": environment.get("JOLT_METAL_DECLARED"),
                "parameter": environment.get("JOLT_METAL_PARAMETER"),
            }
            return SimpleNamespace(
                returncode=0,
                pid=123,
                communicate=mock.Mock(return_value=(json.dumps(output) + "\n", "")),
            )

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            "os.environ",
            {
                "JOLT_METAL_AMBIENT": "forged",
                "DYLD_INSERT_LIBRARIES": "/forged.dylib",
                "PYTHONPATH": "/forged/python",
            },
            clear=False,
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
        self.assertIsNone(output["dyld"])
        self.assertIsNone(output["python"])
        self.assertEqual(output["declared"], "yes")
        self.assertEqual(output["parameter"], "candidate")

    def test_attempt_rejects_declared_loader_and_python_injection(self) -> None:
        declared = evaluator()
        declared["env"] = {"DYLD_INSERT_LIBRARIES": "/forged.dylib"}
        with self.assertRaisesRegex(ValueError, "unsafe state"):
            attempt_runtime._environment(declared, {}, Path("artifacts"))

        with self.assertRaisesRegex(ValueError, "unsafe state"):
            attempt_runtime._environment(
                evaluator(), {"PYTHONPATH": "/forged/python"}, Path("artifacts")
            )

    def test_attempt_passes_only_explicit_controller_artifacts(self) -> None:
        def process(*_args: object, **kwargs: object) -> SimpleNamespace:
            environment = kwargs["env"]
            output = {
                "parent": environment.get(
                    "JOLT_AUTORESEARCH_PARENT_ARTIFACT"
                ),
                "candidate": environment.get(
                    "JOLT_AUTORESEARCH_CANDIDATE_ARTIFACT"
                ),
                "forged": environment.get("JOLT_AUTORESEARCH_FORGED"),
            }
            return SimpleNamespace(
                returncode=0,
                pid=123,
                communicate=mock.Mock(
                    return_value=(json.dumps(output) + "\n", "")
                ),
            )

        context = {
            "JOLT_AUTORESEARCH_PARENT_ARTIFACT": "/sealed/parent",
            "JOLT_AUTORESEARCH_CANDIDATE_ARTIFACT": "/sealed/candidate",
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            "os.environ", {"JOLT_AUTORESEARCH_FORGED": "ambient"}, clear=False
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
                {},
                Path(directory) / "evaluation",
                "screen",
                context_env=context,
                context_record={"artifact_sha256": "a" * 64},
            )

        self.assertEqual(attempt["outcome"], "success")
        self.assertEqual(output["parent"], "/sealed/parent")
        self.assertEqual(output["candidate"], "/sealed/candidate")
        self.assertIsNone(output["forged"])
        self.assertEqual(
            attempt["execution_context"], {"artifact_sha256": "a" * 64}
        )

    def test_attempt_rejects_evaluator_owned_controller_state(self) -> None:
        declared = evaluator()
        declared["env"] = {
            "JOLT_AUTORESEARCH_PARENT_ARTIFACT": "/forged/parent"
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.metal_research.attempt.evaluator_lease", fake_lease
        ), mock.patch(
            "scripts.metal_research.attempt.subprocess.Popen"
        ) as popen:
            attempt, output = run_attempt(
                Path(directory),
                declared,
                {},
                Path(directory) / "evaluation",
                "screen",
            )

        self.assertEqual(attempt["outcome"], "launch_error")
        self.assertIsNone(output)
        popen.assert_not_called()

    def test_non_result_build_attempt_and_prelaunch_check(self) -> None:
        process = SimpleNamespace(
            returncode=0,
            pid=123,
            communicate=mock.Mock(return_value=("cargo output", "")),
        )
        checked = mock.Mock()
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.metal_research.attempt.evaluator_lease", fake_lease
        ), mock.patch(
            "scripts.metal_research.attempt.subprocess.Popen", return_value=process
        ) as popen, mock.patch(
            "scripts.metal_research.attempt.time.monotonic",
            side_effect=[10.0, 11.0],
        ):
            attempt, output = run_attempt(
                Path(directory),
                evaluator(),
                {},
                Path(directory) / "evaluation",
                "sealed_binary:runner",
                parse_result=False,
                prelaunch_check=checked,
            )

        checked.assert_called_once_with()
        popen.assert_called_once()
        self.assertEqual(attempt["outcome"], "success")
        self.assertIsNone(output)

    def test_attempt_rejects_a_partial_controller_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.metal_research.attempt.evaluator_lease", fake_lease
        ), mock.patch(
            "scripts.metal_research.attempt.subprocess.Popen"
        ) as popen:
            attempt, output = run_attempt(
                Path(directory),
                evaluator(),
                {},
                Path(directory) / "evaluation",
                "screen",
                context_env={
                    "JOLT_AUTORESEARCH_PARENT_ARTIFACT": "/sealed/parent"
                },
            )

        self.assertEqual(attempt["outcome"], "launch_error")
        self.assertIsNone(output)
        popen.assert_not_called()

    def test_attempt_rejects_reserved_search_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.metal_research.attempt.evaluator_lease", fake_lease
        ), mock.patch(
            "scripts.metal_research.attempt.subprocess.Popen"
        ) as popen:
            attempt, output = run_attempt(
                Path(directory),
                evaluator(),
                {"JOLT_AUTORESEARCH_PARENT_ARTIFACT": "/forged"},
                Path(directory) / "evaluation",
                "screen",
            )

        self.assertEqual(attempt["outcome"], "launch_error")
        self.assertIsNone(output)
        popen.assert_not_called()


class ArtifactContextTests(unittest.TestCase):
    SOURCE = Path(
        "crates/jolt-kernels/src/metal/solinas/outer_remainder/shader.metal"
    )
    PARAMS = {
        "JOLT_METAL_OUTER_REMAINDER_BINDING_PLAN": "b_only_v1",
        "JOLT_METAL_OUTER_REMAINDER_MATERIALIZE_THREADS": "256",
        "JOLT_METAL_OUTER_REMAINDER_TRANSITION_THREADS": "128",
        "JOLT_METAL_OUTER_REMAINDER_OUTPUT_THREADS": "256",
        "JOLT_METAL_OUTER_REMAINDER_CUTOFF_LOG2": "16",
        "JOLT_METAL_OUTER_REMAINDER_TRACE_CUTOFF_LOG2": "18",
    }

    def state(self, accepted_parent: object) -> dict[str, object]:
        return {
            "template": {
                "baseline_params": self.PARAMS,
                "runtime_artifact": {
                    "kind": "outer_msl_v1",
                    "source_path": self.SOURCE.as_posix(),
                    "plan_parameter": (
                        "JOLT_METAL_OUTER_REMAINDER_BINDING_PLAN"
                    ),
                    "plans": ["b_only_v1"],
                    "tier_id": "screen",
                },
            },
            "accepted_parent": accepted_parent,
        }

    def test_baseline_screen_uses_the_snapshot_for_both_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            run_dir = Path(directory) / "run"
            live = root / self.SOURCE
            baseline = run_dir / "snapshots/baseline" / self.SOURCE
            live.parent.mkdir(parents=True)
            baseline.parent.mkdir(parents=True)
            (run_dir / "artifacts").mkdir()
            live.write_text("kernel void live_candidate() {}")
            baseline.write_text("kernel void sealed_baseline() {}")

            environment, record = runner._runtime_artifact_context(
                root,
                run_dir,
                self.state(None),
                {"id": "screen"},
                dict(self.PARAMS),
            )

            self.assertEqual(
                record["parent"]["artifact_sha256"],
                record["candidate"]["artifact_sha256"],
            )
            self.assertEqual(
                environment["JOLT_AUTORESEARCH_PARENT_ARTIFACT"],
                environment["JOLT_AUTORESEARCH_CANDIDATE_ARTIFACT"],
            )
            self.assertEqual(
                set(environment),
                {
                    "JOLT_AUTORESEARCH_PARENT_ARTIFACT",
                    "JOLT_AUTORESEARCH_CANDIDATE_ARTIFACT",
                },
            )
            self.assertEqual(set(record), {"kind", "parent", "candidate"})

    def test_candidate_artifact_uses_live_source_and_its_own_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            run_dir = Path(directory) / "run"
            live = root / self.SOURCE
            baseline = run_dir / "snapshots/baseline" / self.SOURCE
            live.parent.mkdir(parents=True)
            baseline.parent.mkdir(parents=True)
            (run_dir / "artifacts").mkdir()
            live.write_text("kernel void live_candidate() {}")
            baseline.write_text("kernel void sealed_baseline() {}")
            candidate_params = dict(self.PARAMS)
            candidate_params[
                "JOLT_METAL_OUTER_REMAINDER_TRANSITION_THREADS"
            ] = "256"
            state = self.state(
                {
                    "snapshot": "baseline",
                    "params": dict(self.PARAMS),
                }
            )

            _, record = runner._runtime_artifact_context(
                root,
                run_dir,
                state,
                {"id": "screen"},
                candidate_params,
            )

            self.assertNotEqual(
                record["parent"]["artifact_sha256"],
                record["candidate"]["artifact_sha256"],
            )
            self.assertEqual(
                record["parent"]["manifest"]["dispatch"][
                    "transition_threads"
                ],
                128,
            )
            self.assertEqual(
                record["candidate"]["manifest"]["dispatch"][
                    "transition_threads"
                ],
                256,
            )

    def test_output_fingerprint_must_match_both_controller_artifacts(self) -> None:
        context = {
            "kind": "outer_msl_v1",
            "parent": {"artifact_sha256": "a" * 64},
            "candidate": {"artifact_sha256": "b" * 64},
        }
        output = {
            "fingerprint": {
                "parent_artifact_sha256": "a" * 64,
                "candidate_artifact_sha256": "b" * 64,
            }
        }

        runtime_artifacts.validate_runtime_artifact_output(
            output, "outer_msl_v1", context
        )
        output["fingerprint"]["candidate_artifact_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            runtime_artifacts.validate_runtime_artifact_output(
                output, "outer_msl_v1", context
            )

    def test_nonzero_attempt_records_artifact_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "artifacts").mkdir()
            source = run_dir / "candidate.metal"
            source.write_text("kernel void candidate() {}")
            dispatch = runtime_artifacts.outer_dispatch_from_params(
                self.PARAMS
            )
            artifact = runtime_artifacts.materialize_outer_artifact(
                run_dir, source, "b_only_v1", dispatch
            )
            context = {
                "kind": "outer_msl_v1",
                "parent": artifact,
                "candidate": copy.deepcopy(artifact),
            }
            artifact_source = (
                run_dir / artifact["artifact_path"] / "outer.metal"
            )
            artifact_source.write_text("kernel void tampered() {}")
            attempt = {"outcome": "nonzero_exit", "error": "status 7"}

            valid = runner._seal_attempt_artifacts(
                run_dir, "outer_msl_v1", context, attempt
            )

        self.assertFalse(valid)
        self.assertEqual(attempt["outcome"], "artifact_changed")
        self.assertEqual(attempt["evaluator_outcome"], "nonzero_exit")

    def test_required_runtime_context_cannot_fail_open(self) -> None:
        attempt = {"outcome": "success"}

        valid = runner._seal_attempt_artifacts(
            Path("unused"), "outer_msl_v1", None, attempt
        )

        self.assertFalse(valid)
        self.assertEqual(attempt["outcome"], "artifact_changed")
        self.assertEqual(attempt["evaluator_outcome"], "success")
        self.assertEqual(
            attempt["error"], "required runtime artifact context is missing"
        )

    def test_recovery_rejects_attempt_and_inflight_context_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "artifacts").mkdir()
            source = run_dir / "candidate.metal"
            source.write_text("kernel void candidate() {}")
            artifact = runtime_artifacts.materialize_outer_artifact(
                run_dir,
                source,
                "b_only_v1",
                runtime_artifacts.outer_dispatch_from_params(self.PARAMS),
            )
            context = {
                "kind": "outer_msl_v1",
                "parent": artifact,
                "candidate": copy.deepcopy(artifact),
            }
            error = runner._recovery_execution_context_error(
                run_dir,
                {
                    "template": {
                        "runtime_artifact": {
                            "kind": "outer_msl_v1",
                            "tier_id": "screen",
                        }
                    }
                },
                {"tier_id": "screen", "execution_context": context},
                {"execution_context": None},
            )

        self.assertEqual(error, "attempt and inflight artifact contexts differ")

    def test_recovery_rejects_context_kind_mismatch(self) -> None:
        state = {
            "template": {
                "runtime_artifact": {
                    "kind": "outer_msl_v1",
                    "tier_id": "screen",
                }
            }
        }
        inflight = {
            "tier_id": "screen",
            "execution_context": {"kind": "unknown_v1"},
        }

        error = runner._recovery_execution_context_error(
            Path("unused"), state, inflight, None
        )

        self.assertEqual(
            error,
            "runtime artifact context kind does not match the sealed template",
        )


class SealedBinaryContextTests(unittest.TestCase):
    BINARY_ID = "outer_remainder_eval"
    SOURCE = "runner-source.rs"
    BUILD_COMMAND = ["cargo", "build", "--release"]

    def materialize(
        self, root: Path, run_dir: Path
    ) -> dict[str, object]:
        (root / self.SOURCE).write_text("fn main() {}\n")
        binary = b"sealed evaluator"
        manifest = {
            "schema": binary_artifacts.SEALED_BINARY_SCHEMA,
            "schema_version": binary_artifacts.SEALED_BINARY_SCHEMA_VERSION,
            "id": self.BINARY_ID,
            "binary_file": binary_artifacts.SEALED_BINARY_FILE,
            "binary_bytes": len(binary),
            "binary_sha256": binary_artifacts.sha256(binary),
            "source_sha256": binary_artifacts.declared_source_sha256(
                root, [self.SOURCE]
            ),
            "build_command_sha256": binary_artifacts.sha256(
                binary_artifacts.canonical_json(self.BUILD_COMMAND)
            ),
            "build_environment_sha256": "c" * 64,
        }
        prepared = {
            "artifact_sha256": binary_artifacts.sha256(
                binary_artifacts.canonical_json(manifest) + b"\0" + binary
            ),
            "manifest": manifest,
            "binary": binary,
        }
        return binary_artifacts.materialize_sealed_binary(run_dir, prepared)

    def state(self, record: dict[str, object]) -> dict[str, object]:
        return {
            "template": {
                "budget": {
                    "total": {
                        "max_candidates_admitted": 1,
                        "max_calendar_seconds": 300,
                        "max_active_evaluator_seconds": 300,
                        "max_exclusive_machine_seconds": 300,
                        "max_gpu_active_seconds": 0,
                        "max_tokens": 0,
                        "max_monetary_usd": 0,
                    },
                    "reserves": [],
                },
                "sealed_binaries": {
                    self.BINARY_ID: {
                        "build": {
                            "command": self.BUILD_COMMAND,
                            "output_path": "target/runner",
                            "timeout_seconds": 30,
                        },
                        "source_paths": [self.SOURCE],
                        "consumer_tiers": ["screen"],
                        "result_fingerprint": [
                            "fingerprint",
                            "runner_binary_sha256",
                        ],
                    }
                }
            },
            "sealed_binaries": {self.BINARY_ID: record},
        }

    def tier(self) -> dict[str, object]:
        return {
            "id": "screen",
            "applicable": True,
            "evaluator": {
                "command": [
                    binary_artifacts.sealed_binary_token(self.BINARY_ID)
                ]
            },
        }

    def test_resolves_only_the_whole_token_and_publishes_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "binaries").mkdir()
            record = self.materialize(run_dir, run_dir)

            evaluator, environment, context = runner._sealed_binary_context(
                run_dir, run_dir, self.state(record), self.tier()
            )

            self.assertEqual(
                Path(evaluator["command"][0]).read_bytes(), b"sealed evaluator"
            )
            self.assertEqual(
                environment["JOLT_AUTORESEARCH_RUNNER_SHA256"],
                record["manifest"]["binary_sha256"],
            )
            self.assertEqual(context, {self.BINARY_ID: record})

    def test_post_launch_mutation_invalidates_the_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "binaries").mkdir()
            record = self.materialize(run_dir, run_dir)
            state = self.state(record)
            context = {self.BINARY_ID: record}
            artifact = run_dir / record["artifact_path"]
            executable = artifact / "runner"
            artifact.chmod(0o755)
            executable.chmod(0o755)
            executable.write_bytes(b"mutated evaluator")
            executable.chmod(0o555)
            artifact.chmod(0o555)
            attempt = {"outcome": "success", "error": None}

            valid = runner._seal_attempt_binaries(
                run_dir, state, self.tier(), context, attempt
            )

            self.assertFalse(valid)
            self.assertEqual(attempt["outcome"], "binary_changed")
            self.assertEqual(attempt["evaluator_outcome"], "success")

    def test_result_fingerprint_must_match_the_sealed_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "binaries").mkdir()
            record = self.materialize(run_dir, run_dir)
            state = self.state(record)
            context = {self.BINARY_ID: record}
            output = {
                "fingerprint": {
                    "runner_binary_sha256": record["manifest"][
                        "binary_sha256"
                    ]
                }
            }

            runner._validate_sealed_binary_fingerprint(
                output, state, self.tier(), context
            )
            output["fingerprint"]["runner_binary_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "does not match"):
                runner._validate_sealed_binary_fingerprint(
                    output, state, self.tier(), context
                )

    def test_profiled_evaluator_must_match_the_newly_sealed_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "profile.json"
            evidence = {
                "evaluator": {
                    "runner_binary_sha256": "a" * 64,
                    "runner_source_sha256": "b" * 64,
                }
            }
            payload = runner.canonical_json(evidence)
            evidence_path.write_bytes(payload)
            state = {
                "template": {
                    "iteration_profile": {
                        "evidence_path": "profile.json",
                        "evidence_sha256": runner.sha256(payload),
                    },
                    "evaluation": {
                        "tiers": [
                            {
                                "id": "screen",
                                "role": "proxy",
                                "applicable": True,
                            }
                        ]
                    },
                    "sealed_binaries": {
                        self.BINARY_ID: {"consumer_tiers": ["screen"]}
                    },
                },
                "sealed_binaries": {
                    self.BINARY_ID: {
                        "manifest": {
                            "binary_sha256": "a" * 64,
                            "source_sha256": "b" * 64,
                        }
                    }
                },
            }

            runner._validate_profiled_binary(root, state)
            state["sealed_binaries"][self.BINARY_ID]["manifest"][
                "binary_sha256"
            ] = "c" * 64
            with self.assertRaisesRegex(ValueError, "does not match"):
                runner._validate_profiled_binary(root, state)

    def test_recovery_requires_matching_published_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "binaries").mkdir()
            record = self.materialize(run_dir, run_dir)
            state = self.state(record)
            state["template"]["evaluation"] = {"tiers": [self.tier()]}
            context = {self.BINARY_ID: record}
            error = runner._recovery_sealed_binary_context_error(
                run_dir,
                state,
                {"tier_id": "screen", "sealed_binary_context": context},
                {"sealed_binary_context": None},
            )

            self.assertEqual(
                error, "attempt and inflight sealed binary contexts differ"
            )

    def durable_state(self, root: Path) -> dict[str, object]:
        return {
            "schema_version": 2,
            "status": "sealing_binaries",
            "template": {
                "budget": {
                    "total": {
                        "max_candidates_admitted": 1,
                        "max_calendar_seconds": 300,
                        "max_active_evaluator_seconds": 300,
                        "max_exclusive_machine_seconds": 300,
                        "max_gpu_active_seconds": 0,
                        "max_tokens": 0,
                        "max_monetary_usd": 0,
                    },
                    "reserves": [],
                },
                "sealed_binaries": {
                    self.BINARY_ID: {
                        "build": {
                            "command": ["build-runner"],
                            "output_path": "target/runner",
                            "timeout_seconds": 30,
                        },
                        "source_paths": [self.SOURCE],
                        "consumer_tiers": ["screen"],
                        "result_fingerprint": [
                            "fingerprint",
                            "runner_binary_sha256",
                        ],
                    }
                },
                "scope": {"editable": [], "frozen": []},
            },
            "sealed_binaries": {},
            "usage": runner.empty_usage(),
            "accepted_parent": None,
            "created_at": runner.utc_now(),
            "root": str(root),
        }

    def test_binary_build_is_durable_before_baseline_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            run_dir = Path(directory) / "run"
            root.mkdir()
            runner._initialize_run_files(run_dir)
            (root / self.SOURCE).write_text("fn main() {}\n")
            output = root / "target/runner"
            output.parent.mkdir()
            output.write_bytes(b"durable evaluator")
            output.chmod(0o755)
            state = self.durable_state(root)
            runner.write_state(run_dir, state)

            def build_attempt(
                _root: Path,
                _evaluator: dict[str, object],
                _params: dict[str, str],
                evaluation_dir: Path,
                tier_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], None]:
                evaluation_dir.mkdir(parents=True)
                return (
                    {
                        "schema_version": 1,
                        "tier_id": tier_id,
                        "outcome": "success",
                        "error": None,
                        "command": ["build-runner"],
                        "started_at": runner.utc_now(),
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
                        "result_sha256": None,
                    },
                    None,
                )

            with mock.patch.object(
                runner, "run_attempt", side_effect=build_attempt
            ), mock.patch.object(
                runner, "_continue_initialization", side_effect=lambda *_: state
            ):
                result = runner._continue_binary_sealing(root, run_dir, state)

            record = result["sealed_binaries"][self.BINARY_ID]
            binary_artifacts.verify_sealed_binary_contract(
                root,
                run_dir,
                self.BINARY_ID,
                state["template"]["sealed_binaries"][self.BINARY_ID],
                record,
            )
            self.assertFalse((run_dir / "inflight.json").exists())
            self.assertFalse((run_dir / "binaries").stat().st_mode & 0o222)
            self.assertEqual(
                len((run_dir / "binary-events.jsonl").read_text().splitlines()),
                1,
            )
            self.assertEqual(result["usage"]["active_evaluator_seconds"], 1.0)
            self.assertEqual(result["usage"]["exclusive_machine_seconds"], 1.0)
            self.assertEqual(result["usage"]["gpu_active_seconds"], 0.0)

    def test_failed_binary_build_is_recoverably_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            run_dir = Path(directory) / "run"
            root.mkdir()
            runner._initialize_run_files(run_dir)
            (root / self.SOURCE).write_text("fn main() {}\n")
            state = self.durable_state(root)
            runner.write_state(run_dir, state)

            def failed_attempt(
                _root: Path,
                _evaluator: dict[str, object],
                _params: dict[str, str],
                evaluation_dir: Path,
                tier_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], None]:
                evaluation_dir.mkdir(parents=True)
                return (
                    {
                        "schema_version": 1,
                        "tier_id": tier_id,
                        "outcome": "timeout",
                        "error": "build timed out",
                        "command": ["build-runner"],
                        "started_at": runner.utc_now(),
                        "controller": {
                            "queue_wait_seconds": 0.0,
                            "exclusive_lease_seconds": 30.0,
                            "subprocess_wall_seconds": 30.0,
                        },
                        "resources": {
                            "gpu_active_seconds": None,
                            "gpu_active_charge_seconds": 30.0,
                            "gpu_active_charge_kind": "conservative_wall_upper_bound",
                        },
                        "result_sha256": None,
                    },
                    None,
                )

            with mock.patch.object(
                runner, "run_attempt", side_effect=failed_attempt
            ), self.assertRaisesRegex(ValueError, "build timed out"):
                runner._continue_binary_sealing(root, run_dir, state)

            recovered = runner.load_state(run_dir)
            self.assertEqual(recovered["status"], "sealing_binaries_retryable")
            self.assertFalse((run_dir / "inflight.json").exists())
            self.assertEqual(recovered["sealed_binaries"], {})
            self.assertEqual(recovered["usage"]["active_evaluator_seconds"], 30.0)
            self.assertEqual(recovered["usage"]["exclusive_machine_seconds"], 30.0)
            self.assertEqual(recovered["usage"]["gpu_active_seconds"], 0.0)

    def test_binary_build_is_admitted_against_all_non_gpu_budgets(self) -> None:
        cases = (
            ("max_active_evaluator_seconds", "active_evaluator_seconds"),
            ("max_exclusive_machine_seconds", "exclusive_machine_seconds"),
            ("max_calendar_seconds", "calendar"),
        )
        for cap, message in cases:
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "root"
                run_dir = Path(directory) / "run"
                root.mkdir()
                runner._initialize_run_files(run_dir)
                state = self.durable_state(root)
                state["template"]["budget"]["total"][cap] = 29
                runner.write_state(run_dir, state)

                with mock.patch.object(runner, "run_attempt") as run, self.assertRaisesRegex(
                    runner.BudgetExhausted, message
                ):
                    runner._continue_binary_sealing(root, run_dir, state)

                run.assert_not_called()
                self.assertFalse((run_dir / "inflight.json").exists())
                self.assertEqual(
                    (run_dir / "binary-events.jsonl").read_text(), ""
                )

    def test_binary_build_terminal_event_is_recovered_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            run_dir = Path(directory) / "run"
            root.mkdir()
            runner._initialize_run_files(run_dir)
            (root / self.SOURCE).write_text("fn main() {}\n")
            output = root / "target/runner"
            output.parent.mkdir()
            output.write_bytes(b"durable evaluator")
            output.chmod(0o755)
            state = self.durable_state(root)
            runner.write_state(run_dir, state)

            def build_attempt(
                _root: Path,
                _evaluator: dict[str, object],
                _params: dict[str, str],
                evaluation_dir: Path,
                tier_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], None]:
                evaluation_dir.mkdir(parents=True)
                return (
                    {
                        "schema_version": 1,
                        "tier_id": tier_id,
                        "outcome": "success",
                        "error": None,
                        "command": ["build-runner"],
                        "started_at": runner.utc_now(),
                        "controller": {
                            "queue_wait_seconds": 0.0,
                            "exclusive_lease_seconds": 1.0,
                            "subprocess_wall_seconds": 1.0,
                        },
                        "resources": {
                            "gpu_active_seconds": None,
                            "gpu_active_charge_seconds": 1.0,
                            "gpu_active_charge_kind": (
                                "conservative_wall_upper_bound"
                            ),
                        },
                        "result_sha256": None,
                    },
                    None,
                )

            with mock.patch.object(
                runner, "run_attempt", side_effect=build_attempt
            ), mock.patch.object(
                runner, "charge_attempt", side_effect=RuntimeError("crash")
            ), self.assertRaisesRegex(RuntimeError, "crash"):
                runner._continue_binary_sealing(root, run_dir, state)

            self.assertTrue((run_dir / "inflight.json").is_file())
            self.assertEqual(
                len((run_dir / "binary-events.jsonl").read_text().splitlines()),
                1,
            )
            recovered_state = runner.load_state(
                run_dir, verify_sealed_binaries=False
            )
            inflight = runner.read_json(run_dir / "inflight.json")
            recovered = runner._recover_binary_build(
                root, run_dir, recovered_state, inflight
            )

            self.assertEqual(
                len((run_dir / "binary-events.jsonl").read_text().splitlines()),
                1,
            )
            self.assertIn(self.BINARY_ID, recovered["sealed_binaries"])
            self.assertEqual(recovered["usage"]["active_evaluator_seconds"], 1.0)
            self.assertEqual(recovered["usage"]["gpu_active_seconds"], 0.0)
            self.assertFalse((run_dir / "inflight.json").exists())


class ResultAdapterTests(unittest.TestCase):
    def successor_tier(self) -> dict[str, object]:
        return {
            "id": "screen",
            "role": "proxy",
            "replication": descriptor(pairs=4, warmups=1),
            "evaluator": {
                "result_adapter": "outer_remainder_successor_v1"
            },
            "promotion": {
                "kind": "successor_screen",
                "log_n": 25,
                "clear_loss_ratio": 0.98,
                "minimum_uncertainty": 0.02,
                "maximum_calibration_absolute_log_bias": 0.02,
                "maximum_screen_relative_mad": 0.03,
                "inconclusive_retry_limit": 1,
            },
        }

    def successor_output(
        self, parent_ns: int, candidate_ns: int
    ) -> dict[str, object]:
        orders = [
            ["parent", "candidate"]
            if index % 2 == 0
            else ["candidate", "parent"]
            for index in range(4)
        ]
        speedup = parent_ns / candidate_ns
        return {
            "schema": "outer_remainder_successor_v1",
            "schema_version": 1,
            "kernel": "OuterRemainder",
            "fingerprint": {
                "fixture": "resident-outer-remainder-v1",
                "log_n": 25,
                "pairs": 4,
                "excluded_warmup_pairs": 1,
                "orders": orders,
                "parent_artifact_sha256": "a" * 64,
                "candidate_artifact_sha256": "b" * 64,
            },
            "metrics": {"successor_speedup": speedup},
            "samples": [
                {
                    "pair": index,
                    "order": order,
                    "parent": {
                        "gpu_active_ns": parent_ns,
                        "wall_ns": parent_ns,
                    },
                    "candidate": {
                        "gpu_active_ns": candidate_ns,
                        "wall_ns": candidate_ns,
                    },
                }
                for index, order in enumerate(orders)
            ],
            "excluded_warmup": {
                "order": ["parent", "candidate"],
                "parent": {"gpu_active_ns": parent_ns},
                "candidate": {"gpu_active_ns": candidate_ns},
            },
            "guards": {"all_exact": True},
            "all_exact": True,
            "resources": {
                "gpu_seconds": 5 * (parent_ns + candidate_ns) / 1e9
            },
        }

    def test_outer_successor_adapter_recomputes_four_raw_pairs(self) -> None:
        tier = self.successor_tier()
        output = self.successor_output(105, 100)

        result, charge = adapt_result(tier, output, "outer_remainder")
        validate_tier_result(result, tier)

        self.assertEqual(result["primary"]["value"], 1.05)
        self.assertEqual(result["primary"]["name"], "successor_speedup")
        self.assertEqual(charge["gpu_active_seconds"], 1.025e-6)

    def test_outer_successor_adapter_rejects_unearned_gpu_charge(self) -> None:
        tier = self.successor_tier()
        output = self.successor_output(105, 100)
        output["resources"]["gpu_seconds"] = 1.0

        with self.assertRaisesRegex(ValueError, "disagrees with raw arms"):
            adapt_result(tier, output, "outer_remainder")

    def test_outer_successor_adapter_rejects_legacy_arm_names(self) -> None:
        output = self.successor_output(105, 100)
        output["samples"][0]["order"] = ["optimized", "metal"]

        with self.assertRaisesRegex(ValueError, "arm order"):
            adapt_result(self.successor_tier(), output, "outer_remainder")

    def test_outer_successor_result_closes_fixture_and_replication(self) -> None:
        tier = self.successor_tier()
        output = self.successor_output(105, 100)

        runner._validate_closed_result(ROOT, tier, output, {})

        for field, value in (
            ("fixture", "different"),
            ("log_n", 24),
            ("pairs", 3),
            ("orders", list(reversed(output["fingerprint"]["orders"]))),
        ):
            with self.subTest(field=field):
                tampered = copy.deepcopy(output)
                tampered["fingerprint"][field] = value
                with self.assertRaisesRegex(ValueError, "not closed"):
                    runner._validate_closed_result(ROOT, tier, tampered, {})

    def successor_v2_tier(self) -> dict[str, object]:
        tier = self.successor_tier()
        tier["evaluator"][
            "result_adapter"
        ] = "outer_remainder_successor_v2"
        return tier

    def successor_v2_arm(self, gpu_ns: int) -> dict[str, object]:
        setup_ns = 10
        materialize_ns = gpu_ns // 2
        first_bind_ns = gpu_ns // 4
        dense_ns = gpu_ns // 8
        openings_ns = gpu_ns - materialize_ns - first_bind_ns - dense_ns
        return {
            "gpu_active_ns": gpu_ns,
            "wall_ns": gpu_ns + 5,
            "resource_gpu_active_ns": gpu_ns + setup_ns,
            "setup_gpu_active_ns": setup_ns,
            "setup_wall_ns": setup_ns + 5,
            "phase_gpu_active_ns": {
                "materialize": materialize_ns,
                "first_bind": first_bind_ns,
                "dense_rounds": dense_ns,
                "openings": openings_ns,
            },
            "tail_elements": 1 << 16,
            "initialized_bytes": 4096,
            "storage_owned_bytes": 4096,
            "round_device_buffer_allocations": 0,
            "output_sha256": "d" * 64,
            "dispatch_counts": {
                "materializations": 1,
                "stream_transitions": 1,
                "dense_transitions": 9,
                "cpu_tail_exports": 1,
                "opening_scans": 1,
                "command_buffers": 12,
            },
        }

    def successor_v2_output(self) -> dict[str, object]:
        orders = [
            ["parent", "candidate"]
            if index % 2 == 0
            else ["candidate", "parent"]
            for index in range(4)
        ]
        parent_times = [100, 200, 300, 10_000]
        candidate_times = [100] * 4
        samples = [
            {
                "pair": index,
                "order": orders[index],
                "parent": self.successor_v2_arm(parent_times[index]),
                "candidate": self.successor_v2_arm(candidate_times[index]),
            }
            for index in range(4)
        ]
        warmup = {
            "order": ["parent", "candidate"],
            "parent": self.successor_v2_arm(100),
            "candidate": self.successor_v2_arm(100),
        }
        total_ns = sum(
            arm["resource_gpu_active_ns"]
            for record in [warmup, *samples]
            for arm in (record["parent"], record["candidate"])
        )
        guards = {
            "all_exact": True,
            "correctness_exact": True,
            "target_scale": True,
            "runtime_artifacts_exact": True,
            "resident_row_handle_lifecycle_exact": True,
            "metal_phase_schedule_exact": True,
            "gpu_timestamps_exact": True,
        }
        return {
            "schema": "outer_remainder_successor_v2",
            "schema_version": 2,
            "kernel": "OuterRemainder",
            "fingerprint": {
                "fixture": "resident-outer-remainder-v2",
                "log_n": 25,
                "pairs": 4,
                "excluded_warmup_pairs": 1,
                "orders": orders,
                "parent_artifact_sha256": "a" * 64,
                "candidate_artifact_sha256": "b" * 64,
                "runner_binary_sha256": "c" * 64,
            },
            "metrics": {
                "successor_speedup": 2.5,
                "paired_speedups": [1.0, 2.0, 3.0, 100.0],
            },
            "excluded_warmup": warmup,
            "samples": samples,
            "guards": guards,
            "all_exact": True,
            "resources": {
                "gpu_active_total_ns": total_ns,
                "gpu_seconds": total_ns / 1e9,
            },
            "telemetry": {
                "device_name": "test-metal",
                "device_registry_shared": True,
                "cycles": 1 << 25,
                "parent_binding_plan": "b_only_v1",
                "candidate_binding_plan": "b_only_v1",
                "parent_source_sha256": "e" * 64,
                "candidate_source_sha256": "f" * 64,
                "production_last_owner_release_deferred": True,
                "compilation": {
                    "context_order": ["parent", "candidate"],
                    "parent": {
                        "source_assembly_ns": 1,
                        "library_compile_ns": 2,
                        "source_bytes": 1024,
                        "assembled_source_sha256": "1" * 64,
                        "pipeline_set_ns": [1] * 5,
                        "pipeline_set_total_ns": 5,
                    },
                    "candidate": {
                        "source_assembly_ns": 1,
                        "library_compile_ns": 2,
                        "source_bytes": 1024,
                        "assembled_source_sha256": "2" * 64,
                        "pipeline_set_ns": [1] * 5,
                        "pipeline_set_total_ns": 5,
                    },
                },
            },
        }

    def uniform_successor_v2_output(
        self, parent_ns: int, candidate_ns: int
    ) -> dict[str, object]:
        result = self.successor_v2_output()
        for sample in result["samples"]:
            sample["parent"] = self.successor_v2_arm(parent_ns)
            sample["candidate"] = self.successor_v2_arm(candidate_ns)
        result["excluded_warmup"]["parent"] = self.successor_v2_arm(parent_ns)
        result["excluded_warmup"]["candidate"] = self.successor_v2_arm(
            candidate_ns
        )
        speedup = parent_ns / candidate_ns
        result["metrics"] = {
            "successor_speedup": speedup,
            "paired_speedups": [speedup] * 4,
        }
        total_ns = sum(
            arm["resource_gpu_active_ns"]
            for record in [result["excluded_warmup"], *result["samples"]]
            for arm in (record["parent"], record["candidate"])
        )
        result["resources"] = {
            "gpu_active_total_ns": total_ns,
            "gpu_seconds": total_ns / 1e9,
        }
        return result

    def varying_successor_v2_output(
        self, parent_times: list[int], candidate_ns: int = 100
    ) -> dict[str, object]:
        if len(parent_times) != 4:
            raise ValueError("the successor fixture requires four parent times")
        result = self.successor_v2_output()
        effects = []
        for sample, parent_ns in zip(result["samples"], parent_times):
            sample["parent"] = self.successor_v2_arm(parent_ns)
            sample["candidate"] = self.successor_v2_arm(candidate_ns)
            effects.append(parent_ns / candidate_ns)
        sorted_effects = sorted(effects)
        result["metrics"] = {
            "successor_speedup": (
                sorted_effects[1] + sorted_effects[2]
            )
            / 2,
            "paired_speedups": sorted_effects,
        }
        total_ns = sum(
            arm["resource_gpu_active_ns"]
            for record in [result["excluded_warmup"], *result["samples"]]
            for arm in (record["parent"], record["candidate"])
        )
        result["resources"] = {
            "gpu_active_total_ns": total_ns,
            "gpu_seconds": total_ns / 1e9,
        }
        return result

    def calibration_v2_output(
        self, parent_ns: int = 100, candidate_ns: int = 100
    ) -> dict[str, object]:
        result = self.uniform_successor_v2_output(parent_ns, candidate_ns)
        result["fingerprint"]["candidate_artifact_sha256"] = result[
            "fingerprint"
        ]["parent_artifact_sha256"]
        result["telemetry"]["candidate_source_sha256"] = result["telemetry"][
            "parent_source_sha256"
        ]
        return result

    def test_outer_successor_v2_uses_midpoint_and_full_gpu_charge(self) -> None:
        tier = self.successor_v2_tier()
        output = self.successor_v2_output()

        result, charge = adapt_result(tier, output, "outer_remainder")
        validate_tier_result(result, tier)
        runner._validate_closed_result(ROOT, tier, output, {})

        self.assertEqual(result["primary"]["value"], 2.5)
        self.assertEqual(
            charge["gpu_active_charge_seconds"],
            output["resources"]["gpu_active_total_ns"] / 1e9,
        )
        self.assertEqual(
            result["telemetry"]["candidate_phase_gpu_active_ns"],
            {
                "materialize": 50.0,
                "first_bind": 25.0,
                "dense_rounds": 12.0,
                "openings": 13.0,
            },
        )
        self.assertEqual(
            result["telemetry"]["parent_phase_gpu_active_ns"]["materialize"],
            125.0,
        )

        warmup_changed = self.successor_v2_output()
        warmup_changed["excluded_warmup"]["candidate"] = self.successor_v2_arm(
            1_000_000
        )
        warmup_changed["excluded_warmup"]["parent"] = self.successor_v2_arm(
            1_000_000
        )
        total_ns = sum(
            arm["resource_gpu_active_ns"]
            for record in [
                warmup_changed["excluded_warmup"],
                *warmup_changed["samples"],
            ]
            for arm in (record["parent"], record["candidate"])
        )
        warmup_changed["resources"] = {
            "gpu_active_total_ns": total_ns,
            "gpu_seconds": total_ns / 1e9,
        }
        changed_result, _ = adapt_result(tier, warmup_changed, "outer_remainder")
        self.assertEqual(
            changed_result["telemetry"]["candidate_phase_gpu_active_ns"],
            result["telemetry"]["candidate_phase_gpu_active_ns"],
        )

    def test_outer_successor_v2_rejects_compile_telemetry_drift(self) -> None:
        tier = self.successor_v2_tier()
        output = self.successor_v2_output()

        tampered = copy.deepcopy(output)
        tampered["telemetry"]["compilation"]["parent"][
            "pipeline_set_total_ns"
        ] = 4
        with self.assertRaisesRegex(ValueError, "compilation context"):
            runner._validate_closed_result(ROOT, tier, tampered, {})

        tampered = copy.deepcopy(output)
        tampered["telemetry"]["compilation"]["candidate"][
            "pipeline_set_ns"
        ][0] = 16
        tampered["telemetry"]["compilation"]["candidate"][
            "pipeline_set_total_ns"
        ] = 20
        with self.assertRaisesRegex(ValueError, "exceeds setup wall"):
            runner._validate_closed_result(ROOT, tier, tampered, {})

    def test_outer_successor_v2_rejects_raw_evidence_drift(self) -> None:
        mutations = {
            "arm charge": lambda value: value["samples"][0]["parent"].__setitem__(
                "resource_gpu_active_ns", 111
            ),
            "resource total": lambda value: value["resources"].__setitem__(
                "gpu_active_total_ns", 1
            ),
            "paired metrics": lambda value: value["metrics"][
                "paired_speedups"
            ].__setitem__(1, 2.1),
            "output digest": lambda value: value["samples"][0][
                "parent"
            ].__setitem__("output_sha256", "0" * 64),
            "dispatch": lambda value: value["samples"][0]["parent"][
                "dispatch_counts"
            ].__setitem__("dense_transitions", 8),
            "phase sum": lambda value: value["samples"][0]["parent"][
                "phase_gpu_active_ns"
            ].__setitem__("materialize", 49),
            "phase bool": lambda value: value["samples"][0]["parent"][
                "phase_gpu_active_ns"
            ].__setitem__("materialize", True),
            "zero materialize": lambda value: value["samples"][0]["parent"][
                "phase_gpu_active_ns"
            ].__setitem__("materialize", 0),
            "guard": lambda value: value["guards"].__setitem__(
                "gpu_timestamps_exact", False
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                output = self.successor_v2_output()
                mutate(output)
                with self.assertRaises(ValueError):
                    adapt_result(
                        self.successor_v2_tier(), output, "outer_remainder"
                    )

    def test_successor_promotion_uses_one_as_its_neutral_point(self) -> None:
        tier = self.successor_v2_tier()
        calibration, _ = adapt_result(
            tier, self.calibration_v2_output(), "outer_remainder"
        )
        parent = {
            "metric": 1.0,
            "relative_mad": 0.0,
            "calibration": runner._successor_calibration(tier, calibration),
        }
        improvement, _ = adapt_result(
            tier,
            self.uniform_successor_v2_output(105, 100),
            "outer_remainder",
        )
        clear_loss, _ = adapt_result(
            tier,
            self.uniform_successor_v2_output(95, 100),
            "outer_remainder",
        )

        promoted, _ = runner._promotion_pass(tier, improvement, parent)
        rejected, _ = runner._promotion_pass(tier, clear_loss, parent)

        self.assertTrue(promoted)
        self.assertFalse(rejected)

    def test_baseline_admission_requires_unbiased_stable_a_a(self) -> None:
        tier = self.successor_v2_tier()
        stable, _ = adapt_result(
            tier, self.calibration_v2_output(), "outer_remainder"
        )
        biased, _ = adapt_result(
            tier, self.calibration_v2_output(106, 100), "outer_remainder"
        )
        unstable = copy.deepcopy(stable)
        unstable["replication"]["summary"]["mad"] = 0.031
        wrong_artifact, _ = adapt_result(
            tier,
            self.uniform_successor_v2_output(100, 100),
            "outer_remainder",
        )
        noisy_wall_output = self.calibration_v2_output()
        for index, sample in enumerate(noisy_wall_output["samples"]):
            sample["parent"]["wall_ns"] = 105 if index % 2 == 0 else 205
        noisy_wall, _ = adapt_result(tier, noisy_wall_output, "outer_remainder")
        order_biased_output = self.calibration_v2_output()
        effects = []
        for index, sample in enumerate(order_biased_output["samples"]):
            parent_ns = 103 if index % 2 == 0 else 98
            sample["parent"] = self.successor_v2_arm(parent_ns)
            sample["candidate"] = self.successor_v2_arm(100)
            effects.append(parent_ns / 100)
        order_biased_output["metrics"] = {
            "successor_speedup": 1.005,
            "paired_speedups": sorted(effects),
        }
        total_ns = sum(
            arm["resource_gpu_active_ns"]
            for record in [
                order_biased_output["excluded_warmup"],
                *order_biased_output["samples"],
            ]
            for arm in (record["parent"], record["candidate"])
        )
        order_biased_output["resources"] = {
            "gpu_active_total_ns": total_ns,
            "gpu_seconds": total_ns / 1e9,
        }
        order_biased, _ = adapt_result(
            tier, order_biased_output, "outer_remainder"
        )

        self.assertTrue(runner._baseline_admission(tier, stable)[0])
        self.assertFalse(runner._baseline_admission(tier, biased)[0])
        self.assertFalse(runner._baseline_admission(tier, unstable)[0])
        self.assertFalse(runner._baseline_admission(tier, wrong_artifact)[0])
        self.assertFalse(runner._baseline_admission(tier, noisy_wall)[0])
        admitted, _, diagnostics = runner._baseline_admission(
            tier, order_biased
        )
        self.assertFalse(admitted)
        self.assertGreater(
            diagnostics["gpu"]["maximum_absolute_log_bias"],
            diagnostics["limits"]["maximum_absolute_log_bias"],
        )

    def test_successor_retries_only_noisy_ambiguous_screens(self) -> None:
        tier = self.successor_v2_tier()
        calibration, _ = adapt_result(
            tier, self.calibration_v2_output(), "outer_remainder"
        )
        parent = {
            "metric": 1.0,
            "relative_mad": 0.0,
            "calibration": runner._successor_calibration(tier, calibration),
        }
        potential_clear_loss, _ = adapt_result(
            tier,
            self.varying_successor_v2_output([90, 130, 90, 80]),
            "outer_remainder",
        )
        slight_loss, _ = adapt_result(
            tier,
            self.varying_successor_v2_output([95, 110, 99, 85]),
            "outer_remainder",
        )
        strong, _ = adapt_result(
            tier,
            self.uniform_successor_v2_output(200, 100),
            "outer_remainder",
        )
        decisive_loss, _ = adapt_result(
            tier,
            self.uniform_successor_v2_output(50, 100),
            "outer_remainder",
        )

        self.assertEqual(
            runner._successor_screen_disposition(
                tier, potential_clear_loss, parent, retry_available=True
            )[0],
            "retry",
        )
        self.assertEqual(
            runner._successor_screen_disposition(
                tier, potential_clear_loss, parent, retry_available=False
            )[0],
            "advance",
        )
        self.assertEqual(
            runner._successor_screen_disposition(
                tier, slight_loss, parent, retry_available=True
            )[0],
            "advance",
        )
        self.assertEqual(
            runner._successor_screen_disposition(
                tier, strong, parent, retry_available=True
            )[0],
            "advance",
        )
        self.assertEqual(
            runner._successor_screen_disposition(
                tier, decisive_loss, parent, retry_available=True
            )[0],
            "discard",
        )

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

        output["samples"][0]["order"] = ["parent", "candidate"]
        with self.assertRaisesRegex(ValueError, "arm order"):
            adapt_result(tier, output, "outer_remainder")

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
        tier = copy.deepcopy(tier)
        tier["evaluator"]["result_adapter"] = "outer_remainder_screen_v1"
        tier["replication"]["included_pairs"] = 3
        tier["replication"]["minimum_pairs_per_order_stratum"] = 1
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
    def setUp(self) -> None:
        self.sealing_patch = mock.patch.object(
            runner,
            "_continue_binary_sealing",
            side_effect=self.fake_binary_sealing,
        )
        self.sealing_patch.start()
        self.addCleanup(self.sealing_patch.stop)

    def fake_binary_sealing(
        self, root: Path, run_dir: Path, state: dict[str, object]
    ) -> dict[str, object]:
        records = {}
        for binary_id, contract in state["template"][
            "sealed_binaries"
        ].items():
            binary = b"test sealed evaluator"
            manifest = {
                "schema": binary_artifacts.SEALED_BINARY_SCHEMA,
                "schema_version": binary_artifacts.SEALED_BINARY_SCHEMA_VERSION,
                "id": binary_id,
                "binary_file": binary_artifacts.SEALED_BINARY_FILE,
                "binary_bytes": len(binary),
                "binary_sha256": binary_artifacts.sha256(binary),
                "source_sha256": binary_artifacts.declared_source_sha256(
                    root, contract["source_paths"]
                ),
                "build_command_sha256": binary_artifacts.sha256(
                    binary_artifacts.canonical_json(contract["build"]["command"])
                ),
                "build_environment_sha256": "0" * 64,
            }
            prepared = {
                "artifact_sha256": binary_artifacts.sha256(
                    binary_artifacts.canonical_json(manifest) + b"\0" + binary
                ),
                "manifest": manifest,
                "binary": binary,
            }
            records[binary_id] = binary_artifacts.materialize_sealed_binary(
                run_dir, prepared
            )
        state["sealed_binaries"] = records
        binary_artifacts.seal_sealed_binary_store(run_dir)
        state["status"] = "initializing"
        runner.write_state(run_dir, state)
        initialized = runner._continue_initialization(root, run_dir, state)
        initialized["proxy"]["status"] = "enabled"
        initialized["proxy"]["reason"] = "test sentinel calibration"
        runner.write_state(run_dir, initialized)
        return initialized

    def outer_output(self) -> dict[str, object]:
        from scripts.tests.test_metal_autoresearch import MetalAutoresearchTests

        return MetalAutoresearchTests().outer_remainder_local_contract_fixture()[2]

    def screen_output(
        self,
        output: dict[str, object],
        params: dict[str, str],
        execution_context: dict[str, object],
        sealed_binary_context: dict[str, object],
    ) -> dict[str, object]:
        del output, params
        parent = execution_context["parent"]
        candidate = execution_context["candidate"]
        parent_ns = (
            100
            if parent["artifact_sha256"] == candidate["artifact_sha256"]
            else 105
        )
        result = ResultAdapterTests().uniform_successor_v2_output(
            parent_ns, 100
        )
        runner_record = sealed_binary_context["outer_remainder_eval"]
        result["fingerprint"].update(
            {
                "parent_artifact_sha256": parent["artifact_sha256"],
                "candidate_artifact_sha256": candidate["artifact_sha256"],
                "runner_binary_sha256": runner_record["manifest"][
                    "binary_sha256"
                ],
            }
        )
        result["telemetry"].update(
            {
                "parent_binding_plan": parent["manifest"]["binding_plan"],
                "candidate_binding_plan": candidate["manifest"]["binding_plan"],
                "parent_source_sha256": parent["manifest"][
                    "outer_source_sha256"
                ],
                "candidate_source_sha256": candidate["manifest"][
                    "outer_source_sha256"
                ],
            }
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
                self.screen_output(
                    output,
                    _params,
                    _kwargs["context_record"],
                    _kwargs["sealed_binary_context"],
                )
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
                    "execution_context": _kwargs.get("context_record"),
                    "sealed_binary_context": _kwargs.get(
                        "sealed_binary_context"
                    ),
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

    def order_biased_tier_result(
        self, tier: dict[str, object]
    ) -> dict[str, object]:
        result = self.tier_result(tier, speedup=1.0)
        effects = [1.0, 2.0, 1.0, 2.0, 1.0]
        pairs = result["replication"]["pairs"]
        self.assertEqual(len(pairs), len(effects))
        for pair, effect in zip(pairs, effects):
            pair["arms"]["treatment"]["primary_ns"] = 500.0 / effect
            pair["effect"] = effect
        summary = paired_summary(pairs, tier["replication"])
        result["replication"]["summary"] = summary
        result["primary"]["value"] = summary["median"]
        return result

    def test_representative_stability_rejects_hidden_order_bias(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        tier = next(
            item
            for item in template["evaluation"]["tiers"]
            if item.get("role") == "representative"
        )
        stable = self.tier_result(tier, speedup=1.0)
        biased = self.order_biased_tier_result(tier)

        admitted, _, diagnostics = runner._baseline_admission(tier, biased)
        promoted, reason = runner._promotion_pass(
            tier, biased, runner._tier_record(stable)
        )

        self.assertEqual(diagnostics["relative_mad"], 0.0)
        self.assertGreater(
            diagnostics["order_stratum_log_skew"],
            diagnostics["limits"]["maximum_order_stratum_log_skew"],
        )
        self.assertFalse(admitted)
        self.assertFalse(promoted)
        self.assertIn("stability gate", reason)

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

            self.assertEqual(
                launches,
                ["screen", "representative", "screen", "representative"],
            )
            self.assertEqual(state["accepted_parent"]["id"], "baseline")
            self.assertEqual(decision["verdict"], "discard")
            self.assertEqual(state["usage"]["candidates_admitted"], 1)
            self.assertEqual(
                len((run_dir / "tier-events.jsonl").read_text().splitlines()), 4
            )

    def test_live_state_validation_allows_candidate_shader_to_differ(self) -> None:
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
            source = (
                ROOT / state["template"]["runtime_artifact"]["source_path"]
            ).resolve()
            original_read_bytes = Path.read_bytes

            def changed_candidate(path: Path) -> bytes:
                payload = original_read_bytes(path)
                return payload + b"\n// candidate" if path == source else payload

            with mock.patch.object(Path, "read_bytes", changed_candidate):
                runner._validate_live_state(ROOT, state)

    def test_initialization_rejects_biased_a_a_before_representative(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        screen = next(
            tier
            for tier in template["evaluation"]["tiers"]
            if tier.get("role") == "proxy"
        )
        biased, _ = adapt_result(
            screen,
            ResultAdapterTests().calibration_v2_output(106, 100),
            "outer_remainder",
        )
        launches: list[str] = []

        def execute(
            _root: Path,
            _run_dir: Path,
            _state: dict[str, object],
            tier: dict[str, object],
            _params: dict[str, str],
            _evaluation_id: str,
            **_kwargs: object,
        ) -> tuple[dict[str, object], dict[str, object]]:
            launches.append(str(tier["id"]))
            return self.seal_tier_evaluation(
                _run_dir,
                tier,
                _params,
                _evaluation_id,
                biased,
            )

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "execute_tier", side_effect=execute
        ):
            run_dir = Path(directory) / "run"
            with self.assertRaisesRegex(ValueError, "A/A calibration is biased"):
                runner.init_run(
                    ROOT,
                    ROOT
                    / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                    run_dir,
                )

            state = runner.load_state(run_dir)
            self.assertEqual(state["status"], "initialization_retryable")
            self.assertEqual(launches, ["screen"])
            rejection = json.loads(
                (run_dir / "baseline-events.jsonl").read_text()
            )
            self.assertEqual(rejection["event"], "baseline_rejected")
            self.assertFalse(rejection["admission"]["admitted"])
            self.assertEqual(state["usage"]["active_evaluator_seconds"], 1.0)

            stable, _ = adapt_result(
                screen,
                ResultAdapterTests().calibration_v2_output(),
                "outer_remainder",
            )
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "representative"
            )
            representative_result = self.tier_result(representative)

            def resume_execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                params: dict[str, str],
                evaluation_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                result = stable if tier["role"] == "proxy" else representative_result
                return self.seal_tier_evaluation(
                    _run_dir, tier, params, evaluation_id, result
                )

            with mock.patch.object(
                runner, "execute_tier", side_effect=resume_execute
            ):
                resumed = runner.resume_initialization(ROOT, run_dir)

            self.assertEqual(resumed["status"], "active")
            self.assertEqual(
                [
                    event["evaluation_id"]
                    for event in runner._events(run_dir / "tier-events.jsonl")
                ],
                [
                    "baseline-screen",
                    "baseline-screen-retry-002",
                    "baseline-representative",
                ],
            )
            self.assertEqual(
                runner.load_state(run_dir)["usage"]["active_evaluator_seconds"],
                3.0,
            )

    def test_initialization_accepts_a_contract_valid_correctness_tier(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        representative = next(
            tier
            for tier in template["evaluation"]["tiers"]
            if tier.get("role") == "representative"
        )
        correctness = copy.deepcopy(representative)
        correctness.update({"id": "correctness", "role": "correctness"})
        correctness["promotion"] = {
            "kind": "all_guards",
            "required_guards": ["all_exact"],
        }
        template["evaluation"]["tiers"].insert(0, correctness)
        validate_template(template, ROOT)
        original_search_tiers = runner._search_tiers

        def search_tiers(sealed_template: dict[str, object]):
            return [correctness, *original_search_tiers(sealed_template)]

        launches: list[str] = []
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "_search_tiers", side_effect=search_tiers
        ), mock.patch.object(
            runner,
            "run_attempt",
            side_effect=self.successful_attempt(self.outer_output(), launches),
        ):
            state = runner.init_run(
                ROOT,
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                Path(directory) / "run",
            )

        self.assertEqual(state["status"], "active")
        self.assertEqual(launches, ["correctness", "screen", "representative"])
        self.assertIn("correctness", state["accepted_parent"]["tiers"])

    def test_accepted_calibration_is_recomputed_from_its_sealed_result(self) -> None:
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
            events = list(runner._events(run_dir / "baseline-events.jsonl"))
            events[0]["admission"]["gpu"]["median"] = 2.0
            (run_dir / "baseline-events.jsonl").write_bytes(
                b"\n".join(runner.canonical_json(event) for event in events)
                + b"\n"
            )

            with self.assertRaisesRegex(ValueError, "disagrees"):
                runner._accepted_baseline_records(
                    run_dir, state["template"]
                )

    def test_trial_retries_one_unstable_inconclusive_screen(self) -> None:
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
                if tier.get("role") == "proxy"
            )
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "representative"
            )
            stable, _ = adapt_result(
                screen,
                ResultAdapterTests().uniform_successor_v2_output(99, 100),
                "outer_remainder",
            )
            unstable, _ = adapt_result(
                screen,
                ResultAdapterTests().varying_successor_v2_output(
                    [90, 130, 90, 80]
                ),
                "outer_remainder",
            )
            representative_result = self.tier_result(
                representative, speedup=state["accepted_parent"]["metric"]
            )
            baseline_usage = runner.load_state(run_dir)["usage"]
            evaluation_ids: list[str] = []
            screen_runs = 0

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                _params: dict[str, str],
                evaluation_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                nonlocal screen_runs
                evaluation_ids.append(evaluation_id)
                if tier["role"] == "proxy":
                    screen_runs += 1
                    result = unstable if screen_runs == 1 else stable
                else:
                    result = representative_result
                return self.seal_tier_evaluation(
                    run_dir,
                    tier,
                    _params,
                    evaluation_id,
                    result,
                )

            with mock.patch.object(runner, "execute_tier", side_effect=execute):
                decision, state = runner.trial(
                    ROOT, run_dir, [], "retry the noisy cheap screen"
                )

            self.assertEqual(
                evaluation_ids,
                [
                    "candidate-001-screen",
                    "candidate-001-screen-retry-002",
                    "candidate-001-representative",
                ],
            )
            self.assertEqual(
                [attempt["disposition"] for attempt in decision["screen_attempts"]],
                ["retry", "advance"],
            )
            self.assertEqual(decision["verdict"], "discard")
            candidate_screen_events = [
                event
                for event in runner._events(run_dir / "tier-events.jsonl")
                if str(event["evaluation_id"]).startswith(
                    "candidate-001-screen"
                )
            ]
            reconstructed = runner.load_state(run_dir)
            self.assertEqual(len(candidate_screen_events), 2)
            self.assertEqual(
                reconstructed["usage"]["active_evaluator_seconds"],
                baseline_usage["active_evaluator_seconds"] + 3.0,
            )
            self.assertEqual(
                reconstructed["usage"]["reserve_invocations"], {}
            )

    def test_trial_budget_rejection_leaves_no_fake_inflight_attempt(self) -> None:
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
            tier_event_count = len(
                (run_dir / "tier-events.jsonl").read_text().splitlines()
            )

            with mock.patch.object(
                runner,
                "execute_tier",
                side_effect=runner.BudgetExhausted(
                    "screen budget is reserved for validation"
                ),
            ):
                decision, state = runner.trial(
                    ROOT, run_dir, [], "budget cannot admit the screen"
                )

            self.assertEqual(decision["verdict"], "invalid")
            self.assertEqual(decision["evaluation_ids"], [])
            self.assertFalse((run_dir / "inflight.json").exists())
            self.assertEqual(
                len((run_dir / "tier-events.jsonl").read_text().splitlines()),
                tier_event_count,
            )
            self.assertEqual(state["accepted_parent"]["id"], "baseline")

    def test_recovered_keep_uses_the_same_calibrated_tiers_as_normal_keep(self) -> None:
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
            pre_trial = copy.deepcopy(state)
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "representative"
            )
            improved = self.tier_result(
                representative,
                speedup=float(state["accepted_parent"]["metric"]) * 2.0,
            )
            screen = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier.get("role") == "proxy"
            )
            neutral, _ = adapt_result(
                screen,
                ResultAdapterTests().uniform_successor_v2_output(100, 100),
                "outer_remainder",
            )

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                params: dict[str, str],
                evaluation_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                result = neutral if tier["role"] == "proxy" else improved
                return self.seal_tier_evaluation(
                    run_dir, tier, params, evaluation_id, result
                )

            with mock.patch.object(runner, "execute_tier", side_effect=execute):
                decision, normal = runner.trial(
                    ROOT, run_dir, [], "candidate clears representative"
                )

            pre_trial["phase_checkpoint"] = copy.deepcopy(
                normal["phase_checkpoint"]
            )
            recovered, _ = runner._recover_committed_candidate(
                ROOT,
                run_dir,
                pre_trial,
                {
                    "candidate_id": decision["candidate_id"],
                    "editable_paths_sha256": normal["accepted_parent"][
                        "editable_paths_sha256"
                    ],
                },
            )

            self.assertTrue(recovered)
            self.assertEqual(decision["verdict"], "keep")
            self.assertEqual(
                pre_trial["accepted_parent"]["tiers"],
                normal["accepted_parent"]["tiers"],
            )
            self.assertTrue(
                pre_trial["accepted_parent"]["tiers"]["screen"][
                    "calibration"
                ]["admitted"]
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

    def test_recovery_preserves_and_verifies_published_artifacts(self) -> None:
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
            params = state["accepted_parent"]["params"]
            source = (
                ROOT
                / "crates/jolt-kernels/src/metal/solinas/outer_remainder/"
                "shader.metal"
            )
            artifact = runtime_artifacts.materialize_outer_artifact(
                run_dir,
                source,
                "b_only_v1",
                runtime_artifacts.outer_dispatch_from_params(params),
            )
            context = {
                "kind": "outer_msl_v1",
                "parent": artifact,
                "candidate": copy.deepcopy(artifact),
            }
            runner.write_inflight(
                run_dir,
                {
                    "schema_version": 2,
                    "kind": "candidate",
                    "candidate_id": "candidate-001",
                    "evaluation_id": "candidate-001-screen",
                    "tier_id": "screen",
                    "params": params,
                    "editable_paths_sha256": state["fingerprint"][
                        "editable_paths_sha256"
                    ],
                    "execution_context": context,
                    "started_at": runner.utc_now(),
                },
            )

            @contextmanager
            def recovery_lease(*_args: object, **_kwargs: object):
                yield {}

            with mock.patch.object(
                runner, "evaluator_lease", side_effect=recovery_lease
            ):
                runner.recover(ROOT, run_dir)

            event = json.loads(
                (run_dir / "tier-events.jsonl").read_text().splitlines()[-1]
            )

        self.assertEqual(event["attempt"]["execution_context"], context)
        self.assertTrue(event["attempt"]["artifact_context_valid"])

    def test_recovery_marks_a_required_unpublished_context_invalid(self) -> None:
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
                    "kind": "candidate",
                    "candidate_id": "candidate-001",
                    "evaluation_id": "candidate-001-screen",
                    "tier_id": "screen",
                    "params": state["accepted_parent"]["params"],
                    "editable_paths_sha256": state["fingerprint"][
                        "editable_paths_sha256"
                    ],
                    "started_at": runner.utc_now(),
                },
            )

            @contextmanager
            def recovery_lease(*_args: object, **_kwargs: object):
                yield {}

            with mock.patch.object(
                runner, "evaluator_lease", side_effect=recovery_lease
            ):
                runner.recover(ROOT, run_dir)

            event = json.loads(
                (run_dir / "tier-events.jsonl").read_text().splitlines()[-1]
            )

        self.assertFalse(event["attempt"]["artifact_context_valid"])
        self.assertIn("not published", event["attempt"]["error"])

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
                    **_kwargs,
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

    def test_resume_initialization_recovers_before_first_inflight_record(self) -> None:
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
            state["status"] = "initializing"
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
                    **_kwargs,
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
            state["accepted_parent"]["tiers"]["screen"]["relative_mad"] = 0.0
            runner.write_state(run_dir, state)
            screen_output = ResultAdapterTests().successor_v2_output()
            for sample in screen_output["samples"]:
                parent = sample["parent"]
                parent.update(ResultAdapterTests().successor_v2_arm(90))
            screen_output["metrics"] = {
                "successor_speedup": 0.9,
                "paired_speedups": [0.9] * 4,
            }
            total_ns = sum(
                arm["resource_gpu_active_ns"]
                for record in [
                    screen_output["excluded_warmup"],
                    *screen_output["samples"],
                ]
                for arm in (record["parent"], record["candidate"])
            )
            screen_output["resources"] = {
                "gpu_active_total_ns": total_ns,
                "gpu_seconds": total_ns / 1e9,
            }
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
                return self.seal_tier_evaluation(
                    run_dir,
                    tier,
                    params,
                    evaluation_id,
                    screen_result,
                )

            with mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(runner, "execute_tier", side_effect=execute):
                decision, state = runner.trial(
                    ROOT, run_dir, [], "candidate rejected by the proxy"
                )

            self.assertEqual(launches, ["screen"])
            self.assertEqual(decision["verdict"], "discard")
            self.assertTrue(decision["phase_checkpoint"]["passed"])

    def test_checkpoint_miss_ends_the_mechanism_phase(self) -> None:
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
                if tier["id"] == "screen"
            )
            state["accepted_parent"]["tiers"]["screen"]["relative_mad"] = 0.0
            runner.write_state(run_dir, state)
            screen_result, _ = adapt_result(
                screen,
                ResultAdapterTests().uniform_successor_v2_output(
                    240_000_000, 240_000_000
                ),
                "outer_remainder",
            )
            launches: list[str] = []

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                params: dict[str, str],
                evaluation_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                launches.append(str(tier["id"]))
                return self.seal_tier_evaluation(
                    run_dir,
                    tier,
                    params,
                    evaluation_id,
                    screen_result,
                )

            with mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(runner, "execute_tier", side_effect=execute):
                decision, state = runner.trial(
                    ROOT, run_dir, [], "candidate misses the phase checkpoint"
                )

            self.assertEqual(launches, ["screen"])
            self.assertFalse(decision["phase_checkpoint"]["passed"])
            self.assertEqual(state["status"], "phase_exhausted")
            with mock.patch.object(runner, "_validate_live_state"):
                with self.assertRaisesRegex(ValueError, "not active"):
                    runner.trial(ROOT, run_dir, [], "phase is terminal")

    def test_direct_lane_bypasses_ranking_after_the_required_checkpoint(self) -> None:
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
                if tier["role"] == "representative"
            )
            screen = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier["role"] == "proxy"
            )
            screen_result, _ = adapt_result(
                screen,
                ResultAdapterTests().uniform_successor_v2_output(100, 100),
                "outer_remainder",
            )
            launches: list[str] = []

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                params: dict[str, str],
                evaluation_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                launches.append(str(tier["id"]))
                result = (
                    screen_result
                    if tier["role"] == "proxy"
                    else self.tier_result(representative, speedup=5.0)
                )
                return self.seal_tier_evaluation(
                    run_dir, tier, params, evaluation_id, result
                )

            with mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(
                runner, "execute_tier", side_effect=execute
            ), mock.patch.object(
                runner, "_promotion_pass", return_value=(True, "promoted")
            ), mock.patch.object(
                runner, "_phase_success", return_value=(True, "phase passed")
            ):
                decision, state = runner.trial(
                    ROOT,
                    run_dir,
                    [],
                    "architectural candidate",
                    direct_to_representative=True,
                    direct_reason="the proxy cannot model this ownership change",
                )

            self.assertEqual(launches, ["screen", "representative"])
            self.assertEqual(decision["verdict"], "keep")
            self.assertEqual(decision["lane"]["effective"], "representative_direct")
            self.assertEqual(
                decision["lane"]["reason"],
                "the proxy cannot model this ownership change",
            )
            self.assertTrue(decision["phase_checkpoint"]["passed"])
            self.assertTrue(decision["lane"]["checkpoint_probe"])
            self.assertEqual(
                state["accepted_parent"]["lane"], decision["lane"]
            )
            self.assertEqual(state["proxy"]["status"], "disabled")
            self.assertEqual(
                runner.load_state(run_dir)["proxy"]["status"], "disabled"
            )
            launches.clear()
            with mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(
                runner, "execute_tier", side_effect=execute
            ), mock.patch.object(
                runner, "_promotion_pass", return_value=(True, "promoted")
            ), mock.patch.object(
                runner, "_phase_success", return_value=(True, "phase passed")
            ):
                second, _ = runner.trial(
                    ROOT,
                    run_dir,
                    [],
                    "follow-up after proxy disable",
                )

            self.assertEqual(launches, ["representative"])
            self.assertFalse(second["lane"]["requested"])
            self.assertEqual(
                second["lane"]["effective"], "representative_direct"
            )
            self.assertFalse(second["lane"]["checkpoint_probe"])
            self.assertEqual(
                second["phase_checkpoint"], decision["phase_checkpoint"]
            )

    def test_fixed_sentinel_calibration_enables_proxy_ranking(self) -> None:
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
            state["proxy"]["status"] = "pending_calibration"
            state["proxy"]["reason"] = "test pending calibration"
            runner.write_state(run_dir, state)
            with mock.patch.object(
                runner, "_validate_live_state"
            ), self.assertRaisesRegex(ValueError, "not calibrated"):
                runner.trial(ROOT, run_dir, [], "must calibrate first")
            sentinels = state["template"]["search_policy"]["proxy_calibration"][
                "sentinels"
            ]
            sentinel_rank = {
                sentinel["id"]: index for index, sentinel in enumerate(sentinels)
            }
            launches: list[str] = []

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                _params: dict[str, str],
                evaluation_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                sentinel_id = next(
                    sentinel_id
                    for sentinel_id in sentinel_rank
                    if f"-{sentinel_id}-" in evaluation_id
                )
                rank = sentinel_rank[sentinel_id]
                score = 1.0 + rank * 0.1
                if tier["role"] == "representative":
                    score = 4.0 + rank * 0.4
                launches.append(evaluation_id)
                return (
                    {"attempt": {"error": None, "outcome": "success"}},
                    self.tier_result(tier, speedup=score),
                )

            with mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(runner, "execute_tier", side_effect=execute):
                event, state = runner.calibrate_proxy(ROOT, run_dir)

            self.assertEqual(len(launches), 2 * len(sentinels))
            self.assertEqual(event["event"], "proxy_calibrated")
            self.assertEqual(state["proxy"]["status"], "enabled")
            self.assertEqual(
                state["proxy"]["calibration"]["kendall_tau_b"], 1.0
            )

    def test_proxy_calibration_terminal_event_recovers_without_rerunning(self) -> None:
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
            state["proxy"]["status"] = "pending_calibration"
            state["proxy"]["reason"] = "test pending calibration"
            state["proxy"]["calibration"] = None
            state["search_started_at"] = "2026-08-06T00:00:00Z"
            runner.write_state(run_dir, state)
            sentinels = state["template"]["search_policy"][
                "proxy_calibration"
            ]["sentinels"]
            records = [
                {
                    "id": sentinel["id"],
                    "proxy_score": 1.0 + index * 0.1,
                    "representative_score": 4.0 + index * 0.4,
                }
                for index, sentinel in enumerate(sentinels)
            ]
            calibration = runner._proxy_calibration_decision(
                state["template"]["search_policy"]["proxy_calibration"],
                records,
            )
            evaluation_ids = [
                f"sealed-{index}"
                for index in range(2 * len(sentinels))
            ]
            with mock.patch.object(
                runner,
                "write_state",
                side_effect=RuntimeError("simulated post-ledger crash"),
            ), self.assertRaisesRegex(RuntimeError, "post-ledger crash"):
                runner._finish_proxy_calibration(
                    run_dir, state, calibration, evaluation_ids
                )

            with mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(runner, "_assert_frozen"):
                event, recovered = runner.calibrate_proxy(ROOT, run_dir)

            self.assertEqual(event["event"], "proxy_calibrated")
            self.assertEqual(recovered["proxy"]["status"], "enabled")
            self.assertEqual(
                recovered["search_started_at"], "2026-08-06T00:00:00Z"
            )
            terminals = [
                item
                for item in runner._events(run_dir / "proxy-events.jsonl")
                if item.get("event") == "proxy_calibrated"
            ]
            self.assertEqual(len(terminals), 1)

    def test_proxy_false_negative_audit_disables_screening(self) -> None:
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
            runner.append_event(
                run_dir / "candidate-events.jsonl",
                {
                    "event": "candidate_admitted",
                    "candidate_id": "candidate-001",
                },
            )
            screen = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier["role"] == "proxy"
            )
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier["role"] == "representative"
            )
            screen_result, _ = adapt_result(
                screen,
                ResultAdapterTests().uniform_successor_v2_output(100, 100),
                "outer_remainder",
            )
            launches: list[str] = []

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                params: dict[str, str],
                evaluation_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                launches.append(str(tier["id"]))
                result = (
                    screen_result
                    if tier["role"] == "proxy"
                    else self.tier_result(representative, speedup=5.0)
                )
                return self.seal_tier_evaluation(
                    run_dir,
                    tier,
                    params,
                    evaluation_id,
                    result,
                )

            with mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(
                runner, "execute_tier", side_effect=execute
            ), mock.patch.object(
                runner,
                "_successor_screen_disposition",
                return_value=("discard", "proxy clear loss"),
            ), mock.patch.object(
                runner, "_promotion_pass", return_value=(True, "promoted")
            ), mock.patch.object(
                runner, "_phase_success", return_value=(True, "phase passed")
            ):
                decision, state = runner.trial(
                    ROOT, run_dir, [], "proxy false-negative sentinel"
                )

            self.assertEqual(launches, ["screen", "representative"])
            self.assertFalse(decision["proxy_audit"]["ranking_ok"])
            self.assertEqual(state["proxy"]["status"], "disabled")

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

    def test_run_contract_is_immutable_and_state_is_self_digested(self) -> None:
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
            run_before = (run_dir / "run.json").read_bytes()
            contract = json.loads(run_before)
            self.assertEqual(
                contract["record_kind"], "metal_autoresearch_run_contract_v1"
            )
            self.assertIn("run_sha256", contract)
            raw = json.loads((run_dir / "state.json").read_text())
            self.assertIn("state_sha256", raw)
            self.assertFalse((run_dir / "run.sha256").exists())
            state = runner.load_state(run_dir)
            state["status"] = "active"
            runner.write_state(run_dir, state)
            self.assertEqual((run_dir / "run.json").read_bytes(), run_before)
            raw = json.loads((run_dir / "state.json").read_text())
            raw["status"] = "forged"
            (run_dir / "state.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "state digest"):
                runner.load_state(run_dir)

    def test_fresh_run_publication_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            run_dir = parent / "run"
            with mock.patch.object(
                runner,
                "_write_run_contract",
                side_effect=RuntimeError("simulated publication crash"),
            ), self.assertRaisesRegex(RuntimeError, "publication crash"):
                runner.init_run(
                    ROOT,
                    ROOT
                    / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                    run_dir,
                )

            self.assertFalse(run_dir.exists())
            self.assertEqual(list(parent.glob(".run.initializing-*")), [])

    def test_candidate_admission_has_a_recoverable_transaction(self) -> None:
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
            append = runner.append_event

            def interrupt(path: Path, event: dict[str, object]) -> None:
                if path.name == "candidate-events.jsonl":
                    raise RuntimeError("simulated admission crash")
                append(path, event)

            with mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(
                runner, "append_event", side_effect=interrupt
            ), self.assertRaisesRegex(RuntimeError, "admission crash"):
                runner.trial(ROOT, run_dir, [], "interrupted admission")

            inflight = runner.read_json(run_dir / "inflight.json")
            self.assertEqual(inflight["kind"], "candidate_pending")
            state = runner.load_state(run_dir)
            with mock.patch.object(
                runner, "_restore_with_quarantine", return_value=None
            ):
                recovered = runner._recover_pending_candidate(
                    ROOT, run_dir, state, inflight
                )
            self.assertEqual(recovered["status"], "active")
            self.assertFalse((run_dir / "inflight.json").exists())

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
            screen = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier["role"] == "proxy"
            )
            representative = next(
                tier
                for tier in state["template"]["evaluation"]["tiers"]
                if tier["role"] == "representative"
            )
            screen_result, _ = adapt_result(
                screen,
                ResultAdapterTests().uniform_successor_v2_output(100, 100),
                "outer_remainder",
            )
            representative_result = self.tier_result(
                representative,
                speedup=float(state["accepted_parent"]["metric"]) * 2.0,
            )

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                params: dict[str, str],
                evaluation_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                result = (
                    screen_result
                    if tier["role"] == "proxy"
                    else representative_result
                )
                return self.seal_tier_evaluation(
                    run_dir, tier, params, evaluation_id, result
                )

            with mock.patch.object(
                runner, "execute_tier", side_effect=execute
            ):
                decision, state = runner.trial(
                    ROOT, run_dir, [], "candidate committed before interruption"
                )
            self.assertEqual(decision["verdict"], "keep")
            state["status"] = "kernel_transferred"
            candidate_id = decision["candidate_id"]
            editable_digest = state["accepted_parent"][
                "editable_paths_sha256"
            ]
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

    def test_recovery_preserves_a_terminal_phase_checkpoint(self) -> None:
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
                if tier["role"] == "proxy"
            )
            screen_result, _ = adapt_result(
                screen,
                ResultAdapterTests().uniform_successor_v2_output(
                    240_000_000, 240_000_000
                ),
                "outer_remainder",
            )

            def execute(
                _root: Path,
                _run_dir: Path,
                _state: dict[str, object],
                tier: dict[str, object],
                params: dict[str, str],
                evaluation_id: str,
                **_kwargs: object,
            ) -> tuple[dict[str, object], dict[str, object]]:
                return self.seal_tier_evaluation(
                    run_dir, tier, params, evaluation_id, screen_result
                )

            with mock.patch.object(
                runner, "_validate_live_state"
            ), mock.patch.object(runner, "execute_tier", side_effect=execute):
                decision, state = runner.trial(
                    ROOT, run_dir, [], "candidate misses checkpoint before crash"
                )
            candidate_id = decision["candidate_id"]
            state["status"] = "active"

            recovered, _ = runner._recover_committed_candidate(
                ROOT,
                run_dir,
                state,
                {
                    "candidate_id": candidate_id,
                    "editable_paths_sha256": state["accepted_parent"][
                        "editable_paths_sha256"
                    ],
                },
            )

            self.assertFalse(recovered)
            self.assertEqual(state["status"], "phase_exhausted")


class DispatchAndGoalTests(unittest.TestCase):
    def test_proxy_rank_calibration_enables_or_disables_screening(self) -> None:
        policy = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )["search_policy"]["proxy_calibration"]
        monotone = [
            {"id": "a", "proxy_score": 1.0, "representative_score": 4.0},
            {"id": "b", "proxy_score": 1.1, "representative_score": 4.4},
            {"id": "c", "proxy_score": 1.2, "representative_score": 4.8},
        ]
        admitted = runner._proxy_calibration_decision(policy, monotone)
        self.assertEqual(admitted["status"], "enabled")
        self.assertEqual(admitted["kendall_tau_b"], 1.0)
        self.assertEqual(admitted["material_inversions"], 0)

        inverted = copy.deepcopy(monotone)
        inverted[2]["representative_score"] = 3.0
        rejected = runner._proxy_calibration_decision(policy, inverted)
        self.assertEqual(rejected["status"], "disabled")
        self.assertGreater(rejected["material_inversions"], 0)

    def test_both_cli_surfaces_accept_the_direct_lane(self) -> None:
        argv = [
            "trial",
            "run-dir",
            "--summary",
            "architectural candidate",
            "--direct-to-representative",
            "--direct-reason",
            "proxy cannot model ownership",
        ]
        for parser in (runner.parser(), metal_autoresearch.parser()):
            with self.subTest(parser=parser.prog):
                args = parser.parse_args(argv)
                self.assertTrue(args.direct_to_representative)
                self.assertEqual(args.direct_reason, "proxy cannot model ownership")

    def test_calendar_queue_preserves_validation_reserves(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        state = {
            "created_at": runner.utc_now(),
            "template": template,
            "usage": {
                "calendar_seconds": 14_400.0,
                "reserve_invocations": {},
            },
        }
        with mock.patch.object(
            runner, "_calendar_seconds", return_value=14_400.0
        ):
            queue = runner._queue_budget_for_timeout(
                state, 1_800.0, "representative_revalidation"
            )

        self.assertEqual(queue, 28_800.0)

    def test_phase_checkpoint_uses_validated_candidate_gpu_phases(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        tier = next(
            tier
            for tier in template["evaluation"]["tiers"]
            if tier["id"] == "screen"
        )
        result, _ = adapt_result(
            tier,
            ResultAdapterTests().uniform_successor_v2_output(100, 100),
            "outer_remainder",
        )
        state = {
            "template": template,
            "usage": {"candidates_admitted": 1},
        }

        checkpoint = runner._phase_checkpoint(state, result)
        self.assertTrue(checkpoint["due"])
        self.assertTrue(checkpoint["passed"])
        observed = {
            metric["name"]: metric["observed_ms"]
            for metric in checkpoint["metrics"]
        }
        self.assertEqual(observed["openings_gpu_active_ms"], 0.000013)

        next(
            metric
            for metric in template["mechanism_phase"]["checkpoint"]["metrics"]
            if metric["name"] == "openings_gpu_active_ms"
        )["threshold"] = 0.000012
        failed = runner._phase_checkpoint(state, result)
        self.assertFalse(failed["passed"])

    def test_phase_success_requires_gain_and_latency_ceiling(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        state = {"template": template}
        parent = {"metric": 4.0}
        result = {
            "primary": {"value": 4.2},
            "replication": {
                "pairs": [
                    {"arms": {"treatment": {"primary_ns": 209_000_000}}}
                    for _ in range(5)
                ]
            },
        }
        passed, _ = runner._phase_success(state, result, parent)
        self.assertTrue(passed)

        for pair in result["replication"]["pairs"]:
            pair["arms"]["treatment"]["primary_ns"] = 211_000_000
        passed, reason = runner._phase_success(state, result, parent)
        self.assertFalse(passed)
        self.assertIn("latency", reason)

    def test_search_phase_timebox_is_stricter_than_the_run_budget(self) -> None:
        template = json.loads(
            (
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json"
            ).read_text()
        )
        state = {
            "created_at": runner.utc_now(),
            "template": template,
            "usage": {
                "calendar_seconds": 0.0,
                "candidates_admitted": 2,
            },
        }
        with self.assertRaisesRegex(runner.BudgetExhausted, "phase candidate"):
            runner._require_search_phase_budget(state)

        state["usage"]["candidates_admitted"] = 1
        with mock.patch.object(
            runner,
            "_phase_calendar_seconds",
            return_value=10801.0,
        ), self.assertRaisesRegex(runner.BudgetExhausted, "phase calendar"):
            runner._require_search_phase_budget(state)

    def test_v2_init_rejects_an_edit_racing_the_baseline_snapshot(self) -> None:
        legacy = mock.Mock()
        legacy.path_digest.side_effect = ["snapshot", "live", "frozen"]
        legacy.outside_editable_worktree_digest.return_value = "outside"
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "_legacy", return_value=legacy
        ), mock.patch.object(runner, "_continue_binary_sealing") as seal:
            with self.assertRaisesRegex(ValueError, "baseline snapshot changed"):
                runner.init_run(
                    ROOT,
                    ROOT
                    / "crates/jolt-kernels/autoresearch/outer_remainder.v2.template.json",
                    Path(directory) / "run",
                )

        seal.assert_not_called()

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
        self.assertEqual(result["lifecycle"], "fresh_init")
        self.assertTrue(result["fresh_init_eligible"])

    def test_schema_one_template_is_readable_but_cannot_start_a_fresh_run(self) -> None:
        template = (
            ROOT
            / "crates/jolt-kernels/autoresearch/outer_remainder.template.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with self.assertRaisesRegex(ValueError, "existing-run-only"):
                metal_autoresearch.command_init(
                    SimpleNamespace(root=ROOT, template=template, run_dir=run_dir)
                )
            self.assertFalse(run_dir.exists())

        with mock.patch("builtins.print") as emit:
            exit_code = metal_autoresearch.command_validate_template(
                SimpleNamespace(root=ROOT, template=template)
            )
        result = json.loads(emit.call_args.args[0])
        self.assertEqual(exit_code, 0)
        self.assertEqual(result["lifecycle"], "existing_runs_only")
        self.assertFalse(result["fresh_init_eligible"])

    def test_schema_one_cli_init_rejects_without_acquiring_the_evaluator(self) -> None:
        args = SimpleNamespace(
            command="init",
            root=ROOT,
            template=(
                ROOT
                / "crates/jolt-kernels/autoresearch/outer_remainder.template.json"
            ),
            run_dir=ROOT / "unused-schema-one-run",
            handler=metal_autoresearch.command_init,
        )
        parser = mock.Mock()
        parser.parse_args.return_value = args
        with mock.patch.object(
            metal_autoresearch, "parser", return_value=parser
        ), mock.patch.object(metal_autoresearch, "evaluator_lock") as lock:
            exit_code = metal_autoresearch.main()

        self.assertEqual(exit_code, 2)
        lock.assert_not_called()

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
