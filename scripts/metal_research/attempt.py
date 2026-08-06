from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


EVALUATOR_LOCK_PATH = Path("/private/tmp/jolt-metal-autoresearch-evaluator.lock")
EVALUATOR_LOCK_HELD_ENV = "JOLT_METAL_EVAL_LOCK_HELD"
LOCK_POLL_SECONDS = 0.1


class EvaluatorLeaseTimeout(TimeoutError):
    def __init__(self, waited_seconds: float):
        super().__init__("evaluator lease wait exhausted the calendar budget")
        self.waited_seconds = waited_seconds


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _read_lock_record() -> dict[str, Any]:
    try:
        value = json.loads(EVALUATOR_LOCK_PATH.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


@contextmanager
def evaluator_lease(
    owner: dict[str, Any], timeout_seconds: Optional[float] = None
) -> Iterator[dict[str, Any]]:
    wait_started = time.monotonic()
    inherited_token = os.environ.get(EVALUATOR_LOCK_HELD_ENV)
    if inherited_token is not None and secrets.compare_digest(
        str(_read_lock_record().get("token", "")), inherited_token
    ):
        acquired = time.monotonic()
        telemetry = {
            "queue_wait_seconds": acquired - wait_started,
            "exclusive_lease_seconds": 0.0,
        }
        try:
            yield telemetry
        finally:
            telemetry["exclusive_lease_seconds"] = time.monotonic() - acquired
        return

    descriptor = os.open(EVALUATOR_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    previous_token = inherited_token
    acquired_lock = False
    try:
        if timeout_seconds is None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            acquired_lock = True
        else:
            deadline = wait_started + max(0.0, timeout_seconds)
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired_lock = True
                    break
                except BlockingIOError:
                    now = time.monotonic()
                    if now >= deadline:
                        raise EvaluatorLeaseTimeout(now - wait_started)
                    time.sleep(min(LOCK_POLL_SECONDS, deadline - now))
        acquired = time.monotonic()
        token = secrets.token_hex(32)
        os.environ[EVALUATOR_LOCK_HELD_ENV] = token
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            _encoded(
                {
                    **owner,
                    "pid": os.getpid(),
                    "locked_at": _utc_now(),
                    "token": token,
                }
            ),
        )
        os.fsync(descriptor)
        telemetry = {
            "queue_wait_seconds": acquired - wait_started,
            "exclusive_lease_seconds": 0.0,
            "lock_fd": descriptor,
        }
        try:
            yield telemetry
        finally:
            telemetry["exclusive_lease_seconds"] = time.monotonic() - acquired
    finally:
        if acquired_lock:
            if previous_token is None:
                os.environ.pop(EVALUATOR_LOCK_HELD_ENV, None)
            else:
                os.environ[EVALUATOR_LOCK_HELD_ENV] = previous_token
            os.ftruncate(descriptor, 0)
            os.fsync(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _parse_unique_result(stdout: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    if len(records) != 1:
        raise ValueError("evaluator must emit exactly one JSON result object")
    return records[0]


def _command(evaluator: dict[str, Any], params: dict[str, str]) -> list[str]:
    command = list(evaluator["command"])
    for binding in evaluator.get("parameter_bindings", []):
        parameter = binding.get("parameter")
        flag = binding.get("flag")
        if parameter not in params or not isinstance(flag, str) or not flag:
            raise ValueError("evaluator parameter binding is incomplete")
        command.extend([flag, params[parameter]])
    return command


def _environment(
    evaluator: dict[str, Any],
    params: dict[str, str],
    artifact_dir: Path,
    context_env: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    result = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("JOLT_METAL_")
        and not name.startswith("JOLT_AUTORESEARCH_")
    }
    token = os.environ.get(EVALUATOR_LOCK_HELD_ENV)
    if token is not None:
        result[EVALUATOR_LOCK_HELD_ENV] = token
    declared = {
        str(name): str(value) for name, value in evaluator.get("env", {}).items()
    }
    if any(name.startswith("JOLT_AUTORESEARCH_") for name in declared):
        raise ValueError("evaluator env cannot set controller-owned state")
    result.update(declared)
    if any(name.startswith("JOLT_AUTORESEARCH_") for name in params):
        raise ValueError("search parameters cannot set controller-owned state")
    result.update(params)
    context_env = context_env or {}
    allowed_context = {
        "JOLT_AUTORESEARCH_PARENT_ARTIFACT",
        "JOLT_AUTORESEARCH_CANDIDATE_ARTIFACT",
    }
    if (context_env and set(context_env) != allowed_context) or any(
        not isinstance(value, str) or not value for value in context_env.values()
    ):
        raise ValueError("controller evaluator context is invalid")
    result.update(context_env)
    result["JOLT_AUTORESEARCH_EVAL_DIR"] = str(artifact_dir)
    return result


def _stop_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate()


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recorded_process_owns_lease(identity: dict[str, Any]) -> bool:
    descriptor = os.open(EVALUATOR_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record = _read_lock_record()
            if (
                record.get("evaluation_id") != identity.get("evaluation_id")
                or record.get("launch_token") != identity.get("launch_token")
            ):
                raise RuntimeError("the evaluator lease belongs to another launch")
            controller_pid = record.get("pid")
            if type(controller_pid) is int and _process_exists(controller_pid):
                raise RuntimeError("the evaluator controller is still running")
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _recorded_process_command(pid: int) -> Optional[str]:
    observed = subprocess.run(
        ["ps", "-ww", "-o", "command=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )
    if observed.returncode != 0:
        return None
    return observed.stdout.strip()


def stop_recorded_process_group(identity: dict[str, Any]) -> None:
    pid = identity.get("pid")
    pgid = identity.get("pgid")
    if type(pid) is not int or type(pgid) is not int or pid <= 1 or pgid <= 1:
        raise ValueError("recorded evaluator process identity is invalid")
    if pid != pgid:
        raise ValueError("recorded evaluator is not its process-group leader")
    if not _recorded_process_owns_lease(identity):
        return
    try:
        observed = os.getpgid(pid)
    except ProcessLookupError:
        return
    if observed != pgid:
        raise ValueError("recorded evaluator process group no longer matches")
    command = _recorded_process_command(pid)
    if command is None:
        return
    if (
        "scripts/metal_research/process_wrapper.py" not in command
        or str(identity.get("launch_token")) not in command
    ):
        raise RuntimeError("recorded evaluator command no longer matches")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not _process_group_exists(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_group_exists(pgid):
        raise RuntimeError("recorded evaluator process group did not stop")


def _tracked_command(
    root: Path, command: list[str], process_tracking: Optional[dict[str, str]]
) -> tuple[list[str], Optional[dict[str, str]]]:
    if process_tracking is None:
        return command, None
    required = {"evaluation_id", "launch_token", "identity_path"}
    if set(process_tracking) != required or not all(
        isinstance(process_tracking[name], str) and process_tracking[name]
        for name in required
    ):
        raise ValueError("evaluator process tracking is invalid")
    identity_path = Path(process_tracking["identity_path"])
    if not identity_path.is_absolute():
        raise ValueError("evaluator process identity path must be absolute")
    wrapper = root / "scripts/metal_research/process_wrapper.py"
    tracked = [
        sys.executable,
        str(wrapper),
        "--identity-path",
        str(identity_path),
        "--evaluation-id",
        process_tracking["evaluation_id"],
        "--launch-token",
        process_tracking["launch_token"],
        "--",
        *command,
    ]
    return tracked, process_tracking


def run_attempt(
    root: Path,
    evaluator: dict[str, Any],
    params: dict[str, str],
    evaluation_dir: Path,
    tier_id: str,
    queue_timeout_seconds: Optional[float] = None,
    process_tracking: Optional[dict[str, str]] = None,
    context_env: Optional[dict[str, str]] = None,
    context_record: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = evaluation_dir / "artifacts"
    command = _command(evaluator, params)
    launch_command, tracking = _tracked_command(root, command, process_tracking)
    attempt_started_at = _utc_now()
    stdout = ""
    stderr = ""
    output: Optional[dict[str, Any]] = None
    outcome = "launch_error"
    error_message: Optional[str] = None
    subprocess_wall = 0.0
    lease = {
        "queue_wait_seconds": 0.0,
        "exclusive_lease_seconds": 0.0,
    }

    try:
        lease_owner = {"tier_id": tier_id, "command": command}
        if tracking is not None:
            lease_owner.update(
                {
                    "evaluation_id": tracking["evaluation_id"],
                    "launch_token": tracking["launch_token"],
                }
            )
        with evaluator_lease(lease_owner, queue_timeout_seconds) as lease:
            started = time.monotonic()
            try:
                if tracking is not None and "lock_fd" not in lease:
                    raise OSError("tracked evaluator did not inherit the scheduler lease")
                process = subprocess.Popen(
                    launch_command,
                    cwd=root,
                    env=_environment(
                        evaluator, params, artifact_dir, context_env
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                    pass_fds=(int(lease["lock_fd"]),) if "lock_fd" in lease else (),
                )
                try:
                    stdout, stderr = process.communicate(
                        timeout=float(evaluator["timeout_seconds"])
                    )
                except subprocess.TimeoutExpired:
                    stdout, stderr = _stop_process_group(process)
                    subprocess_wall = time.monotonic() - started
                    outcome = "timeout"
                    error_message = "evaluator timed out"
                except BaseException:
                    _stop_process_group(process)
                    raise
                else:
                    subprocess_wall = time.monotonic() - started
                    if process.returncode != 0:
                        outcome = "nonzero_exit"
                        error_message = (
                            f"evaluator exited with status {process.returncode}"
                        )
                    else:
                        try:
                            output = _parse_unique_result(stdout)
                        except ValueError as error:
                            outcome = "invalid_result"
                            error_message = str(error)
                        else:
                            outcome = "success"
            except (OSError, ValueError) as error:
                subprocess_wall = time.monotonic() - started
                outcome = "launch_error"
                error_message = str(error)
    except EvaluatorLeaseTimeout as error:
        lease["queue_wait_seconds"] = error.waited_seconds
        outcome = "lease_timeout"
        error_message = str(error)

    (evaluation_dir / "stdout").write_text(stdout)
    (evaluation_dir / "stderr").write_text(stderr)
    result_sha256 = None
    if output is not None:
        result_bytes = _encoded(output)
        (evaluation_dir / "raw-result.json").write_bytes(result_bytes)
        result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    attempt = {
        "schema_version": 1,
        "tier_id": tier_id,
        "outcome": outcome,
        "error": error_message,
        "command": command,
        "started_at": attempt_started_at,
        "controller": {
            "queue_wait_seconds": float(lease["queue_wait_seconds"]),
            "exclusive_lease_seconds": float(lease["exclusive_lease_seconds"]),
            "subprocess_wall_seconds": subprocess_wall,
        },
        "resources": {
            "gpu_active_seconds": None,
            "gpu_active_charge_seconds": subprocess_wall,
            "gpu_active_charge_kind": "conservative_wall_upper_bound",
        },
        "result_sha256": result_sha256,
        "process_tracking": tracking,
        "execution_context": context_record,
    }
    (evaluation_dir / "attempt.json").write_bytes(_encoded(attempt))
    return attempt, output
