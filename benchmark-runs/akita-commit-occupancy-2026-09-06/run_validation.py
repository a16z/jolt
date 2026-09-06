#!/usr/bin/env python3
"""Serial finalist checks with immutable logs and bounded resource use."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from run_saturation import ROOT, LOCK, matrix, record

AKITA = Path("/private/tmp/akita-kernel-campaign-20260906")
JOLT = Path("/private/tmp/jolt-commit-occupancy-20260906")
AKITA_TARGET = "/Users/mgeorghiades/worktrees/akita-feat-metal/target"
JOLT_TARGET = "/Users/mgeorghiades/worktrees/jolt/lever-c-jolt/target"
PCS_TESTS = ["--test", "akita_fp128_e2e", "--test", "commitment_contract",
             "--test", "protocol_soundness"]
STEPS = {
    "build": (JOLT, JOLT_TARGET, ["cargo", "build", "--release", "-p", "jolt-prover",
        "--example", "modular_benchmark", "--features", "prover-fixtures,metal,profiling"]),
    "jolt-metal": (JOLT, JOLT_TARGET, ["cargo", "nextest", "run", "-p", "jolt-kernels",
        "--features", "metal", "--test-threads", "1", "--cargo-quiet"]),
    "jolt-clippy-host": (JOLT, JOLT_TARGET, ["cargo", "clippy", "--all", "--features",
        "host", "-q", "--all-targets", "--", "-D", "warnings"]),
    "jolt-clippy-zk": (JOLT, JOLT_TARGET, ["cargo", "clippy", "--all", "--features",
        "host,zk", "-q", "--all-targets", "--", "-D", "warnings"]),
    "akita-clippy-parallel": (AKITA, AKITA_TARGET, ["cargo", "clippy", "--all", "--all-targets",
        "--release", "--no-default-features", "--features", "parallel,disk-persistence,transcript-blake2b",
        "--", "-D", "warnings"]),
    "akita-clippy-serial": (AKITA, AKITA_TARGET, ["cargo", "clippy", "--all", "--all-targets",
        "--release", "--no-default-features", "--features", "transcript-blake2b", "--", "-D", "warnings"]),
    "akita-clippy-pcs": (AKITA, AKITA_TARGET, ["cargo", "clippy", "-p", "akita-pcs", "--all-targets",
        "--release", "--no-default-features", "--features",
        "parallel,schedules-default,response-model-diagnostics,transcript-blake2b", "--", "-D", "warnings"]),
}
for mode, features in (("parallel", "parallel,disk-persistence,schedules-default,akita-planner/catalog-gen,transcript-blake2b"),
                       ("serial", "disk-persistence,schedules-default,akita-planner/catalog-gen,transcript-blake2b")):
    STEPS["pcs-" + mode] = (AKITA, AKITA_TARGET, ["cargo", "nextest", "run", "-p", "akita-pcs",
        *PCS_TESTS, "--profile", "ci", "--cargo-profile", "ci-test", "--no-default-features",
        "--features", features, "--test-threads", "1"])
STEPS["build-resolved-cargo"] = STEPS["build"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    timeout = min(1200, args.deadline_epoch - time.time()) if args.deadline_epoch is not None else 1200
    if timeout < 10:
        raise RuntimeError("insufficient validation epoch reserve")
    cwd, target, command = STEPS[args.step]
    log_path = ROOT / "runs" / ("finalist-" + args.step + ".out")
    if log_path.exists():
        raise RuntimeError("immutable validation log exists; inspect it instead of repeating")
    environment = os.environ.copy()
    environment.update(CARGO_NET_OFFLINE="true", CARGO_TARGET_DIR=target,
        CARGO_BUILD_JOBS="8", RUST_MIN_STACK="67108864")
    environment["PATH"] = "/opt/homebrew/opt/rustup/bin:" + environment["PATH"]
    command = ["/opt/homebrew/opt/rustup/bin/cargo", *command[1:]]
    LOCK.mkdir()
    process = None
    started = time.monotonic()
    peak_rss = 0
    try:
        record("validation_start", step=args.step, command=command, cwd=str(cwd),
            timeout_s=timeout, controller_pid=os.getpid())
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                peak_rss = max(peak_rss, matrix.family_rss(process.pid))
                if peak_rss >= 88 * 2**30 or time.monotonic() - started > timeout:
                    raise RuntimeError("validation resource/time guard")
                if re.search(r"GPU.*watchdog.*abort|watchdog.*timed out|SIGABRT", log_path.read_text(), re.I):
                    raise RuntimeError("validation device abort marker")
                time.sleep(2)
        record("validation_complete", step=args.step, exit_code=process.returncode,
            elapsed_s=time.monotonic() - started, sampled_family_rss_bytes=peak_rss,
            raw_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest())
        return process.returncode
    except BaseException as error:
        record("validation_failure", step=args.step, reason=str(error))
        raise
    finally:
        if process is not None:
            matrix.stop_child(process)
        LOCK.rmdir()


if __name__ == "__main__":
    def stop(_signum, _frame):
        raise KeyboardInterrupt("validation controller interrupted")
    signal.signal(signal.SIGTERM, stop)
    raise SystemExit(main())
