#!/usr/bin/env python3
"""One frozen production-PSO saturation diagnostic, with no automatic retries."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
LOCK = ROOT.parent / "akita-10mhz-studies/scratch/machine.lock"
spec = importlib.util.spec_from_file_location(
    "matrix", ROOT.parent / "akita-five-workload-matrix-2026-09-05/run_matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def record(event, **fields):
    row = dict(event=event, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields)
    with (ROOT / "events.jsonl").open("a") as log:
        log.write(json.dumps(row, sort_keys=True) + "\n")
        log.flush()
        os.fsync(log.fileno())
    print(json.dumps(row, sort_keys=True), flush=True)


def main():
    binary = ROOT / "bin/saturation"
    source = Path("/private/tmp/akita-kernel-campaign-20260906/crates/akita-metal/src/kernels/onehot.metal")
    if hashlib.sha256(source.read_bytes()).hexdigest() != "065827662f06ed94de4974f336349abbb933f72316b0590e1e68c3f6db188c83":
        raise RuntimeError("accepted shader fingerprint mismatch")
    output = ROOT / "runs/d0-saturation.out"
    telemetry = ROOT / "runs/d0-telemetry.jsonl"
    archive = ROOT / "runs/d0-production.bin"
    output.parent.mkdir(exist_ok=True)
    if any(path.exists() for path in (output, telemetry, archive)):
        raise RuntimeError("immutable D0 output exists; inspect instead of overwriting")
    command = ["/usr/bin/time", "-l", str(binary), str(source), str(archive)]
    sampler = Path("/private/tmp/akita-commit-macmon-build-20260906/release/macmon")
    if not sampler.is_file():
        raise RuntimeError("telemetry build must finish before the GPU run")
    LOCK.mkdir()
    process = monitor = None
    try:
        record("d0_cooldown", seconds=120, controller_pid=os.getpid(),
               binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        time.sleep(120)
        start = time.monotonic()
        sampled_rss = 0
        with output.open("x") as log, telemetry.open("x") as samples:
            monitor = subprocess.Popen([str(sampler), "pipe", "-i", "100", "-s", "1800"],
                stdout=samples, stderr=subprocess.STDOUT, start_new_session=True)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            while process.poll() is None:
                sampled_rss = max(sampled_rss, matrix.family_rss(process.pid))
                if sampled_rss >= 88 * 2**30 or time.monotonic() - start > 180:
                    raise RuntimeError("D0 resource/time guard")
                if re.search(r"watchdog|abort|FAILURE", output.read_text(), re.I):
                    raise RuntimeError("D0 failure marker")
                if monitor.poll() is not None:
                    raise RuntimeError("telemetry exited before diagnostic completion")
                time.sleep(1)
        raw = output.read_text()
        if (process.returncode != 0 or re.findall(r"(\d+)\s+swaps", raw) != ["0"]
                or raw.count("SATURATION_COMPLETE target_observations=6 parity=pass") != 1
                or len(re.findall(r"^SATURATION positions=", raw, re.M)) != 8):
            raise RuntimeError("D0 process/output failure")
        record("d0_complete", sampled_family_rss_bytes=sampled_rss,
               raw_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
               archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    except BaseException as error:
        record("d0_failure", reason=str(error))
        raise
    finally:
        if process is not None:
            matrix.stop_child(process)
        if monitor is not None:
            matrix.stop_child(monitor)
        LOCK.rmdir()


if __name__ == "__main__":
    def stop(_signum, _frame):
        raise KeyboardInterrupt("controller interrupted")
    signal.signal(signal.SIGTERM, stop)
    main()
