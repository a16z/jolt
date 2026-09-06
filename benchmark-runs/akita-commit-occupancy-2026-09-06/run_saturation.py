#!/usr/bin/env python3
"""One frozen production-PSO saturation diagnostic, with no automatic retries."""
import argparse
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
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--shared-reservation", action="store_true")
    mode.add_argument("--task-pairing", action="store_true")
    mode.add_argument("--task-interleaving", action="store_true")
    mode.add_argument("--cached-gathers", action="store_true")
    args = parser.parse_args()
    real_input = args.task_pairing or args.task_interleaving or args.cached_gathers
    diagnostic = "d7" if args.cached_gathers else "d6" if args.task_interleaving else "d3" if args.task_pairing else "d1" if args.shared_reservation else "d0"
    binary = ROOT / ("bin/cached-gathers" if args.cached_gathers else "bin/task-interleaving" if args.task_interleaving else "bin/task-pairing" if args.task_pairing else "bin/shared-reservation" if args.shared_reservation else "bin/saturation")
    source = Path("/private/tmp/akita-kernel-campaign-20260906/crates/akita-metal/src/kernels/onehot.metal")
    if hashlib.sha256(source.read_bytes()).hexdigest() != "065827662f06ed94de4974f336349abbb933f72316b0590e1e68c3f6db188c83":
        raise RuntimeError("accepted shader fingerprint mismatch")
    output = ROOT / "runs" / (diagnostic + "-saturation.out")
    telemetry = ROOT / "runs" / (diagnostic + "-telemetry.jsonl")
    archive = ROOT / "runs" / (diagnostic + "-production.variant0.bin" if args.shared_reservation or real_input else "d0-production.bin")
    output.parent.mkdir(exist_ok=True)
    if any(path.exists() for path in (output, telemetry, archive)):
        raise RuntimeError("immutable diagnostic output exists; inspect instead of overwriting")
    command = ["/usr/bin/time", "-l", str(binary), str(source)]
    if real_input:
        command += [str(ROOT / "runs/d2-capture"), str(ROOT / "runs/d3-task-map.u32le"),
                    str(ROOT / "runs" / (diagnostic + "-production"))]
        if args.task_interleaving:
            command.append(str(ROOT / "task-interleaving.metal"))
        if args.cached_gathers:
            command += [str(ROOT / "cached-gathers.metal"), "--cached"]
    elif args.shared_reservation:
        command += [str(ROOT / "shared-reservation.metal"), str(ROOT / "runs/d1-production")]
    else:
        command.append(str(archive))
    sampler = Path("/private/tmp/akita-commit-macmon-build-20260906/release/macmon")
    if not sampler.is_file():
        raise RuntimeError("telemetry build must finish before the GPU run")
    LOCK.mkdir()
    process = monitor = None
    try:
        record(diagnostic + "_cooldown", seconds=120, controller_pid=os.getpid(),
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
                    raise RuntimeError("diagnostic resource/time guard")
                if re.search(r"watchdog|abort|FAILURE", output.read_text(), re.I):
                    raise RuntimeError("diagnostic failure marker")
                if monitor.poll() is not None:
                    raise RuntimeError("telemetry exited before diagnostic completion")
                time.sleep(1)
        raw = output.read_text()
        complete = "CACHED_COMPLETE" if args.cached_gathers else "INTERLEAVING_COMPLETE" if args.task_interleaving else "PAIRING_COMPLETE" if args.task_pairing else "RESERVATION_COMPLETE" if args.shared_reservation else "SATURATION_COMPLETE"
        target_observations = 4 if real_input else 6
        expected_rows = 10 if args.cached_gathers else 11 if args.shared_reservation else 8
        if (process.returncode != 0 or re.findall(r"(\d+)\s+swaps", raw) != ["0"]
                or raw.count(complete + f" target_observations={target_observations} parity=pass") != 1
                or len(re.findall(r"^SATURATION positions=", raw, re.M)) != expected_rows):
            raise RuntimeError("diagnostic process/output failure")
        record(diagnostic + "_complete", sampled_family_rss_bytes=sampled_rss,
               raw_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
               archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    except BaseException as error:
        record(diagnostic + "_failure", reason=str(error))
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
