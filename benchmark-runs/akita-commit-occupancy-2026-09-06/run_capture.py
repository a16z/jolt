#!/usr/bin/env python3
"""Capture one verified real input; timing is diagnostic and never promotional."""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time

from run_saturation import LOCK, ROOT, matrix, record


def census(pairing=False):
    capture = ROOT / "runs/d2-capture"
    output = ROOT / ("runs/d3-pairing-price.out" if pairing else "runs/d2-census.out")
    metadata = json.loads((capture / "metadata.json").read_text())
    command = ["/usr/bin/time", "-l", str(ROOT / ("bin/selector-pairing" if pairing else "bin/selector-census")),
        str(capture / "lanes.u8"), str(capture / "active_zero_rows.u64le")]
    command += [str(metadata[key]) for key in
        ("rows", "columns", "positions", "full_blocks", "zero_mask", "hot_entries")]
    if pairing:
        command += [str(metadata["zero_suffix_start"]), str(ROOT / "runs/d3-task-map.u32le")]
    if output.exists():
        raise RuntimeError("immutable census output exists")
    LOCK.mkdir()
    try:
        with output.open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
        raw = output.read_text()
        if raw.count("producer_hot_match=true") != 1 or re.findall(r"(\d+)\s+swaps", raw) != ["0"]:
            raise RuntimeError("census identity/resource failure")
        record("d3_pairing_price_complete" if pairing else "d2_census_complete",
               raw_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    except BaseException as error:
        record("d3_pairing_price_failure" if pairing else "d2_census_failure", reason=str(error))
        raise
    finally:
        LOCK.rmdir()


def main():
    binary = ROOT / "bin/capture-modular-benchmark"
    output = ROOT / "runs/d2-capture.out"
    capture = ROOT / "runs/d2-capture"
    if output.exists() or capture.exists():
        raise RuntimeError("immutable capture exists; inspect rather than overwrite")
    command = ["/usr/bin/time", "-l", str(binary), "--name", "fibonacci",
               "--scale", "28", "--backend", "metal", "--format", "none"]
    environment = dict(os.environ, CARGO_NET_OFFLINE="true", RUST_MIN_STACK="67108864",
        CARGO_TARGET_DIR="/Users/mgeorghiades/worktrees/jolt/lever-c-jolt/target",
        AKITA_COMMIT_CAPTURE_DIR=str(capture))
    LOCK.mkdir()
    process = None
    try:
        record("d2_cooldown", seconds=120, controller_pid=os.getpid(),
               binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        time.sleep(120)
        start = time.monotonic()
        peak_rss = 0
        with output.open("x") as log:
            process = subprocess.Popen(command, cwd="/private/tmp/jolt-commit-occupancy-20260906",
                env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                peak_rss = max(peak_rss, matrix.family_rss(process.pid))
                if peak_rss >= 88 * 2**30 or time.monotonic() - start > 180:
                    raise RuntimeError("capture process memory/time guard")
                if re.search(r"watchdog|abort", output.read_text(), re.I):
                    raise RuntimeError("capture watchdog/abort marker")
                time.sleep(1)
        raw = output.read_text()
        if (process.returncode != 0 or raw.count("PROOF_VERIFIED backend=metal value=true") != 1
                or raw.count("COMMIT_DIAGNOSTIC_CAPTURE complete=true") != 1
                or re.findall(r"(\d+)\s+swaps", raw) != ["0"]):
            raise RuntimeError("capture process/verification/resource failure")
        timing = re.findall(r"^MATRIX_TIMING (.+)$", raw, re.M)
        rss = re.findall(r"(\d+)\s+maximum resident set size", raw)
        if len(timing) != 1 or len(rss) != 1 or int(rss[0]) >= 88 * 2**30:
            raise RuntimeError("capture timing/RSS parse failure")
        fields = dict(item.split("=", 1) for item in timing[0].split())
        if (fields["name"] != "fibonacci" or fields["backend"] != "metal"
                or int(fields["scale"]) != 28 or int(fields["padded_len"]) != 2**28):
            raise RuntimeError("capture proof identity mismatch")
        metadata = json.loads((capture / "metadata.json").read_text())
        if ((capture / "lanes.u8").stat().st_size != metadata["lanes_bytes"]
                or (capture / "active_zero_rows.u64le").stat().st_size != metadata["active_zero_words"] * 8
                or metadata["rows"] != 2**28 or metadata["positions"] != 2**19):
            raise RuntimeError("capture identity/length mismatch")
        record("d2_capture_complete", metadata=metadata, sampled_family_rss_bytes=peak_rss,
               raw_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), promotion_eligible=False)
    except BaseException as error:
        record("d2_capture_failure", reason=str(error))
        raise
    finally:
        if process is not None:
            matrix.stop_child(process)
        LOCK.rmdir()


if __name__ == "__main__":
    def stop(_signum, _frame):
        raise KeyboardInterrupt("capture controller interrupted")
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", action="store_true")
    parser.add_argument("--price-pairing", action="store_true")
    args = parser.parse_args()
    census(args.price_pairing) if args.census or args.price_pairing else main()
