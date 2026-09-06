#!/usr/bin/env python3
"""D19 full-cost column-major ABBA, one immutable cooled observation per call."""
import argparse
import hashlib
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from run_saturation import ROOT, LOCK, matrix, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=int, required=True, choices=(1, 2, 3, 4))
    observation = parser.parse_args().observation
    variant = (0, 1, 1, 0)[observation - 1]
    prefix = f"d19-{observation}"
    binary = ROOT / "bin/full-panel"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != "dab662ad72d0275c79ee44000e2598fa1dbb5af75fd293d29f1b950a3115625e":
        raise RuntimeError("frozen full-panel binary fingerprint mismatch")
    shader = Path("/private/tmp/akita-kernel-campaign-20260906/crates/akita-metal/src/kernels/onehot.metal")
    if hashlib.sha256(shader.read_bytes()).hexdigest() != "065827662f06ed94de4974f336349abbb933f72316b0590e1e68c3f6db188c83":
        raise RuntimeError("accepted shader fingerprint mismatch")
    hot = ROOT / "runs/d3-task-map.u32le.hot.u64le"
    if hashlib.sha256(hot.read_bytes()).hexdigest() != "fd7594a14202862d3fc5ffbf8289e080d764327f52e76aef844b89c16de933cd":
        raise RuntimeError("frozen hot count fingerprint mismatch")
    output = ROOT / "runs" / (prefix + "-panel.out")
    telemetry = ROOT / "runs" / (prefix + "-telemetry.jsonl")
    archive_prefix = ROOT / "runs" / (prefix + "-production")
    archive = Path(str(archive_prefix) + ".bin")
    generated = Path(str(archive_prefix) + ".generated.metal")
    reference = ROOT / "runs/d19-parent-final.fp128le"
    if any(path.exists() for path in (output, telemetry, archive, generated)):
        raise RuntimeError("immutable observation exists; inspect rather than repeat")
    if reference.exists() != (observation > 1):
        raise RuntimeError("first-parent reference presence/order mismatch")
    if observation > 1:
        previous = ROOT / "runs" / (f"d19-{observation - 1}-panel.out")
        if previous.read_text().count("FULL_PANEL_COMPLETE target_observations=1 parity=pass") != 1:
            raise RuntimeError("previous observation incomplete")
    sampler = Path("/private/tmp/akita-commit-macmon-build-20260906/release/macmon")
    command = ["/usr/bin/time", "-l", str(binary), str(shader), str(ROOT / "runs/d2-capture"),
        str(reference), str(archive_prefix), str(variant)]
    LOCK.mkdir()
    process = monitor = None
    try:
        record(prefix + "_cooldown", seconds=120, controller_pid=os.getpid(), variant=variant,
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        time.sleep(120)
        started = time.monotonic()
        peak_rss = 0
        with output.open("x") as log, telemetry.open("x") as samples:
            monitor = subprocess.Popen([str(sampler), "pipe", "-i", "100", "-s", "1800"],
                stdout=samples, stderr=subprocess.STDOUT, start_new_session=True)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                peak_rss = max(peak_rss, matrix.family_rss(process.pid))
                if peak_rss >= 88 * 2**30 or time.monotonic() - started > 180:
                    raise RuntimeError("full-panel resource/time guard")
                if re.search(r"watchdog|abort|FAILURE", output.read_text(), re.I):
                    raise RuntimeError("full-panel failure marker")
                if monitor.poll() is not None:
                    raise RuntimeError("telemetry exited before completion")
                time.sleep(1)
        raw = output.read_text()
        rss = re.findall(r"(\d+)\s+maximum resident set size", raw)
        rows = re.findall(r"^FULL_PANEL (.+)$", raw, re.M)
        if (process.returncode != 0 or re.findall(r"(\d+)\s+swaps", raw) != ["0"]
            or len(rss) != 1 or int(rss[0]) >= 88 * 2**30 or len(rows) != 1
            or raw.count("FULL_PANEL_COMPLETE target_observations=1 parity=pass") != 1
            or len(re.findall(r"^SATURATION positions=256", raw, re.M)) != 4
            or len(re.findall(r"^PANEL_COMMAND ", raw, re.M)) != 44):
            raise RuntimeError("full-panel process/output/identity failure")
        metrics = dict(field.split("=", 1) for field in rows[0].split())
        if (metrics["variant"] != str(variant) or metrics["tasks"] != "22301"
            or metrics["hot"] != "3263846381" or metrics["zero_copy"] != "true"):
            raise RuntimeError("full-panel useful-work identity mismatch")
        record(prefix + "_complete", metrics=metrics, sampled_family_rss_bytes=peak_rss,
            raw_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
    except BaseException as error:
        record(prefix + "_failure", reason=str(error))
        raise
    finally:
        if process is not None:
            matrix.stop_child(process)
        if monitor is not None:
            matrix.stop_child(monitor)
        LOCK.rmdir()


if __name__ == "__main__":
    def stop(_signum, _frame):
        raise KeyboardInterrupt("full-panel controller interrupted")
    signal.signal(signal.SIGTERM, stop)
    main()
