#!/usr/bin/env python3
"""Full-cost panel ABBA diagnostics, one immutable cooled observation per call."""
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--widened-carry", action="store_true")
    mode.add_argument("--single-task", action="store_true")
    args = parser.parse_args()
    observation = args.observation
    variant = (0, 1, 1, 0)[observation - 1]
    diagnostic = "d21" if args.single_task else "d20" if args.widened_carry else "d19"
    prefix = f"{diagnostic}-{observation}"
    binary = ROOT / ("bin/full-panel-widened" if args.widened_carry else "bin/full-panel")
    binary_hash = "2f5a3cf9a5312f2994c018b39b25b80d9236b7663363a6cf8bf57fd8a934c11e" if args.widened_carry else "dab662ad72d0275c79ee44000e2598fa1dbb5af75fd293d29f1b950a3115625e"
    if args.single_task:
        binary = ROOT / "bin/full-panel-single"
        binary_hash = "b2d4eb47df07a7327fc160d6b32845ec102bf73094bb72099408d32dc589d7ac"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != binary_hash:
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
    reference = ROOT / "runs" / (diagnostic + "-parent-final.fp128le")
    if any(path.exists() for path in (output, telemetry, archive, generated)):
        raise RuntimeError("immutable observation exists; inspect rather than repeat")
    if reference.exists() != (observation > 1):
        raise RuntimeError("first-parent reference presence/order mismatch")
    if args.single_task and observation > 2 and '"event": "d21_futility_stop"' in (ROOT / "events.jsonl").read_text():
        raise RuntimeError("preregistered D21 futility stop already reached")
    if observation > 1:
        previous = ROOT / "runs" / (f"{diagnostic}-{observation - 1}-panel.out")
        if previous.read_text().count("FULL_PANEL_COMPLETE target_observations=1 parity=pass") != 1:
            raise RuntimeError("previous observation incomplete")
    sampler = Path("/private/tmp/akita-commit-macmon-build-20260906/release/macmon")
    command = ["/usr/bin/time", "-l", str(binary), str(shader), str(ROOT / "runs/d2-capture"),
        str(reference), str(archive_prefix), str(variant)]
    if args.widened_carry:
        command.append("--widened-carry")
    if args.single_task:
        command.append("--single-task")
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
            or len(re.findall(r"^SATURATION positions=256", raw, re.M)) != (6 if args.widened_carry or args.single_task else 4)
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
        if args.single_task and observation == 2:
            parent_raw = (ROOT / "runs/d21-1-panel.out").read_text()
            parent = dict(field.split("=", 1) for field in re.search(r"^FULL_PANEL (.+)$", parent_raw, re.M)[1].split())
            gpu_saving = 1 - float(metrics["panel_gpu_ms"]) / float(parent["panel_gpu_ms"])
            wall_saving_ms = float(parent["wall_ms"]) - float(metrics["wall_ms"])
            if gpu_saving < 0.03 or wall_saving_ms < 300:
                record("d21_futility_stop", gpu_saving=gpu_saving, wall_saving_ms=wall_saving_ms,
                    verdict="discard", remaining_observations="not_launched")
                return 2
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
    raise SystemExit(main() or 0)
