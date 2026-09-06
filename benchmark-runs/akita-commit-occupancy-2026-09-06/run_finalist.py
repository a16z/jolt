#!/usr/bin/env python3
"""Frozen twelve-proof D23 validation; no automatic timing retries."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from run_saturation import ROOT, LOCK, matrix, record

DIRECTORY = ROOT / "finalist"
SEQUENCE = [("fibonacci", 1, "parent"), ("fibonacci", 2, "candidate"),
    ("fibonacci", 3, "candidate"), ("fibonacci", 4, "parent")]
for name in ("btreemap", "sha2-chain", "sha3-chain", "collatz"):
    SEQUENCE += [(name, 1, "parent"), (name, 2, "candidate")]


def check_fingerprints(manifest):
    for filename, expected in manifest["frozen_files"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError("finalist fingerprint mismatch: " + filename)


def main():
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    check_fingerprints(manifest)
    marker = DIRECTORY / "started.json"
    if marker.exists():
        raise RuntimeError("finalist epoch already started; inspect ledger before any resumption")
    matrix.ROOT = DIRECTORY
    matrix.EVENTS = ROOT / "events.jsonl"
    (DIRECTORY / "runs").mkdir(exist_ok=True)
    LOCK.mkdir()
    try:
        started = time.time()
        with marker.open("x") as file:
            json.dump(dict(started_epoch=started, deadline_epoch=started + 2400,
                controller_pid=os.getpid()), file)
        record("finalist_start", deadline_epoch=started + 2400,
            manifest_sha256=hashlib.sha256((DIRECTORY / "manifest.json").read_bytes()).hexdigest())
        results = []
        for workload, attempt, variant in SEQUENCE:
            if time.time() + 300 > started + 2400:
                raise RuntimeError("insufficient finalist epoch reserve for another cooled proof")
            selected = dict(manifest[variant], worktree=manifest["worktree"],
                cargo_target_dir=manifest["cargo_target_dir"])
            record("finalist_cell", workload=workload, attempt=attempt, variant=variant,
                binary_sha256=manifest["frozen_files"][selected["binary"]])
            result = matrix.observe(selected, workload, 28, "metal", attempt, 120, started + 2400)
            result["variant"] = variant
            results.append(result)
            if variant == "parent" and abs(result["prove_s"] / manifest["reference_seconds"][workload] - 1) > 0.05:
                raise RuntimeError("fresh parent discrepancy exceeds five percent: " + workload)
            for previous in results[:-1]:
                if previous["workload"] == workload and previous["trace_len"] != result["trace_len"]:
                    raise RuntimeError("same-workload trace identity changed")
            if workload == "fibonacci" and attempt == 4:
                parents = [r["prove_s"] for r in results if r["variant"] == "parent"]
                candidates = [r["prove_s"] for r in results if r["variant"] == "candidate"]
                if abs(parents[0] - parents[1]) / (sum(parents) / 2) > 0.03:
                    raise RuntimeError("Fibonacci parent drift exceeds three percent")
                if (sum(parents) - sum(candidates)) / 2 < 1:
                    record("finalist_futility_stop", reason="Fibonacci mean saving below one second")
                    return 2
        means = {}
        for workload in ("fibonacci", "btreemap", "sha2-chain", "sha3-chain", "collatz"):
            means[workload] = {}
            for variant in ("parent", "candidate"):
                values = [r["prove_s"] for r in results if r["workload"] == workload and r["variant"] == variant]
                means[workload][variant] = sum(values) / len(values)
        rates = {variant: sum(2**28 / row[variant] / 1e6 for row in means.values()) / 5
            for variant in ("parent", "candidate")}
        passed = rates["candidate"] / rates["parent"] >= 1.03 and all(
            row["candidate"] <= row["parent"] for row in means.values())
        check_fingerprints(manifest)
        summary = dict(means=means, arithmetic_mean_mhz=rates, timing_gate_pass=passed,
            production_accepted=False, remaining="nextest/clippy/fmt and inherited-failure accounting")
        with (DIRECTORY / "result.json").open("x") as file:
            json.dump(summary, file, indent=2)
        record("finalist_complete", **summary)
        return 0 if passed else 2
    except BaseException as error:
        record("finalist_failure", reason=str(error))
        raise
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    def stop(_signum, _frame):
        raise KeyboardInterrupt("finalist controller interrupted")
    signal.signal(signal.SIGTERM, stop)
    raise SystemExit(main() or 0)
