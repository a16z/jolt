#!/usr/bin/env python3
"""Guarded four-workload Metal sweep; see specs/akita-metal-andrew-matrix.md."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ("fibonacci", "sha2-chain", "btreemap", "blake2b-chain")
RSS_LIMIT = 88 * 2**30
COOLDOWN = 120
TIMEOUT = 180
BUDGET = 85 * 60


def digest(path):
    with Path(path).open("rb") as source:
        value = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
        return value.hexdigest()


def command_output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def family_rss(pid):
    rows = [tuple(map(int, row.split())) for row in
            command_output(["ps", "-axo", "pid=,ppid=,rss="]).splitlines()]
    family = {pid}
    while True:
        expanded = family | {child for child, parent, _ in rows if parent in family}
        if expanded == family:
            return sum(rss * 1024 for child, _, rss in rows if child in family)
        family = expanded


def stop_child(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def parse_result(raw, workload, scale):
    if re.findall(r"^PROOF_VERIFIED backend=metal value=true$", raw, re.M) != ["PROOF_VERIFIED backend=metal value=true"]:
        raise ValueError("missing or ambiguous proof verification")
    if re.search(r"watchdog|abort|panicked", raw, re.I):
        raise ValueError("watchdog, abort or panic marker")
    timings = re.findall(r"^MATRIX_TIMING (.+)$", raw, re.M)
    rss = re.findall(r"(\d+)\s+maximum resident set size", raw)
    swaps = re.findall(r"(\d+)\s+swaps", raw)
    if len(timings) != 1 or len(rss) != 1 or swaps != ["0"]:
        raise ValueError("missing/ambiguous timing or resource metrics, or nonzero swaps")
    fields = dict(item.split("=", 1) for item in timings[0].split())
    seconds, trace, padded = float(fields["prove_s"]), int(fields["trace_len"]), int(fields["padded_len"])
    if (fields["name"] != workload or fields["backend"] != "metal"
            or int(fields["scale"]) != scale or padded != 1 << scale
            or not padded // 2 < trace <= padded or not math.isfinite(seconds)
            or seconds <= 0 or int(rss[0]) >= RSS_LIMIT):
        raise ValueError("invalid workload, trace scale, timing or RSS")
    if workload == "blake2b-chain":
        iterations = 5000 << (scale - 24)
        expected = bytes([5]) * 64
        for _ in range(iterations):
            expected = hashlib.blake2b(expected).digest()
        checks = re.findall(r"^BLAKE2B_CHAIN_CHECK iterations=(\d+) digest=([0-9a-f]+) value=true$", raw, re.M)
        if checks != [(str(iterations), expected.hex())]:
            raise ValueError("BLAKE2b hash count or independent output mismatch")
    return dict(workload=workload, scale=scale, prove_s=seconds, trace_len=trace,
                padded_len=padded, padded_mhz=padded / seconds / 1e6,
                actual_mhz=trace / seconds / 1e6, rss_gib=int(rss[0]) / 2**30,
                verified=True, swaps=0)


def mean_rates(results):
    means = []
    for scale in sorted({row["scale"] for row in results}):
        rows = [row for row in results if row["scale"] == scale]
        if len(rows) != 4 or {row["workload"] for row in rows} != set(WORKLOADS):
            continue
        rate = statistics.mean(row["padded_mhz"] for row in rows)
        means.append(dict(scale=scale, measured_mean_mhz=rate,
                          measured_10mhz_pass=rate >= 10,
                          projected_m5_mean_mhz=1.13 * rate))
    return means


class Study:
    def __init__(self, output):
        self.output = output

    def record(self, event, **fields):
        row = dict(event=event, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields)
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row, sort_keys=True), flush=True)

    def checked_process(self, command, log, timeout, environment):
        started, peak = time.monotonic(), 0
        self.record("start", command=command, raw=str(log.relative_to(self.output)), timeout_s=timeout)
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    peak = max(peak, family_rss(process.pid))
                    if peak >= RSS_LIMIT:
                        raise RuntimeError(f"88 GiB family-RSS stop: {peak} bytes")
                    if re.search(r"watchdog|abort|panicked", log.read_text(), re.I):
                        raise RuntimeError("watchdog, abort or panic; stop and investigate")
                    if time.monotonic() - started >= timeout:
                        raise RuntimeError("process timeout")
                    time.sleep(0.5)
                if process.returncode:
                    raise RuntimeError(f"process exit {process.returncode}: {log}")
            finally:
                stop_child(process)
        return peak

    def fingerprint(self, binary):
        displays = json.loads(command_output(["system_profiler", "SPDisplaysDataType", "-json"]))
        gpu = [{key: item.get(key) for key in ("sppci_model", "sppci_cores", "spdisplays_metal")}
               for item in displays["SPDisplaysDataType"]]
        return dict(binary=str(binary), binary_sha256=digest(binary),
                    git_revision=command_output(["git", "rev-parse", "HEAD"]),
                    tracked_diff_sha256=hashlib.sha256(command_output(["git", "diff", "HEAD"]).encode()).hexdigest(),
                    lock_sha256=digest(ROOT / "Cargo.lock"), runner_sha256=digest(__file__),
                    macos=command_output(["sw_vers", "-productVersion"]),
                    chip=command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
                    memory_bytes=int(command_output(["sysctl", "-n", "hw.memsize"])),
                    cpu_count=int(command_output(["sysctl", "-n", "hw.ncpu"])), gpu=gpu,
                    environment={key: os.environ.get(key) for key in
                                 ("RAYON_NUM_THREADS", "JOLT_AKITA_DECOMPOSE_MODE", "JOLT_PATH",
                                  "JOLT_METAL_HANG_WATCHDOG", "JOLT_METAL_HANG_WATCHDOG_SECS",
                                  "AKITA_METAL_ROOT_CENSUS_TILE_STRIDE", "CARGO_TARGET_DIR")})

    def load_results(self):
        results = []
        for path in sorted((self.output / "cells").glob("*.json")):
            row = json.loads(path.read_text())
            raw = self.output / row["raw"]
            if digest(raw) != row["raw_sha256"]:
                raise RuntimeError(f"raw evidence changed: {raw}")
            checked = parse_result(raw.read_text(), row["workload"], row["scale"])
            if any(row[key] != value for key, value in checked.items()):
                raise RuntimeError(f"result does not match raw evidence: {path}")
            results.append(row)
        return results

    def report(self):
        results = self.load_results()
        means = mean_rates(results)
        with (self.output / "results.csv").open("w", newline="") as stream:
            keys = ["scale", "workload", "prove_s", "trace_len", "padded_len", "padded_mhz",
                    "actual_mhz", "rss_gib", "verified", "swaps", "raw", "raw_sha256"]
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(results)
        (self.output / "summary.json").write_text(json.dumps(dict(cells=len(results), means=means), indent=2) + "\n")
        lines = ["# Four-workload Metal matrix", "", f"Verified observations: {len(results)}.", "",
                 "Rates below use padded trace rows, not hashes/second. M5 is an optional projection, not measured.", "",
                 "| Scale | Measured mean MHz | Measured >=10 MHz? | Projected M5 mean MHz (1.13x) |",
                 "|---|---:|---|---:|"]
        lines += [f"| 2^{row['scale']} | {row['measured_mean_mhz']:.4f} | {'yes' if row['measured_10mhz_pass'] else 'no'} | {row['projected_m5_mean_mhz']:.4f} |" for row in means]
        lines += ["", "Only complete four-workload scales receive a mean. One observation per cell; no uncertainty interval.",
                  "Individual times, actual rows and memory are in results.csv; machine/build/guest identity is in manifest.json.",
                  "Inspect events.jsonl and raw logs for failures. No omitted/failed cell is treated as a pass.", ""]
        (self.output / "REPORT.md").write_text("\n".join(lines))
        print("\n".join(lines), flush=True)

    def run(self, args):
        binary = args.binary.resolve()
        scales = list(range(args.min_scale, args.max_scale + 1))
        environment = os.environ.copy()
        environment["RUST_MIN_STACK"] = "67108864"
        for key in ("JOLT_METAL_HANG_WATCHDOG", "JOLT_METAL_HANG_WATCHDOG_SECS",
                    "AKITA_METAL_ROOT_CENSUS_TILE_STRIDE", "JOLT_AKITA_DECOMPOSE_MODE"):
            if key in environment:
                raise RuntimeError(f"unset {key}: this matrix uses the production defaults and watchdog")
        manifest_path = self.output / "manifest.json"
        identity = self.fingerprint(binary)
        if identity["memory_bytes"] < 120 * 2**30 and 28 in scales:
            raise RuntimeError("this T28 contract requires a 128 GiB Mac; do not raise the RSS cap")
        for scale in scales:
            for workload in WORKLOADS:
                lock = ROOT / "benchmark-runs" / f"modular_{workload.replace('-', '_')}_akita_{scale}_metal.lock"
                if lock.exists():
                    raise RuntimeError(f"workload lock exists: {lock}; verify no process is alive before manually archiving a stale lock")
        if args.resume:
            manifest = json.loads(manifest_path.read_text())
            if manifest["identity"] != identity or manifest["scales"] != scales:
                raise RuntimeError("machine/build/source/scale identity changed; start a new study")
        else:
            if manifest_path.exists() or (self.output / "events.jsonl").exists():
                raise RuntimeError("output already used; use --resume or a new output directory")
            guest_hashes = {}
            for workload in WORKLOADS:
                log = self.output / f"prepare-{workload}.out"
                self.checked_process([str(binary), "--name", workload, "--prepare-only"], log, 900, environment)
                matches = re.findall(r"^GUEST_PREPARED name=" + re.escape(workload) + r" path=(.+)$", log.read_text(), re.M)
                if len(matches) != 1:
                    raise RuntimeError("missing or ambiguous AOT guest identity")
                guest_hashes[matches[0]] = digest(matches[0])
            manifest = dict(identity=identity, scales=scales, workloads=WORKLOADS,
                            guests=guest_hashes, cooldown_s=COOLDOWN, timeout_s=TIMEOUT,
                            rss_stop_bytes=RSS_LIMIT, projection_factor=1.13,
                            blake2b_t28_hashes=80000, created_epoch=time.time(),
                            deadline_epoch=time.time() + BUDGET)
            with manifest_path.open("x") as stream:
                json.dump(manifest, stream, indent=2)
        (self.output / "cells").mkdir(exist_ok=True)
        completed = {(row["workload"], row["scale"]) for row in self.load_results()}
        for scale in scales:
            for workload in WORKLOADS:
                if (workload, scale) in completed:
                    continue
                stem = f"t{scale}_{workload}"
                log = self.output / "cells" / f"{stem}.out"
                if log.exists():
                    raise RuntimeError(f"incomplete observation retained: {log}; investigate before a new study")
                if time.time() + COOLDOWN + TIMEOUT > manifest["deadline_epoch"]:
                    raise RuntimeError("study deadline; completed cells retained")
                if self.fingerprint(binary) != identity:
                    raise RuntimeError("machine/build/source identity changed during study")
                for path, expected in manifest["guests"].items():
                    if digest(path) != expected:
                        raise RuntimeError(f"AOT guest changed: {path}")
                self.record("cooldown", workload=workload, scale=scale, seconds=COOLDOWN)
                time.sleep(COOLDOWN)
                command = ["/usr/bin/time", "-l", str(binary), "--name", workload,
                           "--scale", str(scale), "--backend", "metal", "--format", "none"]
                if workload == "blake2b-chain":
                    command += ["--target-trace-size", str(171520000 >> (28 - scale))]
                peak = self.checked_process(command, log, TIMEOUT, environment)
                row = parse_result(log.read_text(), workload, scale)
                row.update(raw=str(log.relative_to(self.output)), raw_sha256=digest(log),
                           sampled_family_rss_bytes=peak)
                if self.fingerprint(binary) != identity:
                    raise RuntimeError("machine/build/source identity changed during proof")
                for path, expected in manifest["guests"].items():
                    if digest(path) != expected:
                        raise RuntimeError(f"guest changed during proof: {path}")
                with (self.output / "cells" / f"{stem}.json").open("x") as stream:
                    json.dump(row, stream, indent=2)
                self.record("result", **row)
                self.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/examples/modular_benchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-scale", type=int, choices=range(24, 29), default=24)
    parser.add_argument("--max-scale", type=int, choices=range(24, 29), default=28)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.min_scale > args.max_scale:
        parser.error("min-scale must not exceed max-scale")
    study = Study(args.output.resolve())
    if args.report_only:
        study.report()
        return
    if sys.platform != "darwin":
        parser.error("measurement requires macOS; report-only works elsewhere")
    study.output.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "benchmark-runs/akita-10mhz-studies/scratch/machine.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with (Path(tempfile.gettempdir()) / "jolt-akita-metal-matrix.lock").open("a") as machine:
        fcntl.flock(machine, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.mkdir()
        try:
            study.run(args)
        except BaseException as error:
            study.record("stopped", reason=str(error))
            study.report()
            raise
        finally:
            lock.rmdir()


if __name__ == "__main__":
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("matrix interrupted; completed observations are retained")

    signal.signal(signal.SIGTERM, interrupted)
    main()
