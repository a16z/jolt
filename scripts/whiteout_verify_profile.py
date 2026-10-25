#!/usr/bin/env python3
"""The Whiteout verifier benchmark (specs/whiteout-o0-profile.md).

`run` drives the modular profile harness (`jolt-prover profile`) over the
three workloads, clear Akita and Whiteout interleaved, recording the 1-minute
load average before every run, and appends one JSON line per run. `report`
prints the median [min-max] of each verify time and of each span bucket.

Spans are attributed to the harness's verify-mode spans
(`profile_verifier_{parallel,single_threaded,transported}`) by time window,
and keyed by label within their nearest context span on the same thread, so
the pad-layout replay's stages, the response lookup's Σ setup and the
response proof's opening stay apart from the live ones.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

WORKLOADS = [("muldiv", 16), ("fibonacci", 16), ("fibonacci", 20)]
MODES = ["parallel", "single_threaded", "transported"]
# Spans that rename the work nested under them.
CONTEXTS = (
    "whiteout_pad_layout",
    "whiteout_response_lookup",
    "AkitaResponseProver::verify",
)
LINE = re.compile(
    r"Verifier ([0-9.]+)ms parallel \((\d+) threads\), ([0-9.]+)ms single-threaded"
)
TRANSPORTED = re.compile(r"Verifier ([0-9.]+)ms on transported preprocessing")
PROOF = re.compile(r"Proof size (\d+) bytes")


def load_average():
    out = subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout
    return float(re.search(r"load averages?: ([0-9.]+)", out).group(1))


def split(trace_path):
    """Per verify mode: {key: inclusive ms summed over instances}."""
    events = json.loads(Path(trace_path).read_text())
    stacks = defaultdict(list)
    spans = []
    windows = {}
    for event in events:
        ph, tid = event.get("ph"), event.get("tid")
        if ph == "B":
            name = event["name"]
            fields = event.get("args") or {}
            for field in ("level", "block"):
                if field in fields:
                    name = f"{name}[{str(fields[field]).strip(chr(34))}]"
            stacks[tid].append((name, event["ts"]))
        elif ph == "E" and stacks[tid]:
            name, start = stacks[tid].pop()
            if name.startswith("profile_verifier_"):
                windows[name.removeprefix("profile_verifier_")] = (start, event["ts"])
                continue
            context = next(
                (n for n, _ in reversed(stacks[tid]) if n in CONTEXTS), ""
            )
            key = f"{context}/{name}" if context else name
            spans.append((key, start, event["ts"] - start))
    out = {}
    for mode, (lo, hi) in windows.items():
        totals = defaultdict(float)
        for key, start, duration in spans:
            if lo <= start < hi:
                totals[key] += duration / 1e3
        totals["total"] = (hi - lo) / 1e3
        out[mode] = dict(totals)
    return out


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "runs.jsonl"
    binaries = {"akita": args.clear_bin, "whiteout": args.zk_bin}
    for index in range(args.runs):
        # Alternate which build goes first, so neither always runs warm.
        order = ["akita", "whiteout"] if index % 2 == 0 else ["whiteout", "akita"]
        for name, scale in WORKLOADS:
            for build in order:
                load = load_average()
                started = time.time()
                result = subprocess.run(
                    [binaries[build], "profile", "--name", name, "--scale", str(scale),
                     "--format", "chrome", "--backend", "optimized"],
                    capture_output=True, text=True, check=True,
                )
                stdout = result.stdout
                parallel = LINE.search(stdout)
                transported = TRANSPORTED.search(stdout)
                run_dir = Path(
                    f"benchmark-runs/latest_modular_{name}_{build}_{scale}_optimized"
                ).resolve()
                record = {
                    "run": index,
                    "workload": f"{name}_{scale}",
                    "build": build,
                    "load_1m": load,
                    "started": started,
                    "proof_bytes": int(PROOF.search(stdout).group(1)),
                    "verify_ms": {
                        "parallel": float(parallel.group(1)),
                        "single_threaded": float(parallel.group(3)),
                        "transported": float(transported.group(1)) if transported else None,
                    },
                    "threads": int(parallel.group(2)),
                    "spans_ms": split(run_dir / "trace.json"),
                    "run_dir": str(run_dir),
                }
                with log.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                if not args.keep_traces:
                    (run_dir / "trace.json").unlink(missing_ok=True)
                print(f"run {index} {name}_{scale} {build}: load {load}, "
                      f"verify {record['verify_ms']}", flush=True)


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    median = statistics.median(values)
    if len(values) == 1:
        return f"{median:.2f}"
    return f"{median:.2f} [{min(values):.2f}–{max(values):.2f}]"


def report(args):
    records = [json.loads(line) for line in Path(args.log).read_text().splitlines()]
    loads = [r["load_1m"] for r in records]
    print(f"{len(records)} runs, 1-minute load {min(loads):.2f}–{max(loads):.2f}\n")
    groups = defaultdict(list)
    for record in records:
        groups[(record["workload"], record["build"])].append(record)
    for (workload, build), runs in sorted(groups.items()):
        print(f"## {workload} {build} (n = {len(runs)})")
        print(f"proof bytes {stats([r['proof_bytes'] for r in runs])}")
        for mode in MODES:
            print(f"verify {mode} ms: {stats([r['verify_ms'][mode] for r in runs])}")
        modes = [m for m in MODES if any(m in r["spans_ms"] for r in runs)]
        keys = sorted({k for r in runs for m in modes for k in r["spans_ms"].get(m, {})})
        print("| span | " + " | ".join(modes) + " |")
        print("|---|" + "---|" * len(modes))
        for key in keys:
            cells = [stats([r["spans_ms"].get(m, {}).get(key) for r in runs]) for m in modes]
            print(f"| {key} | " + " | ".join(cells) + " |")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--clear-bin", required=True)
    runner.add_argument("--zk-bin", required=True)
    runner.add_argument("--runs", type=int, default=7)
    runner.add_argument("--out", required=True)
    runner.add_argument("--keep-traces", action="store_true")
    reporter = sub.add_parser("report")
    reporter.add_argument("log")
    parsed = parser.parse_args()
    {"run": run, "report": report}[parsed.command](parsed)


if __name__ == "__main__":
    main()
