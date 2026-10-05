#!/usr/bin/env python3
"""Prints the jolt-metal benchmark results as a Markdown table.

Reads criterion's output for the `fp128_*` groups of
`crates/jolt-metal/benches/fp128.rs` and pairs each GPU benchmark with the CPU
benchmark of the same case. Rates are the benchmark's throughput divided by
criterion's median time: operations per second for the chains, elements per
second otherwise.

Usage: scripts/metal-bench-table.py [criterion directory]
"""

import json
import sys
from pathlib import Path


def results(root):
    """Maps (group, case) to {"gpu": rate, "cpu": rate} in units of 10^9/s."""
    table = {}
    for meta_path in sorted(root.glob("fp128_*/**/new/benchmark.json")):
        meta = json.loads(meta_path.read_text())
        estimates = json.loads((meta_path.parent / "estimates.json").read_text())
        parts = meta["full_id"].split("/")
        modes = {
            "gpu": ("gpu", ""),
            "cpu": ("cpu", ""),
            "gpu_partials": ("gpu", "/partial_kernel"),
            "gpu_complete_wall": ("gpu", "/complete_wall"),
            "cpu_complete_wall": ("cpu", "/complete_wall"),
        }
        sides = [i for i, part in enumerate(parts) if part in modes]
        if len(sides) != 1:
            continue
        side = sides[0]
        device, timing = modes[parts[side]]
        group = "/".join(parts[:side]) + timing
        case = "/".join(parts[side + 1 :])
        nanoseconds = estimates["median"]["point_estimate"]
        rate = meta["throughput"]["Elements"] / nanoseconds
        table.setdefault((group, case), {})[device] = rate
    return table


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "target/criterion")
    table = results(root)
    if not table:
        sys.exit(f"error: no fp128 benchmark results under {root}")
    print("Inner products: `partial_kernel` measures GPU partial reduction only; "
          "`complete_wall` includes submission, readback, and the CPU sum of partials, "
          "with inputs already resident. Only complete results are compared with CPU times.")
    print()
    print("| benchmark | case | GPU (G/s) | CPU (G/s) | GPU / CPU |")
    print("|---|---|---:|---:|---:|")
    for (group, case), rates in sorted(table.items()):
        gpu, cpu = rates.get("gpu"), rates.get("cpu")
        ratio = f"{gpu / cpu:.1f}×" if gpu and cpu else ""
        cells = [f"{rate:.2f}" if rate else "" for rate in (gpu, cpu)]
        print(f"| {group} | {case} | {cells[0]} | {cells[1]} | {ratio} |")


if __name__ == "__main__":
    main()
