#!/usr/bin/env python3
"""Plot peak memory usage from the modular prover's summary.json artifacts.

Each `jolt-prover profile --format chrome` run leaves a `summary.json` in
its run directory (read through the `latest_*` links); this script reads
the getrusage peak (`peak_rss_gib`, falling back to the sampled
`root.peak_memory_gib`) and the run identity (`run.workload`,
`run.scale_log2`) from each and plots peak memory vs scale.
"""

import json
import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

try:
    import plotly.graph_objects as go
except ImportError:
    print("Error: plotly is not installed.")
    print("\nTo install, run one of the following:")
    print("  pip install plotly")
    print("  sudo apt-get install python3-plotly")
    sys.exit(1)

TICK_LABELS = {
    20: "2^20 (1 million)",
    24: "2^24 (16.8 million)",
    26: "2^26 (67 million)",
    27: "2^27 (134 million)",
    28: "2^28 (268 million)",
    29: "2^29 (537 million)",
    30: "2^30 (1 billion)"
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

NICE_NAMES = {
    "btreemap": "BTreeMap",
    "fibonacci": "Fibonacci",
    "sha2-chain": "SHA2-chain",
    "sha3-chain": "SHA3-chain",
}


def read_summary_point(summary_path):
    """Read (benchmark_name, scale, peak_memory_gib) from a summary.json.

    Run identity comes from the summary's `run` metadata. Peak memory
    prefers the process-lifetime getrusage high-water mark (`peak_rss_gib`),
    falling back to the sampled prove-window peak (`root.peak_memory_gib`).
    Returns None if identity or peak memory is missing.
    """
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not read {summary_path}: {e}", file=sys.stderr)
        return None

    run = summary.get("run") or {}
    benchmark_name = run.get("workload")
    scale = run.get("scale_log2")
    if benchmark_name is None or scale is None:
        print(f"Warning: no run identity in {summary_path}", file=sys.stderr)
        return None

    peak = summary.get("peak_rss_gib")
    if peak is None:
        peak = (summary.get("root") or {}).get("peak_memory_gib")
    if peak is None:
        print(f"Warning: no peak memory in {summary_path}", file=sys.stderr)
        return None
    return benchmark_name, scale, peak


def load_memory_data(traces_dir):
    """Load peak memory usage data from all summary.json artifacts.

    Returns: dict mapping benchmark_name -> list of (scale, peak_memory_gib)
    """
    data = defaultdict(list)
    traces_dir = Path(traces_dir)

    if not traces_dir.exists():
        print(f"Error: Traces directory not found at {traces_dir}")
        return dict(data)

    # One summary per (workload, scale): read through the latest_* links
    # so superseded runs in timestamped directories are not double-counted.
    summary_files = sorted(traces_dir.glob("latest_*/summary.json"))

    if not summary_files:
        print(f"Warning: No summary files found in {traces_dir}", file=sys.stderr)
        return dict(data)

    for summary_file in summary_files:
        point = read_summary_point(summary_file)
        if point is not None:
            benchmark_name, scale, peak_memory = point
            data[benchmark_name].append((scale, peak_memory))

    return dict(data)


def create_memory_plot(data, output_path):
    """Create peak memory usage plot with logarithmic x-axis."""
    if not data:
        print("Error: No data to plot")
        return

    fig = go.Figure()

    # Collect all unique scales from data
    all_scales = set()
    for i, (name, points) in enumerate(data.items()):
        if not points:
            continue

        # Sort by scale
        points_sorted = sorted(points)
        scales = [s for s, _ in points_sorted]
        memories = [m for _, m in points_sorted]

        all_scales.update(scales)

        nice_name = NICE_NAMES.get(name, name)
        fig.add_trace(go.Scatter(
            x=scales, y=memories, mode='markers', name=nice_name,
            marker=dict(size=10, color=COLORS[i % len(COLORS)])
        ))

    # Set up x-axis ticks dynamically from data
    ticks = sorted(all_scales)
    labels = [TICK_LABELS.get(n, f"2^{n}") for n in ticks]

    fig.update_layout(
        title="Jolt zkVM Peak Memory Usage<br><sub>Hardware: AMD Threadripper PRO 7975WX 32 cores, 768 GB DDR5 RAM</sub>",
        xaxis=dict(
            title="Trace length (RISCV64IMAC Cycles)",
            tickmode='array',
            tickvals=ticks,
            ticktext=labels,
            tickangle=45),
        yaxis=dict(
            title="Peak Memory Usage (GiB)",
            rangemode='tozero'),
        width=1200,
        height=800,
        margin=dict(b=120))

    fig.write_html(output_path)
    print(f"Memory usage plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate memory usage plot from summary.json artifacts')
    parser.add_argument('--traces-dir', default='benchmark-runs',
                        help='Directory containing latest_*/summary.json artifacts')
    parser.add_argument('--output-dir', default='benchmark-runs',
                        help='Directory to save the output plot')
    parser.add_argument('--output-name', default='memory_usage_plot.html',
                        help='Name of the output HTML file')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading summary artifacts from {args.traces_dir}...")
    data = load_memory_data(args.traces_dir)

    if not data:
        print("No memory data found in summary artifacts.")
        print("\nRun `cargo run --release -p jolt-prover --features profiling -- "
              "benchmark` (or `profile`) with --format chrome first.")
        return

    print(f"Loaded memory data for {len(data)} benchmark types")
    for name, points in data.items():
        print(f"  {name}: {len(points)} data points")

    output_path = os.path.join(args.output_dir, args.output_name)
    create_memory_plot(data, output_path)


if __name__ == '__main__':
    main()
