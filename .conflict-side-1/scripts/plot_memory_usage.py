#!/usr/bin/env python3
"""Plot peak memory usage from Jolt zkVM benchmark perfetto traces."""

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


def extract_peak_memory_from_trace(trace_path):
    """Extract peak memory usage (in GB) from a perfetto trace file.

    Handles both raw traces (with monitor.rs events) and postprocessed traces
    (with counter events). Returns the maximum value of memory_gb found.
    """
    try:
        with open(trace_path, 'r') as f:
            trace = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not read {trace_path}: {e}", file=sys.stderr)
        return None

    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])

    max_memory = 0.0
    memory_found = False

    for event in events:
        args = event.get("args", {})

        # Check postprocessed format: counter events with memory_gb
        if event.get("ph") == "C" and "memory_gb" in args:
            memory_gb = float(args["memory_gb"])
            max_memory = max(max_memory, memory_gb)
            memory_found = True

        # Check raw format: monitor.rs events with counters.memory_gb
        elif 'monitor.rs' in event.get('name', '') and 'counters.memory_gb' in args:
            memory_gb = float(args["counters.memory_gb"])
            max_memory = max(max_memory, memory_gb)
            memory_found = True

    if not memory_found:
        print(f"Warning: No memory_gb counter found in {trace_path}", file=sys.stderr)
        return None

    return max_memory


def parse_trace_filename(filename):
    """Parse benchmark name and scale from trace filename.

    Expected format: {benchmark}_{scale}.json
    Returns: (benchmark_name, scale) or None if parsing fails
    """
    stem = Path(filename).stem
    parts = stem.rsplit('_', 1)

    if len(parts) == 2:
        benchmark_name, scale_str = parts
        try:
            scale = int(scale_str)
            return benchmark_name, scale
        except ValueError:
            pass

    return None


def load_memory_data(traces_dir):
    """Load peak memory usage data from all perfetto traces.

    Returns: dict mapping benchmark_name -> list of (scale, peak_memory_gb)
    """
    data = defaultdict(list)
    traces_dir = Path(traces_dir)

    if not traces_dir.exists():
        print(f"Error: Traces directory not found at {traces_dir}")
        return dict(data)

    trace_files = sorted(traces_dir.glob("*.json"))

    if not trace_files:
        print(f"Warning: No trace files found in {traces_dir}", file=sys.stderr)
        return dict(data)

    for trace_file in trace_files:
        parsed = parse_trace_filename(trace_file.name)
        if not parsed:
            print(f"Warning: Could not parse filename {trace_file.name}, skipping", file=sys.stderr)
            continue

        benchmark_name, scale = parsed
        peak_memory = extract_peak_memory_from_trace(trace_file)

        if peak_memory is not None:
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
            title="Peak Memory Usage (GB)",
            rangemode='tozero'),
        width=1200,
        height=800,
        margin=dict(b=120))

    fig.write_html(output_path)
    print(f"Memory usage plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate memory usage plot from perfetto traces')
    parser.add_argument('--traces-dir', default='benchmark-runs/perfetto_traces',
                        help='Directory containing perfetto trace JSON files')
    parser.add_argument('--output-dir', default='benchmark-runs',
                        help='Directory to save the output plot')
    parser.add_argument('--output-name', default='memory_usage_plot.html',
                        help='Name of the output HTML file')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading perfetto traces from {args.traces_dir}...")
    data = load_memory_data(args.traces_dir)

    if not data:
        print("No memory data found in perfetto traces.")
        print("\nMake sure to:")
        print("  1. Run benchmarks with --format chrome to generate traces")
        print("  2. Run postprocess_trace.py on the traces to convert counter events")
        return

    print(f"Loaded memory data for {len(data)} benchmark types")
    for name, points in data.items():
        print(f"  {name}: {len(points)} data points")

    output_path = os.path.join(args.output_dir, args.output_name)
    create_memory_plot(data, output_path)


if __name__ == '__main__':
    main()
