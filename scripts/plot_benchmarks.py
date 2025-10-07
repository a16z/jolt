#!/usr/bin/env python3
"""Generate interactive plots of Jolt zkVM benchmark results."""

import csv
import os
import sys
import argparse
from collections import defaultdict
import plotly.graph_objects as go

TICK_LABELS = {
    20: "2^20 (1 million)",
    24: "2^24 (16.8 million)",
    26: "2^26 (67 million)",
    27: "2^27 (134 million)",
    28: "2^28 (268 million)"
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load_data(csv_path):
    """Load benchmark data from CSV file."""
    data = defaultdict(list)

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 7 and row[1].isdigit():  # Skip header
                    name, scale, time, _, _, size, size_comp = row[:7]
                    data[name].append(
                        (int(scale), float(time), int(size), int(size_comp)))
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    return dict(data)


def create_speed_plot(data, output_path):
    """Create prover speed plot (Clock speed in KHz vs Trace length)."""
    fig = go.Figure()

    for i, (name, points) in enumerate(data.items()):
        scales, times = zip(*[(s, 2**s / (t * 1000))
                            for s, t, _, _ in sorted(points)])

        fig.add_trace(go.Scatter(
            x=scales, y=times, mode='markers', name=name,
            marker=dict(size=10, color=COLORS[i % len(COLORS)])
        ))

    # Set up x-axis ticks
    ticks = list(range(16, 31))
    labels = [TICK_LABELS.get(n, f"2^{n}") for n in ticks]

    fig.update_layout(
        title="Jolt zkVM Benchmark<br><sub>Hardware: AMD Threadripper PRO 7975WX 32 cores, 768 GB DDR5 RAM</sub>",  # noqa: E501
        xaxis=dict(
            title="Trace length (RISCV64IMAC Cycles)",
            tickmode='array',
            tickvals=ticks,
            ticktext=labels),
        yaxis=dict(
            title="Prover Speed (Cycles proved per millisecond, aka KHz)"),
        width=1200,
        height=800)

    fig.write_html(output_path)
    print(f"Interactive plot saved to {output_path}")


def create_size_plot(data, output_path):
    """Create proof size plot (compressed/uncompressed)."""
    fig = go.Figure()

    for i, (name, points) in enumerate(data.items()):
        color = COLORS[i % len(COLORS)]

        # Extract data
        scales, sizes, sizes_comp = zip(*[(2**s / 1e6, sz / 1024, szc / 1024)
                                          for s, _, sz, szc in sorted(points)])

        # Uncompressed (filled markers)
        fig.add_trace(go.Scatter(
            x=scales, y=sizes, mode='markers', name=f"{name} (uncompressed)",
            marker=dict(size=10, color=color)
        ))

        # Compressed (hollow markers)
        fig.add_trace(go.Scatter(
            x=scales, y=sizes_comp,
            mode='markers', name=f"{name} (compressed)",
            marker=dict(size=10, color='white',
                        line=dict(color=color, width=2))
        ))

    # Set up x-axis ticks
    tick_scales = [18, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    tick_vals = [2**n / 1e6 for n in tick_scales]
    tick_labels = [TICK_LABELS.get(n, f"2^{n}") for n in tick_scales]

    fig.update_layout(
        title="Jolt zkVM Proof Size",
        xaxis=dict(title="Trace length (RISCV64IMAC Cycles)",
                   tickmode='array', tickvals=tick_vals, ticktext=tick_labels),
        yaxis=dict(title="Proof Size (KB)"),
        width=1200, height=800
    )

    fig.write_html(output_path)
    print(f"Proof size plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate benchmark plots from CSV data')
    parser.add_argument('--csv', default='benchmark-runs/results/timings.csv',
                        help='Path to the benchmark CSV file')
    parser.add_argument('--output-dir', default='benchmark-runs',
                        help='Directory to save the output plots')
    parser.add_argument(
        '--plot-type',
        choices=[
            'all',
            'speed',
            'size'],
        default='all',
        help='Type of plot to generate')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.csv)
    if not data:
        print(
            "No data found in CSV file. Run ./scripts/jolt_benchmarks.sh to generate data.")  # noqa: E501
        return

    print(f"Loaded {len(data)} benchmark types from CSV")

    if args.plot_type in ['all', 'speed']:
        create_speed_plot(data, os.path.join(
            args.output_dir, 'benchmark_plot.html'))

    if args.plot_type in ['all', 'size']:
        create_size_plot(data, os.path.join(
            args.output_dir, 'proof_size_plot.html'))


if __name__ == '__main__':
    main()
