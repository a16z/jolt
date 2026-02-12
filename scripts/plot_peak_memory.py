#!/usr/bin/env python3
"""Plot peak memory usage from /usr/bin/time -l benchmark results."""

import argparse
import csv
import os
import sys
from collections import defaultdict

try:
    import plotly.graph_objects as go
except ImportError:
    print("Error: plotly is not installed.")
    print("\nTo install: pip install plotly")
    sys.exit(1)

TICK_LABELS = {
    18: "2^18",
    24: "2^24",
    25: "2^25",
    26: "2^26",
    27: "2^27",
    28: "2^28",
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

NICE_NAMES = {
    "btreemap": "BTreeMap",
    "fibonacci": "Fibonacci",
    "sha2-chain": "SHA2-chain",
    "sha3-chain": "SHA3-chain",
}


def load_data(csv_path):
    """Load peak memory data from CSV.

    CSV format: benchmark_name,scale,peak_memory_bytes
    """
    data = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3 and row[1].isdigit():
                name = row[0]
                scale = int(row[1])
                peak_bytes = int(row[2])
                data[name].append((scale, peak_bytes))
    return dict(data)


def create_plot(data, output_path):
    fig = go.Figure()

    all_exponents = set()
    for i, (name, points) in enumerate(sorted(data.items())):
        points_sorted = sorted(points)
        scales = [2**s for s, _ in points_sorted]
        memory_gb = [b / (1024**3) for _, b in points_sorted]
        all_exponents.update(s for s, _ in points_sorted)

        nice_name = NICE_NAMES.get(name, name)
        fig.add_trace(
            go.Scatter(
                x=scales,
                y=memory_gb,
                mode="markers",
                name=nice_name,
                marker=dict(size=10, color=COLORS[i % len(COLORS)]),
                line=dict(color=COLORS[i % len(COLORS)]),
            )
        )

    minor_exponents = [19, 20, 21, 22, 23]
    all_tick_exponents = sorted(set(TICK_LABELS.keys()) | set(minor_exponents))
    ticks = [2**e for e in all_tick_exponents]
    labels = [TICK_LABELS.get(e, "") for e in all_tick_exponents]

    fig.update_layout(
        title=dict(text="Jolt Prover Peak Memory Usage", x=0.5),
        xaxis=dict(
            title="Trace length (RV64IMAC Cycles)",
            tickmode="array",
            tickvals=ticks,
            ticktext=labels,
            tickangle=45,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
        ),
        yaxis=dict(
            title="Peak Memory (GB)",
            rangemode="tozero",
            gridcolor="lightgray",
        ),
        plot_bgcolor="white",
        width=1200,
        height=800,
        margin=dict(b=120),
    )

    fig.write_html(output_path)
    print(f"Peak memory plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot peak memory usage from benchmark CSV"
    )
    parser.add_argument(
        "--csv",
        default="benchmark-runs/results/peak_memory.csv",
        help="Path to peak memory CSV file",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark-runs",
        help="Directory to save the output plot",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.csv)
    if not data:
        print("No data found. Run ./scripts/bench_peak_memory.sh first.")
        return

    print(f"Loaded {len(data)} benchmark type(s)")
    for name, points in sorted(data.items()):
        print(f"  {name}: {len(points)} data points")
        for scale, peak_bytes in sorted(points):
            print(f"    2^{scale}: {peak_bytes / (1024**3):.2f} GB")

    output_path = os.path.join(args.output_dir, "peak_memory_plot.html")
    create_plot(data, output_path)


if __name__ == "__main__":
    main()
