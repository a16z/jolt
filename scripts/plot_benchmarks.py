#!/usr/bin/env python3
"""
Generate interactive plots of Jolt zkVM benchmark results.
"""

import csv
import argparse
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.offline as pyo


def load_benchmark_data(csv_path):
    """Load benchmark data from CSV file."""
    benchmark_data = {}
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return benchmark_data
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 5:
                try:
                    bench_name = row[0]
                    scale = int(row[1])
                    time = float(row[2])
                    proof_size = int(row[5]) if len(row) > 5 else 0
                    proof_size_comp = int(row[6]) if len(row) > 6 else 0
                    
                    if bench_name not in benchmark_data:
                        benchmark_data[bench_name] = []
                    
                    benchmark_data[bench_name].append((scale, time, proof_size, proof_size_comp))
                except (ValueError, IndexError):
                    continue  # Skip header or malformed rows
    
    return benchmark_data


def create_benchmark_plot(data, output_path):
    """Create prover speed plot (Clock speed in KHz vs Trace length)."""
    fig = go.Figure()
    
    # Define colors for different benchmarks
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for color_idx, (bench_name, points) in enumerate(data.items()):
        # Sort points by scale
        sorted_points = sorted(points, key=lambda x: x[0])
        
        color = colors[color_idx % len(colors)]
        
        x_values = []
        y_values = []
        
        for scale, time, _, _ in sorted_points:
            # Calculate cycles and clock speed
            cycles = float(1 << scale)
            # Clock speed in KHz (cycles per millisecond)
            clock_speed = cycles / (time * 1000.0)
            
            x_values.append(scale)
            y_values.append(clock_speed)
        
        # Add trace with markers only
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            name=bench_name,
            marker=dict(size=10, color=color)
        ))
    
    # Create custom tick labels
    tick_vals = list(range(16, 31))
    tick_text = []
    
    for n in tick_vals:
        if n == 20:
            label = "2^20 (1 million)"
        elif n == 24:
            label = "2^24 (16.8 million)"
        elif n == 26:
            label = "2^26 (67 million)"
        elif n == 27:
            label = "2^27 (134 million)"
        elif n == 28:
            label = "2^28 (268 million)"
        else:
            label = f"2^{n}"
        tick_text.append(label)
    
    # Update layout
    fig.update_layout(
        title="Jolt zkVM Benchmark<br><sub>Hardware: AMD Threadripper PRO 7975WX 32 cores, 768 GB DDR5 RAM</sub>",
        xaxis=dict(
            title="Trace length (RISCV64IMAC Cycles)",
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text
        ),
        yaxis=dict(
            title="Prover Speed (Cycles proved per millisecond, aka KHz)"
        ),
        width=1200,
        height=800
    )
    
    # Save to HTML
    fig.write_html(output_path)
    print(f"Interactive plot saved to {output_path}")


def create_proof_size_plot(data, output_path):
    """Create proof size plot (compressed/uncompressed)."""
    fig = go.Figure()
    
    # Define colors for different benchmarks
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for color_idx, (bench_name, points) in enumerate(data.items()):
        # Sort points by scale
        sorted_points = sorted(points, key=lambda x: x[0])
        
        color = colors[color_idx % len(colors)]
        
        x_values = []
        y_values = []
        x_values_comp = []
        y_values_comp = []
        
        for scale, _, proof_size, proof_size_comp in sorted_points:
            # Convert 2^scale to millions of cycles for x-axis
            cycles_millions = float(1 << scale) / 1_000_000.0
            
            # Convert proof sizes from bytes to KB
            proof_size_kb = proof_size / 1024.0
            proof_size_comp_kb = proof_size_comp / 1024.0
            
            x_values.append(cycles_millions)
            y_values.append(proof_size_kb)
            
            x_values_comp.append(cycles_millions)
            y_values_comp.append(proof_size_comp_kb)
        
        # Add uncompressed proof size with filled markers
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            name=f"{bench_name} (uncompressed)",
            marker=dict(size=10, color=color)
        ))
        
        # Add compressed proof size with hollow markers
        fig.add_trace(go.Scatter(
            x=x_values_comp,
            y=y_values_comp,
            mode='markers',
            name=f"{bench_name} (compressed)",
            marker=dict(
                size=10,
                color='white',
                line=dict(color=color, width=2)
            )
        ))
    
    # Create custom tick labels
    tick_scales = [18, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    tick_vals = []
    tick_text = []
    
    for n in tick_scales:
        cycles_millions = float(1 << n) / 1_000_000.0
        tick_vals.append(cycles_millions)
        
        if n == 20:
            label = "2^20 (1 million)"
        elif n == 24:
            label = "2^24 (16.8 million)"
        elif n == 26:
            label = "2^26 (67 million)"
        elif n == 27:
            label = "2^27 (134 million)"
        elif n == 28:
            label = "2^28 (268 million)"
        else:
            label = f"2^{n}"
        tick_text.append(label)
    
    # Update layout
    fig.update_layout(
        title="Jolt zkVM Proof Size",
        xaxis=dict(
            title="Trace length (RISCV64IMAC Cycles)",
            type="linear",
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text
        ),
        yaxis=dict(
            title="Proof Size (KB)"
        ),
        width=1200,
        height=800
    )
    
    # Save to HTML
    fig.write_html(output_path)
    print(f"Proof size plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark plots from CSV data')
    parser.add_argument(
        '--csv',
        default='benchmark-runs/results/timings.csv',
        help='Path to the benchmark CSV file (default: benchmark-runs/timings.csv)'
    )
    parser.add_argument(
        '--output-dir',
        default='benchmark-runs',
        help='Directory to save the output plots (default: benchmark-runs)'
    )
    parser.add_argument(
        '--plot-type',
        choices=['all', 'speed', 'size'],
        default='all',
        help='Type of plot to generate (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load benchmark data
    data = load_benchmark_data(args.csv)
    
    if not data:
        print("No data found in CSV file. Run benchmarks first to generate data.")
        return
    
    print(f"Loaded {len(data)} benchmark types from CSV")
    
    # Generate plots based on plot type
    if args.plot_type in ['all', 'speed']:
        speed_plot_path = os.path.join(args.output_dir, 'benchmark_plot.html')
        create_benchmark_plot(data, speed_plot_path)
    
    if args.plot_type in ['all', 'size']:
        size_plot_path = os.path.join(args.output_dir, 'proof_size_plot.html')
        create_proof_size_plot(data, size_plot_path)


if __name__ == '__main__':
    main()
