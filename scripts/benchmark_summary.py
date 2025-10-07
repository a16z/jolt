#!/usr/bin/env python3
"""Generate a summary table of benchmark results from CSV data."""

import argparse
import csv
import sys

BENCHMARKS = ['fibonacci', 'sha2-chain', 'sha3-chain', 'btreemap']
BENCHMARK_LABELS = ['Fibonacci', 'SHA2-chain', 'SHA3-chain', 'BTreeMap']

METRICS = {
    'proving_hz': {'label': 'Prover Speed (kHz)', 'format': '{:.1f}', 'suffix': ' kHz'},
    'prover_time_s': {'label': 'Prover Time', 'format': '{:.2f}', 'suffix': 's'},
    'trace_length': {'label': 'Trace Length', 'format': '{:.0f}', 'suffix': ''},
    'proof_size': {'label': 'Proof Size', 'format': '{:.1f}', 'suffix': ' KB'},
    'proof_size_compressed': {'label': 'Compressed Proof', 'format': '{:.1f}', 'suffix': ' KB'},
}


def format_value(value, metric):
    """Format a value according to the metric type."""
    config = METRICS[metric]
    if metric in ['proof_size', 'proof_size_compressed']:
        value = value / 1024  # Convert bytes to KB
    elif metric == 'proving_hz':
        value = value / 1000  # Convert Hz to kHz
    formatted = config['format'].format(value)
    return formatted + config['suffix']


def main():
    parser = argparse.ArgumentParser(
        description='Generate summary table from benchmark CSV')
    parser.add_argument('--csv', default='benchmark-runs/results/timings.csv',
                        help='Path to the benchmark CSV file')
    parser.add_argument('--metric', default='proving_hz',
                        choices=list(METRICS.keys()),
                        help='Metric to display in the table')
    args = parser.parse_args()

    print(f"\nBenchmark Summary ({METRICS[args.metric]['label']})")
    print("=" * 60)

    # Print header
    header = " Scale | " + \
        " | ".join(f"{label:>12}" for label in BENCHMARK_LABELS)
    print(header)
    separator = "-------| " + " | ".join("-" * 12 for _ in BENCHMARKS)
    print(separator)

    # Read data
    data = {}
    try:
        with open(args.csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scale = int(row['scale'])
                bench_type = row['benchmark_name']
                value = float(row[args.metric])
                if scale not in data:
                    data[scale] = {}
                data[scale][bench_type] = value
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Column {e} not found in CSV")
        sys.exit(1)

    # Print table
    for scale in sorted(data.keys()):
        row_data = data[scale]
        values = []
        for bench in BENCHMARKS:
            if bench in row_data:
                formatted = format_value(row_data[bench], args.metric)
                values.append(f"{formatted:>12}")
            else:
                values.append(f"{'':12}")
        scale_str = f"2^{scale}"
        print(f"{scale_str:>6} | " + " | ".join(values))


if __name__ == '__main__':
    main()
