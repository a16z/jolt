#!/usr/bin/env python3
"""Generate a summary table of benchmark results from CSV data."""

import csv
import sys

# Get CSV path from command line or use default
default_csv = 'benchmark-runs/results/timings.csv'
csv_path = (sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '--csv'
            else default_csv)

print("\nBenchmark Summary")
print("=================")
print("Scale | Fibonacci | SHA2-chain | SHA3-chain | BTreeMap")
print("------|-----------|------------|------------|----------")

# Read timings
timings = {}
try:
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scale = int(row['scale'])
            bench_type = row['benchmark_name']
            time = float(row['prover_time_s'])
            if scale not in timings:
                timings[scale] = {}
            timings[scale][bench_type] = time
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}")
    sys.exit(1)

# Print table
for scale in sorted(timings.keys()):
    row = timings[scale]
    fib_time = (f"{row.get('fibonacci', 0):.2f}s"
                if 'fibonacci' in row else "N/A")
    sha2_time = (f"{row.get('sha2-chain', 0):.2f}s"
                 if 'sha2-chain' in row else "N/A")
    sha3_time = (f"{row.get('sha3-chain', 0):.2f}s"
                 if 'sha3-chain' in row else "N/A")
    btree_time = (f"{row.get('btreemap', 0):.2f}s"
                  if 'btreemap' in row else "N/A")
    print(f"2^{scale:2} | {fib_time:9} | {sha2_time:11} | "
          f"{sha3_time:11} | {btree_time:9}")
