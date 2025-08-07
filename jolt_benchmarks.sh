#!/bin/bash

# Usage: ./run_benchmarks.sh [TRACE_LENGTH]
# Default TRACE_LENGTH is 22

TRACE_LENGTH=${1:-22}
echo "Running benchmarks with TRACE_LENGTH=$TRACE_LENGTH"

mkdir -p perfetto_traces

echo "type,scale,time" >perfetto_traces/timings.csv

for scale in $(seq 20 $TRACE_LENGTH); do
  echo "================================================"
  echo "Running benchmarks at scale 2^$scale"
  echo "================================================"

  # Fibonacci
  echo ">>> Fibonacci at scale 2^$scale"
  BENCH_TYPE=fib BENCH_SCALE=$scale \
    cargo run --release -p jolt-core -- profile --name master-benchmark --format chrome

  # SHA3-chain
  echo ">>> SHA3-chain at scale 2^$scale"
  BENCH_TYPE=sha3-chain BENCH_SCALE=$scale \
    cargo run --release -p jolt-core -- profile --name master-benchmark --format chrome

  # BTreeMap
  echo ">>> BTreeMap at scale 2^$scale"
  BENCH_TYPE=btreemap BENCH_SCALE=$scale \
    cargo run --release -p jolt-core -- profile --name master-benchmark --format chrome

  echo ""
done

echo "================================================"
echo "All benchmarks complete!"
echo "Chrome trace files saved in: perfetto_traces/"
echo "Timing summary saved in: perfetto_traces/timings.csv"
echo "================================================"

# Generate summary table
python3 - <<EOF
import csv

print("\nBenchmark Summary")
print("=================")
print("Scale | Fibonacci | SHA3     | BTreeMap")
print("------|-----------|----------|----------")

# Read timings
timings = {}
with open('perfetto_traces/timings.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        scale = int(row['scale'])
        bench_type = row['type']
        time = float(row['time'])
        if scale not in timings:
            timings[scale] = {}
        timings[scale][bench_type] = time

# Print table
for scale in sorted(timings.keys()):
    row = timings[scale]
    fib_time = f"{row.get('fib', 0):.2f}s" if 'fib' in row else "N/A"
    sha3_time = f"{row.get('sha3', 0):.2f}s" if 'sha3' in row else "N/A"
    btree_time = f"{row.get('btreemap', 0):.2f}s" if 'btreemap' in row else "N/A"
    print(f"2^{scale:2} | {fib_time:9} | {sha2_time:9} | {sha3_time:9} | {btree_time:9}")
EOF
