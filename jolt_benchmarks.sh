#!/bin/bash

# Usage: ./jolt_benchmarks.sh [MAX_TRACE_LENGTH] [MIN_TRACE_LENGTH] [--benchmarks "bench1 bench2"]
# Defaults: MAX=21, MIN=18

set -euo pipefail

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
MAX_TRACE_LENGTH=${1:-21}
MIN_TRACE_LENGTH=${2:-18}
BENCHMARKS="fibonacci sha2-chain sha3-chain btreemap"

# Parse optional --benchmarks flag
shift 2 2>/dev/null || shift $# 2>/dev/null
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running benchmarks with TRACE_LENGTH=$MIN_TRACE_LENGTH..$MAX_TRACE_LENGTH"
echo "Benchmarks: $BENCHMARKS"

# Set stack size for Rust
export RUST_MIN_STACK=33554432

# Create output directories
mkdir -p benchmark-runs/perfetto_traces
mkdir -p benchmark-runs/results

# Initialize CSV header only if file doesn't exist
if [ ! -f "benchmark-runs/results/timings.csv" ]; then
    echo "benchmark_name,scale,prover_time_s,trace_length,proving_hz" >benchmark-runs/results/timings.csv
fi

# Build once
echo "Building jolt-core (release)..."
cargo build --release -p jolt-core --features monitor
JOLT_BIN="./target/release/jolt-core"

# Run benchmarks
for scale in $(seq $MIN_TRACE_LENGTH $MAX_TRACE_LENGTH); do
    echo "================================================"
    echo "Running benchmarks at scale 2^$scale"
    echo "================================================"
    
    for bench in $BENCHMARKS; do
        echo ">>> $bench at scale 2^$scale"
        $JOLT_BIN benchmark --name "$bench" --scale "$scale" --format chrome
        
        # Postprocess trace file
        python3 scripts/benchmark/postprocess.py "benchmark-runs/perfetto_traces/${bench}_${scale}.json" 2>/dev/null || true
    done
    
    echo ""
done

echo "================================================"
echo "All benchmarks complete!"
echo "Chrome trace files saved in: benchmark-runs/perfetto_traces/"
echo "Timing summary saved in: benchmark-runs/results/timings.csv"
echo "================================================"

# Generate summary table
python3 - <<EOF
import csv

print("\nBenchmark Summary")
print("=================")
print("Scale | Fibonacci | SHA2-chain | SHA3-chain | BTreeMap")
print("------|-----------|------------|------------|----------")

# Read timings
timings = {}
with open('benchmark-runs/results/timings.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        scale = int(row['scale'])
        bench_type = row['benchmark_name']
        time = float(row['prover_time_s'])
        if scale not in timings:
            timings[scale] = {}
        timings[scale][bench_type] = time

# Print table
for scale in sorted(timings.keys()):
    row = timings[scale]
    fib_time = f"{row.get('fibonacci', 0):.2f}s" if 'fibonacci' in row else "N/A"
    sha2_time = f"{row.get('sha2-chain', 0):.2f}s" if 'sha2-chain' in row else "N/A"
    sha3_time = f"{row.get('sha3-chain', 0):.2f}s" if 'sha3-chain' in row else "N/A"
    btree_time = f"{row.get('btreemap', 0):.2f}s" if 'btreemap' in row else "N/A"
    print(f"2^{scale:2} | {fib_time:9} | {sha2_time:11} | {sha3_time:11} | {btree_time:9}")
EOF
