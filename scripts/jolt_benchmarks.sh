#!/bin/bash

# Usage: ./jolt_benchmarks.sh [MAX_TRACE_LENGTH] [MIN_TRACE_LENGTH] [--benchmarks "bench1 bench2"]
# Defaults: MAX=21, MIN=18

set -euo pipefail

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Default values
MAX_TRACE_LENGTH=${2:-21}
MIN_TRACE_LENGTH=${1:-18}
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
    echo "benchmark_name,scale,prover_time_s,trace_length,proving_hz,proof_size,proof_size_compressed" >benchmark-runs/results/timings.csv
fi

# Build once
echo "Building jolt-core (release)..."
cargo build --release -p jolt-core
JOLT_BIN="./target/release/jolt-core"

# Run benchmarks
for scale in $(seq $MIN_TRACE_LENGTH $MAX_TRACE_LENGTH); do
    echo "================================================"
    echo "Running benchmarks at scale 2^$scale"
    echo "================================================"
    
    for bench in $BENCHMARKS; do
        echo ">>> $bench at scale 2^$scale"
        $JOLT_BIN benchmark --name "$bench" --scale "$scale" --format chrome
    done
    
    echo ""
done

echo "================================================"
echo "All benchmarks complete!"
echo "Chrome trace files saved in: benchmark-runs/perfetto_traces/"
echo "Timing summary saved in: benchmark-runs/results/timings.csv"
echo "================================================"

# Generate summary table
python3 scripts/benchmark_summary.py --csv benchmark-runs/results/timings.csv

# Generate interactive plots
echo ""
echo "Generating interactive plots..."
python3 scripts/plot_benchmarks.py --csv benchmark-runs/results/timings.csv --output-dir benchmark-runs

echo ""
echo "Interactive plots saved:"
echo "  - benchmark-runs/benchmark_plot.html (Prover speed plot)"
echo "  - benchmark-runs/proof_size_plot.html (Proof size plot)"
