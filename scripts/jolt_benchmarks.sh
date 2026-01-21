#!/bin/bash

# Usage: ./jolt_benchmarks.sh [MIN_TRACE_LENGTH] [MAX_TRACE_LENGTH] [--benchmarks "bench1 bench2"] [--resume] [--monitor]
# Defaults: MAX=21, MIN=18
# --resume: Skip benchmarks that already exist
# --monitor: Enable memory/CPU monitoring (may impact performance)

set -euo pipefail

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Default values
MAX_TRACE_LENGTH=${2:-21}
MIN_TRACE_LENGTH=${1:-18}
BENCHMARKS="fibonacci sha2-chain sha3-chain btreemap"
RESUME=false
MONITOR=false

# Parse optional flags
shift 2 2>/dev/null || shift $# 2>/dev/null
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --monitor)
            MONITOR=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running benchmarks with TRACE_LENGTH=$MIN_TRACE_LENGTH..$MAX_TRACE_LENGTH"
echo "Benchmarks: $BENCHMARKS"
if [ "$RESUME" = true ]; then
    echo "Resume mode: enabled (will skip existing results)"
fi
if [ "$MONITOR" = true ]; then
    echo "Monitoring: enabled (memory/CPU tracking)"
fi

# Set stack size for Rust
export RUST_MIN_STACK=33554432
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C embed-bitcode=yes"
export RUST_LOG="info"

# Create output directories
mkdir -p benchmark-runs/perfetto_traces
mkdir -p benchmark-runs/results

# Initialize CSV header only if file doesn't exist
if [ ! -f "benchmark-runs/results/timings.csv" ]; then
    echo "benchmark_name,scale,prover_time_s,trace_length,proving_hz,proof_size,proof_size_compressed" >benchmark-runs/results/timings.csv
fi

# Build once
if [ "$MONITOR" = true ]; then
    echo "Building jolt-core (release with monitor feature)..."
    cargo build --release -p jolt-core --features monitor
else
    echo "Building jolt-core (release)..."
    cargo build --release -p jolt-core
fi
JOLT_BIN="./target/release/jolt-core"

# Track failures
declare -a FAILED_COMMANDS=()
TOTAL_RUN=0
TOTAL_SKIPPED=0
TOTAL_FAILED=0

# Run benchmarks
for scale in $(seq $MIN_TRACE_LENGTH $MAX_TRACE_LENGTH); do
    echo "=== Running benchmarks at scale 2^$scale ==="
    
    for bench in $BENCHMARKS; do        
        # Check if we should skip this benchmark (resume mode)
        RESULT_FILE="benchmark-runs/results/${bench}_${scale}.csv"
        if [ "$RESUME" = true ] && [ -f "$RESULT_FILE" ]; then
            echo "  ⏭ Skipping (found $RESULT_FILE)"
            ((TOTAL_SKIPPED++)) || true
            continue
        fi
        
        # Run the benchmark (disable exit on error for this command)
        set +e
        $JOLT_BIN benchmark --name "$bench" --scale "$scale" --format chrome
        EXIT_CODE=$?
        set -e
        
        # Check if it failed
        if [ $EXIT_CODE -ne 0 ]; then
            CMD="$JOLT_BIN benchmark --name \"$bench\" --scale \"$scale\" --format chrome"
            echo "  ❌ FAILED (exit code: $EXIT_CODE)"
            echo "  Command: $CMD"
            FAILED_COMMANDS+=("$CMD")
            ((TOTAL_FAILED++)) || true
        else
            ((TOTAL_RUN++)) || true
        fi
    done
    
    echo ""
done

echo "================================================"
echo "Benchmark run summary:"
echo "  ✓ Completed: $TOTAL_RUN"
if [ $TOTAL_SKIPPED -gt 0 ]; then
    echo "  ⏭ Skipped: $TOTAL_SKIPPED"
fi
if [ $TOTAL_FAILED -gt 0 ]; then
    echo "  ❌ Failed: $TOTAL_FAILED"
fi
echo "================================================"

# If there were failures, print them
if [ ${#FAILED_COMMANDS[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  FAILED BENCHMARKS:"
    echo "The following commands failed and can be rerun:"
    echo ""
    for cmd in "${FAILED_COMMANDS[@]}"; do
        echo "  $cmd"
    done
    echo ""
    echo "To resume and skip successful benchmarks, use:"
    echo "  ./scripts/jolt_benchmarks.sh $MIN_TRACE_LENGTH $MAX_TRACE_LENGTH --resume"
    echo ""
fi


# Consolidate results again to ensure files are concluded
# This is helpful when running benchmarks in parallel on a cluster with a shared filesystem.
echo "benchmark_name,scale,prover_time_s,trace_length,proving_hz,proof_size,proof_size_compressed" > benchmark-runs/results/timings.csv

for csv_file in benchmark-runs/results/*_*.csv; do
    [ -f "$csv_file" ] && cat "$csv_file" >> benchmark-runs/results/timings.csv
done

# Postprocess perfetto traces to convert monitor.rs events to counter tracks
if [ "$MONITOR" = true ]; then
    echo ""
    echo "Postprocessing perfetto traces..."
    if [ -d "benchmark-runs/perfetto_traces" ] && [ "$(ls -A benchmark-runs/perfetto_traces/*.json 2>/dev/null)" ]; then
        python3 scripts/postprocess_trace.py benchmark-runs/perfetto_traces/*.json
    else
        echo "No perfetto traces found to postprocess"
    fi
fi


# Generate summary and plots
if [ -f "benchmark-runs/results/timings.csv" ]; then
    echo ""
    python3 scripts/benchmark_summary.py --csv benchmark-runs/results/timings.csv
    echo ""
    python3 scripts/plot_benchmarks.py --csv benchmark-runs/results/timings.csv --output-dir benchmark-runs

    # Only generate memory plot if monitoring was enabled
    if [ "$MONITOR" = true ]; then
        echo ""
        python3 scripts/plot_memory_usage.py --traces-dir benchmark-runs/perfetto_traces --output-dir benchmark-runs
    fi
fi

[ $TOTAL_FAILED -gt 0 ] && exit 1
