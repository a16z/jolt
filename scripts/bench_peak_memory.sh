#!/bin/bash

# Usage: ./scripts/bench_peak_memory.sh [MIN_SCALE] [MAX_SCALE] [--benchmarks "bench1 bench2"] [--resume]
# Defaults: MIN=18, MAX=24, benchmarks="sha2-chain"
#
# Measures peak memory (maximum resident set size) via /usr/bin/time -l (macOS)
# or /usr/bin/time -v (Linux) for the Jolt prover at various trace lengths.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MIN_SCALE=${1:-18}
MAX_SCALE=${2:-24}
BENCHMARKS="sha2-chain"
RESUME=false

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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Peak memory benchmark: scale 2^${MIN_SCALE}..2^${MAX_SCALE}"
echo "Benchmarks: $BENCHMARKS"
if [ "$RESUME" = true ]; then
    echo "Resume mode: enabled"
fi

# Detect OS for /usr/bin/time flags and output parsing
if [[ "$(uname)" == "Darwin" ]]; then
    TIME_CMD="/usr/bin/time -l"
    # macOS: "maximum resident set size" in bytes
    MEMORY_REGEX="maximum resident set size"
    MEMORY_UNIT="bytes"
else
    TIME_CMD="/usr/bin/time -v"
    # Linux: "Maximum resident set size (kbytes)" in kilobytes
    MEMORY_REGEX="Maximum resident set size"
    MEMORY_UNIT="kbytes"
fi

export RUST_MIN_STACK=33554432

mkdir -p benchmark-runs/results

echo "Building jolt-core (release)..."
cargo build --release -p jolt-core --message-format=short -q
JOLT_BIN="./target/release/jolt-core"

TOTAL_RUN=0
TOTAL_SKIPPED=0
TOTAL_FAILED=0
declare -a FAILED_COMMANDS=()

CONSOLIDATED="benchmark-runs/results/peak_memory.csv"

for scale in $(seq "$MIN_SCALE" "$MAX_SCALE"); do
    for bench in $BENCHMARKS; do
        RESULT_FILE="benchmark-runs/results/peak_memory_${bench}_${scale}.csv"
        if [ "$RESUME" = true ] && [ -f "$RESULT_FILE" ]; then
            echo "  Skipping ${bench} 2^${scale} (found $RESULT_FILE)"
            ((TOTAL_SKIPPED++)) || true
            continue
        fi

        echo "Running ${bench} at scale 2^${scale}..."
        TMPFILE=$(mktemp)

        set +e
        $TIME_CMD $JOLT_BIN benchmark --name "$bench" --scale "$scale" 2> "$TMPFILE"
        EXIT_CODE=$?
        set -e

        if [ $EXIT_CODE -ne 0 ]; then
            echo "  FAILED (exit code: $EXIT_CODE)"
            FAILED_COMMANDS+=("$bench --scale $scale")
            ((TOTAL_FAILED++)) || true
            rm -f "$TMPFILE"
            continue
        fi

        # Parse peak memory from /usr/bin/time output
        RAW_MEM=$(grep "$MEMORY_REGEX" "$TMPFILE" | awk '{print $1}' | tr -d ' ')
        rm -f "$TMPFILE"

        if [ -z "$RAW_MEM" ]; then
            echo "  WARNING: Could not parse peak memory"
            ((TOTAL_FAILED++)) || true
            continue
        fi

        # Normalize to bytes
        if [ "$MEMORY_UNIT" = "kbytes" ]; then
            PEAK_BYTES=$((RAW_MEM * 1024))
        else
            PEAK_BYTES=$RAW_MEM
        fi

        PEAK_GB=$(echo "scale=2; $PEAK_BYTES / 1073741824" | bc)
        echo "  Peak memory: ${PEAK_GB} GB"

        echo "${bench},${scale},${PEAK_BYTES}" > "$RESULT_FILE"
        ((TOTAL_RUN++)) || true
    done
done

# Consolidate results
> "$CONSOLIDATED"
for csv_file in benchmark-runs/results/peak_memory_*_*.csv; do
    [ -f "$csv_file" ] && cat "$csv_file" >> "$CONSOLIDATED"
done

echo ""
echo "========================================"
echo "Peak memory benchmark summary:"
echo "  Completed: $TOTAL_RUN"
[ $TOTAL_SKIPPED -gt 0 ] && echo "  Skipped: $TOTAL_SKIPPED"
[ $TOTAL_FAILED -gt 0 ] && echo "  Failed: $TOTAL_FAILED"
echo "  Results: $CONSOLIDATED"
echo "========================================"

if [ ${#FAILED_COMMANDS[@]} -gt 0 ]; then
    echo ""
    echo "FAILED BENCHMARKS:"
    for cmd in "${FAILED_COMMANDS[@]}"; do
        echo "  $cmd"
    done
fi

# Generate plot
if [ -f "$CONSOLIDATED" ] && [ -s "$CONSOLIDATED" ]; then
    echo ""
    python3 scripts/plot_peak_memory.py --csv "$CONSOLIDATED" --output-dir benchmark-runs
fi

[ $TOTAL_FAILED -gt 0 ] && exit 1
exit 0
