#!/usr/bin/env bash
# Usage: ./jolt_runner.sh [MAX_TRACE_LENGTH] [MIN_TRACE_LENGTH] [--resume] [--benchmarks "bench1 bench2"]
set -euo pipefail

# Initialize defaults
MAX_TRACE_LENGTH=27
MIN_TRACE_LENGTH=21
RESUME_MODE=false
BENCH_LIST="fibonacci sha2-chain sha3-chain btreemap"  # Default list

ulimit -n 1048576 || echo "Failed to increase ulimit -nto 1048576"
ulimit -l unlimited 2>/dev/null || echo "Failed to increase ulimit -l to unlimited"
ulimit -a

# Set Rayon and Rust environment variables for optimal performance
echo "==> Setting Rayon and Rust environment variables"
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-$(nproc)}
export RAYON_THREAD_STACK_SIZE=${RAYON_THREAD_STACK_SIZE:-67108864}  # 64MB per thread
export RUST_MIN_STACK=${RUST_MIN_STACK:-134217728}  # 128MB stack size
export RUST_STACK_GUARD_SIZE=${RUST_STACK_GUARD_SIZE:-0}  # Disable guard pages for high thread count

# Calculate MB values for display (ensure we have numeric values)
RAYON_STACK_MB=$((RAYON_THREAD_STACK_SIZE/1024/1024))
RUST_STACK_MB=$((RUST_MIN_STACK/1024/1024))

echo "Rayon configuration: threads=$RAYON_NUM_THREADS, stack_size=${RAYON_THREAD_STACK_SIZE} bytes (${RAYON_STACK_MB}MB)"
echo "Rust configuration: min_stack=${RUST_MIN_STACK} bytes (${RUST_STACK_MB}MB), guard_size=$RUST_STACK_GUARD_SIZE"

# Parse command line arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_MODE=true
            shift
            ;;
        --benchmarks)
            if [[ $# -lt 2 ]]; then
                echo "Error: --benchmarks requires a value" >&2
                exit 1
            fi
            BENCH_LIST="$2"
            shift 2
            ;;
        --*)
            echo "Error: Unknown option $1" >&2
            exit 1
            ;;
        *)
            # Collect positional arguments
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Handle positional arguments (MAX_TRACE_LENGTH and MIN_TRACE_LENGTH)
if [[ ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
    MAX_TRACE_LENGTH="${POSITIONAL_ARGS[0]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
    MIN_TRACE_LENGTH="${POSITIONAL_ARGS[1]}"
fi

# Mitigate stack overflow
export RUST_MIN_STACK=33554432

# Find the project root by looking for Cargo.toml
find_project_root() {
    local current_dir="$PWD"
    while [[ "$current_dir" != "/" ]]; do
        if [[ -f "$current_dir/Cargo.toml" ]] && [[ -f "$current_dir/rust-toolchain.toml" ]]; then
            echo "$current_dir"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
    done
    echo "Error: Could not find project root (looking for Cargo.toml and rust-toolchain.toml)" >&2
    return 1
}

# Change to project root to ensure all relative paths work correctly
PROJECT_ROOT=$(find_project_root)
if [[ -z "$PROJECT_ROOT" ]]; then
    exit 1
fi
cd "$PROJECT_ROOT"

echo "Running benchmarks with TRACE_LENGTH=$MIN_TRACE_LENGTH..$MAX_TRACE_LENGTH"

# if [ -d "perfetto_traces" ]; then
#     echo "Clearing existing perfetto_traces directory..."
#     rm -rf perfetto_traces
# fi

mkdir -p benchmark-runs/perfetto_traces benchmark-runs/results

# Detect GNU time (gtime on macOS, /usr/bin/time GNU on Linux)
TIME_CMD=()
TIME_WITH_GB_CONVERSION=false
if command -v gtime >/dev/null 2>&1; then
    if gtime --version >/dev/null 2>&1; then
        TIME_CMD=(gtime -f "%E,%M,%P")
        TIME_WITH_GB_CONVERSION=true
    fi
elif [[ -x /usr/bin/time ]] && /usr/bin/time --version 2>&1 | grep -qi 'GNU'; then
    TIME_CMD=(/usr/bin/time -f "%E,%M,%P")
    TIME_WITH_GB_CONVERSION=true
fi

if [ ! -f "benchmark-runs/results/timings.csv" ]; then
    echo "benchmark_name,scale,prover_time_s,trace_length,proving_hz" >benchmark-runs/results/timings.csv
fi

# Build once, then run the binary for all iterations
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "Building jolt-core (release)..."
    cargo build --release -p jolt-core --features pprof,monitor
fi
JOLT_BIN="./target/release/jolt-core"
if [ ! -x "$JOLT_BIN" ]; then
    echo "Error: built binary not found at $JOLT_BIN"
    exit 1
fi

# Clean up existing SRS files (skip if resuming)
if [ "$RESUME_MODE" = false ]; then
    echo "Cleaning up existing .srs files..."
    rm -f *.srs
else
    echo "Resume mode: preserving existing .srs files"
fi

for scale in $(seq $MIN_TRACE_LENGTH $MAX_TRACE_LENGTH); do
    echo ">>> Running benchmarks at scale 2^$scale"
    
    for bench in $BENCH_LIST; do
        # Check if this benchmark is already completed (resume logic)
        result_file="benchmark-runs/results/${bench}_${scale}.csv"
        if [ "$RESUME_MODE" = true ] && [ -f "$result_file" ]; then
            echo "> $bench at scale 2^$scale: SKIPPED (already completed)"
            continue
        fi
        
        echo "> $bench at scale 2^$scale"
        if [ ${#TIME_CMD[@]} -gt 0 ] && [ "$TIME_WITH_GB_CONVERSION" = true ]; then
            # Capture time output to convert KB to GB
            TIME_OUTPUT_FILE=$(mktemp)
            echo "RUST_MIN_STACK=$RUST_MIN_STACK PPROF_PREFIX=${bench}_${scale}_ ${TIME_CMD[*]} -o \"$TIME_OUTPUT_FILE\" $JOLT_BIN benchmark --name \"$bench\" --scale \"$scale\" --format chrome"
            RUST_MIN_STACK=$RUST_MIN_STACK PPROF_PREFIX="${bench}_${scale}_" "${TIME_CMD[@]}" -o "$TIME_OUTPUT_FILE" "$JOLT_BIN" benchmark --name "$bench" --scale "$scale" --format chrome
            
            # Parse and display with GB conversion
            if [[ -f "$TIME_OUTPUT_FILE" ]]; then
                IFS=, read -r elapsed_time max_rss_kb cpu_pct < "$TIME_OUTPUT_FILE"
                max_rss_gb=$(python3 -c "print(f'{float('$max_rss_kb') / 1024 / 1024:.2f}')")
                
                # Convert M:SS.SS to seconds
                if [[ "$elapsed_time" == *:* ]]; then
                    elapsed_seconds=$(echo "$elapsed_time" | awk -F: '{print $1*60 + $2}')
                    echo "Elapsed: ${elapsed_time} (${elapsed_seconds}s) | Max RSS: ${max_rss_gb} GB | CPU: ${cpu_pct}"
                else
                    echo "Elapsed: ${elapsed_time} | Max RSS: ${max_rss_gb} GB | CPU: ${cpu_pct}"
                fi
                rm -f "$TIME_OUTPUT_FILE"
            fi
        elif [ ${#TIME_CMD[@]} -gt 0 ]; then
            echo "RUST_MIN_STACK=$RUST_MIN_STACK PPROF_PREFIX=${bench}_${scale}_ ${TIME_CMD[*]} $JOLT_BIN benchmark --name \"$bench\" --scale \"$scale\" --format chrome"
            RUST_MIN_STACK=$RUST_MIN_STACK PPROF_PREFIX="${bench}_${scale}_" "${TIME_CMD[@]}" "$JOLT_BIN" benchmark --name "$bench" --scale "$scale" --format chrome
        else
            echo "RUST_MIN_STACK=$RUST_MIN_STACK PPROF_PREFIX=${bench}_${scale}_ $JOLT_BIN benchmark --name \"$bench\" --scale \"$scale\" --format chrome"
            RUST_MIN_STACK=$RUST_MIN_STACK PPROF_PREFIX="${bench}_${scale}_" "$JOLT_BIN" benchmark --name "$bench" --scale "$scale" --format chrome
        fi
        
        # Post-process the generated trace file
        expected_trace_file="benchmark-runs/perfetto_traces/${bench}_${scale}.json"
        echo "Expected trace file: $expected_trace_file"
        if [ -f "$expected_trace_file" ]; then
            echo "  -> Post-processing trace: $expected_trace_file"
            python3 scripts/benchmarks/postprocess.py "$expected_trace_file"
        else
            # Try with wildcards as a fallback
            matching_files=(benchmark-runs/perfetto_traces/${bench}*${scale}*.json)
            if [ ${#matching_files[@]} -gt 0 ] && [ -f "${matching_files[0]}" ]; then
                echo "  -> Post-processing trace: ${matching_files[0]}"
                python3 scripts/benchmarks/postprocess.py "${matching_files[0]}"
            else
                echo "  -> Warning: No trace file found for $bench at scale $scale"
            fi
        fi
    done
done





echo "Creating final consolidated results..."
echo "benchmark_name,scale,prover_time_s,trace_length,proving_hz" > benchmark-runs/results/timings.csv
for csv_file in benchmark-runs/results/*_*.csv; do
    [ -f "$csv_file" ] && cat "$csv_file" && echo
done >> benchmark-runs/results/timings.csv
echo "Results consolidated in benchmark-runs/results/timings.csv"
