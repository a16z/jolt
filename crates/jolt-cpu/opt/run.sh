#!/usr/bin/env bash
# jolt-cpu kernel optimization runner.
#
# Usage:
#   ./opt/run.sh                          # bench + print table
#   ./opt/run.sh --save-baseline          # bench + save as baseline
#   ./opt/run.sh --log "hypothesis here"  # bench + diff + append to log
#   ./opt/run.sh --filter "toom"          # only run matching benchmarks
#   ./opt/run.sh --json                   # output JSON to stdout

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPT_DIR="$SCRIPT_DIR"
BASELINE="$OPT_DIR/baselines/baseline.json"
LATEST="$OPT_DIR/latest.json"
LOG="$OPT_DIR/experiment_log.md"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

SAVE_BASELINE=false
LOG_DESC=""
BENCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --save-baseline)
            SAVE_BASELINE=true
            shift
            ;;
        --log)
            LOG_DESC="$2"
            shift 2
            ;;
        *)
            BENCH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Build and run benchmark
echo "Building and running quick_bench..." >&2
cd "$REPO_ROOT"
RESULT=$(cargo bench -p jolt-cpu --bench quick_bench -q -- --json "${BENCH_ARGS[@]}" 2>/dev/null)

# Save latest
echo "$RESULT" > "$LATEST"

# Save baseline
if $SAVE_BASELINE; then
    SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    DATE=$(date +%Y-%m-%d)
    cp "$LATEST" "$BASELINE"
    cp "$LATEST" "$OPT_DIR/baselines/${DATE}_${SHA}.json"
    echo "Baseline saved: $BASELINE" >&2
fi

# Print table (re-run with baseline if available)
if [[ -f "$BASELINE" ]] && ! $SAVE_BASELINE; then
    cargo bench -p jolt-cpu --bench quick_bench -q -- --baseline "$BASELINE" "${BENCH_ARGS[@]}" 2>&1 >/dev/null
else
    cargo bench -p jolt-cpu --bench quick_bench -q -- "${BENCH_ARGS[@]}" 2>&1 >/dev/null
fi

# Diff against baseline
if [[ -f "$BASELINE" ]] && [[ -f "$LATEST" ]]; then
    echo "" >&2
    echo "=== vs baseline ===" >&2
    python3 -c "
import json, sys
try:
    baseline = json.load(open(sys.argv[1]))
    current = json.load(open(sys.argv[2]))
    print(f'{\"Benchmark\":<22} {\"Baseline\":>10} {\"Current\":>10} {\"Delta\":>10}')
    print('-' * 55)
    for name, c in current.get('benchmarks', {}).items():
        b = baseline.get('benchmarks', {}).get(name, {})
        bval = b.get('ns_per_op', 0)
        cval = c.get('ns_per_op', 0)
        if bval > 0:
            delta = (cval - bval) / bval * 100
            print(f'{name:<22} {bval:>10.2f} {cval:>10.2f} {delta:>+9.1f}%')
        else:
            print(f'{name:<22} {\"n/a\":>10} {cval:>10.2f} {\"new\":>10}')
except Exception as e:
    print(f'Diff error: {e}', file=sys.stderr)
" "$BASELINE" "$LATEST" >&2
fi

# Append to experiment log
if [[ -n "$LOG_DESC" ]]; then
    # Count existing experiments
    COUNT=$(grep -c '^## Experiment' "$LOG" 2>/dev/null || echo "0")
    NEXT=$((COUNT + 1))
    NEXT_FMT=$(printf "%03d" "$NEXT")
    SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    DATE=$(date "+%Y-%m-%d %H:%M")

    {
        echo ""
        echo "---"
        echo ""
        echo "## Experiment $NEXT_FMT: $LOG_DESC"
        echo "- **Date**: $DATE"
        echo "- **Commit**: $SHA"
        echo "- **Hypothesis**: $LOG_DESC"
        echo ""
        echo "### Results"
        echo "| Benchmark | Baseline | Current | Delta |"
        echo "|-----------|----------|---------|-------|"

        python3 -c "
import json, sys
try:
    baseline = json.load(open(sys.argv[1])) if sys.argv[1] != 'none' else {'benchmarks': {}}
    current = json.load(open(sys.argv[2]))
    for name, c in current.get('benchmarks', {}).items():
        b = baseline.get('benchmarks', {}).get(name, {})
        bval = b.get('ns_per_op', 0)
        cval = c.get('ns_per_op', 0)
        if bval > 0:
            delta = (cval - bval) / bval * 100
            print(f'| {name} | {bval:.2f} | {cval:.2f} | {delta:+.1f}% |')
        else:
            print(f'| {name} | n/a | {cval:.2f} | new |')
except Exception as e:
    print(f'| error | {e} | | |', file=sys.stderr)
" "${BASELINE:-none}" "$LATEST"

        echo ""
        echo "### Analysis"
        echo "[TODO: Fill in analysis]"
        echo ""
        echo "### Decision"
        echo "- [ ] Keep"
        echo "- [ ] Revert"
        echo "- [ ] Iterate"
    } >> "$LOG"

    echo "Experiment $NEXT_FMT appended to $LOG" >&2
fi
