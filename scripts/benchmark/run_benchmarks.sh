#!/usr/bin/env bash
# run_benchmarks.sh — Orchestrate machine prep, optional pinning, and benchmark runs
#
# Usage:
#   ./scripts/benchmark/run_benchmarks.sh [flags]
#
# Flags:
#   --prep | --no-prep              Run full system setup (Linux only, requires sudo). Default: --prep on Linux, --no-prep on macOS
#   --pin  | --no-pin               Run benchmarks under NUMA pinning. Default: --no-pin
#   --nodes NODES                   NUMA nodes to pin to (e.g. 0, 0-3, 0,2). Default: env PIN_NODES or 0
#   --policy bind|interleave        NUMA memory policy. Default: env PIN_POLICY or bind
#   --max-scale N                   Max scale exponent (passed to jolt_runner.sh as first arg)
#   --min-scale M                   Min scale exponent (passed as second arg)
#   --benchmarks "bench1 bench2"    Space-separated list of benchmarks to run. Default: "fibonacci sha2-chain sha3-chain btreemap"
#   --resume                        Resume from previous run (skip completed benchmarks)
#   -h | --help                     Show this help
#
# Examples:
#   # Local macOS (no prep/pin)
#   ./scripts/benchmark/run_benchmarks.sh --max-scale 27 --min-scale 20
#
#   # Linux AWS: default does prep; pinned run on nodes 0-3
#   ./scripts/benchmark/run_benchmarks.sh --pin --nodes 0-3 --policy bind --max-scale 27 --min-scale 20
#
#   # Run only specific benchmarks
#   ./scripts/benchmark/run_benchmarks.sh --benchmarks "fibonacci sha2-chain" --max-scale 25
#
# Notes:
#   - On macOS, prep and pin are skipped automatically.
#   - Per-run prep (page cache drop) runs automatically on Linux for clean benchmarks.
#   - This script forwards scale range and benchmark list to jolt_runner.sh.

set -euo pipefail

# =============================================================================
# DEFAULTS - All configurable parameters are defined here
# =============================================================================

# Scale range defaults (can be overridden by flags or env vars)
DEFAULT_MAX_SCALE=30
DEFAULT_MIN_SCALE=21

# Benchmark list default (matches jolt_runner.sh default)
DEFAULT_BENCHMARKS="fibonacci sha2-chain sha3-chain btreemap"

# NUMA pinning defaults (can be overridden by flags or env vars)  
DEFAULT_PIN_NODES=${PIN_NODES:-0}
DEFAULT_PIN_POLICY=${PIN_POLICY:-bind}

# =============================================================================

show_help() {
  sed -n '2,40p' "$0"
}

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

# Ensure we can find the project root (for validation)
PROJECT_ROOT=$(find_project_root)
if [[ -z "$PROJECT_ROOT" ]]; then
    exit 1
fi

OS_NAME=$(uname -s)
IS_LINUX=0
if [[ "$OS_NAME" == "Linux" ]]; then
  IS_LINUX=1
fi

# Initialize variables with defaults
# Prep by default on Linux; skip on macOS
if (( IS_LINUX )); then
  PREP=1
else
  PREP=0
fi
PIN=0
RESUME=0
PIN_NODES_DEFAULT="$DEFAULT_PIN_NODES"
PIN_POLICY_DEFAULT="$DEFAULT_PIN_POLICY"
MAX_SCALE="$DEFAULT_MAX_SCALE"
MIN_SCALE="$DEFAULT_MIN_SCALE"
BENCHMARKS="$DEFAULT_BENCHMARKS"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prep) PREP=1 ;;
    --no-prep) PREP=0 ;;
    --pin) PIN=1 ;;
    --no-pin) PIN=0 ;;
    --resume) RESUME=1 ;;
    --nodes)
      [[ $# -lt 2 ]] && { echo "--nodes requires a value"; exit 1; }
      PIN_NODES_DEFAULT="$2"; shift ;;
    --policy)
      [[ $# -lt 2 ]] && { echo "--policy requires a value (bind|interleave)"; exit 1; }
      PIN_POLICY_DEFAULT="$2"; shift ;;
    --max-scale)
      [[ $# -lt 2 ]] && { echo "--max-scale requires a value"; exit 1; }
      MAX_SCALE="$2"; shift ;;
    --min-scale)
      [[ $# -lt 2 ]] && { echo "--min-scale requires a value"; exit 1; }
      MIN_SCALE="$2"; shift ;;
    --benchmarks)
      [[ $# -lt 2 ]] && { echo "--benchmarks requires a value"; exit 1; }
      BENCHMARKS="$2"; shift ;;
    -h|--help) show_help; exit 0 ;;
    --) shift; break ;;
    *)
      # Positional numeric convenience: first number -> max, second -> min
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        if [[ -z "$MAX_SCALE" ]]; then
          MAX_SCALE="$1"
        elif [[ -z "$MIN_SCALE" ]]; then
          MIN_SCALE="$1"
        else
          echo "Ignoring extra positional argument: $1"
        fi
      else
        echo "Unknown flag/arg: $1"; echo; show_help; exit 1
      fi
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "OS: $OS_NAME | prep=$PREP pin=$PIN resume=$RESUME | nodes=$PIN_NODES_DEFAULT policy=$PIN_POLICY_DEFAULT"

# 1) Prep machine (Linux only)
if (( PREP )); then
  if (( IS_LINUX )); then
    echo "==> Running machine prep (requires sudo)"
    "$SCRIPT_DIR/tune_linux.sh"
  else
    echo "==> Skipping prep on $OS_NAME (Linux-only)"
  fi
fi

# 4) Run benchmarks, optionally pinned (Linux only)
BENCH_SCRIPT="$SCRIPT_DIR/jolt_runner.sh"
if [[ ! -x "$BENCH_SCRIPT" ]]; then
  # Try to make it executable; ignore failure
  chmod +x "$BENCH_SCRIPT" 2>/dev/null || true
fi

# Build arguments for benchmark script
BENCH_ARGS=("$MAX_SCALE" "$MIN_SCALE")
if (( RESUME )); then
  BENCH_ARGS+=("--resume")
fi
BENCH_ARGS+=("--benchmarks" "$BENCHMARKS")

# 3) Per-run prep: drop page cache for clean benchmarks (Linux only)
if (( IS_LINUX )); then
  echo "==> Running per-run prep (cache drop)"
  "$SCRIPT_DIR/tune_linux.sh" --cache-only
fi

if (( PIN )) && (( IS_LINUX )); then
  echo "==> Running benchmarks with NUMA pinning"
  export PIN_NODES="$PIN_NODES_DEFAULT"
  export PIN_POLICY="$PIN_POLICY_DEFAULT"
  "$SCRIPT_DIR/pin_numa.sh" "$BENCH_SCRIPT" "${BENCH_ARGS[@]}"
else
  if (( PIN )) && (( ! IS_LINUX )); then
    echo "==> Skipping pin on $OS_NAME (Linux-only). Running benchmarks directly."
  else
    echo "==> Running benchmarks without pinning"
  fi
  "$BENCH_SCRIPT" "${BENCH_ARGS[@]}"
fi

echo "All benchmarks completed."


