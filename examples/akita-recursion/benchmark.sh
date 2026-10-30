#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

run_dir=${1:-"benchmark-runs/akita-recursion/$(date -u +%Y%m%dT%H%M%SZ)"}
if [[ -e "$run_dir/pair.bin" || -e "$run_dir/cycles.json" ]]; then
    printf 'Choose a fresh run directory; benchmark artifacts already exist in %s\n' "$run_dir" >&2
    exit 2
fi

cargo build --locked --profile ci --target-dir target \
    -p jolt-akita-recursion --features recursion-host

mkdir -p "$run_dir"
git rev-parse HEAD > "$run_dir/jolt-commit"
cp Cargo.lock "$run_dir/Cargo.lock"

RUST_LOG=info RAYON_NUM_THREADS=8 target/ci/akita-recursion generate \
    --trace-log 26 --schedules crates/jolt-akita/schedules \
    --output "$run_dir/pair.bin" 2>&1 | tee "$run_dir/generate.log"

RUST_LOG=info RAYON_NUM_THREADS=1 target/ci/akita-recursion trace \
    --pair "$run_dir/pair.bin" --report "$run_dir/cycles.json" \
    2>&1 | tee "$run_dir/trace.log"

printf 'Cycle report: %s/cycles.json\n' "$run_dir"
