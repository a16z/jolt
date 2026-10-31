#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
run_dir=${1:-"benchmark-runs/saved-outer/$(date -u +%Y%m%dT%H%M%SZ)"}
if [[ -e "$run_dir" ]]; then
    printf 'Choose a fresh run directory: %s already exists\n' "$run_dir" >&2
    exit 2
fi
cargo build --locked --profile ci --target-dir target -p jolt-akita-external --features recursion-host --bins
fixture=examples/akita-recursion/fixtures/outer-2p30.bundle
expected=examples/akita-recursion/fixtures/outer-2p30.output
RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-32} target/ci/outer-verifier-audit --prepare "$fixture" "$run_dir"
RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-32} target/ci/outer-verifier-audit "$fixture" "$expected" "$run_dir/native-timings.json"
git rev-parse HEAD > "$run_dir/jolt-commit"
cp Cargo.lock "$run_dir/Cargo.lock"
RUST_LOG=info target/ci/outer-verifier-trace "$run_dir" 2>&1 | tee "$run_dir/trace.log"
printf 'Results: %s/cycles.json and %s/native-timings.json\n' "$run_dir" "$run_dir"
