#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
run_dir=${1:-"benchmark-runs/akita-recursion-smoke/$(date -u +%Y%m%dT%H%M%SZ)"}
if [[ -e "$run_dir" ]]; then
    printf 'Choose a fresh run directory: %s already exists\n' "$run_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"
run_dir=$(cd "$run_dir" && pwd)
git rev-parse HEAD > "$run_dir/jolt-commit"
cp Cargo.lock "$run_dir/Cargo.lock"

cargo clippy -p jolt-akita-recursion -p jolt-akita-external \
    --all-targets --features recursion-host -- -D warnings
cargo build --locked --profile ci --target-dir target \
    -p jolt-akita-recursion -p jolt-akita-external --features recursion-host --bins
RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-2} target/ci/akita-recursion generate \
    --trace-log 14 --iterations 6 --schedules crates/jolt-akita/schedules \
    --output "$run_dir/pair.bin"
target/ci/outer-verifier-audit --prepare-pair "$run_dir/pair.bin" "$run_dir/material"

export JOLT_AKITA_RECURSION_MATERIAL="$run_dir/material"
export JOLT_AKITA_EXTERNAL_MATERIAL="$run_dir/material"
while read -r guest function heap memory; do
    JOLT_FUNC_NAME="$function" "${JOLT_PATH:-jolt}" build -p "$guest" \
        --mode std --backtrace off --stack-size 33554432 \
        --heap-size "$heap" --memory-size "$memory" -- \
        --release --features guest
done <<'GUESTS'
jolt-akita-recursion-guest verify_before 2147483648 1073741824
jolt-akita-recursion-guest verify_deferred 2147483648 1073741824
jolt-akita-advice-guest verify_advice 1073741824 2147483648
jolt-akita-external-guest verify_external 268435456 2147483648
GUESTS

printf 'Fresh pair replay and guest builds passed: %s\n' "$run_dir"
