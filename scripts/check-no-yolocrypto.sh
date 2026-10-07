#!/usr/bin/env bash
set -euo pipefail

# Keeps spongefish's sponge state private (specs/jolt-transcript-spongefish-state.md,
# invariant 2).
#
# spongefish's `yolocrypto` feature makes `ProverState`/`VerifierState`'s
# duplex state public. Cargo unifies features across the graph, so one crate
# enabling it would expose the state to every crate. No crate may enable it,
# under any feature combination, on any target, in the root workspace or in
# any standalone workspace (fuzz harnesses, external consumers) whose lockfile
# contains spongefish.
export CARGO_TERM_COLOR=never

root=$(git rev-parse --show-toplevel)
cd "$root"

check() {
  local manifest=$1 graph
  if ! graph=$(cargo tree --locked --manifest-path "$manifest" --workspace --all-features \
    --target all -e features -i spongefish --prefix none --format '{p} {f}'); then
    echo "error: cargo tree failed for $manifest" >&2
    exit 1
  fi
  # Self-test: the inverted tree must reach spongefish, or an empty result
  # would make the check below vacuous.
  if ! grep -Eq '^spongefish ' <<<"$graph"; then
    echo "error: self-test failed: spongefish not found in $manifest's graph" >&2
    exit 1
  fi
  if grep -q 'yolocrypto' <<<"$graph"; then
    echo "error: spongefish's yolocrypto feature is enabled in $manifest:" >&2
    grep 'yolocrypto' <<<"$graph" >&2
    exit 1
  fi
}

check Cargo.toml
while IFS= read -r lock; do
  manifest="$(dirname "$lock")/Cargo.toml"
  [[ "$manifest" == "./Cargo.toml" ]] && continue
  if grep -q '^name = "spongefish"$' "$lock"; then
    check "$manifest"
  fi
done < <(git ls-files -- '*Cargo.lock' | sed 's|^|./|')
