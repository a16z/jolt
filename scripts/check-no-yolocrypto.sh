#!/usr/bin/env bash
set -euo pipefail

# Keeps spongefish's sponge state private (specs/jolt-transcript-spongefish-state.md,
# invariant 2).
#
# spongefish's `yolocrypto` feature makes `ProverState`/`VerifierState`'s
# duplex state public. Cargo unifies features across the graph, so one crate
# enabling it would expose the state to every crate. No crate in the
# workspace may enable it, under any feature combination, on any target.
export CARGO_TERM_COLOR=never

features() {
  cargo tree --locked --workspace --all-features --target all -e features \
    -i spongefish --prefix none --format '{p} {f}' 2>/dev/null
}

graph=$(features)
# Self-test: the inverted tree must reach spongefish at all, or an empty
# result would make the check below vacuous.
if ! grep -Eq '^spongefish ' <<<"$graph"; then
  echo "error: self-test failed: spongefish not found in the workspace graph" >&2
  exit 1
fi
if grep -q 'yolocrypto' <<<"$graph"; then
  echo "error: spongefish's yolocrypto feature is enabled:" >&2
  grep 'yolocrypto' <<<"$graph" >&2
  exit 1
fi
