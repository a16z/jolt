#!/usr/bin/env bash
set -euo pipefail

fixture="crates/jolt-sumcheck/tests/external-consumer/Cargo.toml"

cargo check -p jolt-transcript --no-default-features --features bn254 --locked --offline

# The transcript is spongefish, which seeds the prover's private randomness
# from getrandom; both belong to the minimal graph.
for forbidden in ark-bn254 ark-ec ark-ff jolt-crypto jolt-openings jolt-r1cs rayon; do
  if cargo tree --manifest-path "$fixture" --edges normal,build --offline --prefix none | rg -q "^${forbidden} "; then
    echo "unexpected dependency in minimal sumcheck graph: ${forbidden}" >&2
    exit 1
  fi
done
