#!/usr/bin/env bash
set -euo pipefail

fixture="crates/jolt-sumcheck/tests/external-consumer/Cargo.toml"

cargo check -p jolt-transcript --no-default-features --features bn254 --locked --offline

for forbidden in ark-bn254 ark-ec ark-ff getrandom jolt-crypto jolt-openings jolt-r1cs rayon spongefish; do
  if cargo tree --manifest-path "$fixture" --edges normal,build --offline --prefix none | rg -q "^${forbidden} "; then
    echo "unexpected dependency in minimal sumcheck graph: ${forbidden}" >&2
    exit 1
  fi
done
