#!/usr/bin/env bash
set -euo pipefail

cargo nextest run \
  -p jolt-verifier \
  --test fs_obligations \
  --features fs-audit \
  --cargo-quiet

cargo nextest run \
  -p jolt-verifier \
  --test fs_attacks \
  --features fs-audit,prover-fixtures \
  --cargo-quiet

cargo nextest run \
  -p jolt-verifier \
  --test fs_attacks \
  --features fs-audit,prover-fixtures,zk \
  --cargo-quiet

cargo nextest run \
  -p jolt-verifier \
  --test fs_attacks \
  --features akita,fs-audit,prover-fixtures \
  --cargo-quiet

unsupported_log="$(mktemp)"
trap 'rm -f "$unsupported_log"' EXIT
if cargo check \
  -p jolt-verifier \
  --lib \
  --features akita,fs-audit,zk \
  --quiet \
  2>"$unsupported_log"; then
  echo "Akita+ZK now builds; add it to the Fiat-Shamir fixture matrix." >&2
  exit 1
fi
if ! grep -q "features are mutually exclusive" "$unsupported_log"; then
  cat "$unsupported_log" >&2
  echo "Akita+ZK failed for an unexpected reason." >&2
  exit 1
fi
