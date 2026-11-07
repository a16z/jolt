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
  echo "akita+zk built without the explicit whiteout feature" >&2
  exit 1
fi
if ! grep -q "needs the explicit \`whiteout\` feature" "$unsupported_log"; then
  cat "$unsupported_log" >&2
  echo "akita+zk failed for an unexpected reason." >&2
  exit 1
fi
cargo check -p jolt-verifier --lib --features fs-audit,whiteout --quiet
