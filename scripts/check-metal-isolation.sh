#!/usr/bin/env bash
set -euo pipefail

# Keeps Metal out of the verifier (specs/jolt-metal-field.md, invariant 3).
#
# `jolt-metal` and its `objc2*` bindings must not appear in the normal or
# build dependency graph of `jolt-verifier` or `jolt-field`, under any feature
# combination, on any target. Features are additive, so `--all-features`
# covers every combination, and `--target all` includes macOS-only
# dependencies even when this runs on Linux. Dev-dependencies are excluded:
# verifier tests may build a prover.
export CARGO_TERM_COLOR=never

forbidden='^(jolt-metal|objc2[a-z0-9-]*) '

packages() {
  cargo tree --locked -p "$1" --all-features --target all -e normal,build \
    --prefix none --format '{p}' | sed 's/ (\*)$//' | sort -u
}

# Self-test: the pattern must match the crate it guards against, or a
# rename would make the check below vacuous.
if ! packages jolt-metal | grep -Eq '^objc2-metal '; then
  echo "error: self-test failed: objc2-metal not found in jolt-metal's own graph" >&2
  exit 1
fi

status=0
for package in jolt-verifier jolt-field; do
  if leaked="$(packages "$package" | grep -E "$forbidden")"; then
    echo "error: $package depends on Metal crates:" >&2
    printf '%s\n' "$leaked" >&2
    status=1
  else
    echo "$package: no Metal dependency"
  fi
done
exit "$status"
