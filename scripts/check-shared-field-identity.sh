#!/usr/bin/env bash
set -euo pipefail

# Intermediate-state check for the staged Akita field migration.
#
# After the Solinas stack lands in `jolt-field` (this state), the workspace
# must resolve exactly one `jolt-field` package identity. The pre-cutover
# `akita-field` package is still reachable through the temporary bootstrap
# `akita` feature, but only from Jolt's immutable Akita Git pin, never from a
# local path. The final migration PR replaces this check with one that rejects
# every `akita-field` identity.
#
# Structural check over `cargo metadata` package IDs: immune to `cargo tree`
# rendering (CARGO_TERM_COLOR=always colorizes the `(*)` dedup marker, which
# broke the previous text parse). The worst-case color environment is forced
# below as a permanent regression guard.
export CARGO_TERM_COLOR=always

metadata="$(cargo metadata --format-version 1 --locked)"

jolt_identities="$(jq -r '.packages[] | select(.name == "jolt-field") | .id' <<<"$metadata" | sort -u)"
jolt_count="$(grep -c . <<<"$jolt_identities" || true)"

if [[ "$jolt_count" -ne 1 ]]; then
  echo "error: expected exactly one jolt-field package identity, found $jolt_count" >&2
  printf '%s\n' "$jolt_identities" >&2
  exit 1
fi

akita_identities="$(jq -r '.packages[] | select(.name == "akita-field") | .id' <<<"$metadata" | sort -u)"
akita_count="$(grep -c . <<<"$akita_identities" || true)"

if [[ "$akita_count" -gt 0 ]]; then
  if [[ "$akita_count" -ne 1 ]]; then
    echo "error: expected at most one bootstrap akita-field identity, found $akita_count" >&2
    printf '%s\n' "$akita_identities" >&2
    exit 1
  fi

  if ! grep -q 'github.com/LayerZero-Labs/akita' <<<"$akita_identities"; then
    echo "error: bootstrap akita-field must resolve from the pinned Akita Git source" >&2
    printf '%s\n' "$akita_identities" >&2
    exit 1
  fi

  printf 'bootstrap akita-field identity: %s\n' "$akita_identities"
fi

printf 'shared field identity: %s\n' "$jolt_identities"
