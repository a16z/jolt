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

tree="$(cargo tree --workspace --edges normal,build --prefix none)"

jolt_identities="$(
  grep '^jolt-field v' <<<"$tree" \
    | sed 's/ (\*)$//' \
    | sort -u
)"
jolt_count="$(grep -c '^jolt-field v' <<<"$jolt_identities" || true)"

if [[ "$jolt_count" -ne 1 ]]; then
  echo "error: expected exactly one jolt-field package identity, found $jolt_count" >&2
  printf '%s\n' "$jolt_identities" >&2
  exit 1
fi

akita_identities="$(
  { grep '^akita-field v' <<<"$tree" || true; } \
    | sed 's/ (\*)$//' \
    | sort -u
)"

if [[ -n "$akita_identities" ]]; then
  akita_count="$(grep -c '^akita-field v' <<<"$akita_identities" || true)"

  if [[ "$akita_count" -ne 1 ]]; then
    echo "error: expected at most one bootstrap akita-field identity, found $akita_count" >&2
    printf '%s\n' "$akita_identities" >&2
    exit 1
  fi

  if ! grep -q 'https://github.com/LayerZero-Labs/akita' <<<"$akita_identities"; then
    echo "error: bootstrap akita-field must resolve from the pinned Akita Git source" >&2
    printf '%s\n' "$akita_identities" >&2
    exit 1
  fi

  printf 'bootstrap akita-field identity: %s\n' "$akita_identities"
fi

printf 'shared field identity: %s\n' "$jolt_identities"
