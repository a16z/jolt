#!/usr/bin/env bash
set -euo pipefail

# Shared-field identity check for the post-cutover workspace.
#
# The field crate is `jolt-algebra`; `jolt-field` re-exports it so Akita's
# Git dependency on `jolt-field` (redirected to the workspace by the root
# `[patch]` table) keeps building. The workspace must resolve exactly one
# identity for each of the two packages, and no `akita-field` identity at
# all: that crate no longer exists, so any occurrence means a stale Akita pin
# somewhere in the graph.
#
# Structural check over `cargo metadata` package IDs: immune to `cargo tree`
# rendering (CARGO_TERM_COLOR=always colorizes the `(*)` dedup marker, which
# broke the previous text parse). The worst-case color environment is forced
# below as a permanent regression guard.
export CARGO_TERM_COLOR=always

metadata="$(cargo metadata --format-version 1 --locked)"

for name in jolt-algebra jolt-field; do
  identities="$(jq -r --arg name "$name" '.packages[] | select(.name == $name) | .id' <<<"$metadata" | sort -u)"
  count="$(grep -c . <<<"$identities" || true)"

  if [[ "$count" -ne 1 ]]; then
    echo "error: expected exactly one $name package identity, found $count" >&2
    printf '%s\n' "$identities" >&2
    exit 1
  fi
  printf 'shared field identity: %s\n' "$identities"
done

akita_identities="$(jq -r '.packages[] | select(.name == "akita-field") | .id' <<<"$metadata" | sort -u)"
akita_count="$(grep -c . <<<"$akita_identities" || true)"

if [[ "$akita_count" -gt 0 ]]; then
  echo "error: akita-field no longer exists post-cutover; found $akita_count identity(ies) — stale pre-cutover Akita pin in the graph" >&2
  printf '%s\n' "$akita_identities" >&2
  exit 1
fi
