#!/usr/bin/env bash
set -euo pipefail

# Shared-field identity check for the post-cutover workspace.
#
# After the Akita cutover (Jolt #1684 landed the rebuilt `jolt-field` as the
# single shared field crate; Akita #307 rebound Akita onto it and deleted
# `akita-field`), the workspace must resolve exactly one `jolt-field` package
# identity — the workspace path, unified with Akita's Git pin via the
# `[patch]` table in the root manifest — and no `akita-field` identity at
# all: the crate no longer exists, so any occurrence means a stale pre-cutover
# pin somewhere in the graph.
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
  echo "error: akita-field no longer exists post-cutover; found $akita_count identity(ies) — stale pre-cutover Akita pin in the graph" >&2
  printf '%s\n' "$akita_identities" >&2
  exit 1
fi

printf 'shared field identity: %s\n' "$jolt_identities"
