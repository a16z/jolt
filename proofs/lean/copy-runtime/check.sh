#!/usr/bin/env bash
# CI entry point for the copy-runtime Lean proofs.
#
#   ./check.sh            regenerate the extracted data, fail if it differs from
#                         the committed files, build everything (including the
#                         axiom audit in CopyRuntime/Audit.lean), and list the
#                         remaining `sorry`s
#   ./check.sh --no-regen skip the regeneration step (no Rust toolchain needed)
#
# Uses the same environment overrides as regenerate.sh (RUSTC, LLVM_OBJDUMP)
# and LAKE (default: lake).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
LAKE=${LAKE:-lake}
REGEN=true
for argument in "$@"; do
  case "$argument" in
    --no-regen) REGEN=false ;;
    *) echo "usage: $0 [--no-regen]" >&2; exit 2 ;;
  esac
done

if $REGEN; then
  echo "[1/3] Regenerating CopyRuntime/Generated from jolt-platform/src/memory.S"
  "$HERE/regenerate.sh"
  if ! git -C "$HERE" diff --exit-code -- CopyRuntime/Generated; then
    echo "CopyRuntime/Generated is stale: run ./regenerate.sh and commit the result" >&2
    exit 1
  fi
else
  echo "[1/3] Skipping regeneration"
fi

echo "[2/3] Building the proofs and the axiom audit"
(cd "$HERE" && "$LAKE" build)

echo "[3/3] Remaining sorry (each is a theorem whose proof is not finished)"
grep -rn --include='*.lean' -w 'sorry' "$HERE/CopyRuntime" || echo "none"
