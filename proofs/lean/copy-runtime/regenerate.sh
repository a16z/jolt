#!/usr/bin/env bash
# Reassemble jolt-platform/src/memory.S exactly as the guest build does (rustc
# global_asm!, riscv64imac-unknown-none-elf) and regenerate the Lean data in
# CopyRuntime/Generated/. CI runs this and then `git diff --exit-code` on that
# directory, so the proofs always refer to the current assembly.
#
# Environment overrides:
#   RUSTC         rustc of the toolchain pinned by rust-toolchain.toml (default: rustc)
#   LLVM_OBJDUMP  llvm-objdump (default: the one shipped with that toolchain's
#                 llvm-tools component, else llvm-objdump on PATH)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../../.." && pwd)
RUSTC=${RUSTC:-rustc}

if [[ -z "${LLVM_OBJDUMP:-}" ]]; then
  SYSROOT=$(cd "$REPO_ROOT" && "$RUSTC" --print sysroot)
  HOST=$(cd "$REPO_ROOT" && "$RUSTC" -vV | sed -n 's/^host: //p')
  LLVM_OBJDUMP="$SYSROOT/lib/rustlib/$HOST/bin/llvm-objdump"
  [[ -x "$LLVM_OBJDUMP" ]] || LLVM_OBJDUMP=llvm-objdump
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cp "$REPO_ROOT/jolt-platform/src/memory.S" "$WORK/memory.S"
printf '#![no_std]\ncore::arch::global_asm!(include_str!("memory.S"));\n' >"$WORK/asm.rs"
(cd "$REPO_ROOT" && "$RUSTC" --edition 2021 --crate-type lib -C opt-level=3 \
  --target riscv64imac-unknown-none-elf --emit obj \
  -o "$WORK/memory.o" "$WORK/asm.rs")
"$LLVM_OBJDUMP" -d -r --mattr=+c -M no-aliases -M numeric "$WORK/memory.o" >"$WORK/listing"

python3 -I "$HERE/extract.py" "$WORK/memory.o" "$WORK/listing" \
  "$REPO_ROOT/jolt-platform/src/memory.S" "$HERE/CopyRuntime/Generated"
echo "regenerated $HERE/CopyRuntime/Generated"
