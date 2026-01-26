#!/usr/bin/env bash
# Build and run fibonacci-guest for ZeroOS↔Jolt integration verification.
# This builds the guest for no-std mode and runs it on jolt-emu.
#
# The guest feature enables jolt/zeroos-arch-riscv via Cargo.toml feature mapping.

set -euo pipefail

export RUSTUP_NO_UPDATE_CHECK=1

PROFILE="${PROFILE:-dev}"
TIMEOUT="${TIMEOUT:-60s}"
TARGET_TRIPLE="riscv64imac-unknown-none-elf"

# Resolve Jolt repo root based on this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
JOLT_ROOT="$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && git rev-parse --show-toplevel)"

cd "${JOLT_ROOT}"

echo "Building jolt-emu ..." >&2
cargo build -p tracer --bin jolt-emu

# Make `cargo jolt run` deterministic.
export JOLT_EMU_PATH="${JOLT_ROOT}/target/debug/jolt-emu"

OUT_DIR="${JOLT_ROOT}/target/${TARGET_TRIPLE}/$([ "$PROFILE" = "dev" ] && echo debug || echo "$PROFILE")"
BIN="${OUT_DIR}/fibonacci-guest"

echo "Building fibonacci-guest (no-std mode) ..."
cargo jolt build -p fibonacci-guest --target "${TARGET_TRIPLE}" \
    -- --quiet --features guest --profile "${PROFILE}"

echo "Running on Jolt emulator ..."
OUT="$(mktemp)"
trap 'rm -f "${OUT}"' EXIT

timeout "${TIMEOUT}" cargo jolt run "${BIN}" | tee "${OUT}"

# Success is determined by exit code 0 (emulator returned successfully).
# Informational: check for emulator success message (non-fatal if missing).
if grep -q "Program exited successfully" "${OUT}" 2>/dev/null; then
    echo "(emulator reported success)"
fi

echo ""
echo "=== fibonacci-guest verification PASSED ==="

