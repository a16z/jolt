#!/usr/bin/env bash
# Build and run stdlib-guest for ZeroOS↔Jolt std-mode integration verification.
#
# Why `cargo run -p stdlib` instead of `cargo jolt run` on a single ELF?
#   The stdlib example contains multiple #[jolt::provable] functions, which
#   generates multiple guest artifacts (one ELF per provable fn). The host
#   driver (`stdlib` crate) orchestrates building and proving each function.
#   There is no single stdlib-guest ELF to run directly.
#
# This runs the stdlib host driver which internally uses `cargo jolt build`
# to compile each #[jolt::provable] function and runs them via the prover.

set -euo pipefail

export RUSTUP_NO_UPDATE_CHECK=1

PROFILE="${PROFILE:-release}"
TIMEOUT="${TIMEOUT:-300s}"

# Resolve Jolt repo root based on this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
JOLT_ROOT="$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && git rev-parse --show-toplevel)"

cd "${JOLT_ROOT}"

echo "Building jolt-emu ..." >&2
cargo build -p tracer --bin jolt-emu

# Make `cargo jolt run` deterministic for any guest builds triggered by the host driver.
export JOLT_EMU_PATH="${JOLT_ROOT}/target/debug/jolt-emu"

echo "Building/verifying Jolt stdlib example (ZeroOS std-mode integration) ..."
OUT="$(mktemp)"
trap 'rm -f "${OUT}"' EXIT

# The stdlib host driver internally uses `cargo jolt build -p stdlib-guest --mode std ...`
# to build individual guest functions and runs them through the prover.
# NOTE: This requires `cargo jolt` to be available in PATH.
timeout "${TIMEOUT}" cargo run --profile "${PROFILE}" -p stdlib 2>&1 | tee "${OUT}"

# Check that the run completed successfully (host driver prints test results).
# A successful run returns exit code 0.

echo ""
echo "=== ZeroOS + Jolt stdlib verification PASSED ==="
