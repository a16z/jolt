#!/usr/bin/env bash
# Build and run sig-recovery for ZeroOS+Jolt ECDSA signature recovery verification.
#
# This example demonstrates proving ECDSA signature recovery from Ethereum
# transactions inside the Jolt zkVM. The host generates test transactions,
# serializes them, and proves the signature recovery inside the guest.
#
# NOTE: This requires the sig-recovery example to be present in the workspace.

set -euo pipefail

export RUSTUP_NO_UPDATE_CHECK=1

PROFILE="${PROFILE:-release}"
TIMEOUT="${TIMEOUT:-600s}"

# Resolve Jolt repo root based on this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
JOLT_ROOT="$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && git rev-parse --show-toplevel)"

cd "${JOLT_ROOT}"

echo "Building jolt CLI ..." >&2
cargo build -p jolt --release

echo "Building jolt-emu ..." >&2
cargo build -p tracer --release --bin jolt-emu

# Make `jolt run` deterministic for any guest builds triggered by the host driver.
export JOLT_EMU_PATH="${JOLT_ROOT}/target/release/jolt-emu"
export JOLT_PATH="${JOLT_ROOT}/target/release/jolt"

echo "Building/verifying Jolt sig-recovery example (ZeroOS ECDSA verification) ..."
OUT="$(mktemp)"
trap 'rm -f "${OUT}"' EXIT

timeout "${TIMEOUT}" cargo run --profile "${PROFILE}" -p sig-recovery 2>&1 | tee "${OUT}"

echo ""
echo "=== ZeroOS + Jolt sig-recovery verification PASSED ==="
