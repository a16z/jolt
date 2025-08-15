#!/bin/bash

# This updates the fixtures when prover/verifier implementation changes

set -e

echo "🔄 Regenerating test fixtures..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
FIXTURES_DIR="$SCRIPT_DIR/tests/fixtures"

cd "$PROJECT_ROOT"

cargo run --profile build-fast -p fibonacci -- --save

cp /tmp/jolt_verifier_preprocessing.dat "$FIXTURES_DIR/"
cp /tmp/fib_proof.bin "$FIXTURES_DIR/"
cp /tmp/fib_io_device.bin "$FIXTURES_DIR/"

echo "✅ Test fixtures updated successfully!"
