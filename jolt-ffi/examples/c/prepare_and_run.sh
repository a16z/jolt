#!/bin/bash
# Helper script to prepare a guest program and run the C FFI example

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <guest-name> [elf-output] [preprocessing-output]"
    echo ""
    echo "Example: $0 fibonacci-guest"
    echo ""
    echo "Available guests:"
    echo "  - fibonacci-guest"
    echo "  - sha2-guest"
    echo "  - sha3-guest"
    echo "  - merkle-tree-guest"
    echo "  - collatz-guest"
    echo "  - btreemap-guest"
    echo "  - (and more in examples/*/guest)"
    exit 1
fi

GUEST=$1
ELF_OUTPUT=${2:-"${GUEST}.elf"}
PREPROCESSING_OUTPUT=${3:-"${GUEST}-preprocessing.bin"}

echo "========================================="
echo "Jolt FFI Example - Full Workflow"
echo "========================================="
echo ""
echo "Guest: $GUEST"
echo "ELF output: $ELF_OUTPUT"
echo "Preprocessing output: $PREPROCESSING_OUTPUT"
echo ""

# Go to jolt root (3 levels up from this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
JOLT_ROOT="$SCRIPT_DIR/../../.."

cd "$JOLT_ROOT"

# Step 1: Prepare guest
echo "Step 1: Preparing guest program..."
cargo run -q -p jolt-ffi --bin prepare-guest -- \
    --guest "$GUEST" \
    --elf-output "jolt-ffi/examples/c/$ELF_OUTPUT" \
    --preprocessing-output "jolt-ffi/examples/c/$PREPROCESSING_OUTPUT"

echo ""

# Step 2: Build C example
echo "Step 2: Building C example..."
cd jolt-ffi/examples/c
make

echo ""

# Step 3: Show how to run
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo ""
echo "Run the example with:"
echo "  ./jolt_example $ELF_OUTPUT proof.bin $PREPROCESSING_OUTPUT"
echo ""
echo "Or run it now:"
read -p "Run example now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running example..."
    echo "========================================="
    ./jolt_example "$ELF_OUTPUT" proof.bin "$PREPROCESSING_OUTPUT"
    echo ""
    echo "✓ Proof generated: proof.bin"
fi
