#!/bin/bash
set -e

# Script to build jolt-ffi as a static library for iOS
# Supports: iPhone devices (arm64), Apple Silicon simulator (arm64), Intel simulator (x86_64)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FFI_DIR="$PROJECT_ROOT/jolt-ffi"
OUTPUT_DIR="$SCRIPT_DIR/JoltProver/libs"
DEMOS_DIR="$SCRIPT_DIR/JoltProver/demos"

echo "Building jolt-ffi for iOS..."
echo "Project root: $PROJECT_ROOT"
echo "FFI directory: $FFI_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Demos directory: $DEMOS_DIR"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DEMOS_DIR"

# iOS device (arm64)
echo "Building for iOS device (aarch64-apple-ios)..."
cd "$FFI_DIR"
cargo build --release --target aarch64-apple-ios --message-format=short -q

# iOS simulator (Apple Silicon)
echo "Building for iOS simulator - Apple Silicon (aarch64-apple-ios-sim)..."
cargo build --release --target aarch64-apple-ios-sim --message-format=short -q

# # iOS simulator (Intel)
# echo "Building for iOS simulator - Intel (x86_64-apple-ios)..."
# cargo build --release --target x86_64-apple-ios --message-format=short -q

# # Create universal library for simulator (arm64 + x86_64)
# echo "Creating universal simulator library..."
# lipo -create \
#     "$PROJECT_ROOT/target/aarch64-apple-ios-sim/release/libjolt_ffi.a" \
#     "$PROJECT_ROOT/target/x86_64-apple-ios/release/libjolt_ffi.a" \
#     -output "$OUTPUT_DIR/libjolt_ffi_sim.a"

# Copy device library
echo "Copying device library..."
cp "$PROJECT_ROOT/target/aarch64-apple-ios/release/libjolt_ffi.a" "$OUTPUT_DIR/libjolt_ffi_device.a"

echo "Copying sim library..."
cp "$PROJECT_ROOT/target/aarch64-apple-ios-sim/release/libjolt_ffi.a" "$OUTPUT_DIR/libjolt_ffi_sim.a"


# Copy header file
echo "Copying C header file..."
cp "$PROJECT_ROOT/target/aarch64-apple-ios/release/jolt-ffi.h" "$OUTPUT_DIR/jolt-ffi.h"

## Copy Dory URS files if they exist in cache
#echo ""
#echo "Copying Dory URS files..."
#DORY_CACHE=""
#if [ -d "$HOME/Library/Caches/dory" ]; then
#    DORY_CACHE="$HOME/Library/Caches/dory"
#elif [ -d "$HOME/.cache/dory" ]; then
#    DORY_CACHE="$HOME/.cache/dory"
#fi
#
#if [ -n "$DORY_CACHE" ]; then
#    echo "  Found Dory cache at: $DORY_CACHE"
#    cp "$DORY_CACHE"/*.urs "$DEMOS_DIR/" 2>/dev/null || echo "  No URS files to copy"
#    echo "  ✓ Copied URS files to demos directory"
#else
#    echo "  Warning: Dory cache not found. URS files will be generated on first run."
#fi

# Prepare demo examples
echo ""
echo "Preparing demo examples..."
cd "$PROJECT_ROOT"

DEMOS=(
    "fibonacci"
    "sha3-chain"
    "ecdsa-sign"
)

for DEMO in "${DEMOS[@]}"; do
    echo -n "  - $DEMO: "

    ELF_OUTPUT="$DEMOS_DIR/${DEMO}.elf"
    PREPROCESSING_OUTPUT="$DEMOS_DIR/${DEMO}-preprocessing.bin"

    STD_FLAG=""
    case "$DEMO" in
        sha3-chain-guest)
            STD_FLAG="--std"
            ;;
    esac

    cargo run --release -q -p jolt-ffi --bin prepare-guest -- \
        --guest "$DEMO" \
        $STD_FLAG \
        --elf-output "$ELF_OUTPUT" \
        --preprocessing-output "$PREPROCESSING_OUTPUT" \
        --quiet
done

echo "Build complete!"
echo "Libraries created at:"
echo "  - Device: $OUTPUT_DIR/libjolt_ffi_device.a"
echo "  - Simulator: $OUTPUT_DIR/libjolt_ffi_sim.a"
echo "  - Header: $OUTPUT_DIR/jolt-ffi.h"
echo "Demos created at:"
for DEMO in "${DEMOS[@]}"; do
    echo "  - $DEMOS_DIR/${DEMO}.elf"
    echo "  - $DEMOS_DIR/${DEMO}-preprocessing.bin"
done
#echo "Dory URS files:"
#ls -1 "$DEMOS_DIR"/*.urs 2>/dev/null | while read file; do
#    echo "  - $(basename "$file")"
#done || echo "  - (none - will be generated on first run)"
