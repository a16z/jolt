#!/bin/bash
set -e

# Script to build jolt-ffi as a static library for iOS
# Supports: iPhone devices (arm64), Apple Silicon simulator (arm64), Intel simulator (x86_64)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FFI_DIR="$PROJECT_ROOT/jolt-ffi"
OUTPUT_DIR="$SCRIPT_DIR/JoltProver/libs"

echo "Building jolt-ffi for iOS..."
echo "Project root: $PROJECT_ROOT"
echo "FFI directory: $FFI_DIR"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# iOS device (arm64)
echo "Building for iOS device (aarch64-apple-ios)..."
cd "$FFI_DIR"
cargo build --release --target aarch64-apple-ios --message-format=short -q

# iOS simulator (Apple Silicon)
echo "Building for iOS simulator - Apple Silicon (aarch64-apple-ios-sim)..."
cargo build --release --target aarch64-apple-ios-sim --message-format=short -q

# iOS simulator (Intel)
echo "Building for iOS simulator - Intel (x86_64-apple-ios)..."
cargo build --release --target x86_64-apple-ios --message-format=short -q

# Create universal library for simulator (arm64 + x86_64)
echo "Creating universal simulator library..."
lipo -create \
    "$PROJECT_ROOT/target/aarch64-apple-ios-sim/release/libjolt_ffi.a" \
    "$PROJECT_ROOT/target/x86_64-apple-ios/release/libjolt_ffi.a" \
    -output "$OUTPUT_DIR/libjolt_ffi_sim.a"

# Copy device library
echo "Copying device library..."
cp "$PROJECT_ROOT/target/aarch64-apple-ios/release/libjolt_ffi.a" "$OUTPUT_DIR/libjolt_ffi_device.a"

# Copy header file
echo "Copying C header file..."
cp "$PROJECT_ROOT/target/aarch64-apple-ios/release/jolt-ffi.h" "$OUTPUT_DIR/jolt-ffi.h"

echo "Build complete!"
echo "Libraries created at:"
echo "  - Device: $OUTPUT_DIR/libjolt_ffi_device.a"
echo "  - Simulator: $OUTPUT_DIR/libjolt_ffi_sim.a"
echo "  - Header: $OUTPUT_DIR/jolt-ffi.h"
