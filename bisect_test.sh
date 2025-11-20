#!/bin/bash

# Exit on first error
set -e

# Setup step
RUST_LOG=info cargo run --release -p recursion -- generate --example fibonacci

# Test step (this should fail on bad commits)
JOLT_BACKTRACE=1 RUST_BACKTRACE=1 RUST_LOG=info cargo run --release -p recursion -- trace --example fibonacci -d

# If we get here, the commit is good
exit 0