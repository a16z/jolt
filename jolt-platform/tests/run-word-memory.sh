#!/usr/bin/env bash
set -euo pipefail
# Linux rental only. The test executes the same assembly that the guest links.
if [[ "$(uname -s)" != Linux ]]; then
    echo "word-memory tests require the Linux validation box" >&2
    exit 2
fi
test_dir="$(dirname "$(realpath "$0")")"
test_out="${WORD_MEMORY_TEST_OUT:-/tmp/jolt-word-memory-tests}"
mkdir -p "$test_out"
riscv64-linux-gnu-gcc -static -O2 -Wall -Wextra -Werror \
    -fno-builtin -fno-tree-loop-distribute-patterns \
    -march=rv64gc -mabi=lp64d \
    "$test_dir/word_memory.c" "$test_dir/../src/word_memory.S" \
    -o "$test_out/word-memory-tests"
qemu-riscv64 "$test_out/word-memory-tests"
