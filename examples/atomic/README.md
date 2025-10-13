# Atomic Operations Example

This example demonstrates how Jolt properly processes atomic operations by converting them to regular instructions through the `passes=lower-atomic` compiler flag.

## Purpose

The demo tests that:
1. Rust atomic operations (`AtomicU64`) compile successfully 
2. Both `core::sync::atomic` and `portable-atomic` work identically
3. **No actual atomic instructions** are generated in the final binary
4. Atomic operations are lowered to regular load/store instructions that Jolt can trace

## Verification

After running the example, verify that no atomic instructions remain:

```bash
riscv64-unknown-elf-objdump -d /tmp/jolt-guest-targets/atomic-guest-atomic_test_u64/riscv64imac-unknown-none-elf/release/atomic-guest | grep -E "(amoadd|amoswap|amoor|amoand|amomin|amomax|amoxor|lr\.|sc\.)"
```

**Expected result**: No output (meaning no atomic instructions found).

## What It Tests

- `AtomicU64` operations (load, store, fetch_add) using `core::sync::atomic`
- Same operations using `portable-atomic` for compatibility 
- Both produce identical results when processed by `passes=lower-atomic`

## Running the Example

```bash
cd examples/atomic
cargo run --release
```

## Expected Output

```
Testing AtomicU64 compatibility (standard vs portable-atomic)...
Standard atomic result: 44
Portable-atomic result: 44
Results match: true
Proof valid: true
✓ Both standard and portable-atomic operations work correctly!
```

## Key Point

The success of this example proves that `passes=lower-atomic` correctly converts atomic operations into regular instructions that Jolt can trace and prove, eliminating the need for `portable-atomic` or `critical-section` dependencies in guest programs.