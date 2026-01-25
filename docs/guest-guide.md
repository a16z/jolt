# Jolt Guest Program Development Guide

This guide covers how to develop guest programs for proving with Jolt zkVM, including feature configuration, debugging techniques, and binary inspection tools.

**Examples:**
- [muldiv](../examples/muldiv/guest) - Simple no_std example with cycle tracking
- [sig-recovery](../examples/sig-recovery/guest) - std mode with parallel signature recovery via rayon
- [merkle-tree](../examples/merkle-tree/guest) - Trusted/untrusted advice demonstration

## Table of Contents

1. [Guest Program Structure](#guest-program-structure)
2. [Feature Flags and Kernel Abilities](#feature-flags-and-kernel-abilities)
3. [Input Types](#input-types)
4. [Cycle Tracking and Profiling](#cycle-tracking-and-profiling)
5. [Binary Inspection with nm and readelf](#binary-inspection-with-nm-and-readelf)
6. [Troubleshooting](#troubleshooting)

---

## Guest Program Structure

A Jolt guest program is a Rust library containing functions marked with `#[jolt::provable]`. The typical structure is:

```
guest/
├── Cargo.toml
└── src/
    ├── main.rs    # Required entry point (minimal)
    └── lib.rs     # Provable functions
```

### main.rs (Required Entry Point)

```rust
//! Guest program entry point for Jolt zkVM
//!
//! This file is required by Jolt's build system. The actual provable
//! functions are defined in lib.rs.

#![no_main]

#[allow(unused_imports)]
use your_guest_crate::*;
```

The `#![no_main]` attribute prevents Rust from generating a standard main function. The actual entry point is handled by ZeroOS boot sequence.

### lib.rs (Provable Functions)

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyOutput {
    pub result: u32,
}

#[jolt::provable(
    max_input_size = 4096,
    max_output_size = 4096,
    memory_size = 33554432,
    stack_size = 131072,
    max_trace_length = 16777216
)]
pub fn my_function(input: &[u8]) -> MyOutput {
    // Your provable logic here
    MyOutput { result: input.len() as u32 }
}
```

### The `#[jolt::provable]` Macro

The `#[jolt::provable]` macro transforms a function into a provable computation.

#### Macro Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_input_size` | 4096 | Maximum input size in bytes |
| `max_output_size` | 4096 | Maximum output size in bytes |
| `memory_size` | 33554432 (32MB) | Total memory allocation |
| `stack_size` | 4096 | Stack size in bytes |
| `max_trace_length` | 16777216 (16M) | Maximum execution trace length |
| `max_trusted_advice_size` | - | Maximum trusted advice input size |
| `max_untrusted_advice_size` | - | Maximum untrusted advice input size |
| `guest_only` | false | Prevent host compilation (for RISC-V-specific code) |

---

## Feature Flags and Kernel Abilities

The `jolt-sdk` crate provides feature flags to enable special kernel abilities in your guest program.

### Available Features

| Feature | Description |
|---------|-------------|
| `guest-nostd` | Lightweight no-std mode (default) |
| `guest-std` | Enable Rust standard library support with musl runtime |
| `thread` | Cooperative threading support via ZeroOS scheduler |
| `random` | Random number generation (deterministic seed for zkVM) |
| `stdout` | Console output support via VFS |
| `debug` | Debug feature for syscall logging and enhanced output |

### no_std Mode (Default)

From [muldiv](../examples/muldiv/guest):

```toml
# guest/Cargo.toml
[package]
name = "muldiv-guest"
version = "0.1.0"
edition = "2021"

[features]
guest = []

[dependencies]
jolt = { package = "jolt-sdk", path = "../../../jolt-sdk" }
```

```rust
#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;

#[jolt::provable(memory_size = 32768, max_trace_length = 65536)]
fn muldiv(a: u32, b: u32, c: u32) -> u32 {
    use jolt::{end_cycle_tracking, start_cycle_tracking};

    start_cycle_tracking("muldiv");
    let result = black_box(a * b / c); // use black_box to keep code in place
    end_cycle_tracking("muldiv");
    result
}
```

### std Mode with Full Capabilities

From [sig-recovery](../examples/sig-recovery/guest):

```toml
# guest/Cargo.toml
[package]
name = "sig-recovery-guest"
version = "0.1.0"
edition = "2021"

[features]
guest = []

[dependencies]
jolt = { package = "jolt-sdk", path = "../../../jolt-sdk", features = [
    "guest-std",
    "thread",
    "stdout",
] }
reth-primitives-traits = { workspace = true }
reth-ethereum-primitives = { workspace = true }
alloy-eips = { workspace = true }
serde = { workspace = true }
postcard = { workspace = true }
```

```rust
use alloy_eips::eip2718::Decodable2718;
use reth_ethereum_primitives::TransactionSigned;
use reth_primitives_traits::transaction::recover::recover_signers;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub tx_count: u32,
    pub recovered_count: u32,
    pub signers: Vec<[u8; 20]>,
}

#[jolt::provable(
    max_input_size = 1048576,
    max_output_size = 65536,
    memory_size = 33554432,
    stack_size = 131072,
    max_trace_length = 33554432
)]
pub fn verify_txs(txs_bytes: &[u8]) -> VerificationResult {
    jolt::start_cycle_tracking("deserialize");

    let rlp_txs: Vec<Vec<u8>> = postcard::from_bytes(txs_bytes).unwrap_or_default();
    let txs: Vec<TransactionSigned> = rlp_txs
        .iter()
        .filter_map(|rlp| TransactionSigned::decode_2718(&mut rlp.as_slice()).ok())
        .collect();

    jolt::end_cycle_tracking("deserialize");

    jolt::start_cycle_tracking("recover_signers");
    let addresses = recover_signers(&txs).unwrap_or_default();
    jolt::end_cycle_tracking("recover_signers");

    VerificationResult {
        tx_count: txs.len() as u32,
        recovered_count: addresses.len() as u32,
        signers: addresses.into_iter().map(|a| a.0.0).collect(),
    }
}
```

**Note**: The `thread` feature enables ZeroOS cooperative threading, required for parallel processing. The `recover_signers` function uses rayon's `into_par_iter()` when the `rayon` feature is enabled on `reth-primitives-traits`.

### Debug Feature

Enable syscall logging by adding the `debug` feature:

```toml
jolt = { package = "jolt-sdk", path = "path/to/jolt-sdk", features = [
    "guest-std",
    "debug",  # Enable syscall logging
] }
```

When enabled, the trap handler logs each syscall:

```rust
// In jolt-sdk/src/support/trap.rs
#[cfg(feature = "debug")]
zeroos::debug::writeln!("[syscall] {}", zeroos::os::linux::syscall_name((*regs).a7));
```

This outputs syscall names during execution, helping diagnose issues with system calls.

---

## Input Types

Jolt supports three types of inputs, distinguished by what the verifier knows:

### 1. Public Input

Known to both prover and verifier. The verifier receives these values directly:

```rust
#[jolt::provable]
fn add(x: u32, y: u32) -> u32 {
    x + y
}
```

### 2. Untrusted Advice

Known only to the prover. The verifier has no knowledge of these values—not the data itself, nor any commitment. The proof demonstrates the computation was correct for *some* input, but the verifier cannot determine what that input was:

```rust
use jolt::UntrustedAdvice;

#[jolt::provable]
fn compute_with_secret(public_root: [u8; 32], secret_leaf: UntrustedAdvice<[u8; 32]>) -> [u8; 32] {
    let leaf_data = *secret_leaf;  // Dereference to access data
    // Compute with secret data...
    public_root
}
```

### 3. Trusted Advice

The prover has the data; the verifier has a polynomial commitment to it. Before proving, generate a commitment using `commit_trusted_advice_*`. Both prover and verifier receive this commitment, allowing verification that the prover used specific committed data without revealing the data:

```rust
use jolt::TrustedAdvice;

#[jolt::provable]
fn merkle_tree(
    leaf1: &[u8],                           // Public input
    leaf2: TrustedAdvice<[u8; 32]>,          // Verifier has commitment
    leaf3: TrustedAdvice<[u8; 32]>,          // Verifier has commitment
    leaf4: UntrustedAdvice<[u8; 32]>,        // Verifier knows nothing
) -> [u8; 32] {
    // Access trusted advice via deref
    let l2 = *leaf2;
    let l3 = *leaf3;
    // Compute merkle root...
    [0u8; 32]
}
```

**Host-side usage:**

```rust
// 1. Generate commitment to trusted advice before proving
let (commitment, _hint) = guest::commit_trusted_advice_merkle_tree(
    TrustedAdvice::new(leaf2),
    TrustedAdvice::new(leaf3),
    &prover_preprocessing,
);

// 2. Prover receives all data + commitment
let (output, proof, io) = prove(leaf1, TrustedAdvice::new(leaf2),
    TrustedAdvice::new(leaf3), UntrustedAdvice::new(leaf4), commitment.clone());

// 3. Verifier receives only: public inputs, output, commitment, proof
let valid = verify(leaf1, output, io.panic, commitment, proof);
```

---

## Cycle Tracking and Profiling

Measure execution cycles for performance analysis:

### API

```rust
use jolt::{start_cycle_tracking, end_cycle_tracking};

#[jolt::provable]
fn my_function(input: &[u8]) -> u32 {
    start_cycle_tracking("deserialize");
    let data: Vec<u32> = postcard::from_bytes(input).unwrap();
    end_cycle_tracking("deserialize");

    start_cycle_tracking("compute");
    let result = data.iter().sum();
    end_cycle_tracking("compute");

    result
}
```

### Preventing Compiler Optimization

Use `core::hint::black_box()` to prevent the compiler from moving code:

```rust
use core::hint::black_box;
use jolt::{start_cycle_tracking, end_cycle_tracking};

#[jolt::provable]
fn muldiv(a: u32, b: u32, c: u32) -> u32 {
    start_cycle_tracking("muldiv");
    let result = black_box(a * b / c);  // Prevents optimization
    end_cycle_tracking("muldiv");
    result
}
```

### Expected Output

```text
"deserialize": 1523 RV64IMAC cycles, 2847 virtual cycles
"compute": 892 RV64IMAC cycles, 1654 virtual cycles
Trace length: 4501
```

---

## Binary Inspection with nm and readelf

After building your guest program, inspect the resulting ELF binary to verify correct compilation.

### Building the Guest

```bash
# no_std mode (muldiv example)
RUST_LOG=debug cargo build --release -p muldiv-guest

# std mode (sig-recovery example)
RUST_LOG=debug cargo build --release -p sig-recovery-guest
```

The ELF binary is output to:
```
# no_std mode
target/riscv64imac-unknown-none-elf/release/muldiv-guest

# std mode
target/riscv64imac-unknown-linux-musl/release/sig-recovery-guest
```

### Using readelf

#### Check Entry Point

```bash
readelf -h target/riscv64imac-unknown-linux-musl/release/sig-recovery-guest
```

Expected output:
```
ELF Header:
  ...
  Entry point address:               0x80000000  # Should be non-zero, in RAM
  ...
```

**Important**: Entry point should be non-zero and typically near `0x8000_0000`.

#### Check Sections

```bash
readelf -S target/riscv64imac-unknown-linux-musl/release/sig-recovery-guest
```

Verify:
- `.text` section has non-zero size
- `.rodata`, `.data`, `.bss` sections exist

#### Check Symbols

```bash
readelf -s target/riscv64imac-unknown-linux-musl/release/sig-recovery-guest | grep -E "_start|__runtime_bootstrap|__main_entry|__platform_bootstrap"
```

Expected symbols:
```
   123: 0000000080000000   ...  _start
   456: 0000000080001234   ...  __runtime_bootstrap
   789: 0000000080002345   ...  __platform_bootstrap
```

### Using nm

#### List All Symbols

```bash
nm target/riscv64imac-unknown-linux-musl/release/sig-recovery-guest | head -50
```

#### Find Specific Functions

```bash
# Find your provable function
nm target/riscv64imac-unknown-linux-musl/release/sig-recovery-guest | grep verify_txs

# Find boot chain symbols
nm target/riscv64imac-unknown-linux-musl/release/sig-recovery-guest | grep -E "^[0-9a-f]+ [Tt] _"
```

#### Check for Missing Symbols

```bash
# Look for undefined symbols (should be minimal)
nm -u target/riscv64imac-unknown-linux-musl/release/sig-recovery-guest
```

### Using objdump

#### Disassemble Entry Point

```bash
objdump -d target/riscv64imac-unknown-linux-musl/release/sig-recovery-guest | head -100
```

#### Check for Specific Instructions

```bash
# Look for ECALL instructions (syscalls)
objdump -d target/riscv64imac-unknown-linux-musl/release/sig-recovery-guest | grep ecall
```

### Common Issues and What to Look For

| Issue | Symptom | Check |
|-------|---------|-------|
| Empty `.text` section | Entry point is 0x0 | `readelf -S` shows `.text` size 0 |
| Missing boot chain | Panic at preprocessing | `nm` missing `_start` symbol |
| Wrong architecture | Execution fails | `readelf -h` shows wrong machine type |
| Stripped symbols | No backtrace info | `nm` shows minimal symbols |

---

## Troubleshooting

### Insufficient Stack Size

**Symptom**: Unpredictable errors or crashes during execution.

**Solution**: Increase `stack_size` parameter:

```rust
#[jolt::provable(stack_size = 131072, memory_size = 33554432)]
fn my_function() { ... }
```

### Maximum Input/Output Size Exceeded

**Symptom**: Error about input/output size limits.

**Solution**: Increase size limits:

```rust
#[jolt::provable(max_input_size = 1048576, max_output_size = 65536)]
fn my_function(input: &[u8]) -> Vec<u8> { ... }
```

### Guest Fails to Compile on Host

**Symptom**: Compilation errors for RISC-V-specific code.

**Solution**: Use `guest_only` to skip host compilation:

```rust
#[jolt::provable(guest_only)]
fn risc_v_specific() {
    use core::arch::asm;
    unsafe {
        asm!("nop");
    }
}
```

### Entry Point is 0x0

**Symptom**: `readelf -h` shows entry point as 0x0, preprocessing fails.

**Cause**: Missing boot chain (no `_start` symbol).

**Solution**: Ensure ZeroOS runtime is properly linked. Check that:
1. Guest Cargo.toml includes correct jolt-sdk features
2. `main.rs` has `#![no_main]` and imports guest crate

### Empty .text Section

**Symptom**: `readelf -S` shows `.text` size 0.

**Cause**: Linker garbage collection removed all code (no entry point reference).

**Solution**: Verify boot chain symbols exist and are referenced.

### Logs Not Appearing

**Solution**: Set `RUST_LOG` environment variable:

```bash
RUST_LOG=info cargo run --release -p my-example
RUST_LOG=debug cargo run --release -p my-example  # More verbose
```

### Getting Help

If none of the above solve your problem, create a GitHub issue with:
- Jolt commit hash
- Hardware/container configuration
- Minimal guest program to reproduce
- Output of `readelf -h` and `readelf -S` on guest ELF

---

## References

- [Jolt Book](https://jolt.a16zcrypto.com/)
- [Jolt Repository](https://github.com/a16z/jolt)
- [ZeroOS Documentation](https://github.com/a16z/zeroos)
- [RISC-V Specification](https://riscv.org/technical/specifications/)
