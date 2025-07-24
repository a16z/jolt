# Jolt Trace Recorder

A performance analysis tool for inline implementations in Jolt that provides detailed execution metrics including trace length, bytecode size, and instruction frequency analysis for RISC-V guest programs.

## Overview

The Trace Recorder analyzes the execution characteristics of Jolt guest programs by:
- Measuring execution trace length (number of execution steps)
- Computing bytecode size
- Tracking instruction frequency and cycle counts
- Supporting both RV32 and RV64 RISC-V architectures


## Prerequisites

- Rust toolchain (see [rust-toolchain.toml](../rust-toolchain.toml) for required version)
- RISC-V target support: Run `rustup target list` to check if `riscv64imac-unknown-none-elf` is installed, and if not, run `rustup target add riscv64imac-unknown-none-elf` to install it.

## Changes to the Original Code

1. **tracer/src/lib.rs**: The `decode()` function has been modified to support RISC-V 64-bit instructions. Mainly, the `decode()` function now supports compressed instructions and prevents `panic!()` during execution.

2. **Program::build()**: Has been changed to support both 32-bit and 64-bit architectures.

## Execution

Navigate to the trace recorder directory:
```bash
cd trace_recorder
```

### Basic Usage

Run trace analysis on a guest program:
```bash
cargo run --release <guest_name>
```

**Example:**
```bash
cargo run --release blake2-inline
```

### Architecture Options

The tool supports different RISC-V architectures:

**64-bit mode (default):**
```bash
cargo run --release <guest_name>
```

**32-bit mode:**
```bash
cargo run --release --features rv32 --no-default-features <guest_name>
```

**Explicit 64-bit mode:**
```bash
cargo run --release --features rv64 --no-default-features <guest_name>
```

## Available Guest Programs

The trace recorder works with any guest program in the `examples/` directory. Available inline implementations include:

### Hash Functions
- `blake2-inline` - Inline BLAKE2 implementation
- `blake2-default` - Reference BLAKE2 implementation  
- `blake3-inline` - Inline BLAKE3 implementation
- `blake3-64-inline` - Inline BLAKE3 64-bit variant
- `blake3-128-inline` - Inline BLAKE3 128-bit variant
- `blake3-192-inline` - Inline BLAKE3 192-bit variant
- `blake3-256-inline` - Inline BLAKE3 256-bit variant
- `keccak1600-inline` - Inline Keccak implementation
- `keccak1600-default` - Reference Keccak implementation


## Input Configuration

For hash function guest programs, the tool uses two types of inputs:

1. **Hash Input Data** (32 bytes):
   ```rust
   inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
   ```
   *Note: Currently hardcoded in the guest program due to serialization constraints. This input is not used at all.*

2. **Iteration Count**:
   ```rust
   inputs.append(&mut postcard::to_stdvec(&100000u32).unwrap());
   ```
   Controls the number of iterations to perform.

## Output

The tool provides:
- **Trace Length**: Number of execution steps
- **Bytecode Length**: Size of the compiled program
- **Program Output**: Hexadecimal representation of results

**Example Output:**
```
Trace length: 1234567, Bytecode length: 8901
Output is: [a1, b2, c3, ...]
```

*Note: Currently output does not show the expected output and this needs further investigation.*

## Advanced Analysis (Optional)

For detailed instruction-level analysis, uncomment the trace analysis code in `main.rs`:

```rust
let result = program.trace_analyze::<Fr>(&inputs);
```

This provides:
- Instruction frequency counts
- Complete bytecode
- Complete execution trace

**File Output:**
Trace analysis can be saved to files:
- `<guest_name>_RV32_trace.txt` (32-bit mode)
- `<guest_name>_RV64_trace.txt` (64-bit mode)

⚠️ **Warning**: Trace files can be very large. Use smaller iteration counts (100-1000) when saving detailed traces.

## Adding New Guest Programs

1. **Create Guest Program**: Add your program to the `examples/` directory
2. **Update Workspace**: Add the new program to the workspace configuration
3. **Naming Convention**: 
   - Use `-inline` suffix for inline implementations
   - Use `-default` suffix for reference implementations
