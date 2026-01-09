# Inlines

## Overview

Jolt inlines are a unique optimization technique that replaces high-level operations with optimized sequences of native RISC-V instructions. Unlike traditional precompiles that operate in a separate constraint system, inlines remain fully integrated within the main Jolt zkVM execution model, requiring no additional "glue" logic for memory correctness. Similar to the [virtual sequences](https://jolt.a16zcrypto.com/how/architecture/emulation.html#virtual-instructions-and-sequences) already used for certain RISC-V instructions, inlines expand into sequences of simpler operations, but with additional optimizations like extended registers and custom instructions.

### Key Characteristics

**Native RISC-V Integration**: Inlines expand into sequences of RISC-V instructions that execute within the same trace as regular program code. This seamless integration eliminates the complexity of bridging between different proof systems.

**Custom Instructions**: Jolt enables the creation of custom instructions that can accelerate common operations. These custom instructions must have structured multilinear extension (MLE) polynomials, meaning that they can be evaluated efficiently in small space (see [prefix-suffix sumcheck](../instruction-execution.html) for details on structured MLEs). By ensuring all custom instructions maintain this property, Jolt achieves the performance benefits of specialized operations without sacrificing the simplicity of its proof system. This is the core innovation that distinguishes Jolt inlines from traditional precompiles or simple assembly optimizations - we compress complex operations into lookup-friendly instructions that remain fully verifiable within the main zkVM, eliminating the need for complex glue logic or separate constraint systems.

**Extended Register Set**: Inline sequences have access to 32 additional registers beyond the standard RISC-V register set. This expanded register space allows complex operations to maintain state in registers rather than memory, dramatically reducing load/store operations.

## Example Usage

Jolt provides optimized inline implementations for common cryptographic operations. The SHA-256 implementation demonstrates the power of this approach. See the `examples/sha2-chain` directory for a complete example.

### Basic Usage

```rust
use jolt_inlines_sha2::Sha256;

// Simple one-shot hashing
let input = b"Hello, Jolt!";
let hash = Sha256::digest(input);

// Incremental hashing
let mut hasher = Sha256::new();
hasher.update(b"Hello, ");
hasher.update(b"Jolt!");
let hash = hasher.finalize();
```

### Chained Hashing Example

```rust
use jolt_inlines_sha2::Sha256;

#[jolt::provable]
fn sha2_chain(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    let mut hash = input;
    for _ in 0..num_iters {
        hash = Sha256::digest(&hash);
    }
    hash
}
```

### Direct Inline Assembly Access

For advanced use cases, you can invoke inlines directly through inline assembly. Jolt uses a structured encoding scheme for inline instructions:

- **Opcode**: `0x0B` for Jolt core inlines, `0x2B` for user-defined inlines
- **funct7**: Identifies the type of operation (e.g., `0x00` for SHA2)
- **funct3**: Identifies sub-instructions within that operation (e.g., `0x0` for SHA256, `0x1` for SHA256INIT)

#### Jolt Core Inlines Reference

| Inline        | Opcode | funct7 | funct3 | Description                                |
| ------------- | ------ | ------ | ------ | ------------------------------------------ |
| SHA256        | 0x0B   | 0x00   | 0x00   | SHA-256 compression with existing state    |
| SHA256INIT    | 0x0B   | 0x00   | 0x01   | SHA-256 compression with initial constants |
| KECCAK256     | 0x0B   | 0x01   | 0x00   | Keccak-256 permutation                     |
| BLAKE2B       | 0x0B   | 0x02   | 0x00   | BLAKE2b compression                        |
| BLAKE3        | 0x0B   | 0x03   | 0x00   | BLAKE3 compression                         |
| BLAKE3KEYED64 | 0x0B   | 0x03   | 0x01   | BLAKE3 compression keyed                   |
| BIGINT256_MUL | 0x0B   | 0x04   | 0x00   | 256-bit bigint multiplication              |

```rust
unsafe {
    // SHA256 compression with existing state
    // opcode=0x0B (core inline), funct3=0x0 (SHA256), funct7=0x00 (SHA2 family)
    core::arch::asm!(
        ".insn r 0x0B, 0x0, 0x00, x0, {}, {}",
        in(reg) input_ptr,  // Pointer to 16 u32 words
        in(reg) state_ptr,  // Pointer to 8 u32 words
        options(nostack)
    );

    // SHA256 compression with initial constants
    // opcode=0x0B (core inline), funct3=0x1 (SHA256INIT), funct7=0x00 (SHA2 family)
    core::arch::asm!(
        ".insn r 0x0B, 0x1, 0x00, x0, {}, {}",
        in(reg) input_ptr,
        in(reg) state_ptr,
        options(nostack)
    );
}
```

Additional inlines for Secp256k1 operations are available but wrapped in a higher-level API. See `jolt-inlines/secp256k1` for details and `examples/secp256k1-ecdsa-verify` for an example of a higher-level ECDSA verification function using the Secp256k1 inlines.

## Error Handling in Secp256k1

When working with inlines that can fail (like cryptographic verification), it's important to understand how different error handling approaches affect the resulting proof.

**Return `Result`** - Use when you want the guest program to handle errors gracefully:
```rust
let result = ecdsa_verify(z, r, s, q);
match result {
    Ok(()) => { /* signature valid */ }
    Err(e) => { /* handle invalid signature */ }
}
```
The proof is valid regardless of the outcome and proves which branch was taken.

**`.unwrap()`** - Use when an error is unexpected and should terminate execution:
```rust
ecdsa_verify(z, r, s, q).unwrap();
```
If verification fails, the program panics. The proof is still valid and proves that the program panicked at this point.

**`.unwrap_or_spoil_proof()`** - Use when you want to **assert** a condition such that no valid proof can exist if it fails:
```rust
use jolt_inlines_secp256k1::UnwrapOrSpoilProof;

ecdsa_verify(z, r, s, q).unwrap_or_spoil_proof();
```
If verification fails, the proof becomes unsatisfiable. This is appropriate when:
- You want to prove "the signature IS valid" (not "I checked the signature")
- A malicious prover should not be able to produce any proof if the condition fails
- The error case represents something that should be cryptographically impossible

## Benchmarks

The table below compares the performance of reference and inline implementations for each hash function, using identical 32KB inputs and the same API across both reference and inline implementations.

| Hash Function | Implementation | Cycles | Cycles Per Byte (CPB) | Speedup |
|--------------|----------------|----------------|----------------------|---------|
| SHA-256      | [sha2 crate](https://crates.io/crates/sha2)      | 10,414,653     | 317.94               | -       |
| SHA-256      | **Jolt Inline**     | **1,765,207**  | **53.89**            | **5.9×** |
| Keccak-256   | [sha3 crate](https://crates.io/crates/sha3)      | 2,556,519      | 78.04                | -       |
| Keccak-256   | **Jolt Inline**     | **848,224**    | **25.89**            | **3.01×** |
| Blake2B      | [blake2 crate](https://crates.io/crates/blake2)      | 968,562        | 29.57                | -       |
| Blake2B      | **Jolt Inline**     | **340,787**    | **10.40**            | **2.85×** |

*Note: Blake3 currently supports inputs up to 64 bytes. Full implementation for larger inputs is in development.*

#### Proving Time

Proving time is hardware-dependent. The Jolt prover achieves approximately 500 kHz throughput (proving 500,000 RISC-V cycles per second) on a MacBook M4 Max, and 1.5 MHz throughput (1,500,000 cycles per second) on an AMD Threadripper Pro 7975.

**Hardware Specifications:**
- **MacBook M4 Max**: 16 cores, 128 GB RAM
- **AMD Threadripper Pro 7975**: 32 cores

The following table shows the data that can be proved by each of the Jolt inlines per second.

| Hash Function | MacBook M4 Max (500 kHz) | Threadripper Pro 7975WX (1.5 MHz) |
|--------------|---------------------|----------------------|
| SHA-256 Inline | 9.1 KB/s | 27.1 KB/s |
| Keccak-256 Inline | 18.8 KB/s | 56.1 KB/s |
| Blake2B Inline | 47.1 KB/s | 139.1 KB/s |


## Jolt CPU Advantages

The Jolt zkVM architecture provides several unique optimization opportunities that inlines can leverage:

### 1. Extended Virtual Registers

Inline sequences have access to 32 additional virtual registers beyond the standard RISC-V register set. This allows complex operations to maintain their entire working state in registers, eliminating hundreds of load/store operations that would otherwise be required. Importantly, this expanded register usage comes with virtually zero additional cost to the prover, making it an essentially "free" optimization from a proof generation perspective.

### 2. Custom Instructions

Jolt allows creation of custom instructions that can replace common multi-instruction patterns with a single operation. The key innovation here is that these instructions must have structured multilinear extensions (MLEs) that can be evaluated efficiently in small space (see [prefix-suffix sumcheck](../instruction-execution.html)). This is where the real performance gain comes from: by compressing operations into forms that work naturally with Jolt's lookup-based architecture, we achieve dramatic speedups without the complexity of traditional precompiles.

This is fundamentally different from traditional assembly optimization - we're not just rearranging instructions, we're creating new ones that are specifically designed to be "lookupable" within Jolt's proof system. For example, the ROTRI (rotate right immediate) instruction replaces the three-instruction sequence `(x >> imm) | (x << (32-imm))` with a single cycle, while remaining fully verifiable through lookups because it maintains the structured MLE property.

Note that creating custom user-defined instructions is currently only available within the core Jolt codebase and not yet supported in external crates.

### 3. 64-bit Immediate Values

Unlike standard RISC-V which limits immediate values to 12 or 20 bits, inlines can use full 64-bit immediate values. This eliminates the need for multiple instructions to load large constants, reducing both cycle count and register usage.

## Creating Custom Inlines

For implementing custom inlines, refer to the existing implementations in the `jolt-inlines/` directory, particularly the SHA2 implementation in `jolt-inlines/sha2/`.

### Key Requirements and Restrictions

When creating user-defined inlines, you must adhere to these critical requirements:

1. **Opcode Space**: Use opcode `0x2B` for user-defined inlines (`0x0B` is reserved for Jolt core inlines)

2. **Virtual Register Management**:
   - All virtual registers (registers 32-63) must be zeroed out at the end of the inline sequence
   - This ensures clean state for subsequent operations

3. **Register Preservation**:
   - Inlines cannot modify any of the real 32 RISC-V registers, including the destination register (`rd`)
   - The inline must operate purely through memory operations and virtual registers

4. **Instruction Encoding**:
   - Use `funct7` to identify your operation type (must be unique among user-defined inlines)
   - Use `funct3` for sub-instruction variants within your operation

5. **MLE Structure**:
   - All custom instructions must have structured multilinear extensions (see [prefix-suffix sumcheck](../instruction-execution.html))
   - Complex operations may need to be broken down into simpler instructions that maintain this property

### Implementation Structure

A typical inline implementation consists of three main components:

1. **SDK Module**: Provides safe, high-level API for guest programs
2. **Execution Module**: Implements host-side execution logic for testing and verification
3. **Trace Generator**: Generates the optimized RISC-V instruction sequence that replaces the inline

### Design Considerations

When designing your inline, consider:

- **Register Allocation**: Maximize use of the 32 additional virtual registers to minimize memory operations
- **Custom Instructions**: Identify patterns that could benefit from custom instructions (creating custom user-defined instructions is not available at this time)
- **Immediate Values**: Leverage 64-bit immediate values to reduce instruction count
- **Memory Access Patterns**: Structure your algorithm to minimize load/store operations

For concrete examples and implementation patterns, study the existing inline implementations in the Jolt codebase.

## Future Directions

The inline system continues to evolve with planned enhancements:

- **Extended instruction set**: Additional custom instructions for common patterns
- **Automated inline generation**: Compiler-driven inline synthesis for hot code paths
- **Larger register files**: Expanding beyond 32 additional registers for complex algorithms
- **Domain-specific optimizations**: Specialized inlines for bigint arithmetic, elliptic curves, and other cryptographic primitives

Inlines represent a fundamental innovation in zkVM design, demonstrating that significant performance improvements are possible while maintaining the simplicity and verifiability of a RISC-V-based architecture.
