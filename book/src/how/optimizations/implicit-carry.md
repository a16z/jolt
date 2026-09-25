# Implicit carry instructions

> Available behind the `implicit-carry` cargo feature. For now it is mutually exclusive with the `zk` and `akita` features (the build fails on either combination).

## Overview

Multi-precision arithmetic (256-bit multiplication, field operations for ECDSA, etc.) compiles to long chains of `mul`/`mulhu`/`add`/`sltu` instructions: for every limb operation, ordinary RISC-V needs extra instructions just to recover and propagate the carry. The implicit-carry extension removes that bookkeeping by giving the VM a single carry register that arithmetic instructions write and two custom instructions that consume it:

- **`ADDC rd, rs1, rs2`** — `rd = low64(rs1 + rs2 + carry)`, then `carry = high64(rs1 + rs2 + carry)`
- **`MULC rd, rs1, rs2`** — `rd = low64(rs1 * rs2 + carry)`, then `carry = high64(rs1 * rs2 + carry)`

Both are R-type virtual instructions (funct3 `0b000`, funct7 `0x06`/`0x07`). The widening results fit in 128 bits: even `(2^64-1)^2 + (2^64-1) < 2^128`.

The carry register is fed by the **carry producers**: `ADD`, `MUL`, `ADDC`, and `MULC`. Each of them sets `carry` to the high 64 bits of its unsigned widening result. **Every other instruction clobbers the carry to zero**, so a carry chain must be an uninterrupted run of producer instructions; you cannot interleave loads, stores, or branches between the producer and its consumer.

With this extension, one 256-bit multiplication drops from 145 trace cycles of ordinary compiled Rust to 55 cycles of chained `MULC`/`ADDC` assembly (see the `mul256` cycle-count benchmark). Rebuilding the bigint and secp256k1 [inline](./inlines.md) sequences on these instructions roughly halves their row counts.

## No dead-code elimination: `add x0, a, b` is a real carry producer

A chain often needs the carry-out of an addition without the low 64 bits. The idiom for that is

```text
add x0, a, b    # discard low64(a + b), set carry = high64(a + b)
addc d, c, x0   # d = c + carry
```

Writes to `x0` are discarded as usual, but the instruction still executes and still sets the carry. Jolt's tracer and bytecode pipeline **never optimize such instructions away**: an `add` (or `mul`) with `rd = x0` is a deliberate carry producer, not dead code. Conversely, do not assume a "useless" `add` will be removed — under this feature every `ADD`/`MUL` writes the carry register, and the row is proven like any other.

Note this applies to hand-written assembly (inline sequences and `asm!` blocks). The Rust compiler knows nothing about the implicit carry, so compiled code never relies on it; the extension is consumed through the [inline](./inlines.md) sequences (`jolt-inlines-bigint`, `jolt-inlines-secp256k1`, ...) or through explicit `.insn` assembly.

## How it is proven

The trace gains one committed column, **`Carry`**: each row's *incoming* carry (the previous row's carry-out), a dense `u64` column with `Carry(0) = 0` (a fresh CPU starts with zero carry). Two circuit flags drive the constraints:

- `ProducesCarry` — set on `ADD`, `MUL`, `ADDC`, `MULC`
- `UsesCarry` — set on `ADDC`, `MULC`

The carry actually consumed by a row is the product-virtualized

$$\mathsf{CarryUsed} = \mathsf{UsesCarry} \cdot \mathsf{Carry},$$

which joins the existing product constraints (`Product`, `ShouldBranch`, `ShouldJump`) in the product-virtualization sumcheck. The R1CS constraints then say:

- add path: $\mathsf{RightLookupOperand} = \mathsf{LeftInstructionInput} + \mathsf{RightInstructionInput} + \mathsf{CarryUsed}$
- mul path: $\mathsf{RightLookupOperand} = \mathsf{Product} + \mathsf{CarryUsed}$
- if `ProducesCarry`: $\mathsf{RightLookupOperand} = \mathsf{LookupOutput} + 2^{64} \cdot \mathsf{NextCarry}$ — the range-check lookup forces `LookupOutput` to the low 64 bits, so `NextCarry` is exactly the high 64 bits
- if not `ProducesCarry`: $\mathsf{NextCarry} = 0$ — this is what makes non-producers clobber the carry

`NextCarry` is tied to the committed `Carry` column by the shift sumcheck (the same mechanism that relates `NextPC` to `PC`), and a dedicated stage-6 claim reduction batches the column's two openings together with the $\mathsf{Carry}(0) = 0$ boundary condition into the single opening that enters the final batched opening proof.

The extension therefore costs one extra committed `u64` column and one extra product constraint; no new lookup tables and no separate constraint system.
