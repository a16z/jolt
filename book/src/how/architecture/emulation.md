# RISC-V emulation

Jolt's [RISC-V](../appendix/risc-v.md) emulator is implemented within the `tracer` crate.
It originated as a fork of the [`riscv-rust`](https://github.com/takahirox/riscv-rust) project and has since evolved significantly to support Jolt’s unique requirements.
The bulk of the emulator's logic resides in the `instruction` subdirectory of the `tracer` crate, which defines the behavior of individual RISC-V instructions.

## Core traits

Two key traits form the backbone of the RISC-V emulator:

- `RISCVInstruction`
  - Defines core instruction behavior required by the tracer.
  - Includes two associated constants, `MASK` and `MATCH`, used for decoding instructions from raw bytes.
  - Has an associated type `Format`, indicating the RISC-V instruction format (e.g., R-type or I-type).
  - The associated type `RAMAccess` specifies memory access (read/write) required by the instruction, guiding the tracer in capturing memory state before and after execution.
  - Its critical method, `execute`, emulates instruction execution by modifying CPU state and populating memory state changes as needed.

- `RISCVTrace` (extends `RISCVInstruction`)
  - Adds a `trace` method, which by default invokes `execute` and additionally captures pre- and post-execution states of any accessed registers or memory.
  May be overridden for instructions which employ **virtual sequences**, see below.
  - Constructs a `RISCVCycle` struct instance from the captured information and appends it to a trace vector.

## Core enums

`RV32IMInstruction` and `RV32IMCycle` are enums encapsulating all RV32IM instruction variants, wrapping implementations of `RISCVInstruction` and `RISCVCycle`, respectively.

## Virtual Instructions and Sequences

A crucial concept in Jolt is that of **virtual instructions** and **virtual sequences**, introduced in section 6 of the Jolt paper. Virtual instructions don't exist in the official RISC-V ISA but are specifically created to facilitate proving.

### Reasons for Using Virtual Instructions:

#### Complex operations (e.g. division)
Some instructions, like division, don't neatly adhere to the lookup table structure required by [prefix-suffix Shout](./instruction_execution.md).
To handle these cases, the problematic instruction is expanded into a virtual sequence, a series of instructions (some potentially virtual).

For instance, division involves a sequence described in detail in section 6.3 of the Jolt paper, utilizing virtual untrusted "advice" instructions.
In the context of division, the advice instructions store the quotient and remainder in **virtual registers**, which are additional registers used exclusively within virtual sequences as scratch space.
The rest of the division sequence verifies the correctness of the computed quotient and remainder, finally storing the quotient in the destination register specified by the original instruction.

#### Performance optimization (inlines)

Virtual sequences can also be employed to optimize prover performance on specific operations, e.g. hash functions. For details, refer to [Inlines](../optimizations/inlines.md).

### Implementation details

Virtual instructions reside alongside regular instructions within the `instructions` subdirectory of the `tracer` crate.
Instructions employing virtual sequences implement the `VirtualInstructionSequence` trait, explicitly defining the sequence for emulation purposes.
The `execute` method executes the instruction as a single operation, while the `trace` method executes each instruction in the virtual sequence.

### Performance considerations

A first-order approximation of Jolt's prover cost profile is "pay-per-cycle": each cycle in the trace costs *roughly* the same to prove, regardless of which instruction is being proven or whether the given cycle is part of a virtual sequence.
This means that instructions that must be expanded to virtual sequences are more expensive than their unexpanded counterparts.
An instruction emulated by an eight-instruction virtual sequence is approximately eight times more expensive to prove than a single, standard instruction.

On the other hand, virtual sequences can be used to *improve* prover performance on key operations (e.g. hash functions). This is discussed in the [Inlines](../optimizations/inlines.md) section.
