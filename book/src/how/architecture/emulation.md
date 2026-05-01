# RISC-V emulation

Jolt's [RISC-V](../appendix/risc-v.md) emulator is implemented within the `tracer` crate.
It originated as a fork of the [`riscv-rust`](https://github.com/takahirox/riscv-rust) project and has since evolved significantly to support Jolt’s unique requirements.
The bulk of the emulator's logic resides in the `instruction` subdirectory of the `tracer` crate, which defines the behavior of individual RISC-V instructions.

## Privilege and Trap Model

Jolt's RISC-V implementation targets **M-mode-only** execution with **no interrupt hardware**.
All guest programs run under [ZeroOS](https://github.com/a16z/zeroos), a minimal runtime that operates exclusively in Machine mode.
There are no S-mode or U-mode privilege levels, no timer, no CLINT, no PLIC, and no external interrupt sources.

This model has the following consequences for privileged instructions:

- **ECALL** writes `mepc`, `mcause`, `mtval`, and `mstatus` to their virtual registers, then jumps to `mtvec`.
  `mstatus` is set to `0x1800` (MPP = M-mode, MIE = 0, MPIE = 0) unconditionally rather than via read-modify-write.
  This is correct because the privilege mode is always M-mode, and interrupt-enable bits are unused —
  the MIE CSR (0x304) is not in the supported CSR whitelist and cannot be accessed by guest code.

- **MRET** jumps to `mepc` and does not modify `mstatus`.
  The full RISC-V spec requires restoring `MIE` from `MPIE` and resetting `MPP`,
  but these fields are inert in the M-mode-only model.
  The ZeroOS trap trampoline restores `mstatus` via `csrw` before executing `mret`,
  so the virtual register always holds the correct value.

- **CSR access** is restricted to a fixed whitelist: `mstatus` (0x300), `mtvec` (0x305),
  `mscratch` (0x340), `mepc` (0x341), `mcause` (0x342), `mtval` (0x343).
  Accessing any other CSR panics the tracer. Only `CSRRW` and `CSRRS` are implemented;
  `CSRRC` and the immediate CSR variants are not supported.

- **The R1CS constraint system has no privilege-mode or interrupt semantics.**
  CSR virtual registers (vr34–vr39) are treated as ordinary registers by the memory-checking circuit.
  The circuit verifies read/write consistency, not the architectural meaning of individual bits.

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

`Instruction` and `Cycle` are enums encapsulating all RV64IMAC instruction variants, wrapping implementations of `RISCVInstruction` and `RISCVCycle`, respectively.

## Virtual Instructions and Sequences

A crucial concept in Jolt is that of **virtual instructions** and **virtual sequences**, introduced in section 6 of the Jolt paper. Virtual instructions don't exist in the official RISC-V ISA but are specifically created to facilitate proving.

### Reasons for Using Virtual Instructions:

#### Complex operations (e.g. division)
Some instructions, like division, don't neatly adhere to the lookup table structure required by [prefix-suffix Shout](./instruction_execution.md).
To handle these cases, the problematic instruction is expanded into a virtual sequence, a series of instructions (some potentially virtual).

For instance, division involves a sequence described in detail in section 6.3 of the Jolt paper, utilizing virtual untrusted "advice" instructions.
In the context of division, the advice instructions store the quotient and remainder in **virtual registers**, which are additional registers used exclusively within virtual sequences as scratch space.
The rest of the division sequence verifies the correctness of the computed quotient and remainder, finally storing the quotient in the destination register specified by the original instruction.

#### Atomic operations (LR/SC)

The RISC-V "A" extension's Load-Reserved/Store-Conditional (LR/SC) instructions are implemented using virtual sequences with dedicated **reservation registers**. These are a pair of width-specific virtual registers that track the address reserved by the most recent LR instruction:

- **`reservation_w`** (virtual register 32): Set by `LR.W`, checked by `SC.W`
- **`reservation_d`** (virtual register 33): Set by `LR.D`, checked by `SC.D`

When `LR.W` executes, it writes the reservation address into `reservation_w` (vr32) and **clears** `reservation_d` (vr33). Conversely, `LR.D` writes into `reservation_d` (vr33) and clears `reservation_w` (vr32). This cross-clear mechanism prevents mixed-width LR/SC pairing: an `SC.W` after `LR.D` will always fail because the word-width reservation register was cleared by the `LR.D`.

##### SC failure path

`SC.W` and `SC.D` must handle both success and failure within the zkVM's constraint system. This is accomplished using `VirtualAdvice`:

1. The prover supplies a success/failure bit via `VirtualAdvice`, constrained to $\{0, 1\}$ using `VirtualAssertLTE`.
2. A derived `v_success` flag ($1 - \text{v\_result}$) gates the reservation check: $\text{v\_success} \cdot (\text{reservation} - \text{rs1}) = 0$. On success ($\text{v\_success} = 1$), this forces the reservation to match the target address. On failure ($\text{v\_success} = 0$), the constraint is trivially satisfied.
3. The store is conditional: $\text{store\_val} = \text{mem} + (\text{rs2} - \text{mem}) \cdot \text{v\_success}$. On success this writes `rs2`; on failure it writes back the original memory value (a no-op).
4. The destination register is set to 0 on success, 1 on failure.

**Soundness**: A malicious prover cannot claim success without a valid reservation, because the constraint $\text{v\_success} \cdot (\text{reservation} - \text{rs1}) = 0$ would be violated. However, the prover *can* claim spurious failure even with a valid reservation, which is permitted by the RISC-V specification (SC is allowed to fail spuriously).

Per the RISC-V specification, SC always invalidates all reservations regardless of whether the store succeeded or failed. Both `reservation_w` and `reservation_d` are cleared to zero at the end of any SC instruction.

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
