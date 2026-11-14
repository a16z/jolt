# RISC-V emulation

Jolt's [RISC-V](../appendix/risc-v.md) emulator is implemented within the `tracer` crate.
It originated as a fork of the [`riscv-rust`](https://github.com/takahirox/riscv-rust) project and has since evolved significantly to support Jolt’s unique requirements.
The bulk of the emulator's logic resides in the `instruction` subdirectory of the `tracer` crate, which defines the behavior of individual RISC-V instructions.

## Privilege and Trap Model

Jolt's RISC-V implementation targets **M-mode-only** execution with **no interrupt hardware**.
All guest programs run under [ZeroOS](https://github.com/a16z/zeroos), a minimal runtime that operates exclusively in Machine mode.
There are no S-mode or U-mode privilege levels, no timer, no CLINT, no PLIC, and no external interrupt sources.

This model has the following consequences for privileged instructions:

- **`mstatus`** has only two writable bits on such a hart, MIE and MPIE; MPP can hold only M, and every other field is read-only 0.
  Its virtual register holds just MIE and MPIE, and reads add MPP = M, so `mstatus` reads `0x1800` after reset.

- **ECALL** writes `mepc`, `mcause` (11) and `mtval` (0) to their virtual registers,
  moves MIE into MPIE and clears MIE, then jumps to `mtvec`.

- **MRET** moves MPIE into MIE, sets MPIE, and jumps to `mepc`.

- **CSR access** is restricted to a fixed whitelist: `mstatus` (0x300), `mtvec` (0x305),
  `mscratch` (0x340), `mepc` (0x341), `mcause` (0x342), `mtval` (0x343).
  Accessing any other CSR panics the tracer. All six Zicsr instructions are implemented
  (`CSRRW`, `CSRRS`, `CSRRC` and the immediate forms `CSRRWI`, `CSRRSI`, `CSRRCI`).
  Writes are legalized as the privileged specification requires: `mstatus` keeps only MIE and MPIE,
  `mepc` clears bit 0, and an `mtvec` write with a reserved MODE (2 or 3) keeps the previous MODE.
  `mscratch`, `mcause` and `mtval` store the written value.
  See `specs/m-mode-trap-csr-semantics.md` for the argument and the exact sequences.

- **EBREAK** halts the hart, as M-mode `ebreak` does under a debugger that has set `dcsr.ebreakm`.
  It is Jolt's failure halt; `jolt.success` (the word `0x0000305B`) is the success halt.
  A Jolt run must end at one of the two: the tracer rejects a run that stops anywhere else,
  such as at a `j .` self-loop, with `TraceError::EndedWithoutHalt`.

- **The R1CS constraint system has no privilege-mode or interrupt semantics.**
  CSR virtual registers (vr34–vr39) are treated as ordinary registers by the memory-checking circuit.
  The circuit verifies read/write consistency, not the architectural meaning of individual bits.

## Core traits

Two key traits form the backbone of the RISC-V emulator:

- `RISCVInstruction`
  - Defines core instruction behavior required by the tracer.
  - Includes two associated constants, `MASK` and `MATCH`, used for decoding instructions from raw bytes.
  - Has an associated type `Format`, indicating the RISC-V instruction format (e.g., R-type or I-type).
  - The associated type `RamAccess` specifies memory access (read/write) required by the instruction, guiding the tracer in capturing memory state before and after execution.
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

`LR.W` writes the reserved address into `reservation_w` (vr32) and **clears** `reservation_d` (vr33).
`LR.D` writes the reserved address into **both** registers, because a doubleword reservation set also contains the word at the same address.
So an `SC.W` succeeds after an `LR.W` or `LR.D` to the same address, and an `SC.D` succeeds only after an `LR.D` to the same address; an `SC.D` after an `LR.W` fails, because its bytes are not all in the reservation set.
An `SC` elsewhere inside a doubleword reservation set fails; the RISC-V specification allows such spurious failures.
Both `LR` and `SC` first assert that the address is at least `RAM_START`: only ordinary RAM is reservable, which also keeps a cleared reservation (0) from matching.

##### SC success and failure

The `SC` sequences are straight-line, so success is computed rather than branched on:

1. `success = (reservation == rs1)`, computed with `XOR` and `SLTIU`, using the reservation register for the SC's width.
2. The sequence loads the current memory value `mem` and stores $\text{mem} + \text{success} \cdot (\text{rs2} - \text{mem})$: `rs2` on success, the unchanged value on failure.
3. `rd` is set to 0 on success and 1 on failure.

Success is fully determined by the reservation registers, so the prover cannot choose it.
On a single hart the failed-SC store writes back the value already in memory, so memory is unchanged, as the specification requires.

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
