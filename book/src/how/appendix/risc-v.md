# RISC-V

[RISC-V](https://en.wikipedia.org/wiki/RISC-V) is an open-source [instruction set architecture](https://en.wikipedia.org/wiki/Instruction_set_architecture) (ISA) based on [reduced instruction set computer](https://en.wikipedia.org/wiki/Reduced_instruction_set_computer) (RISC) principles. Every RISC-V implementation includes a base integer ISA, with optional extensions for additional functionality.

## Supported Instructions

Jolt currently supports the RV64IMAC instruction set, comprising the 64-bit integer instruction set (RV64I), the integer multiplication/division extension (RV64M), the atomic memory operations extension (RV64A), and the compressed instruction extension (RV64C).

### RV64I (base)

RV64I is the base 64-bit integer instruction set: 32 registers (`x0`--`x31`, each 64 bits wide, with `x0` hardwired to zero), 32-bit fixed-width instructions, and a load/store architecture (all arithmetic operates on registers; memory is accessed only via dedicated load/store instructions with base-plus-offset addressing). It provides arithmetic, logical, shift, branch, and jump operations.

For detailed instruction formats and encoding, refer to **chapter 2** of the [specification](https://riscv.org/wp-content/uploads/2019/12/riscv-spec-20191213.pdf).

### "M" extension (integer multiply/divide)

Adds signed and unsigned multiplication (`MUL`, `MULH`, `MULHU`, `MULHSU`) and division/remainder (`DIV`, `DIVU`, `REM`, `REMU`), plus their 32-bit W-type variants. Divide-by-zero produces a well-defined result rather than a trap.

For detailed instruction formats and encoding, refer to **chapter 7** of the [specification](https://riscv.org/wp-content/uploads/2019/12/riscv-spec-20191213.pdf).

### "A" extension (atomics)

Adds atomic read-modify-write operations (`AMOSWAP`, `AMOADD`, `AMOAND`, `AMOOR`, `AMOXOR`, `AMOMIN`, `AMOMAX`) and load-reserved/store-conditional (`LR`/`SC`) pairs, each in word (32-bit) and doubleword (64-bit) variants.

#### Jolt's LR/SC implementation

Jolt implements LR/SC using [virtual sequences](../architecture/emulation.md#atomic-operations-lrsc) with a pair of width-specific **reservation registers**:

- `reservation_w` (virtual register 32) -- used by `LR.W`/`SC.W` for word (32-bit) reservations
- `reservation_d` (virtual register 33) -- used by `LR.D`/`SC.D` for doubleword (64-bit) reservations

The two reservation registers enforce width-matched pairing: `LR.W` sets `reservation_w` and clears `reservation_d`; `LR.D` does the reverse. This cross-clear means `SC.W` after `LR.D` (or `SC.D` after `LR.W`) will always fail, since the SC checks only its own width's reservation register.

The SC failure path uses prover-supplied `VirtualAdvice` constrained to $\{0, 1\}$. On success, a constraint forces the reservation address to match `rs1`, and `rs2` is stored to memory. On failure, the store is a no-op (the original memory value is written back). The destination register `rd` receives 0 on success, 1 on failure. Both reservation registers are always cleared at the end of any SC instruction, per the RISC-V specification.

For the constraint design and soundness argument, see [RISC-V emulation](../architecture/emulation.md#atomic-operations-lrsc). For the virtual register layout, see [Registers](../architecture/registers.md#virtual-register-layout).

For detailed instruction formats and encoding, refer to **chapter 8** of the [specification](https://riscv.org/wp-content/uploads/2019/12/riscv-spec-20191213.pdf).

### "C" extension (compressed instructions)

Provides 16-bit encodings for common operations, reducing binary size by ~25--30%. Compressed instructions map directly to their 32-bit RV64I equivalents and can be freely intermixed with them; the Jolt [tracer](../architecture/emulation.md) expands them before execution.

For detailed instruction formats and encoding, refer to **chapter 16** of the [specification](https://riscv.org/wp-content/uploads/2019/12/riscv-spec-20191213.pdf).

## Compilation via LLVM

Jolt proves execution of RISC-V ELF binaries, which can be produced from any language with an LLVM frontend (Rust, C, C++, etc.). The `jolt-sdk` crate handles cross-compilation to the RISC-V target automatically via the `#[jolt::provable]` macro. The compilation pipeline is:

```
Source (Rust/C/C++) → LLVM IR → RISC-V ELF → Jolt tracer → proof
```

![Compilation to RISC-V target](../../imgs/compilation_to_riscv.png)

## References
- [RISC-V Specifications](https://riscv.org/technical/specifications/)
- [RV64I Base ISA Specification](https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf)
- [RISC-V Assembly Programmer's Manual](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc)
- [LLVM RISC-V Backend Documentation](https://llvm.org/docs/RISCVUsage.html)
