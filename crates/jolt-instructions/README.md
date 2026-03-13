# jolt-instructions

RISC-V instruction set and lookup tables for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines the instruction abstraction layer for the Jolt lookup argument. Each RISC-V instruction is decomposed into lookup queries against small tables, which are then verified via the Twist/Shout protocol. The crate covers the full RV64IMAC instruction set.

## Public API

### Core Traits

- **`Instruction: Flags`** -- A RISC-V instruction. Methods: `opcode()`, `name()`, `execute(x, y) -> u64`, `lookup_table() -> Option<LookupTableKind>`.
- **`LookupTable<XLEN>`** -- A small evaluation table. Methods: `materialize_entry(index) -> u64`, `evaluate_mle(r) -> F`.
- **`Flags`** -- Static flag configuration. Methods: `circuit_flags() -> [bool; NUM_CIRCUIT_FLAGS]`, `instruction_flags() -> [bool; NUM_INSTRUCTION_FLAGS]`.
- **`ChallengeOps<F>`** -- Arithmetic bounds for challenge-field operations in prefix/suffix evaluation.
- **`FieldOps<C>`** -- Field-side bounds for challenge arithmetic.

### Types

- **`LookupTableKind`** -- `#[repr(u8)]` enum identifying one of 40 distinct lookup table types.
- **`LookupTables<XLEN>`** -- Runtime dispatch enum over all concrete table implementations. Constructed from `LookupTableKind` via `From`.
- **`LookupBits`** -- Compact 17-byte bitvector for lookup index substrings (prefix/suffix decomposition).
- **`CircuitFlags`** -- R1CS-relevant boolean flags (14 variants: `AddOperands`, `Load`, `Store`, `Jump`, etc.).
- **`InstructionFlags`** -- Non-R1CS flags for witness generation (7 variants: `LeftOperandIsPC`, `RightOperandIsImm`, `Branch`, etc.).
- **`JoltInstructionSet`** -- Registry of all RV64IMAC instructions with opcode-indexed dispatch.

### Modules

- **`rv`** -- Concrete RISC-V instruction implementations (arithmetic, arithmetic_w, branch, compare, jump, load, logic, shift, shift_w, store, system).
- **`virtual_`** -- Virtual instructions (advice, arithmetic, assert, bitwise, byte, division, extension, shift, xor-rotate).
- **`tables`** -- Lookup table implementations with prefix/suffix sparse-dense decomposition.

## Dependency Position

`jolt-instructions` depends only on `jolt-field` and `serde`. It is used by `jolt-zkvm`.

## Feature Flags

This crate has no feature flags.

## License

MIT
