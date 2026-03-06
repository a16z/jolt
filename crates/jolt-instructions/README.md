# jolt-instructions

RISC-V instruction set and lookup tables for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines the instruction abstraction layer for the Jolt lookup argument. Each RISC-V instruction is decomposed into lookup queries against small tables, which are then verified via the Twist/Shout protocol. The crate covers the full RV64IMAC instruction set.

## Public API

### Core Traits

- **`Instruction`** — A RISC-V instruction that can be executed and decomposed into lookup queries. Methods: `opcode`, `name`, `execute(x, y) -> u64`, `lookups(x, y) -> Vec<LookupQuery>`.

- **`LookupTable<F>`** — A small evaluation table used by the Twist/Shout lookup argument. Methods: `id`, `name`, `size`, `evaluate(input) -> F`, `materialize() -> Vec<F>`.

### Types

- **`TableId(u16)`** — Unique identifier for a lookup table.
- **`LookupQuery`** — A query pairing a `TableId` with a `u64` input.
- **`JoltInstructionSet`** — Registry of all RV64IMAC instructions with opcode-indexed dispatch.

### Instruction Modules

- **`rv`** — Concrete RISC-V instruction implementations:
  - `arithmetic` / `arithmetic_w` — ADD, SUB, MUL, DIV, REM (64-bit and 32-bit word variants)
  - `branch` — BEQ, BNE, BLT, BGE, BLTU, BGEU
  - `compare` — SLT, SLTU
  - `load` / `store` — LB, LH, LW, LD, SB, SH, SW, SD (with sign extension)
  - `logic` — AND, OR, XOR
  - `shift` / `shift_w` — SLL, SRL, SRA (64-bit and 32-bit word variants)
  - `system` — ECALL, EBREAK

- **`virtual_`** — Virtual instructions that combine multiple lookups:
  - `assert` — Assertion instructions
  - `arithmetic` — Virtual arithmetic operations
  - `bitwise` — Virtual bitwise operations

- **`tables`** — Lookup table implementations backing the instructions

## Dependency Position

`jolt-instructions` depends only on `jolt-field` and `serde`. It is used by `jolt-zkvm` and `jolt-core`.

## Feature Flags

This crate has no feature flags.

## License

MIT
