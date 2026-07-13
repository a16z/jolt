# Rust To Lean Bytecode Expansion Extractor

This crate is a lightweight wrapper around `jolt-riscv` and `jolt-program` to automatically extract Jolt bytecode expansions in Lean.
There is only 1 file `main.rs` and it is heavily commented. 
By tracing the usage command below, and following the comments the design should be self explanatory.
We do not modify or touch the live prover/verifier in any meaningful way, and thus this crate is entirely self contained.

## What Is Not Done

The following instructions that are expandable are not auto-expanded as we are not yet sure what the correct Lean definition looks like.

```rust
// Source-only kinds we deliberately leave hand-written for now.
fn is_unsupported(kind: SourceInstructionKind) -> bool {
    matches!(
        kind.name(),
        // System / CSR
        "ECALL" | "EBREAK" | "MRET" | "CSRRW" | "CSRRS"
        // Load-reserved / store-conditional (hand-coded in LoadReserved.lean)
        | "LRW" | "LRD" | "SCW" | "SCD"
        // Registered inline dispatch (needs an InlineExpansionProvider)
        | "Inline"
    )
}

```

## Usage 

To see bytecode expansion for the `LB` instruction in Lean 

```zsh
cargo run -p jolt-lean-gen -- LB


--- LB ---
rd == x0:
  .instr (.ADDI (.vreg v41) (.xreg rs1) imm) <|
  .instr (.ANDI (.vreg v42) (.vreg v41) (-8 : BitVec 12)) <|
  .instr (.LD .normal (.vreg v42) (.vreg v42) (0 : BitVec 12)) <|
  .instr (.XORI (.vreg v41) (.vreg v41) (7 : BitVec 12)) <|
  .instr (.VirtualMULI (.vreg v41) (.vreg v41) (slliMultiplier 3)) <|
  .instr (.VirtualPow2 (.vreg v43) (.vreg v41)) <|
  .instr (.MUL (.vreg v42) (.vreg v42) (.vreg v43)) <|
  .instr (.VirtualSRAI (.vreg v40) (.vreg v42) (sraiBitmask 56)) <|
  .done RETIRE_SUCCESS
rd != x0:
  .instr (.ADDI (.vreg v40) (.xreg rs1) imm) <|
  .instr (.ANDI (.vreg v41) (.vreg v40) (-8 : BitVec 12)) <|
  .instr (.LD .normal (.vreg v41) (.vreg v41) (0 : BitVec 12)) <|
  .instr (.XORI (.vreg v40) (.vreg v40) (7 : BitVec 12)) <|
  .instr (.VirtualMULI (.vreg v40) (.vreg v40) (slliMultiplier 3)) <|
  .instr (.VirtualPow2 (.vreg v42) (.vreg v40)) <|
  .instr (.MUL (.vreg v41) (.vreg v41) (.vreg v42)) <|
  .instr (.VirtualSRAI (.xreg rd) (.vreg v41) (sraiBitmask 56)) <|
  .done RETIRE_SUCCESS
```


