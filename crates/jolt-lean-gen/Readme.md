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
```

## Generating Full Lean File 

Run the following command to generate automatic expansions.

```zsh
cargo run -p jolt-lean-gen -- --lean --out /path/to/ExpansionsAutomated.lean
```


