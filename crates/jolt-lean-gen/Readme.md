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
        // Registered inline dispatch (needs an InlineExpansionProvider)
        | "Inline"
    )
}

```

LR.W, LR.D, SC.W and SC.D are generated for both destination branches.
The floor-addressed `VirtualLoad{B,H,W}` and `VirtualStore{B,H,W}` rows, and the `VirtualShiftData{B,H,W}` rows that carry an immediate, are generated with their fault class and offset.
The generated file uses the `JoltISA.Encoded.` constructors for 12-bit immediates and the plain constructors otherwise.
Only `VirtualAdvice` introduces a runtime payload parameter; `VirtualAdviceLoad`
retains its fixed byte-count immediate.

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


