//! BN254 Fr native-field coprocessor smoke demo.
//!
//! The `#[jolt::provable]` macro expands to a jolt-core (non-modular)
//! prover pipeline, which does not know how to handle `FieldOp` /
//! `FMov{I2F,F2I}` cycles — those are refactor-crates-only instructions.
//! Running this binary directly panics with "Unexpected instruction".
//!
//! Use the modular-prover integration test at
//! `crates/jolt-equivalence/tests/bn254_fr_smoke.rs` for an actual
//! prove/verify cycle of this guest on the refactor-crates pipeline.
fn main() {
    eprintln!(
        "This binary is a placeholder. For a working end-to-end demo see \
         `crates/jolt-equivalence/tests/bn254_fr_smoke.rs` which runs the \
         same guest through the modular prover."
    );
    std::process::exit(0);
}
