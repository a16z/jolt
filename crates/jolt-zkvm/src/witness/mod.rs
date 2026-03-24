//! Witness generation for the Jolt proving pipeline.
//!
//! Bridges the tracer's execution trace (`Vec<Cycle>`) to the two distinct
//! witness representations the prover needs:
//!
//! 1. **R1CS witnesses** ([`r1cs_inputs`]): Per-cycle variable vectors for Spartan.
//! 2. **Committed polynomial witnesses** ([`cycle_data`] + [`store_sink`]):
//!    Evaluation tables (Inc, Ra one-hot) for sumcheck stages.
//!
//! The [`flags`] module provides the bridge between tracer instruction types
//! and the jolt-instructions flag enums that control operand routing.

pub mod bytecode;
pub mod cycle_data;
pub mod flags;
pub mod generate;
pub mod r1cs_inputs;
pub mod store;
pub mod store_sink;

pub use generate::{generate_witnesses, WitnessOutput};
pub use store::WitnessStore;
