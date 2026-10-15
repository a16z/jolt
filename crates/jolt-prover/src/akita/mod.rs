//! The Akita prove path: native trace-group commitments and heterogeneous batched openings.

pub mod preprocessing;
mod prover;
mod setup;
pub use setup::one_hot_trace_setup_shape;
mod stage0;
mod stage8;
pub mod witness;
pub use prover::prove;
