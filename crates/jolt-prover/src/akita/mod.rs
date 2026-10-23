//! The Akita prove path: native trace-group commitments and heterogeneous batched openings.

pub mod preprocessing;
mod prover;
mod setup;
pub use setup::one_hot_trace_setup_shape;
// A Whiteout build does not prove until W6d wires its tail.
#[cfg(not(feature = "zk"))]
mod stage0;
// Whiteout's stage 8 is the masked Akita opening (W6d).
#[cfg(not(feature = "zk"))]
mod stage8;
pub mod witness;
pub use prover::prove;
#[cfg(not(feature = "zk"))]
mod spine;
